#!/usr/bin/env python3
"""LTAS/EQ 匹配：把源歌声的长期平均谱（LTAS）向参考音色的 LTAS 归一化。

实验背景（见 docs/source-timbre-normalization-survey.md E1）：
SVC 的 whisper 内容特征里残留源歌手信息。用纯线性滤波把源歌声的谱包络向目标/中性
拉近，检验"音频域音色归一化"能否降低 SVC 难度。**线性滤波不改变音高、节奏、动态**。

做法：
  1. 对源与参考各做 STFT，取能量 top-`--voiced-frac` 帧求平均功率谱（LTAS，dB）。
  2. 在 `--bands` 个 mel 频带上求平均带级。
  3. 增益(dB) = strength × (target_band − source_band)，clamp 到 ±--clamp-db。
     strength=1.0 → 完全匹配目标；0.5 → 源与目标的中点（"中性"臂）。
  4. 带级插值成整谱增益曲线 → firwin2 线性相位 FIR → filtfilt（零相位，无时序偏移）。
  5. 输出 RMS 归一化到输入，避免 EQ 改变响度。

用法（GPT-SoVITS venv）:
  GPT-SoVITS/.venv/bin/python tools/source_norm/ltas_match.py \
      --source data/songs/X/sep2/lead.wav --target data/voices/ref_talk_sing.wav \
      --out data/songs/X/lead_ltas_t1.wav --strength 1.0 --json ltas_report.json
"""
import argparse
import json
import os

import librosa
import numpy as np
import scipy.signal as ss
import soundfile as sf


def ltas_bands(x: np.ndarray, sr: int, n_fft: int, hop: int,
               bands: int, voiced_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (mel_centers_hz, band_db, band_std_db)。取能量 top-voiced_frac 帧。"""
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2
    frame_rms = np.sqrt(S.mean(axis=0))
    keep = np.argsort(frame_rms)[-max(1, int(frame_rms.size * voiced_frac)):]
    P = S[:, keep].mean(axis=1)                       # 平均功率谱
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    P = np.maximum(P, 1e-12)
    mel = librosa.mel_frequencies(n_mels=bands, fmin=30.0, fmax=sr / 2)
    centers = 0.5 * (mel[:-1] + mel[1:])
    edges = np.concatenate([[0.0], centers, [sr / 2]])
    band_db = np.zeros(bands - 1)
    band_std = np.zeros(bands - 1)
    for i in range(bands - 1):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if m.sum() == 0:
            continue
        db = 10 * np.log10(P[m])
        band_db[i] = db.mean()
        band_std[i] = db.std()
    return centers, band_db, band_std


def gain_curve(centers: np.ndarray, gain_db: np.ndarray,
               sr: int, n_fft: int) -> np.ndarray:
    """带增益 → 全频带线性增益曲线（线性插值，端点半带值外推）。"""
    pts = np.concatenate([[0.0], centers, [sr / 2]])
    g = np.concatenate([[gain_db[0]], gain_db, [gain_db[-1]]])
    freq_pts = np.linspace(0, sr / 2, n_fft // 2 + 1)
    g_db = np.interp(freq_pts, pts, g)
    return 10 ** (g_db / 20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="归一化强度：1=完全匹配目标 LTAS 形状，0.5=半程，0=不动")
    ap.add_argument("--clamp-db", type=float, default=8.0)
    ap.add_argument("--bands", type=int, default=24, help="mel 频带数（默认 24）")
    ap.add_argument("--voiced-frac", type=float, default=0.7,
                    help="取能量 top 多少比例帧算 LTAS（避开静音/气声帧）")
    ap.add_argument("--shape-only", action="store_true", default=True,
                    help="纯形状匹配：增益去均值（消掉说话/唱歌的整体电平差）+ 限带")
    ap.add_argument("--no-shape-only", dest="shape_only", action="store_false",
                    help="含整体电平差（旧行为，输出会变闷）")
    ap.add_argument("--fmin", type=float, default=150.0)
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--json", help="诊断报告输出（带级表 + 增益统计）")
    args = ap.parse_args()

    x, sr = librosa.load(args.source, sr=None, mono=True)
    t, sr_t = librosa.load(args.target, sr=None, mono=True)
    if sr_t != sr:
        t = librosa.resample(t, orig_sr=sr_t, target_sr=sr)
    n_fft, hop = 4096, 1024

    c_src, b_src, _ = ltas_bands(x, sr, n_fft, hop, args.bands, args.voiced_frac)
    c_tgt, b_tgt, _ = ltas_bands(t, sr, n_fft, hop, args.bands, args.voiced_frac)
    gain_db = args.strength * (b_tgt - b_src)
    if args.shape_only:
        # 纯形状匹配：去掉整体电平差（说话 vs 唱歌本身有 ~10dB 电平/倾角差，
        # 不剔除会让"匹配"退化成整体压高频）。RMS 归一化已经保证输出响度不变。
        gain_db = gain_db - gain_db.mean()
        # 限带：身份信息主要在 150-8000Hz，带外不动（0dB）
        band_fmin, band_fmax = args.fmin, args.fmax
        gain_db = np.where((c_src >= band_fmin) & (c_src <= band_fmax), gain_db, 0.0)
    gain_db = np.clip(gain_db, -args.clamp_db, args.clamp_db)
    curve = gain_curve(c_src, gain_db, sr, n_fft)

    taps = 2049
    fir = ss.firwin2(taps, np.linspace(0, 1, curve.size), curve, window="hamming")
    y = ss.filtfilt(fir, 1.0, x)
    # RMS 归一化到输入（EQ 不改响度）
    rms_in = np.sqrt(np.mean(x ** 2)) + 1e-12
    rms_out = np.sqrt(np.mean(y ** 2)) + 1e-12
    y = y * (rms_in / rms_out)
    # 防削波
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = y * 0.999 / peak

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, y, sr, subtype="PCM_16")

    if args.json:
        rep = {
            "sr": sr, "strength": args.strength, "clamp_db": args.clamp_db,
            "bands": [round(float(c), 1) for c in c_src],
            "source_band_db": [round(float(v), 2) for v in b_src],
            "target_band_db": [round(float(v), 2) for v in b_tgt],
            "gain_db": [round(float(v), 2) for v in gain_db],
            "gain_min_db": round(float(gain_db.min()), 2),
            "gain_max_db": round(float(gain_db.max()), 2),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[ltas] src_ltas={round(float(b_src.mean()),1)}dB tgt_ltas={round(float(b_tgt.mean()),1)}dB "
          f"gain∈[{gain_db.min():+.1f},{gain_db.max():+.1f}]dB  -> {args.out}")


if __name__ == "__main__":
    main()
