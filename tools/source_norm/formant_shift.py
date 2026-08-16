#!/usr/bin/env python3
"""E2: WORLD 共振峰搬移（VTLN 式谱包络弯折）——保 F0、保节奏的"声道归一化"。

实验背景（见 docs/source-timbre-normalization-survey.md E2）：LTAS/EQ 只改谱平衡，
不动共振峰位置。这里用 source-filter 分解（pyworld: f0 + 谱包络 sp + 非周期 ap），
把源歌声的谱包络按"源→目标共振峰比例"沿频率轴弯折，再用**原始 f0/ap** 重合成。
音高与节奏在构造上完全不动。

做法：
  1. 源与参考各做 WORLD 分解，在 voiced 帧上峰值拾取 F1/F2，取中位数。
  2. α = clamp(target_F1med / source_F1med, --alpha-min, --alpha-max)；
     实际弯折因子 α_eff = 1 + strength·(α−1)（strength=1 完全对齐，0.5 半程）。
  3. 每帧把 sp 按 α_eff 弯折（log 域插值，f/α_eff 处取值），f0/ap 原样重合成。
  4. RMS 归一化到输入。

用法（tools/source_norm/.venv）:
  tools/source_norm/.venv/bin/python tools/source_norm/formant_shift.py \
      --source data/songs/X/sep2/lead.wav --target data/voices/ref_talk_sing.wav \
      --out data/songs/X/lead_fshift_s1.wav --strength 1.0 --json fshift.json
"""
import argparse
import json
import os

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf


def world_decompose(x: np.ndarray, fs: int, fft_size: int):
    f0, t = pw.dio(x.astype(np.float64), fs, f0_floor=50.0, f0_ceil=1200.0)
    f0 = pw.stonemask(x.astype(np.float64), f0, t, fs)
    sp = pw.cheaptrick(x.astype(np.float64), f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x.astype(np.float64), f0, t, fs, fft_size=fft_size)
    return f0, sp, ap


def formant_medians(sp: np.ndarray, f0: np.ndarray, fs: int, fft_size: int) -> tuple[float, float]:
    """voiced 帧频带受限峰值拾取 F1/F2，返回中位数（Hz）。

    F1 在 [150,900]Hz 取最大峰（唱歌 /a/ 的 F1 可到 ~900），F2 在 [900,2500]Hz 取最大峰。
    全局最大峰在唱歌音频里常是谐波峰（F1 估计成 2kHz+），所以必须限带。
    """
    freqs = np.linspace(0, fs / 2, sp.shape[1])
    f1s, f2s = [], []
    voiced = f0 > 0
    idx = np.where(voiced)[0][::2]  # 隔帧采样，省时间
    for i in idx:
        db = 20 * np.log10(sp[i] + 1e-12)
        m1 = (freqs >= 150) & (freqs <= 900)
        m2 = (freqs >= 900) & (freqs <= 2500)
        seg1, seg2 = db[m1], db[m2]
        if seg1.size == 0 or seg2.size == 0:
            continue
        f1s.append(freqs[m1][np.argmax(seg1)])
        f2s.append(freqs[m2][np.argmax(seg2)])
    med1 = float(np.median(f1s)) if f1s else 400.0
    med2 = float(np.median(f2s)) if f2s else 1500.0
    return med1, med2


def warp_sp(sp: np.ndarray, alpha: float, fs: int, fft_size: int) -> np.ndarray:
    """把谱包络沿频率轴弯折：新包络在 f 处取旧包络在 f/alpha 处的值（log 域插值）。"""
    nf = sp.shape[1]
    freqs_new = np.linspace(0, fs / 2, nf)
    freqs_old = np.linspace(0, fs / 2, nf)
    src = np.log(sp + 1e-12)
    out = np.empty_like(sp)
    fmax_old = fs / 2 / alpha  # 弯折后高于该频率的旧包络不存在
    for i in range(sp.shape[0]):
        warped = np.interp(freqs_new / alpha, freqs_old, src[i])
        floor = src[i].min()
        out[i] = np.where(freqs_new <= fmax_old, warped, floor)
    return np.exp(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="弯折强度：1=完全对齐目标 F1，0.5=半程，0=不动")
    ap.add_argument("--alpha-min", type=float, default=0.80)
    ap.add_argument("--alpha-max", type=float, default=1.25)
    ap.add_argument("--fft-size", type=int, default=2048)
    ap.add_argument("--json", help="诊断报告")
    args = ap.parse_args()

    x, sr = sf.read(args.source, dtype="float64", always_2d=False)
    t, sr_t = sf.read(args.target, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if t.ndim > 1:
        t = t.mean(axis=1)
    if sr_t != sr:
        # 简易重采样（librosa 不在本 venv；用 scipy 重采样）
        n = int(len(t) * sr / sr_t)
        t = ss.resample_poly(t, sr, sr_t)
        t = t[:n] if len(t) >= n else np.pad(t, (0, n - len(t)))

    f0, sp, ap = world_decompose(x, sr, args.fft_size)
    f0_t, sp_t, _ = world_decompose(t, sr, args.fft_size)
    f1s, f2s = formant_medians(sp, f0, sr, args.fft_size)
    f1t, f2t = formant_medians(sp_t, f0_t, sr, args.fft_size)
    alpha = float(np.clip(f1t / max(f1s, 1e-6), args.alpha_min, args.alpha_max))
    alpha_eff = 1.0 + args.strength * (alpha - 1.0)

    if abs(alpha_eff - 1.0) > 1e-6:
        sp_w = warp_sp(sp, alpha_eff, sr, args.fft_size)
    else:
        sp_w = sp
    y = pw.synthesize(f0, sp_w, ap, sr)
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * (np.sqrt(np.mean(x ** 2)) + 1e-12)
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = y * 0.999 / peak

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, y.astype(np.float32), sr, subtype="PCM_16")

    if args.json:
        rep = {
            "sr": sr, "strength": args.strength,
            "f1_source_hz": round(f1s, 1), "f1_target_hz": round(f1t, 1),
            "f2_source_hz": round(f2s, 1), "f2_target_hz": round(f2t, 1),
            "alpha_raw": round(alpha, 4), "alpha_eff": round(alpha_eff, 4),
            "voiced_frames": int((f0 > 0).sum()),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[fshift] F1 {f1s:.0f}->{f1t:.0f}Hz α_raw={alpha:.3f} α_eff={alpha_eff:.3f} "
          f"voiced={(f0>0).sum()} frames -> {args.out}")


if __name__ == "__main__":
    main()
