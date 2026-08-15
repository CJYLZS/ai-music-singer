#!/usr/bin/env python3
"""评测指标敏感性探针 + SVC 输出低分窗口诊断。

探针：对一段人声做受控扰动，看 SingMOS / NISQA-TTS 的反应，回答
"这些指标惩罚什么 / 优化什么能涨分"。

诊断：对 3s 窗逐窗提取声学特征（响度、频谱质心、平直度、F0 均值/抖动、
清音占比、过零率），与 SingMOS 分数做相关分析，并列出
最差窗口 / 与原唱差距最大的窗口。

用法:
  probe_metric_sensitivity.py <合成vocal.wav> <原唱lead.wav> <逐窗CSV> \
      [--seg-start 60] [--seg-len 60]
"""
import argparse
import json
import os
import sys

import numpy as np
import librosa
import soundfile as sf
from scipy import signal as scisig
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_naturalness import (SR, load_windows, singmos_predict,
                               nisqa_predict_file, _TOOL_DIR)
import torch


def make_variants(base, sr):
    """base: float32 一维数组。返回 {name: array}。"""
    rng = np.random.default_rng(0)
    out = {"base": base}
    p = float(np.sqrt(np.mean(base ** 2)))

    def snr_db(db):
        n = rng.standard_normal(len(base)) * p / (10 ** (db / 20))
        return base + n

    out["noise40"] = snr_db(40)
    out["noise25"] = snr_db(25)

    def butter(sos):
        return scisig.sosfilt(sos, base)

    out["lp7k"] = butter(scisig.butter(6, 7000 / (sr / 2), "lowpass", output="sos"))
    out["hp60"] = butter(scisig.butter(4, 60 / (sr / 2), "highpass", output="sos"))

    # 卷积混响：1.2s 指数衰减 IR，30% 湿
    ir = np.exp(-np.arange(int(1.2 * sr)) / (0.35 * sr)) * rng.standard_normal(int(1.2 * sr))
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = scisig.fftconvolve(base, ir)[:len(base)]
    out["reverb30"] = 0.7 * base + 0.3 * wet

    # 音高抖动：0.8s 块 ±40 cents 随机
    chunk = int(0.8 * sr)
    wob = []
    n_chunks = int(np.ceil(len(base) / chunk))
    for i in range(n_chunks):
        c = base[i * chunk:(i + 1) * chunk]
        if len(c) < 100:
            break
        st = float(rng.uniform(-0.4, 0.4))
        wob.append(librosa.effects.pitch_shift(c, sr=sr, n_steps=st))
    # 交叉淡化拼接
    out["wobble40"] = crossfade_concat(wob, int(0.02 * sr))
    return out


def crossfade_concat(chunks, fade):
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0].copy()
    for c in chunks[1:]:
        seg = np.concatenate([out[-fade:], c])
        ramp = np.linspace(0, 1, fade)
        seg[:fade] = seg[:fade] * (1 - ramp) + out[-fade:] * ramp
        out = np.concatenate([out[:-fade], seg])
    return out


def window_features(y, sr, win):
    """对整段音频按 3s 窗提取特征，返回 (idx列表, dict[name->array])。"""
    n = int(win * sr)
    feats = {k: [] for k in
             ("rms", "centroid", "flatness", "zcr", "f0_mean", "f0_std_semi", "voiced_ratio")}
    idxs = []
    for i in range(0, len(y) - n + 1, n):
        w = y[i:i + n]
        idxs.append(i // n)
        feats["rms"].append(float(np.sqrt(np.mean(w ** 2))))
        spec = np.abs(librosa.stft(w, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        mag = spec.sum(axis=1) + 1e-12
        feats["centroid"].append(float(np.sum(freqs * mag) / np.sum(mag)))
        feats["flatness"].append(float(librosa.feature.spectral_flatness(S=spec).mean()))
        feats["zcr"].append(float(librosa.feature.zero_crossing_rate(w).mean()))
        f0 = librosa.yin(w, fmin=65, fmax=1100, sr=sr, hop_length=256)
        voiced = f0 < 1000
        if voiced.sum() > 0:
            fv = f0[voiced]
            feats["f0_mean"].append(float(np.median(fv)))
            feats["f0_std_semi"].append(
                float(12 * np.log2(np.percentile(fv, 90) / max(np.percentile(fv, 10), 1))))
        else:
            feats["f0_mean"].append(0.0)
            feats["f0_std_semi"].append(0.0)
        feats["voiced_ratio"].append(float(voiced.mean()))
    return idxs, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synth_wav")
    ap.add_argument("orig_wav")
    ap.add_argument("windows_csv")
    ap.add_argument("--seg-start", type=float, default=60.0)
    ap.add_argument("--seg-len", type=float, default=60.0)
    args = ap.parse_args()

    import csv as csvmod
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[singmos] loading predictor ...", flush=True)
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True
    ).to(dev).eval()
    repo = os.path.join(_TOOL_DIR, "..", "nisqa")

    # ================= 探针 =================
    y, _ = librosa.load(args.synth_wav, sr=SR, mono=True)
    seg = y[int(args.seg_start * SR):int((args.seg_start + args.seg_len) * SR)]
    print(f"\n=== 探针: {args.synth_wav} 片段 {args.seg_start}-{args.seg_start + args.seg_len}s ===")
    variants = make_variants(seg, SR)
    rows = []
    for name, v in variants.items():
        wavs, kept, total, _ = load_windows_from_array(v)
        scores = singmos_predict(predictor, wavs, 8, dev)
        # NISQA-TTS
        tmp = os.path.join(_TOOL_DIR, ".tmp", f"probe_{name}.wav")
        sf.write(tmp, v, SR)
        nis = nisqa_predict_file(repo, "nisqa_tts", tmp)
        rows.append((name, float(scores.mean()), float(np.median(scores)), nis["mos_pred"]))
    print(f"{'变体':<12}{'SingMOS mean':>13}{'median':>9}{'NISQA-TTS':>11}{'ΔS':>7}{'ΔT':>7}")
    base_s, base_t = rows[0][1], rows[0][3]
    for name, ms, md, nt in rows:
        print(f"{name:<12}{ms:>13.3f}{md:>9.3f}{nt:>11.3f}{ms - base_s:>+7.3f}{nt - base_t:>+7.3f}")

    # ================= 诊断 =================
    y_o, _ = librosa.load(args.orig_wav, sr=SR, mono=True)
    scores = {}
    with open(args.windows_csv) as f:
        for r in csvmod.DictReader(f):
            scores.setdefault(r["label"], {})[int(r["window_idx"])] = float(r["singmos_mos"])
    print(f"\n=== 诊断: 逐窗特征 vs SingMOS ===")
    for label, yy in (("合成", y), ("原唱", y_o)):
        idxs, feats = window_features(yy, SR, 3.0)
        mask = [i in scores[label] for i in idxs]
        keep = [i for i in idxs if i in scores[label]]
        s = np.array([scores[label][i] for i in keep])
        farr = {k: np.array(v)[mask] for k, v in feats.items()}
        corrs = {}
        for k, v in farr.items():
            rho, p = spearmanr(v, s)
            corrs[k] = rho
        print(f"\n[{label}] {len(s)} 窗  Spearman 相关:")
        for k, rho in sorted(corrs.items(), key=lambda x: -abs(x[1])):
            print(f"  {k:<14} {rho:+.3f}")
        order = np.argsort(s)
        print(f"  最低分 8 窗 (idx, score, rms, f0std_semi, voiced):")
        for pos in order[:8]:
            k = keep[pos]
            print(f"    w{k:>3}  score={s[pos]:.3f}  rms={farr['rms'][pos]:.4f}  "
                  f"f0std={farr['f0_std_semi'][pos]:.2f}st  voiced={farr['voiced_ratio'][pos]:.2f}")

    # 配对差距
    orig_s = scores.get("原唱", {})
    synth_s = scores.get("合成", {})
    common = sorted(set(orig_s) & set(synth_s))
    gaps = [(i, orig_s[i] - synth_s[i]) for i in common]
    gaps.sort(key=lambda x: -x[1])
    print(f"\n=== 原唱-合成 逐窗差距（正=原唱赢）: 共 {len(common)} 窗 ===")
    print("  原唱赢最多的 8 窗:", [(f"w{i}", f"{g:+.2f}") for i, g in gaps[:8]])
    print("  合成赢最多的 8 窗:", [(f"w{i}", f"{g:+.2f}") for i, g in gaps[-8:]])


def load_windows_from_array(seg):
    """把数组切成 3s 窗（探针用，不跳静音）。"""
    n = int(3.0 * SR)
    wavs = [seg[i:i + n] for i in range(0, len(seg) - n + 1, n)]
    return wavs, len(wavs), len(wavs), 1.0


if __name__ == "__main__":
    main()
