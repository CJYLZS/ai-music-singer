#!/usr/bin/env python
"""按音高分桶测量说话人相似度，用于验证"音高距离 -> 相似度下降"的假设。

复用仓库自带的 ERes2NetV2 SV 模型（GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/）。
输入需为 16k 可重采样的 wav；内部统一走 32k -> 16k 的既有路径。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi

REPO = "/home/rookie/voice"
sys.path.insert(0, os.path.join(REPO, "GPT-SoVITS", "GPT_SoVITS", "eres2net"))
from ERes2NetV2 import ERes2NetV2  # noqa: E402

SV_CKPT = os.path.join(REPO, "GPT-SoVITS", "GPT_SoVITS", "pretrained_models",
                       "sv", "pretrained_eres2netv2w24s4ep4.ckpt")


class SV:
    def __init__(self, device="cuda"):
        state = torch.load(SV_CKPT, map_location="cpu")
        m = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        m.load_state_dict(state)
        m.eval()
        self.m = m.to(device)
        self.device = device

    def emb(self, wav_16k: np.ndarray) -> torch.Tensor:
        """wav_16k: 1-D float array at 16 kHz, range -1..1."""
        w = torch.from_numpy(wav_16k).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = torch.stack(
                [Kaldi.fbank(w0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for w0 in w]
            )
            return self.m.forward3(feat).squeeze(0)


def load16k(path: str) -> np.ndarray:
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != 16000:
        d = (
            torchaudio.functional.resample(torch.from_numpy(d).float(), sr, 16000)
            .numpy()
        )
    return d.astype(np.float32)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考音色 wav（prompt）")
    ap.add_argument("--gen", required=True, nargs="+", help="待评测的生成 wav，可多个")
    ap.add_argument("--f0", required=True, help="生成音频对应的源 F0 .npy（50Hz 网格）")
    ap.add_argument("--ref-f0", required=True, help="参考音频的 F0 .npy")
    ap.add_argument("--win", type=float, default=3.0, help="分析窗长(秒)")
    args = ap.parse_args()

    sv = SV()
    ref_emb = sv.emb(load16k(args.ref))
    ref_f0 = np.load(args.ref_f0)
    ref_med = float(np.median(ref_f0[ref_f0 > 0]))
    print(f"reference: {args.ref}  median F0 = {ref_med:.0f} Hz\n")

    f0 = np.load(args.f0)
    F0_RATE = 50

    for gpath in args.gen:
        wav = load16k(gpath)
        win = int(args.win * 16000)
        rows = []
        for start in range(0, len(wav) - win, win):
            seg = wav[start : start + win]
            if np.sqrt(np.mean(seg**2)) < 0.01:
                continue
            t0 = start / 16000
            fa, fb = int(t0 * F0_RATE), int((t0 + args.win) * F0_RATE)
            fseg = f0[fa:fb]
            v = fseg[fseg > 0]
            if len(v) < 20:
                continue
            med = float(np.median(v))
            semis = 12 * np.log2(med / ref_med)
            rows.append((t0, med, semis, cos(sv.emb(seg), ref_emb)))

        if not rows:
            print(f"{gpath}: no usable windows")
            continue

        sims = np.array([r[3] for r in rows])
        dist = np.array([abs(r[2]) for r in rows])
        print(f"=== {os.path.basename(gpath)}   windows={len(rows)}")
        print(f"    overall SIM: mean={sims.mean():.4f}  median={np.median(sims):.4f}")

        # 按与参考的半音距离分桶
        print(f"    {'semitones from ref':<22}{'n':>4}{'meanSIM':>10}")
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 99)]:
            m = (dist >= lo) & (dist < hi)
            if m.sum() >= 2:
                lbl = f"{lo}-{hi if hi < 99 else '+'}"
                print(f"    {lbl:<22}{m.sum():>4}{sims[m].mean():>10.4f}")

        if len(rows) > 4:
            r = np.corrcoef(dist, sims)[0, 1]
            print(f"    corr(pitch distance, SIM) = {r:+.3f}")
        print()


if __name__ == "__main__":
    main()
