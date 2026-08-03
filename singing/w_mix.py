"""人声 + 伴奏混音（SoulX-Singer venv 运行）。

SVC 输出 24kHz 单声道，伴奏为 44.1kHz 立体声：
重采样人声到伴奏采样率、截断/补零对齐长度，按增益混音后写 PCM16 wav。
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True, help="SVC 输出（24kHz 单声道）")
    ap.add_argument("--acc", required=True, help="伴奏（44.1kHz）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocal-gain", type=float, default=1.0)
    ap.add_argument("--acc-gain", type=float, default=0.8)
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    v, sr_v = librosa.load(args.vocal, sr=None, mono=True)
    a, sr_a = librosa.load(args.acc, sr=None, mono=False)

    if sr_a != sr_v:
        v = librosa.resample(v, orig_sr=sr_v, target_sr=sr_a)
        print(f"重采样人声 {sr_v} -> {sr_a}")
    if a.ndim > 1:
        a = a.mean(axis=0)  # 转单声道处理，输出仍立体声

    n = a.shape[-1]
    v = np.pad(v, (0, max(0, n - len(v))))[:n]
    a = a[:n]

    mix = args.vocal_gain * v + args.acc_gain * a
    mix = np.clip(mix, -1.0, 1.0)
    stereo = np.stack([mix, mix], axis=1)  # (frames, 2) for soundfile
    sf.write(args.out, stereo, sr_a)
    print(f"-> {args.out} ({n / sr_a:.1f}s, {sr_a}Hz stereo, vocal {args.vocal_gain:.2f} / acc {args.acc_gain:.2f})")


if __name__ == "__main__":
    main()
