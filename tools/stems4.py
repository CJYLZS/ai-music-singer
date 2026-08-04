#!/usr/bin/env python3
"""4-stem separation: drums / bass / other / vocals.

Usage:
  GPT-SoVITS/.venv/bin/python tools/stems4.py <input.wav> <outdir> [--mix drums+bass]

Uses GPT-SoVITS's Roformer_Loader with the 4-stem BS-RoFormer weights
(bs_roformer_4stems_ft.pth). With --mix, also writes drums+bass summed
into <outdir>/<name>_drum_bass.wav.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GPT-SoVITS", "tools", "uvr5"))

import librosa
import numpy as np
import soundfile as sf
import torch

from bsroformer import Roformer_Loader

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GPT-SoVITS", "tools", "uvr5", "uvr5_weights", "bs_roformer_4stems_ft.pth")
CONFIG = WEIGHTS[:-4] + ".yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--mix", action="store_true", help="also write drums+bass summed")
    ap.add_argument("--drum-bass-out", default=None,
                    help="drums+bass 输出路径（--mix 时覆盖默认的 <outdir>/<name>_drum_bass.wav）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action="store_true", default=True, help="use fp16 (default on)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    loader = Roformer_Loader(WEIGHTS, CONFIG, args.device, args.half)
    loader.model.eval()

    sr = loader.config["audio"]["sample_rate"]
    mix, _ = librosa.load(args.input, sr=sr, mono=False)
    mixture = torch.tensor(mix, dtype=torch.float32)
    stems = loader.demix_track(loader.model, mixture, args.device)

    for name, audio in stems.items():
        path = os.path.join(args.outdir, f"{base}_{name}.wav")
        sf.write(path, audio.T, sr)
        print(f"wrote {path}")

    if args.mix:
        db = stems["drums"] + stems["bass"]
        path = args.drum_bass_out or os.path.join(args.outdir, f"{base}_drum_bass.wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, db.T, sr)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
