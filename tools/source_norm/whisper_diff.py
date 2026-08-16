#!/usr/bin/env python3
"""特征域诊断：源 lead 的各种预处理版本经过 whisper 后，内容特征变了多少。

解释 E1 的 SIM 持平：若 LTAS/EQ 版与原始版的 whisper 特征几乎相同（cos≈1），
SoulX 看到的源就没变，SIM 当然不动。共振峰搬移版若差异更大，则 E2 可能有机会。

用法（SoulX-Singer venv，cwd=SoulX-Singer）:
  PYTHONPATH=/home/rookie/voice/SoulX-Singer /home/rookie/voice/SoulX-Singer/.venv/bin/python \
      /home/rookie/voice/tools/source_norm/whisper_diff.py --song-dir data/songs/白色毛衣 \
      --files lead_ltas_n05.wav lead_ltas_n10.wav lead_fshift_n10.wav
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer")))
from soulxsinger.models.modules.whisper_encoder import WhisperEncoder  # noqa: E402


def load_wav(p, sr):
    y, sr0 = torchaudio.load(p)
    if sr0 != sr:
        y = torchaudio.functional.resample(y, sr0, sr)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--files", nargs="+", required=True,
                    help="相对歌曲目录的 wav（第一个必须是原始 sep2/lead.wav）")
    args = ap.parse_args()
    enc = WhisperEncoder()
    enc.model = enc.model.cuda().eval()
    feats = {}
    for fn in args.files:
        p = fn if os.path.isabs(fn) else os.path.join(args.song_dir, fn)
        wav = load_wav(p, 24000).cuda()
        with torch.no_grad():
            f = enc.encode(wav, sr=24000)[0].cpu().numpy()  # (T, D)
        feats[fn] = f
        print(f"{fn}: {f.shape}")
    names = list(feats)
    base = feats[names[0]]
    print("\nmean frame cosine vs baseline:")
    for n in names[1:]:
        f = feats[n]
        T = min(base.shape[0], f.shape[0])
        cos = (base[:T] * f[:T]).sum(axis=1) / (
            np.linalg.norm(base[:T], axis=1) * np.linalg.norm(f[:T], axis=1) + 1e-9)
        print(f"  {n:28s} mean_cos={cos.mean():.4f}  p10={np.percentile(cos,10):.4f}  "
              f"<0.9 占比={(cos<0.9).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
