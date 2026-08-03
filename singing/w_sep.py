"""两段式人声/伴奏分离（GPT-SoVITS venv 运行）。

用 GPT-SoVITS/tools/uvr5 的 Roformer_Loader 直接调 _path_audio_，
一次性保留人声轨和伴奏轨（uvr5_cli.py 只搬走人声）。

模型:
    bs       -> model_bs_roformer_ep_317_sdr_12.9755 : 歌曲 -> vocal(含和声) + acc(伴奏)
    karaoke  -> mel_band_roformer_karaoke_becruily  : 对 vocal 再分 -> lead(主唱) + backing(和声)

输出（规范名）:
    bs      : <out>/vocal.wav  <out>/acc.wav
    karaoke : <out>/lead.wav   <out>/backing.wav
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UVR5_DIR = ROOT / "GPT-SoVITS" / "tools" / "uvr5"
WEIGHT_DIR = UVR5_DIR / "uvr5_weights"
sys.path.insert(0, str(UVR5_DIR))

MODELS = {
    "bs": {
        "ckpt": "model_bs_roformer_ep_317_sdr_12.9755",
        "out_names": ["vocal.wav", "acc.wav"],  # 目标轨 -> 规范名
        "src_names": ["_vocals.wav", "_other.wav"],
    },
    "karaoke": {
        "ckpt": "mel_band_roformer_karaoke_becruily",
        "out_names": ["lead.wav", "backing.wav"],
        "src_names": ["_Vocals.wav", "_Instrumental.wav"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="输入音频")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", default="True", choices=["True", "False"])
    args = ap.parse_args()

    import torch

    is_half = args.half == "True"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退 CPU", file=sys.stderr)
        args.device = "cpu"
        is_half = False

    m = MODELS[args.model]
    ckpt = WEIGHT_DIR / (m["ckpt"] + ".ckpt")
    yaml = WEIGHT_DIR / (m["ckpt"] + ".yaml")
    for p in (ckpt, yaml):
        if not p.exists():
            sys.exit(f"缺少模型文件: {p}")

    from bsroformer import Roformer_Loader

    loader = Roformer_Loader(
        model_path=str(ckpt),
        config_path=str(yaml),
        device=args.device,
        is_half=is_half,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocal_dir = out_dir / "raw_vocal"
    other_dir = out_dir / "raw_other"
    base = Path(args.src).stem
    loader._path_audio_(args.src, str(other_dir), str(vocal_dir), "wav", False)

    del loader
    torch.cuda.empty_cache()

    for src_suffix, out_name in zip(m["src_names"], m["out_names"]):
        src = vocal_dir / (base + src_suffix)
        if not src.exists():
            src = other_dir / (base + src_suffix)
        if not src.exists():
            sys.exit(f"分离输出缺失: {base + src_suffix}（在 {vocal_dir} / {other_dir}）")
        dst = out_dir / out_name
        if src != dst:
            shutil.move(str(src), str(dst))
        print(f"-> {dst}")
    shutil.rmtree(vocal_dir, ignore_errors=True)
    shutil.rmtree(other_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
