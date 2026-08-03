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
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UVR5_DIR = ROOT / "GPT-SoVITS" / "tools" / "uvr5"
WEIGHT_DIR = UVR5_DIR / "uvr5_weights"
DEFAULT_CACHE_ROOT = ROOT / "data" / ".sep_cache"
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


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="输入音频")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", default="True", choices=["True", "False"])
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT),
                    help="分离缓存根目录（按输入 md5 复用产物，默认 data/.sep_cache）")
    ap.add_argument("--no-cache", action="store_true", help="禁用缓存（重新分离）")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"输入不存在: {args.src}")

    m = MODELS[args.model]
    ckpt = WEIGHT_DIR / (m["ckpt"] + ".ckpt")
    yaml = WEIGHT_DIR / (m["ckpt"] + ".yaml")
    for p in (ckpt, yaml):
        if not p.exists():
            sys.exit(f"缺少模型文件: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 缓存查找: 输入文件 md5 + 权重 md5，两者都一致才算命中 ----
    in_md5 = file_md5(args.src)
    cache_dir: Path | None = None
    if not args.no_cache:
        cache_dir = (Path(args.cache_root) / f"{args.model}-{file_md5(str(ckpt))[:8]}" / in_md5)
        hits = [n for n in m["out_names"]
                if (cache_dir / n).exists() and (cache_dir / n).stat().st_size > 0]
        if len(hits) == len(m["out_names"]):
            for n in m["out_names"]:
                shutil.copy2(str(cache_dir / n), str(out_dir / n))
                print(f"[sep-cache] 命中 {cache_dir / n} -> {out_dir / n}")
            return
        if hits:
            print(f"[sep-cache] 部分命中，重新分离: {cache_dir}", file=sys.stderr)

    import torch

    is_half = args.half == "True"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退 CPU", file=sys.stderr)
        args.device = "cpu"
        is_half = False

    print(f"[sep-cache] miss md5={in_md5[:12]}… 开始分离", file=sys.stderr)
    from bsroformer import Roformer_Loader

    loader = Roformer_Loader(
        model_path=str(ckpt),
        config_path=str(yaml),
        device=args.device,
        is_half=is_half,
    )

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

    # ---- 回填缓存: 先写 .tmp 再原子改名，避免杀进程留下半截文件被误判命中 ----
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for n in m["out_names"]:
            tmp = cache_dir / (n + ".tmp")
            shutil.copy2(str(out_dir / n), str(tmp))
            os.replace(str(tmp), str(cache_dir / n))
        print(f"[sep-cache] 已写入 {cache_dir}")


if __name__ == "__main__":
    main()
