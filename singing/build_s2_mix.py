#!/usr/bin/env python3
"""实验3 数据集构建：说话 + 唱歌按比例混合，供 GPT-SoVITS S2 从头训练。

切分方式沿用唱歌流水线的 `w_select.fine_cands` 两级切分，**不用 slicer2**：
  1. 粗切（长静音 186ms）出乐句边界
  2. 长段内部按短静音（93ms）细分
  3. 仍超上限则在 RMS 最低点硬切
  4. 短段向邻居合并，保证候选严格落在 [--min-sec, --max-sec] 闭区间

为什么不用 slicer2（实测，本仓库同一条 538s 参考）:
  - slicer2 不控上限：唱歌段切出 40.9s 的片段，而 `DistributedBucketSampler`
    最高边界 1900 帧 = 38s，超出的样本被 **静默丢弃**（data_utils.py:1001）。
  - w_select 的 docstring 记录：老切法（frame 2048/top_db 25）把该参考切成 674 段
    med 0.41s，段间空隙中位仅 58ms —— 短语内部微停顿被误当边界。

长度默认 [3, 6] 秒，落在 bucket0（0.64-6.00s）内。AGENTS.md：
  "Keeping clips inside bucket0 and using a modest batch_size matters far more
   than raw dataset duration"，且 S2 每步只监督 segment_size=20480 → 0.64s 波形，
  长片段不增加波形监督。

用法（GPT-SoVITS venv）:
  GPT-SoVITS/.venv/bin/python singing/build_s2_mix.py \
      --talk-src data/voices/ref_talk_sing.wav --talk-end 162 \
      --sing-src data/voices/ref_talk_sing.wav --sing-start 165.166 \
      --name talksing_s2mix50 --sing-ratio 0.5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_fine_cands():
    spec = importlib.util.spec_from_file_location("w_select", ROOT / "singing/w_select.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.fine_cands


def cut_range(src: Path, dst: Path, start: float, end: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - start}"]
    cmd += ["-ac", "1", "-ar", "44100", str(dst)]
    subprocess.run(cmd, check=True)


def segment(wav: Path, out_dir: Path, prefix: str, min_sec: float, max_sec: float,
            top_db: float) -> list[tuple[str, float]]:
    import librosa
    import soundfile as sf
    fine_cands = load_fine_cands()
    y, sr = librosa.load(str(wav), sr=44100, mono=True)
    cands = fine_cands(y, sr, min_sec=min_sec, max_cand=max_sec, top_db=top_db)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (s, e) in enumerate(cands):
        seg = y[s:e]
        peak = float(abs(seg).max())
        if peak < 1e-4:
            continue
        name = f"{prefix}_{i:03d}_{s}_{e}.wav"
        sf.write(out_dir / name, seg / peak * 0.9, sr)
        rows.append((name, (e - s) / sr))
    return rows


def bucket_hist(durs: list[float]) -> dict:
    """DistributedBucketSampler 分桶（frames @ hop640/32k，1 frame = 20ms）。"""
    bounds = [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
              1300, 1400, 1500, 1600, 1700, 1800, 1900]
    h = {}
    dropped = 0
    for d in durs:
        fr = d * 32000 / 640
        idx = None
        for i in range(len(bounds) - 1):
            if bounds[i] < fr <= bounds[i + 1]:
                idx = i
                break
        if idx is None:
            dropped += 1
        else:
            h[idx] = h.get(idx, 0) + 1
    return {"buckets": h, "dropped_out_of_range": dropped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--talk-src", default="data/voices/ref_talk_sing.wav")
    ap.add_argument("--talk-start", type=float, default=0.0)
    ap.add_argument("--talk-end", type=float, default=162.0,
                    help="说话段结束秒（AGENTS.md：162s 之后已在唱，不要用 261）")
    ap.add_argument("--sing-src", default="data/voices/ref_talk_sing.wav")
    ap.add_argument("--sing-start", type=float, default=165.166,
                    help="唱歌段起始秒（实测互相关：ref_sing_only 对应此偏移）")
    ap.add_argument("--sing-end", type=float, default=None)
    ap.add_argument("--name", required=True, help="数据集名，决定输出目录与 .list")
    ap.add_argument("--min-sec", type=float, default=3.0, help="片段下限秒")
    ap.add_argument("--max-sec", type=float, default=6.0,
                    help="片段上限秒（默认 6.0 = bucket0 上界）")
    ap.add_argument("--top-db", type=float, default=30.0)
    ap.add_argument("--sing-ratio", type=float, default=0.5,
                    help="唱歌在总时长中的目标占比（0.5 = 50/50）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep-talk-slices", action="store_true",
                    help="复用 girlish_sliced 的已校对说话切片，不重切说话段")
    args = ap.parse_args()

    random.seed(args.seed)
    tmp = ROOT / ".tmp/exp3"
    tmp.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / f"data/voices/{args.name}_sliced"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        if f.is_file():
            f.unlink()

    import soundfile as sf

    # ---- 说话段 ----
    if args.keep_talk_slices:
        src = ROOT / "data/voices/girlish_sliced"
        talk = [(f.name, sf.info(str(f)).duration) for f in sorted(src.iterdir())
                if f.suffix == ".wav"]
        talk_dir = src
        talk_prefix_copy = True
        print(f"[talk] 复用已校对切片 {len(talk)} 条 {sum(d for _, d in talk):.1f}s")
    else:
        tw = tmp / "talk_cut.wav"
        cut_range(Path(args.talk_src), tw, args.talk_start, args.talk_end)
        talk_dir = tmp / "talk_seg"
        shutil.rmtree(talk_dir, ignore_errors=True)
        talk = segment(tw, talk_dir, "talk", args.min_sec, args.max_sec, args.top_db)
        talk_prefix_copy = False
        print(f"[talk] 两级切分 {len(talk)} 条 {sum(d for _, d in talk):.1f}s")

    talk_tot = sum(d for _, d in talk)

    # ---- 唱歌段 ----
    sw = tmp / "sing_cut.wav"
    cut_range(Path(args.sing_src), sw, args.sing_start, args.sing_end)
    sing_dir = tmp / "sing_seg"
    shutil.rmtree(sing_dir, ignore_errors=True)
    sing = segment(sw, sing_dir, "sing", args.min_sec, args.max_sec, args.top_db)
    print(f"[sing] 两级切分 {len(sing)} 条 {sum(d for _, d in sing):.1f}s")

    # ---- 按目标占比抽样唱歌 ----
    r = args.sing_ratio
    target = talk_tot * r / (1 - r) if r < 1 else sum(d for _, d in sing)
    pool = sing[:]
    random.shuffle(pool)
    picked, acc = [], 0.0
    for name, d in pool:
        if acc + d > target * 1.03:
            continue
        picked.append((name, d))
        acc += d
        if acc >= target * 0.97:
            break
    print(f"[mix] 唱歌抽样 {len(picked)} 条 {acc:.1f}s（目标 {target:.1f}s）")

    for name, _ in talk:
        dst = out_dir / (("talk_" + name) if talk_prefix_copy else name)
        shutil.copy(talk_dir / name, dst)
    for name, _ in picked:
        shutil.copy(sing_dir / name, out_dir / name)

    durs = [sf.info(str(f)).duration for f in out_dir.iterdir() if f.suffix == ".wav"]
    total = sum(durs)
    meta = {
        "name": args.name, "min_sec": args.min_sec, "max_sec": args.max_sec,
        "seed": args.seed, "segmentation": "w_select.fine_cands (两级切分)",
        "talk": {"clips": len(talk), "sec": round(talk_tot, 1),
                 "share": round(talk_tot / total, 4),
                 "source": str(args.talk_src), "range": [args.talk_start, args.talk_end],
                 "reused_corrected_slices": bool(args.keep_talk_slices)},
        "sing": {"clips": len(picked), "sec": round(acc, 1),
                 "share": round(acc / total, 4),
                 "source": str(args.sing_src), "start": args.sing_start},
        "total_clips": len(durs), "total_sec": round(total, 1),
        "dur_min": round(min(durs), 2), "dur_max": round(max(durs), 2),
        "bucket": bucket_hist(durs),
        "min_num_duplication": None if len(durs) >= 100 else max(2, int(100 / len(durs))),
    }
    (out_dir.parent / f"{args.name}_dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n切片 -> {out_dir}")
    print(f"下一步：对 {out_dir} 跑 ASR，人工校对后写 data/voices/{args.name}.list")


if __name__ == "__main__":
    main()
