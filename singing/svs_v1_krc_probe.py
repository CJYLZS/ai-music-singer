#!/usr/bin/env python3
"""V1 探测：逐字 KRC 时间戳到底能不能当词级时间轴用。

docs/svs-manual-calibration.md 记着一条否定结论（绿色：KRC 逐字起点到最近浊音
起点 median 540ms / p75 2700ms）。p75 2700ms 大得可疑，先排除"整体偏移"这类
对齐 bug 再决定是否彻底放弃 —— 本脚本对 offset 做扫描，并分别统计
「每行首字」（人工设定的锚点）与「行内其余字」（可能是线性插值出来的）。

用法: SoulX-Singer/.venv/bin/python singing/svs_v1_krc_probe.py \
        --lrc data/songs/绿色.lrc --f0 data/songs/绿色/target_f0.npy
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svs_qc import voiced_runs, pitch_steps  # noqa: E402

F0_FPS = 50.0


def parse_krc(path: str):
    """-> [(t_sec, char, is_line_head)]，只取汉字。"""
    out = []
    for ln in open(path, encoding="utf-8"):
        toks = re.findall(r"<(\d+):(\d+[.:]\d+)>([^<]*)", ln.strip())
        head = True
        for mm, ss, text in toks:
            t = int(mm) * 60 + float(ss.replace(":", "."))
            for ch in text:
                if re.match(r"[\u4e00-\u9fff]", ch):
                    out.append((t, ch, head))
                    head = False
                    break  # 一个时间标签只对一个字
    return out


def gaps(times: np.ndarray, ref_frames: np.ndarray) -> np.ndarray:
    if len(ref_frames) == 0:
        return np.full(len(times), 9.99)
    ref = ref_frames / F0_FPS
    return np.abs(ref[None, :] - times[:, None]).min(axis=1)


def summarize(tag: str, g: np.ndarray) -> None:
    if len(g) == 0:
        print(f"{tag:<22} n=0")
        return
    print(f"{tag:<22} n={len(g):<4} median={np.median(g)*1000:6.0f}ms "
          f"p75={np.percentile(g,75)*1000:6.0f}ms "
          f"hit@80ms={np.mean(g<=0.08):5.1%} hit@150ms={np.mean(g<=0.15):5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrc", required=True)
    ap.add_argument("--f0", required=True)
    ap.add_argument("--scan", type=float, default=3.0, help="offset 扫描范围(s)")
    args = ap.parse_args()

    krc = parse_krc(args.lrc)
    if not krc:
        sys.exit("no per-char tags in " + args.lrc)
    f0 = np.load(args.f0)
    voiced = f0 > 0
    onsets = voiced_runs(voiced)
    landmarks = np.union1d(onsets, pitch_steps(f0))
    t = np.array([x[0] for x in krc])
    is_head = np.array([x[2] for x in krc])
    print(f"KRC 字数 {len(krc)}（行首 {int(is_head.sum())}）  "
          f"F0 {len(f0)/F0_FPS:.1f}s 浊音起点 {len(onsets)} 地标 {len(landmarks)}")

    # offset 扫描：以「行首字 -> 浊音起点」的 median 为目标
    best = (None, 1e9)
    for off in np.arange(-args.scan, args.scan + 1e-9, 0.02):
        m = float(np.median(gaps(t[is_head] + off, onsets)))
        if m < best[1]:
            best = (float(off), m)
    print(f"\n最优整体偏移 {best[0]:+.2f}s (行首 median {best[1]*1000:.0f}ms)")

    for off, label in ((0.0, "offset=0"), (best[0], f"offset={best[0]:+.2f}s")):
        print(f"\n-- {label}")
        summarize("行首字 vs 浊音起点", gaps(t[is_head] + off, onsets))
        summarize("行内字 vs 浊音起点", gaps(t[~is_head] + off, onsets))
        summarize("全部字 vs 任意地标", gaps(t + off, landmarks))


if __name__ == "__main__":
    main()
