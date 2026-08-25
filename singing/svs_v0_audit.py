#!/usr/bin/env python3
"""V0 可见性审计：量化"F0 证据最多能自动解决多少现有缺陷"。

对 qc 三类缺陷逐一回答：
  SP_VOICED  该休止附近(±500ms)存在 >=160ms 无声段吗？
             有 -> 时间轴修正可救(movable)；没有 -> 结构问题(structural,
             整段连续有声如哼唱间奏，任何词时间轴都救不了)
  ONSET      乐句起始音符 ±250ms 内有几个浊音段起点？恰好 1 个 -> 自动吸附；
             多个 -> 有歧义；0 个 -> F0 盲区
  STEP       换音边界 ±250ms 内有 F0 台阶吗？有 -> 自动吸附；无 -> 盲区
另统计全部有词音符当前位置 ±80ms 内存在任意 F0 地标(浊音边或台阶)的比例，
作为"外部时间轴提案能否被 F0 校验/微调"的基线。

用法: SoulX-Singer/.venv/bin/python singing/svs_v0_audit.py \
        --meta data/songs/X/svs/target/metadata.json [--out x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svs_qc import load_notes, voiced_runs, pitch_steps, SP, F0_FPS


def unvoiced_gaps(voiced: np.ndarray, min_frames: int = 8) -> np.ndarray:
    """无声段中心帧号数组（长度 >= min_frames 才算，默认 160ms）。"""
    d = np.diff(voiced.astype(np.int8), prepend=0, append=0)
    ends = np.flatnonzero(d == -1)
    starts = np.flatnonzero(d == 1)
    out = []
    for s, e in zip(starts, ends):
        if e - s >= min_frames:
            out.append((s + e) // 2)
    return np.array(out, dtype=int)


def audit(meta_path: str) -> dict:
    segs, rows, f0s = load_notes(meta_path)
    stat: Counter = Counter()
    detail = []
    for si, seg in enumerate(segs):
        f0 = f0s[si]
        voiced = f0 > 0
        onsets, steps = voiced_runs(voiced), pitch_steps(f0)
        gaps = unvoiced_gaps(voiced)
        rr = [x for x in rows if x["seg"] == si]
        for ri, r in enumerate(rr):
            a = int(round(r["rel"] * F0_FPS))
            prev = rr[ri - 1] if ri > 0 else None
            item = {"t": round(seg["time"][0] / 1000 + r["rel"], 2),
                    "text": r["text"], "dur": r["dur"]}
            if r["text"] == SP:
                if r["dur"] < 0.15:
                    continue
                win = voiced[a:a + int(r["dur"] * F0_FPS)]
                frac = float(win.mean()) if len(win) else 0.0
                if frac <= 0.35:
                    continue  # 干净休止，不是缺陷
                ng = (float(np.min(np.abs(gaps - a)) / F0_FPS)
                      if len(gaps) else 9.99)
                kind = "SP_MOVABLE" if ng <= 0.5 else "SP_STRUCTURAL"
                stat[kind] += 1
                item.update(kind=kind, near_gap_s=None if ng > 9 else round(ng, 3),
                            voiced_frac=round(frac, 2))
            else:
                landmarks = np.union1d(onsets, steps)
                vis = (float(np.min(np.abs(landmarks - a)) / F0_FPS)
                       if len(landmarks) else 9.99)
                stat["WORD_VISIBLE" if vis <= 0.08 else "WORD_BLIND"] += 1
                is_init = prev is None or prev["text"] == SP
                if is_init:
                    n = int(np.sum(np.abs(onsets - a) <= 0.25 * F0_FPS)) \
                        if len(onsets) else 0
                    kind = {0: "ONSET_BLIND", 1: "ONSET_SNAP"}.get(
                        n, "ONSET_AMBIGUOUS")
                    stat[kind] += 1
                    item.update(kind=kind, n_edges=n)
                elif prev["pitch"] != r["pitch"] and r["type"] != 3:
                    n = int(np.sum(np.abs(steps - a) <= 0.25 * F0_FPS)) \
                        if len(steps) else 0
                    kind = "STEP_OK" if n else "STEP_BLIND"
                    stat[kind] += 1
                    item.update(kind=kind, n_steps=n)
                else:
                    continue  # 句中同音高/slur：无参照物，本就不检查
            if len(detail) < 400:
                detail.append(item)
    return dict(stat), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    stat, detail = audit(args.meta)
    total_fixable = stat.get("SP_MOVABLE", 0) + stat.get("ONSET_SNAP", 0) \
        + stat.get("STEP_OK", 0)
    total_blind = stat.get("SP_STRUCTURAL", 0) + stat.get("ONSET_BLIND", 0) \
        + stat.get("ONSET_AMBIGUOUS", 0) + stat.get("STEP_BLIND", 0)
    words = stat.get("WORD_VISIBLE", 0) + stat.get("WORD_BLIND", 0)
    print(json.dumps(stat, ensure_ascii=False, indent=1))
    print(f"\nF0证据可自动解决: {total_fixable}  需人工/结构决策: {total_blind}")
    vw = stat.get("WORD_VISIBLE", 0)
    print(f"词音符位置±80ms内有F0地标(校验能力): {vw}/{words} = "
          f"{vw / max(words, 1):.1%}")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"stat": stat, "detail": detail}, ensure_ascii=False,
                       indent=1), encoding="utf-8")
        print("detail ->", args.out)


if __name__ == "__main__":
    main()
