#!/usr/bin/env python
"""漏音检测一致性分析 + 听感校准素材导出。

- 对比两个独立说话人嵌入模型（ERes2NetV2 / CAMPPlus）对同一批窗口的漏音判定：
  Spearman 相关、TOP-k 重合度、逐窗散点。
- 尖峰持续性分析：真实漏音来自一个乐句（>1s），孤立 1s 尖峰更像嵌入噪声。
- 把「两个模型都报 / 只有一个报」的窗口连同原唱干声切成 wav 片段，
  供人耳对照（ground truth 校准）。

用法:
  seed-vc/.venv/bin/python singing/leak_agree.py \
      data/songs/雨爱/leak_windows_1s.csv data/songs/雨爱/leak_windows_1s_campplus.csv \
      --out data/songs/雨爱/vocal.wav --target data/songs/雨爱/sep2/lead.wav \
      --clips-dir data/songs/雨爱/leak_clips [--top 12] [--neigh 2]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import soundfile as sf
from scipy.stats import spearmanr


def load_csv(p: str) -> dict[float, dict[str, float]]:
    rows = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            t = float(r["t0"])
            rows[t] = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items()}
    return rows


def top_by(d: dict[float, dict[str, float]], key: str, k: int) -> list[float]:
    items = [(t, r[key]) for t, r in d.items() if not np.isnan(r[key])]
    items.sort(key=lambda x: -x[1])
    return [t for t, _ in items[:k]]


def persistence(d: dict[float, dict[str, float]], key: str, neigh: float) -> None:
    """尖峰（顶部5%）是否孤立：±neigh 秒内还有别的尖峰 = 聚集（更像真实漏音）。"""
    ts = sorted(d)
    vals = np.array([d[t][key] for t in ts])
    thresh = np.percentile(vals, 95)
    spike_idx = [i for i, v in enumerate(vals) if v >= thresh]
    n_isolated = 0
    for i in spike_idx:
        t = ts[i]
        j0 = max(0, np.searchsorted(ts, t - neigh) - 1)
        j1 = min(len(ts), np.searchsorted(ts, t + neigh) + 1)
        others = [k for k in spike_idx if k != i and j0 <= k < j1]
        if not others:
            n_isolated += 1
    print(f"  {key}: 顶部5%尖峰 {len(spike_idx)} 个, 聚集(±{neigh:.0f}s内有其他尖峰) "
          f"{len(spike_idx) - n_isolated}, 孤立 {n_isolated}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="至少两个 1s 漏音 CSV")
    ap.add_argument("--names", nargs="+", default=None, help="CSV 显示名（与 csvs 同序）")
    ap.add_argument("--out", help="输出 vocal.wav（切听感片段用）")
    ap.add_argument("--target", help="原唱干声 lead.wav（切对照片段用）")
    ap.add_argument("--clips-dir", default=None, help="听感校准片段输出目录")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--neigh", type=float, default=2.0, help="持续性邻域(秒)")
    ap.add_argument("--clip-len", type=float, default=2.5, help="听感片段半宽(秒)")
    args = ap.parse_args()

    datas = [load_csv(p) for p in args.csvs]
    names = args.names or [os.path.basename(p).replace("leak_windows_", "").replace(".csv", "") for p in args.csvs]

    # ---- 1. 两两相关（公共窗口）----
    print("== 逐窗指标相关（Spearman，公共窗口）==")
    common = sorted(set(datas[0]) & set(datas[1]))
    print(f"公共窗口 {len(common)}")
    for key in ["A", "C", "C_imp", "excess", "margin"]:
        a = np.array([datas[0][t][key] for t in common])
        b = np.array([datas[1][t][key] for t in common])
        rho, p = spearmanr(a, b)
        print(f"  {key:<7} rho={rho:+.3f}  p={p:.2e}")

    # ---- 2. TOP-k 重合 ----
    print(f"\n== 各模型 TOP-{args.top}（按 excess）==")
    for name, d in zip(names, datas):
        top = top_by(d, "excess", args.top)
        print(f"  {name:<14} {['%.0fs' % t for t in top]}")
    s1, s2 = set(top_by(datas[0], "excess", args.top)), set(top_by(datas[1], "excess", args.top))
    print(f"  TOP-{args.top} 重合: {len(s1 & s2)}/{args.top}  并集 {len(s1 | s2)}")

    # ---- 3. 持续性 ----
    print(f"\n== 尖峰持续性（邻域 ±{args.neigh:.0f}s，真实漏音应成群出现）==")
    for name, d in zip(names, datas):
        persistence(d, "excess", args.neigh)
        persistence(d, "margin", args.neigh)

    # ---- 4. 听感校准片段 ----
    if args.clips_dir and args.out:
        os.makedirs(args.clips_dir, exist_ok=True)
        union = set()
        for d in datas:
            union |= set(top_by(d, "excess", args.top))
        for d in datas:  # margin 最低的也切
            for t in sorted(d, key=lambda t: d[t]["margin"])[:5]:
                union.add(t)
        out_w, sr = sf.read(args.out)
        tgt_w, tgt_sr = sf.read(args.target) if args.target else (None, None)
        half = int(args.clip_len * sr)
        half_tgt = int(args.clip_len * tgt_sr) if tgt_sr else 0
        n = 0
        for t in sorted(union):
            i0 = max(0, int(t * sr) - half)
            i1 = min(len(out_w), int(t * sr) + half)
            tag = f"{t:06.1f}s"
            sf.write(os.path.join(args.clips_dir, f"out_{tag}.wav"), out_w[i0:i1], sr)
            if tgt_w is not None:
                j0 = max(0, int(t * tgt_sr) - half_tgt)
                j1 = min(len(tgt_w), int(t * tgt_sr) + half_tgt)
                sf.write(os.path.join(args.clips_dir, f"orig_{tag}.wav"), tgt_w[j0:j1], tgt_sr)
            n += 1
        print(f"\n听感片段已写入 {args.clips_dir}（{n} 个窗口 × out/orig）")
        print("并集窗口:", sorted(union))


if __name__ == "__main__":
    main()
