#!/usr/bin/env python3
"""探针：emotion2vec 的 9 类分布在短音频上是否退化成 one-hot。

动机（2026-08）：prosody 报告里 prod 是 99.98% neutral、IndexTTS 各臂精确 1.0 happy，
而真人 162s 是 57.4% neutral / 27.5% happy / 12.7% surprised。三者的**窗数**分别是
1 / 1 / 27 —— 如果 emotion2vec 的 utterance softmax 本身接近 one-hot，那么
"prod 100% neutral vs 真人 57% neutral" 根本不是同量纲比较：它是拿 1 个 one-hot 窗
和 27 个 one-hot 窗的平均比，前者天然只能是 0 或 1。

本探针回答三问：
  1. 真人的**单个** 6s 窗是否也接近 one-hot？（若是，则窗级 argmax 无判别力）
  2. 真人窗级标签的经验分布 —— prod 的 100% neutral 落在其中的什么位置？
     （若 neutral 占 57%，那么随机抽一窗得到 100% neutral 的概率就是 57%，无显著性）
  3. 缩短窗长 + 重叠采样能否让 8s 音频拿到足够多窗？代价是分布随窗长漂移多少？

用法（GPT-SoVITS venv）:
    GPT-SoVITS/.venv/bin/python singing/probe_emo_window.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "singing"))

from eval_tts_prosody import EMO_LABELS, REF_WAV, SR, Emo, load  # noqa: E402

REF_TALK_SEC = 162.0
PROD = ROOT / "data/songs/情歌/mv/speech.wav"
IDX = ROOT / "data/songs/情歌/mv/indextts/sweep1"


def windows(y: np.ndarray, win: float, hop: float) -> list[np.ndarray]:
    n, h = int(win * SR), int(hop * SR)
    out = [y[i:i + n] for i in range(0, max(len(y) - n + 1, 1), h)]
    out = [s for s in out if len(s) >= SR]
    return out or [y]


def score_windows(emo: Emo, y: np.ndarray, win: float, hop: float):
    """返回 (窗级分布矩阵 [n_win, 9], 平均分布)。"""
    rows = []
    for s in windows(y, win, hop):
        r = emo.m.generate(s.astype(np.float32), granularity="utterance",
                           extract_embedding=False)[0]
        rows.append(np.asarray(r["scores"], dtype=np.float64))
    M = np.stack(rows)
    return M, M.mean(axis=0)


def fmt(d: np.ndarray, k: int = 4) -> str:
    idx = np.argsort(-d)[:k]
    return "  ".join(f"{EMO_LABELS[i]}={d[i]:.3f}" for i in idx if d[i] > 1e-3)


def main():
    emo = Emo()
    ref = load(str(REF_WAV))[: int(REF_TALK_SEC * SR)]
    prod = load(str(PROD))
    base = load(str(IDX / "base.wav"))
    h06 = load(str(IDX / "h06.wav"))

    print(f"\n真人说话段 {len(ref)/SR:.1f}s | prod {len(prod)/SR:.2f}s | "
          f"idx_base {len(base)/SR:.2f}s")

    # --- Q1: 单窗是否 one-hot ---
    print("\n" + "=" * 78)
    print("Q1. 真人的单个 6s 窗是否也是 one-hot？（不重叠，与现协议同参数）")
    M, avg = score_windows(emo, ref, 6.0, 6.0)
    top1 = M.max(axis=1)
    print(f"  {M.shape[0]} 窗；窗内最大概率 max: 中位 {np.median(top1):.4f} "
          f"p10 {np.percentile(top1, 10):.4f} min {top1.min():.4f}")
    print(f"  top1>0.99 的窗: {int((top1 > 0.99).sum())}/{M.shape[0]}")
    print(f"  窗级平均分布: {fmt(avg)}")

    # --- Q2: 真人窗级 argmax 的经验分布 vs prod 的单窗 ---
    print("\n" + "=" * 78)
    print("Q2. 真人窗级 argmax 的经验分布（prod 单窗 = 从这个分布抽 1 个样本）")
    labs = [EMO_LABELS[i] for i in M.argmax(axis=1)]
    for lab in sorted(set(labs), key=lambda x: -labs.count(x)):
        c = labs.count(lab)
        print(f"  {lab:10s} {c:3d}/{M.shape[0]}  {c / M.shape[0]:.3f}")
    for name, y in (("prod", prod), ("idx_base", base), ("idx_h06", h06)):
        Mg, avgg = score_windows(emo, y, 6.0, 6.0)
        lab = EMO_LABELS[int(avgg.argmax())]
        p = labs.count(lab) / M.shape[0]
        print(f"  {name:9s} {Mg.shape[0]} 窗 -> {lab}；真人窗里该标签占 {p:.3f}"
              f"  => 单窗判决的显著性 p≈{p:.2f}")

    # --- Q3: 缩短窗长 + 重叠 ---
    print("\n" + "=" * 78)
    print("Q3. 窗长/步长扫描：8s 音频能拿到多少窗，真人分布随窗长漂移多少")
    print(f"  {'win/hop':>10s} {'真人窗':>6s} {'真人 neutral':>12s} {'真人 happy':>10s} "
          f"{'prod窗':>6s} {'prod neutral':>12s} {'base窗':>6s} {'base happy':>10s}")
    ni = EMO_LABELS.index("neutral")
    hi = EMO_LABELS.index("happy")
    for win, hop in ((6.0, 6.0), (6.0, 1.0), (3.0, 3.0), (3.0, 0.5),
                     (2.0, 2.0), (2.0, 0.5), (1.5, 0.25)):
        _, ar = score_windows(emo, ref, win, hop)
        Mp, ap = score_windows(emo, prod, win, hop)
        Mb, ab = score_windows(emo, base, win, hop)
        nr = len(windows(ref, win, hop))
        print(f"  {win:4.1f}/{hop:4.2f} {nr:6d} {ar[ni]:12.3f} {ar[hi]:10.3f} "
              f"{Mp.shape[0]:6d} {ap[ni]:12.3f} {Mb.shape[0]:6d} {ab[hi]:10.3f}")


if __name__ == "__main__":
    main()
