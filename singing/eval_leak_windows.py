#!/usr/bin/env python
"""小窗口漏音扫描: 逐窗口输出 A/C/excess/margin，找出漏音最高的短窗口。

协议与 eval_triad 完全一致（窗口 embedding vs 全局锚点、余弦），只是窗口更小、
并逐窗口输出明细。excess = C - mean(C_impostor) 扣掉「分离歌声」域偏置后的
真实原唱泄漏；margin = A - C（正 = 该窗口更像参考）。

用法:
  GPT-SoVITS/.venv/bin/python singing/eval_leak_windows.py --song-dir data/songs/雨爱 \
      --impostor <lead>... [--win 1.0] [--f0 target_f0.npy] [--csv-out leak_1s.csv]

输出:
  - 摘要统计（excess 的 mean/median/p90/max、margin<0 窗口数）
  - 按 excess 排序的最差窗口 TOP-N（含时间戳、A/C/Cimp/excess/margin/F0 中位数）
  - excess 分桶计数（<0 / 0-0.05 / 0.05-0.1 / >0.1）
  - 若给 --csv-out，把全部窗口写 CSV 供进一步分析
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from eval_triad import SV, load16k, windows, cos, score_against


def fmt(x: float) -> str:
    return f"{x:+.4f}" if x < 0 else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", help="data/songs/<name>/，自动推导 ref/target/out")
    ap.add_argument("--ref", help="参考音色真实语音（默认 <song-dir>/refsep2/lead.wav）")
    ap.add_argument("--target", help="原曲分离人声（默认 <song-dir>/sep2/lead.wav）")
    ap.add_argument("--out", default=None, help="转换输出（默认 <song-dir>/vocal.wav）")
    ap.add_argument("--impostor", nargs="*", default=None, metavar="LEAD",
                    help="其他歌手的分离人声 wav，用于扣域偏置")
    ap.add_argument("--win", type=float, default=1.0, help="分析窗长(秒)")
    ap.add_argument("--f0", help="输出音频对应的 F0 .npy（50Hz），给出则打印每窗 F0 中位数")
    ap.add_argument("--top", type=int, default=12, help="打印最差窗口数")
    ap.add_argument("--csv-out", default=None, help="全部窗口明细写到此 CSV")
    args = ap.parse_args()

    d = args.song_dir.rstrip("/") if args.song_dir else None
    ref = args.ref or (os.path.join(d, "refsep2", "lead.wav") if d else None)
    target = args.target or (os.path.join(d, "sep2", "lead.wav") if d else None)
    out = args.out or (os.path.join(d, "vocal.wav") if d else None)
    if not (ref and target and out):
        ap.error("需要 --song-dir，或同时给出 --ref/--target/--out")
    for p in [ref, target, out, *(args.impostor or [])]:
        if not os.path.exists(p):
            ap.error(f"文件不存在: {p}")

    sv = SV()
    ref_w, tgt_w, out_w = load16k(ref), load16k(target), load16k(out)
    a_ref, a_tgt = sv.emb(ref_w), sv.emb(tgt_w)
    imps = [sv.emb(load16k(p)) for p in (args.impostor or [])]
    # 长参考整文件前向会把缓存池撑大（同 eval_triad 注释），embedding 算完即清。
    torch.cuda.empty_cache()

    # ---- B: ref↔target 距离基线（与 eval_triad 同公式：两向窗口均值的平均）----
    _, b_r = score_against(sv, ref_w, {"t": a_tgt}, args.win)
    _, b_t = score_against(sv, tgt_w, {"r": a_ref}, args.win)
    B = float(np.mean([b_r["t"].mean(), b_t["r"].mean()]))

    # ---- 逐窗口扫描 ----
    anchors = {"ref": a_ref, "tgt": a_tgt}
    anchors.update({f"imp{i}": e for i, e in enumerate(imps)})
    times, sc = score_against(sv, out_w, anchors, args.win)
    A_, C_ = sc["ref"], sc["tgt"]
    if len(A_) == 0:
        ap.error("输出无可用窗口")
    C_imp = np.mean([sc[f"imp{i}"] for i in range(len(imps))], axis=0) if imps else np.zeros_like(A_)
    excess = C_ - C_imp
    margin = A_ - C_

    f0 = np.load(args.f0) if args.f0 else None
    med_f0 = np.full(len(times), np.nan)
    if f0 is not None:
        for i, t0 in enumerate(times):
            fa, fb = int(t0 * 50), int((t0 + args.win) * 50)
            v = f0[fa:fb]
            v = v[v > 0]
            if len(v) >= 10:
                med_f0[i] = float(np.median(v))

    print(f"ref    {ref}   {len(ref_w) / 16000:.1f}s")
    print(f"target {target}   {len(tgt_w) / 16000:.1f}s")
    print(f"out    {out}   {len(out_w) / 16000:.1f}s")
    print(f"窗长 {args.win}s，n={len(A_)}，impostor={len(imps)}，B(ref↔target)={B:.4f}\n")

    print(f"{'':<16}{'n':>5}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}")
    for name, arr in [("A", A_), ("C", C_), ("C_imp", C_imp), ("excess", excess), ("margin", margin)]:
        print(f"{name:<16}{len(arr):>5}{arr.mean():>9.4f}{np.median(arr):>9.4f}"
              f"{np.percentile(arr, 90):>9.4f}{arr.max():>9.4f}")
    print(f"\nmargin<0（更像原唱）的窗口: {(margin < 0).sum()}/{len(margin)}  "
          f"({100 * (margin < 0).mean():.1f}%)")
    print(f"excess 分桶: <0: {(excess < 0).sum()}  0-0.05: {((excess >= 0) & (excess < 0.05)).sum()}"
          f"  0.05-0.1: {((excess >= 0.05) & (excess < 0.1)).sum()}"
          f"  >0.1: {(excess >= 0.1).sum()}\n")

    print(f"=== 最差窗口 TOP-{args.top}（按 excess 降序）===")
    order = np.argsort(-excess)[: args.top]
    print(f"{'t0':>8}{'A':>9}{'C':>9}{'Cimp':>9}{'excess':>9}{'margin':>9}{'F0med':>8}")
    for i in order:
        f0s = f"{med_f0[i]:.0f}" if not np.isnan(med_f0[i]) else "-"
        print(f"{times[i]:>8.1f}{A_[i]:>9.4f}{C_[i]:>9.4f}{C_imp[i]:>9.4f}"
              f"{fmt(excess[i]):>9}{fmt(margin[i]):>9}{f0s:>8}")
    print("\n=== margin 最低 TOP-5（最不像参考的窗口）===")
    order2 = np.argsort(margin)[:5]
    print(f"{'t0':>8}{'A':>9}{'C':>9}{'Cimp':>9}{'excess':>9}{'margin':>9}{'F0med':>8}")
    for i in order2:
        f0s = f"{med_f0[i]:.0f}" if not np.isnan(med_f0[i]) else "-"
        print(f"{times[i]:>8.1f}{A_[i]:>9.4f}{C_[i]:>9.4f}{C_imp[i]:>9.4f}"
              f"{fmt(excess[i]):>9}{fmt(margin[i]):>9}{f0s:>8}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0", "A", "C", "C_imp", "excess", "margin", "f0_med"])
            for i in range(len(times)):
                w.writerow([f"{times[i]:.2f}", f"{A_[i]:.4f}", f"{C_[i]:.4f}",
                            f"{C_imp[i]:.4f}", f"{excess[i]:.4f}", f"{margin[i]:.4f}",
                            f"{med_f0[i]:.1f}" if not np.isnan(med_f0[i]) else ""])
        print(f"\n全部窗口已写入 {args.csv_out}")


if __name__ == "__main__":
    main()
