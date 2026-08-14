#!/usr/bin/env python3
"""配对评分：只在两 arm 都通过 RMS 门限的公共窗口上比较，消除 eval_sim 窗数不一致的偏差。

用法: paired_eval.py --ref <refsep2/lead.wav> --f0 <target_f0.npy> --ref-f0 <prompt_f0.npy>
                     --a <base/vocal.wav> --b <knn/vocal.wav> [--win 3.0]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rookie/voice")
from eval_sim import SV, cos, load16k  # noqa: E402

F0_RATE = 50


def windows(wav, win, f0, ref_med):
    """返回 {start: (t0, med, semis)}，只含通过 F0 有效性检查的窗（不做 RMS 门限）。"""
    out = {}
    for start in range(0, len(wav) - win, win):
        t0 = start / 16000
        fa, fb = int(t0 * F0_RATE), int((t0 + win / 16000) * F0_RATE)
        fseg = f0[fa:fb]
        v = fseg[fseg > 0]
        if len(v) < 20:
            continue
        med = float(np.median(v))
        out[start] = (t0, med, 12 * np.log2(med / ref_med))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--f0", required=True)
    ap.add_argument("--ref-f0", required=True)
    ap.add_argument("--a", required=True, help="arm A (base)")
    ap.add_argument("--b", required=True, help="arm B (knn)")
    ap.add_argument("--labels", nargs=2, default=["base", "knn"])
    ap.add_argument("--win", type=float, default=3.0)
    args = ap.parse_args()

    sv = SV()
    ref_emb = sv.emb(load16k(args.ref))
    # 长参考的整文件前向会把 PyTorch 缓存池撑到 ~19GB（537s 素材实测 reserved
    # 19.48GB vs allocated 0.21GB），而后续每个窗只要 0.34GB。不清掉会在 16GB 卡上
    # 随机 OOM（见 eval_sim.py 同处注释）。embedding 已算完，清理不改数值。
    torch.cuda.empty_cache()
    ref_f0 = np.load(args.ref_f0)
    ref_med = float(np.median(ref_f0[ref_f0 > 0]))
    f0 = np.load(args.f0)
    win = int(args.win * 16000)

    wa, wb = load16k(args.a), load16k(args.b)
    ca, cb = windows(wa, win, f0, ref_med), windows(wb, win, f0, ref_med)

    # 公共窗口 + 两侧都过 RMS 门限
    common, gated_a, gated_b = [], 0, 0
    for s in sorted(set(ca) & set(cb)):
        ra = np.sqrt(np.mean(wa[s : s + win] ** 2)) >= 0.01
        rb = np.sqrt(np.mean(wb[s : s + win] ** 2)) >= 0.01
        if ra and rb:
            common.append(s)
        else:
            gated_a += not ra
            gated_b += not rb

    if not common:
        print("no common windows")
        return

    sa = np.array([cos(sv.emb(wa[s : s + win]), ref_emb) for s in common])
    sb = np.array([cos(sv.emb(wb[s : s + win]), ref_emb) for s in common])
    semis = np.array([abs(ca[s][2]) for s in common])

    la, lb = args.labels
    print(f"ref median F0 = {ref_med:.0f} Hz   paired windows = {len(common)}"
          f"   (RMS-gated: {la} {gated_a}, {lb} {gated_b})")
    print(f"  {la:<6} mean={sa.mean():.4f}  median={np.median(sa):.4f}")
    print(f"  {lb:<6} mean={sb.mean():.4f}  median={np.median(sb):.4f}")
    d = sb - sa
    print(f"  gain   mean={d.mean():+.4f}  median={np.median(d):+.4f}"
          f"  wins={int((d > 0).sum())}/{len(d)}"
          f"  p_sign={sign_test(d):.4f}")

    print(f"  {'semitones':<12}{'n':>4}{la:>9}{lb:>9}{'gain':>9}")
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 99)]:
        m = (semis >= lo) & (semis < hi)
        if m.sum():
            print(f"  {f'{lo}-{hi}':<12}{int(m.sum()):>4}{sa[m].mean():>9.4f}"
                  f"{sb[m].mean():>9.4f}{d[m].mean():>+9.4f}")


def sign_test(d):
    """双尾符号检验 p 值（精确二项）。"""
    from math import comb
    n = int((d != 0).sum())
    k = int((d > 0).sum())
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2**n)


if __name__ == "__main__":
    main()
