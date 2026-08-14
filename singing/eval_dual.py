"""双锚点配对评测：SVC 输出同时对 svs_vocal.wav 与真实参考打分。

## 为什么要两个锚点

用户的目标函数是「SVS 的音色已达标、只是音准不行」，所以 SVC 的任务是
**保住 SVS 音色 + 用原唱主唱修回音准**。于是：
  - **主指标 = SIM vs svs_vocal.wav**（音色保住了多少）
  - 参考指标 = SIM vs 真实参考 lead（绝对音色，由 SVS 那一环负责）
  - 音准 = 输出 F0 与歌曲主唱 F0 的偏差（SVC 应当把 SVS 的音准问题修掉）

## 协议要点（都是踩过的坑）

- **公共窗口配对**：`eval_sim` 的 RMS 门（<0.01 丢弃）在每个臂独立生效，两臂会被
  评在不同窗口集上（AGENTS.md 记录冬眠上 60 vs 61 窗把增益从 +0.0073 变成 +0.0042）。
  这里只保留**所有臂都通过门**的窗口。
- 3s 非重叠窗、窗 vs 整文件嵌入，与 `eval_sim` 同协议，所以 A 列可与本仓库历史
  SIM 直接比较。
- 分桶用**歌曲主唱 F0**（target_f0），而不是各臂自己的 F0：桶标签必须在臂之间
  一致才能配对。
- 音准用 50Hz 网格上的逐帧半音差中位值（只在两边都浊音的帧上算）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rookie/voice")
from eval_sim import SV, load16k, cos  # noqa: E402

F0_RATE = 50.0
BANDS = [(0, 300), (300, 380), (380, 460), (460, 547), (547, 900)]


def windows(wav: np.ndarray, win: float) -> list[tuple[float, int]]:
    """返回 [(起始秒, 样本偏移)]，含 RMS 门（与 eval_sim 一致：<0.01 丢弃）。"""
    n = int(win * 16000)
    out = []
    for s in range(0, len(wav) - n, n):
        seg = wav[s:s + n]
        if np.sqrt(np.mean(seg**2)) < 0.01:
            continue
        out.append((s / 16000.0, s))
    return out


def semitone_err(gen_f0: np.ndarray, ref_f0: np.ndarray) -> tuple[float, float]:
    """两条 50Hz F0 的逐帧半音差：返回 (中位绝对误差, >50cents 的帧比例)。"""
    n = min(len(gen_f0), len(ref_f0))
    a, b = gen_f0[:n], ref_f0[:n]
    m = (a > 0) & (b > 0)
    if m.sum() == 0:
        return float("nan"), float("nan")
    d = np.abs(12 * np.log2(a[m] / b[m]))
    return float(np.median(d)), float((d > 0.5).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True, help="待评测的 SVC 输出（可多个臂）")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--svs", required=True, help="主锚点：svs_vocal.wav")
    ap.add_argument("--ref", required=True, help="参考锚点：真实参考人声 lead")
    ap.add_argument("--target-f0", required=True, help="歌曲主唱 F0（分桶 + 音准基准）")
    ap.add_argument("--gen-f0", nargs="*", default=[], help="各臂输出 F0（可选，用于音准）")
    ap.add_argument("--svs-f0", default="", help="SVS 输出 F0（可选，用于对照音准）")
    ap.add_argument("--win", type=float, default=3.0)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(g)) or os.path.basename(g)
                             for g in args.gen]
    if len(labels) != len(args.gen):
        sys.exit("--labels 数量与 --gen 不一致")

    sv = SV()
    anchor_svs = sv.emb(load16k(args.svs))
    anchor_ref = sv.emb(load16k(args.ref))
    # 长参考的整文件前向会把 PyTorch 缓存池撑到 ~19GB（537s 素材实测 reserved
    # 19.48GB vs allocated 0.21GB），而后续每个窗只要 0.34GB。不清掉会在 16GB 卡上
    # 随机 OOM（见 eval_sim.py 同处注释）。embedding 已算完，清理不改数值。
    torch.cuda.empty_cache()
    tf0 = np.load(args.target_f0)

    # ---- 公共窗口：所有臂都过 RMS 门，且该窗歌曲 F0 有足够浊音帧 ----
    wavs = [load16k(g) for g in args.gen]
    sets = [set(int(t * 1000) for t, _ in windows(w, args.win)) for w in wavs]
    common = set.intersection(*sets) if sets else set()
    keep = []
    for ms in sorted(common):
        t = ms / 1000.0
        fs = tf0[int(t * F0_RATE):int((t + args.win) * F0_RATE)]
        v = fs[fs > 0]
        if len(v) < 20:
            continue
        keep.append((t, float(np.median(v))))
    if not keep:
        sys.exit("没有公共可用窗口")
    print(f"公共窗口 {len(keep)}（各臂独立窗数 {[len(s) for s in sets]}）")
    print(f"主锚点 SVS : {args.svs}")
    print(f"参考锚点   : {args.ref}\n")
    # ---- 逐臂逐窗打分 ----
    n = int(args.win * 16000)
    S: dict[str, np.ndarray] = {}   # 对 SVS 的相似度
    R: dict[str, np.ndarray] = {}   # 对真实参考的相似度
    for lbl, w in zip(labels, wavs):
        ss, rr = [], []
        for t, _ in keep:
            e = sv.emb(w[int(t * 16000):int(t * 16000) + n])
            ss.append(cos(e, anchor_svs))
            rr.append(cos(e, anchor_ref))
        S[lbl], R[lbl] = np.array(ss), np.array(rr)

    med = np.array([m for _, m in keep])
    print(f"{'臂':<16}{'S:vs SVS':>11}{'R:vs 真实参考':>15}")
    for lbl in labels:
        print(f"{lbl:<16}{S[lbl].mean():>11.4f}{R[lbl].mean():>15.4f}")

    # SVS 自身对两个锚点的分数 = 天花板/基线参照
    svs_wav = load16k(args.svs)
    ss, rr = [], []
    for t, _ in keep:
        seg = svs_wav[int(t * 16000):int(t * 16000) + n]
        if len(seg) < n:
            continue
        e = sv.emb(seg)
        ss.append(cos(e, anchor_svs))
        rr.append(cos(e, anchor_ref))
    print(f"{'(SVS 自身)':<16}{np.mean(ss):>11.4f}{np.mean(rr):>15.4f}"
          f"   <- S 列的天花板 / R 列的目标值")

    print(f"\n分桶（按歌曲主唱 F0）  S = vs SVS, R = vs 真实参考")
    hdr = f"  {'Hz':<12}{'n':>4}"
    for lbl in labels:
        hdr += f"{lbl[:9]+'/S':>13}{lbl[:9]+'/R':>13}"
    print(hdr)
    for lo, hi in BANDS:
        k = (med >= lo) & (med < hi)
        if k.sum() < 2:
            continue
        line = f"  {str(lo)+'-'+str(hi):<12}{int(k.sum()):>4}"
        for lbl in labels:
            line += f"{S[lbl][k].mean():>13.4f}{R[lbl][k].mean():>13.4f}"
        print(line)

    # ---- 配对差 + 符号检验（第一个臂为基线）----
    if len(labels) >= 2:
        from math import comb
        base = labels[0]
        print(f"\n配对差（基线 = {base}）")
        for lbl in labels[1:]:
            for tag, D in (("S:vs SVS", S[lbl] - S[base]), ("R:vs 参考", R[lbl] - R[base])):
                wins = int((D > 0).sum())
                nn = len(D)
                p = 2 * sum(comb(nn, i) for i in range(min(wins, nn - wins) + 1)) / 2**nn
                print(f"  {lbl:<14}{tag:<12}Δ={D.mean():+.4f}  wins={wins}/{nn}"
                      f"  p_sign={min(p,1.0):.4f}")

    # ---- 音准：输出 F0 vs 歌曲主唱 F0 ----
    f0map = dict(zip(labels, args.gen_f0)) if args.gen_f0 else {}
    if f0map or args.svs_f0:
        print(f"\n音准（vs 歌曲主唱 F0；只在两边都浊音的帧上算）")
        print(f"  {'臂':<16}{'中位|半音差|':>13}{'>50cents帧':>12}")
        if args.svs_f0 and os.path.exists(args.svs_f0):
            e, r = semitone_err(np.load(args.svs_f0), tf0)
            print(f"  {'(SVS 输入)':<16}{e:>13.4f}{r:>12.1%}")
        for lbl in labels:
            p = f0map.get(lbl)
            if p and os.path.exists(p):
                e, r = semitone_err(np.load(p), tf0)
                print(f"  {lbl:<16}{e:>13.4f}{r:>12.1%}")


if __name__ == "__main__":
    main()
