#!/usr/bin/env python3
"""段落级 SVC 的核心逻辑（不跑推理，供 run_segprompt.py 调用）。

动机：探针 downext/singlow（docs/lowregister-rescore.md）发现主歌低音差的真因
是 prompt 低音区只有说话素材；全曲共用一条 prompt（med~331Hz）对低音主歌是
「8 个半音之外的说话素材」。段落级换参考让每段配一条音域匹配的唱歌参考。

用户约束（2026-08）：
  1. 切分后每段**单独算音域**（段 med F0 不能由全曲 med 代替）；
  2. 切分点尽量取**纯人声中最长的静音**，不把一句话切成两半。

切分算法（2026-08 静态重写，不跑 SVC）：
  目标：段长必须落在 [seg_min, seg_max]（默认 12-19s），防止 SoulX 的
  build_vocal_segments 在推理时把段二次切碎，并避免 CFM decoder 单块过长：
  2026-08-26 实测单块推理 >~19s 时输出能量随时间塌缩（末段 RMS 掉到 ~30%），
  但 svc_max=22 时窗口被模型内部切成 ≤19s 块，实测 0/14 衰减、SIM 三档最高，
  所以内容段长上限取 19s：扩展 ±2s 后 23s 超上限，shrink 只裁尾部冗余到 22s，
  前缀 2s 完整保留、后缀恰 1.0s（用户定稿：后缀 1s 足够）。
  手段：候选切分点 = 每个 >= min_run 的静音 run。run 时长 >= 2*pad 出
  **双刀**（前一段在 rs+pad 结束、后一段在 re-pad 开始，中间 L-2*pad 秒的
  纯静音作缓冲留零、不喂 SVC）；run 时长 < 2*pad 出**单刀**（中点一刀，
  前后段均分静音）。每个候选点收益 = 所在 run 时长（切在长静音上收益高），
  在段长硬约束下做 DP 最大化总收益 —— 即「切分点周围静音越长越好」。
  首/尾的大段静音（前奏/尾奏）作为隔离区，不进任何段。

段间冗余（2026-08，run_segprompt.py 内联实现）：每段前后各加 2s 再送 SVC ——
紧贴短静音的段首/段尾是冷启动，开头一两个字音质差；把邻接音频（上一段尾
巴/下一段头 + 共享静音）一起喂给模型，拼接时只取中间 [t0,t1]，把冗余裁掉。
相邻段扩展窗口允许重叠。扩展后总长必须 < 推理上限（SoulX 整段直推判定是
gt < max_seg_sec），超限时只裁尾部冗余（前缀 2s 永远保留，后缀 ≥1s）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F0_RATE = 50.0

# ---------------------------------------------------------------------------
# 切分：静音优先 + 时长约束（DP）
# ---------------------------------------------------------------------------


def silence_runs(f0: np.ndarray, min_run: float) -> List[Tuple[float, float]]:
    """f0==0 且时长 >= min_run 秒的静音 run -> [(start, end)] 秒。"""
    n = len(f0)
    runs = []
    i = 0
    while i < n:
        if f0[i] == 0:
            j = i
            while j < n and f0[j] == 0:
                j += 1
            if (j - i) / F0_RATE >= min_run:
                runs.append((i / F0_RATE, j / F0_RATE))
            i = j
        else:
            i += 1
    return runs


def cut_points(runs: List[Tuple[float, float]], pad: float) -> List[Tuple[float, float, int]]:
    """每个静音 run -> 候选切分点 [(pos, weight, run_id)]。

    run 时长 L >= 2*pad：双刀 -> (rs+pad, L/2) 与 (re-pad, L/2)（收益均分，
    两个都选中总收益 = L；中间的 L-2*pad 秒是段间纯静音缓冲，留零不进 SVC）。
    run 时长 L < 2*pad：单刀 -> 中点 (rs+re)/2，收益 L（前后段各分 L/2 静音）。
    """
    pts = []
    for rid, (rs, re) in enumerate(runs):
        L = re - rs
        if L >= 2 * pad:
            pts.append((rs + pad, L / 2, rid))
            pts.append((re - pad, L / 2, rid))
        else:
            pts.append(((rs + re) / 2, L, rid))
    pts.sort(key=lambda t: t[0])
    return pts


def _dp_segments(pts, A: float, B: float, seg_min: float, seg_max: float):
    """在候选切分点 pts 上做 DP，段长 ∈ [seg_min, seg_max]。

    段 = [A, p_i]（首）/ [p_j, p_i] / [p_i, B]（末），其中 p 是选中的切分点。
    同一 run 的双刀两点之间是缓冲段（无时长约束，纯静音留零）；
    其余段必须满足 seg_min <= 时长 <= seg_max（防止 SoulX 内部再切）。
    收益 = 选中点权重之和；返回选中的切分点列表（升序），无解返回 None。
    """
    m = len(pts)
    pos = [p for p, _, _ in pts]
    w = [w for _, w, _ in pts]
    dp = [-1e18] * m          # 选中 pts[i] 为最后一个边界的最大收益
    prev = [-2] * m           # 前驱索引；-1 = 从 A 起首段
    for i in range(m):        # 首段 [A, pos[i]]
        if seg_min <= pos[i] - A <= seg_max:
            dp[i] = w[i]
            prev[i] = -1
    for i in range(m):
        if dp[i] < 0:
            continue
        for j in range(i + 1, m):
            d = pos[j] - pos[i]
            if pts[i][2] == pts[j][2]:
                valid = True          # 同一 run 双刀两点 = 缓冲段
            else:
                valid = seg_min <= d <= seg_max
            if valid and dp[i] + w[j] > dp[j]:
                dp[j] = dp[i] + w[j]
                prev[j] = i
    best = -1
    for i in range(m):        # 末段 [pos[i], B]
        if dp[i] >= 0 and seg_min <= B - pos[i] <= seg_max:
            if best < 0 or dp[i] > dp[best]:
                best = i
    if best < 0:
        return None
    chain = []
    i = best
    while i >= 0:
        chain.append(pts[i])
        i = prev[i]
    chain.reverse()
    return chain


def split_optimal(f0: np.ndarray, min_run=0.3, pad=1.0,
                  seg_min=12.0, seg_max=19.0, _depth=0) -> List[Tuple[float, float]]:
    """静音优先 + 时长约束的分段，返回 [(start, end)] 秒。

    降级链（无解时逐级放宽，seg_max 永不放松 —— 防内部再切是硬约束）：
      1. min_run 降低：0.3 -> 0.2 -> 0.1（更多候选静音）；
      2. seg_min 降低：15 -> 12 -> 10 -> 8（允许更短段，仍是有效时长约束）。
    段覆盖范围 [A, B]：开头/结尾的 >= min_run 静音 run 作为前奏/尾奏隔离区，
    不进任何段（run_segprompt 摆回时间轴时该处留零）。
    """
    total = len(f0) / F0_RATE
    runs = silence_runs(f0, min_run)
    if not runs:
        raise ValueError(f"min_run={min_run} 无任何静音 run")

    # 前奏/尾奏隔离区：从两端向内扫描，遇到**长静音**就跳过，直到遇到真正的人声。
    # 不能用 runs[0]/runs[-1]——开头可能是 0.4s 气口短静音，真正的长前奏在后面
    # （情歌实测: runs[0]=[0,0.42] 短气口, runs[1]=[0.5,25.9] 才是 25s 前奏；
    # 若 A 定在 0.4，0.4→首个切分点(26s) 之间 25.6s 无点，首段约束无解）。
    # 判据：开头连续 >=1.0s 的静音（或覆盖开头 >=0.5s 的静音）视为前奏，
    # 其终点即 A；结尾对称取 B。短气口（<1s）不构成隔离区。
    voiced = f0 > 0
    # A：从 0 找第一个"实质发声段"的起点 —— 跳过开头所有长静音 run
    A = 0.0
    for rs, re in runs:
        if rs - A < 0.5:          # 与已隔离区相邻的静音（含气口间隙）都并入
            A = max(A, re)
        else:
            break
    # B：从结尾找最后一个"实质发声段"的终点 —— 跳过结尾所有长静音 run
    B = total
    for rs, re in reversed(runs):
        if B - re < 0.5:          # 与已隔离区相邻的静音都并入
            B = min(B, rs)
        else:
            break

    inner = [(rs, re) for rs, re in runs
             if rs >= A - 1e-6 and re <= B + 1e-6]
    pts = cut_points(inner, pad)
    chain = _dp_segments(pts, A, B, seg_min, seg_max)

    if chain is None and _depth == 0:
        for mr in (0.2, 0.1):
            try:
                return split_optimal(f0, min_run=mr, pad=pad,
                                     seg_min=seg_min, seg_max=seg_max, _depth=1)
            except ValueError:
                continue
        for sm in (12.0, 10.0, 8.0):
            try:
                return split_optimal(f0, min_run=min_run, pad=pad,
                                     seg_min=sm, seg_max=seg_max, _depth=1)
            except ValueError:
                continue
        raise ValueError(
            f"切分无解: min_run={min_run} pad={pad} seg_min={seg_min} "
            f"seg_max={seg_max}（歌可能整段连续无 >=0.1s 静音）")
    if chain is None:
        raise ValueError(f"min_run={min_run} 无解")

    # 段生成：相邻选中点若属同一 run（双刀），中间是纯静音缓冲，
    # 不进任何段（前段到前刀结束，后段从后刀开始）；其余点为段边界。
    segs = []
    prev_cut = A
    i = 0
    while i < len(chain):
        p, _, rid = chain[i]
        if i + 1 < len(chain) and chain[i + 1][2] == rid:
            segs.append((prev_cut, p))      # 前段尾部 = 前刀 rs+pad
            prev_cut = chain[i + 1][0]      # 后段头部 = 后刀 re-pad
            i += 2
        else:
            segs.append((prev_cut, p))
            prev_cut = p
            i += 1
    segs.append((prev_cut, B))
    # 剔除空段（防御）
    return [(a, b) for a, b in segs if b - a > 0.05]


# ---------------------------------------------------------------------------
# 兼容旧接口（run_segprompt 早期版本用），语义已切换为静音优先 DP
# ---------------------------------------------------------------------------


def split_silence(f0: np.ndarray, silence_min=0.2, seg_min=12.0,
                  seg_max=19.0, pad=1.0) -> List[Tuple[float, float]]:
    """按静音 + 时长约束切段（新算法：静音优先 DP）。"""
    return split_optimal(f0, min_run=silence_min, pad=pad,
                         seg_min=seg_min, seg_max=seg_max)


def split_bands(f0: np.ndarray, silence_min=0.2, seg_min=12.0,
                band_edges=(260.0, 370.0), pad=1.0,
                seg_max=19.0) -> List[Tuple[float, float, int]]:
    """先按静音切细段，再把相邻的同音域段聚合成组。

    返回 [(start, end, band)]，band=0 低 / 1 中 / 2 高（按 band_edges 分档）。
    切点全在静音处（不切断句子）；每段先单独算 med 再判定音域——满足
    「每段单独算音域」+「切分点在静音」。相邻同 band 段合并成一个参考组；
    组时长同样受 [seg_min, seg_max] 约束（超长组内部按静音再切）。
    """
    base = split_silence(f0, silence_min=silence_min, seg_min=seg_min,
                         seg_max=seg_max, pad=pad)
    if not base:
        return []
    tiles = []  # (start, end, band)
    for a, b in base:
        m = seg_med_f0(f0, a, b)
        band = 0 if m <= band_edges[0] else (2 if m >= band_edges[1] else 1)
        tiles.append((a, b, band))
    # 聚合相邻同 band，但组时长不得超过 seg_max（超了保持细段，防内部再切）
    groups = []
    for a, b, band in tiles:
        if groups and groups[-1][2] == band \
                and (b - groups[-1][0]) <= seg_max:
            groups[-1] = (groups[-1][0], b, band)
        else:
            groups.append((a, b, band))
    return groups


def seg_med_f0(f0: np.ndarray, a: float, b: float) -> float:
    """段内 f0>0 的中位（Hz）；无浊音返回 nan。"""
    v = f0[int(a * F0_RATE): int(b * F0_RATE)]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else float("nan")


def pick_prompt(cands: list, med_f0: float, budget: float,
                max_shift=8) -> List[Tuple[dict, int]]:
    """按段音域选候选：(cand, shift)，移调量最小优先、落进目标音域者优先。

    目标音域 = [med·2^{-1/12}, med·2^{+1/12}]。不足预算时允许未落格候选
    （钳到 max_shift 后最接近）。返回 [(cand, shift)]，调用方负责渲染。
    """
    lo, hi = med_f0 * 2 ** (-1 / 12), med_f0 * 2 ** (1 / 12)
    avail = []
    for c in cands:
        sh = int(round(12 * np.log2(med_f0 / c["med"])))
        sh = max(-max_shift, min(max_shift, sh))
        reached = c["med"] * 2 ** (sh / 12)
        inside = lo <= reached <= hi
        dist = abs(12 * np.log2(reached / med_f0))
        avail.append((0 if inside else 1, abs(sh), dist, c, sh))
    avail.sort(key=lambda t: (t[0], t[1], t[2]))

    picked, used, used_copies = [], 0.0, set()
    for ins, ash, dist, c, sh in avail:
        if used >= budget:
            break
        if (c["idx"], sh) in used_copies:
            continue
        used_copies.add((c["idx"], sh))
        picked.append((c, sh))
        used += c["dur"]
    if not picked:
        # 候选池音域太窄：放宽目标（±max_shift 都够不着才空）
        for c in cands:
            if used >= budget:
                break
            picked.append((c, 0))
            used += c["dur"]
    return picked
