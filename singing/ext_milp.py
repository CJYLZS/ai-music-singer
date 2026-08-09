"""prompt_ext 选段的 MILP 求解器：把「锚点 + ext」两阶段合并成一个 (段, 移调) 指派问题。

## 为什么要合并两阶段

原流水线里 `w_select.py` 先贪心选锚点、`w_ext.py` 再贪心填频段缺口，两者各自独立，
于是同一个源段可以被两阶段各用一次（原声 + 移调版），而候选池本身是**互不重叠的
分割**（143 段 / 95.6s，重叠对数 0），所以"一段素材出现两次"是可以精确定义并消除的。

## 结论先行：本模块不是默认路径（2026-08 撤回）

下面的 clip 实验选出了 `cov_anchor`，但**全曲 10 首配对复现后均值只有 +0.0012**
（5 首显著正 +0.005…+0.013、5 首显著负 -0.003…-0.008，t=0.49，符号检验 p=1.000），
而 `ext` 阶段耗时 12.9s -> 253.6s（19.7×）。故 `w_ext.py --select-method` 默认已改回
`greedy`，本模块保留为可选项。**注意 clip 实验只跑过绿色/冬眠/爱笑的眼睛 3 首，而
它们正是调参用的歌** —— n=3 可以扫参数，不足以定默认值。

效应按歌分向的机制**未找到**：select∩ext 交集（干预实验仅 +0.0034）、移调时长占比
（两版都被 anchor_ratio 锁在 ~45%，压根不是变量）、移调幅度（corr -0.119）、
song_med（阈值 8/10 但 LOO 掉到 5/10 = 瞎猜）四个候选全部否掉。详见 AGENTS.md。

## 三个目标的词典序

用户提的优先级是 ①音域覆盖完整 > ②选段不重叠 > ③移调最小。实测（2026-08，3 首歌
× 6 seed，clip 协议）结论与之**部分不符**，clip 上默认取 `cov_anchor`：

  - ③ 不能排第一：`shift_first` 会退化成"全部 0st"，覆盖塌到 33-54%，连 45s 都填不
    满（素材用光），三首歌全负，最差 -0.0195/-0.0241。
  - ① 排第一有效，但必须配干净锚点下限。纯覆盖优先（`cov_only`）跨歌均值 +0.0033
    但两首为负；加上"0st 段 >= 55% 总时长"后跨歌均值 +0.0050 且 0 负。
  - ② 是免费的：它从来不是瓶颈成本，而且**加上它之后 MILP 求解更快**（绿色 66s vs
    允许复用的 227s），因为它大幅收紧了 LP 松弛。

跨歌实测（vs 现有流水线，clip 90s / 6 seed / 统一窗口集）：

  | 臂 | 绿色 | 冬眠 | 爱笑的眼睛 | 均值 | 负 |
  |---|---|---|---|---|---|
  | shift_first | -0.0195 | -0.0013 | -0.0241 | -0.0150 | 3 |
  | satisfice 80% | -0.0022 | +0.0018 | +0.0070 | +0.0022 | 1 |
  | satisfice 90% | -0.0010 | +0.0023 | +0.0103 | +0.0039 | 1 |
  | **cov_anchor** | **+0.0038** | **-0.0000** | **+0.0112** | **+0.0050** | **0** |
  | cov_only | -0.0009 | -0.0033 | +0.0140 | +0.0033 | 2 |

**不要按歌曲类型切换策略。** 曾按「8st+ 桶占比」分档（<10% 用 satisfice、10-25% 用
cov_anchor、>25% 用 cov_only），同一批数据上看着是 +0.0067 > a55 的 +0.0050，但那是
用同一批数据挑出来的最优，必然过拟合；leave-one-out 检验（阈值只由另两首定）掉到
**+0.0034，反而低于单一 cov_anchor**。n=3 不足以支撑任何分档。

## 覆盖率不是机制，只是这批臂里的可优化量

15 个臂上 `corr(Δ覆盖, gain) = +0.771` 看着很强，但**去掉退化的 shift_first 臂后变成
-0.001**——那个相关完全由一个极端点拉出来。且 Δ覆盖 与 Δw|shift| 本身相关 +0.882，
这批臂无法把两者分开。所以"提高覆盖率"仍不是可信的因果解释，只是 cov_anchor 恰好
是这批可行解里表现最好的那个配置。这与 AGENTS.md 已记录的两次否证一致（select 阶段
前移移调：覆盖 75.8%→84.9%，SIM +0.0002；预移调参考 +5st：覆盖 50.6%→75.7%，
SIM -0.0029）。

## 碎片化不是问题（已用 CER 验证）

MILP 选出的段中位长 0.57s、约 27 段 < 0.5s，远碎于原贪心。用 `singing/eval_cer.py`
（15s 固定切块，绕过 VAD）实测：绿色 -0.0060、冬眠 -0.0380（不升反降），爱笑的眼睛
+0.0256。听感确认吐字无问题。**注意 VAD 路径下爱笑的眼睛会给出 CER 0.72 的假象**——
逐 3s 窗 RMS 显示三臂波形能量几乎一致（无坍缩），手动 15s 切块后字数从 58 回到 109，
即那是 FunASR VAD 提前截断，不是内容退化。
"""

from __future__ import annotations

import numpy as np

F0_RATE = 50.0
GRID_SEMI = 2.0


def build_grids(tv: np.ndarray) -> np.ndarray:
    """与 w_select / w_ext 一致的对数频段网格（2 半音一格）。"""
    lo = float(np.percentile(tv, 1))
    hi = float(np.percentile(tv, 99)) * 2 ** (GRID_SEMI / 12)
    g, x = [], lo
    while x < hi:
        g.append(x)
        x *= 2 ** (GRID_SEMI / 12)
    return np.array(g)


def hist_sec(f0v: np.ndarray, grids: np.ndarray) -> np.ndarray:
    """浊音 F0 序列 -> 每频段秒数。"""
    out = np.zeros(len(grids))
    if len(f0v) == 0:
        return out
    gi = np.floor(12 * np.log2(f0v / grids[0]) / GRID_SEMI).astype(int)
    ok = (gi >= 0) & (gi < len(grids))
    np.add.at(out, gi[ok], 1.0 / F0_RATE)
    return out


def solve(cands: list[dict], target_f0: np.ndarray, max_sec: float,
          anchor_ratio: float = 0.55, max_shift: int = 8, min_shift: int = -4,
          method: str = "cov_anchor", cov_target: float = 0.90,
          no_repeat: bool = True, time_limit: float = 240.0,
          log=print) -> list[tuple[int, int, float, float]]:
    """返回 [(cand_idx, shift, dur, med)]；shift=0 表示原声（不过 vocoder）。

    method:
      cov_anchor   覆盖最大化 + 干净锚点 >= anchor_ratio，再最小化 |shift|（默认）
      cov_only     纯覆盖最大化，再最小化 |shift|（无锚点下限）
      satisfice    覆盖只需达到 cov_target，余量全部用于最小化 |shift|
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    tv = target_f0[target_f0 > 0]
    if len(tv) == 0:
        raise ValueError("target F0 无浊音帧")
    grids = build_grids(tv)
    p_song = hist_sec(tv, grids)
    p_song /= p_song.sum()

    shifts = np.arange(min_shift, max_shift + 1)
    segs = []
    for c in cands:
        f0 = np.load(c["f0"])
        f0 = f0[f0 > 0]
        if len(f0) == 0:
            continue
        # 移调 s 后 F0 整体 x2^(s/12)，直方图沿对数轴平移
        H = np.stack([hist_sec(f0 * 2 ** (s / 12), grids) for s in shifts])
        segs.append((c, H))
    if not segs:
        raise ValueError("候选池为空")

    N, S, G = len(segs), len(shifts), len(grids)
    dur = np.array([c["dur"] for c, _ in segs])
    H = np.stack([h for _, h in segs])          # (N,S,G) 秒
    NX = N * S

    def xi(i, s):
        return i * S + s

    # 变量: x[i,s] (0-1) + d[g] (每格覆盖缺口松弛, 连续)
    integ = np.concatenate([np.ones(NX), np.zeros(G)])
    bnd = Bounds(np.zeros(NX + G), np.concatenate([np.ones(NX), np.full(G, np.inf)]))

    A_cov = np.zeros((G, NX + G))
    for i in range(N):
        for s in range(S):
            A_cov[:, xi(i, s)] = -H[i, s]
    for g in range(G):
        A_cov[g, NX + g] = -1.0

    A_bud = np.zeros((1, NX + G))
    for i in range(N):
        for s in range(S):
            A_bud[0, xi(i, s)] = dur[i]

    cons_fixed = [LinearConstraint(A_bud, -np.inf, max_sec)]
    if no_repeat:
        A_rep = np.zeros((N, NX + G))
        for i in range(N):
            for s in range(S):
                A_rep[i, xi(i, s)] = 1.0
        cons_fixed.append(LinearConstraint(A_rep, -np.inf, 1.0))
    if method == "cov_anchor" and anchor_ratio > 0:
        # 0st 段时长 >= anchor_ratio * 总时长（干净音色锚点，SoulX 的 speaker
        # embedding 需要未过 vocoder 的素材；纯覆盖优先会把它压到 27-36%）
        z = int(np.where(shifts == 0)[0][0])
        A_a = np.zeros((1, NX + G))
        for i in range(N):
            A_a[0, xi(i, z)] = dur[i] * (1 - anchor_ratio)
            for s in range(S):
                if s != z:
                    A_a[0, xi(i, s)] = -dur[i] * anchor_ratio
        cons_fixed.append(LinearConstraint(A_a, 0, np.inf))

    c_def = np.concatenate([np.zeros(NX), np.ones(G)])       # 目标1: 总覆盖缺口
    c_sh = np.zeros(NX + G)                                   # 目标2: 时长加权 |shift|
    for i in range(N):
        for s in range(S):
            c_sh[xi(i, s)] = dur[i] * abs(shifts[s])

    # 需求秒数 demand = p_song * V，其中 V = prompt 最终浊音秒数（依赖解本身）→ 迭代
    V = max_sec * 0.8
    r1 = None
    for _ in range(6):
        demand = p_song * V
        cons = [LinearConstraint(A_cov, -np.inf, -demand)] + cons_fixed
        r1 = milp(c_def, constraints=cons, integrality=integ, bounds=bnd,
                  options={"time_limit": min(90.0, time_limit), "mip_rel_gap": 2e-3})
        if r1.x is None:
            raise RuntimeError(f"MILP 目标1 无解: {r1.message}")
        Vn = float((r1.x[:NX].reshape(N, S)[:, :, None] * H).sum())
        if abs(Vn - V) < 0.05:
            V = Vn
            break
        V = 0.5 * V + 0.5 * Vn

    demand = p_song * V
    cons = [LinearConstraint(A_cov, -np.inf, -demand)] + cons_fixed
    if method == "satisfice":
        cap = max(float(r1.fun) + 1e-6, float(demand.sum()) * (1.0 - cov_target))
    else:
        cap = float(r1.fun) + 1e-6
    A_fix = np.zeros((1, NX + G))
    A_fix[0, NX:] = 1.0
    r2 = milp(c_sh, constraints=cons + [LinearConstraint(A_fix, -np.inf, cap)],
              integrality=integ, bounds=bnd,
              options={"time_limit": time_limit, "mip_rel_gap": 1e-2})
    if r2.x is None:
        # 目标2 超时：退回目标1 的解（覆盖已最优，只是 |shift| 未再压缩）
        log(f"  [warn] MILP 目标2 超时，用目标1 解（{r2.message[:40]}）")
        xv = r1.x
    else:
        xv = r2.x
    x = np.round(xv[:NX]).reshape(N, S).astype(int)

    sel = [(int(segs[i][0]["idx"]), int(shifts[s]), float(dur[i]), float(segs[i][0]["med"]))
           for i in range(N) for s in range(S) if x[i, s]]
    sel.sort(key=lambda t: (t[1], t[0]))
    ach = (x[:, :, None] * H).sum((0, 1))
    tot = sum(d for _, _, d, _ in sel)
    z = sum(d for _, sh, d, _ in sel if sh == 0)
    log(f"  MILP({method}) 覆盖 {np.minimum(ach, demand).sum() / demand.sum():.1%}"
        f"  段 {len(sel)}  唯一源 {len({i for i, _, _, _ in sel})}"
        f"  w|shift| {sum(d * abs(sh) for _, sh, d, _ in sel) / max(tot, 1e-9):.2f}"
        f"  0st {z:.1f}s({z / max(tot, 1e-9):.0%})  总长 {tot:.2f}s")
    return sel
