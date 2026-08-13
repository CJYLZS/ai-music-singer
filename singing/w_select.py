"""参考音频分析 + 片段选择（SoulX-Singer venv 运行）。

流程:
  1. 混合两级切分（默认）: 粗切(186ms 静音)出乐句 + 长段内部细分(93ms) +
     RMS 最低点兜底 + 短段合并，候选严格落在 [min-sec, max-sec] = [1s, 5s]
  2. 对每段提 RMVPE F0，过滤低浊音比例/过低音高的片段
  3. 边际增益贪心: 把歌曲音域按 2 半音分格，每格需求 = 歌曲在该格的浊音秒数；
     每轮选"补满未覆盖格子"增益最大的片段，直到预算用满或全部覆盖
     （旧策略 --legacy: 2~8s 片段 + 单段 0.7*coverage+0.3*voiced 独立打分贪心）
  4. 按时间顺序拼接 -> prompt.wav + prompt_f0.npy（50Hz 网格，与音频严格对齐），
     每段 10ms 淡入淡出避免拼接 click
  5. 写 prompt_stats.json（统计 + 建议变调）与 cands_meta.json（全部候选的
     wav/f0 路径 + F0 统计，供 ext 阶段按音高分层独立移调使用）

约束: 最终 prompt 建议留给 seed-vc 扩展尾部 ~4s 空间，
SoulX whisper 编码器硬上限 30s（3000 mel 帧）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "SoulX-Singer"
sys.path.insert(0, str(SOULX))

RMVPE = SOULX / "pretrained_models" / "SoulX-Singer-Preprocess" / "rmvpe" / "rmvpe.pt"
F0_RATE = 50.0  # 50Hz 网格

MIN_VOICED_RATIO = 0.35
MIN_MED_F0 = 70.0
GRID_SEMI = 2.0       # 音高格宽（半音）
FADE_SEC = 0.01       # 拼接淡入淡出


def grid_index(f0: float, g0: float) -> int:
    return int(np.floor(12 * np.log2(f0 / g0) / GRID_SEMI))


def make_grids(song_v: np.ndarray) -> list[float]:
    """按歌曲浊音 F0 范围建 2 半音格，返回每个格的下边界 Hz。"""
    lo, hi = float(np.percentile(song_v, 1)), float(np.percentile(song_v, 99))
    lo = max(60.0, lo * 2 ** (-3 / 12))
    hi = min(1200.0, hi * 2 ** (3 / 12))
    edges, f = [], lo
    while f < hi:
        edges.append(f)
        f *= 2 ** (GRID_SEMI / 12)
    return edges


def legacy_cands(y, sr, min_sec, max_sec, top_db):
    import librosa
    bounds = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    cands = []
    for s, e in bounds:
        dur = (e - s) / sr
        if dur < min_sec:
            continue
        e = min(e, s + int(max_sec * sr))
        cands.append((int(s), int(e)))
    return cands


def fine_cands(y, sr, min_sec, max_cand, top_db, split_frame=8192, inner_frame=4096):
    """混合两级切分（2026-08 起默认）:
    1. 粗切: 长静音（split_frame 样本，默认 186ms）切出乐句边界
    2. 细分: 长段（> max_cand）内部按短静音（inner_frame，默认 93ms）再切
    3. 仍 > max_cand: 在 max_cand 附近的 RMS 最低点硬切一次
    4. 短段合并: < min_sec 的段向左邻吸收（不行则连续右吸收），保证
       [min_sec, max_cand] 封闭；合并超长时对合并体做一次 RMS 切分。
       终止性靠单调消费保证（每段只进合并流程一次，不递归）。

    依据: 旧切法 frame 2048/top_db 25 把 538s 参考切成 674 段(med 0.41s)，
    段间空隙中位仅 58ms —— 短语内部微停顿被误当边界，候选池 319/429 <1s。
    新切法实测 160 段、[1s,5s] 封闭、med 3.01s。见 AGENTS.md。
    """
    import librosa
    bounds = librosa.effects.split(y, top_db=top_db, frame_length=split_frame,
                                   hop_length=512)

    def hard_cut(seg, target, after=False):
        hop = 512
        rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=hop)[0]
        f = int(target * sr / hop)
        if after:
            # 前视窗: 切点 >= target（用于"填充到 min_sec"，保证合并体达标）
            f = max(1, min(f, len(rms) - 3))
            hi = min(f + 20, len(rms) - 1)
            cut = int(np.argmin(rms[f:hi + 1])) + f
        else:
            # 回溯窗: 切点 <= target（用于"封顶 max_cand"，保证不超上限）
            f = max(2, min(f, len(rms) - 2))
            lo = max(1, f - 20)
            cut = int(np.argmin(rms[lo:f + 1])) + lo
        return cut * hop

    pieces = []
    for s, e in bounds:
        seg = y[s:e]
        if (e - s) / sr <= max_cand:
            pieces.append((s, e))
            continue
        inner = librosa.effects.split(seg, top_db=top_db, frame_length=inner_frame,
                                      hop_length=512)
        for a, b in inner:
            rem = (s + a, s + b)
            while (rem[1] - rem[0]) / sr > max_cand:  # 循环切，剩余部分反复检查
                c = hard_cut(y[rem[0]:rem[1]], max_cand)
                pieces.append((rem[0], rem[0] + c))
                rem = (rem[0] + c, rem[1])
            pieces.append(rem)

    def pdur(p):
        return (p[1] - p[0]) / sr

    cands = []
    i = 0
    while i < len(pieces):
        if pdur(pieces[i]) >= min_sec:
            cands.append((int(pieces[i][0]), int(pieces[i][1])))
            i += 1
            continue
        S = pieces[i]
        if cands:
            L = cands[-1]
            if (S[1] - L[0]) / sr <= max_cand:
                cands[-1] = (L[0], int(S[1]))
                i += 1
                continue
        if i + 1 >= len(pieces):
            cands.append((int(S[0]), int(S[1])))
            i += 1
            continue
        merged = S
        j = i + 1
        while (j < len(pieces) and (pieces[j][1] - merged[0]) / sr <= max_cand
               and pdur(merged) < min_sec):
            merged = (merged[0], pieces[j][1])
            j += 1
        if pdur(merged) >= min_sec:
            cands.append((int(merged[0]), int(merged[1])))
            i = j
            continue
        if j < len(pieces):
            R = pieces[j]
            need = min_sec - pdur(merged)
            seg = y[R[0]:R[1]]
            c = hard_cut(seg, need, after=True)
            if (R[0] + c - merged[0]) / sr <= max_cand:
                cands.append((int(merged[0]), R[0] + c))
                pieces[j] = (R[0] + c, R[1])
                i = j
                continue
        cands.append((int(merged[0]), int(merged[1])))
        i = j

    return cands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考语音（会被切片）")
    ap.add_argument("--target-f0", required=True, help="歌曲人声 F0 .npy（50Hz）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", type=float, default=25.0, help="prompt 总时长上限(秒)")
    ap.add_argument("--min-sec", type=float, default=1.0,
                    help="候选最短(秒)。混合切法的合并目标下限，见 fine_cands")
    ap.add_argument("--max-sec", type=float, default=4.0,
                    help="候选最长(秒)。混合切法的硬上限（RMS 最低点兜底切分）")
    ap.add_argument("--top-db", type=float, default=30.0, help="静音判定阈值 dB")
    ap.add_argument("--split-frame", type=int, default=8192,
                    help="粗切静音帧长(样本@44.1k)。8192=186ms，切乐句边界")
    ap.add_argument("--inner-frame", type=int, default=4096,
                    help="长段内部细分静音帧长。4096=93ms，只断呼吸停顿")
    ap.add_argument("--gain-alpha", type=float, default=1.0,
                    help="贪心排序键的时长归一化指数: gain/dur**alpha。"
                         "1.0（默认）= 单位时长增益，与 w_select_svs.py 一致；"
                         "0 = 绝对增益，长段占优（4s 长音在单格里得 4.0，1s 跨 3 格只得 1.0）。"
                         "默认 1.0 是 2026-08 定的：与 --dbs-group 2 绑定（跨 4 首歌 +0.0165 SIM 的"
                         "实验全部在 alpha=1 上测得），单独改回 0 会脱离已验证配置")
    ap.add_argument("--select-seed", type=int, default=-1,
                    help="RCL 随机种子（--select-rcl>1 时生效；-1 且 rcl>1 时用 0）")
    ap.add_argument("--select-rcl", type=int, default=1,
                    help="RCL 宽度：每步从增益前 K 名里随机抽一个（GRASP 半贪心构造）。"
                         "1（默认）= 纯贪心，与历史产物 bit-identical。"
                         "K>1 用于探索选段空间（多起点取优），注意默认 K=1 时不受 select-seed 影响")
    ap.add_argument("--rcl-eps", type=float, default=0.0,
                    help="值域 RCL：每步从增益 >= 最优×(1-eps) 的候选里随机抽。"
                         "0（默认）= 关；0.1 = 增益在最优值 10% 内的候选都进窗口再随机抽。"
                         "同音高候选增益相近，会整组进入窗口 —— 模拟: eps=0.1 覆盖仅掉 ~2pp，"
                         "eps=0.3 掉 10pp+ 过界")
    ap.add_argument("--dbs-group", type=int, default=2,
                    help="Diverse Beam Search: 输出第几轮贪心的结果（0=纯贪心）。"
                         "第 g 轮的排序键减去 lam×该段被前 g 轮选过的次数，所以必须顺序跑 "
                         "0..g 轮（惩罚是累积的），g 越大越偏离贪心最优覆盖。"
                         "默认 2（2026-08）：跨 4 首歌 × 2-3 seed 实测 vs g0 平均 +0.0165 SIM"
                         "（爱笑的眼睛 +0.0367 / 雨爱 +0.0231 / 绿色 +0.0078 / 冬眠 -0.0016），"
                         "7/8 arm 为正、无显著负例，且零推理成本。0 = 回到历史贪心产物")
    ap.add_argument("--dbs-lam", type=float, default=0.1,
                    help="DBS 多样性惩罚权重（等效覆盖秒数/每次被前轮使用）。"
                         "0.1 = 一个段被前轮用过一次，其排序键降低相当于 0.1s 覆盖增益")
    ap.add_argument("--legacy", action="store_true", help="用旧策略（2~8s 整句 + 单段打分）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not RMVPE.exists():
        sys.exit(f"缺少 RMVPE 模型: {RMVPE}")
    if not Path(args.target_f0).exists():
        sys.exit(f"target F0 不存在: {args.target_f0}（先跑 f0-target 阶段）")

    import librosa
    import soundfile as sf
    from preprocess.tools.f0_extraction import F0Extractor

    out_dir = Path(args.out_dir)
    cand_dir = out_dir / "cands"
    cand_f0_dir = out_dir / "cands_f0"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_f0_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 切片 ----
    y, sr = librosa.load(args.ref, sr=None, mono=True)
    if args.legacy:
        cands = legacy_cands(y, sr, args.min_sec, 8.0, args.top_db)
        tag = "legacy"
    else:
        cands = fine_cands(y, sr, args.min_sec, args.max_sec, args.top_db,
                           args.split_frame, args.inner_frame)
        tag = "fine"
    if not cands:
        sys.exit("参考音频中没有 >= min-sec 的有效片段（试调低 --top-db 或 --min-sec）")
    print(f"候选片段: {len(cands)} 个 ({tag})")

    # ---- 2/3. 逐段 F0 + 质量过滤 ----
    extractor = F0Extractor(str(RMVPE), device=args.device, is_half=False,
                            max_duration=60.0, verbose=False)
    target_f0 = np.load(args.target_f0)
    tv = target_f0[target_f0 > 0]
    if len(tv) == 0:
        sys.exit("target F0 无浊音帧（先检查 f0-target 阶段输出）")

    grids = make_grids(tv)
    grid_sec = np.array([
        float(((tv >= g) & (tv < g * 2 ** (GRID_SEMI / 12))).sum()) / F0_RATE
        for g in grids
    ])  # 歌曲每格浊音秒数
    # 需求 = 歌曲分布 × 预算：素材丰富的格"够用即止"，贪心才会走向稀缺的高音格
    demand = grid_sec / max(float(grid_sec.sum()), 1e-9) * args.budget

    rows: list[dict] = []
    for i, (s, e) in enumerate(cands):
        clip = y[s:e]
        if float(np.sqrt(np.mean(clip**2))) < 0.005:
            continue
        wav = cand_dir / f"c_{i:04d}.wav"
        f0p = cand_f0_dir / f"c_{i:04d}.npy"
        sf.write(str(wav), clip, sr)
        f0 = extractor.process(str(wav), f0_path=str(f0p), verbose=False)
        f0 = f0[: int(len(clip) / 480)]  # 对齐 50Hz 网格
        v = f0[f0 > 0]
        if len(v) == 0:
            continue
        voiced_ratio = len(v) / len(f0)
        if voiced_ratio < MIN_VOICED_RATIO or float(np.median(v)) < MIN_MED_F0:
            continue
        p10, p90 = float(np.percentile(v, 10)), float(np.percentile(v, 90))
        sec_by_grid = np.zeros(len(grids))
        for f in v:
            g = grid_index(f, grids[0])
            if 0 <= g < len(grids):
                sec_by_grid[g] += 1.0 / F0_RATE
        coverage = float(np.mean((tv >= p10) & (tv <= p90)))
        rows.append(dict(i=i, s=s, e=e, wav=str(wav), f0=str(f0p),
                         voiced_ratio=voiced_ratio, med=float(np.median(v)),
                         p10=p10, p90=p90, coverage=coverage,
                         sec_by_grid=sec_by_grid))

    if not rows:
        sys.exit("没有片段通过浊音过滤（参考语音过短或过安静？）")

    # ---- 3. 选择 ----
    if args.legacy:
        selected = []
        used = 0.0
        for r in sorted(rows, key=lambda r: -(0.7 * r["coverage"] + 0.3 * r["voiced_ratio"])):
            dur = (r["e"] - r["s"]) / sr
            if used + dur > args.budget:
                continue
            selected.append(r)
            used += dur
        print(f"[legacy] 选中 {len(selected)} 段 / {used:.1f}s（预算 {args.budget:.0f}s）")
    else:
        total_demand = float(demand.sum())

        def run_group(gidx: int, penalty: dict) -> tuple[list, float, np.ndarray]:
            """跑一组贪心。penalty: 段 idx -> 前组已用次数，排序键减去 dbs_lam×次数。"""
            pool = [r for r in rows]
            selected = []
            used = 0.0
            covered = np.zeros(len(grids))
            rng = None
            if args.select_rcl > 1 or args.rcl_eps > 0:
                seed = (args.select_seed if args.select_seed >= 0 else 0) + gidx
                rng = np.random.default_rng(seed)
            while pool and used < args.budget:
                def gain(r: dict) -> float:
                    g = 0.0
                    s = r["sec_by_grid"]
                    for gi, sec in enumerate(s):
                        if sec <= 0:
                            continue
                        gap = demand[gi] - covered[gi]
                        if gap > 0:
                            g += min(gap, sec)
                    if args.gain_alpha > 0:
                        dur = (r["e"] - r["s"]) / sr
                        g /= max(dur, 1e-6) ** args.gain_alpha
                    return g - args.dbs_lam * penalty.get(r["i"], 0)

                pool.sort(key=gain, reverse=True)
                raw_best = gain(pool[0]) + args.dbs_lam * penalty.get(pool[0]["i"], 0)
                if raw_best < 0.01:
                    break
                bg = gain(pool[0])
                if args.rcl_eps > 0:
                    thresh = bg * (1.0 - args.rcl_eps)
                    n_rcl = sum(1 for r in pool if gain(r) >= thresh)
                    pi = int(rng.integers(n_rcl))
                elif args.select_rcl > 1:
                    k = min(args.select_rcl, len(pool))
                    pi = int(rng.integers(k))
                else:
                    pi = 0
                best = pool[pi]
                dur = (best["e"] - best["s"]) / sr
                if used + dur > args.budget:
                    pool.pop(pi)  # 放不下 -> 永久淘汰
                    continue
                pool.pop(pi)
                selected.append(best)
                used += dur
                covered += best["sec_by_grid"]
                if float((covered >= demand).all()):
                    print(f"所有音高格已满足需求，提前停止")
                    break
            return selected, used, covered

        if args.dbs_group > 0:
            print(f"[DBS] 顺序跑 0..{args.dbs_group} 轮  lam={args.dbs_lam}  取第 {args.dbs_group} 轮")
        used_count: dict = {}
        selected = []
        used = 0.0
        covered = np.zeros(len(grids))
        for g in range(args.dbs_group + 1):
            sel_g, used_g, covered_g = run_group(g, used_count)
            for r in sel_g:
                used_count[r["i"]] = used_count.get(r["i"], 0) + 1
            if g == args.dbs_group:
                selected, used, covered = sel_g, used_g, covered_g
                break
        covered_pct = float(np.minimum(covered, demand).sum()) / total_demand
        print(f"[fine] 选中 {len(selected)} 段 / {used:.1f}s（预算 {args.budget:.0f}s）"
              f"  覆盖 {covered_pct:.1%}")

    selected.sort(key=lambda r: r["s"])  # 按时间顺序拼接，保持语气连贯

    # ---- 4. 拼接 prompt + f0（每段 10ms 淡入淡出）----
    fade_n = int(FADE_SEC * sr)
    fade_in = np.linspace(0.0, 1.0, fade_n)
    fade_out = 1.0 - fade_in
    parts, f0_parts = [], []
    for r in selected:
        seg = y[r["s"]:r["e"]].copy()
        if fade_n > 0:
            seg[:fade_n] *= fade_in
            seg[-fade_n:] *= fade_out
        parts.append(seg)
        f0_parts.append(np.load(r["f0"])[: int((r["e"] - r["s"]) / 480)])
    audio = np.concatenate(parts) if parts else y[:0]
    prompt_f0 = np.concatenate(f0_parts)[: len(audio) // 480]
    prompt_wav = out_dir / "prompt.wav"
    prompt_f0_path = out_dir / "prompt_f0.npy"
    sf.write(str(prompt_wav), audio, sr)
    np.save(str(prompt_f0_path), prompt_f0)

    # ---- 5. 统计 + 建议变调 ----
    pv = prompt_f0[prompt_f0 > 0]
    song_med = float(np.median(tv))
    song_p90 = float(np.percentile(tv, 90))
    prompt_med = float(np.median(pv)) if len(pv) else 0.0
    prompt_p90 = float(np.percentile(pv, 90)) if len(pv) else 0.0
    if prompt_p90 > 0 and song_p90 > 0:
        shift = round(12 * np.log2(song_p90 / prompt_p90))
        shift = max(-6, min(6, shift))
        if abs(shift) < 1:
            shift = 0
    else:
        shift = 0

    cands_meta = [dict(idx=r["i"], wav=r["wav"], f0=r["f0"],
                       start_sec=r["s"] / sr, dur=(r["e"] - r["s"]) / sr,
                       med=r["med"], p10=r["p10"], p90=r["p90"],
                       voiced=r["voiced_ratio"])
                  for r in rows]
    stats = {
        "ref": args.ref,
        "prompt_wav": str(prompt_wav),
        "prompt_f0": str(prompt_f0_path),
        "prompt_dur": float(len(audio) / sr),
        "prompt_med": prompt_med,
        "prompt_p90": prompt_p90,
        "song_med": song_med,
        "song_p90": song_p90,
        "n_cands": len(rows),
        "n_selected": len(selected),
        "selected": [dict(idx=r["i"], start_sec=r["s"] / sr,
                          dur=(r["e"] - r["s"]) / sr,
                          med=r["med"], p10=r["p10"], p90=r["p90"],
                          voiced=r["voiced_ratio"])
                     for r in selected],
        "suggested_shift": shift,
        "strategy": tag,
        "grids_hz": grids,
        "coverage_pct": (float(np.minimum(covered, demand).sum()) / total_demand
                         if not args.legacy else None),
    }
    (out_dir / "prompt_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "cands_meta.json").write_text(json.dumps(cands_meta, indent=1), encoding="utf-8")
    print(f"-> {prompt_wav} ({len(audio) / sr:.2f}s, {sr}Hz)  [strategy={tag}]")
    print(f"-> {prompt_f0_path} ({len(prompt_f0)} frames)")
    print(f"   建议 seed-vc 变调: {shift:+d} 半音 "
          f"(song p90 {song_p90:.0f}Hz vs prompt p90 {prompt_p90:.0f}Hz)")


if __name__ == "__main__":
    main()
