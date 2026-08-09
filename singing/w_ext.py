"""音域扩展 prompt：原始 prompt 作音色锚点 + 按频段缺口追加移调段（SoulX-Singer venv 运行）。

## 为什么是"追加"而不是"替换"

旧版把选中的每一段都移调，拼出的 prompt_ext 里 100% 是 vocoder 重合成音频，
原始干净音色一段不留。实测代价：NSF 在 0 半音（不变调）时 SIM 仍只有 0.8448，
即"过一遍 vocoder"本身就损失约 15% 音色相似度。且旧版输出长度 == 原 prompt 长度，
29s 预算白白浪费近 10s。

新策略:
  1. **原始 prompt 全量保留**在最前（干净音色锚点，SoulX 的 speaker embedding
     从整条 prompt 提取，需要未失真的参考）。超预算才按 --anchor-min-ratio 裁剪。
  2. 剩余预算投给**缺口最大的频段**: gap[g] = 歌曲时长占比 - 原prompt时长占比，
     按 gap 降序逐格填充（时长加权，不是旧版的"每段一个等概率目标点"）。
  3. 每格从全部候选里挑**移调量最小就能到达该格**的段（旧版按段中位升序与目标
     升序一一配对，低音段永远拿低目标，白白浪费了它们的移调余量）。
  4. 同一源段允许生成多个不同移调的副本去填不同频段（素材复用比硬拉 +7 更安全）。
  5. 同移调量的段拼成一条，一次过 PC-NSF-HiFiGAN（--shifter nsf，默认；
     seedvc 需 cwd=seed-vc/ 否则重下 2.4G 模型），按输入比例切回各段。
  6. 每段重新提 RMVPE F0（不能乘常数——vocoder 输出实际 F0 有偏差）。

输入: prompt_stats.json（原 prompt + 选中段）+ cands_meta.json（全部候选）。
输出: prompt_ext.wav / prompt_ext_f0.npy（契约同 select），总长 <= --max-sec。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "SoulX-Singer"
SEEDVC = ROOT / "seed-vc"
SEEDVC_VENV = SEEDVC / ".venv" / "bin" / "python"
sys.path.insert(0, str(SOULX))

PCNSF = ROOT / "tools" / "pcnsf"
PCNSF_VENV = PCNSF / ".venv" / "bin" / "python"
NSF_CONFIG = ROOT / "tools" / "pcnsf_hifigan" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" / "config.json"
NSF_CKPT = ROOT / "tools" / "pcnsf_hifigan" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" / "model.ckpt"

RMVPE = SOULX / "pretrained_models" / "SoulX-Singer-Preprocess" / "rmvpe" / "rmvpe.pt"
MAX_SHIFT_DEFAULT = 8  # 2026-08 A/B 3首全正（见 AGENTS.md）
DIV_WEIGHT_DEFAULT = 0.0  # 源段复用惩罚（等价半音数）；0 = 旧行为。
EXCLUDE_ANCHOR_SEGS_DEFAULT = False  # ext 是否排除锚点已含段；跨 3 首歌均值仅 +0.0034
# （只有重叠 62% 的绿色显著），不够格当默认，与 sing.py 的 Cfg.ext_exclude_anchor 一致。
# 注意：这里保持 0.0 只为「不带 --div-weight 直接跑本脚本时复现旧行为」；
# 实际默认由 sing.py 的 Cfg.ext_div_weight = 3.0 下发（A/B 3 首全正，见 AGENTS.md）。
FADE_SEC = 0.01
F0_RATE = 50.0
GRID_SEMI = 2.0        # 频段格宽（半音），与 w_select 一致


def nsf_shift(wav: str, out_dir: Path, shift: int, device: str) -> str:
    """调 PC-NSF-HiFiGAN 升调 wav（保共振峰，44.1k）；返回输出 wav 路径。"""
    if not PCNSF_VENV.exists():
        sys.exit(f"缺少 PC-NSF-HiFiGAN venv: {PCNSF_VENV}（在 tools/pcnsf/ 建 .venv）")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PCNSF_VENV), str(PCNSF / "scripts" / "pitch_transpose.py"), wav,
           "--transpose", str(shift), "--method", "nsfhifigan",
           "--config", str(NSF_CONFIG), "--ckpt", str(NSF_CKPT),
           "--pitchalgo", "rmvpe", "--rmvpeckpt", str(RMVPE),
           "--out", str(out_dir)]
    print(f"  pc-nsf-hifigan {shift:+d}: {Path(wav).name}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"pc-nsf-hifigan 失败 (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    out = sorted(out_dir.glob("*-out.wav"), key=lambda p: p.stat().st_mtime)
    if not out:
        sys.exit(f"pc-nsf-hifigan 未产生输出于 {out_dir}")
    return str(out[-1])


def nsf_shift_many(jobs: list[dict], work_dir: Path) -> list[dict]:
    """一次进程跑完所有 (wav, shift) 移调任务；返回原 jobs（含 out 路径）。

    替代逐组 fork nsf_shift()：模型加载开销 2.77s 只付一次而不是 N 次。
    输出与逐组路径 bit-identical（multi_transpose.py 逐行照抄 pitch_transpose.py
    的 nsfhifigan 分支），已用 md5 验证。
    """
    if not PCNSF_VENV.exists():
        sys.exit(f"缺少 PC-NSF-HiFiGAN venv: {PCNSF_VENV}（在 tools/pcnsf/ 建 .venv）")
    jobs_json = work_dir / "nsf_jobs.json"
    jobs_json.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    cmd = [str(PCNSF_VENV), str(PCNSF / "scripts" / "multi_transpose.py"),
           "--jobs", str(jobs_json), "--config", str(NSF_CONFIG),
           "--ckpt", str(NSF_CKPT), "--rmvpe", str(RMVPE)]
    print(f"  pc-nsf-hifigan 批量 {len(jobs)} 组: "
          + ", ".join(f"{j['shift']:+d}st" for j in jobs))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"pc-nsf-hifigan 失败 (exit {proc.returncode}):\n"
                 f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    for j in jobs:
        if not Path(j["out"]).exists():
            sys.exit(f"pc-nsf-hifigan 未产生输出: {j['out']}")
    return jobs


def seedvc_shift(wav: str, out_dir: Path, shift: int, device: str) -> str:
    """调 seed-vc CLI 升调 wav；返回输出 wav 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    cmd = [str(SEEDVC_VENV), str(SEEDVC / "inference.py"),
           "--source", wav, "--target", wav,
           "--output", str(out_dir),
           "--f0-condition", "True", "--auto-f0-adjust", "False",
           "--semi-tone-shift", str(shift), "--fp16", "True"]
    print(f"  seed-vc {shift:+d}: {Path(wav).name}")
    proc = subprocess.run(cmd, cwd=str(SEEDVC), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"seed-vc 失败 (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    out = sorted(out_dir.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
    if not out:
        sys.exit(f"seed-vc 未产生输出于 {out_dir}")
    return str(out[-1])


def build_grids(song_v: np.ndarray) -> list[float]:
    """按歌曲浊音 F0 范围建 GRID_SEMI 半音格，返回每格下边界 Hz。"""
    lo, hi = float(np.percentile(song_v, 1)), float(np.percentile(song_v, 99))
    lo = max(60.0, lo * 2 ** (-3 / 12))
    hi = min(1200.0, hi * 2 ** (3 / 12))
    edges, f = [], lo
    while f < hi:
        edges.append(f)
        f *= 2 ** (GRID_SEMI / 12)
    return edges


def sec_by_grid(v: np.ndarray, grids: list[float]) -> np.ndarray:
    """浊音 F0 序列在各频段格的秒数。"""
    out = np.zeros(len(grids))
    if len(v) == 0:
        return out
    g0 = grids[0]
    idx = np.floor(12 * np.log2(np.maximum(v, 1e-6) / g0) / GRID_SEMI).astype(int)
    for gi in idx:
        if 0 <= gi < len(grids):
            out[gi] += 1.0 / F0_RATE
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="prompt_stats.json")
    ap.add_argument("--cands", required=True, help="cands_meta.json")
    ap.add_argument("--target-f0", required=True)
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-f0", required=True)
    ap.add_argument("--seedvc-dir", required=True, help="移调临时输出目录")
    ap.add_argument("--shifter", default="nsf", choices=["seedvc", "nsf"],
                    help="移调引擎: nsf（默认，PC-NSF-HiFiGAN）| seedvc")
    ap.add_argument("--max-sec", type=float, default=29.0,
                    help="prompt_ext 总长上限（whisper 30s 硬顶）")
    ap.add_argument("--anchor-min-ratio", type=float, default=0.55,
                    help="原始干净音频占 prompt_ext 的份额上限（cap，不是下限）: "
                         "锚点超过 max_sec×ratio 即裁剪，余量全给移调段")
    ap.add_argument("--max-shift", type=int, default=MAX_SHIFT_DEFAULT,
                    help=f"单段最大移调半音数（默认 {MAX_SHIFT_DEFAULT}）。逐段保真度扫点显示 "
                         "-4..+6 均值都过 0.85、worst>=0.85 只到 +4，但 ext 是池化平均使用，"
                         "所以单段保真度不是判据：2026-08 A/B 三首歌 ms=8 全胜 ms=5"
                         "（+0.003-0.007, p<0.01），收益来自解锁低音段参与高音格竞争"
                         "（唯一源段 17->21，复用 1.41x->1.10x），不是移调本身变好")
    ap.add_argument("--div-weight", type=float, default=DIV_WEIGHT_DEFAULT,
                    help=f"源段复用惩罚，折算为等价半音数（默认 {DIV_WEIGHT_DEFAULT}）。"
                         "0=旧行为（纯按移调量最小选，会把素材锁在 med 最高的少数段上）；"
                         "越大越倾向用更大移调换更多不同源段（音素多样性）")
    ap.add_argument("--exclude-anchor-segs", action="store_true",
                    default=EXCLUDE_ANCHOR_SEGS_DEFAULT,
                    help="ext 不复用锚点(prompt.wav)里已有的源段（默认关，见下）。这类段会让同一素材"
                         "在 prompt_ext 里出现两次（原声 + 移调版），whisper 看到的内容多样性下降。"
                         "默认关的原因：跨 3 首歌只有重叠比例最高的绿色(62%%)显著受益 "
                         "(22/36 个 ext 段属于此类，+0.0090, 3/3 seed, p=0.0002)，冬眠(39%%) "
                         "-0.0004 与爱笑的眼睛(33%%) +0.0017 都在 ±0.004 噪声底内，均值仅 "
                         "+0.0034。重叠 >~60%% 的歌可手动开。代价是 458Hz+ 高音格从 4.24s 掉到 "
                         "0.93s、8+st 桶 -0.0042，但该桶只占 16%% 窗口且补回高音并不能修复它 "
                         "(已验证软惩罚分层版：高音填充恢复到 4.34-4.62s，8+ 桶仍是 -0.004，"
                         "而 0-6st 收益从 +0.011~+0.014 塌到 +0.001~+0.003)。见 AGENTS.md")
    ap.add_argument("--no-exclude-anchor-segs", dest="exclude_anchor_segs",
                    action="store_false", help="恢复旧行为（允许 ext 复用锚点段）")
    ap.add_argument("--select-method", default="greedy",
                    choices=["cov_anchor", "cov_only", "satisfice", "greedy"],
                    help="选段求解器（默认 greedy = 两阶段贪心：锚点+填缺口）。其余三个走 "
                         "MILP，把锚点与 ext 合并成一个 (段,移调) 指派问题、'一段素材只用一次' "
                         "为硬约束；cov_anchor = 覆盖最大化 + 干净锚点占比下限，cov_only 无该"
                         "下限，satisfice 只要求覆盖达 --cov-target。MILP 曾做过默认后撤回："
                         "3 首调参歌 clip 上 +0.0050(0负)，但全曲 10 首均值仅 +0.0012 —— "
                         "5 首显著正、5 首显著负（效应按歌分向，机制未知），且耗时 19.7×。"
                         "详见 singing/ext_milp.py")
    ap.add_argument("--cov-target", type=float, default=0.90,
                    help="--select-method satisfice 时的覆盖率目标（默认 0.90）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    max_shift = args.max_shift

    import librosa
    import soundfile as sf
    from preprocess.tools.f0_extraction import F0Extractor

    stats = json.load(open(args.stats, encoding="utf-8"))
    cands = {c["idx"]: c for c in json.load(open(args.cands, encoding="utf-8"))}
    sel = stats["selected"]
    if not sel:
        sys.exit("selected 为空（先跑 select 阶段）")
    prompt_wav = stats["prompt_wav"]
    prompt_f0_path = stats["prompt_f0"]
    if not Path(prompt_wav).exists():
        sys.exit(f"原始 prompt 不存在: {prompt_wav}（先跑 select 阶段）")

    tv = np.load(args.target_f0)
    tv = tv[tv > 0]
    if len(tv) == 0:
        sys.exit("target F0 无浊音帧")

    anchor, sr_out = librosa.load(prompt_wav, sr=None, mono=True)
    anchor_f0 = np.load(prompt_f0_path)
    anchor_dur = len(anchor) / sr_out

    # ---- MILP 路径：锚点与 ext 合并求解，直接产出完整 plan ----
    # 与下面的贪心路径互斥。贪心把「选锚点」和「填缺口」分两阶段各自贪心，同一源段
    # 可被两阶段各用一次；MILP 把两者合成一个 (段, 移调) 指派问题，"不重复"是硬约束。
    # 详见 singing/ext_milp.py 的模块 docstring（含跨歌 A/B 与被否证的替代方案）。
    if args.select_method != "greedy":
        import ext_milp
        milp_sel = ext_milp.solve(
            list(cands.values()), np.load(args.target_f0), max_sec=args.max_sec,
            anchor_ratio=args.anchor_min_ratio, max_shift=max_shift,
            method=args.select_method, cov_target=args.cov_target)
        # plan 里 shift=0 的段就是干净锚点素材，与移调段同走一条渲染路径
        # （0st 组在下面会跳过 vocoder），所以这里不再单独拼 anchor。
        plan = [(cands[i], sh, 0) for i, sh, _, _ in milp_sel]
        anchor = np.zeros(0, dtype=np.float32)
        anchor_f0 = np.zeros(0, dtype=np.float64)
        anchor_dur = 0.0
        used_sec = sum(d for _, _, d, _ in milp_sel)
        return _render(args, plan, anchor, anchor_f0, anchor_dur, used_sec,
                       sr_out, F0Extractor, librosa, sf)

    # ---- 1. 原始 prompt 作锚点，必要时裁剪到 anchor_min_ratio ----
    anchor_cap = args.max_sec * args.anchor_min_ratio
    if anchor_dur > anchor_cap:
        keep = int(anchor_cap * sr_out)
        anchor = anchor[:keep]
        anchor_f0 = anchor_f0[: int(anchor_cap * F0_RATE)]
        anchor_dur = len(anchor) / sr_out
        print(f"  锚点裁剪到 {anchor_dur:.2f}s（max_sec {args.max_sec:.0f}s × "
              f"anchor_min_ratio {args.anchor_min_ratio:.2f}）")
    budget = args.max_sec - anchor_dur
    print(f"  锚点(原始干净) {anchor_dur:.2f}s   移调段预算 {budget:.2f}s")

    # ---- 2. 频段缺口: 歌曲占比 - 锚点占比 ----
    grids = build_grids(tv)
    song_sec = sec_by_grid(tv, grids)
    song_pct = song_sec / max(song_sec.sum(), 1e-9)
    av = anchor_f0[anchor_f0 > 0]
    anchor_sec = sec_by_grid(av, grids)
    anchor_pct = anchor_sec / max(anchor_sec.sum(), 1e-9)
    gap_pct = song_pct - anchor_pct
    # 缺口秒数 = 缺口占比 × 预算，按占比归一化只在正缺口上分配
    pos = np.maximum(gap_pct, 0.0)
    if pos.sum() <= 1e-9:
        print("  锚点已覆盖歌曲全部频段分布，无需扩展")
        need_sec = np.zeros(len(grids))
    else:
        need_sec = pos / pos.sum() * budget

    print(f"\n  {'频段Hz':>13} {'歌曲%':>7} {'锚点%':>7} {'缺口pt':>7} {'待补s':>7}")
    for gi, g in enumerate(grids):
        hi = g * 2 ** (GRID_SEMI / 12)
        if song_pct[gi] < 0.005 and need_sec[gi] < 0.01:
            continue
        print(f"  {g:5.0f}-{hi:5.0f} {song_pct[gi] * 100:6.1f}% {anchor_pct[gi] * 100:6.1f}%"
              f" {gap_pct[gi] * 100:+6.1f} {need_sec[gi]:6.2f}")

    # ---- 3. 为每个缺口格挑"移调量最小就能到达"的候选段 ----
    # 候选池 = 全部通过质量过滤的段（不限于 select 选中的 20 段）
    pool = list(cands.values())

    # 锚点里实际保留下来的段（select 按时间顺序拼接，锚点裁剪只截前 anchor_dur 秒）。
    # 只有这些段才会在 prompt_ext 里"原声 + 移调版"各出现一次，构成真正的重复。
    anchor_idx: set[int] = set()
    _t = 0.0
    for _s in sel:
        if _t >= anchor_dur:
            break
        anchor_idx.add(int(_s["idx"]))
        _t += float(_s["dur"])
    if args.exclude_anchor_segs:
        before = len(pool)
        pool = [c for c in pool if c["idx"] not in anchor_idx]
        print(f"  排除锚点已含的 {before - len(pool)} 个源段（避免原声+移调版重复），"
              f"候选池 {before} -> {len(pool)}")
    plan: list[tuple[dict, int, int]] = []   # (cand, shift, grid_idx)
    used_copies: dict[tuple[int, int], bool] = {}
    used_src: dict[int, int] = {}      # 源段 idx -> 已用次数（多样性惩罚用）
    used_sec = 0.0
    order = np.argsort(-need_sec)             # 缺口大的格先填

    for gi in order:
        if need_sec[gi] < 0.01:
            continue
        g_lo = grids[gi]
        g_hi = g_lo * 2 ** (GRID_SEMI / 12)
        g_center = float(np.sqrt(g_lo * g_hi))
        filled = 0.0
        # 该格的候选按「移调量 + 复用惩罚」升序。
        # 移调越小音色损失越小（AGENTS.md 逐段扫点: 0.014-0.020 SIM/半音，且每个
        # 输出音高桶内 corr(|shift|,SIM) 都在 -0.76..-0.99）——但只按 |shift| 排会把
        # 素材池锁死在 med 最高的极少数段上: 实测绿色池有 143 段/95.6s，med>=290Hz
        # 的只有 15 段/7.7s，要填 13s 预算必然复用 2.2x，28 个移调段只来自 13 个源段。
        # div_weight 把「复用一次同源段」折算成等价半音数，让算法可以用更大的移调换
        # 更多不同的源段（音素多样性）。0 = 旧行为。
        # 落不进本格的段允许"尽力靠近"（钳到 max_shift 后取最接近），否则素材
        # med 上限不足时高音格会完全空掉——实测 440-494Hz 格因此只补到 0.3s/2.1s。
        avail = []
        for c in pool:
            raw = 12 * np.log2(g_center / c["med"])
            sh = max(-max_shift, min(max_shift, int(round(raw))))
            if sh == 0:
                continue                      # 本格已由锚点覆盖，无需重合成
            reached = c["med"] * 2 ** (sh / 12)
            inside = g_lo <= reached < g_hi
            avail.append((0 if inside else 1, abs(sh),
                          abs(12 * np.log2(reached / g_center)), c, sh))
        # 贪心逐个取：复用惩罚随 used_src 变化，故每次重算 cost 而非一次性排序
        while filled < need_sec[gi] and used_sec < budget and avail:
            best_i, best_key = None, None
            for i, (ins, ash, dist, c, sh) in enumerate(avail):
                if (c["idx"], sh) in used_copies:
                    continue
                if used_sec + c["dur"] > budget:
                    continue
                # 排序键: 先要"落进本格"的，再按 移调量+复用惩罚+锚点重复惩罚、离格中心近。
                # 锚点惩罚只在同格内比较时起作用（ins 优先级更高），所以高音格里
                # 只有锚点段能 inside 时它仍会被选中 —— 这正是分层的关键。
                key = (ins, ash + args.div_weight * used_src.get(c["idx"], 0), dist)
                if best_key is None or key < best_key:
                    best_key, best_i = key, i
            if best_i is None:
                break
            _, _, _, c, sh = avail.pop(best_i)
            used_copies[(c["idx"], sh)] = True
            used_src[c["idx"]] = used_src.get(c["idx"], 0) + 1
            plan.append((c, sh, int(gi)))
            filled += c["dur"]
            used_sec += c["dur"]
        if filled > 0:
            print(f"  格 {g_lo:.0f}-{g_hi:.0f}Hz: 需 {need_sec[gi]:.2f}s 实补 {filled:.2f}s")

    return _render(args, plan, anchor, anchor_f0, anchor_dur, used_sec,
                   sr_out, F0Extractor, librosa, sf)


def _render(args, plan, anchor, anchor_f0, anchor_dur, used_sec,
            sr_out, F0Extractor, librosa, sf) -> None:
    """把 plan（[(cand, shift, grid_idx)]）渲染成 prompt_ext.wav / _f0.npy。

    贪心路径与 MILP 路径共用：区别只在 plan 怎么来的，以及 MILP 路径把 0st 素材
    也放进 plan（anchor 传空数组），因此 0st 组在下面同样跳过 vocoder。
    """
    if not plan:
        print("  无可用移调段，prompt_ext = 原始 prompt")
        sf.write(args.out_wav, anchor, sr_out)
        np.save(args.out_f0, anchor_f0)
        return

    print(f"\n  计划 {len(plan)} 个移调段 / {used_sec:.2f}s")
    for c, sh, gi in sorted(plan, key=lambda t: t[1]):
        print(f"    seg {c['idx']:3d} med={c['med']:5.0f}Hz {sh:+d}st "
              f"-> {c['med'] * 2 ** (sh / 12):5.0f}Hz  ({c['dur']:.2f}s)")

    # ---- 4. 同移调量分组，一次过 vocoder ----
    extractor = F0Extractor(str(RMVPE), device=args.device, is_half=False,
                            max_duration=60.0, verbose=False)
    shift_dir = Path(args.seedvc_dir)
    shift_dir.mkdir(parents=True, exist_ok=True)
    ext_parts: list[np.ndarray] = []
    ext_f0_parts: list[np.ndarray] = []
    fade_n = int(FADE_SEC * sr_out)

    by_shift: dict[int, list[dict]] = {}
    for c, sh, _ in plan:
        by_shift.setdefault(sh, []).append(c)

    # 每个 shift 组的输入音频先全部落盘，再**一次**调 vocoder 跑完所有组。
    # 动机（本机实测）：pitch_transpose.py 单次进程固定开销 2.77s（import torch 1.02
    # + import 项目模块 0.99 + load RMVPE 0.53 + load NSF 0.11），而净计算只有 0.87s
    # （5.94s 音频）。0.30s 的组和 5.94s 的组 wall 都是 ~3.6-4.7s，即耗时几乎与音频
    # 长度无关、全是开销。5 组 fork 5 次 = ~14s 纯浪费。multi_transpose.py 把模型
    # 加载摊到一次：5 组 18.4s -> 7.6s，输出逐组 md5 bit-identical。
    # 注意 seedvc 引擎没有对应的多 job 入口，仍走逐组 subprocess。
    group_in: dict[int, np.ndarray] = {}      # shift -> 拼接后的输入音频
    group_parts: dict[int, list[np.ndarray]] = {}   # shift -> 各段原始音频（用于切回）
    for sh in sorted(by_shift):
        parts = [librosa.load(c["wav"], sr=sr_out, mono=True)[0] for c in by_shift[sh]]
        group_parts[sh] = parts
        group_in[sh] = np.concatenate(parts)

    shifted_by: dict[int, np.ndarray] = {}
    if 0 in group_in:
        # 0 半音无需重合成，直接用原始音频（省一次 vocoder 损失）
        shifted_by[0] = group_in[0]
        print("  +0st: 直接使用原始音频（跳过 vocoder）")

    nonzero = [sh for sh in sorted(group_in) if sh != 0]
    if nonzero and args.shifter == "nsf":
        jobs = []
        for sh in nonzero:
            tmp_wav = shift_dir / f"group_{sh:+03d}.wav"
            sf.write(str(tmp_wav), group_in[sh], sr_out)
            jobs.append({"wav": str(tmp_wav), "shift": sh,
                         "out": str(shift_dir / f"out_{sh:+03d}.wav")})
        for job in nsf_shift_many(jobs, shift_dir):
            y, sr_s = librosa.load(job["out"], sr=None, mono=True)
            if sr_s != sr_out:
                y = librosa.resample(y, orig_sr=sr_s, target_sr=sr_out)
            shifted_by[int(job["shift"])] = y
    else:
        for sh in nonzero:
            tmp_wav = shift_dir / f"group_{sh:+03d}.wav"
            sf.write(str(tmp_wav), group_in[sh], sr_out)
            out_wav = seedvc_shift(str(tmp_wav), shift_dir / f"out_{sh:+03d}", sh, args.device)
            y, sr_s = librosa.load(out_wav, sr=None, mono=True)
            if sr_s != sr_out:
                y = librosa.resample(y, orig_sr=sr_s, target_sr=sr_out)
            shifted_by[sh] = y

    for sh in sorted(by_shift):
        segs = by_shift[sh]
        parts = group_parts[sh]
        shifted = shifted_by[sh]
        # 按输入比例切回各段
        off = 0
        for y, c in zip(parts, segs):
            seg = shifted[off: off + len(y)]
            off += len(y)
            if len(seg) < fade_n * 2:
                continue
            seg_wav = shift_dir / f"seg_{c['idx']:04d}_{sh:+03d}.wav"
            sf.write(str(seg_wav), seg, sr_out)
            n_frames = int(round(len(seg) / sr_out * F0_RATE))
            f0 = extractor.process(str(seg_wav), verbose=False)[:n_frames]
            if len(f0) < n_frames:
                f0 = np.pad(f0, (0, n_frames - len(f0)))
            seg = seg.copy()
            seg[:fade_n] *= np.linspace(0, 1, fade_n)
            seg[-fade_n:] *= np.linspace(1, 0, fade_n)
            ext_parts.append(seg)
            ext_f0_parts.append(f0)

    # ---- 5. 锚点 + 移调段拼接 ----
    n_anchor_f0 = int(len(anchor) / sr_out * F0_RATE)
    anchor_f0 = anchor_f0[:n_anchor_f0]
    if len(anchor_f0) < n_anchor_f0:
        anchor_f0 = np.pad(anchor_f0, (0, n_anchor_f0 - len(anchor_f0)))
    audio = np.concatenate([anchor] + ext_parts)
    f0 = np.concatenate([anchor_f0] + ext_f0_parts)
    cap = int(args.max_sec * sr_out)
    if len(audio) > cap:
        audio = audio[:cap]
        f0 = f0[: int(args.max_sec * F0_RATE)]
    n_f0 = int(len(audio) / sr_out * F0_RATE)
    f0 = f0[:n_f0] if len(f0) >= n_f0 else np.pad(f0, (0, n_f0 - len(f0)))

    sf.write(args.out_wav, audio, sr_out)
    np.save(args.out_f0, f0)
    v = f0[f0 > 0]
    ext_dur = sum(len(s) for s in ext_parts) / sr_out
    total = len(audio) / sr_out
    # MILP 路径下 0st 素材也走 plan（anchor 为空），所以"干净时长"要把 plan 里
    # shift=0 的段算进来，否则会误报成 "锚点 0.00s (0%)"。
    clean = anchor_dur + sum(c["dur"] for c, sh, _ in plan if sh == 0)
    print(f"\n-> {args.out_wav} ({total:.2f}s, {sr_out}Hz)")
    print(f"   干净(未过 vocoder) {clean:.2f}s ({clean / total * 100:.0f}%)"
          f" + 移调段 {ext_dur - (clean - anchor_dur):.2f}s"
          f"  [共 {len(ext_parts)} 段渲染 + 锚点 {anchor_dur:.2f}s]")
    print(f"-> {args.out_f0} ({len(f0)} frames, med {np.median(v):.1f}Hz, "
          f"p90 {np.percentile(v, 90):.1f}Hz)")


if __name__ == "__main__":
    main()
