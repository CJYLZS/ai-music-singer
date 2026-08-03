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
MAX_SHIFT = 5          # +10 半音退化到 DSP 水平（AGENTS.md 实测）
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
                    help="原始干净音频至少占 prompt_ext 的比例（音色锚点）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

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
    plan: list[tuple[dict, int, int]] = []   # (cand, shift, grid_idx)
    used_copies: dict[tuple[int, int], bool] = {}
    used_sec = 0.0
    order = np.argsort(-need_sec)             # 缺口大的格先填

    for gi in order:
        if need_sec[gi] < 0.01:
            continue
        g_lo = grids[gi]
        g_hi = g_lo * 2 ** (GRID_SEMI / 12)
        g_center = float(np.sqrt(g_lo * g_hi))
        filled = 0.0
        # 该格的候选按 |移调量| 升序：移调越小音色损失越小。
        # 落不进本格的段允许"尽力靠近"（钳到 MAX_SHIFT 后取最接近），否则素材
        # med 上限不足时高音格会完全空掉——实测 440-494Hz 格因此只补到 0.3s/2.1s。
        ranked = []
        for c in pool:
            raw = 12 * np.log2(g_center / c["med"])
            sh = max(-MAX_SHIFT, min(MAX_SHIFT, int(round(raw))))
            if sh == 0:
                continue                      # 本格已由锚点覆盖，无需重合成
            reached = c["med"] * 2 ** (sh / 12)
            inside = g_lo <= reached < g_hi
            # 排序键: 先要"落进本格"的，再按移调量小、离格中心近
            ranked.append((0 if inside else 1, abs(sh),
                           abs(12 * np.log2(reached / g_center)), c, sh))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))
        ranked = [(a, b, c, sh) for _, a, b, c, sh in ranked]
        for _, _, c, sh in ranked:
            if filled >= need_sec[gi] or used_sec >= budget:
                break
            key = (c["idx"], sh)
            if key in used_copies:
                continue           # 同段同移调不重复用
            dur = c["dur"]
            if used_sec + dur > budget:
                continue
            used_copies[key] = True
            plan.append((c, sh, int(gi)))
            filled += dur
            used_sec += dur
        if filled > 0:
            print(f"  格 {g_lo:.0f}-{g_hi:.0f}Hz: 需 {need_sec[gi]:.2f}s 实补 {filled:.2f}s")

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

    for sh in sorted(by_shift):
        segs = by_shift[sh]
        parts = []
        for c in segs:
            y, sr = librosa.load(c["wav"], sr=sr_out, mono=True)
            parts.append(y)
        combined = np.concatenate(parts)
        if sh == 0:
            # 0 半音无需重合成，直接用原始音频（省一次 vocoder 损失）
            shifted = combined
            print(f"  {sh:+d}st: 直接使用原始音频（跳过 vocoder）")
        else:
            tmp_wav = shift_dir / f"group_{sh:+03d}.wav"
            sf.write(str(tmp_wav), combined, sr_out)
            shifter = nsf_shift if args.shifter == "nsf" else seedvc_shift
            out_wav = shifter(str(tmp_wav), shift_dir / f"out_{sh:+03d}", sh, args.device)
            shifted, sr_s = librosa.load(out_wav, sr=None, mono=True)
            if sr_s != sr_out:
                shifted = librosa.resample(shifted, orig_sr=sr_s, target_sr=sr_out)
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
    print(f"\n-> {args.out_wav} ({len(audio) / sr_out:.2f}s, {sr_out}Hz)")
    print(f"   锚点(干净) {anchor_dur:.2f}s ({anchor_dur / (len(audio) / sr_out) * 100:.0f}%)"
          f" + 移调段 {ext_dur:.2f}s ({len(ext_parts)} 段)")
    print(f"-> {args.out_f0} ({len(f0)} frames, med {np.median(v):.1f}Hz, "
          f"p90 {np.percentile(v, 90):.1f}Hz)")


if __name__ == "__main__":
    main()
