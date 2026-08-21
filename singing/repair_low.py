#!/usr/bin/env python3
"""低音区定点修复：低音锚定 prompt（从参考重选真实低音素材）+ 局部重转 + 对齐拼接。

动机（AGENTS.md: 音高距离驱动相似度；prompt 是 CFM 的音色锚）：
低音区（参考中位下方 5-15st）用高音域 prompt 条件是 off-distribution 的。
本脚本为指定跨度构建 ≤30s 的低音锚定 prompt（真实素材优先，零 vocoder 损耗），
用它对跨度局部重转，逐秒 SIM vs 基线，选最佳臂并（达标时）在公共静音点拼接。

验证用法（SoulX-Singer venv）:
  SoulX-Singer/.venv/bin/python singing/repair_low.py \
      --song-dir data/songs/情歌 \
      --spans 25.5-30.87 139.5-144.35 \
      --low-band 170-300 --budget 30 \
      --arms 0.5:0.2:1 1.0:0.4:1 \
      --min-gain 0.015

产出:
  <song-dir>/low_prompt.wav + low_prompt_f0.npy + low_prompt_stats.json
  <song-dir>/vocal_repaired_low.wav（只替换达标跨度，其余与原 vocal 相同）
  逐秒 SIM 对比表（stdout）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_splice import PAD, SR, XF, run_svc, find_shift, find_common_silence, orig_silence_mid  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval_sim import SV, load16k, cos  # noqa: E402

F0_RATE = 50.0
GRID_SEMI = 2.0
FADE_SEC = 0.01
REF = os.path.join(REPO, "data", "voices", "ref_talk_sing.wav")


# ---------------------------------------------------------------- prompt 构建
def build_low_prompt(song_dir: str, low_lo: float, low_hi: float, budget: float,
                     out_wav: str, out_f0: str, out_stats: str) -> None:
    meta = json.load(open(os.path.join(song_dir, "cands_meta.json"), encoding="utf-8"))
    tgt = np.load(os.path.join(song_dir, "target_f0.npy"))

    edges = []
    e = low_lo
    while e < low_hi:
        edges.append(e)
        e *= 2 ** (GRID_SEMI / 12)
    edges.append(low_hi)
    n_grid = len(edges) - 1
    demand = np.zeros(n_grid, dtype=int)
    for k in range(n_grid):
        demand[k] = int(((tgt >= edges[k]) & (tgt < edges[k + 1])).sum())

    cands = []
    for c in meta:
        if c["voiced"] < 0.55:
            continue
        if not (low_lo <= c["med"] <= low_hi):
            continue
        f0 = np.load(c["f0"])
        g = np.zeros(n_grid, dtype=int)
        for k in range(n_grid):
            g[k] = int(((f0 >= edges[k]) & (f0 < edges[k + 1])).sum())
        cands.append({**c, "grids": g})
    if not cands:
        raise SystemExit(f"[build_low_prompt] 无满足 med∈[{low_lo},{low_hi}] 且 voiced≥0.55 的候选")

    # 贪心：每轮选对剩余需求覆盖增益最大的候选，直到预算用满或需求覆盖完
    rem = demand.copy()
    picked: list = []
    used = 0.0
    while used < budget:
        best, best_gain = None, -1
        for c in cands:
            if c in picked:
                continue
            gain = int(np.minimum(c["grids"], rem).sum())
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None or best_gain <= 0:
            break
        picked.append(best)
        rem -= np.minimum(best["grids"], rem)
        used += best["dur"]
    picked.sort(key=lambda c: c["start_sec"])  # 按参考时间序拼接

    wavs, f0s = [], []
    for c in picked:
        y, sr = sf.read(c["wav"])
        assert sr == 44100 and y.ndim == 1, (c["wav"], sr, y.ndim)
        wavs.append(y)
        f0s.append(np.load(c["f0"]))
    # 10ms 余弦淡入淡出拼接
    y = wavs[0]
    for nxt in wavs[1:]:
        xf = int(FADE_SEC * 44100)
        fade = np.cos(np.linspace(0, np.pi / 2, xf))
        tail = y[-xf:].copy()
        head = nxt[:xf].copy()
        y = np.concatenate([y[:-xf], tail * fade + head * (1 - fade), nxt[xf:]])
    f0 = np.concatenate(f0s)
    sf.write(out_wav, y, 44100)
    np.save(out_f0, f0)

    v = f0[f0 > 0]
    cov = np.minimum(demand, sum(c["grids"] for c in picked))
    stats = {
        "n_picked": len(picked),
        "dur": float(len(y) / 44100),
        "med": float(np.median(v)) if len(v) else 0.0,
        "p10": float(np.percentile(v, 10)) if len(v) else 0.0,
        "p90": float(np.percentile(v, 90)) if len(v) else 0.0,
        "grids_hz": edges,
        "demand_frames": demand.tolist(),
        "covered_frames": np.minimum(demand, np.array(sum((c["grids"] for c in picked), np.zeros_like(demand)))).tolist(),
        "picked": [{"idx": c["idx"], "start_sec": c["start_sec"], "dur": c["dur"], "med": c["med"], "voiced": c["voiced"]} for c in picked],
    }
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    cov_sum = int(np.minimum(demand, np.array(sum((c["grids"] for c in picked), np.zeros_like(demand)))).sum())
    print(f"[low_prompt] {len(picked)} 段, {len(y)/44100:.1f}s, med={stats['med']:.0f}Hz "
          f"p10={stats['p10']:.0f} p90={stats['p90']:.0f}, 低段需求覆盖 {cov_sum}/{int(demand.sum())} 帧")


# ---------------------------------------------------------------- 逐秒 SIM
class SimScorer:
    def __init__(self, ref_path: str):
        import torch
        sv = SV()
        self.sv = sv
        self.gt = sv.emb(load16k(ref_path))
        torch.cuda.empty_cache()

    def sim(self, wav16: np.ndarray, t0: float, t1: float) -> float:
        i0, i1 = int(t0 * 16000), int(t1 * 16000)
        seg = wav16[i0:i1]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.005:
            return float("nan")
        return cos(self.gt, self.sv.emb(seg))


def per_second_sims(scorer: SimScorer, vocal16: np.ndarray, reg16: np.ndarray,
                    t0: float, t1: float, shift: float) -> dict:
    """逐 1s 窗 SIM：基线取 vocal 文件时间 t；候选取 region 时间 τ=t-shift-(t0-PAD)。"""
    out = []
    for t in np.arange(t0, t1, 1.0):
        tau = t - shift - (t0 - PAD)
        if tau < 0 or tau + 1.0 > len(reg16) / 16000:
            continue
        base = scorer.sim(vocal16, t, t + 1.0)
        cand = scorer.sim(reg16, tau, tau + 1.0)
        if np.isnan(base) or np.isnan(cand):
            continue
        out.append((t, base, cand))
    return {"rows": out, "n": len(out)}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--spans", required=True, nargs="+", help="t0-t1 秒（静音点界定的乐句组）")
    ap.add_argument("--low-band", default="170-300", help="低音锚定音域 Hz")
    ap.add_argument("--budget", type=float, default=30.0, help="低音 prompt 时长上限(秒)")
    ap.add_argument("--arms", nargs="+", default=["low:0.5:0.2:1", "orig:1.0:0.4:1"],
                    help="候选臂 prompt_tag:cfg:alpha:seed（tag: low/orig/nsf8）")
    ap.add_argument("--min-gain", type=float, default=0.015,
                    help="替换门槛：臂均值 SIM − 基线 ≥ 此值才替换")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--out", default="vocal_repaired_low.wav")
    ap.add_argument("--prompt-wav", default=None, help="外部 prompt wav（跳过低音 prompt 构建）")
    ap.add_argument("--prompt-f0", default=None, help="外部 prompt f0 npy")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    low_lo, low_hi = map(float, args.low_band.split("-"))
    lead = os.path.join(d, "sep2", "lead.wav")
    f0f = os.path.join(d, "target_f0.npy")
    vocal = os.path.join(d, "vocal.wav")
    out_path = os.path.join(d, args.out)
    ref = os.path.abspath(args.ref)
    spans = [tuple(map(float, s.split("-"))) for s in args.spans]
    for p in [lead, f0f, vocal, ref]:
        if not os.path.exists(p):
            raise SystemExit(f"缺少文件: {p}")

    # 1. 低音锚定 prompt（low 臂用；orig/nsf8 臂用现成文件）
    low_prompt = os.path.join(d, "low_prompt.wav")
    low_prompt_f0 = os.path.join(d, "low_prompt_f0.npy")
    build_low_prompt(d, low_lo, low_hi, args.budget, low_prompt, low_prompt_f0,
                     os.path.join(d, "low_prompt_stats.json"))
    prompt_by_tag = {
        "low": (low_prompt, low_prompt_f0),
        "orig": (os.path.join(d, "prompt_ext.wav"), os.path.join(d, "prompt_ext_f0.npy")),
        "nsf8": (os.path.join(d, "_check", "prompt_ext-nsfhifigan--8.0-out.wav"),
                 os.path.join(d, "_check", "prompt_ext_nsf8_f0.npy")),
    }

    # 2. 数据
    vocal_w, _ = sf.read(vocal)
    assert vocal_w.ndim == 1
    lead_w, lead_sr = sf.read(lead)
    if lead_w.ndim > 1:
        lead_w = lead_w.mean(1)
    f0 = np.load(f0f)
    vocal16 = load16k(vocal)
    scorer = SimScorer(ref)

    tmp = tempfile.mkdtemp(prefix="repair_low_")
    print(f"workdir {tmp}")

    final = vocal_w.copy()
    for si, (t0, t1) in enumerate(spans):
        print(f"\n===== 乐句组 {si}: [{t0:.2f}s, {t1:.2f}s] =====")
        i0, i1 = int((t0 - PAD) * F0_RATE), int((t1 + PAD) * F0_RATE)
        j0, j1 = int((t0 - PAD) * lead_sr), int((t1 + PAD) * lead_sr)
        lf = os.path.join(tmp, f"lead{si}.wav")
        ff = os.path.join(tmp, f"f0{si}.npy")
        sf.write(lf, lead_w[j0:j1], lead_sr)
        np.save(ff, f0[i0:i1])

        results = []
        for arm in args.arms:
            tag, cfg, alpha, seed = arm.split(":")
            cfg, alpha, seed = float(cfg), float(alpha), int(seed)
            if tag not in prompt_by_tag:
                raise SystemExit(f"未知 prompt tag: {tag}（可用: {list(prompt_by_tag)}）")
            pwav, pf0 = prompt_by_tag[tag]
            gen = run_svc(pwav, pf0, ref, lf, ff,
                          os.path.join(tmp, f"r{si}_{tag}_c{cfg:g}_a{alpha:g}_s{seed}"),
                          cfg, alpha, seed)
            reg, reg_sr = sf.read(gen)
            assert reg_sr == SR
            shift, corr = find_shift(vocal_w, reg, t0, t1)
            reg16 = load16k(gen)
            comp = per_second_sims(scorer, vocal16, reg16, t0, t1, shift)
            rows, n = comp["rows"], comp["n"]
            base_mean = float(np.mean([r[1] for r in rows])) if n else float("nan")
            cand_mean = float(np.mean([r[2] for r in rows])) if n else float("nan")
            print(f"  臂 {tag} cfg={cfg:g} α={alpha:g} seed={seed}: Δ={shift:+.3f}s (corr={corr:.3f}), "
                  f"n={n}, 基线SIM={base_mean:.3f} → {cand_mean:.3f} (Δ{cand_mean-base_mean:+.3f})")
            results.append((tag, cfg, alpha, seed, shift, rows, base_mean, cand_mean))

        # 逐秒表
        print("  秒段    基线   " + "  ".join(f"{a[0]}c{a[1]:g}a{a[2]:g}s{a[3]}" for a in results))
        all_t = sorted({r[0] for rr in results for r in rr[5]})
        for t in all_t:
            line = f"  {int(t):02d}-{int(t)+1:02d}s  "
            base = next((r[1] for rr in results for r in rr[5] if r[0] == t), None)
            line += f"{base:.3f}   " if base is not None else "  -    "
            for (tag, cfg, alpha, seed, shift, rows, *_ ) in results:
                c = next((r[2] for r in rows if r[0] == t), None)
                line += f"{c:.3f}   " if c is not None else "  -    "
            print(line)

        # 选臂 + 门
        best = max(results, key=lambda r: r[7])
        bm, cm = best[6], best[7]
        if cm - bm >= args.min_gain:
            tag, cfg, alpha, seed, shift, rows = best[:6]
            print(f"  ✔ 选臂 {tag} cfg={cfg:g} α={alpha:g} seed={seed}（{cm-bm:+.3f} ≥ {args.min_gain}），拼接")
            reg, _ = sf.read(os.path.join(tmp, f"r{si}_{tag}_c{cfg:g}_a{alpha:g}_s{seed}"))
            b_start = find_common_silence(vocal_w, reg, t0, shift, t0, lo=-0.5, hi=0.8)
            b_end = find_common_silence(vocal_w, reg, t0, shift, t1, lo=-0.8, hi=0.5)
            if b_start is None or b_end is None or b_end - b_start < 1.0:
                xf_use = 0.25
                fb_s = orig_silence_mid(vocal_w, t0, lo=-0.5, hi=0.8)
                fb_e = orig_silence_mid(vocal_w, t1, lo=-0.8, hi=0.5)
                b_start, b_end = fb_s or (t0 - 0.3), fb_e or (t1 + 0.2)
                print(f"    ⚠ 未找到公共静音，回退原版气口 [{b_start:.3f}, {b_end:.3f}]（淡化 {xf_use}s）")
            else:
                xf_use = XF
            print(f"    拼接边界: [{b_start:.3f}s, {b_end:.3f}s]")
            g0 = int((b_start - shift - (t0 - PAD)) * SR)
            g1 = int((b_end - shift - (t0 - PAD)) * SR)
            assert 0 <= g0 < g1 <= len(reg), (g0, g1, len(reg))
            v0, v1 = int(b_start * SR), int(b_end * SR)
            xf = int(xf_use * SR)
            old = final[v0:v1].copy()
            new = reg[g0:g1].copy()
            n = min(len(old), len(new))
            fade = np.cos(np.linspace(0, np.pi / 2, xf))
            for k in range(min(xf, n)):
                new[k] = new[k] * fade[k] + old[k] * (1 - fade[k])
            for k in range(1, min(xf, n) + 1):
                new[-k] = new[-k] * fade[-k] + old[-k] * (1 - fade[-k])
            final[v0:v0 + n] = new[:n]
            print(f"    已替换 [{b_start:.3f}, {b_end:.3f}]（平移 Δ={shift:+.3f}s）")
        else:
            print(f"  ✘ 无臂达标（最佳 Δ{cm-bm:+.3f} < {args.min_gain}），本段保留原 vocal")

    sf.write(out_path, final, SR)
    print(f"\n已写出 {out_path}")
    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()