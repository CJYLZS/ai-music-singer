#!/usr/bin/env python
"""漏音检测阶段（流水线 worker）：双模型 1s+3s 逐窗扫描 + 共识区域 + 建议修复跨度。

协议（AGENTS.md「漏音检测与局部修复」）：1s 单模型逐窗不可靠，3s 双模型共识才可信。
对每个窗口：excess = C(像原唱) − mean(C_impostor)（扣分离歌声域偏置），
margin = A(像参考) − C。

共识规则（自校准，逐 (模型, 窗长) 独立）：
  窗口标记 = er_excess ≥ max(FLOOR_ER, p85_er) 且 cp_excess ≥ max(FLOOR_CP, p85_cp)
  FLOOR_ER=0.05, FLOOR_CP=0.07（分离歌声域偏置地板，低于它的是域噪声不是漏音）
3s 共识窗合并成核心区域；1s 共识窗（±3s 内）扩展区域边界；区域外扩后锚定到
原版 vocal 的气口（静音窗）得到建议修复跨度 spans，供 leak-repair 阶段使用。

输出 JSON（--out-json）：
  {win, n_windows, B, models{...}, windows[{t0, er_excess, er_margin, cp_excess,
   cp_margin, consensus}], regions[{t0,t1,er_mean,cp_mean,n}], spans[[t0,t1],...],
   impostors[...], ref, vocal}

用法（由 sing.py leak-detect 阶段调用）:
  seed-vc/.venv/bin/python singing/w_leak_detect.py --song-dir data/songs/雨爱 \
      --ref data/voices/ref_talk_sing.wav --vocal data/songs/雨爱/vocal.wav \
      --impostor <lead>... --out-json data/songs/雨爱/leak_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_triad import SV, load16k, cos, windows  # noqa: E402
from eval_leak_campplus import CP  # noqa: E402
from repair_splice import SR  # noqa: E402
from emb_cache import cached_emb  # noqa: E402

FLOOR_ER = 0.05
FLOOR_CP = 0.07
SIL_THR = 0.04
SIL_MINLEN = 0.10


def silence_windows(w, t0, t1, thr=SIL_THR, minlen=SIL_MINLEN, hop=0.01):
    """vocal(24k) 在 [t0,t1] 内的静音窗 [(a,b),...]。"""
    a, b = int(t0 * SR), int(t1 * SR)
    seg = w[a:b]
    n = int(hop * SR)
    r = np.sqrt(np.mean(seg[: len(seg) // n * n].reshape(-1, n) ** 2, axis=1))
    t = t0 + np.arange(len(r)) * hop
    q = r < thr
    s = None
    out = []
    for i, v in enumerate(q):
        if v and s is None:
            s = i
        if not v and s is not None:
            if t[i] - t[s] >= minlen:
                out.append((t[s], t[i]))
            s = None
    if s is not None and t[-1] - t[s] >= minlen:
        out.append((t[s], t[-1]))
    return out


def span_for_region(vocal_w, reg_t0, reg_t1):
    """区域 → 建议修复跨度：向外找原版气口（静音窗）锚定。

    start = [reg_t0-6, reg_t0+0.5] 内最后一个静音窗的 end + 0.03（找不到 → reg_t0-0.5）
    end   = [reg_t1-0.5, reg_t1+6] 内第一个静音窗的 start + 0.10（找不到 → reg_t1+0.5）
    """
    sils = silence_windows(vocal_w, reg_t0 - 7, reg_t1 + 7)
    starts = [s for s in sils if reg_t0 - 6 <= s[1] <= reg_t0 + 0.5]
    ends = [s for s in sils if reg_t1 - 0.5 <= s[0] <= reg_t1 + 6]
    s0 = (max(starts, key=lambda s: s[1])[1] + 0.03) if starts else reg_t0 - 0.5
    s1 = (min(ends, key=lambda s: s[0])[0] + 0.10) if ends else reg_t1 + 0.5
    s0 = max(s0, 0.0)
    return round(s0, 2), round(s1, 2)


def merge_regions(times_3s, flags_3s, times_1s, flags_1s, gap=3.0, expand=3.0):
    """3s 共识窗合并成区域；1s 共识窗（距区域 ≤ expand）扩展边界。"""
    core = sorted(t for t, f in zip(times_3s, flags_3s) if f)
    regions = []
    for t in core:
        if regions and t - regions[-1][1] <= gap:
            regions[-1][1] = t + 3.0
        else:
            regions.append([t, t + 3.0])
    one = sorted(t for t, f in zip(times_1s, flags_1s) if f)
    for t in one:
        for r in regions:
            if t + 1.0 >= r[0] - expand and t <= r[1] + expand:
                r[0] = min(r[0], t)
                r[1] = max(r[1], t + 1.0)
                break
    return regions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--impostor", action="append", default=[], metavar="LEAD",
                    help="可多次指定；nargs=* 的重复 flag 只有最后一个生效，故用 append")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    target = os.path.join(d, "sep2", "lead.wav")
    ref = os.path.abspath(args.ref)
    vocal = os.path.abspath(args.vocal)
    impostors = [os.path.abspath(p) for p in args.impostor]
    for p in [ref, target, vocal, *impostors]:
        if not os.path.exists(p):
            raise SystemExit(f"文件不存在: {p}")

    sv, cp = SV(), CP()
    out16 = load16k(vocal)

    def emb_file(model, tag: str, path: str, device: str) -> torch.Tensor:
        """整文件嵌入 + 磁盘缓存。这里一次要嵌入 537s 参考 + 268s 目标 + 最多 5 个
        268s impostor（两个模型各一遍，合计 ~2.1 小时音频），537s 那一发整文件前向
        瞬时就要 ~15.5GB 显存 —— 16GB 卡上的 OOM 高发点，见 singing/emb_cache.py。
        注意两个模型的输出设备不同：SV.emb 在 self.device，CP.emb 固定返回 CPU
        （CAMPPlus 内部 fbank 在 CPU 上做），窗口嵌入也必须落在同一设备才能算 cos。"""
        e = cached_emb(path, tag, lambda: model.emb(load16k(path)), log=print)
        return e.to(device)

    a_ref = emb_file(sv, "eres2netv2", ref, sv.device)
    a_tgt = emb_file(sv, "eres2netv2", target, sv.device)
    a_ref_c = emb_file(cp, "campplus", ref, "cpu")
    a_tgt_c = emb_file(cp, "campplus", target, "cpu")
    imps_sv = [emb_file(sv, "eres2netv2", p, sv.device) for p in impostors]
    imps_cp = [emb_file(cp, "campplus", p, "cpu") for p in impostors]
    torch.cuda.empty_cache()

    report = {"win": 3.0, "ref": ref, "vocal": vocal, "target": target,
              "impostors": impostors, "n_impostor": len(impostors)}
    windows_out = []

    def scan_model(emb_fn, a_r, a_t, imps, win):
        times, ex, mg = [], [], []
        for t0, seg in windows(out16, win):
            e = emb_fn(seg)
            A, C = cos(e, a_r), cos(e, a_t)
            Ci = float(np.mean([cos(e, a) for a in imps])) if imps else float("nan")
            times.append(t0)
            ex.append((C - Ci) if imps else float("nan"))
            mg.append(A - C)
        return np.array(times), np.array(ex), np.array(mg)

    results = {}
    for win in (3.0, 1.0):
        t3, e3, m3 = scan_model(sv.emb, a_ref, a_tgt, imps_sv, win)
        tc, ec, mc = scan_model(cp.emb, a_ref_c, a_tgt_c, imps_cp, win)
        results[win] = (t3, e3, m3, ec, mc)

    # ---- 阈值（自校准）+ 共识标记 ----
    t3, e3, m3, ec3, mc3 = results[3.0]
    t1, e1, m1, ec1, mc1 = results[1.0]
    th = {
        "er3": max(FLOOR_ER, float(np.percentile(e3, 85))),
        "cp3": max(FLOOR_CP, float(np.percentile(ec3, 85))),
        "er1": max(FLOOR_ER, float(np.percentile(e1, 85))),
        "cp1": max(FLOOR_CP, float(np.percentile(ec1, 85))),
    }
    f3 = (e3 >= th["er3"]) & (ec3 >= th["cp3"])
    f1 = (e1 >= th["er1"]) & (ec1 >= th["cp1"])
    report["thresholds"] = th
    report["models"] = {
        "eres2net": {"mean_excess_3s": round(float(e3.mean()), 4),
                     "p85_excess_3s": round(th["er3"], 4),
                     "mean_margin_3s": round(float(m3.mean()), 4)},
        "campplus": {"mean_excess_3s": round(float(ec3.mean()), 4),
                     "p85_excess_3s": round(th["cp3"], 4),
                     "mean_margin_3s": round(float(mc3.mean()), 4)},
    }

    # 3s 窗口明细（含共识标记）
    for i, t in enumerate(t3):
        windows_out.append({
            "t0": round(float(t), 2),
            "er_excess": round(float(e3[i]), 4), "er_margin": round(float(m3[i]), 4),
            "cp_excess": round(float(ec3[i]), 4), "cp_margin": round(float(mc3[i]), 4),
            "consensus": bool(f3[i]),
        })
    report["n_windows"] = len(t3)
    report["windows"] = windows_out

    # ---- 区域 + 建议跨度 ----
    regions = merge_regions(t3, f3, t1, f1)
    vocal_w, _ = __import__("soundfile").read(vocal)
    spans, r_out = [], []
    for r0, r1 in regions:
        s0, s1 = span_for_region(vocal_w, r0, r1)
        sel3 = (t3 >= r0 - 0.5) & (t3 <= r1 + 0.5)
        r_out.append({"t0": round(r0, 2), "t1": round(r1, 2),
                      "er_excess_mean": round(float(e3[sel3].mean()), 4),
                      "cp_excess_mean": round(float(ec3[sel3].mean()), 4),
                      "n": int(sel3.sum())})
        spans.append([s0, s1])
    # 合并重叠/相邻跨度
    merged = []
    for s in spans:
        if merged and s[0] <= merged[-1][1] + 2.0:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(list(s))
    report["regions"] = r_out
    report["spans"] = merged

    print(f"漏音检测完成: {len(t3)} 个 {report['win']:.0f}s 窗, impostor={len(impostors)}")
    print(f"  阈值: er3={th['er3']:.3f} cp3={th['cp3']:.3f} er1={th['er1']:.3f} cp1={th['cp1']:.3f}")
    print(f"  共识 3s 窗: {int(f3.sum())} 个, 共识 1s 窗: {int(f1.sum())} 个")
    print(f"  漏音区域: {len(regions)} 个")
    for (r0, r1), s in zip(regions, spans):
        print(f"    [{r0:.1f}s, {r1:.1f}s] -> 建议修复跨度 [{s[0]:.2f}s, {s[1]:.2f}s]")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"报告已写入 {args.out_json}")


if __name__ == "__main__":
    main()
