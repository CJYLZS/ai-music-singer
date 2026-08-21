#!/usr/bin/env python3
"""验证 repair_low 的结果：逐秒 SIM 对比（基线 vs 修复后）、拼接边界 1.5s 窗 drift、
输出两段试听片段（基线/修复后对齐同窗口）。GPT-SoVITS venv 或 SoulX venv 均可。"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from eval_sim import SV, load16k, cos

REF = os.path.join(REPO, "data", "voices", "ref_talk_sing.wav")
SPANS = [(25.5, 30.87), (139.5, 144.35)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--repaired", default="vocal_repaired_low.wav")
    ap.add_argument("--out-dir", default="_check")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    base = os.path.join(d, "vocal.wav")
    rep = os.path.join(d, args.repaired)

    sv = SV()
    gt = sv.emb(load16k(REF))
    import torch
    torch.cuda.empty_cache()

    b16 = load16k(base)
    r16 = load16k(rep)
    outd = os.path.join(d, args.out_dir)
    os.makedirs(outd, exist_ok=True)

    def sim(wav16, t0, t1):
        seg = wav16[int(t0 * 16000):int(t1 * 16000)]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.005:
            return float("nan")
        return cos(gt, sv.emb(seg))

    print("=== 逐秒 SIM（基线 → 修复后）===")
    for (t0, t1) in SPANS:
        print(f"[{t0:.1f}-{t1:.1f}s]")
        for t in np.arange(t0, t1, 1.0):
            a, b = sim(b16, t, t + 1), sim(r16, t, t + 1)
            mark = "  <-- 变了" if (not np.isnan(a) and not np.isnan(b) and abs(b - a) > 0.01) else ""
            print(f"  {int(t):02d}-{int(t)+1:02d}s  {a if not np.isnan(a) else float('nan'):.3f} → {b if not np.isnan(b) else float('nan'):.3f}{mark}")

    print("\n=== 拼接边界相邻 1.5s 窗 drift ===")
    for (t0, t1) in SPANS:
        for edge in (t0, t1):
            vals = {"base": [], "rep": []}
            for w in np.arange(edge - 3.0, edge + 1.5, 0.5):
                if w + 1.5 > t0 + 1 and w < t1 - 1:
                    continue  # 只报告边界两侧
                for nm, w16 in (("base", b16), ("rep", r16)):
                    s = sim(w16, w, w + 1.5)
                    if not np.isnan(s):
                        vals[nm].append(s)
            for nm in ("base", "rep"):
                v = vals[nm]
                if len(v) >= 2:
                    dr = float(np.mean(np.abs(np.diff(v))))
                    print(f"  edge {edge:7.2f}s {nm}: n={len(v)} 1.5s窗, 相邻窗|ΔSIM|均值={dr:.3f}")

    # 试听片段（26-30 / 141-144）
    for (t0, t1, tag) in [(26, 30, "26-30"), (141, 144, "141-144")]:
        for nm, p in (("base", base), ("rep", rep)):
            y, sr = sf.read(p)
            sf.write(os.path.join(outd, f"lowfix_{tag}_{nm}.wav"),
                     y[int(t0 * sr):int(t1 * sr)], sr)
        print(f"\n试听片段: {outd}/lowfix_{tag}_base.wav / lowfix_{tag}_rep.wav")


if __name__ == "__main__":
    main()