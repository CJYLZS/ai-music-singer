#!/usr/bin/env python3
"""V2 全曲对比：SOFA 的休止结构 vs 现有 metadata 的 <SP> 结构。

V2 单句实测发现：两者在**边界精度**上都贴着 F0 噪声底（±20~30ms），差别在
**结构** —— metadata 里有大量落在连续人声上的假休止（voiced_frac 高达 1.00、
时长 2~3s），SOFA 不会造这种休止。假休止正是手工量的主要来源：
它会派生出假的"乐句起始"（v0 审计里 19/19 ONSET_BLIND 全部前接高有声休止）。

用法: SoulX-Singer/.venv/bin/python singing/svs_v2_rest_cmp.py \
        --meta data/songs/X/svs/target/metadata.json \
        --seg-dir tools/sofa/segments/v2all --f0-dir <long_cut_wavs>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

F0_FPS = 50.0
RESTS = {"AP", "SP", "<AP>", "<SP>", "pau", "cl"}


def voiced_frac(voiced: np.ndarray, a: float, b: float) -> float:
    w = voiced[max(int(a * F0_FPS), 0):min(int(b * F0_FPS), len(voiced))]
    return float(w.mean()) if len(w) else 0.0


def spans(labels, durs, keep) -> list[tuple[float, float]]:
    t, out = 0.0, []
    for lb, du in zip(labels, durs):
        if keep(lb):
            out.append((t, t + du))
        t += du
    return out


def stat(name: str, sp: list[tuple[float, float]], voiced: np.ndarray) -> dict:
    fr = [voiced_frac(voiced, a, b) for a, b in sp]
    bad = [(a, b, f) for (a, b), f in zip(sp, fr) if f >= 0.5]
    return {"n": len(sp), "bad": len(bad),
            "bad_sec": sum(b - a for a, b, _ in bad),
            "worst": max([(f, b - a) for (a, b, f) in bad], default=(0.0, 0.0)),
            "tot_sec": sum(b - a for a, b in sp)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--f0-dir", required=True)
    args = ap.parse_args()

    trans = {r["name"]: r for r in csv.DictReader(
        open(Path(args.seg_dir) / "transcriptions" / "transcriptions.csv",
             encoding="utf-8"))}
    segs = json.load(open(args.meta, encoding="utf-8"))
    agg = {"asr": [], "sofa": []}
    print(f"{'segment':<24} {'ASR n/坏/坏秒':>16}  {'SOFA n/坏/坏秒':>16}")
    for seg in segs:
        idx = seg["index"]
        if idx not in trans:
            continue
        voiced = np.load(Path(args.f0_dir) / f"{idx}_f0.npy") > 0
        a_sp = spans(seg["text"].split(),
                     [float(x) for x in seg["duration"].split()],
                     lambda x: x == "<SP>")
        r = trans[idx]
        s_sp = spans(r["ph_seq"].split(),
                     [float(x) for x in r["ph_dur"].split()],
                     lambda x: x in RESTS)
        sa, ss = stat("asr", a_sp, voiced), stat("sofa", s_sp, voiced)
        agg["asr"].append(sa)
        agg["sofa"].append(ss)
        print(f"{idx:<24} {sa['n']:3d}/{sa['bad']:2d}/{sa['bad_sec']:5.1f}s"
              f"       {ss['n']:3d}/{ss['bad']:2d}/{ss['bad_sec']:5.1f}s")

    print("\n=== 全曲汇总（坏休止 = voiced_frac>=0.5，即休止槽里其实在唱）")
    for k in ("asr", "sofa"):
        n = sum(x["n"] for x in agg[k])
        bad = sum(x["bad"] for x in agg[k])
        bs = sum(x["bad_sec"] for x in agg[k])
        tot = sum(x["tot_sec"] for x in agg[k])
        w = max((x["worst"] for x in agg[k]), default=(0, 0))
        print(f"{k.upper():<5} 休止 {n:3d} 个 / {tot:6.1f}s   "
              f"坏休止 {bad:3d} 个 / {bs:5.1f}s ({bs/max(tot,1e-9):5.1%})   "
              f"最坏: voiced={w[0]:.2f} dur={w[1]:.2f}s")


if __name__ == "__main__":
    main()
