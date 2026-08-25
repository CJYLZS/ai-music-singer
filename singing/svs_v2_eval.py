#!/usr/bin/env python3
"""V2 评测：SOFA 的时间轴 vs 现有 ASR 时间轴，同一把尺子（F0 证据）量。

尺子取自 svs_qc：乐句起始对"浊音段起点"，句中换音对"F0 音高台阶"。
**关键坑**：F0 浊音起点标的是**元音**开始，不是声母开始。SOFA 给的是音素级
边界，所以要拿它的**元音音素起点**去对 F0，拿**音节起点（声母）**去当 word_durs。
拿声母起点直接对 F0 会凭空多出 40~100ms 的假误差。

用法: SoulX-Singer/.venv/bin/python singing/svs_v2_eval.py \
        --seg-dir tools/sofa/segments/v2 --index vocal_222180_239780 \
        --f0 data/songs/X/svs/target/long_cut_wavs/<index>_f0.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svs_qc import voiced_runs, pitch_steps, F0_FPS  # noqa: E402

# opencpop-extension 里的声母（其余当元音/韵母）
CONSONANTS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q",
              "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"}
REST = {"AP", "SP", "<AP>", "<SP>", "pau", "cl", ""}


def read_trans(path: str, index: str) -> tuple[list[str], list[float]]:
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["name"] == index:
                return (row["ph_seq"].split(),
                        [float(x) for x in row["ph_dur"].split()])
    raise SystemExit(f"{index} not in {path}")


def syllable_onsets(ph_seq: list[str], ph_dur: list[float]):
    """-> [(syl_start, vowel_start, label)]，休止(AP/SP)不计。"""
    t = 0.0
    out = []
    prev_rest = True
    for ph, du in zip(ph_seq, ph_dur):
        base = ph.rstrip("0123456789")
        if ph in REST:
            prev_rest = True
            t += du
            continue
        if base in CONSONANTS:
            out.append({"syl": t, "vowel": None, "ph": ph, "init": prev_rest})
        else:
            if out and out[-1]["vowel"] is None and not out[-1].get("done"):
                out[-1]["vowel"] = t          # 前面是声母，这里是它的元音
                out[-1]["done"] = True
            else:
                out.append({"syl": t, "vowel": t, "ph": ph, "init": prev_rest,
                            "done": True})
        prev_rest = False
        t += du
    for o in out:
        if o["vowel"] is None:
            o["vowel"] = o["syl"]
    return out


def gap_to(ref: np.ndarray, t: float) -> float:
    return float(np.min(np.abs(ref / F0_FPS - t))) if len(ref) else 9.99


def report(tag: str, g: list[float]) -> None:
    a = np.array(g)
    if not len(a):
        print(f"{tag:<26} n=0")
        return
    print(f"{tag:<26} n={len(a):<3} median={np.median(a)*1000:5.0f}ms "
          f"p75={np.percentile(a,75)*1000:5.0f}ms hit@60ms={np.mean(a<=0.06):5.1%} "
          f"hit@120ms={np.mean(a<=0.12):5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--f0", required=True)
    ap.add_argument("--meta", default="", help="对照：现有 metadata")
    args = ap.parse_args()

    d = Path(args.seg_dir)
    ph_seq, ph_dur = read_trans(d / "transcriptions" / "transcriptions.csv",
                                args.index)
    f0 = np.load(args.f0)
    onsets, steps = voiced_runs(f0 > 0), pitch_steps(f0)
    syl = syllable_onsets(ph_seq, ph_dur)
    print(f"SOFA: {len(ph_seq)} 音素 / {len(syl)} 音节, 音频 {len(f0)/F0_FPS:.1f}s, "
          f"浊音起点 {len(onsets)} 台阶 {len(steps)}")

    init_v = [gap_to(onsets, s["vowel"]) for s in syl if s["init"]]
    mid_v = [gap_to(steps, s["vowel"]) for s in syl if not s["init"]]
    report("SOFA 乐句起始(元音)", init_v)
    report("SOFA 句中(元音vs台阶)", mid_v)
    report("SOFA 声母起点 vs 浊音", [gap_to(onsets, s["syl"]) for s in syl
                                     if s["init"]])

    if args.meta:
        segs = json.load(open(args.meta, encoding="utf-8"))
        seg = next(s for s in segs if s["index"] == args.index)
        durs = [float(x) for x in seg["duration"].split()]
        texts, types = seg["text"].split(), [int(x) for x in
                                             seg["note_type"].split()]
        t, mi, mm = 0.0, [], []
        for k, (tx, du, ty) in enumerate(zip(texts, durs, types)):
            if ty != 1 and ty != 3:
                is_init = k == 0 or types[k - 1] == 1
                (mi if is_init else mm).append(
                    gap_to(onsets if is_init else steps, t))
            t += du
        print()
        report("ASR 乐句起始", mi)
        report("ASR 句中", mm)


if __name__ == "__main__":
    main()
