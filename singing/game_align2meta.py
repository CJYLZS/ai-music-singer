#!/usr/bin/env python3
"""GAME align 模式输出 -> SoulX metadata。

align 的输入是完整连续时间轴（词 + <SP> 占位，与 baseline metadata 同构），
输出 note_seq/note_dur/note_slur 的时间轴总长与输入一致，且音符不跨词边界
—— 所以按**累计时间**把音符分配回 token：累计时长越过哪个 token 的边界就
属于哪个 token。token 是 <SP> 的拿 <SP> 音符（pitch 0, type 1）；实字 token
的第一个音符 type=2，其余按 GAME 给的 slur 标志置 type=3。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "singing"))
from svs_g2p import g2p_words  # noqa: E402

NOTE_NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
              "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def parse_note(tok: str):
    """'A#3+32' / 'rest' -> (midi 或 0)。音分偏移四舍五入进半音。"""
    if tok.lower() == "rest":
        return 0
    sign = 1
    body = tok
    for c in "+-":
        if c in body:
            i = body.index(c)
            sign = -1 if c == "-" else 1
            cents = sign * int(body[i + 1:])
            body = body[:i]
            break
    else:
        cents = 0
    name, octv = body[:-1], int(body[-1])
    midi = (octv + 1) * 12 + NOTE_NAMES[name]
    return max(0, min(127, midi + (1 if cents >= 50 else -1 if cents <= -50 else 0)))


def build(meta_base: str, csv_path: str, out_path: str, f0_path: str):
    import numpy as np
    base = json.load(open(meta_base, encoding="utf-8"))
    toks = [t for s in base for t in s["text"].split()]
    durs = [float(x) for s in base for x in s["duration"].split()]

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    seq = rows[0]["note_seq"].split()
    ndur = [float(x) for x in rows[0]["note_dur"].split()]
    slurs = rows[0]["note_slur"].split() if "note_slur" in rows[0] else ["0"] * len(seq)

    # 累计时间映射音符->token
    bounds = np.cumsum([0.0] + durs)
    notes = []
    t = 0.0
    ti = 0
    for k, (tok, du) in enumerate(zip(seq, ndur)):
        while ti + 1 < len(toks) and t >= bounds[ti + 1] - 1e-6:
            ti += 1
        notes.append((t, du, tok, slurs[k], ti))
        t += du

    segs, cur_toks, cur_notes = [], [], []   # 按 token 顺序重组为段
    f0full = np.load(f0_path)

    def flush(tokens, items):
        if not tokens:
            return
        texts = []
        types = []
        pitches = []
        durlist = []
        for tx, its in zip(tokens, items):
            for j, (_, du, ntok, sl, _) in enumerate(its):
                p = parse_note(ntok)
                if tx == "<SP>":
                    texts.append("<SP>"); types.append(1); pitches.append(0)
                else:
                    texts.append(tx)
                    types.append(3 if (sl not in ("0", "False")) or j > 0 else 2)
                    pitches.append(p)
                durlist.append(round(du, 2))
        t0 = items[0][0][0] if items else 0.0
        span = sum(durlist)
        a = int(round(t0 * 50)); b = int(round((t0 + span) * 50))
        segs.append({
            "index": f"game_{len(segs)}_{int(t0*1000)}",
            "language": "Mandarin",
            "time": [int(round(t0 * 1000)), int(round((t0 + span) * 1000))],
            "duration": " ".join(map(str, durlist)),
            "text": " ".join(texts),
            "phoneme": " ".join(g2p_words(texts, "Mandarin")),
            "note_pitch": " ".join(map(str, pitches)),
            "note_type": " ".join(map(str, types)),
            "f0": " ".join(f"{float(x):.1f}" for x in f0full[a:b]),
        })

    cur_tokens: list[str] = []
    cur_items: list[list] = []
    acc = 0.0
    seg_acc = 0.0
    ti = 0
    per_tok: dict[int, list] = {}
    for (t, du, ntok, sl, k) in notes:
        per_tok.setdefault(k, []).append((t, du, ntok, sl, k))
    # 按段切（沿用 baseline 的段结构最稳：直接继承原 time 边界）
    idx = 0
    for s in base:
        n = len(s["text"].split())
        toks_i = s["text"].split()
        items = [per_tok.get(idx + j, []) for j in range(n)]
        flush(toks_i, items)
        idx += n
    json.dump(segs, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    nw = sum(1 for s in segs for ty in s["note_type"].split() if ty == "2")
    ns = sum(1 for s in segs for ty in s["note_type"].split() if ty == "3")
    print(f"[align2meta] {len(seq)} 音符 -> {len(segs)} 段（有词 {nw}，slur {ns}）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-meta", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--f0", default=str(ROOT / "data/songs/爱笑的眼睛/target_f0.npy"))
    a = ap.parse_args()
    build(a.base_meta, a.csv, a.out, a.f0)
