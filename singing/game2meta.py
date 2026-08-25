#!/usr/bin/env python3
"""GAME 输出的 MIDI -> SoulX metadata，供与 ROSVOT 基线同协议对比。

歌词来源：复用 baseline metadata 里已被 LRC 纠正过的词序列（svs_qc.py --lrc
的产物），按时间重叠分配到 GAME 音符上。一个词跨多个音符时，第一个音符拿词、
后续是 slur（type=3，SoulX 惯例重复音节）；没落在任何词里的音符 = <SP>。
分段规则与 svs_song.py 的 max_merge 一致：>2s 的间隙切段，单段 ≤20s。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "singing"))
from svs_g2p import g2p_words  # noqa: E402


def read_notes(midi_path: str):
    """(start_s, dur_s, midi_pitch) 列表，FIFO 配对防同音高重叠丢音符。"""
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000
    raw = []
    for tr in mid.tracks:
        t = 0
        active = {}
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append(t)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                q = active.get((msg.channel, msg.note))
                if q:
                    a = q.pop(0)
                    raw.append((a * tempo / 1e6 / tpb,
                                (t - a) * tempo / 1e6 / tpb,
                                msg.note))
    raw.sort()
    return [(round(s, 3), round(d, 3), p) for s, d, p in raw if d > 0.02]


def baseline_words(meta_path: str):
    """baseline metadata -> [(text, start_s, end_s)]（type==2 的音符即字槽）。"""
    segs = json.load(open(meta_path, encoding="utf-8"))
    out = []
    for s in segs:
        texts = s["text"].split()
        durs = [float(x) for x in s["duration"].split()]
        types = [int(x) for x in s["note_type"].split()]
        t0 = s["time"][0] / 1000.0
        t = 0.0
        for tx, du, ty in zip(texts, durs, types):
            if ty == 2:
                out.append((tx, round(t0 + t, 3), round(t0 + t + du, 3)))
            t += du
    return out


def attach_lyrics(notes, words):
    """把词按时间重叠分给音符。返回 [(start,dur,pitch,text,type)]。"""
    out = []
    wi = 0
    for s, d, p in notes:
        e = s + d
        # 找与该音符重叠最大的词
        best, best_ov = None, 0.0
        for j in range(wi, min(len(words), wi + 12)):
            wtx, ws, we = words[j]
            ov = min(e, we) - max(s, ws)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None and best_ov > 0:
            wi = best
            same_word = getattr(attach_lyrics, "_last", None) == best
            out.append((s, d, p, words[best][0], 3 if same_word else 2))
            attach_lyrics._last = best
        else:
            out.append((s, d, 0, "<SP>", 1))
    return out


def build_meta(notes_tagged, language="Mandarin", max_seg=20.0, gap_split=2.0):
    segs, cur, seg_start = [], [], None

    def flush():
        if not cur:
            return
        t0 = cur[0][0]
        t1 = cur[-1][0] + cur[-1][1]
        texts = [x[3] for x in cur]
        segs.append({
            "index": f"seg_{len(segs)}_{int(t0*1000)}",
            "language": language,
            "time": [int(round(t0 * 1000)), int(round(t1 * 1000))],
            "duration": " ".join(f"{x[1]:.2f}" for x in cur),
            "text": " ".join(texts),
            "phoneme": " ".join(g2p_words(texts, language)),
            "note_pitch": " ".join(str(x[2]) for x in cur),
            "note_type": " ".join(str(x[4]) for x in cur),
            "f0": "",
        })

    prev_end = None
    for n in notes_tagged:
        # 关键：SoulX metadata 的 duration 是**连续**累加的（data_processor 不看
        # 绝对时间），段内任何空隙必须用 <SP> 占位填掉，否则该段之后所有音符
        # 的相对时间整体前移、QC/推理全错位。第一版漏了这一步，把 GAME 自己
        # 准确的转录评成了 |note-F0| median 1.95st 的假象。
        if prev_end is not None and n[0] - prev_end > 0.02:
            cur.append((prev_end, round(n[0] - prev_end, 2), 0, "<SP>", 1))
        if prev_end is not None and n[0] - prev_end > gap_split:
            flush()
            cur = []
        if sum(x[1] for x in cur) > max_seg:
            flush()
            cur = []
        if not cur:
            seg_start = n[0]
        cur.append(n)
        prev_end = n[0] + n[1]
    flush()

    # 填 f0：从绝对时间轴的 target_f0.npy 按段窗口切
    import numpy as np
    f0full = np.load(str(ROOT / "data/songs/爱笑的眼睛/target_f0.npy"))
    for s in segs:
        a = int(round(s["time"][0] / 1000.0 * 50))
        b = int(round(s["time"][1] / 1000.0 * 50))
        s["f0"] = " ".join(f"{float(x):.1f}" for x in f0full[a:b])
    return segs


def extend_tails(notes_tagged, min_voiced=0.5):
    """GAME 会把音符尾音和滑音段剪掉（真音符浊音覆盖率 72.8%，ROSVOT 81.1%），
    剪掉的部分变成音符间空隙 -> 被填成 <SP> -> 句中插气口，即 AGENTS.md 记过的
    「slur 处重新起音」伪影成因。

    规则（无 ML）：两音符之间的空隙若 RMVPE 显示过半数帧仍在唱，就归给**音高
    更接近的邻侧**并延长它（滑音时音高在动、两侧都不匹配，但滑音也是音节的
    延续，SoulX 按音符时长持续发声，延长比插气口正确）。完全清音的空隙保持
    <SP>（真换气）。GAME 音头极准（99% 落在唱声里），所以绝不前移下一音头。
    """
    import numpy as np
    f0full = np.load(str(ROOT / "data/songs/爱笑的眼睛/target_f0.npy"))
    out = [list(n) for n in notes_tagged]
    for i in range(len(out) - 1):
        s, d, p, tx, ty = out[i]
        ns = out[i + 1][0]
        gap0, gap1 = s + d, ns - 0.02
        if gap1 - gap0 <= 0.03:
            continue
        a, b = int(round(gap0 * 50)), max(int(round(gap1 * 50)),
                                          int(round(gap0 * 50)) + 1)
        w = f0full[a:min(b, len(f0full))]
        if len(w) < 2 or (w > 0).mean() < min_voiced:
            continue
        wv = w[w > 0]
        fm = 69 + 12 * np.log2(np.median(wv) / 440.0)
        # 归给音高更近的一侧；两侧都无音高（<SP> 邻接）时不延
        dp_prev = abs(fm - p) if p > 0 else 99
        np_ = out[i + 1][2]
        dp_next = abs(fm - np_) if np_ > 0 else 99
        if dp_prev <= dp_next and p > 0:
            out[i][1] = round(ns - 0.02 - s, 2)
        elif np_ > 0:
            out[i + 1][0] = round(gap0 + 0.02, 2)
            out[i + 1][1] = round(out[i + 1][1] + (ns - 0.02 - gap0 - 0.02), 2)
    return [tuple(x) for x in out]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", required=True)
    ap.add_argument("--base-meta", required=True, help="已纠词的 baseline metadata")
    ap.add_argument("--out", required=True)
    ap.add_argument("--f0", default=str(ROOT / "data/songs/爱笑的眼睛/target_f0.npy"))
    args = ap.parse_args()

    notes = read_notes(args.mid)
    words = baseline_words(args.base_meta)
    tagged = attach_lyrics(notes, words)
    tagged = extend_tails(tagged)
    segs = build_meta(tagged)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(segs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    nw = sum(1 for s in segs for t, ty in zip(s["text"].split(), s["note_type"].split()) if ty == "2")
    print(f"[game2meta] {len(notes)} GAME 音符 -> {len(segs)} 段 / "
          f"{sum(len(s['text'].split()) for s in segs)} 音符（有词 {nw}）")
    print(f"  词槽 baseline {len(words)} vs GAME 有词 {nw}")
