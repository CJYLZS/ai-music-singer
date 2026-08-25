#!/usr/bin/env python3
"""校准待办清单：把几百个音符压成几十个"值得人工看一眼"的位置。

动机（都是实测数字，不是猜的）。在 `亲亲` 的自动 metadata（470 音符）上量：

| 维度 | 实测 | 判读 |
|---|---|---|
| \\|note − 实唱 F0\\| | median **0.09** 半音，>1st 仅 5% | 音高几乎不用动 |
| `<SP>` 槽位里是浊音的比例 | **48.7%** | 一半"休止"其实在唱 |
| 全部浊音帧落在 `<SP>` 里的 | **14.9%** | 一成半歌声被标成休止 |
| 有词音符起点 → 最近浊音起点 | median **140ms**，>120ms 占 53% | 节奏错位的真实来源 |

所以清单**只报时间轴与词/休止边界问题，不报音高**。三类缺陷各自独立打分：

  SP_VOICED   `<SP>` 槽位里在唱歌 —— 漏掉的字，最该修
  ONSET       有词音符的音头离最近的浊音起点太远 —— 节奏错位
  SILENT_WORD 有词音符的窗口里几乎没有浊音 —— 多出来的字/词位空转

为什么不做自动修正：强制对齐实测修不了。FunASR 的
`speech_timestamp_prediction` 在 60s 窗上误差 median 16s（完全崩），逐行切窗
后仍 median 310ms / p90 671ms —— 比要修的 140ms 误差更大。逐字 KRC 歌词也
不行：`绿色` 的 KRC 逐字起点到最近浊音起点 median 540ms / p75 2700ms，比自动
转录**更差**。所以时间轴只能人工，工具的职责是**告诉人去哪儿看**。

歌词是唯一能自动改的：ASR 转出的词与 LRC 有 8.5% 字级差异（实测 `绿色`，还
有整段幻听重复），`--lrc` 会按 LRC 的字序列重写 metadata 的 text。

用法（SoulX venv，要读 f0/g2p）:
    # 只出清单
    SoulX-Singer/.venv/bin/python singing/svs_qc.py \\
        --meta data/songs/X/svs/target/metadata.json --top 60
    # 同时用 LRC 纠正歌词，写出新 metadata
    SoulX-Singer/.venv/bin/python singing/svs_qc.py \\
        --meta ... --lrc data/songs/X.lrc --out .../metadata_lyric.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F0_FPS = 50.0
SP = "<SP>"


def load_notes(meta_path: str):
    """metadata -> [(seg_idx, note_idx_in_seg, abs_start, dur, text, pitch, type)]
    以及每段的 f0（段内相对时间轴）。"""
    with open(meta_path, encoding="utf-8") as f:
        segs = json.load(f)
    rows = []
    f0s = []
    for si, s in enumerate(segs):
        texts = s["text"].split()
        durs = [float(x) for x in s["duration"].split()]
        pitches = [int(x) for x in s["note_pitch"].split()]
        types = [int(x) for x in s["note_type"].split()]
        f0s.append(np.array([float(x) for x in s.get("f0", "").split()]))
        t0 = s["time"][0] / 1000.0
        t = 0.0
        for k, (tx, du, pi, ty) in enumerate(zip(texts, durs, pitches, types)):
            rows.append({"seg": si, "k": k, "rel": t, "abs": t0 + t, "dur": du,
                         "text": tx, "pitch": pi, "type": ty})
            t += du
    return segs, rows, f0s


def voiced_runs(voiced: np.ndarray) -> np.ndarray:
    """浊音段的起点帧号。**只能用于乐句起始**的音头对齐。

    实测（爱笑的眼睛，379 个有词音符）：乐句起始（前一个是真休止）的音头到
    最近浊音起点 median **60ms**；句中音符 median **140ms**、>120ms 占 56%。
    句中的 140ms 不是错位，是连唱本来就没有浊音边界可对 —— 拿它当判据会把
    半首歌都标红（第一版就是这样，193 条 ONSET 里大部分是假阳）。
    句中改用 `pitch_steps`。
    """
    if len(voiced) == 0:
        return np.array([], dtype=int)
    d = np.diff(voiced.astype(np.int8), prepend=0)
    return np.flatnonzero(d == 1)


def pitch_steps(f0: np.ndarray, thresh_st: float = 0.6) -> np.ndarray:
    """F0 曲线上"换音了"的帧号 = 相邻浊音帧半音差超过 thresh 的位置。

    句中音符边界的正确参照物。实测同一首歌里"句中且前后音高不同"的 223 个
    边界，到最近 F0 台阶 median **20ms** / p75 70ms / >150ms 仅 8% —— 信号
    干净，能把真错位挑出来而不是把连唱全标红。
    """
    v = f0 > 0
    idx = np.flatnonzero(v)
    if len(idx) < 3:
        return np.array([], dtype=int)
    semi = np.zeros(len(f0))
    prev, cur = f0[idx[:-1]], f0[idx[1:]]
    semi[idx[1:]] = 12 * np.log2(np.maximum(cur, 1e-6) / np.maximum(prev, 1e-6))
    semi[~v] = 0
    return np.flatnonzero(np.abs(semi) > thresh_st)


def score(segs, rows, f0s) -> list[dict]:
    """给每个音符打三类缺陷分，返回带 issue 的行。"""
    out = []
    for si, seg in enumerate(segs):
        f0 = f0s[si]
        voiced = f0 > 0
        onsets = voiced_runs(voiced)
        steps = pitch_steps(f0)
        rr = [x for x in rows if x["seg"] == si]
        for ri, r in enumerate(rr):
            a = int(round(r["rel"] * F0_FPS))
            b = max(int(round((r["rel"] + r["dur"]) * F0_FPS)), a + 1)
            win = voiced[a:min(b, len(voiced))]
            frac = float(win.mean()) if len(win) else 0.0
            item = dict(r)
            item["voiced_frac"] = frac
            item["seg_index"] = seg["index"]
            if r["text"] == SP:
                # 休止里在唱：按"唱了多久"排序，长的优先（短的多半是尾音余韵）
                if frac > 0.35 and r["dur"] >= 0.15:
                    item["issue"] = "SP_VOICED"
                    item["severity"] = frac * min(r["dur"], 2.0)
                    out.append(item)
                continue
            # 有词音符：音头对齐。参照物按"是否乐句起始"分流，见 voiced_runs /
            # pitch_steps 的注释 —— 用错参照物会产生大量假阳。
            prev = rr[ri - 1] if ri > 0 else None
            is_init = prev is None or prev["text"] == SP
            if is_init:
                ref, kind, tol = onsets, "ONSET", 0.12
            elif prev["pitch"] != r["pitch"] and r["type"] != 3:
                # 句中且换音：对 F0 台阶
                ref, kind, tol = steps, "STEP", 0.15
            else:
                # 句中同音高（或 slur）：没有可靠参照物，跳过时间轴检查
                ref, kind, tol = None, None, None
            gap = None
            if ref is not None:
                gap = (float(np.min(np.abs(ref - a))) / F0_FPS
                       if len(ref) else 9.99)
            item["onset_gap"] = gap
            item["ref"] = kind
            if gap is not None and gap > tol:
                item["issue"] = kind
                item["severity"] = min(gap, 1.0) * (1.5 if kind == "ONSET" else 1.2)
                out.append(item)
            elif frac < 0.25 and r["dur"] >= 0.12:
                item["issue"] = "SILENT_WORD"
                item["severity"] = (0.25 - frac) * 4 * min(r["dur"], 1.0)
                out.append(item)
    out.sort(key=lambda x: -x["severity"])
    return out


# ---------------------------------------------------------------- 歌词替换

def lrc_chars(lrc_path: str) -> list[str]:
    """从 .lrc 抽出纯字序列（逐字 KRC 与行级 LRC 都支持）。

    只要**字序列**，不要时间 —— 实测 KRC 的逐字时间比自动转录更差
    （最近浊音起点 median 540ms vs 140ms），拿它定时间是退步。
    """
    chars: list[str] = []
    for ln in open(lrc_path, encoding="utf-8"):
        ln = ln.strip()
        m = re.match(r"\[(\d+:\d+[.:]\d+)\](.*)", ln)
        if not m:
            continue
        body = m.group(2)
        toks = re.findall(r"<\d+:\d+[.:]\d+>([^<]*)", body)
        text = "".join(toks) if toks else body
        for ch in text:
            if re.match(r"[\u4e00-\u9fff]", ch):
                chars.append(ch)
    return chars


def relabel_from_lrc(segs: list[dict], lrc: list[str]) -> dict:
    """用 LRC 字序列纠正 metadata 的 text。就地修改 segs，返回统计。

    对位规则：metadata 的"字槽"= `note_type == 2` 的音符（type=1 休止、
    type=3 同字延续都不消耗新字）。slur 音符跟随它所属的字。

    **不能按序号硬对位**：ASR 会多字/漏字，一处差异就把后面全推错。实测
    爱笑的眼睛 LRC 376 字 vs 379 槽，开头 ASR 多出两个哼唱「嗯嗯」（真实
    人声，LRC 里没有），硬对位让全曲平移 2 个字 —— 越改越坏。

    改用 difflib 序列对齐，按 opcode 分别处理：
      equal            保持
      replace 等长     逐字替换（这才是要修的 8.5% 错词）
      replace 不等长 / delete / insert
                       **不动**，只计数。因为增删字需要同时增删音符（音符数
                       是时间轴的一部分），那属于人工在编辑器里做的事；这里
                       悄悄改会让 text 与 duration 失配。
    """
    slots: list[tuple[dict, int]] = []
    for s in segs:
        types = [int(x) for x in s["note_type"].split()]
        for k, ty in enumerate(types):
            if ty == 2:
                slots.append((s, k))
    asr = [s["text"].split()[k] for s, k in slots]

    import difflib
    sm = difflib.SequenceMatcher(None, asr, lrc, autojunk=False)
    stat = {"slots": len(slots), "lrc": len(lrc), "fixed": 0,
            "skip_len": 0, "asr_extra": 0, "lrc_extra": 0, "samples": []}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                s, k = slots[i]
                texts = s["text"].split()
                if len(stat["samples"]) < 12:
                    stat["samples"].append((texts[k], lrc[j]))
                texts[k] = lrc[j]
                s["text"] = " ".join(texts)
                stat["fixed"] += 1
        elif tag == "replace":
            stat["skip_len"] += 1
        elif tag == "delete":
            stat["asr_extra"] += i2 - i1
        elif tag == "insert":
            stat["lrc_extra"] += j2 - j1
    # slur 跟随前一个实字
    for s in segs:
        texts = s["text"].split()
        types = [int(x) for x in s["note_type"].split()]
        for k, ty in enumerate(types):
            if ty == 3 and k > 0:
                prev = next((texts[j] for j in range(k - 1, -1, -1)
                             if texts[j] != SP), None)
                if prev:
                    texts[k] = prev
        s["text"] = " ".join(texts)
    return stat


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--top", type=int, default=60, help="清单条数上限")
    ap.add_argument("--lrc", default="", help="用它的字序列覆盖歌词")
    ap.add_argument("--out", default="", help="写出（可能已换词的）metadata")
    ap.add_argument("--csv", default="", help="清单另存 CSV，便于边听边勾")
    args = ap.parse_args()

    segs, rows, f0s = load_notes(args.meta)
    n_note = len(rows)
    n_word = sum(1 for r in rows if r["type"] == 2)

    if args.lrc:
        lrc = lrc_chars(args.lrc)
        st = relabel_from_lrc(segs, lrc)
        print(f"[歌词] LRC {st['lrc']} 字 vs 字槽 {st['slots']} 个 → "
              f"等长替换 {st['fixed']} 字")
        print(f"       未动: 长度不等的替换段 {st['skip_len']} 处、"
              f"ASR 多出 {st['asr_extra']} 字、LRC 多出 {st['lrc_extra']} 字"
              f"（增删字要连音符一起改，请在编辑器里做）")
        if st["samples"]:
            print("       改动示例: " +
                  "  ".join(f"{a}→{b}" for a, b in st["samples"]))
        # 换词后 phoneme 必须重算
        sys.path.insert(0, str(ROOT / "singing"))
        from svs_g2p import g2p_words
        for s in segs:
            s["phoneme"] = " ".join(g2p_words(s["text"].split(), s["language"]))
        # rows 的 text 要跟上换词结果（打分只用时间，但清单要显示新词）。
        # segs 是就地改的，所以从它重建，不要再读一遍磁盘上的旧文件。
        flat = [t for s in segs for t in s["text"].split()]
        if len(flat) != len(rows):
            raise RuntimeError(f"换词后音符数变了: {len(rows)} -> {len(flat)}")
        for r, tx in zip(rows, flat):
            r["text"] = tx

    issues = score(segs, rows, f0s)
    by = {}
    for it in issues:
        by[it["issue"]] = by.get(it["issue"], 0) + 1
    print(f"[清单] {n_note} 音符（其中有词 {n_word}），"
          f"命中 {len(issues)} 处: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print(f"       只列前 {min(args.top, len(issues))} 条（按严重度排序）\n")
    print(f"{'时间':>9}  {'类型':<12} {'词':<4} {'时长':>5} {'浊音':>5} "
          f"{'音头差':>6}  段")
    for it in issues[:args.top]:
        gap = it.get("onset_gap")
        print(f"{mmss(it['abs']):>9}  {it['issue']:<12} "
              f"{it['text']:<4} {it['dur']:>5.2f} {it['voiced_frac']:>5.0%} "
              f"{('%6.0f' % (gap * 1000)) if gap is not None else '     -':>6}  "
              f"{it['seg_index']}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["time", "issue", "text", "dur", "voiced_frac",
                        "onset_gap_ms", "seg", "note_k", "done"])
            for it in issues[:args.top]:
                g = it.get("onset_gap")
                w.writerow([mmss(it["abs"]), it["issue"], it["text"],
                            f"{it['dur']:.2f}", f"{it['voiced_frac']:.2f}",
                            "" if g is None else f"{g * 1000:.0f}",
                            it["seg_index"], it["k"], ""])
        print(f"\n清单 -> {args.csv}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(segs, f, ensure_ascii=False, indent=2)
        print(f"metadata -> {args.out}")


if __name__ == "__main__":
    main()
