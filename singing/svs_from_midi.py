#!/usr/bin/env python3
"""把**手弹录制**的 MIDI（音符 + 歌词，无休止无 slur 标记）转成 SVS 的 metadata。

与 `svs_edit.py import` 的区别：那个是"改回去"，需要 sidecar 提供段结构与原唱
f0；这个是"从零建"，输入只有一份独立乐谱，没有原唱可对齐 —— 用于自己配伴奏、
不复刻原唱时间轴的场景。

三件事必须自己造出来：

1. **`<SP>` 休止**：录制版只有音符，音符之间的空隙就是休止。>`--sp-gap`（默认
   0.2s）的空隙插 `<SP>`；更小的空隙并进前一个音符（弹琴的手指间隙不是休止）。
2. **slur（`note_type=3`）**：歌词是 `-` 或与前一音符同字时算延续。手弹版通常
   一个音符一个字，所以默认没有 slur —— 除非你在编辑器里填了 `-`。
3. **`f0`**：score 模式下 `gt_f0=None`，f0 不进模型，但 `data_processor.py:153`
   仍用 `min(mel2note, len(f0))` 算 `min_frame` 去裁 `mel2note`，**长度不对会把
   合成结果截断**。这里按音符的 MIDI 音高合成一条阶梯 f0（休止=0），长度
   `dur*50`，只为满足这个长度约束。

段的切法：在最长的停顿处切，每段 <= `--max-seg-sec`（默认 20s，ROSVOT 的 21.33s
硬顶在这里不适用，但 whisper 编码器 30s、显存与合成质量都偏好短段）。

用法:
  cd SoulX-Singer && MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
      ../singing/svs_from_midi.py \
      --mid ../data/songs/X/svs/edit/vocal-midi-record130.mid \
      --out ../data/songs/X/svs/target/metadata_rec.json \
      --wav-dir ../data/songs/X/svs_rec/target/long_cut_wavs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

F0_FPS = 50.0
SP = "<SP>"


def read_midi(path: str) -> list[dict]:
    """-> [{start, dur, pitch, text}]，绝对秒。歌词按 tick 就近配给音符。"""
    mid = mido.MidiFile(path)
    tempo = 500000
    raw: list[dict] = []
    lyrics: list[tuple[int, str]] = []
    for track in mid.tracks:
        t = 0
        active: dict[tuple[int, int], list[int]] = {}
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "lyrics":
                tx = msg.text
                try:                    # mido 用 latin-1 读 meta text
                    tx = tx.encode("latin1").decode("utf-8")
                except Exception:
                    pass
                lyrics.append((t, tx))
            elif msg.type in ("note_on", "note_off"):
                key = (msg.channel, msg.note)
                if msg.type == "note_on" and msg.velocity > 0:
                    active.setdefault(key, []).append(t)
                elif active.get(key):
                    st = active[key].pop(0)
                    raw.append({"a": st, "b": t, "pitch": msg.note})
    spb = tempo / 1e6 / mid.ticks_per_beat
    raw.sort(key=lambda n: (n["a"], n["pitch"]))
    lyrics.sort(key=lambda x: x[0])
    # 歌词配给：**按 tick 分桶后按顺序取**，不要用"全局最近"贪心。
    # 实测录制版每条 lyrics 的 tick 与它的 note_on 完全相等（182:182 严格 1:1），
    # 但"最近优先"会让某个词被邻近音符抢走，留下 5 个空音符 —— 看起来像漏词，
    # 实际是匹配算法的锅。同 tick 多条歌词时按序消费。
    by_tick: dict[int, list[str]] = {}
    for lt, tx in lyrics:
        by_tick.setdefault(lt, []).append(tx)
    tol = max(1, mid.ticks_per_beat // 32)
    out = []
    for n in raw:
        text = ""
        # **精确 tick 优先**。只按"距离最近"扫会出错：两个候选 tick 与当前音符
        # 等距时，先被取走的可能是本该属于别人的那条，于是自己的桶被别人掏空，
        # 最后凭空出现 5 个空词音符（实测就是这么来的，而 tick 集合本是严格 1:1）。
        if by_tick.get(n["a"]):
            text = by_tick[n["a"]].pop(0)
        else:
            for cand in sorted(by_tick, key=lambda k: (abs(k - n["a"]), k)):
                if abs(cand - n["a"]) > tol:
                    break
                if by_tick[cand]:
                    text = by_tick[cand].pop(0)
                    break
        out.append({"start": n["a"] * spb, "dur": (n["b"] - n["a"]) * spb,
                    "pitch": n["pitch"], "text": text})
    return out


def add_rests(notes: list[dict], sp_gap: float = 0.2) -> list[dict]:
    """空隙 > sp_gap 的插 <SP>，更小的并进前一个音符（段内必须无缝）。"""
    out: list[dict] = []
    for i, n in enumerate(notes):
        n = dict(n)
        if out:
            gap = n["start"] - (out[-1]["start"] + out[-1]["dur"])
            if gap > sp_gap:
                out.append({"start": out[-1]["start"] + out[-1]["dur"],
                            "dur": gap, "pitch": 0, "text": SP})
            elif gap > 0:
                out[-1]["dur"] += gap          # 手指间隙，不是休止
            elif gap < 0:
                n["start"] = out[-1]["start"] + out[-1]["dur"]
                n["dur"] = max(n["dur"] + gap, 0.02)
        out.append(n)
    return out


def split_segments(notes: list[dict], max_sec: float = 20.0) -> list[list[dict]]:
    """在最长的 <SP> 处切段。段太长会吃显存、也超出 whisper 的 30s 编码窗。"""
    segs, cur = [], []
    for n in notes:
        cur.append(n)
        span = cur[-1]["start"] + cur[-1]["dur"] - cur[0]["start"]
        if span >= max_sec:
            # 回头找本段最长的 <SP> 作为切点（至少留 2 个音符）
            best, bi = 0.0, -1
            for i in range(len(cur) - 2, 0, -1):
                if cur[i]["text"] == SP and cur[i]["dur"] > best:
                    best, bi = cur[i]["dur"], i
            if bi > 0:
                segs.append(cur[:bi])          # <SP> 本身丢掉 = 段间真实停顿
                cur = cur[bi + 1:]
    if cur:
        segs.append(cur)
    return [s for s in segs if any(x["text"] not in (SP, "") for x in s)]


def resolve_slurs(notes: list[dict]) -> tuple[list[dict], int]:
    """在**切段之前**把空歌词解析成前一个字的延续，并打上 slur 标记。

    必须先于 split_segments：切段后再解析，恰好落在段首的延续音符找不到可继承的
    字，只能退化成「啊」（实测 88.42s 那个拖腔就是这样变成新字的）。
    """
    out: list[dict] = []
    n_slur = 0
    for n in notes:
        n = dict(n)
        if n["text"] == SP:
            n["slur"] = False
        elif n["text"].strip() in ("", "-"):
            prev = next((x for x in reversed(out)
                         if x["text"] != SP and not x.get("slur")), None)
            if prev is None:
                n["text"], n["slur"] = "啊", False
            else:
                n["text"], n["slur"] = prev["text"], True
                n_slur += 1
        else:
            n["slur"] = False
        out.append(n)
    return out, n_slur


def synth_f0(seg: list[dict]) -> np.ndarray:
    """按音符音高造阶梯 f0（休止=0）。只为满足 min_frame 的长度约束。"""
    total = sum(x["dur"] for x in seg)
    f0 = np.zeros(max(int(round(total * F0_FPS)), 1))
    t = 0.0
    for n in seg:
        a = int(round(t * F0_FPS))
        b = min(int(round((t + n["dur"]) * F0_FPS)), len(f0))
        if n["text"] != SP and n["pitch"] > 0:
            f0[a:b] = 440.0 * 2 ** ((n["pitch"] - 69) / 12.0)
        t += n["dur"]
    return f0


def build(notes: list[dict], language: str, max_sec: float,
          wav_dir: str | None) -> list[dict]:
    """-> metadata。同时（可选）写出每段的静音占位 wav + _f0.npy。

    为什么要占位 wav：`svs_song.py` 的合成路径要读 `long_cut_wavs/<index>.wav`
    与同名 `_f0.npy` 取时长/帧数。score 模式不看内容（`gt_f0=None`），所以静音
    即可 —— 但**长度必须与 metadata 的 duration 和一致**，否则 min_frame 裁错。
    """
    import soundfile as sf
    from svs_g2p import g2p_words

    segs = split_segments(notes, max_sec)
    out = []
    t_cursor = 0.0
    for si, seg in enumerate(segs):
        t0 = seg[0]["start"]
        durs = [round(x["dur"], 2) for x in seg]
        # **没填词的音符是上一个字的延续（拖腔），不是休止。** 手弹时一个字拖几个
        # 音符只会在第一个音符上填词，后面的空着；当成休止会把拖腔切断，还会凭空
        # 多出一次换气。真正的休止由 add_rests 按空隙插入，text 明确是 <SP>。
        texts = [x["text"] for x in seg]
        pitches = [x["pitch"] for x in seg]
        types = []
        for i, x in enumerate(seg):
            if texts[i] == SP:
                types.append(1)
                pitches[i] = 0
            else:
                types.append(3 if x.get("slur") else 2)
        f0 = synth_f0([{**x, "text": tx} for x, tx in zip(seg, texts)])
        span_ms = int(round(sum(durs) * 1000))
        index = f"rec_{int(round(t0*1000))}_{int(round(t0*1000))+span_ms}"
        if wav_dir:
            Path(wav_dir).mkdir(parents=True, exist_ok=True)
            sr = 24000
            sf.write(str(Path(wav_dir) / f"{index}.wav"),
                     np.zeros(int(sum(durs) * sr), dtype=np.float32), sr)
            np.save(str(Path(wav_dir) / f"{index}_f0.npy"), f0)
        out.append({
            "index": index, "language": language,
            "time": [int(round(t0 * 1000)), int(round(t0 * 1000)) + span_ms],
            "duration": " ".join(f"{d:.2f}" for d in durs),
            "text": " ".join(texts),
            "phoneme": " ".join(g2p_words(texts, language)),
            "note_pitch": " ".join(str(p) for p in pitches),
            "note_type": " ".join(str(t) for t in types),
            "f0": " ".join(f"{x:.1f}" for x in f0),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav-dir", default="", help="写静音占位 wav + _f0.npy")
    ap.add_argument("--sp-gap", type=float, default=0.2,
                    help="大于它的空隙才算休止（默认 0.2s）")
    ap.add_argument("--max-seg-sec", type=float, default=20.0)
    ap.add_argument("--language", default="Mandarin")
    args = ap.parse_args()

    notes = read_midi(args.mid)
    print(f"[midi] {len(notes)} 音符, {notes[0]['start']:.2f}-"
          f"{notes[-1]['start']+notes[-1]['dur']:.2f}s")
    notes, n_slur = resolve_slurs(notes)
    if n_slur:
        print(f"[slur] {n_slur} 个空歌词音符解析成前一个字的延续（拖腔）")
    notes = add_rests(notes, args.sp_gap)
    nsp = sum(1 for n in notes if n["text"] == SP)
    print(f"[rest] 插入 {nsp} 个 <SP>（空隙 > {args.sp_gap}s）")
    meta = build(notes, args.language, args.max_seg_sec, args.wav_dir or None)
    nn = sum(len(s["duration"].split()) for s in meta)
    print(f"[seg ] {len(meta)} 段 / {nn} 音符, 每段 "
          + ", ".join(f"{(s['time'][1]-s['time'][0])/1000:.1f}s" for s in meta))
    Path(args.out).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
