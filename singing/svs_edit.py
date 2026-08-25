#!/usr/bin/env python3
"""SoulX metadata <-> MIDI 无损往返桥，服务手工歌词/MIDI 校准。

为什么不直接用上游 `preprocess/tools/midi_parser.py`：它是有损的。实测
`亲亲` 的 metadata 走一圈 `meta2midi -> midi2meta`：

  段数 12 -> 9、音符 470 -> 476、总时长 216.6s -> 236.8s、<SP> 82 -> 98，
  52 个 slur 里 **10 个歌词被写成 `<SP>`**，`f0` 字段整个丢失。

三个根因，本文件逐个绕开：
  1. 段边界靠"尾部 <SP> > 2s"重新猜（`midi_parser.MAX_GAP_SEC`），与原
     metadata 的 `time[]` 无关 —— 段结构一变，`cli/inference.py` 拼回时间轴
     的依据就变了。这里把 `time[]` 存进 sidecar，导入时原样恢复。
  2. `midi2notes` 给 `note_type=3`（slur，同字跨多音符）回填歌词时取
     `raw_notes[idx-1]["lyric"]`，而 slur 前一个音符可能正是 `<SP>`
     （原 metadata 里 52 个 slur 有 10 个如此），于是歌词变成 `<SP>`。
     这里改为**向前找最近的非 <SP> 歌词**。
  3. MIDI 表达不了 `f0` 数组、`language`、以及"`<SP>` 上带非零音高"这件事
     （原 metadata 里 82 个 <SP> 有 34 个带音高，上游 demo 也有这种写法）。
     全部进 sidecar，按音符序号对齐。

sidecar 的设计取舍：**按音符序号（而非时间）索引**，因为校准的全部动作就是
改时间；时间键会在第一次拖动后失配。音符的增删由 `svs_import` 用
"MIDI 里的音符顺序" 重新分配，sidecar 只保留段边界与每段 f0。

f0 字段与 score 模式的关系：`--control score` 下 `soulxsinger.py:121` 把
`gt_f0=None`，f0 不进模型；但 `data_processor.process` 仍用它算 `min_frame`
去裁 `mel2note`（`data_processor.py:153-155`），所以**不能丢，也不能长度乱**。
时长被编辑后 f0 要按新时长重采样，见 `_resample_f0`。

用法:
    # 导出给编辑器（产出 <dir>/edit/vocal.mid + edit_sidecar.json）
    python3 singing/svs_edit.py export --meta data/songs/X/svs/target/metadata.json
    # 编辑器改完导出 .mid，再导回 metadata
    python3 singing/svs_edit.py import --mid data/songs/X/svs/edit/vocal_edited.mid \
        --sidecar data/songs/X/svs/edit/edit_sidecar.json \
        --out data/songs/X/svs/target/metadata_edited.json
    # 自检：不编辑时往返必须逐字段等价
    python3 singing/svs_edit.py verify --meta data/songs/X/svs/target/metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOULX = ROOT / "SoulX-Singer"

# MIDI 量化：ticks_per_beat 与 tempo 决定时间分辨率。上游用 500/500000
# (= 1ms 一个 tick)；这里放大到 2000 tick/beat @ 120bpm = 0.25ms，让 0.01s
# 的编辑器 snap 精度往返无累积误差。
TICKS_PER_BEAT = 2000
TEMPO = 500000  # microseconds per beat (120 BPM)
SEC_PER_TICK = TEMPO / 1e6 / TICKS_PER_BEAT
VELOCITY = 64
F0_FPS = 50.0  # metadata 的 f0 网格（实测 hop=480 @ 24kHz -> 50Hz）
MIN_FRAME_SEC = 1.0 / F0_FPS  # 20ms：比它还小的重叠在 metadata 里表达不出来

SP = "<SP>"
# 编辑器里 slur 显示成 "-"（上游 midi_parser 的约定，保持一致，用户在
# LyricTable 里看到 "-" 就知道这是同一个字的延续音符）
SLUR_MARK = "-"


@dataclass
class Note:
    """一个音符。start/dur 单位秒，绝对时间轴（与 metadata 的 time[0] 同基准）。"""
    start: float
    dur: float
    text: str
    pitch: int
    ntype: int

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class Doc:
    """一首歌的完整可编辑表示 = 音符序列 + 段结构 + 每段 f0。"""
    notes: list[Note] = field(default_factory=list)
    # 每段: {"index","language","time":[ms,ms],"n_notes",...}
    segs: list[dict] = field(default_factory=list)
    # 每段的 f0 数组（float list），与 segs 一一对应
    f0: list[list[float]] = field(default_factory=list)


def sec_to_ticks(s: float) -> int:
    return int(round(s / SEC_PER_TICK))


def ticks_to_sec(t: int) -> float:
    return t * SEC_PER_TICK


# ---------------------------------------------------------------- metadata IO

def meta_to_doc(meta_path: str | Path) -> Doc:
    """读 metadata.json -> Doc。音符时间轴用每段 time[0] 作起点累加 duration。

    注意 metadata 的 `duration` 之和与 `time` 跨度实测差 ≤11ms（浮点round），
    以 time[0] 为锚、duration 累加是权威做法 —— `data_processor.preprocess`
    也只用 duration，不看 time[1]。
    """
    with open(meta_path, encoding="utf-8") as f:
        segs_raw = json.load(f)
    if not isinstance(segs_raw, list) or not segs_raw:
        raise ValueError(f"{meta_path} 不是非空的段列表")

    doc = Doc()
    for s in segs_raw:
        texts = s["text"].split()
        durs = [float(x) for x in s["duration"].split()]
        pitches = [int(x) for x in s["note_pitch"].split()]
        types = [int(x) for x in s["note_type"].split()]
        n = len(texts)
        if not (len(durs) == len(pitches) == len(types) == n):
            raise ValueError(
                f"段 {s.get('index')} 字段长度不一致: "
                f"text={n} duration={len(durs)} note_pitch={len(pitches)} "
                f"note_type={len(types)}")
        t = s["time"][0] / 1000.0
        for tx, du, pi, ty in zip(texts, durs, pitches, types):
            doc.notes.append(Note(start=t, dur=du, text=tx, pitch=pi, ntype=ty))
            t += du
        f0 = [float(x) for x in s.get("f0", "").split()]
        # `duration` 只有 2 位小数，所以 sum(duration) 与 time 跨度实测差
        # ≤11ms（不是 bug，是原文件自身的量化残差）。把这个残差记下来而不是
        # 重算，否则每次往返 time[1] 都会掉 10ms、f0 长度跟着漂。
        doc.segs.append({
            "index": s["index"],
            "language": s.get("language", "Mandarin"),
            "time": list(s["time"]),
            "n_notes": n,
            "slack_ms": s["time"][1] - (s["time"][0] + int(round(sum(durs) * 1000))),
            "dur_sum": sum(durs),
            "f0_n": len(f0),
        })
        doc.f0.append(f0)
    return doc


def doc_to_meta(doc: Doc) -> list[dict]:
    """Doc -> metadata 段列表。phoneme 由 g2p 重算（歌词可能被改过）。"""
    from svs_g2p import g2p_words  # 延迟导入：需要 SoulX venv 的 g2pM

    out: list[dict] = []
    i = 0
    for si, seg in enumerate(doc.segs):
        n = seg["n_notes"]
        notes = doc.notes[i:i + n]
        i += n
        if not notes:
            continue
        texts = [x.text for x in notes]
        durs = [x.dur for x in notes]
        # time[0] 跟随第一个音符（用户可能把整段音头挪了）。time[1] 用
        # "t0 + 时长和 + 原始 slack" 而不是 notes[-1].end，把原文件的 2 位
        # 小数量化残差原样带回去，保证不编辑时逐字段等价。
        t0 = int(round(notes[0].start * 1000))
        new_dur = sum(durs)
        t1 = t0 + int(round(new_dur * 1000)) + int(seg.get("slack_ms", 0))
        # **f0 的时间基准是段的 t0，所以 t0 变了必须先切掉头部，再谈重采样。**
        # 只做重采样是错的：删掉段首的 <SP>（很常见 —— 幻觉出来的前奏休止）会让
        # t0 右移，若把 f0 原样压缩到新时长，f0 就与音频错位（实测 t0 移 1.59s 后
        # 段内音符落到静音上，QC 凭空多出 16 条 ONSET/STEP，看起来像"手工改坏了"）。
        f0_src = doc.f0[si]
        old_dur = seg.get("dur_sum")
        old_n = seg.get("f0_n")
        shift_ms = t0 - int(seg["t0_ms"])
        if shift_ms:
            drop = int(round(shift_ms / 1000.0 * F0_FPS))
            if drop > 0:
                f0_src = f0_src[drop:]
                if old_n:
                    old_n = max(old_n - drop, 1)
            elif drop < 0:      # 段首往左扩了，前面补清音
                f0_src = [0.0] * (-drop) + list(f0_src)
                if old_n:
                    old_n = old_n - drop
            if old_dur is not None:
                old_dur = old_dur - shift_ms / 1000.0
        f0 = _resample_f0(f0_src, new_dur, old_dur=old_dur, old_n=old_n)
        out.append({
            "index": seg["index"],
            "language": seg["language"],
            "time": [t0, t1],
            "duration": " ".join(f"{x:.2f}" for x in durs),
            "text": " ".join(texts),
            "phoneme": " ".join(g2p_words(texts, seg["language"])),
            "note_pitch": " ".join(str(x.pitch) for x in notes),
            "note_type": " ".join(str(x.ntype) for x in notes),
            "f0": " ".join(f"{x:.1f}" for x in f0),
        })
    if i != len(doc.notes):
        raise ValueError(f"段 n_notes 之和 {i} != 音符数 {len(doc.notes)}")
    return out


def _resample_f0(f0: list[float], new_dur_sec: float,
                 old_dur: float | None = None,
                 old_n: int | None = None) -> list[float]:
    """段时长被编辑后，把 f0 网格重采样到新长度（最近邻，保 0 = 清音）。

    线性插值会在 voiced/unvoiced 边界造出 0 与真实 F0 的中间值（如 130Hz），
    那是个不存在的音高；最近邻不会。f0 在 score 模式下只用于算 min_frame，
    精度要求低，但**长度必须跟着时长走**，否则 mel2note 被裁错。

    `old_dur`/`old_n` 给定时按**相对变化**换算长度而不是 `dur*fps`：原始
    f0 是从切好的 wav 提的，长度包含 `END_EXTENSION_SEC` 之类的尾部余量，
    实测比 `dur*50` 多 1-3 帧。不编辑时必须原样返回。
    """
    if not f0:
        return []
    if old_dur and old_n and old_dur > 0:
        if abs(new_dur_sec - old_dur) < 5e-3:  # 未编辑（<5ms 视为未动）
            return list(f0)
        n_new = max(1, int(round(old_n * new_dur_sec / old_dur)))
    else:
        n_new = int(round(new_dur_sec * F0_FPS))
    if n_new <= 0:
        return []
    if n_new == len(f0):
        return list(f0)
    n_old = len(f0)
    return [f0[min(int(k * n_old / n_new), n_old - 1)] for k in range(n_new)]


# -------------------------------------------------------------------- MIDI IO

def doc_to_midi(doc: Doc, midi_path: str | Path) -> None:
    """Doc -> MIDI。每个音符一个 note_on/off + 一条 lyrics meta。

    `<SP>` 也写成真音符（pitch 用 sidecar 记的原值，0 时落到 MIDI note 0），
    这样编辑器里休止是可见、可拖的块 —— 上游的 <SP> 处理是"靠 gap 反推"，
    用户没法直接调整休止边界，而实测「<SP> 槽位 48.7% 其实在唱」正是要改的
    主要缺陷，必须让它可见。
    """
    import mido

    events: list[tuple[int, int, object]] = []
    for n in doc.notes:
        a = sec_to_ticks(n.start)
        b = sec_to_ticks(n.end)
        if b <= a:
            b = a + 1
        # 编辑器（和上游）把 slur 显示成 "-"；<SP> 保持原样以便区分休止
        label = SLUR_MARK if n.ntype == 3 else n.text
        try:
            label = label.encode("utf-8").decode("latin1")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        # 同 tick 的排序: note_off(0) < lyrics(1) < note_on(2)，与
        # midi_editor/src/lib/midi.ts 的 _sortKey 约定一致
        events.append((b, 0, mido.Message("note_off", note=max(n.pitch, 0),
                                          velocity=0, time=0)))
        events.append((a, 1, mido.MetaMessage("lyrics", text=label, time=0)))
        events.append((a, 2, mido.Message("note_on", note=max(n.pitch, 0),
                                          velocity=VELOCITY, time=0)))
    events.sort(key=lambda x: (x[0], x[1]))

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4,
                               time=0))
    last = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last)
        tr.append(msg)
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    os.makedirs(os.path.dirname(str(midi_path)) or ".", exist_ok=True)
    mid.save(str(midi_path))


def midi_to_notes(midi_path: str | Path) -> list[Note]:
    """MIDI -> Note 列表。歌词按 tick 就近配给 note_on，slur 回填最近的实字。

    与上游 `midi2notes` 的三处差别（都是实测踩到的）:
      * slur("-") 回填歌词时**跳过 <SP>**，上游取 idx-1 会把 <SP> 当歌词，
        实测 52 个 slur 里 10 个被这样毁掉。
      * 不按 gap 插 <SP>、不切段 —— 段结构来自 sidecar。
      * 不做重叠 trim（编辑器自带"消除重叠"，且 trim 会静默改用户的时长）；
        重叠在这里报错，让用户回编辑器点一下那个按钮。
    """
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat
    tempo = TEMPO
    raw: list[dict] = []
    lyrics: list[tuple[int, str]] = []
    for track in mid.tracks:
        abs_t = 0
        # 每个 (channel,pitch) 用**队列**而不是单值。用单值（上游 midi2notes
        # 的写法）在"同音高两个音符重叠"时会被后一个 note_on 覆盖，于是前一个
        # 音符**静默消失**，而且残留的音符互不重叠、连重叠检查都骗过去。
        # 实测触发条件很平常：把某段尾音拖长越过下一段的 <SP>（两者 pitch 都
        # 是 0）就会丢一个音符、段时长反而变短。
        active: dict[tuple[int, int], list[int]] = {}
        for msg in track:
            abs_t += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "lyrics":
                text = msg.text
                try:
                    text = text.encode("latin1").decode("utf-8")
                except Exception:
                    pass
                lyrics.append((abs_t, text))
            elif msg.type in ("note_on", "note_off"):
                key = (msg.channel, msg.note)
                on = msg.type == "note_on" and msg.velocity > 0
                if on:
                    active.setdefault(key, []).append(abs_t)
                elif active.get(key):
                    st = active[key].pop(0)  # FIFO：先开的先关
                    raw.append({"pitch": msg.note, "a": st, "b": abs_t})
    dangling = sum(len(v) for v in active.values())
    if dangling:
        raise ValueError(f"{midi_path} 里有 {dangling} 个 note_on 没有对应的 "
                         "note_off，文件可能损坏")
    if not raw:
        raise ValueError(f"{midi_path} 里没有音符")

    raw.sort(key=lambda n: (n["a"], n["pitch"]))
    lyrics.sort(key=lambda x: x[0])

    # 歌词配给：同 tick 可能有多条（同 tick 多音符），按序消费
    by_tick: dict[int, list[str]] = {}
    for t, tx in lyrics:
        by_tick.setdefault(t, []).append(tx)
    used: dict[int, int] = {}
    tol = max(1, tpb // 100)

    def take(tick: int) -> str:
        for cand in sorted(by_tick, key=lambda k: abs(k - tick)):
            if abs(cand - tick) > tol:
                break
            k = used.get(cand, 0)
            if k < len(by_tick[cand]):
                used[cand] = k + 1
                return by_tick[cand][k]
        return ""

    spt = tempo / 1e6 / tpb
    notes: list[Note] = []
    unlabeled: list[float] = []   # 有音高但没填词的音符（多半是加了忘填）
    for n in raw:
        label = take(n["a"])
        start = n["a"] * spt
        dur = (n["b"] - n["a"]) * spt
        if dur <= 0:
            continue
        if label == SLUR_MARK:
            prev = next((x.text for x in reversed(notes) if x.text != SP), SLUR_MARK)
            notes.append(Note(start, dur, prev, n["pitch"], 3))
        elif label == SP or label == "":
            # 空歌词只可能是用户新加的音符没填词；当作休止更安全（唱"啦"会
            # 在成品里凭空多出一个字，而漏一个休止只是节奏松一点）。
            # 但**必须提醒**：用户加音符通常正是为了补一个漏掉的字，静默当成
            # 休止会让这次编辑白做且毫无提示。
            if label == "" and n["pitch"] > 0:
                unlabeled.append(start)
            notes.append(Note(start, dur, SP, n["pitch"], 1))
        else:
            notes.append(Note(start, dur, label, n["pitch"], 2))

    # 亚帧重叠自动裁掉，不要为它把用户赶回编辑器。来源不是用户拖错：编辑器给
    # 极短音符一个固定像素宽度（实测导出成 31ms），把**邻居**往左挪一格就会撞上
    # 它，重叠量恒为 1.2ms / 11.2ms 这种量级。F0 网格是 50fps = 20ms，比一帧还小
    # 的重叠在 metadata 里根本表达不出来，裁掉是无损的。
    trimmed = 0
    for i in range(len(notes) - 1):
        gap = notes[i].end - notes[i + 1].start
        if 1e-6 < gap <= MIN_FRAME_SEC:
            notes[i].dur = max(notes[i].dur - gap, 1e-4)
            trimmed += 1
    if trimmed:
        print(f"[info] {trimmed} 处亚帧重叠（<{MIN_FRAME_SEC*1000:.0f}ms）已自动裁齐",
              file=sys.stderr)

    over = [(i, notes[i].end, notes[i + 1].start)
            for i in range(len(notes) - 1)
            if notes[i].end > notes[i + 1].start + 1e-6]
    if over:
        i, e, s = over[0]
        raise ValueError(
            f"有 {len(over)} 处音符重叠（超过 {MIN_FRAME_SEC*1000:.0f}ms），"
            f"首处在音符 #{i} (结束 {e:.3f}s > 下一个开始 {s:.3f}s)。"
            "请在编辑器里点「消除重叠」后重新导出。")
    if unlabeled:
        ts = ", ".join(f"{t // 60:.0f}:{t % 60:05.2f}" for t in unlabeled[:6])
        print(f"[warn] {len(unlabeled)} 个音符有音高但没填歌词，已当成休止 <SP> "
              f"处理（若本意是补字，请回编辑器双击音符填词）: {ts}"
              f"{' ...' if len(unlabeled) > 6 else ''}", file=sys.stderr)
    return notes


def fill_gaps(notes: list[Note], min_gap: float = 0.02) -> tuple[list[Note], int]:
    """把音符间的时间空隙补成 `<SP>`，让序列重新成为**无缝链**。

    这一步是必需的，原因在 metadata 的时间表示上：**段内音符是首尾相接的
    duration 链**（实测原始文件段内 0/565 处有 >5ms 间隙），`time[0]` 之后每个
    音符的起点都靠累加 duration 得到 —— 格式里没有"绝对起点"字段。

    所以用户在编辑器里删掉一个音符后若不补空隙，后面所有音符会**整体前移**：
    实测删 1 个音符导致 10 个后续音符起点漂移、最大 0.28s，等于毁掉一句的节奏，
    而且不报任何错。这是"编辑器支持增删音符"与"metadata 无法表达空洞"之间的
    真实落差，必须在导入侧补上。

    补成 `<SP>` 而不是拉长前一个音符：删掉一个字的语义是"这里不该有字"，补静音
    才对；拉长前一个音符会把它的元音拖长，听起来是另一种错误。
    """
    if not notes:
        return notes, 0
    out = [notes[0]]
    filled = 0
    for n in notes[1:]:
        gap = n.start - out[-1].end
        if gap > min_gap:
            out.append(Note(start=out[-1].end, dur=gap, text=SP, pitch=0, ntype=1))
            filled += 1
        elif gap > 0:
            # 亚阈值缝隙（编辑器 0.01s snap 的残留）：并进前一个音符保持严格无缝
            p = out[-1]
            out[-1] = Note(p.start, p.dur + gap, p.text, p.pitch, p.ntype)
        out.append(n)
    return out, filled


# ------------------------------------------------------------------- sidecar

def write_sidecar(doc: Doc, path: str | Path, meta_src: str) -> None:
    """存 MIDI 装不下的东西：段结构 + 每段 f0 + 段边界时间（秒，用于重分段）。

    段边界用**秒**而不是音符序号，因为用户可能增删音符；导入时按"音符起点
    落在哪个段的时间窗内"重新分配，比序号鲁棒。窗口用原 metadata 的
    time[0]，边界取相邻段的中点，落在间隙里的新音符归给更近的那段。
    """
    bounds = []
    for seg in doc.segs:
        bounds.append({"index": seg["index"], "language": seg["language"],
                       "t0": seg["time"][0] / 1000.0,
                       "t1": seg["time"][1] / 1000.0,
                       # 原文件的量化残差与 f0 真实帧数，见 meta_to_doc /
                       # _resample_f0 的注释：不带上就无法保证往返等价
                       "slack_ms": seg.get("slack_ms", 0),
                       "dur_sum": round(seg.get("dur_sum", 0.0), 4),
                       "f0_n": seg.get("f0_n", 0)})
    data = {
        "source_metadata": meta_src,
        "ticks_per_beat": TICKS_PER_BEAT,
        "tempo": TEMPO,
        "f0_fps": F0_FPS,
        "n_notes": len(doc.notes),
        "segments": bounds,
        # f0 存成字符串，和 metadata 同格式，避免 JSON 里几十万个 float
        "f0": [" ".join(f"{x:.1f}" for x in seg) for seg in doc.f0],
    }
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def read_sidecar(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assign_segments(notes: list[Note], side: dict) -> Doc:
    """把编辑后的音符按 sidecar 的段时间窗重新分段。

    分段规则：段 k 的归属区间是 [mid(k-1,k), mid(k,k+1))，mid 取前段 t1 与
    后段 t0 的中点。这样即使用户把某个音头挪过了原边界，它仍归属"意图上"的
    那一段。首段左边界 = -inf，末段右边界 = +inf，保证不丢音符。
    """
    segs = side["segments"]
    if not segs:
        raise ValueError("sidecar 里没有段信息")
    cuts: list[float] = []
    for k in range(len(segs) - 1):
        cuts.append((segs[k]["t1"] + segs[k + 1]["t0"]) / 2.0)

    doc = Doc()
    buckets: list[list[Note]] = [[] for _ in segs]
    for n in notes:
        k = 0
        while k < len(cuts) and n.start >= cuts[k]:
            k += 1
        buckets[k].append(n)

    f0_str = side.get("f0", [])
    empty = [i for i, b in enumerate(buckets) if not b]
    if empty:
        names = ", ".join(segs[i]["index"] for i in empty)
        print(f"[warn] {len(empty)} 个段在编辑后没有音符，已丢弃: {names}",
              file=sys.stderr)
    total_filled = 0
    for k, b in enumerate(buckets):
        if not b:
            continue
        # **按段**补空隙：段内必须无缝（见 fill_gaps 的说明），但段与段之间的
        # 空隙是真实的间奏/停顿，绝不能填 —— 那会让 SVS 在间奏里唱静音音符。
        b, nf = fill_gaps(b)
        total_filled += nf
        doc.notes.extend(b)
        doc.segs.append({
            "index": segs[k]["index"],
            "language": segs[k].get("language", "Mandarin"),
            "time": [int(round(b[0].start * 1000)), int(round(b[-1].end * 1000))],
            "n_notes": len(b),
            "slack_ms": segs[k].get("slack_ms", 0),
            "dur_sum": segs[k].get("dur_sum"),
            "f0_n": segs[k].get("f0_n"),
            # 原始段起点：f0 的时间基准。用户挪动段首音符后要靠它算 f0 该切掉多少
            "t0_ms": int(round(segs[k]["t0"] * 1000)),
        })
        raw = f0_str[k] if k < len(f0_str) else ""
        doc.f0.append([float(x) for x in raw.split()])
    if total_filled:
        print(f"[import] 补了 {total_filled} 个 <SP> 填删除/移动留下的空隙"
              f"（metadata 段内必须无缝，否则后续音符会整体前移）")
    return doc


# ------------------------------------------------------------------------ CLI

def write_f0_curve(doc: Doc, path: str | Path) -> int:
    """给编辑器画底图用的 F0 曲线（绝对时间轴，半音刻度）。

    编辑器的钢琴卷帘纵轴是 MIDI 音高，所以这里直接输出 **MIDI 浮点值**而不是
    Hz —— 前端不必知道 440Hz 换算。清音帧输出 null，让曲线断开而不是掉到 0
    （画成 0 会在低音区拉出一条假线）。

    为什么要这个：编辑器现在只有波形，肉眼没法判断音符该落在哪个音高格。有了
    F0 底图，音头对齐和音高核对都变成"把方块贴到曲线上"。
    """
    pts: list[float | None] = []
    t0 = doc.segs[0]["time"][0] / 1000.0 if doc.segs else 0.0
    # 按段拼到统一时间轴，段间空隙填 null
    grid: dict[int, float] = {}
    for si, seg in enumerate(doc.segs):
        base = seg["time"][0] / 1000.0
        for i, hz in enumerate(doc.f0[si]):
            if hz <= 0:
                continue
            k = int(round((base + i / F0_FPS - t0) * F0_FPS))
            grid[k] = 69.0 + 12.0 * (np_log2(hz / 440.0))
    n = (max(grid) + 1) if grid else 0
    pts = [round(grid[k], 3) if k in grid else None for k in range(n)]
    data = {"t0": round(t0, 3), "fps": F0_FPS, "unit": "midi", "midi": pts}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    return sum(1 for x in pts if x is not None)


def np_log2(x: float) -> float:
    import math
    return math.log2(x)


def cmd_export(args) -> None:
    doc = meta_to_doc(args.meta)
    out_dir = Path(args.out_dir or (Path(args.meta).parent.parent / "edit"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = out_dir / "vocal.mid"
    side = out_dir / "edit_sidecar.json"
    doc_to_midi(doc, mid)
    write_sidecar(doc, side, str(args.meta))
    f0j = out_dir / "f0_curve.json"
    nv = write_f0_curve(doc, f0j)
    print(f"  F0 底图 -> {f0j}（{nv} 个浊音帧）")
    span = (doc.notes[0].start, doc.notes[-1].end)
    n_sp = sum(1 for n in doc.notes if n.text == SP)
    n_slur = sum(1 for n in doc.notes if n.ntype == 3)
    print(f"[export] {len(doc.segs)} 段 / {len(doc.notes)} 音符"
          f"（<SP> {n_sp}，slur {n_slur}），{span[0]:.1f}-{span[1]:.1f}s")
    print(f"  MIDI    -> {mid}")
    print(f"  sidecar -> {side}")


def cmd_import(args) -> None:
    side = read_sidecar(args.sidecar)
    notes = midi_to_notes(args.mid)
    doc = assign_segments(notes, side)
    metas = doc_to_meta(doc)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    n0 = side.get("n_notes", 0)
    print(f"[import] {len(metas)} 段 / {len(doc.notes)} 音符"
          f"（原 {n0}，{len(doc.notes) - n0:+d}） -> {out}")


def _canon(metas: list[dict]) -> list[dict]:
    """比较用的规范化：duration/f0 统一到导出时的小数位，避免格式噪声。"""
    out = []
    for s in metas:
        d = dict(s)
        d["duration"] = " ".join(f"{float(x):.2f}" for x in s["duration"].split())
        d["f0"] = " ".join(f"{float(x):.1f}" for x in s.get("f0", "").split())
        out.append(d)
    return out


def cmd_verify(args) -> None:
    """往返自检：不做编辑时必须逐字段等价。这是本文件存在的理由。"""
    import tempfile

    src = _canon(json.load(open(args.meta, encoding="utf-8")))
    doc = meta_to_doc(args.meta)
    with tempfile.TemporaryDirectory(dir=str(ROOT)) as td:
        mid, side = Path(td) / "v.mid", Path(td) / "v.json"
        doc_to_midi(doc, mid)
        write_sidecar(doc, side, str(args.meta))
        back = assign_segments(midi_to_notes(mid), read_sidecar(side))
        got = _canon(doc_to_meta(back))

    keys = ("index", "language", "time", "duration", "text", "phoneme",
            "note_pitch", "note_type", "f0")
    bad = 0
    if len(src) != len(got):
        print(f"! 段数 {len(src)} -> {len(got)}")
        bad += 1
    for a, b in zip(src, got):
        for k in keys:
            if a.get(k) != b.get(k):
                bad += 1
                if bad <= 12:
                    sa, sb = str(a.get(k)), str(b.get(k))
                    print(f"! 段 {a['index']} 字段 {k} 不一致")
                    print(f"    原: {sa[:140]}")
                    print(f"    回: {sb[:140]}")
    na = sum(len(s["text"].split()) for s in src)
    nb = sum(len(s["text"].split()) for s in got)
    print(f"[verify] 段 {len(src)}/{len(got)}  音符 {na}/{nb}  "
          f"不一致字段 {bad}")
    if bad:
        sys.exit(f"往返不等价（{bad} 处）")
    print("往返逐字段等价 ✓")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="metadata -> MIDI + sidecar")
    e.add_argument("--meta", required=True)
    e.add_argument("--out-dir", default="",
                   help="默认 <meta 的上两级>/edit/")
    e.set_defaults(fn=cmd_export)

    i = sub.add_parser("import", help="编辑后的 MIDI + sidecar -> metadata")
    i.add_argument("--mid", required=True)
    i.add_argument("--sidecar", required=True)
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_import)

    v = sub.add_parser("verify", help="往返等价自检（不编辑）")
    v.add_argument("--meta", required=True)
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
