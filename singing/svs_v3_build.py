#!/usr/bin/env python3
"""V3：把 SOFA 的音节时间轴替掉 ASR 词时间轴，重跑 ROSVOT，产出新 metadata。

为什么这样就够了（见 docs/svs-manual-calibration.md「缺陷的真正来源」）：
ROSVOT 在这条管线里不是自由转写器，`model.py:109` 用 `word_durs` 做帧级词边界，
`regulate_real_note_itv` 还会把音符端点**替换**成词区间。所以只要换掉
`item["words"] / item["word_durs"]`，音符边界就跟着换 —— 接口现成
（`apply_rwbd_ = ("word_durs" not in item)`），不用改上游一行代码。

直接在**已合并**的 long_cut_wavs 段上跑（SOFA 也是在这些段上对齐的），跳过
pipeline 的 merge_short_segments，这样 index/time 与现有 metadata 完全一致，
QC 对比是严格配对的。

`svs_song.py` 默认就会调用 `refine()`（`--no-sofa` 关掉），所以正常出片不用手动跑。
单独跑时（须以 SoulX-Singer/ 为 CWD，PYTHONPATH=.）：

  # 一条命令做完 prep -> SOFA -> rebuild（SOFA 在自己的 venv 里，自动 shell out）
  MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python ../singing/svs_v3_build.py \
      --refine ../data/songs/X/svs/target/metadata.json

  # 或者三段分开（--meta/--seg-dir/--wav-dir/--out，SOFA 要自己先跑）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESTS = {"AP", "SP", "<AP>", "<SP>", "pau", "cl", ""}
CONSONANTS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q",
              "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"}
SP = "<SP>"


def enforce_min_dur(words: list[str], durs: list[float],
                    min_dur: float = 0.021) -> tuple[list[str], list[float]]:
    """把短于 min_dur 的槽并进邻居，自己做掉，别让 ROSVOT 去做。

    上游 `model.py:97-107` 会丢掉 <20ms 的槽，但**只改 word_durs 不改 words**，
    于是 `regulate_real_note_itv` 的 assert
    `word_durs.shape[0] == np.sum(word_bd)+1` 挂掉 → 静默 fallback 到"不吸附"，
    这一段就白换时间轴了。SOFA 会产出 4ms 级的声母（实测 `ch` 0.0042s），必须先合。
    ROSVOT 帧长 = 128/24000 = 5.33ms，min_word_dur=20ms，取 21ms 留余量。

    **绝不能删掉有词的槽**：删了 words 就比 ROSVOT 的 `note2words` 索引短，
    `post_process` 会按错位的下标取字，歌词整段串位（实测 `决 定` 被并掉后，
    段尾变成「我 我 己」）。所以只有 `<SP>` 可以被并掉，有词的短槽改成
    **向邻居借时长**补到 min_dur。
    """
    w = list(words)
    d = list(durs)
    # 1) 先并掉过短的休止槽（休止没有字，删掉不影响对位）
    i = 0
    while i < len(d):
        if w[i] == SP and d[i] < min_dur and len(d) > 1:
            j = i - 1 if i > 0 else i + 1
            d[j] += d[i]
            del w[i], d[i]
            if j < i:
                i = j + 1
            continue
        i += 1
    # 2) 有词的短槽：从最长的邻居借时长，字保留
    for i, dur in enumerate(d):
        if w[i] == SP or dur >= min_dur:
            continue
        need = min_dur - dur
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(d)]
        cands.sort(key=lambda j: -d[j])
        for j in cands:
            take = min(need, max(d[j] - min_dur, 0.0))
            d[j] -= take
            d[i] += take
            need -= take
            if need <= 1e-9:
                break
    return w, d


def sofa_words(ph_seq: list[str], ph_dur: list[float], chars: list[str]):
    """SOFA 音素序列 -> (words, word_durs)，与 _build_words_with_gaps 同构。

    休止(AP/SP)合成一个 <SP> 槽；一个音节的时长 = 其声母+韵母之和；
    音节按顺序取 chars 里的字。相邻 <SP> 会合并（上游的 word_durs 也是连续的）。
    """
    words: list[str] = []
    durs: list[float] = []
    ci = 0
    pending_syl = False   # 上一个音素是声母，正在等韵母
    for ph, du in zip(ph_seq, ph_dur):
        base = ph.rstrip("0123456789")
        if ph in RESTS:
            if words and words[-1] == SP:
                durs[-1] += du
            else:
                words.append(SP)
                durs.append(du)
            pending_syl = False
            continue
        if base in CONSONANTS:
            # 声母：开一个新字槽
            words.append(chars[ci] if ci < len(chars) else "啊")
            durs.append(du)
            ci += 1
            pending_syl = True
        else:
            if pending_syl:
                durs[-1] += du      # 韵母并入刚开的字槽
                pending_syl = False
            else:
                words.append(chars[ci] if ci < len(chars) else "啊")
                durs.append(du)
                ci += 1
    return words, durs, ci


def _process(nt, item: dict) -> dict:
    """`nt.process` 的替代：自己调 infer_sample + post_process，为了拿到
    `note2words`（`post_process` 不返回它）去修歌词错位 bug。

    必须在**每次 ROSVOT 调用**后立刻修，索引才对得上；切段路径拼接的是已修好的
    结果。"""
    from preprocess.tools.note_transcription.model import infer_sample
    rosvot_out = infer_sample(item, nt.hparams, nt.models, device=nt.device,
                              save_dir=None, apply_rwbd=False, save_plot=False,
                              no_save_midi=True, no_save_npy=True,
                              verbose=False)
    res = nt.post_process(metadata=item, segment_info=item,
                          rosvot_out=rosvot_out)
    res["note2words"] = rosvot_out.get("note2words")
    res, nfix = fix_note_text(res, item["words"])
    res["n_text_fix"] = nfix
    return res


def run_maybe_split(nt, item: dict, max_sec: float) -> dict:
    """ROSVOT 有 `max_frames=4000` 硬顶 = **21.33s**（hop 128 / sr 24000）。

    超长段会被 `real_len = min(mel_len, mel2word_len)` 截断，尾部的槽全部消失，
    于是 `word_bd` 的边界数少于 word_durs 的槽数 → assert 挂 → 静默 fallback。
    上游管线没暴露这个问题是因为它在**合并前**的短段上跑 ROSVOT；我们直接用
    合并后的段，所以要自己在**休止处**切开（切在休止处不会切碎音节）。
    """
    total = sum(item["word_durs"])
    if total <= max_sec:
        return _process(nt, item)

    words, durs = item["words"], item["word_durs"]
    # 找离中点最近的 <SP> 槽作为切点
    mid, t, best = total / 2, 0.0, None
    for i, (w, d) in enumerate(zip(words, durs)):
        if w == SP and i > 0:
            if best is None or abs(t + d / 2 - mid) < best[1]:
                best = (i, abs(t + d / 2 - mid), t, d)
        t += d
    if best is None:
        print(f"!! {item['item_name']} {total:.1f}s 超限但找不到切点")
        return _process(nt, item)
    i, _, sp_t, sp_d = best
    cut = sp_t + sp_d / 2          # 在休止正中间切
    print(f"   {item['item_name']} {total:.1f}s > {max_sec}s，在 {cut:.2f}s "
          f"的休止处切成两段")

    import soundfile as sf
    wav, sr = sf.read(item["wav_fn"])
    parts, out = [], {"note_text": [], "note_dur": [], "note_pitch": [],
                      "note_type": []}
    halves = [(words[:i] + [SP], durs[:i] + [sp_d / 2], 0.0, cut),
              ([SP] + words[i + 1:], [sp_d / 2] + durs[i + 1:], cut, total)]
    for hi, (w, d, a, b) in enumerate(halves):
        sub = str(Path(item["wav_fn"]).with_suffix("")) + f".part{hi}.wav"
        sf.write(sub, wav[int(a * sr):int(b * sr)], sr)
        parts.append(sub)
        sub_item = dict(item, wav_fn=sub, origin_wav_fn=sub,
                        item_name=item["item_name"] + f".part{hi}",
                        words=w, word_durs=d)
        res = _process(nt, sub_item)
        for k in ("note_text", "note_dur", "note_pitch", "note_type"):
            out[k] = out[k] + list(res[k])
        out["n_text_fix"] = out.get("n_text_fix", 0) + res.get("n_text_fix", 0)
    for p in parts:
        Path(p).unlink(missing_ok=True)
    out["item_name"] = item["item_name"]
    return out


def fix_note_text(res: dict, words: list[str]) -> tuple[dict, int]:
    """修上游 `post_process` 的一个真 bug：歌词按**未归一化**的 note2words 取。

    `regulate_real_note_itv`（`rosvot_utils.py:12-27`）把 `note2words` 初始化成
    全 0，**只给端点落在词边界上的音符赋值**，最后统一 `+= 1`。所以"整个音符都在
    某个词内部、两端都不碰词边界"的音符会保持 0 → 变成 1 → 被当成**第一个词**。
    `model.py:488-494` 又偏偏用 `note2words_raw` 取歌词、用归一化后的算 note_type，
    两者不一致。实测 `vocal_107500_116660` 的 raw 是
    `[... 9, 9, 10, 1, 1, 10, 11, 12, 13]`，于是段尾冒出「我 我」。

    这里按"单调不减"重建：出现回跳就沿用前一个音符的词。
    """
    texts = list(res["note_text"])
    types = [int(x) for x in res["note_type"]]
    n2w = res.get("note2words")
    if not n2w:
        return res, 0
    fixed = 0
    cur = 0
    norm = []
    for idx in n2w:
        cur = max(cur, int(idx))
        norm.append(cur)
    for i, idx in enumerate(norm):
        if 0 < idx <= len(words):
            want = words[idx - 1]
            if i < len(texts) and texts[i] != want:
                texts[i] = want
                # 与前一个同词 -> 延续音；否则新字
                if i > 0 and texts[i - 1] == want and want != SP:
                    types[i] = 3
                elif want == SP:
                    types[i] = 1
                else:
                    types[i] = 2
                fixed += 1
    return dict(res, note_text=texts, note_type=types), fixed


def clamp_total(res: dict, span_sec: float) -> dict:
    """保证 `duration` 的 2 位小数之和不超过段的 time 跨度。

    `convert_metadata` 把每个 dur 格式化成 `%.2f`，几十个音符累积的舍入误差可以
    达到 ±20ms。上游的 ASR 版恰好总是往下取，我们的版本在某一段往上溢出 20ms，
    结果导出 MIDI 时与下一段首音符重叠，`svs_edit.py verify` 直接报错
    （音符 #507 结束 222.200s > 下一段开始 222.180s）。从最后一个音符扣掉。
    """
    durs = [float(x) for x in res["note_dur"]]
    if not durs:
        return res
    # 必须按**写出去的那个表示**（%.2f）算，而不是按原始 float：clamp 完还要再被
    # convert_metadata 四舍五入一次，只夹原始值会有残差漏过去（实测 +0.010s）。
    # 另外段的 time 边界不保证是 10ms 的整数倍（实测有 16.419s 的段），所以直接
    # 把 2 位小数表示的总和压到 <= floor(span, 0.01)。
    cap = int(span_sec * 100) / 100.0
    for _ in range(64):
        rounded = [round(d, 2) for d in durs]
        over = round(sum(rounded) - cap, 4)
        if over <= 0:
            break
        # 从最长的音符里扣，避免把短音符扣成 0
        order = sorted(range(len(durs)), key=lambda i: -durs[i])
        for i in order:
            take = min(over, max(round(durs[i] - 0.02, 4), 0.0))
            if take <= 0:
                continue
            durs[i] = round(durs[i] - take, 4)
            over = round(over - take, 4)
            if over <= 0:
                break
        else:
            break
    return dict(res, note_dur=durs)


def rebuild(meta_path: str, seg_dir: str, wav_dir: str, out_path: str,
            language: str = "Mandarin", max_sec: float = 21.0,
            device: str = "cuda", quiet: bool = False) -> str:
    """核心逻辑，供 svs_song.py 直接 import（须在 SoulX-Singer 的 venv/CWD 下）。"""
    from preprocess.tools import NoteTranscriber
    from preprocess.utils import convert_metadata
    from preprocess.utils import SegmentMetadata

    segs = json.load(open(meta_path, encoding="utf-8"))
    manifest = json.load(open(Path(seg_dir) / "manifest.json",
                              encoding="utf-8"))
    trans = {r["name"]: r for r in csv.DictReader(
        open(Path(seg_dir) / "transcriptions" / "transcriptions.csv",
             encoding="utf-8"))}

    nt = NoteTranscriber(
        rosvot_model_path="pretrained_models/SoulX-Singer-Preprocess/rosvot/rosvot/model.pt",
        rwbd_model_path="pretrained_models/SoulX-Singer-Preprocess/rosvot/rwbd/model.pt",
        device=device)

    out_meta = []
    for seg in segs:
        idx = seg["index"]
        if idx not in trans:
            print(f"!! {idx} 没有 SOFA 输出，保留原段")
            out_meta.append(seg)
            continue
        chars = [s["char"] for s in manifest[idx]["slots"]]
        r = trans[idx]
        words, durs, used = sofa_words(r["ph_seq"].split(),
                                       [float(x) for x in r["ph_dur"].split()],
                                       chars)
        words, durs = enforce_min_dur(words, durs)
        wav = str(Path(wav_dir) / f"{idx}.wav")
        item = {"item_name": idx, "wav_fn": wav, "origin_wav_fn": wav,
                "start_time_ms": seg["time"][0], "end_time_ms": seg["time"][1],
                "duration": seg["time"][1] - seg["time"][0],
                "words": words, "word_durs": durs, "language": language}
        res = run_maybe_split(nt, item, max_sec)
        res = clamp_total(res, (seg["time"][1] - seg["time"][0]) / 1000.0)
        sm = SegmentMetadata(
            item_name=idx, wav_fn=wav, origin_wav_fn=wav,
            start_time_ms=seg["time"][0], end_time_ms=seg["time"][1],
            language=language,
            note_text=res["note_text"], note_dur=res["note_dur"],
            note_pitch=res["note_pitch"], note_type=res["note_type"])
        out_meta.append(convert_metadata(sm))
        if used != len(chars):
            print(f"!! {idx} 只用掉 {used}/{len(chars)} 个字 —— 大概率有音节不在"
                  f"词典里被 SOFA 静默丢掉，检查 .lab")
        if not quiet:
            print(f"{idx}: 字 {used}/{len(chars)}  槽 {len(words)} -> "
                  f"音符 {len(res['note_pitch'])}"
                  + (f"  歌词错位修正 {res['n_text_fix']}"
                     if res.get("n_text_fix") else ""))
    Path(out_path).write_text(
        json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


SOFA_DIR = Path(__file__).resolve().parent.parent / "tools" / "sofa"
SOFA_CKPT = ("weights/SOFA_model_mandarin_byBaiShuo/"
             "mandarin_1228_byBaiShuo.ckpt")


def refine(meta_path: str, *, language: str = "Mandarin",
           device: str = "cuda", out_path: str | None = None,
           keep_backup: bool = True) -> str | None:
    """prep -> SOFA -> rebuild 一条龙，**默认原地覆盖 metadata.json**。

    只对 Mandarin 生效（词典是 opencpop-extension）；SOFA 缺权重/其它语言时原样
    返回 None，让调用方继续用上游 metadata，不要让出片流程挂掉。
    原文件备份成 `metadata_asr.json`（对照 + 出问题能回退）。
    """
    meta_path = str(Path(meta_path).resolve())
    tgt = Path(out_path or meta_path)
    if language != "Mandarin":
        print(f"[sofa] 跳过：SOFA 词典只覆盖 Mandarin（当前 {language}）")
        return None
    if not (SOFA_DIR / SOFA_CKPT).exists():
        print(f"[sofa] 跳过：缺权重 {SOFA_DIR/SOFA_CKPT}")
        return None

    wav_dir = str(Path(meta_path).parent / "long_cut_wavs")
    if not Path(wav_dir).is_dir():
        print(f"[sofa] 跳过：没有 {wav_dir}")
        return None

    import svs_v2_prep
    seg_dir = SOFA_DIR / "segments" / Path(meta_path).parent.parent.parent.name
    n_syl = svs_v2_prep.prepare(meta_path, wav_dir, str(seg_dir))
    print(f"[sofa] 准备 {n_syl} 个音节 -> {seg_dir}")

    cmd = [str(SOFA_DIR / ".venv/bin/python"), "infer.py",
           "-c", SOFA_CKPT, "-f", f"segments/{seg_dir.name}",
           "-d", "dictionary/opencpop-extension.txt",
           "-m", "force", "-of", "transcriptions.csv", "-sc"]
    env = dict(os.environ, MPLCONFIGDIR=str(SOFA_DIR.parent.parent /
                                            ".uv-cache" / "mpl"))
    # **不要用 text=True**：它按 locale 解码，而 svs_song.py 的运行环境下 locale 是
    # ASCII，SOFA（Lightning）会打 `💡 Tip:` 这种非 ASCII，于是父进程在解码子进程
    # 输出时抛 UnicodeDecodeError 直接挂掉 —— 表现是"预处理跑完就退出、没有 traceback"，
    # 很容易误判成显存/OOM。这里拿 bytes 自己按 UTF-8 解。
    r = subprocess.run(cmd, cwd=SOFA_DIR, env=env, capture_output=True)
    if r.returncode != 0:
        out = (r.stdout or b"").decode("utf-8", "replace")
        err = (r.stderr or b"").decode("utf-8", "replace")
        print("[sofa] 对齐失败，保留上游 metadata：\n" + out[-800:] + err[-800:])
        return None
    print("[sofa] 强制对齐完成")

    if keep_backup:
        bak = tgt.parent / "metadata_asr.json"
        if not bak.exists():
            bak.write_text(Path(meta_path).read_text(encoding="utf-8"),
                           encoding="utf-8")
            print(f"[sofa] 原 metadata 备份 -> {bak}")
    rebuild(meta_path, str(seg_dir), wav_dir, str(tgt),
            language=language, device=device, quiet=True)
    print(f"[sofa] 已写入 {tgt}")
    return str(tgt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", default="",
                    help="一条龙：prep -> SOFA -> rebuild，原地覆盖该 metadata")
    ap.add_argument("--meta", required=False)
    ap.add_argument("--max-sec", type=float, default=21.0,
                    help="ROSVOT max_frames=4000 → 21.33s 硬顶")
    ap.add_argument("--seg-dir")
    ap.add_argument("--wav-dir")
    ap.add_argument("--out")
    ap.add_argument("--language", default="Mandarin")
    args = ap.parse_args()
    if args.refine:
        if not refine(args.refine, language=args.language,
                      out_path=args.out or None):
            sys.exit("refine 未执行（见上面的原因）")
        return
    missing = [k for k in ("meta", "seg_dir", "wav_dir", "out")
               if not getattr(args, k)]
    if missing:
        ap.error("需要 --refine，或者同时给 " + "/".join("--" + m.replace("_", "-")
                                                        for m in missing))
    rebuild(args.meta, args.seg_dir, args.wav_dir, args.out,
            language=args.language, max_sec=args.max_sec)
    print("->", args.out)


if __name__ == "__main__":
    main()
