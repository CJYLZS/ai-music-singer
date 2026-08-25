#!/usr/bin/env python3
"""V2 准备：从 metadata 的一个 segment 生成 SOFA 输入（wav + .lab）。

.lab 内容 = 该段的**无调拼音音节序列**（opencpop-extension 词典要求），由
metadata 的 phoneme 字段反推：去掉 <SP>、去掉 note_type==3 的延续音符，
`zh_xiao4` -> `xiao`。这样音节序列一定和音频一致，V2 只考察**时间轴**质量。

用法: SoulX-Singer/.venv/bin/python singing/svs_v2_prep.py \
        --meta data/songs/X/svs/target/metadata.json --index vocal_222180_239780 \
        --wav-dir data/songs/X/svs/target/long_cut_wavs --out tools/sofa/segments/v2
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

SP = "<SP>"


def norm_syl(syl: str) -> str:
    """g2pM 的拼音写法 -> opencpop-extension 词典的写法。

    **必须做，否则 SOFA 会静默丢掉整个音节**，后面所有字↔音节的对位平移一格。
    实测爱笑的眼睛只有 1 处（「略」= g2pM `lu:e` vs 词典 `lve`），但它让那一段
    20 个音节的歌词全部错位 —— 而且不报错，只能靠 word_seq 与 lab 逐项核对发现。
    校验方法：把 .lab 的音节逐个查 dictionary/opencpop-extension.txt 第一列。
    """
    return syl.replace("u:", "v")


def seg_syllables(seg: dict) -> tuple[list[str], list[dict]]:
    """-> (拼音音节列表, 字槽信息[含段内相对起点/时长])"""
    phones = seg["phoneme"].split()
    types = [int(x) for x in seg["note_type"].split()]
    durs = [float(x) for x in seg["duration"].split()]
    texts = seg["text"].split()
    syls: list[str] = []
    slots: list[dict] = []
    t = 0.0
    for ph, ty, du, tx in zip(phones, types, durs, texts):
        if ty != 1 and ty != 3:  # 新字槽
            syl = norm_syl(re.sub(r"\d+$", "", ph.split("_", 1)[-1]))
            syls.append(syl)
            slots.append({"syl": syl, "char": tx, "rel": t, "dur": du})
        elif ty == 3 and slots:  # 延续：并入上一个字槽
            slots[-1]["dur"] += du
        t += du
    return syls, slots


def prepare(meta_path: str, wav_dir: str, out_dir: str, index: str = "all",
            verbose: bool = False) -> int:
    """写出 SOFA 输入（wav + .lab + manifest.json），返回音节总数。

    会**先清空** out_dir 里的 wav/lab，否则换歌或换段时会残留上一次的段，
    SOFA 把它们一起对齐、manifest 里又没有，rebuild 时报"没有 SOFA 输出"。
    """
    segs = json.load(open(meta_path, encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.wav")) + list(out.glob("*.lab")):
        old.unlink()
    picked = [s for s in segs if index in ("all", s["index"])]
    if not picked:
        raise SystemExit(f"segment {index} not found; have: "
                         + ", ".join(s["index"] for s in segs))
    manifest, total = {}, 0
    for seg in picked:
        syls, slots = seg_syllables(seg)
        src = Path(wav_dir) / f"{seg['index']}.wav"
        if not src.exists():
            raise SystemExit(f"missing wav: {src}")
        shutil.copy(src, out / src.name)
        (out / f"{seg['index']}.lab").write_text(" ".join(syls) + "\n",
                                                 encoding="utf-8")
        manifest[seg["index"]] = {"time_ms": seg["time"], "slots": slots}
        total += len(syls)
        if verbose:
            print(f"{seg['index']}: {len(syls)} 音节 -> {out/src.name}")
            print("  lab:", " ".join(syls)[:160])
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--index", default="all", help="segment index，或 all")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = prepare(args.meta, args.wav_dir, args.out, index=args.index,
                verbose=True)
    print(f"{n} 音节, manifest -> {Path(args.out)/'manifest.json'}")


if __name__ == "__main__":
    main()
