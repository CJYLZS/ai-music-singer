#!/usr/bin/env python3
"""批量语音合成 CLI：一个纯文本文件 -> 多个编号语音文件。

复用 MV 前奏口白的 TTS 接口（listen_lipsync/intro_speech.py 的权重表
GPT_WEIGHTS / girlish.list 转写反查 lookup_ref_text / 子进程封装 run，
底层同为 GPT-SoVITS/infer.py），只是把「一段口白」换成「多段文本逐段合成」。
权重表更新（如换 SoVITS）只改 intro_speech.py 一处，本脚本自动跟随。

输入格式（**空行 = 分段边界**：段与段之间至少一个空行，即两次回车；
连续多个空行等价一个；段内不空行，需要换行直接写）:

    第一段文本。

    第二段文本。

用法（须用 GPT-SoVITS venv 运行，与 intro_speech.py 相同）:
    GPT-SoVITS/.venv/bin/python tts_batch.py script.txt
    GPT-SoVITS/.venv/bin/python tts_batch.py script.txt --out-dir out/ \\
        --tts gpt-ft --ref-audio data/voices/girlish_sliced/xxx.wav --speed 0.9

产物（--out-dir，默认 <输入文件同目录>/<输入名>_tts/）:
    01.wav 02.wav …     每段一个，按出现顺序编号（补零到至少两位，方便 ls 排序）
    01.txt …            每段实际送入 TTS 的文本（复用凭据：文本变了自动重推该段）
    01.tts.log …        每段 infer.py 的完整输出（排错用）
    batch_meta.json     本次参数 + 各段时长快照

注意: 每段起一个 infer.py 子进程，权重重复加载（约 10-20s/段），
段数很多时这是主要开销；已有的编号产物默认复用，断点续跑即可。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "listen_lipsync"))

try:
    # 复用 MV 口白的 TTS 接口：GPT_WEIGHTS / lookup_ref_text / run / GPTV / ROOT
    import intro_speech as IS
except ImportError as e:
    sys.exit(f"error: {e}\n  本脚本须用 GPT-SoVITS venv 运行: "
             f"GPT-SoVITS/.venv/bin/python tts_batch.py ...")

import soundfile as sf

# 与 GPT-SoVITS/infer.py 的 DEFAULT_REF 保持一致（不 import infer.py：会拉起 torch）
DEFAULT_REF = "data/voices/girlish_sliced/ref-vocal.wav_0004307200_0004589760.wav"

# 一个及以上连续空行 = 分段边界（\n\n 即切；再多的空行等价一个）
SEP_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")


def split_segments(text: str) -> list[str]:
    """按『空行』（一个及以上连续空行）切段。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [b.strip() for b in SEP_RE.split(text) if b.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", help="输入纯文本文件（两个及以上连续空行分段）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 <输入文件同目录>/<输入名>_tts/）")
    ap.add_argument("--tts", default="gpt-ft", choices=sorted(IS.GPT_WEIGHTS),
                    help="TTS 权重（默认 gpt-ft = girlish GPT e30 + talksing_s2mix50，"
                         "与 mv.py 前奏口白同一默认）")
    ap.add_argument("--ref-audio", default=str(ROOT / DEFAULT_REF),
                    help="参考音频（角色音色，3~10s；默认与 infer.py 的 DEFAULT_REF 一致）")
    ap.add_argument("--ref-text", default=None,
                    help="参考音频转写（缺省从 girlish.list 反查）")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="语速倍率（GPT-SoVITS 原生时长缩放，<1 = 更慢，0.6-1.65）")
    ap.add_argument("--lang", default="中文",
                    help="目标文本语言（透传 infer.py，默认 中文）")
    ap.add_argument("--seed", type=int, default=42,
                    help="采样种子（默认 42，与 intro_speech.py 一致；各段共用同一 seed，"
                         "保证整批复现。不播种则重跑必变，见 intro_speech.py 的说明）")
    ap.add_argument("--force", action="store_true", help="重推全部段落（默认复用已有编号 wav）")
    args = ap.parse_args()

    src = Path(args.text)
    if not src.is_file():
        sys.exit(f"error: 输入文件不存在: {src}")
    segments = split_segments(src.read_text(encoding="utf-8-sig"))
    if not segments:
        sys.exit("error: 输入无有效文本段（段与段之间须用空行分隔）")

    out_dir = Path(args.out_dir) if args.out_dir else src.parent / f"{src.stem}_tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_abs = Path(args.ref_audio).resolve()
    if not ref_abs.is_file():
        sys.exit(f"error: 参考音频不存在: {ref_abs}")
    ref_text = args.ref_text or IS.lookup_ref_text(str(ref_abs))
    if not ref_text:
        sys.exit(f"error: 无法确定参考音频的转写文本（girlish.list 无 {ref_abs.name}），"
                 f"请用 --ref-text 显式指定")

    w = IS.GPT_WEIGHTS[args.tts]
    n = len(segments)
    width = max(2, len(str(n)))
    print(f"输入: {src}  ->  {n} 段")
    print(f"引擎: {w['label']}  |  ref: {ref_abs.name}  |  speed={args.speed:g}  "
          f"seed={args.seed}  lang={args.lang}")
    print(f"输出: {out_dir}/  ({width} 位编号)\n")

    t_all = time.time()
    made = reused = 0
    durs: dict[str, float] = {}
    for i, seg in enumerate(segments, 1):
        name = f"{i:0{width}d}"
        wav = out_dir / f"{name}.wav"
        txt = out_dir / f"{name}.txt"
        log = out_dir / f"{name}.tts.log"

        if wav.exists() and not args.force:
            if txt.exists() and txt.read_text(encoding="utf-8").strip() == seg:
                d = sf.info(wav).frames / sf.info(wav).samplerate
                durs[name] = d
                print(f"[{i}/{n}] {name}.wav  复用 ({d:.2f}s)")
                reused += 1
                continue
            print(f"[{i}/{n}] 文本已改，重推 {name}.wav")

        txt.write_text(seg + "\n", encoding="utf-8")
        cmd = [
            str(IS.GPTV), str(IS.ROOT / "GPT-SoVITS/infer.py"),
            "-t", "@" + str(txt.resolve()),
            "-o", str(wav.resolve()),
            "--gpt", str(w["gpt"]),
            "--sovits", str(w["sovits"]),
            "--ref", str(ref_abs),
            "--speed", f"{args.speed:g}",
            "--lang", args.lang,
            "--seed", str(args.seed),
            "-q",
        ]
        if ref_text:
            cmd += ["--ref-text", ref_text]
        t0 = time.time()
        IS.run(cmd, IS.ROOT / "GPT-SoVITS", log)   # 失败即退出并打 log 尾部
        if not wav.is_file():
            sys.exit(f"error: infer.py 未产出 {wav}（见 {log}）")
        info = sf.info(wav)
        d = info.frames / info.samplerate
        durs[name] = d
        made += 1
        print(f"[{i}/{n}] {len(seg)} 字 -> {name}.wav  {d:.2f}s @ {info.samplerate}Hz"
              f"  ({time.time() - t0:.0f}s)")

    meta = {
        "params": {k: v for k, v in vars(args).items()},
        "tts_label": w["label"],
        "ref_audio": str(ref_abs),
        "segments": n,
        "made": made,
        "reused": reused,
        "durations": {k: round(v, 3) for k, v in durs.items()},
        "total_sec": round(sum(durs.values()), 3),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    (out_dir / "batch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n完成: 新合成 {made} / 复用 {reused}，共 {sum(durs.values()):.1f}s 音频"
          f"（{time.time() - t_all:.0f}s）-> {out_dir}/")


if __name__ == "__main__":
    main()
