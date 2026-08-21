#!/usr/bin/env python3
"""逐字强制对齐 + 韵母鼻音性分类。给鼻音测量提供取样区间。

为什么必须按字分类（这是整件事的前提）：
  中文有鼻韵尾 -n/-ng 和鼻声母 m/n，**这些字本来就该带鼻化**。整段混在一起测
  A1-P0，得到的主要是"这段文本里鼻韵尾占多少比例"，不是"音色有没有鼻音缺陷"。
  情歌口白 36 字里 CODA_N 8 个 + CODA_NG 5 个 + NAS_INIT 2 个 = 15 个（42%）本该鼻化，
  只有 21 个 ORAL 字的鼻音是**缺陷证据**。

对齐用 FunASR 的 `speech_timestamp_prediction-v1-16k-offline`（已在本地缓存）：
  * paraformer-large 那个 ASR 模型**不输出时间戳**（`output_timestamp=True` 也不给），
    所以走独立的强制对齐模型，输入 (wav, 文本) 两路。
  * 实测 36 字文本 → 36 个 [start_ms, end_ms]，一一对应。

分类（`pypinyin` strict 韵母）:
  CODA_NG  韵母以 ng 结尾（ong/ing/ang...）—— 强鼻化，预期
  CODA_N   韵母以 n 结尾（an/in/uen...）  —— 强鼻化，预期
  NAS_INIT 声母是 m/n，韵母无鼻尾        —— 鼻音会前向协同发音，半预期
  ORAL     其余                          —— **唯一的干净样本**

用法（GPT-SoVITS venv）:
    GPT-SoVITS/.venv/bin/python singing/align_syllables.py \
        --wav data/songs/情歌/mv/speech.wav \
        --text-file data/songs/情歌/mv/intro.txt --out align.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TP_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/iic--speech_timestamp_prediction-v1-16k-offline"
    "/snapshots/master"
)
ASR_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/"
    "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    "/snapshots/master"
)
VAD_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch"
    "/snapshots/master"
)

CLASSES = ("ORAL", "NAS_INIT", "CODA_N", "CODA_NG")


def han_only(s: str) -> str:
    """只留汉字：对齐模型的 token 是汉字，标点/空白会打乱一一对应。"""
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


def classify(text: str) -> list[str]:
    """逐字判鼻音性。返回与 text 等长的标签表。"""
    from pypinyin import Style, lazy_pinyin

    fins = lazy_pinyin(text, style=Style.FINALS, strict=True)
    inis = lazy_pinyin(text, style=Style.INITIALS, strict=True)
    out = []
    for f, i in zip(fins, inis):
        if f.endswith("ng"):
            out.append("CODA_NG")
        elif f.endswith("n"):
            out.append("CODA_N")
        elif i in ("m", "n"):
            out.append("NAS_INIT")
        else:
            out.append("ORAL")
    return out


def seed_all() -> None:
    """FunASR 的 VAD 会消耗全局 RNG，每次 generate 前都要重置（见 eval_cer.py 的注释）。"""
    import numpy as np
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)


def transcribe(wav: str, device: str) -> str:
    """没给文本时先 ASR 拿一份（对齐要求文本与音频内容一致）。"""
    from funasr import AutoModel

    seed_all()
    m = AutoModel(model=ASR_DIR, vad_model=VAD_DIR, device=device,
                  disable_update=True, disable_pbar=True)
    seed_all()
    return han_only(m.generate(input=wav, batch_size_s=300)[0]["text"])


def align(wav: str, text: str, device: str) -> list[dict]:
    """返回 [{char, cls, t0, t1}]，时间单位秒。"""
    from funasr import AutoModel

    for d, what in ((TP_DIR, "timestamp_prediction"),):
        if not os.path.exists(os.path.join(d, "model.pt")):
            sys.exit(f"缺少已缓存的 {what}: {d}/model.pt")
    seed_all()
    m = AutoModel(model=TP_DIR, device=device, disable_update=True, disable_pbar=True)
    seed_all()
    r = m.generate(input=(wav, text), data_type=("sound", "text"))[0]
    ts = r.get("timestamp")
    if not ts:
        sys.exit("对齐模型没有返回 timestamp")
    if len(ts) != len(text):
        sys.exit(f"时间戳数({len(ts)}) 与字数({len(text)}) 不一致，无法一一对应")
    cls = classify(text)
    return [{"char": c, "cls": k, "t0": a / 1000.0, "t1": b / 1000.0}
            for c, k, (a, b) in zip(text, cls, ts)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", help="对齐文本（缺省先 ASR）")
    ap.add_argument("--text-file")
    ap.add_argument("--sec", type=float, default=0.0,
                    help="只取前 N 秒（真人参考用 162，不要用 261）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    wav = args.wav
    if args.sec > 0:
        # 对齐模型吃文件路径，先切一份临时 wav
        import numpy as np
        import soundfile as sf
        import librosa

        y, _ = librosa.load(wav, sr=16000, mono=True)
        y = y[: int(args.sec * 16000)]
        wav = str(Path(args.out or "/tmp/align_cut").with_suffix(".cut.wav"))
        sf.write(wav, y.astype(np.float32), 16000)
        print(f"[cut] 前 {args.sec}s -> {wav}")

    if args.text_file:
        text = han_only(Path(args.text_file).read_text(encoding="utf-8"))
    elif args.text:
        text = han_only(args.text)
    else:
        text = transcribe(wav, args.device)
        print(f"[asr] {len(text)} 字: {text[:60]}...")

    items = align(wav, text, args.device)
    from collections import Counter

    cnt = Counter(it["cls"] for it in items)
    print(f"[align] {len(items)} 字, " +
          ", ".join(f"{k}={cnt.get(k, 0)}" for k in CLASSES))
    dur = sum(it["t1"] - it["t0"] for it in items)
    oral = sum(it["t1"] - it["t0"] for it in items if it["cls"] == "ORAL")
    print(f"[align] 总时长 {dur:.2f}s，其中 ORAL {oral:.2f}s "
          f"({oral / max(dur, 1e-9) * 100:.0f}%)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"wav": wav, "text": text, "items": items}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n对齐: {args.out}")


if __name__ == "__main__":
    main()
