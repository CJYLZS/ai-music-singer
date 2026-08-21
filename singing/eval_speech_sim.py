#!/usr/bin/env python3
"""口白 SIM：整句 + 3s 窗 vs 参考音色，不需要 F0 网格。

为什么单独一个脚本：`eval_sim.py` 是给歌声设计的，`--f0/--ref-f0` 是必填（要按半音
距离分桶），口白没有那个 F0 网格；而 `eval_tts_ckpts.py` 把 SIM 内联在自己的七臂流程
里，没法拿来打任意 wav。这里只复用 `eval_sim.SV`（ERes2NetV2），协议与两者一致，
所以数字可以和历史记录直接比。

参考系有两个，别混用：
  * `--ref` 给单个 prompt 切片（生产参考 6.37s）—— 和 eval_tts_ckpts 的 SIM 同源；
  * `--ref` 给 ref_talk_sing.wav 且加 `--ref-sec 162` —— 和 prosody / Step-Audio 那轮
    的 SIM 同源（162s 说话段，不是 261s）。

用法（GPT-SoVITS venv）:
    GPT-SoVITS/.venv/bin/python singing/eval_speech_sim.py \
        --ref data/voices/ref_talk_sing.wav --ref-sec 162 \
        --gen a.wav b.wav --labels a b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_sim  # noqa: E402

SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="参考音色 wav")
    ap.add_argument("--ref-sec", type=float, default=0.0,
                    help="只取参考的前 N 秒（ref_talk_sing 用 162，不要用 261）")
    ap.add_argument("--gen", required=True, nargs="+")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--win", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.gen]
    if len(labels) != len(args.gen):
        sys.exit("--labels 数量与 --gen 不一致")

    sv = eval_sim.SV(device=args.device)
    ref = eval_sim.load16k(args.ref)
    if args.ref_sec > 0:
        ref = ref[: int(args.ref_sec * SR)]
    ref_emb = sv.emb(ref)
    print(f"[ref] {args.ref} {len(ref) / SR:.2f}s")

    wlen = int(args.win * SR)
    out = {}
    for label, p in zip(labels, args.gen):
        wav = eval_sim.load16k(p)
        whole = float(eval_sim.cos(sv.emb(wav), ref_emb)) if len(wav) / SR <= 100 else None
        ws = [float(eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb))
              for s in range(0, len(wav) - wlen + 1, wlen)]
        out[label] = {"path": p, "sec": round(len(wav) / SR, 2),
                      "whole": round(whole, 4) if whole is not None else None,
                      "win_mean": round(float(np.mean(ws)), 4) if ws else None,
                      "win_min": round(float(np.min(ws)), 4) if ws else None,
                      "win_n": len(ws)}
        w = out[label]
        print(f"[sim] {label:14s} {w['sec']:6.2f}s  whole={w['whole']}  "
              f"win{args.win:g}_mean={w['win_mean']} min={w['win_min']} ({w['win_n']} 窗)")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"ref": args.ref, "ref_sec": args.ref_sec, "win": args.win,
                        "arms": out}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n报告: {args.out}")


if __name__ == "__main__":
    main()
