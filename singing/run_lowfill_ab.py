#!/usr/bin/env python3
"""低音填充 A/B：arm A（现有 prompt_ext）vs arm B（+低音填充 prompt_lowfill）。

复现亲亲生产参数（sing.log 08-24）：cfg 0.5 / rescale 0.75 / max_seg_sec 28 /
num_overlaps 1 / knn 0.2 k4 全池 / pitch_shift 0 / n_steps 32，seed 交给模型默认
（无 --seed 时 inference_svc 的随机性来自 CFM 的 randn；先验证 arm A 是否 bit-identical
复现现有 vocal.wav，是则证明确定性，不是则记录差异量级）。

对每首歌：
  arm A: prompt_ext.wav（已有产物）-> svc_base_replay/
  arm B: prompt_lowfill<sec>.wav（w_ext --low-fill-sec 渲染，须先跑）-> svc_lowfill/

用法（SoulX venv）:
  python singing/run_lowfill_ab.py 冬眠 去有风的地方 亲亲 \
      --low-fill 12 --max-sec 45 --ref-clone data/voices/ref_talk_sing.wav
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "SoulX-Singer"
PY = SOULX / ".venv" / "bin" / "python"
SVC_SCRIPT = SOULX / "cli" / "inference_svc.py"
MODEL = SOULX / "pretrained_models" / "SoulX-Singer" / "model-svc.pt"
CONFIG = SOULX / "soulxsinger" / "config" / "soulxsinger.yaml"

SVC_BASE = "--device cuda:0 --model_path {m} --config {c} --prompt_wav_path {pw} " \
           "--target_wav_path {tw} --prompt_f0_path {pf} --target_f0_path {tf} " \
           "--save_dir {sd} --pitch_shift 0 --n_steps 32 --cfg 0.5 --rescale_cfg 0.75 " \
           "--max_seg_sec 28.0 --num_overlaps 1 --knn_alpha 0.2 --knn_k 4 " \
           "--knn_pool_wav {pool} --fp16"


def run(cmd: list[str], tag: str) -> None:
    print(f"[{tag}] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(SOULX), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tail = "\n".join((proc.stdout or "").splitlines()[-8:])
    print(tail, flush=True)
    if proc.returncode != 0:
        sys.exit(f"[{tag}] 失败 exit {proc.returncode}:\n{proc.stderr[-2000:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="+")
    ap.add_argument("--low-fill", type=float, default=12.0,
                    help="arm B 的低音填充秒数（w_ext --low-fill-sec）")
    ap.add_argument("--max-sec", type=float, default=45.0)
    ap.add_argument("--ref-clone", default="",
                    help="kNN 池（默认从 sing.py 逻辑：ref_clean 或 ref_talk_sing）")
    args = ap.parse_args()

    for s in args.songs:
        d = ROOT / "data" / "songs" / s
        if not d.is_dir():
            sys.exit(f"歌目录不存在: {d}")
        pool = args.ref_clone or str(ROOT / "data" / "voices" / "ref_talk_sing.wav")
        for p in ("prompt_ext.wav", "prompt_lowfill.wav", "target_f0.npy",
                  f"sep2/lead.wav"):
            if not (d / p).exists() and p != "prompt_lowfill.wav":
                sys.exit(f"{d}/{p} 缺失（先跑 select/sep）")
        print(f"\n===== {s} =====", flush=True)

        # ---- arm A：复现生产基线 ----
        a_dir = d / "svc_base_replay"
        a_dir.mkdir(exist_ok=True)
        run([str(PY), str(SVC_SCRIPT),
             *SVC_BASE.format(m=MODEL, c=CONFIG,
                              pw=d / "prompt_ext.wav", tw=d / "sep2" / "lead.wav",
                              pf=d / "prompt_ext_f0.npy", tf=d / "target_f0.npy",
                              sd=a_dir, pool=pool).split()],
            f"{s}/A 基线")

        # ---- arm B：低音填充 prompt ----
        b_dir = d / "svc_lowfill"
        b_dir.mkdir(exist_ok=True)
        low_wav = d / "prompt_lowfill.wav"
        if not low_wav.exists():
            print(f"[{s}] 渲染低音填充 prompt...", flush=True)
            run([str(PY), str(ROOT / "singing" / "w_ext.py"),
                 "--stats", str(d / "prompt_stats.json"),
                 "--cands", str(d / "cands_meta.json"),
                 "--target-f0", str(d / "target_f0.npy"),
                 "--out-wav", str(low_wav),
                 "--out-f0", str(d / "prompt_lowfill_f0.npy"),
                 "--seedvc-dir", str(d / "seedvc_low"),
                 "--shifter", "nsf", "--max-sec", str(args.max_sec),
                 "--anchor-min-ratio", "1.0", "--max-shift", "8",
                 "--div-weight", "3.0", "--low-fill-sec", str(args.low_fill),
                 "--device", "cuda:0"], f"{s}/w_ext低填")
        run([str(PY), str(SVC_SCRIPT),
             *SVC_BASE.format(m=MODEL, c=CONFIG,
                              pw=low_wav, tw=d / "sep2" / "lead.wav",
                              pf=d / "prompt_lowfill_f0.npy", tf=d / "target_f0.npy",
                              sd=b_dir, pool=pool).split()],
            f"{s}/B 低填")


if __name__ == "__main__":
    main()
