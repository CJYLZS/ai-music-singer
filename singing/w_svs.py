#!/usr/bin/env python
"""Step 2: score-control SVS smoke test — synthesize singing from MIDI notes +
lyrics with NO source vocal, using a SPEECH reference for timbre.

This is the first end-to-end run of SoulX-Singer's *other* checkpoint
(`model.pt`, SVS) as opposed to the `model-svc.pt` that `sing.py` uses. In score
mode `soulxsinger.py:121-125` sets gt_f0=None, so the original singer's audio
never enters the model — which is the whole point of the experiment.

Two stages:
  1. prompt metadata: run SoulX's own PreprocessPipeline on our reference with
     vocal_sep=False (our `sing.py` already produced a dry vocal, and the
     karaoke/dereverb weights are deliberately not downloaded). This yields
     note_pitch for the prompt side, which score mode requires for BOTH sides.
  2. synthesis: cli.inference --control score with a target metadata JSON.

Notes on two things that are easy to get wrong:
  - The prompt must be a single segment: cli/inference.py:82 takes
    prompt_meta_list[0] and ignores the rest. `--max-merge-duration` is
    therefore what controls how much of the reference is usable, and the model's
    own 30s-class limits still apply.
  - The prompt WAV passed to inference must be the exact audio the metadata
     describes (its `time` field indexes into it), so we pass the cut/merged wav
     that the pipeline itself wrote, never the original file.

Run (CWD must be SoulX-Singer/ — relative model paths):
  cd SoulX-Singer && PYTHONPATH=. .venv/bin/python ../singing/w_svs.py \
      --ref ../data/songs/绿色/prompt.wav \
      --target-meta example/audio/music.json \
      --out-dir /tmp/opencode/svs_run1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np


def build_prompt_meta(ref_wav: str, out_dir: str, language: str, device: str,
                      max_merge_ms: int) -> tuple[str, str]:
    """Run SoulX's preprocess on the reference; return (metadata_path, wav_path)."""
    from preprocess.pipeline import PreprocessPipeline

    save_dir = os.path.join(out_dir, "prompt_meta")
    os.makedirs(save_dir, exist_ok=True)

    # Feed a mono copy: VocalDetector/ROSVOT expect mono, and vocal_sep=False
    # means the pipeline uses this file verbatim as the "vocal".
    import librosa
    import soundfile as sf
    y, sr = librosa.load(ref_wav, sr=None, mono=True)
    ref_mono = os.path.join(save_dir, "ref_mono.wav")
    sf.write(ref_mono, y, sr)
    print(f"[svs] prompt source {ref_wav}  {len(y)/sr:.2f}s @ {sr}Hz")

    pipe = PreprocessPipeline(
        device=device,
        language=language,
        save_dir=save_dir,
        vocal_sep=False,          # our reference is already a dry vocal
        max_merge_duration=max_merge_ms,
        midi_transcribe=True,     # we need note_pitch for the prompt side
    )
    pipe.run(ref_mono, vocal_sep=False, max_merge_duration=max_merge_ms, language=language)

    meta_path = os.path.join(save_dir, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        metas = json.load(f)
    if not metas:
        sys.exit("! preprocess produced no prompt segments")

    # cli/inference.py:82 uses only the first segment.
    first = metas[0]
    print(f"[svs] prompt segments: {len(metas)} (only [0] is used by inference)")
    for m in metas:
        n = len(m["note_pitch"].split())
        print(f"        {m['index']}  {m['time']}  {n} notes")

    # The merged/cut wav that this metadata indexes into.
    cand = os.path.join(save_dir, "long_cut_wavs", f"{first['index']}.wav")
    if not os.path.exists(cand):
        cand = os.path.join(save_dir, "cut_wavs", f"{first['index']}.wav")
    if not os.path.exists(cand):
        sys.exit(f"! cannot find the wav for prompt segment {first['index']}")

    # Keep only the first segment so the intent is explicit in the artifact.
    solo = os.path.join(save_dir, "prompt_first.json")
    with open(solo, "w", encoding="utf-8") as f:
        json.dump([first], f, ensure_ascii=False, indent=2)
    return solo, cand


def describe(tag: str, meta_path: str) -> None:
    with open(meta_path, encoding="utf-8") as f:
        metas = json.load(f)
    print(f"\n[svs] {tag}: {len(metas)} segment(s)")
    for m in metas[:3]:
        pitch = np.array([int(x) for x in m["note_pitch"].split()])
        dur = np.array([float(x) for x in m["duration"].split()])
        real = pitch[pitch > 0]
        print(f"        {m['index']}: {pitch.size} notes, span {dur.sum():.1f}s, "
              f"pitch median {np.median(real):.0f} range {real.min()}..{real.max()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="speech reference (timbre source)")
    ap.add_argument("--target-meta", required=True,
                    help="target metadata json (notes + lyrics); no audio needed")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--control", default="score", choices=["score", "melody"])
    ap.add_argument("--language", default="Mandarin")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-merge-ms", type=int, default=30000)
    ap.add_argument("--pitch-shift", type=int, default=0)
    ap.add_argument("--auto-shift", action="store_true",
                    help="let the model transpose the song into the reference's register")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--reuse-prompt", default="",
                    help="skip preprocess and use this prompt metadata json")
    ap.add_argument("--reuse-prompt-wav", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.reuse_prompt:
        prompt_meta, prompt_wav = args.reuse_prompt, args.reuse_prompt_wav
        if not prompt_wav:
            sys.exit("! --reuse-prompt requires --reuse-prompt-wav")
    else:
        prompt_meta, prompt_wav = build_prompt_meta(
            args.ref, args.out_dir, args.language, args.device, args.max_merge_ms)

    describe("prompt", prompt_meta)
    describe("target", args.target_meta)

    save_dir = os.path.join(args.out_dir, "generated")
    cmd = [
        sys.executable, "-m", "cli.inference",
        "--device", args.device,
        "--model_path", "pretrained_models/SoulX-Singer/model.pt",
        "--config", "soulxsinger/config/soulxsinger.yaml",
        "--prompt_wav_path", prompt_wav,
        "--prompt_metadata_path", prompt_meta,
        "--target_metadata_path", args.target_meta,
        "--phoneset_path", "soulxsinger/utils/phoneme/phone_set.json",
        "--save_dir", save_dir,
        "--control", args.control,
        "--pitch_shift", str(args.pitch_shift),
        # 本地补丁: 上游 cli/inference.py 从不播种（CFM 初始噪声走全局 RNG），
        # 实测同配置 4 次 md5 全不同、SIM std 0.0065。任何 A/B 都必须固定它。
        "--seed", str(args.seed),
    ]
    if args.auto_shift:
        cmd.append("--auto_shift")
    if args.fp16:
        cmd.append("--fp16")

    print(f"\n[svs] control={args.control} pitch_shift={args.pitch_shift} "
          f"auto_shift={args.auto_shift}")
    print(f"[svs] prompt wav  {prompt_wav}")
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"! inference failed (exit {r.returncode})")

    gen = os.path.join(save_dir, "generated.wav")
    if not os.path.exists(gen):
        sys.exit(f"! inference finished but {gen} missing")
    import soundfile as sf
    info = sf.info(gen)
    print(f"\n[svs] OK -> {gen}  {info.duration:.2f}s @ {info.samplerate}Hz")
    shutil.copy(prompt_wav, os.path.join(args.out_dir, "prompt_used.wav"))


if __name__ == "__main__":
    main()
