#!/usr/bin/env python
"""Fix "转音转得过快 / 短时间转了好多个音" in score-control SVS, by editing the
score metadata only (no audio, no retraining).

DIAGNOSIS (measured — see AGENTS.md "Score-control SVS"):
  Not a pitch problem. Our output is *smoother* than the real singer frame to
  frame (0.174 vs 0.257 st/20ms) and resolves into FEWER distinguishable pitch
  plateaus than the score has notes (0.64x, vs the human's 0.81x). What is
  actually audible is RE-ARTICULATION at slur notes (note_type=3, one word
  carried over several notes). Energy dip at slur boundaries: ours 0.450 vs the
  real singer's 0.196 (0 = perfectly connected, 1 = fully broken) — 7/13 slurs
  clearly re-attack. The ear reads a re-attacked syllable as a new, hurried note.

WHY it happens, and why the obvious fix is wrong:
  `DataProcessor.preprocess` (data_processor.py:94-105) wraps EVERY note in
  <BOW> ... <EOW> word-boundary tokens, slur notes included, and the metadata
  repeats the whole syllable on each slur note (`为/zh_wei4` then `为/zh_wei2`).
  So the model gets a fresh word onset mid-vowel. Tempting fix: replace the slur
  note's phoneme with a "sustain" token. There is no such token — <SP> is
  silence, and repeating the syllable is upstream's OWN convention in every demo
  (zh/en/yue all do it), i.e. it is what the model was trained on. Substituting
  anything else is out-of-distribution.

  The in-distribution fix is to stop the slur from being a separate NOTE:

    --mode merge   fold each slur note into its parent, so the word occupies one
                   continuous note. Kills the re-attack. Cost: the slur's pitch
                   change is lost, so the melody is slightly flattened.

    --f0-out PATH  (combine with --mode merge) additionally synthesize a smooth
                   F0 contour from the ORIGINAL note sequence and write it as a
                   .npy, so pitch movement is restored continuously via
                   `--control melody` instead of as note steps. Still no source
                   vocal anywhere: the curve is generated from the score.

Usage:
  # articulation only (score mode)
  python singing/fix_slur.py --in music.json --out music_merged.json --mode merge

  # articulation + continuous pitch (melody mode) — recommended
  python singing/fix_slur.py --in music.json --out music_legato.json \
      --mode merge --f0-out music_legato_f0.npy --glide-ms 90
"""
from __future__ import annotations

import argparse
import json

import numpy as np

F0_GRID_HZ = 50.0  # 24000 / 480, the grid every f0 field in this repo uses


def parse(seg: dict) -> dict:
    return {
        "duration": [float(x) for x in seg["duration"].split()],
        "text": seg["text"].split(),
        "phoneme": seg["phoneme"].split(),
        "note_pitch": [int(x) for x in seg["note_pitch"].split()],
        "note_type": [int(x) for x in seg["note_type"].split()],
    }


def emit(seg: dict, f: dict, f0: np.ndarray | None) -> dict:
    out = dict(seg)
    out["duration"] = " ".join(f"{d:.2f}" for d in f["duration"])
    out["text"] = " ".join(f["text"])
    out["phoneme"] = " ".join(f["phoneme"])
    out["note_pitch"] = " ".join(str(p) for p in f["note_pitch"])
    out["note_type"] = " ".join(str(t) for t in f["note_type"])
    if f0 is not None:
        out["f0"] = " ".join(f"{x:.1f}" for x in f0)
    return out


def merge_slurs(f: dict) -> tuple[dict, int]:
    """Absorb every slur note (type 3) into the preceding note."""
    keys = ("duration", "text", "phoneme", "note_pitch", "note_type")
    out: dict[str, list] = {k: [] for k in keys}
    merged = 0
    for i, ty in enumerate(f["note_type"]):
        if ty == 3 and out["duration"]:
            out["duration"][-1] += f["duration"][i]
            merged += 1
            continue
        for k in keys:
            out[k].append(f[k][i])
    return out, merged


def synth_f0(f: dict, glide_ms: float, n_frames: int, vibrato_cents: float,
             vibrato_hz: float) -> np.ndarray:
    """Build a smooth F0 contour from the note sequence.

    Rests stay at 0 (unvoiced, which is how f0_to_coarse encodes silence).
    Pitch transitions use a raised-cosine ramp centred on the note boundary, so
    the glide is spread over `glide_ms` instead of being a step.
    """
    grid = F0_GRID_HZ
    # Per-frame target MIDI, with 0 for rests.
    tgt = np.zeros(n_frames)
    t = 0.0
    for d, p in zip(f["duration"], f["note_pitch"]):
        a, b = int(round(t * grid)), int(round((t + d) * grid))
        b = min(b, n_frames)
        if b > a and p > 0:
            tgt[a:b] = p
        t += d

    out = tgt.copy()
    half = max(int(round(glide_ms / 1000.0 * grid / 2)), 1)
    # Smooth only voiced-to-voiced transitions; never bleed across a rest, which
    # would turn silence into a portamento.
    for i in range(1, n_frames):
        if tgt[i] == tgt[i - 1] or tgt[i] <= 0 or tgt[i - 1] <= 0:
            continue
        a, b = max(i - half, 0), min(i + half, n_frames)
        if (tgt[a:b] <= 0).any():
            continue
        p0, p1 = tgt[i - 1], tgt[i]
        k = np.arange(a, b)
        w = 0.5 * (1 - np.cos(np.pi * (k - a) / max(b - a - 1, 1)))
        out[a:b] = p0 + (p1 - p0) * w

    if vibrato_cents > 0:
        # Vibrato only inside sustained stretches; a rest boundary must stay dry.
        v = out > 0
        phase = 2 * np.pi * vibrato_hz * np.arange(n_frames) / grid
        depth = np.zeros(n_frames)
        run = 0
        for i in range(n_frames):
            run = run + 1 if v[i] else 0
            # ramp in over 0.3s so short notes get no vibrato
            depth[i] = min(max(run - 15, 0) / 15.0, 1.0)
        out = np.where(v, out + depth * (vibrato_cents / 100.0) * np.sin(phase), out)

    hz = np.zeros(n_frames)
    v = out > 0
    hz[v] = 440.0 * 2 ** ((out[v] - 69.0) / 12.0)
    return hz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--mode", choices=["merge", "none"], default="merge")
    ap.add_argument("--f0-out", default="",
                    help="also write a synthesized smooth F0 .npy for --control melody")
    ap.add_argument("--glide-ms", type=float, default=90.0,
                    help="transition width; real singer measured ~40-60ms, "
                         "so 80-120 gives a deliberately smoother legato")
    ap.add_argument("--vibrato-cents", type=float, default=0.0,
                    help="vibrato depth on sustained notes (0 = off)")
    ap.add_argument("--vibrato-hz", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        segs = json.load(fh)

    total_merged = 0
    out_segs = []
    f0_all: list[np.ndarray] = []

    for seg in segs:
        orig = parse(seg)
        # F0 is synthesized from the ORIGINAL notes (slurs included), so the
        # melody survives even though the notes get merged for articulation.
        n_frames = int(round(sum(orig["duration"]) * F0_GRID_HZ))
        f0 = None
        if args.f0_out:
            f0 = synth_f0(orig, args.glide_ms, n_frames,
                          args.vibrato_cents, args.vibrato_hz)
            f0_all.append(f0)

        g, merged = merge_slurs(orig) if args.mode == "merge" else (orig, 0)
        total_merged += merged

        # Duration total must be preserved exactly: DataProcessor derives the mel
        # frame count from sum(duration), and cli/inference.py sizes the output
        # buffer from `time`; drift shows up as a broadcast error.
        assert abs(sum(g["duration"]) - sum(orig["duration"])) < 1e-6, "duration drift"
        for k in ("text", "phoneme", "note_pitch", "note_type"):
            assert len(g[k]) == len(g["duration"]), f"{k} length mismatch"
        out_segs.append(emit(seg, g, f0))

    with open(args.dst, "w", encoding="utf-8") as fh:
        json.dump(out_segs, fh, ensure_ascii=False, indent=2)

    n_slur = sum(1 for s in segs for t in s["note_type"].split() if t == "3")
    print(f"mode={args.mode}  slur notes in source: {n_slur}  merged: {total_merged}")
    print(f"wrote {args.dst}")
    if args.f0_out:
        cat = np.concatenate(f0_all)
        np.save(args.f0_out, cat.astype(np.float32))
        v = cat[cat > 0]
        print(f"wrote {args.f0_out}  {len(cat)} frames, voiced {len(v)/len(cat):.1%}, "
              f"median {np.median(v):.1f}Hz, glide {args.glide_ms:.0f}ms")


if __name__ == "__main__":
    main()
