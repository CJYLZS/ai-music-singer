#!/usr/bin/env python
"""Step 1 probe: can a SPEECH reference produce usable note annotation for
SoulX-Singer's score-control mode?

Why this is the go/no-go gate for the whole MIDI+lyrics idea:
  score mode reads note_pitch for BOTH sides (soulxsinger.py:122-123), not just
  the target. Our reference is speech, and speech has no notes. ROSVOT is a
  singing note segmenter; feeding it speech may yield a jittery pseudo-note
  sequence that is useless as an in-context prompt. If this fails, the whole
  approach is dead regardless of how good the target MIDI is.

What it does (no SVS model needed, so it can run before model.pt finishes):
  1. VAD-segment the reference with SoulX's own VocalDetector (the same code
     path the real pipeline uses, so we measure the real thing).
  2. ASR the segment with the seaco paraformer to get words + word durations.
  3. Run ROSVOT note transcription on it.
  4. Report diagnostics that decide usable/unusable, and cross-check the notes
     against RMVPE F0 measured directly from the same audio.

Judging criteria, and why each one matters:
  - notes/sec: singing is ~2-4 notes/sec. A much higher rate means ROSVOT is
    chopping speech intonation into fragments rather than finding notes.
  - median note duration: <0.1s means fragmentation.
  - |note_pitch - F0|: ROSVOT's note pitch should agree with the measured F0 of
    the same frames. Large disagreement means the note track is not describing
    this audio at all.
  - pitch range: speech has a narrower range than singing; a very wide range
    would indicate octave errors.

Run:
  cd SoulX-Singer && PYTHONPATH=. .venv/bin/python ../singing/probe_svs_prompt.py \
      --wav ../data/songs/绿色/prompt.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    out = np.zeros_like(hz, dtype=np.float64)
    v = hz > 0
    out[v] = 69.0 + 12.0 * np.log2(hz[v] / 440.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="speech reference (mono wav)")
    ap.add_argument("--out-dir", default="/tmp/opencode/svs_probe")
    ap.add_argument("--language", default="Mandarin")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seg", type=int, default=1,
                    help="how many VAD segments to transcribe (prompt uses only the first)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pre = "pretrained_models/SoulX-Singer-Preprocess"

    from preprocess.tools import F0Extractor, VocalDetector, LyricTranscriber, NoteTranscriber

    # --- mono/44.1k normalisation: VocalDetector + ROSVOT expect a mono file ---
    import librosa
    import soundfile as sf
    wav_in = os.path.abspath(args.wav)
    y, sr = librosa.load(wav_in, sr=None, mono=True)
    wav_path = os.path.join(args.out_dir, "ref_mono.wav")
    sf.write(wav_path, y, sr)
    print(f"[probe] input {wav_in}\n        {len(y)/sr:.2f}s @ {sr}Hz -> {wav_path}")

    # --- 1. F0 (also used later as ground truth to check the notes against) ---
    f0x = F0Extractor(model_path=f"{pre}/rmvpe/rmvpe.pt", device=args.device)
    f0 = f0x.process(wav_path, f0_path=wav_path.replace(".wav", "_f0.npy"))
    f0 = np.asarray(f0).squeeze()
    voiced = f0[f0 > 0]
    print(f"[probe] F0: {len(f0)} frames, voiced {len(voiced)/max(len(f0),1):.1%}, "
          f"median {np.median(voiced):.1f}Hz -> MIDI {hz_to_midi(np.array([np.median(voiced)]))[0]:.1f}")

    # --- 2. VAD segmentation (same code path as the real pipeline) ---
    vd = VocalDetector(cut_wavs_output_dir=os.path.join(args.out_dir, "cut_wavs"))
    segments = vd.process(wav_path, f0=f0)
    print(f"[probe] VocalDetector -> {len(segments)} segment(s)")
    for s in segments[:5]:
        print(f"          {s['item_name']}  {s.get('start_time_ms')}..{s.get('end_time_ms')} ms")
    if not segments:
        sys.exit("! no segments produced; reference may be too quiet or too short")

    # --- 3. lyric transcription (words + per-word durations feed note alignment) ---
    lt = LyricTranscriber(
        zh_model_path=f"{pre}/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        en_model_path=f"{pre}/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
        device=args.device,
    )
    nt = NoteTranscriber(
        rosvot_model_path=f"{pre}/rosvot/rosvot/model.pt",
        rwbd_model_path=f"{pre}/rosvot/rwbd/model.pt",
        device=args.device,
        verbose=True,
    )

    results = []
    for seg in segments[: args.max_seg]:
        f0x.process(seg["wav_fn"], f0_path=seg["wav_fn"].replace(".wav", "_f0.npy"))
        words, durs = lt.process(seg["wav_fn"], args.language)
        seg["words"], seg["word_durs"], seg["language"] = words, durs, args.language
        print(f"\n[probe] ASR {seg['item_name']}: {len(words)} words")
        print(f"        {' '.join(words)[:200]}")
        meta = nt.process(seg, segment_info=seg)
        results.append((seg, meta))

    # --- 4. diagnostics that decide go/no-go ---
    print("\n" + "=" * 70)
    print("  PROMPT-SIDE NOTE ANNOTATION DIAGNOSTICS")
    print("=" * 70)
    for seg, meta in results:
        pitch = np.asarray(meta.get("note_pitch") or [], dtype=float)
        dur = np.asarray(meta.get("note_dur") or meta.get("ep_notedurs") or [], dtype=float)
        types = np.asarray(meta.get("note_type") or meta.get("ep_types") or [], dtype=int)
        total = float(dur.sum()) if dur.size else 0.0
        real = pitch[pitch > 0]

        print(f"\n{seg['item_name']}   span {total:.2f}s")
        print(f"  notes           {pitch.size} total, {real.size} pitched, "
              f"{int((types == 1).sum())} rest, {int((types == 3).sum())} slur")
        if total > 0:
            print(f"  notes/sec       {pitch.size / total:.2f}   (singing is ~2-4)")
        if dur.size:
            print(f"  note dur        median {np.median(dur):.3f}s  "
                  f"min {dur.min():.3f}  max {dur.max():.3f}  "
                  f"<0.1s: {int((dur < 0.1).sum())}/{dur.size}")
        if real.size:
            print(f"  pitch           median {np.median(real):.1f}  "
                  f"p10 {np.percentile(real,10):.0f}  p90 {np.percentile(real,90):.0f}  "
                  f"range {real.min():.0f}..{real.max():.0f} ({real.max()-real.min():.0f} st)")

            # Cross-check: do the notes agree with the F0 actually measured?
            segf0 = np.load(seg["wav_fn"].replace(".wav", "_f0.npy")).squeeze()
            f0_midi = hz_to_midi(np.asarray(segf0, dtype=float))
            hop_s = total / max(len(f0_midi), 1)
            devs = []
            t = 0.0
            for p, d in zip(pitch, dur):
                if p > 0 and d > 0:
                    a, b = int(t / hop_s), int((t + d) / hop_s)
                    seg_m = f0_midi[a:b]
                    seg_m = seg_m[seg_m > 0]
                    if seg_m.size:
                        devs.append(abs(float(np.median(seg_m)) - float(p)))
                t += d
            if devs:
                devs = np.asarray(devs)
                print(f"  |note - F0|     median {np.median(devs):.2f} st  "
                      f"mean {devs.mean():.2f}  >1st: {int((devs > 1).sum())}/{devs.size}  "
                      f"(agreement check; <0.5 is good)")

        out = os.path.join(args.out_dir, f"{seg['item_name']}_note.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in meta.items()}, f, ensure_ascii=False, indent=2, default=str)
        print(f"  saved           {out}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
