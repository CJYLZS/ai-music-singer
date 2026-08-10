#!/usr/bin/env python3
"""Extract a beat grid from a song for beat-locked character sway.

Writes a small JSON that render.mjs consumes via --sway:

    {"t0": 1.46, "barLen": 1.9673, "bpm": 121.99, "beatsPerBar": 4, ...}

Run with the GPT-SoVITS venv (the only one carrying beat_this's deps):

    GPT-SoVITS/.venv/bin/python singing/beat_grid.py data/songs/绿色.flac \
        -o data/songs/绿色.beats.json

Why a global grid rather than the raw downbeat list
---------------------------------------------------
Beat This! detects downbeats well but intermittently misses one, leaving a bar that
measures twice as long. Driving phase by interpolating between consecutive downbeats makes
the sway rate swing 2.0-11.2x across a song (measured on 5 songs) -- visible as the
character stumbling. Fitting one line t = t0 + k*barLen instead, with outlier rejection,
matches the detected downbeats to p90 ~15ms at ~0 drift, because pop tempo is machine
steady. So the grid is both smoother AND more accurate here.

Feed the FULL MIX, not a separated vocal: beat tracking wants percussive cues.
"""
import argparse, json, sys
import numpy as np


def robust_grid(downbeats, bar0, iters=12):
    """Least-squares fit t = t0 + k*barLen, rejecting outliers.

    Outliers are real: a missed downbeat puts a point half a bar off the grid. Without
    rejection those drag barLen and the residual p90 rises ~10x.
    """
    t0, L = float(downbeats[0]), float(bar0)
    keep = np.ones(len(downbeats), bool)
    for _ in range(iters):
        k = np.round((downbeats - t0) / L)
        r = downbeats - (t0 + k * L)
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med)) + 1e-9
        keep = np.abs(r - med) < max(3 * mad, 0.05)
        if keep.sum() < 4:
            keep = np.ones(len(downbeats), bool)
            break
        A = np.vstack([np.ones(keep.sum()), k[keep]]).T
        t0, L = np.linalg.lstsq(A, downbeats[keep], rcond=None)[0]
    k = np.round((downbeats - t0) / L)
    return float(t0), float(L), (downbeats - (t0 + k * L)), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="full mix (flac/wav/mp3), NOT an isolated vocal")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--checkpoint", default="final0", help="final0 (78MB) | small0 (8MB)")
    ap.add_argument("--device", default="cuda")
    # Octave resolution. The tracker's level is arbitrary; these decide which one we call
    # "the beat". Verify by ear -- no automatic test settles it (see comments below).
    ap.add_argument("--prefer-bpm", type=float, default=75.0,
                    help="fold the beat into prefer-bpm/sqrt2 .. prefer-bpm*sqrt2 (53-106 "
                         "at the default 75), matching the slow felt pulse of ballad pop")
    ap.add_argument("--beat-mult", type=float, default=None,
                    help="skip the fold and scale the tracker beat directly (0.5 = twice as "
                         "fast, 2 = half as fast)")
    ap.add_argument("--grid-beats", type=int, default=2,
                    help="beats spanned by one tracker downbeat interval before folding; "
                         "only affects the pre-fold starting point")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    args = ap.parse_args()

    from beat_this.inference import File2Beats

    beats, downbeats = File2Beats(
        checkpoint_path=args.checkpoint, device=args.device, dbn=False)(args.audio)
    beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
    if len(downbeats) < 4:
        sys.exit(f"only {len(downbeats)} downbeats found; too few to fit a grid")

    bars = np.diff(downbeats)
    bm = np.median(bars)
    inliers = bars[np.abs(bars - bm) < 0.2 * bm]
    if not len(inliers):
        sys.exit("bar lengths too irregular to fit a grid")
    t0, gridLen, resid, keep = robust_grid(downbeats, inliers.mean())

    # `gridLen` is the spacing of what the tracker CALLS a bar, which is not necessarily a
    # musical bar. On 绿色 the tracker locks onto the eighth-note level, reporting 4 beats
    # per "bar" across only 2 real beats, so the naive 60*4/gridLen gives 121.99 BPM for a
    # song that is actually 61. Beat-level (octave) ambiguity is a known-unsolvable
    # perceptual problem, and every automatic test tried here was inconclusive on this song:
    #   - onset strength does not alternate between odd/even tracker beats (ratio 1.008),
    #     so the extra beats are not weak off-beats -- consistent with EITHER reading
    #   - onset autocorrelation peaks at every multiple (0.49/0.98/1.97/3.92s all +0.44..
    #     +0.55), so its ranking cannot choose a level
    #   - Fourier tempogram magnitude *increases* monotonically toward 244 BPM (7.8 / 25.1 /
    #     52.0 / 68.2 at 30.5 / 61 / 122 / 244), i.e. it prefers ever-finer subdivisions
    #   - librosa's own estimate flips with its prior: 123.05 at start_bpm>=100, 61.52 below
    # So resolve it with an explicit perceptual prior instead of pretending to measure it:
    # fold the beat period into a preferred range around --prefer-bpm by halving/doubling.
    # 61 was confirmed by ear for 绿色; --beat-mult overrides when the prior guesses wrong.
    beatLen = gridLen / max(1, args.grid_beats)
    if args.beat_mult:
        beatLen *= args.beat_mult
    else:
        lo = args.prefer_bpm / np.sqrt(2.0)      # geometric window, so the fold is unbiased
        while 60.0 / beatLen > args.prefer_bpm * np.sqrt(2.0):
            beatLen *= 2.0
        while 60.0 / beatLen < lo:
            beatLen /= 2.0

    # Beats per bar: the tracker's own count is unreliable (it switches between quarters and
    # eighths at section boundaries -- on 冬眠, 176 beats at 0.48s vs 181 at 0.93s, an exact
    # 2:1 split), and after the octave fold above it no longer refers to our beat level.
    # Assume 4/4, which is right for essentially all Mandarin pop, and report the tracker's
    # value separately so a disagreement is visible rather than silently absorbed.
    cnt = [int(((beats >= downbeats[i] - 1e-6) & (beats < downbeats[i + 1] - 1e-6)).sum())
           for i in range(len(downbeats) - 1)]
    vals, freq = np.unique(cnt, return_counts=True)
    trackerBpb = int(vals[freq.argmax()]) if len(vals) else 0
    bpb = args.beats_per_bar

    # Shift t0 to the first grid point at or before t=0 so phase is defined from the start.
    t0 -= gridLen * np.ceil(t0 / gridLen)

    drift = float(np.polyfit(downbeats[keep], resid[keep], 1)[0] * downbeats[-1]) \
        if keep.sum() > 2 else 0.0
    bpm = 60.0 / beatLen
    out = {
        "audio": args.audio,
        "t0": round(t0, 6),
        # beatLen is the unit the renderer works in; barLen is kept for readability only.
        "beatLen": round(beatLen, 6),
        "barLen": round(beatLen * bpb, 6),
        "bpm": round(bpm, 3),
        "beatsPerBar": bpb,
        # Diagnostics for the octave decision, so a wrong guess is traceable.
        "trackerGridLen": round(gridLen, 6),
        "trackerBeatsPerGrid": trackerBpb,
        "octaveFold": round(beatLen / (gridLen / max(1, args.grid_beats)), 4),
        "nBeats": len(beats),
        "nDownbeats": len(downbeats),
        "gridInliers": int(keep.sum()),
        "residP90": round(float(np.percentile(np.abs(resid[keep]), 90)), 5),
        "driftSec": round(drift, 5),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"{args.audio}\n  {bpm:.2f} BPM  beat {beatLen:.4f}s  bar {beatLen*bpb:.4f}s "
          f"({bpb}/bar)  t0={t0:.4f}s")
    print(f"  tracker grid {gridLen:.4f}s x{trackerBpb} -> octave fold x{out['octaveFold']}"
          f"  ({keep.sum()}/{len(downbeats)} inliers, resid p90 "
          f"{out['residP90']*1000:.1f}ms, drift {drift*1000:.1f}ms)")
    if out["residP90"] > 0.05:
        print("  WARNING: residual > 50ms -- tempo may not be steady; check by ear")
    if out["octaveFold"] != 1.0:
        print(f"  NOTE: folded to the {args.prefer_bpm}+-41% window. Octave is a perceptual "
              f"choice, not measurable -- verify by ear, override with --beat-mult 0.5/2.")


if __name__ == "__main__":
    main()
