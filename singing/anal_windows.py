#!/usr/bin/env python
"""Task 1: build a per-window table across all SVC songs.

Joins, per 3 s window: leak_report.json (original-singer leakage excess,
eres2net + campplus), natmos_windows.csv (SingMOS of the 合成 vocal),
target_f0.npy (register features) and spectral/breathiness features of the
ORIGINAL lead (sep2/lead.wav, resampled to 16 kHz for analysis).

Read-only on data/. Writes .tmp/anal/windows.csv
Run: SoulX-Singer/.venv/bin/python singing/anal_windows.py
"""
import csv, json, os
import numpy as np
import librosa
import soundfile as sf  # noqa: F401  (availability check)

ROOT = '/home/rookie/voice'
OUT = f'{ROOT}/.tmp/anal/windows.csv'
WIN = 3.0
FPS = 50  # f0 frames per second
SONGS = ['云烟成雨', '亲亲', '冬眠', '去有风的地方', '情歌', '最后一页',
         '爱的回归线', '爱笑的眼睛', '白色毛衣', '绿色', '追光者',
         '那女孩对我说', '陪你看星星', '雨爱']

COLS = ['song', 't0', 'mos', 'er_excess', 'cp_excess', 'excess', 'consensus',
        'med_f0_hz', 'voiced_ratio', 'st_from_prompt', 'f0_range_st',
        'lead_flat', 'lead_hnr', 'lead_hf', 'lead_rms_db']


def prompt_median(song):
    pf = np.load(f'{ROOT}/data/songs/{song}/prompt_ext_f0.npy')
    v = pf[pf > 0]
    return float(np.median(v))


def f0_feats(f0, t0, pm):
    i0, i1 = int(round(t0 * FPS)), int(round((t0 + WIN) * FPS))
    seg = f0[max(i0, 0):min(i1, len(f0))]
    vr = float((seg > 0).mean()) if len(seg) else np.nan
    v = seg[seg > 0]
    out = dict(med_f0_hz=np.nan, voiced_ratio=vr, st_from_prompt=np.nan,
               f0_range_st=np.nan)
    if len(v) >= 5:
        out['med_f0_hz'] = float(np.median(v))
        out['st_from_prompt'] = float(12 * np.log2(out['med_f0_hz'] / pm))
        p10, p90 = np.percentile(v, [10, 90])
        if p10 > 0:
            out['f0_range_st'] = float(12 * np.log2(max(p90, 1e-9) / p10))
    return out


def lead_frame_feats(path):
    """Per-frame features of the original lead at 16 kHz."""
    y, sr = librosa.load(path, sr=16000, mono=True)
    hop, nfft = 512, 1024
    mag = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop))
    pow_spec = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    times = np.arange(mag.shape[1]) * hop / sr
    flat = librosa.feature.spectral_flatness(S=mag)[0]
    hf = pow_spec[freqs > 4000].sum(0) / (pow_spec.sum(0) + 1e-12)
    rms = np.sqrt(pow_spec.sum(0) / nfft)
    rmsdb = 20 * np.log10(rms + 1e-10)
    # HNR-ish proxy: peak of normalised autocorrelation for f0 in 60-500 Hz
    fr = librosa.util.frame(np.pad(y, nfft // 2), frame_length=nfft,
                            hop_length=hop)
    N = 2048
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr * np.hamming(nfft)[:, None],
                                         N, axis=0)) ** 2, N, axis=0)[:268]
    ac /= (ac[0:1] + 1e-12)
    lo, hi = int(sr / 500), int(sr / 60)
    per = np.clip(ac[lo:hi].max(axis=0), 0, 1)
    return dict(times=times.astype(float), flat=flat, hf=hf, hnr=per,
                rmsdb=rmsdb)


def agg_frames(feats, t0):
    i0 = np.searchsorted(feats['times'], t0, side='left')
    i1 = np.searchsorted(feats['times'], t0 + WIN, side='left')
    if i1 <= i0:
        return dict.fromkeys(['lead_flat', 'lead_hnr', 'lead_hf',
                              'lead_rms_db'], np.nan)
    return dict(lead_flat=float(np.mean(feats['flat'][i0:i1])),
                lead_hnr=float(np.mean(feats['hnr'][i0:i1])),
                lead_hf=float(np.mean(feats['hf'][i0:i1])),
                lead_rms_db=float(np.median(feats['rmsdb'][i0:i1])))


def main():
    rows = []
    for song in SONGS:
        d = f'{ROOT}/data/songs/{song}'
        if not all(os.path.exists(f'{d}/{f}') for f in
                   ('vocal.wav', 'leak_report.json', 'natmos_report.json')):
            print(f'[skip] {song}: missing reports/vocal')
            continue
        leak = json.load(open(f'{d}/leak_report.json'))
        lw = {round(w['t0'], 1): w for w in leak['windows']}
        mos = {}
        for r in csv.DictReader(open(f'{d}/natmos_windows.csv')):
            if r['label'] in ('合成', 'synth'):
                mos[round(int(r['window_idx']) * WIN, 1)] = \
                    float(r['singmos_mos'])
        pm = prompt_median(song)
        f0 = np.load(f'{d}/target_f0.npy')
        lf = lead_frame_feats(f'{d}/sep2/lead.wav')
        n = 0
        for t0, w in sorted(lw.items()):
            if t0 not in mos:
                continue
            row = dict(song=song, t0=t0, mos=mos[t0],
                       er_excess=w['er_excess'], cp_excess=w['cp_excess'],
                       excess=(w['er_excess'] + w['cp_excess']) / 2,
                       consensus=int(bool(w.get('consensus'))))
            row.update(f0_feats(f0, t0, pm))
            row.update(agg_frames(lf, t0))
            rows.append(row)
            n += 1
        print(f'{song}: {n} joined windows (of {len(lw)} leak / '
              f'{len(mos)} mos), prompt_med={pm:.1f}Hz')
    with open(OUT, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (r[k] if isinstance(r[k], str) else
                             f'{r[k]:.6g}' if r[k] == r[k] else '')
                         for k in COLS})
    print(f'wrote {len(rows)} rows -> {OUT}')


if __name__ == '__main__':
    main()
