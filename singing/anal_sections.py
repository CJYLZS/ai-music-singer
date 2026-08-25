#!/usr/bin/env python
"""Task 4: verse vs chorus labelling from .lrc lyric-line repetition.

A lyric LINE is "repeated" if the same normalised text occurs in >=2 separate
blocks (runs of consecutive identical lines) whose starts are >=20 s apart;
blocks of repeated lines are chorus candidates -> merged chorus intervals.
Each 3 s window is CHORUS if >=50% overlapped by a chorus interval, VERSE if
unoverlapped, else MIXED. Songs without .lrc are skipped.
Known error modes: varied-wording repeat choruses missed; repeated hooks
(countdowns / post-chorus "哦~" lines) over-called chorus; instrumental
gaps inside a section default to VERSE.
Reads .tmp/anal/windows.csv. Writes .tmp/anal/sections_*.csv
Run: SoulX-Singer/.venv/bin/python singing/anal_sections.py
"""
import csv, glob, math, os, re
from collections import defaultdict
import numpy as np

ROOT = '/home/rookie/voice'
SRC = f'{ROOT}/.tmp/anal/windows.csv'
OUT_W = f'{ROOT}/.tmp/anal/sections_windows.csv'
OUT_S = f'{ROOT}/.tmp/anal/sections_summary.csv'
TAG = re.compile(r'^\[(\d+):(\d+(?:\.\d+)?)\]')
WORDTAG = re.compile(r'<\d+:\d+(?:\.\d+)?>')
GAP = 20.0
SEQ = 3          # consecutive lines that must co-recur to call a chorus


def norm(t):
    t = WORDTAG.sub('', t)
    t = re.sub(r'[（(【\[].*?[）)】\]]', '', t)
    return re.sub(r'[\s、，。！？!?.,·—\-~～"\'“”‘’:：;；]+', '', t)


def parse_lrc(path):
    lines, seen_t = [], set()
    for raw in open(path, encoding='utf-8', errors='replace'):
        s = raw.strip()
        if s.startswith('#'):
            continue
        ts = []
        rest = s
        while True:
            m = TAG.match(rest)
            if not m:
                break
            ts.append(int(m.group(1)) * 60 + float(m.group(2)))
            rest = rest[m.end():]
        tx = norm(rest)
        for t in ts:
            if len(tx) >= 4 and round(t, 2) not in seen_t:
                seen_t.add(round(t, 2))
                lines.append((t, tx))
    return sorted(lines)


def chorus_intervals(lines, song_end):
    n = len(lines)
    seed = np.zeros(n, bool)
    for i in range(n - SEQ + 1):
        seg = [lines[k][1] for k in range(i, i + SEQ)]
        if len(set(seg)) < SEQ:      # kills periodic one-line hooks
            continue
        for j in range(i + SEQ, n - SEQ + 1):
            if lines[j][0] - lines[i][0] >= GAP and \
                    [lines[k][1] for k in range(j, j + SEQ)] == seg:
                seed[i:i + SEQ] = True
                break
    iv, i = [], 0
    while i < n:
        if seed[i]:
            j = i
            while j < n and seed[j]:
                j += 1
            end = lines[j][0] if j < n else min(lines[i][0] + 12, song_end)
            if end - lines[i][0] > 0.5:
                iv.append((lines[i][0], end))
            i = j
        else:
            i += 1
    merged = []
    for a, b in sorted(iv):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def label(iv, t0, win=3.0):
    ov = sum(max(0.0, min(b, t0 + win) - max(a, t0)) for a, b in iv)
    frac = ov / win
    return ('CHORUS' if frac >= 0.5 else 'VERSE' if frac <= 0 else 'MIXED',
            frac)


def main():
    g = {}
    for r in csv.DictReader(open(SRC)):
        try:
            g.setdefault(r['song'], []).append(
                (float(r['t0']), float(r['mos']), float(r['excess'])))
        except ValueError:
            pass
    wrows, srows, dmos, dexc = [], [], [], []
    for song, wins in sorted(g.items()):
        lrcs = glob.glob(f'{ROOT}/data/songs/{song}.lrc')
        if not lrcs:
            print(f'{song}: no .lrc -> skipped')
            continue
        lines = parse_lrc(lrcs[0])
        if len(lines) < SEQ + 1:
            print(f'{song}: .lrc has no usable lyrics -> skipped')
            continue
        song_end = max(t for t, _, _ in wins) + 3.0
        iv = chorus_intervals(lines, song_end)
        lab = {t: label(iv, t) for t, _, _ in wins}

        def mstat(sel):
            ms = [m for t, m, _ in wins if lab[t][0] == sel]
            es = [e for t, _, e in wins if lab[t][0] == sel]
            return (np.mean(ms) if ms else np.nan,
                    np.mean(es) if es else np.nan)

        mv, ev = mstat('VERSE')
        mc, ec = mstat('CHORUS')
        mm, em = mstat('MIXED')
        n = lambda s: sum(1 for t, _, _ in wins if lab[t][0] == s)
        print(f'{song}: lines={len(lines)} intervals={len(iv)} '
              f'| V={n("VERSE")} M={n("MIXED")} '
              f'C={n("CHORUS")} | MOS v={mv:.3f} c={mc:.3f} '
              f'(d={mc - mv:+.3f}) EXC v={ev:.3f} c={ec:.3f} (d={ec - ev:+.3f})')
        if np.isfinite(mc - mv):
            dmos.append(mc - mv)
            dexc.append(ec - ev)
        for sel, mos_v, exc_v in (('VERSE', mv, ev), ('MIXED', mm, em),
                                  ('CHORUS', mc, ec)):
            srows.append(dict(song=song, section=sel, n=n(sel),
                              mos=mos_v, excess=exc_v))
        for t, _, _ in wins:
            wrows.append(dict(song=song, t0=t, section=lab[t][0],
                              chorus_frac=f'{lab[t][1]:.2f}'))
    for path, rows in ((OUT_W, wrows), (OUT_S, srows)):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('wrote', path)
    dmos, dexc = np.array(dmos), np.array(dexc)

    def sig(x):
        pos = int((x > 0).sum())
        p = sum(math.comb(len(x), k) for k in range(min(pos, len(x) - pos)
                                                    + 1)) / 2 ** len(x) * 2
        return min(p, 1.0)

    print(f'\nAcross {len(dmos)} lrc songs: chorus-verse dMOS mean '
          f'{dmos.mean():+.4f} ({(dmos > 0).sum()}/{len(dmos)} positive, '
          f'sign-test p={sig(dmos):.3f}); dEXC mean {dexc.mean():+.4f} '
          f'sign-test p={sig(dexc):.3f}')


if __name__ == '__main__':
    main()
