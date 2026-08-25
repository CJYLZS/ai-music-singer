#!/usr/bin/env python
"""Task 2: does low register predict worse naturalness / more leakage?

Per song (and pooled) Spearman correlations of window MOS and leakage excess
against register/breathiness features, plus a low-vs-high register tercile
split by window median F0.
Reads .tmp/anal/windows.csv, writes .tmp/anal/register_corr.csv,
.tmp/anal/register_terciles.csv. Pure numpy (Spearman = Pearson on ranks).
Run: SoulX-Singer/.venv/bin/python singing/anal_register.py
"""
import csv, math, os
from collections import defaultdict
import numpy as np

ROOT = '/home/rookie/voice'
SRC = f'{ROOT}/.tmp/anal/windows.csv'
OUT_C = f'{ROOT}/.tmp/anal/register_corr.csv'
OUT_T = f'{ROOT}/.tmp/anal/register_terciles.csv'
PAIRS = [('mos', 'st_from_prompt'), ('excess', 'st_from_prompt'),
         ('mos', 'f0_range_st'), ('mos', 'lead_hnr'), ('mos', 'lead_hf'),
         ('mos', 'lead_flat'), ('excess', 'lead_hnr'),
         ('mos', 'voiced_ratio'), ('excess', 'voiced_ratio')]


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    sa = a[order]
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 8:
        return np.nan, n
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]), n


def load():
    g = defaultdict(dict)
    for r in csv.DictReader(open(SRC)):
        for k, v in r.items():
            if k == 'song':
                continue
            try:
                x = float(v)
            except ValueError:
                x = np.nan
            if k == 't0':
                g[r['song']].setdefault('t0', []).append(x)
            else:
                g[r['song']].setdefault(k, []).append(x)
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in g.items()}


def main():
    g = load()
    songs = sorted(g)
    rows = []
    for name in songs + ['POOLED']:
        d = g[name] if name != 'POOLED' else {
            k: [v for s in songs for v in g[s][k]] for k in g[songs[0]]}
        for a, b in PAIRS:
            rho, n = spearman(d[b], d[a])
            rows.append(dict(scope=name, outcome=a, feature=b, rho=rho, n=n))
            print(f'{name:8s} {a:6s} ~ {b:14s} rho={rho:+.3f} n={n}')
    with open(OUT_C, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scope', 'outcome', 'feature',
                                          'rho', 'n'])
        w.writeheader()
        w.writerows(rows)

    # terciles by median F0 within each song
    trows, dm, de = [], [], []
    for s in songs:
        st = np.array(g[s]['st_from_prompt'], float)
        mos = np.array(g[s]['mos'], float)
        exc = np.array(g[s]['excess'], float)
        m = np.isfinite(st) & np.isfinite(mos) & np.isfinite(exc)
        st, mos, exc = st[m], mos[m], exc[m]
        q1, q2 = np.quantile(st, [1 / 3, 2 / 3])
        lo, hi = st <= q1, st >= q2
        d_mos = mos[hi].mean() - mos[lo].mean()
        d_exc = exc[hi].mean() - exc[lo].mean()
        dm.append(d_mos)
        de.append(d_exc)
        trows.append(dict(song=s, n_low=int(lo.sum()), n_high=int(hi.sum()),
                          medf0_low=float(np.median(st[lo])),
                          medf0_high=float(np.median(st[hi])),
                          mos_low=mos[lo].mean(), mos_high=mos[hi].mean(),
                          d_mos=d_mos, exc_low=exc[lo].mean(),
                          exc_high=exc[hi].mean(), d_exc=d_exc))

        def fmt(x):
            return f'{x:.3f}'

        print(f'{s}: LOW(f0~{trows[-1]["medf0_low"]:.1f}st) MOS='
              f'{fmt(trows[-1]["mos_low"])} EXC={fmt(trows[-1]["exc_low"])}'
              f' | HIGH(f0~{trows[-1]["medf0_high"]:.1f}st) MOS='
              f'{fmt(trows[-1]["mos_high"])} EXC={fmt(trows[-1]["exc_high"])}'
              f' | dMOS={d_mos:+.3f} dEXC={d_exc:+.3f}')
    with open(OUT_T, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    dm, de = np.array(dm), np.array(de)

    def sigtest(x):  # two-sided sign test
        pos = int((x > 0).sum())
        n = len(x)
        p = sum(math.comb(n, k) for k in range(min(pos, n - pos) + 1)) \
            / 2 ** n * 2
        return min(p, 1.0)

    print(f'\nAcross {len(dm)} songs: mean dMOS(high-low)='
          f'{dm.mean():+.4f} (range {dm.min():+.3f}..{dm.max():+.3f}, '
          f'{(dm > 0).sum()}/{len(dm)} positive, sign-test p={sigtest(dm):.3f})')
    print(f'mean dEXC(high-low)={de.mean():+.4f} '
          f'({(de < 0).sum()}/{len(de)} negative i.e. less leakage high, '
          f'sign-test p={sigtest(-de):.3f})')


if __name__ == '__main__':
    main()
