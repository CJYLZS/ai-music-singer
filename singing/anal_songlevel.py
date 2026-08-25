#!/usr/bin/env python
"""Task 3: song-level rankings and which INPUT property predicts quality.

Aggregates .tmp/anal/windows.csv to one row per song, parses eval.txt for the
SIM numbers, adds full-song input properties from target_f0.npy / lead, ranks
songs by each output metric and correlates every input property against the
output metrics (Spearman, n=14).
Writes .tmp/anal/songs.csv. Run: SoulX-Singer/.venv/bin/python singing/anal_songlevel.py
"""
import csv, json, os, re
from collections import defaultdict
import numpy as np
from anal_register import spearman

ROOT = '/home/rookie/voice'
SRC = f'{ROOT}/.tmp/anal/windows.csv'
OUT = f'{ROOT}/.tmp/anal/songs.csv'
FPS = 50
SONGS = ['云烟成雨', '亲亲', '冬眠', '去有风的地方', '情歌', '最后一页',
         '爱的回归线', '爱笑的眼睛', '白色毛衣', '绿色', '追光者',
         '那女孩对我说', '陪你看星星', '雨爱']


def parse_eval(song):
    """eval.txt: overall SIM mean/median + corr(pitch distance, SIM)."""
    p = f'{ROOT}/data/songs/{song}/eval.txt'
    if not os.path.exists(p):
        return dict(sim_mean=np.nan, sim_median=np.nan,
                    corr_pitch_sim=np.nan)
    t = open(p, errors='replace').read()
    m = re.search(r'overall SIM:\s*mean=([\d.]+)\s+median=([\d.]+)', t)
    c = re.search(r'corr\(pitch distance, SIM\)\s*=\s*(-?[\d.]+)', t)
    return dict(sim_mean=float(m.group(1)) if m else np.nan,
                sim_median=float(m.group(2)) if m else np.nan,
                corr_pitch_sim=float(c.group(1)) if c else np.nan)


def input_props(song):
    d = f'{ROOT}/data/songs/{song}'
    f0 = np.load(f'{d}/target_f0.npy')
    pf = np.load(f'{d}/prompt_ext_f0.npy')
    v = f0[f0 > 0]
    pv = pf[pf > 0]
    pm = float(np.median(pv))
    st = 12 * np.log2(v / pm)
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    iqr_st = float(12 * np.log2(max(p90, 1e-9) / max(p10, 1e-9)))
    lo = 12 * np.log2(max(p10, 1e-9) / pm)
    hi = 12 * np.log2(max(p90, 1e-9) / pm)
    return dict(in_med_f0=p50, in_p10_f0=p10, in_p90_f0=p90,
                in_iqr_st=iqr_st, in_lo_st=lo, in_hi_st=hi,
                in_share_2st=float((np.abs(st) <= 2).mean()),
                in_share_4st=float((np.abs(st) <= 4).mean()),
                in_voiced=float((f0 > 0).mean()),
                prompt_med_f0=pm)


def main():
    g = {}
    for r in csv.DictReader(open(SRC)):
        g.setdefault(r['song'], defaultdict(list))
        for k, val in r.items():
            try:
                g[r['song']][k].append(float(val))
            except ValueError:
                pass
    rows = []
    for s in SONGS:
        if s not in g:
            continue
        d = g[s]

        def mean(k):
            a = [x for x in d[k] if np.isfinite(x)]
            return float(np.mean(a)) if a else np.nan

        row = dict(song=s, n_win=len(d['mos']),
                   singmos_mean=mean('mos'), singmos_median=mean('mos'))
        mos = np.array([x for x in d['mos'] if np.isfinite(x)])
        row['singmos_median'] = float(np.median(mos))
        for k in ('excess', 'er_excess', 'cp_excess'):
            row[f'{k}_mean'] = mean(k)
        row['consensus_rate'] = mean('consensus')
        for k in ('lead_flat', 'lead_hnr', 'lead_hf', 'voiced_ratio'):
            row[f'{k}_mean'] = mean(k)
        rms = np.array([x for x in d['lead_rms_db'] if np.isfinite(x)])
        row['dyn_range_db'] = float(np.percentile(rms, 90) -
                                    np.percentile(rms, 10))
        stw = np.array([x for x in d['st_from_prompt'] if np.isfinite(x)])
        row['f0_mobility_st'] = float(np.std(stw))
        row.update(parse_eval(s))
        nm = json.load(open(f'{ROOT}/data/songs/{s}/natmos_report.json'))['files']
        orig = next((v.get('mean') for k, v in nm.items()
                     if k != '合成' and isinstance(v, dict)), np.nan)
        row['orig_mos_mean'] = float(orig) if orig is not None else np.nan
        row['mos_delta'] = (row['singmos_mean'] - row['orig_mos_mean']
                            if np.isfinite(row['orig_mos_mean']) else np.nan)
        row.update(input_props(s))
        rows.append(row)

    keys = list(rows[0])
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def rank(metric, high_good=True):
        vals = [(r[metric], r['song']) for r in rows
                if r.get(metric) == r.get(metric)]
        vals.sort(reverse=high_good)
        print(f'\nrank by {metric} ({"high" if high_good else "low"}=best):')
        for i, (v, s) in enumerate(vals, 1):
            mark = ' <<<' if s in ('情歌', '云烟成雨') else ''
            print(f'  {i:2d}. {s:8s} {v:.4f}{mark}')
        return {s: i for i, (v, s) in enumerate(vals, 1)}

    ranks = {}
    ranks['singmos_mean'] = rank('singmos_mean')
    ranks['singmos_median'] = rank('singmos_median')
    ranks['sim_mean'] = rank('sim_mean')
    ranks['excess_mean'] = rank('excess_mean', high_good=False)

    outs = ['singmos_mean', 'sim_mean', 'excess_mean', 'mos_delta']
    ins = ['in_med_f0', 'in_iqr_st', 'in_lo_st', 'in_hi_st',
           'in_share_2st', 'in_share_4st', 'in_voiced', 'lead_hnr_mean',
           'lead_flat_mean', 'lead_hf_mean', 'dyn_range_db',
           'f0_mobility_st']
    print(f'\nSpearman(input -> output), n={len(rows)}; '
          f'|rho|>=0.54 approx p<0.05:')
    print(f'{"input":14s}' + ''.join(f'{o:>16s}' for o in outs))
    for i in ins:
        cells = []
        for o in outs:
            rho, _ = spearman([r[i] for r in rows], [r[o] for r in rows])
            stars = '*' if abs(rho) >= 0.54 else ' '
            cells.append(f'{rho:+.3f}{stars}       ')
        print(f'{i:14s}' + ''.join(f'{c:>16s}' for c in cells))
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
