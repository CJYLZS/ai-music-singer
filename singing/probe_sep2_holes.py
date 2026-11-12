"""probe: sep2/lead 是否在局部「丢人声」——二次分离把同一人声判给了 backing？

背景（2026-11，雪之华）：成品人声在 3:29 附近突然变弱。听感上像是 karaoke 二分
（sep1/vocal.wav -> sep2/lead.wav + sep2/backing.wav）在短片段里把主唱拆到了
backing。本脚本只做测量，不改任何产物。

三个假设，各自可判：
  H1 能量守恒：V1 ≈ L2 + B2（逐帧），并且出问题的位置 B2 显著上升而不是 V1 本身弱。
  H2 同一人声：问题窗内 spectrogram(V1) 与 spectrogram(B2) 的相关明显高于正常窗。
  H3 传导：SVC 输出 vocal.wav 的电平凹陷跟随 L2/V1 的比例（源缺内容 -> 输出缺电平），
     并且 mix 的 --match-lead 以 sep2/lead.wav 为参照会二次压低。

用法:
  GPT-SoVITS/.venv/bin/python singing/probe_sep2_holes.py data/songs/雪之华
"""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf


def load_mono(path: str):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    return x.mean(axis=1), sr


def frame_rms(x: np.ndarray, hop: int, win: int) -> np.ndarray:
    n = max(0, (len(x) - win) // hop + 1)
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1))


def db(x: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, floor))


def stft_mag(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    w = np.hanning(n_fft)
    n = max(0, (len(x) - n_fft) // hop + 1)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1))


def spec_corr(a: np.ndarray, b: np.ndarray) -> float:
    la, lb = np.log1p(a * 1000.0), np.log1p(b * 1000.0)
    la = la - la.mean()
    lb = lb - lb.mean()
    d = np.sqrt((la * la).sum() * (lb * lb).sum())
    return float((la * lb).sum() / d) if d > 0 else 0.0


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/songs/雪之华"
    v1, sr = load_mono(f"{root}/sep1/vocal.wav")
    l2, sr2 = load_mono(f"{root}/sep2/lead.wav")
    b2, sr3 = load_mono(f"{root}/sep2/backing.wav")
    acc, sr4 = load_mono(f"{root}/sep1/acc.wav")
    out, sr5 = load_mono(f"{root}/vocal.wav")
    print(f"sep1/vocal {len(v1)/sr:.2f}s @{sr}  sep2/lead {len(l2)/sr2:.2f}s "
          f"sep2/backing {len(b2)/sr3:.2f}s  svc vocal {len(out)/sr5:.2f}s")
    assert sr == sr2 == sr3 == sr4

    # 对齐：mel-band-roformer 应当零延迟，确认一下
    a, b = v1[60 * sr:70 * sr], l2[60 * sr:70 * sr]
    n = 1 << int(np.ceil(np.log2(2 * len(a))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lag = int(np.argmax(c) - (len(a) - 1))
    print(f"cross-corr lag(L2 vs V1) = {lag} samples ({lag/sr*1000:.2f} ms), "
          f"corr={c.max()/np.sqrt((a*a).sum()*(b*b).sum()):.3f}")

    hop = int(0.05 * sr)
    win = int(0.20 * sr)
    ev = frame_rms(v1, hop, win)
    el = frame_rms(l2, hop, win)
    eb = frame_rms(b2, hop, win)
    es = frame_rms(l2 + b2, hop, win)
    ea = frame_rms(acc, hop, win)
    m = min(len(ev), len(el), len(eb), len(es))
    t = np.arange(m) * 0.05 + 0.1

    print(f"\n[全局] V1 peak {db(np.abs(v1).max()):.1f} dBFS  rms {db(np.sqrt((v1**2).mean())):.1f}")
    print(f"        L2 rms {db(np.sqrt((l2**2).mean())):.1f}  B2 rms {db(np.sqrt((b2**2).mean())):.1f}")
    act = db(ev) > db(np.abs(v1).max()) - 45
    print(f"        V1 有声帧 {act.sum()}/{m} ({act.mean()*100:.1f}%)")
    # H1: 能量守恒
    cons = db(es[act]) - db(ev[act])
    print(f"        |L2+B2| vs |V1| 有声帧: mean {cons.mean():+.2f} dB, "
          f"p5 {np.percentile(cons,5):+.2f}, p95 {np.percentile(cons,95):+.2f}")

    ratio = db(el) - db(ev)
    bratio = db(eb) - db(ev)
    dips = act & (ratio < -6.0)
    print(f"        L2 相对 V1 < -6dB 的有声帧: {dips.sum()} ({dips.sum()/max(act.sum(),1)*100:.1f}% of active), "
          f"合计 {dips.sum()*0.05:.1f}s")

    # 合并连续段（间隔 < 0.25s 视为同一段，最短 0.15s）
    spans = []
    i = 0
    while i < m:
        if not dips[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and dips[j + 1]:
            j += 1
        spans.append([i, j])
        i = j + 1
    merged = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) * 0.05 < 0.25:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    merged = [s for s in merged if (s[1] - s[0] + 1) * 0.05 >= 0.15]

    print(f"\n[H1/H2] {-len(merged)} 段下陷（>=0.15s, 合并 <0.25s 间隙），按时间：")
    print("  t0      t1     dur   L2-V1(dB)          B2-V1(dB)          |L2+B2|-|V1|  B2-L2   specCorr(V1,L2) specCorr(V1,B2)")
    rows = []
    for (i0, i1) in merged:
        sl = slice(i0, i1 + 1)
        r = db(el[sl]) - db(ev[sl])
        br = db(eb[sl]) - db(ev[sl])
        cs = db(es[sl]) - db(ev[sl])
        bl = db(eb[sl]) - db(el[sl])
        s0 = int(i0 * 0.05 * sr)
        s1 = int(min(i1 * 0.05 * sr + win, len(v1)))
        A = stft_mag(v1[s0:s1])
        L = stft_mag(l2[s0:s1])
        B = stft_mag(b2[s0:s1])
        rows.append((t[i0], t[i1], (i1 - i0 + 1) * 0.05, r.mean(), r.min(), br.mean(),
                     cs.mean(), bl.mean(), spec_corr(A, L), spec_corr(A, B)))

    for (t0, t1, dur, rm, rmin, brm, csm, blm, cL, cB) in rows:
        print(f"  {int(t0//60)}:{t0%60:05.2f} {int(t1//60)}:{t1%60:05.2f} {dur:5.2f}  "
              f"{rm:+6.2f}/{rmin:+6.2f}     {brm:+6.2f}            {csm:+6.2f}        {blm:+6.2f}   "
              f"{cL:6.3f}          {cB:6.3f}")

    # 基线：正常有声窗的谱相关
    normal = act & (ratio > -2.0)
    step = int(1.0 / 0.05)
    cLb, cBb = [], []
    for i0 in range(0, m - step, step):
        sl = slice(i0, i0 + step)
        if not normal[sl].all():
            continue
        s0 = int(i0 * 0.05 * sr)
        s1 = s0 + int(1.0 * sr)
        A = stft_mag(v1[s0:s1])
        cLb.append(spec_corr(A, stft_mag(l2[s0:s1])))
        cBb.append(spec_corr(A, stft_mag(b2[s0:s1])))
        if len(cLb) >= 200:
            break
    print(f"\n[基线] 正常 1s 窗（L2-V1 > -2dB）n={len(cLb)}: "
          f"specCorr(V1,L2) {np.mean(cLb):.3f}   specCorr(V1,B2) {np.mean(cBb):.3f}")

    # H3: SVC 输出是否跟随源凹陷
    eo = frame_rms(out, int(0.05 * sr5), int(0.20 * sr5))
    no = min(len(eo), m)
    # 24k -> 帧号同一时间轴（都是 50ms 帧），直接对齐帧号
    eo = eo[:no]
    d_out = db(eo) - db(ev[:no])
    both = act[:no]
    r_ratio = db(el[:no]) - db(ev[:no])
    corr = float(np.corrcoef(r_ratio[both], d_out[both])[0, 1])
    print(f"\n[H3] corr(SVC输出-V1, L2-V1) = {corr:+.3f}  (n={both.sum()})")
    for (i0, i1) in merged:
        sl = slice(i0, min(i1 + 1, no))
        print(f"      {t[i0]:6.1f}s  hole  L2-V1 {np.mean(db(el[sl])-db(ev[sl])):+6.2f}dB   "
              f"SVC-V1 {np.mean(db(eo[sl])-db(ev[sl])):+6.2f}dB")

    # 3:29 附近逐帧打印
    print("\n[3:29 附近] t, V1, L2, B2 (dBFS)")
    i0 = int((209.0 - 0.1) / 0.05)
    for k in range(i0, min(i0 + 60, m), 4):
        print(f"  {t[k]:7.2f}  V1 {db(ev[k]):7.1f}  L2 {db(el[k]):7.1f}  B2 {db(eb[k]):7.1f}  "
              f"L2-V1 {db(el[k])-db(ev[k]):+6.1f}")

    # 如果换用 sep1/vocal 做源：源里多出来的是什么
    print("\n[源替换影响] 正常段剖面")
    nm = act & (ratio > -2.0)
    print(f"  正常帧: (L2+B2)-V1 {np.mean(db(es[nm])-db(ev[nm])):+.2f}dB, "
          f"B2-V1 {np.mean(db(eb[nm])-db(ev[nm])):+.2f}dB, "
          f"acc-V1 {np.mean(db(ea[nm])-db(ev[nm])):+.2f}dB (分离残留能量)")
    print(f"  下陷帧: B2-V1 {np.mean(db(eb[dips])-db(ev[dips])):+.2f}dB  "
          f"(正常帧 {np.mean(db(eb[nm])-db(ev[nm])):+.2f}dB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
