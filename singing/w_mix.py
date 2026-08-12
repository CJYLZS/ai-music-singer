"""人声 + 伴奏混音（SoulX-Singer venv 运行）。

SVC 输出 24kHz 单声道，伴奏 44.1kHz 立体声：重采样人声、对齐长度、混音、写 PCM16。

四个默认行为，都可关：

1. **`--match-lead`（默认开）人声逐帧 RMS 对齐原曲主唱。** SVC 输出的电平包络与原唱
   不一致：乐句内会有 3-5dB 的下陷，而 `--match-acc` 的 5s 窗跟不上这种快速抖动，于是
   混音后这些位置人声比原曲更弱。用 `sep2/lead.wav` 做逐帧参照（200ms 平滑）把包络
   校正回原唱的走向。实测云烟成雨：3s 窗「比原曲弱」的窗从 55.9% → 3.2%，最差
   −5.67dB → −0.39dB，且动态范围与包络相关都**更接近**原曲（22.5→20.4dB / r 0.86→0.94）
   —— 它是在恢复原唱动态，不是压扁。`--no-match-lead` 关闭。
2. **`--match-acc`（默认开）人声增益跟随伴奏 RMS 包络。** 歌曲高潮段伴奏通常比主歌
   响 10-15dB（编曲加厚），而 SVC 人声全程电平平稳 —— 固定增益必然导致后半段人声被
   伴奏压过。原曲不这样是因为母带做过压缩，而分离出的 acc 保留了那层压缩、vocal 没有。
   `--no-match-acc` 回到固定增益。**与 1 是两级不同时间尺度**：这一级管慢变的段落平衡，
   上一级管乐句内的快速抖动。
3. **`--limit-db`（默认 -0.5）软限幅代替硬削波。** 人声抬升后原始和峰值会超过 0dBFS，
   `np.clip` 会把增益换成可听失真。传 0 关闭。
4. **保留伴奏立体声，人声居中。** 旧版把伴奏 `mean(axis=0)` 压成单声道再复制成假立体声
   （L/R 相关 1.000000），丢掉了伴奏的空间信息。`--mono` 输出真单声道。
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def _frame_rms(x: np.ndarray, hop: int) -> np.ndarray:
    """逐 hop 样本的 RMS（不重叠分帧）。"""
    n = len(x) // hop * hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    f = x[:n].astype(np.float64).reshape(-1, hop)
    return np.sqrt(np.mean(f * f, axis=1))


def lead_norm_curve(
    voc: np.ndarray,
    lead: np.ndarray,
    sr: int,
    *,
    smooth_sec: float,
    clip_db: float,
    gate_db: float,
) -> tuple[np.ndarray, dict]:
    """把人声的逐帧 RMS 包络对齐到原曲主唱，返回（采样级增益曲线, 诊断）。

    这是「治本」的一级：SVC 输出的电平抖动在源头就被校正，不依赖后面 `match_gain_curve`
    那个 5s 窗去追。三个设计要点：

    - **只在两者都有声的帧上算比例。** 单侧有声的帧（SVC 多吐了气声 / 原唱有和声尾音）
      比例无意义；静音段用最近的有声值插值填充，避免边界处增益跳变。
    - **`clip_db` 双向限幅（默认 ±12dB）。** 不限幅时静音帧的比例会炸到几百倍，把
      SVC 的底噪连同抖动一起放大 —— 第一版就是这么错的（gain 撞到上限 5.0、峰值 +2.6dBFS）。
    - **`smooth_sec` 决定时间尺度，200ms 是实测最优。** 更慢（1s/2s）跟不上乐句内的下陷，
      std 从 0.43 退化到 0.81/1.03；更快会开始跟踪单个音素的自然起伏。
    """
    hop = max(1, int(sr * 0.02))  # 20ms 分析帧
    fv = _frame_rms(voc, hop)
    fl = _frame_rms(lead, hop)
    m = min(len(fv), len(fl))
    fv, fl = fv[:m], fl[:m]
    if m == 0:
        return np.ones(len(voc), dtype=np.float64), {}

    thr = 10 ** (gate_db / 20)
    both = (fv > fv.max() * thr) & (fl > fl.max() * thr)
    if not both.any():
        return np.ones(len(voc), dtype=np.float64), {}

    lim = 10 ** (clip_db / 20)
    r = np.ones(m, dtype=np.float64)
    r[both] = np.clip(fl[both] / np.maximum(fv[both], 1e-12), 1.0 / lim, lim)
    idx = np.where(both)[0]
    r = np.interp(np.arange(m), idx, r[idx])  # 静音段沿用最近的有声比例

    wl = max(3, int(round(smooth_sec / 0.02)))
    wl = wl if wl % 2 else wl + 1
    wl = min(wl, m if m % 2 else m - 1)
    if wl >= 3:
        from scipy.signal import savgol_filter

        r = np.clip(savgol_filter(r, wl, 1), 1.0 / lim, lim)

    curve = np.interp(
        np.arange(len(voc), dtype=np.float64), np.arange(m) * hop + hop // 2, r
    )
    d = 20 * np.log10(np.maximum(r[both], 1e-12))
    return curve, {
        "frames": int(both.sum()),
        "total": m,
        "min_db": float(d.min()),
        "max_db": float(d.max()),
        "mean_db": float(d.mean()),
    }


def match_gain_curve(
    voc: np.ndarray,
    acc_mono: np.ndarray,
    sr: int,
    *,
    win_sec: float,
    margin_db: float,
    max_boost_db: float,
    gate_db: float,
    vocal_gain: float,
    acc_gain: float,
) -> tuple[np.ndarray, list]:
    """人声跟随伴奏包络所需的时变增益，返回（采样级增益曲线, 诊断行）。

    两个设计要点：

    - **只在人声有声的帧上比较两者 RMS。** 直接对整窗取 RMS 会被间奏/句间换气拉低，
      于是算出虚高的抬升量；门限按人声自身峰值取相对值（`gate_db`）。
    - **只抬不压。** 伴奏安静的主歌段不把人声压下去 —— 那是原曲本来的平衡。
    """
    hop = max(1, int(sr * 0.05))  # 50ms 分析帧
    rv_f = _frame_rms(voc, hop)
    ra_f = _frame_rms(acc_mono, hop)
    m = min(len(rv_f), len(ra_f))
    rv_f, ra_f = rv_f[:m], ra_f[:m]
    if m == 0:
        return np.ones(len(voc), dtype=np.float64), []

    peak = float(np.abs(voc).max()) or 1.0
    active = rv_f > peak * (10 ** (gate_db / 20))

    per_win = max(1, int(round(win_sec / 0.05)))
    target = (10 ** (margin_db / 20)) * acc_gain / max(vocal_gain, 1e-9)
    ceil = 10 ** (max_boost_db / 20)

    xs: list[float] = []
    gs: list[float] = []
    rows: list = []
    for w0 in range(0, m, per_win):
        sl = slice(w0, min(w0 + per_win, m))
        act = active[sl]
        t0 = w0 * hop / sr
        # 该窗几乎无人声（intro/间奏/尾奏）-> 不设控制点，让插值跨过去
        if int(act.sum()) < max(1, per_win // 10):
            rows.append((t0, None, None, None))
            continue
        rv = math.sqrt(float(np.mean(rv_f[sl][act] ** 2)))
        ra = math.sqrt(float(np.mean(ra_f[sl][act] ** 2)))
        g = min(max(target * ra / max(rv, 1e-12), 1.0), ceil)
        xs.append((w0 + min(per_win, m - w0) / 2) * hop)
        gs.append(g)
        rows.append((t0, rv, ra, g))

    if not xs:  # 全曲都没有人声
        return np.ones(len(voc), dtype=np.float64), rows

    g_arr = np.asarray(gs, dtype=np.float64)
    if len(g_arr) >= 5:
        from scipy.signal import savgol_filter

        wl = min(7, len(g_arr) if len(g_arr) % 2 else len(g_arr) - 1)
        if wl >= 3:
            g_arr = np.clip(savgol_filter(g_arr, wl, 1), 1.0, ceil)

    curve = np.interp(np.arange(len(voc), dtype=np.float64), np.asarray(xs), g_arr)
    return curve, rows


def soft_limit(x: np.ndarray, sr: int, ceiling_db: float) -> tuple[np.ndarray, float, int]:
    """前视峰值限幅器。返回（结果, 施加的最大衰减dB, 触发帧数）。

    对超出 ceiling 的位置算所需衰减，取滑动窗最小值（前视，保证峰值到来前已经压下去），
    再用单极点释放平滑，避免逐样本增益跳变造成的失真。
    """
    ceiling = 10 ** (ceiling_db / 20)
    peak = np.abs(x).max(axis=1) if x.ndim > 1 else np.abs(x)
    if float(peak.max()) <= ceiling:
        return x, 0.0, 0

    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))

    # 前视：窗内取最小增益（attack 5ms）
    la = max(1, int(sr * 0.005))
    pad = np.pad(need, (la, la), mode="edge")
    k = la * 2 + 1
    # 滑动最小值：用步长化视图避免 O(n·k)
    from numpy.lib.stride_tricks import sliding_window_view

    env = sliding_window_view(pad, k).min(axis=1)[: len(need)]

    # 释放平滑（50ms 单极点），只在增益回升时生效
    rel = math.exp(-1.0 / (sr * 0.05))
    out = np.empty_like(env)
    g = 1.0
    for i in range(len(env)):
        e = env[i]
        g = e if e < g else e + (g - e) * rel
        out[i] = g

    n_hit = int((need < 1.0).sum())
    red = -20 * math.log10(max(float(out.min()), 1e-12))
    return (x * out[:, None] if x.ndim > 1 else x * out), red, n_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True, help="SVC 输出（24kHz 单声道）")
    ap.add_argument("--acc", required=True, help="伴奏（44.1kHz）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocal-gain", type=float, default=1.0)
    ap.add_argument("--acc-gain", type=float, default=0.8)
    ap.add_argument("--lead", default=None,
                    help="原曲主唱（sep2/lead.wav），用于逐帧 RMS 对齐；缺省则跳过该级")
    ap.add_argument("--match-lead", dest="match_lead", action="store_true", default=True,
                    help="人声逐帧 RMS 对齐原曲主唱（默认开，需 --lead）")
    ap.add_argument("--no-match-lead", dest="match_lead", action="store_false")
    ap.add_argument("--lead-smooth", type=float, default=0.2,
                    help="对齐平滑时长秒（默认 0.2；调大跟不上乐句内下陷）")
    ap.add_argument("--lead-clip", type=float, default=12.0,
                    help="对齐增益双向限幅 dB（默认 12）")
    ap.add_argument("--match-acc", dest="match_acc", action="store_true", default=True,
                    help="人声增益跟随伴奏包络（默认开）")
    ap.add_argument("--no-match-acc", dest="match_acc", action="store_false")
    ap.add_argument("--match-margin", type=float, default=0.0,
                    help="目标：人声 RMS 比伴奏高多少 dB（默认 0）")
    ap.add_argument("--match-max-boost", type=float, default=8.0,
                    help="跟随增益上限 dB（默认 8）")
    ap.add_argument("--match-win", type=float, default=5.0, help="分析窗长秒（默认 5）")
    ap.add_argument("--match-gate", type=float, default=-40.0,
                    help="人声有声门限，相对自身峰值 dB（默认 -40）")
    ap.add_argument("--limit-db", type=float, default=-0.5,
                    help="限幅上限 dBFS，0 = 关闭改用硬削波（默认 -0.5）")
    ap.add_argument("--mono", action="store_true", help="输出单声道（默认保留伴奏立体声）")
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    v, sr_v = librosa.load(args.vocal, sr=None, mono=True)
    a, sr_a = librosa.load(args.acc, sr=None, mono=False)

    if sr_a != sr_v:
        v = librosa.resample(v, orig_sr=sr_v, target_sr=sr_a)
        print(f"重采样人声 {sr_v} -> {sr_a}")
    if a.ndim == 1:
        a = a[None, :]

    n = a.shape[-1]
    v = np.pad(v, (0, max(0, n - len(v))))[:n].astype(np.float64)
    a = a[:, :n].astype(np.float64)
    a_mono = a.mean(axis=0)

    # 第一级（治本）：把 SVC 的电平包络校正到原唱，必须在 match_acc 之前 ——
    # 后者的分析要读校正后的人声，否则它会去补偿一个已经被修掉的下陷。
    if args.match_lead and args.lead:
        ld, sr_l = librosa.load(args.lead, sr=None, mono=True)
        if sr_l != sr_a:
            ld = librosa.resample(ld, orig_sr=sr_l, target_sr=sr_a)
        ld = np.pad(ld, (0, max(0, n - len(ld))))[:n].astype(np.float64)
        lcurve, st = lead_norm_curve(
            v, ld, sr_a, smooth_sec=args.lead_smooth,
            clip_db=args.lead_clip, gate_db=args.match_gate,
        )
        if st:
            v = v * lcurve
            print(f"对齐原唱包络: {st['frames']}/{st['total']} 帧有声, "
                  f"增益 {st['min_db']:+.1f} ~ {st['max_db']:+.1f} dB "
                  f"(均值 {st['mean_db']:+.1f}), 限幅 ±{args.lead_clip:.0f}, "
                  f"平滑 {args.lead_smooth * 1000:.0f}ms")
        else:
            print("对齐原唱包络: 跳过（无共同有声帧）")

    gain = np.ones(n)
    if args.match_acc:
        gain, rows = match_gain_curve(
            v, a_mono, sr_a,
            win_sec=args.match_win, margin_db=args.match_margin,
            max_boost_db=args.match_max_boost, gate_db=args.match_gate,
            vocal_gain=args.vocal_gain, acc_gain=args.acc_gain,
        )
        shown = [r for r in rows if r[3] is not None]
        if shown:
            db = [20 * math.log10(max(r[3], 1e-9)) for r in shown]
            print(f"跟随伴奏包络: {len(shown)} 窗有声, 增益 {min(db):+.1f} ~ {max(db):+.1f} dB "
                  f"(均值 {sum(db) / len(db):+.1f}), 上限 {args.match_max_boost:+.0f}")
            for r, d in zip(shown, db):
                if d >= 3.0:
                    print(f"  {r[0]:6.1f}s  人声 {20 * math.log10(max(r[1], 1e-9)):6.1f} "
                          f"伴奏 {20 * math.log10(max(r[2], 1e-9)):6.1f} dBFS  -> {d:+5.1f} dB")

    mix = args.vocal_gain * (v * gain)[None, :] + args.acc_gain * a
    if args.mono or mix.shape[0] == 1:
        mix = mix.mean(axis=0, keepdims=True)

    raw_peak = float(np.abs(mix).max())
    if args.limit_db < 0:
        mix_t = mix.T
        mix_t, red, n_hit = soft_limit(mix_t, sr_a, args.limit_db)
        mix = mix_t.T
        if n_hit:
            print(f"限幅: 原始峰值 {20 * math.log10(max(raw_peak, 1e-9)):+.2f} dBFS, "
                  f"最大衰减 {red:.2f} dB, 触发 {n_hit / sr_a:.1f}s")
    else:
        clipped = int((np.abs(mix) > 1.0).sum())
        mix = np.clip(mix, -1.0, 1.0)
        if clipped:
            print(f"硬削波 {clipped} 样本 ({clipped / mix.size * 100:.4f}%) "
                  f"— 原始峰值 {20 * math.log10(max(raw_peak, 1e-9)):+.2f} dBFS")

    ch = mix.shape[0]
    out = np.repeat(mix, 2, axis=0) if ch == 1 and not args.mono else mix
    sf.write(args.out, out.T.astype(np.float32), sr_a, subtype="PCM_16")
    print(f"-> {args.out} ({n / sr_a:.1f}s, {sr_a}Hz {'mono' if args.mono else 'stereo'}, "
          f"vocal {args.vocal_gain:.2f} / acc {args.acc_gain:.2f}, "
          f"peak {20 * math.log10(max(float(np.abs(mix).max()), 1e-9)):+.2f} dBFS)")


if __name__ == "__main__":
    main()
