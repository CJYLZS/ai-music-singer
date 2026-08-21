#!/usr/bin/env python3
"""鼻音量化：把"听起来鼻音重"变成数字。按韵母鼻音性分类，vs 真人说话段。

问题来源（2026-08）：生产口白 `data/songs/情歌/mv/speech.wav` 听感鼻音偏重，
而既有全部指标（CER/SIM/SingMOS/NISQA + prosody 七组）都测不到 —— prosody 的
LTAS/centroid 只看总体谱形，鼻音是**特定频段的极点/零点结构**，会被总能量平均掉。

## 声学原理（决定了下面每个量怎么取）

鼻化 = 软腭下降，鼻腔与口腔并联。并联在传递函数上做三件事：
  1. **加一个低频极点 P0**（鼻腔第一共振，名义 250-450Hz）→ 低频多出一个峰
  2. **加一个反共振（零点）**，位置随耦合度上移 → 在零点频率挖一个谷
  3. **F1 幅度 A1 下降、带宽 B1 展宽**（能量被分流+鼻腔壁阻尼大）

经典量是 Chen (1997) 的 **A1−P0**（F1 峰幅 − 鼻极点峰幅，dB）：越小=越鼻化。

## ⚠️ A1−P0 在本素材上会退化，所以它不是主指标

A1−P0 要求能把 P0 和 F1 分开。本素材是女声，**F0 中位 254Hz（真人 249Hz）**，
H1 就落在 250-300Hz 的鼻极点区 —— P0 与 H1 无法区分，A1−P0 退化成 A1−H1，
那是个嗓音质量（breathiness）量，不是鼻音量。文献也记录了这一点：A1−P0 在高 F0
说话人上因"鼻极点附近可能没有谐波"而失效。

所以主指标改为三个不依赖"分离 P0 与 H1"的量：
  A. **P0 区能量比** `lf_ratio` = E(300-500Hz) / E(500-3000Hz)
     鼻化增加的低频极点直接抬这一项。**下沿必须 ≥300Hz 把 H1 排除**（见下方 LF_LO
     的注释：F0 中位 254Hz，H1 一项就占 200-3000Hz 能量的 81%，含进来必然饱和）。
  B. **F1 带宽 B1**（Praat Burg）。鼻腔壁阻尼使 F1 展宽，与谐波位置无关。
  C. **反共振深度 `antires_db`** = 在 700-2500Hz 内，倒谱平滑谱的
     (局部峰均值 − 最深谷) —— 鼻化挖的那个零点会加深这个谷。

外加参考用的 D. **`a1_h1_db`**（就是退化版 A1−P0，只作辅证，别单独下结论）。

## 关键设计：按韵母分类 + 取 ORAL−CODA 之差

中文的 -n/-ng 鼻韵尾和 m/n 声母**本来就该鼻化**，所以：
  * 整段平均 ≈ 测"这段文本鼻韵尾占多少"，不是测缺陷；
  * 真正的判据是 **ORAL 字的鼻音量**，以及 **ORAL 与 CODA 的对比度
    `contrast = ORAL − CODA_N/NG`**。
  `contrast` 是本脚本最重要的输出：它在说话人内部做差，**天然消掉了录音通道、
  整体谱倾斜、说话人嗓音基线**这些混淆项，所以是唯一能跨录音直接比的量。
  对比度小 = 该口腔的字也鼻化了 = 软腭没抬起来 = 听感"鼻音重"。

取样区间来自 `singing/align_syllables.py` 的逐字强制对齐。

用法（GPT-SoVITS venv；先跑 align_syllables.py 生成两份 align json）:
    GPT-SoVITS/.venv/bin/python singing/eval_nasality.py \
        --align data/songs/情歌/mv/align_prod.json \
                data/songs/情歌/mv/align_human162.json \
        --labels prod human162 --out nasality.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16000
CLASSES = ("ORAL", "NAS_INIT", "CODA_N", "CODA_NG")
NASAL_CLS = ("CODA_N", "CODA_NG")

# 频带定义（Hz）
# ⚠️ 第一版把 LF 带取成 200-500 / 参考带 200-3000，结果 lf_ratio 恒在 0.95-0.98 —— 饱和。
# 实测 ORAL 字的能量分布：0-200Hz 3.8%、**200-300Hz 81.4%**、300-500 8.1%、500-800 1.9%、
# 800-1500 3.2%、1500-3000 1.4%。那 81.4% 就是 **H1 自己**（F0 中位 254Hz），
# 于是 lf_ratio 实际在测"基频多强"，与鼻音无关。
# 修正：LF 带下沿抬到 300Hz **把 H1 排除在外**，参考带改成 F1-F3 区（500-3000Hz）。
LF_LO, LF_HI = 300.0, 500.0        # 鼻极点 P0 区，已排除 H1（F0 中位 250Hz 附近）
REF_LO, REF_HI = 500.0, 3000.0     # 归一化参考带 = F1 之上的口腔共振区
AR_LO, AR_HI = 700.0, 2500.0       # 反共振搜索带（F1 之上、F3 之下）
F1_MIN, F1_MAX = 200.0, 1200.0     # F1 合法范围


def load(path: str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float64)


def rms_norm(y: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    return y / r


# ------------------------------------------------------------------ 谱工具

def smooth_spec(seg: np.ndarray, n_fft: int = 1024, lifter: int = 24):
    """倒谱平滑的对数幅度谱（包络）。返回 (freqs, log_mag_smoothed)。

    为什么要倒谱平滑：鼻音是**共振结构**（极点/零点），而原始 FFT 谱上占主导的是
    **谐波栅格**（F0 254Hz → 谐波间距 254Hz，梳齿比共振峰还显眼）。直接在原始谱上
    找峰谷会找到谐波，不是共振。倒谱低通（保留前 `lifter` 个倒谱系数）滤掉周期性
    梳齿，只留下包络 —— 这也是 A1-P0 类测量普遍要求宽带谱/包络的原因。
    """
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    w = np.hanning(len(seg))
    S = np.fft.rfft(seg * w, n=max(n_fft, len(seg)))
    mag = np.abs(S) + 1e-12
    logmag = np.log(mag)
    c = np.fft.irfft(logmag)
    c[lifter:-lifter] = 0.0                      # 倒谱低通 = 谱域平滑
    env = np.fft.rfft(c).real
    freqs = np.fft.rfftfreq(max(n_fft, len(seg)), 1.0 / SR)
    return freqs, env


def band_energy(seg: np.ndarray, lo: float, hi: float, n_fft: int = 1024) -> float:
    """带内线性能量（用原始谱，能量比不需要平滑）。"""
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w, n=max(n_fft, len(seg)))) ** 2
    f = np.fft.rfftfreq(max(n_fft, len(seg)), 1.0 / SR)
    m = (f >= lo) & (f < hi)
    return float(S[m].sum())


def antires_depth(freqs: np.ndarray, env: np.ndarray) -> float | None:
    """反共振深度（dB）：AR 带内 (峰均值 − 最深谷)。鼻化的零点会加深这个谷。"""
    m = (freqs >= AR_LO) & (freqs < AR_HI)
    if m.sum() < 8:
        return None
    e = env[m] * (20.0 / np.log(10.0))            # 自然对数 → dB
    # 局部极值
    peaks = [e[i] for i in range(1, len(e) - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1]]
    if not peaks:
        return None
    return float(np.mean(peaks) - e.min())


def a1_h1(seg: np.ndarray, f0: float) -> float | None:
    """退化版 A1−P0：F1 区峰幅 − H1 幅（dB）。**辅证，不作主指标**（见模块 docstring）。"""
    if not np.isfinite(f0) or f0 <= 0:
        return None
    n_fft = 2048                                  # 窄带：要分辨出单根谐波
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    w = np.hanning(len(seg))
    S = np.abs(np.fft.rfft(seg * w, n=max(n_fft, len(seg)))) + 1e-12
    f = np.fft.rfftfreq(max(n_fft, len(seg)), 1.0 / SR)
    db = 20.0 * np.log10(S)
    # H1: F0 附近 ±15% 的最大值
    mh = (f >= f0 * 0.85) & (f <= f0 * 1.15)
    # A1: 300-1100Hz（F1 区）的最大值
    ma = (f >= 300.0) & (f <= 1100.0)
    if mh.sum() < 2 or ma.sum() < 4:
        return None
    return float(db[ma].max() - db[mh].max())


# ------------------------------------------------------------------ 逐字测量

def measure_file(wav: str, items: list[dict], min_dur: float = 0.06) -> list[dict]:
    """对每个字取样，返回 [{cls, lf_ratio, b1, antires_db, a1_h1_db, f1, dur}]。"""
    import librosa
    import parselmouth

    y = rms_norm(load(wav))
    snd = parselmouth.Sound(y, sampling_frequency=SR)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                             maximum_formant=5500.0)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=500, sr=SR,
                            frame_length=1024, hop_length=160)
    hop_s = 160.0 / SR

    rows = []
    for it in items:
        t0, t1 = it["t0"], it["t1"]
        if t1 - t0 < min_dur:
            continue
        # 取字的中间 60%：避开声母/过渡段，只留韵母稳定部分（鼻音看的是元音的鼻化）
        mid = 0.5 * (t0 + t1)
        half = 0.3 * (t1 - t0)
        a, b = mid - half, mid + half
        i0, i1 = int(a * SR), int(b * SR)
        seg = y[max(i0, 0):min(i1, len(y))]
        if len(seg) < 256:
            continue

        e_lf = band_energy(seg, LF_LO, LF_HI)
        e_ref = band_energy(seg, REF_LO, REF_HI)
        lf = e_lf / max(e_ref, 1e-20)

        freqs, env = smooth_spec(seg)
        ar = antires_depth(freqs, env)

        # F1 / B1: 字中点取值
        f1 = fm.get_value_at_time(1, float(mid))
        b1 = fm.get_bandwidth_at_time(1, float(mid))
        if not (f1 and F1_MIN < f1 < F1_MAX):
            f1, b1 = None, None

        k = int(mid / hop_s)
        fk = f0[k] if 0 <= k < len(f0) else np.nan
        # 必须是 voiced 才算：鼻化是**元音**的属性，清音段的 F1/B1 是 Praat 在噪声上
        # 硬拟合出来的（第一版 prod 的 F1 95%CI 上界冲到 725Hz vs 中位 366Hz 就是这个）。
        if not np.isfinite(fk):
            continue
        ah = a1_h1(seg, float(fk))

        rows.append({"cls": it["cls"], "char": it.get("char", ""),
                     "dur": round(t1 - t0, 3),
                     "lf_ratio": round(float(lf), 5),
                     "b1": round(float(b1), 1) if b1 else None,
                     "antires_db": round(ar, 2) if ar is not None else None,
                     "a1_h1_db": round(ah, 2) if ah is not None else None,
                     "f1": round(float(f1), 1) if f1 else None})
    return rows


def agg(rows: list[dict], cls: tuple[str, ...] | None = None) -> dict:
    """按类聚合（中位数，抗离群）。cls=None 表示全部。"""
    sel = [r for r in rows if cls is None or r["cls"] in cls]
    out = {"n": len(sel)}
    for k in ("lf_ratio", "b1", "antires_db", "a1_h1_db", "f1"):
        v = [r[k] for r in sel if r.get(k) is not None]
        out[k] = round(float(np.median(v)), 5) if v else None
        out[k + "_n"] = len(v)
    return out


def boot_ci(rows: list[dict], key: str, cls: tuple[str, ...], n: int = 2000,
            seed: int = 0) -> tuple[float, float] | None:
    """按字 bootstrap 的 95% CI —— **噪声底，本仓库的硬要求**。

    21 个 ORAL 字的中位数本身有抽样误差；不给 CI 就无法判断两条音频的差是不是真的。
    重采样单位是"字"（不是帧），因为同一字内的帧高度相关，按帧重采样会把 CI 压到假小。
    """
    v = [r[key] for r in rows if r["cls"] in cls and r.get(key) is not None]
    if len(v) < 5:
        return None
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    meds = [float(np.median(rng.choice(v, size=len(v), replace=True))) for _ in range(n)]
    return (round(float(np.percentile(meds, 2.5)), 5),
            round(float(np.percentile(meds, 97.5)), 5))


def print_report(res: dict) -> None:
    labels = list(res.keys())
    w = 22
    line = "=" * (w + 13 * len(labels))
    print("\n" + line)
    print(f"{'指标':<{w}}" + "".join(f"{l:>13}" for l in labels))
    print("-" * (w + 13 * len(labels)))

    def row(title, get, fmt="{:>13.4f}"):
        cells = ""
        for l in labels:
            v = get(res[l])
            cells += fmt.format(v) if isinstance(v, (int, float)) else f"{'-':>13}"
        print(f"{title:<{w}}{cells}")

    print("── 字数")
    row("ORAL 字数", lambda d: d["ORAL"]["n"], "{:>13.0f}")
    row("CODA_N/NG 字数", lambda d: d["CODA"]["n"], "{:>13.0f}")
    for key, name, hint in (
        ("lf_ratio", "P0区/F1-F3 300-500", "↑=更鼻"),
        ("b1", "F1 带宽 B1(Hz)", "↑=更鼻"),
        ("antires_db", "反共振深度(dB)", "↑=更鼻"),
        ("a1_h1_db", "A1−H1(dB,辅证)", "↓=更鼻"),
        ("f1", "F1(Hz)", ""),
    ):
        print(f"── {name}  {hint}")
        row("  ORAL", lambda d, k=key: d["ORAL"][k])
        row("  ORAL 95%CI 下", lambda d, k=key: (d["ci"].get(k) or [None])[0])
        row("  ORAL 95%CI 上", lambda d, k=key: (d["ci"].get(k) or [None, None])[1])
        row("  CODA_N/NG", lambda d, k=key: d["CODA"][k])
        row("  对比度 ORAL−CODA", lambda d, k=key: d["contrast"].get(k))
    print(line)
    print("对比度 = ORAL − CODA_N/NG，说话人内部作差，消掉录音通道/谱倾斜/嗓音基线，")
    print("**是唯一可跨录音直接比的量**。对比度趋 0 = 该口腔的字也鼻化了 = 听感鼻音重。")
    print("绝对值只在同一录音内可比；ORAL 一栏跨录音比时要先看 95%CI 是否重叠。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", required=True, nargs="+",
                    help="align_syllables.py 产出的 json（可多个）")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.align]
    if len(labels) != len(args.align):
        raise SystemExit("--labels 数量与 --align 不一致")

    res = {}
    for label, ap_ in zip(labels, args.align):
        d = json.loads(Path(ap_).read_text(encoding="utf-8"))
        rows = measure_file(d["wav"], d["items"])
        oral, coda = agg(rows, ("ORAL",)), agg(rows, NASAL_CLS)
        contrast = {k: (round(oral[k] - coda[k], 5)
                        if oral.get(k) is not None and coda.get(k) is not None else None)
                    for k in ("lf_ratio", "b1", "antires_db", "a1_h1_db", "f1")}
        ci = {k: boot_ci(rows, k, ("ORAL",))
              for k in ("lf_ratio", "b1", "antires_db", "a1_h1_db", "f1")}
        res[label] = {"wav": d["wav"], "ORAL": oral, "CODA": coda,
                      "contrast": contrast, "ci": ci,
                      "all": agg(rows), "rows": rows}
        print(f"[{label}] {len(rows)} 字可测 (ORAL {oral['n']}, CODA {coda['n']})")

    print_report(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n报告: {args.out}")


if __name__ == "__main__":
    main()
