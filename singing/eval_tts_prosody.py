#!/usr/bin/env python3
"""口白 TTS 韵律 / 清晰度评测：补 eval_tts_ckpts.py 的三项指标（CER/SIM/MOS）之外的缺口。

为什么需要它（2026-08，情歌前奏口白）：
  既有评测只有 CER（吐字）、SIM（音色）、SingMOS/NISQA（自然度）三项，听感上的两个
  问题在这三项里**都不可见**：
    1. 语气不贴合内容 —— 四项里没有任何一项测语调/停顿/语气，SingMOS 是歌声域自然度，
       测"像不像真人发声"，不测"语气对不对"。
    2. 部分段落口齿不清 —— CER 是 ASR 的**离散字判决**，语言模型能从含糊音里猜对字，
       声学上的模糊被判决层吃掉（e10 CER 0.0275 已饱和）。

参照系（本脚本的核心设计）：
  - 真人基线 = data/voices/ref_talk_sing.wav 的 **前 162s 说话段**。
    162s 不是 261s：F0 细扫显示 165s 起 stable(相邻帧 <0.5 半音的比例) 从 ~0.60 跳到
    0.75-0.88、中位 F0 从 ~250Hz 升到 280-364Hz，即 **162s 之后已经在唱**。用 261s 会把
    唱段混进说话统计，F0 std / 语速 / LTAS 全部被污染。
  - 被测 = 情歌 mv 下的 TTS 输出（生产 speech.wav 与 tts_eval*/<arm>/speech.wav）。

七组指标（全部"vs 真人说话段"的相对量，绝对值无意义）：
  A. LTAS      长时平均谱差（dB，1/3 倍频程）+ 高频能量占比 —— "闷/糊"最直接的证据
  B. F0 统计   median / std(半音) / p10-p90 域宽 —— "语气平"的主证据
  C. 语速      音节率(voiced run 计数/秒)、voiced 占比
  D. 停顿      静音段数 / 中位时长 / 最长，vs 文本标点数
  E. ES        emotion2vec 嵌入余弦 + 9 类情感分布（语气是否同源）
  F. 清晰度    2-8kHz 占比、spectral centroid/flux、调制谱 4-16Hz 能量
  G. 元音空间  F1-F2 散布面积（构音清晰度的既有预测量）

用法（GPT-SoVITS venv；emotion2vec 权重在 tools/emotion2vec/.ms-cache）:
  GPT-SoVITS/.venv/bin/python singing/eval_tts_prosody.py \
      --gen data/songs/情歌/mv/speech.wav \
      --labels prod
  # 全臂对比（长文本 7 臂）:
  GPT-SoVITS/.venv/bin/python singing/eval_tts_prosody.py --arms-long
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

REF_WAV = ROOT / "data/voices/ref_talk_sing.wav"
REF_TALK_SEC = 162.0        # 说话段上限，见模块 docstring
SR = 16000

E2V_DIR = ROOT / "tools/emotion2vec/.ms-cache/models/iic--emotion2vec_plus_base/snapshots/master"

EMO_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unk"]
# emotion2vec 取窗参数：**必须重叠**，否则短音频的分布退化成 one-hot（见 Emo.embed）
EMO_WIN = 3.0
EMO_HOP = 0.5


# --------------------------------------------------------------------------- 载入

def load(path: str | Path, limit_sec: float | None = None) -> np.ndarray:
    """16k 单声道加载；limit_sec 截断（真人参考取前 162s 说话段）。"""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=limit_sec)
    return y.astype(np.float32)


def rms_norm(y: np.ndarray, target: float = 0.05) -> np.ndarray:
    """RMS 归一化 —— LTAS/高频占比等谱形指标必须先消掉整体音量差，
    否则测到的是录音电平而不是音色。"""
    r = float(np.sqrt(np.mean(y ** 2))) or 1.0
    return y * (target / r)


def voiced_f0(y: np.ndarray, fmin: float = 80.0, fmax: float = 600.0):
    """pyin 提取 F0，返回 (f0 全帧含 nan, voiced_flag, hop)。
    说话用 pyin 而不是 yin：yin 无 voicing 判决，静音处会吐垃圾值。"""
    import librosa
    hop = 256
    f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR,
                               frame_length=1024, hop_length=hop)
    return f0, vflag, hop


# ------------------------------------------------------------------ A. LTAS / 谱形

def ltas(y: np.ndarray, n_fft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """长时平均谱（dB）。返回 (中心频率, dB)，用 1/3 倍频程带聚合以降噪。"""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    # 只统计有能量的帧（跳静音），否则静音比例不同的两个文件不可比
    frame_e = S.sum(axis=0)
    keep = frame_e >= 0.01 * np.percentile(frame_e, 95)
    if keep.sum() < 4:
        keep = np.ones_like(frame_e, dtype=bool)
    psd = S[:, keep].mean(axis=1)

    edges = 125.0 * (2.0 ** (np.arange(0, 19) / 3.0))     # 125Hz .. 8kHz, 1/3 oct
    cf, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if m.sum() == 0:
            continue
        cf.append(float(np.sqrt(lo * hi)))
        val.append(float(psd[m].mean()))
    v = np.asarray(val)
    db = 10.0 * np.log10(v / v.sum() + 1e-12)              # 谱形归一化：只看分布不看总能量
    return np.asarray(cf), db


def band_ratio(y: np.ndarray, lo: float, hi: float) -> float:
    """[lo,hi] 频带能量占全带比例（只统计有能量帧）。"""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    frame_e = S.sum(axis=0)
    keep = frame_e >= 0.01 * np.percentile(frame_e, 95)
    if keep.sum() < 4:
        keep = np.ones_like(frame_e, dtype=bool)
    psd = S[:, keep].mean(axis=1)
    m = (freqs >= lo) & (freqs < hi)
    return float(psd[m].sum() / (psd.sum() + 1e-12))


# ------------------------------------------------------------------ B. F0 / 语调

def f0_stats(y: np.ndarray) -> dict:
    """F0 统计。std 与域宽一律用**半音**（对数域）—— 用 Hz 会让高音者天然显得
    '语调丰富'，跨说话人不可比。这是'语气平'的主证据。"""
    f0, vflag, hop = voiced_f0(y)
    v = f0[np.isfinite(f0)]
    if len(v) < 10:
        return {"n_voiced": int(len(v))}
    med = float(np.median(v))
    st = 12.0 * np.log2(v / med)                    # 相对中位的半音偏移
    d = np.abs(np.diff(st))
    return {
        "n_voiced": int(len(v)),
        "median_hz": round(med, 1),
        "std_st": round(float(np.std(st)), 3),                       # 语调起伏总量
        "range_p10_p90_st": round(float(np.percentile(st, 90) - np.percentile(st, 10)), 3),
        "iqr_st": round(float(np.percentile(st, 75) - np.percentile(st, 25)), 3),
        "delta_med_st": round(float(np.median(d)), 4),               # 帧间变化速度
        "flat_frac": round(float(np.mean(d < 0.2)), 4),              # 近乎不动的帧占比
        "voiced_frac": round(float(np.mean(np.isfinite(f0))), 4),
    }


# ------------------------------------------------------------------ C/D. 语速 / 停顿

def voiced_runs(y: np.ndarray, top_db: float = 35.0, min_sil: float = 0.12):
    """用 librosa.effects.split 找发声段，返回 [(t0,t1)...] 秒。
    min_sil: 短于此的间隙不算停顿（塞音闭塞期会造成 30-80ms 假静音）。"""
    import librosa
    iv = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    runs = [(a / SR, b / SR) for a, b in iv]
    if not runs:
        return []
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] < min_sil:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def rate_pause_stats(y: np.ndarray) -> dict:
    """语速与停顿结构。音节率用 F0 包络的峰计数估计（中文近似音节 = 一个 voiced 核）。"""
    import librosa
    dur = len(y) / SR
    runs = voiced_runs(y)
    speech = sum(b - a for a, b in runs)
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]

    # 音节率：能量包络在 2-8Hz（音节调制带）的峰计数 / 发声时长
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=256)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5,
                                   delta=0.15 * float(np.std(env)), wait=3)
    syl_rate = len(peaks) / speech if speech > 0.5 else float("nan")

    return {
        "dur_s": round(dur, 2),
        "speech_s": round(speech, 2),
        "speech_frac": round(speech / dur, 4) if dur else None,
        "n_runs": len(runs),
        "syl_rate_hz": round(float(syl_rate), 3),
        "n_pauses": len(gaps),
        "pause_med_s": round(float(np.median(gaps)), 3) if gaps else None,
        "pause_max_s": round(float(np.max(gaps)), 3) if gaps else None,
        "pause_total_s": round(float(np.sum(gaps)), 2) if gaps else 0.0,
    }


# ------------------------------------------------------------------ F. 清晰度 / 谱动态

def clarity_stats(y: np.ndarray) -> dict:
    """'口齿不清'的声学签名。三条独立线索：
      - 高频占比 / centroid：辅音能量塌陷 = 闷
      - spectral flux：谱变化速率，构音含糊时音素间过渡被抹平 → flux 下降
      - 调制谱 4-16Hz：音素级包络调制深度，糊音会衰减（STI 家族的机制）
    """
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    frame_e = (S ** 2).sum(axis=0)
    keep = frame_e >= 0.01 * np.percentile(frame_e, 95)
    if keep.sum() < 4:
        keep = np.ones_like(frame_e, dtype=bool)

    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0][keep]
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.95)[0][keep]
    flat = librosa.feature.spectral_flatness(S=S)[0][keep]
    # flux: 相邻帧归一化谱的 L2 差（只在有能量帧之间）
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-12)
    flux = np.sqrt(((np.diff(Sn, axis=1)) ** 2).sum(axis=0))

    # 调制谱：宽带包络的 FFT，看 4-16Hz（音素级）与 2-4Hz（音节级）能量占比
    env = np.sqrt(frame_e)
    env = env - env.mean()
    fs_env = SR / 256
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    mf = np.fft.rfftfreq(len(env), d=1.0 / fs_env)
    tot = spec[(mf >= 0.5) & (mf <= 20)].sum() + 1e-12
    return {
        "hf_2_8k_ratio": round(band_ratio(y, 2000, 8000), 5),
        "hf_4_8k_ratio": round(band_ratio(y, 4000, 8000), 5),
        "centroid_hz": round(float(np.mean(cent)), 1),
        "rolloff95_hz": round(float(np.mean(roll)), 1),
        "flatness": round(float(np.mean(flat)), 5),
        "flux_mean": round(float(np.mean(flux)), 5),
        "mod_2_4hz": round(float(spec[(mf >= 2) & (mf < 4)].sum() / tot), 4),
        "mod_4_16hz": round(float(spec[(mf >= 4) & (mf < 16)].sum() / tot), 4),
    }


# ------------------------------------------------------------------ G. 元音空间

def vowel_space(y: np.ndarray) -> dict:
    """F1-F2 散布：构音清晰度的既有预测量（元音空间被压扁 = 口齿含糊）。
    用 parselmouth(Praat) 的 Burg 共振峰，只取高能量 voiced 帧。"""
    try:
        import parselmouth
    except Exception:
        return {}
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                             maximum_formant=5500.0)
    f0, vflag, hop = voiced_f0(y)
    times = np.arange(len(f0)) * hop / SR
    pts = []
    for t, ok in zip(times, np.isfinite(f0)):
        if not ok:
            continue
        f1 = fm.get_value_at_time(1, float(t))
        f2 = fm.get_value_at_time(2, float(t))
        if f1 and f2 and 200 < f1 < 1200 and 600 < f2 < 3500:
            pts.append((f1, f2))
    if len(pts) < 50:
        return {"n_formant_frames": len(pts)}
    p = np.asarray(pts)
    # 面积代理: F1/F2 在 bark-ish 对数域的协方差行列式平方根（散布椭圆面积）
    lp = np.log(p)
    cov = np.cov(lp.T)
    area = float(np.sqrt(max(np.linalg.det(cov), 0.0)))
    return {
        "n_formant_frames": len(p),
        "f1_med": round(float(np.median(p[:, 0])), 1),
        "f2_med": round(float(np.median(p[:, 1])), 1),
        "f1_iqr": round(float(np.percentile(p[:, 0], 75) - np.percentile(p[:, 0], 25)), 1),
        "f2_iqr": round(float(np.percentile(p[:, 1], 75) - np.percentile(p[:, 1], 25)), 1),
        "vowel_disp_logarea": round(area, 5),
    }


# ------------------------------------------------------------------ E. emotion2vec ES

class Emo:
    """emotion2vec+base（ModelScope iic/emotion2vec_plus_base，权重在工作区内）。

    长音频必须**分窗**：整条 162s 送进去，utterance 粒度会把整段平均成一个向量，
    9 类分布必然塌到 neutral，等于没测。这里逐窗取嵌入再平均，分布取窗级平均。"""

    def __init__(self, device: str = "cuda:0"):
        if not (E2V_DIR / "model.pt").exists():
            sys.exit(f"缺少 emotion2vec 权重: {E2V_DIR}/model.pt\n"
                     f"  下载: MODELSCOPE_CACHE=tools/emotion2vec/.ms-cache "
                     f"python -c \"from modelscope import snapshot_download; "
                     f"snapshot_download('iic/emotion2vec_plus_base', "
                     f"cache_dir='tools/emotion2vec/.ms-cache')\"")
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from funasr import AutoModel
        self.m = AutoModel(model=str(E2V_DIR), disable_update=True,
                          disable_pbar=True, device=device)

    def embed(self, y: np.ndarray, win: float = EMO_WIN,
              hop: float = EMO_HOP) -> tuple[np.ndarray, np.ndarray, int]:
        """返回 (平均嵌入(已 L2 归一), 平均 9 类分布, 窗数)。

        **必须重叠取窗，且窗不能太长** —— 见 singing/probe_emo_window.py 实测：
        emotion2vec 的 utterance softmax 本身近 one-hot（真人 6s 窗里 16/27 是
        top1>0.99），所以"分布"完全由**窗数**撑起来。win=6/hop=6 时一条 8-10s 的
        TTS 输出只有 1 窗，分布结构上只能是 0 或 1，与真人 162s 的 27 窗不可比：
        当时读出的"prod 100% neutral vs 真人 57.4% neutral"是这个假象，真人随机
        抽一窗得 neutral 的概率就是 0.556，即 p≈0.56，无任何显著性。
        win=3/hop=0.5 让 8s 音频拿到 9-15 窗，真人分布同参数下是
        neutral 0.687 / happy 0.235。"""
        n, h = int(win * SR), max(int(hop * SR), 1)
        segs = [y[i:i + n] for i in range(0, max(len(y) - n + 1, 1), h)]
        segs = [s for s in segs if len(s) >= SR]          # <1s 的碎片丢掉
        if not segs:
            segs = [y]
        embs, dists = [], []
        for s in segs:
            r = self.m.generate(s.astype(np.float32), granularity="utterance",
                                extract_embedding=True)[0]
            e = np.asarray(r["feats"], dtype=np.float64)
            embs.append(e / (np.linalg.norm(e) + 1e-12))
            if "scores" in r:
                dists.append(np.asarray(r["scores"], dtype=np.float64))
        E = np.mean(embs, axis=0)
        E = E / (np.linalg.norm(E) + 1e-12)
        D = np.mean(dists, axis=0) if dists else np.zeros(len(EMO_LABELS))
        return E, D, len(segs)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- 汇总

def measure(y: np.ndarray, do_vowel: bool = True) -> dict:
    yn = rms_norm(y)
    out = {"f0": f0_stats(y), "rate": rate_pause_stats(y), "clarity": clarity_stats(yn)}
    if do_vowel:
        out["vowel"] = vowel_space(y)
    cf, db = ltas(yn)
    out["_ltas"] = {"cf": cf.tolist(), "db": db.tolist()}
    return out


def ltas_diff(ref: dict, gen: dict) -> dict:
    """LTAS 差（gen − ref，dB）。**逐带 dB 是"该带占全带的比例"**（18 带各自 <1，
    所以绝对值必为负）—— 只有**带间差值**有意义，单看某带的 dB 值没有意义。

    分频段用**能量占比之差**汇总（不是 dB 均值之差）：dB 均值会被 -47dB 那种极低带
    主导，而听感对应的是能量份额。正 = 该段能量偏多，负 = 偏少。"""
    cf = np.asarray(ref["_ltas"]["cf"])
    dref = np.asarray(ref["_ltas"]["db"])
    dgen = np.asarray(gen["_ltas"]["db"])
    d = dgen - dref
    bands = {"low_125_500": (125, 500), "mid_500_2k": (500, 2000),
             "hi_2k_4k": (2000, 4000), "hi_4k_8k": (4000, 8000)}
    out = {}
    for k, (lo, hi) in bands.items():
        m = (cf >= lo) & (cf < hi)
        if not m.any():
            continue
        # 该频段能量份额（线性）之差，换算成 dB 比值：正=gen 该段份额更大
        sr_ = float(np.sum(10.0 ** (dref[m] / 10.0)))
        sg_ = float(np.sum(10.0 ** (dgen[m] / 10.0)))
        out[k] = round(10.0 * float(np.log10((sg_ + 1e-12) / (sr_ + 1e-12))), 2)
    out["l1_mean_db"] = round(float(np.abs(d).mean()), 3)
    out["per_band_db"] = [round(float(x), 2) for x in d]
    return out


def rel(gen: dict, ref: dict, keys: list[str]) -> dict:
    """逐键做 gen − ref（缺失键跳过）。"""
    o = {}
    for k in keys:
        a, b = gen.get(k), ref.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            o[k] = round(float(a) - float(b), 4)
    return o


def print_report(ref: dict, arms: dict) -> None:
    labels = list(arms.keys())

    def row(title, get, fmt="{:>9.3f}", refget=None):
        rv = (refget or get)(ref)
        rs = fmt.format(rv) if isinstance(rv, (int, float)) else f"{'-':>9}"
        cells = "".join(fmt.format(get(arms[l])) if isinstance(get(arms[l]), (int, float))
                        else f"{'-':>9}" for l in labels)
        print(f"{title:<22}{rs}{cells}")

    print("\n" + "=" * (22 + 9 * (len(labels) + 1)))
    print(f"{'指标':<22}{'真人162s':>9}" + "".join(f"{l:>9}" for l in labels))
    print("-" * (22 + 9 * (len(labels) + 1)))
    print("── B. F0 / 语调（语气平的主证据）")
    row("F0 median(Hz)", lambda d: d["f0"].get("median_hz"), "{:>9.1f}")
    row("F0 std(半音)↑", lambda d: d["f0"].get("std_st"))
    row("F0 p10-p90(半音)↑", lambda d: d["f0"].get("range_p10_p90_st"))
    row("F0 帧间变化(半音)", lambda d: d["f0"].get("delta_med_st"), "{:>9.4f}")
    row("F0 近平坦帧占比↓", lambda d: d["f0"].get("flat_frac"), "{:>9.4f}")
    print("── C/D. 语速 / 停顿")
    row("音节率(Hz)", lambda d: d["rate"].get("syl_rate_hz"))
    row("发声占比", lambda d: d["rate"].get("speech_frac"), "{:>9.4f}")
    row("停顿数", lambda d: d["rate"].get("n_pauses"), "{:>9.0f}")
    row("停顿中位(s)", lambda d: d["rate"].get("pause_med_s"))
    print("── F. 清晰度（口齿不清的声学签名）")
    row("2-8k 占比↑", lambda d: d["clarity"].get("hf_2_8k_ratio"), "{:>9.5f}")
    row("4-8k 占比↑", lambda d: d["clarity"].get("hf_4_8k_ratio"), "{:>9.5f}")
    row("centroid(Hz)↑", lambda d: d["clarity"].get("centroid_hz"), "{:>9.1f}")
    row("rolloff95(Hz)↑", lambda d: d["clarity"].get("rolloff95_hz"), "{:>9.1f}")
    row("spectral flux↑", lambda d: d["clarity"].get("flux_mean"), "{:>9.5f}")
    row("调制 4-16Hz↑", lambda d: d["clarity"].get("mod_4_16hz"), "{:>9.4f}")
    print("── G. 元音空间（构音清晰度）")
    row("元音散布 logarea↑", lambda d: d.get("vowel", {}).get("vowel_disp_logarea"), "{:>9.5f}")
    row("F1 IQR↑", lambda d: d.get("vowel", {}).get("f1_iqr"), "{:>9.1f}")
    row("F2 IQR↑", lambda d: d.get("vowel", {}).get("f2_iqr"), "{:>9.1f}")
    print("── A. LTAS 差 vs 真人（dB，负=该频段偏少）")
    for k in ("low_125_500", "mid_500_2k", "hi_2k_4k", "hi_4k_8k", "l1_mean_db"):
        row(k, lambda d, k=k: d.get("ltas_diff", {}).get(k), "{:>9.2f}",
            refget=lambda d: None)
    if any("es" in arms[l] for l in labels):
        print("── E. emotion2vec（语气同源性）")
        row("窗数(<5 则分布无效)", lambda d: d.get("emo_windows"), "{:>9.0f}",
            refget=lambda d: None)
        row("ES 余弦↑", lambda d: d.get("es"), "{:>9.4f}", refget=lambda d: None)
        row("neutral 占比", lambda d: d.get("emo_dist", {}).get("neutral"), "{:>9.3f}")
        row("happy 占比", lambda d: d.get("emo_dist", {}).get("happy"), "{:>9.3f}")
        row("surprised 占比", lambda d: d.get("emo_dist", {}).get("surprised"), "{:>9.3f}")
        row("sad 占比", lambda d: d.get("emo_dist", {}).get("sad"), "{:>9.3f}")
    print("=" * (22 + 9 * (len(labels) + 1)))
    print("↑ = 越高越接近真人/越清晰；F0 std 与元音散布低 = 语气平/口齿糊。")
    print("emotion2vec 分布只在窗数足够时可比：softmax 近 one-hot，分布靠窗数撑起来。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", nargs="*", default=[], help="被测 wav（可多个）")
    ap.add_argument("--labels", nargs="*", default=[], help="与 --gen 对应的标签")
    ap.add_argument("--arms-long", action="store_true",
                    help="自动加载 tts_eval_long/<arm>/speech.wav 全 7 臂")
    ap.add_argument("--arms-short", action="store_true",
                    help="自动加载 tts_eval/<arm>/speech.wav 全 7 臂（短口白）")
    ap.add_argument("--prod", action="store_true",
                    help="加入生产口白 data/songs/情歌/mv/speech.wav")
    ap.add_argument("--ref", default=str(REF_WAV))
    ap.add_argument("--ref-sec", type=float, default=REF_TALK_SEC,
                    help=f"真人参考取前 N 秒说话段（默认 {REF_TALK_SEC}，不要用 261：162s 后已在唱）")
    ap.add_argument("--ref-win", type=float, default=0.0,
                    help="真人基线也按 N 秒切窗取中位（0=整段一次算）。用于验证长短音频可比性")
    ap.add_argument("--no-emo", action="store_true", help="跳过 emotion2vec（省 1.1GB 权重加载）")
    ap.add_argument("--no-vowel", action="store_true", help="跳过共振峰/元音空间（parselmouth 较慢）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="", help="报告 JSON 输出路径")
    args = ap.parse_args()

    items: list[tuple[str, str]] = []
    mvd = ROOT / "data/songs/情歌/mv"
    if args.prod:
        items.append(("prod", str(mvd / "speech.wav")))
    for flag, sub in ((args.arms_long, "tts_eval_long"), (args.arms_short, "tts_eval")):
        if flag:
            for arm in ("e5", "e10", "e15", "e20", "e25", "e30", "zs"):
                p = mvd / sub / arm / "speech.wav"
                if p.exists():
                    items.append((arm, str(p)))
    for i, g in enumerate(args.gen):
        items.append((args.labels[i] if i < len(args.labels) else Path(g).parent.name, g))
    if not items:
        sys.exit("没有被测音频：用 --gen / --prod / --arms-long / --arms-short")

    print(f"[ref] {args.ref} 前 {args.ref_sec:g}s（说话段）")
    ref_y = load(args.ref, args.ref_sec)
    ref = measure(ref_y, do_vowel=not args.no_vowel)
    print(f"      voiced {ref['f0'].get('n_voiced')} 帧, "
          f"F0 med {ref['f0'].get('median_hz')}Hz, std {ref['f0'].get('std_st')} 半音")

    emo = None
    if not args.no_emo:
        emo = Emo(args.device)
        ref_e, ref_d, nw = emo.embed(ref_y)
        ref_nwin = nw
        # 打表时 ref 也要有 emo_dist，否则真人那一列全是 "-"（分布行的对照系没了）
        ref["emo_dist"] = {k: round(float(v), 4) for k, v in zip(EMO_LABELS, ref_d)}
        ref["emo_windows"] = nw
        print(f"[emo] 真人基线 {nw} 窗 (win={EMO_WIN}/hop={EMO_HOP}), "
              f"neutral={ref_d[EMO_LABELS.index('neutral')]:.3f}, "
              f"happy={ref_d[EMO_LABELS.index('happy')]:.3f}")

    arms: dict[str, dict] = {}
    for label, path in items:
        print(f"[gen] {label}: {path}")
        y = load(path)
        d = measure(y, do_vowel=not args.no_vowel)
        d["path"] = path
        d["ltas_diff"] = ltas_diff(ref, d)
        if emo is not None:
            e, dist, nw = emo.embed(y)
            d["es"] = round(cos(e, ref_e), 4)
            d["emo_dist"] = {k: round(float(v), 4) for k, v in zip(EMO_LABELS, dist)}
            d["emo_windows"] = nw
            if nw < 5:
                print(f"       ⚠️ {label} 只有 {nw} 窗，分布退化（近 one-hot），不可与"
                      f"真人 {ref_nwin} 窗比 —— 音频太短或 EMO_HOP 太大")
        arms[label] = d

    print_report(ref, arms)

    if args.out:
        rep = {"ref": {"path": args.ref, "sec": args.ref_sec, **{k: v for k, v in ref.items()}},
               "emo_win": EMO_WIN, "emo_hop": EMO_HOP, "arms": arms}
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n报告: {args.out}")


if __name__ == "__main__":
    main()
