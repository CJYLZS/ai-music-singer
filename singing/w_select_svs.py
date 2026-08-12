"""从 SVS 输出里选段构建 prompt_svs（GPT-SoVITS venv 运行，需要 ERes2NetV2）。

## 动机

`prompt_ext` 的音域上限被素材硬卡住：候选池最高段中位 339Hz，`--ext-max-shift 8`
只能到 538Hz，而歌曲有 9.5% 的帧在 547Hz 以上 —— 能到那一格的源段是 **0 段**。
SVS 输出覆盖到 774Hz（音域分布 L1 对歌曲 0.164 vs prompt_ext 的 0.592），
所以它是唯一能补上那一格的素材来源。

## 为什么不按"音色保真度"排序选段

看起来最自然的做法是给每个候选段打 SV 分、挑保真度高的。**AGENTS.md 已记录这条路
是死胡同**，两个独立原因：
  1. 段级 SV 分数在 <1s 音频上基本不可用，且 `corr(段级分数, 时长) = +0.711`
     —— 57% 的方差是时长伪影，不是质量差异。
  2. 留一法实测 prompt 内各段的**真实边际贡献 std 只有 0.001-0.002**，即便存在完美
     排序键，收益也落在 ±0.004 噪声底之下。

所以这里沿用 `w_select` 已验证的**音高覆盖贪心**（边际增益/时长），只把保真度当
诊断量报出来，不作为选择依据。区别于 w_select 的两点：
  - 素材是连续歌声而非说话，最短候选放大到 --min-sec 1.0（说话那边是 0.3s），
    减少碎片；
  - F0 直接复用 svs_f0.npy（50Hz 网格）切片拼接，不重提 —— 与音频严格对齐，
    且避免 F0Extractor 的不可复现性（同文件连跑 4 次 max|diff| 2.2e-4）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

F0_RATE = 50.0
GRID_SEMI = 2.0        # 频段格宽（半音），与 w_select / w_ext 一致
FADE_SEC = 0.01
MIN_VOICED_RATIO = 0.35
MIN_RMS = 0.005


def build_grids(v: np.ndarray) -> list[float]:
    """按歌曲浊音 F0 范围建 2 半音格，返回每格下边界 Hz（同 w_select.make_grids）。"""
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    lo = max(60.0, lo * 2 ** (-3 / 12))
    hi = min(1200.0, hi * 2 ** (3 / 12))
    edges, f = [], lo
    while f < hi:
        edges.append(f)
        f *= 2 ** (GRID_SEMI / 12)
    return edges


def sec_by_grid(v: np.ndarray, grids: list[float]) -> np.ndarray:
    """浊音 F0 序列在各频段格的秒数。"""
    out = np.zeros(len(grids))
    if len(v) == 0:
        return out
    idx = np.floor(12 * np.log2(np.maximum(v, 1e-6) / grids[0]) / GRID_SEMI).astype(int)
    for gi in idx:
        if 0 <= gi < len(grids):
            out[gi] += 1.0 / F0_RATE
    return out


def make_cands(y: np.ndarray, sr: int, min_sec: float, max_sec: float,
               top_db: float) -> list[tuple[int, int]]:
    """发声段切分；长段按 max_sec 滑窗（步长 = max_sec，不重叠，避免同素材重复入池）。"""
    import librosa
    out: list[tuple[int, int]] = []
    for s, e in librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512):
        dur = (e - s) / sr
        if dur < min_sec:
            continue
        if dur <= max_sec:
            out.append((int(s), int(e)))
        else:
            win = int(max_sec * sr)
            t = int(s)
            while t < e:
                if (min(t + win, e) - t) / sr >= min_sec:
                    out.append((t, int(min(t + win, e))))
                t += win
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svs", required=True, help="SVS 输出 wav（素材来源）")
    ap.add_argument("--target-f0", required=True, help="歌曲主唱 F0 .npy（50Hz，定义需求分布）")
    ap.add_argument("--ref", default="", help="真实参考人声（仅用于诊断打分，不参与选择）")
    ap.add_argument("--out-wav", required=True)
    ap.add_argument("--out-f0", required=True)
    ap.add_argument("--svs-f0", default="", help="SVS 输出的 F0 .npy；默认同名 _f0.npy")
    ap.add_argument("--clean", default="",
                    help="干净素材，**应传 prompt.wav（纯选段）而不是 prompt_ext.wav**："
                         "后者 45%% 是 PC-NSF 重合成的移调段，vs 真实参考 0.9259 而 prompt.wav "
                         "是 0.9550，掺进 vocoder 误差与'低中音区用最高保真素材'的立论矛盾。"
                         "给了就走**分层混合**：音高落在干净素材覆盖到的频段优先用它，只有它"
                         "到不了的高音格才用 SVS 素材（SVS 是 460Hz+ 唯一来源）")
    ap.add_argument("--clean-f0", default="", help="干净素材的 F0 .npy；默认同名 _f0.npy")
    ap.add_argument("--clean-bonus", type=float, default=1e9,
                    help="干净素材的优先级权重。默认 1e9 = 硬优先（同格内干净素材总是先用完"
                         "才轮到 SVS）；调小则退化成软偏好")
    ap.add_argument("--clean-cap", type=float, default=-1.0,
                    help="干净素材最多占多少秒。**必须设**，否则硬优先会让干净素材吃光整个"
                         "预算、SVS 一段都进不来（实测 44s 干净素材 100%% 占满 45s 预算，"
                         "分层混合退化成 prompt_ext 本身）。-1 = 自动：预算 − 干净素材"
                         "覆盖不到的缺口秒数（按频段格逐格算），把那部分留给 SVS")
    ap.add_argument("--budget", type=float, default=45.0, help="prompt 总时长上限(秒)")
    ap.add_argument("--min-sec", type=float, default=1.0,
                    help="候选最短时长。连续歌声用 1.0（说话那边是 0.3），减少碎片")
    ap.add_argument("--max-cand-sec", type=float, default=4.0, help="候选最长(秒)")
    ap.add_argument("--top-db", type=float, default=30.0, help="静音判定阈值 dB")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    svs_f0_path = args.svs_f0 or (str(Path(args.svs).with_suffix("")) + "_f0.npy")
    if not Path(svs_f0_path).exists():
        sys.exit(f"缺少 SVS 输出的 F0: {svs_f0_path}（用 w_f0.py 先提，--max-duration 600）")

    y, sr = librosa.load(args.svs, sr=None, mono=True)
    f0_all = np.load(svs_f0_path)
    tv = np.load(args.target_f0)
    tv = tv[tv > 0]
    if len(tv) == 0:
        sys.exit("target F0 无浊音帧")

    # 干净素材（可选）：与 SVS 素材同一套切分/过滤逻辑，只是 src 标记不同
    clean_y = clean_f0 = None
    clean_sr = sr
    if args.clean:
        if not Path(args.clean).exists():
            sys.exit(f"--clean 不存在: {args.clean}")
        cf0p = args.clean_f0 or (str(Path(args.clean).with_suffix("")) + "_f0.npy")
        if not Path(cf0p).exists():
            sys.exit(f"缺少干净素材的 F0: {cf0p}")
        clean_y, clean_sr = librosa.load(args.clean, sr=None, mono=True)
        clean_f0 = np.load(cf0p)
        if clean_sr != sr:
            # 统一到 SVS 的采样率，拼接时才不用重采样（F0 是 50Hz 网格，与 sr 无关）
            clean_y = librosa.resample(clean_y, orig_sr=clean_sr, target_sr=sr)
            clean_sr = sr

    # ---- 需求分布：歌曲各频段时长占比 × 预算（同 w_select）----
    grids = build_grids(tv)
    song_sec = sec_by_grid(tv, grids)
    demand = song_sec / max(float(song_sec.sum()), 1e-9) * args.budget

    # ---- 候选切分 + F0 切片（复用已有 F0，不重提：F0Extractor 不可复现）----
    def collect(yy: np.ndarray, ff: np.ndarray, src: str) -> list[dict]:
        out = []
        for i, (s, e) in enumerate(make_cands(yy, sr, args.min_sec,
                                              args.max_cand_sec, args.top_db)):
            if float(np.sqrt(np.mean(yy[s:e] ** 2))) < MIN_RMS:
                continue
            f0 = ff[int(s / sr * F0_RATE):int(e / sr * F0_RATE)]
            v = f0[f0 > 0]
            if len(f0) == 0 or len(v) == 0 or len(v) / len(f0) < MIN_VOICED_RATIO:
                continue
            out.append(dict(i=i, s=int(s), e=int(e), dur=(e - s) / sr, src=src,
                            med=float(np.median(v)), voiced=len(v) / len(f0),
                            sec_by_grid=sec_by_grid(v, grids)))
        return out

    rows = collect(y, f0_all, "svs")
    if clean_y is not None:
        rows = collect(clean_y, clean_f0, "clean") + rows
    if not rows:
        sys.exit("没有候选通过浊音过滤")
    for src in ("clean", "svs"):
        rs = [r for r in rows if r["src"] == src]
        if rs:
            print(f"候选[{src:5s}]: {len(rs):3d} 段 / {sum(r['dur'] for r in rs):6.1f}s"
                  f"  med {min(r['med'] for r in rs):.0f}-{max(r['med'] for r in rs):.0f}Hz")

    # ---- 干净素材的时长上限 ----
    # 硬优先 + 整段消费会让干净素材吃光预算（44s 干净素材 100% 占满 45s，SVS 进不来），
    # 于是分层混合退化成 prompt_ext 本身。所以先按频段格算出"干净素材供不上的缺口"，
    # 把这部分预算预留给 SVS。
    clean_cap = args.clean_cap
    if clean_y is not None and clean_cap < 0:
        cv = clean_f0[clean_f0 > 0]
        supply = sec_by_grid(cv, grids)
        reserve = float(np.maximum(demand - supply, 0.0).sum())
        avail = float(sum(r["dur"] for r in rows if r["src"] == "clean"))
        # 上限取"预算 − 缺口"，但**不低于干净素材本身的总量**：干净素材是保真度最高的
        # 来源（prompt.wav vs 真实参考 0.9550），能用多少就该用多少，没有理由丢掉。
        # 缺口预留只是为了防止它把预算吃光导致 SVS 一段都进不来（实测那样会退化成
        # prompt_ext 本身）。素材本来就少于预算时，这个预留是多余的。
        clean_cap = max(0.0, args.budget - reserve)
        if clean_cap < avail:
            print(f"  干净素材缺口 {reserve:.2f}s 算出上限 {clean_cap:.2f}s"
                  f" < 素材总量 {avail:.2f}s，放宽到全用（干净素材保真度最高）")
            clean_cap = avail
        else:
            print(f"  干净素材缺口合计 {reserve:.2f}s -> 干净上限 {clean_cap:.2f}s"
                  f"（余下留给 SVS 补高音）")

    # ---- 边际增益贪心（每格"够用即止"，逼贪心走向稀缺的高音格）----
    pool = list(rows)
    selected: list[dict] = []
    used = 0.0
    used_clean = 0.0
    covered = np.zeros(len(grids))
    while pool and used < args.budget:
        def gain(r: dict) -> float:
            g = 0.0
            for gi, sec in enumerate(r["sec_by_grid"]):
                if sec <= 0:
                    continue
                gap = demand[gi] - covered[gi]
                if gap > 0:
                    g += min(gap, sec)
            return g / max(r["dur"], 1e-6)      # 单位时长增益，短段不吃亏

        def key(r: dict) -> tuple[float, float]:
            # 干净素材优先：同样能补上缺口时，永远先用真实 ref 的素材。
            # 只有干净素材在某格已用尽（gain 掉到 ~0）时，SVS 素材才会被选中去补那一格
            # —— 这正是"落在 prompt 范围内就用 prompt，否则才用 svs_vocal"。
            g = gain(r)
            bonus = args.clean_bonus if r["src"] == "clean" else 0.0
            return (-(g * (1.0 + bonus) if g > 1e-4 else g), r["dur"])

        pool.sort(key=key)
        best = pool[0]
        if gain(best) < 1e-4:
            break
        if used + best["dur"] > args.budget:
            pool.pop(0)
            continue
        if best["src"] == "clean" and used_clean + best["dur"] > clean_cap:
            pool.pop(0)          # 干净素材已到上限，剩余预算留给 SVS
            continue
        pool.pop(0)
        selected.append(best)
        if best["src"] == "clean":
            used_clean += best["dur"]
        used += best["dur"]
        covered += best["sec_by_grid"]
        if bool((covered >= demand).all()):
            print("  所有音高格已满足需求，提前停止")
            break
    if not selected:
        sys.exit("贪心未选出任何段")
    cov_pct = float(np.minimum(covered, demand).sum()) / max(float(demand.sum()), 1e-9)
    # 干净素材排在前面（音色锚点靠前，同 w_ext 的 anchor 布局），各源内部按时间序
    selected.sort(key=lambda r: (0 if r["src"] == "clean" else 1, r["s"]))
    clean_sec = sum(r["dur"] for r in selected if r["src"] == "clean")
    print(f"选中 {len(selected)} 段 / {used:.2f}s（预算 {args.budget:.0f}s）覆盖 {cov_pct:.1%}")
    if clean_y is not None:
        print(f"  干净素材 {clean_sec:.2f}s ({clean_sec/used*100:.0f}%)"
              f" + SVS 素材 {used-clean_sec:.2f}s ({(used-clean_sec)/used*100:.0f}%)")

    # ---- 拼接（10ms 淡入淡出，同 w_select）----
    SRC = {"svs": (y, f0_all), "clean": (clean_y, clean_f0)}
    fade_n = int(FADE_SEC * sr)
    fi, fo = np.linspace(0, 1, fade_n), np.linspace(1, 0, fade_n)
    parts, f0_parts = [], []
    for r in selected:
        yy, ff = SRC[r["src"]]
        seg = yy[r["s"]:r["e"]].copy()
        if fade_n > 0 and len(seg) > fade_n * 2:
            seg[:fade_n] *= fi
            seg[-fade_n:] *= fo
        n_fr = int(round(len(seg) / sr * F0_RATE))
        f0 = ff[int(r["s"] / sr * F0_RATE):][:n_fr]
        if len(f0) < n_fr:
            f0 = np.pad(f0, (0, n_fr - len(f0)))
        parts.append(seg)
        f0_parts.append(f0)
    audio = np.concatenate(parts)
    f0 = np.concatenate(f0_parts)
    n_fr = int(len(audio) / sr * F0_RATE)
    f0 = f0[:n_fr] if len(f0) >= n_fr else np.pad(f0, (0, n_fr - len(f0)))
    sf.write(args.out_wav, audio, sr)
    np.save(args.out_f0, f0)
    v = f0[f0 > 0]
    print(f"-> {args.out_wav} ({len(audio)/sr:.2f}s, {sr}Hz)")
    print(f"-> {args.out_f0} ({len(f0)} frames, med {np.median(v):.1f}Hz, "
          f"p90 {np.percentile(v,90):.1f}Hz, >547Hz {float((v>547).mean()):.1%})")

    # ---- 诊断：与真实参考的保真度（只报，不参与选择，理由见 docstring）----
    fid = None
    if args.ref and Path(args.ref).exists():
        try:
            from eval_sim import SV, load16k, cos
            sv = SV(device=args.device)
            fid = cos(sv.emb(load16k(args.out_wav)), sv.emb(load16k(args.ref)))
            print(f"   [诊断] prompt_svs vs 真实参考 = {fid:.4f}"
                  f"（仅参考；主指标是 SVC 输出 vs svs_vocal）")
        except Exception as e:                      # 诊断失败不应影响产物
            print(f"   [诊断] 跳过保真度打分: {e}")

    stats = {
        "svs": args.svs,
        "prompt_wav": args.out_wav,
        "prompt_f0": args.out_f0,
        "prompt_dur": float(len(audio) / sr),
        "prompt_med": float(np.median(v)),
        "prompt_p90": float(np.percentile(v, 90)),
        "coverage_pct": cov_pct,
        "n_cands": len(rows),
        "n_selected": len(selected),
        "clean": args.clean or None,
        "clean_sec": clean_sec,
        "clean_ratio": clean_sec / max(used, 1e-9),
        "fidelity_vs_ref": fid,
        "band_share": {"lt380": float((v < 380).mean()),
                       "380_460": float(((v >= 380) & (v < 460)).mean()),
                       "460_547": float(((v >= 460) & (v < 547)).mean()),
                       "ge547": float((v >= 547).mean())},
        "selected": [dict(idx=r["i"], src=r["src"], start_sec=r["s"] / sr, dur=r["dur"],
                          med=r["med"], voiced=r["voiced"]) for r in selected],
    }
    # 统计文件名跟随 --out-wav，否则 mix 臂会覆盖掉纯 svs 臂的统计
    Path(str(Path(args.out_wav).with_suffix("")) + "_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
