#!/usr/bin/env python
"""三角基准: ref(参考音色) / target(原曲分离人声) / out(转换结果) 之间的相似度。

只测 "out vs ref" 会漏掉两件事:
  B = ref vs target   参考说话人和原唱本身有多远。这不是成绩, 是**任务难度基线**,
                      同时也是 cos 相似度"不同说话人地板"的实测值。
  C = out vs target   输出是否还残留原唱音色 (泄漏)。A 高但 C 也高 = 音色没换过去,
                      只是两者都像而已。

理想形态: A -> 1, C -> B (输出离原唱的距离和参考离原唱一样远)。
退化形态: A ~ B, C -> 1 (基本没转换, 输出还是原唱)。

统一协议
--------
所有分数都是 "某文件的一个窗口 vs 另一文件的全局 embedding", 全局 embedding 用整文件
一次 forward, 和 eval_sim.py 完全一致 —— 所以 A 可以直接和 AGENTS.md 里的历史 SIM 对比,
A 和 C 也可以直接相减 (同一批窗口, 只换锚点)。

不要把这里的数字和内容配对协议 (tools/pcnsf/score_segments.py 的 vs_real 列) 混用,
那个协议宽松约 0.7。也不要和 eval_stab.py 的 1.5s 窗数字混用, 那个低约 0.05。

归一化
------
progress = (A - B) / (1 - B)   0 = 和原唱一样远, 1 = 和参考完全一致
leakage  = (C - B) / (1 - B)   0 = 已完全脱离原唱, 1 = 输出就是原唱
两者都除以说话人对距离, 因此**跨歌曲可比**, 而裸 A 不可比。

用法
----
  GPT-SoVITS/.venv/bin/python singing/eval_triad.py --song-dir data/songs/song_2
  GPT-SoVITS/.venv/bin/python singing/eval_triad.py --song-dir data/songs/song_2 \
      --out a/vocal.wav b/vocal.wav --labels 旧 新
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi

REPO = "/home/rookie/voice"
sys.path.insert(0, os.path.join(REPO, "GPT-SoVITS", "GPT_SoVITS", "eres2net"))
from ERes2NetV2 import ERes2NetV2  # noqa: E402

SV_CKPT = os.path.join(REPO, "GPT-SoVITS", "GPT_SoVITS", "pretrained_models",
                       "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
SR = 16000


class SV:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        m.load_state_dict(torch.load(SV_CKPT, map_location="cpu"))
        self.m = m.eval().to(self.device)

    def emb(self, wav: np.ndarray) -> torch.Tensor:
        w = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = torch.stack([
                Kaldi.fbank(x.unsqueeze(0), num_mel_bins=80, sample_frequency=SR, dither=0)
                for x in w
            ])
            return self.m.forward3(feat).squeeze(0)


def load16k(path: str) -> np.ndarray:
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != SR:
        d = torchaudio.functional.resample(torch.from_numpy(d).float(), sr, SR).numpy()
    return d.astype(np.float32)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def windows(wav: np.ndarray, win_sec: float, rms_min: float = 0.01):
    """切非重叠窗口, 跳过静音。yield (t0, seg)。"""
    win = int(win_sec * SR)
    for start in range(0, max(0, len(wav) - win), win):
        seg = wav[start : start + win]
        if np.sqrt(np.mean(seg**2)) < rms_min:
            continue
        yield start / SR, seg


def score_against(sv: SV, wav: np.ndarray, anchors: dict[str, torch.Tensor],
                  win_sec: float) -> tuple[list[float], dict[str, np.ndarray]]:
    """一批窗口同时对多个锚点打分, 每个窗口只 forward 一次。"""
    times: list[float] = []
    out: dict[str, list[float]] = {k: [] for k in anchors}
    for t0, seg in windows(wav, win_sec):
        e = sv.emb(seg)
        times.append(t0)
        for k, a in anchors.items():
            out[k].append(cos(e, a))
    return times, {k: np.array(v) for k, v in out.items()}


def fmt(x: float) -> str:
    return f"{x:+.4f}" if x < 0 else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", help="data/songs/<name>/，用于自动推导 ref/target/out")
    ap.add_argument("--ref", help="参考音色真实语音（默认 <song-dir>/refsep2/lead.wav）")
    ap.add_argument("--target", help="原曲分离人声（默认 <song-dir>/sep2/lead.wav）")
    ap.add_argument("--out", nargs="*", help="转换输出（默认 <song-dir>/vocal.wav）")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--win", type=float, default=3.0, help="分析窗长(秒)，和 eval_sim 一致")
    ap.add_argument("--f0", help="输出音频对应的 F0 .npy（50Hz），给出则按音高分桶看泄漏")
    ap.add_argument("--impostor", nargs="*", default=None, metavar="LEAD",
                    help="其他歌手的分离人声 wav，用作 C 的冒充者对照。"
                         "裸 C 高度受「分离歌声」域影响（不同歌手互相也有 ~0.78），"
                         "excess = C_own - C_impostor 才是真实的原唱泄漏。")
    args = ap.parse_args()

    d = args.song_dir.rstrip("/") if args.song_dir else None
    ref = args.ref or (os.path.join(d, "refsep2", "lead.wav") if d else None)
    target = args.target or (os.path.join(d, "sep2", "lead.wav") if d else None)
    outs = args.out or ([os.path.join(d, "vocal.wav")] if d else [])
    if not (ref and target and outs):
        ap.error("需要 --song-dir，或同时给出 --ref/--target/--out")
    for p in [ref, target, *outs]:
        if not os.path.exists(p):
            ap.error(f"文件不存在: {p}")
    labels = args.labels or [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in outs]

    sv = SV()
    ref_w, tgt_w = load16k(ref), load16k(target)
    a_ref, a_tgt = sv.emb(ref_w), sv.emb(tgt_w)
    imp = {p: sv.emb(load16k(p)) for p in (args.impostor or []) if os.path.exists(p)}
    # 长参考的整文件前向会把 PyTorch 缓存池撑到 ~19GB（537s 素材实测 reserved
    # 19.48GB vs allocated 0.21GB），而后续每个窗只要 0.34GB。不清掉会在 16GB 卡上
    # 随机 OOM（见 eval_sim.py 同处注释）。embedding 已算完，清理不改数值。
    torch.cuda.empty_cache()

    print(f"ref    {ref}   {len(ref_w) / SR:.1f}s")
    print(f"target {target}   {len(tgt_w) / SR:.1f}s")
    print(f"窗长 {args.win}s，协议: 窗口 vs 全局 embedding（同 eval_sim.py）\n")

    # ---- B: 说话人对距离（任务难度基线 / 归一化分母）----
    _, b_r = score_against(sv, ref_w, {"t": a_tgt}, args.win)
    _, b_t = score_against(sv, tgt_w, {"r": a_ref}, args.win)
    b_r, b_t = b_r["t"], b_t["r"]
    b_glob = cos(a_ref, a_tgt)
    B = float(np.mean([b_r.mean(), b_t.mean()]))

    print("=== B  ref vs target（参考说话人 ↔ 原唱，任务难度基线）")
    print(f"    全局-全局              {b_glob:.4f}")
    print(f"    ref 窗口 vs target全局  mean={b_r.mean():.4f}  median={np.median(b_r):.4f}  n={len(b_r)}")
    print(f"    target 窗口 vs ref全局  mean={b_t.mean():.4f}  median={np.median(b_t):.4f}  n={len(b_t)}")
    print(f"    -> B = {B:.4f}（两向窗口均值的平均，用作归一化分母）")
    print(f"       归一化: progress=(A-B)/(1-B)   leakage=(C-B)/(1-B)\n")

    f0 = np.load(args.f0) if args.f0 else None
    ref_f0_path = os.path.join(d, "prompt_f0.npy") if d else None
    ref_med = None
    if f0 is not None and ref_f0_path and os.path.exists(ref_f0_path):
        rf = np.load(ref_f0_path)
        ref_med = float(np.median(rf[rf > 0]))

    for path, lbl in zip(outs, labels):
        wav = load16k(path)
        anchors = {"ref": a_ref, "tgt": a_tgt}
        anchors.update({f"imp::{p}": e for p, e in imp.items()})
        times, sc = score_against(sv, wav, anchors, args.win)
        A_, C_ = sc["ref"], sc["tgt"]
        if len(A_) == 0:
            print(f"=== {lbl}: 无可用窗口")
            continue
        margin = A_ - C_
        A, C = float(A_.mean()), float(C_.mean())
        prog, leak = (A - B) / (1 - B), (C - B) / (1 - B)

        print(f"=== {lbl}  {path}   windows={len(A_)}  {len(wav) / SR:.1f}s")
        print(f"    A  out vs ref     mean={A:.4f}  median={np.median(A_):.4f}   -> progress {prog:.3f}")
        print(f"    C  out vs target  mean={C:.4f}  median={np.median(C_):.4f}   -> leakage  {leak:.3f}")
        print(f"    A-C margin        mean={fmt(float(margin.mean()))}  "
              f"median={fmt(float(np.median(margin)))}  "
              f"win={100 * float((margin > 0).mean()):.0f}% 更像参考")

        if imp:
            ci = float(np.mean([sc[f"imp::{p}"].mean() for p in imp]))
            print(f"    C_impostor        mean={ci:.4f}  (n={len(imp)} 位其他歌手)")
            print(f"    excess = C-C_imp  {fmt(C - ci)}   <- 扣掉「分离歌声」域偏置后的真实泄漏")
            if C - ci <= 0:
                print("       excess<=0: 未检出原唱音色泄漏，裸 C 高只是域效应")
        else:
            print("    (未给 --impostor: 裸 C 含「分离歌声」域偏置，不同歌手互相也有 ~0.78，")
            print("     不能据此判断泄漏。C>A 单独不构成「转换未生效」的证据。)")
        if C > A and not imp:
            print("    !  C > A，但需 --impostor 对照才能区分泄漏 vs 域偏置")
        elif imp and C - ci > 0 and (C - ci) > (A - B):
            print("    !! 泄漏(excess) 超过获得(A-B)：音色转换弱于原唱残留")

        if f0 is not None and ref_med:
            print(f"    {'semis from ref':<18}{'n':>4}{'A':>9}{'C':>9}{'margin':>9}")
            dist = []
            for t0 in times:
                fa, fb = int(t0 * 50), int((t0 + args.win) * 50)
                v = f0[fa:fb]
                v = v[v > 0]
                dist.append(abs(12 * np.log2(float(np.median(v)) / ref_med)) if len(v) >= 20 else np.nan)
            dist = np.array(dist)
            for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 99)]:
                m = (dist >= lo) & (dist < hi)
                if m.sum() >= 2:
                    lb = f"{lo}-{hi if hi < 99 else '+'}"
                    print(f"    {lb:<18}{m.sum():>4}{A_[m].mean():>9.4f}"
                          f"{C_[m].mean():>9.4f}{fmt(float(margin[m].mean())):>9}")
        print()


if __name__ == "__main__":
    main()
