"""
多 seed 段级选择 worker：段级 SIM 最大化 + 边界一致性惩罚，用 Viterbi 精确求解。

背景（AGENTS.md「Per-segment seed resampling」/「DP 段级选择」）：
整首歌换 seed 的收益只有 ±0.004（噪声量级），但**按段**换 seed 有 +0.012~+0.039 的
可达空间。朴素「每段取最高分」会让相邻段来自不同 seed，音色在乐句边界跳变（drift ↑）。
惩罚项只涉及相邻两段，所以整条链能用 Viterbi 精确最优化，无需贪心：

    max  Σ_k score[c_k][k]  +  λ · Σ_k cos( tail(c_k, k), head(c_{k+1}, k+1) )

λ=0 → 朴素 argmax；λ→∞ → 全曲同一个 seed。实测最优 λ 与段数相关（6 段用 1，12 段用 0）。

关键：**打分只用 prompt_ext.wav**（推理时已有的产物），不碰真实语音，所以这是可部署的
选择器而不是 oracle。评测请另外用 refsep2/lead.wav 跑 eval_sim。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "GPT-SoVITS", "GPT_SoVITS", "eres2net"))
sys.path.insert(0, os.path.join(ROOT, "SoulX-Singer"))

import torchaudio.compliance.kaldi as Kaldi  # noqa: E402
from ERes2NetV2 import ERes2NetV2  # noqa: E402

from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC  # noqa: E402

SV_CKPT = os.path.join(ROOT, "GPT-SoVITS", "GPT_SoVITS", "pretrained_models", "sv",
                       "pretrained_eres2netv2w24s4ep4.ckpt")
BOUND_SEC = 1.5   # 边界一致性取每段首/尾多长
WIN_SEC = 3.0     # 段内打分窗长，与 eval_sim 保持一致
F0_RATE = 50


def load_sv(device: str):
    state = torch.load(SV_CKPT, map_location="cpu")
    model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def embed(model, wav16k: np.ndarray, device: str):
    if wav16k is None or len(wav16k) < 1600:   # < 0.1s，fbank 会退化
        return None
    w = torch.from_numpy(np.ascontiguousarray(wav16k)).float().unsqueeze(0).to(device)
    feat = Kaldi.fbank(w, num_mel_bins=80, sample_frequency=16000, dither=0).unsqueeze(0)
    with torch.no_grad():
        return model.forward3(feat).squeeze(0)


def load16k(path: str) -> np.ndarray:
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(1)
    if sr != 16000:
        d = torchaudio.functional.resample(torch.from_numpy(d.astype("float32")), sr, 16000).numpy()
    return d.astype("float32")


def cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def seg_windows(wav16k: np.ndarray, f0: np.ndarray, t0: float, t1: float):
    """段内 3s 非重叠窗，沿用 eval_sim 的静音/有声过滤（RMS<0.01 或有声帧<20 丢弃）。"""
    out = []
    t = t0
    while t + WIN_SEC <= t1:
        a, b = int(t * 16000), int((t + WIN_SEC) * 16000)
        if b <= len(wav16k):
            seg = wav16k[a:b]
            fseg = f0[int(t * F0_RATE):int((t + WIN_SEC) * F0_RATE)]
            if np.sqrt(np.mean(seg ** 2)) >= 0.01 and (fseg > 0).sum() >= 20:
                out.append(seg)
        t += WIN_SEC
    return out


def viterbi(score: np.ndarray, bcos: np.ndarray, lam: float) -> list[int]:
    """score: (N, S) 段级得分; bcos: (S-1, N, N) 边界一致性。返回每段选的 seed 下标。"""
    n_seeds, n_seg = score.shape
    dp = score[:, 0].copy()
    back = np.zeros((n_seg, n_seeds), dtype=int)
    for k in range(1, n_seg):
        nxt = np.empty(n_seeds)
        for j in range(n_seeds):
            cand = dp + lam * bcos[k - 1, :, j]
            back[k, j] = int(np.argmax(cand))
            nxt[j] = cand[back[k, j]] + score[j, k]
        dp = nxt
    choice = [0] * n_seg
    choice[-1] = int(np.argmax(dp))
    for k in range(n_seg - 1, 0, -1):
        choice[k - 1] = int(back[k, choice[k]])
    return choice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="打分用参考（prompt_ext.wav；不要用真实语音）")
    ap.add_argument("--target-f0", required=True, help="源 lead 的 F0 .npy（决定分段）")
    ap.add_argument("--cands", required=True, nargs="+", help="各 seed 跑出的 vocal.wav")
    ap.add_argument("--out", required=True, help="拼接结果输出路径")
    ap.add_argument("--lam", type=float, default=1.0, help="边界一致性惩罚权重（0=朴素 argmax）")
    ap.add_argument("--max-seg-sec", type=float, default=28.0)
    ap.add_argument("--num-overlaps", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default=None, help="把选择明细写成 json")
    args = ap.parse_args()

    f0 = np.load(args.target_f0).astype(np.float32).squeeze()
    # 与 SVC 完全同一套分段逻辑，否则拼接边界会落在乐句中间
    _, segments = SoulXSingerSVC.build_vocal_segments(
        torch.from_numpy(f0), f0_rate=F0_RATE, uv_frames_th=10,
        min_duration_sec=min(15.0, args.max_seg_sec / 2),
        max_duration_sec=args.max_seg_sec, num_overlaps=args.num_overlaps)
    if not segments:
        raise SystemExit("build_vocal_segments 没切出任何段，无法做段级选择")

    n_seg, n_seeds = len(segments), len(args.cands)
    print(f"  {n_seg} 段 × {n_seeds} 候选   λ={args.lam:g}")

    model = load_sv(args.device)
    ref_emb = embed(model, load16k(args.prompt), args.device)
    wavs16 = [load16k(p) for p in args.cands]

    score = np.zeros((n_seeds, n_seg))
    head: list[list] = [[None] * n_seg for _ in range(n_seeds)]
    tail: list[list] = [[None] * n_seg for _ in range(n_seeds)]
    bw = int(BOUND_SEC * 16000)
    for si, (t0, t1) in enumerate(segments):
        for ri in range(n_seeds):
            w = wavs16[ri]
            wins = seg_windows(w, f0, t0, t1)
            score[ri, si] = float(np.mean([cos(embed(model, x, args.device), ref_emb) for x in wins])) if wins else 0.0
            a, b = int(t0 * 16000), int(t1 * 16000)
            head[ri][si] = embed(model, w[a:min(a + bw, len(w))], args.device)
            tail[ri][si] = embed(model, w[max(a, b - bw):min(b, len(w))], args.device)

    bcos = np.zeros((max(n_seg - 1, 0), n_seeds, n_seeds))
    for k in range(n_seg - 1):
        for i in range(n_seeds):
            for j in range(n_seeds):
                bcos[k, i, j] = cos(tail[i][k], head[j][k + 1])

    choice = viterbi(score, bcos, args.lam)

    # 按方案拼音频。段外区域（间奏/尾奏）所有候选都是精确 0，直接留零即可
    ref_native, sr = sf.read(args.cands[0])
    mixed = np.zeros_like(ref_native)
    natives = [sf.read(p)[0] for p in args.cands]
    for si, (t0, t1) in enumerate(segments):
        a, b = int(round(t0 * sr)), int(round(t1 * sr))
        a, b = max(0, min(a, len(mixed))), max(0, min(b, len(mixed)))
        src = natives[choice[si]]
        seg = src[a:b] if b <= len(src) else np.pad(src[a:], (0, b - len(src)))[: b - a]
        mixed[a:b] = seg
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, mixed, sr)

    picked = float(np.mean([score[choice[k], k] for k in range(n_seg)]))
    per_seed = [float(score[ri].mean()) for ri in range(n_seeds)]
    best_single = int(np.argmax(per_seed))
    n_switch = sum(1 for k in range(n_seg - 1) if choice[k] != choice[k + 1])
    bnd = float(np.mean([bcos[k, choice[k], choice[k + 1]] for k in range(n_seg - 1)])) if n_seg > 1 else 0.0

    print("  单候选段均分: " + "  ".join(f"#{i}={v:.4f}" for i, v in enumerate(per_seed)))
    print(f"  最好单候选: #{best_single} ({per_seed[best_single]:.4f})   "
          f"DP 选择: {picked:.4f} ({picked - per_seed[best_single]:+.4f})")
    print(f"  换 seed 次数: {n_switch}/{max(n_seg - 1, 0)}   边界一致性: {bnd:.4f}")
    print(f"  逐段选择: {choice}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({
                "segments": [[float(a), float(b)] for a, b in segments],
                "cands": list(args.cands),
                "lam": args.lam,
                "score": score.tolist(),
                "choice": [int(c) for c in choice],
                "picked_score": float(picked),
                "per_seed_score": [float(v) for v in per_seed],
                "best_single_idx": int(best_single),
                "n_switch": int(n_switch),
                "boundary_cos": float(bnd),
            }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
