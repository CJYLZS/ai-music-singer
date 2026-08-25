#!/usr/bin/env python
"""诊断实验：原唱混响尾巴被 RMVPE 判为有声 → SVC 把混响当歌声重建。

这是「纯人声里就听得到原唱与目标音色重叠的混响」的候选机制（2026-08 诊断）：

  分离出的 `sep2/lead.wav` 是**湿的**（句尾 300ms 只衰减 3.5dB，参考素材衰减 28.5dB），
  因为 BS-RoFormer/karaoke 分的是「人声轨」，原唱的板式混响属于人声轨。混响尾巴是
  **原唱声带激励经过房间**的产物 —— 它有周期性，于是 RMVPE 在尾巴帧上有 30-55% 判为
  有声（亲亲 19.9s / 全集平均 6.5% 的时长）。SVC 对有声帧一律合成，于是这些帧被当成
  「一个要唱的音」重建出来，电平只比主体低 10-20dB，**且它的内容是原唱的混响**。
  听感就是干声里始终糊着一层原唱的影子。

  注意这不是 w_mix 的 --match-lead（那一级只在混音时再放大一次底噪，是另一个独立问题）。
  本脚本测的是 vocal.wav 自身。

对照臂（同 seed / cfg / knn，只改输入端）：
  base       现状
  gate       在 target_f0 里把「混响尾巴帧」置 0（模型对这些帧输出静音）
  gate+dry   同上 + 对 lead 做尾巴帧衰减（不让混响进 whisper 的 content 特征）

判读：如果 gate 让听感明显干净，机制成立，且修复很便宜（只动 F0 与 lead 的预处理）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

ROOT = "/home/rookie/voice"
SOULX = os.path.join(ROOT, "SoulX-Singer")
SOULX_PY = os.path.join(SOULX, ".venv", "bin", "python")
INF = os.path.join(SOULX, "cli", "inference_svc.py")
MODEL = os.path.join(SOULX, "pretrained_models", "SoulX-Singer", "model-svc.pt")
CONFIG = os.path.join(SOULX, "soulxsinger", "config", "soulxsinger.yaml")
PAD = 2.0
GRID = 0.02   # 50Hz F0 网格


def tail_mask(lead: np.ndarray, sr: int, n_frames: int, *,
              loud_db=-12.0, aud_db=-35.0, max_tail_sec=0.8) -> np.ndarray:
    """标出「原唱混响尾巴」帧：offset 之后仍可听、但已不是主体的帧。

    判据刻意保守（相对自身 p95 电平），因为我们只想抓住衰减段，不想误伤
    真实的弱唱/气声 —— 后者在 offset 之前就已经是 loud 的一部分。
    """
    hop = int(round(sr * GRID))
    m = min(n_frames, len(lead) // hop)
    e = np.sqrt((lead[:m * hop].reshape(-1, hop) ** 2).mean(1))
    ed = 20 * np.log10(np.maximum(e, 1e-10))
    pk = float(np.percentile(ed, 95))
    loud = ed > pk + loud_db
    aud = ed > pk + aud_db
    out = np.zeros(n_frames, bool)
    last = -10 ** 9
    lim = int(round(max_tail_sec / GRID))
    for i in range(m):
        if loud[i]:
            last = i
        elif aud[i] and 0 < i - last <= lim:
            out[i] = True
    return out


def apply_gate(lead: np.ndarray, sr: int, mask: np.ndarray, *, atten_db=-24.0) -> np.ndarray:
    """对尾巴帧做平滑衰减（不硬切，避免咔嗒声）。"""
    hop = int(round(sr * GRID))
    g = np.ones(len(mask))
    g[mask] = 10 ** (atten_db / 20)
    # 帧级增益插值到样本级并平滑 40ms
    x = np.arange(len(lead))
    gs = np.interp(x, np.arange(len(mask)) * hop + hop // 2, g)
    k = int(sr * 0.04) | 1
    ker = np.hanning(k)
    ker /= ker.sum()
    gs = np.convolve(gs, ker, mode="same")
    return lead * gs


def svc(prompt_wav, prompt_f0, lead_f, f0_f, out_dir, *, pool, seed=1, device="cuda:0"):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [SOULX_PY, INF, "--device", device, "--model_path", MODEL, "--config", CONFIG,
           "--prompt_wav_path", prompt_wav, "--prompt_f0_path", prompt_f0,
           "--target_wav_path", lead_f, "--target_f0_path", f0_f,
           "--save_dir", out_dir, "--pitch_shift", "0", "--n_steps", "32",
           "--cfg", "0.5", "--rescale_cfg", "0.75", "--max_seg_sec", "28.0",
           "--num_overlaps", "1", "--knn_alpha", "0.2", "--knn_k", "4",
           "--knn_pool_wav", pool, "--seed", str(seed), "--fp16"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=SOULX,
                       env={**os.environ, "PYTHONPATH": SOULX})
    if r.returncode != 0:
        raise RuntimeError(f"SVC 失败:\n{r.stderr[-2000:]}")
    return os.path.join(out_dir, "generated.wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--span", required=True, help="t0-t1 秒")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--atten-db", type=float, default=-24.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    t0, t1 = map(float, args.span.split("-"))
    lead_p = os.path.join(d, "sep2", "lead.wav")
    f0_p = os.path.join(d, "target_f0.npy")
    p_wav = os.path.join(d, "prompt_ext.wav")
    p_f0 = os.path.join(d, "prompt_ext_f0.npy")
    ref = os.path.join(ROOT, "data", "voices", "ref_talk_sing.wav")

    out = args.out_dir or tempfile.mkdtemp(prefix="tailgate_", dir=ROOT)
    os.makedirs(out, exist_ok=True)
    print(f"输出目录 {out}")

    lw, lsr = sf.read(lead_p)
    if lw.ndim > 1:
        lw = lw.mean(axis=1)
    f0 = np.load(f0_p)

    mask = tail_mask(lw, lsr, len(f0))
    voiced_tail = mask & (f0 > 0)
    print(f"全曲: 尾巴帧 {mask.sum()} ({mask.sum() * GRID:.1f}s), "
          f"其中 RMVPE 判有声 {voiced_tail.sum()} ({voiced_tail.sum() * GRID:.1f}s) "
          f"= 全曲 {voiced_tail.sum() / len(f0) * 100:.1f}%")

    j0, j1 = max(0, int((t0 - PAD) * lsr)), int((t1 + PAD) * lsr)
    i0, i1 = max(0, int((t0 - PAD) / GRID)), int((t1 + PAD) / GRID)
    seg_lead = lw[j0:j1]
    seg_f0 = f0[i0:i1].copy()
    seg_mask = mask[i0:i1]
    print(f"区间内: 有声尾巴帧 {(seg_mask & (seg_f0 > 0)).sum() * GRID:.1f}s / {(t1 - t0):.0f}s")

    arms = {}
    lf = os.path.join(out, "lead_base.wav")
    ff = os.path.join(out, "f0_base.npy")
    sf.write(lf, seg_lead, lsr)
    np.save(ff, seg_f0)
    arms["base"] = (lf, ff)

    # gate: 只把尾巴帧的 F0 置 0
    f0g = seg_f0.copy()
    f0g[seg_mask] = 0.0
    ffg = os.path.join(out, "f0_gate.npy")
    np.save(ffg, f0g)
    arms["gate"] = (lf, ffg)

    # gate+dry: 同时衰减 lead 上的尾巴（不让混响进 content 特征）
    lg = apply_gate(seg_lead, lsr, seg_mask, atten_db=args.atten_db)
    lfg = os.path.join(out, "lead_dry.wav")
    sf.write(lfg, lg, lsr)
    arms["gate+dry"] = (lfg, ffg)

    outs, labels = [], []
    for name, (l, f) in arms.items():
        print(f"\n=== 臂 {name} ===")
        g = svc(p_wav, p_f0, l, f, os.path.join(out, f"svc_{name.replace('+', '_')}"),
                pool=ref, seed=args.seed, device=args.device)
        w, sr = sf.read(g)
        cut = os.path.join(out, f"{name.replace('+', '_')}.wav")
        sf.write(cut, w[int(PAD * sr):int((PAD + t1 - t0) * sr)], sr)
        outs.append(cut)
        labels.append(name)

    v = os.path.join(d, "vocal.wav")
    if os.path.exists(v):
        w, sr = sf.read(v)
        cut = os.path.join(out, "现有成品.wav")
        sf.write(cut, w[int(t0 * sr):int(t1 * sr)], sr)
        outs.append(cut)
        labels.append("现有成品")
    cut = os.path.join(out, "原唱.wav")
    sf.write(cut, lw[int(t0 * lsr):int(t1 * lsr)], lsr)
    outs.append(cut)
    labels.append("原唱")

    print("\n产物:")
    for l, p in zip(labels, outs):
        print(f"  {l:12s} {p}")


if __name__ == "__main__":
    main()
