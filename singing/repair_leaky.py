#!/usr/bin/env python
"""局部漏音修复：只重转共识漏音区域，网格选优后拼回 vocal.wav。

对每个漏音区域:
  1. 切出 lead + F0（各加 1s pad，给首段提供上下文，转换后丢弃 pad）
  2. 跑 (seed × knn_alpha × cfg) 网格（参数镜像原命令，只换 target/seed/alpha/cfg）
  3. ERes2NetV2 + CAMPPlus 双模型 1s 逐窗打分，选 excess 降幅最大、
     margin 不倒退、A 不塌陷的候选（每模型独立计算再取均值）
  4. 候选与原始 vocal 在该区域做 0.15s 交叉淡化拼接

输出 vocal_repaired.wav（原始 vocal.wav 不动），并打印每区域的选择明细。

用法:
  seed-vc/.venv/bin/python singing/repair_leaky.py --song-dir data/songs/雨爱 \
      --regions 73-84 150-162 186-198 210-225 \
      --ref data/voices/ref_talk_sing.wav \
      --impostor data/songs/冬眠/sep2/lead.wav data/songs/绿色/sep2/lead.wav \
                 data/songs/爱笑的眼睛/sep2/lead.wav data/songs/陪你看星星/sep2/lead.wav \
                 data/songs/最后一页/sep2/lead.wav \
      --seeds 1 2 --alphas 0.2 0.4 --cfgs 0.0 0.5 \
      --out vocal_repaired.wav
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

REPO = "/home/rookie/voice"
sys.path.insert(0, os.path.join(REPO, "GPT-SoVITS", "GPT_SoVITS", "eres2net"))
sys.path.insert(0, os.path.join(REPO, "seed-vc"))
sys.path.insert(0, os.path.join(REPO, "seed-vc", "modules"))

from eval_triad import SV, load16k, cos, windows  # noqa: E402
from eval_leak_campplus import CP  # noqa: E402

SOULX_PY = os.path.join(REPO, "SoulX-Singer", ".venv", "bin", "python")
SOULX_DIR = os.path.join(REPO, "SoulX-Singer")
INF_SVC = os.path.join(SOULX_DIR, "cli", "inference_svc.py")
SVC_MODEL = os.path.join(REPO, "SoulX-Singer", "pretrained_models", "SoulX-Singer", "model-svc.pt")
SVC_CONFIG = os.path.join(REPO, "SoulX-Singer", "soulxsinger", "config", "soulxsinger.yaml")

SR_OUT = 24000
PAD = 1.0  # 转换时两侧各加 1s 上下文
XF = 0.15  # 拼接交叉淡化秒数


def fmt(x: float) -> str:
    return f"{x:+.4f}" if x < 0 else f"{x:.4f}"


def region_slice(lead_path: str, f0_path: str, t0: float, t1: float, tmp: str):
    """切 lead+F0 到 [t0-PAD, t1+PAD]，返回 (lead_file, f0_file, pad_frames)。"""
    lead, sr = sf.read(lead_path)
    f0 = np.load(f0_path)
    i0 = max(0, int((t0 - PAD) * 50))
    i1 = min(len(f0), int((t1 + PAD) * 50))
    j0 = max(0, int((t0 - PAD) * sr))
    j1 = min(len(lead), int((t1 + PAD) * sr))
    lf = os.path.join(tmp, "lead.wav")
    ff = os.path.join(tmp, "f0.npy")
    sf.write(lf, lead[j0:j1], sr)
    np.save(ff, f0[i0:i1])
    return lf, ff, int(PAD * 50), int(PAD * sr)


def run_svc(prompt_wav, prompt_f0, pool, lead_f, f0_f, out_dir, cfg, alpha, seed) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [SOULX_PY, INF_SVC,
           "--device", "cuda:0", "--model_path", SVC_MODEL, "--config", SVC_CONFIG,
           "--prompt_wav_path", prompt_wav, "--target_wav_path", lead_f,
           "--prompt_f0_path", prompt_f0, "--target_f0_path", f0_f,
           "--save_dir", out_dir, "--pitch_shift", "0",
           "--n_steps", "32", "--cfg", str(cfg), "--rescale_cfg", "0.75",
           "--max_seg_sec", "28.0", "--num_overlaps", "1",
           "--knn_alpha", str(alpha), "--knn_k", "4",
           "--knn_pool_wav", pool, "--seed", str(seed), "--fp16"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace",
                       cwd=SOULX_DIR, env={**os.environ, "PYTHONPATH": SOULX_DIR})
    if r.returncode != 0:
        raise RuntimeError(f"SVC failed ({cfg=} {alpha=} {seed=})\n"
                           f"--- stderr head ---\n{r.stderr[:3000]}\n"
                           f"--- stderr tail ---\n{r.stderr[-2000:]}")
    return os.path.join(out_dir, "generated.wav")


class Scorer:
    """双模型 1s 逐窗打分（只统计区域内的窗口）。"""

    def __init__(self, ref, target, impostors):
        self.sv, self.cp = SV(), CP()
        ref16, tgt16 = load16k(ref), load16k(target)
        self.anch_sv = {"ref": self.sv.emb(ref16), "tgt": self.sv.emb(tgt16)}
        self.anch_cp = {"ref": self.cp.emb(ref16), "tgt": self.cp.emb(tgt16)}
        self.anch_sv.update({f"imp{i}": self.sv.emb(load16k(p)) for i, p in enumerate(impostors)})
        self.anch_cp.update({f"imp{i}": self.cp.emb(load16k(p)) for i, p in enumerate(impostors)})
        torch.cuda.empty_cache()

    def _one(self, emb_fn, anchors, wav16, t_a, t_b) -> dict:
        A, C, Ci = [], [], []
        for t0, seg in windows(wav16, 1.0):
            if t0 < t_a - 1e-6 or t0 + 1.0 > t_b + 1e-6:
                continue
            e = emb_fn(seg)
            A.append(cos(e, anchors["ref"]))
            C.append(cos(e, anchors["tgt"]))
            imps = [cos(e, anchors[f"imp{i}"]) for i in range(len(anchors) - 2)]
            Ci.append(float(np.mean(imps)))
        return {"A": np.mean(A), "C": np.mean(C), "Ci": np.mean(Ci),
                "excess": np.mean(np.array(C) - np.array(Ci)),
                "margin": np.mean(np.array(A) - np.array(C))}

    def score(self, wav_path: str, t_a: float, t_b: float) -> dict:
        """wav_path 内容对应 [t0-PAD, t1+PAD]（窗口时间相对 wav 起点），
        只统计相对时间 [PAD, PAD+(t1-t0)] 内的 1s 窗口。"""
        w16 = load16k(wav_path)
        w_lo, w_hi = PAD, PAD + (t_b - t_a)
        return {
            "sv": self._one(self.sv.emb, self.anch_sv, w16, w_lo, w_hi),
            "cp": self._one(self.cp.emb, self.anch_cp, w16, w_lo, w_hi),
        }

    def score_baseline(self, vocal_path: str, t_a: float, t_b: float) -> dict:
        w16 = load16k(vocal_path)
        return {
            "sv": self._one(self.sv.emb, self.anch_sv, w16, t_a, t_b),
            "cp": self._one(self.cp.emb, self.anch_cp, w16, t_a, t_b),
        }


def mean_of(d: dict, key: str) -> float:
    return float(np.mean([d[m][key] for m in ("sv", "cp")]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--regions", required=True, nargs="+", help="t0-t1 秒，如 73-84")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--impostor", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--cfgs", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--pool", default=None, help="kNN 池（默认 = --ref）")
    ap.add_argument("--out", default="vocal_repaired.wav")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    lead = os.path.join(d, "sep2", "lead.wav")
    f0f = os.path.join(d, "target_f0.npy")
    prompt_wav = os.path.join(d, "prompt_ext.wav")
    prompt_f0 = os.path.join(d, "prompt_ext_f0.npy")
    vocal = os.path.join(d, "vocal.wav")
    out_path = os.path.join(d, args.out)
    pool = os.path.abspath(args.pool or args.ref)
    ref = os.path.abspath(args.ref)
    impostors = [os.path.abspath(p) for p in args.impostor]
    for p in [lead, f0f, prompt_wav, prompt_f0, vocal, pool, ref]:
        if not os.path.exists(p):
            raise SystemExit(f"缺少文件: {p}")

    regions = []
    for r in args.regions:
        a, b = r.split("-")
        regions.append((float(a), float(b)))

    scorer = Scorer(ref, lead, impostors)

    tmp_root = tempfile.mkdtemp(prefix="repair_")
    print(f"workdir: {tmp_root}")
    vocal_w, vocal_sr = sf.read(vocal)

    for ri, (t0, t1) in enumerate(regions):
        print(f"\n=== 区域 {ri}: [{t0:.0f}s, {t1:.0f}s] ===")
        rt = os.path.join(tmp_root, f"r{ri}")
        os.makedirs(rt, exist_ok=True)
        lf, ff, pad_fr, pad_samp = region_slice(lead, f0f, t0, t1, rt)
        base = scorer.score_baseline(vocal, t0, t1)
        b_ex, b_mg, b_a = mean_of(base, "excess"), mean_of(base, "margin"), mean_of(base, "A")
        print(f"  基线: excess={fmt(b_ex)} margin={fmt(b_mg)} A={b_a:.4f}")

        best = None
        for seed in args.seeds:
            for alpha in args.alphas:
                for cfg in args.cfgs:
                    tag = f"s{seed}_a{alpha:g}_c{cfg:g}"
                    try:
                        gen = run_svc(prompt_wav, prompt_f0, pool, lf, ff,
                                      os.path.join(rt, tag), cfg, alpha, seed)
                    except RuntimeError as e:
                        print(f"  {tag}: FAILED {e}")
                        continue
                    s = scorer.score(gen, t0, t1)
                    d_ex = b_ex - mean_of(s, "excess")
                    d_mg = mean_of(s, "margin") - b_mg
                    a_c = mean_of(s, "A")
                    sc = d_ex + 0.3 * d_mg
                    flag = ""
                    if a_c < b_a - 0.015:
                        flag = "  [A塌陷,弃]"
                        sc = -1e9
                    print(f"  {tag}: excess {fmt(mean_of(s,'excess'))} (Δ{fmt(d_ex)}) "
                          f"margin {fmt(mean_of(s,'margin'))} (Δ{fmt(d_mg)}) A={a_c:.4f} score={sc:+.4f}{flag}")
                    if best is None or sc > best[0]:
                        best = (sc, tag, gen, mean_of(s, "excess"), mean_of(s, "margin"))

        if best is None:
            print("  全部候选失败，保留原样")
            continue
        if best[0] < 0.005:
            print(f"  无候选优于基线（最佳 {best[0]:+.4f}），保留原样")
            continue
        sc, tag, gen, g_ex, g_mg = best
        print(f"  >>> 选用 {tag}: excess {fmt(g_ex)} / margin {fmt(g_mg)}")

        # ---- 拼接：候选内容（去掉 pad）替换 vocal[t0:t1]，交叉淡化 ----
        gen_w, gen_sr = sf.read(gen)
        assert gen_sr == SR_OUT, gen_sr
        g0 = pad_samp
        g1 = g0 + int((t1 - t0) * SR_OUT)
        assert g1 <= len(gen_w), (len(gen_w), g1)
        v0, v1 = int(t0 * SR_OUT), int(t1 * SR_OUT)
        xf = int(XF * SR_OUT)
        # 直接替换（无叠唱），边界做余弦淡化避免咔哒
        old = vocal_w[v0:v1].copy()
        new = gen_w[g0:g1].copy()
        n = min(len(old), len(new))
        fade = np.cos(np.linspace(0, np.pi / 2, min(xf, n)))[:n]
        for k in range(min(xf, n)):
            new[k] = new[k] * fade[k] + old[k] * (1 - fade[k])
        for k in range(1, min(xf, n) + 1):
            new[-k] = new[-k] * fade[-k] + old[-k] * (1 - fade[-k])
        vocal_w[v0 : v0 + n] = new[:n]

    sf.write(out_path, vocal_w, SR_OUT)
    print(f"\n已写出 {out_path}")

    if not args.keep_tmp:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
