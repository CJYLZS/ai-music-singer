#!/usr/bin/env python
"""对齐感知的漏音段拼接修复（v2，替代 repair_leaky 的朴素时间对齐拼接）。

repair_leaky 的问题：拼接边界取漏音窗口边缘（乐句中段），且区域重转的乐句时序
与原版全曲转换不一致（起音可差 0.4s+），导致 150s 处"下一句接得快"、162s 处
"听雨"重复、210/225s 同类问题。

本脚本：
  1. 按原版 vocal 的静音窗确定区域边界（--spans t0-t1，覆盖整个乐句组）
  2. 用指定臂重转（+2s pad）
  3. 包络(10ms RMS)互相关求区域内容相对原版的时间偏移 Δ，平移对齐
  4. 在边界附近搜索"两边都静音"的公共点，拼接（0.1s 交叉淡化）
  5. 双模型打分确认漏音仍低于基线（不低则保留原样）

用法:
  seed-vc/.venv/bin/python singing/repair_splice.py --song-dir data/songs/雨爱 \
      --spans 150.5-165.4 208.55-225.3 \
      --ref data/voices/ref_talk_sing.wav \
      --impostor <5 leads> --arm-seed 1 --arm-alpha 0.2 --arm-cfg 0.5 \
      --out vocal_repaired_v2.wav
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from eval_triad import load16k, windows, cos
from eval_leak_campplus import CP
from eval_triad import SV
from emb_cache import cached_emb

REPO = "/home/rookie/voice"
SOULX_PY = os.path.join(REPO, "SoulX-Singer", ".venv", "bin", "python")
SOULX_DIR = os.path.join(REPO, "SoulX-Singer")
INF_SVC = os.path.join(SOULX_DIR, "cli", "inference_svc.py")
SVC_MODEL = os.path.join(SOULX_DIR, "pretrained_models", "SoulX-Singer", "model-svc.pt")
SVC_CONFIG = os.path.join(SOULX_DIR, "soulxsinger", "config", "soulxsinger.yaml")

SR = 24000
PAD = 2.0      # 转换输入两侧 pad（秒）
XF = 0.10      # 拼接交叉淡化（秒）
HOP = 0.01     # 包络 hop（秒）


def rms_env(w, t0, t1, hop=HOP):
    a, b = int(t0 * SR), int(t1 * SR)
    seg = w[a:b]
    n = int(hop * SR)
    r = np.sqrt(np.mean(seg[: len(seg) // n * n].reshape(-1, n) ** 2, axis=1))
    return t0 + np.arange(len(r)) * hop, r


def find_shift(orig, region, sp0, sp1):
    """包络互相关：region 内容相对 orig 的时间偏移（region 平移 Δ 后与 orig 对齐）。

    region 文件是相对时间轴（0..dur，内容对应原曲 [sp0-PAD, sp1+PAD]）。
    对齐约定：orig 文件时间 t 的内容来自 region 时间 τ = t - (sp0-PAD) - shift。
    """
    ot, oe = rms_env(orig, sp0 - 1.0, sp1 + 1.0)   # orig: 绝对时间
    rt, re_ = rms_env(region, 0.0, (sp1 - sp0) + 2 * PAD)  # region: 相对时间
    o = oe - oe.mean()
    r = re_ - re_.mean()
    best, bshift = -1e9, 0.0
    for shift in np.arange(-1.0, 1.0 + HOP, HOP):
        k0 = int(round((ot[0] - (sp0 - PAD) - shift) / HOP))
        if k0 < 0 or k0 >= len(r):
            continue
        n = min(len(o), len(r) - k0)
        if n < 200:
            continue
        a, b = o[:n], r[k0 : k0 + n]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if c > best:
            best, bshift = c, float(shift)
    return bshift, best


def find_common_silence(orig, region, sp0, shift, t_nom, lo, hi, thr=0.04, span=0.12):
    """在 [t_nom+lo, t_nom+hi] 内找两边都静音的公共点（orig 绝对时间；region 内容在
    orig 文件时间 t 处来自 region 相对时间 τ = t - (sp0-PAD) - shift）。
    先试 span=0.12，找不到再试 0.06，仍找不到返回 None。"""
    ot, oe = rms_env(orig, t_nom + lo - 0.5, t_nom + hi + 0.5)
    # region 全文件包络（相对时间轴）
    rfull_t, rfull = rms_env(region, 0.0, len(region) / SR)
    for sp in (span, 0.06):
        best, bt = None, None
        for t in np.arange(t_nom + lo, t_nom + hi + HOP, HOP):
            om = (ot >= t - sp / 2) & (ot <= t + sp / 2)
            # region 相对时间 τ 对应 orig 文件时间 τ + (sp0-PAD) + shift
            rm = (rfull_t + (sp0 - PAD) + shift >= t - sp / 2) & \
                 (rfull_t + (sp0 - PAD) + shift <= t + sp / 2)
            if om.any() and rm.any() and oe[om].max() < thr and rfull[rm].max() < thr:
                d = abs(t - t_nom)
                if best is None or d < best:
                    best, bt = d, float(t)
        if bt is not None:
            return bt
    return None


def orig_silence_mid(orig, t_nom, lo, hi, thr=0.04, minlen=0.08):
    """原版 vocal 在 [t_nom+lo, t_nom+hi] 内离 t_nom 最近的静音窗中点。"""
    ot, oe = rms_env(orig, t_nom + lo - 0.5, t_nom + hi + 0.5)
    q = oe < thr
    s = None
    wins = []
    for i, v in enumerate(q):
        if v and s is None:
            s = i
        if not v and s is not None:
            if ot[i] - ot[s] >= minlen and t_nom + lo <= ot[s] and ot[i - 1] <= t_nom + hi:
                wins.append((ot[s], ot[i - 1]))
            s = None
    if s is not None and t_nom + lo <= ot[s] and ot[-1] <= t_nom + hi:
        if ot[-1] - ot[s] >= minlen:
            wins.append((ot[s], ot[-1]))
    if not wins:
        return None
    return min(wins, key=lambda w: abs((w[0] + w[1]) / 2 - t_nom))[0] + 0.02


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
                       errors="replace", cwd=SOULX_DIR,
                       env={**os.environ, "PYTHONPATH": SOULX_DIR})
    if r.returncode != 0:
        raise RuntimeError(f"SVC failed: {r.stderr[-1500:]}")
    return os.path.join(out_dir, "generated.wav")


class Scorer:
    def __init__(self, ref, target, impostors):
        self.sv, self.cp = SV(), CP()

        def em(p, model, tag, device):
            """整文件锚点嵌入 + 磁盘缓存。537s 参考的整文件前向在 16GB 卡上瞬时需
            ~15.5GB 显存，是 OOM 高发点（见 singing/emb_cache.py）；这里同样要嵌
            参考 + 目标 + 最多 5 个 impostor，两个模型各一遍。"""
            e = cached_emb(p, tag, lambda: model.emb(load16k(p)), log=print)
            return e.to(device)

        self.anch_sv = {"ref": em(ref, self.sv, "eres2netv2", self.sv.device),
                        "tgt": em(target, self.sv, "eres2netv2", self.sv.device)}
        self.anch_cp = {"ref": em(ref, self.cp, "campplus", "cpu"),
                        "tgt": em(target, self.cp, "campplus", "cpu")}
        self.anch_sv.update({f"imp{i}": em(p, self.sv, "eres2netv2", self.sv.device)
                             for i, p in enumerate(impostors)})
        self.anch_cp.update({f"imp{i}": em(p, self.cp, "campplus", "cpu")
                             for i, p in enumerate(impostors)})
        import torch
        torch.cuda.empty_cache()

    def _one(self, emb_fn, anchors, w16, t_a, t_b):
        A, C, Ci = [], [], []
        for t0, seg in windows(w16, 1.0):
            if t0 < t_a - 1e-6 or t0 + 1.0 > t_b + 1e-6:
                continue
            e = emb_fn(seg)
            A.append(cos(e, anchors["ref"]))
            C.append(cos(e, anchors["tgt"]))
            imps = [cos(e, anchors[f"imp{i}"]) for i in range(len(anchors) - 2)]
            Ci.append(float(np.mean(imps)))
        return np.mean(np.array(C) - np.array(Ci))

    def region_excess(self, wav_path, w_lo, w_hi):
        """wav 自身时间轴 [w_lo, w_hi] 内的 1s 窗双模型平均 excess。"""
        w16 = load16k(wav_path)
        ex = []
        for m in ("sv", "cp"):
            if m == "sv":
                ex.append(self._one(self.sv.emb, self.anch_sv, w16, w_lo, w_hi))
            else:
                ex.append(self._one(self.cp.emb, self.anch_cp, w16, w_lo, w_hi))
        return float(np.mean(ex))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--spans", required=True, nargs="+", help="t0-t1 秒（原版静音窗界定的乐句组）")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--impostor", nargs="*", default=[])
    ap.add_argument("--arm-seed", type=int, default=1)
    ap.add_argument("--arm-alpha", type=float, default=0.2)
    ap.add_argument("--arm-cfg", type=float, default=0.5)
    ap.add_argument("--out", default="vocal_repaired_v2.wav")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    lead = os.path.join(d, "sep2", "lead.wav")
    f0f = os.path.join(d, "target_f0.npy")
    prompt_wav = os.path.join(d, "prompt_ext.wav")
    prompt_f0 = os.path.join(d, "prompt_ext_f0.npy")
    vocal = os.path.join(d, "vocal.wav")
    out_path = os.path.join(d, args.out)
    ref = os.path.abspath(args.ref)
    impostors = [os.path.abspath(p) for p in args.impostor]
    for p in [lead, f0f, prompt_wav, prompt_f0, vocal, ref]:
        if not os.path.exists(p):
            raise SystemExit(f"缺少文件: {p}")

    spans = [tuple(map(float, s.split("-"))) for s in args.spans]
    scorer = Scorer(ref, lead, impostors)

    lead_w, lead_sr = sf.read(lead)
    f0 = np.load(f0f)
    vocal_w, _ = sf.read(vocal)

    tmp = tempfile.mkdtemp(prefix="splice_")
    print(f"workdir {tmp}")

    for si, (t0, t1) in enumerate(spans):
        print(f"\n=== 乐句组 {si}: [{t0:.2f}s, {t1:.2f}s] ===")
        # 1. 切 lead+f0（+PAD）
        i0, i1 = int((t0 - PAD) * 50), int((t1 + PAD) * 50)
        j0, j1 = int((t0 - PAD) * lead_sr), int((t1 + PAD) * lead_sr)
        lf = os.path.join(tmp, f"lead{si}.wav")
        ff = os.path.join(tmp, f"f0{si}.npy")
        sf.write(lf, lead_w[j0:j1], lead_sr)
        np.save(ff, f0[i0:i1])

        # 2. 重转
        tag = f"s{args.arm_seed}_a{args.arm_alpha:g}_c{args.arm_cfg:g}"
        gen = run_svc(prompt_wav, prompt_f0, ref, lf, ff,
                      os.path.join(tmp, f"r{si}_{tag}"), args.arm_cfg, args.arm_alpha, args.arm_seed)
        reg, reg_sr = sf.read(gen)
        assert reg_sr == SR
        # 基线漏音（绝对时间）；重转（相对时间，内容 [t0-PAD, t1+PAD]）
        base_ex = scorer.region_excess(vocal, t0, t1)
        gen_ex = scorer.region_excess(gen, PAD, PAD + (t1 - t0))
        print(f"  区域 excess: 基线 {base_ex:+.4f} → 重转 {gen_ex:+.4f} (Δ{gen_ex-base_ex:+.4f})")

        # 3. 对齐
        shift, corr = find_shift(vocal_w, reg, t0, t1)
        print(f"  包络对齐: Δ={shift:+.3f}s (corr={corr:.3f})")

        # 4. 公共静音边界
        b_start = find_common_silence(vocal_w, reg, t0, shift, t0, lo=-0.5, hi=0.8)
        b_end = find_common_silence(vocal_w, reg, t0, shift, t1, lo=-0.8, hi=0.5)
        if b_start is None or b_end is None or b_end - b_start < 1.0:
            # 回退：取原版自身气口中心（区域侧可能非静音，用加长淡化掩盖）
            xf_use = 0.25
            fb_s = orig_silence_mid(vocal_w, t0, lo=-0.5, hi=0.8)
            fb_e = orig_silence_mid(vocal_w, t1, lo=-0.8, hi=0.5)
            b_start, b_end = fb_s or (t0 - 0.3), fb_e or (t1 + 0.2)
            print(f"  ⚠ 未找到公共静音，回退原版气口 [{b_start:.3f}, {b_end:.3f}]（淡化 {xf_use}s）")
        else:
            xf_use = XF
        print(f"  拼接边界: [{b_start:.3f}s, {b_end:.3f}s]")

        # 5. 拼接：文件时间 t ← region 时间 t-shift；region 覆盖 [t0-PAD, t1+PAD]
        g0 = int((b_start - shift - (t0 - PAD)) * SR)
        g1 = int((b_end - shift - (t0 - PAD)) * SR)
        assert 0 <= g0 < g1 <= len(reg), (g0, g1, len(reg))
        v0, v1 = int(b_start * SR), int(b_end * SR)
        xf = int(xf_use * SR)
        old = vocal_w[v0:v1].copy()
        new = reg[g0:g1].copy()
        n = min(len(old), len(new))
        fade = np.cos(np.linspace(0, np.pi / 2, xf))
        for k in range(min(xf, n)):
            new[k] = new[k] * fade[k] + old[k] * (1 - fade[k])
        for k in range(1, min(xf, n) + 1):
            new[-k] = new[-k] * fade[-k] + old[-k] * (1 - fade[-k])
        vocal_w[v0 : v0 + n] = new[:n]
        print(f"  已替换 [{b_start:.3f}, {b_end:.3f}]（平移 Δ={shift:+.3f}s）")

    sf.write(out_path, vocal_w, SR)
    print(f"\n已写出 {out_path}")
    if not args.keep_tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
