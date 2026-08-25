#!/usr/bin/env python
"""诊断实验：低音段听感差的两个候选机制，在同一区间上做配对 A/B。

要检验的两条假设（2026-08 诊断，用户听感「低音非常差」+「原唱与目标音色重叠的混响」）：

  H1 混响残留：`sep2/lead.wav` 保留了原唱的板式混响（实测句尾 300ms 只掉 3.5dB，
     参考素材掉 28.5dB）。混响尾巴里带的是**原唱共振峰**，whisper 会把它编码进
     content_feat，CFM 再重建出来 —— 听感即「两个音色叠在一起」。
     干预 = 对 lead 先跑 DeEcho-DeReverb，再送 SVC。

  H2 低音区素材缺口：参考素材的低音区**只有说话**。实测唱歌池最低 med F0 = 217Hz、
     <250Hz 的唱歌素材仅 17.1s，而 <250Hz 的说话素材有 83.6s。歌曲低音段（亲亲
     34-54s med 180-210Hz）拿到的 in-context prompt 于是是「说话音色」，
     模型被要求用说话的共振峰去唱长音。
     干预 A = `--pitch-shift auto`（该段 medF0 210Hz，比 prompt 低 7.87st，会触发 +12）
     干预 B = 把低音区素材用 PC-NSF 向下移调补进 prompt（w_ext 目前只在高音格补）

脚本对给定区间切片重转，逐臂打分（SingMOS 自然度 + eval_triad 的 A/C/excess），
不改流水线。用法：

  SoulX-Singer/.venv/bin/python singing/probe_lowreg.py \
      --song-dir data/songs/亲亲 --span 34.0-54.0 \
      --arms base dereverb pitchauto
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
GSV_PY = os.path.join(ROOT, "GPT-SoVITS", ".venv", "bin", "python")
INF = os.path.join(SOULX, "cli", "inference_svc.py")
MODEL = os.path.join(SOULX, "pretrained_models", "SoulX-Singer", "model-svc.pt")
CONFIG = os.path.join(SOULX, "soulxsinger", "config", "soulxsinger.yaml")
RMVPE = os.path.join(SOULX, "pretrained_models", "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt")
PAD = 2.0
SR_OUT = 24000


def svc(prompt_wav, prompt_f0, lead, f0, out_dir, *, pitch_shift="0", cfg=0.5,
        knn_alpha=0.2, seed=1, pool="", device="cuda:0") -> str:
    """跑一次 SVC。参数与 sing.py 默认一致，只有被测臂不同。"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [SOULX_PY, INF, "--device", device, "--model_path", MODEL, "--config", CONFIG,
           "--prompt_wav_path", prompt_wav, "--target_wav_path", lead,
           "--prompt_f0_path", prompt_f0, "--target_f0_path", f0,
           "--save_dir", out_dir, "--pitch_shift", str(pitch_shift),
           "--n_steps", "32", "--cfg", str(cfg), "--rescale_cfg", "0.75",
           "--max_seg_sec", "28.0", "--num_overlaps", "1",
           "--knn_alpha", str(knn_alpha), "--knn_k", "4", "--seed", str(seed), "--fp16"]
    if pool:
        cmd += ["--knn_pool_wav", pool]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=SOULX,
                       env={**os.environ, "PYTHONPATH": SOULX})
    if r.returncode != 0:
        raise RuntimeError(f"SVC 失败:\n{r.stderr[-2000:]}")
    return os.path.join(out_dir, "generated.wav")


def extract_f0(wav: str, out_npy: str, device="cuda:0") -> str:
    """复用 w_f0.py 提 F0，保证与流水线**同一协议**（50Hz 网格 / 同一 extractor）。

    自己直接调 RMVPE 会得到不同的网格与阈值，A/B 就不可比了 —— 必须走同一入口。
    """
    r = subprocess.run([SOULX_PY, os.path.join(ROOT, "singing", "w_f0.py"),
                        "--wav", wav, "--out", out_npy, "--device", device],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"F0 提取失败:\n{r.stderr[-2000:]}")
    print(f"  {r.stdout.strip()}")
    return out_npy


def dereverb(src: str, out_dir: str, model="VR-DeEchoDeReverb", device="cuda:0") -> str:
    """uvr5 DeEcho 去混响。**注意 vocal/ins 是反的**（vr.py 原注释），干声在 ins_root。"""
    uvr = os.path.join(ROOT, "GPT-SoVITS", "tools", "uvr5")
    code = f'''
import sys, os
sys.path.insert(0, {uvr!r})
from vr import AudioPreDeEcho
w = os.path.join({uvr!r}, "uvr5_weights", {model!r} + ".pth")
pre = AudioPreDeEcho(agg=10, model_path=w, device={device!r}, is_half=True)
pre._path_audio_({src!r}, os.path.join({out_dir!r}, "wet"), os.path.join({out_dir!r}, "dry"), "wav")
'''
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([GSV_PY, "-c", code], capture_output=True, text=True,
                       cwd=uvr, env={**os.environ, "PYTHONPATH": uvr})
    dry = os.path.join(out_dir, "dry")
    got = [os.path.join(dry, f) for f in os.listdir(dry)] if os.path.isdir(dry) else []
    if not got:
        raise RuntimeError(f"去混响无产出:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return got[0]


def singmos(paths: list[str], labels: list[str]) -> dict:
    """SingMOS 自然度（独立 venv）。返回 label -> mean/median。"""
    tool = os.path.join(ROOT, "tools", "singmos")
    py = os.path.join(tool, ".venv", "bin", "python")
    out = os.path.join(tempfile.gettempdir(), "probe_lowreg_mos.json")
    args = [f"{l}={p}" for l, p in zip(labels, paths)]
    r = subprocess.run([py, os.path.join(tool, "score_naturalness.py"),
                        "--win", "3.0", "--no-nisqa", "--out", out] + args,
                       capture_output=True, text=True, cwd=tool)
    if r.returncode != 0:
        print(f"  [singmos 失败] {r.stderr[-800:]}")
        return {}
    import json
    d = json.load(open(out))
    return {k: (v["mean"], v["median"], v["n_windows"]) for k, v in d["files"].items()}


def decay_db(path: str, at=(0.1, 0.3, 0.6)) -> list[float]:
    """句尾衰减（真实气口，要求 1.2s 内不回到 loud）。干声应大幅负值。"""
    import librosa
    x, sr = librosa.load(path, sr=None, mono=True)
    hop = int(sr * 0.02)
    m = len(x) // hop * hop
    if m == 0:
        return [float("nan")] * len(at)
    e = np.sqrt((x[:m].reshape(-1, hop) ** 2).mean(1))
    ed = 20 * np.log10(np.maximum(e, 1e-10))
    pk = float(np.percentile(ed, 95))
    loud = ed > pk - 12
    K = int(1.2 / 0.02)
    cur = []
    for i in range(1, len(loud) - K - 1):
        if loud[i] and not loud[i + 1]:
            seg = ed[i:i + K]
            if (seg[3:] > pk - 12).any():
                continue
            cur.append(seg - seg[0])
    if not cur:
        return [float("nan")] * len(at)
    C = np.array(cur)
    return [float(np.median(C[:, int(t / 0.02)])) for t in at]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--span", required=True, help="t0-t1 秒")
    ap.add_argument("--arms", nargs="+",
                    default=["base", "dereverb", "pitchauto", "dereverb+pitchauto"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    t0, t1 = map(float, args.span.split("-"))
    lead = os.path.join(d, "sep2", "lead.wav")
    f0f = os.path.join(d, "target_f0.npy")
    p_wav = os.path.join(d, "prompt_ext.wav")
    p_f0 = os.path.join(d, "prompt_ext_f0.npy")
    ref = os.path.join(ROOT, "data", "voices", "ref_talk_sing.wav")
    for p in (lead, f0f, p_wav, p_f0, ref):
        if not os.path.exists(p):
            raise SystemExit(f"缺少 {p}")

    tmp = tempfile.mkdtemp(prefix="lowreg_", dir=ROOT)
    print(f"workdir {tmp}")

    lw, lsr = sf.read(lead)
    f0 = np.load(f0f)
    j0, j1 = max(0, int((t0 - PAD) * lsr)), int((t1 + PAD) * lsr)
    i0, i1 = max(0, int((t0 - PAD) * 50)), int((t1 + PAD) * 50)
    lf = os.path.join(tmp, "lead.wav")
    ff = os.path.join(tmp, "f0.npy")
    sf.write(lf, lw[j0:j1], lsr)
    np.save(ff, f0[i0:i1])
    seg = f0[int(t0 * 50):int(t1 * 50)]
    sv = seg[seg > 0]
    pf = np.load(p_f0)
    pm = float(np.median(pf[pf > 0]))
    print(f"区间 [{t0}, {t1}]  段内 medF0 {np.median(sv):.1f}Hz  "
          f"prompt medF0 {pm:.1f}Hz  相距 {12 * np.log2(np.median(sv) / pm):+.2f}st")

    dr_lf = dr_ff = None
    if any("dereverb" in a for a in args.arms):
        print("去混响 lead ...")
        dr_lf = dereverb(lf, os.path.join(tmp, "dr"), device=args.device)
        dr_ff = extract_f0(dr_lf, os.path.join(tmp, "f0_dr.npy"), device=args.device)
        a = decay_db(lf)
        b = decay_db(dr_lf)
        print(f"  句尾衰减 100/300/600ms: 原 {a[0]:+.1f}/{a[1]:+.1f}/{a[2]:+.1f} dB"
              f"  →  去混响 {b[0]:+.1f}/{b[1]:+.1f}/{b[2]:+.1f} dB")

    outs, labels = [], []
    for arm in args.arms:
        use_dr = "dereverb" in arm
        ps = "auto" if "pitchauto" in arm else "0"
        print(f"\n=== 臂 {arm} (lead={'去混响' if use_dr else '原始'}, pitch_shift={ps}) ===")
        g = svc(p_wav, p_f0,
                dr_lf if use_dr else lf,
                dr_ff if use_dr else ff,
                os.path.join(tmp, f"arm_{arm.replace('+', '_')}"),
                pitch_shift=ps, seed=args.seed, pool=ref, device=args.device)
        # 裁掉 pad，只留被测区间
        w, sr = sf.read(g)
        cut = os.path.join(tmp, f"{arm.replace('+', '_')}.wav")
        sf.write(cut, w[int(PAD * sr):int((PAD + (t1 - t0)) * sr)], sr)
        outs.append(cut)
        labels.append(arm)

    # 参照：原版全曲输出的同一区间 + 原唱同区间
    vocal = os.path.join(d, "vocal.wav")
    if os.path.exists(vocal):
        w, sr = sf.read(vocal)
        cut = os.path.join(tmp, "现有成品.wav")
        sf.write(cut, w[int(t0 * sr):int(t1 * sr)], sr)
        outs.append(cut)
        labels.append("现有成品")
    cut = os.path.join(tmp, "原唱.wav")
    sf.write(cut, lw[int(t0 * lsr):int(t1 * lsr)], lsr)
    outs.append(cut)
    labels.append("原唱")

    print("\n打分 SingMOS ...")
    mos = singmos(outs, labels)
    print(f"\n{'臂':22s} {'MOS mean':>9s} {'median':>8s} {'窗':>4s}")
    for l, p in zip(labels, outs):
        k = os.path.basename(p)
        if k in mos:
            m, md, n = mos[k]
            print(f"{l:22s} {m:9.3f} {md:8.3f} {n:4d}")
    print(f"\n试听目录: {tmp}")
    if not args.keep:
        print("（--keep 未指定，但为便于试听仍保留目录，请手动清理）")


if __name__ == "__main__":
    main()
