#!/usr/bin/env python
"""诊断实验 H2：低音段的 in-context prompt 音域不匹配，是不是「低音很差」的真因。

背景（2026-08 诊断）：
  - 参考素材的低音区**只有说话**：唱歌池最低 med F0 = 217Hz，<250Hz 的唱歌素材仅 17.1s，
    而 <250Hz 的说话素材有 83.6s。
  - `prompt_ext.wav` 是全曲共用的一条 45s 音频（`pt_mel` 在段循环**外面**算一次，
    `soulxsinger_svc.py:301`），med F0 331Hz。
  - 亲亲 34-54s 段内 med F0 208Hz，比 prompt 低 8.06 个半音。模型被要求
    用一条「中高音 + 说话低音」的 in-context 示例去唱 200Hz 的长音。

对照臂（全部同 seed / 同 cfg / 同 knn，只换 prompt 或 pitch_shift）：
  base        现有默认：全曲共用 prompt_ext.wav
  lowprompt   **只用低音候选段**重新拼一条 prompt（音域对齐该段），其余不变
  singonly    只用「唱歌」区间(>163.1s)的候选段拼 prompt（排除说话素材）
  pitchauto   prompt 不变，但 --pitch-shift auto（该段会拿到 +12）

判读：如果 lowprompt 明显好于 base，问题是 prompt 音域不匹配（可按段换 prompt 解决）；
如果 singonly 好于 base，问题是低音区混进了说话素材（可按「唱歌素材优先」重排选段）；
如果只有 pitchauto 好，问题是模型在低音区本身不行（只能移调规避）。
"""

from __future__ import annotations

import argparse
import json
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
TALK_END = 163.1     # ref_talk_sing.wav 里 0-163.1s 是说话，之后是唱歌（实测互相关确认）
PAD = 2.0


def build_prompt(cands, out_wav, out_f0, *, budget=45.0, pick, sr_out=44100):
    """按 pick(cand)->bool 过滤候选段并拼一条 prompt（与 w_select 同样的拼接方式）。"""
    import librosa

    sel = [c for c in cands if pick(c)]
    if not sel:
        raise SystemExit("过滤后没有候选段")
    parts, f0s, tot = [], [], 0.0
    for c in sel:
        if tot >= budget:
            break
        y, _ = librosa.load(c["wav"], sr=sr_out, mono=True)
        parts.append(y)
        f0s.append(np.load(c["f0"]))
        tot += c["dur"]
    sf.write(out_wav, np.concatenate(parts), sr_out)
    np.save(out_f0, np.concatenate(f0s).astype(np.float32))
    med = np.median(np.concatenate(f0s)[np.concatenate(f0s) > 0])
    print(f"  -> {os.path.basename(out_wav)}  {tot:.1f}s / {len(parts)}段  med F0 {med:.1f}Hz")
    return out_wav, out_f0, med


def svc(prompt_wav, prompt_f0, lead, f0, out_dir, *, pitch_shift="0", seed=1,
        pool="", device="cuda:0"):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [SOULX_PY, INF, "--device", device, "--model_path", MODEL, "--config", CONFIG,
           "--prompt_wav_path", prompt_wav, "--target_wav_path", lead,
           "--prompt_f0_path", prompt_f0, "--target_f0_path", f0,
           "--save_dir", out_dir, "--pitch_shift", str(pitch_shift),
           "--n_steps", "32", "--cfg", "0.5", "--rescale_cfg", "0.75",
           "--max_seg_sec", "28.0", "--num_overlaps", "1",
           "--knn_alpha", "0.2", "--knn_k", "4", "--seed", str(seed), "--fp16"]
    if pool:
        cmd += ["--knn_pool_wav", pool]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=SOULX,
                       env={**os.environ, "PYTHONPATH": SOULX})
    if r.returncode != 0:
        raise RuntimeError(f"SVC 失败:\n{r.stderr[-2000:]}")
    return os.path.join(out_dir, "generated.wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--span", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    t0, t1 = map(float, args.span.split("-"))
    lead = os.path.join(d, "sep2", "lead.wav")
    f0f = os.path.join(d, "target_f0.npy")
    ref = os.path.join(ROOT, "data", "voices", "ref_talk_sing.wav")
    cands = json.load(open(os.path.join(d, "cands_meta.json"), encoding="utf-8"))

    tmp = tempfile.mkdtemp(prefix="regprompt_", dir=ROOT)
    print(f"workdir {tmp}")

    lw, lsr = sf.read(lead)
    f0 = np.load(f0f)
    lf, ff = os.path.join(tmp, "lead.wav"), os.path.join(tmp, "f0.npy")
    sf.write(lf, lw[max(0, int((t0 - PAD) * lsr)):int((t1 + PAD) * lsr)], lsr)
    np.save(ff, f0[max(0, int((t0 - PAD) * 50)):int((t1 + PAD) * 50)])
    seg = f0[int(t0 * 50):int(t1 * 50)]
    seg_med = float(np.median(seg[seg > 0]))
    print(f"区间 [{t0},{t1}] 段内 med F0 {seg_med:.1f}Hz")

    arms = {}
    arms["base"] = (os.path.join(d, "prompt_ext.wav"), os.path.join(d, "prompt_ext_f0.npy"), "0")
    pm = np.load(arms["base"][1])
    print(f"base prompt med F0 {np.median(pm[pm > 0]):.1f}Hz")

    # 音域对齐 prompt：只取 med 落在该段 ±3 半音内的候选段
    lo, hi = seg_med * 2 ** (-3 / 12), seg_med * 2 ** (3 / 12)
    print(f"\nlowprompt: 只取 med ∈ [{lo:.0f}, {hi:.0f}]Hz 的候选段")
    w, f_, _ = build_prompt(cands, os.path.join(tmp, "p_low.wav"),
                            os.path.join(tmp, "p_low_f0.npy"),
                            pick=lambda c: lo <= c["med"] <= hi)
    arms["lowprompt"] = (w, f_, "0")

    # 只用唱歌素材（排除 0-163.1s 的说话）
    print("singonly: 只取参考里 >163.1s（唱歌）的候选段")
    w, f_, _ = build_prompt(cands, os.path.join(tmp, "p_sing.wav"),
                            os.path.join(tmp, "p_sing_f0.npy"),
                            pick=lambda c: c["start_sec"] >= TALK_END)
    arms["singonly"] = (w, f_, "0")

    # 唱歌素材 ∩ 低音（这一臂检验「低音区唱歌素材是否根本不够」）
    print("singlow: 唱歌素材里 med 最低的那些")
    sl = sorted([c for c in cands if c["start_sec"] >= TALK_END], key=lambda c: c["med"])
    keep = {c["idx"] for c in sl[:20]}
    w, f_, _ = build_prompt(cands, os.path.join(tmp, "p_singlow.wav"),
                            os.path.join(tmp, "p_singlow_f0.npy"),
                            pick=lambda c: c["idx"] in keep)
    arms["singlow"] = (w, f_, "0")

    arms["pitchauto"] = (arms["base"][0], arms["base"][1], "auto")

    outs, labels = [], []
    for name, (pw, pf, ps) in arms.items():
        print(f"\n=== 臂 {name} (pitch_shift={ps}) ===")
        g = svc(pw, pf, lf, ff, os.path.join(tmp, f"arm_{name}"),
                pitch_shift=ps, seed=args.seed, pool=ref, device=args.device)
        wv, sr = sf.read(g)
        cut = os.path.join(tmp, f"{name}.wav")
        sf.write(cut, wv[int(PAD * sr):int((PAD + (t1 - t0)) * sr)], sr)
        outs.append(cut)
        labels.append(name)

    vocal = os.path.join(d, "vocal.wav")
    if os.path.exists(vocal):
        wv, sr = sf.read(vocal)
        cut = os.path.join(tmp, "现有成品.wav")
        sf.write(cut, wv[int(t0 * sr):int(t1 * sr)], sr)
        outs.append(cut)
        labels.append("现有成品")
    cut = os.path.join(tmp, "原唱.wav")
    sf.write(cut, lw[int(t0 * lsr):int(t1 * lsr)], lsr)
    outs.append(cut)
    labels.append("原唱")

    print("\n各臂产物：")
    for l, p in zip(labels, outs):
        print(f"  {l:14s} {p}")
    print(f"\n用 tools/singmos/score_naturalness.py 打分这些文件；目录 {tmp}")


if __name__ == "__main__":
    main()
