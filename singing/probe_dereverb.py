#!/usr/bin/env python
"""诊断探针：分离出的 lead 携带原唱混响 → 「原唱与目标音色重叠」的候选机制。

背景（2026-08 诊断）：`sep2/lead.wav` 的句尾衰减实测 **600ms 后比句尾还高 +3.3dB**
（参考素材 −0.2dB，干声）。BS-RoFormer/karaoke 分离的是「人声轨」，原唱的板式混响尾巴
属于人声轨，于是被完整保留下来。SVC 把这条带混响的音频当 content 源，混响尾巴里的
**原唱共振峰**被 whisper 编码进 content_feat，再由 CFM 重建 —— 听感就是「原唱与目标
音色重叠的混响效果」。

本脚本只做测量，不改流水线：对给定 wav 计算
  1. 句尾衰减曲线（T20 近似 + 300/600ms 相对电平），干声应单调下降；
  2. EDR（early-to-late energy ratio）近似：50ms 内 vs 50-300ms 的能量比；
  3. 可选跑 DeEcho-DeReverb / MDX-Net 去混响后重测，量化能去掉多少。

用法（GPT-SoVITS venv，因为 uvr5 在那边）：
  GPT-SoVITS/.venv/bin/python singing/probe_dereverb.py \
      --in data/songs/亲亲/sep2/lead.wav data/voices/ref_talk_sing.wav
  # 加 --dereverb 会额外产出去混响版本并对比
"""

from __future__ import annotations

import argparse
import os

import numpy as np

HOP = 0.05  # 50ms 分析帧


def _env_db(x: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(sr * HOP))
    m = len(x) // hop * hop
    if m == 0:
        return np.zeros(0)
    e = np.sqrt((x[:m].reshape(-1, hop) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(e, 1e-10))


def decay_stats(x: np.ndarray, sr: int) -> dict:
    """句尾衰减：找 loud→quiet 的下降沿，测 300/600ms 后的相对电平。

    干声在气口里应该继续掉（负值）；带混响的音频会因为混响尾巴而持平甚至回升。
    回升（正值）通常意味着下一句起音进入了窗口，所以同时报中位与 n。
    """
    ed = _env_db(x, sr)
    if len(ed) < 20:
        return {}
    pk = float(np.percentile(ed, 95))
    loud = ed > pk - 15
    k3, k6 = int(round(0.3 / HOP)), int(round(0.6 / HOP))
    d3, d6 = [], []
    for i in range(1, len(loud) - k6 - 1):
        if loud[i] and not loud[i + 1]:
            d3.append(ed[i + k3] - ed[i])
            d6.append(ed[i + k6] - ed[i])
    if not d3:
        return {}
    return {
        "n_offsets": len(d3),
        "d300_median": float(np.median(d3)),
        "d600_median": float(np.median(d6)),
    }


def edr(x: np.ndarray, sr: int) -> dict:
    """早/晚能量比近似：句尾后 0-50ms 与 50-300ms 的能量比（dB）。

    值越大越干。这是比 T20 更稳的指标，因为它不依赖找到完整的 −20dB 衰减。
    """
    ed = _env_db(x, sr)
    pk = float(np.percentile(ed, 95))
    loud = ed > pk - 15
    ke, kl = int(round(0.05 / HOP)) or 1, int(round(0.3 / HOP))
    vals = []
    for i in range(1, len(loud) - kl - 1):
        if loud[i] and not loud[i + 1]:
            early = ed[i + 1 : i + 1 + ke]
            late = ed[i + 1 + ke : i + 1 + kl]
            if len(early) and len(late):
                vals.append(float(early.mean() - late.mean()))
    if not vals:
        return {}
    return {"edr_median_db": float(np.median(vals)), "n": len(vals)}


def analyze(path: str) -> dict:
    import librosa

    x, sr = librosa.load(path, sr=None, mono=True)
    out = {"file": path, "sr": sr, "dur": len(x) / sr}
    out.update(decay_stats(x, sr))
    out.update(edr(x, sr))
    return out


def run_dereverb(src: str, out_dir: str, model: str, device: str) -> str:
    """跑 uvr5 的 DeEcho 系模型；返回去混响后的人声路径。

    注意 `AudioPreDeEcho` 的注释「3个VR模型vocal和ins是反的」—— DeEcho 系里
    `ins_root` 拿到的才是去混响后的干声，`vocal_root` 是被剥掉的混响成分。
    这一层反转是踩过的坑，不要按名字直觉取文件。
    """
    import sys

    uvr = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "GPT-SoVITS", "tools", "uvr5")
    sys.path.insert(0, uvr)
    from vr import AudioPreDeEcho  # noqa: E402

    weights = os.path.join(uvr, "uvr5_weights", f"{model}.pth")
    if not os.path.exists(weights):
        raise SystemExit(f"缺少权重 {weights}")
    os.makedirs(out_dir, exist_ok=True)
    dry = os.path.join(out_dir, "dry")
    wet = os.path.join(out_dir, "wet")
    pre = AudioPreDeEcho(agg=10, model_path=weights, device=device, is_half=True)
    pre._path_audio_(src, wet, dry, "wav")
    cands = [os.path.join(dry, f) for f in os.listdir(dry)] if os.path.isdir(dry) else []
    if not cands:
        raise SystemExit(f"{model} 没有产出（检查 {dry}）")
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--dereverb", action="store_true", help="额外跑去混响并对比")
    ap.add_argument("--model", default="VR-DeEchoDeReverb",
                    choices=["VR-DeEchoDeReverb", "VR-DeEchoAggressive", "VR-DeEchoNormal"])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        r = analyze(p)
        rows.append(r)
        if args.dereverb:
            od = args.out_dir or os.path.join(os.path.dirname(p) or ".", "_dereverb")
            dry = run_dereverb(os.path.abspath(p), od, args.model, args.device)
            r2 = analyze(dry)
            r2["file"] = f"{os.path.basename(p)} [{args.model}]"
            rows.append(r2)

    print(f"\n{'file':44s} {'n':>4s} {'300ms':>8s} {'600ms':>8s} {'EDR dB':>8s}")
    for r in rows:
        print(f"{os.path.basename(r['file'])[:44]:44s} {r.get('n_offsets', 0):4d} "
              f"{r.get('d300_median', float('nan')):+8.1f} "
              f"{r.get('d600_median', float('nan')):+8.1f} "
              f"{r.get('edr_median_db', float('nan')):8.1f}")
    print("\n判读：干声的 300/600ms 应为负（继续衰减）、EDR 高；"
          "带混响则 600ms 持平或回升、EDR 低。")


if __name__ == "__main__":
    main()
