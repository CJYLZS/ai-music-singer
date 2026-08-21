#!/usr/bin/env python3
"""SIM 天花板与地板标定：真人自己 vs 自己能到多少分，不同人是多少分。

为什么必须先做这件事：任何"合成输出 SIM = 0.75"都无法单独解读 —— 需要知道
  上界 = 同一个人**另一段录音**对同一锚点的分数（这是 SV 模型在同人跨片段上的极限），
  下界 = 不同说话人的分数（模型的判别地板）。
AGENTS.md 已记录线索：zs 输出 3s 窗 0.7553 竟高于本人切片 0.7511 —— 即该 SV 模型
在这条素材上已分不出合成与真人。本脚本把这个天花板正式量化。

协议与 eval_sim 完全一致（ERes2NetV2、16k、3s 非重叠窗、余弦），这样输出的数字
可以直接和 data/experiments/ 下各臂的 SIM 并列比较。

三类自我相似度都要测，因为它们的差异本身就是结论：
  same-take   同一条录音的不同片段（最乐观，含录音环境/设备的共同偏置）
  cross-take  不同录音场次（ref-vocal.wav 旧参考 vs ref_talk_sing.wav 新录）
  cross-domain 说话段 vs 唱歌段（同人同场次，只有发声方式不同）

用法:
  GPT-SoVITS/.venv/bin/python singing/eval_sim_ceiling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_sim  # noqa: E402

WIN = 3.0
SR = 16000
ANCHOR = ROOT / "data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav"
OLD_REF = ROOT / "data/voices/ref-vocal.wav"          # 旧 163s 参考（girlish 训练源）
NEW_REF = ROOT / "data/voices/ref_talk_sing.wav"      # 新 538s（talk 0-162s / sing 165s+）
TALK_END = 162.0
SING_START = 165.166


def windows(y: np.ndarray) -> list[np.ndarray]:
    w = int(WIN * SR)
    return [y[s:s + w] for s in range(0, len(y) - w + 1, w)]


def load(path, offset=0.0, duration=None):
    y, _ = librosa.load(str(path), sr=SR, mono=True, offset=offset, duration=duration)
    return y


def stats(sv, segs, ref_emb) -> dict:
    sims = [float(eval_sim.cos(sv.emb(s), ref_emb)) for s in segs]
    a = np.array(sims)
    return {"n": len(sims), "mean": round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def main() -> None:
    sv = eval_sim.SV(device="cuda:0")
    anchor_y = load(ANCHOR)
    ref_emb = sv.emb(anchor_y)
    torch.cuda.empty_cache()
    print(f"锚点: {ANCHOR.name} ({len(anchor_y) / SR:.2f}s)，与 experiments 各臂同一锚点\n")

    out = {"protocol": {"model": "ERes2NetV2", "win_sec": WIN, "sr": SR,
                        "anchor": str(ANCHOR.relative_to(ROOT))}, "self": {}, "impostor": {}}

    # ---- 1) same-take：锚点所属那条录音的其余部分 ----
    old = load(OLD_REF)
    segs = windows(old)
    # 排除与锚点重叠的窗（锚点取自 64320-268160 采样点 @44.1k → 1.46-6.08s）
    lo, hi = int(1.0 * SR / (WIN * SR)), int(7.0 / WIN) + 1
    segs_ex = [s for i, s in enumerate(segs) if not (lo <= i <= hi)]
    out["self"]["same_take_old_ref"] = stats(sv, segs_ex, ref_emb)
    print(f"same-take   旧参考其余片段        {out['self']['same_take_old_ref']}")

    # ---- 2) cross-take：新录音的说话段 ----
    talk = load(NEW_REF, 0.0, TALK_END)
    out["self"]["cross_take_new_talk"] = stats(sv, windows(talk), ref_emb)
    print(f"cross-take  新录音说话段 0-162s    {out['self']['cross_take_new_talk']}")

    # ---- 3) cross-domain：新录音的唱歌段 ----
    sing = load(NEW_REF, SING_START, 180.0)
    out["self"]["cross_domain_new_sing"] = stats(sv, windows(sing), ref_emb)
    print(f"cross-domain 新录音唱歌段 165s+    {out['self']['cross_domain_new_sing']}")

    # ---- 4) impostor 地板：其他歌的分离主唱（不同歌手）----
    imp = sorted(ROOT.glob("data/songs/*/sep2/lead.wav"))[:5]
    for p in imp:
        y = load(p, 30.0, 60.0)
        segs = [s for s in windows(y) if float(np.abs(s).max()) > 1e-3]
        if not segs:
            continue
        name = p.parent.parent.name
        out["impostor"][name] = stats(sv, segs, ref_emb)
        print(f"impostor    {name:<12}          {out['impostor'][name]}")

    if out["impostor"]:
        fl = float(np.mean([v["mean"] for v in out["impostor"].values()]))
        out["impostor_floor_mean"] = round(fl, 4)
        print(f"\n不同人地板（impostor 均值）: {fl:.4f}")
    ceil = out["self"]["cross_take_new_talk"]["mean"]
    print(f"同人跨场次天花板（说话）  : {ceil:.4f}")
    print(f"同人跨域（唱歌）          : {out['self']['cross_domain_new_sing']['mean']:.4f}")

    p = ROOT / "data/experiments/sim_ceiling.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
