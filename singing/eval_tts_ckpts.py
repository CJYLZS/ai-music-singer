#!/usr/bin/env python3
"""TTS 微调 checkpoint 横向评测：gpt-ft girlish 各 epoch 口白 vs 零样本基线。

三个指标（与仓库既有协议对齐）：
  - 吐字清晰度 : eval_cer.py --ref-text（用口白真实文本作参考，CER 越低越清晰）
  - 相似度 SIM  : eval_sim 的 ERes2NetV2 SV 模型（整句 + 3s 窗余弦，参考=训练用 girlish 切片）
  - 自然度      : SingMOS naturalness + NISQA-tts（score_naturalness.py）

模型臂：
  e5/e10/e15/e20 —— GPT 与 SoVITS 配对微调（同 epoch）。
  e25/e30         —— SoVITS 只微调到 e20，故 GPT-e2x 与 SoVITS-e20 配对（= 生产默认 e30 的组合）。
  zs              —— v2Pro 官方零样本基线（s1bert + s2Gv2Pro），供对照微调收益。

用法（一律用 GPT-SoVITS venv；SingMOS/NISQA 由其自身 venv 子进程调用）：
  GPT-SoVITS/.venv/bin/python singing/eval_tts_ckpts.py gen            # 生成各臂口白
  GPT-SoVITS/.venv/bin/python singing/eval_tts_ckpts.py score          # 三项指标 + report.json
  GPT-SoVITS/.venv/bin/python singing/eval_tts_ckpts.py all            # gen + score

产物落在 <out>/<label>/speech.wav，报告 <out>/report.json。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPTV = ROOT / "GPT-SoVITS/.venv/bin/python"
SINGMOSV = ROOT / "tools/singmos/.venv/bin/python"

TEXT = ROOT / "data/songs/情歌/mv/intro.txt"
REF_AUDIO = ROOT / "data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav"
GIRLISH = ROOT / "data/voices/girlish.list"

V2PRO = ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro"
GPTW = ROOT / "GPT-SoVITS/GPT_weights_v2Pro"
SOVW = ROOT / "GPT-SoVITS/SoVITS_weights_v2Pro"

ARMS = [
    ("e5",  GPTW / "girlish-e5.ckpt",    SOVW / "girlish_e5_s110.pth",  "GPT e5 + SoVITS e5"),
    ("e10", GPTW / "girlish-e10.ckpt",   SOVW / "girlish_e10_s220.pth", "GPT e10 + SoVITS e10"),
    ("e15", GPTW / "girlish-e15.ckpt",   SOVW / "girlish_e15_s330.pth", "GPT e15 + SoVITS e15"),
    ("e20", GPTW / "girlish-e20.ckpt",   SOVW / "girlish_e20_s440.pth", "GPT e20 + SoVITS e20"),
    ("e25", GPTW / "girlish-e25.ckpt",   SOVW / "girlish_e20_s440.pth", "GPT e25 + SoVITS e20*"),
    ("e30", GPTW / "girlish-e30.ckpt",   SOVW / "girlish_e20_s440.pth", "GPT e30 + SoVITS e20* (生产默认)"),
    ("zs",  ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/"
                  "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            V2PRO / "s2Gv2Pro.pth", "v2Pro 官方零样本"),
]


def lookup_ref_text(ref_wav: Path) -> str | None:
    target = ref_wav.name
    for line in GIRLISH.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("|")
        if len(parts) == 4 and Path(parts[0]).name == target:
            return parts[3].strip()
    return None


def gen(out: Path, args) -> None:
    text = Path(args.text).resolve()
    for label, gpt, sovits, _note in ARMS:
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        wav = d / "speech.wav"
        if wav.exists() and not args.force:
            print(f"[skip] {label}: {wav}")
            continue
        ref_text = args.ref_text or lookup_ref_text(REF_AUDIO)
        cmd = [str(GPTV), str(ROOT / "GPT-SoVITS/infer.py"),
               "-t", "@" + str(text),
               "-o", str(wav.resolve()),
               "--gpt", str(gpt.resolve()),
               "--sovits", str(sovits.resolve()),
               "--ref", str(REF_AUDIO.resolve()),
               "--speed", str(args.speed),
               "-q"]
        if ref_text:
            cmd += ["--ref-text", ref_text]
        print(f"[gen] {label}: {gpt.name} + {sovits.name} (speed {args.speed})")
        with open(d / "tts.log", "w") as f:
            r = subprocess.run(cmd, cwd=str(ROOT / "GPT-SoVITS"), stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"error: 生成失败 {label}（见 {d / 'tts.log'}）")
    print("生成完成。")


def score(out: Path, args) -> dict:
    sys.path.insert(0, str(ROOT))
    import eval_sim
    import numpy as np
    import torch
    import soundfile as sf

    report = {"speed": args.speed, "text": str(Path(args.text).resolve()), "ref": str(REF_AUDIO),
              "arms": {label: {"gpt": gpt.name, "sovits": sovits.name, "note": note}
                       for label, gpt, sovits, note in ARMS}}

    wavs = {label: str(out / label / "speech.wav") for label, *_ in ARMS}
    for label, p in wavs.items():
        if not os.path.exists(p):
            sys.exit(f"缺少 {label} 的 speech.wav（先跑 gen）")

    # ---- 1) 吐字 CER（eval_cer.py 子进程，FunASR paraformer-large）----
    cer_items = {}
    cer_cmd = [str(GPTV), str(ROOT / "singing/eval_cer.py"),
               "--src", str(REF_AUDIO.resolve()),
               "--ref-text", str(Path(args.text).resolve()),
               "--gen"] + [Path(p).resolve() for p in wavs.values()] + \
              ["--labels"] + list(wavs.keys()) + ["--win", "10"]
    if args.no_cer:
        print("[cer] 跳过")
    else:
        print("[cer] running ...")
        r = subprocess.run(cer_cmd, cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        print(r.stdout)
        # 解析 'variant CER S D I 字数 vs ref' 表（表头 8 字段、数据行 7 字段）
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 7 and parts[0] in wavs:
                try:
                    cer = float(parts[1])
                except ValueError:
                    continue
                label = parts[0]
                cer_items[label] = {"cer": cer, "S": int(parts[2]), "D": int(parts[3]),
                                    "I": int(parts[4]), "hyp_len": int(parts[5])}
        if r.returncode != 0:
            sys.exit(f"eval_cer.py 失败 (exit {r.returncode})")
    report["cer"] = cer_items

    # ---- 2) SIM：整句 + 3s 窗 vs 参考（内置 ERes2NetV2 SV）----
    device = args.device
    sv = eval_sim.SV(device=device)
    ref_emb = sv.emb(eval_sim.load16k(str(REF_AUDIO)))
    torch.cuda.empty_cache()
    win = 3.0
    sim_items = {}
    for label, p in wavs.items():
        wav = eval_sim.load16k(p)
        # 超长音频整句前向缓存代价大、意义也不大：>100s 只出 3s 窗统计
        if len(wav) / 16000 <= 100:
            whole = float(eval_sim.cos(sv.emb(wav), ref_emb))
        else:
            whole = None
        wlen = int(win * 16000)
        ws = [eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
              for s in range(0, len(wav) - wlen + 1, wlen)]
        sim_items[label] = {"whole": round(whole, 4) if whole else None,
                            "win3_mean": round(float(np.mean(ws)), 4),
                            "win3_n": len(ws)}
        print(f"[sim] {label}: whole={whole if whole is None else round(whole, 4)}  "
              f"win3_mean={np.mean(ws):.4f} ({len(ws)} 窗)")
    report["sim"] = sim_items

    # ---- 3) 自然度：SingMOS + NISQA(-tts)（singmos venv 子进程，一次跑完）----
    if args.no_nat:
        print("[nat] 跳过")
    else:
        print("[nat] running SingMOS + NISQA ...")
        nat_items = ["=".join([label, str(Path(p).resolve())]) for label, p in wavs.items()]
        nat_cmd = [str(SINGMOSV), str(ROOT / "tools/singmos/score_naturalness.py"),
                   "--out", str((out / "natmos_report.json").resolve()),
                   "--csv", str((out / "natmos_win.csv").resolve())] + nat_items
        r = subprocess.run(nat_cmd, cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        if r.returncode != 0:
            sys.exit(f"score_naturalness.py 失败 (exit {r.returncode})")
        print((r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout))
        with open(out / "natmos_report.json") as f:
            nat = json.load(f)
        nisqa_all = nat.get("nisqa", {})
        report["nat"] = {
            "singmos": {k: v for k, v in nat.get("files", {}).items()},
            "nisqa": {m: {k: nisqa_all.get(m, {}).get(k, {}) for k in wavs}
                      for m in ("nisqa", "nisqa_tts")},
        }
        report["nat"]["csv"] = str(out / "natmos_win.csv")

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return report


def print_table(report: dict) -> None:
    cer = report.get("cer") or {}
    sim = report.get("sim") or {}
    singmos = report.get("nat", {}).get("singmos") or {}
    nisqa = report.get("nat", {}).get("nisqa", {}).get("nisqa_tts") or {}
    print("\n==================== TTS checkpoint 对比 ====================")
    print(f"{'arm':<6}{'CER↓':>8}{'SIM整句':>9}{'SIM3s':>9}{'SingMOS↑':>9}{'NISQA-tts↑':>10}")
    for label, *_ in ARMS:
        c = cer.get(label, {})
        s = sim.get(label, {})
        sm = singmos.get(label, {})
        n = nisqa.get(label, {})
        whole = s.get("whole")
        whole_s = f"{whole:.4f}" if whole is not None else "  -----"
        print(f"{label:<6}{c.get('cer', float('nan')):>8.4f}"
              f"{whole_s:>9}"
              f"{s.get('win3_mean', float('nan')):>9.4f}"
              f"{sm.get('mean', float('nan')):>9.3f}{n.get('mos_pred', float('nan')):>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "score", "all"])
    ap.add_argument("--out", default=ROOT / "data/songs/情歌/mv/tts_eval")
    ap.add_argument("--text", default=str(TEXT), help="口白/评测文本文件（默认短口白 intro.txt）")
    ap.add_argument("--speed", type=float, default=1.0, help="口白语速（恒定对比，默认 1.0）")
    ap.add_argument("--ref-text", default=None, help="参考音频转写（缺省从 girlish.list 反查）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="重新生成口白")
    ap.add_argument("--no-cer", action="store_true")
    ap.add_argument("--no-nat", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.step in ("gen", "all"):
        gen(out, args)
    if args.step in ("score", "all"):
        report = score(out, args)
        print_table(report)
        print(f"\n完整报告: {out / 'report.json'}")


if __name__ == "__main__":
    main()
