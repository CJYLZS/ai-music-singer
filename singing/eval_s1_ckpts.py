#!/usr/bin/env python3
"""S1（GPT）选型扫点：固定 SoVITS，只换 GPT epoch，按吐字 CER 优先排序。

为什么需要独立脚本：既有 `eval_tts_ckpts.py` 的每个臂同时替换 GPT 与 SoVITS
（e15 = GPT e15 + SoVITS e15），所以它的 CER 差异里混着 S2 的贡献，**不能用来选 S1**。
这里把 SoVITS 钉死在一个 checkpoint 上，只扫 GPT，差异才可归因到 S1。

判据口径（本轮用户指定）：吐字 CER 优先，SIM / 自然度只作参考。
文本必须用长文本：AGENTS.md 已记录短口白测不出吐字差（"短句区不出吐字差"）。

CER 参考文本 = 合成时喂进去的原始文案（TTS 有 ground-truth 文本，不需要 ASR 代理），
所以这里直接调 eval_cer.py 的 --ref-text 路径。FunASR VAD 非确定性由 eval_cer.py 内部
播种处理 —— 不要绕过它自己调 ASR。

用法（一律 GPT-SoVITS venv）:
  GPT-SoVITS/.venv/bin/python singing/eval_s1_ckpts.py gen
  GPT-SoVITS/.venv/bin/python singing/eval_s1_ckpts.py score
  GPT-SoVITS/.venv/bin/python singing/eval_s1_ckpts.py all --sovits girlish_e15_s330.pth
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPTV = ROOT / "GPT-SoVITS/.venv/bin/python"
SINGMOSV = ROOT / "tools/singmos/.venv/bin/python"

GPTW = ROOT / "GPT-SoVITS/GPT_weights_v2Pro"
SOVW = ROOT / "GPT-SoVITS/SoVITS_weights_v2Pro"
V2PRO = ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro"
S1_ZS = ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

# 长文本（510 字），AGENTS.md：短口白拉不开吐字差，选型必须长文本
TEXT = ROOT / "data/songs/情歌/mv/tts_eval_long_text.txt"
REF_AUDIO = ROOT / "data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav"
GIRLISH = ROOT / "data/voices/girlish.list"

# 只扫 GPT epoch；SoVITS 由 --sovits 固定
S1_EPOCHS = ["e5", "e10", "e15", "e20", "e25", "e30"]


def arms(sovits: Path):
    out = [(f"gpt-{e}", GPTW / f"girlish-{e}.ckpt", sovits, f"GPT {e} + 固定 SoVITS") for e in S1_EPOCHS]
    out.append(("gpt-zs", S1_ZS, sovits, "零样本 S1 + 同一固定 SoVITS"))
    return out


def lookup_ref_text(ref_wav: Path):
    for line in GIRLISH.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("|")
        if len(parts) == 4 and Path(parts[0]).name == ref_wav.name:
            return parts[3].strip()
    return None


def gen(out: Path, args, ARMS) -> None:
    ref_text = args.ref_text or lookup_ref_text(REF_AUDIO)
    for label, gpt, sovits, note in ARMS:
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        wav = d / "speech.wav"
        if wav.exists() and not args.force:
            print(f"[skip] {label}: {wav}")
            continue
        if not gpt.exists():
            sys.exit(f"error: 缺少 GPT 权重 {gpt}")
        cmd = [str(GPTV), str(ROOT / "GPT-SoVITS/infer.py"),
               "-t", "@" + str(TEXT.resolve()),
               "-o", str(wav.resolve()),
               "--gpt", str(gpt.resolve()),
               "--sovits", str(sovits.resolve()),
               "--ref", str(REF_AUDIO.resolve()),
               "--speed", str(args.speed), "-q"]
        if ref_text:
            cmd += ["--ref-text", ref_text]
        print(f"[gen] {label}: {gpt.name} + {sovits.name}")
        with open(d / "tts.log", "w") as f:
            r = subprocess.run(cmd, cwd=str(ROOT / "GPT-SoVITS"), stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"error: 生成失败 {label}（见 {d / 'tts.log'}）")
    print("生成完成。")


def score(out: Path, args, ARMS) -> dict:
    sys.path.insert(0, str(ROOT))
    import eval_sim
    import numpy as np
    import torch

    wavs = {label: str(out / label / "speech.wav") for label, *_ in ARMS}
    for label, p in wavs.items():
        if not os.path.exists(p):
            sys.exit(f"缺少 {label} 的 speech.wav（先跑 gen）")

    report = {"sovits_fixed": Path(ARMS[0][2]).name, "text": str(TEXT), "speed": args.speed,
              "ref": str(REF_AUDIO),
              "arms": {l: {"gpt": Path(g).name, "sovits": Path(s).name, "note": n} for l, g, s, n in ARMS}}

    # ---- 1) CER（主判据）：参考文本 = 合成用的原始文案 ----
    cer_items = {}
    if args.no_cer:
        print("[cer] 跳过")
    else:
        cmd = [str(GPTV), str(ROOT / "singing/eval_cer.py"),
               "--src", str(REF_AUDIO.resolve()),
               "--ref-text", str(TEXT.resolve()),
               "--gen"] + [str(Path(p).resolve()) for p in wavs.values()] + \
              ["--labels"] + list(wavs.keys()) + ["--win", "10"]
        print("[cer] running ...")
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        print(r.stdout)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 7 and parts[0] in wavs:
                try:
                    cer = float(parts[1])
                except ValueError:
                    continue
                cer_items[parts[0]] = {"cer": cer, "S": int(parts[2]), "D": int(parts[3]),
                                       "I": int(parts[4]), "hyp_len": int(parts[5])}
        if r.returncode != 0:
            sys.exit(f"eval_cer.py 失败 (exit {r.returncode})")
    report["cer"] = cer_items

    # ---- 2) SIM（参考项）----
    sv = eval_sim.SV(device=args.device)
    ref_emb = sv.emb(eval_sim.load16k(str(REF_AUDIO)))
    torch.cuda.empty_cache()
    sim_items = {}
    wlen = int(3.0 * 16000)
    for label, p in wavs.items():
        wav = eval_sim.load16k(p)
        ws = [eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
              for s in range(0, len(wav) - wlen + 1, wlen)]
        whole = float(eval_sim.cos(sv.emb(wav), ref_emb)) if len(wav) / 16000 <= 100 else None
        sim_items[label] = {"whole": round(whole, 4) if whole else None,
                            "win3_mean": round(float(np.mean(ws)), 4), "win3_n": len(ws)}
        print(f"[sim] {label}: win3_mean={np.mean(ws):.4f} ({len(ws)} 窗)")
    report["sim"] = sim_items

    # ---- 3) 自然度（参考项）----
    if args.no_nat:
        print("[nat] 跳过")
    else:
        print("[nat] running SingMOS + NISQA ...")
        items = ["=".join([l, str(Path(p).resolve())]) for l, p in wavs.items()]
        cmd = [str(SINGMOSV), str(ROOT / "tools/singmos/score_naturalness.py"),
               "--out", str((out / "natmos_report.json").resolve()),
               "--csv", str((out / "natmos_win.csv").resolve())] + items
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        if r.returncode != 0:
            sys.exit(f"score_naturalness.py 失败 (exit {r.returncode})")
        nat = json.loads((out / "natmos_report.json").read_text())
        nis = nat.get("nisqa", {})
        report["nat"] = {"singmos": nat.get("files", {}),
                         "nisqa": {m: {k: nis.get(m, {}).get(k, {}) for k in wavs}
                                   for m in ("nisqa", "nisqa_tts")}}

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def print_table(report: dict, ARMS) -> None:
    cer = report.get("cer") or {}
    sim = report.get("sim") or {}
    sm = report.get("nat", {}).get("singmos") or {}
    nq = report.get("nat", {}).get("nisqa", {}).get("nisqa_tts") or {}
    print(f"\n===== S1 选型（SoVITS 固定 = {report.get('sovits_fixed')}）=====")
    print(f"{'arm':<8}{'CER↓':>9}{'S':>5}{'D':>5}{'I':>5}{'SIM3s':>9}{'SingMOS↑':>10}{'NISQA-tts↑':>11}")
    rows = []
    for label, *_ in ARMS:
        c = cer.get(label, {})
        s = sim.get(label, {})
        rows.append((label, c.get("cer"), c, s))
        print(f"{label:<8}{c.get('cer', float('nan')):>9.4f}{c.get('S', -1):>5}{c.get('D', -1):>5}"
              f"{c.get('I', -1):>5}{s.get('win3_mean', float('nan')):>9.4f}"
              f"{(sm.get(label, {}) or {}).get('mean', float('nan')):>10.3f}"
              f"{(nq.get(label, {}) or {}).get('mos_pred', float('nan')):>11.3f}")
    ok = [r for r in rows if r[1] is not None]
    if ok:
        best = min(ok, key=lambda r: r[1])
        print(f"\nCER 最优臂: {best[0]} (CER {best[1]:.4f})  ← 主判据")
        print("注意：差值需超过重复测量噪声才可分辨；CER 相近时按 SIM/自然度做次级排序。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "score", "all"])
    ap.add_argument("--out", default=str(ROOT / "data/experiments/s1_selection"))
    ap.add_argument("--sovits", default="girlish_e15_s330.pth",
                    help="固定的 SoVITS 权重（文件名或路径）。选 S1 时它必须全臂一致")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cer", action="store_true")
    ap.add_argument("--no-nat", action="store_true")
    args = ap.parse_args()

    sov = Path(args.sovits)
    if not sov.is_absolute():
        sov = SOVW / args.sovits
    if not sov.exists():
        sys.exit(f"error: 找不到固定 SoVITS 权重 {sov}")
    ARMS = arms(sov)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.step in ("gen", "all"):
        gen(out, args, ARMS)
    if args.step in ("score", "all"):
        print_table(score(out, args, ARMS), ARMS)
        print(f"\n完整报告: {out / 'report.json'}")


if __name__ == "__main__":
    main()
