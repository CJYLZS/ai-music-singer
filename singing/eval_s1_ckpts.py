#!/usr/bin/env python3
"""S1（GPT）选型 / A-B 扫点：固定 SoVITS 与参考音频，只换 GPT 权重。

为什么需要独立脚本：既有 `eval_tts_ckpts.py` 的每个臂同时替换 GPT 与 SoVITS
（e15 = GPT e15 + SoVITS e15），所以它的 CER 差异里混着 S2 的贡献，**不能用来选 S1**。
这里把 SoVITS 钉死在一个 checkpoint 上，只扫 GPT，差异才可归因到 S1。

两种用法：
  1) 单家族选型   —— `--gpt-prefix girlish`，选 CER 最优 epoch
  2) 跨家族 A/B   —— `--gpt-prefix girlish --gpt-prefix talksing_s2mix50`，
     即"纯说话 S1"对"混 50% 唱歌素材 S1"，验证唱歌素材是否伤吐字

判据口径：
  - **CER 优先**（吐字），SIM / 自然度只作参考。
  - **必须多 seed 配对**：AR 解码走 multinomial，不播种时同一权重重跑 CER 极差可达
    0.009（AGENTS.md 记录），单枪结论会被噪声翻转。默认 3 个 seed 取中位。
  - **必须长文本**：短口白测不出吐字差（AGENTS.md "短句区不出吐字差"）。
  - **必须看时长**：混入唱歌素材的预期损伤不是"字错"而是"字被拖长"。S1 学的是
    phoneme→semantic token 的时长分布，唱段的 phoneme/sec 是 2.2-6.9 而说话段是
    3.8-9.9（本仓实测），所以同一文本合出来更长 = 语速被唱腔拖慢。CER 对此完全盲，
    因为 ASR 只判离散字。故这里把 `dur` / `char_per_sec` 与 CER 并列为一等指标。

CER 参考文本 = 合成时喂进去的原始文案（TTS 有 ground-truth 文本，不需要 ASR 代理），
所以这里直接调 eval_cer.py 的 --ref-text 路径。FunASR VAD 非确定性由 eval_cer.py 内部
播种处理 —— 不要绕过它自己调 ASR。

用法（一律 GPT-SoVITS venv）:
  # 纯说话 vs 混唱歌，3 seed 配对
  GPT-SoVITS/.venv/bin/python singing/eval_s1_ckpts.py all \
      --gpt-prefix girlish --gpt-prefix talksing_s2mix50 \
      --sovits talksing_s2mix50_e12_s240.pth --out data/experiments/s1_mix_ab
  # 只跑生成 / 只打分
  GPT-SoVITS/.venv/bin/python singing/eval_s1_ckpts.py gen ...
  GPT-SoVITS/.venv/bin/python singing/eval_s1_ckpts.py score ...
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPTV = ROOT / "GPT-SoVITS/.venv/bin/python"
SINGMOSV = ROOT / "tools/singmos/.venv/bin/python"

GPTW = ROOT / "GPT-SoVITS/GPT_weights_v2Pro"
SOVW = ROOT / "GPT-SoVITS/SoVITS_weights_v2Pro"
S1_ZS = ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

# 长文本（510 字），AGENTS.md：短口白拉不开吐字差，选型必须长文本
TEXT = ROOT / "data/songs/情歌/mv/tts_eval_long_text.txt"
# 参考音频对所有臂**必须一致** —— 它是推理期输入，与训练集无关；换掉它就混入了新变量
REF_AUDIO = ROOT / "data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav"
GIRLISH = ROOT / "data/voices/girlish.list"


def build_arms(prefixes, epochs, sovits: Path, zs: bool):
    """返回 [(label, gpt_path, sovits_path, note)]。label 同时是输出子目录名。"""
    out = []
    for pre in prefixes:
        for e in epochs:
            ck = GPTW / f"{pre}-e{e}.ckpt"
            out.append((f"{pre}-e{e}", ck, sovits, f"GPT {pre} e{e} + 固定 SoVITS"))
    if zs:
        out.append(("gpt-zs", S1_ZS, sovits, "零样本 S1 + 同一固定 SoVITS"))
    return out


def lookup_ref_text(ref_wav: Path):
    for line in GIRLISH.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("|")
        if len(parts) == 4 and Path(parts[0]).name == ref_wav.name:
            return parts[3].strip()
    return None


def wav_path(out: Path, label: str, seed: int) -> Path:
    return out / label / f"seed{seed}" / "speech.wav"


def gen(out: Path, args, ARMS) -> None:
    ref_text = args.ref_text or lookup_ref_text(REF_AUDIO)
    todo = [(l, g, s, sd) for l, g, s, _ in ARMS for sd in args.seeds]
    for i, (label, gpt, sovits, seed) in enumerate(todo, 1):
        wav = wav_path(out, label, seed)
        wav.parent.mkdir(parents=True, exist_ok=True)
        if wav.exists() and not args.force:
            print(f"[skip {i}/{len(todo)}] {label} seed{seed}")
            continue
        if not gpt.exists():
            sys.exit(f"error: 缺少 GPT 权重 {gpt}")
        cmd = [str(GPTV), str(ROOT / "GPT-SoVITS/infer.py"),
               "-t", "@" + str(TEXT.resolve()),
               "-o", str(wav.resolve()),
               "--gpt", str(gpt.resolve()),
               "--sovits", str(sovits.resolve()),
               "--ref", str(REF_AUDIO.resolve()),
               "--speed", str(args.speed),
               "--seed", str(seed), "-q"]
        if ref_text:
            cmd += ["--ref-text", ref_text]
        print(f"[gen {i}/{len(todo)}] {label} seed{seed}: {gpt.name} + {sovits.name}")
        with open(wav.parent / "tts.log", "w") as f:
            r = subprocess.run(cmd, cwd=str(ROOT / "GPT-SoVITS"),
                               stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"error: 生成失败 {label} seed{seed}（见 {wav.parent / 'tts.log'}）")
    print("生成完成。")


def score(out: Path, args, ARMS) -> dict:
    sys.path.insert(0, str(ROOT))
    import eval_sim
    import numpy as np
    import soundfile as sf
    import torch

    # (label, seed) -> wav
    items = {(l, sd): wav_path(out, l, sd) for l, *_ in ARMS for sd in args.seeds}
    for k, p in items.items():
        if not p.exists():
            sys.exit(f"缺少 {k} 的 speech.wav（先跑 gen）")

    report = {"sovits_fixed": Path(ARMS[0][2]).name, "text": str(TEXT), "speed": args.speed,
              "ref": str(REF_AUDIO), "seeds": args.seeds,
              "arms": {l: {"gpt": Path(g).name, "sovits": Path(s).name, "note": n}
                       for l, g, s, n in ARMS}}

    # ---- 0) 时长 / 语速（一等指标，见 docstring）----
    dur = {}
    for (label, sd), p in items.items():
        info = sf.info(str(p))
        dur[f"{label}|{sd}"] = round(info.frames / info.samplerate, 3)
    report["dur"] = dur

    # ---- 1) CER（主判据）：参考文本 = 合成用的原始文案 ----
    # 一次调用里塞进全部 wav：eval_cer.py 每次 generate 前重置种子，所以顺序无关
    cer = {}
    if args.no_cer:
        print("[cer] 跳过")
    else:
        labels = [f"{l}|{sd}" for (l, sd) in items]
        cmd = [str(GPTV), str(ROOT / "singing/eval_cer.py"),
               "--src", str(REF_AUDIO.resolve()),
               "--ref-text", str(TEXT.resolve()),
               "--gen"] + [str(p.resolve()) for p in items.values()] + \
              ["--labels"] + labels + ["--win", "10"]
        print(f"[cer] running on {len(labels)} files ...")
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        print(r.stdout)
        lset = set(labels)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 7 and parts[0] in lset:
                try:
                    cer[parts[0]] = {"cer": float(parts[1]), "S": int(parts[2]),
                                     "D": int(parts[3]), "I": int(parts[4]),
                                     "hyp_len": int(parts[5])}
                except ValueError:
                    continue
        if r.returncode != 0:
            sys.exit(f"eval_cer.py 失败 (exit {r.returncode})")
    report["cer"] = cer

    # ---- 2) SIM（参考项）----
    sim = {}
    if args.no_sim:
        print("[sim] 跳过")
    else:
        sv = eval_sim.SV(device=args.device)
        ref_emb = sv.emb(eval_sim.load16k(str(REF_AUDIO)))
        torch.cuda.empty_cache()
        wlen = int(3.0 * 16000)
        for (label, sd), p in items.items():
            wav = eval_sim.load16k(str(p))
            ws = [eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
                  for s in range(0, len(wav) - wlen + 1, wlen)]
            sim[f"{label}|{sd}"] = {"win3_mean": round(float(np.mean(ws)), 4),
                                    "win3_n": len(ws)}
            print(f"[sim] {label} seed{sd}: {np.mean(ws):.4f} ({len(ws)} 窗)")
    report["sim"] = sim

    # ---- 3) 自然度（参考项）：只测每臂第一个 seed，省时间 ----
    if args.no_nat:
        print("[nat] 跳过")
    else:
        print("[nat] running SingMOS + NISQA (每臂首个 seed) ...")
        sd0 = args.seeds[0]
        pairs = {l: wav_path(out, l, sd0) for l, *_ in ARMS}
        argv = ["=".join([l, str(p.resolve())]) for l, p in pairs.items()]
        cmd = [str(SINGMOSV), str(ROOT / "tools/singmos/score_naturalness.py"),
               "--out", str((out / "natmos_report.json").resolve()),
               "--csv", str((out / "natmos_win.csv").resolve())] + argv
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
        if r.returncode != 0:
            sys.exit(f"score_naturalness.py 失败 (exit {r.returncode})")
        nat = json.loads((out / "natmos_report.json").read_text())
        nis = nat.get("nisqa", {})
        report["nat"] = {"singmos": nat.get("files", {}),
                         "nisqa": {m: {k: nis.get(m, {}).get(k, {}) for k in pairs}
                                   for m in ("nisqa", "nisqa_tts")}}

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return report


def _agg(d: dict, label: str, seeds, key=None):
    """取某臂在各 seed 上的值列表 -> (中位, 最小, 最大, n)。"""
    vals = []
    for sd in seeds:
        v = d.get(f"{label}|{sd}")
        if v is None:
            continue
        vals.append(v if key is None else v.get(key))
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None, 0
    return statistics.median(vals), min(vals), max(vals), len(vals)


def print_table(report: dict, ARMS, seeds) -> None:
    cer = report.get("cer") or {}
    sim = report.get("sim") or {}
    dur = report.get("dur") or {}
    sm = report.get("nat", {}).get("singmos") or {}
    nq = report.get("nat", {}).get("nisqa", {}).get("nisqa_tts") or {}
    nan = float("nan")

    print(f"\n===== S1 扫点（SoVITS 固定 = {report.get('sovits_fixed')}，"
          f"ref 固定，{len(seeds)} seed 取中位）=====")
    print(f"{'arm':<24}{'CER中位↓':>10}{'CER范围':>16}{'D':>5}{'时长s':>8}"
          f"{'字/秒':>7}{'SIM':>8}{'SingMOS':>9}{'NISQAt':>8}")
    rows = []
    for label, *_ in ARMS:
        cm, clo, chi, n = _agg(cer, label, seeds, "cer")
        dm, dlo, dhi, _ = _agg(dur, label, seeds)
        sm_, _, _, _ = _agg(sim, label, seeds, "win3_mean")
        hyp, _, _, _ = _agg(cer, label, seeds, "hyp_len")
        dmed, _, _, _ = _agg(cer, label, seeds, "D")
        cps = (hyp / dm) if (hyp and dm) else None
        rows.append((label, cm, dm, cps, sm_))
        print(f"{label:<24}{cm if cm is not None else nan:>10.4f}"
              f"{f'{clo:.4f}-{chi:.4f}' if n else '-':>16}"
              f"{int(dmed) if dmed is not None else -1:>5}"
              f"{dm if dm is not None else nan:>8.2f}"
              f"{cps if cps else nan:>7.2f}"
              f"{sm_ if sm_ is not None else nan:>8.4f}"
              f"{(sm.get(label, {}) or {}).get('mean', nan):>9.3f}"
              f"{(nq.get(label, {}) or {}).get('mos_pred', nan):>8.3f}")

    ok = [r for r in rows if r[1] is not None]
    if not ok:
        return
    best = min(ok, key=lambda r: r[1])
    print(f"\nCER 中位最优臂: {best[0]} (CER {best[1]:.4f})  ← 主判据")

    # 家族聚合：跨家族 A/B 的结论应看整族分布，而不是各族的最好一枪
    fams = {}
    for label, cm, dm, cps, sm_ in ok:
        fam = label.rsplit("-e", 1)[0] if "-e" in label else label
        fams.setdefault(fam, []).append((cm, dm, cps))
    if len(fams) > 1:
        print(f"\n----- 家族聚合（每族 {len(next(iter(fams.values())))} 个 epoch）-----")
        print(f"{'family':<24}{'CER中位':>10}{'CER最优':>10}{'时长中位':>10}{'字/秒中位':>10}")
        for fam, vs in fams.items():
            cs = [v[0] for v in vs]
            ds = [v[1] for v in vs if v[1] is not None]
            ps = [v[2] for v in vs if v[2] is not None]
            print(f"{fam:<24}{statistics.median(cs):>10.4f}{min(cs):>10.4f}"
                  f"{statistics.median(ds) if ds else nan:>10.2f}"
                  f"{statistics.median(ps) if ps else nan:>10.2f}")
    print("\n判读：CER 差值需超过 ~0.009 的采样噪声底才可分辨（AGENTS.md）；"
          "时长/字每秒是唱腔拖慢语速的直接证据，CER 对它盲。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "score", "all"])
    ap.add_argument("--out", default=str(ROOT / "data/experiments/s1_selection"))
    ap.add_argument("--sovits", default="talksing_s2mix50_e12_s240.pth",
                    help="固定的 SoVITS 权重（文件名或路径）。选 S1 时它必须全臂一致")
    ap.add_argument("--gpt-prefix", action="append", default=None,
                    help="GPT 权重家族前缀，可重复。默认 girlish")
    ap.add_argument("--epochs", default="5,10,15,20,25,30")
    ap.add_argument("--seeds", default="1234,2234,3234",
                    help="配对 seed 列表。单 seed 结论会被 AR 采样噪声翻转")
    ap.add_argument("--zs", action="store_true", help="额外加零样本 S1 臂")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cer", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-nat", action="store_true")
    args = ap.parse_args()

    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    epochs = [s.strip() for s in args.epochs.split(",") if s.strip()]
    prefixes = args.gpt_prefix or ["girlish"]

    sov = Path(args.sovits)
    if not sov.is_absolute():
        sov = SOVW / args.sovits
    if not sov.exists():
        sys.exit(f"error: 找不到固定 SoVITS 权重 {sov}")
    ARMS = build_arms(prefixes, epochs, sov, args.zs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.step in ("gen", "all"):
        gen(out, args, ARMS)
    if args.step in ("score", "all"):
        print_table(score(out, args, ARMS), ARMS, args.seeds)
        print(f"\n完整报告: {out / 'report.json'}")


if __name__ == "__main__":
    main()
