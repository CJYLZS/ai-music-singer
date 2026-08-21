#!/usr/bin/env python3
"""VoxCPM2 参考长度扫描：更长参考是否会显著变好？

代码层面 VoxCPM2 参考没有硬上限（_encode_wav 整条编码，~6.25 token/s，
max_length=8192 预算充裕）——唯一样本量决定。跑不同参考长度 + 同一段长文本，
用 SIM vs 整条 ref_talk_sing（3s 窗）判断加长收益。

用法（GPT-SoVITS venv；vox 生成作为子进程，vox venv 自带）：
  GPT-SoVITS/.venv/bin/python singing/eval_vox_ref_length.py --lens 6.4 10 15 30 60
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/rookie/voice")
GIRLH_DIR = ROOT / "data/voices/girlish_sliced"
CUR = ROOT / "data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav"
TALK_SRC = ROOT / "data/voices/ref_talk_sing.wav"
TALK_LIMIT_S = 261.0                                          # 说话段：ref_talk_sing 前 261s
REF_SRC = ROOT / "data/voices/ref_talk_sing.wav"
LONG_TEXT = ROOT / "data/songs/情歌/mv/tts_eval_long_text.txt"
OUT = ROOT / "data/songs/情歌/mv/vox_ref_len"

VOXV = ROOT / "tools/voxcpm2/.venv/bin/python"
TEST_GEN = ROOT / "tools/voxcpm2/test_gen.py"
STYLE = "high pitch, sweet young girl voice, steady pace"


def build_concat(path: Path, source_dir=GIRLH_DIR):
    """按句拼接 girlish_sliced（备选材料，默认不用）。"""
    import soundfile as sf
    files = sorted(source_dir.glob("*.wav"),
                   key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1)))
    if path.exists():
        return
    chunks = []
    for p in files:
        x, sr = sf.read(str(p), dtype="float32")
        chunks.append(x)
    import numpy as np
    out = np.concatenate(chunks)
    sf.write(str(path), out, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=float, nargs="+", default=[6.4, 10.0, 15.0, 30.0, 60.0, 120.0])
    ap.add_argument("--mat", choices=["talk", "concat"], default="talk",
                    help="talk=ref_talk_sing 前 261s 直接连续窗口（默认）；"
                         "concat=girlish_sliced 按句硬拼接")
    ap.add_argument("--start", type=float, default=1.0, help="talk 模式下连续窗口起点(秒)，避开开头气口")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import soundfile as sf
    import eval_sim

    # 说话段连续材料：ref_talk_sing[0, TALK_LIMIT_S]
    talk, tsr = sf.read(str(TALK_SRC), dtype="float32", always_2d=True)
    talk = talk.mean(axis=1)
    n_talk = int(TALK_LIMIT_S * tsr)
    talk = talk[:n_talk]
    print(f"[mat] {args.mat}: 参考材料 {TALK_LIMIT_S:.0f}s @{tsr}Hz（连续，未拼接）")

    text = LONG_TEXT.read_text(encoding="utf-8").strip()
    prompt = f"({STYLE}) {text}"

    # ref 准备
    refs = {}
    for L in args.lens:
        if args.mat == "concat" and abs(L - 6.4) < 1e-6:
            p = CUR
        else:
            p = OUT / f"ref_{args.mat}{L:g}s.wav"
            if args.mat == "talk":
                s0 = int((args.start) * tsr); n = int(L * tsr)
                s0 = min(s0, len(talk) - n)
                if not p.exists() or args.force:
                    sf.write(str(p), talk[s0:s0 + n], tsr)
            else:
                concat = OUT / "girlish_all.wav"
                build_concat(concat)
                cc, csr = sf.read(str(concat), dtype="float32")
                n = int(L * csr)
                if not p.exists() or args.force:
                    sf.write(str(p), cc[:n], csr)
        refs[L] = p

    # SV 参考
    sv = eval_sim.SV(device="cuda:0")
    ref_emb = sv.emb(eval_sim.load16k(str(REF_SRC)))
    import torch
    torch.cuda.empty_cache()

    report = {}
    for L, p in refs.items():
        dur = sf.info(str(p)).frames / sf.info(str(p)).samplerate
        gen = OUT / f"gen_{args.mat}{L:g}s.wav"
        if not gen.exists() or args.force:
            print(f"[vox] ref {L:g}s ({p.name}) ...", flush=True)
            # 参考已按句拼接，无转写要求（ref 模式不配 prompt_text）
            r = subprocess.run(
                [str(VOXV), str(TEST_GEN),
                 "--ref", str(p.resolve()),
                 "--text", prompt,
                 "--out", str(gen.resolve()),
                 "--seed", "42"],
                cwd=str(ROOT), env=dict(__import__("os").environ, HF_HUB_OFFLINE="1"),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode != 0:
                print(f"  !! ref {L:g}s 生成失败:\n" + "\n".join(r.stdout.splitlines()[-15:]))
                continue
        wav = eval_sim.load16k(str(gen))
        w = float(eval_sim.cos(sv.emb(wav), ref_emb))
        wlen = 3 * 16000
        ws = np.array([eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
                       for s in range(0, len(wav) - wlen + 1, wlen)])
        gdur = len(wav) / 16000
        report[L] = {"ref": str(p.name), "ref_dur": round(dur, 1), "gen_dur": round(gdur, 1),
                     "sim_full_whole": round(w, 4), "sim_full_win3": round(float(ws.mean()), 4),
                     "win3_n": len(ws)}
        print(f"  ref {L:g}s: SIM whole={w:.4f} win3={ws.mean():.4f} ({len(ws)}窗)   "
              f"gen={gdur:.0f}s")

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"\n{'ref':>6}{'ref_dur':>9}{'gen_dur':>9}{'SIM整段':>10}{'SIM3s':>10}")
    for L in report:
        r = report[L]
        print(f"{L:g}s{L:>3}{r['ref_dur']:>9}{r['gen_dur']:>9}{r['sim_full_whole']:>10.4f}"
              f"{r['sim_full_win3']:>10.4f}")
    print(f"\n报告: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
