#!/usr/bin/env python3
"""对候选参考切片做多采样确认：同一切片多次 zs 生成（每枪独立采样），
统计 SIM(vs 整条 ref_talk_sing) 的走势，判断领先是否稳定（仓库惯例：多 seed 配对）。

用法（GPT-SoVITS venv）:
  GPT-SoVITS/.venv/bin/python singing/eval_zs_slice_confirm.py --slice s16 s01 --draws 3
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/rookie/voice")
GIRLH_DIR = ROOT / "data/voices/girlish_sliced"
REF_SRC = ROOT / "data/voices/ref_talk_sing.wav"
LONG_TEXT = ROOT / "data/songs/情歌/mv/tts_eval_long_text.txt"
OUT = ROOT / "data/songs/情歌/mv/zs_slice_scan/confirm"

GPT = ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SOVITS = ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="append", required=True, dest="labels",
                    help="dirscan 标签，如 s16 s01（用 dirscan_report 的 label）")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import soundfile as sf
    import eval_sim

    files = sorted(GIRLH_DIR.glob("*.wav"),
                   key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1)))
    label_path = {f"s{i:02d}": p for i, p in enumerate(files, 1)}
    gl = [l.rstrip("\n").split("|") for l in
          (ROOT / "data/voices/girlish.list").read_text(encoding="utf-8").splitlines()]
    gl_tr = {Path(g[0]).name: g[3].strip() for g in gl if len(g) == 4 and len(g[3].strip())}

    sv = eval_sim.SV(device="cuda:0")
    ref_emb = sv.emb(eval_sim.load16k(str(REF_SRC)))
    import torch
    torch.cuda.empty_cache()

    report = {}
    for label in args.labels:
        if label not in label_path:
            sys.exit(f"未知标签 {label}（可用: " + " ".join(sorted(label_path)) + "）")
        p = label_path[label]
        tr = gl_tr.get(p.name, "")
        draws = []
        for k in range(1, args.draws + 1):
            gen = OUT / f"{label}_d{k}.wav"
            if not gen.exists() or args.force:
                import subprocess
                cmd = [str(ROOT / "GPT-SoVITS/.venv/bin/python"),
                       str(ROOT / "GPT-SoVITS/infer.py"),
                       "-t", "@" + str(LONG_TEXT.resolve()),
                       "-o", str(gen.resolve()),
                       "--gpt", str(GPT.resolve()),
                       "--sovits", str(SOVITS.resolve()),
                       "--ref", str(p.resolve()), "--speed", "1.0", "-q"]
                if tr:
                    cmd += ["--ref-text", tr]
                with open(OUT / f"{label}_d{k}.log", "w") as f:
                    r = subprocess.run(cmd, cwd=str(ROOT / "GPT-SoVITS"),
                                       stdout=f, stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    print(f"  !! {label} d{k} 失败"); continue
            wav = eval_sim.load16k(str(gen))
            w = float(eval_sim.cos(sv.emb(wav), ref_emb))
            wlen = 3 * 16000
            ws = np.array([eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
                           for s in range(0, len(wav) - wlen + 1, wlen)])
            draws.append({"draw": k, "whole": round(w, 4), "win3": round(float(ws.mean()), 4),
                          "win3_n": len(ws)})
            print(f"  {label} d{k}: whole={w:.4f} win3={ws.mean():.4f}")
        ws_ = np.array([d["win3"] for d in draws]); wh_ = np.array([d["whole"] for d in draws])
        stat = {"file": p.name, "dur": round(sf.info(str(p)).frames / sf.info(str(p)).samplerate, 1),
                "ref_text": tr[:24], "draws": draws,
                "whole": round(float(wh_.mean()), 4) if len(wh_) else None,
                "whole_std": round(float(wh_.std()), 4) if len(wh_) > 1 else None,
                "win3": round(float(ws_.mean()), 4) if len(ws_) else None,
                "win3_std": round(float(ws_.std()), 4) if len(ws_) > 1 else None,
                "win3_med": round(float(np.median(ws_)), 4) if len(ws_) else None,
                "win3_min": round(float(ws_.min()), 4) if len(ws_) else None,
                "win3_max": round(float(ws_.max()), 4) if len(ws_) else None}
        report[label] = stat

    (OUT / "confirm_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n================== 多采样确认 ==================")
    print(f"{'slice':<8}{'dur':>5}{'win3 mean±std':>18}{'win3 [min,max]':>20}{'whole':>10}")
    for label, st in report.items():
        if st["win3_std"] is None:
            print(f"{label:<8}{st['dur']:>5.1f}{st['win3']:>18.4f}{'':>20}{st['whole']:>10.4f}")
        else:
            rng = "[%s,%s]" % (st['win3_min'], st['win3_max'])
            ms = "%s±%s" % (st['win3'], st['win3_std'])
            print(f"{label:<8}{st['dur']:>5.1f}{ms:>18}{rng:>20}{st['whole']:>10.4f}")
    print(f"\n报告: {OUT / 'confirm_report.json'}")


if __name__ == "__main__":
    main()
