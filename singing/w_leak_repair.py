#!/usr/bin/env python
"""漏音修复阶段（流水线 worker）：按 leak_report.json 的建议跨度定点重转+对齐+拼接。

流程（每跨度）：
  1. 切 lead+F0（+2s pad）→ 用固定臂（--arm-seed/--arm-alpha/--arm-cfg）重转
  2. 双模型打分：仅当 重转 excess < 基线 excess 才替换（增益门，防止越修越漏）
  3. 包络互相关对齐 + 公共静音点拼接（回退原版气口 + 加长淡化）
执行前把当前 vocal.wav 复制为 vocal_raw.wav（修复前的原始 SVC 输出）。
全部跨度处理完后再跑一遍 3s 双模型扫描 → leak_report_after.json，打印前后对比。

用法（由 sing.py leak-repair 阶段调用）:
  seed-vc/.venv/bin/python singing/w_leak_repair.py --song-dir data/songs/雨爱 \
      --report data/songs/雨爱/leak_report.json \
      --arm-seed 1 --arm-alpha 0.2 --arm-cfg 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_splice import (  # noqa: E402
    PAD, SR, XF, Scorer, find_common_silence, find_shift, orig_silence_mid, run_svc,
)


def region_excess(scorer, wav_path, t0, t1):
    return scorer.region_excess(wav_path, t0, t1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--arm-seed", type=int, default=1)
    ap.add_argument("--arm-alpha", type=float, default=0.2)
    ap.add_argument("--arm-cfg", type=float, default=0.5)
    ap.add_argument("--vocal", default=None, help="默认 <song-dir>/vocal.wav")
    ap.add_argument("--raw-vocal", default=None, help="修复前备份名，默认 vocal_raw.wav")
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="替换门槛：重转 excess 至少比基线低这么多才替换（默认 0 = 有改善就换）")
    args = ap.parse_args()

    d = os.path.abspath(args.song_dir.rstrip("/"))
    report_path = os.path.abspath(args.report)
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    spans = report.get("spans") or []
    if not spans:
        print("报告无建议修复跨度，无需修复")
        return

    vocal = os.path.abspath(args.vocal or os.path.join(d, "vocal.wav"))
    raw_vocal = os.path.abspath(args.raw_vocal or os.path.join(d, "vocal_raw.wav"))
    lead = os.path.join(d, "sep2", "lead.wav")
    f0f = os.path.join(d, "target_f0.npy")
    prompt_wav = os.path.join(d, "prompt_ext.wav")
    prompt_f0 = os.path.join(d, "prompt_ext_f0.npy")
    ref = report["ref"]
    impostors = report.get("impostors") or []
    for p in [vocal, lead, f0f, prompt_wav, prompt_f0, ref]:
        if not os.path.exists(p):
            raise SystemExit(f"缺少文件: {p}")

    # ---- 备份原始 SVC 输出 ----
    shutil.copy2(vocal, raw_vocal)
    print(f"原始 SVC 输出已备份: {raw_vocal}")

    scorer = Scorer(ref, lead, impostors)
    lead_w, lead_sr = sf.read(lead)
    f0 = np.load(f0f)
    vocal_w, _ = sf.read(vocal)
    tmp = tempfile.mkdtemp(prefix="wrepair_")
    print(f"workdir {tmp}")

    changed = 0
    for si, (t0, t1) in enumerate(spans):
        print(f"\n=== 跨度 {si}: [{t0:.2f}s, {t1:.2f}s] ===")
        i0, i1 = int((t0 - PAD) * 50), int((t1 + PAD) * 50)
        j0, j1 = int((t0 - PAD) * lead_sr), int((t1 + PAD) * lead_sr)
        lf = os.path.join(tmp, f"lead{si}.wav")
        ff = os.path.join(tmp, f"f0{si}.npy")
        sf.write(lf, lead_w[j0:j1], lead_sr)
        np.save(ff, f0[i0:i1])

        tag = f"s{args.arm_seed}_a{args.arm_alpha:g}_c{args.arm_cfg:g}"
        gen = run_svc(prompt_wav, prompt_f0, ref, lf, ff,
                      os.path.join(tmp, f"r{si}_{tag}"),
                      args.arm_cfg, args.arm_alpha, args.arm_seed)
        reg, reg_sr = sf.read(gen)
        assert reg_sr == SR

        base_ex = scorer.region_excess(vocal, t0, t1)
        gen_ex = scorer.region_excess(gen, PAD, PAD + (t1 - t0))
        print(f"  excess: 基线 {base_ex:+.4f} → 重转 {gen_ex:+.4f} (Δ{gen_ex-base_ex:+.4f})")
        if gen_ex >= base_ex - args.min_gain:
            print("  增益门未通过（重转不漏音下降），保留原样")
            continue

        shift, corr = find_shift(vocal_w, reg, t0, t1)
        b_start = find_common_silence(vocal_w, reg, t0, shift, t0, lo=-0.5, hi=0.8)
        b_end = find_common_silence(vocal_w, reg, t0, shift, t1, lo=-0.8, hi=0.5)
        xf_use = XF
        if b_start is None or b_end is None or b_end - b_start < 1.0:
            xf_use = 0.25
            fb_s = orig_silence_mid(vocal_w, t0, lo=-0.5, hi=0.8)
            fb_e = orig_silence_mid(vocal_w, t1, lo=-0.8, hi=0.5)
            b_start, b_end = fb_s or (t0 - 0.3), fb_e or (t1 + 0.2)
            print(f"  ⚠ 未找到公共静音，回退原版气口 [{b_start:.3f}, {b_end:.3f}]（淡化 {xf_use}s）")
        print(f"  对齐 Δ={shift:+.3f}s，拼接边界 [{b_start:.3f}s, {b_end:.3f}s]")

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
        changed += 1
        print(f"  已替换 [{b_start:.3f}, {b_end:.3f}]")

    if not changed:
        print("\n所有跨度均未通过增益门，vocal.wav 保持不变（备份 vocal_raw.wav 已生成）")
        return

    # ---- 写回 + 复检 ----
    out_tmp = vocal + ".repairing.wav"
    sf.write(out_tmp, vocal_w, SR)
    os.replace(out_tmp, vocal)
    print(f"\n修复后 vocal.wav 已写出（原始输出在 vocal_raw.wav）")

    after_report = os.path.join(d, "leak_report_after.json")
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "w_leak_detect.py"),
           "--song-dir", d, "--ref", ref, "--vocal", vocal, "--out-json", after_report]
    for p in impostors:
        cmd += ["--impostor", p]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("---- 修复后复检 ----")
    print(r.stdout[-2000:] if r.returncode == 0 else r.stderr[-1500:])

    import torch
    del scorer
    torch.cuda.empty_cache()

    if os.path.exists(after_report):
        with open(after_report, encoding="utf-8") as f:
            after = json.load(f)
        before = report["models"]
        print("\n全曲 3s excess 对比:")
        print(f"  {'':<10}{'修复前':>9}{'修复后':>9}")
        for m, key in [("eres2net", "eres2net"), ("campplus", "campplus")]:
            b = before[m]["mean_excess_3s"]
            a = after["models"][m]["mean_excess_3s"]
            print(f"  {m:<10}{b:>9.4f}{a:>9.4f}  (Δ{a-b:+.4f})")
        n_cons_b = sum(1 for w in report["windows"] if w["consensus"])
        n_cons_a = sum(1 for w in after["windows"] if w["consensus"])
        print(f"  共识漏音窗: {n_cons_b} → {n_cons_a}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
