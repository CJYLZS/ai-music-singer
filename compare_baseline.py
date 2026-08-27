#!/usr/bin/env python3
"""对比 segprompt 全量重跑前后：读各歌曲目录新 eval.txt，与 docs/song-eval-baseline.md 基线对比。

用法（GPT-SoVITS venv 或系统 python 均可）:
  python3 compare_baseline.py
输出: 新旧 SIM/SingMOS 对照表 + 变化汇总（stdout + docs/segprompt_fullrun_compare.md）
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
SONGS_DIR = ROOT / "data" / "songs"

# 基线（来自 docs/song-eval-baseline.md 汇总表，2026-08-26 记录）
BASELINE = {
    "云烟成雨":   {"sim_mean": 0.8055, "sim_med": 0.8287, "singmos": 4.130, "src": "ext/α0/s1"},
    "亲亲":       {"sim_mean": 0.7493, "sim_med": 0.7938, "singmos": 4.055, "src": "segprompt/α0.2/s1234"},
    "冬眠":       {"sim_mean": 0.7498, "sim_med": 0.7720, "singmos": 3.806, "src": "ext/α0/s1"},
    "去有风的地方": {"sim_mean": 0.7457, "sim_med": 0.7880, "singmos": 3.815, "src": "ext/α0/s1"},
    "情歌":       {"sim_mean": 0.7916, "sim_med": 0.8029, "singmos": 4.349, "src": "ext/α0.2/s1"},
    "最后一页":    {"sim_mean": 0.8034, "sim_med": 0.8232, "singmos": 4.211, "src": "ext/α0/s1"},
    "爱的回归线":  {"sim_mean": 0.7866, "sim_med": 0.8075, "singmos": 3.963, "src": "ext/α0/s1"},
    "爱笑的眼睛":  {"sim_mean": 0.7841, "sim_med": 0.8064, "singmos": 4.120, "src": "ext/α0.2/s1"},
    "白色毛衣":    {"sim_mean": 0.8115, "sim_med": 0.8347, "singmos": 4.236, "src": "ext/α0/s1"},
    "绿色":       {"sim_mean": 0.7756, "sim_med": 0.8370, "singmos": 4.261, "src": "ext/α0/s1"},
    "追光者":     {"sim_mean": 0.7969, "sim_med": 0.8176, "singmos": 4.049, "src": "ext/α0/s1"},
    "那女孩对我说": {"sim_mean": 0.7654, "sim_med": 0.7790, "singmos": 4.284, "src": "ext/α0/s1"},
    "陪你看星星":  {"sim_mean": 0.8042, "sim_med": 0.8132, "singmos": 4.377, "src": "ext/α0/s1"},
    "雨爱":       {"sim_mean": 0.7810, "sim_med": 0.8058, "singmos": 4.213, "src": "ext/α0/s1"},
}


def parse_eval_txt(p: Path) -> dict:
    """从 eval.txt 提取 SIM mean/median + SingMOS 合成 mean。"""
    txt = p.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = re.search(r"overall SIM: mean=([0-9.]+)\s+median=([0-9.]+)", txt)
    if m:
        out["sim_mean"] = float(m.group(1))
        out["sim_med"] = float(m.group(2))
    m = re.search(r"corr\(pitch distance, SIM\) = ([+-]?[0-9.]+)", txt)
    if m:
        out["corr"] = float(m.group(1))
    # SingMOS 表：找 "合成" 行
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("合成") and "mean" in txt and i > 0:
            parts = ln.split()
            # 合成  65   4.130    4.416   3.111   4.823  0.871
            if len(parts) >= 4:
                try:
                    out["singmos"] = float(parts[2])
                except ValueError:
                    pass
    return out


def parse_natmos(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    syn = d.get("files", {}).get("合成", {})
    return {"singmos": syn.get("mean"), "n_win": syn.get("n_windows")}


def main():
    rows = []
    for song_dir in sorted(SONGS_DIR.iterdir()):
        if not song_dir.is_dir():
            continue
        name = song_dir.name
        if name not in BASELINE:
            continue
        eval_txt = song_dir / "eval.txt"
        natmos = song_dir / "natmos_report.json"
        if not eval_txt.exists():
            print(f"[skip] {name}: 无 eval.txt（还没跑完？）")
            continue
        cur = parse_eval_txt(eval_txt)
        if "singmos" not in cur and natmos.exists():
            cur.update(parse_natmos(natmos))
        base = BASELINE[name]
        rows.append((name, base, cur))

    if not rows:
        print("没有可对比的歌曲。")
        return

    # 汇总
    print(f"{'歌曲':<10} {'基线SIM':>8} {'新SIM':>8} {'ΔSIM':>7} | "
          f"{'基线SingMOS':>10} {'新SingMOS':>9} {'Δ':>6} | 基线来源")
    print("-" * 90)
    sim_d = sing_d = 0.0
    n_sim_up = n_sing_up = 0
    for name, base, cur in rows:
        bsim, csim = base["sim_mean"], cur.get("sim_mean")
        bsing, csing = base["singmos"], cur.get("singmos")
        dsim = (csim - bsim) if (bsim is not None and csim is not None) else None
        dsing = (csing - bsing) if (bsing is not None and csing is not None) else None
        if dsim is not None:
            sim_d += dsim
            n_sim_up += dsim > 0
        if dsing is not None:
            sing_d += dsing
            n_sing_up += dsing > 0
        print(f"{name:<10} {bsim:>8.4f} {csim if csim is None else f'{csim:.4f}':>8} "
              f"{dsim if dsim is None else f'{dsim:+.4f}':>7} | "
              f"{bsing:>10.3f} {csing if csing is None else f'{csing:.3f}':>9} "
              f"{dsing if dsing is None else f'{dsing:+.3f}':>6} | {base['src']}")
    n = len(rows)
    print("-" * 90)
    print(f"平均 ΔSIM   : {sim_d / n:+.4f}（{n_sim_up}/{n} 首提升）")
    print(f"平均 ΔSingMOS: {sing_d / n:+.3f}（{n_sing_up}/{n} 首提升）")


if __name__ == "__main__":
    main()
