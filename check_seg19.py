#!/usr/bin/env python3
"""验证 seg-max 19 下 14 首歌的切分可行性与扩展窗口长度（纯 CPU，不跑 SVC）。
模拟 run_segprompt 的扩展 + shrink 逻辑：
  ext_t0 = max(a - 2, 0); ext_t1 = min(b + 2, total)
  若 ext_len >= svc_max(22): ext_t1 -= (ext_len - (22 - 1e-3))  # 只裁尾部
  输出每段: [a,b], ext 窗口, shrink 后 ext 窗口, 前缀/后缀冗余秒数
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "singing"))
from w_segprompt import split_optimal  # noqa: E402

ROOT = Path(__file__).resolve().parent
F0_RATE = 50.0
PAD_EXT = 2.0
SVC_MAX = 22.0
SONGS = ["云烟成雨", "亲亲", "冬眠", "去有风的地方", "情歌", "最后一页",
         "爱的回归线", "爱笑的眼睛", "白色毛衣", "绿色", "追光者",
         "那女孩对我说", "陪你看星星", "雨爱"]

all_ok = True
for name in SONGS:
    f0_path = ROOT / "data" / "songs" / name / "target_f0.npy"
    if not f0_path.exists():
        print(f"[{name}] MISSING target_f0.npy"); all_ok = False; continue
    f0 = np.load(f0_path)
    total = len(f0) / F0_RATE
    try:
        segs = split_optimal(f0, min_run=0.2, pad=1.0, seg_min=12.0, seg_max=19.0)
    except ValueError as e:
        print(f"[{name}] 切分失败: {e}"); all_ok = False; continue
    problems = []
    min_suf = 99.0
    for i, (a, b) in enumerate(segs):
        ext_t0 = max(a - PAD_EXT, 0.0)
        ext_t1 = min(b + PAD_EXT, total)
        shrunk = False
        if ext_t1 - ext_t0 >= SVC_MAX:
            shrink = (ext_t1 - ext_t0) - (SVC_MAX - 1e-3)
            ext_t1 = max(ext_t0 + 1e-3, ext_t1 - shrink)
            shrunk = True
        pref = a - ext_t0
        suf = ext_t1 - b
        min_suf = min(min_suf, suf)
        if not (ext_t1 - ext_t0 < SVC_MAX + 1e-9):
            problems.append(f"段{i} 扩展后 {ext_t1-ext_t0:.2f}s >= 22")
        if a > 1e-6 and pref < 1.99:   # 首段 (a≈0) 前方无音频，前缀短是正常的
            problems.append(f"段{i} 前缀 {pref:.2f}s < 2")
        if suf < 0.99:
            problems.append(f"段{i} 后缀 {suf:.2f}s < 1")
        if b - a > 19.0 + 1e-6:
            problems.append(f"段{i} 内容 {b-a:.2f}s > 19")
    status = "OK" if not problems else "PROBLEM"
    if problems:
        all_ok = False
    print(f"[{name}] {status} 段数={len(segs)} min后缀={min_suf:.2f}s"
          + (f" 问题={problems}" if problems else ""))

print("ALL OK" if all_ok else "HAS PROBLEMS")
