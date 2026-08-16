#!/usr/bin/env python3
"""汇总 source-norm 系列实验各臂的 SIM/CER/SingMOS 到一张表。"""
import re
import os
import sys

EXPS = [("E1 LTAS(旧,含电平)", "e1", ["base", "neutral", "target"]),
        ("E1b LTAS(形状+限带)", "e1b", ["shape05", "shape10"]),
        ("E2 共振峰搬移", "e2", ["fshift05", "fshift10"]),
        ("E3b seed-vc 预转换(说话参考)", "e3b", ["neutral3", "target"]),
        ("E7 预转换(prompt_ext参考)", "A", ["e7"]),
        ("E6 迭代自转换", "A", ["e6p2", "e6p3"]),
        ("E8 kNN×预转换", "A", ["e8a0", "e8a0on", "e8a1on"]),
        ("E9 预转换迭代", "A", ["e9"])]


def parse_arm(song, tag, arm):
    et = f"data/songs/{song}/eval_{tag}_{arm}.txt"
    lg = f"data/songs/{song}/log_{tag}_{arm}.log"
    sim = cer = sing = None
    if os.path.exists(et):
        t = open(et, encoding="utf-8").read()
        m = re.search(r"overall SIM: mean=([\d.]+)\s+median=([\d.]+)", t)
        if m:
            sim = (float(m.group(1)), float(m.group(2)))
        m = re.search(r"合成\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
        if m:
            sing = (float(m.group(1)), float(m.group(3)))  # mean, p10
    if os.path.exists(lg):
        t = open(lg, encoding="utf-8").read()
        m = re.search(rf"^{arm}\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", t, re.M)
        if m:
            cer = float(m.group(1))
    return sim, cer, sing


def main():
    songs = sys.argv[1:] or ["白色毛衣", "去有风的地方"]
    for song in songs:
        print(f"\n===== {song} =====")
        print(f"{'exp':26s} {'arm':9s} {'SIM':>7s} {'med':>7s} {'CER':>7s} {'SingMOS':>7s} {'p10':>6s}")
        for label, tag, arms in EXPS:
            for arm in arms:
                sim, cer, sing = parse_arm(song, tag, arm)
                print(f"{label:26s} {arm:9s} "
                      f"{sim[0] if sim else '-':>7.4f} {sim[1] if sim else '-':>7.4f} "
                      f"{cer if cer is not None else '-':>7.4f} "
                      f"{sing[0] if sing else '-':>7.3f} {sing[1] if sing else '-':>6.3f}")


if __name__ == "__main__":
    main()
