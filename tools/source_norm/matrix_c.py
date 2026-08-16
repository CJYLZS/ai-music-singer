#!/usr/bin/env python3
"""C 组 cfg×α 2×2 矩阵汇总：每首歌 4 格 + 跨歌聚合的 kNN/cfg 效应。"""
import re
import os

SONGS = ["白色毛衣", "去有风的地方", "冬眠", "绿色"]
ARMS = ["c05a02", "c05a00", "c10a02", "c10a00"]


def arm(song, tag):
    et = f"data/songs/{song}/eval_C_{tag}.txt"
    lg = f"data/songs/{song}/log_C_{tag}.log"
    if not os.path.exists(et):
        return None
    t = open(et, encoding="utf-8").read()
    t2 = open(lg, encoding="utf-8").read()
    sim = re.search(r"overall SIM: mean=([\d.]+)\s+median=([\d.]+)", t)
    sing = re.search(r"合成\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
    cer = re.search(rf"^{tag}\s+([\d.]+)", t2, re.M)
    if not sim or not cer or not sing:
        return None
    return (float(sim.group(1)), float(sim.group(2)), float(cer.group(1)),
            float(sing.group(1)), float(sing.group(3)))


def main():
    agg = {a: [] for a in ARMS}
    for song in SONGS:
        print(f"\n===== {song} =====")
        print(f"{'arm':8s} {'cfg':>4s} {'α':>4s} {'SIM':>7s} {'med':>7s} {'CER':>7s} {'SingMOS':>7s} {'p10':>6s}")
        for a in ARMS:
            r = arm(song, a)
            if not r:
                print(f"{a:8s} (缺)")
                continue
            cfg, alpha = a[1:3], a[4:]
            print(f"{a:8s} {cfg:>4s} {alpha:>4s} {r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.3f} {r[4]:6.3f}")
            agg[a].append(r)
    print("\n===== 跨歌聚合（配对均值） =====")
    print(f"{'对比':28s} {'ΔSIM':>8s} {'ΔCER':>8s} {'ΔSingMOS':>9s} {'SIM胜率':>7s}")
    pairs = [
        ("kNN 0.2 vs 0 @ cfg=0.5", "c05a02", "c05a00"),
        ("kNN 0.2 vs 0 @ cfg=1.0", "c10a02", "c10a00"),
        ("cfg 1.0 vs 0.5 @ α=0",   "c10a00", "c05a00"),
        ("cfg 1.0 vs 0.5 @ α=0.2", "c10a02", "c05a02"),
    ]
    for label, a1, a0 in pairs:
        rows = [(x, y) for x, y in zip(agg[a1], agg[a0]) if x and y]
        if not rows:
            continue
        ds = [x[0] - y[0] for x, y in rows]
        dc = [x[2] - y[2] for x, y in rows]
        dm = [x[3] - y[3] for x, y in rows]
        wins = sum(1 for d in ds if d > 0.004)
        print(f"{label:28s} {sum(ds)/len(ds):+8.4f} {sum(dc)/len(dc):+8.4f} "
              f"{sum(dm)/len(dm):+9.3f} {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
