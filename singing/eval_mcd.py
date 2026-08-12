#!/usr/bin/env python3
"""MCD (Mel-Cepstral Distortion): 频谱质量 —— 预测 vs 真实的梅尔倒谱距离。

数值越低越好。SoulX-Singer 论文里用 DTW-MCD（动态时间规整），适配节奏稍有差异的情况。

依赖:
  pip install pymcd     # SoulX-Singer-Eval 用的同一个库

用法（任意 venv 装了 pymcd）:
  python singing/eval_mcd.py --ref data/songs/X/sep2/lead.wav --gen data/songs/X/vocal.wav
  python singing/eval_mcd.py --ref data/songs/X/sep2/lead.wav \
      --gen data/songs/X_a/vocal.wav data/songs/X_b/vocal.wav --labels base new
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(
        description="MCD (Mel-Cepstral Distortion) 评估。\n"
                    "注意: MCD 是**成对**指标，要求 ref 与 gen 是同一段内容的干声。\n"
                    "拿带伴奏的混音当 ref 会得到 14+ 的无意义数值（论文 SOTA 是 3-4）；\n"
                    "零样本换音色场景下 ref 与 gen 的说话人不同，MCD 也不可解释。")
    ap.add_argument("--ref", required=True,
                    help="参考音频，必须是与 gen 同内容的**干声**（分离人声）")
    ap.add_argument("--gen", required=True, nargs="+", help="生成音频，可多个")
    ap.add_argument("--labels", nargs="+", help="与 --gen 对应的标签")
    ap.add_argument("--mode", default="dtw_sl", choices=["plain", "dtw", "dtw_sl"],
                    help="dtw_sl=动态时间规整+静音-浊音对齐（推荐，SoulX 论文用）")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(g)) for g in args.gen]
    if len(labels) != len(args.gen):
        sys.exit("--labels 数量与 --gen 不匹配")

    if not os.path.exists(args.ref):
        sys.exit(f"参考文件不存在: {args.ref}")

    try:
        from pymcd.mcd import Calculate_MCD
    except ImportError:
        sys.exit("缺少 pymcd: pip install pymcd")

    mcd_calc = Calculate_MCD(MCD_mode=args.mode)
    print(f"参考: {args.ref}")
    print(f"模式: {args.mode}  (dtw_sl = DTW + 静音浊音对齐)\n")
    print(f"{'标签':<20} {'MCD':>8}")
    print("-" * 30)

    for lbl, path in zip(labels, args.gen):
        if not os.path.exists(path):
            print(f"{lbl:<20} 文件缺失")
            continue
        try:
            val = mcd_calc.calculate_mcd(args.ref, path)
            print(f"{lbl:<20} {val:8.3f}")
        except Exception as e:
            print(f"{lbl:<20} 失败: {e}")


if __name__ == "__main__":
    main()
