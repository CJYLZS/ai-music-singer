"""RMVPE F0 提取（SoulX-Singer venv 运行）。

输出 50Hz 网格（24000Hz / hop 480）的 f0 数组，0 表示无浊音，
与 SoulX SVC 与 eval_sim.py 的 F0_RATE=50 对齐。
模型只加载一次，可一次处理多个文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "SoulX-Singer"
sys.path.insert(0, str(SOULX))

RMVPE = SOULX / "pretrained_models" / "SoulX-Singer-Preprocess" / "rmvpe" / "rmvpe.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", required=True, help="输入 wav（可多个）")
    ap.add_argument("--out", action="append", required=True, help="输出 .npy（与 --wav 一一对应）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-duration", type=float, default=300.0, help="最长分析时长(秒)")
    args = ap.parse_args()

    if len(args.wav) != len(args.out):
        sys.exit("--wav 与 --out 数量不一致")
    if not RMVPE.exists():
        sys.exit(f"缺少 RMVPE 模型: {RMVPE}")

    from preprocess.tools.f0_extraction import F0Extractor

    extractor = F0Extractor(
        str(RMVPE),
        device=args.device,
        is_half=False,
        max_duration=args.max_duration,
        verbose=False,
    )
    for wav, out in zip(args.wav, args.out):
        f0 = extractor.process(wav, f0_path=out)
        voiced = float((f0 > 0).mean()) if len(f0) else 0.0
        med = float(__import__("numpy").median(f0[f0 > 0])) if (f0 > 0).any() else 0.0
        print(f"{out}  frames={len(f0)}  voiced={voiced:.2f}  med={med:.1f}Hz")


if __name__ == "__main__":
    main()
