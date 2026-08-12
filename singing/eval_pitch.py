#!/usr/bin/env python3
"""音高精度评估: FFE / GPE / VDE (Frame-level F0 Error / Gross Pitch Error / Voicing Decision Error)

三个指标 (Chu et al. 2013, MDB-stem-synth):
  FFE: 有声帧的 F0 平均误差（cents, 100 cents = 1 半音）
  GPE: 有声帧里 F0 误差 >50 cents 的比例（粗大错误）
  VDE: UV 决策错误率（该有声判成无声 + 该无声判成有声）

数值越低越好。SoulX-Singer 论文用 pyworld DIO 提 F0，此处沿用。

依赖（应已安装）:
  pyworld librosa numpy soundfile

用法（GPT-SoVITS venv 或 SoulX-Singer venv）:
  GPT-SoVITS/.venv/bin/python singing/eval_pitch.py \
      --ref data/songs/X/sep2/lead.wav --gen data/songs/X/vocal.wav

  # 多个输出对比
  GPT-SoVITS/.venv/bin/python singing/eval_pitch.py \
      --ref data/songs/X/sep2/lead.wav \
      --gen data/songs/X_a/vocal.wav data/songs/X_b/vocal.wav --labels base new
"""
import argparse
import os
import sys

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf


def extract_f0(wav: np.ndarray, sr: int, f0_min: float = 80.0, f0_max: float = 800.0) -> np.ndarray:
    """pyworld DIO + Stonemask refinement, 返回 F0 数组 (Hz, 无声=0)。"""
    hop_ms = 10.0  # SoulX-Singer 论文用 10ms hop
    if len(wav) == 0:
        return np.array([])
    f0, t = pw.dio(wav.astype(np.float64), sr, f0_floor=f0_min, f0_ceil=f0_max,
                   frame_period=hop_ms)
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, sr)
    return f0


def align_length(ref_f0: np.ndarray, gen_f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """截断或零填充到相同长度（取较短者，误差通常 <1%）。"""
    n = min(len(ref_f0), len(gen_f0))
    return ref_f0[:n], gen_f0[:n]


def best_offset(ref_f0: np.ndarray, gen_f0: np.ndarray, max_ms: float = 600.0,
                hop_ms: float = 10.0) -> tuple[int, float]:
    """搜出让 FFE 最小的整帧时移，返回 (offset_frames, ffe)。

    为什么必须做: SoulX 的 metadata 有 `time` 字段（例: [240, 51240] ms），
    cli/inference.py 把生成内容放在该偏移处，而参考 F0 是从 0 开始的段内序列。
    不补偿这 240ms，FFE 会被抬高 2 倍（实测 131 -> 62 cents），并且会让不同
    臂的排序反转 —— 即得出完全错误的结论。
    """
    best = (0, np.inf)
    for off in range(0, int(max_ms / hop_ms) + 1):
        g = gen_f0[off:]
        n = min(len(g), len(ref_f0))
        if n < 100:
            break
        g, r = g[:n], ref_f0[:n]
        b = (r > 0) & (g > 0)
        if b.sum() < 100:
            continue
        ffe = float(np.abs(1200.0 * np.log2(g[b] / (r[b] + 1e-8))).mean())
        if ffe < best[1]:
            best = (off, ffe)
    return best


def compute_metrics(ref_f0: np.ndarray, gen_f0: np.ndarray, gpe_thresh: float = 50.0,
                    offset: int = 0):
    """
    返回 (FFE_cents, GPE_rate, VDE_rate)。
      FFE: 有声帧 F0 平均误差（cents）
      GPE: 有声帧里 F0 误差 > gpe_thresh cents 的比例
      VDE: UV 决策错误率
    """
    if offset:
        gen_f0 = gen_f0[offset:]
    ref_f0, gen_f0 = align_length(ref_f0, gen_f0)
    if len(ref_f0) == 0:
        return np.nan, np.nan, np.nan

    ref_voiced = ref_f0 > 0
    gen_voiced = gen_f0 > 0

    # VDE: (ref有声 预测无声) + (ref无声 预测有声)
    vde_err = np.sum(ref_voiced != gen_voiced)
    vde = vde_err / len(ref_f0) if len(ref_f0) > 0 else np.nan

    # FFE / GPE: 只在双方都有声的帧上计算
    both_voiced = ref_voiced & gen_voiced
    if both_voiced.sum() == 0:
        return np.nan, np.nan, vde

    ref_v = ref_f0[both_voiced]
    gen_v = gen_f0[both_voiced]

    # F0 误差转 cents: 1200 * log2(gen/ref)
    cents_err = 1200.0 * np.log2(gen_v / (ref_v + 1e-8))
    ffe = float(np.abs(cents_err).mean())

    # GPE: 误差 > 50 cents 的比例
    gpe = float((np.abs(cents_err) > gpe_thresh).mean())

    return ffe, gpe, vde


def main():
    ap = argparse.ArgumentParser(description="音高精度评估 (FFE/GPE/VDE)")
    ap.add_argument("--ref", help="参考音频。必须是**干声**（分离人声），"
                                  "带伴奏的混音会把贝斯/伴奏当基频，结果无意义")
    ap.add_argument("--ref-f0", help="直接给参考 F0 .npy（Hz，无声=0）。"
                                     "比 --ref 更可靠：省掉一次提取，也避免误用混音")
    ap.add_argument("--ref-f0-hz", type=float, default=50.0,
                    help="--ref-f0 的帧率（本仓库的 *_f0.npy 都是 50Hz）")
    ap.add_argument("--ref-meta", help="从 SoulX metadata.json 的 f0 字段读参考（50Hz）")
    ap.add_argument("--gen", required=True, nargs="+", help="生成音频，可多个")
    ap.add_argument("--labels", nargs="+", help="与 --gen 对应的标签")
    ap.add_argument("--f0-min", type=float, default=80.0)
    ap.add_argument("--f0-max", type=float, default=800.0)
    ap.add_argument("--gpe-thresh", type=float, default=50.0, help="GPE 阈值（cents）")
    ap.add_argument("--align", action="store_true", default=True,
                    help="自动补偿时移（默认开）。SoulX metadata 的 time 偏移不补会让 FFE 翻倍")
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.add_argument("--align-max-ms", type=float, default=600.0)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(g)) for g in args.gen]
    if len(labels) != len(args.gen):
        sys.exit("--labels 数量与 --gen 不匹配")

    n_src = sum(bool(x) for x in (args.ref, args.ref_f0, args.ref_meta))
    if n_src != 1:
        sys.exit("需要且只需要一个参考来源: --ref / --ref-f0 / --ref-meta")

    # 参考 F0。统一重采样到 100Hz（10ms hop），与 extract_f0 的栅格一致。
    HOP_HZ = 100.0
    if args.ref_f0 or args.ref_meta:
        if args.ref_f0:
            raw = np.load(args.ref_f0).squeeze().astype(float)
            src_hz, src_desc = args.ref_f0_hz, args.ref_f0
        else:
            import json
            with open(args.ref_meta, encoding="utf-8") as fh:
                segs = json.load(fh)
            raw = np.concatenate([
                np.array([float(x) for x in s["f0"].split()]) for s in segs])
            src_hz, src_desc = 50.0, f"{args.ref_meta} (f0 字段)"
        # 最近邻重采样：F0 有 0=无声的语义，线性插值会在边界造出假基频
        n_out = int(round(len(raw) / src_hz * HOP_HZ))
        idx = np.clip((np.arange(n_out) * src_hz / HOP_HZ).astype(int), 0, len(raw) - 1)
        ref_f0 = raw[idx]
        print(f"参考: {src_desc}")
        print(f"      {len(raw)} frames @{src_hz:g}Hz -> {len(ref_f0)} frames @{HOP_HZ:g}Hz")
    else:
        if not os.path.exists(args.ref):
            sys.exit(f"参考文件不存在: {args.ref}")
        ref_wav, ref_sr = sf.read(args.ref)
        if ref_wav.ndim > 1:
            ref_wav = ref_wav.mean(axis=1)
        ref_f0 = extract_f0(ref_wav, ref_sr, args.f0_min, args.f0_max)
        print(f"参考: {args.ref}")

    rv = ref_f0[ref_f0 > 0]
    if rv.size:
        print(f"      voiced {np.mean(ref_f0 > 0):.1%}  median {np.median(rv):.1f}Hz"
              f"  (= MIDI {69 + 12 * np.log2(np.median(rv) / 440):.1f})")
        if np.median(rv) < 160:
            print("      !! 中位基频偏低，像是带伴奏的混音而非干声；FFE 会失去意义")
    print(f"F0 范围: {args.f0_min}-{args.f0_max} Hz")
    print(f"GPE 阈值: {args.gpe_thresh} cents\n")
    print(f"{'标签':<20} {'FFE(cents)':>12} {'GPE(%)':>10} {'VDE(%)':>10} {'偏移(ms)':>10}")
    print("-" * 68)

    for lbl, path in zip(labels, args.gen):
        if not os.path.exists(path):
            print(f"{lbl:<20} {'文件缺失':>12}")
            continue
        try:
            gen_wav, gen_sr = sf.read(path)
            if gen_wav.ndim > 1:
                gen_wav = gen_wav.mean(axis=1)
            gen_f0 = extract_f0(gen_wav, gen_sr, args.f0_min, args.f0_max)

            off = 0
            if args.align:
                off, _ = best_offset(ref_f0, gen_f0, args.align_max_ms)
            ffe, gpe, vde = compute_metrics(ref_f0, gen_f0, args.gpe_thresh, offset=off)
            if np.isnan(ffe):
                print(f"{lbl:<20} {'无可比帧':>12}")
            else:
                print(f"{lbl:<20} {ffe:12.2f} {100*gpe:10.2f} {100*vde:10.2f} {off*10:10d}")
        except Exception as e:
            print(f"{lbl:<20} 失败: {e}")

    print("\n说明:")
    print("  FFE: 有声帧 F0 平均误差（cents），越低越好，通常 <30 为优秀")
    print("  GPE: 有声帧里 F0 误差 >50 cents 的比例（粗大错误率），越低越好")
    print("  VDE: UV 决策错误率（该有声判无声 + 该无声判有声），越低越好")


if __name__ == "__main__":
    main()
