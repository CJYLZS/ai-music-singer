#!/usr/bin/env python
"""第二种说话人嵌入（CAMPPlus, funasr campplus_cn_common.bin）重跑小窗口漏音扫描。

与 eval_leak_windows.py 完全同协议（窗口 embedding vs 全局锚点、余弦、excess/margin），
只是嵌入模型换成 seed-vc 自带的 CAMPPlus（192 维，80 维 fbank 输入）。
用途：两个独立说话人嵌入模型对同一批窗口给出一致结论的窗口才是可信的漏音；
只有单个模型报尖峰的窗口可能是嵌入噪声/模型偏置。

用法:
  seed-vc/.venv/bin/python singing/eval_leak_campplus.py --song-dir data/songs/雨爱 \
      --impostor data/songs/冬眠/sep2/lead.wav data/songs/绿色/sep2/lead.wav \
      data/songs/爱笑的眼睛/sep2/lead.wav data/songs/陪你看星星/sep2/lead.wav \
      data/songs/最后一页/sep2/lead.wav [--win 1.0] [--csv-out leak_1s_campplus.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchaudio

from eval_triad import load16k, windows, cos

_SV_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "seed-vc")
sys.path.insert(0, _SV_ROOT)  # 让 `modules` 包可导入
sys.path.insert(0, os.path.join(_SV_ROOT, "modules"))
from campplus.DTDNN import CAMPPlus  # noqa: E402

CAMPPLUS_CKPT = os.path.expanduser(
    "~/voice/seed-vc/checkpoints/models--funasr--campplus/snapshots/"
    "e4b6ede7ce16997aff4ae69fbca1f0175e2afede/campplus_cn_common.bin"
)


class CP:
    """CAMPPlus 封装：任意长度波形 → L2 归一化 192 维嵌入。"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        m = CAMPPlus(feat_dim=80, embedding_size=192)
        sd = torch.load(CAMPPLUS_CKPT, map_location="cpu")
        m.load_state_dict(sd)
        m.eval().to(device)
        self.m = m

    @torch.no_grad()
    def emb(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # (1,N) 16k
        f = torchaudio.compliance.kaldi.fbank(
            x, num_mel_bins=80, dither=0, sample_frequency=16000
        )  # (T,80)
        f = f - f.mean(dim=0, keepdim=True)
        e = self.m(f.unsqueeze(0).to(self.device))[0]  # (192,)
        return torch.nn.functional.normalize(e, dim=0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", help="data/songs/<name>/，自动推导 ref/target/out")
    ap.add_argument("--ref", help="参考音色真实语音（默认 <song-dir>/refsep2/lead.wav）")
    ap.add_argument("--target", help="原曲分离人声（默认 <song-dir>/sep2/lead.wav）")
    ap.add_argument("--out", default=None, help="转换输出（默认 <song-dir>/vocal.wav）")
    ap.add_argument("--impostor", nargs="*", default=None, metavar="LEAD")
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--f0", help="输出 F0 .npy（50Hz），给出则打印每窗 F0 中位数")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    d = args.song_dir.rstrip("/") if args.song_dir else None
    ref = args.ref or (os.path.join(d, "refsep2", "lead.wav") if d else None)
    target = args.target or (os.path.join(d, "sep2", "lead.wav") if d else None)
    out = args.out or (os.path.join(d, "vocal.wav") if d else None)
    if not (ref and target and out):
        ap.error("需要 --song-dir，或同时给出 --ref/--target/--out")
    for p in [ref, target, out, *(args.impostor or [])]:
        if not os.path.exists(p):
            ap.error(f"文件不存在: {p}")

    cp = CP()
    ref_w, tgt_w, out_w = load16k(ref), load16k(target), load16k(out)
    a_ref, a_tgt = cp.emb(ref_w), cp.emb(tgt_w)
    imps = [cp.emb(load16k(p)) for p in (args.impostor or [])]
    torch.cuda.empty_cache()

    anchors = {"ref": a_ref, "tgt": a_tgt}
    anchors.update({f"imp{i}": e for i, e in enumerate(imps)})

    times: list[float] = []
    out_sc = {k: [] for k in anchors}
    for t0, seg in windows(out_w, args.win):
        e = cp.emb(seg)
        times.append(t0)
        for k, a in anchors.items():
            out_sc[k].append(cos(e, a))
    A_ = np.array(out_sc["ref"])
    C_ = np.array(out_sc["tgt"])
    C_imp = np.mean([np.array(out_sc[f"imp{i}"]) for i in range(len(imps))], axis=0) if imps else np.zeros_like(A_)
    excess = C_ - C_imp
    margin = A_ - C_
    times = np.array(times)

    f0 = np.load(args.f0) if args.f0 else None
    med_f0 = np.full(len(times), np.nan)
    if f0 is not None:
        for i, t0 in enumerate(times):
            fa, fb = int(t0 * 50), int((t0 + args.win) * 50)
            v = f0[fa:fb]
            v = v[v > 0]
            if len(v) >= 10:
                med_f0[i] = float(np.median(v))

    print(f"窗长 {args.win}s，n={len(A_)}，impostor={len(imps)}，模型 CAMPPlus(192d)")
    print(f"{'':<16}{'n':>5}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}")
    for name, arr in [("A", A_), ("C", C_), ("C_imp", C_imp), ("excess", excess), ("margin", margin)]:
        print(f"{name:<16}{len(arr):>5}{arr.mean():>9.4f}{np.median(arr):>9.4f}"
              f"{np.percentile(arr, 90):>9.4f}{arr.max():>9.4f}")
    print(f"\nmargin<0 窗口: {(margin < 0).sum()}/{len(margin)}")
    print(f"excess 分桶: <0: {(excess < 0).sum()}  0-0.05: {((excess >= 0) & (excess < 0.05)).sum()}"
          f"  0.05-0.1: {((excess >= 0.05) & (excess < 0.1)).sum()}  >0.1: {(excess >= 0.1).sum()}\n")

    print("=== 最差窗口 TOP-12（按 excess 降序）===")
    order = np.argsort(-excess)[:12]
    print(f"{'t0':>8}{'A':>9}{'C':>9}{'Cimp':>9}{'excess':>9}{'margin':>9}{'F0med':>8}")
    for i in order:
        f0s = f"{med_f0[i]:.0f}" if not np.isnan(med_f0[i]) else "-"
        print(f"{times[i]:>8.1f}{A_[i]:>9.4f}{C_[i]:>9.4f}{C_imp[i]:>9.4f}"
              f"{excess[i]:>+9.4f}{margin[i]:>+9.4f}{f0s:>8}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0", "A", "C", "C_imp", "excess", "margin", "f0_med"])
            for i in range(len(times)):
                w.writerow([f"{times[i]:.2f}", f"{A_[i]:.4f}", f"{C_[i]:.4f}",
                            f"{C_imp[i]:.4f}", f"{excess[i]:.4f}", f"{margin[i]:.4f}",
                            f"{med_f0[i]:.1f}" if not np.isnan(med_f0[i]) else ""])
        print(f"\n全部窗口已写入 {args.csv_out}")


if __name__ == "__main__":
    main()
