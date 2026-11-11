"""音色稳定性对比: 相邻窗口 drift / 掉音色窗口数 / 最差窗口 SIM。

mean SIM 会掩盖"音色跳来跳去"。用 1.5s 窗口逐个算 SIM，看:
  drift    = 相邻窗口 |SIM 变化| 的均值（越低越稳）
  dropouts = SIM < dropout 阈值的窗口数
  worst    = 最差窗口 SIM

用法:
  GPT-SoVITS/.venv/bin/python singing/eval_stab.py --ref <真实语音.wav> \
      --gen a.wav b.wav --labels 旧v1 新v2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/home/rookie/voice/GPT-SoVITS/GPT_SoVITS/eres2net")
sys.path.insert(0, "/home/rookie/voice/singing")
import torchaudio.compliance.kaldi as Kaldi  # noqa: E402
from ERes2NetV2 import ERes2NetV2  # noqa: E402
from emb_cache import cached_emb  # noqa: E402

CKPT = "/home/rookie/voice/GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


def load16k(p: str) -> np.ndarray:
    d, sr = sf.read(p)
    if d.ndim > 1:
        d = d.mean(1)
    if sr != 16000:
        d = torchaudio.functional.resample(torch.from_numpy(d).float(), sr, 16000).numpy()
    return d.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="真实语音（ground truth）")
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--win", type=float, default=1.5)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--no-ref-emb-cache", action="store_true",
                    help="禁用参考嵌入磁盘缓存，每次重算（见 singing/emb_cache.py）")
    args = ap.parse_args()

    labels = args.labels or [f"gen{i}" for i in range(len(args.gen))]
    m = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval().cuda()

    def emb(w: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(w).float().unsqueeze(0).cuda()
        with torch.no_grad():
            f = torch.stack([Kaldi.fbank(x.unsqueeze(0), num_mel_bins=80,
                                         sample_frequency=16000, dither=0) for x in t])
            return m.forward3(f).squeeze(0)

    # 参考嵌入走磁盘缓存：537s 参考整文件前向瞬时需 ~15.5GB 显存，与宿主桌面占用
    # 叠加就会 OOM（见 singing/emb_cache.py）。缓存内容与直接前向一致，不改协议。
    gt = cached_emb(args.ref, "eres2netv2", lambda: emb(load16k(args.ref)),
                    use_cache=not args.no_ref_emb_cache).cuda()
    # 长参考的整文件前向会把 PyTorch 缓存池撑到 ~19GB（537s 素材实测 reserved
    # 19.48GB vs allocated 0.21GB），而后续每个窗只要 0.34GB。不清掉会在 16GB 卡上
    # 随机 OOM（见 eval_sim.py 同处注释）。embedding 已算完，清理不改数值。
    torch.cuda.empty_cache()
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))  # noqa: E731

    print(f"参考: {args.ref}   窗口 {args.win}s   dropout 阈值 {args.dropout}\n")
    print(f"{'variant':<12}{'mean':>8}{'median':>8}{'drift':>8}{'p90drift':>10}"
          f"{'dropouts':>10}{'worst':>8}{'n':>6}")
    for lab, path in zip(labels, args.gen):
        y = load16k(path)
        n = int(args.win * 16000)
        sims = []
        for i in range(0, len(y) - n, n):
            seg = y[i:i + n]
            if float(np.sqrt(np.mean(seg ** 2))) < 0.005:
                continue          # 静音窗口不计
            sims.append(cos(gt, emb(seg)))
        s = np.array(sims)
        d = np.abs(np.diff(s))
        print(f"{lab:<12}{s.mean():8.4f}{np.median(s):8.4f}{d.mean():8.4f}"
              f"{np.percentile(d, 90):10.4f}{int((s < args.dropout).sum()):10d}"
              f"{s.min():8.4f}{len(s):6d}")


if __name__ == "__main__":
    main()
