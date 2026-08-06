#!/usr/bin/env python3
"""吐字可懂度评测：用 ASR 字错率(CER)衡量 SVC 输出的咬字清晰度。

为什么需要它：eval_sim / eval_stab / eval_triad 全部只测说话人相似度。kNN 内容特征替换
(--knn-alpha) 按设计就在改内容特征，用纯音色指标验收它会漏掉吐字退化。

参考文本取自**原唱 lead** 的 ASR 结果（不是人工标注）：原唱是同一段旋律同一段词，其 ASR
输出是"这段音频在理想情况下应该被识别成什么"的最好可得代理。CER 因此测的是"转换后
比转换前掉了多少字"，而不是绝对识别率。

用法（GPT-SoVITS venv）:
  GPT-SoVITS/.venv/bin/python singing/eval_cer.py \
      --src data/songs/X_base/sep2/lead.wav \
      --gen data/songs/X_base/vocal.wav data/songs/X_knn/vocal.wav \
      --labels base knn
"""
import argparse
import os
import re
import sys


def norm(s: str) -> str:
    """归一化：只留中文/字母/数字，去标点空白（CER 按字符计）。"""
    s = re.sub(r"[^\u4e00-\u9fff a-zA-Z0-9]", "", s)
    return re.sub(r"\s+", "", s).lower()


def edit_ops(ref: str, hyp: str):
    """Levenshtein，返回 (S, D, I, dist)。O(len(ref)*len(hyp)) 时间，滚动数组省内存。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m, m
    # dp[j] = (dist, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                cur[j] = (cur[j][0], cur[j][1], cur[j][2], cur[j][3])
            else:
                sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
                dele = (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3])
                ins = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2], cur[j - 1][3] + 1)
                cur[j] = min(sub, dele, ins, key=lambda x: x[0])
        prev = cur
    d, S, D, I = prev[m]
    return S, D, I, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原唱 lead.wav，其 ASR 结果作参考文本")
    ap.add_argument("--gen", required=True, nargs="+", help="待评测输出，可多个")
    ap.add_argument("--labels", nargs="+", help="与 --gen 对应的标签")
    ap.add_argument("--ref-text", help="人工参考文本文件（给了就不用 src 的 ASR）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump", help="把各 ASR 转写写到这个目录，便于人工核对")
    ap.add_argument("--allow-download", action="store_true",
                    help="允许联网下载模型（默认强制离线，只用已缓存的 paraformer-large）")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(g)) for g in args.gen]
    if len(labels) != len(args.gen):
        sys.exit("--labels 数量与 --gen 不匹配")

    # 用本地已缓存的 paraformer-large 绝对路径，而不是 "paraformer-zh" 别名 —— 后者指向
    # SeaCo-Paraformer（热词版，另外 990MB），会触发下载。
    cache = os.path.expanduser("~/.cache/modelscope/models")
    asr_dir = os.path.join(
        cache, "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "snapshots", "master")
    vad_dir = os.path.join(
        cache, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")
    for d, what in ((asr_dir, "paraformer-large ASR"), (vad_dir, "fsmn-vad")):
        if not os.path.exists(os.path.join(d, "model.pt")):
            sys.exit(f"缺少已缓存的 {what}: {d}/model.pt\n"
                     f"（有 .incomplete 说明上次下载中断）")
    if not args.allow_download:
        os.environ["MODELSCOPE_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # FunASR VAD 分段有随机性（同一文件多次结果不同），固定种子保证可复现
    import random
    import numpy as _np
    import torch as _torch
    _torch.manual_seed(0); _torch.cuda.manual_seed_all(0)
    _torch.backends.cudnn.deterministic = True; _torch.backends.cudnn.benchmark = False
    _np.random.seed(0); random.seed(0)

    from funasr import AutoModel

    m = AutoModel(model=asr_dir, vad_model=vad_dir,
                  device=args.device, disable_update=True, disable_pbar=True)

    def asr(path: str) -> str:
        r = m.generate(input=path, batch_size_s=300)
        return norm("".join(seg.get("text", "") for seg in r))

    if args.ref_text:
        ref = norm(open(args.ref_text, encoding="utf-8").read())
        print(f"参考文本: {args.ref_text}（人工）")
    else:
        ref = asr(args.src)
        print(f"参考文本: {args.src} 的 ASR 转写")
    print(f"  {len(ref)} 字: {ref[:80]}{'...' if len(ref) > 80 else ''}\n")

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        open(os.path.join(args.dump, "ref.txt"), "w", encoding="utf-8").write(ref)

    print(f"{'variant':<10}{'CER':>8}{'S':>6}{'D':>6}{'I':>6}{'字数':>7}{'vs ref':>9}")
    base_cer = None
    for lab, g in zip(labels, args.gen):
        hyp = asr(g)
        S, D, I, dist = edit_ops(ref, hyp)
        cer = dist / max(1, len(ref))
        if base_cer is None:
            base_cer = cer
        print(f"{lab:<10}{cer:>8.4f}{S:>6}{D:>6}{I:>6}{len(hyp):>7}"
              f"{cer - base_cer:>+9.4f}")
        if args.dump:
            open(os.path.join(args.dump, f"{lab}.txt"), "w", encoding="utf-8").write(hyp)

    print("\nCER = (S+D+I)/len(ref)，越低越好。D 高 = 吞字，S 高 = 咬字变形成别的字。")


if __name__ == "__main__":
    main()
