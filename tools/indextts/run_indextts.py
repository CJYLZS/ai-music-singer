#!/usr/bin/env python3
"""IndexTTS-2.5 推理封装（前奏口白用）。

设计要点：
  * **必须固定 seed**：上游 GPT 生成是 do_sample=True / temperature=0.8 / num_beams=3，
    不播种的话同一组参数两次输出不同 —— 本仓库在 CFM、FunASR VAD、Step-Audio 上
    已经三次踩过这个坑，任何未播种的 A/B 数字都是噪声。
  * emo_vector 八维顺序 = [happy, angry, sad, afraid, disgusted, melancholic,
    surprised, calm]。**normalize_emo_vec 只被 webui.py 调用，infer() 不会自动应用**，
    所以这里传进去的就是原始强度；需要 webui 那套 bias+总和≤0.8 时加 --normalize-emo。
  * 向量总和必须 ≤1：融合式是
    `emovec = emovec_mat + (1 - sum(vec)) * emovec_ref`（infer_v2_5.py:768），
    sum>1 会给参考音色的情感分量一个负权重。
  * base 臂（emo_vector=None）是干净基线：emovec 来自参考音频自身。日志里那句
    "Use the specified emotion vector" 是误导 —— emo_vec 参数永远非 None。
  * 单次只加载一次模型，可以一口气跑多个 arm，省掉重复的权重加载。

用法（先 source env.sh）：
    python run_indextts.py --ref REF.wav --text-file intro.txt --out-dir OUT \
        --arm base --arm 'happy0.6:happy=0.6' --arm 'calm:calm=0.5' --seed 1
"""
import argparse
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "repo")
CKPT = os.path.join(ROOT, "checkpoints")

EMO_NAMES = [
    "happy", "angry", "sad", "afraid",
    "disgusted", "melancholic", "surprised", "calm",
]


def parse_arm(spec):
    """'name:happy=0.6,calm=0.2@alpha=0.8' -> (name, emo_vector|None, kwargs)"""
    name, _, rest = spec.partition(":")
    name = name.strip()
    vec = None
    kw = {}
    rest, _, tail = rest.partition("@")
    for part in filter(None, (p.strip() for p in rest.split(","))):
        key, _, val = part.partition("=")
        key = key.strip()
        if key not in EMO_NAMES:
            sys.exit(f"未知情感维度 {key!r}，可选: {EMO_NAMES}")
        if vec is None:
            vec = [0.0] * 8
        vec[EMO_NAMES.index(key)] = float(val)
    for part in filter(None, (p.strip() for p in tail.split(","))):
        key, _, val = part.partition("=")
        kw[key.strip()] = float(val)
    return name, vec, kw


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="音色参考 wav（上游硬裁 15s）")
    ap.add_argument("--text", help="要合成的文本")
    ap.add_argument("--text-file", help="从文件读文本（与 --text 二选一）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", action="append", default=[],
                    help="臂定义 'name[:dim=v,...][@alpha=..,duration_factor=..]'")
    ap.add_argument("--emo-ref", help="情感参考 wav（emo_audio_prompt 模式）")
    ap.add_argument("--emo-text", help="情感软指令文本（需要 QwenEmotion）")
    ap.add_argument("--normalize-emo", action="store_true",
                    help="套用 webui 的 emo_bias + 总和≤0.8 归一化")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bf16", action="store_true", help="GPT 用 bf16，省显存")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if bool(args.text) == bool(args.text_file):
        sys.exit("--text 与 --text-file 必须且只能给一个")
    text = args.text if args.text else open(args.text_file, encoding="utf-8").read().strip()
    ref = os.path.abspath(args.ref)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    arms = [parse_arm(s) for s in (args.arm or ["base"])]

    sys.path.insert(0, REPO)
    os.chdir(REPO)  # 上游有相对路径依赖
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    need_qwen = bool(args.emo_text)
    t0 = time.perf_counter()
    tts = IndexTTS2(
        cfg_path=os.path.join(CKPT, "config.yaml"),
        model_dir=CKPT,
        use_bf16=args.bf16,
        use_cuda_kernel=False,
        use_qwen_emo=need_qwen,
    )
    print(f">> 模型加载耗时 {time.perf_counter() - t0:.1f}s, "
          f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)

    for name, vec, kw in arms:
        out = os.path.join(out_dir, f"{name}.wav")
        if vec is not None:
            if args.normalize_emo:
                vec = tts.normalize_emo_vec(vec, apply_bias=True)
            elif sum(vec) > 1.0:
                sys.exit(f"[{name}] emo_vector 总和 {sum(vec):.3f} > 1，"
                         f"会让参考情感分量取负权重；请降低或加 --normalize-emo")
        seed_all(args.seed)  # 每个臂前重播种，臂之间才是配对的
        alpha = kw.pop("alpha", 1.0)
        dur = kw.pop("duration_factor", 1.0)
        t1 = time.perf_counter()
        tts.infer(
            spk_audio_prompt=ref, text=text, output_path=out, lang=args.lang,
            emo_audio_prompt=args.emo_ref, emo_alpha=alpha, emo_vector=vec,
            use_emo_text=bool(args.emo_text), emo_text=args.emo_text,
            use_random=False, duration_factor=dur, verbose=args.verbose, **kw,
        )
        print(f">> [{name}] vec={vec} alpha={alpha} dur={dur} "
              f"{time.perf_counter() - t1:.1f}s peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
              f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
