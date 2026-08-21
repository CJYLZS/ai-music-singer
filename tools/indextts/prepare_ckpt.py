#!/usr/bin/env python3
"""把 IndexTTS-2.5 权重组装成一个可写的 checkpoints/ 目录，并预下载辅助权重。

主权重已在 .ms-cache/models/IndexTeam--IndexTTS-2.5/snapshots/master（5.2GB，只读语义），
但 IndexTTS2.__init__ 需要往 model_dir/hf_cache/ 里写 4 份辅助权重：
  w2v-bert-2.0（完整 repo）/ MaskGCT semantic codec / campplus / BigVGAN
所以 checkpoints/ 里主权重用软链，hf_cache/ 是真目录。

用法（先 source env.sh）：
    python prepare_ckpt.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(
    ROOT, ".ms-cache", "models", "IndexTeam--IndexTTS-2.5", "snapshots", "master"
)
DST = os.path.join(ROOT, "checkpoints")

# 只链推理需要的条目，跳过 README/LICENSE/.gitattributes 等
WANT = [
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
    "qwen0.6bemo4-merge",
    "multilingual_zh_ja_yue_char_del.tiktoken",
]


def link_main():
    if not os.path.isdir(SRC):
        sys.exit(f"主权重目录不存在: {SRC}")
    os.makedirs(DST, exist_ok=True)
    for name in WANT:
        src = os.path.join(SRC, name)
        dst = os.path.join(DST, name)
        if not os.path.exists(src):
            sys.exit(f"缺少权重文件: {src}")
        if os.path.islink(dst):
            if os.path.realpath(dst) == os.path.realpath(src):
                continue
            os.unlink(dst)
        elif os.path.exists(dst):
            print(f"   跳过（已是真实文件）: {name}")
            continue
        os.symlink(src, dst)
        print(f"   link {name}")
    print(f">> 主权重就绪: {DST}")


def fetch_aux():
    sys.path.insert(0, os.path.join(ROOT, "repo"))
    from indextts.utils.model_download import ensure_models_available

    paths = ensure_models_available(DST)
    for k, v in paths.items():
        ok = os.path.isdir(v) or os.path.isfile(v)
        print(f"   {'OK ' if ok else 'MISS'} {k}: {v}")
    return paths


if __name__ == "__main__":
    link_main()
    print(">> 拉取辅助权重（w2v-bert / semantic codec / campplus / bigvgan）...")
    fetch_aux()
    print(">> done")
