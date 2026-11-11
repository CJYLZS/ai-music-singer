#!/usr/bin/env python
"""长音频说话人嵌入的磁盘缓存。

**为什么需要**：537s 参考在 16GB 卡上做整文件前向（`SV.emb(load16k(ref))`）的
瞬时显存需求 ~15.5GB，加上 CUDA context 已经贴住 16GB 上限。宿主桌面占用一波动
就 OOM，表现是「SVC 全部成功、只有 eval/leak-detect 挂掉」，而且换歌重试结果不稳定
（12 首里 6 首日志里有 `memory allocation failed` 重试痕迹，多数侥幸成功）。
allocator 的报错文本会指向 `expandable_segments`，但这里不是碎片问题：live 张量
本身就有 11.4GB，再要一块 4.1GB，总计超过卡容量，换分配策略也救不回来。

**做法**：把嵌入按 (wav 绝对路径, 大小, mtime, 模型 tag) 缓存到
`data/emb_cache/<stem>_<pathhash>_<tag>_<size>_<mtime>.npy`。缓存可以**在 CPU 上生成**
（ERes2NetV2 537s CPU 前向实测 ~2.2 分钟，峰值 host RAM ~7GB），数值与 GPU 一致：
40s 输入实测 cos(CPU,GPU) = 1.0000005，max|diff| 2e-3 / 范数 88（相对 2e-5）。
所以它不改协议 —— 锚点仍是**完整 537s 真实语音**算出来的同一个向量，
只是不再每次重算、不再吃 16GB 显存。

**不要用「截断参考」替代**：那会换掉锚点，SIM 绝对值与历史 12 首不可比
（本工作区已记录过一次换锚点导致 0.4979 → 0.7600 的假暴涨）。

用法：
    from emb_cache import cached_emb
    ref_emb = cached_emb(ref_path, "eres2netv2", lambda: sv.emb(load16k(ref_path)),
                         log=print).to(sv.device)
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "emb_cache")


def cache_path(wav_path: str, tag: str, cache_dir: str | None = None,
               extra: str = "") -> str:
    """缓存文件名：内容变化（size/mtime）自动失效，不同文件不会撞名。"""
    p = os.path.abspath(wav_path)
    st = os.stat(p)
    h = hashlib.md5((p + extra).encode()).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(p))[0]
    return os.path.join(cache_dir or CACHE_DIR,
                        f"{stem}_{h}_{tag}_{st.st_size}_{int(st.st_mtime)}.npy")


def cached_emb(wav_path: str, tag: str, build_fn, use_cache: bool = True,
               cache_dir: str | None = None, cache_file: str | None = None,
               extra: str = "", log=print):
    """返回 CPU 上的嵌入张量（调用方自己 .to(device)）。

    build_fn:  无参可调用，返回 torch.Tensor（由各脚本自己的模型/设备决定）。
    cache_file: 显式缓存路径，覆盖默认命名（用于预生成/跨机搬运）。
    """
    if not use_cache:
        return build_fn()
    cp = cache_file or cache_path(wav_path, tag, cache_dir, extra)
    if os.path.exists(cp):
        try:
            e = torch.from_numpy(np.load(cp))
            log(f"[emb-cache] 命中 {os.path.basename(cp)}")
            return e
        except Exception as ex:  # 半截文件/版本不符：重算，不要让缓存把流程卡死
            log(f"[emb-cache] 读取失败({ex})，重算 {os.path.basename(cp)}")
    t = build_fn()
    os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
    np.save(cp, t.detach().cpu().numpy())
    log(f"[emb-cache] 写入 {os.path.basename(cp)}")
    return t
