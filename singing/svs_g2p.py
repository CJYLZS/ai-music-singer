#!/usr/bin/env python3
"""g2p 薄封装：歌词字 -> SoulX phone_set token（`zh_tui1` 这一级）。

单独一个文件的原因：`preprocess.tools.g2p` 在 import 时就构造 G2pM 和
g2p_en（约 2s + 内存），而 `svs_edit.py` 的 export/verify 路径不需要它。
延迟到真正要写 metadata 时再付这个代价。

phoneme 是**音节级不是音素级**（`zh_tui1` 整个是一个 token），所以歌词改字
只要重跑 g2p，不需要 MFA 强制对齐 —— 时间信息全部来自音符时长。
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOULX = ROOT / "SoulX-Singer"


@lru_cache(maxsize=1)
def _phone_set() -> set[str]:
    p = SOULX / "soulxsinger/utils/phoneme/phone_set.json"
    with open(p, encoding="utf-8") as f:
        return set(json.load(f))


def _known(token: str, ps: set[str]) -> bool:
    """token 是否可查。英文的 `en_P-R-IH1-T-IY0` 是**复合**写法。

    `DataProcessor.preprocess`（`data_processor.py:96-99`）对 `en_` 前缀的
    token 会先 `phonemes[3:].split('-')` 拆成 `en_P` / `en_R` / ... 再逐个
    查表，所以整串本身不在 phone_set 里是正常的 —— 上游 `en_target.json`
    就是这么写的。按拆分后的音素校验。
    """
    if token in ps:
        return True
    if token.startswith("en_"):
        return all(("en_" + p) in ps for p in token[3:].split("-") if p)
    return False


def g2p_words(words: list[str], language: str = "Mandarin") -> list[str]:
    """逐音符转 phoneme token，并校验落在 phone_set 内。

    未登录 token 会在 `DataProcessor.load_phoneme_id_map` 的 `phone2idx[x]`
    处抛 KeyError（推理开跑之后才炸），所以在这里就拦下来 —— 校准阶段用户
    可能填进生僻字或英文，早报错比推理到一半失败好。
    """
    if str(SOULX) not in sys.path:
        sys.path.insert(0, str(SOULX))
    from preprocess.tools.g2p import g2p_transform

    out = list(g2p_transform(list(words), language))
    if len(out) != len(words):
        raise ValueError(f"g2p 返回 {len(out)} != 输入 {len(words)}")
    ps = _phone_set()
    bad = sorted({(w, o) for w, o in zip(words, out) if not _known(o, ps)})
    if bad:
        raise ValueError(
            "以下歌词转出的 phoneme 不在 phone_set 里（推理会 KeyError）:\n  " +
            "\n  ".join(f"「{w}」 -> {o}" for w, o in bad))
    return out
