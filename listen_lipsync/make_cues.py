#!/usr/bin/env python3
"""中文唱歌口型时间轴生成：强制对齐(字级起点) + pypinyin(韵母→口型)。

为什么不用 Rhubarb：实测其 phonetic recognizer 在中文唱歌上元音区分力 AUC=0.555
（≈随机，语音是 0.679），口型分布与歌词要求的 L1 距离 0.688 —— 歌词要 20% 大张口(D)
它只给 3.2%。它的长静音检测是好的（>1s 段 100% 命中），但元音形状不可用。

本方案：
  1. FunASR fa-zh 强制对齐（iic/speech_timestamp_prediction-v1-16k-offline, 167MB）
     用「音频 + 歌词文本」得到每个字的起点。实测 64 字对齐 64 个时间戳，
     0/64 落在静音里，无重叠。
  2. pypinyin 取韵母 → 口型（元音主导张口形状，这是 2D 动画既有做法）。

关键修正（实测发现）：该对齐模型 47/64 个字的时长恰好被截在 240ms —— 它为**语音**
设计，不建模唱歌拖腔。所以**只用起点，终点由下一个字的起点决定**，否则长音期间口型
会提前回到默认，看起来像"唱一半就闭嘴"。

用法（GPT-SoVITS venv）:
  python make_cues.py --wav clip.wav --text lyrics.txt --out cues.json
  # 不给 --text 则先用 ASR 转写当歌词（口型准确度略降，但无需人工准备歌词）
"""
import argparse
import json
import os
import random
import re
import sys

# 韵母 → 口型。**当前只启用 A / B / C / E 四种**。
#
# 为什么收缩到四种：D(大开)、F(撅唇)、G(唇齿)、H(长L) 这几张贴图在本立绘上很难用对，
# 尤其 D 一出现就明显违和。四型方案的依据是动画业界的常规做法 ——
# "An extreme is anime — whose limited animation style typically achieves lip sync
#  with two mouth shapes — open and shut (millions of anime fans never complained)"
# （Bristow, "5 things that can go wrong with animated lip sync"）。
# 也就是说，口型数量少不是缺陷；用错口型才是。四型已经能覆盖开合与圆展两个维度。
#
# 四型的官方语义（Rhubarb README.adoc 原文，Preston Blair 名在括号里）：
#   A (MBP)  Closed mouth for "P","B","M". 双唇闭合
#   B (etc)  Slightly open mouth with clenched teeth. 多数辅音 + "EE" as in bee
#   C (E)    Open mouth. "EH" as in men / "AE" as in bat；也是 →D 的中间形，
#            所以用它承接原本给 D 的 a 类元音，语义上是连续的
#   E (O)    Slightly rounded mouth. "AO" as in off / "ER" as in bird；
#            官方强调「开口不得大于 C」，且是 →F 的中间形，
#            所以用它承接原本给 F 的 u/ü 撅唇类，同样是沿官方过渡链回退一步
#
# 归并规则（每条都沿官方过渡链，不跨族）：
#   D → C   （官方 C 是 B→D 的中间形）
#   F → E   （官方 E 是 C/D→F 的中间形）
#   G → B   （唇齿收进微开）
#   H → C   （官方 H 开口介于 C 与 D 之间，取 C）
#   X       仍用于静音（闭嘴），与 A 共用同一张闭嘴图
#
# 只取**韵腹**（主元音）：口型由开口最大的那个元音主导，韵尾 -n/-ng 不改变唇形，
# 介音 i-/u-/ü- 只在起始瞬间存在。
VOWEL_SHAPE = {
    # ---- C: 开口档。承接 a 系全部（含原本给 D 的纯 a）与 e 系 ----
    "a": "C", "ia": "C", "ua": "C",           # 原 D，归并到 C
    "ai": "C", "uai": "C", "ao": "C", "iao": "C",
    "an": "C", "uan": "C", "ang": "C", "iang": "C", "uang": "C",
    "ian": "C", "van": "C",
    "e": "C", "ie": "C", "ei": "C", "en": "C", "eng": "C",
    "ve": "C",
    # ---- B: 官方 "EE as in bee"，i 类高前元音 ----
    "i": "B", "in": "B", "ing": "B",
    # ---- E: 微圆唇。o 系 + 原本给 F 的 u/ü 撅唇类 ----
    "er": "E",
    "o": "E", "io": "E", "ou": "E", "iou": "E", "iu": "E",
    "uo": "E", "ong": "E", "iong": "E",
    "ui": "E", "uei": "E",
    "un": "E", "uen": "E", "ueng": "E",
    "u": "E",                                  # 原 F，归并到 E
    "v": "E", "vn": "E",                       # ü/ün 原 F，归并到 E
}

# 声母 → 字首过渡口型。b/p/m 必须先闭唇（这是唯一不能省的辅音口型，
# 官方与多篇动画教程都强调 M/B/P 要在发声前 1-2 帧闭合）。
# f 原本用 G(唇齿)，四型方案下归并到 B(微开)。
INITIAL_SHAPE = {"b": "A", "p": "A", "m": "A", "f": "B"}

# 拖腔衰减：沿官方过渡链回退一步，四型内部闭合。
DECAY_SHAPE = {
    "C": "C",   # 开口档是中性，保持（早先 C→B 曾让 B 占到 57%）
    "B": "B",   # 已是最小展唇
    "E": "E",   # 四型下 E 已是最圆最小，不再往 F 走
    "A": "A", "X": "X",
}
# 短音（VOCAL 的 Ma）降档表。四型下 C 已是最大开口，且它同时承载 a 与 e 两类，
# 再压会丢失元音辨识度，所以留空 —— 开口过大的问题已由 D→C 的归并解决。
SHORT_SHAPE = {}

# 允许输出的口型集合。渲染端缺图会按 VIS.fallback 退化，但这里直接不产出，
# 保证 cues 文件本身就只含这四型（外加静音 X）。
ENABLED_SHAPES = set("ABCEX")


def load_pinyin():
    from pypinyin import Style, pinyin
    return pinyin, Style


def shape_of(ch, pinyin, Style):
    # strict=True 才能拿到《汉语拼音方案》的真实韵母：strict=False 会把 yu/ju/qu/xu 的
    # ü 按书写形式给成 "u"，于是「与/居/去/需」全部错判成强圆唇的 u 口型（F）而不是 ü（H）。
    # 同理 strict=True 下 yi→i、wu→u、yuan→üan，介音归属也正确。
    try:
        fin = pinyin(ch, style=Style.FINALS, strict=True, errors="ignore")
        ini = pinyin(ch, style=Style.INITIALS, strict=True, errors="ignore")
    except Exception:
        return "C", ""
    f = (fin[0][0] if fin and fin[0] else "").replace("ü", "v").replace("ū", "v")
    i = (ini[0][0] if ini and ini[0] else "").lower()
    s = VOWEL_SHAPE.get(f) or VOWEL_SHAPE.get(f[-2:]) or VOWEL_SHAPE.get(f[-1:]) or "C"
    return s, i


def norm_text(s):
    """只保留汉字（对齐模型按字建模，标点/空白会打乱字数对应）。

    非汉字字符不能删除，要替换成占位汉字（默认「嗯」）—— FunASR 强制对齐把
    文本逐字贴到音频上，英文/数字在音频里真实占有时长，删掉它们模型找不到
    对应的声音，只能把后面的字整体右移（实测雨爱 3 处 rainielove 造成
    96.74s 起全部行错位 +4.5~+7.5s）。

    关键：英文按「词」替换，不是按字符。paraformer 的 tokenizer 把英文按词
    （子词/BPE）合并成一个 token，一个词只占一个对齐单元；按字符替换成
    9 个「嗯」= 9 个 token 塞进一个词的位置，反而把后续汉字再推挤 +1.7s
    （实测：按词 1 恩 窗 +0.12s vs 按字符 9 恩 +1.78s）。替换数量 = 词数 ×
    2（留一点余量给多音节英文词），连续字母串如 rainielove 视为一个词。

    占位字会参与口型生成（「嗯」→ en → C 开口），恰好匹配英文元音的开口
    形状；且占位信息写进 chars 的 placeholder 标记，可被下游剔除或替换。
    """
    out = []
    word_buf = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            if word_buf:
                out.append("嗯嗯")  # 一个英文词占 2 个 token（多音节余量，实测 1-2 最优）
                word_buf = []
            out.append(ch)
        elif ch.isalnum():
            word_buf.append(ch)  # 连续字母/数字 = 一个词，只占一个 token
        # 标点/空白：结束当前词缓冲，不占时长
        else:
            if word_buf:
                out.append("嗯嗯")
                word_buf = []
    if word_buf:
        out.append("嗯嗯")
    return "".join(out)


def load_lrc_ts(path, wav_path):
    """从增强 LRC 读时间戳 → (lyr, ts)。

    行起点 = LRC 时间戳（KRC 人工校对，实测 43/43 落在有声处，<0.2s 偏差）。
    字级时间 = 行内均摊到「行内最后有声时刻」——口型只需要「这行大约何时
    唱」与大致节奏，发音不需要字级精准。

    为什么不用 LRC 的逐字时间戳：个别行的行尾逐字时间不可信（雨爱
    「彩虹的美丽」把「美」「丽」标进 21s 间奏的静音区），而行的能量截止
    天然免疫这类坏数据。行内均匀分布后，拖腔长音由下游 is_voiced/
    first_long_silence 逻辑负责 hold/闭嘴。

    返回 (lyr 字符串, ts 列表[(start_ms, end_ms)])，两者索引一一对应。
    """
    import soundfile as sf
    import numpy as np
    from get_lyrics import read_enhanced
    lines, hdr = read_enhanced(path)
    y, sr = sf.read(wav_path)
    hop = sr // 200
    rms = np.sqrt(np.mean((y[:len(y)//hop*hop].reshape(-1, hop))**2, axis=1))
    gate = rms.max() * 10 ** (-40.0 / 20.0)

    def first_long_silence(t_floor, t_end, min_sil=0.35):
        """[t_floor, t_end) 内第一段「持续」静音 ≥min_sil 的起点；没有则 t_end。

        行的能量截止点。不能用 last_voiced（找最后有声）——它会误把
        **下一行的起音**算进本行（实测「彩虹的美丽」行的下一行「冷」
        气口在 124.4s，于是 5 个字均摊到 27.7s，间奏里张嘴 25s）。
        「第一段长静音」从源头断开：行尾拖腔最多夹带 min_sil 的静音，
        超出即视为本行结束。判据必须持续 ≥min_sil，单帧静音不算。"""
        a, b = int(round(t_floor * 200)), min(int(round(t_end * 200)), len(rms))
        if b <= a:
            return t_end
        v = rms[a:b] > gate
        need = max(1, int(round(min_sil * 200)))
        run = 0
        for i, x in enumerate(v):
            if not x:
                run += 1
                if run >= need:
                    return (a + i + 1 - run) / 200.0
            else:
                run = 0
        return t_end

    lyr, ts = [], []
    for li, ln in enumerate(lines):
        nxt = lines[li + 1]["start"] if li + 1 < len(lines) else \
              (ln.get("end") or len(y) / sr)
        if ln.get("chars"):
            units = [ch for _t, _d, ch in ln["chars"]
                     if ch and not ch.isspace()]
        else:
            units = list(norm_text(ln["text"] or ""))
        if not units:
            continue
        line_end = first_long_silence(ln["start"], nxt)
        span = max(0.1, line_end - ln["start"])
        m = len(units)
        for i, ch in enumerate(units):
            st = ln["start"] + span * i / m
            en = ln["start"] + span * (i + 1) / m
            lyr.append(ch)
            ts.append((int(round(st * 1000)), int(round(en * 1000))))
    return "".join(lyr), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16k 单声道 wav（唱歌干声）")
    ap.add_argument("--lrc", help="增强 LRC（行级或逐字时间戳，get_lyrics.py 缓存格式）。"
                                  "与 --text 二选一，LRC 优先；用它时完全不做 ASR")
    ap.add_argument("--text", help="歌词文件；不给则用 ASR 转写")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--closure-sec", type=float, default=0.06,
                    help="b/p/m/f 字首闭合时长")
    ap.add_argument("--silence-gap", type=float, default=0.30,
                    help="字间隔超过此值判为乐句停顿，插入闭嘴 X")
    ap.add_argument("--hold-to-next", action="store_true", default=True,
                    help="口型保持到下一个字起点（补偿对齐模型 240ms 时长上限）")
    ap.add_argument("--gate-db", type=float, default=-40.0,
                    help="静音门限（相对峰值 dB）。闭嘴只发生在真正无声处，"
                         "而不是按字间隔时长猜——见下方 build_timeline 注释")
    ap.add_argument("--no-energy-gate", action="store_true",
                    help="退回旧的按 --silence-gap 时长判静音（不推荐，唱歌上不成立）")
    ap.add_argument("--max-hold-sil", type=float, default=0.35,
                    help="hold 到下一字时，尾部允许夹带的连续静音上限（秒）。"
                         "is_voiced 是「任意一帧过门限」，混响尾巴或伴奏泄漏擦过门限就会"
                         "把整段停顿判成拖腔，嘴僵在半开；超出则回退到该静音起点")
    ap.add_argument("--melisma-sec", type=float, default=0.45,
                    help="超过此时长的字视为拖腔，按「枣核型」拆成峰值+衰减两段；0=关闭。"
                         "唱歌时 17/27 个字超过 0.8s，不拆则口型长时间定格")
    ap.add_argument("--no-decay", action="store_true",
                    help="拖腔不做衰减降档（保持峰值口型到底）")
    # VOCAL (SIGGRAPH Asia'22) 的 Melodic accent：唱歌时元音承载旋律、辅音只管节奏，
    # Ma 越高则辅音越弱化、相邻元音互相融合、嘴保持半开。
    # 这里用它控制「短元音不给满开口」——论文指出语音模型用于唱歌的典型失败正是
    # 短元音 over-articulate（为每个元音机械地全开全闭下巴）。
    ap.add_argument("--melodic-accent", type=float, default=0.6,
                    help="Ma∈[0,1]：0=按说话方式逐字清晰发音，1=元音充分融合、"
                         "短音不满开口。流行歌 0.5-0.7，说唱接近 0，抒情/美声接近 1")
    ap.add_argument("--short-vowel-sec", type=float, default=0.20,
                    help="说话元音平均时长 τ（Kuwabara 1996，VOCAL 采用 0.2s）。"
                         "短于 τ·(1+Ma) 的元音按 Ma 降档，避免机械式全开全闭")
    args = ap.parse_args()

    cache = os.path.expanduser("~/.cache/modelscope/models")
    fa_dir = os.path.join(cache, "iic--speech_timestamp_prediction-v1-16k-offline",
                          "snapshots", "master")
    asr_dir = os.path.join(
        cache, "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "snapshots", "master")
    vad_dir = os.path.join(cache, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                           "snapshots", "master")
    if not os.path.exists(os.path.join(fa_dir, "model.pt")):
        sys.exit(f"缺少强制对齐模型: {fa_dir}")
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # FunASR 的 VAD/推理消耗全局 RNG，不固定种子同一文件多次结果不同
    # （AGENTS.md 已记录该陷阱）。每次 generate 前都重置。
    import numpy as np
    import torch

    def seed():
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        np.random.seed(0); random.seed(0)

    from funasr import AutoModel
    seed()

    import soundfile as sf
    info = sf.info(args.wav)

    if args.lrc:
        # 直接从增强 LRC 读时间戳（人工校对，无模型、无漂移、无重复段歧义）。
        # 行级时间（KRC 人工校对，实测与音频能量跃变差 <0.2s）是权威锚点；
        # 字级时间：逐字行直接用逐字时间戳，行级行按行内字数均摊——
        # 口型只需要「这行大约何时唱」与大致节奏，发音不需字级精准。
        lyr, ts = load_lrc_ts(args.lrc, args.wav)
        src = args.lrc
        n = len(ts)
        print(f"LRC 时间轴: {n} 字", file=sys.stderr)
    else:
        if args.text:
            lyr = norm_text(open(args.text, encoding="utf-8").read())
            src = args.text
        else:
            m = AutoModel(model=asr_dir, vad_model=vad_dir, device=args.device,
                          disable_update=True, disable_pbar=True)
            seed()
            r = m.generate(input=args.wav, batch_size_s=300)
            lyr = norm_text("".join(s.get("text", "") for s in r))
            src = "ASR 转写"
            del m
        if not lyr:
            sys.exit("歌词为空")

        # 纯汉字歌词池（供 ASR 句子匹配）与索引映射：plain 位置 → lyr 位置。
        # lyr 可能含英文占位「嗯嗯」（见 norm_text），ASR 句子文本是真实汉字，
        # 匹配必须在纯汉字池上做，匹配结果再映射回带占位的 lyr 索引。
        lyr_plain = "".join(ch for ch in lyr if "\u4e00" <= ch <= "\u9fff")
        plain_to_lyr = []
        for i, ch in enumerate(lyr):
            if "\u4e00" <= ch <= "\u9fff":
                plain_to_lyr.append(i)

        seed()
        fam = AutoModel(model=fa_dir, device=args.device, disable_update=True,
                        disable_pbar=True)

        # ---- 分段强制对齐（修复 fa-zh 长音频漂移，方案 A：ASR 句子级对齐）----
        # fa-zh 在整段音频上做对齐时，时间戳随音频增长系统性偏移（实测雨爱全曲
        # 96.74s 起每行错位 +4.5~+7.5s；Issue #2784 确认段内首字偏移 1.5-2s）。
        # 社区共识（WhisperX 论文 / MFA 官方）：长音频强制对齐必须切成短段，
        # 每段独立对齐。本实现的流程：
        #   1. 按长静音（>2s）切出有声段；有声段 >max_seg_sec（默认 60）时
        #      在能量谷点（min-cut）继续切开
        #   2. 用 ASR（paraformer + VAD）对全曲跑句子级时间戳——VAD 时间戳是
        #      帧级可靠的（不受 fa-zh 漂移影响），用它锚定每块的歌词范围
        #   3. 每块独立 fa-zh 对齐
        #   4. 块内所有字符减去「首字对齐时间 − 块内第一有声帧」的系统偏移
        # 歌词到块的分配：用 ASR 句子文本与真歌词做序列匹配（difflib），
        # 每个句子得到其在歌词中的位置，块 = 时间上覆盖该块的句子 → 歌词区间。
        # ASR 文本错字多但句子边界是 VAD 级可靠的，序列匹配只要求大致位置。

        import numpy as np
        _y, _sr = sf.read(args.wav)
        _hop = _sr // 200          # 5ms 分辨率
        _rms = np.sqrt(np.mean((_y[:len(_y)//_hop*_hop].reshape(-1, _hop))**2, axis=1))
        _gate = _rms.max() * 10 ** (args.gate_db / 20.0)

        # 找 >2s 的静音段作为分段边界
        sil = False
        sil_start = 0.0
        bounds = []
        n_frames = len(_rms)
        dur_total = n_frames / 200.0
        for i in range(n_frames):
            v = _rms[i] > _gate
            t = i / 200.0
            if not v and not sil:
                sil, sil_start = True, t
            elif v and sil:
                if t - sil_start > 2.0:
                    bounds.append((sil_start, t))
                sil = False
        if sil and dur_total - sil_start > 2.0:
            bounds.append((sil_start, dur_total))

        # 有声段 = 静音段之间的区间；只保留时长 >2s 的（歌词都在里面）
        segments = []
        prev_end = 0.0
        for s, e in bounds:
            if s - prev_end > 2.0:
                segments.append((prev_end, s))
            prev_end = e
        if dur_total - prev_end > 2.0:
            segments.append((prev_end, dur_total))

        def first_voiced_in(a, b, min_run=0.08):
            """[a, b) 内第一段持续有声的起点；没有则 a。"""
            ia, ib = int(round(a * 200)), min(int(round(b * 200)), n_frames)
            v = _rms[ia:ib] > _gate
            need = max(1, int(round(min_run * 200)))
            run = 0
            for i, x in enumerate(v):
                if x:
                    run += 1
                    if run >= need:
                        return (ia + i + 1 - run) / 200.0
                else:
                    run = 0
            return a

        # min-cut：把过长有声段在能量谷点切开（WhisperX 的 Cut & Merge 思路）
        def min_cut(seg, max_len):
            """把 (a,b) 切成长度 ≤max_len 的小块；切点在能量谷（避免切在字中间）。"""
            a, b = seg
            if b - a <= max_len:
                return [seg]
            out = []
            cur = a
            while b - cur > max_len:
                # 在 [cur+max_len*0.4, cur+max_len*0.6] 内找 RMS 最低点（谷）
                lo = int((cur + max_len * 0.4) * 200)
                hi = min(int((cur + max_len * 0.6) * 200), n_frames)
                if hi <= lo:
                    cut = cur + max_len * 0.5
                else:
                    cut = (lo + int(np.argmin(_rms[lo:hi]))) / 200.0
                out.append((cur, cut))
                cur = cut
            out.append((cur, b))
            return out

        max_seg_sec = 60.0
        blocks = []
        for seg in segments:
            blocks.extend(min_cut(seg, max_seg_sec))

        if len(blocks) <= 1:
            # 没有长静音且不长：退回整段对齐
            seed()
            res = fam.generate(input=(args.wav, lyr), data_type=("sound", "text"))
            ts = res[0].get("timestamp")
            if not ts:
                sys.exit("对齐失败：没有返回时间戳")
            if len(ts) != len(lyr):
                print(f"警告: 时间戳数 {len(ts)} != 字数 {len(lyr)}，按较短者截断",
                      file=sys.stderr)
            n = min(len(ts), len(lyr))
            ts = ts[:n]
        else:
            # 用 ASR 句子级时间戳锚定每块的歌词范围
            from difflib import SequenceMatcher
            seed()
            asr_m = AutoModel(model=asr_dir, vad_model=vad_dir, device=args.device,
                              disable_update=True, disable_pbar=True)
            seed()
            asr_res = asr_m.generate(input=args.wav, batch_size_s=60,
                                     sentence_timestamp=True)
            del asr_m
            asr_sents = []
            for s in asr_res[0].get("sentence_info", []):
                t0, t1 = s["start"] / 1000.0, s["end"] / 1000.0
                text = s.get("text", "") or ""
                asr_sents.append((t0, t1, text))

            def match_lyrics(asr_text, lyr_pool, start_hint, expect_pos):
                """在 lyr_pool（纯汉字池）中找 asr_text 的最佳匹配起点。
                返回 (pos, ratio)；pos 为 None 表示没找到。
                搜索中心 = expect_pos（句子时间占比 × 歌词总长，VAD 级可靠），
                窗口 = expect ± 8% 全文。副歌重复段里文本会匹配到错误的重复处，
                时间占比是唯一能区分重复段的线索，所以窗口必须小。"""
                a = norm_text(asr_text)
                if not a:
                    return None, 0.0
                best_pos, best_ratio = None, 0.0
                span = len(lyr_pool)
                win = max(span // 12, 20)
                lo = max(0, expect_pos - win)
                hi = min(len(lyr_pool) - len(a), expect_pos + win)
                if hi <= lo:
                    hi = max(lo + 1, len(lyr_pool) - len(a))
                for pos in range(lo, hi + 1):
                    r = SequenceMatcher(None, a, lyr_pool[pos:pos + len(a)],
                                        autojunk=False).ratio()
                    if r > best_ratio:
                        best_ratio, best_pos = r, pos
                return best_pos, best_ratio

            # 逐句子匹配：**文本匹配定 lo（非重复段可靠），hi = 下一句 lo**。
            # 副歌重复段（雨爱「雨爱的秘密能一直延续」出现 3 次）文本完全相同，
            # 文本匹配会选错重复处——此时用「有声时长占比」期望位置修正：
            # 文本匹配位置与期望位置偏差 > 全文 4% 视为重复段错位，改用期望。
            # 长句（如 36s 的 [123.84-160.55]）的区间 = [lo, 下一句 lo)，由
            # fa-zh 对齐自身决定句内字分布，不依赖时间占比的密度假设。
            sent_positions = []   # (t0, t1, lyr_pos_lo, ratio)
            # 累计有声时长曲线：t -> 该时刻之前的有声总时长
            cum_voiced = np.zeros(n_frames + 1)
            for i in range(n_frames):
                cum_voiced[i + 1] = cum_voiced[i] + (1.0 if _rms[i] > _gate else 0.0)
            total_voiced = cum_voiced[-1]

            def voiced_frac(t):
                idx = min(max(int(t * 200), 0), n_frames)
                return cum_voiced[idx] / total_voiced if total_voiced > 0 else 0.0

            for t0, t1, text in asr_sents:
                mid = (t0 + t1) / 2.0
                expect = int(voiced_frac(mid) * len(lyr_plain))
                pos, ratio = match_lyrics(text, lyr_plain, 0, expect)
                if pos is not None and ratio >= 0.35 and \
                   abs(pos - expect) <= len(lyr_plain) // 25:
                    lyr_pos = plain_to_lyr[pos]      # 文本匹配可信
                else:
                    lyr_pos = plain_to_lyr[min(max(expect, 0),
                                               len(plain_to_lyr) - 1)]
                sent_positions.append((t0, t1, lyr_pos, ratio if pos is not None else 0.0))
            # hi = 下一句的 lo；首句 lo=0；末句 hi=len(lyr)
            for i in range(len(sent_positions)):
                t0, t1, lo, ratio = sent_positions[i]
                if i == 0:
                    lo = 0
                hi = sent_positions[i + 1][2] if i + 1 < len(sent_positions) else len(lyr)
                if hi <= lo:
                    hi = lo + 1
                sent_positions[i] = (t0, t1, lo, hi, ratio)
            print(f"ASR 锚定: {len(sent_positions)}/{len(asr_sents)} 句（文本 lo + 单调 hi）",
                  file=sys.stderr)

            # ---- 以 ASR 句子为对齐单元（VAD 段 = 天然乐句边界，fa-zh 短段无漂移）----
            # 每句的对齐歌词 = [lo, hi)（已由时间占比 + 文本校验锚定）。
            # 注意：ts 必须与 lyr 索引一一对应（下游 for i in range(n): ch=lyr[i]），
            # 句子对齐失败时不能跳过——用前后句的时间线性插值占位。
            import subprocess as _sp
            ts_all = [None] * len(lyr)   # 占位，失败句由插值填补
            for sent_i, (t0, t1, lo, hi, ratio) in enumerate(sent_positions):
                lo = max(lo, 0)
                hi = min(hi, len(lyr))
                if hi <= lo:
                    hi = lo + 1
                seg_lyr = lyr[lo:hi]
                if not seg_lyr:
                    continue
                tmp_wav = os.path.join(os.path.dirname(args.out),
                                       f".align_sent{sent_i}.wav")
                _sp.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}",
                         "-t", f"{t1 - t0:.3f}", "-i", args.wav, tmp_wav],
                        check=True)
                seed()
                res = fam.generate(input=(tmp_wav, seg_lyr),
                                   data_type=("sound", "text"))
                seg_ts = res[0].get("timestamp")
                if not seg_ts or len(seg_ts) != len(seg_lyr):
                    print(f"警告: 句 {sent_i} 对齐 {len(seg_ts) if seg_ts else 0} "
                          f"!= {len(seg_lyr)} 字，该句留待插值", file=sys.stderr)
                    os.remove(tmp_wav)
                    continue
                # 句内偏移校正：fa-zh 在 VAD 子段上对齐时，首字相对句起点有
                # 系统偏移（实测 1.5-2s，Issue #2784）。句起点应落在第一有声
                # 帧上，所以把「首字对齐时间 − 第一有声帧」作为该句统一偏移。
                fv = first_voiced_in(t0, t1)          # 全局秒
                offset_ms = seg_ts[0][0] - (fv - t0) * 1000.0   # 句内毫秒
                for j, t in enumerate(seg_ts):
                    st = int(round(t0 * 1000 + t[0] - offset_ms))
                    en = int(round(t0 * 1000 + t[1] - offset_ms))
                    ts_all[lo + j] = (st, en)
                os.remove(tmp_wav)
            # 插值填补失败句：对 None 位置用前一个有效值 + 60ms 步进（保持索引对齐）
            last = None
            for i in range(len(ts_all)):
                if ts_all[i] is not None:
                    last = ts_all[i]
                elif last is not None:
                    ts_all[i] = (last[0] + 60, last[1] + 120)
            ts = [v for v in ts_all if v is not None]
            n = len(ts)
            print(f"句子级对齐: {len(sent_positions)} 句 / {n} 字", file=sys.stderr)

    pinyin, Style = load_pinyin()
    cues = []

    # 归并到启用集合：四型方案下 D/F/G/H 一律沿官方过渡链回退。
    # 放在 push 里做兜底，这样无论上游哪条分支产出的口型都不会漏出去。
    FOLD = {"D": "C", "F": "E", "G": "B", "H": "C"}

    def push(a, b, v):
        if b - a <= 1e-4:
            return
        v = FOLD.get(v, v)
        if v not in ENABLED_SHAPES:
            v = "C"
        if cues and cues[-1]["value"] == v and abs(cues[-1]["end"] - a) < 1e-6:
            cues[-1]["end"] = b
        else:
            cues.append({"start": round(a, 3), "end": round(b, 3), "value": v})

    chars = []
    prev_end = 0.0

    # 能量门限：按实际音频判静音，而不是按字间隔时长猜。
    # 唱歌拖腔时字间隔通常 0.6-1.5s，全都有声；乐句间停顿也可能只有 0.3s。
    # 时长判据（silence_gap）对这两种情况都说错，测量结果：9 个 X 段里 8 个有声。
    import numpy as np
    _y, _sr = sf.read(args.wav)
    _hop = _sr // 200          # 5ms 分辨率
    _rms = np.sqrt(np.mean((_y[:len(_y)//_hop*_hop].reshape(-1, _hop))**2, axis=1))
    _gate = _rms.max() * 10 ** (args.gate_db / 20.0)

    def is_voiced(t_start, t_end):
        """该时段内是否有任何有声帧（超过能量门限）。"""
        a = int(round(t_start * 200))
        b = min(int(round(t_end * 200)), len(_rms))
        if b <= a:
            return False
        return bool((_rms[a:b] > _gate).any())

    def last_voiced_before(t, t_floor, min_run=0.08):
        """[t_floor, t) 内最后一段「持续」有声的结束时刻；没有则 None。

        判据必须是持续而非单帧：混响尾巴和伴奏泄漏会让孤立帧擦过门限
        （实测 72.6-73.1s 一整段 max 只有 -40.2dB 却含过限帧），用单帧判据
        会认为「一直有声」，裁剪永不触发。min_run 默认 80ms ≈ 一个音节的下限。
        """
        a = int(round(t_floor * 200))
        b = min(int(round(t * 200)), len(_rms))
        if b <= a:
            return None
        v = _rms[a:b] > _gate
        need = max(1, int(round(min_run * 200)))
        end = None
        run = 0
        for i, x in enumerate(v):
            if x:
                run += 1
                if run >= need:
                    end = i + 1
            else:
                run = 0
        return None if end is None else (a + end) / 200.0

    def first_long_silence(t_start, t_end, min_sil=0.35):
        """[t_start, t_end) 内第一段「持续」静音超过 min_sil 的起点；没有则 t_end。

        裁剪拖腔时的正确判据。last_voiced_before 会误把「下一字的起唱」当成
        本字拖音：扫描 [en, nxt] 时若间奏之后下一字提前发声，返回的是下一字的
        声音结尾，hold_end - q 只剩 0.1s，20s 的间奏被完整 hold 过去
        （实测雨爱 99.07-125.01s 出现 25.94s 的连续张嘴）。找「第一段长静音」
        从源头断开：拖腔尾部最多允许夹带 min_sil 的静音，超出即闭嘴。
        """
        a = int(round(t_start * 200))
        b = min(int(round(t_end * 200)), len(_rms))
        if b <= a:
            return t_end
        v = _rms[a:b] > _gate
        need = max(1, int(round(min_sil * 200)))
        run = 0
        for i, x in enumerate(v):
            if not x:
                run += 1
                if run >= need:
                    return (a + i + 1 - run) / 200.0
            else:
                run = 0
        return t_end

    for i in range(n):
        ch = lyr[i]
        st, en = ts[i][0] / 1000.0, ts[i][1] / 1000.0
        nxt = ts[i + 1][0] / 1000.0 if i + 1 < n else None

        # 字的实际发声结束：
        #   - 旧逻辑（--no-energy-gate）：按时长 silence_gap 猜
        #   - 新逻辑：能量门控——只要到下一个字起点之间还有声音，就 hold 过去
        if args.no_energy_gate:
            hold_end = en
            if args.hold_to_next and nxt is not None:
                gap = nxt - en
                hold_end = nxt if gap <= args.silence_gap else en
        else:
            hold_end = en
            if args.hold_to_next and nxt is not None:
                # 把间隔分成两段：本字尾端到下一字头的中间点。
                # 若中间点之前有声 → 拖腔，hold 到下一字起点。
                # 若中间点之前无声 → 停顿，保持到 en（让 push 插 X）。
                mid = en + (nxt - en) * 0.5
                hold_end = nxt if is_voiced(en, mid) else en
                # is_voiced 是「任意一帧」判据，够宽松才能接住拖腔，但也会被混响尾巴
                # 或伴奏泄漏骗过去（实测 -40.2dB 擦过 -40 门限，把 3.5s 停顿判成拖腔，
                # 嘴僵在半开 5.6s）。所以 hold 之后再回头砍掉尾部的连续静音。
                if hold_end > en and args.max_hold_sil > 0:
                    # 不能直接 last_voiced_before(hold_end, en)：下一字起唱会被误认
                    # 为本字拖音（见 first_long_silence 注释）。改为在长静音起点断开。
                    cut = first_long_silence(en, hold_end, args.max_hold_sil)
                    if cut < hold_end:
                        hold_end = cut

        shape, ini = shape_of(ch, pinyin, Style)

        # Melodic accent（VOCAL 的 Ma）：短元音不给满开口。
        # 论文指出语音模型用于唱歌时的典型失败是短元音 over-articulate ——
        # "robotically opening/closing the jaw completely for each vowel"。
        # 唱歌里一个一闪而过的 a 根本来不及把下巴放到底，视觉上却最扎眼，
        # 所以短于 τ·(1+Ma) 的音按 DECAY_SHAPE 降一档。
        vdur = hold_end - st
        if args.melodic_accent > 0:
            thresh = args.short_vowel_sec * (1.0 + args.melodic_accent)
            if vdur < thresh:
                # 只降「开口过大」的那一档，且不跨唇形族：
                # 短促的 a 来不及把下巴放到底 -> C；其余元音本身开口就不大，
                # 再降会丢掉元音辨识度（这是 B/H/F 泛滥的另一个来源）。
                shape = SHORT_SHAPE.get(shape, shape)

        chars.append({"ch": ch, "start": round(st, 3), "end": round(hold_end, 3),
                      "shape": shape})

        # 字前的间隔：若真的无声才插 X，否则保持上一个口型
        if st - prev_end > 1e-3:
            if not is_voiced(prev_end, st):
                push(prev_end, st, "X")
            else:
                # 有声间隔：上一字口型 hold 到本字起点（已在上一轮 push 了 hold_end=nxt）
                pass   # prev_end==hold_end of prev char already covers it
        cl = INITIAL_SHAPE.get(ini)
        if cl and hold_end - st > args.closure_sec * 2:
            push(st, st + args.closure_sec, cl)
            body_start = st + args.closure_sec
            body_end = hold_end
        else:
            body_start = st
            body_end = hold_end

        # 拖腔处理：「枣核型」两段式，而不是按音量密集切分。
        #
        # 上一版把长音按 180ms 切成十几段、形状跟着 RMS 走，方向是错的：
        #   1. 口型编码的是元音形状，与音量无关——轻声的「啊」嘴也该是大的；
        #   2. 行业规范明确反对逐音节打点（Bristow 第 3 条 "Too many mouth shapes":
        #      "It's a common mistake to try to hit every single phoneme with a key
        #       mouth pose. Identify the important shapes that define the word, and
        #       skip over the lesser ones."）。实测那样做 cue 密度到 3.8/s，
        #      远超真实音节 1.37/s，看起来就是无意义嚅动。
        #
        # 正确做法：一个字只给「峰值 + 衰减」两个口型。汉语音节是枣核型——
        # 中部开口最大、两端收拢，所以长音后半段降一档（DECAY_SHAPE）即可，
        # 既避免长时间定格，又不会制造假动作。
        body_dur = body_end - body_start
        if args.melisma_sec > 0 and body_dur > args.melisma_sec and not args.no_decay:
            # 峰值段占前 45%，其余为衰减段（枣核型的收拢尾）
            peak_end = body_start + body_dur * 0.45
            push(body_start, peak_end, shape)
            push(peak_end, body_end, DECAY_SHAPE.get(shape, shape))
        else:
            push(body_start, body_end, shape)

        prev_end = hold_end

    if info.duration > prev_end:
        # 尾部：保持最后一个口型到「真正唱完」为止，之后闭嘴。
        # 判据同 --max-hold-sil：必须是**持续**有声，不能用 is_voiced 的单帧判据 ——
        # 尾奏/淡出的孤立过限帧会让最后一个口型一路撑到文件结尾（实测全曲版尾部
        # 220.2s 后是数字静音，却被判成有声，嘴张着 C 型僵了 20.5s）。
        tail = last_voiced_before(info.duration, prev_end)
        if tail is None:
            push(prev_end, info.duration, "X")
        else:
            tail = min(info.duration, tail + args.max_hold_sil)
            if tail > prev_end:
                push(prev_end, tail, cues[-1]["value"] if cues else "X")
            if info.duration > tail:
                push(tail, info.duration, "X")

    out = {"metadata": {"soundFile": os.path.abspath(args.wav),
                        "duration": round(info.duration, 3),
                        "lyricsSource": src, "chars": len(chars)},
           "mouthCues": cues, "chars": chars}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    wt = Counter()
    for c in cues:
        wt[c["value"]] += c["end"] - c["start"]
    tot = sum(wt.values())
    print(f"{len(chars)} 字 → {len(cues)} cues，写入 {args.out}")
    print("口型时长占比: " + "  ".join(
        f"{k}={wt[k] / tot * 100:.0f}%" for k in "XABCDEFGH" if wt.get(k)))


if __name__ == "__main__":
    main()
