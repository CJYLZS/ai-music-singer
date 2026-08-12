#!/usr/bin/env python
"""主线对比: score-control SVS vs 现有 SVC 流水线，同歌同参考。

这是回到最初目标的脚本 —— 验证「MIDI + 参考音频 + 歌词」生成的唱歌，是否比
`sing.py` 的 SVC 流水线更好。之前几轮都跑在上游 demo 歌上，而 SVC 的历史成绩在
我们自己的歌上，两者不可比。

做法（全自动，不需要人工乐谱）：
  1. 用 SoulX 自己的预处理，从 <song>/sep2/lead.wav（原唱干声）转出目标 metadata
     —— 音符(ROSVOT) + 歌词(paraformer) + 时长。这一步替代人工精校乐谱，是
     "SVS 能否上生产" 的真正门槛（上游 README 明确警告转录误差影响很大）。
  2. 用 <song>/prompt.wav（参考人说话）或指定的唱歌干声作 prompt，同样自动标注。
  3. score 模式合成；分段推理后拼接成整首。
  4. 与 <song>/vocal.wav（SVC 输出）在同一批指标上对比。

关键取舍：**逐段合成再拼接**。`cli/inference.py` 一次只用 prompt_meta_list[0]，
且 SVS 是段级模型；整首 270s 不可能一次喂进去。所以按预处理切出的段逐段推理，
每段都用同一个 prompt（保证音色一致），最后按 time 字段拼回原时间轴。

用法:
  cd SoulX-Singer && PYTHONPATH=. .venv/bin/python ../singing/svs_song.py \
      --song-dir ../data/songs/绿色          # 产物默认落到 data/songs/绿色/svs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def build_meta(wav: str, save_dir: str, language: str, device: str,
               max_merge_ms: int, tag: str) -> tuple[str, str]:
    """跑 SoulX 预处理，返回 (metadata_path, cut_wav_dir)。"""
    from preprocess.pipeline import PreprocessPipeline
    import librosa
    import soundfile as sf

    os.makedirs(save_dir, exist_ok=True)
    mono = os.path.join(save_dir, "mono.wav")
    if not os.path.exists(mono):
        y, sr = librosa.load(wav, sr=None, mono=True)
        sf.write(mono, y, sr)
        print(f"[{tag}] {wav} -> {len(y)/sr:.1f}s mono")

    meta_path = os.path.join(save_dir, "metadata.json")
    if os.path.exists(meta_path):
        print(f"[{tag}] 复用已有 metadata: {meta_path}")
    else:
        pipe = PreprocessPipeline(
            device=device, language=language, save_dir=save_dir,
            vocal_sep=False,          # 输入已是干声
            max_merge_duration=max_merge_ms,
            midi_transcribe=True,     # score 模式必须有 note_pitch
        )
        pipe.run(mono, vocal_sep=False, max_merge_duration=max_merge_ms,
                 language=language)

    with open(meta_path, encoding="utf-8") as f:
        metas = json.load(f)
    print(f"[{tag}] {len(metas)} 段")
    return meta_path, save_dir


def trim_stratified(y: np.ndarray, sr: int, max_sec: float,
                    f0: np.ndarray | None = None) -> np.ndarray:
    """把 y 裁到 max_sec，但**保持原素材的音高构成**，而不是取时间前缀。

    动机: prompt_ext.wav 按移调量排序拼接（w_ext.py:_render 的
    `for sh in sorted(by_shift)`），高音段全在尾部。取前 N 秒会 100% 切掉音域扩展 ——
    实测这样得到的 prompt 与完全不做 ext 的 prompt.wav 不可分辨（p=0.064）。

    做法: 按 F0 分成低/中/高三层（<380 / 380-460 / >=460 Hz，与评测分桶一致），
    每层按其在原素材中的时长占比分配预算，层内均匀取块。块粒度 0.5s，避免切碎音节。
    F0 缺失时退回按时间均匀抽样（仍比取前缀好，至少覆盖全曲构成）。
    """
    n_keep = int(max_sec * sr)
    if len(y) <= n_keep:
        return y
    blk = int(0.5 * sr)                      # 0.5s 块，避免切碎音节
    n_blk = len(y) // blk
    if n_blk == 0:
        return y[:n_keep]

    if f0 is not None and len(f0) > 0:
        # 每块的中位 F0（f0 是 50Hz 网格）
        band = []
        for i in range(n_blk):
            a = int(i * blk / sr * 50.0)
            b = int((i + 1) * blk / sr * 50.0)
            v = f0[a:b]
            v = v[v > 0]
            m = float(np.median(v)) if len(v) else 0.0
            band.append(0 if m < 380 else (1 if m < 460 else 2))
        band = np.asarray(band)
    else:
        band = np.zeros(n_blk, dtype=int)

    keep_blk = max(1, n_keep // blk)
    chosen: list[int] = []
    for bi in (0, 1, 2):
        idx = np.flatnonzero(band == bi)
        if len(idx) == 0:
            continue
        # 该层配额 = 该层块数占比 × 总配额；层内均匀取（保留时间顺序与语气连贯）
        quota = int(round(keep_blk * len(idx) / n_blk))
        quota = max(1, min(quota, len(idx)))
        pick = idx[np.linspace(0, len(idx) - 1, quota).round().astype(int)]
        chosen.extend(int(x) for x in pick)
    chosen = sorted(set(chosen))[:keep_blk]
    out = np.concatenate([y[i * blk:(i + 1) * blk] for i in chosen])
    return out[:n_keep]


def wav_for(seg_index: str, save_dir: str) -> str:
    for sub in ("long_cut_wavs", "cut_wavs"):
        p = os.path.join(save_dir, sub, f"{seg_index}.wav")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到 {seg_index} 的 wav（{save_dir}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", required=True)
    ap.add_argument("--out-dir", default="",
                    help="产物目录。默认 <song-dir>/svs/ —— 听感产物必须落在 data/ 下，"
                         "不要写 /tmp（会被清理，且用户无法长期回听）")
    ap.add_argument("--prompt-wav", default="",
                    help="参考音色音频。默认 <song>/prompt.wav（说话）。"
                         "传 refsep2/lead.wav 可用参考人的唱歌干声（VDE 更好）")
    ap.add_argument("--prompt-max-sec", type=float, default=20.0,
                    help="prompt 时长上限（inference 只用第一段）")
    ap.add_argument("--prompt-trim", default="stratified",
                    choices=["head", "stratified"],
                    help="prompt 超长时怎么裁。stratified（默认）按音高分层等比抽样，"
                         "head = 旧行为「取前 N 秒」。**对 prompt_ext.wav 必须用 "
                         "stratified**：w_ext.py 的 _render 按移调量排序拼接 "
                         "(for sh in sorted(by_shift))，所有移调高音段都在尾部，"
                         "头部截断会把音域扩展 100% 切掉。实测爱笑的眼睛 45s 的 "
                         "prompt_ext 前 24.76s 是 +0st 干净素材(>380Hz 仅 0.9%)、"
                         "后 20.21s 才是移调段(>380Hz 占 79.9%)，取前 18.55s 得到的 "
                         "prompt 与不做 ext 的 prompt.wav 在 SIM 上不可分辨(p=0.064)")
    ap.add_argument("--language", default="Mandarin")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--control", default="score", choices=["score", "melody"])
    ap.add_argument("--cfg", type=float, default=None,
                    help="CFM 的 CFG 引导强度。不给则用 config.infer.cfg（上游 3.0）")
    ap.add_argument("--max-merge-ms", type=int, default=20000)
    ap.add_argument("--limit-segments", type=int, default=0,
                    help=">0 只跑前 N 段（先验证流程再跑整首）")
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    song = args.song_dir.rstrip("/")
    name = os.path.basename(song)
    src = os.path.join(song, "sep2", "lead.wav")
    prompt_src = args.prompt_wav or os.path.join(song, "prompt.wav")
    for p in (src, prompt_src):
        if not os.path.exists(p):
            sys.exit(f"缺少 {p}")

    if not args.out_dir:
        args.out_dir = os.path.join(song, "svs")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. 目标侧: 从原唱干声自动转录 MIDI + 歌词 ----
    tgt_meta_path, tgt_dir = build_meta(
        src, os.path.join(args.out_dir, "target"), args.language,
        args.device, args.max_merge_ms, "target")

    # ---- 2. prompt 侧 ----
    import librosa
    import soundfile as sf
    pdir = os.path.join(args.out_dir, "prompt")
    os.makedirs(pdir, exist_ok=True)
    ptrim = os.path.join(pdir, "src.wav")
    if not os.path.exists(ptrim):
        y, sr = librosa.load(prompt_src, sr=None, mono=True)
        if args.prompt_trim == "head":
            y = y[: int(args.prompt_max_sec * sr)]
        else:
            # 同名 _f0.npy 若存在就用它分层（prompt_ext.wav -> prompt_ext_f0.npy）
            f0 = None
            cand = os.path.splitext(prompt_src)[0] + "_f0.npy"
            if os.path.exists(cand):
                f0 = np.load(cand)
            before = len(y) / sr
            y = trim_stratified(y, sr, args.prompt_max_sec, f0)
            if before > args.prompt_max_sec:
                print(f"[prompt] 分层裁剪 {before:.2f}s -> {len(y)/sr:.2f}s"
                      f"（f0={'有' if f0 is not None else '无，退回时间均匀'}）")
        sf.write(ptrim, y, sr)
    pmeta_path, pdir_out = build_meta(
        ptrim, pdir, args.language, args.device,
        int(args.prompt_max_sec * 1000) + 5000, "prompt")

    with open(pmeta_path, encoding="utf-8") as f:
        pmetas = json.load(f)
    if not pmetas:
        sys.exit("prompt 预处理没产出段")
    pfirst = pmetas[0]
    pwav = wav_for(pfirst["index"], pdir_out)
    psolo = os.path.join(pdir, "prompt_first.json")
    with open(psolo, "w", encoding="utf-8") as f:
        json.dump([pfirst], f, ensure_ascii=False, indent=2)
    pn = len(pfirst["note_pitch"].split())
    print(f"[prompt] 用 {pfirst['index']}  {pfirst['time']}  {pn} notes  -> {pwav}")

    # ---- 3. score 合成 ----
    # 上游 cli/inference.py 本身就处理多段并按 metadata 的 `time` 拼回原时间轴
    # （generated_len 取最后一段的 time[1]，见 inference.py:90-97）。所以整个
    # metadata 一次传进去即可，**不要**逐段调用后自己拼 —— 每段输出都是"从 0 到
    # 本段 time[1]"的全长缓冲（前面填静音），自己再按 time[0] 偏移叠加会错位重叠。
    with open(tgt_meta_path, encoding="utf-8") as f:
        tmetas = json.load(f)
    if args.limit_segments:
        tmetas = tmetas[: args.limit_segments]
        tgt_meta_path = os.path.join(args.out_dir, "target_limited.json")
        with open(tgt_meta_path, "w", encoding="utf-8") as f:
            json.dump(tmetas, f, ensure_ascii=False, indent=2)

    gen_dir = os.path.join(args.out_dir, "generated")
    gen = os.path.join(gen_dir, "generated.wav")
    if not os.path.exists(gen):
        cmd = [
            sys.executable, "-m", "cli.inference",
            "--device", args.device,
            "--model_path", "pretrained_models/SoulX-Singer/model.pt",
            "--config", "soulxsinger/config/soulxsinger.yaml",
            "--prompt_wav_path", pwav,
            "--prompt_metadata_path", psolo,
            "--target_metadata_path", tgt_meta_path,
            "--phoneset_path", "soulxsinger/utils/phoneme/phone_set.json",
            "--save_dir", gen_dir,
            "--control", args.control,
            "--pitch_shift", "0",
            "--seed", str(args.seed),
            "--fp16",
        ]
        if args.cfg is not None:
            cmd += ["--cfg", str(args.cfg)]
        print(f"[svs] 合成 {len(tmetas)} 段（control={args.control}, seed={args.seed}"
              f"{'' if args.cfg is None else f', cfg={args.cfg}'}）")
        r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=os.getcwd()))
        if r.returncode != 0 or not os.path.exists(gen):
            sys.exit(f"! 合成失败 (exit {r.returncode})")

    y, sr = librosa.load(gen, sr=args.sr, mono=True)
    out = os.path.join(args.out_dir, "svs_vocal.wav")
    sf.write(out, y, args.sr)
    span = (min(m["time"][0] for m in tmetas) / 1000,
            max(m["time"][1] for m in tmetas) / 1000)
    print(f"\n[svs] -> {out}  {len(y)/args.sr:.1f}s"
          f"  (metadata 覆盖 {span[0]:.1f}-{span[1]:.1f}s)")

    # 供 CER 用的真值歌词（来自自动转录的 metadata，与合成同源）
    words = []
    for m in tmetas:
        tx = m["text"].split()
        ty = [int(x) for x in m["note_type"].split()]
        words += [w for w, y in zip(tx, ty) if y not in (1, 3) and not w.startswith("<")]
    ref_txt = os.path.join(args.out_dir, "ref_text.txt")
    with open(ref_txt, "w", encoding="utf-8") as f:
        f.write("".join(words))
    print(f"[svs] 目标歌词 {len(words)} 字 -> {ref_txt}")


if __name__ == "__main__":
    main()
