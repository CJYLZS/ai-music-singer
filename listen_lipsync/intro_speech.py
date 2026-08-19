#!/usr/bin/env python3
"""前奏口白工具：选模生成 → 时长校验 → 口型合并 → 混音 → 新 cues/audio。

把一段由 **txt 文件** 给出的口白合进 MV 的前奏器乐段（唱歌段时间轴完全不动）：

  1. 用选定的说话克隆模型生成口白（gpt-ft 微调 girlish / gpt-zs 基座零样本 / voxcpm2 零样本）
  2. 时长硬校验：offset + dur + margin <= first_lyric —— 口白必须短于前奏、绝不压唱段
  3. 对生成语音做强制对齐得到说话型口型（make_cues.py --melodic-accent 0）
  4. 把口白口型段插进原 cues.json 的前奏区（A/B/C/E/X），唱歌段 cues 原样保留
  5. 把口白混进 44.1k 立体声主轨，另写 16k 单声道 align（cues 的 soundFile 与真实音频一致）

用法（用 GPT-SoVITS venv 运行）; 约定：一首歌的全部 MV 中间素材放在
data/songs/<歌名>/mv/ 下（口白 txt、cues、beats、audio、混音、口型……），即"是否已备料"的凭据:
  GPT-SoVITS/.venv/bin/python listen_lipsync/intro_speech.py \\
      --text data/songs/情歌/mv/intro.txt \\
      --song-cues data/songs/情歌/mv/cues.json \\
      --base-audio data/songs/情歌/mv/audio.wav \\
      --out-dir data/songs/情歌/mv/ \\
      --tts gpt-ft \\
      --first-lyric 27.08 --offset 1.5 --margin 0.6 --speech-gain-db 4.0

产物（全部落在 --out-dir）:
  speech.wav           从 TTS 原样输出的口白
  speech_cues.json     make_cues 产出的口白口型段（0 基线）
  speech_16k.wav       对口白做的 16k 单声道（供对齐与评测）
  cues_intro.json      合并后的全曲口型（唱歌段未动）
  audio_intro.wav      混好口白的 44.1k 立体声主轨
  align_intro.wav      对应 16k 单声道（cues 的 soundFile 指向它）
  intro_meta.json      全部参数 + 校验结果 + 摘要
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
GPTV = ROOT / "GPT-SoVITS/.venv/bin/python"
VOXV = ROOT / "tools/voxcpm2/.venv/bin/python"
MAKE_CUES = ROOT / "listen_lipsync/make_cues.py"
GIRLISH = ROOT / "data/voices/girlish.list"

# 候选 TTS 的权重（gpt 系走 GPT-SoVITS infer.py；voxcpm2 走 test_gen.py）
GPT_WEIGHTS = {
    "gpt-ft": dict(
        gpt=ROOT / "GPT-SoVITS/GPT_weights_v2Pro/girlish-e30.ckpt",
        sovits=ROOT / "GPT-SoVITS/SoVITS_weights_v2Pro/girlish_e20_s440.pth",
        label="gpt-ft (girlish fine-tuned v2Pro)",
    ),
    "gpt-zs": dict(
        gpt=ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models"
                   "/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        sovits=ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        label="gpt-zs (v2Pro base zero-shot)",
    ),
}

EPS = 1e-3


def lookup_ref_text(ref_wav: str) -> str | None:
    """从 girlish.list 按文件名反查参考音频转写，跟 infer.py 保持一致。"""
    target = Path(ref_wav).name
    if not GIRLISH.exists():
        return None
    for line in GIRLISH.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("|")
        if len(parts) == 4 and Path(parts[0]).name == target:
            return parts[3].strip()
    return None


def run(cmd: list[str], cwd: Path, log: Path, env_extra: dict | None = None) -> None:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"error: {' '.join(cmd)}\n  log: {log}", file=sys.stderr)
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print("\n".join("  | " + t for t in tail), file=sys.stderr)
        sys.exit(1)


def resample(x: np.ndarray, sr_src: int, sr_dst: int) -> np.ndarray:
    import torch
    import torchaudio

    if sr_src == sr_dst:
        return x
    t = torch.from_numpy(x).float()
    return torchaudio.functional.resample(t, sr_src, sr_dst).numpy()


def generate_speech(args, out_dir: Path) -> tuple[Path, int]:
    """调 TTS 生成口白，返回 (speech.wav 绝对路径, 采样率)。"""
    speecw_path = out_dir / "speech.wav"
    if speecw_path.exists() and not args.force:
        print(f"复用已有 {speecw_path}")
        info = sf.info(speecw_path)
        return speecw_path, info.samplerate

    text = Path(args.text).read_text(encoding="utf-8").strip()
    out_abs = speecw_path.resolve()          # 子进程 cwd 不同，文件路径必须绝对
    ref_abs = Path(args.ref_audio).resolve()
    if args.tts == "voxcpm2":
        run([
            str(VOXV), str(ROOT / "tools/voxcpm2/test_gen.py"),
            "--ref", str(ref_abs),
            "--text", text,
            "--out", str(out_abs),
            "--seed", "42",
        ], ROOT, out_dir / "tts.log", env_extra={"HF_HUB_OFFLINE": "1"})
        return speecw_path, sf.info(speecw_path).samplerate

    w = GPT_WEIGHTS[args.tts]
    ref_text = args.ref_text or lookup_ref_text(ref_abs)
    cmd = [
        str(GPTV), str(ROOT / "GPT-SoVITS/infer.py"),
        "-t", "@" + str(Path(args.text).resolve()),
        "-o", str(out_abs),
        "--gpt", str(w["gpt"]),
        "--sovits", str(w["sovits"]),
        "--ref", str(ref_abs),
        "-q",
    ]
    if ref_text:
        cmd += ["--ref-text", ref_text]
    run(cmd, ROOT / "GPT-SoVITS", out_dir / "tts.log")
    return speecw_path, sf.info(speecw_path).samplerate


def build_speech_cues(speech16: Path, txt: Path, out: Path, device: str) -> dict:
    """对生成的 16k 口白做说话型口型（不带模型的强制对齐判断，结果落 out）。"""
    run([
        str(GPTV), str(MAKE_CUES),
        "--wav", str(speech16.resolve()),
        "--text", str(txt.resolve()),
        "--out", str(out.resolve()),
        "--melodic-accent", "0",
        "--device", device,
    ], ROOT, out.with_suffix(".log"))
    return json.loads(out.read_text(encoding="utf-8"))


def check_speech_cues(j: dict) -> None:
    """复刻 make_cues 的产出检查：单调、无零长、无同起点崩塌、无微字泛滥。"""
    cues = j["mouthCues"]
    if not cues:
        sys.exit("error: 口白口型为空")
    prev_end = -1.0
    prev_start = None
    micro = 0
    for i, c in enumerate(cues):
        if c["start"] >= c["end"] - 1e-4:
            sys.exit(f"error: 销毁字口型 [{i}] start={c['start']} end={c['end']}")
        if prev_start is not None and abs(c["start"] - prev_start) < 1e-4:
            sys.exit(f"error: 相邻 cue 同起点（对齐崩塌） [{i}] start={c['start']} '{c['value']}'")
        if c["start"] < prev_end - 1e-3:
            sys.exit(f"error: 时间轴倒序 [{i}] {c['start']} < {prev_end}")
        prev_start = c["start"]
        prev_end = c["end"]
        if c["end"] - c["start"] < 0.05:
            micro += 1
    if len(cues) and micro / len(cues) > 0.30:
        sys.exit(f"error: 微字口型占比过高 {micro}/{len(cues)}（<50ms），强制对齐疑似崩塌")


def push(cues: list[dict], a: float, b: float, v: str) -> None:
    """与 make_cues 相同的 push：同值相邻合并。"""
    a, b = round(a, 3), round(b, 3)
    if b - a <= 1e-4:
        return
    if cues and cues[-1]["value"] == v and abs(cues[-1]["end"] - a) < 1e-6:
        cues[-1]["end"] = b
    else:
        cues.append({"start": a, "end": b, "value": v})


def merge_cues(song: dict, speech_cues: dict, offset: float, first_lyric: float,
               chars_speech: int) -> dict:
    mouth = song["mouthCues"]
    # 找出唱歌段起点（第一条非 X 口型的 start = 首句）。前奏区 = [0, first_lyric)。
    k = next((i for i, c in enumerate(mouth) if c["value"] != "X"
              and c["start"] >= first_lyric - EPS), None)
    if k is None:
        sys.exit(f"error: 找不到首句起点（首条非 X 口型 vs --first-lyric {first_lyric}）")
    valid_first = mouth[k]["start"]
    if abs(valid_first - first_lyric) > 0.05:
        print(f"warn: --first-lyric={first_lyric} 与实测首句 {valid_first} 不一致，用实测", file=sys.stderr)
        first_lyric = valid_first

    # 口白段跨度（shift 之后）
    sc = speech_cues["mouthCues"]
    t0 = min(c["start"] for c in sc)
    t1 = max(c["end"] for c in sc)
    if offset < 0:
        sys.exit("error: --offset 不能为负")
    if offset + (t1 - t0) > first_lyric - EPS:
        sys.exit(
            f"error: 口白越界 offset={offset:.2f} + 段长 {t1 - t0:.2f} = "
            f"{offset + t1 - t0:.2f} > 首句 {first_lyric:.2f}\n"
            f"  请缩短口白文本或增大 --offset/--margin")

    region: list[dict] = []
    if offset > EPS:
        push(region, 0.0, offset, "X")
    for c in sc:
        push(region, c["start"] + offset, c["end"] + offset, c["value"])
    end = region[-1]["end"]
    if end < first_lyric - EPS:
        push(region, end, first_lyric, "X")

    new_mouth = region + [dict(c) for c in mouth[k:]]

    # 全表单调校验
    prev = -1.0
    for i, c in enumerate(new_mouth):
        if c["start"] < prev - 1e-3:
            sys.exit(f"error: 合并后时间轴倒序 [{i}] {c['start']} < {prev}")
        prev = c["end"]

    meta = dict(song.get("metadata") or {})
    meta["chars"] = int(meta.get("chars") or 0) + chars_speech
    return {"metadata": meta, "mouthCues": new_mouth, "chars": song.get("chars", [])}


def mix_audio(base: Path, speech: Path, offset: float, gain_db: float, fade_ms: float,
              out_audio: Path, out_align: Path) -> None:
    x, sr = sf.read(base)
    s, ssr = sf.read(speech)
    if s.ndim == 1:
        s = s[:, None]
    s = s.mean(axis=1)                       # 保守：口白取单声道再广播回立体声
    s = resample(s, ssr, sr).astype(np.float32)
    s = s * (10 ** (gain_db / 20.0))
    fade = min(len(s), int(sr * fade_ms / 1000.0))
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        s[:fade] *= ramp
        s[-fade:] *= ramp
    i0 = int(round(offset * sr))
    i1 = min(len(x), i0 + len(s))
    if i0 + len(s) > len(x):
        sys.exit(f"error: 口白越出主轨（offset {offset:.2f}s + {len(s) / sr:.2f}s > {len(x) / sr:.2f}s）")
    if x.ndim == 1:
        x = x[:, None]
    x[i0:i1] = np.clip(np.asarray(x[i0:i1], dtype=np.float32)
                        + np.broadcast_to(s[: i1 - i0, None], x[i0:i1].shape), -1.0, 1.0)
    mono = resample(x.mean(axis=1), sr, 16000)
    sf.write(out_audio, x, sr, subtype="PCM_16", format="WAV")
    sf.write(out_align, mono, 16000, subtype="PCM_16", format="WAV")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="口白文本文件（txt）")
    ap.add_argument("--song-cues", required=True, help="全曲 cues.json")
    ap.add_argument("--base-audio", required=True, help="44.1k 立体声主轨（audio.wav）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tts", default="gpt-ft", choices=["gpt-ft", "gpt-zs", "voxcpm2"])
    ap.add_argument("--ref-audio", required=True, help="参考音频（角色音色，3~10s 内）")
    ap.add_argument("--ref-text", default=None, help="参考音频转写（缺省从 girlish.list 反查）")
    ap.add_argument("--first-lyric", type=float, default=None,
                    help="首句时间（秒），缺省从 song-cues 第一条非 X 口型推出")
    ap.add_argument("--offset", type=float, default=1.5, help="口白起点（秒）")
    ap.add_argument("--margin", type=float, default=0.6, help="口白尾到首句的安全间距（秒）")
    ap.add_argument("--speech-gain-db", type=float, default=4.0, help="口白相对主轨的增益（dB）")
    ap.add_argument("--fade-ms", type=float, default=50.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="重新生成口白（默认复用已有 speech.wav）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_lyric = args.first_lyric
    if first_lyric is None:
        song = json.loads(Path(args.song_cues).read_text(encoding="utf-8"))
        first_lyric = next(
            (c["start"] for c in song["mouthCues"] if c["value"] != "X"), None)
        if first_lyric is None:
            sys.exit("error: --first-lyric 未给且 cues 里找不到第一条非 X 口型")
        print(f"首句由 cues 推出: {first_lyric:.2f}s")

    # 1) 生成口白
    speech, ssr = generate_speech(args, out_dir)
    _sinfo = sf.info(speech)
    dur = _sinfo.frames / _sinfo.samplerate
    print(f"\n口白: {Path(args.text).name}  [{len(Path(args.text).read_text(encoding='utf-8').strip())} 字]")
    print(f"     {args.tts} -> {speech.name}  ({dur:.2f}s @ {ssr}Hz)")

    # 2) 时长校验（核心约束：口白必须短于前奏、不压唱段）
    if dur > first_lyric - EPS:
        sys.exit(f"error: 口白 {dur:.2f}s 已超过前奏 {first_lyric:.2f}s，请缩短文本")
    end = args.offset + dur
    buffer_ = first_lyric - end
    if buffer_ < args.margin - EPS:
        sys.exit(
            f"error: 口白越界 offset={args.offset:.2f} + dur={dur:.2f} = {end:.2f}s，"
            f"距首句只剩 {buffer_:.2f}s < 要求 margin {args.margin:.2f}s\n"
            f"  请缩短文本 或 调 --offset/--margin")
    print(f"     offset={args.offset:.2f}s  end={end:.2f}s  first_lyric={first_lyric:.2f}s  "
          f"buffer={buffer_:.2f}s (margin {args.margin:.2f}s)  [OK]")

    # 3) 16k + 说话型口型
    speech16 = out_dir / "speech_16k.wav"
    _d, _sr = sf.read(speech)
    if _d.ndim > 1:
        _d = _d.mean(axis=1)
    sf.write(speech16, resample(_d, _sr, 16000), 16000, subtype="PCM_16", format="WAV")
    speech_cues_path = out_dir / "speech_cues.json"
    sc = build_speech_cues(speech16, Path(args.text), speech_cues_path, args.device)
    check_speech_cues(sc)
    n_sc = len(sc["mouthCues"])
    print(f"     口型: {sc['metadata']['chars']} 字 -> {n_sc} cues（说话型 --melodic-accent 0）")

    # 4) 合并 cues
    song = json.loads(Path(args.song_cues).read_text(encoding="utf-8"))
    merged = merge_cues(song, sc, args.offset, first_lyric,
                        int(sc["metadata"].get("chars") or 0))
    merged["metadata"]["soundFile"] = str((out_dir / "align_intro.wav").resolve())
    meta = merged["metadata"]
    meta["extra"] = meta.get("extra") or {}
    meta["extra"]["introSpeech"] = {
        "text": Path(args.text).read_text(encoding="utf-8").strip(),
        "txt": str(Path(args.text).resolve()),
        "tts": args.tts,
        "offset": round(args.offset, 3),
        "dur": round(dur, 3),
        "firstLyric": round(first_lyric, 3),
        "margin": args.margin,
        "speechGainDb": args.speech_gain_db,
        "speechWav": str(speech.resolve()),
    }
    cues_out = out_dir / "cues_intro.json"
    cues_out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"     合并: {len(song['mouthCues'])} -> {len(merged['mouthCues'])} cues -> {cues_out.name}")

    # 5) 混音 + align
    audio_out = out_dir / "audio_intro.wav"
    align_out = out_dir / "align_intro.wav"
    mix_audio(Path(args.base_audio), speech, args.offset, args.speech_gain_db,
              args.fade_ms, audio_out, align_out)
    print(f"     混音: {audio_out.name} ({sf.info(audio_out).frames / sf.info(audio_out).samplerate:.2f}s) + "
          f"{align_out.name}")

    meta_json = {
        "params": vars(args),
        "speech": {"wav": str(speech), "sr": ssr, "dur": round(dur, 3)},
        "speech16": str(speech16),
        "checks": {
            "dur_budget": f"{dur:.2f} <= {first_lyric - args.margin - args.offset:.2f}",
            "buffer_sec": round(buffer_, 3),
            "speech_cues": n_sc,
        },
        "outputs": {k: str((out_dir / v).resolve()) for k, v in {
            "cues_intro": "cues_intro.json", "audio_intro": "audio_intro.wav",
            "align_intro": "align_intro.wav", "speech_cues": "speech_cues.json"}.items()},
    }
    (out_dir / "intro_meta.json").write_text(
        json.dumps(meta_json, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n完成 -> {out_dir}\n  cues={cues_out}  audio={audio_out}  "
          f"align={align_out}  meta={out_dir / 'intro_meta.json'}")


if __name__ == "__main__":
    main()
