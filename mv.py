#!/usr/bin/env python3
"""MV 渲染流水线 CLI：歌曲干声/混音 -> MV 视频（可选前奏口白 + KTV 字幕）。

仿 sing.py 的分阶段流水线。默认基于一首歌跑全流程；产物与中间素材一律放在
data/songs/<歌名>/mv/ 下（见 README「素材目录约定」）。某阶段产物齐全且不比输入旧时
自动复用已有素材（--force 强制重跑）。

阶段（--stages all 的默认顺序）:
    slice    从 sing.py 产物切出 mv/align.wav（16k mono，口型用）与 mv/audio.wav
             （44.1k stereo，渲染 mux 用）；两个文件同起点同时长，切完核时长/声道
    lyrics   缓存歌曲同名增强 LRC（get_lyrics.py；只联网一次，之后全走本地 .lrc）
    cues     make_cues.py 对干声做口型时间轴（--lrc 行级锚点，无 ASR、不需要 GPU）
    beats    beat_grid.py 全曲节拍网格（渲染 --sway 摇摆用）
    visemes  一次性口型 sprite（split_visemes.py -> data/pics/visemes/）
    intro    （可选）前奏口白：intro_speech.py 产出 speech.wav + speech_cues.json
             + cues_intro.json + audio_intro.wav + align_intro.wav
             （需 mv/intro.txt 与 --tts-ref；--with-intro 或 --stages ...,intro 开启；
             --tts-speed 调语速，<1 慢；改文本/语速后须 --force 重跑）
    render   a25rig/render.sh 渲片（有 intro 时用 cues_intro/audio_intro，无则用基础版）
    subs     make_ass.py 生成 ASS + ffmpeg 烧 KTV 字幕（--no-subs 关闭）

最简用法（一首歌全流程，无口白）:
    python3 mv.py --song 情歌
带前奏口白:
    python3 mv.py --song 情歌 --with-intro --tts-ref data/voices/girlish_sliced/xxx.wav \
        --tts-speed 0.9
只重渲 / 只烧字幕（复用已有素材）:
    python3 mv.py --song 情歌 --stages render
    python3 mv.py --song 情歌 --stages subs
    python3 mv.py --song 情歌 --with-intro --stages render,subs   # 口白版成片（可省 --tts-ref）
强制重跑某阶段:
    python3 mv.py --song 情歌 --stages cues --force

兼容旧入口: ./render.sh（等价 python3 mv.py --song "${SONG:-情歌}"），SONG/SUBS 环境变量仍有效。

日志: data/songs/<歌名>/mv.log；-v 打印子进程完整输出。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("mv")

# 阶段顺序即 --stages all 的执行顺序。依赖：cues 需要 slice 的 align.wav 与 lyrics 的
# .lrc；intro 需要 cues.json 与 audio.wav；render 需要 cues(_intro)/audio(_intro)/beats/
# visemes；subs 需要渲染完的视频与 .lrc。
ALL_STAGES = ["slice", "lyrics", "cues", "beats", "visemes", "intro", "render", "subs"]
# 可选阶段：intro 需要额外输入（mv/intro.txt + --tts-ref），默认不跑。
# subs 默认开（--no-subs 关闭），其余阶段默认全跑。
OPT_IN_STAGES = {"intro"}

SONGS_DIR = os.path.join(ROOT, "data", "songs")
GSV = os.path.join(ROOT, "GPT-SoVITS", ".venv", "bin", "python")
MAKE_CUES = os.path.join(ROOT, "listen_lipsync", "make_cues.py")
SPLIT_VISEMES = os.path.join(ROOT, "listen_lipsync", "split_visemes.py")
GET_LYRICS = os.path.join(ROOT, "listen_lipsync", "get_lyrics.py")
MAKE_ASS = os.path.join(ROOT, "listen_lipsync", "make_ass.py")
INTRO_SPEECH = os.path.join(ROOT, "listen_lipsync", "intro_speech.py")
BEAT_GRID = os.path.join(ROOT, "singing", "beat_grid.py")

PICS = os.path.join(ROOT, "data", "pics")
FIGURE_PSD = os.path.join(PICS, "figure.psd")
MOUTH_9 = os.path.join(PICS, "mouth_9.png")
VISEMES_DIR = os.path.join(PICS, "visemes")
RAIN_GLASS = os.path.join(PICS, "love_song.jpg")
A25_RENDER = os.path.join(ROOT, "a25rig", "render.sh")
FONTS_CONF = os.path.join(ROOT, ".tmp", "fontconf", "fonts.conf")

# 渲染必须显式点名 NVIDIA 适配器 + ANGLE/GL，否则选到 AMD 核显或回落 SwiftShader
RENDER_ENV = {
    "MESA_LOADER_DRIVER_OVERRIDE": "d3d12",
    "GALLIUM_DRIVER": "d3d12",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
    "A25_GL_ARGS": "--use-gl=angle --use-angle=gl --ignore-gpu-blocklist",
}


def setup_logging(logfile: str, verbose: bool) -> None:
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)


def banner(text: str) -> None:
    log.info("")
    log.info("=" * 70)
    log.info(f"  {text}")
    log.info("=" * 70)


class Fail(RuntimeError):
    pass


def run(cmd: list[str], stage: str = "", cwd: str | None = None, env: dict | None = None,
        timeout: int = 7200) -> str:
    """跑子进程，逐行转发到日志（DEBUG 级，-v 可见）；失败抛 Fail；返回完整 stdout。"""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
        for k, v in sorted(env.items()):
            log.debug(f"    env {k}={v}")
    log.debug(f"    cmd {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        log.debug(f"    | {line}")
        lines.append(line)
    proc.wait(timeout=timeout)
    dt = time.time() - t0
    if proc.returncode != 0:
        log.error(f"  [{stage}] 子进程退出码 {proc.returncode}，最后 20 行输出：")
        for line in lines[-20:]:
            log.error(f"    | {line}")
        raise Fail(f"{stage} 失败（exit {proc.returncode}）")
    log.debug(f"    done in {dt:.1f}s")
    return "\n".join(lines)


def probe(path: str, entries: str, of: str = "csv=p=0") -> str:
    """ffprobe: entries 形如 format=duration 或 stream=width,height。"""
    out = run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", of, path],
              stage="probe")
    return out.strip()


def duration(path: str) -> float:
    return float(probe(path, "format=duration"))


# ---------------------------------------------------------------- Cfg

@dataclass
class Cfg:
    name: str                 # 歌曲名（决定 data/songs/<name>/ 与输出命名）
    song: str                 # 歌曲音频文件路径（beats/lyrics 的输入）
    stages: list[str]         # 本次要跑的阶段（有序）
    vocal: str = ""           # 干声路径；空 = 自动探测 vocal_psvs -> vocal -> cascade
    mix: str = ""             # 混音路径；空 = data/songs/<name>/mix.wav
    start: float = 0.0        # 切片窗口起点（秒）；0 = 从开头
    end: float = 0.0          # 切片窗口终点（秒）；0 = 全长
    force: bool = False       # 强制重跑（无视产物新旧）
    gpu: str = "0"
    # --- intro（前奏口白）---
    with_intro: bool = False
    tts: str = "gpt-ft"
    tts_ref: str = ""
    tts_ref_text: str = ""
    tts_speed: float = 1.0   # 口白语速倍率（GPT-SoVITS 原生，<1 = 更慢，范围 0.6-1.65）
    first_lyric: float | None = None
    offset: float = 1.5
    margin: float = 0.6
    speech_gain_db: float = 4.0
    # --- render ---
    psd: str = FIGURE_PSD
    visemes_dir: str = VISEMES_DIR
    bg: str = ""              # 非空 → 普通背景模式（与 --rain-glass 互斥）
    rain_glass: str = RAIN_GLASS
    streaks: bool = True
    char_scale: float = 1.15
    char_x: float = 0.62
    char_y: float = 0.56
    rest_form: float = 0.35
    fps: int = 30
    size: str = "16:9"
    render_seed: int = 1
    # --- subs ---
    subs: bool = True
    ass: str = ""             # 空 = mv/lyrics.ass
    fc: str = FONTS_CONF
    font: str = "SimHei"

    # --- 路径 ---
    @property
    def out_dir(self) -> str:
        return os.path.join(SONGS_DIR, self.name)

    @property
    def mv_dir(self) -> str:
        return os.path.join(self.out_dir, "mv")

    def p_mv(self, *parts: str) -> str:
        return os.path.join(self.mv_dir, *parts)

    @property
    def lrc(self) -> str:
        return os.path.join(SONGS_DIR, self.name + ".lrc")

    @property
    def lyrics_lrc(self) -> str:
        """cues/字幕用的歌词：优先手工重锚过的 mv/lyrics.lrc（README「删段后重锚」：
        人工唱段删改后行起点要挪到洞后首个发声帧，改好的版本存 mv/lyrics.lrc），
        否则用 get_lyrics.py 缓存的同名 .lrc。"""
        manual = self.p_mv("lyrics.lrc")
        return manual if os.path.exists(manual) else self.lrc

    @property
    def log_file(self) -> str:
        return os.path.join(self.out_dir, "mv.log")

    @property
    def device(self) -> str:
        return f"cuda:{self.gpu}"

    # 切片产物（基础版，intro 在它们之上再混口白）
    @property
    def align_wav(self) -> str:
        return self.p_mv("align.wav")

    @property
    def audio_wav(self) -> str:
        return self.p_mv("audio.wav")

    @property
    def cues_json(self) -> str:
        return self.p_mv("cues.json")

    @property
    def beats_json(self) -> str:
        return self.p_mv("beats.json")

    # 出片用哪一套：有 intro 就用口白版命名（_intro），否则基础版
    @property
    def intro_on(self) -> bool:
        return self.with_intro

    @property
    def video_cues(self) -> str:
        return self.p_mv("cues_intro.json") if self.intro_on else self.cues_json

    @property
    def video_audio(self) -> str:
        return self.p_mv("audio_intro.wav") if self.intro_on else self.audio_wav

    @property
    def video_path(self) -> str:
        return self.p_mv(f"{self.name}_intro.mp4" if self.intro_on else f"{self.name}.mp4")

    @property
    def subvideo_path(self) -> str:
        return self.p_mv(f"{self.name}_intro_lyrics.mp4" if self.intro_on
                         else f"{self.name}_lyrics.mp4")

    @property
    def ass_path(self) -> str:
        return self.ass or self.p_mv("lyrics.ass")


def stale(cfg: Cfg, outputs: list[str], inputs: list[str],
          meta_ok=None, note: str = "") -> bool:
    """True = 需要重跑；False = 产物齐全且不比输入旧，复用。

    meta_ok：附加参数校验（如切片窗口），不匹配也算 stale。
    """
    if cfg.force:
        log.info(f"  --force 强制重跑{note}")
        return True
    missing = [o for o in outputs if not os.path.exists(o)]
    if missing:
        log.info(f"  缺产物: {', '.join(os.path.basename(m) for m in missing)}")
        return True
    if meta_ok is not None and not meta_ok():
        log.info("  参数已变化，重跑")
        return True
    tin = -1.0
    for i in inputs:
        if os.path.exists(i):
            tin = max(tin, os.path.getmtime(i))
    tout = min(os.path.getmtime(o) for o in outputs)
    if tin > tout - 1.0:
        log.info("  输入比产物新，重跑")
        return True
    return False


# ---------------------------------------------------------------- 阶段

def stage_slice(cfg: Cfg) -> None:
    banner("Stage: 切片 (align 16k mono + audio 44.1k stereo)")
    outputs = [cfg.align_wav, cfg.audio_wav]
    inputs = [cfg.vocal, cfg.mix]

    def meta_ok() -> bool:
        mp = cfg.p_mv("slice_meta.json")
        if not os.path.exists(mp):
            return False
        try:
            return json.load(open(mp, encoding="utf-8")) == {
                "vocal": cfg.vocal, "mix": cfg.mix, "start": cfg.start, "end": cfg.end,
            }
        except Exception:
            return False

    if not stale(cfg, outputs, inputs, meta_ok, note="（slice）"):
        log.info(f"  复用已有: {cfg.align_wav} / {cfg.audio_wav}")
        return
    os.makedirs(cfg.mv_dir, exist_ok=True)

    win = cfg.end - cfg.start if cfg.end > 0 else 0.0
    common = ["-v", "error", "-y"]
    if cfg.start > 0:
        common += ["-ss", f"{cfg.start:g}"]
    if win > 0:
        common += ["-t", f"{win:g}"]
    run([*["ffmpeg", *common], "-i", cfg.vocal, "-ar", "16000", "-ac", "1", cfg.align_wav],
        stage="slice")
    run([*["ffmpeg", *common], "-i", cfg.mix, cfg.audio_wav], stage="slice")

    # 核对：README 要求 16k mono / 44.1k stereo / 两文件同时长
    sa, ca = probe(cfg.align_wav, "stream=sample_rate,channels").split(",")
    sb, cb = probe(cfg.audio_wav, "stream=sample_rate,channels").split(",")
    da, db = duration(cfg.align_wav), duration(cfg.audio_wav)
    log.info(f"  align: {da:.2f}s {sa}Hz/{ca}ch    audio: {db:.2f}s {sb}Hz/{cb}ch")
    if abs(float(da) - float(db)) > 0.1:
        log.warning(f"  ⚠ align/audio 时长差 {abs(float(da)-float(db)):.2f}s，口型可能整体偏移")
    if sa != "16000" or ca != "1":
        log.warning(f"  ⚠ align 应为 16k mono，实际 {sa}Hz/{ca}ch")
    if sb != "44100":
        log.warning(f"  ⚠ audio 应为 44.1k，实际 {sb}Hz（--mix-mono 混音是 1ch 属正常）")

    with open(cfg.p_mv("slice_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocal": cfg.vocal, "mix": cfg.mix, "start": cfg.start, "end": cfg.end},
                  f, ensure_ascii=False, indent=1)
    log.info(f"  -> {cfg.align_wav} / {cfg.audio_wav}")


def stage_lyrics(cfg: Cfg) -> None:
    banner("Stage: 歌词缓存 (get_lyrics.py -> .lrc)")
    if not stale(cfg, [cfg.lrc], [cfg.song], note="（lyrics）"):
        log.info(f"  复用已有: {cfg.lrc}")
        return
    cmd = [GSV, GET_LYRICS, cfg.song]
    if cfg.force:
        cmd += ["--force"]
    run(cmd, stage="lyrics")
    if not os.path.exists(cfg.lrc):
        raise Fail(f"get_lyrics 未产生 {cfg.lrc}（可能抓取失败，见上方输出）")
    log.info(f"  -> {cfg.lrc}")


def stage_cues(cfg: Cfg) -> None:
    banner("Stage: 口型时间轴 (make_cues.py --lrc)")
    if not os.path.exists(cfg.align_wav):
        raise Fail(f"缺少 {cfg.align_wav}（先跑 slice 阶段）")
    lrc = cfg.lyrics_lrc
    if not os.path.exists(lrc):
        raise Fail(f"缺少歌词 {lrc}（先跑 lyrics 阶段；也可以删掉它让 cues 走 ASR，"
                   f"但不推荐）")
    if not stale(cfg, [cfg.cues_json], [cfg.align_wav, lrc], note="（cues）"):
        log.info(f"  复用已有: {cfg.cues_json}")
        return
    run([GSV, MAKE_CUES, "--wav", cfg.align_wav, "--lrc", lrc,
         "--out", cfg.cues_json, "--gate-db", "-40", "--device", cfg.device],
        stage="cues")
    if not os.path.exists(cfg.cues_json):
        raise Fail(f"make_cues 未产生 {cfg.cues_json}")
    log.info(f"  -> {cfg.cues_json}（来自 {os.path.basename(lrc)}）")


def stage_beats(cfg: Cfg) -> None:
    banner("Stage: 节拍网格 (beat_grid.py)")
    if not stale(cfg, [cfg.beats_json], [cfg.song], note="（beats）"):
        log.info(f"  复用已有: {cfg.beats_json}")
        return
    run([GSV, BEAT_GRID, cfg.song, "-o", cfg.beats_json, "--device", "cuda"],
        stage="beats", timeout=1800)
    if not os.path.exists(cfg.beats_json):
        raise Fail(f"beat_grid 未产生 {cfg.beats_json}")
    log.info(f"  -> {cfg.beats_json}")


def stage_visemes(cfg: Cfg) -> None:
    banner("Stage: 口型 sprite (split_visemes.py, 一次性)")
    need = [os.path.join(cfg.visemes_dir, v) for v in ("X.png", "A.png", "B.png", "C.png",
                                                       "D.png", "E.png", "F.png", "G.png",
                                                       "H.png", "meta.json")]
    if not stale(cfg, need, [MOUTH_9], note="（visemes）"):
        log.info(f"  复用已有: {cfg.visemes_dir}/")
        return
    if not os.path.exists(MOUTH_9):
        raise Fail(f"缺九宫格素材 {MOUTH_9}，先准备立绘口型图")
    run([GSV, SPLIT_VISEMES, MOUTH_9, "-o", cfg.visemes_dir], stage="visemes")
    missing = [v for v in need if not os.path.exists(v)]
    if missing:
        raise Fail(f"split_visemes 未产全 sprite: {missing}")
    log.info(f"  -> {cfg.visemes_dir}/（X/A/B/C/D/E/F/G/H + meta.json）")


def stage_intro(cfg: Cfg) -> None:
    banner("Stage: 前奏口白 (intro_speech.py)")
    intro_txt = cfg.p_mv("intro.txt")
    for label, p in (("口白文本", intro_txt), ("song cues", cfg.cues_json),
                     ("主轨 audio", cfg.audio_wav), ("参考音频 --tts-ref", cfg.tts_ref)):
        if not os.path.exists(p):
            raise Fail(f"缺少{label}: {p}"
                       + ("（先写 mv/intro.txt，并给 --tts-ref）" if label == "口白文本" else ""))
    outputs = [cfg.p_mv("speech.wav"), cfg.p_mv("speech_cues.json"),
               cfg.p_mv("cues_intro.json"), cfg.p_mv("audio_intro.wav"),
               cfg.p_mv("align_intro.wav"), cfg.p_mv("intro_meta.json")]
    inputs = [intro_txt, cfg.cues_json, cfg.audio_wav, cfg.tts_ref]
    if not stale(cfg, outputs, inputs, note="（intro，--force 才重跑）"):
        log.info(f"  复用已有口白素材（speech.wav / cues_intro.json / audio_intro.wav）")
        return
    cmd = [GSV, INTRO_SPEECH, "--text", intro_txt,
           "--song-cues", cfg.cues_json,
           "--base-audio", cfg.audio_wav,
           "--out-dir", cfg.mv_dir,
           "--tts", cfg.tts,
           "--speed", f"{cfg.tts_speed:g}",
           "--ref-audio", cfg.tts_ref,
           "--offset", f"{cfg.offset:g}",
           "--margin", f"{cfg.margin:g}",
           "--speech-gain-db", f"{cfg.speech_gain_db:g}",
           "--device", cfg.device]
    if cfg.first_lyric is not None:
        cmd += ["--first-lyric", f"{cfg.first_lyric:g}"]
    if cfg.tts_ref_text:
        cmd += ["--ref-text", cfg.tts_ref_text]
    if cfg.force:
        cmd += ["--force"]
    run(cmd, stage="intro", timeout=3600)
    for p in outputs:
        if not os.path.exists(p):
            raise Fail(f"intro_speech 未产生 {p}")
    log.info(f"  -> cues_intro.json / audio_intro.wav / align_intro.wav / intro_meta.json")


def stage_render(cfg: Cfg) -> None:
    banner("Stage: 渲染 (a25rig/render.sh)")
    for label, p in (("口型 cues", cfg.video_cues), ("主轨 audio", cfg.video_audio),
                     ("节拍", cfg.beats_json)):
        if not os.path.exists(p):
            hint = ("（成片选了 intro 命名但没有 intro 素材：先跑 intro 阶段，"
                    "或去掉 --with-intro 用基础版 cues.json/audio.wav）"
                    if p.endswith("_intro.json") or p.endswith("_intro.wav") else "")
            raise Fail(f"缺少{label}: {p}{hint}")
    if not stale(cfg, [cfg.video_path], [cfg.video_cues, cfg.video_audio, cfg.beats_json],
                 note="（render）"):
        log.info(f"  复用已有成片: {cfg.video_path}")
        return
    for label, p in (("立绘 PSD", cfg.psd), ("口型 sprite 目录", cfg.visemes_dir),
                     ("雨水玻璃图", cfg.rain_glass)):
        if not os.path.exists(p):
            raise Fail(f"缺渲染素材{label}: {p}")
    if not os.path.exists(A25_RENDER):
        raise Fail(f"缺渲染器 {A25_RENDER}")
    for sub in ("driver", "driver/node_modules", "driver/nsslib"):
        if not os.path.exists(os.path.join(ROOT, "a25rig", sub)):
            raise Fail(f"缺 a25rig/{sub}（driver 依赖未装？见 a25rig/render.sh 提示）")

    os.makedirs(cfg.mv_dir, exist_ok=True)
    cmd = ["setsid", A25_RENDER,
           "--psd", cfg.psd,
           "--cues", cfg.video_cues,
           "--sway", cfg.beats_json,
           "--audio", cfg.video_audio,
           "--visemes", cfg.visemes_dir,
           "--video", cfg.video_path,
           "--char-scale", f"{cfg.char_scale:g}",
           "--char-x", f"{cfg.char_x:g}",
           "--char-y", f"{cfg.char_y:g}",
           "--rest-form", f"{cfg.rest_form:g}",
           "--fps", str(cfg.fps),
           "--size", cfg.size,
           "--seed", str(cfg.render_seed)]
    if cfg.bg:
        cmd += ["--bg", cfg.bg]
    else:
        cmd += ["--rain-glass", cfg.rain_glass]
        if cfg.streaks:
            cmd += ["--streaks"]
    log.info(f"  素材: cues={os.path.basename(cfg.video_cues)}  audio="
             f"{os.path.basename(cfg.video_audio)}  beats={os.path.basename(cfg.beats_json)}")
    run(cmd, stage="render", env=RENDER_ENV, timeout=7200)
    if not os.path.exists(cfg.video_path):
        raise Fail(f"渲染未产生成片: {cfg.video_path}")
    log.info(f"  -> {cfg.video_path}")


def stage_subs(cfg: Cfg) -> None:
    banner("Stage: 烧字幕 (make_ass.py + ffmpeg)")
    if not os.path.exists(cfg.video_path):
        raise Fail(f"缺少成片 {cfg.video_path}（先跑 render 阶段，或用 SUBS=0 / --no-subs 关闭烧字）")
    dur = duration(cfg.video_path)
    wh = probe(cfg.video_path, "stream=width,height", of="csv=s=x:p=0")

    if not stale(cfg, [cfg.ass_path], [cfg.lyrics_lrc, cfg.video_path], note="（ass）"):
        log.info(f"  复用已有: {cfg.ass_path}")
    else:
        if not os.path.exists(cfg.lyrics_lrc):
            raise Fail(f"缺歌词 {cfg.lyrics_lrc}（先跑 lyrics 阶段缓存歌词）")
        log.info(f"  生成 ASS: {os.path.basename(cfg.lyrics_lrc)} -> {os.path.basename(cfg.ass_path)} "
                 f"(0..{dur:.0f}s @ {wh})")
        run([GSV, MAKE_ASS, cfg.lyrics_lrc, "-o", cfg.ass_path, "--size", wh,
             "--start", "0", "--end", f"{dur:g}", "--pos", "bc", "--font", cfg.font],
            stage="subs")
        if not os.path.exists(cfg.ass_path):
            raise Fail(f"make_ass 未产生 {cfg.ass_path}")

    if not os.path.exists(cfg.fc):
        raise Fail(f"缺字体配置 {cfg.fc}（fonts.conf 必须放工作区，见 README §中文字体）")
    fenv = dict(os.environ, FONTCONFIG_FILE=cfg.fc)
    fr = subprocess.run(["fc-match", cfg.font], capture_output=True, text=True, env=fenv)
    if fr.returncode != 0 or not fr.stdout.strip():
        raise Fail(f"fontconfig 校验失败：FONTCONFIG_FILE={cfg.fc} 下 fc-match {cfg.font} 无结果"
                   f"（否则烧出的字幕会静默消失，见 README §中文字体 的陷阱）")

    log.info(f"  烧字幕: {os.path.basename(cfg.video_path)} -> "
             f"{os.path.basename(cfg.subvideo_path)} (crf18 h264 + aac copy)")
    run(["ffmpeg", "-v", "error", "-y", "-i", cfg.video_path,
         "-vf", f"ass={cfg.ass_path}",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "copy",
         cfg.subvideo_path], stage="subs", timeout=7200, env=fenv)
    ds = duration(cfg.subvideo_path)
    log.info(f"  -> {cfg.subvideo_path}  ({ds:.2f}s, 源 {dur:.2f}s)")


STAGE_FN = {
    "slice": stage_slice,
    "lyrics": stage_lyrics,
    "cues": stage_cues,
    "beats": stage_beats,
    "visemes": stage_visemes,
    "intro": stage_intro,
    "render": stage_render,
    "subs": stage_subs,
}


# ---------------------------------------------------------------- 主流程

def resolve_name_and_song(v: str) -> tuple[str, str | None]:
    """--song 接受歌曲名（自动探测 .flac/.mp3/.wav/.m4a）或歌曲文件路径。"""
    if os.path.sep in v or v.lower().endswith((".flac", ".mp3", ".wav", ".m4a")):
        name = os.path.splitext(os.path.basename(v))[0]
        return name, v
    return v, None


def find_song(name: str) -> str | None:
    for ext in (".flac", ".mp3", ".wav", ".m4a"):
        p = os.path.join(SONGS_DIR, name + ext)
        if os.path.exists(p):
            return p
    return None


def find_vocal(name: str) -> str | None:
    for v in ("vocal_psvs.wav", "vocal.wav", "vocal_cascade.wav"):
        p = os.path.join(SONGS_DIR, name, v)
        if os.path.exists(p):
            return p
    return None


def preflight(cfg: Cfg, stages: list[str]) -> None:
    banner("预检")
    if not os.path.exists(GSV):
        raise Fail(f"venv python 不存在: {GSV}（GPT-SoVITS 未建 .venv？）")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise Fail(f"未找到 {tool}")
    need_song = any(s in ("beats", "lyrics") for s in stages)
    if need_song and not os.path.exists(cfg.song):
        raise Fail(f"歌曲不存在: {cfg.song}\n"
                   f"  先跑 sing.py 出干声/混音（或检查 {cfg.name} 的产物）")
    if need_song:
        log.info(f"  歌曲    : {cfg.song}")
    if "slice" in stages:
        for label, p in (("干声 vocal", cfg.vocal), ("混音 mix", cfg.mix)):
            if not os.path.exists(p):
                raise Fail(f"缺少{label}: {p}\n"
                           f"  先跑 sing.py --name {cfg.name}（含 mix 阶段产出 mix.wav）；"
                           f"干声可 --vocal 指定")
    if "subs" in stages and not shutil.which("fc-match"):
        raise Fail("未找到 fc-match（烧字幕需要 fontconfig）")
    os.makedirs(cfg.mv_dir, exist_ok=True)
    log.info(f"  输出目录: {cfg.mv_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MV 渲染流水线：歌曲干声/混音 -> MV 视频（可选前奏口白 + KTV 字幕）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--song", required=True,
                    help="歌曲（名或文件路径）。按名时自动探测 data/songs/<song>.flac|mp3|wav，"
                         "输出/素材在 data/songs/<song>/mv/ 下")
    ap.add_argument("--vocal", default="",
                    help="干声路径（口型用）。默认自动探测 vocal_psvs.wav -> vocal.wav -> "
                         "vocal_cascade.wav")
    ap.add_argument("--mix", default="",
                    help="混音路径（渲染 mux 用）。默认 data/songs/<name>/mix.wav")
    ap.add_argument("--start", type=float, default=0.0, help="切片窗口起点秒（默认 0=开头）")
    ap.add_argument("--end", type=float, default=0.0, help="切片窗口终点秒（默认 0=全长）")
    ap.add_argument("--stages", default="all",
                    help=f"要执行的阶段，逗号分隔。可选: {','.join(ALL_STAGES)}。"
                         f"默认不跑: intro（需 --with-intro）。显式列出则一定跑")
    ap.add_argument("--with-intro", action="store_true",
                    help="前奏口白：自动带出 intro 阶段，渲片用 cues_intro.json / "
                         "audio_intro.wav，成片命名 <song>_intro.mp4。"
                         "需要 mv/intro.txt 与 --tts-ref")
    ap.add_argument("--tts", default="gpt-ft", choices=["gpt-ft", "gpt-zs", "voxcpm2"],
                    help="口白 TTS 模型（默认 gpt-ft = girlish GPT e30 + talksing_s2mix50 "
                         "SoVITS e12，吐字与高频清晰度最优；权重在 intro_speech.py 定义）")
    ap.add_argument("--tts-speed", dest="tts_speed", type=float, default=1.0,
                    help="口白语速倍率（默认 1.0；GPT-SoVITS 原生走模型时长缩放、音色不变，"
                         "0.6-1.65，<1 = 更慢。gpt-voxcpm2 不生效）")
    ap.add_argument("--tts-ref", default="",
                    help="口白参考音频（角色音色，3~10s）")
    ap.add_argument("--tts-ref-text", default="",
                    help="参考音频转写（缺省从 girlish.list 反查）")
    ap.add_argument("--first-lyric", type=float, default=None,
                    help="首句时间秒（缺省从 cues.json 第一条非 X 口型推出）")
    ap.add_argument("--offset", type=float, default=1.5, help="口白起点秒（默认 1.5）")
    ap.add_argument("--margin", type=float, default=0.6, help="口白尾到首句的安全间距秒")
    ap.add_argument("--speech-gain-db", type=float, default=4.0, help="口白相对主轨增益 dB")
    ap.add_argument("--no-subs", dest="subs", action="store_false", default=True,
                    help="不烧 KTV 字幕（等价 SUBS=0）")
    ap.add_argument("--ass", default="", help="字幕 ASS 路径（默认 mv/lyrics.ass）")
    ap.add_argument("--font", default="SimHei", help="字幕字体（须 fontconfig 可见）")
    ap.add_argument("--fc", default=FONTS_CONF, help="fontconfig 配置文件（默认 .tmp/fontconf/fonts.conf）")
    ap.add_argument("--psd", default=FIGURE_PSD, help="立绘 PSD（默认 data/pics/figure.psd）")
    ap.add_argument("--bg", default="",
                    help="普通背景（图片/CSS），与默认雨天风格互斥（传了就不再 --rain-glass/--streaks）")
    ap.add_argument("--rain-glass", default=RAIN_GLASS, help="雨水玻璃背景图（默认 data/pics/love_song.jpg）")
    ap.add_argument("--no-streaks", dest="streaks", action="store_false", default=True,
                    help="雨天风格但不加落雨丝")
    ap.add_argument("--char-scale", type=float, default=1.15)
    ap.add_argument("--char-x", type=float, default=0.62)
    ap.add_argument("--char-y", type=float, default=0.56)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="16:9", help="16:9 或 WxH")
    ap.add_argument("--render-seed", type=int, default=1, help="渲染随机种子（同种子可复现）")
    ap.add_argument("--force", action="store_true", help="强制重跑（忽略产物新旧）")
    ap.add_argument("--gpu", default="0", help="GPU 编号")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印子进程完整输出")
    args = ap.parse_args()

    name, song = resolve_name_and_song(args.song)
    if song is None:
        song = find_song(name)
        if song is None:
            sys.exit(f"error: 找不到歌曲文件 data/songs/{name}.(flac|mp3|wav|m4a)\n"
                     f"  或把 --song 直接指向文件路径")
    vocal = args.vocal or find_vocal(name) or ""
    mix = args.mix or os.path.join(SONGS_DIR, name, "mix.wav")

    if args.stages == "all":
        stages = [s for s in ALL_STAGES
                  if s not in OPT_IN_STAGES or (s == "intro" and args.with_intro)]
        if not args.subs:
            stages = [s for s in stages if s != "subs"]
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        bad = [s for s in stages if s not in ALL_STAGES]
        if bad:
            sys.exit(f"error: 未知阶段 {bad}，可选: {','.join(ALL_STAGES)}")
        args.with_intro = args.with_intro or ("intro" in stages)
        args.subs = args.subs and ("subs" in stages)

    if args.with_intro and "intro" in stages and not args.tts_ref:
        sys.exit("error: --with-intro 需要 --tts-ref <参考音频>（口白角色音色）\n"
                 "  （仅重渲/重烧字时 --stages render,subs --with-intro 可省略 --tts-ref："
                 "口白素材已存在则复用）")

    cfg = Cfg(name=name, song=song, stages=stages,
              vocal=vocal, mix=mix, start=args.start, end=args.end,
              force=args.force, gpu=args.gpu,
              with_intro=args.with_intro,
              tts=args.tts, tts_ref=args.tts_ref, tts_ref_text=args.tts_ref_text,
              tts_speed=args.tts_speed,
              first_lyric=args.first_lyric, offset=args.offset, margin=args.margin,
              speech_gain_db=args.speech_gain_db,
              psd=args.psd,
              bg=args.bg, rain_glass=args.rain_glass, streaks=args.streaks,
              char_scale=args.char_scale, char_x=args.char_x, char_y=args.char_y,
              fps=args.fps, size=args.size, render_seed=args.render_seed,
              subs=args.subs, ass=args.ass, fc=args.fc, font=args.font)

    setup_logging(cfg.log_file, args.verbose)
    log.info(f"歌曲    : {cfg.name}")
    log.info(f"音频    : {cfg.song}")
    if cfg.vocal:
        log.info(f"干声    : {cfg.vocal}")
    log.info(f"混音    : {cfg.mix}")
    if cfg.with_intro:
        log.info(f"口白    : {cfg.tts}  ref={cfg.tts_ref}")
        log.info(f"成片    : {cfg.video_path}（含前奏口白）")
    log.info(f"阶段    : {' -> '.join(stages)}")
    log.info(f"日志    : {cfg.log_file}")

    t0 = time.time()
    try:
        preflight(cfg, stages)
        for s in stages:
            ts = time.time()
            STAGE_FN[s](cfg)
            log.info(f"  [{s}] 耗时 {time.time() - ts:.1f}s")
        log.info("")
        log.info(f"完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
        if "render" in stages and os.path.exists(cfg.video_path):
            log.info(f"  成片      : {cfg.video_path}")
        if "subs" in stages and os.path.exists(cfg.subvideo_path):
            log.info(f"  成片(字幕): {cfg.subvideo_path}")
        if "render" in stages and "subs" not in stages:
            log.info("  烧字幕    : 已跳过（--no-subs / 未列 subs 阶段）")
    except Fail as e:
        log.error("")
        log.error(f"失败: {e}")
        log.error(f"完整日志: {cfg.log_file}（用 -v 可看子进程全部输出）")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("\n已中断。已完成的阶段产物保留在磁盘，重跑会自动复用")
        sys.exit(130)


if __name__ == "__main__":
    main()