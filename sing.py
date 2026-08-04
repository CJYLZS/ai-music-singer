#!/usr/bin/env python
"""零样本歌声合成 CLI：参考语音 -> 用参考音色唱给定歌曲。

两段式流水线（无单一模型能做"语音参考 -> 唱出目标音色"）:

    song.flac
      ├─ BS-RoFormer          -> 人声(含和声) + 伴奏      (GPT-SoVITS venv)
      │    └─ karaoke 模型    -> 主唱 + 和声
      ├─ 伴奏 -> 4-stem 分离   -> 鼓点+bass 骨架 (sep4/, GPT-SoVITS venv)
      ├─ 主唱 -> RMVPE F0     (SoulX venv)
      ├─ 参考音频(通常带BGM/BGM含人声) 同样两步分离 -> 纯人声 lead
      ├─ 纯人声切片 -> F0 分析 -> 贪心选段 -> prompt.wav (SoulX venv)
      ├─ PC-NSF-HiFiGAN 分层移调补齐高音区 -> prompt_ext.wav (tools/pcnsf venv,
      │   可 --ext-method seedvc 换回 seed-vc 自转换)
      ├─ SoulX-Singer SVC: prompt_ext + 主唱 -> 目标音色人声 (SoulX venv)
      ├─ ERes2NetV2 相似度评测（对照真实语音）  (GPT-SoVITS venv)
      └─ [--mix] 人声 + 伴奏混音 -> 成品         (SoulX venv)

最简用法:
    python3 sing.py --ref data/voices/ref.wav --song data/songs/song.flac --name exp1

先听后段纯人声再混伴奏（默认不含 mix 阶段）:
    python3 sing.py --ref ... --song ... --name exp1
    python3 sing.py --ref ... --song ... --name exp1 --stages mix --acc-gain 0.7

只跑部分阶段（产物在 data/songs/<name>/ 下，可续跑）:
    python3 sing.py --ref ... --song ... --name exp1 --stages sep,select
    python3 sing.py --ref ... --song ... --name exp1 --stages svc

日志写到 data/songs/<name>/sing.log；-v 打印全部子进程输出。
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
log = logging.getLogger("sing")

ALL_STAGES = ["sep", "stems4", "ref-sep", "f0-target", "select", "ext", "svc", "eval", "mix"]
OPT_IN_STAGES = {"mix"}  # 先验收纯人声，再 --mix

VENVS = {
    "gsv": os.path.join(ROOT, "GPT-SoVITS", ".venv", "bin", "python"),
    "soulx": os.path.join(ROOT, "SoulX-Singer", ".venv", "bin", "python"),
    "seedvc": os.path.join(ROOT, "seed-vc", ".venv", "bin", "python"),
}
WORKERS = os.path.join(ROOT, "singing")
SOULX_DIR = os.path.join(ROOT, "SoulX-Singer")
SEEDVC_DIR = os.path.join(ROOT, "seed-vc")

UVR5_WEIGHTS = os.path.join(ROOT, "GPT-SoVITS", "tools", "uvr5", "uvr5_weights")
BS_CKPT = os.path.join(UVR5_WEIGHTS, "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
BS_YAML = os.path.join(UVR5_WEIGHTS, "model_bs_roformer_ep_317_sdr_12.9755.yaml")
KARAOKE_CKPT = os.path.join(UVR5_WEIGHTS, "mel_band_roformer_karaoke_becruily.ckpt")
KARAOKE_YAML = os.path.join(UVR5_WEIGHTS, "mel_band_roformer_karaoke_becruily.yaml")
STEMS4_CKPT = os.path.join(UVR5_WEIGHTS, "bs_roformer_4stems_ft.pth")
STEMS4_YAML = os.path.join(UVR5_WEIGHTS, "bs_roformer_4stems_ft.yaml")
RMVPE = os.path.join(ROOT, "SoulX-Singer", "pretrained_models",
                     "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt")
SVC_MODEL = os.path.join(ROOT, "SoulX-Singer", "pretrained_models", "SoulX-Singer", "model-svc.pt")
SVC_CONFIG = os.path.join(ROOT, "SoulX-Singer", "soulxsinger", "config", "soulxsinger.yaml")

PCNSF_VENV = os.path.join(ROOT, "tools", "pcnsf", ".venv", "bin", "python")
NSF_CONFIG = os.path.join(ROOT, "tools", "pcnsf_hifigan",
                          "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02", "config.json")
NSF_CKPT = os.path.join(ROOT, "tools", "pcnsf_hifigan",
                        "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02", "model.ckpt")


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
        timeout: int = 3600) -> str:
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


@dataclass
class Cfg:
    ref: str
    song: str
    name: str
    gpu: str = "0"
    stages: list[str] = None  # type: ignore[assignment]
    max_prompt: float = 25.0
    semitone_shift: int | None = None
    ext_method: str = "nsf"
    anchor_min_ratio: float = 0.55
    max_seg_sec: float = 28.0
    n_steps: int = 32
    cfg: float = 0.5   # 上游默认 3.0，实测偏高（见 AGENTS.md「cfg 扫点」）
    pitch_shift: int | str = 0
    vocal_gain: float = 1.0
    acc_gain: float = 0.8
    mix: bool = False
    no_sep_cache: bool = False

    @property
    def out_dir(self) -> str:
        return os.path.join(ROOT, "data", "songs", self.name)

    def p(self, *parts: str) -> str:
        return os.path.join(self.out_dir, *parts)

    @property
    def device(self) -> str:
        return f"cuda:{self.gpu}"

    @property
    def lead_wav(self) -> str:
        return self.p("sep2", "lead.wav")

    @property
    def ref_lead(self) -> str:
        return self.p("refsep2", "lead.wav")

    @property
    def ref_clean(self) -> str:
        """选段/评测用的参考：已跑 ref-sep 则用分离后的纯人声 lead，否则退回原音频。"""
        return self.ref_lead if os.path.exists(self.ref_lead) else self.ref

    @property
    def acc_wav(self) -> str:
        return self.p("sep1", "acc.wav")

    @property
    def drum_bass_wav(self) -> str:
        """伴奏中的 鼓点+bass 骨架（stems4 阶段产物，供 --mix 使用）。"""
        return self.p("sep4", "drum_bass.wav")

    @property
    def target_f0(self) -> str:
        return self.p("target_f0.npy")

    @property
    def prompt_wav(self) -> str:
        return self.p("prompt.wav")

    @property
    def prompt_f0(self) -> str:
        return self.p("prompt_f0.npy")

    @property
    def prompt_stats(self) -> str:
        return self.p("prompt_stats.json")

    @property
    def prompt_ext_wav(self) -> str:
        return self.p("prompt_ext.wav")

    @property
    def prompt_ext_f0(self) -> str:
        return self.p("prompt_ext_f0.npy")

    @property
    def vocal_wav(self) -> str:
        return self.p("vocal.wav")

    @property
    def mix_wav(self) -> str:
        return self.p("mix.wav")

    @property
    def log_file(self) -> str:
        return self.p("sing.log")


def preflight(cfg: Cfg) -> None:
    banner("预检")
    if not os.path.exists(cfg.ref):
        raise Fail(f"参考音频不存在: {cfg.ref}")
    if not os.path.exists(cfg.song):
        raise Fail(f"歌曲不存在: {cfg.song}")

    required = [
        ("BS-RoFormer 权重", BS_CKPT),
        ("BS-RoFormer 配置", BS_YAML),
        ("karaoke 权重", KARAOKE_CKPT),
        ("karaoke 配置", KARAOKE_YAML),
        ("4-stem 权重", STEMS4_CKPT),
        ("4-stem 配置", STEMS4_YAML),
        ("RMVPE", RMVPE),
        ("SVC 模型", SVC_MODEL),
        ("SVC 配置", SVC_CONFIG),
    ]
    if cfg.ext_method == "nsf":
        required += [
            ("PC-NSF venv", PCNSF_VENV),
            ("PC-NSF 配置", NSF_CONFIG),
            ("PC-NSF 权重", NSF_CKPT),
        ]
    elif cfg.ext_method == "seedvc":
        required += [("seed-vc venv", VENVS["seedvc"])]
    missing = [(label, p) for label, p in required if not os.path.exists(p)]
    for label, p in required:
        log.info(f"  [{'OK ' if os.path.exists(p) else '缺失'}] {label:<14} {p}")
    if missing:
        raise Fail("缺少预训练模型:\n" + "\n".join(f"    - {l}: {p}" for l, p in missing))

    for key, py in VENVS.items():
        if not os.path.exists(py):
            raise Fail(f"venv python 不存在: {py}（{key} 项目未建 .venv？）")
    if not shutil.which("ffmpeg"):
        raise Fail("未找到 ffmpeg")
    if not shutil.which("nvidia-smi"):
        log.warning("未检测到 nvidia-smi，若用 CPU 会极慢")

    os.makedirs(cfg.out_dir, exist_ok=True)


# ---------------------------------------------------------------- 阶段


def stage_sep(cfg: Cfg) -> None:
    banner("Stage: 人声分离 (BS-RoFormer + karaoke)")
    extra = ["--no-cache"] if cfg.no_sep_cache else []
    run([VENVS["gsv"], os.path.join(WORKERS, "w_sep.py"), "--in", cfg.song,
         "--out-dir", cfg.p("sep1"), "--model", "bs", "--device", cfg.device,
         *extra], stage="sep")
    run([VENVS["gsv"], os.path.join(WORKERS, "w_sep.py"), "--in", cfg.p("sep1", "vocal.wav"),
         "--out-dir", cfg.p("sep2"), "--model", "karaoke", "--device", cfg.device,
         *extra], stage="sep")
    for p in (cfg.p("sep1", "vocal.wav"), cfg.p("sep1", "acc.wav"),
              cfg.p("sep2", "lead.wav"), cfg.p("sep2", "backing.wav")):
        if not os.path.exists(p):
            raise Fail(f"分离产物缺失: {p}")


def stage_stems4(cfg: Cfg) -> None:
    banner("Stage: 伴奏乐器分离 (4-stem -> 鼓点+bass)")
    for p in (cfg.acc_wav,):
        if not os.path.exists(p):
            raise Fail(f"缺少伴奏: {p}（先跑 sep 阶段）")
    run([VENVS["gsv"], os.path.join(ROOT, "tools", "stems4.py"),
         cfg.acc_wav, cfg.p("sep4"), "--mix",
         "--drum-bass-out", cfg.drum_bass_wav,
         "--device", cfg.device], stage="stems4")
    for p in (cfg.drum_bass_wav, cfg.p("sep4", "acc_drums.wav"),
              cfg.p("sep4", "acc_bass.wav")):
        if not os.path.exists(p):
            raise Fail(f"4-stem 分离产物缺失: {p}")
    log.info(f"  -> {cfg.drum_bass_wav}（鼓点+bass 骨架，可用作 --mix 的伴奏）")


def stage_ref_sep(cfg: Cfg) -> None:
    banner("Stage: 参考分离 (BS-RoFormer + karaoke, 去 BGM)")
    extra = ["--no-cache"] if cfg.no_sep_cache else []
    run([VENVS["gsv"], os.path.join(WORKERS, "w_sep.py"), "--in", cfg.ref,
         "--out-dir", cfg.p("refsep1"), "--model", "bs", "--device", cfg.device,
         *extra], stage="ref-sep")
    run([VENVS["gsv"], os.path.join(WORKERS, "w_sep.py"), "--in", cfg.p("refsep1", "vocal.wav"),
         "--out-dir", cfg.p("refsep2"), "--model", "karaoke", "--device", cfg.device,
         *extra], stage="ref-sep")
    for p in (cfg.p("refsep1", "vocal.wav"), cfg.p("refsep2", "lead.wav")):
        if not os.path.exists(p):
            raise Fail(f"参考分离产物缺失: {p}")
    log.info(f"  -> {cfg.ref_lead}")


def stage_f0_target(cfg: Cfg) -> None:
    banner("Stage: 主唱 F0 提取 (RMVPE)")
    run([VENVS["soulx"], os.path.join(WORKERS, "w_f0.py"),
         "--wav", cfg.lead_wav, "--out", cfg.target_f0,
         "--device", cfg.device, "--max-duration", "600"], stage="f0-target")


def stage_select(cfg: Cfg) -> None:
    banner("Stage: 参考选段 + prompt 构建")
    run([VENVS["soulx"], os.path.join(WORKERS, "w_select.py"),
         "--ref", cfg.ref_clean, "--target-f0", cfg.target_f0,
         "--out-dir", cfg.out_dir, "--budget", str(cfg.max_prompt),
         "--device", cfg.device], stage="select")


def stage_ext(cfg: Cfg) -> None:
    engine = "PC-NSF-HiFiGAN" if cfg.ext_method == "nsf" else "seed-vc"
    banner(f"Stage: 音域扩展 ({engine} 分层移调)")
    log.info(f"  扩展引擎: {engine}")
    for p in (cfg.prompt_stats, cfg.p("cands_meta.json"), cfg.target_f0):
        if not os.path.exists(p):
            raise Fail(f"缺少前置产物: {p}（先跑 select 阶段）")
    seedvc_out = cfg.p("seedvc")
    os.makedirs(seedvc_out, exist_ok=True)
    # seed-vc 用相对缓存路径 ./checkpoints（inference.py:5 HF_HUB_CACHE），
    # w_ext.py 内部以 SEEDVC_DIR 为 CWD 启动，避免重新下载模型
    run([VENVS["soulx"], os.path.join(WORKERS, "w_ext.py"),
         "--stats", cfg.prompt_stats, "--cands", cfg.p("cands_meta.json"),
         "--target-f0", cfg.target_f0,
         "--out-wav", cfg.prompt_ext_wav, "--out-f0", cfg.prompt_ext_f0,
         "--seedvc-dir", seedvc_out,
         "--shifter", cfg.ext_method,
         "--anchor-min-ratio", str(cfg.anchor_min_ratio),
         "--max-sec", "29", "--device", cfg.device], stage="ext")


def stage_svc(cfg: Cfg) -> None:
    banner("Stage: 歌声转换 (SoulX-Singer SVC)")
    for p in (cfg.prompt_ext_wav, cfg.prompt_ext_f0, cfg.lead_wav, cfg.target_f0):
        if not os.path.exists(p):
            raise Fail(f"缺少前置产物: {p}")
    svc_dir = cfg.p("svc")
    os.makedirs(svc_dir, exist_ok=True)
    # pitch_shift 只移调「送进模型的 F0 条件」(soulxsinger_svc.py:360 f0_to_coarse
    # 的 f0_shift=pitch_shift*5，1 半音=5 个 20-cent bin)，target_f0.npy 本身不变，
    # 所以 select/ext 仍按原始音域规划。输出音频会跟着高/低相应的半音数。
    if cfg.pitch_shift == "auto":
        log.info("  F0 条件移调: auto（按段对齐到最近八度，男女对唱/跨音区歌曲用）")
    elif cfg.pitch_shift:
        log.info(f"  F0 条件移调: {cfg.pitch_shift:+d} 半音（输出音高同步改变）")
        if cfg.pitch_shift % 12 != 0:
            log.warning(f"  注意: {cfg.pitch_shift:+d} 不是八度整数倍，人声会与原伴奏跑调；"
                        f"±12/±24 才是和声上安全的")
    run([VENVS["soulx"], os.path.join(SOULX_DIR, "cli", "inference_svc.py"),
         "--device", cfg.device,
         "--model_path", SVC_MODEL, "--config", SVC_CONFIG,
         "--prompt_wav_path", cfg.prompt_ext_wav,
         "--target_wav_path", cfg.lead_wav,
         "--prompt_f0_path", cfg.prompt_ext_f0,
         "--target_f0_path", cfg.target_f0,
         "--save_dir", svc_dir,
         "--pitch_shift", str(cfg.pitch_shift),
         "--n_steps", str(cfg.n_steps), "--cfg", str(cfg.cfg),
         "--max_seg_sec", str(cfg.max_seg_sec),
         "--fp16"], stage="svc", cwd=SOULX_DIR,
        env={"PYTHONPATH": SOULX_DIR})  # 直接跑 cli/*.py 时 soulxsinger 包不在 sys.path
    gen = os.path.join(svc_dir, "generated.wav")
    if not os.path.exists(gen):
        raise Fail(f"SVC 未产生输出: {gen}")
    shutil.copy(gen, cfg.vocal_wav)
    log.info(f"  -> {cfg.vocal_wav}")


def stage_eval(cfg: Cfg) -> None:
    banner("Stage: 相似度评测 (ERes2NetV2, 对照真实语音)")
    out = run([VENVS["gsv"], os.path.join(ROOT, "eval_sim.py"),
               "--ref", cfg.ref_clean,
               "--gen", cfg.vocal_wav,
               "--f0", cfg.target_f0,
               "--ref-f0", cfg.prompt_f0], stage="eval")
    with open(cfg.p("eval.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    log.info("  " + out.replace("\n", "\n  "))


def stage_mix(cfg: Cfg) -> None:
    banner("Stage: 混音 (人声 + 伴奏)")
    for p in (cfg.vocal_wav, cfg.acc_wav):
        if not os.path.exists(p):
            raise Fail(f"缺少 {p}")
    run([VENVS["soulx"], os.path.join(WORKERS, "w_mix.py"),
         "--vocal", cfg.vocal_wav, "--acc", cfg.acc_wav, "--out", cfg.mix_wav,
         "--vocal-gain", str(cfg.vocal_gain), "--acc-gain", str(cfg.acc_gain)], stage="mix")


STAGE_FN = {
    "sep": stage_sep,
    "stems4": stage_stems4,
    "ref-sep": stage_ref_sep,
    "f0-target": stage_f0_target,
    "select": stage_select,
    "ext": stage_ext,
    "svc": stage_svc,
    "eval": stage_eval,
    "mix": stage_mix,
}


def _pitch_shift_arg(v: str):
    """--pitch-shift 接受整数半音数或字面量 "auto"（按段对齐到最近八度）。"""
    if isinstance(v, str) and v.lower() == "auto":
        return "auto"
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f'--pitch-shift 需要整数或 "auto"，收到 {v!r}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description="零样本歌声合成：参考语音 -> 用参考音色唱歌曲",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ref", required=True,
                    help="参考音频（通常带 BGM/BGM 含人声；ref-sep 阶段会自动两步分离取纯人声）")
    ap.add_argument("--song", required=True, help="原始歌曲（flac/wav/mp3）")
    ap.add_argument("--name", required=True, help="实验名，决定输出目录 data/songs/<name>/")
    ap.add_argument("--gpu", default="0", help="GPU 编号")
    ap.add_argument("--stages", default="all",
                    help=f"要执行的阶段，逗号分隔。可选: {','.join(ALL_STAGES)}"
                         f"（{'/'.join(sorted(OPT_IN_STAGES))} 默认不跑，需 --mix）")
    ap.add_argument("--max-prompt", type=float, default=25.0,
                    help="选段 prompt 时长上限(秒)，需 <30（whisper 硬顶）")
    ap.add_argument("--semitone-shift", type=int, default=None,
                    help="覆盖自动计算的 seed-vc 变调半音数（默认按 song/prompt 的 p90 自动）")
    ap.add_argument("--ext-method", default="nsf", choices=["seedvc", "nsf"],
                    help="音域扩展引擎: nsf（默认，PC-NSF-HiFiGAN）| seedvc")
    ap.add_argument("--anchor-min-ratio", type=float, default=0.55,
                    help="prompt_ext 中原始干净音频的最低占比（音色锚点，默认 0.55）")
    ap.add_argument("--max-seg-sec", type=float, default=28.0,
                    help="SVC 单段最长秒数（默认 28，16GB 卡无压力；上游默认 30，6GB 卡需降到 20）")
    ap.add_argument("--n-steps", type=int, default=32, help="SVC 采样步数（默认 32）")
    ap.add_argument("--cfg", type=float, default=0.5,
                    help="SVC classifier-free guidance 强度（默认 0.5；上游默认 3.0 实测过强，"
                         "会放大内容特征里的原唱音色残留。0 = 完全关闭 CFG）")
    ap.add_argument("--pitch-shift", type=_pitch_shift_arg, default=0,
                    help='SVC F0 条件移调（半音）。男声歌用 +12（高八度）可对齐女声 prompt；'
                         '"auto" 按段对齐到最近八度（男女对唱/跨音区）；'
                         '非八度整数倍会与原伴奏跑调')
    ap.add_argument("--mix", action="store_true", help="跑混音阶段（验收纯人声后）")
    ap.add_argument("--vocal-gain", type=float, default=1.0)
    ap.add_argument("--acc-gain", type=float, default=0.8)
    ap.add_argument("--no-sep-cache", action="store_true",
                    help="禁用分离缓存（sep/ref-sep 强制重新分离）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印子进程完整输出")
    args = ap.parse_args()

    if args.stages == "all":
        stages = [s for s in ALL_STAGES if s not in OPT_IN_STAGES or args.mix]
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        bad = [s for s in stages if s not in ALL_STAGES]
        if bad:
            sys.exit(f"error: 未知阶段 {bad}，可选: {','.join(ALL_STAGES)}")
        args.mix = args.mix or "mix" in stages

    cfg = Cfg(ref=args.ref, song=args.song, name=args.name, gpu=args.gpu,
              stages=stages, max_prompt=args.max_prompt,
              semitone_shift=args.semitone_shift, ext_method=args.ext_method,
              anchor_min_ratio=args.anchor_min_ratio,
              max_seg_sec=args.max_seg_sec,
              n_steps=args.n_steps, cfg=args.cfg, pitch_shift=args.pitch_shift,
              vocal_gain=args.vocal_gain, acc_gain=args.acc_gain, mix=args.mix,
              no_sep_cache=args.no_sep_cache)

    setup_logging(cfg.log_file, args.verbose)
    log.info(f"实验    : {cfg.name}")
    log.info(f"参考    : {cfg.ref}")
    log.info(f"歌曲    : {cfg.song}")
    log.info(f"阶段    : {' -> '.join(stages)}")
    log.info(f"扩展引擎: {cfg.ext_method}（nsf=PC-NSF-HiFiGAN / seedvc=seed-vc）")
    log.info(f"日志    : {cfg.log_file}")

    t0 = time.time()
    try:
        preflight(cfg)
        for s in stages:
            ts = time.time()
            STAGE_FN[s](cfg)
            log.info(f"  [{s}] 耗时 {time.time() - ts:.1f}s")
        log.info("")
        log.info(f"完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
        log.info(f"  纯人声 : {cfg.vocal_wav}")
        log.info(f"  评测   : {cfg.p('eval.txt')}")
        if cfg.mix_wav in [cfg.p("mix.wav")] and os.path.exists(cfg.mix_wav):
            log.info(f"  成品   : {cfg.mix_wav}")
        else:
            log.info("  混音   : 先验收纯人声，再 --stages mix（可调 --acc-gain）")
    except Fail as e:
        log.error("")
        log.error(f"失败: {e}")
        log.error(f"完整日志: {cfg.log_file}（用 -v 可看子进程全部输出）")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("\n已中断。已完成的阶段产物保留在磁盘，重跑会自动跳过")
        sys.exit(130)


if __name__ == "__main__":
    main()
