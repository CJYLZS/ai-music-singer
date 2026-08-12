#!/usr/bin/env python
"""零样本歌声合成 CLI：参考语音 -> 用参考音色唱给定歌曲。

两条流水线（可独立运行或级联）：

(1) SVC 流水线（纯音色转换，保留原唱旋律细节）:
    song.flac
      ├─ BS-RoFormer          -> 人声(含和声) + 伴奏      (GPT-SoVITS venv)
      │    └─ karaoke 模型    -> 主唱 + 和声
      ├─ 伴奏 -> 4-stem 分离   -> 鼓点+bass 骨架 (sep4/, GPT-SoVITS venv)
      ├─ 主唱 -> RMVPE F0     (SoulX venv)
      ├─ 参考音频(通常带BGM/BGM含人声) 同样两步分离 -> 纯人声 lead
      ├─ 纯人声切片 -> F0 分析 -> 贪心选段 -> prompt.wav (SoulX venv)
      ├─ PC-NSF-HiFiGAN 分层移调补齐高音区 -> prompt_ext.wav (tools/pcnsf venv)
      ├─ SoulX-Singer SVC: prompt_ext + 主唱 -> 目标音色人声 (SoulX venv)
      └─ ERes2NetV2 相似度评测（对照真实语音）  (GPT-SoVITS venv)

(2) SVS 流水线（score-control，MIDI+歌词直接合成，无原唱音色泄漏）:
    song.flac -> 自动转录 MIDI + 歌词 -> SVS 合成（带参考音色）

(3) Cascade 流水线（SVS → SVC，结合两者优势，实测音色最优）:
    SVS 输出 -> 提取 F0 -> SVC 二次转换（复用 prompt_ext）-> vocal_cascade.wav

最简用法（纯 SVC）:
    python3 sing.py --ref data/voices/ref.wav --song data/songs/song.flac --name exp1

Cascade 流程（当前最优配置）:
    python3 sing.py --ref ... --song ... --name exp1 --stages sep,ref-sep,f0-target,select,ext,svs,cascade

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

ALL_STAGES = ["sep", "stems4", "ref-sep", "f0-target", "select", "ext", "svc", "svs", "cascade", "eval", "mix"]
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
    select_min_sec: float = 0.3   # select 候选最短时长（见 AGENTS.md「最小分段长度」）
    semitone_shift: int | None = None
    ext_method: str = "nsf"
    anchor_min_ratio: float = 0.55
    ext_max_shift: int = 8   # ext 单段最大移调半音。原默认 5，2026-08 A/B 3首
    # 结论：ms=8 在复用未饱和的歌上解锁新源段（1.37-1.41×→1.10×），在饱和的歌上
    # 扩展高音格覆盖，3/3 全正（+0.003-0.007，p<0.01）。见 AGENTS.md「ext 移调上限扩展」。
    ext_div_weight: float = 3.0  # ext 源段复用惩罚（等价半音数），见 AGENTS.md「ext 多样性」
    ext_exclude_anchor: bool = False  # ext 是否排除锚点已含的源段。2026-08 跨 3 首歌：
    # 绿色(重叠62%) +0.0090 p=0.0002，但冬眠(39%) -0.0004、爱笑的眼睛(33%) +0.0017 都在
    # 噪声底内，3 首均值仅 +0.0034 —— 不够格当默认。重叠比例高(>~60%)的歌可手动开
    # `--ext-exclude-anchor`。见 AGENTS.md「ext 与锚点的段落重复」
    # 注意：select_method != greedy 时本项无意义（MILP 把"一段只用一次"做成硬约束）。
    select_method: str = "greedy"  # 选段求解器。cov_anchor(MILP) 曾短暂做过默认，已撤回：
    # 它把「选锚点」与「填频段缺口」合并成一个 (段,移调) 指派问题，"一段素材只用一次"是
    # 硬约束，在 3 首歌的 90s clip 上 +0.0050 且 0 负 —— 但全曲 10 首配对复现后均值只有
    # +0.0012：5 首显著正（+0.005…+0.013）、5 首显著负（−0.003…−0.008），效应按歌分向
    # （corr(gain, song_med) = +0.516，高音歌 MILP 胜、低音歌 greedy 胜）。ext 耗时
    # 12.9s -> 253.6s（19.7×）。均值上零收益、一半歌还变差，故回退；若歌曲音域明显高于
    # 参考（song_med 差 >~4st）可对单首歌试 cov_anchor。见 AGENTS.md 同名条目。
    # 仍保留为可选项：MILP 的"不重复"硬约束本身是免费的（收紧 LP 松弛，反而更快收敛）。
    cov_target: float = 0.90  # 仅 select_method=satisfice 时生效
    max_sec: float = 45.0   # prompt_ext 总长上限。原默认 29，2026-08 跨歌 A/B 改为 45：
    # 绿色/白色毛衣在 45s 均比 29s 增益 +0.013…+0.023；爱笑的眼睛 29-45s 平坦（错配
    # 太大导致 prompt 非瓶颈）。45s 是素材 prompt.wav 普遍在 ~25s 时的甜点 (25/0.55)。
    # 见 AGENTS.md「prompt 长度与素材上限的关系」。
    max_seg_sec: float = 28.0
    num_overlaps: int = 1   # 每段往前取几个活性格作上下文（见 AGENTS.md「num_overlaps」）
    knn_alpha: float = 0.2   # kNN 内容特征替换强度（0.1-0.2 是 CER/SIM 双赢区，≥0.3 起吐字退化）
    knn_k: int = 4   # kNN 近邻数（默认 4，同 kNN-VC）
    knn_full_pool: bool = True   # kNN 检索池用完整参考人声而非 28.9s prompt_ext（绕开 whisper 30s）
    n_steps: int = 32
    cfg: float = 0.5   # 上游默认 3.0，实测偏高（见 AGENTS.md「cfg 扫点」）
    rescale_cfg: float = 0.75   # CFG 外推的 std 归一化混合比（上游默认，0=关闭归一化）
    pitch_shift: int | str = 0
    seed: int | None = None   # CFM 初始噪声 seed；None = 上游行为（每次随机、不可复现）
    n_seeds: int = 1   # >1 时跑 N 个 seed 再用 DP 段级选择（见 AGENTS.md「DP 段级选择」）
    dp_lam: float = 0.0   # DP 边界一致性惩罚权重（0=朴素 argmax；听到明显跳变时才往上调）
    vocal_gain: float = 1.0
    acc_gain: float = 0.8
    match_acc: bool = True   # 人声增益跟随伴奏 RMS 包络（高潮段伴奏变厚，固定增益会压住人声）
    match_margin: float = 0.0   # 目标：人声 RMS 比伴奏高多少 dB
    match_max_boost: float = 8.0   # 跟随增益上限 dB
    limit_db: float = -0.5   # 软限幅上限 dBFS；0 = 关闭改用硬削波
    mix_mono: bool = False   # 默认保留伴奏立体声，人声居中
    mix_vocal: str | None = None   # 混音人声输入路径（None=自动选择，优先 cascade）
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

    @property
    def ext_shifted_dir(self) -> str:
        """ext 阶段的移调段输出目录（seedvc/ 或其他）。"""
        return self.p("seedvc")  # w_ext.py 写死这个名字

    @property
    def svs_vocal_wav(self) -> str:
        """SVS 阶段产物（score-control 合成的歌声）。"""
        return self.p("svs", "svs_vocal.wav")

    @property
    def svs_f0(self) -> str:
        """SVS 输出的 F0（cascade 阶段需要）。"""
        return self.p("svs", "svs_f0.npy")

    @property
    def vocal_cascade_wav(self) -> str:
        """cascade 阶段产物（SVS → SVC 二次转换）。"""
        return self.p("vocal_cascade.wav")


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
         "--min-sec", str(cfg.select_min_sec),
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
         "--max-shift", str(cfg.ext_max_shift),
         "--div-weight", str(cfg.ext_div_weight),
         "--max-sec", str(cfg.max_sec),
         ("--exclude-anchor-segs" if cfg.ext_exclude_anchor else "--no-exclude-anchor-segs"),
         "--select-method", cfg.select_method,
         "--cov-target", str(cfg.cov_target),
         "--device", cfg.device], stage="ext")


def _svc_cmd(cfg: Cfg, svc_dir: str, seed: int | None) -> list[str]:
    """构建单次 SVC 推理命令。"""
    cmd = [VENVS["soulx"], os.path.join(SOULX_DIR, "cli", "inference_svc.py"),
           "--device", cfg.device,
           "--model_path", SVC_MODEL, "--config", SVC_CONFIG,
           "--prompt_wav_path", cfg.prompt_ext_wav,
           "--target_wav_path", cfg.lead_wav,
           "--prompt_f0_path", cfg.prompt_ext_f0,
           "--target_f0_path", cfg.target_f0,
           "--save_dir", svc_dir,
           "--pitch_shift", str(cfg.pitch_shift),
           "--n_steps", str(cfg.n_steps), "--cfg", str(cfg.cfg),
           "--rescale_cfg", str(cfg.rescale_cfg),
           "--max_seg_sec", str(cfg.max_seg_sec),
           "--num_overlaps", str(cfg.num_overlaps),
           "--knn_alpha", str(cfg.knn_alpha),
           "--knn_k", str(cfg.knn_k)]
    # kNN 检索池默认用完整参考人声（163s）而非 prompt_ext（28.9s）：检索池不进模型，
    # 不受 whisper 30s 硬顶约束，分块编码后匹配失败帧 9.8% -> 1.9%
    if cfg.knn_alpha > 0 and cfg.knn_full_pool and os.path.exists(cfg.ref_clean):
        cmd += ["--knn_pool_wav", os.path.abspath(cfg.ref_clean)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd + ["--fp16"]


def _svc_multi_seed(cfg: Cfg) -> None:
    """--n-seeds N: 跑 N 个 seed 后用 DP 段级选择拼出最优人声。"""
    log.info(f"  多采样模式: {cfg.n_seeds} 个 seed → DP 段级选择（λ={cfg.dp_lam:g}）")
    base_seed = cfg.seed if cfg.seed is not None else 1
    cands: list[str] = []
    for i in range(cfg.n_seeds):
        s = base_seed + i * 1000   # 每个 seed 之间间隔大，避免各段 seed+idx 出现交叉
        svc_dir = cfg.p(f"svc_s{s}")
        os.makedirs(svc_dir, exist_ok=True)
        log.info(f"  SVC run {i+1}/{cfg.n_seeds}  seed={s}")
        run(_svc_cmd(cfg, svc_dir, s), stage="svc", cwd=SOULX_DIR,
            env={"PYTHONPATH": SOULX_DIR})
        gen = os.path.join(svc_dir, "generated.wav")
        if not os.path.exists(gen):
            raise Fail(f"SVC 未产生输出（seed={s}）: {gen}")
        cands.append(gen)
    # DP 选择
    log.info(f"  DP 段级选择  λ={cfg.dp_lam:g}  {cfg.n_seeds} 候选")
    os.makedirs(cfg.p("svc"), exist_ok=True)
    report = cfg.p("svc", "dp_select.json")
    run([VENVS["gsv"], os.path.join(WORKERS, "w_dpselect.py"),
         "--prompt", cfg.prompt_ext_wav,
         "--target-f0", cfg.target_f0,
         "--cands", *cands,
         "--out", cfg.vocal_wav,
         "--lam", str(cfg.dp_lam),
         "--max-seg-sec", str(cfg.max_seg_sec),
         "--num-overlaps", str(cfg.num_overlaps),
         "--device", cfg.device,
         "--report", report], stage="svc")
    log.info(f"  -> {cfg.vocal_wav}")


def stage_svc(cfg: Cfg) -> None:
    banner("Stage: 歌声转换 (SoulX-Singer SVC)")
    for p in (cfg.prompt_ext_wav, cfg.prompt_ext_f0, cfg.lead_wav, cfg.target_f0):
        if not os.path.exists(p):
            raise Fail(f"缺少前置产物: {p}")
    if cfg.n_seeds > 1:
        _svc_multi_seed(cfg)
        return
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
    if cfg.seed is not None:
        log.info(f"  CFM 噪声 seed: {cfg.seed}（可复现；第 i 段用 seed+i）")
    run(_svc_cmd(cfg, svc_dir, cfg.seed), stage="svc", cwd=SOULX_DIR,
        env={"PYTHONPATH": SOULX_DIR})
    gen = os.path.join(svc_dir, "generated.wav")
    if not os.path.exists(gen):
        raise Fail(f"SVC 未产生输出: {gen}")
    shutil.copy(gen, cfg.vocal_wav)
    log.info(f"  -> {cfg.vocal_wav}")


def stage_svs(cfg: Cfg) -> None:
    banner("Stage: SVS 合成 (score-control, MIDI+lyrics+ref)")
    for p in (cfg.lead_wav, cfg.prompt_wav):
        if not os.path.exists(p):
            raise Fail(f"缺少前置产物: {p}")
    # svs_song.py 默认用 prompt.wav（说话），可传 --prompt-wav 换成唱歌干声
    # 这里保持与 SVC 一致：都用 prompt.wav（说话参考），方便直接对比
    svs_dir = cfg.p("svs")
    os.makedirs(svs_dir, exist_ok=True)
    cmd = [VENVS["soulx"], os.path.join(WORKERS, "svs_song.py"),
           "--song-dir", cfg.out_dir,
           "--out-dir", svs_dir,
           "--prompt-wav", cfg.prompt_wav,
           "--device", cfg.device,
           "--seed", str(cfg.seed) if cfg.seed is not None else "1234",
           "--control", "score"]
    run(cmd, stage="svs", cwd=SOULX_DIR, env={"PYTHONPATH": SOULX_DIR})
    if not os.path.exists(cfg.svs_vocal_wav):
        raise Fail(f"SVS 未产生输出: {cfg.svs_vocal_wav}")
    log.info(f"  -> {cfg.svs_vocal_wav}")


def stage_cascade(cfg: Cfg) -> None:
    banner("Stage: Cascade (SVS → SVC 二次转换)")
    for p in (cfg.svs_vocal_wav, cfg.prompt_ext_wav, cfg.prompt_ext_f0):
        if not os.path.exists(p):
            raise Fail(f"缺少前置产物: {p}（需先跑 svs 和 ext 阶段）")
    # 1. 提取 SVS 输出的 F0（作为 SVC 的 target_f0）
    log.info("  提取 SVS 输出 F0")
    run([VENVS["soulx"], os.path.join(WORKERS, "w_f0.py"),
         "--wav", cfg.svs_vocal_wav, "--out", cfg.svs_f0,
         "--device", cfg.device, "--max-duration", "600"], stage="cascade")
    # 2. 用 SVC 转换（source = SVS 输出，prompt = 同一套 prompt_ext）
    log.info("  SVC 二次转换（source=SVS 输出，复用 prompt_ext）")
    svc_dir = cfg.p("cascade_svc")
    os.makedirs(svc_dir, exist_ok=True)
    # 复用 _svc_cmd 但替换 target 路径，注意所有路径都要转绝对路径（CWD=SOULX_DIR）
    cmd = [VENVS["soulx"], os.path.join(SOULX_DIR, "cli", "inference_svc.py"),
           "--device", cfg.device,
           "--model_path", SVC_MODEL, "--config", SVC_CONFIG,
           "--prompt_wav_path", os.path.abspath(cfg.prompt_ext_wav),
           "--target_wav_path", os.path.abspath(cfg.svs_vocal_wav),  # 关键：source 换成 SVS 输出
           "--prompt_f0_path", os.path.abspath(cfg.prompt_ext_f0),
           "--target_f0_path", os.path.abspath(cfg.svs_f0),           # F0 也是 SVS 的
           "--save_dir", os.path.abspath(svc_dir),
           "--pitch_shift", str(cfg.pitch_shift),
           "--n_steps", str(cfg.n_steps), "--cfg", str(cfg.cfg),
           "--rescale_cfg", str(cfg.rescale_cfg),
           "--max_seg_sec", str(cfg.max_seg_sec),
           "--num_overlaps", str(cfg.num_overlaps),
           "--knn_alpha", str(cfg.knn_alpha),
           "--knn_k", str(cfg.knn_k)]
    if cfg.knn_alpha > 0 and cfg.knn_full_pool and os.path.exists(cfg.ref_clean):
        cmd += ["--knn_pool_wav", os.path.abspath(cfg.ref_clean)]
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    cmd += ["--fp16"]
    run(cmd, stage="cascade", cwd=SOULX_DIR, env={"PYTHONPATH": SOULX_DIR})
    gen = os.path.join(svc_dir, "generated.wav")
    if not os.path.exists(gen):
        raise Fail(f"cascade SVC 未产生输出: {gen}")
    shutil.copy(gen, cfg.vocal_cascade_wav)
    log.info(f"  -> {cfg.vocal_cascade_wav}")


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
    # 用户指定 > cascade 产物 > vocal.wav
    if cfg.mix_vocal:
        # 支持绝对路径或相对于歌曲目录的文件名
        if os.path.isabs(cfg.mix_vocal):
            vocal_input = cfg.mix_vocal
        else:
            vocal_input = cfg.p(cfg.mix_vocal)
    elif os.path.exists(cfg.vocal_cascade_wav):
        vocal_input = cfg.vocal_cascade_wav
    else:
        vocal_input = cfg.vocal_wav
    
    if not os.path.exists(vocal_input):
        raise Fail(f"缺少人声输入: {vocal_input}")
    if not os.path.exists(cfg.acc_wav):
        raise Fail(f"缺少 {cfg.acc_wav}")
    log.info(f"  人声输入: {vocal_input}")
    cmd = [VENVS["soulx"], os.path.join(WORKERS, "w_mix.py"),
           "--vocal", vocal_input, "--acc", cfg.acc_wav, "--out", cfg.mix_wav,
           "--vocal-gain", str(cfg.vocal_gain), "--acc-gain", str(cfg.acc_gain),
           "--match-margin", str(cfg.match_margin),
           "--match-max-boost", str(cfg.match_max_boost),
           "--limit-db", str(cfg.limit_db)]
    if not cfg.match_acc:
        cmd.append("--no-match-acc")
    if cfg.mix_mono:
        cmd.append("--mono")
    run(cmd, stage="mix")


STAGE_FN = {
    "sep": stage_sep,
    "stems4": stage_stems4,
    "ref-sep": stage_ref_sep,
    "f0-target": stage_f0_target,
    "select": stage_select,
    "ext": stage_ext,
    "svc": stage_svc,
    "svs": stage_svs,
    "cascade": stage_cascade,
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
    ap.add_argument("--select-min-sec", type=float, default=0.3,
                    help="select 候选段最短时长(秒)，默认 0.3。降到 0.15 可捕捉极短高音，"
                         "提到 0.8-1.2 可减少拼接点")
    ap.add_argument("--semitone-shift", type=int, default=None,
                    help="覆盖自动计算的 seed-vc 变调半音数（默认按 song/prompt 的 p90 自动）")
    ap.add_argument("--ext-method", default="nsf", choices=["seedvc", "nsf"],
                    help="音域扩展引擎: nsf（默认，PC-NSF-HiFiGAN）| seedvc")
    ap.add_argument("--anchor-min-ratio", type=float, default=0.55,
                    help="原始干净音频在 prompt_ext 中占的份额上限 cap，默认 0.55 时锚点裁到 "
                         "15.95s、ext 拿 45 pct。扫 0.0 到 0.9 即 ext 占比 100 pct 到 14 pct")
    ap.add_argument("--ext-div-weight", type=float, default=3.0,
                    help="ext 源段复用惩罚，折算为等价半音数（默认 3）。0=旧行为。"
                         "只按移调量最小排序会把素材锁在 med 最高的少数段上（实测 143 段池"
                         "只用到 13 个源段，复用 2.2x）；调大用更大移调换更多不同源段")
    ap.add_argument("--ext-max-shift", type=int, default=8,
                    help="ext 阶段单段最大移调半音数（默认 8）。2026-08 A/B 3 首歌：ms=8 vs ms=5 全正 "
                         "(+0.003-0.007, p<0.01)，机制是解锁低音段参与高音格竞争，唯一源段 17→21，"
                         "复用 1.37×→1.10×。见 AGENTS.md「ext 移调上限扩展」")
    ap.add_argument("--select-method", default="greedy",
                    choices=["cov_anchor", "cov_only", "satisfice", "greedy"],
                    help="prompt 选段求解器（默认 greedy = 两阶段贪心）。cov_anchor/cov_only/"
                         "satisfice 走 MILP：锚点与 ext 合并求解，'一段素材只用一次' 为硬约束。"
                         "MILP 曾做过默认后撤回：3 首调参歌的 clip 上 +0.0050(0负)，但全曲 10 "
                         "首仅 +0.0012：5 首显著正、5 首显著负（效应按歌分向，高音歌 MILP 胜），"
                         "且 ext 耗时 19.7×。见 singing/ext_milp.py 与 AGENTS.md")
    ap.add_argument("--cov-target", type=float, default=0.90,
                    help="--select-method satisfice 的覆盖率目标（默认 0.90）")
    ap.add_argument("--ext-exclude-anchor", dest="ext_exclude_anchor",
                    action="store_true", default=False,
                    help="ext 不复用锚点(prompt.wav)已含的源段，避免同一素材在 prompt_ext 里"
                         "以原声+移调版出现两次。默认关：跨 3 首歌只有重叠比例最高的绿色(62%%)"
                         "显著受益 (+0.0090, p=0.0002)，冬眠(39%%) -0.0004 与爱笑的眼睛(33%%) "
                         "+0.0017 都在噪声底内。corr(重叠%%, gain)=+0.91 但 n=3 不足以定标")
    ap.add_argument("--max-sec", type=float, default=45.0,
                    help="prompt_ext 总长上限（默认 45，原 29）。2026-08 跨歌 A/B：绿色 +0.0229、"
                         "白色毛衣 +0.0133（均 3/3 seed 一致）；再加到 50s 则视素材而定 —— 歌曲若有 "
                         "帧超出「素材最高段 med × 2^(max_shift/12)」，高音格填不满会逼贪心堆 +8st，"
                         "反而退化（绿色 45→50 为 −0.0113）。见 AGENTS.md「prompt 长度」")
    ap.add_argument("--max-seg-sec", type=float, default=28.0,
                    help="SVC 单段最长秒数（默认 28，16GB 卡无压力；上游默认 30，6GB 卡需降到 20）")
    ap.add_argument("--num-overlaps", type=int, default=1,
                    help="SVC 每段往前取几个活性格作前置上下文（默认 1）。"
                         "总窗仍受 --max-seg-sec 限制，段本来就接近上限时提高它无效")
    ap.add_argument("--knn-alpha", type=float, default=0.2,
                    help="kNN 内容特征替换强度（默认 0.2）。把 source 的 whisper 帧向参考的帧做 "
                         "kNN 插值，用目标说话人的残留替换原唱残留（mean-shift 删除已实测变差）。"
                         "0.1-0.2 是 CER 与 SIM 双赢区；0.3 起吐字开始退化，0.4 时 CER +5.9 点不可用。"
                         "0=关闭")
    ap.add_argument("--knn-k", type=int, default=4,
                    help="kNN 近邻数（默认 4，同 kNN-VC）")
    ap.add_argument("--knn-full-pool", action="store_true", default=True,
                    help="kNN 检索池用完整参考人声(默认开启)，分块编码绕开 whisper 30s 截断；"
                         "匹配失败帧 9.8%%->1.9%%. --no-knn-full-pool 回退到 prompt_ext")
    ap.add_argument("--no-knn-full-pool", dest="knn_full_pool", action="store_false",
                    help="禁用完整检索池，回退到 prompt_ext（30s 截断行为）")
    ap.add_argument("--n-steps", type=int, default=32, help="SVC 采样步数（默认 32）")
    ap.add_argument("--cfg", type=float, default=0.5,
                    help="SVC classifier-free guidance 强度（默认 0.5；上游默认 3.0 实测过强，"
                         "会放大内容特征里的原唱音色残留。0 = 完全关闭 CFG）")
    ap.add_argument("--rescale-cfg", type=float, default=0.75,
                    help="CFG 外推结果的 std 归一化混合比（1=全归一化，0=用原始外推；上游 0.75）。"
                         "cfg=0 时这一路完全不走")
    ap.add_argument("--pitch-shift", type=_pitch_shift_arg, default=0,
                    help='SVC F0 条件移调（半音）。男声歌用 +12（高八度）可对齐女声 prompt；'
                         '"auto" 按段对齐到最近八度（男女对唱/跨音区）；'
                         '非八度整数倍会与原伴奏跑调')
    ap.add_argument("--seed", type=int, default=None,
                    help="CFM 初始噪声 seed（默认不设 = 每次随机、不可复现）。"
                         "设定后同参数可复现，第 i 段用 seed+i")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="多采样：跑 N 个 seed 再用 DP 段级选择拼出最优人声（1=单次推理）。"
                         "代价 N× 时间，收益 +0.015~+0.04；详见 AGENTS.md「DP 段级选择」")
    ap.add_argument("--dp-lam", type=float, default=0.0,
                    help="DP 边界一致性惩罚权重（默认 0 = 朴素 argmax，均值最高）。"
                         "调高只换来更稳的边界、更低的均值（拿 SIM 换 drift，同 --cfg 的形状）")
    ap.add_argument("--mix", action="store_true", help="跑混音阶段（验收纯人声后）")
    ap.add_argument("--vocal-gain", type=float, default=1.0)
    ap.add_argument("--acc-gain", type=float, default=0.8)
    ap.add_argument("--no-match-acc", dest="match_acc", action="store_false", default=True,
                    help="关闭「人声增益跟随伴奏包络」，回到固定增益")
    ap.add_argument("--match-margin", type=float, default=0.0,
                    help="混音目标：人声 RMS 比伴奏高多少 dB（默认 0）")
    ap.add_argument("--match-max-boost", type=float, default=8.0,
                    help="跟随增益上限 dB（默认 8）")
    ap.add_argument("--limit-db", type=float, default=-0.5,
                    help="混音软限幅上限 dBFS；0 = 关闭改用硬削波（默认 -0.5）")
    ap.add_argument("--mix-mono", action="store_true",
                    help="混音输出单声道（默认保留伴奏立体声）")
    ap.add_argument("--mix-vocal", default=None,
                    help="混音人声输入路径。默认自动选择：优先 vocal_cascade.wav，"
                         "不存在则回退到 vocal.wav。可指定绝对路径或相对于 data/songs/<name>/ 的文件名")
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
              select_min_sec=args.select_min_sec,
              semitone_shift=args.semitone_shift, ext_method=args.ext_method,
              anchor_min_ratio=args.anchor_min_ratio,
              ext_max_shift=args.ext_max_shift,
              ext_div_weight=args.ext_div_weight,
              ext_exclude_anchor=args.ext_exclude_anchor,
              select_method=args.select_method,
              cov_target=args.cov_target,
              max_sec=args.max_sec,
              max_seg_sec=args.max_seg_sec, num_overlaps=args.num_overlaps,
              knn_alpha=args.knn_alpha, knn_k=args.knn_k,
              knn_full_pool=args.knn_full_pool,
              n_steps=args.n_steps, cfg=args.cfg, rescale_cfg=args.rescale_cfg,
              pitch_shift=args.pitch_shift,
              seed=args.seed, n_seeds=args.n_seeds, dp_lam=args.dp_lam,
              vocal_gain=args.vocal_gain, acc_gain=args.acc_gain,
              match_acc=args.match_acc, match_margin=args.match_margin,
              match_max_boost=args.match_max_boost, limit_db=args.limit_db,
              mix_mono=args.mix_mono, mix_vocal=args.mix_vocal, mix=args.mix,
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
        
        # 根据跑过的阶段输出产物路径
        if "svc" in stages and os.path.exists(cfg.vocal_wav):
            log.info(f"  SVC 人声 : {cfg.vocal_wav}")
        if "svs" in stages and os.path.exists(cfg.svs_vocal_wav):
            log.info(f"  SVS 人声 : {cfg.svs_vocal_wav}")
        if "cascade" in stages and os.path.exists(cfg.vocal_cascade_wav):
            log.info(f"  级联人声 : {cfg.vocal_cascade_wav}")
        if "eval" in stages and os.path.exists(cfg.p("eval.txt")):
            log.info(f"  评测     : {cfg.p('eval.txt')}")
        if cfg.mix_wav in [cfg.p("mix.wav")] and os.path.exists(cfg.mix_wav):
            log.info(f"  成品     : {cfg.mix_wav}")
        elif "mix" not in stages:
            log.info("  混音     : 先验收纯人声，再 --stages mix（可调 --acc-gain）")
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
