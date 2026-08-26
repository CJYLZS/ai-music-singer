#!/usr/bin/env python3
"""段落级 SVC 驱动：切分 → 每段选参考 → 渲染 → 段内 SVC → 静音处拼接。

依赖 w_segprompt.py（切分/选段）+ w_ext.py（PC-NSF 移调渲染）。
用法（SoulX venv，CWD=仓库根）:
  python singing/run_segprompt.py 亲亲 --prompt-budget 45 --seed 1234
产物: data/songs/<song>/vocal.wav（默认；--out 可改）+ segprompt_meta.json
      （每段 t0/t1/ext_t0/ext_t1、med_f0、参考候选索引）+ 中间文件在 --tmp-dir。
sing.py 集成：--svc-prompt segprompt（默认）经 stage_svc 调本脚本，输出标准
vocal.wav 供下游 mix/评测直接使用。

效率（2026-08 方案 A）：模型只在单进程内加载一次，所有段共享；
kNN pool 预编码一次传入每段（soulxsinger_svc.py:infer 支持 knn_pool 透传），
避免每段重新加载 2.8GB 模型 + 重编 537s 参考。每阶段耗时打点到输出目录
<tmp>/run.log 便于排查。

段间冗余（2026-08）：每段前后各 pad_ext=2s 一起送 SVC（段首/段尾冷启动音质
差，把邻接音频喂给模型当上下文），拼接时只取中间 [t0,t1]。推理上限
--svc-max-sec 默认 29.0（SoulX 整段直推判定 gt < max_seg_sec；whisper 30s 硬
顶），所以内容段长 --seg-max 默认 25.0 = 29 - 2*2。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "SoulX-Singer"
MODEL = SOULX / "pretrained_models" / "SoulX-Singer" / "model-svc.pt"
CONFIG = SOULX / "soulxsinger" / "config" / "soulxsinger.yaml"
POOL = ROOT / "data" / "voices" / "ref_talk_sing.wav"
F0_RATE = 50.0
SR_OUT = 44100

sys.path.insert(0, str(ROOT / "singing"))
sys.path.insert(0, str(SOULX))
from w_segprompt import (split_silence, split_bands, seg_med_f0,
                         pick_prompt)  # noqa: E402
from w_ext import nsf_shift_many, RMVPE  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402
from soulxsinger.utils.file_utils import load_config  # noqa: E402
from soulxsinger.utils.audio_utils import load_wav  # noqa: E402
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC  # noqa: E402


class _T:
    """每阶段耗时打点，写日志到输出目录。"""

    def __init__(self, log_path: Path):
        self.log = open(log_path, "a", encoding="utf-8")
        self.t0 = time.time()
        self.last = self.t0
        self.log.write(f"=== segprompt start {time.strftime('%F %T')} ===\n")
        self.log.flush()

    def mark(self, tag: str):
        now = time.time()
        self.log.write(f"  [{now - self.t0:7.1f}s] (+{now - self.last:6.1f}s) {tag}\n")
        self.log.flush()
        self.last = now

    def close(self):
        self.log.close()


def render_prompt(picked, tmp: Path, tag: str, extractor, T: _T) -> tuple:
    """[(cand, shift)] -> (prompt_wav, prompt_f0)。按 shift 分组一次跑 vocoder。

    临时文件策略（2026-08）：移调组的 *_in_* / *_out_*.wav 是 PC-NSF vocoder
    的必要文件 IO，用完即删；候选段切片不落盘——F0 用 extractor.process_audio
    内存提，只保留最终的 prompt_segN.wav + _f0.npy。
    """
    by_sh = {}
    for c, sh in picked:
        by_sh.setdefault(sh, []).append(c)
    jobs, part_audio = [], {}
    for sh, cs in sorted(by_sh.items()):
        parts = [sf.read(str(c["wav"]), dtype="float32")[0] for c in cs]
        part_audio[sh] = parts
        if sh == 0:
            continue  # 0st 用原音频
        in_wav = tmp / f"{tag}_in_{sh:+03d}.wav"
        sf.write(str(in_wav), np.concatenate(parts), SR_OUT)
        jobs.append({"wav": str(in_wav), "shift": sh,
                     "out": str(tmp / f"{tag}_out_{sh:+03d}.wav")})
    shifted = {}
    if jobs:
        nsf_shift_many(jobs, tmp)
        for j in jobs:
            y, sr_s = sf.read(j["out"], dtype="float32")
            if sr_s != SR_OUT:
                raise SystemExit(f"采样率不符 {sr_s}")
            shifted[j["shift"]] = y
    for sh in by_sh:
        if sh == 0:
            shifted[0] = np.concatenate(part_audio[0])
    p_parts, p_f0s = [], []
    for sh in sorted(by_sh):
        cs, parts = by_sh[sh], part_audio[sh]
        arr = shifted[sh]
        off = 0
        for c, part in zip(cs, parts):
            seg = arr[off: off + len(part)]
            off += len(part)
            # f0 帧数需 ≥ mel 帧数（infer_segment 的 F.pad 对 batch 维有 bug，
            # 只在 pt_f0.shape[1] < pt_mel.shape[1] 时触发）。加 buffer 保证 ≥。
            nf = int(len(seg) / SR_OUT * F0_RATE) + 4
            ff = extractor.process_audio(seg, SR_OUT, verbose=False)[:nf]
            if len(ff) < nf:
                ff = np.pad(ff, (0, nf - len(ff)))
            p_parts.append(seg)
            p_f0s.append(ff)
    # 清理移调临时文件（vocoder 已跑完，不再需要）
    for f in tmp.glob(f"{tag}_in_*.wav"):
        f.unlink(missing_ok=True)
    for f in tmp.glob(f"{tag}_out_*.wav"):
        f.unlink(missing_ok=True)
    p_wav = tmp / f"prompt_{tag}.wav"
    p_f0 = tmp / f"prompt_{tag}_f0.npy"
    sf.write(str(p_wav), np.concatenate(p_parts), SR_OUT)
    np.save(str(p_f0), np.concatenate(p_f0s))
    return str(p_wav), str(p_f0)


def svc_infer(model, config, p_wav, seg_wav, p_f0, seg_f0,
              knn_pool, device, use_fp16, args, T: _T, si: int):
    """单进程内对一段做 SVC 推理，返回 generated 音频 (numpy float32, 24k)。"""
    T.mark(f"段{si}: 加载 prompt/target")
    pt_wav = load_wav(p_wav, config.audio.sample_rate).to(device)
    gt_wav = load_wav(seg_wav, config.audio.sample_rate).to(device)
    pt_f0 = torch.from_numpy(np.load(p_f0)).unsqueeze(0).to(device)
    gt_f0 = torch.from_numpy(np.load(seg_f0)).unsqueeze(0).to(device)
    T.mark(f"段{si}: wav/f0 就绪")
    with torch.no_grad():
        generated_audio, _ = model.infer(
            pt_wav=pt_wav, gt_wav=gt_wav, pt_f0=pt_f0, gt_f0=gt_f0,
            auto_shift=False, pitch_shift=0,
            n_steps=args.n_steps, cfg=args.cfg,
            use_fp16=use_fp16, max_seg_sec=args.svc_max_sec,
            seed=args.seed, num_overlaps=args.num_overlaps,
            rescale_cfg=args.rescale_cfg,
            knn_alpha=args.knn_alpha, knn_k=args.knn_k,
            knn_pool_wav=POOL if knn_pool is None else None,
            knn_pool=knn_pool,
        )
    T.mark(f"段{si}: infer() 完成")
    return generated_audio.squeeze().float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--prompt-budget", type=float, default=45.0,
                    help="每段参考素材时长预算（秒）。SVC 的 infer_segment 已支持 "
                         ">30s prompt（encode_long chunked 编码绕过 whisper 30s 硬顶，"
                         "commit 7b7f02d，实测 45s 效果更好），与整曲 prompt_ext.wav "
                         "(44.8s) 对齐")
    ap.add_argument("--silence-min", type=float, default=0.3,
                    help="候选切分静音 run 最短秒数（低于此的断句静音不切）")
    ap.add_argument("--pad", type=float, default=1.0,
                    help="切分点距静音 run 边缘的冗余秒数：run 时长 >= 2*pad "
                         "则双刀（前段 rs+pad 结束、后段 re-pad 开始，中间纯静音 "
                         "缓冲留零不进 SVC）；否则单刀中点均分")
    ap.add_argument("--seg-min", type=float, default=15.0,
                    help="段长下限（秒）。低于此会被合并/被 DP 跳过——太短 "
                         "SVC 内容特征不足。勿低于 8")
    ap.add_argument("--seg-max", type=float, default=25.0,
                    help="内容段长上限（秒，不含前后冗余）。硬约束：扩展后 "
                         "(seg+2*pad_ext) 必须 < svc-max-sec，否则 SoulX 把段二次切碎")
    ap.add_argument("--pad-ext", type=float, default=2.0,
                    help="每段前后各加的冗余秒数（一起送 SVC，拼接时裁掉）")
    ap.add_argument("--svc-max-sec", type=float, default=29.0,
                    help="送 SVC 的整段推理上限（秒）。SoulX 整段直推判定是 "
                         "gt_wav < max_seg_sec*sr；whisper 30s 硬顶，故取 29")
    ap.add_argument("--bands", nargs=2, type=float, default=None,
                    metavar=("LOW", "HIGH"),
                    help="启用音域分桶：先按静音切细段，再把相邻同音域段聚合，"
                         "组内用同一参考。传两个边界 Hz（如 260 370）。"
                         "主歌/副歌 ABAB 的歌用这个得到干净的「主歌组/副歌组」")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="vocal_segprompt.wav")
    ap.add_argument("--reuse-svc", action="store_true",
                    help="段内 generated.wav 已存在则跳过推理，只重做拼接")
    ap.add_argument("--fixed-prompt", action="store_true",
                    help="对照臂：同样的切段与逐段推理，但每段都用全曲 "
                         "prompt_ext.wav。用来把「音域匹配」和「切段+30s 短 "
                         "prompt」两个因素分开")
    ap.add_argument("--tmp-dir", default="segprompt_tmp")
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=0.5)
    ap.add_argument("--rescale-cfg", type=float, default=0.75)
    ap.add_argument("--num-overlaps", type=int, default=1)
    ap.add_argument("--knn-alpha", type=float, default=0.2)
    ap.add_argument("--knn-k", type=int, default=4)
    ap.add_argument("--pool-wav", default=None,
                    help="kNN 检索池 wav（覆盖默认 POOL）。sing.py 集成时传参考人声")
    ap.add_argument("--no-knn", action="store_true",
                    help="关闭 kNN（knn_alpha=0，跳过 pool 编码）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-fp16", action="store_true", help="关闭 fp16")
    args = ap.parse_args()

    d = ROOT / "data" / "songs" / args.song
    f0 = np.load(d / "target_f0.npy")
    lead, sr = sf.read(str(d / "sep2" / "lead.wav"), dtype="float32")
    cands = json.load(open(d / "cands_meta.json", encoding="utf-8"))
    tmp = d / args.tmp_dir
    tmp.mkdir(exist_ok=True)
    T = _T(tmp / "run.log")
    device = args.device
    use_fp16 = not args.no_fp16
    if args.no_knn:
        args.knn_alpha = 0.0

    if args.bands:
        groups = split_bands(f0, silence_min=args.silence_min,
                             seg_min=args.seg_min, seg_max=args.seg_max,
                             band_edges=tuple(args.bands), pad=args.pad)
        segs = [(a, b) for a, b, _ in groups]
        T.mark(f"音域分桶: {len(segs)} 组 (band_edges={args.bands}, "
               f"seg_min={args.seg_min}, seg_max={args.seg_max})")
    else:
        segs = split_silence(f0, args.silence_min, args.seg_min,
                             args.seg_max, args.pad)
        T.mark(f"段落切分: {len(segs)} 段 (seg_min={args.seg_min}, "
               f"seg_max={args.seg_max}, pad={args.pad})")

    # ---- 单进程：模型加载一次 + kNN pool 编码一次 ----
    T.mark("加载模型 + CUDA 初始化")
    config = load_config(str(CONFIG))
    model = SoulXSingerSVC(config).to(device)
    ckpt = torch.load(str(MODEL), weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if use_fp16 and device.startswith("cuda"):
        model.half()
        model.mel.float()
    model.eval()
    model.to(device)
    T.mark(f"模型加载完成 ({sum(p.numel() for p in model.parameters())/1e6:.0f}M)")

    knn_pool = None
    if args.knn_alpha > 0.0:
        T.mark("kNN pool 加载 + encode_long (一次)")
        pool_path = Path(args.pool_wav) if args.pool_wav else POOL
        pool_wav = load_wav(str(pool_path), config.audio.sample_rate).to(device)
        if use_fp16:
            pool_wav = pool_wav.half()
        with torch.no_grad():
            knn_pool = model.encode_long(pool_wav)
        T.mark(f"kNN pool 就绪 ({knn_pool.shape[1]} 帧)")

    extractor = F0Extractor(str(RMVPE), device=device, is_half=False,
                            max_duration=60.0, verbose=False)
    placed, meta = [], []  # placed: [(t0, audio)] 按原时间轴摆放
    total_sec = len(f0) / F0_RATE
    for si, (a, b) in enumerate(segs):
        med_f0 = seg_med_f0(f0, a, b)
        T.mark(f"段{si} [{a:.1f}-{b:.1f}] med_f0={med_f0:.0f}Hz")
        if not np.isfinite(med_f0) or med_f0 <= 0:
            T.mark(f"  无浊音，跳过")
            continue
        picked = (None if args.fixed_prompt else
                  pick_prompt(cands, med_f0, args.prompt_budget,
                              args.max_shift))
        if args.fixed_prompt:
            p_wav, p_f0 = d / "prompt_ext.wav", d / "prompt_ext_f0.npy"
            T.mark("  对照臂：用全曲 prompt_ext.wav")
        elif picked:
            p_wav, p_f0 = render_prompt(picked, tmp, f"seg{si}", extractor, T)
            T.mark(f"  参考 {len(picked)} 段 / shift "
                   + str(sorted({sh for _, sh in picked})))
        else:
            p_wav, p_f0 = d / "prompt_ext.wav", d / "prompt_ext_f0.npy"
            T.mark("  候选池无匹配，退回 prompt_ext.wav")

        # 段间冗余：SVC 输入是 [ext_t0, ext_t1]（前后各 +pad_ext），
        # 拼接只取 [a, b]（冗余裁掉）。记录 ext 窗口供后续对账。
        ext_t0 = max(a - args.pad_ext, 0.0)
        ext_t1 = min(b + args.pad_ext, total_sec)
        if ext_t1 - ext_t0 >= args.svc_max_sec:
            shrink = (ext_t1 - ext_t0) - (args.svc_max_sec - 1e-3)
            ext_t1 = max(ext_t0 + 1e-3, ext_t1 - shrink)
        assert ext_t1 - ext_t0 < args.svc_max_sec, \
            f"段{si} 扩展后仍 >= {args.svc_max_sec}s"
        seg_wav = tmp / f"seg{si}.wav"
        seg_f0 = tmp / f"seg{si}_f0.npy"
        sa, sb = int(round(ext_t0 * sr)), int(round(ext_t1 * sr))
        sf.write(str(seg_wav), lead[sa:sb], sr)
        np.save(str(seg_f0), f0[int(ext_t0 * F0_RATE): int(ext_t1 * F0_RATE)])
        out_dir = tmp / f"svc_seg{si}"
        out_dir.mkdir(exist_ok=True)
        # reuse 标记：记录扩展窗口，避免 --reuse-svc 复用旧算法（无冗余）的产物
        marker = out_dir / "window.json"
        stale = (not marker.exists())
        if not stale:
            try:
                wj = json.load(open(marker, encoding="utf-8"))
                stale = (wj.get("ext_t0") != ext_t0 or wj.get("ext_t1") != ext_t1
                         or wj.get("sr") != sr)
            except Exception:
                stale = True
        if args.reuse_svc and (out_dir / "generated.wav").exists() and not stale:
            T.mark("  已有产物，跳过推理")
        else:
            y = svc_infer(model, config, p_wav, seg_wav, p_f0, seg_f0,
                          knn_pool, device, use_fp16, args, T, si)
            gen = out_dir / "generated.wav"
            sf.write(str(gen), y, config.audio.sample_rate)
            json.dump({"ext_t0": ext_t0, "ext_t1": ext_t1, "sr": sr},
                      open(marker, "w", encoding="utf-8"))
            T.mark(f"  generated.wav 写出 ({len(y)/config.audio.sample_rate:.1f}s)")
        y, sr_g = sf.read(str(out_dir / "generated.wav"), dtype="float32")
        # 裁剪冗余：只保留 [a, b] 对应段（y 对齐到 ext_t0）
        keep_s = int(round((a - ext_t0) * sr_g))
        keep_e = keep_s + int(round((b - a) * sr_g))
        keep_s = max(0, keep_s)
        keep_e = min(keep_e, len(y))
        y_keep = y[keep_s:keep_e]
        meta.append({"seg": si, "t0": a, "t1": b, "med_f0": med_f0,
                     "ext_t0": ext_t0, "ext_t1": ext_t1,
                     "prompt": str(p_wav),
                     "n_picked": len(picked) if picked else 0,
                     "picked_idx": [c["idx"] for c, _ in picked]
                     if picked else [],
                     "shifts": sorted({sh for _, sh in picked})
                     if picked else []})
        placed.append((a, y_keep))

    # 摆回原时间轴：段间的间奏/长静音留零，输出与源等长，
    # 这样 target_f0.npy 的窗口标签、与伴奏的对齐、A/B 配窗都成立。
    sr_g = sr_g if placed else 24000
    total = int(round(len(f0) / F0_RATE * sr_g))
    out_audio = np.zeros(total, dtype="float32")
    for t0, y in placed:
        s = int(round(t0 * sr_g))
        e = min(s + len(y), total)
        out_audio[s:e] += y[: e - s]
    out_wav = d / args.out
    sf.write(str(out_wav), out_audio, sr_g)
    meta_path = d / ("segprompt_meta.json"
                     if args.out in ("vocal.wav", "vocal_segprompt.wav")
                     else args.out.replace(".wav", "_meta.json"))
    json.dump(meta, open(meta_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    T.mark(f"-> {out_wav} ({len(out_audio)/sr_g:.1f}s, {len(meta)} 段)")
    T.close()


if __name__ == "__main__":
    main()
