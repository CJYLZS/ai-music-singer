#!/usr/bin/env python3
"""人声自然度打分：SingMOS (singmos_pro, naturalness MOS) + NISQA (多维质量)。

用法:
  score_naturalness.py [--win 3.0] [--rms-floor-frac 0.01] [--batch 8]
                       [--nisqa-repo DIR] [--out report.json]
                       参考=data/voices/ref_talk_sing.wav \
                       原唱=data/songs/x/sep2/lead.wav \
                       合成=data/songs/x/vocal.wav

输出:
  --out 指向的 JSON（每文件聚合统计 + NISQA 分维度 + 逐窗 SingMOS CSV 路径）
  打印汇总表。

设计要点:
  - 统一 16kHz 单声道（SingMOS 训练分布）。
  - SingMOS 按 --win 秒非重叠窗打分，跳过低 RMS 窗（伴奏间隙/静音无意义），
    输出 mean/median/p10/p90/std 与分布。
  - NISQA 对整文件打分（模型内部自带 15s/5s 滑窗），输出
    mos/noi/col/dis/lou 五个维度。
"""
import argparse
import json
import os
import sys

import numpy as np
import librosa
import torch

SR = 16000

# 沙箱/磁盘策略：把 torch.hub 与 s3prl 的权重缓存重定向到本工具目录（默认 ~/.cache 不可写）
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(_TOOL_DIR, ".torch-cache"))
os.environ.setdefault("HF_HOME", os.path.join(_TOOL_DIR, ".hf-cache"))
import s3prl.util.download as _s3prl_download

_s3prl_download.set_dir(os.path.join(_TOOL_DIR, ".s3prl-cache"))


def load_windows(path, win, rms_floor_frac):
    """16k 单声道加载，切成非重叠窗；返回 (保留窗列表, 绝对窗序号列表, 总数, RMS p95)。"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    n = int(round(win * SR))
    wavs, rms = [], []
    for i in range(0, len(y) - n + 1, n):
        w = y[i:i + n]
        wavs.append(w)
        rms.append(float(np.sqrt(np.mean(w ** 2))))
    rms = np.asarray(rms)
    floor = max(1e-4, rms_floor_frac * float(np.percentile(rms, 95)))
    keep = rms >= floor
    kept = [wavs[i] for i in range(len(wavs)) if keep[i]]
    idxs = [i for i in range(len(wavs)) if keep[i]]
    return kept, idxs, len(wavs), float(np.percentile(rms, 95))


def singmos_predict(predictor, wavs, batch, device):
    """SingMOS 批量打分，返回逐窗分数数组（与 wavs 同序）。"""
    scores = []
    for i in range(0, len(wavs), batch):
        chunk = wavs[i:i + batch]
        maxlen = max(len(w) for w in chunk)
        t = torch.zeros(len(chunk), maxlen)
        for j, w in enumerate(chunk):
            t[j, :len(w)] = torch.from_numpy(w)
        lengths = torch.tensor([len(w) for w in chunk], dtype=torch.long)
        with torch.no_grad():
            s = predictor(t.to(device), lengths.to(device))
        scores.extend(s.cpu().numpy().tolist())
    return np.asarray(scores)


def summarize(name, scores):
    s = np.asarray(scores, dtype=float)
    return {
        "file": name,
        "n_windows": int(len(s)),
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "std": float(s.std()),
        "p10": float(np.percentile(s, 10)),
        "p90": float(np.percentile(s, 90)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="label=wav 对")
    ap.add_argument("--win", type=float, default=3.0)
    ap.add_argument("--rms-floor-frac", type=float, default=0.01,
                    help="静音窗阈值 = frac × 文件 RMS p95")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nisqa-repo",
                    default=os.path.join(_TOOL_DIR, "..", "nisqa"),
                    help="NISQA 官方仓库目录（含 run_predict.py 与 weights/）；缺省自动探测")
    ap.add_argument("--no-nisqa", action="store_true",
                    help="跳过 NISQA（默认评测流程只用 SingMOS；NISQA 是语音域模型，唱歌上仅供参考）")
    ap.add_argument("--out", default="natmos_report.json")
    ap.add_argument("--csv", default="", help="逐窗 SingMOS 分数 CSV 输出路径")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    items = []
    for s in args.inputs:
        if "=" in s:
            label, path = s.split("=", 1)
        else:
            label, path = os.path.basename(s), s
        if not os.path.exists(path):
            sys.exit(f"文件不存在: {path}")
        items.append((label, path))

    # ---- SingMOS ----
    print(f"[singmos] loading predictor (device={args.device}) ...", flush=True)
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True
    ).to(args.device).eval()
    predictor.eval()

    report = {"config": vars(args), "files": {}}
    csv_lines = ["label,window_idx,singmos_mos"]
    for label, path in items:
        wavs, idxs, total, rms95 = load_windows(path, args.win, args.rms_floor_frac)
        print(f"[singmos] {label}: {total} 窗, 保留 {len(idxs)} (RMS p95={rms95:.4f})", flush=True)
        scores = singmos_predict(predictor, wavs, args.batch, args.device)
        for i, sc in zip(idxs, scores):
            csv_lines.append(f"{label},{i},{sc:.4f}")
        report["files"][label] = summarize(label, scores)
        report["files"][label]["n_total_windows"] = total
        report["files"][label]["n_skipped_silence"] = total - len(idxs)

    # ---- NISQA ----
    nisqa_out = {}
    if not args.no_nisqa and os.path.exists(args.nisqa_repo):
        print("[nisqa] running ...", flush=True)
        for model_name in ("nisqa", "nisqa_tts"):
            nisqa_out[model_name] = {}
            for label, path in items:
                nisqa_out[model_name][label] = nisqa_predict_file(
                    args.nisqa_repo, model_name, path)
    report["nisqa"] = nisqa_out

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("\n".join(csv_lines) + "\n")
        report["csv"] = args.csv

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 打印汇总表 ----
    print("\n================ SingMOS naturalness MOS ================")
    print(f"{'文件':<8}{'窗数':>6}{'mean':>8}{'median':>9}{'p10':>8}{'p90':>8}{'std':>7}")
    for label, d in report["files"].items():
        print(f"{label:<8}{d['n_windows']:>6}{d['mean']:>8.3f}{d['median']:>9.3f}"
              f"{d['p10']:>8.3f}{d['p90']:>8.3f}{d['std']:>7.3f}")
    for model_name, d in nisqa_out.items():
        print(f"\n================ NISQA-{model_name} ================")
        keys = [k for k in next(iter(d.values())).keys() if k != "file"]
        print(f"{'文件':<8}" + "".join(f"{k:>8}" for k in keys))
        for label, dd in d.items():
            print(f"{label:<8}" + "".join(f"{dd[k]:>8.3f}" for k in keys))
    print(f"\n完整报告: {args.out}")


def nisqa_predict_file(repo_dir, model_name, path):
    """用 NISQA 官方 nisqaModel 对单个 wav 打分（内部自带滑窗聚合）。
    返回 dict（列名 -> float）。
    - checkpoint 参数会被我们传入的 args 覆盖（checkpoint['args'].update(args)），
      所以 ms_max_segments 可以调大（默认 1300 ≈ 52s 上限，长文件会炸）。
    - 沙箱 /tmp 每进程隔离，临时文件放工作区。"""
    import csv as csvmod
    import shutil
    import uuid
    from nisqa.NISQA_model import nisqaModel

    y, _ = librosa.load(path, sr=SR, mono=True)
    td = os.path.join(_TOOL_DIR, ".tmp", f"{model_name}_{uuid.uuid4().hex[:8]}")
    os.makedirs(td, exist_ok=True)
    try:
        wav_path = os.path.join(td, "input.wav")
        import soundfile as sf
        sf.write(wav_path, y, SR)
        out_dir = os.path.join(td, "out")
        os.makedirs(out_dir, exist_ok=True)
        args = {
            "mode": "predict_file",
            "pretrained_model": os.path.join(repo_dir, "weights", f"{model_name}.tar"),
            "deg": wav_path,
            "output_dir": out_dir,
            "ms_channel": 0,
            "ms_max_segments": 200000,   # 覆盖 checkpoint 的 1300（52s 上限）
            "tr_bs_val": 1,
            "tr_num_workers": 0,
        }
        nisqaModel(args).predict()
        csv_path = os.path.join(out_dir, "NISQA_results.csv")
        with open(csv_path) as f:
            row = next(csvmod.DictReader(f))
        skip = {"file", "deg", "model"}
        return {k: float(v) for k, v in row.items() if k not in skip}
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
