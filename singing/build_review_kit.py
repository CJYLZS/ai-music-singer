#!/usr/bin/env python3
"""导出人工校对包：把切好的全部片段 + ASR/已校对转写放进一个目录，供逐条订正。

为什么需要它：`build_s2_mix.py` 只负责切片，转写与筛除是它下游的手工步骤，而上一轮
（talksing_full）的筛除只留在提交记录里 —— 94 条切片里有 16 条被剔掉，理由散落在
对话里，无法复核也无法回滚。这个脚本把三件事固化成一个可编辑目录：

  1. **全部 94 条切片都出现**（含上一轮被剔掉的 16 条），文件名带时间戳与时长，
     按源音频时间排序，便于对着原始录音听。
  2. 每条给一行 TSV，四列：`status / clip / dur / text`。**status 是唯一的取舍开关**，
     不再靠"从 .list 里删行"表达剔除 —— 那样丢失了剔除理由。
  3. 转写有来源标记：已人工校对过的直接沿用（`kept`），只有新 ASR 的标 `asr`
     （需要逐条核）。

**切分是逐位可复现的**（同样 seed/参数下 `fine_cands` 对同一输入给出同名同内容的
片段，实测 55 条与旧集 md5 全等），所以本脚本不重切，直接读现有 `*_sliced/` 目录。

产物 `review.tsv` 的编辑约定（只改这个文件）:
  - `status` 列: `ok` = 收进数据集；`drop` = 剔除；`?` = 待定（视为 drop，但会单独报告）
  - `text` 列: 直接改成正确的转写。留空且 status=ok 会在校验时报错。
  - `reason` 列（可选，第 5 列）: 剔除理由，纯记录用，不进 .list
  - 不要动 `clip` 列；行序可任意，重复 clip 会报错。

校对完用 `--validate` 复核，再用 `--emit` 生成 `.list` 与数据集 json。

用法（GPT-SoVITS venv）:
  # 1) 导出校对包（默认从 talksing_full_sliced 读，写到 data/voices/review_<name>/）
  GPT-SoVITS/.venv/bin/python singing/build_review_kit.py export --name talksing_clean
  # 2) 人工编辑 data/voices/review_talksing_clean/review.tsv
  # 3) 校验
  GPT-SoVITS/.venv/bin/python singing/build_review_kit.py validate --name talksing_clean
  # 4) 生成数据集（写 data/voices/<name>.list + <name>_sliced/ + <name>_dataset.json）
  GPT-SoVITS/.venv/bin/python singing/build_review_kit.py emit --name talksing_clean
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC_SLICED = ROOT / "data/voices/talksing_full_sliced"
# 已人工校对过的转写来源，按优先级；命中即标 kept
CORRECTED_LISTS = [
    ROOT / "data/voices/talksing_full.list",
    ROOT / "data/voices/talksing_s2mix50.list",
    ROOT / "data/voices/girlish.list",
]
# 片段文件名里的偏移是各自源文件的采样点：talk 名来自 ref-vocal.wav(32k) 的切片，
# sing 名来自 165.166s 起的 44.1k 重采样临时文件。实测按此换算与文件时长逐条吻合。
TALK_SR = 32000
SING_SR = 44100
SING_OFFSET = 165.166

HEADER = ["status", "clip", "dur", "text", "reason"]


def read_list(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) == 4:
            out[Path(parts[0]).name] = parts[3].strip()
    return out


def resolve_src(args) -> Path:
    if getattr(args, "src", None):
        return Path(args.src)
    # export 时把源写进了 index.json；validate/emit 优先读它，避免默认指向旧的 talksing_full_sliced
    try:
        idx = ROOT / f"data/voices/review_{args.name}/index.json"
        if idx.exists():
            j = json.loads(idx.read_text(encoding="utf-8"))
            p = j.get("src_sliced")
            if p:
                return Path(p)
    except Exception:
        pass
    return SRC_SLICED


def clip_time(name: str) -> tuple[str, float, float]:
    """返回 (kind, 源起始秒, 源结束秒)。用于排序与人工定位原始录音。"""
    m = re.match(r"sing_(\d+)_(\d+)_(\d+)\.wav$", name)
    if m:
        return ("sing",
                int(m.group(2)) / SING_SR + SING_OFFSET,
                int(m.group(3)) / SING_SR + SING_OFFSET)
    # talk 切片有两种命名：talksing_full 是 talk_ref-vocal...，新切是 ref-vocal...
    m = re.match(r"(?:talk_)?ref-vocal\.wav_(\d+)_(\d+)\.wav$", name)
    if m:
        return ("talk", int(m.group(1)) / TALK_SR, int(m.group(2)) / TALK_SR)
    return ("other", 0.0, 0.0)


def asr_model(device: str):
    """本地缓存的 paraformer-large + fsmn-vad + punc，离线、可复现。

    种子必须在每次 generate 前重置：FunASR 的 VAD 消耗全局 RNG，否则同一文件的
    转写取决于它在批次里的位置（eval_cer.py 记录：58/55/54 字 vs 58/58/58 字）。
    """
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch

    cache = Path(os.path.expanduser("~/.cache/modelscope/models"))
    asr = cache / "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/snapshots/master"
    vad = cache / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/master"
    punc = cache / "iic--punc_ct-transformer_zh-cn-common-vocab272727-pytorch/snapshots/master"
    for d, what in ((asr, "paraformer-large"), (vad, "fsmn-vad"), (punc, "punc-ct")):
        if not (d / "model.pt").exists():
            sys.exit(f"缺少已缓存的 {what}: {d}/model.pt")

    def seed() -> None:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)

    seed()
    from funasr import AutoModel
    m = AutoModel(model=str(asr), vad_model=str(vad), punc_model=str(punc),
                  device=device, disable_update=True, disable_pbar=True)

    def run(path: Path) -> str:
        seed()
        r = m.generate(input=str(path), batch_size_s=300)
        return "".join(s.get("text", "") for s in r).strip()

    return run


def cmd_export(args) -> None:
    import soundfile as sf

    src = Path(args.src) if args.src else SRC_SLICED
    if not src.is_dir():
        sys.exit(f"找不到切片目录 {src}")
    out = ROOT / f"data/voices/review_{args.name}"
    tsv = out / "review.tsv"
    if tsv.exists() and not args.force:
        sys.exit(f"{tsv} 已存在（要重建加 --force；注意会覆盖你的编辑）")
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    corrected: dict[str, str] = {}
    for p in CORRECTED_LISTS:
        for k, v in read_list(p).items():
            corrected.setdefault(k, v)

    wavs = sorted(src.glob("*.wav"))
    if not wavs:
        sys.exit(f"{src} 里没有 wav")
    rows = []
    for p in wavs:
        kind, s, e = clip_time(p.name)
        info = sf.info(str(p))
        rows.append({"clip": p.name, "kind": kind, "start": s, "end": e,
                     "dur": round(info.duration, 2), "sr": info.samplerate})
    rows.sort(key=lambda r: (r["kind"] != "talk", r["start"]))

    need_asr = [r for r in rows if r["clip"] not in corrected]
    texts = {r["clip"]: corrected[r["clip"]] for r in rows if r["clip"] in corrected}
    src_of = {r["clip"]: "kept" for r in rows if r["clip"] in corrected}
    if need_asr and not args.no_asr:
        print(f"[asr] {len(need_asr)} 条无已校对转写，跑 ASR ...")
        run = asr_model(args.device)
        for i, r in enumerate(need_asr, 1):
            t = run(src / r["clip"])
            texts[r["clip"]] = t
            src_of[r["clip"]] = "asr"
            print(f"  [{i}/{len(need_asr)}] {r['clip']}  ->  {t}")
    else:
        for r in need_asr:
            texts[r["clip"]] = ""
            src_of[r["clip"]] = "asr"

    # 人工听审用的副本：文件名带序号/时间/时长，按时间排序，直接在播放器里顺序播
    for i, r in enumerate(rows, 1):
        nice = (f"{i:03d}_{r['kind']}_{r['start']:07.2f}s"
                f"_d{r['dur']:04.2f}_{src_of[r['clip']]}.wav")
        dst = clips_dir / nice
        if not dst.exists() or args.force:
            shutil.copy(src / r["clip"], dst)
        r["review_wav"] = nice

    lines = ["\t".join(HEADER)]
    for r in rows:
        lines.append("\t".join(["ok", r["clip"], f"{r['dur']:.2f}",
                                texts[r["clip"]], ""]))
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "index.json").write_text(json.dumps(
        {"src_sliced": str(src), "name": args.name,
         "sing_offset_sec": SING_OFFSET,
         "rows": [{**r, "text_source": src_of[r["clip"]]} for r in rows]},
        ensure_ascii=False, indent=1), encoding="utf-8")

    n_kept = sum(1 for v in src_of.values() if v == "kept")
    print(f"\n共 {len(rows)} 条：{n_kept} 条沿用已校对转写，{len(rows) - n_kept} 条是新 ASR")
    print(f"听审副本 -> {clips_dir}")
    print(f"待编辑   -> {tsv}")
    print("\n编辑约定：status 列填 ok / drop / ?（drop 与 ? 都不进数据集，? 会单独报告）；"
          "text 列改成正确转写；reason 列可写剔除理由。")
    print(f"改完跑：GPT-SoVITS/.venv/bin/python singing/build_review_kit.py validate --name {args.name}")


def load_tsv(name: str) -> tuple[Path, list[dict]]:
    out = ROOT / f"data/voices/review_{name}"
    tsv = out / "review.tsv"
    if not tsv.exists():
        sys.exit(f"找不到 {tsv}（先跑 export）")
    lines = [l for l in tsv.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        sys.exit(f"{tsv} 是空的")
    body = lines[1:] if lines[0].split("\t")[:2] == HEADER[:2] else lines
    rows = []
    for ln, line in enumerate(body, 2):
        f = line.split("\t")
        while len(f) < 5:
            f.append("")
        rows.append({"line": ln, "status": f[0].strip().lower(), "clip": f[1].strip(),
                     "dur": f[2].strip(), "text": f[3].strip(), "reason": f[4].strip()})
    return out, rows


def check(name: str, src: Path) -> tuple[list[dict], list[str]]:
    _, rows = load_tsv(name)
    errs: list[str] = []
    seen: set[str] = set()
    on_disk = {p.name for p in src.glob("*.wav")}
    for r in rows:
        if not r["clip"]:
            errs.append(f"L{r['line']}: clip 列为空")
            continue
        if r["clip"] in seen:
            errs.append(f"L{r['line']}: clip 重复 {r['clip']}")
        seen.add(r["clip"])
        if r["clip"] not in on_disk:
            errs.append(f"L{r['line']}: {src} 里没有 {r['clip']}")
        if r["status"] not in ("ok", "drop", "?"):
            errs.append(f"L{r['line']}: status 只能是 ok/drop/?，得到 {r['status']!r}")
        if r["status"] == "ok":
            if not r["text"]:
                errs.append(f"L{r['line']}: status=ok 但 text 为空 → {r['clip']}")
            elif not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", r["text"]):
                errs.append(f"L{r['line']}: text 无有效字符 → {r['text']!r}")
    missing = on_disk - seen
    if missing:
        errs.append(f"review.tsv 漏了 {len(missing)} 条切片：{sorted(missing)[:5]} ...")
    return rows, errs


def bucket_hist(durs: list[float]) -> dict:
    """DistributedBucketSampler 分桶（frames @ hop640/32k，1 frame = 20ms）。

    越界样本被 data_utils.py:1001 **静默丢弃**，所以必须在建集时就报出来。
    """
    bounds = [32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
              1300, 1400, 1500, 1600, 1700, 1800, 1900]
    h: dict[int, int] = {}
    dropped = 0
    for d in durs:
        fr = d * 32000 / 640
        idx = None
        for i in range(len(bounds) - 1):
            if bounds[i] < fr <= bounds[i + 1]:
                idx = i
                break
        if idx is None:
            dropped += 1
        else:
            h[idx] = h.get(idx, 0) + 1
    return {"buckets": h, "dropped_out_of_range": dropped}


def report(rows: list[dict], src: Path) -> dict:
    import soundfile as sf

    keep = [r for r in rows if r["status"] == "ok"]
    drop = [r for r in rows if r["status"] == "drop"]
    todo = [r for r in rows if r["status"] == "?"]
    stat = {"total": len(rows), "ok": len(keep), "drop": len(drop), "pending": len(todo)}
    per = {"talk": [], "sing": []}
    for r in keep:
        kind = clip_time(r["clip"])[0]
        d = sf.info(str(src / r["clip"])).duration
        per.setdefault(kind, []).append((r, d))
    durs = [d for v in per.values() for _, d in v]
    tot = sum(durs) or 1.0
    for k, v in per.items():
        s = sum(d for _, d in v)
        stat[k] = {"clips": len(v), "sec": round(s, 1), "share": round(s / tot, 4)}
    stat["total_sec"] = round(tot, 1)
    if durs:
        stat["dur_min"] = round(min(durs), 2)
        stat["dur_max"] = round(max(durs), 2)
        stat["bucket"] = bucket_hist(durs)
    # 复制倍数是阶梯函数 int(100/n)，99→2×(len 198) 而 100→1×(len 100)，步数腰斩。
    n = len(keep)
    stat["min_num_duplication"] = None if n >= 100 else max(2, int(100 / n))
    stat["dataset_len_after_duplication"] = n if n >= 100 else n * stat["min_num_duplication"]
    # S1 侧还有一道独立过滤：phoneme/sec 必须落在 [3,25]（AR/data/dataset.py:170）
    stat["chars_per_sec"] = {}
    if keep:
        cps = sorted((len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", r["text"])) /
                      max(sf.info(str(src / r["clip"])).duration, 0.01)) for r in keep)
        stat["chars_per_sec"] = {"min": round(cps[0], 2), "med": round(cps[len(cps) // 2], 2),
                                "max": round(cps[-1], 2),
                                "below_1.0": sum(1 for c in cps if c < 1.0)}
    return stat


def cmd_validate(args) -> None:
    src = resolve_src(args)
    rows, errs = check(args.name, src)
    if errs:
        print("发现问题：")
        for e in errs:
            print("  -", e)
    stat = report(rows, src)
    print("\n" + json.dumps(stat, ensure_ascii=False, indent=1))
    pending = [r for r in rows if r["status"] == "?"]
    if pending:
        print(f"\n还有 {len(pending)} 条 status='?' 未定（emit 时按 drop 处理）：")
        for r in pending[:20]:
            print(f"  {r['clip']}  {r['text']}")
    if errs:
        sys.exit(1)
    print("\n校验通过。生成数据集："
          f"\n  GPT-SoVITS/.venv/bin/python singing/build_review_kit.py emit --name {args.name}")


def cmd_emit(args) -> None:
    src = resolve_src(args)
    rows, errs = check(args.name, src)
    if errs and not args.force:
        print("校验未通过，拒绝生成（要强行生成加 --force）：")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    keep = [r for r in rows if r["status"] == "ok"]
    if not keep:
        sys.exit("没有 status=ok 的片段")

    out_dir = ROOT / f"data/voices/{args.name}_sliced"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() != src.resolve():
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
        for r in keep:
            shutil.copy(src / r["clip"], out_dir / r["clip"])
    else:
        # 源与输出同目录（5-15 等重切流）：只删被 drop/? 的，其余不动
        keep_names = {r["clip"] for r in keep}
        for f in list(out_dir.iterdir()):
            if f.is_file() and f.name not in keep_names:
                f.unlink()

    lst = ROOT / f"data/voices/{args.name}.list"
    lines = [f"data/voices/{args.name}_sliced/{r['clip']}|{args.name}|ZH|{r['text']}"
             for r in keep]
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stat = report(rows, src)
    stat["name"] = args.name
    stat["source_sliced"] = str(src)
    stat["review_tsv"] = str(ROOT / f"data/voices/review_{args.name}/review.tsv")
    stat["dropped"] = [{"clip": r["clip"], "text": r["text"], "reason": r["reason"]}
                       for r in rows if r["status"] in ("drop", "?")]
    meta = ROOT / f"data/voices/{args.name}_dataset.json"
    meta.write_text(json.dumps(stat, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({k: v for k, v in stat.items() if k != "dropped"},
                     ensure_ascii=False, indent=1))
    print(f"\n切片 -> {out_dir}  ({len(keep)} 条)")
    print(f".list -> {lst}")
    print(f"元数据 -> {meta}")
    if stat["dataset_len_after_duplication"] and len(keep) < 100:
        print(f"\n注意：{len(keep)} 条 < 100，复制倍数 {stat['min_num_duplication']}×，"
              f"dataset len = {stat['dataset_len_after_duplication']}。"
              f"训练日志里核对 wav_data_len 与此一致。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("export", "validate", "emit"):
        p = sub.add_parser(c)
        p.add_argument("--name", required=True, help="数据集名（决定 .list / _sliced 目录名）")
        p.add_argument("--src", default=None,
                       help=f"切片来源目录（默认 {SRC_SLICED}）")
        p.add_argument("--force", action="store_true")
        if c == "export":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--no-asr", action="store_true",
                           help="不跑 ASR，未校对的条目 text 留空自己填")
    args = ap.parse_args()
    {"export": cmd_export, "validate": cmd_validate, "emit": cmd_emit}[args.cmd](args)


if __name__ == "__main__":
    main()
