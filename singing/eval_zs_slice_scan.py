#!/usr/bin/env python3
"""零样本参考切片选择扫描：从 ref_talk_sing 前 162s 里挑「哪一段 + 多长」做零样本参考最佳。

两步走（省时关键：预筛不跑 TTS，直接比切片 embedding 到整条参考嗓音的距离）：
  scan   —— 廉价预筛：对候选 (位置×长度) 切片直接算 ERes2NetV2 embedding 与整条参考的
            余弦相似度，越接近整条嗓音"中心"的切片，零样本克隆理论上越稳。
  verify —— 对选出的切片真正跑 gpt-zs 零样本生成（同一段长文本），算
            SIM vs 整条参考 与  SIM vs 当前 6.37s 生产切片，定夺最佳。

用法（GPT-SoVITS venv）:
  GPT-SoVITS/.venv/bin/python singing/eval_zs_slice_scan.py scan
  GPT-SoVITS/.venv/bin/python singing/eval_zs_slice_scan.py verify \
      --slice 6.4@12.3 --slice 8@35.0 --slice 6@100.0   (len@start 格式)
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path("/home/rookie/voice")
REF_SRC = ROOT / "data/voices/ref_talk_sing.wav"          # 537.8s, talk+sing
TALK_LIMIT_S = 162.0                                       # 只用前 162s 说话段
CUR_SLICE = ROOT / "data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav"
GIRLH_DIR = ROOT / "data/voices/girlish_sliced"            # 按句切好的 22 条句子切片
LONG_TEXT = ROOT / "data/songs/情歌/mv/tts_eval_long_text.txt"
OUT = ROOT / "data/songs/情歌/mv/zs_slice_scan"

LENGTHS = [3.0, 5.0, 6.4, 8.0, 10.0, 12.0]


def voiced_runs(y, sr, limit_s, hop=0.02, rms_frac=0.06, min_gap=0.4, min_len=2.0):
    """RMS 能量 VAD：返回 [(s,e),...] 秒级连通语音段（在 limit_s 内）。"""
    import numpy as np
    n = int(hop * sr)
    rms = [float(np.sqrt(np.mean(y[i:i + n] ** 2))) for i in range(0, len(y) - n, n)]
    rms = np.asarray(rms)
    thr = max(1e-3, rms_frac * float(np.percentile(rms, 95)))
    v = rms > thr
    runs, i = [], 0
    while i < len(v):
        if not v[i]:
            i += 1
            continue
        j = i
        while j < len(v) and v[j]:
            j += 1
        runs.append((i * hop, min(j * hop, limit_s)))
        i = j
    # 合并 < min_gap 的间隙
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged if m[1] - m[0] >= min_len]


def pick_candidates(runs, n_max=12):
    """在 162s 内均匀取样 n_max 个语音段起点。"""
    if len(runs) <= n_max:
        return [s for s, _ in runs]
    starts = sorted(s for s, _ in runs)
    # 均匀取 ntos：按累积语音覆盖取
    total = max(starts[-1], 1.0)
    picks = []
    for i in range(n_max):
        tgt = total * (i + 0.5) / n_max
        picks.append(min(starts, key=lambda s: abs(s - tgt)))
    # 去重保序
    out, seen = [], set()
    for p in picks:
        ks = round(p, 1)
        if ks not in seen:
            seen.add(ks)
            out.append(p)
    return out[:n_max]


def make_slices(runs):
    """(pos × len) 网格 → [{pos,len,path}]，切片都完整落在语音段内。"""
    slices = []
    for pos in pick_candidates(runs):
        run = next(((s, e) for s, e in runs if s <= pos < e), None)
        if run is None:
            continue
        s, e = run
        for L in LENGTHS:
            if s + L <= e + 1e-6:          # 从 pos 起 L 秒能整段落在语音内
                slices.append({"pos": round(pos, 1), "len": float(L),
                               "label": f"t{pos:.1f}_l{L:g}"})
    return slices


def write_slice32(y, sr, s, e, path):
    import numpy as np
    import soundfile as sf
    seg = y[int(s * sr):int(e * sr)]
    if y.ndim > 1:
        seg = seg.mean(axis=1)
    # 44.1k -> 32k
    import torch, torchaudio
    t = torch.from_numpy(seg.astype(np.float32))
    t = torchaudio.functional.resample(t, sr, 32000) if sr != 32000 else t
    sf.write(path, t.numpy(), 32000, subtype="PCM_16", format="WAV")


def full_ref_emb(sv):
    import sys
    sys.path.insert(0, str(ROOT))
    import eval_sim
    import torch
    emb = sv.emb(eval_sim.load16k(str(REF_SRC)))
    torch.cuda.empty_cache()
    return emb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "verify", "dirscan"])
    ap.add_argument("--slice", action="append", default=[], dest="slices",
                    help="verify 用: len@start 秒，可多次（如 6@35.0）")
    ap.add_argument("--top", type=int, default=8, help="scan 后自动 verify 前 top 个")
    ap.add_argument("--force", action="store_true", help="重新生成（默认复用已有 gen）")
    ap.add_argument("--no-verify-baseline", action="store_true",
                    help="verify 不跑当前 6.37s 生产切片基线")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import sys
    sys.path.insert(0, str(ROOT))
    import eval_sim

    print("加载参考源 + VAD ...", flush=True)
    y, sr = sf.read(str(REF_SRC), dtype="float32")
    n_talk = int(TALK_LIMIT_S * sr)
    y = y[:n_talk]
    runs = voiced_runs(y, sr, TALK_LIMIT_S)
    print(f"前 {TALK_LIMIT_S:.0f}s 检出 {len(runs)} 个语音段:")
    for s, e in runs:
        print(f"  [{s:6.1f} - {e:6.1f}]  {e - s:5.1f}s")

    sv = eval_sim.SV(device="cuda:0")
    ref_emb = full_ref_emb(sv)

    def sim_vs_full(slice_path, emb16):
        import numpy as np
        return float(eval_sim.cos(sv.emb(eval_sim.load16k(str(slice_path))), ref_emb))

    if args.step == "scan":
        cands = make_slices(runs)
        print(f"\n[scan] {len(cands)} 个候选 (pos×len)，直接比切片 embedding vs 整条参考 ...")
        rows = []
        for c in cands:
            p = OUT / "slices" / f"{c['label']}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                write_slice32(y, sr, c["pos"], c["pos"] + c["len"], p)
            sc = sim_vs_full(p, None)
            rows.append({**c, "sim_vs_full": round(sc, 4), "path": str(p)})
        rows.sort(key=lambda r: -r["sim_vs_full"])
        (OUT / "scan_report.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n{'位置s':<10}{'长度s':<8}{'SIM vs 整条参考':>18}")
        for r in rows[:40]:
            print(f"t{r['pos']:<8.1f}{r['len']:<8.1g}{r['sim_vs_full']:>18.4f}")
        top_cmd = " ".join(f"{r['len']:g}@{r['pos']:.1f}" for r in rows[:args.top])
        top5 = ", ".join("%gs/%ss=%.4f" % (r['len'], r['pos'], r['sim_vs_full'])
                         for r in rows[:5])
        print(f"\nTop 切片: [{top5}]")
        print(f"完整表: {OUT / 'scan_report.json'}")
        print(f"\n后接: python3 singing/eval_zs_slice_scan.py verify --slice "
              f"{top_cmd.replace(' ', ' --slice ')}")

    if args.step == "dirscan":
        # 把 girlish_sliced 里按句切好的全部切片逐一当零样本参考，长文本生成 + SIM
        gl = [l.rstrip("\n").split("|") for l in
              (ROOT / "data/voices/girlish.list").read_text(encoding="utf-8").splitlines()]
        gl_tr = {Path(g[0]).name: g[3].strip() for g in gl if len(g) == 4 and len(g[3].strip())}
        files = sorted(GIRLH_DIR.glob("*.wav"),
                       key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1)))
        print(f"[dirscan] {len(files)} 条句子切片 × 长文本 gpt-zs 生成 + SIM ...")
        cur_emb = sv.emb(eval_sim.load16k(str(CUR_SLICE)))
        rows = []
        for i, p in enumerate(files, 1):
            label = f"s{i:02d}"
            gen = OUT / "dir" / f"{label}.wav"
            gen.parent.mkdir(parents=True, exist_ok=True)
            if not gen.exists() or args.force:
                ref_text = gl_tr.get(p.name, "")
                import subprocess
                cmd = [str(ROOT / "GPT-SoVITS/.venv/bin/python"),
                       str(ROOT / "GPT-SoVITS/infer.py"),
                       "-t", "@" + str(LONG_TEXT.resolve()),
                       "-o", str(gen.resolve()),
                       "--gpt", str(ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/"
                                            "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
                       "--sovits", str(ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"),
                       "--ref", str(p.resolve()), "--speed", "1.0", "-q"]
                if ref_text:
                    cmd += ["--ref-text", ref_text]
                with open(OUT / f"tts_{label}.log", "w") as f:
                    r = subprocess.run(cmd, cwd=str(ROOT / "GPT-SoVITS"),
                                       stdout=f, stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    print(f"  !! {label} {p.name} 生成失败"); continue
            wav = eval_sim.load16k(str(gen))
            w = float(eval_sim.cos(sv.emb(wav), ref_emb))
            wlen = 3 * 16000
            ws = np.array([eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
                           for s in range(0, len(wav) - wlen + 1, wlen)])
            vs_cur = float(eval_sim.cos(sv.emb(wav), cur_emb))
            dur = sf.info(str(p)).frames / sf.info(str(p)).samplerate
            rows.append({"label": label, "file": p.name, "dur": round(dur, 1),
                         "ref_text": gl_tr.get(p.name, "")[:20],
                         "sim_full_whole": round(w, 4), "sim_full_win3": round(float(ws.mean()), 4),
                         "win3_n": len(ws), "sim_cur_whole": round(vs_cur, 4)})
            print(f"  {label} {p.name} {dur:5.1f}s: whole={w:.4f} win3={ws.mean():.4f} "
                  f"vs6.37s={vs_cur:.4f}")
        rows.sort(key=lambda r: -r["sim_full_win3"])
        (OUT / "dirscan_report.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nTop5: " + ", ".join(f"{r['label']}/{r['dur']}s={r['sim_full_win3']}" for r in rows[:5]))
        print(f"结果: {OUT / 'dirscan_report.json'}")

    else:  # verify
        runs_dict = {s: (s, e) for s, e in runs}
        items = []
        for sp in args.slices:
            if "@" not in sp:
                sys.exit(f"--slice 需 len@start，收到 {sp!r}")
            L, pos = float(sp.split("@")[0]), float(sp.split("@")[1])
            run = next((ss for ss in runs if ss[0] - 0.05 <= pos < ss[1] + 0.05), None)
            if run is None:
                sys.exit(f"start {pos}s 不在任何语音段内（VAD 运行表见上）")
            items.append((f"t{pos:.1f}_l{L:g}", L, pos))
        if not args.no_verify_baseline:
            items.append(("cur6.37s", 6.37, -1))
        print(f"\n[verify] 跑 {len(items)} 个切片 × 同一长文本 gpt-zs 生成 + SIM ...")

        results = []
        # ---- 参考转写：先查 girlish.list，查不到就 ASR（paraformer，离线+种子）并缓存 ----
        gl = [l.rstrip("\n").split("|") for l in
              (ROOT / "data/voices/girlish.list").read_text(encoding="utf-8").splitlines()]
        ref_cache = {}
        rcp = OUT / "reftext_cache.json"
        if rcp.exists():
            ref_cache = json.loads(rcp.read_text(encoding="utf-8"))
        asr_model = None

        def ref_text_for(path: Path) -> str:
            nonlocal asr_model
            name = path.name
            if name in ref_cache:
                return ref_cache[name]
            for parts in gl:
                if len(parts) == 4 and Path(parts[0]).name == name:
                    ref_cache[name] = parts[3].strip()
                    return ref_cache[name]
            # ASR 兜底
            import os as _os
            _os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            if asr_model is None:
                import random
                import torch as _torch
                import numpy as _np
                _torch.manual_seed(0); _torch.cuda.manual_seed_all(0)
                _torch.backends.cudnn.deterministic = True; _torch.backends.cudnn.benchmark = False
                _np.random.seed(0); random.seed(0)
                from funasr import AutoModel
                asr_dir = _os.path.expanduser(
                    "~/.cache/modelscope/models/iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/snapshots/master")
                vad_dir = _os.path.expanduser(
                    "~/.cache/modelscope/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/master")
                asr_model = AutoModel(model=asr_dir, vad_model=vad_dir, device="cuda:0",
                                      disable_update=True, disable_pbar=True)
            import soundfile as _sf
            import torch as _torch
            import torchaudio as _ta
            x, sr = _sf.read(str(path), dtype="float32")
            if sr != 16000:
                x = _ta.functional.resample(_torch.from_numpy(x).float(), sr, 16000).numpy()
            tmp16 = (OUT / f"_asr_{name}.wav")
            _sf.write(tmp16, x, 16000)
            r = asr_model.generate(input=str(tmp16), batch_size_s=300)
            text = "".join(s.get("text", "") for s in r)
            ref_cache[name] = text.strip()
            return ref_cache[name]

        for label, L, pos in items:
            if pos >= 0:
                p = OUT / "slices" / f"{label}.wav"
                if not p.exists():
                    write_slice32(y, sr, pos, pos + L, p)
            else:
                p = CUR_SLICE
            gen = OUT / "gen" / f"{label}.wav"
            gen.parent.mkdir(parents=True, exist_ok=True)
            if not gen.exists() or args.force:
                print(f"  [tts] {label} ref={p.name} ...", flush=True)
                import subprocess
                ref_text = ref_text_for(p)
                cmd = [str(ROOT / "GPT-SoVITS/.venv/bin/python"),
                       str(ROOT / "GPT-SoVITS/infer.py"),
                       "-t", "@" + str(LONG_TEXT.resolve()),
                       "-o", str(gen.resolve()),
                       "--gpt", str(ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/"
                                            "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
                       "--sovits", str(ROOT / "GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"),
                       "--ref", str(p.resolve()), "--speed", "1.0", "-q"]
                if ref_text:
                    cmd += ["--ref-text", ref_text]
                with open(OUT / f"tts_{label}.log", "w") as f:
                    r = subprocess.run(cmd, cwd=str(ROOT / "GPT-SoVITS"),
                                       stdout=f, stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    print(f"  !! {label} 生成失败"); continue
        rcp.write_text(json.dumps(ref_cache, ensure_ascii=False, indent=1), encoding="utf-8")
        cur_emb = sv.emb(eval_sim.load16k(str(CUR_SLICE)))
        results = []
        for label, L, pos in items:
            p = CUR_SLICE if pos < 0 else OUT / "slices" / f"{label}.wav"
            gen = OUT / "gen" / f"{label}.wav"
            import numpy as np
            wav = eval_sim.load16k(str(gen))
            w = eval_sim.cos(sv.emb(wav), ref_emb)
            wlen = 3 * 16000
            ws = np.array([eval_sim.cos(sv.emb(wav[s:s + wlen]), ref_emb)
                           for s in range(0, len(wav) - wlen + 1, wlen)])
            vs_cur = float(eval_sim.cos(sv.emb(wav), cur_emb))
            results.append({"label": label, "sim_full_whole": round(float(w), 4),
                            "sim_full_win3": round(float(ws.mean()), 4),
                            "win3_n": len(ws), "sim_cur_whole": round(vs_cur, 4)})
            print(f"  {label}: SIM vs整条 whole={w:.4f} win3={ws.mean():.4f} ({len(ws)}窗)   "
                  f"vs6.37s切片={vs_cur:.4f}")
        results.sort(key=lambda r: -r["sim_full_win3"])
        (OUT / "verify_report.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n结果: {OUT / 'verify_report.json'}")


if __name__ == "__main__":
    main()
