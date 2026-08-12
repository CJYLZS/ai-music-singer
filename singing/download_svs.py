#!/usr/bin/env python3
"""Download the SoulX-Singer **SVS** weights (score-control: MIDI + lyrics, no source vocal).

Why this exists separately from the SVC path already in use:
  `sing.py` runs SoulX-Singer-**SVC** (`model-svc.pt`), which needs the original
  singer's dry vocal for content features. The score-control mode lives in a
  *different* checkpoint (`model.pt`) and needs no source audio at all — see
  soulxsinger/models/soulxsinger.py:121 where control=="score" sets gt_f0=None.

Resume strategy: fixed "<target>.part" name + `curl -C -`, verified against the
repo's LFS sha256 before promotion. NOT huggingface_hub's snapshot_download,
which renegotiates its `.incomplete` suffix every process start and therefore
restarts from byte 0 (this cost ~10GB of redundant transfer on echomimic3).
Idempotent: complete files are skipped.

Deliberately NOT downloaded (3.42 GB saved), with reasons:
  - mel-band-roformer-karaoke (1.72 GB) + dereverb (0.91 GB): only used by
    PreprocessPipeline(vocal_sep=True). Our own `sing.py --stages sep,ref-sep`
    already produces dry vocals, and the karaoke ckpt is byte-identical to the
    one in GPT-SoVITS/tools/uvr5/uvr5_weights/. Pass vocal_sep=False.
  - parakeet-tdt-0.6b-v2 (2.47 GB): English ASR only. LyricTranscriber loads it
    lazily per language; Mandarin goes through the seaco paraformer below.
  Both become necessary only if you separate inside this pipeline or do English.
"""
import hashlib
import os
import subprocess
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_ROOT = os.path.join(ROOT, "SoulX-Singer", "pretrained_models")

SVS = "Soul-AILab/SoulX-Singer"
PRE = "Soul-AILab/SoulX-Singer-Preprocess"

# (repo_id, repo_filename, path relative to DEST_ROOT)
TARGETS = [
    # --- the SVS model itself (score + melody control) ---
    (SVS, "model.pt", "SoulX-Singer/model.pt"),
    (SVS, "config.yaml", "SoulX-Singer/config.yaml"),

    # --- ROSVOT note transcription: turns a waveform into note events. ---
    # Paths are hard-coded in preprocess/pipeline.py:51-52; additionally
    # note_transcription/model.py:294 derives the pitch extractor as
    # `<ckpt>/../../rmvpe/model.pt`, i.e. rosvot/rmvpe/model.pt is required and
    # is a DIFFERENT file from the top-level rmvpe/rmvpe.pt below.
    (PRE, "rosvot/rosvot/model.pt", "SoulX-Singer-Preprocess/rosvot/rosvot/model.pt"),
    (PRE, "rosvot/rosvot/config.yaml", "SoulX-Singer-Preprocess/rosvot/rosvot/config.yaml"),
    (PRE, "rosvot/rwbd/model.pt", "SoulX-Singer-Preprocess/rosvot/rwbd/model.pt"),
    (PRE, "rosvot/rwbd/config.yaml", "SoulX-Singer-Preprocess/rosvot/rwbd/config.yaml"),
    (PRE, "rosvot/rmvpe/model.pt", "SoulX-Singer-Preprocess/rosvot/rmvpe/model.pt"),

    # --- Mandarin lyric transcription (seaco paraformer, via funasr) ---
    # Not the same model as the paraformer-large already in ~/.cache/modelscope.
    (PRE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pt",
     "SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pt"),
    (PRE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml",
     "SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/config.yaml"),
    (PRE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/configuration.json",
     "SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/configuration.json"),
    (PRE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn",
     "SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/am.mvn"),
    (PRE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json",
     "SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json"),
    (PRE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/seg_dict",
     "SoulX-Singer-Preprocess/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/seg_dict"),
]


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_meta():
    """(repo_id, filename) -> (size, sha256). sha256 is None for non-LFS files."""
    api = HfApi()
    meta = {}
    for repo in sorted({t[0] for t in TARGETS}):
        info = api.model_info(repo, files_metadata=True)
        for s in info.siblings:
            meta[(repo, s.rfilename)] = (s.size, s.lfs.sha256 if s.lfs else None)
    return meta


def download(repo_id, repo_file, dest):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{repo_file}"
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have:
        print(f"    resuming from {have / 1e9:.2f} GB")
    subprocess.run(
        ["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "3",
         "--progress-bar", "-o", part, url],
        check=True,
    )
    return part


def main():
    meta = fetch_meta()
    todo, skipped = [], 0

    for repo_id, repo_file, rel in TARGETS:
        dest = os.path.join(DEST_ROOT, rel)
        size, want_hash = meta.get((repo_id, repo_file), (None, None))
        if size is None:
            sys.exit(f"! {repo_id}/{repo_file} not found in repo listing")
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            skipped += 1
            continue
        todo.append((repo_id, repo_file, rel, dest, size, want_hash))

    total = sum(t[4] or 0 for t in todo)
    print(f"already complete: {skipped} file(s)")
    print(f"to download:      {len(todo)} file(s), {total / 1e9:.2f} GB\n")

    for i, (repo_id, repo_file, rel, dest, size, want_hash) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {rel}  ({(size or 0) / 1e9:.2f} GB)")
        part = download(repo_id, repo_file, dest)

        got_size = os.path.getsize(part)
        if size and got_size != size:
            sys.exit(f"  ! size mismatch: got {got_size}, want {size}")
        if want_hash:
            got = sha256_file(part)
            if got != want_hash:
                sys.exit(f"  ! sha256 mismatch\n    got  {got}\n    want {want_hash}")
            print("  sha256 ok")
        os.replace(part, dest)
        print("  done\n")

    print("=== all SVS weights present and verified ===")
    print(f"model root: {DEST_ROOT}/")


if __name__ == "__main__":
    main()
