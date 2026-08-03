# AGENTS.md — voice workspace

## Workspace layout

The main projects sit side by side at the workspace root. **Their dependencies genuinely conflict** (torch 2.13+cu130 vs 2.6+cu124, `gradio<5` vs 6.x, `transformers<=4.50` vs 4.46), so each carries its own `.venv`. Never install across them. `tools/` holds self-contained helper toolkits with their own venvs too.

```
voice/
├── GPT-SoVITS/       TTS / voice cloning for SPEECH. The git fork (branch main).
│                     Its own .venv (torch 2.13+cu130). Entrypoints: webui.py, infer.py, train.py
├── SoulX-Singer/     Singing voice conversion (SVC). Apache-2.0.
│                     .venv = torch 2.6+cu124. Converts a song's vocal to a target timbre.
├── seed-vc/          Zero-shot VC; range extender fallback (--ext-method seedvc), see tools/pcnsf.
│                     .venv = torch 2.6+cu124. GPL-3.0 (viral — do not redistribute linked).
├── tools/pcnsf/      PC-NSF-HiFiGAN pitch-transposition toolkit (scripts/pitch_transpose.py).
│                     .venv = torch 2.13 CPU-only (~500MB, 无需 CUDA). DEFAULT range extender.
├── tools/pcnsf_hifigan/  PC-NSF-HiFiGAN 44.1k weights (config.json + model.ckpt), 2025.02 release
├── tools/yingmusic/  YingMusic-SVC (zero-shot SVC, seed-vc fork + F0-aware timbre adaptor).
│                     .venv = torch 2.6+cu124. Experimental; see "SVC engine A/B" below.
├── data/             Shared audio. voices/ = speaker source material; songs/ = songs, stems, SVC output
├── logs/             GPT-SoVITS training features + checkpoints
├── AGENTS.md         this file
└── eval_sim.py       speaker-similarity scoring (pitch-bucketed), see "Measuring similarity"
```

`data/`, `logs/`, and `AGENTS.md` live at the root and are **symlinked into `GPT-SoVITS/`** so its relative paths (`data/voices/...`, `logs/<name>/`) keep working. Consequence: `git status` inside `GPT-SoVITS/` reports `T AGENTS.md` (type changed to symlink) — that is expected, do not commit it.

Cross-project scripts under `SoulX-Singer/work/` reference the SV model by absolute path
(`/home/rookie/voice/GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/...`). Moving the workspace breaks them.

## The singing pipeline (verified end to end)

Speech reference → sung output in the target timbre. **No single model does this**; it is two-stage.

```
song.flac
  └─ BS-RoFormer            → vocals + instrumental          (GPT-SoVITS/tools/uvr5_cli.py)
       └─ mel-band-roformer-karaoke → LEAD vocal + backing vocals
            └─ SoulX-Singer SVC  ← prompt: speech reference   → converted vocal
                                    ↑
                  prompt_ext = original prompt (timbre anchor, ≥55%)
                             + PC-NSF-HiFiGAN transposed segments (≤+5st each,
                               allocated by per-band time-share gap vs the song)
```

Measured on a 257s Chinese song with a 163s female speech reference (median F0 247Hz):

| variant | mean SIM | median SIM | >370Hz segments |
|---|---|---|---|
| narrow prompt / contaminated vocal | 0.5960 | 0.6151 | 0.5592 |
| wide prompt / contaminated | 0.6093 | 0.6250 | 0.5697 |
| wide prompt / clean lead | 0.6084 | 0.6236 | 0.5772 |
| **pitch-extended prompt / clean lead** | **0.6119** | **0.6363** | **0.5815** |

### Non-obvious findings from this pipeline

- **Pitch distance drives similarity.** Bucketing 3s windows by semitone distance from the prompt: 0-2st → 0.646, 4-6st → 0.593, 6-8st → 0.539. Correlation −0.48. The single biggest quality lever is keeping the melody near the reference's register.
- **Prompt length caps at 30s, and it is NOT chunked.** `soulxsinger/models/modules/whisper_encoder.py:48` hard-truncates to 3000 mel frames. Audio past 30s is silently discarded. `PROMPT_MAX_SEC_DEFAULT` in `webui_svc.py` is a *separate* soft limit that the CLI does not apply.
- **Extend the reference's range with a neural vocoder, never with a DSP pitch shifter.** A/B'd 2026-08 on this ref/song (eval_sim vs the real voice, unshifted audio as ground truth): rubberband (even `formant=preserved`) scored 0.7728 vs **seed-vc 0.9067** on a +5st speech clip — but **PC-NSF-HiFiGAN now beats seed-vc** on the full 24.8s layered `prompt_ext`: mean 0.7922 / median 0.8061 vs seed-vc 0.7711 / 0.7799, and even on unshifted segments (0-2st bucket 0.8448 vs 0.8170). Also ~2× faster (runs CPU-only). It is the DEFAULT ext engine (`sing.py --ext-method nsf`); seed-vc stays as fallback. seed-vc's mechanics for why this family works: `adjust_f0_semitones` (`inference.py:243`) multiplies the *extracted F0 curve*, then the vocoder resynthesizes — the speaker embedding comes from undistorted audio. +5 semitones is the sweet spot; +10 degrades to 0.7845, i.e. back to DSP quality.
  ```bash
  # PC-NSF-HiFiGAN（默认引擎，CPU 即可）:
  tools/pcnsf/.venv/bin/python tools/pcnsf/scripts/pitch_transpose.py ref.wav \
      --transpose 5 --method nsfhifigan \
      --config tools/pcnsf_hifigan/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json \
      --ckpt  tools/pcnsf_hifigan/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt \
      --pitchalgo rmvpe --rmvpeckpt SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt \
      --out out/
  # seed-vc（回退引擎，须以 seed-vc/ 为 CWD）:
  cd seed-vc && .venv/bin/python inference.py --source ref.wav --target ref.wav \
      --f0-condition True --auto-f0-adjust False --semi-tone-shift 5 --fp16 True
  ```
- **Keep the original prompt as a timbre anchor; APPEND transposed segments, never replace.** The first `w_ext.py` shipped a `prompt_ext` that was 100% vocoder-resynthesized (every selected segment got transposed) and — because it reused the original prompt's length — came out at exactly 19.56s, wasting 9.4s of the 29s budget. Cost is measurable: even at 0 semitones a pass through the vocoder drops SIM to ~0.845, and `prompt_ext` itself scored 0.9122 against the real voice vs 0.9492 for the untouched prompt. Rewritten 2026-08 to "original prompt (≥55%, `--anchor-min-ratio`) + transposed segments filling the remaining budget". Measured on two songs:

  | | song_1 | song_2 |
  |---|---|---|
  | band-distribution L1 vs song, raw prompt | 1.055 | 0.735 |
  | L1, old ext (replace-all) | 0.493 | — |
  | **L1, new ext (anchor+gap)** | **0.377** | **0.224** |
  | `prompt_ext` SIM vs real voice, old → new | 0.9122 → **0.9320** | — |
  | SVC mean SIM, old → new | 0.5219 → **0.5320** | **0.5794** |
  | drift / dropouts / worst | 0.0723 / 7 / 0.2349 | 0.0657 / 1 / 0.2908 |

  Three fixes mattered, in order of impact: (1) keeping clean audio in the prompt at all; (2) allocating the extension budget by **time-weighted band gap** (`song % − anchor %` per 2-semitone grid) instead of the old "one equal-probability target per segment", which silently assumed every segment contributes equal duration; (3) picking, for each gap band, the candidates needing the **smallest** shift — the old code sorted segments by median ascending and paired them with targets ascending, so low-pitched segments always drew low targets and their transposition headroom went unused. Also: skip the vocoder entirely for the `+0` group, and allow one source segment to produce several differently-shifted copies (material reuse beats forcing +7st).
- **A strict "must land inside this band" filter starves the top bands.** First cut of the new selector rejected any candidate whose clamped shift fell outside the target band; with material topping out at med 339Hz, +5st reaches only 452Hz, so the 437-491Hz band got 0.30s of the 2.13s it needed — *worse* than the old algorithm there. Allowing a best-effort fallback (rank in-band candidates first, then nearest-to-center) brought it to 2.38s and cut L1 from 0.463 to 0.377.
- **`--auto_shift` transposes the whole song to fit the speaker.** With a 368Hz song and 247Hz speech it computes −7 semitones, which changes the key. Use `--pitch_shift 0` to preserve the melody.
- **`build_vocal_segments`' 30s ceiling is a 6GB VRAM trap that looks like slowness.** `soulxsinger_svc.py:257` hard-coded `max_duration_sec=30.0`; with `num_overlaps=1` the longest chunk reaches ~33s. On a 6GB card that chunk overflows into host-memory paging: song_2 sat at 8/10 segments for 12+ minutes with VRAM pegged at 5845/6144 MiB while the first five pipeline stages had taken only 4.6 min total. Patched to accept `max_seg_sec` (threaded through `cli/inference_svc.py --max_seg_sec`, `sing.py --max-seg-sec`, default **20.0**); the short-circuit at `soulxsinger_svc.py:234` follows the same value. song_2 then went 10 chunks (max 31.3s) → 14 chunks (max 22.8s) and **finished in 305.8s**. Raise it on a 24GB card. Symptom to recognize: `Inferring segments` stops advancing with no error, and killing `sing.py` leaves the child `inference_svc.py` alive holding all the VRAM — check `ps aux | grep inference_svc` and kill it before re-running.
- **SoulX SVC handles silence correctly on its own.** `soulxsinger_svc.py:263` inits output as `zeros_like(gt_wav)` and `build_vocal_segments(ignore_silent_segments=True)` drops segments under 5% voiced. Do not add a silence gate — regions that look like leakage are usually breath/consonants that RMVPE found no F0 for, and gating them removes real audio.
- **Speech-only references are a known-hard case** ("cross-domain SVC"). In SVCC2023 Task 2 (speech-only target), *no* submitted system reached target-speaker similarity; the winner still sat ~0.4 MOS below. A SIM around 0.6 here is not a failure. The winning system's ablation showed adaptation/fine-tuning beat zero-shot by 0.41 similarity — so **fine-tuning is the next real lever**, not more prompt tweaking.
- **The source material bounds everything.** This reference had 0.4s above 370Hz while the song spent 44.3% of its voiced frames there. Reference *selection* only reshuffles what exists; recording 30-60s of higher-register material is worth more than any parameter.

### SVC engine A/B (2026-08): SoulX-Singer vs YingMusic-SVC

Same inputs (257s lead vocal, same ref), same eval protocol (eval_sim vs real speech, singtest1 dir):

| engine | prompt | mean SIM | median SIM | drift | dropouts | worst |
|---|---|---|---|---|---|---|
| SoulX-Singer (baseline) | prompt_ext | 0.5121 | 0.5310 | **0.0916** | **27** | **0.2123** |
| YingMusic-SVC | prompt_ext (24.7s) | 0.5496 | 0.5670 | 0.1283 | 38 | 0.1698 |
| YingMusic-SVC | **raw prompt (7s, no ext)** | **0.5531** | 0.5524 | 0.1233 | 38 | 0.1723 |

- **YingMusic is ~+0.04 mean SIM but less stable**: adjacent-window drift ~0.03-0.04 higher, more dropouts, worse worst-case window. Diffusion (flow-matching, 100 steps) trades local jitter for better average similarity — don't judge by mean SIM alone (see `singing/eval_stab.py`).
- **The F0-aware timbre adaptor makes the pitch-extension stage unnecessary on this model**: raw 7s prompt ≈ 24.7s extended prompt (0.5531 vs 0.5496). On YingMusic, skip `ext` entirely.
- **YingMusic auto-transposes the song into the ref's register by default** (`mm4.py` `preprocess_voice_conversion`: base shift + adaptive factor + 3.5st, clamp ±8). Must pass `--semi-tone-shift 0` to keep the melody (flag added to `my_inference.py` — upstream argparse doesn't expose it).
- Setup: own `.venv` (torch 2.6+cu124), `checkpoints/` = seed-vc's HF caches copied in (whisper-small, bigvgan_v2_44khz_128band_512x, funasr/campplus, lj1995/VoiceConversionWebUI) + `YingMusic-SVC-full.pt` (731MB). **CWD must be `tools/yingmusic/`** (relative cache paths, same gotcha as seed-vc). RTF ≈ 5.9 at 100 steps fp16 on 6GB.
  ```bash
  cd tools/yingmusic && .venv/bin/python my_inference.py \
      --source ../data/songs/<name>/sep2/lead.wav --target ../data/songs/<name>/prompt.wav \
      --diffusion-steps 100 --checkpoint checkpoints/YingMusic-SVC-full.pt \
      --expname ym --cuda 0 --fp16 True --config ./configs/YingMusic-SVC.yml \
      --semi-tone-shift 0
  ```

### Separation notes

- `tools/uvr5/uvr5_weights/` ships `.ckpt` files but **BS-RoFormer's `.yaml` config was missing**; fetched from `ZFTurbo/Music-Source-Separation-Training`. The karaoke model + config came from `Soul-AILab/SoulX-Singer-Preprocess`.
- **`batch_size` in those yaml files is a real performance trap.** The karaoke model ships `batch_size: 8` with `chunk_size: 485100` (11s/chunk) = 88s of audio resident. On 6GB that overflows into host-memory swapping: VRAM pegged, GPU utilization low, 15+ min per track. Setting `batch_size: 1` gave 2.2GB VRAM, 92-100% utilization, **27 seconds** — ~33× faster. Read from config at `bsroformer.py:117`. Raise it again on a 24GB card.
- `tools/uvr5_cli.py` only writes the vocal stem, but the underlying `_path_audio_(input, others_root, vocal_root, ...)` writes both. Call it directly to keep the instrumental.
- Karaoke separation measurably isolates backing vocals (37/128 windows contaminated, up to backing/lead 0.66) but **only marginally improved output stability** — residual bleed remains. For harmony-heavy songs, YingMusic-SVC is the model actually trained for this (its benchmark is graded by contamination level).

## Measuring similarity

Do not judge by ear alone. `eval_sim.py` scores speaker similarity with the ERes2NetV2 model already in
`GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/`, bucketed by pitch distance from the reference.

**Always score against the real, unmodified speech** as ground truth. Scoring a pitch-shifted variant against
its own shifted prompt hides the very degradation you are trying to measure — that mistake made a real
regression look like a tie.

Mean SIM is also the wrong metric for "the timbre jumps around": use adjacent-window drift and worst-case
SIM to catch local instability. `singing/eval_stab.py` does this for arbitrary files (1.5s windows →
mean/median/drift/p90drift/dropouts/worst); `SoulX-Singer/work/eval3.py` is the original but its paths are
hard-coded to one old experiment.

```bash
GPT-SoVITS/.venv/bin/python singing/eval_stab.py --ref data/songs/<name>/refsep2/lead.wav \
    --gen old/vocal.wav new/vocal.wav --labels 旧 新
```

**Judge a prompt change by three numbers, not one.** Band-distribution L1 against the song (does the prompt
now spend its time where the song does), `prompt_ext` SIM against the real voice (how much timbre the
vocoder cost), and SVC output mean SIM + drift. A change can improve the first two and still lose on drift —
the 2026-08 ext rewrite gained +0.010 mean SIM but drift went 0.0660 → 0.0723 and dropouts 4 → 7, because
28 short transposed segments are less internally consistent than 20 longer ones. Raise `--anchor-min-ratio`
if that trade is wrong for a given song.

## `sing.py` — one-command zero-shot singing CLI

Automates the whole pipeline above; run from the workspace root with **system python3** (stdlib only, no venv needed):

```bash
python3 sing.py --ref data/voices/ref.wav --song data/songs/song.flac --name exp1
# 验收纯人声后，再混伴奏（可调伴奏音量）
python3 sing.py --ref ... --song ... --name exp1 --stages mix --acc-gain 0.7
# 只跑部分阶段（产物在 data/songs/<name>/，可续跑）
python3 sing.py --ref ... --song ... --name exp1 --stages sep,select
```

Stages: `sep` (BS-RoFormer → karaoke, 人声+伴奏+主唱+和声) → `ref-sep` (参考音频同样两步分离取纯人声
lead，产出 `data/songs/<name>/refsep2/lead.wav`；select/eval 自动优先用分离产物，无则退回原 ref) →
`f0-target` (RMVPE 50Hz) → `select`
(参考切片打分贪心选段 → `prompt.wav`+`prompt_f0.npy`+`prompt_stats.json`) → `ext` (原始 prompt 作
音色锚点 + 按频段缺口追加移调段，PC-NSF-HiFiGAN 默认 `--ext-method nsf`，seed-vc 可选；
`--anchor-min-ratio 0.55` 控制干净音频最低占比，拼出 ≤29s 的 `prompt_ext`) → `svc`
(SoulX, `--pitch_shift 0` 保留旋律，`--max-seg-sec 20` 防 6GB 爆显存) → `eval` (eval_sim.py 对照
真实语音) → `mix` (opt-in, `--mix`)。Worker 脚本在 `singing/w_*.py`，每个阶段用对应项目的 venv 跑。

### Gotchas (all hit in practice)

- **seed-vc 的缓存路径是相对的** (`inference.py:5` `HF_HUB_CACHE='./checkpoints/hf_cache'`，
  `hf_utils.py` `cache_dir="./checkpoints"`)。必须以 `seed-vc/` 为 CWD 启动，否则会在调用处新建
  空 `checkpoints/` 并**重新下载 2.4G 模型**。`sing.py` 已处理（`cwd=SEEDVC_DIR`，仅
  `--ext-method seedvc` 时走此路径）。`tools/yingmusic/` 同样有此坑（CWD 必须是它自己）。
- **PC-NSF-HiFiGAN 的准备是手工的**：`tools/pcnsf/` 的 `.venv` 只装了 CPU torch + 少量包
  （`modules/pe/rmvpe/inference.py` 已打补丁接受裸 state_dict 的 rmvpe.pt；config.json 补了
  `audio_sample_rate`/`mel_base`/`mel_scale` 等字段；parselmouth 必须用 0.4.3，新版依赖
  googleads 装不上）。缺任何一环 ext 阶段直接失败。
- **`python cli/inference_svc.py` 找不到 `soulxsinger` 包** — 直接跑 cli 脚本时 sys.path 只有 `cli/`。
  需要 `PYTHONPATH=SoulX-Singer`（`sing.py` 已处理）。
- **`SoulX-Singer/` 里有本地补丁，升级上游会覆盖掉**：`soulxsinger/models/soulxsinger_svc.py`
  的 `infer(max_seg_sec=...)` 参数 + `cli/inference_svc.py` 的 `--max_seg_sec`（显存分段），
  以及 `tools/uvr5/uvr5_weights/*.yaml` 的 `batch_size: 1`。拉新版后要重打。
- **参考素材决定上限**：`select` 只在现有素材里选段。素材太短（如 7s 可用）时 prompt 短、
  SIM 偏低，自动建议的变调数也偏高；要多补高音区，录更长的素材而不是靠变调。
- `--max-prompt` 默认 25s，扩展后 ≤29s（whisper 30s 硬顶）。pt_f0 参与 SVC 的 f0 conditioning，
  不能只拼音频不拼 f0（`w_ext.py` 对扩展段重新提 RMVPE）。

---

# GPT-SoVITS

Everything below concerns the `GPT-SoVITS/` project specifically. Run its commands from inside that directory.

## Architecture

GPT-SoVITS is a **research-grade TTS/VC project**, not a pip package. There is no `setup.py`, `pyproject.toml`, or packaging.

**Pipeline**: Text → GPT (AR text→semantic tokens, `s1`) → SoVITS (VITS/BigVGAN semantic→audio, `s2`)

**6 model versions**: v1, v2, v3, v4, v2Pro, v2ProPlus — each with separate pretrained weights, weight directories, and configs. `config.py` maps versions to paths.

### Key directories

| Path | Purpose |
|---|---|
| `GPT_SoVITS/AR/` | Autoregressive T2S GPT model (modules, models, data, processing) |
| `GPT_SoVITS/module/` | SoVITS/VITS core (synthesizer, discriminators, loss, mel) |
| `GPT_SoVITS/TTS_infer_pack/` | Config-driven v2 inference pipeline (used by `api_v2.py`) |
| `GPT_SoVITS/text/` | Multilingual text frontends (zh/en/ja/ko/yue) |
| `GPT_SoVITS/prepare_datasets/` | Data prep: text→phonemes, audio→SSL, SSL→semantic tokens |
| `GPT_SoVITS/pretrained_models/` | Pretrained model weights (downloaded separately, not in repo) |
| `tools/uvr5/` | Voice separation (webui, mdxnet, bsroformer) |
| `tools/asr/` | ASR tools (Fun-ASR-Nano, Faster-Whisper, FunASR) |
| `tools/AP_BWE_main/` | Audio super-resolution (24k→48k) |
| `tools/i18n/` | Internationalization engine |
| `SoVITS_weights*/` | Trained SoVITS checkpoints (gitignored) |
| `GPT_weights*/` | Trained GPT checkpoints (gitignored) |

### Entrypoints

- **`webui.py`** — Master Gradio WebUI orchestrator. Sets `version=v2Pro`, injects site-packages, spawns subprocesses via `Popen` for each pipeline tab (UVR5, dataset prep, S1/S2 training, TTS inference).
- **`api_v2.py`** — Modern FastAPI TTS API (config-driven via `TTS_infer_pack/TTS.py`). Preferred over `api.py`.
- **`api.py`** — Legacy monolithic FastAPI TTS API. Tightly coupled, manually wires models.
- **`GPT_SoVITS/inference_webui.py`** — Standalone Gradio TTS inference UI (spawned by `webui.py`).
- **`GPT_SoVITS/inference_cli.py`** — CLI batch TTS inference.
- **`GPT_SoVITS/s1_train.py`** — GPT (S1) training via PyTorch Lightning.
- **`GPT_SoVITS/s2_train.py`** — SoVITS (S2) training v1/v2/v2Pro/v2ProPlus.
- **`GPT_SoVITS/s2_train_v3_lora.py`** — SoVITS (S2) training v3/v4 with LoRA.

### Repo-local helper scripts (not upstream)

Two wrappers exist so you don't have to hand-assemble the env-var-only `prepare_datasets/` scripts or remember the i18n/version env vars that inference needs.

#### `train.py` — end-to-end finetune driver

Pipeline: `[vocal] → [denoise] → slice → asr → text → ssl → sv → semantic → s1 → s2`.
`vocal` (UVR5) and `denoise` are opt-in (`OPT_IN_STAGES`) — they need `--vocal` / `--denoise`, or naming them in `--stages`.

```bash
# Full run with defaults (v2Pro, Chinese, GPU 0)
uv run python train.py --ref data/voices/ref.wav --name myvoice

# Recommended for short (~2-3 min) source audio: augmentation packs clips into
# bucket0 and pushes sample count past min_num=100, so epochs become real epochs
uv run python train.py --ref data/voices/ref.wav --name myvoice --augment

# Run a subset of stages (re-run after fixing ASR transcripts, retrain S2 only, ...)
uv run python train.py --ref data/voices/ref.wav --name myvoice --stages slice,asr
uv run python train.py --ref data/voices/ref.wav --name myvoice --stages text,ssl,sv,semantic
uv run python train.py --ref data/voices/ref.wav --name myvoice --stages s2

# Forward all subprocess output (essential when a stage fails)
uv run python train.py --ref data/voices/ref.wav --name myvoice -v

# v2ProPlus, non-Chinese, second GPU
uv run python train.py --ref data/voices/ref.wav --name myvoice \
    --version v2ProPlus --lang ja --gpu 1

# Tune slicing: larger --min-interval only cuts at long (sentence-level) pauses
uv run python train.py --ref data/voices/ref.wav --name myvoice \
    --threshold -30 --min-length 4000 --min-interval 600 --max-sil-kept 800

# Tune training. Defaults: s1 batch 6 / 2 epochs / save every 1,
# s2 batch 4 / 10 epochs / save every 5. On 6GB VRAM keep both batches <= 4.
uv run python train.py --ref data/voices/ref.wav --name myvoice \
    --s1-batch 4 --s1-epochs 30 --s1-save-every 5 \
    --s2-batch 4 --s2-epochs 20 --s2-save-every 5
```

Paths are all derived from `--name`: clips in `data/voices/<name>_sliced/`, annotations at `data/voices/<name>.list`, features under `logs/<name>/`, weights in `GPT_weights_<version>/` and `SoVITS_weights_<version>/`. Logs go to both stdout and `logs/<name>/train.log`.

What it checks for you: all 8 pretrained models + G2PW + ffmpeg + GPU before starting; pre-creates the weight dirs that `savee()` silently needs; prints the bucket histogram with per-bucket padding warnings; flags the `min_num=100` duplication; cross-checks counts across all five stage outputs and **names the specific dropped item**; verifies new weights actually landed after each training stage. Interrupting is safe — both S1 and S2 resume from their latest checkpoint.

#### `infer.py` — TTS CLI

```bash
# Minimal: model/ref default to the verified e20 pair, ref transcript is
# auto-looked-up from the .list file
uv run python infer.py -t "今天天气真好呀，我们一起出去玩吧。" -o out.wav

# Read target text from a file (for long-form synthesis)
uv run python infer.py -t @script.txt -o out.wav

# Pick specific checkpoints — do this to A/B different epochs
uv run python infer.py \
    --gpt GPT_weights_v2Pro/girlish_voice-e20.ckpt \
    --sovits SoVITS_weights_v2Pro/girlish_voice_e20_s120.pth \
    -t "测试文本" -o out.wav

# New voice: point at its own clip + .list
uv run python infer.py \
    --gpt GPT_weights_v2Pro/myvoice-e12.ckpt \
    --sovits SoVITS_weights_v2Pro/myvoice_e10_s180.pth \
    --ref data/voices/myvoice_sliced/aug_000_00_sp100.wav \
    --list data/voices/myvoice.list \
    -t "测试文本" -o out.wav

# Explicit ref transcript (skips .list lookup)
uv run python infer.py --ref path/to/ref.wav --ref-text "参考音频里说的话" \
    -t "测试文本" -o out.wav

# Sampling / speed / language; -q silences upstream library chatter
uv run python infer.py -t "测试文本" -o out.wav \
    --lang 中英混合 --top-k 15 --top-p 1.0 --temperature 1.0 --speed 1.0 -q
```

`--lang` / `--ref-lang` take the Chinese literals (`中文`, `英文`, `日文`, `粤语`, `韩文`, `中英混合`, `多语种混合`) because the script forces `language=zh_CN` — see the `dict_language` gotcha below.

### Packaging a trained voice for reuse

Ship a self-contained bundle so a finished voice can be re-used without re-running the pipeline. Layout that `infer.py` consumes directly:

```
girlish_v2Pro/
├── README.md
├── gpt/      girlish-e15.ckpt  girlish-e20.ckpt  girlish-e30.ckpt   (149M each)
├── sovits/   girlish_e15_s330.pth  girlish_e20_s440.pth             (129M each)
└── ref/      6 clips spanning different deliveries + ref.list
```

Only ship the checkpoints worth A/B-ing (skip under-trained early epochs); each GPT is 149M and each SoVITS 129M, so pruning matters. GPT and SoVITS are independent — any pair works.

**Ship several reference clips, not one.** The reference sets the *emotion and delivery*, not just the timbre, so a single narration-style clip locks every output into that register. Pick clips covering the registers you'll actually need (narration / gentle / laughing / questioning / …), name them with their duration, and keep every one inside the 3–10s limit.

`ref/ref.list` carries the transcripts in standard 4-field format (`path|speaker|LANG|text`) with **paths relative to the bundle root**, so `--list` auto-looks-up the reference text after extraction:

```bash
uv run python infer.py \
  --gpt    girlish_v2Pro/gpt/girlish-e30.ckpt \
  --sovits girlish_v2Pro/sovits/girlish_e15_s330.pth \
  --ref    girlish_v2Pro/ref/01_解说_8.8s.wav \
  --list   girlish_v2Pro/ref/ref.list \
  -t "要合成的文本" -o out.wav
```

Build the zip with Python — **`zip`/`unzip` are not installed** on this host. Store the weights uncompressed; they're already-compressed binaries, so deflate burns minutes for ~1%:

```bash
cd /tmp/opencode/pkg && python3 - <<'EOF'
import zipfile, os
out = "/home/rookie/voice/girlish_v2Pro.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for root, _, files in os.walk("girlish_v2Pro"):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            z.write(p, p, compress_type=zipfile.ZIP_STORED if fn.endswith((".ckpt", ".pth")) else zipfile.ZIP_DEFLATED)
print(f"{os.path.getsize(out) / 1024 / 1024:.0f} MB")
EOF
```

Before zipping, copy the staged dir somewhere else and run one synthesis through the in-bundle relative paths — that is what catches a broken `ref.list`. Verify afterwards with `zipfile.ZipFile(...).testzip()`.

The bundle deliberately **excludes pretrained models**. The target host still needs `GPT_SoVITS/pretrained_models/` (s1v3, chinese-hubert-base, chinese-roberta-wwm-ext-large, `sv/`, `v2Pro/`) plus `GPT_SoVITS/text/G2PWModel` for Chinese. Say so in the bundle README.

`infer.py`'s `DEFAULT_GPT`/`DEFAULT_SOVITS`/`DEFAULT_REF`/`DEFAULT_LIST` are hardcoded to one specific voice and **go stale whenever old weights are deleted**. Re-point them after training a new voice, or the no-argument invocation breaks.

### Ports

| Port | Service |
|---|---|
| 9874 | Main WebUI |
| 9873 | UVR5 WebUI |
| 9872 | TTS Inference WebUI |
| 9871 | SubFix WebUI |
| 9880 | API |

## Dev commands

### Using uv (recommended)

This project uses **uv** to manage the Python virtual environment. No conda needed.

```bash
# 1. Create venv with Python 3.10
uv venv --python 3.10

# 2. Install PyTorch first (match your CUDA version; see install.sh for other options)
uv pip install torch torchcodec --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies in the REQUIRED order (extra-req first, no-deps)
uv pip install -r extra-req.txt --no-deps
uv pip install -r requirements.txt

# 4. Run any entrypoint
uv run python webui.py
uv run python api_v2.py
```

**All commands** (lint, run, python) must be prefixed with `uv run` to use the project venv:

```
uv run pre-commit run --all-files   # lint + format all
uv run ruff check .                 # lint only
uv run ruff format .                # format only (line-length 120, py311)
```

### Pretrained models not managed by uv

Pretrained weights, G2PWModel, UVR5 weights, and other assets must be downloaded separately. See `install.sh` for the full list of URLs. Model weights go in `GPT_SoVITS/pretrained_models/`, are never committed, and are gitignored.

## Important conventions & gotchas

- **No `requirements.txt` alone.** Install order: `pip install -r extra-req.txt --no-deps` then `pip install -r requirements.txt`. The `requirements.txt` line `--no-binary=opencc` breaks pip if run first.
- **`webui.py` is the source of truth for startup.** It patches `sys.path` by writing to `site-packages/users.pth`, sets `version=v2Pro` env, creates `TEMP/`, and initializes `TORCH_DISTRIBUTED_DEBUG=INFO`. Other entrypoints may not work without similar setup.
- **Config lives in `config.py`** (module-level globals), not an env file. GPU detection, default model paths, ports, and weight root dirs are all here.
- **Model weights are downloaded separately**, never committed. Pretrained baselines go in `GPT_SoVITS/pretrained_models/`, trained checkpoints in `SoVITS_weights*/` / `GPT_weights*/` (both gitignored).
- **Gradio < 5** (pinned: `gradio<5`). Do not bump past 4.x.
- **Transformers pinned**: `>=4.43,<=4.50`. PEFT `<0.18.0`. `numpy<2.0`.
- **Chinese TTS requires G2PW** (`G2PWModel` in `GPT_SoVITS/text/`), downloaded and unpacked separately.
- **Version-dependent code paths** are common. Check `version` env var and `config.pretrained_sovits_name`/`pretrained_gpt_name` dicts to understand which branch is hit.
- **`dict_language` keys are i18n-translated.** `inference_webui.py:172-193` builds them via `i18n("中文")`, so in an English locale the key becomes `"Chinese"` and passing `"中文"` raises `KeyError`. Set `os.environ["language"] = "zh_CN"` **before** importing `inference_webui` to keep the Chinese literals (this is what `infer.py` does).
- **`fast_langdetect` needs its cache dir to exist.** Inference fails with `FileNotFoundError: Cache directory not found: .../pretrained_models/fast_langdetect`. Create the dir and the model downloads itself.
- **IPC via subprocess kill signals**: `webui.py` launches `inference_webui.py` and `inference_webui_fast.py` as child processes, managing them with `Popen.pid` + signal handlers. Use `psutil` to find/kill children.
- **No type checking configured.** There is no `mypy` config.
- **`transformers>=4.57` needed for Fun-ASR-Nano**: The default ASR backend uses Fun-ASR-Nano which internally needs Qwen3 support (added in transformers 4.57). Upgrade if ASR fails with `KeyError: 'qwen3'`.
- **`torchaudio.load()` may fail via torchcodec.** torchaudio 2.11 routes `load()` through torchcodec, which needs CUDA-12 NPP libs (`libnppicc.so.12`). If the host only has CUDA 13, this raises `OSError`. Symlinking `.so.13` as `.so.12` does **not** work (symbol version mismatch). Workaround used here: replace `torchaudio.load()` with `soundfile.read()` (patched in `prepare_datasets/2-get-sv.py` and `inference_webui.py:get_spepc`). Note `sf.read` returns `[samples, ch]` — transpose to `[ch, samples]`. Training data loading is unaffected (`tools/my_utils.load_audio` shells out to ffmpeg).
- **Create weight output dirs before training.** `logs_s2_*`/`logs_s1_*` missing → first `save_checkpoint` crashes. `SoVITS_weights_*`/`GPT_weights_*` missing → `process_ckpt.savee()` fails *silently* (G/D checkpoints land fine, but the small inference weights never appear).
- **Reference audio must be 3–10s or inference dies.** `inference_webui.py:857` raises `OSError: 参考音频在3~10秒范围外，请更换！`. Slice filenames encode sample offsets at the *source* rate, so the implied duration does not match the resampled file — always confirm with `ffprobe -show_entries format=duration` instead of doing arithmetic on the filename.
- **Hand-edited ASR transcripts must land in `data/voices/<name>.list`.** `stage_asr` writes its raw output to `data/voices/<name>_asr/<name>_sliced.list` and then copies a filtered version to `data/voices/<name>.list`; the `text` stage reads **only** the latter. Editing the `_asr/` copy has no effect and silently trains on uncorrected text.
- **The ONNX `libcublasLt.so.12` error during inference is harmless.** G2PW's CUDA provider fails to load and falls back to CPU; audio is still produced. Do not chase it.

## Small-dataset training gotchas (source-verified)

Non-obvious behaviors that materially change how training should be configured:

- **Samples are silently duplicated below 100.** `module/data_utils.py:54-59`: if the dataset has `< min_num = 100` usable items, the whole list is repeated `max(2, int(100/len))` times. 21 items → `wav_data_len: 84`. **Configured epochs are then not real epochs** (30 configured ≈ 120+ passes over the same audio). This is a bucketing safeguard, *not* a hint that 100+ samples are required — 1 minute of audio genuinely works for v2Pro.
- **`DistributedBucketSampler` buckets by audio length, and each bucket fills batches independently.** Boundaries `[32,300,400,...,1900]` frames at `hop_length=640`/32kHz mean `bucket0` spans **0.64–6.00s** while every later bucket covers only 2s. Sparse buckets get padded with repeats (`data_utils.py:1015`). Keeping clips inside `bucket0` and using a modest `batch_size` matters far more than raw dataset duration.
- **S2 only supervises 0.64s per step.** `segment_size=20480` @32kHz; `s2_train.py:405,417` random-slices that window for mel/discriminator loss. Audio beyond ~6s adds SSL/spec context but **no extra waveform supervision** — long clips are not inherently better.
- **S1's LR schedule is hard-coded off.** `AR/modules/lr_schedulers.py:58` (`self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！`) overrides everything. `optimizer.lr`, `lr_init`, `lr_end`, `warmup_steps`, `decay_steps` in `s1longer-v2.yaml` are **all inert**. Ignore any external guide that tells you to tune them.
- **`3-get-semantic.py` can silently drop items.** Observed: 21 entries in `2-name2text.txt` but only 20 rows in `6-name2semantic.tsv` (a ~2s clip vanished, no error). Cross-check counts across `2-name2text.txt` / `4-cnhubert` / `5-wav32k` / `7-sv_cn` / `6-name2semantic.tsv` before training.
- **webui's own defaults are the sane baseline**, not the values in the config templates. `webui.py:124-127` for non-v3/v4: `default_sovits_epoch = 8`, `save_every = 4`, `max_sovits_epoch = 25`. The `epochs: 100` sitting in `s2v2Pro.json` is not a recommendation.
- **v2Pro vs v2ProPlus differ only in vocoder width.** Configs differ by two lines (`upsample_initial_channel` 512→768, first kernel 16→20); all 19M extra params are in the decoder (14.9M→33.8M), and both share the same GPT base (`s1v3.ckpt`). They take identical code paths (`models.py:623` `v2pro_set`). Expect no gain on timbre or prosody — only potentially on waveform quality, and only if the source material actually has high-frequency content to reconstruct.
