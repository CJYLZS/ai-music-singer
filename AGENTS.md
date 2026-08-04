# AGENTS.md — voice workspace

## Hardware baseline (2026-08: 3060-Laptop 6GB → RTX 4070 Ti SUPER 16GB)

Many notes below were written against the old 6GB card and keep their historical detail;
**the current GPU is a 16GB 4070 Ti SUPER**. Numbers observed on it: seed-vc training
0.25s/step (was 36s/step on 6GB), karaoke separation 27s/track, SoulX SVC segments are
only VRAM-bound above ~30s. Whenever a note says "6GB card" or "raise on a 24GB card",
read the current limit as ~16GB: `sing.py --max-seg-sec` default is now **28** and
separation yaml `batch_size` can be raised toward 4.

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
├── tools/stems4.py   4-stem separation (drums/bass/other/vocals) via BS-RoFormer 4stems_ft.
│                     Uses GPT-SoVITS venv + uvr5 weights; see "Separation notes".
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
- **`--auto_shift` transposes the whole song to fit the speaker.** With a 368Hz song and 247Hz speech it computes −7 semitones, which changes the key. Use `--pitch_shift 0` to preserve the melody — *but* see the octave-alignment note below: `0` is right for arbitrary shifts, not for octaves.
- **Octave shifts are the exception to "always `--pitch_shift 0`", and they are the biggest lever found so far for low-register songs.** `pitch_shift` moves only the *F0 condition* fed to the model (`f0_to_coarse(gt_f0, f0_shift=pitch_shift*5)`, `soulxsinger_svc.py:360`; 1 semitone = 5 bins of 20 cents), so the model is asked to sing the same content at a different pitch — `target_f0.npy` itself is untouched, which means `select`/`ext` keep planning against the original register and do not fight the shift. Arbitrary shifts change the key and clash with the untouched accompaniment (upstream's webui compensates by shifting the acc too, `webui_svc.py:236`; `sing.py`'s mix stage cannot), but **±12 is musically free** — same pitch class, one octave up, which is exactly how a female singer covers a male song. A/B'd 2026-08 on song_3 (240.9s, 61 windows, female speech ref, prompt median 267.5Hz), scored per-window against the real voice:

  | | mean SIM | median SIM |
  |---|---|---|
  | `--pitch-shift 0` | 0.5018 | 0.5278 |
  | `--pitch-shift 12` (whole song) | 0.5009 | 0.5148 |
  | **`--pitch-shift auto`** (per segment) | **0.5485** | **0.5513** |

  The whole-song numbers look like a tie and are not: song_3 turned out to be a **duet**, low 135-175Hz for 16-71s and 235-337Hz afterwards. Split by source register, `+12` wins the low part by **+0.1834** (0.3926 → 0.5761, 14/14 windows improved, the largest single gain measured in this workspace — prompt tuning moves ±0.01) and loses the already-aligned high part by −0.0647, and the two cancel. **Never judge a pitch-shift A/B by the whole-song mean; bucket by source register.**
- **Per-segment octave alignment (`--pitch-shift auto`) is the general fix, and it nearly saturates.** Implemented as a local patch: `_nearest_octave_shift(seg_median, prompt_median)` rounds `12·log2(ref/seg)` to the nearest multiple of 12 (clamped ±36) and is applied per segment inside the existing loop (`soulxsinger_svc.py`, threaded through `cli/inference_svc.py --pitch_shift auto` and `sing.py --pitch-shift auto`). Two details matter: the decision must read the F0 of the **non-overlapping** part (`segments[idx]`, the audio that actually lands in the output) — using the overlap context lets a low lead-in bias a whole segment; and because SVC already splits at silence, the octave changes land in silence and are seamless. On song_3 it chose `+12` for 16.4-71.5s and `+0` for everything after, reaching 0.5485 against a per-window oracle ceiling of **0.5496**. Rule of thumb: a genuinely low male song (whole song ~150Hz) only needs `--pitch-shift 12`; duets and wide-register songs need `auto`.
- **`eval_sim --f0` must be the F0 of the audio being scored, not the source's.** `sing.py`'s eval stage passes `cfg.target_f0` (the *source* lead's F0), which is correct only when `pitch_shift=0`. On a shifted run the bucket labels are stale — the shift12 run reported the identical window counts (34/7/6/4/8/2) as the baseline, which is the tell. Re-extract the F0 from `vocal.wav` before bucketing a shifted run; the overall mean/median stay valid either way. (Verified the shift really applied by measuring output F0: 261.0Hz → 522.4Hz, exactly 2×.)
- **Non-octave global shifts are not a reliable lever, even when the whole-song pitch coverage looks good on paper.** A/B'd 2026-08 on song (song.flac, 257s, 70 windows): median 369Hz vs prompt 274Hz = +5.2st offset. Frame-level analysis showed −6st maximises ≤2st coverage from 17.5% → 49.0%. Running it: SIM dropped from 0.5313 → 0.5199 (−0.011). Cause: the song's melodic range spans p10-p90 = 254-465Hz, so the low end (254-331Hz) was already inside the prompt's register and had the highest SIM (0.5637 at 0-2st). The −6st shift pushed those already-aligned windows out by −0.064, while windows pulled in from the high register only gained +0.008 per window — pitch distance is no longer the bottleneck for those windows; content-feature leakage (whisper speaker residual) is. The asymmetry: moving an already-aligned window out costs more than moving a misaligned window in gains. Two corollaries: (1) never apply a non-octave global shift without checking what fraction of your best windows are already in range; (2) the `--pitch-shift auto` per-segment octave approach avoids this trap because it only shifts segments that are genuinely far out. Per-segment arbitrary (non-octave) shift is musically invalid — it changes the key mid-phrase, breaking tonality. The only musically safe non-zero shift is ±12/±24 (octave multiples, same pitch class).
- **`--cfg` is the largest single-parameter win found so far, and upstream's default is ~6× too high.** `cfg` is the classifier-free guidance scale in the CFM sampler (`flow_matching.py:293-304`): each Euler step also runs the estimator with the condition zeroed and extrapolates `flow_pred + cfg·(flow_pred − uncond_flow_pred)`, then rescales to the conditional prediction's std (`rescale_cfg=0.75` blends the rescaled and raw versions). The condition here is `content_feat + f0_feat` — **whisper content features plus F0, not a speaker embedding** (`soulxsinger_svc.py:391-394`). That is the whole story: amplifying the condition amplifies whatever speaker residual whisper left in the content features, i.e. CFG pushes the output *toward the original singer*, not toward the prompt. The prompt timbre enters as in-context mel (`pt_mel`), which CFG does not touch. Swept 2026-08 on song (257s, 70 windows), everything else fixed, `sing.py --stages svc,eval --cfg <x>`:

  | cfg | 0 | 0.5 | 1 | 1.5 | 2 | **3 (upstream)** | 5 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | mean SIM | 0.5373 | **0.5554** | 0.5541 | 0.5425 | 0.5357 | 0.5313 | 0.5158 | 0.5093 |
  | median SIM | 0.5478 | 0.5576 | **0.5673** | 0.5575 | 0.5485 | 0.5433 | 0.5257 | 0.5234 |

  Monotone decreasing above ~1, peak at **0.5-1**, and `cfg=0` (CFG fully off) still beats the default. Replicated on two more songs, and it **stacks with `--pitch-shift auto`**:

  | song | cfg=3 (baseline) | cfg=1 | **cfg=0.5** | gain |
  |---|---|---|---|---|
  | song (257s) | 0.5313 | 0.5541 | **0.5554** | +0.024 |
  | song_4 (`--pitch-shift 0`) | 0.5641 | 0.6099 | **0.6207** | +0.057 |
  | song_3 (`--pitch-shift auto`) | 0.5485 | 0.5755 | **0.5873** | +0.039 |

  For scale: every prompt-engineering change in this file moved ±0.01. **Default in `sing.py` changed to 0.5.**
- **Low `cfg` trades local stability for similarity — check `eval_stab` before accepting it.** Same runs, 1.5s windows vs the real voice:

  | | song mean/drift/dropouts/worst | song_4 | song_3 |
  |---|---|---|---|
  | cfg=3 | 0.4914 / **0.0894** / **12** / 0.2437 | 0.5312 / **0.0655** / **0** / **0.3709** | 0.5117 / **0.0793** / **4** / 0.2526 |
  | cfg=1 | 0.5080 / 0.1136 / 15 / 0.2401 | 0.5609 / 0.1002 / 2 / 0.2957 | 0.5375 / 0.0928 / 7 / 0.2343 |
  | cfg=0.5 | **0.5090** / 0.1154 / 15 / 0.2389 | **0.5738** / 0.1049 / 2 / 0.3196 | **0.5450** / 0.0933 / 4 / **0.2851** |

  Drift rises ~0.03-0.04 and dropouts go up on all three. Interpretation: CFG acts as a regularizer that keeps adjacent windows consistent (by leaning on the strongly-conditioned content signal), so removing it lets the output track the prompt more freely and jitter more. The mean gain (+0.024 to +0.057) is 2-6× larger than any prompt change, so it is worth the trade — but on a song where consistency matters more than peak similarity, `cfg=1` is the middle ground (better worst-case than 0.5 on song_4 and song_3's dropouts). Same shape of trade-off as YingMusic vs SoulX.
- **`build_vocal_segments`' 30s ceiling was a 6GB VRAM trap that looked like slowness.** `soulxsinger_svc.py:257` hard-coded `max_duration_sec=30.0`; with `num_overlaps=1` the longest chunk reaches ~33s. On the old 6GB card that chunk overflowed into host-memory paging: song_2 sat at 8/10 segments for 12+ minutes with VRAM pegged at 5845/6144 MiB. Patched to accept `max_seg_sec` (threaded through `cli/inference_svc.py --max_seg_sec`, `sing.py --max-seg-sec`); the short-circuit at `soulxsinger_svc.py:234` follows the same value. On 6GB we used **20.0**; **on the current 16GB card the default is 28.0** (the whisper encoder hard-caps at 30s — that is the real ceiling, not VRAM anymore). Symptom to recognize if it ever paged again: `Inferring segments` stops advancing with no error, and killing `sing.py` leaves the child `inference_svc.py` alive holding all the VRAM — check `ps aux | grep inference_svc` and kill it before re-running.
- **`build_vocal_segments` wasted inference on outro/interlude silence — three separate bugs.** (1) The split point for a silence run was placed at its *midpoint* (`(run_start+run_end)//2`), so the outro got glued to the previous vocal segment; (2) the min-duration merge absorbed a short vocal tail *forward* into the following long silence (`163.8-167.7s` vocal + 28s outro → one 24s segment at 11% voiced); (3) `ignore_silent_segments` scored **overlap** segments (which prepend the previous vocal grid) instead of the non-overlapping ones, masking pure-silence segments. Result: song_2 (outro 27.9s + 10s interlude) inferred 187.0s of its 195.7s. Patched (local, gets overwritten on upgrade): silence runs now split at *both* start and end; merging is activity-first — silence grids never enter segments (short active segments infer standalone instead of being padded with trailing silence), overlaps never cross a silence grid and stay ≤ `max_frames` (which also caps VRAM peak); the 5%-voiced filter now scores non-overlapping segments. song_2: 14 segments / 305.8s → **9 segments / 212.9s (−30%)**, tail 167.8s→end is exact zeros (mix fills the accompaniment), SIM mean 0.5794→0.5771 / median 0.5717→0.5757 (no regression). If the song has a long outro, always run `f0-target` first — segmenting keys off `target_f0.npy`.
- **SoulX SVC handles silence correctly *only after the build_vocal_segments patch*.** Output is inited as `zeros_like(gt_wav)` (`soulxsinger_svc.py:263`) and unvoiced regions simply stay zero. Do not add a silence gate — regions that look like leakage are usually breath/consonants that RMVPE found no F0 for, and gating them removes real audio.
- **Speech-only references are a known-hard case** ("cross-domain SVC"). In SVCC2023 Task 2 (speech-only target), *no* submitted system reached target-speaker similarity; the winner still sat ~0.4 MOS below. A SIM around 0.6 here is not a failure. The winning system's ablation showed adaptation/fine-tuning beat zero-shot by 0.41 similarity — so **fine-tuning is the next real lever**, not more prompt tweaking. **But 30s-scale fine-tuning is NOT that lever** (A/B'd 2026-08, see below) — QQ Music's "30s 微调" works because it's a purpose-built singing model + 30 epochs + matched-content nursery-rhyme recordings, not because 30s of data is enough in general.
- **seed-vc 30s fine-tuning: process works, result is negative.** A/B'd 2026-08 (ref.wav 15-45s cut into 3×10s clips → `train.py` on `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`, 100/500 steps, then `inference.py --checkpoint ft_model.pth` on a 30s song section, scored by `eval_stab.py` vs the real voice): zero-shot mean 0.5466/median 0.5991 → ft100 0.5413/0.5628 → ft500 0.5300/0.5506. **More steps = worse similarity**; loss plateaus at ~0.52. The model memorizes the 3 clips and loses the pretrained speaker-manifold prior that zero-shot relies on; and the ft objective is mel reconstruction, not speaker-similarity. Conclusion: for seed-vc, the right use is zero-shot + good prompt; real fine-tuning needs RVC-scale data (≥5-10 min). Flow is still useful as a reference: `train.py --config ... --dataset-dir <1-30s clips> --run-name x --batch-size 1 --max-steps 500`, output `runs/<name>/ft_model.pth`; resume auto-loads the latest `DiT_epoch_*_step_*.pth` (weights only, iters restart at 0).
- **seed-vc fine-tuning on a 6GB card was a VRAM-paging trap (fixed by hardware).** Full pipeline (DiT 17×768 + whisper-small fp32 + RMVPE + OpenVoiceV2 converter + campplus + BigVGAN, all fp32) peaks ~7.4GB VRAM. On the old 3060-Laptop 6GB it paged: **36s/step** (50 steps took 30+ min), no error, GPU util low. Same run on the 4070 Ti SUPER 16GB: **~0.25s/step**, 500 steps in ~2 min — this is now a non-issue, but if seed-vc training ever crawls again, check `nvidia-smi` for memory.used near the cap before touching any code. Note `checkpoints/hf_cache` (vocoder/whisper) vs `checkpoints/models--*` (DiT/rmvpe/campplus) are separate cache roots; OpenVoiceV2 converter downloads on first `train.py` run.
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
- Setup: own `.venv` (torch 2.6+cu124), `checkpoints/` = seed-vc's HF caches copied in (whisper-small, bigvgan_v2_44khz_128band_512x, funasr/campplus, lj1995/VoiceConversionWebUI) + `YingMusic-SVC-full.pt` (731MB). **CWD must be `tools/yingmusic/`** (relative cache paths, same gotcha as seed-vc). RTF ≈ 5.9 at 100 steps fp16 on 6GB (16GB card is not memory-bound; the bottleneck is the 100 diffusion steps themselves).
  ```bash
  cd tools/yingmusic && .venv/bin/python my_inference.py \
      --source ../data/songs/<name>/sep2/lead.wav --target ../data/songs/<name>/prompt.wav \
      --diffusion-steps 100 --checkpoint checkpoints/YingMusic-SVC-full.pt \
      --expname ym --cuda 0 --fp16 True --config ./configs/YingMusic-SVC.yml \
      --semi-tone-shift 0
  ```

### SVC engine A/B #2 (2026-08): HQ-SVC (AAAI'26 low-resource zero-shot)

HQ-SVC = 解耦 codec + EVA 模块 + DDSP/diffusion 渐进生成，主打低资源零样本。Same 257s song, same ref; here we compared on the full song with the real `prompt_ext` as target (SoulX via `sing.py` full pipeline, HQ-SVC via `tools/HQ-SVC/cli_infer.py --shift 0`):

| engine | eval_sim vs prompt_ext | eval_stab vs real voice (1.5s win) |
|---|---|---|
| | mean / median | mean / median / drift / dropouts / worst |
| SoulX-Singer-SVC (baseline) | **0.5850 / 0.5945** | **0.4886 / 0.5154** / 0.0957 / 16 / 0.2346 |
| HQ-SVC | 0.5060 / 0.5084 | 0.4086 / 0.4125 / **0.0748** / 24 / **0.1869** |

- **HQ-SVC loses ~0.08 mean SIM to SoulX in both protocols** — worse timbre similarity — but its windows are *more internally consistent* (drift 0.0748 vs 0.0957, best worst-case). It sounds "stable but off-timbre"; SoulX sounds closer to the ref with more local jumps. For speech→singing cross-domain this confirms zero-shot SVC quality is still dominated by pretraining scale (SoulX: 42k hours; HQ-SVC: <80h). **Eliminated by ear (2026-08): the timbre gap is clearly audible on the full song; do not wire HQ-SVC into `sing.py`.** Keep the A/B files only as reference data.
- Its selling point is **low-resource training** (single consumer GPU, <80h data), not zero-shot inference quality — training code was still not released as of 2026-08, so it's inference-only for now.
- Setup gotchas (repo dir + weights removed 2026-08 to free 15G; re-download if ever needed): ships a ready-made conda env `environment.tar.gz` (4.7GB → unpack to `venv/`, torch 2.0.1+cu118, python3.9); weights on HF `shawnpi/HQ-SVC` (`utils/pretrain/*`: 250000_step_val_loss_0.50.pth 241M + ns3_facodec_* 450M + rmvpe 368M + nsf_hifigan 57M). `utils/pretrain/nsf_hifigan/` needs `config.json` copied from `configs/hifigan_config.json` (not shipped in place). `LD_LIBRARY_PATH` must be unset before running (`unset LD_LIBRARY_PATH`). Local CLI wrapper: `tools/HQ-SVC/cli_infer.py` (source + 1..N targets, `--shift 0` to keep melody; sys.path must use `append`, not `insert`, or `utils` shadows itself). RTF ≈ 0.05 on 16GB (10 diffusion steps).

### Separation notes

- `tools/uvr5/uvr5_weights/` ships `.ckpt` files but **BS-RoFormer's `.yaml` config was missing**; fetched from `ZFTurbo/Music-Source-Separation-Training`. The karaoke model + config came from `Soul-AILab/SoulX-Singer-Preprocess`.
- **`batch_size` in those yaml files is a real performance trap.** The karaoke model ships `batch_size: 8` with `chunk_size: 485100` (11s/chunk) = 88s of audio resident. On 6GB that overflows into host-memory swapping: VRAM pegged, GPU utilization low, 15+ min per track. Setting `batch_size: 1` gave 2.2GB VRAM, 92-100% utilization, **27 seconds** — ~33× faster. Read from config at `bsroformer.py:117`. **On the current 16GB card `batch_size: 4` is safe** (≈8.8GB with the karaoke chunk size); the yamls in the repo still carry `batch_size: 1` from the 6GB era — bump them if a track feels slow.
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
(SoulX, `--pitch-shift 0` 默认保留旋律；`12` 全曲高八度适配男→女；`auto` 按段自动对齐（男女对唱/跨音区）；
`--max-seg-sec 28` 适配 16GB；`--cfg 0.5` 默认，上游 3.0 会放大原唱音色残留) →
`eval` (eval_sim.py 对照真实语音) → `mix` (opt-in, `--mix`)。
Worker 脚本在 `singing/w_*.py`，每个阶段用对应项目的 venv 跑。

### Gotchas (all hit in practice)

- **sep/ref-sep 有输入内容级缓存**：`w_sep.py` 按「输入文件 md5 + 模型 ckpt md5」两级建目录
  （`data/.sep_cache/<model>-<ckpt8>/<inmd5>/`），全命中时连 torch 都不导入，直接拷产物，
  140s 分离 → 4s。换 `--name` 重跑同一首歌 / 同一参考会全命中。坑：
  - 缓存目录在 `data/.sep_cache`（在 /data/ 下，不进 git）；`sing.py --no-sep-cache` 强制重跑。
  - 模型权重升级会自动失效（路径带 ckpt md5 前 8 位），但 BS-RoFormer 权重若不变、分离逻辑
    改了则需手动 `rm -rf data/.sep_cache`。
  - karaoke 阶段的输入是 `sep1/vocal.wav`：只有它字节稳定（同机同权重确定性分离 OK）缓存才
    能命中；中途杀进程不会污染缓存（先写 `.tmp` 再原子改名）。
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
  的 `infer(max_seg_sec=...)` 参数（显存分段）+ `build_vocal_segments` 活性优先合并（跳过
  尾奏/间奏无人声）+ `_nearest_octave_shift` / `pitch_shift="auto"` 按段八度对齐，
  以及 `cli/inference_svc.py` 的 `--max_seg_sec` 和 `--pitch_shift` 的 `_pitch_shift_arg`
  （改成接受整数或 `"auto"`，上游是 `type=int`），还有
   `tools/uvr5/uvr5_weights/*.yaml` 的 `batch_size: 1`（6GB 时代调的；16GB 卡可提到 4，见 Separation notes）。拉新版后要重打。
- **本地补丁都在自己的 fork 里，可 diff 可回溯**（2026-08）：
  - `SoulX-Singer` → `CJYLZS/SoulX-Singer` 分支 `voice-patches`（基于上游 `81aeb3a`）
  - `tools/pcnsf` → `CJYLZS/pitch-transposition-using-DiffSinger` 分支 `voice-patches`
    （基于上游 `0e40e97`，rmvpe 裸 state_dict 补丁）
  - 本地 remote 名为 `fork`。升级上游后重打：`git fetch fork && git rebase origin/<branch> voice-patches`
    （或 `git cherry-pick`），冲突按 AGENTS.md 的语义修。
  - GPT-SoVITS 的 uvr5 yaml 改动在 `.gitignore` 里 fork 也带不走，仍靠本文件记录。
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
# s2 batch 4 / 10 epochs / save every 5. On the old 6GB card keep both batches <= 4;
# 16GB fits s1 batch 12 / s2 batch 8 comfortably.
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
