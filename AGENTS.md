# AGENTS.md — voice workspace

## Hardware baseline (2026-08: 3060-Laptop 6GB → RTX 4070 Ti SUPER 16GB)

Many notes below were written against the old 6GB card and keep their historical detail;
**the current GPU is a 16GB 4070 Ti SUPER**. Numbers observed on it: seed-vc training
0.25s/step (was 36s/step on 6GB), karaoke separation 27s/track, SoulX SVC segments are
only VRAM-bound above ~30s. Whenever a note says "6GB card" or "raise on a 24GB card",
read the current limit as ~16GB: `sing.py --max-seg-sec` default is now **28** and
separation yaml `batch_size` can be raised toward 4.

**System RAM is the second, easily-forgotten ceiling: this is WSL2 on a 32GB host.**
Without a `memory=` line in `/mnt/c/Users/rookie/.wslconfig`, WSL takes the default 50%
of host RAM = **15.9GB**, which is *less than the GPU-adjacent workloads need for weight
loading*. Video-diffusion checkpoints are the trigger: `echomimic3`'s T5 encoder is a
single 11.36GB `.pth`, and `torch.load(..., map_location="cpu")` without `mmap=True`
peaks at roughly **2× file size (~23GB)** because the file buffer and the materialised
tensors are both resident. Symptom is not a Python traceback but `Killed` / **exit code
137** (the kernel OOM killer), with the log stopping mid-way through model loading — easy
to misread as a hang or a CUDA problem. Check `grep MemTotal /proc/meminfo` and the
`.wslconfig` before debugging anything else. Raising WSL to `memory=24GB` + `swap=8GB`
(edit `.wslconfig`, then `wsl --shutdown` from PowerShell) is the quick fix; the real fix
is `mmap=True`, measured on this host to drop the same T5 load from ~23GB to **0.38GB
peak RSS** (60×). `nvidia-smi` looks completely healthy during these kills — VRAM is not
the constraint.

### Downloading large weights: verify resume, do not assume it

`huggingface_hub`'s `snapshot_download`/`hf_hub_download` write to
`<sha256>.<etag-suffix>.incomplete`, and **the suffix is renegotiated on every process
start**, so interrupting and re-running creates a *new* temp file and restarts from byte
0 while the old partial sits on disk unused. This silently re-downloaded ~10GB of an
11.36GB file here. Tell-tale sign: several `.incomplete` files under
`<model>/.cache/huggingface/download/` sharing one sha256 but differing in the final
suffix.

The HF CDN *does* honour byte ranges (verified: `HTTP 206`, and the returned bytes are
identical to the local partial's tail), so resume works — just not through hf_hub across
restarts. `echomimic3/download_models.py` is the pattern to copy: download to a fixed
`<target>.part` name, resume with `curl -fL -C -`, then verify against the repo's LFS
sha256 (`HfApi().model_info(repo, files_metadata=True)`) before renaming into place. It
is idempotent, so re-running after any interruption transfers only what is missing.
Before trusting a partial file, confirm it is a real prefix:
`curl -sL -H "Range: bytes=<size-32>-<size-1>" <url> | xxd` versus
`tail -c 32 <partial> | xxd`.

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
├── tools/pcnsf/      PC-NSF-HiFiGAN pitch-transposition toolkit (scripts/pitch_transpose.py
│                     单 job；scripts/multi_transpose.py 多 job，ext 阶段用，省 2.2× 进程开销).
│                     .venv = torch 2.13 CPU-only (~500MB, 无需 CUDA). DEFAULT range extender.
├── tools/pcnsf_hifigan/  PC-NSF-HiFiGAN 44.1k weights (config.json + model.ckpt), 2025.02 release
├── tools/yingmusic/  YingMusic-SVC (zero-shot SVC, seed-vc fork + F0-aware timbre adaptor).
│                     .venv = torch 2.6+cu124. Experimental; see "SVC engine A/B" below.
├── tools/voxcpm2/    VoxCPM2 TTS (voice design + style-controlled cloning) 与本轮移调实验脚本.
│                     .venv = torch 2.6+cu124. 结论是否定的，见「参考素材音域扩展」一节
├── tools/vevo2/      Vevo2 (Amphion) 零样本 SVC，speech+singing 联合 tokenizer.
│                     .venv = torch 2.6+cu124. 结论是否定的（whisper 30s 硬顶），见 SVC engine A/B #2
├── tools/hifiglot/   HiFi-Glot 显式共振峰合成（F1-F4 可控）.
│                     .venv = torch 2.6+cu124. 结论是否定的，见「共振峰」一节
├── tools/stems4.py   4-stem separation (drums/bass/other/vocals) via BS-RoFormer 4stems_ft.
│                     Uses GPT-SoVITS venv + uvr5 weights; see "Separation notes".
├── tools/singmos/    SingMOS naturalness MOS 打分（score_naturalness.py）+ 指标敏感性探针
│                     （probe_metric_sensitivity.py）。独立 uv venv（torch 2.2.1+cu121）.
│                     见「SingMOS 自然度评测与 NSF 后处理」
├── tools/nisqa/      NISQA 语音质量/自然度评测（官方仓库浅克隆，权重内置在仓库里）.
│                     重新获取: git clone --depth 1 https://github.com/gabrielmittag/NISQA.git tools/nisqa
├── a25rig/           Anime2.5DRig (MIT) fork + 离线渲染补丁: PSD → 逐帧 PNG（口型吃
│                     make_cues.py 时间轴）. driver/ 是 WSL 内无头 Chrome 驱动.
│                     入口 a25rig/render.sh（mv.py render 阶段也在调；必须用它，
│                     见「Anime2.5DRig 离线渲染」一节）
├── echomimic3/       EchoMimicV3 (AAAI'26, Apache-2.0): 单图 + 音频 → 说话/唱歌视频.
│                     .venv = torch 2.6+cu124. 1.3B params, flash-pro 8-step, ≤768×768.
│                     可吃 GPT-SoVITS 的 TTS 输出或 sing.py 的 SVC 干声。见「EchoMimicV3」一节
├── listen_lipsync/   口型 + 字幕流水线脚本: make_cues.py（歌词→口型时间轴）、
│                     split_visemes.py（九宫格→9 张 sprite）、get_lyrics.py（抓词缓存
│                     成同名 .lrc）、make_ass.py（→ 双行滚动 KTV 字幕）。GPT-SoVITS venv
├── singing/          sing.py 的 worker 脚本（w_*.py = 各阶段，eval_*.py = 各评测协议）.
│                     w_select_svs.py 从 SVS 输出选 prompt、eval_dual.py 双锚点评测，
│                     两者只服务 --svc-prompt svs 臂（素材音域不足时的备选，非默认）
├── data/             Shared audio. voices/ = speaker source material; songs/ = songs, stems, SVC output
├── docs/             调研/实验文档（voice-beautify-survey.md、source-timbre-normalization-survey.md 等）
├── logs/             GPT-SoVITS training features + checkpoints
├── AGENTS.md         this file
├── README.md         MV 出片流程（sing.py 产物 → mv.py），操作向
├── mv.py             MV 渲染流水线入口（分阶段：slice/lyrics/cues/beats/visemes/intro/render/subs，
│                     复用已有素材、可续跑；见 README ② MV 渲染）
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
            └─ SVS (score-control) ← MIDI + 歌词 + prompt_ext → svs_vocal (全音域覆盖)
                 └─ select-svs 选段                          → prompt_svs
                      └─ SoulX-Singer SVC  ← prompt_svs      → converted vocal
                                              ↑
                            prompt_ext = original prompt (timbre anchor, ≥55%)
                                       + PC-NSF-HiFiGAN transposed segments (≤+8st each,
                                         allocated by per-band time-share gap vs the song)
```

Measured on a 257s Chinese song with a 163s female speech reference (median F0 247Hz):

| variant | mean SIM | median SIM | >370Hz segments |
|---|---|---|---|
| narrow prompt / contaminated vocal | 0.5960 | 0.6151 | 0.5592 |
| wide prompt / contaminated | 0.6093 | 0.6250 | 0.5697 |
| wide prompt / clean lead | 0.6084 | 0.6236 | 0.5772 |
| **pitch-extended prompt / clean lead** | **0.6119** | **0.6363** | **0.5815** |

**这张表是窄素材时代（163s 说话参考）的历史数据，锚点是真实参考，`--svc-prompt ext`。**
`ext` 仍是当前默认，但参考素材加强后移调那一层已基本不触发（`--anchor-min-ratio 1.0`），
所以表里"pitch-extended prompt"这一行对新素材不再适用 —— 见「参考素材加强」一节。
中间曾以 `--svc-prompt svs` 为默认（主指标换成 SIM vs `svs_vocal`，与本表不可比），
现已退回备选，见「SVS prompt extraction」一节。

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
- **`--pitch-shift auto` only fires on genuine *octave* separation, and "the song has low parts" is not the same test.** Ran it on the four songs whose `target_f0` has real low content, `--seed 1234` fixed so the arms are paired:

  | song | `<190Hz` share | lowest segment median | base | auto | gain |
  |---|---|---|---|---|---|
  | **song_3** | 26.2% | **153-156Hz** (9.3-9.6st below prompt) | 0.5208 | **0.5796** | **+0.0588** |
  | song_9 | 13.3% | 195Hz (5.4st) | 0.4829 | 0.4829 | 0.0000 |
  | song_8 | 11.1% | 238Hz (1.2st) | 0.5276 | 0.5276 | 0.0000 |
  | song_5 | 9.0% | 211Hz (3.7st) | 0.5618 | 0.5618 | 0.0000 |

  The three zeros are **bit-identical output** (same md5), not noise: `_nearest_octave_shift` rounds `12·log2(ref/seg)` to a multiple of 12, so a segment must sit ≥6st below the prompt before it rounds to +12 at all. song_9's low sections are 4-7st below — closer to the prompt where they are than where +12 would put them (195Hz → 390Hz is 6.6st *above* a 254Hz prompt, i.e. 1.1st worse). **`auto` correctly declines**; there is nothing to win there. Only song_3, whose duet halves are a near-exact octave apart, qualifies. Per-window on song_3: 18/62 windows gained (+0.005…+0.35, median +0.13), 1 lost 0.015, and **43 were bit-identical** — every gain sits in 16-60s where segment medians are 117-178Hz, and everything after 72s (medians 232-350Hz) is untouched.

  So the pre-flight check is not "does the song have low notes" but **"is any segment median ≥6st below the prompt median"**. Compute it from `target_f0.npy` with `build_vocal_segments` before reaching for `auto`; on a song like song_5/8/9 it is a guaranteed no-op and the `+0` alignment is already optimal. Corollary for the three songs that scored worst: their problem is *not* register mismatch, so pitch tooling cannot fix them.
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
- **`rescale_cfg` is a real but sub-noise effect — swept and rejected as a default change.** It is the second CFG knob (`flow_matching.py:262`, upstream hard-codes 0.75 and exposes it nowhere): after the extrapolation `flow_pred_cfg = flow_pred + cfg·(flow_pred − uncond)`, the result is rescaled to the conditional prediction's std and the two are blended, `rescale_cfg·rescaled + (1−rescale_cfg)·raw`. So 1.0 = fully std-normalised (most conservative), 0.0 = raw extrapolation. Now plumbed through the same four layers as `seed` (`sing.py --rescale-cfg` → `inference_svc.py --rescale_cfg` → `infer` → `infer_segment` → `CFMDecoder`). Swept on song_2 at `cfg=0.5`, **seed fixed at 1** so the arms are paired:

  | rescale_cfg | 0.0 | 0.5 | 0.75 (upstream) | 1.0 |
  |---|---|---|---|---|
  | mean SIM | **0.5924** | 0.5900 | 0.5873 | 0.5850 |
  | median SIM | **0.5964** | 0.5922 | 0.5917 | 0.5885 |
  | drift | 0.1053 | 0.1046 | **0.1027** | 0.1019 |

  Monotone: less normalisation → higher SIM, slightly higher drift. Direction replicated on 3 seeds (0.0 beat 0.75 in 3/3: +0.0051 / +0.0043 / +0.0033), **but the mean gain is +0.0042 — exactly the seed-noise floor**, an order of magnitude below `--cfg` (+0.024…+0.057). Left at the upstream 0.75. Worth knowing for two reasons: it is the same mechanism as `cfg` (both control how hard the content condition is amplified, and both trade SIM against drift in the same direction), and if `--cfg` is ever raised for a stability-sensitive song, `rescale_cfg=1.0` is the matching "even more conservative" setting. Also note `cfg=0` skips the whole `if cfg > 0` block, so `rescale_cfg` is dead there.
- **Per-segment seed resampling has measurable headroom but trades drift.** Measured 2026-08 by running song_2 (6 segments, 52 windows) and song_9 (12 segments, 96 windows) with 5 different seeds, then computing three scenarios: (1) single-seed baseline (best of 5: mean 0.5496 / 0.4684); (2) segment-level oracle — for each segment, pick the seed whose windows in that segment score highest against the prompt; (3) per-window oracle (unreachable upper bound). Results:

  | song | 5-seed mean | best single seed | seg oracle | win oracle | seg captures |
  |---|---|---|---|---|---|
  | song_2 | 0.5962 | 0.6003 | 0.6127 **(+0.0124)** | 0.6360 | 42% |
  | song_9 | 0.4633 | 0.4684 | 0.5026 **(+0.0342)** | 0.5166 | 74% |

  All five columns are the same protocol (`eval_sim`, 3s windows, vs `refsep2/lead.wav`) — do not mix in
  `eval_stab`'s 1.5s numbers when comparing, they run ~0.05 lower and the gain looks 5× bigger than it is.

  The low-scoring song gains more. But segment switching introduces cross-segment timbre drift: mixing the oracle-selected segments from 5 seeds raised song_2's mean SIM from 0.5496 → 0.5604 (+0.011), but `eval_stab` drift rose from 0.1034 (most stable single seed) → 0.1199, and worst-case window dropped to 0.3073 (vs 0.2565 baseline). Trade-off is similar to the cfg case: +SIM for +drift. Segment boundaries are silent (切在静音处), but the ear still detects timbre shifts across phrase boundaries. Conclusion: *if* implementing multi-seed selection, monitor drift; the naive "每段选最优 seed" plan needs either (a) acceptance of the drift cost, or (b) a consistency penalty in the selection objective.
- **DP 段级选择（Viterbi）实现了 oracle 的大部分收益，且比朴素 argmax 更稳定。** 用 `prompt_ext.wav`（已有产物，无需真实语音）对每段打分，再用带边界一致性惩罚的 Viterbi 选每段用哪个 seed：`argmax Σ simscore[choice_k][k] + λ·Σ cos(tail_k, head_{k+1})`。λ=0 退化成朴素 argmax，λ→∞ 趋近全曲单 seed。实测 5 seeds × 2 首歌（`dp_select.py`，`eval_sim` 3s 窗对真实语音）：

  | | 最好单 seed | λ=0 (argmax) | λ=1 (DP) | 增益 (vs best single) |
  |---|---|---|---|---|
  | song_2 (6 seg) | 0.6003, drift=0.1104 | 0.6131, drift=0.1130 | **0.6155**, drift=0.1190 | **+0.0152** |
  | song_9 (12 seg) | 0.4684, drift=0.0717, drop=31 | **0.5071**, drift=0.0827, drop=15 | 0.4963 | **+0.0387** |

  `eval_stab` 同时改善：song_2 dropouts 维持 2/103 不变，worst 从 0.2565 升至 0.3073；song_9 dropouts 从 31 降至 15/187，说明 argmax 的换 seed 次数多（9/11）虽然提高了均值，但砍掉了孤立低分窗口，反而改善了尾部。

  **λ 的默认值是 0（朴素 argmax），"段数决定 λ" 是噪声拟合，不要照抄。** 上表 song_2 那格 λ=1 比 λ=0 高 0.0024 —— 低于 0.004 的 seed 噪声底，从来没有被分辨出来。用 `--n-seeds 3` 复测时方向就反了：song_2 λ=0 → 0.6116 vs λ=1 → 0.6027（**−0.0089**），song_9 λ=0 → 0.5017。三次能分辨的对比里 λ=0 赢两次，λ=1 那次赢在噪声内。**λ 真正确定的作用是拿 SIM 换 drift**，和 `--cfg` 完全同一形状：song_2 3-seed 上 λ=0 drift 0.1184 / λ=1 drift 0.1072，λ=1 确实更稳但均值更低。所以 `--dp-lam` 默认 0，只在某首歌听得出乐句边界音色跳变时才往上调。

  **`dp_select.py` 的使用方法**（已在 `/tmp/opencode/dp_select.py`，见下方实施计划）：
  1. 用 `--seed 1/2/3/4/5` 分别跑 N 次 SVC（每次 `--stages svc`，约 1 min/次）
  2. 运行 `dp_select.py song_X run1 run2 run3 ... --lam 0 1 2` 选出最优方案并写出拼接音频
  3. 对拼接音频运行 `eval_sim` 验证，再决定 λ

  代价 N× 推理时间（3 seeds ≈ 3min），可用 `--n-seeds N` 在 `sing.py` 里一键完成（待实现）。
- **`build_vocal_segments`' 30s ceiling was a 6GB VRAM trap that looked like slowness.** `soulxsinger_svc.py:257` hard-coded `max_duration_sec=30.0`; with `num_overlaps=1` the longest chunk reaches ~33s. On the old 6GB card that chunk overflowed into host-memory paging: song_2 sat at 8/10 segments for 12+ minutes with VRAM pegged at 5845/6144 MiB. Patched to accept `max_seg_sec` (threaded through `cli/inference_svc.py --max_seg_sec`, `sing.py --max-seg-sec`); the short-circuit at `soulxsinger_svc.py:234` follows the same value. On 6GB we used **20.0**; **on the current 16GB card the default is 28.0** (the whisper encoder hard-caps at 30s — that is the real ceiling, not VRAM anymore). Symptom to recognize if it ever paged again: `Inferring segments` stops advancing with no error, and killing `sing.py` leaves the child `inference_svc.py` alive holding all the VRAM — check `ps aux | grep inference_svc` and kill it before re-running.
- **`build_vocal_segments` wasted inference on outro/interlude silence — three separate bugs.** (1) The split point for a silence run was placed at its *midpoint* (`(run_start+run_end)//2`), so the outro got glued to the previous vocal segment; (2) the min-duration merge absorbed a short vocal tail *forward* into the following long silence (`163.8-167.7s` vocal + 28s outro → one 24s segment at 11% voiced); (3) `ignore_silent_segments` scored **overlap** segments (which prepend the previous vocal grid) instead of the non-overlapping ones, masking pure-silence segments. Result: song_2 (outro 27.9s + 10s interlude) inferred 187.0s of its 195.7s. Patched (local, gets overwritten on upgrade): silence runs now split at *both* start and end; merging is activity-first — silence grids never enter segments (short active segments infer standalone instead of being padded with trailing silence), overlaps never cross a silence grid and stay ≤ `max_frames` (which also caps VRAM peak); the 5%-voiced filter now scores non-overlapping segments. song_2: 14 segments / 305.8s → **9 segments / 212.9s (−30%)**, tail 167.8s→end is exact zeros (mix fills the accompaniment), SIM mean 0.5794→0.5771 / median 0.5717→0.5757 (no regression). If the song has a long outro, always run `f0-target` first — segmenting keys off `target_f0.npy`.
- **The silence-aware patch silently killed `num_overlaps` — every segment was inferred with zero preceding context.** `append_segment`'s backtrack loop was `while k >= 0 and cnt < num_overlaps and active(k)`, written when silence was merged into neighbouring grids. After the outro/interlude patch split each silence run into its own grid, `start_idx - 1` is *always* a silence grid, so `active(k)` fails on the first iteration and the loop exits immediately. Measured before the fix: song_2 **0/6** segments and song_9 **0/12** had any context (0.00s), at every `num_overlaps` and every `max_seg_sec` — the parameter was dead. Fixed by letting the backtrack step over *short* silence grids (`< min_frames`, same test the forward merge uses) and treating only long silence (interlude/outro) as a hard boundary. song_2 now gets 5.60s of context across 3/6 segments, song_9 10.16s across 2/12; the rest are first-segments or sit right after an interlude, where there is genuinely nothing to prepend. A/B on song_2 with `--seed 1` fixed:

  | | mean SIM | drift | p90drift | worst |
  |---|---|---|---|---|
  | before fix (ctx 0.00s) | 0.5405 | 0.1083 | 0.2373 | 0.2630 |
  | **after fix, `num_overlaps=1` (ctx 5.60s)** | 0.5377 | **0.1027** | **0.2240** | 0.2693 |
  | `num_overlaps=2` (ctx 8.80s) | 0.5379 | 0.1047 | 0.2246 | 0.2693 |

  The fix is a real local-stability gain (drift −0.006, p90drift −0.013) and mean SIM is unchanged (all three within 0.002 = noise). **Raising `num_overlaps` past 1 does nothing** — the first prepended active grid is what matters, a second one costs inference time and nudges drift back up. Exposed as `sing.py --num-overlaps` / `inference_svc.py --num_overlaps` for future experiments but **leave it at 1**. Also note the context budget competes with `max_seg_sec` for the whisper 30s ceiling: at `max_seg_sec=28` a 27.7s segment has almost no room left, which is the second reason a bigger `num_overlaps` cannot help.
- **SoulX SVC handles silence correctly *only after the build_vocal_segments patch*.** Output is inited as `zeros_like(gt_wav)` (`soulxsinger_svc.py:263`) and unvoiced regions simply stay zero. Do not add a silence gate — regions that look like leakage are usually breath/consonants that RMVPE found no F0 for, and gating them removes real audio.
- **Speech-only references are a known-hard case** ("cross-domain SVC"). In SVCC2023 Task 2 (speech-only target), *no* submitted system reached target-speaker similarity; the winner still sat ~0.4 MOS below. A SIM around 0.6 here is not a failure. The winning system's ablation showed adaptation/fine-tuning beat zero-shot by 0.41 similarity — so **fine-tuning is the next real lever**, not more prompt tweaking. **But 30s-scale fine-tuning is NOT that lever** (A/B'd 2026-08, see below) — QQ Music's "30s 微调" works because it's a purpose-built singing model + 30 epochs + matched-content nursery-rhyme recordings, not because 30s of data is enough in general.
- **seed-vc 30s fine-tuning: process works, result is negative.** A/B'd 2026-08 (ref.wav 15-45s cut into 3×10s clips → `train.py` on `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`, 100/500 steps, then `inference.py --checkpoint ft_model.pth` on a 30s song section, scored by `eval_stab.py` vs the real voice): zero-shot mean 0.5466/median 0.5991 → ft100 0.5413/0.5628 → ft500 0.5300/0.5506. **More steps = worse similarity**; loss plateaus at ~0.52. The model memorizes the 3 clips and loses the pretrained speaker-manifold prior that zero-shot relies on; and the ft objective is mel reconstruction, not speaker-similarity. Conclusion: for seed-vc, the right use is zero-shot + good prompt; real fine-tuning needs RVC-scale data (≥5-10 min). Flow is still useful as a reference: `train.py --config ... --dataset-dir <1-30s clips> --run-name x --batch-size 1 --max-steps 500`, output `runs/<name>/ft_model.pth`; resume auto-loads the latest `DiT_epoch_*_step_*.pth` (weights only, iters restart at 0).
- **seed-vc fine-tuning on a 6GB card was a VRAM-paging trap (fixed by hardware).** Full pipeline (DiT 17×768 + whisper-small fp32 + RMVPE + OpenVoiceV2 converter + campplus + BigVGAN, all fp32) peaks ~7.4GB VRAM. On the old 3060-Laptop 6GB it paged: **36s/step** (50 steps took 30+ min), no error, GPU util low. Same run on the 4070 Ti SUPER 16GB: **~0.25s/step**, 500 steps in ~2 min — this is now a non-issue, but if seed-vc training ever crawls again, check `nvidia-smi` for memory.used near the cap before touching any code. Note `checkpoints/hf_cache` (vocoder/whisper) vs `checkpoints/models--*` (DiT/rmvpe/campplus) are separate cache roots; OpenVoiceV2 converter downloads on first `train.py` run.
- **`n_steps` is saturated at the upstream 32 — do not spend time here.** Swept on song_2 at `--seed 1234`, everything else fixed: 32 → 0.5950 / 0.5735, 64 → 0.5959 / 0.5786, 100 → 0.5959 / 0.5827 (mean / median). The +0.0009 mean gain from 32→64 is a quarter of the seed-noise floor, 64→100 is exactly zero on the mean, and each doubling costs a proportional amount of inference time (the Euler loop is the whole cost). Median creeps up ~0.009 across the range, which is the only hint of a real effect and still sub-noise. Note the CFM sampler is a plain fixed-step Euler integrator (`flow_matching.py:292-315`), so more steps only reduce discretisation error — it cannot add speaker information that the condition does not carry. Confirms the older "≥16 converged" note with numbers.
- **Debiasing the whisper content features toward the prompt makes it worse — the speaker residual is load-bearing.** The tempting inference from the `--cfg` result is that the whisper speaker residual in `content_feat` is pure contamination, so subtracting it should help directly. Implemented and tested: take the time-mean of the source's content features over non-padded frames, take the prompt's, and shift the source's features by `α·(pt_mean − gt_mean)` before `content_feat + f0_feat` — stripping the static speaker component while leaving all temporal (phonetic) structure intact. Swept on song_2 at `--seed 1234`: α=0 → **0.5950**, α=0.5 → 0.5931, α=1.0 → 0.5932. Monotonically *worse*, and both arms land ~0.002 below baseline. Reason: the CFM decoder was trained on features with that residual present, so the learned feature→mel mapping already accounts for it; removing it at inference time is an out-of-distribution shift, and the decoder has no way to know the missing bias was speaker rather than content. This is why `--cfg` works and this does not — CFG changes *how hard the condition is amplified* (staying on-manifold), while mean-shifting moves the condition off the manifold entirely. Reverted; not exposed as a flag. General lesson: interventions that edit the conditioning signal in-place lose to interventions that reweight it — **or that substitute it with in-distribution vectors, see the kNN entry below**.
- **kNN retrieval on the content features is the second-largest lever found, and it stacks with `--cfg`.** Same target as the failed mean-shift — the source-singer residual in `content_feat` — but instead of *deleting* the residual it *replaces* it with the target speaker's: each source whisper frame is blended with the mean of its `k` nearest neighbours among the **prompt's own** whisper frames (`_knn_replace_content`, `soulxsinger_svc.py`, exposed as `sing.py --knn-alpha/--knn-k` → `inference_svc.py --knn_alpha/--knn_k`). Every substituted vector is therefore a convex combination of genuine whisper-base final-layer outputs, which is exactly what mean-shift was not. Costs nothing: `pt_content_feat` is already computed one line above (`:414`), the pool is ~1250-1450 frames, brute-force cosine needs no faiss. Same idea as kNN-VC / RVC's `index_rate`.

  Swept on song_2, `--seed 1234`, cfg=0.5 (52 windows, `eval_sim` vs the real voice):

  | α | 0.0 | 0.2 | 0.3 | **0.4** | 0.6 | 1.0 |
  |---|---|---|---|---|---|---|
  | mean SIM | 0.5950 | 0.5985 | 0.6016 | **0.6042** | 0.5918 | 0.5395 |

  Inverted U with a peak at **0.3-0.4**; full replacement (α=1.0) costs −0.056 and flattens
  `corr(pitch distance, SIM)` from −0.42 to −0.086, i.e. it destroys content. Confirmed over 3 seeds
  (1234/2234/3234, paired) so it clears the ±0.004 noise floor, and the gain is **larger at cfg=1.0**
  than at the current default cfg=0.5:

  | | α=0.0 | α=0.4 | gain | wins |
  |---|---|---|---|---|
  | cfg=0.5 | 0.5991 | 0.6066 | +0.0075 | 3/3 |
  | cfg=1.0 | 0.6002 | **0.6134** | **+0.0132** | 3/3 |

  **The triad is what makes this trustworthy: A rises while C *falls*.** Every other SIM gain in this
  file could in principle have been bought with original-singer leakage (`corr(A, excess) = +0.396`);
  this one demonstrably is not (seed 1234, B = 0.4295, 4 impostors):

  | arm | A | C | excess | progress | A−C margin | win% |
  |---|---|---|---|---|---|---|
  | cfg=0.5, α=0.0 | 0.5888 | 0.6492 | 0.1354 | 0.279 | −0.0605 | 28% |
  | cfg=0.5, α=0.4 | 0.5979 | 0.6195 | 0.1291 | 0.295 | −0.0216 | 42% |
  | **cfg=1.0, α=0.4** | **0.6086** | **0.6090** | **0.1188** | **0.314** | **−0.0003** | **55%** |

  `win%` (fraction of windows closer to the reference than to the original singer) crossing 50% is a
  first for this workspace. Note `eval_triad` reports 53 windows where `eval_sim` reports 52, so its A
  runs ~0.006 below the `eval_sim` mean — do not mix the two tools' absolute numbers in one table.
  Stability cost is the familiar shape and mild: at cfg=1.0, `eval_stab` mean 0.5444 → 0.5548,
  drift 0.0933 → **0.1050**, p90drift 0.2144 → 0.2526, dropouts 2 → 2, worst 0.2982 → 0.3074.
- **The kNN gain replicates on every song tested — 11/11 arms positive, and it is *largest where the
  baseline is worst*.** Cross-song check at cfg=1.0, α=0.4 vs α=0.0, all seeds paired:

  | song | n seeds | α=0.0 | α=0.4 | gain | wins | baseline rank |
  |---|---|---|---|---|---|---|
  | song_9 | 1 | 0.4831 | **0.5149** | **+0.0318** | 1/1 | worst song in the set |
  | song_6 | 3 | 0.5264 | 0.5473 | **+0.0209** | 3/3 | worst *progress* (0.220) |
  | song_2 | 3 | 0.6002 | 0.6134 | +0.0132 | 3/3 | mid |
  | song_4 | 3 | 0.6124 | 0.6198 | +0.0074 | 3/3 | best song in the set |
  | song_0 | 1 | 0.5587 | 0.5634 | +0.0047 | 1/1 | mid |
  | **all** | **11** | | | **+0.0146** | **11/11** | |

  Not one negative arm; the smallest gain (+0.0047) still sits at the noise floor and the largest
  (+0.0318) is 8× it. **`corr(baseline SIM, gain)` is strongly negative** — song_9 and song_6, the two
  songs this file previously singled out as hardest, gain 3-4× what the best song gains. That is the
  opposite of the usual pattern where a lever only helps where things already work, and it means kNN
  partially closes the cross-song spread rather than widening it.

  Leakage holds up on the other songs too, so song_2 was not a fluke (`--impostor` × 4, cfg=1.0):

  | song | A α=0→0.4 | C α=0→0.4 | excess α=0→0.4 | win% α=0→0.4 |
  |---|---|---|---|---|
  | song_6 | 0.5271 → 0.5458 | 0.6435 → **0.6175** | 0.1275 → **0.1169** | 10% → 25% |
  | song_9 | 0.4810 → 0.5125 | 0.4771 → **0.4338** | −0.0551 → **−0.0695** | 58% → **80%** |

  Same signature as song_2: A up, C *down*, excess down. song_6 is the one song where `excess` exceeded
  the normalised gain (flagged by `eval_triad` as 泄漏超过获得) and kNN narrows that gap from both
  sides. song_9's win% reaching 80% is the highest recorded here.

  Stability cost is consistently mild and, on song_9, actually *negative* (i.e. it improves):

  | song | mean | drift | p90drift | dropouts | worst |
  |---|---|---|---|---|---|
  | song_4 | 0.5638 → 0.5674 | 0.0999 → 0.1174 | 0.1902 → 0.2107 | 2 → 2 | 0.2842 → 0.3164 |
  | song_6 | 0.4988 → 0.5122 | 0.0841 → 0.0936 | 0.1845 → 0.1933 | 10 → 10 | 0.2077 → 0.2175 |
  | song_9 | 0.4527 → 0.4845 | 0.0759 → 0.0822 | 0.1769 → 0.1818 | **23 → 13** | 0.2102 → 0.2067 |
  | song_0 | 0.5132 → 0.5167 | 0.1120 → 0.1196 | 0.2330 → 0.2504 | 13 → 13 | 0.2398 → 0.2322 |

  drift rises 0.006-0.018 everywhere (the familiar SIM-for-drift trade), but dropouts never worsen and
  song_9 loses **10 of its 23** dropouts. So on the songs that need help most, kNN improves the mean,
  the leakage, *and* the tail simultaneously — there is no trade to weigh there.

  **All of the above was measured on speaker similarity alone, and α=0.4 turned out to be unusable
  once intelligibility was measured.** See the next entry — the α table there supersedes the "α=0.4
  is the peak" conclusion, and the working default is now **0.2**, not 0.4 and not 0.
- **kNN's real cost is articulation, and none of the three existing metrics could see it.**
  `eval_sim` / `eval_stab` / `eval_triad` all score speaker similarity; kNN edits the *content*
  features by construction, so validating it with timbre metrics only was a hole in the measurement
  setup, not a subtle oversight. Found by ear first — specific characters ("退") stopped being
  pronounceable — then reproduced objectively. `singing/eval_cer.py` now measures it: ASR the output,
  ASR the original lead as the reference text, report CER. Two things make the number trustworthy:
  - **The reference text is the original lead's own ASR transcript**, not a hand annotation, so CER
    measures "how many characters were lost *by the conversion*" rather than absolute recognisability.
    Both arms inherit the same ~61-character tail that the ASR drops on long audio, so the absolute
    CER is inflated and only the *relative* comparison is meaningful.
  - **FunASR's VAD is non-deterministic and this silently corrupted the first measurement.** The same
    file scored CER 0.3059 and 0.1912 on two runs; re-ASR'ing one file three times gave 345/281/345
    characters. `eval_cer.py` pins `torch`/`numpy`/`random` seeds and `cudnn.deterministic`, after
    which four consecutive runs are byte-identical. **Any CER number produced without that seeding is
    noise** — this is the same class of trap as the unseeded CFM noise documented above.

  Swept on 陪你看星星 (195.6s), cfg=1.0, k=4, full pool, seed 1234 fixed, `eval_sim` 52 windows:

  | α | 0 (off) | **0.1** | **0.2** | 0.3 | 0.4 |
  |---|---|---|---|---|---|
  | CER | 0.1324 | **0.1206** | **0.1206** | 0.1412 | 0.1912 |
  | ΔCER vs off | — | **−0.0118** | **−0.0118** | +0.0088 | **+0.0588** |
  | SIM | 0.5950 | 0.6042 | 0.6095 | 0.6148 | 0.6195 |
  | ΔSIM vs off | — | +0.0092 | **+0.0145** | +0.0198 | +0.0245 |

  **α=0.1-0.2 improves both metrics at once** — CER *drops* 1.2 points while SIM rises past the
  ±0.004 noise floor. SIM keeps climbing monotonically to α=0.4, so a similarity-only sweep
  (which is what the previous entry did) inevitably picks 0.4 and lands on a 44%-worse CER. The
  substitution-error count is where it shows: S goes 31 → 31 → 37 → **52** across α=0.1…0.4.
  **Default is now `--knn-alpha 0.2`**; 0.3 is the crossover, 0.4 is out.
  **⚠️ 2026-08 更正（两次）：默认 0.2 → 0 → 0.2（最终 0.2）**。第一次改 0 的依据是
  「源音色归一化预处理」C 组 4 首歌 cfg×α 矩阵：上表的验证全在旧协议时代（163s 池 /
  24.7s prompt / seed 1234 / 当时 SIM 基线 0.55-0.62）；当前协议（537s 池 / 44s prompt /
  seed 1 / 基线 0.75-0.82）下 4 首歌显示 α=0.2 净负：SIM 0/4 歌获益（−0.003）、cfg=0.5 下
  CER +1.9 点（4/4 歌变差），仅 SingMOS +0.02~0.03。**但 12 首全量默认参数重跑（2026-08）
  后追光者暴露吐字回归**：α=0 vs α=0.2 的 CER 0.1226 vs 0.0935（**−2.9 点**，吞字 10→4，
  SIM 持平 0.7975 vs 0.7969），证明 **α 对吐字的影响按歌分向** —— C 组 4 首是受害歌、
  追光者是受益歌，全局固定 0 会牺牲受益歌（预转换路线也救不了：seed-vc 预转换源自身
  CER 0.0742 极好，但 SoulX×预转换源交互把输出恶化到 0.1032，机制与去有风一致）。
  **默认最终定为 0.2**；受害歌用 `--knn-alpha 0` 按歌关。α=0.2 在 cfg=1.0 下仍帮 CER
  −1.0 点。

  For scale, compare the exchange rate against the published one. kNN-SVC (ICASSP'25, by a kNN-VC
  author) pays **+0.62 WER for +2.82 EER** and calls that "largely imperceptible". α=0.4 here pays
  **+5.9 CER points for +0.025 SIM** — roughly an order of magnitude worse trade. α=0.2 is not a
  trade at all, which is the only reason it is defensible as a default.
- **The kNN pool must be the whole reference, not `prompt_ext.wav` — the whisper 30s cap does not
  apply to it.** The first implementation matched against `pt_content_feat`, i.e. the 28.9s
  `prompt_ext.wav`, because that is what `infer_segment` already had in hand. But the 30s ceiling
  exists so `pt_mel` stays aligned with `pt_content_feat` for the *in-context prompt*; a **retrieval
  pool never enters the model**, so it is not constrained at all. Propagating that limit to the pool
  was the actual bug behind the articulation damage.

  `encode_long()` (`soulxsinger_svc.py`) chunk-encodes at 28s and concatenates, keeping only each
  chunk's real frames. On this reference that is **8158 frames (163.2s) instead of 1500 (30.0s)**,
  and it is what makes α=0.2 viable:

  | pool | mean nn1 | p10 nn1 | frames with nn1<0.5 | top-10 pool frames cover |
  |---|---|---|---|---|
  | `prompt_ext.wav`, 28.9s | 0.6534 | 0.5057 | 137/1400 (9.8%) | 10.9% of source |
  | **`refsep2/lead.wav`, 163.1s** | **0.7232** | **0.5694** | **26/1400 (1.9%)** | **8.1%** |

  Unmatchable frames drop **5×**. At α=0.4 the full pool alone cut CER 0.3441 → 0.3059 and *raised*
  SIM +0.020 → +0.025, i.e. it improves both sides of the trade — but 0.3059 was still unusable,
  which is what forced the α sweep. Exposed as `sing.py --knn-full-pool` (**on by default**) →
  `inference_svc.py --knn_pool_wav` → `infer(knn_pool_wav=)`; the pool is encoded **once** in `infer`,
  outside the segment loop, since it is identical for every segment. `--no-knn-full-pool` restores the
  30s-truncated behaviour. Note whisper is non-causal, so a 28s chunk taken alone and the first 28s of
  a 60s file give different features (max|diff| 15.6) — irrelevant for retrieval, but do not expect
  bit-identity between the two paths.
- **The kNN pool gain is *pool size*, not pitch coverage — and adding the `ext` transposed segments to
  it is not worth doing.** The natural next step from the entry above was to append the 12.8s of
  PC-NSF material `ext` already writes to `seedvc/seg_*.wav`, since a retrieval pool has no 29s budget
  and the song spends 41.8% of its voiced frames above 380Hz where the reference has 10 frames.
  Measured on 绿色 (query = 9.2k active song frames, `nn1` = best cosine against the pool):

  | pool | mean nn1 | p10 | nn1<0.5 | ≥380Hz mean |
  |---|---|---|---|---|
  | 163s reference (current default) | 0.6997 | 0.5702 | 1.6% | 0.6874 |
  | + `ext` segments (12.8s) | 0.7031 | 0.5735 | 1.4% | 0.6939 |
  | + whole reference transposed +5st (163s) | **0.7125** | **0.5895** | **0.8%** | 0.7000 |

  The `ext` version moves +0.0034 — an 7.8% larger pool buys almost nothing, so the obvious reading is
  "too little material". Scaling it up to a full 163s transposed copy does help (+0.0128, unmatchable
  halved), **but the per-band breakdown kills the pitch-coverage explanation**: the 0-260Hz bucket
  gains +0.0121, the same as 380-440Hz. A +5st pool cannot possibly supply better neighbours for
  *low* query frames.

  Decisive control — same experiment at −5st (moving the reference *away* from the song), all three
  addons generated with `librosa.effects.pitch_shift` so the arms are paired:

  | addon | global | 0-260Hz | 380-440Hz | nn1 taken from the addon (≥380Hz) |
  |---|---|---|---|---|
  | +5st | +0.0130 | +0.0131 | +0.0142 | 59.6% |
  | **−5st** | **+0.0104** | **+0.0108** | **+0.0117** | 60.3% |
  | +0st (resample only) | +0.0042 | +0.0045 | +0.0040 | 50.2% |

  **−5st captures 80% of the +5st gain**, and ~60% of nearest neighbours come from the addon in every
  band regardless of direction. The gain is "a bigger pool has a closer neighbour for everything",
  not "the high notes are now covered". Note even +0st (a pure DSP round-trip, no pitch change) buys
  +0.0042 — which also means the transposer's fidelity is not the binding constraint here.

  **Why**: kNN matches whisper-base final-layer features, an ASR encoder, so the space is organised by
  *phonetic content*, not F0. An /a/ sung at 250Hz and at 350Hz are near-neighbours there, which is
  why the baseline pool's ≥380Hz nn1 (0.687) was already no worse than its 0-260Hz nn1 (0.706) —
  pitch was never the pool's blind spot. **This also re-reads the entry above**: 30s→163s did not help
  by covering more pitch, it helped by covering more phones and prosody.

  Ceiling: +0.0128 on `nn1` scales through the historical ratio (p10 +0.064 → SIM +0.020) to about
  **SIM +0.004**, i.e. the noise floor, and that is already at 2× pool size with diminishing returns.
  Not implemented; `--knn-pool-wav` stays at `refsep2/lead.wav`. The real levers for kNN would be a
  longer reference (163s is near saturation) or richer phone coverage (needs re-recording).
- **Two plausible explanations for *which* frames kNN damages were both wrong; the cause is phone
  coverage.** Worth recording because both looked obvious and each took one measurement to kill:
  - *"It breaks unvoiced consonants"* (the /tʰ/ in 退), which would point at RVC's `protect` knob
    (blend original features back where F0=0). **Unvoiced frames are under-represented among the
    worst matches, not over**: 20% of the 25 lowest-nn1 frames vs a 46.3% baseline unvoiced rate,
    an enrichment of 0.43×. `protect` targets "electro-acoustic tearing on consonants", a different
    failure.
  - *"The pool lacks the high notes"* — the worst-matching frames are indeed high (390-468Hz). But
    binning source voiced frames by pitch against pool availability, the `ext` stage has already
    pushed the pool to median 314Hz / p90 419Hz, and only the >450Hz bin is genuinely thin (6.5:1,
    110 of 931 frames). Pitch is not the binding constraint.

  What is left is phonetic coverage, which is also what the literature says: NeuCoSVC's table gives
  14 unique phones at 5s, 23 at 10s, 38 at 30s, 46 at 60s, 49 at 90s, and kNN-VC's authors state
  that below 30s of reference "intelligibility and target speaker similarity decrease to a point
  where the more complex baselines perform better", saturating around 5 minutes. A 28.9s pool sits
  exactly at the bad end of that curve, which is why enlarging it to 163s helped and why the residual
  damage at high α is irreducible by tuning.
- **The mechanism behind the kNN gain is NOT confirmed, and the obvious prediction failed.** If kNN
  works by swapping the source residual for the target's, then amplifying the condition should stop
  pushing the output toward the original singer, so high `cfg` should hurt *less* with kNN on. Tested
  directly (seed 1234, song_2): the cfg 1.0 → 3.0 penalty is **−0.0331 at α=0.0 and −0.0455 at
  α=0.4** — slightly *worse*, and both curves keep the same shape (peak at cfg=1.0, worst at cfg=3.0).
  So the residual-substitution story is falsified as stated; kNN's α=0.4 gain at cfg=3.0 is only
  +0.0026, i.e. the two knobs are not attacking the same quantity. A plausible alternative is that kNN
  mainly pulls the condition's *statistics* toward the prompt's manifold (a mild smoothing/regularising
  effect). Recorded as unexplained rather than papered over: the effect replicates, the explanation
  does not. **The smoothing half of that guess is now confirmed and the "which would also explain why
  drift rises" half is refuted** — see the next entry: at α=0.2 kNN makes the condition *smoother*
  frame-to-frame, so the measured drift rise must originate in the CFM decoder or the timbre channel,
  not in the candidate selection.
- **kNN at α=0.2 is a *smoother*, not a noise source — which kills the whole "add a temporal constraint"
  family (kNN-SVC, arXiv 2504.05686).** The plan was to port its autoregressive candidate reselection
  (`L = L_src + m·L_concat`, restricting frame *t*'s candidates to pool-continuations of frame *t−1*'s),
  motivated by the local observation that raising `--knn-alpha` raises drift. Measuring the condition
  tensor directly refuted the premise before any of it was implemented. Frame-to-frame jump
  (adjacent-frame `1−cos`, active frames only), 4/4 songs:

  | song | source feats | α=0.2 mixed | ratio | α=1.0 |
  |---|---|---|---|---|
  | 绿色 | 0.08661 | **0.07786** | 0.90× | 0.10854 (1.25×) |
  | 爱笑的眼睛 | 0.08997 | **0.08156** | 0.91× | — |
  | 冬眠 | 0.08974 | **0.07992** | 0.89× | — |
  | 最后一页 | 0.08030 | **0.07200** | 0.90× | — |

  The α curve is a **U**: 0.94× at α=0.1, minimum 0.86× at α=0.4, back to 0.97× at α=0.7, and only
  α=1.0 is rougher than the source (+25%). The default 0.2 sits on the smoothing side. Three more
  independent refutations, each cheap:
  - **The ideal temporal constraint makes it worse.** A greedy autoregressive selector (prefer
    `prev+1` when it appears in top-16) lifts strict-continuity from 9-17% to 55-63% at almost no
    cosine cost (−0.0003…+0.0023), and frame jump *rises* 2.9-3.2% on all four songs.
  - **The long sustained notes the paper targets do not exist in this material.** Stable runs
    (adjacent-frame F0 change <0.5 semitone): `≥0.5s` covers 5.9-48.3% of active frames, `≥1.0s` is
    ~0%, longest run 0.62-1.48s. Mandarin pop diction does not leave room for the multi-second notes
    the paper's Figure 3 is about.
  - **α=0.2 only edits ~1% of the condition, so every reordering variant is fighting over that 1%.**
    `srcKeep` = cos(mixed, source): 0.9914 at the current k=4, and the whole variant space
    (k=1 → 0.9876, consistency-filter τ=0.9 → 0.9879, softmax-weighted → 0.9911, k=16 → 0.9934) spans
    **0.6%**. Scaling by the known anchor (srcKeep 1.0000→0.9914 bought +0.0145 SIM) caps the most
    aggressive variant at **+0.0064**, and even that assumes "further from source ⇒ higher SIM"
    monotonically, which the α sweep already disproves.

  **Confirmed end to end with the variant that fully satisfies the paper's objective.** `k=1` makes
  every substituted vector a genuine pool frame (`onManifold` 0.9395 → **1.0000**) and needs no code
  change at all (`--knn-k 1`). cfg=1.0, seed 1234, paired: 绿色 0.6034 → 0.6028 (**−0.0006**,
  36/70 wins, p=0.905), 爱笑的眼睛 0.5635 → 0.5624 (**−0.0011**, 31/75, p=0.165). The paper's core
  ask is free here, and worthless.

  **Root cause is architectural, and it generalises.** In kNN-VC/kNN-SVC the retrieved frames go
  **straight into the vocoder**, so concatenation smoothness *is* waveform smoothness. Here they are
  blended at α=0.2 into the **CFM condition**, with the source at 80%, followed by 32 Euler steps and
  a separate `pt_mel` timbre channel. Edit 1% of the condition and the ordering stops mattering. This
  also pre-empts the Sinkhorn-OT idea (E2 in OPTIMIZATION.md): it manipulates the same quantity, and
  the k=1…k=16 sweep already spans Gini 0.95→0.79 with SIM flat-to-negative. **Corollary: a high Gini
  in the kNN pool is not a defect.** Treating it as one came from analogy with `ext`'s reuse penalty
  (where it genuinely worked) — but `ext` rewrites the 25s of audio that whisper actually encodes,
  while kNN nudges a tensor the CFM then dilutes 5×.

  Worth keeping from the probes: the current k=4 candidates are **spatially scattered** — 78.5% of
  frames draw their 4 neighbours from pool positions >1s apart, and those candidates agree with each
  other at only **0.742** cosine, i.e. the average really is a blended timbre rather than any real
  frame. That is a genuine defect in the paper's framing and a harmless one here. Constraining
  candidates to a ±1.5s pool window costs **0.0756** of cosine, 30-250× the frame-level constraint's
  cost, so that variant is not worth trying either.
- **`reverse_diffusion_v2` is dead code for a different training recipe — do not wire it up.** `flow_matching.py:320` looks like a free stability win: it builds `xt_input = cat([prompt, xt, prompt])`, i.e. the prompt bracketing the target on *both* sides instead of prefix-only, which is exactly the shape of fix you would want for cross-segment timbre drift. It is unreachable (nothing calls it) and it must stay that way: the training path builds the prompt mask as a contiguous *prefix* (`forward_diffusion`, `flow_matching.py:131-144` — `prompt_len = randint(...)` then `is_prompt[col_indices < prompt_len] = 1`), so the model has never seen a suffix prompt and would be OOD from the first Euler step. It also expects `target_len = cond.shape[1] − prompt_len*2`, which does not match how `CFMDecoder.reverse_diffusion` assembles the condition (`decoder.py:20`, prefix layout). Ruled out by reading the training code rather than by burning a run — worth remembering as the pattern for any other "unused upstream method" that shows up.
- **The source material bounds everything.** This reference had 0.4s above 370Hz while the song spent 44.3% of its voiced frames there. Reference *selection* only reshuffles what exists; recording 30-60s of higher-register material is worth more than any parameter. **Update (2026-08): the material can now be *extended*, not just re-selected** — PC-NSF transposition covers 221-394Hz at SIM ≥0.83 (see "Reference material pitch extension"); TTS voice design cannot substitute (SIM 0.535, synthetic timbre).
- **ext 占比/移调上限/最小分段的单因素扫点（song_0, 2026-08, seed 1234 固定）—— 三个维度里只有 ext 占比有效应。** 三组实验各 5-7 个点、全部 `--seed 1234`（配对），eval_sim 3s 窗 vs 真实语音：

  **Exp1: ext 占比**（`--anchor-min-ratio`，0% = 直接喂 prompt.wav 跳过 ext）。粗扫 7 点 + 精细加密 8 点（0.20-0.65 区间，全部 seed 1234 配对）:

  | ratio | 0.0 | 0.15 | 0.20 | 0.25 | 0.30 | **0.35** | **0.40** | 0.45 |
  |---|---|---|---|---|---|---|---|---|
  | ext占比 | 100% | 85% | 80% | 75% | 70% | **65%** | **60%** | 55% |
  | SIM mean | 0.5660 | 0.5690 | 0.5694 | 0.5710 | 0.5681 | **0.5743** | **0.5737** | 0.5710 |

  | ratio | 0.50 | 0.55 | 0.60 | 0.65 | 0.75 | 0.90 | (0% ext) |
  |---|---|---|---|---|---|---|---|
  | ext占比 | 50% | 45% (默认) | 40% | 35% | 25% | 14% | 0% |
  | SIM mean | 0.5657 | 0.5673 | 0.5657 | 0.5642 | 0.5521 | 0.5489 | **0.5433** |

  峰是 **ratio 0.35-0.40（ext 60-65%）的平台**：r035 与 r040 各 3 seed（+1234/2/3）均值 0.5694 vs 0.5686，差 0.0008 低于噪声底，**不可区分**；而 r035 vs r055 3 seed 配对 +0.0070/+0.0045/+0.0033 三战三胜（均值 0.5694 vs 0.5644），超过噪声底 ±0.004。形状是非单调的山：两头都掉（0% ext 掉 0.024，14% 掉 0.018）。`eval_stab` 同协议：r035 1.5s 窗 mean 0.5264 vs r055 0.5213（+0.0051），drift 0.1229 vs 0.1204（+0.0025）、dropouts 13 vs 12 —— 稳定性代价很小；r040 与 r035 在 eval_stab 上也几乎相同（0.5261/0.5264）。**注意 0% ext（纯锚点 25s）是最差的 0.5433** —— 这个参考的音域本身不够（med 274 vs 歌曲 369），ext 高音素材确实是必要的，但比例不是越大越好。

  **跨歌复现（song_2 + song_6, 2026-08）—— 峰没有复制，song_2 是明确反例，不改默认。** 同协议 3 seed 配对（r035/r040/r055 × 1234/2/3）:

  | 歌曲 | 错配 | r035 (65%) | r040 (60%) | r055 (45%默认) | r035 vs 默认 |
  |---|---|---|---|---|---|
  | song_0 | +9.1st | 0.5694 | 0.5686 | 0.5644 | **+0.0049** 三战三胜 |
  | song_2 | +7.6st | 0.5934 | 0.5918 | **0.5995** | **−0.0061 三战三负** |
  | song_6 | +11.2st | 0.5369 | 0.5331 | 0.5333 | +0.0036 三战三胜(噪声边) |

  song_2 上 r035/r040 都比默认差，`eval_stab` 同方向（r055 0.5399 vs r035 0.5334，且 r035 dropouts 更多）—— 反例在稳定性协议上同样成立，不是单协议假象。三首歌的 r035/r040 prompt_ext 音高分布几乎相同（med 340-350 vs 314-318，ext 18.8s 全填满），排除产物差异。**唯一能解释的方向是歌曲的 prompt-vs-song 音域接近度**：song_2 的 prompt med 277Hz vs song 318Hz 只差 2.4st，素材本身覆盖够，vocoder 重合成的损伤（过一遍 NSF 掉 ~0.02-0.045 SIM）> 高音覆盖收益；song_0/song_6 差 4.6-5.1st，覆盖收益 > 损伤。**结论：默认 0.55 保持不动**；若歌曲与参考音域差 >~4.5st 且想要那 ~+0.004，可对单首歌调 0.35，但收益量级≈噪声底，不值得为此引入歌曲级配置。

  **Exp2: 移调上限**（`--ext-max-shift`，固定 ratio 0.55）:

  | max-shift | 2 | 3 | 4 | **5 (默认)** | 6 |
  |---|---|---|---|---|---|
  | SIM mean | 0.5548 | 0.5602 | 0.5647 | **0.5673** | 0.5598 |

  单调升到 5 ~~再跌 —— **默认 5 已经是最优，无需动**~~。**2026-08 更正**：该扫点在 
  dw=0 时代测得，当时素材被锁在 med 最高的少数段。dw=3 改变了竞争结构，ms=8 能解锁
  低音段参与高音格（med 283-285Hz 的段以 +8st 达到 450+Hz）。A/B 三首（seed 1234 配对）：
  
  | 歌曲 | 初始复用 | ms5→ms8 唯一源 | 复用 | 增益 | wins | p |
  |---|---|---|---|---|---|---|
  | 爱笑的眼睛 | 1.41× | 17→21 | 1.41→1.10× | **+0.0046** | 50/75 | 0.005 |
  | 冬眠 | 1.37× | 19→21 | 1.37→1.10× | **+0.0072** | 46/60 | <0.0001 |
  | 绿色（已饱和） | 1.05× | 22→22 | 1.05→1.05× | **+0.0033** | 49/69 | 0.0006 |
  
  3/3 全正且全部统计显著。绿色已饱和但仍正向，说明即使不解锁新段，高音格覆盖范围扩展
  本身也有轻微收益。**默认改为 8**（2026-08，`sing.py --ext-max-shift` / 
  `w_ext.py MAX_SHIFT_DEFAULT`）。2-4 的缺口恰好是 song_0 歌曲 p90 465Hz vs 素材+shift 上限 378→453Hz 的覆盖不足；~~6 的回落与 AGENTS.md 逐段扫点（+6 worst 掉到 0.81）一致~~（该结论作废，单段保真度不是 ext 的判据，因为 prompt 是池化平均使用）。

  **Exp3: 最小分段**（`--select-min-sec`，固定 ratio 0.55）:

  | min-sec | 0.15 | **0.30 (默认)** | 0.50 | 0.80 | 1.20 |
  |---|---|---|---|---|---|
  | SIM mean | 0.5668 | **0.5673** | 0.5653 | 0.5635 | 0.5622 |

  近于平坦：0.15-0.30 持平（0.15 不产生新候选 —— 现有最短候选就是 0.302s，md5 相同），之后单调慢降 −0.005。**0.30 默认已经是最优**；1.20 的下降部分来自素材池缩到 11 段/17.7s 导致 prompt 变短。短分段不伤 SIM —— 之前担心的"27 段 0.9s 碎片拼接伤音色"没有发生（prompt 是 whisper 一次性编码的整段 mel，不是逐段推理）。**用户提议的"用短窗峰值代替中位数选段"仍然不可行**（见上一条的 L1 模拟：峰值音高高估整段移调后的落格秒数，三首歌 L1 全部变差），但"峰值只用于选择不用于移调"的变体未测。

- **`--anchor-min-ratio` is a *cap*, not a floor, and it binds on every run — an earlier note in this file claimed the opposite and was wrong.** `w_ext.py:165-172` reads `if anchor_dur > max_sec * ratio: 裁剪`, so the flag sets the **maximum** share the clean anchor may occupy, and everything left over goes to the transposed material. At the defaults (`max_sec=29`, `ratio=0.55`) the anchor is clamped to **15.95s** and ext receives **13.05s (45%)** on every song whose `prompt.wav` exceeds 15.95s — which is all of them (`prompt.wav` is 19.6-25.0s). Verified in three logs (`song_0`, `song_2`, `song_6`), each printing `锚点裁剪到 15.95s（max_sec 29s × anchor_min_ratio 0.55）`, and by counting F0 frames in the shipped `prompt_ext_f0.npy`. The superseded note asserted ext gets only 3.8-4.1s and that the clamp was unreachable; it had the inequality backwards. Consequences: (1) `--anchor-min-ratio` is the ext-share knob — sweeping it 0.0→1.0 spans "all transposed" to "all clean", and no code change is needed to test ext share; (2) lowering `--max-prompt` does **not** buy ext more room at ratio 0.55 (the clamp already decides the split), it only shortens the anchor once `prompt.wav` drops below 15.95s; (3) above `ratio ≈ 0.86` the material itself becomes the binding constraint (a 24.96s `prompt.wav` cannot fill 29·0.9), so ratios 0.90 and 1.00 produce identical prompts.
- **ext 的选段排序键把素材池锁死在顶端 8%，加一个复用惩罚就换来 3 首歌全正的增益（2026-08，`--ext-div-weight` 默认 3.0）。** 由听感发现：ext 追加的部分"多样性严重不足，基本都是拿一两个高音段落升调拼接"。量化后比描述更严重 —— 10 首歌里 ext 平均只用到 **11-16 个源段**，复用 1.56-2.55×（28 个移调段来自 11 个源），且跨 10 首歌被选中的源段几乎是同一批（`seg 59/54/39/65/29/114/53/115` 每首都在）。

  **根因不是贪心策略，是排序键的副作用。** `w_ext.py` 原排序是 `(inside, |shift|, 离格中心距离)`，`|shift|` 为第二优先，于是填 380Hz 格时 339Hz 的段（+2st）永远排在 248Hz 的段（+7st→钳到+5）之前。绿色的候选池有 **143 段/95.6s**，但 med≥290Hz 的只有 **15 段/7.7s**，要填 13.05s 的 ext 预算 → 数学上必然复用 2.2×。池里 116-339Hz 的其余 92% 素材从未参与竞争。这个排序本身有实测依据（每半音 0.014-0.020 SIM，每个输出音高桶内 `corr(|shift|,SIM)` 都在 −0.76…−0.99），但它把**单段保真度**当成唯一目标，而 prompt 是**池化平均使用**的 —— whisper 看到的音素多样性只有 13 段的量，正好落在 NeuCoSVC/kNN-VC 那条"<30s 参考吐字与相似度双降"的曲线最差端。

  修法：`used_src[idx]` 计数 + 排序键改为 `(inside, |shift| + div_weight·已用次数, 离格中心距离)`，即把"复用一次同源段"折算成等价半音数，让算法能用更大的移调换更多不同源段。因惩罚随选择过程变化，改为**每次重算 cost 的逐个贪心**而非一次性排序。A/B（3 首 × `div_w ∈ {0,1,3}`，`--cfg 1.0 --seed 1234`，`paired_eval` 3s 公共窗）:

  | 歌曲 | dw=0 | dw=1 | **dw=3** | gain(dw3) | wins | p_sign |
  |---|---|---|---|---|---|---|
  | 绿色 | 0.5925 | 0.6002 | **0.6034** | **+0.0109** | 60/70 | 0.0000 |
  | 爱笑的眼睛 | 0.5583 | 0.5602 | **0.5635** | +0.0052 | 54/75 | 0.0002 |
  | 冬眠 | 0.5454 | 0.5486 | **0.5489** | +0.0034 | 40/60 | 0.0135 |
  | 均值 | | +0.0043 | | **+0.0065** | | |

  9/9 arm 全正，`dw=3 > dw=1` 在 3/3 上一致，符号检验全部显著。**`dw=0` 与旧代码 `prompt_ext.wav` md5 完全一致**，所以重构无副作用、基线可信。

  **机制是干净的：唯一变化的就是源段数。** ext 总时长恒定 ~13s，均 |shift| 几乎不动（绿色 3.82→3.74，爱笑 3.57→3.54，冬眠 3.11→3.46），而唯一源段 13→22 / 11→17 / 13→19，复用 2.15→1.05 / 2.55→1.41 / 2.15→1.37。增益最大的绿色正是多样性提升最多的那首。**即"旧排序的复用不是移调预算的必要代价，纯粹是排序键的浪费"** —— 换来 9 个额外源段几乎没有额外移调损失。

  代价两项都查过（kNN 的教训：动内容侧必须查 CER）：
  - **CER 无系统性代价**，方向不一致 —— 绿色 +0.0141、爱笑的眼睛 **−0.0377**、冬眠 +0.0109，均值 −0.004。与 kNN α=0.4 那种"3 首全负、+5.9 点"性质完全不同。
  - **稳定性代价极小**：drift +0.0016…+0.0020（本文件其他杠杆都是 +0.006…+0.018），dropouts 基本不变（11→11 / 5→5 / 8→7），p90drift 两首反而改善（0.1959→0.1888、0.1924→0.1840）。

  未测：`dw>3`（绿色在 dw=3 已到复用 1.05×，接近饱和，但另两首还有空间）、`dw` 与 `--ext-max-shift` 的交互（放宽多样性后是否值得把 max-shift 提到 6）。**2026-08 已测：见下一条「ext 移调上限扩展」。**
- **`ext` 阶段 2.2× 提速：把逐 shift 组 fork 合并成一次进程，输出 bit-identical（2026-08）。** 该阶段耗时几乎与音频长度无关 —— 逐组计时暴露了形状：0.30s 的组 4.7s、5.94s 的组 4.2s。拆开量：`pitch_transpose.py` 单次进程**固定开销 2.77s**（import torch 1.02 + import 项目模块 0.99 + load RMVPE 0.53 + load NSF-HiFiGAN 0.11），净计算只有 **0.87s**（mel 0.00 + rmvpe f0 0.11 + vocoder 0.76，5.94s 音频）。典型 5 组 = **~14s 纯开销**。

  修法：新增 `tools/pcnsf/scripts/multi_transpose.py`（接 `--jobs` 的 JSON 列表，一个进程跑完所有 `(wav, shift)`），`w_ext.py` 先把各组输入全部落盘再一次性调用（`nsf_shift_many`）。**注意 fork 本身是省不掉的** —— `tools/pcnsf/.venv` 是 torch 2.13+**cpu**、与 `w_ext.py` 所在的 SoulX venv 不同，所以只能把 N 次降到 1 次，不能降到 0。全 10 首实测：

  | | 逐组 fork（旧） | 批量（新） | 加速 |
  |---|---|---|---|
  | 单首范围 | 15.2-25.0s | 8.3-8.6s | 1.77-2.90× |
  | 10 首合计 | **188.4s** | **84.6s** | **2.22×** |

  新路径耗时几乎恒定（8.3-8.6s），因为剩下的就是那一次固定开销。**10/10 首 `prompt_ext.wav` md5 bit-identical**，端到端 SVC 后 `vocal.wav` md5 与 SIM（0.6034/0.6348）也完全不变。`multi_transpose.py` 的 per-job 计算是逐行照抄 `pitch_transpose.py` 的 `nsfhifigan` 分支（同样的 librosa dtype / `get_mel_torch` 参数 / `rmvpe.get_pitch` / `f0 *= 2**(n/12)` / `save_wav`），这是 bit-identical 的原因。`--shifter seedvc` 没有对应多 job 入口，仍走逐组 fork。

  **验证这类改动时有个陷阱：`F0Extractor` 本身不可复现。** 同一进程内对同一文件连跑 4 次，`max|diff|` 达 2.2e-4（相对误差 2.5e-6），所以 `prompt_ext_f0.npy` **永远不会** byte-equal，一度让我误判成改动引入了偏差。判据要用 `prompt_ext.wav` 的 md5（它是确定性的）加下游 `vocal.wav` + SIM。另一个坑：对照基线必须显式带上 `--div-weight 3.0`，否则拿到的是 `dw=0` 时代的存档产物（`DIV_WEIGHT_DEFAULT` 仍是 0.0，3.0 由 `sing.py` 的 `Cfg.ext_div_weight` 下发），段数会从 26 变 28，看起来像本次改动的锅。
- **选段从两阶段贪心换成一个 MILP（`--select-method cov_anchor`）—— clip 上 3 首歌 0 负，但全曲 10 首交叉验证后增益归零，默认已改回 `greedy`。** 原流水线里 `w_select` 先贪心选锚点、`w_ext` 再贪心填频段缺口，两者互不知情，于是同一源段可被两阶段各用一次（原声 + 移调版）。候选池其实是**互不重叠的分割**（绿色 143 段 / 95.6s，重叠对数 0，并集 = 段长和），所以"一段素材出现两次"可以精确定义并消除。`singing/ext_milp.py` 把两阶段合并成一个 (段, 移调) 指派问题：变量 `x[i,s] ∈ {0,1}`（143 段 × 13 shift = 1859 个 0-1 变量），"每段最多用一次" = `Σ_s x[i,s] ≤ 1`，`scipy.milp` 求解，词典序优化覆盖缺口 → 时长加权 `|shift|`。

  用户提的优先级是 **①音域覆盖 > ②不重叠 > ③移调最小**，实测只有一半成立。跨 3 首歌、clip 90s / 6 seed / 统一窗口集、`--seed` 配对：

  | 臂 | 策略 | 绿色 | 冬眠 | 爱笑的眼睛 | 均值 | 负 |
  |---|---|---|---|---|---|---|
  | `shift_first` | ③ 先最小化移调 | −0.0195 | −0.0013 | −0.0241 | −0.0150 | **3** |
  | `satisfice` 80% | 覆盖够 80% 即止 | −0.0022 | +0.0018 | +0.0070 | +0.0022 | 1 |
  | `satisfice` 90% | 覆盖够 90% 即止 | −0.0010 | +0.0023 | +0.0103 | +0.0039 | 1 |
  | **`cov_anchor`** | **① + 干净锚点 ≥55%** | **+0.0038** | **−0.0000** | **+0.0112** | **+0.0050** | **0** |
  | `cov_only` | ① 纯覆盖优先 | −0.0009 | −0.0033 | +0.0140 | +0.0033 | 2 |

  三条可迁移的结论：
  - **③ 不能排第一。** `shift_first` 会退化：移调压到 0 之后全部段都是 0st，覆盖塌到 33-54%，加上"不重叠"连 45s 都填不满（只有 28-34s，素材用光），三首歌全负。
  - **① 排第一有效，但必须配干净锚点下限。** 纯覆盖优先会把 0st 素材压到 27-36%（`base` 是 55%），两首歌因此转负；加上 `0st 段 ≥ 55% 总时长` 的硬约束后 0 负。这与本文件已有的"过一遍 vocoder 就掉 0.02-0.045 SIM"一致。
  - **② 是免费的，而且让求解更快。** 加上不重复约束后绿色 66s 求解完，允许复用反而要 227s —— 它大幅收紧了 LP 松弛。所以没必要把它当成要权衡的成本。

  端到端验证（`sing.py --stages ext` 走新默认，绿色 clip 6 seed）：**base 0.6268 → 新默认 0.6347，+0.0079，5/6 seed 正，p=0.0002**，比实验期的 `a55` 臂还高 +0.0041（收紧了 MILP 的 `mip_rel_gap`）。`--select-method greedy` 保留旧路径，**md5 与改动前产物 bit-identical**（重构无副作用，可用于回归）。
- **然后把它放到全曲 10 首上，增益消失 —— clip 协议过拟合了那 3 首调参歌，`cov_anchor` 已撤回默认。** 全量重跑（`--seed 1234` 与旧基线严格配对，`--pitch-shift 0`，`paired_eval` 公共窗）：

  | | 均值 | 正/负 | t | 符号检验 |
  |---|---|---|---|---|
  | 全 10 首 | **+0.0012** | **5/5** | 0.49 | **p=1.000** |

  95% CI ≈ [−0.0044, +0.0069]，**包含 0**。逐首：爱笑的眼睛 +0.0129、绿色 +0.0094、云烟成雨 +0.0092、爱的回归线 +0.0053、陪你看星星 +0.0047｜冬眠 −0.0033、那女孩对我说 −0.0051、白色毛衣 −0.0064、追光者 −0.0064、最后一页 −0.0080。

  **调参歌 vs 留出歌把过拟合摊开了**，这是本轮最重要的一个数字：

  | | 均值 | 正 |
  |---|---|---|
  | 3 首调参歌（clip 实验用过：绿色/冬眠/爱笑的眼睛） | **+0.0063** | 2/3 |
  | **7 首留出歌（从未参与调参）** | **−0.0010** | **3/7** |

  clip 上选出 `cov_anchor` 的那批 15 臂实验，全部只在这 3 首歌上跑过；全曲复现时它们仍是表现最好的 3 首（+0.0094/−0.0033/+0.0129，与 clip 的 +0.0038/−0.0000/+0.0112 同号且同序）。当时把它读成"clip 过拟合了调参歌"——**这个解释被下一轮全曲双臂复跑推翻了**：7 首留出歌里有 5 首显著，其中 3 首 MILP 胜、2 首 greedy 胜，"留出歌均值为负"是组内相消的假象，真正的规律是效应按歌分向（见下一条）。正确表述：**clip 只测了 3 首歌，不足以定默认值；n=3 定策略、n=10 才勉强够看出"没效果"的结论本身也要修正。**

  代价那一侧是明确的：`ext` 阶段从 **12.9s 涨到 253.6s（19.7×，最慢绿色 584.8s）**，MILP 求解是全部开销。一个零收益的改动换 20 倍耗时，**默认已改回 `--select-method greedy`**；`cov_anchor` 保留为可选项。

  三条可迁移的教训：(1) **clip 用来扫参数可以，用来定默认不行** —— 它只降低单次成本，不增加歌曲样本量，而歌曲间方差（sd 0.0079）才是这里的主导误差；(2) 报"跨 N 首歌 0 负"时必须写清那 N 首是否就是调参用的那 N 首；(3) `corr(gain, 唯一源数) = −0.831` 看起来又是个漂亮的预测因子，**但不要用** —— MILP 把每首歌都推到 60-70 段，自变量跨度只有 60→70，且 n=10，与"覆盖率 +0.771"那次被一个极端点拉出来的情形同构；顺带一提 `corr(gain, MILP 覆盖%) = −0.381` 是**负**的，这是覆盖率不转化为 SIM 的第四次否证。
- **全曲双臂复跑后对结论的修正：「均值 +0.0012、5 正 5 负」是错的读法 —— 真实现象是「两首歌组，组内效应又大又一致」。** 上一轮把 MILP 和 greedy 产物直接归档比较（两版都在 `data/songs/<名>/` 与 `data/songs/<名>_greedy/`），用 `paired_eval` 对 10 首全曲逐窗重比（3s 窗、公共 RMS 门、与 `eval_sim` 同协议），逐首：

  | 歌 | gain | wins/n | win% | p_sign | | 歌 | gain | wins/n | win% | p_sign |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 爱笑的眼睛 | **+0.0129** | 62/75 | 82.7% | 0.0000 | | 追光者 | **−0.0064** | 14/53 | 26.4% | 0.0008 |
  | 绿色 | **+0.0094** | 44/70 | 62.9% | 0.0414 | | 白色毛衣 | **−0.0064** | 19/59 | 32.2% | 0.0086 |
  | 云烟成雨 | **+0.0092** | 43/63 | 68.3% | 0.0052 | | 最后一页 | **−0.0080** | 20/67 | 29.9% | 0.0013 |
  | 爱的回归线 | +0.0053 | 47/70 | 67.1% | 0.0056 | | 冬眠 | −0.0033 | 25/60 | 41.7% | 0.2451 |
  | 陪你看星星 | +0.0047 | 32/52 | 61.5% | 0.1263 | | 那女孩对我说 | −0.0053 | 34/82 | 41.5% | 0.1507 |

  两组内部的效应都是又大又一致（MILP 胜的 5 首：mean +0.0083、69.1% 窗同向；greedy 胜的 5 首：mean −0.0059、65.1% 窗同向）——**这不可能都是噪声**。`t=0.49、p=1.000` 说明的是"方向随机"，但按窗口的符号检验（5 首 p<0.01、其余 2 首 p≈0.13）说明的是"每首歌内部效应真实"——两者同时成立，唯一的解释是**效应存在但按歌分向**。clip 的 3 首恰好 2 首落在 MILP 组（绿色 +0.0038、爱笑的眼睛 +0.0112），才让人误以为全覆盖。

  **分向的机制没找到；四个候选变量全部否掉，包括一度写进本文件的 `song_med`。** 逐个查（全部零 GPU，只读已有日志与 `prompt_stats.json`）：

  | 候选 | 检验 | 结论 |
  |---|---|---|
  | select∩ext 交集（同段被锚点与 ext 各用一次） | 已有**干预实验** `--ext-exclude-anchor`：3 首均值 +0.0034，仅绿色(重叠62%)显著；本轮 `corr(gain, 重复占ext%) = +0.051` | 影响小，且重叠最高的两首(76.6%/76.8%)都是 MILP 胜、58%/53% 的两首是 greedy 胜，无规律 |
  | **移调时长占比** | 两版都被 `anchor_min_ratio 0.55` 锁死在 **44.2-45.3%** | **压根不是变量**，跨度 0.5pp；在它上面算出的 `corr = −0.552` 是拟合噪声 |
  | 移调幅度 | `corr(gain, Δ移调段 w\|shift\|) = −0.119` | 不解释 |
  | `song_med`（歌的调性） | 最佳阈值 >344Hz 可分 8/10，`corr = +0.516`，**但 LOO 掉到 5/10 = 瞎猜基线** | **否证** |

  `song_med` 那条值得单独记，因为它看起来最像规律：最高的 3 首全 MILP 胜、最低的 2 首全 greedy 胜。但中间段是乱的 —— **陪你看星星 318Hz(MILP 胜 +0.0047) 与白色毛衣 319Hz(greedy 胜 −0.0064) 只差 1Hz 而方向相反**，云烟成雨 308Hz 还是 MILP 胜 +0.0092。那个 8/10 全靠挑阈值挑出来。这是本文件第三次踩"n≈10 上挑阈值拟合规律"（前两次：`corr(8st+桶占比, 最大增益)=+0.995`、锚点占比线性拟合），**LOO 是这类规律的标准反例检验，先做再信**。

  另有一个**必须记住的口径错误**：MILP 的移调段其实移得**更狠**（移调段内部 w|shift| 3.59-7.84 vs greedy 4.43-5.88，8/10 首更大），但按"全 prompt 加权"算却更小（1.59-3.52 vs 1.98-2.79）——因为 MILP 段数多一倍(60-70 vs 26-35)、0st 段占比高，把均值稀释了。**比较移调幅度必须只在移调段内部算**，否则会得出方向完全相反的结论（我据此错判过一次）。

  还需注意 10 首里只有 4 首（爱笑的眼睛 +0.0129、绿色 +0.0094、云烟成雨 +0.0092、最后一页 −0.0080）的效应确实超出单 seed 噪声底 ±0.004；其余 6 首绝对值都在 0.0033-0.0064，**单 seed 下方向本身就不可靠**。要定机制得先给这几首补 seed。
  
  **9/10 首 bit-identical 复现，说明整条链（select→ext→svc→eval）是确定性的**；唯一例外 那女孩对我说 的旧归档 vocal 与其 sing.log 时间戳不符（12:36 vs 13:52，归档时目录被复用），用今天的重跑作基线后结论不变（−0.0053）。**这也修正了上一轮的表述**："clip 协议过拟合那 3 首调参歌"解释不了"5 首留出歌显著为正"的事实 —— 正确表述是"clip 只测了 3 首歌，分向现象需要全曲才看得见"。
- **两条路径的分叉点只有 `ext`，但 MILP 不是在 select 结果上做增量 —— 它把 `prompt.wav` 整个丢掉重解。** 逐阶段 md5 对比（绿色）：`sep`/`ref-sep` bit-identical（同一缓存），`select` 的 `selected` 段列表**完全相同**（28/28，idx 与 start 全同）、`prompt.wav` md5 一致，`*_f0.npy` 与 `*_stats.json` 的差异只是路径串与 `F0Extractor` 的 1e-6 浮点噪声（本文件已记录它永不 byte-equal）。真正分叉的是 `prompt_ext.wav`。

  但"分叉只在 ext"容易被误读成"MILP 在锚点之上填缺口"。实际是 `w_ext.py:233-235` 把 `anchor` 置成 `np.zeros(0)`、**完全不读 `prompt.wav`**，直接把 143 段全池交给 `ext_milp.solve()`，锚点（0st 段）由求解器自己重选。实测 MILP 的 35 个 0st 段与 `w_select` 的 28 段只重合 **16 个**（冬眠 14/27），只是时长同样凑到 `45×0.55=24.75s` 上限。**所以 MILP 路径下 `prompt.wav` 是死产物，`select` 阶段对它的唯一贡献是 `cands_meta.json`。**

  `no_repeat` 约束（`Σ_s x[i,s] ≤ 1`）的作用域也常被误解：它只禁止**同一次求解内**一段被用两次（实测 `0st ∩ 移调 = 0`），**不会因为 select 选过某段就禁用它** —— 绿色 select 的 28 段里 25 段被 MILP 再次用到（16 当锚点、9 当移调段）。反而是 **greedy 才会重复用段**：绿色有 16 段在 `prompt_ext` 里出现两次（原声进锚点 + 移调版进 ext），ext 内部还有 6 段用了多个 shift。消除这个重复正是 MILP 的设计动机，也是 `--ext-exclude-anchor` 只对 greedy 有意义的原因。

  两版 ext 的结构差异（10 首一致，绿色为例）：

  | | greedy | MILP |
  |---|---|---|
  | ext 段数 / 时长 | 30 段 / 17.5s（45s 预算剩 ~20s 未用） | **67 段 / 44.9s（填满）** |
  | 源素材复用 | ~1.2×（同段多 shift + 与锚点重复） | **0（硬约束）** |
  | shift 分布 | 集中 +3/+4/+5 | 双峰：35 段 **+0** + 21 段 **+8** |
  | 移调段内部 w\|shift\| | 5.71 | **7.44**（更狠，见上条口径说明） |
- **"覆盖率高就好"仍然不成立 —— 上面那个默认是配置调优，不是找到了机制。** 15 个臂上 `corr(Δ覆盖, gain) = +0.771` 看着很有说服力，但**去掉退化的 `shift_first` 臂后变成 −0.001**：那个相关完全由一个极端点拉出来。而且 Δ覆盖 与 Δw|shift| 本身相关 **+0.882**，这批臂在设计上就无法把两者分开。所以 `cov_anchor` 只是这批可行解里表现最好的配置，不能推论成"提高 prompt 频段覆盖能提分"——那个命题在本文件里已被独立否证两次（select 阶段前移移调：覆盖 75.8%→84.9%，SIM +0.0002；预移调整条参考 +5st：覆盖 50.6%→75.7%，SIM −0.0029）。
- **不要按歌曲类型切换选段策略 —— 分档规则被 leave-one-out 否证。** 每首歌的最优臂确实不同（冬眠→`satisfice90`、绿色→`cov_anchor`、爱笑的眼睛→`cov_only`），且 `corr(8st+ 桶占比, 最大可得增益) = +0.995` 看起来是个漂亮的分档依据（<10% / 10-25% / >25%）。同一批数据上"每歌选最优"能到 **+0.0067** vs 单一 `cov_anchor` 的 +0.0050。但那个 +0.995 和那个 +0.0067 都是**用同一批数据挑出来的**，必然过拟合：做 LOO（阈值只由另两首歌定，预测留出的那首该用哪个臂）后掉到 **+0.0034，反而低于单一 `cov_anchor`**。额外收益 +0.0017 本来就在 ±0.004 噪声底以下。**n=3 不足以支撑任何按歌分档的规则**，这也是本文件里第二次踩"3 个点拟合 2 个参数"（前一次是锚点占比线性拟合，同样被否）。
- **MILP 的碎片化不伤吐字（已用 CER 验证）。** 它选出的段中位长 0.57s、约 27 段 <0.5s，远碎于原贪心（本文件此前担心"短参考音素覆盖不足"）。`singing/eval_cer.py` 实测：绿色 **−0.0060**、冬眠 **−0.0380**（两首不升反降）、爱笑的眼睛 +0.0256。听感也确认吐字无问题。`corr(中位段长, ΔCER) = −0.19`、`corr(<0.5s 占比, ΔCER) = +0.20`，即碎片化与 CER 基本无关。
- **`prompt_ext` 总长的 29s 上限是历史包袱，不是架构限制；默认已改 45s（2026-08，跨 3 首歌）。** whisper 的 30s 硬顶只约束 `pt_content_feat`，而 `encode_long()` 补丁已绕开它；音色通道 `pt_mel` 直接从波形算（`soulxsinger_svc.py:301`），**从来没有 30s 限制**。真正的天花板是 LLaMA 的 RoPE `max_position_embeddings=4096`（`llama.py:142`）：`pt_frames + gt_max` 在 24kHz/hop480 下 = 45s→3650、50s→3900，都还有余量（60s 会超）。实测（90s 代表性 clip，`--cfg 1.0 --knn-alpha 0.2 --pitch-shift auto`，3 seed 配对，统一窗口集）：

  | 歌 | 29s | 45s | 50s | 29→45 | 45→50 |
  |---|---|---|---|---|---|
  | 绿色 | 0.6223 | **0.6452** | 0.6339 | **+0.0229** (59/72) | **−0.0113** (22/72) |
  | 白色毛衣 | 0.6078 | 0.6211 | **0.6231** | **+0.0133** (61/81) | +0.0020 (噪声内) |
  | 爱笑的眼睛 | 0.5814 | 0.5837 | — | +0.0023 (**噪声内**) | — |

  29→45 在两首歌上 3/3 seed 一致且远超 ±0.004 噪声底，这是本文件里第三大的单参数增益（仅次于 `--cfg` 和 kNN）。**注意 `--max-sec` 之前是 `sing.py:374` 硬编码的字符串 `"29"`，压根没接 `Cfg`** —— 现在是 `Cfg.max_sec` / `--max-sec`。
- **45s→50s 的方向由「歌曲需求是否超出素材移调上限」决定，可零 GPU 预判。** 绿色在 50s 退化 −0.0113，白色毛衣不退化，机制是纯粹的组合学：`w_ext.py:226` 的 `need_sec` 按缺口比例把预算摊到各频段格，预算从 20s 涨到 25s 时高音格的配额同比放大，一旦超过近距离素材的供给，贪心（`:272`）只能往下拿移调量更大的候选。判据是**时长加权 mean |shift|**：

  | 歌 | 超「素材最高段 med × 2^(8/12) = 538Hz」的帧占比 | 45→50 Δw\|sh\| | 实测 |
  |---|---|---|---|
  | 绿色 | 9.7% | **+0.61** (hi7 46%→61%) | **−0.0113** |
  | 白色毛衣 | 0.0% | **−0.15** | +0.0020 |

  按已记录的 0.014-0.020 SIM/半音，+0.61st 折算 −0.009…−0.012，与实测 −0.0113 吻合。**具体到段**：`seg 72/127/138` 在 45s 计划里拿 `+4st`，在 50s 计划里全被降级到 `+8st`。两个计划不是嵌套关系（45s 独有 6 个 `(idx,shift)` 组合在 50s 里消失），所以**长度扫点的相邻档之间混着「重规划抖动」，报 SIM 时必须同时报成分统计**（w\|shift\| / hi7 / uniq / reuse），否则无法回溯。

  但 **Δw|shift| 只能判断「继续加长会不会开始变差」，不能判断「加长有没有用」**：绿色 29→45 的 Δw|sh| 也是正的（+0.25）却大幅改善 +0.0229，因为那一段长度和多样性的收益压过了移调成本。
- **「甜点 = prompt_dur / anchor_min_ratio」这个公式已被否证。** 9/10 首歌的 `prompt.wav` 都撞在 `w_select --max-prompt 25s` 上限（24.82-25.00s），算出 `25/0.55 = 45.4s`，与绿色实测最优的 45s 一致 —— 但那是**巧合**，因为这 9 首的自变量没有变化，无法互相验证。唯一能验证的样本是**爱笑的眼睛**（`prompt.wav` 仅 18.55s → 公式预测甜点 33.7s）：实测 29/34/40/45s 四档全部落在 0.5814-0.5839，**跨度 0.0025，低于噪声底**，34s 处没有任何拐点。切换点本身确实存在（日志里锚点占比从 55% 掉到 46%/42%，确认 34s 以后被素材封顶在 18.55s），但**它不产生 SIM 上的拐点**。
- **`--anchor-min-ratio` 在大 `max_sec` 下是惰性参数（已实测确认）。** 锚点取 `min(prompt_dur, max_sec × ratio)`，所以一旦 `max_sec × ratio > prompt_dur`，ratio 就完全失效。白色毛衣 50s 那行日志写着 `锚点(干净) 24.92s (50%)` —— 名义 ratio 0.55、实际占比 50%，因为 `27.5 > 24.92`。**推论：在 45s 以上调 ratio 无效，只有调 `max_sec` 有效**（之前"降 ratio 到 0.45 值得扫"的建议因此撤回）。可迁移的量是 **ext 预算 = `max_sec − min(prompt_dur, max_sec×ratio)`**，绿色最优在 20s、白色毛衣 25s 仍可。
- **若歌曲 8st+ 桶占比 >25%，prompt 侧调参无收益，别花时间。** 爱笑的眼睛的桶分布是 `[11.7 10.2 27.6 20.3 30.2]`（错配 +6.0st，10 首里最大），30% 的帧距参考 8 半音以上、只有 11.7% 在 2 半音内，绝对分也是三首里最低（0.5814 vs 绿色 0.6223 / 白色毛衣 0.6078 @29s）。这种「素材与歌曲根本不匹配」的场景下 prompt 构造不是瓶颈，所以四档全平。**提高 `--ext-max-shift` 也救不了它**：模拟显示 ms 8→12 时 `hi8+` 恒定在 33%、`ext` 恒定 ~15s，只有 `w|sh|` 从 5.18 爬到 5.53（纯加成本不加覆盖）。根因在 `build_grids`——最高格上界是 `p99 × 2^(3/12)`，而 ms=8 时最上面 4 格（547-975Hz，占全曲 13.8%）**候选数为 0**，487-547Hz 格也只有 10 个候选。该做的是换更高音区的素材或 `--pitch-shift`，不是调 prompt 参数。
- **零 GPU 预筛工具（本轮建立，四次验证全中）：`/tmp/opencode/sim_plan.py` 复现 `w_ext.py` 的 `need_sec` 分配 + 贪心，只算成分不做移调。** 校验记录：绿色 45s 预测 `n=36 / ext=20.05 / w|sh|=5.71 / hi7=46% / uniq=30`，真实产物五项**全中**；白色毛衣 45s 三项全中；爱笑的眼睛 29s 与 45s 三项全中，34s/40s 误差 ≤0.06。**关键实现细节**：锚点 F0 必须按 `max_sec × ratio` 裁剪（对应 `w_ext.py:201-209`），否则 45s 档会算出 `w|sh|=6.12 / hi7=56%` 而不是真实的 5.71/46% —— 第一版就是这么错的，且它在 50s 档看起来是对的，很容易误判成"模拟器可用"。
- **`svc` 阶段没有显存可吃，`max_seg_sec` 已到架构顶。** 扫点（绿色，`--cfg 1.0 --seed 1234`）：20 → 78.8s / 5049 MiB，24 → 74.8s / 5257 MiB，28 → 73.7s / 6437 MiB。28 已是 whisper 30s 硬顶下的最大值，且 6437/16376 MiB 说明**瓶颈不是显存**。CFM 的 32 步 Euler 严格串行（`flow_matching.py:292`），CFG 每步两次前向（`:293` 的 `[prompt+xt]` 与 `:302` 的 `[xt]`）长度不同、不能 stack 成一个 batch。另外两处看似浪费的地方实测都不值得动：`infer_segment:513` 每段重算循环不变的 `pt_content_feat`（10 段冗余共 **0.11s** —— whisper-base 单次 28s 只要 **10.9ms**），`encode_long` 的 6 chunk 顺序编码（**0.07s**）。`eval` 的 71 窗 ERes2NetV2 逐窗推理也只有 1.38s。**这些是本文件里"看代码觉得浪费、实测微不足道"的典型，动手前先测。**
- **覆盖率与 SIM 相关 +0.69，但这大概率是难度的代理变量，不要当成因果。** 10 首歌 `corr(select 覆盖%, SIM) = +0.692`（Spearman +0.733），cov<55% 的 5 首均值 0.5654 vs cov≥55% 的 5 首 0.6021（差 +0.0367）。看着像"提高覆盖率就能提分"，但覆盖率低本身就意味着歌曲音域离参考远，而本文件已记录 `corr(B, A) = +0.810`（B = ref vs target，任务难度基线）。同一批数据里 `corr(唯一源数, SIM) = +0.440`、`corr(复用率, SIM) = −0.452`、`corr(均|shift|, SIM) = +0.102` —— 这些变量彼此高度耦合，n=10 的观察性相关无法分离。**多样性那条之所以可信，是因为随后做了干预实验（上一条），而不是因为这个相关系数。**
- **给 prompt「挑更好的段」这一整类思路都会撞上同一面墙：prompt 级指标由总时长主导，单段质量被平均掉。** 2026-08 用留一法直接测出了天花板：从选中的 28 段里逐一移除一段、重拼 prompt、测 SIM 变化，得到每段的*真实*边际贡献 —— 三首歌的贡献 **std 只有 0.00206 / 0.00156 / 0.00104**。即便存在一个完美排序键，换掉若干段的收益也只有千分之几，乘传导系数 k=0.51 后必然落在 ±0.004 噪声底之下。同一实验的 H1 分支说明了原因：移除 10 段「质量最差」使 SIM **+0.0014**（时长掉 7.5s 但质量升，净持平），移除 10 段「最短」使 SIM **−0.0036**。这解释了三个独立实验为何都失败 —— CAM++ 质心选段（本条）、select 阶段前移移调（下一条）、kNN 池加移调段（`:382`，增益来自池子变大而非音高覆盖）。**在动 select/ext 的选段逻辑之前，先用留一法量一下边际贡献的 std；若 <0.005，不必往下做。**

  附带一个可复用的测量事实：**段级 SV 分数在 <1s 音频上基本不可用，且 57% 的方差是时长伪影。** 同一段音频从 0.3s 截到 2.0s，CAM++ 分数抬升 **+0.475**，而整个 143 段候选池的分数跨度只有 0.90 —— 一半以上的"质量差异"其实是"片段更短"。`corr(段级分数, 时长)` = +0.711 (CAM++) / +0.705 (ERes)。同段 +0.05s 偏移的子窗 std 达 0.029-0.043。去时长残差的信噪比其实不差（6.95×），但它与真实边际贡献的相关性跨歌符号都不一致（−0.229 / +0.083 / +0.224，均值 +0.026）——**信噪比高 ≠ 预测力强**。另注意用 CAM++ 选段、CAM++ 打分会自我强化：端到端重拼后 CAM++ 3/3 报正（+0.003…+0.007）而独立的 ERes2NetV2 只有 1/3 正、均值 +0.0030。
- **前移移调到 select 阶段（在 prompt 构建时就覆盖高音区）：已实验，SIM 无改善，已回滚（2026-08）。** 思路是把歌曲目标 F0 分布带入 select 的贪心，把每个候选段虚拟展开成 shift ∈ {−4…+5} 的多个候选，用 `score = (边际增益/时长) × fidelity[shift]` 打分，让高音区缺口在第一次选段时就被填上，省掉 ext 阶段单独的 4s 预算限制。仿真（7首歌，3 seed）：覆盖率 75.8%→84.9%，band L1 0.405→0.273。实际 A/B（song_2 3 seed, song_6 3 seed, seed 配对）：

  | 歌曲 | 旧流程（seed 均值） | 新 select | 差值 |
  |---|---|---|---|
  | song_2（覆盖 62%→88%）| 0.5962（5 seed）| 0.6029（3 seed）| +0.0067 |
  | song_6（覆盖 29%→64%）| 0.5313（3 seed）| 0.5251（3 seed）| −0.0062 |

  两首合并均值差 +0.0002，seed 噪声底 ±0.004，**无显著效果**。根本原因：SoulX 从 `prompt.wav` 里用 WhisperEncoder 提取内容特征，speaker embedding 来自 mel 时序全貌而非 prompt 的音高分布；prompt 高音区覆盖率不在 `content_feat + f0_feat` 这条路径上。和 ext 改写时的结论完全一致——改的是 prompt 结构，不是送给模型的 F0 条件，所以覆盖率涨了但 SIM 不动。附一个在实验中发现的测量陷阱：**FIDELITY 表（`tools/pcnsf/score_segments.py` 的 `vs_real` 列）用的是「内容配对协议」**——对 seg_k 的移调版打分时，真实语音用的是 seg_k 对应时刻的同一段，不是全文件嵌入。这个协议比 `eval_sim` 宽松约 0.7（+5st：0.898 vs ~0.25），不能用来预测 `eval_sim` 分数，也不反映 SoulX prompt 读取时的实际音色保真度。

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

### SVC engine A/B #2 (2026-08): Vevo2 (Amphion, speech+singing joint tokenizer)

**Conclusion: 30s whisper ceiling makes it unusable for full-song SVC.** Vevo2 is a unified speech+singing
framework (7k hours of singing data in the tokenizer training), but its FM-only (style-preserved SVC) path
uses whisper-medium as the content encoder, and whisper hard-truncates to 30s (`whisper.pad_or_trim`,
N_SAMPLES=480000). Everything past 30s is **截断后用最后一帧重复填充** (`vevo2_utils.py:211-214`), so
on song_2 (195.7s) only **15.3%** of the content is real — 1500 whisper frames real, 8285 frames repeated.

Measured (smoke test, 20s source + 25s ref, both < 30s ceiling, 16GB VRAM):
- **VRAM peak: 6.60 GB allocated / 7.53 GB reserved** (comfortable on 16GB)
- **RTF 0.39** (20s audio in 7.9s, after 169s one-time whisper-medium download)
- **No manual segmentation logic** — `inference_fm` sends the whole file to whisper, unlike SoulX which
  auto-segments at silence with `build_vocal_segments`

Why SoulX's 30s limit doesn't hurt but Vevo2's does:
- **SoulX**: whisper only encodes *content*, timbre comes from a separate channel (`pt_mel`, the prompt's
  mel spectrogram, which is global and never segmented). Each segment gets the same `pt_mel` as timbre
  anchor → cross-segment音色一致. The 30s limit only affects content encoding, which is already split at
  silence boundaries.
- **Vevo2**: the `content_style_tokenizer` is a VQ-VAE trained on **whisper + chromagram joint features**
  (`extract_coco_codec`, `:689-699`), so it encodes a混合of content and style, not pure content. The
  timbre reference also gets whisper-encoded and contributes to `diffusion_input_codecs` (`:777`), so the
  30s截断 affects both content *and* timbre. Manually segmenting the source doesn't help: each segment's
  `contentstyle_codecs` would still carry that segment's source-singer style residual (the tokenizer was
  never trained to disentangle them fully), and there's no global timbre anchor like SoulX's `pt_mel`.

Setup (tools/vevo2/.venv, Python 3.10, torch 2.6+cu124, kept for reference but not wired into sing.py):
- Weights: 2.72 GB (FM-only, skipped AR model; `download_fm_only.py` filters by `allow_patterns`)
- Dependencies: 20+ packages resolved by repeated "missing module" loop (transformers 4.57, not 5.x;
  torchvision 0.21+cu124 to match torch 2.6; setuptools<81 for pyworld's pkg_resources; ruamel.yaml, six)
- CWD trap: same as seed-vc/yingmusic — must run from `tools/vevo2/Amphion/` or `whisper_stats.pt` path breaks

Ruled out as a production option. The 7k-hour singing tokenizer advantage cannot overcome the 30s whisper
architectural limit in a full-song zero-shot SVC use case. Changing the encoder (e.g. to WavLM) would
invalidate the pretrained tokenizer (needs retraining with 101k hours of data). Manual segmentation trades
音色一致性 for覆盖, and you'd still lose to SoulX which already has the segmentation + global timbre anchor.

### Score-control SVS (2026-08): MIDI + 歌词直接合成，不用原唱音频 —— 首跑成功，泄漏归零

我们一直只用了 SoulX-Singer 的一半。仓库里的 `cli/inference.py --control score` 走的是**另一个
checkpoint**（`model.pt` 2.82GB，SVS；`sing.py` 用的是 `model-svc.pt`），输入是**音符 + 歌词**，
`soulxsinger.py:121-125` 在 score 模式下把 `gt_f0 = None`，**原唱音频完全不进模型**。上游在
issue #5 里确认："the model takes notes + lyrics and directly generates a full singing
performance. No source vocal is required."

首跑（`singing/w_svs.py`，绿色的 24.9s 语音 prompt + 上游 `example/audio/music.json` 的人工校对
乐谱，`--control score --pitch-shift 0`，fp16）：**51.24s / 18.65s 推理 / 704M 参数**。

**结果的关键在于它把本文件反复记录的那条主线瓶颈直接绕掉了。** `eval_triad`（3s 窗，同 `eval_sim`
协议，4 位 impostor）：

| | B(难度基线) | A(out vs ref) | C(out vs target) | excess | progress | A−C win% |
|---|---|---|---|---|---|---|
| **score SVS** | 0.2715 | **0.6156** | 0.3941 | **−0.1959** | 0.472 | **100%** |
| SVC 历史最好（绿色） | 0.4046-0.4295 | 0.6034-0.6487 | 0.6090-0.6604 | −0.07…+0.14 | 0.22-0.36 | 25-80% |

三个数字要分开读：
- **`excess = −0.1959`，泄漏是负的** —— 输出比"随便另一个歌手"更不像原唱。SVC 路线上 6/9 首歌
  有真实泄漏（song_6 高达 +0.1414），根因是 whisper 内容特征里的原唱残留（也是 `--cfg` 必须从
  上游 3.0 降到 0.5 的原因）。score 模式不读原唱，这类泄漏在原理上不存在，实测确认。
- **`A−C win% = 100%`**（全部 16 个窗都更像参考）。SVC 路线上这个数字跨越 10%-80%，
  kNN 调到最好才刚过 50%，本文件当时把"crossing 50% 是 first for this workspace"单独记了一笔。
- **`progress = 0.472`，是本工作区最高**（SVC 最好 0.355，均值 0.279）。注意 B 只有 0.2715
  ——这首 demo 歌的原唱离我们的参考更远，所以 A=0.6156 的绝对值不能和 SVC 的 A 直接比，
  **progress 才是可比的量**，而它高出 33%。

**音高完全跟谱**：输出 F0 中位 329.0Hz = MIDI 64.0，乐谱中位 MIDI 64.0，误差 0.0。
**吐字可用**：`eval_cer` 0.2542（S=7 D=2 I=6，63 字），D 只有 2 说明没有吞字/内容塌陷 ——
对照 kNN α=0.4 那次是"3 首全负、CER +5.9 点"的量级，这里不是那个性质。

**步骤 1（先做的 go/no-go 门）：语音参考能产出可用的 note 标注。** score 模式对 **prompt 侧也要**
`note_pitch`（`soulxsinger.py:123`），而我们的素材是说话、说话没有音符 —— 这是整条路唯一可能
直接死掉的地方，所以先单独验（`singing/probe_svs_prompt.py`，不需要 SVS 权重，可在下载期间跑）。
绿色的 24.9s `prompt.wav`：

| 指标 | 实测 | 判据 |
|---|---|---|
| `|note − F0|` | **median 0.37 st**（mean 0.65，>1st 仅 14/97） | <0.5 为好 → 音符确实在描述这段音频 |
| note 时长 | median **0.180s**，<0.1s 仅 9/99 | 无碎片化 |
| pitch median | MIDI **61.0** | 与 F0 推出的 60.8 吻合 |
| notes/sec | 5.23 | 看着高于"唱歌 2-4"，但**与 97 个 ASR 字 1:1 对应**（中文语速 ~5 音节/秒），是字级对齐不是切碎 |

**`notes/sec` 那一行是本轮唯一差点误判的地方**：单看它会以为 ROSVOT 把语音语调切成了碎片，
实际对照字数就知道是正常的。判据要用 `|note − F0|` + 中位时长，不要用 notes/sec 单独下结论。

坑与实现要点：
- **`phoneme` 是音节级不是音素级**（`zh_tui1` 整个是一个 token，phone_set 2820 项里中文 1201、
  粤语 1529、英文 70）。所以歌词→phoneme 只要 `g2pM`/`ToJyutping`，**不需要 MFA 强制对齐**；
  `duration` 直接等于音符时长（`preprocess/utils.py:155`），时间信息全部来自 MIDI。
  这意味着可以纯手写 metadata.json，完全不碰音频。
- **prompt 只用 `prompt_meta_list[0]`**（`cli/inference.py:82`），后面的段全部丢弃。所以
  `--max-merge-ms` 才是决定"参考用了多少"的旋钮；`w_svs.py` 把首段单独写成 `prompt_first.json`
  让这件事在产物里显式可见。
- **传给推理的 prompt wav 必须是 metadata 所描述的那个文件**（`time` 字段是它的索引），
  即 pipeline 自己写出的 `long_cut_wavs/<index>.wav`，**不能传原始 `prompt.wav`**。
- **`preprocess/tools/__init__.py` 把 import 失败吞成 `= None`**，于是缺依赖的表现是
  `TypeError: 'NoneType' object is not callable`，而不是 ImportError。排查要逐个手动 import。
- **不要用"缺什么模块就 pip 装同名包"的循环**：`funasr` 缺的 `hydra` 对应的是 `hydra-core`，
  而 PyPI 上真有一个叫 `hydra` 的无关 C 扩展，会编译失败并让循环空转。正确做法是
  `importlib.metadata.requires('funasr')` 拿到真实依赖名单。
- SoulX venv 补装（**全部 `--no-deps`，保住 torch 2.6+cu124 / numpy 1.26**，实测补装后
  SVC 路线的 import 与 CUDA 均正常）：`funasr==1.3.0 matplotlib==3.10.8` + funasr 的核心依赖
  （jamo jieba pytorch-wpe oss2 umap-learn jaconv hydra-core tensorboardX modelscope
  sentencepiece kaldiio torch_complex editdistance pynndescent）+ matplotlib 的
  （pyparsing cycler python-dateutil kiwisolver fonttools contourpy six pillow）+ note 转录的
  （pretty_midi==0.2.11 mido webrtcvad ml_collections pyworld importlib_resources
  **setuptools==80.9.0** —— pyworld 用 `pkg_resources`，81+ 会失败）+ g2p 的
  （ToJyutping==3.2.0 g2p_en g2pM nltk）。
- 权重 `singing/download_svs.py`（固定 `.part` + `curl -C -` + LFS sha256 校验，幂等）。
  **故意不下 3.42GB**：karaoke 1.72GB + dereverb 0.91GB（只有 `vocal_sep=True` 用，我们自己的
  `sep`/`ref-sep` 已产干声，且 karaoke ckpt 与 GPT-SoVITS 里那份同一个文件）、
  parakeet 2.47GB（纯英文 ASR）。实下 13 个文件；注意 `rosvot/rmvpe/model.pt` 与顶层
  `rmvpe/rmvpe.pt` 是**两个不同文件**，前者由 `note_transcription/model.py:294` 按
  `<ckpt>/../../rmvpe/model.pt` 推导，漏了会报路径错。

#### 听感反馈「转音转得过快，短时间转了好多个音」—— 真因不是音高，是 slur 处重新起音

用户听感一次就指对了问题存在，但**指向的原因是错的，而且三个显而易见的音高假设全部被实测否掉**。
这一节值得完整保留，因为"照着听感描述去改音高"会完全走空。

先量谱子，它不碎：86 音符 / 51.0s，**sounding 1.51 音符/秒**，中位时长 0.38s，<0.2s 只有 5/77。
再量三路 F0（RMVPE，50Hz 栅格，与 `target_f0.npy` 同协议）：

| | 相邻帧跳变 mean(st/20ms) | >0.5st 帧 | 稳定音段占比 | 可分辨音高台阶 |
|---|---|---|---|---|
| 真实歌手 | 0.2574 | 14.85% | 31.4% | 62 |
| **我们的输出** | **0.1732** | **6.62%** | **59.4%** | **49** |
| 音符阶梯（谱面） | 0.0648 | 2.51% | 90.4% | 45 |

**我们的输出比真实歌手更平滑，不是更抖** —— 跳变只有真人的 67%，稳定音占比几乎翻倍，
可分辨台阶 49 个还少于真人的 62 个（台阶/谱面音符 = 0.64，真人 0.81）。滑音时长也更长
（median 50ms vs 真人 40ms）。slur 处最大单帧跳变更是只有真人的 **0.54×**。
所以"转太快/转了好多音"在音高维度上**全部指标反向**，四个角度一致，不是噪声。

真因在**能量/发音**维度。测 slur 边界（`note_type=3`，同一个字跨多个音符）处的能量凹陷
（0=完全连贯，1=完全断开，取边界前后 6 帧的 max 与中心 5 帧的 min 之比）：

| | mean dip | 硬断开 >0.5 |
|---|---|---|
| 真实歌手 | **0.196** | **0/13** |
| baseline（score 模式） | **0.434**（4 次 std 0.012） | **5/13** |

**同一个字被重新起音了** —— 辅音在元音中途重新发一次，耳朵把它听成一个新的、赶出来的音，
于是描述成"转音太快"。机制是确定的两条叠加：(1) metadata 在每个 slur 音符上**重复整个音节**
（`为/zh_wei4` 后面又是 `为/zh_wei2`，`群/zh_qun2` 后面又是 `群/zh_qun2`）；
(2) `DataProcessor.preprocess`（`data_processor.py:94-105`）给**每个音符**都套
`<BOW> … <EOW>` 词边界 token，slur 音符也不例外，于是模型在元音中间收到一个全新的词起始信号。

**"把 slur 的 phoneme 换成延音符号"这个最自然的修法是错的，两点各自独立地否掉它：**
phone_set 2820 项里**没有延音 token**（`<SP>` 是静音，特殊符号只有 PAD/SP/AP/UNK/BOW/EOW/BOS/
EOS/MASK/SEP + 10 个占位），而且**重复音节正是上游自己的惯例** —— zh/en/yue 三个 demo
metadata 全都这么写（`懦/懦`、`you're/you're`、`睬/睬`），即模型就是在这种标注上训练的，
换成别的写法是 OOD。

四臂实验（`singing/fix_slur.py` 只改 metadata，不碰音频不重训；`--seed` 配对）：

| 臂 | slur dip | 硬断 | A (out vs ref) | 说明 |
|---|---|---|---|---|
| baseline (score) | 0.407 | 5/13 | **0.5986** | |
| A: merge slur (score) | 0.514 | 7/13 | 0.5762 | **反而更差** |
| B2: melody + 真实 F0 | 0.403 | 5/13 | 0.5385 | 对照臂 |
| **B3: melody + 合成平滑曲线 + 颤音** | **0.244** | **2/13** | 0.4666 | 唯一修好发音的 |
| 真实歌手 | 0.196 | 0/13 | — | |

两个反直觉结论：
- **合并 slur 音符（A 臂）不但没用，还更糟**（dip 0.514）。所以"音符太多"从来不是原因 ——
  这也和上面音高侧的四个指标一致。
- **B2（喂真实 F0）也没修好**（0.403）。所以修好发音的不是"换成 melody 模式"，而是
  **连续 F0 条件本身要足够平滑**：合成曲线（raised-cosine 过渡，glide 70-90ms）的
  相邻帧跳变是 0.0649，真实 F0 是 0.2574，前者才把 dip 压到 0.24。

**代价很大，所以目前不作为默认**：B3 的 A 从 0.5986 掉到 0.4666（**−0.132**，是噪声底的 20 倍），
`progress` 0.449 → 0.268。方向在 2 个 seed 上一致（−0.132 / −0.139）。推测是曲线过于机械
（即使加了 45 cents 颤音，跳变 0.219 仍低于真人 0.257），音色通道被这条"非人类"的 F0 拖累。
未测的下一步：把合成曲线的随机微扰进一步做到真人统计量（drift + 音头微升 + 句末下滑），
或只在 slur 局部用 melody 条件、其余仍走 score。

**这一轮暴露了一个必须先修的测量缺陷：SVS 路线之前完全没有播种。**
`cli/inference.py` 从不调 `manual_seed`，CFM 初始噪声（`flow_matching.py:282`）走全局 RNG，
所以**同配置 4 次 md5 全不同**，A 的 std = **0.0065**（range 0.017），slur dip 的 std = 0.012。
本节最初的四臂对比是 4 次单抽，其中 A_merge 的 −0.039 就在 2σ 边缘、不可信；结论重跑成配对后才成立。
已打本地补丁（与 SVC 那份同构）：`soulxsinger.py:infer(seed=, rescale_cfg=)` →
`CFMDecoder.reverse_diffusion(seed=)`（这层早就有了）+ `cli/inference.py --seed/--rescale_cfg`，
segment *i* 用 `seed + i`。验证：同 seed 两次 **md5 完全一致**。`w_svs.py` 默认 `--seed 1234`。

#### 补上业界标准指标: FFE/GPE/VDE + MCD，并推翻上一节的"修好了"结论

用户听感复核 B3（melody+合成曲线）："**吐字变得更不清晰，音准也变得更差**"，要求回到 baseline。
这与上一节仅凭 slur dip 得出的"唯一修好发音的臂"直接冲突。补齐 SVS 领域标准指标后，
**用户是对的，上一节的结论是错的** —— dip 是个太窄的指标，它只看 slur 边界那 12 帧的能量，
既不测音准也不测吐字。

新增两个工具（业界通用，对齐 SoulX-Singer-Eval / TCSinger2 / DiTSinger 的口径）：
- `singing/eval_pitch.py` — **FFE / GPE / VDE**（pyworld DIO+Stonemask，10ms hop）
- `singing/eval_mcd.py` — **MCD**（pymcd，`dtw_sl` 模式，与 SoulX-Singer-Eval 同一个库）

四臂全指标（参考 = `music.json` 的 `f0` 字段，即真实歌手干声；seed 固定 1234）：

| 臂 | FFE(cents)↓ | GPE%↓ | VDE%↓ | CER↓ | slur dip↓ | A(SIM)↑ |
|---|---|---|---|---|---|---|
| **score baseline** | **61.6** | 35.9 | 14.6 | 0.407 | 0.407 | **0.599** |
| merge only (score) | 100.6 | 47.4 | 15.7 | **0.271** | 0.514 | 0.576 |
| **B3 melody+合成曲线** | 57.6 | 37.5 | 14.6 | 0.322 | **0.244** | 0.467 |
| melody + 真实 F0（对照） | **14.6** | **4.0** | **9.6** | 0.475 | 0.403 | 0.539 |

**为什么用户听到"音准更差"而 FFE 说 B3 更好（57.6 vs 61.6）**：这 4 cents 差异只有 6%，
而 **GPE 反而升高**（37.5% vs 35.9%）—— GPE 数的是 >50 cents 的**粗大错误**，也就是耳朵
真正听成"跑调"的那种错。FFE 是平均值，会被大量微小改善摊平；**听感对应的是 GPE 而非 FFE**。
B3 的合成曲线让平均偏差略降、但明显跑调的地方变多了。

**"吐字更不清晰"也被证实**：B3 的 CER 0.322，S（咬字变形）**从 8 升到 10**。虽然总 CER 比
baseline 低（因为 baseline 的 D=15 吞字更多），但**用户感知的"不清晰"是 S 而不是 D** ——
吞掉的字听不见，唱错的字才刺耳。

**决定：回退到 score baseline（B1）**，按用户要求。`w_svs.py` 默认就是 `--control score`，
不需要改代码；`fix_slur.py` 保留但不进默认流程。

**两个必须记住的测量陷阱（都在本轮踩到并修掉）：**
1. **`music.mp3` 是带伴奏的整首混音，不是干声。** 拿它当音高参考，pyworld 会把贝斯/伴奏
   当基频（中位仅 **111Hz**，而真实人声是 **326.7Hz**），FFE 直接虚高到 **1260 cents**（>1 个八度）
   且四臂排序全乱。正解是用 metadata 的 `f0` 字段（预处理时从分离人声提的），
   `eval_pitch.py` 因此支持 `--ref-meta`/`--ref-f0`，并在中位基频 <160Hz 时主动告警。
   **同理 MCD 在这里不可用**：14.7 vs 论文 SOTA 3-4，差距全部来自"混音 vs 干声"这个域差，
   不是质量差异。MCD 是成对指标，零样本换音色（ref 与 gen 说话人不同）本就不该解释。
2. **SoulX metadata 的 `time` 偏移必须补偿。** `music.json` 的 `time = [240, 51240]` ms，
   `cli/inference.py` 把内容放在 240ms 处，而参考 F0 从 0 开始。不补这 240ms，
   FFE 从 **62 虚高到 131 cents**（2 倍）。`eval_pitch.py` 默认做 ±600ms 整帧搜索，
   实测最优偏移落在 **240-260ms**，与 metadata 完全吻合 —— 这也反向验证了搜索是对的。
   验证链的另一端：喂真实 F0 那一臂 FFE = **14.6 cents / GPE 4.0%**，说明工具本身可信。

**与 SOTA 的差距（同口径对比，论文数据见下）：**

| 指标 | 我们 (score baseline) | SOTA | 差距 |
|---|---|---|---|
| SIM | 0.599（ERes2NetV2 协议） | 0.89-0.93（WavLM） | 协议不同不可直接比，但趋势上差 0.3 |
| FFE | 61.6 cents ≈ 0.36 (>50c 帧率 35.9%) | 0.15-0.24 | **约 1.5-2 倍差** |
| CER | 0.407 | 0.11（SoulX 论文 score 模式） | **约 3.7 倍差** |

注意 SoulX 论文的 FFE 定义是"偏差 >20% 的帧比例"，与我们的 GPE（>50 cents ≈ 3%）口径不同，
上表按可比项做了换算说明。**最大的短板是 CER 0.407 vs 论文 0.11** —— 但论文用的是人工精校
乐谱 + 干净单段，我们这轮也是人工乐谱，所以差距不能全归给转录；下一步应先查是不是
prompt 用了 25s 说话（论文 prompt 是唱歌）导致的域不匹配。

#### 验证「CER 高是因为 prompt 用了说话而非唱歌」—— 假设被推翻，但挖出两个更严重的度量缺陷

上一节推测 CER 0.407 vs 论文 0.11 的主因是"prompt 用 25s 说话、论文用唱歌"的域不匹配。
**做了 6 seed 配对实验，这个假设不成立**（p=0.415）。过程里发现的两个度量缺陷比原假设重要得多。

**缺陷 1：CER 的参考文本本身是错的。** 之前拿 `music.mp3`（带伴奏混音）的 ASR 当参考，
它把「容颜」听成「双颜」、漏掉「从此我开始」。而 `music.json` 的 `text` 字段就是**真值歌词**
（去掉 `<SP>/<AP>` 和 `note_type=3` 的重复字后 64 字）。用真值重测，baseline 的分解从
「S=2 D=19」变成一眼可辨的问题：**几乎没有咬字错误，全是"漏字"**。

**缺陷 2（关键）：`eval_cer` 对 51s 连续歌声整段转写会成块丢内容。**
最差那个 seed 整段测 CER **0.5625**（只转出 29 字），把同一个文件切成 10s 窗逐段转写，
CER 只有 **0.1094** —— **字其实全都唱出来了，0.45 的 CER 差异完全来自 ASR 侧**。
原因是歌声没有语音那样的停顿，VAD 找不到切点，就把整段当一句喂给声学模型。
逐窗转写可见内容是完整的：

```
sp6234 整段:  只是因为在人群中多看了你一一眼想你时你在眼前想你时你在脑海   (29字, CER 0.5625)
sp6234 逐窗:  0-10s 只是因为在人群中多看了你一眼再也没能忘掉
              10-20s 梦的梦想着偶然有一天      20-30s 再相见从此我开始孤单思念
              30-40s 我想你时你在天边          40-50s 你在眼前想你时你在脑海
```

**用已知上限标定窗参数**（原唱混音跑自己，CER 越低说明协议损失越小）：

| win/overlap | 0(整段) | 10/0 | 15/0 | 20/0 | 15/2 | **20/2** | 25/2 | 30/2 |
|---|---|---|---|---|---|---|---|---|
| CER | 0.125 | 0.328 | 0.156 | 0.141 | 0.203 | **0.078** | 0.094 | 0.125 |
| D | 6 | 8 | 5 | 3 | 6 | **0** | 0 | 4 |

`--win 20 --win-overlap 2` 成为新默认（**D=0**，比整段协议还低）。注意 `win=10` 反而最差
（0.328）—— 窗太短会切在字中间，所以"分窗"不是越碎越好，必须标定。

**修好协议后，噪声降到原来的 16%，原假设随即被推翻：**

| | CER 均值 | std | range | 唱歌更好的 seed |
|---|---|---|---|---|
| 语音 prompt (25s) | 0.2031 | **0.0221** | 0.172-0.234 | — |
| 唱歌 prompt (10.4s) | 0.1875 | 0.0328 | 0.156-0.234 | **3/6** |

配对差 −0.0156，**t=−0.888 p=0.415，不显著**。修复前语音臂的 std 是 **0.1407**（range 0.141-0.563，
同配置只换 seed 差 4 倍），那个方差几乎全是 ASR 协议噪声，不是模型行为。
**上一节"CER 0.407"这个数字本身也是协议产物**，真值约 **0.20**。

**但唱歌 prompt 确实有一个显著收益，在另一个指标上：**

| 指标 | 语音 prompt | 唱歌 prompt | Δ | p |
|---|---|---|---|---|
| **VDE%**（有声/无声判断错误率） | 14.88 | **6.95** | **−7.93** | **0.010** |
| FFE cents | 56.72 | 54.22 | −2.50 | 0.143 |
| GPE% | 35.04 | 35.93 | +0.89 | 0.475 |

**VDE 直接减半，3/3 seed 一致。** 说话有大量停顿和不发声段，唱歌是连续发声；用说话当 prompt
让模型对"哪里该发声"不确定。这是听感上"发虚/含混"的真实来源，而 **CER 完全测不到它**
（漏字和"该发声却没发声"是两件事）。音高准确度（FFE/GPE）则无显著变化 —— 与预期一致，
score 模式的音高来自 MIDI，不依赖 prompt。

**因此结论要分两半说**：想改 CER，换唱歌 prompt 没用（p=0.415）；但想改"发虚"，
换唱歌 prompt 有效（VDE 减半，p=0.010）。**两者不能混为一谈**，这也是上一节把
"用户说吐字变差"直接对应到 CER 时犯的同类错误。

**尚未做**：唱歌臂用的是上游 `zh_prompt.mp3`（另一个说话人，10.4s 单段），所以
SIM 不能与语音臂直接比（B 基线不同：0.5167 vs 0.2715）。要干净验证 VDE 收益能否用在
我们自己的音色上，需要拿**绿色这位参考人的唱歌干声**当 prompt 重跑 —— 素材是有的
（`data/songs/绿色/refsep2/lead.wav`），但那需要先给它做 note 标注。

**尚未验证的（这一轮只证明了"能跑且泄漏归零"，没证明"整首歌更好听"）：**
乐谱是上游人工校对过的，**没有测 MIDI 自动转录的误差如何传导**（上游 README 加粗警告转录误差
影响很大）；只跑了 51s 单段，没跑整首；没做与 SVC 的同歌同参考配对 A/B（要用
`singing/paired_eval.py`，`eval_sim` 的 RMS 门在两臂独立生效会导致窗口集不同）；
没听感验收演唱细节（score 模式丢掉原唱的颤音/气声/转音强弱，可能偏平淡）。

### Cascade pipeline (2026-08): SVS → SVC 级联，最优音色 + 最低泄漏

**级联流程已集成到 `sing.py`**。动机：SVS (score-control) 消除了 whisper 内容特征中的原唱残留，
SVC 二次转换时该残留变成**参考音色的真实特征**，不再是污染源。验证（绿色全曲 268.8s，seed=1234）：

| 方法 | A (音色) | progress | excess (泄漏) | A-C margin | 更像参考 |
|---|---|---|---|---|---|
| **SVC** (melody-control) | 0.6347 | 0.360 | 0.0569 | 0.0473 | 65% |
| **SVS** (score-control) | 0.5998 | 0.299 | 0.0435 | 0.0687 | 72% |
| **Cascade** (SVS→SVC) | **0.6525** | **0.391** | **0.0370** | **0.1408** | **75%** |

- **Cascade 最优**：音色 A=0.6525（比 SVC +0.0178，比 SVS +0.0527），progress 最高 0.391
- **泄漏最低**：excess=0.0370（SVC 的 65%，SVS 的 85%），A-C margin 翻倍（0.1408 vs SVC 0.0473）
- **机制验证**：SVS 输出已带参考音色，SVC 的 whisper 内容特征不再携带原唱残留

#### 使用方法

```bash
# 完整级联流程（opt-in 支线，必须显式列出 svs + cascade —— 默认流程不跑它们）
python3 sing.py --ref data/voices/ref.wav --song data/songs/song.flac \
  --name song_cascade --stages sep,ref-sep,f0-target,select,ext,svs,cascade \
  --seed 1234

# 只跑 SVC（快速预览，约 2 分钟）
python3 sing.py --ref data/voices/ref.wav --song data/songs/song.flac \
  --name song_svc --stages sep,ref-sep,f0-target,select,ext,svc \
  --seed 1234
```

#### 产物路径

- SVC：`data/songs/<name>/vocal.wav`
- SVS：`data/songs/<name>/svs/svs_vocal.wav`
- **Cascade**：`data/songs/<name>/vocal_cascade.wav`（这一节内的最优臂；注意 cascade 是
  opt-in 支线，当前默认流程不跑它，成片走 `vocal.wav`，见「参考素材加强」一节）

#### 实现细节

- **SVS 阶段** (`singing/svs_song.py`)：MIDI 转录用 ROSVOT，歌词从 LRC 缓存读取，生成 metadata 后调用 `cli/inference.py --control score`
- **Cascade 阶段**：提取 SVS 输出的 F0（`svs/svs_f0.npy`），复用同一套 `prompt_ext`，把 SVS 输出当作 SVC 的 source
- **kNN 保持开启**：虽然 SVS 输出已是参考音色，kNN 在级联里仍贡献 +0.0116 音色（绿色），因为它替换的是内容特征分布而非音色通道
- **所有路径转绝对路径**：SVC 以 `SOULX_DIR` 为 CWD，相对路径会找不到文件

### 参考素材加强（2026-08）：音域够宽之后，移调与 SVS 两层同时变成多余 — **当前默认**

补录参考素材后（`data/voices/ref_talk_sing.wav` 538s 说话+唱歌纯人声、
`ref_sing_only.wav` 372s 纯唱歌），本文件此前几乎所有"绕开素材音域不足"的机制都失去了前提。
默认已改为 **`--svc-prompt ext` + `--max-prompt 45` + `--anchor-min-ratio 1.0`**，
即"选 45s 干净素材直接当 prompt"，不移调、不跑 SVS。

**素材本身（浊音帧，RMVPE 50Hz 同口径）**：

| | 时长 | 浊音 | p10 | med | p90 | p99 |
|---|---|---|---|---|---|---|
| 旧 `ref.wav` | 163s | 106.6s | 198.7 | 245.8 | 306.1 | 349.8 |
| **`ref_talk_sing.wav`** | **538s** | **379.0s** | 219.3 | **328.5** | **447.2** | **631.0** |
| `ref_sing_only.wav` | 372s | 273.4s | 255.5 | 362.2 | 468.3 | 694.7 |

med 抬了 5.8 半音、p99 抬了 10.2 半音，而歌曲 med 是 291-374Hz —— 素材中位数直接落进歌曲音域。
12 首歌逐首算"不移调时 45s 预算内的需求满足率"（需求 = 歌曲各频段格时长占比 × 预算）：

| 素材 | 满足率 | 完全缺口 | 最差歌 |
|---|---|---|---|
| 旧 ref | 54.5% | 23.2% | 爱笑的眼睛 29.0% |
| **talk_sing** | **99.8%** | **0.0%** | 冬眠 99.6% |
| sing_only | 99.0% | 0.3% | 那女孩对我说 93.1% |

真实 `select`（budget 45s）复核，三首代表歌：覆盖 28.6/45.0/65.2% → **88.2/84.7/87.9%**，
候选池 143 → 429 段，**`suggested_shift` 从 +4…+6 掉到 −1…+1**。后者是最直接的判据 ——
它就是"还需不需要移调"本身。

**`talk_sing` 优于 `sing_only`，说话素材不是冗余**：它补的是低音区。sing_only 在 175-280Hz
只有 0.7-23.0s（talk_sing 是 7.4-46.4s），而歌曲低音段占比可以很高（那女孩对我说的
175-277Hz 占 49.6%），所以 sing_only 在那首掉到 93.1%。

#### 雨爱三臂 A/B（seed 1234 配对，旧素材臂是既有产物）

雨爱是旧素材最吃亏的歌之一（满足率 37.5%、建议移调 +6）：

| 指标 | A: 旧素材+移调+SVS | **C: 新素材·免移调·无SVS** | B: 新素材·免移调+SVS |
|---|---|---|---|
| SIM（新锚点，配对） | 0.7202 | **0.7600** `+0.0398` 55/69 p<1e-4 | 0.7624 |
| SIM（旧锚点，配对） | 0.4979 | **0.5300** `+0.0321` 56/69 p<1e-4 | — |
| eval_stab mean / worst | 0.6680 / 0.3884 | 0.7065 / 0.4030 | 0.7071 / 0.4082 |
| drift / dropouts | 0.0594 / 0 | 0.0641 / 0 | 0.0613 / 0 |
| CER（真歌词 371 字） | 0.3801 | **0.3639** | 0.4016 |
| 音准 >50cents | 4.08% | 5.13% | 4.78% |

- **素材本身的收益 +0.032…+0.040，是本工作区第二大的单一改动**（仅次于 `--cfg` 与
  song_3 低音区的 `--pitch-shift auto`），所有音高桶一致正向。这是"素材决定上限"那一条的
  正面确认，而不是又一次 prompt 结构调整 —— 后者在本文件里已被否证多次（覆盖率 +9pp 换
  +0.0002、预移调 +25pp 换 −0.0029）。
- **去掉 SVS 那层：SIM 不可区分（+0.0024，wins 37/69，p=0.63），但 CER 明显更好
  （0.3639 vs 0.4016，咬字变形 S=62 vs 76）。** 即 SVS 多一层重合成的代价落在**吐字**上，
  不在音色上 —— 只测 SIM 会得出"两者等价、留着无妨"的结论，而它其实是净损失（还白跑
  svs + select-svs 两个阶段）。这是本文件"动内容侧必须查 CER"那条规则的第二次生效。
- SVS 路线的历史数字（爱笑的眼睛 +0.0449，75/75 窗）**仍然成立，但只在素材音域不够时成立**。
  它存在的理由是"PC-NSF 受素材 med 上限约束、到不了 547Hz+"，而 talk_sing 的 p99 是 631Hz。
- 稳定性：mean +0.039、worst 改善、dropouts 三臂全 0，drift 仅 +0.005（本文件其他杠杆是
  +0.006…+0.018）。音准三臂持平（4.1-5.1%，都远好于 SVS 自身的 29.4%）。

#### 必须记住的两个方法论点

- **换素材做 A/B 时，锚点会静默变成上一个参考的缓存。** `Cfg.ref_clean` 原本是"`refsep2/lead.wav`
  存在就用它"，而复用同一 `--name` 时那个文件是旧参考的分离产物（雨爱目录里那份是 8 月 3 日的
  163s 旧 ref）。新增 **`--ref-is-clean`**：跳过 ref-sep，且 `ref_clean` **无条件**返回 `--ref`。
  没有这个开关，整轮对比会变成"新 prompt vs 旧锚点"，而日志里看不出任何异常。
- **不同锚点下的 SIM 绝对值不可比，必须双向检验。** 第一次跑出 C=0.7600 vs A=0.4979 像是
  +0.26 的暴涨，其实两臂锚点不同（新素材 med 350Hz vs 旧 ref med 277Hz）。统一锚点后是
  +0.0398；再用**旧锚点**复核仍是 +0.0321、56/69 窗、全桶正向。只跑对自己有利的那个锚点，
  这个结论是立不住的 —— 反向检验才排除了"新锚点偏袒新素材"。

#### 附带结论

- **`--anchor-min-ratio 1.0` 不是"禁用移调"，它是自适应的。** 锚点 =
  `min(prompt_dur, max_sec × ratio)`，余量才给移调段：素材够宽（select 选满 45s）→ 移调预算
  0.13s → 日志 `无可用移调段，prompt_ext = 原始 prompt`，两者 **md5 一致**，省掉过一遍 vocoder
  的 0.02-0.045 SIM 损失；素材偏窄（只选出 25s）→ 锚点 25s、移调预算 20s，ext 照常补高音格。
  所以它对窄素材没有回归风险，不需要按素材切换。
- **`corr(pitch distance, SIM)` 从 −0.42…−0.48 变成 +0.046。** 本文件开篇"pitch distance drives
  similarity，是最大的质量杠杆"在这批素材上失效了 —— 不是结论错，是瓶颈换了位置。后续再调
  音高相关的参数（`--pitch-shift`、ext 移调上限）应先重算这个相关系数，接近 0 就说明没有可捞的。
- **素材干净度已验证，不需要额外处理**：`<150Hz` 能量 **0.03%**（旧 ref 是 0.11%），无削波，
  底噪 −76.6/−106.3dB(rel) —— 没有伴奏或贝斯残留。音色一致性：talk_sing vs 旧 ref 整体嵌入
  **0.8490**（不同说话人地板 0.37-0.43），talk_sing vs sing_only 0.9686（同批录音）。
  `sing_only vs 旧 ref = 0.6915` 看着偏低，是**唱歌 vs 说话的域差不是音色差**：旧 ref 自己的
  ≥280Hz 窗 SIM 只有 0.7076 vs <280Hz 的 0.8891，同一人同一录音仅因音高就掉 0.18。
- 口径：**单首歌、单 seed**。SIM 的 +0.032…+0.040 是噪声底（±0.004）的 8-10 倍且双锚点一致，
  可信；CER 的 −0.038（S 差 14 字）只有一次测量，方向可信、幅度未定。跨歌复验未做。

### 选段：覆盖率不是目标函数，`--dbs-group 2` 是（2026-08）— **当前默认**

`select` 的贪心一直在最大化"音高格覆盖率"，而**这个目标与 SIM 无关，已被 9 次独立否证**。
决定性的一次是本轮：故意把覆盖率从 86.8% 砍到 **55.1%**（`W7`，完全放弃覆盖目标，改按
"源时间分散 + 段内音高跨度 + p90"选段），SIM 反而 **+0.0113**（48/69 窗，p=0.0016，全音高桶
一致改善）。四个覆盖率 55-80% 的臂 SIM 全落在 0.7686-0.7708（跨度 0.0022 < 噪声底），说明
**覆盖率一旦不再是约束，SIM 就由别的东西决定**。

真正有效的是 **Diverse Beam Search 的第 2 轮**（`--dbs-group 2`，默认）：第 g 轮的排序键减去
`lam × 该段被前 g 轮选过的次数`，惩罚累积，所以必须顺序跑 0..g 轮。跨 4 首歌 × 2-3 seed
（`--seed` 严格配对，`paired_eval` 公共窗）：

| 歌 | g0 基线 | g2 | 增益 | seed 数 | p_sign |
|---|---|---|---|---|---|
| 爱笑的眼睛 | 0.7473-0.7537 | 0.7858-0.7886 | **+0.0367** | 2 | 0.0002 / <0.0001 |
| 雨爱 | 0.7502-0.7595 | 0.7776-0.7798 | **+0.0231** | 3 | 全 <0.0001 |
| 绿色 | 0.7697-0.7725 | 0.7763-0.7814 | +0.0078 | 2 | 0.041 / 0.072 |
| 冬眠 | 0.7514-0.7642 | 0.7533-0.7591 | −0.0016 | 2 | 0.245 / 0.609（**无效应**） |
| **均值** | | | **+0.0165** | **7/8 arm 正** | 无显著负例 |

**这是本工作区唯一不需要权衡的 SIM 杠杆**，三项副指标全部同向或持平：

| | drift | dropouts | worst | CER（S=咬字变形） |
|---|---|---|---|---|
| 雨爱 | 0.0650→**0.0602** | 0→0 | 0.3618→**0.4250** | 0.1914→**0.1779**（S 50→45） |
| 绿色 | 0.0671→**0.0626** | 0→0 | 0.3690→0.3685 | 0.1308→**0.1207**（S 51→48） |
| 冬眠 | 0.0809→**0.0741** | 0→0 | 0.4076→**0.4304** | 0.1538→0.1731（S 35→39） |
| 爱笑的眼睛 | 0.0857→0.0884 | 0→0 | 0.4157→**0.4629** | 0.2011→**0.1708**（S 54→42） |

drift 3/4 改善（本文件其他 SIM 杠杆一律 +0.006…+0.018 恶化），dropouts 全程 0，worst 3/4 改善，
CER 3/4 改善（均值 −0.0087）。**推理成本为零** —— DBS 是确定性的，只是换一个选段方案。
增益与基线负相关（基线最差的爱笑的眼睛收益最大，冬眠/绿色基线最好收益最小），与 kNN 同一模式，
压缩跨歌差距而非放大。

#### 五个假设全部否证 —— g2 的制胜机制仍未知

11 个已测臂 + 16 个 prompt 特征的回归，**没有一个能解释 g2**（回归预测 0.7683，实测 0.7794，
残差 +0.011 全场最大）。逐个记录，因为每个都看起来很合理：

| 假设 | 判据 | 结果 |
|---|---|---|
| 覆盖率高更好 | `corr = +0.031`，加入 W 系列后转负 | 否（W7 55% 覆盖反而 +0.011） |
| 段更短更好 | `corr(med段长)` 从 −0.412 掉到 **−0.007** | 否，**假信号**：−0.412 全由 g2 单点拉出 |
| 宽跨度段多更好 | W6/W7 宽跨占比 97-100% | 否（只 +0.011，不及 g2 的 +0.023） |
| 取唱歌段（参考 200s 后） | `corr(唱歌占比) = −0.344` | 否，**反向**；g2 唱歌占比最低（74%）却最高分 |
| prompt 音高对齐歌曲 | `corr(\|Δp90\| 半音) = +0.086` | 否（W5 对齐到 0.04st 只有 +0.009） |
| 唱中区(200-340s)占比低 | 回归 R²=0.46，15 配对里 11 正 | 方向可信但**解释不了 g2**，LOO 误差 0.0048≈噪声底 |

三条方法论教训：
- **`corr(med段长, SIM) = −0.412` 是本文件第二次"小样本相关系数当规律"**（前一次是
  `corr(8st+桶占比, 最大增益)=+0.995` 被 LOO 否掉）。加 4 个新臂后归零。**n≈10 上的相关系数，
  先补样本再信。**
- **段长扫描（`--select-max-sec` 2.0/2.5/3.0）四臂全部不及 g2**（+0.0045…+0.0076 vs +0.0199），
  且缩短段长会**破坏** g2：池子从 174 涨到 245/296 后 DBS 的 λ 惩罚被稀释，g2 原本选中的
  长宽跨度段在细分后不复存在。`--select-max-sec 4.0` 保持不动。
- **单 seed 的臂间排序不可信**：g0 自己在 3 个 seed 上摆动 0.0093，比 W4-W7 彼此的差距
  （0.0022）还大。W 系列只有"≈+0.011"这个量级可信，内部排序是噪声。

#### 参数与踩过的坑

- **`--dbs-groups`（带 s）已删除，它从来没起作用。** 生成第 g 组只需顺序跑 0..g 轮
  （`w_select.py` 的 `for g in range(args.dbs_group + 1)` 命中即 break），`dbs_groups` 唯一
  被读的地方是一行日志。实测 `--dbs-groups 3` 与 `--dbs-groups 6` 输出 md5 完全一致。
  现在只有 `--dbs-group 2 --dbs-lam 0.1`。
- **`--gain-alpha` 默认同时改成 1.0，与 `--dbs-group 2` 绑定**：+0.0165 的实验全部在 α=1 上
  测得，单独改回 0 会脱离已验证配置。α=1 = 单位时长增益（`gain/dur`）。
- 回归判据（改动后已全部验证）：默认无参数运行 → 与 `雨爱_dbs_g2` **bit-identical**；
  `--dbs-group 0 --gain-alpha 1.0` → 与 `雨爱_a1` **bit-identical**；`sing.py --stages select`
  走 Cfg 默认 → 同样 bit-identical。**判据要用 `prompt.wav` 的 md5**（`prompt_f0.npy` 因
  `F0Extractor` 不可复现永不 byte-equal，见「ext 阶段 2.2× 提速」）。
- **`--dbs-group 2` 会降低覆盖率**（雨爱 86.8%→85.0%，绿色 88.9%→82.1%）。这是预期的，
  不要据此回调 —— 覆盖率与 SIM 的关系已 9 次否证。
- 探索工具留在 `/tmp/opencode/refana/`：`explore.py`（多目标加权贪心生成器，`W1` 与纯覆盖
  贪心产物一致可作自检）、`explore_render.py`、`ref_timeline.py`（测绘参考的说话/唱歌分界，
  本例 200s 前说话 med 234-278Hz、200s 后唱歌 med 321-426Hz）、`feat_regress.py`（16 特征回归）。

#### 仍未解决

- **g2 的机制未知**，所以无法推广成目标函数，也无法预判哪首歌会像冬眠一样无效应。
  下一步若要挖，需对 g2 独有的 15 段做逐段消融，但单 seed 噪声 0.009 会盖住单段效应，
  需要多 seed → 30+ 次 SVC。成本高、把握低。
- `--dbs-group` 是否该按歌选（3/4/5 组在雨爱上分别 +0.0099/+0.0110/+0.0054）**没有验证**，
  且本文件已两次因"按歌分档"被 LOO 否掉，不要在没有跨歌样本时做。
- 冬眠为什么无效应未知（它的 drift/worst 仍改善，只是 SIM 不动）。

### SVS prompt extraction (2026-08): `select-svs` 阶段，从 SVS 输出提取 prompt — **素材音域不足时的备选（曾是默认）**

**动机**：`prompt_ext` 受素材 med 上限约束（实测 248-339Hz），PC-NSF 移调上限 +8st 只能到 452-538Hz，而歌曲常有 547-774Hz 的段落。SVS (score-control) 输出覆盖全域（实测 222-624Hz，med 377Hz）且不含原唱音色（仅 MIDI+歌词+参考音色进模型），可当新的 prompt 素材源。

#### 实测结果（爱笑的眼睛全曲 257.9s，75 公共窗，seed 1234 配对）

| 臂 | prompt 来源 | S: vs svs_vocal | R: vs 真实参考 | 音准 >50c | CER |
|---|---|---|---|---|---|
| ext（旧默认） | prompt_ext（干净 55% + PC-NSF 45%） | 0.7432 | **0.5899** | 5.4% | 0.2793 |
| **svs（新默认）** | **prompt_svs（SVS 输出选段）** | **0.7881** | 0.5735 | **4.8%** | **0.2500** |
| (SVS 自身) | — | 0.8192 | 0.5734 | 37.9% | — |

**主指标 S（vs svs_vocal）：+0.0449（75/75 窗，p<1e-4）**，已达 SVS 自身 0.8192 的 **96%**。音准略好（5.4%→4.8%），稳定性同向改善（drift 0.0783→0.0692，worst 0.4286→0.4941），按真歌词 CER 0.2793→0.2500。

代价是 R（vs 真实参考）0.5899→0.5735（−0.0163），但那正好收敛到 SVS 自身的 0.5734 —— **绝对音色由 SVS 那一环负责**，这一环只需保住 SVS 给的音色。分桶看全音域一致正向，547-900Hz 桶收益最大（+0.036，该桶 5 窗）。

#### 分层混合变体（`--select-svs-mix`）：三个变体都测过，任何权重都不最优

思路是"音高落在干净素材覆盖到的频段就用它（保真度 0.93-0.96），只有到不了的高音格才用 SVS（0.60）"。测了三个 clean 源：

| clean 源 | 音域上限 | 干净占比 | S | R | S+R | CER |
|---|---|---|---|---|---|---|
| prompt_ext | 430Hz | 62% | 0.7585 | 0.5744 | 1.3328 | — |
| prompt | 320Hz | 41% | 0.7566 | 0.5630 | 1.3196 | 0.2473 |
| — (纯 svs) | — | 0% | **0.7881** | 0.5735 | **1.3616** | 0.2500 |

**结论**：扫了全部权重 `w·S + (1−w)·R`，mix 在任何 w 下都不最优（ext 胜出 w<0.27，svs 胜出 w≥0.27）。`prompt.wav` 虽然单段保真度最高（0.9550 vs ext 0.9259），但音域只到 410Hz，换掉 `prompt_ext` 意味着把 345Hz 以上几乎全让给 SVS（占比 41%→59%），净效应负（R 0.5744→0.5630）。

根因：**mix 表现由"高保真素材能覆盖多宽的音域"决定，不是"素材单段有多干净"**。这首歌 71% 的帧在 345Hz 以上，`prompt.wav` 在那里帮不上忙；`prompt_ext` 虽付 vocoder 代价（−0.029），但把高保真覆盖撑到 430Hz，反而更有用。这与 ext 一节已记录的"覆盖 > 单段保真"是同一规律。

保留 `--svc-prompt psvs_mix` 和 `--select-svs-mix`（含自动 `--clean-cap`），不设为默认。

#### 使用方法（**不再是默认**，需显式指定；svs + select-svs 阶段会自动带出）

```bash
# 素材音域不够时走这条（歌曲音域远超素材、suggested_shift 仍是 +4 以上）
python3 sing.py --ref ... --song ... --name exp1 --svc-prompt svs --seed 1234

# 当前默认（选段直接当 prompt，不跑 SVS）
python3 sing.py --ref ... --song ... --name exp1 --seed 1234

# 分层混合（测试用，不推荐）
python3 sing.py --ref ... --song ... --name exp1 --svc-prompt psvs_mix --select-svs-mix
```

#### 产物路径（三臂互不覆盖）

- svs 臂：`data/songs/<name>/vocal_psvs.wav`
- **ext 臂（当前默认）**：`data/songs/<name>/vocal.wav` ← **成片用这个**
- psvs_mix 臂：`data/songs/<name>/vocal_psvs_mix.wav`
- 中间产物：`svs/svs_vocal.wav`（音准未修，不是成品）、`prompt_svs.wav`、`prompt_svs_mix.wav`

#### 实现细节

- **SVS 阶段**（`singing/svs_song.py`）：MIDI 转录用 ROSVOT，歌词从 LRC 缓存读取（`get_lyrics.py`），生成 metadata 后调用 `cli/inference.py --control score`。**必须 `--svs-control score`**（已成默认）：melody 会把原唱 f0 喂进模型（`soulxsinger.py:126`），提出的 prompt 就带原唱音色，整条链路的立论就没了。
- **select-svs 阶段**（`singing/w_select_svs.py`）：从 `svs_vocal.wav` 按音高分布边际增益贪心选段（与 select 同一套算法），拼成 `prompt_svs.wav`。
- **svs_ref**（`--svs-ref ext`，默认）：SVS 吃哪个 prompt 去合成。ext = `prompt_ext`（分层裁剪到 20s，见「SVS 吃 prompt 的截断缺陷」），ref = `refsep2/lead.wav`（直接用分离人声，未测）。SVS 输出的音域由这里的 prompt 决定，所以默认 ext 能让 SVS 覆盖到 PC-NSF 移调后的上限。
- **`--svs-control` 默认 score**：不要改成 melody。见上"必须 score"。
- **音准修正在 SVC**：SVS 输出的 >50cents 帧 37.9%，SVC 用 `target_f0` 修回 4.8%。这是分工：SVS 负责音色覆盖全域，SVC 负责音准。
- **阶段顺序**：`svs` 和 `select-svs` 必须排在 `svc` 之前，否则 `--svc-prompt svs` 找不到 `prompt_svs.wav`。`ALL_STAGES` 已调整；两个阶段都在 `OPT_IN_STAGES` 里，由 `--svc-prompt svs|psvs_mix` 自动带出。
- **eval 阶段自动双锚点**：检测到 `svc_prompt in ("svs","psvs_mix")` 且 `svs_vocal.wav` 存在时，自动调用 `singing/eval_dual.py` 给出 S 列（vs svs_vocal，主指标）和 R 列（vs 真实参考）+ 音准。

#### 踩过的坑

- **SVS 吃 prompt 的截断缺陷**：`svs_song.py` 原来是一句 `y = y[: int(prompt_max_sec * sr)]`（取时间前缀），而 `w_ext.py:_render` 拼 `prompt_ext` 时**按移调量排序**（`for sh in sorted(by_shift)`），所以干净的 +0st 段全在头部、PC-NSF 移调的高音段全在尾部。取前 N 秒等于把音域扩展 100% 切掉。实测 45s prompt 被截成 **98.4% <380Hz / 1.4% 380-460Hz / 0.1% ≥460Hz**，与源素材 **62.5 / 28.6 / 8.8** 完全不符。修法：新增 `--prompt-trim stratified`（默认），按音高分层等比抽样，实测修到 **65.4 / 24.7 / 9.9**。不修这个 bug，`select-svs` 提出的 prompt 就是空的（SVS 输出音域塌到 <380Hz）。
- **干净素材必须用 `prompt.wav`，不是 `prompt_ext`**：后者 45% 是 PC-NSF 重合成的移调段，vs 真实参考 0.9259 而 `prompt.wav` 是 0.9550（差 0.029）。但实测换成 `prompt.wav` 后 mix 臂反而更差，因为它只有 18.55s 且音域只到 320Hz（med），丢掉了 `prompt_ext` 在 345-430Hz 的覆盖。这推翻了"更干净的素材必然更好"，真正的规律是**覆盖范围 > 单段保真度**。
- **`--clean-cap` 自动算法第一版会饿死干净素材**：按"预算 − SVS 独家缺口"算出 11.79s，比素材本身 18.55s 还小，白扔了 6.8s。已修：上限不低于素材总量（缺口预留只为防止吃光预算让 SVS 进不来）。
- **域效应打折扣**：三臂对"别歌 SVS"的 Δ 都大于"本歌 SVS"的 Δ（ext +0.0417 vs +0.0341，svs +0.0417 vs +0.0341，mix +0.0183 vs +0.0151），说明 S 列增益的主要成分是"像 SVS 这类音频"而非歌曲特异性。异人锚点几乎不动（−0.006 / −0.000），没引入别人音色。这一点三臂一致，不改变排序。

#### 口径

单首歌（爱笑的眼睛 257.9s）、单 seed（1234）、75 公共窗；547-900Hz 桶只有 5 窗。mix 三个变体都是同一首歌同一 seed 配对。跨歌复验未做（单首定不了默认值的惯例在这里被打破，因为主指标 +0.0449 / 75/75 窗 / p<1e-4 + 四个副指标全部同向或无差异）。

### SVC engine A/B #3 (2026-08): HQ-SVC (AAAI'26 low-resource zero-shot)

HQ-SVC = 解耦 codec + EVA 模块 + DDSP/diffusion 渐进生成，主打低资源零样本。Same 257s song, same ref; here we compared on the full song with the real `prompt_ext` as target (SoulX via `sing.py` full pipeline, HQ-SVC via `tools/HQ-SVC/cli_infer.py --shift 0`):

| engine | eval_sim vs prompt_ext | eval_stab vs real voice (1.5s win) |
|---|---|---|
| | mean / median | mean / median / drift / dropouts / worst |
| SoulX-Singer-SVC (baseline) | **0.5850 / 0.5945** | **0.4886 / 0.5154** / 0.0957 / 16 / 0.2346 |
| HQ-SVC | 0.5060 / 0.5084 | 0.4086 / 0.4125 / **0.0748** / 24 / **0.1869** |

- **HQ-SVC loses ~0.08 mean SIM to SoulX in both protocols** — worse timbre similarity — but its windows are *more internally consistent* (drift 0.0748 vs 0.0957, best worst-case). It sounds "stable but off-timbre"; SoulX sounds closer to the ref with more local jumps. For speech→singing cross-domain this confirms zero-shot SVC quality is still dominated by pretraining scale (SoulX: 42k hours; HQ-SVC: <80h). **Eliminated by ear (2026-08): the timbre gap is clearly audible on the full song; do not wire HQ-SVC into `sing.py`.** Keep the A/B files only as reference data.
- Its selling point is **low-resource training** (single consumer GPU, <80h data), not zero-shot inference quality — training code was still not released as of 2026-08, so it's inference-only for now.
- Setup gotchas (repo dir + weights removed 2026-08 to free 15G; re-download if ever needed): ships a ready-made conda env `environment.tar.gz` (4.7GB → unpack to `venv/`, torch 2.0.1+cu118, python3.9); weights on HF `shawnpi/HQ-SVC` (`utils/pretrain/*`: 250000_step_val_loss_0.50.pth 241M + ns3_facodec_* 450M + rmvpe 368M + nsf_hifigan 57M). `utils/pretrain/nsf_hifigan/` needs `config.json` copied from `configs/hifigan_config.json` (not shipped in place). `LD_LIBRARY_PATH` must be unset before running (`unset LD_LIBRARY_PATH`). Local CLI wrapper: `tools/HQ-SVC/cli_infer.py` (source + 1..N targets, `--shift 0` to keep melody; sys.path must use `append`, not `insert`, or `utils` shadows itself). RTF ≈ 0.05 on 16GB (10 diffusion steps).

### Reference material pitch extension (2026-08): TTS voice design vs neural vocoder transposition

Goal: extend a 163s speech reference (ref.wav, median F0 245.8Hz) to cover a song at 369.3Hz median — the gap is +5.8st. Two approaches tested on the same 15s material (ref_cut.wav, the highest-register 15s window of `data/songs/song_2/refsep2/lead.wav`, med 263.9Hz), scored by eval_sim vs the real voice unless noted.

#### VoxCPM2 TTS (tools/voxcpm2/, torch 2.6+cu124, Apache 2.0, 48kHz out)

| mode | variant | med F0 | SIM vs real |
|---|---|---|---|
| Style Control (ref + tag) | base (no tag) | 246.1Hz | 0.8456 |
| | high | 255.9Hz | — |
| | girl | 262.1Hz | — |
| | soprano | 263.0Hz | 0.866 |
| Voice Design (no ref) | vd_high | 357.6Hz | 0.535 |
| | vd_soprano | 352.3Hz | — |
| | vd_loli (中文) | 261.4Hz | — |
| Voice Design → Clone | rt_ultimate (ref+cont) | 366.3Hz | 0.909 (vs vd_high) |

- **Style Control ceiling is +1.15st, not enough** (need +5.8st). Tags only modulate *how* the speaker performs within their natural register; the ref channel locks the register itself. Proof: feeding a 263.9Hz window still output 246.1Hz — exactly the full-reference median (245.8Hz), i.e. the model extracted the speaker's *habitual* pitch, not the window's.
- **Voice Design (no ref) reaches 357.6Hz but sounds like "game character voice"** — SIM vs real voice is 0.535 (vs 0.866 for a real clone). Round-tripping the designed voice back as a ref ("VoiceDesign→Clone") keeps pitch (366Hz) and locks that synthetic timbre (SIM 0.909 vs the synthetic, 0.535 vs real) — it does NOT recover the real timbre. So TTS is unsuitable as SVC reference material: pitch is achievable, timbre is not.
- Gotchas: `voxcpm.generate()` has NO `seed` param (README example `seed=42` raises TypeError) — set `torch.manual_seed()` first. `_encode_wav` does NOT truncate the ref; 163s ref = 1020 tokens (max_length 8192) but training capped at 60s, so keep refs ≤ 60s. API: `--mode design` (no ref) vs `--mode clone` (ref + optional style tag). VoiceDesign output is random per run — must generate 3-5 and pick.

#### PC-NSF-HiFiGAN transposition (existing `tools/pcnsf`, winner)

Same ref_cut.wav, SIM vs real (unshifted):

| shift | med F0 | SIM | | shift | med F0 | SIM |
|---|---|---|---|---|---|---|
| −12st | 131.5Hz | 0.6939 | | orig | 263.9Hz | 1.0000 |
| −7st | 176.5Hz | 0.7844 | | +3st | 312.8Hz | 0.9346 |
| −5st | 197.8Hz | 0.8500 | | +4st | 330.6Hz | 0.9139 |
| −3st | 221.3Hz | 0.9163 | | +5st | 351.2Hz | 0.8789 |
| | | | | +6st | 371.9Hz | 0.8540 |
| | | | | +7st | 393.9Hz | 0.8303 |

- **Asymmetry: downshift costs more than upshift.** Equal magnitudes: +3st 0.9346 vs −3st 0.9163 (+0.018); +5st 0.8789 vs −5st 0.8500 (+0.029); +7st 0.8303 vs −7st 0.7844 (+0.046). Cost per semitone ≈ 0.024 up / 0.030 down. −12st (0.6939) is below DSP level (rubberband 0.7728) — unusable. **The per-semitone figures here are inflated** — they include the vocoder resynthesis loss that `orig = 1.0000` hid; see the per-segment sweep below for the corrected 0.014-0.020/st and for the fact that the asymmetry only matters above ±2st.
- **Mechanism: formants stay fixed.** Transposing multiplies F0 but keeps the spectral envelope; down to male register the female formants become impossible for a human, and ERes2NetV2 catches it. Consequence: for low-register songs, use `--pitch-shift auto` (raise the song, don't lower the ref) — this is the same direction the whole pipeline already takes.
- **Voiced rate drops on upshift** (0.78→0.75 at +7st), rises on downshift (→0.81 at −12st) — a real second cost, **but the "RMVPE tracks low pitches more easily" explanation this line used to carry is wrong**. `pitch_transpose.py`/`multi_transpose.py` run `rmvpe.get_pitch` on the **original** audio and only then apply `f0 *= 2**(n/12)`, so the estimator never sees shifted audio and the input-side voiced rate is *identical* for every shift by construction. Re-extracting F0 from the **vocoder output** reproduces the drop (8s segment, CPU: voiced 0.5196 @+0 → 0.4862 @+5 → 0.4848 @+7 → 0.4688 @+12, i.e. −0.035 at +7, matching the −0.03 above on different material), so the loss is created at **synthesis**: the mel is unchanged while F0 rises, and the vocoder must place wider-spaced harmonics under a spectral envelope measured for narrower spacing. Corollary: **a better/ensembled F0 estimator cannot recover it** — see OPTIMIZATION.md. Same run also shows output F0 is systematically **flat** vs target and worsens with shift (−9 cents @+0, −18 @+5, −27 @+12); the +0 offset is the vocoder's intrinsic bias.
- Usable pool: SIM ≥0.87 → −3/orig/+3/+4/+5 (221-351Hz); ≥0.83 → add +6/+7 (up to 394Hz, covers song median 369Hz). **These pilot numbers are inflated (no vocoder baseline); see the per-segment sweep below for corrected figures: mean ≥0.85 → −4…+6, worst ≥0.85 → −4…+4.**
- CLI: `tools/pcnsf/.venv/bin/python tools/pcnsf/scripts/pitch_transpose.py in.wav --transpose 5 --method nsfhifigan --config .../config.json --ckpt .../model.ckpt --pitchalgo rmvpe --rmvpeckpt SoulX-Singer/.../rmvpe.pt --out dir --title name`. Note `--transpose -3` needs `=` form (`--transpose=-3`) or argparse treats it as a flag.

#### Full-reference per-segment sweep −4…+6 (2026-08, supersedes the 15s pilot above)

The table above was one embedding per 15s clip scored against itself at `orig`, which conflated two
different costs and let good passages average over bad ones. Redone on the **whole 163.1s
`refsep2/lead.wav`** (the two-stage separation of `ref.wav`, med F0 245.8Hz, voiced 65.5%), split at
silence into **20 segments of 3.05-8.0s**, every segment scored individually at all 11 shifts.

Three comparisons per segment, and keeping them apart is what makes the numbers interpretable:

| comparison | isolates |
|---|---|
| shifted vs **its own shift=0** (`vs_base`) | transposition cost alone |
| shift=0 vs real (`base_vs_real`) | **vocoder resynthesis floor** — the ceiling for everything else |
| shifted vs real (`vs_real`) | total = resynthesis + transposition |

**The vocoder floor is 0.9748** (long segments; 0.9601 worst). No transposed output can beat it, and the
pilot's `orig = 1.0000` row hid it entirely — roughly a quarter of what the pilot charged to
"transposition" was really resynthesis.

Headline table, **long segments only (n=18)**, `vs_real`:

| shift | out F0 | mean | p10 | worst | verdict |
|---|---|---|---|---|---|
| −4 | 195.1 | 0.8811 | 0.8646 | 0.8543 | OK |
| −3 | 206.7 | 0.9123 | 0.8986 | 0.8838 | OK |
| −2 | 219.0 | 0.9427 | 0.9319 | 0.9266 | OK |
| −1 | 232.0 | 0.9610 | 0.9530 | 0.9273 | OK |
| **0** | 245.8 | **0.9748** | 0.9693 | 0.9601 | vocoder floor |
| +1 | 260.4 | 0.9690 | 0.9654 | 0.9143 | OK |
| +2 | 275.9 | 0.9586 | 0.9528 | 0.9048 | OK |
| +3 | 292.3 | 0.9422 | 0.9323 | 0.9095 | OK |
| +4 | 309.7 | 0.9208 | 0.9079 | 0.8738 | OK |
| +5 | 328.1 | 0.8980 | 0.8777 | **0.8463** | mean/p10 pass, worst fails |
| +6 | 347.6 | 0.8734 | 0.8513 | **0.8114** | mean/p10 pass, worst fails |

- **`−4 … +6` holds at the mean, and that is where the "0.85" claim is true**: −4 → 0.8811, +6 → 0.8734,
  and *every* shift in the range clears 0.85 on both mean and p10. So the working range from the pilot
  survives — but only as an average statement.
- **The tail breaks before the mean does, which is exactly what segmenting was for.** `worst ≥ 0.85`
  only survives **−4 … +4**; at +5 one segment hits 0.8463 and at +6 two fall to 0.8425/0.8114. Pick the
  bound by which statistic matters: `−4…+6` if the material gets pooled and averaged (the `ext` stage
  does), `−4…+4` if any single segment can carry a phrase on its own.
- **Degradation tracks shift *magnitude*, not absolute pitch — this overturns the natural reading of the
  pilot.** Bucketing all shifted segments by **output** F0 and correlating within each bucket, so
  absolute pitch is held fixed:

  | out F0 | ≈180 | ≈200 | ≈220 | ≈240 | ≈260 | ≈280 | ≈300 | ≈320 | ≈340 | ≈360 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | corr(\|shift\|, SIM) | −0.92 | −0.95 | −0.84 | −0.88 | −0.99 | −0.97 | −0.95 | −0.94 | −0.89 | −0.76 |

  Every bucket is strongly negative: reaching 280Hz via +2 beats reaching it via +6. Overall
  `corr(|shift|, vs_base)` is −0.93 up / −0.90 down, while `corr(output F0, vs_base)` is −0.65 up but
  **+0.76 down** — the sign flip proves absolute F0 is only a proxy. Practical consequence: **pick the
  source segment nearest the target pitch**, which is what `w_ext.py`'s "smallest shift wins" selector
  already does, and this is the measurement that justifies it.
- **Per-semitone cost is not constant — it accelerates.** Isolated with `vs_base`: 0.0140/st at +1 rising
  to 0.0195/st at +6 (cumulative loss 0.0131 → 0.1172). The pilot's flat "0.024 up / 0.030 down" was
  inflated by the missing vocoder baseline.
- **The up/down asymmetry is real but grows with magnitude and is negligible when small.** Paired by
  segment, `vs_base`:

  | magnitude | ±1 | ±2 | ±3 | ±4 |
  |---|---|---|---|---|
  | up loss | 0.0131 | 0.0275 | 0.0474 | 0.0694 |
  | down loss | 0.0131 | 0.0297 | 0.0549 | 0.0880 |
  | down/up | **1.01×** | 1.08× | 1.16× | **1.27×** |
  | segments where up is worse | 9/18 | 5/18 | 3/18 | **1/18** |

  At ±1 it is a coin flip (9/18, ratio 1.01) — below ±2 the direction does not matter. By ±4 downshift
  loses in 17/18 segments. Confirms the pilot's direction, corrects its magnitude, and reinforces
  "raise the song rather than lower the reference" (`--pitch-shift auto`) only for the *large* shifts.
- **Which segments fail is not predictable from any obvious feature.** For `+6`, correlation of segment
  SIM against its vocoder floor is only +0.15 (`vs_base`) / +0.34 (`vs_real`), against source F0
  +0.21…+0.40, against duration −0.31. The two casualties (`seg017` 258.9Hz, `seg009` 248.0Hz) have
  perfectly normal floors (0.9601, 0.9737) and mid-range pitch. So there is no cheap pre-filter —
  a segment's transposition headroom has to be measured, not inferred.
- **Segments under ~5s score unreliably**: the two short ones (3.05s, 3.88s) sit at a 0.9501 vocoder
  floor vs 0.9748 for the 18 long ones, i.e. ERes2NetV2 is noisier on short input and depresses them at
  *every* shift. They are excluded from the table above; keep scoring windows ≥5s.
- Cost: 4m24s for all 11 shifts of 163s, CPU-only. `batch_transpose.py` hoists the mel and the RMVPE
  pass out of the shift loop (only `f0 *= 2**(n/12)` changes per shift, `pitch_transpose.py:180-186`),
  so an 11-shift sweep costs barely more than one — the upstream script recomputes both every time.

```bash
# 1. sweep + silence segmentation (CPU, ~4.5min for 11 shifts)
tools/pcnsf/.venv/bin/python tools/pcnsf/batch_transpose.py \
    data/songs/song_2/refsep2/lead.wav --shifts -4 -3 -2 -1 0 1 2 3 4 5 6 \
    --config tools/pcnsf_hifigan/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json \
    --ckpt   tools/pcnsf_hifigan/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt \
    --rmvpe  SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt \
    --out tools/pcnsf/out/sweep_lead
# 2. per-segment F0 labels (86.13Hz grid = hop 512/44.1k, NOT eval_sim's 50Hz)
tools/pcnsf/.venv/bin/python tools/pcnsf/dump_f0.py data/songs/song_2/refsep2/lead.wav \
    --config .../config.json --rmvpe .../rmvpe.pt --out tools/pcnsf/out/sweep_lead/lead_f0.npy
# 3. three-way paired scoring (GPU, needs the SV model)
GPT-SoVITS/.venv/bin/python tools/pcnsf/score_segments.py \
    --sweep-dir tools/pcnsf/out/sweep_lead --real-dir data/songs/song_2/refsep2 \
    --shifts -4 -3 -2 -1 0 1 2 3 4 5 6 --f0 tools/pcnsf/out/sweep_lead/lead_f0.npy \
    --out tools/pcnsf/out/sweep_lead/scores.txt --csv tools/pcnsf/out/sweep_lead/scores_full.csv
```

#### WORLD baseline

Same material, `--method world` (pyworld, in pcnsf venv): +5st 353Hz/0.8289, +7st 394Hz/0.7655. PC-NSF beats WORLD by +0.050/+0.065 at the same pitch — the gap grows with shift size. Ranking so far: PC-NSF (0.879) > WORLD (0.829) > rubberband (0.773).

#### Four-dimension acoustic analysis (parselmouth/Praat, tools/voxcpm2/acoustic4.py)

Why does TTS sound like game voice? Measured on ref_cut (real), NSF +5st, WORLD +5st, and vd_high (TTS):

| dimension | real 263Hz | NSF+5st | WORLD+5st | TTS 358Hz |
|---|---|---|---|---|
| F1 / F2 (formants) | 396/1830 | 430/1809 | 428/1835 | **921/2081** |
| HNR (harm-noise ratio) | 23.23 | 26.08 | 25.20 | **15.72** |
| jitter % | 1.276 | 0.966 | 0.961 | 1.101 |
| shimmer % | 4.830 | 3.876 | 3.894 | 5.627 |
| intensity std (dB) | 14.86 | 15.33 | 14.51 | **10.70** |

- **TTS's F1=921Hz vs real 396Hz (+525Hz) is the smoking gun.** A female F1 of 921Hz is physically wrong (male /i/ territory) — the designed voice's vocal tract doesn't match its pitch. NSF shifts F1 by only +34Hz at +5st.
- **TTS is also noisier (HNR −7.5dB) and flatter in envelope (intensity std −4dB)** — the "game voice" is breathier and lacks dynamic range.
- **Jitter/shimmer are NOT the problem** — all three within normal range (jitter <2%, shimmer <6%). Do not chase perturbation metrics.
- PC-NSF preserves all four dimensions; its only artifact is F1 drift (+34Hz @ +5st, +76Hz @ +7st) — the same physical limit WORLD has.

#### Survey: what else exists for timbre-preserving pitch shift (2026-08)

- **SiFi-GAN** (ICASSP'23, open weights, 24kHz): source-filter HiFi-GAN, MOS beats WORLD at ±6st but LOSES to hn-uSFGAN at F0×2.0; 24kHz is a downgrade vs PC-NSF's 44.1k. Not worth it here.
- **NSF-Pupu-Vocoder** (2025-12): claims to beat PC-NSF at ±12st via bottleneck pitch injection; **no public weights yet** — watch list.
- **HiFi-Glot** (2026-02, open code+ckpt): differentiable all-pole filters, explicit F1-F4 control — the only approach that could fix the downshift asymmetry (move formants with F0). But it's a formant synthesis system, not a drop-in transposer; ~2 weeks to wire in. Only worth it if low-register songs become a frequent target.
- **DDSP/SawSing, FIRNet, CLPCNet**: singing-data vocoders (UV mask) / F0-robust but lower accuracy / <150c prosody-only. None beat PC-NSF for the ±5-6st, 44.1kHz, speech-to-singing material use case.
- **Conclusion: PC-NSF stays the default ext engine.** It wins on formant preservation (+34Hz F1 vs TTS's +525Hz), HNR (26 vs 15.7), envelope (15.3 vs 10.7), and outranks WORLD. The real lever now is *material coverage*, not the transposer: `+4/+5/+6` covers 330-372Hz, and the existing `select`/`ext` stages can pool it.

### Separation notes

- `tools/uvr5/uvr5_weights/` ships `.ckpt` files but **BS-RoFormer's `.yaml` config was missing**; fetched from `ZFTurbo/Music-Source-Separation-Training`. The karaoke model + config came from `Soul-AILab/SoulX-Singer-Preprocess`.
- **`batch_size` in those yaml files is a real performance trap.** The karaoke model ships `batch_size: 8` with `chunk_size: 485100` (11s/chunk) = 88s of audio resident. On 6GB that overflows into host-memory swapping: VRAM pegged, GPU utilization low, 15+ min per track. Setting `batch_size: 1` gave 2.2GB VRAM, 92-100% utilization, **27 seconds** — ~33× faster. Read from config at `bsroformer.py:117`. **But raising it above 1 buys nothing on a 16GB card — see the next entry; the "bump it to 4" advice this line used to carry was never throughput-tested and is withdrawn.**
- **`batch_size` is not a speed knob for any of the three separation models — swept and rejected (2026-08).** The 6GB-era note above only ever established that *too large* is catastrophic (paging); the natural inverse ("so on 16GB go bigger") turned out to be false. Swept on a 257s track with the page cache pre-warmed, 2 runs per point:

  | model | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 |
  |---|---|---|---|---|---|
  | **karaoke** time | 17.3s | 17.3s | 16.7s | 17.0s | — |
  | VRAM MiB | 2467 | 3707 | 6177 | 11121 | — |
  | **sep1 bs_roformer** time | 26.1s | 25.4s | 28.5s | 25.3s | 26.1s |
  | VRAM MiB | 1671 | 2625 | 4539 | 8369 | **16025** |
  | **stems4** time | 31.5s | 31.5s | 31.9s | — | — |
  | VRAM MiB | 1663 | 2797 | 4907 | — | — |

  **Time is flat while VRAM grows 5-10×**; sep1 at bs=16 reaches 16025/16376 MiB — 350MB from OOM — for zero gain. Cause: `nvidia-smi` sampling shows **GPU utilization is already 100% (median, 21/26 samples ≥90%) at `batch_size: 1`**. One chunk is 801-1101 STFT frames, so attention alone (`L²` = 0.64-1.21M) saturates the SMs; batching just lengthens the queue on an already-full GPU. Corollary for the whole file: **"we have spare VRAM" is not a reason to expect speed** — check utilization first, and only batch when the GPU is idling.

  Two facts to know before touching these yamls anyway. (1) **Changing `batch_size` changes the output**: karaoke bs=1 vs bs=4 gave different `lead.wav` md5 at SNR 72dB (`max|diff|` 1.5e-3) because fp16 accumulation order shifts. It stays bit-exact at a *fixed* batch_size (verified by re-running), so reproducibility is intact — but the sep cache keys on input md5 + **ckpt** md5 and *not* the yaml, so any yaml edit requires a manual `rm -rf data/.sep_cache`. (2) Current committed values are **karaoke 1, sep1 4, stems4 2** — an earlier version of this file claimed all three were still at the 6GB-era `1`, which was wrong.
- `tools/uvr5_cli.py` only writes the vocal stem, but the underlying `_path_audio_(input, others_root, vocal_root, ...)` writes both. Call it directly to keep the instrumental.
- Karaoke separation measurably isolates backing vocals (37/128 windows contaminated, up to backing/lead 0.66) but **only marginally improved output stability** — residual bleed remains. For harmony-heavy songs, YingMusic-SVC is the model actually trained for this (its benchmark is graded by contamination level).

## Measuring similarity

Do not judge by ear alone. `eval_sim.py` scores speaker similarity with the ERes2NetV2 model already in
`GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/`, bucketed by pitch distance from the reference.

**长参考的整文件嵌入前向必须清一次 PyTorch 缓存池，否则 16GB 卡随机 OOM。** 537s 素材的
`sv.emb(load16k(ref))` 实测 allocated 只有 0.21GB、**reserved 19.48GB**（CUDA caching
allocator 在长序列上预取整块显存），而后面每个 3s 窗只要 0.34GB。不清掉的话 eval 会在
随机的某一首歌挂掉——表现是 SVC 全部成功、只有 eval OOM，且换歌重试不稳定（前 4 首侥幸
通过、第 5 首失败）。已在 `eval_sim.py` / `eval_dual.py` / `eval_stab.py` / `eval_triad.py` /
`paired_eval.py` 的参考嵌入之后统一加 `torch.cuda.empty_cache()`（embedding 已算完，
清理不改数值）。新增 eval 类脚本时同样处理。

**Always score against the real, unmodified speech** as ground truth. Scoring a pitch-shifted variant against
its own shifted prompt hides the very degradation you are trying to measure — that mistake made a real
regression look like a tie.

**Every SIM number carries ±0.004 of seed noise, so any A/B under ~0.01 is unmeasured.** SVC inference was
non-deterministic until 2026-08: the CFM initial noise `z = torch.randn(...)` (`flow_matching.py:275`) used
the *global* RNG with no `manual_seed` anywhere in the repo (the one `manual_seed(0)` in
`whisper_encoder.py:70` sits in a `__main__` self-test that never runs). Every historical result was a single
random draw. Measured on two songs, 5 seeds each, everything else fixed:

| | song_2 (52 win) | song_9 (96 win) |
|---|---|---|
| whole-song mean, 5 seeds | 0.5890 – 0.6003 | 0.4573 – 0.4684 |
| seed std | **0.0043** | **0.0036** |
| mean per-window std | 0.0273 | 0.0396 |
| windows with max−min > 0.10 | 13 / 52 | 51 / 96 |

Consequences, in order of how much they should change behaviour:

1. **Fix `--seed` for every A/B from now on** (`sing.py --seed N`, threaded to `inference_svc.py --seed`,
   `infer(seed=)`, segment *i* uses `seed + i`). Same seed + same inputs is now bit-exact — verified by
   identical md5 on two runs. Leaving it unset keeps upstream's random behaviour.
2. **Re-read the history with a 0.01 noise floor.** `--cfg` (+0.024…+0.057) and `--pitch-shift auto` on
   song_3's low register (+0.1834) are far above it and stand. The non-octave shift result (−0.011) is ~2.5σ:
   direction probably right, magnitude not trustworthy. Everything described in this file as "prompt tuning
   moves ±0.01" — including the ext rewrite's +0.010 — is **at or below the noise floor and was never
   actually resolved**. Do not build on those numbers.
3. **To resolve a small effect, average over seeds.** 3 seeds cut the noise to ~0.0023, 5 to ~0.0018. One run
   per arm cannot see anything smaller than about 0.01.
4. Whole-song means are *far* more stable than individual windows (std 0.004 vs 0.027-0.040). "Run the song N
   times and keep the best" is therefore near-worthless; per-segment selection is where the headroom is (see
   the per-segment resampling entry above).

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

**Anything that touches the content features needs `singing/eval_cer.py` too — the three SIM tools are
blind to articulation.** `eval_sim`/`eval_stab`/`eval_triad` all score speaker identity, so a change that
edits `content_feat` (kNN replacement is the case in point) can post gains on every one of them while
making the lyrics unintelligible. That happened: α=0.4 kNN looked like an 11/11 cross-song win and was
in fact costing +5.9 CER points. `eval_cer.py` ASRs the output and scores it against the **original
lead's own ASR transcript**, so the number means "characters lost by the conversion", not absolute
recognisability. It reuses the already-cached `paraformer-large` + `fsmn-vad` (zero download; do not pass
`model="paraformer-zh"`, that alias points at SeaCo-Paraformer and pulls another 990MB).

```bash
GPT-SoVITS/.venv/bin/python singing/eval_cer.py \
    --src data/songs/<name>_base/sep2/lead.wav \
    --gen data/songs/<name>_base/vocal.wav data/songs/<name>_knn/vocal.wav --labels base knn
```

**Two measurement traps found while building it, both of which produced wrong numbers first:**
- **FunASR's VAD is non-deterministic.** Re-ASR'ing one unchanged file three times gave 345/281/345
  characters, and the same arm scored CER 0.3059 then 0.1912. `eval_cer.py` now pins
  `torch`/`numpy`/`random` seeds plus `cudnn.deterministic`; four consecutive runs are then identical.
  **A CER measured without that seeding is noise.**
  - **Pinning the seeds once at process start is NOT enough, and the fix above was incomplete until
    2026-08.** The VAD consumes the global RNG on every `generate()`, so file *k* in a `--gen` list sees
    an RNG advanced by the previous *k−1* calls: **a file's transcript depends on its position in the
    list.** Measured on one unchanged file, three consecutive calls: **58 / 55 / 54 chars** without
    re-seeding vs **58 / 58 / 58** when the seed is reset before each call. That was enough to move one
    arm's CER by 0.4+ purely by reordering. `eval_cer.py` now re-seeds inside `asr()`; verified by
    running the same pair in both orders and getting identical CERs. Any CER table produced before this
    fix is only valid for the arm that happened to be scored first.
  - **A CER spike can be VAD truncation rather than degraded audio — check the waveform before believing
    it.** 爱笑的眼睛's MILP arms reported CER 0.72/0.63 with D=68/46 (i.e. "most words swallowed"), which
    looked like catastrophic content collapse. But per-3s-window RMS was **near-identical across arms**
    (0.09/0.12/0.11… vs base 0.08/0.11/0.10…), so nothing collapsed. Bypassing the VAD with fixed 15s
    chunks took the transcript from 58 chars back to **109**. The real ΔCER was +0.026, not +0.42.
    Rule: if D is large, dump the ASR text *and* check window RMS; if the energy profile matches the
    baseline, re-run with fixed chunking before drawing any conclusion.
- **`eval_sim`'s RMS gate (`eval_sim.py:87`, `rms < 0.01`) is applied per arm independently, so two arms
  can be scored on different window sets.** On 冬眠 base reported 60 windows and knn 61 despite
  byte-identical inputs and equal-length outputs; the extra window was a low-scoring one present in only
  one arm, which shifted the gain from +0.0073 to +0.0042. Use `singing/paired_eval.py` for any A/B — it
  scores only windows that pass the gate in **both** arms and adds a sign test:

```bash
GPT-SoVITS/.venv/bin/python singing/paired_eval.py \
    --ref data/songs/<name>_base/refsep2/lead.wav \
    --f0 data/songs/<name>_base/target_f0.npy --ref-f0 data/songs/<name>_base/prompt_f0.npy \
    --a <base>/vocal.wav --b <variant>/vocal.wav --labels base knn
```

### Dual-anchor benchmark (2026-08): `singing/eval_dual.py`

**只适用于 `--svc-prompt svs|psvs_mix` 臂**（不再是默认，见「参考素材加强」一节）：那条路线的
目标函数变了 —— SVS 那一环负责绝对音色，SVC 只需**保住 SVS 给的音色 + 用原唱主唱修回音准**，
所以主指标不是"vs 真实参考"。默认 `ext` 臂只有单锚点，直接看 `eval_sim` / `paired_eval`：

| 列 | 配对 | 含义 |
|---|---|---|
| **S** | out vs `svs_vocal.wav` | **主指标** —— 音色保住了多少（天花板 = SVS 自身的自相似度） |
| R | out vs 真实参考 lead | 参考指标 —— 绝对音色，由 SVS 那一环负责，收敛到 SVS 自身即达标 |
| 音准 | out F0 vs `target_f0` | 中位 \|半音差\| + >50cents 帧率；SVC 应把 SVS 的音准问题修掉 |

```bash
GPT-SoVITS/.venv/bin/python singing/eval_dual.py \
    --gen <a>/vocal.wav <b>/vocal_psvs.wav --labels ext svs \
    --svs <dir>/svs/svs_vocal.wav --ref <dir>/refsep2/lead.wav \
    --target-f0 <dir>/target_f0.npy \
    --gen-f0 <a>/vocal_f0.npy <b>/vocal_psvs_f0.npy --svs-f0 <dir>/svs/svs_f0.npy
```

协议与 `eval_sim` 一致（3s 非重叠窗、窗 vs 整文件嵌入、RMS 门 0.01），所以 S/R 两列都可与本文件
历史 SIM 直接比较。三个要点：**公共窗口配对**（RMS 门在每臂独立生效会让两臂评在不同窗口集上，
见「eval_sim 的 RMS 门」）；**分桶用歌曲主唱 F0** 而不是各臂自己的 F0（桶标签必须臂间一致）；
**同时报 SVS 自身对两个锚点的分数**作为 S 的天花板与 R 的目标值。`eval` 阶段在 svs/psvs_mix 臂
下会自动调用它，结果一并写进 `eval.txt`。

**读这两列时不要只看 S。** 用户的判据是"两个 SIM 综合"时，扫权重 `w·S + (1−w)·R` 才是正确做法
—— 实测 mix 变体在任何 w 下都不最优（见「SVS prompt extraction」的分层混合一节），而只看 S 会
把 mix 误判成"svs 的削弱版"、只看 R 会把整条新流程误判成退化。

### Triad benchmark (2026-08): `singing/eval_triad.py`

`eval_sim`/`eval_stab` only measure **A = out vs ref**. That leaves two blind spots, and adding them changed
how the historical A numbers should be read. Same protocol as `eval_sim` (3s non-overlapping windows, RMS
gate 0.01, window vs whole-file embedding), so A here is directly comparable to every SIM in this file.

| | pair | what it is |
|---|---|---|
| **A** | out vs ref | 已有指标，获得的参考音色 |
| **B** | ref vs target | **不是成绩，是任务难度基线** —— 也是 cos 的"不同说话人地板"实测值 |
| **C** | out vs target | 原唱音色残留，**但裸值不可直接解读**，见下 |

```bash
GPT-SoVITS/.venv/bin/python singing/eval_triad.py --song-dir data/songs/song_2 \
    --impostor data/songs/song_{1,4,8,9}/sep2/lead.wav      # --impostor 不是可选项，见下
```

Measured on all 9 songs with output (single ref speaker, `refsep2/lead.wav` is byte-identical across songs —
verified by md5, so B varies only through the target):

| song | B | A | C | excess | progress |
|---|---|---|---|---|---|
| song_0 | 0.3986 | 0.5672 | 0.6564 | **+0.1241** | 0.280 |
| song_1 | 0.3607 | 0.5431 | 0.5576 | −0.0024 | 0.285 |
| song_2 | 0.4295 | 0.6013 | 0.6470 | +0.0906 | 0.301 |
| song_3 | 0.3058 | 0.5333 | 0.4499 | −0.0724 | 0.328 |
| song_4 | 0.4069 | 0.6177 | 0.5592 | +0.0217 | **0.355** |
| song_5 | 0.4015 | 0.5610 | 0.6159 | +0.0731 | 0.267 |
| song_6 | 0.4046 | 0.5355 | 0.6604 | **+0.1414** | **0.220** |
| song_8 | 0.3640 | 0.5214 | 0.5942 | +0.0615 | 0.247 |
| song_9 | 0.2934 | 0.4544 | 0.5129 | −0.0401 | 0.228 |
| mean | 0.3739 | 0.5483 | 0.5837 | +0.0442 | 0.279 |

- **`C > A` on 7/9 songs, and it does NOT mean the conversion failed.** That was the first read and it is
  wrong. Control: score every output against *other* songs' targets. Different singers' separated vocals
  score **0.7765** with each other (t2↔t6 0.8865, t2↔t5 0.8869) while ref↔target averages 0.4586 — i.e.
  "separated singing" is itself a huge domain cluster in ERes2NetV2 space, far larger than the speaker
  difference being measured. C is dominated by domain, not identity. **Always pass `--impostor` and read
  `excess = C − C_impostor`**; on the impostor-normalised number only **6/9** songs show real leakage, and
  song_3/song_9/song_1 show none (−0.07…0.00) despite raw C looking high. A/C are not on comparable scales
  and `A − C` is not interpretable.
- **B explains most of the cross-song spread in A, so raw A was never comparable across songs.**
  `corr(B, A) = +0.810`. song_9's famously bad 0.4544 comes with the lowest B in the set (0.2934): its
  singer is simply the furthest from the reference speaker, and its `progress` (0.228) is ordinary, not an
  outlier. Normalising as `progress = (A−B)/(1−B)` reorders the songs — song_3 goes from 8th on A to 3rd on
  progress. It does not shrink the variance (std 0.0449 → 0.0422), so it is a **re-attribution**, not a
  noise reduction: part of what looked like output quality was task difficulty all along.
- **The worst song is song_6, and only the triad shows why.** It has a mid-range B (0.4046, 4th highest) yet
  the lowest progress (0.220) *and* the highest leakage (excess +0.1414) — the one song where leakage
  exceeds the normalised gain (net −0.018). Its pitch buckets show margin degrading monotonically to
  −0.1872 at 8+ semitones. That is a specific, actionable diagnosis; A alone (0.5355, unremarkable) hid it.
- **`corr(A, excess) = +0.396`** — mildly *positive*. Higher-A outputs tend to carry *more* original-singer
  residue, not less, which fits the `--cfg` mechanism already documented above (the condition is
  `content_feat + f0_feat`, carrying whisper's speaker residual). Both A and C rise together when the model
  leans harder on the content condition. So a pure-A optimisation can be partly buying score with leakage;
  `excess` is what tells them apart.
- **Ceilings for calibration** (same protocol, window vs own global embedding): ref 0.8689, song_2 target
  0.8005, song_6 target 0.7733. A ≈ 0.60 against a 0.87 self-consistency ceiling on a task whose floor
  (B) is 0.43 — the usable dynamic range is much narrower than "0 to 1" suggests, another reason ±0.004
  seed noise matters as much as it does.
- Reuses the same `pretrained_eres2netv2w24s4ep4.ckpt` as `eval_sim`/`eval_stab`. Fine for A and B, but note
  the caveat recorded under GPT-SoVITS: anything that injects timbre via v2Pro's `sv_emb` shares this exact
  network, so those scores would be self-reinforcing.

## SingMOS 自然度评测与 NSF 后处理（2026-08）— `sing.py` 默认评测含 SingMOS

**评测侧新增第三个读数：SingMOS naturalness MOS**（唱歌域自然度，`singmos_pro` 预测器，wav2vec2
骨干，16kHz）。`eval` 阶段现在默认跑三样：`eval_sim`（音色相似）+ `eval_stab`/CER（按需）+ **SingMOS
naturalness**（`--no-eval-singmos` 关闭；`--eval-nisqa` 额外跑 NISQA，语音域模型，唱歌上不可靠，仅诊断）。
工具在 `tools/singmos/`（独立 uv venv：torch 2.2.1+cu121 + s3prl + librosa；权重缓存重定向到工具目录内
`.torch-cache/.s3prl-cache`，因为默认 `~/.cache` 在沙箱下不可写；**沙箱 /tmp 是每进程隔离的，跨进程临时
文件必须放工作区**）：

```bash
tools/singmos/.venv/bin/python tools/singmos/score_naturalness.py \
    --out <dir>/natmos_report.json --csv <dir>/natmos_windows.csv \
    原唱=<dir>/sep2/lead.wav 合成=<dir>/vocal.wav      # --no-nisqa 只跑 SingMOS
```

协议：统一 16k 单声道、3s 非重叠窗、跳过 RMS < 1%×p95 的静音窗；报 mean/median/p10/p90。**原唱 lead
是"真实演唱"锚点，参考（说话）是唱歌域模型的 OOD 输入，绝对分不与唱歌文件比。** SingMOS 是唱歌域模型，
sanity check 通过（两首歌原唱 > 合成，配对窗 70%/68% 原唱赢）；NISQA 主模型会把"干净的合成"排在
"带分离伪影的原唱"前面（它测干净度不测真实感），**不能当自然度排名**。

### 指标敏感性（探针：60s 片段受控扰动，SingMOS Δ / NISQA-TTS Δ）

| 扰动 | SingMOS | NISQA-TTS |
|---|---|---|
| 混响 30% 湿 | **−0.75** | −0.93 |
| 音高抖动 ±40cents/0.8s | −0.54 | **−1.28** |
| 强噪声 SNR25 | −0.56 | −0.13 |
| 轻噪声 SNR40 | −0.16 | +0.79（不可信） |
| 7kHz 限带 | +0.06 | −0.41 |

SingMOS 喜欢"亮、响、清音多"的窗（3s 窗特征相关：centroid +0.46、zcr +0.34、rms +0.30）。低分窗 =
弱响度+低清音占比 或 大滑音装饰音（f0std 17-30st）——**与原唱的低分窗高度重合**，是歌曲本身难点，
不是 SVC 独有缺陷。含义：**指标评的是干声自然度**；成品混音加了混响分数会掉，那是正常的。

### NSF 后处理（PC-NSF-HiFiGAN 纯重合成，`--post-nsf`，默认关）

对最终 vocal 过一遍 `pitch_transpose.py --transpose 0`（CPU ~2-3 分钟/首，输出升到 44.1kHz），
SingMOS 增益**跨两首歌稳健**，但 CER/NISQA-TTS 因歌而异 —— 所以是 opt-in 阶段（`--stages all` 不含它），
按歌人工决定：

| NSF Δ | 那女孩对我说 | 雨爱 |
|---|---|---|
| SingMOS mean | **+0.068** | **+0.111** |
| SingMOS p10（最差尾部） | +0.620 | +0.287 |
| SingMOS 配对胜率 | 59% | 68% |
| SIM mean（管道协议 ref=参考人声） | **−0.003** | 0.000 |
| CER | −0.002 | **+0.022** |
| NISQA-TTS | +0.635 | **−0.544** |

结论：① **SingMOS 主指标支持默认开启，但 CER 是反例门**（雨爱 +0.022 咬字变糊），建议 per-song
门控：`SingMOS 增益 ≥0.05 且 CER 恶化 ≤0.01` 才采用；② NISQA-TTS 是语音域模型，唱歌内容上跨歌翻向
（探针里加噪还 +0.79），**别拿它当唱歌开关门**；③ NSF 损伤是局部的（7/69、12/83 窗），且与泄漏修复窗
有交互（雨爱 w7 是修复过的泄漏窗，NSF 重磨后 SingMOS 掉 0.17）——下一步候选是**选择性 NSF**
（只重合成 SingMOS 低分窗）。注意 eval_sim 的 A/B 必须**同协议同参考**（管道默认 ref=`ref_clean`，
手动对比时用 `--ref-is-clean` 的同一文件），换参考会整体平移分数。

### 源音色归一化预处理（2026-08）— `sing.py --source-wav`

用户观察"原唱音色越接近目标，SVC 效果越好"→ 调研 + 三组 A/B（见
`docs/source-timbre-normalization-survey.md`）。注入点：`--source-wav <wav>` 替代 `sep2/lead.wav` 进
whisper 内容特征提取（`_svc_cmd` 的 `--target_wav_path`；**F0 仍用 `target_f0.npy`**，光谱编辑不改音高）。

| 方案 | SIM 效应 | 机制（whisper 逐帧 cos 实测） |
|---|---|---|
| LTAS/EQ 匹配（纯形状+限带） | 持平（±0.002） | cos=0.9996 —— whisper 对 EQ 免疫，源没变 |
| WORLD 共振峰搬移（保 F0） | 持平（±0.002） | cos=0.9999 |
| seed-vc 预转换到中性第三歌手 | 持平 | cos=0.988，改变太小 |
| **seed-vc 预转换到目标音色** | **+0.007/+0.011（2/2 歌超噪声底）** | cos=0.9235，16.5% 帧改变——源残留被换成目标残留 |

**seed-vc 预转换到目标是唯一有效的"源音色归一化"**：SIM +0.007~+0.011、SingMOS +0.05（两首歌一致），
与 kNN α=0.2 叠加生效；低基线歌收益更大（同 kNN 规律）。机制 = 音频域预热（预转换源自身 SIM
0.80/0.75 ≈ 基线 SVC 输出），SVC 在已向目标靠拢的源上表现更好。**代价是 CER 继承**：预转换自身
CER 0.139/0.274，最终输出 CER≈输入 CER——按歌门控（预转换源 CER 恶化 ≤0.01 且 SIM 增益 ≥0.005 才用）；
seed-vc 转换质量差的歌（去有风的地方）禁用。产物命名：seed-vc 把 `--output` 当**目录**用，生成
`vc_<src>_<tgt>_*.wav`，`--source-wav` 要指到那个文件（或先改名）。工具：`tools/source_norm/`
（ltas_match.py、formant_shift.py、whisper_diff.py、summarize.py；`tools/source_norm/.venv` 是
pyworld 独立 venv，setuptools 须 <81 否则 pyworld 缺 pkg_resources）。

**A 组进阶实验（2026-08，见 docs/source-timbre-normalization-survey.md §7.1）——最优配置变成「预转换(prompt_ext 参考) + kNN 关」**：

| 配置（全部 cfg=0.5, seed 1） | 白色毛衣 ΔSIM/ΔCER | 去有风 ΔSIM/ΔCER |
|---|---|---|
| 预转换 prompt_ext + kNN 关（e8a0on） | **+0.0082**/+0.0097 | **+0.0171**/**+0.0038** |
| 预转换 prompt_ext + kNN 0.2（e7） | +0.0071/**−0.0032** | +0.0134/+0.0301 |
| kNN 关、无预转换（e8a0） | +0.0020/**−0.0194** | +0.0045/**−0.0451** |
| 迭代自转换 e6p2/e6p3 | +0.006 | **+0.074/+0.217** ——否决 |
| 双遍 seed-vc e9 | — | +0.0147/+0.0264 ——否决 |

要点：① **预转换参考用 `prompt_ext.wav`（下游音色锚点）而不是说话参考**，去有风的地方的
CER 损伤从 +5.3 点降到 +0.4 点——说话域参考是 CER 元凶；② **cfg=0.5 下 kNN α=0.2 可能净负**
（kNN 关时 SIM +0.002/+0.0045 且 CER −0.019/−0.045，两首歌同时更优；历史 kNN 验证全在
cfg=1.0 时代，cfg=0.5 之后未重验）——需 ≥4 首歌的 cfg×α 2×2 才能动默认，若复现则
`cfg=0.5×α=0` + 预转换是新的推荐组合；③ E6 迭代自转换（SelfVC 式）在 CFM 管线上伪影
累积，CER 每轮爆炸，不要做；④ whisper_diff 门控：预转换后内容特征 cos<0.95 才值得启用
（有效臂 0.87-0.92，无效臂 0.988）。

**B 组（2026-08）—— YingMusic-SVC 预转换（`my_inference.py`，100 步，`--semi-tone-shift 0`，
参考 prompt_ext.wav）**：预转换源自身 CER 两首歌都优于 seed-vc（0.091 vs 0.139；0.259 vs
0.274），但**源级优势没传导到最终输出**——白色毛衣最终持平（+0.0083/+0.0065，SingMOS 4.250）；
去有风的地方 SIM 顶到 +0.0179、SingMOS 3.88/3.92（全实验最高）但 CER +0.087（最差）。
机制：SoulX 与预转换源的交互按转换器不同而不同（seed-vc 源 0.274→输出 0.222 被修复；
ym 源 0.259→输出 0.305 被加重，100 步扩散重合成的 whisper 特征更 OOD）。**门控结论**：
高基线歌用 YingMusic 预转换（CER/SingMOS 略优），低基线+保字歌用 seed-vc+α=0；
CER 的根不在预转换器，在"SoulX×预转换源交互"（训练侧）或 lead 质量（分离前端）。

## `sing.py` — one-command zero-shot singing CLI

Automates the whole pipeline above; run from the workspace root with **system python3** (stdlib only, no venv needed):

```bash
# 默认流程（2026-08 起：参考选段 → SVC，不跑 SVS，成片是 vocal.wav）
python3 sing.py --ref data/voices/ref_talk_sing.wav --ref-is-clean \
    --song data/songs/song.flac --name exp1
# 参考已是纯人声就务必加 --ref-is-clean：跳过 ref-sep，且锚点直接指向 --ref
# （不加会沿用 refsep2/lead.wav，换素材复用同一 --name 时那是上一个参考的缓存）
# 验收纯人声后，再混伴奏（可调伴奏音量）
python3 sing.py --ref ... --song ... --name exp1 --stages mix --acc-gain 0.7
# 只跑部分阶段（产物在 data/songs/<name>/，可续跑）
python3 sing.py --ref ... --song ... --name exp1 --stages sep,select
# 素材音域不够时才走 SVS 路线（svs + select-svs 自动带出，成片 vocal_psvs.wav）
python3 sing.py --ref ... --song ... --name exp1 --svc-prompt svs
```

Stages: `sep` (BS-RoFormer → karaoke, 人声+伴奏+主唱+和声) → `ref-sep` (参考音频同样两步分离取纯人声
lead，产出 `data/songs/<name>/refsep2/lead.wav`；select/eval 自动优先用分离产物，无则退回原 ref) →
`f0-target` (RMVPE 50Hz) → `select`
(参考切片打分贪心选段 → `prompt.wav`+`prompt_f0.npy`+`prompt_stats.json`。混合两级切分
[1s,4s]，**默认 `--dbs-group 2 --dbs-lam 0.1 --gain-alpha 1.0`** —— DBS 第 2 轮贪心，跨 4 首歌
+0.0165 SIM 且 drift/CER/worst 同向改善、零推理成本，见「选段：覆盖率不是目标函数」一节；
`--dbs-group 0 --gain-alpha 1.0` 回到纯贪心) → `ext` (选段 + 移调，
PC-NSF-HiFiGAN 默认 `--ext-method nsf`，seed-vc 可选。**默认 `--select-method greedy`**（两阶段
贪心：锚点 + 填频段缺口），此时 `--ext-div-weight 3.0` 的源段复用惩罚与 `--ext-exclude-anchor`
才有意义。MILP（`cov_anchor`）曾短暂做过默认后**已撤回**：clip 上 3 首 +0.0050，但全曲 10 首
均值仅 +0.0012（5 首显著正、5 首显著负，效应按歌分向）且 ext 耗时 19.7×，见同名条目。
`--ext-max-shift 8` 允许单段
最大移调 8st，拼出 ≤`--max-sec` 的 `prompt_ext`（**默认 45s，原 29s**；跨歌 A/B 绿色 +0.0229 /
白色毛衣 +0.0133，再加到 50s 是否退化取决于歌曲是否超出素材移调上限，见「prompt_ext 总长」
一节）。**`--anchor-min-ratio` 默认 1.0（原 0.55）**：它是 cap 而非固定值，锚点 =
`min(prompt_dur, max_sec×ratio)`，所以 1.0 是自适应的 —— 素材够宽时移调预算为 0
（`prompt_ext` 与 `prompt.wav` md5 一致，省掉 vocoder 的 0.02-0.045 SIM 损失），素材窄时
仍用余量补高音格。配套 `--max-prompt` 默认 45（原 25，那是"只选锚点、余量交给移调"时代的值）） →
`svs` (**opt-in**，只在 `--svc-prompt svs|psvs_mix` 时自动带出。score-control 合成，吃
`prompt_ext`；`--svs-ref ext` 默认，`--svs-control score` **必须保持 score**，melody 会把原唱
f0 喂进模型让后面提出的 prompt 带原唱音色) →
`select-svs` (同为 opt-in。从 `svs_vocal.wav` 选段拼 `prompt_svs.wav`；`--select-svs-mix` 走
干净素材优先的分层混合变体) → `svc`
(SoulX, `--svc-prompt ext` **默认**（用 `prompt_ext`，成片 `vocal.wav`）/ `svs`（用
`prompt_svs`，成片 `vocal_psvs.wav`，素材音域不够时用）/ `psvs_mix`；
`--pitch-shift 0` 默认保留旋律；`12` 全曲高八度适配男→女；`auto` 按段
自动对齐（男女对唱/跨音区）；`--max-seg-sec 28` 适配 16GB；`--cfg 0.5` 默认，上游 3.0 会放大原唱
音色残留；`--knn-alpha 0.2` 默认开启，检索池用完整参考人声（`--knn-full-pool`，绕开 whisper 30s）；
0.1-0.2 是 CER 与 SIM 双赢区，0.4 会让 CER 涨 5.9 点，详见「kNN 的吐字代价」一节) →
`eval` (eval_sim.py 对照真实语音；svs/psvs_mix 臂自动追加 `singing/eval_dual.py` 的双锚点评测，
主指标是 **SIM vs svs_vocal**) → `mix` (opt-in, `--mix`；两级动态控制默认都开：先按
`sep2/lead.wav` 逐帧对齐人声包络 `--no-match-lead` 关，再按伴奏包络给时变增益
`--no-match-acc` 关，软限幅 `--limit-db -0.5`，保留伴奏立体声 `--mix-mono` 关，见下)。
`cascade` 也是 opt-in，且**不再被 `--mix` 带出**（它依赖 `svs/svs_vocal.wav`，而默认路线不跑
svs，绑在一起会让 `--mix` 必然 Fail）；要跑级联就显式 `--stages ...,svs,cascade`，见
「Cascade pipeline」一节。Worker 脚本在 `singing/w_*.py`，每个阶段用对应项目的 venv 跑。

**阶段顺序有依赖**：`svs` / `select-svs` 必须在 `svc` 之前（`--svc-prompt svs` 要用
`select-svs` 的产物）。`ALL_STAGES` 已按此排列，手动传 `--stages` 时注意。

score-control SVS 的独立入口仍在：`singing/download_svs.py` 拉权重、
`singing/probe_svs_prompt.py` 验语音参考能否产 note 标注、`singing/w_svs.py` 端到端合成
（上游 demo 歌用）。首跑泄漏 `excess = −0.196`、`progress = 0.472`（本工作区最高），
详见「Score-control SVS」一节。

### Gotchas (all hit in practice)

- **固定增益混音必然在高潮段把人声压掉 —— `mix` 阶段现在默认按伴奏包络给人声时变增益。**
  发现路径是听感（「后半段伴奏压过人声」），量化后原因是两条独立事实叠加：**歌曲伴奏后半段
  比主歌响 15dB**（云烟成雨 acc 40-50s 段 −29.3dBFS → 150-160s 段 −13.4dBFS，编曲加厚，正常），
  而 **SVC 人声全程电平平稳**（−17…−20dBFS 几乎不动）。旧 `w_mix.py` 是一句
  `vocal_gain*v + acc_gain*a` 的固定配比，于是「人声−伴奏」从主歌的 +8…+12dB 翻转到高潮段的
  −3…−5dB。**原曲不这样并不是因为原唱唱得更响** —— 分离出来的 `sep2/lead.wav` 同样平稳
  （−17…−20dBFS），差别在原曲母带做过压缩，而那层压缩留在了 `acc.wav` 里、没留在 vocal 侧。
  所以这不是某首歌的调音问题，是流程缺一级动态控制，每首歌都会复发。

  修法（`singing/w_mix.py` 重写 + `sing.py` 透传）：50ms 分帧算人声与伴奏 RMS，按 `--match-win 5`
  秒窗求所需增益，Savitzky-Golay 平滑后插值到采样级。四首歌实测「人声被伴奏压过 3dB 以上」的
  10s 窗数：

  | | 云烟成雨 | 绿色 | 冬眠 | 白色毛衣 |
  |---|---|---|---|---|
  | 旧固定增益 | **6/20** | 2/23 | 2/20 | 2/19 |
  | 新 `--match-acc`（默认） | **0/20** | 0/23 | 0/20 | 0/19 |
  | 施加的增益（均值 / 峰值） | +2.4 / +7.7 dB | +0.9 / +4.3 | +1.3 / +5.2 | +1.2 / +7.5 |

  三个实现细节，每个都是踩过才知道的：
  - **必须只在人声有声的帧上比较两者 RMS**（`--match-gate -40`，相对人声自身峰值）。第一版手工
    脚本直接对整窗取 RMS，被 intro/间奏/句间换气拉低，算出的抬升量系统性虚高 ~0.9dB：手工版把
    高潮段推到人声比伴奏高 +2…+4dB（偏亮），门控后是 **−0.3…+2.3dB**。两版 mix 的 10s 窗 RMS
    只差 0.9-1.5dB，**听感差异全在这个门控上**，不是限幅或立体声的功劳。
  - **只抬不压**（增益下限 1.0）。伴奏安静的主歌段人声本来就该突出，压下去反而不像原曲。
  - **上限 `--match-max-boost 8`**，且抬升前必须有限幅兜底（下一条）。

- **`--match-acc` 只治了慢变的段落平衡，乐句内的电平下陷要在源头治 —— `mix` 现在默认先把
  SVC 人声的包络对齐 `sep2/lead.wav`（`--match-lead`，200ms）。** 上一条修完后仍有
  「某些窗口人声比原曲弱」的听感，量化确认：云烟成雨 188 个 3s 窗里 **24 个（12.8%）**
  mix 人声弱于原曲，最差 **−2.57 dB**（206-213s）。**根因不在混音，在 SVC 输出本身** ——
  三分量分解 `ΔV = SVC本身电平差 + 匹配增益补偿 + 限幅` 给出负窗均值
  `−0.85 = −3.45 + 2.82 − 0.15`：SVC 有 **59.0%** 的窗比原曲弱（最差 −5.76dB），而
  `--match-win 5` 的 5s 窗 + SG 平滑只追回 +2.82dB。**5s 窗是为跟随伴奏渐强设计的，
  追不上乐句内几秒内 4-5dB 的下陷** —— 这是两个不同时间尺度的问题，不能用一级解决。

  修法：混音前用 `lead_norm_curve()` 按 20ms 帧算 `lead/SVC` 的 RMS 比例，200ms 平滑后
  逐样本校正（`w_mix.py`，`sing.py --no-match-lead` 关）。**必须排在 `match_acc` 之前**，
  否则后者会去补偿一个已被修掉的下陷。实测（3s 窗 vs `sep1/vocal.wav`）：

  | | 更弱窗 | mean | std | p10 | 关键区段最差 |
  |---|---|---|---|---|---|
  | 旧（仅 match-acc） | **24/188 (12.8%)** | +1.38 | 1.21 | −0.31 | 16s −0.76 / 89s −0.92 / 154s −1.57 / 206s −2.57 |
  | **新（+ match-lead）** | **10/188 (5.3%)** | +0.88 | **0.84** | **+0.21** | **+0.33 / +0.84 / +0.40 / −0.17** |

  四个问题区段全部转正或接近 0，原先 24 个负窗只剩 2 个为负（最差 −0.17dB）。

  三个必须记住的点：
  - **它是在恢复原唱动态，不是压扁动态 —— 这是接受该改动的判据。** 一个把包络硬拉平的
    实现同样能让「更弱窗」归零，却会听起来死板。查动态范围 p90−p10：SVC 原始 22.5dB →
    校正后 **20.4dB**，而原唱 lead 是 **17.9dB**（校正后更靠近原唱）；与 lead 包络的相关
    从 **0.86 升到 0.94**。两个指标同向才说明方向对。
  - **平滑时长 200ms 是实测最优，不是随手取的。** 1s / 2s 跟不上乐句内下陷（3s 窗 std
    从 0.43 退化到 0.81 / 1.03，「更弱窗」从 3.2% 回到 6.9%）。
  - **必须双向限幅（`--lead-clip 12`）且只在两侧都有声的帧上算比例。** 第一版用
    `f_v.max()*1e-4` 当门限、不限幅，静音帧的比例炸到上限（gain 撞 5.0、峰值 +2.6dBFS），
    等于把 SVC 底噪连抖动一起放大；`worst` 还报出 **−197dB** 这种明显是伪值的数。
    静音段要用最近的有声比例插值填充，避免边界增益跳变。

  **验收时有个必踩的口径陷阱：改动后 `worst` 从 −2.57 反而「恶化」到 −4.95 dB，那是伪值。**
  最差的几个窗全在 187-189s，那里 `lead − vocal = −6.6 dB`，即 `sep1/vocal.wav` 里绝大部分
  是和声（实测 189-192s `backing −26.2dB` 比 `lead −27.8dB` 还响，是叠唱段）。SVC 只合成
  一条主唱，拿它跟「主唱+和声」比必然吃亏 —— **与本改动无关，是单声道 SVC 的覆盖局限**。
  以真正的对齐目标 `sep2/lead.wav` 为基准重算：`mean +0.64 / std 0.43 / worst −0.39 /
  更弱 3.2%`。所以**评判人声电平必须明确基准是 `lead` 还是 `vocal`**，和声段两者差 6dB 以上。

  回归判据：`--no-match-lead` 与改动前产物 **md5 完全一致**（本次已验），所以重构无副作用。
  另注意开启后限幅工作量变大（原始峰值 +3.09 → +4.76dBFS，最大衰减 3.59 → 5.26dB），
  但成品峰值仍是 −0.50dBFS、触顶 0 样本、L/R 相关 0.79（立体声未退化）。
- **抬人声之后 `np.clip` 会把增益换成失真 —— 已改前视软限幅。** 旧代码结尾是
  `np.clip(mix, -1.0, 1.0)`。人声抬升后云烟成雨的原始和峰值到 **+3.29dBFS**，硬削样本从 214 涨到
  **8570（0.08%）**，即「响了但脏了」。现在 `--limit-db -0.5`（传 0 退回硬削）：按需衰减 → 5ms
  前视滑动最小值 → 50ms 单极点释放，实测峰值 −0.50dBFS、**触顶 0 样本**，最大衰减 3.79dB、
  触发累计仅 0.3s。四首歌原始峰值 +0.44…+3.29dBFS，说明**只要开了 match-acc 就必然需要限幅**。

  验限幅有个坑：我一度量到成片解码峰值 1.4964 ≈ 限幅前的 1.4965，看着像限幅没生效。实际是提取
  音轨时加了 `-ac 1` 降混，两声道相加放大了峰值 —— **测量方法的伪影**。去掉降混后 v1（旧，硬削
  214 样本）解码峰值 1.0288 / 超 1.0 的样本 54 个，v2（新）1.0771 / **5 个**：新版更响但溢出反而
  少一个数量级。AAC 解码峰值超 1.0 本身正常（有码器 overshoot），要判削波看 **PCM 源**。
- **旧 mix 输出的是假立体声，伴奏空间信息被丢掉了。** `w_mix.py` 原来 `a.mean(axis=0)` 把伴奏压成
  单声道再 `np.stack([mix,mix])` 复制两份，成品 **L/R 相关精确等于 1.000000**。而 `acc.wav` 的
  L/R 相关是 **0.6474**（side/mid 0.4564），即有真实立体声内容。现在伴奏保留双声道、人声居中
  叠加，新成品 L/R 相关 0.7917。`--mix-mono` 回到单声道。
- **改 `w_mix.py` 后回归判据是「内容」而非 md5。** 以 `--no-match-acc --limit-db 0 --mono` 复现旧
  行为，逐样本最大差 **1 LSB、759/10618526 个样本（0.007%）**，来自新版内部用 float64 累加而旧版
  是 float32，差异只在 16bit 量化边界上 —— 不是行为变化。md5 必然不同（旧版恒写 2 个相同声道，
  `--mono` 写 1 个），别拿 md5 当判据。
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
- **`SoulX-Singer/` 里有本地补丁，升级上游会覆盖掉**：
  - `soulxsinger/models/soulxsinger_svc.py`：`infer(max_seg_sec=...)`（显存分段）+
    `build_vocal_segments` 活性优先合并（跳过尾奏/间奏无人声）+ `append_segment` 回溯跨短静音格
    （修 `num_overlaps` 失效）+ `num_overlaps` 透传 + `_nearest_octave_shift` /
    `pitch_shift="auto"` 按段八度对齐 + `infer/infer_segment` 的 `seed=` +
    `_knn_replace_content` / `knn_alpha`+`knn_k` 透传（内容特征 kNN 替换，见上文 kNN 一节；
    `valid_pt_frames` 必须传 whisper 实际产出帧数，否则右侧 zero-padding 会进匹配池）+
    `encode_long()` 分块编码（绕开 whisper 30s，给 kNN 检索池用）+ `infer(knn_pool_wav=)`
    在段循环外一次性编码检索池并经 `infer_segment(knn_pool=)` 透传
  - `soulxsinger/models/modules/flow_matching.py`：`reverse_diffusion(seed=)` —— 用
    `torch.Generator` 取初始噪声，上游走全局 RNG 不可复现
  - `soulxsinger/models/modules/decoder.py`：`CFMDecoder.reverse_diffusion(seed=)` 透传
    （这层最容易漏，漏了会 `TypeError: unexpected keyword argument 'seed'`）
  - `cli/inference_svc.py`：`--max_seg_sec` / `--num_overlaps` / `--seed` /
    `--knn_alpha` / `--knn_k` / `--knn_pool_wav`（只在 `knn_alpha>0` 时才 load_wav），以及
    `--pitch_shift` 的 `_pitch_shift_arg`（接受整数或 `"auto"`，上游是 `type=int`）
  - `tools/uvr5/uvr5_weights/*.yaml`（在 GPT-SoVITS 里）的 `inference.batch_size`
    （现值 karaoke 1 / sep1 4 / stems4 2）。**不要动** —— 已实测调大零收益，见 Separation notes

  拉新版后要重打。注意 2026-08 的 `select-svs` / 双锚点评测**没有**改上游：
  `singing/w_select_svs.py`、`singing/eval_dual.py`、`singing/svs_song.py` 的分层裁剪都在
  本仓库自己的文件里，升级 SoulX 不会丢。
- **`preflight` 的检查清单要按「代码读什么」写，不是按「下载器写什么」写。** SVS 那 6 个
  文件是 2026-08 补进去的（此前完全没检查，靠 `model.pt` 恰好在磁盘上才没炸）。两条判据：
  - **假阴性比漏检更糟** —— preflight 误报「缺失」会 raise，把一条本来能跑的流程直接挡死。
    所以只列运行时真正被读取的：`download_svs.py` 的 TARGETS 有 13 个文件，但下载下来的
    `SoulX-Singer/config.yaml` **没人读**（`svs_song.py:232` 传的是 repo 里的
    `soulxsinger/config/soulxsinger.yaml`，即 `SVC_CONFIG`），进了清单只是白增失败面。
    同理 FunASR / g2pM 那类按需下进 HF cache 的不做硬检查，首次跑必然误伤。
  - **贵的权重要按阶段条件检查。** SVS 侧 2.9GB，只在 `"svs" in cfg.stages` 时才要求，
    否则走默认 `--svc-prompt ext` 的人也被逼着下载。`select-svs`/`cascade` 只读 `svs/`
    下的产物、不加载模型，所以**不能**把它们也算进触发条件。`cfg.stages` 早就在 dataclass
    里，不需要改 `preflight` 签名。
  - 清单里 `rosvot/rmvpe/model.pt` 与顶层 `rmvpe/rmvpe.pt` 是**两个不同文件**（前者由
    `note_transcription/model.py` 按 `<rosvot_ckpt>/../../rmvpe/model.pt` 推导），两个都要查。
  - **验一个 preflight 检查是否真的生效，必须临时移走文件看它是否 raise。** 全 OK 的输出
    只能证明文件在，证明不了检查连着 —— 写完就跑一遍「mv 走 → 确认拦截并点名该文件 → 还原」。
- **本地补丁都在自己的 fork 里，可 diff 可回溯**（2026-08）：
  - `SoulX-Singer` → `CJYLZS/SoulX-Singer` 分支 `voice-patches`（基于上游 `81aeb3a`）
  - `tools/pcnsf` → `CJYLZS/pitch-transposition-using-DiffSinger` 分支 `voice-patches`
    （基于上游 `0e40e97`，rmvpe 裸 state_dict 补丁 + `save_wav` 就地修改 bug 修复，见下，
    以及新增 `scripts/multi_transpose.py` 多 job 入口，见「ext 阶段 2.2× 提速」）
  - 本地 remote 名为 `fork`。升级上游后重打：`git fetch fork && git rebase origin/<branch> voice-patches`
    （或 `git cherry-pick`），冲突按 AGENTS.md 的语义修。
  - GPT-SoVITS 的 uvr5 yaml 改动在 `.gitignore` 里 fork 也带不走，仍靠本文件记录。
  - **`tools/pcnsf/utils/infer_utils.py` 的 `save_wav` 有就地修改 bug（已修，2026-08）：** 原代码 `wav *= 32767` 在传入视图（`y_full[start:end]`）时会把 `y_full` 对应区域就地放大，导致后续写入 `full.wav` 的样本被二次放大进而 int16 溢出成垃圾。修法：改为 `wav = wav * 32767`（不就地修改）。受影响的只有 `batch_transpose.py` 写出的 `shift_*/full.wav`；各 `seg***.wav` 各自调用一次未被污染，FIDELITY 表数字不受影响。
- **参考素材决定上限**：`select` 只在现有素材里选段。素材太短（如 7s 可用）时 prompt 短、
  SIM 偏低，自动建议的变调数也偏高；要多补高音区，录更长的素材而不是靠变调。
  2026-08 补充：PC-NSF 移调可把素材池扩到 221-394Hz（SIM≥0.83），见
  「Reference material pitch extension」一节——`select`/`ext` 的素材池可以直接吃移调产物。
- `--max-prompt` 默认 25s（这是 `prompt.wav` 即锚点素材的上限），扩展后 ≤`--max-sec` **默认 45s**。
  「whisper 30s 硬顶」只约束 `pt_content_feat` 且已由 `encode_long()` 绕开；`pt_mel` 无此限制。
  真正的顶是 RoPE 4096（`llama.py:142`），45s+28s 段 = 3650 帧尚有余量，60s 会超。
  pt_f0 参与 SVC 的 f0 conditioning，不能只拼音频不拼 f0（`w_ext.py` 对扩展段重新提 RMVPE）。

---

# EchoMimicV3 — 单图 + 音频 → 说话/唱歌视频（`echomimic3/`）

蚂蚁集团 AAAI'26，Apache-2.0，1.3B 参数。输入一张人像 + 一段音频，输出口型同步、带自然
头部/身体动作的视频。**说话和唱歌都支持**（官方 gallery 有 Singing 分类，训练集含唱歌数据），
所以既能吃 `GPT-SoVITS` 的 TTS 输出，也能吃 `sing.py` 的 SVC 干声。

选它而不选 LongCat-Video-Avatar-1.5 / Wan2.2-S2V-14B 的理由：那两个是 14B 级、要 FP8 量化
+ block-swap 才勉强进 16GB，而 EchoMimicV3-flash-pro 官方标称 **12GB VRAM、768×768、8 步**，
在这张卡上有余量，迭代快，适合先把「素材 → 视频」这条链路的问题（人脸角度、音频格式）暴露完。

## 两条推理路径，只用 flash-pro

| | preview | **flash-pro** |
|---|---|---|
| 步数 | 15-25 | **8** |
| face mask | **需要** | 不需要 |
| 音频编码器 | `wav2vec2-base-960h` | `chinese-wav2vec2-base` |
| 权重 | `transformer/` | `echomimicv3-flash-pro/` |
| 入口 | `infer_preview.py` / `app_mm.py` | `infer_flash.py` / `run_flash.sh` |

flash-pro 不需要 mask，因此 **`tensorflow==2.15.0` + `retina-face` 这两个依赖可以整套跳过**
（只有 `src/face_detect.py` 用，preview 路径才需要），省约 3GB。`mmgp` 也跳过（只有
`app_mm.py` 这个量化 Gradio UI 用）。反过来，上游 `requirements.txt` **漏了 `pyloudnorm`**
（`infer_flash.py:36` 直接 import），不补会在启动时报错。这套增删记在
`echomimic3/requirements_slim.txt` 里，装它而不是上游的 `requirements.txt`。

## 权重 23GB，`flash/` 目录结构

`download_models.py` 会摆成 `run_flash.sh` 期望的样子（该脚本用的是 `flash/` 这个名字）：

```
flash/
├── Wan2.1-Fun-V1.1-1.3B-InP/   base，含 4 个大文件
│   ├── models_t5_umt5-xxl-enc-bf16.pth                       11.36 GB  ← 内存杀手
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth 4.77 GB
│   ├── diffusion_pytorch_model.safetensors                    3.13 GB
│   └── Wan2.1_VAE.pth                                         0.51 GB
├── chinese-wav2vec2-base/       0.38 GB（只要 pytorch_model.bin + config）
└── transformer/                 3.73 GB（flash-pro DiT）
```

- **CLIP 那 4.77GB 不能省**，虽然 flash-pro 不需要 face mask。`infer_flash.py:270` 无条件
  加载它并在 `:294` 传进 pipeline。我一度以为它只服务 preview 路径，是错的。
- **`chinese-wav2vec2-base-fairseq-ckpt.pt`（~1.2GB）要跳过**：`infer_flash.py:210` 走
  `Wav2Vec2Model.from_pretrained()`，只读 `pytorch_model.bin` + `config.json`，fairseq
  原始 ckpt 是纯浪费。`download_models.py` 已按文件白名单下载，不用 `snapshot_download`。
- 下载务必用 `download_models.py`（固定 `.part` 名 + `curl -C -` + sha256 校验），
  **不要用 `snapshot_download`** —— 原因见开头「Downloading large weights」一节，
  这次为此白下了 ~10GB。

## 本地补丁（升级上游会被覆盖，拉新版后重打）

- `infer_flash.py`：**让 `--GPU_memory_mode` 真正生效**。上游在 `:91` 定义、`:193` 读取这个
  参数，然后无条件 `pipeline.to(device)`，**两种 offload 模式从来没被调用过**（默认值
  `sequential_cpu_offload` 也是死的）。改成按模式分别调
  `enable_sequential_cpu_offload()` / `enable_model_cpu_offload()`；并且删掉紧跟其后的
  **第二次** `pipeline.to(device)`（原 `:332`）—— 那一句会把已 offload 的模块重新拉回 GPU，
  把刚装好的 hook 全部作废。
- `src/wan_text_encoder.py` + `src/wan_transformer3d_audio_2512.py`：
  `load_model_dict_into_meta` 的 import 加 fallback。上游只从
  `diffusers.models.modeling_utils` 导入，但 diffusers **0.39 已把它移到
  `diffusers.models.model_loading_utils`**，导致 ImportError → 整个 meta-device 分支被跳过
  → 静默退回 `low_cpu_mem_usage=False` → 峰值内存翻倍。日志里那句
  `The low_cpu_mem_usage mode is not work because cannot import name
  'load_model_dict_into_meta'` 就是它，**不是无害警告**（我第一次读代码时误判成无害，因为
  `:1506` 的默认值本来就是 `False`，且 `:1558` 有注释提到 0.33 重构；但 T5 那条路径
  `:264` 显式传了 `low_cpu_mem_usage=True`，所以确实被这个 ImportError 打掉了）。
  补丁生效的判据：该警告从日志里消失。
- `infer_flash.py:234`：**`load_state_dict(..., assign=True)`，本轮最危险的一个 bug**。
  开启 `low_cpu_mem_usage=True`（`:220`）后 base checkpoint 走 meta 设备加载，而 base 里
  **没有** flash-pro 那 220 个音频注入参数（`audio_injection.*` +
  `blocks.*.cross_attn.*_audio.*`），它们因此留在 meta 上。紧接着上游用不带 `assign=` 的
  `load_state_dict` 加载 flash 权重，而该调用是**就地拷贝**语义 —— 对 meta 张量是
  PyTorch 明确定义的 **no-op**。实测：

  | | 剩余 meta 参数 | load_state_dict 报告 |
  |---|---|---|
  | 上游（无 assign） | **220** | `missing=0, unexpected=0` |
  | `assign=True` | **0** | `missing=0, unexpected=0` |

  两者的返回值完全一样，**失败没有任何信号**。之所以能发现，是因为 offload hook 随后调
  `.to()` 撞上 `NotImplementedError: Cannot copy out of meta tensor`；如果当时图省事改用
  别的 offload 模式绕过报错，模型会带着一整套空的音频层跑完，**产出一个完全没有口型的
  视频且不报任何错**。补丁里加了断言（剩余 meta 参数 >0 直接 raise），复发时不再静默。
  修复后抽查 `audio_injection.proj1.weight`：`is_meta=False, absmean=0.002871`。
- **`sequential_cpu_offload` 在本仓库不可用，别去试。** 它是 `--GPU_memory_mode` 的上游默认值，
  文档上也确实最省显存，但 diffusers 的 submodule 级 offload 需要先把 `modulation` 类参数
  固定到 GPU 并搬 `transformer.freqs`——同源项目 FlashPortrait 的 `infer.py` 里就是这么做的
  （`replace_parameters_by_name(transformer, ["modulation",], device=device)` +
  `transformer.freqs = transformer.freqs.to(device)`），而 **echomimic_v3 的 `src/` 下这两个
  辅助函数根本不存在**（已 grep 确认）。用 **`model_cpu_offload`**：整模型级卸载
  （transformer 用完换 VAE），1.3B 规模下粒度已经足够，实测显存峰值 **13983 MiB / 16376 MiB**。

## 内存：真正的瓶颈不是显存

在 15.9GB 的 WSL 默认配额下，`infer_flash.py` **必然被 OOM 杀掉**（exit 137），
日志停在 VAE 加载完、T5 即将加载处，`nvidia-smi` 全程健康、显存峰值只有 599MiB。
上面两个补丁都是必要的但都不够——`load_file()`/`torch.load()` 仍要把 11.36GB 整个读进内存。

实测数字（本机）：

| 加载方式 | T5 峰值 RSS |
|---|---|
| `torch.load(map_location="cpu")`（上游） | ~23 GB（≈2× 文件） |
| `torch.load(..., mmap=True, weights_only=True)` | **0.38 GB** |

**已修（2026-08）：`src/utils.py` 新增 `load_torch_ckpt_mmap()`，替换 4 个文件共 6 处
`torch.load(..., map_location="cpu")`**（`wan_text_encoder.py` ×2、`wan_vae.py`、
`wan_image_encoder.py`、`wan_transformer3d_audio_2512.py` ×2；`wan_transformer3d_audio.py`
是 preview 路径，未改）。T5/CLIP/VAE 三个文件都实测支持 `mmap=True` + `weights_only=True`
（242/784/194 keys 干净加载），helper 里对需要真正 unpickle 的 checkpoint 保留了退回分支。

**mmap 改善的是内存增长曲线，不是峰值 —— 别指望它省内存。** 同一条 `smoke` 命令，
`/usr/bin/time -v` 实测：

| | 补丁前 | 补丁后 |
|---|---|---|
| **峰值 RSS** | ~23 GB | **21.9 GB**（几乎没降） |
| GPU 开始工作 | 256s | **<90s** |
| 90s 时 VRAM | 92 MiB（仍在换页） | **13983 MiB**（满负荷） |
| Major page faults（真实盘 I/O） | 38592 | **94825**（涨 2.5×） |
| 产物 | 97597 bytes | 97597 bytes（字节一致，无损） |

那个到处流传的「0.38GB」只在**只加载、不使用**时成立：`torch.load(mmap=True)` 返回时张量
还在磁盘上。推理必须 `.to(device)`，这一步把 mmap 页全部 page-in，峰值几乎回到原点。
（我一度把该数字当成推理峰值写进本文件，与同一批监控里 90s 时 RSS 13.8GB 直接矛盾 ——
凡是引用「加载峰值」的结论，都要用 `/usr/bin/time -v` 的 Maximum RSS 复核。）

机制上的收益是真实的：eager 加载要先把 11.36GB 整个读进缓冲、再复制成张量（瞬时 2× 尖峰），
mmap 则按页逐步载入，没有那个尖峰，所以 GPU 能早约 3 倍开始工作；代价是 major page faults
涨 2.5 倍（读盘从「一次大批量」变成「多次按需」）。

**所以 `memory=24GB` 必须保留**：21.9GB 峰值在 16GB 配额下必然 OOM。

## 跑通实测（2026-08，4070 Ti SUPER 16GB + WSL 24GB RAM）

首个成功产物：`smoke` 预设（`LEN=25`，`--sample_size 512 512`）→ 97.6KB / **368×672** /
25 帧 / 1.000s / 双声道音频。用 `echomimic3/run.sh` 跑，不要手敲那串 20+ 参数的命令。

- **`--sample_size` 是面积预算，不是精确尺寸。** 传 `512 512` 而输出 368×672 —— 模型按输入图
  的宽高比（demo 图是 640×1138 竖屏）重排了。Wan 系列的既有行为，不是 bug，但意味着不能靠它
  精确控制输出分辨率。
- **视频时长 = `--video_length ÷ --fps`**，且代码取 `min(音频时长×fps, video_length)`
  （`infer_flash.py:343`），所以 6.5s 的音频配 `LEN=25` 只会用到前 1 秒。帧数须为 8n+1。
- **加载阶段 VRAM 一直是 92MiB，不代表卡住。** `model_cpu_offload` 下权重先进 RAM，只在
  forward 时才拉进 GPU；实测 122s 时 VRAM 仍是 92MiB，256s 时才升到 7945MiB（RSS 8.0GB），
  峰值 13983MiB。T5 无 mmap 读盘本身就慢，别把加载慢误判成 hang。
- **不要用 `nohup ... &` 在受超时限制的会话里跑。** 会话超时会连带杀掉整个进程组，表现是
  日志停在半路、无 traceback、显存归零 —— 极易误判成程序卡死或 OOM（本轮为此浪费了大量时间）。
  用 `setsid`，或直接在自己的终端里跑。

## 关键参数（官方 Tips + `run_flash.sh` 默认值）

- `--audio_guidance_scale`：**1.8~2** 最佳。调高口型更准，调低画面更好。
- `--guidance_scale`（文本 CFG）：3~6。调高更贴 prompt，调低画面更好。
- `--num_inference_steps`：说话头 5 步够，半身/带手势 15~25 步。flash-pro 标称 8 步。
- `--teacache_threshold`：0~0.1。
- `--video_length` > 138 帧要开 Long Video CFG；`--partial_video_length` 调到 81/65 可降显存。
- 音频侧三个社区验证过的坑（LTX/Wan 同源，值得先排除）：**必须立体声 WAV**（mono 或 mp3
  会导致完全没有口型）、**太干净的 TTS 音频反而不同步**（加一点底噪才对上）、
  **唱歌音频要先分离人声**（LongCat 官方 pipeline 内置 `Kim_Vocal_2.onnx` 做这步，
  说明伴奏会干扰音频编码器）。本工作区 `sing.py` 的 SVC 干声天然满足第三条。

---

# Anime2.5DRig 离线渲染（`a25rig/`）

PSD 立绘 + `make_cues.py` 口型时间轴 → MV。上游是交互式浏览器工具，本 fork 加的是
**确定性离线渲染**：无头 Chrome 里跑同一套 rig，固定步长推进，直接出视频。
**详细文档在 `a25rig/RENDER.md`**（含全部旗标、实测数字、物理推导），这里只记
工作区级别的坑。

- **渲染入口是 `mv.py`（workspace 根，分阶段流水线），底层必须走 `a25rig/render.sh` wrapper，**
  不能直接 `node driver/render.mjs`。wrapper 负责设 `LD_LIBRARY_PATH`：puppeteer 自带的
  Chrome 需要 libnss3/libnspr4，本机无 root，这些 `.so` 是 `apt-get download` +
  `dpkg-deb -x` 解到 `driver/nsslib/` 的。直接跑 node 会报
  `libnss3.so: cannot open shared object file`。
- **`render` 阶段在受限沙箱里跑不了，表现是 `page.goto` 30s Navigation timeout**（不是脚本
  bug）：headless Chrome + Mesa D3D12 拿 GPU（ANGLE/GL）需要系统级设备访问，沙箱内 `goto`
  到 `../index.html` 一直等不到 load 就超时，随后 node 以 exit 1 退出。要在能访问 GPU 的正常
  机器上跑，或给该命令提权执行（2026-08 实测提权后 2s 冒烟片 6.9s 出片）。`mv.py --stages render`
  负责拼 `MESA_LOADER_DRIVER_OVERRIDE=d3d12 ...` 那套 env + `setsid`，文档不重复贴。
- **puppeteer 脚本必须放在 `driver/`**（`node_modules` 所在处），否则
  `ERR_MODULE_NOT_FOUND: Cannot find package 'puppeteer'`。
- **GPU 可用，但必须显式点名适配器**，否则选到 AMD 核显（比软渲染更慢）。
  Vulkan 必然回落 SwiftShader（`/usr/share/vulkan/icd.d/` 无 `nvidia_icd.json`）。
  实测 17.3 → 38.3 fps：
  ```bash
  MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  A25_GL_ARGS="--use-gl=angle --use-angle=gl --ignore-gpu-blocklist" ./render.sh ...
  ```
  **`h264_nvenc` 反而更慢**（32.8 vs 38.3 fps）—— 瓶颈是像素读取 + JPEG 编码，不是编码器。
- **雨是两个互不相干的层，不要试图用一个函数生成。** `--rain-glass` 是贴在玻璃上的水
  （raindrop-fx，MIT，画在角色**后面**），`--streaks` 是落雨丝（显式粒子，画在角色
  **前面**，因为真实雨落在主体与镜头之间）。早先自研的 Heartfelt 移植把两者塞进同一个
  函数，导致密度与长度无法独立设定，**已删除**（连带 `--rain-bg` 及 14 个 `--rain-*`
  旗标）。参考片按帧间持久性拆分是 **94/MP 静止水痕 + 10/MP 真正在落**；单帧形状区分不了
  两者（都是宽 ~6px、纵横比 4-5），我曾把 104/MP 全当雨丝，放大了约 10 倍。
- **玻璃的"模糊"与"水珠划过变清晰"是同一个机制**：玻璃上有一层逐帧累积的雾，水珠把雾
  **擦掉**（`[Zero, OneMinusSrcAlpha]`，即雾 alpha `*= 1-dropAlpha`），不是把画面变锐。
  水痕露出的正是 `--glass-blur` 那一层，所以它单独决定玻璃有多透。实测底图锐度占原图：
  0 → 100%、1 → 2.7%、2 → 0.8%、3 → 0.4%。**旧默认 3 时水痕只有周围的 0.5×（根本没变清晰）**，
  故默认已改 **0**（水痕/周围 8.8×）。`--glass-mist-time` 同时决定雾的上限与水痕保持清晰
  多久（恢复就是那个逐帧累加本身）。raindrop-fx 表达不了 0（`upSample(0)` 不写输出 → 黑屏），
  由 `setRainGlass` 包装 `blurBackground` 绕过。
- **落雨丝的长度和倾斜都不是自由参数。** 长度 = 速度 × 曝光时间（`--streak-shutter`
  度数，180° @30fps = 1/60s，Garg & Nayar SIGGRAPH'06）；倾斜 = `atan(U(t)/v_t)`，
  `U(t)` 是**全帧共享的单一风速**（多秒周期谐波叠加），`v_t` 是各滴终端速度
  （Gunn-Kinzer 1949）。曾按滴独立高斯抽样倾斜角，帧内 std 12.2°、范围 0.5-34.5°，
  物理上不可能；且误把 Bringi 的 canting σ（**滴轴摆动**）当轨迹散布。修好后帧内
  std 7.8°、相邻帧漂移 0.55°。
- **合成必须在页面内做（canvas2d），不能用 ffmpeg overlay。** 角色 canvas 是预乘 alpha，
  边缘亮度差：页面内 **−2.9** / ffmpeg overlay **−75.4** / `unpremultiply`+overlay **−49.4**。
- **`CW`/`CH` 直接等于 PSD 尺寸，所有锚点/形变常数都在这套坐标里，不能改**；换比例必须
  另加输出画布层（`--size` + `--char-scale/-x/-y/-anchor`）。
- **`<video>` 背景必须支持 Range 请求（206）**，否则 `currentTime` 赋值永不完成，
  **卡在第一帧且不报错**。
- **未知旗标现在会 exit 2。** 之前 `arg()` 对没问过的参数一概忽略，所以拼错或用了已删的
  旗标会静默渲染出黑背景并 exit 0。**但旗标的"取值"仍无校验，同一类坑低一层还在**：
  `--char-x/--char-y` 是 0..1 归一化比例（`cy=y*h`，`index.html:1338`），传 `--char-y 60`
  等于把角色推到画面下方 60 个画面高处，产出纯背景且 exit 0；`--char-anchor` 只认 `face`
  （`:1340`），`bottom`/`top`/`centre` 全部静默退化成画布中心。
- **`--size` 必须配背景**，否则 `renderFrame` 在 `index.html:1356` 抛
  `Cannot read properties of undefined (reading 'getContext')` 且 0 帧产出 ——
  合成画布 `_c2` 只在 `setBg()` 里创建，`setOutput()` 只建 `_oc`。纯色背景传 `--bg '#000'`。
- **`--out`（PNG 序列）比 `--video` 慢 2.6×，不要用 `--out` 测性能。** 同一条 8s 命令
  （1920×1080 + 玻璃 + 雨丝）只换输出方式：`--out` 25.3s / 9.5 fps / 105ms 每帧，
  `--video` 9.6s / 25.0 fps / 40ms。`--out` 走 PNG 编码 → base64 → `writeFileSync`
  （`render.mjs:401-408`），`--video` 直接 JPEG 管进 ffmpeg。两个反直觉的后果：
  (1) **PNG 耗时取决于画面复杂度而非像素数** —— 1280²(63.8ms) 比 1920×1080(44.2ms) 更慢，
  因为方画幅被角色细节填满而宽画幅有大片平坦区；水珠/雨丝的噪声同理，玻璃在 PNG 路径
  上 +61ms 而在 video 路径上只 +20ms。(2) **在 `--out` 下观察到的变慢与渲染器无关** ——
  本轮据此误判 `--cues` 造成 4× 变慢，实际全是 PNG 路径。`--out` 只在需要逐帧像素分析时
  用，用完即删：8 组 24s/1080p 实验占了 **21 GB**。另外要读驱动打印的 fps 而不是 wall
  时间，进程启动（Chrome + PSD 加载 + 玻璃 warmup）有固定 4-6s，短测试里它占主导。
- **`--cues` 本身零开销**（配对 A/B，8s，其余参数相同）：27.90s / 202092 KB peak RSS
  无 cues vs 28.34s / 201904 KB 有 cues，即 +1.6% 时间、−0.1% 内存，都在噪声内。
  内存大头是 `--rain-glass`（1920×1080 + `--glass-droplets/limit 900` + 多级模糊金字塔
  ≈ 2.4 GB Chrome+node）。
- **传了 `--cues` 但时间轴有空洞，照样落到随机嘴 —— 这是更隐蔽的一层，已修。** 上一条说的是
  完全不传 cues；真正会在每首歌上发生的是 cues 存在但**有洞**：intro、间奏、乐句休止处
  `make_cues.py` 本来就不产 cue，`cueMouth()` 未命中返回 `null`，同一个 `auto.talk` 接手。
  实测一份 100s 的 cues **22.6% 时长完全没有 cue**（intro 19.0s + 间奏 3.3s + 3 个碎片），
  那 22.6% 全程由随机数驱动。已改成按 `OFF.cues` 分支：有时间轴就以它为权威，空洞即闭嘴，
  `--rest-form`（默认 0.35）给一点嘴角上扬（0 是平嘴，19s intro 里显得阴沉）。
  显式 `X` 走同一姿态，否则表情会取决于 make_cues 是吐了 X 还是留了洞。
- **验证「嘴动没动」不能用口腔色块探针，必须用帧间差分 + 面部对照。** `auto.talk` 驱动的是
  单图**形变**而非换贴图，所以色块探针对两个臂都报 ~0，bug 能完整躲过这道检查（本轮就是
  这样差点判成已修好）。正确做法是对嘴部区域做相邻帧差分，并与脸颊/额头对照：合格成片里
  嘴区 **0.324** < 脸颊 1.253 < 额头 1.322，且嘴与脸颊正相关 **+0.61** —— 这个相关性才是
  「残留微动是 `auto.idle` 呼吸带动整张脸」而非「嘴在开合」的判据。
- **口型只启用 A/B/C/E 四种（`make_cues.py` 侧收缩）。** 九宫格素材的 D(大开)/F(撅唇)/
  G(唇齿)/H(长L) 在本立绘上很难用对，D 一出现就明显违和。40 个韵母全部映射到
  C(开口)/E(微圆)/B(微开)，辅音只留 A(闭唇) 给 b/p/m；归并沿 Rhubarb 官方过渡链回退一步
  （D→C、F→E、G→B、H→C），`push()` 里有 `FOLD` 兜底。依据是 Bristow《5 things that can go
  wrong with animated lip sync》——口型过大是新手错误第一位，且 anime 常只用 open/shut
  两型也没人抱怨。**注意 Rhubarb 的 H 是 long "L"（舌尖抬起），不是 ü**，我一度弄错。
 - **口型的文本源优先用真歌词强制对齐，不要用 ASR 转写 —— 这是本轮最大的免费收益。**
   `make_cues.py` 不传 `--text` 会走 paraformer ASR，但它在 SVC 转换后的人声上漏字严重：
   云烟成雨全曲 ASR 只认出 **103 字**，喂真歌词（`get_lyrics.py` 的缓存）能对齐 **347 字**，
   且不再听错词。代价是零 —— 歌词本来就为字幕抓过，347 字对齐只花 14.5s。所以流程顺序应是
   先 `get_lyrics.py`、再 `make_cues.py --text`。
 - **全曲口型时间轴直接用增强 LRC（`make_cues.py --lrc`），不要做 ASR 强制对齐 —— 这是
   2026-08 的最终结论。** fa-zh（funasr 强制对齐）在长音频上系统性漂移（实测全曲 96.74s 起
   每行错位 +4.5~+7.5s；FunASR #2784/#2019 确认是模型 bug 且官方不修，「官方基本不会处理
   了」），逐句切段对齐 + ASR 锚定歌词区间绕不开三个互相纠缠的问题：36s 长句的歌词定位、
   副歌重复段文本匹配错位、句子 lo 乱序。**口型对齐的真实需求是「时间准 + 发音近似」，而
   KRC 行级时间是人工校对、与音频能量跃变 <0.2s 偏差（43/43 行落在有声处）** —— 逐字时间
   反而不可信（雨爱「彩虹的美丽」把「美」「丽」标进 21s 间奏的静音区）。所以 `--lrc` 路径：
   行起点 = LRC 时间戳（权威），行内字数均摊到「行内第一段 ≥0.35s 长静音的起点」
   （`first_long_silence`，与 `--max-hold-sil` 同判据），拖腔/闭嘴交给下游能量门控。
   验证判据：行起点落在有声处 + 间奏/尾奏全闭嘴 + 唱歌段全张嘴。
   **行结束不能用 `last_voiced`（找最后有声）** —— 它会把下一行的起音算进本行（雨爱
   「彩虹的美丽」行的下一行「冷」气口在 124.4s，5 个字因此均摊到 27.7s、间奏张嘴 25s），
   「第一段长静音」从源头断开才对。用法：
   ```bash
   GPT-SoVITS/.venv/bin/python listen_lipsync/make_cues.py \
       --wav vocal.wav --lrc data/songs/<名>.lrc --out cues.json --gate-db -40
   ```
   `--lrc` 优先于 `--text`，完全不加载模型（省 30s+ 启动、零显存）。
 - **被手工删段的人声（剪掉唱得不好的低音段）会打破 `--lrc` 的行锚点，整行塌成 ~100ms 微闪。**
   2026-08 情歌全曲：用户在 vocal.wav 里把 4 段低音唱烂的地方改为静音（25.7-27s / 125-126s /
   140.6-142s / 246.6-249s，1s 分辨率对 pre-edit 基准 `vocal_baseline.wav` 的 voiced 网格差分
   找到，都是「掏成静音、时间轴不位移」的洞）。KRC 其余行锚点依然权威（30-100s 逐 10s 首发声帧
   与旧档一致），但**行起点落进洞的那几行**，`first_long_silence` 当头 0.35s 无声就立即截断 →
   整行字挤进 ~0.1s，下游 cues 的「时间轴倒序 + 退化块(≥2字同起点) + 微字(<50ms)」三连报警。
   修法 = 把受影响行的行起点改锚到洞后首个「持续 80ms 有声」帧（情歌两处：25.73→27.08、
   140.61→141.46），其余行不动，重跑 `--lrc` 即成（唱着却闭嘴 10.5%→4.9%）。实现上用
   `read_enhanced()` 读进内存、改受影响行的 start、重写行级 LRC，不碰原缓存。尾奏修整
   （246.6-249s）**不用**重锚 —— 尾部分支本来就按最后有声处闭嘴。KRC 头行「歌手 - 歌名」做
   cues 前要删（它在无歌的 intro 里也被长静音判据截成 ~0.1s、t=0 嘴微闪；`" - " not in text`
   过滤即可）——但这行字做 ASS 字幕时要**保留**（KTV 片头歌名卡，见字幕节）。
 - **`--text` 强制对齐在重复乐句上塌缩，是「用 `--lrc`」之外的又一个送命点（2026-08 实测）。**
   第二遍「我们在 怀念的演唱会」（~175-184s，与第一遍结构相同）fa-zh 把 我/们/在/怀 四字给同一个
   起点 175.13，其后 念的演唱会礼貌的吻别 全变成 20ms 步进的微字（伴随 `句 33 对齐 11 != 13 字，
   该句留待插值` 告警），下游 cues 直接挂「时间轴有序」检查。注意它只查「唱着却闭嘴 %」看不出来
   （口型没碎、能量也对），必须跑 README ③ 的「产出检查」脚本里的倒序/退化块/微字扫描。
- **尾奏静音会让最后一个口型撑到文件结尾（已修）。** `make_cues.py` 尾部分支原本用
  `is_voiced()` 的**单帧**判据，淡出时的孤立过限帧就判成「还在唱」：全曲版歌词 220.6s 唱完、
  音频 240.8s，尾部 20.2s 是数字静音（−175dB），而嘴张着 `C` 型僵了 20 秒。改为复用
  `last_voiced_before()`（「持续 80ms 有声」，与 `--max-hold-sil` 同一判据）后，尾段实测
  99 帧张嘴 **0%**，`X` 占比 3%→13%（与真实静音量吻合）。**这是同一个坑第二次复发**（第一次
  是「昏」字后 3.5s 静音被判拖腔），凡是「这段有没有声」的判断都必须用持续判据。
  **100s 窗口看不出来** —— 它正好切在歌声中间，没有尾奏；全曲出片才暴露。
- **验收 cue 长度时必须排除 `X`。** 尾奏的 `X` 本来就可以长达 20s，与张嘴 cue 一起统计会
  永远误报「误判拖腔」。判据改成「最长**张嘴** cue」+「最后一个张嘴 cue 是否落在真正唱完处」。
- **不传 `--cues` 时嘴部由随机生成器驱动，且完全静默。** `cueMouth()` 无时间轴返回 `null`，
  代码落到 `else if(auto.talk)`（`index.html:915`）—— 上游"随便动动嘴"空闲动画，
  而 `begin()` 从不关掉 `auto.talk`。720 帧对真实人声包络实测：随机臂 r=**−0.06**、
  「在唱却闭嘴」**43.8%**；关掉 `auto.talk` 嘴部开合归零（0/720 帧张开，证明它是唯一
  驱动源）；带 `--cues` 后 r=+0.22、「在唱却闭嘴」降到 **2.4%**。随机臂 r≈0 正是判据：
  它不是"同步但被扰动"，而是与音频互相独立的随机过程。
- **cues 质量不能用「与人声能量的相关」判断，要看字级对齐。** 带 cues 的 r=+0.22 是
  **正常的**：口型编码**元音形状**（a 大张 / i 微开 / u 收圆），本就与音量无关 ——
  「i」唱得再响嘴也是小的。判据应是「每字都有时间戳、无倒序、起点落在有声处」。
  本轮两次静音阈值用错，先后把同一份对齐判成坏和好：用 RMS **p25** 报出「32.9% 字起点
  落在静音里」，换绝对阈值（−40dB rel peak）后是 **0/79**，且起点后 100ms 内 79/79 有
  能量 —— p25 在唱歌素材上圈的是 −15dB 的连音而非静音；镜像错误是「79% 静音帧嘴大张」，
  那个 p20 阈值圈出的 144 帧里真静音只有 **4 帧**（其中 2 帧张嘴），连音时嘴本该张开。
- **cues 与音频窗口严格绑定**：现成 `cues_hybrid.json` 对应某曲 66-86s，而视频用 60-84s，
  差 6s 即不可用，必须对实际渲染的窗口重新对齐。**在 SVC 干声上对齐、混音版出片是可行的**
  —— SoulX-SVC 保留源时间轴：`vocal.wav` 与 `sep2/lead.wav` 逐 10s 窗互相关最大位移
  **10ms**（= 测量 hop，0.3 帧 @30fps）。有一个窗报 −10s 是伪值（间奏，RMS 0.000000，
  相关无定义）—— 看到离群值先查该窗是否全静音。
- **「角色不见了」不能靠像素统计判断。** 饱和度剖面、暖色掩码、分块帧差在带 `--rain-glass`
  时全部失效 —— 雾和水珠主导了这些统计量，量到的是玻璃不是人。本轮据此连续得出两个错误
  结论（先"水珠糊住了角色"，再"角色其实在画面里"）。**有效判据**：渲染两帧只改
  `--char-scale`，若角色在画，差异必然达 80% 像素级别；最终观感必须由人目视确认。
- **`--visemes data/pics/visemes` 必须传，漏了口型完全不动且日志不报错。** 默认值是
  `null`，`render.mjs` 的 `if (visemes)` 分支整个跳过——渲染照常完成、帧数正确、连嘴部
  区域像素都会「随歌声变化」（那是形变/阴影伪差异），**唯一线索是启动日志里没有
  `visemes: A B C D E F G H X anchor=(679,587) box=89x51` 那一行**。带上的判据：
  唱歌段嘴部暗红像素 800-1000+、间奏/尾部 <5；漏传时唱歌段只有 10-43。完整渲染参数
  清单（雨爱全曲实测）：`--psd data/pics/figure.psd --cues cues.json --visemes
  data/pics/visemes --rain-glass data/pics/rainy_love_bg.png --glass-blur 0
  --glass-mist-blur 4 --glass-mist-time 10 --streaks --streak-density 2.8 --sway
  data/songs/<名>/beats.json --sway-beats 1 --size 1920x1080 --char-scale 1.15
  --char-x 0.62 --char-y 0.56 --video out.mp4 --audio mix.wav --fps 30`（CWD 必须是
  voice 根目录，否则 psd fetch 404；`--rain-glass` 是路径参数不是 flag，裸传会
  `ERR_INVALID_ARG_TYPE`）。
- **`--rnd-pose` 默认改为 0（2026-08）：随机姿势漂移会把卡拍的摆动「糊掉」。**
  随机姿势（`index.html` 的 rnd 通道，每 1.4-4.0s 重采样 + ~0.9s 缓动）与 sway 落在
  同一个 `tgt.body`/`tgt.angleZ` 通道上：它不改变摆动的**相位**（probe 直接读几何值
  验证极值仍卡在拍子 ±12ms 内），但会移动**摆动中心**——0.35 时中心漂移 std 0.059、
  max 0.104（= 摆动幅度的 35%），左右极值不对称（实测 +0.388/−0.183 → +0.259/−0.336），
  视觉上就是「摆着摆着往一边偏、没卡在拍子上」。30s 并排 A/B（body 0.70 放大差异）
  确认是主要嫌疑。需要 idle 生命感时显式传 `--rnd-pose 0.15`（中心漂移降到 ±0.045，
  可忽略）；gaze（眼睛）随机不受影响。
- 确定性靠三件事：固定步长 `1/fps`、播种 mulberry32、`vendor/raindrop-fx.js` 内 6 处
  `Math.random()` 改走 `globalThis.__rdRandom`。另外玻璃 sim 要先 warmup 固定帧数
  （水珠数 44→280 于 ~30s 饱和），否则开场是干玻璃、且帧 N 会依赖起始位置。
  验收方法：同种子渲染两次比 md5。
- **fork 已推到 `CJYLZS/Anime2.5DRig` 分支 `offline-render`**（remote 名 `fork`，
  上游 `852wa/Anime2.5DRig` 为 `origin`）。`driver/node_modules/`、`driver/nsslib/`、
  `out/`、`frames/` 在 `.gitignore` 里。

---

# 前奏口白（`listen_lipsync/intro_speech.py`，2026-08）

MV 成片前可在器乐前奏里用角色音色说一段话。**素材目录约定**：一首歌的全部渲染中间物**和成片**
放 `data/songs/<歌名>/mv/`（口白 txt、cues、beats、audio、混音、口型、预览与最终 mp4，及
`100s/`、`ab/` 子目录），`.tmp` 只留跨命令临时物、`data/video/` 不再放新产物。"是否已备料" =
`mv/` 下 `cues_intro.json`/`audio_intro.wav`/`beats.json`/`speech.wav` 是否齐。渲染入口是
`python3 mv.py`：`--with-intro --tts-ref <参考音频>` 自动带出 intro 阶段（调 intro_speech.py）
并做口白版素材；成片写回 `mv/<歌名>_intro.mp4`，**默认渲完自动烧 KTV 字幕** →
`mv/<歌名>_intro_lyrics.mp4`（复用 `mv/lyrics.ass`，缺则 `make_ass.py` 现建，烧前
`fc-match SimHei` 校验字体配置，`--no-subs` 跳过烧字）。详见 README ⑥/⑥½/⑦。

- **说话模型选型（2026-08 A/B，同一文案，2026-08）**：微调 girlish（`--tts gpt-ft`）> 基座
  零样本（`gpt-zs`）> VoxCPM2（`voxcpm2`）。eval_sim（3s 窗 vs girlish 参考）：
  `gpt-ft 0.8734` / `gpt-zs 0.8406` / `voxcpm2 0.7807` —— 说话→说话同域，天然高于 SVC 的
  0.5-0.6。SingMOS 三者在 ±0.05 噪声内互博、NISQA-TTS 偏好 voxcpm2，**MOS 结论打架**；
  MV 的主导判据是**相似度（音色同源 = 与歌声观感一致）**，故默认 `gpt-ft`。A/B 产物留 `mv/ab/`。
- **硬约束在脚本里 fail-closed**：`dur ≤ first_lyric − offset − margin`，绝不压到唱段。
  情歌首句 27.08s（重锚后），示例口白 5.68s @ offset 1.5 → end 7.18、buffer 19.9s。
- **口白口型 = 单独跑 make_cues（`--melodic-accent 0` 说话型）再并回前奏区**：
  `X(0,offset) + 口白段 + X(尾, first_lyric)`，唱歌段 cues 原样不动（混音侧可断言逐样本
  不变）。别在唱歌参数（Ma=0.6）下生成 —— 短元音会被糊成唱歌腔。
- **合并自检只认「相邻同起点」是崩塌**：相邻 cue `start == 前一个 end`（无缝衔接）是正常连接，
  不是塌缩 —— 初版把后者误报为崩塌（见 `check_speech_cues`）。
- **voxcpm2 在受限沙箱必须 `HF_HUB_OFFLINE=1`**：`from_pretrained` 每次都会重建
  `~/.cache/huggingface/.../snapshots/<rev>` 指针，沙箱 home 只读即 EROFS；离线模式走
  `refs/main` 命中既有缓存，不经网络也不写盘。
- **GPT-SoVITS `infer.py` 的 cwd 必须是 `GPT-SoVITS/`**（模块导入要仓库根），所以
  intro_speech 给它的文件路径全用绝对路径 —— 相对路径会写进 `GPT-SoVITS/.tmp/...` 而读不到。

---

# 歌词获取与 KTV 字幕（`listen_lipsync/get_lyrics.py` + `make_ass.py`）

字幕**烧在已渲染好的视频上**（`ffmpeg -vf ass`），不进 a25rig。理由是改字幕只要重烧几秒
（22s 片段 5.5s、全曲 ~90s），走页面内 canvas2d 则每次都要重渲整片（全曲 4.5min），而 libass
的 `\kf` 原生支持逐字扇光、排版能力（描边/阴影/淡入/对齐）也现成。ffmpeg 已编入
`libass`+`libfreetype`+`libfribidi`+`libharfbuzz`，无需装任何东西。

- **逐字歌词可以匿名拿到，用酷狗 KRC，不要用网易云。** 网易云的逐字字段 `yrc` 实测恒为空
  （云烟成雨与一首对照热门曲都是 0 字节，说明需要客户端级鉴权），只能拿到行级 `lrc`。酷狗
  `krcs.kugou.com/search` + `lyrics.kugou.com/download?fmt=krc` 完全匿名，返回 base64 →
  XOR 固定 16 字节密钥 → zlib 解压，文件头是 `krc1`，格式 `[行起ms,行时长ms]` + 每字
  `<相对起点,时长,0>`。10 首实测：**4 首逐字、5 首行级回落、1 首两个源都没有**。
- **酷狗搜索必须带「歌名 + 歌手」，且歌手要从音频标签读，不要联网猜。** 只传文件名候选恒为 0，
  加上歌手才有 10 个候选。10 首 FLAC 全带 `title`/`artist`/`album` 标签；我一度先去网易云搜歌手，
  猜错两个（爱的回归线 实际是陈韵若、白色毛衣 实际是黄星侨，与网络结果差一个字）。
- **酷狗接口会静默限流，失败长得和「这首歌没歌词」一模一样**（HTTP 200 + 空 candidates）。
  实测同一 query 连发 6 次得到 `10/10/10/0/10/10`，间隔 2s 则 6/6 命中。不限速时测出的命中率
  全是噪声 —— 我先后得出 1/10 和 4/10 两个矛盾结果才发现。`get_lyrics.py` 内置 1.2s 限速 + 重试。
  同类静默非确定性还有 FunASR VAD（见「kNN 的吐字代价」一节），这是本工作区第三次踩。
- **`duration` 参数是模糊筛选而非精确匹配**，传精确时长时候选时长可以是 155-220s（真值 240.8s），
  不传则退化成 13-40s 的碎片。所以候选要自己按 `|候选时长 − 文件时长|` 排序，不能信第一个。
- **LRC 自带的行首时间不能直接用来做扇光，必须本地重新对齐。** 网易云 LRC 系统性提前（给 KTV
  播放器留提示用）：实测中位 **+0.78s**、最坏 **+3.76s**，17 行里 **4 行落在静音处**。相比之下
  酷狗 KRC 与音频能量跃变只差 0.2s（17/17 落在有声处）。真值判据是逐 0.2s 扫能量找静音→有声的
  跃变；两个争议行上 KRC 对、强制对齐错（对齐器漂了 2.7s）。
- **逐字源顺带解决了「行内长静音」，这是强制对齐做不到的。** FunASR `MonotonicAligner` 不建模
  休止，只能把字均摊到整个区间，于是间奏被吞进行内：同一 100s 窗口内强制对齐有 1 行长 11.42s
  且含 3.6s 静音，酷狗 KRC 是 **0 行**、最长行 7.95s、行间留出 0.83s 中位空隙（间奏 4.36s）。
- **缓存成歌曲同名 `.lrc`，格式用标准「增强 LRC」而不是自造格式。** `[mm:ss.xx]<mm:ss.xx>字`
  是通行的逐字扩展：任何编辑器可改、任何播放器可验、只有行级时自然降级。行尾补一个「唱完」
  时刻给扇光收尾。10 首缓存共 **30KB**；`--offline` 强制只读缓存（生成 ASS 0.039s），`--force`
  才重抓。原子写（`.tmp` + rename），中途中断不留半个缓存。
- **抓不到时写「带时间轴骨架的手工填写模板」**，顶部是格式说明，下面是按人声能量自动切的乐句
  锚点。两个坑：骨架必须用**分离出的主唱**（`sep2/lead.wav`），拿混音切只得到 1 个锚点（伴奏
  一直有能量）；静音判据 0.45s 太粗，收到 **0.20s** 后从 7 个变 17 个（对照同参数在云烟成雨上
  切出 36 句 vs KRC 真值 42 行，量级吻合）。
- **两个同样式 ASS 事件时间重叠会触发 libass 防碰撞位移。** 表现是换行时上行槽空了、字留在预览
  槽里（看起来像「不滚动」）—— 实测上行槽 222px vs 预览槽 23352px。`--lead 1.0` + `--hold 0.4`
  超过句间空隙就会重叠（实测重叠 0.58s）。`make_ass.py` 的 `no_overlap()` 在两句之间取中点切开。
- **双行滚动的上下顺序在底部对齐时会反。** ASS 的 `MarginV` 顶部对齐时从顶边量（加值=往下），
  底部对齐时从底边量（加值=往**上**）。所以底部对齐要让**主行**拿更大的 MarginV 才会被顶到上面。
  我第一版直接沿用顶部的公式，注释还写着「底部对齐方向自动对了」，那句判断本身就是错的。
- **验字幕必须与无字幕原片做差分，且要设噪声门。** 直接测颜色会把背景算进去 —— 雨玻璃偏蓝，
  被当成「已唱」，一度得出「扇光 0.5s 冲到 99% 然后僵住」的假结论（ASS 里 `\kf` 之和 5.91s /
  事件 6.31s 本来就是对的）。差分后还要设 `<200` 采样点判为无字幕：h264 重编码会留个别孤立像素，
  全曲抽检时把包围盒从 y920-1039 撑成 y246-982。**看质量分布，不看 min/max。**
- **系统里没有中文字体**（`fc-list :lang=zh` 为 0，`fc-match ':charset=4e00'` 回落到不含汉字的
  DejaVu），直接烧是整屏豆腐块。不装系统字体的做法是 `FONTCONFIG_FILE` 指向
  `/mnt/c/Windows/Fonts`。用 `simhei.ttf`（单 face）而不是 `msyh.ttc`（字体集合，还要选 face）。
- **`FONTCONFIG_FILE` 指向的配置文件必须放工作区，不能依赖 `/tmp` —— 本工作区的命令沙箱会
  在调用之间清空 `/tmp`（2026-08 实测）。** 第一次烧字幕时 fonts.conf 放 `/tmp`，后台烧录 job
  启动时文件已不存在：ffmpeg **exit 0**、stderr 只有 4 行
  `Fontconfig error: Cannot load default config file: No such file: (null)`（libass 拿不到任何
  字体），画面**一个字都没有**，看起来完全正常（如果配着「无字幕真是这个画面正常」就完蛋）。
  改放 `.tmp/fontconf/fonts.conf`（`<dir>/mnt/c/Windows/Fonts</dir>` + cachedir 也放工作区），
  烧前先 `FONTCONFIG_FILE=... fc-match SimHei` 验一下。同理 `/tmp` 里的任何中间产物
  （fonts.conf、抽的帧、clip）都别指望跨命令存在 —— 跨命令要用的放仓库内，单命令内的烧录/抽帧
  分析在一条命令里做完。
- **验「字幕烧进去了」时注意两片会错 1 帧，别直接单帧差分（2026-08 实测）。** 雨丝/晃动/口型
  这类动画面错 1 帧会让全帧差异爆表、字幕带 100% 非零全是假象。可靠做法：原片 vs 烧录片各解码
  1s、逐帧对齐（±3 帧内取整帧 mean absdiff 最小，实测 offset=0），再比「字幕带（y≈900-1040）
  mean absdiff」与「控制带（顶部/中部）」比值 —— 情歌实测 **7.45 vs 0.46/0.78（9.5×）**，
  重编码噪声本身只有 ~0.5-0.8。完全不依赖肉眼/原片的兜底是**黑底 ink test**：`-f lavfi
  -i color=black` + `ass=...` 出一帧，数字幕带亮像素数与两行 bbox（居中双行 ⇒ 文字真排上了）。
  ⚠️ 黑底测试抽 t≈0 会碰上 ASS `\fad` 淡入全透明，抽 t=2s 之类真实事件中段。

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

## 漏音检测与局部修复（2026-08，雨爱实测）

**检测：1s 单模型逐窗不可靠，必须 3s + 双模型共识 + 持续性。** 同一协议（窗口 vs
全局锚点、excess/margin、5 个 impostor）下 ERes2NetV2 与 CAMPPlus（`eval_leak_campplus.py`，
seed-vc 自带 funasr campplus，192d fbank 输入）对 1s 窗口的 excess 判定 Spearman 仅 +0.28、
TOP-12 重合 1/12（3s 窗升到 +0.54 / 5/12）；两模型都报的 3s 窗才是可信漏音区（雨爱：
78/156/192/213/219s，恰为反复副歌句）。`leak_agree.py` 做两 CSV 一致性 + 尖峰持续性 +
听感片段导出（**注意 vocal.wav 24k 与 sep2/lead.wav 44.1k 采样率不同，导出必须按各自
采样率写，否则原唱片段变慢放**）。

**修复：`repair_leaky.py` 局部重转漏音段，网格选优拼接。** 对每个区域：切 lead+F0
（两侧各 1s pad 给上下文）→ (seed × knn_alpha × cfg) 网格重转（子进程须
`cwd=SoulX-Singer` + `PYTHONPATH=SoulX-Singer`，且**所有路径绝对化**，相对路径会在
cwd 下解析失败）→ ERes2NetV2+CAMPPlus 双模型 1s 逐窗打分，选 score =
Δexcess + 0.3·Δmargin 最大、A 不塌陷（ΔA > −0.015）且 score > 0.005 的臂 → 0.15s
交叉淡化拼回。雨爱实测（4 区域 × 8 臂）：**α=0.4 臂漏音降幅最大但 CER +3.0，
α=0.2 臂 CER +1.9 且漏音降幅几乎相同**（R1 150-162s excess −0.013/−0.014 双模型、
R3 210-225s campplus −0.034）；区域 2（186-198s）两模型分歧大（eres2net 无感、
campplus 0.112），α0.4 修复代价 4 吞字，不值。SIM 全版本不降反升（0.7794→0.7820）。
**结论：漏音段修复默认用 α=0.2 臂；CER 是硬约束，每版必须过 eval_cer。**

**流水线集成（2026-08）：leak-detect 默认开，leak-repair 默认关。** `sing.py` 新增两个阶段：
- `leak-detect`（默认，svc→eval 之后）：`w_leak_detect.py` 双模型(ERes2NetV2+CAMPPlus)
  1s+3s 扫描 → `leak_report.json`（逐窗 excess/margin、共识标记、区域、建议修复跨度）。
  共识 = 两模型 excess 都 ≥ max(地板, p85)（地板 er 0.05 / cp 0.07）；跨度锚定原版气口。
  impostor 自动发现 data/songs/ 其他歌的 sep2/lead.wav（最多 5 个），可用
  `--leak-impostor` 覆盖（**注意 nargs="*" 的重复 flag 只有最后一个生效，必须用
  action="append"**）。
- `leak-repair`（`--leak-repair` 开启）：`w_leak_repair.py` 按报告跨度定点重转
  （固定臂 `--leak-arm-seed/alpha/cfg`，默认 1/0.2/0.5）→ 增益门（重转 excess 不低于
  基线则不替换）→ 包络对齐 + 公共静音拼接 → **vocal.wav 换修复版，原输出备份为
  vocal_raw.wav** → 自动复检写 leak_report_after.json。
- 全曲重转一轮无助于整体漏音（seed1 实测：逐窗 55 改善/54 变差，均值 +0.0002），
  定点修复才有用；均值被分离歌声域偏置地板主导，动不了。
- **跨歌验证（2026-08，那女孩对我说）：检测判定"无漏、跳过修复"是真实的。** 3s 双模型
  excess：er mean −0.0009 / cp mean +0.0137（雨爱对照 0.031 / 0.054），两个模型的 p85
  都低于地板阈值（er 0.05 / cp 0.07），er 超阈 5 窗 ∩ cp 超阈 9 窗 = **0 窗共识**，
  于是 leak-repair 直接 no-op（不创建 vocal_raw.wav）。这首歌基线就干净，无需修复。

### 2026-08 三首歌实测后的三个修复（repair_leaky.py）

**① 采样率 bug：拼接整体提前 0.84s。** `region_slice` 的 `pad_samp` 用 lead 采样率
（44.1k）算，但拼接时它作用于 SVC 输出（24k）——1s pad 被切成 44100 样本而非 24000，
多吞 0.84s 内容，修复段内容整体提前（用户听感"拼接靠前"，F0 逐秒互相关实测
−500~−760ms）。修复：`pad_samp = PAD × SR_OUT`。**SVC 输出长度与输入严格等长
（实测 +0ms），时间轴本身是 1:1 的**——错位全在调用方的切片采样率。

**② 拼接点两级算法（用户要求"小节边界 ∩ 静音"）。** 实测发现**绝大多数歌的小节边界
（强拍）处没有连续静音**（雨爱 vocal/lead 双确认，全曲边界 ratio 最低 0.63），
"静音的小节边界"在物理上常常不存在。最终算法（`--bars` 给节拍网格后启用）：
  - 首选 = 静音段中心恰在小节边界上（±0.15s）→ "小节∩静音"；
  - 次选 = 最近的**足够长静音段中心**（RMS < max(0.02, 0.25×中位) 连续 ≥0.3s，
    取段中心保证两侧都有静音缓冲），放弃小节约束，静音切换优先；
  - 扩展上限 4 小节；无可用静音则保持小节边界并警告。
  实测：那女孩对我说 r0 起点 ratio 0.0（真静音小节边界）、最后一页 r2 终点 0.06；
  雨爱全部走"静音次选"（r0 59.4/69.3s、r1 161.5s 静音段 1.35s、r2 186.0/206.5s）。

**③ 拼接手感：150ms 淡化太短 + 音量跳变。** 交叉淡化 0.15 → 0.35s；接缝处 RMS
匹配（头尾 xf 窗的 old/new RMS 比作增益，指数衰减到 1，clip 0.3-3.0）消除音量跳变
（实测 t=66.87s 修复段尾部 RMS 0.036 vs 原 0.012 的跳变被压平）。

**对齐验证方法：F0 逐秒互相关，不是波形互相关。** 波形互相关在转换音色上 corr 只有
0.05-0.08（音色不同，波形本来就不像），完全不可信；F0（50Hz 网格，log-F0 绝对差
最小化）才能定位错位。周期混淆（重复音型）用"前 3 候选 err 对比"识别：err 几乎相等
的多个 lag 是混淆（最后一页 161/167s 的 −360ms 是假阳性），err 唯一最优的 lag 才是
真错位（那女孩 129/132s 的 −60~−100ms，SVC 段边界效应，量级远小于 bug 级错位）。

三首歌修复后全曲：SIM 持平略升（雨爱 0.7783→0.7810、那女孩 0.7613→0.7654、
最后一页 0.8018→0.8034），区域 excess 0.04-0.09 → 0.001-0.035。**修复版已替换
vocal.wav（原版备份 vocal_raw.wav），`--bars` 需先跑 `beat_grid.py` 生成节拍网格
（full mix 输入，80 BPM 档 resid p90 ~14ms）。**
