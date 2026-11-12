# voice workspace — 从歌曲到 MV 全流程

两条独立的流水线，前者产声、后者产画：

```
① sing.py                                       ② MV 渲染（入口 mv.py 分阶段流水线）
歌曲.flac + 参考语音                        vocal.wav / mix.wav → data/songs/<歌名>/mv/
  └─ sep → ref-sep → f0-target → select            ├─ slice    align.wav(16k) + audio.wav(44.1k)
     → ext → svc（segprompt 段落级，默认）          ├─ lyrics  get_lyrics.py（.lrc）
     → eval → mix                                  ├─ cues    make_cues.py（口型时间轴）
        ├─ vocal.wav 干声（口型用这个）             ├─ beats   beat_grid.py（节拍网格）
        └─ mix.wav    人声+伴奏（成片用这个）        ├─ visemes split_visemes.py（口型 sprite）
                                                   ├─ intro   intro_speech.py（可选前奏口白）
                                                   ├─ render  a25rig/render.sh（渲片）
                                                   └─ subs    make_ass.py + ffmpeg（KTV 字幕）
                                                        → data/songs/<歌名>/mv/*.mp4
```
```

`sing.py` 本身的参数、调优结论、各引擎 A/B 见 `AGENTS.md`；本文件只讲**从它的产物做出 MV**。
贯穿示例是 **云烟成雨 0–100s**，路径换成目标歌曲即可。

---

## ⓿ 前置：拿到 vocal.wav 和 mix.wav

```bash
# 默认流程（segprompt 段落级合成，2026-08 起），成片干声 = vocal.wav
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac \
    --name 云烟成雨 --seed 1234

# 旧整曲单 prompt 流程（--svc-prompt ext，输出 vocal_ext.wav）
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac \
    --name 云烟成雨 --svc-prompt ext \
    --stages sep,ref-sep,f0-target,select,ext,svc,eval --seed 1234

# 验收后再混伴奏（mix 自动用当前 --svc-prompt 对应的干声）
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac \
    --name 云烟成雨 --stages mix --acc-gain 0.8
```

产物都在 `data/songs/<name>/`：

| 文件 | 用途 |
|---|---|
| `vocal.wav` | **默认干声（segprompt 段落级合成）**。整曲按静音切段，每段配音域匹配的唱歌参考 + 前后 2s 冗余。亲亲实测 vs 整曲单 prompt：CER −2.1 点、SingMOS +0.106、drift −0.007，SIM 持平 |
| `vocal_ext.wav` | 旧整曲单 prompt 干声（`--svc-prompt ext`），对真实参考的绝对相似度与 segprompt 持平 |
| `segprompt_meta.json` | segprompt 每段切分窗口（t0/t1/ext_t0/ext_t1）、med F0、参考候选索引 |
| `mix.wav` | 人声 + 伴奏。**成片 `--audio` 用它**，否则 MV 没有伴奏 |
| `eval.txt` | 音色相似度 + SingMOS 自然度 + 漏音检测，先看这个决定要不要重跑 |

**为什么默认走 segprompt 段落级合成**：全曲共用一条 prompt 对低音主歌是「8 个半音之外的
说话素材」（med~331Hz vs 段 med~208Hz），段首/段尾紧贴短静音时冷启动音质差。segprompt 给
每段配一条音域匹配的唱歌参考（预算 45s，与整曲 prompt_ext 对齐），前后各 +2s 冗余把邻接
音频喂给模型当上下文。详见 `AGENTS.md`「段落级 SVC」章节。

**两者时长严格一致**（同源，实测都是 240.782s），所以同一时间窗切出来的片段天然对齐 —— 
口型在干声上算、成片用混音出，是安全的。这一点已量化验证过：SoulX-SVC 保留源时间轴，
`vocal.wav` 与 `sep2/lead.wav` 逐 10s 窗互相关最大位移只有 10ms（0.3 帧 @30fps）。

**Mix 阶段自动选择产物**：用当前 `--svc-prompt` 对应的干声（默认 `vocal.wav`），不存在时
回退到 `vocal_cascade.wav`、再回退 `vocal.wav`。也可以用 `--mix-vocal` 手动指定（支持绝对路径
或相对于 `data/songs/<name>/` 的文件名）：

```bash
# 强制使用旧流程的 ext 干声混音
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac \
    --name 云烟成雨 --stages mix --mix-vocal vocal_ext.wav

# 或使用绝对路径
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac \
    --name 云烟成雨 --stages mix --mix-vocal /path/to/custom_vocal.wav
```

`--acc-gain` 是伴奏音量（默认 0.8）；人声侧是 `--vocal-gain`（默认 1.0）。

**mix 阶段默认做两件事，一般不用管，但要知道它们在。** ① 人声增益**跟随伴奏包络**：歌曲高潮段
伴奏通常比主歌响 10-15dB（编曲加厚），而 SVC 人声全程电平平稳，固定配比必然让后半段人声被压住。
② **软限幅**到 −0.5dBFS，因为抬完人声的原始和会超过 0dBFS，硬削会把增益换成失真。

| 想干什么 | 加什么 |
|---|---|
| 人声再亮一点 | `--match-margin 2`（目标：人声比伴奏高 2dB，默认 0） |
| 觉得抬得太狠 | `--match-max-boost 4`（默认 8dB 上限） |
| 完全回到旧的固定增益 | `--no-match-acc --limit-db 0` |
| 成片要单声道 | `--mix-mono`（默认保留伴奏立体声） |

验收看 mix 阶段自己打的那两行日志 —— `增益 +0.0 ~ +7.7 dB (均值 +2.4)` 和
`限幅: 原始峰值 +3.29 dBFS, 最大衰减 3.79 dB`。均值超过 +5dB 或限幅衰减超过 6dB，说明这首歌的
人声与伴奏本来就差太多，该回头调 `--acc-gain` 而不是让跟随去硬扛。

---

> **素材目录约定（2026-08 起）**：一首歌的全部 MV 中间素材**和成片**——前奏口白 txt、cues、
> beats、align/audio 主轨、混音、口型、预览片、最终 mp4 等——一律放在 **`data/songs/<歌名>/mv/`**
> 下。`mv.py` 只从那里读素材、成片也写回那里；某首歌"是否已备料"= 该目录里
> `cues_intro.json`、`audio_intro.wav`、`beats.json`、`speech.wav` 是否都就位（`mv.py`
> 会当场检查并提示缺什么）。别再往 `/tmp`、`.tmp/` 或 `data/video/` 放渲染产物。

## ① 选窗口

扫 1s 步长 RMS，找有声秒数最多的连续窗口。整曲出片可跳过这步。

```bash
GPT-SoVITS/.venv/bin/python - <<'EOF'
import numpy as np, soundfile as sf
y, sr = sf.read("data/songs/云烟成雨/vocal.wav")
hop = sr; n = len(y)//hop
rms = np.array([np.sqrt(np.mean(y[i*hop:(i+1)*hop]**2)) for i in range(n)])
win = 100
best = max(range(max(1, n - win)), key=lambda i: (rms[i:i+win] > 0.05).sum())
print("最佳 %ds 窗口: t=%ds-%ds 有声 %d/%d" % (win, best, best+win, (rms[best:best+win] > 0.05).sum(), win))
EOF
```

## ② 切片（同一窗口，两个规格）

| 文件 | 规格 | 来源 | 用途 |
|---|---|---|---|
| `align.wav` | **16k mono** | `vocal.wav` | make_cues 的硬性要求 |
| `audio.wav` | 44.1k stereo | **`mix.wav`** | 渲染 mux |

```bash
D=data/songs/云烟成雨/mv; mkdir -p "$D"
ffmpeg -v error -ss 0 -t 100 -i data/songs/云烟成雨/vocal.wav -ar 16000 -ac 1 $D/align.wav -y
ffmpeg -v error -ss 0 -t 100 -i data/songs/云烟成雨/mix.wav                  $D/audio.wav -y
```

两个文件必须**同起点同时长**，否则口型整体偏移。切完核一下：

```bash
for f in align audio; do printf "%-6s " $f
  ffprobe -v error -show_entries format=duration:stream=sample_rate,channels -of csv=p=0 $D/$f.wav | tr '\n' ' '; echo
done   # 期望 16000,1 100.0 / 44100,2 100.0
```

---

## ③ cues（口型时间轴）

```bash
GPT-SoVITS/.venv/bin/python listen_lipsync/make_cues.py \
    --wav /tmp/opencode/云烟成雨_100s/align.wav \
    --out /tmp/opencode/云烟成雨_100s/cues.json \
    --gate-db -40
```

### 优先用真歌词做强制对齐，不要用 ASR 转写

不传 `--text` 会走 ASR 自动转写（paraformer-large）。**但只要 ⑦ 的歌词缓存已经存在，就一定要
用 `--text`** —— ASR 在 SVC 转换后的人声上漏字很严重，实测全曲 ASR 只认出 103 字（还听错若干），
喂真歌词能对齐 **347 字**：

```bash
D=data/songs/云烟成雨/mv
# 从缓存的增强 LRC 抽纯文本（只读本地，不联网）
GPT-SoVITS/.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "listen_lipsync")
from get_lyrics import read_enhanced
lines, _ = read_enhanced("data/songs/云烟成雨.lrc")
open("$D/lyrics.txt", "w", encoding="utf-8").write(
    "\n".join(l["text"] for l in lines) + "\n")
EOF

GPT-SoVITS/.venv/bin/python listen_lipsync/make_cues.py \
    --wav $D/align.wav --text $D/lyrics.txt --out $D/cues.json --gate-db -40
```

这一步是**免费的**（歌词本来就为字幕抓过），全曲 347 字对齐只要 14.5s。所以顺序上建议先跑
⑦ 的 `get_lyrics.py`，再回来做 ③。

**全曲口型直接用 `--lrc`（读增强 LRC 行锚点），不要走 ASR 强制对齐。** 见
AGENTS.md「Anime2.5DRig 离线渲染」节：fa-zh 长音频系统性漂移、且**重复乐句会塌缩**（第二遍
「我们在怀念的演唱会」实测把 我/们/在/怀 压成同一时刻、后 10 字全 20ms 步进，下游「时间轴
有序」检查直接报警）。KRC 行级时间是人工校对（与能量跃变 <0.2s）、行内按字数均摊到「行内首个
≥0.35s 长静音的起点」，拖腔/闭嘴交给能量门控 —— 无模型、零显存、秒出：

```bash
GPT-SoVITS/.venv/bin/python listen_lipsync/make_cues.py \
    --wav $D/align.wav --lrc data/songs/情歌.lrc --out $D/cues.json --gate-db -40
```

`--lrc` 优先于 `--text`；只有行级 LRC（无逐字）时才退回 `--text`，且跑完必须过下面「产出检查」
的**全部**判据（含奇异值扫描，光看「唱着却闭嘴 %」查不出塌缩）。喂 `--lrc` 前删掉 KRC 头行
（「歌手 - 歌名」）：它在无歌的 intro 里会被长静音判据截成 ~0.1s、t=0 嘴微闪；字幕侧**反过来
要保留**（KTV 片头歌名卡）。

**人声被手工删段（剪掉唱得不好的部分）后必须重锚受影响的行。** 删段是「掏成静音、时间轴不
位移」，所以只要行起点落进洞，`first_long_silence` 当头 0.35s 无声就把整行塌成 ~100ms 微闪。
检测：对比 pre-edit 基准（sing.py 的 `vocal_baseline.wav`）的 1s voiced 网格找洞；修复 = 把
受影响行的起点挪到洞后**首个发声帧**（情歌两处：25.73→27.08、140.61→141.46），把改好的 LRC
存 `$D/lyrics.lrc` 再喂 `--lrc`，其余行的锚点照旧不动。

### 只用 A/B/C/E 四种口型

九宫格素材里 D(大开)、F(撅唇)、G(唇齿)、H(长L) 在本立绘上很难用对 —— D 一出现就明显违和。
所以 `make_cues.py` 把 40 个韵母全部映射到 **C(开口) / E(微圆) / B(微开)**，辅音只保留
**A(闭唇)** 给 b/p/m。归并沿 Rhubarb 官方的过渡链回退一步，不是随意合并：

| 原 | → | 依据 |
|---|---|---|
| D 大开 | **C** | 官方 C 是 `B→D` 的中间形 |
| F 撅唇 | **E** | 官方 E 是 `C/D→F` 的中间形 |
| G 唇齿 | **B** | 唇齿收进微开 |
| H 长L | **C** | 官方 H 开口介于 C 与 D 之间 |

依据是动画业界常规：Bristow《5 things that can go wrong with animated lip sync》指出口型过大
是新手错误第一位，并说 anime 常只用 open/shut 两型也没人抱怨。`push()` 里有 `FOLD` 兜底，
无论上游哪条分支产出什么都不会漏出四型之外。

### 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gate-db` | −40 | 静音门限（相对峰值）。闭嘴只发生在真正无声处 |
| `--max-hold-sil` | 0.35 | hold 到下一字时尾部允许夹带的连续静音上限 |
| `--melisma-sec` | 0.45 | 超过此时长按「枣核型」拆成峰值+衰减两段 |
| `--melodic-accent` | 0.6 | VOCAL 的 Ma：短元音不给满开口 |
| `--closure-sec` | 0.06 | b/p/m 字首闭合时长 |
| `--silence-gap` | 0.30 | **仅在 `--no-energy-gate` 下生效**，唱歌上不成立，别用 |

**`--gate-db` 取代了旧的 `--silence-gap` 判据。** 按字间隔时长猜停顿在唱歌上是错的：实测 9 个
被判静音的段落有 8 个其实在唱（拖腔的字间隔普遍 >0.3s），首版 X 占 78%、嘴基本不动。
改成能量门控后「唱着却闭嘴」从 28.2% 降到 1.8%。

### 产出检查

```bash
GPT-SoVITS/.venv/bin/python - <<'EOF'
import json, numpy as np, soundfile as sf
D="/tmp/opencode/云烟成雨_100s"
d=json.load(open(f"{D}/cues.json")); cs=d["mouthCues"]
print("口型集合:", sorted(set(c["value"] for c in cs)), " cue 数:", len(cs))
assert not set(c["value"] for c in cs) - set("ABCEX"), "有四型之外的口型"
assert not [1 for a,b in zip(cs,cs[1:]) if b["start"] < a["end"]-1e-6], "时间轴倒序"
y,sr=sf.read(f"{D}/align.wav"); hop=int(sr*0.01); n=len(y)//hop
rms=np.sqrt(np.mean((y[:n*hop].reshape(-1,hop))**2,axis=1)); gate=rms.max()*10**(-40/20)
voiced=rms>gate; open_=np.zeros(n,bool)
for c in cs:
    if c["value"]!="X": open_[int(c["start"]/.01):min(n,int(c["end"]/.01))]=True
print("唱着却闭嘴 %.1f%%  (应 <5%%)" % (100*(voiced&~open_).sum()/voiced.sum()))
# 只看张嘴的 cue：尾奏的 X 本来就可以很长（全曲版尾部静音 20s），一起统计会永远误报
print("最长张嘴 cue %.2fs  (>7s 要查是否误判拖腔)" % max(c["end"]-c["start"] for c in cs if c["value"]!="X"))
tail=[c for c in cs if c["value"]!="X"][-1]
print("最后一个张嘴 cue 结束于 %.2fs / 全长 %.2fs" % (tail["end"], d["metadata"]["duration"]))
EOF
```

**「最后一个张嘴 cue」必须落在真正唱完处，不能顶到文件末尾。** 全曲版这里踩过一次：歌词
220.6s 唱完、音频却有 240.8s，尾部 20.2s 是数字静音（−175dB），而嘴张着 `C` 型僵了 20 秒。
根因是 `make_cues.py` 尾部分支用的是 `is_voiced()` 的**单帧**判据，淡出时的孤立过限帧就让它
判成「还在唱」—— 与 `--max-hold-sil` 要解决的是同一个坑，只是换了位置复发。已改为复用
`last_voiced_before()`（「持续 80ms 有声」）。修复后尾段实测 99 帧张嘴 **0%**。

## ④ 口型素材（一次性准备）

3×3 九宫格 → 9 张带 alpha 的 sprite：

```bash
GPT-SoVITS/.venv/bin/python listen_lipsync/split_visemes.py \
    data/pics/mouth_9.png -o data/pics/visemes
```

按**连通域**切分（自动剔除铺满全图的边框域），产出 `{X,A,B,C,D,E,F,G,H}.png` + `meta.json`。
`meta.json` 里的 `maxContentW/canvasContentW` 供渲染器算 `contentRatio` 修正 padding，不可删。

人眼核对九个口型：

```bash
a25rig/mouth_sheet.sh --psd data/pics/figure.psd \
    --visemes data/pics/visemes --out data/songs/云烟成雨/mv/mouth_sheet --zoom 4
```

输出 9 张单图 + `_sheet.png` 总览。它会冻结 sway/blink/rand/idle，保证画面里只有嘴不同。

## ⑤ beats（节拍网格）

```bash
GPT-SoVITS/.venv/bin/python singing/beat_grid.py \
    data/songs/云烟成雨.flac -o /tmp/opencode/云烟成雨_100s/beats.json
```

- 吃**全曲混音**（.flac），不是分离人声 —— 节拍检测需要打击乐线索
- 全曲网格（`t0 + k·barLen`），**与渲染窗口无关，跑一次各窗口通用**（可跨窗口复制）
- 看 `gridInliers` / `residP90` 判断可信度（示例 68/74 inliers、p90 19.5ms）
- GPU 异常时加 `--device cpu`

---

## ⑥ 渲染

入口是 `python3 mv.py`（分阶段流水线，见文件头部文档；旧 `render.sh` 已并入它）。素材从
`data/songs/<歌名>/mv/` 读（见上方素材目录约定），每阶段按"产物齐全且不比输入旧"自动复用
（`--force` 重跑）。**日常出片直接跑**：

```bash
python3 mv.py --song 云烟成雨            # 默认全流程：slice→lyrics→cues→beats→visemes→render→subs
python3 mv.py --song 云烟成雨 --stages render   # 只重渲（复用已有素材）
python3 mv.py --song 情歌 --with-intro --tts-ref <参考音频>   # 带前奏口白（见 ⑥½）
```

> **⚠️ 沙箱环境跑不了渲染**：`render` 阶段要起 headless Chrome + Mesa D3D12 拿 GPU（ANGLE/GL），
> 在受限沙箱里 `page.goto` 会 30s Navigation timeout（不是脚本 bug，见 AGENTS.md）。需要在能
> 访问 GPU 的正常机器上跑，或给该命令提权执行。

下面两段是渲染器原始参数，仅作底层参考（默认风格下 `mv.py --stages render` 已经拼好这套，
换 `--bg`/`--char-scale` 等风格时才需要直接调 `a25rig/render.sh`）。注意那串环境变量
**必须显式点名 NVIDIA 适配器**，否则会选到 AMD 核显（比软渲染更慢，实测 17.3 → 38.3 fps）；
Vulkan 必然回落 SwiftShader，所以走 ANGLE/GL。

基础版：

```bash
D=data/songs/云烟成雨/mv
MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
A25_GL_ARGS="--use-gl=angle --use-angle=gl --ignore-gpu-blocklist" \
a25rig/render.sh --psd data/pics/figure.psd \
    --cues  $D/cues.json \
    --sway  $D/beats.json \
    --audio $D/audio.wav \
    --visemes data/pics/visemes \
    --video $D/云烟成雨_100s.mp4 \
    --fps 30 --size 16:9 --bg data/pics/bg.jpg
```

雨天版（玻璃水珠 + 落雨丝 + 角色放大右移）—— 注意**去掉 `--bg`**：

```bash
D=data/songs/云烟成雨/mv
MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
A25_GL_ARGS="--use-gl=angle --use-angle=gl --ignore-gpu-blocklist" \
setsid a25rig/render.sh --psd data/pics/figure.psd \
    --cues  $D/cues.json \
    --sway  $D/beats.json \
    --audio $D/audio.wav \
    --visemes data/pics/visemes \
    --video $D/云烟成雨_100s.mp4 \
    --fps 30 --size 16:9 \
    --char-scale 1.15 --char-x 0.62 --char-y 0.56 \
    --rain-glass data/pics/bg.jpg --streaks
```

100s / 1080p / 带玻璃与雨丝实测 **26.6 fps，约 113s 出片**。

### 参数速查

| 参数 | 说明 |
|---|---|
| `--psd` | 立绘（`data/pics/figure.psd`） |
| `--cues` | 口型时间轴（③ 产物）。**不传则嘴由随机生成器驱动，与歌声无关** |
| `--visemes <dir>` | 口型 sprite 目录（④ 产物）。不传则退回单图形变，口型辨识度低 |
| `--rest-form` | 无 cue 处的嘴角上扬，默认 **0.35**（闭嘴微笑）；0=平嘴，0.9=rig 的 smile 预设 |
| `--sway` | 节拍网格（⑤ 产物）；配 `--sway-beats/-body/-roll/-phase` |
| `--audio` | 用 **mix.wav 的切片**；必须与 cues 同窗口 |
| `--size` | `16:9` 或 `WxH`；**必须同时给背景**，否则 0 帧 |
| `--bg` | 纯色/图片背景；`--bg-dim` 压暗、`--bg-blur` 景深 |
| `--rain-glass <img>` | 玻璃水珠（角色**后**）；自带背景，**与 `--bg` 冲突，别同传** |
| `--snow` | 雪花飘落（角色前/后分层）；`--snow-density` 默认 45/MP、`--snow-size` 5.5px@1080p |
| `--glass-blur` | 玻璃模糊，默认 **0**（水痕才看得出变清晰；旧默认 3 时水痕只有周围 0.5×） |
| `--streaks` | 落雨丝（角色**前**）；`--streak-density` 默认 2.8/MP，参考片实测 10/MP |
| `--char-scale/-x/-y` | 缩放 / 位置（**x/y 是 0..1 比例**） |
| `--fps/--seed` | 帧率 / 随机种子（同种子同参数 → 字节级可复现） |

时长由 cues 的 `metadata.duration` 决定，音频由 ffmpeg `-shortest` 截齐。
更多旗标与物理推导见 `a25rig/RENDER.md`。

---

## ⑥½ 前奏口白（可复用的说话语音开头）

成片前，在**器乐前奏**里用角色音色说一段话。整条链路是 `listen_lipsync/intro_speech.py`
一条命令（GPT-SoVITS venv），口白 txt / cues / beats / 混音等产物全落到该歌的
`data/songs/<歌名>/mv/` 下（见上方素材目录约定）：

```bash
# 推荐：直接走 mv.py —— intro 阶段自动调 intro_speech.py（口白生成→口型合并→混音），
# 成片用口白版 cues/audio，命名 <歌名>_intro.mp4，渲完默认接着烧字幕：
python3 mv.py --song 情歌 --with-intro \
    --tts-ref data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav \
    --tts-speed 0.9 \
    --first-lyric 27.08 --offset 1.5 --margin 0.6 --speech-gain-db 4.0
# 成片:            data/songs/情歌/mv/情歌_intro.mp4
# 成片（带字幕）:  data/songs/情歌/mv/情歌_intro_lyrics.mp4（--no-subs 跳过烧字）
# 改了口白文本后重生成：在上一命令加 --stages intro --force（新文本才会真的被念出来）
# 只重渲/重烧字（口白素材已在）：python3 mv.py --song 情歌 --with-intro --stages render,subs
#   （此时可省略 --tts-ref：仅 intro 阶段真跑时才需要）

# 底层手工命令（等价 mv.py 的 intro 阶段；需要单独重算口白/校验时用）：
GPT-SoVITS/.venv/bin/python listen_lipsync/intro_speech.py \
    --text data/songs/情歌/mv/intro.txt \
    --song-cues data/songs/情歌/mv/cues.json \
    --base-audio data/songs/情歌/mv/audio.wav \
    --out-dir data/songs/情歌/mv/ \
    --tts gpt-ft \
    --speed 0.9 \
    --ref-audio data/voices/girlish_sliced/ref-vocal.wav_0000064320_0000268160.wav \
    --first-lyric 27.08 --offset 1.5 --margin 0.6 --speech-gain-db 4.0
#   之后 python3 mv.py --song 情歌 --with-intro --stages render,subs 出片（复用已生成的口白素材）
```

要点：

* **选模：默认 `gpt-ft` = girlish GPT `e30` + `talksing_s2mix50` SoVITS `e12`**（权重在
  `listen_lipsync/intro_speech.py` 的 `GPT_WEIGHTS` 里定义）。SoVITS 于 2026-08 换成
  「说话 50% + 唱歌 50% 混合数据、从 v2Pro 底模从头训练」的 S2，判据是**吐字与清晰度**
  （长文本 436 字、GPT 固定、`--seed 1234` 配对）：CER **0.0505 → 0.0367**（超噪声底 0.0092，
  替换错误 9→5）、4-8kHz 塌陷 **−4.27dB → −0.02dB**（LTAS L1 2.81 → 1.83）。
  跨模型的老 A/B（`gpt-ft` 0.8734 > `gpt-zs` 0.8406 > VoxCPM2 0.7807）仍成立，产物在 `mv/ab/`。
* **别拿 SIM 当选型主判据 —— 它已经饱和。** 同一锚点下真人**自己另一场录音**只有
  **0.7538**、不同人地板 0.3567（`singing/eval_sim_ceiling.py` → `data/experiments/sim_ceiling.json`），
  而各臂都在 0.72-0.79，即 SV 模型已分不出合成与真人。换 SoVITS 后 SIM 0.7880 → 0.7452 看着是
  退步，实则两者都在天花板附近，那 0.04 只占可判别区间（0.397）的 10%。**判据优先级：吐字 CER >
  语速语调自然 > SIM。**
* **语气平是文本问题，不是 checkpoint 问题**：六个臂 F0 std 全在 2.21-2.37（真人 2.853）、
  停顿数 24-27（真人 66），臂间差异远小于噪声底。停顿完全由标点决定 —— 见下面「断句靠标点」。
* **语速 `--tts-speed`（GPT-SoVITS 原生，非后期变速）**：走 VITS 时长模型把 phoneme 时长缩放，
  **音色/音高不变**，比 ffmpeg atempo 自然。范围 0.6-1.65（WebUI 滑杆同款），<1 更慢；实测
  0.9 秒把 34 字口白从 ~6s 拉到 ~10s。仅 gpt-ft/gpt-zs 生效（voxcpm2 不走这条）。
* **断句靠标点，`～` 不可靠**：`。！？` 是成句级强停顿（默认「按中文句号。切」把长文本在 `。`
  处拆句）；`，、；：` 是句内短停；**`～`/`~` 会被替换成省略号 `…`，模型基本不给停顿**
  （实测「真好看～如果」几乎无停顿），想要停顿就用 `。`（平静）/`！`（赞叹上扬）/`，`（轻过渡）。
  反之想完全无停顿就连写不加标点。语速+断句改完用 `--stages intro --force` 重生成试听。
* **硬约束（fail-closed）**：`dur ≤ first_lyric − offset − margin`，**口白绝不允许压到唱段**；
  脚本打印 `offset / end / first_lyric / buffer`。生成后先目检 `mv/preview_intro.mp4`（或先渲
  前 ~11s 低清预览）确认口型与语音对得上再全片。
* **口型**：对生成的口白单独跑 make_cues（`--melodic-accent 0`，说话型逐字清晰），再并回前奏
  区 `X(0,offset) + 口白段 + X(尾,first_lyric)`；唱歌段 cues 原样不动（合并后逐样本核对过）。
* **混音**：口白按 `--speech-gain-db`（默认 +4dB）叠进 audio 的 `[offset, end)`，只动这一小段、
  之后逐样本不变；同时写 16k mono `align_intro.wav`，让 cues 的 `soundFile` 与真实时间轴自洽。

---

## ⑦ KTV 字幕（可选）

**`mv.py` 的 `subs` 阶段已内置这一步**（渲完默认自动烧字 → `mv/<歌名>[_intro]_lyrics.mp4`；
`--no-subs` 关闭）：复用 `mv/lyrics.ass`（没有则用 `make_ass.py` 现建），烧前 `fc-match SimHei`
校验字体配置，防止"烧了但字幕静默消失"。下面是底层手工命令，仅在单独重烧字幕时用。

烧在**已渲染好的视频**上，不参与 a25rig 渲染。这样改字幕只要重烧几秒，不用重渲整片。

```bash
# 1) 抓歌词 → 缓存成歌曲同名 .lrc（只联网一次，之后全走本地）
GPT-SoVITS/.venv/bin/python listen_lipsync/get_lyrics.py data/songs/云烟成雨.flac

# 2) 生成 ASS（--start/--end 必须与视频窗口一致）
D=data/songs/云烟成雨/mv
GPT-SoVITS/.venv/bin/python listen_lipsync/make_ass.py data/songs/云烟成雨.lrc \
    -o $D/lyrics.ass \
    --size 1920x1080 --start 0 --end 100 --pos bc

# 3) 烧进视频（字体走 FONTCONFIG_FILE，不往系统装字体）
export FONTCONFIG_FILE=.tmp/fontconf/fonts.conf
ffmpeg -v error -y -i $D/云烟成雨_100s.mp4 \
    -vf "ass=$D/lyrics.ass" \
    -c:v libx264 -crf 18 -preset medium -c:a copy \
    $D/云烟成雨_100s_lyrics.mp4
```

画面上永远两行：**上行**是正在唱的句子（逐字扇光，唱过的字变色），**下行**是下一句预览
（更小更淡）。唱完自动上滚。1080p 烧字实测：22s 片段 5.5s、全曲 241s 约 90s。

整曲出片就把窗口开满（`--start 0 --end <全长>`，超出全长无害）：

```bash
GPT-SoVITS/.venv/bin/python listen_lipsync/make_ass.py data/songs/云烟成雨.lrc \
    -o /tmp/opencode/云烟成雨_full/lyrics.ass --size 1920x1080 --start 0 --end 241 --pos bc
# 全曲 42 行 → 83 个字幕事件（每行一个主行 + 一个预览行，末行无预览）
```

烧字**不要覆盖无字幕版**：留着它，下次调字幕只需重烧几秒，不必重渲 4.5 分钟。

### 歌词来源与缓存

`get_lyrics.py` 从**音频标签**（`title`/`artist`/`duration`）出发，三级回落：

| 优先级 | 源 | 时间精度 | 鉴权 |
|---|---|---|---|
| 1 | 酷狗 KRC | **逐字** | 无 |
| 2 | LRCLIB → 网易云 LRC | 行级 | 无 |
| 3 | 写手工填写模板 | 乐句锚点 | — |

命中即写 `data/songs/<name>.lrc`，**再次运行直接读本地**（`--force` 才重抓，`--offline` 强制
只读缓存）。10 首实测：4 首逐字、5 首行级、1 首要手填，缓存总共 30KB。

格式是标准**增强 LRC**（`[mm:ss.xx]<mm:ss.xx>字`），不是自造格式 —— 任何编辑器可改、任何
播放器可验，只有行级时也能用。行首是整行起点，每个 `<>` 是该字起点，行尾补一个「唱完」时刻
给扇光收尾。手填模板顶部带格式说明，下面是按人声能量自动切的乐句锚点，只需填字。

**别用网易云 LRC 自带的时间做扇光。** 它是给 KTV 播放器留提示用的，系统性提前（实测中位
+0.78s，最坏 +3.76s，17 行里 4 行落在静音处）。逐字源用酷狗；只有行级时，要么让
`make_ass.py` 按字数均分（`--even`，默认在无字级数据时自动生效），要么先跑
`make_cues.py --text <歌词>` 做强制对齐拿真字级。

**逐字源同时解决了「行内长静音」。** 强制对齐器不建模休止，会把间奏吞进行内（实测一行
11.4s 里含 3.6s 静音，字幕僵在屏上）；酷狗 KRC 的行边界把休止留在行外，同窗口 17 行**零**
行内长静音，最长行 7.95s。

### 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--pos` | `tr` | `tl/tc/tr/bl/bc/br`，常规视频用 **`bc`**（下方居中） |
| `--start/--end` | 0 / ∞ | **必须与视频窗口一致**，脚本会把时间轴归零到 `--start` |
| `--alpha` | `0x40` | 两行整体透明度，0=不透明、255=全透（减少遮挡角色） |
| `--next-alpha` | `0x40` | 预览行在 `--alpha` 之上再淡多少 |
| `--font` / `--font-size` | SimHei / 54 | 字体须 fontconfig 可见 |
| `--next-scale` | 0.72 | 预览行字号比例 |
| `--color-sung` | `#3BC8FF` | 已唱（扇光）色；`--color-pending` 未唱色 |
| `--margin-x/-y` | 48 / 40 | 边距（px） |
| `--lead` | 1.0 | 提前多少秒亮出当前行（未唱色） |
| `--no-preview` | — | 只显示当前行 |
| `--even` | — | 忽略字级时间，按字数均分 |

### 中文字体

**系统里没有中文字体**（`fc-list :lang=zh` 为 0，`fc-match` 回落到不含汉字的 DejaVu），
直接烧会得到整屏豆腐块。不装系统字体的做法是让 fontconfig 指向 Windows 字体目录：

```bash
mkdir -p /tmp/opencode/fonttest/fc && cat > /tmp/opencode/fonttest/fc/fonts.conf <<'EOF'
<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>/mnt/c/Windows/Fonts</dir>
  <cachedir>/tmp/opencode/fonttest/fc/cache</cachedir>
  <include ignore_missing="yes">/etc/fonts/conf.d</include>
</fontconfig>
EOF
export FONTCONFIG_FILE=/tmp/opencode/fonttest/fc/fonts.conf
fc-match SimHei     # 期望 simhei.ttf: "SimHei" "Regular"
```

用 `simhei.ttf` 而不是 `msyh.ttc`：后者是字体集合，还得指定 face。

**fonts.conf 别放 `/tmp`，放工作区（如 `.tmp/fontconf/fonts.conf`，cachedir 也放工作区）。**
本工作区的命令沙箱会在各次调用之间清空 `/tmp`：上一份生成该配置的命令跑过之后、真正烧录时文件
已不存在，结果是 ffmpeg **exit 0**、stderr 只有 4 行
`Fontconfig error: Cannot load default config file: No such file: (null)`，画面**一个字都没有**，
看起来完全正常。凡是跨命令要用的临时文件（fonts.conf、中间产物）一律放仓库内，烧录/抽帧这种
一次性分析在单条命令里做完。

### 验收

字幕像素要用**与无字幕原片做差分**来隔离 —— 直接测颜色会把背景算进去（雨玻璃偏蓝，曾把
背景当成「已唱」，得出「扇光 0.5s 冲到 99% 然后僵住」的假结论）。

抽帧时刻要取**确实在唱**的位置 —— intro 是纯伴奏、屏上本来就没字（本例首句 18.0s 才起唱），
在那里抽帧会得出「字幕没生效」的假结论。时刻直接从 `.lrc` 读：

```bash
cd /tmp/opencode && rm -rf lyrchk && mkdir lyrchk
SUB=/home/rookie/voice/data/songs/云烟成雨/mv/云烟成雨_100s_lyrics.mp4
CLN=/home/rookie/voice/data/songs/云烟成雨/mv/云烟成雨_100s.mp4
# 取第 1 句起唱后 +1/+4/+7s（首句起点由 .lrc 第一行时间决定，这里是 18s）
for t in 19 22 25; do
  ffmpeg -v error -y -ss $t -i $SUB -frames:v 1 lyrchk/s$t.png
  ffmpeg -v error -y -ss $t -i $CLN -frames:v 1 lyrchk/c$t.png
done
/home/rookie/voice/GPT-SoVITS/.venv/bin/python - <<'EOF'
from PIL import Image, ImageChops
for t in (19, 22, 25):
    a=Image.open(f"lyrchk/s{t}.png").convert("RGB"); b=Image.open(f"lyrchk/c{t}.png").convert("RGB")
    dp=ImageChops.difference(a,b).convert("L").load(); w,h=a.size
    pts=[(x,y) for y in range(0,h,2) for x in range(0,w,4) if dp[x,y]>30]
    if len(pts) < 200:                       # 少量散点是重编码噪声，不是文字
        print(f"t={t}s  差分像素仅 {len(pts)} 个 → 该时刻无字幕"); continue
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    print(f"t={t}s  y={min(ys)}..{max(ys)}  x={min(xs)}..{max(xs)}  "
          f"中心偏移 {abs((min(xs)+max(xs))//2 - w//2)}px  ({len(pts)} 采样点)")
EOF
```

`--pos bc` 下实测：主行 y928-982、预览行 y1006-1036（中间有干净空隙，证明**上行是主行**），
中心偏移 2px。透明度合格的判据是同一坐标「带字幕」与「原片」的 RGB 明显不同但非纯色覆盖。

**重编码与原片容易错 1 帧，别直接对单帧抽差分。** 动画面（雨丝/晃动/口型）错 1 帧会让全帧
差异爆表，字幕带 100% 非零全是假象。可靠做法：两片各解码 1s、逐帧对齐（±3 帧内取整帧
mean absdiff 最小），再比「字幕带（y≈900-1040）mean absdiff」与「控制带（顶部/中部）」比值 ——
情歌实测字幕带 7.45 vs 控制带 0.46/0.78（**9.5×**），重编码噪声本身只有 ~0.5-0.8。完全不依赖
肉眼/原片的兜底是**黑底 ink test**：`lavfi` 黑底 + ass 滤镜出一帧，数字幕带亮像素数与两行 bbox
（居中双行 ⇒ 文字排对了）。

---

## ⑧ 验收

```bash
V=data/songs/云烟成雨/mv/云烟成雨_100s.mp4
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,channels \
    -of default=nw=1 $V     # 期望 100.0 / h264 1920x1080 30/1 / aac 2ch
```

**没人声时嘴必须闭合**，这是最容易回归的一条。探针法（`char-scale 1.15 / x 0.62 / y 0.56` 下）：

```bash
cd /tmp/opencode && rm -rf chk && mkdir chk
ffmpeg -v error -i /home/rookie/voice/data/songs/云烟成雨/mv/云烟成雨_100s.mp4 -vf fps=5 -q:v 2 chk/f_%04d.jpg
/home/rookie/voice/GPT-SoVITS/.venv/bin/python - <<'EOF'
import numpy as np, glob, json
from PIL import Image
S=1.15*1080/1280.0                                   # char-scale x 输出缩放
mx=int(round(0.62*1920+(678.54-655.53)*S))           # face 锚点 (655.53,383.94) 落在 (x*W,y*H)
my=int(round(0.56*1080+(586.65-383.94)*S))           # 嘴锚点 (678.54,586.65) 相对偏移
r=50; sig=[]
for f in sorted(glob.glob("chk/f_*.jpg")):
    p=np.array(Image.open(f).convert("RGB")).astype(int)[my-r:my+r, mx-r:mx+r]
    rr,gg,bb=p[:,:,0],p[:,:,1],p[:,:,2]
    sig.append(((rr>100)&(rr<200)&(gg<80)&(bb<110)).sum())   # 口腔暗红色块
sig=np.array(sig,float); t=np.arange(len(sig))/5.0
cs=json.load(open("/tmp/opencode/云烟成雨_100s/cues.json"))["mouthCues"]
has=np.zeros(len(t),bool)
for c in cs:
    if c["value"]!="X": has |= (t>=c["start"])&(t<c["end"])
print("有 cue: 张嘴 %.0f%%  面积均值 %.0f" % (100*(sig[has]>20).mean(), sig[has].mean()))
print("无 cue: 张嘴 %.0f%%  面积均值 %.1f   ← 应 ≤2%%" % (100*(sig[~has]>20).mean(), sig[~has].mean()))
print("面积最大 %.0f" % sig.max())
EOF
```

实测合格线（同一套素材/机位）：

| | 100s 版（ASR 转写） | 全曲 241s（真歌词对齐） |
|---|---|---|
| cue 数 / 字数 | 83 / 103 字 | **284 / 347 字** |
| 有 cue 张嘴 | 98%，面积均值 333 | 98%，面积均值 339 |
| **无 cue 张嘴** | 1%（113 帧里 1 帧） | **2%**（324 帧里） |
| 面积最大 | 502 | 516 |
| 尾段（221s 后） | — | 张嘴 **0%**（99 帧） |
| 渲染 | 26.6 fps / 113s | 27.2 fps / **265s**（7223 帧） |
| 烧字 | 5.5s（22s 片段） | ~90s（全曲） |

「无 cue 张嘴率」是这套流程唯一必查的回归项。面积最大值只能横向比同一套素材/机位 ——
它随 `--char-scale` 和探针半径变化，不是绝对阈值；判断 D 是否在用，直接看 ③ 的口型集合。

全曲版还要多查两项（100s 片段看不出来）：

```bash
V=data/songs/云烟成雨/mv/云烟成雨_full_lyrics.mp4
# 1) 帧数 × fps 必须等于时长：7223 / 30 = 240.767s
ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 $V
# 2) 音频不能中途断流 —— 分段 RMS 应全程在 -12..-20dB，不能出现 -99
ffmpeg -v error -i $V -ar 16000 -ac 1 -f wav - 2>/dev/null | GPT-SoVITS/.venv/bin/python -c "
import sys, io, numpy as np, soundfile as sf
y, sr = sf.read(io.BytesIO(sys.stdin.buffer.read()))
for a in range(0, int(len(y)/sr), 40):
    s = y[a*sr:(a+40)*sr]
    print('  %3d-%3ds %+7.1fdB' % (a, min(a+40, int(len(y)/sr)), 20*np.log10(max(float((s**2).mean()**.5), 1e-9))))"
```

`--count_frames` 要遍历整个文件（全曲约 30s），但它是唯一能抓出「时长对、帧却丢了」的检查。

---

## 常见陷阱

按踩到的代价排序，前三条都曾产出「看起来正常但其实是坏的」成片。

- **没有 cue 的区间，嘴会被上游的随机空闲动画驱动。** `index.html` 原本在 cue 未命中时
  落到 `auto.talk`（Anime2.5DRig 的「随便动动嘴」），而 `begin()` 从不关掉它。实测一份 100s
  cues 有 **22.6% 时长完全没有 cue**（intro 19s + 间奏 3.3s 是大段），那些区间嘴全程乱动。
  已修为「有 cues 就以 cues 为权威，空洞即闭嘴 + `--rest-form` 微笑」。**升级 a25rig 后要重打**。
- **`--audio` 传成 vocal.wav 会得到没有伴奏的 MV。** 口型算在干声上、成片用混音，是两个不同
  的文件；`sing.py` 默认不跑 `mix` 阶段，要显式 `--stages mix`。
- **cues 与音频窗口必须严格对齐。** cues 只对生成它的那段音频有效。现成的 cues 文件换窗口即
  失效（曾有 66–86s 的 cues 配 60–84s 的视频，差 6s 即不可用）。
- **`--rain-glass` 与 `--bg` 不能同传。** 前者在 `render.mjs:340` 是独立分支、自己吃图片当
  背景，与 `--bg` 的 `else if` 链并列，同传会各画一遍。
- **`--size` 不带背景会静默产出 0 帧**（合成画布 `_c2` 只在 `setBg()` 里创建）。纯色用 `--bg '#000'`。
- **`--char-x/--char-y` 是 0..1 归一化比例**，传 60 会把角色推出画面且 exit 0。
  `--char-anchor` 只认 `face`，其余值静默退化成画布中心。
- **未知旗标会 exit 2，但旗标的取值没有校验** —— 上一条就是这个坑的下一层。
- **`--gate-db` 的单帧判据会被混响尾巴骗过。** `is_voiced` 是「任意一帧过门限」，够宽松才能
  接住拖腔，但 −40.2dB 擦过 −40 门限就把 3.5s 纯静音判成拖腔，嘴僵在半开 5.6s。
  `--max-hold-sil` 用「最后一段**持续** 80ms 有声」裁掉尾部静音。
- **判断「嘴动没动」不能只看口腔色块。** `auto.talk` 驱动的是单图形变而非换贴图，色块探针
  抓不到。要用嘴部**帧间差分**，并与脸颊/额头对照 —— 呼吸（`auto.idle`）会带动整张脸位移，
  合格的成片里嘴区差分（0.32）应**小于**脸颊（1.25）和额头（1.32）且与它们正相关。
- **不要用 `nohup ... &` 在受超时限制的会话里跑渲染** —— 会话超时会杀掉整个进程组，表现是
  日志停在半路、无 traceback、显存归零，极易误判成卡死或 OOM。用 `setsid`。
- **`--out`（PNG 序列）比 `--video` 慢 2.6×**，且耗时取决于画面复杂度而非像素数。
  只在需要逐帧像素分析时用，用完即删（8 组 24s/1080p 曾占 21GB）。
- 渲染日志第一行的 `404 (Not Found)` 是 favicon，无害（`rig:` 行正常打印即 OK）。
- **字幕：两个同样式事件时间重叠会触发 libass 防碰撞位移。** 表现是换行时上行槽空了、字留在
  预览槽里（看起来像「不滚动」）。`--lead` + `--hold` 超过句间空隙就会重叠，`make_ass.py` 的
  `no_overlap()` 在两句之间取中点切开。
- **字幕：`--start/--end` 必须与视频窗口一致。** ASS 时间轴按 `--start` 归零，配错窗口整体偏移。
  用 `ffmpeg -ss` 剪过的片段，`-ss` 的值就是 `--start`。
- **字幕：验字幕不能直接测颜色，要与无字幕原片做差分。** 与「验嘴动不能用色块探针」同一类错。
  差分后还要设噪声门（`<200` 采样点判为无字幕）：h264 重编码会留个别孤立像素，全曲抽检时
  它把包围盒从 y920-1039 撑成 y246-982，看起来像字幕跑到画面中间去了。判据要看**质量分布**
  而不是 min/max。**且原片与重编码容易错 1 帧**，动画面下直接单帧差分全屏虚报 —— 先对齐
  （解 1s 逐帧取最小整帧差），再看字幕带 vs 控制带的比值（情歌实测 7.45 vs 0.78，9.5×）。
- **字幕：fonts.conf 放 `/tmp` 会静默烧出无字幕版。** 本工作区命令沙箱跨调用清空 `/tmp`，
  烧录时 libass 拿不到字体，ffmpeg **exit 0**、画面没字，看起来完全正常。配置文件放工作区
  （`.tmp/fontconf/fonts.conf`），烧前 `FONTCONFIG_FILE=... fc-match SimHei` 验一下。
- **口型：全曲用 `--lrc`，`--text` 只在无逐字 KRC 时退回。** ASR 强制对齐在重复乐句会塌缩
  （多字同一起点 + 20ms 微字），且长音频系统性漂移（见 AGENTS.md）；KRC 行锚点人工校对无此问题。
- **口型：ASR 转写在 SVC 人声上漏字约 30%**（全曲 347 字只认出 103 字）。⑦ 的歌词缓存存在时
  一定要用 `make_cues.py --text`，这一步免费且顺带让口型变准。
- **口型：尾奏静音会让最后一个口型撑到文件结尾。** 尾部判据必须是「持续有声」而非单帧，
  否则淡出的孤立过限帧就让嘴张着僵 20 秒（已修，见 ③）。全曲版才暴露 —— 100s 窗口正好切在
  歌声中间，没有尾奏。
- **酷狗歌词接口会静默限流**：HTTP 200 + 空候选，长得和「这首歌没歌词」一样。实测连发 6 次得到
  `10/10/10/0/10/10`，间隔 2s 则 6/6 命中。`get_lyrics.py` 已内置限速与重试 —— 不限速时测出的
  命中率全是噪声（曾先后得出 1/10 与 4/10 两个矛盾结果）。

## 相关文档

| 文件 | 内容 |
|---|---|
| `AGENTS.md` | 工作区总览、`sing.py` 全部调优结论、各 SVC/移调引擎 A/B、硬件与内存陷阱 |
| `a25rig/RENDER.md` | 渲染器全部旗标、雨层物理推导、确定性验收、实测性能 |

脚本入口一览：

| 脚本 | 阶段 | venv |
|---|---|---|
| `sing.py` | ⓿ 产声 | 系统 python3（内部各阶段自选 venv） |
| **`mv.py`** | **② MV 流水线入口**（slice/lyrics/cues/beats/visemes/intro/render/subs） | 系统 python3（内部调各脚本自带 venv） |
| `listen_lipsync/make_cues.py` | ③ 口型时间轴 | GPT-SoVITS |
| `listen_lipsync/split_visemes.py` | ④ 口型 sprite | GPT-SoVITS |
| `singing/beat_grid.py` | ⑤ 节拍网格 | GPT-SoVITS |
| `a25rig/render.sh` | ⑥ 渲染（mv.py render 阶段调用） | node（自带 `driver/node_modules`） |
| `listen_lipsync/get_lyrics.py` | ⑦ 抓词缓存 | GPT-SoVITS |
| `listen_lipsync/make_ass.py` | ⑦ 生成 ASS | GPT-SoVITS（纯 stdlib） |
| `listen_lipsync/intro_speech.py` | ⑥½ 前奏口白（mv.py intro 阶段调用） | GPT-SoVITS |
