# voice workspace — 从歌曲到 MV 全流程

两条独立的流水线，前者产声、后者产画：

```
① sing.py                                       ② MV 渲染
歌曲.flac + 参考语音                              vocal.wav / mix.wav
  └─ sep → ref-sep → f0-target → select            ├─ 选窗口   RMS 扫描
     → ext → svc → eval → mix                      ├─ 切片     align/audio
        ├─ vocal.wav  SVC 干声（口型用这个）        ├─ cues     make_cues.py
        └─ mix.wav    人声+伴奏（成片用这个）        ├─ beats    beat_grid.py
                                                   └─ 渲染     a25rig/render.sh
                                                        → data/video/*.mp4
```

`sing.py` 本身的参数、调优结论、各引擎 A/B 见 `AGENTS.md`；本文件只讲**从它的产物做出 MV**。
贯穿示例是 **云烟成雨 0–100s**，路径换成目标歌曲即可。

---

## ⓿ 前置：拿到 vocal.wav 和 mix.wav

```bash
# 完整流水线（默认不含 mix，先验收纯人声）
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac --name 云烟成雨

# 验收 vocal.wav 后再混伴奏
python3 sing.py --ref data/voices/ref.wav --song data/songs/云烟成雨.flac --name 云烟成雨 \
    --stages mix --acc-gain 0.8
```

产物都在 `data/songs/<name>/`：

| 文件 | 用途 |
|---|---|
| `vocal.wav` | SVC 干声。**cues 转写必须用它** —— 伴奏会干扰 ASR 与能量门控 |
| `mix.wav` | 人声 + 伴奏。**成片 `--audio` 用它**，否则 MV 没有伴奏 |
| `eval.txt` | 音色相似度，先看这个决定要不要重跑 SVC |

**两者时长严格一致**（同源，实测都是 240.782s），所以同一时间窗切出来的片段天然对齐 —— 
口型在干声上算、成片用混音出，是安全的。这一点已量化验证过：SoulX-SVC 保留源时间轴，
`vocal.wav` 与 `sep2/lead.wav` 逐 10s 窗互相关最大位移只有 10ms（0.3 帧 @30fps）。

`--acc-gain` 是伴奏音量（默认 0.8）；人声侧是 `--vocal-gain`（默认 1.0）。

---

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
D=/tmp/opencode/云烟成雨_100s; mkdir -p $D
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

不传 `--text` 就自动 ASR 转写（paraformer-large，已缓存）；有歌词文件时 `--text lyr.txt` 更准。

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
print("最长单 cue %.2fs  (>5s 要查是否误判拖腔)" % max(c["end"]-c["start"] for c in cs))
EOF
```

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
    --visemes data/pics/visemes --out data/video/mouth_sheet --zoom 4
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

前面那串环境变量**必须显式点名 NVIDIA 适配器**，否则选到 AMD 核显（比软渲染更慢，
实测 17.3 → 38.3 fps）。Vulkan 必然回落 SwiftShader，所以走 ANGLE/GL。

基础版：

```bash
MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
A25_GL_ARGS="--use-gl=angle --use-angle=gl --ignore-gpu-blocklist" \
a25rig/render.sh --psd data/pics/figure.psd \
    --cues  /tmp/opencode/云烟成雨_100s/cues.json \
    --sway  /tmp/opencode/云烟成雨_100s/beats.json \
    --audio /tmp/opencode/云烟成雨_100s/audio.wav \
    --visemes data/pics/visemes \
    --video data/video/mv_云烟成雨_100s.mp4 \
    --fps 30 --size 16:9 --bg data/pics/bg.jpg
```

雨天版（玻璃水珠 + 落雨丝 + 角色放大右移）—— 注意**去掉 `--bg`**：

```bash
MESA_LOADER_DRIVER_OVERRIDE=d3d12 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
A25_GL_ARGS="--use-gl=angle --use-angle=gl --ignore-gpu-blocklist" \
setsid a25rig/render.sh --psd data/pics/figure.psd \
    --cues  /tmp/opencode/云烟成雨_100s/cues.json \
    --sway  /tmp/opencode/云烟成雨_100s/beats.json \
    --audio /tmp/opencode/云烟成雨_100s/audio.wav \
    --visemes data/pics/visemes \
    --video data/video/mv_云烟成雨_100s.mp4 \
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
| `--glass-blur` | 玻璃模糊，默认 **0**（水痕才看得出变清晰；旧默认 3 时水痕只有周围 0.5×） |
| `--streaks` | 落雨丝（角色**前**）；`--streak-density` 默认 2.8/MP，参考片实测 10/MP |
| `--char-scale/-x/-y` | 缩放 / 位置（**x/y 是 0..1 比例**） |
| `--fps/--seed` | 帧率 / 随机种子（同种子同参数 → 字节级可复现） |

时长由 cues 的 `metadata.duration` 决定，音频由 ffmpeg `-shortest` 截齐。
更多旗标与物理推导见 `a25rig/RENDER.md`。

## ⑦ 验收

```bash
V=data/video/mv_云烟成雨_100s.mp4
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,channels \
    -of default=nw=1 $V     # 期望 100.0 / h264 1920x1080 30/1 / aac 2ch
```

**没人声时嘴必须闭合**，这是最容易回归的一条。探针法（`char-scale 1.15 / x 0.62 / y 0.56` 下）：

```bash
cd /tmp/opencode && rm -rf chk && mkdir chk
ffmpeg -v error -i /home/rookie/voice/data/video/mv_云烟成雨_100s.mp4 -vf fps=5 -q:v 2 chk/f_%04d.jpg
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

实测合格线：有 cue **98%** 张嘴、无 cue **1%**（113 帧里 1 帧）、面积最大 502。

「无 cue 张嘴率」是这套流程唯一必查的回归项。面积最大值只能横向比同一套素材/机位 ——
它随 `--char-scale` 和探针半径变化，不是绝对阈值；判断 D 是否在用，直接看 ③ 的口型集合。

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

## 相关文档

| 文件 | 内容 |
|---|---|
| `AGENTS.md` | 工作区总览、`sing.py` 全部调优结论、各 SVC/移调引擎 A/B、硬件与内存陷阱 |
| `a25rig/RENDER.md` | 渲染器全部旗标、雨层物理推导、确定性验收、实测性能 |
