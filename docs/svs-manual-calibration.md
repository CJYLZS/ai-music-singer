# SVS 手工歌词/MIDI 校准工作流

> ## ⚠️ 整条 SVS 路线已于 2026-08 放弃，这份文档是负面结论的记录
>
> **结论：score-control SVS 的成品自然度远低于 SVC，且不是标注精度问题。**
> 下面全部的校准工作（SOFA 时间轴、手工修 MIDI、MIDI 键盘录制）都做完了，
> 音准与节奏都达标，听感依然"非常不自然"，无法用。
>
> 三轮递进的证据（详见文末「为什么整条路线失败」）：
>
> | 轮次 | 乐谱来源 | 校准待办 | 音准 | 听感判决 |
> |---|---|---|---|---|
> | ① 自动 | ROSVOT + ASR 词时间轴 | 104 条 | GPE 37.9 | 不可用 |
> | ② SOFA | ROSVOT + SOFA 词时间轴 | **49 条** | GPE 33.7 | 仍不可用 |
> | ③ 手修 | ② + 手工修前 130s | 28 条（区间内） | GPE 35.3 | 仍不可用 |
> | ④ 录制 | **MIDI 键盘手弹 + 填词** | 0（乐谱即真值） | **GPE 25.8 / median 16.3 cents** | **仍不可用** |
>
> ④ 是决定性的：乐谱由人直接演奏，音准与节奏**按定义**正确（对自己的谱 GPE 25.8、
> 中位偏差 16.3 音分是全部臂里最好的），听感依然不行。所以瓶颈不在"音符对不对"。
>
> **根因（用户判断，与实测一致）：转成 MIDI 时被抹掉的不是音高精度，是演唱信息。**
> 音符序列只有 `pitch / duration / lyric` 三个自由度，而颤音、气声、转音的强弱、
> 咬字的软硬、句尾的收放全部在这三个自由度之外。SVC 保留了原唱逐帧的 F0 曲线与
> 频谱细节（这正是它"漏原唱音色"的同一条通道），SVS 把它们量化成阶梯函数后
> **信息已经丢失，无法从 MIDI 还原**。音符级表示对流行唱腔是有损压缩，且损失量
> 大于音色转换引入的损伤。
>
> **保留下来的可复用资产**（这些不依赖 SVS，仍在用）：
> - `tools/sofa/` 歌声强制对齐器 —— 词/音素时间戳比 ASR 准得多（坏休止 28.3s→3.5s），
>   任何需要逐字时间轴的任务（字幕、口型、按句切段）都能用。
> - `singing/beat_grid.py` 节拍网格 —— `repair_leaky.py` 的 `--bars` 拼接点在用。
> - `singing/svs_edit.py` metadata↔MIDI 无损往返桥 + `svs_qc.py` 待办清单 —— 若以后
>   出现真正吃乐谱的模型（如 YingMusic-Singer-Plus 的纯乐谱微调路线）可直接复用。
>
> **不要重做的**：换更强的 MIDI 转录模型（GAME 已否决）、逐字歌词时间轴对齐
> （KRC 已否决）、十六分音符量化（已实测本曲无网格结构）。原因见正文各节。

score-control SVS 的输入是**音符 + 歌词**（`gt_f0=None`，原唱音频完全不进模型），
所以成品质量的上限由 metadata 决定。自动转录（ROSVOT + paraformer）的音高够用，
但**时间轴与词/休止边界不够**，这份文档是修它的流程。

## 先看这三个数字，它们决定了要改什么

在 `亲亲` 的自动 metadata（470 音符）上量，参照物是同源 RMVPE F0：

| 维度 | 实测 | 判读 |
|---|---|---|
| \|note − 实唱 F0\| | median **0.09** 半音，>1st 仅 5% | **音高几乎不用动** |
| `<SP>` 槽位里是浊音的比例 | **48.7%** | 一半"休止"其实在唱 |
| 全部浊音帧落在 `<SP>` 里的 | **14.9%** | 一成半歌声被标成休止 |
| 有词音符起点 → 最近浊音起点 | median **140ms** | 节奏错位的真实来源 |

所以校准的动作是**挪音头 / 调时长 / 修词-休止边界 + 纠错字**，不是拖音高。

**⚠️ 这三个数字是纯 ASR 时间轴时代的。** 后两行（休止里在唱）现在由 SOFA 在
预处理阶段自动解决掉大半 —— `svs_song.py` 默认就走 SOFA 时间轴，爱笑的眼睛的
坏休止时长从 28.3s 降到 3.5s、校准待办从 104 条降到 49 条。诊断与实测见
「缺陷的真正来源」一节，操作见步骤 0.5。**"音高几乎不用动"这条不变**。

两条已经被实测否掉的"捷径"，不要再试：

- **FunASR 自带的强制对齐修不了时间。** `speech_timestamp_prediction` 在 60s 窗上
  误差 median **16s**（完全崩），逐行切窗后仍 median 310ms / p90 671ms —— 比要修的
  140ms 误差更大。（注意这条只否掉 FunASR 的时间戳预测，**不否掉歌声专用的强制
  对齐器**，见下方 SOFA 一节。）
- **逐字 KRC 的时间不能直接用。** `绿色` 的 KRC 逐字起点到最近浊音起点 median
  **540ms** / p75 2700ms。**但 2026-08 复查发现这不是精度问题而是版本错位**：
  误差是双峰的（46 个行首里 20 个 ≤240ms，其中 10 个 ≤40ms），且按时间分段
  0-40s median 2.56s / 40-120s **0.13-0.16s** / 160-200s 1.13s / 200s 后 12s，
  KRC 只覆盖到 204.2s 而人声到 267.1s —— 我们手上的音频与 KRC 参照的版本编排
  不同（intro/间奏/outro 长度不一致）。offset 扫描找不到能修正的常数偏移
  （最优 +0.40s，median 只从 435ms 降到 405ms 且其余指标变差）。
  **结论不变（KRC 只用来纠歌词文本），但理由要改**：想用 KRC 的时间就得先解决
  段级对齐，而那件事 SOFA 做得更好、更稳 —— 直接上 SOFA。
  复查脚本：`singing/svs_v1_krc_probe.py`。

## 缺陷的真正来源：音符边界被吸附到 ASR 词时间轴上

先读代码再动手，2026-08 定位到的机制（不是猜测，是三处代码）：

1. `preprocess/tools/note_transcription/model.py:109` —— ROSVOT 在这条管线里
   **不是自由转写器**：`mel2word = align_word(word_durs)` 把 ASR 的词时长做成
   帧级词边界喂给模型。
2. `utils/rosvot_utils.py:19-26`（`regulate_real_note_itv`）—— 凡音符端点落在词
   边界帧上，其时间就被**替换**成 ASR 的词区间：
   `note_itv_secs[idx,0] = word_itv[word_dur_idx,0]`。**ASR 的时间误差就是这样
   变成音符边界误差的。**
3. `tools/lyric_transcription.py:39-104`（`_word_dur_post_process`）—— 只修一种
   错：`<SP>` 窗口**开头**的浊音并给前一个词（≥90% 浊音则整段并入）。
   下一个字提前起唱（晚 onset）、<100ms 的短浊音尾巴，都不修。
4. `regulate_ill_slur` 把同词内同音高的连续音符合并 —— 重复音高的转音被结构性
   销毁，这就是 upstream issue #37 那个"同字被唱两遍"的来源。

所以**换更强的 MIDI 模型是错的方向**（GAME 实测：音准 0.13st、边界 20ms 与
ROSVOT 打平，端到端反而更差 GPE 42.7 vs 37.9 / CER 0.43 vs 0.13）。要换的是
**词级时间轴的来源**。

### V0 可见性审计：手工量的上下界（`singing/svs_v0_audit.py`）

爱笑的眼睛 470 个音符，用 F0 证据逐个判定"这个边界看得见吗"：

| 类别 | n | 含义 |
|---|---|---|
| 可自动定位 | **315** | 43 SP_MOVABLE + 52 ONSET_SNAP + 220 STEP_OK |
| ONSET_BLIND | 19 | 乐句起始落在连续人声内部 |
| ONSET_AMBIGUOUS | 13 | ±250ms 内有 2 个浊音起点，要选一个 |
| SP_STRUCTURAL | 6 | 连唱正中间的幻觉休止，只能删（人工决策） |
| STEP_BLIND | 3 | 真盲区 |

关键发现：**19/19 的 ONSET_BLIND 全部落在连续浊音内部、且前面都紧接一个
浊音占比 >50% 的 `<SP>`** —— 它们不是独立缺陷，是假休止的派生产物。修好休止
结构，这 19 个自动变成句中边界，而句中边界的 F0 可判定率是 220:3。
于是不可压缩的手工量 ≈ 6 结构 + 3 盲区 + 部分 13 歧义 ≈ **≤22/470 (≤4.7%)**。

### V2 SOFA 实测：赢在结构，不在精度（`singing/svs_v2_*.py`）

SOFA = 歌声专用强制对齐器，输入 wav + 无调拼音 `.lab`，输出音素级时间戳。
**边界精度上它并不比现状好**（爱笑的眼睛某句：乐句起始 median 3ms/hit@60ms 80%
vs ASR 30ms/100%；句中 11ms/75% vs 20ms/87.5%），两者都贴着 F0 噪声底。
它赢在**不会在连续歌声里造休止**，全曲 14 段：

| | 休止个数 | 坏休止（槽内 voiced≥0.5） | 坏休止时长 | 占休止总时长 |
|---|---|---|---|---|
| ASR（现状） | 96 / 64.2s | 50 | 28.3s | 44.1% |
| **SOFA** | 83 / 39.2s | **10** | **3.5s** | **8.9%** |

坏休止时长 **8× 下降**，13/14 段改善；派生的 ONSET_BLIND **19 → 1**。
唯一变差的是 intro 段（ASR 幻听出「嗯嗯」，`.lab` 里有音节但音频里没有）——
那是歌词问题，不是时间问题。

三个必须记住的操作细节：

- **对比 F0 时要用 SOFA 的"元音起点"，不是"音节起点"。** 声母比浊音起点早
  106ms（实测 median），拿声母对 F0 会凭空造出 106ms 的假误差。
  喂 `word_durs` 用音节起点，验精度用元音起点。
- **用 `-m force`，不要用 `-m match`。** match 模式在 intro 段把 60 个音素砍到
  14 个（当作"没匹配上"丢掉），坏休止反而从 1 个/1.01s 涨到 6 个/3.9s。
- **置信度只有 0.40-0.55**（全 14 段）。这是 2h 训练的社区模型跑流行歌干声
  （有伴奏残留）的正常水平，别当成对齐失败，但也别当成可以免检。

### V3 端到端：手工量降到 1/3（`singing/svs_v3_build.py`）

把 SOFA 的音节时间轴当 `word_durs` 喂回 ROSVOT 重跑（接口现成，
`apply_rwbd_ = ("word_durs" not in item)`，不改上游一行），再用 `svs_qc.score()`
打分。三臂对照（②是拿**原 ASR 时间轴**走同一条代码路径，用来把"SOFA 的贡献"
和"我这条路径的贡献"分开）：

| 臂 | 音符 | 休止 | 缺陷 | severity | SP_VOICED | ONSET | STEP |
|---|---|---|---|---|---|---|---|
| ① 原始 metadata（ASR） | 566 | 96 | 104 | 48.3 | 49 | 35 | 18 |
| ② ASR 时间轴走 V3 路径 | 571 | 117 | 118 | 46.7 | 74 | 25 | 17 |
| **③ V3 = SOFA 时间轴** | 587 | **65** | **49** | **19.3** | **11** | **15** | 22 |

②比①还差（我从 metadata 反推词槽时合并相邻 `<SP>` 的方式与
`_word_dur_post_process` 不同，丢了它的浊音头合并），所以**③的收益全部来自
SOFA 的时间轴**，而且 49 这个数字是偏保守的。

**换算成手工量**（这才是要回答的问题）：

| | 待办条数 | 涉及时长 | 占总时长 | 必改(sev>0.5) |
|---|---|---|---|---|
| baseline | 104 | 54.2s | 24.2% | 30 条 / 31.2s |
| **V3** | **49** | **17.7s** | **7.9%** | **9 条 / 3.9s** |

待办 −53%、涉及时长 −67%、必改项 −70%。STEP 从 18 涨到 22 是**判据分流**而非
变差：假休止消失后"乐句起始"从 84 个降到 58 个、"句中"从 386 涨到 448 个，
更多音符改用了更严的 F0 台阶尺子（ONSET+STEP 合计 53 → 37）。

残余 49 条里最重的就是 V0 早已点出的那类：intro 的 1.59s 哼唱（71% 浊音）——
`.lab` 里没有对应音节，SOFA 只能把它放进休止，**必须人工决定**加字还是接受。

#### 顺手挖出两个上游 bug（换时间轴必须先修，否则白干）

1. **歌词按未归一化的 `note2words` 取，会串位。**
   `regulate_real_note_itv`（`rosvot_utils.py:12-27`）把 `note2words` 初始化成
   全 0，**只给端点落在词边界上的音符赋值**，最后统一 `+= 1` —— 于是"整个音符
   都在某个词内部、两端都不碰词边界"的音符保持 0 → 变成 1 → **被当成第一个词**。
   `model.py:488-495` 又用 `note2words_raw` 取歌词、用归一化后的算 `note_type`，
   两者不一致。实测 `vocal_107500_116660` 的 raw 是
   `[... 9, 9, 10, 1, 1, 10, 11, 12, 13]`，段尾冒出「我 我」。
   **ASR 时间轴下这个 bug 一次都不触发**（词边界与 ROSVOT 自己的音符边界高度重合），
   换成 SOFA 后 13/14 段触发、共 39 处 —— 任何换时间轴的尝试都会踩到。
   修法见 `fix_note_text()`：按单调不减重建，回跳就沿用前一个音符的词。
2. **`min_word_dur` 过滤只改 `word_durs` 不改 `words`。** 上游
   `model.py:97-107` 丢掉 <20ms 的槽，于是
   `assert word_durs.shape[0] == np.sum(word_bd)+1` 挂掉 → 静默 fallback 到
   "不吸附词边界"，那一段就白换时间轴了（日志里只有一行 `postprocess failed`）。
   SOFA 会产出 4ms 级的声母（实测 `ch` 0.0042s），所以要**自己先处理**：
   `enforce_min_dur()` 只并掉过短的 `<SP>`，有词的短槽改成**向邻居借时长**——
   删掉有词的槽会让 words 比 `note2words` 索引短，歌词整段串位（实测「决 定」
   被并掉后段尾变成「我 我 己」）。

#### 另外三个坑

- **ROSVOT 有 21.33s 硬顶**（`max_frames=4000` × hop 128 / sr 24000）。上游管线
  在**合并前**的短段上跑 ROSVOT 所以没暴露；直接用 `long_cut_wavs` 会有段超限
  （爱笑的眼睛有 1 段 27.4s），超限部分的槽被 `real_len` 截掉 → 触发上面第 2 条。
  `run_maybe_split()` 在**休止正中间**切成两段再拼（切在休止处不会切碎音节）。
- **`duration` 的 2 位小数会累积溢出。** `convert_metadata` 把每个 dur 格式化成
  `%.2f`，几十个音符累积 ±20ms；溢出段的尾音符会与下一段首音符重叠，
  `svs_edit.py verify` 直接报错（实测音符 #507 结束 222.200s > 下一段 222.180s）。
  `clamp_total()` 负责夹住，但有两处必须注意，第一版都踩了：
  要按**写出去的那个表示**（`round(d,2)`）迭代夹紧，只夹原始 float 会漏残差
  （实测仍溢出 +0.010s）；而且段的 `time` 边界**不保证是 10ms 的整数倍**
  （实测有 span=16.419s 的段），所以上限要取 `floor(span, 0.01)`。
- **`subprocess.run(..., text=True)` 会按 locale 解码，在 `svs_song.py` 的环境里
  是 ASCII，直接把父进程打挂。** SOFA（Lightning）会打 `💡 Tip:`，于是父进程抛
  `UnicodeDecodeError`。表现极具误导性：**预处理跑完就 exit 1、没有 traceback、
  `faulthandler` 也不响**，很容易误判成显存不足或 OOM killer（而 `nvidia-smi`
  和 `MemAvailable` 都健康）。改成拿 bytes 自己 `decode("utf-8","replace")`。
  排查这类"无 traceback 退出"的正确姿势是先把 `subprocess.run` 包一层打印
  returncode，而不是先去查显存。
- **拼音写法必须转成词典的写法。** g2pM 写 `lu:e`（略），opencpop-extension 词典
  里是 `lve`；不在词典里的音节 SOFA **静默丢掉**，后面所有字↔音节对位平移一格
  （实测一处 `lu:e` 让那一段 20 个音节全错位，且不报错）。`norm_syl()` 做
  `u:` → `v`；上线前用词典第一列逐个校验 `.lab`（379 个音节，0 漏）。

## 一次校准的完整步骤

### 0. 准备（每首歌一次）

需要 `<song>/sep2/lead.wav`（原唱干声）与 `<song>/prompt_ext.wav`（音色参考）。
自动转录出 baseline metadata：

```bash
cd SoulX-Singer && MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
    ../singing/svs_song.py --song-dir ../data/songs/<歌名> \
    --prompt-wav ../data/songs/<歌名>/prompt_ext.wav \
    --out-dir ../data/songs/<歌名>/svs_base --control score --seed 1234 --cfg 0.5
```

这一步同时产出对照组音频 `svs_base/svs_vocal.wav`（未校准），以及
`svs_base/target/metadata.json`。

### 0.5 SOFA 时间轴（**已是默认，不用手动跑**）

`svs_song.py` 的 `build_meta(sofa=True)` 会在上游预处理之后自动把词时间轴换成
SOFA 的，产物**原地覆盖** `target/metadata.json`，原文件备份成
`target/metadata_asr.json`（同时也是"已处理过"的哨兵，重跑不会二次处理）。

三首歌实测（`svs_qc.score()`，同一把 F0 尺子），收益稳定复现：

| 歌 | 音符 ASR→SOFA | 休止 | 缺陷 | severity |
|---|---|---|---|---|
| 爱笑的眼睛 | 566 → 587 | 96 → **65** | 104 → **49** | 48.3 → **19.3** |
| 冬眠 | 413 → 427 | 70 → **57** | 94 → **47** | 40.9 → **20.6** |
| 情歌 | 552 → 579 | 89 → **74** | 107 → **64** | 38.2 → **17.1** |

缺陷腰斩、severity 降到 ~1/2.3，三首歌全部 `svs_edit.py verify` 往返等价。

- 关掉：`svs_song.py --no-sofa`
- 非 Mandarin、缺权重、缺 `long_cut_wavs` 时自动跳过并沿用上游 metadata（不会让
  出片流程挂掉），日志里有 `[sofa] 跳过：...`
- 对已有的 `target/` 单独补跑（一条命令做完 prep → SOFA → rebuild）：

```bash
cd SoulX-Singer && MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
    ../singing/svs_v3_build.py --refine ../data/songs/<歌名>/svs/target/metadata.json
```

**必看日志**：每段的 `字 n/n` 要相等，不等会打 `!!` 警告 —— 那说明有音节不在
opencpop-extension 词典里被 SOFA 静默丢掉，那一段后面的字↔音节对位会整体平移。

### 1. 自动纠歌词 + 拿到待办清单

```bash
MPLCONFIGDIR=/tmp/mpl SoulX-Singer/.venv/bin/python singing/svs_qc.py \
    --meta data/songs/<歌名>/svs_base/target/metadata.json \
    --lrc  data/songs/<歌名>.lrc \
    --out  data/songs/<歌名>/svs/target/metadata_lyric.json \
    --csv  data/songs/<歌名>/svs/qc.csv --top 80
```

`--lrc` 用 difflib 对齐后**只做等长替换**，增删字一律不动（增删字要连音符一起
改，属于人工动作）。实测爱笑的眼睛纠了 21 个错字（`系→心`、`赌→嘟`、
`压抑→讶异`），并正确跳过 ASR 多出的 3 个哼唱字 —— 早期版本按序号硬对位会把
全曲平移 2 个字，越改越坏。

清单把 566 个音符压成 ~100 个待看位置，三类缺陷：

| 类型 | 含义 | 怎么改 |
|---|---|---|
| `SP_VOICED` | 休止里在唱歌 —— 漏掉的字 | 把 `<SP>` 换成实字，或缩短它给前后音符让位 |
| `ONSET` | 乐句起始音头离浊音起点太远 | 拖音符左边缘对到 F0 曲线的起点 |
| `STEP` | 句中换音处对不上 F0 台阶 | 拖边界对到曲线的音高台阶 |
| `SILENT_WORD` | 有词音符窗内几乎没浊音 | 多出来的字，删掉或缩短 |

**判据按"乐句起始 / 句中"分流，这一点很重要**：句中连唱本来就没有浊音边界可
对（median 140ms 是地板不是缺陷），用浊音起点当判据会把半首歌标红（第一版就是
这样，242 条里大半假阳）。句中改对 **F0 音高台阶**，实测 median 20ms、>150ms
仅 8%，信号干净。改判据后清单从 242 条降到 104 条。

### 2. 导出给编辑器

```bash
MPLCONFIGDIR=/tmp/mpl SoulX-Singer/.venv/bin/python singing/svs_edit.py export \
    --meta data/songs/<歌名>/svs/target/metadata_lyric.json
```

产出三个文件到 `<song>/svs/edit/`：

- `vocal.mid` —— 编辑器吃的 MIDI（含歌词，`<SP>` 是可见可拖的块，slur 显示成 `-`）
- `edit_sidecar.json` —— MIDI 装不下的东西：段边界、每段 `f0`、`language`、
  以及原文件的 2 位小数量化残差（不带它往返就会每次掉 10ms）
- `f0_curve.json` —— 原唱 F0 曲线，MIDI 刻度，清音帧是 `null`

### 3. 起编辑器，校准

```bash
cd SoulX-Singer/preprocess/tools/midi_editor && npm run dev -- --port 5199
# 浏览器打开 http://127.0.0.1:5199/
```

操作顺序：

1. **导入 MIDI** → 选 `<song>/svs/edit/vocal.mid`
2. **载入 F0 底图** → 选 `f0_curve.json`（本工作区新增的按钮）。橙色曲线就是原唱
   实际唱的音高，校准动作变成"把音符方块贴到曲线上"
3. **对齐音频导入** → 选 `<song>/sep2/lead.wav`，可以边听边改
4. 按 `qc.csv` 的时间码逐条跳过去改（CSV 有 `done` 列可以勾）
5. 改完 **导出 MIDI**，存成 `<song>/svs/edit/vocal_edited.mid`

常用快捷键：拖音符块 = 移动；拖左右边缘 = 改音头/音尾；双击空白 = 加音符；
双击音符 = 改歌词；`Backspace`/`Delete` = 删音符；`Ctrl+滚轮` = 横向缩放（缩放越大
snap 越细，最小 0.01s）。

#### 增删音符是支持的，但有一条隐式约束要知道

编辑器确实能增删（双击空白加、选中按 `Delete` 删），**但 metadata 的时间轴是
"段内首尾相接的 duration 链"，格式里没有"绝对起点"字段**（实测原始文件段内
0/565 处有 >5ms 间隙）。所以"删掉一个音符"在 metadata 里无法表达成一个空洞。

早期版本的导入器没处理这件事，实测**删 1 个音符导致 10 个后续音符起点前移、最大
0.28s，且不报任何错** —— 等于毁掉一句的节奏。现在 `import` 会**按段**把空隙补成
`<SP>`（段与段之间的空隙是真实间奏，绝不填），实测删 1/删 3/加字三种情况下，
剩余有词音符的绝对位置**零错位**。

两个提示要留意：

- **加了音符但没填歌词**会被当成休止 `<SP>`（唱"啦"会在成品里凭空多一个字，
  更糟），导入时会 `[warn]` 报出时间码。若本意是补字，回编辑器双击音符填词。
- **音符重叠会报错而不是静默修改。** 常见触发是把尾音拖长越过了后一个音符 ——
  回编辑器点「消除重叠」再导出。

### 4. 导回 metadata

```bash
MPLCONFIGDIR=/tmp/mpl SoulX-Singer/.venv/bin/python singing/svs_edit.py import \
    --mid      data/songs/<歌名>/svs/edit/vocal_edited.mid \
    --sidecar  data/songs/<歌名>/svs/edit/edit_sidecar.json \
    --out      data/songs/<歌名>/svs/target/metadata.json
```

导入时会：段结构按 sidecar 的时间窗重新分配、段内空隙补 `<SP>`（见上一节）、
`phoneme` 按新歌词重算并校验落在 `phone_set` 内、`f0` 按段时长变化重采样
（最近邻，不污染清音）。重叠与未填词都会提示而不是静默处理。

### 5. 合成并对比

```bash
cd SoulX-Singer && MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
    ../singing/svs_song.py --song-dir ../data/songs/<歌名> \
    --prompt-wav ../data/songs/<歌名>/prompt_ext.wav \
    --control score --seed 1234 --cfg 0.5      # 产物 <song>/svs/svs_vocal.wav
```

与第 0 步的 `svs_base/svs_vocal.wav` 对比。**指标要用 GPE 不要只看 FFE**：
FFE 是平均值，会被大量微小改善摊平；耳朵听成"跑调"的是 >50 cents 的粗大错误，
那是 GPE。这个坑本工作区踩过一次（B3 臂 FFE 更好但用户听到音准更差）。

```bash
# 音准。**参考必须用 target_f0.npy（绝对时间轴），不要用 --ref-meta**：
# 后者把各段 f0 首尾相接（爱笑的眼睛是 11203 帧 = 224s），与音频的 257.9s
# 绝对时间轴错位，实测虚报 GPE 90.7% / 偏移 560ms，正确值是 37.9% / 0ms。
MPLCONFIGDIR=/tmp/mpl SoulX-Singer/.venv/bin/python singing/eval_pitch.py \
    --ref-f0 data/songs/<歌名>/target_f0.npy --ref-f0-hz 50 \
    --gen <出1> <出2> --labels base tuned
# 吐字（eval_cer.py 内部已播种；FunASR 的 VAD 非确定性，不播种的 CER 是噪声）
MPLCONFIGDIR=/tmp/mpl MODELSCOPE_OFFLINE=1 SoulX-Singer/.venv/bin/python \
    singing/eval_cer.py --src data/songs/<歌名>/sep2/lead.wav \
    --gen <出1> <出2> --labels base tuned \
    --ref-text data/songs/<歌名>/svs_base/ref_text.txt
```

### 爱笑的眼睛的 baseline 分数（要打败的靶子）

未校准、自动转录 metadata、`seed 1234 / cfg 0.5`，257.9s 合成耗时 61s：

| FFE(cents) | GPE(%) | VDE(%) | CER |
|---|---|---|---|
| 88.10 | 37.91 | 16.70 | 0.1346 (S=37 D=5 I=9) |

存于 `data/songs/爱笑的眼睛/svs_base/eval_baseline.txt`。GPE 37.9% 与本工作区
历史记录的 score-control SVS 音准水平一致（37.9%），所以这条 baseline 是可信的
对照组，不是跑坏的一次。

## 为什么整条路线失败（2026-08 终局）

四个臂全部在爱笑的眼睛上跑完，`seed 1234 / cfg 0.5` 固定，只换乐谱来源。

| 臂 | 乐谱 | 时长 | GPE | CER | 产物 |
|---|---|---|---|---|---|
| base | ROSVOT + ASR 词时间轴 | 257.9s | 37.91 | 0.1346 | `svs_base/` |
| sofa | ROSVOT + SOFA 词时间轴 | 257.9s | 33.66 | 0.1900 | `svs_sofa/` |
| fix130 | sofa + 手修前 130s | 257.9s | 35.25 | 0.1768 | `svs_fix130/` |
| **rec** | **MIDI 键盘手弹 + 填词** | 97.6s | **25.76**\* | 0.2472\* | `svs_rec/` |

\* rec 臂的参考物是**它自己的 MIDI**（用户已明确放弃与原伴奏对齐，"可以不使用
原版伴奏，我自己配伴奏"），所以 25.76 与前三臂的"对原唱"不可直接比大小；
CER 也不可比（rec 是 1.82 字/秒 vs 其他 1.47，字密度高 24%，休止 9.2s vs 22.2s）。

**三条能确定的结论：**

1. **SOFA 的收益是真的，但方向不对。** 待办 104→49 条、坏休止 28.3s→3.5s、
   GPE +4.25（130-258s 段 +9.06），三首歌一致（冬眠 94→47、情歌 107→64）。
   对照组（ASR 时间轴走同一段新代码）反而变差到 118 条，证明增益来自 SOFA 本身
   而不是代码改动。**但它改善的是标注正确性，听感没有跟着改善。**

2. **手工校准的边际收益为负。** 用户手修前 130s 后，0-130s 的 GPE **反而降 3.74**，
   130s 之后 0.00（干净的差分，说明差异确实来自手修区间）。原因不是手修得差，
   而是 SOFA 的自动结果在这段已经接近该表示能达到的上限，人工把音符挪到"看起来
   对齐 F0"的位置反而引入了新的不一致。**修 MIDI 不是通往自然度的路。**

3. **乐谱正确性与听感自然度在这条路上解耦。** rec 臂把"音符对不对"这个变量彻底
   消掉了（人直接演奏，中位偏差 16.3 音分），听感依然不可用。这是整条路线的
   判决书 —— 如果误差为零仍然不行，那么减小误差的任何努力都不会行。

**丢失的是什么（机制）：** metadata 的每个音符只有 `note_pitch`（MIDI 整数）、
`duration`（2 位小数秒）、`text`、`note_type`（新词/slur/休止）四个自由度。
实测原唱在 `<SP>` 之外的连续浊音段里，F0 逐帧变化的标准差是音符量化台阶的数倍 ——
颤音（5-7Hz 调制）、句尾的滑落、转音的过渡曲线、气声段的 F0 抖动，全部落在
`note_pitch` 的 100 音分栅格之下。SVC 路线把这条 F0 曲线**逐帧**喂给模型
（`gt_f0`，50Hz 网格），所以那些细节原样保留；这也正是它同时把原唱音色残留
带进来的**同一条通道**。所以两条路线的取舍是明确的：

- SVC：保留全部演唱细节 + 携带原唱音色残留（可测、可局部修复）
- SVS：零音色残留 + 演唱细节不可恢复地丢失（不可修复）

**用户的原始判断成立**："转换成 MIDI 之后，一些音高的变化信息、歌手的唱法信息被
抹除了，再想还原就非常难，即使音高和节奏都是准确的，但还是无法合成能听的效果。"
rec 臂就是这句话的对照实验。

**级联（SVS→SVC）不受此结论否决**，因为它只把 SVS 输出当音色锚点用，不当成片
（`vocal_cascade.wav`，excess 0.0370 是历史最低）。但它依赖 `svs/svs_vocal.wav`
的质量，而上面的结论说明那个输入是弱的 —— 若要重启级联，先解决 SVS 自然度。

## 工具的验收状态

`svs_edit.py` 有自检子命令，任何改动后先跑它：

```bash
MPLCONFIGDIR=/tmp/mpl SoulX-Singer/.venv/bin/python singing/svs_edit.py verify \
    --meta <任意 metadata.json>
```

已验证 **8/8 文件往返逐字段等价**：我们的两首歌（亲亲 470 音符、爱笑的眼睛 566
音符）+ 上游 6 个 demo（中/英/粤，含英文复合 phoneme `en_P-R-IH1-T-IY0`）。

**为什么不能用上游的 `preprocess/tools/midi_parser.py`**：它是有损的。同一份
`亲亲` metadata 走一圈 `meta2midi → midi2meta`：段数 12→9、音符 470→476、
总时长 216.6s→236.8s、`<SP>` 82→98、**52 个 slur 里 10 个歌词被写成 `<SP>`**
（`midi2notes` 取 `idx-1` 的歌词，而 slur 前一个正是休止）、`f0` 字段整个丢失。

F0 底图已用无头浏览器实测（不是只看构建通过）：曲线与音符方块的纵向中位偏差
**2.2px / 20px 行高**，即曲线落在它所描述的音符块内，两层共用同一坐标系。

## 踩过的坑（避免重复）

- **metadata 无法表达"时间空洞"，所以删音符必须补 `<SP>`。** 段内是首尾相接的
  duration 链，起点靠累加得到。不补空隙时删 1 个音符会让 10 个后续音符前移
  0.28s，静默无错。补要**按段**补 —— 段间空隙是真实间奏，填了会让模型在间奏里
  唱静音音符。
- **同音高音符重叠会静默丢音符。** 用 `dict[(channel,pitch)] = start` 记录
  未闭合的 note_on（上游写法）时，两个同音高音符重叠会让后一个 note_on 覆盖前
  一个，于是前一个音符**消失**，而且残留音符互不重叠、连重叠检查都骗过去。
  触发条件很平常：把某段尾音拖长越过下一段的 `<SP>`（两者 pitch 都是 0）。
  已改成 FIFO 队列 + 悬挂 note_on 检查。
- **`f0` 长度不能按 `dur × 50` 算。** 原始 f0 是从切好的 wav 提的，含
  `END_EXTENSION_SEC` 尾部余量，实测比 `dur×50` 多 1-3 帧。未编辑时必须原样
  返回（`_resample_f0` 的 `old_dur`/`old_n` 参数），否则往返就不等价。
- **英文 phoneme 是复合写法。** `en_P-R-IH1-T-IY0` 整串不在 phone_set 里是正常
  的，`DataProcessor.preprocess` 会先按 `-` 拆开再逐个查表。校验要按拆分后的
  音素做，否则会把上游自己的 demo 判成非法。
- **`npm install` 会因 `~/.npm` 只读失败**（EROFS）。用
  `npm install --cache /home/rookie/voice/.npm-cache`。
- **无头 Chrome 需要 `LD_LIBRARY_PATH`**：
  `LD_LIBRARY_PATH=/home/rookie/voice/a25rig/driver/nsslib`（libnss3 是从 .deb
  解出来的，没装进系统），且要 `headless: true` + `protocolTimeout` 放大，
  `headless:'shell'` 那个 binary 缺库。
- **SOFA 的权重必须对上代码版本，先查 discussion 再下载。** 官方 release
  **v0.0.1**（"Pretrained Model for Mandarin Singing Voice"）是 **v1.0.0 大重构
  之前**的架构：文件是裸 `state_dict`（无 `pytorch-lightning_version`，前缀
  `encoder/edge_decoder/seg_decoder`），而 v1.0.3 的代码要
  `backbone/head` + `hyper_parameters` 的完整 Lightning ckpt —— 配置翻译救不回来，
  只能换权重。可用的是
  [discussion #40](https://github.com/qiuqiao/SOFA/discussions/40) 指向的
  [BaiShuo v2.1](https://github.com/BaiShuoQwQ/SOFA_Models/releases)（387MB，
  声明代码版本 v1.0.3 + 词典 opencpop-extension，CC-BY-NC-SA）。判断方法：
  `torch.load` 后看有没有 `hyper_parameters` 的六个键。
  权重在 `tools/sofa/weights/SOFA_model_mandarin_byBaiShuo/`。
- **SOFA 环境的三个依赖坑**（venv 建好后按顺序踩）：`librosa 0.9.2` + `numba 0.67`
  报 `cannot cache function '__shear_dense': no locator available`（`NUMBA_CACHE_DIR`
  修不了，要 `librosa>=0.11`，代码只用 `load`/`filters.mel`/`sequence.viterbi`，
  升级安全）→ 缺 `torchaudio`（torch 2.13 配 **torchaudio 2.11**，版本号已解耦，
  不存在 2.13.0）→ torchaudio 2.11 把解码交给 torchcodec，报
  `TorchCodec is required`，装 `torchcodec`（系统已有 ffmpeg 6.1 即可）。
  跑的时候带 `MPLCONFIGDIR=/home/rookie/voice/.uv-cache/mpl`。
  单段对齐 ~1.9s（GPU），全曲 14 段几秒。
