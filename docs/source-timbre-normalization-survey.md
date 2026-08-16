# 源歌声"音色归一化"预处理调研 —— 是否能让 SVC 更简单、效果更好

> 2026-08 调研。问题：不同歌曲 SVC 后 SIM 差异大；听感上"原唱音色越接近目标音色，效果越好"。
> 是否存在一种预处理，先对原唱（源歌手）的音色做归一化修饰——不破坏音高和节奏——从而降低 SVC 难度、获得更好的转换效果？

---

## 0. 结论摘要（TL;DR）

**存在，而且已被 2025 年底的 SVC 论文正面验证**：把源歌声先转换到一个**随机的、与源和目标都无关的辅助音色**，再提取内容特征——即"音色打乱/中性化"。代表是 **YingMusic-SVC**（arXiv 2512.04793，恰好是工作区 `tools/yingmusic/` 的上游项目）里的 **RVC Timbre Shifter**，其消融实验显示这是它全部组件里**贡献最大**的一个（去掉它：SPK-SIM 0.810→0.804，CER 15.70%→16.30%，Mix Vocal 设置下为三个组件中降幅最大）。

但有一个关键前提：**YingMusic 是在训练管线里做这件事的**（模型被训练成"内容特征来自被转换后的音频"）。对于工作区**冻结的 SoulX-Singer**（whisper 编码器 + CFM 解码器），同样的预处理是否有效是**开放问题**，需要 A/B 验证——机制上它同时带来收益与风险：

- **收益机制**：whisper 内容特征里残留源歌手信息（本工作区已实测证明：`eval_triad` 的 excess 泄漏、CFG 放大源歌手、kNN 有效）。源音色被归一化后，残留指向中性/目标方向，泄漏不再拖后腿。这也解释了用户的听感观察：**源≈目标时，泄漏的残留"指对了方向"，所以效果好**。
- **风险机制**：① 二次转换本身有损（每多过一遍 VC 就多一层伪影，CER 风险）；② SoulX 的解码器只见过"自然音频的 whisper 特征"，把"被转换过的音频"喂进去是分布外输入（本工作区 mean-shift 实验失败即同类教训）；③ 若向"目标方向"预转换过头，会双重驱动音色，听感过冲。

另外：用户观察到的"源≈目标效果更好"还有一个**评测地板效应**——输出多少会继承源音色（泄漏），源≈目标时这个继承本身就让 SIM 更高。两种机制叠加，方向一致。

---

## 1. 现象解释：为什么"原唱音色越接近目标，效果越好"

本工作区自己的测量已经把这个现象拆开了（见 AGENTS.md「kNN retrieval」「CFG」两节）：

| 证据 | 结论 |
|---|---|
| `eval_triad`：`corr(A, excess) = +0.396`，C 列（对原唱的相似度）普遍高于 A 列 | 输出确实向**原唱**泄漏，泄漏方向由源音色决定 |
| CFG 放大的是 `content_feat + f0_feat`（whisper 特征）而非说话人嵌入 | whisper 特征里带着源歌手残留，CFG 把它放大 |
| kNN（α=0.2）把内容特征向目标参考的 whisper 流形拉近：SIM +0.0145 且 CER **−1.2pt** | 抑制源残留**同时**改善相似度与咬字——残留是有害的 |
| 歌曲-参考音域差越大 SIM 越低（0-2st → 0.646 vs 6-8st → 0.539） | 音高距离是另一个独立轴（用户说"不破坏音高"，故本调研聚焦音色轴） |

**推论**：如果能把源歌声的**音色（谱包络/声道形状）**先"归一化"（中性化，或向目标拉近一档），whisper 提取到的残留就不再指向原唱——SVC 就"变简单"了。这正是 YingMusic-SVC 的做法。

---

## 2. 已有方案盘点（按"与用户问题贴合度"排序）

### 2.1 音色打乱/中性化预转换 —— YingMusic-SVC 的 RVC Timbre Shifter（最直接对应）

来源：[YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO](https://ar5iv.labs.arxiv.org/html/2512.04793)（arXiv 2512.04793，2025-12）

**做法**（论文 §3.2）：对分离出的源歌声 x，先用一个在 **120 位歌手**上预训练的 RVC（检索式 VC）模型把它转换到**随机辅助音色** s_rand：`x̃ = T_RVC(x, s_rand)`，再从 x̃ 的 mel 提取内容特征。原文："By disrupting the original timbre, the content features become more speaker-invariant and better suited for conversion to the target voice… 模型收到不受源音色 idiosyncrasies 污染的内容特征后，发音与旋律保持都更准。"

**训练侧细节**（§3.1）：随 Seed-VC 的 simulated-inference 掩码，**观测区用原始内容特征、预测区用"打乱音色后的"内容特征**——推理时整曲都是预测区，所以推理路径同样走打乱后的内容特征（§3.5 的推理流程图未再单独声明 shifter，按训练掩码设计推断；以官方代码为准）。

**消融数据**（Table 3，三组件中贡献最大）：

| 设置 | 去掉 RVC shifter 的代价 |
|---|---|
| GT Leading（干净） | SPK-SIM 0.802→0.798，CER 9.20%→9.73%，美学分降 |
| **Mix Vocal（分离残留和声，最接近本工作区场景）** | **SPK-SIM 0.810→0.804，CER 15.70%→16.30%，美学 5.65→5.54/6.33→6.18（降幅最大）** |

**与本工作区的关系**：`tools/yingmusic/` 就是它的实现仓库 fork，但**当前 fork 代码里没有 RVC shifter 组件**（已 grep 确认无 retrieval/rvc/timbre_shifter）。原因：该组件需要 120 歌手 RVC 模型 + 在训练时使用；工作区的 yingmusic 只是推理用零样本转换器。

**启示**：这条路 = "源 →（RVC/任意高质量 VC）→ 中性音色 →（SoulX-Singer）→ 目标"。YingMusic 证明它在**配套训练**的模型上有效；对冻结模型属于可测假设。

### 2.2 同性别/小距离转换（intra-gender SVC）——"源越近越好"的正面文献

来源：[Intra-gender statistical singing voice conversion with direct waveform modification using log-spectral differential](https://www.sciencedirect.com/science/article/pii/S0167639317303710)（Speech Communication 2018，Sakti/Toda/Nakamura 线）；配套 F0 变换研究 [F0 transformation techniques for statistical VC with direct waveform modification with spectral differential](https://ieeexplore.ieee.org/document/7846338)。

**做法**：不做完整谱映射，只对源波形直接施加"源-目标**对数谱差分**"（`d_t = y_t − x_t`，来自统计模型），避免 vocoder 伪影。
**结论**：**同性别内转换所需谱修改量小 → 伪影少、质量高**——这正面支持"源音色与目标越近，转换越容易"的普遍性（不只 whisper 泄漏这一个机制）。异性别跨度的转换质量差是经典结论（AutoVC 等论文也有 m2f/f2m 跨性别劣化的记录）。

### 2.3 声道长度归一化 VTLN / 共振峰搬移（保 F0、保节奏的经典 DSP）

- **VTLN**（ASR 经典，[SPRAAK 教程](https://users.elis.ugent.be/~krdmuync/spraakverwerking/SPRaak/spr_tut_vtln.html)；[TTS 说话人自适应的频率弯折](https://ieeexplore.ieee.org/document/5703015)）：把频谱轴按比例弯折，把说话人声道差异"归一化"到中性长度。ASR 用它减说话人差异；TTS 用它做说话人自适应。**应用到源歌声** = 频率弯折不改变 F0 时间线、不改变节奏，只移动共振峰——天然满足"不破坏音高和节奏"。
- **共振峰搬移**（音高保持版）：Little AlterBoy 一类人声处理插件的 "formant" 旋钮；实现上可分解为 source-filter（WORLD: cheapTrick + D4C）→ 搬移谱包络（乘一个尺度因子或按估计的 F1-F4 对齐）→ 用**原始 F0/非周期分量**重合成。音高与节奏在构造上不动。
- 参考实现家族：[Lifter training and sub-band modeling for voice conversion using spectral differentials](https://ar5iv.labs.arxiv.org/html/2002.06778)（同一个谱差分家族的轻量实现）。

### 2.4 谱包络匹配 / LTAS 均衡匹配（零风险线性滤波）

- 音频制作实践："EQ match"（拿参考人声的平均频谱当 EQ 模板，[自适应均衡系统专利族](https://www.freepatentsonline.com/y2014/0365211.html)）；说话人研究里的 LTAS（[LTAS 与说话人/情感](http://cs.joensuu.fi/pages/tkinnu/webpage/pdf/LTAS_ISCSLP2006.pdf)）。
- 对源歌声做**逐频带增益**，使其长期平均谱（LTAS）匹配目标说话参考的 LTAS。**纯线性滤波 = 音高、节奏、动态完全不变**，是最安全的"音色归一化"，也最容易先做 A/B。

### 2.5 两段式/中间说话人转换 + 语音匿名化（中性化家族）

- 老专利已有"源 → 中间说话人 i → 目标"的三段式设计（[US20100198600A1](https://patents.google.com/patent/US20100198600A1/en)）。
- **语音匿名化**（VoicePrivacy 挑战，[第三届时序综述](https://www.sciencedirect.com/science/article/abs/pii/S0885230826000513)）：把说话人身份抹掉、保留内容——本质上就是"音色中性化"，但面向隐私的模型质量损失大，不适合当 SVC 前置。
- SelfVC 的迭代精化（[SelfVC: Voice Conversion With Iterative Refinement using Self Transformations](https://ui.adsabs.harvard.edu/abs/2023arXiv231009653N/abstract)）是"自己转换自己"的迭代思路，方向相反（精化输出而非预处理输入），但同样暗示多次小步转换比一次大步转换稳。

### 2.6 特征域等价物（工作区已实现并验证）

- **kNN 内容特征替换**（`--knn-alpha 0.2`，本工作区默认开启）：把源 whisper 特征向目标参考的 whisper 流形拉近——**这就是"特征域的音色归一化"**，已实测 SIM +0.0145 / CER −1.2pt，11/11 首歌全正。
- RVC 的 index rate、kNN-VC 是同样的检索式思路；[ContentVec](https://ar5iv.labs.arxiv.org/html/2204.09224) 用对抗训练把说话人信息从 SSL 特征里解耦（模型侧方案，需换编码器）；[聚类音素表示](https://ar5iv.labs.arxiv.org/html/2409.08039)（KM 量化 SSL 特征）同样削弱说话人残留。
- **工作区教训**（决定音频域方案前景）：特征域"直接减均值"（mean-shift 实验）**失败**——解码器把带残留的特征当成了分布内输入；而 kNN（把特征换成真实存在的向量）成功。**类推到音频域：把源音频"换成真实存在的、自然的、音色不同的音频"（VC 预转换）比"线性滤波/搬移"（EQ、formant）更可能成功，因为 whisper 对真实自然音频的特征提取是分布内的**。但线性滤波的优点是零伪影——两者值得都试。

---

## 3. 对 SoulX-Singer（冻结模型）的预期分析

| 方案 | 源残留去向 | 内容特征分布内性 | CER 风险 | 实现成本 |
|---|---|---|---|---|
| 不处理（基线） | 指向原唱 | ✅ 完全自然 | — | — |
| LTAS/EQ 匹配到目标 | 指向目标（弱） | ⚠️ 轻微（EQ 后音频仍自然，whisper 鲁棒） | 很低 | 半小时（纯 scipy） |
| WORLD 共振峰搬移 | 指向目标（中） | ⚠️ 中等（vocoder 重合成，NSF 后处理已证明此类伪影可接受） | 低-中 | 半天（pyworld，CPU） |
| seed-vc 预转换到中性/半程目标 | 指向中性/目标（强） | ⚠️ 中（seed-vc 输出自然度高，但仍有伪影） | 中（二次转换） | 1 天（复用现有 seed-vc 推理） |
| YingMusic 式 RVC 打乱（120 歌手） | 完全随机 | ❓ | 中 | 高（无现成 RVC 模型 + 需配套训练才稳） |

**预期结论**：
1. **"源≈目标效果好"的机制主要是泄漏方向**——任何把源音色往目标方向拉一档的预处理，都应该改善 SIM（前提是不伤内容）。
2. **强归一化（完全打乱）是 YingMusic 在配套训练下的最优**；对冻结 SoulX，中等强度（EQ 匹配 / 共振峰搬移 / 半程预转换）更可能无伤赢分——正好是用户问的"归一化修饰"。
3. 与 kNN（α=0.2）**可叠加**：音频域归一化 + 特征域检索是两道独立的残留抑制。
4. 必须配 CER 与 SingMOS 一起看：二次转换伪影可能吃掉 SIM 收益（本工作区 NSF 的教训：SingMOS 增益 + CER 按歌翻转，需 per-song 门控）。

---

## 4. 建议实验（按性价比排序，全部可复用现有评测协议）

评测协议固定：`eval_sim`（管道协议，ref=ref_talk_sing.wav，配对 seed）+ `eval_cer`（已种子固定）+ SingMOS。每臂跑两首歌：一首"源≈目标"（高基线）、一首"源≫目标"（低基线，如 song_9 这类）——**预测：低基线歌收益更大**（kNN 的增益与基线负相关是前车之鉴）。

- **E1 · LTAS/EQ 匹配**（最先做，半小时）：
  目标 LTAS 取参考说话音频（如 refsep2/lead.wav，若没有歌声参考就用说话参考）的长期平均谱；源取 lead vocal。32-64 频带（或 1/3-octave）增益差，clamp ±12dB，加一点时间平滑避免呼吸噪声被放大。→ `--stages svc,eval` + CER + SingMOS。
  对照臂：**匹配到"中性"LTAS**（源与目标 LTAS 的平均）vs **匹配到目标 LTAS**——直接回答"归一化到中性 vs 归一化到目标"哪个好。
- **E2 · WORLD 共振峰搬移**（E1 有信号再上）：pyworld 分解，估计源/目标 F1-F2（或整包络尺度因子），把源包络向目标搬一档（0.3-0.5 强度），原 F0/AP 重合成。保音高保节奏是构造保证。
- **E3 · seed-vc 半程/中性预转换**（直接检验用户假设）：
  臂 a：源 lead → seed-vc 转换到**中性第三说话人** → SoulX → 目标（YingMusic 思路）；
  臂 b：源 lead → seed-vc 转换到**目标说话参考**（半程：其实这就是"先转换一次再转换一次"，迭代收敛效应）→ SoulX → 目标；
  臂 c：基线。
  用 eval_triad 看 excess 是否下降（泄漏是否真的被压住）。
- **E4（远期）**：若 E1-E3 证明音频域归一化有效但被冻结模型"分布外"拖累 → 真正的解法是 YingMusic 式**配套训练**（SFT 时把源侧换成打乱音色的内容特征），工作区 tools/yingmusic 是现成基座，但训练成本高。

---

## 6. 实验结果（2026-08，本工作区实测）

协议：`--stages svc,eval --seed 1`（配对），eval_sim 管道协议（ref=ref_talk_sing.wav），
CER = 输出 vs 原唱 lead ASR 转写（已种子固定），SingMOS 3s 窗。两首歌：白色毛衣（高基线）/
去有风的地方（低基线）。kNN α=0.2、cfg=0.5 为默认（所有臂一致）。

### 总表（完整）

| 歌 | 臂 | SIM | CER | SingMOS | 说明 |
|---|---|---|---|---|---|
| 白色毛衣 | base | 0.8115 | 0.1618 | 4.236 | |
| | E1 LTAS 旧版 n05/n10 | 0.8124/0.8110 | 0.1553/**0.1392** | 4.248/4.260 | SIM 持平，CER 变好 |
| | E1b LTAS 形状 shape05/10 | 0.8115/0.8123 | 0.1553/**0.1456** | 4.216/4.257 | 同上 |
| | E2 共振峰 fshift05/10 | 0.8130/0.8131 | 0.1586/0.1586 | 4.259/**4.266** | SIM +0.0015，p10 +0.2 |
| | E3 中性预转 neutral3 | 0.8115 | 0.1618 | 4.264 | SIM 持平 |
| | **E3 目标预转 target** | **0.8187** | **0.1489** | **4.292** | **SIM +0.0072 超噪声底** |
| 去有风的地方 | base | 0.7457 | 0.2180 | 3.815 | |
| | E1 LTAS 旧版 n05/n10 | 0.7463/0.7454 | 0.2030/0.2256 | 3.823/3.804 | SIM 持平 |
| | E1b LTAS 形状 shape05/10 | 0.7471/0.7476 | **0.1842**/0.2519 | 3.821/3.816 | shape05 CER −0.034 |
| | E2 共振峰 fshift05/10 | 0.7465/0.7481 | 0.2256/0.2293 | 3.790/3.800 | SIM +0.002 |
| | E3 中性预转 neutral3 | 0.7455 | 0.2857 | 3.857 | CER 恶化 |
| | **E3 目标预转 target** | **0.7568** | 0.2707 | **3.870** | **SIM +0.0111 超噪声底** |

### 核心结论

1. **DSP 谱编辑（LTAS/EQ、共振峰搬移）完全不动 SIM**——whisper 特征对它们免疫
   （实测逐帧余弦：LTAS 形状 0.9996、共振峰搬移 0.9999，改变 <0.1% 帧）。原因：whisper
   是 ASR 编码器，训练时就对 EQ/滤波鲁棒；SoulX 看到的源内容特征几乎没变。CER 的漂移
   （±0.03 内，方向按歌不定）是 EQ 改变音频可懂度特征的直接结果，与 SVC 无关。
   第一版 E1 含整体电平差（说话 vs 唱歌 ~10dB），"匹配"退化成压高频，已作废；
   修正版做纯形状匹配（去均值 + 限带 150-8000Hz，增益 ±6dB 内）。
2. **seed-vc 预转换到目标音色是唯一真正有效的"源音色归一化"**：
   - SIM：白色毛衣 +0.0072、去有风的地方 **+0.0111**（低基线歌收益更大，与 kNN 的
     corr(基线, 增益)<0 规律一致），两首均超 ±0.004 噪声底，且与 kNN α=0.2 叠加生效。
   - SingMOS：+0.056/+0.055（两首一致正）。
   - 机制（whisper 逐帧余弦实测）：预转换后内容特征 cos=0.9235、16.5% 帧显著改变——
     源歌手残留被换成目标残留（与 kNN 同方向，但发生在音频域、更强）。
   - **归因**：预转换源自身 SIM 已达 0.8046/0.7521（≈基线 SVC 输出 0.8115/0.7457），
     最终输出 = 预热启动 + SVC 增量（+0.014/+0.005）——不是纯继承，SVC 在已向目标
     靠拢的源上表现更好。
   - **代价是 CER 继承**：预转换自身 CER 0.1392/0.2744，最终输出 CER 0.1489/0.2707
     基本等于输入 CER。白色毛衣预转换保字率好（CER 反而比基线低）；去有风的地方的
     seed-vc 转换质量差（CER 0.2744），全线继承 → 不可用。
3. **中性第三说话人预转换（YingMusic 思路的近似）无效**：SIM 持平（whisper cos 0.988，
   改变太小），CER 无改善——单次随机中性化不如直接向目标靠拢。
4. **应用建议**：`--source-wav <seed-vc 预转换产物>` 作为可选预处理。采用条件（同 NSF
   教训，per-song 门控）：预转换源自身的 CER 相对原唱不恶化（≤+0.01）且 SIM 增益
   ≥+0.005 才用；去有风的地方这类 seed-vc 转换质量差的歌禁用。更好的预转换器
   （120 歌手 RVC 之类）可能同时改善 SIM 与 CER——见 6.2 展望。

### 6.2 展望（未做）

- **更强的预转换器**：seed-vc 对部分歌的转换质量差（CER 0.27）。换 RVC 类模型
  （YingMusic 的 120 歌手 RVC timbre shifter 就是干这个的）可能把 CER 损伤压下去。
- **配套训练**：YingMusic 的 shifter 收益在训练时注入；对冻结 SoulX 是"预热"，上限有限。
- **预转换 + kNN 的交互**：本实验 kNN α=0.2 全程开启；两者同向（都换源残留），
  预转换后 kNN 的边际贡献待测（可试 α=0.1/0 组合）。
- **强度调节**：seed-vc 预转换是全量。若有"半程"可控（如 timbre 插值），可能
  SIM/CER 交换率更好——目前 seed-vc 无此旋钮。

---

## 7. 下一步路线（2026-08 二次调研，按成本×收益×证据排序）

### 已排除（别再花时间）

| 路线 | 为什么排除 |
|---|---|
| DSP 谱编辑家族（LTAS/EQ、共振峰搬移、任何线性滤波） | whisper 对它们免疫（逐帧 cos >0.999），SoulX 看到的源根本没变；E1/E1b/E2 三组实证 |
| 中性第三说话人预转换（YingMusic 思路的近似） | 单次转换改变太小（cos 0.988），SIM 持平、CER 无改善；YingMusic 的收益在**训练侧** |
| 特征域均值减法（mean-shift） | 已在 kNN 章节证伪：特征离开流形，解码器 OOD |
| NISQA-TTS 当唱歌评测门 | 语音域模型，唱歌上跨歌翻向（NSF 章节结论） |

### A. 管线内零新模型（每项 ≤1 小时，立即可 A/B）

1. **E6 · 迭代自转换（SelfVC 式 / so-vits `-se` second_encoding 同族）**
   [SelfVC: VC With Iterative Refinement using Self Transformations](https://research.nvidia.com/labs/conv-ai/publications/2024/2024-selfvc/)
   ——把 SVC 输出（vocal.wav）经 `--source-wav` 再喂一轮 SoulX：第二轮内容特征来自"已带目标
   音色的合成音频"，残留天然指向目标。机制上就是 so-vits 4.1 的 `--second_encoding`（
   [二次编码说明](https://github.com/shiciki/so-vits-svc)）。风险：合成音频的 whisper 特征略 OOD、
   伪影/CER 累积；但单轮成本 2.5 分钟，值得先看一轮收敛方向。每首歌跑 1-2 轮。

2. **E7 · 预转换参考换成 `prompt_ext.wav`（下游音色锚点）而不是 ref_talk_sing 前 25s**
   当前 E3 用说话参考前 25s 做 seed-vc target——说话域 vs 唱歌域参考差可能是
   去有风的地方 seed-vc CER 0.274 的元凶之一。prompt_ext（25-45s，精选唱歌段）与 SVC
   用**同一音色参考**，消除跨参考损耗，且是唱歌音色（YingMusic 论文强调 singing-trained）。
   顺带可试 ref_talk_sing 537s 里的唱歌段（300s+ 之后能量明显更高）。

3. **E8 · kNN×预转换 2×2 交互**（α∈{0, 0.1, 0.2} × 预转换 {off, on}）
   预转换已把源残留换成目标残留，kNN α=0.2 可能冗余甚至双重驱动；E3b 的全部增益都是
   **在 kNN 开启之上**测得的，两者是加性还是替代关系未知。若预转换+α=0 持平，能同时
   省掉 kNN 的 CER 风险。8 臂 × 2.5min，一晚跑完。

4. **E9 · 预转换迭代**（对 seed-vc 转换质量差的歌，转两遍：每次向目标靠近一点）
   迭代小步转换文献上比一步到位稳（SelfVC 同族）；用 E7 的参考试 2 遍 seed-vc。

### B. 换预转换器（半天-1 天）

5. **E5 · YingMusic-SVC 预转换**（`tools/yingmusic/my_inference.py`，权重已在
   `checkpoints/YingMusic-SVC-full.pt`）
   它是**唱歌域训练**的（SVC engine A/B：作为主引擎比 SoulX +0.04 mean SIM 但 drift 高）。
   作预转换器正好扬长避短：预转换不关心段间 drift，只要求"音色转对 + 内容不坏"，唱歌训练
   应该让"唱歌→目标"比 seed-vc（说话域训练）保内容更好。**必须 `--semi-tone-shift 0`**
   （AGENTS.md 记录过它默认自动移调到参考音域，会改调）。预期直接针对去有风的地方
   CER 0.27 的问题。
   若仍不够：找 RVC 预训练（YingMusic 论文的 120 歌手 timbre shifter 无公开权重，
   社区 RVC v2 模型可替代）。

### C. 模型/训练侧（周级，长期）

6. **换内容编码器**：ContentVec（[对抗解耦说话人](https://ar5iv.labs.arxiv.org/html/2204.09224)）
   或 WavLM+bottleneck（[FreeVC](https://ar5iv.labs.arxiv.org/html/2210.15418) 思路）从根上减残留。
   **必须配套重训解码器**（SoulX 的 CFM 是 whisper 特征训练的，直接换编码器 = mean-shift 式
   OOD，已证失败）。成本高，收益与"预转换+kNN"重叠——优先级低于 A/B。
7. **YingMusic 式 SFT**（训练时注入"打乱音色后的内容特征"，论文消融里 RVC shifter 是三组件
   贡献之首）：需要开源唱歌数据集（M4Singer/OpenCpop/GTSinger 等，论文用了 230h 唱歌 +
   2000h 说话）。工作区只有 12 首歌的 lead，不够。若以后认真做 SVC 训练，这条是"真正的解法"。
8. **换主引擎评估**（2024-2026 零样本 SVC）：[LDM-SVC](https://searchworks.stanford.edu/articles/edsarx__edsarx.2406.05325)
   （latent diffusion + singer guidance，CFG 式抑制源泄漏）、[SaMoye](https://arxiv.org/pdf/2407.07728)
   （解耦 + 相似歌手增强嵌入，跨域鲁棒）、[HQ-SVC](https://ar5iv.labs.arxiv.org/html/2511.08496)
   （统一 codec + EVA 音色-音高对齐，低资源目标）、[R2-SVC](https://ar5iv.labs.arxiv.org/html/2510.20677)
   （F0 扰动 + 和声污染增强，真实世界鲁棒）。工作区已否决 Vevo2（whisper 30s 硬顶）、
   YingMusic（drift 高）——若再试，**R2-SVC / HQ-SVC 最贴近"源泄漏 + 分离伪影"痛点**。
   每个 setup 1-3 天。

### D. 数据/前端（半天）

9. **分离前端升级**：karaoke 分离的 lead 残留和声是真实世界最大伪影源（YingMusic 论文明确
   用内部多轨数据强化分离前端）。工作区已有 `stems4.py`（BS-RoFormer 4stems_ft），
   拿它的 lead 与当前 karaoke lead 各跑一遍基线对比，免费但可能小。
10. **唱歌段进 kNN 池/预转换参考**：kNN 池现在是 163s 说话 lead；ref_talk_sing 537s 含唱歌段
    （300s+ 能量明显更高）。唱歌段的 phone/prosody 覆盖与歌声更匹配（kNN 池增益 = 音素覆盖，
    已实测与音高无关）。

### E. 监控/门控（顺手做）

11. **whisper_diff 做预转换收益预测器**：预转换后 cos<0.95（内容特征真被改变）才值得启用
    （E3 数据：有效的 target 臂 cos=0.923，无效的 neutral3 臂 cos=0.988）。同时把
    "预转换源自身 CER" 做成门（>基线+0.01 禁用）——与 NSF 门控同模式，可写进 sing.py 的
    eval 汇总。

### 推荐执行序

**A 组（今天）**：E6（20min）→ E7（30min）→ E8（1h）→ E9（40min，仅去有风的地方）。
**B 组（明天）**：E5 YingMusic 预转换。
**长期（C/D 组）**：按 C8 → D9/D10 → C6/C7 的顺序，取决于 A/B 组的增量是否够用。
**监控**：E11 随 A 组一起落地（两个数值两行代码，进 summarize.py）。

### 7.1 A 组结果（2026-08 实测，全表 vs 基线：白色毛衣 0.8115/0.1618/4.236，去有风的地方 0.7457/0.2180/3.815）

| 歌 | 臂 | ΔSIM | ΔCER | SingMOS | 判读 |
|---|---|---|---|---|---|
| 白 | **e8a0on**（预转换 prompt_ext + kNN 关） | **+0.0082** | +0.0097 | 4.237 | **最优 SIM，CER 勉强过门** |
| 白 | e7（预转换 prompt_ext + kNN 0.2） | +0.0071 | −0.0032 | **4.290** | SIM/CER/SingMOS 全正 |
| 白 | e8a0（kNN 关，无预转换） | +0.0020 | **−0.0194** | 4.189 | kNN 关本身 CER 大赢 |
| 白 | e6p2/e6p3（迭代自转换） | +0.0061/+0.0059 | **+0.074/+0.217** | 4.128/4.034 | **否决：CER 爆炸** |
| 去 | **e8a0on** | **+0.0171** | **+0.0038** | 3.840 | **最优，CER 几乎持平** |
| 去 | e7 | +0.0134 | +0.0301 | 3.830 | CER 仍伤（kNN 帮倒忙） |
| 去 | e8a0（kNN 关） | +0.0045 | **−0.0451** | 3.797 | kNN 关 CER 大赢 |
| 去 | e6p2（迭代） | +0.0109 | **+0.094** | 3.835 | 否决 |
| 去 | e9（双遍 seed-vc） | +0.0147 | +0.0264 | 3.841 | 不比单遍好，否决 |

**A 组结论：**

1. **E6 迭代自转换否决**：SIM 只多 +0.006，CER 每轮爆炸（+0.074 → +0.217），SingMOS 同步掉
   ——伪影累积没有收敛，SelfVC 式迭代对 CFM 管线不适用（so-vits 的 second_encoding 是
   VITS+浅扩散架构，机制不可移植）。
2. **E9 双遍 seed-vc 否决**：不比单遍好（+0.0147 vs 单遍 +0.0171@α=0），白白多一层损失。
3. **E7/E8 组合是 A 组赢家：预转换参考换 `prompt_ext.wav` + kNN 关（α=0）**：
   SIM +0.008/+0.017（两首歌最佳），CER +0.010/+0.004（基本持平）。相比 E3b 的
   "说话参考预转换 + kNN 开"（去有风的地方 CER +0.053），**prompt_ext 参考把
   去有风的地方的 CER 损伤从 +5.3 点降到 +0.4 点**——说话域参考确实是 CER 元凶之一。
   whisper_diff 佐证：prompt_ext 预转换改变更强（cos 0.87/0.92 vs 说话参考 0.92/0.92，
   白毛衣 32.2% 帧 <0.9）。
4. **新发现：cfg=0.5 下 kNN α=0.2 可能是净负的**。e8a0（kNN 关）在两首歌上都同时
   优于 kNN 开：SIM +0.002/+0.0045、CER **−0.019/−0.045**。历史 kNN 验证全在
   **cfg=1.0** 下做的（α 扫点、11 首歌表、CER 扫点），cfg 降到 0.5 之后 kNN 的
   SIM 增益缩水到噪声内而 CER 代价显现。**默认组合 cfg=0.5×α=0.2 需要一次更宽的
   cfg×α 2×2 验证（≥4 首歌）才能定论**——若复现，默认应改 cfg=0.5×α=0，预转换
   成为替代 kNN 的"音频域残留替换"。这与"预转换+kNN 同向、冗余"的 E8 假设一致。
5. **SingMOS 支持 e7 配置**（白 4.290 全实验最高）——若 CER 门收紧，e7 是安全选择；
   若 SIM 优先，e8a0on。

### 7.2 B 组结果（E5 YingMusic-SVC 预转换，2026-08 实测）

预转换器换成**唱歌域训练**的 YingMusic-SVC（`tools/yingmusic/my_inference.py`，
100 步扩散，`--semi-tone-shift 0`，参考 = prompt_ext.wav），其余同 A 组协议。

| 歌 | 臂 | ΔSIM | ΔCER | SingMOS | 预转换源自身 CER |
|---|---|---|---|---|---|
| 白 | ym0（α=0） | +0.0083 | +0.0065 | 4.250 | **0.0906** |
| 白 | ym02（α=0.2） | +0.0073 | +0.0097 | **4.265**（p10 3.316 最佳） | |
| 白 | 对照 seed-vc e8a0on | +0.0082 | +0.0097 | 4.237 | 0.1392 |
| 去 | **ym0** | **+0.0179（全实验最高）** | +0.0865 | 3.880 | 0.2594 |
| 去 | ym02 | +0.0128 | +0.0790 | **3.916** | |
| 去 | 对照 seed-vc e8a0on | +0.0171 | **+0.0038** | 3.840 | 0.2744 |

**B 组结论：**

1. **YingMusic 预转换源自身 CER 两首歌都优于 seed-vc**（0.091 vs 0.139；0.259 vs 0.274）
   ——唱歌域训练确实更保内容，whisper 门控同量级（cos 0.917/0.917）。
2. **但源级 CER 优势没有传导到最终输出**：白色毛衣两者最终持平（0.168 vs 0.172）；
   去有风的地方 YingMusic 反而差 8.3 点（0.305 vs 0.222）。机制：SoulX 与预转换源的
   交互按转换器不同而不同——seed-vc 源 CER 0.274 → 输出 0.222（SoulX **修复**了部分
   损伤），ym 源 0.259 → 输出 0.305（SoulX **加重**）。YingMusic 输出是 100 步扩散
   重合成，whisper 特征对 SoulX 解码器更 OOD。
3. **去有风的地方的 SIM 天花板被顶到 +0.0179、SingMOS 3.88/3.92（全实验最高）**——
   该歌若允许 CER 牺牲，ym0 是最强音色配置；保字优先仍用 seed-vc prevc + α=0。
4. **综合建议（per-song 门控）**：高基线歌（白色毛衣类）用 YingMusic 预转换
   （CER/SingMOS 略优）；低基线+保字歌用 seed-vc + α=0。**CER 的根不在预转换器**，
   而在"SoulX 与预转换源的交互"（C6/C7 训练侧）或 lead 质量（D9 分离前端）。

### 7.3 C 组结果（cfg×α 2×2 矩阵，2026-08 实测）—— kNN α=0.2 默认改为 0

4 首歌（白色毛衣/去有风的地方/冬眠/绿色）× {cfg 0.5, 1.0} × {α 0, 0.2}，seed 1 配对，无预转换。

跨歌聚合（配对均值）：

| 对比 | ΔSIM | ΔCER | ΔSingMOS |
|---|---|---|---|
| **kNN 0.2 vs 0 @ cfg=0.5** | **−0.0030（0/4 歌赢）** | **+0.0192（4/4 歌变差）** | +0.023 |
| kNN 0.2 vs 0 @ cfg=1.0 | −0.0025（0/4） | −0.0103（帮 CER） | +0.027 |
| cfg 1.0 vs 0.5 @ α=0 | +0.0010 | +0.0030 | +0.017 |
| cfg 1.0 vs 0.5 @ α=0.2 | +0.0015 | −0.0264 | +0.021 |

结论：

1. **kNN α=0.2 在 cfg=0.5 下确认净负**：SIM 0/4 首歌获益（−0.003，两档 cfg 都负），
   CER 4/4 变差（+1.9 点）。历史 kNN 验证（11 首歌 +0.0146 SIM、陪你看星星 CER −1.2 点）
   全在旧协议时代（163s 参考池、24.7s prompt_ext、seed 1234）；当前协议（537s 池、
   44s prompt、seed 1）下 SIM 增益不复现——池与 prompt 变大后残留替换的边际收益消失。
2. **kNN 唯一一致收益是 SingMOS +0.02~0.03**（两档 cfg 都正），不足以抵消 SIM/CER 损失。
3. **cfg=1.0 vs 0.5 基本持平**（SIM ±0.0015 噪声内；SingMOS +0.02）；历史 cfg 扫点的
   0.5-1.0 平台结论保持。低基线歌（去有风的地方/冬眠）per-song 用 cfg=1.0+α=0 可再
   +0.006~+0.013 SIM（绿色/白色毛衣反而 −0.006，按歌定）。
4. **默认配置变更（2026-08）**：`--knn-alpha` 0.2 → **0**（cfg 保持 0.5）。变更后与
   A 组最优组合（预转换 + α=0）一致——预转换成为唯一的"音频域残留替换"手段，
   与 kNN 不再重复。α=0.2 在 cfg=1.0 下帮 CER 的配置保留为按歌实验选项。

## 5. 参考资料

- [YingMusic-SVC (arXiv 2512.04793)](https://ar5iv.labs.arxiv.org/html/2512.04793) / [GitHub](https://github.com/GiantAILab/YingMusic-SVC) —— RVC timbre shifter 全文
- [Intra-gender statistical SVC with direct waveform modification using log-spectral differential](https://www.sciencedirect.com/science/article/pii/S0167639317303710) —— 谱差分/同性别转换
- [Lifter training and sub-band modeling for VC using spectral differentials](https://ar5iv.labs.arxiv.org/html/2002.06778)
- [SPRAAK VTLN 教程](https://users.elis.ugent.be/~krdmuync/spraakverwerking/SPRaak/spr_tut_vtln.html)；[Frequency warping for speaker adaptation of TTS](https://ieeexplore.ieee.org/document/5703015)
- [自适应均衡/频谱模板匹配（EQ match）专利](https://www.freepatentsonline.com/y2014/0365211.html)；[LTAS 与说话人特征](http://cs.joensuu.fi/pages/tkinnu/webpage/pdf/LTAS_ISCSLP2006.pdf)
- [US20100198600A1：源 → 中间说话人 → 目标](https://patents.google.com/patent/US20100198600A1/en)
- [SelfVC 迭代精化](https://ui.adsabs.harvard.edu/abs/2023arXiv231009653N/abstract)；[VoicePrivacy 综述](https://www.sciencedirect.com/science/article/abs/pii/S0885230826000513)
- [ContentVec](https://ar5iv.labs.arxiv.org/html/2204.09224)；[聚类音素表示的零样本 SVC](https://ar5iv.labs.arxiv.org/html/2409.08039)
