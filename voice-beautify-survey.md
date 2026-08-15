# 人声美化调研：评测方法 + 模型/算法（2026-08）

> 目标问题：有哪些模型/算法可以"美化人声、让人声更自然好听"？
> 前置问题：人声的自然程度/好听程度怎么评测？
> 本报告分两大部分：① 评测方法；② 美化方法。结尾给出结合本工作坊（4070 Ti SUPER 16GB、SVC 流水线）的落地建议。
> 说明：调研覆盖说话语音（speech）与唱歌（singing）两个域，二者评测与美化工具差异较大，均已分别列出。

---

## 一、评测：人声"自然/好听"怎么评

### 1.1 主观评测（金标准，永远的最后裁决）

| 方法 | 说明 | 适用 |
|---|---|---|
| **MOS**（ITU-T P.800 ACR，1-5 分绝对等级） | 听者给单条音频打 1-5 分，通常 5 档：差/尚可/一般/好/优 | 单系统绝对质量/自然度 |
| **CMOS / CCR**（成对比较） | 相对评分 −3…+3，比绝对 MOS 更敏感，样本量需求更小 | A/B 两个系统 |
| **MUSHRA**（ITU-R BS.1534） | 带隐藏参考/锚点的多刺激评分，适合精细差异 | 编解码/增强对比 |
| **ABX / 成对偏好** | 二选一/三选一，统计 wins | 快速决策 |

唱歌领域的惯例（SVS/SVC 论文）：按维度分别打 **naturalness（自然度）、similarity（相似度）、expressiveness（表现力）MOS**，每条 3-8s 片段、20+ 听者取均值。⚠️ 听者少时 MOS 噪声很大（±0.3 以上不罕见），**成对偏好通常比绝对 MOS 更能分辨小差异**。

### 1.2 客观无参考：模型预测 MOS（可自动跑，适合批量回归测试）

**说话语音域：**
- **[DNSMOS](https://zhuanlan.zhihu.com/p/540526841)**（微软）：语音增强/合成自然度打分器，业界事实标准之一。
- **[NISQA](https://github.com/gabrielmittag/NISQA)**（[arXiv 2104.09494](https://ar5iv.labs.arxiv.org/html/2104.09494)）：CNN+自注意力，预测总体 MOS 及多维（noisiness 噪声感 / coloration 染色 / discontinuity 断续 / loudness 响度），另有 TTS 自然度版本。
- **[MOSA-Net](https://zhuanlan.zhihu.com/p/540526841)**、**[UTMOS/SpeechMOS (SSL-MOS)](https://speechify.com/zh-hans-cn/blog/measuring-text-to-speech-quality/)**：自监督表征（WavLM/HuBERT）微调，Zero-shot 跨域能力更好。
- **[AttentiveMOS](https://par.nsf.gov/biblio/10690168)**：轻量 attention-only 质量预测。
- **[NatScore](https://github.com/harrrshall/natscore)**（较新）：**偏好监督的自然度打分器**，专为现代神经 TTS 设计——比 MOS 回归更贴近"自然不自然"的感知。
- **[Uni-VERSA](https://ar5iv.labs.arxiv.org/html/2505.20741)**：统一网络同时预测 MOS 类多属性。

**唱歌域（重点，工作坊最相关）：**
- **[SingMOS](https://huggingface.co/datasets/TangRain/SingMOS)**（[arXiv 2406.10911](https://huggingface.co/papers/2406.10911)）：**6583 段中/日文唱歌片段、9.07 小时**，WavLM 微调预测 MOS；即 **VoiceMOS 2024 唱歌轨**数据（SingMOS_v1）。预训练模型在 [github.com/South-Twilight/SingMOS](https://github.com/South-Twilight/SingMOS)。
- **[SingMOS-Pro](https://huggingface.co/datasets/TangRain/SingMOS-Pro)**（[arXiv 2510.01812](https://arxiv.org/abs/2510.01812)，2025-10）：**7981 段、11.15 小时**，官方唱歌质量评估基准，含预训练 MOS 预测器。
- **[VoiceMOS Challenge 2023](https://ar5iv.labs.arxiv.org/html/2310.02640)**：多域（合成语音/说话人转换/唱歌）零样本 MOS 预测基准；唱歌轨的 baseline 思路可参考。

⚠️ 局限（工作坊已有惨痛教训）：MOS 预测器在训练域外会漂移；对"自然度"这一维度的敏感度有限——**SingMOS 类指标测的是整体质量，测不出"咬字被 kNN 搞坏"这类内容损伤**，必须与 CER/WER 等可懂度指标搭配。

### 1.3 客观信号级 / 声学指标（可解释，直接对应主观感受）

| 类别 | 指标 | 测什么 |
|---|---|---|
| 侵入式质量（需干净参考） | PESQ/POLQA、STOI/ESTOI、ViSQOL | 降质/恢复程度，**不直接测"好听"** |
| 分离质量 | SDR/SIR（BSSEval） | 干声提取干净程度 |
| 可懂度/内容 | **CER/WER**（工作坊已有 `eval_cer.py`）、MCD | 字是否被处理丢 |
| 音准（唱歌） | F0 RMSE、cents 偏差、VDE（清浊判定错误率） | 跑不跑调 |
| **嗓音声学（对应"气息感/沙哑/清亮"）** | **jitter**（基频微扰→粗糙）、**shimmer**（振幅微扰→沙哑）、**HNR**（谐噪比→气息感）、**CPP/CPPS**（倒谱峰值突出度→清亮程度）、**AVQI**（综合嗓音质量指数） | 这些正是"自然好听"的物理成分 |
| 唱歌专项 | 颤音质量（vibrato rate/extent/regularity）、歌手共振峰（~3kHz）能量 | 专业感/表现力 |

- jitter/shimmer/HNR/CPP 是嗓音医学与声学语音学的成熟工具（[综述例](https://www.jvoice.org/article/S0892-1997(26)00137-2/fulltext)），开源可算（praat-parselmouth / librosa / opensmile），**适合量化"处理后气息是否自然、有没有电音感/沙哑感"**。
- 合成语音评测常用组合：MOS + MCD + F0 RMSE + CER/WER。

### 1.4 给本工作坊的评测建议

现有 `eval_sim`（音色相似）/ `eval_stab`（稳定性）/ `eval_cer`（可懂度）覆盖"像不像、稳不稳、字清不清楚"；**自然度/好听度补两个自动读数 + 一个主观协议**：

1. **SingMOS 预训练模型**（WavLM 基座，16GB 卡毫无压力）——唱歌输出的总体 MOS。
2. **NISQA 或 DNSMOS**——说话域（TTS 输出、参考素材）的自然度。
3. **主观 AB 偏好**（每次 3-5 人、10-20 对、统计 win rate）作为最终裁决——这是唯一能真正回答"好不好听"的。
4. 所有对比**同协议同数据**做相对比较；**绝对 MOS 不要跨论文/跨数据集比**。

---

## 二、美化：让人声更自然好听的模型与算法

### 2.1 传统 DSP 链（可控、可解释、零门槛）——"自然好听"的基础

对"干净但平淡"或"有明显缺陷"的人声，先上物理链：

- **音准修正**（跑调）：Auto-Tune / Melodyne 类商业插件（Antares、Celemony）；免费/开源替代：MAutoPitch、Autotalent、GSnap。原理 = F0 检测（CREPE/pyin/RMVPE）→ 拉向目标音阶/旋律曲线 → PSOLA/WORLD/vocoder 重合成。**修正幅度小（<50 cents）时完全听不出，这就是"自然"的关键。**
- **动态**：压缩/多段压缩稳定音量；De-esser 去齿音（5-8kHz）；气息噪声门或 breath de-noise（去气口）。
- **音色**：EQ（高通去低频噪、临场频段、齿音区衰减）；激励器/磁带式饱和增加谐波丰富度（"贵气"感的主要来源）。
- **空间**：板式/卷积混响 + 延迟，平行处理（parallel）保留干声锐度。

参考：现代流行人声链路的公开配置（[pop vocal chain](https://bchillmix.com/blogs/news/modern-pop-vocal-chain-settings-for-home-studio-sessions)、[iZotope Nectar](https://www.izotope.com/en/products/nectar/features)），以及"如何让人声听起来自然"的插件使用原则（[Sonarworks](https://www.sonarworks.com/blog/learn/how-to-achieve-a-natural-sound-using-vocal-effects-plugins)）。

### 2.2 神经重合成 / 神经声码器（平滑、去毛刺）

思路：把语音分解为 F0 + 内容 + 音色，用**神经声码器**（HiFi-GAN / BigVGAN / PC-NSF-HiFiGAN…）重合成 → 抹平拾音噪、毛刺、呼吸爆音，声音更"干净顺滑"。

- 工作坊已有此能力：**PC-NSF-HiFiGAN**（`--ext-method nsf`）。注意 AGENTS.md 教训：重合成有音色代价（过一遍 NSF 掉 ~0.02-0.045 相似度），**只对最终产物做、别动 prompt**。
- 说话域增强/修复模型：[**VoiceFixer**](https://github.com/haoheliu/voicefixer)（[arXiv 2109.13731](https://arxiv.org/pdf/2109.13731)，通用语音修复，有预训练）、[**AnyEnhance**](https://ieeexplore.ieee.org/abstract/document/11095999)（prompt 引导的统一生成式增强，2025）——主要面向噪声/失真恢复，顺带提升听感。

### 2.3 唱歌人声美化（SVB）专用模型 —— 与需求最对口的研究方向

**SVB（Singing Voice Beautifying）**：把"唱得不够好"的业余歌声修成"更好听"的歌声——修音准、补气息、提升表现力，同时尽量保留原唱者音色。这是 2022 年才提出的新任务：

| 模型 | 年份/出处 | 特点 | 代码 |
|---|---|---|---|
| **[NeuralSVB](https://neuralsvb.github.io/)** | ACL 2022（清华） | 首个 SVB 模型；F0 修正 + 音色美化，把业余歌声转成专业质感 | [github.com/auzxb/NeuralSVB](https://github.com/auzxb/NeuralSVB)（官方，含预训练；数据 PopBuTFy） |
| **[ConTuner](https://huggingface.co/papers/2404.19187)** | [arXiv 2404.19187](https://arxiv.org/abs/2404.19187)，2024 | **音高 + 表现力双重条件**，用户可控（目标旋律/表现力强度）；MOS/CMOS 评测 | — |
| **[DiffBeautifier](https://github.com/ishine/DiffBeautifier)** | 2024 | ConTuner 的扩散加速版，Hi-Fi 高质量 + 快速推理 | 官方代码 |

⚠️ 注意定位：**SVB 治的是"唱功"（跑调、气息弱、平淡），不是给已经不错的演唱做音色美容**；与工作坊的 SVC（换音色）互补——理想组合是 SVC 换完音色后，再经 SVB 修正音准/表现力。

### 2.4 唱歌增强（SVE：去噪、去伴奏残留、修复）基准

- **[SingVERSE](https://huggingface.co/papers/2509.20969)**（[arXiv 2509.20969](https://scirate.com/arxiv/2509.20969)）：多样化**真实世界**唱歌增强基准——处理手机录音、现场、K 歌软件等真实场景的噪声/失真。对"用户自己录的素材不好听"场景直接对口。
- 真实演出增强：LSTM + 时频掩码的实时人声增强框架（[Speech Communication 2025](https://www.sciencedirect.com/science/article/abs/pii/S0885230825000968)）。

### 2.5 说话语音的美化（"让声音更好听"的另一半）

- **音色转换到"好听"的目标音色**：RVC / So-VITS / seed-vc（工作坊已有整套）；研究上，"likability（讨喜度）控制"是明确方向——[Interspeech 2025：Voice Conversion for Likability Control](https://arxiv.org/abs/2507.01356)、[NTT 的印象控制 VC](https://www.rd.ntt/e/research/JN202511_37040.html)。
- **生成式增强**：见 2.2（VoiceFixer / AnyEnhance）。
- **产品级参考**（商业实现 = DSP 链 + 神经网络混合）：全民K歌"臻品录音/智能修音"（[52audio 技术演讲](https://www.52audio.com/archives/243214.html)）、唱吧打分/修音、[智能修音与增强技术综述](https://developer.baidu.com/article/detail.html?id=7010920)。

### 2.6 结合本工作坊的落地建议（4070 Ti SUPER 16GB）

按投入产出排序：

1. **评测先行（半天）**：装 SingMOS 预训练 MOS 预测器 + NISQA，接入现有 `sing.py --stages eval` 的扩展，对 SVC 输出加两个自然度读数；保留 AB 听感协议。
2. **快赢（1 天）**：DSP 美容链脚本——高通 + 轻压缩 + de-esser + 微量激励 + 板式混响（Python: `pedalboard`/librosa，或 ffmpeg 滤镜链），对 SVC 干声做后处理；用 jitter/HNR/CPP 前后对比验证"气息是否被处理自然"。
3. **中赢（2-3 天）**：神经重合成平滑——对最终 vocal 用 PC-NSF-HiFiGAN 过一遍（复用现成 `tools/pcnsf`），配合 `eval_stab` + SingMOS 验证毛刺/电音感是否下降。
4. **对口尝试（实验性）**：NeuralSVB / DiffBeautifier 跑通官方推理，看"修音准+表现力"是否给 SVC 产物加分；SingVERSE 的基线系统可作素材增强（手机录音修干净）参考。
5. **不要踩的坑**（工作坊已有教训）：MOS 预测器跨域绝对不可比；只信同协议相对值；CER 评测必须固定种子；任何重合成改动都要回查相似度与稳定性。

---

## 关键链接汇总

**评测：**
- [SingMOS 数据集](https://huggingface.co/datasets/TangRain/SingMOS) / [SingMOS-Pro](https://huggingface.co/datasets/TangRain/SingMOS-Pro) / [SingMOS 预训练模型](https://github.com/South-Twilight/SingMOS)
- [NISQA](https://github.com/gabrielmittag/NISQA)（[论文](https://ar5iv.labs.arxiv.org/html/2104.09494)） / [无参考 MOS 评估综述（中文）](https://zhuanlan.zhihu.com/p/540526841) / [NatScore](https://github.com/harrrshall/natscore) / [Uni-VERSA](https://ar5iv.labs.arxiv.org/html/2505.20741)
- [VoiceMOS Challenge 2023](https://ar5iv.labs.arxiv.org/html/2310.02640)

**美化：**
- [NeuralSVB（ACL 2022）](https://github.com/auzxb/NeuralSVB)（[项目页](https://neuralsvb.github.io/)）
- [ConTuner](https://huggingface.co/papers/2404.19187) / [DiffBeautifier](https://github.com/ishine/DiffBeautifier)
- [SingVERSE 唱歌增强基准](https://huggingface.co/papers/2509.20969)
- [VoiceFixer](https://github.com/haoheliu/voicefixer) / [AnyEnhance](https://ieeexplore.ieee.org/abstract/document/11095999)
- [Likability VC（Interspeech 2025）](https://arxiv.org/abs/2507.01356)
- [全民K歌臻品录音技术分享](https://www.52audio.com/archives/243214.html) / [智能修音技术解析](https://developer.baidu.com/article/detail.html?id=7010920)
- [现代流行人声链路参考](https://bchillmix.com/blogs/news/modern-pop-vocal-chain-settings-for-home-studio-sessions)
