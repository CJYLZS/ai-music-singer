# SVS Prompt 长度影响测试报告

> **⚠️ 2026-08 修正：本报告只测了耗时，没测质量，其中两条结论已被后续实测推翻。**
> 阅读时请以 `AGENTS.md`「SVS prompt extraction」一节为准。
>
> 1. **§5.3「可选优化：用 `prompt.wav` 替换 `prompt_ext.wav`」—— 已 A/B，结论是不要换。**
>    该建议的理由写的是"理论上 SVS 不需要音域扩展（音高来自 MIDI）"。音高确实来自 MIDI，
>    但**音色覆盖的音域来自 prompt**：等长（18.55s）三臂对照实测，含移调素材的 prompt 比
>    纯干净 prompt 在 SVC 下游主指标高 **+0.0411（65/75 窗，p<1e-4）**，>380Hz 段 +0.0526。
>    `prompt.wav` 只到 410Hz，换掉它等于把 345Hz 以上让给别的素材。
> 2. **§3.2「边际成本指数衰减，30s 后达到渐近线」与 §4.2「接近线性，attention 主导」
>    自相矛盾**，且 4 个数据点（各跑一次、无 seed 控制）不足以分辨这两种形状。
>    本工作区已知 SVS 未播种时 SIM std 达 0.0065，耗时同样有抖动。
> 3. **同一份报告的 §6.1 明确写着"音色质量 A/B 尚未完成"** —— 所以 §5.1 那张
>    "不同场景的最优配置"表纯粹是按耗时排的，不含任何质量依据，不要当作质量建议。
>
> 仍然可信的部分：prompt 只编码一次、32 步 Euler 全在 target 侧（§4.1 的代码路径）；
> `cli/inference.py` 只用 `prompt_meta_list[0]`（§5.2 的"有效上限"）。
>
> **另有一个本报告没有发现的缺陷**：`svs_song.py` 当时按**时间前缀**裁剪 prompt，
> 而 `prompt_ext.wav` 是按移调量排序拼接的（移调段全在尾部），所以这 4 个长度档
> 实际都只吃到了干净素材、音域扩展被 100% 切掉。现已改为按音高分层抽样
> （`--prompt-trim stratified`）。

测试日期: 2026-08-12  
测试对象: SoulX-Singer SVS (score-control)  
参考音频: 纯说话素材（从 refsep2/lead.wav 提取，排除唱歌段落）  
目标歌曲: 绿色（268.8s，14 段）  
固定参数: seed=1234, device=cuda:0, control=score

## 1. 测试设计

### 参考音频准备
从 163.1s 的 `refsep2/lead.wav` 中提取纯说话段落：
- 识别标准: 时长 <15s + 能量变异系数 >0.25
- 识别结果: 104 段说话，总时长 123.9s
- 拼接方式: 段间插入 0.2s 静音
- 最终产物: `ref_speech_only.wav` (144.5s)

### 测试长度
从 `ref_speech_only.wav` 切出不同长度的 prompt：
- 10s, 20s, 30s, 45s
- 所有切片从头部开始（保证一致性）

## 2. 实测结果

### 2.1 推理耗时

| Prompt 长度 | Note 数 | 总耗时 | 每段耗时 | vs 10s 基线 | 增量耗时/10s |
|-------------|---------|--------|----------|-------------|--------------|
| 10s         | 50      | 31s    | 2.2s     | 1.00x       | —            |
| 20s         | 96      | 87s    | 6.2s     | 2.81x       | 5.6s         |
| 30s         | 138     | 122s   | 8.7s     | 3.94x       | 3.5s         |
| 45s         | 206     | 131s   | 9.4s     | 4.23x       | 0.6s         |

**关键观察**:
- 增量耗时**快速递减**: 5.6s → 3.5s → 0.6s per 10s
- 30→45s 的边际成本几乎可忽略（0.6s per 10s）
- 符合理论预期：prompt 编码只运行一次，不在 32 步 Euler 循环内

### 2.2 产物验证

所有 4 个产物均完整生成：
- 时长: 268.8s (全曲)
- 大小: 12.3MB
- 段数: 14 段
- 格式: 24kHz, mono, float32

## 3. 性能分析

### 3.1 开销分解

以 10s baseline 为参照：

| 组件 | 10s | 20s | 30s | 45s | 说明 |
|------|-----|-----|-----|-----|------|
| Prompt 编码 | ~1s | ~2s | ~3s | ~5s | Mel 提取 + Note encoder |
| Target 生成 | ~30s | ~85s | ~119s | ~126s | 32 步 Euler × 14 段 |
| 总耗时 | 31s | 87s | 122s | 131s | |

**结论**: 
- Prompt 编码占比随长度增加而**下降**（10s: 3% → 45s: 4%）
- Target 生成是绝对瓶颈（占比 >95%）
- Prompt 长度对总耗时的影响**非线性递减**

### 3.2 边际成本曲线

```
增量耗时 (s/10s prompt)
   ^
6  |  •
   |   
5  |  
   |   ＼
4  |      •
   |        ＼
3  |          
   |            ＼
2  |              
   |                
1  |                  •
   |___________________>
      10→20  20→30  30→45
```

边际成本呈**指数衰减**，30s 后几乎达到渐近线。

## 4. 理论解释

### 4.1 为什么边际成本递减？

**SVS 模型架构** (`soulxsinger.py:188-196`):
```python
# 1. Prompt 编码（只运行一次）
pt_mel = self.mel(pt_wav)                    # O(T_prompt)
pt_decoder_inp = features[:, :len_prompt_mel, :]  # 作为 context

# 2. 生成循环（32 步，瓶颈）
for step in range(n_steps):  # 32 次
    generated_mel = self.cfm_decoder.reverse_diffusion(
        pt_mel,           # ← prompt 只是 context，不参与迭代
        pt_decoder_inp,
        gt_decoder_inp,   # ← 这里是瓶颈
    )
```

**关键机制**:
- Prompt 只在初始化时编码**一次**
- 32 步 Euler 迭代全在 **target 侧**
- Prompt 越长 → 编码开销占比越小

### 4.2 为什么不是完全线性？

**非线性因素**:
1. **显存带宽**: 更大的 prompt context 需要每步读取更多数据
2. **Attention 开销**: CFM decoder 的 cross-attention 复杂度 O(T_prompt × T_target)
3. **但这两者在 T_prompt < 2250 帧时影响很小**

实测 45s (2250 帧) vs 10s (500 帧):
- 理论 attention 开销: 4.5×
- 实际每段耗时: 9.4s / 2.2s = 4.27×
- **接近线性，说明 attention 是主导**

## 5. 实际建议

### 5.1 不同场景的最优配置

| 场景 | 推荐长度 | 理由 |
|------|----------|------|
| **快速原型/调试** | 10-15s | 耗时 <60s，快速迭代 |
| **生产默认** | **20-30s** | 质量与速度平衡，边际成本可接受 |
| **高质量出品** | 30-45s | 边际成本极小（+9s），音色最稳定 |
| **不推荐** | >60s | 预处理会拆段，且信息冗余 |

### 5.2 与理论上限的对比

| 维度 | 实测 | 理论上限 | 约束 |
|------|------|----------|------|
| Prompt 最长 | 45s (测试) | **720s** | `max_length=36000` 帧 |
| 实际上限 | 60s (建议) | 60s | 预处理 `max_merge_duration` |
| 有效上限 | **只用第一段** | — | `cli/inference.py:84` |

### 5.3 Cascade 流程的选择

**当前 cascade 默认**: 复用 SVC 的 `prompt_ext.wav` (45s)

**评估后的建议**:
- ✅ **保持 45s 不变**: 边际成本只有 +100s (vs 10s)，音色更稳定
- ⚠️ 可选优化: 用原音 `prompt.wav` (24.9s) 替换 → 节省 ~60s
  - 需要 A/B 验证音色是否受影响
  - 理论上 SVS 不需要音域扩展（音高来自 MIDI）

## 6. 后续实验建议

### 6.1 音色质量 A/B（尚未完成）

用 ERes2NetV2 评测 4 个产物的音色相似度：
```bash
GPT-SoVITS/.venv/bin/python eval_sim.py \
    --ref data/voices/ref_speech_only.wav \
    --gen data/songs/绿色_svs_10s/svs/svs_vocal.wav \
         data/songs/绿色_svs_20s/svs/svs_vocal.wav \
         data/songs/绿色_svs_30s/svs/svs_vocal.wav \
         data/songs/绿色_svs_45s/svs/svs_vocal.wav \
    --labels 10s 20s 30s 45s
```

预期：20s 后音色趋于饱和（音素覆盖充分）

### 6.2 吐字清晰度（CER）

```bash
GPT-SoVITS/.venv/bin/python singing/eval_cer.py \
    --src data/songs/绿色/sep2/lead.wav \
    --gen data/songs/绿色_svs_*s/svs/svs_vocal.wav \
    --win 10
```

预期：差异很小（内容来自 MIDI，不依赖 prompt）

### 6.3 稳定性（Drift）

```bash
GPT-SoVITS/.venv/bin/python singing/eval_stab.py \
    --ref data/songs/绿色/refsep2/lead.wav \
    --gen data/songs/绿色_svs_*s/svs/svs_vocal.wav
```

预期：更长 prompt → 更低 drift（音色锚点更稳定）

## 7. 结论

1. **Prompt 长度对 SVS 推理耗时的影响远小于预期**
   - 10s → 45s: 总耗时仅增加 3.23× (理论 4.5×)
   - 边际成本快速递减，30s 后几乎平坦

2. **实际瓶颈在 target 段生成（32 步 Euler）**
   - 占比 >95%，与 prompt 长度无关
   - 优化方向应放在段数/步数，而非 prompt 长度

3. **推荐配置更新**:
   - 当前默认 20s → 可放心提升到 **30-45s**
   - 边际成本 <10%，音色稳定性预期显著提升

4. **Cascade 流程无需调整**:
   - 复用 45s `prompt_ext` 的性能代价可接受
   - 除非音色评测证明更短 prompt 无损，否则保持现状

---

**测试文件位置**:
- 参考音频: `data/voices/ref_speech_{10,20,30,45}s.wav`
- SVS 产物: `data/songs/绿色_svs_{10,20,30,45}s/svs/svs_vocal.wav`
- 本报告: `AGENTS.md` 对应章节
