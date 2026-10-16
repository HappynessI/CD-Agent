# 双时相变化检测中的精确定位：SoM 与 RULER 论文分析

更新日期：2026-07-22

## 0. 结论先行

Change-Agent 当前的核心矛盾不是“Verifier 看不懂全图”，而是把两个性质不同的任务塞进了同一个生成接口：

1. **语义判断**：T1/T2 中这里究竟是新增、消失、不变，还是证据不足；
2. **几何执行**：应该点击哪一个像素、给出哪个 box，才能让 SimpleClick/SAM3 产生局部且可控的修改。

这两篇论文共同表明：让视觉语言模型从全局图像直接生成精确像素坐标，是不稳定的接口设计。更可靠的做法是在视觉空间与文本输出之间加入一个**可引用的空间中介**：

- SoM 用带编号的候选区域把开放式坐标生成改成区域选择；
- RULER 用与视觉 patch 对齐的“标尺 token”把任意坐标生成改成“引用最近位置 + 小范围偏移”。

对本项目最合适的近期方案不是训练 Verifier 回归坐标，而是：

> **Environment 负责高召回候选和精确几何；Verifier 只负责选择 `region_id`、判断时相语义与置信度；程序再从被选 mask 计算 point/box。**

当前实现已经朝这个方向走了一半：Environment 已生成连通域/XOR/delta proposals，并让 Qwen 对固定 `region_id` 做判断。下一步的重点不应是恢复自由坐标输出，而应是提高候选覆盖率、改善 SoM 式可视化，以及增加“没有候选命中时”的粗到细兜底。

---

## 1. 论文与本地文件

### 1.1 Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V

- 作者：Jianwei Yang 等
- 首次公开：2023-10；本文使用 arXiv v2（2023-11-06）
- arXiv：[2310.11441](https://arxiv.org/abs/2310.11441)
- 项目与代码：[microsoft/SoM](https://github.com/microsoft/SoM)
- 本地 PDF：`2310.11441_Set-of-Mark_Prompting.pdf`，23 页

### 1.2 Improving GUI Grounding with Explicit Position-to-Coordinate Mapping（RULER）

- 作者：Suyuchen Wang 等
- 公开时间：2025-10；本文使用 arXiv:2510.03230 版本
- arXiv：[2510.03230](https://arxiv.org/abs/2510.03230)
- 本地 PDF：`2510.03230_RULER_Position-to-Coordinate_Mapping.pdf`，13 页

RULER 研究的是 GUI grounding，而不是遥感变化检测；因此这里借鉴的是它对“为什么直接坐标生成不可靠”的建模和接口设计，不直接照搬训练数据或指标。

---

## 2. SoM 具体讲了什么

### 2.1 它解决的问题

多模态大模型能回答“图中有什么”，但未必能稳定回答“具体是哪一个、在哪里”。直接要求模型输出数值坐标，会让语言模型同时承担：

- 识别目标；
- 在内部视觉 token 与原图像素之间做坐标换算；
- 以严格数值格式输出结果。

SoM 的关键观察是：模型原本就擅长在文字中引用一个离散符号。因此，与其让模型生成 `(x, y)`，不如先把图像分成有意义的区域，在区域上叠加 `1, 2, 3...` 等可读标记，再让模型回答“目标是区域 7”。

### 2.2 方法链路

SoM 由三个步骤构成。

#### 第一步：生成语义区域

论文使用现成分割模型得到候选 mask，包括 MaskDINO、SEEM、SAM 和 Semantic-SAM。不同分割器提供不同粒度：

- MaskDINO 偏向封闭类别、实例质量较强；
- SEEM 支持更开放的语义；
- SAM / Semantic-SAM 提供类别无关、不同粒度的区域。

这里的本质不是某个具体模型，而是把连续的像素空间变成可枚举的候选集合。

#### 第二步：在区域上放置可读标记

标记可以是数字、字母、box 或 mask overlay。论文特别处理了标记遮挡与重叠：

- 小区域优先分配标记；
- 去除或处理高度重叠的区域；
- 用 distance transform 在 mask 内寻找离边界最远的位置放标签；
- 区域太小时，允许将标签轻微移到附近，避免完全覆盖目标。

这个细节对 Change-Agent 很重要：**mask 的中心点、bbox 中心或轮廓质心未必落在目标内部，而 distance-transform argmax 是更安全的可执行点击点。**

#### 第三步：让模型引用标记

论文采用两种 prompt：

- plain prompt：直接给带标记图像和问题；
- interleaved prompt：将普通图像、带标记图像与文字交错提供。

模型输出的是可读区域 ID，而不是坐标。区域 ID 又能回溯到 mask，因此同一个响应可以支持 referring segmentation、phrase grounding、视频对象分割等像素级任务。

### 2.3 实验说明了什么

论文在 referring expression comprehension/segmentation、开放词汇分割、phrase grounding、视频对象分割等任务上测试 GPT-4V + SoM。

对本项目最有价值的数字不是某个跨数据集 SOTA，而是论文中的接口对照：GPT-4V 直接生成坐标的基线很差（文中对应结果为 25.7），而把输出改成引用区域标记后，grounding 能力显著提高。这支持一个非常具体的工程判断：

> Verifier 的视觉推理能力可能够用，失败的是“视觉语义 → 精确数值坐标”的输出通道。

论文的绝对结果不能直接外推到本项目，因为：

- 使用闭源 GPT-4V；
- 部分实验只在数据集子集上评估；
- 性能受上游分割候选质量强烈制约；
- 自然图像中的物体边界通常比遥感小建筑变化更明显。

### 2.4 SoM 对 Change-Agent 的直接启发

#### 可以立即采用

1. 在 T1、T2、当前 change mask 和局部 crop 上画**同一套稳定 region ID**。
2. Verifier 只返回 `region_id` 和时相状态，不返回自由坐标。
3. 正/负点击点由程序从被选 mask 内计算，优先使用 distance-transform argmax。
4. 一个局部视图最多显示约 5–10 个候选，避免几十个编号互相遮挡。
5. 在全局 overview 中只负责粗定位；选中区域后再用局部 crop 做二次语义确认。

#### 不能机械照搬

SoM 假设上游候选中已经包含正确目标。若漏检变化根本没有进入候选菜单，Verifier 不可能选中它。因此必须把 **proposal recall@K** 作为独立指标，而不能只看最终选择准确率。

---

## 3. RULER 具体讲了什么

### 3.1 它解决的问题

GUI grounding 要把自然语言指令映射到屏幕像素。常规 VLM 直接从视觉特征生成诸如 `x=523, y=217` 的文本 token，但视觉 patch 的位置编码与这些数字 token 之间没有显式对应关系。

论文指出两个后果：

1. **坐标预测不稳定**：模型需要隐式学会 patch 位置到数字的复杂映射；
2. **分辨率泛化差**：训练时学到的坐标范围或屏幕尺寸变化后，映射容易失效。

这与 Change-Agent 的症状高度一致：模型能在全局上描述错误，但数值定位会漂移、落到背景，或退化为整图 box。

### 3.2 RULER tokens

RULER 在输入序列中加入一组辅助位置 token。核心设计是：

- 每个 ruler token 代表一个显式像素参考位置；
- 它与对应图像 patch 共享空间位置 ID；
- 模型不再从零生成任意坐标，而是先引用最近的 ruler token，再预测一个有界的小偏移。

例如目标坐标靠近参考点 `(21, 20)`，模型可先检索该参考，再做小范围算术得到目标。这样把高难度的全局回归改成：

> **离散参考点检索 + 局部残差。**

这和 SoM 的思想相通，但粒度不同：SoM 引用的是语义区域，RULER 引用的是规则空间锚点。

### 3.3 I-MRoPE

论文的第二个贡献是 Interleaved Multidimensional Rotary Position Embedding（I-MRoPE）。

标准多维 RoPE 会把频率维度分块分配给时间、高度和宽度，可能造成宽、高得到的频率表达不均衡。I-MRoPE 将不同维度的频率交错排列，使 width/height 都能使用更完整、均衡的频率范围。

它主要改善高分辨率和跨分辨率泛化。对于固定 256×256 或 512×512 的遥感 patch，这一收益可能小于 GUI 场景；若未来输入分辨率、裁剪尺度变化很大，它才更有吸引力。

### 3.4 训练与结果

论文在 LLaVA-NeXT + Qwen2.5 7B 上从头训练，也在 Qwen2.5-VL 7B 上微调；使用 UGround 的约 800 万条标注、77.5 万张截图，并在 ScreenSpot、ScreenSpot-V2、ScreenSpot-Pro 上评估。

论文报告 RULER 在各基准上稳定改进，且高分辨率界面的增益更明显。例如 ScreenSpot-Pro 中 Qwen2.5-VL 从 34.6 提升到 37.2。

这些数字说明显式空间引用有用，但也揭示成本：RULER 不是纯 prompt 技巧，它需要修改输入 token/位置编码，并用大规模 grounding 数据训练或微调。

### 3.5 RULER 对 Change-Agent 的直接启发

#### 可以借鉴的原则

1. 不要让模型从全图直接生成任意数值坐标；先引用离散锚点。
2. 如果候选 mask 不可靠，可以在全图叠加规则网格/标尺 ID，让模型先选 tile，再在 crop 内选区域。
3. 若必须保留坐标输出，采用“anchor ID + bounded offset”，而不是直接 `(x, y)`。
4. 多尺度 crop 时，坐标必须由程序维护从 crop 到原图的仿射变换；模型不负责换算。

#### 当前不建议直接实现的部分

- 为 Qwen3-VL-2B 改造 RULER token 和 I-MRoPE；
- 立即训练坐标头或大规模 grounding 模型；
- 期待 GUI 训练数据直接迁移到遥感建筑变化。

原因是项目当前瓶颈首先是候选语义是否正确、工具修改是否局部，而不是位置编码本身。候选 ID + 程序几何可以零训练验证假设，成本和风险都更低。

---

## 4. 两篇论文的关系与取舍

| 维度 | SoM | RULER | 对本项目的优先级 |
|---|---|---|---|
| 空间中介 | 语义 mask/box 的编号 | 规则参考坐标 token | SoM 优先 |
| 模型输出 | `region_id` | reference token + offset/coordinate | `region_id` 优先 |
| 是否改模型 | 否 | 是 | 先不改模型 |
| 是否需要训练 | 基本不需要 | 需要大规模 grounding 训练/微调 | 零训练先验证 |
| 候选依赖 | 很强，候选漏了就无法恢复 | 弱，可覆盖整个规则坐标平面 | SoM + 网格兜底 |
| 精度来源 | 上游 mask 的像素边界 | 锚点密度和局部残差 | Environment mask/工具 |
| 最适用阶段 | 当前工程迭代 | 后续训练型 grounding 研究 | 分两阶段推进 |

最合理的组合不是二选一，而是：

> **主路径用 SoM 式语义候选；候选缺失时，用 RULER 式规则网格做粗到细搜索。**

---

## 5. 对 Change-Agent 当前实现的判断

从 `README.md` 与开发记录看，项目已经完成了几个正确的结构性修改：

- Environment 从 change-mask 连通域、T1/T2 object-mask XOR 和 candidate delta 产生稳定 proposals；
- Qwen 对固定 `region_id` 判断，而不是自由生成定位坐标；
- target view、action 和点击锚点由 runtime 推导；
- 当前 change component 与 candidate delta 要求完整覆盖；
- candidate 经过 locality、area、topology 和 pairwise/语义门控再提交；
- 小区域已经有局部 crop 和外侧黄色轮廓辅助。

因此，当前的架构方向与 SoM 已经基本一致。真正尚未解决的不是“要不要菜单”，而是下面三个二阶问题。

### 5.1 菜单覆盖率

现有候选主要来自：

- 当前 change mask 连通域；
- T1/T2 object-mask XOR；
- candidate added/removed delta。

这对已有变化和两时相分割不一致区域有效，但对以下情况仍可能无候选：

- T1/T2 分割模型在同一位置同时漏检；
- 两时相都错误地产生相同 mask；
- 细小建筑或低对比变化未被分割器捕获；
- 变化只出现在 RGB/feature difference 中。

候选池还应加入：

1. 双时相特征差分或 RGB difference 的高响应连通域；
2. SAM/OmniOVCD 的低置信、跨增强不稳定实例；
3. mask 边界带与配准误差带；
4. 多尺度 proposal（小目标与大区域分开）；
5. 当以上都失败时的规则网格 tile。

关键指标是 `proposal_recall@K`：离线用 GT 只做评估，计算每个真实 FP/FN error component 是否被某个 proposal 覆盖。GT 不进入在线 Verifier。

### 5.2 候选呈现是否适合模型选择

只给全图和一个很小的 ring，并不一定让 2B VLM 稳定建立跨图对应。建议把一次决策拆成两层：

1. **全局层**：显示 T1/T2/change/temporal masks 的 overview，候选用同色轮廓与编号；模型只选 1–3 个 coarse IDs；
2. **局部层**：对入选区域生成放大后的四联图，保留相同 ID，必要时把区域细分成 5–10 个 subregion，再判断时相状态。

标号放置采用 SoM 的 distance-transform 方法，避免覆盖小建筑；标号只用于引用，实际执行仍使用原始 mask。

### 5.3 几何解析与工具成功率

选对语义区域不等于工具一定修改正确。建议增加独立的 `GeometryResolver`：

- `negative_point`：在待删除连通域内取 distance-transform argmax；
- `positive_point`：在缺失候选 mask/差分高响应区域内，取高置信内部点或 distance-transform argmax；
- `box`：候选 mask 的 bbox 加固定比例 padding，并裁剪到图像边界；
- 优先级：mask prompt > box prompt > point prompt（取决于现有工具接口可用性）；
- 所有 crop 坐标由程序映射回原图，模型永远不做归一化/反归一化换算。

还要单独统计：`point_hit_rate` 和 `tool_success_rate`。前者测点击是否落在目标区域内；后者测执行后是否真的提升离线 IoU。这样能区分 Verifier 错、几何错和工具响应错。

---

## 6. 推荐的闭环协议

### 6.1 Verifier 输出

Verifier 不再输出 `error_region` 坐标，只输出离散字段：

```json
{
  "region_id": "r7",
  "t1_state": "building",
  "t2_state": "background",
  "diagnosis": "supported_change",
  "confidence": "high",
  "needs_zoom": false
}
```

对于 candidate pairwise gate：

```json
{
  "region_id": "d2",
  "previous_state": "false_change",
  "candidate_state": "background",
  "effect": "beneficial",
  "confidence": "high"
}
```

`target_view`、`suggested_action`、point/box 均由 Environment 根据时相状态、mask occupancy 和候选几何推导。

### 6.2 运行流程

```text
T1/T2 + current masks
        │
        ▼
High-recall ProposalBuilder
  change CC / mask XOR / feature diff / uncertainty / boundary / grid
        │
        ▼
SoM overview：稳定 region_id + 轮廓
        │
        ▼
Global Verifier：选 coarse region_id
        │
        ▼
Local crop / 可选 subregion 菜单
        │
        ▼
Local Verifier：时相语义 + confidence
        │
        ▼
Environment：推导 error type / target view / action
        │
        ▼
GeometryResolver：mask → exact point/box
        │
        ▼
SimpleClick/SAM3 → locality/topology/pairwise gate → commit/rollback
```

### 6.3 没有候选命中时的兜底

不要让模型直接退化为整图 box。改用最多 2–3 层的 coarse-to-fine 搜索：

1. 全图划分 4×4 tile，模型选一个 tile ID；
2. 对该 tile 放大，重新运行 proposal builder；
3. 若仍无 proposal，再细分 3×3 或 4×4；
4. 到最大深度仍不确定则停止，不执行高风险动作。

这相当于一个无需训练的简化 RULER：模型只引用规则空间单元，程序负责坐标变换。

---

## 7. 建议的实验与消融

不要只比较最终 IoU；需要把闭环拆开，否则无法判断到底是哪一层失败。

### 7.1 定位链路指标

| 指标 | 回答的问题 |
|---|---|
| `proposal_recall@K` | 正确错误区域是否进入候选菜单 |
| `selection_accuracy@K` | 正确候选在菜单里时，Verifier 是否选对 |
| `temporal_state_accuracy` | 选中区域后，T1/T2 状态判断是否正确 |
| `point_hit_rate` | 程序点是否落在期望 mask/区域内部 |
| `tool_success_rate` | 工具执行后离线 `ΔIoU > 0` 的比例 |
| `commit_precision` | 被在线 gate 接受的 candidate 中，真实有益比例 |
| `ranking_regret` | 选中 candidate 与 oracle-best candidate 的差距 |

### 7.2 最小消融矩阵

1. 自由坐标输出（历史基线）；
2. Environment candidate ID；
3. candidate ID + SoM 编号/轮廓；
4. 上述 + local crop 二次确认；
5. 上述 + feature-difference/uncertainty proposals；
6. 上述 + grid coarse-to-fine fallback；
7. bbox center vs centroid vs distance-transform point；
8. 单阶段 Verifier vs global-select + local-verify；
9. point/box/mask 三种执行接口。

三样本结果只能验证代码链路，不能支撑方法有效性。建议至少在完整验证集报告均值、改善样本比例、退化样本比例和 bootstrap 置信区间。

---

## 8. 分阶段实施建议

### P0：先验证“引用比坐标好”（低成本）

- 保持现有 Environment proposals；
- 在 overview 和 local crop 上叠加稳定编号；
- Verifier 只选择 ID 和时相状态；
- 统一用 distance-transform point；
- 新增 proposal recall / selection accuracy / point hit 三个离线统计。

### P1：补候选召回与粗到细兜底

- 加入 feature-difference、uncertainty、boundary proposals；
- 做 proposal merge/NMS，但不丢失小区域；
- 增加 4×4 grid → local crop 的两级 fallback；
- 为大编辑与不确定判断保留拒绝/回滚。

### P2：训练型方案

只有在 P0/P1 表明“候选覆盖充足，但 Verifier 仍因空间引用而选错”后，再考虑：

- 用真实 rollout 构建遥感 grounding 训练集；
- 训练 region ranking/heatmap head；
- 采用 anchor ID + bounded offset；
- 最后才评估 RULER token 或 I-MRoPE 改造。

---

## 9. 可形成的研究表述

如果后续目标不仅是修系统，还要形成论文贡献，建议不要把题目表述成“让 Qwen 输出更准的坐标”。更有研究价值的表述是：

> **A semantic-to-geometric action interface for GT-free bi-temporal change correction**：通过高召回候选、双时相局部语义复核、程序化几何解析和受约束的粗到细回退，把全局 VLM 推理安全地转换为局部像素编辑。

这个表述能自然产生三个可验证贡献：

1. 语义判断与几何执行解耦；
2. proposal coverage 与 verifier selection 的可诊断评估；
3. 无 GT 在线闭环中的 locality、pairwise gate 与安全回滚。

---

## 10. 最终建议

“程序提供候选菜单”是当前正确的主方向，但需要补全为以下设计，而不是只给一个有限列表：

1. **高召回候选菜单**负责覆盖；
2. **SoM 式编号和局部 crop**负责让 Verifier 稳定选择；
3. **程序化 GeometryResolver**负责精确 point/box；
4. **规则网格粗到细**负责候选缺失时的兜底；
5. **locality + pairwise gate + rollback**负责阻止错误语义被放大成有害编辑。

短期最值得实现的是 SoM，而不是完整 RULER。RULER 更适合作为后续训练型 grounding 方向，或者被简化为无需训练的“网格 anchor + crop refinement”。

---

## 11. P0 实现状态（2026-07-22）

本仓库的 staged proposal/hybrid 路径已经落地 P0 接口：

```text
Environment proposals
  → deterministic numbered T1/T2/change overview
  → Verifier returns region_ids only
  → local crop evidence + diagnosis
  → runtime distance-transform seed point
  → Environment executes directly
```

`select` 响应严格限制为已有 `region_id`，最多选择三个区域；候选的坐标和
轮廓来自 Environment，模型没有新的坐标字段可写。初始和 accepted-candidate
replan 使用选择菜单；candidate comparison 仍审计全部 delta proposals，以免
候选局部改善掩盖其它区域退化。只有 `false_positive_change` 和
`false_negative` 生成 point；`mixed_error`/`uncertain_region` 暂停，等待后续
子区域拆分或 grid fallback。

Direct rubric 仍保留作对照实验：它有意继续展示全局图并允许模型输出坐标，因而
不能声称已经解决 global-to-pixel gap。其 repair retry 现在锁定原始语义字段，
只修正结构、目标视图或几何，避免 repair 把第一次的 actionable diagnosis 推翻。

当前 focused Slurm regression：49 tests passed（job `44073`）。这只证明协议和
执行链路，不证明无 GT 下 IoU 必然提升；真实收益仍需在完整验证集统计
`proposal_recall@K`、`selection_accuracy@K`、`point_hit_rate`、`tool_success_rate`
和在线 `ΔIoU`。

三样本 CA_0721(13) 首次实跑中，Hybrid aggregate IoU 从 `0.69744116` 提升到
`0.70886178`，但收益只来自一个样本；Proposal-only 的三个动作均退化并被 rollback。
这表明坐标 gap 已被缩小，但当前 change connected-component proposal 仍可能混合 TP/FP，
成为新的主要瓶颈。详见 `ca0721_13_som_ablation_analysis.md`。
