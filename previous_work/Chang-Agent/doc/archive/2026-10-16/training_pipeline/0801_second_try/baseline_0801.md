# Qwen3-VL v12.1 baseline 0801：少样本训练闭环总结

日期：2026-08-01  
仓库分支：`exp/baseline-0801-v21`
冻结 runtime 锚点：`491532f4622667e9fbf299c71aac228fa37f6a04`
最终数据工具提交：`0f234d9`（v7 weighted contrast；只增加训练数据构建、测试和文档，没有修改冻结的在线 verifier、Environment、Executor 或模型加载边界）。

## 先说结论

这次工作把“少量数据 → SFT → 闭环评测 → 根据失败轨迹补数据”的流程完整跑通了，训练和评测都能稳定在 Slurm 上完成，也消除了第一轮最严重的“大框覆盖背景”退化。v5、v6 进一步把“模型诊断 → runtime 生成动作 → 工具执行 → candidate → 安全回滚”链路打通，但动作都删错了真实变化区域。v7 又在不增加新内容的前提下提高关键对照的采样曝光，但仍没有动作或改善。到最终 v7，固定 3 个样本的 selected IoU/F1 仍然与 base 完全相同：

```text
base：              IoU 0.7422312，F1 0.85204673
v2–v7 各轮 selected：IoU 0.7422312，F1 0.85204673
最终 v7：            IoU 0.7422312，F1 0.85204673
```

v1 曾明显退化，v2 才恢复到 base；v3–v7 都没有超过 base。因此本轮没有找到可以宣称“稳定提升”的模型。v2 跑过一次 5 样本 e2e5，之后各轮都未通过 3 样本准入门，所以没有再跑 e2e5。v4–v7 反复出现状态和 checklist 自相矛盾的问题；冻结 parser 或 hard gate 阻止了错误写入，但“被安全挡住”不等于模型变好。v7 把 10 条 matched contrast 的总曝光提高到 3 次后，训练 loss 虽降到 `0.3760`，regional errors 却从 v6 的 5 增到 7，说明简单重复同类样本不能解决根本问题。本轮因此停止继续扩充同类数据。

## 这项训练在做什么（给初学者的解释）

这里的模型不是直接输出一张最终变化图。它先看 T1/T2 两张卫星图和一个候选区域，再回答两件事：

1. 这个区域在两个时间点是不是目标类别（这里是 `building`）？
2. 候选白色区域的七项检查是否通过，例如“白色像素是否有目标支持”“边界是否对齐”。

冻结的 runtime 再把这些回答转换成动作：`positive_point`（补回漏掉的目标）、`negative_point`（删掉假阳性）或 `box`。所以训练数据不能只教模型“看起来像建筑”，还必须教它在“两个时相都是背景”时明确让 checklist 失败，以及在真实改善 candidate 上允许删除错误区域。

一条训练样本是一个多模态 prompt/answer 记录，不等于一张图片。多个 stage/scope 变体可以来自同一个 parent，但 train 和 val 按原始 1024×1024 parent 严格隔离。`test_256` 的固定样本只用于 rollout 完成后的离线评测，不进入训练。

## 数据方案与样本分布

所有版本都使用 `train_original`/`val_original`，GT 只用于离线候选筛选、人工检查和构造局部扰动；raw GT 不出现在模型输入或目标中。v2–v7 的训练配置保持一致：2×A800、BF16 LoRA rank 16/alpha 32/dropout 0.05、有效 batch 8、2 epochs、learning rate `3e-5`、cosine + 0.1 warmup，视觉塔和多模态 projector 冻结。v1 是探索轮，使用 4 epochs 和 `1e-4`；发现退化后，后续各轮都从同一个冻结 base 重新训练，而不是在坏模型上继续叠加。

### v1：先建立基本合同（39 train / 24 val）

首轮包含普通 atomic audit、scan、evidence repair 和 candidate evidence。它暴露了生产几何不匹配：训练中的 scan 正例围绕小 GT component，而线上空预测时审计的是精确 3×3 大 tile。模型在 `test_50_8` 接受覆盖背景的大框，约新增 6518 个 false positive，aggregate IoU/F1 从 base 的 `0.7422312/0.85204673` 降到 `0.3687664/0.5388303`。

### v2：加入 production-scale scan hard negative（48 train / 24 val）

把 scan 正例改成更大的真实目标，并加入 3×3 背景 tile 的正确审计和有害 candidate。结果安全性恢复：`test_50_8` 不再产生 6518 个大框 false positive；但模型变得过于保守。v2 的 candidate 几乎全部是 `delta_added`（“添加白区”），没有真实的 `delta_removed` 改善正例，模型学到的策略接近“不要动/直接 finish”。

这就是 v2 只能与 base 持平的主要原因：它学会了拒绝危险动作，却没有学会“什么时候删除一个已经存在的错误区域是有益的”。所以 e2e3 没有 accepted candidate；e2e5 虽完成评分，5 条样本的 aggregate IoU/F1 为 `0.6762904/0.80688931`，且 `test_78_13` 有 7 条 regional validation errors。这个五样本 aggregate 加入了两个更难样本，不能与三样本数字直接作高低比较；它的用途是暴露稳定性问题。

### v3：补真实单侧 transition 和 `delta_removed`

第一轮 v3 增加了 `train_364`/`train_365` 的真实 SAM3 单侧连通域（面积 236、1205、464）、删除型 candidate、真实误删 harm、时间反转以及精确 production scan tile。初次 Job 52129 因 scope 合同漏计 candidate scope 失败，错误信息为 `v3 scope distribution mismatch`；修正后 Job 52132 成功生成 53/24。

随后继续做两次针对性迭代：

| 版本 | train/val | 训练样本组成（train） | 关键目的 |
|---|---:|---|---|
| v3-action | 53/24 | audit standard 19、scan 9、evidence repair 8、candidate 17 | 补真实 transition/删除动作 |
| v3-bgfp | 59/24 | audit standard 22、scan 9、evidence repair 8、candidate 20 | 加 3 个背景 tile FP audit + 删除改善 |
| v3-bgfp-scope（最后用于 Job 52187） | 59/24 | 与上行相同 | 把背景 FP 的 `target_class_only` 从 pass 改为 fail |
| v4-residual | **65/24** | audit standard 26、scan 9、evidence repair 8、candidate 22 | 加 2 个真实 residual 的 initial/repair/candidate 三联 |
| v5-contrast-repair | **69/24** | audit standard 30、scan 9、evidence repair 8、candidate 22 | 修 residual 标签冲突并补 finish/action 对照 |
| v6-matched-contrast | **63/24** | audit standard 25、scan 9、evidence repair 8、candidate 21 | 回到 53 条 v3 核心，加 10 条等量正反 matched contrast |
| v7-weighted-contrast | **83 exposures（63 unique）/24** | audit standard 37、scan 9、evidence repair 8、candidate 29 | 不加新内容；10 条 matched contrast 各总曝光 3 次 |

v4 的详细 train 分布为：

- semantic state：`background→building` 30、`background→background` 19、`building→building` 8、`building→background` 8；
- candidate outcome：improvement 16、harm 6；其中新增 v4 的 2 条 improvement 是 `delta_removed`；
- parent：train 11 个、val 6 个，无 parent overlap；
- 所有新增 residual 的 GT change overlap 都是 0，并记录了 component rank、面积、box 和 SHA256。

v4 的两条新增 residual（都来自 train-only parent 的局部区域）是：

| parent/candidate | 方向与 rank | 面积 | box | RGB 人工结论 |
|---|---|---:|---|---|
| `train_51` | T2 rank 0 | 1804 | `[124,130,184,181]` | T1 裸地；T2 施工车辆/裸地，无 roofed building |
| `train_85` | T1 rank 1 | 258 | `[74,20,91,41]` | 两个时相都是树冠背景，无 roofed building |

每条 residual 生成三条样本：

1. 初次 standard atomic audit：`background/background`，`white_pixels_supported` 与 `boundary_alignment` 为 fail；
2. 使用线上真实报错 `a white region with no target transition cannot pass as correct` 的 standard repair replay；
3. 执行 `negative_point` 后的 `delta_removed` candidate improvement。

### 最终训练集：v7 的 63 条 unique / 83 条 train exposure / 24 条 val

这里必须区分“不同内容”和“模型实际看到的行数”：v7 仍只有 63 条不同训练内容，但把其中 10 条关键 matched contrast 各复制 2 行，因此每条关键内容总共出现 3 次，训练文件共有 83 行 exposure。它不是 83 个新样本或新场景。同一 parent 可以提供 audit、repair、candidate 等不同训练问题；train 的 11 个 parent 与 val 的 6 个 parent 完全不重叠，固定 `test_256` 评测样本也没有进入训练。

63 条 unique 内容与 v6 完全相同，分布为：

- 按任务：standard audit 25、scan audit 9、evidence-only repair 8、candidate evidence 21，共 63；
- 按时相语义：`background→building` 32、`background→background` 13、`building→background` 10、`building→building` 8，共 63；
- candidate 的结果：改善 13、有害 8，共 21；其余 42 条不是 candidate 任务；
- 按 audit kind：missing 16、present 17、`delta_added` 11、`delta_removed` 10、scan 9，共 63；
- 53 条来自 Job 52132 的冻结 v3 核心，新增的 10 条由两个真实背景 residual 与两个真实 transition 组成匹配对照。

计入显式重复后的 83 条 exposure 分布为：

- 按任务：standard audit 37、scan audit 9、evidence-only repair 8、candidate evidence 29，共 83；
- 按时相语义：`background→building` 36、`background→background` 25、`building→background` 14、`building→building` 8，共 83；
- candidate 的结果：改善 17、有害 12，共 29；
- 按 audit kind：missing 16、present 29、`delta_added` 11、`delta_removed` 18、scan 9，共 83。

v5 的 69 条不是 v6/v7 的父集。v6 主动删掉 v5 中 6 条 production 大背景 tile FP/removal pair，保留行为较好的 53 条 v3 核心，再只加入 10 条 matched contrast；v7 只改变这 10 条的曝光权重。因此“69 降到 63 unique、再变成 83 exposure”是一组受控配方和采样实验，不是数据丢失，也不是新增 20 条独立知识。

## 作业与可追溯结果

下面的数值均来自对应输出目录中的 `validation.json`、`training_summary.json`、`post_sft_summary.json` 或 `protocol_summary.json`。所有计算型作业均通过 Slurm 提交（其中 Job 52129 按预期保留为失败诊断）；训练稳定后按约束使用 30 分钟监控窗口，评测使用 5 分钟窗口（作业提前结束即收取）。

| Job | 内容 | 结果 |
|---:|---|---|
| 51896 | v1 SFT，2×A800 | loss `0.2928593` |
| 51903 | v1 e2e3 | IoU/F1 `0.3687664/0.5388303`，严重 FP 退化 |
| 52009 | v2 数据修复构建 | 48/24，合同通过 |
| 52011 | v2 SFT，2×A800 | loss `0.4735843` |
| 52015 | v2 e2e3 | base 持平，accepted `0`，regional error `0` |
| 52035 | v2 e2e5 | IoU/F1 `0.6762904/0.80688931`，7 regional errors |
| 52129 | v3 初次数据构建 | **失败**：scope distribution 漏计 candidate |
| 52132 | v3-action 数据构建 | 53/24，manifest `0be3b1ad…` |
| 52143 | v3-action SFT | loss `0.4500545`，merged `eff1ea9b…` |
| 52146 | v3-action e2e3 | base 持平，accepted `0`，regional error `1` |
| 52147 | v3-bgfp 数据构建 | 59/24，manifest `6672c2cb…` |
| 52148 | v3-bgfp SFT | loss `0.4401292`，merged `221cf6a0…` |
| 52150 | v3-bgfp e2e3 | base 持平，accepted `0`，regional error `3` |
| 52183 | v3-bgfp-scope 数据构建 | 59/24，manifest `bb09c74c…` |
| 52187 | v3-bgfp-scope SFT | loss `0.4412667`，adapter `19587e52…`，merged `6613b488…` |
| 52218 | v3-bgfp-scope e2e3 | base 持平，accepted `0`，regional error `2` |
| 52244/52250/52252 | v4 residual component 只读诊断与 RGB crop 复核 | 确认 rank/面积/GT overlap，并人工复核两个局部 |
| 52256 | v4-residual 数据构建 | **65/24 合同通过**，manifest `421e3dfc…` |
| 52258 | v4-residual SFT，2×A800 | loss `0.4197052`，merged `633675e0…` |
| 52259 | v4-residual e2e3 | base 持平，accepted `0`，regional error `7`；未进入 e2e5 |
| 52307 | v5-contrast-repair 数据构建 | **69/24 合同通过**，manifest `83a35bb…` |
| 52311 | v5 manifest 首次独立复验 | **失败诊断**：测试重建 serialized Environment 时丢失 `editable_seed_white`；不是数据合同失败 |
| 52315/52319 | 修复后的 v5 manifest 复验 | 69/24 合同再次通过；构建后显式 gate 通过 |
| 52326 | v5 SFT，2×A800 | loss `0.4225744`，merged `13830e23…` |
| 52351 | v5 e2e3，1×A800 | base 持平；tool action `1`、accepted `0`、rejected `1`、regional error `6` |
| 52427 | v6-matched-contrast 数据构建 | **63/24 合同通过**，manifest `0eb835bd…`；冻结 v3 core receipt 通过 |
| 52428 | v6 SFT，2×A800 | loss `0.4220659`，merged `20845c44…` |
| 52429 | v6 e2e3，1×A800 | base 持平；tool action `1`、accepted `0`、rejected `1`、regional error `5`；未进入 e2e5 |
| 52433 | v7-weighted-contrast 数据构建 | **83 exposures / 63 unique / 24 val 合同通过**，manifest `e86da116…` |
| 52434 | v7 SFT，2×A800 | 83 examples、22 steps、loss `0.3760475`，merged `e80049cb…` |
| 52436 | v7 e2e3，1×A800 | base 持平；candidate/action/accepted 均 `0`，regional error `7`；停止，不进入 e2e5 |

完整哈希和路径保存在实验 registry 及各输出目录，不在表中截断替代原始证据。

## 失败轨迹说明：模型到底错在哪里？

### v2 的失败轨迹

v2 的主要错误不是“把背景变成大框”了，而是反过来过度谨慎：candidate 样本缺少真实 `delta_removed` 正例，模型倾向于把 audit 判成 `none`，直接 `finish`，或者不生成可执行动作。安全门因此没有被绕过，但也没有机会改善初始 mask。

### v3-bgfp-scope 的轨迹（Job 52218）

`test_20_15` 初始有 8 个白色 proposal。v3-scope 后 `r6` 已从错误的 `background/background` 改为 `background/building`，所以 regional errors 从 3 降到 2；但 `r1` 和 `r7` 仍然是：

- T1/T2 都回答 `background`；
- 七项 checklist 却全部 `pass`；
- repair 后仍重复 all-pass；
- frozen parser 只能安全地拒绝，因没有可定位的 fail，最终 `plan=None`、candidate/tool action 为 0。

这证明问题在模型的“状态—checklist 一致性”学习，而不是 SimpleClick、SAM3 executor 或 action locality。`test_1_1` 仍把 626 个 false-positive 像素当成 `background→building` 并直接 finish；`test_50_8` 仍漏掉 166 个真实变化并直接 finish。

### v4 的轨迹（Job 52259）

v4 的 6 条 targeted 样本确实降低了训练 loss，但在这 3 个 diagnostic 样本上表现出明显的过拟合/语义迁移迹象：在 `test_20_15` 中只有 `r0` 保持有效；模型在 `r1–r7` 的原始回答里把更多真实变化区域写成 `background/background`，同时又把七项 checklist 全写成 `pass`。这两个判断互相矛盾，冻结 parser 才把解析后的语义状态安全降级为 `uncertain/uncertain`，于是触发 7 条 regional errors。模型并不是直接学会输出 `uncertain`；它学到的是错误的“背景语义 + all-pass”组合，仍没有区分“背景假阳性”与“真实 background→building change”。因此：

- aggregate IoU/F1 没有任何提升；
- accepted candidate、successful tool action 都是 0；
- `test_1_1` 的 FP626 与 `test_50_8` 的 FN166 行为没有改变；
- regional errors 比 v3-scope 的 2 条恶化到 7 条。

这是一条有价值的失败证据：在 2 epoch、65 条少样本数据和当前 3 样本闭环中，增加少量背景 residual repair 与类别/语义过拟合迹象同时出现；单靠继续加同类 checklist 样本不可靠。

### v5 的轨迹（Job 52351）

v5 首次产生了可执行的 `negative_point`，因此工程上比 v4 前进了一步：工具调用成功，形成 1 个 candidate。但模型在 `test_20_15` 选中了错误区域。该动作只减少 16 个 FP，却同时删除 324 个 TP，令该样本 IoU 从 `0.84638554` 降为 `0.79122246`。runtime 的 `point_proposal_combined_change_gate_exceeded` 在进入 candidate evidence 之前拒绝并回滚，所以最终 selected mask 仍是初始 mask。

这说明两个容易混淆的概念：

- “tool action 成功”只表示工具按坐标执行成功，不表示改动方向正确；
- “最终没有退化”来自 hard gate 回滚，并不表示训练模型有提升。

v5 的 aggregate 最终仍为 `0.7422312/0.85204673`，accepted `0`、rejected `1`、regional errors `6`、atomic valid `12/18`。

### v6 的轨迹（Job 52429）

v6 用匹配的“该删/不该删”对照替代继续累积数据，regional errors 从 v5 的 6 降到 5，atomic valid 从 `12/18` 升到 `13/18`，但关键行为没有改变：

- `test_20_15` 的 `r1/r2/r3/r6/r7` 仍输出 `background/background + all-pass`，被 parser fail closed；`r0/r5` 正确；
- `r4` 虽识别为 `background→building`，却把 `white_pixels_supported` 写为 fail，于是错误生成 T2 `negative_point`；
- 这次删除的是 360 像素 component，TP 从 4777 降到 4417，FP 保持 613，FN 从 254 增到 614；candidate IoU/F1 从 `0.84638554/0.91680261` 降为 `0.78260099/0.87804393`；
- 同一个 hard gate 拒绝并回滚，candidate evidence 调用数仍为 0，最终 accepted `0`、rejected `1`；
- `test_1_1` 仍保留 626 个 FP 后 finish；`test_50_8` 的 9 个 scan 全部 finish，仍保留 166 个 FN。

因此 v6 只是把协议错误减少 1 条，没有任何一条样本的 selected IoU/F1 严格提升，也没有 accepted candidate。aggregate 继续与 base 完全相同。

### v7 的轨迹（Job 52436）

v7 的 loss 从 v6 的 `0.4221` 降到 `0.3760`，但这只是“模型更会复述训练答案”的训练信号，不能直接代表它在新图像的闭环里会做对动作。实际 e2e3 恰好证明了这一点：

- `test_20_15` 的 8 个 region 只有 1 个 diagnosis 为 `none`，其余 7 个都是 `uncertain_region`；repair 后仍没有可定位动作，结果为 `no_executable_action`；
- 更具体地说，repair 后只有 `r0` 改成了自洽的 `background→building + all-pass`；`r1/r2/r3/r4/r6/r7` 仍是“状态写 `background/background`、checklist 却全 pass”，`r5` 则在已有白区上只报 `changed_object_extent_complete=fail`，被 runtime 判为不可能的“纯 false negative”。这 7 个回答都只能 fail closed；
- 7 个 `uncertain_region` 使 completion gate 失败并得到 `plan=null`。随后记录的 1 次 `action_attempt_error` 只是“verifier 没授权可执行动作”，并非 SimpleClick/SAM3 工具崩溃；raw action、parsed action 和 tool action 都不存在；
- candidate、successful tool action 和 accepted 都是 0，初始 FP613/FN254 原样保留；
- `test_1_1` 仍错误 finish，留下 626 个 FP；`test_50_8` 的 9 个 scan 仍错误 finish，留下 166 个 FN；
- aggregate selected IoU/F1 仍为 `0.7422312/0.85204673`，没有一条样本严格提升；
- regional errors 从 v6 的 5 恶化到 7，atomic valid 从 `13/18` 降到 `11/18`。

所以“loss 更低”与“闭环更好”是两件事：前者只看训练目标拟合，后者要求模型在未参与训练的固定图像上正确诊断、生成正确动作，并让最终 mask 的 IoU/F1 真正提高。v7 只满足前者，没有满足后者。

## 为什么 v7 不进入 e2e5，以及最终结论

预先设定的进入 e2e5 门槛是：3/3 无退化、successful tool action ≥1、accepted candidate ≥1、至少一条样本的 IoU/F1 同时严格提升、action error=0；regional error 目标为 0，且至少不能差于 v3 的 1。v7 虽然 selected 3/3 无退化，但这是因为没有执行工具动作；它的 accepted=0、candidate=0、successful tool action=0、regional error=7、action attempt error=1、严格提升样本=0、IoU/F1 delta=0，明显不满足。因此不运行 v7 e2e5，也不把较低 loss 当成放大评测的理由。

已经取得的可复用成果是工程和数据合同层面的：

- 所有实际进入 SFT 的训练数据版本都通过 strict parser、scope、hash、parent 隔离和 Slurm 质量门；
- v1 的 6518 FP 大框灾难被 v2/v3 安全 hard negative 消除；
- 训练、合并模型、GT-free rollout、离线评分和 QQ 终态通知链路可复现；
- 失败轨迹已定位到 checklist 语义、监督目标权重/课程设计或 2B 模型能力，而不是继续修改冻结 runtime 就能解决的问题。

最终科学结论是负结果：在固定 v12.1 runtime、Qwen3-VL-2B、当前 LoRA 配置和 3 条 diagnostic 样本下，从 39 条起步并迭代到 63 条 unique / 83 条 exposure，仍未获得相对 base 的稳定提升。工程闭环、数据合同和安全回滚已经可复用，但继续重复或堆叠同类少样本的收益证据不足。v7 的协议统计与加权同时变化，但只有 3 个评测样本，不能把恶化完全归因于采样权重；能确定的是，这种 3×显式重复没有在当前闭环中解决问题。下一阶段应改变方法，例如真正的 per-sample/token loss weighting、结构化 curriculum、使用更强模型做 matched control，或先验证 2B 模型是否具备稳定学习该状态—checklist 合同的能力。

## Git、输出和主 worktree 保护

- v5 数据合同提交为 `1589dd7`，manifest 复验修复为 `48f2bd6/e011d51`；v6 数据合同提交为 `5069200`；v7 weighted sampling 提交为 `0f234d9`。冻结 runtime 锚点仍为 `491532f`。
- 正式输出均写入 `Chang-Agent-local-qwen3vl/outputs/CA_0801(*)` 新目录，没有覆盖旧实验。
- 原始 `Chang-Agent-local-qwen3vl` 主 worktree 中已有的 4 个 tracked deletion 和未跟踪 `doc/training_pipeline/0729_first_try/` 属于用户改动，交付时必须原样保留。
- `experiments/experiment_registry.json` 是登记源，`experiment_registry.xlsx` 由 builder 重建并用 `--check` 校验；两者都应在最终 commit 中同步。

## 权威证据路径

- v7 训练数据：`outputs/CA_0801(34)-qwen3vl-v121-action-data-v7-weighted-contrast/data/{manifest.json,validation.json,train.json,val.json}`
- v7 SFT：`outputs/CA_0801(35)-qwen3vl-v121-action-sft-v7-weighted-contrast/{training_summary.json,config_used.yaml,merged_model/}`
- v7 e2e3：`outputs/CA_0801(36)-qwen3vl-v121-action-e2e3-v7-weighted-contrast/{post_sft_summary.json,protocol_summary.json,evaluation_receipt.json,run/}`
- v6 对照：`outputs/CA_0801(31)`、`(32)`、`(33)`
- v5 对照：`outputs/CA_0801(28)`、`(29)`、`(30)`
- v3 行为对照：`outputs/CA_0801(16)`、`(17)`、`(18)`
- 交接与数据合同：`QWEN3VL_TRAINING_DATA_HANDOFF_0801.md`、`QWEN3VL_V121_BASELINE_DATA_0801.md`
