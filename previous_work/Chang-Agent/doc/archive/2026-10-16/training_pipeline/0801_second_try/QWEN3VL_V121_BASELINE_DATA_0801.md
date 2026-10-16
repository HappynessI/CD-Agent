# Qwen3-VL v12.1 少样本 baseline 数据方案（0801）

## 目标与冻结边界

本轮只训练冻结 v12.1 production live path 实际调用的 `audit` 与
`candidate_evidence` target。runtime 源码继续固定在 `491532f`，不恢复历史
`select`/`decision`/独立 `diagnosis` target，也不使用 0729 select-stage SFT32 数据。

数据只来自 LEVIR-CD `train_original` 和 `val_original`。`test_256` 五条固定诊断样本只用于
训练前后闭环评测，永不进入训练或 held-out 功能集。

## v1 少样本配方

首轮采用“少父图像、多生产协议变体”，预计 24–32 条 train、8–12 条 held-out val：

| 类型 | 建议占比 | 主要能力 |
|---|---:|---|
| standard atomic audit | 40% | correct、FP、FN 与边界/碎片 checklist 一致性 |
| RGB-only scan audit | 20% | 空预测下的 false-finish 与正确 finish 区分 |
| RGB-only evidence repair | 15% | 简短、物理内容锚定的 T1/T2 evidence |
| candidate_evidence | 25% | 实际改善与引入伤害的 delta 语义，减少保守误拒绝 |

同一人工核验区域可以产生多个 stage/scope 变体，但 train/val 按原始 1024×1024 parent
图像严格隔离。类别至少平衡：building→background 与 background→building、changed 与
unchanged、false positive 与 false negative、改善 candidate 与有害 candidate。

## 两阶段构建

1. `tools/prepare_qwen3vl_v121_curation.py` 在 GPU Slurm 作业中确定性选择候选 patch，加载
   冻结 SAM3 一次并保存 T1/T2 初始对象 mask。GT 仅用于候选选择和 curator preview，绝不
   输入 SAM3。
2. 人工查看候选 RGB/GT/SAM3 preview，只登记清晰的 T1/T2 building/background 状态和一条
   可见物理内容 evidence。正式 builder 复用 production `_stage_images`、`_stage_prompt` 与
   strict parser，基于冻结预测 mask 做局部、可追溯的腐蚀/增删/候选扰动。

0801 实际人工门保留 15/18 个候选：train 9 个 parent、val 6 个 parent。两个因目标太小或树冠
遮挡而剔除，一个冗余背景样本剔除；val 特意保留一条 building→background 拆除案例，避免训练
候选全部是新增建筑。每个 changed parent 构造 missing/correct/scan/evidence-repair/candidate-improvement
五种变体；每个 unchanged parent 构造 FP/scan-correct/candidate-harm 三种变体。

正式 manifest 必须记录 source/runtime commit、prompt hash、A/B/label/SAM3 mask hash、parent、
crop、scope、stage、attempt_kind、扰动规则、人工标签版本和 seed。模型输入中不包含 raw GT；
凡由 GT 离线确定的局部扰动和 checklist 必须显式标记，不能描述成纯模型预测事实。

## 质量门

- stage/attempt/scope 只接受冻结交接文档列出的闭集组合；target schema 必须逐条通过 runtime
  strict parser。
- evidence 不得出现 mask/presentation shortcut、占位符、坐标或 machine label。
- concrete FN 必须是 T1/T2 不同状态，concrete FP 必须是相同状态；checklist 派生 error type
  必须与构造类别一致。
- train/val parent、图像 hash 和 target hash 不得重复；任何 `test_256` 路径/hash 立即失败。
- base raw response 只作诊断，不自动写入 gold。
- 首轮仅在 held-out 功能门和固定 3 条闭环均优于 base 时进入 5 条闭环；若没有改善，先按错误
  类型增补最多一轮 hard cases，不盲目扩大样本量或修改冻结 runtime。

## 资源与执行

候选/SAM3 预处理使用 1 GPU；SFT 计划使用 2 GPU BF16 LoRA，最多 3 GPU；闭环评测使用 1 GPU。
所有计算、测试、预处理和训练均通过 Slurm。确认作业稳定运行后，训练监控采用 30 分钟窗口，
闭环评测采用 5 分钟窗口。正式输出统一写入开发仓库 `outputs/` 下新的、不可覆盖目录，并发送
QQ 终态通知。

独立干净 worktree 可以作为固定代码源，但 submitter 的 `PRIMARY_REPO`/正式输出根仍固定为
`Chang-Agent-local-qwen3vl/outputs`；这既保留主 worktree 中用户未提交的文档整理，也使所有
实验产物留在权威开发仓库目录。独立的
`tools/submit_qwen3vl_v121_pipeline_tests.sh` 支持同样的测试隔离方式，冻结通用测试入口保持
不变。

首轮 SFT 固定为 2×A800、BF16 LoRA rank 16/alpha 32、有效 batch 8、4 epoch、LR 1e-4 cosine
与 0.1 warmup；冻结视觉塔和多模态 projector。训练保留 adapter，并在同一 Slurm 作业内合并到
独立 `merged_model/`。后续 production 闭环直接把该目录作为 `--model-path`，不修改冻结
runtime 的模型加载逻辑。

多卡 worker 直接由固定 overlay Python 的 `torch.distributed.run` 执行 LLaMA-Factory
`launcher.py` 训练入口；不在已经建立的两个 rank 内再次调用 `llamafactory.cli`，避免嵌套
launcher 重新按 PATH 搜索另一个 `torchrun`。

## 闭环 hard-case v2

首轮 39 条 train 在固定 e2e3 上暴露出 production 尺度不匹配：6 条 changed scan 正例均围绕
GT component 裁剪，而 runtime 在空预测时审计 3×3 大范围 scan tile。首轮模型把 scan prompt
与 `background -> building` 绑定，在 `test_50_8` 的 9 个 tile 上普遍幻觉新增建筑，并接受了
覆盖背景的大 box candidate。该结果不进入 e2e5。

`--round v2-hardcase` 保留原 standard audit、evidence repair 和 parent 隔离，只做定向修正：
changed scan 正例保留 3 条较大且清晰的目标；从人工通过、GT change 为空的 train parent
`train_127` 按冻结 runtime 的精确 3×3 geometry 构造 9 条 production-scale scan correct；其中
s0/s4/s8 另构造 3 条大区域 candidate-harm。train 因此为 48 条，val 仍为 24 条，test 仍不参与
训练。第二轮从冻结 base 重新训练，保持 2×A800 和 LoRA 结构，但降为 2 epoch、LR 3e-5；仍以
同一 e2e3 样本顺序做第一道闭环门，只有无样本退化且主要指标改善才扩大到 e2e5。

实验 registry 的旧 0729 smoke 数值证据已不在当前输出根，workbook builder 现在在该历史
artifact 缺失时保留结构校验并显式跳过旧行数值复核；本轮 0801 的 Slurm 回执和 manifest
仍逐项保留，不把缺失历史文件伪装成新结果。

## 闭环 action hard-case v3（本轮）

v2 已经能安全地拒绝大范围背景框，但它把所有 candidate 都学成了“不要动”：12 个
candidate 样本全部是 `delta_added`，没有一个真实的 `delta_removed` 正例；另外三个 unchanged
parent 被用同一个 48×48 方块模拟，和线上 SimpleClick/SAM3 连通域不一致。v3 只修这两个合同
缺口，不改变冻结 runtime，也不同时更换优化器或学习率。

训练集仍然是少样本、按 parent 隔离的 59 条（val 保持 24 条）：

| 类型 | train 条数 | 作用 |
|---|---:|---|
| standard atomic audit | 22 | 正确、FN、建筑同态/背景场景下的 FP，以及时间反转后的 t1 目标 |
| production 3×3 scan audit | 9 | 3 条 changed 正例、3 条 unchanged 正例、3 条背景 hard negative |
| RGB-only evidence repair | 8 | 6 条原始 changed + 2 条时间反转 |
| candidate_evidence | 20 | 14 条真实改善（8×`delta_added`、6×`delta_removed`）和 6 条有害动作 |

candidate 的有害动作包括 3 条在真实 background→building 变化上误删 `delta_removed`，以及
3 条对背景 tile 的大 box；它们要求模型拒绝动作。改善动作包括原有的正点补回、2 条时间反转的
t1 正点、unchanged building 上删除伪变化后接受，以及 3 条背景 tile 删除伪变化后接受。unchanged transition 使用 `train_364`
和 `train_365` 的真实 SAM3 单侧连通域（面积 236、464、1205），不再生成固定 2304 像素方块。
背景 false-positive 的 checklist 同时标记 `target_class_only`、`white_pixels_supported` 和
`boundary_alignment` 为 fail，使“白色 proposal 但 T1/T2 都是背景”与建筑同态 FP 的语义边界明确。
每个 candidate 的 region、seed、增删像素和 T1/T2 transition facts 都由
`attach_verifier_regions`/`EvidenceRecord.from_proposal` 重新生成；builder 会对 59/24 行数、
stage/scope 分布、component 面积和 parent 隔离做硬校验。

预期语义状态分布（train）为：background→building 30、building→background 8、
building→building 8、background→background 13。实际 Slurm manifest/sha256 是最终依据；若硬校验
失败，本轮不进入 SFT。

v3 SFT 从冻结 base 重新开始，沿用 v2 的可比配置：2×A800、BF16 LoRA rank 16/alpha 32、dropout
0.05、effective batch 8、2 epochs、learning rate `3e-5`、cosine scheduler 和 0.1 warmup。
SFT worker 会把命令行实际覆盖的 dataset、样本数、学习率、epoch 和输出目录写入
`config_used.yaml`，避免历史 receipt 仍显示 39 条/4 epochs 的旧默认值。

## residual false-positive contract hard-case v4（已完成）

v3-bgfp 的失败轨迹显示，模型可以把 proposal 内 T1/T2 都识别成背景，却仍把七项 checklist
全部输出为 `pass`；冻结 parser 会以
`a white region with no target transition cannot pass as correct` 拒绝这种自相矛盾的结果。v4 不再
增加大范围 scan，也不修改 runtime，只补这个精确合同的两组三联样本。

两条 source 均来自 train-only curation parent，人工检查的是 SAM3 residual 的精确局部，而不是
原先被剔除的 GT candidate 框：

| candidate | 单侧 residual | 面积 | box xyxy | GT change overlap | RGB 结论 |
|---|---|---:|---|---:|---|
| `train_train_51_r3c0_changed` | T2 rank 0 | 1804 | `[124,130,184,181]` | 0 | T1 为裸地，T2 为施工车辆/裸地背景 |
| `train_train_85_r0c0_changed` | T1 rank 1 | 258 | `[74,20,91,41]` | 0 | 两期都是树冠背景 |

builder 对 direction/rank/area/box/GT overlap 全部 fail closed，并在 manifest 记录 component
SHA256。每个 residual 只生成三条：首轮 standard atomic FP audit；带上述 production 原始
validation error 的 full-context standard repair replay；执行 `negative_point` 后的真实
`delta_removed` candidate improvement。repair 不是 evidence-only scope，gold 仍是完整 atomic
target。正确 checklist 固定为 evidence sufficient 和 target-class scope `pass`，white-pixel
support 与 boundary alignment `fail`，holes/extent `not_applicable`，fragment `pass`；parser 必须
机械派生 `false_positive_change`、唯一白 seed view 和 `negative_point`。

因此 v4 是 v3 的最小 `+6 train / +0 val` 增量，预期为 train 65、val 24；train stage 分布为
audit 43、candidate evidence 22，scope 分布为 standard 48、scan 9、evidence-only repair 8。
代码只新增 `--round v4-residual`，保留 v1/v2/v3 输出合同。CPU Slurm 的 unit test、
确定性重建和 manifest 硬门已通过，实际结果如下：

- Job `52256` 完成 v4 数据构建，确认 `65 train / 24 val`；dataset manifest SHA256 为
  `<hash omitted>`。
- Job `52258` 完成 2×A800 SFT，train loss 为 `0.4197051591`；merged model SHA256 为
  `<hash omitted>`。
- Job `52259` 完成固定顺序的 e2e3，verifier-selected IoU/F1 为
  `0.7422312 / 0.85204673`，accepted candidate 为 `0`，regional validation errors 为 `7`。
  `test_20_15` 的 `r1`–`r7` 仍因“T1/T2 同为 background 但 checklist 全 pass”被拒绝，
  未产生可执行 action。

v4 的最终 IoU/F1 与 base 持平，没有 accepted candidate，也没有达到闭环稳定提升的验收条件；
而 regional errors 还增加到 7 条。因此本轮停止继续扩充相似训练数据，不进入 e2e5，后续应转向
检查少样本 token 分布、训练轮数/损失权重或模型能力边界，而不是继续重复同类数据。

## contrast repair v5（已完成）

在用户确认继续后，v5 针对 v4 最明显的标签冲突做了一轮受控修复。v4 residual 的 RGB 结论是
`background -> background`，但旧 target 仍把 `target_class_only` 写成 `pass`；v5 将它统一为
`false_positive_background`，即 `target_class_only`、`white_pixels_supported` 和
`boundary_alignment` 都为 `fail`。此外加入两条真实 `background -> building` finish repair，
以及两条 unchanged-building actionable repair，形成 69 train / 24 val。

Job `52307` 构建出的 train 分布为：standard audit 30、scan 9、evidence-only repair 8、
candidate evidence 22；语义状态为 background→building 32、background→background 19、
building→building 10、building→background 8；candidate improvement/harm 为 16/6。
manifest SHA256 为
`<hash omitted>`。

首次后置 manifest 复验 Job `52311` 失败，并不是数据错误：测试用 `EvidenceRecord.from_proposal()`
重建已经序列化的 Environment record，丢失了权威 `editable_seed_white`，所以原本应定位到 T2 的
`target_view` 变成 `None`。修复为按完整 envelope 直接重建后，Jobs `52315`、`52319` 均通过；
后者还把 manifest gate 固定为数据构建后的显式步骤，避免误读历史 output。

Job `52326` 用 2×A800、2 epochs 从同一冻结 base 重新训练，loss 为 `0.4225744117`，merged model
SHA256 为 `<hash omitted>`。
Job `52351` 的 e2e3 仍为 IoU/F1 `0.7422312 / 0.85204673`，3/3 无退化但零提升；首次得到一个
可执行 `negative_point` 和一个成功 tool action，却删错了真实变化区域。candidate 只减少 16 FP，
同时增加 324 FN，`test_20_15` IoU 从 `0.84638554` 降到 `0.79122246`，被冻结 hard gate 拒绝并
回滚。最终 accepted=0、regional validation errors=6，因此仍不进入 e2e5。

## matched contrast v6（已完成）

v5 证明“动作链打通”还不等于“动作正确”。v6 不再继承 69 条的累积配方，而回到行为最好的
Job `52146` 的 53 条 v3 核心，再加入 10 条一一匹配的对照：

| 对照来源 | T1/T2 方向 | 增量 | 监督目的 |
|---|---|---:|---|
| `train_51` residual | T2，background→background | 3 | FP audit、同报错 repair、正确删除为 better |
| `train_85` residual | T1，background→background | 3 | FP audit、同报错 repair、正确删除为 better |
| `train_407` 真实变化 | T2，background→building | 2 | 同报错 repair 必须 finish、误删为 worse |
| `train_231` 时间反转 | T1，building→background | 2 | 同报错 repair 必须 finish、误删为 worse |

三组 production 大背景 tile 的 FP audit/removal 共 6 条从 v6 排除，但 v3 的 9 条 scan 和 3 条
grid-harm candidate 保留。builder 用 Job `52132` 的 53 条核心逐行 prompt、target、image hash
计算可复现 canonical receipt
`<hash omitted>`；历史交接中另记的
`e10dee1...` receipt 因当时未保存序列化算法，只作为审计来源保留，不替代可复现硬门。

v6 预期为 63 train / 24 val、train/val parent 11/6。train 分布固定为：standard audit 25、
scan 9、evidence-only repair 8、candidate evidence 21；语义状态 background→building 32、
background→background 13、building→background 10、building→building 8；candidate
improvement/harm 13/8。构建器必须机械证明 residual repair 产生正确 T1/T2 `negative_point`、
删除 FP 为 `better`，同时真实变化 repair 产生 `finish`、误删真实变化为 `worse`。数据 worker
在全新输出目录构建后显式执行 `QWEN3VL_V121_V6_MANIFEST` gate；只有该门通过才允许 SFT。

v6 的 e2e3 准入条件固定为：3/3 无退化、successful tool action ≥1、accepted candidate ≥1、
至少一条样本 IoU/F1 同时严格提升、action error=0，regional error 目标为 0 且不得差于 v3 的 1。

实际结果：

- Job `52427` 构建成功，确认 `63 train / 24 val`、train/val parent `11/6` 且无 overlap；
  manifest SHA256 为
  `<hash omitted>`。上述任务、语义、candidate
  和 audit-kind 分布均与硬门完全一致，53 条冻结 core 的 canonical receipt 为
  `<hash omitted>`。
- Job `52428` 用 2×A800、2 epochs 完成 16 optimizer steps，loss 为 `0.42206590995`；adapter
  SHA256 为 `<hash omitted>`，merged model
  SHA256 为 `<hash omitted>`。
- Job `52429` 在 1×A800 上完成固定 e2e3。selected IoU/F1 仍为
  `0.7422312 / 0.85204673`，delta 为 0；tool action `1`、candidate `1`、accepted `0`、rejected
  `1`、action attempt error `1`。regional errors 为 `5`、atomic valid 为 `13/18`，仅比 v5 的
  `6` 和 `12/18` 略好，仍远差于 v3 的 1 条 error。

错误动作位于 `test_20_15/r4`：模型生成 T2 `negative_point`，工具删掉 360 像素 component，TP
从 4777 降至 4417，FP 保持 613，FN 从 254 升至 614，candidate IoU/F1 从
`0.84638554/0.91680261` 降至 `0.78260099/0.87804393`。冻结 runtime 的
`point_proposal_combined_change_gate_exceeded` 拒绝并回滚，所以最终 selected 没有退化，但也不是
提升。`r1/r2/r3/r6/r7` 仍为 raw `background/background + all-pass`；`test_1_1` 保留 FP626，
`test_50_8` 保留 FN166。v6 没有通过门，因此不运行 e2e5，并触发下述采样权重试验。

## weighted contrast v7（已完成，停止继续扩充同类数据）

v7 不扩大 unique 数据集：仍使用 v6 的 63 条不同记录和 24 条 val。唯一变量是训练采样曝光：
给 10 条带 `v6_extension` 标记的 matched-contrast 样本各设置总重复因子 3，即每条在原有 1 次之外
再显式重复 2 次。这样 unique train 仍为 63，但 SFT 实际读取 `63 + 10×2 = 83` 条 exposure。

该设计用于区分两种解释：如果 v7 改善，说明 v6 的对照方向可能正确但权重不足；如果仍失败，
继续添加同类 unique 样本的依据更弱，应转向真正的 loss weighting、结构化 curriculum 或
Qwen3-VL-2B 能力边界验证。为了保持因果可比性，base、LoRA、2 epochs、learning rate `3e-5`、
有效 batch 8、seed 801、冻结模块、runtime 和 e2e3 顺序均不变。

正式输出目录为：

- `outputs/CA_0801(34)-qwen3vl-v121-action-data-v7-weighted-contrast`；
- `outputs/CA_0801(35)-qwen3vl-v121-action-sft-v7-weighted-contrast`；
- `outputs/CA_0801(36)-qwen3vl-v121-action-e2e3-v7-weighted-contrast`。

实际结果：

- Job `52433` 构建成功，确认 `83 train exposures / 63 unique train contents / 24 val`，train/val
  parent 为 `11/6` 且无 overlap。train exposure 分布为 standard audit 37、scan 9、evidence repair
  8、candidate evidence 29；语义状态为 background→building 36、background→background 25、
  building→background 14、building→building 8；candidate improvement/harm 为 17/12。
  manifest SHA256 为
  `<hash omitted>`，63 条 unique receipt 为
  `<hash omitted>`，10 条 weighted source
  receipt 为 `<hash omitted>`。
- Job `52434` 用 2×A800、2 epochs 完成 22 optimizer steps；83 条 exposure 的 loss 为
  `0.3760475462`，adapter SHA256 为
  `<hash omitted>`，merged model SHA256 为
  `<hash omitted>`。
- Job `52436` 在 1×A800 上完成固定 e2e3。selected IoU/F1 仍为
  `0.7422312 / 0.85204673`，delta 为 0，3/3 selected 无退化但严格提升样本数为 0；accepted、
  rejected、candidate、successful/tool action 均为 0，action attempt/error 为 `1/1`，finish 为
  2。regional errors 恶化到 7，atomic valid 降到 `11/18`。

`test_20_15` 的 8 个 region 中只有 1 个 diagnosis 为 `none`，其余 7 个都被降级为
`uncertain_region`；repair 也没有形成可定位动作，最终 `no_executable_action`，初始 FP613/FN254
原样保留。repair 后 `r1/r2/r3/r4/r6/r7` 仍输出 `background/background + all-pass`；`r5` 在已有
白区上只报 extent fail，被拒绝为不可能的纯 false negative。7 个 fail-closed 区域令 `plan=null`，
所以 action attempt error 表示 verifier 没有授权动作，并非工具执行失败。`test_1_1` 仍错误 finish
并留下 FP626；`test_50_8` 仍错误 finish 并留下 FN166。

因此，loss 从 v6 的约 `0.4221` 降至约 `0.3760` 并不代表闭环能力提高。把 10 条 matched contrast
显式重复到总曝光 3 次，没有产生 accepted candidate、动作或任何 IoU/F1 提升，反而让 regional
errors 从 v6 的 5 恶化到 7。v7 未通过预设门，不运行 e2e5；本轮停止继续堆同类数据，下一阶段应
转向真正的 loss weighting、结构化 curriculum、更强模型对照或验证 Qwen3-VL-2B 的能力边界。
由于这里只评测 3 个 diagnostic 样本，不能把协议恶化完全归因于重复权重；可以确定的是，3×显式
重复没有在当前闭环中带来改善。
