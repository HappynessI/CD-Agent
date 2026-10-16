# CA_0721(12)：Codex 作为 Verifier 的 GT-aware 反事实轨迹

## 目的与边界

本文回答的不是“本次 Qwen verifier 实际说了什么”，而是以下反事实问题：**若由
Codex 在同一 Environment 输入上充当 verifier，并且离线知晓 `label_cvt`，它应输出
哪些可执行 verdict/action；这些 action 交给当前执行器后，是否真的能提高 change
detection？**

这是一项 oracle 诊断，GT 在 rollout 完成后才读取。它不能作为线上结果或提示词中的
隐藏信息；它用来检验当前的 Agent—Verifier—Executor 接口是否有能力承载一个正确的
verdict。

- 输入实验：`outputs/CA_0721(12)-bailian-direct-rubric-v3`。
- 所有初始 `t1_mask.npy`/`t2_mask.npy` 与已归档的有效 run
  `change_agent_levir_gpu_closed_loop_20260719_151344` 逐文件 SHA256 完全相同。
- 复建使用当前 `overlap_presence`（threshold=0.25）和同样的 T1/T2 初始 mask；
  不修改初始化、不读取新模型结果。
- 坐标是 Agent/Verifier 公共的 normalized `[0,1000]` XY；括号内给出实际 pixel XY。

## 为什么这些 action 可以真正交给当前 Executor

推荐项全部是 `negative_point`，且点击时对应目标 T1/T2 mask 的 seed 为白色。
现有 `ActionExecutor` 对负点的最终组合规则是
`initial_mask & ~component_containing(initial_mask, point)`：它删除当前 mask 中该点所在的
4-connected 分量。SimpleClick 仍会被调用并做形状验证，但其预测不会参与负点的最终
mask composition。因此以下候选的像素删除、locality 和离线 IoU 是由已保存 mask 与
Executor 规则直接确定的，不是“希望工具会分得更好”的猜测。

所有所选分量都满足：

- GT overlap 为 0（只删除 FP，不减少 TP、也不增加 FN）；
- 选点在分量内，且整个分量位于 64×64 point ROI 内，`outside_roi_ratio=0`；
- 最大删除 408 px，`target_mask_change_ratio <= 408/65536`，远低于 0.25；
- component-count delta 为 -1，低于阈值 4。

故它们不会触发现有的 area/locality/component hard gate。对每步，Codex 的 candidate
verdict 应填入 `intended_error_improved=true`，三个 harm flag 都为 `false`；Direct runtime
因而会推导 `comparison="better"` 并接受候选。

## Codex 的初始 Direct-verifier 输出

下列为完整、可被当前 `DirectQwenVerifier` parser 接受的初始输出。`target_class_only=true`
表示 Codex 正确地把非建筑区域当作非目标，而不是把它们当作 building change。

### `test_20_15`

```json
{"verdict":{"rubric":{"evidence_sufficient":{"pass":true,"evidence":"The registered pair and masks are visually judgeable."},"target_class_only":{"pass":true,"evidence":"The selected isolated white region is not a building change."},"change_semantic_precision":{"pass":false,"evidence":"Unsupported white change pixels remain."},"change_semantic_recall":{"pass":false,"evidence":"Some true building-change pixels remain missing."},"changed_object_extent":{"pass":false,"evidence":"Not every changed building footprint is fully covered."},"change_boundary_alignment":{"pass":false,"evidence":"Some predicted boundaries include non-building pixels."},"change_artifact_control":{"pass":false,"evidence":"A disconnected three-pixel artifact is present."}},"candidate_effect":null,"error_type":"false_positive_change","target_view":"t2","suggested_action":"negative_point","coordinate_normalized_1000":[31,424],"box_normalized_1000":null,"feedback":"Remove the isolated T2 false-positive component at the supplied white seed."}}
```

Executor payload: `{"target_view":"t2","action":"negative_point","coordinate":[31,424]}`
→ pixel `[8,108]`，删除 3 FP，IoU `0.84638554 → 0.84683567`。

这是该样本唯一没有 TP 的白色 component；其余 component 都混有大量真变化。因而 Codex
在这一步后应**保守 abstain**，而不是强行删除大分量。

### `test_78_13`

```json
{"verdict":{"rubric":{"evidence_sufficient":{"pass":true,"evidence":"The registered pair and masks are visually judgeable."},"target_class_only":{"pass":true,"evidence":"The selected T1 component is not a building change."},"change_semantic_precision":{"pass":false,"evidence":"Several unmatched T1 components are pure false positives."},"change_semantic_recall":{"pass":false,"evidence":"Some building-change pixels remain missing."},"changed_object_extent":{"pass":false,"evidence":"Some changed-building extents remain incomplete."},"change_boundary_alignment":{"pass":false,"evidence":"Some change boundaries remain non-building aligned."},"change_artifact_control":{"pass":false,"evidence":"Disconnected false-positive components remain."}},"candidate_effect":null,"error_type":"false_positive_change","target_view":"t1","suggested_action":"negative_point","coordinate_normalized_1000":[894,502],"box_normalized_1000":null,"feedback":"Remove the largest isolated T1 false-positive component at the supplied white seed."}}
```

### `test_85_16`

```json
{"verdict":{"rubric":{"evidence_sufficient":{"pass":true,"evidence":"The registered pair and masks are visually judgeable."},"target_class_only":{"pass":true,"evidence":"The selected T2 component is a non-building false-positive region."},"change_semantic_precision":{"pass":false,"evidence":"Many white pixels are unsupported building change."},"change_semantic_recall":{"pass":false,"evidence":"Some changed-building pixels remain missing."},"changed_object_extent":{"pass":false,"evidence":"Some building extents are incomplete."},"change_boundary_alignment":{"pass":false,"evidence":"Some boundaries are not building aligned."},"change_artifact_control":{"pass":false,"evidence":"Disconnected non-building artifacts remain."}},"candidate_effect":null,"error_type":"false_positive_change","target_view":"t2","suggested_action":"negative_point","coordinate_normalized_1000":[725,467],"box_normalized_1000":null,"feedback":"Remove the largest isolated T2 non-building false-positive component at the supplied white seed."}}
```

## 完整的可执行轨迹与预期结果

候选 state 的 Direct JSON 与上面相同的 rubric/error/action 字段结构；唯一必须新增的是
`candidate_effect`：

```json
{"intended_error_improved":true,"introduced_false_positive":false,"introduced_false_negative":false,"boundary_or_artifact_worsened":false,"evidence":"The prior negative point removed only the isolated non-building component."}
```

这使每一步成为当前 Direct 二元候选协议下合法的 `better/accept=true`，而不是只在 GT
指标上变好却被运行时回滚的候选。

| 样本/step | Codex 交给 Executor 的 action JSON | 删除的目标分量 | TP / FP / FN（动作后） | IoU（动作后） |
| --- | --- | ---: | --- | ---: |
| `20_15` / 1 | `{"target_view":"t2","action":"negative_point","coordinate":[31,424]}` | 3 FP | 4777 / 610 / 254 | 0.84683567 |
| `78_13` / 1 | `{"target_view":"t1","action":"negative_point","coordinate":[894,502]}` | 254 FP | 11506 / 2992 / 430 | 0.77076635 |
| `78_13` / 2 | `{"target_view":"t1","action":"negative_point","coordinate":[918,467]}` | 177 FP | 11506 / 2815 / 430 | 0.78001500 |
| `78_13` / 3 | `{"target_view":"t1","action":"negative_point","coordinate":[102,129]}` | 135 FP | 11506 / 2680 / 430 | 0.78721949 |
| `85_16` / 1 | `{"target_view":"t2","action":"negative_point","coordinate":[725,467]}` | 408 FP | 1379 / 2555 / 156 | 0.33716381 |
| `85_16` / 2 | `{"target_view":"t2","action":"negative_point","coordinate":[576,784]}` | 262 FP | 1379 / 2293 / 156 | 0.36024033 |
| `85_16` / 3 | `{"target_view":"t2","action":"negative_point","coordinate":[796,467]}` | 260 FP | 1379 / 2033 / 156 | 0.38649103 |

`test_78_13` 的第一步也有真实历史执行佐证：相同初始 mask、相同 T1 135-pixel FP
component 的删除曾从 IoU 0.75787116 提升到 0.76467070。这里 Codex 先删更大的
254-pixel pure-FP component；三步都只改变已验证的纯 FP。

若按 `MAX_STEPS=1` 跑 `test_20_15`、按 `MAX_STEPS=3` 分别跑 `test_78_13` 和
`test_85_16`，三样本 oracle-selected 汇总为：TP=17662、FP=5323、FN=840、
**IoU=0.74132214**、Precision=0.76841418、Recall=0.95459950。相对本次 selected
IoU 0.69744116，提升 0.04388098，且没有样本退化。

### 正点与 box 的适用边界

这里没有把“点击必须落在白色 mask”推广到所有点动作。当前 Executor 的
`positive_point` 会使用 SimpleClick 的预测扩张 mask；历史有效 run 已显示从白色 seed
出发的正点可以改善 IoU。因此运行时只应拒绝白色 seed 的 `negative_point`（它必然是
component-removal no-op），而不能拒绝白色 seed 的 `positive_point`。

不过，当前 Direct 的 candidate-effect 协议把任何新 FP 都视作 harm。一个诚实的正点即使
净 IoU 上升，也可能同时新增少量 FP，因而只能报告 `uncertain`，不能被该二元协议 accept。
本 oracle 特意选择纯 FP 的负点，是因为它们同时满足“真实改进”和“当前协议可接受”。新正点
或 box 的效果仍须经 Slurm GPU 运行验证，不能从这份确定性的负点复建中外推。

## 这说明什么，尚未说明什么

结论是肯定的：**一个正确的 Codex verifier 输出可以交给当前 Executor，并实质提高
change detection。** 这不是 box 质量或 SimpleClick 分割能力的阻塞；所需负点编辑的
最终 mask 由 Executor 的确定性 component-removal 规则保证。

当前失败点是 verifier 的语义与动作选择：`CA_0721(12)` 在 `85_16` 错误地输出
`none/finish`，但同一输入含有至少 14 个白色、GT overlap 为零的 T2 components。对
`78_13`，它选择了会新增 868 FP 的大 SAM3 box，而不是 T1 中纯 FP component。

不过这不证明线上 Codex/Qwen 在不可见 GT 条件下必然能找出这些 components。它只建立了
两个事实：(1) 当前 state/action/executor 接口足以表达并执行正确修复；(2) 要获得线上
收益，verifier 必须从 RGB + masks 中可靠地识别“可删除且不含真变化”的 component，并让
candidate effect 如实报告净收益。

还有一个协议缺口：`test_20_15` 删除 3 px 后仍有错误，但没有第二个零风险 component。
Direct schema 目前不允许“仍有错误，但没有安全可执行 action”的 `defer/abstain`。若继续
按 `MAX_STEPS=3`，模型会被迫编造一个动作；实际实验应为该样本设 `MAX_STEPS=1`，或在
`VerifierOutput` 中新增无工具的保守 `defer` 终止状态。这个缺口与 GT 无关。

## 与原始 verifier 轨迹的区别

原始轨迹只证明 Qwen 能输出 JSON、box 和 finish；本文的 Codex 轨迹是以 GT 选择纯 FP
component 的 oracle action policy。它应被用作：

1. 一个离线 upper-bound / executor-compatibility test；
2. 训练或评估 verifier 的 action-ranking target；
3. 检验未来 GT-free verifier 是否能复现同样的 component 排序。

它不能直接被写入线上 prompt 或被宣称为 GT-free 实验结果。
