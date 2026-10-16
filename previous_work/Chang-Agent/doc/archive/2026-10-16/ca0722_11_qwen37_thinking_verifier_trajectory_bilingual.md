# CA_0722(11) `test_85_16`：Verifier 完整轨迹（中英文对照）

## 1. 文档范围 / Scope

本文逐项还原 `CA_0722(11)-bailian-proposal-qwen37-thinking-v9` 中
`test_85_16.png` 的完整 verifier 归档轨迹。之所以选择这个样本，是因为它最清楚地暴露了
本轮失败：离线 IoU 只有 `0.30658070`、precision 只有 `0.31759558`，但 verifier 对三个
已检查区域都给出 `error_type="none"`，最终没有授权任何 mask 编辑。

This document reconstructs the complete persisted verifier trace for `test_85_16.png`
from `CA_0722(11)-bailian-proposal-qwen37-thinking-v9`. This sample is selected because
it exposes the failure most clearly: its offline IoU is only `0.30658070` and its
precision is only `0.31759558`, yet the verifier labels all three inspected regions as
`error_type="none"` and authorizes no mask edit.

数据来源 / Sources:

- 原始轨迹 / raw trajectory:
  `outputs/CA_0722(11)-bailian-proposal-qwen37-thinking-v9/proposal/trajectories/test_85_16/trajectory.json`
- episode 汇总 / episode summary:
  `outputs/CA_0722(11)-bailian-proposal-qwen37-thinking-v9/proposal/trajectories/test_85_16/episode_summary.json`
- 离线指标 / offline metrics:
  `outputs/CA_0722(11)-bailian-proposal-qwen37-thinking-v9/proposal/per_sample_metrics.json`

真实性边界 / Fidelity boundary:

- 轨迹保存了全部 8 次 backend call 的 stage、延迟、token 用量、prompt SHA256、request ID、
  原始文本响应及解析结果；下文逐次完整列出。
- The trace persists the stage, latency, token usage, prompt SHA256, request ID, raw
  text response, and parsed output for all eight backend calls. Every call is reproduced
  below.
- 归档**没有保存明文 prompt**，只保存 `prompt_sha256` 和结构化输入证据。因此本文能够完整
  展示“已归档轨迹”，但不能把 prompt 哈希反推出当时的逐字 prompt。
- The archive does **not** persist plaintext prompts. It stores only each
  `prompt_sha256` and the structured evidence. Therefore this is the complete persisted
  trace, not a verbatim reconstruction of unavailable prompt text.
- GT 在 rollout 期间不可见；本页末尾的 IoU/FP 等指标是在 rollout 完成后离线加入的。
- Ground truth was unavailable during rollout. IoU/FP metrics shown at the end were
  attached only by post-rollout offline evaluation.
- 用户最初指向的 `CA_0722(12)` 在 `Environment.reset` 时因
  `max_completion_tokens=512 < thinking_budget=1024` 失败，没有 verifier 轨迹。因此本文使用
  同模型、thinking 已开启且成功完成的 `(11)`。
- The originally referenced `CA_0722(12)` failed during `Environment.reset` because
  `max_completion_tokens=512 < thinking_budget=1024`, so it contains no verifier trace.
  This document uses the completed `(11)` run with the same model and thinking enabled.

## 2. 运行配置 / Run configuration

| 配置 / Field | 值 / Value |
| --- | --- |
| 样本 / sample | `test_85_16.png` |
| 查询类别 / query | `building` |
| Agent 与 verifier 模型 / model | `qwen3.7-plus-2026-05-26` |
| Verifier | `qwen_staged` |
| 决策协议 / decision mode | `qwen_staged_deterministic_target_resolution_v9` |
| Proposal mode | `proposal` |
| Thinking | enabled, budget `256` per call |
| Sampling | `do_sample=false`, repetition penalty `1.05` |
| 初始审计目录 / initial audit catalog | `19` 个 region |
| 初始每批上限 / initial regions per batch | `1`（run manifest 的 `verifier_max_initial_regions_per_batch`；本轨迹的 selection stage 仍选择 3 个 region） |
| 单次最多选择 / max selected regions | `3` |
| 最小视觉置信度 / min visual confidence | `0.6` |
| 匹配方式 / matching | `overlap_presence`, threshold `0.25` |
| 坐标 / coordinates | verifier: normalized XY `[0,1000]`; environment: pixel XY |
| GT during rollout | `false` |
| Git commit | `381ac3a3941b498447baa7c780beecd410b2c2ef`（dirty worktree） |

整条 episode 耗时 `90.604 s`。其中 8 次模型调用的记录延迟合计 `73.355583 s`，记录的
token 合计为 prompt `25,567`、completion `2,517`、total `28,084`；各调用记录的
`reasoning_tokens` 合计 `2,032`。

The entire episode took `90.604 s`. The eight model calls account for `73.355583 s`
of recorded latency and, as reported by the provider, `25,567` prompt tokens, `2,517`
completion tokens, and `28,084` total tokens. Recorded `reasoning_tokens` sum to `2,032`.

## 3. 轨迹总览 / End-to-end timeline

| 顺序 / Order | 阶段 / Stage | 发生的事情 / What happened |
| ---: | --- | --- |
| 0 | Environment reset | 从初始 T1/T2 mask 构建 19 个白色 change-mask audit regions。 / Build 19 white change-mask audit regions from the initial T1/T2 masks. |
| 1 | `select` | Qwen 选择 `r0,r1,r2`，达到单次最多 3 个区域的上限。 / Qwen selects `r0,r1,r2`, reaching the per-call limit of three regions. |
| 2 | `evidence(r0)` | 判断 `background → building`，置信度 `0.95`。 / Judge `background → building`, confidence `0.95`. |
| 3 | `diagnosis(r0)` | 输出 `error_type=none`。 / Return `error_type=none`. |
| 4 | `evidence(r1)` | 判断 `background → building`，置信度 `0.90`。 / Judge `background → building`, confidence `0.90`. |
| 5 | `diagnosis(r1)` | 输出 `error_type=none`。 / Return `error_type=none`. |
| 6 | `evidence(r2)` | 判断 `background → building`，置信度 `0.95`。 / Judge `background → building`, confidence `0.95`. |
| 7 | `diagnosis(r2)` | 输出 `error_type=none`。 / Return `error_type=none`. |
| 8 | `decision` | 给初始状态质量分 `0.95`，称 16 个区域仍未审计。 / Score the initial state `0.95` while noting 16 regions remain unaudited. |
| 9 | Runtime completion gate | 因只审计 `3/19`，禁止 `finish`；plan 被清空，输出 `uncertain_region`，但没有动作和坐标。 / Reject `finish` because only `3/19` regions were audited; clear the plan and emit `uncertain_region` without an action or coordinates. |
| 10 | Action execution | 因 `suggested_action=null`，执行器报告 “verifier did not authorize an executable action”。 / Because `suggested_action=null`, the executor reports “verifier did not authorize an executable action.” |
| 11 | Episode stop | `action_retry_exhaustion_without_state_change`；只有 `step_000`，mask 未变化。 / Stop with `action_retry_exhaustion_without_state_change`; only `step_000` exists and the mask is unchanged. |

## 4. Environment 提供的完整区域目录 / Complete region catalog

19 个区域的公共结构均为：`audit_kind=present`、`change_mask_state=white`、
`temporal_difference_state=present`、`t1` seed 为黑、`t2` seed 为白，并允许
`positive_point`、`negative_point`、`box`。表内坐标均为 normalized `[0,1000]` XY。

All 19 regions share these fields: `audit_kind=present`, `change_mask_state=white`,
`temporal_difference_state=present`, a black T1 seed, a white T2 seed, and allowed actions
`positive_point`, `negative_point`, and `box`. All coordinates below use normalized
`[0,1000]` XY.

| ID | box `[x1,y1,x2,y2]` | component seed `[x,y]` | T1 mask px | T2 mask px | component area | 是否被审计 / Audited |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `r0` | `[376,239,604,443]` | `[459,314]` | 0 | 684 | 684 | 是 / yes |
| `r1` | `[647,310,792,498]` | `[737,420]` | 0 | 408 | 408 | 是 / yes |
| `r2` | `[314,812,475,933]` | `[380,875]` | 0 | 472 | 401 | 是 / yes |
| `r3` | `[780,541,894,682]` | `[851,592]` | 0 | 287 | 287 | 否 / no |
| `r4` | `[541,761,678,871]` | `[584,812]` | 0 | 270 | 262 | 否 / no |
| `r5` | `[776,384,882,498]` | `[831,431]` | 0 | 260 | 260 | 否 / no |
| `r6` | `[678,875,773,988]` | `[733,914]` | 0 | 214 | 214 | 否 / no |
| `r7` | `[847,871,988,984]` | `[890,918]` | 0 | 241 | 208 | 否 / no |
| `r8` | `[380,922,510,1000]` | `[463,957]` | 0 | 173 | 173 | 否 / no |
| `r9` | `[937,773,1000,863]` | `[980,812]` | 0 | 164 | 164 | 否 / no |
| `r10` | `[467,627,584,714]` | `[506,663]` | 0 | 210 | 162 | 否 / no |
| `r11` | `[498,576,612,663]` | `[533,608]` | 0 | 267 | 161 | 否 / no |
| `r12` | `[941,502,1000,596]` | `[980,549]` | 0 | 160 | 160 | 否 / no |
| `r13` | `[400,710,498,792]` | `[439,745]` | 0 | 178 | 158 | 否 / no |
| `r14` | `[525,529,643,616]` | `[565,565]` | 0 | 217 | 156 | 否 / no |
| `r15` | `[635,702,718,788]` | `[671,737]` | 0 | 153 | 153 | 否 / no |
| `r16` | `[376,765,490,855]` | `[412,796]` | 0 | 233 | 146 | 否 / no |
| `r17` | `[792,310,867,388]` | `[831,349]` | 0 | 102 | 102 | 否 / no |
| `r18` | `[855,957,933,1000]` | `[890,988]` | 0 | 85 | 83 | 否 / no |

匹配证据还记录：`t1_instance_count=0`、`t2_instance_count=19`、`candidate_pairs=[]`、
`split_merge_ambiguity=false`。

The matching evidence additionally records `t1_instance_count=0`,
`t2_instance_count=19`, `candidate_pairs=[]`, and `split_merge_ambiguity=false`.

## 5. 八次 backend call 的完整记录 / All eight backend calls

### Call 1 — 区域选择 / Region selection

- Stage: `select`
- Latency: `9.726062 s`
- Request ID: `chatcmpl-2ec80996-de97-9a12-941e-2ffc44bc22f0`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `2709`, completion `300`, total `3009`; reasoning `256`, text `44`; image `194`, text prompt `2515`

Raw response / 原始响应：

```json
{"selection":{"region_ids":["r0","r1","r2"],"reason":"Largest change-mask components with significant area discrepancies between component_area and t2_mask_pixels, indicating potential segmentation errors"}}
```

中文对照：选择 `r0、r1、r2`；理由是“这些是最大的 change-mask 分量，且
`component_area` 与 `t2_mask_pixels` 存在显著差异，说明可能有分割错误”。需要注意，原始
目录中 `r0` 的两者均为 `684`，`r1` 的两者均为 `408`，只有 `r2` 是 `401` 对 `472`；
因此这条选择理由与目录数值并不完全一致。

English note: select `r0,r1,r2` because they are said to be the largest components with
significant discrepancies between `component_area` and `t2_mask_pixels`. However, the
catalog reports exact equality for `r0` (`684/684`) and `r1` (`408/408`); only `r2`
differs (`401/472`). The stated rationale is therefore not fully consistent with the
catalog values.

### Call 2 — `r0` 视觉证据 / Visual evidence

- Stage: `evidence`
- Latency: `9.307964 s`
- Request ID: `chatcmpl-cc0a0cba-4c4e-9690-9ff8-0d9b4ec2c344`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `3295`, completion `328`, total `3623`; reasoning `256`, text `72`; image `564`, text prompt `2731`

Raw response / 原始响应：

```json
{
  "region_id": "r0",
  "visual_judgment": {
    "t1_state": "background",
    "t2_state": "building",
    "visual_confidence": 0.95,
    "evidence_quality": "clear"
  }
}
```

中文对照：`r0` 在 T1 是背景、T2 是建筑，视觉置信度 `0.95`，证据清晰。

English: `r0` is background in T1 and building in T2, with `0.95` visual confidence and
clear evidence.

### Call 3 — `r0` 诊断 / Diagnosis

- Stage: `diagnosis`
- Latency: `9.126106 s`
- Request ID: `chatcmpl-d7c0c0ef-3fe7-9adf-90a0-61570ce93ae8`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `3442`, completion `297`, total `3739`; reasoning `244`, text `53`; image `564`, text prompt `2878`

Raw response / 原始响应：

```json
{
  "region_id": "r0",
  "diagnosis": {
    "error_type": "none",
    "target_view": null,
    "confidence": 0.95
  }
}
```

中文对照：`r0` 没有错误，不需要目标视图，诊断置信度 `0.95`。

English: diagnose no error in `r0`; no target view is needed; confidence `0.95`.

### Call 4 — `r1` 视觉证据 / Visual evidence

- Stage: `evidence`
- Latency: `9.368443 s`
- Request ID: `chatcmpl-57c85867-758a-9c0e-9b1d-aa072ec3c2eb`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `3335`, completion `327`, total `3662`; reasoning `256`, text `71`; image `604`, text prompt `2731`

Raw response / 原始响应：

```json
{
  "region_id": "r1",
  "visual_judgment": {
    "t1_state": "background",
    "t2_state": "building",
    "visual_confidence": 0.9,
    "evidence_quality": "clear"
  }
}
```

中文对照：`r1` 在 T1 是背景、T2 是建筑，视觉置信度 `0.90`，证据清晰。

English: `r1` is background in T1 and building in T2, with `0.90` visual confidence and
clear evidence.

### Call 5 — `r1` 诊断 / Diagnosis

- Stage: `diagnosis`
- Latency: `8.872339 s`
- Request ID: `chatcmpl-4935d9f5-1159-9f1c-b918-56ec67898cfc`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `3481`, completion `309`, total `3790`; reasoning `256`, text `53`; image `604`, text prompt `2877`

Raw response / 原始响应：

```json
{
  "region_id": "r1",
  "diagnosis": {
    "error_type": "none",
    "target_view": null,
    "confidence": 0.9
  }
}
```

中文对照：`r1` 没有错误，不需要目标视图，诊断置信度 `0.90`。

English: diagnose no error in `r1`; no target view is needed; confidence `0.90`.

### Call 6 — `r2` 视觉证据 / Visual evidence

- Stage: `evidence`
- Latency: `8.967392 s`
- Request ID: `chatcmpl-daa0f1a4-e570-9843-b181-bcf757899bca`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `3285`, completion `328`, total `3613`; reasoning `256`, text `72`; image `554`, text prompt `2731`

Raw response / 原始响应：

```json
{
  "region_id": "r2",
  "visual_judgment": {
    "t1_state": "background",
    "t2_state": "building",
    "visual_confidence": 0.95,
    "evidence_quality": "clear"
  }
}
```

中文对照：`r2` 在 T1 是背景、T2 是建筑，视觉置信度 `0.95`，证据清晰。

English: `r2` is background in T1 and building in T2, with `0.95` visual confidence and
clear evidence.

### Call 7 — `r2` 诊断 / Diagnosis

- Stage: `diagnosis`
- Latency: `10.603720 s`
- Request ID: `chatcmpl-dc13b0a5-f54e-9c85-950d-50676a8ef5dd`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `3432`, completion `305`, total `3737`; reasoning `252`, text `53`; image `554`, text prompt `2878`

Raw response / 原始响应：

```json
{
  "region_id": "r2",
  "diagnosis": {
    "error_type": "none",
    "target_view": null,
    "confidence": 0.95
  }
}
```

中文对照：`r2` 没有错误，不需要目标视图，诊断置信度 `0.95`。

English: diagnose no error in `r2`; no target view is needed; confidence `0.95`.

### Call 8 — 初始状态决策 / Initial-state decision

- Stage: `decision`
- Latency: `7.383557 s`
- Request ID: `chatcmpl-3233ef84-b0a5-9b01-84a6-c8060732da1a`
- Prompt SHA256: `<hash omitted>`
- Usage: prompt `2588`, completion `323`, total `2911`; reasoning `256`, text detail `323`; text prompt `2588`

Raw response / 原始响应：

```json
{"decision":{"quality_score":0.95,"feedback":"All audited regions (r0-r2) show no errors with high confidence. Remaining 16 regions (r3-r18) are unaudited but exhibit consistent present-state characteristics with white change masks and valid seed points."}}
```

中文对照：质量分为 `0.95`。所有已审计区域 `r0-r2` 都以高置信度判断为无错误；其余
16 个区域 `r3-r18` 尚未审计，但呈现一致的 present-state 特征，拥有白色 change mask 和
有效 seed。

English: quality score `0.95`. All audited regions `r0-r2` show no errors with high
confidence. The remaining 16 regions are unaudited but appear to have consistent
present-state characteristics, white change masks, and valid seeds.

说明：provider 在本次 decision call 中记录 `completion_tokens=323`，同时 details 记录
`reasoning_tokens=256`、`text_tokens=323`；details 之和大于 completion 总数。本文忠实保留
provider 字段，不自行修正。

Note: for this decision call, the provider reports `completion_tokens=323` while its
details report `reasoning_tokens=256` and `text_tokens=323`, whose sum exceeds the stated
completion total. This document preserves the provider fields without silently fixing
them.

## 6. Runtime 后处理与最终 VerifierOutput / Runtime post-processing

模型调用结束时的三个 judgment/diagnosis 为：

The three judgments and diagnoses after the model calls are:

| Region | T1 → T2 | Evidence | Diagnosis | Confidence |
| --- | --- | --- | --- | ---: |
| `r0` | `background → building` | `clear` | `none` | `0.95` |
| `r1` | `background → building` | `clear` | `none` | `0.90` |
| `r2` | `background → building` | `clear` | `none` | `0.95` |

由于三个 diagnosis 都是 `none`，动作计划原本等价于 `finish`。但 v9 completion gate 要求
Environment 的**每一个** audit region 都经过 judgment 和 diagnosis。本次只选择了 3 个，而
目录共有 19 个，因此 gate 返回：

Because all three diagnoses are `none`, the action plan initially amounts to `finish`.
However, the v9 completion gate requires **every** Environment audit region to receive a
judgment and diagnosis. Only three of 19 were selected, so the gate returns:

```text
state_completion_gate_passed = false
state_completion_gate_reason = "audited 3 of 19 region(s); unselected regions are not evidence of correctness"
```

Runtime 随后把未审计区域代表设为 `uncertain_region`，清空 plan，并形成以下公共输出。
注意 `candidate_accepted=true` 只表示 reset 的初始状态被保存为当前基线，不表示执行或接受
了一个修复动作。

The runtime then represents the unaudited remainder as `uncertain_region`, clears the
plan, and produces the public output below. `candidate_accepted=true` means only that the
reset state is retained as the baseline; it does not mean that a corrective action was
executed or accepted.

```json
{
  "quality_score": 0.95,
  "progress_score": 0.0,
  "score_delta": 0.0,
  "comparison": "initial",
  "error_type": "uncertain_region",
  "target_view": "t2",
  "error_region": null,
  "coordinate_space": "normalized_0_1000",
  "suggested_action": null,
  "feedback": "All audited regions (r0-r2) show no errors with high confidence. Remaining 16 regions (r3-r18) are unaudited but exhibit consistent present-state characteristics with white change masks and valid seed points.; completion gate: audited 3 of 19 region(s); unselected regions are not evidence of correctness",
  "accept": false,
  "verifier_valid": true,
  "localization_valid": false,
  "stop": false
}
```

这形成一个无法继续的状态：`stop=false` 表示不能结束；`suggested_action=null` 和
`error_region=null` 又表示没有可执行动作。runner 尝试进入第一个 loop 后立即得到：

This creates a non-progressing state: `stop=false` means the episode may not finish,
while `suggested_action=null` and `error_region=null` mean there is no executable action.
On entering loop 1, the runner immediately records:

```json
{
  "loop_index": 1,
  "error": "verifier did not authorize an executable action"
}
```

Episode 最终记录 / Final episode record:

```json
{
  "status": "success",
  "stop_reason": "action_retry_exhaustion_without_state_change",
  "loop_count": 1,
  "candidate_count": 0,
  "accepted_candidate_count": 0,
  "rejected_candidate_count": 0,
  "tool_action_count": 0,
  "action_attempt_count": 1,
  "invalid_action_attempt_count": 1,
  "selected_step": 0,
  "verifier_best_step": 0
}
```

这里的 `status="success"` 表示程序正常写完 episode 文件，并不表示 verifier 成功改善 mask。

Here `status="success"` means the program completed and wrote the episode files; it does
not mean the verifier successfully improved the mask.

## 7. Rollout 后离线结果 / Post-rollout offline result

只存在 `step_000`，没有 candidate step。初始、last、selected 和 verifier-best 都指向同一
mask：

Only `step_000` exists; there is no candidate step. Initial, last, selected, and
verifier-best all point to the same mask:

| Metric | Initial | Last/Selected | Delta |
| --- | ---: | ---: | ---: |
| IoU | `0.30658070` | `0.30658070` | `0` |
| Precision | `0.31759558` | `0.31759558` | `0` |
| Recall | `0.89837134` | `0.89837134` | `0` |
| F1 | `0.46928705` | `0.46928705` | `0` |
| TP | `1379` | `1379` | `0` |
| FP | `2963` | `2963` | `0` |
| FN | `156` | `156` | `0` |
| TN | `61038` | `61038` | `0` |

Mask hashes / mask 哈希：

```text
t1_mask_sha256     = <hash omitted>
t2_mask_sha256     = <hash omitted>
change_mask_sha256 = <hash omitted>
```

## 8. 从轨迹可以直接得出的结论 / Conclusions directly supported by the trace

1. **模型完成了视觉问答，却没有形成闭环动作。** 它对三个区域做了高置信度判断，但三个
   diagnosis 都是 `none`，因此没有产生 point、box 或 finish 中任何一个最终可执行动作。

   **The model completed the visual reasoning calls but produced no closed-loop action.**
   It judged three regions with high confidence, but all diagnoses were `none`, so no
   point, box, or authorized finish action survived runtime post-processing.

2. **thinking budget 不是本次表面失败的直接原因。** 八次调用都返回了合法 JSON，记录的
   reasoning tokens 共 `2032`；失败发生在语义判断和后续协议衔接，而不是 JSON 解析。

   **The thinking budget is not the immediate cause of this failure.** All eight calls
   returned valid JSON and recorded `2,032` reasoning tokens in total. The failure lies
   in semantic judgment and protocol handoff, not JSON parsing.

3. **存在协议级死路。** 本轮配置单次最多选 3 个区域，但 completion gate 要求一次 initial
   verification 覆盖全部 19 个区域；当前三个区域均被判为 `none` 时，runtime 禁止 finish，
   却没有执行下一批审计的状态或动作，于是得到 `stop=false + suggested_action=null`。

   **There is a protocol-level dead end.** A call may select at most three regions, while
   the completion gate requires the initial verification to cover all 19. When the first
   three regions are all diagnosed as `none`, runtime forbids finish but has no state or
   action for auditing the next batch, yielding `stop=false + suggested_action=null`.

4. **最终 mask 没有任何改善。** 没有工具调用、没有 candidate、没有 mask delta；离线指标
   完全保持初始化值。因此这条轨迹不能被解释为“旗舰模型做了更保守的有效选择”，只能解释
   为 verifier 没有完成可执行闭环。

   **The final mask does not improve at all.** There is no tool call, candidate, or mask
   delta, and every offline metric remains at its initialization value. This is not a
   successful conservative correction; it is an incomplete executable verifier loop.
