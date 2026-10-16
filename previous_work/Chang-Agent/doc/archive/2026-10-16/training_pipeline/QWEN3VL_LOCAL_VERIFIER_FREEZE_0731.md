# 本地 Qwen3-VL Verifier 训练前冻结审计（0731–0801）

## 结论口径

本轮只决定本地推理/评测协议是否足够稳定，可以据此开始构建训练数据；固定五样本是
开发诊断集，不用于宣称 LEVIR-CD 泛化性能。

最终结论（2026-08-01）：建议冻结提交 `491532f` 的 v12.1 本地推理/评测框架并进入
正式训练数据构建。这里冻结的是 compact schema、输入范围、parser/fail-closed 约束、
Environment/Executor/rollback 安全门和固定评测 harness，不是未训练 Qwen3-VL-2B 的
视觉语义能力。历史 v12、select-only SFT32 和修复前轨迹不得混入 v12.1 训练 target。

## 为什么需要精简

历史 `CA_0730(7)-qwen3vl-base-protocol-fix-e2e5` 中，5 个样本共有 45 个
Environment region，全部最终进入 audit。模型 select 调用 18 次，只改变全量集合的
顺序，没有筛除任何 region；随后产生 90 次 audit（含 retry），audit backend latency
约 1279.2 秒，占全部 backend latency 约 97.9%，原始 audit 响应约 19.9 万字符。

最终 39 个 region 记录为 `screening_resolution` mismatch，但逐 attempt 回看显示这些
响应同时普遍缺少模型被要求回填的 `change_mask_state`。因此不能把 retry 节省错误归因
于只删除 screening；正确拆分是：

1. 删除不改变覆盖集合的模型 select 和 screening 关系；
2. 由 Environment 注入权威 `change_mask_state`；
3. 再单独比较 v11.3 冗长 atomic audit 与 v12 compact audit。

## v12 保留与删除

模型仍必须输出 region ID、T1/T2 RGB 状态、一条可观察 evidence，以及七项 checklist
状态。七项 checklist 分别覆盖证据充分性、目标类别、白像素精度、边界、内部空洞、
变化目标完整性和碎片伪影，均保留。

运行时机械派生 `change_mask_state`、evidence quality、`mask_assessment`、
`error_type`、target view、动作几何、quality 和 finish readiness。删除的是重复的
逐项 evidence、模型回填的派生字段、初始 feedback-only decision 调用，不是安全检查。
所有 Environment region 仍按确定顺序全量审计，completion gate 仍要求全覆盖。

新增的 fail-closed 守卫包括：

- 白色 proposal 的明确无变化 RGB、黑色 proposal 的明确变化 RGB 不能全 pass；
- FN 目标时相必须同时满足 RGB 语义和 Environment seed editability，不能切换时相；
- 除 `evidence_sufficient` 外全为 `not_applicable` 时派生为 uncertain；
- 空 full-state proposal 只有在 Environment 明确报告零未覆盖像素时才可 finish；
- region ID、RGB/checklist enum、proposal 极性和动作 editability 继续严格校验。

## 必要改动与实现落点

本轮确认有必要精简，且修改集中在模型协议和可审计边界，没有重写 Environment/Executor：

- `change_agent/adapters/staged_verifier.py`：按 Environment 顺序全量 audit，删除不筛选
  region 的 live select 和初始 feedback-only decision；机械派生冗余字段，增加一致性、
  completion、candidate-plan 和 region-level fail-closed 门；仅在首轮唯一错误为 evidence
  grounding 时建立 deep-copy anchor，并只替换 evidence 后重跑完整 parser。
- `change_agent/adapters/stage_backends.py`：compact audit schema；scan 只看权威 tile box 内
  T1/T2 RGB；field-local repair 只看权威 proposal box 内两张 RGB crop，并在 local/hosted
  history 记录 `visual_input_scope`。
- `change_agent/verifier_protocol.py`：严格校验 RGB state/checklist/evidence，拒绝已观测到的
  schema boilerplate、machine label 和 mask/presentation shortcut，不猜测或清洗模型语义。
- `tools/analyze_staged_verifier_run.py`、`tools/compare_local_verifier_ab.py` 及 Slurm harness：
  校验 scope/stage/attempt、调用记账、固定 profile、commit/hash/环境和逐样本指标来源。
- 对应 backend、parser、analyzer、comparator、Environment 与 action protocol 回归测试均已
  补齐；最终 full regression 共 297 项。

特意没有把 `black/white` 映射成 `building/background`，也没有放宽 Environment enum。
`change_mask_state` 仍只表示当前 proposal 的运行时极性；它被权威注入不等于 RGB 语义或
七项 checklist 正确，后两者仍必须由模型输出并经过 fail-closed 校验。

## 可复现实验

最终 CPU 测试输出：

`outputs/CA_0801(1)-local-verifier-v12_1-evidence-repair-tests`

其中 targeted Job `51723` 覆盖十组相关测试并通过；全量 regression Job `51724` 为
`297 tests` 全部通过。两者均由 Slurm 顺序提交，各自写终态 receipt 并发送 QQ：

```bash
bash tools/submit_local_verifier_tests.sh
```

GPU 迭代使用三个闭集 profile，均由 harness 固定评测 ID、样本及顺序：`smoke1`
为 `DEV-SMOKE-1-V1`/`test_20_15`；`e2e3` 为 `DEV-CORE-3-V1`/
`test_20_15,test_1_1,test_50_8`；`e2e5` 为 `DEV-EXTENDED-5-V1`/既定五样本。
三者都把 profile、ID 和样本顺序交叉写入 submission、终态 receipt、input manifest、
baseline summary 与 protocol summary。`smoke1` 和 `e2e3` 必须显式指定 `OUTPUT`，只用于
快速迭代；正式 A/B comparator 仍硬性只接受 `e2e5`，三样本结果不能冒充冻结证据。

GPU A/B 使用相同 base 模型、五样本顺序、seed=42、CLI、A800 类型及后处理。控制臂固定
`f80e270`：它已经删除 live select/screening 并注入 Environment mask state，但保留
v11.3 冗长 audit；处理臂固定最终 clean v12.1 提交 `491532f`。两臂在同一预检节点
`gpu48` 顺序运行，总并发一张卡：

```bash
NODE=gpuXX bash tools/submit_local_verifier_e2e5.sh control /path/to/clean-f80e270-repo
NODE=gpuXX DEPENDENCY=afterany:<control-job-id> \
  bash tools/submit_local_verifier_e2e5.sh compact
```

正式两臂分别为：

- Job `51730`：`outputs/CA_0801(4)-qwen3vl-v11_3-control-evidence-repair-harness-e2e5`
- Job `51731`：`outputs/CA_0801(5)-qwen3vl-v12_1-evidence-repair-e2e5`

两臂完成后，CPU Slurm 作业验证固定 arm/protocol/schema、input/model/tool hashes、同一
Slurm 节点与 partition、相同 Python/CUDA/package 环境、harness/OmniOVCD commit，以及
逐样本初始 region ID 和 coverage count，再生成机械对比。比较器还要求两臂 worker 的
终态 receipt 为 `status=success`、`exit_code=0`，并把 manifest、job ID、run 路径与四份
summary 交叉绑定；baseline 指标、动作、outcome、退化样本和工具成败均从原始
`per_sample_metrics.json`、episode summary 与 tool report 重新计算，不信任二手 totals：

```bash
bash tools/submit_local_verifier_ab_compare.sh
```

最终 comparator Job `51762` 写入
`outputs/CA_0801(6)-qwen3vl-v11_3-v12_1-evidence-repair-comparison`。每个 GPU worker
在启动和 rollout 后均验证 arm/harness commit 未漂移且 tracked worktree clean；GT 仍只在
GT-free rollout marker 之后用于离线评分。comparison 显式列出逐样本 selected IoU、
F1、相对 initial 的 delta，以及相对 control 的退化信号。

## 最终结果

分级门均通过：Job `51725` 的 `smoke1` 为 8/8 atomic audit valid、0 regional error，
记录 1 次 field-local evidence repair；Job `51727` 的固定 `e2e3` 为 18/18 valid、
0 regional error，并记录 1 次 field-local evidence repair 和 10 次
`authoritative_pair_rgb_only_scan`。两者均无工具失败或 selected 指标退化。

正式 matched e2e5 的机械 comparison 显示两臂 profile、评测 ID、样本顺序、输入、模型、
工具、节点、分区、运行环境和 GT boundary 全部可比，且初始覆盖均为 45 个 region：
下表是 v11.3 到最终 v12.1 的累计协议对比，不能把全部改善单独归因于最后一次
field-local evidence repair。

| 指标 | v11.3 control | v12.1 treatment | 变化 |
|---|---:|---:|---:|
| backend calls | 95 | 65 | -31.6% |
| audit calls | 90 | 60 | -33.3% |
| repair calls | 45 | 15 | -66.7% |
| atomic audit valid | 0/45 | 37/45 | +37 |
| regional errors | 45 | 8 | -82.2% |
| raw response chars | 205608 | 29918 | -85.4% |
| backend latency | 1312.658 s | 226.457 s | -82.7% |
| episode elapsed | 1403.891 s | 381.417 s | -72.8% |
| invalid/no-action samples | 5 | 1 | -80.0% |

Treatment 的 7 个工具调用全部成功，7 个 candidate 全部被效果或 hard gate 安全拒绝，
0 tool failure、0 accepted candidate。两臂 aggregate selected IoU/F1 都保持
`0.67629040/0.80688931`，五个样本逐项 selected 与 initial 相同，无退化样本。
唯一的 `action_attempt_error` 位于 `test_78_13`，内容是 verifier 未授权下一项可执行
动作；它以 `no_executable_action` 安全终止，不是 action payload/protocol validation error、
未授权执行或工具失败。
rollout 后的离线 GT 复核同时说明 base 模型仍过于保守：`test_20_15` 的三个有害候选均
正确拒绝，但 `test_78_13` 的首个候选实际把 IoU `0.75787116→0.76640245`，
`test_85_16` 的首个候选实际把 IoU `0.30658070→0.31563287`，二者也被拒绝。这不破坏
保守选择的安全性，却说明未训练模型尚不能可靠判断 candidate benefit。

Treatment 剩余 8 个 regional error 已逐条回看 raw response：`test_78_13` 的
`r1/r4/r7` 和 `test_85_16` 的 `r3/r6/r11/r16/r18` 均完成 initial + repair；所有 JSON
均正常闭合且存在 `parsed_output`。失败来自 `mixed` actionable state、白区无目标转移却
全 pass、纯 false-negative 极性冲突，或直到普通 repair 才引用 mask evidence，不是格式
截断、scope 错配或漏建安全 anchor。它们都保持 `standard` scope，因为首轮错误并非唯一的
evidence-grounding 错误；相对地，`test_85_16` 另有 4 次 field-local evidence repair 成功。
这些专用 repair response 只提供新 evidence；runtime 从 deep-copy anchor 保留首轮
states/checklist，合并后重新运行完整 parser，模型不能借 repair 改写其他已验证字段。

这 8 个 region 最终全部变为 `uncertain/insufficient`、`target_view=null`，completion gate
为 false，未从错误 region 授权动作或 finish。regional fail closed 不要求整步 verifier
invalid；独立有效 region 仍可形成 plan，但候选继续经过原有 Executor/rollback 门。这里未
发现 runtime fail-open，因此不再增加词法启发式或重试次数，错误作为训练 hard cases 保留。

## 冻结门槛

只有同时满足以下条件才建议冻结本地推理/评测框架：

1. targeted 与全量 regression 全部通过；
2. 两臂 5/5 完成，输入、模型和工具 hash 完全一致，无 `.run_incomplete`；
3. v12 无 derived-field 协议依赖，所有无效输出继续 region-level fail closed；
4. 无未授权执行、action payload/protocol validation error 或工具失败；安全的
   no-authorized-action 终止单独记录，selected 指标无样本退化；
5. compact audit 的调用/repair、响应长度或 latency 至少一项实质改善，且没有用减少
   region 覆盖换取改善。

若安全与协议门通过但 base 2B 仍因视觉判断错误而无有效动作，可以冻结“数据格式与
评测框架”并进入训练；不能据此声称 base 模型行为已经可用。若出现 fail-open、错误
时相动作、覆盖下降或不可解释的指标退化，则不得冻结。

## 最终冻结范围与训练入口

五项冻结门槛均已满足。冻结范围为 v12.1 compact 输出契约、Environment 权威字段与全量
coverage、scan/evidence-repair 输入 scope、strict parser 与 region-level fail closed、
candidate plan 绑定、Executor locality/hard gate/rollback、轨迹分析和 A/B provenance
harness。除非后续复现新的未授权动作、错误时相执行、coverage/provenance 回退或 parser
fail-open，否则训练阶段不再改这套推理/评测协议。

未冻结的是 base Qwen3-VL-2B 的语义行为。`test_1_1` 仍把 626 个 FP 判断为可 finish，
`test_50_8` 仍漏掉 166 个 FN 并 finish；8 个 regional error 也说明 states、evidence 与
checklist 一致性尚需训练，两个实际改善却被拒绝的 candidate 还暴露 benefit 判断偏保守。
这些样本和 field-local repair 后仍语义含混的 evidence 应进入 hard-case/strict-scorer
集合，而不是通过 parser 放宽来“修好”。

训练数据现在应只生成 v12.1 compact target，并保留原始 paired RGB、权威 region box、
Environment 字段、模型 raw response、parser 结果和 fail-closed 标签。该冻结结论只支持
进入训练，不支持声称未训练 2B verifier 已可用或具备数据集泛化性能。
