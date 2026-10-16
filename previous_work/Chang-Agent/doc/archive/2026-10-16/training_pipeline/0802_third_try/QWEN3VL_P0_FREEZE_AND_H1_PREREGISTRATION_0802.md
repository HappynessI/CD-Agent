# Qwen3-VL P0 冻结说明与 H0/H1 harness 预注册（0802）

- 状态：**设计合同已冻结；P0 验收待执行；H1 未实现；P1 未授权执行**
- 日期：2026-08-02（UTC）
- 冻结 runtime 锚点：`491532f4622667e9fbf299c71aac228fa37f6a04`
- 分析基线 HEAD：`0a5300ddc1d738a64c5bc93db6b0069f5c250fa7`

## 1. 本文件冻结什么、不冻结什么

本文件冻结 H0/H1 的实验语义、GT 边界、预算含义、输出标识、停止门和执行前检查。它不实现 H1，不创建 worktree，也不授权提交测试、训练或 GPU 作业。P0/P1 的 `CA_0802(1)`–`CA_0802(6)` 新目录名已经在机器可读预注册中预留，但当前均不存在；预留不等于创建、运行或批准。

在 P1 执行前仍必须补齐并再次冻结：用户批准的 exact commit、clean run worktree、工具与模型哈希、CPU fixture manifest/hash、resolved run config、Slurm 节点资源和提交脚本。任何这些字段变化都需要形成显式 preregistration amendment；不能在作业启动后静默修改。P0 contract validation 的观察结果必须写入仓库内独立的 `QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json`，不能回填或改写本设计 manifest；最终 receipt 再引用 `CA_0802(1)` 日志和证据。

机器可读合同：

- harness 预注册：`doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_PREREGISTRATION_0802.json`；
- 输入 manifest：`doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json`；按文件内 canonical JSON 规则得到 SHA256 `<hash omitted>`；
- v2 重建证据：`doc/training_pipeline/0802_third_try/QWEN3VL_V2_RECONSTRUCTED_RESOLVED_CONFIG_0802.json`。

状态被拆成三个正交维度：设计合同 `frozen`；P0 acceptance receipt 在设计冻结时为 `pending`；P1 `not_execution_ready` 且 `p1_execution_authorized=false`。这些 pending/readiness 值描述不可变的设计冻结时点，后续观察结果只写入独立 receipt/amendment。因此“冻结”不表示任何测试已经通过或可以提交 P1 作业。

历史 `outputs/CA_0801_archive/` 只作为只读证据。不得修改旧 trajectory、receipt、summary、模型、数据 JSON 或哈希。

## 2. 共同不变量

- model I/O protocol 继续使用 `v12.1` atomic audit/candidate schema；H1 只改变同一 accepted Environment state 上的调度，不修改训练 target schema。
- target class 固定为 `building`。
- base model 固定为本地 `/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`；P1 冻结阶段先只比较同一 base model 的 H0/H1，不引入 SFT 权重差异。
- H0/H1 使用相同模型、工具、输入、样本顺序、推理 seed、generation 参数和 Environment 初始状态。
- frozen parser、geometry authorization、locality hard gate、candidate benefit-without-harm 和 rollback 必须保持逐项一致；H1 不得放宽或绕过任何安全门。
- 若离线审计发现 hard gate 可能误拦，P1 不得就地“校准”。任何 gate 变化必须另立 arm、取得用户批准，并重新执行 H0 parity、安全测试及 base control。
- 所有在线触发、region/scan 选择、模型输入、动作授权、candidate 审核和停止决策均为 GT-free。GT 只能在 rollout 完成标记之后由离线 scorer 加载。
- H0/H1 只用于 diagnostic harness 冻结，不能据 CORE3 结果宣称 LEVIR-CD 泛化提升。

唯一因果变量命名为 `terminal_review_scheduler_v1`：H0 为 legacy single-round terminal policy，H1 为一次 GT-free independent terminal-review round。protocol/arm ID 和 `max_verifier_rounds` 的差异只是该 treatment 的标识与实现；模型权重、model-I/O protocol、工具、安全门、A/B 输入及顺序、seed、generation 参数、Environment 初始状态、每轮 stage attempt 上限和 environment step 上限均为控制变量。除机器可读 manifest 列出的 allowed arm differences 外，任何共变都必须另立 arm。

## 3. 独立标识

| 项目 | H0 legacy control | H1 terminal-review treatment |
|---|---|---|
| harness protocol ID | `qwen3vl_harness_h0_legacy_single_round_v1` | `qwen3vl_harness_h1_gtfree_terminal_review_v1` |
| experiment arm ID | `v12_1-h0-legacy-single-round-base-0802` | `v12_1-h1-gtfree-terminal-review-base-0802` |
| summary schema ID | `qwen3vl_e2e_summary_v2` | `qwen3vl_e2e_summary_v2` |
| receipt schema ID | `qwen3vl_evaluation_receipt_v2` | `qwen3vl_evaluation_receipt_v2` |
| model I/O protocol | `v12.1`（保持） | `v12.1`（保持） |

这里的“独立”指 protocol/arm 标识及每个 arm 的 summary/receipt **artifact** 独立；H0/H1 共用同一公共、版本化 schema，不平行制造两个结构漂移的 schema。新输出不得继续登记为旧的 `v12_1-compact-post-sft` arm，也不得覆盖任何 0801 目录。预留的每阶段/arm 唯一输出路径已写入机器可读 manifest；提交前仍必须再次验证目录不存在，任一冲突都要停止并显式 amendment，不能现场顺延序号。

schema ID 已冻结不等于实现完成：summary v2 definition/validator 仍在实现中；receipt v2 definition/validator 尚未实现。两者均通过 happy/error/empty/tie/count-regression 合同测试前，manifest 保持 `not_execution_ready`，不得运行 P1。

## 4. 三类预算

| 预算 | H0 | H1 | 精确定义 |
|---|---:|---:|---|
| `max_verifier_rounds` | 1 | 2 | 同一 accepted Environment state 上的独立语义观察轮数；H1 的第 2 轮只由首轮 provisional `finish` 或 `plan=null` 触发。 |
| `max_stage_attempts` | 每轮最多 2 | 每轮最多 2 | 单个 stage/region 的总 generation 次数，包含 initial；即最多 initial + 1 次协议 repair，不是两次额外探索。 |
| `max_environment_steps` | 3 | 3 | 实际 action、candidate、accept/rollback 后的 Environment 状态迭代上限；它不保证一定执行三步。 |

当前冻结 runtime 的 `action_retries` CLI 是 deprecated compatibility option，并不会让授权 geometry 多执行：每个已授权动作的有效工具尝试上限为 1。必须把 stage initial、schema repair、本地 backend network retry（有效上限 0）、tool action attempt 和 tool failure 分开记账；增加 verifier round 也不得自动增加 environment step 或强制产生动作。

## 5. H0 合同

H0 是旧 v12.1 调度的行为保持 control：首轮合法 `finish` 立即结束，首轮无法形成安全 plan 时以 `no_executable_action` 结束；不会为了用满预算再调用独立 verifier round。

H0 的冻结条件不是“指标与旧 summary 字符串相同”，而是相同 fixture 下的 region 顺序、model call 数、repair 数、runtime diagnosis、plan、动作、candidate、安全门和终止原因与旧行为一致。P0 新 summary 可以修正指标口径，但不能改变 H0 rollout 行为。

## 6. H1 合同

### 6.1 GT-free 触发

H1 只允许两个 trigger：

1. 首轮 runtime 将 `finish` 判为结构合法但尚未独立确认；
2. 首轮在完成允许的 protocol repair 后仍得到 `plan=null`。

trigger 只能读取当前 accepted Environment state、首轮 protocol/runtime 结果和预算计数。不得读取 GT、离线 FP/FN、IoU/F1、人工 failure 标签或“该样本历史上是 false finish”等信息。

### 6.2 独立且非伪重复的第二轮

- 第二轮必须带 `round_index=2` 和明确的 `round_kind=finish_confirmation|plan_null_alternative`。
- `finish_confirmation` 从同一 accepted Environment state 重新构建一次独立审计请求，不继承首轮模型 hidden state，也不能直接复制首轮 raw prompt/response 当作新观察。
- `plan_null_alternative` 按 Environment 原始确定性顺序选择尚未得到自洽结论的下一 region；若没有未审 region，则对首个 unresolved region 使用单独版本化的 alternative-diagnosis prompt。具体 prompt 与选择算法必须在实现 commit 和 CPU fixture 中冻结后才可执行 GPU smoke。
- 第二轮仍使用 production stage image scope；不得添加 GT crop、label、离线错误类型或由 GT 派生的提示。
- 第二轮最多使用自己的 initial + 1 repair；不能递归触发第三 verifier round。

### 6.3 第二轮后的决策

- 若第二轮独立产生通过全部原安全门的唯一可执行 plan，才允许进入原 Environment action/candidate 流程。
- 若首轮和第二轮都合法 `finish`，终止原因为 `confirmed_finish`。
- 若任一轮保留 unresolved/plan-null、两轮互相冲突，或第二轮预算耗尽，则不授权动作，终止为 `no_executable_action_after_review`，不得记录为语义收敛。
- 完整 trigger×outcome 状态表已在机器可读预注册中冻结。特别地，首轮 `plan=null`、第二轮 `finish` 属于 round conflict，只能 fail closed 为 `no_executable_action_after_review`；任一 trigger 后只有第二轮唯一 safe plan 才能进入既有 action/candidate 安全流程，进入流程本身仍不代表动作已经获准。
- 不得为了满足 candidate 数量门而把 `uncertain`、mixed 或无唯一 target view 的输出转成动作。

## 7. 固定测试顺序

1. CPU Slurm replay/contract fixtures：至少覆盖 legacy finish、legacy plan-null、第二轮安全 plan、第二轮仍不确定、两轮冲突、repair 耗尽、预算耗尽和安全门拒绝。
2. 1 GPU smoke：base model、`test_1_1`，先 H0 后 H1；用途仅为证明真实 false-terminal 流程能进入独立第二轮。
3. 1 GPU diagnostic CORE：`DEV-CORE-3-V1`，固定顺序 `test_20_15`、`test_1_1`、`test_50_8`；先完成 H0，再运行 H1，不并发。

推理 seed 固定为 `42`。H1 冻结前不运行 post-SFT 模型、EXTENDED5 或 locked final。

smoke/CORE 的顺序及 `test_20_15`、`test_1_1`、`test_50_8` 的 A/B/label_cvt SHA256 已冻结在 input manifest。A/B 是在线输入；label_cvt 的 hash 只用于 provenance，文件只能在 rollout-complete 后由离线 scorer 打开。CPU fixture manifest 路径预留为 `QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json`，当前文件与 hash 均不存在，是提交 CPU fixture 和任何 GPU 作业前的显式 P1 blocker。

## 8. Summary 与 receipt 最低字段

H0/H1 必须各自保存独立的 summary artifact；两个 artifact 共同服从 `qwen3vl_e2e_summary_v2`：

- model state、model hash、harness/model-I/O protocol、arm、commit、工具 hash、样本 ID/hash/order；
- 单次 summary 保存 initial 与本模型 selected，并明确 `model_state`；跨模型 comparison receipt 必须引用匹配的 T0 base summary hash 与 post-SFT summary hash，再分列 initial、base-verifier selected、post-SFT selected。post-SFT summary 不得复制或伪造未在本次 rollout 中生成的 base-selected，且禁止把 initial 称为 base-selected；
- per-sample、micro 和 macro IoU/F1，以及 TP/FP/FN 和逐样本变化；
- 每轮 raw output、round/attempt kind、region、protocol errors、runtime diagnosis、plan、action、candidate、安全门与终止原因；
- 三类预算的 configured/consumed 值，以及 stage initial、schema repair、network retry、tool action attempt/tool failure 的分账；deprecated `action_retries` CLI 值只作兼容 provenance；
- candidate 的 offline gain、false accept、false reject、hard-gate-before-Qwen 分流；
- `gt_loaded_before_rollout_complete=false`、requested/scored、tool failure 和 dirty-worktree receipt。

P1 harness 安全冻结的 primary endpoints 是协议与安全硬门，不是 IoU/F1：H0 fixture 行为 parity 必须为 `1.0`；所有 eligible terminal-trigger fixture 必须恰好进入一次第二轮；paired H1 final-region protocol-valid rate 不低于 H0；pre-rollout GT violation、safety-boundary relaxation、requested/scored mismatch 和 tool failure rate 必须为 `0`。各 endpoint 的精确分子、分母、零分母规则与比较符见机器可读预注册。IoU/F1、count delta、candidate oracle 和是否产生新诊断/candidate 全部作为 diagnostic/promotion 指标，不能驱动在线 trigger/region/action，也不是 harness 安全冻结的必要条件。

零分母口径在实现前固定为：GT 与 prediction 都为空时 sample IoU/F1/precision/recall 均为 1；仅 prediction 为空且 GT 非空时 precision=0、recall=0；仅 GT 为空且 prediction 非空时 precision=0、recall=1，IoU/F1=0。macro 是固定样本的 sample metric 算术平均，micro 先汇总 TP/FP/FN 再按同一公式计算。

逐样本 degradation 定义为：IoU 或 F1 任一下降；若 IoU/F1 都持平，则 TP 下降或 FP/FN 任一增加也算 degradation。旧 v1 的零 IoU 下新增 6518 FP 必须被新 summary 判为退化。

candidate confusion 使用互斥 leaf class：harmful+accepted=`false_accept`；beneficial+Qwen rejected=`qwen_false_reject`；beneficial+pre-Qwen hard-gated=`hard_gate_false_reject`。Qwen false-reject 的分母只含到达 Qwen 的 beneficial candidates，hard-gate false-reject 与 all-false-reject 的分母是全部 beneficial candidates；indeterminate 不强塞进二分类。完整 3×3 oracle/decision-path 映射与分母见机器可读预注册。

receipt 必须引用 summary hash、launch command/resolved config、rollout marker、trajectory 清单和离线 scorer 输入；路径同时区分历史运行位置与当前证据位置。receipt 中任何必需路径不存在时不得标记 complete。

## 9. H1 冻结停止门

任一条件失败即停止，不冻结 H1、不进入训练：

1. H0 replay 不能保持旧调度行为；
2. finish 或 plan-null fixture 没有进入恰好一次第二 verifier round，或完整 trigger×outcome transition table 未被覆盖；
3. 第二轮 raw request 与首轮完全相同，或 region/prompt 选择无法确定性回放；
4. rollout complete 前出现 GT、FP/FN、IoU/F1 或人工 failure label 访问；
5. parser、geometry/locality、candidate 或 rollback 安全门发生放宽；
6. 任一预算超限、计数缺失或 `verifier_retries=2` 被误解释为 initial 后再两次 repair；
7. replay fixture 无法证明“第二轮可形成安全 plan/candidate，同时仍可拒绝危险动作”；
8. live smoke 的历史 finish case 没有实际进入第二轮，或第二轮不是 schema-valid 独立 audit；第二轮是否产生新诊断/candidate 只作诊断，不是安全冻结门；
9. 同样本 H1 的 final-region protocol validity 低于 H0；raw-attempt invalid 数必须另报，不能靠 repair 隐藏；
10. requested/scored 不一致、`tool_failure_count != 0`、receipt/summary/schema ID 缺失或回放不确定；
11. exact commit、clean worktree、唯一新输出路径、summary/receipt validator、输入/fixture manifest hash 或用户批准任一缺失。

H1 是否改善 IoU/F1不是 harness 安全冻结的充分条件，也不能用 GT 指标反向选择第二轮策略。若仅模型语义仍错误，应如实进入后续 capability diagnosis，而不是继续调 harness 迎合 CORE3。

## 10. Slurm、GPU 与 Git preflight

- 用户批准前不创建正式运行 worktree、不提交作业。
- 所有测试和计算通过 Slurm；管理节点只做文档、文件检查和轻量诊断。
- 先运行 `scontrol ping`，再以 `squeue -u guisongxia` 检查重复作业；不得使用路径名 `wangyihan` 作为 Slurm 用户。
- 从用户批准的 exact commit 创建新的 detached clean worktree；核验 HEAD、空 `git status --porcelain`、runtime diff 和工具 hash。
- 逐项核验 base model、解释器、LEVIR `test_256`、SimpleClick、SAM3、BPE 和输出根路径。
- CPU fixtures 不申请 GPU；H0/H1 smoke/CORE 每次只申请 1×A800，并验证 CUDA 可见卡数恰为 1、A800 和 BF16。
- 任一时刻本任务 GPU 总数不得超过 3；H0/H1 必须顺序运行，不用并发抢占预算。
- 输出目录必须全新且不存在，stdout/stderr 写入其 `logs/`；不得写到 `outputs/` 根或覆盖 0801 archive。
- 长作业成功或失败均发送不含凭据的 QQ 终态通知，记录 Job、状态、耗时、关键结果和产物路径。

P0 验收的冻结入口为 `tools/submit_qwen3vl_p0_validation.sh`，worker 为
`tools/run_qwen3vl_p0_validation.slurm`。入口只允许从 exact commit 的 clean run worktree
执行；独立 worktree 的 Git-ignored `outputs` 必须是指向开发仓库固定 artifact root 的
symlink，submitter/worker 都以 `realpath` fail closed 校验，不能假设 ignored archive 会随
`git worktree add` 被 checkout。它顺序提交两个 0-GPU CPU Job：`CA0802P0Target` 执行三组
targeted unittest、shell syntax、workbook `--check` 和只读 v1 archive summary integration；
`CA0802P0Regress` 以 `afterany` 依赖运行全部 CPU unittest。两个 Job 分别保存独立 result/status
receipt，共用本节预注册的 `logs/slurm-%j.out/.err` 模板；targeted 在全部检查前后对八个 v1
输入文件做 SHA256 清单比较。任何一项失败均不生成 P0 acceptance pass receipt。

本段与下表保留原始设计冻结时点，不随执行结果回填。首次 `(1)` 和 R1 均失败后，当前
submitter/worker 的唯一有效 allocation 已由
`QWEN3VL_P0_VALIDATION_AMENDMENT_R2_0802.{md,json}` 追加为 `(1R2)`、
`CA0802P0TargetR2/CA0802P0RegressR2`。R2 还要求逐文件复核 `(1)` 与 `(1R1)` 两次失败
证据。旧 preregistration 与 R1 amendment 不修改，P1/GPU 状态不变。

预留目录与 QQ title 如下；六个目录在设计冻结时均不存在，submit preflight 必须在每次提交前重新验证，不能因冲突现场换号：

| 阶段/arm | 预留输出目录 | QQ title |
|---|---|---|
| P0 contract | `CA_0802(1)-qwen3vl-p0-contract-validation` | `CA0802 P0 contract validation` |
| P1 CPU H0/H1 | `CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures` | `CA0802 P1 H0-H1 CPU fixtures` |
| smoke H0 | `CA_0802(3)-qwen3vl-h0-base-smoke1` | `CA0802 P1 H0 base smoke1` |
| smoke H1 | `CA_0802(4)-qwen3vl-h1-base-smoke1` | `CA0802 P1 H1 base smoke1` |
| CORE3 H0 | `CA_0802(5)-qwen3vl-h0-base-core3` | `CA0802 P1 H0 base CORE3` |
| CORE3 H1 | `CA_0802(6)-qwen3vl-h1-base-core3` | `CA0802 P1 H1 base CORE3` |

每个目录的 stdout/stderr 精确模板均为自身 `logs/slurm-%j.out` / `logs/slurm-%j.err`。QQ terminal body 必须含 `job_name`、`job_id`、`terminal_status`、`elapsed_seconds`、`key_result_or_error`、`artifact_path`，覆盖 success、failed、cancelled、timeout、out-of-memory 和 node-fail，且禁止包含凭据或代理值。

## 11. v2 resolved-config 证据冻结

v2 Job `52011` 的历史 `config_used.yaml` 是未应用命令行覆盖的模板快照，不能再称为 resolved config。只读重建结果及逐字段证据见：

`doc/training_pipeline/0802_third_try/QWEN3VL_V2_RECONSTRUCTED_RESOLVED_CONFIG_0802.json`

当前冻结版本 SHA256 为 `<hash omitted>`。该 JSON 明确标为 reconstructed evidence：48 train examples、2 epochs、12 optimizer steps 和 `3e-5` learning rate 来自 validation、trainer state、训练日志与 registry 的交叉证据；其中原始 submit-shell 环境字符串没有被持久化，不能伪造为直接 receipt。重建文件不得作为 archive 重训入口。

## 12. 当前冻结状态

- H0：语义合同已预注册，尚未针对未来 implementation commit 生成 parity receipt。
- H1：只完成设计预注册，代码不存在，`p1_execution_authorized=false`。
- v2：只新增重建证据映射；历史 archive 未修改。
- P0 验收：receipt 在设计冻结时为 `pending`；该历史状态不回填。后续执行观察已写入本节
  下方独立 receipt，并由外部 hash 绑定。
- P1 阻断项：CPU fixture manifest、receipt v2 definition/validator、implementation commit、clean worktree、模型/工具/worker hash、resolved config、Slurm 资源/脚本和用户批准均未完成。
- 下一动作：汇报 P0 acceptance；再由用户决定是否批准 P1 实现。当前不得把 P0 通过解释
  成 P1 execution-ready。

## 13. P0 执行观察（不回填原 manifest）

R2 validation commit 为 `e9fbcd690a9af2e91384231ced46555fe400d9ba`。0-GPU Jobs
`52598/52599` 均为 `COMPLETED/0:0`：targeted 三组测试 `19/19 + 14/14 + 49/49`，
full regression 为 357 passed、3 skipped、360 total；workbook、真实 v1 summary/assertion 和
八个命名输入的 hash window 均通过。

独立结果见 `QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json`：canonical SHA256
`<hash omitted>`，raw SHA256
`<hash omitted>`。receipt 记录三次
attempt、六个 Job、32 个 evidence hash 和 16/16 gates；P0 accepted，但 H1 未实现、
`p1_execution_authorized=false`、GPU 未授权。
