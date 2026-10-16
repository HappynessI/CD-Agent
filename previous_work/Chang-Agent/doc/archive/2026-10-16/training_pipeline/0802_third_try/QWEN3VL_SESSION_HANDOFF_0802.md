# Qwen3-VL 0802 P1 会话交接

- 交接时间：2026-08-02（UTC）
- 开发仓库：`/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl`
- P1 收尾 worktree：
  `/guisongxia01/pangchao/wangyihan/wyh/Chang-Agent-qwen3vl-p1-closeout-0802`
- 当前阶段：P0/P1 已验收；P2 formal generation/scoring 已完成且科学 gate=false；final candidate C
  已由零 GPU Job `53323` 验收，本提交形成只绑定 C receipt 的 acceptance D；D 的终态由外置
  exact-D receipt 绑定
- 当前结论：**P2 工程链通过，但 test_1_1 在冻结有限主动作空间中无 beneficial candidate；禁止
  构造 C-ISO/T-TRAJ**
- 执行状态：P1 B4 与 P2 freeze B 的 detached runtime worktree保持只读；不得重复提交既有 P1/P2
  formal attempt，也不得绕过 P2 no-go 开始训练
- 本文用途：供新会话在不丢失 Git、协议、失败历史和科学解释边界的前提下，从已完成 P1 恢复并进入
  已授权的下一阶段

## 0. 当前权威恢复点（覆盖后文历史快照）

implementation A4 为 `a88404ac6a014f42692d1119d54de42c980fb310`，final freeze B4 为
`5210f89f5ca13ee1e636f5aa6a44631206a80123`，且 `B4^ == A4`。开发仓库和 detached run
worktree 当前都 clean、exact B4；run worktree 保持 detached，B4 submitter `--check-only` 已在正式
提交前通过。

B4 冻结摘要保持不变：

- config raw：`<hash omitted>`
- fixture canonical：`<hash omitted>`
- amendment raw：`<hash omitted>`
- amendment canonical：`<hash omitted>`

precommit Job `53151` 已通过 300 targeted、470 full、3 skipped、golden validator 与 shell
syntax；hash Job `53152` 的 25-entry manifest SHA256 为
`<hash omitted>`。

retry3 正式链已完整结束：

| role | Job | Slurm state | elapsed | result |
|---|---:|---|---:|---|
| formal CPU | `53155` | `COMPLETED / 0:0` | 00:00:25 | receipt complete；300 targeted、470 full、3 skipped |
| smoke H0 | `53156` | `COMPLETED / 0:0` | 00:01:16 | summary/receipt valid |
| smoke H1 | `53157` | `COMPLETED / 0:0` | 00:01:58 | summary/receipt/smoke pair valid |
| CORE3 H0 | `53158` | `COMPLETED / 0:0` | 00:04:18 | 3/3；summary/receipt valid |
| CORE3 H1 | `53159` | `COMPLETED / 0:0` | 00:05:59 | 3/3；summary/receipt/CORE pair valid |
| dependency guard | `53160` | `COMPLETED / 0:0` | 00:00:02 | synthesized none；unresolved none |

独立 CORE pair 审计通过 18 项 controlled-variable checks、25 项 hard gates、两个 individual
receipt validators 和 pair validator；`cross_attempt_pair_composition=false`。H0 round two 为 0，
H1 为 2；两次额外调用都安全地执行为 fresh-context finish confirmation，但都再次返回 finish，没有
safe plan 或 tool action，因此 selected masks 没有变化。

CORE3 最终结果：

- micro IoU/F1：H0 与 H1 均为 `0.74223120 / 0.85204673`，delta `0 / 0`；
- macro IoU/F1：H0 与 H1 均为 `0.28212851 / 0.30560087`，delta `0 / 0`；
- 三个样本的 selected state 和离线指标逐项精确一致；
- `test_20_15` 的四步 H1 trajectory 已成功保存，确认越过旧 Job `53145` 的自环崩溃点。

逐样本失败机制是：

- `test_20_15`：三次成功 tool action 产生的 candidate 都更差并被安全拒绝；非终局 passthrough，
  H1 不应调用 round two，最终 budget exhausted；
- `test_1_1`：无变化样本保留 626 个 FP；fresh review 重复了 false-positive finish；
- `test_50_8`：有变化样本保留空 mask 和 166 个 FN；fresh review 重复了 false-negative finish。

分析 Job `53164` 因本地分析器错误期待 pair validator 文本为 `status=complete` 而失败；冻结合同的
正确值是 `status=valid`。只修正分析器读取后，Job `53165` 从 clean B4 成功生成
`trajectory_analysis.json`，SHA256 为
`<hash omitted>`。`53164` 的失败日志/status
与 `53165` 的成功证据分别保留，未拼接部分结果。

详细结果报告：
`doc/training_pipeline/0802_third_try/QWEN3VL_P1_RETRY3_RESULT_0802.md`。

当前训练数据主要是孤立 stage snapshots，缺少同 episode 的
`audit → action → candidate → next-state` 完整后果链；本轮“能再次 audit、但不能纠错”的现象与此
假设一致，但**没有证明因果，也不能把它写成唯一根因**。CORE3 只是三个反复使用的诊断样本，不是
LEVIR-CD 泛化证明。

P1 closeout parent `d64ec522de1d46df27891cea1f9f9ab0fa1b7be0` 已由零 GPU Job `53178`
完整验证：workbook check、15 个 registry targeted tests、471 个 full tests（3 skipped）、冻结合同、
result/analysis receipt 和 repository immutability 全通过。closeout receipt SHA256 为
`<hash omitted>`。

P2 final candidate C 为 `d412759857d030b8ab766a077ea54b603585e72a`，已由 clean detached
exact-C 零 GPU Job `53323` 验收。Acceptance D 只绑定已验证的 C receipt；D 不自写 commit hash，
自身终态由 Git 外部 exact-D receipt 绑定。若该外置 D receipt 尚不存在，则下一恢复动作是跑 exact-D
终验；若已存在且全部通过，则只需确认主开发分支已 fast-forward 到 D。任一情况都不改变科学 no-go，
并停在数据构造门前，等待对冻结动作空间或独立 gate arm 的新设计/审批。

下文第 1–15 节是此前各时点的历史证据；第 16 节记录 P1 retry3；第 17 节是当前 P2 权威结果与收尾
边界。前文“下一步直接创建训练数据”等旧时点文字均由第 17 节的 gate=false 覆盖。

## 1. 冻结锚点与当前 Git 状态

### 1.1 P0 已验收锚点

- 分支：`feat/local-qwen3vl-verifier`
- 当前开发 HEAD：`00164f8e8ec141c0df9427ca86fc58cf1a13ebf5`
- commit subject：`docs(training): record P0 acceptance`
- P0 acceptance receipt canonical SHA256：
  `<hash omitted>`
- P0 acceptance receipt raw SHA256：
  `<hash omitted>`
- H0/H1 preregistration canonical SHA256：
  `<hash omitted>`
- evaluation input manifest canonical SHA256：
  `<hash omitted>`

对应文件：

- `doc/training_pipeline/0802_third_try/QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json`
- `doc/training_pipeline/0802_third_try/QWEN3VL_P0_FREEZE_AND_H1_PREREGISTRATION_0802.md`
- `doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_PREREGISTRATION_0802.json`
- `doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json`

### 1.2 开发 worktree

开发 worktree 当前为 dirty，HEAD 仍是上述 P0 commit。没有 staged change。必须保留全部现有
修改，不得 `git reset --hard`、不得覆盖或回退用户已有工作。

已修改的 tracked files：

- `change_agent/adapters/staged_verifier.py`
- `change_agent/adapters/stage_backends.py`
- `change_agent/environment.py`
- `change_agent/verifier_protocol.py`
- `change_agent/verifier_regions.py`
- `tests/test_stage_backends.py`
- `tools/analyze_staged_verifier_run.py`
- `tools/run_levir_change_agent.py`
- `tools/summarize_qwen3vl_e2e_run.py`

当前 untracked P1 files：

- `change_agent/adapters/terminal_review_verifier.py`
- `change_agent/terminal_review.py`
- `doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json`
- `doc/training_pipeline/0802_third_try/QWEN3VL_H0_V12_1_GOLDEN_FIXTURE_RECEIPT_0802.json`
- `doc/training_pipeline/0802_third_try/QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json`
- `doc/training_pipeline/0802_third_try/QWEN3VL_P1_EXECUTION_AMENDMENT_0802.md`
- `doc/training_pipeline/0802_third_try/QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json`
- `tests/test_terminal_review.py`
- `tests/test_terminal_review_integration.py`
- `tests/test_qwen3vl_e2e_summary_v2.py`
- `tests/test_qwen3vl_evaluation_receipt_v2.py`
- `tests/test_qwen3vl_h0_golden_fixture.py`
- `tests/test_qwen3vl_p1_execution.py`
- `tools/build_qwen3vl_evaluation_receipt.py`
- `tools/build_qwen3vl_p1_pair_comparison.py`
- `tools/build_qwen3vl_runtime_input_manifest.py`
- `tools/run_qwen3vl_p1_cpu_fixtures.slurm`
- `tools/run_qwen3vl_p1_dependency_guard.slurm`
- `tools/run_qwen3vl_p1_evaluation.slurm`
- `tools/submit_qwen3vl_p1.sh`
- `tools/validate_qwen3vl_e2e_summary.py`
- `tools/validate_qwen3vl_evaluation_receipt.py`
- `tools/validate_qwen3vl_h0_golden_fixture.py`
- `tools/validate_qwen3vl_p1_pair_comparison.py`

本交接文档也属于新增未提交文件。

### 1.3 正式运行 worktree

- 路径：`/guisongxia01/pangchao/wangyihan/Chang-Agent-baseline-0802-run`
- 状态：detached、clean
- HEAD：`00164f8e8ec141c0df9427ca86fc58cf1a13ebf5`

它仍停在 P0 commit，没有任何 P1 草稿。只有在实现 commit、冻结 commit 和所有 hash binding
真实形成后，才能把该 worktree 更新到 exact freeze commit。更新后必须再次验证 detached、
clean、HEAD 精确匹配 amendment。

## 2. 授权和实验边界

用户已经授权进入 P1，范围仅包括：

- 完成 H0/H1 terminal-review harness；
- 通过 Slurm 运行 CPU fixtures；
- 同一 local base model 上运行 smoke H0/H1 与 CORE3 H0/H1；
- 独立验证 receipts，并分析、归档 P1 结果。

当前未授权：P2、训练、SFT/post-SFT、EXTENDED5、locked final 或任何额外模型 arm。

共同固定项：

- base model：`/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- input root：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256`
- model-I/O protocol：`v12.1`
- seed：42
- GPU 上限：整个任务最多 3 张；当前串行设计峰值仅 1 张
- 计算节点方案：`gpu48`，partition `gre`
- 每个 GPU job：恰好 1 张 A800，运行时验证单卡可见和 BF16
- H0/H1 不并发；smoke 不保存 checkpoint
- GT/label 只能在 `rollout_complete.json` 之后由 offline scorer 访问
- 长任务无论成功或失败都必须发送 QQ 终态通知，且日志只能进入对应输出的 `logs/`

CORE3 只能作为反复使用过的 diagnostic promotion gate，不能解释为 LEVIR-CD 泛化证明。

## 3. 已完成的工作

### 3.1 失败分析、P0 harness 和预注册

- 已完成 0801 多版本 SFT/E2E 失败轨迹分析及 0802 分阶段实验计划：
  `QWEN3VL_FAILURE_ANALYSIS_AND_EXPERIMENT_PLAN_0802.md`。
- 已完成 P0 独立复核 harness、R1/R2 修订、输入 manifest、预注册和 acceptance receipt。
- P0 已由 commit `00164f8...` 记录为 complete，正式运行 worktree 也固定在该 commit。

### 3.2 Terminal-review 核心草稿

`change_agent/terminal_review.py` 已实现：

- H0 identity policy 与 H1 exactly-once round-two policy；
- 两类 trigger × 七类 outcome 的完整 14 项 transition table；
- 每个 accepted-state 的 exactly-once token，禁止第三轮；
- GT-free trigger 输入白名单；
- fresh backend context，禁止 hidden-state 和首轮 raw-response 继承；
- verifier/stage/network/tool/environment 的独立预算字段与 `counter_scope`；
- safe plan 只返回既有 Environment 流程，scheduler 没有工具执行 surface；
- 冲突、unresolved、unsafe/non-unique、schema invalid 和预算耗尽均 fail closed。

`change_agent/adapters/terminal_review_verifier.py` 和
`change_agent/adapters/staged_verifier.py` 已接入第二轮 review 草稿，包括 accepted-state 内容哈希、
round-two backend call audit、completion/schema/safe-plan 信息。最新草稿还加入了从 staged trace 与
backend calls 重建 plan-null trigger、Environment-order 单 region selection、两个冻结 prompt version ID、
显式 network retry 计数及 versioned empty-mask scan proposal；这些仍须完成严格负例并在 Slurm 验证。

### 3.3 Environment 和 runner 集成草稿

`change_agent/environment.py` 与 `tools/run_levir_change_agent.py` 已实现或起草：

- initial terminal result 与 accepted candidate 后的 terminal review；
- accepted `finish` 也进入 post-accept review；
- `Environment.done` 使用 review 后 feedback，避免首轮 finish 提前结束；
- `confirmed_finish` 映射为 converged，review no-action 映射为 no-executable-action；
- H0 不包装 decorator，保留 legacy control；H1 才包装 terminal-review decorator；
- P1 arm/protocol/budget/GT/safety audit；
- `run/p1_harness_audit.json`、episode review history 和预算汇总；
- post-review envelope、rollback replan 和 nested round evidence 的初步去重计数。
- rollout 前 runner input validation 不再 stat `label_cvt`，offline label validation 要求先存在并校验
  `rollout_complete.json`；对应 raw access boundary fixture 已写入 CPU suite 草稿。

静态审计已确认 accepted-state ID 不包含 `step_index`，相同内容状态不能通过改变步号获得第二次
review；safe plan 仍需通过原 parser、geometry、locality、candidate-benefit 和 rollback 安全流。

### 3.4 Summary、receipt 和 pair 工具草稿

已形成以下独立工具：

- `tools/analyze_staged_verifier_run.py`
- `tools/summarize_qwen3vl_e2e_run.py`
- `tools/validate_qwen3vl_e2e_summary.py`
- `tools/build_qwen3vl_runtime_input_manifest.py`
- `tools/build_qwen3vl_evaluation_receipt.py`
- `tools/validate_qwen3vl_evaluation_receipt.py`
- `tools/build_qwen3vl_p1_pair_comparison.py`
- `tools/validate_qwen3vl_p1_pair_comparison.py`

receipt v2 builder 已按当前 worker CLI 起草，绑定 baseline/protocol summary、P1 harness audit、
resolved config、static/runtime input manifest、run manifest、launch command、rollout marker、offline
metrics、submission/status 和 summary-validator output。最新小修将 status 路径统一到
`OUTPUT/logs/status-<job_id>.txt`，并为 analyzer 补充 `--harness-protocol-id`。

summary builder 已加入从 raw trajectory 重建 `final_region_protocol_valid` 的草稿；独立 validator 与
evaluation receipt 仍需使用各自实现验证 H0 raw staged envelope、H1 invoked/schema-valid round two、
missing evidence、budget/no-proposal 及 declared-value tamper，不能回退信任最终 feedback。

这些工具目前只有代码和 synthetic fixture，尚未在 Slurm 上执行，因此不能声明 receipt v2 已闭环。

### 3.5 执行合同和 Slurm 编排草稿

已新增：

- `QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json`
- `QWEN3VL_H0_V12_1_GOLDEN_FIXTURE_RECEIPT_0802.json`
- `QWEN3VL_P1_EXECUTION_AMENDMENT_0802.md/.json`
- `QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json`
- `tools/run_qwen3vl_p1_cpu_fixtures.slurm`
- `tools/run_qwen3vl_p1_dependency_guard.slurm`
- `tools/run_qwen3vl_p1_evaluation.slurm`
- `tools/submit_qwen3vl_p1.sh`
- `tools/validate_qwen3vl_h0_golden_fixture.py`

拟定唯一串行链：

`CPU fixtures → smoke H0 → smoke H1 → CORE3 H0 → CORE3 H1`

所有 GPU job 依赖 `afterok`，每个仅 1×A800。submitter 使用 held jobs，在完整 submission receipt
形成后统一 release；worker 包含 commit/worktree/hash/resource/status/QQ 的 fail-closed 检查。
辅助 dependency guard 为 0 GPU `afterany:<all five formal jobs>`，只对 `sacct` 证明 dependency-cancelled
且从未启动、同时缺 worker status 的下游原子补 status/QQ-attempt 证据；它不属于五个正式 allocation，
也不改变四条正式 `afterok` 边。

### 3.6 测试代码草稿

已新增 terminal-review、Environment integration、stage prompt、独立 H0 golden、summary v2、receipt
v2 和 P1 execution tests。CPU fixture manifest 已加入 GT-boundary、plan-null trigger/selection/prompt/
raw-call、真实 H1 empty-mask、network retry、final-region raw reconstruction 与 dependency-guard 的必需
test ID。本次只做 source-presence 静态核对，当前 70 个 ID 均能在对应源码中找到精确 `def`；这不
代表测试执行或通过。冻结前 CPU worker 仍须重复同一 preflight。没有任何测试真正运行过，也没有
形成 CPU fixture receipt。

## 4. 尚未产生的实验与产物

以下五个保留目录均不存在：

- `outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures`
- `outputs/CA_0802(3)-qwen3vl-h0-base-smoke1`
- `outputs/CA_0802(4)-qwen3vl-h1-base-smoke1`
- `outputs/CA_0802(5)-qwen3vl-h0-base-core3`
- `outputs/CA_0802(6)-qwen3vl-h1-base-core3`

因此当前没有：

- CPU fixture 日志或 receipt；
- smoke H0/H1 trajectory、summary 或 evaluation receipt；
- CORE3 H0/H1 trajectory、summary、pair comparison 或 evaluation receipt；
- P1 实验结果分析；
- P1 experiments registry/workbook 更新；
- P1 结果 commit。

本次只读 `squeue -u guisongxia` 检查受到当前工具沙箱限制，返回
`Error creating slurm stream socket: Operation not permitted`。这不是 Slurm 控制面 DOWN 的证据，
但新会话在任何提交前必须重新执行 `scontrol ping` 和精确的重复作业检查；不得据本文件假设队列
状态。

## 5. 当前文档/配置的真实状态

- `QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json` 明确仍为
  `record_status=draft`、`execution_readiness=not_execution_ready`，且
  `implementation_commit=null`。
- `QWEN3VL_P1_EXECUTION_AMENDMENT_0802.md` 明确写着“当前不得提交任何 P1 Job”。
- `QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json` 已纠正为
  `configuration_status=draft`、`execution_readiness=not_execution_ready`、
  `p1_execution_authorized=false`；不存在“implementation complete”的声明。
- resolved config 中 model/tool/worker/builder/validator 的 hash binding 仍为 `null`。
- amendment 中 node/partition/CPU/memory/time 等仍有 `null`；resolved config 虽起草了
  `gpu48/gre` 与具体资源，但尚未回填、复核和冻结到 amendment。
- CPU fixture manifest 已加入独立 H0 golden receipt 路径、receipt raw/canonical hash、P0 receipt
  hash 和 validator hash；其 `status=frozen` 仅表示 fixture **设计清单**冻结，新增字段明确整体仍为
  `not_execution_ready` 且不授权 P1。CPU worker 会在 unittest 前运行 validator，但尚无 Slurm success
  receipt，整个 manifest/worker 的最终 freeze hash 仍待 implementation/freeze commit 后闭环。

## 6. 必须先解决的 blocker

### 6.1 独立 H0 golden fixture 已实现，尚待 Slurm 验证

已新增只读 `QWEN3VL_H0_V12_1_GOLDEN_FIXTURE_RECEIPT_0802.json`：它显式固定单 region
finish/plan-null 输入、region/call/repair/diagnosis/plan/action/candidate/gate/termination 行为向量，
并绑定 P0 成功 commit、八个历史 source blob 及 P0 acceptance receipt raw/canonical hash。独立
validator 不导入当前 verifier/decorator/runner，且内置 exact oracle，能够拒绝 receipt 与 manifest
同步重哈希的协同篡改。新 CPU tests 分别让当前 legacy H0 与 H0 identity decorator 直接对照该
receipt，而不是只互相比对。当前仅做静态检查，validator、parity 和 tamper fixtures 仍须通过
Slurm CPU Job；在此之前 6.1 不能视为执行门已通过。

### 6.2 rollout 前 label 零访问修复已落盘，尚待 Slurm 验证

`tools/run_levir_change_agent.py::_validate_inputs()` 的草稿已移除 rollout 前
`input_root/label_cvt/<sample>.png` 的 existence/stat 检查；新增 offline validation 只允许在已原子落盘
且内容有效的 `rollout_complete.json` 之后打开 label。CPU suite 已加入 runner raw `Path.exists` spy、
marker-before-open 和 runtime-manifest pre-rollout no-hash fixtures。它们尚未在 Slurm 执行，因此 6.2
只能记为“实现草稿已落盘”，不能记为通过。

### 6.3 plan-null trigger 已有 raw 重建草稿，严格 call-chain 仍待闭环

`terminal_review_verifier.py` 已新增从 staged type/schema/trace/backend calls 重建 plan-null 的路径，
并拒绝缺 trace、非 null plan、缺显式 retry counter，以及无对应 repair 的 regional error。剩余冻结门
是逐条 raw call 必须完整、initial/最多一次 repair 链必须可回放、不能通过跳过 malformed call 或只信
normalized `trace.plan` 得到 trigger。manifest 已列入 complete-call-group 与 identity 负例；在这些 def
全部落盘并通过 Slurm 前，其他 `suggested_action=None` 仍不得泛化为合法 trigger。

### 6.4 plan-null Environment-order selection 已接入，prompt/replay 严格性仍待闭环

当前草稿已按预注册实现：

1. 优先选 Environment 确定性顺序中的下一条尚未审 region；
2. 只有没有未审 region 时，才用版本化的 alternative-diagnosis prompt 复核首个 unresolved
   region；
3. 新 prompt/region identity 必须独立记录，不能只是排序变化。

两个 prompt version ID 已进入 raw terminal-review context，plan-null round two 只复核一个 selected
region，不再重排后全量复核。剩余 blocker 是 `_stage_prompt()` 必须产生两套不同且冻结的任务文本并
拒绝 unknown/mismatched version；selection replay 必须要求 trace catalog 与 Environment catalog 精确
同序，且单-region review 不能借 completion gate 声称全状态覆盖。manifest 已登记相应正反 test ID。

### 6.5 versioned empty-mask proposal 与真实 H1 fixture 已落盘，尚待完整验证

Environment 草稿已在 dual-empty initial state 生成确定性 `s0...s8` synthetic scan proposals，并记录
`qwen3vl_empty_dual_mask_scan_grid_v1`；integration suite 已改用真实
`TerminalReviewVerifier(StagedQwenVerifier(...))` 的 H1 initial-finish 路径。冻结前仍须确保 proposal
version 被 prompt/raw evidence 绑定、H0 prompt 不受 H1 metadata 影响，并修完/静态核对 fixture 后由
Slurm 实际运行。没有 CPU receipt 前，6.5 仍不是已通过执行门。

### 6.6 `final_region_protocol_valid` raw 重建草稿已出现，独立双路径尚未闭环

summary builder 已开始从 hash-bound trajectory/review history 重建该 gate，并已有 H0 declared-value
flip、malformed backend evidence 和 H1 budget-exhausted/no-round-two 负例。冻结要求仍是 builder 与独立
validator 各自实现重建而非共享一个受信 helper：H1 只有 invoked、raw round-two evidence 完整且
schema-valid 时为 true；H0 必须有完整 staged schema/trace/region calls，programmatic/no-trace、hard-
gate-only 和 invalid output 均为 false。evaluation receipt 还必须拒绝 bound trajectory/episode tamper。
H1 的 final effective envelope 自身必须携带合法 terminal review；若 later effective envelope 未经 review，
不得回退复用旧 envelope 的 review。manifest 已登记这些精确 test ID，但尚无 Slurm 结果。

### 6.7 显式 network retry 重建已落盘，尚待 Slurm 与 hosted 兼容审计

local backend call audit 已显式记录 `network_retry_attempt_count=0`；round-one、round-two 和 runner
accounting 草稿均从 backend evidence 求和，并拒绝缺失、bool、负数和超预算计数。P1 固定 local
backend，因此冻结 fixture 必须证明零 retry 路径。若保留 hosted/Bailian runner，还需统一其 logical-call
retry audit，避免一个网络 retry 的多条 history 与当前严格计数语义冲突。

### 6.8 独立 validator 全链路仍未实证

即使若干检查已经写入草稿，仍需由独立 validator 和 tamper fixtures 证明：

- H1 eligible terminal trigger 时 `terminal_review=null` 必须 fail closed；
- `counter_scope`、trigger、round-two kind/outcome、observed/effective outcome、transition 全部重建；
- exactly-once、no-third-round、fresh context、nonduplicate prompt/region；独立性 prior 覆盖经 call-group
  验证的全部 round-one initial 与 repair 身份，failed initial 也按 trace group 归属，不能借 repair 绕过；
- safe plan 只返回原 Environment flow，且 unchanged hard gates 可拒绝它；
- analyzer 能展开 nested `round_one_evidence`/`round_two_evidence`，不漏计也不双计；
- `confirmed_finish → episode_outcome=converged → finish_count` 完整闭环；
- trajectory 的 step 0、best step、raw/offline metrics 和
  `final_region_protocol_valid` 均独立重建；
- `step_index` 必须拒绝 bool；
- 完整 `terminal_reviews` history 与兼容字段 `terminal_review` 一致；
- summary、evaluation receipt、runtime manifest、status、pair receipt 的逐 artifact tamper 都失败。

### 6.9 receipt builder 最新修改未运行

`tools/build_qwen3vl_evaluation_receipt.py` 和
`tests/test_qwen3vl_evaluation_receipt_v2.py` 是本轮末尾新增；只做过人工字段对齐，最后一次
status-path 小修后连 AST/shell 静态检查也未重跑。必须在 Slurm CPU fixture 中验证 builder 与当前
validator/worker CLI 完全一致，尤其 artifact 顺序、字段名、runtime tool hash、model hash 和
stable status 绑定。

### 6.10 dependency guard 已实现，尚待 Slurm fixture 实证

submitter 当前一次性提交 held 的完整 `afterok` 链。若上游失败，Slurm 的 invalid dependency 可能
使下游 worker 根本不启动，从而无法由 worker 发送 cancelled/failed QQ 终态通知。需要在不破坏
严格串行和 output-reservation 合同的前提下，由 submitter/monitor 为未启动下游补充可审计的终态
status 与 QQ attempt，或改成逐阶段放行；必须补 fixture。

2026-08-02 实现草稿已增加独立的零 GPU `CA0802P1DependencyGuard`：它不属于五个正式
allocation，以 `afterany` 依赖全部五个正式 Job，而正式四条 `afterok` 边保持不变。guard 只对
`sacct` 证明 dependency-cancelled 且从未启动、同时缺 worker status 的下游原子补
`logs/status-<job_id>.txt`，并保存 best-effort QQ attempt receipt；已启动但缺 status 时 fail
closed。fake `sacct`/notifier fixtures 已写入 CPU suite 草稿，但截至本交接更新仍未通过 Slurm
执行，不能把本 blocker 标为实证关闭。

### 6.11 尚未形成冻结 Git/hash 闭环

当前还没有：

- P1 implementation commit；
- 真实 model/tool/worker/builder/validator/config hashes；
- frozen/authorized amendment canonical digest；
- exact freeze commit；
- 更新到 freeze commit 的 clean detached run worktree；
- submitter `--check-only` 成功 receipt；
- CPU fixture success receipt。

在这些缺失项补齐前，任何 P1 submission 都属于越过冻结门。

冻结采用明确的 A/B 合同：amendment 的 `implementation_commit` 记录完整实现提交 A；freeze commit B
必须是 A 的直接子提交（`B^ == A`），只加入真实 hash/resource/authorization binding。冻结文件不得
自引用 B；clean detached run worktree 与 `runtime_commit` 必须等于 B，submission/evaluation receipt
同时记录 A、B，且 receipt v2 hash-bind P1 amendment。

## 7. 新会话建议执行顺序

### 阶段 A：恢复现场，只读核验

1. 完整阅读 `/guisongxia01/pangchao/wangyihan/AGENTS.md`、本文、P1 amendment、resolved config、
   CPU fixture manifest 和 P0 receipt。
2. 在开发仓库运行只读 `git status --short --branch`、`git diff --stat`、`git diff`；确认没有他人
   新增改动，不丢弃本节列出的 dirty files。
3. 核验正式 worktree 仍为 clean detached `00164f8...`。
4. 核验 `(2)`–`(6)` 输出路径仍全部不存在。
5. 运行 `scontrol ping`；只用 `squeue -u guisongxia`/明确 Job ID 检查队列和重复 job。若控制面
   报错，先处理连接，不要创建输出目录或反复提交。

### 阶段 B：修复协议 blocker

按第 6 节继续：GT boundary、golden H0、empty-mask proposal、retry accounting 和 dependency guard 已有
草稿，先完成 plan-null prompt/catalog/raw-call strictness 与 builder/validator 双路径 final-region
reconstruction；再统一 summary/receipt/pair/analyzer 的 tamper contracts。每次代码修改同步更新
amendment、resolved config 和 CPU fixture manifest。

所有测试仍必须通过 Slurm；管理节点只能做编辑、文件检查和轻量静态诊断。

### 阶段 C：形成 implementation/freeze commits

1. blocker 全部有对应 fixtures 后，做一次实现范围审计并创建 P1 implementation commit A；该 commit
   必须包含完整实现、tests 与维护文档，并由 amendment 的 `implementation_commit` 记录。
2. 从该 commit 和真实文件计算 model/tool/worker/builder/validator/config hash；禁止填猜测值。
3. 更新 amendment/config/fixture manifest，消除所有 placeholder，设置真实状态并计算 canonical
   digest。
4. 创建 A 的直接子提交 freeze commit B（`B^ == A`）；冻结文件不得自引用 B。submission/receipt
   必须另行同时记录 A 与 B，并 hash-bind P1 amendment。
5. 将正式 worktree 更新到 exact B，验证 detached、clean、HEAD/runtime_commit/worker/hash 全部一致。

若 CPU fixtures 暴露问题，应回到新 implementation commit + 新 freeze commit，不得修改已冻结
worktree 后继续复用旧 digest。

### 阶段 D：Slurm preflight 和 P1 串行执行

1. 先运行 submitter 的 `--check-only`，确认 controller、`gpu48`、A800、模型/工具、输入、hash、
   worktree、重复 job 和输出目录 absence。
2. 只有 check-only 全部通过后才提交正式链。
3. CPU fixtures 必须先成功并生成完整 receipt；任何失败都不释放 smoke。
4. smoke 固定 `test_1_1`，顺序 H0 → H1；H1 必须实际出现一次合法独立 round two。
5. smoke hard gates 全过后，CORE3 才按
   `test_20_15, test_1_1, test_50_8` 顺序运行 H0 → H1。
6. 任一上游失败，停止后续阶段；发送包含 job/status/error/elapsed/artifact 的 QQ 终态通知。

### 阶段 E：独立验证、分析和 Git 收尾

1. 在与 builder 独立的路径验证 H0/H1 summary、evaluation receipts 和 pair receipt。
2. 先报告协议/安全门，再报告行为和 IoU/F1；不得用 GT 指标反调在线 trigger/region/action。
3. 更新 `doc/training_pipeline/0802_third_try` 的 P1 结果报告和失败分析。
4. 更新 `experiments` registry/workbook，并绑定 job IDs、commit、hash、receipts 和 artifact paths。
5. 创建独立结果 commit，确认开发 worktree clean后再向用户汇报。
6. 到此停止；没有新的用户授权，不进入 P2 或训练。

## 8. 新会话启动检查清单

- [ ] 已阅读 workspace `AGENTS.md` 和本文
- [ ] 已确认开发仓库 dirty 内容完整、没有覆盖用户改动
- [ ] 已确认正式 worktree clean detached 且仍在记录的 commit
- [ ] 已确认 `(2)`–`(6)` 输出不存在
- [ ] 已确认没有同名 P1 job，Slurm controller 可用
- [ ] 已逐项关闭第 6 节 blocker
- [ ] 已补独立 H0 golden receipt 与真实 H1 empty-mask fixture
- [ ] 已通过 Slurm CPU fixtures，而非管理节点本地测试
- [ ] 已形成 implementation commit 和独立 freeze commit
- [ ] amendment/config/manifest 无 placeholder，所有 hash 真实可复算
- [ ] 正式 worktree 已更新到 exact freeze commit 且 clean
- [ ] submitter `--check-only` 全过
- [ ] 五阶段链严格串行，峰值 GPU 为 1
- [ ] 每个终态有 status receipt、logs 和 QQ attempt
- [ ] summary/receipt/pair 均经独立 validator 和 tamper checks
- [ ] 实验报告、registry 和结果 commit 已完成

## 9. 最重要的禁止事项

- 不从 dirty 开发仓库运行正式实验。
- 不覆盖任何已有输出，不复用失败链的 `(2)`–`(6)` 目录。
- 不在管理节点跑测试、数据处理、模型推理或评测。
- 不在 rollout-complete 前访问或 stat `label_cvt`。
- 不放宽 parser、geometry、locality、candidate-benefit-without-harm 或 rollback。
- 不增加第三轮、network retry、tool retry 或 environment-step budget。
- 不因为 CORE3 指标更好就修改预注册 trigger/selection。
- 不把 CORE3 结果写成泛化结论。
- 不打印、复制或提交 SMTP/API/SSH 等凭据。
- 不使用 `wangyihan` 作为 Slurm 用户；固定使用 `guisongxia`。
- 未经用户新授权，不进入 P2、SFT/post-SFT、EXTENDED5 或 locked final。

## 10. 2026-08-02 后续实现记录：冻结 hash 与实际文件绑定

runtime input manifest 的 pre-rollout 构建现在会先验证 resolved config 为
`frozen`、`execution_ready`、`p1_execution_authorized=true`，并要求
`freeze_commit_hash_bindings` 三组 hash 全部已解析为合法 SHA-256。随后对冻结的 model 子集、完整
tool exact key set 和 evaluation worker 逐文件重新计算 hash；任一声明或外部文件漂移都会在 rollout
前 fail closed。evaluation worker 的 21 个 `--tool-file` 名称已与 config 的冻结 tool 集合对齐，runtime
manifest 同时记录这些 tool 的绝对路径，供后续独立复核。

submitter、CPU worker、evaluation worker 与 dependency guard 现各自读取 resolved config，并将
workspace、artifact repository、clean run worktree、output/allocation 与适用的 input/model/checkpoint/BPE
路径按 realpath 精确比对。`PYTHON` 必须解析到 config 的 `omniovcd_python`；evaluation 同时绑定
`OMNIOVCD_PYTHON` 与 `SEGAGENT_PYTHON`。环境变量只能提供解析到同一冻结实体的别名，不能静默改换
执行路径；这些检查发生在 rollout 与任何 label 访问之前。

evaluation receipt builder 与独立 validator 现在分别比较 resolved config、P1 amendment、runtime
manifest、receipt provenance 中的 model/tool/evaluation-worker bindings，并重新读取实际 model/tool
文件计算 hash。amendment 的 `worker_sha256` 必须使用
`cpu_fixtures/evaluation/dependency_guard` canonical 三键，并与 config exact mapping 一致；evaluation 项
还必须与实际 worker/provenance 一致。fixtures 已加入 rollout 前外部 model/tool/worker 漂移、tool exact-set 漂移、config
授权状态漂移，以及 receipt 构建前后外部 model/tool、amendment model/tool 声明和重绑定 config 授权
篡改的负例。上述 fixture 尚未通过 Slurm CPU 作业执行；CPU fixture manifest 当前登记 70 个 required
test IDs，仍须由正式 CPU 作业形成执行 receipt。

## 11. 2026-08-02 P1 implementation 预提交 Slurm 回归

在不创建正式 `CA_0802(2)`–`(6)` 输出、也不申请 GPU 的独立 diagnostic 路径上完成了三次
`gpu48/gre` CPU 回归：

- Job `53035` 在 `test_local_verifier_analysis.py` 暴露两个错误消息与统一
  `visual input scope` 断言不一致；只调整错误消息，不改变验证逻辑。
- Job `53036` 在 `test_qwen3vl_p1_execution.py` 暴露 H1 synthetic round-two context 缺少
  `proposal_version=None`、dependency guard 缺少显式 Python executable identity 表达式，以及
  submitter source fixture 与实际字典写法不一致；均按既有 fail-closed 合同修正。
- Job `53037` 为最终成功复跑：Slurm `COMPLETED/0:0`，2 CPU、16G、0 GPU、耗时 24 秒。
  独立 H0 golden validator 有效；11 个 targeted patterns 全部通过；full regression 为
  `466 tests / OK / 3 skipped`；四个 P1 shell 文件的 `bash -n` 通过。

当前 manifest 已统一为 70 个唯一 required test IDs 和 11 个 targeted patterns。该 diagnostic success
证明当前 implementation fixture 在 Slurm 计算节点通过，但不是正式
`CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures` receipt，也不改变 amendment/config/manifest 的
`draft / not_execution_ready / unauthorized` 状态。

implementation commit A 已创建为
`f97f9f090369bc5d97f2a60b3d247aae6166b16e`。随后零 GPU Job `53047` 在 `gpu48`
从 clean A 读取并计算 1 个 model 文件、21 个 tool/checkpoint 文件和 3 个 worker 的真实 SHA-256，
Slurm `COMPLETED/0:0`，耗时 46 秒。hash/resource/implementation 证据已经回填到仍保持
non-authorizing 的 amendment/config。用户随后已明确要求“执行上述流程，得到对比结果，做好记录和
管理”，授权正式 P1 严格串行链；仍须先形成 A 的直接子 non-authorizing freeze candidate 并把 clean
detached worktree 对齐，才允许翻转三层 authorization、计算最终派生 digest，并 amend 为最终 freeze B。

## 12. 2026-08-02 最终 freeze B 激活记录

本节是首次 final freeze B 的历史激活记录；当前权威恢复点见第 0 与第 15 节。implementation A、precommit CPU 回归、外部 hash Job、
non-authorizing candidate 及其 detached worktree 均已完成独立核验；只读审计未发现新增 blocker。
用户明确授权以 `git commit --amend` 将 candidate 替换为最终 freeze B，并立即把 detached run
worktree 同步到 B。包含本节的 amend commit 就是 B；B 的 hash 由 Git 与后续 receipts 外部记录，
不写入自身冻结文件。

冻结后的三层机器状态必须同时保持：

```text
resolved config: frozen / execution_ready / p1_execution_authorized=true
amendment:       frozen / execution_ready / p1_execution_authorized=true
fixture manifest:frozen / execution_ready / p1_execution_authorized=true
```

固定摘要锚点为：

- config raw：`<hash omitted>`
- fixture canonical：`<hash omitted>`
- amendment raw：`<hash omitted>`
- amendment canonical：`<hash omitted>`

最终 B 形成后必须立即验证 `B^ == A`、branch `HEAD == B`、run worktree `HEAD == B`、detached、
clean，再从 run worktree 执行 submitter `--check-only`。只有全部成功才可提交正式链。正式 CPU
receipt 仍待运行时生成；GPU 阶段不能凭 precommit Job `53037` 解锁，只能由正式 CPU Job 的
`afterok` 依赖依次释放。到 P1 结果登记完成即停止，不进入 P2、训练、EXTENDED5 或 locked final。

### 12.1 训练数据限制（后续实验必须保留）

当前训练数据主要是彼此孤立的单阶段 stage snapshots，缺少同一 episode 内机械关联的完整
`audit → action → candidate → next-state` 流程轨迹监督。单阶段样本因此未必足以让模型学会解决
完整多阶段任务。这是重要的、已有数据结构支持的解释方向，但目前仍是**待验证假设**，不能写成
唯一根因或既定结论；只有满足预注册条件时，才能通过 token/来源/optimizer steps 匹配的 T3
trajectory-linked 对照验证。

## 13. 2026-08-02 首次正式链失败与 retry1 计划

### 13.1 append-only attempt ledger

| role | Job | Slurm state | worker/scientific status | artifact |
|---|---:|---|---|---|
| formal CPU | `53064` | `COMPLETED / 0:0` | engineering success；296 targeted + 466 full tests | `outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures` |
| smoke H0 | `53065` | `FAILED / 3:0` | rollout/summary valid；receipt build engineering failure；not canonical | `outputs/CA_0802(3)-qwen3vl-h0-base-smoke1` |
| smoke H1 | `53066` | `CANCELLED / 0:0` | never started；not evaluated | `outputs/CA_0802(4)-qwen3vl-h1-base-smoke1` |
| CORE3 H0 | `53067` | `CANCELLED / 0:0` | never started；not evaluated | `outputs/CA_0802(5)-qwen3vl-h0-base-core3` |
| CORE3 H1 | `53068` | `CANCELLED / 0:0` | never started；not evaluated | `outputs/CA_0802(6)-qwen3vl-h1-base-core3` |
| auxiliary guard | `53069` | `FAILED / 1:0` | guard compatibility failure；QQ attempt exit 0 | `(2)/dependency_guard_receipt.txt` |

旧目录及其 submission/status/summary/logs 不得修改、移动、补 receipt 或复用。`53065` 的 H0
summary 显示 requested/scored `1/1`、GT/safety/tool/requested gates 为 0、final-region valid `1/1`、
round two `0`，IoU/F1 均为 `0.0` 且模型直接 finish；这些只用于诊断，不是可与 H1 比较的 canonical
结果。

### 13.2 根因和最小修复

1. submitter 原 receipt identity 使用 `A && H0 || B && H1 || N/A`；Bash 的 `&&`/`||` 同优先级且
   左结合，使 H0 同时输出 H0 与 H1 字符串。retry1 改为由 hash-bound resolved config 的
   `frozen_identity` 产生两个 exact 字段，并新增防拼接 fixture。
2. 本集群的 Slurm controller 仍保留 `Reason=DependencyNeverSatisfied` 与
   `Dependency=afterok:<job>(...)`，但 SlurmDBD/sacct 的终态 Reason 为 `None`。retry1 guard 只有在
   sacct 同时证明 `CANCELLED + Start missing` 后，才允许用 live scontrol 的唯一 `JobId`、`JobState`、
   `Reason`、`Dependency` 字段补证；额外边、重复字段、错误前驱或任一不匹配继续 fail closed。合成
   status 还必须持久化 controller state/reason/dependency，避免 controller 记录过期后证据不可复核。

### 13.3 retry1 唯一输出

```text
outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry1
outputs/CA_0802(3)-qwen3vl-h0-base-smoke1-retry1
outputs/CA_0802(4)-qwen3vl-h1-base-smoke1-retry1
outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry1
outputs/CA_0802(6)-qwen3vl-h1-base-core3-retry1
```

科学 treatment、样本顺序、资源和串行依赖不变。retry1 必须完整重跑 formal CPU，不能用 `53064`
直接释放新 GPU Job，因为 worker/config hashes 与 freeze commit 都将变化。

### 13.4 当前 release gates

零 GPU retry1 precommit 首次 attempt `53102` 在 implementation candidate
`25eaa5982ce03d40cf18bb14ed804c7f59c24f4e` 上 `FAILED / 1:0 / 00:00:14`。失败仅来自新增
identity fixture 调用 submitter receipt writer 时漏设 `PARTITION`、`NODE`；golden validator 和此前
10 个 targeted patterns 已通过。该 attempt 只读保留，补齐 fixture 后仍须从新 implementation commit
重新跑完整 72-ID、11-pattern 和 full regression，不能把 `53102` 的部分通过拼成成功证据。

- [x] implementation A2 `15cdca7ddc4611ad773576dd582e5b4513188d8e` 包含修复、72 个 required IDs 与维护文档
- [x] clean A2 上的零 GPU Slurm precommit Job `53103` 全过（298 targeted；468 full；3 skipped）
- [x] hash Job `53106` 从 clean A2 重算 25 个 model/tool/worker 条目
- [x] candidate `03f95e53…` 的 activation content 三层收敛为 frozen/execution_ready/true，待本次 amend 生成最终 B2
- [ ] detached run worktree 精确同步 B2、clean，并通过 submitter `--check-only`
- [ ] 五个 retry1 输出在提交前均不存在

全部勾选前不得提交正式 retry1 作业。完成 P1 对比、独立 receipts 验证、失败/成功 attempts 登记、
workbook 重建与结果 commit 后停止，不进入 P2 或训练。

## 14. 2026-08-02 retry1 第二次失败与 retry2 当前计划

### 14.1 append-only retry1 attempt ledger

| role | Job | Slurm state | worker/scientific status | artifact |
|---|---:|---|---|---|
| formal CPU | `53109` | `COMPLETED / 0:0` | engineering success；298 targeted + 468 full | `outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry1` |
| smoke H0 | `53110` | `COMPLETED / 0:0` | summary/receipt valid baseline reference；pair 尚不存在 | `outputs/CA_0802(3)-qwen3vl-h0-base-smoke1-retry1` |
| smoke H1 | `53111` | `FAILED / 3:0` | rollout complete；`result=building_summary`；not canonical | `outputs/CA_0802(4)-qwen3vl-h1-base-smoke1-retry1` |
| CORE3 H0 | `53112` | `CANCELLED / 0:0` | never started；not evaluated | `outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry1` |
| CORE3 H1 | `53113` | `CANCELLED / 0:0` | never started；not evaluated | `outputs/CA_0802(6)-qwen3vl-h1-base-core3-retry1` |
| auxiliary guard | `53114` | `COMPLETED / 0:0` | synthesized `53112/53113`；unresolved none | `(2)-retry1/dependency_guard_receipt.txt` |

旧 retry1 目录不得修改、移动、补 receipt 或复用。`53110` 可作为有效 H0 reference，但没有 H1
receipt 和 pair receipt，故整个 P1 科学结果仍是 `not_evaluated`。

`53111` exact error 为：

```text
p1_harness_audit.json.per_sample.test_1_1.terminal_reviews[0] trigger differs from raw round-one evidence
```

detached run worktree 在 B3 提交前快照仍 clean、detached、exact B2；B3 形成后必须立即精确同步，
并在 `--check-only` 通过前保持不提交。registry 的科学状态保持 `not_evaluated`，首次链与 retry1
两组六个 attempts 都必须 append-only 保留，不能覆盖。

### 14.2 根因与最小修复

生产 `StagedQwenVerifier` 将 protocol-level `StageTrace.decision` 与最终公开 `VerifierOutput` 分层保存：
initial trace 合法值是 `comparison=initial, accept=false, stop=false`，而 `_output()` 根据完整 finish plan、
completion gate 与全 none diagnoses 派生 `accept=true, stop=true`。scheduler 使用后者并正确触发
`provisional_finish`。summary builder 和独立 validator 却错误地从 trace 要求 true/true，因此在真实
H1 initial finish 上系统性失败。

retry2 的 production 修复仅修改：

1. `tools/summarize_qwen3vl_e2e_run.py` 的 trigger rebuilder；
2. `tools/validate_qwen3vl_e2e_summary.py` 的独立 trigger rebuilder；
3. `tests/test_qwen3vl_e2e_summary_v2.py` 的 production-shaped initial/candidate 正负 fixtures。

模型、输入、seed、generation、H0/H1 treatment、round/tool/environment budgets 和全部 safety gates 不变。

### 14.3 retry2 precommit candidate 失败

implementation candidate `47d67d45be5011eb91008fbd46c1692cef86e00c` 上的零 GPU Job `53134`
为 `FAILED / 1:0 / 00:00:15`。它已通过 golden validator 及此前 targeted patterns，随后在
`test_submitter_arm_identity_is_exact_and_non_concatenating` 停止，未运行 full regression：

```text
resolved config tool_files_sha256 contains an unresolved hash
```

retry2 草稿按合同把已改变的 summarizer 和 summary validator hash 置 null，等待 clean implementation
hash Job；synthetic submitter fixture 却仍只替换旧版会变化的 worker/submitter hash，因而把两个新 null
复制进私有 frozen clone。production submitter 正确拒绝了不完整绑定。修复仅让该私有 fixture
确定性补齐所有 draft-null tool hashes，并把 fixture 输出名同步到 `-retry2`；不改 production 路径。
`53134` 的日志和 status append-only 保留，不能与后续通过项拼接；新 clean A3 必须完整重跑。

### 14.4 retry2 唯一输出与 release gates

```text
outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry2
outputs/CA_0802(3)-qwen3vl-h0-base-smoke1-retry2
outputs/CA_0802(4)-qwen3vl-h1-base-smoke1-retry2
outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry2
outputs/CA_0802(6)-qwen3vl-h1-base-core3-retry2
```

- [x] implementation A3 `e31fcb01d34cc28924e9c4189f319d7f44af4911` 包含最小修复、73 required IDs 与维护文档
- [x] clean A3 的零 GPU Job `53135` 全过（299 targeted、469 full、3 skipped）
- [x] clean A3 的 25-entry hash Job `53140` 成功，所有 changed hash 已回填
- [x] 包含本记录的直接子提交形成 freeze B3，三层状态收敛为 frozen/execution_ready/true
- [x] detached run worktree 精确同步 B3、clean，并通过 submitter `--check-only`
- [x] 五个 retry2 输出在 freeze 前均不存在，且没有重复同名 Job

冻结摘要：config raw
`<hash omitted>`；fixture canonical
`<hash omitted>`；amendment raw
`<hash omitted>`；amendment canonical
`<hash omitted>`。

全部勾选前不得提交 retry2 正式链。正式链仍从 formal CPU 开始并保持峰值 1 GPU。

### 14.5 P1 完成后的用户授权边界

用户已授权在当前 P1 对比、独立验证、registry/workbook/报告和 Git 收尾完成后，基于届时最新框架
重新开展“训练数据创建 → 训练 → 评测 → 失败轨迹分析 → 下一轮改进”。下一阶段优先把当前孤立的
stage snapshots 扩展为同 episode 机械关联的完整
`audit → action → candidate → next-state` 轨迹，同时保留单阶段控制，验证而非预设 T3 假设。

每轮都必须有版本化 data manifest/hash、来源和泄漏审计、预注册对照、Slurm Job/资源/状态、模型
checkpoint lineage、评测 receipt、失败轨迹分类、registry/workbook 与独立 Git 提交。该授权尚未执行，
也不授权跳过新阶段的 readiness/safety gates、直接进入 EXTENDED5 或 locked final。

## 15. 2026-08-02 retry2 第三次 fail-closed 与 retry3 当前计划

### 15.1 B3 与 retry2 append-only attempt ledger

- implementation A3：`e31fcb01d34cc28924e9c4189f319d7f44af4911`
- final freeze B3：`64871db9f03d19d6ddb9ab3edf502c3ecb698ee5`
- 父子关系：`B3^ == A3`
- detached run worktree：clean、detached、exact B3；B3 submitter `--check-only` 已通过

| role | Job | Slurm state | worker/scientific status | artifact |
|---|---:|---|---|---|
| formal CPU | `53141` | `COMPLETED / 0:0` | receipt valid；299 targeted + 469 full + 3 skipped | `outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry2` |
| smoke H0 | `53142` | `COMPLETED / 0:0` | summary/evaluation receipt valid | `outputs/CA_0802(3)-qwen3vl-h0-base-smoke1-retry2` |
| smoke H1 | `53143` | `COMPLETED / 0:0` | summary/evaluation receipt/smoke pair valid | `outputs/CA_0802(4)-qwen3vl-h1-base-smoke1-retry2` |
| CORE3 H0 | `53144` | `COMPLETED / 0:0` | standalone summary/evaluation receipt valid | `outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry2` |
| CORE3 H1 | `53145` | `FAILED / 1:0` | `running_rollout`; `RecursionError`; not evaluated | `outputs/CA_0802(6)-qwen3vl-h1-base-core3-retry2` |
| auxiliary guard | `53146` | `COMPLETED / 0:0` | synthesized none；unresolved none | `(2)-retry2/dependency_guard_receipt.txt` |

CORE3 H0 standalone reference 的 micro IoU/F1 为 `0.7422312 / 0.85204673`，macro IoU/F1 为
`0.28212851 / 0.30560087`。这些指标有效，但没有 canonical CORE3 H1 summary/receipt 或 CORE pair，
所以整个 P1 仍为：

```text
scientific_outcome = not_evaluated
```

不得把 smoke pair、CORE3 H0 reference 或 H1 partial rollout 提升或拼接为完整科学结论。上述六个
Jobs、五个 retry2 目录及其 receipts/failure/status 全部只读保留。

### 15.2 53145 精确根因与修复边界

失败样本是 `test_20_15`。H1 第一轮给出非终止 `false_positive_change → negative_point`，没有 terminal
trigger，也没有 round two。passthrough 分支原实现为：

```python
final_evidence = round_one_evidence
terminal_review["round_one_evidence"] = round_one_evidence
final_evidence["terminal_review"] = terminal_review
```

因此形成对象自环：

```text
execution.verifier_evidence
└─ terminal_review
   └─ round_one_evidence
      └─ terminal_review
         └─ round_one_evidence
            └─ ...
```

`Trajectory.save()` 已先写 `step_000.npy`，随后在 JSON 转换中递归溢出；没有产生 trajectory JSON、
`rollout_complete.json`、summary、evaluation receipt 或 pair receipt。`.run_incomplete`、failure/status
证据正确阻止该目录被误判为科学成功。

retry3 的最小功能修复是：

```python
final_evidence = copy.deepcopy(round_one_evidence)
```

新增 `test_h1_nonterminal_passthrough_trajectory_is_acyclic`，覆盖真实
`Environment.reset() → Trajectory.save() → json.loads()`，并检查外层与 nested round-one evidence
不是同一对象。第 74 个 required ID 已进入 manifest。通用 `_json_safe` active-recursion-stack detector
可作为后续防御增强；本次不使用 cycle marker，不删除字段，不把坏证据包装成成功 trajectory。

### 15.3 retry3 非授权状态与唯一输出

三层机器状态当前必须是：

```text
resolved config: frozen / execution_ready / p1_execution_authorized=true
amendment:       frozen / execution_ready / p1_execution_authorized=true
fixture manifest:frozen / execution_ready / true
```

`terminal_review_verifier` hash 已绑定为
`<hash omitted>`。新输出只能是：

```text
outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry3
outputs/CA_0802(3)-qwen3vl-h0-base-smoke1-retry3
outputs/CA_0802(4)-qwen3vl-h1-base-smoke1-retry3
outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry3
outputs/CA_0802(6)-qwen3vl-h1-base-core3-retry3
```

### 15.4 release gates

- [x] implementation A4 `a88404ac6a014f42692d1119d54de42c980fb310` 包含最小修复、74 required IDs 与维护文档
- [x] clean A4 的零 GPU Job `53151` 完整通过 74-ID source gate、11 patterns、full regression、
  golden validator 与 shell syntax；实测 300 targeted、470 full、3 skipped
- [x] clean A4 的 25-entry hash Job `53152` 成功，terminal-review verifier 新 hash 已回填
- [x] config/fixture/amendment 的真实 hashes、Job IDs 和 counts 已交叉复算
- [x] 包含本记录的提交形成 A4 直接子 non-authorizing candidate
- [x] 按用户授权 amend candidate；包含本记录的 amend 提交形成 final freeze B4
- [ ] detached run worktree exact B4、clean、detached，submitter `--check-only` 通过
- [ ] 五个 retry3 输出在提交前均不存在，且没有同名重复 Job

全部门禁通过前不得提交正式 retry3 作业。通过后也必须从 formal CPU 开始完整重跑五段链；不得复用
`53141–53144`，不得拿 B3 H0 与 B4 H1 组成 pair。formal CPU、smoke H0、smoke H1、CORE3 H0、
CORE3 H1 仍以 `afterok` 严格串行，峰值 1 GPU；guard 仍是独立 0-GPU `afterany`。

### 15.5 P1 后独立训练迭代

只有 retry3 完整 pair、独立 validation、失败/成功 attempts、registry/workbook、结果报告和结果 Git
commit 都完成后，才执行用户已授权的最新版框架迭代：

```text
训练数据创建 → 训练 → 评测 → 失败轨迹分析 → 下一轮改进
```

核心待验证假设是：孤立单阶段 snapshots 不足以教授完整任务，应增加同 episode 机械关联的
`audit → action → candidate → next-state` 轨迹监督，同时保留单阶段 control。每轮必须重新冻结 data
manifest/hash、来源和泄漏审计、token/optimizer-step 对照、Slurm 资源与 Jobs、checkpoint lineage、评测
receipt、失败轨迹 taxonomy、registry/workbook 和独立 Git commit；不得把该授权解释为跳过 readiness、
EXTENDED5 或 locked final 门禁。

## 16. 2026-08-02 retry3 正式结果与 P1 收尾

### 16.1 B4 正式链 append-only ledger

retry3 使用 implementation A4
`a88404ac6a014f42692d1119d54de42c980fb310` 与 final freeze B4
`5210f89f5ca13ee1e636f5aa6a44631206a80123`，且 `B4^ == A4`。正式提交前已经验证 detached
run worktree clean、detached、exact B4 和 submitter `--check-only`。

| role | Job | Slurm state | elapsed | artifact/result |
|---|---:|---|---:|---|
| formal CPU | `53155` | `COMPLETED / 0:0` | 00:00:25 | retry3 CPU receipt；300 targeted、470 full、3 skipped |
| smoke H0 | `53156` | `COMPLETED / 0:0` | 00:01:16 | H0 summary/receipt valid |
| smoke H1 | `53157` | `COMPLETED / 0:0` | 00:01:58 | H1 summary/receipt 与 smoke pair valid |
| CORE3 H0 | `53158` | `COMPLETED / 0:0` | 00:04:18 | H0 summary/receipt valid |
| CORE3 H1 | `53159` | `COMPLETED / 0:0` | 00:05:59 | H1 summary/receipt 与 CORE pair valid |
| auxiliary guard | `53160` | `COMPLETED / 0:0` | 00:00:02 | synthesized none；unresolved none |

五个 retry3 目录、正式 receipts、pair receipts、submission/status/logs 和 guard receipt 现均为只读历史
证据。不得用后续文档收尾修改它们，也不得与 retry1/retry2 arm 组成新 pair。

### 16.2 独立验证和科学结果

CORE3 pair 的两个 individual receipt validators、pair validator、18 项 controlled-variable checks
和 25 项 hard gates 全部通过；没有跨 attempt 组合。H0 round two 调用为 0，H1 为 2，
`final_region_protocol_valid` 两臂均为 2。

| selected 指标 | H0 | H1 | delta |
|---|---:|---:|---:|
| micro IoU | 0.74223120 | 0.74223120 | 0 |
| micro F1 | 0.85204673 | 0.85204673 | 0 |
| macro IoU | 0.28212851 | 0.28212851 | 0 |
| macro F1 | 0.30560087 | 0.30560087 | 0 |

三样本 selected state、per-sample metrics 与汇总动作统计全部相同。H1 的两次额外调用都是
`finish_confirmation`：fresh context 按协议工作，但模型仍返回 finish，没有 safe plan、tool action
或新 candidate，故 mask 不变。

逐样本机制：

- `test_20_15`：三次成功 tool action 的 candidate 都被安全拒绝，最终 max-steps budget
  exhausted；该样本非 terminal eligible，H1 正确 passthrough；
- `test_1_1`：GT 无变化，但保留 626 个 FP；fresh review 重新确认了错误 finish；
- `test_50_8`：GT 有 166 个正像素，但 selected mask 为空；fresh review 同样重新确认了错误
  finish。

因此 P1 的工程结论为 terminal-review 协议、安全性、证据保存和完整执行链通过；科学结论仅为
**冻结 CORE3 上未观察到 H1 质量提升**。CORE3 是反复使用的三样本诊断 gate，不能解释为
LEVIR-CD 泛化结论。

### 16.3 独立 trajectory analysis attempt ledger

| Job | Slurm state | exact result |
|---:|---|---|
| `53164` | `FAILED / 1:0 / 00:00:02` | 分析器错误期待 `status=complete;`，而 frozen pair validator 合同实际为 `status=valid;` |
| `53165` | `COMPLETED / 0:0 / 00:00:01` | 从 clean B4 完成 3 样本、18 controls、25 hard gates 分析 |

`53164` 没有形成可提升的科学结果；只修正独立分析器对既有 validator 状态字面量的读取后重新执行
`53165`，没有改 frozen receipt。成功产物为：

`/guisongxia01/pangchao/wangyihan/wyh/diagnostics/qwen3vl_p1_retry3_result_0802/trajectory_analysis.json`

其 SHA256 为
`<hash omitted>`。详细解释见
`QWEN3VL_P1_RETRY3_RESULT_0802.md`。

registry/workbook preparation Job `53171` 已把 workbook 重建为 `14/52/21/38`，15 个 targeted tests
通过，workbook SHA256 为
`<hash omitted>`。clean candidate
`03af3c82b92d352de984eec80ba20d5de2c87f6d` 的 closeout Job `53174` 通过 result/analysis、workbook 和
registry targeted 后，在 471 项 full regression 中发现唯一旧测试边界错误：P0 测试把 P0 当时冻结
的 workbook hash 与持续维护的当前 workbook 直接比较。该 attempt 为 `FAILED / 1:0`，不与后续验证
拼接；修复仅保留 P0 历史 hash 字面量并允许当前 registry append-only 演进，不修改任何 P0 frozen
receipt/amendment。

### 16.4 训练数据假设和后续授权

“孤立 stage snapshots 可能不足以教授完整任务，需增加同 episode 的
`audit → action → candidate → next-state` 机械关联轨迹”已经明确登记。本轮轨迹与该解释一致：
模型能重复 audit，却没有产生纠错动作。不过本轮没有训练数据对照，只有三个诊断样本，所以该解释
仍是**待受控验证的假设，而非已证明根因**。

P1 当前只剩以下收尾，不再提交 retry3 正式作业：

- [x] retry3 正式 pair 与所有独立 validators 完成；
- [x] 三样本失败轨迹分析与结果报告完成；
- [x] registry/workbook append-only 登记完成；
- [x] registry/workbook/result 的零 GPU Slurm 收尾验证完成；
- [x] 独立 P1 result commit 与历史 workbook 测试边界 follow-up commit 已形成。

全部收尾已经完成，按用户授权开启独立新阶段：

```text
训练数据创建 → 训练 → 评测 → 失败轨迹分析 → 下一轮改进
```

新阶段优先预注册 isolated-stage control 与 same-episode linked-trajectory treatment，并匹配来源 split、
泄漏策略、token budget、optimizer steps、base checkpoint 和评测协议。每一轮继续保留版本化 data
manifest/hash、Slurm ledger、checkpoint lineage、receipt、registry/workbook 和独立 Git commit。

## 17. 2026-08-03 P2 动作可达性正式结果（gate=false）

P2 已按冻结设计完整执行，implementation A 为
`8252a57b191a2a7c4aa176a3f82cca5218119533`，manifest-only freeze B 为
`cda314ce9caeb6e9dd69ba0d08fe5254a8689fdd`，且 `B^ == A`。P1 B4 detached run worktree没有改动；
P2 exact-B runtime worktree继续保持 detached、clean、exact B。

| 角色 | Job | 资源 | Slurm 终态 | 结果 |
|---|---:|---|---|---|
| manifest materialization | `53268` | 0GPU/2CPU/16G | `COMPLETED / 0:0 / 00:00:24` | 两份 manifest staged/target bytes 与独立验证通过；未访问 label |
| exact-B contract/full regression | `53269` | 0GPU/2CPU/16G | `COMPLETED / 0:0 / 00:01:18` | P2 `37/37`、registry `15/15`、full `508/508`，3 skipped |
| formal candidate generation | `53272` | 1×A800/4CPU/48G | `COMPLETED / 0:0 / 00:19:45` | 36/36 requested/generated/tool-success；27 primary + 9 diagnostic；GT=false |
| strict `afterok:53272` scoring | `53273` | 0GPU/2CPU/16G | `COMPLETED / 0:0 / 00:00:15` | 36/36 scored；decision complete；gate=false |

36 个 candidate states 全部唯一，没有重复。事后 oracle 分类为：

- all：beneficial/harmful/indeterminate = `4/31/1`；
- primary：`3/23/1`，其中 beneficial 且 gate-pass 为 2，hard-gate false reject 为 1；
- diagnostic executor-only：`1/8/0`，唯一 beneficial 被原 point combined-change gate 拦截；
- Qwen 未调用；`eligible_for_qwen_not_evaluated` 只表示通过 GT-free hard gate，不代表模型接受。

两个预注册 target 的结论不同：

- `test_1_1`：唯一 primary
  `test_1_1--primary--r0--negative_point--t2` 把 FP 从 626 减为 566，但因为 GT 正像素为 0、仍有
  566 FP，IoU/F1 继续为 `0/0`，所以按冻结定义只能记为 indeterminate，primary beneficial 数为 0。
  diagnostic positive-point 候选虽把 mask 清空并得到 IoU/F1=`1/1`，但它不是 planner-reachable
  primary，而且被 `point_proposal_combined_change_gate_exceeded` 拦截，不能用于通过 readiness gate。
- `test_50_8`：18 个 primary 中有 2 个 beneficial。`s8--box--t1` 将 IoU/F1 提到
  `0.02682178/0.05224233`，但因 `component_count_delta_exceeded` 被原 gate 拦截；最佳
  `test_50_8--primary--s8--box--t2` 通过 gate，ΔIoU=`+0.09965337954939342`、
  ΔF1=`+0.18124507486209615`，TP `+115`、FP `+988`、FN `-115`。

预注册要求 `test_1_1` 与 `test_50_8` **都**至少有一个 primary beneficial + gate-pass 候选。因此最终
机器决定为：

```text
proceed_to_c_iso_and_t_traj_data_construction = false
next_action = stop_training_and_review_frozen_action_space
gate_modification_authorized = false
```

完整结果见
`doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.md`；formal attempt 位于
`outputs/CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001`。score report SHA256 为
`<hash omitted>`，decision SHA256 为
`<hash omitted>`。

解释边界必须保持严格：这只说明冻结 CORE3、预注册 36 个有限单步动作中，`test_1_1` 未观察到
primary beneficial；它不证明更大/多步动作空间不存在解，也不是 LEVIR-CD 泛化结论。此前“孤立
stage snapshots 可能不足以教授完整任务”的判断仍是待受控训练验证的假设，P2 没有训练，不能把它
升级成已证明根因。

当前收尾边界：registry 只追加 1 个 P2 科学结果与 Jobs `53272/53273` 两条 formal attempt，目标行数
`15/52/21/40`；**不追加 data row，不创建 C-ISO/T-TRAJ**。收尾诊断 Jobs `53315/53316` 分别暴露
registry 文本契约与测试夹具错误，preparation Job `53317` 通过后，旧 candidate
`f73f50bfc982d71705910f19d5571ef92870e931` 的 Job `53321` 又抓到结果 JSON 键名与正式 score report
不一致；这些失败 attempt 全部 append-only 保留。修正后的 preparation Job `53322` 通过 16 个 registry
和 12 个 P2 targeted tests；final C `d412759857d030b8ab766a077ea54b603585e72a` 的 exact-C Job
`53323` 通过独立证据重算、workbook、targeted、521 个 full tests（3 skipped）与 repository
immutability。其 receipt SHA256 为
`<hash omitted>`。Acceptance D 只能新增
机器 acceptance 文件并更新本 handoff/结果 Markdown；result JSON、registry、XLSX、validator 与执行面
必须保持字节不变。exact-D 外置终验通过后才把主开发分支快进到 D；即使收尾通过，也因科学
gate=false 停在训练数据构造门前，后续动作空间或 gate arm 必须另行设计和审批。
