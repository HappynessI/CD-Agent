# Qwen3-VL P1 retry3 execution amendment freeze（0802）

- 当前状态：**frozen / execution_ready / p1_execution_authorized=true**
- 日期：2026-08-02（UTC）
- 机器可读 retry3 freeze：`QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json`
- 首次 freeze B：`1dfbdbeda9f8b1ace5e3074a4dee9d52b8f86efe`
- retry1 freeze B2：`a59382656ad6cdd4ee62b76a655f84c2279f023c`
- 原预注册 canonical SHA256：
  `<hash omitted>`
- P0 acceptance receipt canonical SHA256：
  `<hash omitted>`
- P0 acceptance receipt raw SHA256：
  `<hash omitted>`
- 上一版 resolved config raw SHA256：
  `<hash omitted>`
- 上一版 CPU fixture manifest canonical SHA256：
  `<hash omitted>`
- 上一版 amendment raw SHA256：
  `<hash omitted>`
- 上一版 amendment canonical SHA256：
  `<hash omitted>`
- retry1 implementation A2：`15cdca7ddc4611ad773576dd582e5b4513188d8e`
- retry1 precommit：Job `53103`，`COMPLETED / 0:0`，298 targeted、468 full、3 skipped
- retry1 hash：Job `53106`，`COMPLETED / 0:0`，25 entries；manifest SHA256
  `<hash omitted>`
- retry1 resolved config raw SHA256：
  `<hash omitted>`
- retry1 CPU fixture manifest canonical SHA256：
  `<hash omitted>`
- retry1 amendment raw SHA256：
  `<hash omitted>`
- retry1 amendment canonical SHA256：
  `<hash omitted>`
- retry2 implementation A3：`e31fcb01d34cc28924e9c4189f319d7f44af4911`
- retry2 pre-amend freeze candidate：`053e3d3709ccd8ec6f2d4eae3a1c1778f3a01550`，已验证
  `candidate^ == A3`；本次 amend 在既有授权范围内将其替换为 final B3
- retry2 precommit：Job `53135`，`COMPLETED / 0:0`，299 targeted、469 full、3 skipped
- retry2 hash：Job `53140`，`COMPLETED / 0:0`，25 entries；manifest SHA256
  `<hash omitted>`
- retry2 resolved config raw SHA256：
  `<hash omitted>`
- retry2 CPU fixture manifest canonical SHA256：
  `<hash omitted>`
- retry2 amendment raw SHA256：
  `<hash omitted>`
- retry2 amendment canonical SHA256：
  `<hash omitted>`
- retry2 final freeze B3：`64871db9f03d19d6ddb9ab3edf502c3ecb698ee5`，且
  `B3^ == e31fcb01d34cc28924e9c4189f319d7f44af4911`
- retry3 implementation A4：`a88404ac6a014f42692d1119d54de42c980fb310`
- retry3 precommit：Job `53151`，`COMPLETED / 0:0`，300 targeted、470 full、3 skipped
- retry3 hash：Job `53152`，`COMPLETED / 0:0`，25 entries；manifest SHA256
  `<hash omitted>`
- retry3 `terminal_review_verifier` SHA256：
  `<hash omitted>`
- retry3 pre-amend non-authorizing candidate：`8a862e700ecc7d7d81ea512a65b5aca5430347ba`，已验证
  `candidate^ == A4`；本次 amend 在用户明确授权范围内将其替换为 final B4
- retry3 resolved config raw SHA256：
  `<hash omitted>`
- retry3 CPU fixture manifest canonical SHA256：
  `<hash omitted>`
- retry3 amendment raw SHA256：
  `<hash omitted>`
- retry3 amendment canonical SHA256：
  `<hash omitted>`

第 1–9 节保留此前 freeze 与失败链的合同和审计历史；当前恢复点以第 10 节、机器可读草稿及现场
Git/Slurm 证据为准。

## 1. 历史 freeze B 的授权、状态与不可回填边界

用户已在当前会话批准进入 P1，实现并按 CPU fixture、smoke、CORE 的阶段顺序验证
H0/H1 harness。该批准不绕过预注册的技术门槛，也不批准 P2、训练、post-SFT、EXTENDED5
或 locked final。

本节记录首次 freeze B 当时的最终 amendment 冻结事务，不修改或回填原
`QWEN3VL_H0_H1_PREREGISTRATION_0802.json`，也不修改已经 complete 的
`QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json`。原文件中的历史
`p1_execution_authorized=false` 保持不变；该次执行授权只由当时的 amendment、resolved config 和
CPU fixture manifest 三层共同表达。三者当时已由真实证据闭环并收敛为
`frozen / execution_ready / p1_execution_authorized=true`。用户又明确授权：
“确认授权 amend candidate 为最终 freeze B，并立即同步 detached worktree；全部检查通过前不提交正式作业”。

该次授权只允许完成 freeze B 事务与既定 P1 串行链，不绕过运行前门禁：必须先让最终 B 满足
`B^ == A`，把 detached run worktree 精确同步到 B 并保持 clean，再通过 submitter
`--check-only`。这些检查全部通过前，**不得提交任何正式 P1 Job**；通过后也只能先提交 CPU
fixture，首个 GPU smoke 仍只能由正式 CPU Job 的 `afterok` 依赖释放。

截至该次同步，implementation commit A 已形成：
`f97f9f090369bc5d97f2a60b3d247aae6166b16e`。零 GPU precommit Slurm Job `53037`
已通过独立 golden validator、11 个 targeted patterns、466-test full regression 与 shell syntax；
其 worker wall-clock 为 24 秒，`sacct` 口径为 26 秒。
零 GPU Job `53047` 已从 A 的 clean worktree 计算模型、checkpoint、21 个工具和 3 个 worker 的真实
SHA-256。non-authorizing candidate `5293b3a4fb1237ffdbecd6f718c954d0ff22e980` 已验证为 A 的
直接子提交，detached run worktree 在 amend 前已 clean 对齐该 candidate。包含本记录的 amend
提交即最终 freeze B；为避免 Git 自引用，B 的 hash 不写入冻结文件，而由 Git ancestry、run
worktree、submission receipt 和 evaluation receipts 外部绑定。正式 CPU fixture receipt 仍未产生。

## 2. 冻结 treatment 与共同不变量

唯一因果变量继续是 `terminal_review_scheduler_v1`：

- H0：`qwen3vl_harness_h0_legacy_single_round_v1`，arm
  `v12_1-h0-legacy-single-round-base-0802`，保持 legacy single-round terminal policy；
- H1：`qwen3vl_harness_h1_gtfree_terminal_review_v1`，arm
  `v12_1-h1-gtfree-terminal-review-base-0802`，只增加一次 GT-free independent terminal
  review，禁止第三轮；
- 两者共同使用 base model `Qwen3-VL-2B-Instruct`、model-I/O protocol `v12.1`、seed 42、
  `qwen3vl_e2e_summary_v2` 和 `qwen3vl_evaluation_receipt_v2`；
- 除 protocol/arm ID、`max_verifier_rounds`、terminal-review 及其 trigger/selection/decision
  字段外，模型、工具、输入顺序、generation、Environment 初态、每轮 stage-attempt 上限、
  environment-step 上限和全部安全门必须相同。

H0/H1 的预算仍为：每轮最多 2 次 stage generation（initial + 最多一次 repair），
environment steps 最多 3；H0 最多 1 个 verifier round，H1 最多 2 个。

H1 plan-null 复核固定使用 `environment_ordered_terminal_review_v1`：先选 Environment 顺序中的
下一条未审 region，并使用 `qwen3vl_plan_null_next_unreviewed_region_v1`；若不存在未审 region，才选
首个 unresolved region，并使用 `qwen3vl_plan_null_alternative_diagnosis_v1`。两种 prompt、region 及
raw backend-call identity 必须可独立回放，未知或错配 version 必须 fail closed。

## 3. 五个 P1 allocation

下表的输出目录和 QQ title 来自原预注册，保持精确不变。Job name、资源与 `afterok` 链已经
按实现冻结；用户已经明确授权执行正式 P1 严格串行链。授权只在包含本记录的 freeze B 形成、clean
detached worktree 精确同步到 B 且 submitter `--check-only` 成功后进入可提交状态。每个目录提交前
必须仍然不存在。

| allocation | 阶段/arm | 建议 Job name | GPU | 输出目录 | QQ title | dependency |
|---|---|---|---:|---|---|---|
| `(2)` | CPU H0/H1 fixtures | `CA0802P1CPUFixtures` | 0 | `CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures` | `CA0802 P1 H0-H1 CPU fixtures` | 无 |
| `(3)` | smoke H0 | `CA0802P1SmokeH0` | 1×A800 | `CA_0802(3)-qwen3vl-h0-base-smoke1` | `CA0802 P1 H0 base smoke1` | `afterok:<cpu_fixture_job_id>` |
| `(4)` | smoke H1 | `CA0802P1SmokeH1` | 1×A800 | `CA_0802(4)-qwen3vl-h1-base-smoke1` | `CA0802 P1 H1 base smoke1` | `afterok:<smoke_h0_job_id>` |
| `(5)` | CORE3 H0 | `CA0802P1CoreH0` | 1×A800 | `CA_0802(5)-qwen3vl-h0-base-core3` | `CA0802 P1 H0 base CORE3` | `afterok:<smoke_h1_job_id>` |
| `(6)` | CORE3 H1 | `CA0802P1CoreH1` | 1×A800 | `CA_0802(6)-qwen3vl-h1-base-core3` | `CA0802 P1 H1 base CORE3` | `afterok:<core_h0_job_id>` |

另设一个不属于上述五个正式 allocation 的辅助收尾 Job
`CA0802P1DependencyGuard`。它申请 0 GPU、1 CPU、1G memory、10 分钟，且不固定 GPU
节点；依赖为 `afterany:<全部五个正式 job id>`。五个正式 Job 之间的四条 `afterok` 边保持
不变，因此 guard 只观察终态和补通知证据，不会释放、重排或并发运行任何正式阶段。

每个 allocation 的 stdout/stderr 都固定在自身输出目录的
`logs/slurm-%j.out` / `logs/slurm-%j.err`。CPU fixture 不申请 GPU；每个 smoke/CORE Job
只能申请一张 NVIDIA A800，并验证 CUDA 可见卡数为 1 和 BF16。H0/H1 不并发，本任务任一
时刻总 GPU 数不超过 3。

固定 node/partition 为 `gpu48/gre`。CPU fixture 为 2 CPU、16G、45 分钟；smoke 各为
4 CPU、48G、1 GPU、1 小时；CORE3 各为 4 CPU、48G、1 GPU、3 小时。dependency guard 为
1 CPU、1G、10 分钟且不固定节点。

## 4. 固定执行顺序和停止门

唯一顺序为：

`CPU fixtures → smoke H0 → smoke H1 → CORE3 H0 → CORE3 H1`

下游一律使用 `afterok`，任何上游失败都不释放下游：

1. CPU fixtures 至少覆盖 legacy finish/plan-null parity、round-two safe plan/uncertain、
   round conflict、repair exhausted、budget exhausted、safety-gate rejection、rollout 前 label
   零访问、plan-null trigger/selection/prompt/raw-call replay、真实 H1 empty-mask review、显式 network
   retry 分账、raw trajectory final-region 重建以及 dependency guard。H0 parity
   不得仅用当前 legacy verifier 与 H0 decorator 互比；必须先通过只读
   `QWEN3VL_H0_V12_1_GOLDEN_FIXTURE_RECEIPT_0802.json` 及其独立 validator。该 receipt 固定
   P0 成功 commit/source blobs、P0 acceptance receipt raw/canonical hash、finish/plan-null 输入与
   行为向量，并由 CPU manifest 绑定 receipt raw/canonical hash 和 validator hash；协同修改 receipt
   与 manifest 也必须 fail closed。H0 parity 必须为 1；eligible trigger 必须恰好进入一次可确定性
   回放的独立第二轮；完整 transition
   table、GT boundary、安全门、预算分账、requested/scored、tool failure、final-region 重建和 schema/receipt
   validator 全部通过后才能释放 GPU。
2. smoke 固定 base model、seed 42、`test_1_1`，先 H0 后 H1。H1 必须让历史 finish case
   实际进入 schema-valid、fresh-context、非重复、可回放的第二轮；否则不进入 CORE3。
3. CORE3 固定顺序 `test_20_15`、`test_1_1`、`test_50_8`，H0 完整成功后才运行 H1。
4. 原预注册的 19 项 H1 freeze gate 必须 19/19 通过；任一失败即不冻结 H1、不训练。

IoU/F1 改善或是否出现新 candidate 不是 harness 安全冻结门，也不能用于反向调整在线 trigger、
region 或 action。若安全门通过但没有新诊断/candidate，只能如实进入 capability diagnosis。

## 5. 通知合同

每个 Job 在 success、failed、cancelled、timeout、out-of-memory 或 node-fail 时均发送 QQ
终态通知。body 必须包含 `job_name`、`job_id`、`terminal_status`、`elapsed_seconds`、
`key_result_or_error`、`artifact_path`；不得包含凭据或代理值。没有外部 delivery receipt 时，
只能声明 worker 尝试通知，不能声明收件人已收到。

若 `afterok` 前驱失败导致下游 worker 从未启动，辅助 dependency guard 只在以下证据同时成立
时补 `logs/status-<job_id>.txt`：目标 status 不存在、`sacct` 显示目标为 dependency 原因取消且
没有 start time、直接前驱不是 `COMPLETED`。补写使用同文件系统原子 no-replace 发布；已经
启动但缺 status 的 Job fail closed，不伪造终态。每次代发 QQ 都另存
`logs/qq-attempt-<job_id>.txt`，记录 notifier exit code 并固定
`delivery_claimed=false`。该辅助 Job 自身的 stdout/stderr、status、receipt 和 QQ attempt 均写在
`(2)` 的 `logs/` 或 `(2)` 根目录，不新增或复用正式输出目录。

## 6. retry1 冻结状态与 release 门（历史）

本节描述 retry1 freeze B2 形成前的历史状态，不再是当前授权状态。implementation、precommit Slurm 验证、外部 hash、worker/tool/model 绑定、精确资源、QQ title
验证和用户执行授权均已完成。non-authorizing candidate 及其 clean detached worktree 已核实；三层
状态现为 `frozen / execution_ready / p1_execution_authorized=true`，最终派生 digest 已写入本文件
和 session handoff。包含本记录的 amend 提交被定义为最终 freeze B。

仍需按顺序通过以下 release 门：

1. amend 后验证 `B^ == A` 且开发分支 HEAD 精确等于 B；
2. 立即把 detached run worktree 同步到 B，并验证 detached、clean、HEAD 精确相等；
3. 从该 run worktree 执行 submitter `--check-only`，重新检查 controller、重复作业、路径、hash、
   资源和五个输出目录 absence；
4. 只有 1–3 全过才可提交正式链；正式 CPU allocation 必须再次生成 formal success receipt，
   首个 GPU smoke 只能由其 `afterok` 释放。

任何冻结合同变化都需要新的显式授权与新的 A/B 闭环，禁止静默修改或复用旧 digest。

Git 冻结必须拆成两个提交：implementation commit **A** 先包含完整实现、tests 和维护文档，并写入
amendment 的 `implementation_commit`；随后才从真实文件计算 hash/resource binding，形成 A 的直接
子提交 freeze commit **B**，即 `B^ == A`。冻结文件不得写入自身 B 的 hash；正式 detached run
worktree 与 `runtime_commit` 必须精确等于 B 且保持 clean。submission receipt 与 evaluation receipt
必须同时记录 `implementation_commit=A` 和 `freeze_commit=B`，receipt v2 还必须把 P1 amendment
作为 hash-bound artifact。

## 7. 禁止事项

- 不修改原 preregistration、P0 receipt 或任何 `CA_0801_archive` 证据；
- 不在管理节点运行测试、数据处理、训练或评测；
- rollout-complete 前不得读取 GT、label、FP/FN、IoU/F1、人工失败标签或历史
  false-finish 身份；
- 不放宽 parser、geometry、locality、candidate-benefit-without-harm 或 rollback；
- 不继承首轮 hidden state、复制首轮 raw response 伪装独立复核、执行第三轮、增加 network/tool
  retry 或因为多一轮而增加 environment steps；
- 不为 candidate quota 强制动作；冲突、unresolved、unsafe/nonunique、schema-invalid 或预算耗尽
  必须 fail closed；
- 当前 P1 对比与 append-only 登记完成前不运行 post-SFT、EXTENDED5、locked final、训练或 P2；
- 不复用、覆盖或现场顺延 `(2)`–`(6)`，不把日志写入 `outputs/` 根；smoke 不保存 checkpoint；
- 不并发 H0/H1，不给单个 P1 smoke/CORE Job 分配超过一张 GPU；
- receipt 必需路径或 digest 缺失时不得标记 complete；CORE3 不得解释为 LEVIR-CD 泛化证明。

## 8. retry1 fail-closed 记录与重新冻结门

首次正式链由 freeze B `1dfbdbeda9f8b1ace5e3074a4dee9d52b8f86efe` 提交：

- CPU Job `53064`：`COMPLETED / 0:0`，formal receipt、12 个日志 hash、296 targeted tests 与
  466-test full regression 独立复核通过；
- smoke H0 Job `53065`：rollout、summary 和 summary validator 均成功，但在 evaluation receipt
  builder 处 `FAILED / 3:0`；
- H1 `53066`、CORE3 H0 `53067`、CORE3 H1 `53068`：因 `afterok` 未满足，均在 worker 启动前
  `CANCELLED`；
- guard `53069`：因本集群 SlurmDBD 把 dependency-cancelled Job 记为
  `CANCELLED / Start=None / Reason=None`，按旧合同 fail closed。

H0 failure 的根因是 submitter 用同优先级 `&&`/`||` 模拟三元表达式，导致 H0 的
`experiment_arm_id` 和 `harness_protocol_id` 各自拼接 H0+H1 两个值。receipt builder 正确拒绝了
该 submission receipt；既有输出禁止补写、修改或提升为 canonical success。本次属于工程失败，
科学结果为 `not_evaluated`，不能据 H0 diagnostic summary 得出 H0/H1 对比结论。

retry1 只修复编排与证据层，不改变模型、输入、seed、generation、H0/H1 因果变量、预算或安全门：

1. submitter 从冻结 config 的 `frozen_identity` 解析单一 arm/protocol identity；
2. guard 仅在 `sacct` reason 缺失时查询 live `scontrol`；只接受单行记录中唯一且精确解析出的
   `JobId`、`JobState=CANCELLED`、`Reason=DependencyNeverSatisfied` 与单一
   `Dependency=afterok:<直接前驱>(...)`，拒绝额外边、重复字段和错误前驱，并把 controller 三个证据字段
   写入 synthetic status 供事后独立复核；
3. 原 allocation IDs `(2)`–`(6)` 不重编号，retry1 使用五个全新且不存在的 `-retry1` 输出目录；
4. 原 Jobs `53064`–`53069`、旧输出、freeze B 和全部 hashes 保持只读，作为失败 attempt ledger。

旧 precommit Job `53037` 只覆盖原 70-ID suite，不验证新增的两个 retry1 fixtures；机器记录已把它
明确降为 previous evidence，并把当前 72-ID retry1 validation 重置为 `not_run`。

第一次 retry1 precommit Job `53102` 在 14 秒后以 `FAILED / 1:0` 停在新增 identity fixture：fixture
执行从 submitter 原样提取的 receipt writer 时漏设其所需的 `PARTITION` 与 `NODE` shell 变量。此前
golden validator 与其余 10 个 targeted patterns 均已通过；这是 test-fixture wiring failure，不是正式
链或科学结果。失败 attempt 保留，补齐两个变量后必须从新 implementation commit 重跑全部门禁。

修正后的 implementation A2 为 `15cdca7ddc4611ad773576dd582e5b4513188d8e`。零 GPU precommit
Job `53103` 已 `COMPLETED / 0:0`：72 个 required IDs 均有源码绑定，11 个 targeted patterns 共
298 tests，full regression 468 passed / 3 skipped，golden validator 与 shell syntax 全过。零 GPU hash
Job `53106` 也在 clean A2 上 `COMPLETED / 0:0`，重新绑定 1 个 model、21 个 tools/checkpoints 和
3 个 workers；25-entry manifest SHA256 为
`<hash omitted>`。

non-authorizing candidate `03f95e53f0070665a56e746ac454047242d6e369` 已固定为 A2 的直接子并
通过独立 activation 审计；本 amendment 随用户授权的 `git commit --amend` 将该 candidate 替换为
满足 `B2^ == A2` 的最终 freeze B2。B2 不在文件内自引用；amend 后必须立即同步 clean detached
worktree 并通过 `--check-only`，此前仍不得提交 retry1 正式 Job。通过后也只能从 retry1 CPU formal
allocation 开始，GPU 只能由其 `afterok` 释放。

## 9. retry1 第二次 fail-closed、retry2 修复与后续授权

retry1 final freeze B2 为 `a59382656ad6cdd4ee62b76a655f84c2279f023c`，implementation A2 为
`15cdca7ddc4611ad773576dd582e5b4513188d8e`。在 detached worktree 与 `--check-only` 全部通过后，
正式链 `53109 → 53110 → 53111 → 53112 → 53113` 及 guard `53114` 才被提交：

- `53109` formal CPU：`COMPLETED / 0:0`，298 targeted、468 full、3 skipped，receipt 有效；
- `53110` smoke H0：`COMPLETED / 0:0`，summary 与 evaluation receipt 独立验证有效；
- `53111` smoke H1：rollout 与 offline metrics 已完成，但 summary builder 以 `FAILED / 3:0`
  fail closed；`53112`、`53113` 因依赖不满足在 worker 启动前取消；
- `53114` guard：`COMPLETED / 0:0`，成功为 `53112`、`53113` 合成带 controller 证据的终态，
  `unresolved_missing_status_job_ids=none`。

因此 retry1 科学状态仍是 `not_evaluated`，不得把 H0 的有效 receipt 或 H1 的 raw rollout 提升为
H0/H1 对比结论。旧输出全部只读保留。

精确根因在证据重建端。生产 `StagedQwenVerifier` 的 initial `StageTrace.decision` 按设计保存
`comparison=initial, accept=false, stop=false`；随后 `_output()` 根据完整 `finish` plan、completion gate
与全 `none` diagnoses 派生公开的 `VerifierOutput accept=true, stop=true`。scheduler 正确按公开输出触发
`provisional_finish`，但 summary builder 与独立 validator 错误要求中间 decision 自身已经 true/true，
因而拒绝合法轨迹。

retry2 只修复两个彼此独立的 trigger rebuilder，并新增 production-shaped initial/candidate 正负例；
不修改模型、输入、seed、generation、H0/H1 因果变量、预算、parser、geometry、locality、
candidate-benefit-without-harm 或 rollback。原 allocation IDs 不变，五个输出改为全新 `-retry2` 目录。

首个 implementation candidate `47d67d45be5011eb91008fbd46c1692cef86e00c` 的零 GPU precommit Job
`53134` 为 `FAILED / 1:0 / 00:00:15`，停在 synthetic submitter identity fixture，精确错误是
`resolved config tool_files_sha256 contains an unresolved hash`。retry2 草稿有意把改变过的 summarizer
和 summary validator hash 置 null 等待 clean hash Job，但该私有 fixture 只替换了旧版预期变化的
worker/submitter hash。production submitter 正确 fail closed；没有正式 allocation 或科学结果受影响。
本次只让 fixture 确定性解析所有 draft-null tool hashes，并同步其临时输出名到 `-retry2`。失败 attempt
及日志只读保留，不能与后续结果拼接，新 implementation A3 必须完整重跑全部 precommit。

retry2 freeze 事务已完成前四步；包含本记录的提交定义为 final B3，且不得在冻结文件中自引用其
commit hash。正式 Job 提交前仍必须完成第 5 步：

1. [x] 创建包含实现、73 个 required IDs 与维护文档的 implementation A3；
2. [x] 从 clean A3 在 Slurm 运行零 GPU、11-pattern、full-regression precommit；
3. [x] 从 clean A3 重算 25 个 model/tool/worker hashes；
4. [x] 创建直接子 freeze B3，使三层状态收敛到 `frozen/execution_ready/true`；
5. [x] 同步 clean detached worktree 到 B3，并通过 submitter `--check-only`。

Job `53135` 在 clean A3 上完整通过，Job `53140` 也绑定同一 A3；两条 changed tool hashes 分别为
summarizer `<hash omitted>` 与独立 validator
`<hash omitted>`。其余 23 项与 retry1
hash manifest 一致。B3 满足 `B3^ == A3`；detached sync 和 `--check-only` 均已在正式链提交前通过。

用户另行授权：仅在当前 P1 对比、独立验证、registry/workbook/报告与 Git 收尾完成后，才可基于届时
最新框架启动版本化的“训练数据创建 → 训练 → 评测 → 失败轨迹分析 → 下一轮改进”。该授权重点保留
完整 `audit → action → candidate → next-state` 轨迹假设，并要求每轮保存数据 manifest/hash、Slurm
作业、产物、结论和 Git lineage；它不是泛化的 P2 授权，也不绕过新阶段自身的预注册、冻结与安全门。

## 10. retry2 CORE3 H1 fail-closed 与 retry3 草案

retry2 final freeze B3 为 `64871db9f03d19d6ddb9ab3edf502c3ecb698ee5`，implementation A3 为
`e31fcb01d34cc28924e9c4189f319d7f44af4911`。detached worktree、clean/exact 检查和 submitter
`--check-only` 均通过后，才提交正式链 `53141 → 53142 → 53143 → 53144 → 53145` 及 guard
`53146`：

- `53141` formal CPU：`COMPLETED / 0:0`，299 targeted、469 full、3 skipped，receipt 有效；
- `53142` smoke H0 与 `53143` smoke H1：均 `COMPLETED / 0:0`，两臂 summary、evaluation
  receipt 及 smoke pair receipt 独立验证有效；
- `53144` CORE3 H0：`COMPLETED / 0:0`，standalone summary/receipt 有效；micro IoU/F1 为
  `0.7422312 / 0.85204673`，macro IoU/F1 为 `0.28212851 / 0.30560087`；
- `53145` CORE3 H1：`FAILED / 1:0`，在 `test_20_15` reset entry 的 `Trajectory.save()` 中触发
  `RecursionError: maximum recursion depth exceeded`；没有 `rollout_complete.json`、summary、
  evaluation receipt 或 CORE pair receipt；
- `53146` guard：`COMPLETED / 0:0`，所有已启动 worker status 均存在，未合成 status，
  `unresolved_missing_status_job_ids=none`。

因此 P1 整体科学状态仍是 `not_evaluated`。smoke pair 是有效的协议/安全诊断，CORE3 H0 是独立
reference，但两者都不能替代完整 CORE3 H0/H1 pair，更不能与新 H1 局部重跑拼接成结论。retry2
目录、receipts 和失败副产物全部只读保留。

精确根因是 H1 非终止 passthrough 分支让 `final_evidence` 与 `round_one_evidence` 指向同一 dict，随后
又把包含 `round_one_evidence` 的 `terminal_review` 挂回 `final_evidence`，形成：

```text
execution.verifier_evidence
└─ terminal_review
   └─ round_one_evidence
      └─ terminal_review
         └─ ...
```

retry3 的最小功能修复仅把外层 evidence 改为 `copy.deepcopy(round_one_evidence)`，并新增真实
H1 nonterminal `reset → Trajectory.save → json.loads` 回归。它不改变模型、输入、seed、generation、
H0/H1 因果变量、预算或任何 safety gate。通用 serializer cycle detector 作为后续防御性增强登记；
本次不写 cycle marker，也不把无效证据伪装成成功 trajectory。

五个正式输出改用全新 `-retry3` 目录。工程门禁与 freeze 激活已完成；正式 Job 提交前仍必须完成
detached sync 与 `--check-only`：

1. [x] 创建包含最小修复、74 个 required IDs 与维护文档的 implementation A4；
2. [x] 从 clean A4 在 Slurm 完整运行零 GPU 11-pattern 与 full regression；Job `53151` 实测
   300 targeted、470 full、3 skipped；
3. [x] 从同一 clean A4 重算 25-entry model/tool/worker hash manifest；Job `53152` 成功；
4. [x] 回填真实 Job/count/hash；包含本记录的提交定义为 A4 直接子 non-authorizing candidate；
5. [x] 按用户已授权的 amend workflow 计算最终派生 digests；包含本记录的 amend 提交定义为
   final freeze B4；
6. [ ] 立即同步 clean detached worktree 到 B4，并通过 submitter `--check-only`；
7. [ ] 只有上述门禁全过，才从 formal CPU 开始完整重跑 retry3 五段链，不能复用 `53141–53144`。

P1 完整 receipts、pair、分析、registry/workbook 和结果 commit 收尾后，进入用户补充授权的独立阶段：
按届时最新框架重新走“训练数据创建 → 训练 → 评测 → 失败轨迹分析 → 下一轮迭代”，重点以同 episode
机械关联的完整 `audit → action → candidate → next-state` 监督对照孤立 stage snapshots；这仍是待验证
假设而非唯一根因，并保留单阶段 control。每轮仍须
独立预注册、data manifest/hash、来源与泄漏审计、Slurm、checkpoint lineage、评测 receipt、失败轨迹
分类、registry/workbook 和 Git 提交。
