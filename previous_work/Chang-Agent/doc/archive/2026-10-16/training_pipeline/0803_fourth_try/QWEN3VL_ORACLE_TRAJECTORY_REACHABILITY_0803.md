# Qwen3-VL oracle trajectory 可达性 pilot 预注册（2026-08-03）

## 当前状态

本阶段状态为 **exact-B3 formal complete / scientific no-go / training blocked**。implementation A3 与
freeze B3 已固定为
`9b4754dedb163119dc6ab4f2ce03f84b67b8b70f → be3bd9ea74a4991c9af6e0988d8fa67474bdb158`；B3 只修改
机器配置和本 0803 文档，并已由外部 append-only manifest、零 GPU contract receipt 和正式 attempt-0002
共同绑定。Job `54394` 的 exact-B3 contract 全部通过，Jobs `54396/54399` 的真实工具三 shard 与独立
validator 也都以 `COMPLETED/0:0` 结束。workflow 验证成功，但正式科学结论是 `scientific_no_go`：
4 个需要修改的 episode 中 endpoint success 为 0，`training_authorized=false`，C-ISO/T-TRAJ 均为 0 条，
因此不构建训练集、不启动 SFT。

A2/B2 已固定为
`13db539cb382ad37fdfeb66bf21f144c07a996e8 → f839cea8f638dc37976ef1c27545a71187d794ad`；其 exact-B2
contract 最终通过，三个 GPU shard 也全部成功，但独立 validator 在读取首条真实 episode 时发现两处此前只用
合成 fixture 未覆盖的序列化差异。因此 B2 formal attempt-0001 只保留为失败诊断，不能产生科学结论或训练授权。
A3 已让 validator 重建与 runner 的 pixel-coordinate action 精确同构，并把浮点等价的 metric delta 用容差
验值、再用独立复算值排序；Jobs `54375/54393` 已分别完成旧六轨迹诊断重放和 A3 候选全量回归，B3 随后
重新生成三 shard，未跨 freeze 复用 B2 receipt/runtime/hash。首轮 implementation/freeze
`e65f6099e3a792718a73a0e37f4c54e9cb79e80a → ea6a51d4b273170f4e56e72865fbbe2e725328a9`
在 exact-B submitter 的 Slurm 提交前检查中暴露 shell 成功返回码缺陷；两次管理端调用均在创建 diagnostics 和
Slurm Job 前以 code 1 退出，没有运行 contract 或 GPU pilot。该 B 不具运行资格，只保留为诊断节点。
新 A2 已为 `verify_freeze_delta` 的合法成功路径增加显式 `return 0` 和回归合同，并由零 GPU Slurm Job
`54305` 完成修复后回归。下述旧轨迹重放只用来说明 validator 修复过程；正式结论只取 exact-B3
attempt-0002。机器可读合同位于
`configs/qwen3vl_oracle_trajectory_pilot_0803.json`。

实现冻结前的真实输入 preflight 曾于 2026-08-03 用零 GPU Slurm Job `53958` 验证通过（`COMPLETED/0:0`，
10:30，2 CPU/16G）：6 个样本、18 个原 parent、18 个历史 curation artifact、6528 个 forbidden-test
文件、6 个 runtime source 与 2 个外部源码仓库在当时版本上通过身份/隔离检查。该 Job 发生在本轮 endpoint、
hard-gate 独立复算和派生合同收紧之前，只能作为历史输入基线，不能验收当前未提交代码；它更不能替代后续
clean detached exact-B contract/full regression，也不构成 pilot 科学结果。

实现期零 GPU 诊断 ledger（均为候选代码检查，不冒充正式验收）：

| Job | 角色 | 终态 | 结果 |
|---:|---|---|---|
| `53914` | validator evidence chain 迭代检查 | `COMPLETED/0:0`，2 CPU/4G，00:00:02 | 22/22 tests |
| `53916` | session rebase evidence 增补检查 | `COMPLETED/0:0`，2 CPU/4G，00:00:02 | 23/23 tests |
| `53953` | preflight contract 定向检查 | `COMPLETED/0:0`，2 CPU/8G，00:00:03 | 29/29 tests |
| `53958` | 定向测试 + 真实全输入 preflight | `COMPLETED/0:0`，2 CPU/16G，00:10:30 | 29/29 tests；真实 preflight passed |
| `54115` | 首次 pre-freeze 定向 + 全量回归 | `FAILED/1:0`，0GPU/2CPU/16G，00:00:50 | 新 runner 21/21、validator 38/38；全量 564 中 1 failure + 7 errors，均归因为 0802 证据归档或旧 detached worktree 清理后的历史路径漂移；冻结 evidence hash 复核无丢失 |
| `54217` | 历史路径受限解析首轮定向 | `FAILED/1:0`，0GPU/2CPU/16G，00:00:11 | registry 18/18、P0 19/19、P2 prereg 16/16；仅剩 parent closeout validator 仍把旧 receipt hash 与当前维护文件比较 |
| `54227` | 历史路径/exact-commit blob 修复复验 | `COMPLETED/0:0`，0GPU/2CPU/16G，00:00:11 | registry 18/18、P0 19/19、P2 prereg 16/16、P2 result 14/14；workbook `--check` passed |
| `54248` | 最终 implementation-A 候选定向 + 全量回归 | `COMPLETED/0:0`，0GPU/2CPU/16G，00:01:19 | runner 21/21、validator 44/44；full regression 591/591 passed，3 skipped |
| `54274` | 首轮 freeze-B submitter 修复后的候选回归 | `FAILED/1:0`，0GPU/2CPU/16G，00:01:01 | runner 21/21、validator 44/44；full regression 591 中仅 1 failure、3 skipped：旧 P2 acceptance 测试错误要求任意后续 `HEAD^ == d412759...`，没有暴露 oracle runner/validator 失败 |
| `54305` | descendant-aware 历史验收修复后的 A2 候选回归 | `COMPLETED/0:0`，0GPU/2CPU/16G，00:01:20 | P2 result 16/16、runner 21/21、validator 44/44；full regression 593/593 passed，3 skipped |
| `54328` | clean detached exact-B2 contract attempt-0001 | `FAILED/1:0`，0GPU/2CPU/16G，00:01:42 | preflight、runner 21/21、validator 44/44 passed；新 worktree 尚未同步 ignored `outputs` 证据链接，历史回归 575 中 4 failures + 4 errors；未创建 formal attempt |
| `54340` | exact-B2 contract attempt-0002 | `COMPLETED/0:0`，0GPU/2CPU/16G，00:01:20 | 同步与 B4/P2 worktree 相同的 `outputs → primary outputs` 链接后，preflight、runner 21/21、validator 44/44、full regression 593/593 全通过，3 skipped |

exact-B2 formal execution ledger（全部 append-only，失败不覆盖）：

| Job | 角色 | 终态 | 结果 |
|---:|---|---|---|
| `54351` | `0-2%3` 三 shard 真实工具生成 | 三个 task 均 `COMPLETED/0:0` | shard 0/1/2 分别 2 episodes；总计 4 accepted steps、26 candidates、Qwen calls=0；所有 completion/terminal receipts 存在 |
| `54352` | `afterany` 独立 validator | `FAILED/1:0`，0GPU/2CPU/16G，00:00:03 | fail closed 于 runner action 含 `coordinate_space=pixel`、独立 inventory 漏该字段；未写 validation report/matched data，不能给科学结论 |
| `54369` | 第一轮修复后旧轨迹零 GPU重放 | `FAILED/1:0`，0GPU/2CPU/16G，00:00:18 | runner 21/21、validator 44/44 passed；继续暴露 runner/validator 数学等价但浮点表示不逐 bit 相同的 `metric_delta` exact-equality 缺口 |
| `54375` | 第二轮修复后旧轨迹重放 + 全回归 | `COMPLETED/0:0`，0GPU/2CPU/16G，00:01:22 | runner 21/21、validator 46/46、full regression 595/595 passed（3 skipped）；旧六轨迹完整独立复算并安全得到诊断 `scientific_no_go` |
| `54393` | A3 最终候选定向 + 全量回归 | `COMPLETED/0:0`，0GPU/2CPU/16G，00:01:27 | P2 result 16/16、runner 21/21、validator 46/46；full regression 595/595 passed，3 skipped |

Job `54375` 的诊断结果不能跨 freeze 提升为正式结果，但足以指导下一轮：6 个样本中 1 个 `already_endpoint`、
1 个 `already_exact`、3 个 `no_beneficial`、仅 `train_364_r0c2` 为连续 3 步改善后 `max_steps`；endpoint
success 为 0，accepted transitions 为 4，候选工具为 SAM3/SimpleClick 各 13 次，selected replay 全为 4 次
SAM3 box。20/26 candidates 通过 hard gate；3 个会改善指标的候选被 hard gate 拦截。诊断派生的 train/val
C-ISO 与 T-TRAJ 都是 0 条，`training_authorized=false`、`raw_artifacts_sft_ready=false`。因此当前不启动 SFT；
而且增加 `max_steps` 最多直接帮助一个样本，不能修复另外三个 `no_beneficial` 或两个初始已达门样本，不能把
“有一个三步耗尽案例”误写为整轮增步已获放行。

Jobs `54115/54217` 保留为失败诊断，不冒充验收成功。对应修复不恢复旧软链、不修改冻结 P0/P2
result/acceptance/receipt JSON：0802 当前证据显式登记到 `CA_0802_archive`；冻结旧绝对路径只允许受限映射并
继续逐文件验 SHA；被清理的 implementation-A worktree 文件只从固定历史 commit 的同路径 Git blob 验证。
Job `54248` 只证明未提交候选通过回归，仍不能替代提交后 clean detached exact-B contract。

P2 result 的 commit-context 验证按历史验收节点而非当前开发提交解释：固定 acceptance D 为
`a094cc59d9104471b70b0624b4e10cbacfefe2e5`，并固定验证其唯一父提交为 candidate C
`d412759857d030b8ab766a077ea54b603585e72a`；当前 `HEAD` 只允许是 D 本身或 D 的后代，不再错误要求
`HEAD^ == C`。P2 冻结执行面在历史 D 上复核，acceptance/result JSON 则继续绑定 D 的精确 Git blob、字节数与
SHA256；candidate closeout receipt 及其 exact-C worker/validator/registry/workbook blob 校验均不放宽。该段只
记录代码语义；候选回归已由 Job `54305` 通过，提交后状态仍须 exact-B2 contract 复核。

本 pilot 是 P2 单步 no-go 后的独立动作空间检查，不修改、重解释或覆盖任何冻结 P2 文件。P2 仍然
保持 `proceed_to_c_iso_and_t_traj_data_construction=false`；只有本 pilot 自己完成实现、冻结、验证和
正式执行，并满足下述终点门，才允许从 endpoint-success episode 派生候选 C-ISO/T-TRAJ 数据。

## 要回答的问题

P2 只检查了冻结 CORE3 的有限单步动作。这里改为最多三步、每步一个 accepted action，问题是：

> 在真实 SAM3 初始化、真实 SimpleClick/SAM3 工具、production action geometry 和原 hard gates 下，
> 是否能形成一条可重放、每一步都严格改善、并在最多 3 步内到达预注册 change-mask 质量终点的轨迹？

该问题只测试 production-tool reachability，不测试 Qwen 是否能发现或选择这条轨迹。Qwen 不加载、
调用次数必须为 0。

这里恢复的是 LEVIR label 定义的 **change mask**。LEVIR 没有提供两侧 T1/T2 building semantic mask
GT，因此三 mask hash 一致只能证明 branch/canonical replay 和状态传递一致，不能声称 T1/T2 语义 mask
分别恢复了 GT。

## 样本与隔离

pilot 不再使用反复查看的 test/CORE3，而是冻结三个 `train_original` parent 和三个
`val_original` parent。每个 parent 只取一个 256×256 crop，坐标均为 0-based、右下角 exclusive。

| pilot sample | split/role | parent crop | GT | 0801 provisional RGB 语义 |
|---|---|---|---|---|
| `train_407_r3c2` | train，可派生训练数据 | `[512,768,768,1024]` | non-empty | T1 cultivated ground；T2 newly built roofed house |
| `train_114_r3c1` | train，可派生训练数据 | `[256,768,512,1024]` | non-empty | T1 trees/bare ground；T2 roofed buildings |
| `train_364_r0c2` | train，可派生训练数据 | `[512,0,768,256]` | empty | 同一组 roofed residential buildings 在 T1/T2 均存在 |
| `val_43_r1c1` | held-out validation only | `[256,256,512,512]` | non-empty | T1 grass/trees；T2 roofed house |
| `val_41_r3c2` | held-out validation only | `[512,768,768,1024]` | non-empty | T1 light-roofed building；T2 grass |
| `val_60_r3c0` | held-out validation only | `[0,768,256,1024]` | empty | T1/T2 均为 open fields，无 roofed building |

权威输入是 LEVIR-CD 原始 1024×1024 parent 的 A/B/label PNG，而不是旧数据集导出的训练图片。
JSON 已绑定 18 个 parent 文件 SHA256，并保留 0801 curation crop SHA 作为来源复核。正式运行前必须从
原 parent 重建 crop 并复核身份。

三个 train parent 与三个 validation parent 无交集；派生记录继承 parent split。validation episode
不能进入训练，任何 test/test_256 路径或内容 hash 出现都必须 fail closed。

## 非盲与 provisional 限制

六个 crop 都在 0801 数据 curation 中被人工看过，本次选择也有意覆盖
background→building、building→background、unchanged-building 和 unchanged-background。因此它是
**non-blinded feasibility pilot**，不能当作 LEVIR-CD 泛化估计。

0801 的 RGB evidence 只冻结宽粒度、可见的时相语义，不是逐像素标注，也不能代替 GT scoring。
如果 runtime crop 与来源不一致、RGB 语义不能维持，或动作与该语义冲突，该 episode 不能通过。
当前没有按 runtime proposal 区域重新人工标注 RGB；所以该 gate 是 provisional 的动作方向检查，不能包装成
像素级语义证明。这一限制与非盲选择一起保留在最终报告中。

## 生成与 offline selection 边界

每个样本从 fresh production SAM3 初始化开始。每一层按以下顺序执行：

1. 从当前 accepted state 读取 Environment 的 full live proposals，按 production 映射完成整个 action
   catalog；此时不读取 label。
2. offline oracle 打开 label，先丢弃与冻结 RGB 语义冲突的动作，再按 proposal box 内当前 change-error
   pixels 从多到少、action identity 字典序，冻结最多 4 个 action shortlist。
3. 在同一 parent state 上完整执行 shortlist 的真实工具候选；工具、Environment 和 hard gate 看不到
   label。候选 records/states 会在内存中收齐后再做最终排序，但真实 tool runs 与 candidate mask artifacts
   会随着候选执行逐个落盘，其中 mask artifacts 由 runner 原子写入。当前实现**没有**在最终 GT
   scoring/selection 前另写一个独立的
   “candidate batch complete” marker：该完成事实只在 episode 结束后写入 `episode.json` 的
   `candidate_batch_complete_before_final_gt_scoring` 审计字段。不能把它描述成已经落盘的 pre-scoring
   completion marker。
4. offline selector 只从 hard-gate pass 且满足严格 step 改善门的候选中选一个；排序为最低
   pixel error、最高 IoU、最高 F1、最后按 action identity 字典序。
5. 同一个 offline-construction worker 可以为 oracle selection 读取 label，但只把选中 action 和绑定
   receipt 交给 canonical Environment；label、metrics 和任何 GT-derived hint 不得进入 proposal、
   SimpleClick、SAM3、hard gate 或 Qwen 输入。
6. 在同一个 `max_steps=3` canonical Environment 中，从当前 accepted state 再执行一次选中动作；其
   T1/T2/change hashes 必须与候选分支完全一致，随后只连续传递该 canonical state。

这里必须区分三类落盘证据：第一步完整 GT-free catalog 会在首次 GT access 前以
`catalog_freeze_before_gt.json` 原子落盘；后续步的 catalog 在该步 shortlist/执行前以
`catalog_freeze_step-*.json` 原子落盘，但此时 label 已由同一 offline worker 打开，不能再声称它们早于首次
GT access。`episode.json` 只在该 episode 完整结束后原子落盘；shard 的 `generation_complete.json` 则要等两个 episode 和
`shard_index.json` 都写完后才原子落盘。前两类都不能冒充 shard completion marker；若中途失败，允许留下
未被 completion marker 授权的诊断 artifact。

GT 因此只用于 offline candidate shortlisting/selection/scoring。这里采用同一进程内的函数级隔离以减少
重复加载工具，并不声称 generator 进程本身无法打开 label；正式证据必须记录首次 GT access 在完整
action catalog marker 之后，并证明 label/metrics 没有作为实参或环境输入传给 SAM3 initializer、
SimpleClick、SAM3 box、hard gate、proposal builder 或 Qwen。

pre-execution shortlist 的排序输入是冻结 RGB 语义、当前 change-error 在 proposal box 内的像素数和 action
identity；post-execution selector 的排序输入是候选的独立评分、原 hard/RGB/per-step gate 与 action identity。
candidate-state hashes 用来绑定候选 artifact、metrics 和 receipt，而不是额外的质量排序信号。尤其是
canonical replay 只在 action 已选中后发生，所以 replay mask hash 只用于 branch/canonical 一致性验收，
绝不是 selector input。

`ProductionRuntimeFactory` 只能收到显式 allowlist 后的工具运行配置：解释器/权重路径、seed、设备、
matching、proposal 和 hard-gate 数值。整份 manifest、samples、label identity、`gt_pixels`、offline metric
合同及数据派生策略均不能进入 factory；输出中的 `label_passed_to_runtime_factory=false` 以此结构隔离为依据。

## 冻结 useful executor-reachable 动作策略

| live proposal | 枚举动作 |
|---|---|
| 普通 full-live proposal | 两侧 seed 都枚举 `positive_point`（既允许从黑 seed 添加，也允许在白 seed 上 refinement）；仅白 seed 侧枚举 `negative_point`；两侧都枚举 proposal-owned `box` |
| `scan` | 两侧只枚举 scan seed 的 `positive_point` 以及 `box_pixels` 的 `box`；即使异常/未来 proposal 漂移出白 seed 标志，也不枚举 `negative_point` |
| `delta_added` / `delta_removed` | 只作为 candidate 审核证据，不进入下一轮动作来源 |

上述动作全部使用 Environment 给出的 seed/box 并能通过现有 Environment/Executor 接口执行；RGB gate
在 top-4 前再过滤物理方向。它是为本次工具可达性冻结的 useful executor policy，比当前
`StagedQwenVerifier` 的 deterministic planner 更宽，但不声称穷举任意人工坐标或所有数学上可调用的工具
组合。结果必须按 `reachability_tier` 报告；若成功依赖 non-planner tier，进入模型训练/上线前还需明确让
planner 暴露相应合法动作，不能把“工具能执行”偷换为“当前 Qwen planner 已能选择”。

候选审核用的 `d* / delta_added / delta_removed` proposal 不能变成下一步动作来源；不允许人工坐标或
finish。accepted candidate 后必须使用 Environment 自动刷新出的 full live proposal catalog。

## 步数、候选和 replay 预算

- 每个 episode 最多 3 步；每步最多 1 个 accepted action。
- 每步最多执行 4 个 candidate actions，故 distinct candidates 理论上最多 12 个/样本。
- 包含 selected-action replay 在内，真实 tool execution 最多 24 次/样本。
- 计划 3 个 shard，每个 shard 1×A800、2 个样本；峰值不超过 3 GPU。

以上 `4 × 3 / distinct 12 / tool 24` 是 exact freeze contract，不接受缺失字段、较小值或较大值的
运行时覆盖；proposal `audit_kind` 只接受 `present/missing/mixed/scan/delta_added/delta_removed`，未知值
必须 fail closed，不能自动套用普通 proposal 动作策略。

catalog 本身始终完整记录；本 pilot 的冻结 oracle policy 只实跑上述 top-4 shortlist，未入 shortlist 的
catalog action 是已声明的 policy limitation，不算 runtime 截断。只有 shortlist 内动作、candidate 或
selected-action replay 因 24 次总 cap 无法执行时，才记为 `inconclusive_budget_exhausted`。任何受截断的
partial result 都不能提升为 no-go、success 或数据构造许可。

## 原 hard gates

本 pilot 逐项继承 P2 的 production hard-gate 数值：

- `max_selection_area_delta=0.25`
- `max_locality_outside_ratio=0.1`
- `max_target_mask_change_ratio=0.25`
- `max_component_count_delta=4`
- `max_point_proposal_change_ratio=0.25`
- `max_point_proposal_change_pixels=256`
- `max_point_image_change_ratio=0.004`
- `point_proposal_change_gate_enabled=true`

本阶段不授权修改、放松或事后调参。beneficial 但被 gate 拦截的候选只能报告为 false-reject diagnostic，
不能用于构造数据。

## 冻结局部改善与终点定义

定义 `pixel_error = FP + FN`，所有比较在舍入前完成，epsilon 为 `1e-12`。

每个 accepted step 必须同时满足：

1. 每一步都通过原 hard gate、provisional RGB semantic check 和 selected-action exact replay；
2. 每一步 `pixel_error` 严格下降；
3. non-empty GT：每一步 IoU 与 F1 都严格上升；
4. empty GT：每一步 FP 严格下降；
5. 没有任何候选或 replay 预算截断。

原先“最终 error reduction 至少 5%”不足以证明任务解决，现只命名为 `locally_reachable`。它是诊断量，
即使通过也不能产正式 C-ISO/T-TRAJ。

只有达到以下绝对终点才记为 `endpoint_success`：

- non-empty GT：最终 `IoU >= 0.90`，并复核 `F1 >= 18/19 ≈ 0.947368`；对同一个 binary mask，
  二者由 `F1=2*IoU/(1+IoU)` 联系，F1 是一致性复核而非第二份独立证据；
- empty GT：最终 `FP <= 64`，即 256×256 crop 的残留不足 0.1%；empty GT 在非 exact 时 IoU/F1
  恒为 0，故不能用它们定义“接近”；
- 初始未达终点、至少经过 1 个 accepted action，首次达到终点立即停止，且没有 budget hit。

`exact_recovery` 仍按最终 `FP+FN==0` 单独报告，它是 endpoint success 的严格子集。最终达到 5% 改善但
没到终点记为 `locally_reachable_only`，不得误称 success。

初始 `pixel_error == 0` 单列为 `already_exact`；初始非 exact 但已经达到终点单列为 `already_endpoint`。
两者都不证明工具轨迹恢复、不产生 C-ISO/T-TRAJ，也不进入 trajectory recovery denominator。完整搜索
无合格轨迹记为 `not_reachable_within_frozen_pilot`；基础设施、证据或访问边界失败记为 `not_evaluated`；
预算截断记为 `inconclusive_budget_exhausted`。

六个样本是人工挑选的非盲 feasibility 集，因此全局放行门不是“过一部分”：必须 6/6 都为
`endpoint_success`、全部有非零 accepted trajectory，且没有 inconclusive/not-evaluated，才允许进入下一轮
正式训练数据迭代。任何较低比例都只形成失败分析和候选轨迹，不放行训练；独立 validator 此时应报告
`scientific_no_go`。只要合同、证据和独立复算本身都合法，`scientific_no_go` 是一次成功完成的科学结论，
validation Slurm 应正常退出，而不是把“工具在冻结边界内未达到 6/6”伪装成基础设施失败。

## 同源 C-ISO / T-TRAJ 派生

只有 endpoint-success episode 可以派生候选数据：

- T-TRAJ 保存同一个 episode 的完整
  `audit → action → real-tool candidate → hard-gate → accepted next-state` 链；
- C-ISO 对该 T-TRAJ 的每个 accepted transition 各派生一个 isolated stage record；
- 两者必须绑定同一 parent、动作、candidate 和 state hashes，禁止跨 episode 拼接；
- train/validation split 原样继承；`already_exact`、`already_endpoint`、`locally_reachable_only`、失败和
  inconclusive episode 均不产数据。

这里有两层不同的门，不能混写。episode-level endpoint 门决定哪些 episode 可以进入派生文件；global 6/6
门决定整轮是否获准进入下一次正式训练数据迭代。因此在 `scientific_no_go` 下，validator 可以把少数
endpoint-success episode 写成隔离的候选/诊断 C-ISO 与 T-TRAJ artifact，但这些文件没有训练放行资格，
不得提升为正式训练集或据此启动训练。

输出必须把 train 与 val 写到物理分离目录，并携带 RGB parent/crop、source/next masks、proposal/audit、
tool receipt、hard/RGB gate、metrics 和 lineage 引用，才能恢复/replay。同源产物是 canonical trajectory
source corpus；它还不是 Qwen ShareGPT conversation，后续进入 SFT 前必须再做确定性的模型可见格式转换。
冻结输出路径为 `data/train/{c_iso,t_traj}.jsonl` 与 `data/val/{c_iso,t_traj}.jsonl`；每个 isolated
transition 显式携带 source/candidate/next/replay state artifact receipt，不能只留下 state hash。

后续若进入训练，C-ISO 与 T-TRAJ 还必须匹配 base checkpoint、来源 parent、token exposure 或 optimizer
steps 和评测协议，才能检验“孤立阶段 vs 完整轨迹”假设。

### 已记录的完整流程训练数据缺口

此前已经明确的核心缺口是：现有训练数据缺少足够的**完整任务轨迹**；单阶段样本只监督“给定某个中间
状态时做一次局部动作”，不足以直接教会模型怎样跨步骤读取新状态、连续决策、处理前一步结果并把整个任务
推进到终点。C-ISO 仍有价值，但不能单独充当完整流程监督的替代品。

因此本轮要求从同一个 endpoint-success episode 同源、机械地同时派生 C-ISO 与 T-TRAJ：前者保留逐阶段
学习信号，后者保留从初始审计到终点的有序 accepted transitions 与状态传递。后续匹配训练与评测才检验
“加入完整轨迹是否补上该缺口”；这目前是待验证假设，不是预写的训练收益结论。若 6/6 global gate 未通过，
该缺口仍保持未解决，不能用局部改善、失败轨迹或拼接的单阶段记录强行填补。

独立 validator 不信任 generator 自报的 `hard_gate`：它从已验证的 source/candidate masks 与冻结
action/proposal geometry 重新计算 change-area、point/box ROI、outside/target change ratio、四连通 component
delta 和 point combined gate，并对 branch 与 canonical replay 各复核一次；receipt 与 runtime execution
一起协调伪造也必须失败。

### 放行前验收器补强（2026-08-03，待 Slurm 回归）

本轮又关闭了两个可能造成误放行的缺口。第一，validator 不再按 episode status 接受一组宽松的
`termination.reason`：它从初始/最终 mask、endpoint、完整 selection rounds、候选 batch、accepted replay 和
冻结工具预算唯一推导 `already_exact`、`already_endpoint`、`budget_hit`、`exact_recovery`、
`endpoint_reached`、`no_beneficial` 或 `max_steps`。因此“一步后停止却自报 max_steps”等不可能轨迹必须失败。
报告同时独立汇总终止原因、action-catalog/candidate RGB、candidate metric gate、beneficial 与 hard-gate
reject reason，并单列“指标本可改善但被 hard gate 拒绝”的潜在 false reject；candidate 与 selected replay
也分别统计动作类型以及 SimpleClick/SAM3 使用量。

第二，训练授权改为机器可判定并在落盘前 fail closed：只有全局冻结的 6/6/0/6 release checks 全部通过时，
train split 的同源 C-ISO/T-TRAJ 记录才写 `training_authorized=true`；validation split 永远为 false，no-go 时
所有记录均为 false。该字段贯穿 C-ISO、T-TRAJ 及其 transitions、matching transitions、manifest 和 split
artifact。writer 会复算 release checks、授权、bijection、counts 与 canonical hashes，任何缺失或矛盾都拒绝
创建输出。即使授权为 true，raw artifacts 仍显式标为非 SFT-ready，必须先做确定性的 GT-free 模型可见格式
转换。本段只记录实现合同；通过状态须以后续 Slurm 定向与全量回归 receipt 为准。

## 资源、证据和正式提交门

正式运行使用 3 个 1-GPU shard：

- shard 0：`train_407_r3c2`、`val_43_r1c1`
- shard 1：`train_114_r3c1`、`val_41_r3c2`
- shard 2：`train_364_r0c2`、`val_60_r3c0`

每个 shard 使用 4 CPU、48G、2 小时时限。三个 shard 各自端到端处理两个样本：每一步先完成 GT-free
action catalog 和真实工具候选 batch，再在同一离线数据构造 worker 内用 GT 选择动作，并由 canonical
Environment 真执行一次作 replay。三个 shard 终止后，依赖的 0GPU Job 使用 2 CPU、16G 独立复算
指标、验证 receipt 并派生数据。finalizer 使用 `afterany`，以便 shard 失败时也写终态和发 QQ；只有三个
completion marker 都合法时才进入科学复算。正式任务固定在 `gpu48` 的 A800；调度过程实际同时运行峰值为
2 卡，仍满足不超过 3 卡。所有长任务均写共享工作区日志并执行 QQ 终态通知路径；通知器不保存
投递回执，因此只能证明调用路径执行，不能从本地证据证明邮箱送达。

独立复算的终态要区分 `global_release_passed` 与 `scientific_no_go`：二者都表示 validation workflow 正常
完成，区别只是后者禁止下一轮训练数据放行；只有合同漂移、证据不完整、独立复算不一致或基础设施故障才是
validation failure 和非零退出。

正式提交前还必须完成：

- pilot implementation A、config+doc-only freeze B 和 `B^ == A`；先由 A 固化实现、测试、运行脚本及其
  preregistration 基线，再让直接子提交 B 必须且只能同时修改本 config 与本 0803 文档。B 中 config 的
  `pilot_implementation_commit` 必须写 A 的精确 commit，`pilot_freeze_commit` 必须保持 `null` 以避免自引用，
  status 进入 `frozen_pending_exact_commit_validation`；A/B 两个精确 commit 由外部 append-only manifest 与
  receipts 同时绑定；
- clean detached exact-B 零 GPU contract/full regression；
- 18 个原 parent 文件、crop、tool/checkpoint/interpreter/runtime source 的 hash 复核；
- SegAgent（含 bundled SimpleClick）与 OmniOVCD/SAM3 外部源码仓库的精确 commit 和 tracked-clean 复核；
- 首次 GT access 晚于 catalog freeze，以及 label 对 proposal/tool/hard-gate/Qwen 调用边界不可见的测试；
- action inventory、候选预算、offline selector、hard gate 与 replay 的负例；
- train/val parent/content 隔离及 test exclusion；
- 首次正式输出目录不存在；重试时 campaign 下只允许既有 append-only `attempt-NNNN`，新运行原子选择首个
  不存在编号；无本 campaign 活动作业，Slurm 控制面正常。

contract 与正式提交使用 diagnostics 中的非阻塞 `flock`，而不是永久目录锁。contract 失败保留当前
diagnostics attempt，下一次自动选择新的 `attempt-NNNN`；pilot 只接受 exact B、config、manifest、非零测试
日志及 preflight receipt/log hash 全部重新绑定通过的 contract receipt v3。正式 attempt 仅在所有只读检查
通过、即将 `sbatch` 时创建。若 array 尚未被 Slurm 接受，预留目录移动回 diagnostics，不在正式 outputs
伪装成实验；若任何 Job 已被接受而后续记录失败，统一 EXIT handler 取消已知 Job、轮询 `squeue` 尽可能确认
退出，并写 submission-failure receipt 与 QQ 通知。worker 在参数 `realpath` 前安装 bootstrap trap；共享挂载或
参数错误也尽可能留下 terminal receipt/QQ。单一 pilot array 固定 `0-2%3`、每 task 一卡，全局 pilot 提交锁和
活作业检查共同守住峰值三卡。

上述检查由 exact-B3 Job `54394` 全部通过后才提交正式 pilot；未用候选回归或 B2 receipt 代替。

## exact-B3 正式结果与训练决策

正式 attempt：
`/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl/outputs/CA_0803(2)-qwen3vl-oracle-trajectory-pilot/attempt-0002`。

| Job | 角色 | 终态 | 关键结果 |
|---:|---|---|---|
| `54394` | exact-B3 零 GPU contract | `COMPLETED/0:0`，2 CPU/16G，00:01:57 | preflight passed；runner 21/21、validator 46/46、full regression 595/595，3 skipped |
| `54396_0` | shard 0 | `COMPLETED/0:0`，1 GPU/4 CPU/48G，00:06:44 | 2 episodes、1 accepted step、10 candidates、Qwen=0 |
| `54396_1` | shard 1 | `COMPLETED/0:0`，1 GPU/4 CPU/48G，00:03:00 | 2 episodes、0 accepted steps、4 candidates、Qwen=0 |
| `54396_2` | shard 2 | `COMPLETED/0:0`，1 GPU/4 CPU/48G，00:08:26 | 2 episodes、3 accepted steps、12 candidates、Qwen=0 |
| `54399` | `afterany` 独立 validator | `COMPLETED/0:0`，0 GPU/2 CPU/16G，00:00:08 | validation workflow success；`scientific_no_go`、global release failed |

正式报告为 `validation/validation_report.json`，SHA256
`<hash omitted>`；matching manifest SHA256 为
`<hash omitted>`。三个 completion marker、四个
terminal receipt、runtime/config/runner/contract hash 与 size 均由 validator 重新核对。6/6 episode 都先冻结
GT-free live action catalog，之后才首次访问 GT；GT 未传给 runtime factory、Environment、SAM3/SimpleClick
或模型输入，Qwen 调用总数为 0。

| episode | 终止 | accepted | 初始 → 最终 | 解释 |
|---|---|---:|---|---|
| `train_407_r3c2` | `no_beneficial` | 1 | IoU `0→0.283115`，F1 `0→0.441294`，pixel error `6564→6305` | 首步有改善，下一轮没有可接受的继续动作 |
| `train_114_r3c1` | `already_endpoint` | 0 | IoU `0.916249`，F1 `0.956294` 不变 | SAM3 初始状态已经过 endpoint，不是非平凡训练轨迹 |
| `train_364_r0c2` | `max_steps` | 3 | empty-GT FP `790→508→347→213`，要求 `≤64` | 三步持续改善但预算耗尽，唯一直接支持另行测试更长步数的样本 |
| `val_43_r1c1` | `no_beneficial` | 0 | IoU `0.749492`，F1 `0.856811` 不变 | 增加步数不能产生当前动作清单里不存在的首个有效动作 |
| `val_41_r3c2` | `no_beneficial` | 0 | IoU `0.302006`，F1 `0.463909` 不变 | 同上 |
| `val_60_r3c0` | `already_exact` | 0 | empty-GT FP `0` | 初始精确，不是非平凡训练轨迹 |

正式汇总为 6 episodes、4 个非初始达标 episode、26 candidates、4 accepted transitions；endpoint@1/2/3
和最终 endpoint reachability 都是 `0/4=0%`，exact recovery 也是 `0/4`。4 个被接受转移的真实工具执行、
hard gate、RGB semantics、单调改善和 canonical replay 均为 `4/4`，说明状态传递链本身工作正常，但不能把
“每个已选步骤都合法”误写成“轨迹能到目标”。候选包含 SAM3 box 13 个、SimpleClick positive point 13 个；
最终 4 个被选动作全部是 SAM3 box。20/26 候选通过 hard gate，8/26 通过 metric gate，5/26 指标有益；其中
3 个本可改善指标的候选被 hard gate 拒绝，但不能把它们写成三个 `no_beneficial` 的直接原因：它们分别发生在
`train_407` step 1（SAM3 box 改善 130，同轮入选 box 改善 259）和 `train_364` step 2/3（SimpleClick point
分别改善 161/134，同轮已有等改善的 SAM3 box 入选），均不在终止性的 `no_beneficial` round，也没有改变
本次 oracle 选择或 endpoint 结论。它们仍值得作为 gate 过严风险独立审计。

相反，三个终止性 `no_beneficial` round 的已执行候选都没有被 gate 隐藏的正改善：`train_407` step 2 的
pixel-error reduction 为 `-806/-210`，`val_43` step 1 为 `-620/-41/-734/-22`，`val_41` step 1 为
`-2881/-43/-276/-66`；后者首个 box 还被 component gate 拒绝，但指标本身也显著变差。因此这些终止案例应
优先检查 frozen top-4 shortlist、proposal 覆盖/geometry、真实工具行为与 greedy policy，而不是先放宽 hard gate。

冻结 config 的 `already_endpoint.historical_initialization_check` 原断言“六个 0801 initial mask 都低于
endpoint”；正式 fresh-SAM3 初始化却得到 `train_114=already_endpoint`、`val_60=already_exact`，直接反驳
了该历史初始化假设。冻结的 global gate 又要求 6 个 endpoint success、6 个 evaluated episode 和 6 条非平凡
accepted trajectory，因此本固定六样本设计对该 gate **结构性不可能达到 6/6**。本轮按 fail-closed 记为
runtime-initialization assumption / sample-gate mismatch，不事后修改 config。这是正式 pilot 暴露出的设计问题：
不能把初始成功冒充动作轨迹，也不能把它们当工具不可达失败。下一轮应重新选择 fresh SAM3 初始未过 endpoint
的样本，或重新预注册清楚的分母与初始成功处理规则；不得重标 attempt-0002。

matching manifest 的 endpoint-success/transition/C-ISO/T-TRAJ 计数全部为 0；train/val 四个 JSONL 文件均
为空，`training_authorized=false`、`authorization_scope=none`、`raw_artifacts_sft_ready=false`。因此本轮
不创建正式训练集、不做格式转换、不启动 SFT。下一步先分开处理三个问题：为 gate 分母重选或重预注册样本；
对三个 `no_beneficial` 样本分析 proposal/action policy/geometry；独立审计 3 个 hard-gate false reject，但不预设
它们造成了终止。只对 `train_364` 这类连续改善且三步耗尽的样本，另行预注册更大 `max_steps` 的独立实验。
新实验必须使用新样本合同、新 freeze 和新 append-only attempt。

`max_steps=3` 是用户指定的正式当前预算，也是本 cheap pilot 的结论边界。框架默认和 SegAgent 的常见
评测允许更多步；所以三步失败只表示“冻结 top-4 greedy oracle policy 在三步内未达终点”，不能外推为
更长轨迹或完整 executor 动作空间永远不可达。

## Pilot 结果后的决策分支（用户补充，2026-08-03）

本轮执行期间不得根据中间结果临时增大 `max_steps`，也不得降低 6/6 endpoint gate。最终结果先按冻结的
三步合同独立验收并保留失败轨迹，然后按证据进入以下分支：

- 若 6/6 endpoint gate、全部证据门和 train/val 隔离均通过，则同源 T-TRAJ/C-ISO 获得下一阶段候选资格。
  在完成确定性 Qwen 可见格式转换及数据 schema/hash 复核后，可以设计小规模、匹配 token exposure 或
  optimizer steps 的 SFT 对比；“工具 oracle 可达”本身仍不等于“Qwen 已学会选动作”。
- 若主要失败轨迹每步持续严格改善、只是到第 3 步仍未到 endpoint，才把“增加 `max_steps`”列为有证据支持的
  下一项实验。该实验必须另行预注册步数、候选和工具执行预算，使用新 freeze/attempt，不能覆盖或重标本轮
  三步结果。
- 若失败源于首步就没有有效候选、RGB/safety gate 拦截、真实工具失败或状态/replay 不一致，盲目增加步数
  通常不会解决根因；应优先分析动作空间、proposal、gate 或工具链，再决定是否修改框架。

因此，证据完整 episode 的 `scientific_no_go` 必须按 `max_steps` 耗尽、`no_beneficial` 和 gate false reject
等原因拆分统计；工具或证据失败则由 shard/validation terminal receipt 单列为 validation failure，不能伪装成
科学 no-go。是否增步或进入 SFT 由最终 receipt 与失败轨迹决定，不由主观预期决定。

## 本轮工程八荣八耻落实

- 查档求真：以真实 Environment/Executor、工具 receipt 和独立 mask 重建为准，不用口头假设代替执行证据。
- 对齐需求：本轮只回答工具层三步轨迹可达性；Qwen 能否自己选中动作仍是后续问题。
- 请示规则：GT 只在离线 shortlist/selector/scoring 使用，train/val/test 和 endpoint 门预先冻结。
- 复用存量：复用 production SAM3、SimpleClick、MaskPairProcessor、hard gate 和 Slurm/QQ 链。
- 完备测例：先 exact-B 零 GPU contract/full regression，再运行六样本 pilot 和独立 validator。
- 恪守规范：管理节点不跑计算，GPU 峰值不超过三张，失败 attempt append-only 且不覆盖。
- 坦诚存疑：RGB gate、非盲样本、三步预算和 non-planner tier 都保留限制，不包装成泛化结论。
- 分步迭代：只有六样本 endpoint 6/6 放行，才把同源轨迹提升为下一轮训练候选。
