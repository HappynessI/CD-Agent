# Qwen3-VL 0801 失败分析与 0802 实验计划

> 状态：**P0（0 GPU）已验收通过；等待向用户汇报；P1 未授权**
>
> 日期：2026-08-02
>
> 开发仓库：`/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl`
>
> 分支：`feat/local-qwen3vl-verifier`
>
> 分析时 HEAD：`0a5300ddc1d738a64c5bc93db6b0069f5c250fa7`
>
> 冻结 runtime 锚点：`491532f4622667e9fbf299c71aac228fa37f6a04`
>
> 资源硬约束：任一时刻最多使用 3 张 GPU；本计划建议峰值 2 张 GPU

P0 首次 validation Jobs `52553/52554` 因把 Slurm spool 脚本路径误当作 Git worker
路径，在任何测试开始前 fail closed；R1 Jobs `52562/52563` 已验证 spool/repo worker 字节
身份修正有效，但暴露出 legacy rollout marker 与 requested/scored order 被混成同一字段语义，
targeted/full regression 均以相同三个 error 失败。两轮都是 0 GPU，均未产生 H0/H1 科学结果；
旧输出完整保留。首次 amendment 与独立 R2 amendment 分别为
`QWEN3VL_P0_VALIDATION_AMENDMENT_0802.{md,json}` 和
`QWEN3VL_P0_VALIDATION_AMENDMENT_R2_0802.{md,json}`。当前只预留唯一 R2，P1 状态不变。
R2 机器可读 amendment 的 canonical SHA256 为
`<hash omitted>`。

## 0. 结论先行

0801 的 v1–v7 **没有任何一个版本证明优于 base verifier**：

- v1 接受了错误的大框候选，使固定 e2e3 的 selected micro IoU/F1 从 `0.74223120/0.85204673` 降到 `0.36876640/0.53883030`。
- v2–v7 的 selected micro IoU/F1 都与 base 完全相同，但这不是提升。v2–v4、v7 没有成功 tool action/candidate；v5/v6 生成了有害候选，依靠 hard gate 和 rollback 才回到初始 mask。
- v2–v7 都在 `test_1_1` 保留 `FP=626`、在 `test_50_8` 保留 `FN=166` 后错误 finish，属于 false convergence。
- v5/v6 的候选在 Qwen candidate-evidence stage 前就被 hard gate 拒绝，因此这两次评测不能证明 candidate-evidence SFT 有效或无效。
- 旧评测和汇总口径还会掩盖失败：固定 3 条反复用于选择、只报 micro aggregate、把 post-SFT rollout 的 initial mask 称作“训练前 E2E”、只用 IoU 下降判断 sample degradation，且 24 条 val 从未做生成式功能评测。

因此，0802 **不应直接堆一个 v8 SFT**。推荐顺序是：

1. 先修证据、指标、registry、resolved config 和路径合同，不使用 GPU；
2. 版本化并冻结真正支持独立复核轮的探索 harness；
3. 做离线 actionability oracle，确认当前动作空间确实存在可达的有益动作；
4. 做 stage-level 能力门，区分模型能力、监督泛化、动作空间和 hard gate 问题；
5. 只有前述门通过，才做单变量、固定预算的最小 SFT 消融；
6. 先过 CORE3、再过 EXTENDED5，最后才一次性打开至少 30 条的 locked set。

本文只给出分析和预注册计划。**截至本文完成，没有提交训练、评测、数据构建或 GPU 作业，也没有创建正式运行 worktree。**

上述句子记录本文首次冻结时点。后续 P0 时间线只包含 0-GPU Slurm contract validation：
正式 clean worktree 已创建；两次失败与唯一 R2 的授权、输出和证据 hash 见两份独立
amendment。它们不构成 P1、训练、评测或 GPU 授权。

## 1. 分析范围、证据和术语

### 1.1 权威输入

- 交接文档：`doc/training_pipeline/0801_second_try/QWEN3VL_SESSION_HANDOFF_0801.md`
- 0801 基线记录：`doc/training_pipeline/0801_second_try/baseline_0801.md`
- 数据交接与设计：
  - `doc/training_pipeline/0801_second_try/QWEN3VL_TRAINING_DATA_HANDOFF_0801.md`
  - `doc/training_pipeline/0801_second_try/QWEN3VL_V121_BASELINE_DATA_0801.md`
- 归档证据：`outputs/CA_0801_archive/`
- 实验登记：`experiments/experiment_registry.json` 及生成的 workbook
- 冻结 runtime：`491532f4622667e9fbf299c71aac228fa37f6a04`

分析时已用 Git 核对：当前 HEAD 的 `change_agent/` 相对该 runtime 锚点 diff 为空。旧实验的 repo commit 虽不完全相同，但 runtime 核心保持冻结；0802 仍需冻结整个运行 commit，避免 builder、submitter、summarizer 的演进混入模型对照。

已核对的本地依赖路径均存在：

- base model：`/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- 数据：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/train_original`、`val_original`、`test_256`
- 0801 归档：`outputs/CA_0801_archive`

归档只作为只读证据。不能就地修改归档 JSON、模型或 summary，也不能直接拿已经失效的归档数据路径重训。

### 1.2 三类指标必须分开

后续报告不得再把以下三者混为一谈：

1. **initial segmentation mask**：当前 rollout 初始化得到的 SAM3 mask；
2. **base-verifier selected**：base Qwen verifier 在冻结 harness 中完成决策后的 selected mask；
3. **post-SFT-verifier selected**：SFT 模型在完全相同 harness 中完成决策后的 selected mask。

当前 registry 中所谓“训练前 E2E”实际是第 1 项，不是真实的第 2 项。0802 的主比较必须是第 3 项相对第 2 项，并同时保留第 1 项作诊断。

### 1.3 事实、推断和待验证假设

本文遵循三层证据口径：

- **轨迹事实**：可从 trajectory、summary、receipt、trainer state 或日志直接复核；
- **受限结论**：事实支持，但只适用于当前小样本、单 seed 和当前协议；
- **待验证假设**：包括 2B 能力上限、冻结视觉塔影响、视觉 grounding/text shortcut、轨迹监督必要性等，未通过对照前不得写成根因。

## 2. 0801 实验复盘

### 2.1 e2e3 主结果

固定 CORE3 顺序为 `test_20_15`、`test_1_1`、`test_50_8`。表中 IoU/F1 均为 selected mask 的 micro aggregate。

| 版本 | train / val | epoch / optimizer step | train loss | selected IoU / F1 | candidate / accepted | action error | audit valid / total；regional error |
|---|---:|---:|---:|---:|---:|---:|---:|
| base，Job 51727 | — | — | — | `0.7422312 / 0.85204673` | `3 / 0` | 0 | `18/18；0` |
| v1 | `39 / 24` | `4 / 20` | `0.292859` | `0.3687664 / 0.5388303` | `4 / 1` | 1 | `16/19；3` |
| v2 hardcase | `48 / 24` | `2 / 12` | `0.473584` | 与 base 相同 | `0 / 0` | 1 | `18/18；0` |
| v3 action | `53 / 24` | `2 / 14` | `0.450054` | 与 base 相同 | `0 / 0` | 1 | `17/18；1` |
| v3-bgfp | `59 / 24` | `2 / 16` | `0.440129` | 与 base 相同 | `0 / 0` | 1 | `15/18；3` |
| v3-bgfp-scope | `59 / 24` | `2 / 16` | `0.441267` | 与 base 相同 | `0 / 0` | 1 | `16/18；2` |
| v4 residual | `65 / 24` | `2 / 18` | `0.419705` | 与 base 相同 | `0 / 0` | 1 | `11/18；7` |
| v5 contrast | `69 / 24` | `2 / 18` | `0.422574` | 与 base 相同 | `1 / 0` | 1 | `12/18；6` |
| v6 matched | `63 / 24` | `2 / 16` | `0.422066` | 与 base 相同 | `1 / 0` | 1 | `13/18；5` |
| v7 weighted | `83 exposure / 63 unique / 24` | `2 / 22` | `0.376048` | 与 base 相同 | `0 / 0` | 1 | `11/18；7` |

这张表支持四个直接结论：

1. v1 是明确的 completed scientific negative，不是“训练没跑起来”。
2. v2–v7 没有 selected 增益；“与 base 持平”来自不动作或回滚。
3. loss 不能代理闭环能力。v7 loss 低于 v6，但 successful tool action/candidate 归零，仍有 1 次 action attempt error，regional error 从 5 增至 7。
4. v6→v7 同时改变 exposure 和 optimizer steps（16→22），所以旧实验无法单独归因于 weighting。

### 2.2 失败轨迹

#### F1：v1 false accept 造成灾难性 FP

证据：

`outputs/CA_0801_archive/CA_0801(10)-qwen3vl-v121-post-sft-e2e3-v1/run/trajectories/test_50_8/trajectory.json`

- 初始 `TP/FP/FN = 0/0/166`；
- scan box `[0,0,329,329]` 映射到 pixel ROI `[0,0,84,84]`；
- 模型把错误区域判断为 `background -> building, clear` 并接受；
- selected 变成 `TP/FP/FN = 0/6518/166`。

该样本的 IoU/F1 前后仍为 0，所以旧 summarizer 将其记录为 `degraded=false`；但新增的 6518 FP 使整体 micro IoU/F1 大幅下降。这同时暴露了模型错误和 summary 退化诊断口径的盲点；后者不是导致在线候选被接受的 action safety gate。

#### F2：v2–v7 false finish / false convergence

所有 v2–v7 都重复出现：

- `test_1_1`：留下 `FP=626`、sample IoU=0，仍授权 finish；
- `test_50_8`：留下 `FN=166`、sample IoU=0，9 个 scan tile 都被判断为 `background -> background`，checklist 没有给出 fail/uncertain（部分字段为 `not_applicable`），随后 finish。

因此 summary 中的 `converged` 只表示流程终止，不表示分割正确。现有 `max_steps=3` 也没有带来三轮探索：initial verifier 一旦 finish，会在环境 loop 前退出；`plan=null` 也会在首轮终止。

#### F3：semantic states 与 checklist 自相矛盾

v4/v7 在 `test_20_15` 的多个 proposal 上输出 `background -> background`，同时七项 checklist 全 pass。runtime repair 后仍矛盾，只能 fail closed 为 `uncertain_region`，使 atomic-audit 合同有效记录降到 `11/18`；这不是整个 rollout protocol 的完成率。

这不是 JSON 能否解析的问题，而是模型没有稳定学会“state/polarity/checklist/action”之间的逻辑约束。

#### F4：v5/v6 触发工具候选，但未进入 candidate-evidence stage

- v5 的 negative-point 候选删除真实变化，IoU 从 `0.84638554` 降到 `0.79122246`；
- v6 的候选同样删除真实 component，IoU 降到 `0.78260099`，FN 从 254 增至 614；
- 两者都被 `point_proposal_combined_change_gate_exceeded` 拒绝并 rollback。

这说明安全门正常发挥了保护作用，但不说明 verifier 已学会 candidate 判断。两次 `candidate_evidence` 调用数都是 0，Qwen candidate stage 在 hard gate 后根本没有执行。

#### F5：存在真正有益 candidate 被 false reject

base e2e5：

- `test_78_13`：IoU `0.75787116 -> 0.76640245`，FP `3246 -> 3077`，仍被拒绝；
- `test_85_16`：IoU `0.30658070 -> 0.31563287`，FP `2963 -> 2834`，仍被拒绝。

v2 e2e5 的 `test_85_16` 更明显：

- IoU `0.30658070 -> 0.32554297`；
- F1 `0.46928705 -> 0.49118433`；
- FP `2963 -> 2701`，TP/FN 不变；
- verifier 仍把 removed delta 判断为 `background -> building, clear` 并判 worse。

所以当前失败不是单向的“门太松”：既有 v1 false accept，也有 base/v2 false reject。0802 必须同时报告 candidate false-accept、false-reject 和 hard-gate-before-Qwen 三种路径。

### 2.3 训练、评测与统计设计缺口

#### 训练功能门缺失

- 所有 SFT 配置均为 `do_eval=false`、`do_predict=false`。
- 24 条 val 只通过数据 schema/分布合同，从未进行生成式 stage-level 功能评测。
- trainer state 没有 `eval_loss` 或功能指标；因此无法知道模型是在 train-like 样本上没学会，还是只在 parent-disjoint val 上不能泛化。

#### 数据和 seed 不足

- v1–v7 只有 39–83 个 exposure、9–11 个 train parent、6 个 val parent；
- v7 的 83 行实际只有 63 个 unique content，额外 20 行是重复曝光；
- 全部训练只使用 seed 801；
- stage snapshot 没有机械关联完整的 `audit -> action -> candidate -> next-state` 闭环。

这些是确定的实验设计事实；“轨迹监督不足就是唯一根因”仍是待验证假设。

#### 固定小样本和聚合方式掩盖失败

- e2e3 仅 3 条，其中 2 条 sample IoU=0；initial/base-selected micro IoU/F1 为 `0.7422312/0.85204673`，macro 只有 `0.28212851/0.30560087`。
- e2e5 仅 5 条，base micro IoU 为 `0.6762904`，macro IoU 为 `0.38216748`。
- CORE3 被 v1–v7 反复查看；EXTENDED5 也已用于 base 和 v2。两者都只能作为 diagnostic promotion gate，不能再证明 LEVIR-CD 泛化。
- 旧 `degraded_sample_count` 只检查 per-sample IoU 是否下降；对“零 IoU 不变但 FP/FN 恶化”失明。

#### 预算命名误导

- `max_steps=3` 是 environment action budget，不是 3 个独立 verifier round；
- `verifier_retries=2` 表示 initial generation 加一次 repair，共 2 个 stage attempt，不是两次新探索；
- finish/plan-null 提前退出，使最需要复核的失败恰好没有第二次独立观察。

0802 必须把 `max_verifier_rounds`、`max_stage_attempts`、`max_environment_steps` 三个预算拆开并单独记账。

当前 runner 的 `action_retries` 是 deprecated compatibility option，授权 geometry 实际只执行一次；不能再把默认值 3 解释成工具最多重试三次。0802 receipt 必须分别记录 verifier round、stage initial、schema repair、本地 backend network retry、tool action attempt 和 tool failure。

### 2.4 归档、registry 和复现缺口

归档移动后，内容哈希链仍可验证，但大量绝对路径已失效：

- 10/10 个 `evaluation_receipt.model_path` 不再存在；
- 遍历归档内全部 `train.json`/`val.json` 得到的非去重引用（包含失败和被替代目录）中，4312/4312 个 train image reference、1611/1611 个 val image reference 失效；这不是唯一图片数；
- summary 中的 rollout marker、source metrics、trajectory 路径也保留归档前地址。

不能就地修归档，因为这会破坏历史 SHA 和追溯性。新轮需要在新目录重建数据，采用可迁移路径合同，并生成新的 receipt。

另有以下登记问题：

- v2 `config_used.yaml` 仍写 39 samples、`1e-4`、4 epochs；实际日志/trainer state 是 48 samples、`3e-5`、2 epochs、12 steps。该文件不能作为 resolved config。
- registry 缺失真实 base e2e3/e2e5，也缺失 v1 completed-negative。
- registry 的 0801 路径仍是归档前历史路径；应并列保存“历史运行路径”和“当前证据路径”。
- v1 data 的 artifact producer 与成功 receipt 不是同一 Job：51876/commit `41a2bd...` 已写出 manifest 后才在 summary inline Python 报错，51883/commit `9fad7f...` 只复核现有 artifact 并闭合成功 receipt。两者必须分列，不能用成功 retry 的 commit 冒充 manifest producer。
- workbook checker 遇到缺失的 0729 artifact 会直接 return，导致后续固定样本 hash 校验也被跳过。
- registry 中仍有 v11.1 标题、CORE membership 和 EXTENDED5 状态陈旧问题。

正面证据也必须保留：

- 已核对的 manifest/model/evaluation hash 链匹配；
- 48/48 条轨迹记录 `git_dirty=false`；
- rollout 均记录 `gt_loaded=false`；
- 成功评测的 `tool_failure_count=0`。

因此没有证据表明“基础设施故障”或“在线读取 GT”导致了零提升。确有过 `torchrun` 路径、数据 summary 语法、scope 统计和 contract 单测等基础设施失败，但对应正式训练/评测已成功重跑，应与 scientific negative 分开登记。

## 3. 失败原因分层判断

### 3.1 已证实的直接失败机制

1. **语义合同不稳定**：state、polarity、checklist 和 action/finish 之间出现矛盾。
2. **false convergence**：残留明显 FP/FN 时授权 finish，且当前 harness 没有真正的独立复核轮。
3. **候选方向判断双向出错**：既接受过灾难性坏候选，也拒绝过可量化的好候选。
4. **安全回滚掩盖模型无效**：selected 不退化不等于模型改进；v5/v6 的 headline 持平来自 hard gate。
5. **监督 loss 与闭环指标脱钩**：更低 loss、更多重复曝光没有转化为 protocol validity、action 或 accepted candidate。
6. **评测口径不足**：micro-only、小样本、单 seed、错误 degradation 定义和 initial/base/post 混写，使结论容易过度乐观。
7. **复现合同不完整**：归档绝对路径、resolved config 和 registry 覆盖不足，阻碍可靠复现。

### 3.2 尚未证实、必须用对照区分的假设

| 假设 | 当前证据 | 判别实验 |
|---|---|---|
| Qwen3-VL-2B 容量不足 | 复杂逻辑不稳定，但没有 matched stronger-model control | P3 stage-level gate；更强本地模型仅在路径、许可和资源明确后作为独立 arm |
| 冻结 vision tower/projector 妨碍小目标 grounding | 全部旧 SFT 都冻结，`test_50_8` 小目标漏检 | 先做视觉/文本对照能力门；只有结果支持时再单独预注册 unfreeze arm |
| 数据不平衡导致 all-pass/finish 偏置 | v2 后从有害动作摆向过度保守，v7 重复曝光无效 | P3 confusion matrix；P4 固定预算的真实 weighting 单变量消融 |
| 缺少 trajectory-linked supervision 是主因 | 当前只有孤立 stage snapshot | 仅当 stage 能力改善但 E2E 不改善时运行 T3 trajectory-linked arm |
| 当前动作空间不可达，或 hard gate 误拦改善 | 已观察到好候选被拒，但尚无系统 actionability 统计 | P2 离线 oracle actionability 与 gate confusion audit |
| scan 标签口径不一致影响学习 | 零白像素 precision 在历史标签中有 `not_applicable/pass` 两种口径 | P0 统一 schema，重建后做确定性合同测试 |

在这些对照完成前，不得宣称“2B 不行”“视觉塔必须解冻”或“加轨迹数据一定能解决”。

## 4. 0802 分阶段实验计划

### P0：证据卫生、指标和预注册（0 GPU）

目标：先保证之后每个数字可解释、可复现，避免带着错误口径继续训练。

工作项：

1. 迁移 registry schema/README，把正交状态拆开：
   - `execution_status`：`planned/running/completed/failed/cancelled`；
   - `scientific_outcome`：`not_evaluated/baseline_reference/pass/negative/inconclusive`；
   - `lineage_status`：`canonical/superseded`；
   - 单独 attempt ledger 登记全部 Slurm 尝试。v1 应为 `completed + negative + canonical`，infra failed attempt 不冒充科学结论。
2. registry 同时记录历史运行路径、当前 archive 证据路径、Job ID、commit、manifest/model/tool hash。
3. 修复 workbook checker 的 early return，并让 archive receipt、固定样本顺序/hash 都独立校验。
4. 修改 summarizer，至少同时输出：
   - initial、base-selected、post-selected 三组指标；
   - per-sample 与 micro/macro IoU/F1；
   - TP/FP/FN 及变化；
   - candidate oracle gain、false accept、false reject、hard-gate-before-Qwen；
   - action/protocol/tool/termination 原因。
5. 每个 0802 新 run 保存真正的 resolved config，而不是只复制模板 YAML。0801 原始 archive 保持不变；对 v2 只能在 archive 外生成带证据来源和 `reconstructed_resolved_config` 标记的映射，不能冒充原始文件。
6. 新数据采用新目录、可迁移路径和完整 receipt；历史 archive 保持只读。
7. 写入预注册 manifest：唯一变量、seed、样本 hash/order、主指标、停止门、资源、输出目录和通知标题。当前冻结文件为 `QWEN3VL_H0_H1_PREREGISTRATION_0802.json`，输入字节身份另由 `QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json` 固定；其 canonical SHA256 为 `<hash omitted>`。
8. 冻结 metric/candidate 判定合同，并对空 GT、空 prediction 和零分母写单测。

metric 合同从 TP/FP/FN/TN 像素 counts 独立重算指标，并在四舍五入前比较；counts 只在 IoU/F1 同时持平时作为退化 tie-break：

- `IoU = TP/(TP+FP+FN)`；分母为 0 时记 1；
- precision 分母为 0 时：若 GT 也无正例记 1，否则记 0；
- recall 分母为 0 时记 1；F1 为 precision/recall 的调和平均，两者之和为 0 时记 0；
- candidate oracle-beneficial 定义为 `ΔIoU > 1e-12` 且 `ΔF1 > 1e-12`；
- oracle-harmful 定义为任一指标 `< -1e-12`，或两指标均持平但 TP 下降、FP/FN 任一增加；
- accepted harmful candidate 计 false accept，rejected beneficial candidate 计 false reject；beneficial candidate 在 Qwen 前被 gate 拒绝时单列 hard-gate false reject；其余边界情形列为 indeterminate，不强塞进二分类。

机器可读 confusion 合同使用互斥 leaf class，并固定分母：false-accept 分母为全部 oracle-harmful candidates；Qwen false-reject 分母为到达 Qwen 的 oracle-beneficial candidates；hard-gate false-reject 与 all-false-reject 分母为全部 oracle-beneficial candidates。hard-gate leaf 可向上汇总到 all-false-reject，但不得与 Qwen false-reject 重复计数。

P0 准入门：

- registry JSON 与 XLSX 确定性同步；
- checker 不因任一无关旧 artifact 缺失而跳过 0801/0802 检查；
- summary 在 v1 轨迹上能把新增 6518 FP 判为退化；
- 空/非空 GT × 空/非空 prediction 的 metric truth table 和 candidate confusion 单测全部通过；
- 0802 新 run 的 resolved config 与日志/trainer state 一致；0801 重建映射明确标记 reconstructed，且不改 archive；
- 工作树和正式运行 worktree 的 commit/dirty 状态可审计。

任一项不满足，不提交 GPU 作业。

预注册状态必须按时间轴解释：`design_status=frozen` 只表示设计不能静默变化；P0 acceptance receipt 在设计冻结时仍为 `pending`；P1 保持 `not_execution_ready`、`p1_execution_authorized=false`。P0 作业完成后的观察结果写入仓库内独立 receipt，不回填不可变设计 manifest。

### P1：版本化探索 harness（先 0 GPU 测试，再 1 GPU smoke/CORE）

先做行为保持的预算拆分，再比较探索 arm：

| arm | verifier rounds | stage attempts | environment steps | finish/plan-null 行为 |
|---|---:|---:|---:|---|
| H0 | 1 | 2（initial + repair） | 3 | 保持旧 v12.1 行为 |
| H1 | 2 | 每轮最多 2 | 3 | 首次 finish 或 plan-null 后，允许恰好一次 GT-free、非重复的独立复核 |

H1 必须使用独立的 harness protocol ID、experiment arm ID、summary artifact 和 receipt artifact；H0/H1 两个 artifact 都由同一组公共、版本化且可验证的 `qwen3vl_e2e_summary_v2` / `qwen3vl_evaluation_receipt_v2` schema 约束。不得继续登记成旧 `v12.1/v12_1-compact-post-sft`。确切 ID 在 P0 预注册后冻结。

唯一因果变量为 `terminal_review_scheduler_v1`：H0 使用 legacy single-round terminal policy，H1 增加一次 GT-free independent terminal-review round。模型、工具、A/B 输入 hash/order、seed、generation、Environment 初始状态、stage attempt/environment step 预算和全部安全门均为控制变量。完整 `round1 trigger × round2 outcome` 状态表必须在实现前固定；其中 `plan_null + finish` 是 conflict/no-action，只有第二轮唯一 safe plan 可进入原安全流程。

H1 约束：

- 第二轮不能读取或利用 GT，也不能因为 GT 指示“还有错”而触发；
- 复核使用预先定义的 GT-free region/proposal/scan 选择，不重复同一 raw prompt 伪装成探索；
- 不降低 frozen parser、geometry authorization、locality gate、candidate benefit-without-harm 或 rollback；
- 保存每轮 raw output、attempt 类型、region、plan、动作、终止理由和三个预算计数；
- base 和所有 post-SFT 模型必须在完全相同的冻结 H1 commit 上运行。

P1 harness freeze 的 primary endpoints 只衡量协议与安全：H0 parity、eligible trigger 恰好一次第二轮、paired H1 final-region protocol-valid rate 不低于 H0，以及 GT boundary violation、safety relaxation、requested/scored mismatch、tool failure 全为 0。IoU/F1、candidate oracle gain 和是否出现新诊断/candidate 是 diagnostic/promotion 指标，不得驱动在线选择或为满足 quota 强制动作。

H1 冻结门：

- 行为保持单测证明 H0 与旧协议一致；
- 指定 finish/plan-null case 确实进入第二 verifier round；
- rollout marker 前 `gt_loaded=false`；
- requested/scored 对齐，tool failure 为 0；
- protocol validity 不低于 H0；
- 至少一个历史 false-finish case 必须实际进入 schema-valid、独立且可确定性回放的第二轮；第二轮是否形成新的合法诊断或 candidate 作为诊断指标报告，不作为 harness 安全冻结门，也不得据此强制动作；
- 上述安全/协议任一项未过即停止，不冻结 H1。若安全门均过但没有形成新的合法诊断/candidate，如实进入 capability diagnosis，不为迎合 CORE3 调整 harness。

### P2：离线 oracle actionability（工具执行可用 1 GPU；GT 只在事后评分）

目的：回答“当前动作空间里是否存在 verifier 有可能选对的改善动作”，避免在不可达任务上继续 SFT。

方法：

1. 对预注册诊断样本枚举当前协议授权的 point、box、scan-derived action；
2. 使用正式 tool chain 生成 candidate；
3. candidate 生成完成后，才在离线 evaluator 中加载 GT 计算 IoU/F1、TP/FP/FN；
4. 分别记录：最佳可达 gain、无改善动作、hard gate 正确拒绝、hard gate 误拒、Qwen false accept/false reject；
5. GT 分数永不进入 verifier 输入、prompt、第二轮触发或在线选择。

运行前必须冻结有限 oracle manifest：样本及图片 hash、H1 commit、action-generator 代码 hash、允许的 action family、region/proposal 清单、point/box 构造规则、seed、每类预算、canonical 去重规则和 tool hash。若有限枚举没有改善，只能表述为“在该预注册 generator/budget 内未找到可达改善”，不能外推为连续动作空间全局不可达。所有 gate confusion 必须报告分子、分母和 Wilson 95% CI，不使用没有预注册阈值的“系统性”措辞。

决策：

- 若 `test_1_1`/`test_50_8` 等目标失败类型在预注册 generator/budget 内没有任何授权动作可改善，停止 verifier SFT，先单独评审动作空间/tool proposal；
- 若任一 oracle-beneficial candidate 被 hard gate 拒绝，P2 只报告、不直接修改 gate。任何 gate 变化都必须另行向用户审批，建立独立 gate arm，并从 P1 安全门和 T0 base control 重新运行；不得混入既有 H1/SFT 对照，更不得静默放宽；
- 若动作可达且 gate 可通过，而 verifier 仍选错，再进入 P3/P4。

### P3：stage-level capability gate（1 GPU，先评模型，不训练）

建立 parent-disjoint 的生成式功能集，覆盖：

1. T1/T2 semantic states；
2. state 与 white/black mask polarity 一致性；
3. 七项 checklist 与 state 的逻辑一致性；
4. audit 到 runtime action/finish 的派生；
5. candidate `better/worse`；
6. scan true-change/background；
7. schema、enum、scope 和 repair 行为。

首次生成前冻结 train-like 与 parent-disjoint 两份 manifest。每个 split 至少包含：四种 semantic state 各 20 个 unique content、action/finish 各 20、candidate better/worse 各 20、scan true-change/background 各 20；样本可跨能力维度复用，但同一 prompt+image+target 不重复计数。parent-disjoint 集覆盖 6 个保留 val parent，且与 train-like 无 parent 重叠。若数据无法在不复制内容的情况下达到下限，应报告 capability set 不可构建并停止，不能看过模型输出后降低下限。

先评 base、archived v6、archived v7；分别报告 train-like 和 parent-disjoint val，不先训练新 v8。归档模型若因 receipt 路径失效，使用 archive 内实际 merged model 并生成新的只读映射 receipt，不修改旧 receipt。

建议准入阈值：

- strict JSON/schema：100%；
- state–polarity–checklist 逻辑一致性：100%；
- semantic state、action/finish derivation、candidate polarity、scan polarity：macro 和每一类别都至少 90%，并报告分子/分母、parent 覆盖和 confusion matrix；
- train-like 未过：先做 job-local、结束不保 checkpoint 的 overfit smoke，定位训练实现或模型能力，不跑 E2E；
- train-like 通过而 parent-disjoint val 未过：判定为泛化/覆盖问题，先改数据设计，不跑 E2E；
- P3 诊断完成后才决定 P4 的单变量 treatment；任一模型只有通过 parent-disjoint 功能门，才能进入 P5 E2E。

更强模型 matched control 是条件 arm：本地模型路径、显存、许可和 worker 均明确后再预注册，不能与数据课程、解冻或 harness 变化混在同一 arm。

### P4：最小 SFT 消融（每个训练 2 GPU，严格顺序执行）

| arm | 作用 | 唯一变化 |
|---|---|---|
| T0 | frozen base control | 不训练；在同一 H1 上获得真实 base-verifier selected |
| T1 | 可复现训练 control | 在新路径重建 v6 的 63 unique joint 数据，复用已验证超参 |
| T2-W | weighting 消融 | 与 T1 内容、顺序、总 token、optimizer steps、超参完全一致，只改变真正的 loss/sample weighting；不靠复制样本增加总更新 |
| T3（条件） | trajectory-linked 消融 | 仅当 T2-W 过 stage gate 但 E2E 不改善时，把同源 episode 的 snapshot 替换为机械关联的 `audit -> action -> candidate -> next-state` 表达；不直接追加曝光 |

实现 T2-W 前必须查清当前 trainer 对 sample/token weighting 的真实接口；若不支持，先设计可测试的 deterministic sampler，不得臆造配置字段。

T3 使用与 T2-W 相同的底层 episode/image/semantic label，optimizer steps 必须完全相同，输入+target 总 token 预算误差不超过 1%。若无法同时匹配内容来源和预算，T3 必须标为 composite exploratory arm，不能把差异单独归因于 trajectory linkage。

训练约束：

- base model 固定为本地 `Qwen3-VL-2B-Instruct`；
- T1/T2-W 首轮固定 seed 801；
- 只有同一 arm 同时通过 stage gate、CORE3、EXTENDED5 后，才补 seed 802/803；
- 不选择最好 seed，所有 seed 和负结果完整报告；
- 训练数据只来自 train split，val parent-disjoint；`test_256` 不进入训练或数据选择；
- raw GT 可用于离线构造/机械校验标签，但不得进入 model input/target 中未授权字段或在线 rollout；
- 每个 arm 保存 data manifest、resolved config、trainer state、adapter/merged-model hash、代码/工具/环境 receipt。

### P5：E2E promotion gate（每次 1 GPU）

所有比较均使用同一冻结 commit、H1、工具版本、样本顺序和 seed。先 CORE3，过门后才 EXTENDED5。

CORE3 硬门：

1. 3/3 完成，requested=scored；
2. `tool_failure_count=0`、`action_attempt_error=0`、`regional_error_count=0`；
3. successful tool action 至少 1 个；
4. accepted candidate 至少 1 个；
5. 至少一条样本的 IoU 和 F1 同时严格提升；
6. 相对 T0 base-verifier selected，micro 和 macro IoU/F1 都严格提升；
7. 任一样本 IoU 或 F1 下降即失败；若 IoU/F1 持平，则 TP 不得下降、FP/FN 不得增加；指标严格改善时允许可解释的 count tradeoff，但必须逐样本报告；
8. 按 P0 的 oracle 定义全量报告 candidate false-accept、false-reject、hard-gate-before-Qwen 和 indeterminate；
9. protocol validity 不得靠 retry 数增加掩盖，raw attempt 和终止理由必须可审计。

EXTENDED5 使用相同规则；任一 gate 未过，不进入多 seed 或 locked final。回滚后 headline 持平、低 train loss、tool 调用成功、或 3 条流程完成，都不能单独视为 promotion。

### P6：一次性 locked final（每个存活模型 1 GPU）

在任何 final 结果打开前：

- 从从未进入训练、curation、失败分析或人工挑选的 `test_256` 样本中，按预注册确定性 seed 和分层规则锁定至少 30 条；
- 排除长期反复使用的 5 条：`test_20_15`、`test_1_1`、`test_50_8`、`test_78_13`、`test_85_16`；
- 固定 manifest、顺序、图片 hash、工具 hash、模型 hash、主指标和停止门；
- 所有存活 arm/seed 在同一 locked manifest 上运行，查看结果后不得再调模型、prompt、gate 或阈值。

最终报告：

- 主指标：相对 base-verifier selected 的 paired macro `ΔIoU`、`ΔF1`；
- paired bootstrap 按样本成对重采样 10,000 次，固定 seed 20260802，使用 percentile 95% CI；同时报告 micro IoU/F1、TP/FP/FN、action/protocol/tool error 和逐样本结果；
- 三个训练 seed 的主分析先对每个样本的 paired delta 跨 seed 取均值，再对样本做 bootstrap；同时逐 seed 报点估计，三个 seed 的 macro `ΔIoU/ΔF1` 点估计都必须为正，不挑最好 seed；
- 稳定提升要求 arm-level macro `ΔIoU` 和 `ΔF1` 的 95% CI 下界都大于 0；
- catastrophic regression 预注册为任一 seed、任一样本 `ΔIoU < -0.05` 或 `ΔF1 < -0.05`，或违反 frozen geometry/locality/candidate/rollback 安全边界；阈值必须在 locked manifest hash 生成前固定，结果打开后不能修改；
- 未同时满足稳定提升和无 catastrophic regression，只能报告 scientific negative 或 inconclusive。

## 5. 最小执行矩阵与资源

| 阶段 | 典型作业 | GPU | 是否可并行 | 输出 |
|---|---|---:|---|---|
| P0 | registry/checker/summary 单测 | 0 | 轻量步骤可串行 | 新 commit 与测试记录 |
| P1 | harness smoke / CORE | 1×A800 | H0/H1 顺序 | 独立版本输出目录 |
| P2 | tool candidate 枚举 | 最多 1×A800 | 顺序 | oracle actionability receipt |
| P3 | stage-level generation | 1×A800 | 模型顺序 | capability report |
| P4 | SFT | 2×A800 | 所有 arm 顺序 | adapter、merged model、receipt |
| P5/P6 | E2E | 1×A800 | 按 promotion 顺序 | 完整 trajectory/summary |

计划峰值为 2 张 GPU，低于“最多 3 张卡”的硬限制。即使集群空闲，也不并发两个 2-GPU 训练，不用并发掩盖实验顺序和停止门。

集群执行规范：

- 管理节点只做编辑、轻量检查和 Slurm 管理；数据构建、测试、训练、评测均按负载通过 Slurm；
- 提交前运行 `scontrol ping`，使用 `squeue -u guisongxia` 检查重复作业；attempt ledger 记录本轮所有 Job ID，preflight 汇总处于 RUNNING/COMPLETING 的本轮 GPU allocation，并验证“现有本轮占卡 + 新请求卡数 <= 3”。不能只靠 Job name；无法确认时不提交；
- smoke 不保存 checkpoint；正式输出只保存完整有效产物；
- stdout/stderr 放入对应 `outputs/CA_0802(...)/logs/`，不写在 `outputs/` 根目录；
- 每个输出目录不覆盖，包含 Job ID、commit、配置和 receipt；
- 长作业无论成功或失败，都用账号专属 QQ 工具发送最终状态、关键结果、耗时和产物路径；不得打印通知凭据。

P0 manifest 已预留且在设计冻结时验证不存在的输出为：`CA_0802(1)-qwen3vl-p0-contract-validation`、`(2)-qwen3vl-h0-h1-cpu-fixtures`、`(3)/(4)-qwen3vl-h0/h1-base-smoke1`、`(5)/(6)-qwen3vl-h0/h1-base-core3`。每个目录、stdout/stderr 模板和 exact QQ title 均以机器可读预注册为准；submit preflight 必须再次验证 absent，冲突时停止并 amendment，不能现场顺延序号。

## 6. Git、worktree 和输出管理

### 6.1 当前状态

分析开始时开发树已有用户改动：7 个旧路径 tracked 文档删除，`0729_first_try/`、`0801_second_try/` 为未跟踪目录；移动前后 7 个文件内容 hash 一致，另有新的 handoff 文档。`0802_third_try/` 是本任务唯一新增目录。本文不得覆盖、恢复或混入前述用户改动。

旧 0801 参考 worktree 的 Git registration 已显示 `prunable`，实际目录又被移动。不能复制或移动旧 worktree 来创建 0802 worktree，也不在未确认前执行 prune/repair。

### 6.2 推荐提交边界

1. 用户已有 7 个 exact 文档 rename，单独提交；
2. 新增的 0801 session handoff，单独提交；
3. 本文 0802 分析文档，单独提交；
4. registry schema/README、checker、summarizer 按关注点拆成小提交；
5. 探索 harness 与对应测试，单独提交；
6. data contract、trajectory curriculum、weighting 分别按单变量提交。

每个提交只精确 stage 自己的文件；提交前检查 `git diff --check`、测试记录和 `git status --short`，不使用 destructive reset 或 force 操作。

### 6.3 正式运行 worktree

只有用户批准、所需代码提交完成且 exact commit 明确后，才从该 commit 创建全新的 detached clean worktree。根据 0801 的 Git registration，旧活动 worktree 原本位于 workspace root，之后被手工移动进 `run_worktree_archive/` 才变为 `prunable`；因此 0802 默认创建新的活动路径，而不是直接写入 archive：

```bash
git -C /guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl \
  worktree add --detach \
  /guisongxia01/pangchao/wangyihan/Chang-Agent-baseline-0802-run \
  <approved_commit>
```

创建后必须核验：

- `HEAD == <approved_commit>`；
- `git status --porcelain` 为空；
- frozen runtime diff 和工具 hash 符合预注册；
- 数据、模型、解释器和输出路径存在；
- 没有同名或重复 Slurm 作业。

代码从 clean run worktree 执行；正式产物继续显式写入开发仓库下新的 `outputs/CA_0802(...)` 目录，保证代码树和结果归档职责分离。

若用户希望新 worktree 从一开始就位于 `run_worktree_archive/`，应在批准 P1 前明确该路径，并直接用 `git worktree add` 创建到最终位置；不能先创建后用文件移动。实验结束后的归档也必须使用 Git-aware 的 repair/move 流程，避免重演 0801 的 stale registration。

## 7. 总停止条件

| 失败点 | 必须停止的后续动作 |
|---|---|
| P0 provenance/summary/checker 未通过 | 不提交任何 GPU 作业 |
| P1 任一 H1 冻结门未通过 | 不冻结 H1，不训练 |
| P2 在预注册 generator/budget 内没有可达改善动作 | 不继续 verifier SFT；先评审动作空间/tool proposal |
| P2 发现 hard-gate false reject | 保持原 gate；另行汇报审批，任何 gate arm 从 P1/T0 重跑 |
| P3 train-like 功能门失败 | 不跑 E2E；先定位训练实现或能力边界 |
| P3 parent-disjoint val 失败 | 不跑 E2E；先修数据覆盖/泛化 |
| CORE3 未过 | 不跑 EXTENDED5 |
| EXTENDED5 未过 | 不补 seed，不打开 locked final |
| locked final 未过 | 只报告负结果/inconclusive，不根据 final 调参 |

## 8. 工程八荣八耻在本轮的落实

- 查档求真：所有接口、预算语义、trainer weighting 能力先读代码/配置，不猜字段。
- 对齐需求：当前只分析、写计划和汇报；用户批准前不启动实验。
- 请示规则：GT-free、数据 split、promotion gate 和 final-set 使用均以冻结协议/预注册为准。
- 复用存量：复用现有 submitter、worker、parser、安全门、receipt 和 registry 生成链，不平行造一套隐式流程。
- 完备测例：每阶段有合同测试、正/反/边界样例和明确停止门；先 smoke 再正式运行。
- 恪守规范：不降低 frozen parser、geometry/locality/candidate/rollback 边界，不在管理节点跑计算。
- 坦诚存疑：2B 上限、视觉塔冻结、轨迹监督和动作空间均标为待验证假设。
- 分步迭代：证据修复、harness、能力门、SFT、promotion、final 逐步提交和验收，不把多变量混入一个 arm。

## 9. 待用户批准的首个动作

用户已经批准并完成 **P0（0 GPU）**：证据口径、registry/checker/summarizer、预注册
manifest 和验证链已通过下面记录的 Slurm CPU 验收。下一动作仅是向用户汇报；是否批准
P1 由用户另行决定，当前不实现 H1、不提交 CPU fixture/GPU 作业，也不批准新一轮 SFT。

## 10. P0 阶段验收结果（执行观察）

- validation commit：`e9fbcd690a9af2e91384231ced46555fe400d9ba`；正式 detached
  worktree 在启动、终态及 receipt 提交前均 clean。
- R2 targeted Job `52598`：`COMPLETED/0:0`，P0 contracts `19/19`、registry
  `14/14`、local verifier `49/49`；shell syntax、workbook、真实 v1 summary/assertion 和
  八文件 hash window 全通过。
- R2 regression Job `52599`：`COMPLETED/0:0`，360 项中 357 passed、3 skipped；两个
  Job 都是 0 GPU、2 CPU、8 GiB。
- receipt：`QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json`；canonical SHA256
  `<hash omitted>`，raw SHA256
  `<hash omitted>`。
- receipt 完整保留首次/R1 两次失败和 R2 成功：3 attempts、6 Jobs、32 个逐文件 hash、
  16/16 gates；`p0_accepted=true`、`failed_gates=[]`。
- P0 只证明证据与 harness 合同可独立复核。H1 尚未实现，没有观察 H0/H1 科学行为；
  `p1_execution_authorized=false`、GPU 未授权。
