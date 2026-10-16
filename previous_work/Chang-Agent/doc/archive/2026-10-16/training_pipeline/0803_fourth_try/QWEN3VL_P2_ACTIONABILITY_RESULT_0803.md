# Qwen3-VL P2 动作可达性结果（2026-08-03）

## 1. 一句话结论

P2 的工程链完整通过，但预注册的科学 gate **未通过**。在冻结的有限 CORE3 单步动作空间中，
`test_50_8` 存在 primary beneficial 且通过原 hard gate 的候选，`test_1_1` 却没有 primary
beneficial 候选。因此：

- `proceed_to_c_iso_and_t_traj_data_construction=false`；
- `next_action=stop_training_and_review_frozen_action_space`；
- 当前禁止创建 C-ISO 孤立阶段数据和 T-TRAJ 完整轨迹数据，也不进入训练、EXTENDED5 或 locked final。

这里的 `negative` 是一次成功完成的科学负结果，不是基础设施失败。它只表示本次冻结集合没有满足
停止门，不能外推成“更大的动作空间没有解”或“模型在 LEVIR-CD 上无效”。

## 2. 冻结身份

- P1 执行 freeze B4：`5210f89f5ca13ee1e636f5aa6a44631206a80123`
- P1 closeout acceptance：`17e9bb9914266691aee0c3d2b5e78b6bf64d7a00`
- P2 implementation A：`8252a57b191a2a7c4aa176a3f82cca5218119533`
- P2 manifest freeze B：`cda314ce9caeb6e9dd69ba0d08fe5254a8689fdd`
- 已核对 `B^ == A`，且 B 相对 A 只新增两份正式 manifest。
- exact-B detached worktree：
  `/guisongxia01/pangchao/wangyihan/wyh/Chang-Agent-qwen3vl-p2-b-0803-run`
- 正式结果 attempt：
  `/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl/outputs/CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001`

两份冻结 manifest 的身份为：

| 文件 | raw SHA256 | canonical SHA256 |
|---|---|---|
| `QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json` | `<hash omitted>` | `<hash omitted>` |
| `QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json` | `<hash omitted>` | `<hash omitted>` |

execution manifest 是 generator 唯一收到的 GT-free manifest。materialization receipt 明确记录
`label_file_accessed=false`。

## 3. 作业与证据 ledger

| Job | 角色 | Slurm 终态 | 资源 | 耗时 | 关键结果 |
|---:|---|---|---|---:|---|
| `53268` | manifest materialization | `COMPLETED / 0:0` | gpu48，0GPU/2CPU/16G | 00:00:24 | 两份 manifest materialize 并独立校验 |
| `53269` | exact-B contract validation | `COMPLETED / 0:0` | gpu48，0GPU/2CPU/16G | 00:01:18 | P2 targeted 37/37、registry 15/15、full 508/508（3 skipped） |
| `53272` | candidate generation | `COMPLETED / 0:0` | gpu48，1×A800/4CPU/48G | 00:19:45 | 36/36 生成、36/36 tool success，GT 未加载 |
| `53273` | offline scoring | `COMPLETED / 0:0` | gpu48，0GPU/2CPU/16G | 00:00:15 | 严格 `afterok:53272`；36/36 评分并写出停止决策 |

关键 receipt 与结果哈希：

- materialization receipt：`<hash omitted>`
- contract receipt：`<hash omitted>`
- formal submission intent / receipt：
  `<hash omitted>` /
  `<hash omitted>`
- generation receipt / complete marker / candidate index：
  `<hash omitted>` /
  `<hash omitted>` /
  `<hash omitted>`
- scoring receipt / runtime scoring manifest / score report / decision：
  `<hash omitted>` /
  `<hash omitted>` /
  `<hash omitted>` /
  `<hash omitted>`

## 4. 生成、评分与 GT 边界

固定顺序是 `test_20_15 → test_1_1 → test_50_8`。共请求并成功生成 36 个候选：27 个
planner-primary 和 9 个 diagnostic executor-only。36 个 candidate state 全部唯一，没有重复状态。

全体候选的 oracle 分类为 4 beneficial、31 harmful、1 indeterminate；primary 为
3/23/1，diagnostic executor-only 为 1/8/0。hard-gate leaf 总计为：

- harmful gate-pass 15；
- hard-gate correct reject 16；
- beneficial gate-pass 2；
- indeterminate gate-pass 1；
- hard-gate false reject 2。

Qwen 全程没有被调用。本次 oracle 回答“真实 executor 产生的候选是否改善以及是否过 gate”，不回答
Qwen 能否自主选中动作。generation marker、candidate index、36 个 record 和全部 mask 的 raw/semantic
hash 先完成验证，之后 scorer 才首次读取 label；runtime scoring manifest 对 generator 不可见。

## 5. 两个目标样本

### `test_1_1`

- 初始 IoU/F1 为 `0/0`，TP/FP/FN 为 `0/626/0`。
- primary 只有 1 个候选，没有 beneficial，也没有 beneficial + gate-pass。
- `test_1_1--primary--r0--negative_point--t2` 通过 hard gate，减少 60 FP，但 IoU/F1 仍为
  `0/0`，按预注册规则属于 indeterminate。
- executor-only diagnostic 的 positive-point 候选虽被 oracle 标为 beneficial，却不属于 planner-primary，
  且被 `point_proposal_combined_change_gate_exceeded` 拦截，不能用于通过主决策。

所以冻结主动作空间对该目标的结论是
`no_primary_beneficial_candidate_in_frozen_enumeration`。

### `test_50_8`

- 初始 IoU/F1 为 `0/0`，TP/FP/FN 为 `0/0/166`。
- 18 个 primary 中有 2 个 beneficial，其中 1 个通过 hard gate。
- 最佳 gate-pass 候选是 `test_50_8--primary--s8--box--t2`：
  ΔIoU=`+0.09965337954939342`，ΔF1=`+0.18124507486209615`，TP `+115`、FP `+988`、FN `-115`。
- 另一个 beneficial 候选 `test_50_8--primary--s8--box--t1` 被
  `component_count_delta_exceeded` 拦截；本次没有修改或放松 gate。

该目标本身满足 `primary_actionable_and_gate_pass`，但全局 gate 要求两个目标同时满足，所以仍然停止。

## 6. 为什么现在不能创建训练数据

预注册规则要求 `test_1_1` 与 `test_50_8` 各自至少有一个 primary beneficial 且通过未修改 hard gate
的候选。当前只有后者满足，故全局结果必须是 false。任何直接创建 C-ISO/T-TRAJ、训练 adapter 或继续
正式评测的动作都会绕过已冻结的停止门。

此前登记的“孤立 stage snapshots 可能不足以教授完整任务，same-episode
`audit → action → candidate → next-state` 轨迹可能更有效”仍是**未验证假设**。P2 没有构造两类数据，
没有训练模型，也没有做这项受控比较，因此既不能证明也不能否定该假设。

## 7. 解释边界

本结果严格限于冻结 CORE3 的有限、single-step、固定 proposal/action inventory。它不证明：

- 更大的动作集合、多步动作或修改后的工具空间不存在有益解；
- Qwen 能或不能发现已观察到的有益候选；
- 在完整 LEVIR-CD 分布上具有相同结果；
- C-ISO 与 T-TRAJ 哪一种训练数据更有效。

CORE3 是长期反复查看的三个 diagnostic 样本，不是无偏泛化测试。

## 8. Candidate C 与 Acceptance D 收尾状态

本结果候选要求 registry/workbook 更新后达到 `15/52/21/40`：15 个实验行、52 个字段字典行、
21 个数据行、40 个 attempt 行。P2 不产生训练数据，因此数据表必须保持 21 行；attempt ledger 只新增
正式 generation Job `53272` 和 scoring Job `53273` 两行。

机器结果 JSON 继续保持 `closeout_status=pending_exact_candidate_validation`、`attempts=[]`；这是为了
避免 candidate commit 自引用自身未来的验证结果。实际收尾状态由独立 acceptance 文件与外置 receipt
补充记录，不回写科学结果 JSON。

最终 candidate C 为 `d412759857d030b8ab766a077ea54b603585e72a`，其父提交精确为 freeze B。
零 GPU Job `53323` 从 clean detached exact C 完成独立证据重算、workbook check、12 个 P2 targeted、
16 个 registry targeted、521 个 full-regression tests（3 skipped）及 repository immutability 检查，
终态为 `COMPLETED / 0:0 / 00:01:08`。candidate receipt SHA256 为
`<hash omitted>`。

`QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json` 只绑定上述 C/Job/receipt，并明确保持科学决定不变：
`proceed_to_c_iso_and_t_traj_data_construction=false`。Acceptance D 自身不写入自己的 commit hash；其
最终 exact-D 验收状态由 Git 外部的零 GPU closeout receipt 绑定，避免形成无限自引用提交链。

收尾期间的失败 attempt 均 append-only 保留：Jobs `53315/53316` 分别发现 registry 措辞契约和测试
夹具问题，旧 candidate `f73f50bfc982d71705910f19d5571ef92870e931` 的 Job `53321` 发现结果 JSON
键名与正式 score report 不一致；修正后的 preparation Job `53322` 与最终 C Job `53323` 均通过。
