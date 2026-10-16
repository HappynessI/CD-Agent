# Qwen3-VL P1 retry3 H0/H1 结果（2026-08-02）

## 1. 一句话结论

P1 retry3 已完整跑通：H1 的 exactly-once terminal review 协议、隔离上下文、安全边界、来源绑定和
receipt 链都通过；但在冻结的 DEV-CORE-3-V1 三个样本上，H1 相对 H0 的 selected mask、IoU 和 F1
完全没有变化。因此本轮结论是：

- 工程结论：**通过**。第二轮终局复核能够按协议安全执行，旧的 trajectory 自环崩溃也已消失。
- 科学结论：**在 CORE3 上未观察到提升**。这不是“模型整体无效”或“LEVIR-CD 上无效”的证明。

给初学者的直观解释：H0 让模型作一次终局判断，H1 在模型准备结束时再请一个全新上下文复核一次。
这次额外复核确实运行了，但两次都只是再次说“可以结束”，没有提出可执行修改，所以最后的 mask
自然和 H0 相同，分数也不会变化。

## 2. 冻结身份与证据边界

- implementation A4：`a88404ac6a014f42692d1119d54de42c980fb310`
- final freeze B4：`5210f89f5ca13ee1e636f5aa6a44631206a80123`
- 父子关系：`B4^ == A4`
- 正式 detached worktree：
  `/guisongxia01/pangchao/wangyihan/Chang-Agent-baseline-0802-run`
- 执行时与收尾核对时，detached worktree 均为 clean、detached、exact B4；B4
  `--check-only` 已在提交前通过。
- 模型：`Qwen3-VL-2B-Instruct`
- 固定样本顺序：`test_20_15 → test_1_1 → test_50_8`
- 分析主证据：
  `/guisongxia01/pangchao/wangyihan/wyh/diagnostics/qwen3vl_p1_retry3_result_0802/trajectory_analysis.json`
- 分析主证据 SHA256：
  `<hash omitted>`

retry1、retry2、retry3 的 outputs、receipts、失败日志和 status 都是 append-only 历史证据。本报告没有
修改或补写任何历史产物，也没有跨 attempt 拼接 H0/H1 pair。

## 3. 正式作业与独立分析 ledger

| Job | 角色 | Slurm 终态 | 耗时 | 结果 |
|---:|---|---|---:|---|
| `53155` | formal CPU fixtures | `COMPLETED / 0:0` | 00:00:25 | 300 targeted、470 full、3 skipped；receipt complete |
| `53156` | smoke H0 | `COMPLETED / 0:0` | 00:01:16 | summary、evaluation receipt 有效 |
| `53157` | smoke H1 | `COMPLETED / 0:0` | 00:01:58 | summary、evaluation receipt、smoke pair 有效 |
| `53158` | CORE3 H0 | `COMPLETED / 0:0` | 00:04:18 | 3/3 样本完成，receipt 有效 |
| `53159` | CORE3 H1 | `COMPLETED / 0:0` | 00:05:59 | 3/3 样本完成，receipt 与 CORE pair 有效 |
| `53160` | dependency guard | `COMPLETED / 0:0` | 00:00:02 | `synthesized=none`，`unresolved=none` |
| `53164` | trajectory analysis attempt 1 | `FAILED / 1:0` | 00:00:02 | 分析器状态字面量预期错误；无科学结果 |
| `53165` | trajectory analysis attempt 2 | `COMPLETED / 0:0` | 00:00:01 | 3 样本分析完成并写出 hash-bound JSON |
| `53171` | registry workbook preparation | `COMPLETED / 0:0` | 00:00:03 | XLSX 重建为 14/52/21/38；15 个 registry targeted tests 通过 |
| `53174` | closeout candidate validation 1 | `FAILED / 1:0` | 00:00:15 | result/workbook/targeted 均通过；全量回归发现 P0 历史 workbook hash 测试边界错误 |
| `53178` | closeout candidate validation 2 | `COMPLETED / 0:0` | 00:00:14 | workbook/15 targeted/471 full（3 skipped）/frozen/result 全通过 |

`53164` 的失败只发生在独立分析器：它错误要求 frozen pair validator 文本以
`status=complete;` 开头，而真实且合法的 validator 合同是 `status=valid;`。修正分析器对既有合同
的读取后，`53165` 从 clean B4 仓库重新执行并通过。没有改写 frozen pair receipt，也没有把
`53164` 的部分结果与 `53165` 拼接；两次 status 与日志都保留。

`53174` 发现的是维护测试边界问题：P0 amendment 中的 workbook SHA256 是 P0 当时的历史快照，旧
测试却把它与会持续 append-only 更新的当前 workbook 文件直接比较，等价于禁止 registry 增加任何
后续实验。修复只把 P0 测试改为验证历史 hash 字面量，并明确要求它不同于当前 P1 workbook；P0
amendment、acceptance receipt 和旧证据均未修改。当前 workbook 由 Job `53171` 重建，SHA256 为
`<hash omitted>`。
修复后的 clean parent commit `d64ec522de1d46df27891cea1f9f9ab0fa1b7be0` 由 Job `53178` 验证
通过；其 closeout receipt SHA256 为
`<hash omitted>`。

## 4. 工程和协议门禁

独立分析确认：

- 18 项 controlled-variable checks 全通过；
- 25 项 CORE pair hard gates 全通过；
- H0、H1 两个 evaluation receipt validator 均通过；
- CORE pair validator 通过；
- `cross_attempt_pair_composition=false`；
- H0 round two 调用数为 0，H1 为 2；
- H0/H1 的 `final_region_protocol_valid` 计数均为 2；
- 两臂都是 3 次 tool action 全部执行成功、3 个 candidate 全部被安全拒绝、0 个 candidate accepted；
- H1 没有 scheduler 越权执行、没有复制第一轮 raw response、没有 contract violation。

此外，`53159` 成功保存了 `test_20_15` 的四步 trajectory，明确越过 retry2 Job `53145` 的旧
`Trajectory.save()` 自环崩溃点。修复仍是最小边界：

```python
final_evidence = copy.deepcopy(round_one_evidence)
```

它避免外层 evidence 与 nested round-one evidence 共用同一个 dict；本轮没有用 cycle marker
掩盖坏证据。

## 5. CORE3 H0/H1 对比

| selected 指标 | H0 | H1 | H1 − H0 |
|---|---:|---:|---:|
| micro IoU | 0.74223120 | 0.74223120 | 0.00000000 |
| micro precision | 0.79404920 | 0.79404920 | 0.00000000 |
| micro recall | 0.91918414 | 0.91918414 | 0.00000000 |
| micro F1 | 0.85204673 | 0.85204673 | 0.00000000 |
| macro IoU | 0.28212851 | 0.28212851 | 0.00000000 |
| macro precision | 0.29542362 | 0.29542362 | 0.00000000 |
| macro recall | 0.64983767 | 0.64983767 | 0.00000000 |
| macro F1 | 0.30560087 | 0.30560087 | 0.00000000 |

两臂每个样本的 selected state 与离线指标精确相同，汇总动作统计也相同。H1 的额外调用增加了审计
证据，但没有产生 safe plan、tool action 或新 candidate，因此没有改变 selected mask。

## 6. 三个样本的失败轨迹

### 6.1 `test_20_15`：安全拒绝了三次有害修改，但没有增益

- selected IoU/F1：`0.84638554 / 0.91680261`
- 像素误差：613 FP、254 FN
- 三次 tool action 都成功生成 candidate，但三个 candidate 均被 verifier 拒绝；
- episode 最终因 `max_steps_reached` / budget exhausted 结束；
- 该过程不是 terminal trigger，H1 正确走 nonterminal passthrough，没有调用 round two。

这里协议行为是正确的：它阻止了有害 candidate 覆盖当前较好结果。不过 H1 的能力只在终局 trigger
出现时生效，因此无法改善这个“预算耗尽且候选都更差”的样本。

### 6.2 `test_1_1`：无变化样本上的完整 false positive

- GT 正像素为 0，selected mask 却有 626 个正像素；
- H0 直接 finish；
- H1 在 fresh context 中执行一次 finish confirmation，但仍把区域判断为
  `background → building`，再次确认 finish；
- round two 没有返回 safe plan，scheduler 没有执行工具，mask 保持不变。

这是“复核存在但复核判断也错了”。fresh context 能防止隐藏状态泄漏，却不会自动给模型增加识别
能力。

### 6.3 `test_50_8`：有变化样本上的空 mask false negative

- GT 有 166 个正像素，selected mask 正像素为 0；
- H0 直接 finish；
- H1 在 fresh context 中复核多个区域，仍把它们解释为背景或非建筑变化，并确认 finish；
- 同样没有 safe plan、tool action 或状态变化。

这个样本说明第二轮“再看一遍”不等于“知道如何纠错”。如果复核模型仍缺少从审计结论走向可执行
动作、再判断 candidate 后果的能力，它只能更完整地重复原错误。

## 7. 为什么多两次 review 仍然零增益

H1 的两次额外调用都属于 `finish_confirmation`，分别发生在 `test_1_1` 和
`test_50_8`。它们都满足协议要求：fresh backend context、exactly once、无 raw-response 继承、
无越权工具执行；但输出仍是 finish。由于没有提出可执行计划，系统没有状态转移，最终 mask 与 H0
逐像素相同。

所以本轮区分出两个层面：

1. terminal-review **机制**是否能安全、可审计地运行——答案是能；
2. 当前模型是否具备利用额外复核纠正 false finish 的**能力**——CORE3 上没有观察到。

## 8. 完整流程训练数据假设

此前提出的数据限制已正式保留：当前训练数据以彼此孤立的 stage snapshots 为主，可能教会模型填写
局部 schema 或作单阶段判断，却没有充分教授同一 episode 中机械关联的
`audit → action → candidate → next-state` 后果链。这与本轮“能再次 audit，但不能产生纠错动作”的
轨迹一致。

但它目前只是**与观察一致、尚未证明因果的假设**，不能写成唯一根因。CORE3 只有三个反复使用的
诊断样本，也没有进行训练数据对照。下一轮应比较：

- control：孤立 stage snapshots；
- treatment：同 episode、可追溯的完整流程轨迹；
- 严格匹配：来源 split、泄漏策略、token budget、optimizer steps、base checkpoint 和评测协议。

只有完成这个受控训练实验，才能判断完整流程数据是否真的改善 false-finish recovery。

## 9. 解释边界与下一阶段

本结果只支持“在冻结 DEV-CORE-3-V1 上没有观察到 H1 指标提升”。它不证明 LEVIR-CD 泛化表现，
也不能据此直接进入 EXTENDED5 或 locked final。

当前 P1 的 registry/workbook 更新、零 GPU 收尾验证和独立 result commit 已完成；用户授权的独立
版本化迭代现已生效：

```text
训练数据创建 → 训练 → 评测 → 失败轨迹分析 → 下一轮改进
```

新阶段必须重新建立 preregistration、data manifest/hash、来源与泄漏审计、Slurm Job 和资源记录、
checkpoint lineage、evaluation receipt、失败轨迹 taxonomy、registry/workbook 以及独立 Git commit。
P1 冻结 JSON 与历史 outputs 不作为新阶段可改写的草稿。

## 10. 关键只读产物

- formal CPU：
  `outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry3`
- smoke H0/H1：
  `outputs/CA_0802(3)-qwen3vl-h0-base-smoke1-retry3`、
  `outputs/CA_0802(4)-qwen3vl-h1-base-smoke1-retry3`
- CORE3 H0/H1：
  `outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry3`、
  `outputs/CA_0802(6)-qwen3vl-h1-base-core3-retry3`
- dependency guard：
  `outputs/CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry3/dependency_guard_receipt.txt`
- 独立逐轨迹分析：
  `/guisongxia01/pangchao/wangyihan/wyh/diagnostics/qwen3vl_p1_retry3_result_0802/`
