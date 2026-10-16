# Qwen3-VL P2 动作可达性预注册

- 日期：2026-08-03（UTC）
- 状态：**设计和执行链正在冻结，P2 尚未运行，没有 P2 结果或 Job ID**
- 上游状态张量：P1 H0 retry3 Job `53158` 的三个 CORE3 cached initialization masks；H1 retry3
  Job `53159` 与冻结 H0/H1 pair receipt 负责绑定 false-finish 来源，并证明 H1 与这些 H0 masks 的
  selected step 和离线指标逐样本一致
- 目的：构造新训练数据之前，先确认当前框架里确实存在能把错误状态改好的动作

## 1. 这一步用通俗的话在做什么

P1 表明模型会再次检查，但仍可能在错误状态上直接结束。现在不能马上生成训练数据，因为还不知道
“模型即使选对动作，现有工具和安全门能否真正把结果改好”。如果动作空间里根本没有可用纠错动作，
训练模型去选择这些动作也没有意义。

P2 因而像一次有限的“穷举试按按钮”：从每个完全相同的 P1 初始状态出发，每次只按一个预先登记的
按钮，保存候选 mask，再在全部候选生成完毕并通过哈希验证以后，才由离线评分器读取 GT 判断候选变好、
变坏还是无法判定。这个 oracle 不调用 Qwen，也不评价 Qwen 能否自己选中动作。

## 2. 冻结的检查范围

样本顺序固定为：

1. `test_20_15`
2. `test_1_1`
3. `test_50_8`

随机种子固定为 `42`。每个候选都是 single step，而且从同一份 P1 H0 Job `53158` cached
selected-step-zero T1/T2 mask 独立 reset；前一个候选不能影响下一个候选。H0 rollout/evaluation
绑定实际读取的 mask；H1 evaluation、H1 rollout marker 与 H0/H1 pair receipt 则提供 false-finish
provenance，并复核两个独立 Job/arm、controlled-variable checks、hard gates、summary bindings、相同
offline-metrics payload，以及三个样本 selected step/metrics 相等。H1 provenance 不会把实际 reset
来源改写成 H1 文件。

proposal 参数固定为：

- `max_regions=1`
- `max_delta_regions=1`
- `min_component_area=4`
- `padding_ratio=0.25`
- `empty_mask_scan_grid_size=3`

共登记 36 个 action specs：27 个 planner-primary 与 9 个 diagnostic executor-only。27 个 primary 包含
`test_20_15` 的 8 个 T2 negative point、`test_1_1` 的 1 个 T2 negative point，以及空双 mask
`test_50_8` 上 9 个扫描框分别作用于 T1/T2 的 18 个 box。另有 9 个 positive-point diagnostic，
仅用于观察 executor 和 gate，不参与本次主决策。这里的“executor 可执行”不等于“当前 planner 可达”：
present proposal 在 `EvidenceRecord` 中是 white region；atomic audit 与独立 diagnosis validator 都禁止
white region 产生 `false_negative`，而 present-region positive point 的 planner 分支只可能来自
`false_negative`。因此这 9 项保留为 executor-only diagnostic 是有意的合同，不是漏掉 planner 动作。

这个有限枚举不能证明更大的动作空间里绝对没有有益动作；负结论只能表述为“在本次冻结的单步集合中
没有观察到”。

## 3. GT 隔离

仓库中最终冻结两份 JSON：

- preregistration/scoring contract：包含离线 label 的 path 与已冻结 SHA256；
- execution manifest：它是 preregistration 的严格 GT-free 投影，也是 generator 唯一收到的 manifest。

builder 不打开、不 `stat`、不重新 hash label 文件，只复制已经由 P1 冻结的 label identity。generator
不收到 label、input root、preregistration 或 runtime scoring manifest。它必须先写完 generation marker、
candidate index、36 个 candidate records 与所有 T1/T2/change masks。

scoring worker 随后先验证 marker/index、execution binding、freeze-B runtime commit、每个 action spec、
record/raw file hash、三份语义 mask hash、hard-gate 和 tool-success evidence。只有全部验证成功，validator
才首次打开 label，并一次性创建 runtime scoring manifest。

## 4. A/B 冻结和运行步骤

1. implementation commit A 只包含 builder、独立 validator、generator、scorer、四个 Slurm workers、
   两个 submitter、release helper、tests 和文档；A 不包含两份正式 JSON。
2. 从 clean detached exact A 提交零 GPU materialization 作业。作业在 A worktree 中执行代码，但只把
   两份 manifest 写入 clean、同为 exact A 的开发分支 worktree 的目标文档目录；receipt/status/logs 写到
   仓库外 diagnostics，不能污染即将形成的 B。
3. 人工核对 materialization receipt 后，以“只新增这两份 manifest”形成 freeze commit B；必须满足
   `B^ == A`。manifest 只记录 A，避免自引用尚不存在的 B；B 由后续外部 receipts 绑定。
4. 同步一个新的 clean detached exact B run worktree。既有 P1 B4 worktree保持不动。
5. 默认运行 `submit_qwen3vl_p2_oracle.sh` 时只能提交零 GPU contract validation。该作业验证 A/B 父子、
   B 只含两份文件、manifest、targeted/full tests、registry tests、shell syntax、P1 runtime 未变化以及
   repository immutability。
6. 只有 contract Job 明确为 `COMPLETED / 0:0` 且 receipt 全部通过，才能显式使用
   `--formal CONTRACT_JOB_ID CONTRACT_RECEIPT` 提交 generation 和它的 `afterok` scoring。

manifest materialization 必须使用专用 submitter；它会验证 exact detached clean A、clean 目标分支、
Slurm controller/固定节点/重复作业，并以 `--export=NIL` 加位置参数提交，避免继承凭据、代理或
`PYTHONOPTIMIZE`。每次提交自动保留独立、append-only 的 `attempt-NNNN`：

```bash
cd "$EXACT_A_ROOT"
EXPECTED_COMMIT="$A" TARGET_REPO_ROOT="$TARGET_REPO_ROOT" \
  bash tools/submit_qwen3vl_p2_manifest_build.sh
```

正式输出 base 预留为：

`Chang-Agent-local-qwen3vl/outputs/CA_0803(1)-qwen3vl-p2-oracle-actionability`

每次 formal 提交在该 base 下原子预留新的 `attempt-NNNN`，generation、afterok scoring、日志和
receipts 只能使用同一个 attempt；失败 attempt 永不覆盖，也不得与后续 attempt 拼接。首次提交前
base 可以不存在；已有 base 只能继续追加新的 attempt。P2 的任何 receipt 或结果都不得预写成已经完成。

## 5. 评分和停止门

沿用 P0 定义及 `epsilon=1e-12`：IoU 与 F1 都严格提高才是 beneficial；任一下降，或两指标持平但
TP 下降、FP 上升或 FN 上升，则是 harmful；其余是 indeterminate。比较发生在舍入前。

本次主决策只看 primary。只有 `test_1_1` 和 `test_50_8` **各自**至少存在一个 primary beneficial
候选，而且该候选通过了未修改的 pre-Qwen hard gate，才允许进入 C-ISO 孤立阶段数据与 T-TRAJ
完整轨迹数据的构造、匹配训练、评测和失败轨迹分析。

如果 beneficial 候选被 hard gate 拦截，只报告候选和原因，不在本次 P2 内修改或放松 gate。任一目标
没有满足停止门，就停止训练数据构造并复核冻结动作空间。P2 通过也只授权下一步“数据构造设计”，不等于
训练、EXTENDED5 或 locked final 已经自动获批。

## 6. 当前待办

- 形成 implementation A；
- 在 Slurm 上从 clean detached exact A materialize 两份正式 JSON；
- 形成只含两份 JSON 的 B；
- 在新的 detached exact B worktree 跑零 GPU contract validation；
- 验收全通过后才运行 P2 generation/scoring；
- 根据 P2 gate 决定是否构造 C-ISO/T-TRAJ。

因此截至本文冻结时，P2 **尚未运行**，不得填写结果、Job ID、registry outcome 或任何训练结论。
