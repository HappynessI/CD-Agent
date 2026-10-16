# Qwen3-VL P0 validation R2 amendment（0802）

- 状态：**冻结；只授权 P0 的唯一 0-GPU R2；P1/GPU 仍未授权**
- 日期：2026-08-02（UTC）
- 机器可读合同：`QWEN3VL_P0_VALIDATION_AMENDMENT_R2_0802.json`
- 机器可读合同 canonical SHA256：
  `<hash omitted>`

## 1. R1 得到了什么、没有得到什么

R1 Jobs `52562/52563` 均在 commit
`28786e82a08fa9e755cc3bd2fd01885df333316c` 的 clean detached worktree 上运行，worker
实际/预期 SHA256 均为
`<hash omitted>`；每个 Job
申请 2 CPU、8 GiB、0 GPU。spool-copy 身份修复有效，测试确实开始。

- targeted：P0 contract `18/18`、registry `14/14` 通过；local verifier 共 48 项，
  `45 passed + 3 errors`。
- full regression：358 项，`352 passed + 3 errors + 3 skipped`。
- 两边相同的三个 error 都发生在 baseline summarizer 读取 rollout marker 时。
- targeted 因此没有到达 shell syntax、workbook `--check`、真实 v1 summary 生成和独立
  false-accept assertion；`v1_post_sft_summary_v2.json` 不存在，不能把 R1 写成这些检查通过。
- v1 archive 八个输入文件的 before/after manifest 逐字节相同，只能陈述
  `archive hash window passed`；文件系统并未提供永久只读保证，不能写成“archive 永久 immutable”。

R1 的 11 个 submission/result/status/stdout/stderr/hash-manifest 文件 SHA256 已完整冻结在
机器可读 R2 amendment 中。首次失败 `(1)` 的 9 个文件也继续保留并再次引用；两个旧目录均
不得删除、覆盖或复用。

## 2. 根因与精确合同

生产 writer `tools/run_levir_change_agent.py` 的 legacy 语义是：

1. `rollout_complete.json.samples` 是 rollout 完成成员列表，固定写成
   `sorted(requested_samples)`；
2. `per_sample_metrics.json.samples` 的 key 顺序才是 requested/scored 顺序；
3. protocol/baseline summary 的 `sample_order` 与 `per_sample` 也保持 requested 顺序。

真实 v1 archive 因而合法地保存：marker 为
`test_1_1, test_20_15, test_50_8`，metrics 为
`test_20_15, test_1_1, test_50_8`。R1 summarizer 错把两者要求成同一种顺序。

R2 不退回 set comparison。analyzer、summarizer、comparator 必须分别对 legacy marker 做
`rollout_samples == sorted(expected_samples)` 的精确 list 校验；metrics 和 summary 仍精确
保持 requested order。任何重排、重复、遗漏或未知 sample 都 fail closed。未来若在 H1
marker 中新增显式 requested-order 字段，必须在 P1 另立 schema/实现，本次 P0 不偷跑。

## 3. 唯一 R2 预留

- 输出：`CA_0802(1R2)-qwen3vl-p0-contract-validation`
- targeted Job：`CA0802P0TargetR2`
- regression Job：`CA0802P0RegressR2`
- dependency：`afterany:<targeted_job_id>`
- QQ title：`CA0802 P0 contract validation R2`
- 资源：两个 Job 都是 0 GPU、2 CPU、8 GiB；时限分别 20/45 分钟

submitter 和 worker 必须逐文件复核 `(1)` 与 `(1R1)` 的冻结 SHA256，校验 exact commit、
clean worktree、worker spool/repo 双重字节身份和固定 outputs symlink。R2 目录提交前必须不
存在；若仍有冲突或合同变化，停止并另立 amendment，不删除旧现场或临时换号。

## 4. 验收与边界

只有 targeted、完整 regression、workbook checker、真实 v1 summary/assertion 和 archive
hash window 全部通过，才能另建
`QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json`。receipt 必须同时引用首次失败、R1 失败和 R2
通过的 commit、Job、日志与 SHA256；不能回填原 preregistration 或 R1 amendment。
机器可读 amendment 已冻结 `qwen3vl_p0_acceptance_receipt_v1` 的必需字段、16 个 gate ID、
三次 attempt/job 字段和 complete-if 规则；两个 R2 Job 必须同时为 `COMPLETED/0:0`，全部
路径与 raw hash 可复算，且 `p1_execution_authorized=false`。QQ 只允许声称 worker 已尝试
发送；没有外部 delivery receipt 时不得声称收件人已收到。

H0/H1 设计、唯一 treatment、P1 的 `(2)`–`(6)` 预留均未改变。H1 仍未实现；P1、CPU
fixture 和任何 GPU 作业仍未授权。
