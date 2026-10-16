# Qwen3-VL P0 validation amendment（0802）

- 状态：**冻结；只授权 P0 0-GPU corrective retry；P1 仍未授权**
- 日期：2026-08-02（UTC）
- 机器可读合同：`QWEN3VL_P0_VALIDATION_AMENDMENT_0802.json`

## 1. 为什么必须 amendment

首次 P0 CPU validation 使用 commit
`383521f72e846b459697dc24208b8245243975ec`，Jobs `52553/52554` 均未申请
GPU，但在 2–3 秒内于 worker preflight 失败，三组 targeted unittest、workbook check、
archive summary integration 和全量 regression 都没有开始。

根因不是代码 checkout 不干净，也不是 worker 字节漂移。`sbatch` 会把脚本复制到 Slurm
controller/compute-node spool 后执行，因此运行时 `BASH_SOURCE[0]` 合法地指向 spool copy，
不能要求该路径位于 Git worktree。原检查把“文件路径身份”误当成“文件字节身份”。

首次输出
`outputs/CA_0802(1)-qwen3vl-p0-contract-validation` 必须完整保留，不删除、不覆盖、
不作为 pass receipt；其 submission/result/status/stdout/stderr SHA256 已冻结在机器可读
amendment 中。

## 2. 唯一实现修正

retry worker 同时计算并校验：

1. Slurm 实际执行的 spool copy SHA256；
2. exact clean run worktree 内 pinned worker SHA256；
3. submitter 在提交前记录的 expected worker SHA256。

三者的两个实际 digest 都必须等于 expected digest；路径是否相同不再作为字节身份条件。
exact commit、空 Git status、ignored `outputs` symlink realpath、0 GPU、输入 archive、
summary/registry/workbook 合同均不变。

## 3. 新的唯一预留

- 输出：`CA_0802(1R1)-qwen3vl-p0-contract-validation`
- targeted Job：`CA0802P0TargetR1`
- regression Job：`CA0802P0RegressR1`
- dependency：`afterany:<targeted_job_id>`
- stdout/stderr：新输出自身 `logs/slurm-%j.out/.err`
- QQ title：`CA0802 P0 contract validation R1`
- 资源：两个 Job 均 0 GPU、2 CPU、8 GiB；时限分别 20/45 分钟

submitter 和 worker 必须先验证首次失败证据 hash 未变化；新目录提交前必须不存在。若
`1R1` 再发生冲突或合同变化，继续停止并另立 amendment，不能删除旧输出或现场换号。

## 4. 未改变的边界

- H0/H1 treatment、GT-free 语义、metric/candidate 合同没有变化；
- `CA_0802(2)`–`CA_0802(6)` 的 P1 预留不变；
- H1 仍未实现，P1/GPU 仍未授权；
- 最终 P0 acceptance receipt 必须同时引用首次 failed attempt 和本次 retry，不能隐去失败轨迹。
