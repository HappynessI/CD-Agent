# Qwen3-VL-2B 训练前完整 Verifier 基线（0730）

本基线使用未训练的本地 `Qwen3-VL-2B-Instruct`，在固定
`DEV-EXTENDED-5-V1` 五条数据上运行完整的 Verifier-planned
Environment → Executor 闭环。它补的是训练前端到端基线，不是 32 条 select
专项前测，也不复用历史 CA_0723 指标。

## 固定口径

- 数据根目录：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256`
- 样本顺序：`test_20_15`、`test_78_13`、`test_85_16`、`test_1_1`、`test_50_8`
- 模型：`/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- Verifier：`qwen_staged`，backend=`local`，protocol=`v11.1`
- visual context：`proposal`；select 会覆盖所有 pending region，随后逐候选
  audit，并在候选动作后运行 candidate evidence/decision。
- rollout：最多 3 步，`conservative_best`，模型 greedy，seed 42。
- GT 边界：完整 rollout 写出 `rollout_complete.json` 后才读取 `label_cvt`；
  worker 也只在该 marker 存在并完成离线评分后才生成 `input_manifest.json`，固定
  A/B/GT 与模型元数据哈希。

运行参数固定在 `tools/run_qwen3vl_base_e2e5.slurm`，包括 32 个最大候选、每次
最多选择 3 个候选、0.25 crop padding、3×3 empty-mask scan、
`overlap_presence@0.25`、Verifier 1536 tokens/2 retries、SimpleClick/SAM3
路径及当前 point proposal 安全门。训练后对照必须复用相同 CLI 参数和代码版本，
唯一实验变量是 Verifier 权重状态。当前 worker 明确只接受 base 模型；重训后还需为
`GroundingModelQwen3VL` 增加并验证 LoRA adapter 加载路径，不能把 adapter 目录直接
冒充完整模型目录。

## 提交

管理节点只做路径、样本和 Slurm preflight；模型、SAM3、SimpleClick 和汇总均在
单 GPU Slurm 作业中运行：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent
bash tools/submit_qwen3vl_base_e2e5.sh
```

默认输出是 `outputs/CA_0730(4)-qwen3vl-base-e2e5`，实际闭环在 `run/`，Slurm
日志和终态 receipt 在 `logs/`。长作业无论成功或失败都会发送 QQ 通知。
本次提交固定到预先核对过的 A800 节点；提交前先确认该节点仍为 `IDLE`，例如：

```bash
NODE=gpu25 bash tools/submit_qwen3vl_base_e2e5.sh
```

## 验收与登记

成功作业必须同时满足：五个固定 sample 全部评分；没有 `.run_incomplete`；存在
GT-free rollout marker、`per_sample_metrics.json` 和 `baseline_summary.json`；汇总器
重算的 selected micro counts 与 runner aggregate 完全一致。`baseline_summary.json`
统一记录：

- initial/selected micro IoU、Precision、Recall、F1 及 Δ；
- candidate、accepted、rejected、finish、tool action、invalid attempt；
- episode outcome、无有效动作/Verifier invalid 样本数、退化样本数和工具成功率；
- 每条样本的 initial/selected 指标与 stop reason。

这些五条样本长期用于开发，是固定 diagnostic set，不是随机、盲测或无偏 test。
结果只能用于同代码、同数据、同参数下的 base/post-SFT 对照，不能宣称 LEVIR-CD
泛化性能。当前 0729 smoke 的 adapter 已删除，因此本次只能补齐训练前 E2E；训练后
E2E 必须在按标准重训且保留 adapter 后另行执行。

## 本次结果

Job `50248` 已在 `gpu25` 的单张 `NVIDIA A800-SXM4-80GB` 上完成：

- Slurm `COMPLETED 0:0`，elapsed `00:22:23`，batch MaxRSS `16,096,348 KiB`；
- 请求/评分 `5/5`，五条样本顺序与 `DEV-EXTENDED-5-V1` 完全一致；
- initial 与 verifier-selected micro 指标相同：IoU `0.6762904`、Precision
  `0.70338511`、Recall `0.94611099`、F1 `0.80688931`；
- `ΔIoU=0.0`、`ΔF1=0.0`，退化样本 `0`；
- candidates/accepted/rejected/finish/tool actions 均为 `0`；只发生 1 次 action
  attempt，且为 error；
- episode outcome 为 `verifier_invalid=4`、`no_executable_action=1`，即五条均未
  产生有效改进动作。

因此 E2E 框架 pipeline、GT-free rollout marker 和离线评分链路成功，但 base verifier
的有效闭环能力没有通过这五条诊断。该结果是训练前 base-only diagnostic；0729 adapter
已删除，训练后 E2E 仍明确记为未运行。

失败轨迹同时暴露了疑似协议/运行时基础设施问题：四条样本的 select 响应表达了区域
选择意图，但外层 JSON 少结束括号，repair 重试仍未修复；另一条样本完成 select、audit
和 decision，却把 RGB 语义枚举写成 mask 的 `black/white` 状态，随后被 fail-closed 为
`plan=null`。因此本提交只如实记录“baseline 已运行且没有形成可执行动作”，不把五条
失败全部归因于模型能力，也不在本提交中改变 parser、prompt/schema 或动作授权逻辑。
这些疑似基础设施问题将在后续独立提交中改进，并在 clean、稳定代码快照上重新运行
base/post 配对。

代码 provenance 需要分开解释：runner 在 `04:44:08Z` 创建 base manifest 时记录
Chang-Agent commit `d849c6945396f9d0041a830562d6e4642ba96943`、`git_dirty=true`、
worktree SHA256
`<hash omitted>`；作业运行中
仓库因文档提交切换到 `f4fa5e0aa5b8fb2ab4291ddf5c761bc49b03c6e5`，所以 rollout 后
生成的 `input_manifest.json` 记录的是后处理时 HEAD，不能冒充 runner 启动快照。
OmniOVCD commit 为 `46957e47805eaa273f4aa7bbbd25a88bca9108ce`，模型权重 SHA256
始终为 `<hash omitted>`。
因此本结果是有效的 base-only diagnostic，但未来严格 base/post 配对必须在同一稳定
代码快照上重跑 base，不能直接把本次结果锁定为最终 control arm。
