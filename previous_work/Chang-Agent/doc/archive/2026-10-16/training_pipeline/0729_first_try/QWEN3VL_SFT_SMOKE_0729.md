# Qwen3-VL-2B SFT smoke（0729）

本链路用固定版本 LLaMA-Factory 验证 32 条多模态样本从预处理、LoRA SFT、同集评估到生成预测的完整路径。它是 overfit smoke，不是独立验证集指标。

## 固定组件

- 基座模型：`/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- LLaMA-Factory commit：`9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d`
- 独立包 overlay：`/guisongxia01/pangchao/wangyihan/conda_envs/qwen3vl-sft-env`
- 训练配置：`configs/qwen3vl_sft_smoke_0729.yaml`
- 首次尝试 run root：`outputs/CA_0729(1)-qwen3vl-sft32-smoke`
- 启动修复后的重试 run root：`outputs/CA_0729(2)-qwen3vl-sft32-smoke`

核心依赖固定为 NumPy 1.26.4、SciPy 1.17.1、Transformers 4.57.6、Datasets 4.0.0、Accelerate 1.11.0、PEFT 0.18.1、TRL 0.24.0、TorchData 0.11.0 和 qwen-vl-utils 0.0.14。NumPy 固定在 1.x ABI，是因为 LLaMA-Factory 源包未声明上界，而同一工作区的 SAM3/Change-Agent 明确要求 `numpy<2`；SciPy 同步固定为与该 ABI 兼容的 1.17.1。环境使用 `venv --system-site-packages` 只读复用 `omniovcd-env` 的 Torch/CUDA 栈，所有新增/升级包写入 overlay，不修改 base；安装脚本还会安装并导入校验 torchaudio、matplotlib、fire、tyro 及 LLaMA-Factory 的声明依赖。该方案的边界是 Torch/Python 运行时仍依赖 base 环境路径，因此版本清单会同时记录 overlay 和 base。

## 执行顺序

1. 在仓库根目录运行 `REPO_ROOT="$PWD" bash tools/submit_qwen3vl_sft_prepare.sh`。它提交 CPU prepare 作业，在 Slurm 计算节点创建/校验独立环境、运行 data/scorer 单测并构造 32 条数据；不要在管理节点直接运行 setup、测试或预处理。
2. 监控 prepare Job ID，等待其成功结束，并明确检查正式 run root 中出现 `.prepare_complete`。只有 Slurm 状态为 `COMPLETED` 且该 marker 存在时才能继续；`.prepare_submitted` 只表示提交成功。
3. 运行 `REPO_ROOT="$PWD" bash tools/submit_qwen3vl_sft_smoke.sh`。launcher 会检查 `.prepare_complete`、Slurm 控制面、重复作业、固定 commit、环境、模型及数据后，申请 gre 分区单 GPU。
4. `tools/run_qwen3vl_sft_smoke.slurm` 使用 `levir_qwen3vl_sft32_train` 训练，并用 `levir_qwen3vl_sft32_eval` 做同 32 条数据的 eval/predict；结束后由独立 `tools/score_qwen3vl_sft_smoke.py` 在 adapter 被清理前完成协议验收。

### 2026-07-29 首次 GPU 尝试与重试约束

Job `49550` 已在 `gpu16` 获得单张 A800 80GB 并通过 32 行数据、固定依赖、
LLaMA-Factory import、单卡和 BF16 preflight，但在训练开始前失败。失败点是 worker
设置了 `FORCE_TORCHRUN=1`，固定 launcher 随后按 `PATH` 查找 `torchrun`，而提交环境
没有暴露该入口，最终报 `FileNotFoundError: torchrun`。这次失败未产生 adapter、预测
或 smoke metrics；原始 receipt、Slurm 日志和 `status-49550.txt` 保留在首次 run 中。

修复后的单卡 worker 显式清除继承的 torch distributed launch 变量，使用固定
`$TRAIN_ENV/bin/python -m llamafactory.cli train` 的原生单进程分支。这样不会通过
另一个 console script 绕过 package overlay；未来只有多卡或 DeepSpeed 路径才应恢复
显式 torchrun，并且必须用同一 overlay Python 启动。

首次 run 已有 `.training_submitted` 和失败审计，不能删除 marker 或原地覆盖。由于
ShareGPT 中的图像是指向 run 内 `data/images/` 的绝对路径，重试先通过 CPU prepare
在新的干净 run 中确定性重建同一 32 条数据，再走标准 GPU submitter：

```bash
RUN_DIR="$PWD/outputs/CA_0729(2)-qwen3vl-sft32-smoke" \
JOB_NAME=CA29Q3VLPREP32R1 \
REPO_ROOT="$PWD" bash tools/submit_qwen3vl_sft_prepare.sh

# 仅当新的 prepare Job 为 COMPLETED 且 .prepare_complete 存在后执行：
RUN_DIR="$PWD/outputs/CA_0729(2)-qwen3vl-sft32-smoke" \
JOB_NAME=CA29Q3VLSFT32R1 \
REPO_ROOT="$PWD" bash tools/submit_qwen3vl_sft_smoke.sh
```

### 实际完成结果

| Job | 阶段 | 状态 | 节点/资源 | 耗时 | 关键结果 |
|---|---|---|---|---:|---|
| 50128 | clean retry prepare | `COMPLETED (0:0)` | gpu14；0 GPU、2 CPU、16 GiB | 6m12s | 32 条，16 FN + 16 FP，32 parent，`valid=true` |
| 50129 | LoRA train + eval + predict + score | `COMPLETED (0:0)` | gpu14；1×A800 80GB、4 CPU、32 GiB | 11m25s | `pipeline=success`，`acceptance=passed` |

Job `50129` 完成 32 个 optimizer steps（有效 4 个 epoch），`train_loss=1.07037`，
最后一个 step loss 为 `0.0732`。同 32 条数据上的 eval/predict 指标一致：BLEU-4
`90.3757`、ROUGE-1 `93.7041`、ROUGE-2 `92.8933`、ROUGE-L `95.6694`。
`generated_predictions.jsonl` 恰好 32 行，严格 JSON 和精确 schema 均为 32/32，
unknown region 为 0，`region_id` 为 32/32，字符串 exact-match 为 21/32。协议硬门
全部通过；reason 文本的非精确匹配不影响协议验收。

Slurm batch MaxRSS 为 6,221,784 KiB；job-local adapter 确实生成并在独立 scorer
完成后清理，共享 run 未保留 adapter 或 checkpoint。`eval_loss`/`predict_loss` 按当前
generation 路径为空，符合下文边界。评测时 jieba 尝试替换共享 `/tmp/jieba.cache`
产生一次非致命 `PermissionError` 日志，随后在内存中成功构建词典并完成全部
eval/predict/scorer；最终 `pipeline_errors=[]`。

## 队列监控与邮件通知

`tools/monitor_qwen3vl_sft_job.sh` 用于在管理节点做轻量 Slurm 监控，默认每 120 秒查询一次 `guisongxia` 的 24-GPU association 配额。脚本内默认的 Job `49550` 只对应首次失败尝试；重试必须同时显式传入新的 `--job-id` 和 `--run-dir`。它不会登录计算节点或读取其他作业的 GPU 利用率，只依据 Slurm 的正式分配状态判断。

通知采用账号专属 `wangyihan-send-qq`，不读取或打印 SMTP 凭据，并按状态转移去重：

- pending 期间从 0 张空闲变为至少 1 张空闲时发送一次；配额再次占满后，下一次释放可再次通知；
- Job 进入 `CONFIGURING` 或 `RUNNING` 时发送一次启动通知；
- Job 进入终态时发送一次终态通知并停止监控；
- 查询暂时失败时只写日志，不改变通知状态。

状态、锁和日志都写入本次 run 的 `logs/`：

```text
gpu-monitor-<job-id>.state
gpu-monitor-<job-id>.lock
gpu-monitor-<job-id>.log
```

单次无邮件诊断可运行：

```bash
bash tools/monitor_qwen3vl_sft_job.sh --once --dry-run
```

常驻监控应放在独立 tmux 会话中；脚本使用 `flock`，重复启动同一 Job 的 monitor 会直接拒绝，不会产生重复邮件。

训练 adapter 和中间 checkpoint 只写到计算节点的 `SLURM_TMPDIR` 或 `/tmp`，eval/predict 结束后删除。共享 run root 只保留日志、配置、版本、`train/eval/predict_results.json`、`generated_predictions.jsonl`、`smoke_metrics.json` 和 `completion_summary.json`。

配置中的 `gradient_accumulation_steps=4`、`max_steps=32` 是 32 条同集 overfit smoke 的特配：单卡 batch size 1 时相当于约 4 个数据遍历（128 个样本步），且 `max_steps` 会覆盖 `num_train_epochs`。它用于放大学习信号、提高打通链路时的可观测性，不等同于 `doc/baseline_0729.md` 中面向正式数据规模的概念 baseline；扩展到完整 train/held-out val 时应按有效 batch size 和 epoch 重新定步数，不能照搬 32 steps。

## 验收与边界

- 数据必须正好 32 条，两个 dataset key 必须指向同一份 `sharegpt.json`，每条 `<image>` 数量与图像路径严格匹配。
- worker 要看到且只能看到一张支持 BF16 的 GPU；训练结束前必须实际产生 job-local `adapter_model.safetensors`。
- scorer 要求 `generated_predictions.jsonl` 的原始非空行数、可解析 JSON 行数和 object 行数都恰好为 32；不会去除 Markdown fence，也不会从说明文字中抽取 JSON。所有比率固定以 32 为分母，损坏或缺失行不会被跳过后缩小分母。
- `smoke_metrics.json` 报告 loss/ROUGE/BLEU、严格 JSON、精确 response schema、unknown region、`region_id` 准确率和字符串 exact-match，并把 `pipeline_status` 与 `acceptance_status` 分开。协议硬门是 `strict_json_count=32`、`schema_valid_count=32`、`unknown_region_id_count=0`；`region_id_accuracy` 和 exact-match 是质量指标，不是链路硬门。
- scorer 退出码 `0` 表示链路完整且协议门通过，`3` 表示预测/metrics 文件或外层 32 行链路不完整，`4` 表示 32 个外层 object 完整但响应协议门失败。三种路径都会先写 `smoke_metrics.json`、`smoke_manifest.json` 和 `completion_summary.json`；后两种会让 Slurm 作业失败，但指标和失败原因仍保留在共享 run root，并进入 QQ 的 `key_result`。
- 同集结果只证明链路和短程可学习性，不能代表 LEVIR-CD val/test 泛化。
- 当前 `predict_with_generate=true` 路径会为生成评测移除 labels，因此 `eval_loss`/`predict_loss` 可能为空；应以 `train_loss`、ROUGE/BLEU 和上述协议指标验收 smoke。若要比较 held-out loss，需另跑不启用 generation 的 val 作业。
- 当前 32 条数据中，错误候选是 `missing/present`，三个干扰候选是 `scan`。这是为了稳定验证协议链路的明显 shortcut；高 `region_id` 准确率不能证明模型真正学会了遥感视觉判别。
- 固定 commit 的 `llamafactory-cli eval` 不可用。本链路使用 `llamafactory-cli train` 的 `do_eval + do_predict + predict_with_generate`，并在 adapter 被清理前完成评估。
