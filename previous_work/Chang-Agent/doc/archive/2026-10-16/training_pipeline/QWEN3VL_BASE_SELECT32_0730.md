# Qwen3-VL-2B base select32 前测（0730）

## 状态与目的

本路径已由 Job `50295` 成功执行并通过产物审计。它只加载未经本项目 SFT 的
`Qwen3-VL-2B-Instruct` base 模型，对成功 0729 retry 使用的同一批 32 条
select-stage 数据执行一次 `do_predict`，补齐训练前功能前测。

这不是训练任务，不产生 adapter、checkpoint、optimizer 或 trainer state；也不是
held-out、E2E 或视觉泛化评测。它只能与既有 post-SFT same32 结果组成同集功能对照。

launcher 默认输出目录是 `(3)`；本次成功 clean run 使用独立目录：

```text
/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CA_0730(6)-qwen3vl-base-select32
```

## 固定输入与实现文件

- base 模型：`/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- 数据：`outputs/CA_0729(2)-qwen3vl-sft32-smoke/data`
- eval key：`levir_qwen3vl_sft32_eval`
- reference train key：`levir_qwen3vl_sft32_train`，仅用于声明与 post-SFT
  对照的数据身份，本作业不加载训练集分支也不训练
- LLaMA-Factory commit：`9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d`
- 配置：`configs/qwen3vl_base_select32_0730.yaml`
- GPU worker：`tools/run_qwen3vl_base_select32.slurm`
- submitter：`tools/submit_qwen3vl_base_select32.sh`
- scorer：`tools/score_qwen3vl_sft_smoke.py --run-kind base_predict`
- scorer 单测：`tests/test_qwen3vl_sft_smoke_score.py`

源数据身份固定为：

| 文件 | SHA256 |
|---|---|
| `manifest.json` | `<hash omitted>` |
| `validation.json` | `<hash omitted>` |
| `sharegpt.json` | `<hash omitted>` |

不能改用配置历史默认曾指向的 `CA_0729(1)` 数据。两次构建中的绝对图片路径不同，
所以必须使用成功训练作业 `50129` 实际读取的 `(2)` 数据，才能形成严格同输入对照。

## Predict-only 合同

专用 YAML 和 worker 双重固定以下有效参数：

```yaml
stage: sft
do_train: false
do_eval: false
do_predict: true
predict_with_generate: true
finetuning_type: full
eval_dataset: levir_qwen3vl_sft32_eval
template: qwen3_vl_nothink
cutoff_len: 4096
per_device_eval_batch_size: 1
bf16: true
do_sample: false
temperature: 1.0
top_p: 1.0
top_k: 50
max_new_tokens: 256
```

配置中不得出现 `dataset`、`adapter_name_or_path` 或任何 `lora_*` 字段。
`do_eval=false` 是为了避免对同一 32 条数据重复生成；SFT workflow 的
`do_predict` 已经使用 `eval_dataset` 写出所需预测。

固定 LLaMA-Factory 版本的独立 `eval` CLI 不可用，因此命令入口仍是：

```text
python -m llamafactory.cli train <predict-only-config>
```

这里的 `train` 是 workflow 入口名称。`do_train=false` 会让模型以
`is_trainable=false` 加载并进入 eval 模式；workflow 不进入 `trainer.train()` 或
`save_model()`。

## 提交与监控

只允许从管理节点运行 submitter，由 Slurm 在计算节点做模型加载和生成：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent
REPO_ROOT="$PWD" bash tools/submit_qwen3vl_base_select32.sh
```

需要固定已知空闲节点时可显式指定：

```bash
NODE=gpuXX REPO_ROOT="$PWD" bash tools/submit_qwen3vl_base_select32.sh
```

默认申请 `gre` 分区、1 张 NVIDIA A800、2 CPU、24 GiB 内存、30 分钟。24 GiB
高于同环境成功 SFT32 作业的 6.0 GiB batch MaxRSS，也高于完整 E2E5 基线的
15.4 GiB batch MaxRSS，同时避免在账号其他任务占用接近 1 TiB 用户级配额时继续申请
不必要的 32 GiB。submitter 在提交前检查
Slurm 控制面、`guisongxia` 同名重复作业、固定模型/环境/代码路径、LLaMA-Factory
commit、predict-only 配置、32 条数据结构及上述哈希。它将输出和源数据目录分离，
拒绝覆盖非空 run 目录。

按 90–120 秒窗口用正式 Job ID 监控：

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,MaxRSS
```

worker 无论成功或失败都会把状态写入 `logs/status-<job>.txt` 并发送 QQ 通知。

首次 Job `50227` 在模型加载前因多余 CLI `save_strategy=no` 被解析为布尔值而失败，
没有预测或 metrics；Job `50285` 又在 worker 启动前收到调度 signal 53，未执行应用
代码。二者均按实验登记规则不进入主表。最终 clean retry 删除多余覆盖、固定
`gpu25` 并使用 24 GiB 申请，以 Job `50295` 成功完成。

## Worker 执行门

worker 在加载 4 GB 模型前完成：

1. 再次核对 predict-only YAML，拒绝训练、LoRA 或 adapter 配置。
2. 运行数据构造器 `--validate-only`，再核对三项固定 SHA256 和 32 行合同。
3. 运行 scorer 单测；单测失败时不加载模型。
4. 核对固定 Python 包、LLaMA-Factory commit、Qwen3-VL model type。
5. 确认只看到一张支持 BF16 的 NVIDIA A800 Slurm GPU，并清除分布式 launcher 环境变量。

预测结束后必须存在：

- `generated_predictions.jsonl`
- `predict_results.json`

同时 local output 必须不存在：

- `adapter_config.json`、`adapter_model.*`
- `checkpoint-*`
- `train_results.json`、`eval_results.json`、`trainer_state.json`
- `optimizer.pt`、`scheduler.pt` 或其他模型训练权重

worker 和 scorer 都会独立执行训练产物污染检查。job-local scratch 在结果复制和
scorer 完成后清理；共享 run 目录不保存模型权重。`runtime_identity.json` 会记录
实际 base 模型全部文件和关键 worker/scorer/config 文件 SHA256，避免同路径换权重或
dirty worktree 无法追溯。

## Scorer 语义与验收

`base_predict` 是对现有 scorer 的向后兼容扩展；默认 `sft_smoke` 行为不变。
base 模式只要求 predict metrics，不伪造空的 train/eval JSON，并在三份报告中明确写：

```text
run_kind=base_predict
training_performed=false
model_state=base
adapter_created=false
same32_reference=true
overfit_smoke=false
```

作业硬门是 pipeline 完整性，而不是要求未经训练的 base 模型通过协议门：

- `generated_predictions.jsonl` 恰好 32 个非空行、32 个可解析外层 JSON object；
- 32 条 label 均可解释为合法 select target；
- predict metrics 可读；
- 没有任何 adapter/checkpoint/training artifact。

pipeline 完整时，即使 base 输出没有达到严格 JSON/schema 门，scorer 仍返回成功完成，
但保留 `acceptance_status=failed` 和 `protocol_gate.passed=false`。数据或产物不完整仍返回
pipeline failure。

必须记录而不作为 base 作业硬门的质量指标包括：

- strict JSON count/rate
- exact schema count/rate
- unknown region ID count/rate
- region ID accuracy
- exact string match
- BLEU-4、ROUGE-1/2/L
- generation runtime 和吞吐

完成后可与已保存的 post-SFT same32 指标计算功能差值，但必须继续注明：32 条数据既是
SFT train/predict 同集，又包含 GT-derived candidate/proposal/facts 和明显 text-only
shortcut。因此该差值只能回答“在这批同集协议样本上，SFT 是否改变输出”，不能回答
held-out 泛化、遥感视觉理解或完整 verifier/E2E 是否改善。

## 本次结果

Job `50295` 在 `gpu25` 的单张 `NVIDIA A800-SXM4-80GB` 上 `COMPLETED 0:0`：

- Slurm elapsed `00:09:47`，worker elapsed `584s`，batch MaxRSS `2,530,388 KiB`；
- 32/32 个外层 JSONL object 完整，pipeline status 为 `success`；
- strict JSON `28/32 = 0.875`，exact schema `28/32 = 0.875`；
- unknown region ID `0/32`，region ID `26/32 = 0.8125`；
- exact string match `0/32`；
- BLEU-4 `50.02246875`，ROUGE-1/2/L 分别为
  `62.30181875 / 57.06209375 / 68.44459375`；
- `protocol_gate.passed=false`、`acceptance_status=failed`，原因是严格 JSON/schema
  各缺 4 条；这是 base 模型质量测量，不是 pipeline 失败；
- `training_performed=false`、`adapter_created=false`、
  `forbidden_local_artifacts=[]`，没有 adapter、checkpoint 或训练状态产物。

同一 32 条 post-SFT 的 `region_id_accuracy=1.0`，因此同集功能差值为 `+0.1875`。
该差值仍受 train=eval、GT-derived facts 和 text-only shortcut 限制，不能解释成视觉
泛化提升。运行时记录的 Chang-Agent commit 为
`f4fa5e0aa5b8fb2ab4291ddf5c761bc49b03c6e5`，模型权重 SHA256 为
`<hash omitted>`。

## 共享产物

成功 run 实际只保留：

```text
.base_predict_submitted
.base_predict_complete
config_source.yaml
config_effective.yaml
launch_command.txt
llamafactory_commit.txt
environment_versions.txt
predict_results.json
generated_predictions.jsonl
all_results.json
base_metrics.json
base_manifest.json
completion_summary.json
runtime_identity.json
logs/
```

核验后的 base same32 结果已回填 `0729(1)` 同一实验行，没有把失败尝试登记为实验，
也没有新增一行伪装成独立训练实验。该路径不调用或修改任何 E2E launcher。
