# Qwen3-VL SFT32 数据契约（0729）

入口：`tools/build_qwen3vl_sft_smoke.py`。

默认输出目录：

```text
/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CA_0729(1)-qwen3vl-sft32-smoke/data
```

数据构造只读取 `LEVIR-CD/train_original/{A,B,label}`。1024×1024 原图按固定
4×4 网格切成 256×256 patch；选择 16 个 change-positive patch 和 16 个空标注
patch，并保证 32 个 patch 来自不同原图。positive 样本从一块 GT change 中删除
一个连通子块形成 FN；empty 样本加入由样本 ID 决定的紧凑白色区域形成 FP。
raw GT label image 不会作为独立图像或字段直接输入模型；但 FN 的 candidate change
mask/overview、`t1_mask`/`t2_mask` facts 和 proposal geometry 都由 GT 派生并对模型
可见。这是刻意用于打通链路的 shortcut，不是无 GT 泄漏的真实 verifier 数据。

每条模型输入只有一张复用当前 Proposal 风格的组合图：

```text
T1 RGB | T2 RGB | candidate change mask
```

每图固定四个 `r0`–`r3` 候选；真实错误候选的 ID 按样本哈希打乱。assistant
只学习当前 `select` stage 的严格 JSON：

```json
{"selection":{"region_ids":["r2"],"reason":"short observable reason"}}
```

训练数据不含 `target_view`、action、coordinate、accept、quality 或 CoT 监督。
输出包含：

- `sharegpt.json`：32 条 `conversations + images`，每条一个绝对图像路径；
- `dataset_info.json`：`levir_qwen3vl_sft32_train` 与
  `levir_qwen3vl_sft32_eval`，两者指向同一 overfit smoke 文件；
- `manifest.json`：源 split、原图分组、crop、类别、目标 region 和哈希；
- `validation.json`：样本数、类别平衡、占位符、绝对路径、重复项和 schema 门。

实际数据预处理必须通过 CPU Slurm 作业执行；管理节点只做代码编辑和静态检查。
32 条 train/eval 同集仅用于验证训练链路和 overfit，不得作为泛化指标；尤其不能
据此宣称模型学会了从原始双时相图像自主发现 mask 错误。

## CPU prepare 作业

统一入口会在一个 `gre` CPU 作业中顺序完成独立环境安装/审计、数据单测、32 条
样本构建和持久化结果二次校验；管理节点上的 submitter 只做 Slurm preflight 和
提交：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent
bash tools/submit_qwen3vl_sft_prepare.sh
```

默认资源为 0 GPU、2 CPU、16 GiB 内存和 1 小时时限。stdout/stderr 写入
`outputs/CA_0729(1)-qwen3vl-sft32-smoke/logs/prepare-slurm-<job>.{out,err}`；
作业无论成功或失败都会发送不含凭据的 QQ 状态通知。可用提交回显中的 Job ID
按 90–120 秒窗口监控：

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

Slurm 会把脚本复制到 spool 目录执行，因此 worker 通过 submitter 显式导出的
`REPO_ROOT` 定位仓库，不能依赖 `BASH_SOURCE[0]` 反推源文件目录。失败重试时保留已有
`prepare-slurm-*`/status 日志用于审计，不隐式删除已生成的数据目录。

成功门包括：

- `tools/setup_qwen3vl_sft_env.sh install` 完成且实际多模态训练模块可导入；
- `tests/test_qwen3vl_sft_smoke_data.py` 全部通过；
- `validation.json` 为 `valid=true`、恰好 32 条、16 FN + 16 FP 且来自 32 个原图；
- run 根目录生成 `.prepare_complete`，日志目录生成
  `prepare-status-<job>.txt`。

默认的环境、数据和运行目录分别可用 `TRAIN_ENV`、`DATASET_ROOT` 和 `RUN_DIR`
覆盖，`NODE` 可固定节点。例如仅指定节点时：

```bash
NODE=gpuXX bash tools/submit_qwen3vl_sft_prepare.sh
```

submitter 会检查 Slurm controller、同名活动作业、安全输出路径和状态 marker，且
拒绝覆盖已有 `data/` 或复用不干净的 run。失败结果保留用于审计；需要重试时使用
新的 `RUN_DIR`，不要直接删除或覆盖旧结果。只有 `.prepare_complete` 和
`validation.json` 均通过后才进入单 GPU smoke。prepare 生成的 train/eval 是同一
32 条 overfit 集，仍然不能用于报告泛化性能。
