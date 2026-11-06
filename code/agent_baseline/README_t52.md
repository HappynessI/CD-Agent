# T5.2：LEVIR 点反馈修正

本次交付代码，未运行实验。下面的处理、缓存检查、回放和 Agent 命令通过 `sbatch` 在计算节点执行。使用现有 `run_cpu.sbatch`、`run_agent.sbatch`；它们在成功或失败后发送完成通知。资源是申请估计，尚未实测。提交前运行 `scontrol ping`，检查 `squeue -u guisongxia`、分区和重复任务。控制面连接失败时先处理连接。

## 固定口径

- 给定掩码：`$AGENT_RUN/init/sam3_imc_p85/val`，沿用 T5.1 的 0.85 掩码。
- 8 连通，与 `common.components` 和掩码工具一致。漏检是给定掩码覆盖真值组件严格小于 50%，点击区域取未覆盖部分。误检是给定组件与真值重叠严格小于 50%，点击区域取整个给定组件。恰好 50% 不选为错误。
- 点击取区域内欧氏距离变换的最大值；图外作背景，并列先取最小 y，再取最小 x。坐标为整数 `[x, y]`。漏检未覆盖部分即使断开，也仍属于同一个源错误。
- 两档混排错误区域，按实际点击区域面积降序；同面积先漏检，再按源组件编号升序。`gt:N`、`given:N` 分别记录源组件编号。k=3 最多三个，不足时全部取；k=1 与 k=3 的首个点击一致。
- 指令保留原 T5.1 全文，追加 `User feedback:`、点坐标和类型，以及 `Correct only the places the user pointed out and keep the rest of mask#0 unchanged.`。无错误时追加 `The user reviewed the mask and pointed out no errors. Keep mask#0 unchanged.`。所有完整措辞见 `tasks.t52_instruction` 和根目录 `NOTES.md`。
- 反馈 JSON 与清单的 `meta.feedback` 只记录点、类型、面积、源编号，不含真值像素。Agent 消息只使用指令，反馈面积、源编号、标签路径、错误掩码均不发送给模型。真值由离线生成器、评测和原有单向 Recorder 使用。
- 固定响应不调用规划器或验证器。漏检用一个正点，若该点属于 T2 的 SAM3 建筑实例，选 T2，否则选 T1。直接使用 `read_soft('val', name, 1, .85, .5)`：512 网格、像素 0.85、分数 0.5，和 `segment_text` 的 LEVIR 缓存口径一致。误检删除给定掩码中含点的 8 连通域。各点独立，以“给定掩码删除结果 ∪ 所有正点分割”合成；新增分割与被删组件重叠时保留新增分割，顺序不影响结果。
- `SharedPrompter.for_image('val', name)` 读取 `$OVCD_RUN/sam_emb/val/<stem>.pt`，不重新编码两期图。新增 `--sam-prompt-device cuda` 将该解码器放在 GPU；默认仍为 cpu。Agent 申请一张 GPU 供点提示和需要的 SAM3 worker 使用。
- T5.2 清单入口先核对两期路径是否对应 LEVIR 的 `val_original/A,B/<name>`，按源图名取缓存，允许输出 `id` 与源图名不同。默认 T1.1、T5.1 的清单分支不变。

## 逐步运行

每步是一条提交或设置命令。完成依赖作业并检查退出状态后再进行下一步。下面从含 `code/` 的当前交付目录执行，`AGENT_CODE` 指向本次代码；已有认证由原环境提供，不在文档中保存认证内容。`BAILIAN_CSV` 须已由账号配置导出，`run_agent.sbatch` 会检查它存在。

1. 设置公共环境。这里设置现有共享运行目录，后续代码只读取已有 G2 结果，T5.2 结果写入新配置和 `t52/`。

```bash
export AGENT_CODE="$PWD/code/agent_baseline" AGENT_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run OVCD_CODE="$PWD/code/ovcd_v0" OVCD_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/ovcd_v0_run PYTHONPATH="$PWD/code:$PWD/code/ovcd_v0${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AGENT_ASK_SLOTS=3 USE_LOCAL_WORKERS=1 USE_ASK_WORKER=0 T52_PARTITION=gre
```

2. 创建 Slurm 日志目录。已有 launcher 的 stdout/stderr 保留在共享盘，通知会给出状态和日志位置。

```bash
mkdir -p "$AGENT_RUN/logs"
```

3. 执行新增单元测试及原有回归。CPU 1，内存 4 GiB，GPU 0，预计时限 20 分钟。新模块用小数组和临时 JSON/PNG，固定响应使用假 prompter。全量回归由计算节点执行，本次仅实际执行了新模块。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-tests --cpus-per-task=1 --mem=4G --time=00:20:00 "$AGENT_CODE/run_cpu.sbatch" -m unittest discover -s tests -t .. -v
```

4. 生成 64 图反馈、两档区域掩码和两份清单。CPU 1，内存 4 GiB，GPU 0，预计时限 30 分钟。生成器要求恰好 64 张验证图；缺标签或给定掩码即失败。正式清单生成成功后再入库反馈 JSON，不用临时或空文件代替。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-feedback --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.make_t52_feedback --run-root "$AGENT_RUN" --output "$PWD/results/agent_baseline/data/t52_feedback.json"
```

输出为 `results/agent_baseline/data/t52_feedback.json`、`$AGENT_RUN/t52/regions/k{1,3}/<stem>/<source>_<N>.png` 和 `$AGENT_RUN/t52/tasks/T5.2_k{1,3}.jsonl`。无错误图片在 JSON 和清单中保留空反馈，无区域文件。区域 PNG 只用于离线检查，禁止作为 Agent 输入。

5. 检查清单缓存路径与真实 SAM3 软缓存。CPU 1，内存 4 GiB，GPU 0，预计时限 10 分钟；不加载模型。k1、k3 图单相同，检查 k1 即覆盖 64 图两期路径。实际解码和模型环境在后面的试跑验证。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-cache --cpus-per-task=1 --mem=4G --time=00:10:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.make_t52_feedback --check-caches "$AGENT_RUN/t52/tasks/T5.2_k1.jsonl"
```

检查逐图读取 `sam3_soft/t512s256/val/<stem>.npz` 的两期数据，使用与 `SegmentBackend` 相同的 `read_soft`；同时检查 `sam_emb/val/<stem>.pt` 存在。缺缓存时停止，按原项目缓存准备流程补齐后再提交。此步骤不会生成缓存、重新编码图片或调用 API。

6. 两档各 10 张 Agent 试跑，单个作业内顺序执行。CPU 4，主机内存 32 GiB，GPU 1（建议至少 24 GiB 显存），预计时限 1 小时。保留 SAM3 worker 以支持 Agent 的现有分割工具；默认 building/512/.85/.5 查询走 LEVIR 软缓存，点提示走预存 SAM 编码。Python、skill、plan 保持关闭。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-smoke --cpus-per-task=4 --mem=32G --time=01:00:00 "$AGENT_CODE/run_agent.sbatch" --tasks "$AGENT_RUN/t52/tasks/T5.2_k1.jsonl" --task T5.2 --split val --config json_qwen37flash_g2_t52k1_smoke_s1 --limit 10 --seed-offset 1 --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch t52_smoke --max-cost-cny 5 ::: --tasks "$AGENT_RUN/t52/tasks/T5.2_k3.jsonl" --task T5.2 --split val --config json_qwen37flash_g2_t52k3_smoke_s1 --limit 10 --seed-offset 1 --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch t52_smoke --max-cost-cny 5
```

验收试跑：每档 10 个 `final.json`；核对 `input.json` 的给定掩码路径、指令点与 JSON 一致；检查 `steps.jsonl` 中默认 `segment_text` 的 `backend=sam3_soft_cache` 和 `cache_hit=true`。点提示记录为 `backend=sam_prompt`，T5.2 路径已固定为预存编码入口，不传 `rgb`。实际有点提示调用的会话应成功读入编码；若 Agent 未调用点提示，下一步固定响应的漏检样本验证这条路径。检查所有错误和兜底会话，不按得分更换样本、门限或指令。无反馈图应原样提交，偏离计入区域外改动。

7. 固定响应 k=1。CPU 2，主机内存 16 GiB，GPU 1（建议至少 8 GiB 显存），预计时限 30 分钟。复用 CPU launcher 的通知和环境，Slurm 显式分配 GPU。无需 API 认证。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-fixed-k1 --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t52_fixed --tasks "$AGENT_RUN/t52/tasks/T5.2_k1.jsonl" --output-dir "$AGENT_RUN/t52/fixed/k1" --device cuda
```

8. 固定响应 k=3，资源同上。`responses.json` 记录每个点所选时相、SAM 自评分或删除的给定组件编号。默认用 GPU 解码；`--device cpu` 可在 CPU 计算作业上运行，未实测耗时。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-fixed-k3 --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t52_fixed --tasks "$AGENT_RUN/t52/tasks/T5.2_k3.jsonl" --output-dir "$AGENT_RUN/t52/fixed/k3" --device cuda
```

9. 试跑和固定响应验收后，全量 2 档 × 64 图 × 3 次，共 384 会话。CPU 4，主机内存 32 GiB，GPU 1（建议至少 24 GiB 显存），预计时限 6 小时。以下 Bash 命令一次提交一个作业，六组顺序执行；正式配置前缀分别为 `json_qwen37flash_g2_t52k1`、`json_qwen37flash_g2_t52k3`。沿用 G2 参数，不修改其他任务。

```bash
args=(); for n in 1 2 3; do for k in 1 3; do args+=(--tasks "$AGENT_RUN/t52/tasks/T5.2_k${k}.jsonl" --task T5.2 --split val --config "json_qwen37flash_g2_t52k${k}_s${n}" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch t52_g2_full --max-cost-cny 40 :::); done; done; sbatch --partition="$T52_PARTITION" --job-name=t52-full --cpus-per-task=4 --mem=32G --time=06:00:00 "$AGENT_CODE/run_agent.sbatch" "${args[@]}"
```

正式运行排在线 B 试跑之后、线 B 全量之前，合计最多两个 Agent 作业同时运行。约 24 元仅为旧配置推算，实际看账本；命令设批次估算上限 40 元。额度或预算停止时先查 `STOP.json`、`budget/t52_g2_full/STOP.json` 及日志，处理实际原因后再原配置续跑；已存在 `final.json` 会跳过。错误兜底会话保留，评测计数，不能静默删掉后重跑。试跑配置与正式三次重复分开。

10. 写指标输入清单。CPU 1，内存 2 GiB，GPU 0，预计时限 5 分钟。默认读取已有 G2 的 `json_qwen37flash_g2_s{1,2,3}/T5.1/val`，与 T1.1 的重跑结果无关。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-spec --cpus-per-task=1 --mem=2G --time=00:05:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t52_metrics --write-spec --run-root "$AGENT_RUN" --output "$AGENT_RUN/t52/metrics_spec.json"
```

11. 汇总 JSON 和一张 Markdown 表。CPU 1，内存 4 GiB，GPU 0，预计时限 30 分钟。输出 `metrics.json`、`metrics.md`；spec 中路径均可调整，固定响应输出目录由其 `fixed_dir` 指定。

```bash
sbatch --partition="$T52_PARTITION" --job-name=t52-metrics --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t52_metrics --spec "$AGENT_RUN/t52/metrics_spec.json" --output "$AGENT_RUN/t52/metrics.json"
```

12. 原有 G1 历史回放。CPU 1，内存 4 GiB，GPU 0，预计每组时限 1 小时。下面一次提交六个只读归档的回放作业；每组输出独立。检查这些作业成功后才声称历史回放通过。本次未执行。

```bash
for n in 1 2 3; do for task in T1.1xC0 T5.1; do sbatch --partition="$T52_PARTITION" --job-name="t52-replay-s${n}-${task}" --cpus-per-task=1 --mem=4G --time=01:00:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.replay_g1 --config "json_qwen37flash_g1_s${n}" --task "$task" --split val --reference-code "$AGENT_RUN/code_g" --output-dir "$AGENT_RUN/t52/regression/g1_s${n}_${task}" || break 2; done; done
```

使用 `squeue -u guisongxia`、`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS` 检查作业；运行中每 90–120 秒检查一次。通知只报状态、耗时和结果/错误日志位置。Slurm 成功后仍须核对会话数、错误/兜底及缓存信息。缓存检查或模型加载失败时停止依赖步骤，保留 traceback；不替换依赖或返回假成功。

## 指标与验收

整图 F1^c、P、R 使用 `common.confusion/scores`，每次先累计像素再取三次均值，同时给逐次值和相对给定掩码的 ΔF1。给定原样、固定响应各报告一遍；不带反馈的 G2 与 Agent 都报告三次。修复率按被点源组件累计：漏检最终覆盖完整真值组件至少 50% 才算修好，误检最终保留完整给定组件严格小于 50% 才算修好。无被点组件时比例记 null，Markdown 显示 —。

未指出区域以被点完整源组件的并集为起点，按欧氏距离外扩半径 32 像素；之外按数据集累计增删像素，并计算该区域的累计 F1 变化。空反馈时整图都属于未指出区域。全部被排除或区域内没有真值及预测时 F1 按 `common.scores` 为 0；原样结果的变化为 0。两档都使用相同口径评测全部对照。

Agent 对固定响应、对给定掩码分别按图配对 bootstrap 4000 次，种子 1016，三次重复固定且共享每次图像抽样，不对重复再次抽样。报告误差只反映图像抽样变化。缺会话、缺已有路径指向的掩码直接报错；已有的明确 `error + empty` 兜底记录按空预测计入，另计错误和兜底。JSON 保存全部逐次指标，Markdown 同时给三次均值和逐次行。

T5.3 复用 `error_regions(truth, initial)` 和 `region_mask(..., full=False/True)`，已在独立模块实现语言反馈。G3 下 T5.2 的自动核验与默认 `ask()` 限于公开点击半径 32 像素内相交的组件及相对给定掩码改动过的组件；空反馈只核验改动。G3 关闭时保留本手册描述的 G2 行为。G3 开关、T5.3 的生成、对照、指标和回归命令见 [README_t53.md](README_t53.md)。

本次未生成正式反馈 JSON、区域 PNG、清单、模型掩码或指标文件；这些由以上计算作业生成。没有真实缓存和模型验收结果；极小单元测试的通过仅覆盖代码逻辑与小规模文件读写。
