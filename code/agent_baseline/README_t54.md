# T5.4：旧建筑地图与新影像

本批仅写代码并运行假后端单元测试。下面的批量准备、SAM3、API 和指标命令供合并后执行，通过 Slurm 提交。当前代码未运行这些实验。

## 输入与输出

T1 是 WHU-CD 的 2012 年建筑地图：`slabel1 == 1` 渲染成黑底白建筑的 RGB PNG。保留原始像素网格，不叠加 2012 年照片。T2 是 `img2` 的 2016 年影像。任务体系 HTML 的 T5.4 要求“没有前时相影像，对照地图要素找出不一致处”，这里用完整建筑轮廓的栅格实现。

英文指令固定为：

> Update the old building map using the new image. Output a binary change mask of newly built and demolished buildings.

真值为 `(slabel1 == 1) XOR (slabel2 == 1)`，包括新建和拆除。另保存 `new` 和 `demolished` 两个子集。WHU 的 0 是有标注的背景，不沿用 SECOND 的“0=未标注”口径；相同建筑占用像素保持为无变化。现成变化标签只用于一致率核对，不能替代本任务的真值。三个训练图块的轻量核对发现二者并非完全一致，完整核对结果由 `t54/preparation.json` 给出。

生成器读取冻结的 `R/whu/names_val128.txt`（报告）和 `names_dev60.txt`（开发）。保留图单中的空真值图，不重新抽样。两份图单必须互不重叠，只允许 `train_*.png`。不读取官方 `list/test.txt`。图单不存在时明确失败，不自动生成划分。

产物：

- `R/t54/maps/<name>.png`：输入地图，只从 slabel1 生成。
- `R/t54/gt/{any,new,demolished}/<name>.png`：离线评分真值。
- `R/t54/tasks/T5.4_{dev,val}.jsonl`：任务清单，`image_t1` 指向输入地图，`image_t2` 指向 img2。
- `R/t54/preparation.json`：图单、非空真值图数、实际编码、异或与变化标签的一致率和不一致像素数。

## 会话和工具

必须同时传入 T5.4 清单和 `--map-input`，默认关闭。Python 接口为 `Toolbox(rgb, directory, ..., map_input=True)`，其中 `rgb[0]` 是渲染好的地图。`run_session(..., task='T5.4')` 要求已启用地图输入。`read_tasks` 只校验清单，实际会话开关由执行入口控制。

`map#0` 保存旧地图的建筑掩码，支持 inspect、view、combine、morph、add/remove_region 和 submit 等所有接受 Mask 的工具。它不占普通掩码编号，不自动成为当前答案，也不作为 T5.1 给定变化掩码。G2/G3 的最近答案查找排除输入地图；模型未生成答案时仍按原规则使用空预测。首条消息说明颜色、时间和句柄。`view` 的 T1 一直是地图，包括局部视图和叠加视图。记录器将输入句柄归档为 `masks/map_0.png`，工具调用轨迹保留 map#0 输入引用。

地图上调用 `segment_text`（SAM3）、`segment_prompt`、`refine_region` 以及语义分割、无类别分割、特征提取、分类工具会得到 `unsupported_input`，工具后端不会运行。地图已经提供建筑轮廓，光学模型未经验证可识别这种栅格。T2 上的工具照常工作；跨两期光学特征比较和示例匹配也返回明确错误。掩码运算可以直接使用 `map#0`。现有 combine 的 subtract、union 可组合成异或，无需新增工具或改变旧 schema 的运算枚举。

只有旧地图进入 Session。`slabel2` 和 `label_unch0chi1` 不作为在线输入；清单的 `gt_mask` 只由 Recorder 消费，文件路径和评分结果不进入模型消息。不开启地图输入时，现有 G1–G3 的消息、图像字节和工具说明保持一致。

## 固定路线、规则和指标

固定路线只在 T2 调用一次 `segment_text('building')`：512 分块、像素门限 0.85、实例门限 0.5、worker confidence 0.1，复用 SegmentBackend 和现有 SAM3 worker。实例求并集后与 `map#0` 异或。不做实例匹配。输出 `fixed/<split>/xor` 和 `fixed/<split>/buildings`，分别报相同变化真值上的累计像素 F1/P/R。后者就是“不异或”的参照，不能解读成建筑分割指标。

默认不做小组件过滤（min_area=0）。可选过滤只作用于异或变化掩码，使用 8 连通组件，面积单位为原图像素。正面积门限必须提供开发图单；`--write-selection` 保存开发设置，报告图只能通过 `--frozen-settings` 应用正面积门限。同分时保留较小门限。使用默认 0 的完整命令见下文；本批没有选择过面积门限。

规则对照对异或掩码的每个组件使用四幅图：旧地图近景、新影像近景、旧地图上下文、新影像上下文。复用 SECOND `f37_corr` 的 qwen3.7-flash 参数、图像渲染、组件 seed、费用上限和逐组件续跑。仅删除解析为 `no` 的组件；`yes` 和未解析回答保留。问法为：

> T1 is an old building map (white buildings, black background); T2 is a new optical image. The red contour marks the same region in both views. Does building presence inside the contour differ between the old map and the new image, through construction or demolition?

回答为 JSON，类型限 new_building、demolished、no_change、uncertain。在线 ask 和离线规则使用相同地图口径。问题要求比较地图几何与影像中的建筑存在情况，因此提供这个规则对照；模型对这种输入的能力尚未实测，不能报告有效性或收益。返回 no 的组件才删除，不把地图上的颜色差异当作变化。

规则输出为 `rule/<split>/any/masks`、`answers.jsonl` 和 `summary.json`。失败后保留已完成回答，原命令可续跑。改变输入掩码、图像或设置时使用新目录；费用停止后先核对账本及授权，不自动删除 STOP 文件。`--max-cost-cny` 必填，默认串行。费用取决于候选数，完成前不能当作实测值。

`t54_metrics.py` 直接复用 second_metrics 的 counts、score、paired_bootstrap。每次重复累计 TP/FP/FN 后算 F1/P/R，Agent 报三次均值；按图配对 bootstrap 4000 次、种子 1016，三次重复共享抽样。报告 Agent 对固定路线、规则和原始建筑参照的 ΔF1 及 95% 区间。也支持仅有固定路线/规则时汇总，并报告异或路线相对参照的配对区间。所有图均纳入，错误会话计为空预测；缺会话、缺掩码、尺寸不同会明确报错。区间只表示图像抽样变化。

## 环境与资源

准备和指标使用 exp5-env，1 CPU、4 GiB、0 GPU。规则使用 exp5-env，1 CPU、4 GiB、0 GPU，等待 API。固定路线主进程使用 exp5-env，SAM3 worker 使用 omniovcd-env，共 4 CPU、24 GiB、1 GPU，建议显存至少 24 GiB。资源配置参照 `R/probe/refs_t12c1t.sbatch`；尚未运行，耗时和峰值内存未测。

`analysis/run_t54.sbatch` 默认 1 CPU / 4 GiB / 2 小时。固定路线命令显式增加 GPU、CPU、内存和时限。脚本在同一作业内启动 localhost SAM3 worker，等健康检查成功再处理，退出时清理 worker。日志写在对应产物目录的 logs/。成功和失败都调用工作区 QQ 通知工具，包含任务、退出状态、耗时和目录。本批未发送通知。

## 执行命令

以下每一步执行一条命令。设当前目录为合并后的仓库根，代码在 code/。先确认数据与两份图单存在。先完成一个作业并检查状态，再执行后续依赖步骤。

1. 设置路径、解释器和单线程环境。

```bash
export R=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run AGENT_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run CODE_ROOT="$PWD/code" PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python PYTHONPATH="$PWD/code" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

2. 创建共享日志目录。

```bash
mkdir -p "$R/t54/logs" "$R/t54/fixed/dev/logs" "$R/t54/fixed/val/logs" "$R/t54/rule/dev/logs" "$R/t54/rule/val/logs"
```

3. 检查 Slurm 控制面。失败则先处理连接。

```bash
scontrol ping
```

4. 查看重复作业和可用分区。

```bash
squeue -u guisongxia
```

5. 确认 gre 分区与 GPU 资源，按实际可用节点调整提交参数。

```bash
sinfo -p gre -o '%P %a %l %D %G'
```

6. 提交地图、真值与两份清单准备（CPU 1 / 4 GiB / GPU 0）。

```bash
sbatch --output="$R/t54/logs/prepare-%j.out" --error="$R/t54/logs/prepare-%j.err" code/agent_baseline/analysis/run_t54.sbatch prepare "$R/t54" --run-root "$R"
```

7. 提交开发集固定路线（CPU 4 / 24 GiB / GPU 1），默认不做小组件过滤。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=04:00:00 --output="$R/t54/fixed/dev/logs/fixed-%j.out" --error="$R/t54/fixed/dev/logs/fixed-%j.err" code/agent_baseline/analysis/run_t54.sbatch fixed "$R/t54/fixed/dev" --tasks "$R/t54/tasks/T5.4_dev.jsonl" --run-root "$R" --min-area 0 --dev-names "$R/whu/names_dev60.txt" --write-selection "$R/t54/selection.json"
```

可选：在第 7 步用独立的输出目录分别试 min_area=16、64，依据完整开发集 counts.json 的 xor F1 选设置，并把选中设置保存到 selection.json。此时改写 selection.json 的路径应指向选中那次运行；不能依据报告图选门限。默认运行无需这一步。

8. 报告集应用已经保存的设置（CPU 4 / 24 GiB / GPU 1）。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=04:00:00 --output="$R/t54/fixed/val/logs/fixed-%j.out" --error="$R/t54/fixed/val/logs/fixed-%j.err" code/agent_baseline/analysis/run_t54.sbatch fixed "$R/t54/fixed/val" --tasks "$R/t54/tasks/T5.4_val.jsonl" --run-root "$R" --frozen-settings "$R/t54/selection.json"
```

9. 指定现有本地凭据导出文件的路径。替换下面的占位路径；run_with_key.py 在作业内读取该文件，凭据只保存在进程环境中。

```bash
export BAILIAN_CSV=/path/to/your/local/credential-export.csv
```

10. 开发集核验规则（CPU 1 / 4 GiB / GPU 0）。费用上限示例 10 元。

```bash
sbatch --output="$R/t54/rule/dev/logs/rule-%j.out" --error="$R/t54/rule/dev/logs/rule-%j.err" code/agent_baseline/analysis/run_t54.sbatch rule "$R/t54/rule/dev" --tasks "$R/t54/tasks/T5.4_dev.jsonl" --mask-dir "$R/t54/fixed/dev/xor" --max-cost-cny 10 --concurrency 1
```

11. 报告集核验规则，保持相同问法与设置。

```bash
sbatch --output="$R/t54/rule/val/logs/rule-%j.out" --error="$R/t54/rule/val/logs/rule-%j.err" code/agent_baseline/analysis/run_t54.sbatch rule "$R/t54/rule/val" --tasks "$R/t54/tasks/T5.4_val.jsonl" --mask-dir "$R/t54/fixed/val/xor" --max-cost-cny 10 --concurrency 1
```

12. 生成报告配置（只写一个小 JSON，无实验计算，可在管理节点执行）。

```bash
"$PY" -B -m agent_baseline.analysis.t54_metrics --write-spec --run-root "$R" --output "$R/t54/report_spec.json"
```

13. 提交累计像素指标及配对区间（CPU 1 / 4 GiB / GPU 0）。

```bash
sbatch --output="$R/t54/logs/metrics-%j.out" --error="$R/t54/logs/metrics-%j.err" code/agent_baseline/analysis/run_t54.sbatch metrics "$R/t54" --spec "$R/t54/report_spec.json" --output "$R/t54/report.json"
```

14. 查看作业状态。JOB_ID 替换成提交返回的编号。

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

15. 在当前代码快照中运行本任务假后端测试（登录节点允许，单线程；临时目录在当前目录）。先创建临时目录。

```bash
mkdir -p "$PWD/code/agent_baseline/.tmp"
```

16. 执行测试。

```bash
TMPDIR="$PWD/code/agent_baseline/.tmp" "$PY" -B -m unittest agent_baseline.tests.test_t54 -v
```

后续 Agent 会话通过实验作业入口运行，命令参数包括 `--tasks "$R/t54/tasks/T5.4_val.jsonl" --map-input --task T5.4 --pixel-thr .85 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence`，以及所用 G3/G4 开关和 SAM3 worker。三次重复采用不同 seed-offset 和输出 config。地图设置只对 T5.4 生效。Agent 跑完后，在第 12 步增加 `--agent-prefix t54_agent`，得到三份 `R/t54_agent_s1..3/T5.4/val` 的路径，再重跑指标步骤。
