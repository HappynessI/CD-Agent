# T3.2、T3.3、T3.4 跨图示例

本轮只交付代码与假后端测试。不运行模型、API、Slurm、真实任务批量生成或指标计算。以下实验命令供合并后使用，当前未执行。定义参照主克隆 `doc/agent_baseline/task_system.html` 的 T3 表格；TASK.md 所指 `doc/cd_agent_task_design.html` 不在主克隆快照中。

## 标签与示例规则

三组图单为 `R/second/second_val128.txt`（128 张报告图）、`R/second_dev/names.txt`（60 张开发图）与 `R/second_dev/rest.txt`（示例图）。三组必须互斥，rest 必须非空且无重复。只在 rest 里取示例；开发图和报告图均只用于检测与各自阶段的评分。每个任务只输出评测图的掩码，示例图不进评测。

SECOND 语义标签 0 表示未标注，类别 1–6 分别为低矮植被、裸地、树、水、建筑、运动场。令 `changed = (T1 != T2) & (T1 != 0) & (T2 != 0)`。

| 类型 | 示例 | 评测图真值 | 相似度 |
|---|---|---|---|
| T3.2 | rest 图 T2 某类的一个 8 连通组件，仅 T2 图块 | `changed & (T2 == target)`；每张图每个出现类别一个任务 | 提议 T2 外观均值与示例 T2 均值的余弦 |
| T3.3 | rest 图同转换 `source -> target` 的一个 8 连通组件，两期图块 | `changed & (T1 == source) & (T2 == target)`；每个非空转换一个任务 | 提议 `mean(T2)-mean(T1)` 与示例同方向差向量的余弦 |
| T3.4 | 正示例同 T3.3；负示例来自同一 rest 图的另一种有方向转换 | 同 T3.3 | `cos(提议差向量, 正例差向量) - cos(提议差向量, 负例差向量)` |

示例组件面积在 **16–4096 原生像素（含边界）**。真值保留全部匹配像素，包括小于 16 和大于 4096 的组件。T3.2 示例只在 SECOND 已标注的 T2 类别区域取，可能是实际地物的局部，不能据此宣称得到完整地物实例。类别“出现”指有方向的类别进入，不含类别保持不变。

固定种子默认为 1016，图名与任务类型、类别编号共同生成局部种子；候选按图名、类别、连通组件编号排序。T3.4 先从具备异类负例的正例候选中均匀抽取，再在同图异类组件中均匀取负例。无法为非空真值找到合格示例时明确报错。任务数与按类型、转换统计写入每个阶段的 `inventory.json`，不凭估计填写真实数量。

裁剪复用 `render.crop_box`：边长为组件包围框长边的 4 倍、至少 64 像素，上限为原图短边，中心靠边时整体移入图内。两期图块与示例掩码使用同一裁剪框。磁盘保存原生分辨率 RGB 与二值掩码；首条消息显示 384×384 JPEG，带掩码时画红色轮廓。DINOv2 读取原生 RGB 图块，红色轮廓只用于展示。

## 会话接口和消息顺序

新入口为 `--external-examples`，默认关闭。启用时须同时传 `--tasks` 与包含 `match_exemplar` 的 `--tools-t13`。关闭时原有 schema、系统提示、首条消息与工具回复保持不变；示例任务缺少清单或开关时会在运行前报错。

任务清单在原有字段之外增加 `examples` 数组。先正后负，每项包含 `role`（positive/negative）、`image_t1` 和/或 `image_t2`、可选 `mask`。相对路径以清单所在目录解析。T3.2 必须有 T2；T3.3、T3.4 必须有两期。`meta.example_provenance` 记录示例来源、组件与裁剪框，仅供离线追溯。`gt_mask` 仅交给 Recorder。

程序接口为 `attach_examples(session, load_examples(specs, worker_pool))`。不传示例时不添加会话属性。示例有独立 `Example` / `ExampleMask` 类型，与评测图尺寸无关，不加入当前答案，也不能提交为评测掩码。

句柄共用 Session 的递增编号。例如无初始掩码的 T3.4：正例 `example#0`、其掩码 `exmask#1`、负例 `example#2`、其掩码 `exmask#3`。没有示例掩码时只分配 example 句柄。句柄以实际首条消息为准。

首条用户消息依次包含：

1. 原有任务指令。
2. 评测图 `view#0 T1`、T1 图片、`view#0 T2`、T2 图片。
3. 正例说明：角色、example 句柄、日期、调用方式、可选 exmask 句柄和图块坐标约定。
4. 正例 T1 标签及图片（若有），再 T2 标签及图片。
5. T3.4 的负例说明、负例 T1 标签及图片、负例 T2 标签及图片。
6. T3.4 正负分数的组合说明。

示例说明文字模板是 `External positive example: example#0; dates: T1, T2. Use match_exemplar(exemplar={"example":"example#0"}, mode="change"). Example mask: exmask#1, shown by the red contour. Example coordinates belong only to this crop; submit a mask for the evaluation pair.` 无掩码时相应位置为 `Use the whole crop.`。T3.2 为 dates T2 与 mode t2；负例角色为 negative。

调用示例：

```json
{"props":"props#4","exemplar":{"example":"example#0"},"mode":"t2"}
```

```json
{"props":"props#4","exemplar":{"mask":"exmask#1"},"negative":{"example":"example#2"},"mode":"change"}
```

example 句柄使用其附带掩码，无掩码时使用整块。exmask 句柄定位到所属示例，使用该图块自己的特征。T3.2 的一期示例不能使用 change 模式。T3.4 一次调用产生组合分数，继续用 `select(where="score >= 门限")` 与 `to_mask` 取并集。同图 T3.1 的 `{"mask":"mask#0"}` 和点选路径保留。

## 固定路线和指标

固定路线合并评测图两期 SAM ViT-H 提议，以 DINOv2 均值向量计算上述分数，保留 `score >= threshold` 的提议并取并集。T3.2 的固定对照只使用 T2 外观相似度，没有额外的时间变化门限；可能选到未变化目标，这是这条固定路线的限制。

开发阶段按任务类型分别最大化汇总像素 F1，平局取较小门限。T3.2、T3.3 搜索 [-1,1]，T3.4 搜索 [-2,2]，步长均为 0.05。报告阶段必须传完整开发报告与 `--mask-dir`，核对图单、生成设置、后端配置，再使用冻结门限。报告图只跑一次，禁止用报告结果回调门限。

输出是 `mask-dir/<任务 id>.png`、逐任务 TP/FP/FN 与汇总报告。跨图任务评测图不排除任何示例位置。`t3x_metrics.py` 复用 `t3_metrics.py` 计数、F1 和配对抽样；分别报告类型与转换。由于一张图有多个转换任务，先按图汇总计数，再做 4000 次图像配对抽样（种子 1016）。三个 Agent 重复共用抽样。指标入口接受配置根目录，读取 `配置根/T3.x/val/<id>/final.json`。

## 资源估计

以下为估计，尚无真实数据或 GPU 实测。

| 步骤 | CPU | 内存 | GPU | 时间估计 |
|---|---:|---:|---|---|
| 任务生成（扫描 rest，逐图释放标签） | 1 | 4 GiB | 0 | 数分钟到数十分钟，取决于 rest 数量和共享盘 |
| 固定路线 | 2 | 24 GiB | 1，显存至少 16 GiB，建议 24 GiB | 数小时；新示例图块需额外 DINOv2 提取 |
| 指标和图像配对抽样 | 1 | 4 GiB | 0 | 数分钟 |
| 假后端单元测试 | 1 | 2 GiB | 0 | 秒到数分钟，受共享盘导入影响 |

缓存复用 `R/t13_probe/.runtime/vision-cache`。已有评测图的 SAM、DINOv2 条目可命中；示例图块为新路径，要计算自己的特征。改输出路径或重写图块会使基于路径、大小、修改时间的缓存条目失效；任务生成成功后固定清单与图块，不重复生成。固定路线没有训练和 checkpoint，也不调用外部 API。所有 GPU 操作须由 Slurm 调度。

## 合并后运行命令

从含 `code/` 的仓库根执行。每步等待上一作业成功、核对日志后再进行下一步。当前任务没有执行本节命令。Slurm 脚本结束时发送 QQ 通知（成功或失败、退出码、耗时、作业号、产物目录、最后三行结果或错误与日志路径）；通知配置沿用工作区现有命令，不写入仓库。

设置运行环境：

```bash
export R=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run PYTHONPATH="$PWD/code" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 T3X="$PWD/code/.runtime/t3x"
```

建立输出与临时目录：

```bash
mkdir -p "$T3X/logs" "$PWD/code/.runtime/tmp"
```

检查控制面：

```bash
scontrol ping
```

检查重复作业与现有资源：

```bash
squeue -u guisongxia
```

检查分区可用资源，默认脚本使用 gre：

```bash
sinfo -p gre
```

从现有模板准备后端配置；按本机已验证解释器和权重路径填写，避免覆盖已有配置：

```bash
cp -n code/agent_baseline/vision_t13.example.json "$T3X/vision.json"
```

生成开发与报告清单、示例图块、真值和统计：

```bash
sbatch --cpus-per-task=1 --mem=4G --time=02:00:00 --output="$T3X/logs/make-%j.out" --error="$T3X/logs/make-%j.err" code/agent_baseline/analysis/run_t3x.sbatch make "$T3X/tasks" --run-root "$R" --output-root "$T3X/tasks" --phase both
```

查看生成后的任务统计：

```bash
cat "$T3X/tasks/dev/inventory.json" "$T3X/tasks/val/inventory.json"
```

开发集选门限：

```bash
sbatch --gres=gpu:1 --cpus-per-task=2 --mem=24G --time=12:00:00 --output="$T3X/logs/dev-%j.out" --error="$T3X/logs/dev-%j.err" code/agent_baseline/analysis/run_t3x.sbatch fixed "$T3X/dev.json" --tasks "$T3X/tasks/dev/tasks.jsonl" --phase dev --run-root "$R" --vision-config "$T3X/vision.json" --cache "$R/t13_probe/.runtime/vision-cache" --output "$T3X/dev.json"
```

冻结设置后报告一次，并保存逐任务掩码：

```bash
sbatch --gres=gpu:1 --cpus-per-task=2 --mem=24G --time=12:00:00 --output="$T3X/logs/val-%j.out" --error="$T3X/logs/val-%j.err" code/agent_baseline/analysis/run_t3x.sbatch fixed "$T3X/val.json" --tasks "$T3X/tasks/val/tasks.jsonl" --phase val --run-root "$R" --vision-config "$T3X/vision.json" --cache "$R/t13_probe/.runtime/vision-cache" --settings "$T3X/dev.json" --mask-dir "$T3X/fixed_masks" --output "$T3X/val.json"
```

监控作业（JOBID 替换为提交返回值）：

```bash
sacct -j JOBID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS
```

失败时保留日志与缓存，修复明确报错后原命令重跑。输出报告仅在整个阶段成功后写入；失败阶段的部分掩码不构成有效结果，需整阶段成功覆盖后再评分。更改生成规则、图单、示例或后端配置时使用新的输出目录，并重新在开发集选门限。

Agent 正式实验须另行安排；入口在现有 `loop_json` 命令上增加 `--external-examples --tools-t13 --vision-config 配置路径 --tasks 清单路径`，并使用合并后确定的 harness。完成三次 Agent 运行后，用三个配置根进行配对比较：

```bash
sbatch --cpus-per-task=1 --mem=4G --output="$T3X/logs/metrics-%j.out" --error="$T3X/logs/metrics-%j.err" code/agent_baseline/analysis/run_t3x.sbatch metrics "$T3X/metrics.json" --tasks "$T3X/tasks/val/tasks.jsonl" --fixed-dir "$T3X/fixed_masks" --agent-runs "$R/t3x_agent_s1" "$R/t3x_agent_s2" "$R/t3x_agent_s3" --output "$T3X/metrics.json"
```

## 本轮允许的单元验证

登录节点仅运行新加和直接相关的假后端测试，使用 exp5-env 单线程，TMPDIR 位于当前目录。以下命令可独立复现；不运行模型和网络：

```bash
mkdir -p "$PWD/code/.runtime/tmp"
```

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code" OVCD_CODE="$PWD/code/ovcd_v0" TMPDIR="$PWD/code/.runtime/tmp" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m unittest agent_baseline.tests.test_t3x agent_baseline.tests.test_tools_t13 agent_baseline.tests.test_replay -v
```

测试用小尺寸隔离图像覆盖任务生成到清单读取、首条消息、匹配、冻结设置与指标，不代表真实 SECOND 数据或 GPU 验证。G1–G3 的关闭路径与只读 code_t 比较实际编码请求字节；已有 test_tools_t13 保留旧 T3.1 与 G1 回放检查。完整旧测试集未在本轮管理节点执行。
