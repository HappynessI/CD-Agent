# T1.3、T1.2、T3 工具与离线摸底

本批只交付代码。未运行模型、API、Slurm 或真实数据实验。管理节点允许的新增假后端小测试见文末。所有生成路径默认为当前检出目录内；运行根 R 只用于读取图单和既有 SAM3 结果。

## 工具接口与开关

`Toolbox(..., tools_t13=True, vision=backend)` 开启全部六个工具。也可以传工具名列表或逗号分隔字符串。JSON 循环对应 `--tools-t13` 或 `--tools-t13 segment_everything,extract_features,compare_features`，并提供 `--vision-config .runtime/vision_t13.json`。默认 `False`；schema 中没有新工具，execute 也拒绝调用。不开启时 building/any 的 schema、系统提示、验证器提示和请求保持原字节。

| 工具 | 参数 | 返回与含义 |
|---|---|---|
| segment_everything | time=t1/t2 | props#k，提议数、并集面积、分数范围；初始 score=1，仅作占位 |
| extract_features | time=t1/t2 | feat#k，DINOv2 网格形状；模型不参与类别判断 |
| compare_features | props, feat_t1, feat_t2 | 新 props#k，每项 score=1−cos(mean(f1),mean(f2))，范围 0..2；摘要给分数范围 |
| classify | props, time, classes=[字符串,…] | 新 props#k；每个 ID 的各类概率；softmax 词表为 background 加 classes，背景概率保留在后端输出中 |
| segment_semantic | time, text | background 与 text 二分类的稠密 mask#k，逐像素 argmax；不改变当前答案 |
| match_exemplar | props, exemplar={mask:句柄} 或 {point:[x,y]}, mode=change/t1/t2 | 新 props#k，每项 score 是与示例的余弦相似度，范围 −1..1；返回按分数排序的 ID 列表 |

提取与比较保留两个工具。feat# 可以在比较和多个示例检索中复用，不重复运行 DINOv2。变化检索默认用 `mean(f2)−mean(f1)`，候选和示例均保持时间方向。t1/t2 模式用单期外观。零向量的余弦约定为 0；空示例报错。

`props#` 复用 Instances 的不可变稀疏像素表示。`select(set="props#0", ids=[1,3])` 按 ID 选；`select(set="props#0", where="score >= 0.4")` 按当前分数选；输出仍为 props#。`to_mask(sets=["props#1"])` 求并集并设为当前答案。分类结果按概率清单选 ID；classify 保留已有 score，不把它改成类别概率。inspect、view、add_region 复用原操作。同一回复中新建的句柄须下一回复才能使用。

## 后端、环境和缓存

`backends_t13.WorkerPool` 按后端启动独立、串行复用的本地子进程。主循环只导入 NumPy，不导入模型。worker 只用 GPU，未发现 CUDA 时明确失败。JSON 通过标准输入输出传输，数组写入共享缓存；stderr 写入缓存目录。源图、时间、类别词表、后端配置及文件修改时间进入缓存索引。索引和每项缓存加文件锁，可供分片复用。每个脚本结束会关闭 worker；循环退出也关闭。模型失败返回明确错误，不把空掩码当成功。

| 后端 | Python 环境 | 服务器上的现有实现与权重 |
|---|---|---|
| SAM ViT-H | conda_envs/ovcd-de-env | DynamicEarth 的 dynamic_earth.sam_ext.MaskProposal；models/SAM/sam_vit_h_4b8939.pth |
| DINOv2-base | conda_envs/ovcd-de-env | DynamicEarth comparator.bi_match.extract_features；batch56_ovcd_de/weights/hf/dinov2-base |
| SegEarth-OV | conda_envs/ovcd-de-env | identifier.segearth_ov_ext.SegEarth_OV；batch56 的 weights/clip/ViT-B-16.pt 和 weights/simfeatup/xclip_jbu_one_million_aid.ckpt |
| AnyChange H1 | conda_envs/exp5-env | torchange.models.segment_any_change.AnyChange；同一 SAM 权重 |

DynamicEarth 原版在 `/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth`。默认配置使用 batch56 中已做离线权重适配的克隆，路径见 `vision_t13.example.json`，无需访问网络。另有 HF 的 facebook/dinov2-base 缓存，本批没有使用它。

实际导入核对：`PYTHONNOUSERSITE=1` 下 ovcd-de-env 使用 NumPy 1.26.4，MaskProposal、bi_match 能导入；独立导入 AutoImageProcessor/AutoModel 时 transformers/models 目录返回共享文件系统 Input/output error，DINOv2 接口尚未导入成功；SegEarth-OV 在当前管理节点缺少 libGL.so.1，尚未验证计算节点。未设置该变量时用户级 NumPy 2.2.6 污染环境，出现 PyTorch 兼容警告。worker 固定禁用用户级包。AnyChange 在 exp5-env 可以导入。omniovcd-env 缺少 mmseg/mmcv/scipy，不适合运行此后端。以上均未实例化模型；不能据此宣称 GPU 推理通过。

SAM 参数沿用 DynamicEarth demo 和 batch56 M-C-I：points_per_side=32、points_per_batch=64、pred_iou_thresh=.5、stability_score_thresh=.95、stability_score_offset=.9、box_nms_thresh=.7、min_mask_region_area=0；match_hist=False、area_thresh=.8。每个工具只生成一个时相；T1.3 固定路线合并两期提议，不做跨期 NMS，所以与 M-C-I 的两期联合 proposal.forward 有区别。

DINOv2 按原图、不 resize、不 center crop；原版 patch grid 经 align_corners=True 双线性插值后做区域均值。本实现通过累计插值权重获得同一均值，避免两张 H×W×768 特征图占满内存，磁盘只存原始 float32 网格。SegEarth-OV 使用 CLIP ViT-B/16、448×448 预处理、slide_crop=224、stride=112、cls_token_lambda=−.3、logit_scale=50、SimFeatUp。调用整个模型的 eval() 关闭 Dropout。分类概率是 softmax 后对提议区域求均值，未作概率校准。

LEVIR 可直接用 `pool.for_pair(paths, mci=缓存文件)` 读取 `common.MCI/<split>/<stem>.npz` 的 source_time、masks_packed，格式与 queues.read_dump 一致。缓存中的提议经过两期联合 NMS，不与新单期生成路径混记；新特征比较仍计算 1−cos。该入口不会用缓存中的标签或评测结果。

AnyChange 沿用 exp5_anychange.py 的官方 H1：points_per_side=32、stability=.95、change_confidence_threshold=145、use_normalized_feature=True、bitemporal_match=True，不使用检测器或 H2 点查询。`--anychange` 开启参照；后端错误中止并留下 traceback。未开启时报告明确写出未运行原因；若实际不可运行，保留错误日志并以 --anychange-skip-reason "实际原因" 重新运行 T1.3 主路线，将原因写进 JSON/Markdown，不把失败写成零分。

## 验证器范围

新增 `scope_t13.py` 和 `scopes/any_change.json`。`Session(..., target="any change")` 判断不限类别的地类转换；同类替换、阴影、季节和错位按范围卡处理。`Session(..., targets=["tree","water"])` 判断涉及任一所列类别的变化，两个时相不必都属于目标列表。manifest 可用 targets 列表。二者仅支持 change_direction=any。

公开 ask 参数不变。会话属性控制验证器 SYSTEM、VERIFY、CORRESPONDENCE 和规划器范围说明；默认 building 路径仍调用原 prompts 的函数。真值路径、转换类型和数据集名保留在 recorder/离线分析，不进入 Agent 输入。T3 manifest 的 exemplar_mask 单独加载为示例句柄，初始视图叠加轮廓；它不成为初始答案。示例掩码是该任务明确给定的输入。

## 指标与开发、报告分离

所有脚本检查 R/second_dev/names.txt 恰好 60 张、R/second/second_val128.txt 恰好 128 张且无重叠。开发阶段只读取 60 张影像标签。完整开发报告冻结设置；val 必须给出 `--settings`，后端配置也须与开发一致。分片报告不能直接充当设置，须先完整汇总。不得根据 128 张结果更改设置再报告。任务生成规则预先固定，生成脚本不进行质量选择。

T1.3 真值为两期类别不同且两期都非 0。门限网格 0..2，间隔 .02，按开发集累计像素 F1 选，平分取较小门限；128 张只算这一门限。覆盖率=真值与全部提议并集交集/真值面积。像素 oracle 允许完美裁掉提议内假阳性，上限为 2C/(G+C)。整提议 oracle 只允许选择完整提议并求并集：用像素成员模式压缩成原子区域，再用 MILP 求最大 F1。每图限时 60 秒，报告经过证明的 lower/upper 及 exact；限时未收敛不能称为精确最优。逐图界限和全数据集像素 oracle 分开报告，不把逐图 F1 平均成累计 F1。

T1.2 每类单独与 background 做分割。类名固定：low vegetation、non-vegetated ground（标签名 N.V.G surface）、tree、water、building、playground。只在本期 label!=0 区域算 TP/FP/FN，分别报告两期 P/R/IoU。SAM3 的同图单报告并列表格；完整 val 报告必须提供它。开发建议规则预定为两期 IoU 都至少 .25，按较差一期排序。它是可用性筛选，不代表统计显著性。没有通过的类别就报告空列表。

T3.1 每图每种有向转换一个任务。按 8 连通域定义实例；从面积至少 16 的实例中按固定种子 1016 均匀选一个示例。局部种子由图名、转换类型和固定种子决定，与分片及处理顺序无关。真值包含该转换的全部连通域，包括小于 16 像素的实例和示例本身。若没有足够大的示例，该转换不生成任务。生成清单记录每种转换的任务数、示例与全部实例。标签转换类型不写入公开指令。

检索按 DINOv2 变化差向量相似度排序。IoU≥.5 贪心一对一匹配；同分按提议顺序；重复命中算 FP。AP=各首次真命中的 precision 之和/真值实例数，另报 R@1/5/10/20/50 和全部提议召回，任务间宏平均及逐转换统计。包含示例和排除示例两组结果同时报告；排除示例时命中示例的候选仍占排序位置并计 FP，只有一个真值实例的任务在这组平均中不计分。

## 运行手册

以下命令是供计算节点后续运行的手册，本批未提交。所有路径以仓库根为当前目录。当前交付目录已经有 code/；合并到主仓库时只迁移 NOTES 列出的修改。先在计算节点确认 SegEarth-OV 的 libGL.so.1 依赖可用；需修复环境时应修复后端环境，不增加备用实现。

1. 创建本地日志和运行目录。

```bash
mkdir -p .runtime/probe-logs results/agent_baseline/data
```

2. 准备后端配置，核对其中环境和权重路径。

```bash
cp code/agent_baseline/vision_t13.example.json .runtime/vision_t13.json
```

3. 提交前确认 Slurm 控制面；失败时停止提交。

```bash
scontrol ping
```

4. 查看已有作业，避免重复提交。

```bash
squeue -u guisongxia
```

5. T1.3 开发集（含 AnyChange 官方参照）。

```bash
sbatch --output=.runtime/probe-logs/t13-dev-%j.out --error=.runtime/probe-logs/t13-dev-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch t13 --phase dev --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache --anychange
```

6. 上一步成功后只报告 128 张。

```bash
sbatch --output=.runtime/probe-logs/t13-val-%j.out --error=.runtime/probe-logs/t13-val-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch t13 --phase val --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache --anychange --settings results/agent_baseline/data/probe_t13_dev_0of1.json
```

7. T1.2 开发集。

```bash
sbatch --output=.runtime/probe-logs/t12-dev-%j.out --error=.runtime/probe-logs/t12-dev-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch t12 --phase dev --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache
```

8. T1.2 128 张报告，与已有同图单 SAM3 摸底结果并列。

```bash
sbatch --output=.runtime/probe-logs/t12-val-%j.out --error=.runtime/probe-logs/t12-val-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch t12 --phase val --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache --settings results/agent_baseline/data/probe_t12_segearth_dev_0of1.json --sam3-report /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/second/t12_probe.json
```

9. 生成 T3 开发任务清单、示例 PNG 和真值 PNG。

```bash
sbatch --output=.runtime/probe-logs/t3-make-dev-%j.out --error=.runtime/probe-logs/t3-make-dev-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch make-t3 --phase dev
```

10. T3 开发集检索，生成冻结设置。

```bash
sbatch --output=.runtime/probe-logs/t3-dev-%j.out --error=.runtime/probe-logs/t3-dev-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch t3 --phase dev --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache
```

11. 生成 T3 报告集任务，规则与开发集一致。

```bash
sbatch --output=.runtime/probe-logs/t3-make-val-%j.out --error=.runtime/probe-logs/t3-make-val-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch make-t3 --phase val
```

12. T3 报告集检索。

```bash
sbatch --output=.runtime/probe-logs/t3-val-%j.out --error=.runtime/probe-logs/t3-val-%j.err code/agent_baseline/analysis/run_probe_t13.sbatch t3 --phase val --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache --settings results/agent_baseline/data/probe_t3_dev_0of1.json
```

所有入口都支持 `--shard i --shards n`，按图单位置取模分片；默认 0/1。例：将第 5 步追加 `--shard 0 --shards 2`，另一次追加 `--shard 1 --shards 2`。得到全部分片后，将同一命令的分片参数去掉并追加 `--merge 分片0.json 分片1.json`，通过同一 Slurm 入口汇总。T1.3 汇总仍须带 --anychange；val 汇总仍须带 --settings；T1.2 val 汇总还须 --sam3-report。汇总拒绝缺图、重复图、混用阶段和后端配置。任务生成分片各自写独立子目录，其计数可按转换相加；probe 自己用相同确定性规则重建任务，不依赖拼接清单。

脚本也可直接在已分配 GPU 的计算节点运行，例如：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code" /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.probe_t13 --phase dev --vision-config .runtime/vision_t13.json --cache .runtime/vision-cache --anychange --shard 0 --shards 1
```

对应其余三个入口，替换模块为 agent_baseline.analysis.probe_t12_segearth、agent_baseline.analysis.make_t3_tasks（不传 vision-config/cache）、agent_baseline.analysis.probe_t3；完整命令参数见前面的 sbatch 示例。不得在管理节点运行这些实验命令。

| 脚本 | CPU | 内存 | GPU | 粗略时间/每阶段 | 缓存和产物 |
|---|---:|---:|---|---|---|
| probe_t13 | 4 | 32 GB | 1×16–24 GB | 1–4 小时，AnyChange 另加时间 | 缓存约 1–5 GB，JSON+Markdown |
| probe_t12_segearth | 4 | 24–32 GB | 1×12–24 GB | 1–3 小时 | 六类稠密概率约 2–4 GB |
| make_t3_tasks | 1 | 4 GB | 不做模型推理；统一脚本保留单 GPU 配额 | 少于 5 分钟 | 实例清单、掩码、计数 |
| probe_t3 | 4 | 24–32 GB | 1×16–24 GB | .5–2 小时 | 复用 T1.3 缓存 |

以上是预算估计，尚无 GPU 实测。统一脚本默认 gre 分区、4 CPU、32 GB、1 GPU、8 小时，按集群空闲资源调整；短摸底可分片，避免同时抢占优先任务的 GPU。脚本成功或失败都会发 QQ 通知，包含状态、耗时、产物和日志路径；通知失败也会使作业失败。stdout/stderr 置于共享 .runtime/probe-logs，不污染正式结果。用 squeue/sacct 查询 guisongxia 或明确 Job ID。失败时保留缓存与日志，修复依赖后同参数重跑；缓存只原子写入完整结果。输出 JSON/Markdown 写入 results/agent_baseline/data/，可用 --output-root 指定其他明确目录。

## 小测试

只运行新增假后端小测试；不加载模型。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code" TMPDIR="$PWD/.runtime" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m unittest agent_baseline.tests.test_tools_t13 -v
```

测试覆盖冻结 G1 fixture、G2 关闭开关的请求字节、analysis.replay_g1 的合成会话回放、工具句柄、分数与差向量、验证器范围、示例与真值隔离、确定性任务生成、标注区域指标、整提议 oracle 的枚举对照及分片检查。真实历史归档、GPU 推理及完整实验不在本批执行范围。

## 合并版本验证

本目录已合入线 B 的 region#0、T1.1xC2/C4 和 T5.2。Session 同时保留独立区域句柄与多类别属性；Recorder 按完整句柄区分 region_0.png、mask_0.png，并单独保存 feat# 网格。任务读取保留区域与 T5.2 反馈校验、LEVIR 缓存路径检查；循环保留区域输入及候选排除、--sam-prompt-device，并接入新工具与 T3 示例。

在本交付目录检查 code_merged/ 副本时，显式设置两个代码路径，防止导入旧 code/ 副本。下列命令只运行指定的极小隔离测试；单线程，不加载模型。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code_merged" OVCD_CODE="$PWD/code_merged/ovcd_v0" TMPDIR="$PWD/.runtime" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m unittest agent_baseline.tests.test_tools_t13 agent_baseline.tests.test_second_boxes agent_baseline.tests.test_t52 agent_baseline.tests.test_generalize -v
```

前面的实验手册使用主仓库正常的 code/ 布局。完成移植后将 code_merged/ 的内容应用到主仓库 code/，再按手册运行。此追加任务不提交实验，也不修改原 code/ 副本。

合并验收：上述四组在 exp5-env 单线程完整重跑 56 项全部通过，无跳过；日志位于本交付目录 .runtime/merge_tests_final.log。较早尝试中的共享文件系统写入错误及依赖导入异常保留在 NOTES.md；未修改测试或加入异常跳过。
