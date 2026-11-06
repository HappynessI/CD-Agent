# SECOND 建筑方向迁移准备

本次只交代码，不跑实验。所有数据处理、模型、API 和统计命令均在已分配的计算节点执行。登录节点只编辑、读少量文件、跑合成数组小测试。仓库根目录是当前稀疏克隆；不操作主克隆，不提交 Git。

## 已核对的数据事实

数据根为 `/guisongxia01/pangchao/wangyihan/datasets/lim1/SECOND`。`dataset_index.json` 登记 2968 张 `train_*.png`、1694 张 `test_*.png`。两期影像在 `img1/img2`，灰度语义标签在 `slabel1/slabel2`，另有对应 RGB 标签和二值变化标签目录。只读首张验证图的 PNG 头确认两期 RGB 和两期灰度标签均为 512×512、8 bit。另轻量读取同一张验证图：两期 0 支持完全一致，类别不同区域与二值变化标签完全一致；T1 值为 0/1/2/3，T2 值为 0/2/3/5。未遍历全数据像素。

`cls_gray.json`：0=non-change，1=low vegetation，2=N.V.G surface，3=tree，4=water，5=building，6=playground。`cls_rgb.json` 的未变化色为白色，建筑为 `[128,0,0]`。0 表示未变化且没有地类标注，不是背景地类。因此 T1.2 只在各期非零语义标签区域评分；不能解释为全图分割质量。`second_labels.py` 检查编码、数组形状、0 支持一致性。

真值：new=`T2==5 & T1!=5 & changed`；demolished=`T1==5 & T2!=5 & changed`；c0 为两者并集。changed 要求两期非零且类别不同。建筑→建筑（包括原址换楼）不计入。图可仅有一个方向变化，另一方向真值自然为空。

验证来源确认：

- `code/probe/exp8_common.py` 从旧 probe 导入数据接口，不单独定义验证图单。
- 主克隆 `code/probe/exp8s2_judge.py::image_list('second_cv')` 明确用 `exp6/second_cv/manifest.json` 的 `verifier_val_images` 核对 297 张图。
- `exp8s3_common.py` 将非 test 的 SECOND 映射为 `second_cv`，`exp8s3_audit.py` 从 `exp8_batch22/protocol.json` 取 `val_images`。
- 只读核对上述 protocol 的 297 张 val、`exp8s_batch30/interim/exp8s/second_cv/manifest.json` 的图单，以及 `/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/exp6/second_cv/verifier_val.txt`：排序完全一致。后者也与 `exp6/second_cv/manifest.json` 的 `verifier_val_images` 完全一致，seed=1016，train/val=2671/297。
- `doc/exp8_object_pool.md` 和 `doc/exp8s3_stage1_judge.md` 说明验证沿用交叉拟合划分。新准备代码只读独立文本图单和原数据，不导入旧 exp8 模块，不读旧候选、预测或池目录。文本文件是普通文件，不是符号链接。旧 exp8 的其他功能确实依赖旧池，但新流程不依赖它们。

固定抽样：对 297 张验证图筛选有任意建筑变化的图，按名称排序，用 NumPy `default_rng(1016)` 不放回抽 128 张，再按名称排序。少于 128 张直接报错。正式图单只由计算节点生成后交 Claude 入库，本次没有生成占位图单。

## 冻结设置和接口

像素门限 0.85，实例门限 0.5，SAM3 检索置信门限 0.1，512 分块，匹配 overlap 0.25；不在新数据调参。缓存由现有 SAM3 worker 的源路径命名空间存储，准备、Agent 和固定路线共用。worker 在原浮点像素值上用 0.85 二值化；旧 LEVIR 软缓存的读取和默认路径原样保留。

新任务为 `T1.1xC1`、`T5.1xC1`；T5 必须有给定掩码。五份清单（T1 新建/拆除、T1 C0、T5 新建/拆除）共用图单。指令是第 10.6 节原文。`scope=building_rebuild_excluded`，target 仍为 building。卡里明确原址换楼答 no、类型 rebuilt；方向 counts 分别只有新建或拆除。未指定 scope 仍用原 building 卡，其文字不变。默认 schema、指令、提示词由 G1 fixture 检查。

两个方向使用不同 config。`loop_json.py` 会拒绝同 config 下同任务同图的双方向清单，防止输出覆盖。范围卡名、卡的文字都不含数据集名。图路径、GT、meta 留在 harness；不进入规划器或验证器消息。本地 ask worker 可固定 `--scope`，混合范围清单优先逐会话配置的 API ask。

## 计算节点运行手册

以下命令都在仓库根、已分配的计算节点执行。每一步是一条命令；先设置公共环境。CPU/内存/GPU 为保守申请估计，未实测；没有给出新数据上的速度或费用实测值。轻量 CPU 脚本使用含 NumPy/SciPy/Pillow 的 Python；SAM3 本地步骤用现有 `omniovcd-env`。API 认证沿用已有环境配置，文档不保存认证细节。

1. 设置运行根和解释器。CPU 1，内存 1 GiB，GPU 0。`AGENT_RUN` 应指向共享盘；后续保持同一值。

```bash
export PYTHONPATH="$PWD/code" AGENT_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run PY=/guisongxia01/pangchao/wangyihan/conda_envs/ovcd-rea-env/bin/python SAM_PY=/guisongxia01/pangchao/wangyihan/conda_envs/omniovcd-env/bin/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

2. 合成数据与冻结回归。CPU 1，内存 2 GiB，GPU 0；无模型/API。若集群解释器不可用，选有上述依赖的解释器替换 `$PY`。

```bash
"$PY" -m unittest agent_baseline.tests.test_second agent_baseline.tests.test_generalize agent_baseline.tests.test_loop agent_baseline.tests.test_replay -v
```

3. 抽样并生成五份清单和三种真值。CPU 1，内存 2 GiB，GPU 0。检查 `second/preparation.json` 和 128 行图单；已有不同图单会拒绝覆盖。

```bash
"$PY" -m agent_baseline.analysis.make_second_tasks --run-root "$AGENT_RUN" --names-output results/agent_baseline/data/second_val128.txt
```

4. 本地 SAM3 预缓存并生成不分方向给定掩码。CPU 4，主机内存 16 GiB，GPU 1（建议 24 GiB 显存起，按现有 SAM3 环境申请）。默认单分片完整准备；`--shard/--shards` 可拆分，分片输出在 `second/prepare_shards/`，分片结束后用第 5 步 any 路线生成完整掩码。重复执行复用缓存。

```bash
"$SAM_PY" -m agent_baseline.analysis.second_prepare --tasks "$AGENT_RUN/second/tasks/T1.1xC0_any.jsonl" --run-root "$AGENT_RUN" --local
```

已有 SAM3 worker 时可用 `--sam3-worker "$SAM3_WORKER"` 替代 `--local`。worker 需与读取方共享同一个 `AGENT_RUN` 和源文件路径。prepare 成功后 Agent 的 `segment_text(building,tile=512,score_thr=.5)` 和固定路线可直接命中缓存；其他文本/阈值需要已准备的对应缓存或 worker。

5. 三条固定路线及逐图 TP/FP/FN。CPU 1，内存 4 GiB，GPU 0（准备完整后）。any 输出同时是 T5 给定掩码；不能按方向重新构造 T5 输入。

```bash
for d in any appeared disappeared; do t=T1.1xC1; [ "$d" != any ] || t=T1.1xC0; "$PY" -m agent_baseline.fixed_routes imc --tasks "$AGENT_RUN/second/tasks/${t}_${d}.jsonl" --direction "$d" --pixel-thr 0.85 --output-dir "$AGENT_RUN/second/fixed/$d" || break; done
```

如第 4 步使用了分片，在本步前用同一条 any 路线、输出目录改为 `$AGENT_RUN/init/second_sam3_imc_p85/val`，生成 T5 给定掩码。显式图对入口也可用：`fixed_routes imc --image-t1 PATH --image-t2 PATH --gt-mask GT --direction appeared --scope building_rebuild_excluded --pixel-thr 0.85 --output-dir DIR`。GT 只用于结果统计。

6. 带方向规则。CPU 2，内存 4 GiB，GPU 0；会调用 flash API。每方向独立预算与答案目录，可续跑；只删除 answer=no，未解析答案保留并计数。费用上限是已有 RunGuard 的请求间检查，存在在途请求的少量超额可能。

```bash
for d in appeared disappeared; do "$PY" -m agent_baseline.analysis.second_rule --tasks "$AGENT_RUN/second/tasks/T1.1xC1_${d}.jsonl" --mask-dir "$AGENT_RUN/second/fixed/$d" --output-dir "$AGENT_RUN/second/rule" --max-cost-cny 10 --concurrency 4 || break; done
```

模型、max_tokens、logprobs、JSON、四幅近景/上下文图、seed 公式均复用 `ask_eval.py` 的 `f37_corr`。只注入会话的方向和范围卡，不在新数据改问法。答案是 `second/rule/<direction>/answers.jsonl`，掩码是其 `masks/`。

7. 先做 10 张 Agent 试跑，两个任务、两个方向都覆盖。CPU 4，内存 8 GiB，GPU 1。没有预存 SAM 编码的图，`segment_prompt` 每次编码整张图：CPU 单线程实测约 150 秒，GPU 0.4–0.8 秒（每个进程约 6 GB 显存，几个进程共用 1 张卡）。不要用 GPU 0 跑。本条顺序跑四组，不超过两个 Agent 作业并发；规划器和验证器共用 flash 限额。沿用 G1 的 v4、status、固定核验，Python/skill/plan 关闭；新配置加 gate2。若 G2 尚未验收或不如 G1，去掉 `--gate2` 并把 config 前缀换成 `second_g1`，记录原因。

```bash
for t in T1.1xC1 T5.1xC1; do for d in appeared disappeared; do AGENT_ASK_SLOTS=3 "$PY" -m agent_baseline.loop_json --tasks "$AGENT_RUN/second/tasks/${t}_${d}.jsonl" --split val --config "second_smoke_g2_${d}_s1" --limit 10 --seed-offset 1 --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --batch second_smoke_g2 --max-cost-cny 10 || exit; done; done
```

人工核对范围、两期图、方向、三种 GT 和给定掩码。试跑结果保留在独立 config；不用试跑调门限或问法，不混入三次正式重复。

8. 试跑核对后，五个子任务各三次。CPU 4，内存 8 GiB，GPU 1（理由见第 7 步）；当前命令顺序执行，最多一个 Agent 进程。1536 个 C1 会话按文档旧单价估约 94 元，C0 另算，此处共享估算预算上限 140 元，实际以账本为准。续跑用相同 config 跳过已有 final；检查账本停止原因后才处理对应 STOP 文件。

```bash
for n in 1 2 3; do for t in T1.1xC1 T5.1xC1 T1.1xC0; do ds='appeared disappeared'; [ "$t" != T1.1xC0 ] || ds=any; for d in $ds; do AGENT_ASK_SLOTS=3 "$PY" -m agent_baseline.loop_json --tasks "$AGENT_RUN/second/tasks/${t}_${d}.jsonl" --split val --config "second_g2_${d}_s$n" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --batch second_g2 --max-cost-cny 140 || exit; done; done; done
```

9. 写出汇总输入 spec。CPU 1，内存 1 GiB，GPU 0。若正式 config 改为 G1，设置 `--agent-prefix second_g1`。spec 显式列出五组清单、三个重复目录和基线目录，便于审查。

```bash
"$PY" -m agent_baseline.analysis.second_metrics --write-spec --run-root "$AGENT_RUN" --agent-prefix second_g2 --output "$AGENT_RUN/second/metrics_spec.json"
```

10. 汇总 JSON 和一张 Markdown 表。CPU 1，内存 4 GiB，GPU 0。三次各自累计 TP/FP/FN，报告 F1/P/R 三次均值和逐次值；方向违反率为反方向 GT 上的预测像素/所有预测像素（空预测记 0）。T5 另报 ΔF1 given。对固定路线、规则分别给按图配对的 4000 次 bootstrap 区间，三次重复固定、共享抽样图索引；区间只含图像抽样变化。缺 final 或必需基线会报错；已有错误/兜底 final 保留并计数。另报在线 rebuilt 回答总次数、区域数和缺账本数，以及离线去重 rebuilt 区域数。在线重复核验不去重，避免把“回答次数”和“区域数”混为一谈。

```bash
"$PY" -m agent_baseline.analysis.second_metrics --spec "$AGENT_RUN/second/metrics_spec.json" --output "$AGENT_RUN/second/metrics.json"
```

11. T1.2 离线摸底。CPU 4，主机内存 16 GiB，GPU 1（建议 24 GiB 显存起）。不调用 API，使用同一份 128 图名单；逐类两期分割，建筑复用已有缓存，其他类别走同一 worker 缓存。每期每类单独报 P/R/IoU，只在非零语义区域算，分母为 0 记 null。提示词固定为 water、tree、low vegetation、non-vegetated ground、playground、building，不据结果挑问法。

```bash
"$SAM_PY" -m agent_baseline.analysis.second_t12_probe --names-file results/agent_baseline/data/second_val128.txt --run-root "$AGENT_RUN" --local --output "$AGENT_RUN/second/t12_probe.json"
```

12. 完整 G1 历史离线回放须在计算节点执行，CPU 2，内存 4 GiB，GPU 0。下面针对一个已知归档 config；其他两次重复与两个任务同样回放，实际 config 按 `$AGENT_RUN` 中的归档目录选择。脚本拦截网络，验证器请求由冻结 code_g 重建；不初始化模型、不读原始 GT。输出必须在归档目录外。本次只跑合成回放单测，不宣称重新验过全部 384 个历史会话。

```bash
"$PY" -m agent_baseline.analysis.replay_g1 --config json_qwen37flash_g1_s1 --task T5.1 --split val --reference-code "$AGENT_RUN/code_g" --output-dir "$AGENT_RUN/second/regression/g1_s1_T5"
```

仅验证集准备和运行；测试集等待 Ethan 决定。没有生成模型结果、费用实测值、正式 128 图单或对新数据的性能结论。

轨迹展示也识别 T5.1xC1 的给定掩码，并从 input.json 的 gt_mask_path 读参照。通用 diagnose 可传 --tasks 和 --fixed-dir；SECOND 的正式多方向统计仍用 second_metrics。

## 线 B：T1.1 × C2 和 T1.1 × C4

C2 是框内任意建筑变化（direction=any）；C4 是框内新建（direction=appeared）。两者均使用 `scope=building_rebuild_excluded`。指令按 README_ai.md 第 10.7 节原文填写，只替换整数坐标。框采用 `[x0,y0,x1,y1]`，右界 x1 和下界 y1 为开区间，和工具 bbox 一致。真值与框逐像素取交集，跨边界建筑会被截断。

清单可选 `region_mask` PNG 路径按清单位置解析。会话登记 `region#0`，它是 Mask，可用于 `combine(op="intersect", a=答案句柄, b="region#0")`、`view(overlay="region#0")` 等。第一轮消息显示面积、bbox、开区间说明；路径、数据集名和真值不进入规划器输入。它不占用普通句柄编号，也不作为给定掩码、当前答案或自动核验候选。Agent 显式提交它时按普通掩码提交和核验。轨迹保存为 `masks/region_0.png`，与 `masks/mask_0.png` 分开。工具列表不增加，Python 保持关闭；没有区域字段的旧任务沿用原消息和编号。

以下命令均在仓库根、已经分配的计算节点执行，一步一条命令。资源为估计值。这里交付代码和极小单测；未生成正式框文件，未执行正式数据处理、模型或 API。`results/agent_baseline/data/second_boxes.json` 由第 3 步生成后纳入交付；不能用未核对真值的占位框替代。正式运行前等待线 A 结论；若采用新分割设置，先重算缓存、固定路线、规则及历史 C0/C1，使用新目录和前缀，旧结果保留。Agent 作业总并发最多两个。

1. 公共环境。CPU 1，内存 1 GiB，GPU 0。`BOX_BASE` 用新的输出目录，改变来源后也换目录。历史参照前缀 `HISTORY_PREFIX` 可替换为线 A 重跑配置。

```bash
export PYTHONPATH="$PWD/code/ovcd_v0:$PWD/code" AGENT_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 BOX_BASE=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/second/box_baselines HISTORY_PREFIX=second_g2
```

2. 极小单测。CPU 1，内存 2 GiB，GPU 0。覆盖旧 SECOND、小数组框及真实临时文件流程，无模型/API。完整 G1 fixture 和 replay 回归仍由计算节点执行原手册的测试命令及历史回放命令。

```bash
"$PY" -B -m unittest agent_baseline.tests.test_second agent_baseline.tests.test_second_boxes -v
```

3. 生成固定框、区域 PNG、裁剪真值和两个清单。CPU 1，内存 2 GiB，GPU 0。读取现有 `second/second_val128.txt`、C0 清单及 `gt/{c0,new}`；要求恰好 128 图。随机数为 NumPy default_rng(1016)，按名称排序，先固定抽 64 张，再按图依次生成 C2/C4 框。宽高分别在 `[ceil(.4×边长), floor(.7×边长)]` 整数中均匀采样；位置均匀。要求相交的框整个重采样，空真值记录豁免。JSON 记录每框的要求、实际交集、尺寸及尝试次数。预检查全部输出；任意已有不同内容即报错，同内容可重复执行。

```bash
"$PY" -B -m agent_baseline.analysis.make_second_boxes --run-root "$AGENT_RUN" --names-file "$AGENT_RUN/second/second_val128.txt" --seed 1016 --output results/agent_baseline/data/second_boxes.json
```

输出：`second/regions/{c2,c4}/`、`second/gt/{c2,c4}/`、`second/tasks/T1.1xC2_any.jsonl`、`second/tasks/T1.1xC4_appeared.jsonl`。框 JSON 无需保存运行路径，可跨克隆复用。运行路径在清单内，仅由 harness 读取。

需要保持已有运行根只读时，增加 `--source-run-root 原运行根`，将 `--run-root` 指向新的可写目录。图单、C0 清单和原始 c0/new 真值从来源目录读取，区域 PNG、裁剪真值和两个清单只写入新的运行根；`--output` 指定框 JSON 的可写位置。此模式生成的框内容与原地生成完全一致。后续 C4 指标的 `demolished_dir` 仍须指向含完整拆除真值的来源目录。

4. 选择最终分割设置对应的对照来源。CPU 1，内存 1 GiB，GPU 0。线 A 采用时替换这里四个目录及第 1 步历史前缀；不改框文件。

```bash
export FIXED_ANY="$AGENT_RUN/second/fixed/any" FIXED_APPEARED="$AGENT_RUN/second/fixed/appeared" RULE_ANY="$AGENT_RUN/second/rule/any/masks" RULE_APPEARED="$AGENT_RUN/second/rule/appeared/masks"
```

5. 生成 C2 不分方向的规则掩码。CPU 2，内存 4 GiB，GPU 0；调用现有 flash API，预算 10 元、并发 4。`second_rule.run` 和 CLI 已支持 scope 和 direction any，无需改算法。只删 answer=no，其余保留。方向和范围写入 settings，输入改变时使用新的 output-dir，并同步修改 `RULE_ANY`。C4 复用已有 appeared 规则；线 A 改动时须先重算。

```bash
"$PY" -B -m agent_baseline.analysis.second_rule --tasks "$AGENT_RUN/second/tasks/T1.1xC0_any.jsonl" --mask-dir "$FIXED_ANY" --output-dir "$AGENT_RUN/second/rule" --scope building_rebuild_excluded --direction any --max-cost-cny 10 --concurrency 4
```

6. 所有对照取框内交集。CPU 1，内存 2 GiB，GPU 0；无 API。C2 为 fixed_any、rule、posthoc_s1/s2/s3；C4 另外有 fixed_direction。历史 C0 来自 `${HISTORY_PREFIX}_any_s{1,2,3}/T1.1xC0/val`，C1 新建来自 `${HISTORY_PREFIX}_appeared_s{1,2,3}/T1.1xC1/val`。源 final 缺失即报错；已有错误或空兜底照常计入，来源诊断写到 preparation.json。

```bash
"$PY" -B -m agent_baseline.analysis.second_box_baselines --run-root "$AGENT_RUN" --fixed-any-dir "$FIXED_ANY" --fixed-appeared-dir "$FIXED_APPEARED" --rule-any-dir "$RULE_ANY" --rule-appeared-dir "$RULE_APPEARED" --agent-prefix "$HISTORY_PREFIX" --output-dir "$BOX_BASE"
```

7. 10 张试跑，顺序运行两个子任务。CPU 4，内存 8 GiB，GPU 1（建议 24 GiB 显存）。没有预存 SAM 编码时 `segment_prompt` 需要编码整张图，GPU 约 0.4–0.8 秒，CPU 单线程约 150 秒；这些是现有 C0 运行观察，未对 C2/C4 实测。沿用 G2 的 v4/status/verify-stage/gate2、flash 模型、pixel-thr=.85、seed-offset=1、concurrency=6、ASK_SLOTS=3。Python/skill/plan 不开启。

```bash
for k in c2 c4; do t=T1.1xC2; d=any; [ "$k" != c4 ] || { t=T1.1xC4; d=appeared; }; AGENT_ASK_SLOTS=3 "$PY" -B -m agent_baseline.loop_json --tasks "$AGENT_RUN/second/tasks/${t}_${d}.jsonl" --split val --config "second_smoke_g2_${k}_s1" --limit 10 --seed-offset 1 --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --batch second_boxes_smoke_g2 --max-cost-cny 10 || exit; done
```

检查首轮 region 面积和 bbox、区域 PNG、GT 与框交集、方向范围及 combine 的结果。查看轨迹中工具耗时，确认 GPU 编码。试跑用独立配置，不混入正式三次重复，也不用结果调整框或门限。

8. 线 A 结论确定且试跑检查通过后全量：2 × 128 × 3 = 768 个会话。CPU 4，内存 8 GiB，GPU 1（原因见第 7 步）。顺序执行，config 为 `second_g2_c2_s{1,2,3}` 和 `second_g2_c4_s{1,2,3}`。约 47 元的估计沿用既有单价，预算上限 60 元；实际看账本。原 G2 的回复预算和提交门逻辑保持不变。相同 config 可续跑。

```bash
for n in 1 2 3; do for k in c2 c4; do t=T1.1xC2; d=any; [ "$k" != c4 ] || { t=T1.1xC4; d=appeared; }; AGENT_ASK_SLOTS=3 "$PY" -B -m agent_baseline.loop_json --tasks "$AGENT_RUN/second/tasks/${t}_${d}.jsonl" --split val --config "second_g2_${k}_s$n" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate2 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --batch second_boxes_g2 --max-cost-cny 60 || exit; done; done
```

9. 写出统计输入 spec。CPU 1，内存 1 GiB，GPU 0。`--agent-prefix second_g2` 对应新 C2/C4 配置；历史前缀已在第 6 步使用。spec 可显式替换目录，保留三个不同重复目录。

```bash
"$PY" -B -m agent_baseline.analysis.second_box_metrics --write-spec --run-root "$AGENT_RUN" --baseline-root "$BOX_BASE" --agent-prefix second_g2 --output "$AGENT_RUN/second/box_metrics_spec.json"
```

10. 全量统计。CPU 1，内存 2 GiB，GPU 0。JSON 和同名 Markdown 表同时生成。Agent 预测按原样评分，不自动裁剪，所以能观察框外违反。P/R/F1 按每次重复累计像素后计算，再报三次均值和逐次值。框外违反率为框外预测像素/全部预测像素；C4 方向违反率为完整拆除真值上的预测像素/全部预测像素，空预测均记 0。每个对照都有 Agent 减对照的配对 bootstrap 4000 次、种子 1016、95% 区间；三次重复共享抽样图索引，事后裁剪参照也保留三个重复。缺 final 或必需掩码会报错，错误/兜底会话逐次报告。

```bash
"$PY" -B -m agent_baseline.analysis.second_box_metrics --spec "$AGENT_RUN/second/box_metrics_spec.json" --output "$AGENT_RUN/second/box_metrics.json"
```

全部命令只是运行手册，本次没有执行。正式框 JSON、10 张模型试跑、768 个会话、全量 fixture/replay 及性能结论仍需计算节点验收。
