# T1.2、C1 转换与 T1.3 问法选择

本批写代码，不运行实验。下面的计算、标签生成和 API 命令由合并者通过 Slurm 执行。所有路径以仓库根为当前目录。默认新会话属性为空；新任务须显式提供清单。原有工具开关仍默认关闭，公开工具参数不变。

## 真值、指令与范围

SECOND 灰度标签：0=未变化且无地类标注；1=low vegetation；2=N.V.G surface（正式英文 non-vegetated ground surface）；3=tree；4=water；5=building；6=playground。读取 slabel1/slabel2，并核对 cls_gray.json。两期非零支持不一致直接报错。0 不当作裸地。

T1.2：两期均非零、类别不同，并且任一期属于 {1,5}。指令是 `Find changes involving buildings or low vegetation between T1 and T2. Output a binary change mask.`。清单含 `targets=['building','low vegetation']`；read_tasks 将其保留为会话 target 元组，使用已有多类别验证器路径。`needs_init` 与 T1.1xC0 一样为 False；清单默认 init_mask=null，从零生成。若显式给 init_mask，应填冻结的本任务固定路线输出；循环会加载它，生成器不另造给定掩码。

C1t：T1∈{1,2} 且 T2=5。反方向为 T1=5 且 T2∈{1,2}。分别保存 gt/new 和 gt/demolished，后者只用于方向违反分析。指令是 `Find buildings newly built on low vegetation or bare ground between T1 and T2. Output a binary change mask.`；target=building，change_direction=appeared。新增 source_classes 属性是两个正式英文类名。它进入规划器范围卡和验证器 SYSTEM 中的附加范围条款；VERIFY 正文沿用 SECOND C1。图单、标签和数据集名不进入请求。

生成器复用 R/second_dev/names.txt 的 60 张及 R/second/second_val128.txt 的 128 张，不抽样、不剔除空真值。preparation.json 记录每组非空图数。清单名分别为 T1.2xC0_any.jsonl、T1.1xC1t_appeared.jsonl；开发清单加 `_dev`。

## 固定路线与选择

T1.2 的每类分别调用两期 segment_semantic 等价实现（同一 worker 的 classify 单类加 background、argmax），取异或，再对类别并集。两期完全对称。另提供 building 使用 SECOND SAM3 IMC any 输出的设置。小组件面积候选为 0、16、64，单位为像素。所有类别词固定为已有摸底的 building、low vegetation。

C1t 输入是 SECOND SAM3 IMC appeared：tile=512、pixel_thr=.85、score_thr=.5、overlap=.25。与 T1 的 low vegetation 和 non-vegetated ground 两次二分类掩码并集求交。分割文本 `non-vegetated ground` 与已有摸底保持一致，对应正式类别名 non-vegetated ground surface。候选为逐像素求交，或组件内 T1 地类覆盖率至少 .25/.5 时保留整组件；再选 0/16/64 面积过滤。另输出 unintersected，不参与求交方式选择。

完整 60 张按累计像素 F1 选择；同分取声明顺序第一项。report JSON 冻结 selected、setups 和后端配置。128 张只应用这一项，C1t 同时报告 unintersected。输出 `R/<t12|c1t>/fixed/<设置>/<dev|val>/`。支持 --shard/--shards；分片按图单步进分配，重跑跳过已有完整 PNG。分片开发报告不能充当设置，必须 --merge 完整汇总。后端配置或已有目录的设置改变会报错，应另设输出目录。

诊断包括求交前像素数、求交前 TP、求交后 TP，以及把候选中全部 TP 切掉的图片数。检查开发报告的 diagnostics，比较 unintersected 的 F1/P/R；若求交路线明显不合理，先记录原因并复核工具，禁止按 128 张重新选择设置。尚未运行真实掩码，不能宣称裁剪损失已可接受。

## 环境、资源与费用

| 步骤 | 主进程环境 | CPU / 内存 / GPU | 估计 |
|---|---|---|---|
| 标签与清单 | exp5-env | 1 / 4 GiB / 0 | 数分钟，无 API |
| SAM3 缓存 | omniovcd-env | 4 / 24 GiB / 1，建议显存 24 GiB | 约 1–3 小时，无 API |
| SegEarth-OV 固定路线 | exp5-env 启动 ovcd-de-env worker | 4 / 32 GiB / 1，显存至少 16 GiB | 每组约 1–4 小时，无 API |
| T1.3 缓存重建 | exp5-env | 1 / 8 GiB / 0 | 约数分钟至一小时，无 API |
| 规则及问法 | exp5-env | 1 / 8 GiB / 0 | 一组件一请求；主要等待 API |
| 指标和 bootstrap | exp5-env | 1 / 8 GiB / 0 | 数分钟，无 API |

SegEarth-OV、SAM ViT-H、DINOv2-base 共用 README_tools_t13.md 的 ovcd-de-env。配置复制 vision_t13.example.json 后核对实际路径。本批不重复验证这些模型环境；缓存缺失或模型依赖失败会明确停止。禁止用缺失缓存当空预测。

API 均复用 SECOND second_rule 的 f37_corr：qwen3.7-flash、temperature=0、max_tokens=256、logprobs、top_logprobs=5、JSON、四幅近景/上下文图及同一组件 seed。仅切换范围和问法。估计每次输入 8,000、输出 128 token，按现有价格约 0.0017024 元。C 个组件约 C×0.0017024 元；1,000 次约 1.70 元，5,000 次约 8.51 元。T1.2 与 C1t 各约一次组件总数；T1.3 开发集三种约 3C 次、C=5,000 时合计约 25.54 元；报告集只选中一种。真实图像 token 和账单可能不同。--max-cost-cny 必填；RunGuard 按请求间检查，存在当前请求或在途请求的少量超额。T1.3 串行，三种共用一个预算。规则示例 concurrency=1。

错误时保留已写 answers.jsonl，修复后原命令续跑。费用停止后，先核对账本和新的授权总额，再删除对应 `budget/ledger/STOP.json` 并提高 --max-cost-cny；历史账本保留，续跑继续累计。全局 STOP.json 的配额错误须先解决，不自动清除。改变掩码或问法要另用输出目录。

## 命令

先完成合并。以下步骤有依赖，前一步成功后再提交下一步。脚本默认 CPU 作业；GPU 步骤显式申请 GPU。长作业退出时发送 QQ 状态、耗时和产物路径，错误看 Slurm traceback。CPU1 是单线程；实际耗时未测。

1. 设置环境和运行根。

```bash
export R=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run AGENT_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python PYTHONPATH="$PWD/code" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

2. 创建日志和本地临时目录。

```bash
mkdir -p "$R/t12/logs" "$R/c1t/logs" "$R/t13_ask/logs" "$PWD/code/.runtime/tmp"
```

3. 检查控制面。失败时停止提交。

```bash
scontrol ping
```

4. 检查重复作业。

```bash
squeue -u guisongxia
```

5. 准备后端配置。

```bash
cp code/agent_baseline/vision_t13.example.json code/.runtime/vision_t13.json
```

6. 生成 T1.2 两份清单及真值。

```bash
sbatch --output="$R/t12/logs/prepare-%j.out" --error="$R/t12/logs/prepare-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch make-t12 "$R/t12" --run-root "$R"
```

7. 生成转换任务及反方向真值。

```bash
sbatch --output="$R/c1t/logs/prepare-%j.out" --error="$R/c1t/logs/prepare-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch make-c1t "$R/c1t" --run-root "$R"
```

8. 合并两份转换清单供 SAM3 缓存和原始固定路线使用。此步只连接清单文本，可在管理节点执行。

```bash
cat "$R/c1t/tasks/T1.1xC1t_appeared_dev.jsonl" "$R/c1t/tasks/T1.1xC1t_appeared.jsonl" > "$R/c1t/tasks/all.jsonl"
```

9. 若建筑缓存未齐，计算节点补齐。已有完整缓存可直接下一步。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=08:00:00 --export=ALL,PYTHON_BIN=/guisongxia01/pangchao/wangyihan/conda_envs/omniovcd-env/bin/python --output="$R/c1t/logs/sam3-%j.out" --error="$R/c1t/logs/sam3-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch sam3 "$R/init/second_sam3_imc_p85/val" --tasks "$R/c1t/tasks/all.jsonl" --run-root "$R" --local
```

10. 建筑 C0 参照。这里只使用输出掩码；脚本附带的转换真值分数不作为 T1.2 指标。

```bash
sbatch --output="$R/t12/logs/sam3-any-%j.out" --error="$R/t12/logs/sam3-any-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch sam3-fixed "$R/references/any" imc --tasks "$R/c1t/tasks/all.jsonl" --direction any --pixel-thr .85 --output-dir "$R/references/any"
```

11. 建筑 appeared 参照。

```bash
sbatch --output="$R/c1t/logs/sam3-appeared-%j.out" --error="$R/c1t/logs/sam3-appeared-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch sam3-fixed "$R/references/appeared" imc --tasks "$R/c1t/tasks/all.jsonl" --direction appeared --pixel-thr .85 --output-dir "$R/references/appeared"
```

12. 开发集 T1.2 设置选择，含 SAM3 建筑选项。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --output="$R/t12/logs/fixed-dev-%j.out" --error="$R/t12/logs/fixed-dev-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch t12-fixed "$R/t12/fixed" --phase dev --run-root "$R" --vision-config code/.runtime/vision_t13.json --cache "$R/vision-cache" --sam3-mask-dir "$R/references/any"
```

13. 开发集转换求交方式选择。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --output="$R/c1t/logs/fixed-dev-%j.out" --error="$R/c1t/logs/fixed-dev-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch c1t-fixed "$R/c1t/fixed" --phase dev --run-root "$R" --vision-config code/.runtime/vision_t13.json --cache "$R/vision-cache" --sam3-mask-dir "$R/references/appeared"
```

如需分片，第 12 或 13 步每个作业追加 --shard K --shards N。全部完成后同参数追加 `--merge report_0ofN.json ... report_N-1ofN.json`，改用 CPU 作业；完整输出名为 report_0of1_merged.json，后续 --settings 指向它。并列设置与后端配置必须一致。

14. 应用冻结的 T1.2 设置，报告 128 张一次。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --output="$R/t12/logs/fixed-val-%j.out" --error="$R/t12/logs/fixed-val-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch t12-fixed "$R/t12/fixed" --phase val --run-root "$R" --vision-config code/.runtime/vision_t13.json --cache "$R/vision-cache" --sam3-mask-dir "$R/references/any" --settings "$R/t12/fixed/dev/report_0of1.json"
```

15. 应用冻结的转换设置，同时输出不求交参照。

```bash
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --output="$R/c1t/logs/fixed-val-%j.out" --error="$R/c1t/logs/fixed-val-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch c1t-fixed "$R/c1t/fixed" --phase val --run-root "$R" --vision-config code/.runtime/vision_t13.json --cache "$R/vision-cache" --sam3-mask-dir "$R/references/appeared" --settings "$R/c1t/fixed/dev/report_0of1.json"
```

16. 读取开发集选中的设置名称，仅解析 JSON。

```bash
export T12_SETTING="$("$PY" -c 'import json,os; print(json.load(open(os.environ["R"]+"/t12/fixed/dev/report_0of1.json"))["selected"])')" C1T_SETTING="$("$PY" -c 'import json,os; print(json.load(open(os.environ["R"]+"/c1t/fixed/dev/report_0of1.json"))["selected"])')"
```

17. T1.2 规则。示例上限 10 元，预计 C×0.0017024 元。

```bash
sbatch --output="$R/t12/logs/rule-%j.out" --error="$R/t12/logs/rule-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch rule "$R/t12/rule" --tasks "$R/t12/tasks/T1.2xC0_any.jsonl" --mask-dir "$R/t12/fixed/$T12_SETTING/val" --output-dir "$R/t12/rule/$T12_SETTING" --max-cost-cny 10 --concurrency 1
```

18. 转换规则。复用 second_rule，source_classes 自动进范围卡。示例上限 10 元，估算同上。

```bash
sbatch --output="$R/c1t/logs/rule-%j.out" --error="$R/c1t/logs/rule-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch rule "$R/c1t/rule" --tasks "$R/c1t/tasks/T1.1xC1t_appeared.jsonl" --mask-dir "$R/c1t/fixed/$C1T_SETTING/val" --output-dir "$R/c1t/rule/$C1T_SETTING" --max-cost-cny 10 --concurrency 1
```

开发集规则对应把清单换为 `_dev.jsonl`、掩码路径换为 dev、输出目录单独设为 rule_dev，预算独立。只删除 no；yes 和未解析答案保留并报告。

## T1.3 问法

三种候选定义在 scope_t13.ASK_CANDIDATES，来源是 TASK.md 的三种建议，英文在开发前固定。共同使用 scopes/any_change.json。category 问两期地类是否改变；surface 问真实地表覆盖变化并排除光照、季节和错位；describe 要求在 reason 中先识别两期类别，再判断。describe 的描述留在原有 reason 字段，JSON schema 不变。

候选组件来自 probe_t13 的两期 SAM 提议并集，经 DINOv2 区域均值 1−cos≥0.22 后并集的 8 连通域。probe 报告只存计数，不存掩码；prepare 读取原缓存 `CACHE/dynamic/inventory.json`、item_*.npz，复用 probe_t13.route 重建。需给同一配置、完整相应图单的 probe JSON，开发 JSON 的 selected_threshold 必须是 .22。不会启动 WorkerPool，缓存不足报错，由计算节点重新准备缓存后续跑。

组件标签为组件内真值比例≥50%（含等号）。AUC 不按面积加权；无正例或无负例时为 null。缺失 p_yes 不编造分数：不参与 AUC，概率门限规则保留该组件，报告缺失数。所有图均计入像素 F1/P/R，固定掩码为空也计漏检。直接删除 no 与 p_yes 门限分别报告；问法按完整开发集 delete-no F1 选，概率门限再在该问法的开发网格上选，同分取小门限。128 张固定问法、固定概率门限，只报告一次。

19. 指向已有摸底缓存和两个完整 probe 报告。按真实归档修改三个路径，不能用空目录代替。

```bash
export T13_CACHE="$R/vision-cache" T13_DEV_PROBE="$PWD/results/agent_baseline/data/probe_t13_dev_0of1.json" T13_VAL_PROBE="$PWD/results/agent_baseline/data/probe_t13_val_0of1.json"
```

20. 从开发缓存重建固定候选，无 API。

```bash
sbatch --output="$R/t13_ask/logs/prepare-dev-%j.out" --error="$R/t13_ask/logs/prepare-dev-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch t13-ask "$R/t13_ask/candidates/dev" prepare --phase dev --run-root "$R" --vision-config code/.runtime/vision_t13.json --cache "$T13_CACHE" --probe-report "$T13_DEV_PROBE" --candidates-dir "$R/t13_ask/candidates/dev"
```

21. 三种问法逐组件调用并选择，示例总上限 30 元。

```bash
sbatch --output="$R/t13_ask/logs/select-%j.out" --error="$R/t13_ask/logs/select-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch t13-ask "$R/t13_ask/dev" answers --phase dev --run-root "$R" --candidates-dir "$R/t13_ask/candidates/dev" --output-dir "$R/t13_ask/dev" --max-cost-cny 30
```

22. 以下两步由 Claude 决定是否执行。先重建 128 张缓存候选，无 API。

```bash
sbatch --output="$R/t13_ask/logs/prepare-val-%j.out" --error="$R/t13_ask/logs/prepare-val-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch t13-ask "$R/t13_ask/candidates/val" prepare --phase val --run-root "$R" --vision-config code/.runtime/vision_t13.json --cache "$T13_CACHE" --probe-report "$T13_VAL_PROBE" --candidates-dir "$R/t13_ask/candidates/val"
```

23. 128 张单次报告。只调用 selection.json 选中的一种，示例上限 15 元。

```bash
sbatch --output="$R/t13_ask/logs/report-%j.out" --error="$R/t13_ask/logs/report-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch t13-ask "$R/t13_ask/val" answers --phase val --run-root "$R" --candidates-dir "$R/t13_ask/candidates/val" --output-dir "$R/t13_ask/val" --settings "$R/t13_ask/dev/selection.json" --max-cost-cny 15
```

prepare 的 complete.json 在调用 API 前保存逐图组件数、每种问法的调用数和估计费用。selection.json 保存完整问法、来源、选中项、组件数、预计调用数与费用；每种问法目录含 answers.jsonl、report.json、masks/。val/report.json 保存正式报告，不重新选择问法。

## 指标与测试

固定报告和问法报告已给累计像素 F1/P/R。Agent 运行后复用 second_metrics 的 JSON spec 和 paired_bootstrap：每组设置 manifest、三个 agent_runs、fixed_dir、undirected_dir、rule_dir、rule_answers。T1.2 的 direction=any；C1t 使用 gt/new，反方向从相邻 gt/demolished 自动读取，方向违反率=预测与反方向真值交集/预测面积，空预测为 0。C1t 的 undirected_dir 填 unintersected/val，汇总行命名为 fixed_unintersected，表示“新建但未限制前期地类”；标准比较是选中求交固定路线与求交后规则。bootstrap 共享每图抽样索引，4000 次，三个重复固定。

24. 写好含上述字段的 spec 后提交完整指标汇总。

```bash
sbatch --output="$R/c1t/logs/metrics-%j.out" --error="$R/c1t/logs/metrics-%j.err" code/agent_baseline/analysis/run_t12_c1t.sbatch metrics "$R/c1t/metrics.json" --spec "$R/c1t/metrics_spec.json" --output "$R/c1t/metrics.json" --bootstrap-samples 4000
```

25. 本批允许的单线程假后端测试。可在管理节点运行；不读取真实数据或加载模型。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code" OVCD_CODE="$PWD/code/ovcd_v0" TMPDIR="$PWD/code/.runtime/tmp" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m unittest agent_baseline.tests.test_t12_c1t agent_baseline.tests.test_second agent_baseline.tests.test_generalize agent_baseline.tests.test_replay -v
```

冻结 fixture、工具关闭、G3 及合成 replay 的直接相关测试也应保留。真实历史回放、完整既有测试、GPU/API 运行由后续计算节点流程完成，本批不执行全量实验。
