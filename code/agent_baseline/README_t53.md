# G3 与 T5.3 语言反馈

本次只交付代码和单元测试，未运行模型、API、Slurm、真实数据处理或正式回放。以下为审核合并后的计算节点运行手册。资源和费用上限是申请估计，未实测。长任务沿用 `run_cpu.sbatch`、`run_agent.sbatch` 的完成通知；每个步骤成功并核对结果后再运行依赖步骤。

## 固定行为

`--gate3` 隐含 G2，必须同时给 `--verify-stage`。默认关闭。大框检查只作用于 `segment_prompt` 的框，点提示和 `refine_region` 沿用原行为。框按现有 `box_pixels` 取整后计算面积，严格超过 `--big-box-share` 才返回 `box_too_large`，不调用 SAM；恰好等于阈值可用。常量 `tools.BIG_BOX_SHARE=.10`，命令行可覆盖 `(0,1]`。G3 的 `input.json` 和 `final.json` 的 `backends` 都记录 `gate3`、隐含的 `gate2` 和 `big_box_share`。

提交前报告及 G2 晚到报告共用 `Toolbox.report`。有区域时列出区域外像素总数和涉及的当前组件 ID；有方向时根据最近一次可识别的验证器 `type` 列出反方向 ID，`change` 字段未解析时仍可使用类型。映射取现有 `PSEUDO_TYPES` 的交集：appeared → `demolished`（building）、`removed_tree`（tree）；disappeared → `new_building`、`new_tree`。`rebuilt` 和伪变化类型没有唯一反方向，不列入此项。没有违反就不增加文字；报告不修改掩码。

G3 的反馈会话只核验公开反馈范围及相对 `mask#0` 改动过的组件：T5.2 用点击为圆心、半径 32 像素的闭圆与组件相交；T5.3 用方位词对应的完整格子与组件相交。当前组件与原组件的完整像素轮廓不同即算改动，涵盖新增、面积变化、拆分与合并；被删除或缩小的原组件也核验。未改变且远离反馈的组件不欠核验答案。空反馈时只核验改动。`pending`、`owed`、`stage`、提交门、晚到报告和默认 `ask()`/省略 `ids` 的 `ask(regions=...)` 使用同一范围；显式指定 `ids` 仍可主动核验其他组件。原来的每次 64 个默认 ask 上限、重试规则、回复预算不变。

清单 `meta.feedback` 在 harness 中经 `public_feedback` 白名单解析，会话只接收点坐标或格子名称。区域编号、面积、生成用的点（T5.3）、真值路径和错误区域掩码不进入 Agent 请求。工具定义、系统提示词、验证器问法保持原样；Python 默认关闭。

## T5.3 生成与对照

复用 T5.2 同一套 64 张 LEVIR 验证图和 `init/sam3_imc_p85/val`。错误判定、8 连通、严格小于 50% 的覆盖条件、最远点和平局规则均沿用 T5.2。

3×3 方位顺序为 `upper left, top, upper right, left, center, right, lower left, bottom, lower right`。错误区域按其最远点 `[x,y]` 归属 `floor(3*x/width), floor(3*y/height)`；边界归右侧/下侧，非整除图像尺寸也覆盖每个像素恰好一次。每个错误只归属一个格子，跨格子的其他部分仍保留。先统计所有类型的错误；仅所在格子有一个归属错误的区域可选。可选区域按面积降序，每张最多一条；同面积先漏检，再源编号。

指令由 T5.1 原文追加 `User feedback:` 和下列句子组成：

- 漏检：`A building in the {where} was missed.`
- 误检：`One of the marked changes in the {where} is not a building change.`（10 张试跑后由 `The marked change in the {where} ...` 改来：原句没说是格子里的哪一个，规划器把整格组件都删了）
- 有反馈时结尾：`Correct only what the user described and keep the rest of mask#0 unchanged.`
- 无可用错误时：`The user reviewed the mask and pointed out no errors. Keep mask#0 unchanged.`

不在指令中提供坐标。输出为 `results/agent_baseline/data/t53_feedback.json`、`$AGENT_RUN/t53/tasks/T5.3.jsonl`、`$AGENT_RUN/t53/regions/<stem>/<gt|given>_<id>.png`。JSON 同时保留全部错误的格子归属，便于核对唯一性；PNG 保存被描述的错误区域，漏检为未覆盖部分。

两种固定对照共用一个候选池，不调用规划器：

- 漏检候选来自 T1 和 T2 的 SAM3 缓存（512 网格、像素 .85、分数 .5），要求给定掩码覆盖严格小于 50%。实例按自身最远点归属格子，只选与反馈相同格子的实例；保留完整实例。相同轮廓去重，T1 优先。候选排序为面积降序、T1 在 T2 前、源编号升序。
- 误检候选来自给定掩码的完整连通组件，也按自身最远点归属格子。
- 对照 1：取面积最大者，漏检并入，误检删除。
- 对照 2：对全部候选使用 `ask_eval.api_answer` 和规则对照的 `VARIANTS['f37_corr']`，即 flash 四图对应问法、同模型参数、裁剪方式与 seed 公式。漏检选最大 `p_yes`，误检选最小 `p_yes`，同分按候选顺序。没有有效 `p_yes` 的候选保留诊断，不参与排序；全无分数则原样，并明确记录 `no_scored_candidates`。
- 两种对照的空候选、空反馈均保持原样。后端失败直接终止，已完成答案可续跑。`largest/`、`verifier/` 各自保存 `settings.json`、`masks/`、`traces/`；验证器另存逐候选 `answers.jsonl` 和 `budget/`。同目录设置不一致时失败。相同输入续跑复用答案和完整图片输出。

对照缺陷检查：同时取两期避免 T5.2 A 单期选择偏差；完全重复实例去重避免重复调用。仍可能没有覆盖目标建筑的 SAM3 实例，也可能一个实例合并多栋楼、跨格子最远点与错误最远点归属不同，或同格子有多个正常候选。这些是该候选池的实际限制。10 张试跑必须离线查看 `traces/` 的候选及选中实例与 `regions/` 的交集，统计空池、目标无候选覆盖和跨格子情况后再决定是否全量；不能用真值替对照选候选。本次尚无真实检查结论。

## 逐步命令

从含 `code/` 的交付目录运行。已有认证由账号环境提供，`BAILIAN_CSV` 只传路径，不打印内容。API 作业提交前按工作区规范完成节点、网络与 endpoint 探测，将 `NODE` 固定为探测成功的同一节点；下面所有 API 作业使用该节点。资源检查与网络探测不在本次任务中执行。

1. 设置环境（1 条命令）。`NODE` 须已设置为探测成功节点；`T53_PARTITION` 可按集群情况修改。

```bash
export AGENT_CODE="$PWD/code/agent_baseline" AGENT_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run OVCD_CODE="$PWD/code/ovcd_v0" OVCD_RUN=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/ovcd_v0_run PYTHONPATH="$PWD/code:$PWD/code/ovcd_v0${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AGENT_ASK_SLOTS=3 USE_LOCAL_WORKERS=1 USE_ASK_WORKER=0 T53_PARTITION=gre PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
```

2. 提交前检查控制面。出现 socket 错误或控制器 DOWN 时停止提交，先解决连接。

```bash
scontrol ping
```

3. 检查账号作业和重复任务。

```bash
squeue -u guisongxia
```

4. 创建共享日志目录。

```bash
mkdir -p "$AGENT_RUN/logs"
```

5. 新增及直接相关单元测试。CPU 1，4 GiB，GPU 0，预计 10 分钟。全部测试可另用原有 discover 命令在计算节点执行。

```bash
sbatch --partition="$T53_PARTITION" --job-name=g3-t53-tests --cpus-per-task=1 --mem=4G --time=00:10:00 "$AGENT_CODE/run_cpu.sbatch" -m unittest agent_baseline.tests.test_g3_t53 agent_baseline.tests.test_t52 agent_baseline.tests.test_loop agent_baseline.tests.test_tools agent_baseline.tests.test_generalize agent_baseline.tests.test_second_boxes agent_baseline.tests.test_replay -v
```

6. G1 历史回放：2 任务 × 3 次，每个 CPU 1、4 GiB、GPU 0，预计每组 1 小时。不调用模型/API，完整归档回放仍在计算节点运行。

```bash
for n in 1 2 3; do for task in T1.1xC0 T5.1; do sbatch --partition="$T53_PARTITION" --job-name="g3-replay-s${n}-${task}" --cpus-per-task=1 --mem=4G --time=01:00:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.replay_g1 --config "json_qwen37flash_g1_s${n}" --task "$task" --split val --reference-code "$AGENT_RUN/code_g" --output-dir "$AGENT_RUN/t53/regression/g1_s${n}_${task}" || break 2; done; done
```

7. 生成 T5.3 正式反馈和清单：CPU 1，4 GiB，GPU 0，预计 30 分钟。

```bash
sbatch --partition="$T53_PARTITION" --job-name=t53-feedback --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.make_t53_feedback --run-root "$AGENT_RUN" --output "$PWD/results/agent_baseline/data/t53_feedback.json"
```

8. T5.3 Agent 10 张试跑：CPU 4，32 GiB，GPU 1（建议 24 GiB 显存），预计 1 小时。费用估算上限 5 元，不保存 checkpoint。

```bash
sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=t53-smoke --cpus-per-task=4 --mem=32G --time=01:00:00 "$AGENT_CODE/run_agent.sbatch" --tasks "$AGENT_RUN/t53/tasks/T5.3.jsonl" --task T5.3 --split val --config json_qwen37flash_g3_t53_smoke_s1 --limit 10 --seed-offset 1 --concurrency 6 --prompt-version v4 --status --verify-stage --gate3 --big-box-share 0.10 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch t53_smoke --max-cost-cny 5
```

9. 对照 1 的相同 10 张：CPU 1，4 GiB，GPU 0，预计 10 分钟，只读现有 SAM3 实例缓存。

```bash
sbatch --partition="$T53_PARTITION" --job-name=t53-largest-smoke --cpus-per-task=1 --mem=4G --time=00:10:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t53_fixed --tasks "$AGENT_RUN/t53/tasks/T5.3.jsonl" --output-dir "$AGENT_RUN/t53/fixed_smoke" --mode largest --limit 10
```

10. 对照 2 的相同 10 张：CPU 1，4 GiB，GPU 0，预计 30 分钟，串行 API，估算上限 2 元。账号 wrapper 在计算节点读认证；不写认证内容。

```bash
sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=t53-verifier-smoke --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" /guisongxia01/pangchao/wangyihan/Chang-Agent/tools/run_with_bailian_csv.py "${BAILIAN_CSV:?configured credential path required}" "$PY" -B -m agent_baseline.analysis.t53_fixed --tasks "$AGENT_RUN/t53/tasks/T5.3.jsonl" --output-dir "$AGENT_RUN/t53/fixed_smoke" --mode verifier --limit 10 --max-cost-cny 2
```

试跑验收：10 个 Agent `final.json`，两个对照各 10 个 PNG 和 trace；核对原掩码路径、无坐标语言指令、空反馈图；检查核验记录只覆盖反馈格子及变化组件；核对超大框报错发生在后端调用前。检查全部错误/兜底。离线查看两个对照的候选是否含目标错误、选择结果是否正确，记录空候选和无分数个数。不能据得分更换图像或门限。通过后再执行以下全量步骤。

11. LEVIR G3 回归：T1.1、T5.1 各 64 张 × 3 次，顺序执行。CPU 4，32 GiB，GPU 1，预计 6 小时，共享估算上限 100 元用于 G3 回归（含后两步），实际需看账本。

```bash
args=(); for n in 1 2 3; do for task in T1.1xC0 T5.1; do args+=(--task "$task" --split val --config "json_qwen37flash_g3_s${n}" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate3 --big-box-share 0.10 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch g3_regression --max-cost-cny 100 :::); done; done; sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=g3-levir --cpus-per-task=4 --mem=32G --time=06:00:00 "$AGENT_CODE/run_agent.sbatch" "${args[@]}"
```

12. SECOND C2/C4：沿用已冻结的框和清单，各 128 张 × 3 次。CPU 4，32 GiB，GPU 1，预计 6 小时。与第 11、13 步最多同时运行两个 Agent 作业。

```bash
args=(); for n in 1 2 3; do for kind in c2 c4; do if [[ $kind == c2 ]]; then task=T1.1xC2; direction=any; else task=T1.1xC4; direction=appeared; fi; args+=(--tasks "$AGENT_RUN/second/tasks/${task}_${direction}.jsonl" --task "$task" --split val --config "second_g3_${kind}_s${n}" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate3 --big-box-share 0.10 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --batch g3_regression --max-cost-cny 100 :::); done; done; sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=g3-c2c4 --cpus-per-task=4 --mem=32G --time=06:00:00 "$AGENT_CODE/run_agent.sbatch" "${args[@]}"
```

13. T5.2 回归：两档 k=1、3，各 64 张 × 3 次，复用原清单。CPU 4，32 GiB，GPU 1，预计 6 小时。对照仍使用已有固定响应 C（两期点提示取较小者）。

```bash
args=(); for n in 1 2 3; do for k in 1 3; do args+=(--tasks "$AGENT_RUN/t52/tasks/T5.2_k${k}.jsonl" --task T5.2 --split val --config "json_qwen37flash_g3_t52k${k}_s${n}" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate3 --big-box-share 0.10 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch g3_regression --max-cost-cny 100 :::); done; done; sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=g3-t52 --cpus-per-task=4 --mem=32G --time=06:00:00 "$AGENT_CODE/run_agent.sbatch" "${args[@]}"
```

14. T5.3 对照 1 全量：CPU 1，4 GiB，GPU 0，预计 30 分钟。正式输出与试跑分开。

```bash
sbatch --partition="$T53_PARTITION" --job-name=t53-largest --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t53_fixed --tasks "$AGENT_RUN/t53/tasks/T5.3.jsonl" --output-dir "$AGENT_RUN/t53/fixed" --mode largest
```

15. T5.3 对照 2 全量：CPU 1，4 GiB，GPU 0，预计 2 小时，估算上限 10 元。

```bash
sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=t53-verifier --cpus-per-task=1 --mem=4G --time=02:00:00 "$AGENT_CODE/run_cpu.sbatch" /guisongxia01/pangchao/wangyihan/Chang-Agent/tools/run_with_bailian_csv.py "${BAILIAN_CSV:?configured credential path required}" "$PY" -B -m agent_baseline.analysis.t53_fixed --tasks "$AGENT_RUN/t53/tasks/T5.3.jsonl" --output-dir "$AGENT_RUN/t53/fixed" --mode verifier --max-cost-cny 10
```

16. T5.3 Agent 全量 64 张 × 3 次：CPU 4，32 GiB，GPU 1，预计 4 小时，估算上限 25 元。

```bash
args=(); for n in 1 2 3; do args+=(--tasks "$AGENT_RUN/t53/tasks/T5.3.jsonl" --task T5.3 --split val --config "json_qwen37flash_g3_t53_s${n}" --seed-offset "$n" --concurrency 6 --prompt-version v4 --status --verify-stage --gate3 --big-box-share 0.10 --planner-model qwen3.7-flash --ask-backend api --ask-model qwen3.7-flash --ask-correspondence --pixel-thr 0.85 --sam-prompt-device cuda --init-name sam3_imc_p85 --batch t53_full --max-cost-cny 25 :::); done; sbatch --partition="$T53_PARTITION" --nodelist="${NODE:?set NODE after preflight}" --job-name=t53-full --cpus-per-task=4 --mem=32G --time=04:00:00 "$AGENT_CODE/run_agent.sbatch" "${args[@]}"
```

17. 写评测输入配置：CPU 1，2 GiB，GPU 0，预计 5 分钟。默认包含给定掩码、两对照、G2 T5.1、G3 T5.3；检测到现有 `json_qwen37flash_g2_t52_k1_s{1,2,3}` 目录时加入 T5.2 行。部分缺失不会静默跳过，评测时报错；可在 spec 中明确指定所需的三个 T5.2 重复目录。

```bash
sbatch --partition="$T53_PARTITION" --job-name=t53-spec --cpus-per-task=1 --mem=2G --time=00:05:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t53_metrics --write-spec --run-root "$AGENT_RUN" --output "$AGENT_RUN/t53/metrics_spec.json"
```

18. 全量指标和按图配对 bootstrap：CPU 1，4 GiB，GPU 0，预计 30 分钟。

```bash
sbatch --partition="$T53_PARTITION" --job-name=t53-metrics --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t53_metrics --spec "$AGENT_RUN/t53/metrics_spec.json" --output "$AGENT_RUN/t53/metrics.json"
```

19. 查询已提交作业状态。运行中的查询间隔采用 90–120 秒；不在登录节点长时间 sleep。

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

预算/额度停止后检查各目录的 `budget/STOP.json` 与全局 `STOP.json`。按实际原因处理后，保持配置续跑；不得删除失败会话以制造更好分数。固定对照用逐候选答案续跑，Agent 沿用 `final.json` 跳过规则。

## 指标与回归验收

T5.3 输出 JSON 保存逐图、逐次统计，Markdown 给各方法整图累计 F1^c/P/R、相对给定 ΔF1、漏检/误检修复率、区域外增删与 ΔF1、定位比例、描述区域被改动率、错误/兜底及 Agent 对各参照的配对区间。沿用 T5.2 的完整源组件修复规则和外扩 32 像素的未指出区域。定位组件定义为 `final XOR initial` 的 8 连通域，分子为与被描述错误区域相交者；漏检目标取未覆盖部分。无改动时定位比例为 null；有反馈但没有改动时区域被改动率为 0；无反馈时该率为 null。三次重复先独立累计再取均值。bootstrap 按图配对 4000 次、种子 1016，重复固定，两侧共享图像抽样。T5.2 行按同一 T5.3 错误重新评测，作为配对参照。

G3 回归检查 G1 夹具和归档回放、旧配置请求内容、会话数、错误/兜底、费用及三种改动是否按条件生效。必须报告区间和失败会话，不预先声称无损。以下补充命令均在计算节点执行，CPU 1、4 GiB、GPU 0、预计各 30 分钟。

20. LEVIR T1.1/T5.1 对 G2 的按图配对区间，三次重复固定。复用本次的计分和 bootstrap 函数；缺会话或掩码直接报错。

```bash
sbatch --partition="$T53_PARTITION" --job-name=g3-levir-metrics --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -c '
import os
from pathlib import Path
from common import names, load_gt, write_json
from agent_baseline.analysis.t52_metrics import counts, score, session_prediction
from agent_baseline.analysis.t53_metrics import paired_delta
run = Path(os.environ["AGENT_RUN"])
truth = {name: load_gt("val", name) for name in names("val")}
report = {}
for task in ("T1.1xC0", "T5.1"):
    arrays, summaries = {}, {}
    for gate in ("g2", "g3"):
        arrays[gate], summaries[gate] = [], []
        for n in (1, 2, 3):
            data, diagnostics = [], []
            root = run / f"json_qwen37flash_{gate}_s{n}" / task / "val"
            for name, gt in truth.items():
                pred, diag = session_prediction(root, Path(name).stem, gt.shape)
                data.append(counts(pred, gt))
                diagnostics.append(diag)
            arrays[gate].append(data)
            summaries[gate].append(dict(score(data), errors=sum(d["error"] for d in diagnostics), fallbacks=sum(d["fallback"] for d in diagnostics)))
    report[task] = dict(per_run=summaries, g3_minus_g2=paired_delta(arrays["g3"], arrays["g2"]))
write_json(run / "t53/regression/levir_g3_vs_g2.json", report)
'
```

21. C2/C4 写新 spec，沿用既有冻结基线，G3 输出前缀与第 12 步一致。

```bash
sbatch --partition="$T53_PARTITION" --job-name=g3-box-spec --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.second_box_metrics --write-spec --run-root "$AGENT_RUN" --baseline-root "$AGENT_RUN/second/box_baselines" --agent-prefix second_g3 --output "$AGENT_RUN/t53/regression/box_g3_spec.json"
```

22. C2/C4 计算 F1、区域/方向违反率与配对区间。和已有 `second/box_metrics.json` 并列报告，不覆盖旧文件。

```bash
sbatch --partition="$T53_PARTITION" --job-name=g3-box-metrics --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.second_box_metrics --spec "$AGENT_RUN/t53/regression/box_g3_spec.json" --output "$AGENT_RUN/t53/regression/box_g3_metrics.json"
```

23. T5.2 写 G3 spec。固定响应 C 的现有目录为 `t52/fixed_bothdates/k{1,3}`；原 G2 和 T5.1 参照目录沿用已经核实的 `metrics_spec_bothdates.json`。

```bash
sbatch --partition="$T53_PARTITION" --job-name=g3-t52-spec --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -c '
import os, json
from pathlib import Path
from common import write_json
run = Path(os.environ["AGENT_RUN"])
spec = json.loads((run / "t52/metrics_spec_bothdates.json").read_text())
for key, group in spec["groups"].items():
    group["agent_runs"] = [str(run / f"json_qwen37flash_g3_t52{key}_s{n}" / "T5.2/val") for n in (1, 2, 3)]
write_json(run / "t53/regression/t52_g3_spec.json", spec)
'
```

24. T5.2 的整图、修复和未指出区域指标，原 G2 输出保持原样。

```bash
sbatch --partition="$T53_PARTITION" --job-name=g3-t52-metrics --cpus-per-task=1 --mem=4G --time=00:30:00 "$AGENT_CODE/run_cpu.sbatch" -m agent_baseline.analysis.t52_metrics --spec "$AGENT_RUN/t53/regression/t52_g3_spec.json" --output "$AGENT_RUN/t53/regression/t52_g3_metrics.json"
```

以上命令尚未执行。真实缓存、候选覆盖、模型环境、费用及 G3 精度变化都需由计算节点试跑和全量结果判断。
