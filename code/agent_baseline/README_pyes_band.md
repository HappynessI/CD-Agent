# p_yes 离线分析

入口：`analysis/pyes_band.py`。仅依赖已有 JSONL、固定路线 PNG 和真值 PNG，不导入模型客户端，不调用 API。运行目录为本次 staging 目录。所有写入固定在当前目录的 `tmp/` 和 `results/`。

```bash
TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python -u \
code/agent_baseline/analysis/pyes_band.py

TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/code" \
/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python -u \
-m unittest agent_baseline.tests.test_pyes_band
```

本次 TASK.md 明确要求离线单线程、不提交 Slurm；本入口按此执行。不运行原有调用模型或多进程的分析入口。I/O 复用 `analysis/io_retry.py`。

先完整盘点，再计算。组件用 scipy 的 8 连通标号，与 `common.components` 一致，最小面积为 1。通过 bincount 统计每组件 TP/FP；FN 包含所有候选外真值像素。规则只删 `answer=no`，缺失/未解析答案保留。p_yes 仅接受有限的 [0,1] 数值，不用 yes/no 填充。

分段为 [0,0.1)、…、[0.9,1]，缺失另列。选择区间为 0.1 网格上全部 a<b 的闭区间 [a,b]，加“不恢复 no”。只使用完整开发集，整集像素 F1 最高者胜；并列先窄后低。选择冻结到 `results/pyes_selection.json`，之后才计算报告集分数。bootstrap 固定区间，不重复选型，按图配对 5000 次，种子 1016。空候选图仍计入图抽样。

完美复核按 TP>FP 的整组件保留，分别报告所选区间和全部 no。它用真值，仅作天花板参照；不等于实际规划器可实现收益，也不是所有子集的数学最优值。

SECOND 开发 ask 路径含两层方向：`second_dev/rule_card_defined/<direction>/<direction>/`。开发 scope 名称为 `dev_card_defined`，报告为 `building_rebuild_excluded`；脚本核对 `code_k` 的开发卡与本地最终卡内容相同。WHU 开发缺掩码时不从报告集选区间；补问组件数标未知，预算暂按报告集组件密度估计。

产物：`results/pyes_band.json`、`results/pyes_band.md`、六张分段 PNG、冻结选择 JSON。`tmp/pyes_inventory.json` 保存仅含组件像素计数和安全元数据的中间盘点；`--reuse-inventory` 可在原始输入没有改变的前提下重用，不能用它声称重新验证原始文件。原始 ask 内容、密钥、网络地址不写入产物。

测试：分段边界/缺失、像素 F1、开发集选型及反向报告示例、并列规则、不恢复 no、TP 主导整组件、连通域编号、配对 bootstrap。

最终 8 项构造测试通过，包含未解析答案按原规则保留。核查见 `results/pyes_validation.json`：报告规则 F1 与项目文档一致，分段计数及 TP/FP 守恒，六张 PNG 可读。WHU 缺开发候选池的面板明确标缺数据。

本次 LEVIR 选 [0.1,0.3]，报告 ΔF1 约 -0.002；SECOND 新建选 [0.1,0.5]，报告 -0.88；拆除不恢复 no。三者均未达到报告 +1 分。SECOND 全部 no 的 TP 主导真值复核天花板 +4.29/+6.92 与区间迁移结果分开解释。完整盘点、费用和局限见报告及 NOTES.md。
