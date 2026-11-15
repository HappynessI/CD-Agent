# 补充预声明：召回—精度折中诊断

时点：原95%TP保留主实验与局部核验调参已完成；confirm的原Top8和主门控结果已曝光。此为主审核发现保守约束可能隐藏真实折中后追加的探索性诊断，不是事先未知的新确认集，也不替换原主结论。

只复用原tune已算六阈值，不增加扫描。每个固定排序取tune全体累计像素F1最高者，平局低阈值，无TP保留约束；由现有记录得到固定delta=0.6、joint=0.8。二者tune真阳性仅保留约63%，此代价必须与F1和空目标改善并列呈现。

只对这两个冻结值各运行一次confirm30，同原R1池、原delta/joint排序和Top8预算，过滤不补位。不做API、语言桥接、局部核验、候选修改或动态路由。原95%TP主标准和原frozen.json保持不变。即使确认F1提高，也不自动启动在线或将召回损失忽略。

报告原60与固定tune/confirm、非空/空/≥16诊断、目标类/地表类的P/R/F1、空正确率/FP、非空全拒、TP损失/保留和逐例集中度。候选选择只读相似度；GT只用于离线评分。输出独立保存在tradeoff_diagnostic，旧结果不覆盖。

Slurm最终CPU诊断入口：

```bash
PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/code \
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B \
-m agent_baseline.analysis.n7_t3_tradeoff \
--run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run \
--out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115
```

可同作业再执行`n7_t3_top1_audit --out <同根>`与最终`n7_t3_diagnostics --out <同根>`；仍由主Agent提交统一exp-loop，不重复提交。
