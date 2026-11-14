# 下一最小假设与真 Agent 接入边界

历史假设记录（保留原提出过程）：后续候选粒度审计已发现tune Top20的246个候选全部单一8连通，本文提出的“多组件单位差异”在该池没有得到支持；未实施拆分工具。最新判断见candidate_geometry_tune及geometry_route。

本方案只据当时tune与已审查工具接口提出，不是当前待运行方案，也不是收益声明。

调参证据：LEVIR V2 比 V0 的 K10 F1 高7.9825点，但仍比初始答案低4.5225点（图像配对区间[-8.1971,-1.7362]），5张图均未正增益；主要是少接受低收益候选。SECOND 新建 V4 比初始仅+0.2182，区间跨0；拆除 V0 的增益集中于1张图。继续减少yes不能自动转化为补漏能力。

最小不同假设：现有 Agent 缺少对候选几何粒度的显式审查，而不只是问法不够保守。`CachedMissedCandidates.candidates` 扣当前答案后仍按一个原proposal返回，可能含多个断开部分；`to_mask/ask` 再按连通组件处理。这使TopK的proposal单位与核验单位不同。既有 `segment_prompt` 和 `refine_region` 可以返回局部修改，后者支持 `set_current=false`；无需再造图像模型工具。

先做不调用API的固定单位审计：在现有冻结候选和固定初始答案上记录每个proposal扣起点前后面积、连通组件数、实际ask组件数、来源和排序；按组报告TopK与实际核验次数差异。GT只在之后评分TP/FP及召回，不参与分解或排序。不得把细条候选中少数正确像素当作检索能力提升。

若单位差异确实明显，唯一下一处理是：候选工具显式返回逐组件句柄、共同parent和source_rank；保持原排序/父候选预算并报告真实子组件预算，不以碎片增加请求优势。Agent可选择inspect，再用现有refine_region(set_current=false)修改候选句柄，决定是否合并。禁止工具直接写最终答案。完整对象缓存答案不能用于裁切后的新mask；裁切/细化后必须新核验。

真实Agent实验仍为冻结6个开发任务×旧新×3独立seed，初始mask同旧base。旧runner只支持four/som/structured和旧来源池，不能直接宣称执行了夜间门控/evidence或V2/V4。要接入夜间策略，需独立新provider读取冻结rank manifest及gate provenance；在current裁切后保持父排序并重新编号，核验renderer显式支持冻结V0或按任务V2/V4。既有默认工具和ask保持不变，planner显式qwen3.7-flash。应先验证新provider与离线固定初始答案输出一致，再启动真Agent会话。接入代码由主Agent统一共享接口；此方案尚未实现或上线。

若确认阶段没有可靠净收益，保持该阶段为机制诊断，不强行以“更少伤害”晋级线上。若冻结主方法有收益，也须检查逐图集中度、区间、候选召回与实际工具采用率，再决定真实Agent3seed；确认结果只检验已冻结策略，不用于再选最好的K/方向组合。

最新接线状态：可选冻结rank provider及真实runner入口已实现并通过11项Slurm单测，详见online_ranked_integration_preparation.md；默认关闭，原6任务新版manifest仍未准备，真实Agent会话仍为0。上文“尚未实现”保留为提出方案时的历史状态，不再代表当前代码状态。
