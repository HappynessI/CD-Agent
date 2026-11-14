# WHU 官方有效区域与排名输入审查

WHU 审计包含 any(C0)、appeared(C1)、disappeared(C1)，各使用固定 dev60。官方路径为 `whu/gt_official_v3/c0`、`new`、`demolished`。C1 通过 `evaluation.load_reference` 使用同一 canonical `ignore`，返回已裁掉 ignore 的 truth 与 valid。审计函数对历史 baseline 和各来源都先与 valid 相交，再计算 TP/FP/FN、unique 和 add-all；忽略区不会变成 FP 或 FN。C0 不应用方向性 ignore。来源保持历史候选池，因此是新官方 GT 评分旧候选，不是重新生成来源，也不覆盖旧 GT 记录。

排名准备读取 `candidate_gates_pilot/score_summary.json` 的冻结 selection[group].gate，并从对应 prepared NPZ 解码该 gate 替换 S5；S1–S4 原样保留。prepare 不读取 GT、不重做门控选择。selection 本身由 tune TP保留至少80%后最小FP规则产生；confirmation 仅使用该选择。核验器另由 `selected_verifier.json` 冻结，提交时显式设置 NIGHT_VERIFIER。

实际接续顺序：冻结核验器与完成候选门控评分后，运行 `run_night_missed_rank.sbatch prepare`，它执行测试、构造排名候选、写 `rank_<V>/request_inventory.json`；再用verify --split all覆盖固定tune与confirm，选择器事先冻结，结果仍分组评分；最后 score 和夜间报告。qwen3.7-flash，串行；已有完全相同 mask 和同核验器答案复用，新增或裁切 mask 重新核验。V0 新候选保持原四图和原问法。

30图、3种排名各Top20的理论上界为1800个候选槽；排序重叠与旧答案复用会减少实际新增请求。该上界不是观测调用数，也不含最多3次/槽的暂态重试。新增 inventory 依据实际 manifest/answers/usage 分列已完成、可恢复响应、旧答案可复用、新请求槽以及重试上界；不以估价拦截实验。该计数须在 Slurm 实际生成，未完成前不能宣称实测值。
