# 完整 clean-confirm 入口预声明（仅准备，未运行）

新增入口仅覆盖完整 T3.1 clean-confirm 106 任务和 T3.3 clean-confirm 156 任务；不读取 report manifest，也不影响原在线 20 任务的 fixed 配对报告。执行必须提供用户明确批准记录（status=approved 与 approval_reference），以及两类任务各自保存的 tune 参数选择文件。两类任务分别使用其冻结 delta/joint、Top5/8；confirm 不扫描或重选参数。

推理严格复用 N9 的 RGB/E/SAM/DINO 路径：同图去自匹配后重新池化净区域；外部示例与查询坐标独立，不扣查询中的示例，不引入语义路由或全图拒识。所有推理池与排序在读取GT前落盘。随后仅评估 Baseline-Fixed 原R1 delta Top8 与 N9-Fixed 本类型的 tune 冻结配置。T3.1 评分扣 E，T3.3 不扣外部 E；若有额外 ignore 则两组一致应用。

输出 `fixed_full_confirm/frozen.json` 在计算前记录全部任务ID、两类配置与批准来源；逐任务缓存池、排序、预测和指标，恢复时跳过已完成任务。优先复用历史准确源缓存及本轮 tune 的 vision_cache，缺失部分仅使用本地模型补算，无模型API。报告分别提供全部/目标类/地表类 P/R/F1、TP/FP/FN、空预测数、任务数/独立图数、GT量和候选池覆盖召回。

新文件：analysis/n9_t3_full_confirm.py、tests/test_n9_t3_full_confirm.py、tools/run_n9_t3_full_confirm.sh。脚本申请 GPU1/CPU2/16G/2h，以支持外部示例特征缺失补算；若缓存已齐，可用模块 `--backend cache-only` 在CPU作业运行。批准文件与两份选择文件通过 N9_APPROVAL_FILE、N9_TUNE_SELECTION_T31、N9_TUNE_SELECTION_T33 显式传入。测试尚未执行；主Agent统一安排Slurm，人工批准前不得正式运行。
