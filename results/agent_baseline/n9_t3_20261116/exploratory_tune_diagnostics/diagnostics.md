# N9 tune 固定检索探索诊断

仅复用已计算JSON，无新推理、GT读取或调参；人工批准前不执行正式confirm/online。任务共享图像，任务macro/胜负不是独立图像统计检验。

| 类型 | 分组 | 任务/图 | 方法 | pooled F1 | macro F1 | 胜/平/负（对原Top8） | 空预测 |
|---|---|---|---|---|---|---|---|
| T3.1 | 全部 | 116/29 | baseline_R1_delta_top8 | 12.43881 | 10.37112 | 0/116/0 | 0 |
| T3.1 | 全部 | 116/29 | clean_delta_top5 | 14.02938 | 11.34324 | 53/41/22 | 0 |
| T3.1 | 全部 | 116/29 | clean_delta_top8 | 15.02897 | 10.87917 | 27/55/34 | 0 |
| T3.1 | 全部 | 116/29 | clean_joint_top5 | 13.84375 | 11.44716 | 58/41/17 | 0 |
| T3.1 | 全部 | 116/29 | clean_joint_top8 | 15.00708 | 10.96389 | 39/42/35 | 0 |
| T3.1 | 目标类 | 63/27 | baseline_R1_delta_top8 | 17.54220 | 12.19037 | 0/63/0 | 0 |
| T3.1 | 目标类 | 63/27 | clean_delta_top5 | 20.21467 | 13.15235 | 25/26/12 | 0 |
| T3.1 | 目标类 | 63/27 | clean_delta_top8 | 21.65558 | 13.02572 | 17/25/21 | 0 |
| T3.1 | 目标类 | 63/27 | clean_joint_top5 | 19.89373 | 13.54339 | 28/26/9 | 0 |
| T3.1 | 目标类 | 63/27 | clean_joint_top8 | 21.20692 | 12.79647 | 19/24/20 | 0 |
| T3.1 | 地表类 | 53/24 | baseline_R1_delta_top8 | 7.51529 | 8.20862 | 0/53/0 | 0 |
| T3.1 | 地表类 | 53/24 | clean_delta_top5 | 7.98813 | 9.19279 | 28/15/10 | 0 |
| T3.1 | 地表类 | 53/24 | clean_delta_top8 | 8.35723 | 8.32760 | 10/30/13 | 0 |
| T3.1 | 地表类 | 53/24 | clean_joint_top5 | 7.88573 | 8.95543 | 30/15/8 | 0 |
| T3.1 | 地表类 | 53/24 | clean_joint_top8 | 8.85229 | 8.78553 | 20/18/15 | 0 |
| T3.3 | 全部 | 182/30 | baseline_R1_delta_top8 | 9.94386 | 6.79877 | 0/182/0 | 2 |
| T3.3 | 全部 | 182/30 | clean_delta_top5 | 7.63106 | 5.92524 | 42/92/48 | 2 |
| T3.3 | 全部 | 182/30 | clean_delta_top8 | 9.94386 | 6.79877 | 0/182/0 | 2 |
| T3.3 | 全部 | 182/30 | clean_joint_top5 | 7.66906 | 6.77766 | 50/86/46 | 2 |
| T3.3 | 全部 | 182/30 | clean_joint_top8 | 10.28236 | 7.64943 | 54/91/37 | 2 |
| T3.3 | 目标类 | 96/30 | baseline_R1_delta_top8 | 16.05095 | 9.23838 | 0/96/0 | 2 |
| T3.3 | 目标类 | 96/30 | clean_delta_top5 | 12.09886 | 7.85071 | 24/47/25 | 2 |
| T3.3 | 目标类 | 96/30 | clean_delta_top8 | 16.05095 | 9.23838 | 0/96/0 | 2 |
| T3.3 | 目标类 | 96/30 | clean_joint_top5 | 11.87500 | 9.37358 | 24/44/28 | 2 |
| T3.3 | 目标类 | 96/30 | clean_joint_top8 | 15.50982 | 10.20623 | 24/50/22 | 2 |
| T3.3 | 地表类 | 86/30 | baseline_R1_delta_top8 | 4.17658 | 4.07548 | 0/86/0 | 0 |
| T3.3 | 地表类 | 86/30 | clean_delta_top5 | 3.52496 | 3.77587 | 18/45/23 | 0 |
| T3.3 | 地表类 | 86/30 | clean_delta_top8 | 4.17658 | 4.07548 | 0/86/0 | 0 |
| T3.3 | 地表类 | 86/30 | clean_joint_top5 | 3.84032 | 3.87990 | 26/42/18 | 0 |
| T3.3 | 地表类 | 86/30 | clean_joint_top8 | 5.32476 | 4.79533 | 30/41/15 | 0 |

## 候选池覆盖与Top8/Top5

- T3.1 全部：原/净池GT覆盖召回 20.5792%/20.5067%，覆盖损失 1090 像素；GT总量 1503685。
  - delta Top8相对Top5：任务胜/平/负 30/37/49。
  - joint Top8相对Top5：任务胜/平/负 32/40/44。
- T3.1 目标类：原/净池GT覆盖召回 28.1392%/28.1347%，覆盖损失 32 像素；GT总量 710439。
  - delta Top8相对Top5：任务胜/平/负 19/22/22。
  - joint Top8相对Top5：任务胜/平/负 14/24/25。
- T3.1 地表类：原/净池GT覆盖召回 13.8083%/13.6750%，覆盖损失 1058 像素；GT总量 793246。
  - delta Top8相对Top5：任务胜/平/负 11/15/27。
  - joint Top8相对Top5：任务胜/平/负 18/16/19。
- T3.3 全部：原/净池GT覆盖召回 19.6693%/19.6693%，覆盖损失 0 像素；GT总量 1926028。
  - delta Top8相对Top5：任务胜/平/负 48/92/42。
  - joint Top8相对Top5：任务胜/平/负 49/89/44。
- T3.3 目标类：原/净池GT覆盖召回 31.1371%/31.1371%，覆盖损失 0 像素；GT总量 874847。
  - delta Top8相对Top5：任务胜/平/负 25/47/24。
  - joint Top8相对Top5：任务胜/平/负 24/45/27。
- T3.3 地表类：原/净池GT覆盖召回 10.1253%/10.1253%，覆盖损失 0 像素；GT总量 1051181。
  - delta Top8相对Top5：任务胜/平/负 23/45/18。
  - joint Top8相对Top5：任务胜/平/负 25/44/17。

## 典型得失（每类型/对照按F1差两端各最多2例）

### n9_T31_train_786_1to5_e1 · clean_delta_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 16.50026 → 0.26127；TP/FP/FN [157, 1589, 0] → [2, 1372, 155]。原/净池GT覆盖 157/153，原池未覆盖 0，清理覆盖损失 4，当前TopK未取用的池覆盖 151。
候选ID（0起始）：{"pool_ids_zero_based": [164, 250, 293, 24, 132], "original_geometry_ids_zero_based": [165, 252, 295, 24, 133], "raw_sam_ids_zero_based": [266, 389, 447, 60, 219]}；参照：{"pool_ids_zero_based": [128, 199, 165, 252, 295, 24, 133, 305], "original_geometry_ids_zero_based": [128, 199, 165, 252, 295, 24, 133, 305], "raw_sam_ids_zero_based": [213, 312, 266, 389, 447, 60, 219, 460]}。
去自匹配/净区处理后池GT覆盖下降：有可量化候选支持损失，但JSON不能区分具体被剔除候选语义。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_786_1to5_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_786_1to5_e1/metrics.json)。

### n9_T31_train_127_2to5_e11 · clean_delta_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 15.79479 → 2.07188；TP/FP/FN [2814, 1667, 28337] → [341, 1425, 30810]。原/净池GT覆盖 20340/20340，原池未覆盖 10811，清理覆盖损失 0，当前TopK未取用的池覆盖 19999。
候选ID（0起始）：{"pool_ids_zero_based": [207, 3, 151, 51, 106], "original_geometry_ids_zero_based": [210, 3, 154, 52, 109], "raw_sam_ids_zero_based": [507, 13, 385, 164, 307]}；参照：{"pool_ids_zero_based": [107, 12, 210, 3, 154, 52, 109, 226], "original_geometry_ids_zero_based": [107, 12, 210, 3, 154, 52, 109, 226], "raw_sam_ids_zero_based": [305, 28, 507, 13, 385, 164, 307, 571]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_2to5_e11/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_2to5_e11/metrics.json)。

### n9_T31_train_1798_2to5_e9 · clean_delta_top5 vs baseline_R1_delta_top8 · win

目标类；F1 14.52831 → 64.97752；TP/FP/FN [3328, 8588, 30570] → [19362, 6336, 14536]。原/净池GT覆盖 26232/26232，原池未覆盖 7666，清理覆盖损失 0，当前TopK未取用的池覆盖 6870。
候选ID（0起始）：{"pool_ids_zero_based": [19, 4, 20, 11, 13], "original_geometry_ids_zero_based": [22, 5, 23, 14, 16], "raw_sam_ids_zero_based": [107, 27, 108, 80, 83]}；参照：{"pool_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "original_geometry_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "raw_sam_ids_zero_based": [206, 149, 116, 78, 67, 3, 107, 27]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/metrics.json)。

### n9_T31_train_908_1to4_e27 · clean_delta_top5 vs baseline_R1_delta_top8 · win

地表类；F1 22.92835 → 50.63639；TP/FP/FN [736, 4350, 598] → [736, 837, 598]。原/净池GT覆盖 736/736，原池未覆盖 598，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [22, 4, 35, 37, 51], "original_geometry_ids_zero_based": [22, 4, 36, 38, 52], "raw_sam_ids_zero_based": [44, 13, 74, 80, 134]}；参照：{"pool_ids_zero_based": [26, 22, 4, 36, 38, 52, 53, 75], "original_geometry_ids_zero_based": [26, 22, 4, 36, 38, 52, 53, 75], "raw_sam_ids_zero_based": [53, 44, 13, 74, 80, 134, 136, 215]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_908_1to4_e27/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_908_1to4_e27/metrics.json)。

### n9_T31_train_2928_5to2_e3 · clean_delta_top8 vs baseline_R1_delta_top8 · loss

目标类；F1 64.89323 → 42.30849；TP/FP/FN [7111, 4700, 2994] → [7111, 16399, 2994]。原/净池GT覆盖 9615/9615，原池未覆盖 490，清理覆盖损失 0，当前TopK未取用的池覆盖 2504。
候选ID（0起始）：{"pool_ids_zero_based": [0, 13, 29, 49, 54, 7, 68, 32], "original_geometry_ids_zero_based": [0, 14, 31, 51, 57, 8, 71, 34], "raw_sam_ids_zero_based": [0, 27, 68, 116, 131, 15, 168, 74]}；参照：{"pool_ids_zero_based": [7, 55, 27, 0, 14, 31, 51, 57], "original_geometry_ids_zero_based": [7, 55, 27, 0, 14, 31, 51, 57], "raw_sam_ids_zero_based": [14, 122, 62, 0, 27, 68, 116, 131]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2928_5to2_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2928_5to2_e3/metrics.json)。

### n9_T31_train_786_1to5_e1 · clean_delta_top8 vs baseline_R1_delta_top8 · loss

目标类；F1 16.50026 → 0.17331；TP/FP/FN [157, 1589, 0] → [2, 2149, 155]。原/净池GT覆盖 157/153，原池未覆盖 0，清理覆盖损失 4，当前TopK未取用的池覆盖 151。
候选ID（0起始）：{"pool_ids_zero_based": [164, 250, 293, 24, 132, 303, 149, 131], "original_geometry_ids_zero_based": [165, 252, 295, 24, 133, 305, 150, 132], "raw_sam_ids_zero_based": [266, 389, 447, 60, 219, 460, 245, 218]}；参照：{"pool_ids_zero_based": [128, 199, 165, 252, 295, 24, 133, 305], "original_geometry_ids_zero_based": [128, 199, 165, 252, 295, 24, 133, 305], "raw_sam_ids_zero_based": [213, 312, 266, 389, 447, 60, 219, 460]}。
去自匹配/净区处理后池GT覆盖下降：有可量化候选支持损失，但JSON不能区分具体被剔除候选语义。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_786_1to5_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_786_1to5_e1/metrics.json)。

### n9_T31_train_1798_2to5_e9 · clean_delta_top8 vs baseline_R1_delta_top8 · win

目标类；F1 14.52831 → 71.63834；TP/FP/FN [3328, 8588, 30570] → [23564, 8324, 10334]。原/净池GT覆盖 26232/26232，原池未覆盖 7666，清理覆盖损失 0，当前TopK未取用的池覆盖 2668。
候选ID（0起始）：{"pool_ids_zero_based": [19, 4, 20, 11, 13, 27, 30, 23], "original_geometry_ids_zero_based": [22, 5, 23, 14, 16, 31, 35, 27], "raw_sam_ids_zero_based": [107, 27, 108, 80, 83, 140, 166, 117]}；参照：{"pool_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "original_geometry_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "raw_sam_ids_zero_based": [206, 149, 116, 78, 67, 3, 107, 27]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/metrics.json)。

### n9_T31_train_555_3to5_e2 · clean_delta_top8 vs baseline_R1_delta_top8 · win

目标类；F1 15.94002 → 29.11960；TP/FP/FN [287, 1722, 1305] → [812, 3173, 780]。原/净池GT覆盖 1041/1041，原池未覆盖 551，清理覆盖损失 0，当前TopK未取用的池覆盖 229。
候选ID（0起始）：{"pool_ids_zero_based": [64, 6, 88, 48, 21, 1, 85, 68], "original_geometry_ids_zero_based": [66, 6, 90, 50, 21, 1, 87, 70], "raw_sam_ids_zero_based": [204, 24, 279, 137, 61, 8, 266, 217]}；参照：{"pool_ids_zero_based": [39, 40, 92, 66, 6, 90, 50, 21], "original_geometry_ids_zero_based": [39, 40, 92, 66, 6, 90, 50, 21], "raw_sam_ids_zero_based": [105, 109, 286, 204, 24, 279, 137, 61]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_555_3to5_e2/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_555_3to5_e2/metrics.json)。

### n9_T31_train_2822_1to5_e1 · clean_joint_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 19.23488 → 2.77585；TP/FP/FN [4412, 3992, 33059] → [581, 3809, 36890]。原/净池GT覆盖 5555/5555，原池未覆盖 31916，清理覆盖损失 0，当前TopK未取用的池覆盖 4974。
候选ID（0起始）：{"pool_ids_zero_based": [25, 30, 17, 35, 11], "original_geometry_ids_zero_based": [25, 30, 17, 36, 11], "raw_sam_ids_zero_based": [87, 95, 75, 106, 57]}；参照：{"pool_ids_zero_based": [33, 25, 31, 17, 39, 30, 11, 36], "original_geometry_ids_zero_based": [33, 25, 31, 17, 39, 30, 11, 36], "raw_sam_ids_zero_based": [101, 87, 96, 75, 111, 95, 57, 106]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2822_1to5_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2822_1to5_e1/metrics.json)。

### n9_T31_train_1754_2to1_e2 · clean_joint_top5 vs baseline_R1_delta_top8 · loss

地表类；F1 12.34896 → 0.00000；TP/FP/FN [930, 10732, 2470] → [0, 4846, 3400]。原/净池GT覆盖 1632/1632，原池未覆盖 1768，清理覆盖损失 0，当前TopK未取用的池覆盖 1632。
候选ID（0起始）：{"pool_ids_zero_based": [170, 81, 153, 23, 9], "original_geometry_ids_zero_based": [171, 82, 154, 23, 9], "raw_sam_ids_zero_based": [309, 135, 264, 33, 12]}；参照：{"pool_ids_zero_based": [75, 171, 23, 84, 82, 95, 154, 20], "original_geometry_ids_zero_based": [75, 171, 23, 84, 82, 95, 154, 20], "raw_sam_ids_zero_based": [122, 309, 33, 142, 135, 167, 264, 30]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1754_2to1_e2/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1754_2to1_e2/metrics.json)。

### n9_T31_train_1798_2to5_e9 · clean_joint_top5 vs baseline_R1_delta_top8 · win

目标类；F1 14.52831 → 64.97752；TP/FP/FN [3328, 8588, 30570] → [19362, 6336, 14536]。原/净池GT覆盖 26232/26232，原池未覆盖 7666，清理覆盖损失 0，当前TopK未取用的池覆盖 6870。
候选ID（0起始）：{"pool_ids_zero_based": [20, 4, 19, 13, 11], "original_geometry_ids_zero_based": [23, 5, 22, 16, 14], "raw_sam_ids_zero_based": [108, 27, 107, 83, 80]}；参照：{"pool_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "original_geometry_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "raw_sam_ids_zero_based": [206, 149, 116, 78, 67, 3, 107, 27]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/metrics.json)。

### n9_T31_train_443_1to5_e3 · clean_joint_top5 vs baseline_R1_delta_top8 · win

目标类；F1 49.89919 → 70.46263；TP/FP/FN [990, 1442, 546] → [990, 284, 546]。原/净池GT覆盖 990/990，原池未覆盖 546，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [82, 89, 98, 61, 109], "original_geometry_ids_zero_based": [83, 90, 99, 61, 110], "raw_sam_ids_zero_based": [336, 352, 367, 272, 388]}；参照：{"pool_ids_zero_based": [70, 83, 90, 99, 61, 110, 111, 71], "original_geometry_ids_zero_based": [70, 83, 90, 99, 61, 110, 111, 71], "raw_sam_ids_zero_based": [307, 336, 352, 367, 272, 388, 389, 308]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_443_1to5_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_443_1to5_e3/metrics.json)。

### n9_T31_train_2928_5to2_e3 · clean_joint_top8 vs baseline_R1_delta_top8 · loss

目标类；F1 64.89323 → 49.98771；TP/FP/FN [7111, 4700, 2994] → [7117, 11253, 2988]。原/净池GT覆盖 9615/9615，原池未覆盖 490，清理覆盖损失 0，当前TopK未取用的池覆盖 2498。
候选ID（0起始）：{"pool_ids_zero_based": [0, 29, 49, 54, 13, 7, 68, 21], "original_geometry_ids_zero_based": [0, 31, 51, 57, 14, 8, 71, 22], "raw_sam_ids_zero_based": [0, 68, 116, 131, 27, 15, 168, 45]}；参照：{"pool_ids_zero_based": [7, 55, 27, 0, 14, 31, 51, 57], "original_geometry_ids_zero_based": [7, 55, 27, 0, 14, 31, 51, 57], "raw_sam_ids_zero_based": [14, 122, 62, 0, 27, 68, 116, 131]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2928_5to2_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2928_5to2_e3/metrics.json)。

### n9_T31_train_1754_2to1_e2 · clean_joint_top8 vs baseline_R1_delta_top8 · loss

地表类；F1 12.34896 → 0.00000；TP/FP/FN [930, 10732, 2470] → [0, 16433, 3400]。原/净池GT覆盖 1632/1632，原池未覆盖 1768，清理覆盖损失 0，当前TopK未取用的池覆盖 1632。
候选ID（0起始）：{"pool_ids_zero_based": [170, 81, 153, 23, 9, 168, 123, 83], "original_geometry_ids_zero_based": [171, 82, 154, 23, 9, 169, 124, 84], "raw_sam_ids_zero_based": [309, 135, 264, 33, 12, 303, 220, 142]}；参照：{"pool_ids_zero_based": [75, 171, 23, 84, 82, 95, 154, 20], "original_geometry_ids_zero_based": [75, 171, 23, 84, 82, 95, 154, 20], "raw_sam_ids_zero_based": [122, 309, 33, 142, 135, 167, 264, 30]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1754_2to1_e2/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1754_2to1_e2/metrics.json)。

### n9_T31_train_1798_2to5_e9 · clean_joint_top8 vs baseline_R1_delta_top8 · win

目标类；F1 14.52831 → 63.97412；TP/FP/FN [3328, 8588, 30570] → [21653, 12142, 12245]。原/净池GT覆盖 26232/26232，原池未覆盖 7666，清理覆盖损失 0，当前TopK未取用的池覆盖 4579。
候选ID（0起始）：{"pool_ids_zero_based": [20, 4, 19, 13, 11, 23, 3, 27], "original_geometry_ids_zero_based": [23, 5, 22, 16, 14, 27, 4, 31], "raw_sam_ids_zero_based": [108, 27, 107, 83, 80, 117, 20, 140]}；参照：{"pool_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "original_geometry_ids_zero_based": [40, 34, 26, 13, 11, 1, 22, 5], "raw_sam_ids_zero_based": [206, 149, 116, 78, 67, 3, 107, 27]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_1798_2to5_e9/metrics.json)。

### n9_T31_train_127_3to1_e3 · clean_joint_top8 vs baseline_R1_delta_top8 · win

地表类；F1 12.88576 → 26.36741；TP/FP/FN [476, 4406, 2030] → [699, 2097, 1807]。原/净池GT覆盖 703/703，原池未覆盖 1803，清理覆盖损失 0，当前TopK未取用的池覆盖 4。
候选ID（0起始）：{"pool_ids_zero_based": [35, 21, 63, 8, 25, 89, 29, 24], "original_geometry_ids_zero_based": [35, 21, 64, 8, 25, 90, 29, 24], "raw_sam_ids_zero_based": [194, 128, 348, 60, 151, 471, 169, 146]}；参照：{"pool_ids_zero_based": [43, 35, 21, 70, 3, 90, 24, 78], "original_geometry_ids_zero_based": [43, 35, 21, 70, 3, 90, 24, 78], "raw_sam_ids_zero_based": [227, 194, 128, 376, 36, 471, 146, 422]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_3to1_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_3to1_e3/metrics.json)。

### n9_T31_train_908_1to4_e27 · clean_delta_top8 vs clean_delta_top5 · loss

地表类；F1 50.63639 → 20.19759；TP/FP/FN [736, 837, 598] → [736, 5218, 598]。原/净池GT覆盖 736/736，原池未覆盖 598，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [22, 4, 35, 37, 51, 52, 74, 67], "original_geometry_ids_zero_based": [22, 4, 36, 38, 52, 53, 75, 68], "raw_sam_ids_zero_based": [44, 13, 74, 80, 134, 136, 215, 181]}；参照：{"pool_ids_zero_based": [22, 4, 35, 37, 51], "original_geometry_ids_zero_based": [22, 4, 36, 38, 52], "raw_sam_ids_zero_based": [44, 13, 74, 80, 134]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_908_1to4_e27/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_908_1to4_e27/metrics.json)。

### n9_T31_train_443_1to5_e3 · clean_delta_top8 vs clean_delta_top5 · loss

目标类；F1 70.46263 → 45.02046；TP/FP/FN [990, 284, 546] → [990, 1872, 546]。原/净池GT覆盖 990/990，原池未覆盖 546，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [82, 89, 98, 61, 109, 110, 70, 63], "original_geometry_ids_zero_based": [83, 90, 99, 61, 110, 111, 71, 63], "raw_sam_ids_zero_based": [336, 352, 367, 272, 388, 389, 308, 275]}；参照：{"pool_ids_zero_based": [82, 89, 98, 61, 109], "original_geometry_ids_zero_based": [83, 90, 99, 61, 110], "raw_sam_ids_zero_based": [336, 352, 367, 272, 388]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_443_1to5_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_443_1to5_e3/metrics.json)。

### n9_T31_train_127_2to5_e11 · clean_delta_top8 vs clean_delta_top5 · win

目标类；F1 2.07188 → 19.72737；TP/FP/FN [341, 1425, 30810] → [3618, 1911, 27533]。原/净池GT覆盖 20340/20340，原池未覆盖 10811，清理覆盖损失 0，当前TopK未取用的池覆盖 16722。
候选ID（0起始）：{"pool_ids_zero_based": [207, 3, 151, 51, 106, 223, 89, 132], "original_geometry_ids_zero_based": [210, 3, 154, 52, 109, 226, 90, 135], "raw_sam_ids_zero_based": [507, 13, 385, 164, 307, 571, 263, 343]}；参照：{"pool_ids_zero_based": [207, 3, 151, 51, 106], "original_geometry_ids_zero_based": [210, 3, 154, 52, 109], "raw_sam_ids_zero_based": [507, 13, 385, 164, 307]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_2to5_e11/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_2to5_e11/metrics.json)。

### n9_T31_train_555_3to5_e2 · clean_delta_top8 vs clean_delta_top5 · win

目标类；F1 16.65699 → 29.11960；TP/FP/FN [287, 1567, 1305] → [812, 3173, 780]。原/净池GT覆盖 1041/1041，原池未覆盖 551，清理覆盖损失 0，当前TopK未取用的池覆盖 229。
候选ID（0起始）：{"pool_ids_zero_based": [64, 6, 88, 48, 21, 1, 85, 68], "original_geometry_ids_zero_based": [66, 6, 90, 50, 21, 1, 87, 70], "raw_sam_ids_zero_based": [204, 24, 279, 137, 61, 8, 266, 217]}；参照：{"pool_ids_zero_based": [64, 6, 88, 48, 21], "original_geometry_ids_zero_based": [66, 6, 90, 50, 21], "raw_sam_ids_zero_based": [204, 24, 279, 137, 61]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_555_3to5_e2/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_555_3to5_e2/metrics.json)。

### n9_T31_train_443_1to5_e3 · clean_joint_top8 vs clean_joint_top5 · loss

目标类；F1 70.46263 → 45.02046；TP/FP/FN [990, 284, 546] → [990, 1872, 546]。原/净池GT覆盖 990/990，原池未覆盖 546，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [82, 89, 98, 61, 109, 110, 63, 70], "original_geometry_ids_zero_based": [83, 90, 99, 61, 110, 111, 63, 71], "raw_sam_ids_zero_based": [336, 352, 367, 272, 388, 389, 275, 308]}；参照：{"pool_ids_zero_based": [82, 89, 98, 61, 109], "original_geometry_ids_zero_based": [83, 90, 99, 61, 110], "raw_sam_ids_zero_based": [336, 352, 367, 272, 388]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_443_1to5_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_443_1to5_e3/metrics.json)。

### n9_T31_train_2928_5to2_e3 · clean_joint_top8 vs clean_joint_top5 · loss

目标类；F1 65.63596 → 49.98771；TP/FP/FN [7111, 4452, 2994] → [7117, 11253, 2988]。原/净池GT覆盖 9615/9615，原池未覆盖 490，清理覆盖损失 0，当前TopK未取用的池覆盖 2498。
候选ID（0起始）：{"pool_ids_zero_based": [0, 29, 49, 54, 13, 7, 68, 21], "original_geometry_ids_zero_based": [0, 31, 51, 57, 14, 8, 71, 22], "raw_sam_ids_zero_based": [0, 68, 116, 131, 27, 15, 168, 45]}；参照：{"pool_ids_zero_based": [0, 29, 49, 54, 13], "original_geometry_ids_zero_based": [0, 31, 51, 57, 14], "raw_sam_ids_zero_based": [0, 68, 116, 131, 27]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2928_5to2_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_2928_5to2_e3/metrics.json)。

### n9_T31_train_127_2to5_e11 · clean_joint_top8 vs clean_joint_top5 · win

目标类；F1 3.52368 → 21.80315；TP/FP/FN [589, 1691, 30562] → [4035, 1827, 27116]。原/净池GT覆盖 20340/20340，原池未覆盖 10811，清理覆盖损失 0，当前TopK未取用的池覆盖 16305。
候选ID（0起始）：{"pool_ids_zero_based": [89, 207, 160, 151, 3, 223, 205, 103], "original_geometry_ids_zero_based": [90, 210, 163, 154, 3, 226, 208, 105], "raw_sam_ids_zero_based": [263, 507, 401, 385, 13, 571, 497, 300]}；参照：{"pool_ids_zero_based": [89, 207, 160, 151, 3], "original_geometry_ids_zero_based": [90, 210, 163, 154, 3], "raw_sam_ids_zero_based": [263, 507, 401, 385, 13]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_2to5_e11/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_127_2to5_e11/metrics.json)。

### n9_T31_train_908_1to5_e3 · clean_joint_top8 vs clean_joint_top5 · win

目标类；F1 46.33016 → 55.79668；TP/FP/FN [8247, 5770, 13337] → [10670, 5992, 10914]。原/净池GT覆盖 17768/17768，原池未覆盖 3816，清理覆盖损失 0，当前TopK未取用的池覆盖 7098。
候选ID（0起始）：{"pool_ids_zero_based": [35, 58, 67, 38, 66, 54, 69, 70], "original_geometry_ids_zero_based": [35, 59, 68, 38, 67, 55, 70, 71], "raw_sam_ids_zero_based": [111, 168, 208, 115, 199, 154, 211, 215]}；参照：{"pool_ids_zero_based": [35, 58, 67, 38, 66], "original_geometry_ids_zero_based": [35, 59, 68, 38, 67], "raw_sam_ids_zero_based": [111, 168, 208, 115, 199]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_908_1to5_e3/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T31_train_908_1to5_e3/metrics.json)。

### n9_T33_train_145_2to5_e4 · clean_delta_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 70.31727 → 35.04180；TP/FP/FN [21188, 7529, 10359] → [7083, 1796, 24464]。原/净池GT覆盖 29012/29012，原池未覆盖 2535，清理覆盖损失 0，当前TopK未取用的池覆盖 21929。
候选ID（0起始）：{"pool_ids_zero_based": [30, 16, 22, 3, 27], "original_geometry_ids_zero_based": [30, 16, 22, 3, 27], "raw_sam_ids_zero_based": [120, 70, 80, 13, 105]}；参照：{"pool_ids_zero_based": [30, 16, 22, 3, 27, 19, 13, 6], "original_geometry_ids_zero_based": [30, 16, 22, 3, 27, 19, 13, 6], "raw_sam_ids_zero_based": [120, 70, 80, 13, 105, 73, 44, 24]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/metrics.json)。

### n9_T33_train_2571_1to5_e5 · clean_delta_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 34.33993 → 0.00000；TP/FP/FN [4054, 13286, 2217] → [0, 9950, 6271]。原/净池GT覆盖 5786/5786，原池未覆盖 485，清理覆盖损失 0，当前TopK未取用的池覆盖 5786。
候选ID（0起始）：{"pool_ids_zero_based": [80, 106, 67, 103, 81], "original_geometry_ids_zero_based": [80, 106, 67, 103, 81], "raw_sam_ids_zero_based": [172, 221, 145, 215, 173]}；参照：{"pool_ids_zero_based": [80, 106, 67, 103, 81, 45, 65, 60], "original_geometry_ids_zero_based": [80, 106, 67, 103, 81, 45, 65, 60], "raw_sam_ids_zero_based": [172, 221, 145, 215, 173, 114, 143, 136]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_2571_1to5_e5/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_2571_1to5_e5/metrics.json)。

### n9_T33_train_1659_2to3_e6 · clean_delta_top5 vs baseline_R1_delta_top8 · win

地表类；F1 22.51537 → 34.37831；TP/FP/FN [1135, 7428, 384] → [1135, 3949, 384]。原/净池GT覆盖 1135/1135，原池未覆盖 384，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [7, 6, 21, 0, 17], "original_geometry_ids_zero_based": [7, 6, 21, 0, 17], "raw_sam_ids_zero_based": [45, 44, 132, 6, 114]}；参照：{"pool_ids_zero_based": [7, 6, 21, 0, 17, 18, 12, 3], "original_geometry_ids_zero_based": [7, 6, 21, 0, 17, 18, 12, 3], "raw_sam_ids_zero_based": [45, 44, 132, 6, 114, 116, 94, 18]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_2to3_e6/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_2to3_e6/metrics.json)。

### n9_T33_train_1659_1to2_e1 · clean_delta_top5 vs baseline_R1_delta_top8 · win

地表类；F1 33.39857 → 44.22513；TP/FP/FN [2901, 6401, 5169] → [2868, 2032, 5202]。原/净池GT覆盖 6506/6506，原池未覆盖 1564，清理覆盖损失 0，当前TopK未取用的池覆盖 3638。
候选ID（0起始）：{"pool_ids_zero_based": [13, 42, 30, 41, 22], "original_geometry_ids_zero_based": [13, 42, 30, 41, 22], "raw_sam_ids_zero_based": [55, 112, 93, 110, 81]}；参照：{"pool_ids_zero_based": [13, 42, 30, 41, 22, 45, 10, 24], "original_geometry_ids_zero_based": [13, 42, 30, 41, 22, 45, 10, 24], "raw_sam_ids_zero_based": [55, 112, 93, 110, 81, 123, 43, 85]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_1to2_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_1to2_e1/metrics.json)。

### n9_T33_train_2571_2to5_e1 · clean_joint_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 45.03550 → 7.08038；TP/FP/FN [11133, 9206, 17969] → [1190, 3322, 27912]。原/净池GT覆盖 23886/23886，原池未覆盖 5216，清理覆盖损失 0，当前TopK未取用的池覆盖 22696。
候选ID（0起始）：{"pool_ids_zero_based": [0, 79, 1, 86, 48], "original_geometry_ids_zero_based": [0, 79, 1, 86, 48], "raw_sam_ids_zero_based": [3, 225, 5, 254, 136]}；参照：{"pool_ids_zero_based": [8, 5, 53, 58, 83, 38, 50, 47], "original_geometry_ids_zero_based": [8, 5, 53, 58, 83, 38, 50, 47], "raw_sam_ids_zero_based": [21, 15, 145, 167, 236, 120, 138, 134]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_2571_2to5_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_2571_2to5_e1/metrics.json)。

### n9_T33_train_145_2to5_e4 · clean_joint_top5 vs baseline_R1_delta_top8 · loss

目标类；F1 70.31727 → 35.04180；TP/FP/FN [21188, 7529, 10359] → [7083, 1796, 24464]。原/净池GT覆盖 29012/29012，原池未覆盖 2535，清理覆盖损失 0，当前TopK未取用的池覆盖 21929。
候选ID（0起始）：{"pool_ids_zero_based": [30, 22, 3, 16, 27], "original_geometry_ids_zero_based": [30, 22, 3, 16, 27], "raw_sam_ids_zero_based": [120, 80, 13, 70, 105]}；参照：{"pool_ids_zero_based": [30, 16, 22, 3, 27, 19, 13, 6], "original_geometry_ids_zero_based": [30, 16, 22, 3, 27, 19, 13, 6], "raw_sam_ids_zero_based": [120, 70, 80, 13, 105, 73, 44, 24]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/metrics.json)。

### n9_T33_train_1136_5to2_e5 · clean_joint_top5 vs baseline_R1_delta_top8 · win

目标类；F1 0.00000 → 40.91091；TP/FP/FN [0, 1572, 1000] → [512, 991, 488]。原/净池GT覆盖 659/659，原池未覆盖 341，清理覆盖损失 0，当前TopK未取用的池覆盖 147。
候选ID（0起始）：{"pool_ids_zero_based": [55, 30, 49, 7, 15], "original_geometry_ids_zero_based": [55, 30, 49, 7, 15], "raw_sam_ids_zero_based": [134, 82, 118, 21, 36]}；参照：{"pool_ids_zero_based": [30, 62, 50, 55, 49, 27, 54, 51], "original_geometry_ids_zero_based": [30, 62, 50, 55, 49, 27, 54, 51], "raw_sam_ids_zero_based": [82, 156, 121, 134, 118, 76, 133, 122]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1136_5to2_e5/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1136_5to2_e5/metrics.json)。

### n9_T33_train_1808_5to3_e1 · clean_joint_top5 vs baseline_R1_delta_top8 · win

目标类；F1 0.00000 → 23.67242；TP/FP/FN [0, 3137, 210] → [185, 1168, 25]。原/净池GT覆盖 185/185，原池未覆盖 25，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [27, 12, 19, 31, 11], "original_geometry_ids_zero_based": [27, 12, 19, 31, 11], "raw_sam_ids_zero_based": [114, 43, 55, 139, 42]}；参照：{"pool_ids_zero_based": [27, 19, 1, 3, 11, 31, 14, 2], "original_geometry_ids_zero_based": [27, 19, 1, 3, 11, 31, 14, 2], "raw_sam_ids_zero_based": [114, 55, 14, 23, 42, 139, 45, 17]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1808_5to3_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1808_5to3_e1/metrics.json)。

### n9_T33_train_1659_2to3_e6 · clean_joint_top8 vs baseline_R1_delta_top8 · loss

地表类；F1 22.51537 → 0.00000；TP/FP/FN [1135, 7428, 384] → [0, 10414, 1519]。原/净池GT覆盖 1135/1135，原池未覆盖 384，清理覆盖损失 0，当前TopK未取用的池覆盖 1135。
候选ID（0起始）：{"pool_ids_zero_based": [7, 17, 12, 3, 18, 6, 10, 8], "original_geometry_ids_zero_based": [7, 17, 12, 3, 18, 6, 10, 8], "raw_sam_ids_zero_based": [45, 114, 94, 18, 116, 44, 76, 50]}；参照：{"pool_ids_zero_based": [7, 6, 21, 0, 17, 18, 12, 3], "original_geometry_ids_zero_based": [7, 6, 21, 0, 17, 18, 12, 3], "raw_sam_ids_zero_based": [45, 44, 132, 6, 114, 116, 94, 18]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_2to3_e6/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_2to3_e6/metrics.json)。

### n9_T33_train_1065_3to5_e1 · clean_joint_top8 vs baseline_R1_delta_top8 · loss

目标类；F1 23.03014 → 12.57115；TP/FP/FN [1761, 9501, 2270] → [1270, 14904, 2761]。原/净池GT覆盖 1761/1761，原池未覆盖 2270，清理覆盖损失 0，当前TopK未取用的池覆盖 491。
候选ID（0起始）：{"pool_ids_zero_based": [21, 31, 24, 50, 63, 9, 3, 58], "original_geometry_ids_zero_based": [21, 31, 24, 50, 63, 9, 3, 58], "raw_sam_ids_zero_based": [68, 86, 72, 123, 168, 29, 17, 155]}；参照：{"pool_ids_zero_based": [21, 24, 31, 52, 59, 57, 20, 67], "original_geometry_ids_zero_based": [21, 24, 31, 52, 59, 57, 20, 67], "raw_sam_ids_zero_based": [68, 72, 86, 129, 157, 154, 67, 190]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1065_3to5_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1065_3to5_e1/metrics.json)。

### n9_T33_train_555_5to3_e4 · clean_joint_top8 vs baseline_R1_delta_top8 · win

目标类；F1 0.00000 → 46.03225；TP/FP/FN [0, 1182, 733] → [728, 1702, 5]。原/净池GT覆盖 728/728，原池未覆盖 5，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [16, 50, 37, 55, 14, 4, 1, 22], "original_geometry_ids_zero_based": [16, 50, 37, 55, 14, 4, 1, 22], "raw_sam_ids_zero_based": [79, 257, 177, 286, 74, 46, 27, 94]}；参照：{"pool_ids_zero_based": [16, 37, 50, 55, 22, 18, 14, 53], "original_geometry_ids_zero_based": [16, 37, 50, 55, 22, 18, 14, 53], "raw_sam_ids_zero_based": [79, 177, 257, 286, 94, 84, 74, 266]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_555_5to3_e4/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_555_5to3_e4/metrics.json)。

### n9_T33_train_1136_5to2_e5 · clean_joint_top8 vs baseline_R1_delta_top8 · win

目标类；F1 0.00000 → 33.06426；TP/FP/FN [0, 1572, 1000] → [512, 1585, 488]。原/净池GT覆盖 659/659，原池未覆盖 341，清理覆盖损失 0，当前TopK未取用的池覆盖 147。
候选ID（0起始）：{"pool_ids_zero_based": [55, 30, 49, 7, 15, 50, 27, 54], "original_geometry_ids_zero_based": [55, 30, 49, 7, 15, 50, 27, 54], "raw_sam_ids_zero_based": [134, 82, 118, 21, 36, 121, 76, 133]}；参照：{"pool_ids_zero_based": [30, 62, 50, 55, 49, 27, 54, 51], "original_geometry_ids_zero_based": [30, 62, 50, 55, 49, 27, 54, 51], "raw_sam_ids_zero_based": [82, 156, 121, 134, 118, 76, 133, 122]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1136_5to2_e5/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1136_5to2_e5/metrics.json)。

### n9_T33_train_1659_2to3_e6 · clean_delta_top8 vs clean_delta_top5 · loss

地表类；F1 34.37831 → 22.51537；TP/FP/FN [1135, 3949, 384] → [1135, 7428, 384]。原/净池GT覆盖 1135/1135，原池未覆盖 384，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [7, 6, 21, 0, 17, 18, 12, 3], "original_geometry_ids_zero_based": [7, 6, 21, 0, 17, 18, 12, 3], "raw_sam_ids_zero_based": [45, 44, 132, 6, 114, 116, 94, 18]}；参照：{"pool_ids_zero_based": [7, 6, 21, 0, 17], "original_geometry_ids_zero_based": [7, 6, 21, 0, 17], "raw_sam_ids_zero_based": [45, 44, 132, 6, 114]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_2to3_e6/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_2to3_e6/metrics.json)。

### n9_T33_train_1659_1to2_e1 · clean_delta_top8 vs clean_delta_top5 · loss

地表类；F1 44.22513 → 33.39857；TP/FP/FN [2868, 2032, 5202] → [2901, 6401, 5169]。原/净池GT覆盖 6506/6506，原池未覆盖 1564，清理覆盖损失 0，当前TopK未取用的池覆盖 3605。
候选ID（0起始）：{"pool_ids_zero_based": [13, 42, 30, 41, 22, 45, 10, 24], "original_geometry_ids_zero_based": [13, 42, 30, 41, 22, 45, 10, 24], "raw_sam_ids_zero_based": [55, 112, 93, 110, 81, 123, 43, 85]}；参照：{"pool_ids_zero_based": [13, 42, 30, 41, 22], "original_geometry_ids_zero_based": [13, 42, 30, 41, 22], "raw_sam_ids_zero_based": [55, 112, 93, 110, 81]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_1to2_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1659_1to2_e1/metrics.json)。

### n9_T33_train_145_2to5_e4 · clean_delta_top8 vs clean_delta_top5 · win

目标类；F1 35.04180 → 70.31727；TP/FP/FN [7083, 1796, 24464] → [21188, 7529, 10359]。原/净池GT覆盖 29012/29012，原池未覆盖 2535，清理覆盖损失 0，当前TopK未取用的池覆盖 7824。
候选ID（0起始）：{"pool_ids_zero_based": [30, 16, 22, 3, 27, 19, 13, 6], "original_geometry_ids_zero_based": [30, 16, 22, 3, 27, 19, 13, 6], "raw_sam_ids_zero_based": [120, 70, 80, 13, 105, 73, 44, 24]}；参照：{"pool_ids_zero_based": [30, 16, 22, 3, 27], "original_geometry_ids_zero_based": [30, 16, 22, 3, 27], "raw_sam_ids_zero_based": [120, 70, 80, 13, 105]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/metrics.json)。

### n9_T33_train_2571_1to5_e5 · clean_delta_top8 vs clean_delta_top5 · win

目标类；F1 0.00000 → 34.33993；TP/FP/FN [0, 9950, 6271] → [4054, 13286, 2217]。原/净池GT覆盖 5786/5786，原池未覆盖 485，清理覆盖损失 0，当前TopK未取用的池覆盖 1732。
候选ID（0起始）：{"pool_ids_zero_based": [80, 106, 67, 103, 81, 45, 65, 60], "original_geometry_ids_zero_based": [80, 106, 67, 103, 81, 45, 65, 60], "raw_sam_ids_zero_based": [172, 221, 145, 215, 173, 114, 143, 136]}；参照：{"pool_ids_zero_based": [80, 106, 67, 103, 81], "original_geometry_ids_zero_based": [80, 106, 67, 103, 81], "raw_sam_ids_zero_based": [172, 221, 145, 215, 173]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_2571_1to5_e5/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_2571_1to5_e5/metrics.json)。

### n9_T33_train_1808_5to3_e1 · clean_joint_top8 vs clean_joint_top5 · loss

目标类；F1 23.67242 → 10.45493；TP/FP/FN [185, 1168, 25] → [185, 3144, 25]。原/净池GT覆盖 185/185，原池未覆盖 25，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [27, 12, 19, 31, 11, 3, 1, 2], "original_geometry_ids_zero_based": [27, 12, 19, 31, 11, 3, 1, 2], "raw_sam_ids_zero_based": [114, 43, 55, 139, 42, 23, 14, 17]}；参照：{"pool_ids_zero_based": [27, 12, 19, 31, 11], "original_geometry_ids_zero_based": [27, 12, 19, 31, 11], "raw_sam_ids_zero_based": [114, 43, 55, 139, 42]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1808_5to3_e1/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_1808_5to3_e1/metrics.json)。

### n9_T33_train_119_2to5_e2 · clean_joint_top8 vs clean_joint_top5 · loss

目标类；F1 64.05721 → 52.86396；TP/FP/FN [851, 849, 106] → [886, 1509, 71]。原/净池GT覆盖 886/886，原池未覆盖 71，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [24, 29, 1, 11, 15, 19, 26, 20], "original_geometry_ids_zero_based": [24, 29, 1, 11, 15, 19, 26, 20], "raw_sam_ids_zero_based": [145, 154, 14, 62, 76, 85, 149, 92]}；参照：{"pool_ids_zero_based": [24, 29, 1, 11, 15], "original_geometry_ids_zero_based": [24, 29, 1, 11, 15], "raw_sam_ids_zero_based": [145, 154, 14, 62, 76]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_119_2to5_e2/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_119_2to5_e2/metrics.json)。

### n9_T33_train_555_5to3_e4 · clean_joint_top8 vs clean_joint_top5 · win

目标类；F1 0.00000 → 46.03225；TP/FP/FN [0, 898, 733] → [728, 1702, 5]。原/净池GT覆盖 728/728，原池未覆盖 5，清理覆盖损失 0，当前TopK未取用的池覆盖 0。
候选ID（0起始）：{"pool_ids_zero_based": [16, 50, 37, 55, 14, 4, 1, 22], "original_geometry_ids_zero_based": [16, 50, 37, 55, 14, 4, 1, 22], "raw_sam_ids_zero_based": [79, 257, 177, 286, 74, 46, 27, 94]}；参照：{"pool_ids_zero_based": [16, 50, 37, 55, 14], "original_geometry_ids_zero_based": [16, 50, 37, 55, 14], "raw_sam_ids_zero_based": [79, 257, 177, 286, 74]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_555_5to3_e4/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_555_5to3_e4/metrics.json)。

### n9_T33_train_145_2to5_e4 · clean_joint_top8 vs clean_joint_top5 · win

目标类；F1 35.04180 → 70.31727；TP/FP/FN [7083, 1796, 24464] → [21188, 7529, 10359]。原/净池GT覆盖 29012/29012，原池未覆盖 2535，清理覆盖损失 0，当前TopK未取用的池覆盖 7824。
候选ID（0起始）：{"pool_ids_zero_based": [30, 22, 3, 16, 27, 13, 19, 6], "original_geometry_ids_zero_based": [30, 22, 3, 16, 27, 13, 19, 6], "raw_sam_ids_zero_based": [120, 80, 13, 70, 105, 44, 73, 24]}；参照：{"pool_ids_zero_based": [30, 22, 3, 16, 27], "original_geometry_ids_zero_based": [30, 22, 3, 16, 27], "raw_sam_ids_zero_based": [120, 80, 13, 70, 105]}。
原几何池未覆盖全部GT：存在候选支持不足；不能仅归咎排序。 候选池覆盖GT多于TopK命中：存在排序/截断下的未取用支持；不证明可用单一候选或仅改排序即可补齐。
证据：[inference](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/inference.json)；[metrics](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116/exploratory_fixed_tune/tasks/n9_T33_train_145_2to5_e4/metrics.json)。

