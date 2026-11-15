# N7 T3 paired diagnostics

Failure attribution is count-based. Segmentation errors vs semantic errors require visual evidence; overlap is not causal evidence. Empty-task pixel F1 is zero under the unchanged historical convention, so use empty accuracy/FP separately.

| Split | Method | Positive TP retention | Positive F1 before | Positive F1 after | Empty FP before | Empty FP after | Image wins/losses/ties | Criteria |
|---|---|---:|---:|---:|---:|---:|---|---|
| tune | delta_threshold | 1.0 | 11.276883880000918 | 11.276883880000918 | 191988 | 174022 | 0/0/30 | not promoted |
| tune | joint_threshold | 1.0 | 12.12042979026825 | 12.339560061328577 | 193639 | 193639 | 0/0/30 | not promoted |
| confirm | delta_threshold | 1.0 | 16.04369540444158 | 16.04369540444158 | 133931 | 133931 | 0/0/30 | not promoted |
| confirm | joint_threshold | 1.0 | 15.352517674529219 | 15.352517674529219 | 146672 | 146672 | 0/0/30 | not promoted |

## Per-task losses and bottlenecks

| Split | Task | Rank | Pool covered | Top8 TP/FP/FN | Threshold TP/FP/FN | Verified TP/FP/FN | Count-based causes |
|---|---|---|---:|---|---|---|---|
| tune | train_1065_2to5 | delta | 0 | [0, 5427, 0] | [0, 5427, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1065_2to5 | joint | 0 | [0, 3532, 0] | [0, 3532, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1136_5to1 | delta | 0 | [0, 5445, 3] | [0, 5445, 3] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_1136_5to1 | joint | 0 | [0, 5043, 3] | [0, 5043, 3] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_1166_3to5 | delta | 0 | [0, 10722, 0] | [0, 10722, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1166_3to5 | joint | 0 | [0, 14310, 0] | [0, 14310, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_119_6to2 | delta | 0 | [0, 40930, 0] | [0, 40930, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_119_6to2 | joint | 0 | [0, 37409, 0] | [0, 37409, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_127_3to2 | delta | 0 | [0, 2024, 0] | [0, 2024, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_127_3to2 | joint | 0 | [0, 2248, 0] | [0, 2248, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1322_2to4 | delta | 0 | [0, 14871, 0] | [0, 14871, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1322_2to4 | joint | 0 | [0, 10300, 0] | [0, 10300, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1377_5to3 | delta | 0 | [0, 4574, 0] | [0, 4574, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1377_5to3 | joint | 0 | [0, 6748, 0] | [0, 6748, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_145_1to2 | delta | 0 | [0, 2567, 0] | [0, 2567, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_145_1to2 | joint | 0 | [0, 2567, 0] | [0, 2567, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_147_1to5 | delta | 0 | [0, 5757, 0] | [0, 5757, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_147_1to5 | joint | 0 | [0, 6459, 0] | [0, 6459, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1538_1to3 | delta | 5121 | [5121, 4863, 122] | [5121, 4863, 122] | None |  |
| tune | train_1538_1to3 | joint | 5121 | [5121, 4863, 122] | [5121, 4863, 122] | None |  |
| tune | train_1659_1to2 | delta | 3514 | [1927, 1755, 5793] | [1927, 1755, 5793] | None | Top8_truncation_loses_pool_coverage |
| tune | train_1659_1to2 | joint | 3514 | [1960, 1352, 5760] | [1960, 1352, 5760] | None | Top8_truncation_loses_pool_coverage |
| tune | train_1745_5to2 | delta | 1474 | [678, 1066, 11773] | [678, 1066, 11773] | None | Top8_truncation_loses_pool_coverage |
| tune | train_1745_5to2 | joint | 1474 | [678, 785, 11773] | [678, 785, 11773] | None | Top8_truncation_loses_pool_coverage |
| tune | train_1754_5to1 | delta | 311 | [0, 771, 6067] | [0, 771, 6067] | None | Top8_truncation_loses_pool_coverage |
| tune | train_1754_5to1 | joint | 311 | [0, 734, 6067] | [0, 734, 6067] | None | Top8_truncation_loses_pool_coverage |
| tune | train_1798_2to6 | delta | 0 | [0, 15025, 0] | [0, 15025, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1798_2to6 | joint | 0 | [0, 15030, 0] | [0, 15030, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_1808_2to1 | delta | 2100 | [2100, 64283, 5541] | [2100, 64283, 5541] | None |  |
| tune | train_1808_2to1 | joint | 2100 | [2100, 64283, 5541] | [2100, 64283, 5541] | None |  |
| tune | train_2571_4to5 | delta | 0 | [0, 5094, 6] | [0, 5094, 6] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_2571_4to5 | joint | 0 | [0, 4966, 6] | [0, 4966, 6] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_2677_2to1 | delta | 0 | [0, 23340, 0] | [0, 23340, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2677_2to1 | joint | 0 | [0, 23340, 0] | [0, 23340, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2822_1to2 | delta | 0 | [0, 8461, 0] | [0, 8461, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2822_1to2 | joint | 0 | [0, 8461, 0] | [0, 8461, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2848_1to4 | delta | 0 | [0, 24510, 0] | [0, 6544, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2848_1to4 | joint | 0 | [0, 24514, 0] | [0, 24514, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2890_5to1 | delta | 0 | [0, 13985, 492] | [0, 13985, 492] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_2890_5to1 | joint | 0 | [0, 13985, 492] | [0, 13985, 492] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_2912_5to2 | delta | 402 | [0, 13845, 429] | [0, 13845, 429] | None | Top8_truncation_loses_pool_coverage |
| tune | train_2912_5to2 | joint | 402 | [0, 8110, 429] | [0, 8110, 429] | None | Top8_truncation_loses_pool_coverage |
| tune | train_2928_3to1 | delta | 0 | [0, 10138, 0] | [0, 10138, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_2928_3to1 | joint | 0 | [0, 13529, 0] | [0, 13529, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_385_5to6 | delta | 0 | [0, 2942, 0] | [0, 2942, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_385_5to6 | joint | 0 | [0, 3738, 0] | [0, 3738, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_443_3to2 | delta | 0 | [0, 2855, 0] | [0, 2855, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_443_3to2 | joint | 0 | [0, 2868, 0] | [0, 2868, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_555_4to2 | delta | 0 | [0, 5259, 0] | [0, 5259, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_555_4to2 | joint | 0 | [0, 6071, 0] | [0, 6071, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_7_1to2 | delta | 0 | [0, 10131, 262] | [0, 10131, 262] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_7_1to2 | joint | 0 | [0, 5500, 262] | [0, 5500, 262] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_786_3to5 | delta | 0 | [0, 1605, 0] | [0, 1605, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_786_3to5 | joint | 0 | [0, 1647, 0] | [0, 1647, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_908_4to1 | delta | 0 | [0, 1307, 0] | [0, 1307, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_908_4to1 | joint | 0 | [0, 1194, 0] | [0, 1194, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_915_3to4 | delta | 0 | [0, 2889, 1] | [0, 2889, 1] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_915_3to4 | joint | 0 | [0, 2889, 1] | [0, 0, 1] | None | original_geometry_pool_zero_GT_coverage |
| tune | train_987_3to5 | delta | 0 | [0, 9674, 0] | [0, 9674, 0] | None | similarity_gate_retains_empty_target_FP |
| tune | train_987_3to5 | joint | 0 | [0, 9674, 0] | [0, 9674, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1129_4to5 | delta | 0 | [0, 8423, 0] | [0, 8423, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1129_4to5 | joint | 0 | [0, 10434, 0] | [0, 10434, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1177_2to3 | delta | 235 | [0, 19815, 1686] | [0, 19815, 1686] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_1177_2to3 | joint | 235 | [0, 8300, 1686] | [0, 8300, 1686] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_1344_2to5 | delta | 337 | [176, 2860, 15880] | [176, 2860, 15880] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_1344_2to5 | joint | 337 | [176, 2860, 15880] | [176, 2860, 15880] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_1370_3to1 | delta | 499 | [241, 8319, 550] | [241, 8319, 550] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_1370_3to1 | joint | 499 | [499, 6960, 292] | [499, 6960, 292] | None |  |
| confirm | train_1496_2to1 | delta | 4 | [4, 103972, 0] | [4, 103972, 0] | None |  |
| confirm | train_1496_2to1 | joint | 4 | [4, 103972, 0] | [4, 103972, 0] | None |  |
| confirm | train_1527_5to2 | delta | 0 | [0, 11422, 0] | [0, 11422, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1527_5to2 | joint | 0 | [0, 10063, 0] | [0, 10063, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1535_4to1 | delta | 14 | [14, 31049, 0] | [14, 31049, 0] | None |  |
| confirm | train_1535_4to1 | joint | 14 | [14, 31049, 0] | [14, 31049, 0] | None |  |
| confirm | train_156_2to3 | delta | 0 | [0, 41300, 0] | [0, 41300, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_156_2to3 | joint | 0 | [0, 43641, 0] | [0, 43641, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1884_4to3 | delta | 0 | [0, 7023, 0] | [0, 7023, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_1884_4to3 | joint | 0 | [0, 7023, 0] | [0, 7023, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2043_2to3 | delta | 369 | [337, 1181, 49] | [337, 1181, 49] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2043_2to3 | joint | 369 | [337, 1222, 49] | [337, 1222, 49] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_206_3to4 | delta | 0 | [0, 3030, 0] | [0, 3030, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_206_3to4 | joint | 0 | [0, 3371, 0] | [0, 3371, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2075_4to2 | delta | 0 | [0, 10748, 0] | [0, 10748, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2075_4to2 | joint | 0 | [0, 14847, 0] | [0, 14847, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2130_2to4 | delta | 0 | [0, 5682, 0] | [0, 5682, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2130_2to4 | joint | 0 | [0, 5706, 0] | [0, 5706, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2192_1to5 | delta | 22527 | [17170, 4515, 8247] | [17170, 4515, 8247] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2192_1to5 | joint | 22527 | [21111, 12604, 4306] | [21111, 12604, 4306] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2227_3to2 | delta | 206 | [92, 2511, 724] | [92, 2511, 724] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2227_3to2 | joint | 206 | [92, 2843, 724] | [92, 2843, 724] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2255_2to5 | delta | 3031 | [3031, 11318, 828] | [3031, 11318, 828] | None |  |
| confirm | train_2255_2to5 | joint | 3031 | [3031, 11318, 828] | [3031, 11318, 828] | None |  |
| confirm | train_23_2to1 | delta | 2870 | [1973, 3528, 1262] | [1973, 3528, 1262] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_23_2to1 | joint | 2870 | [1973, 3602, 1262] | [1973, 3602, 1262] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2383_2to5 | delta | 14230 | [11776, 10185, 2792] | [11776, 10185, 2792] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2383_2to5 | joint | 14230 | [5447, 5024, 9121] | [5447, 5024, 9121] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2437_1to3 | delta | 0 | [0, 33664, 0] | [0, 33664, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2437_1to3 | joint | 0 | [0, 33664, 0] | [0, 33664, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_2438_5to1 | delta | 1573 | [1284, 2930, 11741] | [1284, 2930, 11741] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2438_5to1 | joint | 1573 | [731, 3866, 12294] | [731, 3866, 12294] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2461_5to3 | delta | 410 | [11, 3709, 876] | [11, 3709, 876] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2461_5to3 | joint | 410 | [11, 2673, 876] | [11, 2673, 876] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2590_3to5 | delta | 671 | [124, 2169, 890] | [124, 2169, 890] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2590_3to5 | joint | 671 | [229, 2789, 785] | [229, 2789, 785] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2792_1to5 | delta | 4481 | [0, 13705, 5613] | [0, 13705, 5613] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2792_1to5 | joint | 4481 | [0, 13428, 5613] | [0, 13428, 5613] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2828_1to5 | delta | 1119 | [1107, 3055, 42793] | [1107, 3055, 42793] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_2828_1to5 | joint | 1119 | [1107, 2678, 42793] | [1107, 2678, 42793] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_368_1to2 | delta | 6106 | [714, 7057, 69863] | [714, 7057, 69863] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_368_1to2 | joint | 6106 | [714, 7057, 69863] | [714, 7057, 69863] | None | Top8_truncation_loses_pool_coverage |
| confirm | train_37_4to3 | delta | 0 | [0, 9451, 0] | [0, 9451, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_37_4to3 | joint | 0 | [0, 14404, 0] | [0, 14404, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_378_1to4 | delta | 0 | [0, 2456, 0] | [0, 2456, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_378_1to4 | joint | 0 | [0, 2932, 0] | [0, 2932, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_427_3to1 | delta | 29 | [29, 1364, 335] | [29, 1364, 335] | None |  |
| confirm | train_427_3to1 | joint | 29 | [29, 1286, 335] | [29, 1286, 335] | None |  |
| confirm | train_615_3to4 | delta | 0 | [0, 732, 0] | [0, 732, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_615_3to4 | joint | 0 | [0, 587, 0] | [0, 587, 0] | None | similarity_gate_retains_empty_target_FP |
| confirm | train_968_5to3 | delta | 0 | [0, 1142, 62] | [0, 1142, 62] | None | original_geometry_pool_zero_GT_coverage |
| confirm | train_968_5to3 | joint | 0 | [0, 1220, 62] | [0, 1220, 62] | None | original_geometry_pool_zero_GT_coverage |
