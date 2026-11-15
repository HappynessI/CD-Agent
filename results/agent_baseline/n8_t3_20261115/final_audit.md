# N8 T3 final counts and failure decomposition

All stage outputs retained; count-based failure decomposition is offline diagnosis, not a GT-based inference rule. Request slots differ from HTTP attempts; returned-usage failures are charged once.

| Split | Tasks | Original candidates | Clean | Removed | Modified retained | Inst routes | Dense routes | Rebuilt | New masks | Triggered | Request slots | New-mask slots | Unknown answers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tune | 30 | 2320 | 2282 | 38 | 16 | 19 | 11 | 1651 | 35 | 6 | 30 | 3 | 0 |
| confirm | 30 | 2575 | 2527 | 48 | 25 | 16 | 14 | 1466 | 218 | 11 | 33 | 17 | 0 |

Returned-usage known cost CNY 0.017536200000000005; known failed-response cost 0.0; HTTP attempts 63; missing-usage attempts 0. Not an account invoice.

| Split | Task | Remaining GT | Original TP/FP/FN | Pool TP/FP/FN | TopK TP/FP/FN | Gate TP/FP/FN | Verified TP/FP/FN | Cause |
|---|---|---:|---|---|---|---|---|---|
| tune | train_1065_2to5 | 0 | [0, 5427, 0] | [0, 46700, 0] | [0, 2560, 0] | [0, 2560, 0] | [0, 2560, 0] |  |
| tune | train_1136_5to1 | 3 | [0, 5445, 3] | [0, 49375, 3] | [0, 1303, 3] | [0, 0, 3] | [0, 0, 3] | rebuilt_pool_zero_GT_coverage |
| tune | train_1166_3to5 | 0 | [0, 10722, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_119_6to2 | 0 | [0, 40930, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_127_3to2 | 0 | [0, 2024, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_1322_2to4 | 0 | [0, 14871, 0] | [0, 45338, 0] | [0, 8192, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_1377_5to3 | 0 | [0, 4574, 0] | [0, 30376, 0] | [0, 1284, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_145_1to2 | 0 | [0, 2567, 0] | [0, 36178, 0] | [0, 1399, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_147_1to5 | 0 | [0, 5757, 0] | [0, 61447, 0] | [0, 3106, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_1538_1to3 | 5243 | [5121, 4863, 122] | [5121, 70903, 122] | [5121, 1912, 122] | [5121, 1912, 122] | [5121, 1912, 122] |  |
| tune | train_1659_1to2 | 7720 | [1927, 1755, 5793] | [0, 0, 7720] | [0, 0, 7720] | [0, 0, 7720] | [0, 0, 7720] | rebuilt_pool_zero_GT_coverage |
| tune | train_1745_5to2 | 12451 | [678, 1066, 11773] | [1474, 19277, 10977] | [0, 541, 12451] | [0, 541, 12451] | [0, 381, 12451] | topK_truncation_loses_covered_TP |
| tune | train_1754_5to1 | 6067 | [0, 771, 6067] | [311, 5350, 5756] | [0, 367, 6067] | [0, 0, 6067] | [0, 0, 6067] | topK_truncation_loses_covered_TP |
| tune | train_1798_2to6 | 0 | [0, 15025, 0] | [0, 140534, 0] | [0, 11023, 0] | [0, 11023, 0] | [0, 1687, 0] |  |
| tune | train_1808_2to1 | 7641 | [2100, 64283, 5541] | [0, 0, 7641] | [0, 0, 7641] | [0, 0, 7641] | [0, 0, 7641] | rebuilt_pool_zero_GT_coverage |
| tune | train_2571_4to5 | 6 | [0, 5094, 6] | [0, 33569, 6] | [0, 2382, 6] | [0, 0, 6] | [0, 0, 6] | rebuilt_pool_zero_GT_coverage |
| tune | train_2677_2to1 | 0 | [0, 23340, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_2822_1to2 | 0 | [0, 8461, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_2848_1to4 | 0 | [0, 24510, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_2890_5to1 | 492 | [0, 13985, 492] | [0, 26857, 492] | [0, 10788, 492] | [0, 0, 492] | [0, 0, 492] | rebuilt_pool_zero_GT_coverage |
| tune | train_2912_5to2 | 429 | [0, 13845, 429] | [402, 151255, 27] | [0, 5291, 429] | [0, 0, 429] | [0, 0, 429] | topK_truncation_loses_covered_TP |
| tune | train_2928_3to1 | 0 | [0, 10138, 0] | [0, 106646, 0] | [0, 1744, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_385_5to6 | 0 | [0, 2942, 0] | [0, 74676, 0] | [0, 1643, 0] | [0, 1643, 0] | [0, 427, 0] |  |
| tune | train_443_3to2 | 0 | [0, 2855, 0] | [0, 45076, 0] | [0, 1269, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_555_4to2 | 0 | [0, 5259, 0] | [0, 34135, 0] | [0, 2716, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_7_1to2 | 262 | [0, 10131, 262] | [0, 0, 262] | [0, 0, 262] | [0, 0, 262] | [0, 0, 262] | rebuilt_pool_zero_GT_coverage |
| tune | train_786_3to5 | 0 | [0, 1605, 0] | [0, 27433, 0] | [0, 533, 0] | [0, 533, 0] | [0, 167, 0] |  |
| tune | train_908_4to1 | 0 | [0, 1307, 0] | [0, 2815, 0] | [0, 1624, 0] | [0, 0, 0] | [0, 0, 0] |  |
| tune | train_915_3to4 | 1 | [0, 2889, 1] | [0, 2532, 1] | [0, 1589, 1] | [0, 0, 1] | [0, 0, 1] | rebuilt_pool_zero_GT_coverage |
| tune | train_987_3to5 | 0 | [0, 9674, 0] | [0, 22910, 0] | [0, 2261, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_1129_4to5 | 0 | [0, 8423, 0] | [0, 42337, 0] | [0, 4686, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_1177_2to3 | 1686 | [0, 19815, 1686] | [235, 155985, 1451] | [0, 3611, 1686] | [0, 0, 1686] | [0, 0, 1686] | topK_truncation_loses_covered_TP |
| confirm | train_1344_2to5 | 16056 | [176, 2860, 15880] | [337, 45458, 15719] | [0, 1781, 16056] | [0, 0, 16056] | [0, 0, 16056] | topK_truncation_loses_covered_TP |
| confirm | train_1370_3to1 | 791 | [241, 8319, 550] | [499, 105868, 292] | [241, 2416, 550] | [0, 0, 791] | [0, 0, 791] | topK_truncation_loses_covered_TP, relative_gate_loses_TP |
| confirm | train_1496_2to1 | 4 | [4, 103972, 0] | [0, 41461, 4] | [0, 29049, 4] | [0, 29049, 4] | [0, 29049, 4] | rebuilt_pool_zero_GT_coverage |
| confirm | train_1527_5to2 | 0 | [0, 11422, 0] | [0, 94538, 0] | [0, 4905, 0] | [0, 4905, 0] | [0, 0, 0] |  |
| confirm | train_1535_4to1 | 14 | [14, 31049, 0] | [0, 0, 14] | [0, 0, 14] | [0, 0, 14] | [0, 0, 14] | rebuilt_pool_zero_GT_coverage |
| confirm | train_156_2to3 | 0 | [0, 41300, 0] | [0, 97, 0] | [0, 3, 0] | [0, 3, 0] | [0, 3, 0] |  |
| confirm | train_1884_4to3 | 0 | [0, 7023, 0] | [0, 31539, 0] | [0, 23794, 0] | [0, 23794, 0] | [0, 23778, 0] |  |
| confirm | train_2043_2to3 | 386 | [337, 1181, 49] | [0, 671, 386] | [0, 36, 386] | [0, 0, 386] | [0, 0, 386] | rebuilt_pool_zero_GT_coverage |
| confirm | train_206_3to4 | 0 | [0, 3030, 0] | [0, 2928, 0] | [0, 10, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_2075_4to2 | 0 | [0, 10748, 0] | [0, 31598, 0] | [0, 15, 0] | [0, 15, 0] | [0, 2, 0] |  |
| confirm | train_2130_2to4 | 0 | [0, 5682, 0] | [0, 2429, 0] | [0, 21, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_2192_1to5 | 25417 | [17170, 4515, 8247] | [22403, 68210, 3014] | [12147, 1048, 13270] | [12147, 1048, 13270] | [12147, 1048, 13270] | topK_truncation_loses_covered_TP |
| confirm | train_2227_3to2 | 816 | [92, 2511, 724] | [0, 0, 816] | [0, 0, 816] | [0, 0, 816] | [0, 0, 816] | rebuilt_pool_zero_GT_coverage |
| confirm | train_2255_2to5 | 3859 | [3031, 11318, 828] | [2672, 79852, 1187] | [2672, 13762, 1187] | [0, 0, 3859] | [0, 0, 3859] | relative_gate_loses_TP |
| confirm | train_23_2to1 | 3235 | [1973, 3528, 1262] | [0, 0, 3235] | [0, 0, 3235] | [0, 0, 3235] | [0, 0, 3235] | rebuilt_pool_zero_GT_coverage |
| confirm | train_2383_2to5 | 14568 | [11776, 10185, 2792] | [14230, 91204, 338] | [5447, 2483, 9121] | [0, 0, 14568] | [0, 0, 14568] | topK_truncation_loses_covered_TP, relative_gate_loses_TP |
| confirm | train_2437_1to3 | 0 | [0, 33664, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_2438_5to1 | 13025 | [1284, 2930, 11741] | [1573, 42409, 11452] | [350, 1235, 12675] | [350, 1235, 12675] | [350, 1012, 12675] | topK_truncation_loses_covered_TP |
| confirm | train_2461_5to3 | 887 | [11, 3709, 876] | [410, 47240, 477] | [11, 1367, 876] | [11, 1367, 876] | [0, 804, 887] | topK_truncation_loses_covered_TP, evidence_verifier_loses_TP |
| confirm | train_2590_3to5 | 1014 | [124, 2169, 890] | [671, 46347, 343] | [106, 2121, 908] | [0, 0, 1014] | [0, 0, 1014] | topK_truncation_loses_covered_TP, relative_gate_loses_TP |
| confirm | train_2792_1to5 | 5613 | [0, 13705, 5613] | [4481, 65017, 1132] | [0, 4608, 5613] | [0, 4608, 5613] | [0, 1190, 5613] | topK_truncation_loses_covered_TP |
| confirm | train_2828_1to5 | 43900 | [1107, 3055, 42793] | [1119, 6592, 42781] | [138, 1460, 43762] | [0, 0, 43900] | [0, 0, 43900] | topK_truncation_loses_covered_TP, relative_gate_loses_TP |
| confirm | train_368_1to2 | 70577 | [714, 7057, 69863] | [5969, 82006, 64608] | [577, 1422, 70000] | [577, 1422, 70000] | [577, 1422, 70000] | topK_truncation_loses_covered_TP |
| confirm | train_37_4to3 | 0 | [0, 9451, 0] | [0, 96397, 0] | [0, 1836, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_378_1to4 | 0 | [0, 2456, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_427_3to1 | 364 | [29, 1364, 335] | [0, 0, 364] | [0, 0, 364] | [0, 0, 364] | [0, 0, 364] | rebuilt_pool_zero_GT_coverage |
| confirm | train_615_3to4 | 0 | [0, 732, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] | [0, 0, 0] |  |
| confirm | train_968_5to3 | 62 | [0, 1142, 62] | [0, 12778, 62] | [0, 204, 62] | [0, 204, 62] | [0, 204, 62] | rebuilt_pool_zero_GT_coverage |
