# R1 direct4 exemplar-overlap audit

Original fixed R1 Top8 retained; live match_exemplar and eligible do not remove exemplar overlap. Scoring alone excludes exemplar pixels.
Candidate-summed counts double count overlaps; complete-task unions are pooled separately. Accepted overlap FP association does not prove self-matching caused decisions. Pending answers stay pending.

| GT | Group | N | Answers complete | Candidate category | Candidates | Yes/no/unparsed/pending | Accepted candidate-sum TP/FP/FN |
|---|---|---:|---:|---|---:|---|---|
| all | 全部 | 30 | 30 | no_overlap | 180 | {'yes': 68, 'no': 112, 'unparsed': 0, 'pending': 0} | [6701, 80630, 95761] |
| all | 全部 | 30 | 30 | overlap_nonempty | 48 | {'yes': 44, 'no': 4, 'unparsed': 0, 'pending': 0} | [210, 80499, 58946] |
| all | 全部 | 30 | 30 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| all | 目标类 | 15 | 15 | no_overlap | 89 | {'yes': 41, 'no': 48, 'unparsed': 0, 'pending': 0} | [518, 60331, 44666] |
| all | 目标类 | 15 | 15 | overlap_nonempty | 31 | {'yes': 28, 'no': 3, 'unparsed': 0, 'pending': 0} | [0, 25684, 33309] |
| all | 目标类 | 15 | 15 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| all | 地表类 | 15 | 15 | no_overlap | 91 | {'yes': 27, 'no': 64, 'unparsed': 0, 'pending': 0} | [6183, 20299, 51095] |
| all | 地表类 | 15 | 15 | overlap_nonempty | 17 | {'yes': 16, 'no': 1, 'unparsed': 0, 'pending': 0} | [210, 54815, 25637] |
| all | 地表类 | 15 | 15 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| positive | 全部 | 11 | 11 | no_overlap | 66 | {'yes': 27, 'no': 39, 'unparsed': 0, 'pending': 0} | [6701, 28316, 95761] |
| positive | 全部 | 11 | 11 | overlap_nonempty | 16 | {'yes': 14, 'no': 2, 'unparsed': 0, 'pending': 0} | [210, 35796, 58946] |
| positive | 全部 | 11 | 11 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| positive | 目标类 | 6 | 6 | no_overlap | 36 | {'yes': 17, 'no': 19, 'unparsed': 0, 'pending': 0} | [518, 20293, 44666] |
| positive | 目标类 | 6 | 6 | overlap_nonempty | 12 | {'yes': 10, 'no': 2, 'unparsed': 0, 'pending': 0} | [0, 3686, 33309] |
| positive | 目标类 | 6 | 6 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| positive | 地表类 | 5 | 5 | no_overlap | 30 | {'yes': 10, 'no': 20, 'unparsed': 0, 'pending': 0} | [6183, 8023, 51095] |
| positive | 地表类 | 5 | 5 | overlap_nonempty | 4 | {'yes': 4, 'no': 0, 'unparsed': 0, 'pending': 0} | [210, 32110, 25637] |
| positive | 地表类 | 5 | 5 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| empty | 全部 | 19 | 19 | no_overlap | 114 | {'yes': 41, 'no': 73, 'unparsed': 0, 'pending': 0} | [0, 52314, 0] |
| empty | 全部 | 19 | 19 | overlap_nonempty | 32 | {'yes': 30, 'no': 2, 'unparsed': 0, 'pending': 0} | [0, 44703, 0] |
| empty | 全部 | 19 | 19 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| empty | 目标类 | 9 | 9 | no_overlap | 53 | {'yes': 24, 'no': 29, 'unparsed': 0, 'pending': 0} | [0, 40038, 0] |
| empty | 目标类 | 9 | 9 | overlap_nonempty | 19 | {'yes': 18, 'no': 1, 'unparsed': 0, 'pending': 0} | [0, 21998, 0] |
| empty | 目标类 | 9 | 9 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |
| empty | 地表类 | 10 | 10 | no_overlap | 61 | {'yes': 17, 'no': 44, 'unparsed': 0, 'pending': 0} | [0, 12276, 0] |
| empty | 地表类 | 10 | 10 | overlap_nonempty | 13 | {'yes': 12, 'no': 1, 'unparsed': 0, 'pending': 0} | [0, 22705, 0] |
| empty | 地表类 | 10 | 10 | empty_residual | 0 | {'yes': 0, 'no': 0, 'unparsed': 0, 'pending': 0} | [0, 0, 0] |

| Task | GT pixels | Known answers | Accepted overlap union TP/FP/FN | Full verified TP/FP/FN |
|---|---:|---:|---|---|
| train_1065_2to5 | 0 | 8/8 | [0, 128, 0] | [0, 4932, 0] |
| train_1136_5to1 | 3 | 8/8 | [0, 437, 3] | [0, 3192, 3] |
| train_1166_3to5 | 0 | 8/8 | [0, 1467, 0] | [0, 1467, 0] |
| train_119_6to2 | 0 | 8/8 | [0, 36, 0] | [0, 6189, 0] |
| train_127_3to2 | 0 | 8/8 | [0, 0, 0] | [0, 352, 0] |
| train_1322_2to4 | 0 | 8/8 | [0, 386, 0] | [0, 386, 0] |
| train_1377_5to3 | 0 | 8/8 | [0, 1589, 0] | [0, 3463, 0] |
| train_145_1to2 | 0 | 8/8 | [0, 304, 0] | [0, 2567, 0] |
| train_147_1to5 | 0 | 8/8 | [0, 2465, 0] | [0, 5259, 0] |
| train_1538_1to3 | 5243 | 8/8 | [0, 1440, 5243] | [3005, 4125, 2238] |
| train_1659_1to2 | 7720 | 8/8 | [0, 157, 7720] | [1827, 456, 5893] |
| train_1745_5to2 | 12451 | 8/8 | [0, 468, 12451] | [518, 589, 11933] |
| train_1754_5to1 | 6067 | 8/8 | [0, 84, 6067] | [0, 160, 6067] |
| train_1798_2to6 | 0 | 8/8 | [0, 11187, 0] | [0, 15025, 0] |
| train_1808_2to1 | 7641 | 8/8 | [210, 30388, 7431] | [210, 30388, 7431] |
| train_2571_4to5 | 6 | 8/8 | [0, 0, 6] | [0, 4685, 6] |
| train_2677_2to1 | 0 | 2/2 | [0, 12577, 0] | [0, 12577, 0] |
| train_2822_1to2 | 0 | 8/8 | [0, 3462, 0] | [0, 6830, 0] |
| train_2848_1to4 | 0 | 8/8 | [0, 51, 0] | [0, 51, 0] |
| train_2890_5to1 | 492 | 8/8 | [0, 1210, 492] | [0, 1210, 492] |
| train_2912_5to2 | 429 | 8/8 | [0, 1086, 429] | [0, 11114, 429] |
| train_2928_3to1 | 0 | 8/8 | [0, 3157, 0] | [0, 3157, 0] |
| train_385_5to6 | 0 | 8/8 | [0, 1577, 0] | [0, 2004, 0] |
| train_443_3to2 | 0 | 8/8 | [0, 57, 0] | [0, 732, 0] |
| train_555_4to2 | 0 | 8/8 | [0, 0, 0] | [0, 1873, 0] |
| train_7_1to2 | 262 | 8/8 | [0, 0, 262] | [0, 1029, 262] |
| train_786_3to5 | 0 | 8/8 | [0, 0, 0] | [0, 465, 0] |
| train_908_4to1 | 0 | 8/8 | [0, 0, 0] | [0, 975, 0] |
| train_915_3to4 | 1 | 2/2 | [0, 0, 1] | [0, 0, 1] |
| train_987_3to5 | 0 | 8/8 | [0, 176, 0] | [0, 1753, 0] |
