# Positive endpoint mechanism pairing

Fixed predicted classes; strict versus positive-only candidate support. Full confirm30 denominators; no GT class prompts. No verification. Cache generation is local SAM3 only.
Exploratory confirmation on multiply exposed images; B2 preselected from positive-stratum label agreement. Tune TP gain was concentrated in train1808, not established stability. No online Agent evaluation.

| Variant | GT | Group | N | Method | F1 | P | R | TP | FP | FN | Covered/truth |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| B0 | all | 全部 | 30 | positive_only_all@0 | 11.193093546034723 | 66.77114293434207 | 6.10854583386891 | 12356 | 6149 | 189918 | 12356/202274 |
| B0 | all | 全部 | 30 | positive_only_delta@8 | 8.370839980323376 | 59.72577576592534 | 4.500825612782661 | 9104 | 6139 | 193170 | 9104/202274 |
| B0 | all | 全部 | 30 | strict_all@0 | 11.193093546034723 | 66.77114293434207 | 6.10854583386891 | 12356 | 6149 | 189918 | 12356/202274 |
| B0 | all | 全部 | 30 | strict_delta@8 | 8.370839980323376 | 59.72577576592534 | 4.500825612782661 | 9104 | 6139 | 193170 | 9104/202274 |
| B0 | all | 目标类 | 12 | positive_only_all@0 | 18.04058986713389 | 98.22720407027586 | 9.932396041832462 | 12356 | 223 | 112045 | 12356/124401 |
| B0 | all | 目标类 | 12 | positive_only_delta@8 | 13.616715774989157 | 97.71385639154234 | 7.318269145746417 | 9104 | 213 | 115297 | 9104/124401 |
| B0 | all | 目标类 | 12 | strict_all@0 | 18.04058986713389 | 98.22720407027586 | 9.932396041832462 | 12356 | 223 | 112045 | 12356/124401 |
| B0 | all | 目标类 | 12 | strict_delta@8 | 13.616715774989157 | 97.71385639154234 | 7.318269145746417 | 9104 | 213 | 115297 | 9104/124401 |
| B0 | all | 地表类 | 18 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5926 | 77873 | 0/77873 |
| B0 | all | 地表类 | 18 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5926 | 77873 | 0/77873 |
| B0 | all | 地表类 | 18 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5926 | 77873 | 0/77873 |
| B0 | all | 地表类 | 18 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5926 | 77873 | 0/77873 |
| B0 | positive | 全部 | 19 | positive_only_all@0 | 11.19862237730548 | 67.16677538595347 | 6.10854583386891 | 12356 | 6040 | 189918 | 12356/202274 |
| B0 | positive | 全部 | 19 | positive_only_delta@8 | 8.375036797173976 | 60.15594026694859 | 4.500825612782661 | 9104 | 6030 | 193170 | 9104/202274 |
| B0 | positive | 全部 | 19 | strict_all@0 | 11.19862237730548 | 67.16677538595347 | 6.10854583386891 | 12356 | 6040 | 189918 | 12356/202274 |
| B0 | positive | 全部 | 19 | strict_delta@8 | 8.375036797173976 | 60.15594026694859 | 4.500825612782661 | 9104 | 6030 | 193170 | 9104/202274 |
| B0 | positive | 目标类 | 10 | positive_only_all@0 | 18.040721570459706 | 98.23501351566227 | 9.932396041832462 | 12356 | 222 | 112045 | 12356/124401 |
| B0 | positive | 目标类 | 10 | positive_only_delta@8 | 13.616817607334895 | 97.72434521253757 | 7.318269145746417 | 9104 | 212 | 115297 | 9104/124401 |
| B0 | positive | 目标类 | 10 | strict_all@0 | 18.040721570459706 | 98.23501351566227 | 9.932396041832462 | 12356 | 222 | 112045 | 12356/124401 |
| B0 | positive | 目标类 | 10 | strict_delta@8 | 13.616817607334895 | 97.72434521253757 | 7.318269145746417 | 9104 | 212 | 115297 | 9104/124401 |
| B0 | positive | 地表类 | 9 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5818 | 77873 | 0/77873 |
| B0 | positive | 地表类 | 9 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5818 | 77873 | 0/77873 |
| B0 | positive | 地表类 | 9 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5818 | 77873 | 0/77873 |
| B0 | positive | 地表类 | 9 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5818 | 77873 | 0/77873 |
| B0 | empty | 全部 | 11 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 109 | 0 | 0/0 |
| B0 | empty | 全部 | 11 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 109 | 0 | 0/0 |
| B0 | empty | 全部 | 11 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 109 | 0 | 0/0 |
| B0 | empty | 全部 | 11 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 109 | 0 | 0/0 |
| B0 | empty | 目标类 | 2 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 0 | 0/0 |
| B0 | empty | 目标类 | 2 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 1 | 0 | 0/0 |
| B0 | empty | 目标类 | 2 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 0 | 0/0 |
| B0 | empty | 目标类 | 2 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 1 | 0 | 0/0 |
| B0 | empty | 地表类 | 9 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 108 | 0 | 0/0 |
| B0 | empty | 地表类 | 9 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 108 | 0 | 0/0 |
| B0 | empty | 地表类 | 9 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 108 | 0 | 0/0 |
| B0 | empty | 地表类 | 9 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 108 | 0 | 0/0 |
| B2 | all | 全部 | 30 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 9308 | 202274 | 0/202274 |
| B2 | all | 全部 | 30 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 9308 | 202274 | 0/202274 |
| B2 | all | 全部 | 30 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 9308 | 202274 | 0/202274 |
| B2 | all | 全部 | 30 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 9308 | 202274 | 0/202274 |
| B2 | all | 目标类 | 12 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 124401 | 0/124401 |
| B2 | all | 目标类 | 12 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 124401 | 0/124401 |
| B2 | all | 目标类 | 12 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 124401 | 0/124401 |
| B2 | all | 目标类 | 12 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 124401 | 0/124401 |
| B2 | all | 地表类 | 18 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5839 | 77873 | 0/77873 |
| B2 | all | 地表类 | 18 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5839 | 77873 | 0/77873 |
| B2 | all | 地表类 | 18 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5839 | 77873 | 0/77873 |
| B2 | all | 地表类 | 18 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5839 | 77873 | 0/77873 |
| B2 | positive | 全部 | 19 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 202274 | 0/202274 |
| B2 | positive | 全部 | 19 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 202274 | 0/202274 |
| B2 | positive | 全部 | 19 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 202274 | 0/202274 |
| B2 | positive | 全部 | 19 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 202274 | 0/202274 |
| B2 | positive | 目标类 | 10 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 124401 | 0/124401 |
| B2 | positive | 目标类 | 10 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 0 | 124401 | 0/124401 |
| B2 | positive | 目标类 | 10 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 124401 | 0/124401 |
| B2 | positive | 目标类 | 10 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 0 | 124401 | 0/124401 |
| B2 | positive | 地表类 | 9 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 77873 | 0/77873 |
| B2 | positive | 地表类 | 9 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 77873 | 0/77873 |
| B2 | positive | 地表类 | 9 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 77873 | 0/77873 |
| B2 | positive | 地表类 | 9 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 5798 | 77873 | 0/77873 |
| B2 | empty | 全部 | 11 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 3510 | 0 | 0/0 |
| B2 | empty | 全部 | 11 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 3510 | 0 | 0/0 |
| B2 | empty | 全部 | 11 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 3510 | 0 | 0/0 |
| B2 | empty | 全部 | 11 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 3510 | 0 | 0/0 |
| B2 | empty | 目标类 | 2 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 0 | 0/0 |
| B2 | empty | 目标类 | 2 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 0 | 0/0 |
| B2 | empty | 目标类 | 2 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 0 | 0/0 |
| B2 | empty | 目标类 | 2 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 3469 | 0 | 0/0 |
| B2 | empty | 地表类 | 9 | positive_only_all@0 | 0.0 | 0.0 | 0.0 | 0 | 41 | 0 | 0/0 |
| B2 | empty | 地表类 | 9 | positive_only_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 41 | 0 | 0/0 |
| B2 | empty | 地表类 | 9 | strict_all@0 | 0.0 | 0.0 | 0.0 | 0 | 41 | 0 | 0/0 |
| B2 | empty | 地表类 | 9 | strict_delta@8 | 0.0 | 0.0 | 0.0 | 0 | 41 | 0 | 0/0 |

Incomplete stages are not full-cohort evidence. Positive-only can retain unchanged or ambiguous support and raise FP. All-candidate union isolates support exclusions; Top8 additionally reflects fixed delta ranking on changed supports. B2 has cross-date input information and cannot be called a pure spotlight intervention.
