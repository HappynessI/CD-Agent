# N7 concentration follow-up

One predeclared rule: call for empty current answer or largest component >= half of current-mask area.
Same cached endpoints; no new API. Exposed-development internal confirmation only.

| endpoint | split | group | F1/P/R | macro Δ never | pooled Δ never | macro Δ always | pooled Δ always | macro Δ coverage | pooled Δ coverage | calls | W/L/T | added TP/FP | verifier calls | replay CNY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced_v0_k10 | tune | all | 87.7716/84.5987/91.1918 | 5.9862 | 0.6837 | 5.0998 | 11.1457 | 0.5421 | 0.1234 | 8/15 | 3/0/12 | 14674/9756 | 66 | 0.016403 |
| balanced_v0_k10 | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 9.3188 | 12.5050 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | tune | second_appeared | 85.7868/88.2379/83.4682 | 7.3954 | 2.9717 | 5.9805 | 7.3599 | 1.6263 | 0.6607 | 3/5 | 2/0/3 | 13020/8682 | 27 | 0.006695 |
| balanced_v0_k10 | tune | second_disappeared | 35.4949/22.0271/91.3453 | 10.5633 | 25.1987 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 | 1/0/4 | 1654/1074 | 39 | 0.009708 |
| balanced_v0_k10 | confirm | all | 77.4587/72.3273/83.3737 | -0.2175 | -1.5539 | 2.3244 | 8.2414 | 0.3114 | 3.1857 | 9/15 | 2/3/10 | 24162/61474 | 89 | 0.022279 |
| balanced_v0_k10 | confirm | levir_any | 86.2924/79.8707/93.8371 | 0.0000 | 0.0000 | 12.6554 | 12.5195 | 7.5599 | 5.7940 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | confirm | second_appeared | 49.1461/66.1096/39.1104 | 0.5816 | 2.9395 | -5.6822 | -8.5708 | -6.6258 | -9.2709 | 4/5 | 1/2/2 | 4801/6783 | 39 | 0.009802 |
| balanced_v0_k10 | confirm | second_disappeared | 41.7225/30.3828/66.5672 | -1.2342 | 15.0106 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 | 1/1/3 | 19361/54691 | 50 | 0.012476 |
| gated_evidence_v0_k10 | tune | all | 87.9091/84.7327/91.3328 | 6.5160 | 0.8211 | 2.2317 | 2.4013 | 0.5276 | 0.1137 | 8/15 | 3/0/12 | 15475/8931 | 64 | 0.015891 |
| gated_evidence_v0_k10 | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0.5320 | 0.5258 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | tune | second_appeared | 86.5030/89.2317/83.9363 | 8.5878 | 3.6880 | 6.1630 | 7.8954 | 1.5829 | 0.5878 | 3/5 | 2/0/3 | 13673/7291 | 27 | 0.006693 |
| gated_evidence_v0_k10 | tune | second_disappeared | 35.8293/21.9378/97.6864 | 10.9601 | 25.5332 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 | 1/0/4 | 1802/1640 | 37 | 0.009197 |
| gated_evidence_v0_k10 | confirm | all | 76.1543/72.5943/80.0816 | -3.3123 | -2.8582 | -0.3218 | 0.4949 | -0.9824 | -0.3607 | 9/15 | 2/4/9 | 5129/51838 | 89 | 0.022346 |
| gated_evidence_v0_k10 | confirm | levir_any | 86.2924/79.8707/93.8371 | 0.0000 | 0.0000 | 1.2032 | 1.1194 | 0.2668 | 0.1249 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | confirm | second_appeared | 49.3628/65.8843/39.4660 | 0.7058 | 3.1562 | -2.1685 | -3.0097 | -3.2139 | -3.6995 | 4/5 | 1/2/2 | 5124/7134 | 39 | 0.009840 |
| gated_evidence_v0_k10 | confirm | second_disappeared | 14.3903/12.1610/17.6204 | -10.6427 | -12.3216 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 | 1/2/2 | 5/44704 | 50 | 0.012506 |

All called-harm and skipped-benefit records are retained in paired_cases.json; empty-GT FP counts are in summary.json.
