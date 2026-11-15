# N7 find_missed trigger replay

Frozen coverage threshold: 0.05. New API requests: 0.
Historical exposed development pool only; no independent generalization or online Agent claim.
Verifier token costs are usage-based replay estimates, not new spending; candidate generation time/cost is unavailable and excluded.
Candidate signals are post-generation diagnostics, not inputs to the call gate. C1 direction is confounded with source in this pool.

| endpoint | policy | split | group | F1/P/R | macro Δ | pooled Δ | calls | W/L/T | added TP/FP | verifier requests | replay CNY |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced_v0_k10 | never | tune | all | 87.0879/85.6178/88.6094 | 0.0000 | 0.0000 | 0/15 | 0/0/15 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | tune | second_appeared | 82.8151/93.7967/74.1353 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | tune | second_disappeared | 10.2962/6.8767/20.4799 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | confirm | all | 79.0126/78.8316/79.1944 | 0.0000 | 0.0000 | 0/15 | 0/0/15 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | confirm | levir_any | 86.2924/79.8707/93.8371 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | confirm | second_appeared | 46.2066/72.8880/33.8247 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | never | confirm | second_disappeared | 26.7119/55.3102/17.6078 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | always | tune | all | 76.6258/65.5302/92.2449 | 0.8865 | -10.4621 | 15/15 | 3/7/5 | 20658/191139 | 136 | 0.033328 |
| balanced_v0_k10 | always | tune | levir_any | 76.5671/64.5624/94.0557 | -9.3188 | -12.5050 | 5/5 | 0/5/0 | 1440/148863 | 50 | 0.012016 |
| balanced_v0_k10 | always | tune | second_appeared | 78.4269/71.5778/86.7254 | 1.4148 | -4.3882 | 5/5 | 2/2/1 | 17564/41202 | 47 | 0.011604 |
| balanced_v0_k10 | always | tune | second_disappeared | 35.4949/22.0271/91.3453 | 10.5633 | 25.1987 | 5/5 | 1/0/4 | 1654/1074 | 39 | 0.009708 |
| balanced_v0_k10 | always | confirm | all | 69.2173/58.0629/85.6767 | -2.5419 | -9.7953 | 15/15 | 4/7/4 | 37476/234814 | 149 | 0.036675 |
| balanced_v0_k10 | always | confirm | levir_any | 73.7729/60.3777/94.8063 | -12.6554 | -12.5195 | 5/5 | 1/4/0 | 4340/172685 | 50 | 0.011934 |
| balanced_v0_k10 | always | confirm | second_appeared | 57.7169/70.2260/48.9904 | 6.2638 | 11.5103 | 5/5 | 2/2/1 | 13775/7438 | 49 | 0.012265 |
| balanced_v0_k10 | always | confirm | second_disappeared | 41.7225/30.3828/66.5672 | -1.2342 | 15.0106 | 5/5 | 1/1/3 | 19361/54691 | 50 | 0.012476 |
| balanced_v0_k10 | coverage | tune | all | 87.6482/85.3836/90.0361 | 5.4441 | 0.5602 | 7/15 | 2/0/13 | 8107/3001 | 56 | 0.013923 |
| balanced_v0_k10 | coverage | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | coverage | tune | second_appeared | 85.1261/92.6107/78.7609 | 5.7690 | 2.3111 | 2/5 | 1/0/4 | 6453/1927 | 17 | 0.004215 |
| balanced_v0_k10 | coverage | tune | second_disappeared | 35.4949/22.0271/91.3453 | 10.5633 | 25.1987 | 5/5 | 1/0/4 | 1654/1074 | 39 | 0.009708 |
| balanced_v0_k10 | coverage | confirm | all | 74.2730/66.0084/84.9033 | -0.5289 | -4.7396 | 10/15 | 3/3/9 | 33005/129825 | 99 | 0.024634 |
| balanced_v0_k10 | coverage | confirm | levir_any | 80.4985/70.4491/93.8918 | -7.5599 | -5.7940 | 1/5 | 0/1/4 | 245/70455 | 10 | 0.002384 |
| balanced_v0_k10 | coverage | confirm | second_appeared | 58.4169/73.2571/48.5765 | 7.2074 | 12.2103 | 4/5 | 2/1/2 | 13399/4679 | 39 | 0.009773 |
| balanced_v0_k10 | coverage | confirm | second_disappeared | 41.7225/30.3828/66.5672 | -1.2342 | 15.0106 | 5/5 | 1/1/3 | 19361/54691 | 50 | 0.012476 |
| balanced_v0_k10 | direction | tune | all | 85.8465/80.4711/91.9915 | 3.9927 | -1.2414 | 10/15 | 3/2/10 | 19218/42276 | 86 | 0.021312 |
| balanced_v0_k10 | direction | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | direction | tune | second_appeared | 78.4269/71.5778/86.7254 | 1.4148 | -4.3882 | 5/5 | 2/2/1 | 17564/41202 | 47 | 0.011604 |
| balanced_v0_k10 | direction | tune | second_disappeared | 35.4949/22.0271/91.3453 | 10.5633 | 25.1987 | 5/5 | 1/0/4 | 1654/1074 | 39 | 0.009708 |
| balanced_v0_k10 | direction | confirm | all | 78.2950/72.6246/84.9260 | 1.6765 | -0.7175 | 10/15 | 3/3/9 | 33136/62129 | 99 | 0.024741 |
| balanced_v0_k10 | direction | confirm | levir_any | 86.2924/79.8707/93.8371 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| balanced_v0_k10 | direction | confirm | second_appeared | 57.7169/70.2260/48.9904 | 6.2638 | 11.5103 | 5/5 | 2/2/1 | 13775/7438 | 49 | 0.012265 |
| balanced_v0_k10 | direction | confirm | second_disappeared | 41.7225/30.3828/66.5672 | -1.2342 | 15.0106 | 5/5 | 1/1/3 | 19361/54691 | 50 | 0.012476 |
| gated_evidence_v0_k10 | never | tune | all | 87.0879/85.6178/88.6094 | 0.0000 | 0.0000 | 0/15 | 0/0/15 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | tune | second_appeared | 82.8151/93.7967/74.1353 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | tune | second_disappeared | 10.2962/6.8767/20.4799 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | confirm | all | 79.0126/78.8316/79.1944 | 0.0000 | 0.0000 | 0/15 | 0/0/15 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | confirm | levir_any | 86.2924/79.8707/93.8371 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | confirm | second_appeared | 46.2066/72.8880/33.8247 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | never | confirm | second_disappeared | 26.7119/55.3102/17.6078 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | always | tune | all | 85.5078/79.9373/91.9128 | 4.2843 | -1.5801 | 15/15 | 3/7/5 | 18771/46502 | 134 | 0.032888 |
| gated_evidence_v0_k10 | always | tune | levir_any | 88.5462/83.8893/93.7506 | -0.5320 | -0.5258 | 5/5 | 0/5/0 | 139/5503 | 50 | 0.012048 |
| gated_evidence_v0_k10 | always | tune | second_appeared | 78.6077/72.2450/86.1993 | 2.4249 | -4.2074 | 5/5 | 2/2/1 | 16830/39359 | 47 | 0.011642 |
| gated_evidence_v0_k10 | always | tune | second_disappeared | 35.8293/21.9378/97.6864 | 10.9601 | 25.5332 | 5/5 | 1/0/4 | 1802/1640 | 37 | 0.009197 |
| gated_evidence_v0_k10 | always | confirm | all | 75.6594/71.2673/80.6283 | -2.9905 | -3.3532 | 15/15 | 3/9/3 | 8290/64987 | 149 | 0.036724 |
| gated_evidence_v0_k10 | always | confirm | levir_any | 85.1730/77.9513/93.8695 | -1.2032 | -1.1194 | 5/5 | 0/5/0 | 145/12994 | 50 | 0.011938 |
| gated_evidence_v0_k10 | always | confirm | second_appeared | 52.3725/67.4939/42.7865 | 2.8743 | 6.1659 | 5/5 | 2/2/1 | 8140/7289 | 49 | 0.012280 |
| gated_evidence_v0_k10 | always | confirm | second_disappeared | 14.3903/12.1610/17.6204 | -10.6427 | -12.3216 | 5/5 | 1/2/2 | 5/44704 | 50 | 0.012506 |
| gated_evidence_v0_k10 | coverage | tune | all | 87.7954/85.5227/90.1921 | 5.9883 | 0.7074 | 7/15 | 2/0/13 | 8993/2176 | 54 | 0.013388 |
| gated_evidence_v0_k10 | coverage | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | coverage | tune | second_appeared | 85.9152/93.7487/79.2899 | 7.0049 | 3.1002 | 2/5 | 1/0/4 | 7191/536 | 17 | 0.004191 |
| gated_evidence_v0_k10 | coverage | tune | second_disappeared | 35.8293/21.9378/97.6864 | 10.9601 | 25.5332 | 5/5 | 1/0/4 | 1802/1640 | 37 | 0.009197 |
| gated_evidence_v0_k10 | coverage | confirm | all | 76.5150/72.8777/80.5344 | -2.3299 | -2.4976 | 10/15 | 3/4/8 | 7747/50332 | 99 | 0.024669 |
| gated_evidence_v0_k10 | coverage | confirm | levir_any | 86.1676/79.6525/93.8433 | -0.2668 | -0.1249 | 1/5 | 0/1/4 | 28/1448 | 10 | 0.002384 |
| gated_evidence_v0_k10 | coverage | confirm | second_appeared | 53.0623/71.1204/42.3175 | 3.9197 | 6.8557 | 4/5 | 2/1/2 | 7714/4180 | 39 | 0.009778 |
| gated_evidence_v0_k10 | coverage | confirm | second_disappeared | 14.3903/12.1610/17.6204 | -10.6427 | -12.3216 | 5/5 | 1/2/2 | 5/44704 | 50 | 0.012506 |
| gated_evidence_v0_k10 | direction | tune | all | 85.8817/80.6121/91.8884 | 4.4617 | -1.2062 | 10/15 | 3/2/10 | 18632/40999 | 84 | 0.020840 |
| gated_evidence_v0_k10 | direction | tune | levir_any | 89.0720/84.8649/93.7180 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | direction | tune | second_appeared | 78.6077/72.2450/86.1993 | 2.4249 | -4.2074 | 5/5 | 2/2/1 | 16830/39359 | 47 | 0.011642 |
| gated_evidence_v0_k10 | direction | tune | second_disappeared | 35.8293/21.9378/97.6864 | 10.9601 | 25.5332 | 5/5 | 1/0/4 | 1802/1640 | 37 | 0.009197 |
| gated_evidence_v0_k10 | direction | confirm | all | 76.4511/72.7057/80.6033 | -2.5895 | -2.5615 | 10/15 | 3/4/8 | 8145/51993 | 99 | 0.024786 |
| gated_evidence_v0_k10 | direction | confirm | levir_any | 86.2924/79.8707/93.8371 | 0.0000 | 0.0000 | 0/5 | 0/0/5 | 0/0 | 0 | 0.000000 |
| gated_evidence_v0_k10 | direction | confirm | second_appeared | 52.3725/67.4939/42.7865 | 2.8743 | 6.1659 | 5/5 | 2/2/1 | 8140/7289 | 49 | 0.012280 |
| gated_evidence_v0_k10 | direction | confirm | second_disappeared | 14.3903/12.1610/17.6204 | -10.6427 | -12.3216 | 5/5 | 1/2/2 | 5/44704 | 50 | 0.012506 |

Per-image failures and skipped benefits: paired_cases.json. Confidence intervals and leave-one-out stability: summary.json.
Elapsed replay seconds: 29.528. Slurm job: 187166.
