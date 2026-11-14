# B2 fixed mask complementarity

Fixed B2 SAM positive-only full-union mask versus B2 SegEarth dense mask. Disjoint SAM-only/dense-only/intersection and fixed union are descriptive regions, not GT-selected deployment policies. GT only scores these fixed masks. FN for each region is relative to the full remaining GT; regional FN counts must not be added.
Original split retained; available-case tables are partial until30/30, never a reduced benchmark. Confirmation images were multiply exposed. No online Agent evaluation.
tune: complete 30/30; pending 0.

| GT | Group | Available N | Region | F1 | P | R | TP | FP | FN |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| all | 全部 | 30 | SAM_only | 10.26358759038955 | 18.55215360664245 | 7.094133697135061 | 2860 | 12556 | 37455 |
| all | 全部 | 30 | dense_only | 3.98458627128912 | 2.662726832607135 | 7.91268758526603 | 3190 | 116612 | 37125 |
| all | 全部 | 30 | intersection | 2.873656600357466 | 22.37888647866956 | 1.5354086568274836 | 619 | 2147 | 39696 |
| all | 全部 | 30 | union | 7.480692544545959 | 4.833169063079778 | 16.542229939228577 | 6669 | 131315 | 33646 |
| all | 目标类 | 15 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 4150 | 19448 |
| all | 目标类 | 15 | dense_only | 8.800901789488517 | 6.061488296261791 | 16.05820649938297 | 3123 | 48399 | 16325 |
| all | 目标类 | 15 | intersection | 0.0 | 0.0 | 0.0 | 0 | 2030 | 19448 |
| all | 目标类 | 15 | union | 8.095917044718082 | 5.4122907351564935 | 16.05820649938297 | 3123 | 54579 | 16325 |
| all | 地表类 | 15 | SAM_only | 17.801014533345782 | 25.38611752174685 | 13.705851344227728 | 2860 | 8406 | 18007 |
| all | 地表类 | 15 | dense_only | 0.15031352709569587 | 0.09812536613942589 | 0.321081132889251 | 67 | 68213 | 20800 |
| all | 地表类 | 15 | intersection | 5.730685552932463 | 84.10326086956522 | 2.9664062874394976 | 619 | 117 | 20248 |
| all | 地表类 | 15 | union | 7.011438570821264 | 4.416930320619815 | 16.993338764556476 | 3546 | 76736 | 17321 |
| positive | 全部 | 11 | SAM_only | 11.072396438250097 | 25.2093433230498 | 7.094133697135061 | 2860 | 8485 | 37455 |
| positive | 全部 | 11 | dense_only | 10.935518151588907 | 17.695678704165974 | 7.91268758526603 | 3190 | 14837 | 37125 |
| positive | 全部 | 11 | intersection | 3.015173287220829 | 83.1989247311828 | 1.5354086568274836 | 619 | 125 | 39696 |
| positive | 全部 | 11 | union | 18.93768369042041 | 22.144375083012353 | 16.542229939228577 | 6669 | 23447 | 33646 |
| positive | 目标类 | 6 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 79 | 19448 |
| positive | 目标类 | 6 | dense_only | 18.52203309412253 | 21.878940731399748 | 16.05820649938297 | 3123 | 11151 | 16325 |
| positive | 目标类 | 6 | intersection | 0.0 | 0.0 | 0.0 | 0 | 8 | 19448 |
| positive | 目标类 | 6 | union | 18.47437072968736 | 21.74639649049509 | 16.05820649938297 | 3123 | 11238 | 16325 |
| positive | 地表类 | 5 | SAM_only | 17.801014533345782 | 25.38611752174685 | 13.705851344227728 | 2860 | 8406 | 18007 |
| positive | 地表类 | 5 | dense_only | 0.5442729488220959 | 1.7852384758859579 | 0.321081132889251 | 67 | 3686 | 20800 |
| positive | 地表类 | 5 | intersection | 5.730685552932463 | 84.10326086956522 | 2.9664062874394976 | 619 | 117 | 20248 |
| positive | 地表类 | 5 | union | 19.365408770684287 | 22.5071405902888 | 16.993338764556476 | 3546 | 12209 | 17321 |
| empty | 全部 | 19 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 4071 | 0 |
| empty | 全部 | 19 | dense_only | 0.0 | 0.0 | 0.0 | 0 | 101775 | 0 |
| empty | 全部 | 19 | intersection | 0.0 | 0.0 | 0.0 | 0 | 2022 | 0 |
| empty | 全部 | 19 | union | 0.0 | 0.0 | 0.0 | 0 | 107868 | 0 |
| empty | 目标类 | 9 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 4071 | 0 |
| empty | 目标类 | 9 | dense_only | 0.0 | 0.0 | 0.0 | 0 | 37248 | 0 |
| empty | 目标类 | 9 | intersection | 0.0 | 0.0 | 0.0 | 0 | 2022 | 0 |
| empty | 目标类 | 9 | union | 0.0 | 0.0 | 0.0 | 0 | 43341 | 0 |
| empty | 地表类 | 10 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |
| empty | 地表类 | 10 | dense_only | 0.0 | 0.0 | 0.0 | 0 | 64527 | 0 |
| empty | 地表类 | 10 | intersection | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |
| empty | 地表类 | 10 | union | 0.0 | 0.0 | 0.0 | 0 | 64527 | 0 |

Per-case regions and counts remain under tasks/. The union is specified before scoring and is not chosen per case from GT. No claim of calibrated abstention or deployable combination follows from overlap counts. Pending cases are listed in summary.json.
