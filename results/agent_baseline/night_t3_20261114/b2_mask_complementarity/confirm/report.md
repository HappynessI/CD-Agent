# B2 fixed mask complementarity

Fixed B2 SAM positive-only full-union mask versus B2 SegEarth dense mask. Disjoint SAM-only/dense-only/intersection and fixed union are descriptive regions, not GT-selected deployment policies. GT only scores these fixed masks. FN for each region is relative to the full remaining GT; regional FN counts must not be added.
Original split retained; available-case tables are partial until30/30, never a reduced benchmark. Confirmation images were multiply exposed. No online Agent evaluation.
confirm: complete 30/30; pending 0.

| GT | Group | Available N | Region | F1 | P | R | TP | FP | FN |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| all | 全部 | 30 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 931 | 202274 |
| all | 全部 | 30 | dense_only | 4.8732212429943536 | 5.149090268681688 | 4.6254090985494924 | 9356 | 172346 | 192918 |
| all | 全部 | 30 | intersection | 0.0 | 0.0 | 0.0 | 0 | 8377 | 202274 |
| all | 全部 | 30 | union | 4.757884887257045 | 4.898172870530339 | 4.6254090985494924 | 9356 | 181654 | 192918 |
| all | 目标类 | 12 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 890 | 124401 |
| all | 目标类 | 12 | dense_only | 8.858619203463121 | 10.781894901888446 | 7.517624456395045 | 9352 | 77386 | 115049 |
| all | 目标类 | 12 | intersection | 0.0 | 0.0 | 0.0 | 0 | 2579 | 124401 |
| all | 目标类 | 12 | union | 8.715425333631552 | 10.367266398394802 | 7.517624456395045 | 9352 | 80855 | 115049 |
| all | 地表类 | 18 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 41 | 77873 |
| all | 地表类 | 18 | dense_only | 0.004628638543830314 | 0.004212122488521966 | 0.0051365685154032845 | 4 | 94960 | 77869 |
| all | 地表类 | 18 | intersection | 0.0 | 0.0 | 0.0 | 0 | 5798 | 77873 |
| all | 地表类 | 18 | union | 0.004477378047415433 | 0.003968135868972154 | 0.0051365685154032845 | 4 | 100799 | 77869 |
| positive | 全部 | 19 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 0 | 202274 |
| positive | 全部 | 19 | dense_only | 5.861567762630313 | 7.999452795020435 | 4.6254090985494924 | 9356 | 107602 | 192918 |
| positive | 全部 | 19 | intersection | 0.0 | 0.0 | 0.0 | 0 | 5798 | 202274 |
| positive | 全部 | 19 | union | 5.757007045503492 | 7.621623382971097 | 4.6254090985494924 | 9356 | 113400 | 192918 |
| positive | 目标类 | 10 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 0 | 124401 |
| positive | 目标类 | 10 | dense_only | 10.483658518813302 | 17.3153119792631 | 7.517624456395045 | 9352 | 44658 | 115049 |
| positive | 目标类 | 10 | intersection | 0.0 | 0.0 | 0.0 | 0 | 0 | 124401 |
| positive | 目标类 | 10 | union | 10.483658518813302 | 17.3153119792631 | 7.517624456395045 | 9352 | 44658 | 115049 |
| positive | 地表类 | 9 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 0 | 77873 |
| positive | 地表类 | 9 | dense_only | 0.005680970877923037 | 0.006354451293130838 | 0.0051365685154032845 | 4 | 62944 | 77869 |
| positive | 地表类 | 9 | intersection | 0.0 | 0.0 | 0.0 | 0 | 5798 | 77873 |
| positive | 地表类 | 9 | union | 0.0054563187581418504 | 0.005818520350274925 | 0.0051365685154032845 | 4 | 68742 | 77869 |
| empty | 全部 | 11 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 931 | 0 |
| empty | 全部 | 11 | dense_only | 0.0 | 0.0 | 0.0 | 0 | 64744 | 0 |
| empty | 全部 | 11 | intersection | 0.0 | 0.0 | 0.0 | 0 | 2579 | 0 |
| empty | 全部 | 11 | union | 0.0 | 0.0 | 0.0 | 0 | 68254 | 0 |
| empty | 目标类 | 2 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 890 | 0 |
| empty | 目标类 | 2 | dense_only | 0.0 | 0.0 | 0.0 | 0 | 32728 | 0 |
| empty | 目标类 | 2 | intersection | 0.0 | 0.0 | 0.0 | 0 | 2579 | 0 |
| empty | 目标类 | 2 | union | 0.0 | 0.0 | 0.0 | 0 | 36197 | 0 |
| empty | 地表类 | 9 | SAM_only | 0.0 | 0.0 | 0.0 | 0 | 41 | 0 |
| empty | 地表类 | 9 | dense_only | 0.0 | 0.0 | 0.0 | 0 | 32016 | 0 |
| empty | 地表类 | 9 | intersection | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |
| empty | 地表类 | 9 | union | 0.0 | 0.0 | 0.0 | 0 | 32057 | 0 |

Per-case regions and counts remain under tasks/. The union is specified before scoring and is not chosen per case from GT. No claim of calibrated abstention or deployable combination follows from overlap counts. Pending cases are listed in summary.json.
