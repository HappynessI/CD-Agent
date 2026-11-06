score descending, stable proposal-index ties; greedy one-to-one IoU >= .5; duplicates are FP; AP sum precision at true hit / GT count; macro average per task; report with and without exemplar

| 范围 | AP | R@1 | R@5 | R@10 | R@20 | R@50 | R全部 |
|---|---:|---:|---:|---:|---:|---:|---:|
| including_exemplar | 26.41 | 21.98 | 26.34 | 29.14 | 32.14 | 35.75 | 40.65 |
| excluding_exemplar | 6.17 | 0.53 | 9.38 | 14.63 | 20.27 | 26.83 | 35.47 |

| 转换 | 任务数 |
|---|---:|
| building -> low vegetation | 22 |
| building -> non-vegetated ground | 30 |
| building -> playground | 2 |
| building -> tree | 15 |
| building -> water | 1 |
| low vegetation -> building | 32 |
| low vegetation -> non-vegetated ground | 35 |
| low vegetation -> playground | 1 |
| low vegetation -> tree | 16 |
| low vegetation -> water | 8 |
| non-vegetated ground -> building | 50 |
| non-vegetated ground -> low vegetation | 35 |
| non-vegetated ground -> playground | 2 |
| non-vegetated ground -> tree | 29 |
| non-vegetated ground -> water | 6 |
| playground -> low vegetation | 1 |
| playground -> non-vegetated ground | 1 |
| tree -> building | 13 |
| tree -> low vegetation | 13 |
| tree -> non-vegetated ground | 18 |
| tree -> water | 4 |
| water -> building | 3 |
| water -> low vegetation | 6 |
| water -> non-vegetated ground | 9 |
| water -> tree | 4 |
