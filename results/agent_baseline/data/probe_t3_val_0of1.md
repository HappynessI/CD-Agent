score descending, stable proposal-index ties; greedy one-to-one IoU >= .5; duplicates are FP; AP sum precision at true hit / GT count; macro average per task; report with and without exemplar

| 范围 | AP | R@1 | R@5 | R@10 | R@20 | R@50 | R全部 |
|---|---:|---:|---:|---:|---:|---:|---:|
| including_exemplar | 28.30 | 24.79 | 28.09 | 30.83 | 34.04 | 38.32 | 43.27 |
| excluding_exemplar | 5.20 | 0.52 | 7.41 | 13.03 | 19.33 | 27.72 | 37.16 |

| 转换 | 任务数 |
|---|---:|
| building -> low vegetation | 31 |
| building -> non-vegetated ground | 88 |
| building -> tree | 23 |
| building -> water | 2 |
| low vegetation -> building | 70 |
| low vegetation -> non-vegetated ground | 77 |
| low vegetation -> playground | 2 |
| low vegetation -> tree | 36 |
| low vegetation -> water | 10 |
| non-vegetated ground -> building | 106 |
| non-vegetated ground -> low vegetation | 70 |
| non-vegetated ground -> playground | 4 |
| non-vegetated ground -> tree | 54 |
| non-vegetated ground -> water | 8 |
| playground -> building | 2 |
| playground -> low vegetation | 1 |
| playground -> non-vegetated ground | 2 |
| tree -> building | 28 |
| tree -> low vegetation | 17 |
| tree -> non-vegetated ground | 41 |
| tree -> water | 1 |
| water -> building | 7 |
| water -> low vegetation | 7 |
| water -> non-vegetated ground | 11 |
| water -> tree | 5 |
