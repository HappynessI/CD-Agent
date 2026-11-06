| Group | Row | F1^c | P | R | Direction violation | ΔF1 given | Δ fixed [95% CI] | Δ rule [95% CI] | Rebuilt | Error / fallback |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| T1.1xC1_appeared | Agent mean (3) | 72.30 | 83.03 | 64.04 | 0.0033 | — | 15.27 [9.99, 21.18] | 8.57 [3.04, 15.35] | 63 | 0 / 12 |
| T1.1xC1_appeared | fixed_direction | 57.04 | 56.69 | 57.38 | 0.0037 | — | — | — | — | — |
| T1.1xC1_appeared | fixed_any | 49.03 | 42.59 | 57.78 | 0.0795 | — | — | — | — | — |
| T1.1xC1_appeared | rule_direction | 63.74 | 83.49 | 51.54 | 0.0028 | — | — | — | 20 | — |
| T1.1xC1_disappeared | Agent mean (3) | 50.29 | 68.15 | 39.99 | 0.0070 | — | 17.45 [10.87, 24.03] | 12.33 [4.01, 21.08] | 110 | 0 / 15 |
| T1.1xC1_disappeared | fixed_direction | 32.84 | 30.12 | 36.09 | 0.0123 | — | — | — | — | — |
| T1.1xC1_disappeared | fixed_any | 13.11 | 7.95 | 37.33 | 0.4259 | — | — | — | — | — |
| T1.1xC1_disappeared | rule_direction | 37.96 | 56.84 | 28.50 | 0.0218 | — | — | — | 36 | — |
| T1.1xC0_any | Agent mean (3) | 64.24 | 81.01 | 53.27 | 0.0000 | — | 12.41 [8.32, 16.52] | — | 305 | 0 / 25 |
| T1.1xC0_any | fixed_direction | 51.83 | 50.54 | 53.20 | 0.0000 | — | — | — | — | — |
| T1.1xC0_any | fixed_any | 51.83 | 50.54 | 53.20 | 0.0000 | — | — | — | — | — |
| T5.1xC1_appeared | Agent mean (3) | 65.67 | 83.91 | 53.96 | 0.0082 | 16.64 | 8.64 [4.13, 13.52] | 1.94 [-0.91, 5.30] | 103 | 0 / 6 |
| T5.1xC1_appeared | fixed_direction | 57.04 | 56.69 | 57.38 | 0.0037 | — | — | — | — | — |
| T5.1xC1_appeared | fixed_any | 49.03 | 42.59 | 57.78 | 0.0795 | — | — | — | — | — |
| T5.1xC1_appeared | rule_direction | 63.74 | 83.49 | 51.54 | 0.0028 | — | — | — | 20 | — |
| T5.1xC1_appeared | given_unchanged | 49.03 | 42.59 | 57.78 | 0.0795 | — | — | — | — | — |
| T5.1xC1_appeared | given_direction_only | 57.04 | 56.69 | 57.38 | 0.0037 | — | — | — | — | — |
| T5.1xC1_disappeared | Agent mean (3) | 37.52 | 55.45 | 28.47 | 0.0449 | 24.41 | 4.69 [-1.75, 11.39] | -0.43 [-4.16, 3.55] | 339 | 0 / 5 |
| T5.1xC1_disappeared | fixed_direction | 32.84 | 30.12 | 36.09 | 0.0123 | — | — | — | — | — |
| T5.1xC1_disappeared | fixed_any | 13.11 | 7.95 | 37.33 | 0.4259 | — | — | — | — | — |
| T5.1xC1_disappeared | rule_direction | 37.96 | 56.84 | 28.50 | 0.0218 | — | — | — | 36 | — |
| T5.1xC1_disappeared | given_unchanged | 13.11 | 7.95 | 37.33 | 0.4259 | — | — | — | — | — |
| T5.1xC1_disappeared | given_direction_only | 32.84 | 30.12 | 36.09 | 0.0123 | — | — | — | — | — |

F1/P/R 按每次重复累计像素后计算，Agent 行报三次均值。方向违反率分母为预测像素数，空预测记 0。CI 按图配对抽样，三次重复固定。Rebuilt：在线为账本回答次数（含重复核验），离线为去重区域回答次数。错误会话计入兜底，缺失会话拒绝汇总。
