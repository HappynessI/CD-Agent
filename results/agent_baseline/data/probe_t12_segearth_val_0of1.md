只在本期非零标签区域算 P/R/IoU；不能据此推断未标注区域或全图质量。
编码 2（N.V.G surface）的文本固定为 non-vegetated ground；其余：low vegetation、tree、water、building、playground。
各类分别与 background 做二分类，概率 argmax 决定掩码；eval() 关闭上采样器 Dropout。SAM3 沿用 tile=512、pixel_thr=.85、score_thr=.5。

| 时期 | 类别 | SegEarth P | SegEarth R | SegEarth IoU | SAM3 P | SAM3 R | SAM3 IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| t1 | low vegetation | 50.64 | 79.09 | 44.66 | 68.23 | 10.67 | 10.17 |
| t1 | non-vegetated ground | 42.96 | 75.29 | 37.65 | 37.01 | 13.54 | 11.01 |
| t1 | tree | 21.36 | 87.05 | 20.70 | 64.06 | 2.44 | 2.41 |
| t1 | water | 10.49 | 29.81 | 8.41 | 41.31 | 36.62 | 24.09 |
| t1 | building | 47.29 | 79.90 | 42.26 | 85.18 | 43.00 | 40.01 |
| t1 | playground | 0.00 | 0.00 | 0.00 | — | 0.00 | 0.00 |
| t2 | low vegetation | 38.74 | 81.76 | 35.66 | 48.45 | 13.11 | 11.51 |
| t2 | non-vegetated ground | 35.46 | 59.58 | 28.58 | 37.15 | 11.52 | 9.64 |
| t2 | tree | 39.76 | 82.66 | 36.70 | 72.61 | 3.14 | 3.11 |
| t2 | water | 15.44 | 35.28 | 12.03 | 74.91 | 16.85 | 15.95 |
| t2 | building | 69.83 | 76.54 | 57.52 | 96.84 | 61.48 | 60.27 |
| t2 | playground | 5.32 | 37.68 | 4.89 | — | 0.00 | 0.00 |

SAM3 来源：/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/second/t12_probe.json
开发集选择规则：两期 IoU 均至少 0.25，按较低一期排序；这是可用性筛选，非统计显著性判断。
冻结建议类别：low vegetation, building
