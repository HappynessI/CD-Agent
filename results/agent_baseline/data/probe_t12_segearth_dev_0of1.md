只在本期非零标签区域算 P/R/IoU；不能据此推断未标注区域或全图质量。
编码 2（N.V.G surface）的文本固定为 non-vegetated ground；其余：low vegetation、tree、water、building、playground。
各类分别与 background 做二分类，概率 argmax 决定掩码；eval() 关闭上采样器 Dropout。SAM3 沿用 tile=512、pixel_thr=.85、score_thr=.5。

| 时期 | 类别 | SegEarth P | SegEarth R | SegEarth IoU | SAM3 P | SAM3 R | SAM3 IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| t1 | low vegetation | 44.27 | 83.91 | 40.81 | — | — | — |
| t1 | non-vegetated ground | 54.33 | 73.89 | 45.58 | — | — | — |
| t1 | tree | 16.38 | 62.20 | 14.89 | — | — | — |
| t1 | water | 20.28 | 72.11 | 18.81 | — | — | — |
| t1 | building | 31.52 | 80.50 | 29.28 | — | — | — |
| t1 | playground | 0.00 | 0.00 | 0.00 | — | — | — |
| t2 | low vegetation | 51.38 | 77.30 | 44.65 | — | — | — |
| t2 | non-vegetated ground | 27.79 | 67.93 | 24.57 | — | — | — |
| t2 | tree | 40.92 | 77.14 | 36.49 | — | — | — |
| t2 | water | 48.98 | 62.37 | 37.81 | — | — | — |
| t2 | building | 70.58 | 86.49 | 63.57 | — | — | — |
| t2 | playground | 0.00 | 0.00 | 0.00 | — | — | — |

SAM3 来源：未提供同图单的 SAM3 结果；表中留空，不伪造数值。
开发集选择规则：两期 IoU 均至少 0.25，按较低一期排序；这是可用性筛选，非统计显著性判断。
冻结建议类别：low vegetation, building
