# N8 find_missed：重构候选、残差触发与证据核验

完整执行FM1→FM2→FM3；新候选经真实qwen3.7-flash V0/V0_evidence核验。原始候选源重建，未以N7旧表代替。
冻结配置：alpha0.0_res1024_V0_k5；语义门槛 0.2。实际usage累计费用 ¥0.099017。
N8 pooled与三组pooled均值晋级条件：False；同时要求三组task-macro均值也提高：False。

| split/config | group | pooled F1/P/R | macro F1 | pooled Δ | macro Δ | added TP/FP | VLM tasks | W/L/T |
|---|---|---|---|---|---|---|---|---|
| tune/alpha0.0_res1024_V0_k5 | all | 85.9429/81.2811/91.1721 | 55.2043 | -1.1450 | +3.3720 | 14562/34731 | 11/15 | 3/7/5 |
| tune/alpha0.0_res1024_V0_k5 | levir_any | 88.6051/83.9799/93.7693 | 88.4029 | -0.4670 | -0.4523 | 219/5004 | 5/5 | 0/5/0 |
| tune/alpha0.0_res1024_V0_k5 | second_appeared | 79.8281/76.2376/83.7736 | 63.2684 | -2.9869 | +3.2017 | 13446/29587 | 4/5 | 2/2/1 |
| tune/alpha0.0_res1024_V0_k5 | second_disappeared | 26.6421/17.2133/58.9117 | 13.9417 | +16.3459 | +7.3667 | 897/140 | 2/5 | 1/0/4 |
| confirm/alpha0.0_res1024_V0_k5 | all | 78.7483/77.2838/80.2694 | 50.4094 | -0.2642 | +0.9625 | 6215/13458 | 9/15 | 2/7/6 |
| confirm/alpha0.0_res1024_V0_k5 | levir_any | 85.2450/78.0943/93.8371 | 84.5758 | -1.0475 | -1.3588 | 0/11966 | 5/5 | 0/5/0 |
| confirm/alpha0.0_res1024_V0_k5 | second_appeared | 52.5505/74.2457/40.6672 | 46.0748 | +6.3439 | +4.2658 | 6215/1385 | 3/5 | 2/1/2 |
| confirm/alpha0.0_res1024_V0_k5 | second_disappeared | 26.6572/54.8440/17.6078 | 20.5775 | -0.0547 | -0.0195 | 0/107 | 1/5 | 0/1/4 |
| confirm/alpha0.0_res1024_V0_k10 | all | 78.6988/76.2401/81.3214 | 51.1477 | -0.3137 | +1.7008 | 12297/23574 | 9/15 | 3/6/6 |
| confirm/alpha0.0_res1024_V0_k10 | levir_any | 84.6400/76.8397/94.2029 | 83.9697 | -1.6524 | -1.9649 | 1638/21244 | 5/5 | 0/5/0 |
| confirm/alpha0.0_res1024_V0_k10 | second_appeared | 56.0790/74.9356/44.8046 | 48.2652 | +9.8724 | +6.4563 | 9973/2184 | 3/5 | 2/1/2 |
| confirm/alpha0.0_res1024_V0_k10 | second_disappeared | 28.8827/56.9928/19.3425 | 21.2082 | +2.1707 | +0.6112 | 686/146 | 1/5 | 1/0/4 |
| confirm/alpha0.0_res1024_V0_evidence_k5 | all | 78.6378/77.0712/80.2694 | 50.2855 | -0.3748 | +0.8387 | 6215/15115 | 9/15 | 2/7/6 |
| confirm/alpha0.0_res1024_V0_evidence_k5 | levir_any | 85.1236/77.8909/93.8371 | 84.3262 | -1.1688 | -1.6084 | 0/13371 | 5/5 | 0/5/0 |
| confirm/alpha0.0_res1024_V0_evidence_k5 | second_appeared | 52.4565/73.8716/40.6672 | 45.9529 | +6.2499 | +4.1439 | 6215/1637 | 3/5 | 2/1/2 |
| confirm/alpha0.0_res1024_V0_evidence_k5 | second_disappeared | 26.6572/54.8440/17.6078 | 20.5775 | -0.0547 | -0.0195 | 0/107 | 1/5 | 0/1/4 |
| confirm/alpha0.0_res1024_V0_evidence_k10 | all | 78.5350/76.0369/81.2028 | 50.8198 | -0.4776 | +1.3730 | 11611/25006 | 9/15 | 2/7/6 |
| confirm/alpha0.0_res1024_V0_evidence_k10 | levir_any | 84.5195/76.6413/94.2029 | 83.7241 | -1.7729 | -2.2105 | 1638/22665 | 5/5 | 0/5/0 |
| confirm/alpha0.0_res1024_V0_evidence_k10 | second_appeared | 56.0597/74.8666/44.8046 | 48.1580 | +9.8531 | +6.3490 | 9973/2234 | 3/5 | 2/1/2 |
| confirm/alpha0.0_res1024_V0_evidence_k10 | second_disappeared | 26.6572/54.8440/17.6078 | 20.5775 | -0.0547 | -0.0195 | 0/107 | 1/5 | 0/1/4 |

tune/alpha0.0_res1024_V0_k5：三组pooled F1均值 65.0251（基线60.7277）；三组task-macro均值 55.2043（基线51.8323）。
confirm/alpha0.0_res1024_V0_k5：三组pooled F1均值 54.8176（基线53.0703）；三组task-macro均值 50.4094（基线49.4469）。
confirm/alpha0.0_res1024_V0_k10：三组pooled F1均值 56.5339（基线53.0703）；三组task-macro均值 51.1477（基线49.4469）。
confirm/alpha0.0_res1024_V0_evidence_k5：三组pooled F1均值 54.7458（基线53.0703）；三组task-macro均值 50.2855（基线49.4469）。
confirm/alpha0.0_res1024_V0_evidence_k10：三组pooled F1均值 55.7455（基线53.0703）；三组task-macro均值 50.8198（基线49.4469）。

## 证据边界与成本
候选生成/特征读取在所有任务执行（find_missed生成率100%）；VLM触发率另列。残差信号在FM1之后获得，不声称提前节省候选生成成本。
15任务只对应10张物理影像；cluster区间与逐图剔除在image_cluster_sensitivity.json。macro为任务等权，pooled为累计像素，指标均为百分数。
历史候选设置及confirm已曝光，本轮是开发池内验证。GT仅进入标定、评分、诊断；不得称独立泛化。
全部API尝试含失败分别保留usage/，request输入在inputs/，解析答案在answers/；逐配置answer-cost为反事实选中答案费用，不等于整个搜索实际支出。
中间候选逐阶段TP/FP在candidate_diagnostics_tune/confirm.json；每门槛召回变化在frozen_semantic.json及fm1_tune_scores.json。
复现顺序：run_n8_missed_inventory.sbatch → run_n8_missed_fill.sbatch → run_n8_missed_prepare.sbatch → run_n8_missed_verify.sbatch tune → run_n8_missed_score.sbatch tune → verify confirm → score confirm → run_n8_missed_report.sbatch。由主Agent统一exp-loop Slurm提交。

未达到在线红线，停止在线推进。十例图像诊断见[诊断索引](diagnostics/index.html)，保留成败及预声明灾难例。
