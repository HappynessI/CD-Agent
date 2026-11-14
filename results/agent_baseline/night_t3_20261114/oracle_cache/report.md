# Historical oracle versus predicted text proposals

Oracle uses GT transition labels in historical SAM3 requests: diagnostic only. No new model/API calls. Existing strict directed-proposal coverage is checked against cached masks; relaxed intersections diagnose constraints, not a selected new method.

| Split | GT | Group | N | Method | F1 | P | R | Covered/GT pixels | Aligned/GT instances |
|---|---|---|---:|---|---:|---:|---:|---|---|
| all | all | 全部 | 60 | historical_oracle | 21.2867 | 76.4743 | 12.3641 | 29994/242589 | 7/132 |
| all | all | 全部 | 60 | predicted | 9.3130 | 54.2882 | 5.0934 | 12356/242589 | 3/132 |
| all | all | 目标类 | 27 | historical_oracle | 32.9522 | 78.5752 | 20.8476 | 29989/143849 | 7/87 |
| all | all | 目标类 | 27 | predicted | 15.3793 | 73.3991 | 8.5896 | 12356/143849 | 3/87 |
| all | all | 地表类 | 33 | historical_oracle | 0.0100 | 0.4739 | 0.0051 | 5/98740 | 0/45 |
| all | all | 地表类 | 33 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/98740 | 0/45 |
| all | positive | 全部 | 30 | historical_oracle | 21.5745 | 84.5831 | 12.3641 | 29994/242589 | 7/132 |
| all | positive | 全部 | 30 | predicted | 9.4505 | 65.3722 | 5.0934 | 12356/242589 | 3/132 |
| all | positive | 目标类 | 16 | historical_oracle | 33.6454 | 87.1368 | 20.8476 | 29989/143849 | 7/87 |
| all | positive | 目标类 | 16 | predicted | 15.7469 | 94.4432 | 8.5896 | 12356/143849 | 3/87 |
| all | positive | 地表类 | 14 | historical_oracle | 0.0100 | 0.4785 | 0.0051 | 5/98740 | 0/45 |
| all | positive | 地表类 | 14 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/98740 | 0/45 |
| all | empty | 全部 | 30 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| all | empty | 全部 | 30 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| all | empty | 目标类 | 11 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| all | empty | 目标类 | 11 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| all | empty | 地表类 | 19 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| all | empty | 地表类 | 19 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| tune | all | 全部 | 30 | historical_oracle | 0.0181 | 0.1040 | 0.0099 | 4/40315 | 0/33 |
| tune | all | 全部 | 30 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/40315 | 0/33 |
| tune | all | 目标类 | 15 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/19448 | 0/18 |
| tune | all | 目标类 | 15 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/19448 | 0/18 |
| tune | all | 地表类 | 15 | historical_oracle | 0.0383 | 44.4444 | 0.0192 | 4/20867 | 0/15 |
| tune | all | 地表类 | 15 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/20867 | 0/15 |
| tune | positive | 全部 | 11 | historical_oracle | 0.0198 | 4.1667 | 0.0099 | 4/40315 | 0/33 |
| tune | positive | 全部 | 11 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/40315 | 0/33 |
| tune | positive | 目标类 | 6 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/19448 | 0/18 |
| tune | positive | 目标类 | 6 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/19448 | 0/18 |
| tune | positive | 地表类 | 5 | historical_oracle | 0.0383 | 44.4444 | 0.0192 | 4/20867 | 0/15 |
| tune | positive | 地表类 | 5 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/20867 | 0/15 |
| tune | empty | 全部 | 19 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| tune | empty | 全部 | 19 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| tune | empty | 目标类 | 9 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| tune | empty | 目标类 | 9 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| tune | empty | 地表类 | 10 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| tune | empty | 地表类 | 10 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| confirm | all | 全部 | 30 | historical_oracle | 25.2389 | 84.7774 | 14.8264 | 29990/202274 | 7/99 |
| confirm | all | 全部 | 30 | predicted | 11.1931 | 66.7711 | 6.1085 | 12356/202274 | 3/99 |
| confirm | all | 目标类 | 12 | historical_oracle | 37.7862 | 87.3576 | 24.1067 | 29989/124401 | 7/69 |
| confirm | all | 目标类 | 12 | predicted | 18.0406 | 98.2272 | 9.9324 | 12356/124401 | 3/69 |
| confirm | all | 地表类 | 18 | historical_oracle | 0.0025 | 0.0956 | 0.0013 | 1/77873 | 0/30 |
| confirm | all | 地表类 | 18 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/77873 | 0/30 |
| confirm | positive | 全部 | 19 | historical_oracle | 25.2400 | 84.8014 | 14.8264 | 29990/202274 | 7/99 |
| confirm | positive | 全部 | 19 | predicted | 11.1986 | 67.1668 | 6.1085 | 12356/202274 | 3/99 |
| confirm | positive | 目标类 | 10 | historical_oracle | 37.7862 | 87.3576 | 24.1067 | 29989/124401 | 7/69 |
| confirm | positive | 目标类 | 10 | predicted | 18.0407 | 98.2350 | 9.9324 | 12356/124401 | 3/69 |
| confirm | positive | 地表类 | 9 | historical_oracle | 0.0025 | 0.0965 | 0.0013 | 1/77873 | 0/30 |
| confirm | positive | 地表类 | 9 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/77873 | 0/30 |
| confirm | empty | 全部 | 11 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| confirm | empty | 全部 | 11 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| confirm | empty | 目标类 | 2 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| confirm | empty | 目标类 | 2 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| confirm | empty | 地表类 | 9 | historical_oracle | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |
| confirm | empty | 地表类 | 9 | predicted | 0.0000 | 0.0000 | 0.0000 | 0/0 | 0/0 |

## Fixed tune positive examples

| Task | Pair label agreement | Predicted proposals | Oracle covered pixels | Predicted covered pixels | Predicted constraint evidence |
|---|---|---:|---:|---:|---|
| train_1136_5to1 | True | 2 | 0 | 0 | {"status": "complete", "truth_pixels": 3, "source_at_t1": 0, "target_at_t2": 0, "positive_intersection": 0, "after_source_loss": 0, "after_target_gain": 0, "strict_loss_and_gain": 0} |
| train_1538_1to3 | False | 0 | 0 | 0 | {"status": "complete", "truth_pixels": 5243, "source_at_t1": 0, "target_at_t2": 0, "positive_intersection": 0, "after_source_loss": 0, "after_target_gain": 0, "strict_loss_and_gain": 0} |
| train_1659_1to2 | False | 0 | 0 | 0 | {"status": "unknown_or_same_class_skip", "covered_pixels": 0} |
| train_1745_5to2 | False | 0 | 0 | 0 | {"status": "unknown_or_same_class_skip", "covered_pixels": 0} |
| train_1754_5to1 | False | 0 | 0 | 0 | {"status": "unknown_or_same_class_skip", "covered_pixels": 0} |
| train_1808_2to1 | False | 0 | 4 | 0 | {"status": "unknown_or_same_class_skip", "covered_pixels": 0} |
| train_2571_4to5 | False | 0 | 0 | 0 | {"status": "complete", "truth_pixels": 6, "source_at_t1": 0, "target_at_t2": 0, "positive_intersection": 0, "after_source_loss": 0, "after_target_gain": 0, "strict_loss_and_gain": 0} |
| train_2890_5to1 | False | 5 | 0 | 0 | {"status": "complete", "truth_pixels": 492, "source_at_t1": 0, "target_at_t2": 482, "positive_intersection": 0, "after_source_loss": 0, "after_target_gain": 0, "strict_loss_and_gain": 0} |
| train_2912_5to2 | False | 0 | 0 | 0 | {"status": "unknown_or_same_class_skip", "covered_pixels": 0} |
| train_7_1to2 | False | 0 | 0 | 0 | {"status": "unknown_or_same_class_skip", "covered_pixels": 0} |
| train_915_3to4 | False | 0 | 0 | 0 | {"status": "complete", "truth_pixels": 1, "source_at_t1": 0, "target_at_t2": 1, "positive_intersection": 0, "after_source_loss": 0, "after_target_gain": 0, "strict_loss_and_gain": 0} |
