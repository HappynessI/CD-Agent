# Fixed paired image diagnostics

Read-only metrics comparison. No images, features, labels or API are loaded. Partial cohorts cannot justify promotion. Bootstrap uncertainty does not select methods or weights; historical exposure of dev60 remains unchanged.

Bootstrap: paired images with replacement, seed 20261115, 5000 replicates, percentile 95% intervals. Each fixed dev60 task is one unique image (enforced by the original split). Tune and confirmation are never pooled. F1 differences are percentage points; counts are pooled pixels. Empty-GT F1 is zero by the original scoring convention, so inspect FP. Small strata may have degenerate intervals; intervals are descriptive, not selection tests.

## r1_endpoint/tune: complete (30/30)

Strict pair: R1_geometry_dino → R1_equal_joint, both Top8. Positive means remaining GT > 0.

| GT | Group | Images | ΔTP | ΔFP | ΔFN | ΔF1 [95%] |
|---|---|---:|---:|---:|---:|---|
| all | 全部 | 30 | 33 | -9966 | -33 | 0.168 [-0.16221108361563166, 0.717477787787286] |
| all | 目标类 | 15 | 0 | -4692 | 0 | 0.027 [-0.09667115524961196, 0.17623140818286168] |
| all | 地表类 | 15 | 33 | -5274 | -33 | 0.257 [-0.40495903939494055, 1.7230925592261477] |
| positive | 全部 | 11 | 33 | -11617 | -33 | 0.844 [0.06725696854123811, 3.888483218070523] |
| positive | 目标类 | 6 | 0 | -6583 | 0 | 0.275 [0.0, 0.8278816326962626] |
| positive | 地表类 | 5 | 33 | -5034 | -33 | 0.798 [0.0, 7.811686495424453] |
| empty | 全部 | 19 | 0 | 1651 | 0 | 0.000 [0.0, 0.0] |
| empty | 目标类 | 9 | 0 | 1891 | 0 | 0.000 [0.0, 0.0] |
| empty | 地表类 | 10 | 0 | -240 | 0 | 0.000 [0.0, 0.0] |

Full TP/FP/FN/F1 intervals and per-image deltas are in summary.json. Extremes are descriptive cases, not newly selected evaluation subsets.

tp benefits:
- [train_1659_1to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1659_1to2/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 33.0, 'fp': -403.0, 'fn': -33.0, 'f1': 1.7319073953540993}

tp losses:
- None observed.

fp benefits:
- [train_2912_5to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_2912_5to2/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -5735.0, 'fn': 0.0, 'f1': 0.0}
- [train_7_1to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_7_1to2/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -4631.0, 'fn': 0.0, 'f1': 0.0}
- [train_1322_2to4](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1322_2to4/metrics.json): empty, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -4571.0, 'fn': 0.0, 'f1': 0.0}
- [train_119_6to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_119_6to2/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -3521.0, 'fn': 0.0, 'f1': 0.0}
- [train_1065_2to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1065_2to5/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -1895.0, 'fn': 0.0, 'f1': 0.0}

fp losses:
- [train_1166_3to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1166_3to5/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 3588.0, 'fn': 0.0, 'f1': 0.0}
- [train_2928_3to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_2928_3to1/metrics.json): empty, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 3391.0, 'fn': 0.0, 'f1': 0.0}
- [train_1377_5to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1377_5to3/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 2174.0, 'fn': 0.0, 'f1': 0.0}
- [train_555_4to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_555_4to2/metrics.json): empty, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 812.0, 'fn': 0.0, 'f1': 0.0}
- [train_385_5to6](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_385_5to6/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 796.0, 'fn': 0.0, 'f1': 0.0}

f1 benefits:
- [train_1659_1to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1659_1to2/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 33.0, 'fp': -403.0, 'fn': -33.0, 'f1': 1.7319073953540993}
- [train_1745_5to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/tune/tasks/train_1745_5to2/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -281.0, 'fn': 0.0, 'f1': 0.1929206042674565}

f1 losses:
- None observed.

## r1_endpoint/confirm: complete (30/30)

Strict pair: R1_geometry_dino → R1_equal_joint, both Top8. Positive means remaining GT > 0.

| GT | Group | Images | ΔTP | ΔFP | ΔFN | ΔF1 [95%] |
|---|---|---:|---:|---:|---:|---|
| all | 全部 | 30 | -2578 | 3108 | 2578 | -0.857 [-5.804896409817051, 2.510946978107381] |
| all | 目标类 | 12 | -2836 | 3524 | 2836 | -2.498 [-12.55456380412049, 3.1796725996698894] |
| all | 地表类 | 18 | 258 | -416 | -258 | 0.139 [-0.1601718480798206, 0.7280754912267255] |
| positive | 全部 | 19 | -2578 | -9633 | 2578 | -0.691 [-6.914736498242707, 3.635074275902677] |
| positive | 目标类 | 10 | -2836 | 2872 | 2836 | -2.647 [-13.920178028699802, 3.0775609872348673] |
| positive | 地表类 | 9 | 258 | -12505 | -258 | 0.338 [-0.00964708436354833, 3.1564212855025726] |
| empty | 全部 | 11 | 0 | 12741 | 0 | 0.000 [0.0, 0.0] |
| empty | 目标类 | 2 | 0 | 652 | 0 | 0.000 [0.0, 0.0] |
| empty | 地表类 | 9 | 0 | 12089 | 0 | 0.000 [0.0, 0.0] |

Full TP/FP/FN/F1 intervals and per-image deltas are in summary.json. Extremes are descriptive cases, not newly selected evaluation subsets.

tp benefits:
- [train_2192_1to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2192_1to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 3941.0, 'fp': 8089.0, 'fn': -3941.0, 'f1': -1.5026462408996508}
- [train_1370_3to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_1370_3to1/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 258.0, 'fp': -1359.0, 'fn': -258.0, 'f1': 6.9424407695822525}
- [train_2590_3to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2590_3to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 105.0, 'fp': 620.0, 'fn': -105.0, 'f1': 3.8598829563072083}

tp losses:
- [train_2383_2to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2383_2to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': -6329.0, 'fp': -5161.0, 'fn': 6329.0, 'f1': -20.966673521787058}
- [train_2438_5to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2438_5to1/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': -553.0, 'fp': 936.0, 'fn': 553.0, 'f1': -6.600008088597782}

fp benefits:
- [train_1177_2to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_1177_2to3/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -11515.0, 'fn': 0.0, 'f1': 0.0}
- [train_2383_2to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2383_2to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': -6329.0, 'fp': -5161.0, 'fn': 6329.0, 'f1': -20.966673521787058}
- [train_1370_3to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_1370_3to1/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 258.0, 'fp': -1359.0, 'fn': -258.0, 'f1': 6.9424407695822525}
- [train_1527_5to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_1527_5to2/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -1359.0, 'fn': 0.0, 'f1': 0.0}
- [train_2461_5to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2461_5to3/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -1036.0, 'fn': 0.0, 'f1': 0.13853974176488765}

fp losses:
- [train_2192_1to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2192_1to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 3941.0, 'fp': 8089.0, 'fn': -3941.0, 'f1': -1.5026462408996508}
- [train_37_4to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_37_4to3/metrics.json): empty, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 4953.0, 'fn': 0.0, 'f1': 0.0}
- [train_2075_4to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2075_4to2/metrics.json): empty, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 4099.0, 'fn': 0.0, 'f1': 0.0}
- [train_156_2to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_156_2to3/metrics.json): empty, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 2341.0, 'fn': 0.0, 'f1': 0.0}
- [train_1129_4to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_1129_4to5/metrics.json): empty, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 2011.0, 'fn': 0.0, 'f1': 0.0}

f1 benefits:
- [train_1370_3to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_1370_3to1/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 258.0, 'fp': -1359.0, 'fn': -258.0, 'f1': 6.9424407695822525}
- [train_2590_3to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2590_3to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 105.0, 'fp': 620.0, 'fn': -105.0, 'f1': 3.8598829563072083}
- [train_427_3to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_427_3to1/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -78.0, 'fn': 0.0, 'f1': 0.15335577624836327}
- [train_2461_5to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2461_5to3/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -1036.0, 'fn': 0.0, 'f1': 0.13853974176488765}
- [train_2828_1to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2828_1to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -377.0, 'fn': 0.0, 'f1': 0.03641961417954054}

f1 losses:
- [train_2383_2to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2383_2to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': -6329.0, 'fp': -5161.0, 'fn': 6329.0, 'f1': -20.966673521787058}
- [train_2438_5to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2438_5to1/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': -553.0, 'fp': 936.0, 'fn': 553.0, 'f1': -6.600008088597782}
- [train_2192_1to5](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2192_1to5/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 3941.0, 'fp': 8089.0, 'fn': -3941.0, 'f1': -1.5026462408996508}
- [train_2043_2to3](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2043_2to3/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 41.0, 'fn': 0.0, 'f1': -0.746203365665032}
- [train_2227_3to2](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_endpoint/confirm/tasks/train_2227_3to2/metrics.json): positive, 地表类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 332.0, 'fn': 0.0, 'f1': -0.47633198174549385}

## r1_residual/tune: complete (30/30)

Strict pair: R1_direct4_verified → R1_residual_verified, both Top8. Positive means remaining GT > 0.

| GT | Group | Images | ΔTP | ΔFP | ΔFN | ΔF1 [95%] |
|---|---|---:|---:|---:|---:|---|
| all | 全部 | 30 | 0 | 726 | 0 | -0.027 [-0.1817727847271485, 0.02656054118760616] |
| all | 目标类 | 15 | 0 | 726 | 0 | -0.011 [-0.07282818929129681, 0.01088534803769018] |
| all | 地表类 | 15 | 0 | 0 | 0 | 0.000 [0.0, 0.0] |
| positive | 全部 | 11 | 0 | 726 | 0 | -0.076 [-0.6316611274544844, 0.08734781085832263] |
| positive | 目标类 | 6 | 0 | 726 | 0 | -0.044 [-0.22961016394929512, 0.037114519734934515] |
| positive | 地表类 | 5 | 0 | 0 | 0 | 0.000 [0.0, 0.0] |
| empty | 全部 | 19 | 0 | 0 | 0 | 0.000 [0.0, 0.0] |
| empty | 目标类 | 9 | 0 | 0 | 0 | 0.000 [0.0, 0.0] |
| empty | 地表类 | 10 | 0 | 0 | 0 | 0.000 [0.0, 0.0] |

Full TP/FP/FN/F1 intervals and per-image deltas are in summary.json. Extremes are descriptive cases, not newly selected evaluation subsets.

tp benefits:
- None observed.

tp losses:
- None observed.

fp benefits:
- [train_1136_5to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_residual/tune/tasks/train_1136_5to1/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': -197.0, 'fn': 0.0, 'f1': 0.0}

fp losses:
- [train_2890_5to1](/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_residual/tune/tasks/train_2890_5to1/metrics.json): positive, 目标类; ΔTP/FP/FN/F1={'tp': 0.0, 'fp': 923.0, 'fn': 0.0, 'f1': 0.0}

f1 benefits:
- None observed.

f1 losses:
- None observed.

