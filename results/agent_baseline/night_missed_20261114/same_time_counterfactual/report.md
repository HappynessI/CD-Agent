# 固定六例同日期反事实

同一RGB分别作为T1/T1或T2/T2；保留原mask/bbox/方向/问法/模型参数。原四图重新渲染，GT不参与。yes违反此构造的无时相变化条件，但不等于自然任务收益、人工语义标签或线上晋级证据。固定6例来自既有review_pack，已有视觉曝光。

|UID|同日期输入|原V0|反事实回答|finish_reason|响应证据|
|---|---|---|---|---|---|
|levir_any:train_106:1|T1_T1|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/levir_any__train_106__1__T1_T1.json|
|levir_any:train_106:1|T2_T2|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/levir_any__train_106__1__T2_T2.json|
|levir_any:train_106:3|T1_T1|no|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/levir_any__train_106__3__T1_T1.json|
|levir_any:train_106:3|T2_T2|no|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/levir_any__train_106__3__T2_T2.json|
|second_appeared:train_1065:1|T1_T1|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_appeared__train_1065__1__T1_T1.json|
|second_appeared:train_1065:1|T2_T2|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_appeared__train_1065__1__T2_T2.json|
|second_appeared:train_1065:4|T1_T1|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_appeared__train_1065__4__T1_T1.json|
|second_appeared:train_1065:4|T2_T2|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_appeared__train_1065__4__T2_T2.json|
|second_disappeared:train_1065:1|T1_T1|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_disappeared__train_1065__1__T1_T1.json|
|second_disappeared:train_1065:1|T2_T2|yes|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_disappeared__train_1065__1__T2_T2.json|
|second_disappeared:train_1065:2|T1_T1|no|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_disappeared__train_1065__2__T1_T1.json|
|second_disappeared:train_1065:2|T2_T2|no|no|stop|/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/events/second_disappeared__train_1065__2__T2_T2.json|

完成 12/12；已记录usage费用 ¥0.0029352。每配置最多一次调用；未完成/失败不自动重试。
