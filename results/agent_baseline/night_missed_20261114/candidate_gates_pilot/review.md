# 固定30记录 S5门控完整审查

证据：本目录`score_summary.json`、`prepared/<group>/<stem>.json`及`scores/<group>/<stem>.json`。30任务方向×图像记录全部完成；3组各5 tune/5 confirm。30/30真实特征缓存可用，26图原0.25重匹配与历史缓存逐像素一致，无不一致。其余4图无已核验种子（新建train1166/train119、拆除train1065/train1177），S5为空，不是丢缓存。

以下仅应用tune已选gate，不按确认改阈值。TP/FP保留分母是原S5候选union，不是完整五来源池。

|组|冻结gate|tune TP保留/FP保留|confirm TP原→门控|confirm FP原→门控|confirm TP保留/FP保留|
|---|---|---|---|---|---|
|SECOND新建|dino_020|99.792%/7.939%|6763→4432|21454→5375|65.533%/25.054%|
|SECOND拆除|dino_020|100%/6.546%|6681→6679|21364→949|99.970%/4.442%|
|LEVIR|dino_010|92.178%/68.913%|6926→6926|953350→919494|100%/96.449%|

新建gate在tune满足≥80%TP规则，但确认保留率降到65.53%；这不是本轮重新择优的理由，而是该阈值稳定性不足的证据。确认丢2331TP，其中train127丢1592（3762→2170，保留57.68%）、train1370丢719（719→0），合99.14%；另train1177丢20。train1370的719TP/5035FP全部被清空。门控少FP同时少TP，不能只强调FP减少。

拆除gate保留几乎所有TP，并显著去FP：全部20415FP减少来自train127（11452→481，零TP图）与train1370（9504→60，TP17→15）。train119的6664TP/400FP完全保留，占保留TP99.78%。因此方向上有候选价值密度改善，但正像素收益高度集中，需看后续固定排名和核验的实际结果。

LEVIR gate确认只有train124发生变化：1248TP不变、FP246405→212549，减少33856FP；另外4图完全不变。总FP仍919494，远不能称已解决LEVIR误候选污染。特征缓存全部存在，不能把弱门控效果解释为缺特征。

## 未核验add-all指标的边界

相对旧S5 add-all，确认新建ΔF1+2.7316，图像配对95%区间[-3.4227,12.7699]；拆除+11.6206，[0,17.5836]；LEVIR+0.7869，[0,2.1234]。这些是把候选全加入起点的离线诊断，不是VLM核验或Agent表现。新建聚合F1上升不能抵消其TP保留明显下降这一风险；LEVIR正差异也不代表与起点相比净改善。

## S4独有像素与来源互补

独有定义为S4相对其余全部来源union；以下tune/confirm各5图，绝非320任务全开发结果。

|组|tune独有TP/FP|confirm独有TP/FP|
|---|---|---|
|SECOND新建|6274/38171|997/4046|
|SECOND拆除|4/109|1773/7869|
|LEVIR|0/0|0/0|

新建确认的train1177/119/127/1322/1370分别有S4独有TP210/65/114/476/132；其中train1322还伴独有FP3474，说明同一来源含补漏价值和污染，不能据几何分支的大FP候选裁掉整源。拆除确认独有TP主要train119=1769，另train1322=4；其独有FP亦集中train119=6420。LEVIR本池S4独有为零，不可据此否定SECOND上的互补作用。

## 全30记录逐图核查表

|组|划分|图像|原S5 TP/FP|冻结gate TP/FP|S4相对全部其余源独有TP/FP|
|---|---|---|---|---|---|
|second_appeared|tune|train_1065|274/26|274/26|1393/5844|
|second_appeared|tune|train_1129|1611/387|1611/387|3835/7254|
|second_appeared|tune|train_1136|57/14|51/14|572/25035|
|second_appeared|tune|train_1166|0/0|0/0|6/0|
|second_appeared|confirm|train_1177|2253/15134|2233/5159|210/24|
|second_appeared|confirm|train_119|0/0|0/0|65/467|
|second_appeared|confirm|train_127|3762/1124|2170/56|114/33|
|second_appeared|confirm|train_1322|29/161|29/160|476/3474|
|second_appeared|tune|train_1344|942/6047|942/87|468/38|
|second_appeared|confirm|train_1370|719/5035|0/0|132/48|
|second_disappeared|tune|train_1065|0/0|0/0|0/0|
|second_disappeared|tune|train_1129|0/4|0/3|0/0|
|second_disappeared|tune|train_1136|821/1585|821/1087|4/78|
|second_disappeared|tune|train_1166|0/2|0/1|0/0|
|second_disappeared|confirm|train_1177|0/0|0/0|0/105|
|second_disappeared|confirm|train_119|6664/400|6664/400|1769/6420|
|second_disappeared|confirm|train_127|0/11452|0/481|0/0|
|second_disappeared|confirm|train_1322|0/8|0/8|4/1253|
|second_disappeared|tune|train_1344|0/29651|0/954|0/31|
|second_disappeared|confirm|train_1370|17/9504|15/60|0/91|
|levir_any|tune|train_106|1268/164030|1260/82519|0/0|
|levir_any|tune|train_109|2897/320091|2734/201820|0/0|
|levir_any|tune|train_11|1942/314673|1331/171027|0/0|
|levir_any|tune|train_116|2830/271618|2830/271618|0/0|
|levir_any|confirm|train_118|10/154124|10/154124|0/0|
|levir_any|confirm|train_119|2004/196352|2004/196352|0/0|
|levir_any|confirm|train_121|522/198022|522/198022|0/0|
|levir_any|confirm|train_122|3142/158447|3142/158447|0/0|
|levir_any|tune|train_123|1060/73978|1060/61645|0/0|
|levir_any|confirm|train_124|1248/246405|1248/212549|0/0|

## 下一判断边界

保留所有来源、三组冻结gate、全局V0及主evidence@10。等排名实际完成后，分别比较gated balanced对旧balanced（候选门控）、evidence对gated balanced（排序），检查图像配对区间、逐图净TP/FP、完整池与TopK召回及费用；当前不能预言排名结果。新建DINO门控的召回损失尤其需要报告是否由其他来源覆盖，不能把S5损失直接等同全池损失，也不能假设自动补回。

WHU官方审计另作独立口径，尚未由本报告声称完成。历史S3/S4阈值使用过全组开发GT，包括当前确认图；本轮选择隔离成立，但候选池不是独立未见确认集，不宣称独立泛化。
