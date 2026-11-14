# 历史候选池的选择曝光限制

已核实本轮复用batch99候选池的S3/S4阈值使用过现在确认图的GT。**本轮gate/verifier的tune选择隔离成立，但候选池不是全新未见确认集，不能宣称独立泛化。** 阈值没有在本轮重扫；原缓存与历史记录未改。

原始代码快照：`/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch99_find_missed_pre/code/agent_baseline/analysis/missed_sources.py`。`statistics_group`调用全组`rows_for`，`scan_thresholds`遍历每图GT，对S3/S4计算起点union来源的累计像素F1，从0–2步长0.05网格选最大值（平局取较小阈值）；之后`raw_sources`应用该阈值并生成`candidates/<group>/<source>/<stem>.npz`。

实际保存的配置与结果根目录：`/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch99_find_missed_pre/artifacts/statistics/`。

|组|参与图数|S3阈值|S4阈值|记录|
|---|---:|---:|---:|---|
|second_appeared|60|0.40|0.35|second_appeared/summary.json 与 settings.json|
|second_disappeared|60|0.25|0.40|second_disappeared/summary.json 与 settings.json|
|levir_any|200|0.45|0.85|levir_any/summary.json 与 settings.json|

settings.json 的inputs包括每图base、raw和GT路径。直接证据包括当前确认图：SECOND train1177/train1322的`second_dev/second/gt/new/...png`；LEVIR train118/train124的`datasets/LEVIR-CD/train_original/label/...png`。`summary.json`保留threshold_scan与images数，证明不是仅用本轮tune五图选择。

因此本轮图像划分仅控制新的核验器/门控选择；当前确认更适合作为历史已曝光开发池中的探索性配对检验。不能将其改称测试集，也不能通过更名split消除历史标签参与。GT未进入本轮在线候选选择或推理；历史离线选阈值是另一层需要披露的事实。
