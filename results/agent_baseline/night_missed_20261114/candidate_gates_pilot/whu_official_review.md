# WHU 官方来源审计复核

依据已完成 `whu_official_summary.json`：any、appeared、disappeared 各60条，共180条开发任务记录。本次仅读取既有汇总与实现，没有重新评分或实验。

口径：官方 `whu/gt_official_v3/c0`、`new`、`demolished`。C0 any 不传 ignore；C1 新建/拆除使用同目录 `ignore/<stem>.png`。`night_missed_candidates.whu` 对 baseline 与每个来源 union 都先与 valid 相交，再计算 TP/FP/FN；所以 ignore 区域不成为 FP，也不成为 FN。各来源内部按 mask union 去重；来源之间有重叠，不可将各行相加当联合候选收益。这里是旧缓存对新官方 GT 的审计，不是重建来源或核验器实验。

下表 TP/FP 是来源补漏像素；FN 是将该来源全部并入原始 baseline 后剩余的官方漏检像素（baseline FN 减 covered），不是孤立来源预测整幅图的 FN。baseline 原有 FP 不包括在来源 FP 中。逐图原始 baseline TP/FP/FN 与 add_all TP/FP/FN 均保存在 summary 的 records；不与旧 GT baseline_counts 混用。

|组|来源|补漏TP|来源FP|并入后FN|相对其余全部来源独有TP|独有FP|
|---|---|---:|---:|---:|---:|---:|
|any|S1_02|85125|257854|421035|6490|120081|
|any|S2|105587|182003|400573|18608|64566|
|any|S3|198026|44123|308134|12148|30478|
|any|S4|194674|352223|311486|10455|262588|
|any|S5|190413|926037|315747|88461|756287|
|appeared|S1_02|67169|198176|389429|3450|152725|
|appeared|S2|84633|81399|371965|20832|48644|
|appeared|S3|197362|38581|259236|12070|26064|
|appeared|S4|190319|334246|266279|9602|302410|
|appeared|S5|176362|52959|280236|90030|22621|
|disappeared|S1_02|8001|56379|18725|344|28841|
|disappeared|S2|15108|96288|11618|8521|71849|
|disappeared|S3|0|27576|26726|0|0|
|disappeared|S4|9379|736912|17347|2857|689340|
|disappeared|S5|477|10856|26249|6|5198|

原始 baseline 官方 FN 分别为506160、456598、26726。S3在 any/新建中提供较高 TP、相对低 FP；S5的新建独有 TP 90030 值得保留，但 any 的926037 FP 显示不能跨方向套用同一结论。拆除 S4 有2857独有TP，同时689340独有FP，不能把来源互补等同于可直接接受。拆除 S3 此池无TP且无独有FP，说明它与其他来源重叠；这只是本开发池观察，不据此重选门槛或剪除来源。没有在线Agent或真实Verifier收益结论。

历史曝光限制：候选仍来自 batch99 的历史 S3/S4 全组 GT 阈值选择，不能称新的未见确认池。证据根目录 `/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch99_find_missed_pre/artifacts/statistics/`，每个 `whu_*/settings.json` 含旧 `whu/gt/...` 输入，`summary.json` 保存60图 threshold_scan。历史阈值 any/appeared 为 S3=.15、S4=.55，disappeared 为 S3=.10、S4=.00。旧阈值使用旧 GT，不应称以官方 GT 调参；此次仅用官方 GT 重评分，没有消除历史标签选择曝光，也没有覆盖旧记录。SECOND/LEVIR 新 gate/verifier 的 tune 隔离与历史候选池曝光是不同层次。

一次性排名落盘检查：`rank_V0/progress.json` 当前 stage=prepare_rank，completed=23，expected=30；未见最终排名确认结论，本审查不推测其收益。
