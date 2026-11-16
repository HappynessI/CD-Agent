# N9 固定检索探索预声明（仅dev_tune，数据仍待人工审核）

本阶段新增入口不会读取clean confirm/report任务，不运行Agent或VLM，不用N8的相对z/p拒识及SegEarth硬路由。数据构造QC/六例审阅不是批准；本阶段是明确标记的探索性开发检查。正式确认/在线仍等用户明确同意新版数据。

在相同clean任务上比较：Baseline-Fixed为原R1几何候选的delta Top8，不去自匹配；N9-Fixed为delta/joint各Top5/8。R1重新基于新E计算面积比[.25,4]、宽高比[.333333,3]与双时相RGB均值RMS≤.25的原固定几何筛选。delta=cos(c2−c1,e2−e1)，endpoint=.5[cos(c1,e1)+cos(c2,e2)]，joint=.5(delta+endpoint)。候选来自查询两期原SAM提议，以T1然后T2顺序保留；相似度平局保留原索引顺序。

T3.1在排序前剔除候选交叠比例|C∩E|/|C|≥.15或IoU≥.15；其余候选取C\E并重新池化DINO特征。**没有N8的微小净区<16过滤，也无最小净区、分数门限或去重追加规则**。新E的特征无条件重新池化；旧N8已排好序的池、旧E向量、旧核验答案均不复用。Baseline仍以原C池化/排序，所有T3.1最终评分统一扣E。

T3.3按原外部R1定义，用独立示例两期RGB与E计算参考几何/特征；不把外部E坐标从查询候选或GT扣除，不进行同图自匹配剔除。它的delta Top8臂应与Baseline一致，可作为实现一致性对照；只对比表示与K，不能宣称跨图去自匹配带来收益。新版外部示例采用完整原生源图，因此历史裁剪图特征不可冒充精确缓存。

只可复用来源路径、文件元信息及操作完全匹配的原SAM/DINO网格缓存。CPU入口先报告两类任务的缺缓存操作，并完成T3.1缓存探索；缺失外部源图DINO由后续显式GPU入口按原本地模型真实补算，不调用API、不用替代示例。新缓存仅写N9结果目录。候选、分数、排序与推理白名单先落盘，之后才读取GT评分；任务transition类别只用于离线分组，不进入纯检索函数或模型输入。

按T3.1/T3.3分别最大化本类型dev_tune pooled像素F1；平局取较小K，再取delta。目标类/地表类只作分层诊断，不另选门限。输出是待批准数据上的探索性推荐，不自动触发确认/在线。完整P/R/F1、TP/FP/FN、空预测、池覆盖与实际任务/图像支持量全部保留；多任务可能共享图像，不算独立图像扩增。

CPU复现：`tools/run_n9_t3_fixed_tune.sh`，1CPU/8G/1h，4项专项测试→T3.1/T3.3缓存需求→T3.1缓存固定探索。GPU补算复现：`tools/run_n9_t3_fixed_tune_gpu.sh`，gpu46单GPU/2CPU/16G/2h，4项测试→两类dev_tune补缺缓存与比较。统一由主Agent exp-loop提交；输出`exploratory_fixed_tune/`。没有formal confirm或online CLI开关。

## 后续在线设计（尚未实现/运行）

三组是Baseline-Fixed、N9-Fixed、N9-Agent，固定两组离线跑；只有N9-Agent在获批且confirm≥20任务后运行3seed真实Agent。无需增加baseline-Agent组。

采用显式N9 T3会话开关，在此会话内给match_exemplar接入冻结的去自匹配/净区特征重排，保留props/IDs schema与Agent自主选择编辑，不影响其他任务或默认match_exemplar。通用view、SegEarth辅助、掩码编辑正常开放；新示例和候选仍用纯推理输入，不带GT端点类别。

必须同时处理planner与ask的same-transition语义：N9 T3专用system明确“正向示例T1→T2的同一地物类别转移”，去掉any-category宽泛定义和鼓励空提交措辞；空提交接口可保留但不作鼓励。ask每个候选使用示例T1/T2支持图及候选T1/T2支持图，区分示例与候选坐标，要求判断候选是否真实发生且与示例为同方向同类别转移，而非仅泛land-cover change。其依据仅RGB/E/C，可记录模型自述端点但不能从meta.transition填答案。完整实际图像输入/问题/usage保存，新的核验输入必须发新qwen3.7-flash请求。

G3提交阶段也必须使用同一N9 T3核验路径，不能手动ask正确却在submit退回泛变化核验。实施时测试默认其他任务prompt/schema不变、T3新会话不含GT词串/路径、初始/补核验一致、候选只返回不自动提交。此为接入设计，不意味着获人工批准或已经运行。
