# T3.1 工具诊断

入口：`python -m agent_baseline.analysis.t3_diag --help`。只面向开发图，真实推理通过
Slurm GPU 作业运行。默认三路线全部执行，无 API。`--tasks` 接受开发任务全集，
先用 `t3_subset.pick(seed=1016)` 固定每图一任务，再以 `--names` 文件筛图；完整运行
要求 60 图。子集图名支持带或不带扩展名。固定 SAM+DINO 门限 0.6，并报告默认
0.00–0.95、步长 0.05 的扫描；`--thresholds` 可改扫描但必须含 0.6。

SAM+DINO 路线完全复用 t3_fixed 的 `WorkerPool`、两期 SAM 提议和 `task_scores`
（T2−T1 区域平均 DINOv2 向量对示例变化向量余弦）。`--vision-config` 使用现有
配置，缓存到 `--cache/vision`，不改旧缓存。`--device cuda[:N]` 控制 SAM3，动态
worker 绑定对应可见 GPU 后仍使用其原 cuda:0 实现。Slurm 的可见设备编号优先。

SAM3 安装包的 `Sam3Processor.add_geometric_prompt(box, label, state)` 支持纯视觉
正示例，box 是归一化 cxcywh；未设文字时内部使用 `visual`。现有 HTTP worker 和
ovcd_v0/sam3_t7 只实现文字调用，所以诊断新增独立 stdin/stdout worker，在线工具
接口不变。通过 `--sam3-python`、`--sam3-root`（sam3 包的父目录）、
`--sam3-checkpoint`、`--sam3-bpe` 指向已安装环境和本地权重；禁止隐式下载。

视觉路线将参考期整图置于左侧、查询期整图置于右侧，示例掩码外接框落在左侧，
调用同图几何提示检索，再只取右侧实例。对 source 用 T1 示例，对 target 用 T2
示例，各检索两期。这样跨期提示不会错误地框选已拆除物体所在空地。SAM3 的公开
processor 没有跨图片独立 exemplar 参数，拼接是明确的适配方法。整张拼接图缩放
到 resolution=1008，查询图的有效横向分辨率因此下降；本批未验证影响。

文字路线从 SECOND 类别编号/任务转换元数据取 source 和 target 名称，分别对
两期执行与 `segment_text` 相同的 set_text_prompt。两条 SAM3 路线都用
`source(T1) & ~source(T2) & target(T2) & ~target(T1)`，同时约束示例转换的起点和
终点。这比只检查单一目标类别的出现/消失更严格，所有规则在开发评分前固定。
对每个 source(T1)/target(T2) 实例裁出该变化区域形成候选，保留实例粒度，不再
借真值拆分。SAM3 全图推理，confidence=.1、实例分数≥.5、像素概率>.85；文字
路线是带类别信息的参照，不声称数学意义的上限。视觉模式不发送类别名。

所有路线在预测、真值和提议中移除示例像素。真值实例为剩余真值的 8 连通组件；
实例对齐召回上限是每个真值实例独立寻找任一提议 IoU≥0.5，不做贪心匹配。
像素覆盖上限是所有提议并集覆盖的真值比例；理想像素裁剪 F1 上限另外给出，
不能把它当成整实例选择可达到的成绩。报告保存逐任务分数、提议上限、累计像素
F1/P/R、开发扫描最高门限以及中文结论。没有跑 Agent 时，不对工作流原因下确定
结论；提议覆盖和匹配损失的观察与未验证的工作流归因分开陈述。

缓存包括可读请求/源文件状态/模型路径/推理设置，配置不同需新缓存目录。
`report.partial.json` 保存处理中已完成任务；原命令重跑会复用模型缓存。
`report.json` 和 `report.md` 仅在全部指定任务完成后写出。I/O 三次尝试不包裹模型
进程调用。真实模型、峰值显存、效果尚未验证，资源和命令见本批 NOTES.md。
