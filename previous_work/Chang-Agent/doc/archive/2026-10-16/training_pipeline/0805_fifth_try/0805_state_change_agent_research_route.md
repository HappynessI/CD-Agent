# 从“分割修补闭环”转向“目标状态变化智能体”：技术路线、数据闭环与论文策略

> 日期：2026-08-05  
> 性质：研究路线决策稿，不是实现 freeze、实验预注册或训练授权  
> 核心问题：当前 Chang-Agent 是否还有继续做的价值；若目标是发表“有状态理解能力的变化检测智能体”，研究问题、数据、模型、Agent 与评测应怎样重新定义。

## 0. 执行结论

### 0.1 一句话判断

**项目还有机会，但必须做研究问题级 pivot：保留现有工具与审计基础设施，把“单图分割后匹配、再修 change mask”降级为 baseline；论文主任务改成“可查询、完整性可评测、风险可校准的跨时相目标状态理解”。**

继续完成当前 0805 规划中的工具路由、Verifier 阈值、停止步数和运行效率，只能得到一个更成熟的 segmentation correction loop，不能自然长成状态理解论文。它们可以以后作为系统执行层，但不应继续支配近期研究资源。

### 0.2 最重要的五个判断

1. **当前主张与实际任务不一致。** 现有系统的事实链是 T1/T2 独立建筑分割、实例重叠匹配、生成 change mask、局部修补；这不是联合学习对象身份与状态演变的端到端变化检测。
2. **漏检不是一个后处理 bug，而是任务不完整。** 若某对象没有进入 proposal/change-mask 支持集，Verifier 无论多强都看不到它。必须增加独立于当前预测的全场景发现与“未解释残差”机制，并以 changed-object recall 直接评测。
3. **LEVIR-CD 的 change mask 监督不可辨识。** 同一 change mask 对应许多不同的 T1/T2 object-mask 组合，无法唯一监督修改哪一期、对象身份、状态方向或“重建/损毁”。
4. **“Agent”“状态”“自然语言查询”“主动放大”本身在 2026 年都不是创新。** 已有 Change-Agent、ChangeChat、Referring Change Detection、FootprintNet、GeoVista、FarmSeeker 等直接工作。论文必须提出它们尚未解决的、可测量的新问题。
5. **最值得验证的核心假设是：完整对象状态图 + 漏检风险建模 + 预算化主动取证，是否能产生不可由现有模块组合替代的收益。** 若 action oracle 在等算力条件下没有显著收益，就不要强行做 Agent；先发 object-state benchmark/model 反而更稳。

### 0.3 建议的主任务与系统工作名

- 研究任务：**Queryable Object-State Change Understanding（Q-OSCU，可查询目标状态变化理解）**。
- 结构化输出：**Temporal Object-State Graph（TOSG，跨时相目标状态图）**。
- 智能体工作名：**StateGraph-Agent**。正式投稿前需要做名称检索；不建议继续把 `Change-Agent` 当作新名称，因为 2024 年已有同名工作。

目标不再是“输出更好的 change mask”，而是：

> 给定多时相影像与一个状态谓词，在具有完整 GT 的 benchmark 上可评测地返回满足条件的实体，同时估计场景级残余漏检风险，并输出每一期的对象证据、跨期身份关系、状态转移、置信度和不确定性；binary change mask 只是结构化结果的一个投影。

可以把整篇工作的核心科学问题压缩成一句：

> **在有限观测预算下，系统能否比最强静态模型和最邻近模块组合更完整、更校准地返回满足查询的可观测对象状态转移，并给出可核验的跨时相身份与视觉证据？**

### 0.4 最大风险：状态档案缺少独立 GT

TOSG 可以作为模型生成的对象状态假设，但当前 LEVIR-CD/LEVIR-MCI 的 binary change mask 不能提供 audit-complete 的真实档案：稳定对象未被标注，T1/T2 存在性、方向、实例身份和关联也无法唯一恢复。若先用模型预测生成档案，再用同一档案训练或证明模型正确，就会形成循环验证；前置漏检还会同时从档案、Verifier 和评测中消失。

少量人工 gold data 只能承担独立审计和小范围适配，不能单独证明广泛泛化。进入 Agent 开发前，必须先做跨城市、传感器或数据集的同任务 go/no-go：若基础 object-state 模型在隔离测试域明显失效，或不能稳定优于强静态基线，就应缩小为特定域任务，补充独立 GT，或停止完整状态 Agent 路线。SFT、伪标签、FM 特征插值和 Agent 工具调用都不能补回数据中不存在的状态真值。

---

## 1. 最近文档已经说明：问题位于任务层，而不只在实现层

本节只用仓库事实界定研究问题，不展开工具实现细节。

### 1.1 当前系统实际上是什么

- [`README.md`](../../../README.md) 第 3–10 行明确写明，production 路径是 `Verifier planner → Environment → deterministic Executor`，**没有独立 Agent policy**。
- [`doc/pipeline.md`](../../pipeline.md) 第 12–22、60–66 行明确给出主链：分别分割 T1/T2 building，做连通实例与 overlap matching，未匹配实例并成 change mask，再由 Verifier 规划局部修改。
- 同一文档第 103–110 行再次指出，当前准确名称不是独立 Agent，而是 Verifier 同时承担诊断与动作规划。

所以目前最准确的研究描述是：

> 一套带语义审计与安全回滚的、GT-free 双时相分割修补框架。

它有工程价值，也能成为有意义的 baseline，但不能支撑“模型已经理解目标状态演变”的主张。

### 1.2 最近实验并没有产生 Agent 或状态理解证据

- [`0803 oracle trajectory 文档`](../0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md) 第 397–424 行显示，4 个非初始达标 episode 的 endpoint reachability 为 `0/4`，正式 C-ISO/T-TRAJ 均为 0，`training_authorized=false`。
- [`0804 文档`](../0804_fifith_try/0804.md) 第 26–35 行显示，0803/0804 的 Qwen 调用均为 0，只测试真实工具轨迹可达性；0804 又因第四样本取消而为不完整 closeout，`scientific_conclusion=null`。
- 因而最近两轮最多是在回答“当前动作空间能否局部改善 mask”，没有回答模型能否发现漏掉的对象、建立身份对应、判断状态或做规划。

### 1.3 数据不可辨识已经被仓库自己确认

[`0804 文档`](../0804_fifith_try/0804.md) 第 287–301、343–368 行已明确区分：

- 有 T1/T2 building masks 时，可以分别评价两期分割并发现两侧错误抵消；
- 只有 change mask 的 LEVIR-CD 无法唯一反推应修改 T1 还是 T2，也不能证明单时相 building mask 正确；
- 即使假设 `Change = T1 XOR T2`，仅知道 XOR 也无法唯一恢复两个输入。

这不是样本量再增加一点就能解决的问题，而是监督变量缺失造成的不可辨识性。

### 1.4 0805 五分支解决不了论文硬伤

[`0805.md`](./0805.md) 第 434–479 行规划了效率、point/box 候选、mixed error、router、gate 与 horizon。它们能改善“已有候选怎样修”，却没有重新定义：

- 全场景对象如何被穷尽发现；
- 两期对象怎样共同建模与关联；
- 状态标签及其可观测证据是什么；
- 验证器怎样判断一个完整的对象—属性—状态假设；
- 论文如何评价对象级漏检、身份错误和状态错误。

因此建议：**暂停把五分支当作论文主路线。只保留低成本维护与可复用资产，不再在任务定义完成前启动大规模 SFT 或工具穷举。**

---

## 2. 2026 年竞争格局：哪些“创新点”已经不够

以下判断基于截至 2026-08-05 的公开论文和 arXiv 元数据。2025–2026 年部分工作仍是预印本，引用它们是为了判断竞争格局，不等同于同行评审背书。

| 方向 | 代表工作 | 已经覆盖的主张 | 对本项目的含义 |
|---|---|---|---|
| 交互式变化 Agent | [Change-Agent, 2024](https://arxiv.org/abs/2403.19646)、[ChangeGPT, 2026](https://arxiv.org/abs/2601.02757)、[Forest-Chat, 2026](https://arxiv.org/abs/2601.14637) | LLM 编排变化检测、描述、计数、问答与工具 | “把检测器接给 LLM、能对话”不能作为核心创新 |
| 交互式/指令变化理解 | [ChangeChat, 2024](https://arxiv.org/abs/2409.08582)、[DeltaVLM, 2025](https://arxiv.org/abs/2507.22346)、[ChangeIMTI/ChangeVG, 2025](https://arxiv.org/abs/2509.23105) | 大规模 instruction 数据，caption、分类、计数、定位、multi-turn | checklist 或 instruction tuning 本身不新；必须证明结构化状态与完整性 |
| 查询答案与 mask grounding | [VisTA/QAG-360K, 2024](https://arxiv.org/abs/2410.23828)、[TEOChat, 2025](https://arxiv.org/abs/2410.06234) | 问题条件下联合文本答案与视觉 mask；1–8 帧 EO 对话、建筑变化与损毁评估 | “用户提问 + 定位 mask + 解释”已有强直接竞品 |
| 检测与语言统一 | [ChangeMinds, 2024](https://arxiv.org/abs/2410.10047)、[Delta-LLaVA, 2026](https://arxiv.org/abs/2604.14044)、[JL1-CC&QA, 2026](https://arxiv.org/abs/2606.31745) | change mask、caption、VQA/推理的联合学习 | “既分割又解释”已经拥挤；自由文本不是可靠解释性 |
| 端到端双时相视觉建模 | [BIT, 2021](https://arxiv.org/abs/2103.00208)、[ChangeFormer, 2022](https://arxiv.org/abs/2201.01293)、[Changer, 2023](https://arxiv.org/abs/2209.08290)、[ChangeMamba, 2024](https://arxiv.org/abs/2404.03425) | 双时相 feature interaction、token/Transformer/SSM 联合建模；覆盖 binary/semantic/damage 等任务 | 独立单图分割后 XOR/匹配必须作为弱 baseline，而非端到端主方法 |
| 语义/开放词汇变化 | [Semantic-CD, 2025](https://arxiv.org/abs/2501.06808)、[Referring Change Detection, 2025](https://arxiv.org/abs/2512.11719)、[ReasonCD, 2025](https://arxiv.org/abs/2512.19354)、[Seg2Change, 2026](https://arxiv.org/abs/2604.11231) | 文本指定类别或 change-of-interest，开放词汇/隐式意图 | “给自然语言条件，再输出对应 change mask”单独看也不够新 |
| SAM/实例式开放变化 | [AnyChange, 2024](https://arxiv.org/abs/2402.01188)、[OmniOVCD, 2026](https://arxiv.org/abs/2601.13895)、[ReA-OVCD, 2026](https://arxiv.org/abs/2606.20032) | training-free SAM latent matching、point query、SAM3、实例/像素一致性与开放词汇变化 | 训练模型输出 point/box 或接入 SAM/SAM3 只能是实现组件 |
| 对象追踪与建设/拆除 | [SpaceNet 7 / MUDS + SCOT, 2021](https://arxiv.org/abs/2102.04420) | exhaustive building polygons、跨月 unique ID、construction/demolition tracking、对象级 SCOT | “首次做对象身份/生命周期追踪”不能成立；增量必须来自查询、可观测性、漏检风险和主动验证 |
| 状态/多时相动态 | [Bi-SRNet, 2021](https://arxiv.org/abs/2108.06103)、[SCanNet, 2022](https://arxiv.org/abs/2212.05245)、[SMART-HC-VQA, 2026](https://arxiv.org/abs/2605.10739)、[FootprintNet, 2026](https://arxiv.org/abs/2607.27969) | `from-to` 语义转移、施工属性/活动阶段、多次变化区间、显式 state-action transition constraint | “从变化上升到目标—属性—状态”这个口号也已不新；需要对象身份、完整性和查询闭环 |
| 主动感知与遥感 Agent | [ZoomEarth, 2025](https://arxiv.org/abs/2511.12267)、[GeoEyes, 2026](https://arxiv.org/abs/2602.14201)、[GeoVista, 2026](https://arxiv.org/abs/2605.14475)、[GeoLens, 2026](https://arxiv.org/abs/2607.25993)、[FarmSeeker, 2026](https://arxiv.org/abs/2607.28186) | 高分辨率主动裁剪、证据跟踪、多工具调用、信息增益、学习停止 | “会 zoom、会调工具”也不能单独立项；必须是变化任务特有的状态假设检验 |

FootprintNet 所称 `action` 是潜在地表变化过程，不是 Agent 的工具调用或主动观测动作；但它已经直接占据“显式状态转移建模”的创新空间。

### 2.1 待验证的研究假设

从这些工作的公开任务定义看，可以提出一个待验证、而非预先成立的交叉假设：

1. 最终输出不是一张变化图或一段描述，而是**带身份、每期 mask、属性、转移和不确定性的 temporal entity graph**；
2. 任务在具有完整 GT 的场景上**显式评价实体检索完整性**，并估计残余漏检风险，而不是只解释已有 change proposal；
3. Agent 对对象假设执行主动取证、配对、重关联与验证，动作改变后续可用证据；
4. 评测同时覆盖事件召回、状态正确性、证据一致性、校准和成本，而不是只报 change IoU 或语言分数。

这不是“绝对无人做”的结论，也不意味着把几个模块拼起来就自动形成创新。正式投稿前仍需系统 related-work review，并用最邻近组合基线证明增量；但这个假设比“更好的 mask 修正 Agent”有更清楚的论文边界。

需要特别警惕四个直接撞车点：

- `Change-Agent` 已是 TGRS 2024 论文名称和公开系统，不应再把同名或“首次交互式变化 Agent”作为 claim；
- VisTA、Referring Change Detection 和 ReasonCD 已覆盖 query-conditioned grounding/检测与解释；
- SMART-HC-VQA 与 FootprintNet 已直接触及属性—状态演进或 state-transition；
- GeoVista、GeoLens、FarmSeeker 等已覆盖主动区域搜索、工具调用、信息增益和轨迹训练。

因此剩余创新不能靠术语拼接，而必须落到**audit-complete entity inventory、显式 identity/lineage、前置漏检可恢复性和事件级等预算评测**上。

---

## 3. 重新定义任务：从 change mask 到跨时相目标状态图

### 3.1 输入与输出

给定：

- 同一地理区域的两期或多期影像 `I_1 ... I_K`；
- 必要的时间、传感器、分辨率和配准元信息；
- 一个自然语言查询，先编译为可执行的 typed state predicate。

查询负责规定“要检索和报告哪类状态”，可以影响 Agent 的取证优先级，但不能反过来创造视觉事实。底层 object inventory、identity 与 state 必须能被影像和元数据独立核验。

例如：

```text
category = building
presence(T1) = present
presence(T2) = absent
geometry_change = any
condition = observable
```

系统输出一个实体集合。每个实体至少包含：

```text
entity_id / category
mask_or_geometry@T1 ... mask_or_geometry@TK
presence@T1 ... presence@TK
cross-time association / lineage
geometry and dataset-supported attributes
transition state
uncertainty / unknown reason
grounded visual evidence
```

传统 binary change mask 由这些实体状态渲染得到，而不是反过来作为所有对象假设的唯一来源。

更严格地说，TOSG 应由“每一期实际存在的对象节点”和“跨期关系边”组成：appeared/disappeared 对应 birth/death 或缺边，而不是给不存在的对象伪造一张 mask；split/merge 使用一对多或多对一关系。`same entity` 也不是天然视觉真值——同址拆除后新建在双时相下可能不可辨识，必须事先规定基于官方 track ID、空间连续性或人工判读的 annotation convention，并允许 `identity-unknown`。

### 3.2 状态要因子化，不要写成一个巨大的类别表

建议把状态拆成可组合的轴：

- **存在性**：present / absent / unknown；
- **身份关系**：same / unmatched / split / merge / uncertain；
- **几何属性**：stable / expanded / shrunk / reshaped / topologically changed；
- **语义属性**：class-from → class-to；
- **条件属性**：normal / damaged 等，仅在数据真正提供标签时启用；
- **可观测性**：clear / occluded / cloud / registration-ambiguous / resolution-insufficient。

这样可以跨数据集做 partial supervision，也方便未来扩展类别，而不需要为所有交叉组合都收集大量样本。

`geometry-modified` 必须在 pilot 后冻结尺度归一化阈值、最小面积变化、配准容差和 annotation-jitter 规则；否则所谓状态变化会再次退化成建筑边缘的小幅修补。

### 3.3 必须区分“观测状态”和“因果解释”

双时相影像通常只能可靠支持：

- appeared / disappeared；
- persisted；
- geometry-modified；
- semantic `A → B`；
- unknown/ambiguous。

它通常**不能仅凭两张图证明**：

- “重建”：至少需要三期以上的消失—再出现过程、可靠身份或外部档案；
- “损毁”：需要灾害上下文和 xBD 一类专门 damage label，不能把 disappeared 自动叫 destroyed；
- “施工原因”“自然原因”：需要额外时序、SAR/光学/气象/地图或外部知识证据。

论文可以输出“replacement-like evidence”或“possible damage”，但不能把不可观测因果当作 GT 类别。**克制的 ontology 会增强而不是削弱创新可信度。**

`unknown/unobservable` 是拒答与可观测性标签，不应与四个普通状态混在同一个 macro-F1 中。它应通过 coverage、selective risk、拒答正确率和校准误差单独评价。

### 3.4 “漏检”不是状态类别，而是完整性失败

漏检必须有独立定义：在 audit-complete 场景中，某个满足查询谓词的 GT entity/event 没有被系统返回。

完整任务因此必须同时要求：

1. 发现 T1 与 T2 中的对象，包括 stable object；
2. 建立跨期身份或明确 unmatched/unknown；
3. 判断状态；
4. 对当前对象图无法解释的双时相证据继续搜索；
5. 只有当估计的残余漏检风险已校准到阈值以下，或继续取证的预期价值低于成本时，才允许停止。

跨期 association 应显式允许匹配到 `null/dummy` 节点：T1 object → `null` 表示消失候选，`null` → T2 object 表示新增候选；低置信或证据不足则进入 unknown，而不是被强行匹配。需要同时用单期 inventory recall 区分“真实 absent”与“感知模型漏了对象”。

这会从定义上修复“拿 change mask 去判断哪个从未进入 change mask 的对象漏了”的逻辑错误。

这里能承诺的是**经验完整性可评测和风险可校准**，不是数学保证“绝无漏检”。风险量也必须预先定义，例如：

```text
P(scene 中仍有至少一个 missed queried event | 当前证据)
```

或 expected missed-event count；二者不能在实验后按结果择优。

### 3.5 什么才算这里的“端到端”

不要求整个系统必须是一个不可解释的大模型。这里的端到端至少意味着：

- 双时相特征在对象发现前就联合交互，而不是两次独立分割后才匹配；
- 训练目标联合监督 paired masks、presence、association、state 和 rendered change；
- 任意真实 changed object 都存在从 RGB/feature 到对象假设的可学习入口，不依赖它先出现在当前 change mask；
- 最终状态错误可以反向约束前面的发现、配对与分割。

单纯训练 Qwen 输出坐标，仍然只是 action grounding；它可以借鉴 SegAgent，但不等于端到端状态变化检测，也不足以成为论文主创新。

---

## 4. 推荐的核心科学假设

论文最好围绕少数可证伪假设组织，而不是围绕工具清单组织。

### H1：联合 temporal entity 建模优于“独立分割 + 后匹配”

一个 joint temporal object model 同时预测实体的 T1/T2 masks、存在性、身份对应（包含显式 dummy/null）和状态。在相同 backbone、数据和算力下，它应在：

- changed-object recall；
- association accuracy；
- state macro-F1；
- 配准/外观扰动鲁棒性；

上显著优于当前独立分割匹配 baseline。若这一点不成立，整个“对象状态”主张缺乏基础。

### H2：独立的 temporal residual discovery 能恢复前置漏检

对象流之外，需要一条从 paired RGB/features 出发的“未解释时相残差”流。它不以当前 change mask 为候选来源，而是问：

> 当前 temporal object graph 还无法解释哪些局部证据？

它应显著提升 initial-miss recovery recall，并且不能靠规则网格穷举制造虚假 Agent 效果。由于 residual branch 很容易只是“第二个 change detector”，必须与容量/算力匹配的独立 detector、cascade 和 ensemble 对照，并分别测试自然漏检与人工注入漏检。

### H3：结构化状态证据比自由文本解释更可靠

每个状态结论必须绑定 paired masks/crops、身份关系、属性差异和置信度。自由文本由这些字段生成，只是用户界面。删除关键证据时结论应相应变化，扰动无关区域时结论应保持。

### H4：Agent 在预算受限的大场景中有不可约价值

Agent 的价值不应来自对 256×256 全图做三次固定 refinement，而应来自部分可观测、代价受限的主动取证，例如：

- 先看低分辨率全局图，再选择原生分辨率区域；
- 在对象分割、跨期匹配、状态专家之间选择；
- 对高风险 entity 做二次证据请求；
- 在事件召回与计算预算之间学习停止。

实验协议必须真的制造部分可观测性：初始只开放低分辨率全局观测，高分辨率局部影像或专项模态按区域、按成本解锁。若 full-resolution 全图从第一步就已输入，crop 主要是 adaptive computation；论文应诚实称预算化自适应计算，而不是主动获取新证据。

必须先做 action-oracle：在同等工具/算力预算下，oracle action 是否明显优于最强 one-shot 与 equal-compute iterative baseline。若 oracle 都没有 headroom，就应取消 Agent 主叙事。

### H5：因子化状态与 partial supervision 可以跨数据集学习

不同数据集只监督自己真实拥有的轴：SpaceNet 监督 footprint/identity，xBD 用 dataset-specific conditional ontology 监督 disaster damage，SECOND/Hi-UCD 监督 class transition，LEVIR/S2Looking 只监督最终 binary change projection。模型不能把缺失标签当负例。

这仍只是可检验假设：partial loss 不会自动消除 ontology、传感器、分辨率和 label-missing mechanism 的差异。change-only projection loss 必须依附于已有对象监督；若单独使用，模型仍可用许多退化的对象图解释同一 change mask。

---

## 5. Agent 应该是什么，而不应该是什么

### 5.1 一个成立的变化检测 Agent

一个可发表的 Agent 至少要有：

- **显式目标**：执行 typed state query，而不是泛泛“把 mask 修好”；
- **结构化 memory**：当前 temporal object-state graph、未解释残差、已检区域、拒绝历史、不确定性和剩余预算；
- **会改变信息状态的动作**：请求高分辨率 crop、提出新 entity、重分 paired masks、重新关联、调用状态专家、验证/弃权/停止；
- **outcome-aware policy**：根据动作后事件正确性与成本学习，而不是 checklist 硬编码工具；
- **面向状态假设的 Verifier**：验证 `(paired object, association, transition, evidence)`，不是只看 change-mask delta；
- **可校准停止**：基于残余漏检风险、预期信息增益和预算，而不是固定三步。

### 5.2 不足以支撑 Agent 论文的做法

- LLM 根据固定规则在几个工具间路由；
- checklist 输出后由程序直接映射 point/box；
- 让模型自由生成坐标，但动作并不改变它可见的信息；
- 把同一网络反复运行三次称为 planning；
- 只与较弱 one-shot baseline 比，不做等算力 iterative baseline；
- 只报最终 IoU，不评价轨迹正确性、成本与有害动作。

### 5.3 checklist 与 SegAgent 应怎样保留

- checklist 可以变成状态 ontology、证据合同和训练/评测 schema；它不是模型能力本身。
- SegAgent 式坐标/交互轨迹可用于 action grounding head 与 oracle imitation；“能输出点/框”不是核心贡献。
- 当前 SAM3/SimpleClick、rollback、trajectory、GT firewall、hash/receipt 可以保留为可靠执行与审计基础设施。

---

## 6. 数据是否够：当前不够；公共数据可分别验证组件，但不能直接拼成统一监督

### 6.1 先纠正对 SpaceNet 7 的认识

[SpaceNet 7 / MUDS](https://arxiv.org/abs/2102.04420) 不是“标注数量很小”：它包含 24 个月、100+ 地理区域、超过 1,100 万个 building polygon annotations，并为建筑提供跨时间唯一 ID。这恰好是现有公开数据中最接近对象身份/生命周期监督的资源。

它真正的不足是：

- 独立地理区域只有约 101 个，不能把同一 AOI 的月份或切片当成独立泛化样本；
- 4 m 影像与高精度 polygon 存在 label-fidelity/resolution 问题；
- 类别基本限于 building；
- 多地理、跨传感器和精细损伤状态仍不足。

因此正确定位是：**对象身份和多时相状态预训练 anchor，而不是唯一 test benchmark。**

尤其要注意其 `omniscient labeling`：标注精度可以高于 4 m 影像真正可见的细节。标注真值不等于视觉可判定真值；用于状态解释时必须做 observability 筛选，否则会用影像中不可见的标签惩罚模型，又声称结果具有视觉证据。

### 6.2 候选数据资源与可监督内容

| 数据 | 已公开的关键监督 | 最适合用途 | 不能直接支持什么 |
|---|---|---|---|
| [SpaceNet 7 / MUDS](https://arxiv.org/abs/2102.04420) | 24 月影像、exhaustive building footprints、跨月 ID，11M+ annotations | entity tracking、出现/消失、长期状态、对象级预训练 | 单独支撑跨地理/跨类别结论；精细 damage |
| [FOTBCD, 2026](https://arxiv.org/abs/2601.22596) | 约 28,000 对、28 个法国省份、0.2 m binary change；另有数千对 instance subset | 建筑跨地域泛化、instance schema 候选 | 摘要未证明完整 T1/T2 identity/状态，使用前必须审计 schema 和许可 |
| WHU Building Change | 两期大幅航空影像及逐期 building annotations | T1/T2 building segmentation、跨期派生状态 | 地理范围窄；身份/因果状态需重新构造和审计 |
| [xBD](https://arxiv.org/abs/1911.09296) | pre/post imagery、building polygons、damage ordinal labels，850,736 annotations | damage 专项状态、灾害域 | [官方 baseline](https://github.com/DIUx-xView/xView2_baseline) 说明 post polygon 由 pre footprint 转置，并非独立 T2 footprint GT；不适合监督新建/消失或精确 T2 实例分割 |
| [DynamicEarthNet](https://arxiv.org/abs/2203.12560) | 75 AOIs、daily multispectral、monthly 7-class semantic masks | 多时相语义状态、时序预训练 | 细粒度实例 identity；地理数量仍有限 |
| [Hi-UCD](https://arxiv.org/abs/2011.03247) / SECOND | 多时相或双时相语义标签、`from-to` land-cover states | 多类别 semantic transition | 实例级 entity lineage 和完整对象图 |
| [LEVIR-CD](https://doi.org/10.3390/rs12101662) / [S2Looking](https://arxiv.org/abs/2107.09244) | LEVIR-CD 有 637 对、31,333 个 building-change instances；S2Looking 有 5,000 对和 65,920+ change instances；均以 change 标注为主 | binary projection 监督、跨域 change 测试 | 唯一恢复 T1/T2 objects、stable objects、identity 和状态方向 |
| ChangeChat/Delta-QA/JL1-CC&QA 等 | captions、QA、instruction、部分 mask | 语言接口和 query paraphrase | audit-complete object graph；自动生成文本不能替代真实状态 GT |

结论是：

- **足够分别验证 tracking、damage、semantic transition、binary projection 和语言接口等组件；**
- **尚不能证明这些异构数据可以直接联合训练一个统一 object-state model；**
- **不够直接训练并严格评测一个跨类别、带身份与完整性评价的状态 Agent；**
- 要发表完整路线，几乎不可避免地需要一个规模不必极大、但标注定义严格的新 gold anchor/benchmark。

仓库现场截至本次审计只落地了 LEVIR-CD，没有已准备好的 paired T1/T2 object-mask 或多类别状态数据，因此数据路线实际上还没有开始。

### 6.3 建议的四层数据体系

1. **Gold anchor**  
   每个 test scene 完整标注 T1/T2 objects、stable objects、association、核心状态、unknown/occlusion 与配准质量。测试标签必须人工审核。

2. **强但部分监督的公共数据**  
   SpaceNet 7 提供 building identity；xBD 通过灾害条件专属 head/ontology 提供 damage；SECOND/Hi-UCD/DynamicEarthNet 提供 semantic transition。使用 masked/partial losses，不伪造不存在的标签，同时显式处理 ontology 与 domain mismatch。

3. **change-only 弱监督**  
   LEVIR-CD、S2Looking、FOTBCD-Binary 只作为依附于对象监督的 projection auxiliary loss，约束由状态图渲染出的 binary change；它们不能单独唯一指定对象分解、配对或状态方向。

4. **单时相与无标注时序**  
   用于 object prior、跨时相一致性、配准鲁棒性和自监督预训练。可借鉴 [ChangeStar/STAR](https://arxiv.org/abs/2108.07002)、[ChangeStar2](https://arxiv.org/abs/2406.15694)、[Self-Pair](https://arxiv.org/abs/2212.10236) 与 [Changen2](https://arxiv.org/abs/2406.17998) 的单时相/合成监督思想。合成数据只用于预训练、增广或压力测试，不能替代真实 held-out benchmark。

### 6.4 新 benchmark 不需要先追求“百万规模”

第一步先构建 200–300 对 audit-complete pilot，用于回答：

- 核心状态的自然频率与长尾程度；
- 两位标注者对 association/state/unknown 的一致性；
- 配准误差对 geometry-modified 的影响；
- 单场景完整标注的时间与成本；
- action oracle 的理论收益。

然后以**独立事件数和 AOI 数**做 power analysis 决定正式规模，而不是按 tile 数拍脑袋。一个 scene graph 可以程序生成很多 typed queries，数据杠杆远高于为每个问题单独写自由文本答案。

### 6.5 扩展到 building 之外必须使用 typed entities

不同类别的“对象”结构不同：

- building、ship、vehicle 是 instance-like；
- forest、water、cropland 是 areal-region-like；
- road、river 是 network-like，关键属性是连通性与拓扑，而非单个闭合 mask。

因此不要宣称一个 building instance matcher 能自然扩展到所有类别。共享的是 presence/state/query 框架；geometry、identity 与评价应由 entity type 决定。**第一篇论文建议只做 building，优先完成跨 AOI、跨国家和跨传感器泛化；**象征性加入一个 areal 类别不如把核心任务做扎实。跨类别与道路网络留到后续工作。

---

## 7. 建议的学习路线：先模型，再证明 Agent headroom

### 阶段 A：联合对象状态感知

训练一个真正双时相联合的 temporal entity set predictor。高层目标是同时预测：

- paired T1/T2 masks or geometry；
- presence；
- association/lineage；
- category/attributes；
- transition state 与 uncertainty；
- 可渲染的 binary/semantic change。

这一阶段不需要 LLM 成为核心。先证明任务可学、对象状态优于独立分割匹配。

### 阶段 B：多源 partial supervision

- full graph 数据监督全部可用字段；
- change-only 数据只监督 rendered change consistency；
- single-date 数据监督 object/category prior；
- xBD 只监督 damage axis；
- semantic datasets 只监督 class-from/to；
- unlabeled sequences 做 temporal correspondence 与 invariance/change disentanglement。

所有数据集必须按 AOI/geography/time 做隔离，不能把相邻 tile、同一对象或同一 SpaceNet AOI 跨 split。

### 阶段 C：构造真实的 Agent 错误状态

从训练集 GT 图出发，系统化制造：

- missing entity；
- wrong association；
- wrong state；
- incomplete/over mask；
- split/merge confusion；
- registration ambiguity；
- false confident stop。

然后让 oracle 在**只使用训练时 GT 计算 outcome、但不把 GT 放入模型输入**的条件下选择动作。保留正、负、无效和有害动作，不只保存 winner。

### 阶段 D：Agent policy/value/stop 学习

优先顺序：

1. oracle imitation；
2. pairwise candidate preference/value learning；
3. calibrated stop/miss-risk learning；
4. 只有 imitation 已显示 headroom 且 reward 可验证时，再考虑 RL。

RL 不是“更像 Agent”的必要条件。一个有明确状态、动作、观测变化、记忆和成本目标的 imitation-trained policy，远比只有自然语言思维链的固定循环更可信。

---

## 8. 评测闭环必须覆盖“发现—配对—状态—解释—成本”

### 8.1 主要指标

**对象与状态：**

- paired-object AP/PQ 或 track-aware matching quality；
- association/lineage accuracy，以及适用时的 SCOT、IDF1/HOTA 或清晰定义的 match F1；
- appeared/disappeared/modified/unchanged 四状态 macro-F1；unknown/unobservable 另报 coverage、selective risk 与拒答正确率；
- query-conditioned entity/event precision、recall、AP；
- T1/T2 mask IoU 与 boundary quality。

匹配阈值、event unit、query universe、split/merge 记分方式必须预先固定。需要加入全场景 no-change negatives 和 duplicate-event penalty，防止系统靠重复提出同一对象提高表面召回。

**漏检完整性：**

- initial changed-object recall；
- miss recovery recall；
- final changed-object recall；
- 未解释 residual 的覆盖与误报；
- false-stop rate，尤其是仍有漏检时错误停止的比例。

**Agent：**

- success/recall–cost curve 与 AUC；
- steps、crop pixels、tool calls、latency、GPU-hour；
- harmful-action rate、rollback rate、cycle rate；
- learned policy 相对 action oracle 的 gap closure；
- selective risk、confidence calibration 与 abstention coverage。

**兼容旧任务：**

- binary change IoU/F1；
- semantic `from-to` 指标。

旧像素指标保留，但不再是唯一主指标。

统计显著性应以 AOI/城市/灾害事件为独立单元做 bootstrap confidence intervals，而不是把同一 AOI 的成千上万个对象当作独立样本制造虚假显著性。

### 8.2 可解释性不能只评自由文本

至少做三类 grounding/fidelity 测试：

1. **证据删除**：移除决定状态的 T1/T2 object evidence，结论应降低置信或改变；
2. **无关扰动**：改变与目标 entity 无关区域，结论应保持；
3. **trace consistency**：action、返回证据、图状态更新和最终文字必须一致。

文本 BLEU/CIDEr 可以作为界面质量指标，不能证明状态推理正确。

### 8.3 鲁棒性必须单列

- 配准偏移；
- 阴影、季节、光照和传感器差异；
- 云/遮挡与低分辨率；
- 地理域迁移；
- 稠密小目标、split/merge、边界接触；
- 查询组合的 seen/unseen split。

---

## 9. 必做基线与消融：专门回答“为什么需要 Agent”

### 9.1 基线阶梯

1. 当前 independent segmentation + overlap matching + correction loop；
2. 强端到端 pixel CD，例如 ChangeStar2/ChangeFormer/ChangeMamba 类方法；
3. semantic/open-vocabulary CD；
4. joint temporal object-state model，**不带 Agent**；
5. one-pass query-conditioned object-state model；
6. 先构建完整图、再做 query post-filter 的数据库式 baseline；
7. equal-compute fixed iterative refinement；
8. 容量/算力匹配的 independent detector、cascade 与 ensemble；
9. SpaceNet tracker/SCOT + 状态分类，及 FootprintNet + 后验 query filtering；
10. joint state model + 固定预算 crop；
11. 完整 Agent；
12. proposal/state/action oracle upper bounds。

若完整 Agent 只赢早期弱基线，却不赢最邻近组合与等算力基线，审稿人会合理地认为提升来自更强 backbone、第二检测器或更多算力，而不是 agentic decision。若 query-conditioned perception 不优于“先构图、再过滤”，自然语言 query 应降级为交互层，不再作为方法贡献。

### 9.2 关键消融

- independent masks vs joint temporal entity queries；
- 无 temporal residual discovery；
- Verifier 只看 change mask vs 验证完整 entity-state hypothesis；
- hard-coded geometry vs learned grounding；
- fixed/shuffled policy vs learned policy；
- 无 memory；
- fixed steps vs learned stop；
- full-resolution one-shot vs budget-matched active crops；
- 低分辨率初始观测 + 按成本解锁高分辨率，vs 一开始可见全分辨率；
- 不同 gold data 比例、无 weak supervision、无 synthetic corruption；
- seen predicate vs held-out category × state composition；
- 无 uncertainty/unknown 类。

### 9.3 三道 Agent 放行门

1. **Oracle headroom gate**：同预算 oracle action 明显优于最强 one-shot/equal-compute baseline；
2. **Complementarity gate**：不同动作在不同错误状态上确实互补，而不是一个工具全面占优；
3. **Policy gate**：learned policy 能稳定关闭相当一部分 oracle gap，且有害动作和 false-stop 可校准。

阈值应在 200–300 对 pilot 后预注册。若任一门失败，停止 Agent 扩张，转投 object-state model/benchmark。

---

## 10. 最小可行论文路线

### 10.1 第一阶段只承诺四个可观测 building 状态，加一个拒答标签

```text
unchanged
appeared
disappeared
geometry-modified
```

`unknown / unobservable` 作为独立 abstention 标签，不计入普通四状态 macro-F1。

暂不把 reconstruction、destroyed、任意多类别因果推理放进主 claim。damage 可用 xBD 做独立的条件轴/外部实验；第一篇先完成 building 的跨域证据，多类别只保留接口与后续路线，不用象征性小实验支撑“通用”主张。

### 10.2 推荐贡献组合必须条件化，不能一篇论文堆满模块

“新 benchmark + joint model + residual detector + query + Agent + explanation”实际上接近两到三篇论文的工作量。更可靠的组织是：第一篇由贡献 1–2 构成闭环；只有 oracle headroom、真实部分可观测性和最邻近组合基线三项都通过，贡献 3 才升级为同一篇论文的主贡献，否则留作下一篇。

**贡献 1：新任务与 gold benchmark**  
可查询、audit-complete 的 object-state change task；有 paired objects、identity、states、unknown 与 AOI-disjoint split。

**贡献 2：joint temporal entity model + residual miss discovery**  
直接学习对象—属性—状态的联合演变，并显式搜索当前图未解释的变化证据。

**贡献 3：completeness-aware StateGraph-Agent**  
在大幅多分辨率场景中，以事件召回/状态正确性和成本为目标主动取证、验证、重关联与停止。

三者是一条候选因果链，不是天然创新组合：没有 1 无法训练和评价；没有 2 仍是单图后匹配；没有 3 则应诚实称 object-state model，而非 Agent。即便三者都有，也必须胜过对应模块的最邻近组合基线。

### 10.3 更稳与更激进的投稿路径

**路径 A：更稳，优先推荐**

- 先完成 object-state benchmark + joint temporal model + miss discovery；
- 目标偏 TGRS / ISPRS Journal of Photogrammetry and Remote Sensing 等遥感主流期刊；
- Agent 只作为有 headroom 的扩展，不影响主体成立。

**路径 B：高风险、高上限**

- 新任务/benchmark、强 joint model、真实 UHR budgeted active perception、完整等算力对照全部到位；
- 才尝试 CVPR / ICCV / ECCV 或强多模态会议；
- 必须证明方法意义不只局限于一个建筑数据集，并开源足以复现的 benchmark/code。

**当前路线若不 pivot**

- 最多适合写成安全、可审计的 GT-free iterative mask correction 系统或应用论文；
- 不应声称端到端状态理解 Agent，论文上限会明显较低。

---

## 11. 建议的近期决策顺序

### 第 0 阶段：先停止错误投入

- 暂停大规模当前格式 SFT；
- 暂停把 router/gate/horizon 五分支当作论文主创新；
- 保留现有工具、rollback、轨迹和审计系统；
- 冻结一份当前方法作为 baseline，避免边做新任务边不断移动旧基线。

### 第 1 阶段：两周级任务与数据可行性审计

1. 冻结四个可观测状态、unknown/abstention 规则和 paired-entity annotation protocol；
2. 审计 SpaceNet 7、FOTBCD-Instances、WHU、xBD、SECOND/Hi-UCD/DynamicEarthNet 的许可、schema、identity、配准与 split；
3. 做小规模人工双标，测 agreement；
4. 明确哪些状态能从标签可靠派生，哪些必须人工判断；
5. 设计 event-level 与 miss-recovery 指标。

### 第 2 阶段：先做两个最便宜的 Go/No-Go 实验

1. **Joint-model headroom**：joint temporal model 是否优于 independent segmentation + matching；
2. **Action-oracle headroom**：在固定预算下，主动区域/重关联/重分割 oracle 是否优于 one-shot/equal-compute iterative。

这两个实验比先训练一个会输出坐标的通用模型更重要。它们分别决定“对象状态路线是否成立”和“Agent 是否必要”。

### 第 3 阶段：只有过门后再扩大数据和 Agent

- 建正式 gold benchmark；
- 做 partial supervision；
- 训练 residual discovery；
- 由训练集 GT 构造 error-state trajectories；
- 训练 planner/value/stop；
- 完成 AOI-disjoint、cross-dataset、equal-compute 和 calibration 实验。

---

## 12. Go / No-Go 表

| 检查点 | Go 条件 | No-Go 后动作 |
|---|---|---|
| 状态可标注性 | 核心状态具有高一致性；unknown 可覆盖模糊样本 | 缩减 ontology，只保留可观测状态 |
| 数据完整性 | test scenes 对 stable/changed objects 均 audit-complete | 不做“漏检恢复”主张，只做局部状态分类 |
| 查询贡献 | query-conditioned perception 优于“先完整构图、再过滤”，或显著节省同等正确率成本 | 将语言 query 降级为交互/检索接口 |
| Joint model | 对 event/state/association 明显优于独立分割匹配 | 停止 Agent，先解决 temporal representation |
| Residual discovery | 真正提升前置 miss recall，且 FP 可控 | 不再把 Verifier 改名为漏检模块 |
| 部分可观测性 | 动作确实解锁此前不可见的高分辨率/新模态证据 | 收缩为预算化 adaptive computation，不声称主动取证 |
| Action oracle | 对最邻近组合与等预算基线有稳定、显著 headroom | 取消 Agent 主叙事，发 model/benchmark |
| Learned policy | 能关闭 oracle gap，harm/false-stop 可校准 | 保留 oracle analysis，不包装 learned Agent |
| 泛化 | AOI-disjoint 与外部数据上仍成立 | 收缩 claim 为单域方法或补数据 |
| 解释 fidelity | 状态结论对关键证据敏感、对无关扰动稳定 | 不声称可解释，只报告结构化输出 |

---

## 13. 最终建议

### 应继续保留

- 真实分割工具和可回放执行；
- rollback、安全门与 GT firewall；
- trajectory、receipt、hash 与审计意识；
- checklist 作为状态 schema/证据合同；
- SegAgent 作为 action grounding 参考。

### 应降级为 baseline 或工程层

- 单图 building segmentation；
- overlap instance matching；
- change-mask 驱动 proposal；
- 人工/规则坐标；
- 只会修轮廓的 point/box loop；
- 只用 change IoU/F1 的最终评价。

### 应成为新的论文核心

1. **任务**：queryable, audit-complete object-state change understanding；
2. **表示**：temporal object-state graph，而非仅 change mask；
3. **漏检**：独立 residual discovery 与显式 miss-risk；
4. **模型**：joint temporal entity learning，而非独立分割后匹配；
5. **Agent**：围绕状态假设主动取证、重关联、验证和停止；
6. **证据**：event/state/association/completeness/cost/calibration 全套评测。

**最诚实也最有机会的路线，是先证明 object-state task 和 joint model 成立，再用 oracle 证明 Agent 有不可替代的 headroom。不要先决定“必须是 Agent”，再寻找能让循环看起来像 Agent 的实现。**

---

## 参考资料（精选）

### 当前项目文档

- [`README.md`](../../../README.md)
- [`doc/pipeline.md`](../../pipeline.md)
- [`0803 Qwen3-VL oracle trajectory reachability`](../0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md)
- [`0804 Qwen3-VL 工具上限、Proposal 与数据集路线`](../0804_fifith_try/0804.md)
- [`0805 系统动作路由、Verifier、停止策略与效率`](./0805.md)
- [`0805 五分支并行开发说明`](./0805_parallel_branches.md)

### 任务、模型与 Agent

- Liu et al., [Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis](https://doi.org/10.1109/TGRS.2024.3425815), TGRS 2024.
- Wu et al., [ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning](https://arxiv.org/abs/2409.08582), 2024.
- [Show Me What and Where Has Changed? / VisTA and QAG-360K](https://arxiv.org/abs/2410.23828), 2024.
- [TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](https://arxiv.org/abs/2410.06234), ICLR 2025.
- [ChangeMinds: Multi-task Framework for Detecting and Describing Changes in Remote Sensing](https://arxiv.org/abs/2410.10047), 2024.
- [Semantic-CD: Remote Sensing Image Semantic Change Detection towards Open-vocabulary Setting](https://arxiv.org/abs/2501.06808), 2025.
- [DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception](https://arxiv.org/abs/2507.22346), 2025.
- [Referring Change Detection in Remote Sensing Imagery](https://arxiv.org/abs/2512.11719), 2025.
- [OmniOVCD: Streamlining Open-Vocabulary Change Detection with SAM 3](https://arxiv.org/abs/2601.13895), 2026.
- [Decoding the Delta: Unifying Remote Sensing Change Detection and Understanding with Multimodal Large Language Models](https://arxiv.org/abs/2604.14044), 2026.
- [ChangeQuery: Advancing Remote Sensing Change Analysis ... from Visual Detection to Semantic Understanding](https://arxiv.org/abs/2604.22333), 2026.
- [SMART-HC-VQA](https://arxiv.org/abs/2605.10739), 2026.
- [GeoVista: Visually Grounded Active Perception for Ultra-High-Resolution Remote Sensing Images](https://arxiv.org/abs/2605.14475), 2026.
- [FootprintNet: State-Transition-Guided Dynamic Footprint Learning for Multi-temporal Remote Sensing Change Detection](https://arxiv.org/abs/2607.27969), 2026.
- [Think with Extra-Image: A Farmland Segmentation Agent Driven by Spatio-Temporal Information Gain](https://arxiv.org/abs/2607.28186), 2026.
- [Agentic AI for Remote Sensing: Technical Challenges and Research Directions](https://arxiv.org/abs/2604.24919), 2026.

### 数据与监督

- Van Etten et al., [The Multi-Temporal Urban Development SpaceNet Dataset](https://arxiv.org/abs/2102.04420), 2021.
- Shen et al., [S2Looking: A Satellite Side-Looking Dataset for Building Change Detection](https://arxiv.org/abs/2107.09244), 2021.
- Gupta et al., [xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://arxiv.org/abs/1911.09296), 2019.
- Toker et al., [DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation](https://arxiv.org/abs/2203.12560), 2022.
- Tian et al., [Hi-UCD: A Large-scale Dataset for Urban Semantic Change Detection](https://arxiv.org/abs/2011.03247), 2020/2024 revision.
- [FOTBCD: A Large-Scale Building Change Detection Benchmark from French Orthophotos and Topographic Data](https://arxiv.org/abs/2601.22596), 2026.
- Zheng et al., [Change is Everywhere / ChangeStar](https://arxiv.org/abs/2108.07002), 2021.
- Zheng et al., [Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection / ChangeStar2](https://arxiv.org/abs/2406.15694), 2024.
- Seo et al., [Self-Pair: Synthesizing Changes from Single Source for Object Change Detection](https://arxiv.org/abs/2212.10236), 2022.
- Zheng et al., [Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model](https://arxiv.org/abs/2406.17998), TPAMI 2024.
