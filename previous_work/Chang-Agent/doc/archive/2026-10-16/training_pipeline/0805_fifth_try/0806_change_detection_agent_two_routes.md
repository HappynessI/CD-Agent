# 变化检测 × Agent 双路线研究决策：可核验状态转移与预算化主动取证

> 日期：2026-08-06  
> 性质：论文选题与研究路线决策稿，不是实现 freeze、训练授权或实验预注册  
> 承接文档：[0805_state_change_agent_research_route.md](./0805_state_change_agent_research_route.md)  
> 目标：快速读懂相关论文的领域、方法和贡献；明确“状态转移与可解释性”以及“Agent 主流方向”两条路线如何形成真正的科研问题。

## 0. 执行结论

### 0.1 不应再研究“怎样让 Agent 把建筑 mask 修得更准”

当前代码的主要链路仍是：

~~~text
T1/T2 独立建筑分割
→ 连通实例与 overlap matching
→ 生成 change mask
→ Verifier 诊断已有候选
→ SAM3 / SimpleClick 局部修补
→ commit 或 rollback
~~~

这个系统可以成为可靠的工程基础设施和强 baseline，但它没有形成新的变化检测科学问题。分割、实例匹配、point/box 路由、候选 gate、缓存和停止步数都只能回答“如何执行”，不能回答“系统究竟理解了什么变化”。

更合适的统一母问题是：

> **在配准误差、尺度差异、遮挡和观测预算受限的条件下，模型能否形成可辨识、可证伪、可拒答的对象状态转移假设，并取得足够证据，使变化事件既完整又可信？**

两条路线不是平行关系：

~~~text
路线一：状态转移表示与证据核验
回答“系统应该推理什么、解释什么”
                  ↓ 作为 world model / memory / reward
路线二：预算化主动取证 Agent
回答“系统下一步应该看哪里、取什么证据、何时停止”
~~~

### 0.2 推荐优先级

| 路线 | 真正的核心问题 | 相对可发表性 | 风险 | 建议 |
| --- | --- | ---: | ---: | --- |
| 路线一：证据约束的对象状态转移 | 如何把真实状态变化与观测扰动分离，并给出可反事实核验的证据 | 高 | 中 | **第一优先，先形成独立论文闭环** |
| 路线二：预算化时相证据 Agent | 如何在大幅、多分辨率、多时相数据中以有限成本找到并验证稀疏变化 | 高上限 | 高 | 先做 action-oracle，过门后再做 |
| 过渡路线：无 GT 状态转移审计器 | 如何判断候选变化事件、方向和证据是否可信 | 中等、开发较快 | 中高，容易被认为是 MQA 的领域迁移 | 可做路线一的低成本 pilot，不宜单独包装成通用 Agent |
| 现路线：交互式 mask 修补 | 如何选择 point/box 并反复改善 mask | 低 | 低工程风险、高论文风险 | 降级为 baseline 与执行层 |

### 0.3 最值得押注的第一篇论文

建议把第一篇论文收缩为：

> **Evidence-Grounded Object-State Change Detection：用显式状态转移瓶颈和反事实证据约束，提升变化事件识别的鲁棒性、完整性与置信度校准。**

第一篇不把自然语言聊天、通用工具调用、RL、多 Agent、跨所有地物类别同时列为贡献。先回答三个可证伪问题：

1. 显式状态转移瓶颈是否比 endpoint feature difference 更能区分真实变化和配准、光照、季节等伪变化？
2. 证据约束是否真的反向改善检测，而不是在最终 mask 后生成一段解释？
3. 模型能否对不可观测或不可辨识样本正确拒答，并校准仍有漏检的风险？

只有在预算匹配的 action oracle 明显优于最强 one-shot、fixed scan 和 equal-compute iterative baseline 后，才把 Agent 升级为论文主贡献。

---

## 1. 原路线稿的核心贡献：一分钟速读

### 1.1 领域

原路线稿位于三个方向的交叉处：

- 遥感建筑变化检测；
- 对象级、多时相状态理解；
- 多模态 Agent 的主动感知、工具调用和可验证推理。

它试图把项目从“GT-free segmentation correction loop”升级为“能够查询和解释对象状态变化的 Agent”。

### 1.2 它识别的核心问题

原路线稿最重要的判断不是某个新网络结构，而是发现当前任务定义本身有问题：

1. T1/T2 独立分割再匹配并不等于联合理解对象身份和状态演变；
2. 只审计已有 change proposal 无法恢复从未进入候选集的漏检对象；
3. LEVIR-CD 的单张 binary change mask 不能唯一监督 T1/T2 对象、状态方向和身份；
4. “能聊天、能 zoom、能调用 SAM、能输出 point/box”在 2026 年都不能单独构成创新；
5. Agent 是否必要，必须由等预算 action-oracle 先证明。

### 1.3 它提出的方法框架

原稿提出了三个工作概念：

- **Q-OSCU**：Queryable Object-State Change Understanding；
- **TOSG**：Temporal Object-State Graph；
- **StateGraph-Agent**：围绕对象状态图进行取证、验证、重关联与停止的 Agent。

模型输出不再只是一张 change mask，而是对象的逐期存在性、mask/geometry、跨期关系、状态、证据、不确定性和拒答原因。change mask 只是状态图的一个投影。

### 1.4 它的核心贡献

原稿可以概括为四项研究路线贡献：

1. **任务级 pivot**：把局部 mask 修补降级为 baseline，把对象状态变化理解设为主任务；
2. **结构化表示**：以 temporal entity/state graph 替代单一 binary mask；
3. **完整性意识**：引入未解释残差、事件召回和残余漏检风险；
4. **Agent 放行门**：要求 oracle headroom、动作互补性和 learned-policy gap closure。

### 1.5 原稿仍需收缩的地方

“新 benchmark + joint model + residual detector + query + Agent + explanation”接近两到三篇论文的体量。原稿把正确的部件都列了出来，但第一篇论文的唯一核心问题仍不够尖锐。

本稿进一步收缩为：

> **路线一先解决状态假设的可辨识性和解释 fidelity；路线二再解决有限预算下的证据获取。**

---

## 2. 本地参考论文：领域、方法、贡献与选题约束

本节对 reference 目录中的八篇真实论文 PDF 做全文级速读。另有两个单页 PDF 是 SegAgent 的可视化插图，不是独立论文。

### 2.1 八篇论文核心贡献

| 论文 | 领域与问题 | 方法 | 核心贡献 | 对本项目的直接含义 |
| --- | --- | --- | --- | --- |
| [Change-Agent](../../../../reference/Change-Agent.pdf), TGRS 2024 | 交互式遥感变化解释；检测只有位置、caption 缺少像素定位 | 双分支 MCI 模型联合 change detection 与 caption；BI³ 做双时相局部/全局交互；LLM 编排工具 | Change-Agent 系统、LEVIR-MCI 数据集、多层次变化解释模型 | “LLM + 检测器 + caption + 计数/问答”已经被直接覆盖；聊天和工具编排不能再当主创新 |
| [Forest-Chat](../../../../reference/Forest-Chat.pdf), 2026 | 森林变化的交互式检测与解释、跨域适配 | supervised MCI，加 AnyChange 与 GPT-4o 的 zero-shot 检测/caption/refinement，LLM 负责 orchestration | Forest-Chat；Forest-Change、LEVIR-MCI-Trees、JL1-CD-Trees；森林域适配与迁移分析 | 简单把 Change-Agent 换到新地物类别也已有先例；其 prompt 驱动执行和域泛化限制恰好说明需要结构化状态与系统评测 |
| [Towards Comprehensive Interactive Change Understanding / ChangeIMTI and ChangeVG](<../../../../reference/Towards Comprehensive Interactive Change Understanding in Remote Sensing- A Large-scale Dataset and Dual-granularity Enhanced VLM.pdf>), 2025 | 变化 caption、分类、计数和定位的统一交互理解 | 89,628 条多任务 instruction；fine-grained 与 global 双分支视觉引导特征作为 Qwen2.5-VL 辅助 prompt | ChangeIMTI 数据集、ChangeVG 模型、四任务与 multi-turn 统一评测 | “多任务指令微调 + 双粒度视觉特征 + 多轮问答”已经拥挤；它不是会主动改变观测的 Agent |
| [SegAgent](../../../../Chang-Agent/third_party/SegAgent/SegAgent.pdf), CVPR 2025 | MLLM 的像素级理解和交互分割 | 把点击式标注建成 MDP；自动生成 annotation trajectory；SFT、StaR+、PRM-guided tree search | HLMAT 新任务、SegAgent、HRES 数据和视觉中心多步决策协议 | 让 MLLM 输出正负点、模仿人工修 mask、用轨迹训练已不是新问题 |
| [SAM-Veteran](../../../../reference/SAM-Veteran/SAM-veteran.pdf), ICLR 2026 | reasoning segmentation 中的人类式 SAM 使用 | MLLM 先生成 box，再生成 refinement points，并自适应停止；多任务 GRPO 与 dynamic sampling | 首个统一 box、迭代修补和 adaptive termination 的 SAM Agent；跨域 SOTA | 当前 point/box router、迭代 mask refinement、RL 和停止规则几乎都被正面覆盖；继续沿此方向只能做领域应用 |
| [Audit After Segmentation / MQA-RefAVS](../../../../reference/MQA-RefAVS/MQA.pdf), 2026 | 无 GT 条件下的多模态 mask 质量评估 | 给定视频、音频、文本和候选 mask，预测 IoU、六类错误和 accept/revise/reject；SFT 一个专用 auditor | MQA-RefAVS 新任务、26,061 个 mask 的 MQ-RAVSBench、MQ-Auditor | “无 GT Verifier 判断 mask 好坏并建议动作”也已经成为独立赛道；本项目若做 auditor，必须审计变化状态和时相证据，而不是照搬 mask taxonomy |
| [LensWalk](../../../../reference/LensWalk.pdf), 2026 | 长视频中的 agentic active observation 和 test-time scaling | reason-plan-observe；Agent 控制时域、采样密度；Scan Search、Segment Focus、Stitched Verify；timestamp 与 entity memory | 不微调即可在多个长视频 benchmark 提升 5% 以上；系统评测准确率、工具互补性和 token/帧成本 | 这是最值得迁移的 Agent 范式：动作决定“看什么、看多细”，不是只对固定输入重复推理 |
| [FM-ChangeNet](../../../../reference/google.pdf), 2026 | 把变化检测从静态端点比较改写为特征空间连续 transport | 对 T0/T1 特征做中间插值，学习 time-conditioned velocity field；多尺度 flow、alignment、trajectory consistency 和 segmentation loss | pathwise supervision；velocity magnitude 作为定位和解释 cue；报告多数据集 SOTA | “状态转移、连续演化、可解释变化场”这组措辞已被占据；但其路径是构造的 latent interpolation，速度目标本质上来自端点差，不是物理对象状态、真实中间观测或可反事实证据 |

### 2.2 两个容易误读的本地资源

- [SegAgent/images/segact-demo.pdf](../../../../Chang-Agent/third_party/SegAgent/images/segact-demo.pdf) 和 [mask-quality-comapre.pdf](../../../../Chang-Agent/third_party/SegAgent/images/mask-quality-comapre.pdf) 都是单页可视化，不是新论文。
- reference 下名为 [OmniOVCD](../../../../Chang-Agent/third_party/OmniOVCD/README.md) 的目录，当前内容实际是 facebookresearch/sam3 的仓库与《SAM 3: Segment Anything with Concepts》说明，而不是 OmniOVCD 论文。正式 related work 中不能把这个本地路径当成 OmniOVCD 的全文证据。

### 2.3 这些论文共同说明了什么

2024 年的“Agent”主要是 LLM 对固定视觉模型和 Python 工具做 orchestration。2025–2026 年的前沿已经明显转向：

1. **主动观察**：模型根据当前假设决定下一次看哪里、以什么尺度看；
2. **test-time scaling**：把推理预算分配给困难样本，而不是固定多跑几轮；
3. **可学习决策**：通过 imitation、preference/value、PRM 或 RL 学动作与停止；
4. **专用 verifier/auditor**：无 GT 评估结果质量并驱动纠错；
5. **结构化 memory**：持续记录实体、证据、冲突、已搜索范围和剩余风险。

本项目最适合切入第 1、2、4、5 点；第 3 点是训练手段，不应被写成科学问题本身。

---

## 3. 原路线稿所引论文：竞争版图

下表综合原路线稿列出的代表工作。对没有本地 PDF 的 2025–2026 论文，当前结论来自原稿记录的公开元数据与任务摘要；正式投稿前必须重新获取全文，核验任务定义、数据 split、指标和 claim。

| 研究簇 | 代表工作 | 已被覆盖的贡献 | 仍可能存在的空白 |
| --- | --- | --- | --- |
| 对话式变化解释 | Change-Agent、ChangeChat、Forest-Chat、ChangeGPT | change detection/caption、问答、计数、原因分析、LLM 工具编排 | 解释是否忠于证据、是否覆盖全部事件、Agent 是否真正改变观测 |
| 指令与 query grounding | VisTA/QAG-360K、TEOChat、ChangeIMTI/ChangeVG、DeltaVLM、Referring CD、ReasonCD | 多任务 instruction、文本条件定位、answer + mask、多轮交互 | query 是否改变感知，还是只对已构建结果做过滤；组合状态 query 的可靠性 |
| 检测与语言统一 | ChangeMinds、Delta-LLaVA、JL1-CC&QA | mask、caption、VQA 和推理联合学习 | 自由文本的 factuality 和 explanation fidelity；事件级完整性 |
| SAM 与开放词汇变化 | AnyChange、OmniOVCD、ReA-OVCD、SAM3 | training-free latent matching、point/text prompt、开放词汇实例分割 | 分割不再是稀缺能力；跨时相身份、状态方向和证据核验仍未自动解决 |
| 状态与动态过程 | Bi-SRNet、SCanNet、FootprintNet、FM-ChangeNet、SpaceNet 7 | from-to semantic transition、显式 state-transition constraint、feature transport、跨月 building ID | 可辨识的对象状态、unknown/observability、反事实解释和残余漏检风险 |
| 遥感主动感知 | ZoomEarth、GeoEyes、GeoVista、GeoLens、FarmSeeker | UHR zoom、区域搜索、多工具、信息增益、学习停止 | 成对时相动作、变化假设证伪、事件完整性和 miss-risk stopping |
| 视觉决策与审计 | SegAgent、SAM-Veteran、MQA-RefAVS | click/box、mask refinement、adaptive stop、GRPO、无 GT mask audit | 从“mask 是否好”升级为“状态转移是否由时相证据支持” |

### 3.1 已经不能再声称的创新

- 首次把 LLM/Agent 用于遥感变化检测；
- 首次让用户通过自然语言查询变化；
- 首次联合 change mask 和 caption；
- 首次让 Agent 输出 point/box 并多步修 mask；
- 首次使用 RL 或 adaptive stopping 做交互分割；
- 首次使用 zoom、工具调用或 verifier；
- 首次提出 state transition 或可解释 change field；
- 首次做建筑跨期身份或 construction/demolition tracking。

### 3.2 仍有机会形成强 claim 的交叉空白

更可防守的增量不是术语拼接，而是四个可以严格评测的性质：

1. **Identifiability-aware**：承认同一个 change mask 对应多种对象分解和状态方向，显式表示 unknown；
2. **Evidence-faithful**：状态结论绑定最小充分的双时相证据，并通过干预验证；
3. **Completeness-calibrated**：评价是否仍有未解释、未返回的变化事件；
4. **Budget-aware active verification**：Agent 为具体状态假设主动取得新证据，并报告成本。

---

## 4. 为什么当前代码走错了科研路线

### 4.1 当前证据只能支持工程结论

- production 路径由 Verifier 规划、Executor 确定性执行，没有独立 learned Agent policy；
- 0803 非初始达标 episode 的 endpoint reachability 为 0/4；
- 0803/0804 的相关 teacher 运行没有调用 Qwen，不能证明模型理解状态或学会规划；
- 0805 的 1776 个候选和 30 个选中动作只证明真实工具有局部修改能力；
- LEVIR-CD 只有 change GT，无法唯一判断 T1 还是 T2 的对象 mask 应怎样修改。

因此，当前代码可以声称：

> “这是一个可审计、可回滚的 GT-free 双时相分割修补框架。”

不能声称：

> “这是一个理解对象状态转移并主动发现变化的 Agent。”

### 4.2 工程组件与科研贡献的边界

| 当前组件 | 正确定位 | 为什么不是核心创新 |
| --- | --- | --- |
| SAM3 / SimpleClick | 感知与执行工具 | 通用分割能力已有成熟基础模型 |
| overlap matching | baseline / 数据接口 | SpaceNet tracking、对象 CD 和大量匹配方法已覆盖 |
| point/box router | 执行策略组件 | SegAgent、SAM-Veteran 已直接研究 |
| mixed_error decomposition | 协议与安全工程 | 改善 actionability，但没有新任务或新认知变量 |
| Verifier gate / rollback | 可靠系统基础 | MQA 等已将无 GT 审计单独建模 |
| cache、常驻模型、batch | 实验基础设施 | 影响吞吐，不改变科学问题 |
| fixed/adaptive horizon | Agent 评测组件 | 只有与成本、信息增益和任务成功联合建模时才有研究价值 |
| 状态转移假设 | 候选核心表示 | 必须可监督、可拒答并反向约束检测 |
| 证据 fidelity 与 miss risk | 候选核心贡献 | 直接回答“为何相信变化、是否仍漏了变化” |

### 4.3 应怎样重写最初直觉

原始直觉是：

> 状态转移和解释也许能帮助变化建筑物分割得更准。

这个表述仍把“分割更准”设为科研终点。更有论文价值的表述是：

> **endpoint difference 对真实状态变化和观测扰动存在系统性混淆；显式、可核验的状态转移瓶颈能否解除这种混淆，并因此改善事件识别、鲁棒性、拒答和最终 mask？**

最终 mask 的提升是验证该科学假设的一项结果，而不是创新定义。

---

## 5. 统一任务定义

### 5.1 输入、假设与证据

给定双时相或多时相观测：

~~~text
I1, I2, ..., IK
传感器、分辨率、时间、配准质量等元数据
可选的目标类别或 typed state predicate
~~~

系统对每个候选实体输出：

~~~text
entity / entity type
presence@T1 ... presence@TK
cross-time identity or lineage
geometry / semantic attributes
transition hypothesis
observability and unknown reason
paired evidence regions
confidence and residual miss risk
~~~

第一篇论文建议只做 building，并只承诺：

~~~text
unchanged
appeared
disappeared
geometry-modified
unknown / unobservable
~~~

“destroyed、reconstructed、caused by disaster”等带因果含义的状态，只有在真实多期观测和专门标签支持时才能加入。

### 5.2 可解释性的操作性定义

本项目的解释不应是一段由 VLM 自由生成的文字，而应是一个 **transition evidence certificate**：

- 支持该实体存在或不存在的逐期区域；
- 支持 same/null/split/merge 的身份关系；
- 支持状态方向的视觉差异；
- 影响结论的配准、遮挡、云和分辨率因素；
- 可校准置信度与拒答原因。

解释 fidelity 至少通过四种干预评价：

1. **Sufficiency**：只保留证据区域，结论应基本保持；
2. **Comprehensiveness**：删除关键证据，状态置信度应显著下降；
3. **Temporal swap**：交换 T1/T2 后 appeared 与 disappeared 应互换；
4. **Nuisance invariance**：扰动无关区域、轻微配准和光照后，结论应稳定或正确降置信。

自由文本只负责把结构化 certificate 翻译给用户，不能作为解释正确性的主要证据。

---

## 6. 路线一：可反事实核验的对象状态转移

### 6.1 核心科学问题

传统 binary CD 把两张图压缩为一个 endpoint label。它没有显式回答：

- 哪个对象发生了变化；
- T1 与 T2 是否是同一实体；
- 状态方向是什么；
- 变化是否可能来自错配、阴影、季节或分辨率；
- 哪些证据真正决定了结论；
- 当前输出是否还漏掉了事件。

路线一研究：

> **显式状态转移瓶颈能否提供比 endpoint-only feature comparison 更可辨识、更鲁棒、更可校准的变化表示？**

### 6.2 三项可组成论文闭环的创新

#### 创新 A：可辨识、可拒答的状态转移任务

建立 audit-complete 的对象级标注协议，至少覆盖：

- T1/T2 实际存在的对象，包括 stable objects；
- appeared/disappeared 对应的 null association；
- same、split、merge、identity-unknown；
- geometry state；
- observability、registration quality 和 unknown reason。

贡献不在于“又做一个数据集”，而在于修复 binary change supervision 的不可辨识性，并提供事件级完整性评价。

#### 创新 B：证据约束的状态转移瓶颈

模型内部必须显式预测状态与证据，而不是先得到 mask 后用规则生成状态字段。可以采用：

~~~text
joint temporal encoder
→ entity / region queries
→ structured transition bottleneck
→ evidence certificate
→ state-conditioned change renderer
~~~

训练约束包括：

- state 与 rendered change 一致；
- presence 与 null association 一致；
- temporal swap equivariance；
- 证据 sufficiency / comprehensiveness；
- nuisance invariance；
- unknown 与 observability 一致。

若删除 state/evidence loss 后，检测、鲁棒性和校准均不下降，就不能声称“解释帮助了变化检测”。

#### 创新 C：解释完整性与残余漏检风险

在完整测试场景中估计：

~~~text
P(仍至少存在一个未解释变化事件 | 当前对象图与已观察证据)
~~~

这是“解释是否完整”的问题，不是一个普通第二检测器。必须与同容量 detector、cascade 和 ensemble 对照，并分别测试自然漏检与人工注入漏检。

为控制第一篇体量，创新 C 可以先只做 calibrated scene-level miss risk；复杂的 residual discovery 与主动搜索留给路线二。

### 6.3 与 FM-ChangeNet、FootprintNet 和 semantic CD 的差异

| 最近工作 | 它的“状态/转移” | 本路线必须提供的额外东西 |
| --- | --- | --- |
| FM-ChangeNet | T0/T1 feature 间构造连续 latent path，速度场幅值定位变化 | 真实可观测对象状态、身份、unknown、时间方向和干预式 evidence fidelity |
| FootprintNet | 根据原路线稿记录，使用显式 state-transition constraint 描述动态 footprint | 对象级证据 certificate、可辨识性、拒答、完整性和 Agent 接口 |
| semantic CD | 像素级 class-from → class-to | entity identity/lineage、观测条件、事件级完整性 |
| caption / VQA | 自由文本描述变化 | 可机器验证、对证据干预敏感的结构化解释 |

不能声称“首次建模状态转移”。更准确的 claim 是：

> 首次系统研究 **identifiability-aware、evidence-faithful、observability-calibrated object-state transition** 对变化检测的价值。

“首次”仍需在投稿前完成全文检索后才能保留。

### 6.4 数据路线

| 数据 | 可提供的监督 | 本路线中的角色 | 风险 |
| --- | --- | --- | --- |
| SpaceNet 7 / MUDS | 多月 building polygons 与跨月 ID | identity、出现/消失、多时相预训练 anchor | 4 m 影像、omniscient labels、独立 AOI 数有限 |
| WHU Building Change | T1/T2 building masks | paired mask、四状态构造与 pilot | 地理域窄，identity 需审计 |
| FOTBCD-Instances | 大规模建筑实例变化候选 | 跨地域外部验证 | 必须先核验实例 schema、许可和 split |
| SECOND / Hi-UCD | semantic from-to labels | 状态转移辅助监督 | 像素语义不等于实例 identity |
| xBD | pre/post building 与 damage label | 条件 damage 轴 | post footprint 标注协议不适合新建/消失主任务 |
| LEVIR-CD / S2Looking | binary change mask | rendered change 的 projection loss 与兼容评测 | 不能单独监督 T1/T2 objects、方向和 stable objects |

最先做的实验不是训练完整 Agent，而是：

1. 在 GT object crops / masks 上验证四状态是否视觉可判；
2. 以 200–300 对 audit-complete pilot 测双人一致性；
3. 单独报告不可观测样本比例；
4. 按 AOI、城市、传感器和时间隔离 split。

### 6.5 必做基线

1. 当前 independent segmentation + overlap matching；
2. 强 pixel CD：ChangeFormer、ChangeMamba、ChangeStar2 或同级方法；
3. FM-ChangeNet；
4. joint temporal entity model，不带 state/evidence constraint；
5. joint model + post-hoc state classifier；
6. joint model + state head，但无 fidelity loss；
7. 完整 state-transition evidence bottleneck；
8. 同容量 detector/cascade/ensemble；
9. proposal/state oracle。

### 6.6 主要指标

**主指标：**

- changed-event precision、recall、F1/AP；
- appeared/disappeared/modified/unchanged macro-F1；
- association / lineage F1；
- unknown coverage、selective risk、ECE/Brier；
- temporal-swap consistency；
- explanation sufficiency、comprehensiveness；
- registration、光照、季节、遮挡和分辨率鲁棒性；
- scene-level false-complete / residual-miss calibration。

**兼容指标：**

- binary change IoU/F1；
- T1/T2 mask IoU 与 boundary quality。

统计显著性以 AOI、城市或灾害事件为独立单元，而不是把同一 AOI 的对象都当独立样本。

### 6.7 Go / No-Go

| 检查点 | Go | No-Go 后动作 |
| --- | --- | --- |
| 状态可观测性 | GT crop 上四状态显著高于随机且 unknown 有稳定边界 | 缩减 ontology |
| 标注一致性 | identity/state/unknown 达到可接受双人一致性 | 不建 benchmark 主张 |
| 状态增益 | 同 backbone/算力下 event、鲁棒性或校准稳定改善 | 状态只保留为报告字段 |
| 解释 fidelity | 关键证据删除与无关扰动产生预期响应 | 不声称可解释 |
| 完整性 | miss-risk 能校准，固定 FP 下提高 miss recovery | 删除 completeness claim |
| 跨域 | AOI-disjoint 和外部数据仍成立 | 收缩为单域方法 |

### 6.8 投稿定位

- **更稳的版本**：新任务/标注协议 + state-transition bottleneck + fidelity/robustness，适合遥感主流期刊路线；
- **更强的视觉论文版本**：需要跨数据集、显著的 robustness/calibration 收益、公开 benchmark 和充分的最近工作对照；
- **不要做的版本**：只有一个 state head，最终仍只报告 LEVIR-CD IoU，解释只展示几张热力图。

---

## 7. Agent 主流热点排序

用户关心的不只是遥感 Agent，还包括近期通用 Agent 中容易与变化检测结合、相对好形成论文的方向。按“热点、任务契合、可证伪性、开发速度”综合排序如下。

| 方向 | 热点 | 与变化检测契合 | 开发速度 | 论文判断 |
| --- | ---: | ---: | ---: | --- |
| 主动视觉观察 / agentic perception | 5/5 | 5/5 | 3/5 | **最推荐；LensWalk 已证明范式，但要做时相特有创新** |
| test-time scaling / 动态预算与学习停止 | 5/5 | 5/5 | 3/5 | 与 UHR 稀疏变化天然契合，必须做等预算比较 |
| 专用 verifier / reward model / self-correction | 4/5 | 5/5 | 4/5 | 最快利用现有代码，但必须超越普通 mask audit |
| 多模态工具使用与 RL | 5/5 | 4/5 | 2/5 | GRPO 很热，但只是优化手段；reward 不可信时风险很高 |
| 结构化长期 memory / world model | 4/5 | 5/5 | 3/5 | 状态图可自然充当 Agent memory，适合作为路线一与二的桥 |
| 多 Agent 专家协作 | 4/5 | 2/5 | 3/5 | 容易变成多个 prompt 的工程编排，不建议优先 |
| RAG / 地理知识问答 | 3/5 | 3/5 | 4/5 | 容易做 demo，难证明改善视觉变化判断 |

最值得选择的组合是：

> **主动观察 + test-time budget allocation + 结构化状态 memory + calibrated verifier。**

RL、PRM 或 preference learning只在这个问题定义下作为策略学习手段出现。

---

## 8. 路线二：预算化时相证据获取 Agent

### 8.1 核心科学问题

大幅遥感影像中变化事件稀疏、尺度差异巨大，而全分辨率穷举成本高。双时相或多时相任务还要求在相同地理区域取得成对证据。

路线二研究：

> **Agent 能否依据不断更新的状态假设，在有限像素、token、工具调用或模态预算下，主动决定看哪里、看哪一期、看多细，并在完整性风险足够低时停止？**

这与当前 mask refinement 的根本区别是：动作必须改变 Agent 能获得的信息。

### 8.2 一个成立的部分可观测环境

初始只向 Agent 开放：

- 低分辨率全局 T1/T2 或多时相 overview；
- 影像元数据；
- 总预算；
- 可选的 typed change query。

高分辨率 crop、额外日期、SAR/光学专项证据或昂贵专家模型只能通过带成本的动作解锁。

如果第一步已经输入 full-resolution 全图，后续 crop 只能诚实称为 adaptive computation，不能称 active perception。

### 8.3 Agent 状态、动作与记忆

**结构化 memory：Transition Hypothesis Ledger**

~~~text
已搜索区域与尺度
候选实体及状态假设
支持/反对证据 ID
跨期身份与冲突
observability / uncertainty
未解释 residual
剩余预算
残余漏检风险
~~~

**最小动作集：**

- coarse scan：低成本搜索新线索；
- paired zoom：同步查看 T1/T2 的同一地理区域；
- asymmetric inspect：某一期因遮挡或尺度问题需要更细观察；
- retrieve time：请求中间时相以区分 disappeared、temporary occlusion 或 reconstruction-like sequence；
- call specialist：调用对象、配准、状态或云检测专家；
- verify / reject hypothesis；
- stop / abstain。

第一篇 Agent 论文不要一次加入所有模态。最小版本只保留 coarse scan、paired zoom、verify 和 stop。

### 8.4 三项候选创新

#### 创新 A：变化检测专用的 budgeted active-perception benchmark

benchmark 必须有：

- 真实大幅或可重建大场景，而不是把独立 256×256 patch 随机拼起来；
- 初始低分辨率观测；
- 按 crop 像素、token、工具调用和延迟计价；
- audit-complete event GT；
- paired temporal action；
- event recall–cost 与 false-stop 评价。

#### 创新 B：transition-aware evidence acquisition

Agent 不是泛化地“寻找有趣区域”，而是选择最能区分竞争状态假设的观测。例如：

~~~text
H1: building disappeared
H2: T2 被阴影/云遮挡
H3: 配准偏移导致假消失
~~~

下一动作应最大化区分 H1/H2/H3 的预期信息价值，而不是只选择 change score 最大的 crop。

#### 创新 C：completeness-aware stopping

停止条件联合：

- 当前已验证事件的价值；
- 仍有遗漏事件的风险；
- 下一次观测的预期收益；
- 剩余成本。

关键指标是“场景中仍有漏检时错误停止”的 false-stop rate，而不是平均只用了几步。

### 8.5 与 LensWalk、GeoVista 和 FarmSeeker 的差异

| 相关方向 | 已有能力 | 本路线必须额外证明 |
| --- | --- | --- |
| LensWalk | reason-plan-observe；动态时域和采样密度；Scan/Focus/Stitch | 成对时相空间动作、状态方向、事件完整性、遥感 UHR 成本 |
| GeoVista / GeoLens | UHR 区域搜索、视觉 grounding、工具调用 | 搜索目标是变化状态假设而非单图语义对象 |
| FarmSeeker | 时空信息增益、学习停止 | 从单类分割扩展到双时相竞争假设和 miss-risk calibration |
| SegAgent / SAM-Veteran | mask 状态上的点框编辑 | 动作获取新证据，而不是只改变当前 mask |
| 当前 Chang-Agent | Verifier 诊断、受控执行、rollback | learned observation policy、结构化 memory、预算曲线和等成本收益 |

最可防守的 claim 是：

> **state-hypothesis-driven paired temporal observation under a calibrated completeness objective**。

### 8.6 训练顺序

1. **Action-oracle upper bound**：训练集 GT 只用于离线评价动作结果，不进入 policy 输入；
2. **Oracle imitation**：学习何时 scan、zoom、verify、stop；
3. **Candidate value / preference learning**：比较动作的事件收益、风险和成本；
4. **Calibrated miss-risk / stop model**；
5. 只有 reward 稳定、oracle 有 headroom、imitation 可关闭部分 gap 时，再使用 GRPO 或其他 RL。

直接从 GRPO 开始风险很高：如果奖励仍是 change IoU，模型学到的只是更复杂的分割优化器。

### 8.7 必做基线

1. full-resolution one-shot；
2. uniform scan；
3. fixed coarse-to-fine scan；
4. uncertainty / change-score greedy；
5. equal-compute iterative refinement；
6. LensWalk-style generic prompt planner；
7. 路线一的 one-shot state model；
8. state model + fixed crop；
9. 完整 learned Agent；
10. random、shuffled、no-memory、fixed-stop ablation；
11. action oracle。

所有方法必须匹配 crop pixels、视觉 token、工具调用或 GPU-seconds中的至少一个主要预算，并同时报告其余成本。

### 8.8 主要指标

- event recall / state F1 @ cost；
- recall–cost、accuracy–cost 曲线与 AUC；
- false-stop rate；
- oracle gap closure；
- harmful action、cycle、rollback；
- 证据覆盖与重复观察率；
- confidence / miss-risk calibration；
- token、crop pixels、tool calls、latency、GPU-hours；
- AOI-disjoint 与跨传感器泛化。

### 8.9 Agent 路线的 Go / No-Go

| 检查点 | Go | No-Go |
| --- | --- | --- |
| 真实部分可观测性 | 动作解锁此前不可见的高分辨率或新时相证据 | 改称 adaptive computation |
| Oracle headroom | 同预算 oracle 稳定胜 one-shot、fixed scan 和 equal-compute | 取消 Agent 主叙事 |
| 动作互补性 | scan/zoom/verify 在不同错误上各有优势 | 精简为单一模型 |
| Learned policy | 稳定关闭明显一部分 oracle gap | 只报告 oracle analysis |
| 停止校准 | false-stop 和 miss risk 可控 | 不声称 completeness-aware |
| 成本收益 | 在相同 recall 下显著节省成本，或相同成本下提高 recall | Agent 没有实用或科学增量 |

---

## 9. 最快可发与最高上限：三种论文包装

### 9.1 推荐：状态转移主论文

**暂定题目：** Evidence-Grounded Object-State Change Detection under Observation Nuisance

**贡献组合：**

1. 可辨识、可拒答的对象状态变化任务与 pilot benchmark；
2. state-transition evidence bottleneck；
3. temporal swap、证据删除、无关扰动和 calibration 评测；
4. 在 event、robustness 和最终 mask 上验证状态约束的作用。

**优点：** 科学问题清晰，不依赖 Agent 一定有效；能直接区别于 Change-Agent、SegAgent 和 SAM-Veteran。  
**风险：** 数据标注与 FootprintNet/FM-ChangeNet 的全文对照必须扎实。

### 9.2 快速过渡：状态转移审计器

**暂定题目：** Reference-Free Auditing of Object-State Transitions in Bi-temporal Remote Sensing

输入候选 entity、T1/T2 证据和 transition，输出：

- transition valid / invalid / unknown；
- 错误来源：identity、direction、geometry、nuisance、unobservable；
- 证据 certificate；
- accept / reacquire / abstain。

可以复用当前 Verifier schema、region visualization、GT firewall、trajectory 和 rollback 基础设施。

**必须超越 MQA-RefAVS 的地方：**

- 审计的是跨时相状态假设，不是 mask morphology；
- 包含 temporal swap、错配、伪变化和 observability；
- 评价证据 fidelity 与事件级错误，而不只预测 IoU；
- hard negatives 来自真实 CD 模型错误与时相干预，不只来自形态学腐蚀。

**判断：** 开发最快，适合作为路线一 pilot；若只是把六类 mask error 换成遥感数据，论文新颖性不足。

### 9.3 高上限：主动时相证据 Agent

**暂定题目：** Budgeted Temporal Evidence Agents for Complete Change Discovery

**贡献组合：**

1. 大幅多时相部分可观测 benchmark；
2. transition-ledger memory；
3. hypothesis-driven paired zoom；
4. miss-risk calibrated stopping；
5. event recall–cost 和 equal-compute 评测。

**优点：** 同时连接 agentic perception、test-time scaling、遥感 UHR 和变化检测。  
**风险：** 数据、环境、公平成本计量和 learned-policy 训练都较重；没有 oracle headroom 时整篇不成立。

---

## 10. 建议的实施顺序

### 阶段 0：立即冻结错误投入

- 暂停把当前格式的大规模 SFT 当作论文主线；
- 暂停把 point/box router、mixed_error、Verifier threshold 和 horizon 当核心创新；
- 保留当前代码版本和结果作为 independent segmentation + correction baseline；
- 保留 SAM3/SimpleClick、rollback、receipt、hash、轨迹和 GT firewall。

### 阶段 1：两周可行性审计

1. 完成 FootprintNet、FM-ChangeNet、GeoVista、FarmSeeker、LensWalk、MQA、SAM-Veteran 的 claim matrix；
2. 冻结四状态、unknown 和 observability 定义；
3. 审计 SpaceNet 7、WHU、FOTBCD 的 schema、许可、分辨率和 split；
4. 标注 50–100 对双人 pilot；
5. 在 GT object crops 上验证 state observability；
6. 构造 temporal swap、misregistration、occlusion 和 hard identity negatives。

### 阶段 2：路线一最便宜的 Go/No-Go

1. 训练或提示一个 transition verifier；
2. 比较 endpoint baseline、post-hoc state 和 evidence-constrained state；
3. 做证据删除、无关扰动、时间交换；
4. 测 state F1、event recall、selective risk 和 calibration；
5. 若无明确增益，停止扩大 benchmark 和 Agent。

### 阶段 3：Agent oracle

1. 将真实大场景封装为 low-res overview + paid paired crops；
2. 定义统一成本；
3. 比较 uniform、greedy、fixed coarse-to-fine 和 action oracle；
4. 只有 oracle 有稳定 headroom，才生成 imitation/value/stop 数据。

### 阶段 4：选择论文分叉

- 状态路线已成立、Agent oracle 未过门：发表 state benchmark/model；
- 状态路线已成立、Agent oracle 过门：继续做主动证据 Agent；
- 状态模型较弱、审计器较强：收缩为 transition auditing 论文；
- 两者均无增量：停止 Agent 包装，回到强 CD model 或数据问题。

---

## 11. 审稿人会追问的十个问题

1. 状态是不是由最终 T1/T2 masks 确定性后处理得到的？
2. 与 FM-ChangeNet、FootprintNet 和 semantic from-to detection 的区别是什么？
3. 解释是否经过干预验证，还是一段看起来合理的 caption？
4. 同一个 binary mask 对应多种状态分解时，标签如何定义？
5. unknown 是真正的 observability 建模，还是吸收错误的垃圾类？
6. 漏检对象如果从未进入 proposal，系统如何发现？
7. Agent 的动作是否真的获得新信息？
8. Agent 是否胜过相同视觉 token、crop pixels 或 GPU 成本的固定策略？
9. 提升来自 Agent policy，还是来自多调用了一个强分割器？
10. 同一 AOI 的相邻 tile 是否泄漏到了不同 split？

如果论文设计不能在实验前回答这些问题，就不应启动大规模训练。

---

## 12. 建议的最终贡献表述

### 12.1 第一篇论文可以争取的表述

> 我们研究 endpoint-only change supervision 的状态不可辨识与解释不忠实问题，提出一个可拒答的对象状态转移任务和证据约束瓶颈。该表示将状态、身份、观测条件与最小双时相证据绑定，并通过 temporal swap、证据删除和 nuisance intervention 进行核验。实验评价事件识别、鲁棒性、校准与最终 change mask，而不是仅展示自由文本解释。

### 12.2 第二篇 Agent 论文可以争取的表述

> 我们研究大幅多时相影像中的预算化变化取证。Agent 维护结构化状态假设账本，选择成对时相观测来区分竞争假设，并依据残余漏检风险和预期观测价值停止。贡献由 event recall–cost、false-stop、oracle gap 和等预算比较验证。

### 12.3 不应再使用的表述

- “我们首次用 Agent 做变化检测”；
- “状态让模型更可解释”，但只有一张 heatmap；
- “Agent 学会了规划”，但动作由 Python checklist 固定映射；
- “主动感知”，但第一步已输入全分辨率图；
- “端到端状态理解”，但状态来自独立 mask 后处理；
- “完整变化发现”，但测试集没有 stable object 和 audit-complete event GT。

---

## 13. 参考资料

### 13.1 本地全文

- [Change-Agent](../../../../reference/Change-Agent.pdf)
- [Forest-Chat](../../../../reference/Forest-Chat.pdf)
- [LensWalk](../../../../reference/LensWalk.pdf)
- [FM-ChangeNet](../../../../reference/google.pdf)
- [ChangeIMTI / ChangeVG](<../../../../reference/Towards Comprehensive Interactive Change Understanding in Remote Sensing- A Large-scale Dataset and Dual-granularity Enhanced VLM.pdf>)
- [SegAgent](../../../../Chang-Agent/third_party/SegAgent/SegAgent.pdf)
- [SAM-Veteran](../../../../reference/SAM-Veteran/SAM-veteran.pdf)
- [MQA-RefAVS](../../../../reference/MQA-RefAVS/MQA.pdf)
- [SAM 3 local repository](../../../../Chang-Agent/third_party/OmniOVCD/README.md)

### 13.2 当前项目与原路线稿

- [0805 状态变化 Agent 研究路线](./0805_state_change_agent_research_route.md)
- [0805 系统动作路由、Verifier、停止与效率记录](./0805.md)
- [0805 五分支并行开发说明](./0805_parallel_branches.md)
- [Chang-Agent README](../../../README.md)

### 13.3 原路线稿中的代表外部工作

- Liu et al., Change-Agent, TGRS 2024.
- Wu et al., ChangeChat, 2024.
- VisTA / QAG-360K, 2024.
- TEOChat, ICLR 2025.
- ChangeMinds, 2024.
- Semantic-CD, 2025.
- DeltaVLM, 2025.
- Referring Change Detection, 2025.
- OmniOVCD, 2026.
- GeoVista, 2026.
- FootprintNet, 2026.
- Think with Extra-Image / FarmSeeker, 2026.
- Agentic AI for Remote Sensing: Technical Challenges and Research Directions, 2026.
- SpaceNet 7 / MUDS, 2021.
- xBD, 2019.
- DynamicEarthNet, 2022.
- Hi-UCD, 2020/2024 revision.
- ChangeStar2, 2024.

---

## 14. 最终建议

不要把两条路线理解为“状态模型”和“通用 Agent”二选一。更合理的科研结构是：

> **状态转移是 Agent 要维护和证伪的 world model；可解释性是状态结论与证据之间可干预验证的关系；Agent 则是在预算下为尚未解决的状态假设主动获取证据。**

短期先做路线一，因为即使 Agent oracle 失败，状态任务、模型和 fidelity benchmark 仍可独立成立。路线二只有在真实部分可观测环境和 action-oracle 都证明存在不可替代收益时再进入。

当前项目最有价值的存量不是 point/box 本身，而是已经形成的可审计执行、回滚、轨迹、receipt、GT firewall 和 Verifier 接口。把这些资产迁移到“状态假设—证据—风险”的新任务上，才可能从一个成熟的 mask 修补系统转成有明确科学问题的变化检测 Agent 项目。

---

## 15. 方案二动机核验更新：从动态选区到风险控制的主动时序取证

> 更新日期：2026-08-06  
> 本节记录对 LensWalk、遥感主动感知、非 Agent 粗到细方法以及候选数据集的进一步核验。若本节与前文“纯空间主动 zoom”表述冲突，以本节为准。LEVIR-CD 后续只适合作为代码 smoke test，不再承担论文动机或主实验。

### 15.1 核验结论

研究动机**存在，但必须改写**：

1. “大图、目标小、变化稀疏”只能证明需要节省计算，不能证明需要顺序 Agent；
2. “低分辨率扫描后动态选择 crop”通常可被 ROI、FPN、quadtree、一次性 top-K 或 coarse-to-fine cascade 替代；
3. 真正成立的 Agent 动机是：在长时间序列、不同分辨率和可选传感器构成的证据池中，前一步返回的新证据会改变下一步最有价值的日期、区域、模态、工具和停止决策；
4. 因而主轴应从“主动空间 zoom”改为**主动日期选择**，空间聚焦作为定位动作，SAR、配准和其他专家工具作为遇到歧义后的条件分支。

一句话判断：

> **变化检测中的 Agent 不应只是决定放大哪里，而应像诊断过程一样，根据刚获得的时序证据决定下一次最有价值的观测，并在残余漏检风险可控时停止。**

### 15.2 LensWalk 为什么能够发表

[LensWalk 已由 CVF 官方收录为 CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_LensWalk_Agentic_Video_Understanding_by_Planning_How_You_See_in_CVPR_2026_paper.html)，不能再仅写成 arXiv 预印本。它成立，不是因为“动态选择数据”这个动作本身，而是因为它构造了一个真实的闭环信息问题：长视频含有大量帧，模型只能观察很小一部分；关键证据稀疏，第一次观察后才知道下一步应该查看哪段视频、使用何种采样密度以及验证哪个新假设。

它的有效贡献组合包括：

- `Scan Search` 做宽范围线索发现；
- `Segment Focus` 对目标区间进行高密度事实提取；
- `Stitched Verify` 把多个不连续时段放入同一视觉上下文进行联合验证；
- Timestamp Anchor 和 Subject Memory 支持精确重访与跨轮实体一致性；
- 在多个模型组合和六类长视频/推理 benchmark 上验证，并做工具、记忆、成本和行为消融；
- 不需要额外训练即可作为 plug-and-play test-time observation framework 使用。

补充材料 Table 8 的 same-evidence 对照尤其重要：把 Agent 实际访问过的帧并成一次静态输入，只提升约 `0.8–2.6` 分；完整闭环提升约 `6–7` 分。这说明收益不只来自“选到好帧”，还来自逐轮改变子问题、观察范围、证据组合和验证策略。但该证据也不是无争议的：静态对照使用 256 帧预算，而完整 LensWalk 在 Video-MME Long 平均使用约 387 帧；Table 6 中 LensWalk 在线时间约 `190.35s`，单次 o3 约 `38.9s`。所以它主要节省全视频预处理和峰值上下文，不代表在线延迟必然更低。

### 15.3 为什么纯空间版本缺乏 Agent 必要性

已有非 Agent 工作已经覆盖“粗看全局、细看局部”：

- [AutoFocus](https://arxiv.org/abs/1812.01600) 用 coarse-to-fine FocusChips 减少高分辨率推理区域，实现约 `2.5×` 加速，并可用 `5×` 更少的 pyramid pixels 换取约 1% mAP 损失；
- [QueryDet](https://arxiv.org/abs/2103.09136) 先在低分辨率特征上预测小目标位置，再稀疏计算高分辨率特征，在 COCO/VisDrone 上报告约 `2.3–3.0×` 加速。

遥感主动感知本身也已拥挤：

- [GeoVista](https://arxiv.org/abs/2605.14475) 已覆盖全局规划、多区域检查和显式证据去重；
- [GeoLens](https://arxiv.org/abs/2607.25993) 已指出单一 zoom 在全局搜索、多区域比较、路径规划和分散证据任务上会饱和，并转向多工具视觉推理；
- [FarmSeeker](https://arxiv.org/abs/2607.28186) 已按信息增益查询额外时空影像并学习停止。

因此，“Agent 会 zoom”“Agent 会选择额外图像”“Agent 使用思考—行动—观察循环”均不能单独成为变化检测论文的核心创新。变化检测必须额外提供**竞争变化假设、结果条件化取证、残余漏检风险和可证伪停止**。

### 15.4 建议的问题定义与现实价值

建议任务名：**Budgeted Active Temporal Evidence Acquisition for Change Detection**。

> 给定两个端点时相的低成本全局观测、一个包含未标注中间日期/不同分辨率/可选模态的冻结证据池以及预算 `B`，Agent 能否顺序决定“何时、何地、用什么证据观察”，更完整、更校准地输出端点 change mask，并在残余漏检风险低于阈值时停止或拒答？

典型条件分支：

1. 端点影像显示某建筑疑似新增；
2. 中间日期显示建筑已经存在，则向更早日期搜索；
3. 中间日期仍无建筑，则向更晚日期搜索；
4. 光学影像被云遮挡，则换日期或调用 SAR；
5. 轮廓差异疑似由错位造成，则调用局部配准或成对验证；
6. 多份证据一致后修改 change mask；证据冲突则继续取证或送人工；
7. 只有当剩余 missed-change risk 足够低时才允许停止。

这解决的现实问题不是“怎样更聪明地裁图”，而是：

> **在城市地图更新、违建巡查、灾害评估等大范围监测中，当高分影像、额外日期、计算和人工审核均有成本时，下一份最值得读取、计算、购买或交给专家的变化证据是什么？**

[WorldStrat](https://arxiv.org/abs/2207.06418) 论文报告的案例显示，即使获得慈善折扣，购买覆盖 Darfur 的 VHR mosaic 仍约需 400 万美元。这只能支持“高分证据存在真实成本”的动机，不能替代变化检测 GT 或 Agent 效果实验。

### 15.5 更换 LEVIR-CD 后的数据路线

目前没有一个公开数据集同时满足“大幅面 UHR、长时序、多模态、完整变化 GT、顺序动作评测”。不能假装已有统一 benchmark。建议分轴验证，或新建小型 gold anchor。

| 数据 | 最适合验证什么 | 主要边界 |
| --- | --- | --- |
| [SpaceNet 7 / MUDS](https://arxiv.org/abs/2102.04420) | 建筑方向首选 pilot；约 24 个月、超过 100 个区域、超过 4 万 km²、1100 万以上 building polygons，并有跨月 unique ID | 4 m 中分辨率；标签精度高于可见影像，需做 observability 过滤；全读 24 月可能仍便宜 |
| [DynamicEarthNet](https://arxiv.org/abs/2203.12560) | 主动日期选择；75 个 AOI、逐日多光谱观测、逐月 7 类语义标签 | 非建筑实例任务；无 identity；不能声称精确日级事件时刻 |
| [DynamicVL](https://arxiv.org/abs/2505.21076) | 高分多时相城市理解；1 m NAIP、42 个城市、每场景 5–10 期，可做状态假设和日期选择外测 | 预裁 `1024×1024` patch；无 SAR；不等同于连续大幅面巡检 |
| [Delta-SN6](https://arxiv.org/abs/2604.05527) | 光学初判后是否按需调用 SAR | 只有双时相、单一区域，不能独立证明长时序策略 |
| [FOTBCD](https://arxiv.org/abs/2601.22596) | 法国大范围 0.2 m 建筑变化，可重建大 canvas 验证空间成本 | 公开 benchmark 已切成 `512×512` 且仅两期 RGB；单独使用仍容易退化为 adaptive computation |
| [WorldStrat](https://arxiv.org/abs/2207.06418) | 真实高/低分辨率获取成本和跨分辨率预训练 | 只有一个高分日期且没有变化 GT，不能作为主评测 |

数据决策：

1. **SpaceNet 7** 最贴近当前建筑主线，适合作为最快可证伪 pilot；换用它后，路线一的“完全没有对象档案 GT”瓶颈会部分解除，但仍需处理 4 m 可观测性和标签来源问题；
2. **DynamicEarthNet** 最适合验证日期选择是否存在真实 adaptivity；
3. **DynamicVL** 可补高分、多时相城市场景；
4. **Delta-SN6** 只承担 SAR fallback 外测；
5. 若要在一个统一分布中同时主张空间、时间和模态 Agent，需要新增小型独立 gold benchmark。

### 15.6 方案二不需要完整建筑状态档案 GT

若主任务仍限定为端点 binary change detection，所需独立监督可以只有：

```text
同一 AOI 的 T0/T1 endpoint change GT
```

中间日期、高分 crop、SAR 和其他专家返回都可作为无标签 evidence pool。训练阶段可从同一 Agent state 执行不同动作，用最终 endpoint GT 离线标注每个动作造成的：

- `ΔFN pixels`、`ΔFP pixels`；
- `ΔF1 / ΔIoU`；
- miss recovery；
- harmful action；
- risk reduction / cost。

这样可以评价最终 change mask、漏检恢复、伪变化减少、成本曲线和停止风险；但不能在没有相应 GT 时声称精确事件日期、对象身份、生命周期或变化原因。部署时 policy 只能读取影像、元数据、历史和工具结果，不能读取 GT。

### 15.7 三项候选创新与可核验解释

#### 创新 A：变化专用的竞争假设状态

Agent 不保存自由文本 CoT 作为解释，而维护结构化假设，例如：

```text
real change / registration error / shadow-seasonal change
/ cloud-occlusion / resolution-insufficient / unknown
```

每份新证据必须明确支持、反驳或无法判定哪些假设。

#### 创新 B：反事实 action value

从同一状态缓存多个候选日期、区域、模态和工具的实际结果，学习：

```text
expected residual-change-risk reduction / action cost
```

这比固定工具链或不确定性 top-K 更强，也允许评价 action regret、冗余调用和有害动作。

#### 创新 C：风险控制停止与证据证书

Agent 估计：

```text
P(场景中仍有至少一个 queried change event 被遗漏 | 当前证据)
```

或 expected remaining FN pixels。解释输出不是推理轨迹，而是可审计证书：准确日期、区域、传感器、工具 provenance；支持/反驳的假设；调用前后置信度；删除该证据后的结论变化；停止时的残余漏检风险。

### 15.8 证明“必须使用 Agent”的生死实验

核心量不是 selection gain，而是 **adaptivity gap**：

```text
G_adapt(B) = J(pi_adaptive, B) - J(pi_open_loop, B)
```

- `pi_open_loop` 只能依据初始 overview，一次性选完预算内的日期/区域/模态；
- `pi_adaptive` 可以依据前序动作真实返回的观测决定下一动作；
- 两者都不能读取测试样本 GT；另列 clairvoyant GT oracle 作为不可达上界；
- 在 AOI-disjoint 测试集上报告不同预算的 AOI-bootstrap confidence interval。

必做比较：

1. endpoint-only detector；
2. full-resolution / all-date / all-modal one-shot fusion；
3. uniform、random、固定日期和固定 coarse-to-fine；
4. learned/uncertainty top-K；
5. ROI、quadtree、QueryDet-style sparse refinement；
6. 将 Agent 最终访问的证据并成一次性输入的 same-evidence baseline；
7. open-loop planner 与 observation replay/shuffle；
8. 轻量非 LLM value-of-information controller；
9. sequential Agent；
10. optimal adaptive policy、optimal non-adaptive policy、exhaustive 与 GT oracle ceiling。

所有方法必须同时报告解锁像素、读取字节、视觉 token、FLOPs/GPU-seconds、tool calls 和串行 wall-clock。若 `G_adapt(B)` 接近零，或者 all-evidence fusion 在现实成本内始终更强，应取消 Agent 主叙事，诚实定位为 budgeted adaptive computation。

### 15.9 最新路线建议

方案二最值得继续的版本是：

> **以主动日期选择为主轴，用变化竞争假设决定空间聚焦和模态 fallback，以反事实 action value 选择下一证据，以校准残余漏检风险决定停止。**

进入大规模训练前，先在 SpaceNet 7 或 DynamicEarthNet 上缓存一个 action-outcome table，比较最佳 open-loop 策略、adaptive policy、exhaustive 和 GT ceiling。只有 adaptive policy 在多个预算和隔离 AOI 上稳定优于 open-loop，才进入 Qwen/VLM Agent、imitation 或 RL；否则停止 Agent 包装。即使闭环策略成立，还必须证明 Qwen/VLM Agent 胜过轻量 value controller，否则“需要顺序策略”仍不等于“需要 LLM Agent”。
