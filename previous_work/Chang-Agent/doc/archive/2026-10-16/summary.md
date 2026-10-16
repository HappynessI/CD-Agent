# 变化检测迭代优化系统总结

本文基于现已统一到 `Chang-Agent` 的本地 Qwen 分支代码及 0723—0804 实践记录整理。主线是：**先得到双时相建筑分割和初始变化图，再由视觉语言模型判断错误、规划工具动作、复核候选，循环更新变化图。** 下述以实际使用的 staged Proposal 闭环为主；历史商业模型实验、本地模型训练和离线工具诊断分别说明。

## 1. 整体 pipeline

```text
输入：LEVIR-CD 双时相 RGB 图像 T1/T2 + 文本查询 building
  ↓
变化检测：SAM3 分别分割 T1/T2 建筑
          → OmniOVCD 风格实例重叠匹配 → 初始二值变化图 C0
  ↓
迭代优化：Qwen Verifier 读取图像、预测 mask、区域证据和历史
          → 诊断错误并规划时相、区域及点/框动作
          → SimpleClick 点修正 / SAM3 框分割
          → 更新目标时相 mask，重新匹配并生成候选变化图
          → 运行时约束检查 + Verifier 复核候选
             ├─ 接受：提交状态，重新生成区域并进入下一轮
             └─ 拒绝：回滚，保留此前状态
  ↓  满足结束条件、动作预算耗尽，或无法继续有效执行
输出：selected 二值变化图 + 双时相中间 mask + 完整轨迹和停止原因
  ↓  正常推理全部结束后才读取 GT
离线评价：IoU、Precision、Recall、F1 及 initial/selected 对比
```

这里的 verifier 同时负责**诊断、动作规划和候选评价**。实际主运行路径没有另一套独立 Agent policy；Environment 管理状态与动作区域，Executor 调用分割工具。所谓“迭代优化”主要是推理时反复修正 mask，循环中不更新模型权重。

## 2. 输入数据与模型型号

| 环节 | 数据 / 模型 | 实际用途与路径 |
| --- | --- | --- |
| 输入数据 | **LEVIR-CD，256×256 双时相建筑变化切片** | 常用诊断集：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256`；`A/<sample>.png` 为 T1，`B/<sample>.png` 为 T2；查询为 `building` |
| 离线标签 | LEVIR-CD 二值变化 GT | 同一数据根目录的 `label_cvt/`；正常闭环不读取。该标签并非两侧建筑语义分割 GT |
| 初始分割模型 | **SAM3** | `/guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt`；对两个时相分别执行文本提示分割，也用于后续框修正 |
| 变化检测组合 | **SAM3 + OmniOVCD 风格实例匹配** | `change_agent/adapters/omniovcd_adapter.py` 的 `MaskPairProcessor`；匹配部分为确定性算法，没有额外的变化检测网络权重 |
| 点修正模型 | **SimpleClick，ViT-Large，COCO/LVIS 权重** | `/guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth`；通过 SegAgent 工具适配层执行正点/负点交互分割 |
| 本地 verifier | **Qwen3-VL-2B-Instruct** | `/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`；实际本地 staged 基线及后续 SFT 均围绕这一 2B 基座开展 |
| 商业 verifier / controller | **Qwen3.7：`qwen3.7-plus-2026-05-26`**；对照 **`qwen3-vl-plus`** | 通过百炼调用；Qwen3.7 比较 thinking 与 non-thinking，早期较好的五样本结果来自 non-thinking，不能归到本地 2B 模型名下 |

固定五个开发诊断样本是 `test_20_15`、`test_78_13`、`test_85_16`、`test_1_1`、`test_50_8`。这些样本被反复用于调试，不能视为完整 LEVIR-CD 的泛化评测。后续工具可达性 pilot 另用了 train/val 切片。

## 3. 各阶段具体处理

### 3.1 输入 → 初始变化检测

入口为 `tools/run_levir_change_agent.py`。每个样本重新调用 SAM3，分别得到 T1/T2 建筑预测 mask，并记录初始化的置信信息与提示配置；该入口不接受缓存 mask 代替初始化。

随后提取两侧 mask 的连通域作为实例，默认使用 `overlap_presence`、阈值 `0.25`。对每个实例，计算它与另一时相实例的交集占自身面积比例；达到阈值则认为另一时相存在对应目标。两侧未匹配实例的并集构成变化图。因而变化图来自**实例存在性比较**，不是简单地将两个语义 mask 逐像素异或。

### 3.2 初始变化图 → Verifier 诊断与规划

Environment 从预测状态建立带编号的候选区域（Proposal），提供 T1/T2 RGB、建筑预测 mask、变化 mask、区域位置及历史。Verifier 判断目标是否存在、是否真的变化，以及是否存在误报、漏检、边界、空洞或碎片问题。

早期 v11 系列包含全局筛查与区域 atomic audit；本地 v12.1 精简为按 Environment 顺序全量审计，移除不减少覆盖范围的模型 select。模型保留双时相 RGB 状态、可观察证据和七项 checklist；运行时派生 mask 状态、错误类型、目标时相和动作几何等字段。

Proposal 路径由 Environment 确定执行区域和点框位置，减少模型自行给坐标的不稳定性。双时相初始 mask 都为空时，增加 `3×3` 网格扫描以寻找漏检区域；扫描存在也不保证模型能识别小目标。

### 3.3 Verifier → 工具修正 → 候选复核

- **点动作**：SimpleClick 根据目标图像、当前 mask 和点击历史执行正点/负点修正。已验证的负点设置使用完整已接受会话 mask 作为模型输入，但只允许写入 Proposal ROI 内被点击的当前连通域。
- **框动作**：SAM3 根据指定框对目标时相进行分割，由 Executor 合并为候选状态。
- **重建变化图**：每次工具执行后重新提取实例、跨时相匹配，得到候选变化图。实际修改的是时相分割状态，变化结果通过匹配间接更新。
- **提交或回滚**：先检查修改范围、面积及连通域变化等运行时约束；通过后，Verifier 根据真实新增/删除区域及对应 RGB 判断候选效果。只有满足提交条件的候选进入已接受状态链；接受后重新审计，拒绝则回滚。

实验通常每个样本最多执行 3 步，采用 `conservative_best` 选择正式输出。该选择依赖 verifier 判断，**不保证 GT 指标单调提升**。停止原因要区分 verifier 允许结束、预算耗尽、输出无效和无可执行动作。

### 3.4 输出与评价

每次实验保存到独立 `outputs/<实验名>/`，主要产物为：

| 产物 | 含义 |
| --- | --- |
| `predictions/selected/<sample>.png` | 正式选定的最终二值变化图 |
| `predictions/initial/`、`verifier_best/`、`last/` | 初始、verifier 最佳记录与最后尝试状态；最后尝试可能被拒绝 |
| `trajectories/<sample>/trajectory.json` | 诊断、动作、工具结果、接受/拒绝与状态转换 |
| `trajectories/<sample>/episode_summary.json` | 所选步骤、停止原因和 episode 结果 |
| `masks/`、`tool_runs/`、`verifier_feedback/` | 中间 mask、初始化/工具证据和反馈 |
| `per_sample_metrics.json` | rollout 完成后读取 GT 得到的离线指标 |

正常推理在写出 `rollout_complete.json` 后才读取标签。汇总指标先累加各样本 TP/FP/FN/TN 再计算，是 micro 汇总；最终结果应取 `selected`，不能直接取 `last`。

## 4. 实践进展与结果边界

| 实践阶段 | 已观察到的结果 | 可以支持的结论 |
| --- | --- | --- |
| CA_0723(5)，商业 Qwen3.7 non-thinking，v11，五样本 | 9 次动作、7 接受、2 拒绝；micro IoU `0.67629040 → 0.70009134`；没有样本以 verifier 授权 finish 结束 | 存在有效的局部闭环修正；尚未证明收敛或泛化 |
| CA_0730，本地未训练 2B，v11.1，五样本 | 无有效工具动作，最终 IoU 保持 `0.67629040` | 运行与评分链路跑通，但本地 verifier 的有效决策能力不足 |
| CA_0801，本地 2B，v12.1 协议对照，五样本 | 相对 v11.3，调用数 `95 → 65`，有效区域审计 `0/45 → 37/45`；7 个候选全被拒绝，最终指标仍不变 | 精简协议改善了格式有效性和效率，尚未带来检测收益；其中也拒绝了真实有益候选 |
| 0801 后续本地 SFT | 多轮少量数据训练改善部分协议和动作可执行性，也出现大框误覆盖、错误删除；后续若干轮由回滚保住初态，但未观察到 selected 提升 | 训练 loss、格式正确率和工具调用成功率不能代替变化检测效果 |
| 0802 终局复核 | 在三个诊断样本上增加一次独立上下文复核，selected mask 和指标不变 | 额外复核本身未解决已有语义错误 |
| 0803—0804 离线工具可达性诊断 | 0803 多步 pilot 的 4 个需修正 episode 均未到预定终点；0804 GT-direct 完成 3/4 样本，三个完成样本均未到终点 | 需要同时检查候选区域、工具、匹配与限制条件；0804 不完整，不能报告完整四样本结论 |

后两次工具 pilot 不调用 Qwen：0803 通过离线标签评价候选，0804 进一步使用 GT teacher 直接生成点框。它们用于诊断有限动作空间和工具能力，不属于正常 GT-free 部署流程，也不能作为训练后 verifier 成功的证据。

## 5. 代码与实践中出现的问题

1. **Verifier 语义判断与候选收益判断不可靠。** 本地 2B 会把有误报或漏检的状态判断为可结束，也会拒绝有益修正。商业模型也有误接受：CA_0730 live 三样本中，`test_20_15` 的一次删除被接受，但 IoU 从 `0.84638554` 降到 `0.81428825`。
2. **协议复杂度曾阻碍实际执行。** 早期重复字段、多阶段筛查和修复调用造成无效输出与高延迟；v12.1 明显改善协议有效性，但仍存在 RGB 状态、证据和 checklist 不一致，格式修复不能替代视觉理解。
3. **动作空间与变化图组合存在限制。** 点框工具直接优化单时相分割，未必改善跨时相匹配后的变化图；初始双空 mask 曾使 Proposal 无法发现漏检。有限步数、候选位置及修改限制也可能挡住有益动作。离线 pilot 暴露了这些问题，但不能据此断言更大动作空间或更长轨迹无解。
4. **少量 SFT 数据未稳定转化为闭环收益。** 模型可能学会输出合规字段，却误判该删哪一块、是否应接受候选；历史 select 专项训练也不能代替完整闭环训练与评价。
5. **初始化和运行条件影响结果可比性。** fresh SAM3 初态与固定回放初态不同，控制器又会选择不同轨迹；因此历史回放和 live 运行的最终分数差不能全部归因于某一项代码改动。
6. **工程维护曾出现路径、超时和序列化问题。** 旧仓库路径引发 `sam3` / `third_party` 导入失败，SimpleClick 冷启动超过旧 300 秒超时；后续还有轨迹自引用、动作序列化与验收器不一致等问题。应按具体记录区分工程故障与模型/方法失败。当前 README、默认配置及历史文档也跨多个协议版本，理解实验时应以对应启动脚本和结果记录为准。

## 6. 主要依据

- 实际入口：[run_levir_change_agent.py](../tools/run_levir_change_agent.py)；匹配：[omniovcd_adapter.py](../change_agent/adapters/omniovcd_adapter.py)；执行器：[executor.py](../change_agent/executor.py)；分阶段 verifier：[staged_verifier.py](../change_agent/adapters/staged_verifier.py)。
- 历史商业闭环：[pipeline.md](pipeline.md)、[SimpleClick live 验证](CA_0730_SIMPLECLICK_LIVE_VALIDATION.md)。
- 本地模型：[训练前基线](training_pipeline/QWEN3VL_BASE_E2E5_0730.md)、[v12.1 协议对照](training_pipeline/QWEN3VL_LOCAL_VERIFIER_FREEZE_0731.md)、[SFT 实践](training_pipeline/0801_second_try/baseline_0801.md)、[终局复核结果](training_pipeline/0802_third_try/QWEN3VL_P1_RETRY3_RESULT_0802.md)。
- 工具诊断：[0803 多步可达性](training_pipeline/0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md)、[0804 GT-direct 记录](training_pipeline/0804_fifith_try/0804.md)。
