# 之前的工作：Chang-Agent 项目（2026-07 至 2026-08）

本目录是 Chang-Agent 项目的代码与文档快照（2026-10-16 工作目录状态），用来交代这次摸底实验之前做过什么、试过什么、遇到了什么问题。仓库根目录的 `results/` 是之后（2026-10-16）的摸底实验，两者的关系见文末。

快照的处理：

- **未包含**：实验产物 `outputs/`（约 5.6 GB）、第三方依赖 `third_party/`（OmniOVCD/SAM3、SegAgent/SimpleClick、LLaMA-Factory，说明见 `Chang-Agent/third_party_README.md`）、原始诊断日志 `diagnostics/`，以及集群内部规则文件 `AGENTS.md`。
- **为公开仓库做的替换**：API key 文件名统一改为 `bailian-apikey.csv`。文档、登记表和测试中的 SHA-256 等哈希值统一替换为 `<hash omitted>`（共 960 处），所以部分测试中的期望值不再是原值，这份代码只作阅读参考。
- 文档中的绝对路径（`/guisongxia01/...`）、旧实验目录名均为服务器上的原始记录，旧实验输出已清理。

## 1. 一句话说明

**用 SAM3 分别分割 T1、T2 的建筑 → 实例匹配得到初始变化图 → Qwen-VL 作为 Verifier 诊断错误并规划点/框动作 → SimpleClick / SAM3 修改单期 mask → 重新匹配，Verifier 复核候选，更好则提交、否则回滚。** 整个闭环不读 GT，结束后才离线评估。任务为 LEVIR-CD 建筑变化检测（256×256 切块）。

```text
T1/T2 + 文本 "building"
  → SAM3 双期建筑分割 → OmniOVCD 风格 overlap-presence 匹配 → 初始 change mask
  → Verifier（Qwen3-VL / 百炼 Qwen3.7）：区域审计、错误诊断、动作规划
  → Executor：SimpleClick 点修正 / SAM3 框分割（修改某一期的建筑 mask）
  → 重新匹配 → 候选 change mask → Verifier 复核 → 提交或回滚 → 下一轮
  → 预算耗尽或 Verifier 允许结束 → selected 结果 → 离线 IoU/F1
```

最好的入门材料：[`Chang-Agent/doc/archive/2026-10-16/summary.md`](Chang-Agent/doc/archive/2026-10-16/summary.md)（系统总结：pipeline、模型、各阶段做法、结果与问题），以及 [`Chang-Agent/doc/old_pipeline.md`](Chang-Agent/doc/old_pipeline.md)（完整 pipeline 入门，以 CA_0723(5) 为例）。

## 2. 代码结构

| 路径 | 内容 |
| --- | --- |
| `Chang-Agent/change_agent/` | 核心框架：`state.py`（状态与动作协议）、`environment.py`（区域、执行几何、提交/回滚）、`executor.py`（工具调用）、`verifier*.py`、`agent.py`、`runner.py`、`trajectory.py`；`adapters/` 下为 SAM3、OmniOVCD 匹配、SimpleClick、Qwen3-VL、百炼、staged/direct verifier 等适配层 |
| `Chang-Agent/tools/` | 运行入口与 Slurm 提交脚本。主入口 `run_levir_change_agent.py`；`submit_ca07xx_*` 为商业模型（百炼）实验，`*qwen3vl*` 为本地 Qwen3-VL 的数据准备、SFT、评测与诊断 |
| `Chang-Agent/configs/` | 运行配置与 SFT 配置 |
| `Chang-Agent/tests/` | 单元测试（协议、解析、验收器等） |
| `Chang-Agent/experiments/` | 实验登记表（`experiment_registry.json`，含执行状态、科学结论与重试谱系） |
| `Chang-Agent/doc/` | 当前文档入口 `README.md`、本次任务说明 `change_dection_agent.md`；历史文档全部在 `archive/2026-10-16/` |

## 3. 尝试过程与结果（按时间）

详细见 `doc/archive/2026-10-16/summary.md` 第 4、5 节和 `training_pipeline/` 各目录。

| 时间 | 做了什么 | 结果 | 文档 |
| --- | --- | --- | --- |
| 07-19 至 07-22 | 搭闭环原型；Verifier 协议多版迭代（direct / proposal / hybrid、rubric、SoM 等消融） | 发现闭环问题：Verifier 误判、协议过繁、动作不可执行等 | `closed_loop_issue_analysis_20260719.md`、`ca0721_*`、`ca0722_*`、`proposal_ablation.md` |
| 07-23 | **商业模型（百炼 Qwen3.7 non-thinking）+ v11 staged Proposal**，5 个固定样本 | 9 次动作、7 次接受、2 次拒绝，micro IoU 0.676 → 0.700；没有一个样本以 Verifier 允许结束 | `pipeline.md`、`verifier_protocol_maintenance.md` |
| 07-25 至 07-30 | 组件范围、输入写入门控、SimpleClick 实测 | 商业模型也会误接受有害修改（如 test_20_15 IoU 0.846 → 0.814） | `CA_0725_*`、`CA_0726_*`、`CA_0730_*` |
| 07-29 至 07-31 | 换本地 **Qwen3-VL-2B**（v11.1 → v12.1 精简协议） | 未训练的 2B 没有有效动作，IoU 不变；v12.1 让有效审计从 0/45 升到 37/45，但 7 个候选全被拒，包括有益候选 | `training_pipeline/QWEN3VL_*_0730/0731.md` |
| 07-29 至 08-02 | 少量数据多轮 SFT | 格式与动作可执行性改善，但出现大框误覆盖、错误删除；回滚保住初态，selected 指标没有提升；终局独立复核也没有改变结果 | `0729_first_try/`、`0801_second_try/`、`0802_third_try/` |
| 08-03 至 08-04 | 不调用模型的**工具可达性诊断**：GT 评价候选（0803）、GT teacher 直接给点框（0804） | 需修正的 episode 均未到达预定终点：有限动作空间、匹配规则、修改约束会挡住有益动作 | `0803_fourth_try/`、`0804_fifith_try/0804.md` |
| 08-05 至 08-06 | 反思与路线讨论：动作路由、停止策略、效率；五分支并行计划；**研究路线重定位** | 结论：继续把 mask 修得更准的路线论文价值低，建议转向“对象状态变化”或“预算化主动取证”，并把召回（漏检）作为一等问题 | `0805_fifth_try/`（`0805.md`、`0805_parallel_branches.md`、`0805_state_change_agent_research_route.md`、`0806_change_detection_agent_two_routes.md`） |

## 4. 主要问题（摘自 summary.md 第 5 节）

1. **Verifier 的语义判断和候选收益判断不可靠**：本地 2B 会把有误报/漏检的状态判为可结束，也会拒绝有益修改；商业模型也有误接受。
2. **协议复杂度一度阻碍执行**：多阶段筛查、重复字段导致无效输出和高延迟；精简协议改善了格式有效性，但不能替代视觉理解。
3. **动作空间与变化图组合的限制**：点/框工具优化的是单期分割，不一定改善跨期匹配后的变化图；双期 mask 都为空时 Proposal 无法发现漏检。
4. **少量 SFT 没有转化为闭环收益**：模型学会了合规字段，但不会判断该删哪块、该不该接受候选。
5. **漏检是任务结构问题**：没有进入候选集合的漏检，Verifier 再强也看不到（0805 路线文档的核心判断之一）。
6. **LEVIR-CD 的二值变化标签无法唯一监督“改哪一期、哪个对象、什么状态变化”**，限制了训练 Verifier/Agent 的可能。

## 5. 与本仓库摸底实验的关系

导师要求“不能放过漏检”，0806 的路线讨论也把召回列为核心问题，于是在 2026-10-16 先做了三个不训练的摸底实验（仓库根目录 `results/`）。实验不再沿用上面的闭环原型，而是在强检测器（SChanger、MambaSCD）的输出上量化：误差主要是哪一类、SAM3 在完美提示下能把精度侧和召回侧各提多少、MLLM 能否当验证者。主要结论：

- 精度侧（用工具修已有预测）在四个数据集上的上界都很低，与上面闭环原型“修 mask 收益有限”的经验一致。
- 召回侧在通用、语义变化数据（SECOND）上空间最大；漏检位置上检测器没有信号，瓶颈在“找候选”和“确认”。
- 零样本 MLLM 偏向回答“有变化”，不能过滤检测器输出，这与原型中 Verifier 误判的问题一致。只有在建筑的召回确认上，它能接近检测器。
