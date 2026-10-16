# 文档入口

现有的 59 个文件已于 2026-10-16 集中归档到 [archive/2026-10-16](archive/2026-10-16/)，保留原文件名、目录结构和内容。它们覆盖早期实现、历史实验和研究设想，不能直接当作当前接口或下一阶段开发要求。

后续新的设计、使用说明和实验报告直接在 `doc/` 下维护。

## 当前实验

| 内容 | 入口 |
| --- | --- |
| 变化检测 agent 摸底实验（误差分解、SAM3 oracle 上界、MLLM 验证能力），2026-10-16 | 任务说明 [change_dection_agent.md](change_dection_agent.md)；脚本 [`probe/`](../probe/README.md)；结果 `outputs/CD_1016_probe/results/`（`summary.md` 为一页总结） |

## 历史资料入口

| 内容 | 入口 |
| --- | --- |
| 旧系统整体流程、结果与问题 | [系统总结](archive/2026-10-16/summary.md) |
| 早期开发过程 | [开发记录](archive/2026-10-16/DEVELOPMENT.md)、[开发工作报告](archive/2026-10-16/DEVELOPMENT_WORK_REPORT.md) |
| 旧流程与 Verifier 协议 | [流程说明](archive/2026-10-16/pipeline.md)、[协议维护记录](archive/2026-10-16/verifier_protocol_maintenance.md)、[Proposal 对照](archive/2026-10-16/proposal_ablation.md) |
| 商业模型阶段分析 | 归档目录下的 `CA_0725_*`、`CA_0726_*`、`CA_0730_*` 和 `ca0721_*`、`ca0722_*` |
| 本地模型开发与训练尝试 | [训练过程目录](archive/2026-10-16/training_pipeline/)；内部仍按 0729、0801、0802、0803、0804、0805 分组 |
| 后期研究路线设想 | [状态变化路线](archive/2026-10-16/training_pipeline/0805_fifth_try/0805_state_change_agent_research_route.md)、[双路线讨论](archive/2026-10-16/training_pipeline/0805_fifth_try/0806_change_detection_agent_two_routes.md) |

## 原路径的使用范围

旧脚本和实验登记仍引用 `doc/training_pipeline`、`doc/pipeline.md`、`doc/verifier_protocol_maintenance.md`，这三个路径通过符号链接指向归档。它们用于定位旧资料，不代表对应说明仍适用于当前实现。

归档原文中的日期、运行路径、内部引用和阶段性判断保持原样；部分相对链接仍基于原目录位置。旧实验输出已清理，复验相关结论须重新生成产物。历史工具验证记录另保存在项目 `diagnostics/legacy/`。
