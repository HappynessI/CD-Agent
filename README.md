# CD-Agent：变化检测 agent 摸底实验

目的：在决定变化检测 agent 的主攻方向（召回侧还是精度侧）之前，用三个实验量出现有检测器的改进空间。任务说明见 [`doc/change_dection_agent.md`](doc/change_dection_agent.md)。

截至 2026-10-16 的状态：**四个数据集（LEVIR-CD、WHU-CD、SYSU-CD、SECOND）的三个实验都已完成，结论写在 `results/`**。唯一在运行的是 LEVIR/WHU 上的 GPT（gpt-6.1-sol）补测（转发通道间歇断开，较慢），完成后会更新 `results/exp3_mllm_verify.md`。

## 从哪里读

| 路径 | 内容 |
| --- | --- |
| [`results/summary.md`](results/summary.md) | **一页总结（四个数据集）**：哪类误差为主、精度侧与召回侧上界、MLLM 能否验证、方向判断 |
| [`results/exp1_error_breakdown.md`](results/exp1_error_breakdown.md) | 实验 1：误差分解（目标级匹配、面积分桶、阈值扫描；SECOND 按变化类型） |
| [`results/exp2_oracle.md`](results/exp2_oracle.md) | 实验 2：SAM3 + 真值框提示的 oracle 上界 |
| [`results/exp3_mllm_verify.md`](results/exp3_mllm_verify.md) | 实验 3：MLLM（Qwen3.5-4B、百炼 qwen3.7-plus）判断“是否真变化” |
| [`results/env.md`](results/env.md) | 数据、检测器、模型和运行环境 |
| `results/exp3_error_examples*/` | 实验 3 的 MLLM 判错样例（t1、t2 裁块，红框为候选区域） |
| `figures/exp1_vis/<数据集>/<检测器>/` | 实验 1 每类误差 5 个样例：t1、t2、真值、预测叠加（绿 TP、红 FP、蓝 FN） |
| `interim/exp1/tables.md`、`interim/exp2/tables.md`、`interim/exp3/tables_*.md` | 四个数据集的完整表格（`results/` 中的表格摘自这里） |
| `interim/*/*.json` | 表格对应的原始汇总 |
| `code/probe/` | 实验脚本（Slurm 集群上运行；路径为服务器路径） |

## 设置要点

- 数据：LEVIR-CD test（128 张 1024²）、WHU-CD test（690 张 512²）、SYSU-CD test（4000 张 256²，通用变化）、SECOND test（1694 张 512²，语义变化，6 类）。
- 检测器：SChanger-small/base（官方权重，LEVIR/WHU/SYSU 上复现的 F1 与论文一致），MambaSCD-Base（SECOND，F1 74.17），SAM3+OmniOVCD 零样本基线（仅 LEVIR/WHU）。
- 实验 1 的判定：8 连通目标；IoU ≥ 0.5 为命中，0.1–0.5 为部分命中，< 0.1 为整目标漏检；与所有真值 IoU < 0.1 的预测为整目标误检。SECOND 的真值目标先按（t1 类别，t2 类别）拆分再取连通域。另补充“无信号漏检”（预测覆盖目标像素 < 10%），因为在 SECOND 上，大预测块会让已被覆盖的小目标 IoU 很低。
- 实验 3 的 MLLM 输入：候选区域外扩裁块，t1 和 t2 各一张并画红框，要求输出 yes/no 和置信度；LEVIR/WHU 问“建筑变化”，SYSU/SECOND 问“真实地物变化”。

## 核心结论（详见 `results/summary.md`）

1. **误差结构随数据变化**：LEVIR 以边界为主（74%）；数据越“通用”，漏检越重（SECOND 整目标漏检 36%、部分命中 31%）。SECOND 上难漏的是改建和植被等**语义变化**，不是小目标。降阈值几乎补不回漏检。
2. **用真值框提示 SAM3 的上界**：精度侧在四个数据集上都不值得做（有完美判断器也只有 +0.5 到 +1.6 F1）；召回侧的上界从 LEVIR +0.5 增大到 SECOND **+7.8**。
3. **零样本 MLLM 不能过滤检测器输出**（偏向回答“是”）。只在建筑的召回侧能与检测器持平（百炼在 WHU 上 0.93）；在通用变化上各项都不如检测器的低概率排序。
4. **方向**：主攻召回侧，以 SECOND 型多类语义变化为主要评测场景。瓶颈在候选生成和确认，不在分割工具。
