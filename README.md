# CD-Agent：变化检测 agent 摸底实验

目的：在决定变化检测 agent 的主攻方向（召回侧还是精度侧）之前，用三个实验量出现有检测器的改进空间。任务说明见 [`doc/change_dection_agent.md`](doc/change_dection_agent.md)。

截至 2026-10-16 的状态：**LEVIR-CD 和 WHU-CD 的三个实验已完成，结论写在 `results/`**。SYSU-CD 和 SECOND 是后来加入的，部分结果还在运行，目前只有中间表格（`interim/`），尚未写进 `results/` 的结论。

## 从哪里读

| 路径 | 内容 |
| --- | --- |
| [`results/summary.md`](results/summary.md) | **一页总结（LEVIR/WHU）**：哪类误差为主、精度侧与召回侧上界、MLLM 能否验证、方向判断 |
| [`results/exp1_error_breakdown.md`](results/exp1_error_breakdown.md) | 实验 1：误差分解（目标级匹配、面积分桶、阈值扫描） |
| [`results/exp2_oracle.md`](results/exp2_oracle.md) | 实验 2：SAM3 + 真值框提示的 oracle 上界 |
| [`results/exp3_mllm_verify.md`](results/exp3_mllm_verify.md) | 实验 3：MLLM（Qwen3.5-4B、百炼 qwen3.7-plus）判断“是否真变化” |
| [`results/env.md`](results/env.md) | 数据、检测器、模型和运行环境 |
| `results/exp3_error_examples*/` | 实验 3 的 MLLM 判错样例（t1、t2 裁块，红框为候选区域） |
| `figures/exp1_vis/<数据集>/<检测器>/` | 实验 1 每类误差 5 个样例：t1、t2、真值、预测叠加（绿 TP、红 FP、蓝 FN） |
| `interim/exp1/tables.md` | 实验 1 全部表格，**含 SECOND**（按变化类型拆分） |
| `interim/exp2/tables.md` | 实验 2 全部表格（LEVIR/WHU） |
| `interim/exp3/tables_qwen35_4b.md`、`tables_qwen37plus.md` | 实验 3 表格，**含 SYSU-CD 和 SECOND** |
| `interim/*/*.json` | 上述表格对应的原始汇总 |
| `code/probe/` | 实验脚本（Slurm 集群上运行；路径为服务器路径） |

## 设置要点

- 数据：LEVIR-CD test（128 张 1024²）、WHU-CD test（690 张 512²）、SYSU-CD test（4000 张 256²，通用变化）、SECOND test（1694 张 512²，语义变化，6 类）。
- 检测器：SChanger-small/base（官方权重，LEVIR/WHU/SYSU 上复现的 F1 与论文一致），MambaSCD-Base（SECOND，F1 74.17），SAM3+OmniOVCD 零样本基线（仅 LEVIR/WHU）。
- 实验 1 的判定：8 连通目标；IoU ≥ 0.5 为命中，0.1–0.5 为部分命中，< 0.1 为整目标漏检；与所有真值 IoU < 0.1 的预测为整目标误检。SECOND 的真值目标先按（t1 类别，t2 类别）拆分再取连通域。另补充“无信号漏检”（预测覆盖目标像素 < 10%），因为在 SECOND 上，大预测块会让已被覆盖的小目标 IoU 很低。
- 实验 3 的 MLLM 输入：候选区域外扩裁块，t1 和 t2 各一张并画红框，要求输出 yes/no 和置信度；LEVIR/WHU 问“建筑变化”，SYSU/SECOND 问“真实地物变化”。

## LEVIR/WHU 的结论（详见 `results/summary.md`）

1. LEVIR 以边界误差为主（74%）；WHU 整目标漏检更多（15% 的目标）。漏检目标的检测器概率接近 0，降阈值补不回来。
2. 用真值框提示 SAM3 时，精度侧直接替换反而降分（LEVIR −1.5），即使有完美判断器也只有 +0.5 到 +0.9；召回侧补齐漏检为 +0.5（LEVIR）和 +1.2（WHU）。
3. MLLM 不能过滤检测器输出（偏向回答“是”），但能确认召回候选：百炼在 WHU 上区分漏检建筑和未变化建筑的 AUC 为 0.93。

## SYSU-CD / SECOND 中间结果（尚未写入 results）

- **SECOND 实验 1**（`interim/exp1/tables.md`）：F1 74.16；误差像素中整目标漏检 36%、部分命中 31%、误检 18%、边界只有 15%，与 LEVIR 的边界主导完全不同。按变化类型看，新建建筑几乎不漏（无信号漏检像素 1.8–5.4%），改建（建筑→建筑）为 40%，植被相关变化为 25–43%，地表↔植被为 21–26%，即难漏的是语义变化而不是小目标。阈值降到 0.4 时 F1 略升（74.45），说明阈值未校准。按变化类型拆分后有 47.5% 的目标小于 20 px，但只占 0.1% 的变化像素，是类别交界处的碎片。
- **SYSU/SECOND 实验 3**（`interim/exp3/`）：
  - 在检测器已预测的区域上，MLLM 仍不如检测器自身的置信度：百炼的 AUC 为 0.73（SYSU）和 0.81（SECOND），检测器为 0.97 和 0.93。
  - 召回侧（漏检 vs 随机未变化区域）上，百炼为 0.71 和 0.81，低于检测器低概率排序的 0.91 和 0.88，也低于它在 WHU 建筑上的 0.93。通用地物变化上 MLLM 的确认能力明显弱于建筑变化。
  - 注意：SYSU/SECOND 的召回侧负样本是尺寸匹配的随机未变化框，不是 LEVIR/WHU 的“未变化建筑”，两者不可直接比较。
- 运行中：SYSU 的实验 1/2（SChanger-base 推理刚开始）、SECOND 的实验 2（SAM3 oracle 与合成），以及 LEVIR/WHU 上的 GPT（gpt-6.1-sol）补测（转发通道间歇断开，较慢）。完成后会更新本仓库。
