# 任务：实验 5，召回侧补测

给服务器上的 Claude。前置结果见 `results/summary.md` 中的"实验 4"一节，以及 `results/exp4a_candidates.md`、`exp4b_confirm.md`、`exp4c_end_to_end.md`。本任务沿用实验 4 的代码（`code/probe/exp4*.py`）、口径、样本和运行方式，只补四件事。

## 为什么要补测
实验 4 的结论是"零样本的找、确认、补全流水线在 SECOND 上拿不到正收益，第一瓶颈是确认"。下这个结论之前还缺几块：
1. 两个本该测的现成工具没有用上：**AnyChange**（找候选）和 **RemoteCLIP**（确认"两期是否同一种地物"）。实验 4 的任务文档版本里没有它们，这次补上。
2. 补全用的是"候选外接框提示 SAM3"。在 oracle 确认下，SECOND 新增的 FP 像素（1148 万）仍多于 TP（898 万），所以 +1.46 可能低估了"找"的贡献。
3. 检测器自己的低概率是最好的确认信号，但还没试过"把可疑区域放大后让检测器再看一遍"。
4. 4c 用 B + G0.05，但 4a 没统计这两者并集的覆盖率。

## 使用的工具（已定）

| 环节 | 工具 | 来源 |
|---|---|---|
| 检测器 | MambaSCD-Base（SECOND）、SChanger-small（SYSU） | 服务器已有 |
| 找 | **AnyChange**（Segment Any Change，NeurIPS 2024） | `pip install torchange`，或 github.com/Z-Zheng/pytorch-change-models（代码在 `torchange/models/segment_any_change`）；按其示例下载所需的 SAM 权重 |
| 确认 | **RemoteCLIP** | github.com/ChenDelong1999/RemoteCLIP，权重 huggingface.co/chendelong/RemoteCLIP（优先 ViT-L-14，显存或速度不够再用 ViT-B-32） |
| 确认 | 检测器局部复检、实验 4 已有信号 | 服务器已有 |
| 补全 | SAM3 框提示，或直接用候选区域 | 服务器已有 |

**AnyChange 和 RemoteCLIP 允许下载**（代码和上表所列权重），装在新的独立 conda 环境里，不要动现有的 omniovcd-env 和 vlm-vs-onejev-env。安装或下载失败就停下来报告，不要换成其他模型。除上表外不下载新的大模型。

## 第 0 步
在 `results/env.md` 追加"实验 5 资源"一节：AnyChange 和 RemoteCLIP 的安装方式、版本、权重路径，以及各在一张 SECOND 图上跑通的确认。

## 5a：AnyChange 作为候选来源
- **H1 全图模式**：AnyChange 默认设置对两期影像做变化提议，取输出的变化掩码，分数用其自带的变化置信度。
- **H2 点查询**：以检测器在同一张图上高置信目标（概率 > 0.9）的中心点为查询点，用 AnyChange 点查询找同类变化。
- 统计与实验 4a 完全相同（无信号漏检集合 M、覆盖定义、纯度、按变化类型和面积拆分、前 k 曲线、去掉面积上限的覆盖率）。
- 补统计并集：B+G0.05、B+G0.05+H1、B+C+G0.05+H1，以及 H1 与 A–E 的两两并集。
- 输出：`results/exp5a_anychange.md`，可视化放 `figures/exp5a/`。

## 5b：新的确认信号
在实验 4b 的**同一批**候选（每个数据集 300 个，B 与 G0.05 各 75 正 75 负）上加两类信号；如果 5a 中 H1 的覆盖率高于 B，再从 H1 按同样方式抽 75 正 75 负加进来，并单独报告。
1. **RemoteCLIP**：
   - 两期裁块（裁块方式同实验 3/4b）嵌入的余弦相似度，越低越像变化；
   - 两期对 SECOND 六类文本（如 "a satellite image of buildings / trees / low vegetation or farmland / water / bare ground or road / a playground"）的零样本分类结果是否不同，以及两期类别分布的差异度（如 1 − Σ p1·p2）。
2. **检测器局部复检**：以候选外接框为中心外扩裁块，送回同一个检测器重新推理，取候选区域内的平均概率。试两种：
   - 裁块放大到检测器的原始输入尺寸（如 SECOND 512²）；
   - 裁块不放大，在原尺寸上推理（只改变上下文）。
   记录用的外扩比例。
- 在逻辑回归（沿用实验 4b 的实现和 5 折交叉验证）里加入这些信号，与 4b 的最好结果比较。
- 指标同 4b：AUC（95% CI）、召回 80% / 90% 时的误检率。
- 另报告一个更贴近真实的指标：按真实候选池的正例率（各来源在 4b 中记录的正负数量）重新加权后，召回 80% 时补进来的候选里真变化的比例。
- 输出：`results/exp5b_confirm.md`，附 10 个判错样例。

## 5c：补全方式的变体（重做 4c 的 oracle 部分 + 最好信号的端到端）
在 SECOND 和 SYSU 上，候选仍为 B + G0.05（若 5a 中 H1 明显更好，再加一组含 H1 的）：
1. **oracle 确认 + 直接并入候选区域**（不经 SAM3）。
2. **oracle 确认 + 候选切细后再补**：大候选（例如面积 > 目标中位面积的 4 倍，阈值自己定并写明）先切成小块，切法任选一种并写明理由（SAM3 自动实例、超像素，或与检测器概率 > 0.05 区域求交），再分别用"直接并入"和"SAM3 框提示"两种方式补。
3. **最好的确认信号（来自 5b）+ 上面效果最好的补全方式**，扫规则阈值（同 4c 的 0.5–0.9 和召回约 80%），不用真值。
- 报告 F1、P、R、ΔF1、新增 TP / FP 像素，并与实验 2 的上界（SECOND +7.81、SYSU +1.89）和 4c 的结果对照。
- 输出：`results/exp5c_end_to_end.md`。

## 判断标准（写在 summary 里）
补测的目的是决定下一步是继续零样本，还是转向训练验证器。请明确回答：
1. 有没有任何确认信号或组合，在平衡样本上召回 80% 时误检率降到 15% 以下？
2. 不靠真值的端到端有没有任何设定拿到正的 ΔF1？
3. oracle 确认下，换了补全方式后，SECOND 能拿到上界的几成？
如果 1 和 2 都是否定的，就说明确认环节必须训练，下一个实验将是训练漏检验证器。

## 可以自己判断的
AnyChange 的超参数、RemoteCLIP 文本提示的措辞、复检的外扩比例、候选切细的方法和阈值、可视化样式。每个改动在结果里写一句理由。

## 不要做的
- 不训练任何网络（逻辑回归除外），不改检测器，不实现完整 agent。
- 不用真值语义图生成候选。
- SECOND/SYSU 的裁块可以发百炼（已有授权），本实验不需要再调 MLLM；不发 GPT。

## 交付
`results/exp5a_anychange.md`、`results/exp5b_confirm.md`、`results/exp5c_end_to_end.md`，在 `results/summary.md` 末尾追加"实验 5"一节，回答上面的三个判断问题。代码放 `code/probe/exp5_*.py`。
