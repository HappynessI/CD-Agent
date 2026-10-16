# Qwen3-VL-2B SFT baseline（0729）

## 结论

首选 **LLaMA-Factory** 搭建 Qwen3-VL-2B 的 SFT 链路，先做 BF16 + LoRA 的 staged Verifier baseline。

选择依据：

- `Chang-Agent/third_party/SegAgent/README.md` 已明确建议把轨迹转换为 LLaMA-Factory 格式训练。
- 当前官方版本已经提供 Qwen3-VL 多图 ShareGPT 数据、`qwen3_vl_nothink` 模板、LoRA，以及视觉塔/多模态投影层冻结选项。
- 与现有 Change-Agent 的输入输出最容易对齐：多张图像输入，assistant 输出严格 JSON。
- `ms-swift` 可作为第二实现；TRL/Transformers 需要自写多模态 collator，不作为首个 baseline。

## 训练对象

训练当前 staged Verifier 的语义决策，不训练自由坐标回归：

- 输入：T1、T2、当前 T1/T2 预测 mask、change mask，以及 Environment 生成的编号 proposal/crop。
- 输出：`region_id`、T1/T2 状态、audit checklist、错误类型和简短可观察证据。
- 坐标、point/box、`accept`、quality、rollback 和执行边界继续由 Environment/Executor 负责。
- 使用 `qwen3_vl_nothink`，assistant 只输出一个严格 JSON 对象，不训练隐式 CoT。

## 数据构造

数据根目录：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD`。

当前数据规模：

| split | 原始配对数 | 图像尺寸 | 备注 |
|---|---:|---|---|
| train | 445 | 1024×1024 | `train_original/{A,B,label}` |
| val | 64 | 1024×1024 | `val_original/{A,B,label}` |
| test | 128 | 1024×1024 | `test_original/{A,B,label}`，只用于最终评测 |
| test patch | 2048 | 256×256 | 已有 `test_256`，不要回流训练 |

首轮建议把 train/val 原图按固定、无重叠的 4×4 网格切成 256 patch，分别得到约 7120/1024 对。按原图分组后再切分，不能让同一原图的 patch 跨 split。

每个训练样本：

1. 用与评测相同的 fresh SAM3 流程得到初始 T1/T2 mask。
2. 用腐蚀、膨胀、局部增删和真实工具候选构造 FP/FN/mixed/none 状态。
3. 仅在离线构造标签时读取 GT；原始 GT label 图不进入模型输入。当前 32 条 smoke 为了稳定打通链路，会把由 GT 派生的 candidate mask、proposal 几何和 mask facts 放进模型可见 overview/prompt；因此它不是无泄漏的视觉泛化训练集，完整 baseline 必须改用冻结预测器/工具输出再生成这些 facts。
4. 只有确实知道被扰动视图时才生成 `target_view`；禁止按样本编号交替伪造该标签。
5. 转成 LLaMA-Factory ShareGPT JSON 数组：`conversations` 中放 `<image>` 占位符，`images` 数组放对应图像路径；占位符数量必须完全匹配。

建议优先复用现有 staged 输入/提示构造逻辑，避免训练格式与推理格式漂移：

- `change_agent/adapters/stage_backends.py`
- `change_agent/adapters/staged_verifier.py`
- `change_agent/verifier_protocol.py`

## 首轮配置

```yaml
model_name_or_path: /guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct
template: qwen3_vl_nothink
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true
bf16: true
cutoff_len: 4096
image_max_pixels: 262144
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
gradient_checkpointing: true
train_on_prompt: false
packing: false
```

先用单 GPU、32 条样本做 overfit/smoke；通过后再提交完整 train。训练、数据处理和评测均通过 Slurm，不在管理节点直接运行计算任务。首基线暂不引入 QLoRA、全参数训练、ZeRO 或视觉塔解冻。

## 验收指标

- **数据门**：无原图跨 split 泄漏；图像占位符与路径一一对应；保存类别和样本构造统计。
- **协议门**：验证集严格 JSON/schema 合法率 ≥99%；未知 `region_id` 为 0。
- **离线门**：诊断 macro-F1、region selection accuracy、candidate better/worse accuracy 高于同一 Qwen3-VL-2B zero-shot。
- **闭环门**：固定 SAM3 初始化、proposal、工具、seed 和最大步数，对比 zero-shot/SFT 的 selected IoU/F1、`ΔIoU`、退化样本率、commit precision、tool success、invalid/no-action 率。
- 超参数只使用 val 选择；锁定后才运行 `test_original`/`test_256` 的最终报告。

## 边界与风险

- 445 个独立 train 场景较少；切成 7120 patch 不等于 7120 个独立场景，存在过拟合风险。
- LEVIR-CD 是二值变化标注，没有逐时相建筑实例 GT；自然场景下的 `target_view/action` 监督不完整，合成标签和 teacher 标签必须单独记录。
- SFT 是静态行为模仿，不保证多步闭环一定提升，也不会直接生成最终像素 mask。
- 效果上限受 proposal recall、SAM3/SimpleClick 和安全 gate 限制；候选未覆盖真实错误时，Verifier 无法凭空修复。
- 冻结视觉塔的结果只说明语言/协议侧适配效果；遥感视觉域适配应作为后续独立消融。
- Qwen3-VL-Plus 可用于生成辅助证据，但不是无误标签器；应保留人工/规则一致性过滤。
- 训练环境必须独立于旧 `segagent-env`（其 Transformers 4.31 不支持 Qwen3-VL）；同时固定 LLaMA-Factory、Transformers、PEFT、Accelerate 版本和 commit。

## 现有链路参考

- Qwen3-VL-2B adapter：`change_agent/adapters/qwen3vl_adapter.py`
- staged backend：`change_agent/adapters/stage_backends.py`
- 训练 schema 约束：`tests/test_verifier_training_schema.py`
- proposal/grounding 设计：`doc/GROUNDING_PAPERS_ANALYSIS.md`
- 既有闭环结果与安全边界：`doc/verifier_protocol_maintenance.md`
