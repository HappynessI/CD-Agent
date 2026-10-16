# Change-Agent 开发工作报告

## 2026-07-19：修复 SAM3 隔离 worker 的 CUDA 混合精度上下文

迁移后的首个 A800 单卡 smoke 作业 `40907` 已进入 fresh SAM3 初始化，但在 fused
ViT MLP 中因 BF16 激活与 FP32 权重不一致而失败。根因是隔离的 segmentation
worker 没有像 SAM3 官方推理示例一样进入 CUDA autocast 上下文，并非数据或模型
文件损坏。

现已将 SAM3 的双时相文本初始化和 box 推理包在局部 autocast 中：支持 BF16 的
GPU 使用 BF16，否则回退 FP16；CPU 路径保持空上下文。已增加 CPU、BF16 和 FP16
回退的单元测试。对应 GPU 资源已由 Slurm 释放；最终闭环成功后，失败诊断目录已
按输出清理要求删除。

后续作业 `40909` 已通过 fused MLP，说明混合精度修复生效；它进一步发现 SAM3
在某个时相没有检测到 building 时会返回显式的空数组。Adapter 旧逻辑对空数组
执行 `max`，现改为输出同图尺寸的全零 mask 和 confidence；只有完全缺少任何
mask 输出字段时仍抛出集成错误。该边界已有回归测试，资源已释放；失败诊断目录
已在最终闭环成功后删除。

作业 `40910` 随后完成了 SAM3 推理，仅在持久化 BF16 诊断 Tensor 时失败；PyTorch
不能把 BF16 storage 直接暴露给 NumPy。现仅在诊断序列化边界将 BF16 提升为
FP32，不改变模型推理精度或 mask 计算，并用真实 BF16 Tensor 增加回归测试。
GPU 已释放；失败诊断目录已在最终闭环成功后删除。

最终单卡作业 `40911` 在一张 A800 上用 114 秒完成三个固定 LEVIR-CD 样本，Slurm
状态为 `COMPLETED (0:0)`，释放记录为 `squeue_entry_after_completion=0`。保守选择
结果的汇总 IoU/F1 为 `0.69744116`/`0.82175592`。本轮验证了：不带
`coordinate_frame` 的 point 可执行；重试耗尽不再触发合成 box；未提升或面积跳变
过大的 candidate 会回滚，最终 selected prediction 不受污染。

仍需后续解决的模型策略问题是：共 10 个 Agent 输出缺少必需坐标；6 次 Verifier
定位全部退化为整图；`test_85_16` 的第三个候选在闭环结束后计算出的 IoU 从
`0.30658070` 提升到 `0.38875878`，但在线 GT-free Verifier 分数始终为 0，因此被
保守策略拒绝。完整审计见
`outputs/change_agent_levir_gpu_smoke_20260719_030624/CLOSED_LOOP_AUDIT.md`。

## 2026-07-18：向 Verifier 暴露双时相 predicted object mask

此前 Qwen Verifier 只能看到 T1/T2 原图和最终 current change mask，无法直接判断
变化来自哪一个时相的 object mask。现已将输入固定为五张带标签的图像：
`T1 original image`、`T2 original image`、`Predicted T1 object mask`、
`Predicted T2 object mask`、`Current change mask`。Prompt 明确说明两个 predicted
mask 不是 GT，current change mask 是由它们和 OmniOVCD matching 共同重建的。

这样 Verifier 可以直接区分新增、消失、误标和时相归因；输入顺序和标签已加入
回归测试。五张图会增加 Qwen 视觉编码开销，下一轮 GPU smoke 需记录显存和耗时。

## 2026-07-18：删除自动 box fallback，并回滚未接受 candidate

上次三个样本唯一实际执行的工具动作都是重试耗尽后由 runner 自动生成的 SAM3
大框，且全部降低离线 IoU。现已删除该 fallback：动作重试耗尽时不再调用任何
分割工具、不改变当前 mask，记录全部原始无效输出和
`episode_stop_reason=action_retry_exhaustion_without_state_change`，随后安全导出当前
及历史最佳结果；没有工具动作不再被当作 runner 异常。

Environment 新增 candidate commit 边界。工具候选只有在 Verifier 有效、分数提升
超过 `selection_epsilon`、mask 面积绝对变化不超过
`max_selection_area_delta` 时才成为下一轮当前状态。否则候选 mask、工具证据、
Verifier 输出、面积变化和拒绝原因仍完整写入 trajectory，但 live state 与 feedback
恢复为上一轮已接受版本；step index 继续推进以避免反复失败造成无限循环。最终
selected state 排除 rejected candidate，`verifier_best` 仅保留为原始评分审计产物。

## 2026-07-18：Verifier 两阶段诊断与动作程序推导

针对上次闭环中 `quality_score`、`error_type`、`suggested_action` 和 `accept`
互相矛盾，以及大量缺少 `error_region` 的反馈，现将 Qwen 输出降级为诊断输入：
只读取质量分数、错误类型、目标时相、可选错误区域和文本说明。

当诊断指出有错误但没有区域时，运行时保留错误类型和文本，并单独发起只要求
`error_region` 的定位请求。定位再次失败才判定 `verifier_valid=false`；此时同时
设置 `localization_valid=false`、`suggested_action=null`、`stop=false`，保留上一轮
有效 feedback，不再向 Agent 发送 `finish`。

`accept`、`stop` 和 `suggested_action` 不再由模型生成或信任：`none` 统一推导为
`finish`，只有质量分数达到阈值才 accept/stop；false positive 推导负点，false
negative 推导正点，mixed/uncertain 推导 box。Environment 只依据 `stop` 结束。

## 2026-07-18：坐标协议改由系统单方定义

复盘 `change_agent_levir_fresh_qwen_20260718_133335` 后确认，三个样本共有 5 次
本可执行的 point 动作仅因没有复述 `coordinate_frame` 而被拒，并导致每个样本都
耗尽动作重试、进入 box safety fallback。现已从 Qwen action schema 与 runner
fallback 中删除该字段要求：系统始终将公开 action 坐标解释为 `[0,1000]` 归一化
XY/XYXY，模型只输出 `target_view`、`action` 和对应的 `coordinate`/`box`。

Parser 对缺少该字段的 point/box 动作直接接受，同时仅为旧产物兼容而接受值为
`normalized_1000_xy` 的历史字段；冲突值仍会拒绝，不能覆盖系统协议。相应的
ActionParser、Qwen prompt、Environment trajectory 与 fallback 回归测试均已更新。

## 2026-07-18：首轮闭环问题修复

首轮三样本闭环确认了工程链路可执行，但 9 次动作均未提升离线 IoU。针对审计
发现的问题，本轮完成以下修复：

1. Agent 和 Verifier 的公开坐标全部统一为 `[0,1000]` 归一化 XY/XYXY，
   `ActionParser` 之后的 Environment 和工具边界才使用原图像素坐标。
2. 三样本正式入口删除缓存 mask 参数，每个样本重新加载一次 SAM3，并在同一
   worker 中依次执行 T1/T2 文本提示初始化。
3. 初始化目录保存 T1/T2 输入、mask、confidence map、presence/object score、
   prompt、checkpoint、resolution、命令、stdout/stderr 和 report。
4. 删除此前按样本序号交替生成的伪 `target_view` 训练标签；Verifier head、loss、
   NPZ schema 和 checkpoint schema 均不再包含 target-view 监督，也暂不训练
   action head；第一阶段只保留 quality、error-map、error-type。
5. 新增共享 Agent 权重的 Qwen3-VL zero-shot Verifier，输出结构化质量分数、错误
   类型、真实视觉推断的目标时相、归一化错误区域、建议动作和 accept。规则
   Verifier 只保留为显式消融。
6. 增加 `coordinate_frame=normalized_1000_xy` 必填校验、连续小坐标告警，并在
   trajectory 中同时保存 raw normalized payload 与 parsed pixel action。
7. 增加 `verifier_best`、`conservative_best`、`initial` 三种选择策略；所有 step
   mask 以及 initial/verifier-best/last/selected 预测均保留。

当前实现只完成代码与无权重单元测试，尚未启动下一轮长时间 GPU 推理。

首次 fresh-SAM3 真实启动暴露了概率阈值问题：SAM3 public processor 的
`semantic_mask_logits` 实际已在 `[0,1]`，旧 adapter 用 `>0` 会产生全图 mask。
现已按 OmniOVCD 配置恢复 semantic/instance/object/presence 融合与 `0.4` 阈值；
失败运行目录保留为诊断证据，修复后重新启动独立输出目录。

第二次真实启动进一步暴露了 prompt-only 的首轮 finish 约束无效：Qwen 连续输出
finish，耗尽三步但没有工具动作。Environment 现已在首个真实分割动作之前强制
拒绝 finish，并将具体 validation error 注入下一次 Qwen retry Prompt。

第三次启动确认当前 Qwen zero-shot 仍可能忽略三次 retry 提示。为避免闭环在
首个样本直接终止，runner 增加了明确标记的安全兜底：从当前可见 change mask
计算一个有边界的 T1/T2 SAM3 box，所有模型原始输出和 fallback 都写入轨迹旁的
`invalid_agent_outputs.json` 与 rollout record，不把 fallback 伪装成模型动作。

随后同一轮的首个 SAM3 box 已执行成功，但 Verifier 又因两次非结构化/缺少
`error_region` 的诊断而中止。现已改为 Verifier 级别的可审计 `uncertain_region`
安全返回：保留上一质量分、不接受候选并记录全部 validation errors，允许闭环继续
暴露后续步骤，同时不把无效输出当作有效诊断。

## 2026-07-17：Matching 决策落地与三样本完整闭环入口

本轮将默认 matching 从确定性一对一 greedy 改为 OmniOVCD 原始的双向
overlap-presence 语义，默认阈值为 `0.25`。一对一 greedy 保留为
`greedy_one_to_one` 消融，不再是默认方案。状态 evidence 新增方向性 coverage、
候选配对、实例数量、面积阈值与 `split_merge_ambiguity`。

新增真实三样本入口 `tools/run_levir_change_agent.py`。该入口不调用
`OmniOVCD/eval.py`，而是用 `OmniOVCDAdapter` 装载先前真实 OmniOVCD/SAM3
三样本运行保存的 T1/T2 初始 mask，再依次执行 Qwen3-VL action、ActionParser、
隔离环境中的 SimpleClick/SAM3 worker、Environment rebuild 和
RuleBasedVerifier。所有样本 trajectory 保存完成并写入 `rollout_complete.json`
之后，才打开 `label_cvt` 计算离线指标。

完整命令、逐样本 action、工具结果、指标和产物清单记录在：

```text
/Data/wyh/CD-SegAgent/doc/NEXT_TASK_FULL_CHANGE_AGENT_RUN.md
```

更新时间：2026-07-17
代码仓库：`git@github.com:HappynessI/Chang-Agent.git`
当前分支：`main`
当前提交：`d2002a1 fix: enable single-GPU smoke execution`

本文记录本轮开发实际完成的代码、模型、测试、资源测量和 Git 管理工作。模型权重和数据集没有提交到 Git，只保留在本地路径。

## 1. 最终代码结构

### 1.1 Agent 运行闭环

- `change_agent/state.py`
  - 定义 `AgentAction`、`AgentObservation`、`ChangeState`、`VerifierOutput` 等状态对象。
  - Agent 观察只包含 T1、T2、当前 change mask、query、Verifier 反馈和历史摘要。
  - 隐藏的 T1/T2 mask、实例、匹配关系和模型证据保留在 Environment 内部。
- `change_agent/action_parser.py`
  - 从普通文本或 Markdown JSON fence 中提取一个 JSON action。
  - 校验 `target_view`、动作类型、点坐标、box 坐标和图像边界。
  - 将 `[0,1000]` 归一化坐标转换成像素坐标。
- `change_agent/environment.py`
  - 实现 GT-free 的 `reset` / `step` 运行时环境。
  - `reset` 不接受 GT；运行时只依赖 T1、T2 和 query。
  - 支持 point、box、finish 三类动作，维护最大步数和终止条件。
- `change_agent/executor.py`
  - 将 point 动作交给 SimpleClick 边界，将 box 动作交给 SAM3/OmniOVCD 边界。
- `change_agent/trajectory.py`
  - 保存每一步 raw action、解析后的 action、Verifier 输出、state 快照和执行证据。
  - 支持 JSON trajectory 与每步 mask 文件落盘。
- `change_agent/runner.py`、`change_agent/agent.py`
  - 提供脚本 Agent 和统一 rollout runner。
  - 支持历史最佳 state 选择，而不是只返回最后一步。

### 1.2 Mask、实例和 change state

- `change_agent/adapters/omniovcd_adapter.py`
  - 通过回调隔离 OmniOVCD/SAM3 的重模型依赖。
  - 实现 T1/T2 mask 初始化、box segmentation、8 邻域 connected components、实例匹配和 change mask 重建。
  - 初版曾使用确定性一对一 greedy；2026-07-17 已由本文开头记录的
    `overlap_presence` 默认策略取代，greedy 仅保留为消融。
- `change_agent/adapters/sam3_adapter.py`
  - 接入 OmniOVCD 的公开 `set_image`、`set_text_prompt`、`add_geometric_prompt` API。
  - 将 SAM3 logits/masks 转成 NumPy bool mask 和 confidence map。
  - 新增 `segment_text`，支持低内存场景按视图分阶段调用。
- `change_agent/adapters/segagent_adapter.py`
  - 包装 SegAgent 的 SimpleClick 推理模型。
  - 将 Environment 当前目标视图 mask 作为 SimpleClick `prev_mask` 传入。
- `change_agent/perturbations.py`
  - 仅供离线 Verifier 监督使用，生成 erode、dilate、local add/delete 等候选 mask 和误差标签。

### 1.3 Qwen3-VL Adapter

文件：`change_agent/adapters/qwen3vl_adapter.py`

完成内容：

1. 使用 `AutoProcessor` 和 `Qwen3VLForConditionalGeneration` 加载本地 Qwen3-VL-2B-Instruct。
2. 通过现代 `apply_chat_template` API 构造多模态消息。
3. 明确标记三类图像：T1 earlier image、T2 later image、current change mask。
4. 强制模型只返回一个结构化 action JSON。
5. 将生成文本交给统一 `ActionParser`，避免 Adapter 自己重复坐标和动作校验。

模型本地路径：

```text
/Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct
```

模型文件不在 Git 中。

## 2. Verifier 实现

### 2.1 规则 Verifier

文件：`change_agent/verifier.py`

实现了可直接运行的 rule-based baseline：

- 根据 change mask 比例、模型 evidence 和动作类型计算质量分数。
- 对 finish 动作进行接受/拒绝判断。
- 输出 quality score、error type、target view、recommended action 和 error map。

该 Verifier 是运行闭环基线，不代表最终研究模型。

### 2.2 可训练 Verifier head

文件：`change_agent/verifier_model.py`

实现了冻结视觉特征上的轻量训练头：

- 两层卷积 encoder。
- quality regression head。
- error-map segmentation head。
- error-type classification head。
- action classification head。
- 组合 MSE、BCE、Dice 和多项 cross entropy loss。

训练入口：`tools/train_verifier.py`

支持两种输入：

- `--smoke`：确定性合成数据，用于代码和 optimizer smoke。
- `--samples path.npz`：真实离线样本。

### 2.3 真实 Verifier 样本构建

新增：`tools/build_verifier_samples.py`

该工具读取离线数据集的：

- `A/<name>.png`
- `B/<name>.png`
- `label_cvt/<name>.png`

利用 GT 只在离线样本构建阶段生成候选 mask 和监督标签，输出特征不包含原始 GT mask。当前特征由以下内容组成：

- T1 RGB：3 通道
- T2 RGB：3 通道
- T1/T2 absolute difference：3 通道
- candidate change mask：1 通道

输出字段：

```text
features
quality
error_map
error_type
action
```

注意：旧版 smoke 曾使用 `index % 2` 构造 `target_view`，该字段不是真实标签，
已在 2026-07-18 的 schema v2 中删除，禁止用于正式训练。

已用 LEVIR-CD `test_256` 的 2 个样本完成构建和训练 smoke。

## 3. 真实模型和适配器验证

由于 SAM3、SimpleClick 和 Qwen 使用不同 Python 依赖栈，验证采用独立进程；没有强行把三个大模型塞进同一个 Python 环境。

### 3.1 Qwen3-VL CPU 验证

入口：`tools/qwen3vl_smoke.py`

结果：

```text
加载时间：约 5.1 秒
CPU RSS 峰值：约 982 MB
生成 JSON：成功
ActionParser：成功
```

### 3.2 Qwen3-VL GPU 验证

使用单张 L20：`CUDA_VISIBLE_DEVICES=7`

结果：

```text
PyTorch：2.9.1+cu128
可见 CUDA device：1
模型加载：6.893 秒
CUDA allocated：约 4,067 MB
CUDA reserved：约 4,174 MB
CUDA peak allocated：约 4,147 MB
```

### 3.3 Qwen3-VL → Environment GPU rollout

入口：`tools/rollout_smoke.py --agent qwen3vl`

结果：

```text
Environment rollout：成功
步数：1
Verifier/finish：成功
总耗时：14.188 秒
CUDA allocated：约 4,067 MB
CUDA peak allocated：约 4,147 MB
trajectory：outputs/qwen_gpu_rollout/trajectory.json
```

### 3.4 SAM3 GPU 验证

入口：`tools/sam3_smoke.py --device cuda --resolution 1008`

本地 checkpoint：

```text
/Data/wyh/CD-SegAgent/models/sam3/sam3.pt
```

验证了：

- text prompt
- box prompt
- processor state 转 NumPy mask
- SAM3 mask evidence 提取

资源结果：

```text
耗时：13.540 秒
CUDA allocated：约 3,525 MB
CUDA reserved：约 4,166 MB
CUDA peak allocated：约 3,909 MB
CPU RSS 峰值：约 7,488 MB
```

### 3.5 SimpleClick GPU 验证

入口：`tools/simpleclick_smoke.py --device cuda`

本地 checkpoint：

```text
/Data/wyh/CD-SegAgent/models/SimpleClick/cocolvis_vit_large.pth
```

验证了：

- checkpoint 加载
- external `prev_mask` 传递
- positive click
- 256x256 mask 结果回传

资源结果：

```text
耗时：7.167 秒
CUDA allocated：约 1,249 MB
CUDA reserved：约 1,450 MB
CUDA peak allocated：约 1,390 MB
CPU RSS 峰值：约 3,230 MB
```

## 4. SAM3 CPU 兼容性修复

为了让 OmniOVCD/SAM3 在 CPU 和 GPU 两种上下文都能工作，在独立的 `OmniOVCD` 仓库中提交了以下源码修复：

### 4.1 Decoder coordinate cache

文件：`OmniOVCD/sam3/model/decoder.py`

- cache 创建不再硬编码 CUDA。
- cache 记录 feature size 和 device。
- CPU/CUDA 切换后自动重建或迁移坐标 cache。
- 修复 cached coordinates 与 reference boxes device 不一致的问题。

### 4.2 Positional encoding cache

文件：`OmniOVCD/sam3/model/position_encoding.py`

- 原始实现预计算时固定 `device="cuda"`。
- 现在根据 `torch.cuda.is_available()` 选择 CPU 或 CUDA。

### 4.3 Geometry encoder

文件：`OmniOVCD/sam3/model/geometry_encoders.py`

- CUDA 使用 pinned memory 和 non-blocking copy。
- CPU 使用普通 tensor copy，避免没有 CUDA runtime 时调用 `pin_memory()`。

OmniOVCD 本地提交：

```text
c9b306e fix: make SAM3 decoder coordinate cache device aware
4746803 fix: allow SAM3 positional cache on CPU
816a1ec fix: avoid hardcoded CUDA decoder cache
c2f6115 fix: support SAM3 geometry encoding on CPU
```

OmniOVCD 中原有的用户修改没有被覆盖，也没有把它们混入这些提交。

## 5. GPU 可见性问题定位

最初 Python smoke 显示 CUDA 不可用，但 `nvidia-smi` 后来确认主机有 8 张 L20。根因是两种执行上下文不同：

- 受限沙箱：看不到 `/dev/nvidia*`，Python 会得到 `cuda.is_available() == false`。
- 设备可见执行上下文：能访问 NVML 和 CUDA device，PyTorch 正常工作。

确认命令：

```bash
nvidia-smi -L
CUDA_VISIBLE_DEVICES=7 \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

新增单卡配置：

```text
configs/runtime_gpu_l20.json
```

内容要点：

- `cuda_visible_devices: "7"`
- `device_map: "auto"`
- 单 GPU smoke policy
- 记录受限沙箱可能误报 CUDA 的说明

## 6. 测试和验证清单

### 6.1 单元测试

命令：

```bash
PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  -m unittest discover -s tests -v
```

结果：

```text
13 tests passed
```

覆盖内容：

- action JSON/fence 解析
- 坐标和 box 转换
- Environment reset/step
- 隐藏 state 不泄露
- finish 接受逻辑
- trajectory 保存
- connected components 和 instance matching
- Qwen 消息格式
- SAM3 public processor API
- SimpleClick external prev mask forwarding

### 6.2 Verifier smoke

```bash
PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py --smoke --count 8 --epochs 2
```

结果：loss 从约 `5.3406` 降至约 `5.3378`，CPU RSS 约 `630 MB`。

### 6.3 真实 LEVIR-CD Verifier smoke

```bash
PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/build_verifier_samples.py \
  --dataset-root /Data/wyh/CD-SegAgent/OmniOVCD/dataset/LEVIR-CD/test_256 \
  --max-samples 2 \
  --output outputs/verifier_levir_smoke.npz

PYTHONPATH=. \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py \
  --samples outputs/verifier_levir_smoke.npz \
  --epochs 1 --batch-size 1 \
  --output outputs/verifier_levir_smoke.pt
```

结果：2 个样本、10 通道特征、1 epoch 训练成功，CPU RSS 约 `864 MB`。

## 7. Git 管理记录

Change-Agent 仓库的主要提交：

```text
2cf4ee2  初始 GT-free change-agent loop
da935f6  v0-v3 架构和验证文档
456002c  合并远端仓库基线
eb826f5  模型、Verifier、rollout、runtime smoke 工具
7b2597d  SAM3 processor adapter
60ca391  记录 Qwen3-VL CPU smoke
355b84e  真实 Qwen3-VL 接入 rollout environment
9863472  真实 adapter 和 Verifier 数据准备
d2002a1  单卡 GPU smoke、GPU 配置和 CUDA 资源记录
```

远端推送目标：

```text
git@github.com:HappynessI/Chang-Agent.git
```

Git 忽略规则确保以下大文件不进入仓库：

- `models/`
- 本地训练 checkpoint
- smoke 输出和日志
- Python cache

## 8. 当前边界和下一步

已经完成的是代码闭环、真实权重加载、独立适配器运行、单卡 GPU rollout 和 Verifier 训练准备。以下属于正式实验阶段，而不是基础代码缺失：

1. 用完整 LEVIR-CD/Change-Agent 数据集进行多 epoch Verifier 训练。
2. 对 point、box、finish 策略运行正式 ablation 和数据集级指标统计。
3. 将 SAM3、SimpleClick、Qwen 拆成服务或分阶段 pipeline，以便在单卡内完成完整双视图真实模型 rollout。
4. 在固定 GPU、固定随机种子和正式数据 split 下生成最终实验报告。

当前推荐策略是继续使用单卡 `GPU 7` 做短 smoke，正式大规模训练前再根据显存余量决定是否切换到多卡或服务化部署。

## 9. OmniOVCD 三样本可视化推理

按检查需求，从 LEVIR-CD `test_256` 中选择三个正样本：

```text
test_85_16.png：GT positive pixels 1535
test_20_15.png：GT positive pixels 5031
test_78_13.png：GT positive pixels 11936
```

使用单张 L20（`CUDA_VISIBLE_DEVICES=7`）运行 OmniOVCD/SAM3，模型可视化和 MMSeg visualization hook 同时开启。为了让每次推理的产物独立保存，`OmniOVCD/eval.py` 增加了：

- `--work-dir`：指定 config snapshot、runner log 和 results.txt 目录。
- 可用的 `--show_dir`：实际启用 `SegVisualizationHook` 并设置 visualizer save directory。
- `--wait-time`：补齐 `--show` 需要的显示等待参数。

OmniOVCD 本地提交：

```text
5d8830b feat: support isolated visualization output directories
```

推理结果：

```text
样本数：3
aAcc：96.08
mIoU：82.70
mAcc：95.98
mFscore：89.97
building IoU：69.70
```

每样本 building IoU：

```text
test_20_15.png：0.852889
test_78_13.png：0.760079
test_85_16.png：0.301231
```

完整产物目录：

```text
/Data/wyh/CD-SegAgent/outputs/omniovcd_levir_3sample_vis_20260717_145327
```

该目录保存了输入 T1/T2、GT、最终预测、T1/T2 semantic mask、change mask、MMSeg 合成可视化、resolved config、源码快照、runner log、完整 stdout/stderr、metrics、GPU 状态、artifact inventory 和 SHA-256 校验和。

## 10. 2026-07-23 CA_0723(5) 后续代码审查与 hardening

审查确认 Proposal v11 的生产路径不是“固定 Agent、只替换 Verifier”：
Verifier 同时负责诊断和动作规划，Environment 校验协议与安全门，Executor
确定性调用 SimpleClick/SAM3。runner 已停止初始化未使用的 hosted Agent，
launcher 和维护文档统一改称 controller/Verifier ablation。

本轮按用户确认范围实施，并将新运行协议标为 v11.1，避免与历史 v11
指标混淆：

1. episode 产物拆分 `execution_status` 与 `episode_outcome`，增加收敛率和失败
   manifest；CA_0723(5) 的历史指标不改写。
2. 分割子进程移除 API key/token/secret/password 环境变量；Bailian 错误信息
   完整脱敏，429/5xx/临时网络错误做有界指数重试。
3. 双时相对象 mask 均为空时启用 3x3 全图扫描，Verifier 选择
   Environment-owned tile，漏检通过 SAM3 box 执行并重新验收。
4. Proposal rollback 更新持久化 diagnosis，避免下一轮 candidate prompt 携带
   被拒绝区域的旧诊断。
5. 新 Proposal launcher 删除无效的 Agent、action retry、min-area 和 v10
   batch 参数；兼容 CLI 暂时保留，清理/回滚表见
   `doc/ablation_parameters.txt`。
6. action-scoped delta 聚合暂不改，只登记效率与局部伤害权衡。
7. 旧 replay 的 whole-component negative deletion 与当前 Executor 不一致，
   已标为 legacy 并要求显式 acknowledgement。

对应定向测试覆盖空 mask 网格、box candidate 方向、rollback diagnosis、
网络重试/脱敏、worker 环境隔离、失败日志和运行状态语义。计算型测试必须
继续通过 Slurm 在计算节点执行。

## 11. 2026-07-25 Proposal-local point 候选动作改造

固定 `test_20_15`、`qwen3.7-plus-2026-05-26`、non-thinking、`gpu46` 和
direct Bailian 网络路径完成修改前后对照。修改前 Job `46601` 的三个负点
候选均使用完整 `5390` 像素 T2 session mask 和固定半径 ROI，全部被回滚；
Verifier 最终保留初始 IoU/F1 `0.84638554/0.91680261`。

候选动作边界现执行以下 GT-free 策略：

1. runner 将 Staged Verifier 选中的 `region_id` 作为独立可信元数据传入
   Environment，不把 ROI 加进模型 Action JSON。
2. Environment 必须在 live proposal 中解析同一 ID，并校验 point 等于权威
   component seed；Executor 使用 proposal box 生成自适应 ROI。
3. SimpleClick `prev_mask` 只保留 ROI 内的当前 target mask；positive 只增、
   negative 只删，且写入均裁剪到同一 ROI。
4. 轨迹记录 region ID、proposal box/component area、effective ROI、完整/局部
   session 像素、local/nonlocal 改动和 proposal change ratio。
5. proposal-local composition mode 继续进入 action-scoped delta 聚合，碎片只
   产生一个全覆盖 candidate record，不会因审计标签变化退回逐碎片调用。
6. 单次 point 改动超过 proposal component 面积 `25%` 时在 Qwen candidate
   verification 前硬拒绝，避免语义误诊一次提交大范围改写。

中间 Job `46634` 暴露了该安全门的必要性：Qwen 把 `r3` 的 265 像素 removal
误判为纯背景并提交，离线 IoU 降到 `0.81714081`。Slurm Job `46637` 的离线
诊断确认点击所在 192 像素 removal component 全部为 GT positive；这不是
SimpleClick 越界，而是模型语义误判。新增 gate 使用运行时可得的
`265/834=31.8%` proposal ratio 将其 fail closed，不使用 GT。

最终 Job `46641`（post3）在 `3:58` 内 `COMPLETED 0:0`，stderr 为空。三步
均使用正确 `r2/r0/r1` proposal-local session，ROI 外改动均为 0：第一步
`1262/1057` 由新 hard gate 拒绝；后两步 `64/1390`、`130/1116` 进入聚合后
candidate audit 并由 Verifier 拒绝。最终仍保留初始 IoU/F1
`0.84638554/0.91680261`，没有发生 post2 的回退。

Slurm 测试最终 attempt 3 为 Jobs `46638/46639`：定向测试全部通过，完整
回归 `184` 项全部通过。测试与实验产物分别位于：

```text
outputs/CA_0725(3)-point-action-logic-tests
outputs/CA_0725(2)-point-action-sample/baseline-qwen37-no-thinking
outputs/CA_0725(2)-point-action-sample/post3-qwen37-no-thinking
```
