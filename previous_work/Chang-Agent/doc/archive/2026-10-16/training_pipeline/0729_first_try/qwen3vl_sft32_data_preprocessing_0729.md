# Qwen3-VL-2B SFT32 数据预处理流程（0729）

## 1. 用途与结论

本文记录 Qwen3-VL-2B 单卡 LoRA smoke 所用 32 条数据的实际构造流程、数据契约、校验结果和边界。实现入口为 `tools/build_qwen3vl_sft_smoke.py`。首次尝试与成功重试的数据目录分别为：

```text
/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CA_0729(1)-qwen3vl-sft32-smoke/data
/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CA_0729(2)-qwen3vl-sft32-smoke/data
```

这批数据用于验证以下链路能否完整运行：

```text
LEVIR-CD 原图
  -> 确定性 patch 选择
  -> 合成 FN/FP candidate mask
  -> 四候选 select-stage 输入
  -> LLaMA-Factory ShareGPT 数据
  -> LoRA SFT / 同集生成 / 独立评分
```

它是刻意构造的同集 overfit/pipeline smoke，不是正式泛化训练集。raw GT label 图没有直接输入模型，但 candidate mask、proposal geometry 和 mask facts 由 GT 派生且对模型可见，候选文本也存在明显 shortcut。因此，高同集指标只能证明训练与协议链路可工作，不能证明模型学会了遥感视觉判断。

截至 2026-07-29，CPU prepare Job `49535` 已成功完成首次数据构建和结构校验。GPU
smoke Job `49550` 随后已在 `gpu16` 获得单张 A800 80GB，但在训练前因 worker 强制
torchrun 而提交环境的 `PATH` 中没有该入口失败；数据、依赖、单卡和 BF16 preflight
均已通过。首次 run 的失败审计保留。启动修复后的 clean retry prepare Job `50128`
在 `outputs/CA_0729(2)-qwen3vl-sft32-smoke` 重新确定性构建了带新绝对图像路径的
32 条数据；GPU smoke Job `50129` 随后完成 train/eval/predict/scorer 并通过协议门。

## 2. 输入来源

构造器只读取：

```text
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/train_original/A
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/train_original/B
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/train_original/label
```

约束如下：

- `A`、`B`、`label` 必须按同名 PNG 一一对应。
- 必须恰好有 445 张 train label。
- 每张原图必须为 1024×1024。
- label 灰度值 `> 0` 转成布尔变化掩码。
- 不读取 `val_original`、`test_original` 或 `test_256`。

构造过程复用现有运行时表示：

- `change_agent.adapters.stage_backends._proposal_overview`
- `change_agent.adapters.stage_backends._stage_prompt`
- `change_agent.state.ChangeState`

这样可减少训练输入与 verifier `select` stage 的格式漂移。不过前两个是内部函数；以后修改 runtime prompt 或 overview 时，必须同步重建并重新校验数据。

## 3. 原图切分与 patch 索引

每张 1024×1024 原图固定切成 4×4 个 256×256 patch，不做随机 crop、缩放或增强：

```text
445 parent scenes × 16 patches = 7120 patches
```

patch 标识和裁剪框为：

```text
sample_id = train_<parent_index>_r<row>c<column>
crop_xyxy_exclusive = [column*256, row*256, (column+1)*256, (row+1)*256]
```

每条 `PatchRecord` 保存 parent、网格位置、GT 变化像素数及对应 T1、T2、label 路径。实际 manifest 统计为：

- 总 patch：7120
- empty patch（`gt_pixels == 0`）：3953
- non-empty patch：3167

manifest 的 non-empty 统计使用 `gt_pixels > 0`，而 FN 采样资格门槛是 `gt_pixels >= 64`；正式扩展时应把“非空统计”和“合格 positive 数量”分开记录。

## 4. 确定性选择 32 条样本

固定 seed 为 `729`，选择：

- 16 个 `gt_pixels >= 64` 的 positive patch，用于构造 FN；
- 16 个 `gt_pixels == 0` 的 empty patch，用于构造 FP；
- 32 个 patch 来自 32 个不同 parent scene。

选择不是简单随机抽样：

1. 为池中每条记录生成 seeded 随机 tie-break 值。
2. 按 `(gt_pixels, tie-break, parent_index, row, column)` 排序。
3. 在排序后的池中取 16 个等频区间的中心位置，以覆盖变化面积分布。
4. 每个位置选择距离最近且 parent 尚未使用的记录。
5. positive 使用 seed `729`；empty 使用 seed `730`，并排除 positive 已占用的 parent。

该策略保证相同输入和代码下可重复生成同一集合，也避免 positive 全部集中在近似变化面积，但不保证目标 `region_id` 严格均衡。

## 5. False negative 合成

对每个 positive patch：

1. 从 GT 掩码中提取八连通分量，按面积从大到小排序。
2. 取最大连通分量。
3. 用 `SHA256(sample_id + ":fn")` 决定起始像素。
4. 从该点做确定性的八邻域扩展，邻居顺序也由样本哈希决定。
5. 删除像素目标数为：

   ```text
   min(max(16, component_area // 3), 1024)
   ```

   若原分量大于一个像素，至少保留一个原分量像素。
6. 构造：

   ```text
   predicted_change = gt_change AND NOT removed_chunk
   synthetic_error = gt_change AND NOT predicted_change
   t1_mask = gt_change
   t2_mask = all-zero mask
   ```

`removed_chunk` 是唯一受监督的 missing region。其他 GT 变化仍保留在 candidate mask 中，使样本表达“局部漏检”，而非完全空预测。

## 6. False positive 合成

对每个 empty patch，先强制确认 GT 为空，再生成由 sample ID 唯一确定的椭圆：

```text
width  = 36 + sha256(sample_id)[0] % 29   # 36..64
height = 36 + sha256(sample_id)[1] % 29   # 36..64
```

位置同样由哈希决定，并在 256×256 边界内保留外边距。构造：

```text
predicted_change = synthetic ellipse
synthetic_error = predicted_change AND NOT gt_change
t1_mask = all-zero mask
t2_mask = predicted_change
```

椭圆白块是唯一受监督的 present-but-wrong region。它适合稳定的链路 smoke，但不代表真实分割模型常见的边缘毛刺、粘连、碎片或大范围误检。

## 7. 四候选 proposal 构造

### 7.1 目标候选

先得到合成错误区域：

- FN：`gt_change AND NOT predicted_change`
- FP：`predicted_change AND NOT gt_change`

取最大八连通分量，得到 component area、seed 和 bounding box；box 向外 padding 6 像素并裁剪到图像范围。目标候选的 `audit_kind` 为：

- FN：`missing`
- FP：`present`

### 7.2 三个干扰候选

干扰框固定为 48×48，候选左上角来自 anchor `{8, 72, 136, 200}`。构造器：

1. 计算候选框与真实错误 mask 的重叠像素数；
2. 先按重叠从小到大、再按 sample hash 排序；
3. 选择三个两两不重叠的框；
4. 将其 `audit_kind` 设为 `scan`。

### 7.3 proposal 字段与编号

每个 proposal 包含：

- `region_id`
- `audit_kind`
- `change_mask_state`
- `temporal_difference_state`
- `box_normalized_1000`
- `component_seed_normalized_1000`
- `t1_mask_pixels` / `t2_mask_pixels`
- `component_area`
- `editable_seed_white`
- `allowed_actions=["box"]`

四个 proposal 按 `SHA256(sample_id + ":proposal-order:" + logical_index)` 排序后编号为 `r0`–`r3`，所以正确位置随样本变化。实际目标分布为：

| target ID | 数量 |
|---|---:|
| r0 | 3 |
| r1 | 8 |
| r2 | 9 |
| r3 | 12 |

四个 ID 都出现，但并不均衡。当前 validator 只要求目标至少分布到两个不同 ID，没有四类均衡硬门。

## 8. 模型输入与监督目标

### 8.1 模型可见图像

每条样本只生成一张 768×256 PNG：

```text
T1 RGB (256×256) | T2 RGB (256×256) | current change mask (256×256)
```

该 overview 使用运行时 proposal 风格叠加候选编号和几何信息。`sharegpt.json` 保存位于当前 `data/images/` 下的唯一绝对路径。

### 8.2 user prompt

user 内容由以下部分拼接：

```text
Numbered global proposal overview: T1 | T2 | current change mask
<image>
<由 _stage_prompt("select", payload) 生成的 proposal catalog 和 schema 约束>
```

payload 固定包含：

```json
{
  "target_class": "building",
  "proposal_catalog": ["four proposal records"],
  "max_selected_regions": 1,
  "schema": "persistent_region_screening_v11"
}
```

### 8.3 assistant target

assistant 只输出一个无 Markdown fence 的严格 JSON object：

```json
{"selection":{"region_ids":["r2"],"reason":"short observable reason"}}
```

FN 和 FP 使用各自固定的英文 reason 模板。监督明确不包含：

- `target_view`
- action 或 coordinate
- accept 或 quality score
- hidden chain of thought

因此当前 SFT 只训练 region selection envelope，不训练编辑动作、视图选择、坐标回归或完整闭环决策。

## 9. LLaMA-Factory 格式

`dataset_info.json` 注册：

- `levir_qwen3vl_sft32_train`
- `levir_qwen3vl_sft32_eval`

两者都指向同一个 `sharegpt.json`。每条记录包含：

- 唯一 `id`
- 一轮 `human` + `gpt` conversation
- user 内容中恰好一个 `<image>`
- `images` 中恰好一个绝对路径

所以当前 eval 是同 32 条数据上的生成评估，仅用于检查 overfit 和协议链路，不能视为 held-out 指标。

## 10. 落盘产物

### `sharegpt.json`

32 条训练记录，保存 prompt、assistant target 和图像路径。

### `dataset_info.json`

保存 LLaMA-Factory ShareGPT 列名和 role tag 映射；train/eval key 指向同一文件。

### `manifest.json`

顶层记录 schema version、seed、source split、采样统计和监督边界。逐样本记录：

- parent ID、patch row/column、exclusive crop box
- GT、candidate 和 synthetic error 像素数
- error type、正确 region ID、proposal 数量
- overview 路径
- 源 T1/T2/label SHA256
- assistant target SHA256

manifest 明确声明：

```text
test_data_used = false
raw_gt_label_image_model_visible = false
gt_derived_synthetic_facts_model_visible = true
```

### `validation.json`

保存结构校验状态、错误列表、样本/parent/class 统计、目标 ID 集合、核心 JSON SHA256 和 32 张模型可见图像的 SHA256。

## 11. Prepare 作业与实际结果

环境安装、单元测试和数据构建都通过 Slurm CPU 作业执行。正式入口：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent
bash tools/submit_qwen3vl_sft_prepare.sh
```

worker 依次执行：

1. 检查仓库、数据、模型、输出路径和 marker，拒绝覆盖已有正式 data。
2. 创建或校验独立训练环境。
3. 运行 4 个数据单测。
4. 运行 4 个 scorer 单测。
5. 用 seed `729` 构造并落盘 32 条数据。
6. 用 `--validate-only` 重新打开持久化文件，做第二次结构校验。
7. 只有 `valid=true`、32 条、16 FN + 16 FP、32 parent 全满足后，才原子写入 `.prepare_complete`。
8. 无论成功或失败都保留共享日志并发送 QQ 通知。

成功记录：

| Job | 状态 | 资源 | 节点 | 耗时 | MaxRSS |
|---|---|---|---|---:|---:|
| 49535 | COMPLETED (`0:0`) | 0 GPU, 2 CPU, 16 GiB | gpu18 | 3m52s | 约 685 MiB |
| 50128 | COMPLETED (`0:0`) | 0 GPU, 2 CPU, 16 GiB | gpu14 | 6m12s | 1,311,312 KiB |

两个 run 的 32 张模型可见图像 SHA256 逐张一致。`sharegpt.json` 的 SHA256 不同是
因为其中保存的是各自 run 下的绝对图像路径；样本选择、类别、parent、target ID 和
图像像素内容保持一致。

`validation.json` 实际结果：

| 检查项 | 结果 |
|---|---:|
| examples | 32 |
| false negative | 16 |
| false positive | 16 |
| unique parent scenes | 32 |
| target region IDs | r0, r1, r2, r3 |
| image placeholder mismatches | 0 |
| unknown region IDs | 0 |
| errors | 0 |
| valid | true |

实际 synthetic error 面积范围为 69–2318 像素；GT change pixels 范围为 0–21145。

## 12. 结构级 validator 的检查范围

`build_qwen3vl_sft_smoke.py::validate_output` 检查：

- `sharegpt.json` 恰好 32 条，ID 非空且不重复；
- 每条只有一轮 human/gpt，role 和字段正确；
- `<image>` 数量与 `images` 数量一致且恰好为 1；
- 路径为绝对路径，位于当前 `data/images/`，文件存在且不重复；
- overview 可读取且尺寸为 768×256；
- target 可直接 `json.loads`，且只有 `selection.region_ids/reason`；
- region ID 只能是 `r0`–`r3`，每条只选择一个；
- prompt/target 不出现 action、coordinate、accept、target_view 监督；
- manifest 与 ShareGPT 的 ID、target ID 完全一致；
- 类别为 16 FN + 16 FP，32 个 parent 唯一；
- dataset key、source split、test/GT 可见性声明正确；
- target ID 确实被打乱到不止一个位置。

该 validator 是持久化结构和数据契约门，不等价于真实 tokenizer/processor 编码门。

## 13. 已知 shortcut 与边界

### GT 衍生信息可见

raw label PNG 没有直接输入，但以下内容由 GT 生成后进入模型输入：

- FN candidate mask
- `t1_mask` / `t2_mask` 派生像素 facts
- target proposal geometry 和 seed
- `missing` / `present` audit kind

因此“raw GT 不可见”不等于“无 GT 泄漏”。

### 文本 shortcut 很强

正确候选的 `audit_kind` 是 `missing` 或 `present`，三个干扰候选全是 `scan`；assistant reason 也只有两条固定模板。模型可以仅依据文本字段完成选择，不一定需要理解 T1/T2。

### Train/eval 同集

两个 key 指向同一个文件。高 region accuracy、exact match、ROUGE/BLEU 只能证明记忆和短程可学习性，不能报告为 val/test 泛化。

### 错误分布理想化

当前只有一个局部 FN 或一个规整椭圆 FP，不包含：

- mixed FP+FN
- 正确无需动作的 none 样本
- proposal 未覆盖真实错误
- 真实 SAM3/SimpleClick 的边界、粘连、碎片和工具失败
- 多目标选择或跨时相 target-view 决策

### 路径和 API 边界

ShareGPT 保存绝对路径，迁移 run root 后不可直接复用。overview/prompt 依赖 runtime 内部函数，代码变更后旧数据不会自动同步。

## 15. 扩展到正式 baseline 的必要改造

1. 用冻结的真实初始化模型和工具链生成 candidate/proposal；GT 只用于离线判断候选质量，不生成模型可见 facts。
2. 让正确候选和 hard negative 使用相同字段类型，消除 `missing/present` 对 `scan` 的直接文本答案。
3. 按 parent scene 划分 train/val/test，再切 patch，禁止同一原图跨 split。
4. 增加真实 FP、FN、mixed、none、proposal miss 和 tool failure。
5. 平衡 target position、错误类型、面积、候选数和 reason 表达。
6. 保留冻结的 zero-shot Qwen3-VL-2B 对照，在 held-out val 上报告协议、region selection、candidate quality 和闭环 mask 指标。
8. 版本化 prompt/overview schema；运行时表示变化时强制重建数据。

## 16. 相关文件

- 构造器：`tools/build_qwen3vl_sft_smoke.py`
- 数据单测：`tests/test_qwen3vl_sft_smoke_data.py`
- CPU prepare worker：`tools/run_qwen3vl_sft_prepare.slurm`
- prepare submitter：`tools/submit_qwen3vl_sft_prepare.sh`
- 训练配置：`configs/qwen3vl_sft_smoke_0729.yaml`
- GPU worker：`tools/run_qwen3vl_sft_smoke.slurm`
- 独立 scorer：`tools/score_qwen3vl_sft_smoke.py`
- 简版数据契约：`doc/QWEN3VL_SFT32_DATA_0729.md`
