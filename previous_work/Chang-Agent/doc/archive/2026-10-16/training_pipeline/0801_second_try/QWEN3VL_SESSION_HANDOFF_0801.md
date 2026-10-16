# Qwen3-VL v12.1 baseline 0801 会话交接

更新时间：2026-08-02（UTC）

目标仓库：`/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl`

用途：供新的会话窗口直接接手 0801 second try；本文记录的是实验结束并归档后的当前状态。

> 本文是当前会话交接入口。相同目录中的
> `QWEN3VL_TRAINING_DATA_HANDOFF_0801.md` 是正式训练集构建前的协议交接，仍有历史价值，
> 但其中“正式 v12.1 训练集尚未构建”等状态已经过期。实验完整细节以
> `baseline_0801.md` 和 `QWEN3VL_V121_BASELINE_DATA_0801.md` 为准。

## 0. 新窗口先读这一段

0801 已完成从 v1 到 v7 的少样本数据构建、SFT、3 样本闭环评测和失败轨迹分析。冻结的
v12.1 runtime 没有在数据迭代中被修改，训练与评测链路能够稳定运行，安全门能够拦截并回滚
有害动作；但是最终 v7 仍没有相对 base 获得任何 selected IoU/F1 提升。

最终 v7 在固定 `DEV-CORE-3-V1` 三样本上的结果是：

```text
base selected：IoU 0.74223120，F1 0.85204673
v7 selected：  IoU 0.74223120，F1 0.85204673
delta：        IoU 0，F1 0
accepted candidate：0
successful tool action：0
regional validation errors：7
```

因此当前结论不是“训练成功提升”，而是：

1. 工程闭环、数据合同、GT-free rollout、离线评分和安全回滚已经跑通；
2. v1 的大框假阳性退化已经消除；
3. v2–v7 都未超过 base，v7 也没有通过扩大到 e2e5 的准入门；
4. 继续简单堆叠或重复同类 checklist 样本的证据不足；
5. 下一阶段应先解决当前 Git/归档路径问题，再升级评测探索机制，最后才开始新一轮训练并验证
   2B 模型能力边界。

当前没有本 baseline 的 Slurm 作业在运行。2026-08-02 复核时 Slurm controller 为 UP，账号
`guisongxia` 下另有其他项目的运行作业；它们不属于本任务，禁止取消或修改。

## 1. 当前仓库和 Git 状态

### 1.1 权威代码状态

- 当前分支：`feat/local-qwen3vl-verifier`
- 当前 HEAD：`0a5300ddc1d738a64c5bc93db6b0069f5c250fa7`
- 同一 HEAD 也由交付分支 `exp/baseline-0801-v21-delivery` 指向
- 冻结 runtime 锚点：`491532f4622667e9fbf299c71aac228fa37f6a04`
- 最终 v7 数据工具提交：`0f234d9bb6734819cda553d47f4dac5846f26db2`

从 runtime 锚点到当前 HEAD，`change_agent/` 没有差异；后续提交只增加或修改训练数据
builder、配置、测试、Slurm 脚本、实验登记和文档。因此不要为了迁就新训练标签而改动冻结的
Verifier、Environment、Executor、parser 或 rollback 边界。

### 1.2 当前工作树不是 clean

用户已经把 0729/0801 文档按轮次移动到：

```text
doc/training_pipeline/0729_first_try/
doc/training_pipeline/0801_second_try/
```

Git 当前把这次整理显示为 7 个 tracked deletion 加两个 untracked 目录。0801 的三个移动后文件
与 HEAD 中原文件 SHA256 完全一致，说明它们是纯路径搬移，不是正文被修改；但这次目录整理
尚未提交。不要恢复、删除或把它与其他实验代码混在一个提交里。

这件事是下一次提交 Slurm 作业前的实际 blocker：当前三个标准 submitter 都要求
`git status --porcelain` 为空，dirty worktree 会直接退出。新窗口应先与用户确认目录整理的提交
边界，然后单独提交或完成这次移动。不要使用 `git reset --hard` 或 `git checkout --`。

建议新窗口首先执行：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl
git status --short --branch
git log -5 --oneline
git diff --name-only 491532f4622667e9fbf299c71aac228fa37f6a04..HEAD -- change_agent
scontrol ping
squeue -u guisongxia
```

最后一条命令中的 Unix/Slurm 用户必须是 `guisongxia`，不是路径名里的 `wangyihan`。

## 2. 当前系统到底由谁判断、由谁动作

“由 runtime 决定是否增删 mask”不表示 Verifier 删除了纠错能力。必须区分：

- Qwen Verifier 模型负责看 RGB、给出 T1/T2 语义状态、区域证据和 checklist；
- Python runtime 把这些输出与 Environment 的权威 mask/几何事实组合，生成受约束动作；
- Executor 修改 T1 或 T2 的预测 object mask，随后重新计算 change mask；
- candidate evidence 再审核本次 action-scoped delta，runtime 决定接受还是回滚。

```text
Qwen states/checklist
  -> runtime error_type/target_view/action/geometry
  -> Executor 修改 object mask
  -> candidate delta 审核
  -> better 时提交，worse/unsafe 时回滚
```

当前映射是：

| 诊断 | runtime 动作 | 几何来源 |
|---|---|---|
| `false_positive_change` | `negative_point` | Environment component seed |
| 普通 `false_negative` | `positive_point` | Environment component seed |
| scan 中的 `false_negative` | `box` | Environment scan tile box |
| 全部区域正确并完成覆盖 | `finish` | 不需要几何 |
| mixed、uncertain、目标时相不唯一 | 不授权动作 | fail closed |

模型不再自由编写动作类型或坐标，但完整 `StagedQwenVerifier` 仍会输出
`suggested_action`。执行后只有 `better` candidate 才能进入 accepted state。

### 2.1 不同阶段看到的图片不同

| 任务 | Qwen 的视觉输入 |
|---|---|
| standard audit | marked T1/T2/change overview；局部 T1/T2 RGB；局部 T1/T2 object mask、change mask、component geometry，共 7 张 |
| scan audit | 当前一个九宫格 tile 的 T1/T2 原始 RGB crop，共 2 张 |
| evidence-only repair | 权威 region 内的 T1/T2 原始 RGB crop，共 2 张 |
| candidate evidence | delta contour RGB、当前/之前的 object/change mask、新增/删除 delta 等，共 11 张 |

scan 只在 T1 mask、T2 mask 和 change mask 都为空时触发。mask 数组仍由 runtime 读取，Qwen
也会在文本 Environment facts 中看到 `change_mask_state=black`、像素计数、box 等结构化事实；
但 T1/T2 mask 和 change mask 不作为 scan 的视觉图片输入。每次调用只审核九宫格中的一个 tile，
不是把九个 tile 一次性都交给模型。

预测 mask 只是可出错的中间结果，不能称为 GT。训练构造可以使用 train split GT 做离线筛选，
但 raw GT 不出现在模型输入或 target 中；在线 rollout 决策路径也是 GT-free。

## 3. 已完成的 v1–v7 迭代

固定三样本主评测集是：

```text
DEV-CORE-3-V1
1. test_20_15
2. test_1_1
3. test_50_8
```

它是反复查看的 diagnostic set，不是无偏测试集，不能据此宣称 LEVIR-CD 泛化性能。

| 版本 | 数据规模 | SFT/评测 Job | 结果和主要发现 |
|---|---:|---|---|
| v1 | 39 train / 24 val | 51896 / 51903 | 大 scan box 引入约 6518 FP；IoU/F1 退化到 `0.3687664/0.5388303` |
| v2 | 48 / 24 | 52011 / 52015 | 消除大框退化并回到 base；accepted=0，过于保守 |
| v2 e2e5 | 同上 | 52035 | 五样本 `0.6762904/0.80688931`，7 regional errors；不能与三样本 aggregate 直接比较 |
| v3-action | 53 / 24 | 52143 / 52146 | 加入真实 transition 和 `delta_removed`；仍与 base 持平，1 regional error |
| v3-bgfp | 59 / 24 | 52148 / 52150 | 加背景 FP 删除对；仍持平，3 regional errors |
| v3-bgfp-scope | 59 / 24 | 52187 / 52218 | 修正 `target_class_only`；仍持平，2 regional errors |
| v4-residual | 65 / 24 | 52258 / 52259 | 加两个真实 residual 三联；仍持平，regional errors 恶化到 7 |
| v5-contrast | 69 / 24 | 52326 / 52351 | 首次执行 1 个 `negative_point`，但删错真实变化，被 hard gate 回滚；6 errors |
| v6-matched | 63 / 24 | 52428 / 52429 | 用等量“该删/不该删”对照；仍误删，回滚；5 errors |
| v7-weighted | 83 exposure / 63 unique / 24 val | 52434 / 52436 | 关键对照总曝光提高到 3 次；loss 降低但无动作、无提升，7 errors |

v2–v7 的最终 selected IoU/F1 都与 base 相同。v5/v6 的“最终没有退化”来自 hard gate 拒绝
有害 candidate，不代表模型学会了正确动作。v7 因没有可执行动作而保持原 mask，也不代表提升。

## 4. 最终 v7 训练数据

### 4.1 当前实际路径

实验输出已由用户归档到：

```text
outputs/CA_0801_archive/CA_0801(34)-qwen3vl-v121-action-data-v7-weighted-contrast
outputs/CA_0801_archive/CA_0801(35)-qwen3vl-v121-action-sft-v7-weighted-contrast
outputs/CA_0801_archive/CA_0801(36)-qwen3vl-v121-action-e2e3-v7-weighted-contrast
```

归档中的 `.data_complete`、`.training_complete`、训练摘要、merged model、评测摘要和 rollout
证据均存在。旧文档及 registry 中的 `outputs/CA_0801(34)` 等路径是实验运行当时的历史路径；
当前读取产物时需要在中间增加 `CA_0801_archive/`。

### 4.2 83 exposure 不等于 83 个独立场景

v7 有 63 条不同训练内容。10 条 matched-contrast 重点内容各额外复制两行，使它们每条总曝光
3 次，因此训练文件共有 `63 + 10×2 = 83` 行。train 的 11 个 parent 与 val 的 6 个 parent
没有重叠。

| 任务 | unique 内容 | train exposure | 含义 |
|---|---:|---:|---|
| standard audit | 25 | 37 | 审核已有或缺失 proposal，输出 states、RGB evidence、七项 checklist |
| scan audit | 9 | 9 | mask 全空时逐 tile 找漏检，只看 tile T1/T2 RGB |
| evidence-only repair | 8 | 8 | states/checklist 已合法，只重写不合格 evidence |
| candidate evidence | 21 | 29 | 审核动作新增/删除 delta 是否覆盖真实变化 |
| 合计 | 63 | 83 | 另有 24 条 val，不参与训练梯度 |

计入重复后的其他分布：

- semantic states：`background→building` 36、`background→background` 25、
  `building→background` 14、`building→building` 8；
- candidate：improvement 17、harm 12；
- audit kind：missing 16、present 29、`delta_added` 11、`delta_removed` 18、scan 9。

manifest SHA256：
`<hash omitted>`。

### 4.3 最终训练配置和模型

- base model：`/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- 2×A800，BF16 LoRA
- rank 16，alpha 32，dropout 0.05，target `all`
- vision tower 和 multimodal projector 冻结
- per-device batch 1，gradient accumulation 4，有效 batch 8
- 2 epochs，learning rate `3e-5`，cosine，warmup ratio 0.1
- seed/data seed 801
- v7：83 examples、22 optimizer steps、train loss `0.3760475462`
- adapter SHA256：`<hash omitted>`
- merged model SHA256：`<hash omitted>`

当前 merged model 路径：

```text
/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl/outputs/CA_0801_archive/CA_0801(35)-qwen3vl-v121-action-sft-v7-weighted-contrast/merged_model
```

## 5. 最终 v7 评测情况

Job `52436` 在 1×A800 上完成固定 e2e3：

| 指标 | initial | selected | 变化 |
|---|---:|---:|---:|
| IoU | 0.74223120 | 0.74223120 | 0 |
| F1 | 0.85204673 | 0.85204673 | 0 |
| TP | 4777 | 4777 | 0 |
| FP | 1239 | 1239 | 0 |
| FN | 420 | 420 | 0 |

动作统计：candidate 0、accepted 0、rejected 0、successful/tool action 0、finish 2、
action attempt/error 1/1。这里的 action error 是“Verifier 没有授权可执行动作”，不是
SimpleClick 或 SAM3 崩溃。

逐样本情况：

- `test_20_15`：FP613/FN254；8 个 region 中 7 个变成 `uncertain_region`，`plan=null`，
  `no_executable_action`；mask 未改变。
- `test_1_1`：仍有 FP626，却被错误 `finish`；IoU/F1 仍为 0。
- `test_50_8`：仍有 FN166，9 个 scan tile 后仍错误 `finish`；IoU/F1 仍为 0。

预先设定的 e2e5 准入要求包括：3/3 无退化、至少 1 个 successful tool action、至少 1 个
accepted candidate、至少 1 条 IoU/F1 同时严格提升、action error=0，且 regional errors 不能
恶化。v7 明显不满足，所以没有运行 v7 e2e5。

### 5.1 `max_steps=3` 没有转化为三轮探索

正式 e2e3/e2e5 都设置了 `--max-steps 3`，含义是每个样本最多允许 3 次 Environment
state/action step。它只是上限，不保证一定执行 3 轮。当前 runner 在初始 Verifier 输出
`finish` 或无法形成安全 plan 时立即结束，不会为了利用剩余预算再请求一次独立诊断或替代动作。

v7 e2e3 的实际情况是：

| 样本 | 初始 Verifier 行为 | 实际 mask-edit 迭代 |
|---|---|---:|
| `test_20_15` | 8 个 region 审核和 repair 后仍为 `plan=null`，立即 `no_executable_action` | 0 |
| `test_1_1` | 初始合法但错误地授权 `finish` | 0 |
| `test_50_8` | 9 个 scan tile 均诊断 `none`，随后授权 `finish` | 0 |

因此虽然理论预算是每个样本最多 3 step，本轮三个样本都只留下 1 个 trajectory step，真正修改
mask 的 tool step、candidate 和 candidate-evidence 调用全部为 0。增加 `max_steps` 本身不能解决
这个问题，因为 episode 在进入第一个 mask action 前已经提前终止。

### 5.2 `verifier_retries=2` 已使用，但它不是探索重试

`--verifier-retries 2` 在当前实现中表示每个 stage 最多 2 次总尝试，即：

```text
attempt 1：initial generation
attempt 2：repair generation
```

它不是“initial 后再重试两次”，也不是新的闭环 step。v7 e2e3 确实使用了这个参数：

```text
initial audit calls：18
repair audit calls：8
audit backend calls：26
```

其中 `test_20_15` 是 8 次 initial 加 8 次 repair；`test_1_1` 只有 1 次 initial；
`test_50_8` 有 9 次 scan initial。repair 只在 JSON/schema、evidence grounding、states/checklist
一致性等可由协议检测的问题上触发。`test_1_1` 和 `test_50_8` 的错误 finish 在没有在线 GT 的
情况下属于“结构合法的语义误判”，所以不会触发 repair。

当前框架因此缺少第三类预算：

```text
1. verifier_retries：修复单个回答的协议错误（已有）
2. max_steps：合法动作之后的状态迭代（已有，但本轮未进入）
3. finish/plan=null 后的探索性 verifier round（缺失）
```

用户已决定：下一阶段训练开始前先修改评测框架，给 Qwen 更多独立 verifier round、替代诊断和
试错机会。该改动必须保持在线 GT-free，并继续保留动作授权、locality hard gate、candidate 审核
和 rollback；不能为了用满预算而强制执行不安全动作。

## 6. 当前最重要的问题

### P0：归档后绝对路径已经失效

v7 `train.json` 的 `images` 字段仍记录归档前的绝对路径，例如：

```text
.../outputs/CA_0801(34)-qwen3vl-v121-action-data-v7-weighted-contrast/data/images/...
```

该路径当前不存在；实际图片在 `outputs/CA_0801_archive/...`。归档中的
`config_used.yaml`、`post_sft_summary.json`、旧文档和 experiment registry 也保留运行时旧路径。
因此：

- 归档结果可以作为只读证据查看；
- 不能直接使用归档中的旧 `train.json`/`config_used.yaml` 重跑训练；
- 推荐为下一轮建立新的不覆盖输出目录并重新构建数据，让 builder 生成正确绝对路径；
- 不要就地改写归档 JSON，否则会破坏已登记 SHA256 和实验可追溯性；
- 如果用户确实需要原样复现旧 v7，应先明确采用重建、受控 symlink 还是迁移 manifest 的方案。

### P1：模型的 states 与 checklist 经常自相矛盾

典型错误是模型先写：

```text
t1_state=background, t2_state=background
```

随后却把七项 checklist 全部写成 `pass`。当 Environment 的 change region 是白色时，这表示
“两期没有建筑变化”却又说“当前变化 mask 完全正确”，逻辑矛盾。冻结 parser 只能 fail closed
为 `uncertain_region`，因此没有安全动作。v7 的 `test_20_15` 有 6 个 region 仍表现为该模式。

另一个错误是已有白色变化区域时只报告 recall-side extent fail，形成不可能的“纯 false
negative”。这些都是语义学习问题，不是继续增加 JSON 词法 parser 可以解决的问题。

### P2：false finish 没有解决

- `test_1_1` 有 626 FP，模型仍授权 finish；
- `test_50_8` 有 166 FN，scan 仍授权 finish。

模型学会了安全地“不乱动”，但没有学会识别什么时候必须纠错。v2 恢复到 base 的主要原因正是
拒绝 v1 的危险大框，而不是学会了产生有益修改。

### P3：动作链打通过，但动作方向不可靠

v5/v6 均执行过 `negative_point`，但删除了真实变化：

- v5：减少 16 FP，同时删除 324 TP；
- v6：删除 360 像素真实 component，FN 从 254 增到 614。

两次都由 runtime hard gate 拒绝并回滚。安全性有效，但 candidate 从未被接受，闭环没有获得
正向收益。v7 则退回到完全没有 tool action。

### P4：降低训练 loss 没有转化为闭环提升

v6 loss 约 `0.4221`，v7 通过重复关键样本降到约 `0.3760`；但 regional errors 从 5 增到 7，
动作和 accepted candidate 都变成 0。显式复制训练行只让模型更容易拟合训练答案，没有证明其
能在未参与训练的图像上正确组合 states 与 checklist。

### P5：当前评测只能作为 diagnostic gate

主结论来自固定 3 条、长期反复查看的样本。它足以否决明显失败轮次，却不足以证明稳定泛化
提升。即使未来通过 e2e3，也必须再运行固定 e2e5，并最终增加与训练 hard case 无关的 held-out
样本，才能讨论“稳定提升”。

### P6：scan checklist 标签有一处口径不统一

当前 9 条 scan exposure 中，3 条真实漏检把没有白像素对应的 precision 项写成
`not_applicable`；6 条无变化 scan 则把这些项写成 `pass`。runtime 都能把后者解释为
`error_type=none`，但这与 prompt 中“零预测的 white-pixel precision checks 应为
not_applicable”的文字不完全一致。下一轮数据构建应先统一标签政策并增加 strict gate，避免小模型
同时学习两种表达。

### P7：尚未区分“数据课程问题”和“2B 能力上限”

现有实验只证明当前联合 target、2 epochs 和显式重复没有奏效。还没有完成一个干净的能力测试，
去回答 Qwen3-VL-2B 能否稳定学习：

1. RGB 的 T1/T2 building/background 四类组合；
2. semantic states 与 mask polarity 的一致性；
3. 七项 checklist；
4. candidate delta 的 better/worse 方向。

下一阶段应通过结构化 curriculum 或更强模型 matched control 把这两种原因分开。

### P8：评测把“最大三步”错误地变成了“首轮可提前终止”

当前 `max_steps=3` 只约束已经进入 Environment action loop 后的上限。错误 `finish` 和
`plan=null` 都会在首轮终止，所以模型没有机会通过第二次独立观察修正语义、尝试另一个 region
或生成替代动作。这会低估“允许试错后能否纠正”的能力，也使 candidate 阶段长期不可达。

下一版评测应把三个概念拆成独立、可登记的预算：

- `max_verifier_rounds`：同一 accepted state 上允许多少次独立语义探索；
- `max_stage_attempts`：单次 region 回答的协议 repair 次数，即当前 `verifier_retries`；
- `max_environment_steps`：实际动作、candidate、接受/回滚后的状态迭代次数。

对于 `finish`，可以要求额外确认或独立复核；对于 `plan=null`，可以允许替代 region/诊断探索。
具体策略必须预先固定，并记录每轮为什么继续或停止。在线逻辑不能读取 GT；GT 仍只能在 rollout
结束后离线评分。旧 v12.1 结果必须保留为 control，不能用新策略覆盖原实验目录或原指标。

### P9：训练覆盖两个 stage，但没有完整轨迹级监督

当前 83 次 exposure 已覆盖：

- `audit` 54 次，其中 standard 37、scan 9、evidence-only repair 8；
- `candidate_evidence` 29 次，其中 improvement 17、harm 12。

所以“v7 没有动作”不能简单归因于训练集中完全缺少 audit 或 candidate stage。真正缺少的是把
这些独立 SFT 行连接成完整 episode 的监督：

```text
actionable audit
  -> runtime 派生正确 action/target/geometry
  -> Executor 产生 action-scoped delta
  -> candidate_evidence
  -> better/accept 或 worse/rollback
  -> 下一 accepted state 再 audit
```

Qwen 的 loss 只监督当前行的 states/checklist 或 candidate states/evidence_quality，没有直接承担
“这个 audit 最终是否产生合法动作”“动作是否改善 mask”“下一轮是否收敛”的轨迹级结果。当前
candidate 样本虽然存在，但只要首个 audit 输出错误 finish 或矛盾 checklist，线上就永远到不了
candidate stage。v7 的实际链路正是：

```text
错误 all-pass / uncertain / finish
  -> plan=null 或 finish
  -> Executor 不运行
  -> candidate=0
  -> 29 次 candidate 训练知识没有被触发
```

这是一项合理原因假设，而不是已经证明的唯一原因。下一轮数据应在保持 stage-specific 输出
schema 的前提下，为相关行增加可审计的 trajectory/parent action chain，并机械验证每个 audit
gold 能派生预期 action/finish、每个 delta gold 能派生预期 better/worse、整条链能够到达下一状态。

## 7. 下一窗口建议路线

### 阶段 A：先恢复可提交、可复现的工作状态

1. 确认并单独提交用户的 `0729_first_try/`、`0801_second_try/` 文档目录整理；保留本文。
2. 再次确认 `change_agent/` 相对 runtime 锚点无差异。
3. 把 `outputs/CA_0801_archive/` 视为只读历史证据，不修改其 JSON、模型或日志。
4. 在新文档/registry 中区分“历史运行路径”和“当前归档路径”；如果更新 registry JSON，必须
   重建 XLSX 并运行：

   ```bash
   python experiments/build_experiment_workbook.py
   python experiments/build_experiment_workbook.py --check
   ```

   2026-08-02 复核时现有 JSON/XLSX 同步，`--check` 通过，工作簿 SHA256 为
   `<hash omitted>`；但其中部分输出路径仍是
   归档前的历史字符串。

### 阶段 B：训练前先升级评测探索机制

新窗口应先设计和验证一个独立的新评测 arm，不追改旧 v12.1 结果：

1. 保留当前 v12.1 commit、启动参数和归档结果作为 control；新逻辑使用新的版本号、实验 arm
   和输出目录。
2. 把 `max_verifier_rounds`、`max_stage_attempts`、`max_environment_steps` 分开配置和记账。
3. 对初始 `finish` 增加 GT-free 的确认轮；对 `plan=null` 增加替代诊断或替代 region 的探索轮。
4. 每轮保存 Qwen 原始输出、repair/探索类型、runtime diagnosis、plan、终止原因和已消耗预算。
5. 不降低 geometry authorization、locality hard gate、candidate benefit-without-harm 或 rollback
   安全门。
6. 先用 smoke 样本证明能够真正进入第二轮、能够产生 candidate、且不会强制危险动作，再把它
   固定为下一阶段训练前后的共同评测 harness。

这里只应增加“再次观察和安全试错的机会”，不能使用离线 GT 告诉模型 finish 是错的，也不能
自动把不确定区域变成工具动作。评测框架稳定后，训练前 base 与训练后模型必须使用完全相同的
新探索预算，避免把 harness 改善误记为 SFT 改善。

### 阶段 C：再做能力诊断，不立即堆 v8 同类数据

建议建立 train/val parent 隔离的小型 curriculum，对每一步分别验收：

1. 只学习 T1/T2 semantic states；
2. 在 states 稳定后学习 mask polarity 与 correct/FP/FN 的一致性；
3. 再学习七项 checklist；
4. 最后加入 candidate delta 的 improvement/harm 对照；
5. 用同一数据给 2B 与更强模型做 matched control，判断是否存在明显能力差距。

模型输出 schema 仍保持 v12.1。新的探索调度必须单独版本化，不能倒改旧 control；课程训练可以
分阶段组织 loss 或采样，但最终生成仍必须通过 strict parser 和 Environment 一致性门。

除单项 curriculum 外，下一轮数据设计还应把同一 action chain 的 audit、candidate 和 next-state
记录用稳定 trajectory ID 关联起来，增加完整链 receipt。candidate 数据不能只作为与 initial
audit 无关的孤立样本存在。

优先验证以下最小行为：

- `background/background + white change` 必须产生 precision-side fail，而不是 all-pass；
- `background/building + black change` 必须产生 recall-side fail；
- `background/building + 正确白 change` 必须 all-pass/finish，不能误删；
- 删除 background/background residual 的 delta 应为 better；
- 删除真实 background/building change 的 delta 应为 worse；
- 空 mask scan 中真实变化 tile 不能 finish，背景 tile 不能产生大框 FP。

只有这些能力在 train-like 和 parent-disjoint validation 上都稳定后，才值得提交正式 SFT。

### 阶段 D：下一轮正式训练与评测

1. 使用新的输出目录重新 build 数据，不复用失效的归档绝对路径。
2. builder 必须继续检查 strict JSON、enum、scope、prompt/image/target hash、parent split、
   raw-GT 不可见和 deterministic receipt。
3. 数据构建、测试、训练、批处理和评测全部通过 Slurm；管理节点只做编辑和轻量检查。
4. 训练可用 2 张 GPU，最多 3 张；若没有足够 GPU，停止迭代并向用户汇报，不改为管理节点直跑。
5. 确认训练稳定后使用 30 分钟监控窗口；确认评测稳定后使用 5 分钟监控窗口。
6. 长作业成功或失败都发送 QQ 终态通知，并记录 Job、耗时、关键结果和产物路径。
7. 使用阶段 B 冻结的新探索 harness，先同时跑 base 与训练后固定 e2e3；只有训练后相对同 harness
   的 base 满足动作、accepted、严格提升和无退化门槛后才跑 e2e5。
8. 每轮同步更新 `experiments/experiment_registry.json` 和由它生成的 XLSX，以及本文或新的维护文档。

不要把较低 train loss、tool 调用成功或 hard gate 回滚后的“最终没退化”单独当作提升证据。

## 8. 关键文件与路径

### 当前文档

```text
doc/training_pipeline/0801_second_try/QWEN3VL_SESSION_HANDOFF_0801.md
doc/training_pipeline/0801_second_try/baseline_0801.md
doc/training_pipeline/0801_second_try/QWEN3VL_V121_BASELINE_DATA_0801.md
doc/training_pipeline/0801_second_try/QWEN3VL_TRAINING_DATA_HANDOFF_0801.md
```

### 数据、模型和评测

```text
base model:
/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct

train source:
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/train_original

validation source:
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/val_original

diagnostic test:
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256

v7 data:
outputs/CA_0801_archive/CA_0801(34)-qwen3vl-v121-action-data-v7-weighted-contrast/data

v7 merged model:
outputs/CA_0801_archive/CA_0801(35)-qwen3vl-v121-action-sft-v7-weighted-contrast/merged_model

v7 evaluation:
outputs/CA_0801_archive/CA_0801(36)-qwen3vl-v121-action-e2e3-v7-weighted-contrast
```

### 训练数据和 Slurm 入口

```text
configs/qwen3vl_v121_curation_0801.json
configs/qwen3vl_v121_baseline_sft_0801.yaml
tools/build_qwen3vl_v121_baseline_data.py
tests/test_qwen3vl_v121_baseline_data.py
tools/submit_qwen3vl_v121_data_build.sh
tools/submit_qwen3vl_v121_baseline_sft.sh
tools/submit_qwen3vl_v121_post_sft_e2e.sh
```

submitter 会检查 clean worktree、重复作业、节点状态、输出目录不存在和完成 marker。不要绕过这些
检查，也不要直接登录没有属于自己作业的 GPU 节点运行实验。

### 实验登记

```text
experiments/experiment_registry.json
experiments/experiment_registry.xlsx
experiments/build_experiment_workbook.py
```

JSON 是可审计源，XLSX 是生成物；更新时必须两者同步。

## 9. 新会话可以直接使用的任务描述

```text
继续 /guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl 的 Qwen3-VL
v12.1 baseline。先完整阅读
doc/training_pipeline/0801_second_try/QWEN3VL_SESSION_HANDOFF_0801.md，检查仓库级
AGENTS.md、git status、Slurm、数据/模型/归档路径。保留用户已有的文档目录整理，不要修改冻结
runtime。先解决 dirty worktree 和 CA_0801_archive 导致的绝对路径失效，再设计能够区分
state-checklist curriculum 问题与 Qwen3-VL-2B 能力边界的下一轮实验。开始新训练前，先建立
一个单独版本化的探索评测 arm：区分 verifier protocol repair、finish/plan-null 后的独立探索轮
和实际 Environment action step；保留旧 v12.1 结果为 control，保持在线 GT-free、安全动作授权、
candidate 审核和 rollback。随后再设计带完整 audit→action→candidate→next-state 关联的训练数据。
任何训练/评测通过 Slurm，使用 2～3 张 GPU；新 harness 下 base 和训练后模型使用相同预算，正式
评测先 e2e3，通过既定门槛后再 e2e5。每轮维护 registry、文档、Git 和 QQ 终态通知。
```

## 10. 当前验收状态

- 已完成：0801 数据构建、v1–v7 SFT、固定三样本评测、失败轨迹、registry、归档和总结文档。
- 已完成：`baseline_0801.md`，现位于 `0801_second_try/`。
- 未完成：训练后模型相对 base 的稳定闭环提升。
- 已确认的评测框架问题：`max_steps=3` 是上限，但错误 `finish`/`plan=null` 会在首轮提前终止；
  `verifier_retries=2` 本轮已经用于 8 次 repair，但只修协议错误，不能提供独立探索。
- 已确认的训练覆盖情况：两个线上 stage 都有样本，但现有行是孤立的 stage snapshot，缺少完整
  audit→action→candidate→next-state 轨迹级监督。
- 当前不应宣称：v7 优于 base、低 loss 等于泛化提升、安全回滚等于模型学会纠错。
- 下一窗口的顺序：恢复干净且路径正确的实验入口 → 版本化并验证探索评测框架 → 构建轨迹关联
  的下一轮数据 → SFT → 在同一新 harness 下做 base/post-SFT 对照。
