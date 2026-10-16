# Change-Agent 完整 Pipeline 入门

> 核对日期：2026-07-24
>
> 项目目录：`/guisongxia01/pangchao/wangyihan/Chang-Agent`
>
> 本文以最近一次完整、有效的模型实验 `CA_0723(5)` 为实例，同时给出当前代码
> v11.1 的标准运行方法。

## 1. 先看结论

Change-Agent 处理的是双时相遥感建筑变化检测：

- 输入同一地点两个时间点的图像：T1 和 T2。
- 先用 SAM3 分别得到 T1、T2 的建筑 mask（掩码）。
- 再通过实例匹配得到初始 change mask（变化掩码）。
- Qwen Verifier 检查变化掩码中的区域，同时规划下一步修改。
- Environment 校验动作，Executor 调用 SimpleClick 或 SAM3 修改某一个时相的建筑
  mask。
- 修改后重新生成 change mask，再由 Verifier 判断候选结果是更好还是更差。
- 更好的候选才提交；更差的候选回滚。
- 完成整段无 GT 的 rollout 后，才读取标签计算 IoU、Precision、Recall 和 F1。

最近一次完整模型实验是：

```text
outputs/CA_0723(5)-bailian-proposal-atomic-v11-5sample
```

其中推荐解读的实验组是：

```text
qwen37-no-thinking
```

这组的五样本汇总 IoU 从 `0.67629040` 提升到 `0.70009134`，共执行 9 次动作，
接受 7 次、拒绝 2 次。三个 Slurm 作业都为 `COMPLETED`，对应 stderr 为空。

但要先建立一个重要认识：

> “作业成功完成”不等于“每个样本已经收敛”。

CA_0723(5) 中没有样本以 `verifier_authorized_finish` 结束。部分样本是用完三步动作
预算后停止，另一些样本是 Verifier 没有给出可执行动作。因此，这次结果证明闭环能够
安全地产生有效改进，不代表问题已经完全解决。

另外，CA_0723(5) 使用的是协议 v11；当前工作树已经进入 v11.1。最新目录
`CA_0723(6)-post-audit-hardening-tests` 是 v11.1 的测试结果，不是新的模型实验结果。
因此不能把 v11.1 的代码行为倒推成 CA_0723(5) 当时的行为。

## 2. 一张图理解完整流程

```text
LEVIR-CD
  ├── A/<sample>.png          T1 图像
  ├── B/<sample>.png          T2 图像
  └── label_cvt/<sample>.png  GT，只在 rollout 完成后评估时读取
             │
             ▼
  SAM3 分别分割 T1/T2 中的 building
             │
             ▼
  T1/T2 二值 mask → 连通组件 → overlap_presence 匹配
             │
             ▼
  未匹配实例的并集 = 初始 change mask
             │
             ▼
  Environment 枚举可审计区域 r0、r1、……
             │
             ▼
  Qwen Verifier：全局筛查 → 逐区域原子审计 → 诊断并规划动作
             │
             ▼
  Environment：解析坐标、检查动作和安全约束
             │
             ▼
  Executor
  ├── positive_point / negative_point → SimpleClick
  └── box                             → SAM3
             │
             ▼
  重建 T1/T2 实例匹配和候选 change mask
             │
             ▼
  Verifier 检查真实 mask delta（修改前后的像素差）
       ┌─────┴─────┐
       │           │
   better       worse / uncertain
   且通过门控      │
       │           ▼
       ▼          回滚
      提交          │
       └─────继续审计，最多 3 次动作─────┘
             │
             ▼
  保存 trajectory 和 rollout_complete.json
             │
             ▼
  此时才读取 label_cvt，离线计算每一步和最终指标
```

这条生产路径没有一个独立的“Agent 模型”。当前准确的架构名称是：

```text
Verifier planner → Environment → deterministic Executor
```

这里的 Verifier 既做质量检查，也做动作规划。因此三个模型组是
controller/Verifier 对照实验，而不是“固定 Agent、只更换 Verifier”的纯 Verifier
消融实验。

## 3. 先认识会反复出现的术语

| 术语 | 初学者解释 |
| --- | --- |
| T1 / T2 | 同一地点较早和较晚两个时间点的图像。 |
| mask（掩码） | 与原图同尺寸的二值图；白色表示目标像素，黑色表示背景。 |
| semantic mask | 某一类目标的整体掩码；本项目的查询类别是 `building`。 |
| change mask | 系统认为 T1 与 T2 发生建筑变化的位置。 |
| instance（实例） | 一栋或一组连通建筑对应的独立区域。 |
| connected component（连通组件） | 二值图中彼此相连的一团白色像素；项目使用 8 邻域连接。 |
| GT / Ground Truth | 人工标注的真实变化 mask。在线闭环不能看 GT，评估阶段才使用。 |
| Verifier | 视觉“审核员”；检查当前结果、诊断错误，并规划修改动作。 |
| Executor | 动作“执行器”；将点或框真正交给 SimpleClick/SAM3。 |
| Proposal | Environment 编号并提供几何信息的候选区域。模型选区域，不能随意发明几何。 |
| rollout / episode | 一个样本从初始化到停止的完整闭环过程。 |
| candidate | 执行动作后产生、但尚未正式提交的新状态。 |
| commit / rollback | 接受候选叫提交；拒绝候选并恢复上一个有效状态叫回滚。 |
| arm | 对照实验中的一个实验组。 |
| ablation（消融） | 固定其他条件，只改变一个因素来比较效果。 |
| checkpoint | 已保存的模型权重文件。 |
| Slurm | 集群作业调度系统；GPU/CPU 计算通过 `sbatch` 交给计算节点。 |
| Bailian / 百炼 | 本实验调用托管 Qwen 模型的服务。 |
| thinking | 模型是否显式使用额外推理预算；不保证开启后一定更好。 |

## 4. 本次 Pipeline 的输入、模型和路径

### 4.1 数据

数据根目录是：

```text
/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256
```

目录约定为：

```text
test_256/
├── A/          # T1
├── B/          # T2
└── label_cvt/  # 二值 GT
```

CA_0723(5) 和当前标准 v11.1 launcher 都使用以下五个样本：

```text
test_20_15
test_78_13
test_85_16
test_1_1
test_50_8
```

三类文件必须同名，例如：

```text
A/test_85_16.png
B/test_85_16.png
label_cvt/test_85_16.png
```

### 4.2 Python 环境和模型

| 用途 | 路径 |
| --- | --- |
| 主 runner、SAM3 | `/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python` |
| SimpleClick worker | `/guisongxia01/pangchao/wangyihan/segagent-env/bin/python` |
| SimpleClick checkpoint | `/guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth` |
| SAM3 checkpoint | `/guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt` |
| SAM3 BPE 词表 | `/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz` |

SAM3 初始化、SAM3 box 和 SimpleClick point worker 的单次超时均为 900 秒。
该上限覆盖共享存储冷缓存时的 checkpoint 加载；超时仍会保留有界 stderr/stdout
和调用目录，不能把冷加载超时解释为 mask 策略效果。

Qwen Verifier 通过百炼调用，不在本地加载 Qwen 权重。标准脚本通过
`tools/run_with_bailian_csv.py` 只把凭据注入主进程；分割 worker 会移除 key、token、
secret、password 等环境变量。

凭据文件只应该由 wrapper 读取。不要使用 `cat`、`head`、日志打印或复制粘贴查看它，
也不要把其中内容写进命令、文档或 Git。

## 5. 一个样本在系统内部到底经历什么

下面按 `tools/run_levir_change_agent.py` 的真实顺序走一遍。

### 5.1 输入检查与输出目录创建

标准 preflight、launcher 和 runner 会共同检查：

- T1、T2、GT 文件是否存在；
- 两个 Python 解释器是否可用；
- SimpleClick、SAM3 checkpoint 和 BPE 文件是否存在；
- 参数范围是否有效。

输出目录必须不存在。这样可以避免新实验静默覆盖旧实验。runner 创建的主要子目录有：

```text
predictions/
trajectories/
masks/
verifier_feedback/
logs/
tool_runs/
```

运行开始时会写 `.run_incomplete`。v11.1 正常完成后删除它；未捕获异常则保留它，并写
经过脱敏的 `failure.json`。

### 5.2 用 SAM3 做双时相初始化

对每个样本，runner 只先读取：

```text
A/<sample>.png
B/<sample>.png
```

然后用文本提示 `building` 分别对 T1 和 T2 做一次全新 SAM3 初始化，不读取缓存
mask。初始化在独立子进程中完成，产物保存在：

```text
tool_runs/<sample>/sam3_initialization/initialize_000/
```

这里能看到：

- `t1_image.png`、`t2_image.png`：送入初始化的图像；
- `t1_mask.npy`、`t2_mask.npy`：双时相建筑掩码；
- `report.json`：模型、参数、seed 和 worker 状态；
- `evidence/`：logits、置信度、presence/object score 等初始化证据。

`npy` 是 NumPy 数组文件，适合程序精确读取；普通 PNG 更适合人眼查看。

### 5.3 从两个建筑 mask 得到初始 change mask

`MaskPairProcessor` 先把 T1、T2 建筑 mask 拆成 8 连通实例，再使用：

```text
matching_mode = overlap_presence
overlap_threshold = 0.25
```

直观理解如下：

- 如果 T1 的某个建筑实例和 T2 某个实例有足够重叠，认为建筑仍然存在；
- 没有找到足够重叠对象的 T1 或 T2 实例，认为它可能发生了变化；
- 所有未匹配实例的并集形成 change mask。

这里的阈值 `0.25` 是重叠覆盖率门槛，不是最终 IoU 指标。实例匹配本身可能受配准偏差、
建筑粘连或分裂影响，所以初始 change mask 会同时包含：

- 真变化；
- 误报（FP）；
- 漏报（FN，通常表现为本应是白色却没有进入 change mask）。

### 5.4 Environment 建立 Proposal 区域

Environment 从当前 mask 和实例关系中建立编号区域，例如 `r0`、`r1`。每个区域带有：

- 精确的组件 mask；
- 归一化 box 和点；
- T1/T2 mask 占用情况；
- change mask 是白还是黑；
- 允许使用的动作。

归一化坐标范围是 `[0,1000]`。例如 `[500,500]` 表示图像中心附近，不是第 500
个像素。真正执行前，`ActionParser` 会统一转换为原图 pixel XY（像素坐标）。

Proposal 模式的重要安全边界是：

> Qwen 选择区域和动作，Environment 拥有并校验几何。

模型不能随意编造一个未经审计的点或框。

在历史 v11 中，如果 T1/T2 两个预测 mask 都为空，就没有 Proposal。当前 v11.1 为此
增加了确定性的 `3 × 3` 全图扫描网格；如果某个黑色 tile 被可靠判断为漏检，
Environment 使用该 tile 自己的 box 调用 SAM3。

### 5.5 Verifier 做全局筛查和逐区域原子审计

`qwen_staged + proposal` 的 v11 核心过程是：

1. 全局筛查所有区域，记录每个区域可能有什么问题。
2. 对每个选中的区域单独做一次 atomic audit（原子审计）。
3. 在一次区域回答中同时检查 T1、T2 的真实视觉状态、整个组件 mask、候选错误和
   checklist。
4. 要求局部审计明确确认、反驳或标记全局筛查提出的假设为不确定。
5. 从安全且可执行的诊断中优先选择较小的组件，规划下一动作。

“原子”表示一个区域的关键判断放在同一次结构化回答中，避免前一阶段说“这里像误报”，
下一阶段却无声地改成“这里完全正确”。

常见诊断包括：

- `false_positive_change`：不该被判为变化，却在 change mask 中为白；
- `false_negative_change`：本应是变化，却没有被标出；
- `none`：没有发现需要改的错误；
- `uncertain_region`：证据不足或结构化回答存在矛盾，按保守策略处理。

模型回答无效时，系统会进行有界重试。重试后仍无效的区域按 fail closed（失败即关闭）
原则处理：不能授权危险动作，也不能凭它宣布 `finish`。

### 5.6 动作怎样变成真正的 mask 修改

Verifier 可以规划四类动作：

| 动作 | 工具 | 作用 |
| --- | --- | --- |
| `positive_point` | SimpleClick | 在目标时相中加入点击位置对应的预测组件。 |
| `negative_point` | SimpleClick | 从目标时相中减去点击引导的错误区域。 |
| `box` | SAM3 | 在指定 box 内重新分割并只替换 box 内部。 |
| `finish` | 无 | 请求结束；仍需通过完成条件。 |

当前 Executor 对修改范围有明确限制：

- positive point 只合并包含点击点的预测连通组件；
- negative point 只删除 SimpleClick 给出的差异，并裁剪在确定性的 point ROI 内；
- box 只替换 box 内部像素，box 外沿用旧 mask；
- 工具不能静默重写远处区域。

### 5.7 安全门控、候选复核、提交与回滚

工具执行后先得到 candidate，而不是立刻覆盖有效状态。Environment 检查：

| 门控 | 当前标准值 | 意义 |
| --- | ---: | --- |
| change-mask 面积变化 | `≤ 0.25` | 一步不能让变化区面积跳变过大。 |
| ROI 外变化比例 | `≤ 0.10` | 修改应主要留在授权区域内。 |
| 目标时相 mask 变化比例 | `≤ 0.25` | 一步不能重写过多目标 mask。 |
| 连通组件数量变化 | `≤ 4` | 防止组件数量异常爆炸或消失。 |

通过硬门控后，Verifier 会看到：

- 上一个已接受状态；
- 本次实际执行的动作；
- 修改前后精确的 added/removed delta；
- 原始 RGB 局部证据；
- 上一轮持久化的错误诊断。

只有同时满足以下条件才提交：

```text
comparison == better
accept == true
所有 runtime 安全门控通过
```

其余候选全部保存在 trajectory 中以便审计，但 live state 回滚到上一个已接受状态。
被拒绝的同一动作不能在同一状态上原样重复。提交成功后，系统从新状态重新建立区域目录并
完整审计。

整个过程最多执行 `max_steps=3` 次工具动作。

### 5.8 何时停止

一个 episode 可能因为以下原因停止：

- Verifier 合法授权 `finish`；
- 达到三步动作预算；
- 初始 Verifier 输出持续无效；
- Verifier 没有给出可执行动作；
- 程序或外部服务真正失败。

v11.1 把两个概念拆开：

- `execution_status=completed`：程序正常走到了结果写出阶段；
- `episode_outcome`：算法层面是 `converged`、`budget_exhausted`、
  `verifier_invalid` 还是 `no_executable_action`。

所以判断结果时，不能只看 Slurm 是不是 `COMPLETED`。

### 5.9 GT 为什么直到最后才读取

对全部样本完成闭环并保存 trajectory 后，runner 先写：

```text
rollout_complete.json
```

其中 `gt_loaded=false` 是一个时间边界证明：写这个 marker 之前，决策链没有读取 GT。
之后 `evaluate_after_rollout` 才打开 `label_cvt/`：

1. 读取每个 step 保存的 change mask；
2. 和 GT 逐像素比较；
3. 把 `offline_metrics` 回写进 trajectory；
4. 生成 `per_sample_metrics.json`；
5. 更新 `run_manifest.md`。

这就是项目所说的 GT-free：在线诊断、动作和接受/拒绝不使用 GT；实验结束后的评估仍然
需要 GT。

## 6. 最近一次有效输出：CA_0723(5)

### 6.1 为什么选它

截至核对日期：

- `CA_0723(5)` 是最近一次包含完整模型 rollout、预测、轨迹和离线指标的实验；
- `CA_0723(6)` 只有 v11.1 hardening 测试，177 个回归测试和 78 个定向测试通过；
- CA_0723(5) 的三个作业 `44878`、`44879`、`44880` 均为 `COMPLETED`；
- 三个 arm 都有五样本 `rollout_complete.json`、`per_sample_metrics.json` 和
  `trajectory.json`；
- 三个正式实验 stderr 均为空。

历史输出的 `run_manifest.md` 标题仍写着 “three-sample”，这是遗留标题；其中的
`samples` 列表、rollout marker 和实际目录都明确包含五个样本。

### 6.2 三个对照组

| Controller / Verifier | Slurm Job | 耗时 | 动作 | 接受/拒绝 | selected IoU | selected F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.7 non-thinking | `44879` | 00:36:43 | 9 | 7 / 2 | **0.70009134** | **0.82359262** |
| Qwen3.7 thinking-256 | `44878` | 00:59:14 | 9 | 6 / 3 | 0.69462518 | 0.81979802 |
| Qwen3-VL-Plus | `44880` | 00:39:57 | 8 | 8 / 0 | 0.68465376 | 0.81281243 |

三组的共同初始 IoU/F1 是 `0.67629040 / 0.80688931`。

推荐 non-thinking 不是因为它“接受最多”，而是因为：

- 它的 selected IoU 和 F1 最高；
- 7 个被接受候选在事后离线评估中都提高了 IoU；
- 2 个被拒绝候选在事后评估中确实更差；
- Qwen3-VL-Plus 虽然接受了全部 8 个候选，却错误接受了 test20 的有害删除。

这也说明 thinking 不是“越多越好”的开关，接受率也不能替代结果质量。

### 6.3 推荐组的逐样本结果

| 样本 | 初始 IoU | selected IoU | 变化 | 动作 | 接受/拒绝 | selected step | 历史 v11 停止原因 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test_20_15` | 0.84638554 | 0.84989316 | +0.00350762 | 3 | 1 / 2 | 1 | `max_steps_reached` |
| `test_78_13` | 0.75787116 | 0.78081913 | +0.02294797 | 3 | 3 / 0 | 3 | `max_steps_reached` |
| `test_85_16` | 0.30658070 | 0.33580247 | +0.02922177 | 3 | 3 / 0 | 3 | `max_steps_reached` |
| `test_1_1` | 0.00000000 | 0.00000000 | 0 | 0 | 0 / 0 | 0 | `action_retry_exhaustion_without_state_change` |
| `test_50_8` | 0.00000000 | 0.00000000 | 0 | 0 | 0 / 0 | 0 | `initial_verifier_invalid` |

这是 v11 的旧 stop-reason 命名。v11.1 会把它们进一步映射为明确的 episode outcome。

### 6.4 用 test_85_16 走一遍真实闭环

`test_85_16` 最适合观察连续改进：

| step | 动作 | 归一化坐标 | 工具实际变化像素 | 接受 | IoU |
| ---: | --- | --- | ---: | --- | ---: |
| 0 | 初始化 | — | — | 初始状态 | 0.30658070 |
| 1 | T2 `negative_point` | `[980,549]` | 169 | 是 | 0.31854932 |
| 2 | T2 `negative_point` | `[890,988]` | 103 | 是 | 0.32037691 |
| 3 | T2 `negative_point` | `[412,796]` | 195 | 是 | 0.33580247 |

三个动作都是 SimpleClick 负点，记录的 ROI 外变化比例均为 `0.0`。每一步提交后，
系统都从新状态重新审计，而不是一次性连续删除。

最终：

- FP 从 `2963` 降到 `2515`，减少 448；
- FN 从 `156` 增到 `175`，增加 19；
- Precision 从 `0.31759558` 升到 `0.35096774`；
- Recall 从 `0.89837134` 降到 `0.88599349`；
- IoU 从 `0.30658070` 升到 `0.33580247`。

这说明系统主要删掉了误报，但也损失了少量真阳性。综合 IoU 仍逐步变好。第三步之后是
动作预算耗尽，并非 Verifier 宣布所有问题已经解决。

### 6.5 用 test_20_15 理解回滚

`test_20_15` 的第一步只修改 32 个像素，IoU 从 `0.84638554` 升到
`0.84989316`，因此提交。

后两步候选分别修改 223 和 336 个像素。Verifier 判断为 `worse`，系统拒绝并回滚。
它们的离线 IoU 分别只有 `0.81925555` 和 `0.81494855`，事后证明拒绝是正确的。

因此该样本有：

```text
initial step = 0
selected step = 1
last attempted step = 3
```

这里最容易犯的错误是把 `last` 当最终结果。生产输出使用 `selected`，拒绝候选不会进入
最终 live state。

### 6.6 汇总指标怎样理解

推荐组的 selected 汇总为：

| 指标 | 数值 |
| --- | ---: |
| IoU | 0.70009134 |
| Precision | 0.73022119 |
| Recall | 0.94434326 |
| F1 | 0.82359262 |
| TP | 17629 |
| FP | 6513 |
| FN | 1039 |
| TN | 302499 |

项目先把五个 selected prediction 的 TP/FP/FN/TN 相加，再计算汇总指标。这是
micro aggregation（微平均式汇总），不是五个样本 IoU 的算术平均。

常用公式是：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
IoU       = TP / (TP + FP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

从共同初始状态到推荐组 selected 状态：

```text
TP: 17662 → 17629
FP:  7448 →  6513
FN:  1006 →  1039
```

也就是说，本次提升主要来自大幅减少 FP，同时付出了少量 TP 损失。不能只报告 IoU，
还应该同时看 Precision、Recall、FP 和 FN。

## 7. 现在怎样阅读已有结果

先进入仓库并设置两个只读路径变量：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent

CA_RUN='outputs/CA_0723(5)-bailian-proposal-atomic-v11-5sample'
CA_ARM="${CA_RUN}/qwen37-no-thinking"
```

建议按以下顺序阅读。

### 7.1 看实验设计

```bash
sed -n '1,160p' "${CA_RUN}/experiment_manifest.md"
```

它告诉你节点、网络模式、样本、三个 arm 和协议版本。

### 7.2 看该 arm 是否完整

```bash
sed -n '1,240p' "${CA_ARM}/run_manifest.md"
sed -n '1,120p' "${CA_ARM}/rollout_complete.json"
wc -c "${CA_ARM}/logs/"*.err
```

历史 CA_0723(5) 的成功特征是：

- manifest 中 `status: success`；
- rollout marker 列出五个样本；
- `gt_loaded` 为 `false`；
- Slurm stderr 为 0 字节。

### 7.3 看整体和逐样本指标

```bash
sed -n '1,460p' "${CA_ARM}/per_sample_metrics.json"
```

这个文件同时包含每一步、initial、selected、verifier_best 和 last 的指标。

### 7.4 看某个样本为什么停止

```bash
sed -n '1,200p' \
  "${CA_ARM}/trajectories/test_85_16/episode_summary.json"
```

先看 summary，再决定是否打开体积更大的 `trajectory.json`。

### 7.5 看最终 mask

```text
predictions/initial/<sample>.png        初始 change mask
predictions/selected/<sample>.png       最终正式选择
predictions/verifier_best/<sample>.png  Verifier 记录的最佳状态
predictions/last/<sample>.png           最后一次尝试，可能被拒绝
predictions/<sample>.png                selected 的便捷副本
```

论文或汇报中的最终预测应使用 `selected/`，不要默认使用 `last/`。

### 7.6 看完整动作链

```text
trajectories/<sample>/trajectory.json
masks/<sample>/step_000.npy
masks/<sample>/step_001.npy
...
verifier_feedback/<sample>.json
tool_runs/<sample>/point_000/
tool_runs/<sample>/sam3_initialization/
```

`trajectory.json` 保存原始动作、解析后的像素动作、Verifier 回答、候选是否接受、安全门控、
工具证据和离线指标，是定位“为什么改、改了什么、为什么接受”的主证据。

历史目录中的 `invalid_agent_outputs.json` 是兼容旧命名；当前生产路径并没有独立 Agent。

## 8. 当前 v11.1 的标准运行方式

这一节用于发起一个新的正式实验，不会复写 CA_0723(5)。

### 8.1 管理节点上只做检查和提交

不要在当前管理节点直接运行：

- `python tools/run_levir_change_agent.py`；
- GPU smoke test；
- 模型推理、数据预处理或批量可视化；
- 长时间测试。

这些都必须通过 Slurm。管理节点只做 Git/路径检查、`scontrol`、`squeue`、`sacct` 和
`sbatch`。

### 8.2 先确认代码版本和未提交改动

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent
git status --short --branch
git log -1 --oneline --decorate
```

当前仓库存在用户未提交改动，不能用 `git reset --hard` 或直接切换分支清掉它们。正式
实验会在 manifest 中记录 commit、dirty 状态和源码指纹。

CA_0723(5) 的 v11 已归档为：

```text
stable-ca0723-5-bailian-proposal-atomic-v11
```

当前 `tools/submit_ca0723_atomic_v11_5sample.sh` 会主动退出，防止把 v11.1 代码误标为
历史 v11。新实验应使用：

```text
tools/submit_ca0723_atomic_v11_1_5sample.sh
```

若必须精确复现历史 v11，应使用独立 Git worktree 检出 stable tag，不能在这个脏工作树
上直接 checkout。普通新实验不需要做这一步。

### 8.3 检查数据、环境和模型路径

下面只检查存在性，不会读取凭据内容：

```bash
test -d /guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256
test -x /guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python
test -x /guisongxia01/pangchao/wangyihan/segagent-env/bin/python
test -f /guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth
test -f /guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt
test -f /guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz
test -f /guisongxia01/pangchao/wangyihan/wyh/bailian-apikey.csv
```

这些命令没有输出通常表示检查通过；可用 `echo $?` 查看上一条命令的退出码，`0` 表示
成功。

### 8.4 检查 Slurm 和重复作业

```bash
scontrol ping
squeue -u guisongxia
sinfo -N -o '%N %T %G %C %m'
```

注意：

- Slurm 用户固定是 `guisongxia`，不是路径中的 `wangyihan`；
- `scontrol ping` 必须显示 controller 为 `UP`；
- 若出现 `Error creating slurm stream socket` 或 controller `DOWN`，先处理控制面，
  不要反复提交或重建临时脚本；
- 先检查是否已经有相同实验，避免重复占用 GPU。

### 8.5 固定节点并做百炼网络探测

从可用节点中选择一个候选，例如 `gpuXX`。不要照抄历史的 `gpu46`，因为节点状态会变化。

如果此前证据表明 GPU 节点上的 localhost 代理不可用，而清空代理后 endpoint 能返回
HTTP 响应，先测试 direct：

```bash
NODE=gpuXX BAILIAN_NETWORK_MODE=direct \
  bash tools/probe_ca0721_bailian_node.sh
```

probe 与 CA_0726 live launcher 默认使用同一个工作区 CSV 路径；确需使用另一个本地
workspace 导出文件时可通过 `CSV=/absolute/path/to/export.csv` 覆盖。脚本只检查文件
存在性，并由凭据 wrapper 在子进程内加载，不打印 CSV 内容、API key 或 endpoint。

脚本会提交一个五分钟以内的小型 Slurm probe，并打印 Job ID、stdout 和 stderr 路径。
等待约 90–120 秒后检查：

```bash
squeue -j <probe-job-id>
sacct -j <probe-job-id> \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

再查看脚本打印的 probe 日志。成功日志应确认：

- Python 依赖可用；
- GPU 可见；
- 固定作业节点与请求的 `NODE` 一致；
- 工作区、解释器和模型文件存在；
- 管理节点和计算节点的不可逆代理指纹一致；
- endpoint 返回三位 HTTP 状态码；
- 使用 proxy 时，两端代理监听检查通过、指纹一致，且 endpoint HTTP 探测通过。

日志只打印不可逆指纹，不能打印代理明文或 API key。
核验后不要把 probe stdout/stderr 长期留在共享 `.slurm-logs`：需要保留时，待对应
正式实验的 child `logs/` 建立后归档为 probe 日志；确认归档后清理共享临时副本。

如果 direct 不通过，只有在 proxy 监听可用且指纹一致时，才能以
`BAILIAN_NETWORK_MODE=proxy` 重新 probe。最终正式作业必须使用刚刚 probe 通过的同一
`NODE` 和同一 network mode。

### 8.6 选择新 run index

v11.1 输出名是：

```text
outputs/CA_0723(<run-index>)-bailian-proposal-atomic-v11.1-5sample
```

例如准备使用编号 7 时，先检查目录不存在：

```bash
CA_RUN_INDEX=7
test ! -e "outputs/CA_0723(${CA_RUN_INDEX})-bailian-proposal-atomic-v11.1-5sample"
```

若目录已经存在，选择新的编号。不要删除或覆盖已有正式输出。

### 8.7 提交三个 arm

使用 probe 通过的节点和网络模式：

```bash
NODE=gpuXX BAILIAN_NETWORK_MODE=direct \
  bash tools/submit_ca0723_atomic_v11_1_5sample.sh "${CA_RUN_INDEX}"
```

脚本会一次提交三个独立作业：

```text
qwen37-thinking
qwen37-no-thinking
qwen3vl-plus
```

每个作业申请：

```text
1 GPU
8 CPU
24 GB 内存
3 小时时限
固定到 probe 通过的 NODE
```

关键实验参数为：

```text
query                         building
samples                       5 个固定样本
max_steps                     3
selection_policy              conservative_best
proposal_mode                 proposal
matching_mode                 overlap_presence
overlap_threshold             0.25
empty_mask_scan_grid_size     3
verifier_max_new_tokens       1536
verifier_retries              2
seed                          42
```

launcher 会打印三个 Job ID 和各自输出目录。把 Job ID 记在实验记录里。

### 8.8 监控，不要高频轮询

```bash
squeue -j <job1>,<job2>,<job3> \
  -o '%.18i %.28j %.2t %.10M %.6D %R'
```

未完成时按 90–120 秒左右的窗口检查，不要每几秒刷新。作业离开队列后使用：

```bash
sacct -j <job1>,<job2>,<job3> \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList
```

三个 launcher 作业无论成功或失败都会发送 QQ 完成通知，并把正式 stdout/stderr 归档到
各 arm 的 `logs/`。不要依赖计算节点 `/tmp` 回读日志。

### 8.9 完成后的验收清单

设定新输出路径：

```bash
CA_NEW_RUN="outputs/CA_0723(${CA_RUN_INDEX})-bailian-proposal-atomic-v11.1-5sample"
```

对每个 arm 检查：

```bash
CA_NEW_ARM="${CA_NEW_RUN}/qwen37-no-thinking"

test -f "${CA_NEW_ARM}/run_manifest.md"
test -f "${CA_NEW_ARM}/rollout_complete.json"
test -f "${CA_NEW_ARM}/per_sample_metrics.json"
test ! -e "${CA_NEW_ARM}/.run_incomplete"

sed -n '1,260p' "${CA_NEW_ARM}/run_manifest.md"
sed -n '1,160p' "${CA_NEW_ARM}/rollout_complete.json"
wc -c "${CA_NEW_ARM}/logs/"*.err
```

然后逐个检查：

```text
trajectories/<sample>/episode_summary.json
predictions/selected/<sample>.png
```

v11.1 的最终验收应回答三个不同问题：

1. Slurm 和 runner 是否完成？
2. 每个 episode 是收敛、预算耗尽、Verifier 无效，还是没有可执行动作？
3. selected prediction 的离线指标是否比 initial 更好？

三者都不能互相替代。

## 9. 输出目录速查

```text
<run>/
├── experiment_manifest.md
├── qwen37-thinking/
├── qwen37-no-thinking/
└── qwen3vl-plus/
    ├── run_manifest.md
    ├── rollout_complete.json
    ├── per_sample_metrics.json
    ├── logs/
    ├── predictions/
    │   ├── initial/
    │   ├── selected/
    │   ├── verifier_best/
    │   └── last/
    ├── trajectories/<sample>/
    │   ├── episode_summary.json
    │   ├── trajectory.json
    │   └── action_attempt_errors.json   # 仅在有无效动作时出现
    ├── masks/<sample>/step_XXX.npy
    ├── verifier_feedback/<sample>.json
    └── tool_runs/<sample>/
        ├── sam3_initialization/
        ├── point_XXX/
        └── box_XXX/
```

最小阅读顺序是：

```text
experiment_manifest
→ run_manifest
→ rollout_complete
→ episode_summary
→ per_sample_metrics
→ selected prediction
→ trajectory（需要追因时）
→ tool_runs（需要检查工具像素行为时）
```

## 10. 常见问题

### 10.1 `output already exists`

这是防覆盖保护。换一个新的 run index，不要删除旧正式输出。

### 10.2 `NODE must be set`

先用标准 probe 检查节点，再把同一节点写入 `NODE`。不能让正式百炼作业随机调度到未探测
节点。

### 10.3 Slurm controller 报错

若 `scontrol ping` 失败，停止提交。反复 `sbatch` 不会修复控制面，也不要把问题误认为
脚本文件损坏。

### 10.4 百炼 401/403

这是认证或非临时请求错误，v11.1 不会盲目重试。检查 wrapper、workspace endpoint 和
凭据是否按既有配置存在，但绝不能打印 key 或凭据文件内容。

### 10.5 百炼 429/5xx 或临时网络失败

v11.1 最多重试两次，并从 1 秒开始指数退避。若最终失败，查看脱敏后的
`failure.json` 和 arm 的 Slurm 日志。

### 10.6 `.run_incomplete` 还在

先用 `squeue` 判断作业是否仍在运行。若已经结束，再看：

```text
run_manifest.md
failure.json
logs/slurm-<job>.out
logs/slurm-<job>.err
```

不要把不完整目录当正式结果。

### 10.7 `rollout_complete.json` 中为什么仍是 `gt_loaded=false`

它证明 marker 写入时 GT 尚未加载。marker 写完后才执行离线评估，所以同一目录随后出现
`per_sample_metrics.json` 是正常设计。

### 10.8 `selected` 和 `last` 为什么不一样

`last` 可能是被拒绝的候选；`selected` 是沿已接受状态链按
`conservative_best` 得到的正式结果。报告最终指标时用 `selected`。

### 10.9 Slurm `COMPLETED` 为什么样本仍是 `verifier_invalid`

`COMPLETED` 只说明程序正常处理并记录了这种算法结果。Verifier 无法给出有效答案是
episode outcome，不一定是进程崩溃。

### 10.10 为什么 test_50_8 在 v11 完全没有动作

它的两个初始建筑 mask 都为空，v11 无法建立 Proposal，但 GT 中实际有 166 个变化像素。
v11.1 新增 `3 × 3` 黑区扫描正是为了解决这一结构性盲点。是否真正改善仍需新的 v11.1
模型实验验证，不能拿测试通过代替效果结论。

## 11. 最后用一句话复述 Pipeline

Change-Agent 先用 SAM3 在 T1/T2 中找建筑，通过实例匹配形成初始变化图；再由 Qwen
Verifier 在不看 GT 的前提下逐区域诊断并规划点或框动作，Environment 和 Executor
受限地修改 mask，候选只有在 Verifier 判定更好且通过安全门控时才提交，否则回滚；
全部 rollout 保存完成后，系统才读取 GT 计算并报告最终指标。
