# Chang-Agent 三样本闭环测试问题分析与解决方案

## 1. 分析对象与结论

分析对象是成功运行目录：

`outputs/change_agent_levir_gpu_smoke_20260719_030624`

固定样本为 `test_20_15`、`test_78_13`、`test_85_16`。在线闭环使用 fresh
SAM3、Qwen3-VL-2B Agent、SimpleClick/SAM3 工具、Qwen zero-shot Verifier，最多
3 步，最终采用 `conservative_best`。GT 只在 `rollout_complete.json` 写入后用于
离线评估，本文中的 IoU、FP/FN 和 oracle 对比都属于闭环后的诊断，不能反馈给
在线决策。

核心结论如下：

1. **工程链路成功，安全机制有效，但优化闭环没有真正改善最终结果。** 三个样本
   最终都选择 step 0，实际被接受的 refinement 数为 0。最终汇总 IoU 从上次的
   `0.69690657` 变为 `0.69744116`，差值来自 fresh SAM3 初始预测的变化，不能归功
   于 Agent 闭环。
2. **Agent、Verifier、工具三层同时失效。** Agent 共 10 次遗漏必需坐标，导致前
   两个样本没有工具动作；Verifier 的 6 次定位全部是整图，且分数对候选没有区分
   能力；SimpleClick 的单点更新是全局 mask 替换，前两个候选发生大面积漂移。
3. **回滚策略避免了明显退化，但也暴露出 Verifier 的漏选。** `test_85_16` 的
   step 1/2 是灾难性候选，被拒绝是正确的；step 3 的离线 IoU 从 `0.30658070`
   提升到 `0.38875878`，仍因 Verifier 分数保持 0 而被拒绝。
4. **当前测试只能证明“不会轻易破坏初始结果”，不能证明“闭环能够优化结果”。**
   下一轮工作的重点不应是增加步数或放松回滚，而应是先修复动作参数生成、候选
   局部性和 Verifier 的相对比较能力。

## 2. 量化复盘

### 2.1 样本级结果

| 样本 | 初始 T1/T2 前景像素 | 初始 IoU | 初始 Verifier | 无效 Agent 输出 | 工具动作 | 最佳生成 step/IoU | 最终选择 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| `test_20_15` | 0 / 5390 | 0.84638554 | 0.00 / false negative | 3 | 0 | step 0 / 0.84638554 | step 0 |
| `test_78_13` | 566 / 14186 | 0.75787116 | 0.85 / mixed error | 3 | 0 | step 0 / 0.75787116 | step 0 |
| `test_85_16` | 0 / 4342 | 0.30658070 | 0.00 / false negative | 4 | 3 | step 3 / 0.38875878 | step 0 |

汇总现象：

- 结构化动作失败 10 次，均为 point 缺 `coordinate` 或 box 缺 `box`。
- 只有 `test_85_16` 执行了工具；3 个 candidate 全部被回滚。
- 6 个 Verifier entry 的 `error_region` 全是 `[0,0,1000,1000]`。
- accepted refinement 数为 0，selected prediction 与 initial prediction 完全一致。
- 如果在闭环后用 GT 对每个样本生成过的 step 做 oracle 选择，汇总 IoU/F1 为
  `0.75206355`/`0.85848889`，高于实际的 `0.69744116`/`0.82175592`。这只说明
  “候选集合中存在更优结果但选择器没有识别”，不构成在线使用 GT 的理由。

### 2.2 `test_20_15`

T1 是基本没有建筑的林地，T2 出现多栋新建筑。SAM3 给出空 T1 mask 和 5390 像素
的 T2 mask，初始 change mask 与 GT 已高度一致，IoU 为 `0.84638554`。

Verifier 却给出 0 分和 `false_negative`，理由是“T1 mask 全黑，不能代表真实场景”。
这与图像内容和 change 结果不一致。离线错误中 FP=613、FN=254，错误像素的
70.7% 实际是 FP，并非以 FN 为主。错误诊断随后要求在 T1 做 positive point；Agent
连续 3 次只输出 action 类型而不输出坐标，最终安全停止。这里没有执行工具反而
保护了一个已经较好的初始结果。

### 2.3 `test_78_13`

初始 IoU 为 `0.75787116`。Verifier 同时输出 `quality_score=0.85` 和
`mixed_error`，定位仍为整图。0.85 已高于配置的 accept threshold 0.82，但运行时
只有 `error_type=none` 才可能 accept，因此该分数与错误语义缺少一致的标尺。

离线错误为 FP=3246、FN=430，88.3% 的错误像素是 FP。Agent 被要求执行 box，但
连续 3 次只输出 `{"target_view":"t1","action":"box"}`，没有 box 坐标，样本
同样以零工具动作结束。

### 2.4 `test_85_16`

这是本轮最能暴露闭环问题的样本：

- SAM3 在 T1 检出 0 个对象，在 T2 返回 21 个检测 mask，二值合并后为 19 个连通
  组件；T2 score 均值仅 `0.61458`，明显低于 `test_20_15` 的 `0.73047` 和
  `test_78_13` 的 `0.82109`。
- T2 场景里有大量屋顶状目标、车辆或临时结构，`building` 文本提示把许多非 GT
  建筑变化一起纳入。GT 有 1535 个前景像素，初始预测有 4342 个；FP=2963、
  FN=156，95.0% 的错误像素是 FP。
- Verifier 仍诊断为 `false_negative`，程序据此推导 `positive_point`。也就是说，
  工具方向与主要误差方向相反。

三个真实动作如下：

| step | 归一化点 / 像素点 | candidate 面积比例 | 离线 IoU | Verifier 分数 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `[420,600]` / `[107,153]` | 0.474716 | 0.00107326 | 0.0 | 分数未提升、面积跳变，拒绝 |
| 2 | `[427,427]` / `[109,109]` | 0.265015 | 0.00026458 | 0.0 | 分数未提升，拒绝 |
| 3 | `[500,300]` / `[128,76]` | 0.012772 | 0.38875878 | 0.0 | 分数未提升，拒绝 |

由于每次拒绝后都会回滚到初始状态，这三个候选是从同一个 4342 像素的 T2 mask
分别生成的。step 1/2 的 SimpleClick 输出分别膨胀到 31111 和 17368 像素；step 3
收缩到 837 像素，显著提高精度但损失召回。Verifier 对跨度极大的四个状态始终给
0 分、整图定位和 `false_negative`，说明它没有形成可用于排序的质量信号。

## 3. 根因定位

### 3.1 指标解释错误：本轮提升来自初始化，不来自闭环

`Trajectory.best_entry` 只在 accepted entries 中按 Verifier 分数选择。本轮没有任何
candidate 被接受，所以三个样本全部选择初始 step 0。将本轮汇总指标与上次直接
比较会造成“闭环略有提升”的错觉；实际上两次 fresh SAM3 初始 mask 略有差异。

此外，runner 只设置了 Python 和 NumPy seed（`tools/run_levir_change_agent.py:82-83`），
没有设置父进程与 worker 的 PyTorch/CUDA seed，也没有声明 deterministic 策略。
trajectory 中 `git_commit` 还是 `null`，因为 `git rev-parse` 依赖启动目录而没有显式
指定项目 cwd。这些都会削弱跨次结果可比性。

### 3.2 初始分割：通用 `building` 提示对遥感小目标存在语义过检

SAM3 public processor 已用 0.5 detection confidence threshold 过滤实例，但不同
样本的 score 分布差异明显。`test_85_16` 的低置信、多实例结果把屋顶状车辆和
临时结构大量视作建筑变化。当前 `t12_min_instance_area=0`、
`cd_min_instance_area=0`，所有小连通组件都会进入 change mask。

基于已存档实例做的**闭环后诊断消融**显示：

| 二次最小 detection score | `20_15` IoU | `78_13` IoU | `85_16` IoU |
| ---: | ---: | ---: | ---: |
| 0.55 | 0.858175 | 0.774137 | 0.408680 |
| 0.65 | 0.772727 | 0.797128 | 0.566463 |
| 0.70 | 0.569996 | 0.797128 | 0.630369 |

较高阈值能显著改善 `85_16`，但会损害 `20_15`，因此不能从三个样本直接硬编码
0.65 或 0.70。对实际 step-0 change mask 做连通域过滤时，最小面积 256 的离线
IoU 分别变成 `0.846836`、`0.794120`、`0.365967`，说明面积过滤值得在独立验证集
上调参，但它也无法单独解决 `85_16` 的大块语义误检。

### 3.3 Agent：需要编辑 T1/T2 mask，却看不到这两个 mask

Agent 输入在 `change_agent/adapters/qwen3vl_adapter.py:42-64` 只有 T1 原图、T2
原图和 current change mask；它看不到自己实际要编辑的 predicted T1/T2 object
mask。Verifier 虽然看五张图，但传给 Agent 的只是一段文本反馈。本轮 feedback
又全部是整图区域，因此 Agent 缺少可落到局部对象的视觉依据。

动作生成还把“选择 view/action”和“输出精确坐标”放在同一次自由文本生成中，
没有 JSON grammar 或 schema constrained decoding。retry 只把校验错误追加到原
prompt（`tools/run_levir_change_agent.py:261-279`）；greedy generation 在相近输入上
容易重复同一缺字段 JSON。增加 retry 次数不会从根本上解决问题。

### 3.4 Verifier：没有比较基准，却承担 candidate 排序

当前 Verifier 每次只看到 candidate 的五张图。它收到 previous score 和 previous
action 文本，但看不到 previous change mask、previous T1/T2 mask 或 candidate delta。
`score_delta` 是“新一次绝对分数减旧分数”，不是对两个状态做视觉配对比较。因此
它很难判断一次工具动作究竟改善还是破坏了状态。

本轮的具体失真包括：

- 高 IoU 的 `20_15` 得 0 分，较低 IoU 的 `78_13` 得 0.85 分，绝对分数未校准。
- `85_16` 从 IoU 0.00026 到 0.38876 的候选都得 0 分，样本内也无法排序。
- `20_15` 和 `85_16` 的错误主要是 FP，却被判断为 false negative，程序进而派生
  出相反方向的 positive point。
- 第二阶段 localization 返回整图仍通过 `_parse_region` 的语法校验；代码没有检查
  region 是否局部、是否与白色 change mask 或 candidate delta 满足诊断语义。
- Agent 和 Verifier 共享同一个 Qwen3-VL-2B 权重，节省显存但会产生相关错误，
  不能把同一模型的自我评价当作独立证据。

### 3.5 SimpleClick：名义上的点修正实际是全局 mask 替换

`SimpleClickAdapter.refine` 把 initial mask 作为输入，但直接返回完整 prediction；
`ActionExecutor` 随后用它替换整个目标时相 mask。单个 point 因而可以把 6.6% 的
mask 改成 47.5%，并非局部编辑。

当前安全门只有：Verifier 有效、分数提升、change-mask 绝对面积变化不超过 0.25。
面积门只能拦住 step 1；step 2 的面积变化约 0.199，虽未超过 0.25，离线 IoU 却
接近 0。它没有检查变化是否集中在点击/box 附近，也没有检查连通域或拓扑是否发生
异常全局变化。

### 3.6 可观测性不足

runner 在内存中的 `rollout_records` 保存了 `episode_stop_reason`，但最终没有把完整
记录持久化。`invalid_agent_outputs.json` 也是扁平列表，没有 loop step、attempt
index、对应 feedback、prompt hash 等信息。因此可以确认前两个样本重试耗尽，却
无法从最终文件精确还原每个无效输出属于哪一轮动作。

## 4. 解决方案

### 4.1 P0：先补齐可复现和审计，不改变模型决策

1. 新增每样本 `episode_summary.json`，持久化 stop reason、accepted/rejected 数、
   action retry 的 loop index/attempt index、原始输出、校验错误和 prompt hash。
2. 显式从项目路径执行 `git -C <repo> rev-parse HEAD`，记录 dirty diff 状态、模型
   文件 checksum、PyTorch/CUDA/Transformers 版本。
3. 父进程及 SAM3/SimpleClick worker 都设置 Python、NumPy、PyTorch、CUDA seed；
   明确记录是否启用 deterministic algorithms，而不是只写 `seed=42`。
4. 离线报告新增 action-valid rate、accepted-refinement rate、degenerate-localization
   rate、Verifier ranking regret 和 oracle-step 指标。oracle 只用于闭环后诊断。

验收标准：同配置复跑能解释所有差异；每个 stop 都有持久化原因；无效输出能映射
到具体 step/attempt；trajectory 的 commit 不再是 null。

### 4.2 P1：把 Agent 改成“离散决策 + 受约束定位”

1. Agent 视觉输入与 Verifier 对齐，增加带标签的 predicted T1 object mask 和
   predicted T2 object mask。它必须看到自己要编辑的状态。
2. 将一次生成拆成两阶段：
   - 阶段 A 只输出 `target_view` 和 `action`；
   - 阶段 B 只完成缺失的 point/box 参数。
3. 优先避免让 2B 模型自由生成数值坐标。系统从以下证据生成有限 region proposal：
   change-mask 连通域、SAM3 低置信实例、previous/candidate delta 连通域、Verifier
   error map。模型只选择 `region_id` 和正负动作，系统再把 region 转为中心点或 box。
   这属于“模型选择的、证据驱动的参数化动作”，不是恢复任意 runner fallback。
4. 若仍使用 JSON 数值，接入 constrained decoding/JSON schema；repair prompt 只问
   缺失字段，并回显无效 payload 和一个同 action 的最小合法例子。
5. 参数补全失败继续安全 no-op，不恢复自动大框。

验收标准：固定三样本 action schema 有效率 100%；当 Verifier 给出 actionable
feedback 时，不再因缺字段导致整样本零工具动作。

### 4.3 P1：Verifier 从绝对打分改为 pairwise candidate gate

candidate commit 不应依赖零样本绝对分数差。推荐拆成两个任务：

1. **Pairwise gate**：同时展示 previous 与 candidate 的 T1 mask、T2 mask、change
   mask，以及高亮的 delta mask，输出 `better`、`worse` 或 `uncertain`。commit 只
   依赖 pairwise 结论和硬安全门。
2. **Absolute stop head**：只判断当前状态是否足够好，可以 finish；它不承担相邻
   candidate 排序。
3. localization 不再完全自由生成 box。优先在 candidate delta 或可疑实例 proposal
   中选择；若仍返回整图，除非 delta 本身覆盖大部分图像，否则标记
   `localization_valid=false` 并重试。
4. 为 FP/FN 加程序一致性检查：false positive region 应主要覆盖当前白色 change；
   false negative region 应主要位于当前 change 外。诊断文本、error type、target
   view 与 region 不一致时不能派生工具动作。
5. 不把共享 Qwen 的单一结论视为充分证据。至少加入 mask area、SAM3 score、局部
   delta、时相 overlap 等独立 proxy；中期接入专门训练的 verifier head。

验收标准：在存档候选上能把 `85_16` step 1/2 排在 initial 之后，并识别 step 3
优于 initial；整图 localization 比例从 100% 降到 0；FP/FN 方向与离线审计在该
三样本上不再出现明显相反。验收使用 GT 仅做离线评分，不进入在线输入。

### 4.4 P1：把 point/box 工具变成真正的局部编辑

1. positive point：只合并 prediction 中包含点击点的连通组件，或只在选定 ROI 内
   更新；不能用整张 prediction 替换旧 mask。
2. negative point：从旧 mask 中移除点击组件或 ROI 内被工具判定为负的区域，保留
   其他组件。
3. box：只允许 box 内替换，box 外保持旧 mask。
4. 新增 locality gate：记录 target-mask XOR，检查变化像素落在 action ROI 外的比例、
   最大变化连通域、组件数变化和相对面积变化。绝对面积阈值 0.25 只能作为最后一
   道粗门。
5. 每个动作可生成少量 candidate（例如局部 merge、局部 replace），交给 pairwise
   gate 选择；不要通过增加 max_steps 盲目试错。

验收标准：单点动作的绝大多数变化位于约定 ROI；不再出现一次 point 把 mask 从
6.6% 扩到 47.5%；拒绝候选不会影响 live state。

### 4.5 P2：校准初始化，而不是硬编码本次最优阈值

1. 保留 SAM3 每实例 mask 与 score，构造多阈值/稳定性候选，而不是先永久合并成
   一个二值 mask。对跨阈值稳定的实例提高置信，对只在低阈值出现的实例降权。
2. 在独立验证集上联合调 `confidence_threshold`、`t12_min_instance_area`、
   `cd_min_instance_area` 和 overlap threshold。不能依据这三个测试样本选择 0.65、
   0.70 或面积 256。
3. 对遥感场景增加更明确的语义提示或候选集，例如永久建筑、屋顶、临时结构，
   再由时相一致性和专用 verifier 判别；避免单一 `building` 把车辆/临时棚屋全部
   当作目标。
4. 训练 verifier 时使用真实 SAM3/SimpleClick rollout candidate，而不仅是从 GT
   做腐蚀/膨胀的合成扰动。按地理区域/场景划分 train/validation/test，避免样本
   泄漏；运行时仍不读取 GT。

建议评价指标包括 pairwise accuracy、Spearman/Kendall 排序相关、quality
calibration error、error-type macro F1、localization IoU，以及最终 accepted action
对离线 IoU 的正/负影响比例。

## 5. 推荐实施顺序

1. **审计与复现**：episode summary、attempt 索引、git/seed/version 记录。
2. **Agent 可执行性**：增加 T1/T2 object mask 输入，两阶段动作和 region ID 协议。
3. **Verifier 可比较性**：previous/candidate pairwise gate、delta proposal、整图区域
   失效规则。
4. **工具局部性**：point/box 局部 merge 与 locality gate。
5. **初始化校准**：在更大验证集上做 score/面积/匹配消融。
6. **再跑闭环**：先复用这三个固定样本做回归，再扩展到至少 30--100 个独立样本
   统计稳定性，不在三个样本上决定正式阈值。

下一轮三样本回归至少应满足：

- 3/3 样本结构化动作可执行；
- degenerate full-image localization 为 0；
- accepted harmful candidate 为 0；
- 至少 1 个样本 selected step 优于 initial step；
- 汇总 selected 指标不低于同一次运行的 initial 指标；
- 在线链路继续保持 `gt_loaded=false`。

## 6. 不建议采用的“快捷修复”

- 只增加 action retries：当前近似 greedy 的错误会重复。
- 恢复自动大 box fallback：上次已经证明会稳定扩大错误。
- 将 `selection_epsilon` 调成负数或接受同分 candidate：会同时放行
  `85_16` step 1/2 的灾难性候选。
- 直接选择 last state：本轮 step 1/2 已证明最后状态不具备安全性。
- 根据三个样本硬编码 SAM3 score 0.7 或面积 256：存在明显样本依赖和测试集过拟合。
- 把闭环后的 GT 指标交给 Agent/Verifier：这会破坏 GT-free 评估边界。
- 在 Verifier 尚无排序能力时增加 max steps：只会增加 GPU 开销和错误候选数量。

## 7. 证据路径与代码定位

- 运行参数：`outputs/change_agent_levir_gpu_smoke_20260719_030624/run_manifest.md`
- 指标：`outputs/change_agent_levir_gpu_smoke_20260719_030624/per_sample_metrics.json`
- 轨迹：`outputs/change_agent_levir_gpu_smoke_20260719_030624/trajectories/*/trajectory.json`
- 无效动作：`outputs/change_agent_levir_gpu_smoke_20260719_030624/trajectories/*/invalid_agent_outputs.json`
- SAM3 初始化证据：`outputs/change_agent_levir_gpu_smoke_20260719_030624/tool_runs/*/sam3_initialization/initialize_000/report.json`
- Agent 视觉输入与 prompt：`change_agent/adapters/qwen3vl_adapter.py`
- Verifier 视觉输入、定位和评分：`change_agent/adapters/qwen3vl_verifier.py`
- candidate commit/rollback：`change_agent/environment.py`
- SimpleClick 全 mask 返回：`change_agent/adapters/segagent_adapter.py`
- retry 与 GT 隔离：`tools/run_levir_change_agent.py`

本文初版分析没有修改运行逻辑，也没有使用 GT 重新选择或覆盖本次成功输出；所有
阈值消融只用于解释问题和制定下一阶段实验方案。下述第 8 节记录后续实现。

## 8. 2026-07-19 第一轮修复记录

本轮只修改 Prompt、Verifier 输入/评分协议和 Agent 动作格式提示，不启动 GPU 闭环：

1. Verifier Prompt 明确定义四种时相关系：新增建筑、消失建筑、不变建筑和不变
   背景。只有新增/消失属于最终 change mask；不再使用“仅存在于一张图”作为未
   区分方向的判断依据。
2. Prompt 明确 predicted T1/T2 mask 是辅助模型输出而非 GT，空 T1 或空 T2 mask
   不自动代表错误；Verifier 必须优先评价最终 candidate change mask，并回看两张
   原图判断空 mask 是否合理。
3. candidate Verifier 固定接收 T1/T2 原图，同时接收上一有效状态的 T1/T2/change
   mask、当前 candidate 的 T1/T2/change mask，以及产生 candidate 的归一化动作。
   被拒绝候选之后仍从上一有效状态比较，不把 rejected candidate 当作基准。
4. 第一阶段只输出 `quality_score`、`progress_score`、`error_type` 和 `feedback`。
   `quality_score` 是 candidate 的绝对质量；`progress_score` 位于 `[-1,1]`，独立
   表示 candidate 相对上一有效状态的改善或退化。初始状态固定为 0。动作候选门
   使用 `progress_score`，旧 `score_delta` 仅保留为审计/兼容字段。
5. actionable error 的第二阶段定位同样看到 previous/candidate 成对 mask 和当前
   动作，只输出 `target_view` 与 `error_region`；运行时继续派生 suggested action。
6. Agent 正常 Prompt 根据 Verifier 的 `suggested_action` 动态注入唯一一个 point、
   box 或 finish JSON 格式示例。point 示例必含 `coordinate`，box 示例必含 `box`，
   finish 不含二者，并继续禁止模型输出 `coordinate_frame`。
7. action retry 现在同时回传上一条无效 JSON。若 point 缺少/写错 `coordinate`，
   或 box 缺少/写错 `box`，repair prompt 会保留原 `target_view` 和 `action`，给出
   同结构精确模板并要求用 `[0,1000]` 数值替换占位符。

相关实现：

- `change_agent/adapters/qwen3vl_verifier.py`
- `change_agent/adapters/qwen3vl_adapter.py`
- `change_agent/environment.py`
- `change_agent/state.py`
- `change_agent/verifier.py`
- `change_agent/trajectory.py`
- `tools/run_levir_change_agent.py`

本轮 CPU 单元测试覆盖成对视觉输入、四关系 Prompt、独立 progress 解析、动态单示例、
缺字段定向重试、previous-state 传递和回滚兼容。下一步应先用固定三样本做闭环回归，
再根据轨迹中的 `quality_score`/`progress_score` 与离线 IoU 进行审计。

## 9. 2026-07-19 第二轮闭环安全修复记录

本轮继续完成分析报告中优先级最高的四项工程修复，仍未启动 GPU 闭环：

1. **局部工具组合与 locality gate**
   - positive point 只合并 SimpleClick prediction 中包含点击点的连通组件；其他
     全局 prediction 组件不会进入 live mask。
   - negative point 只移除当前 target mask 中包含点击点的连通组件。
   - box 只替换 action box 内像素，box 外严格保留上一 target mask。
   - 每次动作记录 target-mask XOR、ROI 外变化比例、target-mask 变化比例、最大
     变化组件、动作前后组件数；环境增加对应硬拒绝门。
2. **Verifier localization 与 FP/FN 一致性**
   - near-full-image region 在 candidate delta 不够大时标记为退化定位。
   - false-positive region 必须覆盖足够的当前白色 change；false-negative region
     必须主要位于白色 change 外。
   - 定位失败不再立即结束，而是携带具体校验错误重试；所有 localization attempts
     和检查统计进入 Verifier evidence。
3. **Agent 可见 predicted T1/T2 masks**
   - `AgentObservation` 和 Qwen3-VL messages 新增当前 predicted T1/T2 object mask，
     并明确它们是待编辑的模型输出而非 GT。
4. **审计与复现**
   - 无效动作记录 `loop_index`、`attempt_index`、原始输出、错误、prompt SHA-256
     和时间戳。
   - 每样本新增 `episode_summary.json`，记录 stop reason、候选接受/拒绝数量、工具
     数量、无效尝试、耗时和最终选择。
   - trajectory 显式从仓库路径解析 commit/dirty；run manifest 增加 Python、平台、
     parent seed、模型元数据 checksum 和 deterministic-policy 状态。
   - 隔离工具通过 seed wrapper 在模型构造前设置 Python/NumPy/PyTorch seed，并将
     seed runtime 写入 worker report。

新增/更新的 CPU 回归测试覆盖 clicked-component merge/remove、box ROI preservation、
global point rejection、full-image localization retry、FP region consistency、Agent mask
输入、prompt hash、step/attempt audit、Git metadata、模型 metadata checksum 和父子进程
seed 记录。正式有效性仍需下一轮固定三样本 GPU 闭环及闭环后 GT 审计确认。

## 10. 2026-07-19 SimpleClick 会话与 initial mask 根因修复

复查第二轮实现后确认，局部组合和 locality gate 只阻止了整图 prediction 直接写入
live mask，但当时仍存在两个底层问题：每次 point 都新建 `Clicker`，且 SegAgent 的
`SegmentationModel.get_prediction(..., mask=...)` 实际忽略 `mask` 参数。因此该版本
不能称为真正的连续 SimpleClick 编辑。

本轮从会话协议和 predictor 调用链修复：

1. Environment 分别维护 T1/T2 的 point session initial mask 与已接受点击历史。
2. 每次隔离 worker 从 session initial mask 开始，按顺序逐次重放历史点击，再执行
   当前候选点击。这保留了隔离进程设计，同时等价重建 SimpleClick Clicker 会话。
3. 候选只有通过 progress、area、locality 和 Verifier 门后才提交点击；被拒绝的点击
   不会污染后续历史。
4. accepted box 会以 box 编辑后的 mask 重开对应视图的 point session；另一视图的
   session 不受影响。
5. `SimpleClickAdapter` 绕过会吞掉 `mask` 的 SegAgent 包装方法，直接调用底层
   `predictor.get_prediction(clicker, prev_mask=...)`，并设置 external-mask 所需的
   `click_indx_offset=1`。
6. worker report 与 action evidence 记录 session click count、accepted click history
   和 session initial-mask 像素数，便于闭环后审计。

local clicked-component composition 与 locality gate 继续保留为最终安全边界。CPU
回归覆盖 initial mask 的实际逐次传递、点击顺序重放、reject 不提交、T1/T2 会话隔离
以及 accepted box 重置会话；真实模型效果仍需下一轮固定样本 GPU 闭环确认。

## 11. 2026-07-19 Delta-effect Verifier 与重复候选修复

对 `outputs/change_agent_levir_gpu_closed_loop_20260719_173613` 的复盘确认：
`test_20_15` 的 149 个 candidate-added 白色像素被错误标成 false negative，
`test_78_13` 的逐像素相同候选先后得到 worse/invalid/better，而 verbose 六区域 JSON
仍可能在后部截断。本轮据此完成以下结构性修改：

1. 初始区域输出压缩为 `region_id -> [verdict,target_view]`，不再要求每区域自然语言；
   普通白色 change component 不允许标成 false negative，该标签必须来自
   `temporal_difference_missing` proposal。
2. 候选阶段只展示本次 candidate delta：新增像素聚合成一个 panel、删除像素聚合成
   一个 panel。Qwen 只输出 added/removed supported/unsupported/uncertain 效果标签。
3. `better/worse/uncertain` 完全由程序从效果标签推导，删除第二次 Qwen pairwise 自由
   判断；任何 harmful 标签即 worse，任何 uncertain 即 uncertain，全部 beneficial 才
   能 better。
4. Verifier 按 previous masks、candidate masks 和 action 的 SHA256 缓存候选裁决；
   Environment 同时禁止在未变化 live state 上再次执行完全相同的已拒绝动作。
5. replay challenge 按线上局部 point/box composition 重建 temporal candidate mask；
   Verifier 保持 compact schema，但默认输出预算提高到 1024，避免偶发格式漂移或
   retry correction 被截断。Git 审计新增 tracked diff 与 untracked contents 的
   `git_worktree_sha256`，避免再次只留下 `commit + dirty=true`。

CPU 全量回归通过；没有在未重新确认迁移后 GPU 配额与通知策略的情况下启动 GPU
任务。下一步应先 replay 上述 9 个存档候选，再跑固定三样本闭环，重点检查
`test_85_16` 的 T1-empty 语义是否仍被模型误解。

## 12. 2026-07-19 审查后 Delta 分量与 Replay 强校验

后续审查指出“每种极性一个聚合 panel”仍可能把空间上相离、语义上相反的候选变化
压成一个标签。现改为按 delta 连通分量生成最多三个独立 panel，并记录
`candidate_delta_pixels`、`covered/uncovered_pixels` 和 `coverage_ratio`；超过预算而
未覆盖的候选不调用 Qwen、不能进入 accepted state。

效果标签改为无歧义的 `added_true_change`、`added_false_change`、
`removed_false_positive`、`removed_true_change`、`mixed`、`uncertain`。程序只接受
全部有益的分量；mixed、uncertain 或有益/有害并存均保守拒绝。面积、locality 和
拓扑硬门前移到 Verifier 之前，上一轮 `test_78_13` 这类已确定越界的候选不再消耗
Qwen 推理。

候选缓存键扩展到原图、完整 previous/candidate masks、像素动作、query、Verifier
schema/model/generation 配置、proposal 与区域事实。轨迹逐步保存 T1/T2/change hash；
replay 沿线上 accepted-state 链重建候选并复用 matching 配置，任一 replay hash 与
线上记录不一致即终止，不能产生 Verifier 评价。Verifier 输出预算保持 1024。

第一次 GPU 回归 `change_agent_levir_gpu_closed_loop_20260719_203105` 暴露了新的初始
停止漏洞：三个样本均把六个 panel 全判为 true_change 后直接 finish，实际没有进入
delta 分支；其中 `test_85_16` 的六个 panel 只覆盖 54.8% 待审计像素。现对初始
`change OR temporal-missing` 像素同样记录精确覆盖率，覆盖不全时由程序定位最大遗漏
分量并输出 uncertain/box，禁止 finish。同时严格执行 compact schema：
true_change/uncertain 的 target_view 必须为 null，模型输出 t1/t2 时重试。

## 13. 2026-07-19 双视觉一致性门与 invalid baseline 安全停止

正式回归 `change_agent_levir_gpu_closed_loop_20260719_203719` 证明，程序推导只能消除
自由 pairwise 的随机性，不能修正 Qwen 对局部事实本身的反向判断：
`test_20_15` 将 189 个纯新增 FP 标成 `added_true_change` 并接受；
`test_78_13` 则将实际删除 135 个 FP 的有益候选标成 `removed_true_change` 并回滚。

候选效果现改为双视觉一致性：第一次保留 previous/candidate masks 与 mask overlay，
第二次隐藏 predicted masks，只展示干净 T1/T2 crop、精确 delta 二值图和 RGB 时相差。
两个调用对每个 region_id 的标签必须完全一致；不一致由程序改成 uncertain 并拒绝。
所有 mode、raw output、重试、标签和 agreement 均写入 evidence，缓存 schema 同步升级。

同时修复 invalid baseline 控制流：初始 Verifier 重试耗尽后，正式 runner 不再要求
Agent 继续探索，也不调用分割工具，而以 `initial_verifier_invalid` 安全结束并保留初始
mask，避免在没有有效基准诊断时提交局部候选。

## 14. 2026-07-19 RGB 时相事实判定与有益候选放行

`outputs/change_agent_levir_gpu_closed_loop_20260719_205825` 表明“双视觉抽象标签一致”
仍不是可靠的安全门。`test_78_13` 的候选删除了 135 个 false-positive 像素，离线 IoU
由 `0.75787116` 提升到 `0.76467070`，但两个分支都输出
`removed_true_change`，导致真实有益修改被一致地误拒绝。与此同时，另外两个样本的
初始 `true_change` 因携带无实际意义的 `target_view=t1/t2` 而耗尽重试并停止。

本轮不再要求 Qwen 在第二视觉分支理解“删除动作是否有益”这一抽象语义，而只判断
delta 对应 RGB 区域在 T1/T2 中分别是 `building/background/mixed/uncertain`。程序结合
`effect_kind=added/removed` 推导：确定的 T1/T2 状态相异表示真实时相变化，相同表示
无变化；新增真实变化和删除无变化区域为有益，新增无变化区域和删除真实变化为有害。
RGB 请求本身不暴露 action、effect_kind 或新增/删除统计，避免这些动作语义反向诱导
局部视觉分类。

mask-context 标签只用于审计，不再拥有否决权。它与 RGB 推导冲突、甚至连续产生无效
JSON 时，仍继续执行 RGB 时相判断；只要 RGB 状态明确，候选照常进入程序门控。RGB
本身为 `mixed/uncertain` 或无效时仍保守拒绝。这样既避免审计分支误杀有益候选，又不
放松 locality、面积、coverage、拓扑和 replay hash 等环境硬约束。

初始阶段则把 `true_change/uncertain` 的多余 target view 规范化为 `null` 并记录
`schema_warnings`，不再把无动作意义的字段错误升级为整轮 invalid。真正缺字段、非法
枚举、事实冲突或未覆盖区域仍按原安全策略处理。

## 15. 2026-07-19 全量 Delta 分批与初始状态程序推导

GPU 作业 `41377` 的输出
`outputs/change_agent_levir_gpu_closed_loop_20260719_135752` 证明上一版仍有两个前置
过滤器，使新的 RGB candidate Verifier 没有得到实际调用机会。`test_20_15` 的候选只
改变 9 个像素，离线 IoU 从 `0.84638554` 提升到 `0.84762580`，但 top-3 proposal 只
覆盖 8 个像素；`test_85_16` 也有 5 个尾部分量像素未覆盖。两者均在 Qwen 前 invalid。
`test_78_13` 则连续输出完整但语义不可能的初始 FN JSON，直接停止在 baseline。

本轮不降低 coverage 阈值，也不忽略单像素分量。candidate proposal 保留所有 added/
removed 连通分量，`max_delta_regions=3` 仅作为每次模型调用的 batch size。Verifier 按
稳定 region_id 分批调用并汇总，所有批次覆盖总和必须等于完整 delta；任一 decisive
RGB batch 无效仍拒绝整个候选。这样既允许 3+1 分量的有益候选进入语义判断，也不会
放过小型有害尾部修改。

初始阶段同步取消 Qwen 自由输出 `true_change/FP/FN/target_view`。模型只看干净 T1/T2
RGB、精确 audit component 和 RGB difference，并输出两时相的 building/background/
mixed/uncertain。程序结合 present/missing geometry、predicted mask 覆盖和状态方向推导
verdict、target view 与 positive/negative/box。普通已有白色区域由协议保证不会被抽象
标签直接判成 FN。

重复 rejected action 的硬拦截继续保留，但 trajectory history 现在保存完整归一化
`rejected_action` JSON，retry prompt 将其列为 forbidden，并要求改变 action type 或
coordinate/box、转向其他未解决区域。该修复避免把坐标合法但已失败的动作误当成普通
格式错误反复修复。

## 16. 2026-07-19 全量分批实测与精确组件锚点

GPU 作业 `41396`（`outputs/change_agent_levir_gpu_closed_loop_20260719_143908`）证明
上一轮的全量分批机制已经在线生效：`test_85_16` 的 96 个 candidate delta 像素被拆成
13 个连通分量、五个 batch，`covered_delta_pixels=96`、`coverage_ratio=1.0`。因此当前
主要问题已经从“候选没进入 Verifier”转为“初始动作定位和 RGB 局部语义仍不够精确”。

三个样本第一次 negative point 的像素坐标分别为 `[115,147]`、`[129,151]` 和
`[96,61]`，都来自 padded proposal box 的模型自由选点，未命中对应连通分量，最终
`changed_pixels=0`。更重要的是，三个最大 initial present component 与离线 GT 的重合
率分别为 90.79%、70.95% 和 92.98%，Qwen 却均输出 background/background。若只把
自由坐标换成精确点，而不提高局部视觉对应性，会更准确地删除真实变化，不能接受。

本轮同时收紧动作几何和改善视觉证据：每个 proposal 保存 Environment 计算的分量 seed
像素及其 `[0,1000]` 坐标；point 型反馈把该 seed 作为退化 `error_region`，Agent 不再
自行在 padded box 内选点，而是复制精确锚点。T1/T2 crop 在分量外侧绘制黄色一像素
轮廓，明确要求判断轮廓内的原始 RGB。轮廓不覆盖被审计像素，并且绝对差分图仍从未
标注的 raw RGB 计算，避免人工颜色成为时相变化证据。box 型不确定动作继续使用完整
padded box。

作业 `41396` 中 `test_85_16` 的 96-pixel candidate 离线 IoU 从 `0.30658070` 提升到
`0.31390233`，但其中恰好 48 像素为 GT change、48 像素为 FP，并非“全部有益候选”。
因此不能为放行它而降低 mixed/uncertain 的安全门；下一轮应首先验证黄色轮廓能否修正
initial component 语义，以及精确锚点能否在语义正确时产生非空局部编辑。

## 17. 2026-07-19 精确错误动作与初始全量分批修复

GPU 作业 `41407`（`outputs/change_agent_levir_gpu_closed_loop_20260719_145637`）验证了
精确锚点确实会命中组件，但也把 Verifier 的语义错误从“无效空操作”放大成了真实有害
编辑。`test_85_16` 的 r0 共 684 像素，negative point 精确删除整个组件；candidate RGB
把 T1/T2 都判为 building，程序推导 `removed_false_positive` 并接受。闭环后 GT 审计
显示其中 636 像素属于真实变化，IoU 从 `0.30658070` 降至 `0.16696629`，聚合 IoU 降至
`0.67360342`。这是语义安全失败，不是 locality 或 coverage 失败。

同一离线审计也给出了正确的改进方向。旧初始 proposal 按面积从大到小只取六个，导致
真正低风险的小型 FP 根本不可见：`test_78_13` 的 127/135/177/254 像素组件均为纯 FP，
`test_85_16` 的多个 83–173 像素组件也为纯 FP。继续优先最大 FP 判断，会让一次 2B 模型
错误删除主变化区域；简单恢复“mask-context 一票否决”又会重新过滤已知的 135-pixel
有益删除。

修复采用分级证据而非统一放宽或统一收紧：初始 proposal 像 candidate delta 一样保留
全部连通分量，六个只表示每次 Qwen call 的 batch size；所有 batch 成功才形成诊断。
同类可行动错误优先选择面积最小的组件，以最小化单次判断错误的损失。candidate delta
小于等于 previous change mask 的 5% 时，继续由 clean-RGB 时相事实决定，mask-context
只审计，确保小型有益删除不会被旧抽象标签误杀；任一组件或总 delta 超过 5% 时，两类
证据必须一致，否则程序将 final effect 改为 uncertain 并拒绝。这会直接拦截本轮
684/4342=15.75% 的有害删除，同时不拦截已知 135/14752=0.92% 的有益删除。

## 18. 2026-07-19 验收闭环结果

GPU 作业 `41410`（`outputs/change_agent_levir_gpu_closed_loop_20260719_151344`）在提交
`c87b707` 上成功完成，耗时 115 秒。三个样本的 initial audit coverage 均为 1.0，组件
数量/批次数分别为 8/2、8/2、19/4；进入 candidate Verifier 的 delta 也保持 1.0 覆盖。

`test_78_13` 精确选择 135-pixel 小组件，在 t1 执行 negative point。该组件离线 GT
重叠为 0，删除后 TP/FN 保持 11506/430，FP 从 3246 降至 3111，IoU 从 0.75787116
提升至 0.76467070。其 delta 仅占 previous change mask 的 0.915%，所以即使旧
mask-context 仍误报 removed_true_change，clean-RGB 的 building/building 判断仍能按
设计放行 removed_false_positive，证明“不要过滤有益小修改”的目标已经实现。

与此同时，`test_20_15` 的 352-pixel 删除占 6.53%，mask-context 与 RGB 冲突后被
large-delta consensus gate 改为 uncertain；离线 IoU 虽会降至 0.78909026，但候选没有
进入 live state。`test_85_16` 的全图 positive candidate 被面积、locality 和 target-mask
变化硬门拒绝。两者最终都保留 initial。最终聚合 IoU/F1 为 0.70117909/0.82434482，
高于 initial 的 0.69744116/0.82175592，并且三个样本均未低于各自 initial。本轮结果
达到既定验收条件，可作为下一阶段回归基线。
## 2026-07-20 后续设计修正：Qwen 恢复为语义 Verifier

上一版已经在三样本闭环中首次达到验收标准，因此用 Git 标签
`closed-loop-effective-v1` 固化，作为有效但偏工程化的第一版闭环基线。它解决了候选
重复裁决、覆盖缺失、局部灾难和输出截断问题，但把 Qwen 收缩为
`building/background/mixed/uncertain` 分类器，再由程序决定 FP/FN 与 better/worse，确实
弱化了最初希望 Verifier 承担的语义诊断和纠错职责，尤其无法正面处理 mixed/uncertain。

新协议恢复三类 Qwen 责任：初始局部错误诊断、候选 delta 效果诊断、全局质量/进度/
better-worse/纠错综合。局部输出重新包含详细自然语言证据、置信度、严重度、目标时相和
建议动作；全局输出由 Qwen 对有益和有害部分进行权衡。程序不再执行“全部有益才 better、
出现 mixed 就 uncertain”规则，只保留可验证的结构和安全边界。1024 token 预算保持不变，
并通过局部分批加一次全局综合控制单次长度。

这不是取消安全机制：Environment 仍保证区域真实存在、delta 全覆盖、region ID 到坐标的
精确映射、added/removed 标签方向一致；闭环仍保留 identical-state、SHA256 缓存、重复动作
拒绝、rollback、locality 与面积硬门。区别是这些机制不再替 Qwen 做视觉语义判断。

首轮 rich 协议 GPU 作业 `41502` 进一步表明，恢复职责后首先要解决的是“语义定义能否被
2B 模型稳定执行”。六区域长输出既会在 1024 token 内截断，也会把单时相 object mask 的
空白误叫成最终 change mask 的 FN。修复因此不是重新程序化判断，而是把每批降到两个区域、
明确 FP/FN 只针对最终 change mask、为黑色且正确不变的 proposal 增加
`correct_unchanged`，并把每个 proposal 的几何允许标签直接写进提示。全局判断仍完全由
Qwen 完成。

后续作业 `41503` 证明两区域分批已彻底消除截断，输出均完整覆盖两个 region；但模型稳定
写出大写 `"T1"/"T2"` 和字符串 `"null"`，被严格枚举解析拒绝。这类差异不涉及视觉判断，
因此解析层现在只做格式规范化后再校验，原始生成仍完整留档；程序不会借此改变 verdict、
comparison、score 或 correction 的语义。

作业 `41504` 首次让 rich Verifier 全流程有效，但局部 panel 把 mask 颜色混入 RGB，Qwen
反复把紫/粉/青色诊断图当成真实场景，最终把 `test_85_16` 的 19 个组件全部判成
`true_change`。v10 改为每次只看一个精确区域；上排仅保留黄色外轮廓的干净 T1/T2 RGB，
下排明确为二值几何和原始差分。全局 Qwen 获取 region 面积、框、极性和长诊断，但不再
接收局部阶段容易产生锚定的 advisory action，最终 target/action 仍由 Qwen 独立规划。

作业 `41505` 的 aggregate IoU 达到 `0.69816521`，并在 `test_20_15`、`test_85_16`
接受了真实有益候选，说明 rich synthesis 已具备工作能力。但所有 initial 区域仍机械输出
`true_change`，candidate 也倾向首个 `added_true_change`，甚至把两时相共有的黄色定位环
当作亮度变化。v11 因此去掉全部彩色 RGB 标注，要求 Qwen 先输出局部 `t1_state/t2_state`，
再直接给出 FP/FN/true/mixed 或 candidate effect。状态只作为 Qwen 自己的可审计依据链，
程序不会根据状态对推理结果做语义映射。

作业 `41506` 中，Qwen 已能稳定看出首个区域是 `background/background`，但仍把“当前白色、
两时相无变化”错误命名为 `correct_unchanged` 或 FN。v12 要求它先回显 Environment 提供的
`white_predicted_change/black_predicted_unchanged`，再依据四格语义表给 verdict。局部术语
冲突不再让整轮立即失效，而是带 `geometry_consistency=false` 进入全局 Qwen 复核；最终全局
输出仍受白色不能为 FN、黑色不能为 FP 的结构检查。局部视觉同时增加无着色 focus tile：
组件内 RGB 完全不变，只压暗组件外上下文。

作业 `41507` 已在三个样本上首次稳定生成语义正确的局部 FP：`white +
background/background -> false_positive`，反馈也明确指出白色 change 不受 RGB 支持。唯一
失败是模型把几何回显缩写为 `"white"`。解析器现在只把 white/black 展开为协议全名后核对
Environment geometry；Qwen 生成的 FP 结论保持原样。

作业 `41508` 表明 focus tile 的外部压暗会诱导 Qwen 输出 `dark/dark`，而 dark 不能安全映射
成 building/background。v13 删除 focus 和所有 RGB 标注，只提供逐字节不变的 T1/T2 crop、
独立二值 geometry 和 raw difference；状态链、长诊断和 Qwen 全局决策职责保持不变。

作业 `41509` 验证了 v13：35 个局部区域全部生成完整长 JSON，不再截断；三个样本中多数
白色区域已由 Qwen 依据 `background/background` 正确诊断为 FP（分别为 6/8、7/8、18/19）。
但全局综合仍会选中自己刚判为正确的 `test_85_16/r11`，并在另外两个样本中把 negative
point 放到目标 mask 为黑色的组件 seed 上，形成空操作。因此最终没有候选被接受，聚合 IoU
保持 `0.69744116`。这说明当前主要矛盾已经从局部语义识别转为 Qwen 自身的全局一致性和
可执行纠错规划。

v14 不用程序替代 Qwen 判断 FP/FN 或 better/worse，而是补充精确组件级 T1/T2 mask 占用、
点击 seed 占用和逐字节不变的 tight T1/T2 crop。全局局部摘要按组件从小到大呈现，降低默认
选择最大/首个区域的锚定。运行时新增的约束只检查 Qwen 自身输出与确定几何：全局不能把局部
已明确判为正确的同一区域再次作为错误纠正；negative point 不能点击目标 object mask 的黑色
seed。发生冲突时要求 Qwen 重试，不由程序生成新的语义裁决。
