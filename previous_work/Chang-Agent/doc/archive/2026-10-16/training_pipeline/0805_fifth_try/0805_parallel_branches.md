# 0805 五分支并行开发说明（2026-08-05）

## 1. 目的与当前事实

本文件用于管理五条并行开发线，避免多个窗口或代理在同一工作树中互相覆盖，并统一最终合并、实验和 freeze
标准。这里的“开发基线”只是并行开发共同起点，不代表系统能力已经冻结，也不授权 SFT。

当前仓库信息：

- 仓库：`/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl`
- 当前分支：`feat/qwen3vl-gt-direct-teacher-0804`
- 当前 HEAD：`365c3a3ae79fcbd1756cbc595eea4c455bf85336`
- 0804 `train_407_r3c2` reference：
  `outputs/CA_0804(1)-qwen3vl-gt-direct-teacher/attempt-0001/shards/shard-0/episodes/train_407_r3c2/episode.json`
- 当前 `0805_fifth_try/` 仍是未跟踪目录。创建 worktree 前，应先形成只包含 0805 规划文档的共同基线提交，并在下表
  填入其完整 commit；不能让五个窗口从不同的 dirty state 起步。

当前 SAM3 不是“完全没有接入”：SAM3 box 已接入 Executor，并用于 fresh initialization、empty-mask scan 和离线
GT-direct teacher。缺口是普通非空 FP/FN 区域尚未进入统一 point/box 配对评测面，production planner 仍把普通
FP/FN 固定路由到 SimpleClick。因此“效率优化”和“SAM3 接入评测框架”是两个独立验收项。

## 2. 并行开发总规则

1. 一个窗口只操作一个独立 Git worktree 和一个分支。禁止五个窗口共同写原仓库目录。
2. 原仓库作为集成工作树；功能开发不直接落在集成分支。
3. 子代理用于单分支内部的只读审计、测试设计和结果复核。若要写代码，必须明确文件所有权；不能让多个代理同时
   修改共享文件。
4. 每个分支从同一个完整 baseline commit 创建，提交保持原子化，并同步维护本文件的状态、commit、Job ID、输出
   路径和结论。
5. 任何计算型测试、实验和批处理都通过 Slurm；管理节点只做编辑、静态检查和轻量审计。Slurm/Unix 用户固定为
   `guisongxia`。
6. 所有输出采用新的 append-only `attempt-NNNN`，日志放对应实验目录的 `logs/`；不得覆盖 0803/0804 artifact。
7. 长任务无论成功、失败或取消，都发送 QQ 终态通知。日志、receipt 和通知不得包含凭据或代理值。
8. 五条线均未通过前，不建立新科学 freeze，不启动大规模实验，不授权 SFT。

建议 worktree 结构（路径和分支名冻结后再执行）：

```text
Chang-Agent-local-qwen3vl                  集成工作树
Chang-Agent-wt-0805-efficiency             feat/0805-efficiency-resident
Chang-Agent-wt-0805-sam3-eval              feat/0805-sam3-eval-bridge
Chang-Agent-wt-0805-router-mixed            feat/0805-action-router-mixed
Chang-Agent-wt-0805-verifier                exp/0805-verifier-calibration
Chang-Agent-wt-0805-horizon                 exp/0805-adaptive-horizon
```

## 3. 依赖与合并顺序

```text
共同开发基线
  ├── 1. efficiency resident/cache/batch
  ├── 2. SAM3 evaluation bridge
  ├── 4. Verifier calibration
  └── 3/5 可并行设计接口和测试，但暂不做正式 GPU 结论

1 通过 hash 与 SLA ───────────────┐
2 完成普通区域 point/box 评测面 ───┼── 3. router + mixed_error
                                    └── 5. adaptive horizon 正式运行

建议集成顺序：1 → 2 → 3 → 4 的已冻结规则 → 5 → 全量验收 → 新 freeze → SFT
```

效率分支先合并，因为后续 paired routing、Verifier replay 和长 horizon 都依赖可接受吞吐。SAM3 evaluation bridge
可以与效率代码并行开发，但正式性能和科学验收必须先 rebase 到已通过效率门的集成版本。Verifier calibration 可以
并行生成 replay 证据；只有阈值预注册并通过后，才允许把规则改动合入 production。

## 4. 分支一：模型常驻、缓存、embedding 复用与同图分组

### 分支与范围

- 分支：`feat/0805-efficiency-resident`
- 目标：像本地 vLLM 服务一样，让 SAM3 与 SimpleClick 各自在自己的 Python 环境中常驻；不再每候选重载模型。
- SAM3 与 SimpleClick 依赖栈不同，必须保留两个隔离常驻进程，不能硬塞进一个 Python 进程。
- SAM3 复用按 image hash 缓存的 vision backbone state。SimpleClick 没有可安全独立复用的 prompt-free image
  embedding，只复用常驻模型，并为每个候选重置 predictor、ROI、`prev_prediction` 和 click history。
- “同图 batch”首版定义为一个服务请求内分组、共享模型/图像状态、按旧顺序独立执行。SAM3 多 box 联合 prompt 和
  SimpleClick 真 tensor batch 都可能改变数值路径；没有逐候选 hash 证据前不能宣称真 GPU batch 等价。

### 当前阻塞点

- 四小时不是 `max_steps=3` 本身造成，而是大量候选乘以逐调用冷启动：`train_407_r3c2` 有 588 个逻辑候选和
  12 次 replay，每次都新建 subprocess、重新加载 SAM3 或 SimpleClick。
- 相同 backend 输入跨 branch/track 重复执行；SAM3 还反复计算同两张图的 image embedding。
- 候选串行调用，无同图请求分组。最大可消除成本是重复进程启动、模型加载、embedding 计算和 raw 推理。

### 实现方式

- 不从零重写模型服务。复用 SAM3 官方 `Sam3Processor`、SimpleClick 的 `load_is_model/get_predictor`、项目现有
  isolated environment、seed、timeout、artifact 和 receipt 逻辑。
- 新增薄本地服务层：用标准本地 IPC 管理两个隔离常驻 worker；每个 job 中 SAM3、SimpleClick 各加载一次。
- SAM3 按 image hash 复用 backbone state；SimpleClick 复用模型，但每候选重置 predictor/session。
- 使用显式 content-addressed raw-mask cache；同图请求分组、按旧顺序独立执行。canonical replay 强制真实推理，
  不读取 candidate cache。
- vLLM 不支持此类分割模型；Triton/Ray Serve 仍需编写相同模型专属状态与 cache handler，当前单作业离线评测阶段
  不引入重型 serving 平台。未来需要多用户在线服务时再评估迁移。

### 缓存与 replay 语义

- SAM3 raw cache key 至少包含：image 内容 hash、query、box 的确定性浮点表示、checkpoint、BPE、resolution、
  device、autocast dtype、seed、代码/配置 commit。
- SimpleClick raw cache key还必须包含：model-input mask hash、coordinate、polarity、有序 click history、ROI/session
  context 和模型配置。
- 缓存 raw tool mask；Environment 仍对每个逻辑候选独立 composition、hard gate、metric 和 artifact 记账。
- 普通候选允许 `read_write` cache。selected canonical replay 必须 `force_execute`，不能直接读取同一个 raw result；
  replay 仍需独立真实推理，并与 candidate branch 比较 T1/T2/change 三个 hash。
- 每条 track 的 fresh initialization 可复用 SAM3 image embedding，但必须重新建立 prompt state、重新 grounding，不能
  复用最终 initialization mask 冒充 fresh reset。

### reference 与硬验收

`train_407_r3c2` 从 fresh state 完整执行四条三步 track；不从 step 3 续跑，因为效率验收必须覆盖旧版全部候选。

- 旧版：588 logical candidates、12 selected replay、4 次 fresh initialization、墙钟 `04:01:17`。
- 按 source state + action/proposal 统计：204 个候选签名、384 次重复。
- 按真实 backend 输入统计：73 个唯一 raw request（SAM3 49、SimpleClick 24）、515 次重复；使用最终 cache key 后须
  重新报告实际 hit/miss，不能只抄理论值。
- 588/588 candidate T1/T2/change state hashes 与 reference 一致。
- 12/12 selected replay hashes 一致，而且 replay 都是 `force_execute`。
- 初态、四条 track 的 selected IDs/action sequence、metric、hard/RGB gate、termination、final hashes 全部一致。
- cache off/on、batch size 1/>1 的逐候选 mask hash 一致；任一工具漂移时，该工具退回 resident sequential，不能
  放宽 hash 门。
- 每个 job SAM3/SimpleClick model load 各不超过 1 次；明确记录 embedding hit/miss、logical/physical calls、cache
  hit/miss、batch 数、p50/p95、GPU 峰值和 `MaxRSS`。
- 硬 SLA：完整 `train_407_r3c2` 不超过 30 分钟、相对旧版至少 8 倍、单样本不超过 0.5 GPU-hour；stretch goal
  为 15 分钟。未达硬 SLA，停止，不运行大实验。
- `qwen_calls=0`、`teacher_only=true`、`training_authorized=false` 保持不变。

建议输出：`outputs/CA_0805(1)-efficiency-resident/attempt-NNNN/`。

## 5. 分支二：SAM3 接入普通区域评测框架

### 分支与范围

- 分支：`feat/0805-sam3-eval-bridge`
- 目标：把 SAM3 box 从“empty-mask 特例/离线 teacher 工具”提升为普通非空 FP/FN 区域的可评测候选。
- 同一 source state、region 和 target view 上，普通 FP 配对 `negative point/box`，普通 FN 配对
  `positive point/box`；geometry 只来自正常 GT-free proposal。
- 本分支建立 evaluation surface 和反事实数据，不训练最终 Planner，也不直接改变 production 默认工具。

### 数据收集与状态展开

1. 先跑 baseline shadow smoke：当前 production policy、hard gate 和 Verifier 决定官方下一状态；SAM3 在状态副本上
   执行，只保存结果，不提交。即使事后 GT 显示 SAM3 更好，也不能改写官方轨迹。
2. 接线通过后跑 `max_steps=3` bounded counterfactual tree：每个节点独立执行 point/box；两个候选若都形成被
   Environment 接受的不同新状态，则两个子状态都继续展开。reject/rollback、相同 state hash、no-op、cycle、
   endpoint 停止展开。二选一三步理想上限为 14 次工具执行，不做 588 候选穷举。
3. 每个子状态重新诊断 region，因此树覆盖可达的 point/box 混合历史，而不是只生成 SimpleClick-only、SAM3-only
   轨迹。树是 teacher-only 数据构造，不是部署时同时维护多条状态。
4. SAM3 同时项、不同 region/box 可继续执行；完全相同的 `image + query + box` raw request 结果相同，必须
   cache/no-op/prune。当前不引入累积 prompt 或 mask-prompt 式 SAM3 内部迭代。

### GT、标签与并行开发边界

- 两个候选全部完成后才能读取 GT。GT 只计算 improvement、harm、regret 和 tool-preference label；不能选择
  region、geometry、target view、在线动作或官方下一状态。
- 分层保存 `raw_oracle_winner`、`hard_gate_safe_winner`、`verifier_selected`、`tool_preference` 和 `tie/neither`。
  当前 Verifier accept/reject 不能直接作为训练真值，避免把尚未校准的偏差教给 Planner。
- 分支二开发期只依赖稳定的 `PointBackend.refine/BoxBackend.segment_box` 接口：fake backend 完成 tree/schema/
  validator，legacy backend 跑极小 smoke。`prefetch` 缺失时为 no-op，正确性不能依赖 cache。
- 分支一通过 hash/SLA 并先合入后，分支二再 rebase，启用 resident、SAM3 embedding cache 和 raw cache，执行完整
  Slurm 数据构造。原则是并行开发、顺序放行。

### 硬验收

- 非空 mask 的普通 FP/FN 都能产生合法 point/box pair；两个候选的 source hashes 完全相同。
- 每个候选保存 candidate ID、source hashes、region/target、tool/action、raw tool mask hash、组合后 T1/T2/change
  hashes、gate、metric、延迟和 cache/batch receipt。
- runtime 输入无 GT；raw、hard gate、Verifier、GT 结论分层，Qwen 调用和证据可审计。
- point/box replay 三 mask hash 一致；相同 SAM3 request、重复 state 和 cycle 被确定性剪枝。
- 覆盖 SAM3 add/refine/replace 的实际 composition 边界，不把 change IoU 改善错误解释成 T1/T2 building mask 正确。
- train/val/test 按 parent 隔离；`teacher_only=true`、`training_authorized=false`，独立 validator 全通过。
- 输出工具胜率时按样本、错误极性、target view、region size 和 gate 结果分层；不能用 0804 的 `27/30 box`
  直接证明 SAM3 普遍更强。

正式构造前仍需冻结 utility/harm/tie 阈值、pilot 样本数和数据规模；rejected candidate 默认只保存、不继续展开。

建议输出：`outputs/CA_0805(2)-sam3-eval-bridge/attempt-NNNN/`。

## 6. 分支三：GT-free router 与 mixed_error 原子动作

### 分支与范围

- 分支：`feat/0805-action-router-mixed`
- 依赖：SAM3 evaluation bridge 的统一候选接口；正式性能验收依赖 efficiency 分支。
- 普通区域不再硬编码“FP/FN 一律 SimpleClick”。router 在 point/box 候选之间使用 GT-free 特征选择。
- offline GT 只负责生成 router 训练/评测 label，不进入 production router 输入。
- `mixed_error` 拆成有明确 polarity、target view、region 和单工具动作的子目标；每步只修一个，随后重新审计。
- 无法安全拆解时，使用独立 `mixed_unresolved`/plan-null 语义，不能伪装成默认 `t2` 动作或普通 schema error。

### 硬验收

- 覆盖普通 FP、普通 FN、FP+FN mixed、mixed 与可执行纯区域并存、uncertain、parse-fail uncertain。
- 对 mixed 候选明确记录 decomposition 依据、先修哪一极性、目标 view、重新审计结果和 stop reason。
- router 输入字段逐项证明 GT-free；相同输入确定性复现相同选择。
- paired evaluator 提供 point/box oracle label、router prediction、regret、失败类型；不能只报总体准确率。
- 检查 early plan-null 与最后预算步 plan-null 的记账差异；顶层 fallback `target_view=t2` 不得当作真实路由决定。
- production 默认路由只有在离线比较、GT-free replay 和安全门通过后才改变。

建议输出：`outputs/CA_0805(3)-router-mixed/attempt-NNNN/`。

## 7. 分支四：Verifier 严格度校准

### 分支与范围

- 分支：`exp/0805-verifier-calibration`
- 目标：验证当前 magnitude-blind zero-harm rollback 是否过严，而不是先假定它过严。
- 保持职责分层：Qwen 输出局部视觉事实 JSON；Python 派生 improvement/harm；Environment 决定 commit/rollback。
- 先做 GT-blind decision，再离线打开 GT 复算；禁止把 GT 信息泄漏进 prompt、JSON 或 runtime decision。
- 比较至少两套预注册规则：现有 zero-harm 与 bounded-tradeoff。阈值在看正式结果前冻结。

### 硬验收

- 统计 false accept、false reject、rollback regret、accepted harm、net pixel/IoU/F1 gain，并按错误类型和工具分层。
- 对每个 reject 保留 before/candidate/replayed state hashes、Qwen JSON、Python derived fields 和最终 Environment reason。
- 区分 semantic reject、hard-gate reject、schema/parse fail、plan-null；不能混成一个失败计数。
- bounded-tradeoff 不能只优化均值；必须设 accepted-harm 上限、尾部风险和回滚安全约束。
- 在校准证据完成前，本分支不修改 production 默认 Verifier 规则。最终规则改动由集成分支单独提交。

建议输出：`outputs/CA_0805(4)-verifier-calibration/attempt-NNNN/`。

## 8. 分支五：adaptive horizon 与停止规则

### 分支与范围

- 分支：`exp/0805-adaptive-horizon`
- 依赖：正式 GPU 运行必须等 efficiency SLA 通过；最终 production stop 还依赖 router/gate 语义稳定。
- 离线 teacher stop 与线上 GT-free stop 分开设计。teacher 可读 GT metric；production stop 只能读状态、Verifier、
  action history 和 GT-free progress proxy。
- stop reason 至少覆盖：endpoint reached、no beneficial/executable action、gain below threshold with patience、state/action
  cycle、mixed unresolved、hard cap。
- `T_max` 只是安全上限，不再等同于所有样本固定执行步数。

### 硬验收

- 先用已有 step-3 state 做便宜的 step-4+ 诊断，再从 fresh state 跑最终可比轨迹；续跑结果不能替代全轨迹验收。
- 预注册 `tau_gain`、patience、cycle key、`T_max` 和 endpoint；不得看完结果后反复调阈值并只报告最好版本。
- 与固定 `max_steps=3` 比较 endpoint success、额外收益、无效动作、rollback、wall time 和 GPU-hour。
- `train_407`/`train_364` 用于检查三步后是否仍有收益；`val_43` 同时区分“增加步数有效”与“gate 已阻断，
  增步无效”。
- 线上停止策略必须在 GT-blind replay 中单独验收，不能直接复用离线 IoU 阈值。

建议输出：`outputs/CA_0805(5)-adaptive-horizon/attempt-NNNN/`。

## 9. 文件所有权与冲突规则

| 范围 | 首要所有分支 | 说明 |
|---|---|---|
| resident worker、IPC、raw cache、batch、profiling | 分支一 | 优先新增独立模块，保留 legacy reference path |
| 普通区域 paired point/box evaluator | 分支二 | 优先新增 runner/config/test，不借 GT 改 production planner |
| staged planner、tool router、mixed decomposition | 分支三 | 不在此分支顺带修改 Verifier tradeoff 阈值 |
| Verifier replay、阈值分析、校准报告 | 分支四 | production rule 由集成提交落地 |
| adaptive stop/horizon runner | 分支五 | 不绕过效率门提交长作业 |
| `change_agent/environment.py`、共享 schema、公共 runner 入口 | 集成窗口 | 多分支需要时先约定接口，最终由单一 owner 合并 |
| 本文件与最终 freeze 文档 | 集成窗口 | 分支只提交状态更新，集成窗口解决冲突 |

若分支必须修改非本分支所有的共享文件：先提交最小接口 commit，通知其他窗口固定该 commit，再继续开发。禁止各自
复制一套近似接口，最后靠大块人工冲突合并。

## 10. 统一测试与 artifact 门

每条分支按以下顺序放行：

1. exact-commit、clean worktree、配置/模型/数据/代码 hash preflight。
2. Slurm 零 GPU contract/unit tests；失败不提交 GPU 作业。
3. 单 GPU fixed micro smoke；覆盖新增工具路径和失败路径。
4. 单样本完整验收；独立 validator 读取 append-only artifact 复算。
5. 达到分支正确性和资源 SLA 后，才允许扩大样本。

统一 receipt 至少记录：branch、完整 commit、dirty=false、config hash、模型/tool identity、sample、Job ID、节点、资源、
开始/结束时间、logical/physical calls、结果路径、validator 结论和 QQ 通知状态。

## 11. 分支状态表

| 分支 | 状态 | baseline commit | branch HEAD | Slurm/输出 | 放行结论 |
|---|---|---|---|---|---|
| `feat/0805-efficiency-resident` | planned | 待冻结 | — | — | 未放行 |
| `feat/0805-sam3-eval-bridge` | planned | 待冻结 | — | — | 未放行 |
| `feat/0805-action-router-mixed` | planned | 待冻结 | — | — | 未放行 |
| `exp/0805-verifier-calibration` | planned | 待冻结 | — | — | 未放行 |
| `exp/0805-adaptive-horizon` | planned | 待冻结 | — | — | 未放行 |

状态只允许按 `planned → implementing → contract_passed → smoke_passed → sample_validated → integration_ready`
推进。任何 hash、GT firewall、安全或 SLA 门失败，状态保持未放行并记录失败 receipt，不能删掉失败尝试重跑后假装首次成功。

## 12. 最终集成与 freeze 条件

最终集成窗口按固定顺序 cherry-pick/rebase 小提交，逐分支复跑 contract。满足以下全部条件后才能建立新 freeze：

- 效率分支逐候选 hash 等价且达到 `train_407_r3c2` 硬 SLA。
- SAM3 已进入普通非空区域 paired evaluation surface，GT firewall 与 replay 完整。
- router 和 mixed_error 有 GT-free、可执行、可审计的动作语义。
- Verifier 规则由 blind-first 校准支持，false accept/false reject 与 rollback regret 达到预注册门。
- adaptive horizon 的离线和线上停止逻辑分离，并在固定样本上通过验收。
- 所有 Slurm 作业绑定 exact commit，artifact append-only，独立 validator 全通过。
- 新 freeze 后才并行启动 SFT 与 teacher-only 能力边界实验；两条数据线 lineage、授权和 train/val/test 隔离。

## 13. 八荣八耻落实

- 查档求真：以源码、reference artifact、mask hash、receipt 和独立复算为准。
- 对齐需求：效率、SAM3 评测桥、router/mixed、Verifier、horizon 五条线分别回答问题，不互相冒充结论。
- 请示规则：影响科学问题或 production 默认行为的阈值、路由和协议先预注册，再放大实验。
- 复用存量：复用 0804 的 588 候选和 12 replay reference，不重复跑四小时 legacy full baseline。
- 完备测例：cache off/on、batch 1/>1、force replay、普通 FP/FN、mixed、uncertain、rollback、cycle、hard cap 全覆盖。
- 恪守规范：独立 worktree、Slurm、exact commit、append-only、QQ 终态通知、凭据不落日志。
- 坦诚存疑：不把 box 多写成 SAM3 普遍更强，不把请求分组写成真 tensor batch，不把离线 teacher 写成线上能力。
- 分步迭代：先吞吐和评测面，再 router/mixed 与 Verifier，最后 horizon、freeze 和 SFT。
