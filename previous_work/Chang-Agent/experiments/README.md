# Change-Agent 实验登记

2026-10-16 保留范围调整：本目录登记表作为历史实验报告保留，旧实验的模型、逐步轨迹、数据副本及运行输出已按用户要求清理。表内历史路径、当前证据路径和下文的归档验收口径描述实验当时状态，不代表产物仍在磁盘上；需要重放或重新验收时先重跑实验。后续代码统一在工作区 `Chang-Agent` 目录维护。

本目录维护一份可追溯、可筛选的实验工作簿：

- `experiment_registry.xlsx`：给人查看和筛选的四子表工作簿。
- `experiment_registry.json`：工作簿的可审计源数据；修改实验记录时先改这里。
- `build_experiment_workbook.py`：只使用 Python 标准库，从 JSON 确定性重建并校验工作簿。

工作簿固定包含四张子表：

1. `01_实验记录`：一行对应一条科学实验结论；执行是否完成、科学结论和 retry lineage 分列记录。
2. `02_字段字典`：staged verifier 的功能划分、各 stage JSON contract、proposal 字段和枚举语义。
3. `03_数据登记`：固定 1/3/5 条开发评测数据、训练数据版本、样本身份、文件哈希、GT 使用策略和构造 pipeline。
4. `04_运行尝试`：一行对应一个与科学 pipeline 有关的 Job/attempt；成功、失败、取消和被 retry 取代的运行都保留。它不是所有无关集群作业的流水账。

## 三轴状态与 attempt lineage

`execution_status`、`scientific_outcome`、`lineage_status` 是正交字段，禁止把它们压成一个“总体状态”：

- `execution_status`：`planned | running | completed | failed | cancelled`，只描述运行执行状态。
- `scientific_outcome`：`not_evaluated | baseline_reference | pass | negative | inconclusive`，只描述可由科学评测支持的结论。
- `lineage_status`：`canonical | superseded`，只描述该行是否是当前保留的权威 attempt，或已被明确的 retry 取代。
- `failure_class` 仅用于 attempt ledger：失败使用 `infra_failed | contract_failed | test_failed`，未失败写 `not_applicable`。

例如，v1 Job 51903 同时是 `completed + negative + canonical`；Job 51891 是
`failed + not_evaluated + superseded + infra_failed`，不能把后者当成模型科学负结果。失败
attempt 与 negative 是两回事：前者没有完成可解释的科学评测，后者是成功完成后得到的负结论。

`retry_of_attempt_id` 只指向同实验、同 stage 的明确前驱；`depends_on_attempt_ids` 是 JSON
字符串数组，只列上游依赖。两者不能混写。例如成功重跑数据构建是 retry，而 SFT 使用成功数据
是 dependency。superseded attempt 必须恰有一个已登记 retry successor。

## 历史路径、archive 与 resolved config

实验主表、数据登记和 attempt ledger 都把 `历史运行路径` 与 `当前证据路径` 分列。路径只允许
安全的 repo-relative `outputs/...`；0801 的历史路径是归档前的 `outputs/CA_0801(...)`，当前可读
证据必须位于 `outputs/CA_0801_archive/CA_0801(...)`。历史绝对路径失效不等于科学证据缺失，
但 archive 内旧 `train.json`、`config_used.yaml` 等绝对路径不能直接用来重训，也不得就地改写。

0802 P1 也已归档：原始运行身份继续记为 `outputs/CA_0802(...)`，当前权威证据
则必须显式记为 `outputs/CA_0802_archive/CA_0802(...)`。构建器按 `0801`/`0802`
命名空间参数化校验，并对 P1 额外强制历史路径与 archive 路径一对一；
不允许“原路径不存在就猜 archive”的
静默 fallback。冻结 runtime receipt 中的历史绝对路径不改写；校验器先验证其原
仓库根和日期命名空间，再唯一映射到当前 archive 并重算 size/SHA256。

v2 的
`doc/training_pipeline/0802_third_try/QWEN3VL_V2_RECONSTRUCTED_RESOLVED_CONFIG_0802.json`
是 archive 外的只读历史重建映射，不是原始 runtime-emitted resolved-config receipt，不能作为重训
输入。每个新 run 必须由 worker 在启动后写出真正的 resolved config，并把它与 launch receipt、
trainer state、Job ID、commit、数据/模型 hash 绑定；复制模板 YAML 不算 resolved receipt。

## 缺失值规则

表中不允许用空白或数字 `0` 代替缺失结果：

- `未运行`：已经确认该阶段没有执行。
- `未记录`：可能执行过，但没有可追溯产物证明。
- `未测量`：运行存在，但该指标没有测量。
- `不适用`：实验设计本身不产生该指标。
- `null`：原始 JSON 明确为 null，必须原样保留。
- 数字 `0`：只表示真实测得为零。

## 评测框架口径

开发评测集均来自 `LEVIR-CD/test_256`：

- `DEV-FIXED-3-V1`：`test_20_15`、`test_78_13`、`test_85_16`。这是历史拟定的三样本快速集。
- `DEV-SMOKE-1-H1-V1`：固定 `test_1_1`。这是 0802 P1 terminal-review harness 的
  单样本协议 promotion gate；只验证 H0 不调用第二轮、H1 恰好调用一次合法 fresh-context 第二轮，
  不是独立科学 test。
- `DEV-CORE-3-V1`：`test_20_15`、`test_1_1`、`test_50_8`。这是 0801 v12.1 baseline
  实际使用的 e2e3 主评测门，顺序固定；v2、v3、v3-bgfp、v3-bgfp-scope 和 v4 主指标均按这
  3 条登记。
- `DEV-EXTENDED-5-V1`：`test_20_15`、`test_78_13`、`test_85_16`、`test_1_1`、
  `test_50_8`。0801 v2 的 e2e5 只作为附加稳定性检查，不与 `DEV-CORE-3-V1` 的主指标混算。

这些样本都被长期反复查看，没有随机抽样 seed，也不是无偏 test。任何结果只能称为固定
diagnostic set 结果，不能宣称 LEVIR-CD 泛化性能。`DEV-FIXED-3-V1` 与
`DEV-CORE-3-V1` 的第 2、3 条不同，因此两个数据 ID 的 aggregate 不能互换或拼接。0801
迭代必须保持 `DEV-CORE-3-V1` 的身份和顺序不变；若运行五样本，需另列
`DEV-EXTENDED-5-V1` 的请求数、评分数、绝对指标和协议错误。

同一训练数据消融必须固定以下内容：

- 同一个基座模型、prompt/schema、proposal generator 和推理代码版本；
- 同一组 T1/T2 输入、SAM3/SimpleClick checkpoint、seed 和 rollout 参数；
- 同一训练方法和超参数；唯一变量为训练数据版本；
- 训练前 base verifier 和训练后 adapter 均运行相同的功能专项评测；
- 快速门固定运行相同的 `DEV-CORE-3-V1` 完整闭环；通过后再以
  `DEV-EXTENDED-5-V1` 作附加稳定性检查；
- 分开记录请求样本和实际完成评分样本；
- GT 只能在整个 GT-free rollout 完成并写 marker 后用于离线 IoU/Precision/Recall/F1。

功能专项指标根据训练目标选择，但必须预先写入 `功能主指标`。完整闭环至少记录：initial/selected IoU、Precision、Recall、F1，动作数、accepted/rejected、finish 数、invalid/no-action、工具成功率、样本退化数和证据目录。

## 当前 0729 记录边界

`0729(1)` 当前只在实验主表登记成功完成训练与评分的 clean retry：GPU Job `50129` 完成 32 条 LoRA SFT 和同 32 条 select-stage 生成评分。训练后 strict JSON、schema 和 region ID 都是 32/32，exact string match 是 21/32。此前失败 attempt 不应进入实验主表，但后续证据回填时必须进入 `04_运行尝试`，不得继续采用“失败不登记”的旧政策。

该记录明确标为“不完整实验”：训练前 select baseline 未运行，held-out 未运行，固定 3/5 条完整闭环未运行，adapter 也按 smoke 策略清理。它只能证明数据加载、LoRA、生成和 JSON 协议链路成功，不能证明完整 verifier 或视觉泛化得到改进。

## 0802 P1 H0/H1 closeout

P1 主表分为两个科学 arm：`0802(P1-H0)` 是
`completed + baseline_reference + canonical` 的冻结 single-round control；
`0802(P1-H1)` 是 `completed + pass + canonical` 的 GT-free terminal-review treatment。
这里的 `pass` 只表示预注册的 harness 协议、provenance、GT boundary 和 safety primary gates
全部通过。

retry3 CORE3 的 H0/H1 micro IoU/F1 都是 `0.7422312 / 0.85204673`，H1-H0 差值为 0。
因此不得写成分割提升、等价性证明或 LEVIR-CD 泛化结论。失败轨迹显示
`test_1_1` 保留 626 个 false-positive pixels，`test_50_8` 保留 166 个 false-negative
pixels；两次 H1 fresh finish confirmation 都再次选择 finish，没有产生动作、candidate 或状态变化。

P1 ledger append-only 保留四代各六条记录：初始 Jobs `53064–53069`、retry1
`53109–53114`、retry2 `53141–53146`、retry3 `53155–53160`。前三代每个 stage 都由下一代
同 stage 的 retry 明确取代；只有 retry3 六条是 canonical。CPU、guard 和 smoke 的
`scientific_outcome` 保守登记为 `not_evaluated`；成功 CORE H0 是
`baseline_reference`，最终 CORE H1 是 `pass`。不得把 smoke pair、standalone H0 或 partial
rollout 跨 attempt/freeze 拼成完整 CORE pair。

四代 24 条 ledger 的 `历史运行路径` 保留归档前身份；两条 P1 主表、每条
attempt 的 `当前证据路径` 和 `权威证据`、以及 DEV-CORE/SMOKE 的引用均指向
`outputs/CA_0802_archive/`。

结果由 `QWEN3VL_P1_RETRY3_RESULT_0802.json` 和 Slurm Job `53165` 的 trajectory analysis
共同绑定。完整训练流程数据不足仍是待验证假设：下一阶段应以 mechanically linked
`audit → action → candidate → next-state` 轨迹 treatment 对照 isolated stage snapshots，并匹配
source split、泄漏策略、token、optimizer steps、base checkpoint 和评测协议。

## 0803 P2 动作可达性 no-go

P2 在冻结 CORE3 step-zero masks 上机械枚举了 36 个有限单步候选：27 个 primary、9 个
`diagnostic_executor_only`。generation Job `53272` 完成 36/36 工具执行，且在完整 marker 和全部
candidate hashes 验证前没有访问 GT；严格 `afterok:53272` 的 scoring Job `53273` 随后完成事后
离线评分。Qwen 在本阶段从未调用。

36 个 candidate states 全部唯一；总体 beneficial/harmful/indeterminate 为 `4/31/1`，primary 为
`3/23/1`，diagnostic 为 `1/8/0`。`test_50_8` 存在一个 beneficial 且通过原 hard gate 的 primary
候选；但 `test_1_1` 唯一 primary 候选只把 FP 从 626 减到 566，IoU/F1 仍为 0，因此没有
primary beneficial candidate。预注册要求两个 target 都可行动，所以最终登记为
`completed + negative + canonical`，`proceed_to_c_iso_and_t_traj_data_construction=false`。

这里的 negative 是完整执行后得到的有效科学 no-go，不是作业或基础设施失败。结论只覆盖冻结
CORE3 的预注册有限单步动作清单：不能据此断言更大/多步动作空间无解，也不能声称 LEVIR-CD
泛化。hard gate 本轮不修改；C-ISO/T-TRAJ 尚未创建，因此 `03_数据登记` 不增加训练数据行。

## 更新与校验

工作簿仍由以下确定性命令重建和检查；按本集群规范，它们及 unittest 必须由零 GPU Slurm
closeout validation Job 在共享工作区执行，不能在管理节点直接运行：

```bash
python experiments/build_experiment_workbook.py
python experiments/build_experiment_workbook.py --check
```

构建器会校验四张表的完整列名与顺序、唯一实验/运行/attempt/Job ID、三轴闭集、failure-class
跨轴约束、retry/dependency 引用、repo-relative path、必填缺失值、0801/0802 显式 archive
映射、P1 冻结绝对绑定的归档后 size/SHA256，以及 0802 P1 四代 append-only lineage。0729
artifact 若存在则核对其指标；若它已迁走，只跳过 0729 自身，后续固定样本顺序与 A/B/GT
SHA256、Jobs `51727/51731/51903` 的 `sample_order`、v1 data/SFT Job+commit+success receipt、
v3/v5 失败 retry 仍独立执行。校验失败时不得把工作簿作为实验结论来源。

当前 attempt ledger 覆盖 40 行：原 0801 backfill 14 行、P1 四代 24 行，以及 P2 formal generation/
scoring 2 行。0801 部分包含
base/v1 八行、v3 data Jobs `52129/52132`，以及 v5 data Job `52307` 与 manifest-test Jobs
`52311/52315/52319`。其中 `52129=contract_failed`、`52311=test_failed`；`52315`
虽成功但被带显式 post-build manifest gate 的 `52319` 取代。P1 的四个失败
`53065/53069/53111/53145` 都是 fail-closed contract failures，不是模型科学负结果。P2 generation
单独登记为 `not_evaluated`，scoring 登记为 `negative`；manifest materialization 与 contract
validation 是工程 gate，不冒充两条 formal science attempts。后续每个科学相关 pipeline attempt
必须及时追加 ledger，不能等成功后再选择性登记。

v1 data 的历史 provenance 有一个必须保留的双层事实：Job `51876` 在 post-build summary
报错前已经产出 manifest 字节，因此 manifest 内 `runtime_commit=41a2bd...`；Job `51883`
使用 `9fad7f...` 复核现有 artifact 并闭合成功 receipt。前者仍是 failed attempt，后者是
canonical validation/receipt closure；不得把 `51883` 的 commit 伪称为 artifact producer。
checker 同时绑定这两个 commit、40 条 attempt identity/status evidence，以及 0801/P1 失败的 traceback、
P2 formal receipts/decision SHA256 与错误签名，避免仅凭一行状态猜测 failure class 或科学结论。
