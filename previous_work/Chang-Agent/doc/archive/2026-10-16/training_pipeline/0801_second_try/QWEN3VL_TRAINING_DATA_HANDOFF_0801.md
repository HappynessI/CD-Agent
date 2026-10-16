# Qwen3-VL v12.1 训练集构建交接（0801）

## 一句话结论

后续训练集构建必须使用
`/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl` 的
`feat/local-qwen3vl-verifier` 分支文件。冻结的 runtime 源码锚点是 `491532f`；不要从
稳定仓库 `Chang-Agent`、历史 control worktree 或 0729 select-stage SFT32 文件开始构建。

当前可以冻结 v12.1 数据格式、parser、Environment/Executor/rollback 安全门和评测
harness，开始训练集工作；不能把未训练 Qwen3-VL-2B 的输出当成正确监督标签。

## 1. 权威仓库、分支和提交

- 开发仓库：
  `/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl`
- 开发分支：`feat/local-qwen3vl-verifier`
- 冻结 runtime 提交：`491532f4622667e9fbf299c71aac228fa37f6a04`
- 最终冻结报告提交：`f6ddfc6`；它只修改文档，runtime 仍以 `491532f` 为锚点。
- 模型：
  `/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct`
- 所有新数据、测试和训练输出必须写入本地开发仓库的 `outputs/`，不得写回稳定仓库。

新会话应停留在开发分支当前 HEAD，不要 detached checkout 到 `491532f`；该提交只用于证明
runtime 边界。开始修改前执行：

```bash
cd /guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl
git branch --show-current
git status --short --branch
git log -3 --oneline
git diff --name-only 491532f..HEAD -- change_agent tools tests configs
```

在训练集工作开始前，最后一条命令应为空；`491532f` 之后只有文档提交。若不为空，必须先
判断是不是新会话或用户已有的训练工作，不能覆盖或回滚。

以下两个目录只作参考，不是训练集代码来源：

- `/guisongxia01/pangchao/wangyihan/Chang-Agent`：稳定闭源模型框架，存在用户自己的工作，
  不得修改或把其中旧文件复制覆盖到本分支。
- `/guisongxia01/pangchao/wangyihan/wyh/Chang-Agent-control-f80e270`：v11.3 A/B control，
  只用于解释历史对照，不包含最终 v12.1 输入与 repair 合同。

如果需要新建训练专用分支，应从当前 `feat/local-qwen3vl-verifier` 分支派生，而不是从稳定
仓库或 `f80e270` 派生。

## 2. 训练 target 的闭集协议

- protocol：`v12.1`
- experiment arm：`v12_1-compact`
- schema：`staged_verifier_environment_ordered_compact_audit_v12_1`

当前 baseline 只构造 production live path 实际使用的 `audit` 和 `candidate_evidence` 两类
模型 target；两类 schema 不能混用。`select`、初始 feedback-only decision 以及历史独立
`evidence/diagnosis` stage 不进入本轮训练集，除非以后另立明确 ablation。

标准 atomic `audit` 的模型 target 只能包含：

```json
{
  "region_id": "r0",
  "visual_judgment": {
    "t1_state": "building",
    "t2_state": "background",
    "evidence": "One short paired-RGB observation."
  },
  "diagnosis": {
    "audit_checklist": {
      "evidence_sufficient": "pass",
      "target_class_only": "pass",
      "white_pixels_supported": "pass",
      "boundary_alignment": "pass",
      "internal_holes_absent": "pass",
      "changed_object_extent_complete": "pass",
      "fragment_artifacts_absent": "pass"
    }
  }
}
```

合同约束：

- `t1_state/t2_state` 只能是 `building/background/mixed/uncertain`，表示未修改 RGB 中的
  目标类别状态，不表示黑白颜色或 mask 极性。
- checklist status 只能是 `pass/fail/not_applicable/uncertain`；
  `evidence_sufficient` 不能为 `not_applicable`。
- evidence 必须是锚定 T1/T2 可见物理内容的简短自然语言，不能引用 object/change/component
  mask、outline、marker、schema placeholder 或 machine label。
- `change_mask_state` 是 Environment 对当前 proposal 的权威运行时极性，不是 GT，也不是
  模型 target。
- 在 atomic audit target 中，`evidence_quality`、`mask_assessment`、`error_type`、target view、
  动作、坐标、quality、accept/stop 和 finish readiness 均由 runtime 派生，不得重新加入；
  下述 `candidate_evidence` 是具有独立 schema 的 stage-specific 例外。
- 七项 checklist 只保留 status，不恢复历史逐项 prose evidence。

`candidate_evidence` 负责观察 attempted action 的 delta 像素，使用另一套 target：

```json
{
  "region_id": "d0",
  "visual_judgment": {
    "t1_state": "building",
    "t2_state": "background",
    "evidence_quality": "clear"
  }
}
```

其中 RGB state enum 与 audit 相同，但描述的是 attempted-action delta 像素下的物理内容；
`evidence_quality` 只能是
`clear/ambiguous/insufficient`。candidate 的 comparison、benefit、accept/reject 和下一动作
继续由 runtime 联合 attempted action、持久化 diagnosis、delta polarity 与安全门派生，不能
塞进该模型 target。用于纠正“实际改善 candidate 被保守拒绝”的数据必须按
`stage=candidate_evidence` 构建，不能伪装成 atomic checklist audit。

不得使用历史 v12、v11.3 或 select-stage JSON 作为 v12.1 target，即使它们能被强模型解析。

## 3. 三种视觉输入必须分开建样本

数据记录必须保存 `stage`、`attempt_kind` 和 `visual_input_scope`；builder/scorer 必须按这三个
字段的闭集组合派发并校验 schema，不能把三类输入混成一种：

1. `standard`
   - 默认 scope，不等于单一训练 stage。它覆盖首轮普通 proposal audit、首错不是唯一
     evidence-grounding 错误时的 full-context audit repair，以及 `candidate_evidence` 等
     非专用调用。
   - 必须再用 `stage+attempt_kind+visual_input_scope` 区分 target；仅
     `(audit, repair)` 就同时存在 standard full-context atomic repair 和 field-local evidence
     repair，不能把 repair/candidate 样本错删或编成同一种 atomic audit。
   - builder/scorer 至少要显式支持并校验以下组合及 protocol/schema version：
     `(audit, initial|repair, standard)` 与
     `(audit, initial|repair, authoritative_pair_rgb_only_scan)` 使用完整 atomic schema；
     `(audit, repair, authoritative_pair_rgb_only_evidence_repair)` 使用 evidence envelope；
     `(candidate_evidence, initial|repair, standard)` 使用 candidate schema。其他组合默认拒绝，
     不能静默猜测。
   - 使用对应 production backend 的原始 RGB、proposal/mask/geometry 上下文；预测 mask
     只是可错的辅助信息，不能作为标签真值。
2. `authoritative_pair_rgb_only_scan`
   - 仅用于 Environment 生成的 scan tile。
   - 只提供权威 tile box 内的 T1/T2 RGB crop，不提供全黑 mask、marked overview、白色
     geometry crop 或完整九宫格 catalog。
3. `authoritative_pair_rgb_only_evidence_repair`
   - 只有首轮唯一失败为 evidence grounding、且 region ID、states、checklist 和全部运行时
     一致性已通过时才能建立 deep-copy anchor。
   - repair 只看权威 proposal box 内两张 T1/T2 RGB crop，唯一新的语义字段是 evidence。
   - production prompt、训练 target 和后续 strict scorer 要求完整顶层 envelope：匹配 anchor
     的 `region_id`、`visual_judgment.evidence` 和空 `diagnosis`；不能输出裸 evidence 字符串。
   - 当前 runtime merge 的消费边界更窄：只强制 response 是 Mapping、`region_id` 匹配且存在
     `visual_judgment.evidence`，其他新字段不参与语义；不要把这种兼容性当成训练 target。
     runtime 机械保留 anchor 的 states/checklist，合并后重新跑完整 parser；一旦建立 anchor，
     后续 retry 不回退 full-context。

如果训练 evidence-repair 子任务，必须单独标识 scope；evidence 是唯一语义监督字段，但
完整 envelope 和匹配的 `region_id` 是训练/strict-scorer 的结构合同。不能把空 `diagnosis`
当成标准 audit target，也不能让 repair 重写 states/checklist。

## 4. 已完成的代码修改

主要权威文件及职责：

- `change_agent/adapters/staged_verifier.py`
  - v12.1 live path 按 Environment 顺序审计全部 region，不再调用只排序而不筛选的 select。
  - 删除初始 feedback-only model decision，机械派生冗余字段。
  - 增加状态/极性、completion、target-view/editability、candidate-plan/geometry 一致性门。
  - 实现 evidence-only anchor、merge 和 region-level fail closed。
- `change_agent/adapters/stage_backends.py`
  - 定义 compact audit prompt/schema。
  - 实现 scan RGB-only 与 field-local evidence-repair 消息和 crop。
  - local/hosted history 记录 `visual_input_scope`，并保留旧 hosted override 兼容签名。
- `change_agent/verifier_protocol.py`
  - 定义合法 RGB/checklist enum 和 evidence-grounding 错误分类。
  - 拒绝已观测到的 placeholder、JSON literal、纯数值/坐标、snake_case machine label、
    `all_black` 以及 mask/presentation shortcut。
  - 没有把 `black/white` 猜测映射为 `building/background`。
- `tools/analyze_staged_verifier_run.py`
  - 校验 scope 白名单、类型以及 stage/attempt/scope 组合。
- `tools/compare_local_verifier_ab.py`
  - 校验 scope 调用记账、固定五样本身份、输入/模型/工具/环境 hash、commit 和指标来源。
- `tools/run_levir_change_agent.py`、相关 Slurm submitter/worker
  - 固定 `smoke1/e2e3/e2e5` profile、回执、终止信号、GT boundary 和输出 provenance。

Executor locality、hard gate、rollback 与 candidate benefit-without-harm 仍保留；本轮没有用
GT 替换在线 Verifier 判断，也没有重写稳定仓库。

## 5. 最终验证证据

| 阶段 | Job | 输出 | 结论 |
|---|---:|---|---|
| targeted | 51723 | `CA_0801(1)-local-verifier-v12_1-evidence-repair-tests` | 相关测试通过 |
| full regression | 51724 | 同上 | 297 tests 通过 |
| smoke1 | 51725 | `CA_0801(2)-qwen3vl-v12_1-evidence-repair-smoke1` | 8/8 valid，0 regional error |
| e2e3 | 51727 | `CA_0801(3)-qwen3vl-v12_1-evidence-repair-e2e3` | 18/18 valid，0 regional error |
| v11.3 control | 51730 | `CA_0801(4)-qwen3vl-v11_3-control-evidence-repair-harness-e2e5` | 5/5 completed |
| v12.1 treatment | 51731 | `CA_0801(5)-qwen3vl-v12_1-evidence-repair-e2e5` | 5/5 completed |
| CPU comparator | 51762 | `CA_0801(6)-qwen3vl-v11_3-v12_1-evidence-repair-comparison` | comparability 全通过 |

matched e2e5 保持两臂 45 个 initial region 完整覆盖。v12.1 相对 v11.3：audit
`90→60`、repair `45→15`、regional error `45→8`、raw chars 减少 85.4%、backend
latency 减少 82.7%，五个样本 selected 指标均无退化。这是 v11.3 到最终 v12.1 的累计
协议效果，不能全部归因于最后一次 evidence repair。

详细结论见：

- `doc/training_pipeline/QWEN3VL_LOCAL_VERIFIER_FREEZE_0731.md`
- `doc/training_pipeline/QWEN3VL_LOCAL_MODEL_DEVELOPMENT.md`
- `doc/verifier_protocol_maintenance.md`

## 6. 模型能力边界与训练 hard-case 设计

`atomic_audit_valid` 只表示结构和协议通过，不表示视觉判断正确。正式 treatment 仍有：

- `test_78_13` 的 3 个 regional error；
- `test_85_16` 的 5 个 regional error；
- `test_1_1` 把仍含 626 FP 的初始结果错误判断为 finish；
- `test_50_8` 漏掉 166 FN，scan 后仍错误 finish；
- `test_78_13/test_85_16` 各有一个离线实际改善 candidate 被模型保守拒绝。

8 个 regional error 都是完整可解析 JSON，但 states/evidence/checklist 与白区极性或
actionable concrete-state 约束不一致；最终均转为 `uncertain/insufficient`，没有从错误
region 授权动作。不要为这些错误继续堆 parser 词法规则或增加 retry，应在训练集中构造
同类语义一致性、false-finish 和 candidate-benefit 样本。

上述固定样本来自 `test_256`，只能作为 hard-case 分类和冻结回归证据，不能直接混入正式
train split。应从 `train_original` 中寻找同类训练样本；若把 test 样本用于显式 overfit
smoke，必须单独命名并永久取消其泛化评测资格。

## 7. 训练数据来源与尚未完成的工作

已确认的数据根目录：

- train：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/train_original`
- validation：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/val_original`
- 冻结诊断 test：`/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256`

正式 v12.1 训练集尚未构建，新的 builder、label policy、train/val manifest、strict scorer
和数据验收作业仍是下一阶段任务。开始前必须先明确：

1. 如何从 train GT、模型 proposal 与人工/机械规则生成七项 checklist label；
2. 哪些字段是 Environment 事实、哪些是监督标签、哪些只作审计元数据；
3. standard audit/full-context repair、scan、evidence-repair 与 `candidate_evidence` 的比例和
   去重策略；
4. parent image 级 train/val 隔离，防止 256 crop 泄漏；
5. base-model raw response 只作 hard negative 或诊断，不能自动成为 gold target；
6. manifest 中记录源码 commit、prompt hash、原图/label hash、crop box、scope、生成规则和 seed。

冻结实验的 rollout 决策路径是 GT-free；训练阶段可以使用 train split GT 构造监督，但必须在
manifest 中明确其来源，且不得把 GT、GT 派生提示或 test label 泄漏到未来在线 Verifier 输入。

## 8. 禁止直接复用的 0729 文件

以下文件属于旧 select-stage 32 条 overfit smoke，不是 v12.1 audit 数据入口：

- `tools/build_qwen3vl_sft_smoke.py`
- `configs/qwen3vl_sft_smoke_0729.yaml`
- `tools/submit_qwen3vl_sft_prepare.sh`
- `tools/run_qwen3vl_sft_prepare.slurm`
- `tools/submit_qwen3vl_sft_smoke.sh`
- `tools/run_qwen3vl_sft_smoke.slurm`
- `doc/training_pipeline/0729_first_try/`

旧 builder 的 target 是 `levir_select_sft_smoke_v1`，且模型可见 GT-derived candidate mask/
facts；其默认输出还指向稳定仓库的 `CA_0729(1)`。它只能作为数据打包、环境和 overfit smoke
的历史参考，不能原样运行或只改输出路径后冒充 v12.1 数据。训练环境脚本若要复用，必须先
改成新的显式 dataset/output 路径和 v12.1 scorer，并通过 Slurm smoke。

## 9. 新会话的建议起步顺序

1. 阅读本文、最终冻结报告和 `verifier_protocol_maintenance.md`。
2. 确认开发分支、runtime 锚点、Git 状态、数据/模型路径；保留所有已有改动。
3. 先写 v12.1 数据契约与 label policy，再实现新的 builder；不要修改已冻结 runtime 来迁就
   builder。
4. builder 必须复用 production prompt/message 构造，不能手抄一个近似 schema；同时增加
   strict JSON、enum、scope、hash、split leakage 和 deterministic rebuild 测试。
5. 数据预处理、测试、批处理和训练全部通过 Slurm；管理节点只做编辑和轻量检查。
6. 首先提交 CPU 小规模 build/validate，再做单 GPU overfit smoke；最多使用 3 张 GPU，短
   验证优先 1 张。
7. 新输出使用本地仓库 `outputs/` 下不覆盖历史结果的明确目录，并归档 `logs/`、submission、
   terminal receipt、manifest 和 QQ 完成通知。
8. 只有 strict scorer、split/hash/provenance 和小规模 overfit 均通过后，才扩大正式数据量。

## 10. 当前工作树提醒

撰写本文时，工作树已有一组用户进行中的文档整理：四个 0729 文档显示为 tracked deletion，
对应文件位于未跟踪的 `doc/training_pipeline/0729_first_try/`。这组改动不属于本交接文档提交，
不得擅自恢复、删除或混入后续 commit；新会话应先重新检查状态，并按用户后续处理结果为准。
