# Qwen3-VL 本地模型开发分支

2026-10-16 维护说明：本地模型代码已收拢到工作区 `Chang-Agent`；旧工作目录已移除。下文保留当时的隔离开发过程，后续从统一目录继续开发。历史输出已按用户要求清理。

## 隔离边界

本地 Qwen3-VL 开发使用分支 `feat/local-qwen3vl-verifier`，基线为提交
`7dd5bb5f1bd2286c930c096429fce9cf1ed7ad3d`。该基线保留已经用于闭源
Qwen3-VL-Plus 的 staged verifier 行为；本地模型适配不得直接改变对应稳定分支。

开发 worktree 位于：

```text
/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl
```

实验修改按单一变量拆分提交和评测。闭源模型可运行不等于 Environment 事实、派生
枚举或冗余字段关系已经被证明正确；强模型可能补偿提示词或协议歧义。因此
Environment 字段注入、枚举收紧、schema 精简必须分别审计，不能与格式修复混成一个
提交。

## JSON completion guard

本地 Transformers backend 启用 JSON completion guard：生成文本尚未形成当前 stage
可解析的完整 JSON object 时，禁止配置的 EOS token；一旦完整，强制下一 token 为
EOS。该 guard：

- 不补括号、不清洗 response，也不修改模型已经生成的 token；
- 不改变闭源 Bailian backend；
- 不约束 `change_mask_state`、RGB state、checklist、`error_type` 或其他语义；
- 达到 `max_new_tokens` 仍不完整时继续由原 parser fail closed；
- 在 backend audit 中记录 `json_completion_guard=true`。

这项修改只解决本地 base 模型提前输出 EOS 导致的外层 JSON 未闭合。后续是否由
Environment 注入字段，必须先验证字段来源、枚举映射、轨迹一致性以及对可执行动作的
影响，再作为独立提交和消融实验处理。

## Environment 顺序的全量审计（v11.2）

`CA_0730(7)-qwen3vl-base-protocol-fix-e2e5` 的五条轨迹表明，v11.1 的
`select` 会按每批最多三个 region 循环，直至所有 45 个 Environment region 都被选中，
随后仍逐 region 审计。它没有减少 audit 集合，只改变顺序并生成一条
`screening_hypothesis`。实际发生 18 次 select 和 90 次 audit；90 次 audit 中 39 个
region 最终仅因 `screening_resolution` 与 `mask_assessment` 的固定映射不一致而失败。

v11.2 因此只移除这组无信息增益的关系：

- region 集合、完整覆盖门和逐 region 原子 audit 均不变；
- audit 顺序直接采用 Environment 已确定的 proposal 顺序；
- 不再调用模型 select，也不再生成或要求回填 `screening_hypothesis` /
  `screening_resolution`；
- 未审计 region 仍不能作为正确证据，任一 region 协议失败仍按
  `uncertain_region` fail closed。

这一提交不改变模型填写的 checklist、`change_mask_state`、RGB state、
`mask_assessment`、目标视图或动作解析，便于把“移除穷举排序”作为单一变量验证。

## 注入 Environment mask state（v11.3）

`change_mask_state` 表示当前 proposal 在程序 change mask 中的极性，不是建筑变化
GT。其来源已沿调用链核对：`attach_verifier_regions` 在当前 state 上同步创建 proposal；
`EvidenceRecord.from_proposal` 将 `present`/`delta_added` 映射为 `white`，将
`missing`/`scan`/`delta_removed` 映射为 `black`，其余明确为 `mixed`；atomic audit
持有不可变的同一条 record，执行几何也继续引用该 record。它因此属于运行时事实，
不需要模型重新抄写。

v11.3 从模型输出 schema 删除 `change_mask_state`，parser 直接把
`record.change_mask_state` 写入 `EvidenceJudgment` 轨迹。模型仍在
`ENVIRONMENT_FACTS` 中看到该字段以理解白/黑 proposal 的语义，但不能覆盖它；若输出
旧字段会因严格 key 集合失败并按 region fail closed。本提交不改变 RGB 枚举、
checklist、`mask_assessment` 或动作决策，单独验证 Environment 事实注入。

## 本地 compact audit（v12，最终收紧为 v12.1）

0730 的 90 次 base audit 产生约 19.9 万字符响应，模型经常为七个 checklist 项重复
同一句证据；39 个 `screening_resolution` 冲突之外还出现 checklist 与
`mask_assessment` 冲突，且大量正确判断仍输出非空/大写 `target_view`。这些字段并不
提供独立监督信号：`mask_assessment` 和 `error_type` 由 checklist 唯一确定，
`evidence_quality` 由 `evidence_sufficient` 唯一确定，安全 target view 可由 RGB state
与 Environment seed editability 确定，初始 decision 调用只生成不参与决策的反馈文本。

v12 引入、v12.1 保留的模型输出只包含：

```json
{
  "region_id": "r0",
  "visual_judgment": {
    "t1_state": "background",
    "t2_state": "building",
    "evidence": "one short observable statement"
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

运行时继续严格校验 region ID、RGB enum、七项 checklist enum 和 proposal 极性，并派生
`change_mask_state`、quality、`mask_assessment`、`error_type`、target view、动作几何和
finish gate。可执行 FP/FN 若没有明确 RGB state 或没有唯一安全 target 会停止授权，
不会猜测；RGB 语义指向的时相与 Environment seed editability 不一致时也不会静默切换
到另一时相。白色 proposal 的明确无变化 RGB、黑色 proposal 的明确变化 RGB 均不能以
全 pass 完成，除 `evidence_sufficient` 外全为 `not_applicable` 也派生为 uncertain。
候选改进后的 full-state proposal 即使为空，也必须有合法且为零的 Environment
`initial_audit_uncovered_pixels` 才能授权 finish；覆盖事实缺失、类型错误或仍有未覆盖
像素都会停止 finish，不能用“空目录”绕过 completion gate。
初始反馈由已验证 diagnosis 生成，不再增加 decision 模型调用。轨迹字段
从 `selected_region_ids` 改为 `audited_region_ids`，可视化工具仍兼容读取历史轨迹。

因此 `CA_0730(6)-qwen3vl-base-select32` 及对应 select-only SFT32 只保留为历史格式
诊断，不能再作为正式训练数据口径。后续数据集应直接监督上述 v12.1 compact audit；在
本次推理/评测验证完成前不开始构建正式训练集。

### 0731 base 2B 的 mask-color 歧义

首轮 v12 五样本作业 `51095` 完整审计了 45 个 Environment region，但 89 次 audit
调用中只有 1 个 region 通过协议；44 个 region 在 initial 后进入 repair 并仍失败。
原始 JSON 和 checklist 能被正常抽取，失败集中为 RGB enum：白色 `present` proposal
几乎都输出 `t1_state=black,t2_state=white`，全黑 `scan` tile 几乎都输出
`black/black`，个别 T1 mask 还输出 `green/black`。这说明 base 2B 把展示用的 object/
change mask 颜色写进了应表示物理目标类别的字段，不是 Environment mask state 注入
错位，也不能安全地把颜色机械映射成建筑/背景。

历史 v12 prompt 因此保留原严格 parser，只增强语义边界：schema placeholder 直接列出
`building/background/mixed/uncertain`，正文说明两个 state 只能描述未修改 RGB 中的
目标类别、禁止颜色名，并把白/黑规则统一写成显式的 Environment
`change_mask_state` 条件。invalid RGB repair 会再次列出合法 enum 和颜色反例，仍不
清洗、不猜测或改写模型 token。该改动需要新的五样本 A/B 证明是否足以恢复协议遵循；
首轮 `51095` 只作为 preliminary 失败轨迹，不能作为冻结依据。

同轮 control `51094` 在 verifier 前因 control clone 的相对路径推导不到 workspace
`Chang-Agent/third_party/OmniOVCD`，以 `ModuleNotFoundError: sam3` 失败。实验 worker 现显式把固定
workspace 的 OmniOVCD、SegAgent 和 SimpleClick 源码目录加入 `PYTHONPATH`；control
与 treatment 使用相同且不继承外部污染的路径，不修改固定 control commit。A/B
submitter 不再接受外部 worker override，而是强制使用 harness worker；submission、
manifest 和 terminal receipt 三处记录并交叉校验该 worker 的 SHA-256，避免 control
clone 内的旧 worker 重新推导错误 workspace。

### 候选新增区域的时相方向门

候选复核不能只判断 delta 像素上“是否存在建筑变化”。`positive_point`/FN box 把像素
加入指定时相 mask，RGB 变化方向必须与 `previous_action.target_view` 一致：T1 只接受
`building -> background`，T2 只接受 `background -> building`。相反方向虽然仍是建筑
变化，但会把像素加到错误时相；运行时将其标记为 introduced false positive 并判为
`worse`，不得进入 replan 或更新 accepted state。这里的
`introduced_false_positive` 是“向错误时相 object mask 加入 target-object FP”的
候选伤害载体，不等同于 binary change mask 上“无真实变化”的 FP。删除不支持区域的
negative action
仍要求 delta 上不存在真实目标变化。该规则由 Environment 已授权动作的 target view
和模型 RGB observation 联合派生；若与持久化 initial diagnosis 的类型或时相不一致也
直接按证据不足拒绝，不允许模型另行输出或改写方向字段。

### Audit evidence grounding

修复前正式五样本作业 `51114` 暴露旧 schema 把 evidence 示例写成了完整 JSON 字面量，
base 2B 会逐字复制，结构校验却只能看到“非空字符串”；部分响应还把
`change_mask_state` 或全黑 presentation 当作 RGB 无变化证据。v12.1 将输出模板改为真正的
`<SHORT_OBSERVABLE_RGB_EVIDENCE>` placeholder，prompt 要求 region-specific 的配对 RGB
可观察事实，并明确禁止用 proposal/change-mask 颜色或存在性支撑 T1/T2 语义。

v12.1 parser 要求 evidence 必须是 JSON string，拒绝旧模板、placeholder，以及已观测到
的窄模式 mask/presentation color shortcut；scan tile 也不能以全黑 mask 推断无变化。
专用 repair 仍失败时按 region 记为 uncertain。该门只能机械排除已知 shortcut，不能
证明一句看似具体的 RGB 描述真实，因此模型视觉判断仍需可靠 label 监督。

正式 v12.1 首次运行 `51135` 又暴露了一个小模型格式退化：未加引号的 evidence
placeholder 与 base 2B 在 `visual_judgment` 后额外开启一层 JSON object 的现象高度
相关；首样本 16 次 audit 中 14 次因此未能闭合并跑满生成上限，耗时反而高于 v11.3
control。作业在保留完整首样本轨迹后主动取消，不作为正式 A/B 结果。输出模板现把
evidence sentinel 写成带引号的 JSON string；若模型原样复制，既有 parser 仍会拒绝并
触发 repair。这是针对已观测格式的最小修复假设，是否消除退化必须由 fresh A/B 重跑
确认。同时 parser 新增窄门，拒绝整段 JSON literal/collection、纯数字、RGB tuple、
box 或 coordinate list 充当 evidence，要求以自然语言描述成对未修改 RGB 中可观察的
区域事实。该门不拒绝包含颜色词的真实屋顶描述，也不声称能机械证明描述为真。

单样本 fresh smoke `51275`（输出
`CA_0731(11)-qwen3vl-v12_1-evidence-guard-smoke1`）证明只修 evidence sentinel 仍不足：
8 个 initial region 中 5 个把 `diagnosis` 错嵌入 `visual_judgment`，随后重复数字或坐标直至
生成上限；最终仅 3/8 个 atomic audit 有效，15 次 audit 共耗时约 243 秒。轨迹还暴露
`<RGB_IMAGE_1>`/`<RGB_IMAGE>` 媒体 marker 会漏过旧 evidence 门并进入 executable plan。
因此该 smoke 只证明 fail-closed 执行门有效，不支持冻结模型行为，也没有继续提交正式
五样本 A/B。本轮把 compact audit 模板中的 T1/T2 state 和全部 checklist status sentinel
都改成带引号字符串，使整段模板可直接被 `json.loads`；parser 同时拒绝有长度上限的
angle-bracket image/schema placeholder 并触发专用 repair。prompt 要求 evidence 同时锚定
可见物理内容和 T1/T2 时相，不把颜色本身当作建筑语义；parser 仍允许明确描述未修改
RGB 中白色屋顶等真实外观，但明确引用 component/change/object mask 颜色仍 fail-closed。
fresh smoke 必须重新验证格式闭合、全区域覆盖和最终
evidence 合同，不能把 `51275` 当作通过结果。

后续 fresh smoke `51642`（输出
`CA_0731(16)-qwen3vl-v12_1-json-template-smoke1-rerun`）证明枚举 sentinel 加引号只解决了
部分结构退化：初始 malformed audit 从 5 个降至 3 个，但仍有 1 次 repair 未闭合并跑满，
最终仅 2/8 个 atomic audit 有效、6 个 regional error，14 次 audit 约耗时 204 秒，因此
不得晋级 `e2e3`。四条 malformed 响应都在 evidence 位置生成
`<RGB_IMAGE_OBSERVATION>`，并把 `diagnosis` 错嵌入 `visual_judgment`；这说明 angle-bracket
evidence placeholder 本身仍在诱导 base 2B 生成媒体 marker。本轮把模板改为合法 JSON
字符串 `REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT`，同时由 parser 将复制该 sentinel 继续
视为 schema boilerplate 并 repair。该单变量假设仍必须先通过新的 `smoke1`，不能用作
冻结结论。

`51135` 的取消还暴露了实验回执缺口：旧 worker 在 SIGTERM 后进入 EXIT trap 时可能
取得 `rc=0`，从而把 Slurm 已取消且仍有 `.run_incomplete` 的作业误写成
`status=success/exit_code=0`。worker 现显式捕获 TERM/INT，分别以 143/130 退出并标记
`status=cancelled`；非成功回执在 best-effort QQ 通知前通过同目录临时文件加原子 rename
落盘，成功回执仍只在完整 epilogue 后发布。comparison 继续只接受真实
`status=success`、`exit_code=0` 且完成产物齐全的两臂。

为避免把临时样本参数误当成正式评测，提交器和 worker 现在只接受闭集
`EVALUATION_PROFILE=e2e5|e2e3|smoke1`：`smoke1` 固定为 `DEV-SMOKE-1-V1` 和
`test_20_15`；`e2e3` 固定为 `DEV-CORE-3-V1` 以及 `test_20_15`、`test_1_1`、
`test_50_8`，用于覆盖格式退化和历史 FP/FN 错误 finish；`e2e5` 固定为
`DEV-EXTENDED-5-V1` 及既定五样本；profile、评测 ID 和样本顺序同时写入
submission、terminal receipt 与 input manifest，并由 baseline 与
protocol 两类 summary 重新校验和持久化；正式 comparison 只接受固定的 `e2e5` 身份并
交叉绑定全部产物。不得通过额外参数替换闭集样本，`smoke1` 和 `e2e3` 必须显式指定
非空 `OUTPUT`，避免落入带 `e2e5` 标签的默认路径。fresh 验证先用单卡 `smoke1`
检查 JSON
是否正常闭合、8 个 region 是否全覆盖、生成长度与回执/完成标记是否正常；只有该门
通过后才运行固定 `e2e3`；仅当三样本覆盖、协议和安全门也通过，才顺序提交重新绑定
同一 harness hash 的 control/compact `e2e5` 正式两臂。正式 comparator 仍只接受 `e2e5`，
不会把 smoke/e2e3 冒充冻结证据。这样能尽早拦截系统性退化，同时不并行占用不必要的 GPU。

non-angle sentinel 版本的 fresh `smoke1` 作业 `51650` 已达到 8/8 initial audit
有效、0 repair、0 regional/validation error，backend latency 从旧合法模板作业 `51642`
的约 205.1 秒降至约 25.8 秒。随后固定 `e2e3` 作业 `51654` 完成 3/3 样本与
18/18 initial region，只有 1 次 repair、0 regional/validation error；但人工合同审计发现
`test_50_8` 的 scan evidence 使用 `no_visible_building_in_both_images` 等 whole-string
snake_case 标签，`s7` repair 后的 `all_black` 仍被 parser 接受，`test_20_15/r3` 还引用
了 marked overview 的 `red outline`。因此 `atomic_audit_valid_records` 只表示结构校验成功，
不能解释为 RGB grounding 成功，`51654` 不晋级正式 `e2e5`。

这些漏口采用不依赖 GT 的窄修复。parser 拒绝整句 underscore-delimited machine label、
纯 `all black` 类颜色状态，以及颜色词与 outline/contour/marker/border 等展示标注在短窗口
内共现；真实 `white roof`、`red metal roof` 等物理外观仍允许。scan tile 的 box 已由
Environment 权威给出，且 scan 只在两期预测 object mask 都为空时产生，因此 audit 输入
从 marked global overview、三张全黑 mask crop、全白 geometry crop 和两张 RGB crop
精简为仅两张未修改的 T1/T2 tile RGB crop。该输入精简消除重复 Environment presentation，
裁剪严格使用权威 tile box；若极小 tile 不满足 provider 的 11 像素下限，只在框内以最近邻
复制像素，不再向邻接 tile 扩张；每次 atomic scan prompt 也不再重复序列化完整九宫格
catalog。该路径还会在任一 T1/T2/change mask 非空或 box 缺失时 fail closed。不把预测
mask 当 GT，也不能机械修复 `test_50_8/s8` 的真实 FN 或 `test_1_1` 的 FP；后两者
仍应作为训练 hard case 和 strict scorer 目标。修改后必须重新通过 CPU regression、
`smoke1` 和 `e2e3`，才可提交最终 matched `e2e5`。

首个修复后 `smoke1` 作业 `51682` 没有过门：虽然工具动作全部安全拒绝、selected 指标
无退化，但只有 7/8 initial audit 有效，2 次 repair 中 `r0` 两次都未闭合，`r2` 初次响应
在合法 JSON 后反复生成说明文字直至上限；10 次 audit latency 约 137.3 秒。根因是本轮
把 scan 的自然语言/annotation 说明误加进所有非 scan audit prompt，破坏了此前
`b52a03f` 已验证稳定的 `test_20_15` prompt。最终方案恢复 non-scan prompt 的原有字节级
措辞，只在 `audit_kind=scan` 分支精简图像、catalog 和追加 instruction；parser 守卫仍对
所有 audit 生效。该轨迹还暴露 `a white area appears in the T2 mask` 会绕过旧 color shortcut
正则，因此新增仅拒绝显式 `T1/T2 ... mask` 引用的窄门，不拒绝 `T2 RGB` 中可见白色屋顶。
`51682` 保留为 failed smoke 证据，不得用于晋级。

恢复 non-scan 指令后的 fresh `smoke1` 作业 `51689`（commit `3a7146a`，输出
`CA_0801(2)-qwen3vl-v12_1-scan-grounding-smoke1-rerun`）恢复了正常生成速度：8 个
initial region 中 7 个有效，只有 1 次 repair，audit latency `25.101823s`、全部 backend
latency `27.847401s`，且三个候选都被安全拒绝、selected 指标无退化、无工具失败。但它仍
未通过 smoke 门：`r3` 初次 evidence 引用 marked overview 的 `red outline`，repair 虽收到
明确禁止语句，仍在同一组 overview/object/change/component presentation 上输出
`white pixel mask indicating a change`；两次都被 parser 正确拒绝，最终留下 1 个 regional
error。该样本全是 `present` region，因此 `51689` 不能验证 scan RGB-only 路径，后者仍需
在包含 `test_50_8` 的 fresh `e2e3` 中审计。

后续修复不再叠加词法启发式，而把这类 retry 收窄为 field-local evidence repair。只有当
首轮除 evidence 外的 region ID、T1/T2 states、七项 checklist 及全部运行时一致性约束均可
独立通过时，runtime 才授权专用 retry；repair 只接收 Environment 权威 box 内的两张
未修改 T1/T2 RGB crop，不再接收 marked overview、object/change masks 或 component
geometry。极小框仅在框内最近邻放大。repair 仍返回 audit 顶层 JSON 以复用 completion
guard，但 runtime 只取新的 evidence，机械保留首轮 states/checklist 后重新执行完整 parser；
anchor 会被 deep copy，避免 backend 原地修改嵌套字段；一旦建立 anchor，后续所有 retry
始终 field-local，不得回退成 full-context semantic rewrite。若无法建立可靠 anchor，则继续
走原 full-context repair 并 fail closed。专用 prompt 使用固定正向措辞，不回显 validation
error、坏 evidence、mask/presentation 诱导词、proposal catalog 或 Environment envelope。
轨迹新增 `visual_input_scope`；analyzer 校验其 stage/attempt 组合，comparator 校验白名单和
完整 call accounting，使固定代码版本下的轨迹可以审计 runtime 选择的输入合同。该改动
必须先通过 CPU regression 和新的 `smoke1`，再用 `e2e3` 同时验证 non-scan repair 与
九宫格 scan，不得直接晋级正式 `e2e5`。

candidate 还必须匹配 verifier 上一轮持久化的实际 executable plan：diagnosis、动作
allowlist、target view、动作类型和像素几何必须自洽且与 attempted action 完全相同。
缺失 plan、跨进程无状态自举、时相/动作/几何替换会在 candidate MLLM 调用前按
uncertain 拒绝；非 finish 的 identical-state candidate 也不能继承上一轮 accept。
由于 evidence contract 和 candidate acceptance gate 都发生了兼容性收紧，当前口径为
protocol `v12.1`、experiment arm `v12_1-compact`、schema
`staged_verifier_environment_ordered_compact_audit_v12_1`；历史 v12 轨迹只保留为修复前
对照，不能与 v12.1 训练 target 混用。

### 0801 最终分级验证与冻结

field-local evidence repair 的最终提交为 `491532f`。CPU targeted Job `51723` 和全量
regression Job `51724` 均成功，后者 297 tests 全过；`smoke1` Job `51725` 达到 8/8
atomic audit valid、0 regional error，固定 `e2e3` Job `51727` 达到 18/18 valid、
0 regional error，并实际覆盖 RGB-only scan 和 evidence-repair scope。

正式 matched e2e5 在同一 `gpu48`、A800、模型、五样本、seed、工具和 harness 下顺序运行：
v11.3 control Job `51730` 为 95 calls、45 repair、45 regional error、backend latency
`1312.657716s`；v12.1 treatment Job `51731` 为 65 calls、15 repair、8 regional error、
backend latency `226.456989s`。CPU comparator Job `51762` 证明全部固定输入/环境/provenance
可比，raw response chars 降低 85.4%，backend latency 降低 82.7%，episode elapsed 降低
72.8%；两臂 45 个 initial region 覆盖一致，逐样本 selected 指标均无退化。这是
v11.3→最终 v12.1 的累计协议效果，不能全部单独归因于 field-local repair。
离线 GT 只在 rollout 后用于复核：`test_20_15` 的三个有害候选均被拒绝，但
`test_78_13`、`test_85_16` 各有一个实际改善候选也被保守拒绝，说明 base 模型的
candidate benefit 判断仍需训练。

treatment 剩余错误只位于 `test_78_13` 的 3 个 region 和 `test_85_16` 的 5 个 region。
它们都是完整、可解析 JSON，但 states/evidence/checklist 与白区极性或 actionable concrete
state 约束不一致；每项跑满 initial + repair 后均转成 uncertain/insufficient，completion
gate 为 false，未授权错误 region 的动作。由于首轮都不是唯一 evidence-grounding 错误，
保持 full-context `standard` repair 而不建立 field-local anchor 符合合同；同一正式 run 中
另有 5 次 field-local evidence repair 和 10 次 RGB-only scan，说明新 scope 正常工作。
field-local response 只提供新 evidence；runtime 机械保留 deep-copy anchor 的 states/checklist
并重新运行完整 parser。唯一 `action_attempt_error` 是 `test_78_13` 没有下一项授权动作，
以 `no_executable_action` fail closed，不是 action payload/protocol 或工具错误。

因此不再增加词法规则或 retry。当前可冻结 v12.1 数据格式、parser、Environment/
Executor/rollback 安全门及评测 harness，并开始构建训练集；不能冻结或宣称 base 2B 语义
能力可用。`test_1_1` 的 FP finish、`test_50_8` 的 FN finish、上述 8 个 regional error 及
语义含混 evidence 均作为训练 hard cases。最终简明结论与完整路径见
`doc/training_pipeline/QWEN3VL_LOCAL_VERIFIER_FREEZE_0731.md`。
