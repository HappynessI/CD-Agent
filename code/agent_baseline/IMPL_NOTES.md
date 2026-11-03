# batch69 BF 实现与维护说明

交付目录为本目录 `agent_baseline/`。安装时整体放到 `CD-Agent/code/agent_baseline/`。
本次只修改暂存目录；未安装到目标仓库，未提交作业、调用外部 API 或加载模型。
开始与结束检查时，目标仓库均显示已有未跟踪的 `code/agent_baseline/`。

## 实现范围与模块边界

- `state.py`：每图 Session；mask/inst 共用单调序号。数组独立复制、只读；mask 缓存全图 8 连通编号，实例筛选保留 ID。
- `instances.py`：保留原有稀疏实例表示与方向性最大单实例覆盖规则；增加实例 ID 和可选并集连通域基准。
- `view.py`：原图裁剪、最小 64 像素区域、长边 768 等比显示、28 像素外边框刻度、细红轮廓、避让编号、JPEG q92。小于 64 的原图使用整图范围，不补造像素。
- `tools.py`：全部指定工具与 OpenAI function schemas；统一校验和错误返回；内部 provenance 与公开结果分开。同轮句柄限制由循环传入的句柄快照执行。
- `backends.py`：旧 SAM3 缓存、SAM3 worker、共享 SAM Prompter、本地/API 核实器。模型均延迟加载；共享 Prompter 的锁覆盖整个调用。
- `planner.py`：urllib 请求、六次总尝试上限/响应、超时 180 秒、指数退避与抖动；每图总尝试上限 180。凭据仅在请求发送时读取，不进入记录。
- `loop_json.py`：30 次成功响应预算、每轮最多执行 8 个调用、顺序执行、合法 submit 后跳过、完整 tool 回复后追加图像消息。保留初始图对及最近三个 view 图对。每图异常独立兜底，已有 final 文件直接跳过。
- `recorder.py`：唯一在线 GT 消费者；按不可变句柄缓存像素评测，对象统计只在初始/最终计算；评测不返回 Session。保存原始请求/回复、调用状态边、JPEG、mask PNG、packed instance NPZ，以及最终指标和费用。
- `fixed_routes.py`：直接调用 Toolbox 的同一执行入口生成 imc；验证集门限扫描与冻结参数应用。
- `summarize.py`：失败会话也参与整集统计，包含路线、修改统计、T5.1 初始比较和 `--errors` 原始诊断。
- `workers/`：标准库 HTTP 服务，仅监听回环地址。ask 单推理线程、最多 8 对、50 ms 合批；SAM3 推理和缓存写入共用一把锁。
- 两个 sbatch 脚本：固定环境、日志目录、退出通知；agent 脚本清理代理变量、启动和检查 worker、退出时回收 worker，支持 `USE_LOCAL_WORKERS=0`。

`common/render/prompts/sam_tools/sam3_t7/mllm` 都直接 import，未复制源码。它们内部采用脚本式绝对 import，故 `__init__.py` 将相邻 `ovcd_v0` 加入模块搜索路径；暂存环境使用现有源码目录，也可用 `OVCD_CODE` 覆盖。增加的 `__init__.py` 仅支持包导入、直接脚本执行及 unittest discovery。

## 相对 TASK 的偏离与解释

1. **系统提示一处措辞调整。** 指定原文的分数说明含隔离测试禁止外发的词组。实际使用 `Tool scores provide limited evidence:`，后面的三种分数含义和限制原样保留，不增加路线建议。提示文件和实际展开文本都进入 input 快照。
2. **SAM3 在线截断门限。** 默认仍为 `conf=0.1`；仅当工具请求 `score_thr<0.1` 时使用 `conf=score_thr`，并写入独立缓存路径和元数据。固定 0.1 会先丢弃用户明确请求的低分实例，因此不能诚实实现该参数。
3. **Slurm 日志目录。** `#SBATCH` 不展开 shell 环境变量，静态指令使用任务指定的默认输出根；自定义 `AGENT_RUN` 时必须在提交命令中覆盖 stdout/stderr 路径。脚本另将运行日志和 worker 日志写到实际 `$AGENT_RUN/logs/`。
4. **缺失输入的统计。** RGB 无法读取、但 GT 可读时，由 Recorder 直接归档空预测与错误，仍计入汇总。GT 本身不可读时仅能留下输入错误记录，汇总会明确失败，无法制造有效指标。T5.1 初始资产缺失时，预测会话仍计入总指标，但初始对比字段为 null 并列出缺失图，避免把兜底空图当初始化对照。

此前讨论中的默认 false、union matching、inst inspect、自由 question 等与 TASK 冲突的建议均未采用：这里默认 current 更新、`MATCH_BASIS='raw'`、mask-only inspect、固定 VERIFY。

## 运行入口（本次未执行实验）

从安装目录可直接运行以下入口，或从其父目录用 `python -m agent_baseline.<module>`：

```text
loop_json.py --task T1.1xC0 --split val --config NAME --ask-backend none
loop_json.py --task T5.1 --split val --config NAME --ask-backend local --ask-worker URL --sam3-worker URL
fixed_routes.py imc --split val --procs 1
fixed_routes.py tune --split val --procs 1
fixed_routes.py tune --apply-test --procs 1
summarize.py --config NAME --task T5.1 --split val --errors
```

正式计算应通过 Slurm。CPU 脚本接收 Python 入口和参数；agent 脚本接收循环参数，多组参数用 `:::` 分隔并串行运行。提交前创建共享日志目录并检查控制面、资源和重复作业。`BAILIAN_CSV` 必须由提交环境提供；脚本不内置文件路径。关闭 worker 时应显式传 `--ask-backend api` 或 `none`。这两种模式仍可使用既有 SAM3 缓存及 CPU SAM 提示；需在线 SAM3 而没有 worker 时返回不可用。

默认输出根取 `AGENT_RUN`，未设置时为任务指定目录。会话产物路径含 config/task/split/stem。imc 初始化 PNG 写入 `init/sam3_imc/<split>/`。tune 报告写入 `fixed/tune_val.json`；`--apply-test` 只读选中值，不重扫，并检查匹配基准一致。

每个 `messages.jsonl` 行对应一次实际尝试，含 round/attempt ID、完整消息、工具及参数、原始响应/错误类型、usage 和耗时。API ask 尝试也以 backend 标记写入该文件，并关联 tool_call_id。图像均为实际发送 JPEG 的本地引用；`clean_messages(..., encode=True)` 可恢复发送图像内容。认证头从不记录。API ask 的部分成功用量在后续区域失败时仍计入。

费用按任务指定的统一标价计算；cached 是 input 的子集，先扣除再计价。`tool_calls` 统计模型声明的全部调用，包含错误、超额和 submit 后跳过的调用；逐条 status 可以区分。`planning_rounds` 只计成功规划器响应，API attempts 包含规划器与 API ask。

summary 的修改工具集合为 combine/morph/remove_region/add_region/refine_region；修改收益是这些成功调用造成的 current 状态边 ΔF1 之和的逐图平均，中间候选未更新 current 时为 0。T5.1 的整集前后差另外从累计像素计数计算，未用逐图均值代替。

## 已执行验证

全部使用小型合成图、注入的假模型/假客户端，在任务指定解释器本机运行。
测试目录中的临时图像和轨迹随测试自动清理。完整真实输出见 `tests/unittest_output.txt`。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B \
-m unittest discover -s agent_baseline/tests -t . -v
```

最终实际输出末尾：

```text
----------------------------------------------------------------------
Ran 43 tests in 1.362s

OK
```

覆盖任务要求的实例匹配、坐标/ID/current/回退、同轮句柄、局部 refine、受限表达式、工具预算与消息顺序、留图、替换 GT 后外发请求逐字一致、隔离负例、导入边界、固定路线一致性；另验证退避、缓存分流、完整调用锁、API 核实器读数/费用、微批顺序、SAM3 packed 缓存及空检测结果、资产回读和失败会话汇总。SAM3 worker 与旧接口一致，在 scores 为空时不读取 masks，并正常保存空实例缓存。
前两轮实际结果分别为 `Ran 30 tests in 1.116s / OK`、`Ran 39 tests in 1.207s / OK`；扩充测试对应新增的后端和轨迹检查。

另运行 `bash -n` 检查两个 sbatch 文件，以及循环、固定路线、汇总和两个 worker 共五个入口的 `--help`，均成功。

## 未验证部分

未调用真实规划器/API 核实器；未读取真实分割缓存做实验；未加载 SAM、SAM3 或本地 Qwen；未启动真实 HTTP 服务、提交 Slurm 或发送通知。模型显存共存、真实吞吐、API 服务对 seed/并行工具参数的实际支持、网络环境和数据资产完整性尚未验证，seed 接受状态记录为 `not_reported`。本次没有验证集/测试集结果或费用实测，标价费用公式仅用假 usage 验证。
