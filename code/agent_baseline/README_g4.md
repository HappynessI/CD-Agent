# G4

本次只交付代码和单元测试。未运行模型、API、Slurm 或真实数据实验。

`loop_json.py --gate4` 隐含 `--gate3` 和 `--gate2`，需要 `--verify-stage`。
默认关闭。`--tool-hints {off,a,b}` 独立控制用途说明，默认 `off`，仅对
`--tools-t13` 开启的工具生效。两个开关关闭时保留 G1/G2/G3 请求内容。

## 当前答案提醒

`gate4.latest_generated_mask(session, calls)` 是纯函数，按实际调用和每次
调用的 `handles_created` 顺序查找新建的 `mask#`。所有产生掩码的工具都计入，
不受 `set_current=false` 影响。给定的初始掩码和 `region#0` 不计入；没有给定
初始掩码时，工具生成的 `mask#0` 计入。引用旧句柄不更新顺序。

G4 在最后 `REVIEW_REPLIES=3` 次回复期间的报告，以及预算补报告中检查：
当前句柄与最新句柄不同时，追加当前答案、最新掩码的像素数及来源工具。
当前答案为空时写 `Your current answer is empty.`。提醒只比较句柄，即使两张
掩码像素相同也提醒。自动报告保留报告前的当前答案；兜底继续提交当前答案，
没有当前答案则提交空掩码。显式 submit 的原有提案行为保留。

自动提醒写入 `steps.jsonl` 的报告 note，工具报告中的提醒写入调用的
`result_text`。实际请求继续由 recorder 记录到 `messages.jsonl`。

## 提交区域约束

G4 且会话具有 `region_mask` 时，在实际提交处将答案与 `region#0` 求交，
显式 submit、`submit(empty)`、预算耗尽及错误兜底共用 `clip_submission`。
给规划器的提交前报告仍报告原候选的框外像素和方向违反；方向违反只报告。
最终验证和评分使用裁剪后的答案。

原掩码与裁剪后掩码都保存在 recorder 的 `masks/` 中。裁剪产生新句柄，
包括删除像素数为零的情况。`final.json` 增加：

- `pre_region_handle`：求交前的提交句柄。
- `pre_region_mask_path`：求交前掩码，相对会话目录的路径。
- `region_clipped_px`：删去的框外像素数。
- `submission_report`：K 大于零时为 `Removed K pixels outside region#0 at submission`，否则为空字符串。

已有 `final_handle`、`mask_path` 指向裁剪后答案。显式提交的工具结果包含
非空的提交报告；兜底报告写入 summary。无 region 的会话不增加裁剪字段。

## 用途说明

a 版逐字使用 README_ai.md 10.11 的草稿。组合句只放在 `compare_features`
末尾，`segment_semantic`、`match_exemplar` 各追加原文；`segment_everything`
保留原说明。b 版在同样三处追加短祈使句，全文见 `gate4.TOOL_HINTS` 与
交付根目录 `NOTES.md`。两个版本只改变 description，参数及其他请求内容不变。
按工具名部分开启 T1.3 工具时，也只修改已开启工具的 description。

## 离线季节范围卡

`scopes/any_change_season.json` 完整保留原卡，并在 `not_counts` 追加指定的
季节颜色句。原卡已有季节外观、阴影和光照条款，这次保留并追加，以便比较
明确这一句带来的影响。在线默认仍使用 `any_change.json`。

`analysis/t13_ask_select.py --scope any_change|any_change_season` 默认原卡，
两个设置都固定使用已选 `describe` 问法，不重新选择问题。范围卡只改变
SYSTEM 和范围卡字段，VERIFY、CORRESPONDENCE 和伪变化类型保持一致。
answers 的 settings 记录 scope 与完整提示词；不同卡使用不同输出目录，
不能复用另一张卡的已保存答案。val 继续要求同一问法和范围的开发设置。
prepare 的候选缓存不依赖 scope，可共用。
历史 Python 调用 `answer_questions` 未传 scope 时保留原来的三问选择行为，
便于既有分析代码和测试继续使用；命令行始终显式传入所选 scope，固定 describe。

## 本次可复现检查

从交付根目录执行，每步一条命令。CPU 为 1，内存估计小于 1 GB，无 GPU。
本节仅运行轻量单元测试和参数帮助；完整数据、API 比较或模型实验另按项目
Slurm 规范提交，本次未执行。

1. 创建当前目录内临时目录，CPU 1，内存小于 10 MB。

```bash
mkdir -p .runtime/tmp
```

2. 设置单线程和当前目录内的临时路径，CPU 1，内存小于 10 MB。

```bash
export TMPDIR="$PWD/.runtime/tmp" PYTHONPATH="$PWD/code" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

3. 新增 G4 单元测试，CPU 1，内存估计小于 512 MB。

```bash
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -m unittest agent_baseline.tests.test_g4 -v
```

4. 直接相关回归，CPU 1，内存估计小于 1 GB。

```bash
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -m unittest agent_baseline.tests.test_g3_t53 agent_baseline.tests.test_tools_t13 agent_baseline.tests.test_loop agent_baseline.tests.test_tools agent_baseline.tests.test_generalize agent_baseline.tests.test_replay agent_baseline.tests.test_t13_task agent_baseline.tests.test_t12_c1t -v
```

5. 查看在线循环开关，CPU 1，内存估计小于 256 MB。

```bash
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -m agent_baseline.loop_json --help
```

6. 查看离线比较参数，CPU 1，内存估计小于 512 MB。

```bash
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -m agent_baseline.analysis.t13_ask_select --help
```

G1 使用既有 `g1_prompts.json`。新增 `g3_requests.json` 是从未修改的冻结
`R/code_t` 执行脚本化 G3 单元会话得到的三个完整规划器请求，包含区域报告、
方向约束、`set_current=false` 候选和提交。JSON 序列化后逐字节比较。
已有 replay_g1 的单元测试继续验证离线回放；本次不运行历史全量回放。

## 合并位置

主要逻辑位于新文件 `gate4.py`。`loop_json.py` 只增加开关、调用记录、报告
追加和兜底裁剪；没有修改 tasks.py 或 state.py。`tools.py` 仅增加 schema
参数传递和实际 submit 裁剪，`recorder.py` 仅追加最终裁剪字段。
