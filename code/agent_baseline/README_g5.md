# G5

只交付代码和单元测试。未运行模型、API、Slurm、GPU 或真实数据实验。

`loop_json.py --gate5 --verify-stage` 开启。默认关闭。
G5 自动启用 G4，再沿用 G4 启用 G3/G2。需要 `--verify-stage`。
Python 调用可用 `run_session(..., gate5=True)` 或 `Toolbox(..., gate5=True)`。
`--tool-hints off/a/b` 继续单独控制，默认 off；G5 不自动开启 T1.3 工具。

G4 的当前答案提醒、提交与 region#0 求交、前后掩码归档保持原实现。
提醒只提示，不替换当前答案。显式提交、空提交和兜底均保留区域约束。

## 比 G4 多的行为

只有 G5 开启，`segment_text` 和已开启的 `segment_everything` 的 schema
才出现可选字符串 `region`。例如：

```json
{"time":"t2","text":"building","region":"region#0"}
```

区域使用会话内同图的 `Mask` 句柄：`region#k`、`mask#k`，以及旧地图
输入的 `map#0`。`exmask#k` 属于另一张示例图，不适用。未知句柄、实例集、
特征句柄或本轮尚不可见的句柄返回 `bad_handle`，错误文字说明 region 无效。

先解析区域，再照常调用整图分割后端；region 不进入后端参数或缓存地址。
命中缓存后也只裁剪返回的实例，不写回缓存。不传 region 时与 G4 完全一致。
G5 关闭时，region 不出现在 schema；传入会按原参数校验返回 bad_args。

裁剪对每个实例与区域求交。空实例丢弃。沿用 `instances.from_dense` 的
非空规则，至少 1 像素就保留。SAM 提案使用 `min_mask_region_area=0`，
SAM3 实例读取也没有额外面积门限。不使用离线候选队列的 50 像素门限。
保留 ID、分数等属性，重算 idx、area、bbox；断开的交集仍为同一个实例。
结果集 parents 记录区域句柄，不改变当前答案。

返回 `report` 说明 `Results clipped to region#k: removed n instances / x pixels.`。
即使删除数量为零也说明已裁剪。像素数按实例并集去重，与返回的 area 一致。

## 轨迹和离线读取

`steps.jsonl` 的调用 `args.region`、`raw_arguments` 保存原参数。
`provenance.region_clip` 保存：

- region：实际区域句柄。
- instances_before / instances_after / dropped_instances：实例数量。
- pixels_before / pixels_after / clipped_px：裁剪前后并集像素及差值。
- min_area_px：1。

Recorder 继续保存输出实例集及区域掩码。分割原始实例集不另增句柄。
G4 的提交前后掩码仍单独保存。无 region 时不增加 region_clip 或 report。

回放页 `make_demo.render` 识别 G5，显示区域句柄、前后数量和删除量。
`analysis.g5_region_clip.read_clips(session_dir)` 可供归因脚本逐调用读取。
也可离线导出 JSONL；这里只说明用法，本次未分析真实轨迹：

```bash
python -m agent_baseline.analysis.g5_region_clip --run 会话或运行目录 --output 裁剪统计.jsonl
```

`--run` 可重复，递归读取 steps.jsonl。输出包含会话、步骤、调用 ID、
工具名、完整参数、输出句柄及裁剪统计。历史无 G5 字段的调用跳过。
工具目录页可用 `make_tools_page --gate5 --out-dir 目录` 显示新参数；
默认目录页保留原 schema。

## 验证

`tests/test_gate5.py` 使用合成掩码、脚本化规划器和假后端。
复用 G1/G3 请求 fixture，新增 `fixtures/g4_requests.json`。
G4 fixture 用本次输入副本中未修改的 tools.py、loop_json.py 生成，
沿用 test_g4.compatibility_script 的 3 个完整请求；没有模型或 API。
G5 关闭时逐字节比较；G5 开启且未传 region 时，只剔除新增 schema
参数后与同一 G4 fixture 逐字节比较，检查提示和工具结果未变。

在交付根目录设置临时目录和单线程环境，再运行全部单元测试：

```bash
mkdir -p .runtime/tmp
export TMPDIR="$PWD/.runtime/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OVCD_CODE=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_v/ovcd_v0
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
cd code
$PY -m unittest discover -s agent_baseline/tests -t .
```

测试结果及沙箱限制见交付根目录 NOTES.md。
