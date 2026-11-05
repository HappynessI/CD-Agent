# Agent baseline 通用化接口

默认仍为 `target=building`、`change_direction=any`，Python 工具关闭。
`tests/fixtures/g1_prompts.json` 中的默认 schema、系统提示、任务指令来自 G1；
新增的 `verifier` 项来自冻结的 `code_g` 源码，包含范围卡、伪变化类型及三个验证器提示。
这些字面量由单元测试保护，不能随通用化改写默认措辞。

## 会话与范围卡

`Session` 保存 `target`、`change_direction`、`instruction`、`constraint`。
方向支持 `any`、`appeared`、`disappeared`。`ask` 的公开参数仍只有 `regions`。
系统提示、工具说明、`ask` 与固定验证阶段从会话取目标和方向。
`ovcd_v0.prompts` 的原有全局导出继续代表 building/any。

`scopes/<target>.json` 定义单复数名称、范围卡、方向专属 counts、伪变化类型、
对应问法和 SAM3 缓存门限。缺卡报错。`building.json` 保留原有数值，
`tree.json` 是未经过实验校准的示例；其缓存配置为空，必须调用匹配图像的 worker。
现有 building 缓存的数据格式及目录不变；其他类别通过按文本命名的 worker 缓存。

本地验证器 worker 可用 `--target` 与 `--change-direction` 固定一个任务范围；
请求中的问题必须与其范围一致。混合目标/方向的清单可使用逐会话配置的 API 验证器，
或拆成范围一致的清单并使用对应本地 worker。默认本地 worker 的问题不变。

## 清单

`loop_json.py --tasks FILE` 接受 JSONL，每行字段：

```json
{"id":"image.png","image_t1":"A/image.png","image_t2":"B/image.png","instruction":"Find the changed objects.","task_type":"T1.1xC0","constraint":null,"target":"building","change_direction":"any","init_mask":null,"gt_mask":"label/image.png","meta":{}}
```

图片、初始掩码和标注的相对路径以清单所在目录为基准。ID 是输出目录标识；
从该 ID 对应行的 `image_t1` 取图像文件名，用原有图名公式计算种子，不依赖行序或 ID 的命名。
`--task` 可筛选任务类型；不提供 `--tasks` 时保留旧入口。
清单路径、图片路径、标注路径及 `meta` 都不进入模型消息；
`instruction` 和 `constraint` 是任务作者明确提供给 Agent 的文本。
标注只交给 Recorder 打分，不进入 Session、Toolbox 或子进程。

`analysis/make_levir_tasks.py` 生成验证集两个任务的清单；默认初始掩码目录为
`init/sam3_imc_p85/val`，可用 `--init-root` 指定。两种任务指令保持 G1 原文。
显式图片路径匹配旧数据集路径时复用旧缓存及 SAM embedding；其他图片交给支持
`image_path` 的 SAM3 worker，并使用独立的按源路径命名的缓存。自定义图片的 SAM
prompt 用完整 CPU predictor，不能复用按数据集图名保存的 embedding。
`summarize.py --tasks FILE` 使用清单长度核对会话数，并从仅供记录器的输入档案读取标注路径。

## Python 开关

`--python` 才公开 `python(code, set_current=true)`；关闭时调用也会被拒绝。
子进程通过 `masks[handle]`、`instances[handle]`、`images['t1'/'t2']` 获取只读数组，
可使用 `np` 和 `ndi`，调用 `save_mask(array)` 返回新掩码。单次最多输出 32 张。
父进程登记句柄；当前回复内不能使用刚生成的句柄，提交仍走已有 `submit` / 验证阶段。

每次调用使用独立临时目录、新解释器和精简环境，无 Session、标注路径或轨迹目录。
预加载输入并关闭文件后安装 Linux seccomp 规则，禁止新文件访问及路径写操作、联网、创建进程和
读取其他进程内存。依赖 `libseccomp.so.2`；加载失败时拒绝执行生成的代码。
上限为 5 秒 CPU（从导入完成后算起）、60 秒墙钟（含解释器启动和导入）、2 GiB 地址空间；不继承用户 site 或 `PYTHONPATH`。
额外包的延迟导入可能因禁用文件访问而失败；应使用预加载的 NumPy/SciPy 操作。
`summarize.py` 输出 Python 调用会话数、成功会话数、调用次数、成功次数、
按调用成功率和按会话成功率。无调用时成功率为 null。

## 离线回放

`analysis/replay_g1.py SESSION --output-dir DIR` 或
`--config CONFIG --task TASK --limit N --output-dir DIR`。
SESSION 可为会话目录或其 `messages.jsonl`。用 `--reference-code` 指定冻结的 `code_g`，
`--run-root` 指定批量档案根目录。输出目录必须在原始会话目录外。

规划器请求：`planner.py` 归档的是去掉 `_view`、图片 URL 为本地文件路径的请求；
实际 HTTP 请求把这些 JPEG 文件原字节编码为 base64。回放读取归档 JPEG 原字节，
不重新编码归档图像。新代码生成的图片单独渲染，并比较完整 base64。
请求参数、schema、消息、字段顺序及字符串空白都参与比较。
差异报告列出首处字段路径、字符串偏移、长度和两侧片段。

验证器请求：G1 的 `ApiAsk.one` 删除了归档事件中的 `request`，只保留回复和尝试信息。
因此无法从现有档案证明原始验证器网络请求逐字节相同。脚本用冻结版本的请求构建器、
渲染函数、归档组件掩码、原图和记录选项独立重建期望请求，再比较新请求和所有图片字节；
报告明确标记为重建比较。如果档案含验证器请求，则直接比较归档请求。

工具正常执行；SAM3 优先读已有缓存，缺失时用归档实例输出；SAM prompt 用归档掩码。
G1 没有保存 refine 的原始 SAM 解码掩码，因此 refine 只能回填归档的完整工具结果，
不能据此验证 refine 内部实现。记录的外部 SAM 错误也直接替换；这些次数单独统计。
其他工具返回文本、生成掩码和最终提交均与档案核对。
脚本注入记录回复并拦截意外 HTTP 请求，不初始化模型，不读原始标注，
只在隔离临时输出中使用零数组满足记录器接口。批量报告保留通过数及失败列表，失败退出码为 1。

## 已做的验证

单元测试在 Slurm 计算节点通过。G1 的 384 个会话全部回放通过：比较 6,423 个规划器请求和
21,926 个验证器请求，另有 151 次归档后备调用。第 8 次 Codex 讨论指出的边界：
验证器请求是重建比较；SAM prompt、refine 和部分外部失败用归档结果；
回放不走清单命令行、新种子计算、Python 和非默认类别。
Python 开启的三次重复（P1）已完成，结果见 `doc/agent_baseline/README_ai.md` 第 10.2 节。

## 换新任务前要做的事

- `fixed_routes.py` 固定 building 且合并两个方向，不能直接作为别的类别或单方向的固定对照。
- `code/ovcd_v0/prompts.py` 的 BOX 仍写 building；当前 Agent 的 SAM 提示工具不调用它。
