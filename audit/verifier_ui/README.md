# MLLM 判断器审核前端

顶部提供“MLLM 判断器审核”和“漏检候选审核”两种模式；后者的离线构建与启动说明见本文末节。

只覆盖“MLLM 做判断器”这一环：提示词版本、ICL 示例（零样本 / 少示例 / 多示例 / 多示例批量；示例形式 A / B / A+B）、示例池审核、模型原始输出与指标、示例数量对指标的影响。候选生成、分类和调度智能体不在范围内。exp6k 起的 ICL 实验都在这里审，不再另做审核页。

架构：服务器上一个 SQLite 文件（`verifier.db`）是唯一的数据源。准备脚本登记示例池、开发集和测试集，推理脚本通过 `store.RunRecorder` 写库；`app.py`（FastAPI）读库，只写审核结果和指标缓存；`static/index.html` 是单页前端。图片不打包，按库里登记的路径直接从服务器磁盘读。

## 启动

```bash
pip install fastapi uvicorn numpy pillow
cd audit/verifier_ui
VERIFIER_DB=/path/to/verifier.db uvicorn app:app --port 8765
# 本机：ssh -L 8765:localhost:8765 <服务器>，然后打开 http://localhost:8765
```

没有服务器数据时可以先看合成数据：`python demo_fixture.py /tmp/vdemo`，再用 `VERIFIER_DB=/tmp/vdemo/verifier.db` 启动。合成数据里的图像、标签和分数都是编出来的，故意放了两处划分违规，用来展示检查。

指标第一次打开时计算（每个运行每个数据集约 1–2 秒，含 2000 次自助法），之后缓存在库里的 `metric_cache` 表。样本标签或权重改了之后，删掉这张表的内容即可重算。

## 写库：样本集（exp6k 第 4 部分）

```python
import store
db = store.connect('verifier.db')
with db:
    store.add_sample_set(db, 'exp6k_demos_second', 'demo_pool', 'second', demos)   # 只能来自 train
    store.add_sample_set(db, 'exp6k_dev_second', 'dev', 'second', dev)            # 只能来自 val
    store.add_sample_set(db, 'exp6k_test_second', 'test', 'second', test)         # 只能来自 test
    store.add_prompt(db, 'second_sys_v1_zh', SYSTEM_ZH, note='出处：……；归纳自训练难负例：……')
```

每个样本是一个字典：

| 键 | 含义 |
| --- | --- |
| `dataset`、`split`、`cid`、`image` | 数据集、划分（train / val / test）、单元 id、原图名。划分检查和“同图”判断都按原图 |
| `label` | 1 有变化，0 无变化（伪变化也是 0） |
| `rationale` | 一句理由。正例写变化内容，负例写伪变化类型，例如“屋顶换色，属于无变化” |
| `pseudo_type` | 负例的伪变化类型（季节 / 物候、光照与阴影、色调 / 成像差异、屋顶换色或翻新、车辆等临时物体、配准错位……） |
| `change_kind` | SECOND 正例的“t1 类别 → t2 类别” |
| `semantic_text` | SECOND 两期语义的文字说明，随 B 形式给出。若同时给了 `semantic_t1` / `semantic_t2`（候选区域两期类别直方图 `[{name, fraction}, ...]`），登记和显示时都改用 `store.semantic_line` 的格式：第一行是候选区域，例如“T1 非植被地表 → T2 树木（占候选 74%）”；给了 `semantic_full`（整图两期直方图）时第二行是整图两期主要类别。“只在变化区提供地类”这类说明放在悬停提示里，不进文字。原文存进 `semantic_text_long`。评测脚本给模型的 B 形式文字也请用 `store.semantic_line`，和前端保持一致 |
| `area`、`source`、`det_mean`、`m1_score` | 新增像素面积、候选来源、检测器均值 D、M1-ign3 分数（D 和 M1-ign3 是参照线的分数） |
| `bbox` | 候选在整图中的 `[x0, y0, x1, y1]`，用来画整图位置缩略图 |
| `inputs` | 图像路径：A 形式 `t1`、`t2`（近景，红色轮廓）；B 形式 `full_t1`、`full_t2`、`mask`；可选 `thumb`（预渲染的位置缩略图，不给就用 `full_t2` + `bbox` 现画） |
| `stratum`、`weight` | 分层（`pos2` ② 正例、`pos_other` 其他正例、`hard_neg` 难负例、`rand_neg` 随机负例）和抽样权重（抽样概率的倒数） |

其余键存进 `meta`。重复登记同一个样本集会更新分层和权重，不会碰审核结果。

## 写库：评测运行

```python
from store import RunRecorder, icl_template
with RunRecorder('verifier.db', name='second_dev_plus_AB_k8_s0', model='qwen3-vl-plus',
                 prompt_name='second_sys_v1_zh', prompt_text=SYSTEM_ZH, demos=demos,
                 mode='few', form='AB', pool='exp6k_demos_second', eval_set='exp6k_dev_second',
                 readout='verbal_confidence', strategy='按类别均衡随机', seed=0,
                 template=icl_template(len(demos), SYSTEM_ZH, form='AB')) as rec:
    for q in queries:
        rec.record(q, answer='no', confidence=80, raw_text=text,
                   usage=dict(input_tokens=..., output_tokens=..., cost=...))
```

- `mode`：`zero` / `few` / `many` / `batched_many`；`form`：`A` / `B` / `AB`。
- `readout`：`token_prob`（有 P(yes)）、`verbal_confidence`（只有口头置信度，例如百炼接口拿不到词元概率时）、`answer_only`。AUC / AP 的分数依次用 P(yes)、口头置信度换算的 P(yes)、回答本身。
- `usage` 记一次 API 调用的输入 / 输出词元数和费用（默认币种 CNY），运行结束时汇总到 `run` 表。
- 批量提问：同一张原图的全部候选放一次调用，用 `rec.record_batch(queries, raw_text, per_query=[...], group_key=image, usage={...})`，并在 `RunRecorder` 里写 `batch_by='image'`。整段原始输出存一次，每个候选记批次和位置。
- `template` 是实际发给模型的消息序列，图像写成引用（`demo[3].full_t2`、`query[0].t1`），示例答案写成 `{"type": "text", "ref": "demo[3].answer"}`（前端显示答案和理由），SECOND 语义写成 `demo[3].semantic`。脚本的排布和 `icl_template` 不同时，传自己的模板即可。
- 提示词按全文哈希去重，可以用 `prompt_parent` 记它从哪个版本改来。

已有的 exp3 / exp6j 零样本结果可以用 `import_existing.py` 导入（这些运行没有关联样本集，在前端的“未指定样本集”里）。

## 提示词分组与迁移

`prompt_version` 新增可空字段 `grp`（分组键）、`title`（中文显示名）、`kind`（`template` / `rendered`，空值按模板显示）。`store.connect` 打开旧库时自动补列，已有记录的新字段保持空值；`add_prompt` 可选传入同名参数，原调用方式继续可用。`name`、`id` 和父版本关系保持不变。

`GET /api/prompts` 保留原字段，并返回这三个字段及 `rendered_ids`：模板对应的直接渲染实例 id 列表，渲染条目自身返回空列表。分组顺序为 SYSU、SECOND、SYSU + SECOND 共用、LEVIR、WHU、LEVIR + WHU 共用、未分组，对应键为 `sysu`、`second`、`sysu_second`、`levir`、`whu`、`levir_whu`、空值。

用数据库副本核对现有提示词的名称映射，再按需应用（在本目录执行）：

```bash
python migrate_prompt_meta.py --db /path/to/copy.db          # 只读打印对照表
python migrate_prompt_meta.py --db /path/to/copy.db --apply  # 备份后写入
```

写入前创建 `<db>.bak-prompt-meta`；需要写入时若备份已存在则报错，绝不覆盖。所有字段已一致时重复执行不写库、不再备份。未匹配名称、渲染实例缺少模板父版本或分组与父模板不符都会报错并停止。迁移只更新这三个字段。

## 审核与导出

在“样本集与审核”页逐个看示例：点“通过 / 剔除”，“撤销”回到未审，备注框按回车保存，结果直接写回库里。审核结果按样本存（`sample_review` 表），同一个样本在 64 池和 256 池里共享：任一池里审过，另一池同步显示，导出也按共享结果。旧库里按池存的审核结果在第一次连接时自动迁移过来（同一样本在多个池里审过时取最近一次）。评测前导出：

```bash
python export_demos.py --db verifier.db --set exp6k_demos_second --out demos_second.json             # 去掉剔除的
python export_demos.py --db verifier.db --set exp6k_demos_second --out demos_second.json --only-passed  # 只要通过的
```

## 指标

- **主指标**：② 正例 vs 全部补漏负例的 AUC / AP，按抽样权重还原。另列全部正例的 AUC / AP 和加权准确率。
- **分层**：只分四栏，② 正例、其他正例（①③ 合并）、难负例、随机负例，每栏给不加权的准确率和答“是”比例。
- **参照**：同一批样本上的 D（检测器均值）和 M1-ign3 分数，作为表格里的参照行和曲线上的点线。
- **CI**：2000 次按原图整群自助法（与 exp8s3 一致）。
- 准确率优先用生成的回答，没有时用 P(yes) > 0.5。
- 开发集 / 测试集用页面顶部的切换按钮切换，依据是运行的 `eval_set`。

## 划分检查

示例池只能来自 train，开发集来自 val，测试集来自 test，三者（以及运行用过的示例集）按原图互不重叠。违反时每页顶部标红，单个示例和运行详情里也会标出；示例池单图超过 2 个示例也会标出。

## 数据表

| 表 | 内容 |
| --- | --- |
| `prompt_version` | 最终提示词全文、哈希 id、父版本、备注（出处） |
| `sample` | 所有出现过的候选（示例或查询）及上面那张表里的字段 |
| `sample_set` / `sample_set_item` | 示例池、开发集、测试集；每个样本在集合里的分层和权重 |
| `sample_review` | 每个样本的审核结果（通过 / 剔除 / 备注），所有示例池共享 |
| `demo_set` / `demo_set_item` | 某次运行实际用的示例：模式、形式、k、来自哪个示例池、选法、种子、顺序和提示中写的答案 |
| `run` | 模型、提示词、示例集、评测样本集、读数方式、批量方式、调用次数、词元数、费用、消息模板、git 提交号、完整配置 |
| `call` | 批量调用的原始输出和该次调用的用量 |
| `output` | 每个查询的 P(yes)、logit 差、yes/no 概率质量、解析后的回答、置信度、原始文本 |

## 前端五个视图

1. **提示词**：左侧只列模板，按数据集分为可折叠小节，组内按创建时间排序；主行显示中文名，小字显示原名、id 和运行次数。右侧查看全文、父版本和可展开的渲染实例，实例可与模板逐词对比；按住 Shift 点第二个模板仍可对比。点击模板只更新详情和选中高亮，切换页面时在当前会话保存列表滚动位置和分组折叠状态。未迁移元数据的旧库全部显示在“未分组”。
2. **样本集与审核**：示例池、开发集、测试集的构成统计（正负比例、分层、伪变化类型、变化种类、面积档、来源、划分、单图最多几个），每个样本显示 A 形式近景、整图位置缩略图、可展开的 B 形式（整图、掩码、SECOND 语义）、理由和各字段；示例池可以直接审。
3. **ICL 示例集**：某次运行实际用的示例、顺序和构成。
4. **运行与输出**：按数据集分组，先列 D 和 M1-ign3 参照行，再列各运行的 ② AUC / AP、全部 AUC / AP、准确率、四栏分层、调用次数和费用。点进去看实际发给模型的输入（批量调用会显示同一张图的其他候选）、划分与泄漏检查，以及原始输出表，可按数据集、分层、候选来源、答错 / 误报 / 漏报筛选。
5. **示例数量曲线**：按数据集分面，横轴为示例数（对数刻度），颜色区分少示例 / 多示例 / 多示例批量，线型区分示例形式 A / B / A+B；零样本、D 和 M1-ign3 画成灰色参照线；可以切换 ② AUC、② AP、全部 AUC / AP 和准确率。

## 漏检候选审核

此模式用于讲解冻结候选生成流程。七步页面只有读取和查看操作，使用 `WALKTHROUGH_DIR` 中的 PNG/JSON；接口不会连接审核数据库。原有五个审核 tab 和写库接口保持原样。未设置该环境变量、未生成某划分的 `index.json` 时，显示“未生成”。

### 启动和打开

在 `audit/verifier_ui/` 执行，前端环境使用已有的 exp6-v2-env：

```bash
WALKTHROUGH_DIR=$STAGING/batch54_assets \
VERIFIER_DB=/path/to/verifier.db \
$CONDA_ENVS/exp6-v2-env/bin/uvicorn app:app --host 127.0.0.1 --port 8767
```

本机沿用 SSH 转发：`ssh -L 8767:127.0.0.1:8767 <服务器>`。打开：

```text
http://127.0.0.1:8767/?mode=candidates&ds=second&split=val&img=train_1049.png&step=2
```

URL 保存模式、数据集、划分、图名、步骤；刷新、浏览器前进/后退会恢复这些位置。来源选择、热力图开关和播放进度是当前步骤内的临时状态。顶部切换可返回原审核页面。新页面明确显示“图池共 N 张”。图名搜索、补全、随机选图仅使用当前图池已渲染的图片；池外图名有明确提示。七步漏斗和右侧的“整个划分平均”读取全量 `stats.json`，与图池的大小、典型图比例无关。

第 2、3、5 步及局部示例支持前后对比滑块、逐帧滑块、播放/暂停。第 1 步默认只显示 B，可选 G0.05、SECOND 的 H1 或全部叠加。第 6 步点击候选掩码内部或选择候选编号，显示 T1/T2/新增像素裁剪；重叠像素选最后绘制的候选，列表可访问其他重叠候选。点击使用离线 `pick.png` 中的编号，浏览器只读取 PNG 像素，不计算分割或过滤。②覆盖未达 50% 与完全未覆盖分别统计，只有新增覆盖为零的②目标画虚线框。

### 冻结范围与数据出处

下面路径中的 `STAGING` 指实验暂存目录（`$STAGING`），`CONDA_ENVS` 指 conda 环境目录，`OUT` 指 `$WORKSPACE/Chang-Agent/outputs/CD_1016_probe`（检测器概率等缓存）。

| 数据集 | 验证 / 测试图数 | 冻结单元 | 第 3 步 | 第 5 步 |
| --- | --- | --- | --- | --- |
| SECOND | second_cv 297 / 官方测试 1694 | `batch48/run/second/units/{val,test}.npz` | U5_x4，面积 >1088 像素才切块 | F3A200 |
| SYSU | pool_val 800 / 官方测试 4000 | `batch48/run/sysu/units/{val,test}.npz` | U5_x4，面积 >4056 像素才切块 | F2 |
| LEVIR | 官方验证 64 / 测试 128 | `batch50/run/levir/units/{val,test}.npz` | U5_x16，面积 >3176 像素才切块 | batch50 冻结 F3A200 |
| WHU | 无冻结验证 / 测试 690 | `batch50/run/whu/units/test.npz` | 冻结 `objects_direct_pairs`，没有 U5 切块 | batch50 冻结 F3A200 |

WHU 没有对应验证单元表，前端只提供测试。LEVIR、WHU 的适配调用 `batch50/exp8s8_common.py`，每次构建进程只处理一个数据集，沿用其原来的几何读取规则。

| 步骤 | 读取或复用的内容 |
| --- | --- |
| 0 | SECOND/SYSU 调用 `batch46/exp8s5_common.pixels/geometry`：`OUT/exp8s_batch30{,_sysu}/interim/exp8s/<raw>/<image>/pixels.npz` 的 GT、D、probability，以及 batch34 格缓存；按 batch46 规则重建修正格。LEVIR/WHU 使用 batch50 的 `raw_pixels/geometry`。RGB 调用原 `common.load_rgb`。 |
| 1 | `OUT/exp8_batch18/pool/<raw>/objects_none/*.npz`，SYSU 对应 batch19；来源、成员及几何来自同目录 `images/*.json` 的 `units.objects_none.details`。这些就是原始候选掩码缓存，无需重跑分割。 |
| 2 | 同目录 `objects_direct_pairs/*.npz` 与对应 details。跨来源 IoU≥0.5 或交集/较小面积≥0.8；按 IoU 降序、原编号升序选直接对，每个原候选最多参与一对，合并后不再重新比较。 |
| 3 | 同目录冻结 `U5_x4` / `U5_x16` 与 `parent_object` 元数据。原生成代码 `batch18/exp8_runtime.split_objects`，SYSU 对应 batch19：超过阈值的对象与已缓存 SAM 自动分区求交，块面积至少 10；普通对象保持完整。WHU 保留去重对象。新增像素严格取候选减 D。 |
| 4 | SECOND/SYSU 调用 `batch46/exp8s5_common.load`，读取 batch43 的单元 ID 并复算单元自身 `r<0.1`，验证与缓存补漏集合一致；LEVIR/WHU 调用 batch50 的 `load`。保留冻结库存实际使用的 U5 单元与自身 r 口径。 |
| 5 | 直接调用 `batch47/exp8s6_filter.units/keep`。F3A200：新增掩码最大内切半径≥2且面积≥200，整单元保留或丢弃；F2：新增掩码3×3开运算后面积≥100。逐图核对最终 ID、数量、逐单元新增掩码、面积与 batch48/batch50 的 packed 掩码完全一致；不一致写 `failures/<图名>.json` 并失败。 |
| 6 | 使用上述核对通过的单元；②按 GT 目标去重，以 D 与候选新增并集覆盖目标≥50%算命中，复用 `batch46/exp8s5_common.coverage`。 |

原图与标签目录来自 `Chang-Agent/probe/common.py`：SECOND/SYSU 为 `datasets/lim1/{SECOND,SYSU-CD}/{img1,img2,label_unch0chi255}`；SECOND 按 `slabel1/slabel2` 的类别转换定义 GT 对象。LEVIR 为 `datasets/LEVIR-CD/{val_original,test_original}/{A,B,label}`；WHU 为 `datasets/lim1/WHU-CD/test/{A,B,label}`。LEVIR 验证概率来自 `OUT/exp8_batch18/preds/levir_val/schanger_base/prob`，LEVIR/WHU 测试概率来自 `OUT/preds/<dataset>/schanger_base/prob`。SECOND/SYSU 使用已有像素包里的冻结概率。

需要区分需求简述与冻结实现的两个细节：

- G0.05 的完整候选来自 `p>0.05` 的连通域，再检查面积≥10和 D 已报比例≤0.5；减去 D 后的新增部分才是 `0.05<p≤0.5`。直接对该概率区间重新取连通域会改变冻结候选。
- ①②③是 GT 评估分类。①②对应目标检测覆盖率<10%的漏检块，按到 D 的棋盘距离≤1 / >1区分；③对应目标检测覆盖率≥10%的剩余漏检。⑤为 GT 目标或漏检块<20像素。实际分流读取候选自身 r。batch42 的按成员分流比较不用于替换 batch48 实际冻结的 U5 库存。

各候选集合统计的“②覆盖”统一采用目标≥50%的口径。第 0 步数量是检测器8连通域数，第 1–6 步是对应候选数；切块可能让数量上升。每步均值以冻结划分的全部图片为分母，包含零候选图。

### 两阶段构建、分片和续跑

CPU解释器为 `$CONDA_ENVS/exp5-env/bin/python`，依赖沿用现有NumPy、SciPy、Pillow。全部阶段只用CPU，不调用模型、不安装包。输出只允许写入 `STAGING/batch54_assets` 或其子目录。

1. `stats`：对整个冻结划分计算七步数量、②覆盖、各来源数量/覆盖、典型图特征、各类典型区域的编号和裁剪坐标，并逐图核对第5步冻结掩码。不读取RGB，不调用绘图、PNG编码或保存。按冻结清单的全局序号 `i % shards` 分片；结果写为 `stats_parts/<i>-of-<n>/<image>.json`，每图一个小JSON，便于中断后续跑。
2. `merge`：读取全部统计分片，校验输入和图片清单，写唯一的 `stats.json`。它包含完整划分均值、各步②覆盖汇总、逐图记录、典型图、固定种子随机图及最终图池。缺图片时正式merge失败。②覆盖汇总采用命中目标数之和除以目标数之和；平均数量采用候选总数除以完整图片数，包含零候选图。
3. `render`：只读取 `stats.json` 中选出的图池，通常每划分10张，沿用stats选定的典型区域。渲染后再次核对统计和冻结像素，最后写 `index.json`。该索引仅记录图池和渲染状态，不存均值。应用从同目录 `stats.json` 读取全部划分统计。

全量有7个划分、7673张统计输入、至多70张渲染图片。每划分最多5张典型图，再从其余图片中用种子 **20261028** 无放回随机抽5张；两组不重复。少量验收不足10张时保留实际数量。默认 `render` 拒绝不完整统计，验收需显式 `--allow-partial`；此时页面只显示“统计预览”，全量均值仍显示待统计。

PNG统一采用 `optimize=True, compress_level=9`。T1/T2只各存一张，叠加图和候选裁剪保存相对背景的RGBA差分：未变化的背景像素透明，变化像素保存原来的准确RGB。页面以CSS复用原图及裁剪位置，不计算分割、过滤或新标记。此方式无损保留整图和裁剪分辨率，现有四个数据集最长边均≤1024，无需降采样。`pick.png` 始终保持原始像素编号。

全部PNG目标≤1 GB（十进制）。代码为7个划分各分配 `1_000_000_000 // 7` 字节，写入每张PNG前检查该划分累计空间，超额停止并报错；不降低图像质量或跳过候选来制造成功。重新选池时，只清理旧完成清单中声明、且已不在新池内的渲染文件，保留统计和其他文件。正式运行每划分只有一个render作业，避免同时修改该划分的空间统计和索引。

在本worktree根目录进行轻量自测，每个数据集限1–2张：

```bash
WALK_PY=$CONDA_ENVS/exp5-env/bin/python
WALK_SCRIPT=audit/verifier_ui/walkthrough_build.py
"$WALK_PY" -B "$WALK_SCRIPT" stats --dataset second --split val --images train_1049.png
"$WALK_PY" -B "$WALK_SCRIPT" merge --dataset second --split val --allow-partial
"$WALK_PY" -B "$WALK_SCRIPT" render --dataset second --split val --images train_1049.png --allow-partial --force
```

`--force` 用于输入或代码修改后的重算/重渲染。正常续跑不带它，统计结果和渲染完成清单通过输入大小、修改时间、版本及PNG大小检查后跳过。JSON/PNG先写临时文件再原子替换。`--images` 必须属于冻结划分；render还要求图片在所选图池内，不能用该参数扩充图池。stats的 `--limit` 取冻结清单前N张，render的 `--limit` 取已选图池前N张。限定图片不会改变它们在全局清单中的分片归属。merge的 `--shards` 必须与stats一致。

全量由Claude通过Slurm执行，本次没有提交作业。下面是一个划分的依赖顺序；其他划分分别修改dataset/split。推荐SECOND测试8片、其他划分4片，数组同时最多4个任务。每任务1 CPU、4 GiB、无GPU，stats每片1小时，merge30分钟，render每划分1小时。提交前检查当前分区和空间。

```bash
scontrol ping
squeue -u guisongxia
mkdir -p ../batch54_assets/logs
PARTITION=your_cpu_partition
WALK_UI="$PWD/audit/verifier_ui"
WALK_PY=$CONDA_ENVS/exp5-env/bin/python
STATS_JOB=$(sbatch --parsable -p "$PARTITION" --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --array=0-7%4 --output="$PWD/../batch54_assets/logs/second-test-stats-%A_%a.out" \
  --error="$PWD/../batch54_assets/logs/second-test-stats-%A_%a.err" \
  --wrap="'$WALK_PY' -B '$WALK_UI/walkthrough_build.py' stats --dataset second --split test --shard \$SLURM_ARRAY_TASK_ID --shards 8")
MERGE_JOB=$(sbatch --parsable -p "$PARTITION" --dependency="afterok:$STATS_JOB" \
  --cpus-per-task=1 --mem=4G --time=00:30:00 \
  --output="$PWD/../batch54_assets/logs/second-test-merge-%j.out" \
  --error="$PWD/../batch54_assets/logs/second-test-merge-%j.err" \
  --wrap="'$WALK_PY' -B '$WALK_UI/walkthrough_build.py' merge --dataset second --split test --shards 8")
sbatch -p "$PARTITION" --dependency="afterok:$MERGE_JOB" --cpus-per-task=1 --mem=4G --time=01:00:00 \
  --output="$PWD/../batch54_assets/logs/second-test-render-%j.out" \
  --error="$PWD/../batch54_assets/logs/second-test-render-%j.err" \
  --wrap="'$WALK_PY' -B '$WALK_UI/walkthrough_build.py' render --dataset second --split test --force"
```

只读接口：`GET /api/walkthrough` 返回图池索引与 `stats.json` 的统计摘要；`GET /api/walkthrough/<dataset>/<split>/<图名去扩展名>` 返回图池内的七步JSON，池外图片返回404；`GET /walkthrough-assets/<相对路径>` 提供根目录内的PNG/JSON。未完成统计/渲染或两个版本不匹配时页面显示未生成。

### 典型图与典型区域选择规则

merge 在完整统计记录中确定性地选最多5张典型图。先按“尚未展示的特征数”降序贪心补齐，再按是否保留②、总特征数、最终命中②数降序，最后按图名字典序。特征为：发生合并、有可切大对象、有普通对象、含①②③、补漏/精修两组、去掉细条、去掉小块、过滤后保留含②的候选。小规模预览可不足5张；全量样本选择只用于讲解，不据此调整冻结参数。

每步局部示例同样可复现：

- 去重：面积最大的已合并对象；同面积按原成员编号。
- 大对象：至少切出2块，优先第二大子块面积最大，再按父对象面积和编号；普通对象选面积最大者。裁剪外扩1.35倍、最小边64，裁到图内。
- ①②③：每类面积最大的漏检块，同面积按编号；只突出本块、D、D膨胀环及与本块相交像素最多的候选，标出该候选 r 和分流结果。没有相交候选时明确标为 r 无定义。两种分流另各取最大面积单元。
- 过滤：被丢弃且半径<2作为细条；其余被丢弃单元作为面积不足例（F2同时显示开运算后面积）；保留例优先过滤后仍含②，再按新增面积和编号。列出过滤前后面积和最大内切半径。
- 某图缺少某类例子时明确提示，并链接到当前图池中按图名字典序找到的对应例图；图池均缺少时显示“暂无”，不会通过例图链接扩大图池。WHU 明确标注未启用切块。

### 验证

使用真实构建结果检查接口、所有PNG、图中候选点击编号与冻结掩码、内嵌JavaScript：

```bash
$CONDA_ENVS/exp6-v2-env/bin/python -B \
  audit/verifier_ui/walkthrough_check.py \
  --assets $STAGING/batch54_assets
```

该脚本直接调用路由，校验所有 JSON/PNG、未设置目录、缺失图片、路径越界和不创建审核数据库；从 `index.html` 用 HTMLParser 提取脚本并执行 `node --check`。有 socketpair 权限的环境可另加 `--http`，用现有 FastAPI TestClient 实际请求 JSON/PNG；无权限时在 socketpair 预检处明确失败。它没有浏览器布局或鼠标事件验收能力。实际验收图名、数值、CPU/空间估算和未执行项目见本 worktree 的 `check/REPORT.md`。

### 两阶段资源估算

2026-10-28仍只用原来的7张真实验收图。单图stats约0.03–0.17秒；按7673张直接外推约9分钟，按冻结候选平均数量修正后约17分钟。考虑缓存预热偏差、NFS和分片启动，建议预留 **0.5–1个单CPU作业小时**，32个统计分片，最多4个并发。这是小规模外推，未执行全量stats。

7张图的663张PNG合计 **15.68 MB**；每划分扩大到10张直接外推约 **156.75 MB**。按冻结候选平均数量修正后约 **601 MB**，再留25%余量约 **0.75 GB**，低于1 GB目标。统计JSON按当前记录大小外推，逐图分片与合并文件合计约0.05–0.1 GB。实际典型图可能更复杂，运行时仍执行每划分约142.86 MB的PNG硬上限；超额会明确失败。实际生成名单、统计覆盖、空间和未执行项目见 `check/REPORT.md`。
