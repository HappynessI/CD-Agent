# Agent baseline：给 AI 看的开发文档

日期：2026-11-03。读者：服务器上负责实现和运行的 Claude 会话，以及之后接手的任何 AI 会话。
给 Ethan 看的版本是同目录的 `index.html`（图 > 表 > 文）。两份文档描述同一件事，改一份就要同步改另一份。

---

## 0. 一句话

在 LEVIR-CD 上做第一个**真正由 Agent 调度工具**的变化检测 baseline：Agent 拿到两期影像和一句自然语言指令，自己调用原子工具生成 building 变化掩码，自己决定要不要核实、补漏、修边界，最后提交。整个过程留下完整轨迹。

## 1. 背景：为什么要做这个（先读）

- 研究方向已经从「接在检测器后面优化掩码」转为「变化检测 Agent 接管整个工作流」。创新点：① Agent 接管 CD 工作流；② Agent 能优化掩码（一类任务）；③ 可能有一个 benchmark。
- 之前的 `code/ovcd_v0/`（11 轮，结果见 `results/ovcd_v0_baseline.md`）是**程序编排**：初始化方法写死（M-C-I / I-M-C / ReA-OVCD），A→B→C 顺序写死，MLLM 只回答 yes/no 和画框。设计文档 `doc/ovcd_agent_v0.md` 里的 M5「MLLM 自己选动作」一直没做。
- 本 baseline 和 ovcd_v0 的根本区别：**没有任何「一键出变化掩码」的工具**。M-C-I、I-M-C 被拆成原子工具，由 Agent 自己组合；A/B/C 变成 Agent 可以选择调用的修改工具。
- 任务体系（另一条线在讨论，未定）：提示形式（给定类别 / 自然语言 / 视觉示例）× 目标层次（L1 单类 … L4 需要外部知识），「已有掩码」的优化作为并列一类。**本 baseline 只做 LEVIR + building，但接口必须能接受自然语言指令，并能接受一个给定初始掩码**，以后不改框架就能扩到这些任务。

## 2. Ethan 定下的开发规范（必须遵守）

1. **文档两份**：本文件（给 AI，方便快速上手）和 `index.html`（给 Ethan，图 > 表 > 文）。每完成一个里程碑，两份都更新。
2. **完整轨迹**：从输入的图 + 指令开始，到规划器每一步的输入输出、每个工具的输入输出，再到最终掩码和评测结果，全部保存在服务器上（格式见第 7 节）。另外挑**一份高质量轨迹**用于展示，放进仓库（第 8 节）。
3. **目录**：文档都在 `doc/agent_baseline/`；代码放 `code/agent_baseline/`；结果放 `results/agent_baseline/`。
4. **和 GPT 合作**：在固定节点找 Codex 讨论（第 10 节），讨论要点写进文档。实现细节由你判断，写明理由即可，不必停下来问 Ethan。
5. **数据集**：只用 LEVIR-CD。验证 64 对用于一切选择和调试，测试 128 对每个配置只评一次。

## 3. 任务定义

| 项 | 定义 |
|---|---|
| 输入 | `T1`、`T2`（LEVIR 1024×1024 RGB）+ 一句指令 `instruction` + 可选的初始掩码 `init_mask`（本 baseline 不给） |
| 默认指令 | `"Find all buildings that changed between T1 and T2 (newly built or demolished). Output a binary change mask."` 另跑一个中文版做稳定性对照 |
| 输出 | 二值变化掩码 + 轨迹 |
| Agent 可见 | 系统提示、工具说明、building 口径卡（复用 `code/ovcd_v0/prompts.py` 的 `SCOPE_CARD`）、预算 |
| Agent 不可见 | 数据集名、真值、任何评测数字、「哪条路线最好」的提示 |
| 训练 | 不训练 |
| 真值 | LEVIR 二值标签，`code/ovcd_v0/common.py::load_gt` |

**关键约束：真值只在记录器里用来算每一步的 F1，绝不能出现在 Agent 的任何输入里。** 写一个单元测试检查规划器收到的消息里没有任何 F1/IoU/GT 字段。

## 4. 工具集

原则：
- 每个工具只做一件事。**禁止**提供 `run_mci()`、`run_imc()`、`run_reaovcd()` 这类组合工具，也不提供 SChanger 等在 LEVIR 上训练过的检测器（另一种设定再说）。
- 大对象（掩码、提议集、特征图）存在工具服务的**状态仓库**里，按句柄引用：`mask#3`、`props#1`、`feat#2`。工具只返回句柄 + 简短统计（个数、面积、分数分布）。
- 只有 `view` 返回图像。
- 每个工具有 OpenAI function calling 格式的 JSON schema（名字、描述、参数）。工具描述就是 Agent 能看到的全部说明，写清楚输入输出和适用场景，**不写**「哪个组合效果最好」。
- 工具就是普通 Python 函数，带 schema；JSON 循环和 CodeAgent 共用同一份实现。以后发 benchmark 时再套 MCP 外壳，工具本身不改。

| 类别 | 工具 | 参数 | 返回 | 底层实现（已有） |
|---|---|---|---|---|
| 看 | `view` | `region`（可选 xyxy，默认全图）、`overlay`（可选 mask 句柄）、`label_components`（bool） | T1、T2 两张图（放大到 768² 以内），可叠加红色轮廓和连通域编号 | `code/ovcd_v0/render.py` |
| 看 | `inspect` | `mask` | 连通域清单：编号、面积、外接框，若有来源分数一并给出；最多列 50 个，按面积降序 | `common.components` |
| 分割 | `segment_everything` | `time`（t1/t2） | `props#k`：无类别 SAM 提议集，个数和面积分布 | SAM ViT-H；第一版直接读 M-C-I 导出缓存（`common.MCI`，`queues.read_dump`），每图约 600 个 |
| 分割 | `segment_text` | `time`、`text`、`tile`（0 或 512）、`score_thr`（默认 0.5） | `inst#k`：实例集，个数、分数 | SAM3（`code/ovcd_v0/sam3_t7.py`，已有 text="building" 的缓存；其他文本要现算） |
| 分割 | `segment_prompt` | `time`、`box` 或 `point` | `mask#k`：一个目标 | SAM ViT-H 提示模式，`code/ovcd_v0/sam_tools.py`（嵌入缓存 + CPU 解码） |
| 特征 | `extract_features` | `time`、`model`（dinov2） | `feat#k` | DINOv2；同上，第一版可读 M-C-I 缓存 |
| 比较 | `compare_features` | `feat_t1`、`feat_t2`、`props` | 给提议集每个提议附上两期特征差异分数（−cos），返回分布 | M-C-I 的 C 步，缓存里已有 |
| 比较 | `match_instances` | `inst_t1`、`inst_t2`、`iou_thr`（默认 0.25） | `appeared#k`、`disappeared#k`、`unchanged#k` 三个实例集 | OmniOVCD overlap_presence |
| 语义 | `classify` | `props`、`time`、`text` | 给每个提议附上该时相属于 `text` 的概率 | SegEarth-OV，M-C-I 的 I 步，缓存里已有 building 两期概率 |
| 语义 | `ask` | `regions`（掩码句柄 + 连通域编号列表，**支持批量**）、`question` | 每个区域的 P(yes)、类型、一句理由 | Qwen3.5-4B 本地 + `code/ovcd_v0/mllm.py` 的 yes/no 读数 |
| 运算 | `select` | `set`、`where`（如 `score>0.57 and p_t1<0.5`）、`area`（min/max） | 子集句柄 | numpy |
| 运算 | `to_mask` | `set` | `mask#k`（并集） | numpy |
| 运算 | `combine` | `op`（union/intersect/subtract）、`a`、`b` | `mask#k` | numpy |
| 运算 | `morph` | `mask`、`op`（open/close/fill_holes）、`radius` | `mask#k` | scipy |
| 修改 | `remove_region` | `mask`、`ids` | `mask#k` | — |
| 修改 | `add_region` | `mask`、`other`（掩码或实例句柄 + 编号） | `mask#k` | — |
| 修改 | `refine_region` | `mask`、`id`、`prompt`（box/point，可选） | `mask#k`：该连通域用 SAM 重分割后替换 | v0 的 T6 |
| 结束 | `submit` | `mask`、`note` | — | — |

用这套工具：
- **M-C-I 路线** = `segment_everything(t1/t2)` → `extract_features` ×2 → `compare_features` → `classify` ×2 → `select` → `to_mask`。
- **I-M-C 路线** = `segment_text(t1,"building")`、`segment_text(t2,"building")` → `match_instances` → `combine(union, appeared, disappeared)`。
  注意：DynamicEarth 原版 I-M-C 用 APE 做识别（测试 69.67），这里用 SAM3，数字会不同；要单独报。
- 用固定脚本把这两条路线拼出来，就是本 baseline 的「无 Agent」下界，也顺便验证工具拆分没有引入偏差：M-C-I 路线必须复现测试集 53.76（差 0.2 以内）。

可以按需要增删工具，但每次增删要写进第 11 节的变更记录，并说明理由。

## 5. 两种 Agent 形态（都做，同一套工具）

| 形态 | 每一步模型输出 | 实现 |
|---|---|---|
| **JSON 循环**（主线候选 1） | 一个或多个并行的 function call | 自己写，约 300 行 ReAct，`code/agent_baseline/loop_json.py`；OpenAI 兼容接口，本地 vLLM 和百炼共用 |
| **CodeAgent**（主线候选 2） | 一段 Python，里面调用工具函数 | smolagents `CodeAgent`，`code/agent_baseline/loop_code.py`；工具包装成 smolagents tool；代码执行用它的本地沙箱，限制 import |

哪种当主线，看验证集上的 F1、步数、失败率再定，写进第 11 节。

**CodeAgent 要额外记录**：模型写出的代码原文；代码里每一次工具调用（在工具包装层记录，和 JSON 循环同一格式），这样两种形态的轨迹可以用同一个回放页显示。另外统计「写死脚本」比例：一段代码里对所有区域按阈值批量删改、之后不再调用 `view`/`ask` 的图有多少。

## 6. 规划器模型与预算

- **两个规划器并行跑**（Ethan 2026-11-03 确认）：
  - 百炼 `qwen3.7-plus`（之前已有 LEVIR 图像授权）。**全量跑之前**先在 5 张验证图上量 token，估出 192 张图 × 两种形态的费用，写进第 11 节并告诉 Ethan，确认后再全量跑。
  - 本地开源 VLM：先查 `models/` 下有没有 ≥7B、支持工具调用的 Qwen VL 模型；没有就把建议下载的模型和大小报给 Ethan。用 vLLM 部署，开启 tool calling。
- `ask` 工具背后固定用本地 Qwen3.5-4B，不随规划器变。
- 预算（每张图）：最多 30 次规划器调用；上下文里最多保留最近 8 张图（更早的图换成文字占位「[image dropped: view#5]」）；超出就强制 `submit` 当前最新的掩码，轨迹里记 `submitted_by: "budget"`。没有任何掩码时提交空掩码并记 `submitted_by: "empty"`。
- 温度 0；每张图种子固定。
- GPU：SAM 嵌入、M-C-I 缓存、SAM3 building 缓存都在 `ovcd_v0_run/` 下，能复用就复用。开跑前估 GPU 小时，单轮超过 8 小时就自己做减法并写明。

## 7. 轨迹格式（服务器上的完整轨迹）

目录：`$AGENT_RUN/<config>/<split>/<image_stem>/`，`$AGENT_RUN` 默认 `/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run`。`<config>` 例如 `json_qwen37plus_v1`。

```
<image_stem>/
  input.json          # 两期图路径、instruction、init_mask（无则 null）、系统提示全文的 sha1 与路径
  messages.jsonl      # 规划器每一次请求的完整消息（含图像路径引用）和完整原始回复，逐条追加；能逐字复现对话
  steps.jsonl         # 每一步一行，见下
  images/             # view 返回的图、每一步的掩码叠加图（PNG）
  masks/              # 每个 mask 句柄的 PNG（mask_<k>.png）
  final.json          # 最终掩码路径、submitted_by、本图 F1/IoU/P/R、步数、token、耗时
```

`steps.jsonl` 每一行：

```json
{
  "step": 3,
  "planner": {"request_id": "...", "input_tokens": 5123, "output_tokens": 212, "seconds": 2.1,
               "thought": "模型的推理文字（如果有）", "raw": "原始回复全文",
               "code": "CodeAgent 写的代码（JSON 循环为 null）"},
  "calls": [
    {"tool": "segment_text", "args": {"time": "t2", "text": "building", "tile": 512},
     "ok": true, "error": null, "seconds": 0.8,
     "result_text": "inst#2: 41 instances, score median 0.71",
     "result_images": [], "handles_created": ["inst#2"]}
  ],
  "current_mask": "mask#4",
  "eval": {"f1": 0.62, "iou": 0.45, "delta_f1": 0.07}
}
```

- `eval` 只由记录器写，用来画「每一步 F1 怎么变」，**绝不回传给规划器**。
- `current_mask` 定义：最近一次被 `submit` 之外的修改/生成工具产出的整图掩码句柄（由记录器按规则判断，规则写进代码注释）。
- 汇总：`$AGENT_RUN/<config>/<split>/summary.json`（整集 F1^c、IoU^c、P、R 用 `common.evaluate` 累计；整目标漏检率、误检率；逐图平均 F1；平均步数、token、耗时、费用；失败率；路线分类统计）。
- 路线分类（每张图一个标签，规则写死在代码里）：`mci_like`（用了 segment_everything + compare_features）、`imc_like`（用了两次 segment_text + match_instances）、`mixed`（两者都用）、`other`。另记「是否调用过修改类工具」和修改带来的 ΔF1。

## 8. 展示用的高质量轨迹

- 测试集评完后，从**测试集**里挑 1 份（再备选 2 份），标准：
  1. 正常 `submit`，不是预算强制提交；
  2. 至少用到 3 类工具（看 / 分割 / 比较或语义 / 修改）；
  3. 至少有一次「看图后发现问题并修正」，且这一步 ΔF1 > 0；
  4. 最终单图 F1 不低于该配置的整集 F1；
  5. 步数在 8–20 之间，读得完。
- 挑选理由写进 `results/agent_baseline/showcase/README.md`，明确写「这是挑出来的展示样例，不代表平均水平」，并附上同一配置的一份典型失败轨迹作对照。
- 仓库里放：`results/agent_baseline/showcase/<config>_<image>/`，包含 steps.jsonl、input.json、final.json、所有 images/ 和 masks/ 的缩小版（长边 ≤ 512，总量 < 15 MB），以及一个单文件回放页 `replay.html`：左边是 T1/T2/真值，中间是步骤时间线（每步一张卡片：模型说了什么 → 调了哪个工具、参数 → 返回什么 → 掩码变化和 F1 曲线），右边是当前掩码叠加图。

## 9. 评测与对照

在测试集 128 对上，每个配置只评一次，报：

| 行 | 说明 |
|---|---|
| M-C-I 路线（固定脚本，原子工具拼出） | 应复现 53.76 |
| I-M-C 路线（固定脚本，SAM3 版） | 新数字 |
| DynamicEarth I-M-C（APE，已有复现） | 69.67，引用 |
| ReA-OVCD / ReA + 编排式 v0（r10） | 84.52 / 85.37，引用 |
| Agent：JSON × {qwen3.7-plus, 本地模型} | 主结果 |
| Agent：CodeAgent × {qwen3.7-plus, 本地模型} | 主结果 |
| 中文指令 | 稳定性 |

每行报 F1^c、IoU^c、P、R、整目标漏检率、整目标误检率、逐图平均 F1；Agent 行另报平均步数、token、耗时、费用、失败率、路线分布、修改工具 ΔF1。

## 10. 里程碑

| 编号 | 内容 | 完成标准 | Codex |
|---|---|---|---|
| B0 | 工具层 + 状态仓库 + schema；固定脚本拼出 M-C-I、I-M-C(SAM3) | M-C-I 测试集差 ≤ 0.2；工具单元测试通过；GT 隔离测试通过 | **工具接口定稿前讨论一次** |
| B1 | JSON 循环 + 轨迹记录；5 张验证图跑通；量 token、估费用 | 5 张图都有完整轨迹；费用报给 Ethan | — |
| B2 | CodeAgent 接同一套工具；同 5 张图跑通 | 同上 | — |
| B3 | 验证集 64 对：四个配置；只改系统提示和工具描述，每次改动写原因 | 每个配置有验证集 summary | **第一次验证集跑通后讨论一次** |
| B4 | 测试集 128 对，每个配置评一次 | 第 9 节的表 | **测试集评测前讨论一次（确认配置冻结）** |
| B5 | 挑展示轨迹、回放页；更新两份文档 | 第 8 节 | — |

每个 Codex 讨论写进第 11 节：讨论了什么、Codex 的意见、采纳了哪些、为什么。

## 11. 状态与变更记录（每个里程碑后追加，新的在上）

| 日期 | 里程碑 | 做了什么 / 改了什么 | 结果 | 下一步 |
|---|---|---|---|---|
| 2026-11-03 | — | 文档建立：任务、工具、轨迹格式、里程碑与 Ethan 确认 | — | B0 |

## 12. 可复用代码索引

| 已有代码 | 用途 |
|---|---|
| `code/ovcd_v0/common.py` | 路径常量、`load_rgb`、`load_gt`、`components`、`evaluate`（与 DynamicEarth 评测逐像素一致）、`object_counts`、`image_f1` |
| `code/ovcd_v0/queues.py::read_dump` | 读 M-C-I 导出：全部 SAM 提议（packed masks）、DINOv2 变化分数、两期 building 概率、是否进 M0 |
| `code/ovcd_v0/sam_tools.py` | SAM ViT-H 嵌入缓存（GPU）与框/点提示解码（CPU） |
| `code/ovcd_v0/sam3_t7.py` | SAM3 文本分割，整图或 512² 分块，结果缓存 `ovcd_v0_run/sam3[_t512]/` |
| `code/ovcd_v0/render.py` | 近景裁剪、红色轮廓渲染 |
| `code/ovcd_v0/mllm.py`、`prompts.py` | Qwen3.5-4B 批量推理、yes/no 读数、口径卡 |
| `code/ovcd_v0/page/` | v0 审核页的生成脚本和 CSS，回放页可参考 |
| `results/ovcd_v0_baseline.md` | 11 轮的经验：SAM3 用 512² 分块召回更高；按变化分数升序核实最有效；Qwen3.5-4B 核实 AUC 0.85 但面积加权只有 0.69 |
