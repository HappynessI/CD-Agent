# 多任务回放（G3 / G4）

打开 `index.html`。页面是单文件，图片内嵌，离线可以打开。格式和结构图沿用 `../demo_g1/`。

| 文件 | 内容 |
|---|---|
| `index.html` | 入口：子页面卡片、系统流程（G3、G4）、9 个任务的表、术语 |
| `baselines.html` | 3 种做法：Agent、固定路线、固定规则的结构（按任务切换），LEVIR val_21 的分步示例 |
| `levir_c0.html` … `second_t31.html` | 9 个回放。每个任务 1 个会话，最终 F1 接近这个任务全部会话的中位数 |
| `selection.json` | 每页的会话、harness，以及这张图上 3 种做法的 F1 |

表中的 F1 是这一张图上的分数，不代表整个任务。全部图上的分数见 `baselines.html` 中每条路线的结构图下方。

生成分两步（`code/agent_baseline/make_site.py`）：

1. 数据步骤（计算节点，只在数据变化时运行）：`python -m agent_baseline.make_site --stage data --run-root <运行目录> --cache <缓存目录> --out-dir <任意> --selections <pick_demo --typical 的输出>`。读取轨迹、对照掩码、验证器答案和原图，把图片和分数写入缓存。
2. 页面步骤（登录节点）：`python -m agent_baseline.make_site --stage page --cache <缓存目录> --out-dir results/agent_baseline/demo_g3`。只读缓存。修改文字、样式、结构图后只运行这一步。

程序不运行模型。
