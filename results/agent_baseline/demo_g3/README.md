# 多任务回放（G3 / G4）

打开 `index.html`。页面是单文件，图片内嵌，离线可以打开。格式和结构图沿用 `../demo_g1/`。

| 文件 | 内容 |
|---|---|
| `index.html` | 入口：子页面卡片、系统流程（G3、G4）、9 个任务的表、术语 |
| `baselines.html` | 3 种做法：Agent、固定路线、固定规则的结构（按任务切换），LEVIR val_21 的分步示例 |
| `levir_c0.html` … `second_t31.html` | 9 个回放。每个任务 1 个会话，最终 F1 接近这个任务全部会话的中位数 |
| `selection.json` | 每页的会话、harness，以及这张图上 3 种做法的 F1 |

表中的 F1 是这一张图上的分数，不代表整个任务。全部图上的分数见 `baselines.html` 中每条路线的结构图下方。

生成：`python -m agent_baseline.make_site --run-root <运行目录> --out-dir results/agent_baseline/demo_g3 --selections <pick_demo --typical 的输出>`。程序只读取已有轨迹、对照掩码和验证器答案，不运行模型。
