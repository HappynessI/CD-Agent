# Chang-Agent 运行依赖

2026-10-16 从原工作区 `reference` 移入以下正在使用的依赖，保留各自的仓库内容和已有修改。Chang-Agent 及 Chang-Agent-local-qwen3vl 共用这些路径。

- `OmniOVCD`：SAM3 分割实现及文本词表。
- `SegAgent`：其中的 SimpleClick 交互式掩码修正实现。
- `LLaMA-Factory`：Qwen3-VL 训练入口。

活动启动脚本、SAM3 词表默认路径和 Python 可编辑安装引用均已更新。原 `reference` 目录已清理。模型权重继续使用工作区 `models` 中的现有文件。
