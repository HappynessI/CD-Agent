# 第 0 步：环境确认（2026-10-16）

任务说明：`doc/change_dection_agent.md`。本目录 `outputs/CD_1016_probe/` 存放三个摸底实验的产物，`results/` 下为结果文件，运行日志放 `logs/`。

## 代码（旧原型，仅作参考）

| 项 | 位置 / 调用方式 |
| --- | --- |
| 仓库 | `/guisongxia01/pangchao/wangyihan/Chang-Agent`（唯一开发目录） |
| 入口 | `tools/run_levir_change_agent.py`：SAM3 双时相文本分割 → OmniOVCD 实例重叠匹配（`overlap_presence`，阈值 0.25）→ Verifier 规划 → 工具修正 |
| SAM3 | `change_agent/adapters/sam3_adapter.py`，源码 `third_party/OmniOVCD`，权重 `models/sam3/sam3.pt`，环境 `conda_envs/omniovcd-env`；支持文本提示和框提示（Executor 生成归一化中心/尺寸） |
| SimpleClick | `change_agent/adapters/segagent_adapter.py`，源码 `third_party/SegAgent`，权重 `models/SimpleClick/cocolvis_vit_large.pth`，环境 `conda_envs/segagent-env` |

## 数据

| 数据集 | 状态 | 说明 |
| --- | --- | --- |
| LEVIR-CD | 有 | `datasets/LEVIR-CD`。`test_original/{A,B,label}` 128 对 1024×1024；`test_256/{A,B,label_cvt}` 为 2048 个 256×256 非重叠切块（标签已二值化 >127→255） |
| WHU-CD | 有 | `datasets/lim1/WHU-CD`，软链接到 `haoxiangyu/rs_datasets/WHU-CD`（2026-10-16 新建，方式同 SYSU-CD/SECOND）。来自官方整图 train/test 分区，切成 512×512：train 1260、test 690（`test/{A,B,label}`，标签 0/255）。另有 `slabel1/slabel2` 两期建筑语义标签，可以直接区分新建和拆除。690×4=2760，与 SChanger 使用的 WHU-CD256-HANet 测试切块数一致，测试区域相同、和训练区不重叠 |
| SYSU-CD | 有（2026-10-16 按用户要求加入） | `datasets/lim1/SYSU-CD`，test 4000 张 256²（`img1/img2/label_unch0chi255`，文件名 `test_*`），通用变化，二值标签 |
| SECOND | 有（2026-10-16 按用户要求加入） | `datasets/lim1/SECOND`，test 1694 张 512²，两期 6 类地物标签（`slabel1/slabel2`，0 = 未变化）加二值变化标签 |

## 检测器

服务器上原本**没有**在 LEVIR-CD 或 WHU-CD 上训练的检测器权重或预测结果（无 BIT、ChangeFormer）。按确认的方案补充和使用：

| 检测器 | 来源 | 输出 | 复现 F1（LEVIR / WHU） |
| --- | --- | --- | --- |
| SChanger-small | HuggingFace `Zy-Zhou/schanger` 官方权重，已下载到 `models/SChanger/schanger_small_{levir,whu}.pth`；源码 `Vlm_vs_OneJev/vendor/SChanger` | sigmoid 概率图 | 92.44 / 93.51（官方 92.44 / 93.51） |
| SChanger-base | 同上，`schanger_base_{levir,whu}.pth` | sigmoid 概率图 | 92.88 / 93.20（官方 92.88 / 93.20） |
| SAM3 + OmniOVCD | 旧原型初态：SAM3 "building" 两期分割 + overlap-presence 匹配 | **仅二值**，阈值扫描与实验 3 不使用；只用于 LEVIR/WHU | 73.13 / 56.37 |
| SChanger-small / base（SYSU） | 官方 SYSU 权重 `models/SChanger/schanger_{small,base}_sysu.pth` | sigmoid 概率图 | SYSU 84.58 / 84.17（与官方一致） |
| MambaSCD-Base（SECOND） | 官方权重 `Vlm_vs_OneJev/checkpoints/MambaSCD_Base_SECOND_SeK_0.2292.pth`，二值变化头概率取自 Vlm_vs_OneJev 已有的正式推理（`probe/import_mambascd.py`） | softmax 变化概率 | SECOND 74.17（与参考 micro F1 一致） |

## 模型（MLLM）

| 模型 | 路径 | 说明 |
| --- | --- | --- |
| Qwen3-VL-2B-Instruct | `models/Qwen3-VL-2B-Instruct` | 旧原型本地 verifier，历史记录显示判断能力不足 |
| Qwen3.5-4B | `models/Qwen3.5-4B` | 原生多模态（含 vision_config） |
| OneJev-4B | `models/OneJev-4B` | Qwen3.5 架构，变化评审方向的模型 |
| 百炼 `qwen3.7-plus` / `qwen3-vl-plus` | 远程 | 已有授权：仅限 LEVIR 图像与 mask 派生证据 |

显存：集群分区 `gre`。Vlm_vs_OneJev 的 GPU 前检（`diagnostics/setup/gpu_preflight.json`）记录为 NVIDIA A800-SXM4-80GB，单卡足够运行 SAM3 和 4B 级 MLLM。

## 已确认的决定（2026-10-16）

- 检测器：SChanger-small 和 SChanger-base 的官方 LEVIR/WHU 权重（从 HuggingFace 下载）加上 SAM3+OmniOVCD 零样本基线，共 3 个。
- 数据：LEVIR-CD 和 WHU-CD 的 test 集；WHU-CD 按用户要求从 `haoxiangyu/rs_datasets` 软链接，未下载新数据。随后按用户要求加入 SYSU-CD 和 SECOND（实验 1/2 用全部 test，实验 3 抽样；SECOND 目标按变化类型拆分）。
- MLLM：本地 Qwen3.5-4B（两个数据集）；用户随后要求补充百炼，已用 qwen3.7-plus 跑 LEVIR 和 WHU（WHU 经用户 2026-10-16 授权；节点 gpu44，direct 模式）。GPT（gpt-6.1-sol，本地转发）获授权用于 LEVIR/WHU，通道间歇断开，正在逐条续跑。SYSU/SECOND 授权百炼，不发 GPT。

## 运行环境

- 脚本：`Chang-Agent/probe/`（说明见 `probe/README.md`），统一用 `probe/run.sbatch` 经 Slurm 运行；SAM3 使用 `conda_envs/omniovcd-env`，其余使用 `conda_envs/vlm-vs-onejev-env`。
- 运行中发现 gpu37 的 `/tmp` 已写满，导致 SAM3 后处理的 triton 编译失败、速度下降 6 倍；已将每个作业的 TMPDIR 和 triton 缓存改到工作区内的作业目录，结束后删除。
