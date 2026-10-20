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
- MLLM：本地 Qwen3.5-4B（两个数据集）；用户随后要求补充百炼，已用 qwen3.7-plus 跑 LEVIR 和 WHU（WHU 经用户 2026-10-16 授权；节点 gpu44，direct 模式）。GPT（gpt-6.1-sol，本地转发）获授权用于 LEVIR/WHU，图像请求最终被通道持续断开，补测未完成（见 exp3_mllm_verify.md）。SYSU/SECOND 授权百炼，不发 GPT。

## 运行环境

- 脚本：`Chang-Agent/probe/`（说明见 `probe/README.md`），统一用 `probe/run.sbatch` 经 Slurm 运行；SAM3 使用 `conda_envs/omniovcd-env`，其余使用 `conda_envs/vlm-vs-onejev-env`。
- 运行中发现 gpu37 的 `/tmp` 已写满，导致 SAM3 后处理的 triton 编译失败、速度下降 6 倍；已将每个作业的 TMPDIR 和 triton 缓存改到工作区内的作业目录，结束后删除。

## 实验 4 资源确认（2026-10-16）

任务：仓库 `doc/exp4_recall_candidates.md`（召回侧候选生成与确认）。

| 资源 | 状态 | 用法 |
| --- | --- | --- |
| MambaSCD-Base 两期语义头输出 | **已有**：`Vlm_vs_OneJev/outputs/formal/SECOND/probabilities/test_*_T1.npy`、`_T2.npy`，7×512×512 softmax（类 0 = 未变化，1–6 为 SECOND 六类），覆盖全部 1694 张 test，无需补跑 | 来源 A：在类 1–6 内取 argmax |
| SYSU 的语义头 | **无**：SChanger 只有二值头 | SYSU 不做来源 A |
| 独立的地物分类 / 语义分割模型 | **无** | 来源 B 改用 SAM3 文本提示 |
| 图像编码器 | SAM3 图像编码器可用（`vision_features`，256 维，72×72 patch） | 来源 C（改建）、D、E |
| DINOv3（`facebook/dinov3-vitl16-pretrain-sat493m`） | HF 缓存里**只有配置文件**，权重为 `.incomplete` 未完成下载，**不可用**；按要求不补下载（后已更正：服务器上另有完整权重，见下方“实验 6 资源”的验证器骨干一行） | — |
| DINOv2、CLIP 等其他编码器 | 无 | — |
| AnyChange 代码或权重 | **无**（segagent-env 里只有 SAM1 的 `segment_anything` 包，没有 SAM ViT-H 权重） | 来源 D 自实现：SAM3 编码器两期特征差 |

## 实验 5 资源（2026-10-17）

任务：仓库 `doc/exp5_followup.md`。

| 资源 | 安装方式与版本 | 路径 | 跑通确认 |
| --- | --- | --- | --- |
| 独立环境 | conda 新建（清华镜像，python 3.11），`probe/exp5_build_env.sbatch` 在计算节点构建。torch 2.5.1+cu124，torchvision 0.20.1。没有改动 omniovcd-env 和 vlm-vs-onejev-env | `conda_envs/exp5-env` | 构建作业中 `import torch, open_clip, torchange` 及 `from torchange.models.segment_any_change import AnyChange` 成功，CUDA 可用 |
| AnyChange | PyPI `torchange==0.0.4`。wheel 预先下载到 `wheels/exp5/`，里面包含 `torchange/models/segment_any_change` 的完整代码 | 代码在 exp5-env 中 | 在 SECOND test_0/1/10 上跑通：H1 每图 68–172 个变化掩码。H2（点查询）在 3 张中有 2 张因查询点没得到掩码而报错，处理方式见 exp5a |
| SAM ViT-H 权重（AnyChange 官方示例用 vit_h） | 官方地址 dl.fbaipublicfiles.com 因管理节点代理中途失效无法下载，改为直连 hf-mirror 上的公开副本 `ybelkada/segment-anything`；大小 2,564,550,879 字节，与官方一致 | `models/SAM/sam_vit_h_4b8939.pth` | 同上 |
| RemoteCLIP | `open_clip_torch` 3.3.0；权重 `chendelong/RemoteCLIP` 的 `RemoteCLIP-ViT-L-14.pt`（经 hf-mirror 直连下载，1,710,613,765 字节，与源站一致） | `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt` | 在 4b 全部 600 个候选（含 SECOND 300 个）上跑通，`probe/exp5b_clip.py` |

- **网络情况**：管理节点的 HTTP 代理在下载途中失效（连接立即被拒）。PyPI 包和权重最终都改为直连清华镜像和 hf-mirror。计算节点同样直连可用。

## 实验 6 资源（2026-10-18，第 0 步最终状态）

| 项目 | 状态 | 说明 |
|---|---|---|
| SECOND train | 可用 | 2,968 对（test 1,694 对），每对有两期 RGB 512×512、两期语义标签（0 未变化，1–6 为低矮植被、非植被地表、树、水、建筑、运动场）、二值变化标签 |
| MambaSCD 官方训练代码 | 可用 | ChangeMamba 的 `train_MambaSCD.py` / `SCDTrainer`；官方 SCD 调度：batch 16、512 裁块、50,000 次迭代、AdamW lr 1e-4、wd 5e-3、每 10,000 次 lr×0.5、每 500 次验证。本地目录名与官方不同，用 `exp6_prepare_folds.py` 建软链接目录适配 |
| VMamba-Base 初始化权重 | 已下载 | 官方 Zenodo 记录 15479555，`models/VMamba/vssm_base_0229_ckpt_epoch_237.pth`（354,373,070 字节）。已有的 MambaSCD SECOND 权重见过完整 train，不能用于交叉拟合 |
| 第二个 SECOND 检测器 | 已下载，未加载 | SCanNet（ResNet34 + CSWin Transformer，结构与 MambaSCD 不同）。代码 `Chang-Agent/third_party/SCanNet/`；官方 SECOND 权重 `models/SCanNet/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth`（111,811,247 字节）。模型构造默认下载 ImageNet ResNet34，加载完整权重时需关掉 |
| 第三个双时相语义数据集 | 服务器已有 | `/guisongxia01/pangchao/haoxiangyu/rs_datasets/` 下：lsSCD（train/test 1,120/480，512²，8 类）、SCSCD7（1,400/322，512²，7 类）、CCSCD5（700/253）、LevirSCD（3,225/645，256²）。HRSCD 划分文件与实际文件数不一致；Hi-CNA 没有 test 语义标签 |
| 验证器骨干 | 可用 | SAM ViT-H（`models/SAM/`）、RemoteCLIP ViT-L-14（`models/RemoteCLIP/`）可用。卫星版 DINOv3：hf-mirror 下载返回 403（gated）。**更正（2026-10-20）**：服务器上同组成员的工作区已有完整的 `dinov3-vitl16-pretrain-sat493m`（含 `model.safetensors`），经用户指定后以软链接接入 `models/dinov3/dinov3-vitl16-pretrain-sat493m`；6b 的 DINOv3 结果用的就是这份卫星版 ViT-L/16 权重，用 transformers 的 `DINOv3ViTModel` 加载 |
| RingMo-SAM | 有公开 demo | HF Space `AI-Cyber/RingMo-SAM` 有代码和约 2.59 GB 光学权重，但未找到论文或作者主页对该地址的确认，Space 当前运行报错；未下载 |
| Zero-OVCD | 停止 | 官方仓库只有 README，源码待论文接收后公开；按任务要求停止该子项 |

**6a 耗时实测与资源决定（2026-10-18）**

- 单卡 A800 80GB 上 batch 16 显存不足；用 2×8 梯度累积时 4.16 秒/步，峰值显存 71.9 GiB；内部验证 0.094 秒/张（`exp6/second_cv/timing_fold0.json`）。
- 按此单卡速度，K=3 完整调度约 7.5 GPU 天，超过任务的 3 天门槛。已报告用户，用户同意按官方完整调度跑，改为每折 2 卡 DDP（每卡 batch 8，全局 16），三折并行共 6 卡。
- 实际：三折各 2 卡，每折约 29.4 小时（2026-10-18 07:27 至 10-19 12:52 UTC），共约 176 GPU 小时（约 7.3 GPU 天）。
- 运行注意：PyTorch DataLoader 在 `TMPDIR` 下建 AF_UNIX socket，工作区长路径会超长，训练作业用 `JOB_TMP_ROOT=/tmp`；gpu04、gpu06、gpu37 上作业异常，已排除。

**6a 候选池与 6b V1 实际耗时（2026-10-19 至 10-20，单卡 A800，按 Slurm 记录的作业时长）**

| 步骤 | 作业 | 耗时 |
|---|---|---|
| 6a second_cv 候选：AnyChange H1 | 4 分片 × 1 GPU | 各约 37 分钟，共约 2.5 GPU 小时 |
| 6a second_cv 候选：SAM3 实例/类别（B、C） | 2 分片 × 1 GPU | 各约 28 分钟，共约 0.9 GPU 小时 |
| 6a 候选合并 + SLIC 切细 + 标注 | CPU，8 核，各数据集 1–6 分片 | 每分片约 2 分钟；second_cv 共约 14 分钟，test 约 8 分钟 |
| 6b 特征提取 SAM ViT-H | second_cv 4 分片 + test 2 分片 | 共约 1.6 GPU 小时 |
| 6b 特征提取 DINOv3 | 同上 | 共约 0.9 GPU 小时 |
| 6b 特征提取 RemoteCLIP | 同上 | 共约 0.8 GPU 小时 |
| 6b MLP 训练（早停） | 1 GPU | none 6 分钟，SAM 13 分钟，DINOv3 15 分钟，RemoteCLIP 9 分钟 |
| 6b 评估（验证集定阈值 + 测试集） | CPU，8 核 | 每个版本约 18 分钟（验证 5–6 分钟，测试 12–13 分钟） |
| 6b 学习曲线重跑（30 轮，不早停） | 1 GPU | 每轮约 1.25 分钟 |

6a 候选池加 6b V1 合计约 7.4 GPU 小时，另有约 2 小时 CPU 作业。实测内存峰值：MLP 训练与评估 1–11 GB，V2 训练约 12.6 GB；之后按峰值申请（V1 24G、V2 32G）。

