"""Round-4 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_4/'

ROUND = dict(
    id=4, title='第 4 轮', stage='第 4 轮 · M4 完成（已评测试集）',
    config=[['数据集 / 类别', 'LEVIR-CD / building'], ['初始化', '**DynamicEarth I-M-C（APE-DINO）**，val 73.01 / test 69.67'],
            ['MLLM / 分割', 'Qwen3.5-4B / SAM ViT-H'],
            ['阶段 A', 'I-M-C 连通域；排序在验证集上从 score_asc / area_desc / area_asc 中选（属性借自重叠的 M-C-I 提议）'],
            ['阶段 B', 'M-C-I 全部 SAM 提议中被新 M0 覆盖 ≤ 50% 的前 20 个（含 M-C-I 已接受的）；核实后直接加入'],
            ['阶段 C', 'SAM 外接框重分割 + IoU 门限（可选“不修”）'], ['预算', 'k = 10 / 20 / 50；A 最多 75%'],
            ['GPU', '约 1.8 GPU 小时（I-M-C 推理 1.1，Qwen 0.65）']],
    changed='只改一个主要因素：初始化换成 I-M-C。依据：验证集上 I-M-C 73.01，比 M-C-I 高 15.2，比第 3 轮 Agent 最好结果高 12.1。随之必须调整 A 的待查项、属性来源和 B 的候选池（见配置）。Codex 同意（batch61），并建议加固定 score_asc 的对照。',
    sections=dict(
        M0=dict(did='官方 I-M-C：LEVIR-CD-256 切块（验证集按测试集同样的 4×4 行优先顺序切成 1,024 块，并在测试集上核对了切块顺序），APE 阈值 0.2、DINO 阈值 135°，再拼回 1024²。',
                result='测试集 **69.67 / 53.46**（原文 69.7 / 53.5），切块累计与拼回整图完全一致；验证集 73.01 / 57.49（P 64.8、R 83.7）。',
                issues='环境需要编译 detectron2/detrex，detrex 固定的 omegaconf 2.1 与 hydra-core 冲突，改为不带依赖安装并补齐 wandb 等。每块约 1.2 秒，CPU 内存峰值约 20 GB。',
                next='在 I-M-C 的 M0 上重新做 M1，并重建 A/B 队列。',
                figs=[dict(src=F + 'm0_repro.png', caption='两种框架的复现')]),
        M1=dict(did='同第 1 轮的口径，基准换成 I-M-C 的 M0。',
                result='- 整目标漏检率 验证 27.4% / 测试 36.8%（M-C-I 43.1% / 50.5%）；整目标误检率 28.4% / 29.6%。\n- 误检像素中 **82%（验证）是命中目标外多出的像素**，整块误检只占 18%。\n- 漏检像素 ② 孤立 52%、③ 有信号 36%。\n- 理想 A +3.2，理想 B +5.4（验证）。\n- T7（M-C-I 提议）前 20 个只触及 11% 的整目标漏检。',
                issues='A 删整块能处理的误检不到两成；B 的现有来源很难碰到 I-M-C 的漏检。',
                next='A 在验证集上重选排序；B、C 照常进入网格，由验证集决定是否保留。',
                figs=[dict(src=F + 'm1_composition.png', caption='I-M-C M0 的误检与漏检构成'),
                      dict(src=F + 'm1_queue.png', caption='I-M-C 上的排序理想曲线与 T7 覆盖')]),
        M2=dict(did='I-M-C 连通域 → T2 → T3 → P(yes) < τ_A 删除；排序与 τ_A 一起在验证集上选。另跑固定 score_asc 的对照（只在验证集上报告）。',
                result='- 验证集：score_asc 73.50 / 73.60 / 73.52，area_desc 73.01 / 73.02 / 73.20，area_asc 在 k=50 时 73.55 被选中。对照与主线在 k=10/20 选出相同配置，k=50 只差 0.03。\n- 判别力：删/留 AUC 0.86，按面积加权 0.72。\n- **测试集 A 相对 M0：+0.43 / +0.50 / +0.47**；整目标误检率 29.6% → 21.3%（k=20）。',
                issues='A 的空间本来就小（理想 +3.2），实际只拿到约六分之一。小目标被判为季节色调删除的情况仍在（test_95）。',
                next='A 可用的预算还可以更多：B 关闭后有 25% 的调用浪费在 B 上。',
                figs=[dict(src=F + 'm2_tau_a.png', caption='验证集：选中排序下的 τ_A 曲线'),
                      dict(src=F + 'm2_m3_pyes.png', caption='P(yes) 分布：左 A，右 B')]),
        M3=dict(did='M-C-I 全部提议中被 I-M-C 覆盖 ≤ 50% 的，按软接受分取前 20 个；核实后直接加入。',
                result='- 验证集上**每个 τ_B 都让 F1 下降**（k=20：τ_B 0.95 为 72.83，不补为 73.60），选中 τ_B = 1.01（不补）。\n- useful 标签只占 7.7%（M-C-I 上 20%）；P(yes) AUC 0.80（M-C-I 上 0.94）；非 useful 候选 46% 被答“是”。τ_B=0.95 时新增像素精度只有 3–15%，提高 F1 需要约 37%。',
                issues='来源本身不缺：全部 M-C-I 原始提议覆盖了 I-M-C ② 孤立漏检像素的 94.8%，过滤后 82.8%，按软接受分截断到前 20 个后只剩 22%（Codex 分层核算）。损失在候选的选择排序和核实后的新增像素质量。τ_B=1.01 时 B 仍逐个核实，占实际调用的 35–44%。',
                next='给 B 换候选来源，或把预算全给 A（见讨论）。',
                figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：τ_B 曲线，所有阈值都低于“不补”')]),
        M4=dict(did='同前，网格含“不修”。', result='验证集上每个门限都变差（总是替换 −3.4，门限 0.9 也 −0.1），选中“不修”。',
                issues='I-M-C 的掩码来自 APE 实例分割加两期筛选，没有经过 SAM。门限 0.7–0.9 时 C 确实补回了一些真阳性（k=20 门限 0.8：+1.2 万 TP），但同时净增 7.6 万误检，净变化的真阳性比例 22% 低于 F1/2。', next='C 在 I-M-C 上暂时关闭。',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的门限，全部低于“不修”')]),
    ),
    summary_table=table(4, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。M0 行是 I-M-C。相对 M0 +0.43 / +0.50 / +0.47；相对第 3 轮最终结果 +15.5 / +14.9 / +14.2。B、C 在验证集上被关掉，所以三个状态相同。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=20）')],
    examples=[dict(cfg='r4_k20', split='test', name='test_6.png',
                   why='A 删掉 9 个被判为季节色调或无变化的误检（P(yes) 0.04–0.32），本图 F1 +0.37。', f1=f1s('test_6.png', 'r4_k20')),
              dict(cfg='r4_k20', split='test', name='test_95.png',
                   why='变差的例子：整张图唯一的真值是右下角 12×19 像素的小目标，I-M-C 找到了它，Qwen 判为 seasonal_or_color（P(yes)=0.20）删除；单图 F1 0.83 → 0，对累计 F1 几乎没有影响。', f1=f1s('test_95.png', 'r4_k20'))],
    discussion='**Codex（batch62）**：预算全给 A 用缓存核算只有 +0.10 / −0.005 / −0.009；“只保留核实为是的”若连未访问的一起删，F1 掉到 11–48。纠正了 Claude 初稿的三处表述（I-M-C 未经 SAM 细化；I-M-C 漏检并非不在 M-C-I 提议里，而是在排序截断中丢失；7.7% 是 useful 标签比例），已改正。\n\n**第 5 轮改什么**（双方一致）：T7 换成 SAM3 的两期文本分割差异，(S1 XOR S2) 且不在 M0 内的连通域，按 SAM3 实例分数取前 20；A、C 固定为本轮选中的配置，只在验证集上重选 τ_B（含不补）。若验证集仍选“不补”，就如实记录新来源无效。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 4] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
