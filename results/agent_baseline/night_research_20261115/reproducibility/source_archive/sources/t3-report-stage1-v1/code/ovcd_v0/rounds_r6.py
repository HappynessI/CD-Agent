"""Round-6 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_6/'
F4 = '../figures/ovcd_v0/round_4/'

ROUND = dict(
    id=6, title='第 6 轮', stage='第 6 轮 · M4 完成（已评测试集）',
    config=[['数据集 / 类别', 'LEVIR-CD / building'], ['初始化', 'DynamicEarth I-M-C'], ['MLLM / 分割', 'Qwen3.5-4B / SAM ViT-H；SAM3（文本 building）'],
            ['阶段 A', '同第 4 轮（回答复用）'],
            ['阶段 B', 'T7 = SAM3 两期差异；**SAM3 改为 512² 分块推理**；前 20 个，核实后直接加入'],
            ['阶段 C', '网格照常，验证集选“不修”'], ['预算', 'k = 10 / 20 / 50；A 最多 75%'], ['GPU', '约 0.35 GPU 小时（SAM3 0.15，Qwen 0.2）']],
    changed='只改一项：SAM3 从整图推理改为每期 4 块 512² 分块推理。依据：第 5 轮验证集诊断显示整图 SAM3 只覆盖 46% 的变化真值像素，小对象覆盖尤其低（Codex batch63）。',
    sections=dict(
        M0=dict(did='同第 4 轮（I-M-C）。', result='测试集 69.67；验证集 73.01。', issues='—', next='—',
                figs=[dict(src=F + 'm0_repro.png', caption='复现对比')]),
        M1=dict(did='新增分块 SAM3 来源的覆盖统计。',
                result='- 两期都没检出建筑的图：验证 9 → 5 张，测试 13 → 6 张。\n- 候选池平均每图约 425 个连通域。\n- 前 20 个候选触及 **16.6%** 的整目标漏检（第 5 轮 8.0%，M-C-I 来源 11.1%），前 50 个 27.0%。',
                issues='碎片更多，仍有 5 张有变化的验证图检不出建筑。', next='—',
                figs=[dict(src=F + 'm1_queue.png', caption='右：分块 SAM3 候选的覆盖曲线（验证集）')]),
        M2=dict(did='同第 4 轮，回答复用。', result='测试集 A 相对 M0 +0.43 / +0.50 / +0.47。', issues='—', next='—',
                figs=[dict(src=F4 + 'm2_tau_a.png', caption='第 4 轮的 τ_A 曲线（本轮相同）')]),
        M3=dict(did='分块 SAM3（每期 4 块 512²）→ (S1 XOR S2) & ~M0 的连通域 → 按 SAM3 分数取前 20 → 核实 → 加入候选掩码。',
                result='- 核实：useful 比例 **18.2%**（第 5 轮 10.6%）；P(yes) AUC **0.85**（0.74）；非 useful 答“是” **20%**（33%）。\n- **验证集 B 相对 A：+0.04 / +0.52 / +1.00**，在很宽的阈值范围内都优于“不补”；选中 τ_B 0.9 / 0.8 / 0.4。\n- **测试集 B 相对 A：+0.13 / +0.37 / +1.23**；补入像素精度 72% / 75% / 62%；k=50 召回 78.1 → 80.3。',
                issues='k=50 选中的 τ_B=0.4 偏低，会加入回答本身倾向“否”的候选（test_6）。',
                next='见讨论（Codex batch64）。',
                figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：τ_B 曲线（分块 SAM3 来源），大部分阈值高于“不补”'),
                      dict(src=F + 'm2_m3_pyes.png', caption='右：分块 SAM3 候选的 P(yes) 分布')]),
        M4=dict(did='同第 4 轮。', result='验证集仍选“不修”。', issues='—', next='—',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的门限')]),
    ),
    summary_table=table(6, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。M0 行是 I-M-C。相对第 5 轮 +0.12 / +0.37 / +0.88；相对 M0 **+0.56 / +0.87 / +1.70**。验证集最终 73.54 / 74.12 / 74.55。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=50）'), dict(src=F + 'test_budget.png', caption='测试集预算曲线')],
    examples=[dict(cfg='r6_k50', split='test', name='test_73.png',
                   why='7 个分块 SAM3 候选被核实为 new_building 并加入，补上了 I-M-C 漏掉的一排小建筑：本图 F1 0.43 → 0.65。', f1=f1s('test_73.png', 'r6_k50')),
              dict(cfg='r6_k50', split='test', name='test_6.png',
                   why='变差的例子：k=50 选中的 τ_B=0.4 偏低，一个 P(yes)=0.47、类型答 seasonal_or_color 的候选也被加入，单步 −0.10。A 在这张图上的收益（0.34 → 0.71）仍然保留。', f1=f1s('test_6.png', 'r6_k50'))],
    discussion='**Codex（batch64）**：分块让 SAM3 对变化真值像素的覆盖从 46% 升到 92%；面积合格池覆盖 84% 的 ② 漏检像素，前 20 个只保留 38%。逐图留一后 B−A 三档都不转负（第 5 轮 k=50 会转负），但 k=20/50 依赖 val_44。k=50 时逐图平均 F1 下降 0.84、整目标误检多 46 个，收益集中在少数图上。\n\n**第 7 轮计划**（双方一致）：只把 k=50 的 a_share 从 0.75 调到 0.60，让 B 访问全部前 20 个候选；缓存回放验证集 +0.14（留一 +0.10～+0.17），不需要新调用。b_max 扩到 50、两来源合并、256² 分块排在后面。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 6] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
