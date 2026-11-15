"""Round-3 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_3/'
F1 = '../figures/ovcd_v0/round_1/'

ROUND = dict(
    id=3, title='第 3 轮', stage='第 3 轮 · M4 完成（已评测试集）',
    config=[['数据集 / 类别', 'LEVIR-CD / building'], ['初始化', 'DynamicEarth M-C-I'], ['MLLM / 分割', 'Qwen3.5-4B / SAM ViT-H'],
            ['阶段 A', '**按 M-C-I 变化分数从低到高核实（score_asc）**，P(yes) < τ_A 删除'],
            ['阶段 B', '核实通过后直接加入候选掩码（同第 2 轮）'], ['阶段 C', 'SAM 外接框重分割 + IoU 门限'],
            ['预算', 'k = 10 / 20 / 50；A 最多 75%'], ['GPU', '0（复用缓存）']],
    changed='只改一项：阶段 A 的排序从面积降序改为变化分数升序。第 1 轮的面积降序是按理想核实曲线选的，但 MLLM 在大连通域上判别弱（面积加权 AUC 0.69）。Codex 用验证集缓存核算（batch59），并在第 2 轮流程下复核了对 B 的影响（batch60）：完整网格最终 F1 比第 2 轮 +0.93 / +0.94 / +0.66。',
    sections=dict(
        M0=dict(did='同第 1 轮（M-C-I）。', result='测试集 53.76 / 36.76；I-M-C 69.67 / 53.46；I-M-C 在验证集上 73.01（M-C-I 57.82）。',
                issues='三轮 Agent 在 M-C-I 上最多 +2.2（测试），与两种初始化之间 16 个点的差距相比很小。', next='第 4 轮换 I-M-C 初始化。',
                figs=[dict(src=F + 'm0_repro.png', caption='复现对比')]),
        M1=dict(did='同第 1 轮。', result='见第 1 轮 M1。', issues='—', next='—',
                figs=[dict(src=F1 + 'm1_queue.png', caption='第 1 轮的排序理想曲线：面积降序在理想核实下最好，但实际 MLLM 配合下变化分数升序更好')]),
        M2=dict(did='M0 的每个连通域按 M-C-I 变化分数从低到高排队；T2 → T3 → P(yes) < τ_A 则删除。τ_A 在验证集上按 k 分别选。',
                result='- 验证集 τ_A 曲线整体抬高，最好点移到 0.6–0.8；选中 0.8 / 0.6 / 0.6。验证集 A 相对 M0 **+1.06 / +1.06 / +0.89**（第 1、2 轮约 +0.1–0.2）。\n- **测试集 A 相对 M0：+0.40 / +0.59 / +0.60**，三档第一次都为正。k=20 时 A 删掉 31.7 万误检像素、5.1 万真阳性像素。\n- 最终状态的整目标误检率从第 2 轮的 35–39% 降到 29–34%。',
                issues='阈值高了以后，A 会删掉一些 P(yes) 略高于 0.5 的真建筑。`test_53`：P(yes)=0.56、回答 new_building 的连通域被删，单步 F1 −0.127；同图另一个被判 seasonal_or_color 的真变化也被删。',
                next='A 暂不再改；第 4 轮换初始化后重新在验证集上选排序。',
                figs=[dict(src=F + 'm2_tau_a.png', caption='验证集：score_asc 下的 τ_A 曲线')]),
        M3=dict(did='同第 2 轮。', result='选中 τ_B 0.95 / 0.9 / 0.9；测试集 B 相对 A +0.30 / +0.73 / +1.49。A 改排序后，B 访问和接受的候选集合与第 2 轮相同（Codex 核对）。',
                issues='B 中的高置信误判（第 2 轮已记录）仍在。', next='—',
                figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：τ_B 曲线')]),
        M4=dict(did='同第 2 轮。', result='门限 0.8 / 0.8 / 0.7；测试集 C 相对 B +0.18 / +0.18 / +0.12。', issues='同前。', next='—',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的门限')]),
    ),
    summary_table=table(3, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。相对第 2 轮：+0.42 / +0.59 / +0.70；相对 M0：+0.88 / +1.50 / +2.21。验证集最终 59.47 / 60.04 / 60.88。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=20）'),
                  dict(src=F + 'test_budget.png', caption='测试集预算曲线')],
    examples=[dict(cfg='r3_k20', split='test', name='test_107.png',
                   why='A 先查变化分数最低的连通域，删掉 5 个误检（P(yes) 0.04–0.38），B 再补入一处新建筑：本图 F1 0.10 → 0.56。', f1=f1s('test_107.png', 'r3_k20')),
              dict(cfg='r3_k20', split='test', name='test_53.png',
                   why='变差的例子：τ_A=0.6 时，一个 P(yes)=0.56、回答 new_building 的真建筑也被删除；另一个被判 seasonal_or_color 的真变化也被删。本图 F1 0.638 → 0.485。', f1=f1s('test_53.png', 'r3_k20'))],
    discussion='**第 4 轮改什么**：换初始化为 I-M-C。验证集证据：I-M-C 73.01，比 M-C-I 高 15.2、比第 3 轮 Agent 最好结果高 12.1。\n\n**Codex（batch61）同意**，并把它排到自己上一轮建议的“拒绝大近景 B 候选”之前。它的补充：I-M-C 剩余误检 82% 附着在已命中目标上，A 的理想删除空间降到 +3.2；B 前 20 个候选只触及 11% 的整目标漏检，其中 91% 来自 M-C-I 已接受的提议，这些提议必须保留但必须经过核实（直接全并入的新增像素精度只有 11%）；建议加固定 score_asc 的对照，用于区分初始化与排序搜索的贡献。\n\n**上一轮的分歧如何解决**：batch60 时 Codex 把换初始化排在最后（范围大、缓存作废），Claude 主张先换。拿到 I-M-C 的验证集数字后双方一致。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 3] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
