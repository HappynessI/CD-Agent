"""Round-5 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_5/'
F4 = '../figures/ovcd_v0/round_4/'

ROUND = dict(
    id=5, title='第 5 轮', stage='第 5 轮 · M4 完成（已评测试集）',
    config=[['数据集 / 类别', 'LEVIR-CD / building'], ['初始化', 'DynamicEarth I-M-C'], ['MLLM / 分割', 'Qwen3.5-4B / SAM ViT-H；**SAM3（文本 building）**'],
            ['阶段 A', '同第 4 轮（回答复用；选中 score_asc/0.6、score_asc/0.4、area_asc/0.4）'],
            ['阶段 B', '**T7 = SAM3 两期分割的差异 (S1 XOR S2) 且不在 M0 内的连通域**，按 SAM3 分数取前 20，核实后直接加入'],
            ['阶段 C', '网格照常，验证集选“不修”'], ['预算', 'k = 10 / 20 / 50；A 最多 75%'], ['GPU', '约 0.6 GPU 小时（SAM3 0.4，Qwen 0.2）']],
    changed='只改一项：B 的候选来源从 M-C-I 近阈值提议换成 SAM3 的两期文本分割差异。依据：第 4 轮 M-C-I 来源在 I-M-C 上全面变差，预算全给 A 的缓存核算几乎为零（Codex batch62）。',
    sections=dict(
        M0=dict(did='同第 4 轮（I-M-C）。', result='测试集 69.67 / 53.46；验证集 73.01。', issues='—', next='—',
                figs=[dict(src=F + 'm0_repro.png', caption='复现对比')]),
        M1=dict(did='同第 4 轮；新增 SAM3 来源的覆盖统计。',
                result='- SAM3 整图推理：验证集 9 张、测试集 13 张图两期都没检出建筑，22 / 30 张图至少一期为 0。\n- 候选池平均每图约 165 个连通域，多为两期边缘不一致的碎片。\n- 前 20 个候选触及 8.0% 的整目标漏检（M-C-I 来源 11.1%），前 50 个 13.2%。',
                issues='SAM3 在 1024² 整图上对小建筑召回低，覆盖受限。', next='候选碎片多、覆盖低，是下一轮可改的地方。',
                figs=[dict(src=F + 'm1_queue.png', caption='右：SAM3 候选的覆盖曲线（验证集）')]),
        M2=dict(did='同第 4 轮，回答复用。', result='验证集选择复现了第 4 轮的 A 配置；测试集 A 相对 M0 +0.43 / +0.50 / +0.47。', issues='—', next='—',
                figs=[dict(src=F4 + 'm2_tau_a.png', caption='第 4 轮的 τ_A 曲线（本轮相同）')]),
        M3=dict(did='SAM3（文本 building，分辨率 1008，实例门限 0.5）分别分割两期 → (S1 XOR S2) & ~M0 的连通域，面积 50–20,000 → 按重叠实例的最高 SAM3 分数排序取前 20 → T3 核实 → P(yes) ≥ τ_B 则加入候选掩码。',
                result='- 验证集 893 个候选中 useful 占 10.6%（M-C-I 来源 7.7%）；P(yes) AUC 0.74；非 useful 中答“是”33%（M-C-I 来源 46%）。\n- 验证集 B 相对 A：**+0.08 / 0（不补）/ +0.06**；第 4 轮在最好的有效阈值下是 −0.45 / −0.77 / −1.39。\n- **测试集 B 相对 A：+0.01 / 0 / +0.35**；k=50 补入 9.8 万 TP、10.6 万 FP，新增像素精度 48%，高于 F1/2（约 35%）。',
                issues='新来源更“干净”但覆盖更低：很多图上 SAM3 一个建筑都没检出。高置信误判仍在（test_96）。',
                next='见讨论：提高 SAM3 对小建筑的召回（分块推理）或合并两种来源。',
                figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：τ_B 曲线（SAM3 来源），接近持平'),
                      dict(src=F + 'm2_m3_pyes.png', caption='右：SAM3 候选的 P(yes) 分布')]),
        M4=dict(did='同第 4 轮。', result='验证集仍选“不修”。', issues='—', next='—',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的门限')]),
    ),
    summary_table=table(5, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。M0 行是 I-M-C。相对第 4 轮 +0.01 / 0 / +0.35；相对 M0 +0.44 / +0.50 / +0.82。k=20 时验证集选“不补”，结果与第 4 轮相同。没有候选的图上 B 不消耗调用，平均调用低于 k。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=50）'), dict(src=F + 'test_budget.png', caption='测试集预算曲线')],
    examples=[dict(cfg='r5_k50', split='test', name='test_110.png',
                   why='一个 SAM3 差异候选正好是 I-M-C 漏掉的新建筑，P(yes)=0.98，加入后本图 F1 0.33 → 0.68。', f1=f1s('test_110.png', 'r5_k50')),
              dict(cfg='r5_k50', split='test', name='test_96.png',
                   why='变差的例子：SAM3 差异候选被以 P(yes)=0.95 判为 new_building 加入，实际不是建筑变化；本图 F1 0.94 → 0.74。', f1=f1s('test_96.png', 'r5_k50'))],
    discussion='**Codex（batch63）**：SAM3 整图推理两期并集只覆盖验证集 46% 的变化真值像素，面积 <1024 像素的变化对象只有约 27% 被覆盖一半以上；候选 77% 是 <50 像素的碎片，前 20 个里 68% 与变化真值完全不相交。两种来源互补（各取前 20 时 SAM3 额外覆盖 43 个、M-C-I 额外覆盖 67 个整目标漏检），但同预算下两种简单合并在验证集上都选“不补”。k=50 的小幅收益集中在少数验证图上（去掉 val_36 后转负）。\n\n**第 6 轮改什么**（双方一致）：SAM3 改为 512² 分块推理，检验小建筑召回能否提高；其余不变。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 5] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
