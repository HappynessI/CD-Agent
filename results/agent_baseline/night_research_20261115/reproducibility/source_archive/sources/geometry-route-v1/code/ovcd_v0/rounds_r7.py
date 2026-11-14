"""Round-7 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_7/'
F6 = '../figures/ovcd_v0/round_6/'
F4 = '../figures/ovcd_v0/round_4/'

ROUND = dict(
    id=7, title='第 7 轮', stage='第 7 轮 · M4 完成（已评测试集）',
    config=[['初始化', 'DynamicEarth I-M-C'], ['MLLM / 分割', 'Qwen3.5-4B；SAM3（文本 building，512² 分块）'],
            ['阶段 A', '同第 4 轮'], ['阶段 B', '同第 6 轮（分块 SAM3 两期差异候选）'],
            ['阶段 C', '**用分块 SAM3 的两期建筑掩码并集 S1∪S2 裁剪当前掩码**（半径与覆盖门限在验证集上选）'],
            ['预算', 'k = 10 / 20 / 50'], ['GPU', '0（复用第 6 轮缓存，C 在 CPU 上）']],
    changed='只改一项：阶段 C 的精修工具从“SAM 外接框重分割”换成“SAM3 建筑支撑裁剪”。依据：I-M-C 的误检 82% 是命中目标外多出的像素；分块 SAM3 两期并集覆盖 92% 的变化真值像素；在第 6 轮验证集 B 状态上离线核算，直接取交集从 74.12 升到 81.56。',
    sections=dict(
        M0=dict(did='同第 4 轮（I-M-C）。', result='测试集 69.67；验证集 73.01。', issues='—', next='—', figs=[]),
        M1=dict(did='沿用第 4、6 轮的 M1。', result='I-M-C 误检像素 82% 在命中目标之外（验证集）。', issues='—', next='—',
                figs=[dict(src=F4 + 'm1_composition.png', caption='I-M-C M0 的误检与漏检构成（第 4 轮图）')]),
        M2=dict(did='同第 4 轮。', result='测试集 A 相对 M0 +0.43 / +0.50 / +0.47。', issues='—', next='—', figs=[]),
        M3=dict(did='同第 6 轮。', result='测试集 B 相对 A +0.13 / +0.37 / +1.23。', issues='—', next='—', figs=[]),
        M4=dict(did='对 B 之后的每个连通域：若至少 c 的像素落在 S = dilate(S1∪S2, r) 内，就只保留落在 S 内的部分；否则原样保留。网格：不修、SAM 外接框重分割（6 个门限）、SAM3 裁剪（r ∈ {0,1,2,4} × c ∈ {0,0.3,0.5}）。',
                result='- 验证集（k=20）：不修 74.12；SAM 框重分割 70.6–74.0；**SAM3 裁剪 r=0、c=0：81.56**；半径越大越差（r=4：77.96）。三档都选 r=0、c=0，即 M ∩ (S1∪S2)。\n- 验证集最终 **81.03 / 81.56 / 81.85**。\n- **测试集 C 相对 B：+4.27 / +4.29 / +4.30**；精度 64 → 82，召回 80 → 70；整目标误检率降到 12.6–14.0%。',
                issues='测试集收益比验证集小（+4.3 对 +7.3），因为召回掉得多：SAM3 没检出的建筑，其变化像素被整块裁掉（test_45：密集小住宅区，SAM3 几乎没检出，单图 0.64 → 0.01）。',
                next='提高 SAM3 支撑的召回（低门限、256² 分块），或在 SAM3 明显漏检时不裁；按验证集诊断选择（Codex batch65）。',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的三类做法，SAM3 裁剪明显最好')]),
    ),
    summary_table=table(7, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。M0 行是 I-M-C。相对第 6 轮 +4.27 / +4.29 / +4.30；相对 M0 **+4.83 / +5.16 / +6.00**。验证集最终 81.03 / 81.56 / 81.85。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=50），C 带来主要提升'), dict(src=F + 'test_budget.png', caption='测试集预算曲线')],
    examples=[dict(cfg='r7_k50', split='test', name='test_17.png', why='C 把 I-M-C 外溢到建筑周围的像素裁掉，本图 +0.21。', f1=f1s('test_17.png', 'r7_k50')),
              dict(cfg='r7_k50', split='test', name='test_45.png', why='变差的例子：T2 是密集小住宅区，SAM3 几乎没检出建筑，裁剪把整张图的预测都删掉了，本图 0.64 → 0.01。说明需要提高 SAM3 支撑的召回或在漏检时不裁。', f1=f1s('test_45.png', 'r7_k50'))],
    discussion='见结果文档第 7 轮之后的讨论（Codex batch65）。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 7] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
