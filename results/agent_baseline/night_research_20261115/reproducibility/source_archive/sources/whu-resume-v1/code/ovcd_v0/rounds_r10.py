"""Round-10 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_10/'

ROUND = dict(
    id=10, title='第 10 轮', stage='第 10 轮 · M4 完成（已评测试集）',
    config=[['初始化', '**ReA-OVCD**（官方 LEVIR 配置，复现 84.52 / 73.19）'], ['阶段 A', 'ReA 连通域，排序与 τ_A 在验证集上选'],
            ['阶段 B', '分块 SAM3 两期差异中不在新 M0 内的候选，前 20'], ['阶段 C', '不修 / SAM 框重分割 / SAM3 裁剪（门限 0.3–0.5），验证集选'],
            ['预算', 'k = 10 / 20 / 50'], ['GPU', '约 0.8 GPU 小时（ReA 0.2，Qwen 0.6）']],
    changed='只改一个主要因素：初始化换成 ReA-OVCD（任务文档列出的候选）。依据：本地复现测试集 84.52、验证集 85.50，高于第 8 轮 Agent 在验证集上的 82.29。',
    sections=dict(
        M0=dict(did='ReA-OVCD 官方代码：1024² 原图按 512² 滑窗、SAM3、提示 building,roof,house；脚本由 Codex 编写（batch66），环境在 omniovcd-env 上叠加 scipy。',
                result='测试集 **84.52 / 73.19**（原文 84.52 / 73.19）；验证集 85.50（P 78.6、R 93.7）。',
                issues='—', next='在 ReA 的 M0 上重建 A/B 队列并重选全部阈值。',
                figs=[dict(src=F + 'm0_repro.png', caption='三种初始化的复现')]),
        M1=dict(did='同口径错误分析。',
                result='- 整目标漏检率 13.6% / 14.2%，整目标误检率 6.6% / 6.0%（验证 / 测试）。\n- 误检像素占错误 80%，其中 80% 是命中目标外多出的像素。\n- 理想 A +2.1、理想 B +1.2（验证）。\n- 离线：SAM3 裁剪 +0.72，腐蚀 1 像素 +0.17、2 像素 −3.3。',
                issues='ReA 已经很强，剩下的空间主要在边界。', next='—',
                figs=[dict(src=F + 'm1_composition.png', caption='ReA-OVCD M0 的误检与漏检构成')]),
        M2=dict(did='同前。', result='该删的连通域只占 7%，AUC 0.82；验证集最好约 +0.1～+0.2。', issues='A 在强初始化上空间很小。', next='—',
                figs=[dict(src=F + 'm2_tau_a.png', caption='验证集：τ_A 曲线')]),
        M3=dict(did='同前（分块 SAM3 差异候选）。', result='τ_B 网格几乎平，选 0.95，收益约为 0。', issues='B 的候选来源对 ReA 的漏检帮助很小。', next='可考虑用 ReA 自身被后处理过滤掉的区域作为候选。',
                figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：τ_B 曲线')]),
        M4=dict(did='C 网格：不修、SAM 框重分割、SAM3 裁剪（门限 0.5 / 0.4 / 0.3 × 半径 × 覆盖）。',
                result='- 验证集选中 SAM3 裁剪、门限 0.3、覆盖 0.5：**86.47 / 86.44 / 86.38**。\n- **测试集 85.37 / 85.25 / 85.31**，比 ReA-OVCD +0.73～+0.85。',
                issues='SAM3 没覆盖到的真变化仍会被裁掉（test_93）。', next='—',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的网格')]),
    ),
    summary_table=table(10, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。M0 行是 ReA-OVCD。相对 ReA-OVCD **+0.85 / +0.73 / +0.79**；相对第 8 轮 +8.9 / +8.5 / +7.7。验证集最终 86.47 / 86.44 / 86.38。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=10）')],
    examples=[dict(cfg='r10_k10', split='test', name='test_97.png', why='C 裁掉 ReA 外溢到建筑之外的像素，本图 +0.16。', f1=f1s('test_97.png', 'r10_k10')),
              dict(cfg='r10_k10', split='test', name='test_93.png', why='变差的例子：SAM3 没覆盖到的真变化被裁掉，本图 −0.20。', f1=f1s('test_93.png', 'r10_k10'))],
    discussion='第 10 轮由 Claude 依据验证集复现结果决定（ReA 验证集 85.50 > 第 8 轮 Agent 82.29）；Codex 负责 ReA-OVCD 的运行脚本（batch66）。第 10 轮之后的讨论见结果文档。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 10] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
