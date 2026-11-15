"""Round-2 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_2/'
F1 = '../figures/ovcd_v0/round_1/'
IMC = (69.67, 53.46)

ROUND = dict(
    id=2, title='第 2 轮', stage='第 2 轮 · M4 完成（已评测试集）',
    config=[['数据集 / 类别', 'LEVIR-CD / building（同第 1 轮）'], ['初始化', 'DynamicEarth M-C-I（同第 1 轮）'],
            ['MLLM / 分割', 'Qwen3.5-4B / SAM ViT-H（同第 1 轮）'], ['阶段 A', '面积降序，P(yes) < τ_A 删除（同第 1 轮）'],
            ['阶段 B', '**核实通过后直接加入候选掩码（不再 T5 画框）**'], ['阶段 C', 'SAM 外接框重分割 + IoU 门限（同第 1 轮）'],
            ['预算', 'k = 10 / 20 / 50；A 最多 75%'], ['GPU', '0（全部复用第 1 轮的缓存）']],
    changed='只改一项：`use_t5 = False`。依据是第 1 轮验证集诊断：真漏检候选上“框 + SAM 并集”与真值的 IoU 0.13，低于候选掩码本身的 0.22，只有 28% 的情况框更好；B 的核实 AUC 0.94。Codex 同意（batch59）。',
    sections=dict(
        M0=dict(did='沿用第 1 轮的 M-C-I 输出。本轮补跑了 I-M-C（APE-DINO）：官方 LEVIR-CD-256 设置，2048 个切块，再拼回 1024² 评测。',
                result=f'M-C-I 53.76 / 36.76（同第 1 轮）。**I-M-C 测试集 {IMC[0]} / {IMC[1]}**（原文 69.7 / 53.5），切块累计与拼回整图的结果完全一致。',
                issues='I-M-C 比 M-C-I 高约 16 个点，也比本轮 Agent 的最好结果高约 14 个点。M-C-I 作为初始化“误差多、Agent 空间大”，但上限也低。',
                next='初始化暂不换，先把 A 的排序改好（第 3 轮）；I-M-C 作为初始化列为后续候选，需要先在验证集上看它的 M0。',
                figs=[dict(src=F + 'm0_repro.png', caption='DynamicEarth 两种框架的复现')]),
        M1=dict(did='与第 1 轮相同（同一份 M0）。', result='见第 1 轮 M1。', issues='—', next='—',
                figs=[dict(src=F1 + 'm1_composition.png', caption='误检与漏检像素构成（第 1 轮图）')]),
        M2=dict(did='与第 1 轮相同：面积降序核实，τ_A 在验证集上选。', result='选出的 τ_A 与 A 状态的数字都和第 1 轮相同（测试集 A：53.78 / 53.76 / 53.57）。',
                issues='面积降序与 MLLM 判别力错配的问题仍在。', next='第 3 轮改排序。',
                figs=[dict(src=F + 'm2_tau_a.png', caption='验证集：τ_A 曲线（与第 1 轮相同）')]),
        M3=dict(did='每个 T7 候选（每图前 20 个、未被当前掩码覆盖）：T2 → T3 核实（1 次调用）→ P(yes) ≥ τ_B 则直接把候选掩码加入。',
                result='- 验证集 τ_B 曲线不再整体为负，0.9–0.95 最好；选中 0.95 / 0.9 / 0.9。验证集 B 相对 A：+0.39 / +0.95 / +1.94。\n- 被接受候选的新增像素精度：验证 68% / 58% / 51%，测试 50% / 44% / 42%。\n- **测试集 B 相对 A：+0.29 / +0.73 / +1.53**，三档都为正；整目标漏检率下降 0.6–4.1 个点。',
                issues='Codex（batch60）：k=50 时验证集接受的 213 个候选中 40 个与真值完全不相交，贡献 B 新增误检的一半；其中 39 个被答成 new_building，P(yes) 中位数 0.95。高分也会错，但把 τ_B 提到 0.95 在验证集上反而略差。近景边长 ≥512 的 5 个接受候选新增像素精度只有 5%。',
                next='第 3 轮不动 B；“拒绝大近景候选”留作后续候选。',
                figs=[dict(src=F + 'm3_tau_b.png', caption='验证集：τ_B 曲线（无 T5）')]),
        M4=dict(did='同第 1 轮。', result='验证集门限 0.7–0.8 最好，选中 0.8 / 0.8 / 0.7；测试集 C 相对 B +0.15 / +0.18 / +0.17。',
                issues='同第 1 轮：只约束形状，不判断对错；整目标误检率随 C 上升。', next='暂不改。',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的替换门限')]),
    ),
    summary_table=table(2, IMC),
    summary_note='测试集，本轮只评这一次。相对第 1 轮最终 F1：+0.21 / +3.53 / +4.36。验证集最终：58.54 / 59.09 / 60.23。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=20）'),
                  dict(src=F + 'test_budget.png', caption='测试集预算曲线：本轮第一次随 k 单调上升')],
    examples=[dict(cfg='r2_k20', split='test', name='test_15.png',
                   why='B 直接加入候选掩码后补对了右侧几栋新建筑：本图 F1 0.363 → 0.665（第 1 轮同图到 0.451）。', f1=f1s('test_15.png', 'r2_k20')),
              dict(cfg='r2_k20', split='test', name='test_96.png',
                   why='仍然变差的例子：整图框不再出现，但 4 个并非建筑变化的候选被以 0.95–0.99 的 P(yes) 接受，本图 F1 0.880 → 0.073。这是 MLLM 高置信误判，阈值挡不住。', f1=f1s('test_96.png', 'r2_k20'))],
    discussion='**第 3 轮改什么**：双方一致，只把阶段 A 的排序改为按 M-C-I 变化分数升序（score_asc）。Codex（batch60）在第 2 轮流程下逐像素复现了全部 A/B 网格，确认改排序后 B 访问和接受的候选集合不变，第 3 轮完整验证集网格最终 F1 59.47 / 60.04 / 60.88（比第 2 轮 +0.93 / +0.94 / +0.66）。\n\n**分歧**：Codex 建议第 4 轮拒绝近景边长 ≥512 的 B 候选（验证集 B +0 / +0.07 / +0.25），并把换初始化排在最后，理由是范围最大、缓存全部作废。Claude 倾向第 4 轮换成 I-M-C 初始化：I-M-C 测试集 69.67，比当前 Agent 最好结果高约 14 个点，B 的 512 过滤只影响 3 张验证图。两种理由都记录，最终依据验证集上 I-M-C 的 M0 和完整网格决定（见第 3 轮）。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 2] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
