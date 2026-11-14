"""Round-11 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_11/'
F10 = '../figures/ovcd_v0/round_10/'

ROUND = dict(
    id=11, title='第 11 轮', stage='第 11 轮 · M4 完成（已评测试集）',
    config=[['初始化', 'ReA-OVCD'], ['阶段 A / B', '同第 10 轮'], ['阶段 C', 'SAM3 裁剪 + **之后对 M0 像素按 ReA 变化分数过滤（阈值在验证集上选）**'],
            ['GPU', '0；Qwen 新调用 0 次']],
    changed='只改一项：C 之后加一条“按初始化方法的变化分数过滤”（B 补入的像素保留）。依据：Codex batch67 用缓存核算验证集 +1.0，逐图留一全部为正。',
    sections=dict(
        M0=dict(did='同第 10 轮。', result='ReA-OVCD：测试集 84.52。', issues='—', next='—', figs=[]),
        M1=dict(did='Codex batch67 的剩余错误分析。', result='剩余误检 80% 附着在已命中目标上，58% 离真值 ≤3 像素；大连通域贡献 76% 的误检。', issues='—', next='—', figs=[]),
        M2=dict(did='同第 10 轮。', result='A 在 ReA 上只有 +0.1～+0.2。', issues='—', next='—', figs=[]),
        M3=dict(did='同第 10 轮。', result='B 约为 0。', issues='—', next='—', figs=[]),
        M4=dict(did='SAM3 裁剪（门限 0.3、覆盖 0.5）之后，对仍保留的 M0 像素要求 ReA 变化分数 > t。',
                result='- 验证集三档都选 t = 0.75：**87.47 / 87.38 / 87.47**。\n- **测试集 85.90 / 85.78 / 85.89**（第 10 轮 85.37 / 85.25 / 85.31）。\n- **对照（不调用 MLLM）**：只把 ReA 阈值在验证集上调到 0.75，测试集 **85.94**；ReA + SAM3 裁剪 + 分数过滤 85.87。',
                issues='在 ReA 上，完整 Agent 与“重调 ReA 阈值”基本打平：增益来自对初始化的后处理，MLLM 的 A、B 只带来 +0.1～+0.2。理想上限：MLLM 挽回被删连通域最多 +0.40，整块删除误检最多 +0.75。',
                next='在强初始化上，以整块核实为单位的 MLLM 动作天花板约 1 个点；要让 MLLM 起更大作用，需要更强的核实器（接口模型，需先确认费用）或更难的场景（多类别、弱初始化）。',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的网格（含分数过滤）')]),
    ),
    summary_table=table(11, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。M0 行是 ReA-OVCD。对照：只在验证集上把 ReA 阈值调到 0.75（不调用 MLLM）测试集 85.94；ReA + SAM3 裁剪 + 分数过滤 85.87。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=10）')],
    examples=[dict(cfg='r11_k10', split='test', name='test_97.png', why='C 的裁剪与分数过滤共同去掉建筑外的误检。', f1=f1s('test_97.png', 'r11_k10')),
              dict(cfg='r11_k10', split='test', name='test_93.png', why='变差的例子：被裁掉和被分数过滤掉的部分里有真变化。', f1=f1s('test_93.png', 'r11_k10'))],
    discussion='**Codex（batch67）**推荐本轮的分数过滤。**Claude 补充的对照**：只调 ReA 阈值（不跑 Agent）就能得到 85.94，说明本轮与第 10 轮的增益主要来自对初始化的后处理，不是 MLLM。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 11] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
