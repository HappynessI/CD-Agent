"""Round-8 content of results/ovcd_v0_rounds.json."""
import json

from common import REPO
from rounds_r1 import f1s, table

F = '../figures/ovcd_v0/round_8/'

ROUND = dict(
    id=8, title='第 8 轮', stage='第 8 轮 · M4 完成（已评测试集）',
    config=[['初始化', 'DynamicEarth I-M-C'], ['阶段 A / B', '同第 6、7 轮'],
            ['阶段 C', 'SAM3 建筑支撑裁剪；**支撑用的 SAM3 实例门限在验证集上选（0.5 / 0.4 / 0.3 / 0.2）**'],
            ['预算', 'k = 10 / 20 / 50'], ['GPU', '约 0.3 GPU 小时（门限 0.1 的 SAM3 推理）']],
    changed='只改一项：C 的 SAM3 支撑实例门限。依据：Codex batch65 在验证集上离线比较 13 种支撑来源 × 3 类裁剪规则，512² 分块门限 0.4 直接裁剪最好（+0.41～+0.44，逐图留一全为正）。B 的候选仍用门限 0.5，保持单因素。',
    sections=dict(
        M0=dict(did='同第 4 轮。', result='I-M-C：测试集 69.67。', issues='—', next='—', figs=[]),
        M1=dict(did='Codex batch65 的裁剪损失分析。', result='裁掉的 TP 约 10 万像素（验证集），近一半在三张密集住宅图；69% 来自面积 <1024 像素的小目标。', issues='整图保护规则分不开“SAM3 漏检”和“输入误检多”。', next='降低 SAM3 门限。', figs=[]),
        M2=dict(did='同前。', result='—', issues='—', next='—', figs=[]),
        M3=dict(did='同前。', result='—', issues='—', next='—', figs=[]),
        M4=dict(did='SAM3 支撑门限 ∈ {0.5, 0.4, 0.3, 0.2} × 半径 {0,1,2,4} × 覆盖 {0,0.3,0.5}。',
                result='- 验证集三档都选门限 0.4、半径 0、直接裁剪：81.44 / 82.00 / 82.29。\n- **测试集 76.49 / 76.80 / 77.60**，比第 7 轮 +2.0；召回 70 → 72–74。',
                issues='测试集收益比验证集大，说明第 7 轮的召回损失在测试集上更严重；收益集中在密集住宅类图。',
                next='第 9 轮：A、B 的阈值改在 C 之后的状态上选。',
                figs=[dict(src=F + 'm4_gate.png', caption='验证集：C 的网格（含门限）')]),
    ),
    summary_table=table(8, (69.67, 53.46)),
    summary_note='测试集，本轮只评这一次。相对第 7 轮 +1.99 / +1.97 / +1.93；相对 I-M-C 的 M0 **+6.82 / +7.13 / +7.93**。验证集最终 81.44 / 82.00 / 82.29。',
    summary_figs=[dict(src=F + 'test_states.png', caption='测试集四个状态（k=50）')],
    examples=[dict(cfg='r8_k50', split='test', name='test_45.png', why='第 7 轮的反例：门限 0.4 后 SAM3 检出了更多密集小房屋，本图从第 7 轮的 0.01 回到 0.45。', f1=f1s('test_45.png', 'r8_k50')),
              dict(cfg='r8_k50', split='test', name='test_86.png', why='相对第 7 轮变差最多的图（−0.06）：低门限多出的 SAM3 实例保留了一些非建筑的误检。', f1=f1s('test_86.png', 'r8_k50'))],
    discussion='**Codex（batch65）**：首选门限 0.4，逐图留一三档都为正，但新增 TP 的 77% 来自三张密集住宅图；256² 分块、两尺度并集、连通域保护、整图保护都没超过它。\n\n**第 9 轮**（Claude 提议）：A、B 的阈值改在固定 C 之后的状态上选；理由是裁剪会切掉 B 落在建筑之外的部分，原来按“C 前”选出的阈值不再最优。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 8] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
