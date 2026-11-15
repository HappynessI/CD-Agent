"""Round-9 content (selection protocol only; same chosen configs as round 8, test not re-run)."""
import json

from common import REPO
from rounds_r1 import table

ROUND = dict(
    id=9, title='第 9 轮', stage='第 9 轮 · 选参协议（选择与第 8 轮相同，未重评测试集）',
    config=[['改动', 'A、B 的阈值改在固定 C（第 8 轮裁剪）之后的状态上选'], ['其余', '同第 8 轮'], ['GPU', '0']],
    changed='只改选参协议。理由：有了裁剪，B 落在建筑之外的部分会被切掉，原来按“C 前”选出的阈值可能不再最优。',
    sections=dict(
        M0=dict(did='同第 8 轮。', result='—', issues='—', next='—', figs=[]),
        M1=dict(did='—', result='—', issues='—', next='—', figs=[]),
        M2=dict(did='A 的排序与 τ_A 按裁剪后的验证集 F1 选。', result='选出 score_asc/0.6、score_asc/0.4、area_asc/0.4，与第 8 轮相同。裁剪之后 A 相对“不删除”只多 +0.4（81.04 → 81.4）。', issues='—', next='—', figs=[]),
        M3=dict(did='τ_B 按裁剪后的验证集 F1 选。', result='选出 0.9 / 0.8 / 0.4，与第 8 轮相同；B 在裁剪后再多 +0.5～+0.9。', issues='—', next='—', figs=[]),
        M4=dict(did='C 同第 8 轮。', result='验证集 81.44 / 82.00 / 82.29，与第 8 轮相同；测试集必然相同，未重评。', issues='“I-M-C + SAM3 裁剪”本身（不调用 MLLM）在验证集上已有 81.04，MLLM 的核实再加约 1.2。', next='换更强的初始化：ReA-OVCD（本地复现 84.52）。', figs=[]),
    ),
    summary_table=table(8, (69.67, 53.46)),
    summary_note='配置与第 8 轮完全相同，下表即第 8 轮的测试集结果。',
    summary_figs=[],
    examples=[dict(cfg='r8_k50', split='test', name='test_45.png', why='与第 8 轮相同的配置，回放沿用第 8 轮。')],
    discussion='本轮是一次协议检查：在 C 之后选 A、B 不改变选择，说明第 8 轮的配置对选参顺序不敏感。',
)

if __name__ == '__main__':
    p = REPO / 'results' / 'ovcd_v0_rounds.json'
    data = json.loads(p.read_text())
    data['rounds'] = sorted([r for r in data['rounds'] if r['id'] != 9] + [ROUND], key=lambda r: r['id'])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(p)
