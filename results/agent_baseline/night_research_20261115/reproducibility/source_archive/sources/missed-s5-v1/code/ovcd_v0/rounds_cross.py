"""cross_round section of results/ovcd_v0_rounds.json (test-set comparison of finished rounds)."""
import json
import sys

from common import REPO, RUN

rounds = [int(r) for r in sys.argv[1].split(',')]
cols = ['轮次', '改动', 'k=10', 'k=20', 'k=50', '验证集 k=20']
change = {1: '首轮（T5 画框）', 2: '去掉 T5，B 直接用候选掩码', 3: 'A 改为按变化分数升序', 4: '初始化换成 I-M-C', 5: 'T7 换成 SAM3 两期差异', 6: 'SAM3 改 512² 分块推理', 7: 'C 换成 SAM3 建筑掩码裁剪', 8: '裁剪的 SAM3 门限 0.4', 9: 'A、B 阈值在 C 之后选（同第 8 轮）', 10: '初始化换成 ReA-OVCD', 11: 'C 后按 ReA 分数过滤', 12: '—'}
rows = []
for r in rounds:
    t = [json.loads((RUN / 'eval' / f'r{r}_k{k}_test.json').read_text())['states']['C']['f1'] for k in (10, 20, 50)]
    v = json.loads((RUN / 'eval' / f'r{r}_k20_val.json').read_text())['states']['C']['f1']
    rows.append([f'第 {r} 轮', change.get(r, ''), *[f'{x*100:.2f}' for x in t], f'{v*100:.2f}'])
rows.append(['参照', 'M0（M-C-I）', '53.76', '53.76', '53.76', '57.82'])
rows.append(['参照', 'I-M-C（APE-DINO）复现', '69.67', '69.67', '69.67', '—'])
rows.append(['参照', 'ReA-OVCD 复现', '84.52', '84.52', '84.52', '85.50'])
rows.append(['对照', 'ReA-OVCD 只在验证集上调阈值到 0.75（无 MLLM）', '85.94', '85.94', '85.94', '87.33'])
p = REPO / 'results' / 'ovcd_v0_rounds.json'
data = json.loads(p.read_text())
data['cross_round'] = dict(
    text='测试集最终状态（M0+A+B+C）的 F1^c。每轮只改一个因素，依据上一轮验证集诊断决定，测试集每轮只评一次。',
    figs=[dict(src='../figures/ovcd_v0/cross_round.png', caption='左：各轮测试集最终 F1^c 随 k 的变化；右：阶段 B 的贡献（B − A）')],
    table=dict(columns=cols, rows=rows))
p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
print(rows)
