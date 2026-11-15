"""Cross-round comparison on test: final-state F1^c vs k per round, with M0 / I-M-C references.

    python figs_cross.py --rounds 1,2,3
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import FIGS, RUN, read_json

sys.path.insert(0, str(Path(__file__).parent / 'page'))
import make_figs as MF  # noqa: E402
from figs_round import C, save  # noqa: E402

IMC = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch56_ovcd_de/run/imc/test/metrics.json')


def main(a):
    MF.configure_font(None)
    rounds = [int(r) for r in a.rounds.split(',')]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    m0 = None
    for i, r in enumerate(rounds):
        ks, f1, b = [], [], []
        for k in (10, 20, 50):
            e = read_json(RUN / 'eval' / f'r{r}_k{k}_test.json')
            ks.append(k)
            f1.append(100 * e['states']['C']['f1'])
            b.append(100 * (e['states']['B']['f1'] - e['states']['A']['f1']))
            m0 = 100 * e['states']['M0']['f1']
        axes[0].plot(ks, f1, 'o-', color=C[i % len(C)], label=f'第 {r} 轮')
        axes[1].bar([x + (i - (len(rounds) - 1) / 2) * 0.8 / len(rounds) for x in range(3)], b,
                    0.8 / len(rounds), color=C[i % len(C)], label=f'第 {r} 轮')
    m0s = {}
    for r in rounds:
        e = read_json(RUN / 'eval' / f'r{r}_k20_test.json')
        m0s[e['config'].get('m0', 'mci')] = 100 * e['states']['M0']['f1']
    for kind, ls in (('mci', ':'), ('imc', '--')):
        if kind in m0s:
            axes[0].axhline(m0s[kind], color='gray', ls=ls, lw=1,
                            label=f"M0 = {'M-C-I' if kind == 'mci' else 'I-M-C'}（{m0s[kind]:.2f}）")
    axes[0].set_xlabel('每图 MLLM 调用上限 k')
    axes[0].set_ylabel('测试集最终 $F_1^c$ (%)')
    axes[0].set_title('各轮测试集最终状态（M0+A+B+C）')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color='gray', lw=.8)
    axes[1].set_xticks(range(3), ['k=10', 'k=20', 'k=50'])
    axes[1].set_ylabel('百分点')
    axes[1].set_title('阶段 B 的贡献（B − A，测试集）')
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, FIGS / 'cross_round.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', required=True)
    main(p.parse_args())
