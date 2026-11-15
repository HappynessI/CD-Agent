"""Per-round diagnostic figures (M0-M4 sections) from the RUN json files.

    python figs_round.py --round 1 --tag r1 --select r1 --runs r1_k10,r1_k20,r1_k50 --main r1_k20
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, RUN, read_json

sys.path.insert(0, str(Path(__file__).parent / 'page'))
import make_figs as MF  # noqa: E402

C = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']


def save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)
    print(out)


def m0_repro(out, mci_test, imc=None):
    rows = [('M-C-I 原文', 53.6, 36.6), ('M-C-I 本次', 100 * mci_test['f1'], 100 * mci_test['iou'])]
    if imc:
        rows += [('I-M-C 原文', 69.7, 53.5), ('I-M-C 本次', 100 * imc['f1'], 100 * imc['iou'])]
    else:
        rows += [('I-M-C 原文', 69.7, 53.5)]
    rea = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch66_ovcd_reaovcd/run/rea/test/metrics.json')
    if rea.exists():
        f = read_json(rea)['dynamic_earth']['fraction']
        rows += [('ReA-OVCD 原文', 84.52, 73.19), ('ReA-OVCD 本次', 100 * f['F1_c'], 100 * f['IoU_c'])]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(rows))
    for j, (lab, col) in enumerate((('$F_1^c$', C[0]), ('$IoU^c$', C[1]))):
        v = [r[1 + j] for r in rows]
        ax.bar(x + (j - .5) * .38, v, .38, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - .5) * .38, vi + 1, f'{vi:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel('%')
    ax.set_ylim(0, 95)
    ax.set_title('LEVIR-CD 测试集：初始化方法的复现')
    ax.legend(frameon=False)
    save(fig, out)


def m1_composition(out, m1v, m1t):
    labels = ['验证', '测试']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fp = [[m['fp_pixels']['whole_false_objects'], m['fp_pixels']['extra_on_hit_objects']] for m in (m1v, m1t)]
    fn = [[m['fn_cells'][k] for k in ('2', '1', '3', '5')] for m in (m1v, m1t)]
    for ax, data, names, title in ((axes[0], fp, ['整块误检（IoU<0.1）', '命中目标外多出的像素'], '误检像素构成'),
                                   (axes[1], fn, ['② 孤立无信号', '① 贴边无信号', '③ 有信号（覆盖≥10%）', '⑤ 极小'], '漏检像素构成')):
        base = np.zeros(2)
        for i, n in enumerate(names):
            v = np.array([d[i] for d in data], float) / 1e6
            ax.bar(labels, v, .55, bottom=base, color=C[i], label=n)
            base += v
        ax.set_ylabel('百万像素')
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    save(fig, out)


def m1_orders(out, diag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    for i, (o, curve) in enumerate(diag['order_oracle'].items()):
        ns = [int(n) for n in curve if int(n) <= 50]
        ax.plot(ns, [100 * curve[str(n)] for n in ns], 'o-', ms=3, color=C[i % 6], label=o)
    ax.axhline(100 * diag['f1_m0'], color='gray', ls=':', label='M0')
    ax.set_xlabel('每图核实的连通域数 n')
    ax.set_ylabel('理想删除后的 $F_1^c$ (%)')
    ax.set_title('阶段 A 排序（验证集，理想核实）')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    t7 = diag['t7']['by_topn']
    ns = [int(n) for n in t7 if int(n) <= 50]
    ax.plot(ns, [100 * t7[str(n)]['hit_share'] for n in ns], 'o-', color=C[2], label='命中的整目标漏检占比')
    ax2 = ax.twinx()
    ax2.plot(ns, [100 * t7[str(n)]['oracle_f1'] for n in ns], 's--', color=C[3], label='理想补入后的 $F_1^c$')
    ax.set_xlabel('每图取前 n 个 T7 候选')
    ax.set_ylabel('命中占比 (%)')
    ax2.set_ylabel('$F_1^c$ (%)')
    ax.set_title('T7 候选的覆盖（验证集）')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='lower right')
    ax2.spines[['top']].set_visible(False)
    save(fig, out)


def grid_curve(out, sel, stage, key, xlabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (k, b) in enumerate(sel['budgets'].items()):
        rows = b[stage]
        if stage == 'A':
            rows = [r for r in rows if r['config']['a_order'] == b['chosen']['a_order']]
        xs = [r['config'][key] for r in rows]
        ax.plot(xs, [100 * r['f1'] for r in rows], 'o-', ms=3, color=C[i], label=f'k={k}')
    m0 = sel.get('m0_f1')
    if m0:
        ax.axhline(100 * m0, color='gray', ls=':', label='M0')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('验证集 $F_1^c$ (%)')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out)


def gate_bars(out, sel):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ks = list(sel['budgets'])
    rows0 = sel['budgets'][ks[0]]['C']
    labels = ['不修' if r['config']['c_mode'] == 'none' else f"框 {r['config']['c_gate']:.1f}" if r['config']['c_mode'] == 'box'
              else f"裁 q{r['config'].get('c_conf', 0.5):g} r{r['config'].get('c_radius', 0)} c{r['config'].get('c_cover', 0):.1f}" for r in rows0]
    x = np.arange(len(labels))
    w = .8 / len(ks)
    for i, k in enumerate(ks):
        v = [100 * r['f1'] for r in sel['budgets'][k]['C']]
        ax.bar(x + (i - (len(ks) - 1) / 2) * w, v, w, color=C[i], label=f'k={k}')
    lo = min(100 * r['f1'] for k in ks for r in sel['budgets'][k]['C'])
    ax.set_ylim(lo - 3, None)
    ax.set_xticks(x, labels, fontsize=7, rotation=60)
    ax.set_ylabel('验证集 $F_1^c$ (%)')
    ax.set_title('阶段 C：不修 / SAM 框重分割（门限）/ SAM3 建筑支撑裁剪（半径 r、覆盖 c）')
    ax.legend(frameon=False, fontsize=8)
    save(fig, out)


def pyes_hist(out, diag):
    A, B = diag['A'], diag['B']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 1, 21)
    ax = axes[0]
    ax.hist([r['p'] for r in A if not r['delete']], bins, alpha=.7, color=C[0], label='应保留')
    ax.hist([r['p'] for r in A if r['delete']], bins, alpha=.7, color=C[3], label='应删除')
    s = diag['summary']['A']
    ax.set_title(f"阶段 A：M0 连通域的 P(yes)（AUC {s['auc_delete']:.2f}，按面积加权 {s['auc_delete_fp_weighted']:.2f}）", fontsize=9)
    ax.set_xlabel('P(yes)')
    ax.set_ylabel('连通域数')
    ax.legend(frameon=False)
    ax = axes[1]
    ax.hist([r['p'] for r in B if not r['useful']], bins, alpha=.7, color=C[0], label='非漏检')
    ax.hist([r['p'] for r in B if r['useful']], bins, alpha=.7, color=C[2], label='真漏检')
    s = diag['summary']['B']
    ax.set_title(f"阶段 B：T7 候选的 P(yes)（AUC {s['auc_useful']:.2f}）", fontsize=9)
    ax.set_xlabel('P(yes)')
    ax.legend(frameon=False)
    save(fig, out)


def box_quality(out, diag):
    B = [r for r in diag['B'] if r['useful'] and 'box_iou' in r]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter([r['cand_iou'] for r in B], [r['box_iou'] for r in B], s=10, alpha=.6, color=C[0])
    ax.plot([0, 1], [0, 1], color='gray', ls=':')
    ax.set_xlabel('候选掩码本身与真值的 IoU（近景内）')
    ax.set_ylabel('T5 框 + SAM 并集与真值的 IoU')
    ax.set_title('真漏检候选：画框是否更好', fontsize=10)
    save(fig, out)


def main(a):
    MF.configure_font(None)
    out = FIGS / f'round_{a.round}'
    mci = read_json(Path(a.mci_metrics))['metrics']
    imc = read_json(Path(a.imc_metrics)) if a.imc_metrics and Path(a.imc_metrics).exists() else None
    m0_repro(out / 'm0_repro.png', dict(f1=mci['f1_score_change'], iou=mci['iou_change']),
             dict(f1=imc['stitched']['metrics']['f1_score_change'], iou=imc['stitched']['metrics']['iou_change']) if imc else None)
    m1v = read_json(RUN / 'm1' / a.tag / 'val.json')['summary']
    m1t = read_json(RUN / 'm1' / a.tag / 'test.json')['summary']
    m1_composition(out / 'm1_composition.png', m1v, m1t)
    m1_orders(out / 'm1_queue.png', read_json(RUN / 'm1' / a.tag / 'val_queue_diag.json'))
    diag = read_json(RUN / 'diag' / a.tag / 'val_mllm.json')
    pyes_hist(out / 'm2_m3_pyes.png', diag)
    box_quality(out / 'm3_box.png', diag)
    sel = read_json(RUN / 'select' / a.select / 'val.json')
    sel['m0_f1'] = m1v['f1']
    grid_curve(out / 'm2_tau_a.png', sel, 'A', 'tau_a', '删除阈值 τ_A（P(yes) < τ_A 则删）', '阶段 A：删除阈值（验证集）')
    grid_curve(out / 'm3_tau_b.png', sel, 'B', 'tau_b', '加入阈值 τ_B（P(yes) ≥ τ_B 则补）', '阶段 B：加入阈值（验证集，A 已固定）')
    gate_bars(out / 'm4_gate.png', sel)
    if a.runs:
        evs = {r: read_json(RUN / 'eval' / f'{r}_test.json') for r in a.runs.split(',')}
        main_ev = evs[a.main]
        MF.plot_bars(dict(title=f'测试集四个状态（{a.main}）', states=[
            dict(label=lab, **{k: main_ev['states'][st][k] for k in ('f1', 'iou', 'precision', 'recall')})
            for st, lab in zip(('M0', 'A', 'B', 'C'), ('M0', 'M0+A', 'M0+A+B', 'M0+A+B+C'))]),
            out / 'test_states.png')
        series = []
        for st, lab in (('A', 'M0+A'), ('B', 'M0+A+B'), ('C', 'M0+A+B+C')):
            ks = [evs[r]['config']['k'] for r in evs]
            series.append(dict(label=lab, k=ks, f1=[evs[r]['states'][st]['f1'] for r in evs]))
        MF.plot_budget(dict(title='测试集预算曲线', series=series), out / 'test_budget.png')
        MF.plot_composition(dict(title=f'测试集误检与漏检像素（{a.main}）', labels=['M0', 'A', 'B', 'C'],
                                 ylabel='像素数', segments=[
                                     dict(label='误检 FP', values=[main_ev['states'][s]['fp'] for s in ('M0', 'A', 'B', 'C')]),
                                     dict(label='漏检 FN', values=[main_ev['states'][s]['fn'] for s in ('M0', 'A', 'B', 'C')])]),
                            out / 'test_composition.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--select', required=True)
    p.add_argument('--runs', default='')
    p.add_argument('--main', default='')
    p.add_argument('--mci-metrics', default='/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch56_ovcd_de/run/mci/test/metrics.json')
    p.add_argument('--imc-metrics', default='/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch56_ovcd_de/run/imc/test/metrics.json')
    main(p.parse_args())
