"""Compare repeated runs of configurations on val: per-run F1^c, run-to-run spread, paired differences, cache and cost.

    python compare_runs.py --name v1=json_qwen37plus_v1fix_s1,json_qwen37plus_v1fix_s2 --name v2=...
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'agent_baseline'
    import agent_baseline

import argparse
import json
import os
from pathlib import Path
import numpy as np
from common import load_gt, load_mask_png, names, confusion, image_f1, scores, object_counts, write_json

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
TASKS = ('T1.1xC0', 'T5.1')


def load_run(config, task, split, gts):
    rows = {}
    for name in names(split):
        d = RUN / config / task / split / Path(name).stem
        final = json.loads((d / 'final.json').read_text())
        pred = load_mask_png(d / final['mask_path'])
        rows[name] = dict(c=confusion(pred, gts[name]), f=image_f1(pred, gts[name]), o=object_counts(pred, gts[name]),
                          by=final['submitted_by'], rounds=final['planning_rounds'], cost=final['cost_cny'],
                          tok=final['tokens'], sec=final['elapsed_seconds'], same_init=final.get('identical_to_init'))
    return rows


def f1c(rows, keys=None):
    keys = list(rows) if keys is None else keys
    tot = {k: sum(rows[n]['c'][k] for n in keys) for k in ('tp', 'fp', 'fn')}
    return tot


def stats(rows):
    t = f1c(rows)
    s = scores(t)
    o = {k: sum(r['o'][k] for r in rows.values()) for k in ('gt_objects', 'missed', 'pred_objects', 'false_alarms')}
    tok = {k: sum(r['tok'][k] for r in rows.values()) for k in ('input', 'cached', 'output')}
    return dict(f1=100 * s['f1'], iou=100 * s['iou'], p=100 * s['precision'], r=100 * s['recall'],
                miss=100 * o['missed'] / o['gt_objects'], fa=100 * o['false_alarms'] / max(o['pred_objects'], 1),
                mean_f=100 * float(np.mean([r['f'] for r in rows.values()])), rounds=float(np.mean([r['rounds'] for r in rows.values()])),
                fail=100 * float(np.mean([r['by'] != 'model' for r in rows.values()])), errors=sum(r['by'] == 'error' for r in rows.values()),
                cost=sum(r['cost'] for r in rows.values()), hit=100 * tok['cached'] / max(tok['input'], 1),
                input_m=tok['input'] / 1e6, sec=float(np.mean([r['sec'] for r in rows.values()])),
                same_init=sum(bool(r['same_init']) for r in rows.values()))


def paired_ci(a_runs, b_runs, keys, n=4000, seed=0):
    """Image and run resampling for mean F1^c(a) - F1^c(b); b_runs may be a single fixed control."""
    rng = np.random.default_rng(seed)
    def cube(runs):
        return np.array([[[r[k]['c'][x] for x in ('tp', 'fp', 'fn')] for k in keys] for r in runs], float)
    A, B = cube(a_runs), cube(b_runs)
    def f(t):
        return 200 * t[..., 0] / np.maximum(2 * t[..., 0] + t[..., 1] + t[..., 2], 1)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        ra = rng.integers(0, len(a_runs), len(a_runs))
        rb = rng.integers(0, len(b_runs), len(b_runs))
        out.append(f(A[ra][:, idx].sum(1)).mean() - f(B[rb][:, idx].sum(1)).mean())
    return [float(np.percentile(out, q)) for q in (2.5, 97.5)], float(np.mean(np.array(out) > 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', action='append', required=True, help='label=config1,config2,...')
    p.add_argument('--split', default='val')
    p.add_argument('--out')
    a = p.parse_args()
    keys = names(a.split)
    gts = {n: load_gt(a.split, n) for n in keys}
    groups = dict(item.split('=') for item in a.name)
    imc = {n: dict(c=confusion(load_mask_png(RUN / 'init' / 'sam3_imc' / a.split / n), gts[n])) for n in keys}
    best = json.loads((RUN / 'fixed' / 'tune_val.json').read_text())
    out = dict(controls=dict(imc=100 * scores(f1c(imc))['f1'], tuned=100 * best['f1'], tuned_thr=best['selected_score_thr']), tasks={})
    print('controls', out['controls'])
    for task in TASKS:
        runs = {g: {c: load_run(c, task, a.split, gts) for c in cfgs.split(',')} for g, cfgs in groups.items()}
        res = {}
        for g, rs in runs.items():
            each = {c: stats(r) for c, r in rs.items()}
            f = [e['f1'] for e in each.values()]
            res[g] = dict(runs=each, f1_mean=float(np.mean(f)), f1_sd=float(np.std(f, ddof=1)) if len(f) > 1 else None,
                          ci_vs_imc=paired_ci(list(rs.values()), [imc], keys)[0], p_above_imc=paired_ci(list(rs.values()), [imc], keys)[1])
            print(task, g, 'F1 per run', [round(x, 2) for x in f], 'mean', round(res[g]['f1_mean'], 2), 'sd', res[g]['f1_sd'] and round(res[g]['f1_sd'], 2),
                  'CI vs imc', [round(x, 2) for x in res[g]['ci_vs_imc']], 'P(>imc)', round(res[g]['p_above_imc'], 3))
            for c, e in each.items():
                print('   ', c, {k: round(v, 1) if isinstance(v, float) else v for k, v in e.items()})
            # how unstable is each image across runs?
            spread = {n: float(np.ptp([r[n]['f'] for r in rs.values()])) for n in keys}
            print('    images with F1 range > 0.2 across runs:', sum(v > .2 for v in spread.values()), 'mean range', round(float(np.mean(list(spread.values()))), 3))
            res[g]['unstable_images'] = sorted(spread, key=spread.get, reverse=True)[:8]
        labels = list(runs)
        for i, x in enumerate(labels):
            for y in labels[i + 1:]:
                ci, pos = paired_ci(list(runs[y].values()), list(runs[x].values()), keys)
                print(task, f'{y} - {x}', 'diff', round(res[y]['f1_mean'] - res[x]['f1_mean'], 2), 'CI', [round(v, 2) for v in ci], 'P(>0)', round(pos, 3))
                res[f'{y}-{x}'] = dict(diff=res[y]['f1_mean'] - res[x]['f1_mean'], ci=ci, p_pos=pos)
        out['tasks'][task] = res
    if a.out:
        write_json(a.out, out)


if __name__ == '__main__':
    main()
