"""SECOND T1.1xC1 on G3: same baselines as the G2 C1 run, plus a paired G3 - G2 Agent delta and big-box error counts."""
import json
from pathlib import Path
import numpy as np
from agent_baseline.analysis.second_metrics import evaluate_group, counts, score, mask
from agent_baseline.tasks import read_tasks

R = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
spec = json.loads((R / 'second/metrics_spec.json').read_text())


def agent_counts(group, runs):
    rows = read_tasks(group['manifest'])
    direction = rows[0]['change_direction']
    out, big = [], []
    for run in runs:
        per, b = [], 0
        for r in rows:
            gt = mask(r['gt_mask'])
            kind = 'demolished' if direction == 'appeared' else 'new'
            opp = mask(Path(r['gt_mask']).parent.parent / kind / Path(r['gt_mask']).name, gt.shape)
            d = Path(run) / Path(r['id']).stem
            fin = json.loads((d / 'final.json').read_text())
            pred = mask(d / fin['mask_path'], gt.shape) if fin.get('mask_path') else np.zeros_like(gt)
            per.append(counts(pred, gt, opp))
            b += (d / 'steps.jsonl').read_text().count('"box_too_large"')
        out.append(per)
        big.append(b)
    return np.asarray(out, np.int64), big


def paired(a, b, samples=4000, seed=1016):
    rng = np.random.default_rng(seed)
    f = lambda x, idx: np.mean([score(r[idx])['f1_c'] for r in x])
    vals = []
    for _ in range(samples):
        idx = rng.integers(0, a.shape[1], a.shape[1])
        vals.append(f(a, idx) - f(b, idx))
    allidx = np.arange(a.shape[1])
    return dict(delta_f1=float(f(a, allidx) - f(b, allidx)), ci95=[float(v) for v in np.percentile(vals, [2.5, 97.5])])


result = {}
for name, group in spec['groups'].items():
    if not name.startswith('T1.1xC1'):
        continue
    g2_runs = group['agent_runs']
    g3_runs = [p.replace('second_g2_', 'second_g3_') for p in g2_runs]
    res = evaluate_group(dict(group, agent_runs=g3_runs))
    A3, big3 = agent_counts(group, g3_runs)
    A2, big2 = agent_counts(group, g2_runs)
    res['g2_agent_mean'] = {k: float(np.mean([score(a)[k] for a in A2])) for k in ('f1_c', 'precision', 'recall', 'direction_violation')}
    res['comparisons']['g2_agent'] = paired(A3, A2)
    res['big_box_errors'] = dict(g3=big3, g2=big2)
    result[name] = res
(R / 'results/agent_baseline/data/second_metrics_g3_c1.json').write_text(json.dumps(result, indent=1))
for name, res in result.items():
    print(name, 'G3', {k: round(v, 2) for k, v in res['agent_mean'].items()})
    print(name, 'G2', {k: round(v, 2) for k, v in res['g2_agent_mean'].items()})
    for k, c in res['comparisons'].items():
        print('  G3 -', k, round(c['delta_f1'], 2), [round(x, 2) for x in c['ci95']])
    print('  big-box errors', res['big_box_errors'], 'diag', [(d['error_sessions'], d['fallback_sessions']) for d in res['run_diagnostics']])
