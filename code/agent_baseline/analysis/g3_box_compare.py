"""C2/C4 on G3: same baselines as the G2 line-B run, plus a paired G3 - G2 Agent delta (shared image bootstrap)."""
import json
from pathlib import Path
import numpy as np
from agent_baseline.analysis.second_box_metrics import evaluate_group, counts, score, paired_bootstrap
from agent_baseline.analysis.second_metrics import mask
from agent_baseline.analysis.second_box_baselines import final_prediction
from agent_baseline.tasks import read_tasks

R = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
spec = json.loads((R / 'second/box_metrics_spec.json').read_text())


def agent_counts(group, runs):
    rows = read_tasks(group['manifest'])
    task = rows[0]['task_type']
    out = []
    for run in runs:
        per = []
        for r in rows:
            truth = mask(r['gt_mask'])
            region = mask(r['region_mask'], truth.shape)
            opp = mask(Path(group['demolished_dir']) / Path(r['id']).name, truth.shape) if task == 'T1.1xC4' else None
            pred, _ = final_prediction(Path(run) / Path(r['id']).stem, truth.shape)
            per.append(counts(pred, truth, region, opp))
        out.append(per)
    return np.asarray(out, np.int64)


result = {}
for kind, group in spec['groups'].items():
    g2_runs = group['agent_runs']
    g3_runs = [p.replace('second_g2box_', 'second_g3box_') for p in g2_runs]
    g3 = dict(group, agent_runs=g3_runs)
    res = evaluate_group(g3)
    A3, A2 = agent_counts(g3, g3_runs), agent_counts(group, g2_runs)
    g2_mean = {k: float(np.mean([score(a)[k] for a in A2])) for k in ('f1_c', 'precision', 'recall', 'outside_violation', 'direction_violation')}
    res['g2_agent_mean'] = g2_mean
    res['comparisons']['g2_agent'] = dict(delta_f1=res['agent_mean']['f1_c'] - g2_mean['f1_c'], ci95=paired_bootstrap(A3, A2))
    result[kind] = res
(R / 'results/agent_baseline/data/second_box_metrics_g3.json').write_text(json.dumps(result, indent=1))
for kind, res in result.items():
    print(kind, 'G3', {k: round(v, 2) for k, v in res['agent_mean'].items()})
    print(kind, 'G2', {k: round(v, 2) for k, v in res['g2_agent_mean'].items()})
    for name, c in res['comparisons'].items():
        print('  G3 -', name, round(c['delta_f1'], 2), [round(x, 2) for x in c['ci95']])
    print('  diag', res['run_diagnostics'])
