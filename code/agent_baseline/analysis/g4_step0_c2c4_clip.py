"""Step 0.1: G3 C2/C4 Agent final masks intersected with the box after the fact, paired vs post-hoc ∩ box."""
import json
from pathlib import Path
import numpy as np
from agent_baseline.analysis.second_box_metrics import counts, score, paired_bootstrap
from agent_baseline.analysis.second_metrics import mask
from agent_baseline.analysis.second_box_baselines import final_prediction
from agent_baseline.tasks import read_tasks

R = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
spec = json.loads((R / 'second/box_metrics_spec.json').read_text())
out = {}
for kind, group in spec['groups'].items():
    rows = read_tasks(group['manifest'])
    task = rows[0]['task_type']
    runs = [p.replace('second_g2box_', 'second_g3box_') for p in group['agent_runs']]
    raw, clip, inside_lost = [], [], []
    for run in runs:
        a, c = [], []
        for r in rows:
            truth = mask(r['gt_mask']); region = mask(r['region_mask'], truth.shape)
            opp = mask(Path(group['demolished_dir']) / Path(r['id']).name, truth.shape) if task == 'T1.1xC4' else None
            pred, _ = final_prediction(Path(run) / Path(r['id']).stem, truth.shape)
            a.append(counts(pred, truth, region, opp)); c.append(counts(pred & region, truth, region, opp))
        raw.append(a); clip.append(c)
    raw, clip = np.asarray(raw, np.int64), np.asarray(clip, np.int64)
    post = group['baselines']['posthoc']
    B = np.asarray([[counts(mask(Path(d) / (Path(r['id']).stem + '.png')), mask(r['gt_mask']), mask(r['region_mask']),
                            mask(Path(group['demolished_dir']) / Path(r['id']).name) if task == 'T1.1xC4' else None)
                     for r in rows] for d in post])
    m = lambda X: {k: float(np.mean([score(x)[k] for x in X])) for k in ('f1_c', 'precision', 'recall', 'outside_violation')}
    out[kind] = dict(agent=m(raw), agent_clipped=m(clip), posthoc=m(B),
                     clipped_minus_posthoc=dict(delta=m(clip)['f1_c'] - m(B)['f1_c'], ci95=paired_bootstrap(clip, B)),
                     raw_minus_posthoc=dict(delta=m(raw)['f1_c'] - m(B)['f1_c'], ci95=paired_bootstrap(raw, B)))
(R / 'results/agent_baseline/data/g4_step0_c2c4_clip.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, {x: (round(y['f1_c'], 2), round(y['precision'], 2), round(y['recall'], 2)) for x, y in v.items() if 'f1_c' in y},
          'clip-post', round(v['clipped_minus_posthoc']['delta'], 2), [round(z, 2) for z in v['clipped_minus_posthoc']['ci95']],
          'raw-post', round(v['raw_minus_posthoc']['delta'], 2))
