"""T3.2/3/4 paired metrics. Reuse T3 pixel counts and paired bootstrap.

All evaluation pixels count. External crops are never evaluation images.
Resample images (all transitions on an image together), separately per type.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from common import write_json
from agent_baseline.tasks import read_tasks
from agent_baseline.examples_t3x import INSTRUCTIONS
from .t3_metrics import load, counts, score, paired


def grouped_counts(rows, agent, fixed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['image_t1']].append(i)
    indices = [groups[k] for k in sorted(groups)]
    return (np.stack([agent[:, idx].sum(axis=1) for idx in indices], axis=1),
            np.stack([fixed[idx].sum(axis=0) for idx in indices]))


def compare(rows, fixed_dir, agent_runs, samples=4000):
    if not rows or any(r['task_type'] not in INSTRUCTIONS for r in rows):
        raise ValueError('Require nonempty T3.2/3/4 tasks')
    fixed, agents = [], [[] for _ in agent_runs]
    diagnostics = [dict(fallback=0, error=0) for _ in agent_runs]
    for row in rows:
        truth = load(row['gt_mask'])
        pred = load(fixed_dir / (row['id'] + '.png'))
        if pred.shape != truth.shape:
            raise ValueError('Fixed mask shape differs from truth')
        fixed.append(counts(pred, truth))
        for k, run in enumerate(agent_runs):
            directory = run / row['task_type'] / 'val' / row['id']
            fin = json.loads((directory / 'final.json').read_text())
            diagnostics[k]['fallback'] += fin['submitted_by'] != 'model'
            diagnostics[k]['error'] += fin['submitted_by'] == 'error' or bool(fin.get('error'))
            pred = load(directory / fin['mask_path']) if fin.get('mask_path') else np.zeros_like(truth)
            if pred.shape != truth.shape:
                raise ValueError('Agent mask shape differs from truth')
            agents[k].append(counts(pred, truth))
    A, F = np.asarray(agents, np.int64), np.asarray(fixed, np.int64)
    def summary(indices):
        subset = [rows[i] for i in indices]
        grouped_a, grouped_f = grouped_counts(subset, A[:, indices], F[indices])
        runs = [score(a) for a in grouped_a]
        return dict(tasks=len(indices), images=len(grouped_f), fixed=score(grouped_f), agent_per_run=runs,
            agent_mean={k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
            agent_minus_fixed=paired(grouped_a, grouped_f, samples=samples))
    per_type = {k: summary([i for i, r in enumerate(rows) if r['task_type'] == k])
                for k in sorted({r['task_type'] for r in rows})}
    per_transition = {}
    for kind in per_type:
        for transition in sorted({r['meta']['transition'] for r in rows if r['task_type'] == kind}):
            ids = [i for i, r in enumerate(rows) if r['task_type'] == kind and r['meta']['transition'] == transition]
            per_transition[kind + ':' + transition] = dict(tasks=len(ids), fixed=score(F[ids]),
                agent_per_run=[score(a[ids]) for a in A])
    return dict(tasks=len(rows), per_type=per_type, per_transition=per_transition, diagnostics=diagnostics,
        per_task={r['id']: dict(fixed=F[i].tolist(), agent=A[:, i].tolist()) for i, r in enumerate(rows)},
        rule=f'All evaluation pixels; external examples never scored; pooled pixel F1; paired image bootstrap {samples}, seed 1016')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--fixed-dir', type=Path, required=True)
    p.add_argument('--agent-runs', type=Path, nargs=3, required=True, help='Three config roots, containing T3.x/val/<id>')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = compare(read_tasks(a.tasks), a.fixed_dir, a.agent_runs)
    write_json(a.output, report)
    print(json.dumps(report['per_type'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
