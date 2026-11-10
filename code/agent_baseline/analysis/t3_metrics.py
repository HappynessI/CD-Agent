"""T3.1 Agent vs fixed route on the same task subset; pooled pixel F1, paired task bootstrap.

The exemplar's pixels are removed from every prediction and from the truth (same rule as t3_fixed).
One task per image, so the bootstrap unit (task) is also the image. Three Agent repeats stay fixed
and share each resample; 4000 samples, seed 1016. Labels only score.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import write_json


def load(path):
    with Image.open(path) as image:
        return np.asarray(image) > 0


def counts(pred, truth):
    return [int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())]


def score(c):
    tp, fp, fn = np.asarray(c, float).sum(axis=0)
    return dict(f1_c=200 * tp / max(1, 2 * tp + fp + fn), precision=100 * tp / max(1, tp + fp),
                recall=100 * tp / max(1, tp + fn))


def paired(agent, base, samples=4000, seed=1016):
    rng = np.random.default_rng(seed)
    f = lambda idx: np.mean([score(a[idx])['f1_c'] for a in agent]) - score(base[idx])['f1_c']
    values = [f(rng.integers(0, base.shape[0], base.shape[0])) for _ in range(samples)]
    return dict(delta_f1=float(f(np.arange(base.shape[0]))), ci95=[float(v) for v in np.percentile(values, [2.5, 97.5])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--fixed-dir', type=Path, required=True)
    p.add_argument('--agent-runs', type=Path, nargs=3, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    fixed, agent, diag = [], [[] for _ in a.agent_runs], [dict(fallback=0, error=0) for _ in a.agent_runs]
    for r in rows:
        ex = load(r['exemplar_mask'])
        truth = load(r['gt_mask']) & ~ex
        fixed.append(counts(load(a.fixed_dir / (r['id'] + '.png')) & ~ex, truth))
        for k, run in enumerate(a.agent_runs):
            d = run / r['id']
            fin = json.loads((d / 'final.json').read_text())
            diag[k]['fallback'] += fin['submitted_by'] != 'model'
            diag[k]['error'] += fin['submitted_by'] == 'error' or bool(fin.get('error'))
            pred = load(d / fin['mask_path']) if fin.get('mask_path') else np.zeros_like(truth)
            agent[k].append(counts(pred & ~ex, truth))
    A, F = np.asarray(agent, np.int64), np.asarray(fixed, np.int64)
    per_run = [score(x) for x in A]
    report = dict(tasks=len(rows), agent_mean={k: float(np.mean([r[k] for r in per_run])) for k in per_run[0]},
                  agent_per_run=per_run, fixed=score(F), agent_minus_fixed=paired(A, F), diagnostics=diag,
                  rule='exemplar pixels removed from predictions and truth; pooled pixel F1; paired task bootstrap 4000, seed 1016')
    write_json(a.output, report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
