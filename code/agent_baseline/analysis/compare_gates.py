"""Paired comparison of two Agent configurations (e.g. G4 vs G3), three repeats each, on one manifest.

Pooled pixel F1 per repeat, mean over repeats; paired image bootstrap with indices shared by all six
repeats (4000 samples, seed 1016). Optional region clipping and exemplar removal follow the task's
own metric: C2/C4 score the raw prediction (as in second_box_metrics), T3.x removes exemplar pixels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def load(path, shape=None):
    with Image.open(path) as image:
        m = np.asarray(image) > 0
    return m if shape is None or m.shape == shape else None


def prediction(directory, shape):
    final = json.loads((directory / 'final.json').read_text())
    if final.get('mask_path'):
        return load(directory / final['mask_path']), final['submitted_by'] != 'model'
    return np.zeros(shape, bool), True


def counts(pred, truth):
    return [int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())]


def f1(c):
    tp, fp, fn = np.asarray(c, float).sum(axis=0)
    return 200 * tp / max(1, 2 * tp + fp + fn)


def compare(manifest, runs_a, runs_b, samples=4000, seed=1016):
    rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    def side(runs):
        out, fb = [], []
        for run in runs:
            per, n = [], 0
            for r in rows:
                truth = load(r['gt_mask'])
                pred, fallback = prediction(Path(run) / Path(r['id']).stem, truth.shape)
                if r.get('exemplar_mask'):
                    ex = load(r['exemplar_mask'])
                    truth, pred = truth & ~ex, pred & ~ex
                per.append(counts(pred, truth)); n += fallback
            out.append(per); fb.append(n)
        return np.asarray(out, np.int64), fb
    A, fa = side(runs_a)
    B, fb = side(runs_b)
    rng = np.random.default_rng(seed)
    mean = lambda X, idx: np.mean([f1(x[idx]) for x in X])
    idx_all = np.arange(A.shape[1])
    values = []
    for _ in range(samples):
        idx = rng.integers(0, A.shape[1], A.shape[1])
        values.append(mean(A, idx) - mean(B, idx))
    return dict(images=len(rows), a=dict(f1=mean(A, idx_all), per_run=[f1(x) for x in A], fallback=fa),
                b=dict(f1=mean(B, idx_all), per_run=[f1(x) for x in B], fallback=fb),
                delta=float(mean(A, idx_all) - mean(B, idx_all)), ci95=[float(v) for v in np.percentile(values, [2.5, 97.5])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--a', type=Path, nargs=3, required=True, help='Three repeat directories of configuration A')
    p.add_argument('--b', type=Path, nargs=3, required=True, help='Three repeat directories of configuration B')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = compare(a.manifest, a.a, a.b)
    a.output.write_text(json.dumps(r, indent=1))
    print(json.dumps(dict(a=round(r['a']['f1'], 2), b=round(r['b']['f1'], 2), delta=round(r['delta'], 2),
                          ci95=[round(x, 2) for x in r['ci95']], fallback_a=r['a']['fallback'], fallback_b=r['b']['fallback'])))


if __name__ == '__main__':
    main()
