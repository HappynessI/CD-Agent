"""LEVIR val: G3 vs G2, paired by image, 3 repeats each. Dataset-level F1 (summed counts), mean over repeats.
Also counts box_too_large tool errors in the G3 runs. Labels only score."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
R = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
sys.path[:0] = [str(R / 'code_p/ovcd_v0'), str(R / 'code_p')]
from common import names, load_gt

TAGS = sys.argv[1:] or ['g2', 'g3']


def work(a):
    tag, task, n, name = a
    stem = Path(name).stem
    d = R / f'json_qwen37flash_{tag}_s{n}' / task / 'val' / stem
    if not (d / 'final.json').exists():
        return None
    gt = load_gt('val', name)
    fin = json.load(open(d / 'final.json'))
    m = np.asarray(Image.open(d / fin['mask_path'])) > 0
    big = 0
    for line in open(d / 'steps.jsonl'):
        big += line.count('"box_too_large"')
    return dict(tag=tag, task=task, n=n, stem=stem, tp=int((m & gt).sum()), fp=int((m & ~gt).sum()),
                fn=int((~m & gt).sum()), big=big, cost=fin.get('cost_cny', 0))


def f1(tp, fp, fn):
    return 200 * tp / max(1, 2 * tp + fp + fn)


if __name__ == '__main__':
    todo = [(t, k, n, x) for t in TAGS for k in ('T1.1xC0', 'T5.1') for n in (1, 2, 3) for x in names('val')]
    with Pool(8) as p:
        res = [r for r in p.map(work, todo) if r]
    out = {}
    rng = np.random.default_rng(0)
    for task in ('T1.1xC0', 'T5.1'):
        stems = sorted({r['stem'] for r in res if r['task'] == task})
        cube = {}
        for t in TAGS:
            for n in (1, 2, 3):
                rows = {r['stem']: r for r in res if r['tag'] == t and r['task'] == task and r['n'] == n}
                cube[t, n] = rows
        common = [s for s in stems if all(s in cube[t, n] for t in TAGS for n in (1, 2, 3))]
        arr = {t: np.array([[[cube[t, n][s][k] for k in ('tp', 'fp', 'fn')] for s in common] for n in (1, 2, 3)]) for t in TAGS}
        score = lambda a, idx: np.mean([f1(*a[n][idx].sum(0)) for n in range(3)])
        allidx = np.arange(len(common))
        row = {t: dict(f1=score(arr[t], allidx),
                       per_seed=[f1(*arr[t][n].sum(0)) for n in range(3)],
                       sessions=sum(len(cube[t, n]) for n in (1, 2, 3)),
                       big_box_errors=sum(cube[t, n][s]['big'] for n in (1, 2, 3) for s in cube[t, n]),
                       sessions_with_big=sum(cube[t, n][s]['big'] > 0 for n in (1, 2, 3) for s in cube[t, n]))
               for t in TAGS}
        if len(TAGS) == 2:
            a, b = TAGS
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(common), len(common))
                boots.append(score(arr[b], idx) - score(arr[a], idx))
            row['delta'] = dict(mean=row[b]['f1'] - row[a]['f1'], ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
        row['images'] = len(common)
        out[task] = row
    json.dump(out, open(R / 'results/agent_baseline/data/g3_vs_g2_levir.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
