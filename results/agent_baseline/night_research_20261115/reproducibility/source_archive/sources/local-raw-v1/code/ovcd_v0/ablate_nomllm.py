"""No-MLLM comparison rows on the ReA-OVCD initializer (thresholds chosen on val, test scored once).

  score-only : M0 & (change_score > t)
  clip+score : SAM3 support clip of M0 (same rule as the agent's stage C: 512 tiles, instance score
               >= c_conf, cover >= c_cover, radius 0), then keep only pixels with change_score > t
t = 0 means no score filter.  Writes RUN/eval/nomllm_rea.json.

    python ablate_nomllm.py --c-conf 0.3 --c-cover 0.5
"""
from __future__ import annotations
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from common import EIGHT, REA, RUN, load_gt, load_m0, names, scores, write_json
from sam3_t7 import read

TS = [0.0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def support(split, name, conf):
    S = np.zeros((1024, 1024), bool)
    src = dict(conf=0.5) if conf >= 0.5 else dict(conf=0.1, min_score=conf)
    for m, _ in read(split, name, 512, **src):
        if len(m):
            S |= m.any(0)
    return S


def clip(M, S, cover):
    lab, n = ndi.label(M, EIGHT)
    inside = np.bincount(lab[S & M], minlength=n + 1)
    total = np.bincount(lab.ravel(), minlength=n + 1)
    ok = inside >= cover * np.maximum(total, 1)
    ok[0] = False
    return (M & ~ok[lab]) | (ok[lab] & S)


def work(args):
    split, name, a = args
    gt = load_gt(split, name)
    M = load_m0('rea', split, name)
    sc = np.load(REA / split / 'score' / (Path(name).stem + '.npz'))['change_score'].astype(np.float32)
    Mc = clip(M, support(split, name, a.c_conf), a.c_cover)
    out = {}
    for mode, base in (('score', M), ('clip+score', Mc)):
        for t in TS:
            P = base & (sc > t) if t else base
            out[(mode, t)] = [int((P & gt).sum()), int((P & ~gt).sum()), int((~P & gt).sum())]
    return out


def agg(res, key):
    tp, fp, fn = (sum(r[key][i] for r in res) for i in range(3))
    return dict(tp=tp, fp=fp, fn=fn, **scores(dict(tp=tp, fp=fp, fn=fn)))


def main(a):
    out = {}
    with Pool(a.procs) as pool:
        val = pool.map(work, [('val', n, a) for n in names('val')])
        test = pool.map(work, [('test', n, a) for n in names('test')])
    for mode in ('score', 'clip+score'):
        grid = {t: agg(val, (mode, t)) for t in TS}
        t_best = max(TS, key=lambda t: grid[t]['f1'])
        out[mode] = dict(val_grid={str(t): g['f1'] for t, g in grid.items()}, chosen_t=t_best,
                         val=grid[t_best], test=agg(test, (mode, t_best)))
        print(mode, 'val grid', {t: round(g['f1'] * 100, 2) for t, g in grid.items()}, 'chosen', t_best,
              'test F1 %.2f P %.1f R %.1f' % tuple(100 * out[mode]['test'][x] for x in ('f1', 'precision', 'recall')), flush=True)
    write_json(RUN / 'eval' / 'nomllm_rea.json', dict(args=vars(a), **out))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--c-conf', type=float, default=0.3)
    p.add_argument('--c-cover', type=float, default=0.5)
    p.add_argument('--procs', type=int, default=4)
    main(p.parse_args())
