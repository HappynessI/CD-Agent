"""Validation diagnostic: clip the current mask to the SAM3 building support S = dilate(S1 | S2, r).

For each connected component of the input mask: if at least `cover` of it lies inside S, keep only
its pixels inside S; otherwise leave it unchanged (SAM3 may have missed that building).  Inputs are
a run's stage masks (e.g. r6_k20 B) or the M0.  No MLLM calls.

    python diag_sam3clip.py --run r6_k20 --state B --tile 512
"""
from __future__ import annotations
import argparse
from itertools import product
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

from common import EIGHT, RUN, load_gt, load_mask_png, names, scores, write_json
from sam3_t7 import read

RADII = [0, 2, 4, 8, 16]
COVERS = [0.0, 0.3, 0.5, 0.7]


def clip(M, S, cover):
    lab, n = ndi.label(M, EIGHT)
    if n == 0:
        return M
    inside = np.bincount(lab[S & M], minlength=n + 1)
    total = np.bincount(lab.ravel(), minlength=n + 1)
    ok = inside >= cover * np.maximum(total, 1)
    ok[0] = False
    clipped = ok[lab] & S
    return (M & ~ok[lab]) | clipped


def work(args):
    a, name = args
    gt = load_gt('val', name)
    M = load_mask_png(RUN / 'agent' / a.run / 'val' / 'masks' / a.state / name)
    inst = read('val', name, a.tile)
    S0 = np.zeros_like(gt)
    for m, _ in inst:
        if len(m):
            S0 |= m.any(0)
    out = {}
    for r in RADII:
        S = ndi.binary_dilation(S0, EIGHT, iterations=r) if r else S0
        for c in COVERS:
            P = clip(M, S, c)
            out[(r, c)] = [int((P & gt).sum()), int((P & ~gt).sum()), int((~P & gt).sum())]
    out['base'] = [int((M & gt).sum()), int((M & ~gt).sum()), int((~M & gt).sum())]
    return out


def main(a):
    with Pool(a.procs) as pool:
        res = pool.map(work, [(a, n) for n in names('val')])
    rows = []
    for key in ['base'] + list(product(RADII, COVERS)):
        tp, fp, fn = (sum(r[key][i] for r in res) for i in range(3))
        s = scores(dict(tp=tp, fp=fp, fn=fn))
        rows.append(dict(key=str(key), tp=tp, fp=fp, fn=fn, **s))
        print(key, f"F1 {s['f1']*100:.2f} P {s['precision']*100:.1f} R {s['recall']*100:.1f}", flush=True)
    write_json(RUN / 'diag' / f'sam3clip_{a.run}_{a.state}_t{a.tile}.json', rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='r6_k20')
    p.add_argument('--state', default='B')
    p.add_argument('--tile', type=int, default=512)
    p.add_argument('--procs', type=int, default=4)
    main(p.parse_args())
