"""M1 queue diagnostics on val (decides the stage-A order and the A/B budget split).

1. Stage-A order: for each candidate order, an oracle that verifies the first n components of each
   image and makes the best possible deletions among them (best precision-sorted prefix); F1 vs n.
2. Stage-B (T7) coverage: share of whole-missed GT objects (IoU < 0.1 with M0) that a top-n
   candidate hits (candidate covers >= 30% of the object, or >= 50% of the candidate is on it),
   and the F1 of an oracle that adds every hitting candidate mask among the top n.

    python m1_queue_diag.py --tag r1 --split val
"""
from __future__ import annotations
import argparse

import numpy as np
from scipy import ndimage as ndi

from common import load_m0, EIGHT, MCI, RUN, load_gt, load_mask_png, names, read_json, scores, write_json
import items as I
from m1_errors import best_deletion

ORDERS = ['area_asc', 'area_desc', 'score_asc', 'bprob_asc', 'bother_desc', 'nprops_asc']
NS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 1000]


def main(a):
    m1 = read_json(RUN / 'm1' / a.tag / f'{a.split}.json')['summary']
    f0 = m1['f1']
    tot = dict(tp=m1['tp'], fp=m1['fp'], fn=m1['fn'])
    comps = read_json(RUN / 'm1' / a.tag / f'{a.split}_components.json')
    by_img = {}
    for r in comps:
        by_img.setdefault(r['name'], []).append(r)
    order_res = {}
    for o in ORDERS:
        key, d = o.rsplit('_', 1)
        curve = {}
        for n in NS:
            seen = [r for rows in by_img.values() for r in sorted(rows, key=lambda r: r[key], reverse=d == 'desc')[:n]]
            curve[n] = best_deletion(seen, tot['tp'], tot['fp'], tot['fn'])[0]['f1']
        order_res[o] = curve
    # T7 coverage
    hit_n = {n: 0 for n in NS}
    add = {n: 0 for n in NS}
    missed_total = 0
    missed_px = 0
    for name in names(a.split):
        m0 = load_m0(a.m0, a.split, name)
        gt = load_gt(a.split, name)
        gl, _ = ndi.label(gt, EIGHT)
        pl, _ = ndi.label(m0, EIGHT)
        missed = []
        for i, sl in enumerate(ndi.find_objects(gl), 1):
            obj = gl == i
            t = np.unique(pl[obj]); t = t[t > 0]
            other = np.isin(pl, t)
            if (obj & other).sum() / max((obj | other).sum(), 1) < 0.1:
                missed.append(obj)
        missed_total += len(missed)
        missed_px += sum(int(o.sum()) for o in missed)
        st = I.Store(a.tag, a.split, name)
        cands = [st.mask(it['uid']) for it in st.meta['B']]
        for n in NS:
            got = np.zeros(gt.shape, bool)
            hit = 0
            for obj in missed:
                h = [c for c in cands[:n] if (c & obj).sum() >= 0.3 * obj.sum() or (c & obj).sum() >= 0.5 * c.sum()]
                if h:
                    hit += 1
                    for c in h:
                        got |= c
            hit_n[n] += hit
            add[n] += int((got & gt & ~m0).sum()) - 0  # TP gain
            add.setdefault(f'fp{n}', 0)
            add[f'fp{n}'] += int((got & ~gt & ~m0).sum())
    cover = {n: dict(hit_share=hit_n[n] / max(missed_total, 1),
                     oracle_f1=scores(dict(tp=tot['tp'] + add[n], fp=tot['fp'] + add[f'fp{n}'],
                                           fn=tot['fn'] - add[n]))['f1']) for n in NS}
    out = dict(f1_m0=f0, order_oracle=order_res, t7=dict(missed_objects=missed_total, missed_pixels=missed_px,
                                                         by_topn=cover))
    write_json(RUN / 'm1' / a.tag / f'{a.split}_queue_diag.json', out)
    for o, c in order_res.items():
        print(o, {n: round(v, 4) for n, v in c.items()})
    print('T7', {n: (round(v['hit_share'], 3), round(v['oracle_f1'], 4)) for n, v in cover.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='r1')
    p.add_argument('--m0', default='mci', choices=['mci', 'imc', 'rea'])
    p.add_argument('--split', default='val')
    main(p.parse_args())
