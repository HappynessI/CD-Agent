"""T7 source "ReA near-threshold / filtered": regions where ReA-OVCD's normalized change score before
its post-filter exceeds `thr` but which are not in M0 (removed by the boundary / opening / <150 px
filters, or just below the 0.5 change threshold when thr < 0.5).

  queue (CPU): A = the A items of --from-tag (same uids and masks; cached answers copied from
         --calls-from); B = 8-connected components of (score_before_filter > thr) & ~M0 with area in
         [min_area, max_area], ranked by their mean score (ties: component id), top keep_b,
         uid = 300000 + component id.

    python rea_t7.py --tag r11 --from-tag r10a --calls-from r10 --split val --thr 0.3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from common import REA, RUN, components, load_m0, names, read_jsonl, write_json
import items as I


def main(a):
    rows = []
    for name in names(a.split):
        old = I.Store(a.from_tag, a.split, name)
        m0 = load_m0('rea', a.split, name)
        z = np.load(REA / a.split / 'score' / (Path(name).stem + '.npz'))
        sc = z['change_score_before_filter'].astype(np.float32)
        lab, comps = components((sc > a.thr) & ~m0)
        B, store = [], {}
        for c in comps:
            if not a.min_area <= c['area'] <= a.max_area:
                continue
            cm = lab == c['id']
            B.append(dict(uid=300000 + c['id'], bbox=c['bbox'], area=c['area'], score=float(sc[cm].mean()),
                          source='rea_filtered'))
            store[300000 + c['id']] = cm
        B.sort(key=lambda it: (-it['score'], it['uid']))
        B = B[:a.keep_b]
        A = old.meta['A']
        for it in A:
            store[it['uid']] = old.mask(it['uid'])
        I.save(a.tag, a.split, name, A, B, {it['uid']: store[it['uid']] for it in A + B}, m0.shape)
        src = {r['uid']: r for r in read_jsonl(RUN / 'calls' / a.calls_from / a.split / (Path(name).stem + '.jsonl'))}
        dst = RUN / 'calls' / a.tag / a.split / (Path(name).stem + '.jsonl')
        have = {r['uid'] for r in read_jsonl(dst)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('a') as f:
            for it in A:
                if it['uid'] not in have and it['uid'] in src:
                    f.write(json.dumps(src[it['uid']]) + '\n')
        rows.append(dict(name=name, A=len(A), B=len(B), B_pool=len(comps)))
        print(rows[-1], flush=True)
    write_json(RUN / 'items' / a.tag / f'{a.split}_counts.json', dict(args=vars(a), rows=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--from-tag', default='r10a')
    p.add_argument('--calls-from', default='r10')
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--thr', type=float, default=0.3)
    p.add_argument('--min-area', type=int, default=20)
    p.add_argument('--max-area', type=int, default=20000)
    p.add_argument('--keep-b', type=int, default=50)
    main(p.parse_args())
