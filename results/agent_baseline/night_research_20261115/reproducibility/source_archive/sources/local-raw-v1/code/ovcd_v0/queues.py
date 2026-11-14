"""Build the per-image agent state from the M-C-I dump: stage-A queue and stage-B queue (T7).

Stage A items = 8-connected components of M0 (uid = component label).  Attributes, from the M0
SAM proposals that fall inside the component (>= 50% of the proposal):
  score   max DINOv2 change score (-cos) among them        (M-C-I comparator)
  bprob   max building probability at the proposal's own time  (SegEarth-OV identifier)
  bother  building probability at the other time for that proposal
  area    component area
Stage B items (T7, source "M-C-I near-threshold"): SAM proposals not in M0, at most 50% covered
by M0, area in [min_area, max_area], ranked by the soft M-C-I acceptance
  near = sigmoid((score - s0) / 0.05) * p_own * (1 - p_other),   s0 = cos(145 deg)
(the hard rule keeps a proposal iff score > s0 and p_own >= 0.5 > p_other), then greedy mask NMS
(drop IoU > 0.5 or >= 80% inside an earlier one); the top `keep_b` are stored (uid = 100000+index).

    python queues.py --tag r1 --split val
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np

from common import MCI, components, load_mask_png, names, write_json, RUN
import items as I

S0 = math.cos(math.radians(145))


def read_dump(split, name):
    z = np.load(MCI / split / (Path(name).stem + '.npz'), allow_pickle=False)
    h, w = map(int, z['image_shape'])
    masks = np.unpackbits(z['masks_packed'], axis=1, count=h * w, bitorder='little').reshape(-1, h, w).astype(bool)
    d = {k: z[k] for k in z.files if k != 'masks_packed'}
    return masks, d


def build(split, name, tag, min_area, max_area, keep_b, m0_dir=None):
    alt = m0_dir is not None  # M0 from another initializer (e.g. I-M-C); M-C-I proposals still feed A attributes and T7
    m0 = load_mask_png((Path(m0_dir) / split / 'm0' if alt else MCI / split / 'm0') / name)
    masks, d = read_dump(split, name)
    src = d['source_time'].astype(int) - 1  # dump: 1 = T1, 2 = T2 -> 0/1
    p = np.nan_to_num(np.stack([d['building_prob_t1'], d['building_prob_t2']], 1).astype(float))  # NaN: empty after resize
    p_own = p[np.arange(len(src)), src]
    p_other = p[np.arange(len(src)), 1 - src]
    score = d['dino_neg_cos'].astype(float)
    in_m0 = d['in_m0'].astype(bool)
    if not alt:
        union = masks[in_m0].any(0) if in_m0.any() else np.zeros(m0.shape, bool)
        assert np.array_equal(union, m0), 'M0 png differs from the union of in_m0 proposals'
    pool = np.ones(len(masks), bool) if alt else in_m0  # proposals that describe A components

    lab, comps = components(m0)
    A, store = [], {}
    for c in comps:
        cm = lab == c['id']
        idx = [i for i in np.flatnonzero(pool) if (masks[i] & cm).sum() >= 0.5 * masks[i].sum()]
        if not idx:  # proposal straddles components: fall back to any overlap
            idx = [i for i in np.flatnonzero(pool) if (masks[i] & cm).any()]
        if idx:
            j = max(idx, key=lambda i: p_own[i])
            A.append(dict(uid=c['id'], bbox=c['bbox'], area=c['area'], nprops=len(idx),
                          score=float(max(score[i] for i in idx)), bprob=float(p_own[j]),
                          bother=float(p_other[j])))
        else:  # no M-C-I evidence at all: lowest change score, so score_asc verifies it first
            A.append(dict(uid=c['id'], bbox=c['bbox'], area=c['area'], nprops=0, score=-1.0, bprob=0.0, bother=0.0))
        store[c['id']] = cm

    near = 1 / (1 + np.exp(-(score - S0) / 0.05)) * p_own * (1 - p_other)
    area = masks.reshape(len(masks), -1).sum(1)
    cover = (masks & m0).reshape(len(masks), -1).sum(1) / np.maximum(area, 1)
    ok = (np.ones_like(in_m0) if alt else ~in_m0) & (area >= min_area) & (area <= max_area) & (cover <= 0.5)
    B, chosen = [], []
    for i in np.argsort(-near):
        if not ok[i]:
            continue
        mi = masks[i]
        if any((mi & masks[j]).sum() > 0.5 * (mi | masks[j]).sum() or (mi & masks[j]).sum() >= 0.8 * area[i]
               for j in chosen):
            continue
        chosen.append(i)
        ys, xs = np.nonzero(mi)
        B.append(dict(uid=100000 + int(i), bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                      area=int(area[i]), near=float(near[i]), score=float(score[i]), p_own=float(p_own[i]),
                      p_other=float(p_other[i]), source_time=int(src[i]) + 1, m0_cover=float(cover[i]),
                      pass_compare=bool(d['pass_compare'][i]), pass_identify=bool(d['pass_identify'][i]),
                      mci_m0=bool(in_m0[i])))
        store[100000 + int(i)] = mi
        if len(B) >= keep_b:
            break
    I.save(tag, split, name, A, B, store, m0.shape)
    return len(A), len(B), int(ok.sum())


def main(a):
    rows = []
    for name in names(a.split)[a.shard::a.shards]:
        nA, nB, npool = build(a.split, name, a.tag, a.min_area, a.max_area, a.keep_b, a.m0_dir)
        rows.append(dict(name=name, A=nA, B=nB, B_pool=npool))
        print(name, nA, nB, npool, flush=True)
    write_json(RUN / 'items' / a.tag / f'{a.split}_counts_{a.shard}.json',
               dict(args=vars(a), rows=rows))


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--tag', default='r1')
    pa.add_argument('--split', choices=['val', 'test'], required=True)
    pa.add_argument('--min-area', type=int, default=50)
    pa.add_argument('--max-area', type=int, default=20000)
    pa.add_argument('--keep-b', type=int, default=50)
    pa.add_argument('--m0-dir', help='alternative M0 root with <split>/m0/<name> (default: M-C-I)')
    pa.add_argument('--shard', type=int, default=0)
    pa.add_argument('--shards', type=int, default=1)
    main(pa.parse_args())
