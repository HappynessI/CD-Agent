"""Instance-set primitives shared by the tools and the fixed routes.

An instance set is stored sparsely: a list of dicts with `idx` (sorted flat pixel indices), `bbox`
(xyxy, exclusive end), `area`, `score`.  Each 1024x1024 bool mask would cost 1 MB, and an image holds
100+ instances across several sets for up to 32 concurrent sessions, so dense storage does not fit.
"""
from __future__ import annotations

import numpy as np


def from_dense(masks, scores, shape=None):
    out = []
    for number, (m, s) in enumerate(zip(masks, scores), 1):
        if not m.any():
            continue
        h, w = m.shape
        idx = np.flatnonzero(m.ravel()).astype(np.int32)
        ys, xs = idx // w, idx % w
        out.append(dict(id=number, idx=idx, area=int(idx.size), score=float(s),
                        bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return out


def to_mask(insts, shape):
    m = np.zeros(int(shape[0]) * int(shape[1]), bool)
    for it in insts:
        m[it['idx']] = True
    return m.reshape(shape)


def _boxes_touch(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def overlaps(a, b):
    """All intersecting pairs: (i, j, intersection, area_i, area_j)."""
    out = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if not _boxes_touch(x['bbox'], y['bbox']):
                continue
            inter = int(np.intersect1d(x['idx'], y['idx'], assume_unique=True).size)
            if inter:
                out.append((i, j, inter, x['area'], y['area']))
    return out


def match(a, b, thr=0.25, mode='overlap'):
    """Split two instance sets into disappeared (a only), appeared (b only) and unchanged (of b).

    mode='overlap' is OmniOVCD overlap_presence: an instance is present in the other time when some
    instance there covers at least `thr` of its own area, judged per direction.  mode='iou' needs a
    pair with IoU >= thr, which makes the relation symmetric.
    """
    hit_a, hit_b = set(), set()
    for i, j, inter, area_i, area_j in overlaps(a, b):
        if mode == 'overlap':
            if inter / area_i >= thr:
                hit_a.add(i)
            if inter / area_j >= thr:
                hit_b.add(j)
        elif mode == 'iou':
            if inter / (area_i + area_j - inter) >= thr:
                hit_a.add(i)
                hit_b.add(j)
        else:
            raise ValueError(f'unknown mode {mode}')
    disappeared = [x for i, x in enumerate(a) if i not in hit_a]
    appeared = [y for j, y in enumerate(b) if j not in hit_b]
    unchanged = [y for j, y in enumerate(b) if j in hit_b]
    return appeared, disappeared, unchanged


def union_components(insts, shape):
    """Optional matching basis; component instances intentionally have no score."""
    from common import components
    lab, rows = components(to_mask(insts, shape))
    return [dict(c, idx=np.flatnonzero(lab.ravel() == c['id']).astype(np.int32)) for c in rows]
