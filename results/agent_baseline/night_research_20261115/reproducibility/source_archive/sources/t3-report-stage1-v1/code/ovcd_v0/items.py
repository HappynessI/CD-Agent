"""Per-image item store: the stage-A queue (connected components of M0) and the stage-B queue
(T7 suspicious regions).  One JSON (attributes) + one NPZ (masks cropped to their bbox) per image.

items/<round>/<split>/<stem>.json : {"name", "A": [item...], "B": [item...]}
    item = {"uid", "bbox": [x0,y0,x1,y1], "area", ...source attributes}
items/<round>/<split>/<stem>.npz  : off (n+1,), bits (packed bool crops), uids (n,)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from common import RUN, read_json, write_json


def paths(tag, split, name):
    stem = Path(name).stem
    base = RUN / 'items' / tag / split
    return base / f'{stem}.json', base / f'{stem}.npz'


def save(tag, split, name, A, B, masks, shape):
    """masks: {uid: full-size bool mask}; uids must be unique across A and B."""
    js, nz = paths(tag, split, name)
    uids = [it['uid'] for it in A + B]
    assert len(set(uids)) == len(uids)
    crops = []
    for it in A + B:
        x0, y0, x1, y1 = it['bbox']
        crops.append(masks[it['uid']][y0:y1, x0:x1].ravel())
    off = np.cumsum([0] + [len(c) for c in crops])
    bits = np.packbits(np.concatenate(crops)) if crops else np.zeros(0, np.uint8)
    nz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, off=off, bits=bits, uids=np.array(uids, int), shape=np.array(shape))
    write_json(js, dict(name=name, A=A, B=B))


class Store:
    def __init__(self, tag, split, name):
        js, nz = paths(tag, split, name)
        self.meta = read_json(js)
        z = np.load(nz)
        self.off, self.uids, self.shape = z['off'], z['uids'].tolist(), tuple(z['shape'])
        self.flat = np.unpackbits(z['bits'])[: self.off[-1]].astype(bool) if len(z['bits']) else None
        self.by_uid = {it['uid']: it for it in self.meta['A'] + self.meta['B']}

    def mask(self, uid):
        i = self.uids.index(uid)
        x0, y0, x1, y1 = self.by_uid[uid]['bbox']
        m = np.zeros(self.shape, bool)
        m[y0:y1, x0:x1] = self.flat[self.off[i]:self.off[i + 1]].reshape(y1 - y0, x1 - x0)
        return m
