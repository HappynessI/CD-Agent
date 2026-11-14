"""Reference loading shared by online recorder and offline evaluators, not tools."""
from pathlib import Path
import numpy as np
from PIL import Image


def load_reference(path, ignore_path=None):
    path = Path(path)
    if path.parent.parent.name == 'gt' and path.parent.parent.parent.name == 'whu':
        raise ValueError('Retired WHU XOR reference; use gt_official_v3')
    # Canonical WHU directions always carry their shared uncertainty mask,
    # including direct-path scorers that do not consume a task manifest.
    if path.parent.parent.name == 'gt_official_v3' and path.parent.name in ('new','demolished'):
        canonical = path.parent.parent/'ignore'/path.name
        if ignore_path is not None and Path(ignore_path).resolve() != canonical.resolve():
            raise ValueError('WHU direction requires its canonical ignore mask')
        ignore_path = canonical
    with Image.open(path) as im:
        gt = np.array(im) > 0
    valid = np.ones_like(gt, bool)
    if ignore_path is not None:
        with Image.open(ignore_path) as im:
            ignore = np.array(im) > 0
        if ignore.shape != gt.shape:
            raise ValueError('GT and ignore shapes differ')
        valid = ~ignore
    return gt & valid, valid


def pixel_counts(pred, gt, valid=None):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError('Prediction and reference shapes differ')
    if valid is not None:
        pred, gt = pred & valid, gt & valid
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))
