"""SECOND semantic encoding: 0=unlabelled unchanged; changed pixels use classes 1..6.

0 is NOT a land-cover class. Class 5 is building. Building->building, including
replacement at one location, is excluded. Read slabel1/slabel2, not RGB labels.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

DATASET = Path('/guisongxia01/pangchao/wangyihan/datasets/lim1/SECOND')
VALIDATION = Path('/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/exp6/second_cv/verifier_val.txt')
ENCODING = {'non-change': 0, 'low vegetation': 1, 'N.V.G surface': 2, 'tree': 3,
            'water': 4, 'building': 5, 'playground': 6}
DIRECTION_GT = {'appeared': 'new', 'disappeared': 'demolished', 'any': 'c0'}
SCOPE = 'building_rebuild_excluded'


def check_encoding(dataset):
    actual = json.loads((Path(dataset) / 'cls_gray.json').read_text())
    if actual != ENCODING:
        raise ValueError(f'Unexpected semantic encoding: {actual}')
    return actual


def building_truth(t1, t2):
    t1, t2 = np.asarray(t1), np.asarray(t2)
    if t1.ndim != 2 or t1.shape != t2.shape:
        raise ValueError('Semantic labels must be matching 2D arrays')
    if not (np.isin(t1, range(7)).all() and np.isin(t2, range(7)).all()):
        raise ValueError('Semantic labels must use grayscale class IDs 0..6')
    if np.any((t1 == 0) != (t2 == 0)):
        raise ValueError('Unchanged/unlabelled support differs between dates')
    changed = (t1 != t2) & (t1 != 0) & (t2 != 0)
    new = changed & (t2 == 5) & (t1 != 5)
    demolished = changed & (t1 == 5) & (t2 != 5)
    return dict(new=new, demolished=demolished, c0=new | demolished)


def load_labels(dataset, name):
    return tuple(np.asarray(Image.open(Path(dataset) / part / name)) for part in ('slabel1', 'slabel2'))


def validation_names(filename=VALIDATION):
    seq = Path(filename).read_text().split()
    if len(seq) != len(set(seq)) or not seq:
        raise ValueError('Validation inventory must be nonempty and unique')
    if any(Path(n).name != n or not n.startswith('train_') or not n.endswith('.png') for n in seq):
        raise ValueError('Validation inventory must contain training-image filenames')
    return sorted(seq)


def pixel_counts(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if pred.shape != truth.shape:
        raise ValueError('Prediction and truth shapes differ')
    return [int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())]
