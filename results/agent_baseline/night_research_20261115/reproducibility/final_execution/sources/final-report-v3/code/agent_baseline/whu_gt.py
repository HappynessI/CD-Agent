"""Official WHU reference contract; labels and ignore masks never enter Agent tools."""
from pathlib import Path
import numpy as np
from PIL import Image

VERSION = 'whu_official_v3_20261114'
GT_DIR = 'gt_official_v3'
KINDS = {'any': 'c0', 'appeared': 'new', 'disappeared': 'demolished'}


def truth(t1, t2, official):
    a, b, raw = map(np.asarray, (t1, t2, official))
    if a.ndim != 2 or a.shape != b.shape or a.shape != raw.shape:
        raise ValueError('WHU semantic and official labels must have identical 2D shapes')
    change = raw > 0
    appeared = change & (a == 0) & (b == 1)
    disappeared = change & (a == 1) & (b == 0)
    ignore = change & ~(appeared | disappeared)
    return dict(c0=change, new=appeared, demolished=disappeared, ignore=ignore)


def read_raw(path):
    with Image.open(path) as im:
        return np.array(im)


def load_truth(dataset, name):
    return truth(*(read_raw(Path(dataset)/p/name) for p in ('slabel1','slabel2','label_unch0chi1')))


def paths(run, name, direction):
    root = Path(run).resolve()/'whu'/GT_DIR
    return root/KINDS[direction]/name, (root/'ignore'/name if direction != 'any' else None)


def update_task(row, run):
    row = dict(row)
    gt, ignore = paths(run, Path(row['id']).name, row.get('change_direction','any'))
    row.update(gt_mask=str(gt), gt_ignore_mask=str(ignore) if ignore else None)
    meta = dict(row.get('meta') or {})
    for key in ('legacy_gt_mask', 'legacy_manifest', 'data_version', 'review_status'):
        meta.pop(key, None)
    meta.update(whu_gt_version=VERSION, gt_definition='official change; semantic direction only inside official change',
                ignore_definition='official changed pixels lacking a reliable 0-to-1 or 1-to-0 direction',
                correction_authorized=True)
    row['meta'] = meta
    return row
