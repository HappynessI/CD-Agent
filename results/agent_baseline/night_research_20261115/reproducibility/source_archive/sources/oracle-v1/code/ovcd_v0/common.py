"""OVCD-v0 shared paths, data access, connected components and metrics.

LEVIR-CD 1024x1024 pairs; val (64) is used for every choice, test (128) is scored once per round.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

WORKSPACE = Path('/guisongxia01/pangchao/wangyihan')
REPO = Path(__file__).resolve().parents[2]
LEVIR = WORKSPACE / 'datasets/LEVIR-CD'
RUN = Path(os.environ.get('OVCD_RUN', WORKSPACE / 'wyh/exp6_staging/ovcd_v0_run'))
MCI = Path(os.environ.get('OVCD_MCI', WORKSPACE / 'wyh/exp6_staging/batch56_ovcd_de/run/mci'))
IMC = WORKSPACE / 'wyh/exp6_staging/batch56_ovcd_de/run/imc'
REA = WORKSPACE / 'wyh/exp6_staging/batch66_ovcd_reaovcd/run/rea'
M0_DIRS = {'mci': MCI, 'imc': IMC, 'rea': REA}  # initializers; each has <split>/m0/<name>.png
DYNAMIC_EARTH = WORKSPACE / 'wyh/exp6_staging/batch56_ovcd_de/DynamicEarth'
SAM_CKPT = WORKSPACE / 'models/SAM/sam_vit_h_4b8939.pth'
QWEN = WORKSPACE / 'models/Qwen3.5-4B'
FIGS = REPO / 'figures/ovcd_v0'
SPLITS = ('val', 'test')
EIGHT = np.ones((3, 3), bool)


def names(split):
    return sorted(p.name for p in (LEVIR / f'{split}_original/A').glob('*.png'))


def load_rgb(split, name):
    from PIL import Image
    return [np.asarray(Image.open(LEVIR / f'{split}_original/{v}/{name}').convert('RGB')) for v in 'AB']


def load_gt(split, name):
    from PIL import Image
    return np.asarray(Image.open(LEVIR / f'{split}_original/label/{name}').convert('L')) > 127


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def components(mask, min_area=1):
    """8-connected components: (label image, list of dicts with id/area/bbox xyxy)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask, EIGHT)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        if area < min_area:
            continue
        out.append(dict(id=i, area=area, bbox=[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]))
    return lab, out


# ---------------- metrics ----------------

def confusion(pred, gt):
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))


def scores(c):
    tp, fp, fn = c['tp'], c['fp'], c['fn']
    eps = 1e-7  # same constant as DynamicEarth eval/evaluate.py
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return dict(f1=2 * p * r / (p + r + eps), iou=tp / (tp + fp + fn + eps), precision=p, recall=r)


def image_f1(pred, gt):
    """Per-image F1; an image with no GT and no prediction scores 1."""
    c = confusion(pred, gt)
    if c['tp'] + c['fp'] + c['fn'] == 0:
        return 1.0
    return 2 * c['tp'] / (2 * c['tp'] + c['fp'] + c['fn'])


def object_counts(pred, gt, thr=0.1):
    """Whole-object misses / false alarms: a GT (pred) component whose IoU with the
    prediction (GT) is < thr.  IoU of a component C against mask M is |C & M| / |C | (M restricted
    to the components of M touching C)|, i.e. the object-to-overlapping-objects IoU."""
    from scipy import ndimage as ndi
    out = {}
    for key, a, b in (('gt', gt, pred), ('pred', pred, gt)):
        la, na = ndi.label(a, EIGHT)
        lb, _ = ndi.label(b, EIGHT)
        bad = 0
        for i, sl in enumerate(ndi.find_objects(la), 1):
            comp = la == i
            touching = np.unique(lb[comp])
            touching = touching[touching > 0]
            other = np.isin(lb, touching) if len(touching) else np.zeros_like(b)
            inter = (comp & other).sum()
            union = (comp | other).sum()
            if inter / max(union, 1) < thr:
                bad += 1
        out[f'{key}_objects'] = int(na)
        out[f'{key}_bad'] = bad
    return dict(gt_objects=out['gt_objects'], missed=out['gt_bad'],
                pred_objects=out['pred_objects'], false_alarms=out['pred_bad'])


def evaluate(preds, split, gts=None):
    """preds: {name: bool mask}.  Cumulative F1^c/IoU^c/P/R, per-image mean F1, object rates."""
    tot = dict(tp=0, fp=0, fn=0)
    obj = dict(gt_objects=0, missed=0, pred_objects=0, false_alarms=0)
    per = {}
    for name, pred in preds.items():
        gt = gts[name] if gts is not None else load_gt(split, name)
        c = confusion(pred, gt)
        for k in tot:
            tot[k] += c[k]
        o = object_counts(pred, gt)
        for k in obj:
            obj[k] += o[k]
        per[name] = image_f1(pred, gt)
    res = scores(tot)
    res.update(tot)
    res.update(obj)
    res['miss_rate'] = obj['missed'] / max(obj['gt_objects'], 1)
    res['false_alarm_rate'] = obj['false_alarms'] / max(obj['pred_objects'], 1)
    res['mean_image_f1'] = float(np.mean(list(per.values())))
    res['images'] = len(preds)
    return res, per


def pack(mask):
    return dict(shape=list(mask.shape), bits=np.packbits(mask.ravel()))


def save_mask_png(path, mask):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def load_m0(kind, split, name):
    return load_mask_png(M0_DIRS[kind] / split / 'm0' / name)


def load_mask_png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('L')) > 127
