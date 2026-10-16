"""Object-level matching between a predicted change mask and the ground truth.

Definitions (doc/change_dection_agent.md, experiment 1):
- objects are 8-connected components of the GT and of the prediction;
- a GT object is a hit if its best IoU with any predicted object is >= 0.5, partial if
  0.1 <= best IoU < 0.5, and a whole-object miss if best IoU < 0.1 (including no overlap);
- a predicted object is a whole-object false positive if its IoU with every GT object is < 0.1.
For semantic-change datasets (SECOND) GT objects are 8-connected components within each
(t1 class, t2 class) transition, so adjacent changes of different types are separate objects
(see gt_label). Error pixels are attributed by the object that owns them: an FN pixel by its GT object's
class, an FP pixel by its predicted object's class (>=0.5 boundary, 0.1-0.5 partial,
<0.1 whole-object FP). Thus boundary error = FP+FN pixels of hit objects.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCTURE = np.ones((3, 3), bool)
HIT, PARTIAL_LOW = 0.5, 0.1


def label(mask: np.ndarray):
    return ndimage.label(mask, STRUCTURE)


def gt_label(dataset, name, gt):
    """GT object labels: plain 8-connectivity, or split by class transition when available.

    Returns (labels, count, transitions) where transitions maps object id -> [c1, c2] or is empty.
    """
    from common import load_classes

    classes = load_classes(dataset, name)
    if classes is None:
        gl, ng = label(gt)
        return gl, ng, {}
    c1, c2 = (np.asarray(c, np.int64) for c in classes)
    pair = np.where(gt, c1 * 16 + c2, -1)
    gl = np.zeros(gt.shape, np.int32)
    transitions, ng = {}, 0
    for key in np.unique(pair[pair >= 0]):
        part, n = label(pair == key)
        gl[part > 0] = part[part > 0] + ng
        for k in range(ng + 1, ng + n + 1):
            transitions[k] = [int(key // 16), int(key % 16)]
        ng += n
    return gl, ng, transitions


def classify(iou: float) -> str:
    return "hit" if iou >= HIT else "partial" if iou >= PARTIAL_LOW else "miss"


def match(gt: np.ndarray, pred: np.ndarray, gt_labels=None):
    """Return (gt_labels, pred_labels, gt_objects, pred_objects, error_pixels).

    gt_labels: optional (labels, count, transitions) from gt_label(); default 8-connectivity.
    """
    gl, ng, transitions = gt_labels if gt_labels is not None else (*label(gt), {})
    pl, npr = label(pred)
    g_area = np.bincount(gl.ravel(), minlength=ng + 1)
    p_area = np.bincount(pl.ravel(), minlength=npr + 1)
    both = (gl > 0) & (pl > 0)
    pairs, inter = np.unique(gl[both].astype(np.int64) * (npr + 1) + pl[both], return_counts=True)
    g_best = np.zeros(ng + 1)
    g_best_id = np.zeros(ng + 1, np.int64)
    p_best = np.zeros(npr + 1)
    p_best_id = np.zeros(npr + 1, np.int64)
    for key, count in zip(pairs, inter):
        g, p = divmod(int(key), npr + 1)
        iou = count / (g_area[g] + p_area[p] - count)
        if iou > g_best[g]:
            g_best[g], g_best_id[g] = iou, p
        if iou > p_best[p]:
            p_best[p], p_best_id[p] = iou, g
    covered = ndimage.sum_labels(pred, gl, np.arange(1, ng + 1)) if ng else np.zeros(0)
    g_slices = ndimage.find_objects(gl)
    p_slices = ndimage.find_objects(pl)
    gt_objects = []
    for g in range(1, ng + 1):
        sl = g_slices[g - 1]
        gt_objects.append({
            "id": g, "area": int(g_area[g]), "bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
            "best_iou": float(g_best[g]), "best_pred": int(g_best_id[g]), "class": classify(g_best[g]),
            "coverage": float(covered[g - 1] / g_area[g]), "any_overlap": bool(covered[g - 1] > 0),
            **({"transition": transitions[g]} if transitions else {}),
        })
    pred_objects = []
    for p in range(1, npr + 1):
        sl = p_slices[p - 1]
        pclass = classify(p_best[p])
        pred_objects.append({
            "id": p, "area": int(p_area[p]), "bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
            "best_iou": float(p_best[p]), "best_gt": int(p_best_id[p]),
            "class": {"hit": "matched", "partial": "partial", "miss": "fp"}[pclass],
        })
    # Pixel attribution.
    g_class = np.array(["none"] + [o["class"] for o in gt_objects])
    p_class = np.array(["none"] + [o["class"] for o in pred_objects])
    fn = gt & ~pred
    fp = pred & ~gt
    fn_cls = g_class[gl[fn]]
    fp_cls = p_class[pl[fp]]
    errors = {
        "miss": int((fn_cls == "miss").sum()),
        "partial": int((fn_cls == "partial").sum() + (fp_cls == "partial").sum()),
        "fp": int((fp_cls == "fp").sum()),
        "boundary": int((fn_cls == "hit").sum() + (fp_cls == "matched").sum()),
        "boundary_fn": int((fn_cls == "hit").sum()), "boundary_fp": int((fp_cls == "matched").sum()),
        "partial_fn": int((fn_cls == "partial").sum()), "partial_fp": int((fp_cls == "partial").sum()),
    }
    return gl, pl, gt_objects, pred_objects, errors
