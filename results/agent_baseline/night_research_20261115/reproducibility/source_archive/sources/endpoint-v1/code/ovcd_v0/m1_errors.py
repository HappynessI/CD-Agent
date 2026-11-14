"""M1: error analysis of an initial mask M0 (val for decisions, test for description).

Outputs RUN/m1/<tag>/<split>.json with
  - pixel metrics, whole-object miss / false-alarm rates (IoU < 0.1)
  - FP pixels split into whole false objects vs. extra pixels on hit objects
  - FN pixels split into the exp8s5 cells with base = M0:
      5 tiny (object or miss blob < 20 px), 3 object covered >= 10% by M0, 1 blob within 1 px
      (chessboard) of M0, 2 isolated (results/miss_share.md cell 2)
  - oracle ceilings: A* deletes every M0 component whose precision is below F1/2 (exactly the
    deletions that raise cumulative F1), B* adds every GT object M0 misses (IoU < 0.1)
  - per-component table (for stage-A ranking): precision, area, and the stage-A item attributes

    python m1_errors.py --tag r1 --split val
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from common import load_m0, EIGHT, MCI, RUN, evaluate, load_gt, load_mask_png, names, scores, write_json
import items as I


def cells(gt, base):
    gl, ng = ndi.label(gt, EIGHT)
    ga = np.bincount(gl.ravel(), minlength=ng + 1)
    gb = np.bincount(gl[base], minlength=ng + 1)
    chess = ndi.distance_transform_cdt(~base, metric='chessboard') if base.any() else None
    cell = np.zeros(gt.shape, np.uint8)
    for target, sl in enumerate(ndi.find_objects(gl), 1):
        if sl is None:
            continue
        cc, _ = ndi.label((gl[sl] == target) & ~base[sl], EIGHT)
        for k, box in enumerate(ndi.find_objects(cc), 1):
            m = cc[box] == k
            ys = slice(sl[0].start + box[0].start, sl[0].start + box[0].stop)
            xs = slice(sl[1].start + box[1].start, sl[1].start + box[1].stop)
            dist = int(chess[ys, xs][m].min()) if chess is not None else None
            g = (5 if min(ga[target], m.sum()) < 20 else 3 if gb[target] / ga[target] >= .1
                 else 1 if dist is not None and dist <= 1 else 2)
            cell[ys, xs][m] = g
    assert np.array_equal(cell > 0, gt & ~base)
    return cell


def best_deletion(rows, tp, fp, fn):
    best, n_best, dtp, dfp = scores(dict(tp=tp, fp=fp, fn=fn)), 0, 0, 0
    for i, r in enumerate(sorted(rows, key=lambda r: r['precision']), 1):
        dtp += r['tp']
        dfp += r['area'] - r['tp']
        s = scores(dict(tp=tp - dtp, fp=fp - dfp, fn=fn + dtp))
        if s['f1'] > best['f1']:
            best, n_best = s, i
    return best, n_best


def main(a):
    preds, gts, comp_rows = {}, {}, []
    fp_whole = fp_extra = 0
    fn_cells = {k: 0 for k in (1, 2, 3, 5)}
    for name in names(a.split):
        m0 = load_m0(a.m0, a.split, name)
        gt = load_gt(a.split, name)
        preds[name], gts[name] = m0, gt
        lab, n = ndi.label(m0, EIGHT)
        gl, _ = ndi.label(gt, EIGHT)
        store = None
        if a.tag and I.paths(a.tag, a.split, name)[0].exists():
            store = I.Store(a.tag, a.split, name)
        attrs = {}
        if store is not None:
            for it in store.meta['A']:
                attrs[it['uid']] = it
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            comp = lab[sl] == i
            tp = int((comp & gt[sl]).sum())
            area = int(comp.sum())
            touching = np.unique(gl[sl][comp]); touching = touching[touching > 0]
            other = np.isin(gl, touching)
            full = lab == i
            iou = (full & other).sum() / max((full | other).sum(), 1)
            if iou < 0.1:
                fp_whole += area - tp
            else:
                fp_extra += area - tp
            row = dict(name=name, comp=i, area=area, tp=tp, precision=tp / area, iou=float(iou))
            if i in attrs:  # stage-A items use the M0 component label as uid
                row.update({k: v for k, v in attrs[i].items() if k not in ('uid', 'bbox', 'area')})
            comp_rows.append(row)
        c = cells(gt, m0)
        for k in fn_cells:
            fn_cells[k] += int((c == k).sum())
    res, per = evaluate(preds, a.split, gts)
    fp, fn = res['fp'], res['fn']
    res['fp_share_of_errors'] = fp / max(fp + fn, 1)
    res['fp_pixels'] = dict(whole_false_objects=fp_whole, extra_on_hit_objects=fp_extra)
    res['fn_cells'] = fn_cells
    res['fn_cells_share'] = {k: v / max(fn, 1) for k, v in fn_cells.items()}
    # oracle A*: delete the best precision-sorted prefix of components (the F1/2 rule applied
    # with the updated F1 after each deletion; Codex review batch58 #3)
    res['oracle_A'], res['oracle_A_deleted'] = best_deletion(comp_rows, res['tp'], fp, fn)
    # oracle B*: add whole missed GT objects
    add = 0
    for name in preds:
        gl, _ = ndi.label(gts[name], EIGHT)
        pl, _ = ndi.label(preds[name], EIGHT)
        for i, sl in enumerate(ndi.find_objects(gl), 1):
            obj = gl == i
            t = np.unique(pl[obj]); t = t[t > 0]
            other = np.isin(pl, t)
            if (obj & other).sum() / max((obj | other).sum(), 1) < 0.1:
                add += int((obj & ~preds[name]).sum())
    res['oracle_B'] = scores(dict(tp=res['tp'] + add, fp=fp, fn=fn - add))
    res['components'] = len(comp_rows)
    res['components_per_image'] = len(comp_rows) / len(preds)
    # attribute AUCs (FA component = IoU < 0.1) for choosing the stage-A order
    aucs = {}
    keys = [k for k in (comp_rows[0] if comp_rows else {}) if k not in
            ('name', 'comp', 'tp', 'precision', 'iou', 'kind')]
    y = np.array([r['iou'] < 0.1 for r in comp_rows])
    for k in keys:
        x = np.array([r.get(k, np.nan) for r in comp_rows], float)
        ok = np.isfinite(x)
        if ok.sum() and 0 < y[ok].sum() < ok.sum():
            from scipy.stats import rankdata
            r = rankdata(x[ok])
            npos = y[ok].sum(); nneg = ok.sum() - npos
            aucs[k] = float((r[y[ok]].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    res['fa_auc_by_attribute'] = aucs  # AUC of "attribute high => false alarm"
    write_json(RUN / 'm1' / (a.tag or 'm0') / f'{a.split}.json', dict(summary=res, per_image_f1=per))
    write_json(RUN / 'm1' / (a.tag or 'm0') / f'{a.split}_components.json', comp_rows)
    print({k: v for k, v in res.items() if not isinstance(v, (list,))})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='r1')
    p.add_argument('--m0', default='mci', choices=['mci', 'imc', 'rea'])
    p.add_argument('--split', choices=['val', 'test'], required=True)
    main(p.parse_args())
