"""Validation diagnostics of the MLLM answers (no thresholds involved).

A: label "delete" = component precision < F1(M0)/2 (the deletions that raise cumulative F1);
   AUC of (1 - P(yes)) for "delete", pixel-weighted AUC, and the share answering each pseudo type.
B: label "useful" = candidate covers >= 30% of a whole-missed GT object (IoU < 0.1 with M0, as in
   common.object_counts) or >= 50% of it lies on GT;
   AUC of P(yes); T5 box quality = IoU of the SAM union from the box with the GT inside the crop,
   against the candidate mask's own IoU.

    python diag_mllm.py --tag r1 --split val
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import load_m0, MCI, RUN, load_gt, load_mask_png, names, read_json, read_jsonl, write_json
import items as I
import render


def auc(score, y, w=None):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    w = np.ones_like(score) if w is None else np.asarray(w, float)
    order = np.argsort(score)
    s, yy, ww = score[order], y[order], w[order]
    # weighted Mann-Whitney with ties
    pos_w, neg_w = ww[yy].sum(), ww[~yy].sum()
    if pos_w == 0 or neg_w == 0:
        return None
    total, cum_neg = 0.0, 0.0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        pw, nw = ww[i:j][yy[i:j]].sum(), ww[i:j][~yy[i:j]].sum()
        total += pw * (cum_neg + 0.5 * nw)
        cum_neg += nw
        i = j
    return float(total / (pos_w * neg_w))


def main(a):
    from sam_tools import Prompter
    pr = Prompter('cpu')
    f0 = read_json(RUN / 'm1' / a.tag / f'{a.split}.json')['summary']['f1']
    comps = {(r['name'], r['comp']): r for r in read_json(RUN / 'm1' / a.tag / f'{a.split}_components.json')}
    A, B = [], []
    for name in names(a.split):
        calls = {r['uid']: r for r in read_jsonl(RUN / 'calls' / a.tag / a.split / (Path(name).stem + '.jsonl'))}
        st = I.Store(a.tag, a.split, name)
        gt = load_gt(a.split, name)
        m0 = load_m0(a.m0, a.split, name)
        for it in st.meta['A']:
            c, r = calls.get(it['uid']), comps[(name, it['uid'])]
            if c:
                A.append(dict(p=c['verify']['p_yes'], delete=r['precision'] < f0 / 2, area=r['area'],
                              type=c['verify']['type'], fp=r['area'] - r['tp']))
        from scipy import ndimage as ndi
        gl, _ = ndi.label(gt, np.ones((3, 3), bool))
        pl, _ = ndi.label(m0, np.ones((3, 3), bool))
        missed = set()
        for i in range(1, gl.max() + 1):
            obj = gl == i
            t = np.unique(pl[obj]); t = t[t > 0]
            other = np.isin(pl, t)
            if (obj & other).sum() / max((obj | other).sum(), 1) < 0.1:
                missed.add(i)
        for it in st.meta['B'][:a.b_max]:
            c = calls.get(it['uid'])
            if not c:
                continue
            m = st.mask(it['uid'])
            on = (m & gt).sum()
            useful = on >= 0.5 * m.sum()
            ids = np.unique(gl[m & gt]); ids = ids[ids > 0]
            for i in ids:
                obj = gl == i
                if i in missed and (m & obj).sum() >= 0.3 * obj.sum():
                    useful = True
            row = dict(p=c['verify']['p_yes'], useful=bool(useful), type=c['verify']['type'])
            x0, y0, x1, y1 = c['crop']
            g = gt[y0:y1, x0:x1]
            row['cand_iou'] = float((m[y0:y1, x0:x1] & g).sum() / max((m[y0:y1, x0:x1] | g).sum(), 1))
            if c.get('box') and c['box']['box'] is not None:
                fb = render.crop_to_full(c['box']['box'], c['crop'])
                u = pr.box(a.split, name, 0, fb)[0] | pr.box(a.split, name, 1, fb)[0]
                row['box_iou'] = float((u[y0:y1, x0:x1] & g).sum() / max((u[y0:y1, x0:x1] | g).sum(), 1))
            B.append(row)
    out = dict(
        A=dict(n=len(A), delete_share=float(np.mean([r['delete'] for r in A])),
               auc_delete=auc([1 - r['p'] for r in A], [r['delete'] for r in A]),
               auc_delete_fp_weighted=auc([1 - r['p'] for r in A], [r['delete'] for r in A], [r['area'] for r in A]),
               types_delete=Counter(r['type'] for r in A if r['delete']).most_common(),
               types_keep=Counter(r['type'] for r in A if not r['delete']).most_common()),
        B=dict(n=len(B), useful_share=float(np.mean([r['useful'] for r in B])),
               auc_useful=auc([r['p'] for r in B], [r['useful'] for r in B]),
               useful_yes=float(np.mean([r['p'] >= 0.5 for r in B if r['useful']])),
               notuseful_yes=float(np.mean([r['p'] >= 0.5 for r in B if not r['useful']])),
               box_valid=float(np.mean(['box_iou' in r for r in B])),
               useful_cand_iou=float(np.mean([r['cand_iou'] for r in B if r['useful']])),
               useful_box_iou=float(np.mean([r.get('box_iou', 0) for r in B if r['useful']])),
               useful_box_better=float(np.mean([r.get('box_iou', 0) > r['cand_iou'] for r in B if r['useful']]))))
    write_json(RUN / 'diag' / a.tag / f'{a.split}_mllm.json', dict(summary=out, A=A, B=B))
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='r1')
    p.add_argument('--m0', default='mci', choices=['mci', 'imc', 'rea'])
    p.add_argument('--split', default='val')
    p.add_argument('--b-max', type=int, default=20)
    main(p.parse_args())
