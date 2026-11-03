"""What limits F1 on LEVIR val: error make-up, oracle ceilings of the current tools, and run agreement. No model calls.

Oracles use labels, so they measure ceilings only and never reach an agent.
    python levir_analysis.py --out analysis.json --runs T5.1=cfgA,cfgB,cfgC --runs T1.1xC0=cfgA,cfgB,cfgC
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'agent_baseline'
    import agent_baseline

import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from common import EIGHT, components, confusion, load_gt, load_mask_png, names, object_counts, scores, write_json
import sam3_t7
from . import instances as I

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
SPLIT = 'val'


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def split_fp_fn(pred, gt):
    """FP pixels in components that barely touch GT (whole false) vs spill around true objects; FN likewise."""
    out = {}
    for key, a, b in (('fp', pred, gt), ('fn', gt, pred)):
        lab, comps = components(a)
        whole = part = 0
        for c in comps:
            m = lab == c['id']
            overlap = (m & b).sum() / c['area']
            miss = int((m & ~b).sum())
            if overlap < 0.1:
                whole += miss
            else:
                part += miss
        out[key + '_whole'], out[key + '_part'] = whole, part
    return out


def oracle_masks(imc, gt, low):
    """Keep/remove/add at the granularity the tools offer."""
    lab, comps = components(imc)
    delete = imc.copy()
    for c in comps:
        m = lab == c['id']
        if (m & gt).sum() / c['area'] < 0.5:
            delete[m] = False
    added = delete.copy()
    for rows in low:
        for it in rows:
            mask = np.zeros(gt.size, bool)
            mask[it['idx']] = True
            mask = mask.reshape(gt.shape)
            if (mask & gt).sum() / it['area'] >= 0.5:
                added |= mask
    return delete, added


def work(args):
    name, runs = args
    gt = load_gt(SPLIT, name)
    imc = load_mask_png(RUN / 'init' / 'sam3_imc' / SPLIT / name)
    inst = sam3_t7.read(SPLIT, name, 512, conf=0.1)
    low = [I.from_dense(m, s) for m, s in inst]
    masks = {'imc': imc}
    masks['oracle_delete'], masks['oracle_delete_add'] = oracle_masks(imc, gt, low)
    for r in (1, 2, 3):
        masks[f'imc_open{r}'] = ndi.binary_opening(imc, disk(r))
        masks[f'imc_erode{r}'] = ndi.binary_erosion(imc, disk(r))
        masks[f'oracle_delete_erode{r}'] = ndi.binary_erosion(masks['oracle_delete'], disk(r))
    row = dict(name=name, gt_px=int(gt.sum()), counts={k: confusion(v, gt) for k, v in masks.items()},
               objects={k: object_counts(masks[k], gt) for k in ('imc', 'oracle_delete', 'oracle_delete_add')},
               split={'imc': split_fp_fn(imc, gt)}, runs={})
    for task, cfgs in runs.items():
        preds = [load_mask_png(RUN / c / task / SPLIT / Path(name).stem / json.loads((RUN / c / task / SPLIT / Path(name).stem / 'final.json').read_text())['mask_path']) for c in cfgs]
        vote = np.sum(preds, axis=0) >= (len(preds) // 2 + 1)
        row['runs'][task] = dict(vote=confusion(vote, gt), each=[confusion(p, gt) for p in preds],
                                 split=[split_fp_fn(p, gt) for p in preds],
                                 agree=float((np.sum(preds, axis=0) % len(preds) == 0).mean()))
    return row


def f1(rows, get):
    tot = {k: sum(get(r)[k] for r in rows) for k in ('tp', 'fp', 'fn')}
    s = scores(tot)
    return dict(f1=round(100 * s['f1'], 2), p=round(100 * s['precision'], 2), r=round(100 * s['recall'], 2), **tot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--runs', action='append', default=[])
    a = p.parse_args()
    runs = {x.split('=')[0]: x.split('=')[1].split(',') for x in a.runs}
    with Pool(a.procs) as pool:
        rows = pool.map(work, [(n, runs) for n in names(SPLIT)], chunksize=1)
    out = dict(masks={k: f1(rows, lambda r, k=k: r['counts'][k]) for k in rows[0]['counts']})
    out['objects'] = {k: {x: sum(r['objects'][k][x] for r in rows) for x in rows[0]['objects'][k]} for k in rows[0]['objects']}
    out['imc_split'] = {x: sum(r['split']['imc'][x] for r in rows) for x in rows[0]['split']['imc']}
    for task, cfgs in runs.items():
        out[task] = dict(vote=f1(rows, lambda r: r['runs'][task]['vote']),
                         each=[f1(rows, lambda r, i=i: r['runs'][task]['each'][i]) for i in range(len(cfgs))],
                         split=[{x: sum(r['runs'][task]['split'][i][x] for r in rows) for x in rows[0]['split']['imc']} for i in range(len(cfgs))])
    write_json(a.out, out)
    print(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == '__main__':
    main()
