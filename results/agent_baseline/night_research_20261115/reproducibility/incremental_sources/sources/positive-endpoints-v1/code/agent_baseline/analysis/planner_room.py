"""How much room is there for the planner beyond the rule? Oracle bounds on val, labels only score.

Zoomed re-segmentation: instances from the 256 grid (2x zoom of the 512 grid) matched across the dates; regions the
fixed mask misses, kept by an oracle when mostly true. Local boundary: each kept instance rethresholded on its soft
output, the threshold picked by an oracle.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'ovcd_v0'), str(Path(__file__).resolve().parents[2])]
import agent_baseline
from agent_baseline import instances
from common import names, load_gt, components

R = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
SPLIT = 'val'
RULE = {json.loads(l)['uid']: json.loads(l).get('answer') for l in open(R / 'ask_eval/val_sam3_imc_p85/f37_corr.jsonl')}
THRS = (.5, .6, .7, .8, .85, .9, .95)


def read(stem, tile, step, index, pixel_thr=.85, score_thr=.5):
    with np.load(R / f'sam3_soft/t{tile}s{step}' / SPLIT / f'{stem}.npz') as z:
        scores, origins, offs = z[f'scores{index}'], z[f'origins{index}'], z[f'offs{index}']
        idx, val, (h, w) = z[f'idx{index}'], z[f'val{index}'], z['shape']
    rows = []
    for i in np.flatnonzero(scores >= score_thr):
        if origins[i][0] % tile or origins[i][1] % tile:
            continue
        pixels, values = idx[offs[i]:offs[i + 1]], val[offs[i]:offs[i + 1]]
        keep = pixels[values >= round(pixel_thr * 255)]
        if len(keep):
            keep = np.sort(keep).astype(np.int32)
            ys, xs = keep // w, keep % w
            rows.append(dict(id=len(rows) + 1, idx=keep, area=int(keep.size), score=float(scores[i]), soft=(pixels, values),
                             bbox=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return rows, (int(h), int(w))


def route(stem, tile, step):
    a, shape = read(stem, tile, step, 0)
    b, _ = read(stem, tile, step, 1)
    appeared, disappeared, _ = instances.match(a, b, .25)
    return list(appeared) + list(disappeared), shape


def cnt(m, gt):
    return np.array([(m & gt).sum(), (m & ~gt).sum(), (~m & gt).sum()], float)


def work(name):
    stem = Path(name).stem
    gt = load_gt(SPLIT, name)
    init = np.asarray(Image.open(R / f'init/sam3_imc_p85/{SPLIT}/{stem}.png')) > 0
    labels, comps = components(init)
    no = [c['id'] for c in comps if RULE.get(f"{stem}:{c['id']}") == 'no']
    rule = init & ~np.isin(labels, no)
    out = dict(init=cnt(init, gt), rule=cnt(rule, gt))
    # Zoom: regions of the 256-grid route outside the fixed mask.
    rows, shape = route(stem, 256, 128)
    h, w = shape
    union = instances.to_mask(rows, shape) if rows else np.zeros(shape, bool)
    lab2, comps2 = components(union)
    add_oracle, add_all = rule.copy(), rule.copy()
    n_new = n_true = 0
    area_new = 0
    for c in comps2:
        region = lab2 == c['id']
        if init[region].mean() > .1 or c['area'] < 30:
            continue
        n_new += 1
        area_new += int(c['area'])
        add_all |= region
        if gt[region].mean() >= .5:
            add_oracle |= region
            n_true += 1
    out.update(zoom_oracle=cnt(add_oracle, gt), zoom_all=cnt(add_all, gt), n_new=n_new, n_true=n_true, area_new=area_new)
    # Boundary: every kept instance at its oracle threshold, on the 512 grid.
    rows, shape = route(stem, 512, 256)
    flat = np.zeros(h * w, bool)
    flat085 = np.zeros(h * w, bool)
    gt_flat = gt.ravel()
    F = .86
    for r in rows:
        ys, xs = np.divmod(r['idx'][r['idx'].size // 2], w)
        if not rule[ys, xs]:
            # Use the majority of the instance rather than its middle pixel.
            if rule.ravel()[r['idx']].mean() < .5:
                continue
        pixels, values = r['soft']
        best, chosen = -1e18, r['idx']
        for t in THRS:
            sel = pixels[values >= round(t * 255)]
            tp = gt_flat[sel].sum()
            gain = tp - F / 2 * sel.size
            if gain > best:
                best, chosen = gain, sel
        flat[chosen] = True
        flat085[r['idx']] = True
    out.update(boundary_base=cnt(flat085.reshape(h, w), gt), boundary_oracle=cnt(flat.reshape(h, w), gt))
    return out


def f1(c):
    return round(float(200 * c[0] / (2 * c[0] + c[1] + c[2])), 2)


def main():
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        res = pool.map(work, names(SPLIT), chunksize=2)
    total = lambda k: sum(r[k] for r in res)
    report = {k: f1(total(k)) for k in ('init', 'rule', 'zoom_oracle', 'zoom_all', 'boundary_base', 'boundary_oracle')}
    report.update(new_regions=sum(r['n_new'] for r in res), new_true=sum(r['n_true'] for r in res), new_area=sum(r['area_new'] for r in res),
                  rule_counts=[int(v) for v in total('rule')], zoom_oracle_counts=[int(v) for v in total('zoom_oracle')])
    print(json.dumps(report, indent=1))
    (R / 'probe/planner_room.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
