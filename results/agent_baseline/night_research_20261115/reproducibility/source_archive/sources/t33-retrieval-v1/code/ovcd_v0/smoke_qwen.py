"""Smoke: T3/T5 on GT components (positives) and random empty boxes (negatives) of 4 val images.
Checks the yes/no readout, timing, and whether T5 boxes come back in 0-1000 crop coordinates."""
import random
import sys
import numpy as np
from common import *
import render
from mllm import Qwen

q = Qwen(QWEN)
rows = []
rng = random.Random(0)
for name in names('val')[:6]:
    rgb, gt = load_rgb('val', name), load_gt('val', name)
    lab, comps = components(gt, 20)
    for c in comps[:3]:
        crop = render.crop_box(c['bbox'], gt.shape)
        t1, t2 = render.render_pair(rgb, lab == c['id'], crop)
        v, b = q.verify(t1, t2), q.box(t1, t2)
        gt_box_norm = [(c['bbox'][0]-crop[0])/(crop[2]-crop[0])*1000, (c['bbox'][1]-crop[1])/(crop[2]-crop[0])*1000,
                       (c['bbox'][2]-crop[0])/(crop[2]-crop[0])*1000, (c['bbox'][3]-crop[1])/(crop[2]-crop[0])*1000]
        rows.append(dict(name=name, kind='gt', verify=v, box=b, gt_box_norm=gt_box_norm))
        print('POS', name, round(v['p_yes'], 3), v['raw'][:160], '|', b['raw'], '| gt', [round(x) for x in gt_box_norm], flush=True)
    for _ in range(2):
        x, y = rng.randrange(0, 960), rng.randrange(0, 960)
        m = np.zeros(gt.shape, bool); m[y:y+48, x:x+48] = True
        if (m & gt).any():
            continue
        crop = render.crop_box([x, y, x+48, y+48], gt.shape)
        t1, t2 = render.render_pair(rgb, m, crop)
        v = q.verify(t1, t2)
        rows.append(dict(name=name, kind='neg', verify=v))
        print('NEG', name, round(v['p_yes'], 3), v['raw'][:160], flush=True)
write_json(RUN / 'smoke/qwen_smoke.json', dict(meta=q.meta, rows=rows))
import torch; print('peak GiB', torch.cuda.max_memory_allocated()/2**30)
