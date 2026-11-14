"""Retired WHU v2 entry point; shared scoring helpers remain for historical tools.
Current official C0/C1 correction is implemented by finalize_whu_gt.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def read(path):
    with Image.open(path) as im:
        return np.asarray(im) > 0


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def counts(p, g):
    return dict(tp=int((p & g).sum()), fp=int((p & ~g).sum()), fn=int((~p & g).sum()))


def metrics(c):
    t, f, n = (c[k] for k in ('tp', 'fp', 'fn'))
    return dict(c, f1=2*t/(2*t+f+n) if 2*t+f+n else 1., iou=t/(t+f+n) if t+f+n else 1.,
                precision=t/(t+f) if t+f else float(not n), recall=t/(t+n) if t+n else float(not f))


def components(mask, official):
    lab, n = ndi.label(mask, np.ones((3, 3)))
    area = np.bincount(lab.ravel(), minlength=n+1)
    hit = np.bincount(lab[official].ravel(), minlength=n+1)
    rows=[]
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        rows.append(dict(id=i, pixels=int(area[i]), official_overlap=int(hit[i]),
                         official_fraction=float(hit[i]/area[i]),
                         bbox=[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
                         qc='outside_official' if not hit[i] else 'partial_official' if hit[i] != area[i] else 'inside_official'))
    return rows


def build(a):
    raise SystemExit('Historical v2 audit retired. Use analysis/finalize_whu_gt.py for official C0/C1 with ignore. Existing reports are retained.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    build(p.parse_args())
