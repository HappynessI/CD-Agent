"""Experiment 2, step 1: oracle prompts for every GT change object (detector independent).

For each 8-connected GT object: its bbox, one positive point (the interior pixel farthest
from the object boundary), the crop sent to SAM3, and which date(s) to segment.
Date choice: WHU has per-date building labels, so an object that is building on t2 only
is "new" (segment t2), building on t1 only is "demolished" (segment t1); otherwise, and
for all LEVIR, SYSU and SECOND objects, both dates are run and the higher IoU is kept.
SECOND objects are split by class transition (objects.gt_label).
Run in conda_envs/vlm-vs-onejev-env (CPU).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage

from common import DATASETS, OUT, load_label, load_semantic, names
from objects import gt_label
from render import crop_box

CROP_SCALE = 3.0      # crop side = max(3 x object long side, 128 px), clipped to the image
CROP_MIN = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    for dataset in parser.parse_args().datasets:
        rows = []
        for name in names(dataset):
            gt = load_label(dataset, name)
            gl, ng, transitions = gt_label(dataset, name, gt)
            sem = load_semantic(dataset, name)
            for g, sl in enumerate(ndimage.find_objects(gl), start=1):
                obj = gl[sl] == g
                dist = ndimage.distance_transform_edt(np.pad(obj, 1))[1:-1, 1:-1]
                py, px = np.unravel_index(np.argmax(dist), dist.shape)
                bbox = [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
                if sem is None:
                    kind = "unknown"
                else:
                    f1 = float(sem[0][sl][obj].mean())
                    f2 = float(sem[1][sl][obj].mean())
                    kind = "new" if f2 >= 0.5 > f1 else "demolished" if f1 >= 0.5 > f2 else "unknown"
                views = {"new": ["t2"], "demolished": ["t1"], "unknown": ["t1", "t2"]}[kind]
                rows.append({"image": name, "id": g, "area": int(obj.sum()), "bbox": bbox,
                             "point": [int(px + sl[1].start), int(py + sl[0].start)],
                             "crop": list(crop_box(bbox, gt.shape, CROP_SCALE, CROP_MIN)),
                             "kind": kind, "views": views,
                             **({"transition": transitions[g]} if transitions else {})})
        path = OUT / "exp2" / dataset / "prompts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        kinds = {k: sum(r["kind"] == k for r in rows) for k in ("new", "demolished", "unknown")}
        print(dataset, len(rows), kinds, flush=True)


if __name__ == "__main__":
    main()
