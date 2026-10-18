"""Experiment 4c, step 2: SAM3 box prompts on the selected candidates (GPU, omniovcd-env).

Same as experiment 2 (SAM1-task interactive head, single mask, crop around the box), but the
box comes from the candidate, not from GT. When the date is unknown both dates are run and the
mask with the higher SAM3 predicted quality is kept (no GT). Writes exp4c/<dataset>/sam3_<kind>/<image>.npz
with one crop mask per selected candidate (index in file order) and the chosen view/score.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from common import OUT, load_rgb
from sam3_tools import autocast, build_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--kind", required=True, help="reads <base>/<dataset>/selected_<kind>.jsonl")
    parser.add_argument("--base", default="exp4c")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    processor = build_processor(inst_interactivity=True)
    model = processor.model
    base = OUT / args.base / args.dataset
    rows = [json.loads(line) for line in open(base / f"selected_{args.kind}.jsonl")]
    by_image = defaultdict(list)
    for i, r in enumerate(rows):
        by_image[r["image"]].append((i, r))
    out = base / f"sam3_{args.kind}"
    out.mkdir(exist_ok=True)
    for n, (name, items) in enumerate(sorted(by_image.items())):
        if n % args.shards != args.shard:
            continue
        rgb = {v: load_rgb(args.dataset, name, v) for v in ("t1", "t2")}
        masks, meta = {}, []
        for i, r in items:
            left, top, right, bottom = r["crop"]
            x0, y0, x1, y1 = r["bbox"]
            box = np.array([[x0 - left, y0 - top, x1 - left, y1 - top]], np.float32)
            best = None
            for view in r["views"]:
                with autocast():
                    state = processor.set_image(Image.fromarray(rgb[view][top:bottom, left:right]))
                    m, score, _ = model.predict_inst(state, box=box, multimask_output=False)
                if best is None or float(score[0]) > best[0]:
                    best = (float(score[0]), view, m[0] > 0)
            masks[f"m{i}"] = best[2]
            meta.append({"index": i, "view": best[1], "sam_score": best[0]})
        np.savez_compressed(out / name.replace(".png", ".npz"), meta=json.dumps(meta), **masks)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
