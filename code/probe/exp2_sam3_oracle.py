"""Experiment 2, step 2: SAM3 (SAM1-task interactive head) on oracle prompts.

For every GT object in exp2/<dataset>/prompts.jsonl and each listed date, crop the image,
run SAM3 predict_inst with (a) the GT box and (b) the GT box + one positive point, using
single-mask output as in the official box examples. Saves crop-sized masks per image in
exp2/<dataset>/sam3/<image>.npz with keys "<id>_<view>_<variant>" and a scores jsonl.
Run in conda_envs/omniovcd-env on a GPU node.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
from PIL import Image

from common import DATASETS, OUT, load_rgb
from sam3_tools import autocast, build_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    processor = build_processor(inst_interactivity=True)
    model = processor.model
    base = OUT / "exp2" / args.dataset
    rows = [json.loads(line) for line in open(base / "prompts.jsonl")]
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image"]].append(row)
    images = sorted(by_image)[: args.limit or None]
    (base / "sam3").mkdir(parents=True, exist_ok=True)
    scores = open(base / "sam3_scores.jsonl", "w")
    start = time.time()
    for index, name in enumerate(images):
        rgb = {view: load_rgb(args.dataset, name, view) for view in ("t1", "t2")}
        masks = {}
        for row in by_image[name]:
            left, top, right, bottom = row["crop"]
            x0, y0, x1, y1 = row["bbox"]
            box = np.array([x0 - left, y0 - top, x1 - left, y1 - top], np.float32)
            point = np.array([[row["point"][0] - left, row["point"][1] - top]], np.float32)
            for view in row["views"]:
                crop = Image.fromarray(rgb[view][top:bottom, left:right])
                with autocast():
                    state = processor.set_image(crop)
                    for variant, kwargs in (("box", {"box": box[None]}),
                                            ("boxpt", {"box": box, "point_coords": point,
                                                       "point_labels": np.array([1])})):
                        out, score, _ = model.predict_inst(state, multimask_output=False, **kwargs)
                        masks[f"{row['id']}_{view}_{variant}"] = out[0] > 0
                        scores.write(json.dumps({"image": name, "id": row["id"], "view": view,
                                                 "variant": variant, "score": float(score[0])}) + "\n")
        np.savez_compressed(base / "sam3" / name.replace(".png", ".npz"), **masks)
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{len(images)} {time.time() - start:.0f}s", flush=True)
    scores.close()
    print(f"done {len(images)} images in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
