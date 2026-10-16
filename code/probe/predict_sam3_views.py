"""Zero-shot baseline, step 1: SAM3 text prompt "building" on each date.

Images are processed as non-overlapping 256x256 tiles (the prototype's LEVIR input size)
and stitched back. Saves per-date building masks and confidence maps; the change mask is
built afterwards by build_sam3_change.py (OmniOVCD overlap-presence matching).
Run in conda_envs/omniovcd-env on a GPU node.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import DATASETS, PRED, load_rgb, names, save_mask, write_json
from sam3_tools import autocast, build_processor, text_adapter

TILE = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    adapter = text_adapter(build_processor())
    out = PRED / args.dataset / "sam3_omniovcd"
    items = names(args.dataset)[: args.limit or None]
    start = time.time()
    for index, name in enumerate(items):
        for view in ("t1", "t2"):
            image = load_rgb(args.dataset, name, view)
            height, width = image.shape[:2]
            mask = np.zeros((height, width), bool)
            conf = np.zeros((height, width), np.float32)
            for y in range(0, height, TILE):
                for x in range(0, width, TILE):
                    with autocast():
                        tile_mask, evidence = adapter.segment_text(image[y:y + TILE, x:x + TILE], "building")
                    mask[y:y + TILE, x:x + TILE] = tile_mask
                    conf[y:y + TILE, x:x + TILE] = evidence["confidence_map"]
            save_mask(out / f"{view}_mask" / name, mask)
            (out / f"{view}_conf").mkdir(parents=True, exist_ok=True)
            np.save(out / f"{view}_conf" / name.replace(".png", ".npy"), conf.astype(np.float16))
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{len(items)} {time.time() - start:.0f}s", flush=True)
    write_json(out / "views_report.json", {"dataset": args.dataset, "images": len(items), "tile": TILE,
                                           "query": "building", "mask_threshold": 0.4,
                                           "seconds": round(time.time() - start, 1)})


if __name__ == "__main__":
    main()
