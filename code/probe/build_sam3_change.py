"""Zero-shot baseline, step 2: OmniOVCD overlap-presence matching on stitched SAM3 masks.

Same rule as change_agent/adapters/omniovcd_adapter.py (MaskPairProcessor, mode
overlap_presence, threshold 0.25): an 8-connected building instance on one date is
unchanged if at least 25% of its area is covered by any instance on the other date;
unmatched instances from both dates form the change mask. Uses scipy for speed.
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

from common import DATASETS, PRED, load_label, names, pixel_metrics, save_mask, write_json

STRUCTURE = np.ones((3, 3), bool)
THRESHOLD = 0.25


def unmatched(labels: np.ndarray, count: int, other: np.ndarray) -> np.ndarray:
    if count == 0:
        return np.zeros(labels.shape, bool)
    index = np.arange(1, count + 1)
    area = ndimage.sum_labels(np.ones(labels.shape), labels, index)
    covered = ndimage.sum_labels(other, labels, index)
    keep = np.zeros(count + 1, bool)
    keep[1:] = covered / area < THRESHOLD
    return keep[labels]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    args = parser.parse_args()
    base = PRED / args.dataset / "sam3_omniovcd"
    counts = np.zeros(3, np.int64)
    items = names(args.dataset)
    for name in items:
        m1 = np.asarray(Image.open(base / "t1_mask" / name)) > 0
        m2 = np.asarray(Image.open(base / "t2_mask" / name)) > 0
        l1, n1 = ndimage.label(m1, STRUCTURE)
        l2, n2 = ndimage.label(m2, STRUCTURE)
        change = unmatched(l1, n1, m2) | unmatched(l2, n2, m1)
        save_mask(base / "mask" / name, change)
        truth = load_label(args.dataset, name)
        counts += [(change & truth).sum(), (change & ~truth).sum(), (~change & truth).sum()]
    report = {"dataset": args.dataset, "detector": "sam3_omniovcd", "images": len(items),
              "matching": "overlap_presence", "overlap_threshold": THRESHOLD,
              "metrics": pixel_metrics(*map(int, counts))}
    write_json(base / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()
