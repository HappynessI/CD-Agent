"""Estimate experiment 6c geometry from train GT only (CPU, exp5-env).

Without train detector masks use all train GT objects >=20 px. With an explicit
complete --train_predictions mask directory, retain objects with coverage <0.1,
as in experiment 6a/5c. No detector training is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import OUT, TRAIN_DATASETS, write_json
from objects import gt_label

EXPECTED = {"levir": 445, "whu": 1260, "lsscd": 1120, "scscd7": 1400}
DETECTOR = {"levir": "schanger_base", "whu": "schanger_base",
            "lsscd": "mambascd_base", "scscd7": "mambascd_base"}
MIN_AREA = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--train_predictions", type=Path,
                        help="Complete train PNG mask directory from the designated detector")
    args = parser.parse_args()
    spec = TRAIN_DATASETS[args.dataset]
    root = spec["root"]
    items = sorted(p.name for p in (root / spec["dirs"][0]).glob(spec["prefix"] + "*.png"))
    if len(items) != EXPECTED[args.dataset]:
        raise ValueError(f"{args.dataset}: expected {EXPECTED[args.dataset]} train pairs, found {len(items)}")
    if args.train_predictions is not None:
        missing = [n for n in items if not (args.train_predictions / n).is_file()]
        if missing:
            raise FileNotFoundError(f"Train prediction directory is incomplete: {missing[:5]}")
    out = OUT / "exp6/geometry" / f"{args.dataset}.json"
    out.unlink(missing_ok=True)
    areas, per_image = [], []
    for name in items:
        with Image.open(root / spec["dirs"][2] / name) as image:
            gt = np.asarray(image) > 127
        # Same transition-aware 8-connectivity as 6a. LEVIR/WHU load_classes returns None.
        labels, count, _ = gt_label(args.dataset, name, gt)
        object_areas = np.bincount(labels.ravel(), minlength=count + 1)
        selected = object_areas >= MIN_AREA
        selected[0] = False
        if args.train_predictions is not None:
            with Image.open(args.train_predictions / name) as image:
                pred = np.asarray(image) > 0
            if pred.shape != gt.shape:
                raise ValueError(f"{name}: prediction {pred.shape} differs from GT {gt.shape}")
            intersections = np.bincount(labels[pred], minlength=count + 1)
            coverage = intersections / np.maximum(object_areas, 1)
            selected &= coverage < 0.1
        retained = object_areas[selected]
        areas.extend(retained.tolist())
        per_image.append({"image": name, "objects": count, "selected_objects": len(retained)})
    if not areas:
        raise ValueError(f"{args.dataset}: no train objects under the selected rule")
    method = "train_missed_gt" if args.train_predictions is not None else "train_all_gt"
    report = {
        "dataset": args.dataset, "split": "train", "root": str(root), "dirs": spec["dirs"],
        "prefix": spec["prefix"], "images": len(items), "objects": len(areas),
        "method": method, "min_area": MIN_AREA, "median_area": float(np.median(areas)),
        "object_definition": ("8_connected_by_transition" if "class1" in spec else "8_connected_binary"),
        "detector": DETECTOR[args.dataset],
        "train_predictions": str(args.train_predictions) if args.train_predictions is not None else None,
        "max_coverage_exclusive": 0.1 if args.train_predictions is not None else None,
        "test_gt_used": False, "per_image": per_image,
        "second_difference": (
            "SECOND uses frozen legacy test missed-object area 298; this uses train missed objects."
            if args.train_predictions is not None else
            "SECOND uses frozen legacy test missed-object area 298; this uses all train GT objects >=20 px."
        ),
    }
    write_json(out, report)
    print(f"{args.dataset}: {len(items)} train images, {len(areas)} objects, "
          f"median={report['median_area']}, method={method}, report={out}", flush=True)


if __name__ == "__main__":
    main()
