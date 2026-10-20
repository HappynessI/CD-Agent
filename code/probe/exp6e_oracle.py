"""Experiment 6e.1: false-positive removal oracles (CPU, conda_envs/exp5-env).

Use the saved detector 0.5 mask and exp5c's SLIC split. oracle_piece keeps all
pieces < 10 px; oracle_whole_fp independently deletes zero-overlap connected
components of any size. Piece records include small retained pieces; bbox is
[x0, y0, x1, y1) and piece_id is 1-based within each image. Metrics and delta_f1
are fractions, micro-aggregated over the complete test split.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy import ndimage
from skimage.segmentation import slic

from common import OUT, load_label, load_pred, load_prob, load_rgb, names, pixel_metrics, write_json
from objects import label

DETECTOR = {"second": "mambascd_base", "sysu": "schanger_small"}
VARIANTS = ("oracle_piece", "oracle_whole_fp")
MIN_AREA, GT_FRAC = 10, 0.1


def process(name, dataset, median_area):
    gt = load_label(dataset, name)
    pred = load_pred(dataset, DETECTOR[dataset], name)
    prob = load_prob(dataset, DETECTOR[dataset], name)
    labels, _ = label(pred)
    masks = {key: pred.copy() for key in VARIANTS}
    counts = dict.fromkeys(VARIANTS, 0)
    records, segments = [], None
    for component_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[sl] == component_id
        local_gt, local_prob = gt[sl], prob[sl]
        if not local_gt[region].any():
            masks["oracle_whole_fp"][sl][region] = False
            counts["oracle_whole_fp"] += 1

        split = int(region.sum()) > 4 * median_area
        if split:
            if segments is None:
                stack = np.concatenate([load_rgb(dataset, name, v) for v in ("t1", "t2")], axis=2)
                segments = slic(stack, n_segments=max(4, int(gt.size / median_area)),
                                compactness=10, channel_axis=-1, start_label=1)
            local_segments = segments[sl]
            pieces = (region & (local_segments == i) for i in np.unique(local_segments[region]))
        else:
            pieces = (region,)
        for piece in pieces:
            area = int(piece.sum())
            gt_frac = float(local_gt[piece].mean())
            deleted = area >= MIN_AREA and gt_frac < GT_FRAC
            if deleted:
                masks["oracle_piece"][sl][piece] = False
                counts["oracle_piece"] += 1
            ys, xs = np.nonzero(piece)
            records.append({
                "image": name, "piece_id": len(records) + 1, "component_id": component_id,
                "bbox": [int(xs.min()) + sl[1].start, int(ys.min()) + sl[0].start,
                         int(xs.max()) + sl[1].start + 1, int(ys.max()) + sl[0].start + 1],
                "area": area, "det_prob": float(local_prob[piece].mean()),
                "gt_frac": gt_frac, "split": split, "deleted_oracle_piece": deleted,
            })

    totals = {}
    for key, mask in {"original": pred, **masks}.items():
        totals[key] = [int((mask & gt).sum()), int((mask & ~gt).sum()), int((~mask & gt).sum())]
    removed = {}
    for key, mask in masks.items():
        deleted = pred & ~mask
        removed[key] = [int((deleted & gt).sum()), int((deleted & ~gt).sum()), counts[key]]
    return records, totals, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DETECTOR), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    config_path = OUT / "exp5c" / args.dataset / "metrics_5c_B_G0.05.json"
    median_area = float(json.loads(config_path.read_text())["config"]["median_target_area"])
    items = names(args.dataset)
    if not items:
        raise ValueError(f"No test images found for {args.dataset}")
    base = OUT / "exp6e" / args.dataset
    base.mkdir(parents=True, exist_ok=True)
    totals = {key: np.zeros(3, np.int64) for key in ("original", *VARIANTS)}
    removed = {key: np.zeros(3, np.int64) for key in VARIANTS}
    worker = partial(process, dataset=args.dataset, median_area=median_area)
    with ProcessPoolExecutor(args.workers) as pool, (base / "pieces.jsonl").open("w") as f:
        for index, (records, image_totals, image_removed) in enumerate(
                pool.map(worker, items, chunksize=4), start=1):
            for record in records:
                f.write(json.dumps(record) + "\n")
            for key in totals:
                totals[key] += image_totals[key]
            for key in removed:
                removed[key] += image_removed[key]
            if index % 100 == 0:
                print(args.dataset, index, "/", len(items), flush=True)

    report = {key: pixel_metrics(*map(int, values)) for key, values in totals.items()}
    for key in VARIANTS:
        report[key]["delta_f1"] = report[key]["f1"] - report["original"]["f1"]
        report[key]["deleted_tp_px"], report[key]["deleted_fp_px"], report[key]["deleted_regions"] = (
            map(int, removed[key]))
    report["config"] = {
        "dataset": args.dataset, "detector": DETECTOR[args.dataset], "images": len(items),
        "workers": args.workers, "aggregation": "micro", "prediction": "saved_0.5_mask",
        "connectivity": 8, "median_target_area_source": str(config_path),
        "median_target_area": median_area, "large_area": 4 * median_area,
        "slic_n_segments": "max(4, int(image_area / median_target_area))", "slic_compactness": 10,
        "min_piece_area": MIN_AREA, "oracle_piece_gt_frac_lt": GT_FRAC,
        "oracle_whole_fp": "delete whole components with zero GT overlap, any area",
        "deleted_regions": "pieces for oracle_piece; components for oracle_whole_fp",
    }
    write_json(base / "oracle.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
