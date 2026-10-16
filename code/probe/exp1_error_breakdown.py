"""Experiment 1: error decomposition of each detector on the LEVIR-CD / WHU-CD test sets.

Writes, per dataset and detector, under outputs/CD_1016_probe/exp1/<dataset>/<detector>/:
  objects_gt.jsonl / objects_pred.jsonl  per-object records at threshold 0.5
  summary.json                           pixel metrics, error shares, area-bucket miss rates
  sweep.json                             threshold sweep (probability detectors only)
  vis/<class>_<k>.png                    5 examples per error class (t1 | t2 | GT | pred)
and outputs/CD_1016_probe/exp1/tables.md with all tables.
Run in conda_envs/vlm-vs-onejev-env (CPU).
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from common import (AREA_BUCKETS, DATASETS, OUT, PROB_DETECTORS, area_bucket, detectors, load_label,
                    load_pred, names, pixel_metrics, write_json)
from objects import gt_label, match
from render import object_panel

EXP = OUT / "exp1"
SWEEP = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
CLASSES = ("miss", "partial", "fp", "boundary")


def analyse_image(name, dataset, detector, threshold):
    gt = load_label(dataset, name)
    pred = load_pred(dataset, detector, name, threshold if detector in PROB_DETECTORS else None)
    _, _, gobjs, pobjs, errors = match(gt, pred, gt_label(dataset, name, gt))
    pixels = [int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())]
    return name, gobjs, pobjs, errors, pixels


def run_threshold(dataset, detector, threshold, workers):
    items = names(dataset)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(partial(analyse_image, dataset=dataset, detector=detector, threshold=threshold),
                                items, chunksize=4))
    return results


def summarise(results, dataset=None):
    tp = fp = fn = 0
    errors = {}
    gts, preds = [], []
    for name, gobjs, pobjs, err, pixels in results:
        tp, fp, fn = tp + pixels[0], fp + pixels[1], fn + pixels[2]
        for key, value in err.items():
            errors[key] = errors.get(key, 0) + value
        gts += [{"image": name, **o} for o in gobjs]
        preds += [{"image": name, **o} for o in pobjs]
    total_error = errors["miss"] + errors["partial"] + errors["fp"] + errors["boundary"]
    assert total_error == fp + fn, (total_error, fp, fn)
    counts = {c: sum(o["class"] == c for o in gts) for c in ("hit", "partial", "miss")}
    # Object counts without sub-20-px fragments (SECOND transition boundaries produce many).
    counts_ge20 = {c: sum(o["class"] == c and o["area"] >= 20 for o in gts) for c in ("hit", "partial", "miss")}
    buckets = {}
    for _, _, label in AREA_BUCKETS:
        group = [o for o in gts if area_bucket(o["area"]) == label]
        n = len(group)
        buckets[label] = {"gt_objects": n,
                          **{f"{c}_rate": (sum(o["class"] == c for o in group) / n if n else None)
                             for c in ("hit", "partial", "miss")}}
    by_transition = {}
    if gts and "transition" in gts[0]:
        classes = DATASETS[dataset]["classes"]
        for o in gts:
            key = f"{classes[o['transition'][0]]} -> {classes[o['transition'][1]]}"
            entry = by_transition.setdefault(key, {"gt_objects": 0, "pixels": 0, "miss": 0, "partial": 0,
                                                    "miss_pixels": 0})
            entry["gt_objects"] += 1
            entry["pixels"] += o["area"]
            entry["miss"] += o["class"] == "miss"
            entry["partial"] += o["class"] == "partial"
            entry["miss_pixels"] += o["area"] if o["class"] == "miss" else 0
            entry["no_signal_pixels"] = entry.get("no_signal_pixels", 0) + (o["area"] if o["coverage"] < 0.1 else 0)
    # Supplement: "no-signal" misses by coverage (< 10% of the object's pixels predicted), which
    # does not depend on the size of the predicted component (merged blobs lower the IoU of
    # well-covered objects; this matters on SECOND).
    nosig = [o for o in gts if o["coverage"] < 0.1]
    total_gt_pixels = sum(o["area"] for o in gts)
    no_signal = {"objects": len(nosig), "objects_area_ge20": sum(o["area"] >= 20 for o in nosig),
                 "pixel_share_of_gt": (sum(o["area"] for o in nosig) / total_gt_pixels) if total_gt_pixels else None,
                 "iou_misses_with_coverage_ge_0.5": sum(o["class"] == "miss" and o["coverage"] >= 0.5 for o in gts),
                 "rate_by_area": {label: (sum(o["coverage"] < 0.1 for o in gts if area_bucket(o["area"]) == label)
                                          / max(1, sum(area_bucket(o["area"]) == label for o in gts)))
                                  for _, _, label in AREA_BUCKETS}}
    partial = [o for o in gts if o["class"] == "partial"]
    miss = [o for o in gts if o["class"] == "miss"]
    return {
        "pixel": pixel_metrics(tp, fp, fn),
        "error_pixels": errors, "total_error_pixels": total_error,
        "error_share": {c: errors[c] / total_error for c in CLASSES},
        "gt_objects": len(gts), "gt_object_counts": counts, "gt_object_counts_area_ge20": counts_ge20,
        "pred_objects": len(preds), "fp_objects": sum(o["class"] == "fp" for o in preds),
        "partial_pred_objects": sum(o["class"] == "partial" for o in preds),
        "area_buckets": buckets,
        "partial_gt_coverage_ge_0.5": (sum(o["coverage"] >= 0.5 for o in partial) / len(partial)) if partial else None,
        "miss_without_any_overlap": (sum(not o["any_overlap"] for o in miss) / len(miss)) if miss else None,
        "no_signal_miss": no_signal,
        **({"by_transition": by_transition} if by_transition else {}),
    }, gts, preds


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def visualise(dataset, detector, gts, preds, out):
    rng = random.Random(1016)
    pools = {
        "miss": [("gt", o) for o in gts if o["class"] == "miss" and o["area"] >= 20],
        "partial": [("gt", o) for o in gts if o["class"] == "partial" and o["area"] >= 20],
        "fp": [("pred", o) for o in preds if o["class"] == "fp" and o["area"] >= 20],
        # Hit objects whose boundary error is visible: IoU in the lower half of hits.
        "boundary": [("gt", o) for o in gts if o["class"] == "hit" and o["best_iou"] < 0.75 and o["area"] >= 50],
    }
    for cls, pool in pools.items():
        for k, (kind, o) in enumerate(rng.sample(pool, min(5, len(pool)))):
            title = f"{dataset} {detector} {cls} | {o['image']} {kind}#{o['id']} area={o['area']} IoU={o['best_iou']:.2f}"
            object_panel(dataset, detector, o["image"], o["bbox"], title).save(out / "vis" / f"{cls}_{k}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        for detector in detectors(dataset):
            out = EXP / dataset / detector
            (out / "vis").mkdir(parents=True, exist_ok=True)
            summary, gts, preds = summarise(run_threshold(dataset, detector, 0.5, args.workers), dataset)
            write_jsonl(out / "objects_gt.jsonl", gts)
            write_jsonl(out / "objects_pred.jsonl", preds)
            write_json(out / "summary.json", summary)
            visualise(dataset, detector, gts, preds, out)
            print(dataset, detector, json.dumps(summary["pixel"]), json.dumps(summary["error_share"]), flush=True)
            if detector not in PROB_DETECTORS:
                continue
            base_miss = {(o["image"], o["id"]) for o in gts if o["class"] == "miss"}
            sweep = []
            for threshold in SWEEP:
                s, g2, p2 = summarise(run_threshold(dataset, detector, threshold, args.workers), dataset) if threshold != 0.5 \
                    else (summary, gts, preds)
                recovered = [o for o in g2 if (o["image"], o["id"]) in base_miss and o["class"] != "miss"]
                sweep.append({"threshold": threshold, "pixel": s["pixel"], "gt_object_counts": s["gt_object_counts"],
                              "fp_objects": s["fp_objects"], "pred_objects": s["pred_objects"],
                              "recovered_misses": len(recovered),
                              "recovered_misses_as_hit": sum(o["class"] == "hit" for o in recovered),
                              "base_misses": len(base_miss)})
                print(dataset, detector, threshold, json.dumps(sweep[-1]["pixel"]), flush=True)
            write_json(out / "sweep.json", sweep)


if __name__ == "__main__":
    main()
