"""Experiment 2, step 3: compose oracle-refined predictions and measure F1.

For each detector (threshold 0.5) and SAM3 prompt variant (box / box+point):
  2a  every hit GT object (best IoU >= 0.5): delete its matched predicted component, except
      pixels lying on other GT objects, and paste the SAM3 mask instead;
  2a_strict  same but delete the whole matched component (first run; it also deleted
      correct predictions of neighbouring GT objects merged into that component);
  2a_select  supplement: replace only hits where the SAM3 IoU beats the detector IoU
      (an upper bound for an agent with a perfect accept/reject verifier);
  2b  every whole-object miss (best IoU < 0.1): add the SAM3 mask;
  both 2a + 2b;   both_select  2a_select + 2b;
  2c  (supplement) every partial object (0.1 <= IoU < 0.5): add the SAM3 mask (union only,
      because its predicted component is often shared with neighbouring objects).
The SAM3 date for "unknown" objects is chosen by the higher IoU with the GT object (oracle).
Run in conda_envs/vlm-vs-onejev-env (CPU).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from common import (AREA_BUCKETS, DATASETS, OUT, area_bucket, detectors, load_label, load_pred, names,
                    pixel_metrics, write_json)
from objects import gt_label, match

VARIANTS = ("box", "boxpt")
SETTINGS = ("original", "2a", "2a_strict", "2a_select", "2b", "both", "both_select", "2c")


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def process(name, dataset, prompts):
    gt = load_label(dataset, name)
    stored = np.load(OUT / "exp2" / dataset / "sam3" / name.replace(".png", ".npz"))
    rows = {r["id"]: r for r in prompts}
    # Full-size SAM3 masks for each object/variant, with the oracle date choice.
    gt_labels = gt_label(dataset, name, gt)
    gl = gt_labels[0]
    sam, sam_iou = {}, {}
    per_object = []
    for g, row in rows.items():
        obj = gl == g
        left, top, right, bottom = row["crop"]
        for variant in VARIANTS:
            best = None
            view_iou = {}
            for view in row["views"]:
                full = np.zeros_like(gt)
                full[top:bottom, left:right] = stored[f"{g}_{view}_{variant}"]
                view_iou[view] = iou(full, obj)
                if best is None or view_iou[view] > best[0]:
                    best = (view_iou[view], view, full)
            sam[(g, variant)] = best[2]
            sam_iou[(g, variant)] = best[0]
            per_object.append({"image": name, "id": g, "variant": variant, "kind": row["kind"],
                               "area": row["area"], "view_iou": view_iou, "chosen_view": best[1],
                               "sam_iou": best[0]})
    results = {}
    for detector in detectors(dataset):
        pred = load_pred(dataset, detector, name)
        _, pl, gobjs, _, _ = match(gt, pred, gt_labels)
        cls = {o["id"]: o for o in gobjs}
        for o in per_object:
            o.setdefault("detector", {})[detector] = {"class": cls[o["id"]]["class"],
                                                     "det_iou": cls[o["id"]]["best_iou"]}
        for variant in VARIANTS:
            hits = [g for g, o in cls.items() if o["class"] == "hit"]
            misses = [g for g, o in cls.items() if o["class"] == "miss"]
            partials = [g for g, o in cls.items() if o["class"] == "partial"]
            strict = pred.copy()
            refined_hits = pred.copy()
            # Objects without a SAM3 result (SECOND fragments < 20 px, not prompted) stay unchanged.
            hits = [g for g in hits if (g, variant) in sam]
            misses = [g for g in misses if (g, variant) in sam]
            partials = [g for g in partials if (g, variant) in sam]
            for g in hits:
                component = pl == cls[g]["best_pred"]
                strict &= ~component
                # Keep the component's pixels that lie on other GT objects: they are the
                # prediction for neighbouring targets, not this target's boundary error.
                refined_hits &= ~(component & ~(gt & (gl != g)))
            for g in hits:
                strict |= sam[(g, variant)]
                refined_hits |= sam[(g, variant)]
            # Perfect-verifier bound: replace only where SAM3 beats the detector on that object.
            select = pred.copy()
            better = [g for g in hits if sam_iou[(g, variant)] > cls[g]["best_iou"]]
            for g in better:
                select &= ~((pl == cls[g]["best_pred"]) & ~(gt & (gl != g)))
            for g in better:
                select |= sam[(g, variant)]
            add_miss = np.zeros_like(gt)
            for g in misses:
                add_miss |= sam[(g, variant)]
            add_partial = np.zeros_like(gt)
            for g in partials:
                add_partial |= sam[(g, variant)]
            masks = {"original": pred, "2a": refined_hits, "2a_strict": strict, "2a_select": select, "2b": pred | add_miss,
                     "both": refined_hits | add_miss, "both_select": select | add_miss,
                     "2c": pred | add_partial}
            for setting, m in masks.items():
                results[(detector, variant, setting)] = [int((m & gt).sum()), int((m & ~gt).sum()),
                                                         int((~m & gt).sum())]
    return results, per_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    tables = {}
    for dataset in args.datasets:
        base = OUT / "exp2" / dataset
        prompts = defaultdict(list)
        for line in open(base / "prompts.jsonl"):
            row = json.loads(line)
            prompts[row["image"]].append(row)
        items = sorted(prompts)
        totals = defaultdict(lambda: np.zeros(3, np.int64))
        objects = []
        with ProcessPoolExecutor(args.workers) as pool:
            for results, per_object in pool.map(partial(_run, dataset=dataset), [(n, prompts[n]) for n in items],
                                                chunksize=4):
                for key, value in results.items():
                    totals[key] += value
                objects += per_object
        # Images with no GT objects still contribute FP pixels to "original"; add them.
        empty = [n for n in names(dataset) if n not in prompts]
        for name in empty:
            gt = load_label(dataset, name)
            for detector in detectors(dataset):
                pred = load_pred(dataset, detector, name)
                for variant in VARIANTS:
                    for setting in SETTINGS:
                        totals[(detector, variant, setting)] += [0, int(pred.sum()), 0]
        metrics = {f"{d}|{v}|{s}": pixel_metrics(*map(int, c)) for (d, v, s), c in totals.items()}
        sam_quality = {}
        for variant in VARIANTS:
            rows = [o for o in objects if o["variant"] == variant]
            entry = {"all_mean_iou": float(np.mean([o["sam_iou"] for o in rows]))}
            for _, _, label in AREA_BUCKETS:
                vals = [o["sam_iou"] for o in rows if area_bucket(o["area"]) == label]
                entry[f"mean_iou_{label}"] = float(np.mean(vals)) if vals else None
            for detector in detectors(dataset):
                hits = [o for o in rows if o["detector"][detector]["class"] == "hit"]
                misses = [o for o in rows if o["detector"][detector]["class"] == "miss"]
                entry[detector] = {
                    "hits": len(hits),
                    "hit_det_mean_iou": float(np.mean([o["detector"][detector]["det_iou"] for o in hits])) if hits else None,
                    "hit_sam_mean_iou": float(np.mean([o["sam_iou"] for o in hits])) if hits else None,
                    "hit_sam_better_frac": float(np.mean([o["sam_iou"] > o["detector"][detector]["det_iou"] for o in hits])) if hits else None,
                    "misses": len(misses),
                    "miss_sam_mean_iou": float(np.mean([o["sam_iou"] for o in misses])) if misses else None,
                    "miss_sam_iou_ge_0.5_frac": float(np.mean([o["sam_iou"] >= 0.5 for o in misses])) if misses else None,
                }
            both = [o for o in rows if len(o["view_iou"]) == 2]
            entry["two_date_objects"] = len(both)
            entry["two_date_chosen_t2_frac"] = float(np.mean([o["chosen_view"] == "t2" for o in both])) if both else None
            sam_quality[variant] = entry
        write_json(base / "metrics.json", {"pixel": metrics, "sam_quality": sam_quality,
                                           "images_without_gt": len(empty)})
        with open(base / "objects.jsonl", "w") as f:
            for o in objects:
                f.write(json.dumps(o) + "\n")
        tables[dataset] = metrics
        for detector in detectors(dataset):
            line = " ".join(f"{s}={metrics[f'{detector}|box|{s}']['f1']:.4f}" for s in SETTINGS)
            print(dataset, detector, "box", line, flush=True)


def _run(item, dataset):
    name, prompts = item
    return process(name, dataset, prompts)


if __name__ == "__main__":
    main()
