"""Experiment 3, step 1: sample candidate regions and build the MLLM inputs.

Detector: SChanger-base (probability maps). Per dataset, three main sources:
  hit     predicted objects (threshold 0.5) matched to a GT object, IoU >= 0.5   -> positive
  fp      predicted objects with IoU < 0.1 to every GT object                    -> negative
  lowthr  components of prob > 0.05 that do not touch any 0.5 prediction (regions that
          only appear after lowering the threshold); positive if >= 50% of the region
          is GT change, negative if < 10%, otherwise dropped as ambiguous. (0.2 gave only
          3 positive / 7 negative regions >= 20 px on LEVIR, so the lowest swept value is used.)
Supplementary recall-side set (not in the task definition):
  miss       GT objects the detector misses entirely (best IoU < 0.1)            -> positive
  unchanged  LEVIR/WHU: SAM3 "building" instances (zero-shot baseline masks, t1 or t2) with
             < 10% GT change and no 0.5 prediction: unchanged buildings a proposal step
             would also surface. SYSU/SECOND (general change, no building baseline): boxes
             with the size of a randomly drawn sampled miss, placed at random where GT change
             < 10% and no 0.5 prediction                                          -> negative
Detector per dataset: SChanger-base (LEVIR, WHU), SChanger-small (SYSU), MambaSCD-Base (SECOND). SECOND GT
objects are split by class transition (objects.gt_label).
Regions under MIN_AREA pixels are excluded (too small to show in a crop); the excluded
share is recorded. Each candidate gets t1/t2 crops with the region's box drawn in red.
Run in conda_envs/vlm-vs-onejev-env (CPU).
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from common import DATASETS, OUT, PRED, load_label, load_prob, load_rgb, names, write_json
from objects import gt_label, label, match
from render import crop_box

# SYSU uses SChanger-small: its official SYSU F1 (84.58) is higher than base (84.17).
DETECTOR = {"levir": "schanger_base", "whu": "schanger_base", "sysu": "schanger_small", "second": "mambascd_base"}
LOW_THRESHOLD = 0.05
# SYSU/SECOND recall-side positives must be genuinely unpredicted (coverage < 10%): on SECOND
# many IoU<0.1 "misses" are covered by a merged predicted blob. LEVIR/WHU candidates were built
# before this rule; their IoU misses had zero overlap in 82-91% of cases.
MISS_MAX_COVERAGE = {"sysu": 0.1, "second": 0.1}
MIN_AREA = 20
PER_SOURCE = {"hit": 100, "fp": 100, "lowthr_pos": 100, "lowthr_neg": 100, "miss": 100, "unchanged": 100}
CROP_SCALE, CROP_MIN, SIDE = 4.0, 96, 448


def region_records(dataset, name):
    gt = load_label(dataset, name)
    prob = load_prob(dataset, DETECTOR[dataset], name)
    pred = prob > 0.5
    gl, pl, gobjs, pobjs, _ = match(gt, pred, gt_label(dataset, name, gt))
    p_prob = np.bincount(pl.ravel(), weights=prob.ravel(), minlength=len(pobjs) + 1)
    p_gt = np.bincount(pl.ravel(), weights=gt.ravel(), minlength=len(pobjs) + 1)
    records = []
    for o in pobjs:
        cls = {"matched": "hit", "fp": "fp"}.get(o["class"])
        if cls is None:
            continue
        records.append({"source": cls, "label": int(cls == "hit"), "area": o["area"], "bbox": o["bbox"],
                        "det_mean_prob": float(p_prob[o["id"]] / o["area"]),
                        "gt_frac": float(p_gt[o["id"]] / o["area"])})
    ll, n = label(prob > LOW_THRESHOLD)
    touches = np.zeros(n + 1, bool)
    touches[np.unique(ll[pred])] = True
    area = np.bincount(ll.ravel(), minlength=n + 1)
    l_prob = np.bincount(ll.ravel(), weights=prob.ravel(), minlength=n + 1)
    l_gt = np.bincount(ll.ravel(), weights=gt.ravel(), minlength=n + 1)
    for k, sl in enumerate(ndimage.find_objects(ll), start=1):
        frac = float(l_gt[k] / area[k])
        if touches[k] or 0.1 <= frac < 0.5:
            continue
        records.append({"source": "lowthr_pos" if frac >= 0.5 else "lowthr_neg", "label": int(frac >= 0.5),
                        "area": int(area[k]), "bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
                        "det_mean_prob": float(l_prob[k] / area[k]), "gt_frac": frac})
    g_prob = np.bincount(gl.ravel(), weights=prob.ravel(), minlength=len(gobjs) + 1)
    for o in gobjs:
        if o["class"] == "miss" and o["coverage"] < MISS_MAX_COVERAGE.get(dataset, 1.01):
            records.append({"source": "miss", "label": 1, "area": o["area"], "bbox": o["bbox"],
                            "det_mean_prob": float(g_prob[o["id"]] / o["area"]), "gt_frac": 1.0})
    for view in ("t1", "t2") if DATASETS[dataset]["task"] == "building" else ():
        building = np.asarray(Image.open(PRED / dataset / "sam3_omniovcd" / f"{view}_mask" / name)) > 0
        bl, nb = label(building)
        area = np.bincount(bl.ravel(), minlength=nb + 1)
        b_gt = np.bincount(bl.ravel(), weights=gt.ravel(), minlength=nb + 1)
        b_prob = np.bincount(bl.ravel(), weights=prob.ravel(), minlength=nb + 1)
        hit_pred = np.zeros(nb + 1, bool)
        hit_pred[np.unique(bl[pred])] = True
        for k, sl in enumerate(ndimage.find_objects(bl), start=1):
            if hit_pred[k] or b_gt[k] / area[k] >= 0.1:
                continue
            records.append({"source": "unchanged", "label": 0, "area": int(area[k]), "view": view,
                            "bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
                            "det_mean_prob": float(b_prob[k] / area[k]), "gt_frac": float(b_gt[k] / area[k])})
    for r in records:
        r["image"] = name
    return records


def render(dataset, record, out_dir, cid):
    paths = {}
    box = crop_box(record["bbox"], load_label(dataset, record["image"]).shape, CROP_SCALE, CROP_MIN)
    left, top, right, bottom = box
    k = SIDE / (right - left)
    x0, y0, x1, y1 = record["bbox"]
    for view in ("t1", "t2"):
        rgb = load_rgb(dataset, record["image"], view)[top:bottom, left:right]
        image = Image.fromarray(rgb).resize((SIDE, SIDE), Image.Resampling.BICUBIC)
        ImageDraw.Draw(image).rectangle([(x0 - left) * k - 3, (y0 - top) * k - 3, (x1 - left) * k + 2, (y1 - top) * k + 2],
                                        outline=(255, 0, 0), width=3)
        paths[view] = str(out_dir / f"{cid}_{view}.png")
        image.save(paths[view])
    return box, paths


def random_unchanged(dataset, misses, rng, count, tries=20000):
    """Size-matched random boxes in unchanged, unpredicted area (general-change datasets)."""
    items = names(dataset)
    records = []
    for _ in range(tries):
        if len(records) >= count:
            break
        x0, y0, x1, y1 = rng.choice(misses)["bbox"]
        w, h = x1 - x0, y1 - y0
        name = rng.choice(items)
        gt = load_label(dataset, name)
        H, W = gt.shape
        if w >= W or h >= H:
            continue
        x, y = rng.randrange(0, W - w), rng.randrange(0, H - h)
        prob = load_prob(dataset, DETECTOR[dataset], name)[y:y + h, x:x + w]
        frac = float(gt[y:y + h, x:x + w].mean())
        if frac >= 0.1 or (prob > 0.5).any():
            continue
        records.append({"source": "unchanged", "label": 0, "area": int(w * h), "bbox": [x, y, x + w, y + h],
                        "det_mean_prob": float(prob.mean()), "gt_frac": frac, "image": name,
                        "random_box": True})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    for dataset in parser.parse_args().datasets:
        rng = random.Random(1016)
        pool = []
        for name in names(dataset):
            pool += region_records(dataset, name)
        if DATASETS[dataset]["task"] != "building":
            misses = [r for r in pool if r["source"] == "miss" and r["area"] >= MIN_AREA]
            pool += random_unchanged(dataset, misses, random.Random(2016), 2 * PER_SOURCE["unchanged"])
        stats = {}
        chosen = []
        for source, quota in PER_SOURCE.items():
            group = [r for r in pool if r["source"] == source]
            eligible = [r for r in group if r["area"] >= MIN_AREA]
            stats[source] = {"all": len(group), "area_ge_min": len(eligible),
                             "pixels_below_min_frac": (sum(r["area"] for r in group if r["area"] < MIN_AREA)
                                                       / max(1, sum(r["area"] for r in group))),
                             "sampled": min(quota, len(eligible))}
            chosen += rng.sample(eligible, min(quota, len(eligible)))
        out_dir = OUT / "exp3" / dataset / "inputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(OUT / "exp3" / dataset / "candidates.jsonl", "w") as f:
            for i, r in enumerate(chosen):
                r["cid"] = f"{dataset}_{i:04d}"
                r["crop"], r["inputs"] = render(dataset, r, out_dir, r["cid"])
                f.write(json.dumps(r) + "\n")
        write_json(OUT / "exp3" / dataset / "candidate_stats.json",
                   {"detector": DETECTOR[dataset], "low_threshold": LOW_THRESHOLD, "min_area": MIN_AREA,
                    "crop_scale": CROP_SCALE, "crop_min": CROP_MIN, "render_side": SIDE, "sources": stats})
        print(dataset, json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
