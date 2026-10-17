"""Experiment 4a, step 2: build recall candidates from sources A-G and measure what they cover.

Task: doc/exp4_recall_candidates.md. Detector: MambaSCD-Base (SECOND), SChanger-small (SYSU).
Target set M: GT objects (SECOND split by class transition) with detector coverage < 10% and
area >= 20 px ("no-signal misses"). M_partial: GT pixels of partial-hit objects not predicted.

A   (SECOND only) MambaSCD semantic heads: argmax over classes 1-6 differs between dates;
    score = mean(1 - sum_c p1_c p2_c) with p renormalised over classes 1-6.
B   SAM3 text-prompt land-cover maps (exp4_sam3_extract): both dates labelled and different;
    score = mean(conf_t1 * conf_t2).
C1  SAM3 "building" instances whose best cross-date IoU < 0.5 (t2: new, t1: demolished);
    score = 1 - best IoU.
C2  matched building pairs (IoU >= 0.5), region = pair union; score = mean SAM3 feature
    distance inside. Only the top 20% by score over the whole dataset are kept (no GT used),
    applied in exp4_analyze.
D   SAM3 feature cosine distance, per-image quantile thresholds (D5/D10/D20 = top 5/10/20%
    of pixels); score = mean distance.
E   exemplar similarity (top 10% of pixels in images that have exemplars); score = mean sim.
F   64 px grid, stride 32; score = mean detector probability.
G   detector probability > 0.05 / 0.1 / 0.2 components; score = mean probability.
Every candidate must overlap the detector 0.5 prediction by <= 50% of its area and have
area >= 10 px. GT is read only for the coverage/purity fields, never to form candidates.
Run in conda_envs/vlm-vs-onejev-env (CPU).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image
from scipy import ndimage

from common import OUT, WORKSPACE, area_bucket, load_label, load_prob, names, write_json
from objects import gt_label, label, match

DETECTOR = {"sysu": "schanger_small", "second": "mambascd_base"}
MAMBA_SEM = WORKSPACE / "Vlm_vs_OneJev/outputs/formal/SECOND/probabilities"
MIN_TARGET, MIN_CAND = 20, 10
MAX_PRED_FRAC = 0.5
COVER_FRAC, MAX_AREA_RATIO = 0.5, 4.0
D_QUANTILES = {"D5": 0.95, "D10": 0.90, "D20": 0.80}
E_QUANTILE = 0.90
G_THRESHOLDS = {"G0.05": 0.05, "G0.1": 0.1, "G0.2": 0.2}
GRID, STRIDE = 64, 32
# SECOND classes: 1 low vegetation, 2 non-vegetated surface, 3 tree, 4 water, 5 building, 6 playground.


def transition_group(t):
    c1, c2 = t
    if c1 == 5 and c2 == 5:
        return "rebuilt (building->building)"
    if c2 == 5:
        return "new building"
    if c1 == 5:
        return "demolished building"
    if {c1, c2} == {1, 3}:
        return "vegetation <-> vegetation"
    if {c1, c2} <= {1, 2, 3} and 2 in (c1, c2):
        return "surface <-> vegetation"
    if 4 in (c1, c2):
        return "water related"
    return "other"


def upsample(grid_map, shape):
    image = Image.fromarray(np.asarray(grid_map, np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def region_stats(mask, maps):
    return {k: float(v[mask].mean()) for k, v in maps.items()}


def process(name, dataset):
    gt = load_label(dataset, name)
    shape = gt.shape
    prob = load_prob(dataset, DETECTOR[dataset], name)
    pred = prob > 0.5
    gt_labels = gt_label(dataset, name, gt)
    gl, _, gobjs, _, _ = match(gt, pred, gt_labels)
    transitions = gt_labels[2]
    undetected = gt & ~pred
    # Targets M and M_partial.
    tmap = np.zeros(shape, np.int32)
    targets = []
    for o in gobjs:
        if o["coverage"] < 0.1 and o["area"] >= MIN_TARGET:
            x0, y0, x1, y1 = o["bbox"]
            t = {"tid": len(targets) + 1, "area": o["area"], "bbox": o["bbox"],
                 "bbox_area": (x1 - x0) * (y1 - y0), "bucket": area_bucket(o["area"])}
            if transitions:
                t["transition"] = transitions[o["id"]]
                t["group"] = transition_group(transitions[o["id"]])
            tmap[gl == o["id"]] = t["tid"]
            targets.append(t)
    partial_ids = [o["id"] for o in gobjs if o["class"] == "partial"]
    m_partial = np.isin(gl, partial_ids) & ~pred if partial_ids else np.zeros(shape, bool)
    t_area = np.array([0] + [t["area"] for t in targets])
    t_box = np.array([0] + [t["bbox_area"] for t in targets])

    # Region maps for the 4b signals.
    z = np.load(OUT / "exp4" / dataset / "sam3" / name.replace(".png", ".npz"))
    dist = upsample(z["dist"], shape)
    maps = {"det_prob": prob, "D_dist": dist,
            "B_agree": ((z["cls_t1"] == z["cls_t2"]) & (z["cls_t1"] > 0)).astype(np.float32),
            "B_conf": z["conf_t1"].astype(np.float32) * z["conf_t2"].astype(np.float32)}
    esim = upsample(z["esim"], shape) if z["esim"].size else None
    if esim is not None:
        maps["E_sim"] = esim
    a_change = None
    if dataset == "second":
        p1 = np.load(MAMBA_SEM / name.replace(".png", "_T1.npy"))[1:]
        p2 = np.load(MAMBA_SEM / name.replace(".png", "_T2.npy"))[1:]
        p1 = p1 / p1.sum(0, keepdims=True)
        p2 = p2 / p2.sum(0, keepdims=True)
        a1, a2 = p1.argmax(0), p2.argmax(0)
        a_change = 1 - (p1 * p2).sum(0)
        maps["A_agree"] = (a1 == a2).astype(np.float32)
        maps["A_change"] = a_change

    candidates = []
    unions = {}

    def add(source, mask, score, extra=None):
        area = int(mask.sum())
        if area < MIN_CAND:
            return
        pred_frac = float(pred[mask].mean())
        if pred_frac > MAX_PRED_FRAC:
            return
        unions[source] = unions.get(source, np.zeros(shape, bool)) | mask
        ys, xs = np.nonzero(mask)
        inter = np.bincount(tmap[mask], minlength=len(targets) + 1)
        loose = inter >= COVER_FRAC * t_area
        loose[0] = False
        ok = loose & (area <= MAX_AREA_RATIO * t_box)
        candidates.append({
            "source": source, "score": float(score), "area": area,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "pred_frac": pred_frac, "undetected_frac": float(undetected[mask].mean()),
            "gt_frac": float(gt[mask].mean()), "covers": np.nonzero(ok)[0].tolist(),
            "covers_loose": np.nonzero(loose)[0].tolist(),
            "partial_px": int(m_partial[mask].sum()), **region_stats(mask, maps), **(extra or {})})

    def add_components(source, binary, score_map):
        labels, n = label(binary)
        for k, sl in enumerate(ndimage.find_objects(labels), start=1):
            local = labels[sl] == k
            mask = np.zeros(shape, bool)
            mask[sl] = local
            add(source, mask, score_map[sl][local].mean())

    if a_change is not None:
        add_components("A", a1 != a2, a_change)
    add_components("B", (z["cls_t1"] > 0) & (z["cls_t2"] > 0) & (z["cls_t1"] != z["cls_t2"]), maps["B_conf"])
    # C: building instances.
    inst = {v: z[f"inst_{v}"].astype(np.int32) for v in ("t1", "t2")}
    counts = {v: len(z[f"inst_scores_{v}"]) for v in ("t1", "t2")}
    area = {v: np.bincount(inst[v].ravel(), minlength=counts[v] + 1) for v in inst}
    both = (inst["t1"] > 0) & (inst["t2"] > 0)
    pairs, inter = np.unique(inst["t1"][both] * (counts["t2"] + 1) + inst["t2"][both], return_counts=True)
    iou = {}
    for key, n in zip(pairs, inter):
        i, j = divmod(int(key), counts["t2"] + 1)
        iou[(i, j)] = n / (area["t1"][i] + area["t2"][j] - n)
    for view, other in (("t1", "t2"), ("t2", "t1")):
        for k in range(1, counts[view] + 1):
            if area[view][k] == 0:
                continue
            best = max([v for (i, j), v in iou.items() if (i if view == "t1" else j) == k], default=0.0)
            if best < 0.5:
                add("C1", inst[view] == k, 1 - best, {"view": view})
    for (i, j), v in iou.items():
        if v >= 0.5:
            mask = (inst["t1"] == i) | (inst["t2"] == j)
            add("C2", mask, dist[mask].mean(), {"pair_iou": float(v)})
    for source, q in D_QUANTILES.items():
        add_components(source, dist >= np.quantile(dist, q), dist)
    if esim is not None:
        add_components("E10", esim >= np.quantile(esim, E_QUANTILE), esim)
    h, w = shape
    for y in range(0, h - GRID + 1, STRIDE):
        for x in range(0, w - GRID + 1, STRIDE):
            mask = np.zeros(shape, bool)
            mask[y:y + GRID, x:x + GRID] = True
            add("F", mask, prob[y:y + GRID, x:x + GRID].mean())
    for source, t in G_THRESHOLDS.items():
        add_components(source, prob > t, prob)
    union_stats = {src: {"union_px": int(u.sum()), "partial_px": int((u & m_partial).sum())}
                   for src, u in unions.items()}
    return {"image": name, "targets": targets, "partial_px_total": int(m_partial.sum()), "unions": union_stats,
            "has_exemplar": esim is not None, "image_px": int(h * w), "candidates": candidates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DETECTOR), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    items = names(args.dataset)[: args.limit or None]
    out = OUT / "exp4" / args.dataset / "candidates.jsonl"
    with ProcessPoolExecutor(args.workers) as pool, open(out, "w") as f:
        for record in pool.map(partial(process, dataset=args.dataset), items, chunksize=4):
            f.write(json.dumps(record) + "\n")
    write_json(OUT / "exp4" / args.dataset / "candidates_config.json", {
        "detector": DETECTOR[args.dataset], "min_target_area": MIN_TARGET, "min_candidate_area": MIN_CAND,
        "max_pred_frac": MAX_PRED_FRAC, "cover_frac": COVER_FRAC, "max_area_ratio": MAX_AREA_RATIO,
        "d_quantiles": D_QUANTILES, "e_quantile": E_QUANTILE, "g_thresholds": G_THRESHOLDS,
        "grid": GRID, "stride": STRIDE, "images": len(items)})
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
