"""Experiment 5c: completion variants for recall candidates (CPU, conda_envs/exp5-env for skimage).

Candidates: sources given by --sources (default B and G0.05), all test images. Masks are
regenerated exactly (exp5b_recheck.candidate_mask).
Variants scored here (pixel F1 after merging into the detector 0.5 mask):
  oracle_direct        candidates with gt_frac >= 0.5, merged as their own region (no SAM3);
  oracle_split_direct  candidates larger than 4 x the median target area are split by SLIC
                       superpixels computed on the stacked t1/t2 image (n_segments = image area /
                       median target area); pieces >= 10 px with gt_frac >= 0.5 are merged as is;
  rule_direct@k        no GT: candidates whose mean detector probability is >= the threshold that
                       gives recall k on the 4b sample (k = 0.8, 0.6, 0.4, 0.2), merged as is;
  rule_split_direct@k  no GT: the same split as oracle_split_direct, then every piece whose own
                       mean detector probability is >= that threshold is merged as is.
Also writes exp5c/<dataset>/selected_oracle_split.jsonl (piece boxes) for SAM3 box completion
(exp4c_sam3.py --base exp5c --kind oracle_split).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.segmentation import slic

from common import OUT, load_label, load_pred, load_prob, load_rgb, pixel_metrics, write_json
from exp5b_recheck import DETECTOR, candidate_mask
from render import crop_box

RECALLS = (0.8, 0.6, 0.4, 0.2)


def thresholds_from_4b(dataset, sources):
    rows = [json.loads(line) for line in open(OUT / "exp4b" / dataset / "candidates.jsonl")]
    pos = np.sort([r["det_prob"] for r in rows if r["label"] and r["source"] in sources])[::-1]
    return {k: float(pos[int(np.ceil(k * len(pos))) - 1]) for k in RECALLS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sources", nargs="+", default=["B", "G0.05"])
    parser.add_argument("--candidates", default="exp4", help="exp4 or exp5 candidates.jsonl")
    parser.add_argument("--every", type=int, default=1, help="deterministic image subset: every n-th image by name")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(OUT / args.candidates / args.dataset / "candidates.jsonl")]
    median_area = float(np.median([t["area"] for r in rows for t in r["targets"]]))   # always on all images
    rows = sorted(rows, key=lambda r: r["image"])[:: args.every]
    large = 4 * median_area
    thr = thresholds_from_4b(args.dataset, args.sources)
    base = OUT / "exp5c" / args.dataset
    base.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(lambda: np.zeros(3, np.int64))
    added = defaultdict(lambda: np.zeros(2, np.int64))
    counts = defaultdict(int)
    suffix = "" if args.every == 1 else f"_every{args.every}"
    pieces_out = open(base / f"selected_oracle_split{suffix}.jsonl", "w")
    for r in rows:
        name = r["image"]
        cands = [c for c in r["candidates"] if c["source"] in args.sources]
        gt = load_label(args.dataset, name)
        pred = load_pred(args.dataset, DETECTOR[args.dataset], name)
        totals["original"] += [(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()]
        prob = load_prob(args.dataset, DETECTOR[args.dataset], name)
        merged = {k: np.zeros_like(gt) for k in ["oracle_direct", "oracle_split_direct"]
                  + [f"rule_direct@{k}" for k in RECALLS] + [f"rule_split_direct@{k}" for k in RECALLS]}
        cache, segments = {}, None
        for c in cands:
            mask = candidate_mask(args.dataset, name, c["source"], c["bbox"], cache)
            if c["gt_frac"] >= 0.5:
                merged["oracle_direct"] |= mask
                counts["oracle_direct"] += 1
            for k in RECALLS:
                if c["det_prob"] >= thr[k]:
                    merged[f"rule_direct@{k}"] |= mask
                    counts[f"rule_direct@{k}"] += 1
            if c["area"] > large:
                if segments is None:
                    stack = np.concatenate([load_rgb(args.dataset, name, v) for v in ("t1", "t2")], axis=2)
                    n_seg = max(4, int(gt.size / median_area))
                    segments = slic(stack, n_segments=n_seg, compactness=10, channel_axis=-1, start_label=1)
                ids = np.unique(segments[mask])
                pieces = [mask & (segments == i) for i in ids]
            else:
                pieces = [mask]
            for p in pieces:
                area = int(p.sum())
                if area < 10:
                    continue
                piece_prob = prob[p].mean()
                for k in RECALLS:
                    if piece_prob >= thr[k]:
                        merged[f"rule_split_direct@{k}"] |= p
                        counts[f"rule_split_direct@{k}"] += 1
                if gt[p].mean() < 0.5:
                    continue
                merged["oracle_split_direct"] |= p
                counts["oracle_split_direct"] += 1
                ys, xs = np.nonzero(p)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                pieces_out.write(json.dumps({"image": name, "source": c["source"], "bbox": bbox,
                                             "crop": list(crop_box(bbox, gt.shape, 3.0, 128)),
                                             "views": ["t1", "t2"], "rule_score": c["det_prob"],
                                             "gt_frac": float(gt[p].mean())}) + "\n")
        for k, m in merged.items():
            new = pred | m
            extra = new & ~pred
            added[k] += [(extra & gt).sum(), (extra & ~gt).sum()]
            totals[k] += [(new & gt).sum(), (new & ~gt).sum(), (~new & gt).sum()]
    pieces_out.close()
    report = {k: pixel_metrics(*map(int, v)) for k, v in totals.items()}
    for k in added:
        report[k]["delta_f1"] = report[k]["f1"] - report["original"]["f1"]
        report[k]["added_tp_px"], report[k]["added_fp_px"] = map(int, added[k])
        report[k]["merged_regions"] = counts[k]
    report["config"] = {"every": args.every, "images": len(rows), "sources": args.sources, "candidates": args.candidates, "median_target_area": median_area,
                        "large_area": large, "slic_compactness": 10, "rule_signal": "det_prob",
                        "rule_thresholds_by_4b_recall": thr}
    write_json(base / f"metrics_5c_{'_'.join(args.sources)}{suffix}.json", report)
    print(args.dataset, {k: round(v["f1"], 4) for k, v in report.items() if k != "config"}, flush=True)


if __name__ == "__main__":
    main()
