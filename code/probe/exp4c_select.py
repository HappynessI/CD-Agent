"""Experiment 4c, step 1: apply the 4b confirmation rule to every candidate of the chosen sources.

Rule (exp4b analysis_4b.json "rule_4c"): no-MLLM logistic regression on region signals, threshold
set on CV predictions to recall about 0.8. Overlapping candidates (box IoU > 0.5) keep the one
with the higher rule score. Also writes the GT-oracle selection (gt_frac >= 0.5) as a bound that
isolates "finding" from "confirming". View for SAM3: C1 instances know their date; others "both".
Writes exp4c/<dataset>/selected_{rule,oracle}.jsonl.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import OUT, load_label, write_json
from exp4a_analyze import load
from exp4b_analyze import matrix, predict
from render import crop_box

CROP_SCALE, CROP_MIN = 3.0, 128     # as experiment 2


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def dedupe(cands):
    kept = []
    for c in sorted(cands, key=lambda c: -c["rule_score"]):
        if all(box_iou(c["bbox"], k["bbox"]) <= 0.5 for k in kept):
            kept.append(c)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args()
    rule = json.load(open(OUT / "exp4b" / args.dataset / "analysis_4b.json"))["rule_4c"]
    w, mu, sd, fill = (np.asarray(rule[k]) for k in ("weights", "mean", "std", "fill"))
    rows, _ = load(args.dataset)
    base = OUT / "exp4c" / args.dataset
    base.mkdir(parents=True, exist_ok=True)
    stats = {"rule_threshold": rule["threshold"], "sources": args.sources}
    outs = {k: open(base / f"selected_{k}.jsonl", "w") for k in ("rule", "oracle")}
    counts = {k: 0 for k in outs}
    for r in rows:
        cands = [c for c in r["candidates"] if c["source"] in args.sources]
        if not cands:
            continue
        for c in cands:
            c["B_disagree"] = 1 - c["B_agree"]
            if "A_agree" in c:
                c["A_disagree"] = 1 - c["A_agree"]
        X, _ = matrix(cands, rule["features"], fill)
        scores = predict(w, (X - mu) / sd)
        for c, s in zip(cands, scores):
            c["rule_score"] = float(s)
        shape = load_label(args.dataset, r["image"]).shape
        for kind, keep in (("rule", lambda c: c["rule_score"] >= rule["threshold"]),
                           ("oracle", lambda c: c["gt_frac"] >= 0.5)):
            for c in dedupe([c for c in cands if keep(c)]):
                view = [c["view"]] if c.get("view") else ["t1", "t2"]
                outs[kind].write(json.dumps({"image": r["image"], "source": c["source"], "bbox": c["bbox"],
                                             "crop": list(crop_box(c["bbox"], shape, CROP_SCALE, CROP_MIN)),
                                             "views": view, "rule_score": c["rule_score"],
                                             "gt_frac": c["gt_frac"]}) + "\n")
                counts[kind] += 1
    for f in outs.values():
        f.close()
    stats["selected"] = counts
    write_json(base / "select_stats.json", stats)
    print(args.dataset, stats, flush=True)


if __name__ == "__main__":
    main()
