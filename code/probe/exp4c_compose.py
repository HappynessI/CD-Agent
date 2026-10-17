"""Experiment 4c, step 3: merge SAM3 masks of confirmed candidates into the detector mask; F1.

Reports, per dataset: original F1, rule-confirmed (no GT anywhere in the pipeline except the
4b-fitted threshold), and oracle-confirmed (gt_frac >= 0.5, isolates finding from confirming);
plus added TP / FP pixels, compared with the experiment 2 2b bound. The rule run is also scored
at stricter rule-score thresholds (only candidates above each threshold are merged), because the
4b threshold was set on a balanced sample while real candidates are mostly spurious.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import OUT, load_label, load_pred, names, pixel_metrics, write_json

DETECTOR = {"sysu": "schanger_small", "second": "mambascd_base"}
STRICT = (0.5, 0.6, 0.7, 0.8, 0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["second", "sysu"])
    args = parser.parse_args()
    for dataset in args.datasets:
        base = OUT / "exp4c" / dataset
        totals = defaultdict(lambda: np.zeros(3, np.int64))
        added = defaultdict(lambda: np.zeros(2, np.int64))
        rows = {k: [json.loads(line) for line in open(base / f"selected_{k}.jsonl")] for k in ("rule", "oracle")}
        for name in names(dataset):
            gt = load_label(dataset, name)
            pred = load_pred(dataset, DETECTOR[dataset], name)
            totals["original"] += [(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()]
            for kind in rows:
                path = base / f"sam3_{kind}" / name.replace(".png", ".npz")
                variants = [(kind, None)] + ([(f"rule>={t}", t) for t in STRICT] if kind == "rule" else [])
                z = np.load(path) if path.exists() else None
                meta = json.loads(str(z["meta"])) if z is not None else []
                for label, thr in variants:
                    new = pred.copy()
                    for m in meta:
                        r = rows[kind][m["index"]]
                        if thr is not None and r["rule_score"] < thr:
                            continue
                        left, top, right, bottom = r["crop"]
                        new[top:bottom, left:right] |= z[f"m{m['index']}"]
                    extra = new & ~pred
                    added[label] += [(extra & gt).sum(), (extra & ~gt).sum()]
                    totals[label] += [(new & gt).sum(), (new & ~gt).sum(), (~new & gt).sum()]
        report = {k: pixel_metrics(*map(int, v)) for k, v in totals.items()}
        for label in added:
            kind, thr = ("rule", float(label.split(">=")[1])) if label.startswith("rule>=") else (label, None)
            report[label]["selected_candidates"] = sum(1 for r in rows[kind] if thr is None or r["rule_score"] >= thr)
            report[label]["added_tp_px"], report[label]["added_fp_px"] = map(int, added[label])
            report[label]["delta_f1"] = report[label]["f1"] - report["original"]["f1"]
        write_json(base / "metrics_4c.json", report)
        print(dataset, {k: round(v["f1"], 4) for k, v in report.items()}, flush=True)


if __name__ == "__main__":
    main()
