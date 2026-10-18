"""Experiment 5b: compare the new confirmation signals (RemoteCLIP, detector local re-check) with
the experiment 4b signals on the same candidates (exp4b/<dataset>, plus exp5b/<dataset> for H*).

Metrics as 4b (AUC with bootstrap CI, false-positive rate at recall 0.8 / 0.9 on the balanced
sample) plus a pool-weighted precision: each sampled candidate is weighted by
(pool count / sampled count) of its (source, label) cell, using the real pool sizes recorded in
sample_stats.json; the threshold is set at weighted recall 0.8 and the weighted share of true
changes among accepted candidates is reported. Logistic regression (numpy IRLS, 5-fold CV) is run
on the 4b features, the new features, and both.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

import exp4b_analyze as b4
from common import OUT, write_json
from exp3_analyze import auc, auc_ci

NEW = ["clip_dissim_ctx", "clip_cls_diff_ctx", "clip_dist_change_ctx", "clip_dissim_tight", "clip_cls_diff_tight",
       "clip_dist_change_tight", "recheck_zoom", "recheck_native"]


def load(subdir, dataset):
    base = OUT / subdir / dataset
    rows = [json.loads(line) for line in open(base / "candidates.jsonl")]
    for fname in ("clip.jsonl", "recheck.jsonl"):
        extra = {x["cid"]: x for x in map(json.loads, open(base / fname))}
        for r in rows:
            r.update({k: v for k, v in extra[r["cid"]].items() if k != "cid"})
    for tag, key in (("qwen35_4b", "mllm_qwen35"), ("qwen37plus", "mllm_qwen37plus")):
        path = base / f"mllm_{tag}.jsonl"
        if path.exists():
            m = {x["cid"]: x for x in map(json.loads, open(path))}
            for r in rows:
                r[key] = m[r["cid"]]["p_yes"]
    for r in rows:
        r["B_disagree"] = 1 - r["B_agree"]
        if "A_agree" in r:
            r["A_disagree"] = 1 - r["A_agree"]
    stats = json.load(open(base / "sample_stats.json"))["per_source"]
    for r in rows:
        s = stats[r["source"]]
        r["_w"] = (s["available_pos"] / s["sampled_pos"]) if r["label"] else (s["available_neg"] / s["sampled_neg"])
    return rows


def weighted_precision_at_recall(rows, scores, recall=0.8):
    y = np.array([r["label"] for r in rows])
    w = np.array([r["_w"] for r in rows])
    s = np.asarray(scores, float)
    order = np.argsort(-s)
    cum_pos = np.cumsum((w * y)[order])
    k = int(np.searchsorted(cum_pos, recall * cum_pos[-1]))
    accepted = order[: k + 1]
    return float((w[accepted] * y[accepted]).sum() / w[accepted].sum())


def evaluate(rows, signals):
    y = [r["label"] for r in rows]
    rng = np.random.default_rng(1016)
    out = {}
    for name, vals in signals.items():
        vals = np.asarray(vals, float)
        entry = {"auc": auc(y, vals), "auc_ci95": auc_ci(y, vals, rng)}
        for rec in (0.8, 0.9):
            entry[f"fpr_at_recall_{rec}"], _ = b4.fpr_at_recall(y, vals, rec)
        entry["pool_precision_at_recall_0.8"] = weighted_precision_at_recall(rows, vals)
        out[name] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["second", "sysu"])
    parser.add_argument("--subdir", default="exp4b")
    args = parser.parse_args()
    for dataset in args.datasets:
        rows = load(args.subdir, dataset)
        old = [f for f in b4.BASE_FEATURES if any(f in r for r in rows)]
        mllm = [k for k in ("mllm_qwen35", "mllm_qwen37plus") if all(r.get(k) is not None for r in rows)]
        signals = {f: [r.get(f, np.nan) for r in rows] for f in old + NEW + mllm}
        signals = {k: np.where(np.isnan(np.asarray(v, float)), np.nanmean(np.asarray(v, float)), v) for k, v in signals.items()}
        signals["logreg_4b"] = b4.cv_scores(rows, old)
        signals["logreg_new_only"] = b4.cv_scores(rows, NEW)
        signals["logreg_4b+new"] = b4.cv_scores(rows, old + NEW)
        if mllm:
            signals["logreg_all+mllm"] = b4.cv_scores(rows, old + NEW + mllm)
        report = {"n": len(rows), "positives": int(sum(r["label"] for r in rows)), "signals": evaluate(rows, signals),
                  "by_source": {}}
        for s in sorted({r["source"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["source"] == s]
            sub = [rows[i] for i in idx]
            report["by_source"][s] = {k: auc([r["label"] for r in sub], np.asarray(v)[idx])
                                      for k, v in signals.items() if not k.startswith("logreg")}
        # Rule for 5c: logistic regression on the best feature set, fit on all samples.
        best_set = max(("logreg_4b", "logreg_new_only", "logreg_4b+new"), key=lambda k: report["signals"][k]["auc"])
        feats = {"logreg_4b": old, "logreg_new_only": NEW, "logreg_4b+new": old + NEW}[best_set]
        X, fill = b4.matrix(rows, feats)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        w = b4.fit_logreg((X - mu) / sd, np.array([r["label"] for r in rows]))
        _, thr = b4.fpr_at_recall([r["label"] for r in rows], signals[best_set], 0.8)
        report["rule_5c"] = {"name": best_set, "features": feats, "weights": w.tolist(), "mean": mu.tolist(),
                             "std": sd.tolist(), "fill": fill.tolist(), "threshold": thr}
        write_json(OUT / args.subdir / dataset / "analysis_5b.json", report)
        best = max(report["signals"], key=lambda k: report["signals"][k]["auc"])
        vals = np.asarray(signals[best], float)
        _, t = b4.fpr_at_recall([r["label"] for r in rows], vals, 0.8)
        wrong = [i for i, r in enumerate(rows) if (vals[i] >= t) != bool(r["label"])]
        out = OUT / "exp5b_error_examples" / dataset
        out.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(random.Random(1016).sample(wrong, min(10, len(wrong)))):
            b4.error_panel(rows[i], best, vals[i], t).save(out / f"{k}_{rows[i]['cid']}.png")
        print(dataset, "best", best, {k: round(v["auc"], 3) for k, v in report["signals"].items()}, flush=True)


if __name__ == "__main__":
    main()
