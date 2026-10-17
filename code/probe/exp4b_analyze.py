"""Experiment 4b, step 2: which signal separates real from spurious changes on real candidates.

Signals per candidate: detector mean probability; source score (percentile within its source);
two-date semantic consistency (A: MambaSCD heads, SECOND only; B: SAM3 text-prompt maps) as
"disagreement" = 1 - agreement and the change/confidence maps; SAM3 feature distance (D);
exemplar similarity (E, where available); MLLM P(yes) for Qwen3.5-4B and Bailian qwen3.7-plus.
Combination: L2-regularised logistic regression (numpy IRLS), 5-fold stratified CV, with and
without MLLM features. Metrics: AUC with 2000-sample bootstrap CI, and false-positive rate at
recall 0.8 / 0.9. Also saves a no-MLLM rule (fit on all samples, threshold at CV recall 0.8)
for exp4c, and 10 error examples of the best signal.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw

from common import OUT, write_json
from exp3_analyze import auc, auc_ci

TAGS = {"mllm_qwen35": "qwen35_4b", "mllm_qwen37plus": "qwen37plus"}
BASE_FEATURES = ["det_prob", "score_pct", "D_dist", "B_disagree", "B_conf", "A_disagree", "A_change", "E_sim"]


def fpr_at_recall(y, s, recall):
    y, s = np.asarray(y), np.asarray(s, float)
    pos = np.sort(s[y == 1])[::-1]
    if not len(pos):
        return None, None
    thr = pos[int(np.ceil(recall * len(pos))) - 1]
    return float((s[y == 0] >= thr).mean()), float(thr)


def fit_logreg(X, y, lam=1.0, iters=50):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = lam * np.eye(Xb.shape[1])
    reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) + reg @ w
        hess = Xb.T @ (Xb * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.hstack([X, np.ones((len(X), 1))]) @ w))


def matrix(rows, features, fill=None):
    X = np.array([[r.get(f, np.nan) for f in features] for r in rows], float)
    fill = np.nanmean(X, 0) if fill is None else fill
    fill = np.where(np.isnan(fill), 0, fill)
    X = np.where(np.isnan(X), fill, X)
    return X, fill


def cv_scores(rows, features, folds=5, seed=1016):
    y = np.array([r["label"] for r in rows])
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    fold_of = {}
    for cls in (0, 1):
        idx = [i for i in order if y[i] == cls]
        for k, i in enumerate(idx):
            fold_of[i] = k % folds
    out = np.zeros(len(rows))
    for k in range(folds):
        tr = [i for i in range(len(rows)) if fold_of[i] != k]
        te = [i for i in range(len(rows)) if fold_of[i] == k]
        Xtr, fill = matrix([rows[i] for i in tr], features)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        w = fit_logreg((Xtr - mu) / sd, y[tr])
        Xte, _ = matrix([rows[i] for i in te], features, fill)
        out[te] = predict(w, (Xte - mu) / sd)
    return out


def error_panel(r, signal, value, thr):
    canvas = Image.new("RGB", (2 * 448 + 10, 448 + 50), "white")
    for i, view in enumerate(("t1", "t2")):
        canvas.paste(Image.open(r["inputs"][view]), (i * 458, 50))
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), f"{r['cid']} {r['image']} source={r['source']} GT={'change' if r['label'] else 'no change'} "
                   f"gt_frac={r['gt_frac']:.2f} area={r['area']}", fill="black")
    d.text((4, 22), f"{signal}={value:.3f} (threshold {thr:.3f} at recall 0.8) det_prob={r['det_prob']:.3f}", fill="red")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["second", "sysu"])
    args = parser.parse_args()
    for dataset in args.datasets:
        base = OUT / "exp4b" / dataset
        rows = [json.loads(line) for line in open(base / "candidates.jsonl")]
        for key, tag in TAGS.items():
            path = base / f"mllm_{tag}.jsonl"
            if path.exists():
                m = {x["cid"]: x for x in map(json.loads, open(path))}
                for r in rows:
                    r[key] = m[r["cid"]]["p_yes"]
        for r in rows:
            r["B_disagree"] = 1 - r["B_agree"]
            if "A_agree" in r:
                r["A_disagree"] = 1 - r["A_agree"]
        y = [r["label"] for r in rows]
        rng = np.random.default_rng(1016)
        base_feats = [f for f in BASE_FEATURES if any(f in r for r in rows)]
        mllm_feats = [k for k in TAGS if all(r.get(k) is not None for r in rows)]
        signals = {f: [r.get(f, np.nan) for r in rows] for f in base_feats + mllm_feats}
        signals["logreg_no_mllm"] = cv_scores(rows, base_feats)
        if mllm_feats:
            signals["logreg_with_mllm"] = cv_scores(rows, base_feats + mllm_feats)
        report = {"n": len(rows), "positives": int(sum(y)), "features": base_feats, "mllm": mllm_feats,
                  "by_source": {}, "signals": {}}
        for s in sorted({r["source"] for r in rows}):
            report["by_source"][s] = {"n": sum(r["source"] == s for r in rows),
                                      "positives": sum(r["label"] for r in rows if r["source"] == s)}
        for name, vals in signals.items():
            vals = np.asarray(vals, float)
            ok = ~np.isnan(vals)
            yy = np.asarray(y)[ok]
            entry = {"n": int(ok.sum()), "auc": auc(yy, vals[ok]),
                     "auc_ci95": auc_ci(yy, vals[ok], rng) if 0 < yy.sum() < len(yy) else None}
            for rec in (0.8, 0.9):
                entry[f"fpr_at_recall_{rec}"], entry[f"threshold_at_recall_{rec}"] = fpr_at_recall(yy, vals[ok], rec)
            report["signals"][name] = entry
        # Rule for 4c: no-MLLM logistic regression fit on all samples; threshold = CV threshold at recall 0.8.
        X, fill = matrix(rows, base_feats)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        w = fit_logreg((X - mu) / sd, np.asarray(y))
        report["rule_4c"] = {"features": base_feats, "weights": w.tolist(), "mean": mu.tolist(), "std": sd.tolist(),
                             "fill": fill.tolist(), "threshold": report["signals"]["logreg_no_mllm"]["threshold_at_recall_0.8"]}
        write_json(base / "analysis_4b.json", report)
        # Error examples for the best single or combined signal.
        best = max(report["signals"], key=lambda k: report["signals"][k]["auc"] or 0)
        thr = report["signals"][best]["threshold_at_recall_0.8"]
        vals = np.asarray(signals[best], float)
        wrong = [i for i, r in enumerate(rows) if not np.isnan(vals[i]) and (vals[i] >= thr) != bool(r["label"])]
        out = OUT / "exp4b_error_examples" / dataset
        out.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(random.Random(1016).sample(wrong, min(10, len(wrong)))):
            error_panel(rows[i], best, vals[i], thr).save(out / f"{k}_{rows[i]['cid']}.png")
        print(dataset, "best", best, {k: round(v["auc"], 3) for k, v in report["signals"].items() if v["auc"]}, flush=True)


if __name__ == "__main__":
    main()
