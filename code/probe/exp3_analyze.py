"""Experiment 3, step 3: AUC / accuracy of MLLM readouts vs detector confidence.

Scores compared per dataset, on the main candidate set and within each subset:
  mllm_p_yes   token probability of "yes";
  mllm_verbal  confidence if answer is yes else 100 - confidence (unparsed -> 50);
  detector     SChanger-base mean probability over the region.
"detector-positive" = hit vs fp (both predicted at 0.5); "low-threshold" = lowthr_pos vs
lowthr_neg (both below 0.5); "recall-side" = missed GT objects vs unchanged SAM3
buildings (supplement). Accuracy uses fixed thresholds: p_yes > 0.5, the generated
yes/no answer, detector mean prob > 0.5. 95% CI from 2000 bootstrap resamples.
Writes exp3/<dataset>/analysis_<tag>.json and 10 wrong-answer panels under errors_<tag>/.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from PIL import Image, ImageDraw

from common import DATASETS, OUT, write_json

SUBSETS = {"main": ("hit", "fp", "lowthr_pos", "lowthr_neg"), "detector-positive": ("hit", "fp"),
           "low-threshold": ("lowthr_pos", "lowthr_neg"), "recall-side (supplement)": ("miss", "unchanged")}


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def auc_ci(labels, scores, rng, n=2000):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        value = auc(labels[idx], scores[idx])
        if value is not None:
            values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    for dataset in DATASETS:
        base = OUT / "exp3" / dataset
        if not (base / f"mllm_{args.tag}.jsonl").exists():
            continue
        cands = {r["cid"]: r for r in map(json.loads, open(base / "candidates.jsonl"))}
        rows = []
        for m in map(json.loads, open(base / f"mllm_{args.tag}.jsonl")):
            r = {**cands[m["cid"]], **m}
            r["verbal"] = 50.0 if r["answer"] is None or r["confidence"] is None else \
                (r["confidence"] if r["answer"] == "yes" else 100 - r["confidence"])
            rows.append(r)
        rng = np.random.default_rng(1016)
        has_p = all(r["p_yes"] is not None for r in rows)
        masses = [r["yes_no_mass"] for r in rows if r["yes_no_mass"] is not None]
        report = {"tag": args.tag, "n": len(rows), "parsed": sum(r["answer"] is not None for r in rows),
                  "errors": sum("error" in r for r in rows), "p_yes_available": has_p,
                  "mean_yes_no_mass": float(np.mean(masses)) if masses else None, "subsets": {}}
        for subset, sources in SUBSETS.items():
            group = [r for r in rows if r["source"] in sources]
            y = [r["label"] for r in group]
            entry = {"n": len(group), "positives": int(sum(y))}
            for key, score in (("mllm_p_yes", "p_yes"), ("mllm_verbal", "verbal"), ("detector", "det_mean_prob")):
                if score == "p_yes" and not has_p:
                    entry[key] = {"auc": None, "auc_ci95": None}
                    continue
                s = [r[score] for r in group]
                entry[key] = {"auc": auc(y, s), "auc_ci95": auc_ci(y, s, rng) if 0 < sum(y) < len(y) else None}
            entry["acc_mllm_p_yes"] = float(np.mean([(r["p_yes"] > 0.5) == r["label"] for r in group])) if has_p else None
            entry["acc_mllm_answer"] = float(np.mean([(r["answer"] == "yes") == r["label"] for r in group]))
            entry["acc_detector_0.5"] = float(np.mean([(r["det_mean_prob"] > 0.5) == r["label"] for r in group]))
            entry["mllm_yes_rate"] = float(np.mean([r["answer"] == "yes" for r in group]))
            entry["per_source_yes_rate"] = {src: float(np.mean([r["answer"] == "yes" for r in group if r["source"] == src]))
                                            for src in sorted({r["source"] for r in group})}
            report["subsets"][subset] = entry
        write_json(base / f"analysis_{args.tag}.json", report)
        print(dataset, json.dumps({k: (v["mllm_p_yes"]["auc"], v["detector"]["auc"]) for k, v in report["subsets"].items()}))
        wrong = [r for r in rows if r["source"] in SUBSETS["main"] and (r["answer"] == "yes") != bool(r["label"])]
        out = base / f"errors_{args.tag}"
        out.mkdir(exist_ok=True)
        for k, r in enumerate(random.Random(1016).sample(wrong, min(10, len(wrong)))):
            canvas = Image.new("RGB", (2 * 448 + 10, 448 + 50), "white")
            for i, view in enumerate(("t1", "t2")):
                canvas.paste(Image.open(r["inputs"][view]), (i * 458, 50))
            d = ImageDraw.Draw(canvas)
            d.text((4, 4), f"{r['cid']} {r['image']} source={r['source']} GT={'change' if r['label'] else 'no change'} "
                           f"area={r['area']} det_prob={r['det_mean_prob']:.2f}", fill="black")
            p_text = "n/a" if r["p_yes"] is None else f"{r['p_yes']:.2f}"
            d.text((4, 22), f"MLLM: {r['raw'][:120]}  p_yes={p_text}", fill="red")
            canvas.save(out / f"{k}_{r['cid']}.png")


if __name__ == "__main__":
    main()
