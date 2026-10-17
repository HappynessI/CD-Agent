"""Experiment 4b, step 1: sample real recall candidates for confirmation and render MLLM inputs.

Pool: candidates of the chosen sources (from exp4/<dataset>/candidates.jsonl, C2 filtered as in
exp4a_analyze). Label by GT only for evaluation: positive if >= 50% of the region is GT change,
negative if < 10%, otherwise dropped. Sample up to 150 positives and fill to ~300 with
negatives, split evenly across sources. Each candidate keeps its region signals and its score
percentile within its source (scores of different sources are not on one scale). Crops are
rendered exactly as in experiment 3 (exp3_candidates.render).
Writes exp4b/<dataset>/candidates.jsonl and inputs/.
"""
from __future__ import annotations

import argparse
import json
import random
from bisect import bisect_left

from common import OUT, write_json
from exp3_candidates import render
from exp4a_analyze import load

TARGET, MAX_POS = 300, 150


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    args = parser.parse_args()
    rows, _ = load(args.dataset)
    rng = random.Random(1016)
    pools = {s: {"pos": [], "neg": []} for s in args.sources}
    scores = {s: [] for s in args.sources}
    for r in rows:
        for c in r["candidates"]:
            if c["source"] not in pools:
                continue
            scores[c["source"]].append(c["score"])
            item = {**c, "image": r["image"]}
            if c["gt_frac"] >= 0.5:
                pools[c["source"]]["pos"].append(item)
            elif c["gt_frac"] < 0.1:
                pools[c["source"]]["neg"].append(item)
    for s in scores:
        scores[s].sort()
    per_source = TARGET // len(args.sources)
    chosen, stats = [], {}
    for s, pool in pools.items():
        n_pos = min(MAX_POS // len(args.sources), len(pool["pos"]))
        n_neg = min(per_source - n_pos, len(pool["neg"]))
        stats[s] = {"available_pos": len(pool["pos"]), "available_neg": len(pool["neg"]),
                    "sampled_pos": n_pos, "sampled_neg": n_neg}
        for kind, n in (("pos", n_pos), ("neg", n_neg)):
            for c in rng.sample(pool[kind], n):
                c["label"] = int(kind == "pos")
                c["score_pct"] = bisect_left(scores[s], c["score"]) / max(1, len(scores[s]))
                chosen.append(c)
    base = OUT / "exp4b" / args.dataset
    (base / "inputs").mkdir(parents=True, exist_ok=True)
    with open(base / "candidates.jsonl", "w") as f:
        for i, c in enumerate(chosen):
            c["cid"] = f"{args.dataset}_4b_{i:04d}"
            c["crop"], c["inputs"] = render(args.dataset, c, base / "inputs", c["cid"])
            c["det_mean_prob"] = c["det_prob"]
            f.write(json.dumps(c) + "\n")
    write_json(base / "sample_stats.json", {"sources": args.sources, "per_source": stats})
    print(args.dataset, json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
