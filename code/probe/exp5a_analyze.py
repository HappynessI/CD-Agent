"""Experiment 5a: coverage statistics for AnyChange (H1, H2) and SAM3-AnyChange (H3), with the
experiment 4a definitions, plus the unions requested in doc/exp5_followup.md.
Reads exp5/<dataset>/candidates.jsonl (exp5_candidates.py). Writes exp5/<dataset>/analysis_5a.json
and figures/exp5a/<dataset>/.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import exp4a_analyze as a4
from common import OUT, write_json

SOURCES = ("H1", "H2", "H3", "A", "B", "C", "D10", "E10", "G0.05")


def load(dataset):
    rows = [json.loads(line) for line in open(OUT / "exp5" / dataset / "candidates.jsonl")]
    c2 = [c["score"] for r in rows for c in r["candidates"] if c["source"] == "C2"]
    cut = float(np.quantile(c2, 0.8)) if c2 else None
    for r in rows:
        kept = []
        for c in r["candidates"]:
            if c["source"] == "C2" and c["score"] < cut:
                continue
            if c["source"] in ("C1", "C2"):
                kept.append({**c, "source": "C"})
            if c["source"] in SOURCES or c["source"] in ("C1", "C2"):
                kept.append(c)
        r["candidates"] = kept
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["second", "sysu"])
    args = parser.parse_args()
    for dataset in args.datasets:
        rows = load(dataset)
        present = [s for s in SOURCES if any(c["source"] == s for r in rows for c in r["candidates"])]
        report = {"images": len(rows), "sources": {s: a4.summarise(rows, [s]) for s in present}, "unions": {}}
        base = [s for s in ("A", "B", "C", "D10", "E10") if s in present]
        combos = [["B", "G0.05"], ["B", "G0.05", "H1"], ["B", "C", "G0.05", "H1"],
                  ["B", "G0.05", "H3"], ["B", "C", "G0.05", "H3"], ["H1", "H3"]]
        combos += [["H1", s] for s in base] + [["H3", s] for s in base]
        for combo in combos:
            if all(s in present for s in combo):
                report["unions"]["+".join(combo)] = a4.summarise(rows, combo)
        write_json(OUT / "exp5" / dataset / "analysis_5a.json", report)
        figdir = OUT / "figures" / "exp5a" / dataset
        figdir.mkdir(parents=True, exist_ok=True)
        a4.figures_for(dataset, rows, figdir, [s for s in ("H1", "H2", "H3") if s in present])
        print(dataset, {s: round(v["coverage_objects"], 3) for s, v in report["sources"].items()}, flush=True)


if __name__ == "__main__":
    main()
