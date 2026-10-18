"""Experiment 4a, step 3: coverage, cost and purity of each candidate source; tables + figures.

Reads exp4/<dataset>/candidates.jsonl (exp4_candidates.py). Main variants used for unions:
A, B, C (= C1 + C2 top 20% by score over the dataset), D10, E10. Writes
exp4/<dataset>/analysis_4a.json, exp4/tables_4a.md and figures/exp4a/<dataset>/<source>_{hit,miss}_k.png.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from common import AREA_BUCKETS, DISPLAY, OUT, load_label, load_rgb, write_json

TOPK = (1, 3, 5, 10, 20, None)
MAIN = ("A", "B", "C", "D10", "E10")
ORDER = ("A", "B", "C", "C1", "C2", "D5", "D10", "D20", "E10", "F", "G0.05", "G0.1", "G0.2")
GROUPS = ("rebuilt (building->building)", "new building", "demolished building", "vegetation <-> vegetation",
          "surface <-> vegetation", "water related", "other")


def load(dataset):
    rows = [json.loads(line) for line in open(OUT / "exp4" / dataset / "candidates.jsonl")]
    c2 = [c["score"] for r in rows for c in r["candidates"] if c["source"] == "C2"]
    c2_cut = float(np.quantile(c2, 0.8)) if c2 else None
    for r in rows:
        kept = []
        for c in r["candidates"]:
            if c["source"] == "C2" and c["score"] < c2_cut:
                continue
            kept.append(c)
            if c["source"] in ("C1", "C2"):
                kept.append({**c, "source": "C"})
        r["candidates"] = kept
    return rows, c2_cut


def coverage(rows, sources, k=None, key="covers"):
    """Covered target keys when, per image and source, only the top-k candidates by score count."""
    covered = set()
    for r in rows:
        for s in sources:
            cands = sorted((c for c in r["candidates"] if c["source"] == s), key=lambda c: -c["score"])
            for c in cands[:k] if k else cands:
                covered.update((r["image"], t) for t in c[key])
    return covered


def summarise(rows, sources):
    targets = {(r["image"], t["tid"]): t for r in rows for t in r["targets"]}
    total_px = sum(t["area"] for t in targets.values())
    cands = [c for r in rows for c in r["candidates"] if c["source"] in sources]
    cov = coverage(rows, sources)
    loose = coverage(rows, sources, key="covers_loose")
    entry = {
        "covered_objects": len(cov), "targets": len(targets),
        "coverage_objects": len(cov) / max(1, len(targets)),
        "coverage_pixels": sum(targets[t]["area"] for t in cov) / max(1, total_px),
        "coverage_objects_no_area_cap": len(loose) / max(1, len(targets)),
        "candidates_per_image": len(cands) / len(rows),
        "purity": float(np.mean([c["undetected_frac"] >= 0.5 for c in cands])) if cands else None,
        "topk_coverage": {str(k or "all"): len(coverage(rows, sources, k)) / max(1, len(targets)) for k in TOPK},
    }
    if len(sources) == 1:
        s = sources[0]
        px = sum(r["image_px"] for r in rows)
        part_total = sum(r["partial_px_total"] for r in rows)
        entry["area_fraction"] = sum(r["unions"].get(s, {}).get("union_px", 0) for r in rows) / px
        entry["partial_pixel_coverage"] = sum(r["unions"].get(s, {}).get("partial_px", 0) for r in rows) / max(1, part_total)
        entry["by_bucket"] = {}
        for _, _, b in AREA_BUCKETS:
            keys = [t for t in targets if targets[t]["bucket"] == b]
            entry["by_bucket"][b] = [len(keys), sum(t in cov for t in keys) / max(1, len(keys))]
        if any("group" in t for t in targets.values()):
            entry["by_group"] = {}
            for g in GROUPS:
                keys = [t for t in targets if targets[t].get("group") == g]
                px_g = sum(targets[t]["area"] for t in keys)
                entry["by_group"][g] = [len(keys), sum(t in cov for t in keys) / max(1, len(keys)),
                                        sum(targets[t]["area"] for t in keys if t in cov) / max(1, px_g)]
    return entry


def panel(dataset, row, target, cands, title):
    """t1 | t2 | target (white) | t2 with candidate boxes (green covering, red other)."""
    name = row["image"]
    t1, t2 = load_rgb(dataset, name, "t1"), load_rgb(dataset, name, "t2")
    h, w = t1.shape[:2]
    x0, y0, x1, y1 = target["bbox"]
    side = int(min(max(3 * max(x1 - x0, y1 - y0), 96), h, w))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
    top = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
    k = 224 / side
    gt = load_label(dataset, name)
    tmask = np.zeros((h, w, 3), np.uint8)
    tmask[gt] = 90
    tmask[y0:y1, x0:x1][gt[y0:y1, x0:x1]] = 255
    imgs = []
    for arr in (t1, t2, tmask, t2):
        imgs.append(Image.fromarray(arr[top:top + side, left:left + side]).resize((224, 224), Image.Resampling.NEAREST))
    draw = ImageDraw.Draw(imgs[3])
    for c in cands:
        bx0, by0, bx1, by1 = c["bbox"]
        if bx1 < left or by1 < top or bx0 > left + side or by0 > top + side:
            continue
        colour = (0, 255, 0) if target["tid"] in c["covers"] else (255, 60, 60)
        draw.rectangle([(bx0 - left) * k, (by0 - top) * k, (bx1 - left) * k, (by1 - top) * k], outline=colour, width=2)
    for im in imgs[:3]:
        ImageDraw.Draw(im).rectangle([(x0 - left) * k - 2, (y0 - top) * k - 2, (x1 - left) * k + 1, (y1 - top) * k + 1],
                                     outline=(255, 255, 0), width=2)
    canvas = Image.new("RGB", (4 * 229, 258), "white")
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * 229, 34))
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), title, fill="black")
    d.text((4, 17), "t1 | t2 | GT (bright = target) | candidate boxes on t2 (green = covers target, red = other)", fill="gray")
    return canvas


def figures(dataset, rows, figdir):
    figures_for(dataset, rows, figdir, ("A", "B", "C", "D10", "E10", "F", "G0.1"))


def figures_for(dataset, rows, figdir, sources):
    rng = random.Random(1016)
    for s in sources:
        hits, misses = [], []
        for r in rows:
            cands = [c for c in r["candidates"] if c["source"] == s]
            covered = {t for c in cands for t in c["covers"]}
            for t in r["targets"]:
                (hits if t["tid"] in covered else misses).append((r, t, cands))
        for kind, pool in (("hit", hits), ("miss", misses)):
            for i, (r, t, cands) in enumerate(rng.sample(pool, min(3, len(pool)))):
                group = t.get("group", "")
                title = f"{DISPLAY[dataset]} source {s} {kind} | {r['image']} target#{t['tid']} area={t['area']} {group}"
                panel(dataset, r, t, cands, title).save(figdir / f"{s}_{kind}_{i}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["second", "sysu"])
    args = parser.parse_args()
    for dataset in args.datasets:
        rows, c2_cut = load(dataset)
        present = [s for s in ORDER if any(c["source"] == s for r in rows for c in r["candidates"])]
        report = {"c2_score_cut": c2_cut, "images": len(rows),
                  "exemplar_image_fraction": float(np.mean([r["has_exemplar"] for r in rows])),
                  "sources": {s: summarise(rows, [s]) for s in present}}
        main = [s for s in MAIN if s in present]
        report["unions"] = {"+".join(p): summarise(rows, list(p)) for p in itertools.combinations(main, 2)}
        report["unions"]["+".join(main)] = summarise(rows, main)
        report["unions"]["+".join(main) + "+G0.1"] = summarise(rows, main + ["G0.1"])
        write_json(OUT / "exp4" / dataset / "analysis_4a.json", report)
        figdir = OUT / "figures" / "exp4a" / dataset
        figdir.mkdir(parents=True, exist_ok=True)
        figures(dataset, rows, figdir)
        print(dataset, {s: round(v["coverage_objects"], 3) for s, v in report["sources"].items()}, flush=True)


if __name__ == "__main__":
    main()
