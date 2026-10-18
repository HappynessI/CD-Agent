"""Experiment 5a: turn AnyChange (H1, H2) and SAM3-AnyChange (H3) masks into candidate records
with exactly the experiment 4 rules and fields (target set M, coverage, purity, region signals),
then merge them with the experiment 4 candidates of the same image.

Writes exp5/<dataset>/candidates.jsonl: one line per image with all exp4 candidates plus H1/H2/H3.
Run in conda_envs/vlm-vs-onejev-env (CPU).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

import exp4_candidates as e4
from common import OUT, load_label, load_prob
from objects import gt_label, match


def unpack(z, key, shape):
    packed = z[f"{key}_masks"]
    if packed.size == 0:
        return []
    n = shape[0] * shape[1]
    return [np.unpackbits(row)[:n].reshape(shape).astype(bool) for row in packed]


def process(record, dataset):
    name = record["image"]
    gt = load_label(dataset, name)
    shape = gt.shape
    prob = load_prob(dataset, e4.DETECTOR[dataset], name)
    pred = prob > 0.5
    gl, _, gobjs, _, _ = match(gt, pred, gt_label(dataset, name, gt))
    undetected = gt & ~pred
    tmap = np.zeros(shape, np.int32)
    tid = 0
    t_area, t_box = [0], [0]
    for o in gobjs:                       # same order and ids as exp4_candidates
        if o["coverage"] < 0.1 and o["area"] >= e4.MIN_TARGET:
            tid += 1
            x0, y0, x1, y1 = o["bbox"]
            tmap[gl == o["id"]] = tid
            t_area.append(o["area"])
            t_box.append((x1 - x0) * (y1 - y0))
    t_area, t_box = np.array(t_area), np.array(t_box)
    partial_ids = [o["id"] for o in gobjs if o["class"] == "partial"]
    m_partial = np.isin(gl, partial_ids) & ~pred if partial_ids else np.zeros(shape, bool)
    z4 = np.load(OUT / "exp4" / dataset / "sam3" / name.replace(".png", ".npz"))
    dist = e4.upsample(z4["dist"], shape)
    maps = {"det_prob": prob, "D_dist": dist,
            "B_agree": ((z4["cls_t1"] == z4["cls_t2"]) & (z4["cls_t1"] > 0)).astype(np.float32),
            "B_conf": z4["conf_t1"].astype(np.float32) * z4["conf_t2"].astype(np.float32)}
    unions = record["unions"]
    cands = []
    for source, path, key in (("H1", "anychange", "h1"), ("H2", "anychange", "h2"), ("H3", "sam3_anychange", "h3")):
        f = OUT / "exp5" / dataset / path / name.replace(".png", ".npz")
        if not f.exists():
            continue
        z = np.load(f)
        union = np.zeros(shape, bool)
        for mask, score in zip(unpack(z, key, shape), z[f"{key}_scores"]):
            area = int(mask.sum())
            if area < e4.MIN_CAND or pred[mask].mean() > e4.MAX_PRED_FRAC:
                continue
            union |= mask
            inter = np.bincount(tmap[mask], minlength=len(t_area))
            loose = inter >= e4.COVER_FRAC * t_area
            loose[0] = False
            ok = loose & (area <= e4.MAX_AREA_RATIO * t_box)
            ys, xs = np.nonzero(mask)
            cands.append({"source": source, "score": float(score), "area": area,
                          "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                          "pred_frac": float(pred[mask].mean()), "undetected_frac": float(undetected[mask].mean()),
                          "gt_frac": float(gt[mask].mean()), "covers": np.nonzero(ok)[0].tolist(),
                          "covers_loose": np.nonzero(loose)[0].tolist(), "partial_px": int(m_partial[mask].sum()),
                          **{k: float(v[mask].mean()) for k, v in maps.items()}})
        unions[source] = {"union_px": int(union.sum()), "partial_px": int((union & m_partial).sum())}
    record["candidates"] = record["candidates"] + cands
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    records = (json.loads(line) for line in open(OUT / "exp4" / args.dataset / "candidates.jsonl"))
    out = OUT / "exp5" / args.dataset / "candidates.jsonl"
    with ProcessPoolExecutor(args.workers) as pool, open(out, "w") as f:
        for rec in pool.map(partial(process, dataset=args.dataset), records, chunksize=4):
            f.write(json.dumps(rec) + "\n")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
