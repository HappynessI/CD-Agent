"""Figures for results/pipeline.html: one SECOND test image followed through every pipeline stage.

--select   rank test images by how well they show the pipeline (several sources each find a
           no-signal miss, a large candidate gets split, the detector has false positives) and
           render the top --render_top images, so one can be picked by eye.
--image    render only that image.
Candidate masks are rebuilt with the experiment 4/5 rules (B, G0.05, C1/C2, H1); the per-source
candidate counts are checked against exp5/second/candidates.jsonl. GT is used only to colour the
figures and to apply the oracle, exactly as in exp5c / exp6e.
Writes figures/pipeline/<image stem>/*.jpg and example.json. Run in conda_envs/exp5-env (CPU).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage
from skimage.segmentation import find_boundaries, slic

import exp4_candidates as e4
from common import OUT, load_classes, load_label, load_prob, load_rgb, pixel_metrics, write_json
from exp5_candidates import unpack
from objects import gt_label, label, match

DS, DET = "second", "mambascd_base"
FIG = OUT / "figures/pipeline"
SECOND_RGB = OUT.parents[2] / "datasets/lim1/SECOND"
SIDE = 384
SOURCES = ("H1", "B", "G0.05", "C")
MEDIAN_AREA = 298.0                      # exp5c: median area of all SECOND no-signal misses
CLASSES = {1: "低矮植被", 2: "非植被地表", 3: "树木", 4: "水体", 5: "建筑", 6: "运动场"}
PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
                    [0, 128, 128], [220, 190, 255], [170, 110, 40], [128, 0, 0], [170, 255, 195]])


def save(path, array, nearest=True):
    image = Image.fromarray(np.asarray(array, np.uint8))
    if image.size[0] != SIDE:
        image = image.resize((SIDE, SIDE), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    image.convert("RGB").save(path, quality=88)


def heat(prob):
    return (colormaps["inferno"](np.clip(prob, 0, 1))[..., :3] * 255).astype(np.uint8)


def dim(rgb, k=0.55):
    return (rgb.astype(np.float32) * k).astype(np.uint8)


def paint(base, mask, colour, alpha=0.55):
    out = base.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(colour, np.float32)
    return out.astype(np.uint8)


def outline(base, mask, colour, thick=2):
    edge = find_boundaries(mask, mode="inner")
    if thick > 1:
        edge = ndimage.binary_dilation(edge, iterations=thick - 1) & ndimage.binary_dilation(mask, iterations=thick)
    out = base.copy()
    out[edge] = colour
    return out


def errors(rgb, pred, gt):
    out = paint(rgb, pred & gt, (0, 200, 0))
    out = paint(out, pred & ~gt, (230, 30, 30))
    return paint(out, ~pred & gt, (40, 120, 255))


def components(binary, score):
    labels, _ = label(binary)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = np.zeros(binary.shape, bool)
        mask[sl] = labels[sl] == k
        yield mask, float(score[mask].mean())


def candidates(name, prob, pred):
    """Candidate masks per source, same rules as exp4_candidates / exp5_candidates."""
    shape = pred.shape
    z = np.load(OUT / "exp4" / DS / "sam3" / name.replace(".png", ".npz"))
    raw = defaultdict(list)
    b_conf = z["conf_t1"].astype(np.float32) * z["conf_t2"].astype(np.float32)
    raw["B"] = list(components((z["cls_t1"] > 0) & (z["cls_t2"] > 0) & (z["cls_t1"] != z["cls_t2"]), b_conf))
    raw["G0.05"] = list(components(prob > 0.05, prob))
    inst = {v: z[f"inst_{v}"].astype(np.int32) for v in ("t1", "t2")}
    counts = {v: len(z[f"inst_scores_{v}"]) for v in ("t1", "t2")}
    area = {v: np.bincount(inst[v].ravel(), minlength=counts[v] + 1) for v in inst}
    both = (inst["t1"] > 0) & (inst["t2"] > 0)
    pairs, inter = np.unique(inst["t1"][both] * (counts["t2"] + 1) + inst["t2"][both], return_counts=True)
    iou = {}
    for key, n in zip(pairs, inter):
        i, j = divmod(int(key), counts["t2"] + 1)
        iou[(i, j)] = n / (area["t1"][i] + area["t2"][j] - n)
    for view in ("t1", "t2"):
        for k in range(1, counts[view] + 1):
            if area[view][k]:
                best = max([v for (i, j), v in iou.items() if (i if view == "t1" else j) == k], default=0.0)
                if best < 0.5:
                    raw["C1"].append((inst[view] == k, 1 - best))
    dist = e4.upsample(z["dist"], shape)
    for (i, j), v in iou.items():
        if v >= 0.5:
            mask = (inst["t1"] == i) | (inst["t2"] == j)
            raw["C2"].append((mask, float(dist[mask].mean())))
    zh = np.load(OUT / "exp5" / DS / "anychange" / name.replace(".png", ".npz"))
    raw["H1"] = list(zip(unpack(zh, "h1", shape), zh["h1_scores"]))
    kept = {s: [(m, float(sc)) for m, sc in v if m.sum() >= e4.MIN_CAND and pred[m].mean() <= e4.MAX_PRED_FRAC]
            for s, v in raw.items()}
    return kept


def targets(name, gt, pred):
    gl, _, gobjs, _, _ = match(gt, pred, gt_label(DS, name, gt))
    tmap = np.zeros(gt.shape, np.int32)
    t_area, t_box = [0], [0]
    for o in gobjs:
        if o["coverage"] < 0.1 and o["area"] >= e4.MIN_TARGET:
            x0, y0, x1, y1 = o["bbox"]
            tmap[gl == o["id"]] = len(t_area)
            t_area.append(o["area"])
            t_box.append((x1 - x0) * (y1 - y0))
    return tmap, np.array(t_area), np.array(t_box)


def covers(mask, tmap, t_area, t_box):
    inter = np.bincount(tmap[mask], minlength=len(t_area))
    ok = (inter >= e4.COVER_FRAC * t_area) & (mask.sum() <= e4.MAX_AREA_RATIO * t_box)
    ok[0] = False
    return np.nonzero(ok)[0].tolist()


def split(mask, segments):
    if mask.sum() <= 4 * MEDIAN_AREA:
        return [mask], False
    return [mask & (segments == i) for i in np.unique(segments[mask])], True


def crop_box(bbox, shape):
    """Verifier crop: square centred on the piece, side max(4 x long side, 64), as in exp5b."""
    x0, y0, x1, y1 = bbox
    side = int(min(max(4 * max(x1 - x0, y1 - y0), 64), *shape))
    left = int(np.clip(round((x0 + x1) / 2 - side / 2), 0, shape[1] - side))
    top = int(np.clip(round((y0 + y1) / 2 - side / 2), 0, shape[0] - side))
    return [left, top, left + side, top + side]


def _box(shape, x, y, side):
    box = np.zeros(shape, bool)
    box[y:y + side, x:x + side] = True
    return box


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def f1_of(mask, gt):
    m = pixel_metrics(int((mask & gt).sum()), int((mask & ~gt).sum()), int((~mask & gt).sum()))
    return {k: round(m[k] * 100, 2) for k in ("f1", "precision", "recall")}


def render(name, out, records):
    out.mkdir(parents=True, exist_ok=True)
    t1, t2 = load_rgb(DS, name, "t1"), load_rgb(DS, name, "t2")
    gt, prob = load_label(DS, name), load_prob(DS, DET, name)
    pred = prob > 0.5
    c1, c2 = (np.asarray(c, np.int64) for c in load_classes(DS, name))
    tmap, t_area, t_box = targets(name, gt, pred)
    cands = candidates(name, prob, pred)
    check = Counter(c["source"] for c in records.get(name, {}).get("candidates", []))
    counts = {s: len(v) for s, v in cands.items()}
    mismatch = {s: [counts.get(s, 0), check.get(s, 0)] for s in ("H1", "B", "G0.05", "C1", "C2")
                if counts.get(s, 0) != check.get(s, 0)}

    # Stage 1: inputs and reference labels.
    save(out / "t1.jpg", t1, False)
    save(out / "t2.jpg", t2, False)
    save(out / "gt.jpg", np.where(gt[..., None], 255, 0).repeat(3, 2))
    for view in ("1", "2"):
        save(out / f"sem_t{view}.jpg", np.asarray(Image.open(SECOND_RGB / f"slabel{view}_rgb" / name).convert("RGB")))
    # Stage 2: detector.
    save(out / "prob.jpg", heat(prob))
    save(out / "mask.jpg", np.where(pred[..., None], 255, 0).repeat(3, 2))
    miss = tmap > 0
    err = outline(errors(t2, pred, gt), miss, (255, 230, 0))
    save(out / "errors.jpg", err)

    # Stage 3: candidate sources.
    target_ol = lambda img: outline(img, miss, (255, 230, 0))
    per_source = {}
    for src in SOURCES:
        items = cands.get(src, []) if src != "C" else cands.get("C1", []) + cands.get("C2", [])
        img = dim(t2)
        hit = 0
        for k, (m, _) in enumerate(items):
            img = paint(img, m, PALETTE[k % len(PALETTE)], 0.45)
        for m, _ in items:
            if covers(m, tmap, t_area, t_box):
                img = outline(img, m, (255, 255, 255), 2)
                hit += 1
        save(out / f"src_{src}.jpg", target_ol(img))
        found = sorted({t for m, _ in items for t in covers(m, tmap, t_area, t_box)})
        per_source[src] = {"candidates": len(items), "covering_candidates": hit, "targets_found": found}

    # Stage 4: split. Pick the large candidate (B, G0.05, H1) with the most positive pieces.
    stack = np.concatenate([t1, t2], axis=2)
    segments = slic(stack, n_segments=max(4, int(gt.size / MEDIAN_AREA)), compactness=10,
                    channel_axis=-1, start_label=1)
    pool = [(s, m, sc) for s in ("B", "G0.05", "H1") for m, sc in cands.get(s, [])]
    best, best_key = None, (-1, -1)
    all_pieces = []
    for s, m, sc in pool:
        pieces, was_split = split(m, segments)
        labelled = []
        for p in pieces:
            if p.sum() < e4.MIN_CAND:
                continue
            g = float(gt[p].mean())
            labelled.append((p, g, 1 if g >= 0.5 else 0 if g < 0.1 else None))
        all_pieces += [(s, p, g, y) for p, g, y in labelled]
        if was_split:
            pos = sum(y == 1 for _, _, y in labelled)
            neg = sum(y == 0 for _, _, y in labelled)
            key = (min(pos, 6) + min(neg, 6), m.sum())
            if pos >= 2 and neg >= 2 and m.sum() <= 40000 and key > best_key:
                best, best_key = (s, m, labelled), key
    split_info = None
    if best is not None:
        s, m, labelled = best
        # Zoom: square around the region with a 25% margin, so pieces are visible.
        x0, y0, x1, y1 = bbox_of(m)
        side = int(min(max(1.1 * max(x1 - x0, y1 - y0), 96), *gt.shape))
        zx = int(np.clip((x0 + x1) // 2 - side // 2, 0, gt.shape[1] - side))
        zy = int(np.clip((y0 + y1) // 2 - side // 2, 0, gt.shape[0] - side))
        zoom = (slice(zy, zy + side), slice(zx, zx + side))
        save(out / "split_where.jpg", outline(dim(t2, 0.8), _box(gt.shape, zx, zy, side), (0, 200, 255), 3))
        save(out / "split_region.jpg", outline(paint(dim(t2), m, (0, 200, 255), 0.5), m, (0, 200, 255))[zoom])
        seg_img = dim(t2, 0.85)
        seg_img[find_boundaries(segments, mode="inner")] = (255, 255, 255)
        save(out / "split_slic.jpg", outline(seg_img, m, (0, 200, 255))[zoom])
        img = dim(t2)
        for p, g, y in labelled:
            img = paint(img, p, (0, 210, 80) if y == 1 else (230, 40, 40) if y == 0 else (150, 150, 150), 0.6)
            img = outline(img, p, (255, 255, 255), 1)
        save(out / "split_pieces.jpg", outline(img, gt, (255, 230, 0), 1)[zoom])
        split_info = {"zoom_xy_side": [zx, zy, side], "source": s, "region_area": int(m.sum()), "pieces": len(labelled),
                      "positive": sum(y == 1 for *_, y in labelled), "negative": sum(y == 0 for *_, y in labelled),
                      "discarded": sum(y is None for *_, y in labelled)}

    # Stage 5: verifier input, one positive and one negative piece (largest of each).
    examples = {}
    for tag, want in (("pos", 1), ("neg", 0)):
        chosen = [x for x in all_pieces if x[3] == want]
        if not chosen:
            continue
        s, p, g, y = max(chosen, key=lambda x: x[1].sum())
        bb = bbox_of(p)
        x0, y0, x1, y1 = crop_box(bb, gt.shape)
        sl = (slice(y0, y1), slice(x0, x1))
        save(out / f"v_{tag}_t1.jpg", t1[sl], False)
        save(out / f"v_{tag}_t2.jpg", t2[sl], False)
        save(out / f"v_{tag}_mask.jpg", np.where(p[sl][..., None], 255, 0).repeat(3, 2))
        save(out / f"v_{tag}_prob.jpg", heat(prob[sl]))
        pp = prob[p]
        record = {"image": name, "source": s, "piece_bbox_xyxy": bb, "piece_area_px": int(p.sum()),
                  "crop_xyxy": [x0, y0, x1, y1], "crop_side_px": x1 - x0,
                  "detector_prob": {"mean": round(float(pp.mean()), 4), "max": round(float(pp.max()), 4),
                                    "p90": round(float(np.quantile(pp, 0.9)), 4)},
                  "label_rule": "gt_frac >= 0.5 -> 1, < 0.1 -> 0", "gt_frac": round(g, 4), "label": y}
        if y == 1:
            a, b = Counter(c1[p & gt].tolist()).most_common(1)[0][0], Counter(c2[p & gt].tolist()).most_common(1)[0][0]
            record["transition"] = f"t1 {CLASSES.get(a, a)} -> t2 {CLASSES.get(b, b)}"
        examples[tag] = record

    # Stage 6a: oracle merge of positive pieces (B + G0.05 + H1, as exp5c "+H1").
    merged = pred.copy()
    for s, p, g, y in all_pieces:
        if y == 1:
            merged |= p
    save(out / "merge_after.jpg", errors(t2, merged, gt))
    save(out / "merge_before.jpg", errors(t2, pred, gt))
    # Stage 6b: oracle false-positive removal (exp6e rule: pieces of the 0.5 mask with gt_frac < 0.1).
    labels, _ = label(pred)
    removed = pred.copy()
    fp_img = dim(t2)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        comp = np.zeros(gt.shape, bool)
        comp[sl] = labels[sl] == k
        for p in split(comp, segments)[0]:
            if p.sum() >= 10 and gt[p].mean() < 0.1:
                removed &= ~p
                fp_img = paint(fp_img, p, (230, 40, 40), 0.65)
            else:
                fp_img = paint(fp_img, p, (0, 210, 80), 0.45)
            fp_img = outline(fp_img, p, (255, 255, 255), 1)
    save(out / "fp_pieces.jpg", fp_img)
    save(out / "fp_after.jpg", errors(t2, removed, gt))
    final = (merged & removed) | (merged & ~pred)
    save(out / "final.jpg", errors(t2, final, gt))
    save(out / "final_mask.jpg", np.where(final[..., None], 255, 0).repeat(3, 2))

    info = {"image": name, "no_signal_misses": int(len(t_area) - 1), "candidate_count_check_mismatch": mismatch,
            "per_source": per_source, "split": split_info, "verifier_examples": examples,
            "f1_image": {"detector": f1_of(pred, gt), "oracle_merge": f1_of(merged, gt),
                         "oracle_fp_removal": f1_of(removed, gt), "both": f1_of(final, gt)}}
    write_json(out / "example.json", info)
    return info


def train_example(name, out):
    """A SECOND train image with all its labels (used only on the training side)."""
    out.mkdir(parents=True, exist_ok=True)
    for view in ("t1", "t2"):
        save(out / f"{view}.jpg", load_rgb(DS, name, view), False)
    save(out / "gt.jpg", np.where(load_label(DS, name)[..., None], 255, 0).repeat(3, 2))
    for view in ("1", "2"):
        save(out / f"sem_t{view}.jpg", np.asarray(Image.open(SECOND_RGB / f"slabel{view}_rgb" / name).convert("RGB")))


def select(records, pieces_fp):
    ranked = []
    for name, r in records.items():
        found = defaultdict(set)
        for c in r["candidates"]:
            src = "C" if c["source"] in ("C1", "C2") else c["source"]
            if src in SOURCES:
                for t in c["covers"]:
                    found[t].add(src)
        sources = {s for v in found.values() for s in v}
        multi = sum(len(v) >= 2 for v in found.values())
        score = 2 * len(sources) + min(multi, 4) + min(len(found), 6) * 0.5 + min(pieces_fp.get(name, 0), 4) * 0.5
        if len(sources) >= 3 and pieces_fp.get(name, 0) >= 1:
            ranked.append((score, name, sorted(sources), len(found), pieces_fp.get(name, 0)))
    return sorted(ranked, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--render_top", type=int, default=4)
    parser.add_argument("--train_image", default="train_1010.png")
    args = parser.parse_args()
    records = {}
    with open(OUT / "exp5" / DS / "candidates.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if args.select or r["image"] == args.image:
                records[r["image"]] = r
    names = [args.image] if args.image else []
    if args.select:
        fp = Counter()
        with open(OUT / "exp6e" / DS / "pieces.jsonl") as f:
            for line in f:
                p = json.loads(line)
                if p["deleted_oracle_piece"] and p["area"] >= 100:
                    fp[p["image"]] += 1
        ranked = select(records, fp)
        write_json(FIG / "select.json", [dict(zip(("score", "image", "sources", "targets_found", "fp_pieces"), r))
                                         for r in ranked[:30]])
        names = [r[1] for r in ranked[:args.render_top]]
    for name in names:
        info = render(name, FIG / name.replace(".png", ""), records)
        print(json.dumps({k: info[k] for k in ("image", "no_signal_misses", "candidate_count_check_mismatch",
                                               "split", "f1_image")}, ensure_ascii=False))
    train_example(args.train_image, FIG / "train_example")


if __name__ == "__main__":
    main()
