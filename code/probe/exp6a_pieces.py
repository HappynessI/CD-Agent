"""Experiment 6a: recall and prediction pieces (CPU, conda_envs/exp5-env).

Rebuild B/G0.05/C1/C2/H1 from raw outputs as in exp6_pipeline_figs.candidates.
C2 is the raw candidate set, before exp4a_analyze's dataset-wide top-20% filter.
GT only annotates pieces. SLIC uses the two RGB images, once per image.
Writes exp6/pool/<dataset>/pieces*.jsonl, summaries and bbox-local packed masks.
Run shards with the same --shards, then --merge --shards N. IDs start at 1 per image,
across both kinds; candidate IDs follow B, G0.05, C1, C2, H1, then prediction components.
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.segmentation import slic

import exp4_candidates as e4
from common import OUT, load_classes, load_label, load_pred, load_prob, load_rgb, names
from exp6_pool_io import DETECTOR, POOL, load_piece_masks  # noqa: F401 (re-exported)
from objects import gt_label, label, match

# exp5c/<dataset>/metrics_5c_B_G0.05.json, config.median_target_area.
# Freeze the SECOND test-derived 5c geometry for second_cv; never estimate it on train.
MEDIAN_AREA = {"second": 298, "second_cv": 298, "sysu": 972, "second_scannet": 298,
               "levir": None, "whu": None, "lsscd": None, "scscd7": None}
RECALL_SOURCES = ("B", "G0.05", "C", "H1")


@lru_cache(maxsize=None)
def geometry(dataset):
    """Freeze existing geometry; new datasets require a completed train-only estimate."""
    median = MEDIAN_AREA[dataset]
    if median is not None:
        return median, {"method": "frozen_exp5c", "split": "test"}
    path = OUT / "exp6/geometry" / f"{dataset}.json"
    report = json.loads(path.read_text())
    if (report["dataset"] != dataset or report["split"] != "train"
            or report["test_gt_used"] or report["min_area"] != e4.MIN_TARGET
            or report["detector"] != DETECTOR[dataset]
            or report["method"] not in ("train_all_gt", "train_missed_gt")):
        raise ValueError(f"{path}: incompatible train geometry; rerun exp6c_median_area.py")
    median = float(report["median_area"])
    if not np.isfinite(median) or median < e4.MIN_TARGET:
        raise ValueError(f"{path}: invalid median_area={median}")
    return median, {"path": str(path), "method": report["method"], "split": "train"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def components(binary, score):
    labels, _ = label(binary)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = np.zeros(binary.shape, bool)
        mask[sl] = labels[sl] == k
        yield mask, float(score[mask].mean())


def candidates(dataset, name, prob, pred):
    """Yield raw experiment 4/5 candidates, preserving masks even with equal bboxes."""
    with np.load(OUT / "exp4" / dataset / "sam3" / (Path(name).stem + ".npz")) as z:
        conf = z["conf_t1"].astype(np.float32) * z["conf_t2"].astype(np.float32)
        binary = (z["cls_t1"] > 0) & (z["cls_t2"] > 0) & (z["cls_t1"] != z["cls_t2"])
        for mask, score in components(binary, conf):
            yield "B", None, mask, score
        for mask, score in components(prob > 0.05, prob):
            yield "G0.05", None, mask, score
        inst = {v: z[f"inst_{v}"].astype(np.int32) for v in ("t1", "t2")}
        counts = {v: len(z[f"inst_scores_{v}"]) for v in inst}
        areas = {v: np.bincount(inst[v].ravel(), minlength=counts[v] + 1) for v in inst}
        both = (inst["t1"] > 0) & (inst["t2"] > 0)
        pairs, inter = np.unique(
            inst["t1"][both] * (counts["t2"] + 1) + inst["t2"][both], return_counts=True)
        iou = {}
        for key, n in zip(pairs, inter):
            i, j = divmod(int(key), counts["t2"] + 1)
            iou[(i, j)] = n / (areas["t1"][i] + areas["t2"][j] - n)
        for view in ("t1", "t2"):
            for k in range(1, counts[view] + 1):
                if not areas[view][k]:
                    continue
                best = max([v for (i, j), v in iou.items()
                            if (i if view == "t1" else j) == k], default=0.0)
                if best < 0.5:
                    yield "C", "C1", inst[view] == k, 1 - best
        dist = e4.upsample(z["dist"], pred.shape)
        for (i, j), v in iou.items():
            if v >= 0.5:
                mask = (inst["t1"] == i) | (inst["t2"] == j)
                yield "C", "C2", mask, float(dist[mask].mean())
    path = OUT / "exp5" / dataset / "anychange" / (Path(name).stem + ".npz")
    if dataset == "sysu" and not path.exists():
        return
    with np.load(path) as z:
        if tuple(z["shape"]) != pred.shape or len(z["h1_masks"]) != len(z["h1_scores"]):
            raise ValueError(f"{path}: H1 shape or mask/score count mismatch")
        for packed, score in zip(z["h1_masks"], z["h1_scores"]):
            mask = np.unpackbits(packed, count=pred.size).reshape(pred.shape).astype(bool)
            yield "H1", None, mask, float(score)


def targets(dataset, name, gt, pred):
    gl, _, objects, _, _ = match(gt, pred, gt_label(dataset, name, gt))
    tmap = np.zeros(gt.shape, np.int32)
    areas, boxes = [0], [0]
    for obj in objects:
        if obj["coverage"] < 0.1 and obj["area"] >= e4.MIN_TARGET:
            x0, y0, x1, y1 = obj["bbox"]
            tmap[gl == obj["id"]] = len(areas)
            areas.append(obj["area"])
            boxes.append((x1 - x0) * (y1 - y0))
    return tmap, np.asarray(areas), np.asarray(boxes)


def covers(mask, target_data):
    tmap, areas, boxes = target_data
    inter = np.bincount(tmap[mask], minlength=len(areas))
    ok = (inter >= e4.COVER_FRAC * areas) & (mask.sum() <= e4.MAX_AREA_RATIO * boxes)
    ok[0] = False
    return bool(ok.any())


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def crop_box(bbox, shape):
    """Use exp5b_recheck's square size, clipping and integer truncation."""
    x0, y0, x1, y1 = bbox
    side = int(min(max(4 * max(x1 - x0, y1 - y0), 64), *shape))
    left = int(np.clip((x0 + x1) / 2 - side / 2, 0, shape[1] - side))
    top = int(np.clip((y0 + y1) / 2 - side / 2, 0, shape[0] - side))
    return [left, top, left + side, top + side]


def process(name, dataset):
    prob = load_prob(dataset, DETECTOR[dataset], name)
    recall_pred = prob > 0.5                       # exp4/5 candidate filtering
    pred = load_pred(dataset, DETECTOR[dataset], name)  # exp6e saved 0.5 PNG
    shape = prob.shape
    median, _ = geometry(dataset)
    gt = load_label(dataset, name)
    classes = load_classes(dataset, name)
    target_data = targets(dataset, name, gt, recall_pred)
    records, packed, boxes, shapes = [], [], [], []
    segments = None
    discarded_small = Counter()
    candidate_counts = Counter()
    cand_id = 0

    def add(kind, source, subsource, mask, score):
        nonlocal segments, cand_id
        cand_id += 1
        cand_area = int(mask.sum())
        candidate_counts[f"{kind}/{subsource or source}"] += 1
        split = cand_area > 4 * median
        covered = covers(mask, target_data)
        if split:
            if segments is None:
                stack = np.concatenate([load_rgb(dataset, name, v) for v in ("t1", "t2")], axis=2)
                segments = slic(stack, n_segments=max(4, int(prob.size / median)),
                                compactness=10, channel_axis=-1, start_label=1)
            pieces = (mask & (segments == i) for i in np.unique(segments[mask]))
        else:
            pieces = (mask,)
        for piece in pieces:
            area = int(piece.sum())
            if kind == "recall" and area < e4.MIN_CAND:
                discarded_small[f"{kind}/{source}"] += 1
                continue
            bbox = bbox_of(piece)
            x0, y0, x1, y1 = bbox
            local = piece[y0:y1, x0:x1]
            gt_frac = float(gt[piece].mean())
            target = 1 if gt_frac >= 0.5 else 0 if gt_frac < 0.1 else None
            transition = None
            if target == 1 and classes is not None:
                changed = piece & gt
                pairs = np.stack([c[changed] for c in classes], axis=1)
                values, counts = np.unique(pairs, axis=0, return_counts=True)
                transition = values[np.argmax(counts)].astype(int).tolist()
            values = prob[piece]
            records.append({
                "image": name, "kind": kind, "piece_id": len(records) + 1,
                "source": source, "subsource": subsource, "cand_id": cand_id,
                "cand_score": float(score), "cand_area": cand_area, "split": split,
                "area": area, "bbox": bbox, "crop": crop_box(bbox, shape),
                "det_prob": {"mean": float(values.mean()), "max": float(values.max()),
                             "min": float(values.min()), "p90": float(np.quantile(values, 0.9))},
                "gt_frac": gt_frac, "label": target, "covers_target": covered,
                "transition": transition,
            })
            packed.append(np.packbits(local.ravel()))
            boxes.append(bbox)
            shapes.append(local.shape)

    for source, subsource, mask, score in candidates(dataset, name, prob, recall_pred):
        if mask.sum() >= e4.MIN_CAND and recall_pred[mask].mean() <= e4.MAX_PRED_FRAC:
            add("recall", source, subsource, mask, score)
    for mask, score in components(pred, prob):
        add("pred", DETECTOR[dataset], None, mask, score)
    path = POOL / dataset / "masks" / (Path(name).stem + ".npz")
    temporary = path.with_suffix(".tmp.npz")
    offsets = np.concatenate(([0], np.cumsum([len(p) for p in packed], dtype=np.int64)))
    np.savez_compressed(
        temporary, image_shape=np.asarray(shape, np.int32),
        piece_ids=np.arange(1, len(records) + 1, dtype=np.int32),
        bboxes=np.asarray(boxes, np.int32).reshape(-1, 4),
        shapes=np.asarray(shapes, np.int32).reshape(-1, 2), offsets=offsets,
        packed=np.concatenate(packed) if packed else np.empty(0, np.uint8))
    temporary.replace(path)
    h1_path = OUT / "exp5" / dataset / "anychange" / (Path(name).stem + ".npz")
    return records, {
        "image": name, "pieces": len(records), "candidates": dict(candidate_counts),
        "discarded_small": dict(discarded_small), "h1_available": h1_path.exists(),
        "saved_mask_vs_prob_pixels": int(np.count_nonzero(pred != recall_pred)),
    }


def empty_groups(dataset):
    return {f"{kind}/{source}": dict(pieces=0, positive=0, negative=0, discarded=0,
                                    small_retained=0, discarded_small=0)
            for kind, sources in (("recall", RECALL_SOURCES), ("pred", (DETECTOR[dataset],)))
            for source in sources}


def count_record(groups, record):
    group = groups[f'{record["kind"]}/{record["source"]}']
    group["pieces"] += 1
    group["positive" if record["label"] == 1 else
          "negative" if record["label"] == 0 else "discarded"] += 1
    group["small_retained"] += int(record["area"] < e4.MIN_CAND)


def summary(dataset, groups, images, shard, shards):
    for image in images:
        for key, count in image["discarded_small"].items():
            groups[key]["discarded_small"] += count
    for group in groups.values():
        usable = group["positive"] + group["negative"]
        group["positive_rate"] = group["positive"] / usable if usable else None
        group["positive_rate_all"] = group["positive"] / group["pieces"] if group["pieces"] else None
    return {
        "config": {"dataset": dataset, "detector": DETECTOR[dataset], "shard": shard, "shards": shards,
                   "median_area": geometry(dataset)[0], "geometry": geometry(dataset)[1],
                   "c2_policy": "raw_all",
                   "candidate_prediction": "prob > 0.5", "prediction_pieces": "saved_0.5_mask",
                   "min_recall_piece_area": e4.MIN_CAND, "pred_keeps_small": True,
                   "slic": {"compactness": 10, "start_label": 1, "channels": 6,
                            "n_segments": "max(4, int(H*W / median_area))"}},
        "images": len(images), "pieces": sum(g["pieces"] for g in groups.values()),
        "groups": groups, "per_image": images,
        "h1_missing_images": [r["image"] for r in images if not r["h1_available"]],
        "saved_mask_vs_prob_pixels": sum(r["saved_mask_vs_prob_pixels"] for r in images),
        "rates": {"positive_rate": "positive / (positive + negative)",
                  "positive_rate_all": "positive / all saved pieces",
                  "discarded": "label is null; retained on disk",
                  "discarded_small": "recall pieces <10 px; absent from JSONL and masks"},
    }


def rows(handle):
    for line in handle:
        yield json.loads(line)


def merge(dataset, shards):
    base = POOL / dataset
    items = names(dataset)
    images, reports = [], []
    for shard in range(shards):
        report = json.loads((base / f"pieces_summary_{shard}.json").read_text())
        expected = items[shard::shards]
        if (report["config"]["dataset"] != dataset or report["config"]["shard"] != shard
                or report["config"]["shards"] != shards
                or [r["image"] for r in report["per_image"]] != expected):
            raise ValueError(f"Shard {shard}: configuration or image coverage mismatch")
        if reports:
            for key, value in reports[0]["config"].items():
                if key != "shard" and report["config"][key] != value:
                    raise ValueError(f"Shard {shard}: inconsistent {key}")
        images.extend(report["per_image"])
        reports.append(report)
    images.sort(key=lambda r: r["image"])
    expected_counts = {r["image"]: r["pieces"] for r in images}
    for name in items:
        if not (base / "masks" / (Path(name).stem + ".npz")).exists():
            raise FileNotFoundError(f"Missing mask archive for {name}")
    groups, actual_counts = empty_groups(dataset), Counter()
    temporary = base / "pieces.jsonl.tmp"
    previous = None
    with ExitStack() as stack, temporary.open("w") as output:
        streams = [rows(stack.enter_context((base / f"pieces_{s}.jsonl").open())) for s in range(shards)]
        for record in heapq.merge(*streams, key=lambda r: (r["image"], r["piece_id"])):
            key = record["image"], record["piece_id"]
            if previous is not None and key <= previous:
                raise ValueError(f"Duplicate or unordered piece: {key}")
            previous = key
            if record["image"] not in expected_counts:
                raise ValueError(f"Unexpected image: {record['image']}")
            actual_counts[record["image"]] += 1
            if record["piece_id"] != actual_counts[record["image"]]:
                raise ValueError(f"Non-contiguous piece ID: {key}")
            count_record(groups, record)
            output.write(json.dumps(record, allow_nan=False) + "\n")
    if any(actual_counts[name] != count for name, count in expected_counts.items()):
        raise ValueError("Shard JSONL counts differ from completed-image summaries")
    result = summary(dataset, groups, images, None, shards)
    if result["config"]["median_area"] != reports[0]["config"]["median_area"]:
        raise ValueError("Train geometry changed after shard generation")
    for group, values in result["groups"].items():
        for field in ("pieces", "positive", "negative", "discarded", "small_retained", "discarded_small"):
            if values[field] != sum(r["groups"][group][field] for r in reports):
                raise ValueError(f"Shard summary mismatch: {group}/{field}")
    temporary.replace(base / "pieces.jsonl")
    write_json(base / "pieces_summary.json", result)
    print(dataset, "merged", len(items), "images", result["pieces"], "pieces", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DETECTOR), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("require workers/shards >=1 and 0 <= shard < shards")
    geometry(args.dataset)
    if args.merge:
        if args.shards == 1:
            parser.error("--merge requires --shards > 1; one-shard runs already write final files")
        merge(args.dataset, args.shards)
        return
    items = names(args.dataset)
    if not items:
        raise ValueError(f"No images for {args.dataset}")
    items = items[args.shard::args.shards]
    base = POOL / args.dataset
    (base / "masks").mkdir(parents=True, exist_ok=True)
    suffix = "" if args.shards == 1 else f"_{args.shard}"
    output = base / f"pieces{suffix}.jsonl"
    report_path = base / f"pieces_summary{suffix}.json"
    # A failed rerun must not leave an old completion summary usable by --merge.
    report_path.unlink(missing_ok=True)
    (base / "pieces_summary.json").unlink(missing_ok=True)
    groups, images = empty_groups(args.dataset), []
    worker = partial(process, dataset=args.dataset)
    temporary = output.with_suffix(".jsonl.tmp")
    with ProcessPoolExecutor(args.workers) as pool, temporary.open("w") as handle:
        batch_size = 4 * args.workers
        for start in range(0, len(items), batch_size):
            for records, image in pool.map(worker, items[start:start + batch_size]):
                images.append(image)
                for record in records:
                    count_record(groups, record)
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(args.dataset, args.shard, len(images), "/", len(items), flush=True)
    temporary.replace(output)
    write_json(report_path, summary(args.dataset, groups, images, args.shard, args.shards))


if __name__ == "__main__":
    main()
