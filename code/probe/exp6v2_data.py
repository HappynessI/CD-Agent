"""Build image-disjoint V2 samples and contour-marked paired crops (CPU job)."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from common import load_rgb
from exp4_candidates import transition_group
from exp6_pool_io import load_piece_masks
from exp6b_common import ROOT, SEED, rows_by_image, source_of, verifier_split, write_json

SIDE = 384
CLASSES = {1: "低矮植被", 2: "非植被地表", 3: "树", 4: "水", 5: "建筑", 6: "运动场"}
GROUPS = {
    "new building": "新建建筑", "demolished building": "拆除建筑",
    "rebuilt (building->building)": "改建", "vegetation <-> vegetation": "植被互变",
    "surface <-> vegetation": "地表与植被互变", "water related": "水体相关", "other": "其他",
}
NEGATIVE = "两期为同一地物，仅色调或成像差异，不是真实变化。"


def eligible(dataset, images=None):
    allowed = None if images is None else set(images)
    for name, rows in rows_by_image(dataset):
        if allowed is None or name in allowed:
            for row in rows:
                if row["kind"] == "recall" and row["label"] in (0, 1):
                    yield row


def cell(row):
    return f"{source_of(row)}/{row['label']}"


def reservoir(rows, capacities, rng, stratified):
    samples = {key: [] for key in capacities}
    counts = Counter()
    positive = 0
    for row in rows:
        key = cell(row) if stratified else "all"
        counts[key] += 1
        positive += row["label"]
        bucket, capacity = samples[key], capacities[key]
        if len(bucket) < capacity:
            bucket.append(row)
        else:
            index = rng.randrange(counts[key])
            if index < capacity:
                bucket[index] = row
    result = []
    strata = {}
    for key, bucket in samples.items():
        if not bucket:
            raise ValueError(f"No eligible samples in {key}")
        weight = counts[key] / len(bucket)
        strata[key] = dict(population=counts[key], sampled=len(bucket), weight=weight)
        result.extend({**r, "weight": weight, "stratum": key} for r in bucket)
    return result, dict(population=sum(counts.values()), positive=positive, strata=strata)


def allocate(counts, requested):
    """Equal allocation across nonempty source/label cells, capped by availability."""
    total = min(requested, sum(counts.values()))
    if not counts or total < len(counts):
        raise ValueError("Test sample size must cover every nonempty source/label cell")
    capacity = {key: 0 for key in sorted(counts)}
    remaining = total
    while remaining:
        for key in capacity:
            if remaining and capacity[key] < counts[key]:
                capacity[key] += 1
                remaining -= 1
    return capacity


def targets(row):
    answer = "是" if row["label"] else "否"
    if row["label"]:
        transition = row["transition"]
        if transition is None or len(transition) != 2 or any(c not in CLASSES for c in transition):
            raise ValueError(f"Missing SECOND transition: {row['image']}/{row['piece_id']}")
        c1, c2 = transition
        reason = (f"t1 为{CLASSES[c1]}，t2 为{CLASSES[c2]}，地物类别改变，"
                  f"是真实变化（{GROUPS[transition_group(transition)]}）。")
    else:
        reason = NEGATIVE
    return f"答案：{answer}", f"{reason}\n答案：{answer}"


class Renderer:
    """Cache one image pair and its saved masks; inference never reads GT."""

    def __init__(self):
        self.key = None

    def render(self, dataset, row):
        key = (dataset, row["image"])
        if key != self.key:
            self.rgb = [load_rgb(*key, view) for view in ("t1", "t2")]
            self.masks = load_piece_masks(*key)
            self.key = key
        left, top, right, bottom = row["crop"]
        bbox, local = self.masks[row["piece_id"]]
        x0, y0, x1, y1 = bbox
        height, width = self.rgb[0].shape[:2]
        if not (0 <= left <= x0 < x1 <= right <= width
                and 0 <= top <= y0 < y1 <= bottom <= height):
            raise ValueError(f"Crop does not contain piece: {key}/{row['piece_id']}")
        mask = np.zeros((bottom - top, right - left), np.uint8)
        mask[y0 - top:y1 - top, x0 - left:x1 - left] = local * 255
        enlarged = Image.fromarray(mask).resize((SIDE, SIDE), Image.Resampling.NEAREST)
        # Padding makes contours on the crop boundary visible as well.
        padded = Image.new("L", (SIDE + 4, SIDE + 4))
        padded.paste(enlarged, (2, 2))
        inner = padded.filter(ImageFilter.MinFilter(3)).crop((2, 2, SIDE + 2, SIDE + 2))
        edge = (np.asarray(enlarged) > 0) & (np.asarray(inner) == 0)
        edge = np.asarray(Image.fromarray(edge.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
        images = []
        for rgb in self.rgb:
            crop = Image.fromarray(rgb[top:bottom, left:right]).resize(
                (SIDE, SIDE), Image.Resampling.BICUBIC)
            pixels = np.array(crop)
            pixels[edge] = (255, 0, 0)
            images.append(Image.fromarray(pixels))
        return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "v2/data")
    parser.add_argument("--train-size", type=int, default=24000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.test_size) < 1:
        parser.error("Sample sizes must be positive")
    train, val = verifier_split()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(seed=args.seed, train_images=train, val_images=val, side=SIDE,
                  train_sampling="uniform_without_replacement_no_class_balancing",
                  test_sampling="equal_source_label_cells_capped_by_population",
                  detector="mambascd_base", splits={})
    renderer = Renderer()
    for offset, (split, dataset, images, size) in enumerate((
        ("train", "second_cv", train, args.train_size),
        ("val", "second_cv", val, args.val_size),
        ("test", "second", None, args.test_size),
    )):
        stratified = split == "test"
        capacities = allocate(Counter(cell(r) for r in eligible(dataset)), size) if stratified else {"all": size}
        rows, stats = reservoir(eligible(dataset, images), capacities,
                                random.Random(args.seed + offset), stratified)
        folder = args.out / split
        folder.mkdir()
        rows.sort(key=lambda r: (r["image"], r["piece_id"]))
        negatives = sum(r["label"] == 0 for r in rows)
        stats.update(requested=size, sampled=len(rows), sampled_positive=len(rows) - negatives,
                     population_positive_rate=stats["positive"] / stats["population"],
                     sampled_positive_rate=(len(rows) - negatives) / len(rows),
                     generic_negative_template_count=negatives if split != "test" else 0,
                     generic_template_fraction=negatives / len(rows) if split != "test" else None,
                     generic_template_fraction_among_negatives=1.0 if negatives and split != "test" else None)
        with (args.out / f"{split}.jsonl").open("x") as handle:
            for i, row in enumerate(rows):
                paths = {}
                for view, image in zip(("t1", "t2"), renderer.render(dataset, row)):
                    path = folder / f"{Path(row['image']).stem}_{row['piece_id']}_{view}.png"
                    image.save(path)
                    paths[view] = str(path.resolve())
                record = {k: row[k] for k in ("image", "piece_id", "label", "area", "gt_frac",
                                              "crop", "weight", "stratum")}
                record.update(dataset=dataset, split=split, kind="recall",
                              source=source_of(row), inputs=paths)
                if split != "test":
                    record["target_a"], record["target_b"] = targets(row)
                    record["transition"] = row["transition"]
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if (i + 1) % 500 == 0:
                    print(split, i + 1, len(rows), flush=True)
        report["splits"][split] = stats
    write_json(args.out / "data_report.json", report)


if __name__ == "__main__":
    main()
