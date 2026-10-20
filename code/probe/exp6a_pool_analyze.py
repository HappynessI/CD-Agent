"""Experiment 6a: train/test pool distributions and test oracles (CPU, exp5-env).

Read completed pools from exp6a_pieces.py. Stream JSONL; retain only probability
samples for exact quantiles. Oracle masks are unioned per image before micro scoring.
Optional --split_verifier uses the existing image manifest; it never selects thresholds.
Writes exp6/pool/analysis_6a.json and tables_6a.md.
"""
from __future__ import annotations

import argparse
import json
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, islice
from pathlib import Path

import numpy as np

from common import OUT, area_bucket, load_label, load_pred, names, pixel_metrics
from exp6a_pieces import DETECTOR, POOL, load_piece_masks, write_json

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
REFERENCES = {"recall_BG": 6.96, "recall_BGH1": 8.50, "remove_pred": 5.74}


def quantiles(values):
    if not len(values):
        return {str(q): None for q in QUANTILES}
    return dict(zip(map(str, QUANTILES), map(float, np.quantile(values, QUANTILES))))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def group_keys(record):
    keys = [("all", "*"), (record["kind"], "*"), (record["kind"], record["source"])]
    if record["subsource"] is not None:
        keys.append((record["kind"], record["subsource"]))
    return keys


def new_group():
    return {"pieces": 0, "positive": 0, "negative": 0, "discarded": 0,
            "pixels": 0, "positive_pixels": 0, "negative_pixels": 0, "discarded_pixels": 0,
            "gt_pixels": 0, "small_pieces": 0, "probability": array("d"),
            "area_buckets": Counter(), "per_image": Counter()}


def add_record(groups, record):
    label_key = "positive" if record["label"] == 1 else "negative" if record["label"] == 0 else "discarded"
    for key in group_keys(record):
        if key not in groups:
            groups[key] = new_group()
        group = groups[key]
        area = record["area"]
        group["pieces"] += 1
        group[label_key] += 1
        group["pixels"] += area
        group[label_key + "_pixels"] += area
        group["gt_pixels"] += int(round(record["gt_frac"] * area))
        group["small_pieces"] += int(area < 10)
        group["probability"].append(record["det_prob"]["mean"])
        group["area_buckets"][area_bucket(area)] += 1
        group["per_image"][record["image"]] += 1


def finish_groups(groups, images):
    result = {}
    for (kind, source), group in sorted(groups.items()):
        parent = groups[(kind, "*")]
        counts = np.asarray([group["per_image"][name] for name in images], dtype=np.int64)
        result[f"{kind}/{source}"] = {
            **{k: v for k, v in group.items() if k not in ("probability", "per_image", "area_buckets")},
            "positive_rate_count": ratio(group["positive"], group["pieces"]),
            "positive_rate_count_labeled": ratio(group["positive"], group["positive"] + group["negative"]),
            "positive_rate_pixels": ratio(group["positive_pixels"], group["pixels"]),
            "positive_rate_pixels_labeled": ratio(
                group["positive_pixels"], group["positive_pixels"] + group["negative_pixels"]),
            "gt_pixel_fraction": ratio(group["gt_pixels"], group["pixels"]),
            "det_prob_mean_quantiles": quantiles(group["probability"]),
            "source_share_count": ratio(group["pieces"], parent["pieces"]),
            "source_share_pixels": ratio(group["pixels"], parent["pixels"]),
            "area_buckets": {key: {"count": group["area_buckets"][key],
                                  "fraction": ratio(group["area_buckets"][key], group["pieces"])}
                             for key in ("<100", "100-1000", ">1000")},
            "pieces_per_image": {"mean": float(counts.mean()) if len(counts) else None,
                                 "quantiles": quantiles(counts),
                                 "min": int(counts.min()) if len(counts) else None,
                                 "max": int(counts.max()) if len(counts) else None,
                                 "zero_images": int((counts == 0).sum()),
                                 "values": dict(zip(images, map(int, counts)))},
        }
    return {"images": len(images), "groups": result}


def completed_summary(dataset):
    report = json.loads((POOL / dataset / "pieces_summary.json").read_text())
    if report["config"]["dataset"] != dataset:
        raise ValueError(f"{dataset}: wrong summary dataset")
    if [r["image"] for r in report["per_image"]] != names(dataset):
        raise ValueError(f"{dataset}: pool must cover all dataset images")
    return report


def image_records(dataset, report):
    """Yield every image, including images with no pieces; require sorted complete JSONL."""
    with (POOL / dataset / "pieces.jsonl").open() as handle:
        records = (json.loads(line) for line in handle)
        grouped = groupby(records, key=lambda r: r["image"])
        current = next(grouped, None)
        for image in report["per_image"]:
            name = image["image"]
            if current is not None and current[0] < name:
                raise ValueError(f"{dataset}: unexpected or unordered image {current[0]}")
            rows = []
            if current is not None and current[0] == name:
                rows = list(current[1])
                current = next(grouped, None)
            if len(rows) != image["pieces"] or any(r["piece_id"] != i for i, r in enumerate(rows, 1)):
                raise ValueError(f"{dataset}/{name}: piece count or IDs differ from completed summary")
            yield name, rows
        if current is not None:
            raise ValueError(f"{dataset}: extra JSONL image {current[0]}")


def distributions(dataset, report, subsets=None):
    subsets = subsets or {}
    images = [r["image"] for r in report["per_image"]]
    pools = {dataset: images, **{key: sorted(value) for key, value in subsets.items()}}
    accumulators = {}
    for key in pools:
        accumulators[key] = {("all", "*"): new_group(), ("recall", "*"): new_group(),
                             ("pred", "*"): new_group()}
        for source in ("B", "G0.05", "C", "C1", "C2", "H1"):
            accumulators[key][("recall", source)] = new_group()
        accumulators[key][("pred", DETECTOR[dataset])] = new_group()
    for name, records in image_records(dataset, report):
        active = [dataset] + [key for key, selected in subsets.items() if name in selected]
        for record in records:
            for key in active:
                add_record(accumulators[key], record)
    if accumulators[dataset][("all", "*")]["pieces"] != report["pieces"]:
        raise ValueError(f"{dataset}: summary total differs from JSONL")
    return {key: finish_groups(groups, pools[key]) for key, groups in accumulators.items()}


def oracle_image(item):
    name, records = item
    gt = load_label("second", name)
    pred = load_pred("second", DETECTOR["second"], name)
    masks = load_piece_masks("second", Path(name).stem)
    if set(masks) != {r["piece_id"] for r in records}:
        raise ValueError(f"{name}: NPZ and JSONL piece IDs differ")
    added_bg, added_h1, removed, pred_union = (np.zeros_like(pred) for _ in range(4))
    pred_area = 0
    for record in records:
        bbox, local = masks[record["piece_id"]]
        if bbox != record["bbox"] or int(local.sum()) != record["area"]:
            raise ValueError(f"{name}/{record['piece_id']}: NPZ and JSONL mask geometry differ")
        x0, y0, x1, y1 = bbox
        sl = (slice(y0, y1), slice(x0, x1))
        if local.shape != (y1 - y0, x1 - x0):
            raise ValueError(f"{name}/{record['piece_id']}: invalid local mask shape")
        actual_gt = int(gt[sl][local].sum())
        if actual_gt != int(round(record["gt_frac"] * record["area"])):
            raise ValueError(f"{name}/{record['piece_id']}: stored GT fraction differs from mask")
        expected_label = 1 if record["gt_frac"] >= 0.5 else 0 if record["gt_frac"] < 0.1 else None
        if record["label"] != expected_label:
            raise ValueError(f"{name}/{record['piece_id']}: invalid label")
        if record["kind"] == "recall" and record["label"] == 1:
            if record["source"] in ("B", "G0.05"):
                added_bg[sl] |= local
            elif record["source"] == "H1":
                added_h1[sl] |= local
        elif record["kind"] == "pred":
            pred_union[sl] |= local
            pred_area += record["area"]
            if record["area"] >= 10 and record["gt_frac"] < 0.1:
                removed[sl] |= local
    if not np.array_equal(pred_union, pred) or pred_area != int(pred.sum()):
        raise ValueError(f"{name}: prediction pieces do not partition the saved prediction")
    variants = {"original": pred, "recall_BG": pred | added_bg,
                "recall_BGH1": pred | added_bg | added_h1, "remove_pred": pred & ~removed}
    return {key: [int((mask & gt).sum()), int((mask & ~gt).sum()), int((~mask & gt).sum())]
            for key, mask in variants.items()}


def oracles(report, workers):
    totals = {key: np.zeros(3, np.int64) for key in ("original", *REFERENCES)}
    images = iter(image_records("second", report))
    with ProcessPoolExecutor(workers) as pool:
        done = 0
        while True:
            batch = list(islice(images, 4 * workers))
            if not batch:
                break
            for result in pool.map(oracle_image, batch):
                for key, values in result.items():
                    totals[key] += values
            done += len(batch)
            print("test oracle", done, "/", report["images"], flush=True)
    metrics = {key: pixel_metrics(*map(int, value)) for key, value in totals.items()}
    for key, reference in REFERENCES.items():
        delta = 100 * (metrics[key]["f1"] - metrics["original"]["f1"])
        metrics[key].update(delta_f1_pp=delta, reference_delta_f1_pp=reference,
                            difference_pp=delta - reference)
    return {"images": done, "aggregation": "micro", "delta_unit": "percentage_points",
            "mask_geometry_and_gt_checked": True, "metrics": metrics}


def compare(train, test):
    result = {}
    for key, value in train["groups"].items():
        reference = test["groups"][key]
        delta = {}
        for field in ("positive_rate_count", "positive_rate_count_labeled", "positive_rate_pixels",
                      "positive_rate_pixels_labeled", "gt_pixel_fraction",
                      "source_share_count", "source_share_pixels"):
            a, b = value[field], reference[field]
            delta[field + "_difference_pp"] = 100 * (a - b) if a is not None and b is not None else None
        delta["det_prob_mean_quantile_differences"] = {
            q: value["det_prob_mean_quantiles"][q] - reference["det_prob_mean_quantiles"][q]
            if value["det_prob_mean_quantiles"][q] is not None and reference["det_prob_mean_quantiles"][q] is not None
            else None for q in map(str, QUANTILES)}
        delta["area_bucket_share_differences_pp"] = {
            bucket: 100 * (entry["fraction"] - reference["area_buckets"][bucket]["fraction"])
            if entry["fraction"] is not None and reference["area_buckets"][bucket]["fraction"] is not None
            else None for bucket, entry in value["area_buckets"].items()}
        delta["pieces_per_image_mean_difference"] = (
            value["pieces_per_image"]["mean"] - reference["pieces_per_image"]["mean"])
        result[key] = delta
    return result


def number(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def markdown(report):
    lines = [
        "# 实验 6a 候选池分布与口径自检", "",
        "按块正例率 = 正块数 / 全部切块数；按像素正例率 = 正块面积 / 全部切块面积。"
        "另列去掉 label=null 后的正例率与真实变化像素占比。来源之间和候选之间的重叠按记录重复计入分布，"
        "oracle 并入按像素并集去重。分位数按切块等权统计。", "",
        "面积桶沿用 common.area_bucket：<100、100≤面积<1000、面积≥1000；"
        "最后一桶的旧名称为 >1000。每图统计包含零切块图。C1/C2 是 C 的子项。", "",
    ]
    for dataset, distribution in report["datasets"].items():
        lines += [f"## {dataset}（{distribution['images']} 张图）", "",
                  "| 类型/来源 | 切块数 | 正/负/丢弃 | 按块正例率 | 有标签按块正例率 | 按像素正例率 | 有标签按像素正例率 | 真值像素占比 | 来源块占比 | 来源像素占比 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, group in distribution["groups"].items():
            rates = [number(group[field], True) for field in
                     ("positive_rate_count", "positive_rate_count_labeled", "positive_rate_pixels",
                      "positive_rate_pixels_labeled", "gt_pixel_fraction", "source_share_count", "source_share_pixels")]
            lines.append(f"| {key} | {group['pieces']} | {group['positive']}/{group['negative']}/{group['discarded']} | "
                         + " | ".join(rates) + " |")
        lines += ["", "| 类型/来源 | 概率均值分位数 10/25/50/75/90% | 面积桶块数 <100/100–1000/>1000 | 每图平均块数 | 每图块数分位数 10/25/50/75/90% | 零块图数 |",
                  "|---|---|---|---:|---|---:|"]
        for key, group in distribution["groups"].items():
            probability = " / ".join(number(v) for v in group["det_prob_mean_quantiles"].values())
            areas = " / ".join(str(v["count"]) for v in group["area_buckets"].values())
            per_image = group["pieces_per_image"]
            counts = " / ".join(number(v) for v in per_image["quantiles"].values())
            lines.append(f"| {key} | {probability} | {areas} | {per_image['mean']:.2f} | {counts} | {per_image['zero_images']} |")
    lines += ["", "## 训练池减测试池", "",
              "| 切分 | 类型/来源 | 按块正例率差（百分点） | 按像素正例率差（百分点） | 概率均值中位数差 | 每图块数差 |",
              "|---|---|---:|---:|---:|---:|"]
    for split, differences in report["comparisons"].items():
        for key, entry in differences.items():
            lines.append(f"| {split} | {key} | {number(entry['positive_rate_count_difference_pp'])} | "
                         f"{number(entry['positive_rate_pixels_difference_pp'])} | "
                         f"{number(entry['det_prob_mean_quantile_differences']['0.5'])} | "
                         f"{number(entry['pieces_per_image_mean_difference'])} |")
    lines += ["", "## 测试池口径自检", "",
              "| 设定 | F1 | 精度 | 召回 | ΔF1（百分点） | 参考 ΔF1 | 与参考差（百分点） |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"original": "原始检测器", "recall_BG": "补漏 B+G0.05",
              "recall_BGH1": "补漏 B+G0.05+H1", "remove_pred": "删除预测负块（≥10 px）"}
    for key, metrics in report["oracle"]["metrics"].items():
        lines.append(f"| {labels[key]} | {number(metrics['f1'], True)} | "
                     f"{number(metrics['precision'], True)} | {number(metrics['recall'], True)} | "
                     f"{number(metrics.get('delta_f1_pp'))} | {number(metrics.get('reference_delta_f1_pp'))} | "
                     f"{number(metrics.get('difference_pp'))} |")
    lines += ["", "## 解释与限制", "",
              "参考值来自 results/exp5c_end_to_end.md（+6.96、+8.50）和 "
              "results/exp6e_fp_removal.md（+5.74），以两位小数记录；这里报告实际差值，不据此调整参数。",
              "切块几何与真值占比已逐条从测试池位图复核；预测切块已检查完整、不重叠地覆盖保存的预测。"
              "训练池与测试池的正例率、面积和概率差异会影响验证器阈值迁移；只在 verifier_val 选择阈值，"
              "测试池用于固定口径的分布检查和 oracle 参照。没有在此运行验证器或推断泛化成败。",
              "C2 当前保留原始候选，未应用 4a 分析阶段的前 20% 筛选；参考的三个 oracle 均不涉及 C。",
              "5c 的旧掩码恢复按 bbox 取首个匹配候选；本池保留每个原始位图。若出现差值，先检查"
              "同框多掩码、保存 PNG 与概率阈值差异、skimage 环境及原始候选完整性。", ""]
    for dataset, metadata in report["pool_metadata"].items():
        lines.append(f"- {dataset}：H1 缺失图数 {len(metadata['h1_missing_images'])}；"
                     f"保存预测与 prob>0.5 不同的像素数 {metadata['saved_mask_vs_prob_pixels']}。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split_verifier", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    summaries = {dataset: completed_summary(dataset) for dataset in ("second_cv", "second")}
    if any(report["h1_missing_images"] for report in summaries.values()):
        raise ValueError("SECOND pools require H1 for every image")
    if summaries["second_cv"]["config"]["median_area"] != summaries["second"]["config"]["median_area"]:
        raise ValueError("Train/test median-area geometry differs")
    if summaries["second_cv"]["config"]["c2_policy"] != summaries["second"]["config"]["c2_policy"]:
        raise ValueError("Train/test C2 policies differ")
    subsets = {}
    if args.split_verifier:
        manifest = json.loads((OUT / "exp6/second_cv/manifest.json").read_text())
        train, val = (manifest[key] for key in ("verifier_train_images", "verifier_val_images"))
        if len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError("Duplicate verifier split images")
        if set(train) & set(val) or set(train) | set(val) != set(names("second_cv")):
            raise ValueError("Verifier train/val must partition second_cv")
        subsets = {"verifier_train": set(train), "verifier_val": set(val)}
    datasets = distributions("second_cv", summaries["second_cv"], subsets)
    datasets.update(distributions("second", summaries["second"]))
    result = {
        "definitions": {
            "positive": "gt_frac >=0.5", "negative": "gt_frac <0.1", "discarded": "otherwise; label=null",
            "positive_rate_count": "positive pieces / all pieces",
            "positive_rate_pixels": "area of positive pieces / area of all pieces",
            "labeled_rates": "exclude label=null from denominator",
            "gt_pixel_fraction": "sum(gt_frac * area) / sum(area)",
            "overlap": "counted per piece for distributions; unioned for oracle metrics",
            "source_share": "within kind; C1/C2 are subgroups of C",
            "area_buckets": "<100; [100,1000); [1000,infinity), using existing common.area_bucket labels",
        },
        "pool_metadata": {key: {field: value[field] for field in
                               ("config", "images", "pieces", "h1_missing_images", "saved_mask_vs_prob_pixels")}
                          for key, value in summaries.items()},
        "datasets": datasets,
        "comparisons": {key + "_minus_second": compare(value, datasets["second"])
                        for key, value in datasets.items() if key != "second"},
        "oracle": oracles(summaries["second"], args.workers),
    }
    write_json(POOL / "analysis_6a.json", result)
    temporary = POOL / "tables_6a.md.tmp"
    temporary.write_text(markdown(result), encoding="utf-8")
    temporary.replace(POOL / "tables_6a.md")
    print("wrote", POOL / "analysis_6a.json", "and tables_6a.md", flush=True)


if __name__ == "__main__":
    main()
