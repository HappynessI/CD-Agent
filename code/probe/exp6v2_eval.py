"""Weighted held-out scoring, validation-only timing gate and full-pool E2E."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from common import OUT
from exp6b_common import (
    ROOT, SEED, Pool, binary_metrics, grouped_metrics, rows_by_image,
    save_npz, verifier_split, write_json,
)
from exp6v2_common import load_model, read_adapter, read_rows, score_one, score_rows
from exp6v2_data import Renderer


def weighted_metrics(labels, scores, weights):
    """Weighted tie-aware ROC with the same >= threshold convention as binary_metrics."""
    y, s, w = (np.asarray(x, np.float64) for x in (labels, scores, weights))
    if (len(y) != len(s) or len(y) != len(w) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(s).all() or not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError("Invalid weighted metric inputs")
    positive, negative = float((w * y).sum()), float((w * (1 - y)).sum())
    result = dict(n=len(y), population_estimate=float(w.sum()), positive=positive,
                  negative=negative, auc=None)
    for recall in (80, 90):
        result[f"recall_{recall}"] = dict(threshold=None, fpr=None, achieved_recall=None)
    if not len(y) or not positive or not negative:
        return result
    order = np.argsort(-s, kind="stable")
    yy, ss, ww = y[order], s[order], w[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp, fp = np.cumsum(ww * yy)[ends], np.cumsum(ww * (1 - yy))[ends]
    tpr, fpr = np.r_[0, tp / positive], np.r_[0, fp / negative]
    result["auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    for recall in (80, 90):
        i = min(int(np.searchsorted(tp, positive * recall / 100)), len(ends) - 1)
        result[f"recall_{recall}"] = dict(
            threshold=float(ss[ends[i]]), fpr=float(fp[i] / negative),
            achieved_recall=float(tp[i] / positive))
    return result


def metrics_for(rows, results):
    y = [r["label"] for r in rows]
    s = [r["logit_diff"] for r in results]
    w = [r["weight"] for r in rows]
    report = dict(overall=weighted_metrics(y, s, w), by_source={})
    for source in sorted({r["source"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["source"] == source]
        report["by_source"][source] = weighted_metrics(
            [y[i] for i in indices], [s[i] for i in indices], [w[i] for i in indices])
    return report


def frozen_threshold_metrics(rows, results, thresholds):
    y = np.asarray([r["label"] for r in rows])
    w = np.asarray([r["weight"] for r in rows])
    s = np.asarray([r["logit_diff"] for r in results])
    report = {}
    for recall in (80, 90):
        threshold = thresholds[f"recall_{recall}"]["threshold"]
        accepted = s >= threshold
        report[f"validation_recall_{recall}"] = dict(
            threshold=threshold,
            test_recall=float(w[accepted & (y == 1)].sum() / w[y == 1].sum()),
            test_fpr=float(w[accepted & (y == 0)].sum() / w[y == 0].sum()))
    return report


def historical_comparison():
    # Actual exp4b/5b JSON, not the rounded values in the prose report.
    comparison = {}
    for stage in ("4b", "5b"):
        path = OUT / "exp4b/second" / f"analysis_{stage}.json"
        data = json.loads(path.read_text())
        comparison[stage] = dict(
            path=str(path), metrics=data["signals"]["mllm_qwen35"],
            scope="SECOND B+G0.05, 300 balanced whole candidates; not the exp6 piece pool")
    return comparison


def v1_comparison(pool, rows, directory):
    path = directory / "second/scores.npz"
    with np.load(path, allow_pickle=False) as z:
        indices = pool.feature_indices(z["image"], z["piece_id"])
        values = np.full(len(pool), np.nan, np.float64)
        values[indices] = z["score"]
    take = pool.feature_indices(np.asarray([r["image"] for r in rows]),
                                np.asarray([r["piece_id"] for r in rows]))
    metrics = metrics_for(rows, [dict(logit_diff=float(v)) for v in values[take]])
    return dict(path=str(path), same_sample=metrics,
                full_pool=json.loads((directory / "eval_second.json").read_text())["recall"],
                selected_on="second_cv validation; caller supplies the frozen V1 run")


def time_estimate(processor, model, val, variant, max_new_tokens, test_pool, val_pool, val_images):
    pilot = random.Random(SEED).sample(val, min(128, len(val)))
    # Warm-up excluded. Rendering is timed separately using the same saved piece masks.
    score_one(processor, model, pilot[0], variant, max_new_tokens)
    _, seconds = score_rows(processor, model, pilot, variant, max_new_tokens)
    renderer = Renderer()
    start = time.monotonic()
    render_seconds = 0.0
    selected = {(r["image"], r["piece_id"]) for r in pilot}
    for name, rows in rows_by_image("second_cv"):
        for row in rows:
            if (name, row["piece_id"]) in selected:
                render_start = time.monotonic()
                renderer.render("second_cv", row)
                render_seconds += time.monotonic() - render_start
    scan_seconds = time.monotonic() - start - render_seconds
    # Conservative sum: persisted-image inference + in-memory rendering cost.
    per_piece = (seconds + render_seconds) / len(pilot)
    nval = int(val_pool.image_mask(val_images).sum())
    # Each pool is scanned once; do not multiply a full-file scan by the piece count.
    test_seconds = per_piece * len(test_pool) + scan_seconds * len(test_pool) / len(val_pool)
    val_seconds = per_piece * nval + scan_seconds
    return dict(pilot_split="second_cv/verifier_val_images", pilot_n=len(pilot),
                inference_seconds=seconds, rendering_seconds=render_seconds, cv_pool_scan_seconds=scan_seconds,
                seconds_per_piece=per_piece, pieces_per_second=1 / per_piece,
                test_pieces=len(test_pool), validation_pieces=nval,
                test_gpu_hours=test_seconds / 3600,
                validation_gpu_hours=val_seconds / 3600,
                total_gpu_hours=(test_seconds + val_seconds) / 3600,
                note="Includes pred and unlabeled recall; estimate excludes model loading and CPU E2E")


def score_full(processor, model, pool, images, variant, max_new_tokens, out, run_id):
    selected = pool.image_mask(images)
    scores = np.full(len(pool), np.nan, np.float32)
    deltas = np.full(len(pool), np.nan, np.float32)
    renderer = Renderer()
    allowed, statuses = set(images), Counter()
    folder = out / pool.dataset
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "raw_scores.jsonl").open("x") as handle:
        for name, rows in rows_by_image(pool.dataset):
            if name not in allowed:
                continue
            sl = pool.slices[name]
            if [r["piece_id"] for r in rows] != pool.piece_id[sl].tolist():
                raise ValueError("Pool row order changed")
            for index, row in zip(range(sl.start, sl.stop), rows):
                # Input assembly uses only RGB, mask and crop, even when labels exist.
                result = score_one(processor, model, {}, variant, max_new_tokens,
                                   renderer.render(pool.dataset, row))
                scores[index] = result["score"]
                deltas[index] = result["logit_diff"]
                statuses[result["status"]] += 1
                handle.write(json.dumps(dict(image=name, piece_id=row["piece_id"], **result),
                                        ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{pool.dataset}/{name} complete", flush=True)
    if not np.isfinite(scores[selected]).all():
        raise ValueError("Full-pool scores are incomplete")
    arrays = pool.score_arrays(scores)
    save_npz(folder / "scores.npz", **{k: v[selected] for k, v in arrays.items()},
             logit_diff=deltas[selected], run_id=run_id, dataset=pool.dataset)
    return scores, deltas, dict(statuses)


def run_full(args, processor, model, metadata, test_pool, val_pool, val_images, out):
    from exp6b_eval import calibrate, evaluate_pixels

    # All validation pieces (including unlabeled recall and pred) are scored before test.
    val_scores, val_deltas, val_status = score_full(
        processor, model, val_pool, val_images, args.variant, args.max_new_tokens, out, metadata["run_id"])
    calibration = calibrate(val_pool, val_scores, val_images, out, metadata["run_id"])
    scores, deltas, test_status = score_full(
        processor, model, test_pool, test_pool.images, args.variant, args.max_new_tokens, out, metadata["run_id"])
    classification = grouped_metrics(test_pool, deltas, np.ones(len(test_pool), bool))
    # Pixel evaluation uses the saved probability representation, including its ties.
    probability_metrics = grouped_metrics(test_pool, scores, np.ones(len(test_pool), bool))
    pixels = evaluate_pixels(test_pool, scores, test_pool.images, calibration,
                             probability_metrics["overall"]["recall_80"]["threshold"])
    # Only recall correction is in scope; V2 was never trained on prediction pieces.
    pixels = {k: v for k, v in pixels.items()
              if k == "baseline" or k.startswith("recall/") or k == "pixel_precision_at_recall_80"}
    report = dict(run_id=metadata["run_id"], variant=args.variant, dataset="second",
                  threshold_from="second_cv/verifier_val_images", recall=classification, pixels=pixels,
                  validation_recall=grouped_metrics(val_pool, val_deltas, val_pool.image_mask(val_images)),
                  classification_score="logit_diff", pixel_threshold_score="sigmoid(logit_diff), float32",
                  best_combination=calibration["best_combination"],
                  applied_thresholds={k: v["threshold"] for k, v in calibration["recall"].items()},
                  generation_status=dict(validation=val_status, test=test_status),
                  zero_shot_delta_f1_pp=-0.73, piece_oracle_delta_f1_pp=[6.96, 8.50],
                  e2e_reference="results/exp5c_end_to_end.md; source combinations differ")
    return report


def write_summary(out, report):
    comparisons = report["comparison"]
    current = report["classification"]["overall"]
    v1 = comparisons["v1"]["same_sample"]["overall"]
    lines = ["# V2 测试评估", "", f"- 模型：{report['run_id']}",
             "- V2 与 V1 同一抽样切块、同一分层权重；历史零样本为 300 个平衡整候选。",
             "- 下表召回 80/90% 是测试 ROC 诊断，不用于端到端阈值选择。", "",
             "| 方法 | AUC | FPR@80% | FPR@90% |", "|---|---:|---:|---:|"]
    for name, entry in (("V2", current), ("V1 同池抽样", v1)):
        lines.append(f"| {name} | {entry['auc']:.4f} | {entry['recall_80']['fpr']:.4f} "
                     f"| {entry['recall_90']['fpr']:.4f} |")
    for stage, item in comparisons["zero_shot"].items():
        entry = item["metrics"]
        lines.append(f"| 零样本 Qwen {stage}（历史整候选） | {entry['auc']:.4f} "
                     f"| {entry['fpr_at_recall_0.8']:.4f} | {entry['fpr_at_recall_0.9']:.4f} |")
    lines.extend(["", f"全量测试推理估计：{report['timing']['test_gpu_hours']:.2f} GPU 小时；"
                  f"另需验证池 {report['timing']['validation_gpu_hours']:.2f} 小时。",
                  "详细分来源指标、固定验证阈值结果、生成格式统计见 sample_report.json。",
                  "抽样结果不提供像素去重后的端到端 F1；全池结果见 full/eval_second.json。"])
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("a", "b"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "v2/data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v1-run", type=Path, help="V1 chosen on validation; required for final reports")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--timing-only", action="store_true", help="Only validation pilot, no test inference")
    parser.add_argument("--max-full-hours", type=float, default=8.0)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_full_hours <= 0:
        parser.error("Invalid token or time budget")
    if not args.timing_only and args.v1_run is None:
        parser.error("--v1-run is required for V1 comparison")
    metadata = read_adapter(args.adapter, args.variant)
    if args.max_new_tokens != metadata["config"]["max_new_tokens"]:
        parser.error("Use the same generation limit as adapter validation")
    args.out.mkdir(parents=True, exist_ok=True)
    leaf = args.out / ("timing" if args.timing_only else "full" if args.full else "sample")
    leaf.mkdir(exist_ok=False)
    val = read_rows(args.data / "val.jsonl", "val")
    test_pool, val_pool = Pool("second"), Pool("second_cv")
    _, val_images = verifier_split()
    processor, model = load_model(metadata["model"], args.adapter)
    timing = time_estimate(processor, model, val, args.variant, args.max_new_tokens,
                           test_pool, val_pool, val_images)
    write_json(leaf / "timing.json", dict(run_id=metadata["run_id"], **timing))
    if args.timing_only:
        return
    if args.full:
        if timing["total_gpu_hours"] > args.max_full_hours:
            raise RuntimeError("Estimated test + validation inference exceeds the full-mode budget; see timing.json")
        # Check reference inputs before committing GPU time to the full pool.
        references = historical_comparison()
        rows = read_rows(args.data / "test.jsonl", "test")
        v1 = v1_comparison(test_pool, rows, args.v1_run)
        report = run_full(args, processor, model, metadata, test_pool, val_pool, val_images, leaf)
        with np.load(leaf / "second/scores.npz", allow_pickle=False) as z:
            idx = test_pool.feature_indices(np.asarray([r["image"] for r in rows]),
                                            np.asarray([r["piece_id"] for r in rows]))
            paired = metrics_for(rows, [dict(logit_diff=float(v)) for v in z["logit_diff"][idx]])
        report.update(timing=timing, comparison=dict(v1=v1, zero_shot=references),
                      same_sample=paired)
        write_json(leaf / "eval_second.json", report)
        return
    references = historical_comparison()
    rows = read_rows(args.data / "test.jsonl", "test")
    v1 = v1_comparison(test_pool, rows, args.v1_run)
    val_results, val_seconds = score_rows(processor, model, val, args.variant, args.max_new_tokens)
    val_metrics = binary_metrics([r["label"] for r in val], [r["logit_diff"] for r in val_results])
    if val_metrics["auc"] is None:
        raise ValueError("Validation sample needs both classes")
    write_json(leaf / "validation_thresholds.json", dict(run_id=metadata["run_id"], **val_metrics))
    with (leaf / "raw_scores.jsonl").open("x") as handle:
        results, seconds = score_rows(processor, model, rows, args.variant, args.max_new_tokens, handle)
    report = dict(run_id=metadata["run_id"], variant=args.variant, classification=metrics_for(rows, results),
                  validation=val_metrics, validation_seconds=val_seconds, test_seconds=seconds,
                  frozen_threshold_test=frozen_threshold_metrics(rows, results, val_metrics),
                  timing=timing, comparison=dict(v1=v1, zero_shot=references),
                  generation_status=dict(Counter(r["status"] for r in results)))
    # Horvitz-Thompson ratio over candidate-pixel incidences; overlaps are counted per piece.
    threshold = report["classification"]["overall"]["recall_80"]["threshold"]
    accepted = [r for r, s in zip(rows, results) if s["logit_diff"] >= threshold]
    area = sum(r["weight"] * r["area"] for r in accepted)
    true_area = sum(r["weight"] * r["area"] * r["gt_frac"] for r in accepted)
    report["sampled_pixel_precision_diagnostic"] = dict(
        precision=true_area / area if area else None,
        scope="labeled sampled pieces; overlaps counted repeatedly; includes already-predicted pixels",
        threshold_from="test_weighted_ROC_80_not_for_deployment",
        second_reference_f1_half=0.3708,
        unique_added_pixel_precision="requires --full and evaluate_pixels")
    write_json(leaf / "sample_report.json", report)
    write_summary(leaf, report)


if __name__ == "__main__":
    main()
