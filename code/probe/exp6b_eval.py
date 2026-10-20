"""Calibrate on second_cv validation, then evaluate a frozen verifier on a target pool."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import AREA_BUCKETS, area_bucket, load_label, load_pred, pixel_metrics
from objects import gt_label
from exp6_pool_io import load_piece_masks
from exp6b_common import (
    BACKBONES, DETECTOR, SOURCES, Features, Pool, binary_metrics, classifier,
    grouped_metrics, predict, run_dir, save_npz, verifier_split, write_json,
)

COMBINATIONS = {
    "B+G0.05": ("B", "G0.05"),
    "B+G0.05+H1": ("B", "G0.05", "H1"),
    "all": ("B", "G0.05", "C1", "C2", "H1"),
}
ABOVE_ONE = float(np.nextafter(np.float64(1), np.float64(2)))
PRED_INDEX = SOURCES.index("pred")


def image_maps(pool, scores, images):
    """Max score implements recall unions; min score implements deletion unions."""
    for number, name in enumerate(images, 1):
        gt = load_label(pool.dataset, name)
        base = load_pred(pool.dataset, DETECTOR[pool.dataset], name)
        if gt.shape != base.shape:
            raise ValueError(f"{name}: GT and detector shape mismatch")
        add = {key: np.full(gt.shape, -np.inf, np.float64) for key in COMBINATIONS}
        remove = {key: np.full(gt.shape, np.inf, np.float64)
                  for key in ("verifier", "detector_mean")}
        masks = load_piece_masks(pool.dataset, name)
        sl = pool.slices[name]
        if set(masks) != set(map(int, pool.piece_id[sl])):
            raise ValueError(f"{name}: metadata and mask piece IDs differ")
        pred_union = np.zeros(gt.shape, bool)
        for i in range(sl.start, sl.stop):
            bbox, local = masks[int(pool.piece_id[i])]
            x0, y0, x1, y1 = bbox
            region = np.s_[y0:y1, x0:x1]
            source = SOURCES[pool.source[i]]
            score = float(scores[i])
            if source == "pred":
                pred_union[region] |= local
                for key, value in (("verifier", score), ("detector_mean", float(pool.meta[i, 0]))):
                    view = remove[key][region]
                    view[local] = np.minimum(view[local], value)
            else:
                for key, sources in COMBINATIONS.items():
                    if source in sources:
                        view = add[key][region]
                        view[local] = np.maximum(view[local], score)
        if not np.array_equal(pred_union, base):
            raise ValueError(f"{name}: pred pieces do not reconstruct the saved detector mask")
        yield name, gt, base, add, remove
        if number % 50 == 0:
            print(f"{pool.dataset}: pixel evaluation {number}/{len(images)} images", flush=True)


def pixel_counts(gt, mask):
    return np.asarray([np.count_nonzero(gt & mask), np.count_nonzero(~gt & mask),
                       np.count_nonzero(gt & ~mask)], np.int64)


def curve_counts(values, thresholds, operation):
    """Number of qualifying pixels at every threshold; no repeated full-image masks."""
    values = values[np.isfinite(values)]
    bins = np.searchsorted(thresholds, values, side="right")
    histogram = np.bincount(bins, minlength=len(thresholds) + 1).astype(np.int64)
    if operation == "recall":
        return np.cumsum(histogram[:0:-1])[::-1]  # score >= threshold
    return np.cumsum(histogram[:-1])             # score < threshold


def threshold_grid(values, operation):
    values = np.unique(np.asarray(values, np.float64))
    if operation == "remove":
        # A deletion changes immediately above a score, since the rule is strictly <.
        values = np.nextafter(values, np.inf)
    return np.unique(np.r_[0.0, values, ABOVE_ONE])


def curve_summary(thresholds, tp, fp, fn, base, operation, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp), np.float64), where=denominator > 0)
    baseline = pixel_metrics(*map(int, base))["f1"]
    best = np.flatnonzero(f1 == f1.max())
    # With equal F1, choose the setting that changes fewer pixels.
    best = int(best[-1] if operation == "recall" else best[0])
    delta = f1 - baseline
    save_npz(path, threshold=thresholds, tp=tp, fp=fp, fn=fn, f1=f1,
             delta_f1=delta, delta_f1_pp=100 * delta, operation=operation)
    near = np.flatnonzero(f1 >= f1[best] - 0.001)
    left = right = best
    while left > 0 and f1[left - 1] >= f1[best] - 0.001:
        left -= 1
    while right + 1 < len(f1) and f1[right + 1] >= f1[best] - 0.001:
        right += 1
    # The NPZ retains all breakpoints; the overview also retains the optimum and its neighbours.
    shown = np.unique(np.r_[np.linspace(0, len(f1) - 1, min(len(f1), 4001), dtype=int),
                            np.arange(max(0, best - 2), min(len(f1), best + 3)), left, right])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(thresholds[shown], delta[shown] * 100, linewidth=1)
    axis.axhline(0, color="grey", linewidth=0.7)
    axis.axvline(thresholds[best], color="tab:red", linestyle="--", label="Selected threshold")
    axis.axvspan(thresholds[left], thresholds[right], color="tab:green", alpha=0.15,
                 label="Within 0.1 F1 pp")
    axis.set(xlabel="Threshold", ylabel="Delta micro F1 (percentage points)", title=path.stem)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)
    return dict(
        threshold=float(thresholds[best]), f1=float(f1[best]), delta_f1=float(delta[best]),
        delta_f1_pp=float(100 * delta[best]), points=len(thresholds), curve=str(path),
        plot=str(path.with_suffix(".svg")),
        near_best_tolerance_pp=0.1, near_best_points=len(near),
        contiguous_near_best_interval=[float(thresholds[left]), float(thresholds[right])],
        neighbours=[dict(threshold=float(thresholds[i]), delta_f1_pp=float(delta[i] * 100))
                    for i in range(max(0, best - 2), min(len(f1), best + 3))],
    )


def calibrate(pool, scores, images, out, run_id):
    selected = pool.image_mask(images)
    grids = {}
    for key, sources in COMBINATIONS.items():
        keep = selected & np.isin(pool.source, [SOURCES.index(s) for s in sources])
        grids["recall/" + key] = threshold_grid(scores[keep], "recall")
    pred = selected & (pool.source == PRED_INDEX)
    grids["remove/verifier"] = threshold_grid(scores[pred], "remove")
    grids["remove/detector_mean"] = threshold_grid(pool.meta[pred, 0], "remove")
    counts = {key: np.zeros((2, len(grid)), np.int64) for key, grid in grids.items()}
    baseline = np.zeros(3, np.int64)
    for _, gt, base, add, remove in image_maps(pool, scores, images):
        baseline += pixel_counts(gt, base)
        for operation, maps in (("recall", add), ("remove", remove)):
            for key, score_map in maps.items():
                full_key = operation + "/" + key
                pixels = ~base if operation == "recall" else base
                for cls, mask in enumerate((gt, ~gt)):
                    counts[full_key][cls] += curve_counts(
                        score_map[pixels & mask], grids[full_key], operation)
    report = dict(run_id=run_id, threshold_from="second_cv",
                  validation_images=images, baseline=pixel_metrics(*map(int, baseline)),
                  recall={}, remove={}, search="all_score_breakpoints_exact_micro_F1")
    for key, changed in counts.items():
        operation, tag = key.split("/")
        sign = 1 if operation == "recall" else -1
        tp = baseline[0] + sign * changed[0]
        fp = baseline[1] + sign * changed[1]
        fn = baseline[2] - sign * changed[0]
        path = out / "curves" / f"{operation}_{tag}.npz"
        report[operation][tag] = curve_summary(grids[key], tp, fp, fn, baseline, operation, path)
    # Stable ordering favours the smaller source set if validation F1 is exactly tied.
    report["best_combination"] = max(COMBINATIONS, key=lambda k: report["recall"][k]["f1"])
    write_json(out / "thresholds_second_cv.json", report)
    return report


def new_accumulator():
    return dict(counts=np.zeros(3, np.int64), added=np.zeros(2, np.int64),
                removed=np.zeros(2, np.int64),
                objects={key: [0, 0] for key in ("all", *(b[2] for b in AREA_BUCKETS))})


def accumulate(accumulator, gt, base, final, object_labels, object_areas, object_buckets):
    accumulator["counts"] += pixel_counts(gt, final)
    added, removed = final & ~base, base & ~final
    accumulator["added"] += [np.count_nonzero(added & gt), np.count_nonzero(added & ~gt)]
    accumulator["removed"] += [np.count_nonzero(removed & gt), np.count_nonzero(removed & ~gt)]
    covered = np.bincount(object_labels[final], minlength=len(object_areas))
    hit = covered[1:] * 2 >= object_areas[1:]
    accumulator["objects"]["all"][0] += int(hit.sum())
    accumulator["objects"]["all"][1] += len(hit)
    for bucket in accumulator["objects"]:
        if bucket == "all":
            continue
        take = object_buckets == bucket
        accumulator["objects"][bucket][0] += int(hit[take].sum())
        accumulator["objects"][bucket][1] += int(take.sum())


def finish(accumulator, baseline_f1):
    report = pixel_metrics(*map(int, accumulator["counts"]))
    report.update(
        delta_f1=report["f1"] - baseline_f1, delta_f1_pp=(report["f1"] - baseline_f1) * 100,
        added_tp=int(accumulator["added"][0]), added_fp=int(accumulator["added"][1]),
        removed_tp=int(accumulator["removed"][0]), removed_fp=int(accumulator["removed"][1]),
        target_recall={key: dict(hit=hit, total=total, recall=hit / total if total else None)
                       for key, (hit, total) in accumulator["objects"].items()},
    )
    return report


def evaluate_pixels(pool, scores, images, calibration, diagnostic_threshold):
    keys = (["baseline"] + ["recall/" + k for k in COMBINATIONS]
            + ["remove/verifier", "remove/detector_mean",
               "combined/verifier", "combined/detector_mean"])
    accumulators = {key: new_accumulator() for key in keys}
    diagnostic = np.zeros(2, np.int64)
    best = calibration["best_combination"]
    for name, gt, base, add, remove in image_maps(pool, scores, images):
        labels, count, _ = gt_label(pool.dataset, name, gt)
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        buckets = np.asarray([area_bucket(int(area)) for area in areas[1:]])
        common = (labels, areas, buckets)
        accumulate(accumulators["baseline"], gt, base, base, *common)
        for key in COMBINATIONS:
            accepted = add[key] >= calibration["recall"][key]["threshold"]
            accumulate(accumulators["recall/" + key], gt, base, base | accepted, *common)
        # Add only outside the original mask. Deletions cannot be undone by overlapping recall pieces.
        addition = (add[best] >= calibration["recall"][best]["threshold"]) & ~base
        for key in ("verifier", "detector_mean"):
            deletion = (remove[key] < calibration["remove"][key]["threshold"]) & base
            cleaned = base & ~deletion
            accumulate(accumulators["remove/" + key], gt, base, cleaned, *common)
            accumulate(accumulators["combined/" + key], gt, base, cleaned | addition, *common)
        if diagnostic_threshold is not None:
            pixels = (add["all"] >= diagnostic_threshold) & ~base
            diagnostic += [np.count_nonzero(pixels & gt), np.count_nonzero(pixels & ~gt)]
    baseline_f1 = pixel_metrics(*map(int, accumulators["baseline"]["counts"]))["f1"]
    result = {key: finish(acc, baseline_f1) for key, acc in accumulators.items()}
    precision = float(diagnostic[0] / diagnostic.sum()) if diagnostic.sum() else None
    result["pixel_precision_at_recall_80"] = dict(
        diagnostic_only=True, threshold_from="current_dataset_labeled_recall_pieces",
        threshold=diagnostic_threshold, added_tp=int(diagnostic[0]), added_fp=int(diagnostic[1]),
        precision=precision, f1_half=baseline_f1 / 2,
        exceeds_f1_half=precision > baseline_f1 / 2 if precision is not None else None,
        second_reference_f1_half=0.3708 if pool.dataset == "second" else None,
    )
    result["combined_sources"] = best
    result["combination_rule"] = "(original_mask minus rejected_pred) union (accepted_recall minus original_mask)"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--include_pred", action="store_true")
    parser.add_argument("--dataset", choices=tuple(DETECTOR), required=True)
    parser.add_argument("--threshold_from", choices=["second_cv"], default="second_cv")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    out = run_dir(args.backbone, args.include_pred)
    report_path = out / f"eval_{args.dataset}.json"
    if report_path.exists():
        raise FileExistsError(f"{report_path}: evaluation already exists; review it before rerunning")
    checkpoint = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    completed = json.loads((out / "val_metrics.json").read_text())
    if (checkpoint["run_id"] != completed["run_id"] or checkpoint["backbone"] != args.backbone
            or checkpoint["include_pred"] != args.include_pred):
        raise ValueError("Model and completed training report do not identify the same run")
    train_images, val_images = verifier_split()
    if checkpoint["train_images"] != train_images or checkpoint["val_images"] != val_images:
        raise ValueError("Verifier split changed since training")
    threshold_path = out / "thresholds_second_cv.json"
    calibration = None
    if threshold_path.exists():
        calibration = json.loads(threshold_path.read_text())
        if (calibration["run_id"] != checkpoint["run_id"]
                or calibration["validation_images"] != val_images):
            raise ValueError("Frozen thresholds belong to a different model or validation split")
    elif args.dataset != "second_cv":
        raise FileNotFoundError("First run --dataset second_cv to freeze validation thresholds")
    pool = Pool(args.dataset)
    features = Features(pool, args.backbone, checkpoint["feature_config"])
    model = classifier(checkpoint["input_dim"], checkpoint["config"]["dropout"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    scores = predict(model, features, np.ones(len(pool), bool), checkpoint["mean"].numpy(),
                     checkpoint["std"].numpy(), args.batch_size, args.device)
    save_npz(out / args.dataset / "scores.npz", **pool.score_arrays(scores),
             run_id=checkpoint["run_id"], dataset=args.dataset)
    images = val_images if args.dataset == "second_cv" else pool.images
    selected = pool.image_mask(images)
    if calibration is None:
        calibration = calibrate(pool, scores, images, out, checkpoint["run_id"])
    classification = grouped_metrics(pool, scores, selected)
    pred = selected & (pool.source == PRED_INDEX) & (pool.label >= 0)
    report = dict(
        run_id=checkpoint["run_id"], backbone=args.backbone, include_pred=args.include_pred,
        dataset=args.dataset, split="verifier_val_images" if args.dataset == "second_cv" else "test",
        images=len(images), scored_pieces=len(pool), evaluated_pieces=int(selected.sum()),
        threshold_from="second_cv/verifier_val_images",
        thresholds_file=str(threshold_path), feature_config=features.config,
        recall=classification,
        pred=dict(verifier=binary_metrics(pool.label[pred], scores[pred]),
                  detector_mean=binary_metrics(pool.label[pred], pool.meta[pred, 0])),
        pixels=evaluate_pixels(
            pool, scores, images, calibration,
            classification["overall"]["recall_80"]["threshold"]),
        applied_thresholds=dict(
            recall={k: v["threshold"] for k, v in calibration["recall"].items()},
            remove={k: v["threshold"] for k, v in calibration["remove"].items()},
            combined_sources=calibration["best_combination"]),
    )
    write_json(report_path, report)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
