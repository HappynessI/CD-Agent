"""Experiment 5a: AnyChange (torchange 0.0.4, SAM ViT-H) as a recall-candidate source.

H1  full-image mode with the settings of the official example: points_per_side=32,
    stability_score_thresh=0.95, change_confidence_threshold=145, normalised features,
    bitemporal match. Candidates = output change masks, score = change_confidence
    (negative cosine between the two dates' mean features inside the mask).
H2  point query (multi_points_match): query points are the interior centres (distance-transform
    maximum) of the up-to-5 largest detector components with probability > 0.9; each point is
    used on both dates. Images without such components are skipped and counted.
Writes exp5/<dataset>/anychange/<image>.npz with packed masks and scores for H1 and H2.
Run in conda_envs/exp5-env on a GPU. No GT is read.
Supports second_cv and --h1_only; H1-only files contain empty H2 arrays and h2_computed=False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/guisongxia01/pangchao/wangyihan/Chang-Agent/probe")
from common import OUT, load_prob, load_rgb, names  # noqa: E402

DETECTOR = {"sysu": "schanger_small", "second": "mambascd_base", "second_cv": "mambascd_base",
            "second_scannet": "scannet", "levir": "schanger_base", "whu": "schanger_base",
            "lsscd": "mambascd_base", "scscd7": "mambascd_base"}
SAM_CKPT = "/guisongxia01/pangchao/wangyihan/models/SAM/sam_vit_h_4b8939.pth"
MAX_QUERY = 5


def pack(cmasks, shape):
    from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask
    masks = [rle_to_mask(r) for r in cmasks["rles"]]
    scores = cmasks["change_confidence"].float().cpu().numpy() if len(masks) else np.zeros(0, np.float32)
    packed = np.packbits(np.stack(masks).reshape(len(masks), -1), axis=1) if masks else np.zeros((0, 0), np.uint8)
    return packed, scores


def query_points(prob):
    labels, n = ndimage.label(prob > 0.9, np.ones((3, 3), bool))
    if n == 0:
        return []
    areas = ndimage.sum_labels(np.ones_like(prob), labels, np.arange(1, n + 1))
    points = []
    for k in np.argsort(-areas)[:MAX_QUERY] + 1:
        dist = ndimage.distance_transform_edt(labels == k)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        points += [np.array([x, y, 1]), np.array([x, y, 2])]
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DETECTOR), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--h1_only", action="store_true")
    parser.add_argument("--output", type=Path, help="Override the anychange directory, e.g. for a smoke run")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("require --shards >= 1 and 0 <= --shard < --shards")
    from torchange.models.segment_any_change import AnyChange

    model = AnyChange("vit_h", sam_checkpoint=SAM_CKPT)
    model.make_mask_generator(points_per_side=32, stability_score_thresh=0.95)
    model.set_hyperparameters(change_confidence_threshold=145, use_normalized_feature=True, bitemporal_match=True)
    out = args.output or OUT / "exp5" / args.dataset / "anychange"
    out.mkdir(parents=True, exist_ok=True)
    items = names(args.dataset)[: args.limit or None]
    items = [n for i, n in enumerate(items) if i % args.shards == args.shard]
    start, no_query, h2_failed = time.time(), 0, 0
    skipped = 0
    for index, name in enumerate(items):
        path = out / name.replace(".png", ".npz")
        if path.exists():
            if not args.h1_only:
                with np.load(path) as saved:
                    if "h2_computed" in saved and not bool(saved["h2_computed"]):
                        raise ValueError(f"{path}: H1-only output exists; move it before requesting H2")
            skipped += 1
            continue
        img1, img2 = load_rgb(args.dataset, name, "t1"), load_rgb(args.dataset, name, "t2")
        shape = img1.shape[:2]
        cmasks, _, _ = model.forward(img1, img2)
        h1_masks, h1_scores = pack(cmasks, shape)
        model.clear_cached_embedding()
        points = [] if args.h1_only else query_points(load_prob(args.dataset, DETECTOR[args.dataset], name))
        h2 = None
        # torchange's multi_points_match raises IndexError when a query point yields no mask;
        # retry with t2-only, then t1-only points, else record H2 as empty for this image.
        for subset in (points, [p for p in points if p[2] == 2], [p for p in points if p[2] == 1]):
            if not subset:
                continue
            try:
                h2 = model.multi_points_match(subset, img1, img2)
                break
            except IndexError:
                h2_failed += 1
            finally:
                model.clear_cached_embedding()
        if h2 is not None:
            h2_masks, h2_scores = pack(h2, shape)
        else:
            h2_masks, h2_scores = np.zeros((0, 0), np.uint8), np.zeros(0, np.float32)
            no_query += int(not args.h1_only)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, h1_masks=h1_masks, h1_scores=h1_scores,
                            h2_masks=h2_masks, h2_scores=h2_scores, shape=np.array(shape),
                            n_query=np.array(len(points) // 2), h2_computed=np.array(not args.h1_only))
        temporary.replace(path)
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(items)} {time.time() - start:.0f}s", flush=True)
    (out.parent / f"anychange_report_{args.shard}.json").write_text(json.dumps(
        {"dataset": args.dataset, "detector_for_h2": None if args.h1_only else DETECTOR[args.dataset],
         "images": len(items), "skipped_images": skipped, "h1_only": args.h1_only,
         "shard": args.shard, "shards": args.shards,
         "no_query_or_failed_images": no_query, "h2_failed_attempts": h2_failed, "sam": "vit_h", "points_per_side": 32,
         "stability_score_thresh": 0.95, "change_confidence_threshold": 145,
         "max_query_components": MAX_QUERY, "seconds": round(time.time() - start, 1)}, indent=2))
    print("done", len(items), "no query", no_query, flush=True)


if __name__ == "__main__":
    main()
