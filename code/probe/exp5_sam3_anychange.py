"""Experiment 5a H3: the AnyChange recipe with SAM3 instead of SAM1 (GPU, omniovcd-env).

AnyChange: class-agnostic mask proposals on each date (SAM automatic mask generation from a grid
of point prompts), then for every proposal compare the mean image-encoder feature inside it on t1
vs t2 (bitemporal: proposals from both dates are scored), keep those whose features differ.
Here both parts come from SAM3:
- proposals: SAM3's SAM1-task interactive head (predict_inst) prompted with a 32x32 grid of single
  positive points, batched; per point the best of the 3 multimask outputs by predicted IoU;
  kept if predicted IoU >= 0.8 and area < 80% of the image; box NMS at IoU 0.7. This mirrors
  SAM's automatic mask generator; SAM3's text grounding needs a category, so it is not used here.
- features: SAM3 vision_features (256-d, 72x72), bilinearly upsampled; score = -cos(mean f_t1,
  mean f_t2) inside the mask (AnyChange's change_confidence).
- threshold: AnyChange's auto_threshold option (per-image Otsu over pixel-wise -cos), because the
  default 145-degree threshold was calibrated on SAM1 features.
Writes exp5/<dataset>/sam3_anychange/<image>.npz (packed masks, scores, threshold). No GT is read.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import batched_nms

from common import OUT, load_rgb, names
from sam3_tools import autocast, build_processor

GRID, CHUNK = 32, 64
IOU_MIN, AREA_MAX, NMS = 0.8, 0.8, 0.7


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[np.argmax(var)])


def proposals(processor, model, image):
    h, w = image.shape[:2]
    with autocast():
        state = processor.set_image(Image.fromarray(image))
    feat = state["backbone_out"]["vision_features"][0].float()
    ys, xs = (np.arange(GRID) + 0.5) / GRID * h, (np.arange(GRID) + 0.5) / GRID * w
    pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2).astype(np.float32)
    masks, scores = [], []
    for i in range(0, len(pts), CHUNK):
        chunk = pts[i:i + CHUNK]
        with autocast():
            m, s, _ = model.predict_inst(state, point_coords=chunk, point_labels=np.ones((len(chunk), 1)),
                                         multimask_output=True)
        m, s = np.asarray(m), np.asarray(s)
        if m.ndim == 3:                       # single prompt returns C x H x W
            m, s = m[None], s[None]
        best = s.argmax(1)
        masks.append(m[np.arange(len(m)), best] > 0)
        scores.append(s[np.arange(len(s)), best])
    masks, scores = np.concatenate(masks), np.concatenate(scores)
    area = masks.reshape(len(masks), -1).sum(1)
    keep = (scores >= IOU_MIN) & (area > 0) & (area < AREA_MAX * h * w)
    masks, scores = masks[keep], scores[keep]
    if not len(masks):
        return masks, feat
    boxes = []
    for m in masks:
        yy, xx = np.nonzero(m)
        boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    keep = batched_nms(torch.tensor(boxes, dtype=torch.float32), torch.from_numpy(scores).float(),
                       torch.zeros(len(boxes), dtype=torch.long), NMS).numpy()
    return masks[keep], feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    processor = build_processor(inst_interactivity=True)
    model = processor.model
    out = OUT / "exp5" / args.dataset / "sam3_anychange"
    out.mkdir(parents=True, exist_ok=True)
    items = names(args.dataset)[: args.limit or None]
    items = [n for i, n in enumerate(items) if i % args.shards == args.shard]
    start = time.time()
    for index, name in enumerate(items):
        if (out / name.replace(".png", ".npz")).exists():
            continue
        imgs = [load_rgb(args.dataset, name, v) for v in ("t1", "t2")]
        h, w = imgs[0].shape[:2]
        (m1, f1), (m2, f2) = proposals(processor, model, imgs[0]), proposals(processor, model, imgs[1])
        f1 = F.interpolate(f1[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        f2 = F.interpolate(f2[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        neg_cos = -F.cosine_similarity(f1, f2, dim=0).cpu().numpy()
        thr = otsu(neg_cos.ravel())
        kept, scores = [], []
        for m in list(m1) + list(m2):
            mt = torch.from_numpy(m).to(f1.device)
            s = float(-F.cosine_similarity(f1[:, mt].mean(1), f2[:, mt].mean(1), dim=0))
            if s > thr:
                kept.append(m)
                scores.append(s)
        packed = np.packbits(np.stack(kept).reshape(len(kept), -1), axis=1) if kept else np.zeros((0, 0), np.uint8)
        np.savez_compressed(out / name.replace(".png", ".npz"), h3_masks=packed,
                            h3_scores=np.array(scores, np.float32), shape=np.array((h, w)), threshold=np.array(thr),
                            n_proposals=np.array([len(m1), len(m2)]))
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(items)} {time.time() - start:.0f}s", flush=True)
    (out.parent / f"sam3_anychange_report_{args.shard}.json").write_text(json.dumps(
        {"images": len(items), "grid": GRID, "pred_iou_min": IOU_MIN, "area_max": AREA_MAX, "nms": NMS,
         "threshold": "per-image Otsu over pixel -cos (AnyChange auto_threshold)",
         "seconds": round(time.time() - start, 1)}, indent=2))
    print("done", len(items), flush=True)


if __name__ == "__main__":
    main()
