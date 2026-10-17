"""Experiment 4a, step 1: everything that needs SAM3, one pass per image (GPU, omniovcd-env).

Per image and date: set_image once, then text prompts.
  B  six SECOND land-cover classes via text prompts (several prompts per class, max over them);
     per pixel class = argmax over classes if the best confidence >= CLASS_MIN_CONF else 0.
     Confidence of a prompt = max over kept instances of (instance score x mask probability).
     Land-cover "stuff" classes rarely reach SAM3's default instance threshold 0.5 (92% of pixels
     stayed unlabelled in a smoke test), so instances are kept from score 0.1 for B; the
     "building" instances used by C still require score >= 0.5.
  C  "building" instances (score > 0.5) as a label map, for cross-date instance matching.
  D  cosine distance between L2-normalised t1/t2 SAM3 vision features (256-d, 72x72 patches).
  E  exemplar retrieval: detector components with probability > 0.9 (on the 72x72 patch grid)
     give exemplar vectors [f_t1, f_t2]; every patch gets its max cosine similarity to any
     exemplar. Images without exemplars store an empty map and are counted.
Writes exp4/<dataset>/sam3/<name>.npz. No GT is read.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import OUT, load_prob, load_rgb, names
from sam3_tools import autocast, build_processor

DETECTOR = {"sysu": "schanger_small", "second": "mambascd_base"}
# SECOND classes: 1 low vegetation, 2 non-vegetated ground surface, 3 tree, 4 water, 5 building, 6 playground.
PROMPTS = {1: ("grass", "farmland"), 2: ("bare ground", "road", "parking lot"), 3: ("tree",),
           4: ("water",), 5: ("building",), 6: ("sports field",)}
CLASS_MIN_CONF = 0.1
INSTANCE_MIN_SCORE = 0.1      # processor keep threshold for the class prompts (B)
BUILDING_MIN_SCORE = 0.5      # instances kept for C
EXEMPLAR_PROB = 0.9


def prompt_conf(state, shape):
    masks = state.get("masks_logits")
    if masks is None or len(state["scores"]) == 0:
        return np.zeros(shape, np.float32), None
    prob = masks[:, 0].float() * state["scores"].float()[:, None, None]
    return prob.max(0).values.cpu().numpy(), state


def instance_map(state, shape):
    labels = np.zeros(shape, np.uint16)
    if state is None or len(state["scores"]) == 0:
        return labels, np.zeros(0, np.float32)
    scores = state["scores"].float().cpu().numpy()
    masks = state["masks"][:, 0].cpu().numpy()
    keep = np.nonzero(scores >= BUILDING_MIN_SCORE)[0]
    scores, masks = scores[keep], masks[keep]
    for k in np.argsort(scores):            # higher scores painted last
        labels[masks[k]] = k + 1
    return labels, scores


def components_grid(mask):
    """8-connected components on a small boolean grid (no scipy in this environment)."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    count = 0
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        count += 1
        stack = [(y, x)]
        labels[y, x] = count
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DETECTOR), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    processor = build_processor()
    processor.confidence_threshold = INSTANCE_MIN_SCORE
    out = OUT / "exp4" / args.dataset / "sam3"
    out.mkdir(parents=True, exist_ok=True)
    items = names(args.dataset)[: args.limit or None]
    no_exemplar = 0
    start = time.time()
    for index, name in enumerate(items):
        result, feats = {}, {}
        for view in ("t1", "t2"):
            image = load_rgb(args.dataset, name, view)
            shape = image.shape[:2]
            with autocast():
                state = processor.set_image(Image.fromarray(image))
                feat = state["backbone_out"]["vision_features"][0].float()
                feats[view] = F.normalize(feat, dim=0)                       # 256 x 72 x 72
                conf = np.zeros((7,) + shape, np.float32)
                building_state = None
                for cls, prompts in PROMPTS.items():
                    for prompt in prompts:
                        processor.reset_all_prompts(state)
                        state = processor.set_text_prompt(prompt=prompt, state=state)
                        c, kept = prompt_conf(state, shape)
                        conf[cls] = np.maximum(conf[cls], c)
                        if prompt == "building":
                            building_state = {k: kept[k] for k in ("masks", "scores")} if kept is not None else None
            best = conf[1:].max(0)
            cls_map = (conf[1:].argmax(0) + 1).astype(np.uint8)
            cls_map[best < CLASS_MIN_CONF] = 0
            result[f"cls_{view}"] = cls_map
            result[f"conf_{view}"] = best.astype(np.float16)
            labels, scores = instance_map(building_state, shape)
            result[f"inst_{view}"] = labels
            result[f"inst_scores_{view}"] = scores
        f1, f2 = feats["t1"], feats["t2"]
        result["dist"] = (1 - (f1 * f2).sum(0)).cpu().numpy().astype(np.float16)
        grid = f1.shape[1:]
        prob = torch.from_numpy(load_prob(args.dataset, DETECTOR[args.dataset], name))[None, None]
        prob_grid = F.interpolate(prob, size=grid, mode="area")[0, 0].numpy()
        labels, count = components_grid(prob_grid > EXEMPLAR_PROB)
        concat = F.normalize(torch.cat([f1, f2], 0), dim=0).flatten(1)       # 512 x P
        if count:
            flat = torch.from_numpy(labels.ravel()).to(concat.device)
            exemplars = torch.stack([concat[:, flat == k].mean(1) for k in range(1, count + 1)])
            sim = (F.normalize(exemplars, dim=1) @ concat).max(0).values.reshape(grid)
            result["esim"] = sim.cpu().numpy().astype(np.float16)
        else:
            result["esim"] = np.zeros((0, 0), np.float16)
            no_exemplar += 1
        result["n_exemplars"] = np.array(count)
        np.savez_compressed(out / name.replace(".png", ".npz"), **result)
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(items)} {time.time() - start:.0f}s", flush=True)
    (OUT / "exp4" / args.dataset / "sam3_report.json").write_text(json.dumps(
        {"images": len(items), "no_exemplar_images": no_exemplar, "prompts": PROMPTS,
         "class_min_conf": CLASS_MIN_CONF, "instance_min_score": INSTANCE_MIN_SCORE,
         "building_min_score": BUILDING_MIN_SCORE, "exemplar_prob": EXEMPLAR_PROB,
         "seconds": round(time.time() - start, 1)}, indent=2))
    print("done", len(items), "no exemplar", no_exemplar, flush=True)


if __name__ == "__main__":
    main()
