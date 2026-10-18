"""Experiment 5b: RemoteCLIP (ViT-L-14) confirmation signals (GPU, conda_envs/exp5-env).

Per candidate and date, an image crop is embedded; signals:
  clip_dissim      1 - cos(e_t1, e_t2)
  clip_cls_diff    1 if the zero-shot argmax over the six SECOND classes differs between dates
  clip_dist_change 1 - sum_c p_t1(c) p_t2(c) over the six-class zero-shot distribution
Two crops: "ctx" = the experiment 3/4b crop (square, 4 x box long side, >= 96 px) taken from the
raw image without the red box; "tight" = 1.5 x box long side, >= 32 px (a small candidate is a
small part of the ctx crop). Writes <subdir>/<dataset>/clip.jsonl. No GT is read.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/guisongxia01/pangchao/wangyihan/Chang-Agent/probe")
from common import OUT, load_rgb  # noqa: E402
from render import crop_box  # noqa: E402

CKPT = "/guisongxia01/pangchao/wangyihan/models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"
CLASSES = ["buildings", "trees", "low vegetation or farmland", "water", "bare ground or road", "a playground"]
PROMPTS = [f"a satellite image of {c}" for c in CLASSES]
CROPS = {"ctx": (4.0, 96), "tight": (1.5, 32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subdir", default="exp4b")
    args = parser.parse_args()
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14")
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        text = model.encode_text(tokenizer(PROMPTS).cuda()).float()
        text = text / text.norm(dim=-1, keepdim=True)
    base = OUT / args.subdir / args.dataset
    rows = [json.loads(line) for line in open(base / "candidates.jsonl")]
    with open(base / "clip.jsonl", "w") as f:
        for r in rows:
            imgs = {v: load_rgb(args.dataset, r["image"], v) for v in ("t1", "t2")}
            out = {"cid": r["cid"]}
            for crop_name, (scale, min_side) in CROPS.items():
                left, top, right, bottom = crop_box(r["bbox"], imgs["t1"].shape[:2], scale, min_side)
                batch = torch.stack([preprocess(Image.fromarray(imgs[v][top:bottom, left:right])) for v in ("t1", "t2")])
                with torch.no_grad():
                    e = model.encode_image(batch.cuda()).float()
                    e = e / e.norm(dim=-1, keepdim=True)
                    p = (100.0 * e @ text.T).softmax(-1)
                out[f"clip_dissim_{crop_name}"] = float(1 - (e[0] * e[1]).sum())
                out[f"clip_cls_diff_{crop_name}"] = float(p[0].argmax() != p[1].argmax())
                out[f"clip_dist_change_{crop_name}"] = float(1 - (p[0] * p[1]).sum())
            f.write(json.dumps(out) + "\n")
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
