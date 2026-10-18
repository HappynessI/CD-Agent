"""Experiment 5b: detector local re-check of candidates (GPU, vlm-vs-onejev-env).

For each candidate, a square crop centred on its box (side = max(4 x long side, 64 px), clipped to
the image, the same expansion as the experiment 3/4b MLLM crops) is passed again through the same
detector, and the mean change probability inside the candidate mask is recorded:
  recheck_zoom    crop resized to the detector's native input size (SECOND 512, SYSU 256);
  recheck_native  crop at its original resolution, zero-padded to a multiple of 32 (context only).
Candidate masks are regenerated deterministically (B: SAM3 class-difference components; G0.05:
detector probability > 0.05 components; H*: stored masks) and matched by bounding box.
Writes <subdir>/<dataset>/recheck.jsonl. No GT is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import OUT, load_prob, load_rgb
from objects import label

VOJ = "/guisongxia01/pangchao/wangyihan/Vlm_vs_OneJev"
NATIVE = {"second": 512, "sysu": 256}
DETECTOR = {"sysu": "schanger_small", "second": "mambascd_base"}
SCALE, MIN_SIDE = 4.0, 64


def build(dataset):
    if dataset == "sysu":
        sys.path.insert(0, f"{VOJ}/vendor/SChanger")
        from cd.SChanger import schanger_small
        model = schanger_small("sysu-cd", pretrained=False)
        model.load_state_dict(torch.load(f"{VOJ}/checkpoints/schanger_small_sysu.pth", map_location="cpu",
                                         weights_only=True)["state_dict"], strict=True)
        stats = json.load(open(f"{VOJ}/vendor/SChanger/evaluation/normalization/sysu.json"))
        norm = [(np.asarray(stats[g]["means"]) * 255, np.asarray(stats[g]["stds"]) * 255) for g in ("A", "B")]
    else:
        sys.path.insert(0, f"{VOJ}/vendor/ChangeMamba")
        from changedetection.configs.config import get_config
        from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
        from changedetection.script.script_utils import get_vssm_kwargs, load_checkpoint
        cfg = get_config(SimpleNamespace(cfg=f"{VOJ}/vendor/ChangeMamba/changedetection/configs/vssm1/vssm_base_224.yaml", opts=None))
        model = ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=None, **get_vssm_kwargs(cfg))
        load_checkpoint(model, f"{VOJ}/checkpoints/MambaSCD_Base_SECOND_SeK_0.2292.pth")
        m, s = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
        norm = [(m, s), (m, s)]
    return model.cuda().eval(), norm


def infer(dataset, model, norm, a, b):
    t = []
    for img, (m, s) in zip((a, b), norm):
        x = ((img.astype(np.float32) - m.astype(np.float32)) / s.astype(np.float32)).transpose(2, 0, 1)
        t.append(torch.from_numpy(np.ascontiguousarray(x))[None].cuda())
    with torch.inference_mode():
        if dataset == "sysu":
            return model((t[0], t[1]))[0, 0].float()
        return model(t[0], t[1])[0].softmax(1)[0, 1].float()


def candidate_mask(dataset, name, source, bbox, cache):
    key = (name, source)
    if key not in cache:
        if source == "G0.05":
            labels, _ = label(load_prob(dataset, DETECTOR[dataset], name) > 0.05)
        elif source == "B":
            z = np.load(OUT / "exp4" / dataset / "sam3" / name.replace(".png", ".npz"))
            labels, _ = label((z["cls_t1"] > 0) & (z["cls_t2"] > 0) & (z["cls_t1"] != z["cls_t2"]))
        else:
            path = "anychange" if source in ("H1", "H2") else "sam3_anychange"
            z = np.load(OUT / "exp5" / dataset / path / name.replace(".png", ".npz"))
            shape = tuple(z["shape"])
            packed = z[f"{source.lower()}_masks"]
            masks = [np.unpackbits(r)[:shape[0] * shape[1]].reshape(shape).astype(bool) for r in packed]
            cache[key] = ("masks", masks)
            return pick(cache[key], bbox)
        cache[key] = ("labels", labels, ndimage.find_objects(labels))
    return pick(cache[key], bbox)


def pick(entry, bbox):
    x0, y0, x1, y1 = bbox
    if entry[0] == "labels":
        labels, slices = entry[1], entry[2]
        for k, sl in enumerate(slices, start=1):
            if sl and [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop] == [x0, y0, x1, y1]:
                return labels == k
    else:
        for m in entry[1]:
            ys, xs = np.nonzero(m)
            if len(xs) and [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] == [x0, y0, x1, y1]:
                return m
    raise KeyError(f"candidate mask not found for bbox {bbox}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subdir", default="exp4b")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    model, norm = build(args.dataset)
    base = OUT / args.subdir / args.dataset
    rows = [json.loads(line) for line in open(base / "candidates.jsonl")]
    cache = {}
    with open(base / "recheck.jsonl", "w") as f:
        for r in rows:
            name = r["image"]
            mask = candidate_mask(args.dataset, name, r["source"], r["bbox"], cache)
            a, b = load_rgb(args.dataset, name, "t1"), load_rgb(args.dataset, name, "t2")
            h, w = mask.shape
            x0, y0, x1, y1 = r["bbox"]
            side = int(min(max(SCALE * max(x1 - x0, y1 - y0), MIN_SIDE), h, w))
            left = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
            top = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
            ca, cb = a[top:top + side, left:left + side], b[top:top + side, left:left + side]
            cm = torch.from_numpy(mask[top:top + side, left:left + side]).cuda()
            n = NATIVE[args.dataset]
            za = F.interpolate(torch.from_numpy(ca).permute(2, 0, 1)[None].float(), size=(n, n), mode="bilinear",
                               align_corners=False)[0].permute(1, 2, 0).numpy()
            zb = F.interpolate(torch.from_numpy(cb).permute(2, 0, 1)[None].float(), size=(n, n), mode="bilinear",
                               align_corners=False)[0].permute(1, 2, 0).numpy()
            pz = infer(args.dataset, model, norm, za, zb)
            pz = F.interpolate(pz[None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0]
            pad = (-side) % 32
            pa = np.pad(ca, ((0, pad), (0, pad), (0, 0)), mode="reflect")
            pb = np.pad(cb, ((0, pad), (0, pad), (0, 0)), mode="reflect")
            pn = infer(args.dataset, model, norm, pa, pb)[:side, :side]
            f.write(json.dumps({"cid": r["cid"], "recheck_zoom": float(pz[cm].mean()),
                                "recheck_native": float(pn[cm].mean()), "crop_side": side}) + "\n")
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
