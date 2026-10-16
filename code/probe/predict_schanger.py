"""Run official SChanger LEVIR/WHU checkpoints and save full-image probability maps.

Protocol follows the vendored SChanger evaluation: non-overlapping 256x256 tiles, no
resize/TTA, per-dataset training normalization, threshold 0.5 for the binary mask.
Run in conda_envs/vlm-vs-onejev-env on a GPU node.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from common import DATASETS, PRED, load_label, load_rgb, names, pixel_metrics, save_mask, write_json

SCHANGER = "/guisongxia01/pangchao/wangyihan/Vlm_vs_OneJev/vendor/SChanger"
WEIGHTS = "/guisongxia01/pangchao/wangyihan/models/SChanger"
TILE = 256
# Reference test F1 (%) from the vendored SChanger evaluation README.
REFERENCE_F1 = {("small", "levir"): 92.4391, ("base", "levir"): 92.8831,
                ("small", "whu"): 93.5141, ("base", "whu"): 93.2020,
                ("small", "sysu"): 84.5759, ("base", "sysu"): 84.1692}


def main():
    parser = argparse.ArgumentParser()
    with_weights = [d for d in DATASETS if "schanger" in DATASETS[d]]
    parser.add_argument("--dataset", choices=(*with_weights, "all"), default="all")
    parser.add_argument("--model", choices=("small", "base", "all"), default="all")
    args = parser.parse_args()
    for dataset in with_weights if args.dataset == "all" else (args.dataset,):
        for model in ("small", "base") if args.model == "all" else (args.model,):
            run(dataset, model)


def run(dataset, model_size):
    args = argparse.Namespace(dataset=dataset, model=model_size)
    sys.path.insert(0, SCHANGER)
    from cd.SChanger import schanger_base, schanger_small

    suffix = DATASETS[args.dataset]["schanger"]
    stats = json.load(open(f"{SCHANGER}/evaluation/normalization/{suffix}.json"))
    mean = [np.asarray(stats[g]["means"], np.float32) * 255 for g in ("A", "B")]
    scale = [1 / (np.asarray(stats[g]["stds"], np.float32) * 255) for g in ("A", "B")]
    builder = {"small": schanger_small, "base": schanger_base}[args.model]
    model = builder(f"{suffix}-cd", pretrained=False)
    state = torch.load(f"{WEIGHTS}/schanger_{args.model}_{suffix}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    model.cuda().eval()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    out = PRED / args.dataset / f"schanger_{args.model}"
    counts = np.zeros(3, dtype=np.int64)
    start = time.time()
    items = names(args.dataset)
    for index, name in enumerate(items):
        images = [load_rgb(args.dataset, name, view) for view in ("t1", "t2")]
        height, width = images[0].shape[:2]
        coords = [(y, x) for y in range(0, height, TILE) for x in range(0, width, TILE)]
        tensors = []
        for image, m, s in zip(images, mean, scale):
            tiles = [(image[y:y + TILE, x:x + TILE].astype(np.float32) - m) * s for y, x in coords]
            tensors.append(torch.from_numpy(np.stack(tiles).transpose(0, 3, 1, 2).copy()).cuda())
        with torch.inference_mode():
            probs = model((tensors[0], tensors[1]))[:, 0].float().cpu().numpy()
        prob = np.zeros((height, width), np.float32)
        for (y, x), tile in zip(coords, probs):
            prob[y:y + TILE, x:x + TILE] = tile
        (out / "prob").mkdir(parents=True, exist_ok=True)
        np.save(out / "prob" / name.replace(".png", ".npy"), prob.astype(np.float16))
        pred = prob > 0.5
        save_mask(out / "mask" / name, pred)
        truth = load_label(args.dataset, name)
        counts += [(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()]
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(items)}", flush=True)
    metrics = pixel_metrics(*map(int, counts))
    report = {"dataset": args.dataset, "detector": f"schanger_{args.model}", "images": len(items),
              "threshold": 0.5, "tile": TILE, "metrics": metrics,
              "reference_f1_percent": REFERENCE_F1[(args.model, suffix)],
              "seconds": round(time.time() - start, 1)}
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
