"""Official SCanNet SECOND test binary head, without ImageNet download or skimage.

Run in vlm-vs-onejev-env. Install the accompanying models/SCanNet.py constructor
change first. Normalization and no-TTA inference follow datasets/RS_ST.py and
pred_SCD.py. Semantic Fscd in the checkpoint filename is not binary change F1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import PRED, WORKSPACE, load_label, load_rgb, names, pixel_metrics, save_mask, write_json

VENDOR = WORKSPACE / "Chang-Agent/third_party/SCanNet"
CHECKPOINT = WORKSPACE / "models/SCanNet/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth"
MEANS = ([113.40, 114.08, 116.45], [111.07, 114.04, 118.18])
STDS = ([48.30, 46.27, 48.14], [49.41, 47.01, 47.94])
DATASET = "second_scannet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=PRED / DATASET / "scannet")
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; requires a separate --output")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit and args.output.resolve() == (PRED / DATASET / "scannet").resolve():
        parser.error("Smoke runs require --output outside the formal prediction directory")
    items = names(DATASET)
    if len(items) != 1694:
        raise ValueError(f"Expected 1694 SECOND test pairs, found {len(items)}")
    items = items[:args.limit or None]
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    sys.path.insert(0, str(VENDOR))
    from models.SCanNet import SCanNet

    model = SCanNet(in_channels=3, num_classes=7, input_size=512, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)  # official train_SCD_psd.py saves net.state_dict()
    del state
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    (args.output / "prob").mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.unlink(missing_ok=True)
    counts = np.zeros(3, np.int64)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, name in enumerate(items):
            pair = []
            for view, mean, std in zip(("t1", "t2"), MEANS, STDS):
                image = load_rgb(DATASET, name, view)
                if image.shape != (512, 512, 3):
                    raise ValueError(f"{name}/{view}: expected RGB 512x512, got {image.shape}")
                # The official loader normalizes in float64, then casts to float32 on GPU.
                normalized = ((image - np.asarray(mean)) / np.asarray(std)).astype(np.float32)
                pair.append(torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].cuda())
            logits, _, _ = model(*pair)
            prob = logits.sigmoid()[0, 0].float().cpu().numpy()
            pred = prob > 0.5
            np.save(args.output / "prob" / (Path(name).stem + ".npy"), prob.astype(np.float16))
            save_mask(args.output / "mask" / name, pred)
            truth = load_label(DATASET, name)
            counts += [(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()]
            if (index + 1) % 50 == 0:
                print(f"{index + 1}/{len(items)}", flush=True)
    report = {
        "dataset": DATASET, "source_dataset": "second", "split": "test", "detector": "scannet",
        "checkpoint": str(args.checkpoint), "images": len(items), "image_names": items,
        "full_test": len(items) == 1694, "threshold": 0.5, "threshold_before_float16": True,
        "probability": "sigmoid(binary_change_head)", "tta": False, "precision": "float32",
        "normalization": {"mean_t1_t2": MEANS, "std_t1_t2": STDS},
        "imagenet_pretrained": False, "checkpoint_load": "strict",
        "metrics": pixel_metrics(*map(int, counts)),
        "reference_binary_f1": None,
        "reference_note": "Checkpoint Fscd63.66 is semantic change F-score, not binary F1.",
        "seconds": time.perf_counter() - started,
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
