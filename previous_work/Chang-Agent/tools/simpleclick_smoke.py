#!/usr/bin/env python3
"""Load the real SimpleClick checkpoint and execute one external-mask point action."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/Data/wyh/CD-SegAgent/models/SimpleClick/cocolvis_vit_large.pth",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("/tmp/change_agent_simpleclick_smoke.json"))
    args = parser.parse_args()
    from evaltools.model_loader import SegmentationModel
    from third_party.SimpleClick.isegm.inference import utils
    from third_party.SimpleClick.isegm.inference.predictors import get_predictor

    start = time.monotonic()
    before = rss_mb()
    device = torch.device(args.device)
    model = utils.load_is_model(args.checkpoint, device, False)
    predictor = get_predictor(
        model,
        "NoBRS",
        device,
        prob_thresh=0.49,
        with_flip=False,
        # The ViT-L checkpoint has a 28x28 positional grid (448px input).
        # Force the full image through the same resize path used by SegAgent.
        zoom_in_params={"target_size": (448, 448), "skip_clicks": -1},
        predictor_params={"optimize_after_n_clicks": 1},
    )
    segmentation_model = SegmentationModel(predictor)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    initial_mask = np.zeros((256, 256), dtype=bool)
    # Exercise the same adapter used by Environment, including [H,W] -> [1,1,H,W].
    from change_agent.adapters.segagent_adapter import SimpleClickAdapter

    result_mask = SimpleClickAdapter(segmentation_model).refine(
        image, initial_mask, (128, 128), True
    )
    cuda_stats = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        cuda_stats.update(
            allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 2),
            reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 2),
            peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 2),
        )
    result = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "rss_before_mb": round(before, 2),
        "rss_peak_mb": round(rss_mb(), 2),
        "mask_shape": list(result_mask.shape),
        "mask_pixels": int(result_mask.sum()),
        "cuda": cuda_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
