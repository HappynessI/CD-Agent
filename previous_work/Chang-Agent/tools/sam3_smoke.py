#!/usr/bin/env python3
"""Load the local OmniOVCD SAM3 checkpoint and exercise text/box prompts."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Data/wyh/CD-SegAgent/models/sam3/sam3.pt")
    parser.add_argument("--bpe", default="/Data/wyh/CD-SegAgent/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--output", type=Path, default=Path("/tmp/change_agent_sam3_smoke.json"))
    args = parser.parse_args()
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    start = time.monotonic()
    before = rss_mb()
    model = build_sam3_image_model(
        bpe_path=args.bpe,
        checkpoint_path=args.checkpoint,
        device=args.device,
        load_from_HF=False,
    )
    # SAM3's upstream code caches a device selected during construction. Explicitly
    # reset it after checkpoint loading so CPU smoke cannot mix cached CUDA tensors.
    import torch

    model.to(args.device)
    model._device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    processor = Sam3Processor(model, resolution=args.resolution, device=args.device)
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt="building", state=state)
    text_count = int(state.get("masks", np.zeros((0,))).shape[0])
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt="building", state=state)
    state = processor.add_geometric_prompt(
        box=[0.5, 0.5, 0.5, 0.5], label=True, state=state
    )
    box_count = int(state.get("masks", np.zeros((0,))).shape[0])
    result = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "resolution": args.resolution,
        "load_and_prompt_seconds": round(time.monotonic() - start, 3),
        "rss_before_mb": round(before, 2),
        "rss_peak_mb": round(rss_mb(), 2),
        "text_mask_count": text_count,
        "box_mask_count": box_count,
    }
    if torch.cuda.is_available():
        result["cuda"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
