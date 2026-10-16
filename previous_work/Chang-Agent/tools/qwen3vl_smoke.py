#!/usr/bin/env python3
"""Load a local Qwen3-VL checkpoint and run one structured-action generation."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.state import AgentObservation


def rss_mb() -> float:
    # Linux reports KiB; macOS reports bytes. This script runs on Linux servers.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


def cuda_memory() -> dict[str, object]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2),
        }
    except Exception as error:
        return {"cuda_available": False, "error": repr(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen3vl_smoke.json"))
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Use cpu for fallback or auto to place the model on visible CUDA devices.",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    observation = AgentObservation(
        t1_image=image,
        t2_image=image.copy(),
        query="building change",
        change_mask=np.zeros((64, 64), dtype=bool),
    )
    before = rss_mb()
    start = time.monotonic()
    adapter = GroundingModelQwen3VL(
        str(model_path),
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
    )
    load_seconds = time.monotonic() - start
    after_load = rss_mb()
    raw, action = adapter.act(observation)
    result = {
        "model_path": str(model_path),
        "device_map": args.device_map,
        "load_seconds": round(load_seconds, 3),
        "rss_before_mb": round(before, 2),
        "rss_after_load_mb": round(after_load, 2),
        "rss_delta_mb": round(after_load - before, 2),
        "cuda": cuda_memory(),
        "raw_response": raw,
        "parsed_action": action.to_dict(),
        "transformers": __import__("transformers").__version__,
        "pid": os.getpid(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
