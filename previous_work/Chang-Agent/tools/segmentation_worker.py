#!/usr/bin/env python3
"""Execute one real SimpleClick point or SAM3 box action for a parent rollout."""

from __future__ import annotations

import argparse
import json
import resource
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


def point(args: argparse.Namespace, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    from third_party.SimpleClick.isegm.inference import utils
    from third_party.SimpleClick.isegm.inference.predictors import get_predictor

    from change_agent.adapters.segagent_adapter import SimpleClickAdapter

    device = torch.device(args.device)
    model = utils.load_is_model(args.checkpoint, device, False)
    predictor = get_predictor(
        model,
        "NoBRS",
        device,
        prob_thresh=0.49,
        with_flip=False,
        zoom_in_params={"target_size": (448, 448), "skip_clicks": -1},
        predictor_params={"optimize_after_n_clicks": 1},
    )
    initial_mask = np.asarray(np.load(args.initial_mask), dtype=bool)
    click_history = tuple(
        ((int(x), int(y)), bool(is_positive))
        for x, y, is_positive in args.history_click
    )
    result = SimpleClickAdapter(predictor).refine(
        image,
        initial_mask,
        tuple(args.coordinate),
        bool(args.is_positive),
        click_history,
    )
    return result, {
        "tool": "simpleclick",
        "checkpoint": args.checkpoint,
        "coordinate": args.coordinate,
        "is_positive": bool(args.is_positive),
        "accepted_click_history": [
            {"coordinate": list(coordinate), "is_positive": is_positive}
            for coordinate, is_positive in click_history
        ],
        "session_click_count": len(click_history) + 1,
        "initial_mask_pixels": int(initial_mask.sum()),
    }


def box(args: argparse.Namespace, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter

    processor = _build_sam3_processor(args)
    with _sam3_autocast(args.device):
        result = SAM3ProcessorAdapter(processor).segment_box(
            image, tuple(args.box), args.query
        )
    return result, {
        "tool": "sam3",
        "checkpoint": args.checkpoint,
        "bpe": args.bpe,
        "resolution": args.resolution,
        "query": args.query,
        "box_cxcywh_normalized": args.box,
    }


def initialize(
    args: argparse.Namespace, image1: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Run fresh SAM3 text prompting for both temporal views with one model load."""

    from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter

    image2 = np.asarray(Image.open(args.image_t2).convert("RGB"))
    if image2.shape[:2] != image1.shape[:2]:
        raise ValueError("T1 and T2 initialization images must have the same shape")
    adapter = SAM3ProcessorAdapter(_build_sam3_processor(args))
    with _sam3_autocast(args.device):
        t1_mask, t1_evidence = adapter.segment_text(image1, args.query)
        t2_mask, t2_evidence = adapter.segment_text(image2, args.query)
    np.save(args.output_mask_t2, np.asarray(t2_mask, dtype=np.uint8))
    evidence_dir = args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "t1": _persist_arrays(t1_evidence, evidence_dir, "t1"),
        "t2": _persist_arrays(t2_evidence, evidence_dir, "t2"),
    }
    return t1_mask, {
        "tool": "sam3",
        "operation": "fresh_dual_view_text_initialization",
        "checkpoint": args.checkpoint,
        "bpe": args.bpe,
        "resolution": args.resolution,
        "query": args.query,
        "t1_mask": str(args.output_mask),
        "t2_mask": str(args.output_mask_t2),
        "intermediate_artifacts": artifacts,
    }


def _build_sam3_processor(args: argparse.Namespace):
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        bpe_path=args.bpe,
        checkpoint_path=args.checkpoint,
        device=args.device,
        load_from_HF=False,
    )
    model.to(args.device)
    model._device = torch.device(args.device)
    return Sam3Processor(model, resolution=args.resolution, device=args.device)


def _sam3_autocast(device: str):
    """Match the mixed-precision context required by the official SAM3 inference path."""

    import torch

    if torch.device(device).type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _persist_arrays(
    evidence: dict[str, object], output_dir: Path, prefix: str
) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for name, value in evidence.items():
        array = np.asarray(value)
        path = output_dir / f"{prefix}_{name}.npy"
        np.save(path, array)
        numeric = array.astype(float, copy=False) if np.issubdtype(array.dtype, np.number) else None
        record: dict[str, object] = {
            "file": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if numeric is not None and array.size:
            record.update(
                min=float(np.nanmin(numeric)),
                max=float(np.nanmax(numeric)),
                mean=float(np.nanmean(numeric)),
            )
        manifest[name] = record
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("point", "box", "initialize"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-t2", type=Path)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-mask-t2", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-mask", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coordinate", nargs=2, type=int)
    parser.add_argument("--is-positive", type=int, choices=(0, 1))
    parser.add_argument(
        "--history-click",
        action="append",
        nargs=3,
        type=int,
        default=[],
        metavar=("X", "Y", "IS_POSITIVE"),
    )
    parser.add_argument("--bpe")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--query")
    parser.add_argument("--box", nargs=4, type=float)
    args = parser.parse_args()
    if args.mode == "point" and (
        args.initial_mask is None or args.coordinate is None or args.is_positive is None
    ):
        parser.error("point requires --initial-mask, --coordinate, and --is-positive")
    if args.mode == "point" and any(
        is_positive not in (0, 1) for _, _, is_positive in args.history_click
    ):
        parser.error("point history IS_POSITIVE values must be 0 or 1")
    if args.mode == "box" and (args.bpe is None or args.query is None or args.box is None):
        parser.error("box requires --bpe, --query, and --box")
    if args.mode == "initialize" and (
        args.bpe is None
        or args.query is None
        or args.image_t2 is None
        or args.output_mask_t2 is None
        or args.evidence_dir is None
    ):
        parser.error(
            "initialize requires --bpe, --query, --image-t2, --output-mask-t2, and --evidence-dir"
        )

    start = time.monotonic()
    before = rss_mb()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    if args.mode == "point":
        result, details = point(args, image)
    elif args.mode == "box":
        result, details = box(args, image)
    else:
        result, details = initialize(args, image)
    result = np.asarray(result, dtype=bool)
    if result.shape != image.shape[:2]:
        raise ValueError(f"worker mask shape {result.shape} != image shape {image.shape[:2]}")
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, result.astype(np.uint8))
    report = {
        "status": "success",
        "mode": args.mode,
        "device": args.device,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "rss_before_mb": round(before, 2),
        "rss_peak_mb": round(rss_mb(), 2),
        "output_shape": list(result.shape),
        "output_pixels": int(result.sum()),
        **details,
    }
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda"] = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
                "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2),
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            }
    except ImportError:
        report["cuda_available"] = False
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
