#!/usr/bin/env python3
"""Train or smoke-test the lightweight frozen-feature Verifier head.

The ``--smoke`` path creates deterministic synthetic candidates and never reads GT
inside the runtime Environment. Real training should provide an ``.npz`` containing
features and the target arrays documented in ``--help``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from change_agent.perturbations import make_training_target, perturb_mask
from change_agent.verifier_model import build_verifier_head, verifier_loss


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


def make_synthetic_samples(count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    features, quality, error_map = [], [], []
    error_type = []
    type_ids = {"none": 0, "false_positive_change": 1, "false_negative": 2}
    for index in range(count):
        gt = np.zeros((16, 16), dtype=bool)
        y, x = 3 + index % 4, 3 + (index * 3) % 4
        gt[y : y + 5, x : x + 5] = True
        kind = ("erode", "dilate", "local_delete", "local_add")[index % 4]
        candidate = perturb_mask(gt, kind, radius=1, rng=rng)
        target = make_training_target(candidate, gt)
        # A real pipeline replaces these frozen maps with SAM3 features/evidence.
        image_features = rng.normal(0, 1, (9, 16, 16)).astype(np.float32)
        image_features[6] = gt.astype(np.float32)
        image_features[7] = candidate.astype(np.float32)
        image_features[8] = (candidate ^ gt).astype(np.float32)
        features.append(image_features)
        quality.append(target.quality)
        error_map.append((target.false_positive_map | target.false_negative_map)[None])
        error_type.append(type_ids[target.error_type])
    return {
        "features": np.stack(features),
        "quality": np.asarray(quality, dtype=np.float32),
        "error_map": np.stack(error_map).astype(np.float32),
        "error_type": np.asarray(error_type, dtype=np.int64),
    }


def load_samples(path: Path | None, count: int, seed: int) -> dict[str, np.ndarray]:
    if path is None:
        return make_synthetic_samples(count, seed)
    with np.load(path) as data:
        required = {"features", "quality", "error_map", "error_type"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"sample file is missing keys: {sorted(missing)}")
        return {key: data[key] for key in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, help="NPZ with verifier training arrays")
    parser.add_argument("--smoke", action="store_true", help="Use deterministic synthetic data")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/tmp/change_agent_verifier_smoke.pt"))
    args = parser.parse_args()
    if args.samples is None and not args.smoke:
        parser.error("pass --smoke or --samples")
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Verifier training requires the train dependencies") from error

    torch.manual_seed(args.seed)
    samples = load_samples(args.samples, args.count, args.seed)
    tensors = {key: torch.from_numpy(value) for key, value in samples.items()}
    model = build_verifier_head(int(tensors["features"].shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    start = time.monotonic()
    losses: list[float] = []
    for _ in range(args.epochs):
        order = torch.randperm(tensors["features"].shape[0])
        for begin in range(0, len(order), args.batch_size):
            indices = order[begin : begin + args.batch_size]
            batch = {key: value[indices] for key, value in tensors.items()}
            predictions = model(batch["features"])
            loss = verifier_loss(predictions, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "seed": args.seed,
            "schema_version": 3,
            "target_view_supervision": "omitted",
        },
        args.output,
    )
    result = {
        "mode": "smoke" if args.smoke else "dataset",
        "samples": int(tensors["features"].shape[0]),
        "epochs": args.epochs,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "rss_after_mb": round(rss_mb(), 2),
        "cuda_available": bool(torch.cuda.is_available()),
        "output": str(args.output),
        "pid": os.getpid(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
