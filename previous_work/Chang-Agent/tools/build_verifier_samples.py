#!/usr/bin/env python3
"""Build offline Verifier samples from paired change-detection images.

This is an offline/GT-facing utility: labels are read only while constructing
training targets and are never passed into ``ChangeAgentEnvironment``.  The
feature tensor intentionally contains image evidence and a perturbed candidate
mask, not the ground-truth mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from change_agent.perturbations import make_training_target, perturb_mask


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.dataset_root
    rng = np.random.default_rng(args.seed)
    labels = sorted((root / "label_cvt").glob("*.png"))
    if args.max_samples > 0:
        labels = labels[: args.max_samples]
    if not labels:
        raise SystemExit(f"no label_cvt PNG files found under {root}")

    features, quality, error_map = [], [], []
    error_type = []
    type_ids = {"none": 0, "false_positive_change": 1, "false_negative": 2}
    for index, label_path in enumerate(labels):
        t1_path, t2_path = root / "A" / label_path.name, root / "B" / label_path.name
        if not t1_path.exists() or not t2_path.exists():
            continue
        t1, t2, gt = read_rgb(t1_path), read_rgb(t2_path), read_mask(label_path)
        kind = ("erode", "dilate", "local_delete", "local_add")[index % 4]
        candidate = perturb_mask(gt, kind, radius=1, rng=rng)
        target = make_training_target(candidate, gt)
        absdiff = np.abs(t2 - t1)
        # 3 channels T1 + 3 channels T2 + 3 abs-difference + candidate/error cues.
        candidate_f = candidate.astype(np.float32)[..., None]
        features.append(np.concatenate([t1, t2, absdiff, candidate_f], axis=2).transpose(2, 0, 1))
        quality.append(target.quality)
        error_map.append((target.false_positive_map | target.false_negative_map)[None])
        error_type.append(type_ids[target.error_type])
    if not features:
        raise SystemExit("no complete A/B/label triplets found")
    arrays = {
        "features": np.stack(features).astype(np.float32),
        "quality": np.asarray(quality, dtype=np.float32),
        "error_map": np.stack(error_map).astype(np.float32),
        "error_type": np.asarray(error_type, dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "dataset_root": str(root),
        "samples": len(features),
        "feature_channels": int(arrays["features"].shape[1]),
        "seed": args.seed,
        "target_view_supervision": "omitted; no real target-view labels are available",
        "schema_version": 3,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
