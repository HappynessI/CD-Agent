"""Offline GT-based candidate generation for verifier training only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

PerturbationName = Literal["erode", "dilate", "local_delete", "local_add"]


@dataclass(frozen=True)
class VerifierTrainingTarget:
    quality: float
    false_positive_map: np.ndarray
    false_negative_map: np.ndarray
    error_type: str


def perturb_mask(
    gt_mask: np.ndarray,
    kind: PerturbationName,
    *,
    radius: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a candidate. This function must never be called by runtime Environment."""

    gt = np.asarray(gt_mask, dtype=bool)
    if gt.ndim != 2:
        raise ValueError("gt_mask must be 2-D")
    if radius < 1:
        raise ValueError("radius must be positive")
    if kind == "erode":
        return _binary_reduce(gt, radius, mode="all")
    if kind == "dilate":
        return _binary_reduce(gt, radius, mode="any")

    rng = rng or np.random.default_rng()
    height, width = gt.shape
    box_height = min(height, max(1, radius * 2))
    box_width = min(width, max(1, radius * 2))
    y = int(rng.integers(0, height - box_height + 1))
    x = int(rng.integers(0, width - box_width + 1))
    candidate = gt.copy()
    candidate[y : y + box_height, x : x + box_width] = kind == "local_add"
    return candidate


def make_training_target(
    candidate_mask: np.ndarray, gt_mask: np.ndarray
) -> VerifierTrainingTarget:
    candidate = np.asarray(candidate_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if candidate.shape != gt.shape:
        raise ValueError("candidate and GT masks must have identical shape")
    intersection = int(np.logical_and(candidate, gt).sum())
    union = int(np.logical_or(candidate, gt).sum())
    quality = 1.0 if union == 0 else intersection / union
    fp = np.logical_and(candidate, ~gt)
    fn = np.logical_and(~candidate, gt)
    if fp.sum() > fn.sum():
        error_type = "false_positive_change"
    elif fn.any():
        error_type = "false_negative"
    else:
        error_type = "none"
    return VerifierTrainingTarget(quality, fp, fn, error_type)


def _binary_reduce(mask: np.ndarray, radius: int, mode: Literal["all", "any"]) -> np.ndarray:
    padded = np.pad(mask, radius, constant_values=(mode == "all"))
    windows = []
    size = radius * 2 + 1
    for y in range(size):
        for x in range(size):
            windows.append(padded[y : y + mask.shape[0], x : x + mask.shape[1]])
    stack = np.stack(windows)
    return stack.all(axis=0) if mode == "all" else stack.any(axis=0)

