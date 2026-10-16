#!/usr/bin/env python3
"""Independently validate GT-direct, teacher-only tool reachability artifacts.

The validator never executes SAM3, SimpleClick, or Qwen.  It reloads every
saved mask, recomputes geometry, metrics, endpoint decisions, production hard
gates, track eligibility, selection order, and replay identity.  Scientific
failure is a valid result (exit status zero); malformed or incomplete evidence
is an infrastructure failure (non-zero).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


CONFIG_SCHEMA_ID = "qwen3vl_gt_direct_teacher_preregistration_v1"
EPISODE_SCHEMA_ID = "qwen3vl_gt_direct_teacher_episode_v1"
INDEX_SCHEMA_ID = "qwen3vl_gt_direct_teacher_shard_index_v1"
COMPLETE_SCHEMA_ID = "qwen3vl_gt_direct_teacher_generation_complete_v1"
REPORT_SCHEMA_ID = "qwen3vl_gt_direct_teacher_validation_report_v1"
PREFLIGHT_SCHEMA_ID = "qwen3vl_gt_direct_teacher_preflight_receipt_v1"
ARTIFACT_MANIFEST_SCHEMA_ID = "qwen3vl_gt_direct_teacher_validation_artifacts_v1"

TRACK_NAMES = (
    "matched_safe",
    "wide_safe",
    "wide_hard",
    "wide_raw",
)
SHA256_LENGTH = 64
GIT_SHA_LENGTH = 40
EXPECTED_SHARD_COUNT = 4
DEFAULT_ENDPOINT_IOU = 0.90
DEFAULT_ENDPOINT_F1 = 18.0 / 19.0
DEFAULT_EMPTY_GT_MAX_FP = 64
DEFAULT_EPSILON = 1e-12
RUNTIME_BOUNDARY_LAYERS = {
    "environment": {
        "complete_gt_derived_proposal_metadata_present": True,
        "raw_label_pixels_or_path_present": False,
    },
    "executor_and_receipt": {
        "gt_derived_fields": [
            "point_coordinate_or_box",
            "proposal_box",
            "component_area",
            "authorized_region_id_encoding_teacher_route",
        ],
        "raw_label_pixels_or_path_present": False,
    },
    "sam3_backend": {
        "gt_derived_fields": ["box"],
        "raw_label_pixels_or_path_present": False,
    },
    "simpleclick_backend": {
        "gt_derived_fields": [
            "point_coordinate",
            "proposal_localized_current_mask_context",
        ],
        "raw_label_pixels_or_path_present": False,
    },
}


class ValidationError(ValueError):
    """An artifact failed its frozen validation contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    require(isinstance(value, list), f"{location} must be a list")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == GIT_SHA_LENGTH
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_nonfinite(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite JSON number at {location}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{location}[{index}]")


def read_json(path: Path, location: str | None = None) -> dict[str, Any]:
    label = location or str(path)

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    _reject_nonfinite(value, label)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValidationError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha256(mask_hashes: Mapping[str, str]) -> str:
    return canonical_sha256(
        {
            "t1_mask_sha256": mask_hashes["t1"],
            "t2_mask_sha256": mask_hashes["t2"],
            "change_mask_sha256": mask_hashes["change"],
        }
    )


def _inside(root: Path, relative: Any, location: str) -> Path:
    require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute(),
        f"{location} must be a non-empty relative path",
    )
    resolved_root = root.resolve()
    path = (resolved_root / str(relative)).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValidationError(f"{location} escapes artifact root") from error
    return path


def _verify_file_spec(root: Path, value: Any, location: str) -> Path:
    spec = _mapping(value, location)
    path = _inside(root, spec.get("path"), f"{location}.path")
    require(path.is_file(), f"{location} is missing: {path}")
    expected = spec.get("sha256")
    require(_is_sha256(expected), f"{location}.sha256 is invalid")
    require(sha256_file(path) == str(expected).lower(), f"{location} SHA256 differs")
    if "size_bytes" in spec:
        require(
            type(spec["size_bytes"]) is int
            and spec["size_bytes"] == path.stat().st_size,
            f"{location}.size_bytes differs",
        )
    return path


def _load_binary_mask(path: Path) -> np.ndarray:
    try:
        if path.suffix.lower() == ".npy":
            value = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                value = np.asarray(image.convert("L"))
    except (OSError, ValueError) as error:
        raise ValidationError(f"cannot load mask {path}: {error}") from error
    require(value.ndim == 2, f"mask must be rank two: {path}")
    if value.dtype == np.bool_:
        return np.asarray(value, dtype=bool)
    unique = set(np.unique(value).tolist())
    require(unique.issubset({0, 1, 255}), f"mask is not binary: {path}")
    return np.asarray(value != 0, dtype=bool)


def _verified_mask(root: Path, value: Any, location: str) -> np.ndarray:
    spec = _mapping(value, location)
    mask = _load_binary_mask(_verify_file_spec(root, spec, location))
    expected = spec.get("mask_sha256")
    require(_is_sha256(expected), f"{location}.mask_sha256 is invalid")
    require(mask_sha256(mask) == str(expected).lower(), f"{location} semantic hash differs")
    require(spec.get("shape") == list(mask.shape), f"{location} shape differs")
    return mask


@dataclass(frozen=True)
class StateArtifact:
    masks: dict[str, np.ndarray]
    mask_hashes: dict[str, str]
    identity: str
    receipt: dict[str, Any]


def _load_state(root: Path, value: Any, location: str) -> StateArtifact:
    item = _mapping(value, location)
    container = _mapping(item.get("masks", item), f"{location}.masks")
    aliases = {
        "t1": ("t1_mask", "t1"),
        "t2": ("t2_mask", "t2"),
        "change": ("change_mask", "change"),
    }
    masks: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name, names in aliases.items():
        matches = [key for key in names if key in container]
        require(matches, f"{location}.{name}_mask is missing")
        if len(matches) > 1:
            require(
                container[matches[0]] == container[matches[1]],
                f"{location}.{name}_mask aliases differ",
            )
        masks[name] = _verified_mask(
            root, container[matches[0]], f"{location}.{name}_mask"
        )
        hashes[name] = mask_sha256(masks[name])
    require(
        masks["t1"].shape == masks["t2"].shape == masks["change"].shape,
        f"{location} mask shapes differ",
    )
    identity = state_sha256(hashes)
    declared = item.get("state_sha256", item.get("state_hash"))
    require(_is_sha256(declared), f"{location}.state_sha256 is invalid")
    require(str(declared).lower() == identity, f"{location} state SHA256 differs")
    declared_hashes = item.get("state_mask_sha256")
    if declared_hashes is not None:
        require(dict(_mapping(declared_hashes, f"{location}.state_mask_sha256")) == hashes,
                f"{location}.state_mask_sha256 differs")
    return StateArtifact(masks, hashes, identity, copy.deepcopy(dict(item)))


def _same_state(left: StateArtifact, right: StateArtifact, location: str) -> None:
    require(left.identity == right.identity, f"{location} state identity differs")
    for name in ("t1", "t2", "change"):
        require(np.array_equal(left.masks[name], right.masks[name]),
                f"{location} {name} mask differs")


def _rebuild_change_mask(
    t1_mask: np.ndarray, t2_mask: np.ndarray, matching_value: Any
) -> np.ndarray:
    matching = _mapping(matching_value, "matching")
    mode = matching.get("mode", "overlap_presence")
    threshold = float(matching.get("overlap_threshold", 0.25))
    t12_minimum = int(matching.get("t12_min_instance_area", 0))
    cd_minimum = int(matching.get("cd_min_instance_area", 0))
    require(mode in {"overlap_presence", "greedy_one_to_one"},
            "matching mode is invalid")
    require(0 <= threshold <= 1 and t12_minimum >= 0 and cd_minimum >= 0,
            "matching thresholds are invalid")
    t1_instances = _connected_components(t1_mask)
    t2_instances = _connected_components(t2_mask)
    candidates: list[tuple[int, int, float, float]] = []
    for left, mask1 in enumerate(t1_instances):
        area1 = int(mask1.sum())
        for right, mask2 in enumerate(t2_instances):
            intersection = int(np.logical_and(mask1, mask2).sum())
            if intersection:
                candidates.append(
                    (left, right, intersection / area1,
                     intersection / int(mask2.sum()))
                )
    if mode == "overlap_presence":
        matched_t1 = {left for left, _, coverage, _ in candidates if coverage >= threshold}
        matched_t2 = {right for _, right, _, coverage in candidates if coverage >= threshold}
    else:
        ranked = sorted(
            (
                (min(left_coverage, right_coverage), left, right)
                for left, right, left_coverage, right_coverage in candidates
                if min(left_coverage, right_coverage) >= threshold
            ),
            reverse=True,
        )
        matched_t1: set[int] = set()
        matched_t2: set[int] = set()
        for _, left, right in ranked:
            if left not in matched_t1 and right not in matched_t2:
                matched_t1.add(left)
                matched_t2.add(right)
    change = np.zeros_like(np.asarray(t1_mask, dtype=bool))
    for index, instance in enumerate(t1_instances):
        if index not in matched_t1 and int(instance.sum()) >= t12_minimum:
            change |= instance
    for index, instance in enumerate(t2_instances):
        if index not in matched_t2 and int(instance.sum()) >= t12_minimum:
            change |= instance
    if cd_minimum:
        filtered = np.zeros_like(change)
        for component in _connected_components(change):
            if int(component.sum()) >= cd_minimum:
                filtered |= component
        change = filtered
    return change


def _require_rebuilt_change(
    state: StateArtifact, context: "ValidationContext", location: str
) -> None:
    rebuilt = _rebuild_change_mask(
        state.masks["t1"], state.masks["t2"], context.base_config.get("matching", {})
    )
    require(np.array_equal(rebuilt, state.masks["change"]),
            f"{location} change mask differs from independent MaskPairProcessor rebuild")


def compute_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    require(prediction.shape == reference.shape, "prediction/GT shape mismatch")
    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, ~reference).sum())
    fn = int(np.logical_and(~prediction, reference).sum())
    tn = int(prediction.size) - tp - fp - fn
    union = tp + fp + fn
    predicted = tp + fp
    gt = tp + fn
    iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if predicted == 0 and gt == 0 else (
        0.0 if predicted == 0 else tp / predicted
    )
    recall = 1.0 if gt == 0 else tp / gt
    denominator = 2 * tp + fp + fn
    f1 = 1.0 if denominator == 0 else 2 * tp / denominator
    return {
        "pixel_error": fp + fn,
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": predicted,
        "gt_pixels": gt,
    }


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pixel_error_reduction": int(before["pixel_error"]) - int(after["pixel_error"]),
        "iou": float(after["iou"]) - float(before["iou"]),
        "f1": float(after["f1"]) - float(before["f1"]),
        "tp": int(after["tp"]) - int(before["tp"]),
        "fp": int(after["fp"]) - int(before["fp"]),
        "fn": int(after["fn"]) - int(before["fn"]),
    }


def _number_equal(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return isinstance(left, (int, float)) and math.isclose(
            float(left), right, rel_tol=0.0, abs_tol=1e-12
        )
    return left == right


def _require_numeric_mapping(
    recorded: Any, expected: Mapping[str, Any], location: str, *, exact_keys: bool = False
) -> None:
    value = _mapping(recorded, location)
    if exact_keys:
        require(set(value) == set(expected), f"{location} field inventory differs")
    for key, expected_value in expected.items():
        require(key in value, f"{location}.{key} is missing")
        require(_number_equal(value[key], expected_value),
                f"{location}.{key} differs from recomputation")


@dataclass(frozen=True)
class EndpointContract:
    minimum_iou: float
    minimum_f1: float
    empty_gt_max_fp: int
    epsilon: float


def endpoint_gate(metrics: Mapping[str, Any], contract: EndpointContract) -> dict[str, Any]:
    nonempty = int(metrics["gt_pixels"]) > 0
    if nonempty:
        iou_passed = float(metrics["iou"]) + contract.epsilon >= contract.minimum_iou
        f1_passed = float(metrics["f1"]) + contract.epsilon >= contract.minimum_f1
        fp_passed = None
        passed = iou_passed and f1_passed
    else:
        iou_passed = None
        f1_passed = None
        fp_passed = int(metrics["fp"]) <= contract.empty_gt_max_fp
        passed = fp_passed
    return {
        "passed": bool(passed),
        "nonempty_gt": nonempty,
        "minimum_iou": contract.minimum_iou if nonempty else None,
        "minimum_f1": contract.minimum_f1 if nonempty else None,
        "empty_gt_max_fp": contract.empty_gt_max_fp if not nonempty else None,
        "iou_passed": iou_passed,
        "f1_passed": f1_passed,
        "fp_residual_passed": fp_passed,
        "comparison_epsilon": contract.epsilon,
    }


def _connected_components(mask: np.ndarray, *, four_neighbour: bool = False) -> list[np.ndarray]:
    value = np.asarray(mask, dtype=bool)
    require(value.ndim == 2, "component mask must be rank two")
    remaining = value.copy()
    height, width = remaining.shape
    result: list[np.ndarray] = []
    if four_neighbour:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = tuple(
            (dx, dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dx or dy
        )
    while remaining.any():
        start_y, start_x = np.argwhere(remaining)[0]
        remaining[start_y, start_x] = False
        stack = [(int(start_x), int(start_y))]
        pixels: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            pixels.append((x, y))
            for dx, dy in offsets:
                next_x, next_y = x + dx, y + dy
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and remaining[next_y, next_x]
                ):
                    remaining[next_y, next_x] = False
                    stack.append((next_x, next_y))
        component = np.zeros_like(value, dtype=bool)
        xs, ys = zip(*pixels)
        component[np.asarray(ys), np.asarray(xs)] = True
        result.append(component)
    return result


def _error_mask(source: StateArtifact, ground_truth: np.ndarray, kind: str) -> np.ndarray:
    normalized = kind.lower().replace("-", "_")
    if normalized in {"false_negative", "fn", "missing", "missing_change"}:
        return np.logical_and(ground_truth, ~source.masks["change"])
    if normalized in {"false_positive", "fp", "spurious", "spurious_change"}:
        return np.logical_and(source.masks["change"], ~ground_truth)
    raise ValidationError(f"unknown teacher error kind: {kind}")


def _distance_transform_seed(mask: np.ndarray) -> tuple[int, int]:
    source = np.asarray(mask, dtype=bool)
    require(source.any(), "cannot seed an empty component")
    remaining = source.copy()
    distance = np.zeros(source.shape, dtype=np.int32)
    while remaining.any():
        distance[remaining] += 1
        padded = np.pad(remaining, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(remaining)
        for offset_y in range(3):
            for offset_x in range(3):
                eroded &= padded[
                    offset_y : offset_y + remaining.shape[0],
                    offset_x : offset_x + remaining.shape[1],
                ]
        remaining = eroded
    seed_y, seed_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(seed_x), int(seed_y)


def _padded_component_box(
    tight: Sequence[int], shape: tuple[int, int], ratio: float, *, production_minimum: bool
) -> tuple[list[int], int, bool]:
    x1, y1, x2, y2 = (int(item) for item in tight)
    height, width = shape
    padding = round(max(x2 - x1 + 1, y2 - y1 + 1) * ratio)
    if production_minimum:
        padding = max(4, padding)
    padded = [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    ]
    adjusted = False
    if padded[0] == padded[2]:
        adjusted = True
        if padded[2] < width - 1:
            padded[2] += 1
        else:
            padded[0] -= 1
    if padded[1] == padded[3]:
        adjusted = True
        if padded[3] < height - 1:
            padded[3] += 1
        else:
            padded[1] -= 1
    return padded, padding, adjusted


def _teacher_component_inventory(
    source: StateArtifact, ground_truth: np.ndarray
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for error_kind in ("fn", "fp"):
        components = _connected_components(_error_mask(source, ground_truth, error_kind))
        components.sort(
            key=lambda component: (
                -int(component.sum()),
                int(np.nonzero(component)[0].min()),
                int(np.nonzero(component)[1].min()),
                int(np.nonzero(component)[0].max()),
                int(np.nonzero(component)[1].max()),
                mask_sha256(component),
            )
        )
        for component in components:
            ys, xs = np.nonzero(component)
            raw.append(
                {
                    "error_kind": error_kind,
                    "component": component,
                    "component_area": int(component.sum()),
                    "component_box_pixels": [
                        int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    ],
                    "component_mask_sha256": mask_sha256(component),
                }
            )
    raw.sort(
        key=lambda item: (
            -int(item["component_area"]),
            str(item["error_kind"]),
            (
                int(item["component_box_pixels"][1]),
                int(item["component_box_pixels"][0]),
                int(item["component_box_pixels"][3]),
                int(item["component_box_pixels"][2]),
            ),
            str(item["component_mask_sha256"]),
        )
    )
    for rank, item in enumerate(raw[:4]):
        item["component_rank"] = rank
        item["component_id"] = f"{item['error_kind']}-c{rank:03d}"
    return raw[:4]


def _proposal_from_geometry(value: Any, location: str) -> dict[str, Any]:
    geometry = dict(_mapping(value, location))
    seed = geometry.get("component_seed_pixels", geometry.get("seed_xy"))
    box = geometry.get("box_pixels", geometry.get("box_xyxy"))
    area = geometry.get("component_area")
    require(
        isinstance(seed, list)
        and len(seed) == 2
        and all(type(item) is int for item in seed),
        f"{location} seed is invalid",
    )
    require(
        isinstance(box, list)
        and len(box) == 4
        and all(type(item) is int for item in box),
        f"{location} box is invalid",
    )
    require(type(area) is int and area > 0, f"{location}.component_area is invalid")
    geometry["component_seed_pixels"] = list(seed)
    geometry["box_pixels"] = list(box)
    geometry["component_area"] = area
    return geometry


def _validate_teacher_geometry(
    value: Any,
    *,
    source: StateArtifact,
    ground_truth: np.ndarray,
    location: str,
) -> dict[str, Any]:
    geometry = _proposal_from_geometry(value, location)
    require(
        geometry.get("teacher_only") is True,
        f"{location} must declare teacher_only=true",
    )
    source_name = geometry.get("geometry_source", geometry.get("source"))
    require(
        source_name in {None, "gt_direct", "offline_gt_direct_teacher"}
        and geometry.get("gt_derived_geometry", True) is True,
        f"{location} is not GT-direct geometry",
    )
    family = geometry.get(
        "geometry_family",
        geometry.get("family", geometry.get("geometry_mode")),
    )
    require(family in {"matched", "wide"}, f"{location} geometry family is invalid")
    error_kind = geometry.get("error_kind", geometry.get("error_type"))
    require(isinstance(error_kind, str), f"{location} error kind is invalid")
    error = _error_mask(source, ground_truth, error_kind)
    seed_x, seed_y = geometry["component_seed_pixels"]
    height, width = error.shape
    require(
        0 <= seed_x < width and 0 <= seed_y < height and error[seed_y, seed_x],
        f"{location} seed is outside declared GT error",
    )
    matching = [component for component in _connected_components(error) if component[seed_y, seed_x]]
    require(len(matching) == 1, f"{location} does not identify one GT error component")
    component = matching[0]
    require(
        int(component.sum()) == geometry["component_area"],
        f"{location} component area differs",
    )
    if "component_mask_sha256" in geometry:
        require(
            geometry["component_mask_sha256"] == mask_sha256(component),
            f"{location} component semantic hash differs",
        )
    ys, xs = np.nonzero(component)
    tight = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    inventory = _teacher_component_inventory(source, ground_truth)
    expected_component = next(
        (
            item for item in inventory
            if item["component_mask_sha256"] == mask_sha256(component)
            and item["error_kind"] == str(error_kind).lower().replace("false_negative", "fn").replace("false_positive", "fp")
        ),
        None,
    )
    require(expected_component is not None,
            f"{location} component is outside frozen top-four teacher inventory")
    require(
        geometry.get("component_rank") == expected_component["component_rank"]
        and geometry.get("component_id") == expected_component["component_id"]
        and geometry.get("component_box_pixels") == tight,
        f"{location} component rank/identity/tight box differs",
    )
    require(
        geometry["component_seed_pixels"] == list(_distance_transform_seed(component))
        and geometry.get("seed_algorithm")
        == "8_neighbor_repeated_erosion_distance_argmax_row_major",
        f"{location} deterministic interior seed differs",
    )
    x1, y1, x2, y2 = geometry["box_pixels"]
    require(
        0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height,
        f"{location} box is out of frame",
    )
    require(
        x1 <= tight[0] and y1 <= tight[1] and x2 >= tight[2] and y2 >= tight[3],
        f"{location} box does not contain GT error component",
    )
    padding_ratio = geometry.get("padding_ratio")
    require(
        isinstance(padding_ratio, (int, float))
        and float(padding_ratio) in {0.0, 0.1, 0.25, 0.5},
        f"{location} padding ratio is invalid",
    )
    if family == "matched":
        require(float(padding_ratio) == 0.25, f"{location} matched padding differs")
    expected_box, expected_padding, expected_adjusted = _padded_component_box(
        tight,
        component.shape,
        float(padding_ratio),
        production_minimum=family == "matched",
    )
    require(
        geometry["box_pixels"] == expected_box
        and geometry.get("padding_pixels") == expected_padding
        and geometry.get("minimum_extent_adjusted") is expected_adjusted
        and geometry.get("box_algorithm")
        == (
            "component_bbox_ratio_padding_production_min4"
            if family == "matched"
            else "component_bbox_ratio_padding_no_minimum"
        ),
        f"{location} deterministic padded box differs",
    )
    target = geometry.get("authorized_target_view")
    action_name = geometry.get("authorized_action")
    require(target in {"t1", "t2"}, f"{location} authorized target is invalid")
    expected_white = bool(source.masks[str(target)][seed_y, seed_x])
    require(
        geometry.get("component_seed_t1_mask_white") is bool(source.masks["t1"][seed_y, seed_x])
        and geometry.get("component_seed_t2_mask_white") is bool(source.masks["t2"][seed_y, seed_x]),
        f"{location} live seed polarity differs",
    )
    if action_name in {"positive_point", "negative_point"}:
        require(
            action_name == "positive_point" or expected_white,
            f"{location} negative point is unauthorized on black live seed",
        )
    else:
        require(action_name == "box", f"{location} authorized action is invalid")
    if "component_tight_box_xyxy" in geometry:
        require(
            geometry["component_tight_box_xyxy"] == tight,
            f"{location} tight component box differs",
        )
    require(
        geometry.get("gt_used_for") in {
            None,
            "offline_geometry_and_scoring_only",
            "offline_teacher_geometry_only",
            "offline_teacher_geometry_and_scoring",
        },
        f"{location}.gt_used_for is invalid",
    )
    return geometry


def _four_component_sizes(mask: np.ndarray) -> list[int]:
    return [int(component.sum()) for component in _connected_components(mask, four_neighbour=True)]


@dataclass(frozen=True)
class HardGateContract:
    max_selection_area_delta: float = 0.25
    max_locality_outside_ratio: float = 0.1
    max_target_mask_change_ratio: float = 0.25
    max_component_count_delta: int = 4
    max_point_proposal_change_ratio: float = 0.25
    max_point_proposal_change_pixels: int = 256
    max_point_image_change_ratio: float = 0.004
    point_proposal_change_gate_enabled: bool = True


def _hard_gate_contract(config: Mapping[str, Any]) -> HardGateContract:
    value = _mapping(
        config.get("hard_gate_config", config.get("gate_config", {})),
        "hard_gate_config",
    )
    defaults = HardGateContract()
    return HardGateContract(
        max_selection_area_delta=float(value.get("max_selection_area_delta", defaults.max_selection_area_delta)),
        max_locality_outside_ratio=float(value.get("max_locality_outside_ratio", defaults.max_locality_outside_ratio)),
        max_target_mask_change_ratio=float(value.get("max_target_mask_change_ratio", defaults.max_target_mask_change_ratio)),
        max_component_count_delta=int(value.get("max_component_count_delta", defaults.max_component_count_delta)),
        max_point_proposal_change_ratio=float(value.get("max_point_proposal_change_ratio", defaults.max_point_proposal_change_ratio)),
        max_point_proposal_change_pixels=int(value.get("max_point_proposal_change_pixels", defaults.max_point_proposal_change_pixels)),
        max_point_image_change_ratio=float(value.get("max_point_image_change_ratio", defaults.max_point_image_change_ratio)),
        point_proposal_change_gate_enabled=bool(value.get("point_proposal_change_gate_enabled", defaults.point_proposal_change_gate_enabled)),
    )


def recompute_hard_gate(
    source: StateArtifact,
    candidate: StateArtifact,
    action: Mapping[str, Any],
    proposal_value: Any,
    contract: HardGateContract,
    location: str,
) -> dict[str, Any]:
    proposal = _proposal_from_geometry(proposal_value, f"{location}.teacher_geometry")
    target_view = action.get("target_view")
    require(target_view in {"t1", "t2"}, f"{location} target_view is invalid")
    before = source.masks[str(target_view)]
    after = candidate.masks[str(target_view)]
    height, width = before.shape
    box = proposal["box_pixels"]
    box_x1, box_y1, box_x2, box_y2 = box
    require(0 <= box_x1 <= box_x2 < width and 0 <= box_y1 <= box_y2 < height,
            f"{location} proposal box is out of frame")
    action_name = action.get("action", action.get("type"))
    if action_name in {"positive_point", "negative_point"}:
        coordinate = action.get("coordinate")
        require(
            isinstance(coordinate, list)
            and len(coordinate) == 2
            and all(type(item) is int for item in coordinate),
            f"{location} point coordinate is invalid",
        )
        point_x, point_y = coordinate
        require(coordinate == proposal["component_seed_pixels"],
                f"{location} point differs from teacher seed")
        require(box_x1 <= point_x <= box_x2 and box_y1 <= point_y <= box_y2,
                f"{location} point is outside proposal box")
        roi = [
            max(0, min(box_x1, point_x - 4)),
            max(0, min(box_y1, point_y - 4)),
            min(width - 1, max(box_x2, point_x + 4)),
            min(height - 1, max(box_y2, point_y + 4)),
        ]
        composition_mode = (
            "merge_clicked_prediction_component_within_proposal_roi"
            if action_name == "positive_point"
            else "remove_simpleclick_delta_within_clicked_component"
        )
    else:
        require(action_name == "box", f"{location} action is invalid")
        require(action.get("box") == box, f"{location} action box differs from teacher box")
        roi = list(box)
        composition_mode = "replace_box_roi_only"

    changed = np.logical_xor(before, after)
    inside = np.zeros_like(changed)
    x1, y1, x2, y2 = roi
    inside[y1 : y2 + 1, x1 : x2 + 1] = True
    changed_pixels = int(changed.sum())
    outside_pixels = int(np.logical_and(changed, ~inside).sum())
    before_components = _four_component_sizes(before)
    after_components = _four_component_sizes(after)
    changed_components = _four_component_sizes(changed)
    locality = {
        "composition_mode": composition_mode,
        "roi_xyxy": roi,
        "changed_pixels": changed_pixels,
        "inside_roi_pixels": changed_pixels - outside_pixels,
        "outside_roi_pixels": outside_pixels,
        "local_changed_pixels": changed_pixels - outside_pixels,
        "nonlocal_changed_pixels": outside_pixels,
        "outside_roi_ratio": outside_pixels / changed_pixels if changed_pixels else 0.0,
        "target_mask_change_ratio": changed_pixels / changed.size,
        "largest_changed_component_pixels": max(changed_components, default=0),
        "components_before": len(before_components),
        "components_after": len(after_components),
        "component_count_delta": len(after_components) - len(before_components),
    }
    previous_area_ratio = float(source.masks["change"].mean())
    candidate_area_ratio = float(candidate.masks["change"].mean())
    candidate_area_delta = abs(candidate_area_ratio - previous_area_ratio)
    reasons: list[str] = []
    if candidate_area_delta > contract.max_selection_area_delta:
        reasons.append("mask_area_delta_exceeded")
    if locality["outside_roi_ratio"] > contract.max_locality_outside_ratio:
        reasons.append("locality_outside_roi_exceeded")
    if locality["target_mask_change_ratio"] > contract.max_target_mask_change_ratio:
        reasons.append("target_mask_change_exceeded")
    if abs(locality["component_count_delta"]) > contract.max_component_count_delta:
        reasons.append("component_count_delta_exceeded")

    point_authorization: dict[str, Any] | None = None
    if action_name in {"positive_point", "negative_point"}:
        component_area = proposal["component_area"]
        proposal_ratio = changed_pixels / component_area
        ratio_exceeded = proposal_ratio > contract.max_point_proposal_change_ratio
        pixels_exceeded = changed_pixels > contract.max_point_proposal_change_pixels
        image_exceeded = locality["target_mask_change_ratio"] > contract.max_point_image_change_ratio
        hard_reject = bool(
            contract.point_proposal_change_gate_enabled
            and ratio_exceeded
            and pixels_exceeded
            and image_exceeded
        )
        change_gate = {
            "enabled": contract.point_proposal_change_gate_enabled,
            "proposal_change_ratio": proposal_ratio,
            "max_proposal_change_ratio": contract.max_point_proposal_change_ratio,
            "changed_pixels": changed_pixels,
            "max_changed_pixels": contract.max_point_proposal_change_pixels,
            "image_change_ratio": locality["target_mask_change_ratio"],
            "max_image_change_ratio": contract.max_point_image_change_ratio,
            "proposal_ratio_exceeded": ratio_exceeded,
            "absolute_pixels_exceeded": pixels_exceeded,
            "image_ratio_exceeded": image_exceeded,
            "hard_reject": hard_reject,
            "policy": "reject_only_when_ratio_and_absolute_and_image_limits_exceed",
        }
        if hard_reject:
            reasons.append("point_proposal_combined_change_gate_exceeded")
        point_authorization = {
            "proposal_box_xyxy": box,
            "proposal_component_area": component_area,
            "proposal_change_ratio": proposal_ratio,
            "effective_roi_xyxy": roi,
            "change_gate": change_gate,
        }
    return {
        "passed": not reasons,
        "rejected": bool(reasons),
        "rejection_reasons": reasons,
        "verifier_skipped_by_hard_gate": bool(reasons),
        "previous_area_ratio": previous_area_ratio,
        "candidate_area_ratio": candidate_area_ratio,
        "candidate_area_delta": candidate_area_delta,
        "locality": locality,
        "point_authorization": point_authorization,
    }


def _require_hard_gate_receipt(
    candidate: Mapping[str, Any],
    expected: Mapping[str, Any],
    location: str,
) -> bool:
    gate = _mapping(candidate.get("hard_gate"), f"{location}.hard_gate")
    expected_gate = {
        key: expected[key]
        for key in ("passed", "rejected", "rejection_reasons", "verifier_skipped_by_hard_gate")
    }
    require(dict(gate) == expected_gate,
            f"{location}.hard_gate differs from independent recomputation")
    execution = _mapping(
        candidate.get("runtime_execution", candidate.get("execution")),
        f"{location}.runtime_execution",
    )
    require(
        execution.get("candidate_accepted") is expected["passed"]
        and execution.get("verifier_skipped_by_hard_gate") is expected["rejected"]
        and execution.get("candidate_rejection_reasons") == expected["rejection_reasons"],
        f"{location} runtime hard-gate decision differs",
    )
    _require_numeric_mapping(
        execution,
        {key: expected[key] for key in (
            "previous_area_ratio", "candidate_area_ratio", "candidate_area_delta"
        )},
        f"{location}.runtime_execution",
    )
    _require_numeric_mapping(
        execution.get("locality"), expected["locality"],
        f"{location}.runtime_execution.locality",
    )
    if expected["point_authorization"] is not None:
        point = _mapping(
            execution.get("point_authorization"),
            f"{location}.runtime_execution.point_authorization",
        )
        expected_point = expected["point_authorization"]
        _require_numeric_mapping(
            point,
            {key: expected_point[key] for key in (
                "proposal_box_xyxy", "proposal_component_area",
                "proposal_change_ratio", "effective_roi_xyxy",
            )},
            f"{location}.runtime_execution.point_authorization",
        )
        _require_numeric_mapping(
            point.get("change_gate"), expected_point["change_gate"],
            f"{location}.runtime_execution.point_authorization.change_gate",
        )
    return bool(expected["passed"])


@dataclass(frozen=True)
class ValidationContext:
    repository: Path
    config_path: Path
    config_sha256: str
    config_canonical_sha256: str
    base_config: dict[str, Any]
    base_config_sha256: str
    baseline_report: dict[str, Any]
    baseline_report_sha256: str
    endpoint: EndpointContract
    hard_gate: HardGateContract
    ordered_samples: tuple[str, ...]
    samples: dict[str, dict[str, Any]]
    expected_initial_hashes: dict[str, dict[str, str]]


def _repo_path(repository: Path, value: Any, location: str) -> Path:
    require(isinstance(value, str) and value, f"{location} path is invalid")
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (repository / raw).resolve()
    try:
        path.relative_to(repository.resolve())
    except ValueError as error:
        raise ValidationError(f"{location} escapes repository") from error
    return path


def _sample_map(base_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    value = base_config.get("samples")
    require(isinstance(value, Mapping), "base config samples must be an object")
    return {str(key): dict(_mapping(item, f"base sample {key}")) for key, item in value.items()}


def validate_config(config: Mapping[str, Any], config_path: Path) -> ValidationContext:
    config_path = config_path.resolve()
    repository = config_path.parent.parent.resolve()
    require(config_path.parent == repository / "configs", "config must be inside repository/configs")
    require(config.get("schema_id") == CONFIG_SCHEMA_ID, "teacher config schema is invalid")
    require(config.get("status") == "frozen_preregistration",
            "teacher config status is not frozen")
    require(config.get("result_fields_intentionally_absent") is True,
            "teacher preregistration contains or permits result fields")

    scope = _mapping(config.get("scope_and_access_audit"), "scope_and_access_audit")
    expected_scope = {
        "teacher_only": True,
        "change_only_teacher_upper_bound": True,
        "raw_label_pixels_passed_to_environment": False,
        "raw_label_pixels_passed_to_tools": False,
        "raw_label_path_passed_to_runtime": False,
        "gt_derived_geometry_passed_to_environment": True,
        "gt_derived_geometry_passed_to_tools": True,
        "gt_derived_metadata_passed_to_environment": True,
        "executor_receipt_contains_teacher_encoded_region_id": True,
        "qwen_invoked": False,
        "training_authorized": False,
        "test_split_used": False,
        "generalization_claim_allowed": False,
    }
    require(all(scope.get(key) is value for key, value in expected_scope.items()),
            "teacher access scope differs from frozen teacher-only contract")
    require(
        scope.get("runtime_boundary_layers") == RUNTIME_BOUNDARY_LAYERS,
        "teacher runtime boundary layers differ from frozen contract",
    )

    base_spec = _mapping(config.get("base_config_0803"), "base_config_0803")
    base_path = _repo_path(repository, base_spec.get("path"), "base_config_0803")
    require(base_path.is_file(), f"0803 base config is missing: {base_path}")
    base_sha = sha256_file(base_path)
    require(
        _is_sha256(base_spec.get("sha256"))
        and base_sha == str(base_spec["sha256"]).lower(),
        "0803 base config SHA256 differs",
    )
    base_config = read_json(base_path, "0803 base config")
    require(base_config.get("schema_id") == base_spec.get("schema_id"),
            "0803 base config schema differs")

    baseline_spec = _mapping(config.get("production_baseline_0803"),
                             "production_baseline_0803")
    baseline_path = _repo_path(
        repository,
        baseline_spec.get("validation_report_path"),
        "production_baseline_0803.validation_report",
    )
    require(baseline_path.is_file(), f"0803 validation report is missing: {baseline_path}")
    baseline_sha = sha256_file(baseline_path)
    require(
        _is_sha256(baseline_spec.get("validation_report_sha256"))
        and baseline_sha == str(baseline_spec["validation_report_sha256"]).lower(),
        "0803 validation report SHA256 differs",
    )
    baseline = read_json(baseline_path, "0803 validation report")
    require(
        baseline.get("schema_id") == "qwen3vl_oracle_trajectory_validation_report_v1"
        and baseline.get("status") == "validated"
        and baseline.get("scientific_conclusion") == baseline_spec.get("scientific_status")
        and baseline.get("endpoint_success_episode_count")
        == baseline_spec.get("nontrivial_endpoint_successes")
        and baseline.get("evaluated_nonexact_episode_count")
        == baseline_spec.get("nontrivial_denominator"),
        "0803 baseline report summary differs from preregistration",
    )

    ordered_raw = _list(config.get("ordered_samples"), "ordered_samples")
    require(
        len(ordered_raw) == 4
        and len(set(ordered_raw)) == 4
        and all(isinstance(item, str) and item for item in ordered_raw),
        "ordered_samples must contain four unique sample IDs",
    )
    ordered = tuple(str(item) for item in ordered_raw)
    base_samples = _sample_map(base_config)
    require(all(sample_id in base_samples for sample_id in ordered),
            "teacher sample is absent from 0803 base config")
    samples = {sample_id: base_samples[sample_id] for sample_id in ordered}
    require(
        all(samples[sample_id].get("split") in {"train", "val"} for sample_id in ordered),
        "teacher sample split is invalid",
    )
    train_parents = {
        str(samples[item].get("parent_id")) for item in ordered
        if samples[item].get("split") == "train"
    }
    val_parents = {
        str(samples[item].get("parent_id")) for item in ordered
        if samples[item].get("split") == "val"
    }
    require(not train_parents.intersection(val_parents), "teacher parents cross train/val")

    expected_hashes_raw = _mapping(
        config.get("expected_initial_state_mask_sha256"),
        "expected_initial_state_mask_sha256",
    )
    require(list(expected_hashes_raw) == list(ordered),
            "expected initial hash sample order differs")
    expected_hashes: dict[str, dict[str, str]] = {}
    for sample_id in ordered:
        hashes = dict(_mapping(expected_hashes_raw[sample_id],
                               f"expected initial hashes {sample_id}"))
        require(
            set(hashes) == {"t1", "t2", "change"}
            and all(_is_sha256(value) for value in hashes.values()),
            f"expected initial hashes {sample_id} are invalid",
        )
        expected_hashes[sample_id] = {key: str(value).lower() for key, value in hashes.items()}

    evidence = _mapping(config.get("production_episode_evidence"),
                        "production_episode_evidence")
    require(list(evidence) == list(ordered), "production episode evidence order differs")
    baseline_episodes = {
        str(item.get("sample_id")): item
        for item in _list(baseline.get("episodes"), "0803 baseline episodes")
    }
    for sample_id in ordered:
        spec = _mapping(evidence[sample_id], f"production episode {sample_id}")
        path = _repo_path(repository, spec.get("path"), f"production episode {sample_id}")
        require(path.is_file(), f"production episode {sample_id} is missing")
        require(
            _is_sha256(spec.get("sha256"))
            and sha256_file(path) == str(spec["sha256"]).lower(),
            f"production episode {sample_id} SHA256 differs",
        )
        episode = read_json(path, f"production episode {sample_id}")
        require(
            episode.get("sample_id") == sample_id
            and _mapping(episode.get("termination"), f"production episode {sample_id}.termination").get("reason")
            == spec.get("termination"),
            f"production episode {sample_id} identity/termination differs",
        )
        initial = _mapping(episode.get("initial_state"), f"production episode {sample_id}.initial_state")
        recorded_hashes = initial.get("mask_sha256", initial.get("state_mask_sha256"))
        require(
            dict(_mapping(recorded_hashes, f"production episode {sample_id}.initial hashes"))
            == expected_hashes[sample_id],
            f"production episode {sample_id} initial hashes differ",
        )
        baseline_item = baseline_episodes.get(sample_id)
        require(
            baseline_item is not None
            and baseline_item.get("termination_reason") == spec.get("termination"),
            f"production episode {sample_id} is not bound to baseline report",
        )

    protocol = _mapping(config.get("execution_protocol"), "execution_protocol")
    require(
        protocol.get("max_steps") == 3
        and protocol.get("one_action_per_step") is True
        and protocol.get("fresh_sam3_initializer") is True
        and protocol.get("state_continuity_required") is True
        and protocol.get("selected_action_canonical_replay_required") is True
        and protocol.get("authorized_teacher_region_id_required") is True
        and protocol.get("qwen_loads_and_calls_allowed") == 0,
        "execution protocol differs from frozen three-step teacher contract",
    )
    composition = dict(
        _mapping(
            protocol.get("change_composition"),
            "execution_protocol.change_composition",
        )
    )
    expected_composition = {
        "implementation": "MaskPairProcessor",
        "mode": "overlap_presence",
        "overlap_threshold": 0.25,
        "t12_min_instance_area": 0,
        "cd_min_instance_area": 0,
    }
    require(
        composition == expected_composition,
        "change composition differs from frozen production MaskPairProcessor contract",
    )
    base_matching = dict(
        _mapping(base_config.get("matching"), "base_config_0803.matching")
    )
    require(
        {key: composition[key] for key in base_matching} == base_matching,
        "change composition differs from bound 0803 matching contract",
    )
    geometry = _mapping(config.get("teacher_geometry"), "teacher_geometry")
    require(
        geometry.get("component_connectivity") == 8
        and _mapping(geometry.get("matched"), "teacher_geometry.matched").get("padding_ratio") == 0.25
        and _mapping(geometry.get("matched"), "teacher_geometry.matched").get("max_candidates_per_step") == 4
        and _mapping(geometry.get("wide"), "teacher_geometry.wide").get("padding_ratios") == [0.0, 0.1, 0.25, 0.5]
        and _mapping(geometry.get("wide"), "teacher_geometry.wide").get("max_candidates_per_step") == 96,
        "teacher geometry contract differs",
    )
    tracks = _list(config.get("tracks"), "tracks")
    require([item.get("track_id") for item in tracks if isinstance(item, Mapping)] == list(TRACK_NAMES),
            "teacher track inventory/order differs")
    expected_tracks = {
        "matched_safe": ("matched", True, True, "always"),
        "wide_safe": ("wide", True, True, "matched_safe_endpoint_failed"),
        "wide_hard": ("wide", True, False, "wide_safe_endpoint_failed"),
        "wide_raw": ("wide", False, False, "wide_hard_endpoint_failed"),
    }
    for raw_track in tracks:
        track = _mapping(raw_track, "track")
        track_id = str(track.get("track_id"))
        require(
            (
                track.get("geometry"),
                track.get("production_hard_gate_required"),
                track.get("frozen_rgb_semantic_gate_required"),
                track.get("run_condition"),
            ) == expected_tracks[track_id],
            f"track {track_id} policy differs",
        )

    resource = _mapping(config.get("resource_plan"), "resource_plan")
    diagnostic_smoke = dict(
        _mapping(resource.get("diagnostic_smoke"), "resource_plan.diagnostic_smoke")
    )
    require(
        diagnostic_smoke
        == {
            "required_before_formal_pilot": True,
            "sample_id": "train_407_r3c2",
            "shard_index": 0,
            "num_shards": 4,
            "gpus": 1,
            "time_limit": "12:00:00",
            "output_scope": "diagnostics_only",
            "scientific_result_authoritative": False,
            "geometry_modes_executed": ["matched"],
            "step_indices_executed": [1],
            "max_real_candidates": 2,
            "required_tool_action_pairs": [
                ["sam3", "box"],
                ["simpleclick", "positive_point"],
            ],
            "endpoint_search_performed": False,
        },
        "diagnostic smoke resource contract differs",
    )
    shards = _list(resource.get("shards"), "resource_plan.shards")
    require(
        resource.get("slurm_required") is True
        and resource.get("maximum_total_gpus") == 4
        and resource.get("planned_generation_shards") == EXPECTED_SHARD_COUNT
        and resource.get("gpus_per_shard") == 1
        and resource.get("peak_gpu_concurrency") == 4
        and resource.get("independent_validation_gpus") == 0
        and len(shards) == EXPECTED_SHARD_COUNT,
        "four-shard resource plan differs",
    )
    shard_coverage: list[str] = []
    for index, raw_shard in enumerate(shards):
        shard = _mapping(raw_shard, f"resource_plan.shards[{index}]")
        require(shard.get("shard_id") == index, "resource shard identity differs")
        assigned = _list(shard.get("samples"), f"resource shard {index}.samples")
        require(len(assigned) == 1, f"resource shard {index} must contain one sample")
        shard_coverage.extend(str(item) for item in assigned)
    require(shard_coverage == list(ordered), "resource shard sample coverage/order differs")

    endpoint_value = _mapping(config.get("endpoint"), "endpoint")
    require(endpoint_value.get("stop_at_first_endpoint") is True,
            "endpoint must stop at first success")
    endpoint = EndpointContract(
        float(endpoint_value.get("nonempty_gt_minimum_iou", DEFAULT_ENDPOINT_IOU)),
        float(endpoint_value.get("nonempty_gt_minimum_f1", DEFAULT_ENDPOINT_F1)),
        int(endpoint_value.get("empty_gt_maximum_fp", DEFAULT_EMPTY_GT_MAX_FP)),
        DEFAULT_EPSILON,
    )
    base_hard_gate = _hard_gate_contract(base_config)
    return ValidationContext(
        repository=repository,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        config_canonical_sha256=canonical_sha256(config),
        base_config=base_config,
        base_config_sha256=base_sha,
        baseline_report=baseline,
        baseline_report_sha256=baseline_sha,
        endpoint=endpoint,
        hard_gate=base_hard_gate,
        ordered_samples=ordered,
        samples=samples,
        expected_initial_hashes=expected_hashes,
    )


def run_preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    context = validate_config(config, config_path)
    runner = context.repository / "tools" / "run_qwen3vl_gt_direct_teacher.py"
    validator = Path(__file__).resolve()
    require(runner.is_file(), f"teacher runner is missing: {runner}")
    return {
        "schema_id": PREFLIGHT_SCHEMA_ID,
        "status": "preflight_passed",
        "teacher_only": True,
        "training_authorized": False,
        "qwen_invoked": False,
        "config": {
            "path": str(context.config_path),
            "sha256": context.config_sha256,
            "canonical_sha256": context.config_canonical_sha256,
        },
        "base_config_0803": {
            "sha256": context.base_config_sha256,
            "canonical_sha256": canonical_sha256(context.base_config),
        },
        "production_baseline_0803": {
            "sha256": context.baseline_report_sha256,
            "canonical_sha256": canonical_sha256(context.baseline_report),
            "endpoint_successes": 0,
            "denominator": 4,
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "validator": {"path": str(validator), "sha256": sha256_file(validator)},
        "ordered_samples": list(context.ordered_samples),
        "shard_count": EXPECTED_SHARD_COUNT,
        "tracks": list(TRACK_NAMES),
    }


def _action_name(action: Mapping[str, Any]) -> str:
    value = action.get("action", action.get("type"))
    require(value in {"positive_point", "negative_point", "box"},
            "candidate action type is invalid")
    return str(value)


def _validate_action(
    action_value: Any, geometry: Mapping[str, Any], location: str
) -> dict[str, Any]:
    action = dict(_mapping(action_value, f"{location}.action"))
    name = _action_name(action)
    require(action.get("target_view") in {"t1", "t2"},
            f"{location}.action target view is invalid")
    require(action.get("coordinate_space", "pixel") == "pixel",
            f"{location}.action coordinate space differs")
    require(
        action.get("target_view") == geometry.get("authorized_target_view")
        and name == geometry.get("authorized_action"),
        f"{location}.action differs from teacher authorization",
    )
    if name == "box":
        require(action.get("box") == geometry["box_pixels"],
                f"{location}.action box differs from teacher geometry")
        require("coordinate" not in action, f"{location}.action box has a point")
    else:
        require(action.get("coordinate") == geometry["component_seed_pixels"],
                f"{location}.action point differs from teacher geometry")
        require("box" not in action, f"{location}.action point has a box")
    return action


def _forbid_raw_gt(value: Any, location: str) -> None:
    """Forbid raw labels/metrics in tool-facing evidence, while allowing geometry."""

    forbidden = {
        "gt",
        "gt_mask",
        "ground_truth",
        "label",
        "labels",
        "error_mask",
        "metrics",
        "metric_delta",
        "pixel_error",
        "iou",
        "f1",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            require(
                normalized not in forbidden
                and not normalized.startswith("gt_mask")
                and "ground_truth" not in normalized,
                f"{location} contains forbidden raw GT/metric field {key!r}",
            )
            _forbid_raw_gt(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_raw_gt(child, f"{location}[{index}]")


def _validate_tool_receipt(
    candidate: Mapping[str, Any],
    action: Mapping[str, Any],
    geometry: Mapping[str, Any],
    location: str,
) -> Mapping[str, Any]:
    receipt = _mapping(candidate.get("tool_receipt"), f"{location}.tool_receipt")
    execution = _mapping(
        candidate.get("runtime_execution", candidate.get("execution")),
        f"{location}.runtime_execution",
    )
    name = _action_name(action)
    point = name in {"positive_point", "negative_point"}
    expected_tool = "simpleclick" if point else "sam3"
    expected_mode = "point" if point else "box"
    require(receipt.get("tool") == expected_tool, f"{location} tool differs")
    tool_input = _mapping(receipt.get("tool_input"), f"{location}.tool_input")
    region_id = geometry.get("region_id", geometry.get("teacher_region_id"))
    require(isinstance(region_id, str) and region_id,
            f"{location} teacher region ID is invalid")
    require(tool_input.get("authorized_region_id") == region_id,
            f"{location} tool authorization region differs")
    _forbid_raw_gt(tool_input, f"{location}.tool_input")
    if point:
        require(
            tool_input.get("coordinate") == action.get("coordinate")
            and tool_input.get("is_positive") is (name == "positive_point"),
            f"{location} point tool input differs from action",
        )
        authorization = _mapping(
            receipt.get("point_authorization"),
            f"{location}.point_authorization",
        )
        require(
            authorization.get("region_id") == region_id
            and authorization.get("proposal_box_xyxy") == geometry["box_pixels"],
            f"{location} point authorization differs",
        )
    else:
        authorization = _mapping(
            receipt.get("box_authorization"),
            f"{location}.box_authorization",
        )
        require(
            authorization.get("region_id") == region_id
            and authorization.get("proposal_box_xyxy") == action.get("box")
            and authorization.get("geometry_match") is True,
            f"{location} box authorization differs",
        )
    result = _mapping(receipt.get("tool_result"), f"{location}.tool_result")
    report = _mapping(result.get("worker_report"), f"{location}.worker_report")
    require(
        result.get("worker_mode") == expected_mode
        and report.get("status") == "success"
        and report.get("mode") == expected_mode,
        f"{location} real tool receipt is not successful",
    )
    _forbid_raw_gt(result, f"{location}.tool_result")
    subset = {
        key: execution[key]
        for key in (
            "tool", "tool_input", "tool_result", "locality",
            "point_authorization", "box_authorization",
        )
        if key in execution
    }
    require(dict(receipt) == subset, f"{location} tool receipt/execution differ")
    return execution


def _expected_rgb_state(sample: Mapping[str, Any], target_view: str) -> str:
    semantic = _mapping(sample.get("semantic"), "sample.semantic")
    value = semantic.get(f"{target_view}_state", semantic.get(target_view))
    require(value in {"building", "background", "mixed", "uncertain"},
            "sample RGB semantic state is invalid")
    return str(value)


def _validate_rgb_gate(
    gate_value: Any,
    action: Mapping[str, Any],
    sample: Mapping[str, Any],
    location: str,
) -> bool:
    gate = _mapping(gate_value, f"{location}.rgb_semantic_gate")
    target_view = str(action.get("target_view"))
    action_name = _action_name(action)
    observed = _expected_rgb_state(sample, target_view)
    required = "background" if action_name == "negative_point" else "building"
    expected = {
        "passed": observed == required,
        "target_view": target_view,
        "action": action_name,
        "observed_rgb_state": observed,
        "required_rgb_state": required,
        "policy": "negative_requires_background_positive_or_box_requires_building",
    }
    require(dict(gate) == expected, f"{location} RGB gate differs from frozen semantics")
    return bool(expected["passed"])


def _strict_beneficial(
    before: Mapping[str, Any], after: Mapping[str, Any], epsilon: float
) -> bool:
    reduction = int(before["pixel_error"]) - int(after["pixel_error"])
    if int(before["gt_pixels"]) > 0:
        return bool(
            reduction > 0
            and float(after["iou"]) - float(before["iou"]) > epsilon
            and float(after["f1"]) - float(before["f1"]) > epsilon
        )
    return bool(reduction > 0 and int(after["fp"]) < int(before["fp"]))


def _track_policy(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["track_id"]): dict(item)
        for item in _list(config.get("tracks"), "tracks")
        if isinstance(item, Mapping)
    }


def expected_gate_policy(track_id: str) -> dict[str, Any]:
    require(track_id in TRACK_NAMES, f"unknown track: {track_id}")
    return {
        "require_hard_gate": track_id != "wide_raw",
        "require_rgb_semantic_gate": track_id in {"matched_safe", "wide_safe"},
        "require_strict_metric_improvement": True,
        "geometry_mode": "matched" if track_id == "matched_safe" else "wide",
        "counterfactual_gate_bypass_allowed": track_id == "wide_raw",
        "policy_id": {
            "matched_safe": "matched_geometry_hard_rgb_metric",
            "wide_safe": "wide_geometry_hard_rgb_metric",
            "wide_hard": "wide_geometry_hard_metric_ignore_rgb",
            "wide_raw": "wide_geometry_metric_only_ignore_hard_and_rgb",
        }[track_id],
    }


@dataclass
class CandidateEvidence:
    candidate_id: str
    geometry_family: str
    state: StateArtifact
    action: dict[str, Any]
    hard_passed: bool
    rgb_passed: bool
    beneficial: bool
    eligible: bool
    metrics: dict[str, Any]
    delta: dict[str, Any]


def _validate_candidate(
    candidate_value: Any,
    *,
    root: Path,
    source: StateArtifact,
    ground_truth: np.ndarray,
    source_metrics: Mapping[str, Any],
    sample: Mapping[str, Any],
    track: Mapping[str, Any],
    context: ValidationContext,
    location: str,
) -> CandidateEvidence:
    candidate = _mapping(candidate_value, location)
    candidate_id = candidate.get("candidate_id")
    require(isinstance(candidate_id, str) and candidate_id,
            f"{location}.candidate_id is invalid")
    geometry = _validate_teacher_geometry(
        candidate.get("teacher_geometry"),
        source=source,
        ground_truth=ground_truth,
        location=f"{location}.teacher_geometry",
    )
    require(
        candidate.get("region_id") == geometry.get("region_id")
        and str(geometry.get("region_id")) in str(candidate_id),
        f"{location} candidate/teacher region authorization differs",
    )
    require(
        geometry.get(
            "geometry_family",
            geometry.get("family", geometry.get("geometry_mode")),
        ) == track.get("geometry"),
        f"{location} geometry does not belong to track",
    )
    action = _validate_action(candidate.get("action"), geometry, location)
    candidate_state = _load_state(
        root, candidate.get("candidate_state"), f"{location}.candidate_state"
    )
    _require_rebuilt_change(candidate_state, context, f"{location}.candidate_state")
    metrics = compute_metrics(candidate_state.masks["change"], ground_truth)
    _require_numeric_mapping(candidate.get("raw_metrics"), metrics,
                             f"{location}.raw_metrics")
    delta = _metric_delta(source_metrics, metrics)
    _require_numeric_mapping(candidate.get("metric_delta"), delta,
                             f"{location}.metric_delta", exact_keys=True)
    beneficial = _strict_beneficial(source_metrics, metrics, context.endpoint.epsilon)
    raw_metric_gate = candidate.get("raw_metric_beneficial")
    if isinstance(raw_metric_gate, Mapping):
        require(raw_metric_gate.get("passed") is beneficial,
                f"{location}.raw_metric_beneficial.passed differs")
        expected_gate = {
            "passed": beneficial,
            "pixel_error_strictly_decreased": int(delta["pixel_error_reduction"]) > 0,
            "nonempty_gt": int(source_metrics["gt_pixels"]) > 0,
            "iou_strictly_increased": (
                float(delta["iou"]) > context.endpoint.epsilon
                if int(source_metrics["gt_pixels"]) > 0 else None
            ),
            "f1_strictly_increased": (
                float(delta["f1"]) > context.endpoint.epsilon
                if int(source_metrics["gt_pixels"]) > 0 else None
            ),
            "fp_strictly_decreased": (
                int(metrics["fp"]) < int(source_metrics["fp"])
                if int(source_metrics["gt_pixels"]) == 0 else None
            ),
            "comparison_epsilon": context.endpoint.epsilon,
        }
        require(dict(raw_metric_gate) == expected_gate,
                f"{location}.raw_metric_beneficial gate differs")
    else:
        require(raw_metric_gate is beneficial,
                f"{location}.raw_metric_beneficial differs")
    if "endpoint_gate" in candidate:
        require(candidate.get("endpoint_gate") == endpoint_gate(metrics, context.endpoint),
                f"{location}.endpoint_gate differs")
    execution = _validate_tool_receipt(candidate, action, geometry, location)
    independent_hard = recompute_hard_gate(
        source, candidate_state, action, geometry, context.hard_gate, location
    )
    hard_passed = _require_hard_gate_receipt(candidate, independent_hard, location)
    rgb_passed = _validate_rgb_gate(
        candidate.get("rgb_semantic_gate"), action, sample, location
    )
    require(
        execution.get("teacher_only") in {None, True}
        and execution.get("training_authorized") in {None, False},
        f"{location} runtime evidence authorizes teacher data",
    )
    eligible = bool(
        beneficial
        and (hard_passed or track.get("production_hard_gate_required") is False)
        and (rgb_passed or track.get("frozen_rgb_semantic_gate_required") is False)
    )
    require(candidate.get("eligible_under_track") is eligible,
            f"{location}.eligible_under_track differs")
    return CandidateEvidence(
        str(candidate_id), str(track.get("geometry")), candidate_state, action,
        hard_passed, rgb_passed, beneficial, eligible, metrics, delta,
    )


def _validate_catalog(
    value: Any,
    *,
    source: StateArtifact,
    ground_truth: np.ndarray,
    track_id: str,
    step_index: int,
    sample_id: str,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    location: str,
) -> None:
    catalog = _mapping(value, location)
    geometry_mode = "matched" if track_id == "matched_safe" else "wide"
    require(
        catalog.get("schema_id") == "qwen3vl_gt_direct_teacher_catalog_v1"
        and catalog.get("geometry_mode") == geometry_mode
        and catalog.get("step_index") == step_index
        and catalog.get("source_state_mask_sha256") == source.mask_hashes
        and catalog.get("source_state_sha256") == source.identity
        and catalog.get("gt_derived_geometry") is True
        and catalog.get("raw_gt_mask_in_catalog") is False
        and catalog.get("qwen_invoked") is False,
        f"{location} identity/access audit differs",
    )
    fn_mask = _error_mask(source, ground_truth, "fn")
    fp_mask = _error_mask(source, ground_truth, "fp")
    fn_components = _connected_components(fn_mask)
    fp_components = _connected_components(fp_mask)
    all_count = len(fn_components) + len(fp_components)
    expected_inventory = {
        "fn_pixels": int(fn_mask.sum()),
        "fn_component_count": len(fn_components),
        "fp_pixels": int(fp_mask.sum()),
        "fp_component_count": len(fp_components),
        "selected_component_count": min(4, all_count),
        "all_component_count": all_count,
        "unselected_component_count": max(0, all_count - 4),
    }
    require(dict(_mapping(catalog.get("error_inventory"), f"{location}.error_inventory")) == expected_inventory,
            f"{location}.error_inventory differs")
    expected_components = _teacher_component_inventory(source, ground_truth)
    recorded_components = _list(catalog.get("selected_components"),
                                f"{location}.selected_components")
    require(len(recorded_components) == len(expected_components),
            f"{location}.selected_components count differs")
    for index, (recorded_value, expected) in enumerate(zip(recorded_components, expected_components)):
        recorded = _mapping(recorded_value, f"{location}.selected_components[{index}]")
        require(
            recorded.get("error_kind") == expected["error_kind"]
            and recorded.get("component_area") == expected["component_area"]
            and recorded.get("component_box") == expected["component_box_pixels"]
            and recorded.get("component_mask_sha256") == expected["component_mask_sha256"]
            and recorded.get("component_rank") == expected["component_rank"]
            and recorded.get("component_id") == expected["component_id"],
            f"{location}.selected_components[{index}] differs",
        )
    paddings = [0.25] if geometry_mode == "matched" else [0.0, 0.1, 0.25, 0.5]
    proposal_count = len(expected_components) * len(paddings)
    routes_per_padding = 0
    candidate_groups: list[list[tuple[str, str, str, int]]] = []
    for component in expected_components:
        seed_x, seed_y = _distance_transform_seed(component["component"])
        # Production surface: positive + box on every view; white seeds also
        # authorize a negative point on that same view.
        routes_per_padding += 4 + sum(
            bool(source.masks[view][seed_y, seed_x]) for view in ("t1", "t2")
        )
        group: list[tuple[str, str, str, int]] = []
        for padding_index, padding_ratio in enumerate(paddings):
            base_region = (
                f"teacher-{component['component_id']}-"
                f"p{round(padding_ratio * 1000):04d}"
            )
            route_index = 0
            for view in ("t1", "t2"):
                actions = (
                    ["negative_point", "positive_point", "box"]
                    if bool(source.masks[view][seed_y, seed_x])
                    else ["positive_point", "box"]
                )
                for action_name in actions:
                    region_id = f"{base_region}--{action_name}--{view}"
                    candidate_id = (
                        f"{sample_id}--{geometry_mode}--step-{step_index:03d}--"
                        f"{region_id}"
                    )
                    group.append((candidate_id, action_name, view, padding_index * 10 + route_index))
                    route_index += 1
        group.sort(key=lambda item: (item[3] // 10, item[3] % 10, item[0]))
        candidate_groups.append(group)
    action_count = routes_per_padding * len(paddings)
    if geometry_mode == "matched":
        flattened = [
            (candidate_id, action_name, view, int(component["component_area"]))
            for component, group in zip(expected_components, candidate_groups)
            for candidate_id, action_name, view, _ in group
        ]
        flattened.sort(key=lambda item: (-item[3], item[0]))
        expected_ids = [
            candidate_id
            for candidate_id, action_name, view, _ in flattened
            if _expected_rgb_state(sample, view)
            == ("background" if action_name == "negative_point" else "building")
        ][:4]
    else:
        expected_ids = []
        max_group = max((len(group) for group in candidate_groups), default=0)
        for index in range(max_group):
            for group in candidate_groups:
                if index < len(group):
                    expected_ids.append(group[index][0])
        expected_ids = expected_ids[:96]
    require(
        catalog.get("padding_ratios") == paddings
        and catalog.get("proposal_geometry_count") == proposal_count
        and catalog.get("action_count_before_cap") == action_count
        and catalog.get("execution_candidate_count") == len(candidates)
        and catalog.get("candidate_ids")
        == expected_ids
        and [str(_mapping(item, "candidate").get("candidate_id")) for item in candidates]
        == expected_ids,
        f"{location} candidate inventory differs",
    )
    cap = 4 if geometry_mode == "matched" else 96
    require(len(candidates) <= cap, f"{location} exceeds candidate cap")


@dataclass
class ValidatedTrack:
    track_id: str
    executed: bool
    endpoint_success: bool
    selected_steps: int
    candidate_count: int
    hard_pass_count: int
    rgb_pass_count: int
    beneficial_count: int
    eligible_count: int
    hard_rejected_beneficial_count: int
    initial_metrics: dict[str, Any] | None
    final_metrics: dict[str, Any] | None
    termination_reason: str | None


@dataclass
class ValidatedEpisode:
    sample_id: str
    parent_id: str
    split: str
    source_episode_sha256: str
    winning_track: str | None
    episode_status: str
    conclusion: str
    tracks: dict[str, ValidatedTrack]
    initial_metrics: dict[str, Any]


def _validate_replay(
    replay_value: Any,
    *,
    root: Path,
    source: StateArtifact,
    selected: CandidateEvidence,
    selected_record: Mapping[str, Any],
    ground_truth: np.ndarray,
    sample: Mapping[str, Any],
    track_id: str,
    canonical_available: bool,
    context: ValidationContext,
    location: str,
) -> StateArtifact:
    replay = _mapping(replay_value, location)
    replay_state = _load_state(root, replay.get("state"), f"{location}.state")
    _require_rebuilt_change(replay_state, context, f"{location}.state")
    _same_state(selected.state, replay_state, f"{location} candidate/replay")
    require(
        replay.get("candidate_branch_state_mask_sha256") == selected.state.mask_hashes
        and replay.get("canonical_replay_state_mask_sha256") == replay_state.mask_hashes
        and replay.get("all_three_mask_hashes_match") is True,
        f"{location} three-mask replay receipt differs",
    )
    hard_rejected = not selected.hard_passed
    expected_counterfactual = track_id == "wide_raw" and hard_rejected
    expected_mode = (
        "canonical_environment_counterfactual_gate_bypass"
        if expected_counterfactual and canonical_available
        else "second_branch_counterfactual_gate_bypass"
        if expected_counterfactual
        else "persistent_canonical_environment_replay"
        if canonical_available
        else "second_branch_canonical_environment_replay"
    )
    require(
        replay.get("mode") == expected_mode
        and replay.get("counterfactual_gate_bypass") is expected_counterfactual
        and replay.get("environment_candidate_accepted") is (not hard_rejected)
        and replay.get("carried_state_source")
        == (
            "rejected_candidate_state_teacher_only"
            if expected_counterfactual else "accepted_environment_state"
        ),
        f"{location} replay mode/gate carry differs",
    )
    metrics = compute_metrics(replay_state.masks["change"], ground_truth)
    _require_numeric_mapping(replay.get("metrics"), metrics, f"{location}.metrics")
    require(replay.get("endpoint_gate") == endpoint_gate(metrics, context.endpoint),
            f"{location}.endpoint_gate differs")
    geometry = _mapping(selected_record.get("teacher_geometry"),
                        f"{location}.teacher_geometry")
    action = _mapping(selected_record.get("action"), f"{location}.action")
    replay_candidate = {
        "tool_receipt": replay.get("tool_receipt"),
        "runtime_execution": replay.get("runtime_execution"),
        "hard_gate": replay.get("hard_gate"),
    }
    _validate_tool_receipt(replay_candidate, action, geometry, f"{location}.tool")
    expected_hard = recompute_hard_gate(
        source, replay_state, action, geometry, context.hard_gate, location
    )
    _require_hard_gate_receipt(replay_candidate, expected_hard, f"{location}.tool")
    return replay_state


def _validate_track(
    track_value: Any,
    *,
    track_id: str,
    should_execute: bool,
    prior_winner: str | None,
    root: Path,
    initial: StateArtifact,
    ground_truth: np.ndarray,
    sample: Mapping[str, Any],
    sample_id: str,
    config_track: Mapping[str, Any],
    context: ValidationContext,
    location: str,
) -> tuple[ValidatedTrack, StateArtifact | None]:
    track = _mapping(track_value, location)
    require(
        track.get("track_name") == track_id
        and track.get("executed") is should_execute
        and track.get("teacher_only") is True
        and track.get("training_authorized") is False
        and track.get("qwen_invoked") is False
        and track.get("qwen_calls") == 0
        and track.get("gate_policy") == expected_gate_policy(track_id),
        f"{location} identity/policy/audit differs",
    )
    if not should_execute:
        require(
            prior_winner is not None
            and track.get("skip_reason") == "prior_track_endpoint_success"
            and track.get("prior_winner") == prior_winner
            and track.get("track_status") == "skipped_adaptive"
            and track.get("selection_rounds") == []
            and track.get("steps") == [],
            f"{location} adaptive skip receipt differs",
        )
        return ValidatedTrack(track_id, False, False, 0, 0, 0, 0, 0, 0, 0,
                              None, None, None), None

    require(
        track.get("max_steps") == 3
        and track.get("candidate_limit_per_step") == (4 if track_id == "matched_safe" else 96)
        and track.get("initial_state_mask_sha256") == initial.mask_hashes
        and track.get("expected_initial_state_mask_sha256") == initial.mask_hashes
        and track.get("fresh_initial_hash_match") is True,
        f"{location} fresh initial binding differs",
    )
    initial_metrics = compute_metrics(initial.masks["change"], ground_truth)
    _require_numeric_mapping(track.get("initial_metrics"), initial_metrics,
                             f"{location}.initial_metrics")
    initial_endpoint = endpoint_gate(initial_metrics, context.endpoint)
    require(track.get("initial_endpoint_gate") == initial_endpoint,
            f"{location}.initial_endpoint_gate differs")
    require(not initial_endpoint["passed"], f"{location} formal denominator starts at endpoint")

    current = initial
    rounds = _list(track.get("selection_rounds"), f"{location}.selection_rounds")
    require(len(rounds) <= 3, f"{location} exceeds max_steps")
    selected_steps: list[dict[str, Any]] = []
    counters = Counter()
    candidate_total = 0
    final_stop: str | None = None
    canonical_available = True
    for round_index, round_value in enumerate(rounds, 1):
        round_record = _mapping(round_value, f"{location}.round[{round_index}]")
        round_location = f"{location}.round[{round_index}]"
        require(
            round_record.get("step_index") == round_index
            and round_record.get("source_state_mask_sha256") == current.mask_hashes
            and round_record.get("source_state_sha256") == current.identity
            and round_record.get("candidate_batch_complete_before_selection") is True,
            f"{round_location} source/batch identity differs",
        )
        source_metrics = compute_metrics(current.masks["change"], ground_truth)
        _require_numeric_mapping(round_record.get("source_metrics"), source_metrics,
                                 f"{round_location}.source_metrics")
        require(round_record.get("source_endpoint_gate") == endpoint_gate(source_metrics, context.endpoint),
                f"{round_location}.source_endpoint_gate differs")
        raw_candidates = _list(round_record.get("candidates"), f"{round_location}.candidates")
        require(round_record.get("candidate_count") == len(raw_candidates),
                f"{round_location}.candidate_count differs")
        _validate_catalog(
            round_record.get("teacher_catalog"), source=current,
            ground_truth=ground_truth, track_id=track_id, step_index=round_index,
            sample_id=sample_id,
            sample=sample,
            candidates=raw_candidates, location=f"{round_location}.teacher_catalog",
        )
        ids: set[str] = set()
        validated: list[tuple[CandidateEvidence, Mapping[str, Any]]] = []
        for ordinal, raw_candidate in enumerate(raw_candidates):
            candidate_record = _mapping(raw_candidate, f"{round_location}.candidate[{ordinal}]")
            require(candidate_record.get("ordinal") == ordinal,
                    f"{round_location}.candidate[{ordinal}] ordinal differs")
            evidence = _validate_candidate(
                candidate_record, root=root, source=current, ground_truth=ground_truth,
                source_metrics=source_metrics, sample=sample, track=config_track,
                context=context, location=f"{round_location}.candidate[{ordinal}]",
            )
            require(evidence.candidate_id not in ids, f"{round_location} duplicate candidate ID")
            ids.add(evidence.candidate_id)
            reasons: list[str] = []
            if config_track.get("production_hard_gate_required") and not evidence.hard_passed:
                reasons.append("hard_gate_failed")
            if config_track.get("frozen_rgb_semantic_gate_required") and not evidence.rgb_passed:
                reasons.append("rgb_semantic_gate_failed")
            if not evidence.beneficial:
                reasons.append("strict_metric_improvement_failed")
            require(candidate_record.get("eligibility_rejection_reasons") == reasons,
                    f"{round_location}.candidate[{ordinal}] rejection reasons differ")
            boundary = _mapping(candidate_record.get("teacher_boundary_audit"),
                                f"{round_location}.candidate[{ordinal}].teacher_boundary_audit")
            require(
                boundary.get("raw_label_pixels_passed_to_environment") is False
                and boundary.get("raw_label_pixels_passed_to_tools") is False
                and boundary.get("gt_derived_geometry_passed_to_environment") is True
                and boundary.get("gt_derived_geometry_passed_to_tools") is True
                and boundary.get("gt_derived_metadata_passed_to_environment") is True
                and boundary.get("executor_receipt_contains_teacher_encoded_region_id") is True
                and boundary.get("runtime_boundary_layers") == RUNTIME_BOUNDARY_LAYERS
                and boundary.get("geometry_match") is True,
                f"{round_location}.candidate[{ordinal}] teacher boundary differs",
            )
            no_qwen = _mapping(candidate_record.get("no_qwen_audit"),
                               f"{round_location}.candidate[{ordinal}].no_qwen_audit")
            require(no_qwen.get("qwen_invoked") is False and no_qwen.get("qwen_calls") == 0,
                    f"{round_location}.candidate[{ordinal}] Qwen audit differs")
            require(candidate_record.get("proposal") == candidate_record.get("teacher_geometry"),
                    f"{round_location}.candidate[{ordinal}] proposal aliases differ")
            require(candidate_record.get("source_state_mask_sha256") == current.mask_hashes
                    and candidate_record.get("source_state_sha256") == current.identity,
                    f"{round_location}.candidate[{ordinal}] source differs")
            validated.append((evidence, candidate_record))
            counters["hard_pass"] += evidence.hard_passed
            counters["rgb_pass"] += evidence.rgb_passed
            counters["beneficial"] += evidence.beneficial
            counters["eligible"] += evidence.eligible
            counters["hard_rejected_beneficial"] += evidence.beneficial and not evidence.hard_passed
        candidate_total += len(validated)
        eligible = [(item, record) for item, record in validated if item.eligible]
        eligible.sort(key=lambda pair: (-int(pair[0].delta["pixel_error_reduction"]), pair[0].candidate_id))
        expected_ids = [item.candidate_id for item, _ in eligible]
        require(round_record.get("beneficial_candidate_ids") == expected_ids,
                f"{round_location}.beneficial_candidate_ids sorting differs")
        selected_id = round_record.get("selected_candidate_id")
        if not eligible:
            require(selected_id is None and round_record.get("canonical_replay") is None,
                    f"{round_location} selected without eligible candidate")
            final_stop = "no_eligible_beneficial"
            require(round_index == len(rounds), f"{round_location} continued after no candidate")
            break
        require(selected_id == eligible[0][0].candidate_id,
                f"{round_location} selected candidate sorting differs")
        selected, selected_record = eligible[0]
        require(
            sum(_mapping(record, "candidate").get("selected") is True for _, record in validated) == 1
            and selected_record.get("selected") is True,
            f"{round_location} selected flag differs",
        )
        replay_state = _validate_replay(
            round_record.get("canonical_replay"), root=root, source=current,
            selected=selected, selected_record=selected_record, ground_truth=ground_truth,
            sample=sample, track_id=track_id, canonical_available=canonical_available,
            context=context,
            location=f"{round_location}.canonical_replay",
        )
        selected_steps.append({
            "record": selected_record,
            "candidate": selected,
            "replay": replay_state,
            "replay_receipt": round_record.get("canonical_replay"),
        })
        # Only the original max_steps=3 canonical Environment can persist.
        # A second branch has max_steps=1 and is exhausted even when accepted.
        canonical_available = canonical_available and selected.hard_passed
        current = replay_state
        if endpoint_gate(compute_metrics(current.masks["change"], ground_truth), context.endpoint)["passed"]:
            final_stop = "exact_recovery" if compute_metrics(current.masks["change"], ground_truth)["pixel_error"] == 0 else "endpoint_reached"
            require(round_index == len(rounds), f"{round_location} continued after endpoint")
            break

    if final_stop is None:
        final_stop = "max_steps" if len(selected_steps) == 3 else "no_eligible_beneficial"
    step_receipts = _list(track.get("steps"), f"{location}.steps")
    require(len(step_receipts) == len(selected_steps), f"{location}.steps count differs")
    for index, (step_value, validated_step) in enumerate(zip(step_receipts, selected_steps), 1):
        step = _mapping(step_value, f"{location}.steps[{index}]")
        record = validated_step["record"]
        replay_receipt = validated_step["replay_receipt"]
        require(
            step.get("step_index") == index
            and step.get("candidate_id") == record.get("candidate_id")
            and step.get("region_id") == record.get("region_id")
            and step.get("action") == record.get("action")
            and step.get("teacher_geometry") == record.get("teacher_geometry")
            and step.get("source_state_mask_sha256") == record.get("source_state_mask_sha256")
            and step.get("candidate_state") == record.get("candidate_state")
            and step.get("canonical_replay") == replay_receipt
            and step.get("metrics_before") == track["selection_rounds"][index - 1]["source_metrics"]
            and step.get("metrics_after") == replay_receipt["metrics"]
            and step.get("metric_delta") == record.get("metric_delta")
            and step.get("raw_metric_beneficial") == record.get("raw_metric_beneficial")
            and step.get("hard_gate") == record.get("hard_gate")
            and step.get("rgb_semantic_gate") == record.get("rgb_semantic_gate")
            and step.get("counterfactual_gate_bypass") == replay_receipt["counterfactual_gate_bypass"]
            and step.get("all_three_mask_hashes_match") is True
            and step.get("qwen_invoked") is False,
            f"{location}.steps[{index}] differs from selected round evidence",
        )

    final_state = _load_state(root, track.get("final_state"), f"{location}.final_state")
    _require_rebuilt_change(final_state, context, f"{location}.final_state")
    _same_state(current, final_state, f"{location} carried/final")
    final_metrics = compute_metrics(final_state.masks["change"], ground_truth)
    _require_numeric_mapping(track.get("final_metrics"), final_metrics,
                             f"{location}.final_metrics")
    final_endpoint = endpoint_gate(final_metrics, context.endpoint)
    expected_success = bool(final_endpoint["passed"])
    expected_status = "endpoint_success" if expected_success else "not_reachable_within_track"
    require(
        track.get("final_state_mask_sha256") == final_state.mask_hashes
        and track.get("final_state_sha256") == final_state.identity
        and track.get("endpoint_reachability") == {
            "initial_gate": initial_endpoint,
            "final_gate": final_endpoint,
            "success_step": len(selected_steps) if expected_success else None,
            "exact_recovery": int(final_metrics["pixel_error"]) == 0,
        }
        and track.get("track_status") == expected_status,
        f"{location} final state/endpoint status differs",
    )
    termination = _mapping(track.get("termination"), f"{location}.termination")
    require(
        termination.get("reason") == final_stop
        and termination.get("selected_step_count") == len(selected_steps)
        and termination.get("candidate_count") == candidate_total
        and termination.get("replay_count") == len(selected_steps)
        and track.get("counterfactual_selected_steps")
        == sum(bool(item["replay_receipt"]["counterfactual_gate_bypass"]) for item in selected_steps),
        f"{location}.termination differs",
    )
    return ValidatedTrack(
        track_id, True, expected_success, len(selected_steps), candidate_total,
        counters["hard_pass"], counters["rgb_pass"], counters["beneficial"],
        counters["eligible"], counters["hard_rejected_beneficial"],
        initial_metrics, final_metrics, final_stop,
    ), final_state


def _expected_episode_conclusion(winner: str | None) -> dict[str, Any]:
    if winner == "matched_safe":
        classification = "proposal_geometry_or_coverage_bottleneck_supported_matched"
        plain = "Matched GT-derived geometry reaches endpoint under production gates; current Proposal is at least one bottleneck."
    elif winner == "wide_safe":
        classification = "proposal_geometry_or_coverage_bottleneck_supported"
        plain = "Wider GT-derived geometry succeeds under production gates; current Proposal is at least one bottleneck."
    elif winner == "wide_hard":
        classification = "rgb_semantic_gate_bottleneck_supported"
        plain = "Real tools and hard gates succeed only after ignoring frozen RGB routing."
    elif winner == "wide_raw":
        classification = "hard_gate_bottleneck_supported_counterfactual"
        plain = "Metric-only real-tool candidates succeed after wide_hard failed; production hard gate blocks the bounded trajectory."
    else:
        classification = "tool_upper_bound_not_reached_within_frozen_protocol"
        plain = "The bounded greedy wide GT-derived protocol did not find an endpoint in three real-tool steps; this does not isolate a tool-only ceiling."
    return {
        "classification": classification,
        "plain_language": plain,
        "winning_track": winner,
        "proposal_bottleneck_supported": winner in {"matched_safe", "wide_safe"},
        "proposal_only_claim_allowed": False,
        "tool_upper_bound_reached": winner is not None,
        "production_safe_upper_bound_reached": winner in {"matched_safe", "wide_safe"},
    }


def validate_episode(
    payload: Mapping[str, Any],
    *,
    root: Path,
    expected_sample_id: str,
    source_episode_sha256: str,
    config: Mapping[str, Any],
    context: ValidationContext,
) -> ValidatedEpisode:
    location = f"episode {expected_sample_id}"
    sample = context.samples[expected_sample_id]
    require(
        payload.get("schema_id") == EPISODE_SCHEMA_ID
        and payload.get("status") == "completed"
        and payload.get("episode_id") == expected_sample_id
        and payload.get("sample_id") == expected_sample_id
        and payload.get("split") == sample.get("split")
        and payload.get("parent_id") == sample.get("parent_id")
        and payload.get("teacher_only") is True
        and payload.get("training_authorized") is False
        and payload.get("claim_scope") == "LEVIR_change_mask_only_teacher_tool_upper_bound"
        and payload.get("qwen_invoked") is False
        and payload.get("qwen_calls") == 0,
        f"{location} identity/safety audit differs",
    )
    base_protocol = _mapping(context.base_config.get("execution_protocol"),
                             "base execution_protocol")
    require(payload.get("query") == base_protocol.get("query"),
            f"{location} query differs")
    expected_crop = _mapping(sample.get("crop"), f"sample {expected_sample_id}.crop")
    crop = expected_crop.get("xyxy_exclusive", expected_crop.get("crop_xyxy_exclusive"))
    require(payload.get("crop_xyxy_exclusive") == crop,
            f"{location} crop differs")
    source_hashes = _mapping(payload.get("source_hashes"), f"{location}.source_hashes")
    expected_source = _mapping(sample.get("source_parent"),
                               f"sample {expected_sample_id}.source_parent")
    source_specs: dict[str, Mapping[str, Any]] = {}
    for name, aliases in {"t1": ("t1", "parent_a"), "t2": ("t2", "parent_b"), "label": ("label", "parent_label")}.items():
        base_value = next((expected_source[key] for key in aliases if key in expected_source), None)
        base_spec = _mapping(base_value, f"sample {expected_sample_id}.source_parent.{name}")
        source_specs[name] = base_spec
        require(source_hashes.get(name) == base_spec.get("sha256"),
                f"{location} source {name} SHA256 differs")
        parent_spec = _mapping(_mapping(payload.get("source_parent"), f"{location}.source_parent").get(name),
                               f"{location}.source_parent.{name}")
        require(parent_spec.get("sha256") == base_spec.get("sha256")
                and Path(str(parent_spec.get("path"))).resolve() == Path(str(base_spec.get("path"))).resolve(),
                f"{location} source parent {name} differs")

    ground_truth = _verified_mask(root, payload.get("gt_mask"), f"{location}.gt_mask")
    require(int(ground_truth.sum()) == expected_crop.get("gt_pixels"),
            f"{location} GT pixels differ")
    label_path = Path(str(source_specs["label"].get("path"))).resolve()
    require(
        label_path.is_file()
        and _is_sha256(source_specs["label"].get("sha256"))
        and sha256_file(label_path) == str(source_specs["label"]["sha256"]).lower(),
        f"{location} frozen parent label identity differs",
    )
    x1, y1, x2, y2 = (int(item) for item in crop)
    try:
        with Image.open(label_path) as image:
            parent_ground_truth = np.asarray(
                image.convert("L").crop((x1, y1, x2, y2)), dtype=np.uint8
            ) > 0
    except (OSError, ValueError) as error:
        raise ValidationError(f"{location} cannot load frozen parent label: {error}") from error
    require(
        parent_ground_truth.shape == ground_truth.shape
        and np.array_equal(parent_ground_truth, ground_truth),
        f"{location} saved GT differs spatially from frozen parent label crop",
    )
    audit = _mapping(payload.get("gt_access_audit"), f"{location}.gt_access_audit")
    require(
        audit.get("access_count") == 1
        and audit.get("first_access_phase") == "after_fresh_sam3_initialization_and_hash_match"
        and audit.get("label_passed_to_runtime_factory") is False
        and audit.get("raw_label_path_passed_to_runtime") is False
        and audit.get("raw_label_pixels_passed_to_environment") is False
        and audit.get("raw_label_pixels_passed_to_tools") is False
        and audit.get("raw_label_pixels_passed_to_qwen") is False
        and audit.get("gt_derived_geometry_passed_to_environment") is True
        and audit.get("gt_derived_geometry_passed_to_tools") is True
        and audit.get("gt_derived_metadata_passed_to_environment") is True
        and audit.get("executor_receipt_contains_teacher_encoded_region_id") is True
        and audit.get("runtime_boundary_layers") == RUNTIME_BOUNDARY_LAYERS
        and audit.get("runtime_factory_sample_type") == "RuntimeEpisode_GT_free"
        and audit.get("offline_label_loader_only") is True,
        f"{location} GT access boundary differs",
    )
    initial = _load_state(root, payload.get("initial_state"), f"{location}.initial_state")
    _require_rebuilt_change(initial, context, f"{location}.initial_state")
    initializer = _mapping(
        _mapping(payload.get("initial_state"), f"{location}.initial_state").get("initializer_receipt"),
        f"{location}.initial_state.initializer_receipt",
    )
    initializer_report = _mapping(
        initializer.get("worker_report"),
        f"{location}.initial_state.initializer_receipt.worker_report",
    )
    require(
        initializer.get("worker_mode") == "initialize"
        and initializer_report.get("status") == "success"
        and initializer_report.get("mode") == "initialize"
        and initializer_report.get("tool") == "sam3"
        and initializer_report.get("operation") == "fresh_dual_view_text_initialization",
        f"{location} initial state is not from fresh real SAM3",
    )
    require(
        initial.mask_hashes == context.expected_initial_hashes[expected_sample_id]
        and payload.get("expected_initial_state_mask_sha256") == initial.mask_hashes
        and payload.get("fresh_initial_hash_match") is True,
        f"{location} fresh SAM3 initial hashes differ",
    )
    initial_metrics = compute_metrics(initial.masks["change"], ground_truth)
    _require_numeric_mapping(payload.get("initial_metrics"), initial_metrics,
                             f"{location}.initial_metrics")
    initial_endpoint = endpoint_gate(initial_metrics, context.endpoint)
    require(payload.get("initial_endpoint_gate") == initial_endpoint and not initial_endpoint["passed"],
            f"{location} initial endpoint receipt differs")
    rgb = _mapping(payload.get("rgb_semantics"), f"{location}.rgb_semantics")
    semantic = _mapping(sample.get("semantic"), f"sample {expected_sample_id}.semantic")
    require(
        rgb.get("default") == {"t1": semantic.get("t1_state"), "t2": semantic.get("t2_state")}
        and rgb.get("evidence") == semantic.get("rgb_evidence", "")
        and rgb.get("source") == semantic.get("source", "")
        and rgb.get("status") == semantic.get("status", ""),
        f"{location} RGB semantic provenance differs",
    )

    adaptive = _mapping(payload.get("adaptive_execution"), f"{location}.adaptive_execution")
    tracks_value = _mapping(payload.get("tracks"), f"{location}.tracks")
    require(list(tracks_value) == list(TRACK_NAMES), f"{location} track order differs")
    require(
        adaptive.get("track_order") == list(TRACK_NAMES)
        and adaptive.get("stops_after_first_endpoint_success") is True,
        f"{location} adaptive contract differs",
    )
    config_tracks = _track_policy(config)
    winner: str | None = None
    executed: list[str] = []
    validated_tracks: dict[str, ValidatedTrack] = {}
    for track_id in TRACK_NAMES:
        should_execute = winner is None
        track, _ = _validate_track(
            tracks_value[track_id], track_id=track_id, should_execute=should_execute,
            prior_winner=winner, root=root, initial=initial, ground_truth=ground_truth,
            sample=sample, sample_id=expected_sample_id,
            config_track=config_tracks[track_id], context=context,
            location=f"{location}.tracks.{track_id}",
        )
        validated_tracks[track_id] = track
        if should_execute:
            executed.append(track_id)
            if track.endpoint_success:
                winner = track_id
    require(
        adaptive.get("executed_tracks") == executed
        and adaptive.get("winning_track") == winner,
        f"{location} adaptive execution/winner differs",
    )
    expected_conclusion = _expected_episode_conclusion(winner)
    require(payload.get("conclusion") == expected_conclusion,
            f"{location} conclusion differs")
    expected_status = "teacher_endpoint_success" if winner is not None else "teacher_endpoint_not_reached"
    require(payload.get("episode_status") == expected_status,
            f"{location} episode_status differs")
    return ValidatedEpisode(
        expected_sample_id, str(sample.get("parent_id")), str(sample.get("split")),
        source_episode_sha256, winner, expected_status,
        str(expected_conclusion["classification"]), validated_tracks, initial_metrics,
    )


def _qwen_teacher_safety(value: Mapping[str, Any], location: str) -> None:
    require(
        value.get("teacher_only") is True
        and value.get("training_authorized") is False
        and value.get("qwen_invoked") is False
        and value.get("qwen_calls") == 0,
        f"{location} teacher/Qwen/training safety audit differs",
    )


def _verify_complete_marker(
    root: Path, index_path: Path, index: Mapping[str, Any], sample_id: str
) -> dict[str, Any]:
    marker_path = root / "generation_complete.json"
    marker = read_json(marker_path, f"generation marker {root}")
    require(
        marker.get("schema_id") == COMPLETE_SCHEMA_ID
        and marker.get("status") == "completed"
        and marker.get("shard_index") == index.get("shard_index")
        and marker.get("num_shards") == EXPECTED_SHARD_COUNT
        and marker.get("episode_ids") == [sample_id]
        and marker.get("runtime_commit") == index.get("runtime_commit")
        and marker.get("counts") == index.get("counts"),
        f"generation marker {root} differs from shard index",
    )
    _qwen_teacher_safety(marker, f"generation marker {root}")
    artifact = _mapping(marker.get("shard_index_artifact"),
                        f"generation marker {root}.shard_index_artifact")
    require(
        artifact.get("path") == "shard_index.json"
        and artifact.get("sha256") == sha256_file(index_path)
        and artifact.get("canonical_sha256") == canonical_sha256(index)
        and artifact.get("size_bytes") == index_path.stat().st_size
        and marker.get("shard_index_sha256") == sha256_file(index_path),
        f"generation marker {root} index binding differs",
    )
    return marker


def _expected_index_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    tracks = _mapping(payload.get("tracks"), "episode.tracks")
    return {
        "episodes": 1,
        "teacher_endpoint_success": int(payload.get("episode_status") == "teacher_endpoint_success"),
        "teacher_endpoint_not_reached": int(payload.get("episode_status") == "teacher_endpoint_not_reached"),
        "executed_tracks": sum(_mapping(value, "track").get("executed") is True for value in tracks.values()),
        "selected_steps": sum(len(_list(_mapping(value, "track").get("steps"), "track.steps")) for value in tracks.values()),
        "candidates": sum(
            sum(int(_mapping(round_value, "round").get("candidate_count"))
                for round_value in _list(_mapping(value, "track").get("selection_rounds"), "track.selection_rounds"))
            for value in tracks.values()
        ),
    }


def validate_inputs(
    config: Mapping[str, Any],
    config_path: Path,
    input_roots: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = validate_config(config, config_path)
    require(len(input_roots) == EXPECTED_SHARD_COUNT,
            "validation requires exactly four input shard roots")
    runner_path = context.repository / "tools" / "run_qwen3vl_gt_direct_teacher.py"
    require(runner_path.is_file(), "teacher runner is missing")
    runner_sha = sha256_file(runner_path)
    runtime_commits: set[str] = set()
    observed_shards: set[int] = set()
    payloads: dict[str, tuple[dict[str, Any], Path, str]] = {}
    shard_bindings: list[dict[str, Any]] = []
    for input_root in input_roots:
        root = input_root.resolve()
        require(root.is_dir(), f"input shard root is missing: {root}")
        index_path = root / "shard_index.json"
        index = read_json(index_path, f"shard index {root}")
        require(index.get("schema_id") == INDEX_SCHEMA_ID and index.get("status") == "completed",
                f"shard index {root} schema/status differs")
        _qwen_teacher_safety(index, f"shard index {root}")
        shard_id = index.get("shard_index")
        require(type(shard_id) is int and 0 <= shard_id < EXPECTED_SHARD_COUNT,
                f"shard index {root} identity is invalid")
        require(shard_id not in observed_shards, "input shard identity is duplicated")
        observed_shards.add(shard_id)
        sample_id = context.ordered_samples[shard_id]
        require(index.get("num_shards") == EXPECTED_SHARD_COUNT
                and index.get("episode_ids") == [sample_id],
                f"shard {shard_id} sample assignment differs")
        index_config = _mapping(index.get("config"), f"shard {shard_id}.config")
        require(
            index_config.get("sha256") == context.config_sha256
            and index_config.get("canonical_sha256") == context.config_canonical_sha256
            and Path(str(index_config.get("path"))).name == context.config_path.name,
            f"shard {shard_id} config binding differs",
        )
        base_binding = _mapping(index.get("base_config_0803"),
                                f"shard {shard_id}.base_config_0803")
        require(
            base_binding.get("sha256") == context.base_config_sha256
            and base_binding.get("canonical_sha256") == canonical_sha256(context.base_config)
            and Path(str(base_binding.get("path"))).name == "qwen3vl_oracle_trajectory_pilot_0803.json",
            f"shard {shard_id} base config binding differs",
        )
        runtime_commit = index.get("runtime_commit")
        require(_is_git_sha(runtime_commit), f"shard {shard_id} runtime commit is invalid")
        runtime_commits.add(str(runtime_commit).lower())
        runner = _mapping(index.get("runner"), f"shard {shard_id}.runner")
        require(
            Path(str(runner.get("path"))).name == runner_path.name
            and runner.get("sha256") == runner_sha,
            f"shard {shard_id} runner binding differs",
        )
        entries = _list(index.get("episodes"), f"shard {shard_id}.episodes")
        require(len(entries) == 1, f"shard {shard_id} must contain one episode")
        entry = _mapping(entries[0], f"shard {shard_id}.episodes[0]")
        require(entry.get("episode_id") == sample_id,
                f"shard {shard_id} episode identity differs")
        episode_path = _inside(root, entry.get("path"), f"shard {shard_id} episode path")
        require(episode_path.is_file(), f"shard {shard_id} episode is missing")
        episode_sha = sha256_file(episode_path)
        require(
            entry.get("sha256") == episode_sha
            and entry.get("size_bytes") == episode_path.stat().st_size,
            f"shard {shard_id} episode raw identity differs",
        )
        payload = read_json(episode_path, f"episode {sample_id}")
        require(entry.get("canonical_sha256") == canonical_sha256(payload),
                f"shard {shard_id} episode canonical identity differs")
        require(
            entry.get("episode_status") == payload.get("episode_status")
            and entry.get("winning_track") == _mapping(payload.get("adaptive_execution"), "adaptive").get("winning_track")
            and entry.get("conclusion") == _mapping(payload.get("conclusion"), "conclusion").get("classification"),
            f"shard {shard_id} episode summary identity differs",
        )
        expected_counts = _expected_index_counts(payload)
        require(index.get("counts") == expected_counts,
                f"shard {shard_id} counts differ from episode")
        _verify_complete_marker(root, index_path, index, sample_id)
        require(sample_id not in payloads, "episode appears in multiple shards")
        payloads[sample_id] = (payload, root, episode_sha)
        shard_bindings.append({
            "shard_index": shard_id,
            "sample_id": sample_id,
            "root": str(root),
            "shard_index_sha256": sha256_file(index_path),
            "generation_complete_sha256": sha256_file(root / "generation_complete.json"),
            "runtime_commit": str(runtime_commit).lower(),
            "runner_sha256": runner_sha,
        })
    require(observed_shards == set(range(EXPECTED_SHARD_COUNT)),
            "four-shard identity coverage differs")
    require(len(runtime_commits) == 1, "runtime commit differs across shards")
    require(set(payloads) == set(context.ordered_samples),
            "combined shard sample coverage differs")
    episodes = [
        validate_episode(
            payloads[sample_id][0], root=payloads[sample_id][1],
            expected_sample_id=sample_id, source_episode_sha256=payloads[sample_id][2],
            config=config, context=context,
        )
        for sample_id in context.ordered_samples
    ]
    report = build_report(episodes, context, sorted(shard_bindings, key=lambda item: item["shard_index"]),
                          next(iter(runtime_commits)), runner_sha)
    manifest = {
        "schema_id": ARTIFACT_MANIFEST_SCHEMA_ID,
        "status": "validated",
        "teacher_only": True,
        "training_authorized": False,
        "config_sha256": context.config_sha256,
        "runtime_commit": next(iter(runtime_commits)),
        "validated_episode_sha256": {
            episode.sample_id: episode.source_episode_sha256 for episode in episodes
        },
    }
    return report, manifest


def _wilson_rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(type(numerator) is int and type(denominator) is int
            and 0 <= numerator <= denominator, "invalid rate")
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": None,
            "wilson95_lower": None,
            "wilson95_upper": None,
        }
    z = 1.959963984540054
    rate = numerator / denominator
    scale = 1.0 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(
        rate * (1 - rate) / denominator
        + z * z / (4 * denominator * denominator)
    ) / scale
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "wilson95_lower": max(0.0, center - radius),
        "wilson95_upper": min(1.0, center + radius),
    }


def build_report(
    episodes: Sequence[ValidatedEpisode],
    context: ValidationContext,
    shard_bindings: Sequence[Mapping[str, Any]],
    runtime_commit: str,
    runner_sha256: str,
) -> dict[str, Any]:
    require(len(episodes) == 4, "report requires four validated episodes")
    winners = Counter(episode.winning_track or "none" for episode in episodes)
    reached = sum(episode.winning_track is not None for episode in episodes)
    safe_reached = sum(
        episode.winning_track in {"matched_safe", "wide_safe"}
        for episode in episodes
    )
    if safe_reached == 4:
        scientific_conclusion = "proposal_bottleneck_supported"
        detailed_conclusion = (
            "All four 0803 production failures reached endpoint with GT-direct "
            "geometry while retaining production hard and RGB gates. Proposal "
            "geometry/coverage is demonstrated as at least one bottleneck, not "
            "necessarily the only bottleneck."
        )
    elif reached == 4:
        scientific_conclusion = "tool_upper_bound_supported"
        detailed_conclusion = (
            "All four samples reached endpoint somewhere in the bounded GT-direct "
            "track ladder, but at least one required bypassing a production safety "
            "layer. Real-tool capacity is supported; Proposal alone is not a valid "
            "global explanation."
        )
    elif reached == 0:
        scientific_conclusion = "greedy_protocol_inconclusive"
        detailed_conclusion = (
            "No sample reached endpoint even under wide metric-only GT-direct "
            "geometry within three steps. This bounded greedy protocol did not "
            "demonstrate reachability, but it does not isolate a tool-only ceiling; "
            "failure cannot be assigned solely to Proposal or solely to the tools."
        )
    else:
        scientific_conclusion = "inconclusive"
        detailed_conclusion = (
            "Samples have mixed bottlenecks: some GT-direct tracks reached endpoint "
            "and others did not. Report per-sample layers; do not force one global "
            "Proposal-versus-tool label."
        )

    track_reports: dict[str, Any] = {}
    cumulative_success = 0
    for track_id in TRACK_NAMES:
        track_values = [episode.tracks[track_id] for episode in episodes]
        executed = [item for item in track_values if item.executed]
        winner_count = winners.get(track_id, 0)
        cumulative_success += winner_count
        track_reports[track_id] = {
            "gate_policy": expected_gate_policy(track_id),
            "executed_episode_count": len(executed),
            "adaptive_winner_count": winner_count,
            "endpoint_success_among_executed": _wilson_rate(
                sum(item.endpoint_success for item in executed), len(executed)
            ),
            "cumulative_endpoint_reachability_over_original_four": _wilson_rate(
                cumulative_success, 4
            ),
            "selected_step_count": sum(item.selected_steps for item in executed),
            "candidate_count": sum(item.candidate_count for item in executed),
            "hard_gate_pass_count": sum(item.hard_pass_count for item in executed),
            "rgb_gate_pass_count": sum(item.rgb_pass_count for item in executed),
            "raw_beneficial_count": sum(item.beneficial_count for item in executed),
            "eligible_count": sum(item.eligible_count for item in executed),
            "beneficial_but_hard_rejected_count": sum(
                item.hard_rejected_beneficial_count for item in executed
            ),
            "termination_reasons": dict(sorted(Counter(
                item.termination_reason for item in executed if item.termination_reason
            ).items())),
        }

    per_episode = []
    for episode in episodes:
        per_episode.append({
            "sample_id": episode.sample_id,
            "parent_id": episode.parent_id,
            "split": episode.split,
            "teacher_only": True,
            "training_authorized": False,
            "episode_status": episode.episode_status,
            "winning_track": episode.winning_track,
            "detailed_conclusion": episode.conclusion,
            "proposal_bottleneck_supported": episode.winning_track in {"matched_safe", "wide_safe"},
            "tool_upper_bound_reached": episode.winning_track is not None,
            "source_episode_sha256": episode.source_episode_sha256,
            "initial_metrics": episode.initial_metrics,
            "tracks": {
                track_id: {
                    "executed": value.executed,
                    "endpoint_success": value.endpoint_success,
                    "selected_steps": value.selected_steps,
                    "candidate_count": value.candidate_count,
                    "termination_reason": value.termination_reason,
                    "final_metrics": value.final_metrics,
                }
                for track_id, value in episode.tracks.items()
            },
        })
    validator_path = Path(__file__).resolve()
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "status": "validated",
        "infrastructure_validation_passed": True,
        "scientific_conclusion": scientific_conclusion,
        "detailed_conclusion": detailed_conclusion,
        "teacher_only": True,
        "change_only_teacher_upper_bound": True,
        "training_authorized": False,
        "authorization_scope": "none",
        "formal_c_iso_or_t_traj_allowed": False,
        "raw_artifacts_sft_ready": False,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "generalization_claim_allowed": False,
        "max_steps": 3,
        "episode_count": 4,
        "production_baseline_0803": {
            "endpoint_successes": 0,
            "denominator": 4,
            "rate": 0.0,
            "validation_report_sha256": context.baseline_report_sha256,
        },
        "teacher_upper_bound": {
            "endpoint_successes": reached,
            "denominator": 4,
            "rate": reached / 4,
            "production_safe_endpoint_successes": safe_reached,
            "proposal_bottleneck_supported_samples": safe_reached,
            "winner_distribution": dict(sorted(winners.items())),
        },
        "tracks": track_reports,
        "episodes": per_episode,
        "bindings": {
            "runtime_commit": runtime_commit,
            "config": {
                "path": str(context.config_path),
                "sha256": context.config_sha256,
                "canonical_sha256": context.config_canonical_sha256,
            },
            "base_config_0803": {
                "sha256": context.base_config_sha256,
                "canonical_sha256": canonical_sha256(context.base_config),
            },
            "production_validation_report_0803": {
                "sha256": context.baseline_report_sha256,
                "canonical_sha256": canonical_sha256(context.baseline_report),
            },
            "runner": {"sha256": runner_sha256},
            "validator": {"path": str(validator_path), "sha256": sha256_file(validator_path)},
            "shards": [dict(item) for item in shard_bindings],
        },
        "limitations": [
            "LEVIR-CD supplies change GT but no T1/T2 building-mask GT.",
            "GT-derived coordinates are teacher-only and unavailable during deployment.",
            "Four previously inspected pilot samples do not support a generalization claim.",
            "Failure is bounded to current tools, composition, geometry family and max_steps=3.",
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_preflight_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    target = path.resolve()
    require(not target.exists(), f"refusing to overwrite preflight receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    require(not temporary.exists(), f"temporary preflight receipt exists: {temporary}")
    try:
        _write_json(temporary, receipt)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(
    output: Path, report: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    target = output.resolve()
    require(not target.exists(), f"refusing to overwrite validation output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        report_path = temporary / "validation_report.json"
        _write_json(report_path, report)
        completed_manifest = dict(manifest)
        completed_manifest["validation_report"] = {
            "path": "validation_report.json",
            "sha256": sha256_file(report_path),
            "canonical_sha256": canonical_sha256(report),
            "size_bytes": report_path.stat().st_size,
        }
        _write_json(temporary / "artifact_manifest.json", completed_manifest)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = read_json(args.config, "teacher config")
    if args.preflight_only:
        if args.input_root or args.output is not None:
            parser.error("--preflight-only cannot use --input-root/--output")
        if args.preflight_receipt is None:
            parser.error("--preflight-only requires --preflight-receipt")
        receipt = run_preflight(config, args.config)
        write_preflight_receipt(args.preflight_receipt, receipt)
        print(json.dumps({"status": "preflight_passed", "teacher_only": True,
                          "training_authorized": False}, sort_keys=True))
        return 0
    if args.preflight_receipt is not None:
        parser.error("--preflight-receipt requires --preflight-only")
    if args.output is None or not args.input_root:
        parser.error("validation requires four --input-root values and --output")
    if len(args.input_root) != EXPECTED_SHARD_COUNT:
        parser.error("validation requires exactly four --input-root values")
    report, manifest = validate_inputs(config, args.config, args.input_root)
    write_outputs(args.output, report, manifest)
    print(json.dumps({
        "status": "validated",
        "scientific_conclusion": report["scientific_conclusion"],
        "teacher_only": True,
        "training_authorized": False,
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
