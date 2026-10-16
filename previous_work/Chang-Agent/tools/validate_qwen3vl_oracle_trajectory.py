#!/usr/bin/env python3
"""Independently validate and derive the Qwen3-VL oracle-trajectory pilot.

This module is deliberately independent from the generator.  It reads only
completed shard artifacts, reloads every saved mask, recomputes all GT metrics,
and derives matched C-ISO/T-TRAJ records only from successful episodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


CONFIG_SCHEMA = "qwen3vl_oracle_trajectory_pilot_preregistration_v1"
INDEX_SCHEMAS = {
    "qwen3vl_oracle_trajectory_shard_index_v1",
    "qwen3vl_oracle_trajectory_index_v1",
}
COMPLETE_SCHEMAS = {
    "qwen3vl_oracle_trajectory_generation_complete_v1",
    "qwen3vl_oracle_trajectory_shard_complete_v1",
}
EPISODE_SCHEMAS = {
    "qwen3vl_oracle_trajectory_episode_v1",
    "qwen3vl_oracle_trajectory_canonical_episode_v1",
}
REPORT_SCHEMA = "qwen3vl_oracle_trajectory_validation_report_v1"
T_TRAJ_SCHEMA = "qwen3vl_t_traj_data_v1"
C_ISO_SCHEMA = "qwen3vl_c_iso_data_v1"
MATCHING_SCHEMA = "qwen3vl_c_iso_t_traj_matching_manifest_v1"
PREFLIGHT_SCHEMA = "qwen3vl_oracle_trajectory_preflight_receipt_v1"
MAX_STEPS = 3
MAX_CANDIDATES_PER_STEP = 4
MAX_DISTINCT_CANDIDATES = MAX_STEPS * MAX_CANDIDATES_PER_STEP
MAX_TOOL_EXECUTIONS = 24
EXPECTED_OUTPUT_ROOT = (
    "/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/"
    "CA_0803(2)-qwen3vl-oracle-trajectory-pilot"
)
DEFAULT_ENDPOINT_IOU = 0.90
DEFAULT_ENDPOINT_F1 = 18.0 / 19.0
DEFAULT_EMPTY_GT_MAX_FP = 64
DEFAULT_LOCAL_ERROR_REDUCTION_RATIO = 0.05
WILSON_Z_95 = 1.959963984540054
SHA256_LENGTH = 64
GIT_SHA_LENGTH = 40
CATALOG_SCHEMA = "qwen3vl_oracle_trajectory_catalog_freeze_v1"
FULL_LIVE_AUDIT_KINDS = frozenset({"present", "missing", "mixed", "scan"})
EVIDENCE_ONLY_AUDIT_KINDS = frozenset({"delta_added", "delta_removed"})
KNOWN_AUDIT_KINDS = FULL_LIVE_AUDIT_KINDS | EVIDENCE_ONLY_AUDIT_KINDS
CONFIG_STATUSES = {
    "preregistered_pending_implementation_and_runtime_validation",
    "frozen_pending_exact_commit_validation",
}


class ValidationError(ValueError):
    """Raised when an artifact cannot satisfy the frozen pilot contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
    """Canonical accepted-state identity shared by validator and tests."""

    payload = {
        "t1_mask_sha256": mask_hashes["t1"],
        "t2_mask_sha256": mask_hashes["t2"],
        "change_mask_sha256": mask_hashes["change"],
    }
    return canonical_sha256(payload)


def compute_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    require(prediction.shape == reference.shape, "prediction/GT shape mismatch")
    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, np.logical_not(reference)).sum())
    fn = int(np.logical_and(np.logical_not(prediction), reference).sum())
    union = tp + fp + fn
    pred_count = tp + fp
    gt_count = tp + fn
    iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if pred_count == 0 and gt_count == 0 else (
        0.0 if pred_count == 0 else tp / pred_count
    )
    recall = 1.0 if gt_count == 0 else tp / gt_count
    f1_denominator = 2 * tp + fp + fn
    f1 = 1.0 if f1_denominator == 0 else 2 * tp / f1_denominator
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pixel_error": fp + fn,
        "predicted_pixels": pred_count,
        "gt_pixels": gt_count,
    }


def endpoint_gate(
    metrics: Mapping[str, Any],
    *,
    minimum_iou: float,
    minimum_f1: float,
    empty_gt_max_fp: int,
    epsilon: float,
) -> dict[str, Any]:
    """Independently recompute the preregistered absolute endpoint gate."""

    nonempty_gt = int(metrics["gt_pixels"]) > 0
    if nonempty_gt:
        iou_passed = float(metrics["iou"]) + epsilon >= minimum_iou
        f1_passed = float(metrics["f1"]) + epsilon >= minimum_f1
        fp_passed = None
        passed = iou_passed and f1_passed
    else:
        iou_passed = None
        f1_passed = None
        fp_passed = int(metrics["fp"]) <= empty_gt_max_fp
        passed = fp_passed
    return {
        "passed": bool(passed),
        "nonempty_gt": nonempty_gt,
        "minimum_iou": minimum_iou if nonempty_gt else None,
        "minimum_f1": minimum_f1 if nonempty_gt else None,
        "empty_gt_max_fp": empty_gt_max_fp if not nonempty_gt else None,
        "iou_passed": iou_passed,
        "f1_passed": f1_passed,
        "fp_residual_passed": fp_passed,
        "comparison_epsilon": epsilon,
    }


def wilson_rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(
        type(numerator) is int
        and type(denominator) is int
        and 0 <= numerator <= denominator,
        "invalid rate numerator/denominator",
    )
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": None,
            "wilson95_lower": None,
            "wilson95_upper": None,
        }
    rate = numerator / denominator
    z2 = WILSON_Z_95 * WILSON_Z_95
    scale = 1.0 + z2 / denominator
    center = (rate + z2 / (2.0 * denominator)) / scale
    radius = (
        WILSON_Z_95
        * math.sqrt(
            rate * (1.0 - rate) / denominator
            + z2 / (4.0 * denominator * denominator)
        )
        / scale
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "wilson95_lower": max(0.0, center - radius),
        "wilson95_upper": min(1.0, center + radius),
    }


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


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{location} must be an object")
    return value


def _pick(value: Mapping[str, Any], names: Sequence[str], location: str) -> Any:
    found = [name for name in names if name in value]
    require(found, f"{location} is missing ({'/'.join(names)})")
    first = value[found[0]]
    for name in found[1:]:
        require(value[name] == first, f"{location} aliases disagree")
    return first


def _inside(root: Path, relative: Any, location: str) -> Path:
    require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute(),
        f"{location} must be a non-empty relative path",
    )
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"{location} escapes shard root") from error
    return path


def _verify_file_spec(root: Path, spec: Any, location: str) -> Path:
    item = _mapping(spec, location)
    path = _inside(root, item.get("path"), f"{location}.path")
    require(path.is_file(), f"{location} file is missing: {path}")
    expected = item.get("sha256")
    require(_is_sha256(expected), f"{location}.sha256 is invalid")
    require(sha256_file(path) == expected.lower(), f"{location} raw SHA256 differs")
    if "size_bytes" in item:
        require(
            type(item["size_bytes"]) is int
            and item["size_bytes"] == path.stat().st_size,
            f"{location} size differs",
        )
    return path


def _load_binary_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = np.asarray(Image.open(path).convert("L"))
    require(value.ndim == 2, f"mask must be rank two: {path}")
    if value.dtype == np.bool_:
        return np.asarray(value, dtype=bool)
    unique = set(np.unique(value).tolist())
    require(
        unique.issubset({0, 1, 255}),
        f"mask contains values outside 0/1/255: {path}",
    )
    return np.asarray(value != 0, dtype=bool)


def _verified_mask(root: Path, spec: Any, location: str) -> np.ndarray:
    item = _mapping(spec, location)
    path = _verify_file_spec(root, item, location)
    mask = _load_binary_mask(path)
    expected = item.get("mask_sha256")
    require(_is_sha256(expected), f"{location}.mask_sha256 is invalid")
    require(mask_sha256(mask) == expected.lower(), f"{location} semantic SHA256 differs")
    require(
        isinstance(item.get("shape"), list)
        and item["shape"] == list(mask.shape),
        f"{location} shape differs",
    )
    return mask


@dataclass(frozen=True)
class StateArtifact:
    masks: dict[str, np.ndarray]
    mask_hashes: dict[str, str]
    identity: str
    receipt: dict[str, Any]


def _load_state(root: Path, value: Any, location: str) -> StateArtifact:
    item = _mapping(value, location)
    container = item.get("masks", item)
    container = _mapping(container, f"{location}.masks")
    aliases = {
        "t1": ("t1", "t1_mask"),
        "t2": ("t2", "t2_mask"),
        "change": ("change", "change_mask"),
    }
    masks: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name, keys in aliases.items():
        spec = _pick(container, keys, f"{location}.{name}_mask")
        masks[name] = _verified_mask(root, spec, f"{location}.{name}_mask")
        hashes[name] = mask_sha256(masks[name])
    shape = masks["change"].shape
    require(
        masks["t1"].shape == shape and masks["t2"].shape == shape,
        f"{location} mask shapes differ",
    )
    identity = state_sha256(hashes)
    expected = item.get("state_sha256", item.get("state_hash"))
    if expected is not None:
        require(_is_sha256(expected), f"{location}.state_sha256 is invalid")
        require(expected.lower() == identity, f"{location} state SHA256 differs")
    return StateArtifact(
        masks=masks,
        mask_hashes=hashes,
        identity=identity,
        receipt=copy.deepcopy(dict(item)),
    )


def _same_state(left: StateArtifact, right: StateArtifact, location: str) -> None:
    require(left.identity == right.identity, f"{location} state identity differs")
    for name in ("t1", "t2", "change"):
        require(
            np.array_equal(left.masks[name], right.masks[name]),
            f"{location} {name} mask differs",
        )


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Independently reproduce production's deterministic 8-connectivity."""

    value = np.asarray(mask, dtype=bool)
    require(value.ndim == 2, "matching mask must be rank two")
    height, width = value.shape
    visited = np.zeros_like(value, dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(value)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and value[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros_like(value, dtype=bool)
        ys, xs = zip(*pixels)
        component[np.asarray(ys), np.asarray(xs)] = True
        components.append(component)
    return tuple(components)


def _matching_parameters(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    mode = raw.get("mode", "overlap_presence")
    threshold = raw.get("overlap_threshold", 0.25)
    t12_minimum = raw.get("t12_min_instance_area", 0)
    cd_minimum = raw.get("cd_min_instance_area", 0)
    require(
        mode in {"overlap_presence", "greedy_one_to_one"},
        "matching mode is invalid",
    )
    require(
        isinstance(threshold, (int, float)) and 0 <= threshold <= 1,
        "matching overlap threshold is invalid",
    )
    require(
        type(t12_minimum) is int
        and type(cd_minimum) is int
        and t12_minimum >= 0
        and cd_minimum >= 0,
        "matching area threshold is invalid",
    )
    return {
        "mode": mode,
        "overlap_threshold": float(threshold),
        "t12_min_instance_area": t12_minimum,
        "cd_min_instance_area": cd_minimum,
    }


def _rebuild_change_mask(
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    matching: Mapping[str, Any] | None,
) -> np.ndarray:
    """Recompute MaskPairProcessor output without trusting runner evidence."""

    parameters = _matching_parameters(matching)
    t1_instances = _connected_components(t1_mask)
    t2_instances = _connected_components(t2_mask)
    candidates: list[tuple[int, int, float, float]] = []
    for left, mask1 in enumerate(t1_instances):
        area1 = int(mask1.sum())
        for right, mask2 in enumerate(t2_instances):
            intersection = int(np.logical_and(mask1, mask2).sum())
            if intersection:
                candidates.append(
                    (left, right, intersection / area1, intersection / int(mask2.sum()))
                )
    threshold = parameters["overlap_threshold"]
    if parameters["mode"] == "overlap_presence":
        matched_t1 = {
            left
            for left, _, t1_coverage, _ in candidates
            if t1_coverage >= threshold
        }
        matched_t2 = {
            right
            for _, right, _, t2_coverage in candidates
            if t2_coverage >= threshold
        }
    else:
        ranked = sorted(
            (
                (min(t1_coverage, t2_coverage), left, right)
                for left, right, t1_coverage, t2_coverage in candidates
                if min(t1_coverage, t2_coverage) >= threshold
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
        if (
            index not in matched_t1
            and int(instance.sum()) >= parameters["t12_min_instance_area"]
        ):
            change |= instance
    for index, instance in enumerate(t2_instances):
        if (
            index not in matched_t2
            and int(instance.sum()) >= parameters["t12_min_instance_area"]
        ):
            change |= instance
    if parameters["cd_min_instance_area"]:
        filtered = np.zeros_like(change)
        for instance in _connected_components(change):
            if int(instance.sum()) >= parameters["cd_min_instance_area"]:
                filtered |= instance
        change = filtered
    return change


def _require_rebuilt_change(
    state: StateArtifact,
    matching: Mapping[str, Any] | None,
    location: str,
) -> None:
    rebuilt = _rebuild_change_mask(state.masks["t1"], state.masks["t2"], matching)
    require(
        np.array_equal(rebuilt, state.masks["change"]),
        f"{location} change mask differs from production MaskPairProcessor rebuild",
    )


def _qwen_is_disabled(value: Mapping[str, Any], location: str) -> None:
    audit = value.get("no_qwen_audit")
    containers = [value]
    if isinstance(audit, Mapping):
        containers.append(audit)
    claims = [
        container["qwen_invoked"]
        for container in containers
        if "qwen_invoked" in container
    ]
    require(claims and all(claim is False for claim in claims), f"{location} qwen_invoked must be false")
    for container in containers:
        for key in ("qwen_call_count", "qwen_calls", "model_load_count"):
            if key in container:
                require(container[key] == 0, f"{location}.{key} must be zero")


def _one_action(step: Mapping[str, Any], location: str) -> dict[str, Any]:
    if "actions" in step:
        actions = step["actions"]
        require(
            isinstance(actions, list) and len(actions) == 1,
            f"{location} must contain exactly one action",
        )
        action = actions[0]
    else:
        action = step.get("action")
    action = _mapping(action, f"{location}.action")
    action_name = action.get("action", action.get("type"))
    require(
        action_name in {"positive_point", "negative_point", "box"},
        f"{location} action is not production-primary",
    )
    require(action.get("target_view") in {"t1", "t2"}, f"{location} target_view invalid")
    return dict(action)


def _tool_passed(step: Mapping[str, Any], location: str) -> bool:
    if "tool_success" in step:
        require(isinstance(step["tool_success"], bool), f"{location}.tool_success invalid")
        return step["tool_success"]
    tool = step.get("tool_evidence", step.get("tool_result"))
    tool = _mapping(tool, f"{location}.tool_evidence")
    if isinstance(tool.get("success"), bool):
        return bool(tool["success"])
    report = tool.get("worker_report", tool)
    report = _mapping(report, f"{location}.worker_report")
    return report.get("status") == "success"


def _gate_passed(step: Mapping[str, Any], location: str) -> bool:
    gate = _mapping(
        step.get("hard_gate", step.get("hard_gate_audit")),
        f"{location}.hard_gate",
    )
    passed = gate.get("passed", gate.get("gate_passed"))
    require(isinstance(passed, bool), f"{location}.hard_gate.passed invalid")
    if "rejected" in gate:
        require(gate["rejected"] is (not passed), f"{location} gate flags disagree")
    return bool(passed)


def _semantic_passed(step: Mapping[str, Any], location: str) -> bool:
    semantic = step.get(
        "rgb_semantic",
        step.get("rgb_semantic_check", step.get("semantic_check")),
    )
    semantic = _mapping(semantic, f"{location}.rgb_semantic")
    passed = semantic.get("valid", semantic.get("passed"))
    require(isinstance(passed, bool), f"{location}.rgb_semantic.valid invalid")
    if "evidence_sufficient" in semantic:
        passed = passed and semantic["evidence_sufficient"] is True
    if "harm" in semantic:
        passed = passed and semantic["harm"] is False
    return bool(passed)


def _budget_hit(value: Mapping[str, Any]) -> bool:
    if "budget_hit" in value:
        require(isinstance(value["budget_hit"], bool), "budget_hit must be boolean")
        return bool(value["budget_hit"])
    budget = value.get("budget")
    if isinstance(budget, Mapping) and "hit" in budget:
        require(isinstance(budget["hit"], bool), "budget.hit must be boolean")
        return bool(budget["hit"])
    return False


def _validate_initializer_receipt(initial: Mapping[str, Any], location: str) -> None:
    receipt = _mapping(initial.get("initializer_receipt"), f"{location}.initializer_receipt")
    require(
        receipt.get("worker_mode") == "initialize",
        f"{location} initializer is not the SAM3 initialize worker",
    )
    report = _mapping(
        receipt.get("worker_report"), f"{location}.initializer_receipt.worker_report"
    )
    require(
        report.get("status") == "success"
        and report.get("mode") == "initialize"
        and report.get("tool") == "sam3"
        and report.get("operation") == "fresh_dual_view_text_initialization",
        f"{location} initializer receipt is not a successful fresh dual-view SAM3 run",
    )


def _tool_receipt_subset(execution: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: execution[key]
        for key in (
            "tool",
            "tool_input",
            "tool_result",
            "locality",
            "point_authorization",
            "box_authorization",
        )
        if key in execution
    }


def _validate_tool_receipt(
    receipt_value: Any,
    action: Mapping[str, Any],
    region_id: Any,
    location: str,
) -> Mapping[str, Any]:
    receipt = _mapping(receipt_value, f"{location}.tool_receipt")
    action_name = action.get("action", action.get("type"))
    point_action = action_name in {"positive_point", "negative_point"}
    expected_tool = "simpleclick" if point_action else "sam3"
    expected_mode = "point" if point_action else "box"
    require(receipt.get("tool") == expected_tool, f"{location} production tool differs")
    tool_input = _mapping(receipt.get("tool_input"), f"{location}.tool_input")
    require(
        tool_input.get("authorized_region_id") == region_id,
        f"{location} tool authorization region differs",
    )
    if point_action:
        require(
            tool_input.get("coordinate") == action.get("coordinate")
            and tool_input.get("is_positive") is (action_name == "positive_point")
            and tool_input.get("accepted_click_history") == [],
            f"{location} point tool input differs from action",
        )
        authorization = _mapping(
            receipt.get("point_authorization"), f"{location}.point_authorization"
        )
        require(
            authorization.get("region_id") == region_id,
            f"{location} point authorization differs",
        )
    else:
        authorization = _mapping(
            receipt.get("box_authorization"), f"{location}.box_authorization"
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
        f"{location} worker receipt is not successful",
    )
    return receipt


def _validate_session_rebase(
    execution: Mapping[str, Any],
    action: Mapping[str, Any],
    accepted_state: StateArtifact,
    location: str,
) -> None:
    update = _mapping(
        execution.get("point_session_update"), f"{location}.point_session_update"
    )
    target_view = action.get("target_view")
    require(
        update.get("policy") == "rebase_to_accepted_live_mask"
        and update.get("target_view") == target_view
        and update.get("base_mask_pixels")
        == int(accepted_state.masks[str(target_view)].sum()),
        f"{location} accepted-mask session rebase differs",
    )


def _four_connected_component_sizes(mask: np.ndarray) -> list[int]:
    """Independently match Executor locality's four-neighbour components."""

    remaining = np.asarray(mask, dtype=bool).copy()
    height, width = remaining.shape
    sizes: list[int] = []
    while remaining.any():
        start_y, start_x = np.argwhere(remaining)[0]
        remaining[start_y, start_x] = False
        stack = [(int(start_x), int(start_y))]
        size = 0
        while stack:
            x, y = stack.pop()
            size += 1
            for next_x, next_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and remaining[next_y, next_x]
                ):
                    remaining[next_y, next_x] = False
                    stack.append((next_x, next_y))
        sizes.append(size)
    return sizes


def _recompute_hard_gate_evidence(
    source: StateArtifact,
    candidate: StateArtifact,
    action: Mapping[str, Any],
    proposal: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    """Recompute production hard-gate quantities from verified masks and geometry."""

    target_view = action.get("target_view")
    require(target_view in {"t1", "t2"}, f"{location} hard-gate target view invalid")
    before = source.masks[str(target_view)]
    after = candidate.masks[str(target_view)]
    require(before.shape == after.shape, f"{location} hard-gate target shapes differ")
    height, width = before.shape
    proposal_box = proposal.get("box_pixels")
    require(
        isinstance(proposal_box, list)
        and len(proposal_box) == 4
        and all(type(value) is int for value in proposal_box),
        f"{location} hard-gate proposal box invalid",
    )
    box_x1, box_y1, box_x2, box_y2 = proposal_box
    require(
        0 <= box_x1 <= box_x2 < width and 0 <= box_y1 <= box_y2 < height,
        f"{location} hard-gate proposal box out of frame",
    )
    action_name = action.get("action")
    if action_name in {"positive_point", "negative_point"}:
        coordinate = action.get("coordinate")
        require(
            isinstance(coordinate, list)
            and len(coordinate) == 2
            and all(type(value) is int for value in coordinate),
            f"{location} hard-gate point coordinate invalid",
        )
        point_x, point_y = coordinate
        require(
            box_x1 <= point_x <= box_x2 and box_y1 <= point_y <= box_y2,
            f"{location} hard-gate point is outside proposal box",
        )
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
        require(action_name == "box", f"{location} hard-gate action invalid")
        require(
            action.get("box") == proposal_box,
            f"{location} hard-gate box differs from proposal geometry",
        )
        roi = list(proposal_box)
        composition_mode = "replace_box_roi_only"

    changed = np.logical_xor(before, after)
    inside = np.zeros_like(changed)
    x1, y1, x2, y2 = roi
    inside[y1 : y2 + 1, x1 : x2 + 1] = True
    changed_pixels = int(changed.sum())
    outside_pixels = int(np.logical_and(changed, ~inside).sum())
    inside_pixels = changed_pixels - outside_pixels
    before_components = _four_connected_component_sizes(before)
    after_components = _four_connected_component_sizes(after)
    changed_components = _four_connected_component_sizes(changed)
    locality = {
        "composition_mode": composition_mode,
        "roi_xyxy": roi,
        "changed_pixels": changed_pixels,
        "inside_roi_pixels": inside_pixels,
        "outside_roi_pixels": outside_pixels,
        "local_changed_pixels": inside_pixels,
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
    if candidate_area_delta > 0.25:
        reasons.append("mask_area_delta_exceeded")
    if locality["outside_roi_ratio"] > 0.1:
        reasons.append("locality_outside_roi_exceeded")
    if locality["target_mask_change_ratio"] > 0.25:
        reasons.append("target_mask_change_exceeded")
    if abs(locality["component_count_delta"]) > 4:
        reasons.append("component_count_delta_exceeded")

    point_authorization: dict[str, Any] | None = None
    if action_name in {"positive_point", "negative_point"}:
        component_area = proposal.get("component_area")
        require(
            type(component_area) is int and component_area > 0,
            f"{location} hard-gate proposal component area invalid",
        )
        proposal_change_ratio = changed_pixels / component_area
        proposal_ratio_exceeded = proposal_change_ratio > 0.25
        absolute_pixels_exceeded = changed_pixels > 256
        image_ratio_exceeded = locality["target_mask_change_ratio"] > 0.004
        hard_reject = bool(
            proposal_ratio_exceeded
            and absolute_pixels_exceeded
            and image_ratio_exceeded
        )
        change_gate = {
            "enabled": True,
            "proposal_change_ratio": proposal_change_ratio,
            "max_proposal_change_ratio": 0.25,
            "changed_pixels": changed_pixels,
            "max_changed_pixels": 256,
            "image_change_ratio": locality["target_mask_change_ratio"],
            "max_image_change_ratio": 0.004,
            "proposal_ratio_exceeded": proposal_ratio_exceeded,
            "absolute_pixels_exceeded": absolute_pixels_exceeded,
            "image_ratio_exceeded": image_ratio_exceeded,
            "hard_reject": hard_reject,
            "policy": "reject_only_when_ratio_and_absolute_and_image_limits_exceed",
        }
        if hard_reject:
            reasons.append("point_proposal_combined_change_gate_exceeded")
        point_authorization = {
            "proposal_box_xyxy": list(proposal_box),
            "proposal_component_area": component_area,
            "proposal_change_ratio": proposal_change_ratio,
            "effective_roi_xyxy": roi,
            "change_gate": change_gate,
        }
    return {
        "previous_area_ratio": previous_area_ratio,
        "candidate_area_ratio": candidate_area_ratio,
        "candidate_area_delta": candidate_area_delta,
        "locality": locality,
        "point_authorization": point_authorization,
        "rejection_reasons": reasons,
    }


def _require_recomputed_mapping(
    observed_value: Any, expected: Mapping[str, Any], location: str
) -> None:
    observed = _mapping(observed_value, location)
    for key, expected_value in expected.items():
        require(key in observed, f"{location}.{key} is missing")
        actual = observed[key]
        if isinstance(expected_value, float):
            require(
                isinstance(actual, (int, float))
                and math.isclose(
                    float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
                ),
                f"{location}.{key} differs from independent recomputation",
            )
        else:
            require(
                actual == expected_value,
                f"{location}.{key} differs from independent recomputation",
            )


def _validate_hard_gate(
    gate_value: Any,
    execution_value: Any,
    location: str,
    *,
    source: StateArtifact,
    candidate: StateArtifact,
    action: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> bool:
    gate = _mapping(gate_value, f"{location}.hard_gate")
    execution = _mapping(execution_value, f"{location}.runtime_execution")
    independent = _recompute_hard_gate_evidence(
        source, candidate, action, proposal, location
    )
    reasons = independent["rejection_reasons"]
    rejected = bool(reasons)
    expected = {
        "passed": not rejected,
        "rejected": rejected,
        "rejection_reasons": reasons,
        "verifier_skipped_by_hard_gate": rejected,
    }
    require(
        dict(gate) == expected,
        f"{location} hard-gate receipt differs from independent recomputation",
    )
    require(
        execution.get("verifier_skipped_by_hard_gate") is rejected
        and execution.get("candidate_rejection_reasons") == reasons
        and execution.get("candidate_accepted") is (not rejected),
        f"{location} runtime hard-gate decision differs from independent recomputation",
    )
    _require_recomputed_mapping(
        execution,
        {
            "previous_area_ratio": independent["previous_area_ratio"],
            "candidate_area_ratio": independent["candidate_area_ratio"],
            "candidate_area_delta": independent["candidate_area_delta"],
        },
        f"{location}.runtime_execution",
    )
    _require_recomputed_mapping(
        execution.get("locality"),
        independent["locality"],
        f"{location}.runtime_execution.locality",
    )
    point_expected = independent["point_authorization"]
    if point_expected is not None:
        authorization = _mapping(
            execution.get("point_authorization"),
            f"{location}.runtime_execution.point_authorization",
        )
        _require_recomputed_mapping(
            authorization,
            {
                key: point_expected[key]
                for key in (
                    "proposal_box_xyxy",
                    "proposal_component_area",
                    "proposal_change_ratio",
                    "effective_roi_xyxy",
                )
            },
            f"{location}.runtime_execution.point_authorization",
        )
        _require_recomputed_mapping(
            authorization.get("change_gate"),
            point_expected["change_gate"],
            f"{location}.runtime_execution.point_authorization.change_gate",
        )
        tool_input = _mapping(
            execution.get("tool_input"), f"{location}.runtime_execution.tool_input"
        )
        _require_recomputed_mapping(
            tool_input,
            {
                "proposal_box_xyxy": point_expected["proposal_box_xyxy"],
                "proposal_component_area": point_expected[
                    "proposal_component_area"
                ],
                "effective_roi_xyxy": point_expected["effective_roi_xyxy"],
            },
            f"{location}.runtime_execution.tool_input",
        )
    if rejected:
        verifier_evidence = _mapping(
            execution.get("verifier_evidence"),
            f"{location}.runtime_execution.verifier_evidence",
        )
        require(
            verifier_evidence.get("type") == "runtime_hard_gate_precheck"
            and verifier_evidence.get("verifier_skipped") is True
            and verifier_evidence.get("rejection_reasons") == reasons
            and verifier_evidence.get("gt_available") is False,
            f"{location} hard-gate verifier evidence differs",
        )
    return not rejected


def _require_metrics(
    recorded_value: Any,
    actual: Mapping[str, Any],
    shape: tuple[int, int],
    location: str,
) -> None:
    recorded = _mapping(recorded_value, location)
    expected = {
        key: actual[key]
        for key in ("pixel_error", "iou", "f1", "precision", "recall", "tp", "fp", "fn")
    }
    expected["tn"] = shape[0] * shape[1] - int(actual["tp"]) - int(actual["fp"]) - int(actual["fn"])
    for key, expected_value in expected.items():
        require(key in recorded, f"{location}.{key} is missing")
        observed = recorded[key]
        if isinstance(expected_value, float):
            require(
                isinstance(observed, (int, float))
                and math.isclose(float(observed), expected_value, rel_tol=0.0, abs_tol=1e-12),
                f"{location}.{key} differs from recomputation",
            )
        else:
            require(observed == expected_value, f"{location}.{key} differs from recomputation")


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pixel_error_reduction": int(before["pixel_error"]) - int(after["pixel_error"]),
        "iou": float(after["iou"]) - float(before["iou"]),
        "f1": float(after["f1"]) - float(before["f1"]),
        "tp": int(after["tp"]) - int(before["tp"]),
        "fp": int(after["fp"]) - int(before["fp"]),
        "fn": int(after["fn"]) - int(before["fn"]),
    }


def _require_metric_delta(
    recorded_value: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    recorded = _mapping(recorded_value, location)
    expected = _metric_delta(before, after)
    require(set(recorded) == set(expected), f"{location} inventory differs")
    for key, expected_value in expected.items():
        observed = recorded[key]
        if isinstance(expected_value, float):
            require(
                isinstance(observed, (int, float))
                and math.isclose(float(observed), expected_value, rel_tol=0.0, abs_tol=1e-12),
                f"{location}.{key} differs from recomputation",
            )
        else:
            require(observed == expected_value, f"{location}.{key} differs from recomputation")
    return expected


def _require_per_step_gate(
    recorded_value: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    epsilon: float,
    location: str,
) -> bool:
    gate = _mapping(recorded_value, location)
    delta = _metric_delta(before, after)
    nonempty = bool(before["gt_pixels"])
    pixel_pass = int(delta["pixel_error_reduction"]) > 0
    iou_pass = float(delta["iou"]) > epsilon
    f1_pass = float(delta["f1"]) > epsilon
    fp_pass = int(after["fp"]) < int(before["fp"])
    expected = {
        "passed": bool(pixel_pass and ((iou_pass and f1_pass) if nonempty else fp_pass)),
        "pixel_error_strictly_decreased": pixel_pass,
        "nonempty_gt": nonempty,
        "iou_strictly_increased": iou_pass if nonempty else None,
        "f1_strictly_increased": f1_pass if nonempty else None,
        "fp_strictly_decreased": fp_pass if not nonempty else None,
        "comparison_epsilon": epsilon,
    }
    require(dict(gate) == expected, f"{location} differs from recomputation")
    return bool(expected["passed"])


def _expected_rgb_state(
    expected_sample: Mapping[str, Any],
    step_index: int,
    region_id: str,
    target_view: str,
) -> str:
    semantic = _mapping(expected_sample.get("semantic"), "sample.semantic")
    default = semantic.get(target_view, semantic.get(f"{target_view}_state"))
    by_region = semantic.get("by_region", {})
    by_step = semantic.get("by_step", {})
    selected: Any = None
    if isinstance(by_step, Mapping):
        step = by_step.get(str(step_index), by_step.get(step_index))
        if isinstance(step, Mapping):
            selected = step.get(region_id)
    if selected is None and isinstance(by_region, Mapping):
        selected = by_region.get(region_id)
    if isinstance(selected, Mapping):
        default = selected.get(target_view, selected.get(f"{target_view}_state", default))
    require(
        default in {"building", "background", "mixed", "uncertain"},
        "sample RGB semantic state is invalid",
    )
    return str(default)


def _validate_rgb_gate(
    gate_value: Any,
    action: Mapping[str, Any],
    expected_sample: Mapping[str, Any],
    step_index: int,
    region_id: str,
    location: str,
) -> bool:
    gate = _mapping(gate_value, f"{location}.rgb_semantic_gate")
    action_name = action.get("action", action.get("type"))
    target_view = action.get("target_view")
    observed = _expected_rgb_state(
        expected_sample, step_index, region_id, str(target_view)
    )
    required = "background" if action_name == "negative_point" else "building"
    expected = {
        "passed": observed == required,
        "target_view": target_view,
        "action": action_name,
        "observed_rgb_state": observed,
        "required_rgb_state": required,
        "policy": "negative_requires_background_positive_or_box_requires_building",
    }
    require(dict(gate) == expected, f"{location} RGB-semantic receipt differs from config")
    return bool(expected["passed"])


def _verify_json_identity(root: Path, spec: Any, location: str) -> dict[str, Any]:
    item = _mapping(spec, location)
    path = _verify_file_spec(root, item, location)
    payload = read_json(path, location)
    expected = item.get("canonical_sha256")
    require(_is_sha256(expected), f"{location}.canonical_sha256 is invalid")
    require(
        canonical_sha256(payload) == expected.lower(),
        f"{location} canonical SHA256 differs",
    )
    return payload


def _forbid_gt_fields(value: Any, location: str) -> None:
    forbidden = {
        "gt",
        "gt_mask",
        "label",
        "labels",
        "ground_truth",
        "metric",
        "metrics",
        "iou",
        "f1",
        "pixel_error",
        "error_mask",
        "score",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            require(
                normalized not in forbidden
                and not normalized.startswith("gt_")
                and not normalized.endswith("_gt")
                and "ground_truth" not in normalized
                and "label" not in normalized,
                f"{location} contains forbidden GT-derived field {key!r}",
            )
            _forbid_gt_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_gt_fields(child, f"{location}[{index}]")


@dataclass(frozen=True)
class EvidenceStep:
    action: dict[str, Any]
    candidate: StateArtifact
    canonical: StateArtifact
    replay: StateArtifact


def _rebuild_useful_action_inventory(
    episode_id: str,
    step_index: int,
    proposals: Sequence[Any],
    location: str,
) -> list[dict[str, Any]]:
    """Independently reproduce the frozen useful executor policy."""

    inventory: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for ordinal, raw_proposal in enumerate(proposals):
        proposal = _mapping(raw_proposal, f"{location}.live_proposals[{ordinal}]")
        region_id = proposal.get("region_id")
        audit_kind = proposal.get("audit_kind")
        require(
            isinstance(region_id, str)
            and region_id
            and region_id not in seen_regions,
            f"{location} live proposal region IDs are invalid",
        )
        require(
            audit_kind in KNOWN_AUDIT_KINDS,
            f"{location} proposal {region_id} has unsupported audit_kind {audit_kind!r}",
        )
        seen_regions.add(region_id)
        planned: list[tuple[str, dict[str, Any]]] = []
        if audit_kind in FULL_LIVE_AUDIT_KINDS:
            seed = proposal.get("component_seed_pixels")
            box = proposal.get("box_pixels")
            require(
                isinstance(seed, list)
                and len(seed) == 2
                and all(type(value) is int for value in seed),
                f"{location} proposal {region_id} seed is invalid",
            )
            require(
                isinstance(box, list)
                and len(box) == 4
                and all(type(value) is int for value in box),
                f"{location} proposal {region_id} box is invalid",
            )
            flags: dict[str, bool] = {}
            for view in ("t1", "t2"):
                value = proposal.get(f"component_seed_{view}_mask_white")
                require(
                    type(value) is bool,
                    f"{location} proposal {region_id} {view} seed flag is invalid",
                )
                flags[view] = value
            unique_views = sum(flags.values()) == 1
            for view in ("t1", "t2"):
                if flags[view] and audit_kind != "scan":
                    planned.append(
                        (
                            "runtime_planner_primary"
                            if unique_views and audit_kind == "present"
                            else "executor_reachable_extended_negative",
                            {
                                "target_view": view,
                                "action": "negative_point",
                                "coordinate_space": "pixel",
                                "coordinate": list(seed),
                            },
                        )
                    )
                planned.append(
                    (
                        "executor_reachable_scan_point"
                        if audit_kind == "scan"
                        else "runtime_planner_primary"
                        if audit_kind == "missing"
                        and unique_views
                        and not flags[view]
                        else "executor_reachable_refinement"
                        if flags[view]
                        else "executor_reachable_complement"
                        if audit_kind == "present"
                        else "executor_reachable_extended_positive",
                        {
                            "target_view": view,
                            "action": "positive_point",
                            "coordinate_space": "pixel",
                            "coordinate": list(seed),
                        },
                    )
                )
                planned.append(
                    (
                        "runtime_planner_primary"
                        if audit_kind == "scan"
                        else "executor_reachable_box",
                        {
                            "target_view": view,
                            "action": "box",
                            "coordinate_space": "pixel",
                            "box": list(box),
                        },
                    )
                )
        for reachability_tier, action in planned:
            candidate_id = "--".join(
                (
                    episode_id,
                    f"step-{step_index:03d}",
                    region_id,
                    str(action["action"]),
                    str(action["target_view"]),
                )
            )
            inventory.append(
                {
                    "candidate_id": candidate_id,
                    "region_id": region_id,
                    "reachability_tier": reachability_tier,
                    "action": action,
                    "proposal": copy.deepcopy(dict(proposal)),
                }
            )
    return inventory


def _validate_evidence_chain(
    payload: Mapping[str, Any],
    *,
    root: Path,
    episode_id: str,
    expected_sample: Mapping[str, Any],
    ground_truth: np.ndarray,
    initial: StateArtifact,
    steps: Sequence[Any],
    epsilon: float,
    endpoint_minimum_iou: float,
    endpoint_minimum_f1: float,
    endpoint_empty_gt_max_fp: int,
    matching: Mapping[str, Any] | None,
) -> list[EvidenceStep]:
    location = f"episode {episode_id}"
    initial_value = _mapping(payload.get("initial_state"), f"{location}.initial_state")
    _validate_initializer_receipt(initial_value, f"{location}.initial_state")
    _require_rebuilt_change(initial, matching, f"{location}.initial_state")

    audit = _mapping(payload.get("gt_access_audit"), f"{location}.gt_access_audit")
    require(audit.get("access_count") == 1, f"{location} GT access count must equal one")
    require(
        audit.get("first_access_phase")
        == "after_initial_live_action_catalog_freeze_before_offline_top4_shortlist",
        f"{location} first GT access phase differs",
    )
    for key in (
        "label_passed_to_runtime_factory",
        "label_passed_to_environment",
        "label_passed_to_initializer_or_tools",
    ):
        require(audit.get(key) is False, f"{location} {key} must be false")
    require(
        audit.get("runtime_factory_sample_type") == "RuntimeEpisode_GT_free"
        and audit.get("offline_label_loader_only") is True,
        f"{location} GT-free runtime audit differs",
    )

    rounds = payload.get("selection_rounds")
    require(
        isinstance(rounds, list) and 1 <= len(rounds) <= MAX_STEPS,
        f"{location}.selection_rounds must contain one to three rounds",
    )
    audit_freezes = audit.get("catalog_freezes")
    require(
        isinstance(audit_freezes, list) and len(audit_freezes) == len(rounds),
        f"{location} catalog-freeze audit coverage differs",
    )
    require(
        audit.get("catalog_freeze_before_first_access") == audit_freezes[0],
        f"{location} first catalog-freeze identity differs",
    )

    current = initial
    step_cursor = 0
    selected_steps: list[EvidenceStep] = []
    seen_candidate_ids: set[str] = set()
    for round_position, raw_round in enumerate(rounds, start=1):
        round_location = f"{location} selection round {round_position}"
        round_record = _mapping(raw_round, round_location)
        require(
            round_record.get("step_index") == round_position,
            f"{round_location} index is not contiguous",
        )
        require(
            round_record.get("source_state_sha256") == current.identity
            and round_record.get("source_state_mask_sha256") == current.mask_hashes,
            f"{round_location} source state differs",
        )
        source_metrics = compute_metrics(current.masks["change"], ground_truth)
        _require_metrics(
            round_record.get("source_metrics"),
            source_metrics,
            ground_truth.shape,
            f"{round_location}.source_metrics",
        )
        source_endpoint = _mapping(
            round_record.get("source_endpoint_gate"),
            f"{round_location}.source_endpoint_gate",
        )
        expected_source_endpoint = endpoint_gate(
            source_metrics,
            minimum_iou=endpoint_minimum_iou,
            minimum_f1=endpoint_minimum_f1,
            empty_gt_max_fp=endpoint_empty_gt_max_fp,
            epsilon=epsilon,
        )
        require(
            dict(source_endpoint) == expected_source_endpoint,
            f"{round_location} source endpoint gate differs",
        )

        freeze_identity = round_record.get("gt_free_catalog_freeze")
        require(
            freeze_identity == audit_freezes[round_position - 1],
            f"{round_location} catalog-freeze identity differs from GT audit",
        )
        catalog = _verify_json_identity(
            root, freeze_identity, f"{round_location}.gt_free_catalog_freeze"
        )
        require(catalog.get("schema_id") == CATALOG_SCHEMA, f"{round_location} catalog schema differs")
        require(
            catalog.get("episode_id") == episode_id
            and catalog.get("step_index") == round_position,
            f"{round_location} catalog identity differs",
        )
        require(
            catalog.get("source_state_sha256") == current.identity
            and catalog.get("source_state_mask_sha256") == current.mask_hashes,
            f"{round_location} frozen source state differs",
        )
        frozen_proposals = catalog.get("live_proposals")
        frozen_actions = catalog.get("actions")
        require(
            isinstance(frozen_proposals, list) and isinstance(frozen_actions, list),
            f"{round_location} frozen catalog lists are invalid",
        )
        require(
            catalog.get("live_proposals_canonical_sha256")
            == canonical_sha256(frozen_proposals)
            and catalog.get("action_catalog_canonical_sha256")
            == canonical_sha256(frozen_actions)
            and catalog.get("action_count") == len(frozen_actions),
            f"{round_location} frozen catalog hashes/count differ",
        )
        rebuilt_actions = _rebuild_useful_action_inventory(
            episode_id,
            round_position,
            frozen_proposals,
            round_location,
        )
        require(
            frozen_actions == rebuilt_actions,
            f"{round_location} frozen action inventory differs from useful executor policy",
        )
        require(
            catalog.get("gt_access_count_before") == (0 if round_position == 1 else 1)
            and catalog.get("contains_gt_derived_fields") is False,
            f"{round_location} catalog was not frozen before GT access",
        )
        _qwen_is_disabled(catalog, f"{round_location} frozen catalog")
        _forbid_gt_fields(frozen_proposals, f"{round_location}.live_proposals")
        _forbid_gt_fields(frozen_actions, f"{round_location}.actions")
        require(
            round_record.get("live_proposals") == frozen_proposals
            and round_record.get("live_proposals_canonical_sha256")
            == canonical_sha256(frozen_proposals),
            f"{round_location} live proposals differ from frozen catalog",
        )

        action_catalog = round_record.get("action_catalog")
        require(isinstance(action_catalog, list), f"{round_location}.action_catalog invalid")
        require(
            round_record.get("action_catalog_count") == len(action_catalog)
            and len(action_catalog) == len(frozen_actions),
            f"{round_location} action-catalog count differs",
        )
        catalog_by_id: dict[str, Mapping[str, Any]] = {}
        independently_eligible: list[tuple[int, str]] = []
        current_error_mask = np.logical_xor(
            current.masks["change"], ground_truth
        )
        for ordinal, (raw_action, raw_frozen) in enumerate(
            zip(action_catalog, frozen_actions)
        ):
            action_item = _mapping(raw_action, f"{round_location}.action_catalog[{ordinal}]")
            frozen_item = _mapping(raw_frozen, f"{round_location}.actions[{ordinal}]")
            candidate_id = action_item.get("candidate_id")
            require(
                isinstance(candidate_id, str)
                and candidate_id
                and candidate_id not in catalog_by_id,
                f"{round_location} action candidate IDs are invalid",
            )
            for key in ("candidate_id", "region_id", "reachability_tier", "proposal", "action"):
                require(
                    action_item.get(key) == frozen_item.get(key),
                    f"{round_location} action {candidate_id} differs from frozen catalog",
                )
            require(
                type(action_item.get("selected_for_execution")) is bool,
                f"{round_location} action selection flag is invalid",
            )
            action = _one_action(
                action_item, f"{round_location}.action_catalog[{ordinal}]"
            )
            region_id = action_item.get("region_id")
            require(
                isinstance(region_id, str) and region_id,
                f"{round_location} action region_id is invalid",
            )
            rgb_eligible = _validate_rgb_gate(
                action_item.get("rgb_semantic_gate"),
                action,
                expected_sample,
                round_position,
                region_id,
                f"{round_location}.action_catalog[{ordinal}]",
            )
            proposal = _mapping(
                action_item.get("proposal"),
                f"{round_location}.action_catalog[{ordinal}].proposal",
            )
            box = proposal.get("box_pixels")
            require(
                isinstance(box, list)
                and len(box) == 4
                and all(type(value) is int for value in box),
                f"{round_location} action proposal box is invalid",
            )
            x1, y1, x2, y2 = box
            height, width = current_error_mask.shape
            require(
                0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height,
                f"{round_location} action proposal box is out of frame",
            )
            box_error = int(current_error_mask[y1 : y2 + 1, x1 : x2 + 1].sum())
            require(
                action_item.get("proposal_box_current_change_error_pixels")
                == box_error,
                f"{round_location} proposal-box error pixels differ",
            )
            if rgb_eligible:
                independently_eligible.append((box_error, str(candidate_id)))
            catalog_by_id[candidate_id] = action_item

        execution_shortlist = round_record.get("execution_shortlist")
        require(
            isinstance(execution_shortlist, list)
            and all(isinstance(item, str) for item in execution_shortlist)
            and len(execution_shortlist) == len(set(execution_shortlist)),
            f"{round_location} execution shortlist is invalid",
        )
        require(
            len(execution_shortlist) <= 4,
            f"{round_location} execution shortlist exceeds frozen top-4 limit",
        )
        independently_eligible.sort(key=lambda item: (-item[0], item[1]))
        expected_shortlist = [
            candidate_id for _, candidate_id in independently_eligible[:4]
        ]
        require(
            execution_shortlist == expected_shortlist,
            f"{round_location} execution shortlist order differs from independent top-4",
        )
        require(
            {
                candidate_id
                for candidate_id, item in catalog_by_id.items()
                if item.get("selected_for_execution") is True
            }
            == set(expected_shortlist),
            f"{round_location} action selection flags differ from independent top-4",
        )
        candidates = round_record.get("candidates")
        require(isinstance(candidates, list), f"{round_location}.candidates invalid")
        batch_complete = round_record.get(
            "candidate_batch_complete_before_final_gt_scoring"
        )
        candidate_ids = [
            _mapping(item, f"{round_location}.candidate").get("candidate_id")
            for item in candidates
        ]
        require(
            round_record.get("candidate_count") == len(candidates),
            f"{round_location} candidate count differs",
        )
        if candidate_ids != execution_shortlist:
            require(
                not candidates
                and (
                    source_endpoint.get("passed") is True
                    or (_budget_hit(payload) and batch_complete is False)
                ),
                f"{round_location} candidate batch differs from execution shortlist",
            )
        if batch_complete is not True:
            require(
                _budget_hit(payload)
                and round_position == len(rounds)
                and not candidates,
                f"{round_location} candidate batch is incomplete",
            )

        candidates_by_id: dict[str, tuple[Mapping[str, Any], StateArtifact, bool]] = {}
        independent_delta_by_id: dict[str, dict[str, Any]] = {}
        beneficial: list[Mapping[str, Any]] = []
        for ordinal, raw_candidate in enumerate(candidates):
            candidate_location = f"{round_location} candidate {ordinal}"
            candidate = _mapping(raw_candidate, candidate_location)
            candidate_id = candidate.get("candidate_id")
            require(
                candidate_id == execution_shortlist[ordinal]
                and candidate_id not in seen_candidate_ids
                and candidate.get("ordinal") == ordinal,
                f"{candidate_location} identity/ordinal differs",
            )
            seen_candidate_ids.add(str(candidate_id))
            catalog_item = catalog_by_id[str(candidate_id)]
            for key in ("region_id", "reachability_tier", "proposal", "action"):
                require(
                    candidate.get(key) == catalog_item.get(key),
                    f"{candidate_location} differs from full action catalog",
                )
            action = _one_action(candidate, candidate_location)
            region_id = candidate.get("region_id")
            require(
                isinstance(region_id, str) and region_id,
                f"{candidate_location} region_id is invalid",
            )
            require(
                candidate.get("source_state_mask_sha256") == current.mask_hashes,
                f"{candidate_location} source-state masks differ",
            )
            candidate_state = _load_state(
                root, candidate.get("candidate_state"), f"{candidate_location}.candidate_state"
            )
            target_view = str(action["target_view"])
            other_view = "t1" if target_view == "t2" else "t2"
            require(
                np.array_equal(
                    candidate_state.masks[other_view], current.masks[other_view]
                ),
                f"{candidate_location} non-target view mask changed",
            )
            _require_rebuilt_change(candidate_state, matching, candidate_location)
            _qwen_is_disabled(candidate, candidate_location)
            receipt = _validate_tool_receipt(
                candidate.get("tool_receipt"), action, region_id, candidate_location
            )
            execution = _mapping(
                candidate.get("runtime_execution"),
                f"{candidate_location}.runtime_execution",
            )
            require(
                dict(receipt) == _tool_receipt_subset(execution),
                f"{candidate_location} tool receipt differs from runtime execution",
            )
            hard_passed = _validate_hard_gate(
                candidate.get("hard_gate"),
                execution,
                candidate_location,
                source=current,
                candidate=candidate_state,
                action=action,
                proposal=_mapping(
                    candidate.get("proposal"), f"{candidate_location}.proposal"
                ),
            )
            rgb_passed = _validate_rgb_gate(
                candidate.get("rgb_semantic_gate"),
                action,
                expected_sample,
                round_position,
                region_id,
                candidate_location,
            )
            require(rgb_passed, f"{candidate_location} was not RGB-eligible for execution")
            candidate_metrics = compute_metrics(
                candidate_state.masks["change"], ground_truth
            )
            _require_metrics(
                candidate.get("metrics"),
                candidate_metrics,
                ground_truth.shape,
                f"{candidate_location}.metrics",
            )
            delta = _require_metric_delta(
                candidate.get("metric_delta"),
                source_metrics,
                candidate_metrics,
                f"{candidate_location}.metric_delta",
            )
            independent_delta_by_id[str(candidate_id)] = delta
            metric_passed = _require_per_step_gate(
                candidate.get("per_step_metric_gate"),
                source_metrics,
                candidate_metrics,
                epsilon,
                f"{candidate_location}.per_step_metric_gate",
            )
            if hard_passed and rgb_passed and metric_passed:
                beneficial.append(candidate)
            candidates_by_id[str(candidate_id)] = (
                candidate,
                candidate_state,
                metric_passed,
            )

        beneficial.sort(
            key=lambda candidate: (
                -int(
                    independent_delta_by_id[str(candidate["candidate_id"])][
                        "pixel_error_reduction"
                    ]
                ),
                -float(
                    independent_delta_by_id[str(candidate["candidate_id"])]["iou"]
                ),
                -float(
                    independent_delta_by_id[str(candidate["candidate_id"])]["f1"]
                ),
                str(candidate["candidate_id"]),
            )
        )
        beneficial_ids = [str(item["candidate_id"]) for item in beneficial]
        require(
            round_record.get("beneficial_shortlist") == beneficial_ids,
            f"{round_location} beneficial shortlist/ranking differs",
        )
        selected_id = round_record.get("selected_candidate_id")
        replay_value = round_record.get("canonical_replay")
        if selected_id is None:
            require(replay_value is None, f"{round_location} has replay without selection")
            require(
                round_position == len(rounds) and step_cursor == len(steps),
                f"{round_location} stops before all accepted steps",
            )
            if source_endpoint.get("passed") is True:
                require(
                    not candidates and round_position == len(rounds),
                    f"{round_location} endpoint round executed unnecessary candidates",
                )
            continue

        require(
            beneficial_ids and selected_id == beneficial_ids[0],
            f"{round_location} selected candidate is not beneficial_shortlist[0]",
        )
        require(
            step_cursor < len(steps),
            f"{round_location} selection has no canonical step",
        )
        selected_record, selected_state, _ = candidates_by_id[str(selected_id)]
        replay = _mapping(replay_value, f"{round_location}.canonical_replay")
        require(
            replay.get("branch_candidate_state_mask_sha256")
            == selected_state.mask_hashes
            and replay.get("canonical_state_mask_sha256")
            == selected_state.mask_hashes
            and replay.get("all_three_mask_hashes_match") is True,
            f"{round_location} canonical replay hash receipt differs",
        )
        replay_state = _load_state(
            root, replay.get("state"), f"{round_location}.canonical_replay.state"
        )
        _same_state(selected_state, replay_state, f"{round_location} branch/replay")
        _require_rebuilt_change(replay_state, matching, f"{round_location} canonical replay")
        selected_action = _one_action(selected_record, f"{round_location} selected candidate")
        selected_region = selected_record.get("region_id")
        replay_receipt = _validate_tool_receipt(
            replay.get("tool_receipt"),
            selected_action,
            selected_region,
            f"{round_location} canonical replay",
        )
        replay_execution = _mapping(
            replay.get("runtime_execution"),
            f"{round_location}.canonical_replay.runtime_execution",
        )
        require(
            dict(replay_receipt) == _tool_receipt_subset(replay_execution),
            f"{round_location} replay receipt differs from runtime execution",
        )
        require(
            _validate_hard_gate(
                replay.get("hard_gate"),
                replay_execution,
                f"{round_location} canonical replay",
                source=current,
                candidate=replay_state,
                action=selected_action,
                proposal=_mapping(
                    selected_record.get("proposal"),
                    f"{round_location} selected candidate.proposal",
                ),
            ),
            f"{round_location} canonical replay failed hard gate",
        )
        _validate_session_rebase(
            replay_execution,
            selected_action,
            replay_state,
            f"{round_location} canonical replay",
        )
        replay_metrics = compute_metrics(replay_state.masks["change"], ground_truth)
        _require_metrics(
            replay.get("metrics"),
            replay_metrics,
            ground_truth.shape,
            f"{round_location}.canonical_replay.metrics",
        )
        expected_replay_endpoint = endpoint_gate(
            replay_metrics,
            minimum_iou=endpoint_minimum_iou,
            minimum_f1=endpoint_minimum_f1,
            empty_gt_max_fp=endpoint_empty_gt_max_fp,
            epsilon=epsilon,
        )
        require(
            replay.get("endpoint_gate") == expected_replay_endpoint,
            f"{round_location} replay endpoint gate differs",
        )

        step_location = f"{location} step {step_cursor + 1}"
        step = _mapping(steps[step_cursor], step_location)
        require(
            step.get("step_index", step.get("step_id")) == round_position,
            f"{step_location} does not correspond to selection round",
        )
        for key in ("candidate_id", "proposal", "reachability_tier", "action"):
            require(
                step.get(key) == selected_record.get(key),
                f"{step_location} {key} differs from selected candidate",
            )
        require(
            step.get("source_state_sha256") == current.identity
            and step.get("source_state_mask_sha256", current.mask_hashes)
            == current.mask_hashes,
            f"{step_location} source evidence differs",
        )
        step_candidate = _load_state(
            root, step.get("candidate_state"), f"{step_location}.candidate_state"
        )
        step_canonical = _load_state(
            root,
            step.get("canonical_state", step.get("next_state")),
            f"{step_location}.canonical_state",
        )
        step_replay = _load_state(
            root, step.get("replay_state"), f"{step_location}.replay_state"
        )
        _same_state(selected_state, step_candidate, f"{step_location} selected/candidate")
        _same_state(replay_state, step_canonical, f"{step_location} round/canonical")
        _same_state(replay_state, step_replay, f"{step_location} round/replay")
        require(
            step.get("tool_receipt") == replay.get("tool_receipt"),
            f"{step_location} tool receipt differs from canonical replay",
        )
        require(
            step.get("hard_gate") == selected_record.get("hard_gate"),
            f"{step_location} hard gate differs from selected candidate",
        )
        require(
            step.get("rgb_semantic") == selected_record.get("rgb_semantic_gate"),
            f"{step_location} RGB semantic evidence differs from selected candidate",
        )
        _qwen_is_disabled(step, step_location)
        require(
            step.get("tool_success") is True
            and step.get("candidate_accepted_by_oracle_selector") is True
            and step.get("branch_replay_mask_hash_match") is True
            and not _budget_hit(step),
            f"{step_location} selected/replay flags differ",
        )
        require(
            step.get("endpoint_gate") == expected_replay_endpoint,
            f"{step_location} endpoint gate differs from canonical replay",
        )
        if expected_replay_endpoint["passed"]:
            require(
                round_position == len(rounds) and step_cursor + 1 == len(steps),
                f"{step_location} continued after reaching the endpoint",
            )
        selected_steps.append(
            EvidenceStep(
                action=selected_action,
                candidate=selected_state,
                canonical=replay_state,
                replay=replay_state,
            )
        )
        current = replay_state
        step_cursor += 1

    require(
        step_cursor == len(steps),
        f"{location} selection-round/canonical-step coverage differs",
    )
    return selected_steps


@dataclass(frozen=True)
class EndpointContract:
    epsilon: float
    minimum_local_reduction: float
    minimum_iou: float
    minimum_f1: float
    empty_gt_max_fp: int
    required_endpoint_successes: int
    required_evaluated_episodes: int
    maximum_inconclusive_or_not_evaluated: int
    minimum_nontrivial_accepted_trajectories: int


@dataclass
class ValidatedEpisode:
    episode_id: str
    sample_id: str
    parent_id: str
    split: str
    status: str
    termination_reason: str
    exact_recovery: bool
    initial_metrics: dict[str, Any]
    final_metrics: dict[str, Any]
    final_error_reduction_ratio: float | None
    success_step: int | None
    transitions: list[dict[str, Any]]
    source_episode_sha256: str
    initial_endpoint_gate: dict[str, Any]
    final_endpoint_gate: dict[str, Any]
    derivation_context: dict[str, Any]
    diagnostics: dict[str, Any]


def _action_tool(action: Mapping[str, Any]) -> str:
    action_type = action.get("action")
    require(
        action_type in {"positive_point", "negative_point", "box"},
        f"validated action type is invalid: {action_type!r}",
    )
    return "SAM3" if action_type == "box" else "SimpleClick"


def _independent_budget_hit(selection_rounds: Sequence[Any]) -> bool:
    """Derive a terminal budget stop from executed/planned work, not receipts."""

    require(bool(selection_rounds), "termination requires at least one selection round")
    completed_executions = 0
    for position, raw_round in enumerate(selection_rounds, start=1):
        round_value = _mapping(raw_round, f"selection round {position}")
        candidates = round_value.get("candidates")
        shortlist = round_value.get("execution_shortlist")
        beneficial = round_value.get("beneficial_shortlist")
        require(
            isinstance(candidates, list)
            and isinstance(shortlist, list)
            and isinstance(beneficial, list),
            f"selection round {position} termination lists are invalid",
        )
        is_last = position == len(selection_rounds)
        batch_complete = round_value.get(
            "candidate_batch_complete_before_final_gt_scoring"
        )
        if batch_complete is not True:
            return bool(
                is_last
                and not candidates
                and completed_executions + len(shortlist)
                > MAX_TOOL_EXECUTIONS
            )
        completed_executions += len(candidates)
        if round_value.get("selected_candidate_id") is not None:
            completed_executions += 1
        elif beneficial:
            return bool(
                is_last and completed_executions + 1 > MAX_TOOL_EXECUTIONS
            )
    return False


def _derive_termination_reason(
    *,
    selection_rounds: Sequence[Any],
    transitions: Sequence[Mapping[str, Any]],
    initial_metrics: Mapping[str, Any],
    initial_endpoint: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    final_endpoint: Mapping[str, Any],
) -> tuple[str, bool]:
    """Derive the unique stop reason from independently validated evidence."""

    if int(initial_metrics["pixel_error"]) == 0:
        require(not transitions, "already-exact termination contains transitions")
        return "already_exact", False
    if initial_endpoint.get("passed") is True:
        require(not transitions, "already-endpoint termination contains transitions")
        return "already_endpoint", False

    budget_hit = _independent_budget_hit(selection_rounds)
    if budget_hit:
        return "budget_hit", True
    if int(final_metrics["pixel_error"]) == 0:
        require(bool(transitions), "exact recovery has no accepted transition")
        return "exact_recovery", False
    if final_endpoint.get("passed") is True:
        require(bool(transitions), "endpoint recovery has no accepted transition")
        return "endpoint_reached", False

    if len(transitions) == MAX_STEPS:
        require(
            len(selection_rounds) == MAX_STEPS
            and all(
                _mapping(value, "selection round").get("selected_candidate_id")
                is not None
                for value in selection_rounds
            ),
            "max_steps termination did not consume three selected/replayed steps",
        )
        return "max_steps", False

    last_round = _mapping(selection_rounds[-1], "terminal selection round")
    require(
        len(selection_rounds) == len(transitions) + 1
        and len(transitions) < MAX_STEPS
        and last_round.get("candidate_batch_complete_before_final_gt_scoring")
        is True
        and last_round.get("selected_candidate_id") is None
        and last_round.get("canonical_replay") is None
        and last_round.get("beneficial_shortlist") == [],
        "termination is neither independently no-beneficial nor max_steps",
    )
    return "no_beneficial", False


def _episode_diagnostics(selection_rounds: Sequence[Any]) -> dict[str, Any]:
    counters = {
        "action_catalog_rgb": Counter(),
        "candidate_rgb": Counter(),
        "candidate_hard_gate": Counter(),
        "candidate_hard_reject_reasons": Counter(),
        "candidate_metric_gate": Counter(),
        "candidate_beneficial": Counter(),
        "hard_gate_false_reject": Counter(),
        "hard_gate_false_reject_reasons": Counter(),
        "candidate_action_types": Counter(),
        "candidate_tools": Counter(),
        "selected_replay_action_types": Counter(),
        "selected_replay_tools": Counter(),
    }
    for position, raw_round in enumerate(selection_rounds, start=1):
        round_value = _mapping(raw_round, f"selection round {position}")
        beneficial_ids = {
            str(value) for value in round_value.get("beneficial_shortlist", [])
        }
        for raw_action in round_value.get("action_catalog", []):
            action_value = _mapping(raw_action, "action catalog item")
            gate = _mapping(
                action_value.get("rgb_semantic_gate"), "action catalog RGB gate"
            )
            counters["action_catalog_rgb"][
                "passed" if gate.get("passed") is True else "failed"
            ] += 1
        candidates_by_id: dict[str, Mapping[str, Any]] = {}
        for raw_candidate in round_value.get("candidates", []):
            candidate = _mapping(raw_candidate, "executed candidate")
            candidate_id = str(candidate.get("candidate_id"))
            candidates_by_id[candidate_id] = candidate
            action = _one_action(candidate, f"candidate {candidate_id}")
            action_type = str(action["action"])
            counters["candidate_action_types"][action_type] += 1
            counters["candidate_tools"][_action_tool(action)] += 1
            rgb_gate = _mapping(
                candidate.get("rgb_semantic_gate"), "candidate RGB gate"
            )
            counters["candidate_rgb"][
                "passed" if rgb_gate.get("passed") is True else "failed"
            ] += 1
            hard_gate = _mapping(candidate.get("hard_gate"), "candidate hard gate")
            rejected = hard_gate.get("rejected") is True
            counters["candidate_hard_gate"][
                "rejected" if rejected else "passed"
            ] += 1
            for reason in hard_gate.get("rejection_reasons", []):
                counters["candidate_hard_reject_reasons"][str(reason)] += 1
            metric_gate = _mapping(
                candidate.get("per_step_metric_gate"), "candidate metric gate"
            )
            metric_passed = metric_gate.get("passed") is True
            counters["candidate_metric_gate"][
                "passed" if metric_passed else "failed"
            ] += 1
            counters["candidate_beneficial"][
                "beneficial"
                if candidate_id in beneficial_ids
                else "not_beneficial"
            ] += 1
            if rejected:
                false_reject = bool(
                    metric_passed and rgb_gate.get("passed") is True
                )
                counters["hard_gate_false_reject"][
                    "would_improve_but_hard_rejected"
                    if false_reject
                    else "hard_rejected_without_metric_improvement"
                ] += 1
                if false_reject:
                    for reason in hard_gate.get("rejection_reasons", []):
                        counters["hard_gate_false_reject_reasons"][
                            str(reason)
                        ] += 1
        selected_id = round_value.get("selected_candidate_id")
        if selected_id is not None:
            require(
                str(selected_id) in candidates_by_id,
                "selected replay candidate is absent from candidate batch",
            )
            action = _one_action(
                candidates_by_id[str(selected_id)], "selected replay candidate"
            )
            action_type = str(action["action"])
            counters["selected_replay_action_types"][action_type] += 1
            counters["selected_replay_tools"][_action_tool(action)] += 1
    return {
        name: dict(sorted(counter.items())) for name, counter in counters.items()
    }


def validate_episode(
    payload: Mapping[str, Any],
    *,
    root: Path,
    expected_sample: Mapping[str, Any],
    epsilon: float,
    minimum_reduction: float,
    source_episode_sha256: str,
    expected_source_hashes: Mapping[str, str] | None = None,
    matching: Mapping[str, Any] | None = None,
    endpoint_minimum_iou: float = DEFAULT_ENDPOINT_IOU,
    endpoint_minimum_f1: float = DEFAULT_ENDPOINT_F1,
    endpoint_empty_gt_max_fp: int = DEFAULT_EMPTY_GT_MAX_FP,
) -> ValidatedEpisode:
    schema = payload.get("schema_id")
    require(schema in EPISODE_SCHEMAS, f"episode schema is invalid: {schema!r}")
    episode_id = payload.get("episode_id", payload.get("sample_id"))
    sample_id = payload.get("sample_id", episode_id)
    require(isinstance(episode_id, str) and episode_id, "episode_id is invalid")
    require(isinstance(sample_id, str) and sample_id, "sample_id is invalid")
    require(sample_id == expected_sample.get("sample_id", sample_id), "sample identity differs")
    parent_id = payload.get("parent_id")
    split = payload.get("split", payload.get("parent_split"))
    require(parent_id == expected_sample.get("parent_id"), f"{episode_id} parent differs")
    require(split == expected_sample.get("split"), f"{episode_id} split differs")
    require(
        payload.get("max_steps") == MAX_STEPS,
        f"episode {episode_id} max_steps must equal {MAX_STEPS}",
    )
    query = payload.get("query")
    require(isinstance(query, str) and query, f"episode {episode_id} query is invalid")
    if "query" in expected_sample:
        require(query == expected_sample.get("query"), f"episode {episode_id} query differs")
    _qwen_is_disabled(payload, f"episode {episode_id}")

    source_hashes = _mapping(payload.get("source_hashes"), f"episode {episode_id}.source_hashes")
    require(
        set(source_hashes) == {"t1", "t2", "label"},
        f"episode {episode_id} source_hashes inventory differs",
    )
    normalized_source_hashes: dict[str, str] = {}
    for key, value in source_hashes.items():
        require(_is_sha256(value), f"episode {episode_id} source hash {key} invalid")
        normalized_source_hashes[key] = str(value).lower()
    if expected_source_hashes is not None:
        require(
            normalized_source_hashes == dict(expected_source_hashes),
            f"episode {episode_id} source_hashes differ from config",
        )
    expected_source_parent = expected_sample.get("source_parent")
    if expected_source_parent is not None:
        require(
            payload.get("source_parent") == expected_source_parent,
            f"episode {episode_id} source_parent differs from config",
        )
    expected_crop = _mapping(expected_sample.get("crop"), f"sample {sample_id}.crop")
    if "xyxy_exclusive" in expected_crop:
        require(
            payload.get("crop_xyxy_exclusive")
            == expected_crop.get("xyxy_exclusive"),
            f"episode {episode_id} crop coordinates differ from config",
        )

    gt = _verified_mask(root, payload.get("gt_mask"), f"episode {episode_id}.gt_mask")
    require(
        int(gt.sum()) == int(expected_sample["crop"]["gt_pixels"]),
        f"episode {episode_id} GT pixels differ from config",
    )
    initial = _load_state(root, payload.get("initial_state"), f"episode {episode_id}.initial_state")
    require(initial.masks["change"].shape == gt.shape, f"episode {episode_id} GT shape differs")
    initial_metrics = compute_metrics(initial.masks["change"], gt)
    initial_value = _mapping(
        payload.get("initial_state"), f"episode {episode_id}.initial_state"
    )
    _require_metrics(
        initial_value.get("metrics"),
        initial_metrics,
        gt.shape,
        f"episode {episode_id}.initial_state.metrics",
    )
    steps = payload.get("steps", [])
    require(isinstance(steps, list), f"episode {episode_id}.steps must be a list")
    require(len(steps) <= MAX_STEPS, f"episode {episode_id} exceeds max_steps=3")
    evidence_steps = _validate_evidence_chain(
        payload,
        root=root,
        episode_id=episode_id,
        expected_sample=expected_sample,
        ground_truth=gt,
        initial=initial,
        steps=steps,
        epsilon=epsilon,
        endpoint_minimum_iou=endpoint_minimum_iou,
        endpoint_minimum_f1=endpoint_minimum_f1,
        endpoint_empty_gt_max_fp=endpoint_empty_gt_max_fp,
        matching=matching,
    )

    current = initial
    previous_metrics = initial_metrics
    transitions: list[dict[str, Any]] = []
    endpoint_success_step: int | None = None
    selection_rounds = payload.get("selection_rounds")
    require(isinstance(selection_rounds, list), f"episode {episode_id} rounds invalid")
    for position, raw_step in enumerate(steps, start=1):
        location = f"episode {episode_id} step {position}"
        step = _mapping(raw_step, location)
        step_index = step.get("step_id", step.get("step_index", position))
        require(step_index == position, f"{location} index is not contiguous")
        source_identity = step.get(
            "source_state_sha256",
            step.get("parent_state_sha256", step.get("state_sha256")),
        )
        require(source_identity == current.identity, f"{location} source-state hash differs")
        action = _one_action(step, location)
        evidence = evidence_steps[position - 1]
        require(action == evidence.action, f"{location} action differs from evidence chain")
        _qwen_is_disabled(step, location)
        tool_passed = _tool_passed(step, location)
        gate_passed = _gate_passed(step, location)
        semantic_passed = _semantic_passed(step, location)
        require(tool_passed, f"{location} tool did not succeed")
        require(gate_passed, f"{location} hard gate did not pass")
        require(semantic_passed, f"{location} RGB semantic check did not pass")
        require(not _budget_hit(step), f"{location} hit candidate/replay budget")

        candidate = evidence.candidate
        canonical = evidence.canonical
        replay = evidence.replay
        _same_state(candidate, canonical, f"{location} candidate/canonical")
        _same_state(canonical, replay, f"{location} canonical/replay")
        accepted = step.get(
            "candidate_accepted_by_oracle_selector",
            step.get("candidate_accepted", step.get("accepted")),
        )
        require(accepted is True, f"{location} is not an accepted oracle transition")

        metrics = compute_metrics(canonical.masks["change"], gt)
        require(
            metrics["pixel_error"] < previous_metrics["pixel_error"],
            f"{location} pixel_error did not strictly decrease",
        )
        if initial_metrics["gt_pixels"] == 0:
            require(metrics["fp"] < previous_metrics["fp"], f"{location} FP did not decrease")
        else:
            require(
                metrics["iou"] > previous_metrics["iou"] + epsilon,
                f"{location} IoU did not strictly increase",
            )
            require(
                metrics["f1"] > previous_metrics["f1"] + epsilon,
                f"{location} F1 did not strictly increase",
            )
        transition_endpoint_gate = endpoint_gate(
            metrics,
            minimum_iou=endpoint_minimum_iou,
            minimum_f1=endpoint_minimum_f1,
            empty_gt_max_fp=endpoint_empty_gt_max_fp,
            epsilon=epsilon,
        )
        if endpoint_success_step is None and transition_endpoint_gate["passed"]:
            endpoint_success_step = position
        transition_id = f"{episode_id}:{position}"
        transitions.append(
            {
                "transition_id": transition_id,
                "episode_id": episode_id,
                "sample_id": sample_id,
                "parent_id": parent_id,
                "split": split,
                "step_id": position,
                "source_state_sha256": current.identity,
                "source_state": copy.deepcopy(current.receipt),
                "action": action,
                "action_sha256": canonical_sha256(action),
                "candidate_state_sha256": candidate.identity,
                "candidate_state": copy.deepcopy(step.get("candidate_state")),
                "next_state_sha256": canonical.identity,
                "next_state": copy.deepcopy(
                    step.get("canonical_state", step.get("next_state"))
                ),
                "replay_state_sha256": replay.identity,
                "replay_state": copy.deepcopy(step.get("replay_state")),
                "metrics_before": previous_metrics,
                "metrics_after": metrics,
                "endpoint_gate": transition_endpoint_gate,
                "tool_success": True,
                "hard_gate_passed": True,
                "rgb_semantic_valid": True,
                "monotonic": True,
                "replay_passed": True,
                "accepted_step_evidence": copy.deepcopy(step),
                "selection_round_evidence": copy.deepcopy(
                    selection_rounds[position - 1]
                ),
            }
        )
        current = canonical
        previous_metrics = metrics

    final_metrics = previous_metrics
    initial_endpoint = endpoint_gate(
        initial_metrics,
        minimum_iou=endpoint_minimum_iou,
        minimum_f1=endpoint_minimum_f1,
        empty_gt_max_fp=endpoint_empty_gt_max_fp,
        epsilon=epsilon,
    )
    final_endpoint = endpoint_gate(
        final_metrics,
        minimum_iou=endpoint_minimum_iou,
        minimum_f1=endpoint_minimum_f1,
        empty_gt_max_fp=endpoint_empty_gt_max_fp,
        epsilon=epsilon,
    )
    termination_reason, budget_hit = _derive_termination_reason(
        selection_rounds=selection_rounds,
        transitions=transitions,
        initial_metrics=initial_metrics,
        initial_endpoint=initial_endpoint,
        final_metrics=final_metrics,
        final_endpoint=final_endpoint,
    )
    require(
        payload.get("budget_hit") is budget_hit,
        f"episode {episode_id} top-level budget flag differs from independent derivation",
    )
    if initial_metrics["pixel_error"] == 0:
        require(not steps, f"episode {episode_id} already exact but contains accepted steps")
        status = "already_exact"
        reduction_ratio = None
    elif initial_endpoint["passed"]:
        require(
            not steps,
            f"episode {episode_id} already satisfies endpoint but contains accepted steps",
        )
        status = "already_endpoint"
        reduction_ratio = 0.0
    elif budget_hit:
        status = "inconclusive_budget_exhausted"
        reduction_ratio = (
            initial_metrics["pixel_error"] - final_metrics["pixel_error"]
        ) / initial_metrics["pixel_error"]
    else:
        reduction_ratio = (
            initial_metrics["pixel_error"] - final_metrics["pixel_error"]
        ) / initial_metrics["pixel_error"]
        if transitions and final_endpoint["passed"]:
            status = "endpoint_success"
        elif transitions and reduction_ratio + epsilon >= minimum_reduction:
            status = "locally_reachable_only"
        else:
            status = "not_reachable_within_frozen_pilot"
    declared = payload.get("episode_status", payload.get("status"))
    require(declared == status, f"episode {episode_id} status differs")
    exact = np.array_equal(current.masks["change"], gt)
    require(
        payload.get("final_state_mask_sha256") == current.mask_hashes
        and payload.get("final_state_sha256") == current.identity,
        f"episode {episode_id} final state identity differs",
    )
    _require_metrics(
        payload.get("final_metrics"),
        final_metrics,
        gt.shape,
        f"episode {episode_id}.final_metrics",
    )
    declared_reduction = payload.get("final_error_reduction_ratio")
    if reduction_ratio is None:
        require(
            declared_reduction is None,
            f"episode {episode_id} final reduction ratio differs",
        )
    else:
        require(
            isinstance(declared_reduction, (int, float))
            and abs(float(declared_reduction) - reduction_ratio) <= epsilon,
            f"episode {episode_id} final reduction ratio differs",
        )
    local_progress = bool(
        transitions
        and reduction_ratio is not None
        and reduction_ratio + epsilon >= minimum_reduction
    )
    require(
        payload.get("local_reachability")
        == {
            "passed": local_progress,
            "minimum_final_error_reduction_ratio": minimum_reduction,
        },
        f"episode {episode_id} local reachability receipt differs",
    )
    require(
        payload.get("endpoint_reachability")
        == {
            "initial_gate": initial_endpoint,
            "final_gate": final_endpoint,
            "success_step": (
                endpoint_success_step if status == "endpoint_success" else None
            ),
            "exact_recovery": bool(exact),
        },
        f"episode {episode_id} endpoint reachability receipt differs",
    )
    termination = _mapping(
        payload.get("termination"), f"episode {episode_id}.termination"
    )
    candidate_count = sum(
        len(_mapping(round_value, "selection round").get("candidates", []))
        for round_value in selection_rounds
    )
    require(
        termination.get("selected_step_count") == len(transitions)
        and termination.get("candidate_count") == candidate_count
        and termination.get("exact") is bool(exact)
        and termination.get("budget_hit") is budget_hit
        and termination.get("reason") == termination_reason,
        f"episode {episode_id} termination receipt differs",
    )
    budget = _mapping(payload.get("budget"), f"episode {episode_id}.budget")
    distinct_candidate_ids = {
        str(_mapping(candidate, "candidate").get("candidate_id"))
        for round_value in selection_rounds
        for candidate in _mapping(round_value, "selection round").get(
            "candidates", []
        )
    }
    require(
        budget.get("hit") is budget_hit
        and budget.get("max_candidates_per_step") == MAX_CANDIDATES_PER_STEP
        and budget.get("max_distinct_candidates") == MAX_DISTINCT_CANDIDATES
        and budget.get("max_total_tool_executions") == MAX_TOOL_EXECUTIONS
        and budget.get("distinct_candidate_count")
        == len(distinct_candidate_ids)
        and budget.get("tool_execution_count")
        == candidate_count + len(transitions),
        f"episode {episode_id} budget receipt differs",
    )
    rgb_semantics = _mapping(
        payload.get("rgb_semantics"), f"episode {episode_id}.rgb_semantics"
    )
    expected_semantic = _mapping(
        expected_sample.get("semantic"), f"sample {sample_id}.semantic"
    )
    require(
        rgb_semantics.get("default")
        == {
            "t1": expected_semantic.get("t1_state"),
            "t2": expected_semantic.get("t2_state"),
        }
        and rgb_semantics.get("evidence")
        == expected_semantic.get("rgb_evidence", "")
        and rgb_semantics.get("source") == expected_semantic.get("source", "")
        and rgb_semantics.get("status") == expected_semantic.get("status", ""),
        f"episode {episode_id} RGB semantic provenance differs",
    )
    derivation_context = {
        "artifact_root": str(root.resolve()),
        "query": payload.get("query"),
        "rgb_semantics": copy.deepcopy(dict(rgb_semantics)),
        "source_parent": copy.deepcopy(payload.get("source_parent")),
        "source_hashes": dict(normalized_source_hashes),
        "crop": copy.deepcopy(dict(expected_crop)),
        "semantic": copy.deepcopy(expected_sample.get("semantic")),
        "gt_mask": copy.deepcopy(payload.get("gt_mask")),
        "initial_state": copy.deepcopy(payload.get("initial_state")),
        "gt_access_audit": copy.deepcopy(payload.get("gt_access_audit")),
        "initial_metrics": copy.deepcopy(initial_metrics),
        "initial_endpoint_gate": copy.deepcopy(initial_endpoint),
    }
    return ValidatedEpisode(
        episode_id=episode_id,
        sample_id=sample_id,
        parent_id=str(parent_id),
        split=str(split),
        status=status,
        termination_reason=termination_reason,
        exact_recovery=bool(exact),
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        final_error_reduction_ratio=reduction_ratio,
        success_step=(
            endpoint_success_step if status == "endpoint_success" else None
        ),
        transitions=transitions,
        source_episode_sha256=source_episode_sha256,
        initial_endpoint_gate=initial_endpoint,
        final_endpoint_gate=final_endpoint,
        derivation_context=derivation_context,
        diagnostics=_episode_diagnostics(selection_rounds),
    )


def _source_parent_hashes(sample: Mapping[str, Any]) -> dict[str, str]:
    source = _mapping(sample.get("source_parent"), "sample.source_parent")
    hashes: dict[str, str] = {}
    for name in ("t1", "t2", "label"):
        item = _mapping(source.get(name), f"sample.source_parent.{name}")
        path = item.get("path")
        expected = item.get("sha256")
        require(isinstance(path, str) and path, f"source {name} path invalid")
        normalized_parts = {part.lower() for part in Path(path).parts}
        require(
            not normalized_parts.intersection(
                {"test", "test_256", "test_original"}
            ),
            "test source is forbidden",
        )
        require(_is_sha256(expected), f"source {name} SHA256 invalid")
        source_path = Path(path)
        require(source_path.is_file(), f"source {name} missing: {source_path}")
        require(sha256_file(source_path) == expected.lower(), f"source {name} SHA256 differs")
        hashes[name] = expected.lower()
    return hashes


def _preflight_repository_path(
    repository: Path, value: Any, location: str
) -> Path:
    require(isinstance(value, str) and value, f"{location} path is invalid")
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (repository / raw).resolve()
    try:
        path.relative_to(repository.resolve())
    except ValueError as error:
        raise ValidationError(f"{location} escapes the frozen repository") from error
    return path


def _preflight_file_identity(
    value: Any,
    location: str,
    *,
    executable: bool = False,
) -> tuple[Path, str]:
    item = _mapping(value, location)
    raw_path = item.get("path")
    expected = item.get("sha256")
    require(isinstance(raw_path, str) and raw_path, f"{location}.path is invalid")
    require(_is_sha256(expected), f"{location}.sha256 is invalid")
    path = Path(raw_path).resolve()
    require(path.is_file(), f"{location} file is missing: {path}")
    require(
        sha256_file(path) == str(expected).lower(),
        f"{location} SHA256 differs",
    )
    if executable:
        require(os.access(path, os.X_OK), f"{location} is not executable")
    return path, str(expected).lower()


def _preflight_git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError(
            f"cannot verify external source repository: {repository}"
        ) from error
    return completed.stdout.strip()


def _preflight_rgb(path: Path, location: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValidationError(f"cannot read {location}: {path}") from error


def _preflight_curation_artifact(
    manifest_path: Path,
    declared: Any,
    candidate_id: str,
    artifact_name: str,
) -> tuple[Path, bool]:
    require(
        isinstance(declared, str) and declared,
        f"curation artifact {candidate_id}.{artifact_name} path is invalid",
    )
    declared_path = Path(declared)
    indexes = [
        index
        for index, component in enumerate(declared_path.parts)
        if component == "curation"
    ]
    require(
        indexes,
        f"curation artifact {candidate_id}.{artifact_name} has no curation anchor",
    )
    suffix = declared_path.parts[indexes[-1] + 1 :]
    expected_filename = {
        "t1": "t1.png",
        "t2": "t2.png",
        "gt_change": "gt_change.png",
    }[artifact_name]
    require(
        suffix == ("samples", candidate_id, expected_filename),
        f"curation artifact {candidate_id}.{artifact_name} suffix differs",
    )
    curation_root = manifest_path.parent.resolve()
    relocated = (curation_root / Path(*suffix)).resolve()
    try:
        relocated.relative_to(curation_root)
    except ValueError as error:
        raise ValidationError(
            f"curation artifact {candidate_id}.{artifact_name} escapes archive"
        ) from error
    require(
        relocated.is_file(),
        f"curation artifact {candidate_id}.{artifact_name} is missing: {relocated}",
    )
    original = declared_path.resolve()
    return relocated, relocated != original


def _preflight_forbidden_inventory(
    dataset_root: Path,
    forbidden_splits: Sequence[str],
) -> tuple[set[str], list[dict[str, str]]]:
    hashes: set[str] = set()
    inventory: list[dict[str, str]] = []
    dataset_root = dataset_root.resolve()
    for split in sorted(set(forbidden_splits)):
        split_root = (dataset_root / split).resolve()
        try:
            split_root.relative_to(dataset_root)
        except ValueError as error:
            raise ValidationError(f"forbidden split escapes dataset root: {split}") from error
        if not split_root.exists():
            continue
        require(split_root.is_dir(), f"forbidden split is not a directory: {split_root}")
        for path in sorted(split_root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(dataset_root)
            except ValueError as error:
                raise ValidationError(
                    f"forbidden split file escapes dataset root: {path}"
                ) from error
            digest = sha256_file(resolved)
            hashes.add(digest)
            inventory.append({"path": relative.as_posix(), "sha256": digest})
    return hashes, inventory


def run_preflight(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    """Verify every immutable input needed before any GPU pilot is submitted."""

    config_path = config_path.resolve()
    require(config_path.is_file(), f"pilot config is missing: {config_path}")
    disk_config = read_json(config_path, "pilot config")
    require(
        canonical_sha256(disk_config) == canonical_sha256(config),
        "in-memory pilot config differs from config file",
    )
    require(config.get("schema_id") == CONFIG_SCHEMA, "pilot config schema invalid")
    require(config.get("status") in CONFIG_STATUSES, "pilot config status invalid")

    # Run the complete frozen scientific/runtime contract before any expensive
    # filesystem or image hashing.  Preflight must not bless a drifted budget,
    # gate, resource plan, or data-derivation policy merely because its files
    # still exist.
    _validate_config(config)

    tools = _mapping(config.get("tools_and_models"), "tools_and_models")
    repository_raw = tools.get("repository")
    workspace_raw = tools.get("workspace_root")
    require(
        isinstance(repository_raw, str) and repository_raw,
        "tools_and_models.repository is invalid",
    )
    require(
        isinstance(workspace_raw, str) and workspace_raw,
        "tools_and_models.workspace_root is invalid",
    )
    repository = Path(repository_raw).resolve()
    workspace = Path(workspace_raw).resolve()
    require(repository.is_dir(), f"frozen repository is missing: {repository}")
    require(workspace.is_dir(), f"workspace root is missing: {workspace}")
    try:
        repository.relative_to(workspace)
    except ValueError as error:
        raise ValidationError("frozen repository is outside workspace root") from error
    runtime_repository = config_path.parent.parent.resolve()
    require(
        config_path.parent == runtime_repository / "configs",
        "pilot config is not inside a runtime repository",
    )
    require(
        (runtime_repository / ".git").exists(),
        "runtime repository Git identity is missing",
    )

    scope = _mapping(config.get("scope_and_interpretation"), "scope_and_interpretation")
    protocol = _mapping(config.get("execution_protocol"), "execution_protocol")
    qwen = _mapping(protocol.get("qwen_policy"), "execution_protocol.qwen_policy")
    disabled_model = _mapping(
        tools.get("qwen_base_model_disabled"),
        "tools_and_models.qwen_base_model_disabled",
    )
    require(scope.get("test_split_used") is False, "config says test split was used")
    require(scope.get("qwen_invoked") is False, "config says Qwen was invoked")
    require(
        qwen.get("enabled") is False
        and qwen.get("model_loaded") is False
        and qwen.get("allowed_call_count") == 0,
        "config Qwen policy is not fully disabled",
    )
    require(
        disabled_model.get("load_for_pilot") is False,
        "Qwen base model load is enabled",
    )

    provenance = _mapping(config.get("provenance"), "provenance")
    manual_spec = _mapping(
        provenance.get("manual_semantic_config"),
        "provenance.manual_semantic_config",
    )
    manual_path = _preflight_repository_path(
        runtime_repository,
        manual_spec.get("path"),
        "provenance.manual_semantic_config",
    )
    require(manual_path.is_file(), f"manual semantic config is missing: {manual_path}")
    require(
        _is_sha256(manual_spec.get("sha256")),
        "manual semantic config SHA256 is invalid",
    )
    manual_sha256 = sha256_file(manual_path)
    require(
        manual_sha256 == str(manual_spec["sha256"]).lower(),
        "manual semantic config SHA256 differs",
    )
    manual = read_json(manual_path, "manual semantic config")
    require(
        manual.get("schema") == manual_spec.get("schema"),
        "manual semantic config schema differs",
    )
    manual_samples_raw = manual.get("samples")
    require(isinstance(manual_samples_raw, list), "manual semantic samples must be a list")
    manual_samples: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(manual_samples_raw):
        item = _mapping(value, f"manual semantic sample {position}")
        candidate_id = item.get("candidate_id")
        require(
            isinstance(candidate_id, str) and candidate_id,
            f"manual semantic sample {position} candidate_id is invalid",
        )
        require(candidate_id not in manual_samples, "manual semantic candidate is duplicated")
        manual_samples[candidate_id] = item

    manifest_spec = _mapping(
        provenance.get("curation_manifest"), "provenance.curation_manifest"
    )
    manifest_raw = manifest_spec.get("path")
    require(isinstance(manifest_raw, str) and manifest_raw, "curation manifest path is invalid")
    manifest_path = Path(manifest_raw).resolve()
    require(manifest_path.is_file(), f"curation manifest is missing: {manifest_path}")
    require(_is_sha256(manifest_spec.get("sha256")), "curation manifest SHA256 is invalid")
    manifest_sha256 = sha256_file(manifest_path)
    require(
        manifest_sha256 == str(manifest_spec["sha256"]).lower(),
        "curation manifest SHA256 differs",
    )
    manifest = read_json(manifest_path, "curation manifest")
    require(
        manifest.get("schema") == manifest_spec.get("schema"),
        "curation manifest schema differs",
    )
    require(
        manifest.get("seed") == manifest_spec.get("curation_seed"),
        "curation manifest seed differs",
    )
    require(manifest.get("test_data_used") is False, "curation manifest used test data")
    manifest_samples_raw = manifest.get("samples")
    require(isinstance(manifest_samples_raw, list), "curation manifest samples must be a list")
    manifest_samples: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(manifest_samples_raw):
        item = _mapping(value, f"curation manifest sample {position}")
        candidate_id = item.get("candidate_id")
        require(
            isinstance(candidate_id, str) and candidate_id,
            f"curation manifest sample {position} candidate_id is invalid",
        )
        require(candidate_id not in manifest_samples, "curation manifest candidate is duplicated")
        manifest_samples[candidate_id] = item

    dataset = _mapping(config.get("dataset_contract"), "dataset_contract")
    dataset_root_raw = dataset.get("dataset_root")
    require(isinstance(dataset_root_raw, str) and dataset_root_raw, "dataset root is invalid")
    dataset_root = Path(dataset_root_raw).resolve()
    require(dataset_root.is_dir(), f"dataset root is missing: {dataset_root}")
    require(
        isinstance(manifest.get("dataset_root"), str)
        and Path(str(manifest["dataset_root"])).resolve() == dataset_root,
        "curation manifest dataset root differs",
    )
    allowed_splits = dataset.get("allowed_source_splits")
    require(
        isinstance(allowed_splits, list)
        and set(allowed_splits) == {"train_original", "val_original"},
        "allowed source splits differ",
    )
    require(
        manifest.get("source_splits") == allowed_splits,
        "curation manifest source splits differ",
    )
    declared_forbidden = dataset.get("forbidden_source_splits")
    require(
        isinstance(declared_forbidden, list)
        and {"test", "test_256"}.issubset(set(declared_forbidden)),
        "forbidden source split declaration differs",
    )
    forbidden_splits = sorted(set(declared_forbidden) | {"test_original"})
    parent_shape = dataset.get("parent_shape")
    crop_shape = dataset.get("crop_shape")
    require(
        isinstance(parent_shape, list)
        and len(parent_shape) == 2
        and all(type(item) is int and item > 0 for item in parent_shape),
        "dataset parent shape is invalid",
    )
    require(
        isinstance(crop_shape, list)
        and len(crop_shape) == 2
        and all(type(item) is int and item > 0 for item in crop_shape),
        "dataset crop shape is invalid",
    )
    require(
        dataset.get("crop_coordinates") == "xyxy_exclusive_in_parent_pixels",
        "crop coordinate convention differs",
    )

    ordered = config.get("ordered_samples")
    samples = _mapping(config.get("samples"), "samples")
    require(
        isinstance(ordered, list)
        and all(isinstance(item, str) and item for item in ordered)
        and len(ordered) == len(set(ordered)),
        "ordered_samples is invalid",
    )
    require(len(ordered) == 6, "preflight requires six frozen pilot samples")
    require(list(samples) == ordered, "sample inventory/order differs")

    selected_paths: set[Path] = set()
    selected_hashes: dict[str, set[str]] = {"train": set(), "val": set()}
    selected_parents: dict[str, set[str]] = {"train": set(), "val": set()}
    selected_rgb_crop_fingerprints: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
    }
    sample_receipts: list[dict[str, Any]] = []
    channel_directory = {"t1": "A", "t2": "B", "label": "label"}
    for sample_id in ordered:
        sample = _mapping(samples[sample_id], f"sample {sample_id}")
        split = sample.get("split")
        require(split in {"train", "val"}, f"sample {sample_id} split is invalid")
        parent_id = sample.get("parent_id")
        require(
            isinstance(parent_id, str) and parent_id,
            f"sample {sample_id} parent_id is invalid",
        )
        require(
            parent_id not in selected_parents[str(split)],
            f"sample parent is duplicated within split: {parent_id}",
        )
        selected_parents[str(split)].add(parent_id)
        legacy_id = sample.get("legacy_candidate_id")
        require(
            isinstance(legacy_id, str) and legacy_id,
            f"sample {sample_id} legacy candidate is invalid",
        )
        require(legacy_id in manual_samples, f"sample {sample_id} is absent from manual semantics")
        require(legacy_id in manifest_samples, f"sample {sample_id} is absent from curation manifest")
        semantic = _mapping(sample.get("semantic"), f"sample {sample_id}.semantic")
        manual_sample = manual_samples[legacy_id]
        require(manual_sample.get("use") is True, f"manual semantic sample is not selected: {legacy_id}")
        for source_key, config_key in (
            ("t1_state", "t1_state"),
            ("t2_state", "t2_state"),
            ("evidence", "rgb_evidence"),
        ):
            require(
                manual_sample.get(source_key) == semantic.get(config_key),
                f"sample {sample_id} manual semantic {source_key} differs",
            )

        source = _mapping(sample.get("source_parent"), f"sample {sample_id}.source_parent")
        source_paths: dict[str, Path] = {}
        source_hashes: dict[str, str] = {}
        expected_source_split = f"{split}_original"
        for channel in ("t1", "t2", "label"):
            identity = _mapping(
                source.get(channel), f"sample {sample_id}.source_parent.{channel}"
            )
            raw_path = identity.get("path")
            expected_hash = identity.get("sha256")
            require(
                isinstance(raw_path, str) and raw_path,
                f"sample {sample_id} source {channel} path is invalid",
            )
            require(
                _is_sha256(expected_hash),
                f"sample {sample_id} source {channel} SHA256 is invalid",
            )
            path = Path(raw_path).resolve()
            try:
                relative = path.relative_to(dataset_root)
            except ValueError as error:
                raise ValidationError(
                    f"sample {sample_id} source {channel} is outside dataset root"
                ) from error
            require(
                len(relative.parts) == 3
                and relative.parts[0] == expected_source_split
                and relative.parts[1] == channel_directory[channel]
                and Path(relative.parts[2]).stem == parent_id,
                f"sample {sample_id} source {channel} path/split differs",
            )
            require(
                not {part.lower() for part in relative.parts}.intersection(
                    {"test", "test_256", "test_original"}
                ),
                "test source is forbidden",
            )
            require(path.is_file(), f"sample {sample_id} source {channel} is missing: {path}")
            digest = sha256_file(path)
            require(
                digest == str(expected_hash).lower(),
                f"sample {sample_id} source {channel} SHA256 differs",
            )
            require(path not in selected_paths, f"source parent path is reused: {path}")
            selected_paths.add(path)
            selected_hashes[str(split)].add(digest)
            source_paths[channel] = path
            source_hashes[channel] = digest

        t1_parent = _preflight_rgb(source_paths["t1"], f"sample {sample_id} T1 parent")
        t2_parent = _preflight_rgb(source_paths["t2"], f"sample {sample_id} T2 parent")
        label_parent = _load_binary_mask(source_paths["label"])
        expected_parent_array_shape = (int(parent_shape[0]), int(parent_shape[1]))
        require(
            t1_parent.shape == (*expected_parent_array_shape, 3)
            and t2_parent.shape == (*expected_parent_array_shape, 3)
            and label_parent.shape == expected_parent_array_shape,
            f"sample {sample_id} parent shape differs",
        )

        crop = _mapping(sample.get("crop"), f"sample {sample_id}.crop")
        row = crop.get("row")
        column = crop.get("column")
        coordinates = crop.get("xyxy_exclusive")
        require(
            type(row) is int
            and row >= 0
            and type(column) is int
            and column >= 0,
            f"sample {sample_id} crop row/column is invalid",
        )
        require(
            isinstance(coordinates, list)
            and len(coordinates) == 4
            and all(type(item) is int for item in coordinates),
            f"sample {sample_id} crop coordinates are invalid",
        )
        x1, y1, x2, y2 = coordinates
        require(
            [y2 - y1, x2 - x1] == crop_shape
            and crop.get("shape") == crop_shape
            and x1 == column * int(crop_shape[1])
            and y1 == row * int(crop_shape[0])
            and 0 <= x1 < x2 <= int(parent_shape[1])
            and 0 <= y1 < y2 <= int(parent_shape[0]),
            f"sample {sample_id} crop coordinates/shape differ",
        )
        t1_crop = np.asarray(t1_parent[y1:y2, x1:x2], dtype=np.uint8)
        t2_crop = np.asarray(t2_parent[y1:y2, x1:x2], dtype=np.uint8)
        label_crop = np.asarray(label_parent[y1:y2, x1:x2], dtype=bool)
        gt_pixels = crop.get("gt_pixels")
        require(
            type(gt_pixels) is int and gt_pixels == int(label_crop.sum()),
            f"sample {sample_id} GT pixel count differs",
        )
        require(
            (sample.get("kind") == "changed") == (gt_pixels > 0),
            f"sample {sample_id} kind differs from GT",
        )

        manifest_sample = manifest_samples[legacy_id]
        expected_manifest_fields = {
            "split": split,
            "parent_id": parent_id,
            "patch_row": row,
            "patch_column": column,
            "kind": sample.get("kind"),
            "gt_pixels": gt_pixels,
            "component_pixels": crop.get("largest_curated_component_pixels"),
        }
        for key, expected_value in expected_manifest_fields.items():
            require(
                manifest_sample.get(key) == expected_value,
                f"sample {sample_id} curation manifest {key} differs",
            )
        manifest_source_paths = _mapping(
            manifest_sample.get("source_paths"), f"curation {legacy_id}.source_paths"
        )
        manifest_source_hashes = _mapping(
            manifest_sample.get("source_sha256"), f"curation {legacy_id}.source_sha256"
        )
        for channel in ("t1", "t2", "label"):
            require(
                isinstance(manifest_source_paths.get(channel), str)
                and Path(str(manifest_source_paths[channel])).resolve()
                == source_paths[channel]
                and manifest_source_hashes.get(channel) == source_hashes[channel],
                f"sample {sample_id} curation source {channel} differs",
            )

        artifacts = _mapping(
            manifest_sample.get("artifacts"), f"curation {legacy_id}.artifacts"
        )
        artifact_hashes = _mapping(
            manifest_sample.get("artifact_sha256"),
            f"curation {legacy_id}.artifact_sha256",
        )
        prior_hashes = _mapping(
            crop.get("prior_curation_png_sha256"),
            f"sample {sample_id}.crop.prior_curation_png_sha256",
        )
        artifact_receipt: dict[str, Any] = {}
        for artifact_name, prior_name in (
            ("t1", "t1"),
            ("t2", "t2"),
            ("gt_change", "label"),
        ):
            expected_hash = prior_hashes.get(prior_name)
            require(
                _is_sha256(expected_hash)
                and artifact_hashes.get(artifact_name) == expected_hash,
                f"sample {sample_id} curation artifact {artifact_name} hash binding differs",
            )
            artifact_path, relocated = _preflight_curation_artifact(
                manifest_path,
                artifacts.get(artifact_name),
                legacy_id,
                artifact_name,
            )
            require(
                sha256_file(artifact_path) == str(expected_hash).lower(),
                f"sample {sample_id} curation artifact {artifact_name} SHA256 differs",
            )
            if artifact_name == "gt_change":
                artifact_pixels = _load_binary_mask(artifact_path)
                require(
                    np.array_equal(artifact_pixels, label_crop),
                    f"sample {sample_id} curated GT crop differs from parent crop",
                )
            else:
                artifact_pixels = _preflight_rgb(
                    artifact_path,
                    f"sample {sample_id} curated {artifact_name}",
                )
                expected_pixels = t1_crop if artifact_name == "t1" else t2_crop
                require(
                    np.array_equal(artifact_pixels, expected_pixels),
                    f"sample {sample_id} curated {artifact_name} crop differs from parent crop",
                )
            artifact_receipt[artifact_name] = {
                "path": str(artifact_path),
                "sha256": str(expected_hash).lower(),
                "relocated_from_manifest_path": relocated,
            }

        rgb_fingerprint = canonical_sha256(
            {
                "t1": artifact_receipt["t1"]["sha256"],
                "t2": artifact_receipt["t2"]["sha256"],
            }
        )
        require(
            rgb_fingerprint not in selected_rgb_crop_fingerprints[str(split)],
            f"duplicate RGB crop content within {split}",
        )
        selected_rgb_crop_fingerprints[str(split)].add(rgb_fingerprint)
        sample_receipts.append(
            {
                "sample_id": sample_id,
                "legacy_candidate_id": legacy_id,
                "split": split,
                "parent_id": parent_id,
                "kind": sample.get("kind"),
                "source_parent": {
                    channel: {
                        "path": str(source_paths[channel]),
                        "sha256": source_hashes[channel],
                    }
                    for channel in ("t1", "t2", "label")
                },
                "crop": {
                    "row": row,
                    "column": column,
                    "xyxy_exclusive": coordinates,
                    "shape": crop_shape,
                    "gt_pixels": gt_pixels,
                    "rgb_pair_sha256": rgb_fingerprint,
                },
                "curation_artifacts": artifact_receipt,
            }
        )

    require(len(selected_paths) == 18, "preflight did not verify exactly 18 parent files")
    require(
        not selected_parents["train"].intersection(selected_parents["val"]),
        "train/validation parent IDs overlap",
    )
    require(
        not selected_hashes["train"].intersection(selected_hashes["val"]),
        "train/validation source content hashes overlap",
    )
    require(
        not selected_rgb_crop_fingerprints["train"].intersection(
            selected_rgb_crop_fingerprints["val"]
        ),
        "train/validation RGB crop content overlaps",
    )
    isolation = _mapping(
        dataset.get("parent_split_isolation"), "dataset_contract.parent_split_isolation"
    )
    require(
        isolation.get("required") is True
        and isolation.get("intersection_must_be_empty") is True,
        "parent split isolation is not required",
    )
    require(
        isolation.get("train_parent_ids")
        == [
            str(_mapping(samples[item], f"sample {item}").get("parent_id"))
            for item in ordered
            if _mapping(samples[item], f"sample {item}").get("split") == "train"
        ]
        and isolation.get("validation_parent_ids")
        == [
            str(_mapping(samples[item], f"sample {item}").get("parent_id"))
            for item in ordered
            if _mapping(samples[item], f"sample {item}").get("split") == "val"
        ],
        "declared parent split isolation inventory differs",
    )

    forbidden_hashes, forbidden_inventory = _preflight_forbidden_inventory(
        dataset_root, forbidden_splits
    )
    all_selected_source_hashes = selected_hashes["train"] | selected_hashes["val"]
    require(
        not all_selected_source_hashes.intersection(forbidden_hashes),
        "selected source content hash occurs in a forbidden test split",
    )

    tool_receipts: dict[str, Any] = {}
    for name, executable in (
        ("segagent_python", True),
        ("omniovcd_python", True),
        ("simpleclick_checkpoint", False),
        ("sam3_checkpoint", False),
        ("sam3_bpe", False),
    ):
        path, digest = _preflight_file_identity(
            tools.get(name), f"tools_and_models.{name}", executable=executable
        )
        tool_receipts[name] = {"path": str(path), "sha256": digest}

    manifest_sam3 = _mapping(manifest.get("sam3"), "curation manifest sam3")
    require(
        isinstance(manifest_sam3.get("checkpoint"), str)
        and Path(str(manifest_sam3["checkpoint"])).resolve()
        == Path(tool_receipts["sam3_checkpoint"]["path"])
        and manifest_sam3.get("checkpoint_sha256")
        == tool_receipts["sam3_checkpoint"]["sha256"]
        and isinstance(manifest_sam3.get("bpe"), str)
        and Path(str(manifest_sam3["bpe"])).resolve()
        == Path(tool_receipts["sam3_bpe"]["path"])
        and manifest_sam3.get("bpe_sha256") == tool_receipts["sam3_bpe"]["sha256"]
        and manifest_sam3.get("resolution") == protocol.get("sam3_resolution")
        and manifest_sam3.get("query") == protocol.get("query"),
        "curation manifest SAM3 identity differs from pilot",
    )

    runtime_sources = tools.get("runtime_sources")
    require(
        isinstance(runtime_sources, list) and runtime_sources,
        "runtime source inventory is invalid",
    )
    runtime_receipts: list[dict[str, str]] = []
    runtime_paths: set[Path] = set()
    for position, value in enumerate(runtime_sources):
        item = _mapping(value, f"runtime source {position}")
        path = _preflight_repository_path(
            runtime_repository, item.get("path"), f"runtime source {position}"
        )
        expected = item.get("sha256")
        require(_is_sha256(expected), f"runtime source {position} SHA256 is invalid")
        require(path.is_file(), f"runtime source is missing: {path}")
        require(path not in runtime_paths, f"runtime source is duplicated: {path}")
        runtime_paths.add(path)
        digest = sha256_file(path)
        require(digest == str(expected).lower(), f"runtime source SHA256 differs: {path}")
        runtime_receipts.append(
            {
                "path": path.relative_to(runtime_repository).as_posix(),
                "sha256": digest,
            }
        )

    external_sources = tools.get("external_source_repositories")
    require(
        isinstance(external_sources, list) and len(external_sources) == 2,
        "external source repository inventory must contain two repositories",
    )
    external_receipts: list[dict[str, Any]] = []
    external_names: set[str] = set()
    external_paths: set[Path] = set()
    for position, value in enumerate(external_sources):
        item = _mapping(value, f"external source repository {position}")
        name = item.get("name")
        raw_path = item.get("path")
        commit = item.get("commit")
        require(isinstance(name, str) and name, f"external source {position} name is invalid")
        require(name not in external_names, f"external source name is duplicated: {name}")
        external_names.add(name)
        require(isinstance(raw_path, str) and raw_path, f"external source {name} path is invalid")
        path = Path(raw_path).resolve()
        require(path.is_dir(), f"external source repository is missing: {path}")
        try:
            path.relative_to(workspace)
        except ValueError as error:
            raise ValidationError(f"external source {name} is outside workspace") from error
        require(path not in external_paths, f"external source path is duplicated: {path}")
        external_paths.add(path)
        require(_is_git_sha(commit), f"external source {name} commit is invalid")
        require(
            item.get("tracked_worktree_must_be_clean") is True,
            f"external source {name} does not require a clean tracked worktree",
        )
        top_level = Path(
            _preflight_git_output(path, "rev-parse", "--show-toplevel")
        ).resolve()
        require(top_level == path, f"external source {name} path is not repository root")
        observed_commit = _preflight_git_output(path, "rev-parse", "HEAD")
        require(
            observed_commit == str(commit).lower(),
            f"external source {name} commit differs",
        )
        tracked_status = _preflight_git_output(
            path, "status", "--porcelain=v1", "--untracked-files=no"
        )
        require(not tracked_status, f"external source {name} tracked worktree is dirty")
        external_receipts.append(
            {
                "name": name,
                "path": str(path),
                "commit": observed_commit,
                "tracked_worktree_clean": True,
            }
        )
    require(
        external_names
        == {"SegAgent_with_bundled_SimpleClick", "OmniOVCD_SAM3"},
        "external source repository names differ",
    )

    return {
        "schema_id": PREFLIGHT_SCHEMA,
        "status": "passed",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_sha256(config),
            "schema_id": config.get("schema_id"),
            "status": config.get("status"),
        },
        "repositories": {
            "runtime": str(runtime_repository),
            "configured_primary": str(repository),
        },
        "qwen": {"loaded": False, "calls": 0, "policy_enabled": False},
        "provenance": {
            "manual_semantic_config": {
                "path": str(manual_path),
                "sha256": manual_sha256,
                "schema": manual.get("schema"),
            },
            "curation_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "schema": manifest.get("schema"),
                "seed": manifest.get("seed"),
            },
        },
        "dataset": {
            "root": str(dataset_root),
            "sample_count": len(sample_receipts),
            "parent_file_count": len(selected_paths),
            "curation_artifact_count": len(sample_receipts) * 3,
            "train_parent_ids": sorted(selected_parents["train"]),
            "validation_parent_ids": sorted(selected_parents["val"]),
            "train_validation_parent_overlap": [],
            "train_validation_source_hash_overlap": [],
            "train_validation_rgb_crop_overlap": [],
            "forbidden_splits": forbidden_splits,
            "forbidden_file_count": len(forbidden_inventory),
            "forbidden_inventory_canonical_sha256": canonical_sha256(
                forbidden_inventory
            ),
            "selected_source_hash_in_forbidden_split": [],
            "samples": sample_receipts,
        },
        "tools_and_models": tool_receipts,
        "runtime_sources": runtime_receipts,
        "external_source_repositories": external_receipts,
    }


def _entries(index: Mapping[str, Any]) -> list[Any]:
    value = index.get("episodes", index.get("samples"))
    require(isinstance(value, list), "shard index episodes/samples must be a list")
    return value


def _entry_identity(entry: Any) -> tuple[str, str, str | None]:
    if isinstance(entry, str):
        return entry, f"episodes/{entry}/episode.json", None
    item = _mapping(entry, "shard index episode")
    episode_id = item.get("episode_id", item.get("sample_id"))
    require(isinstance(episode_id, str) and episode_id, "index episode_id invalid")
    path = item.get("path", f"episodes/{episode_id}/episode.json")
    expected_hash = item.get("sha256")
    if expected_hash is not None:
        require(_is_sha256(expected_hash), f"index episode {episode_id} SHA256 invalid")
    return episode_id, path, expected_hash


def _verify_complete_marker(
    root: Path,
    index_path: Path,
    index: Mapping[str, Any],
    expected_ids: list[str],
) -> dict[str, Any]:
    marker = read_json(root / "generation_complete.json", "generation complete marker")
    require(marker.get("schema_id") in COMPLETE_SCHEMAS, "generation marker schema invalid")
    require(marker.get("status", marker.get("execution_status")) in {"completed", "success"}, "generation is incomplete")
    _qwen_is_disabled(marker, "generation complete marker")
    marker_ids = marker.get("episode_ids", marker.get("samples", marker.get("episodes")))
    if marker_ids is not None:
        normalized = [
            item if isinstance(item, str) else item.get("episode_id", item.get("sample_id"))
            for item in marker_ids
        ]
        require(normalized == expected_ids, "generation marker episode order differs")
    require(
        marker.get("shard_index") == index.get("shard_index")
        and marker.get("num_shards") == index.get("num_shards"),
        "generation marker shard identity differs",
    )
    require(
        marker.get("runtime_commit") == index.get("runtime_commit"),
        "generation marker runtime commit differs",
    )
    expected_index_hash = marker.get("shard_index_sha256", marker.get("index_sha256"))
    require(_is_sha256(expected_index_hash), "generation marker index SHA256 invalid")
    actual_index_hash = sha256_file(index_path)
    require(actual_index_hash == expected_index_hash.lower(), "generation marker index SHA256 differs")
    artifact = _mapping(
        marker.get("shard_index_artifact"),
        "generation marker shard_index_artifact",
    )
    artifact_path = _verify_file_spec(root, artifact, "generation marker shard index")
    require(artifact_path == index_path.resolve(), "generation marker index path differs")
    require(
        artifact.get("canonical_sha256") == canonical_sha256(index),
        "generation marker index canonical SHA256 differs",
    )
    return marker


def _validate_config(config: Mapping[str, Any]) -> EndpointContract:
    require(config.get("schema_id") == CONFIG_SCHEMA, "pilot config schema invalid")
    require(config.get("status") in CONFIG_STATUSES, "pilot config status invalid")
    protocol = _mapping(config.get("execution_protocol"), "execution_protocol")
    require(
        protocol.get("seed") == 42
        and protocol.get("query") == "building"
        and protocol.get("sam3_resolution") == 1008
        and protocol.get("tool_device") == "cuda"
        and protocol.get("tool_timeout_seconds") == 900
        and protocol.get("max_steps") == MAX_STEPS
        and protocol.get("accepted_actions_per_step") == 1,
        "config frozen execution protocol differs",
    )
    qwen = _mapping(protocol.get("qwen_policy"), "execution_protocol.qwen_policy")
    require(
        qwen.get("enabled") is False
        and qwen.get("model_loaded") is False
        and qwen.get("allowed_call_count") == 0,
        "config Qwen policy differs",
    )
    budget = _mapping(
        config.get("candidate_and_replay_budget"),
        "candidate_and_replay_budget",
    )
    require(
        budget.get("max_candidate_actions_per_step") == MAX_CANDIDATES_PER_STEP
        and budget.get("max_steps_per_sample") == MAX_STEPS
        and budget.get("max_distinct_candidate_actions_per_sample")
        == MAX_DISTINCT_CANDIDATES
        and budget.get(
            "max_total_tool_executions_including_selected_action_replay_per_sample"
        )
        == MAX_TOOL_EXECUTIONS,
        "config candidate/replay budget must equal 4 x 3, 12, and 24",
    )
    proposal_config = _mapping(config.get("proposal_config"), "proposal_config")
    require(
        proposal_config.get("max_regions") == 1
        and proposal_config.get("max_delta_regions") == 1
        and proposal_config.get("min_component_area") == 4
        and proposal_config.get("padding_ratio") == 0.25
        and proposal_config.get("empty_mask_scan_grid_size") == 3,
        "config proposal contract differs",
    )
    require(
        config.get("authorized_negative_point_input_scope") == "full_session",
        "config negative-point input scope differs",
    )
    matching = _mapping(config.get("matching"), "matching")
    require(
        matching.get("mode") == "overlap_presence"
        and matching.get("overlap_threshold") == 0.25
        and matching.get("t12_min_instance_area") == 0
        and matching.get("cd_min_instance_area") == 0,
        "config matching contract differs",
    )
    hard_gate = _mapping(config.get("hard_gate_config"), "hard_gate_config")
    expected_hard_gate = {
        "max_selection_area_delta": 0.25,
        "max_locality_outside_ratio": 0.1,
        "max_target_mask_change_ratio": 0.25,
        "max_component_count_delta": 4,
        "max_point_proposal_change_ratio": 0.25,
        "max_point_proposal_change_pixels": 256,
        "max_point_image_change_ratio": 0.004,
        "point_proposal_change_gate_enabled": True,
        "modification_or_retuning_authorized": False,
    }
    require(
        all(hard_gate.get(key) == value for key, value in expected_hard_gate.items()),
        "config production hard-gate contract differs",
    )
    gt_boundary = _mapping(config.get("gt_boundary"), "gt_boundary")
    require(
        all(
            gt_boundary.get(key) is False
            for key in (
                "gt_visible_to_sam3_initializer",
                "gt_visible_to_simpleclick",
                "gt_visible_to_sam3_box_tool",
                "gt_visible_to_environment_proposal_enumeration",
                "gt_visible_to_hard_gates",
                "gt_visible_to_qwen",
            )
        )
        and gt_boundary.get("action_catalog_complete_before_gt_shortlisting")
        is True
        and gt_boundary.get(
            "shortlisted_candidate_batch_complete_before_final_gt_selection"
        )
        is True,
        "config GT access boundary differs",
    )
    derivation = _mapping(
        config.get("data_derivation_contract"), "data_derivation_contract"
    )
    require(
        derivation.get("enabled_only_for_endpoint_success_episodes") is True
        and derivation.get("same_source_episode_required") is True
        and derivation.get("cross_episode_splicing_forbidden") is True
        and all(
            derivation.get(key) is False
            for key in (
                "already_exact_generates_data",
                "already_endpoint_generates_data",
                "locally_reachable_only_generates_data",
                "inconclusive_generates_data",
                "failed_episode_generates_data",
            )
        ),
        "config endpoint-only data derivation contract differs",
    )
    artifact_plan = _mapping(config.get("artifact_plan"), "artifact_plan")
    require(
        artifact_plan.get("output_root") == EXPECTED_OUTPUT_ROOT,
        "config artifact output root differs",
    )
    tools = _mapping(config.get("tools_and_models"), "tools_and_models")
    qwen_model = _mapping(
        tools.get("qwen_base_model_disabled"),
        "tools_and_models.qwen_base_model_disabled",
    )
    require(
        qwen_model.get("load_for_pilot") is False,
        "config Qwen model load policy differs",
    )
    ordered_samples = config.get("ordered_samples")
    samples = _mapping(config.get("samples"), "samples")
    require(
        isinstance(ordered_samples, list)
        and len(ordered_samples) == 6
        and len(set(ordered_samples)) == 6
        and set(ordered_samples) == set(samples),
        "config six-sample inventory differs",
    )
    resource = _mapping(config.get("resource_plan"), "resource_plan")
    require(
        resource.get("slurm_required") is True
        and resource.get("partition") == "gre"
        and resource.get("planned_generation_shards") == 3
        and resource.get("maximum_total_gpus") == 3
        and resource.get("maximum_generate_select_rounds") == 3
        and resource.get("gpus_per_shard") == 1
        and resource.get("accelerator") == "A800"
        and resource.get("cpus_per_shard") == 4
        and resource.get("memory_per_shard") == "48G"
        and resource.get("time_per_shard") == "02:00:00"
        and resource.get("peak_gpu_concurrency") == 3,
        "config resource envelope differs",
    )
    shards = resource.get("shards")
    require(
        isinstance(shards, list) and len(shards) == 3,
        "config must contain exactly three shards",
    )
    expected_shard_samples = [
        ["train_407_r3c2", "val_43_r1c1"],
        ["train_114_r3c1", "val_41_r3c2"],
        ["train_364_r0c2", "val_60_r3c0"],
    ]
    covered: list[str] = []
    for position, raw_shard in enumerate(shards):
        shard = _mapping(raw_shard, f"resource_plan.shards[{position}]")
        shard_samples = shard.get("samples")
        require(
            shard.get("shard_id") == position
            and isinstance(shard_samples, list)
            and shard_samples == expected_shard_samples[position]
            and all(sample_id in samples for sample_id in shard_samples),
            f"config shard {position} identity/coverage differs",
        )
        require(
            {
                _mapping(samples[sample_id], f"sample {sample_id}").get("split")
                for sample_id in shard_samples
            }
            == {"train", "val"},
            f"config shard {position} must contain one train and one val sample",
        )
        covered.extend(shard_samples)
    require(
        len(covered) == len(set(covered))
        and set(covered) == set(ordered_samples),
        "config shard coverage differs from ordered samples",
    )
    postvalidation = _mapping(
        resource.get("independent_post_generation_validation"),
        "resource_plan.independent_post_generation_validation",
    )
    require(
        postvalidation.get("gpus") == 0
        and postvalidation.get("cpus") == 2
        and postvalidation.get("memory") == "16G",
        "config post-generation validation resources differ",
    )
    metric = _mapping(config.get("metric_and_success_contract"), "metric contract")
    epsilon = metric.get("comparison_epsilon")
    local = _mapping(metric.get("locally_reachable"), "locally reachable contract")
    minimum = local.get("minimum_final_error_reduction_ratio")
    endpoint = _mapping(metric.get("endpoint_success"), "endpoint success contract")
    minimum_iou = endpoint.get("nonempty_gt_minimum_final_iou")
    minimum_f1 = endpoint.get("nonempty_gt_minimum_final_f1")
    empty_gt_max_fp = endpoint.get("empty_gt_maximum_final_fp")
    require(epsilon == 1e-12, "comparison epsilon differs from frozen contract")
    require(minimum == 0.05, "minimum reduction differs from frozen contract")
    require(
        minimum_iou == DEFAULT_ENDPOINT_IOU,
        "endpoint minimum IoU differs from frozen contract",
    )
    require(
        minimum_f1 == DEFAULT_ENDPOINT_F1,
        "endpoint minimum F1 differs from frozen contract",
    )
    require(
        empty_gt_max_fp == DEFAULT_EMPTY_GT_MAX_FP,
        "empty-GT endpoint FP threshold differs from frozen contract",
    )
    require(
        local.get("episode_status_if_endpoint_not_reached")
        == "locally_reachable_only"
        and local.get("may_generate_c_iso_or_t_traj") is False,
        "local reachability/data contract differs",
    )
    require(
        endpoint.get("episode_status") == "endpoint_success"
        and endpoint.get("may_generate_matched_candidate_data") is True,
        "endpoint success/data contract differs",
    )
    already_endpoint = _mapping(metric.get("already_endpoint"), "already endpoint contract")
    require(
        already_endpoint.get("episode_status") == "already_endpoint"
        and already_endpoint.get("generate_c_iso_or_t_traj") is False,
        "already-endpoint data contract differs",
    )
    global_gate = _mapping(
        metric.get("global_pilot_release_gate"), "global pilot release gate"
    )
    integer_gate_fields = {
        "required_endpoint_successes": global_gate.get("required_endpoint_successes"),
        "required_evaluated_episodes": global_gate.get("required_evaluated_episodes"),
        "maximum_inconclusive_or_not_evaluated": global_gate.get(
            "maximum_inconclusive_or_not_evaluated"
        ),
        "minimum_nontrivial_accepted_trajectories": global_gate.get(
            "minimum_nontrivial_accepted_trajectories"
        ),
    }
    require(
        all(type(value) is int and value >= 0 for value in integer_gate_fields.values()),
        "global release gate counts are invalid",
    )
    require(
        integer_gate_fields
        == {
            "required_endpoint_successes": 6,
            "required_evaluated_episodes": 6,
            "maximum_inconclusive_or_not_evaluated": 0,
            "minimum_nontrivial_accepted_trajectories": 6,
        },
        "global release gate differs from frozen 6/6 contract",
    )
    require(
        global_gate.get("proceed_to_next_training_data_iteration_only_if_passed")
        is True,
        "global release gate is not mandatory",
    )
    return EndpointContract(
        epsilon=float(epsilon),
        minimum_local_reduction=float(minimum),
        minimum_iou=float(minimum_iou),
        minimum_f1=float(minimum_f1),
        empty_gt_max_fp=empty_gt_max_fp,
        required_endpoint_successes=integer_gate_fields[
            "required_endpoint_successes"
        ],
        required_evaluated_episodes=integer_gate_fields[
            "required_evaluated_episodes"
        ],
        maximum_inconclusive_or_not_evaluated=integer_gate_fields[
            "maximum_inconclusive_or_not_evaluated"
        ],
        minimum_nontrivial_accepted_trajectories=integer_gate_fields[
            "minimum_nontrivial_accepted_trajectories"
        ],
    )


def derive_matched_data(
    episodes: Sequence[ValidatedEpisode],
    *,
    global_release_passed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require(
        type(global_release_passed) is bool,
        "global release authorization must be a strict boolean",
    )
    c_iso: list[dict[str, Any]] = []
    t_traj: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.status != "endpoint_success":
            continue
        trajectory_steps: list[dict[str, Any]] = []
        training_authorized = bool(
            global_release_passed and episode.split == "train"
        )
        for transition in episode.transitions:
            shared = {
                "transition_id": transition["transition_id"],
                "episode_id": episode.episode_id,
                "sample_id": episode.sample_id,
                "parent_id": episode.parent_id,
                "split": episode.split,
                "step_id": transition["step_id"],
                "state_sha256": transition["source_state_sha256"],
                "action": transition["action"],
                "action_sha256": transition["action_sha256"],
                "next_state_sha256": transition["next_state_sha256"],
                "source_episode_sha256": episode.source_episode_sha256,
                "training_authorized": training_authorized,
            }
            rich_transition = {
                **shared,
                "candidate_state_sha256": transition[
                    "candidate_state_sha256"
                ],
                "source_state": copy.deepcopy(transition["source_state"]),
                "candidate_state": copy.deepcopy(transition["candidate_state"]),
                "next_state": copy.deepcopy(transition["next_state"]),
                "replay_state": copy.deepcopy(transition["replay_state"]),
                "replay_state_sha256": transition["replay_state_sha256"],
                "metrics_before": copy.deepcopy(transition["metrics_before"]),
                "metrics_after": copy.deepcopy(transition["metrics_after"]),
                "endpoint_gate": copy.deepcopy(transition["endpoint_gate"]),
                "tool_success": transition["tool_success"],
                "hard_gate_passed": transition["hard_gate_passed"],
                "rgb_semantic_valid": transition["rgb_semantic_valid"],
                "monotonic": transition["monotonic"],
                "selected_action_replay_passed": transition["replay_passed"],
                "accepted_step_evidence": copy.deepcopy(
                    transition["accepted_step_evidence"]
                ),
                "selection_round_evidence": copy.deepcopy(
                    transition["selection_round_evidence"]
                ),
            }
            c_iso.append(
                {
                    "schema_id": C_ISO_SCHEMA,
                    **copy.deepcopy(rich_transition),
                    "source_context": copy.deepcopy(
                        episode.derivation_context
                    ),
                    "derivation": {
                        "mode": "isolated_stage_from_endpoint_success_trajectory",
                        "source_episode_status": episode.status,
                        "cross_episode_splicing": False,
                    },
                }
            )
            trajectory_steps.append(rich_transition)
            mapping.append(
                {
                    "transition_id": transition["transition_id"],
                    "episode_id": episode.episode_id,
                    "step_id": transition["step_id"],
                    "split": episode.split,
                    "state_sha256": transition["source_state_sha256"],
                    "action_sha256": transition["action_sha256"],
                    "next_state_sha256": transition["next_state_sha256"],
                    "training_authorized": training_authorized,
                }
            )
        t_traj.append(
            {
                "schema_id": T_TRAJ_SCHEMA,
                "episode_id": episode.episode_id,
                "sample_id": episode.sample_id,
                "parent_id": episode.parent_id,
                "split": episode.split,
                "source_episode_sha256": episode.source_episode_sha256,
                "training_authorized": training_authorized,
                "source_context": copy.deepcopy(episode.derivation_context),
                "terminal": {
                    "episode_status": episode.status,
                    "success_step": episode.success_step,
                    "final_metrics": copy.deepcopy(episode.final_metrics),
                    "final_endpoint_gate": copy.deepcopy(
                        episode.final_endpoint_gate
                    ),
                    "exact_recovery": episode.exact_recovery,
                },
                "transitions": trajectory_steps,
            }
        )
    c_ids = [item["transition_id"] for item in c_iso]
    t_ids = [
        step["transition_id"]
        for episode in t_traj
        for step in episode["transitions"]
    ]
    require(len(c_ids) == len(set(c_ids)), "C-ISO transition IDs are not unique")
    require(c_ids == t_ids, "C-ISO/T-TRAJ transition bijection differs")
    split_artifacts: dict[str, Any] = {}
    for split in ("train", "val"):
        split_c_iso = [item for item in c_iso if item["split"] == split]
        split_t_traj = [item for item in t_traj if item["split"] == split]
        split_artifacts[split] = {
            "c_iso_count": len(split_c_iso),
            "t_traj_count": len(split_t_traj),
            "c_iso_canonical_sha256": canonical_sha256(split_c_iso),
            "t_traj_canonical_sha256": canonical_sha256(split_t_traj),
            "training_authorized": bool(
                global_release_passed and split == "train"
            ),
        }
    manifest = {
        "schema_id": MATCHING_SCHEMA,
        "status": "matched",
        "endpoint_success_episode_count": len(t_traj),
        "transition_count": len(c_iso),
        "c_iso_transition_count": len(c_ids),
        "t_traj_transition_count": len(t_ids),
        "bijection_100_percent": True,
        "cross_episode_splicing": False,
        "failed_episode_records_written": 0,
        "training_authorized": global_release_passed,
        "authorization_scope": (
            "train_split_only" if global_release_passed else "none"
        ),
        "authorized_splits": ["train"] if global_release_passed else [],
        "raw_artifacts_sft_ready": False,
        "requires_deterministic_gt_free_format_conversion": True,
        "split_counts": dict(sorted(Counter(item["split"] for item in c_iso).items())),
        "split_artifacts": split_artifacts,
        "transitions": mapping,
        "c_iso_canonical_sha256": canonical_sha256(c_iso),
        "t_traj_canonical_sha256": canonical_sha256(t_traj),
    }
    return c_iso, t_traj, manifest


def build_report(
    episodes: Sequence[ValidatedEpisode],
    endpoint_contract: EndpointContract | None = None,
) -> dict[str, Any]:
    endpoint_contract = endpoint_contract or EndpointContract(
        epsilon=1e-12,
        minimum_local_reduction=DEFAULT_LOCAL_ERROR_REDUCTION_RATIO,
        minimum_iou=DEFAULT_ENDPOINT_IOU,
        minimum_f1=DEFAULT_ENDPOINT_F1,
        empty_gt_max_fp=DEFAULT_EMPTY_GT_MAX_FP,
        required_endpoint_successes=6,
        required_evaluated_episodes=6,
        maximum_inconclusive_or_not_evaluated=0,
        minimum_nontrivial_accepted_trajectories=6,
    )
    evaluated = [
        episode
        for episode in episodes
        if episode.status
        in {
            "endpoint_success",
            "locally_reachable_only",
            "not_reachable_within_frozen_pilot",
        }
    ]
    endpoint_successes = [
        episode for episode in evaluated if episode.status == "endpoint_success"
    ]
    exact_denominator = [
        episode
        for episode in episodes
        if episode.status
        not in {
            "not_evaluated",
            "inconclusive_budget_exhausted",
            "already_exact",
            "already_endpoint",
        }
    ]
    transitions = [transition for episode in episodes for transition in episode.transitions]
    statuses = Counter(episode.status for episode in episodes)
    termination_reasons = Counter(
        episode.termination_reason for episode in episodes
    )
    diagnostic_names = (
        "action_catalog_rgb",
        "candidate_rgb",
        "candidate_hard_gate",
        "candidate_hard_reject_reasons",
        "candidate_metric_gate",
        "candidate_beneficial",
        "hard_gate_false_reject",
        "hard_gate_false_reject_reasons",
        "candidate_action_types",
        "candidate_tools",
        "selected_replay_action_types",
        "selected_replay_tools",
    )
    diagnostics = {name: Counter() for name in diagnostic_names}
    for episode in episodes:
        for name in diagnostic_names:
            diagnostics[name].update(episode.diagnostics.get(name, {}))
    rates = {
        "endpoint_success_at_1": wilson_rate(
            sum(episode.success_step == 1 for episode in endpoint_successes),
            len(evaluated),
        ),
        "endpoint_success_at_2": wilson_rate(
            sum(
                (episode.success_step or MAX_STEPS + 1) <= 2
                for episode in endpoint_successes
            ),
            len(evaluated),
        ),
        "endpoint_success_at_3": wilson_rate(
            sum(
                (episode.success_step or MAX_STEPS + 1) <= 3
                for episode in endpoint_successes
            ),
            len(evaluated),
        ),
        "final_endpoint_reachability": wilson_rate(
            len(endpoint_successes), len(evaluated)
        ),
        "exact_recovery": wilson_rate(sum(episode.exact_recovery for episode in exact_denominator), len(exact_denominator)),
        "tool_success": wilson_rate(sum(item["tool_success"] for item in transitions), len(transitions)),
        "hard_gate_pass": wilson_rate(sum(item["hard_gate_passed"] for item in transitions), len(transitions)),
        "rgb_semantic_valid": wilson_rate(sum(item["rgb_semantic_valid"] for item in transitions), len(transitions)),
        "monotonic_step": wilson_rate(sum(item["monotonic"] for item in transitions), len(transitions)),
        "selected_action_replay": wilson_rate(
            sum(item["replay_passed"] for item in transitions), len(transitions)
        ),
    }
    inconclusive_or_not_evaluated = (
        statuses.get("inconclusive_budget_exhausted", 0)
        + statuses.get("not_evaluated", 0)
    )
    nontrivial_endpoint_trajectories = sum(
        bool(episode.transitions) for episode in endpoint_successes
    )
    release_checks = {
        "endpoint_successes": {
            "observed": len(endpoint_successes),
            "required": endpoint_contract.required_endpoint_successes,
            "passed": len(endpoint_successes)
            >= endpoint_contract.required_endpoint_successes,
        },
        "evaluated_episodes": {
            "observed": len(evaluated),
            "required": endpoint_contract.required_evaluated_episodes,
            "passed": len(evaluated)
            >= endpoint_contract.required_evaluated_episodes,
        },
        "inconclusive_or_not_evaluated": {
            "observed": inconclusive_or_not_evaluated,
            "maximum": endpoint_contract.maximum_inconclusive_or_not_evaluated,
            "passed": inconclusive_or_not_evaluated
            <= endpoint_contract.maximum_inconclusive_or_not_evaluated,
        },
        "nontrivial_accepted_trajectories": {
            "observed": nontrivial_endpoint_trajectories,
            "required": endpoint_contract.minimum_nontrivial_accepted_trajectories,
            "passed": nontrivial_endpoint_trajectories
            >= endpoint_contract.minimum_nontrivial_accepted_trajectories,
        },
    }
    global_release_passed = all(
        check["passed"] for check in release_checks.values()
    )
    global_release_gate = {
        "passed": global_release_passed,
        "proceed_to_next_training_data_iteration": global_release_passed,
        "checks": release_checks,
    }
    per_episode = []
    for episode in episodes:
        per_episode.append(
            {
                "episode_id": episode.episode_id,
                "sample_id": episode.sample_id,
                "parent_id": episode.parent_id,
                "split": episode.split,
                "status": episode.status,
                "termination_reason": episode.termination_reason,
                "training_authorized": bool(
                    global_release_passed
                    and episode.split == "train"
                    and episode.status == "endpoint_success"
                ),
                "accepted_steps": len(episode.transitions),
                "endpoint_success_step": episode.success_step,
                "exact_recovery": episode.exact_recovery,
                "initial_metrics": episode.initial_metrics,
                "final_metrics": episode.final_metrics,
                "initial_endpoint_gate": episode.initial_endpoint_gate,
                "final_endpoint_gate": episode.final_endpoint_gate,
                "final_error_reduction_ratio": episode.final_error_reduction_ratio,
                "source_episode_sha256": episode.source_episode_sha256,
            }
        )
    return {
        "schema_id": REPORT_SCHEMA,
        "status": "validated",
        "scientific_conclusion": (
            "global_release_passed"
            if global_release_passed
            else "scientific_no_go"
        ),
        "training_authorized": global_release_passed,
        "authorization_scope": (
            "train_split_only" if global_release_passed else "none"
        ),
        "authorized_splits": ["train"] if global_release_passed else [],
        "raw_artifacts_sft_ready": False,
        "requires_deterministic_gt_free_format_conversion": True,
        "qwen_invoked": False,
        "max_steps": MAX_STEPS,
        "episode_count": len(episodes),
        "evaluated_nonexact_episode_count": len(evaluated),
        "endpoint_success_episode_count": len(endpoint_successes),
        "locally_reachable_only_episode_count": statuses.get(
            "locally_reachable_only", 0
        ),
        "accepted_transition_count": len(transitions),
        "rates": rates,
        "global_release_gate": global_release_gate,
        "termination_reason_distribution": dict(
            sorted(termination_reasons.items())
        ),
        "candidate_diagnostics": {
            name: dict(sorted(counter.items()))
            for name, counter in diagnostics.items()
        },
        "failure_taxonomy": {
            "endpoint_success": statuses.get("endpoint_success", 0),
            "locally_reachable_only": statuses.get("locally_reachable_only", 0),
            "already_exact": statuses.get("already_exact", 0),
            "already_endpoint": statuses.get("already_endpoint", 0),
            "not_reachable_within_frozen_pilot": statuses.get("not_reachable_within_frozen_pilot", 0),
            "inconclusive_budget_exhausted": statuses.get("inconclusive_budget_exhausted", 0),
            "not_evaluated": statuses.get("not_evaluated", 0),
        },
        "episodes": per_episode,
    }


def validate_inputs(
    config: Mapping[str, Any],
    input_roots: Sequence[Path],
    *,
    config_raw_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    endpoint_contract = _validate_config(config)
    require(_is_sha256(config_raw_sha256), "pilot config raw SHA256 is invalid")
    config_raw_sha256 = config_raw_sha256.lower()
    config_canonical_sha256 = canonical_sha256(config)
    matching_parameters = _matching_parameters(
        config.get("matching") if isinstance(config.get("matching"), Mapping) else None
    )
    runner_path = Path(__file__).with_name("run_qwen3vl_oracle_trajectory.py").resolve()
    require(runner_path.is_file(), "sibling oracle-trajectory runner is missing")
    runner_sha256 = sha256_file(runner_path)
    anchors = config.get("anchors", {})
    anchors = _mapping(anchors, "config.anchors")
    freeze_anchor = anchors.get("pilot_freeze_commit")
    if freeze_anchor is not None:
        require(_is_git_sha(freeze_anchor), "config pilot freeze commit is invalid")
        freeze_anchor = str(freeze_anchor).lower()
    expected_samples = _mapping(config.get("samples"), "config.samples")
    ordered_samples = config.get("ordered_samples")
    require(isinstance(ordered_samples, list), "config ordered_samples invalid")
    require(list(expected_samples) == ordered_samples, "config sample order differs")
    expected_source_hashes = {
        sample_id: _source_parent_hashes(_mapping(expected_samples[sample_id], f"sample {sample_id}"))
        for sample_id in ordered_samples
    }
    train_source_hashes = {
        digest
        for sample_id in ordered_samples
        if _mapping(expected_samples[sample_id], f"sample {sample_id}").get("split")
        == "train"
        for digest in expected_source_hashes[sample_id].values()
    }
    validation_source_hashes = {
        digest
        for sample_id in ordered_samples
        if _mapping(expected_samples[sample_id], f"sample {sample_id}").get("split")
        == "val"
        for digest in expected_source_hashes[sample_id].values()
    }
    require(
        not train_source_hashes.intersection(validation_source_hashes),
        "train/validation source content hashes overlap",
    )

    resource_plan = _mapping(config.get("resource_plan"), "resource_plan")
    configured_shards = resource_plan.get("shards")
    require(isinstance(configured_shards, list), "resource_plan.shards must be a list")
    expected_shards: dict[int, list[str]] = {}
    for position, raw_shard in enumerate(configured_shards):
        shard = _mapping(raw_shard, f"resource_plan.shards[{position}]")
        shard_id = shard.get("shard_id")
        samples = shard.get("samples")
        require(type(shard_id) is int and shard_id >= 0, "configured shard_id is invalid")
        require(shard_id not in expected_shards, "configured shard_id is duplicated")
        require(
            isinstance(samples, list)
            and all(isinstance(item, str) for item in samples)
            and len(samples) == len(set(samples)),
            f"configured shard {shard_id} sample list is invalid",
        )
        expected_shards[shard_id] = list(samples)
    configured_coverage = [
        sample_id
        for shard_id in sorted(expected_shards)
        for sample_id in expected_shards[shard_id]
    ]
    require(
        len(configured_coverage) == len(set(configured_coverage))
        and set(configured_coverage) == set(ordered_samples),
        "configured shard sample coverage differs",
    )
    require(
        len(input_roots) == len(expected_shards),
        "input shard count differs from config",
    )

    payloads_by_id: dict[str, tuple[dict[str, Any], Path, str]] = {}
    observed_ids: list[str] = []
    observed_shards: set[int] = set()
    runtime_commits: set[str] = set()
    observed_runner_hashes: set[str] = set()
    shard_bindings: list[dict[str, Any]] = []
    for input_root in input_roots:
        root = input_root.resolve()
        require(root.is_dir(), f"input root is not a directory: {root}")
        index_path = root / "shard_index.json"
        index = read_json(index_path, f"shard index {root}")
        require(index.get("schema_id") in INDEX_SCHEMAS, f"shard index schema invalid: {root}")
        require(index.get("status") == "completed", f"shard index is incomplete: {root}")
        _qwen_is_disabled(index, f"shard index {root}")
        index_config = _mapping(index.get("config"), f"shard index {root}.config")
        require(
            index_config.get("sha256") == config_raw_sha256
            and index_config.get("canonical_sha256") == config_canonical_sha256,
            f"shard index {root} config binding differs",
        )
        require(
            isinstance(index_config.get("path"), str) and index_config.get("path"),
            f"shard index {root} config path is invalid",
        )
        runtime_commit = index.get("runtime_commit")
        require(_is_git_sha(runtime_commit), f"shard index {root} runtime commit is invalid")
        runtime_commit = str(runtime_commit).lower()
        if freeze_anchor is not None:
            require(
                runtime_commit == freeze_anchor,
                f"shard index {root} runtime commit differs from config freeze anchor",
            )
        runtime_commits.add(runtime_commit)
        runner = _mapping(index.get("runner"), f"shard index {root}.runner")
        require(
            isinstance(runner.get("path"), str)
            and Path(str(runner["path"])).name == runner_path.name,
            f"shard index {root} runner path differs",
        )
        require(
            runner.get("sha256") == runner_sha256,
            f"shard index {root} runner SHA256 differs from sibling runner",
        )
        observed_runner_hashes.add(str(runner.get("sha256")))
        entries = _entries(index)
        ids = [_entry_identity(entry)[0] for entry in entries]
        require(len(ids) == len(set(ids)), f"duplicate episode in shard {root}")
        shard_id = index.get("shard_index")
        require(
            type(shard_id) is int and shard_id in expected_shards,
            f"shard index identity is invalid: {root}",
        )
        require(shard_id not in observed_shards, "input shard occurs more than once")
        observed_shards.add(shard_id)
        require(
            ids == expected_shards[shard_id],
            f"shard {shard_id} sample order/coverage differs from config",
        )
        declared_num_shards = index.get("num_shards")
        require(
            declared_num_shards == len(expected_shards),
            f"shard {shard_id} num_shards differs from config",
        )
        marker = _verify_complete_marker(root, index_path, index, ids)
        shard_bindings.append(
            {
                "shard_index": shard_id,
                "root": str(root),
                "shard_index_sha256": sha256_file(index_path),
                "generation_complete_sha256": sha256_file(
                    root / "generation_complete.json"
                ),
                "runtime_commit": runtime_commit,
                "config_sha256": config_raw_sha256,
                "config_canonical_sha256": config_canonical_sha256,
                "runner_sha256": runner_sha256,
                "marker_runtime_commit": marker["runtime_commit"],
            }
        )
        for entry in entries:
            require(isinstance(entry, Mapping), f"shard {shard_id} episode identity invalid")
            episode_id, relative, expected_hash = _entry_identity(entry)
            path = _inside(root, relative, f"episode {episode_id} path")
            require(path.is_file(), f"episode file missing: {path}")
            actual_hash = sha256_file(path)
            require(
                expected_hash is not None and actual_hash == expected_hash.lower(),
                f"episode {episode_id} SHA256 differs",
            )
            payload = read_json(path, f"episode {episode_id}")
            require(
                entry.get("canonical_sha256") == canonical_sha256(payload),
                f"episode {episode_id} canonical SHA256 differs",
            )
            require(
                entry.get("size_bytes") == path.stat().st_size,
                f"episode {episode_id} size differs",
            )
            require(
                episode_id not in payloads_by_id,
                "episode occurs in multiple shards",
            )
            payloads_by_id[episode_id] = (payload, root, actual_hash)
            observed_ids.append(episode_id)
    require(len(observed_ids) == len(set(observed_ids)), "episode occurs in multiple shards")
    require(
        observed_shards == set(expected_shards),
        "input shard identity coverage differs from config",
    )
    require(len(runtime_commits) == 1, "runtime commit differs across shards")
    require(len(observed_runner_hashes) == 1, "runner binding differs across shards")
    require(
        set(observed_ids) == set(ordered_samples)
        and len(observed_ids) == len(ordered_samples),
        "combined shard episode coverage differs",
    )
    # Resource shards intentionally interleave train/validation samples.  Validation
    # and derived-data order is the config authority, independent of input-root order.
    all_payloads = [payloads_by_id[sample_id] for sample_id in ordered_samples]

    parent_splits: dict[str, str] = {}
    episodes: list[ValidatedEpisode] = []
    configured_query = str(
        _mapping(config.get("execution_protocol"), "execution_protocol").get(
            "query", ""
        )
    )
    require(configured_query, "execution protocol query is invalid")
    for (payload, root, episode_hash), sample_id in zip(all_payloads, ordered_samples):
        expected = dict(_mapping(expected_samples[sample_id], f"sample {sample_id}"))
        expected["sample_id"] = sample_id
        expected["query"] = configured_query
        episode = validate_episode(
            payload,
            root=root,
            expected_sample=expected,
            epsilon=endpoint_contract.epsilon,
            minimum_reduction=endpoint_contract.minimum_local_reduction,
            source_episode_sha256=episode_hash,
            expected_source_hashes=expected_source_hashes[sample_id],
            matching=matching_parameters,
            endpoint_minimum_iou=endpoint_contract.minimum_iou,
            endpoint_minimum_f1=endpoint_contract.minimum_f1,
            endpoint_empty_gt_max_fp=endpoint_contract.empty_gt_max_fp,
        )
        previous_split = parent_splits.setdefault(episode.parent_id, episode.split)
        require(previous_split == episode.split, f"parent {episode.parent_id} crosses splits")
        episodes.append(episode)
    train_parents = {episode.parent_id for episode in episodes if episode.split == "train"}
    validation_parents = {episode.parent_id for episode in episodes if episode.split == "val"}
    require(not train_parents.intersection(validation_parents), "train/validation parents overlap")
    report = build_report(episodes, endpoint_contract)
    global_release_passed = report["global_release_gate"]["passed"]
    c_iso, t_traj, matching = derive_matched_data(
        episodes,
        global_release_passed=global_release_passed,
    )
    report["matching"] = {
        "endpoint_success_episode_count": matching[
            "endpoint_success_episode_count"
        ],
        "transition_count": matching["transition_count"],
        "bijection_100_percent": matching["bijection_100_percent"],
        "split_artifacts": matching["split_artifacts"],
        "training_authorized": matching["training_authorized"],
        "authorization_scope": matching["authorization_scope"],
        "authorized_splits": matching["authorized_splits"],
        "raw_artifacts_sft_ready": matching["raw_artifacts_sft_ready"],
        "requires_deterministic_gt_free_format_conversion": matching[
            "requires_deterministic_gt_free_format_conversion"
        ],
    }
    report["bindings"] = {
        "runtime_commit": next(iter(runtime_commits)),
        "config": {
            "sha256": config_raw_sha256,
            "canonical_sha256": config_canonical_sha256,
        },
        "runner": {"path": str(runner_path), "sha256": runner_sha256},
        "pilot_freeze_commit_anchor": freeze_anchor,
        "shards": sorted(shard_bindings, key=lambda item: item["shard_index"]),
    }
    return report, c_iso, t_traj, matching


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_preflight_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path = path.resolve()
    require(not path.exists(), f"refusing to overwrite preflight receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                receipt,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
            stream.write("\n")


def _release_gate_passed(report: Mapping[str, Any]) -> bool:
    gate = _mapping(report.get("global_release_gate"), "report.global_release_gate")
    checks = _mapping(gate.get("checks"), "report.global_release_gate.checks")
    contracts = {
        "endpoint_successes": ("required", 6, lambda observed, limit: observed >= limit),
        "evaluated_episodes": ("required", 6, lambda observed, limit: observed >= limit),
        "inconclusive_or_not_evaluated": (
            "maximum",
            0,
            lambda observed, limit: observed <= limit,
        ),
        "nontrivial_accepted_trajectories": (
            "required",
            6,
            lambda observed, limit: observed >= limit,
        ),
    }
    require(set(checks) == set(contracts), "release-gate check inventory differs")
    independently_passed: list[bool] = []
    for name, (limit_key, frozen_limit, predicate) in contracts.items():
        check = _mapping(checks.get(name), f"release check {name}")
        observed = check.get("observed")
        limit = check.get(limit_key)
        require(
            type(observed) is int
            and observed >= 0
            and type(limit) is int
            and limit == frozen_limit,
            f"release check {name} count/limit differs",
        )
        expected = bool(predicate(observed, limit))
        require(
            check.get("passed") is expected,
            f"release check {name} passed flag differs",
        )
        independently_passed.append(expected)
    passed = all(independently_passed)
    require(
        gate.get("passed") is passed
        and gate.get("proceed_to_next_training_data_iteration") is passed,
        "global release-gate decision differs from independent checks",
    )
    return passed


def _validate_training_authorization_bundle(
    report: Mapping[str, Any],
    c_iso: Sequence[Mapping[str, Any]],
    t_traj: Sequence[Mapping[str, Any]],
    matching: Mapping[str, Any],
) -> None:
    global_release_passed = _release_gate_passed(report)
    expected_scope = "train_split_only" if global_release_passed else "none"
    expected_splits = ["train"] if global_release_passed else []
    require(
        report.get("training_authorized") is global_release_passed
        and report.get("authorization_scope") == expected_scope
        and report.get("authorized_splits") == expected_splits
        and report.get("raw_artifacts_sft_ready") is False
        and report.get("requires_deterministic_gt_free_format_conversion")
        is True
        and report.get("scientific_conclusion")
        == (
            "global_release_passed"
            if global_release_passed
            else "scientific_no_go"
        ),
        "report training authorization differs from release gate",
    )
    report_episodes = report.get("episodes")
    require(isinstance(report_episodes, list), "report episodes are invalid")
    for position, raw_episode in enumerate(report_episodes):
        episode = _mapping(raw_episode, f"report episode {position}")
        split = episode.get("split")
        expected = bool(
            global_release_passed
            and split == "train"
            and episode.get("status") == "endpoint_success"
        )
        require(
            split in {"train", "val"}
            and episode.get("training_authorized") is expected,
            f"report episode {position} training authorization differs",
        )

    def validate_record(record: Mapping[str, Any], location: str) -> None:
        split = record.get("split")
        expected = bool(global_release_passed and split == "train")
        require(
            split in {"train", "val"}
            and record.get("training_authorized") is expected,
            f"{location} training authorization differs",
        )

    for position, record in enumerate(c_iso):
        validate_record(record, f"C-ISO record {position}")
    for position, record in enumerate(t_traj):
        validate_record(record, f"T-TRAJ record {position}")
        steps = record.get("transitions")
        require(isinstance(steps, list), f"T-TRAJ record {position} transitions invalid")
        for step_position, raw_step in enumerate(steps):
            step = _mapping(
                raw_step, f"T-TRAJ record {position} transition {step_position}"
            )
            validate_record(
                step, f"T-TRAJ record {position} transition {step_position}"
            )
            require(
                step.get("split") == record.get("split"),
                f"T-TRAJ record {position} transition split differs",
            )

    require(
        matching.get("training_authorized") is global_release_passed
        and matching.get("authorization_scope") == expected_scope
        and matching.get("authorized_splits") == expected_splits
        and matching.get("raw_artifacts_sft_ready") is False
        and matching.get("requires_deterministic_gt_free_format_conversion")
        is True,
        "matching-manifest training authorization differs",
    )
    manifest_transitions = matching.get("transitions")
    require(isinstance(manifest_transitions, list), "matching transitions are invalid")
    for position, raw_transition in enumerate(manifest_transitions):
        validate_record(
            _mapping(raw_transition, f"matching transition {position}"),
            f"matching transition {position}",
        )

    c_ids = [record.get("transition_id") for record in c_iso]
    t_ids = [
        step.get("transition_id")
        for record in t_traj
        for step in record.get("transitions", [])
    ]
    expected_mapping = [
        {
            "transition_id": record.get("transition_id"),
            "episode_id": record.get("episode_id"),
            "step_id": record.get("step_id"),
            "split": record.get("split"),
            "state_sha256": record.get("state_sha256"),
            "action_sha256": record.get("action_sha256"),
            "next_state_sha256": record.get("next_state_sha256"),
            "training_authorized": record.get("training_authorized"),
        }
        for record in c_iso
    ]
    require(
        len(c_ids) == len(set(c_ids))
        and c_ids == t_ids
        and manifest_transitions == expected_mapping
        and matching.get("endpoint_success_episode_count") == len(t_traj)
        and matching.get("transition_count") == len(c_iso)
        and matching.get("c_iso_transition_count") == len(c_ids)
        and matching.get("t_traj_transition_count") == len(t_ids)
        and matching.get("bijection_100_percent") is True
        and matching.get("c_iso_canonical_sha256") == canonical_sha256(c_iso)
        and matching.get("t_traj_canonical_sha256") == canonical_sha256(t_traj),
        "matching-manifest counts, bijection, or hashes differ",
    )
    expected_split_counts = dict(
        sorted(Counter(record.get("split") for record in c_iso).items())
    )
    require(
        matching.get("split_counts") == expected_split_counts,
        "matching split counts differ",
    )
    split_artifacts = _mapping(
        matching.get("split_artifacts"), "matching.split_artifacts"
    )
    require(set(split_artifacts) == {"train", "val"}, "matching split inventory differs")
    for split in ("train", "val"):
        artifact = _mapping(split_artifacts.get(split), f"matching split {split}")
        split_c_iso = [record for record in c_iso if record.get("split") == split]
        split_t_traj = [record for record in t_traj if record.get("split") == split]
        expected_authorized = bool(global_release_passed and split == "train")
        require(
            artifact.get("c_iso_count") == len(split_c_iso)
            and artifact.get("t_traj_count") == len(split_t_traj)
            and artifact.get("c_iso_canonical_sha256")
            == canonical_sha256(split_c_iso)
            and artifact.get("t_traj_canonical_sha256")
            == canonical_sha256(split_t_traj)
            and artifact.get("training_authorized") is expected_authorized,
            f"matching split {split} authorization or hashes differ",
        )
    report_matching = _mapping(report.get("matching"), "report.matching")
    for key in (
        "endpoint_success_episode_count",
        "transition_count",
        "bijection_100_percent",
        "split_artifacts",
        "training_authorized",
        "authorization_scope",
        "authorized_splits",
        "raw_artifacts_sft_ready",
        "requires_deterministic_gt_free_format_conversion",
    ):
        require(
            report_matching.get(key) == matching.get(key),
            f"report/matching field {key} differs",
        )


def write_outputs(
    output: Path,
    report: Mapping[str, Any],
    c_iso: Sequence[Mapping[str, Any]],
    t_traj: Sequence[Mapping[str, Any]],
    matching: Mapping[str, Any],
) -> None:
    output = output.resolve()
    require(not output.exists(), f"refusing to overwrite output: {output}")
    _validate_training_authorization_bundle(report, c_iso, t_traj, matching)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_json(temporary / "validation_report.json", report)
        require(
            all(item.get("split") in {"train", "val"} for item in c_iso)
            and all(item.get("split") in {"train", "val"} for item in t_traj),
            "derived record split is invalid",
        )
        for split in ("train", "val"):
            _write_jsonl(
                temporary / "data" / split / "c_iso.jsonl",
                [item for item in c_iso if item.get("split") == split],
            )
            _write_jsonl(
                temporary / "data" / split / "t_traj.jsonl",
                [item for item in t_traj if item.get("split") == split],
            )
        _write_json(temporary / "data" / "matching_manifest.json", matching)
        os.replace(temporary, output)
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
    config = read_json(args.config, "pilot config")
    if args.preflight_only:
        if args.input_root or args.output is not None:
            parser.error("--preflight-only cannot be combined with --input-root/--output")
        if args.preflight_receipt is None:
            parser.error("--preflight-only requires --preflight-receipt")
        receipt = run_preflight(config, args.config)
        write_preflight_receipt(args.preflight_receipt, receipt)
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "receipt": str(args.preflight_receipt.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_receipt is not None:
        parser.error("--preflight-receipt requires --preflight-only")
    if not args.input_root or args.output is None:
        parser.error("validation requires --input-root and --output")
    report, c_iso, t_traj, matching = validate_inputs(
        config,
        args.input_root,
        config_raw_sha256=sha256_file(args.config),
    )
    write_outputs(args.output, report, c_iso, t_traj, matching)
    print(json.dumps({"status": "validated", "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
