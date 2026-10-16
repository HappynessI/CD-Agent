#!/usr/bin/env python3
"""Independently validate P2 manifests and enforce the GT access boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


PREREG_SCHEMA = "qwen3vl_p2_actionability_preregistration_v1"
EXECUTION_SCHEMA = "qwen3vl_p2_oracle_execution_manifest_v1"
INDEX_SCHEMA = "qwen3vl_p2_oracle_candidates_v1"
COMPLETE_SCHEMA = "qwen3vl_p2_candidate_generation_complete_v1"
SCORING_SCHEMA = "qwen3vl_p2_oracle_scoring_manifest_v1"
SOURCE_COMMIT = "5210f89f5ca13ee1e636f5aa6a44631206a80123"
ACCEPTANCE_COMMIT = "17e9bb9914266691aee0c3d2b5e78b6bf64d7a00"
P1_IMPLEMENTATION_COMMIT = "a88404ac6a014f42692d1119d54de42c980fb310"
P1_OUTPUT_BASE = Path(
    "/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs"
)
P1_H0_OUTPUT = P1_OUTPUT_BASE / "CA_0802(5)-qwen3vl-h0-base-core3-retry3"
P1_H1_OUTPUT = P1_OUTPUT_BASE / "CA_0802(6)-qwen3vl-h1-base-core3-retry3"
P1_ARCHIVE_OUTPUT_BASE = P1_OUTPUT_BASE / "CA_0802_archive"
P1_H0_ARCHIVE_OUTPUT = P1_ARCHIVE_OUTPUT_BASE / P1_H0_OUTPUT.name
P1_H1_ARCHIVE_OUTPUT = P1_ARCHIVE_OUTPUT_BASE / P1_H1_OUTPUT.name
P1_H0_RECEIPT_SHA256 = "<hash omitted>"
P1_H1_RECEIPT_SHA256 = "<hash omitted>"
P1_H0_ROLLOUT_SHA256 = "<hash omitted>"
P1_H1_ROLLOUT_SHA256 = "<hash omitted>"
P1_PAIR_SHA256 = "<hash omitted>"
P1_ARM_IDS = {
    "h0": "v12_1-h0-legacy-single-round-base-0802",
    "h1": "v12_1-h1-gtfree-terminal-review-base-0802",
}
P1_PROTOCOL_IDS = {
    "h0": "qwen3vl_harness_h0_legacy_single_round_v1",
    "h1": "qwen3vl_harness_h1_gtfree_terminal_review_v1",
}
SAMPLES = ["test_20_15", "test_1_1", "test_50_8"]
TARGETS = ["test_1_1", "test_50_8"]
COUNTS = {"total": 36, "primary": 27, "diagnostic_executor_only": 9}
PRIMARY_BY_SAMPLE = {"test_20_15": 8, "test_1_1": 1, "test_50_8": 18}
DIAGNOSTIC_BY_SAMPLE = {"test_20_15": 8, "test_1_1": 1, "test_50_8": 0}
RUNTIME_SOURCES = [
    "change_agent/action_parser.py",
    "change_agent/adapters/omniovcd_adapter.py",
    "change_agent/adapters/sam3_adapter.py",
    "change_agent/adapters/segagent_adapter.py",
    "change_agent/adapters/staged_verifier.py",
    "change_agent/adapters/subprocess_adapters.py",
    "change_agent/environment.py",
    "change_agent/executor.py",
    "change_agent/state.py",
    "change_agent/trajectory.py",
    "change_agent/verifier.py",
    "change_agent/verifier_protocol.py",
    "change_agent/verifier_regions.py",
    "change_agent/coordinates.py",
    "tools/seeded_segmentation_worker.py",
    "tools/segmentation_worker.py",
]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_EXECUTION_KEYS = {
    "gt",
    "gt_path",
    "ground_truth",
    "ground_truth_path",
    "label",
    "label_path",
    "label_cvt",
    "offline_label",
    "input_root",
    "scoring_manifest",
}
FORBIDDEN_EXECUTION_PARTS = {
    "gt",
    "ground_truth",
    "label",
    "labels",
    "label_cvt",
}


class ValidationError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def read_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{location} must be an object")
    return value


def require_sha(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{location} must be a lowercase SHA256")
    return value


def require_exact_identity(
    value: Any,
    expected_path: Path,
    expected_sha256: str,
    location: str,
) -> Mapping[str, Any]:
    identity = require_mapping(value, location)
    if (
        identity.get("path") != str(expected_path)
        or identity.get("sha256") != expected_sha256
    ):
        raise ValidationError(f"{location} frozen identity differs")
    return identity


def materialize_p1_file(logical_path: Path, location: str) -> Path:
    """Find exact archived P1 bytes without weakening their frozen identity.

    Direct evidence is authoritative when still present.  Otherwise only the
    same relative path below one of the two explicitly named CA_0802 archive
    roots is accepted.  The archive target must stay contained in that exact
    root; no basename search or alternate retry is permitted.
    """

    logical_path = Path(os.path.abspath(logical_path))
    if logical_path.is_file():
        return logical_path.resolve(strict=True)
    relocations = (
        (P1_H0_OUTPUT, P1_H0_ARCHIVE_OUTPUT),
        (P1_H1_OUTPUT, P1_H1_ARCHIVE_OUTPUT),
    )
    for frozen_root, archive_root in relocations:
        try:
            relative = logical_path.relative_to(frozen_root)
        except ValueError:
            continue
        try:
            resolved_archive_root = archive_root.resolve(strict=True)
            materialized = (resolved_archive_root / relative).resolve(strict=True)
        except OSError as error:
            raise ValidationError(
                f"{location} exact archived P1 evidence is unavailable: {error}"
            ) from error
        if not materialized.is_relative_to(resolved_archive_root):
            raise ValidationError(f"{location} escapes its exact P1 archive root")
        if not materialized.is_file():
            raise ValidationError(f"{location} archived P1 evidence is not a file")
        return materialized
    raise ValidationError(f"{location} is not a file: {logical_path}")


def exact_materialized_p1_file(
    raw_path: Any, expected_logical_path: Path, location: str
) -> Path:
    """Require the historical path string, then resolve its exact archive copy."""

    if raw_path != str(expected_logical_path):
        raise ValidationError(f"{location} frozen path differs")
    return materialize_p1_file(expected_logical_path, location)


def resolve_identity(
    value: Any,
    repo_root: Path,
    location: str,
    *,
    verify_hash: bool,
) -> Path:
    identity = require_mapping(value, location)
    raw_path = identity.get("path")
    expected = require_sha(identity.get("sha256"), f"{location}.sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValidationError(f"{location}.path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if path.is_file():
        materialized = path.resolve(strict=True)
    elif path.is_absolute() and (
        path.is_relative_to(P1_H0_OUTPUT) or path.is_relative_to(P1_H1_OUTPUT)
    ):
        materialized = materialize_p1_file(path, location)
    else:
        materialized = path
    if verify_hash:
        if not materialized.is_file():
            raise ValidationError(f"{location} is not a file: {path}")
        if sha256_file(materialized) != expected:
            raise ValidationError(f"{location} SHA256 mismatch")
    return materialized


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    seen = np.zeros_like(source)
    result: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(source)):
        if seen[start_y, start_x]:
            continue
        seen[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
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
                        and source[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        component = np.zeros_like(source)
        ys, xs = zip(*pixels)
        component[np.asarray(ys), np.asarray(xs)] = True
        result.append(component)
    return result


def rebuild_change(t1: np.ndarray, t2: np.ndarray, overlap: float) -> np.ndarray:
    left = connected_components(t1)
    right = connected_components(t2)
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    for left_index, left_mask in enumerate(left):
        left_area = int(left_mask.sum())
        for right_index, right_mask in enumerate(right):
            intersection = int(np.logical_and(left_mask, right_mask).sum())
            if not intersection:
                continue
            if intersection / left_area >= overlap:
                matched_left.add(left_index)
            if intersection / int(right_mask.sum()) >= overlap:
                matched_right.add(right_index)
    change = np.zeros_like(t1, dtype=bool)
    for index, component in enumerate(left):
        if index not in matched_left:
            change |= component
    for index, component in enumerate(right):
        if index not in matched_right:
            change |= component
    return change


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def interior_seed(mask: np.ndarray) -> tuple[int, int]:
    remaining = np.asarray(mask, dtype=bool).copy()
    distance = np.zeros(remaining.shape, dtype=np.int32)
    while remaining.any():
        distance[remaining] += 1
        padded = np.pad(remaining, 1, constant_values=False)
        eroded = np.ones_like(remaining)
        for offset_y in range(3):
            for offset_x in range(3):
                eroded &= padded[
                    offset_y : offset_y + remaining.shape[0],
                    offset_x : offset_x + remaining.shape[1],
                ]
        remaining = eroded
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


def normalize(value: int, extent: int) -> int:
    return 0 if extent == 1 else round(value / (extent - 1) * 1000)


def normalized_point(point: tuple[int, int], shape: tuple[int, int]) -> list[int]:
    height, width = shape
    return [normalize(point[0], width), normalize(point[1], height)]


def normalized_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> list[int]:
    height, width = shape
    return [
        normalize(box[0], width),
        normalize(box[1], height),
        normalize(box[2], width),
        normalize(box[3], height),
    ]


def padded_box(
    box: tuple[int, int, int, int], shape: tuple[int, int], ratio: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    height, width = shape
    padding = max(4, round(max(x2 - x1 + 1, y2 - y1 + 1) * ratio))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    )


def overlap_fraction(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0]) + 1) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]) + 1
    )
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    right_area = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    return intersection / min(left_area, right_area)


def present_proposals(
    t1: np.ndarray, t2: np.ndarray, change: np.ndarray, max_regions: int, ratio: float
) -> list[dict[str, Any]]:
    temporal = np.logical_xor(t1, t2)
    missing = np.logical_and(temporal, ~change)
    sources = [
        ("change_component", change, 1),
        ("temporal_difference_missing", missing, 2),
        ("temporal_difference", temporal, 3),
    ]
    raw: list[dict[str, Any]] = []
    for source, mask, priority in sources:
        components = sorted(connected_components(mask), key=lambda item: int(item.sum()), reverse=True)
        for component in components:
            raw.append(
                {
                    "source": source,
                    "priority": priority,
                    "area": int(component.sum()),
                    "component": component,
                    "box": mask_box(component),
                }
            )
    raw.sort(key=lambda item: (item["priority"], -item["area"]))
    selected: list[dict[str, Any]] = []
    for item in raw:
        duplicate = next(
            (record for record in selected if overlap_fraction(record["box"], item["box"]) >= 0.8),
            None,
        )
        if duplicate is not None:
            duplicate["sources"].add(item["source"])
            duplicate["component_area"] = max(duplicate["component_area"], item["area"])
        else:
            selected.append(
                {
                    "box": item["box"],
                    "sources": {item["source"]},
                    "component_area": item["area"],
                    "component": item["component"],
                    "primary_source": item["source"],
                }
            )
    result: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        component = item["component"]
        seed = interior_seed(component)
        box = padded_box(item["box"], change.shape, ratio)
        crop = np.zeros_like(change)
        crop[box[1] : box[3] + 1, box[0] : box[2] + 1] = True
        result.append(
            {
                "region_id": f"r{index}",
                "batch_index": index // max_regions,
                "sources": sorted(item["sources"]),
                "audit_kind": (
                    "present"
                    if item["primary_source"] == "change_component"
                    else "missing"
                    if item["primary_source"] == "temporal_difference_missing"
                    else "mixed"
                ),
                "component_area": int(item["component_area"]),
                "component_seed_pixels": list(seed),
                "component_seed_normalized": normalized_point(seed, change.shape),
                "box_pixels": list(box),
                "box_normalized": normalized_box(box, change.shape),
                "change_pixels": int(np.logical_and(change, crop).sum()),
                "temporal_difference_pixels": int(np.logical_and(temporal, crop).sum()),
                "candidate_delta_pixels": 0,
                "candidate_added_pixels": 0,
                "candidate_removed_pixels": 0,
                "t1_mask_pixels": int(np.logical_and(t1, crop).sum()),
                "t2_mask_pixels": int(np.logical_and(t2, crop).sum()),
                "component_t1_mask_pixels": int(np.logical_and(t1, component).sum()),
                "component_t2_mask_pixels": int(np.logical_and(t2, component).sum()),
                "component_seed_t1_mask_white": bool(t1[seed[1], seed[0]]),
                "component_seed_t2_mask_white": bool(t2[seed[1], seed[0]]),
            }
        )
    return result


def scan_proposals(shape: tuple[int, int], grid: int, max_regions: int) -> list[dict[str, Any]]:
    height, width = shape
    result: list[dict[str, Any]] = []
    for row in range(grid):
        y1, y2 = row * height // grid, (row + 1) * height // grid - 1
        for column in range(grid):
            x1, x2 = column * width // grid, (column + 1) * width // grid - 1
            index = row * grid + column
            seed = ((x1 + x2) // 2, (y1 + y2) // 2)
            box = (x1, y1, x2, y2)
            result.append(
                {
                    "region_id": f"s{index}",
                    "proposal_version": "qwen3vl_empty_dual_mask_scan_grid_v1",
                    "batch_index": index // max_regions,
                    "sources": ["empty_dual_mask_full_frame_scan"],
                    "audit_kind": "scan",
                    "component_area": (x2 - x1 + 1) * (y2 - y1 + 1),
                    "component_seed_pixels": list(seed),
                    "component_seed_normalized": normalized_point(seed, shape),
                    "box_pixels": list(box),
                    "box_normalized": normalized_box(box, shape),
                    "change_pixels": 0,
                    "temporal_difference_pixels": 0,
                    "candidate_delta_pixels": 0,
                    "candidate_added_pixels": 0,
                    "candidate_removed_pixels": 0,
                    "t1_mask_pixels": 0,
                    "t2_mask_pixels": 0,
                    "component_t1_mask_pixels": 0,
                    "component_t2_mask_pixels": 0,
                    "component_seed_t1_mask_white": False,
                    "component_seed_t2_mask_white": False,
                    "scan_tile_row": row,
                    "scan_tile_column": column,
                }
            )
    return result


def action_record(
    sample: str,
    proposal: Mapping[str, Any],
    tier: str,
    action: str,
    target: str,
) -> dict[str, Any]:
    geometry = proposal["box_pixels"] if action == "box" else proposal["component_seed_pixels"]
    result = {
        "candidate_id": "--".join((sample, tier, str(proposal["region_id"]), action, target)),
        "sample": sample,
        "tier": tier,
        "region_id": proposal["region_id"],
        "action": action,
        "target_view": target,
        "geometry_pixels": list(geometry),
    }
    result["box" if action == "box" else "coordinate"] = list(geometry)
    return result


def actions_for(sample: str, proposals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    for proposal in proposals:
        if proposal["audit_kind"] == "scan":
            primary.extend(action_record(sample, proposal, "primary", "box", view) for view in ("t1", "t2"))
            continue
        if proposal["audit_kind"] != "present":
            continue
        white = [view for view in ("t1", "t2") if proposal[f"component_seed_{view}_mask_white"]]
        black = [view for view in ("t1", "t2") if not proposal[f"component_seed_{view}_mask_white"]]
        if len(white) != 1 or len(black) != 1:
            raise ValidationError(f"{sample} present proposal is not one-sided")
        primary.append(action_record(sample, proposal, "primary", "negative_point", white[0]))
        diagnostic.append(
            action_record(sample, proposal, "diagnostic_executor_only", "positive_point", black[0])
        )
    return primary + diagnostic


def classify_candidate(
    before: Mapping[str, int | float], after: Mapping[str, int | float], epsilon: float
) -> str:
    delta_iou = float(after["iou"]) - float(before["iou"])
    delta_f1 = float(after["f1"]) - float(before["f1"])
    if delta_iou > epsilon and delta_f1 > epsilon:
        return "beneficial"
    tied = abs(delta_iou) <= epsilon and abs(delta_f1) <= epsilon
    worse = (
        int(after["tp"]) < int(before["tp"])
        or int(after["fp"]) > int(before["fp"])
        or int(after["fn"]) > int(before["fn"])
    )
    if delta_iou < -epsilon or delta_f1 < -epsilon or (tied and worse):
        return "harmful"
    return "indeterminate"


def assert_gt_free(value: Any, location: str = "execution") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_EXECUTION_KEYS:
                raise ValidationError(f"GT/scoring key leaked into {location}.{key}")
            assert_gt_free(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_gt_free(item, f"{location}[{index}]")
    elif isinstance(value, str):
        parts = {part.lower() for part in Path(value).parts}
        if parts & FORBIDDEN_EXECUTION_PARTS:
            raise ValidationError(f"GT path leaked into {location}")


def expected_execution(prereg: Mapping[str, Any]) -> dict[str, Any]:
    protocol = require_mapping(prereg["execution_protocol"], "execution_protocol")
    runtime = require_mapping(prereg["runtime_files"], "runtime_files")
    action_generator = require_mapping(runtime["action_generator"], "runtime_files.action_generator")
    samples = require_mapping(prereg["samples"], "samples")
    projected_samples = {
        name: {key: value for key, value in require_mapping(samples[name], name).items() if key != "offline_label"}
        for name in SAMPLES
    }
    payload = {
        "schema_id": EXECUTION_SCHEMA,
        "status": "frozen",
        "source_freeze_commit": SOURCE_COMMIT,
        "implementation_commit": prereg["anchors"]["implementation_commit"],
        "preregistration_canonical_sha256": canonical_sha256(prereg),
        "action_generator_sha256": action_generator["sha256"],
        "runtime_source_files": runtime["generation_runtime_sources"],
        "ordered_samples": SAMPLES,
        "expected_candidate_counts": COUNTS,
        "seed": protocol["seed"],
        "query": protocol["query"],
        "single_step_independent_reset": True,
        "authorized_negative_point_input_scope": protocol["authorized_negative_point_input_scope"],
        "matching": protocol["matching"],
        "proposal_config": protocol["proposal_config"],
        "gate_config": protocol["gate_config"],
        "tool_device": protocol["tool_device"],
        "sam3_resolution": protocol["sam3_resolution"],
        "tool_timeout_seconds": protocol["tool_timeout_seconds"],
        "tools": prereg["tools"],
        "samples": projected_samples,
    }
    assert_gt_free(payload)
    return payload


def validate_structure(prereg: Mapping[str, Any]) -> None:
    if prereg.get("schema_id") != PREREG_SCHEMA or prereg.get("design_status") != "frozen":
        raise ValidationError("preregistration schema/design status differs")
    if prereg.get("execution_status") != "not_run":
        raise ValidationError("repository preregistration must remain not_run")
    anchors = require_mapping(prereg.get("anchors"), "anchors")
    if anchors.get("p1_runtime_source_freeze_commit_b4") != SOURCE_COMMIT:
        raise ValidationError("P1 source freeze commit differs")
    if anchors.get("p1_acceptance_commit") != ACCEPTANCE_COMMIT:
        raise ValidationError("P1 acceptance commit differs")
    if COMMIT_RE.fullmatch(str(anchors.get("implementation_commit"))) is None:
        raise ValidationError("implementation commit A is missing or malformed")
    h0_source = require_mapping(prereg.get("source_h0_retry3"), "source_h0_retry3")
    expected_h0_source = {
        "job_id": "53158",
        "state": "COMPLETED",
        "arm": "h0",
        "experiment_arm_id": P1_ARM_IDS["h0"],
        "harness_protocol_id": P1_PROTOCOL_IDS["h0"],
        "runtime_commit": SOURCE_COMMIT,
        "output_root": str(P1_H0_OUTPUT),
    }
    if any(h0_source.get(key) != value for key, value in expected_h0_source.items()):
        raise ValidationError("H0 retry3 source identity differs")
    require_exact_identity(
        h0_source.get("rollout_complete"),
        P1_H0_OUTPUT / "run/rollout_complete.json",
        P1_H0_ROLLOUT_SHA256,
        "source_h0_retry3.rollout_complete",
    )
    require_exact_identity(
        h0_source.get("evaluation_receipt"),
        P1_H0_OUTPUT / "evaluation_receipt.json",
        P1_H0_RECEIPT_SHA256,
        "source_h0_retry3.evaluation_receipt",
    )
    source = require_mapping(prereg.get("source_h1_retry3"), "source_h1_retry3")
    expected_source = {
        "job_id": "53159",
        "state": "COMPLETED",
        "arm": "h1",
        "experiment_arm_id": P1_ARM_IDS["h1"],
        "harness_protocol_id": P1_PROTOCOL_IDS["h1"],
        "runtime_commit": SOURCE_COMMIT,
        "output_root": str(P1_H1_OUTPUT),
        "paired_h0_output_root": str(P1_H0_OUTPUT),
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise ValidationError("H1 retry3 source identity differs")
    require_exact_identity(
        source.get("rollout_complete"),
        P1_H1_OUTPUT / "run/rollout_complete.json",
        P1_H1_ROLLOUT_SHA256,
        "source_h1_retry3.rollout_complete",
    )
    require_exact_identity(
        source.get("evaluation_receipt"),
        P1_H1_OUTPUT / "evaluation_receipt.json",
        P1_H1_RECEIPT_SHA256,
        "source_h1_retry3.evaluation_receipt",
    )
    require_exact_identity(
        source.get("pair_comparison_receipt"),
        P1_H1_OUTPUT / "pair_comparison_receipt.json",
        P1_PAIR_SHA256,
        "source_h1_retry3.pair_comparison_receipt",
    )
    runtime = require_mapping(prereg.get("runtime_files"), "runtime_files")
    runtime_sources = runtime.get("generation_runtime_sources")
    if not isinstance(runtime_sources, list) or [
        require_mapping(item, f"runtime_source[{index}]").get("path")
        for index, item in enumerate(runtime_sources)
    ] != RUNTIME_SOURCES:
        raise ValidationError("generation runtime source inventory differs")
    for name in ("action_generator", "scorer"):
        require_sha(
            require_mapping(runtime.get(name), f"runtime_files.{name}").get("sha256"),
            f"runtime_files.{name}.sha256",
        )
    for index, item in enumerate(runtime_sources):
        require_sha(item.get("sha256"), f"runtime_source[{index}].sha256")
    if prereg.get("ordered_samples") != SAMPLES or prereg.get("target_samples") != TARGETS:
        raise ValidationError("sample order or target set differs")
    protocol = require_mapping(prereg.get("execution_protocol"), "execution_protocol")
    exact_protocol = {
        "seed": 42,
        "query": "building",
        "single_step_independent_reset": True,
        "max_actions_per_candidate": 1,
        "candidate_to_candidate_state_carryover": False,
        "qwen_invoked": False,
        "tool_device": "cuda",
        "sam3_resolution": 1008,
        "tool_timeout_seconds": 900,
        "authorized_negative_point_input_scope": "full_session",
    }
    for key, expected in exact_protocol.items():
        if protocol.get(key) != expected:
            raise ValidationError(f"execution_protocol.{key} differs")
    if protocol.get("matching") != {
        "mode": "overlap_presence",
        "overlap_threshold": 0.25,
        "t12_min_instance_area": 0,
        "cd_min_instance_area": 0,
    }:
        raise ValidationError("matching contract differs")
    if protocol.get("proposal_config") != {
        "max_regions": 1,
        "max_delta_regions": 1,
        "min_component_area": 4,
        "padding_ratio": 0.25,
        "empty_mask_scan_grid_size": 3,
    }:
        raise ValidationError("Environment proposal config differs")
    if protocol.get("gate_config") != {
        "max_selection_area_delta": 0.25,
        "max_locality_outside_ratio": 0.1,
        "max_target_mask_change_ratio": 0.25,
        "max_component_count_delta": 4,
        "max_point_proposal_change_ratio": 0.25,
        "max_point_proposal_change_pixels": 256,
        "max_point_image_change_ratio": 0.004,
        "point_proposal_change_gate_enabled": True,
    }:
        raise ValidationError("hard-gate thresholds differ")
    metric = require_mapping(prereg.get("metric_contract"), "metric_contract")
    if metric.get("comparison_epsilon") != 1e-12:
        raise ValidationError("comparison epsilon differs from P0")
    if classify_candidate(
        {"iou": 0.5, "f1": 0.5, "tp": 5, "fp": 1, "fn": 1},
        {"iou": 0.5, "f1": 0.5, "tp": 5, "fp": 2, "fn": 1},
        1e-12,
    ) != "harmful":
        raise ValidationError("independent harmful metric boundary failed")
    boundary = require_mapping(prereg.get("gt_boundary"), "gt_boundary")
    if not all(
        boundary.get(key) is expected
        for key, expected in {
            "generation_manifest_contains_labels": False,
            "generation_process_receives_input_root": False,
            "generation_process_receives_scoring_manifest": False,
            "generation_process_receives_only_explicit_a_b_and_frozen_mask_files": True,
            "candidate_generation_complete_marker_required_before_any_label_file_is_opened": True,
            "all_candidate_records_and_masks_hash_verified_before_any_label_file_is_opened": True,
            "labels_are_offline_scorer_only": True,
        }.items()
    ):
        raise ValidationError("GT boundary differs")


def validate_inventory(
    prereg: Mapping[str, Any], repo_root: Path, *, verify_files: bool
) -> None:
    sample_map = require_mapping(prereg.get("samples"), "samples")
    if list(sample_map) != SAMPLES:
        raise ValidationError("sample mapping order differs")
    all_proposals: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    observed_primary: dict[str, int] = {}
    observed_diagnostic: dict[str, int] = {}
    protocol = prereg["execution_protocol"]
    for name in SAMPLES:
        sample = require_mapping(sample_map[name], f"samples.{name}")
        frozen_mask_root = (
            P1_H0_OUTPUT
            / "run/tool_runs"
            / name
            / "sam3_initialization/initialize_000"
        )
        for field, filename in (("t1_mask", "t1_mask.npy"), ("t2_mask", "t2_mask.npy")):
            mask_record = require_mapping(sample.get(field), f"samples.{name}.{field}")
            if mask_record.get("path") != str(frozen_mask_root / filename):
                raise ValidationError(
                    f"{name} {field} is not the canonical H0 retry3 cached mask"
                )
        proposals = sample.get("proposals")
        actions = sample.get("actions")
        if not isinstance(proposals, list) or not isinstance(actions, list):
            raise ValidationError(f"{name} proposal/action inventory is malformed")
        if canonical_sha256(proposals) != sample.get("proposals_canonical_sha256"):
            raise ValidationError(f"{name} proposal canonical hash differs")
        if canonical_sha256(actions) != sample.get("actions_canonical_sha256"):
            raise ValidationError(f"{name} action canonical hash differs")
        if len({item.get("candidate_id") for item in actions}) != len(actions):
            raise ValidationError(f"{name} action candidate IDs are not unique")
        observed_primary[name] = sum(item.get("tier") == "primary" for item in actions)
        observed_diagnostic[name] = sum(
            item.get("tier") == "diagnostic_executor_only" for item in actions
        )
        if verify_files:
            for field in ("t1_image", "t2_image", "t1_mask", "t2_mask"):
                resolve_identity(sample.get(field), repo_root, f"samples.{name}.{field}", verify_hash=True)
            t1_path = resolve_identity(sample["t1_mask"], repo_root, "t1_mask", verify_hash=False)
            t2_path = resolve_identity(sample["t2_mask"], repo_root, "t2_mask", verify_hash=False)
            t1 = np.asarray(np.load(t1_path), dtype=bool)
            t2 = np.asarray(np.load(t2_path), dtype=bool)
            if mask_sha256(t1) != sample["t1_mask"].get("mask_sha256"):
                raise ValidationError(f"{name} T1 semantic mask hash differs")
            if mask_sha256(t2) != sample["t2_mask"].get("mask_sha256"):
                raise ValidationError(f"{name} T2 semantic mask hash differs")
            change = rebuild_change(t1, t2, float(protocol["matching"]["overlap_threshold"]))
            state_hashes = {"t1": mask_sha256(t1), "t2": mask_sha256(t2), "change": mask_sha256(change)}
            if sample.get("initial_state_mask_sha256") != state_hashes:
                raise ValidationError(f"{name} initial state mask hashes differ")
            config = protocol["proposal_config"]
            reconstructed = (
                scan_proposals(change.shape, config["empty_mask_scan_grid_size"], config["max_regions"])
                if not t1.any() and not t2.any()
                else present_proposals(t1, t2, change, config["max_regions"], config["padding_ratio"])
            )
            if reconstructed != proposals:
                raise ValidationError(f"{name} independent proposal reconstruction differs")
            if actions_for(name, reconstructed) != actions:
                raise ValidationError(f"{name} independent action reconstruction differs")
        all_proposals.extend(proposals)
        all_actions.extend(actions)
    if observed_primary != PRIMARY_BY_SAMPLE or observed_diagnostic != DIAGNOSTIC_BY_SAMPLE:
        raise ValidationError("per-sample action counts differ")
    counts = {
        "total": len(all_actions),
        "primary": sum(item["tier"] == "primary" for item in all_actions),
        "diagnostic_executor_only": sum(item["tier"] == "diagnostic_executor_only" for item in all_actions),
    }
    enumeration = prereg["action_enumeration_contract"]
    if counts != COUNTS or enumeration.get("expected_candidate_counts") != COUNTS:
        raise ValidationError("global action counts differ")
    if canonical_sha256(all_proposals) != enumeration.get("all_proposals_canonical_sha256"):
        raise ValidationError("global proposal canonical hash differs")
    if canonical_sha256(all_actions) != enumeration.get("all_actions_canonical_sha256"):
        raise ValidationError("global action canonical hash differs")
    if canonical_sha256({"proposals": all_proposals, "actions": all_actions}) != enumeration.get(
        "combined_inventory_canonical_sha256"
    ):
        raise ValidationError("combined inventory canonical hash differs")


def _validated_p1_receipt(
    path: Path,
    *,
    arm: str,
    job_id: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if sha256_file(path) != expected_sha256:
        raise ValidationError(f"P1 {arm.upper()} evaluation receipt hash differs")
    receipt = read_json(path)
    if (
        receipt.get("schema_id") != "qwen3vl_evaluation_receipt_v2"
        or receipt.get("status") != "complete"
        or receipt.get("terminal_state") != "success"
    ):
        raise ValidationError(f"P1 {arm.upper()} evaluation receipt is incomplete")
    identity = require_mapping(receipt.get("identity"), f"p1.{arm}.identity")
    expected_identity = {
        "arm": arm,
        "experiment_arm_id": P1_ARM_IDS[arm],
        "harness_protocol_id": P1_PROTOCOL_IDS[arm],
        "evaluation_profile": "e2e3",
        "evaluation_id": "DEV-CORE-3-V1",
        "sample_order": SAMPLES,
        "runtime_commit": SOURCE_COMMIT,
        "harness_commit": SOURCE_COMMIT,
        "job_id": job_id,
        "model_state": "base",
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValidationError(f"P1 {arm.upper()} evaluation identity differs")
    provenance = require_mapping(receipt.get("provenance"), f"p1.{arm}.provenance")
    if (
        provenance.get("implementation_commit") != P1_IMPLEMENTATION_COMMIT
        or provenance.get("freeze_commit") != SOURCE_COMMIT
        or provenance.get("git_dirty") is not False
    ):
        raise ValidationError(f"P1 {arm.upper()} provenance differs")
    gates = require_mapping(receipt.get("gates"), f"p1.{arm}.gates")
    expected_gates = {
        "gt_loaded_before_rollout_complete": False,
        "pre_rollout_gt_access_count": 0,
        "safety_boundary_relaxation_count": 0,
        "requested_sample_count": 3,
        "scored_sample_count": 3,
        "requested_scored_mismatch_count": 0,
        "tool_failure_count": 0,
        "offline_scoring_after_rollout_complete": True,
        "summary_validator_passed": True,
    }
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        raise ValidationError(f"P1 {arm.upper()} gates differ")
    artifacts = require_mapping(receipt.get("artifacts"), f"p1.{arm}.artifacts")
    summary_binding = require_mapping(
        artifacts.get("baseline_summary"), f"p1.{arm}.baseline_summary"
    )
    logical_output_root = P1_H0_OUTPUT if arm == "h0" else P1_H1_OUTPUT
    logical_summary_path = logical_output_root / "baseline_summary.json"
    summary_path = exact_materialized_p1_file(
        summary_binding.get("path"),
        logical_summary_path,
        f"p1.{arm}.baseline_summary.path",
    )
    if (
        summary_binding.get("sha256") != sha256_file(summary_path)
        or summary_path
        != materialize_p1_file(
            logical_summary_path, f"p1.{arm}.baseline_summary.expected"
        )
    ):
        raise ValidationError(f"P1 {arm.upper()} baseline summary binding differs")
    summary = read_json(summary_path)
    return receipt, summary, summary_path


def _validate_p1_summary(
    summary: Mapping[str, Any], *, arm: str, job_id: str
) -> Mapping[str, Any]:
    expected = {
        "summary_schema_version": "qwen3vl_e2e_summary_v2",
        "status": "completed",
        "evaluation_profile": "e2e3",
        "evaluation_id": "DEV-CORE-3-V1",
        "model_state": "base",
        "job_id": job_id,
        "experiment_arm": P1_ARM_IDS[arm],
        "harness_protocol_id": P1_PROTOCOL_IDS[arm],
        "runtime_commit": SOURCE_COMMIT,
        "harness_commit": SOURCE_COMMIT,
        "requested_samples": 3,
        "scored_samples": 3,
        "sample_order": SAMPLES,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValidationError(f"P1 {arm.upper()} summary identity differs")
    return require_mapping(summary.get("per_sample"), f"p1.{arm}.per_sample")


def _selected_signature(record: Any, location: str) -> tuple[Any, ...]:
    sample = require_mapping(record, location)
    selected = require_mapping(sample.get("selected"), f"{location}.selected")
    return (
        selected.get("iou"),
        selected.get("precision"),
        selected.get("recall"),
        selected.get("f1"),
        selected.get("tp"),
        selected.get("fp"),
        selected.get("fn"),
        selected.get("tn"),
    )


def validate_h1_pair_source(prereg: Mapping[str, Any]) -> None:
    h0_source = require_mapping(prereg.get("source_h0_retry3"), "source_h0_retry3")
    source = require_mapping(prereg.get("source_h1_retry3"), "source_h1_retry3")
    require_exact_identity(
        h0_source.get("rollout_complete"),
        P1_H0_OUTPUT / "run/rollout_complete.json",
        P1_H0_ROLLOUT_SHA256,
        "source_h0_retry3.rollout_complete",
    )
    require_exact_identity(
        source.get("rollout_complete"),
        P1_H1_OUTPUT / "run/rollout_complete.json",
        P1_H1_ROLLOUT_SHA256,
        "source_h1_retry3.rollout_complete",
    )
    h0_rollout_path = resolve_identity(
        h0_source.get("rollout_complete"),
        Path.cwd(),
        "source_h0_retry3.rollout_complete",
        verify_hash=True,
    )
    h1_rollout_path = resolve_identity(
        source.get("rollout_complete"),
        Path.cwd(),
        "source_h1_retry3.rollout_complete",
        verify_hash=True,
    )
    for arm, path in (("h0", h0_rollout_path), ("h1", h1_rollout_path)):
        rollout = read_json(path)
        if (
            rollout.get("execution_status") != "completed"
            or rollout.get("gt_loaded") is not False
            or rollout.get("samples")
            != ["test_1_1", "test_20_15", "test_50_8"]
        ):
            raise ValidationError(f"P1 {arm.upper()} rollout marker differs")
    require_exact_identity(
        h0_source.get("evaluation_receipt"),
        P1_H0_OUTPUT / "evaluation_receipt.json",
        P1_H0_RECEIPT_SHA256,
        "source_h0_retry3.evaluation_receipt",
    )
    require_exact_identity(
        source.get("evaluation_receipt"),
        P1_H1_OUTPUT / "evaluation_receipt.json",
        P1_H1_RECEIPT_SHA256,
        "source_h1_retry3.evaluation_receipt",
    )
    h0_receipt_path = resolve_identity(
        h0_source.get("evaluation_receipt"),
        Path.cwd(),
        "source_h0_retry3.evaluation_receipt",
        verify_hash=True,
    )
    h1_receipt_path = resolve_identity(
        source.get("evaluation_receipt"),
        Path.cwd(),
        "source_h1_retry3.evaluation_receipt",
        verify_hash=True,
    )
    h0_receipt, h0_summary, h0_summary_path = _validated_p1_receipt(
        h0_receipt_path,
        arm="h0",
        job_id="53158",
        expected_sha256=P1_H0_RECEIPT_SHA256,
    )
    h1_receipt, h1_summary, h1_summary_path = _validated_p1_receipt(
        h1_receipt_path,
        arm="h1",
        job_id="53159",
        expected_sha256=P1_H1_RECEIPT_SHA256,
    )
    h0_samples = _validate_p1_summary(h0_summary, arm="h0", job_id="53158")
    h1_samples = _validate_p1_summary(h1_summary, arm="h1", job_id="53159")
    if list(h0_samples) != SAMPLES or list(h1_samples) != SAMPLES:
        raise ValidationError("P1 paired summary sample inventory differs")
    for arm, receipt, rollout_path, rollout_hash in (
        ("h0", h0_receipt, h0_rollout_path, P1_H0_ROLLOUT_SHA256),
        ("h1", h1_receipt, h1_rollout_path, P1_H1_ROLLOUT_SHA256),
    ):
        rollout_binding = require_mapping(
            require_mapping(receipt.get("artifacts"), f"p1.{arm}.artifacts").get(
                "rollout_marker"
            ),
            f"p1.{arm}.rollout_marker",
        )
        logical_rollout_path = (
            P1_H0_OUTPUT if arm == "h0" else P1_H1_OUTPUT
        ) / "run/rollout_complete.json"
        if (
            rollout_binding.get("path") != str(logical_rollout_path)
            or rollout_binding.get("sha256") != rollout_hash
            or rollout_path
            != materialize_p1_file(
                logical_rollout_path, f"p1.{arm}.rollout_marker.expected"
            )
        ):
            raise ValidationError(
                f"P1 {arm.upper()} receipt/rollout binding differs"
            )

    require_exact_identity(
        source.get("pair_comparison_receipt"),
        P1_H1_OUTPUT / "pair_comparison_receipt.json",
        P1_PAIR_SHA256,
        "source_h1_retry3.pair_comparison_receipt",
    )
    pair_path = resolve_identity(
        source.get("pair_comparison_receipt"),
        Path.cwd(),
        "source_h1_retry3.pair_comparison_receipt",
        verify_hash=True,
    )
    pair = read_json(pair_path)
    if (
        pair.get("schema_id") != "qwen3vl_h0_h1_pair_comparison_v1"
        or pair.get("status") != "complete"
        or pair.get("evaluation_profile") != "e2e3"
        or pair.get("evaluation_id") != "DEV-CORE-3-V1"
        or pair.get("sample_order") != SAMPLES
        or pair.get("implementation_commit") != P1_IMPLEMENTATION_COMMIT
        or pair.get("freeze_commit") != SOURCE_COMMIT
    ):
        raise ValidationError("P1 pair receipt identity differs")
    arms = require_mapping(pair.get("arms"), "p1.pair.arms")
    for arm, expected_path, expected_hash, expected_job, summary_path in (
        ("h0", h0_receipt_path, P1_H0_RECEIPT_SHA256, "53158", h0_summary_path),
        ("h1", h1_receipt_path, P1_H1_RECEIPT_SHA256, "53159", h1_summary_path),
    ):
        arm_record = require_mapping(arms.get(arm), f"p1.pair.arms.{arm}")
        receipt_binding = require_mapping(
            arm_record.get("receipt"), f"p1.pair.arms.{arm}.receipt"
        )
        logical_output_root = P1_H0_OUTPUT if arm == "h0" else P1_H1_OUTPUT
        logical_receipt_path = logical_output_root / "evaluation_receipt.json"
        logical_summary_path = logical_output_root / "baseline_summary.json"
        if (
            receipt_binding.get("path") != str(logical_receipt_path)
            or receipt_binding.get("sha256") != expected_hash
            or expected_path
            != materialize_p1_file(
                logical_receipt_path, f"p1.pair.arms.{arm}.receipt.expected"
            )
            or arm_record.get("job_id") != expected_job
            or arm_record.get("experiment_arm_id") != P1_ARM_IDS[arm]
            or arm_record.get("harness_protocol_id") != P1_PROTOCOL_IDS[arm]
        ):
            raise ValidationError(f"P1 pair {arm.upper()} binding differs")
        summary_binding = require_mapping(
            arm_record.get("baseline_summary"),
            f"p1.pair.arms.{arm}.baseline_summary",
        )
        if (
            summary_binding.get("path") != str(logical_summary_path)
            or summary_binding.get("sha256") != sha256_file(summary_path)
            or summary_path
            != materialize_p1_file(
                logical_summary_path,
                f"p1.pair.arms.{arm}.baseline_summary.expected",
            )
        ):
            raise ValidationError(f"P1 pair {arm.upper()} summary binding differs")
    controlled = require_mapping(
        pair.get("controlled_variable_checks"), "p1.pair.controlled_variable_checks"
    )
    hard_gates = require_mapping(pair.get("hard_gates"), "p1.pair.hard_gates")
    if not controlled or any(value is not True for value in controlled.values()):
        raise ValidationError("P1 pair controlled-variable checks failed")
    if not hard_gates or any(value is not True for value in hard_gates.values()):
        raise ValidationError("P1 pair hard gates failed")

    comparison = require_mapping(pair.get("comparison"), "p1.pair.comparison")
    h0_comp = require_mapping(comparison.get("h0"), "p1.pair.comparison.h0")
    h1_comp = require_mapping(comparison.get("h1"), "p1.pair.comparison.h1")
    for metric in ("initial_iou", "initial_f1", "selected_iou", "selected_f1"):
        if h0_comp.get(metric) != h1_comp.get(metric):
            raise ValidationError(f"P1 paired aggregate {metric} differs")
    if (
        comparison.get("diagnostic_delta_selected_iou_h1_minus_h0") != 0.0
        or comparison.get("diagnostic_delta_selected_f1_h1_minus_h0") != 0.0
    ):
        raise ValidationError("P1 paired aggregate selected-state metrics differ")
    h0_metrics_binding = require_mapping(
        require_mapping(h0_receipt.get("artifacts"), "p1.h0.artifacts").get(
            "offline_scorer_input"
        ),
        "p1.h0.offline_scorer_input",
    )
    h1_metrics_binding = require_mapping(
        require_mapping(h1_receipt.get("artifacts"), "p1.h1.artifacts").get(
            "offline_scorer_input"
        ),
        "p1.h1.offline_scorer_input",
    )
    if (
        h0_metrics_binding.get("sha256") != h1_metrics_binding.get("sha256")
        or h0_metrics_binding.get("sha256")
        != "<hash omitted>"
    ):
        raise ValidationError("P1 paired offline metrics artifacts differ")
    metrics_payloads: dict[str, Mapping[str, Any]] = {}
    for arm, binding, output_root in (
        ("h0", h0_metrics_binding, P1_H0_OUTPUT),
        ("h1", h1_metrics_binding, P1_H1_OUTPUT),
    ):
        logical_metrics_path = output_root / "run/per_sample_metrics.json"
        metrics_path = exact_materialized_p1_file(
            binding.get("path"),
            logical_metrics_path,
            f"p1.{arm}.offline_scorer_input.path",
        )
        if (
            metrics_path
            != materialize_p1_file(
                logical_metrics_path, f"p1.{arm}.offline_scorer_input.expected"
            )
            or binding.get("sha256") != sha256_file(metrics_path)
        ):
            raise ValidationError(f"P1 {arm.upper()} offline metrics binding differs")
        metrics_payloads[arm] = read_json(metrics_path)
    if metrics_payloads["h0"] != metrics_payloads["h1"]:
        raise ValidationError("P1 H0/H1 selected-step metrics payloads differ")
    metric_samples = require_mapping(
        metrics_payloads["h1"].get("samples"), "p1.h1.metrics.samples"
    )
    for sample_name in SAMPLES:
        if _selected_signature(
            h0_samples[sample_name], f"p1.h0.{sample_name}"
        ) != _selected_signature(h1_samples[sample_name], f"p1.h1.{sample_name}"):
            raise ValidationError(f"P1 paired selected metrics differ for {sample_name}")
        metric_record = require_mapping(
            metric_samples.get(sample_name), f"p1.h1.metrics.{sample_name}"
        )
        if metric_record.get("verifier_selected_step") != 0:
            raise ValidationError(f"P1 H1 selected state is not step zero for {sample_name}")
        selected = dict(
            require_mapping(
                metric_record.get("verifier_selected"),
                f"p1.h1.metrics.{sample_name}.verifier_selected",
            )
        )
        selected.pop("step_index", None)
        if tuple(selected.get(key) for key in ("iou", "precision", "recall", "f1", "tp", "fp", "fn", "tn")) != _selected_signature(
            h1_samples[sample_name], f"p1.h1.{sample_name}"
        ):
            raise ValidationError(f"P1 H1 selected-state metrics drifted for {sample_name}")


def validate_file_bindings(
    prereg: Mapping[str, Any], repo_root: Path, *, labels: bool
) -> None:
    h0_source = require_mapping(prereg["source_h0_retry3"], "source_h0_retry3")
    for field in ("rollout_complete", "evaluation_receipt"):
        resolve_identity(
            h0_source[field],
            repo_root,
            f"source_h0_retry3.{field}",
            verify_hash=True,
        )
    source = require_mapping(prereg["source_h1_retry3"], "source_h1_retry3")
    for field in (
        "rollout_complete",
        "evaluation_receipt",
        "pair_comparison_receipt",
    ):
        resolve_identity(
            source[field],
            repo_root,
            f"source_h1_retry3.{field}",
            verify_hash=True,
        )
    validate_h1_pair_source(prereg)
    runtime = require_mapping(prereg["runtime_files"], "runtime_files")
    resolve_identity(runtime["action_generator"], repo_root, "action_generator", verify_hash=True)
    resolve_identity(runtime["scorer"], repo_root, "scorer", verify_hash=True)
    for index, record in enumerate(runtime["generation_runtime_sources"]):
        resolve_identity(record, repo_root, f"runtime_source[{index}]", verify_hash=True)
    tools = require_mapping(prereg["tools"], "tools")
    workspace = require_mapping(tools["workspace_root"], "workspace_root")
    if not Path(str(workspace.get("path"))).resolve().is_dir():
        raise ValidationError("workspace root is missing")
    for name in ("segagent_python", "omniovcd_python", "simpleclick_checkpoint", "sam3_checkpoint", "sam3_bpe"):
        resolve_identity(tools[name], repo_root, f"tools.{name}", verify_hash=True)
    if labels:
        for name in SAMPLES:
            resolve_identity(
                prereg["samples"][name]["offline_label"],
                repo_root,
                f"samples.{name}.offline_label",
                verify_hash=True,
            )


def validate_execution(prereg: Mapping[str, Any], execution: Mapping[str, Any]) -> None:
    expected = expected_execution(prereg)
    if execution != expected:
        raise ValidationError("GT-free execution manifest is not the exact preregistration projection")
    assert_gt_free(execution)


def inside(root: Path, relative: Any, location: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValidationError(f"{location} must be a relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"{location} escapes candidate root") from error
    return path


def verified_relative_file(root: Path, spec: Any, location: str) -> Path:
    record = require_mapping(spec, location)
    path = inside(root, record.get("path"), f"{location}.path")
    if not path.is_file() or sha256_file(path) != require_sha(record.get("sha256"), f"{location}.sha256"):
        raise ValidationError(f"{location} file/hash differs")
    return path


def verified_mask(
    root: Path,
    spec: Any,
    location: str,
) -> tuple[Path, str]:
    record = require_mapping(spec, location)
    path = verified_relative_file(root, record, location)
    mask = np.asarray(np.load(path), dtype=bool)
    semantic_hash = mask_sha256(mask)
    if semantic_hash != require_sha(record.get("mask_sha256"), f"{location}.mask_sha256"):
        raise ValidationError(f"{location} semantic hash differs")
    shape = record.get("shape")
    if shape is not None and shape != list(mask.shape):
        raise ValidationError(f"{location} shape differs")
    return path, semantic_hash


def verify_generation_before_labels(
    candidate_root: Path,
    execution_path: Path,
    execution: Mapping[str, Any],
    expected_runtime_commit: str,
) -> tuple[Path, Path]:
    if COMMIT_RE.fullmatch(expected_runtime_commit) is None:
        raise ValidationError("expected P2 runtime commit is malformed")
    root = candidate_root.resolve(strict=True)
    execution_path = execution_path.resolve(strict=True)
    execution_raw_hash = sha256_file(execution_path)
    execution_canonical_hash = canonical_sha256(execution)
    preregistration_canonical_hash = require_sha(
        execution.get("preregistration_canonical_sha256"),
        "execution.preregistration_canonical_sha256",
    )
    marker_path = root / "candidate_generation_complete.json"
    if not marker_path.is_file():
        raise ValidationError("generation complete marker is absent")
    marker = read_json(marker_path)
    if (
        marker.get("schema_id") != COMPLETE_SCHEMA
        or marker.get("execution_status") != "completed"
        or marker.get("gt_loaded") is not False
        or marker.get("ordered_samples") != SAMPLES
        or marker.get("counts") != COUNTS
        or marker.get("p2_runtime_commit") != expected_runtime_commit
        or marker.get("preregistration_canonical_sha256")
        != preregistration_canonical_hash
        or marker.get("execution_manifest", {}).get("sha256")
        != execution_raw_hash
    ):
        raise ValidationError("generation complete marker differs")
    index_path = verified_relative_file(root, marker.get("candidate_index"), "marker.candidate_index")
    index = read_json(index_path)
    if (
        index.get("schema_id") != INDEX_SCHEMA
        or index.get("gt_loaded") is not False
        or index.get("ordered_samples") != SAMPLES
        or index.get("counts") != COUNTS
        or index.get("p2_runtime_commit") != expected_runtime_commit
        or index.get("source_freeze_commit") != SOURCE_COMMIT
        or index.get("preregistration_canonical_sha256")
        != preregistration_canonical_hash
        or index.get("execution_manifest", {}).get("sha256") != execution_raw_hash
        or index.get("execution_manifest", {}).get("canonical_sha256")
        != execution_canonical_hash
    ):
        raise ValidationError("candidate index identity differs")
    index_execution_path = index.get("execution_manifest", {}).get("path")
    if (
        not isinstance(index_execution_path, str)
        or Path(index_execution_path).resolve(strict=True) != execution_path
    ):
        raise ValidationError("candidate index execution path differs")

    initial_states = require_mapping(index.get("initial_states"), "initial_states")
    if list(initial_states) != SAMPLES:
        raise ValidationError("initial-state inventory differs")
    for sample_name in SAMPLES:
        state = require_mapping(initial_states[sample_name], f"initial_states.{sample_name}")
        expected_hashes = execution["samples"][sample_name]["initial_state_mask_sha256"]
        for output_name, expected_name in (
            ("t1_mask", "t1"),
            ("t2_mask", "t2"),
            ("change_mask", "change"),
        ):
            _, observed_hash = verified_mask(
                root,
                state.get(output_name),
                f"initial_states.{sample_name}.{output_name}",
            )
            if observed_hash != expected_hashes[expected_name]:
                raise ValidationError(
                    f"initial state {sample_name} {expected_name} differs from execution"
                )

    records = index.get("candidates")
    if not isinstance(records, list) or len(records) != COUNTS["total"]:
        raise ValidationError("candidate record inventory differs")
    expected_actions = [
        action
        for sample_name in SAMPLES
        for action in execution["samples"][sample_name]["actions"]
    ]
    if len(expected_actions) != COUNTS["total"]:
        raise ValidationError("execution action inventory differs")
    seen: set[str] = set()
    tiers = {"primary": 0, "diagnostic_executor_only": 0}
    for number, identity in enumerate(records):
        identity = require_mapping(identity, f"candidates[{number}]")
        record_path = inside(root, identity.get("record_path"), f"candidate[{number}].record_path")
        if not record_path.is_file() or sha256_file(record_path) != identity.get("record_sha256"):
            raise ValidationError(f"candidate record {number} hash differs")
        record = read_json(record_path)
        expected_action = require_mapping(
            expected_actions[number], f"expected_actions[{number}]"
        )
        candidate_id = record.get("candidate_id")
        tier = record.get("tier")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in seen
            or candidate_id != identity.get("candidate_id")
            or record.get("sample") != identity.get("sample")
            or tier != identity.get("tier")
            or tier not in tiers
            or any(record.get(key) != value for key, value in expected_action.items())
        ):
            raise ValidationError(f"candidate record {number} identity differs")
        if record.get("initial_state_mask_sha256") != execution["samples"][
            record["sample"]
        ]["initial_state_mask_sha256"]:
            raise ValidationError(f"candidate {candidate_id} initial-state binding differs")
        seen.add(candidate_id)
        tiers[tier] += 1
        no_qwen = require_mapping(
            record.get("no_qwen_audit"), f"candidate.{candidate_id}.no_qwen_audit"
        )
        if no_qwen.get("qwen_invoked") is not False:
            raise ValidationError(f"candidate {candidate_id} reports a Qwen call")
        gate = require_mapping(record.get("hard_gate"), f"candidate.{candidate_id}.hard_gate")
        passed = gate.get("passed")
        rejected = gate.get("rejected")
        if (
            not isinstance(passed, bool)
            or not isinstance(rejected, bool)
            or passed == rejected
            or gate.get("verifier_skipped_by_hard_gate") != rejected
            or no_qwen.get("decision_path")
            != ("pre_qwen_hard_gate" if rejected else "eligible_for_qwen_not_evaluated")
        ):
            raise ValidationError(f"candidate {candidate_id} hard gate booleans differ")
        tool_evidence = require_mapping(
            record.get("tool_evidence"), f"candidate.{candidate_id}.tool_evidence"
        )
        expected_tool = "sam3" if record.get("action") == "box" else "simpleclick"
        if (
            tool_evidence.get("tool") != expected_tool
            or not isinstance(tool_evidence.get("tool_input"), Mapping)
        ):
            raise ValidationError(f"candidate {candidate_id} tool evidence differs")
        observed_state_hashes: dict[str, str] = {}
        for view in ("t1", "t2", "change"):
            mask_spec = require_mapping(record.get("candidate_masks"), "candidate_masks").get(view)
            _, observed_state_hashes[view] = verified_mask(
                root, mask_spec, f"candidate.{candidate_id}.{view}"
            )
        if (
            record.get("candidate_state_mask_sha256") != observed_state_hashes
            or identity.get("candidate_state_mask_sha256") != observed_state_hashes
        ):
            raise ValidationError(f"candidate {candidate_id} state hash binding differs")
    if tiers != {"primary": 27, "diagnostic_executor_only": 9}:
        raise ValidationError("candidate tier counts differ")
    return marker_path, index_path


def runtime_scoring_manifest(
    prereg_path: Path,
    prereg: Mapping[str, Any],
    execution_path: Path,
    execution: Mapping[str, Any],
    marker_path: Path,
    index_path: Path,
    runtime_commit: str,
) -> dict[str, Any]:
    return {
        "schema_id": SCORING_SCHEMA,
        "status": "frozen",
        "preregistration": {
            "path": str(prereg_path.resolve()),
            "sha256": sha256_file(prereg_path.resolve()),
            "canonical_sha256": canonical_sha256(prereg),
        },
        "execution_manifest": {
            "path": str(execution_path.resolve()),
            "sha256": sha256_file(execution_path.resolve()),
            "canonical_sha256": canonical_sha256(execution),
        },
        "execution_manifest_sha256": sha256_file(execution_path.resolve()),
        "preregistration_canonical_sha256": canonical_sha256(prereg),
        "p2_runtime_commit": runtime_commit,
        "candidate_generation_complete_sha256": sha256_file(marker_path),
        "candidate_index_sha256": sha256_file(index_path),
        "ordered_samples": SAMPLES,
        "target_samples": TARGETS,
        "expected_candidate_counts": COUNTS,
        "candidate_flow_counts": {
            "requested": COUNTS["total"],
            "generated": COUNTS["total"],
            "tool_success": COUNTS["total"],
        },
        "comparison_epsilon": 1e-12,
        "labels": {
            name: prereg["samples"][name]["offline_label"] for name in SAMPLES
        },
        "gt_access_boundary": {
            "generation_verified_before_manifest_creation": True,
            "all_candidate_records_and_masks_verified_before_manifest_creation": True,
            "runtime_scoring_manifest_not_available_to_generator": True,
        },
    }


def write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise ValidationError(f"refusing to overwrite runtime scoring manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        # The scorer requires labels to preserve ordered_samples insertion order.
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--phase", choices=("structure", "frozen", "post_generation"), default="structure")
    parser.add_argument("--expected-preregistration-canonical-sha256")
    parser.add_argument("--expected-execution-canonical-sha256")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--runtime-scoring-output", type=Path)
    parser.add_argument("--expected-runtime-commit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prereg_path = args.preregistration.resolve(strict=True)
        repo_root = args.repo_root.resolve(strict=True)
        prereg = read_json(prereg_path)
        validate_structure(prereg)
        validate_inventory(prereg, repo_root, verify_files=args.phase != "structure")
        observed_prereg = canonical_sha256(prereg)
        if args.expected_preregistration_canonical_sha256 is not None and (
            observed_prereg != require_sha(
                args.expected_preregistration_canonical_sha256,
                "expected preregistration canonical SHA256",
            )
        ):
            raise ValidationError("preregistration canonical SHA256 differs")
        execution: dict[str, Any] | None = None
        execution_path: Path | None = None
        if args.execution_manifest is not None:
            execution_path = args.execution_manifest.resolve(strict=True)
            execution = read_json(execution_path)
            validate_execution(prereg, execution)
            observed_execution = canonical_sha256(execution)
            if args.expected_execution_canonical_sha256 is not None and (
                observed_execution != require_sha(
                    args.expected_execution_canonical_sha256,
                    "expected execution canonical SHA256",
                )
            ):
                raise ValidationError("execution canonical SHA256 differs")
        elif args.phase != "structure":
            raise ValidationError("execution manifest is required outside structure phase")
        if args.phase == "frozen":
            validate_file_bindings(prereg, repo_root, labels=False)
        if args.phase == "post_generation":
            if (
                execution is None
                or execution_path is None
                or args.candidate_root is None
                or args.expected_runtime_commit is None
            ):
                raise ValidationError(
                    "post_generation requires execution manifest, candidate root, and runtime commit"
                )
            marker_path, index_path = verify_generation_before_labels(
                args.candidate_root,
                execution_path,
                execution,
                args.expected_runtime_commit,
            )
            # This is deliberately the first point at which label files are opened.
            validate_file_bindings(prereg, repo_root, labels=True)
            if args.runtime_scoring_output is None:
                raise ValidationError("post_generation requires runtime scoring output")
            write_new(
                args.runtime_scoring_output,
                runtime_scoring_manifest(
                    prereg_path,
                    prereg,
                    execution_path,
                    execution,
                    marker_path,
                    index_path,
                    args.expected_runtime_commit,
                ),
            )
    except (OSError, ValidationError) as error:
        print(f"qwen3vl_p2_preregistration: {error}", file=sys.stderr)
        return 3
    print(
        "status=valid;"
        f"phase={args.phase};"
        f"preregistration_canonical_sha256={canonical_sha256(prereg)};"
        f"actions={COUNTS['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
