#!/usr/bin/env python3
"""Score completed P2 candidates after proving that GT-free generation is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_ID = "qwen3vl_p2_oracle_scoring_manifest_v1"
COMPLETE_SCHEMA_ID = "qwen3vl_p2_candidate_generation_complete_v1"
CANDIDATE_INDEX_SCHEMA_ID = "qwen3vl_p2_oracle_candidates_v1"
REPORT_SCHEMA_ID = "qwen3vl_p2_oracle_score_report_v1"
DEFAULT_EPSILON = 1e-12
WILSON_Z_95 = 1.959963984540054
STATE_IDENTITY_POLICY = "sample+t1_mask_sha256+t2_mask_sha256+change_mask_sha256"
CORE3_ORDERED_SAMPLES = ("test_20_15", "test_1_1", "test_50_8")
CORE3_TARGET_SAMPLES = ("test_1_1", "test_50_8")
CORE3_CANDIDATE_COUNTS = {
    "total": 36,
    "primary": 27,
    "diagnostic_executor_only": 9,
}


@dataclass(frozen=True)
class VerifiedGeneration:
    marker: dict[str, Any]
    index: dict[str, Any]
    initial_masks: dict[str, np.ndarray]
    candidates: tuple[tuple[dict[str, Any], np.ndarray], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scoring output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _inside(root: Path, relative: Any, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes candidate root") from error
    return path


def _verified_file(root: Path, spec: Any, *, field: str) -> Path:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{field} must be a file identity object")
    path = _inside(root, spec.get("path"), field=f"{field}.path")
    if not path.is_file():
        raise FileNotFoundError(f"{field} missing: {path}")
    expected = spec.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"{field} SHA256 mismatch")
    return path


def _verified_mask(root: Path, spec: Any, *, field: str) -> np.ndarray:
    path = _verified_file(root, spec, field=field)
    value = np.asarray(np.load(path), dtype=bool)
    expected_mask_hash = spec.get("mask_sha256")
    if not isinstance(expected_mask_hash, str) or mask_sha256(value) != expected_mask_hash:
        raise ValueError(f"{field} semantic mask SHA256 mismatch")
    shape = spec.get("shape")
    if shape is not None and list(value.shape) != shape:
        raise ValueError(f"{field} shape mismatch")
    return value


def validate_scoring_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_id") != SCHEMA_ID or manifest.get("status") != "frozen":
        raise ValueError("scoring manifest must be the frozen P2 v1 schema")
    if not isinstance(manifest.get("labels"), Mapping):
        raise ValueError("scoring manifest labels mapping is required")
    ordered = manifest.get("ordered_samples")
    if not isinstance(ordered, list) or ordered != list(manifest["labels"]):
        raise ValueError("labels must preserve the exact ordered sample list")
    if len(ordered) != len(set(ordered)):
        raise ValueError("ordered samples must be unique")
    if ordered != list(CORE3_ORDERED_SAMPLES):
        raise ValueError("P2 scorer requires the exact frozen CORE3 sample order")
    if manifest.get("target_samples") != list(CORE3_TARGET_SAMPLES):
        raise ValueError("P2 scorer requires the exact frozen target sample list")
    if manifest.get("expected_candidate_counts") != CORE3_CANDIDATE_COUNTS:
        raise ValueError("P2 scorer requires exactly 27 primary plus 9 diagnostic candidates")
    expected_flow_counts = {
        "requested": CORE3_CANDIDATE_COUNTS["total"],
        "generated": CORE3_CANDIDATE_COUNTS["total"],
        "tool_success": CORE3_CANDIDATE_COUNTS["total"],
    }
    if manifest.get("candidate_flow_counts") != expected_flow_counts:
        raise ValueError("P2 scorer requires 36 successful frozen tool executions")


def normalize_runtime_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("p2_runtime_commit must be a full 40-character Git SHA")
    return value.lower()


def verify_candidate_audit(record: Mapping[str, Any], candidate_id: str) -> None:
    action = record.get("action")
    if action == "box":
        expected_tool, expected_worker_mode = "sam3", "box"
    elif action in {"positive_point", "negative_point"}:
        expected_tool, expected_worker_mode = "simpleclick", "point"
    else:
        raise ValueError(f"candidate {candidate_id} has an unsupported action")

    gate = record.get("hard_gate")
    no_qwen = record.get("no_qwen_audit")
    if not isinstance(gate, Mapping) or not isinstance(no_qwen, Mapping):
        raise ValueError(f"candidate {candidate_id} gate/no-Qwen audit is malformed")
    passed = gate.get("passed")
    rejected = gate.get("rejected")
    initial_stub_calls = no_qwen.get("initial_stub_calls")
    candidate_stub_calls = no_qwen.get("candidate_stub_calls")
    if (
        not isinstance(passed, bool)
        or not isinstance(rejected, bool)
        or passed == rejected
        or gate.get("verifier_skipped_by_hard_gate") != rejected
        or no_qwen.get("qwen_invoked") is not False
        or type(initial_stub_calls) is not int
        or initial_stub_calls != 1
        or type(candidate_stub_calls) is not int
        or candidate_stub_calls != (0 if rejected else 1)
        or no_qwen.get("decision_path")
        != ("pre_qwen_hard_gate" if rejected else "eligible_for_qwen_not_evaluated")
    ):
        raise ValueError(f"candidate {candidate_id} gate/no-Qwen audit is malformed")
    hard_reasons = gate.get("rejection_reasons")
    if (
        not isinstance(hard_reasons, list)
        or (rejected and not hard_reasons)
        or (not rejected and hard_reasons)
    ):
        raise ValueError(f"candidate {candidate_id} hard-gate reasons are malformed")

    evidence = record.get("tool_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"candidate {candidate_id} lacks tool success evidence")
    tool_result = evidence.get("tool_result")
    worker_report = (
        tool_result.get("worker_report") if isinstance(tool_result, Mapping) else None
    )
    verifier_evidence = evidence.get("verifier_evidence")
    expected_verifier_type = (
        "runtime_hard_gate_precheck"
        if rejected
        else "p2_no_qwen_reject_verifier"
    )
    if (
        evidence.get("tool") != expected_tool
        or not isinstance(evidence.get("tool_input"), Mapping)
        or not isinstance(tool_result, Mapping)
        or tool_result.get("worker_mode") != expected_worker_mode
        or not isinstance(worker_report, Mapping)
        or worker_report.get("status") != "success"
        or worker_report.get("mode") != expected_worker_mode
        or evidence.get("verifier_skipped_by_hard_gate") != rejected
        or evidence.get("candidate_accepted") is not False
        or not isinstance(verifier_evidence, Mapping)
        or verifier_evidence.get("type") != expected_verifier_type
    ):
        raise ValueError(f"candidate {candidate_id} tool success evidence is malformed")
    if rejected:
        if (
            verifier_evidence.get("verifier_skipped") is not True
            or verifier_evidence.get("rejection_reasons") != hard_reasons
        ):
            raise ValueError(f"candidate {candidate_id} hard-gate evidence is malformed")
    elif verifier_evidence.get("qwen_invoked") is not False:
        raise ValueError(f"candidate {candidate_id} no-Qwen evidence is malformed")


def verify_generation_before_gt(
    scoring_manifest: Mapping[str, Any], candidate_root: Path
) -> VerifiedGeneration:
    """Validate marker, index, records, and every candidate mask before label I/O."""

    candidate_root = candidate_root.resolve()
    marker_path = candidate_root / "candidate_generation_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError("candidate generation complete marker is missing")
    expected_marker_hash = scoring_manifest.get("candidate_generation_complete_sha256")
    if not isinstance(expected_marker_hash, str) or sha256_file(marker_path) != expected_marker_hash:
        raise ValueError("candidate generation complete marker SHA256 mismatch")
    marker = _read_json(marker_path)
    if (
        marker.get("schema_id") != COMPLETE_SCHEMA_ID
        or marker.get("execution_status") != "completed"
        or marker.get("gt_loaded") is not False
        or marker.get("ordered_samples") != scoring_manifest.get("ordered_samples")
    ):
        raise ValueError("candidate generation marker is incomplete or not GT-free")

    expected_counts = scoring_manifest.get("expected_candidate_counts")
    marker_counts = marker.get("counts")
    if not isinstance(expected_counts, Mapping) or marker_counts != expected_counts:
        raise ValueError("marker/scoring candidate count binding mismatch")

    index_path = _verified_file(
        candidate_root, marker.get("candidate_index"), field="marker.candidate_index"
    )
    expected_index_hash = scoring_manifest.get("candidate_index_sha256")
    if not isinstance(expected_index_hash, str) or sha256_file(index_path) != expected_index_hash:
        raise ValueError("candidate index SHA256 mismatch against scoring manifest")
    index = _read_json(index_path)
    if (
        index.get("schema_id") != CANDIDATE_INDEX_SCHEMA_ID
        or index.get("gt_loaded") is not False
        or index.get("ordered_samples") != scoring_manifest.get("ordered_samples")
    ):
        raise ValueError("candidate index identity mismatch")
    index_counts = index.get("counts")
    if not isinstance(index_counts, Mapping) or index_counts != marker_counts:
        raise ValueError("marker/index/scoring candidate count binding mismatch")
    expected_execution_hash = scoring_manifest.get("execution_manifest_sha256")
    marker_execution_identity = marker.get("execution_manifest")
    index_execution_identity = index.get("execution_manifest")
    if (
        not isinstance(expected_execution_hash, str)
        or not isinstance(marker_execution_identity, Mapping)
        or not isinstance(index_execution_identity, Mapping)
        or marker_execution_identity.get("sha256") != expected_execution_hash
        or index_execution_identity.get("sha256") != expected_execution_hash
    ):
        raise ValueError("execution manifest binding mismatch")
    expected_preregistration_hash = scoring_manifest.get(
        "preregistration_canonical_sha256"
    )
    if (
        not isinstance(expected_preregistration_hash, str)
        or marker.get("preregistration_canonical_sha256")
        != expected_preregistration_hash
        or index.get("preregistration_canonical_sha256")
        != expected_preregistration_hash
    ):
        raise ValueError("preregistration canonical binding mismatch")
    expected_runtime_commit = normalize_runtime_commit(
        scoring_manifest.get("p2_runtime_commit")
    )
    if (
        marker.get("p2_runtime_commit") != expected_runtime_commit
        or index.get("p2_runtime_commit") != expected_runtime_commit
    ):
        raise ValueError("P2 runtime commit binding mismatch")

    initial_specs = index.get("initial_states")
    if not isinstance(initial_specs, Mapping) or list(initial_specs) != scoring_manifest["ordered_samples"]:
        raise ValueError("initial-state inventory mismatch")
    initial_masks: dict[str, np.ndarray] = {}
    for sample in scoring_manifest["ordered_samples"]:
        state_specs = initial_specs[sample]
        if not isinstance(state_specs, Mapping):
            raise ValueError(f"initial state for {sample} is malformed")
        # Verify all three masks, even though only change is scored.
        _verified_mask(candidate_root, state_specs.get("t1_mask"), field=f"initial.{sample}.t1")
        _verified_mask(candidate_root, state_specs.get("t2_mask"), field=f"initial.{sample}.t2")
        initial_masks[sample] = _verified_mask(
            candidate_root, state_specs.get("change_mask"), field=f"initial.{sample}.change"
        )

    raw_candidates = index.get("candidates")
    counts = index_counts
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate inventory/counts are malformed")
    if counts.get("total") != len(raw_candidates):
        raise ValueError("candidate count binding mismatch")
    verified: list[tuple[dict[str, Any], np.ndarray]] = []
    seen: set[str] = set()
    tier_counts = {"primary": 0, "diagnostic_executor_only": 0}
    for index_number, identity in enumerate(raw_candidates):
        if not isinstance(identity, Mapping):
            raise ValueError(f"candidate identity {index_number} is malformed")
        record_path = _inside(
            candidate_root, identity.get("record_path"), field=f"candidates[{index_number}].record_path"
        )
        if not record_path.is_file() or sha256_file(record_path) != identity.get("record_sha256"):
            raise ValueError(f"candidate record {index_number} SHA256 mismatch")
        record = _read_json(record_path)
        candidate_id = record.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in seen
            or candidate_id != identity.get("candidate_id")
            or record.get("sample") != identity.get("sample")
            or record.get("tier") != identity.get("tier")
        ):
            raise ValueError(f"candidate record {index_number} identity mismatch")
        seen.add(candidate_id)
        tier = record.get("tier")
        if tier not in tier_counts:
            raise ValueError(f"candidate {candidate_id} has unsupported tier")
        tier_counts[tier] += 1
        if record.get("sample") not in initial_masks:
            raise ValueError(f"candidate {candidate_id} names an unknown sample")
        mask_specs = record.get("candidate_masks")
        if not isinstance(mask_specs, Mapping):
            raise ValueError(f"candidate {candidate_id} lacks masks")
        _verified_mask(candidate_root, mask_specs.get("t1"), field=f"candidate.{candidate_id}.t1")
        _verified_mask(candidate_root, mask_specs.get("t2"), field=f"candidate.{candidate_id}.t2")
        change = _verified_mask(
            candidate_root, mask_specs.get("change"), field=f"candidate.{candidate_id}.change"
        )
        state_hashes = record.get("candidate_state_mask_sha256")
        expected_state_hashes = {
            name: mask_specs.get(name, {}).get("mask_sha256")
            if isinstance(mask_specs.get(name), Mapping)
            else None
            for name in ("t1", "t2", "change")
        }
        if (
            state_hashes != expected_state_hashes
            or identity.get("candidate_state_mask_sha256") != expected_state_hashes
            or any(not isinstance(value, str) for value in expected_state_hashes.values())
        ):
            raise ValueError(f"candidate {candidate_id} semantic state hash mismatch")
        verify_candidate_audit(record, candidate_id)
        verified.append((record, change))
    if tier_counts != {
        "primary": int(counts.get("primary", -1)),
        "diagnostic_executor_only": int(counts.get("diagnostic_executor_only", -1)),
    }:
        raise ValueError("candidate tier counts mismatch")
    return VerifiedGeneration(marker, index, initial_masks, tuple(verified))


def metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, int | float]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if prediction.shape != reference.shape:
        raise ValueError("prediction/reference shapes differ")
    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, ~reference).sum())
    fn = int(np.logical_and(~prediction, reference).sum())
    tn = int(np.logical_and(~prediction, ~reference).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def classify_candidate(
    initial: Mapping[str, int | float],
    candidate: Mapping[str, int | float],
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[str, dict[str, int | float]]:
    delta_iou = float(candidate["iou"]) - float(initial["iou"])
    delta_f1 = float(candidate["f1"]) - float(initial["f1"])
    deltas: dict[str, int | float] = {
        "iou": delta_iou,
        "f1": delta_f1,
        "tp": int(candidate["tp"]) - int(initial["tp"]),
        "fp": int(candidate["fp"]) - int(initial["fp"]),
        "fn": int(candidate["fn"]) - int(initial["fn"]),
    }
    if delta_iou > epsilon and delta_f1 > epsilon:
        return "beneficial", deltas
    tied = abs(delta_iou) <= epsilon and abs(delta_f1) <= epsilon
    worse_counts = deltas["tp"] < 0 or deltas["fp"] > 0 or deltas["fn"] > 0
    if delta_iou < -epsilon or delta_f1 < -epsilon or (tied and worse_counts):
        return "harmful", deltas
    return "indeterminate", deltas


def gate_leaf(oracle_label: str, hard_gate_rejected: bool) -> str:
    return {
        ("beneficial", True): "hard_gate_false_reject",
        ("beneficial", False): "beneficial_gate_pass",
        ("harmful", True): "hard_gate_correct_reject",
        ("harmful", False): "harmful_gate_pass",
        ("indeterminate", True): "indeterminate_hard_gate_rejected",
        ("indeterminate", False): "indeterminate_gate_pass",
    }[(oracle_label, hard_gate_rejected)]


def wilson_interval(numerator: int, denominator: int, z: float = WILSON_Z_95) -> dict[str, Any]:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("invalid Wilson numerator/denominator")
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": None,
            "wilson95_lower": None,
            "wilson95_upper": None,
        }
    rate = numerator / denominator
    z2 = z * z
    scale = 1 + z2 / denominator
    center = (rate + z2 / (2 * denominator)) / scale
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / denominator + z2 / (4 * denominator * denominator))
        / scale
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "wilson95_lower": max(0.0, center - radius),
        "wilson95_upper": min(1.0, center + radius),
    }


def summarize_gate_confusion(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    oracle_counts = {"beneficial": 0, "harmful": 0, "indeterminate": 0}
    leaf_counts: dict[str, int] = {}
    hard_gate_count = 0
    for item in scored:
        oracle_label = str(item["oracle_label"])
        leaf = str(item["gate_leaf"])
        oracle_counts[oracle_label] += 1
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
        hard_gate_count += int(bool(item["hard_gate"]["rejected"]))
    beneficial = oracle_counts["beneficial"]
    harmful = oracle_counts["harmful"]
    summary = {
        "candidate_count": len(scored),
        "oracle_counts": oracle_counts,
        "gate_leaf_counts": leaf_counts,
        "gate_rates_with_wilson95": {
            "hard_gate_false_reject_rate": wilson_interval(
                leaf_counts.get("hard_gate_false_reject", 0), beneficial
            ),
            "beneficial_gate_pass_rate": wilson_interval(
                leaf_counts.get("beneficial_gate_pass", 0), beneficial
            ),
            "hard_gate_correct_reject_rate": wilson_interval(
                leaf_counts.get("hard_gate_correct_reject", 0), harmful
            ),
            "hard_gate_before_qwen_rate": wilson_interval(
                hard_gate_count, len(scored)
            ),
        },
    }
    if all("candidate_state_mask_sha256" in item for item in scored):
        summary.update(summarize_candidate_states(scored))
    return summary


def candidate_state_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    hashes = record.get("candidate_state_mask_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("candidate state semantic hashes are missing")
    values = tuple(hashes.get(name) for name in ("t1", "t2", "change"))
    if any(not isinstance(value, str) for value in values):
        raise ValueError("candidate state semantic hashes are malformed")
    return (str(record["sample"]), *values)


def summarize_candidate_states(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_count = len(records)
    unique_count = len({candidate_state_identity(record) for record in records})
    return {
        "action_spec_count": action_count,
        "unique_candidate_state_count": unique_count,
        "candidate_state_duplicate_count": action_count - unique_count,
        "state_identity_policy": STATE_IDENTITY_POLICY,
    }


def _default_label_loader(spec: Mapping[str, Any], sample: str) -> np.ndarray:
    # This function is called only after verify_generation_before_gt returns.
    raw_path = spec.get("path")
    expected_hash = spec.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"label identity for {sample} is malformed")
    path = Path(raw_path).resolve()
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"label identity mismatch for {sample}")
    return np.asarray(Image.open(path)) > 0


def score_verified_generation(
    scoring_manifest: Mapping[str, Any],
    verified: VerifiedGeneration,
    *,
    label_loader: Callable[[Mapping[str, Any], str], np.ndarray] = _default_label_loader,
) -> dict[str, Any]:
    epsilon = float(scoring_manifest.get("comparison_epsilon", DEFAULT_EPSILON))
    labels: dict[str, np.ndarray] = {}
    initial_metrics: dict[str, dict[str, int | float]] = {}
    # First GT access in the scorer occurs here, after complete candidate verification.
    for sample in scoring_manifest["ordered_samples"]:
        labels[sample] = np.asarray(label_loader(scoring_manifest["labels"][sample], sample), dtype=bool)
        initial_metrics[sample] = metrics(verified.initial_masks[sample], labels[sample])

    scored: list[dict[str, Any]] = []
    for record, change_mask in verified.candidates:
        sample = str(record["sample"])
        candidate_metrics = metrics(change_mask, labels[sample])
        oracle_label, deltas = classify_candidate(
            initial_metrics[sample], candidate_metrics, epsilon
        )
        hard_rejected = bool(record["hard_gate"]["rejected"])
        leaf = gate_leaf(oracle_label, hard_rejected)
        scored.append(
            {
                "candidate_id": record["candidate_id"],
                "sample": sample,
                "tier": record["tier"],
                "region_id": record["region_id"],
                "action": record["action"],
                "target_view": record["target_view"],
                "candidate_state_mask_sha256": record[
                    "candidate_state_mask_sha256"
                ],
                "hard_gate": record["hard_gate"],
                "decision_path": record["no_qwen_audit"]["decision_path"],
                "initial_metrics": initial_metrics[sample],
                "candidate_metrics": candidate_metrics,
                "metric_and_count_deltas": deltas,
                "oracle_label": oracle_label,
                "gate_leaf": leaf,
            }
        )

    total = len(scored)
    aggregate_summary = summarize_gate_confusion(scored)
    by_tier = {
        tier: summarize_gate_confusion(
            [item for item in scored if item["tier"] == tier]
        )
        for tier in ("primary", "diagnostic_executor_only")
    }

    targets = scoring_manifest.get("target_samples", scoring_manifest["ordered_samples"])
    if not isinstance(targets, list) or any(item not in initial_metrics for item in targets):
        raise ValueError("target_samples must be a subset of ordered_samples")
    target_decisions: dict[str, Any] = {}
    for sample in targets:
        primary = [item for item in scored if item["sample"] == sample and item["tier"] == "primary"]
        primary_beneficial = [item for item in primary if item["oracle_label"] == "beneficial"]
        gate_pass = [item for item in primary_beneficial if item["hard_gate"]["passed"]]
        best_primary = max(
            primary,
            key=lambda item: (
                item["metric_and_count_deltas"]["iou"],
                item["metric_and_count_deltas"]["f1"],
                item["candidate_id"],
            ),
            default=None,
        )
        best_beneficial_gate_pass = max(
            gate_pass,
            key=lambda item: (
                item["metric_and_count_deltas"]["iou"],
                item["metric_and_count_deltas"]["f1"],
                item["candidate_id"],
            ),
            default=None,
        )
        decision = (
            "primary_actionable_and_gate_pass"
            if gate_pass
            else "primary_actionable_but_all_beneficial_hard_gated"
            if primary_beneficial
            else "no_primary_beneficial_candidate_in_frozen_enumeration"
        )
        target_decisions[sample] = {
            "decision": decision,
            "primary_candidate_count": len(primary),
            "primary_beneficial_count": len(primary_beneficial),
            "primary_beneficial_gate_pass_count": len(gate_pass),
            "best_primary_candidate_id": (
                best_primary["candidate_id"] if best_primary else None
            ),
            "best_primary_candidate_oracle_label": (
                best_primary["oracle_label"] if best_primary else None
            ),
            "best_primary_candidate_gate_passed": (
                best_primary["hard_gate"]["passed"] if best_primary else None
            ),
            "best_primary_candidate_deltas": (
                best_primary["metric_and_count_deltas"] if best_primary else None
            ),
            "best_primary_beneficial_candidate_id": (
                best_beneficial_gate_pass["candidate_id"]
                if best_beneficial_gate_pass
                else None
            ),
            "best_primary_beneficial_deltas": (
                best_beneficial_gate_pass["metric_and_count_deltas"]
                if best_beneficial_gate_pass
                else None
            ),
        }

    state_summary = summarize_candidate_states(scored)
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_epsilon": epsilon,
        "gt_access_boundary": {
            "candidate_generation_complete_verified_before_label_access": True,
            "all_candidate_hashes_verified_before_label_access": True,
            "generator_reported_gt_loaded": False,
        },
        "qwen_evaluation": {
            "qwen_invoked": False,
            "qwen_false_accept_reject_available": False,
            "gate_pass_semantics": "eligible_for_qwen_not_evaluated",
        },
        "initial_metrics": initial_metrics,
        "candidate_count": total,
        **state_summary,
        "aggregate_all_tiers_diagnostic_only": aggregate_summary,
        "by_tier": by_tier,
        "p2_pass_stop_decision_scope": "primary_only",
        # Compatibility aliases remain aggregate-only and must not drive P2 decisions.
        "oracle_counts": aggregate_summary["oracle_counts"],
        "gate_leaf_counts": aggregate_summary["gate_leaf_counts"],
        "gate_rates_with_wilson95": aggregate_summary[
            "gate_rates_with_wilson95"
        ],
        "target_primary_actionability": target_decisions,
        "candidates": scored,
    }


def score(
    scoring_manifest_path: Path,
    candidate_root: Path,
    *,
    label_loader: Callable[[Mapping[str, Any], str], np.ndarray] = _default_label_loader,
) -> dict[str, Any]:
    scoring_manifest = _read_json(scoring_manifest_path.resolve())
    validate_scoring_manifest(scoring_manifest)
    verified = verify_generation_before_gt(scoring_manifest, candidate_root)
    return score_verified_generation(scoring_manifest, verified, label_loader=label_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = score(args.scoring_manifest, args.candidate_root)
    _write_json(args.output.resolve(), report)
    decisions = report["target_primary_actionability"]
    print(
        "status=complete;gt_access_after_verified_generation=true;"
        f"candidates={report['candidate_count']};targets={len(decisions)}"
    )


if __name__ == "__main__":
    main()
