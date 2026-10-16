#!/usr/bin/env python3
"""Compare fixed-input staged-verifier control and compact-protocol runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SAMPLE_COUNT = 5
CANONICAL_SAMPLES = (
    "test_20_15",
    "test_78_13",
    "test_85_16",
    "test_1_1",
    "test_50_8",
)
EXPECTED_EVALUATION_PROFILE = "e2e5"
EXPECTED_EVALUATION_ID = "DEV-EXTENDED-5-V1"
EXPECTED_MODEL_PATH = (
    "/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct"
)
EXPECTED_ARMS = {
    "control": "v11_3-control",
    "treatment": "v12_1-compact",
}
EXPECTED_PROTOCOLS = {"control": "v11.3", "treatment": "v12.1"}
EXPECTED_SCHEMAS = {
    "control": "staged_verifier_environment_ordered_audit_v11_3",
    "treatment": "staged_verifier_environment_ordered_compact_audit_v12_1",
}
CONTROL_COMMIT = "f80e2702793d265a78b9539d438b60dd812d5fd1"
FINISH_REASONS = {
    "initial_verifier_authorized_finish",
    "verifier_authorized_finish",
}
COUNT_KEYS = ("tp", "fp", "fn", "tn")
METRIC_KEYS = ("iou", "precision", "recall", "f1")
DELTA_COUNT_KEYS = ("tp", "fp", "fn")
SUMMARY_SCHEMA_VERSION = "qwen3vl_e2e_summary_v2"
COMPARISON_EPSILON = 1e-12
CANDIDATE_CLASSIFICATIONS = (
    "beneficial",
    "harmful",
    "indeterminate",
)
CANDIDATE_DECISIONS = ("accepted", "rejected", "unavailable")
CANDIDATE_ORACLE_STATUSES = ("available", "unavailable")
CANDIDATE_DECISION_PATHS = ("qwen", "pre_qwen_hard_gate", "unavailable")
CANDIDATE_LEAF_CLASSES = (
    "true_accept",
    "qwen_false_reject",
    "hard_gate_false_reject",
    "false_accept",
    "true_reject",
    "hard_gate_correct_reject",
    "indeterminate_accepted",
    "indeterminate_qwen_rejected",
    "indeterminate_hard_gate_rejected",
    "unavailable",
)
FALSE_REJECT_COUNT_SEMANTICS = "alias_of_all_false_reject_count"
BASELINE_ACTION_FIELDS = (
    "candidates",
    "accepted",
    "rejected",
    "tool_actions",
    "action_attempts",
    "action_attempt_errors",
    "fallback_actions",
    "successful_tool_actions",
    "tool_failure_count",
    "finish_count",
)
REQUIRED_FILE_KEYS = ("A", "B", "label_cvt")
REQUIRED_TOOL_KEYS = (
    "harness_worker",
    "simpleclick_checkpoint",
    "sam3_checkpoint",
    "sam3_bpe",
    "segmentation_worker",
    "seeded_segmentation_worker",
)
REQUIRED_RUNTIME_SOURCE_KEYS = (
    "change_agent/adapters/stage_backends.py",
    "change_agent/adapters/staged_verifier.py",
    "change_agent/verifier_protocol.py",
    "tools/run_levir_change_agent.py",
)
REQUIRED_COMMIT_KEYS = (
    "chang_agent",
    "expected_chang_agent",
    "harness",
    "expected_harness",
    "omniovcd",
)
REQUIRED_ENVIRONMENT_KEYS = (
    "python",
    "torch",
    "cuda",
    "device",
    "transformers",
    "accelerate",
    "numpy",
    "pillow",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[0-9]+$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-harness-commit", required=True)
    return parser.parse_args(argv)


def read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return payload


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite, found {value!r}")
    return float(value)


def nonnegative_number(value: Any, *, name: str) -> float:
    number = finite_number(value, name=name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, found {value!r}")
    return number


def nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def read_receipt(path: Path, *, name: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read required {name} {path}: {error}") from error
    result: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            raise ValueError(f"{name} contains a malformed line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"{name} contains a duplicate or empty key: {key!r}")
        result[key] = value
    if not result:
        raise ValueError(f"{name} is empty: {path}")
    return result


def metric_counts(value: Any, *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    counts = {
        key: nonnegative_int(value.get(key), name=f"{name}.{key}")
        for key in COUNT_KEYS
    }
    expected = metrics_from_counts(counts)
    for key in METRIC_KEYS:
        observed = finite_number(value.get(key), name=f"{name}.{key}")
        if not 0.0 <= observed <= 1.0 or not math.isclose(
            observed, expected[key], rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"{name}.{key} disagrees with pixel counts")
    return counts


def rates_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in COUNT_KEYS)
    del tn
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    rates = rates_from_counts(counts)
    return {key: round(rates[key], 8) for key in METRIC_KEYS}


def add_counts(target: dict[str, int], values: Mapping[str, int]) -> None:
    for key in COUNT_KEYS:
        target[key] += values[key]


def count_deltas(
    initial: Mapping[str, int], selected: Mapping[str, int]
) -> dict[str, int]:
    return {key: selected[key] - initial[key] for key in DELTA_COUNT_KEYS}


def is_degraded(
    initial: Mapping[str, int], selected: Mapping[str, int]
) -> bool:
    initial_rates = rates_from_counts(initial)
    selected_rates = rates_from_counts(selected)
    delta_iou = selected_rates["iou"] - initial_rates["iou"]
    delta_f1 = selected_rates["f1"] - initial_rates["f1"]
    if delta_iou < -COMPARISON_EPSILON or delta_f1 < -COMPARISON_EPSILON:
        return True
    metrics_tied = (
        abs(delta_iou) <= COMPARISON_EPSILON
        and abs(delta_f1) <= COMPARISON_EPSILON
    )
    return metrics_tied and (
        selected["tp"] < initial["tp"]
        or selected["fp"] > initial["fp"]
        or selected["fn"] > initial["fn"]
    )


def macro_metrics(sample_counts: Sequence[Mapping[str, int]]) -> dict[str, float]:
    if not sample_counts:
        raise ValueError("macro metrics require at least one sample")
    rates = [rates_from_counts(counts) for counts in sample_counts]
    return {
        key: round(sum(item[key] for item in rates) / len(rates), 8)
        for key in METRIC_KEYS
    }


def optional_oracle_counts(step: Any) -> dict[str, int] | None:
    if not isinstance(step, Mapping):
        return None
    metrics = step.get("offline_metrics")
    if not isinstance(metrics, Mapping):
        return None
    counts: dict[str, int] = {}
    for key in DELTA_COUNT_KEYS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        counts[key] = value
    return counts


def explicit_candidate_decision(step: Any) -> str:
    if not isinstance(step, Mapping):
        return "unavailable"
    decisions: list[bool] = []
    direct = step.get("candidate_accepted")
    if isinstance(direct, bool):
        decisions.append(direct)
    execution = step.get("execution")
    if isinstance(execution, Mapping):
        nested = execution.get("candidate_accepted")
        if isinstance(nested, bool):
            decisions.append(nested)
    if not decisions or any(value != decisions[0] for value in decisions[1:]):
        return "unavailable"
    return "accepted" if decisions[0] else "rejected"


def explicit_hard_gate_before_qwen(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    execution = step.get("execution")
    if not isinstance(execution, Mapping):
        return False
    if execution.get("verifier_skipped_by_hard_gate") is True:
        return True
    evidence = execution.get("verifier_evidence")
    return (
        isinstance(evidence, Mapping)
        and evidence.get("type") == "runtime_hard_gate_precheck"
        and evidence.get("verifier_skipped") is True
    )


def explicit_candidate_decision_path(step: Any, *, hard_gate: bool) -> str:
    if not isinstance(step, Mapping):
        return "unavailable"
    execution = step.get("execution")
    if not isinstance(execution, Mapping):
        return "unavailable"
    evidence = execution.get("verifier_evidence")
    qwen_evidence = (
        execution.get("verifier_skipped_by_hard_gate") is False
        or isinstance(evidence, Mapping)
        and evidence.get("type") == "staged_qwen_verifier"
    )
    if hard_gate and qwen_evidence:
        return "unavailable"
    if hard_gate:
        return "pre_qwen_hard_gate"
    return "qwen" if qwen_evidence else "unavailable"


def candidate_leaf_class(
    *, classification: str, oracle_status: str, decision: str, decision_path: str
) -> str:
    if oracle_status != "available" or decision == "unavailable":
        return "unavailable"
    if decision == "accepted":
        if decision_path != "qwen":
            return "unavailable"
        return {
            "beneficial": "true_accept",
            "harmful": "false_accept",
            "indeterminate": "indeterminate_accepted",
        }.get(classification, "unavailable")
    if decision != "rejected":
        return "unavailable"
    if decision_path == "qwen":
        return {
            "beneficial": "qwen_false_reject",
            "harmful": "true_reject",
            "indeterminate": "indeterminate_qwen_rejected",
        }.get(classification, "unavailable")
    if decision_path == "pre_qwen_hard_gate":
        return {
            "beneficial": "hard_gate_false_reject",
            "harmful": "hard_gate_correct_reject",
            "indeterminate": "indeterminate_hard_gate_rejected",
        }.get(classification, "unavailable")
    return "unavailable"


def classify_candidate_counts(
    reference: Mapping[str, int], candidate: Mapping[str, int]
) -> str:
    reference_rates = rates_from_counts({**reference, "tn": 0})
    candidate_rates = rates_from_counts({**candidate, "tn": 0})
    delta_iou = candidate_rates["iou"] - reference_rates["iou"]
    delta_f1 = candidate_rates["f1"] - reference_rates["f1"]
    if delta_iou > COMPARISON_EPSILON and delta_f1 > COMPARISON_EPSILON:
        return "beneficial"
    metrics_tied = (
        abs(delta_iou) <= COMPARISON_EPSILON
        and abs(delta_f1) <= COMPARISON_EPSILON
    )
    worse_counts = (
        candidate["tp"] < reference["tp"]
        or candidate["fp"] > reference["fp"]
        or candidate["fn"] > reference["fn"]
    )
    if (
        delta_iou < -COMPARISON_EPSILON
        or delta_f1 < -COMPARISON_EPSILON
        or (metrics_tied and worse_counts)
    ):
        return "harmful"
    return "indeterminate"


def candidate_oracle_summary(
    trajectory: Mapping[str, Any], *, candidate_count: int, source: str
) -> dict[str, Any]:
    raw_steps = trajectory.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{source}.steps must contain the initial step")
    steps = raw_steps
    if len(steps) != candidate_count + 1:
        raise ValueError(
            f"{source} step/candidate accounting differs: "
            f"{len(steps)} != {candidate_count}+1"
        )
    for expected_index, step in enumerate(steps):
        observed_index = step.get("step_index") if isinstance(step, Mapping) else None
        if (
            isinstance(observed_index, bool)
            or not isinstance(observed_index, int)
            or observed_index != expected_index
        ):
            raise ValueError(
                f"{source}.steps[{expected_index}] must have "
                f"step_index={expected_index}"
            )

    reference_counts = optional_oracle_counts(steps[0])
    reference_step_index: int | None = 0
    records: list[dict[str, Any]] = []
    classification_counts = Counter({key: 0 for key in CANDIDATE_CLASSIFICATIONS})
    decision_counts = Counter({key: 0 for key in CANDIDATE_DECISIONS})
    oracle_status_counts = Counter({key: 0 for key in CANDIDATE_ORACLE_STATUSES})
    decision_path_counts = Counter({key: 0 for key in CANDIDATE_DECISION_PATHS})
    leaf_class_counts = Counter({key: 0 for key in CANDIDATE_LEAF_CLASSES})
    false_accept_count = 0
    qwen_false_reject_count = 0
    hard_gate_before_qwen_count = 0
    hard_gate_false_reject_count = 0
    all_false_reject_count = 0

    for candidate_index in range(1, candidate_count + 1):
        step = steps[candidate_index]
        candidate_counts = optional_oracle_counts(step)
        decision = explicit_candidate_decision(step)
        hard_gate = explicit_hard_gate_before_qwen(step)
        decision_path = explicit_candidate_decision_path(step, hard_gate=hard_gate)
        step_index = candidate_index

        unavailable_reason: str | None = None
        delta: dict[str, int] | None = None
        delta_iou: float | None = None
        delta_f1: float | None = None
        if reference_counts is None:
            classification = "indeterminate"
            oracle_status = "unavailable"
            unavailable_reason = "reference_offline_counts_unavailable"
        elif candidate_counts is None:
            classification = "indeterminate"
            oracle_status = "unavailable"
            unavailable_reason = "candidate_offline_counts_unavailable"
        else:
            oracle_status = "available"
            delta = count_deltas(reference_counts, candidate_counts)
            classification = classify_candidate_counts(
                reference_counts, candidate_counts
            )
            reference_rates = rates_from_counts({**reference_counts, "tn": 0})
            candidate_rates = rates_from_counts({**candidate_counts, "tn": 0})
            delta_iou = round(candidate_rates["iou"] - reference_rates["iou"], 8)
            delta_f1 = round(candidate_rates["f1"] - reference_rates["f1"], 8)

        leaf_class = candidate_leaf_class(
            classification=classification,
            oracle_status=oracle_status,
            decision=decision,
            decision_path=decision_path,
        )
        false_accept = leaf_class == "false_accept"
        qwen_false_reject = leaf_class == "qwen_false_reject"
        hard_gate_false_reject = leaf_class == "hard_gate_false_reject"
        all_false_reject = qwen_false_reject or hard_gate_false_reject
        false_reject = all_false_reject
        classification_counts[classification] += 1
        decision_counts[decision] += 1
        oracle_status_counts[oracle_status] += 1
        decision_path_counts[decision_path] += 1
        leaf_class_counts[leaf_class] += 1
        false_accept_count += int(false_accept)
        qwen_false_reject_count += int(qwen_false_reject)
        hard_gate_before_qwen_count += int(hard_gate)
        hard_gate_false_reject_count += int(hard_gate_false_reject)
        all_false_reject_count += int(all_false_reject)
        records.append(
            {
                "candidate_index": candidate_index,
                "step_index": step_index,
                "reference_step_index": reference_step_index,
                "decision": decision,
                "decision_path": decision_path,
                "classification": classification,
                "oracle_status": oracle_status,
                "leaf_class": leaf_class,
                "reference_counts": (
                    dict(reference_counts) if reference_counts is not None else None
                ),
                "candidate_counts": (
                    dict(candidate_counts) if candidate_counts is not None else None
                ),
                "delta_counts": delta,
                "delta_iou": delta_iou,
                "delta_f1": delta_f1,
                "hard_gate_before_qwen": hard_gate,
                "false_accept": false_accept,
                "qwen_false_reject": qwen_false_reject,
                "all_false_reject": all_false_reject,
                "false_reject": false_reject,
                "hard_gate_false_reject": hard_gate_false_reject,
                "unavailable_reason": unavailable_reason,
            }
        )

        if decision == "accepted":
            reference_counts = candidate_counts
            reference_step_index = step_index
        elif decision == "unavailable":
            reference_counts = None
            reference_step_index = None

    return {
        "candidate_count": candidate_count,
        "classification_counts": {
            key: classification_counts[key] for key in CANDIDATE_CLASSIFICATIONS
        },
        "decision_counts": {key: decision_counts[key] for key in CANDIDATE_DECISIONS},
        "oracle_status_counts": {
            key: oracle_status_counts[key] for key in CANDIDATE_ORACLE_STATUSES
        },
        "decision_path_counts": {
            key: decision_path_counts[key] for key in CANDIDATE_DECISION_PATHS
        },
        "leaf_class_counts": {
            key: leaf_class_counts[key] for key in CANDIDATE_LEAF_CLASSES
        },
        "false_accept_count": false_accept_count,
        "qwen_false_reject_count": qwen_false_reject_count,
        "hard_gate_before_qwen_count": hard_gate_before_qwen_count,
        "hard_gate_false_reject_count": hard_gate_false_reject_count,
        "all_false_reject_count": all_false_reject_count,
        "false_reject_count": all_false_reject_count,
        "false_reject_count_semantics": FALSE_REJECT_COUNT_SEMANTICS,
        "candidates": records,
    }


def empty_candidate_oracle_totals() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "classification_counts": {key: 0 for key in CANDIDATE_CLASSIFICATIONS},
        "decision_counts": {key: 0 for key in CANDIDATE_DECISIONS},
        "oracle_status_counts": {key: 0 for key in CANDIDATE_ORACLE_STATUSES},
        "decision_path_counts": {key: 0 for key in CANDIDATE_DECISION_PATHS},
        "leaf_class_counts": {key: 0 for key in CANDIDATE_LEAF_CLASSES},
        "false_accept_count": 0,
        "qwen_false_reject_count": 0,
        "false_reject_count": 0,
        "hard_gate_before_qwen_count": 0,
        "hard_gate_false_reject_count": 0,
        "all_false_reject_count": 0,
        "false_reject_count_semantics": FALSE_REJECT_COUNT_SEMANTICS,
    }


def add_candidate_oracle_totals(
    total: dict[str, Any], sample: Mapping[str, Any]
) -> None:
    total["candidate_count"] += sample["candidate_count"]
    for key in CANDIDATE_CLASSIFICATIONS:
        total["classification_counts"][key] += sample["classification_counts"][key]
    for key in CANDIDATE_DECISIONS:
        total["decision_counts"][key] += sample["decision_counts"][key]
    for key in CANDIDATE_ORACLE_STATUSES:
        total["oracle_status_counts"][key] += sample["oracle_status_counts"][key]
    for key in CANDIDATE_DECISION_PATHS:
        total["decision_path_counts"][key] += sample["decision_path_counts"][key]
    for key in CANDIDATE_LEAF_CLASSES:
        total["leaf_class_counts"][key] += sample["leaf_class_counts"][key]
    for key in (
        "false_accept_count",
        "qwen_false_reject_count",
        "false_reject_count",
        "hard_gate_before_qwen_count",
        "hard_gate_false_reject_count",
        "all_false_reject_count",
    ):
        total[key] += sample[key]


def sample_order(value: Any, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or value != list(CANONICAL_SAMPLES)
        or len(value) != EXPECTED_SAMPLE_COUNT
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain {EXPECTED_SAMPLE_COUNT} unique sample IDs")
    return list(value)


def region_ids(value: Any, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain unique non-empty region IDs")
    return list(value)


def sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256 digest")
    return value


def commit(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not GIT_COMMIT_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 40-hex Git commit")
    return value


def sha256_mapping(
    value: Any,
    *,
    name: str,
    required_keys: Sequence[str] | None = None,
    nested_keys: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    if required_keys is not None and set(value) != set(required_keys):
        raise ValueError(f"{name} keys differ from the required key set")
    for key, item in value.items():
        nonempty_string(key, name=f"{name} key")
        if isinstance(item, Mapping):
            if nested_keys is None or set(item) != set(nested_keys):
                raise ValueError(f"{name}.{key} keys differ from the required key set")
            for nested_key in nested_keys:
                sha256(item.get(nested_key), name=f"{name}.{key}.{nested_key}")
        else:
            sha256(item, name=f"{name}.{key}")
    return value


def require_manifest(
    payload: Mapping[str, Any], *, name: str, arm_role: str
) -> tuple[list[str], str, str]:
    if payload.get("evaluation_profile") != EXPECTED_EVALUATION_PROFILE:
        raise ValueError(
            f"{name}.evaluation_profile must be {EXPECTED_EVALUATION_PROFILE!r}"
        )
    if payload.get("evaluation_id") != EXPECTED_EVALUATION_ID:
        raise ValueError(
            f"{name}.evaluation_id must be {EXPECTED_EVALUATION_ID!r}"
        )
    samples = sample_order(payload.get("sample_order"), name=f"{name}.sample_order")
    arm = nonempty_string(payload.get("experiment_arm"), name=f"{name}.experiment_arm")
    protocol = nonempty_string(payload.get("protocol_version"), name=f"{name}.protocol_version")
    if arm != EXPECTED_ARMS[arm_role] or protocol != EXPECTED_PROTOCOLS[arm_role]:
        raise ValueError(f"{name} has the wrong fixed {arm_role} arm label/protocol")
    sha256_mapping(
        payload.get("files_sha256"),
        name=f"{name}.files_sha256",
        required_keys=samples,
        nested_keys=REQUIRED_FILE_KEYS,
    )
    if payload.get("model_path") != EXPECTED_MODEL_PATH:
        raise ValueError(f"{name}.model_path is not the pinned base model")
    sha256_mapping(
        payload.get("model_files_sha256"), name=f"{name}.model_files_sha256"
    )
    sha256_mapping(
        payload.get("tool_files_sha256"),
        name=f"{name}.tool_files_sha256",
        required_keys=REQUIRED_TOOL_KEYS,
    )
    input_root = nonempty_string(payload.get("input_root"), name=f"{name}.input_root")
    expected_input_root = "/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256"
    if input_root != expected_input_root:
        raise ValueError(f"{name}.input_root is not the pinned diagnostic dataset")
    if payload.get("gt_hashed_after_rollout") is not True:
        raise ValueError(f"{name}.gt_hashed_after_rollout must be true")
    nonempty_string(payload.get("slurm_node"), name=f"{name}.slurm_node")
    if payload.get("slurm_partition") != "gre":
        raise ValueError(f"{name}.slurm_partition must be gre")
    slurm_job_id = payload.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not JOB_ID_RE.fullmatch(slurm_job_id):
        raise ValueError(f"{name}.slurm_job_id must be numeric")
    source_hashes = sha256_mapping(
        payload.get("runtime_source_sha256"),
        name=f"{name}.runtime_source_sha256",
        required_keys=REQUIRED_RUNTIME_SOURCE_KEYS,
    )
    del source_hashes
    runtime_commits = payload.get("runtime_commits")
    if not isinstance(runtime_commits, Mapping) or set(runtime_commits) != set(
        REQUIRED_COMMIT_KEYS
    ):
        raise ValueError(f"{name}.runtime_commits has an unexpected key set")
    for key in REQUIRED_COMMIT_KEYS:
        commit(runtime_commits.get(key), name=f"{name}.runtime_commits.{key}")
    if runtime_commits["chang_agent"] != runtime_commits["expected_chang_agent"]:
        raise ValueError(f"{name}.runtime_commits arm commit does not match expected")
    if runtime_commits["harness"] != runtime_commits["expected_harness"]:
        raise ValueError(f"{name}.runtime_commits harness commit does not match expected")
    if arm_role == "control" and (
        runtime_commits["chang_agent"]
        != runtime_commits["expected_chang_agent"]
        or runtime_commits["chang_agent"] != CONTROL_COMMIT
    ):
        raise ValueError(f"{name} is not pinned to the full control commit")
    if arm_role == "treatment" and (
        runtime_commits["chang_agent"] != runtime_commits["expected_chang_agent"]
        or runtime_commits["chang_agent"] != runtime_commits["harness"]
        or runtime_commits["harness"] != runtime_commits["expected_harness"]
    ):
        raise ValueError(f"{name} compact arm commit must equal its harness commit")
    environment = payload.get("runtime_environment")
    if not isinstance(environment, Mapping) or set(environment) != set(
        REQUIRED_ENVIRONMENT_KEYS
    ):
        raise ValueError(f"{name}.runtime_environment has an unexpected key set")
    for key, value in environment.items():
        nonempty_string(key, name=f"{name}.runtime_environment key")
        nonempty_string(value, name=f"{name}.runtime_environment.{key}")
    return samples, arm, protocol


def resolved_path(value: Any, *, name: str) -> Path:
    text = nonempty_string(value, name=name)
    try:
        return Path(text).resolve()
    except OSError as error:
        raise ValueError(f"{name} is not a valid path: {error}") from error


def validate_run_receipts(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    samples: Sequence[str],
    name: str,
    expected_dependency: str,
) -> None:
    """Bind the submission and terminal worker receipts to one arm output."""

    root = root.resolve()
    commits = manifest.get("runtime_commits")
    if not isinstance(commits, Mapping):
        raise ValueError(f"{name} manifest lacks runtime commits")
    job_id = manifest.get("slurm_job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise ValueError(f"{name} manifest has an invalid Slurm job ID")
    submission = read_receipt(root / "submission.txt", name=f"{name} submission receipt")
    expected_submission = {
        "job_id": job_id,
        "arm": manifest.get("experiment_arm"),
        "protocol": manifest.get("protocol_version"),
        "expected_commit": commits.get("expected_chang_agent"),
        "expected_harness_commit": commits.get("expected_harness"),
        "evaluation_profile": manifest.get("evaluation_profile"),
        "evaluation_id": manifest.get("evaluation_id"),
        "samples": " ".join(samples),
        "dependency": expected_dependency,
        "resources": "gpu:1,cpu:2,mem:32G,time=03:00:00",
    }
    # These fields are written by submit_local_verifier_e2e5.sh.  Requiring
    # them prevents an old receipt from being silently paired with a new run.
    for key, expected in expected_submission.items():
        if key not in submission:
            raise ValueError(f"{name} submission receipt is missing {key}")
        if submission[key] != expected:
            raise ValueError(f"{name} submission receipt {key} is not bound")
    if "node" not in submission or submission["node"] != manifest.get("slurm_node"):
        raise ValueError(f"{name} submission receipt node is not bound")
    if "partition" not in submission or submission["partition"] != manifest.get("slurm_partition"):
        raise ValueError(f"{name} submission receipt partition is not bound")
    if "output" not in submission or resolved_path(
        submission["output"], name=f"{name} submission output"
    ) != root:
        raise ValueError(f"{name} submission receipt output is not this arm root")
    nonempty_string(submission.get("arm_repo"), name=f"{name} submission arm_repo")
    expected_worker = Path(__file__).resolve().with_name(
        "run_qwen3vl_base_e2e5.slurm"
    )
    if "worker" not in submission or resolved_path(
        submission["worker"], name=f"{name} submission worker"
    ) != expected_worker:
        raise ValueError(f"{name} submission did not use the comparison harness worker")
    worker_sha256 = sha256(
        submission.get("worker_sha256"),
        name=f"{name} submission worker_sha256",
    )
    manifest_worker_sha256 = sha256(
        manifest.get("tool_files_sha256", {}).get("harness_worker"),
        name=f"{name} manifest harness_worker SHA-256",
    )
    executed_worker_sha256 = sha256(
        manifest.get("executed_worker_sha256"),
        name=f"{name} manifest executed_worker_sha256",
    )
    if not (
        worker_sha256 == manifest_worker_sha256 == executed_worker_sha256
    ):
        raise ValueError(f"{name} worker SHA-256 bindings disagree")

    status_path = root / "logs" / f"status-{job_id}.txt"
    status = read_receipt(status_path, name=f"{name} terminal status receipt")
    if status.get("status") != "success" or status.get("exit_code") != "0":
        raise ValueError(f"{name} terminal status is not successful")
    if status.get("job_id") != job_id:
        raise ValueError(f"{name} terminal status job ID is not bound")
    if status.get("worker_sha256") != worker_sha256:
        raise ValueError(f"{name} terminal status worker SHA-256 is not bound")
    expected_status_identity = {
        "evaluation_profile": manifest.get("evaluation_profile"),
        "evaluation_id": manifest.get("evaluation_id"),
        "samples": " ".join(samples),
    }
    for key, expected in expected_status_identity.items():
        if status.get(key) != expected:
            raise ValueError(f"{name} terminal status {key} is not bound")
    expected_run = (root / "run").resolve()
    if "run_dir" not in status or resolved_path(
        status["run_dir"], name=f"{name} terminal run_dir"
    ) != expected_run:
        raise ValueError(f"{name} terminal status run_dir is not bound")
    elapsed_text = status.get("elapsed_seconds")
    if not isinstance(elapsed_text, str) or not elapsed_text.strip():
        raise ValueError(f"{name} terminal elapsed_seconds is malformed")
    try:
        elapsed = nonnegative_number(
            float(elapsed_text), name=f"{name} terminal elapsed_seconds"
        )
    except ValueError as error:
        raise ValueError(f"{name} terminal elapsed_seconds is malformed") from error
    del elapsed
    nonempty_string(status.get("key_result"), name=f"{name} terminal key_result")
    if status.get("error") != "":
        raise ValueError(f"{name} successful terminal status contains an error")


def validate_summary_binding(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root: Path,
    *,
    name: str,
    baseline: bool,
) -> None:
    """Ensure a generated summary identifies this exact arm and run."""

    commits = manifest.get("runtime_commits")
    if not isinstance(commits, Mapping):
        raise ValueError(f"{name} manifest lacks runtime commits")
    expected = {
        "evaluation_profile": manifest.get("evaluation_profile"),
        "evaluation_id": manifest.get("evaluation_id"),
        "experiment_arm": manifest.get("experiment_arm"),
        "protocol_version": manifest.get("protocol_version"),
        "job_id": manifest.get("slurm_job_id"),
        "runtime_commit": commits.get("chang_agent"),
        "harness_commit": commits.get("harness"),
        "sample_order": manifest.get("sample_order"),
    }
    if baseline:
        expected.update(
            {
                "model_state": "base",
                "rollout_marker": str((root / "run" / "rollout_complete.json").resolve()),
                "source_metrics": str((root / "run" / "per_sample_metrics.json").resolve()),
            }
        )
    else:
        expected["run_dir"] = str((root / "run").resolve())
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{name}.{key} is not bound to its input manifest/run")


def require_completed_summary(
    payload: Mapping[str, Any],
    *,
    name: str,
    samples: Sequence[str],
    protocol: bool,
) -> None:
    if payload.get("status") != "completed":
        raise ValueError(f"{name}.status must be completed")
    if payload.get("sample_order") != list(samples):
        raise ValueError(f"{name} sample order differs from input manifest")
    if protocol:
        totals = payload.get("totals")
        per_sample = payload.get("per_sample")
        versions = payload.get("schema_versions")
        if not isinstance(totals, Mapping) or not isinstance(per_sample, Mapping):
            raise ValueError(f"{name} lacks totals or per-sample protocol results")
        if list(per_sample) != list(samples):
            raise ValueError(f"{name} protocol sample identity/order differs")
        if (
            not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], str)
            or not versions[0]
        ):
            raise ValueError(f"{name}.schema_versions must contain exactly one version")
    else:
        if payload.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"{name}.summary_schema_version must be {SUMMARY_SCHEMA_VERSION!r}"
            )
        if (
            isinstance(payload.get("requested_samples"), bool)
            or isinstance(payload.get("scored_samples"), bool)
            or payload.get("requested_samples") != len(samples)
            or payload.get("scored_samples") != len(samples)
        ):
            raise ValueError(f"{name} must score all {len(samples)} requested samples")
        boundary = payload.get("gt_boundary")
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("rollout_marker_claims_gt_free") is not True
            or boundary.get("offline_metrics_are_post_rollout") is not True
        ):
            raise ValueError(f"{name} lacks a GT-free rollout boundary")
        per_sample = payload.get("per_sample")
        if not isinstance(per_sample, Mapping) or list(per_sample) != list(samples):
            raise ValueError(f"{name} baseline sample identity/order differs")


def close_number(observed: Any, expected: float, *, name: str) -> None:
    number = finite_number(observed, name=name)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{name} disagrees with reconstructed evidence")


def validate_baseline_totals(
    root: Path,
    payload: Mapping[str, Any],
    *,
    name: str,
    samples: Sequence[str],
) -> None:
    """Reconstruct every baseline metric/action total from raw run evidence."""

    run_dir = (root / "run").resolve()
    if (run_dir / ".run_incomplete").exists():
        raise ValueError(f"{name} run is still marked incomplete")
    rollout = read_object(run_dir / "rollout_complete.json")
    rollout_samples = rollout.get("samples")
    if (
        rollout.get("execution_status") != "completed"
        or rollout.get("gt_loaded") is not False
        or not isinstance(rollout_samples, list)
        or any(not isinstance(item, str) or not item for item in rollout_samples)
        or rollout_samples != sorted(samples)
    ):
        raise ValueError(f"{name} rollout marker is not the canonical completed run")
    metrics = read_object(run_dir / "per_sample_metrics.json")
    raw_samples = metrics.get("samples")
    if not isinstance(raw_samples, Mapping) or list(raw_samples) != list(samples):
        raise ValueError(f"{name} raw metrics sample identity/order differs")
    summary_samples = payload.get("per_sample")
    if not isinstance(summary_samples, Mapping) or list(summary_samples) != list(samples):
        raise ValueError(f"{name} baseline per-sample identity/order differs")

    initial_total = {key: 0 for key in COUNT_KEYS}
    selected_total = {key: 0 for key in COUNT_KEYS}
    initial_sample_counts: list[dict[str, int]] = []
    selected_sample_counts: list[dict[str, int]] = []
    candidate_oracle_totals = empty_candidate_oracle_totals()
    action_totals: Counter[str] = Counter()
    outcome_totals: Counter[str] = Counter()
    degraded_total = 0
    for sample in samples:
        raw = raw_samples.get(sample)
        summary = summary_samples.get(sample)
        if not isinstance(raw, Mapping) or not isinstance(summary, Mapping):
            raise ValueError(f"{name}.{sample} evidence is malformed")
        raw_initial = raw.get("initial")
        raw_selected = raw.get("verifier_selected")
        initial_counts = metric_counts(
            raw_initial, name=f"{name}.raw.{sample}.initial"
        )
        selected_counts = metric_counts(
            raw_selected, name=f"{name}.raw.{sample}.selected"
        )
        summary_initial = metric_counts(
            summary.get("initial"), name=f"{name}.{sample}.initial"
        )
        summary_selected = metric_counts(
            summary.get("selected"), name=f"{name}.{sample}.selected"
        )
        if summary_initial != initial_counts or summary_selected != selected_counts:
            raise ValueError(f"{name}.{sample} metrics differ from raw evidence")
        add_counts(initial_total, initial_counts)
        add_counts(selected_total, selected_counts)
        initial_sample_counts.append(initial_counts)
        selected_sample_counts.append(selected_counts)
        initial_metrics = metrics_from_counts(initial_counts)
        selected_metrics = metrics_from_counts(selected_counts)
        expected_delta_iou = round(
            selected_metrics["iou"] - initial_metrics["iou"], 8
        )
        expected_delta_f1 = round(
            selected_metrics["f1"] - initial_metrics["f1"], 8
        )
        close_number(
            summary.get("delta_iou"),
            expected_delta_iou,
            name=f"{name}.{sample}.delta_iou",
        )
        close_number(
            summary.get("delta_f1"),
            expected_delta_f1,
            name=f"{name}.{sample}.delta_f1",
        )
        if summary.get("delta_counts") != count_deltas(
            initial_counts, selected_counts
        ):
            raise ValueError(f"{name}.{sample}.delta_counts disagrees with metrics")
        degraded = is_degraded(initial_counts, selected_counts)
        if summary.get("degraded") is not degraded:
            raise ValueError(f"{name}.{sample}.degraded disagrees with metrics")
        degraded_total += int(degraded)

        episode = read_object(
            run_dir / "trajectories" / sample / "episode_summary.json"
        )
        if (
            episode.get("sample") != sample
            or episode.get("execution_status") != "completed"
        ):
            raise ValueError(f"{name}.{sample} episode is not completed/canonical")
        outcome = nonempty_string(
            episode.get("episode_outcome"), name=f"{name}.{sample}.episode_outcome"
        )
        stop_reason = nonempty_string(
            episode.get("stop_reason"), name=f"{name}.{sample}.stop_reason"
        )
        if (
            summary.get("episode_outcome") != outcome
            or summary.get("stop_reason") != stop_reason
        ):
            raise ValueError(f"{name}.{sample} outcome differs from episode evidence")
        outcome_totals[outcome] += 1
        candidate_count = nonnegative_int(
            episode.get("candidate_count"), name=f"{name}.{sample}.candidate_count"
        )
        accepted = nonnegative_int(
            episode.get("accepted_candidate_count"),
            name=f"{name}.{sample}.accepted_candidate_count",
        )
        rejected = nonnegative_int(
            episode.get("rejected_candidate_count"),
            name=f"{name}.{sample}.rejected_candidate_count",
        )
        tool_actions = nonnegative_int(
            episode.get("tool_action_count"),
            name=f"{name}.{sample}.tool_action_count",
        )
        attempts = nonnegative_int(
            episode.get("action_attempt_count"),
            name=f"{name}.{sample}.action_attempt_count",
        )
        errors = nonnegative_int(
            episode.get("action_attempt_error_count"),
            name=f"{name}.{sample}.action_attempt_error_count",
        )
        fallback_actions = nonnegative_int(
            episode.get("empty_mask_scan_fallback_action_count"),
            name=f"{name}.{sample}.fallback_action_count",
        )
        if candidate_count != accepted + rejected or attempts < candidate_count or errors > attempts:
            raise ValueError(f"{name}.{sample} episode action accounting is inconsistent")
        per_sample_expected = {
            "candidate_count": candidate_count,
            "accepted_candidate_count": accepted,
            "rejected_candidate_count": rejected,
            "tool_action_count": tool_actions,
            "action_attempt_error_count": errors,
        }
        for key, expected in per_sample_expected.items():
            if nonnegative_int(summary.get(key), name=f"{name}.{sample}.{key}") != expected:
                raise ValueError(f"{name}.{sample}.{key} differs from episode evidence")

        trajectory = read_object(run_dir / "trajectories" / sample / "trajectory.json")
        expected_candidate_oracle = candidate_oracle_summary(
            trajectory,
            candidate_count=candidate_count,
            source=f"{name}.{sample}.trajectory",
        )
        if summary.get("candidate_oracle") != expected_candidate_oracle:
            raise ValueError(
                f"{name}.{sample}.candidate_oracle disagrees with trajectory evidence"
            )
        add_candidate_oracle_totals(
            candidate_oracle_totals, expected_candidate_oracle
        )

        action_successes = 0
        action_failures = 0
        tool_root = run_dir / "tool_runs" / sample
        action_call_dirs = (
            [
                path
                for path in tool_root.iterdir()
                if path.is_dir()
                and (path.name.startswith("point_") or path.name.startswith("box_"))
            ]
            if tool_root.is_dir()
            else []
        )
        for call_dir in action_call_dirs:
            report_path = call_dir / "report.json"
            if not report_path.is_file():
                action_failures += 1
                continue
            report = read_object(report_path)
            if report.get("status") == "success":
                action_successes += 1
            else:
                action_failures += 1
        if action_successes != tool_actions:
            raise ValueError(f"{name}.{sample} tool reports disagree with episode")
        action_totals.update(
            candidates=candidate_count,
            accepted=accepted,
            rejected=rejected,
            tool_actions=tool_actions,
            action_attempts=attempts,
            action_attempt_errors=errors,
            fallback_actions=fallback_actions,
            successful_tool_actions=action_successes,
            tool_failure_count=action_failures,
            finish_count=int(stop_reason in FINISH_REASONS),
        )

    raw_aggregate = metrics.get("aggregate")
    if not isinstance(raw_aggregate, Mapping):
        raise ValueError(f"{name} raw selected aggregate is missing")
    raw_aggregate_counts = metric_counts(
        raw_aggregate, name=f"{name}.raw.aggregate"
    )
    if raw_aggregate_counts != selected_total:
        raise ValueError(f"{name} raw aggregate counts differ from raw samples")
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"{name}.aggregate must be an object")
    if aggregate.get("aggregation") != "micro_count_sum":
        raise ValueError(f"{name}.aggregate aggregation contract is malformed")
    initial_aggregate = metric_counts(
        aggregate.get("initial"), name=f"{name}.aggregate.initial"
    )
    selected_aggregate = metric_counts(
        aggregate.get("selected"), name=f"{name}.aggregate.selected"
    )
    if initial_aggregate != initial_total or selected_aggregate != selected_total:
        raise ValueError(f"{name} aggregate counts differ from per-sample sums")
    initial_metrics = metrics_from_counts(initial_total)
    selected_metrics = metrics_from_counts(selected_total)
    close_number(
        aggregate.get("delta_iou"),
        round(selected_metrics["iou"] - initial_metrics["iou"], 8),
        name=f"{name}.aggregate.delta_iou",
    )
    close_number(
        aggregate.get("delta_f1"),
        round(selected_metrics["f1"] - initial_metrics["f1"], 8),
        name=f"{name}.aggregate.delta_f1",
    )
    if aggregate.get("delta_counts") != count_deltas(initial_total, selected_total):
        raise ValueError(f"{name}.aggregate.delta_counts disagrees with sample sums")

    macro = payload.get("macro")
    if not isinstance(macro, Mapping):
        raise ValueError(f"{name}.macro must be an object")
    if macro.get("aggregation") != "sample_arithmetic_mean":
        raise ValueError(f"{name}.macro aggregation contract is malformed")
    expected_initial_macro = macro_metrics(initial_sample_counts)
    expected_selected_macro = macro_metrics(selected_sample_counts)
    for state, expected in (
        ("initial", expected_initial_macro),
        ("selected", expected_selected_macro),
    ):
        observed = macro.get(state)
        if not isinstance(observed, Mapping):
            raise ValueError(f"{name}.macro.{state} must be an object")
        for key in METRIC_KEYS:
            close_number(
                observed.get(key), expected[key], name=f"{name}.macro.{state}.{key}"
            )
    close_number(
        macro.get("delta_iou"),
        round(expected_selected_macro["iou"] - expected_initial_macro["iou"], 8),
        name=f"{name}.macro.delta_iou",
    )
    close_number(
        macro.get("delta_f1"),
        round(expected_selected_macro["f1"] - expected_initial_macro["f1"], 8),
        name=f"{name}.macro.delta_f1",
    )
    if payload.get("candidate_oracle") != candidate_oracle_totals:
        raise ValueError(f"{name}.candidate_oracle disagrees with sample totals")

    actions = payload.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError(f"{name}.actions must be an object")
    for key in BASELINE_ACTION_FIELDS:
        actual = nonnegative_int(actions.get(key), name=f"{name}.actions.{key}")
        if actual != action_totals[key]:
            raise ValueError(f"{name}.actions.{key} disagrees with run evidence")
    tool_denominator = action_totals["successful_tool_actions"] + action_totals["tool_failure_count"]
    expected_success_rate = (
        action_totals["successful_tool_actions"] / tool_denominator
        if tool_denominator
        else None
    )
    observed_success_rate = actions.get("tool_success_rate")
    if expected_success_rate is None:
        if observed_success_rate is not None:
            raise ValueError(f"{name}.actions.tool_success_rate must be null")
    else:
        close_number(
            observed_success_rate,
            expected_success_rate,
            name=f"{name}.actions.tool_success_rate",
        )

    observed_outcomes = payload.get("episode_outcome_counts")
    if not isinstance(observed_outcomes, Mapping):
        raise ValueError(f"{name}.episode_outcome_counts must be an object")
    normalized_outcomes = {
        key: nonnegative_int(value, name=f"{name}.episode_outcome_counts.{key}")
        for key, value in observed_outcomes.items()
        if isinstance(key, str) and key
    }
    if len(normalized_outcomes) != len(observed_outcomes) or normalized_outcomes != dict(
        sorted(outcome_totals.items())
    ):
        raise ValueError(f"{name}.episode_outcome_counts disagrees with episodes")
    invalid_or_no_action = outcome_totals.get("verifier_invalid", 0) + outcome_totals.get(
        "no_executable_action", 0
    )
    if nonnegative_int(
        payload.get("invalid_or_no_action_samples"),
        name=f"{name}.invalid_or_no_action_samples",
    ) != invalid_or_no_action:
        raise ValueError(f"{name}.invalid_or_no_action_samples disagrees with episodes")
    if nonnegative_int(
        payload.get("degraded_sample_count"), name=f"{name}.degraded_sample_count"
    ) != degraded_total:
        raise ValueError(f"{name}.degraded_sample_count disagrees with metrics")


PROTOCOL_SCALAR_FIELDS = (
    "trajectory_steps",
    "backend_call_count",
    "raw_response_chars",
    "validation_error_count",
    "regional_validation_error_count",
    "atomic_audit_records",
    "atomic_audit_valid_records",
    "audited_region_records",
    "initial_environment_region_records",
    "initial_atomic_audit_records",
    "verifier_invalid_steps",
    "localization_invalid_steps",
    "hard_gate_skipped_steps",
    "action_attempt_error_count",
    "tool_action_count",
)
PROTOCOL_COUNTER_FIELDS = (
    "stage_calls",
    "attempt_kinds",
    "visual_input_scopes",
    "stage_raw_response_chars",
    "diagnosis_types",
    "plans",
)
PROTOCOL_VISUAL_INPUT_SCOPES = frozenset(
    (
        "standard",
        "authoritative_pair_rgb_only_scan",
        "authoritative_pair_rgb_only_evidence_repair",
        "unreported",
    )
)


def validate_protocol_totals(
    payload: Mapping[str, Any], *, name: str, samples: Sequence[str]
) -> None:
    totals = payload.get("totals")
    per_sample = payload.get("per_sample")
    if not isinstance(totals, Mapping) or not isinstance(per_sample, Mapping):
        raise ValueError(f"{name} lacks totals or per-sample protocol results")
    scalar_sums = {field: 0 for field in PROTOCOL_SCALAR_FIELDS}
    counter_sums: dict[str, dict[str, int]] = {
        field: {} for field in PROTOCOL_COUNTER_FIELDS
    }
    latency_sum: dict[str, float] = {}
    elapsed_sum = 0.0
    outcome_sums: Counter[str] = Counter()
    schema_versions: set[str] = set()
    for sample in samples:
        result = per_sample.get(sample)
        if not isinstance(result, Mapping):
            raise ValueError(f"{name} missing per-sample protocol result: {sample}")
        ids = region_ids(
            result.get("initial_audited_region_ids"),
            name=f"{name}.{sample}.initial_audited_region_ids",
        )
        environment_count = nonnegative_int(
            result.get("initial_environment_region_records"),
            name=f"{name}.{sample}.initial_environment_region_records",
        )
        atomic_count = nonnegative_int(
            result.get("initial_atomic_audit_records"),
            name=f"{name}.{sample}.initial_atomic_audit_records",
        )
        if environment_count != len(ids) or atomic_count != len(ids):
            raise ValueError(
                f"{name}.{sample} initial coverage counts do not equal region IDs"
            )
        outcome = nonempty_string(
            result.get("episode_outcome"), name=f"{name}.{sample}.episode_outcome"
        )
        outcome_sums[outcome] += 1
        versions = result.get("schema_versions")
        if (
            not isinstance(versions, list)
            or not versions
            or any(not isinstance(version, str) or not version for version in versions)
            or len(versions) != len(set(versions))
        ):
            raise ValueError(f"{name}.{sample}.schema_versions is malformed")
        schema_versions.update(versions)
        elapsed = nonnegative_number(
            result.get("episode_elapsed_seconds"),
            name=f"{name}.{sample}.episode_elapsed_seconds",
        )
        elapsed_sum += elapsed
        for field in PROTOCOL_SCALAR_FIELDS:
            value = nonnegative_int(result.get(field), name=f"{name}.{sample}.{field}")
            scalar_sums[field] += value
        for field in PROTOCOL_COUNTER_FIELDS:
            values = result.get(field)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name}.{sample}.{field} must be an object")
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{name}.{sample}.{field} has an invalid key")
                number = nonnegative_int(value, name=f"{name}.{sample}.{field}.{key}")
                counter_sums[field][key] = counter_sums[field].get(key, 0) + number
        visual_input_scopes = result.get("visual_input_scopes")
        if not isinstance(visual_input_scopes, Mapping) or not set(
            visual_input_scopes
        ) <= PROTOCOL_VISUAL_INPUT_SCOPES:
            raise ValueError(
                f"{name}.{sample}.visual_input_scopes has an unsupported scope"
            )
        sample_backend_calls = nonnegative_int(
            result.get("backend_call_count"),
            name=f"{name}.{sample}.backend_call_count",
        )
        for field in ("stage_calls", "attempt_kinds", "visual_input_scopes"):
            values = result[field]
            call_count = sum(
                nonnegative_int(
                    value,
                    name=f"{name}.{sample}.{field}.{key}",
                )
                for key, value in values.items()
            )
            if call_count != sample_backend_calls:
                raise ValueError(
                    f"{name}.{sample}.{field} disagrees with backend_call_count"
                )
        latencies = result.get("stage_latency_seconds")
        if not isinstance(latencies, Mapping):
            raise ValueError(f"{name}.{sample}.stage_latency_seconds must be an object")
        sample_latency = 0.0
        for key, value in latencies.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name}.{sample}.stage_latency_seconds has an invalid key")
            stage_latency = nonnegative_number(
                value, name=f"{name}.{sample}.stage_latency_seconds.{key}"
            )
            sample_latency += stage_latency
            latency_sum[key] = latency_sum.get(key, 0.0) + stage_latency
        backend_latency = nonnegative_number(
            result.get("backend_latency_seconds"),
            name=f"{name}.{sample}.backend_latency_seconds",
        )
        if not math.isclose(
            backend_latency, sample_latency, rel_tol=1e-6, abs_tol=1e-5
        ):
            raise ValueError(
                f"{name}.{sample}.backend_latency_seconds disagrees with stages"
            )
        if elapsed + 0.01 < sample_latency:
            raise ValueError(f"{name}.{sample} elapsed time is below backend latency")
        stage_calls = result.get("stage_calls")
        if not isinstance(stage_calls, Mapping):
            raise ValueError(f"{name}.{sample}.stage_calls must be an object")
        records = nonnegative_int(
            result.get("atomic_audit_records"),
            name=f"{name}.{sample}.atomic_audit_records",
        )
        valid = nonnegative_int(
            result.get("atomic_audit_valid_records"),
            name=f"{name}.{sample}.atomic_audit_valid_records",
        )
        regional = nonnegative_int(
            result.get("regional_validation_error_count"),
            name=f"{name}.{sample}.regional_validation_error_count",
        )
        audit_calls = nonnegative_int(
            stage_calls.get("audit"),
            name=f"{name}.{sample}.stage_calls.audit",
        )
        if valid + regional != records or audit_calls < records:
            raise ValueError(f"{name}.{sample} audit record accounting is inconsistent")
    for field, expected in scalar_sums.items():
        actual = nonnegative_int(totals.get(field), name=f"{name}.totals.{field}")
        if actual != expected:
            raise ValueError(f"{name}.totals.{field} disagrees with per-sample sum")
    for field, expected in counter_sums.items():
        actual = totals.get(field)
        if not isinstance(actual, Mapping):
            raise ValueError(f"{name}.totals.{field} must be an object")
        normalized = {
            str(key): nonnegative_int(value, name=f"{name}.totals.{field}.{key}")
            for key, value in actual.items()
        }
        if normalized != expected:
            raise ValueError(f"{name}.totals.{field} disagrees with per-sample sum")
    actual_latency = totals.get("stage_latency_seconds")
    if not isinstance(actual_latency, Mapping):
        raise ValueError(f"{name}.totals.stage_latency_seconds must be an object")
    if set(actual_latency) != set(latency_sum) or any(
        not math.isclose(
            nonnegative_number(actual_latency[key], name=f"{name}.totals.stage_latency_seconds.{key}"),
            value,
            rel_tol=1e-6,
            abs_tol=1e-5,
        )
        for key, value in latency_sum.items()
    ):
        raise ValueError(f"{name}.totals.stage_latency_seconds disagrees with per-sample sum")
    actual_elapsed = nonnegative_number(
        totals.get("episode_elapsed_seconds"),
        name=f"{name}.totals.episode_elapsed_seconds",
    )
    if not math.isclose(
        actual_elapsed, elapsed_sum, rel_tol=1e-6, abs_tol=1e-5
    ):
        raise ValueError(
            f"{name}.totals.episode_elapsed_seconds disagrees with per-sample sum"
        )
    actual_outcomes = totals.get("episode_outcomes")
    if not isinstance(actual_outcomes, Mapping):
        raise ValueError(f"{name}.totals.episode_outcomes must be an object")
    normalized_outcomes = {
        key: nonnegative_int(value, name=f"{name}.totals.episode_outcomes.{key}")
        for key, value in actual_outcomes.items()
        if isinstance(key, str) and key
    }
    if len(normalized_outcomes) != len(actual_outcomes) or normalized_outcomes != dict(
        sorted(outcome_sums.items())
    ):
        raise ValueError(f"{name}.totals.episode_outcomes disagrees with samples")
    if payload.get("schema_versions") != sorted(schema_versions):
        raise ValueError(f"{name}.schema_versions disagrees with per-sample schemas")
    backend_latency = nonnegative_number(
        totals.get("backend_latency_seconds"),
        name=f"{name}.totals.backend_latency_seconds",
    )
    if not math.isclose(
        backend_latency,
        sum(latency_sum.values()),
        rel_tol=1e-6,
        abs_tol=1e-5,
    ):
        raise ValueError(f"{name}.backend_latency_seconds disagrees with stage latency")
    backend_calls = nonnegative_int(
        totals.get("backend_call_count"), name=f"{name}.totals.backend_call_count"
    )
    if backend_calls != sum(counter_sums["stage_calls"].values()):
        raise ValueError(f"{name}.backend_call_count disagrees with stage calls")
    if backend_calls != sum(counter_sums["attempt_kinds"].values()):
        raise ValueError(f"{name}.backend_call_count disagrees with attempt kinds")
    if backend_calls != sum(counter_sums["visual_input_scopes"].values()):
        raise ValueError(f"{name}.backend_call_count disagrees with visual input scopes")
    raw_chars = nonnegative_int(
        totals.get("raw_response_chars"), name=f"{name}.totals.raw_response_chars"
    )
    if raw_chars != sum(counter_sums["stage_raw_response_chars"].values()):
        raise ValueError(f"{name}.raw_response_chars disagrees with stage response chars")
    valid = nonnegative_int(
        totals.get("atomic_audit_valid_records"),
        name=f"{name}.totals.atomic_audit_valid_records",
    )
    records = nonnegative_int(
        totals.get("atomic_audit_records"), name=f"{name}.totals.atomic_audit_records"
    )
    regional = nonnegative_int(
        totals.get("regional_validation_error_count"),
        name=f"{name}.totals.regional_validation_error_count",
    )
    audit_calls = nonnegative_int(
        totals["stage_calls"].get("audit"), name=f"{name}.totals.stage_calls.audit"
    )
    if valid + regional != records or audit_calls < records:
        raise ValueError(f"{name} audit record accounting is inconsistent")


def nested_number(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            raise ValueError(f"{name} is missing at {key}")
        if key not in current:
            raise ValueError(f"{name} is missing at {key}")
        current = current.get(key)
    return (
        nonnegative_number(current, name=name)
        if nonnegative
        else finite_number(current, name=name)
    )


def delta(control: float, treatment: float) -> dict[str, float | None]:
    return {
        "control": control,
        "treatment": treatment,
        "absolute": treatment - control,
        "treatment_over_control": treatment / control if control else None,
        "reduction_ratio": (control - treatment) / control if control else None,
    }


def comparable_manifests(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {
        "evaluation_profile": control.get("evaluation_profile")
        == treatment.get("evaluation_profile"),
        "evaluation_id": control.get("evaluation_id") == treatment.get("evaluation_id"),
        "sample_order": control.get("sample_order") == treatment.get("sample_order"),
        "input_root": control.get("input_root") == treatment.get("input_root"),
        "input_files_sha256": control.get("files_sha256")
        == treatment.get("files_sha256"),
        "model_path": control.get("model_path") == treatment.get("model_path"),
        "model_files_sha256": control.get("model_files_sha256")
        == treatment.get("model_files_sha256"),
        "tool_files_sha256": control.get("tool_files_sha256")
        == treatment.get("tool_files_sha256"),
        "slurm_node": control.get("slurm_node") == treatment.get("slurm_node"),
        "slurm_partition": control.get("slurm_partition")
        == treatment.get("slurm_partition"),
        "runtime_environment": control.get("runtime_environment")
        == treatment.get("runtime_environment"),
        "gt_boundary": control.get("gt_hashed_after_rollout") is True
        and treatment.get("gt_hashed_after_rollout") is True,
    }
    checks["all"] = all(checks.values())
    return checks


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    control_root = args.control.resolve()
    treatment_root = args.treatment.resolve()
    comparison_path = args.output.resolve()
    control_manifest = read_object(control_root / "input_manifest.json")
    treatment_manifest = read_object(treatment_root / "input_manifest.json")
    control_protocol = read_object(control_root / "protocol_summary.json")
    treatment_protocol = read_object(treatment_root / "protocol_summary.json")
    control_baseline = read_object(control_root / "baseline_summary.json")
    treatment_baseline = read_object(treatment_root / "baseline_summary.json")
    control_order, control_arm, control_version = require_manifest(
        control_manifest, name="control manifest", arm_role="control"
    )
    treatment_order, treatment_arm, treatment_version = require_manifest(
        treatment_manifest, name="treatment manifest", arm_role="treatment"
    )
    if (
        control_root == treatment_root
        or comparison_path == control_root
        or comparison_path == treatment_root
        or control_root in comparison_path.parents
        or treatment_root in comparison_path.parents
    ):
        raise ValueError("control, treatment, and comparison roots must be distinct")
    comparability = comparable_manifests(control_manifest, treatment_manifest)
    if not comparability["all"]:
        failed = sorted(key for key, passed in comparability.items() if not passed)
        raise ValueError(f"A/B input comparability failed: {failed}")
    if control_order != treatment_order:
        raise ValueError("A/B manifests have different sample order")
    if control_manifest.get("slurm_job_id") == treatment_manifest.get("slurm_job_id"):
        raise ValueError("A/B manifests unexpectedly reuse one Slurm job ID")
    if control_arm != EXPECTED_ARMS["control"] or treatment_arm != EXPECTED_ARMS["treatment"]:
        raise ValueError("A/B arm labels are not in the fixed control/treatment order")
    if control_version != EXPECTED_PROTOCOLS["control"] or treatment_version != EXPECTED_PROTOCOLS["treatment"]:
        raise ValueError("A/B protocol versions are not in the fixed control/treatment order")
    if control_manifest.get("slurm_node") != treatment_manifest.get("slurm_node"):
        raise ValueError("A/B runs used different Slurm nodes")
    if control_manifest.get("slurm_partition") != treatment_manifest.get("slurm_partition"):
        raise ValueError("A/B runs used different Slurm partitions")
    if control_manifest.get("runtime_environment") != treatment_manifest.get("runtime_environment"):
        raise ValueError("A/B runtime environments differ")
    control_commits = control_manifest["runtime_commits"]
    treatment_commits = treatment_manifest["runtime_commits"]
    expected_harness_commit = str(
        getattr(args, "expected_harness_commit", "")
    )
    if expected_harness_commit:
        commit(expected_harness_commit, name="expected_harness_commit")
        if (
            control_commits["harness"] != expected_harness_commit
            or treatment_commits["harness"] != expected_harness_commit
        ):
            raise ValueError("A/B manifests are not bound to the comparison harness")
    if control_commits["harness"] != treatment_commits["harness"]:
        raise ValueError("A/B harness commits differ")
    if control_commits["omniovcd"] != treatment_commits["omniovcd"]:
        raise ValueError("A/B OmniOVCD commits differ")
    if control_commits["chang_agent"] == treatment_commits["chang_agent"]:
        raise ValueError("A/B arm commits are unexpectedly identical")
    control_sources = control_manifest["runtime_source_sha256"]
    treatment_sources = treatment_manifest["runtime_source_sha256"]
    if control_sources == treatment_sources:
        raise ValueError("A/B runtime source hashes are unexpectedly identical")
    sample_order = control_order
    validate_run_receipts(
        control_root,
        control_manifest,
        samples=sample_order,
        name="control",
        expected_dependency="none",
    )
    validate_run_receipts(
        treatment_root,
        treatment_manifest,
        samples=sample_order,
        name="treatment",
        expected_dependency=f"afterany:{control_manifest['slurm_job_id']}",
    )
    require_completed_summary(
        control_protocol, name="control protocol", samples=sample_order, protocol=True
    )
    require_completed_summary(
        treatment_protocol, name="treatment protocol", samples=sample_order, protocol=True
    )
    require_completed_summary(
        control_baseline, name="control baseline", samples=sample_order, protocol=False
    )
    require_completed_summary(
        treatment_baseline,
        name="treatment baseline",
        samples=sample_order,
        protocol=False,
    )
    validate_summary_binding(
        control_protocol,
        control_manifest,
        control_root,
        name="control protocol",
        baseline=False,
    )
    validate_summary_binding(
        treatment_protocol,
        treatment_manifest,
        treatment_root,
        name="treatment protocol",
        baseline=False,
    )
    validate_summary_binding(
        control_baseline,
        control_manifest,
        control_root,
        name="control baseline",
        baseline=True,
    )
    validate_summary_binding(
        treatment_baseline,
        treatment_manifest,
        treatment_root,
        name="treatment baseline",
        baseline=True,
    )
    validate_baseline_totals(
        control_root,
        control_baseline,
        name="control baseline",
        samples=sample_order,
    )
    validate_baseline_totals(
        treatment_root,
        treatment_baseline,
        name="treatment baseline",
        samples=sample_order,
    )
    validate_protocol_totals(
        control_protocol, name="control protocol", samples=sample_order
    )
    validate_protocol_totals(
        treatment_protocol, name="treatment protocol", samples=sample_order
    )
    for name, summary, version, arm in (
        ("control protocol", control_protocol, control_version, control_arm),
        ("treatment protocol", treatment_protocol, treatment_version, treatment_arm),
    ):
        schema_version = summary["schema_versions"][0]
        role = "control" if name.startswith("control") else "treatment"
        if schema_version != EXPECTED_SCHEMAS[role]:
            raise ValueError(f"{name} has unexpected schema version: {schema_version}")

    control_totals = control_protocol.get("totals")
    treatment_totals = treatment_protocol.get("totals")
    if not isinstance(control_totals, Mapping) or not isinstance(
        treatment_totals, Mapping
    ):
        raise ValueError("protocol summaries lack totals")
    if (
        nonnegative_int(control_totals["stage_calls"].get("select"), name="control.select_calls")
        != 0
        or nonnegative_int(
            treatment_totals["stage_calls"].get("select"), name="treatment.select_calls"
        )
        != 0
    ):
        raise ValueError("A/B live verifier must not issue select calls")
    count_paths = {
        "backend_calls": ("backend_call_count",),
        "select_calls": ("stage_calls", "select"),
        "audit_calls": ("stage_calls", "audit"),
        "decision_calls": ("stage_calls", "decision"),
        "repair_calls": ("attempt_kinds", "repair"),
        "regional_validation_errors": ("regional_validation_error_count",),
        "validation_errors": ("validation_error_count",),
        "verifier_invalid_steps": ("verifier_invalid_steps",),
        "localization_invalid_steps": ("localization_invalid_steps",),
        "atomic_audit_valid_records": ("atomic_audit_valid_records",),
        "raw_response_chars": ("raw_response_chars",),
        "backend_latency_seconds": ("backend_latency_seconds",),
        "episode_elapsed_seconds": ("episode_elapsed_seconds",),
        "action_attempt_errors": ("action_attempt_error_count",),
        "tool_actions": ("tool_action_count",),
    }
    protocol_deltas = {
        name: delta(
            nested_number(
                control_totals, path, name=f"control.{name}", nonnegative=True
            ),
            nested_number(
                treatment_totals, path, name=f"treatment.{name}", nonnegative=True
            ),
        )
        for name, path in count_paths.items()
    }
    metric_paths = {
        "aggregate_selected_iou": ("aggregate", "selected", "iou"),
        "aggregate_selected_f1": ("aggregate", "selected", "f1"),
        "aggregate_delta_iou": ("aggregate", "delta_iou"),
        "aggregate_delta_f1": ("aggregate", "delta_f1"),
        "invalid_or_no_action_samples": ("invalid_or_no_action_samples",),
        "degraded_sample_count": ("degraded_sample_count",),
        "accepted_candidates": ("actions", "accepted"),
        "tool_failure_count": ("actions", "tool_failure_count"),
    }
    outcome_deltas = {
        name: delta(
            nested_number(
                control_baseline,
                path,
                name=f"control.{name}",
                nonnegative=name
                in {
                    "invalid_or_no_action_samples",
                    "degraded_sample_count",
                    "accepted_candidates",
                    "tool_failure_count",
                },
            ),
            nested_number(
                treatment_baseline,
                path,
                name=f"treatment.{name}",
                nonnegative=name
                in {
                    "invalid_or_no_action_samples",
                    "degraded_sample_count",
                    "accepted_candidates",
                    "tool_failure_count",
                },
            ),
        )
        for name, path in metric_paths.items()
    }
    per_sample: dict[str, Any] = {}
    selected_metric_regressions: list[dict[str, str]] = []
    control_samples = control_baseline.get("per_sample")
    treatment_samples = treatment_baseline.get("per_sample")
    if not isinstance(control_samples, Mapping) or not isinstance(
        treatment_samples, Mapping
    ):
        raise ValueError("baseline summaries lack per-sample results")
    for sample in sample_order:
        left = control_samples.get(sample)
        right = treatment_samples.get(sample)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"missing per-sample result: {sample}")
        for side, result in (("control", left), ("treatment", right)):
            nonempty_string(
                result.get("episode_outcome"),
                name=f"{sample}.{side}.episode_outcome",
            )
            nonempty_string(
                result.get("stop_reason"), name=f"{sample}.{side}.stop_reason"
            )
        if not isinstance(left.get("degraded"), bool) or not isinstance(
            right.get("degraded"), bool
        ):
            raise ValueError(f"per-sample degraded flag is malformed: {sample}")
        control_protocol_sample = control_protocol["per_sample"].get(sample)
        treatment_protocol_sample = treatment_protocol["per_sample"].get(sample)
        if not isinstance(control_protocol_sample, Mapping) or not isinstance(
            treatment_protocol_sample, Mapping
        ):
            raise ValueError(f"missing protocol per-sample result: {sample}")
        control_initial_ids = region_ids(
            control_protocol_sample.get("initial_audited_region_ids"),
            name=f"{sample}.control.initial_audited_region_ids",
        )
        treatment_initial_ids = region_ids(
            treatment_protocol_sample.get("initial_audited_region_ids"),
            name=f"{sample}.treatment.initial_audited_region_ids",
        )
        if control_initial_ids != treatment_initial_ids:
            raise ValueError(f"initial audited region IDs differ between A/B arms: {sample}")
        control_initial = left.get("initial")
        treatment_initial = right.get("initial")
        if not isinstance(control_initial, Mapping) or not isinstance(
            treatment_initial, Mapping
        ):
            raise ValueError(f"initial metric block is malformed: {sample}")
        if any(
            control_initial.get(key) != treatment_initial.get(key)
            for key in (*METRIC_KEYS, *COUNT_KEYS)
        ):
            raise ValueError(f"initial offline metrics differ between A/B arms: {sample}")
        selected_iou = delta(
            nested_number(left, ("selected", "iou"), name=f"{sample}.control_iou"),
            nested_number(right, ("selected", "iou"), name=f"{sample}.treatment_iou"),
        )
        selected_f1 = delta(
            nested_number(left, ("selected", "f1"), name=f"{sample}.control_f1"),
            nested_number(right, ("selected", "f1"), name=f"{sample}.treatment_f1"),
        )
        for metric_name, values in (("iou", selected_iou), ("f1", selected_f1)):
            if float(values["treatment"]) < float(values["control"]) - 1e-8:
                selected_metric_regressions.append(
                    {"sample": sample, "metric": metric_name}
                )
        per_sample[str(sample)] = {
            "control_outcome": left.get("episode_outcome"),
            "treatment_outcome": right.get("episode_outcome"),
            "control_stop_reason": left.get("stop_reason"),
            "treatment_stop_reason": right.get("stop_reason"),
            "initial_iou": float(control_initial["iou"]),
            "initial_f1": float(control_initial["f1"]),
            "selected_iou": selected_iou,
            "selected_f1": selected_f1,
            "delta_iou": delta(
                nested_number(left, ("delta_iou",), name=f"{sample}.control_delta_iou"),
                nested_number(right, ("delta_iou",), name=f"{sample}.treatment_delta_iou"),
            ),
            "delta_f1": delta(
                nested_number(left, ("delta_f1",), name=f"{sample}.control_delta_f1"),
                nested_number(right, ("delta_f1",), name=f"{sample}.treatment_delta_f1"),
            ),
            "degraded": {
                "control": left["degraded"],
                "treatment": right["degraded"],
            },
            "initial_audited_region_ids": control_initial_ids,
        }
    return {
        "status": "completed",
        "evaluation_profile": EXPECTED_EVALUATION_PROFILE,
        "evaluation_id": EXPECTED_EVALUATION_ID,
        "control": str(control_root),
        "treatment": str(treatment_root),
        "sample_order": sample_order,
        "comparability": comparability,
        "arms": {
            "control": {
                "experiment_arm": control_manifest.get("experiment_arm"),
                "protocol_version": control_manifest.get("protocol_version"),
                "runtime_commits": control_manifest.get("runtime_commits"),
                "runtime_source_sha256": control_manifest.get(
                    "runtime_source_sha256"
                ),
            },
            "treatment": {
                "experiment_arm": treatment_manifest.get("experiment_arm"),
                "protocol_version": treatment_manifest.get("protocol_version"),
                "runtime_commits": treatment_manifest.get("runtime_commits"),
                "runtime_source_sha256": treatment_manifest.get(
                    "runtime_source_sha256"
                ),
            },
        },
        "protocol_deltas": protocol_deltas,
        "outcome_deltas": outcome_deltas,
        "selected_metric_regressions": selected_metric_regressions,
        "no_selected_metric_regression": not selected_metric_regressions,
        "per_sample": per_sample,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        comparison = summarize(args)
    except ValueError as error:
        print(f"local_verifier_ab: {error}", file=sys.stderr)
        return 3
    atomic_write(args.output, comparison)
    protocol = comparison["protocol_deltas"]
    outcome = comparison["outcome_deltas"]
    print(
        "comparison=completed;"
        f"audit_calls={protocol['audit_calls']['control']}"
        f"->{protocol['audit_calls']['treatment']};"
        f"regional_errors={protocol['regional_validation_errors']['control']}"
        f"->{protocol['regional_validation_errors']['treatment']};"
        f"invalid_or_no_action={outcome['invalid_or_no_action_samples']['control']}"
        f"->{outcome['invalid_or_no_action_samples']['treatment']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
