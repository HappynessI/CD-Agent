#!/usr/bin/env python3
"""Validate and summarize one fixed-sample verifier E2E run.

The rollout runner reports the selected-mask micro aggregate.  This helper also
reconstructs the initial-mask micro aggregate and records action/outcome totals
needed by the experiment registry.  It never opens the source GT images; all
counts come from post-rollout ``per_sample_metrics.json`` and trajectory
``offline_metrics`` written after the GT-free rollout marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SAMPLES = (
    "test_20_15",
    "test_78_13",
    "test_85_16",
    "test_1_1",
    "test_50_8",
)
EVALUATION_PROFILES = {
    "e2e5": ("DEV-EXTENDED-5-V1", DEFAULT_SAMPLES),
    "e2e3": (
        "DEV-CORE-3-V1",
        ("test_20_15", "test_1_1", "test_50_8"),
    ),
    "smoke1": ("DEV-SMOKE-1-V1", ("test_20_15",)),
    "p1_smoke1": ("DEV-SMOKE-1-H1-V1", ("test_1_1",)),
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
FINISH_REASONS = {
    "initial_verifier_authorized_finish",
    "verifier_authorized_finish",
    "confirmed_finish",
}
P1_AUDIT_SCHEMA_VERSION = "qwen3vl_p1_harness_audit_v1"
P1_FACTOR_ID = "terminal_review_scheduler_v1"
P1_MODEL_IO_PROTOCOL = "v12.1"
P1_SAFETY_BOUNDARIES = (
    "parser",
    "geometry_authorization",
    "locality_hard_gate",
    "candidate_benefit_without_harm",
    "rollback",
)
P1_BUDGET_COUNTER_FIELDS = (
    "verifier_round_count",
    "stage_initial_attempt_count",
    "schema_repair_attempt_count",
    "network_retry_attempt_count",
    "tool_action_attempt_count",
    "tool_failure_count",
    "environment_step_count",
)
P1_ARM_CONTRACTS = {
    "v12_1-h0-legacy-single-round-base-0802": {
        "arm": "h0",
        "harness_protocol_id": "qwen3vl_harness_h0_legacy_single_round_v1",
        "terminal_review": False,
        "configured_budgets": {
            "max_verifier_rounds": 1,
            "max_stage_attempts_per_round": 2,
            "max_environment_steps": 3,
            "effective_network_retry_limit": 0,
            "effective_tool_attempts_per_authorized_action": 1,
        },
    },
    "v12_1-h1-gtfree-terminal-review-base-0802": {
        "arm": "h1",
        "harness_protocol_id": "qwen3vl_harness_h1_gtfree_terminal_review_v1",
        "terminal_review": True,
        "configured_budgets": {
            "max_verifier_rounds": 2,
            "max_stage_attempts_per_round": 2,
            "max_environment_steps": 3,
            "effective_network_retry_limit": 0,
            "effective_tool_attempts_per_authorized_action": 1,
        },
    },
}
P1_ATTEMPT_KINDS = ("initial", "repair")
P1_COUNTER_SCOPE = {
    "verifier_round_count": "accepted_environment_state",
    "stage_initial_attempt_count": "accepted_environment_state",
    "schema_repair_attempt_count": "accepted_environment_state",
    "network_retry_attempt_count": "accepted_environment_state",
    "tool_action_attempt_count": "terminal_review_scheduler_only",
    "tool_failure_count": "terminal_review_scheduler_only",
    "environment_step_count": "episode_cumulative_snapshot",
}
P1_TRIGGER_INPUT_FIELDS = (
    "accepted_environment_state",
    "budget_counts",
    "round_one_protocol_result",
    "round_one_runtime_result",
)
P1_TRIGGERS = ("provisional_finish", "plan_null_after_allowed_repair")
P1_ROUND_KINDS = ("finish_confirmation", "plan_null_alternative")
P1_OUTCOMES = (
    "finish",
    "unique_safe_plan",
    "plan_null",
    "unresolved",
    "unsafe_or_nonunique_plan",
    "schema_invalid_after_allowed_repair",
    "budget_exhausted",
)
P1_TERMINAL_AUDIT_KEYS = (
    "schema_version",
    "factor_id",
    "policy",
    "accepted_state_id",
    "gt_free",
    "trigger_input_fields",
    "forbidden_trigger_inputs_present",
    "eligible",
    "trigger",
    "round_two_invoked",
    "round_two_count",
    "round_index",
    "round_two_kind",
    "observed_outcome",
    "effective_outcome",
    "transition",
    "fresh_backend_context",
    "inherits_hidden_state",
    "copied_round_one_raw_response",
    "nonduplicate_prompt_or_region",
    "round_three_allowed",
    "contract_violation",
    "safe_plan_returned_to_existing_flow",
    "tool_executed_by_scheduler",
    "counter_scope",
    "counters_before",
    "counters_after",
    "backend_calls",
)
P1_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
P1_STATE_ID_RE = P1_SHA256_RE
P1_SCAN_REGION_RE = re.compile(r"^s[0-9]+$")
P1_NON_SCAN_AUDIT_KINDS = {
    "present",
    "missing",
    "delta_added",
    "delta_removed",
    "mixed",
}
P1_EMPTY_MASK_SCAN_PROPOSAL_VERSION = (
    "qwen3vl_empty_dual_mask_scan_grid_v1"
)
P1_TERMINAL_REVIEW_CONTEXT_KEYS = (
    "round_index",
    "round_kind",
    "trigger",
    "fresh_backend_context",
    "inherits_hidden_state",
    "copies_round_one_raw_response",
    "selection_policy_version",
    "preferred_region_id",
    "prompt_version",
    "selection_branch",
    "review_region_id",
    "proposal_version",
)
P1_NORMALIZED_ROUND_TWO_CALL_KEYS = (
    "round_index",
    "stage",
    "region_id",
    "attempt_kind",
    "prompt_sha256",
    "raw_response",
    "parsed_output",
    "visual_input_scope",
    "network_retry_attempt_count",
)
P1_RAW_CALL_REQUIRED_KEYS = (
    "backend",
    "stage",
    "attempt_kind",
    "network_retry_attempt_count",
    "prompt_sha256",
    "raw_response",
    "visual_input_scope",
)
P1_FAIL_CLOSED_REASONS = {
    "round_two_already_used_for_accepted_state",
    "verifier_round_budget_exhausted_before_round_two",
    "no_environment_proposal_available_for_round_two",
    "no_eligible_plan_null_review_region",
}
P1_STAGED_SCHEMA_VERSION = "staged_verifier_environment_ordered_compact_audit_v12_1"
P1_PLAN_NULL_PROMPT_VERSIONS = {
    "qwen3vl_plan_null_next_unreviewed_region_v1": (
        "next_unreviewed_environment_region"
    ),
    "qwen3vl_plan_null_alternative_diagnosis_v1": (
        "first_unresolved_environment_region"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-profile", choices=tuple(EVALUATION_PROFILES), required=True
    )
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--model-state", choices=("base", "post_sft"), required=True)
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--experiment-arm", default="")
    parser.add_argument("--protocol-version", default="")
    parser.add_argument("--harness-protocol-id", default="")
    parser.add_argument("--runtime-commit", default="")
    parser.add_argument("--harness-commit", default="")
    parser.add_argument("--p1-harness-audit", type=Path)
    parser.add_argument("--expected-samples", nargs="+", default=list(DEFAULT_SAMPLES))
    return parser.parse_args(argv)


def read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"required artifact is missing: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"required artifact is not readable JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"required artifact must be a JSON object: {path}")
    return payload


def require_nonnegative_int(payload: Mapping[str, Any], key: str, *, source: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}.{key} must be a non-negative integer, found {value!r}")
    return value


def counts_from_metrics(payload: Mapping[str, Any], *, source: str) -> dict[str, int]:
    counts = {
        key: require_nonnegative_int(payload, key, source=source)
        for key in COUNT_KEYS
    }
    for key in METRIC_KEYS:
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{source}.{key} must be a finite rate, found {value!r}")
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


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, int | float]:
    rates = rates_from_counts(counts)
    return {
        **{key: round(rates[key], 8) for key in METRIC_KEYS},
        **{key: counts[key] for key in COUNT_KEYS},
    }


def add_counts(total: dict[str, int], values: Mapping[str, int]) -> None:
    for key in COUNT_KEYS:
        total[key] += values[key]


def count_deltas(
    initial: Mapping[str, int], selected: Mapping[str, int]
) -> dict[str, int]:
    return {key: selected[key] - initial[key] for key in DELTA_COUNT_KEYS}


def is_degraded(
    initial: Mapping[str, int], selected: Mapping[str, int]
) -> bool:
    """Apply the frozen metric-first degradation contract before rounding."""

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
    """Classify candidates using post-rollout counts and explicit runtime decisions."""

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
            # The next reference could be either the prior or candidate state.
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


def close_metrics(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(
        observed.get(key) == expected[key]
        if key in COUNT_KEYS
        else isinstance(observed.get(key), (int, float))
        and math.isclose(
            float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-8
        )
        for key in (*METRIC_KEYS, *COUNT_KEYS)
    )


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


def require_exact_keys(
    payload: Mapping[str, Any], keys: Sequence[str], *, source: str
) -> None:
    if set(payload) != set(keys):
        raise ValueError(
            f"{source} keys differ from the frozen contract: "
            f"{sorted(payload)!r} != {sorted(keys)!r}"
        )


def require_nonempty_string(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} must be a non-empty string, found {value!r}")
    return value


def require_string_list(value: Any, *, source: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{source} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{source} must not contain duplicates")
    return list(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash required artifact {path}: {error}") from error
    return digest.hexdigest()


def artifact_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required bound artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def normalize_budget_accounting(
    value: Any,
    *,
    source: str,
    configured: Mapping[str, int],
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be an object")
    require_exact_keys(value, P1_BUDGET_COUNTER_FIELDS, source=source)
    counters = {
        key: require_nonnegative_int(value, key, source=source)
        for key in P1_BUDGET_COUNTER_FIELDS
    }
    if counters["verifier_round_count"] > configured["max_verifier_rounds"]:
        raise ValueError(f"{source}.verifier_round_count exceeds its frozen budget")
    if counters["environment_step_count"] > configured["max_environment_steps"]:
        raise ValueError(f"{source}.environment_step_count exceeds its frozen budget")
    if (
        counters["network_retry_attempt_count"]
        > configured["effective_network_retry_limit"]
    ):
        raise ValueError(f"{source}.network_retry_attempt_count exceeds its frozen budget")
    if counters["schema_repair_attempt_count"] > counters["stage_initial_attempt_count"]:
        raise ValueError(f"{source} has more schema repairs than initial attempts")
    if counters["tool_failure_count"] > counters["tool_action_attempt_count"]:
        raise ValueError(f"{source} has more tool failures than tool attempts")
    return counters


def normalize_terminal_review_audit(
    value: Any,
    *,
    source: str,
    experiment_arm: str,
    arm_contract: Mapping[str, Any],
    budget_accounting: Mapping[str, int],
) -> dict[str, Any] | None:
    if value is None:
        if arm_contract["arm"] == "h1":
            raise ValueError(f"{source} must retain an H1 scheduler audit")
        return None
    if arm_contract["arm"] == "h0":
        raise ValueError(f"{source} must be null for the H0 identity arm")
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be an object or null")
    require_exact_keys(value, P1_TERMINAL_AUDIT_KEYS, source=source)
    policy = value.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{source}.policy must be an object")
    configured = arm_contract["configured_budgets"]
    expected_policy = {
        "factor_id": P1_FACTOR_ID,
        "arm": arm_contract["arm"],
        "harness_protocol_id": arm_contract["harness_protocol_id"],
        "experiment_arm_id": experiment_arm,
        "terminal_review": arm_contract["terminal_review"],
        **configured,
    }
    require_exact_keys(policy, tuple(expected_policy), source=f"{source}.policy")
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"{source}.policy.{key} differs from the frozen contract")
    if value.get("schema_version") != "qwen3vl_terminal_review_audit_v1":
        raise ValueError(f"{source}.schema_version is not frozen")
    if value.get("factor_id") != P1_FACTOR_ID:
        raise ValueError(f"{source}.factor_id is not frozen")
    if value.get("gt_free") is not True:
        raise ValueError(f"{source}.gt_free must be true")
    if value.get("forbidden_trigger_inputs_present") != []:
        raise ValueError(f"{source} contains forbidden trigger inputs")
    if value.get("trigger_input_fields") != list(P1_TRIGGER_INPUT_FIELDS):
        raise ValueError(f"{source}.trigger_input_fields differs from the frozen set")
    accepted_state_id = value.get("accepted_state_id")
    if (
        not isinstance(accepted_state_id, str)
        or P1_STATE_ID_RE.fullmatch(accepted_state_id) is None
    ):
        raise ValueError(f"{source}.accepted_state_id is malformed")
    if value.get("counter_scope") != P1_COUNTER_SCOPE:
        raise ValueError(f"{source}.counter_scope differs from the frozen semantics")
    for key in (
        "inherits_hidden_state",
        "copied_round_one_raw_response",
        "round_three_allowed",
        "tool_executed_by_scheduler",
    ):
        if value.get(key) is not False:
            raise ValueError(f"{source}.{key} must be false")
    invoked = value.get("round_two_invoked")
    if not isinstance(invoked, bool):
        raise ValueError(f"{source}.round_two_invoked must be a boolean")
    expected_round_two_count = 1 if invoked else 0
    round_two_count = value.get("round_two_count")
    if (
        isinstance(round_two_count, bool)
        or not isinstance(round_two_count, int)
        or round_two_count != expected_round_two_count
    ):
        raise ValueError(f"{source}.round_two_count disagrees with invocation")
    eligible = value.get("eligible")
    if not isinstance(eligible, bool):
        raise ValueError(f"{source}.eligible must be a boolean")
    trigger = value.get("trigger")
    observed_outcome = value.get("observed_outcome")
    effective_outcome = value.get("effective_outcome")
    contract_violation = value.get("contract_violation")
    if contract_violation is not None and (
        not isinstance(contract_violation, str) or not contract_violation
    ):
        raise ValueError(f"{source}.contract_violation is malformed")
    passthrough = trigger is None
    if passthrough:
        if (
            eligible
            or invoked
            or value.get("round_two_kind") is not None
            or observed_outcome is not None
            or effective_outcome is not None
            or value.get("transition") is not None
            or value.get("round_index") is not None
            or value.get("fresh_backend_context") is not None
            or value.get("nonduplicate_prompt_or_region") is not None
            or contract_violation is not None
            or value.get("safe_plan_returned_to_existing_flow") is not False
        ):
            raise ValueError(f"{source} has malformed non-trigger passthrough semantics")
    elif trigger not in P1_TRIGGERS:
        raise ValueError(f"{source}.trigger is not a registered H1 trigger")
    else:
        expected_kind = {
            "provisional_finish": "finish_confirmation",
            "plan_null_after_allowed_repair": "plan_null_alternative",
        }[trigger]
        if value.get("round_two_kind") != expected_kind:
            raise ValueError(f"{source}.round_two_kind disagrees with its trigger")
    if invoked:
        if not eligible:
            raise ValueError(f"{source} invoked round two while ineligible")
        if value.get("round_index") != 2:
            raise ValueError(f"{source}.round_index must be 2")
        if value.get("fresh_backend_context") is not True:
            raise ValueError(f"{source}.fresh_backend_context must be true")
        if value.get("nonduplicate_prompt_or_region") is not True:
            raise ValueError(f"{source} did not retain independent round-two evidence")
        if observed_outcome not in P1_OUTCOMES or effective_outcome not in P1_OUTCOMES:
            raise ValueError(f"{source} has an unregistered round-two outcome")
        if contract_violation not in {None, "environment_step_budget_exhausted_after_review"}:
            raise ValueError(f"{source} has an unregistered invoked-round violation")
    elif not passthrough:
        if value.get("round_index") is not None:
            raise ValueError(f"{source}.round_index must be null when round two was not invoked")
        if (
            not eligible
            or observed_outcome is not None
            or effective_outcome != "budget_exhausted"
            or contract_violation not in P1_FAIL_CLOSED_REASONS
            or value.get("fresh_backend_context") is not None
            or value.get("nonduplicate_prompt_or_region") is not None
        ):
            raise ValueError(f"{source} lacks a structured fail-closed transition")
    raw_before = value.get("counters_before")
    raw_after = value.get("counters_after")
    if not isinstance(raw_before, Mapping) or not isinstance(raw_after, Mapping):
        raise ValueError(f"{source} lacks before/after budget snapshots")
    before = normalize_budget_accounting(
        raw_before,
        source=f"{source}.counters_before",
        configured=configured,
    )
    after = normalize_budget_accounting(
        raw_after,
        source=f"{source}.counters_after",
        configured=configured,
    )
    if any(after[key] < before[key] for key in P1_BUDGET_COUNTER_FIELDS):
        raise ValueError(f"{source} budget snapshots are not monotonic")
    if any(after[key] > budget_accounting[key] for key in P1_BUDGET_COUNTER_FIELDS):
        raise ValueError(f"{source} snapshot exceeds final episode accounting")
    if before["verifier_round_count"] != 1 or after["verifier_round_count"] != (
        1 + int(invoked)
    ):
        raise ValueError(f"{source} does not use one round-one plus optional round-two")
    if before["tool_action_attempt_count"] or after["tool_action_attempt_count"]:
        raise ValueError(f"{source} scheduler scope contains a tool attempt")
    if before["tool_failure_count"] or after["tool_failure_count"]:
        raise ValueError(f"{source} scheduler scope contains a tool failure")
    if before["environment_step_count"] != after["environment_step_count"]:
        raise ValueError(f"{source} terminal review changed Environment steps")
    if invoked:
        exhausted_after_review = (
            observed_outcome == "unique_safe_plan"
            and before["environment_step_count"]
            >= configured["max_environment_steps"]
        )
        expected_effective = (
            "budget_exhausted" if exhausted_after_review else observed_outcome
        )
        expected_violation = (
            "environment_step_budget_exhausted_after_review"
            if exhausted_after_review
            else None
        )
        if effective_outcome != expected_effective or contract_violation != expected_violation:
            raise ValueError(f"{source} effective outcome/violation differs")
    calls = value.get("backend_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{source}.backend_calls must be a list")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, call in enumerate(calls):
        call_source = f"{source}.backend_calls[{index}]"
        if not isinstance(call, Mapping):
            raise ValueError(f"{call_source} must be an object")
        require_exact_keys(
            call,
            P1_NORMALIZED_ROUND_TWO_CALL_KEYS,
            source=call_source,
        )
        if call.get("round_index") != 2:
            raise ValueError(f"{call_source}.round_index must be 2")
        stage = require_nonempty_string(call.get("stage"), source=f"{call_source}.stage")
        region = require_nonempty_string(
            call.get("region_id"), source=f"{call_source}.region_id"
        )
        attempt_kind = call.get("attempt_kind")
        if attempt_kind not in P1_ATTEMPT_KINDS:
            raise ValueError(f"{call_source}.attempt_kind is not frozen")
        network_retries = call.get("network_retry_attempt_count")
        if (
            isinstance(network_retries, bool)
            or not isinstance(network_retries, int)
            or network_retries < 0
            or network_retries > configured["effective_network_retry_limit"]
        ):
            raise ValueError(f"{call_source}.network_retry_attempt_count is invalid")
        if not isinstance(call.get("raw_response"), str):
            raise ValueError(f"{call_source}.raw_response must be retained as text")
        parsed_output = call.get("parsed_output")
        if parsed_output is not None and (
            not isinstance(parsed_output, Mapping)
            or parsed_output.get("region_id") != region
        ):
            raise ValueError(f"{call_source}.parsed_output differs from region_id")
        visual_input_scope = call.get("visual_input_scope")
        if not isinstance(visual_input_scope, str) or not visual_input_scope:
            raise ValueError(
                f"{call_source}.visual_input_scope must be retained"
            )
        prompt_sha256 = call.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or P1_SHA256_RE.fullmatch(prompt_sha256) is None:
            raise ValueError(f"{call_source}.prompt_sha256 is malformed")
        grouped.setdefault((stage, region), []).append(call)
    for key, grouped_calls in grouped.items():
        attempts = [call["attempt_kind"] for call in grouped_calls]
        if len(attempts) > configured["max_stage_attempts_per_round"]:
            raise ValueError(f"{source} stage/region {key!r} exceeds its attempt budget")
        if attempts[0] != "initial" or attempts.count("initial") != 1:
            raise ValueError(f"{source} stage/region {key!r} lacks one leading initial")
        if attempts.count("repair") > 1:
            raise ValueError(f"{source} stage/region {key!r} exceeds repair budget")
        if grouped_calls[-1].get("parsed_output") is None or (
            len(grouped_calls) == 2
            and grouped_calls[0].get("parsed_output") is not None
        ):
            raise ValueError(f"{source} stage/region {key!r} has invalid parse evidence")
    deltas = {key: after[key] - before[key] for key in P1_BUDGET_COUNTER_FIELDS}
    expected_deltas = {
        "verifier_round_count": expected_round_two_count,
        "stage_initial_attempt_count": sum(
            call.get("attempt_kind") == "initial" for call in calls
        ),
        "schema_repair_attempt_count": sum(
            call.get("attempt_kind") == "repair" for call in calls
        ),
        "network_retry_attempt_count": sum(
            int(call.get("network_retry_attempt_count", 0)) for call in calls
        ),
        "tool_action_attempt_count": 0,
        "tool_failure_count": 0,
        "environment_step_count": 0,
    }
    if deltas != expected_deltas:
        raise ValueError(f"{source} counter deltas disagree with round-two calls")
    if passthrough:
        if calls or before != after:
            raise ValueError(f"{source} passthrough consumed a scheduler budget")
    else:
        expected_transition = _expected_p1_transition(trigger, effective_outcome)
        if value.get("transition") != expected_transition:
            raise ValueError(f"{source}.transition differs from the frozen table")
        safe_plan = effective_outcome == "unique_safe_plan"
        if value.get("safe_plan_returned_to_existing_flow") is not safe_plan:
            raise ValueError(f"{source}.safe_plan_returned_to_existing_flow differs")
    return dict(value)


def _expected_p1_transition(trigger: str, outcome: str) -> dict[str, str]:
    if trigger == "provisional_finish" and outcome == "finish":
        decision = "confirmed_finish"
        next_state = "terminate_confirmed_finish"
        authorization = "forbidden"
    elif outcome == "unique_safe_plan":
        decision = "safe_plan_enters_existing_flow"
        next_state = "existing_environment_action_and_candidate_flow"
        authorization = "not_yet_authorized_existing_safety_flow_required"
    elif (
        trigger == "provisional_finish" and outcome == "plan_null"
        or trigger == "plan_null_after_allowed_repair" and outcome == "finish"
    ):
        decision = "round_conflict_no_action"
        next_state = "terminate_no_executable_action_after_review"
        authorization = "forbidden"
    else:
        decision = "no_executable_action_after_review"
        next_state = "terminate_no_executable_action_after_review"
        authorization = "forbidden"
    return {
        "round_one_trigger": trigger,
        "round_two_outcome": outcome,
        "review_decision": decision,
        "next_state": next_state,
        "tool_action_authorization": authorization,
    }


def normalize_gt_boundary_audit(value: Any, *, source: str) -> dict[str, Any]:
    keys = (
        "pre_rollout_gt_access_count",
        "violation_ids",
        "rollout_completed_before_offline_scoring",
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be an object")
    require_exact_keys(value, keys, source=source)
    count = require_nonnegative_int(value, keys[0], source=source)
    violation_ids = require_string_list(value.get("violation_ids"), source=f"{source}.violation_ids")
    if count != len(violation_ids):
        raise ValueError(f"{source} violation count differs from violation IDs")
    if value.get("rollout_completed_before_offline_scoring") is not True:
        raise ValueError(f"{source} does not prove the offline-scoring boundary")
    if count:
        raise ValueError(f"{source} records pre-rollout GT access")
    return {
        "pre_rollout_gt_access_count": count,
        "violation_ids": violation_ids,
        "rollout_completed_before_offline_scoring": True,
    }


def normalize_safety_boundary_audit(value: Any, *, source: str) -> dict[str, Any]:
    keys = ("relaxation_count", "relaxed_boundaries")
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be an object")
    require_exact_keys(value, keys, source=source)
    count = require_nonnegative_int(value, "relaxation_count", source=source)
    relaxed = require_string_list(
        value.get("relaxed_boundaries"), source=f"{source}.relaxed_boundaries"
    )
    if count != len(relaxed):
        raise ValueError(f"{source} relaxation count differs from boundary IDs")
    if any(item not in P1_SAFETY_BOUNDARIES for item in relaxed):
        raise ValueError(f"{source} names an unknown safety boundary")
    if count:
        raise ValueError(f"{source} records a frozen safety-boundary relaxation")
    return {"relaxation_count": count, "relaxed_boundaries": relaxed}


def terminal_review_records_from_trajectory(
    trajectory: Mapping[str, Any], *, source: str
) -> list[dict[str, Any]]:
    """Return each terminal-review envelope exactly once in trajectory order."""

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{source}.steps must be a list")
    records: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"{source}.steps[{index}] must be an object")
        execution = step.get("execution")
        if not isinstance(execution, Mapping):
            continue
        post_evidence: Any = None
        if "post_accept_terminal_review" in execution:
            post = execution.get("post_accept_terminal_review")
            if not isinstance(post, Mapping) or "verifier_evidence" not in post:
                raise ValueError(
                    f"{source}.steps[{index}] post-review envelope is malformed"
                )
            post_evidence = post.get("verifier_evidence")
            if not isinstance(post_evidence, Mapping):
                raise ValueError(
                    f"{source}.steps[{index}] post-review evidence is malformed"
                )
            selected = post_evidence
        else:
            selected = execution.get("verifier_evidence")
            if "verifier_evidence" in execution and not isinstance(selected, Mapping):
                raise ValueError(
                    f"{source}.steps[{index}] verifier evidence is malformed"
                )
        selected_blocks: list[Mapping[str, Any]] = []
        if isinstance(selected, Mapping):
            selected_blocks.append(selected)
        if "rollback_replan" in execution:
            rollback = execution.get("rollback_replan")
            if not isinstance(rollback, Mapping) or "verifier_evidence" not in rollback:
                raise ValueError(
                    f"{source}.steps[{index}] rollback envelope is malformed"
                )
            rollback_evidence = rollback.get("verifier_evidence")
            if not isinstance(rollback_evidence, Mapping):
                raise ValueError(
                    f"{source}.steps[{index}] rollback evidence is malformed"
                )
            selected_blocks.append(rollback_evidence)
        for block_index, evidence in enumerate(selected_blocks):
            if "terminal_review" not in evidence:
                continue
            review = evidence.get("terminal_review")
            if not isinstance(review, Mapping):
                raise ValueError(
                    f"{source}.steps[{index}].terminal_review[{block_index}] is malformed"
                )
            round_one = review.get("round_one_evidence")
            if not isinstance(round_one, Mapping):
                raise ValueError(
                    f"{source}.steps[{index}] terminal review lacks round-one evidence"
                )
            round_two = review.get("round_two_evidence")
            if round_two is not None and not isinstance(round_two, Mapping):
                raise ValueError(
                    f"{source}.steps[{index}] terminal review round-two evidence is malformed"
                )
            bare = {
                key: value
                for key, value in review.items()
                if key not in {"round_one_evidence", "round_two_evidence"}
            }
            records.append(
                {
                    "audit": bare,
                    "round_one_evidence": round_one,
                    "round_two_evidence": round_two,
                }
            )
    return records


def _protocol_region_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        region_id = item.get("region_id") if isinstance(item, Mapping) else None
        if not isinstance(region_id, str) or not region_id or region_id in result:
            return None
        expected_version = _protocol_proposal_version(region_id)
        if (
            item.get("proposal_version") != expected_version
            or (
                item.get("audit_kind") != "scan"
                if expected_version is not None
                else item.get("audit_kind") not in P1_NON_SCAN_AUDIT_KINDS
            )
        ):
            return None
        result.append(region_id)
    return tuple(result)


def _protocol_proposal_version(region_id: str) -> str | None:
    return (
        P1_EMPTY_MASK_SCAN_PROPOSAL_VERSION
        if P1_SCAN_REGION_RE.fullmatch(region_id) is not None
        else None
    )


def _protocol_visual_scope_valid(
    *, scan: bool, attempt_kind: Any, visual_input_scope: Any
) -> bool:
    if not isinstance(visual_input_scope, str) or not visual_input_scope:
        return False
    if attempt_kind == "initial":
        expected = (
            "authoritative_pair_rgb_only_scan" if scan else "standard"
        )
        return visual_input_scope == expected
    if attempt_kind == "repair":
        expected = (
            "authoritative_pair_rgb_only_scan" if scan else "standard"
        )
        return visual_input_scope in {
            expected,
            "authoritative_pair_rgb_only_evidence_repair",
        }
    return False


def _protocol_trace_region_contracts(
    trace: Mapping[str, Any],
) -> dict[str, tuple[str, str | None]] | None:
    mode = trace.get("mode")
    raw_groups = (
        (trace.get("evidence"),)
        if mode == "initial"
        else (trace.get("evidence"), trace.get("replan_evidence"))
        if mode == "candidate"
        else ()
    )
    if not raw_groups:
        return None
    result: dict[str, tuple[str, str | None]] = {}
    for raw_records in raw_groups:
        region_ids = _protocol_region_ids(raw_records)
        if region_ids is None or not isinstance(raw_records, list):
            return None
        for raw_record, region_id in zip(raw_records, region_ids, strict=True):
            if region_id in result:
                return None
            result[region_id] = (
                str(raw_record["audit_kind"]),
                raw_record.get("proposal_version"),
            )
    return result or None


def _protocol_trace_proposal_versions(
    trace: Mapping[str, Any],
) -> dict[str, str | None] | None:
    contracts = _protocol_trace_region_contracts(trace)
    return (
        {region_id: contract[1] for region_id, contract in contracts.items()}
        if contracts is not None
        else None
    )


def _protocol_id_list(value: Any) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return tuple(value)


def _protocol_records_match(value: Any, ids: tuple[str, ...]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(ids)
        and all(
            isinstance(item, Mapping) and item.get("region_id") == region_id
            for item, region_id in zip(value, ids, strict=True)
        )
    )


def _protocol_trace_groups(
    trace: Mapping[str, Any],
) -> tuple[tuple[str, str], ...] | None:
    if "plan" not in trace:
        return None
    mode = trace.get("mode")
    if mode == "initial":
        ids = _protocol_region_ids(trace.get("evidence"))
        if (
            not ids
            or _protocol_id_list(trace.get("audited_region_ids")) != ids
            or not _protocol_records_match(trace.get("judgments"), ids)
            or not _protocol_records_match(trace.get("diagnoses"), ids)
        ):
            return None
        return tuple(("audit", region_id) for region_id in ids)
    if mode != "candidate":
        return None
    candidate_ids = _protocol_region_ids(trace.get("evidence"))
    if (
        not candidate_ids
        or _protocol_id_list(trace.get("audited_region_ids")) != candidate_ids
        or not _protocol_records_match(trace.get("judgments"), candidate_ids)
        or trace.get("diagnoses") != []
    ):
        return None
    groups = [("candidate_evidence", region_id) for region_id in candidate_ids]
    replan_ids = _protocol_region_ids(trace.get("replan_evidence"))
    if replan_ids is None:
        return None
    if replan_ids:
        if (
            _protocol_id_list(trace.get("replan_audited_region_ids")) != replan_ids
            or not _protocol_records_match(trace.get("replan_judgments"), replan_ids)
            or not _protocol_records_match(trace.get("replan_diagnoses"), replan_ids)
        ):
            return None
        groups.extend(("audit", region_id) for region_id in replan_ids)
    elif any(
        trace.get(key) != []
        for key in (
            "replan_audited_region_ids",
            "replan_judgments",
            "replan_diagnoses",
        )
    ):
        return None
    return tuple(groups)


def staged_region_protocol_valid_from_raw_evidence(
    evidence: Any,
    *,
    require_round_two: bool,
) -> bool:
    """Independently reconstruct one schema-valid complete staged region result."""

    if (
        not isinstance(evidence, Mapping)
        or evidence.get("type") != "staged_qwen_verifier"
        or evidence.get("schema_version") != P1_STAGED_SCHEMA_VERSION
        or evidence.get("verifier_valid") is not True
        or evidence.get("validation_errors") != []
        or evidence.get("regional_validation_errors") != []
    ):
        return False
    trace = evidence.get("stage_trace")
    calls = evidence.get("backend_calls")
    if not isinstance(trace, Mapping) or not isinstance(calls, list):
        return False
    groups = _protocol_trace_groups(trace)
    region_contracts = _protocol_trace_region_contracts(trace)
    if not groups or region_contracts is None:
        return False
    if require_round_two:
        meta = evidence.get("terminal_review_round")
        if not isinstance(meta, Mapping) or meta.get("schema_valid") is not True:
            return False
    call_index = 0
    for expected_stage, expected_region in groups:
        if call_index >= len(calls):
            return False
        initial = calls[call_index]
        if (
            not isinstance(initial, Mapping)
            or initial.get("stage") != expected_stage
            or initial.get("attempt_kind") != "initial"
        ):
            return False
        attempt_group = [initial]
        call_index += 1
        if (
            call_index < len(calls)
            and isinstance(calls[call_index], Mapping)
            and calls[call_index].get("stage") == expected_stage
            and calls[call_index].get("attempt_kind") == "repair"
        ):
            attempt_group.append(calls[call_index])
            call_index += 1
        parsed_states: list[bool] = []
        for call in attempt_group:
            retry_count = call.get("network_retry_attempt_count")
            prompt = call.get("prompt_sha256")
            raw_response = call.get("raw_response")
            if (
                any(key not in call for key in P1_RAW_CALL_REQUIRED_KEYS)
                or not isinstance(call.get("backend"), str)
                or not call.get("backend")
                or not _protocol_visual_scope_valid(
                    scan=region_contracts[expected_region][0] == "scan",
                    attempt_kind=call.get("attempt_kind"),
                    visual_input_scope=call.get("visual_input_scope"),
                )
                or isinstance(retry_count, bool)
                or retry_count != 0
                or not isinstance(prompt, str)
                or P1_SHA256_RE.fullmatch(prompt) is None
                or "raw_response" not in call
                or not isinstance(raw_response, str)
            ):
                return False
            if "parsed_output" in call:
                parsed = call.get("parsed_output")
                if (
                    "parse_error" in call
                    or not isinstance(parsed, Mapping)
                    or parsed.get("region_id") != expected_region
                ):
                    return False
                parsed_states.append(True)
            else:
                if (
                    not isinstance(call.get("parse_error"), str)
                    or not call.get("parse_error")
                ):
                    return False
                parsed_states.append(False)
            context = call.get("terminal_review")
            if require_round_two:
                if (
                    not isinstance(context, Mapping)
                    or set(context) != set(P1_TERMINAL_REVIEW_CONTEXT_KEYS)
                    or context.get("round_index") != 2
                    or context.get("review_region_id") != expected_region
                    or context.get("proposal_version")
                    != region_contracts[expected_region][1]
                    or context.get("fresh_backend_context") is not True
                    or context.get("inherits_hidden_state") is not False
                    or context.get("copies_round_one_raw_response") is not False
                    or context.get("selection_policy_version")
                    != "environment_ordered_terminal_review_v1"
                ):
                    return False
                trigger = context.get("trigger")
                if trigger == "plan_null_after_allowed_repair":
                    prompt_version = context.get("prompt_version")
                    if (
                        len(groups) != 1
                        or context.get("round_kind") != "plan_null_alternative"
                        or context.get("preferred_region_id") != expected_region
                        or P1_PLAN_NULL_PROMPT_VERSIONS.get(prompt_version)
                        != context.get("selection_branch")
                    ):
                        return False
                elif trigger == "provisional_finish":
                    if (
                        context.get("round_kind") != "finish_confirmation"
                        or context.get("preferred_region_id") is not None
                        or context.get("prompt_version") is not None
                        or context.get("selection_branch") is not None
                    ):
                        return False
                else:
                    return False
            elif "terminal_review" in call:
                return False
        if not parsed_states[-1] or (
            len(parsed_states) == 2 and parsed_states[0]
        ):
            return False
    return call_index == len(calls)


def _round_one_attempt_groups(
    evidence: Any,
) -> tuple[tuple[str, str, tuple[Mapping[str, Any], ...]], ...] | None:
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("type") != "staged_qwen_verifier"
        or evidence.get("schema_version") != P1_STAGED_SCHEMA_VERSION
        or evidence.get("verifier_valid") is not True
        or evidence.get("validation_errors") != []
    ):
        return None
    trace = evidence.get("stage_trace")
    calls = evidence.get("backend_calls")
    if not isinstance(trace, Mapping) or not isinstance(calls, list):
        return None
    expected = _protocol_trace_groups(trace)
    region_contracts = _protocol_trace_region_contracts(trace)
    if not expected or region_contracts is None:
        return None
    result: list[tuple[str, str, tuple[Mapping[str, Any], ...]]] = []
    call_index = 0
    for stage, region_id in expected:
        if call_index >= len(calls):
            return None
        initial = calls[call_index]
        if (
            not isinstance(initial, Mapping)
            or initial.get("stage") != stage
            or initial.get("attempt_kind") != "initial"
        ):
            return None
        attempts = [initial]
        call_index += 1
        if (
            call_index < len(calls)
            and isinstance(calls[call_index], Mapping)
            and calls[call_index].get("stage") == stage
            and calls[call_index].get("attempt_kind") == "repair"
        ):
            attempts.append(calls[call_index])
            call_index += 1
        parsed_states: list[bool] = []
        for call in attempts:
            retry_count = call.get("network_retry_attempt_count")
            prompt = call.get("prompt_sha256")
            raw_response = call.get("raw_response")
            if (
                any(key not in call for key in P1_RAW_CALL_REQUIRED_KEYS)
                or not isinstance(call.get("backend"), str)
                or not call.get("backend")
                or "terminal_review" in call
                or not _protocol_visual_scope_valid(
                    scan=region_contracts[region_id][0] == "scan",
                    attempt_kind=call.get("attempt_kind"),
                    visual_input_scope=call.get("visual_input_scope"),
                )
                or isinstance(retry_count, bool)
                or retry_count != 0
                or not isinstance(prompt, str)
                or P1_SHA256_RE.fullmatch(prompt) is None
                or "raw_response" not in call
                or not isinstance(raw_response, str)
            ):
                return None
            if "parsed_output" in call:
                parsed = call.get("parsed_output")
                if (
                    "parse_error" in call
                    or not isinstance(parsed, Mapping)
                    or parsed.get("region_id") != region_id
                ):
                    return None
                parsed_states.append(True)
            else:
                if (
                    not isinstance(call.get("parse_error"), str)
                    or not call.get("parse_error")
                ):
                    return None
                parsed_states.append(False)
        if not parsed_states[-1] or (
            len(parsed_states) == 2 and parsed_states[0]
        ):
            return None
        result.append((stage, region_id, tuple(attempts)))
    if call_index != len(calls):
        return None
    regional_errors = evidence.get("regional_validation_errors")
    if not isinstance(regional_errors, list):
        return None
    repair_regions = {
        region_id
        for stage, region_id, attempts in result
        if stage == "audit" and len(attempts) == 2
    }
    seen: set[str] = set()
    for item in regional_errors:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("region_id"), str)
            or not isinstance(item.get("error"), str)
            or not item.get("error")
            or item["region_id"] in seen
            or item["region_id"] not in repair_regions
        ):
            return None
        seen.add(str(item["region_id"]))
    return tuple(result)


def rebuild_terminal_trigger_from_round_one(evidence: Any) -> str | None:
    attempts = _round_one_attempt_groups(evidence)
    if attempts is None or not isinstance(evidence, Mapping):
        return None
    trace = evidence.get("stage_trace")
    if not isinstance(trace, Mapping):
        return None
    plan = trace.get("plan")
    mode = trace.get("mode")
    diagnosis_key = "diagnoses" if mode == "initial" else "replan_diagnoses"
    diagnoses = trace.get(diagnosis_key)
    if not isinstance(diagnoses, list):
        return None
    decision = trace.get("decision")
    production_trace_decision = bool(
        isinstance(decision, Mapping)
        and (
            (
                mode == "initial"
                and decision.get("comparison") == "initial"
                and decision.get("accept") is False
                and decision.get("stop") is False
            )
            or (
                mode == "candidate"
                and decision.get("comparison") == "better"
                and decision.get("accept") is True
                and decision.get("stop") is False
            )
        )
    )
    if (
        isinstance(plan, Mapping)
        and plan.get("action") == "finish"
        and trace.get("state_completion_gate_passed") is True
        and diagnoses
        and all(
            isinstance(item, Mapping) and item.get("error_type") == "none"
            for item in diagnoses
        )
        # StageTrace retains the protocol decision before _output() derives
        # the public VerifierOutput.  Initial decisions are deliberately
        # false/false; a complete error-free finish plan is what makes the
        # public output accept/stop true.  Candidate decisions similarly keep
        # stop=false and let _output() derive terminal stop.
        and production_trace_decision
        and evidence.get("regional_validation_errors") == []
    ):
        return "provisional_finish"
    if "plan" in trace and plan is None:
        return "plan_null_after_allowed_repair"
    return None


def rebuild_round_two_outcome(evidence: Any) -> str | None:
    if not isinstance(evidence, Mapping):
        return None
    meta = evidence.get("terminal_review_round")
    trace = evidence.get("stage_trace")
    if not isinstance(meta, Mapping) or not isinstance(trace, Mapping):
        return None
    if meta.get("schema_valid") is not True:
        return "schema_invalid_after_allowed_repair"
    diagnoses = trace.get("diagnoses")
    if not isinstance(diagnoses, list) or not diagnoses:
        return None
    error_types = [
        item.get("error_type") if isinstance(item, Mapping) else None
        for item in diagnoses
    ]
    if any(not isinstance(item, str) for item in error_types):
        return None
    safe_count = meta.get("safe_plan_candidate_count")
    if isinstance(safe_count, bool) or not isinstance(safe_count, int) or safe_count < 0:
        return None
    plan = trace.get("plan")
    action = plan.get("action") if isinstance(plan, Mapping) else None
    if "uncertain_region" in error_types:
        return "unresolved"
    if (
        safe_count == 1
        and sum(item != "none" for item in error_types) == 1
        and action in {"positive_point", "negative_point", "box"}
    ):
        return "unique_safe_plan"
    if safe_count > 0 or any(
        item in {"false_positive_change", "false_negative", "mixed_error"}
        for item in error_types
    ):
        return "unsafe_or_nonunique_plan"
    if (
        meta.get("completion_gate_passed") is True
        and action == "finish"
        and all(item == "none" for item in error_types)
    ):
        return "finish"
    return "plan_null"


def rebuild_final_region_protocol_valid(
    trajectory: Mapping[str, Any],
    *,
    arm: str,
    review_records: Sequence[Mapping[str, Any]],
) -> bool:
    """Rebuild the declared gate from the hash-bound trajectory only."""

    if arm not in {"h0", "h1"} or (arm == "h0" and review_records):
        return False
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return False
    final_block: Any = None
    block_seen = False
    for step in steps:
        if not isinstance(step, Mapping):
            return False
        execution = step.get("execution")
        if not isinstance(execution, Mapping):
            continue
        if "post_accept_terminal_review" in execution:
            post = execution.get("post_accept_terminal_review")
            final_block = (
                post.get("verifier_evidence") if isinstance(post, Mapping) else None
            )
            block_seen = True
        elif "verifier_evidence" in execution:
            final_block = execution.get("verifier_evidence")
            block_seen = True
        if "rollback_replan" in execution:
            rollback = execution.get("rollback_replan")
            final_block = (
                rollback.get("verifier_evidence")
                if isinstance(rollback, Mapping)
                else None
            )
            block_seen = True
    if not block_seen:
        return False
    if arm == "h1":
        raw_review = (
            final_block.get("terminal_review")
            if isinstance(final_block, Mapping)
            else None
        )
        if not isinstance(raw_review, Mapping) or not review_records:
            return False
        final_record = {
            "audit": {
                key: value
                for key, value in raw_review.items()
                if key not in {"round_one_evidence", "round_two_evidence"}
            },
            "round_one_evidence": raw_review.get("round_one_evidence"),
            "round_two_evidence": raw_review.get("round_two_evidence"),
        }
        if final_record != review_records[-1]:
            return False
        audit = final_record["audit"]
        if (
            audit.get("eligible") is not True
            or audit.get("round_two_invoked") is not True
            or audit.get("round_two_count") != 1
            or audit.get("round_index") != 2
        ):
            return False
        return staged_region_protocol_valid_from_raw_evidence(
            final_record["round_two_evidence"],
            require_round_two=True,
        )
    return staged_region_protocol_valid_from_raw_evidence(
        final_block,
        require_round_two=False,
    )


def validate_terminal_review_independence(
    record: Mapping[str, Any], audit: Mapping[str, Any], *, source: str
) -> None:
    round_two = record.get("round_two_evidence")
    if not audit["round_two_invoked"]:
        if round_two is not None or audit["backend_calls"]:
            raise ValueError(f"{source} claims backend activity without invocation")
        return
    if not isinstance(round_two, Mapping):
        raise ValueError(f"{source} lacks raw round-two evidence")
    raw_round_two_calls = round_two.get("backend_calls")
    if not isinstance(raw_round_two_calls, list):
        raise ValueError(f"{source} raw round-two calls are malformed")
    if len(raw_round_two_calls) != len(audit["backend_calls"]):
        raise ValueError(f"{source} normalized/raw round-two call counts differ")
    round_one = record.get("round_one_evidence")
    if not isinstance(round_one, Mapping):
        raise ValueError(f"{source} lacks raw round-one evidence")
    round_one_groups = _round_one_attempt_groups(round_one)
    if round_one_groups is None:
        raise ValueError(f"{source} raw round-one calls are malformed")
    round_two_trace = round_two.get("stage_trace")
    region_contracts = (
        _protocol_trace_region_contracts(round_two_trace)
        if isinstance(round_two_trace, Mapping)
        else None
    )
    if region_contracts is None:
        raise ValueError(f"{source} round-two proposal versions are malformed")

    prior_identities: set[tuple[str, str]] = set()
    for _, expected_region, attempts in round_one_groups:
        for call in attempts:
            parsed = call.get("parsed_output")
            region = (
                parsed["region_id"]
                if isinstance(parsed, Mapping)
                else expected_region
            )
            prompt = call.get("prompt_sha256")
            if not isinstance(region, str) or not isinstance(prompt, str):
                raise ValueError(
                    f"{source} validated round-one identity is malformed"
                )
            prior_identities.add((region, prompt))
    round_two_identities: set[tuple[str, str]] = set()
    expected_plan_null_selection: tuple[str, str, str] | None = None
    if audit.get("trigger") == "plan_null_after_allowed_repair":
        trace = round_one.get("stage_trace")
        if not isinstance(trace, Mapping):
            raise ValueError(f"{source} plan-null round one lacks a stage trace")
        if trace.get("mode") == "initial":
            raw_regions = trace.get("evidence")
            raw_audited = trace.get("audited_region_ids")
            raw_diagnoses = trace.get("diagnoses")
        elif trace.get("mode") == "candidate":
            raw_regions = trace.get("replan_evidence")
            raw_audited = trace.get("replan_audited_region_ids")
            raw_diagnoses = trace.get("replan_diagnoses")
        else:
            raise ValueError(f"{source} plan-null trace mode is malformed")
        region_ids = _protocol_region_ids(raw_regions)
        audited_ids = _protocol_id_list(raw_audited)
        if not region_ids or audited_ids is None:
            raise ValueError(f"{source} plan-null region order is malformed")
        next_unreviewed = next(
            (region_id for region_id in region_ids if region_id not in audited_ids),
            None,
        )
        if next_unreviewed is not None:
            expected_plan_null_selection = (
                next_unreviewed,
                "qwen3vl_plan_null_next_unreviewed_region_v1",
                "next_unreviewed_environment_region",
            )
        else:
            diagnosis_by_region = {
                item.get("region_id"): item
                for item in raw_diagnoses
                if isinstance(item, Mapping)
                and isinstance(item.get("region_id"), str)
            } if isinstance(raw_diagnoses, list) else {}
            malformed = {
                item.get("region_id")
                for item in round_one.get("regional_validation_errors", [])
                if isinstance(item, Mapping)
                and isinstance(item.get("region_id"), str)
            }
            unresolved = next(
                (
                    region_id
                    for region_id in region_ids
                    if region_id in malformed
                    or diagnosis_by_region.get(region_id, {}).get("error_type")
                    != "none"
                ),
                None,
            )
            if unresolved is not None:
                expected_plan_null_selection = (
                    unresolved,
                    "qwen3vl_plan_null_alternative_diagnosis_v1",
                    "first_unresolved_environment_region",
                )
        if expected_plan_null_selection is None:
            raise ValueError(f"{source} plan-null selection cannot be rebuilt")
    for index, (raw_call, normalized_call) in enumerate(
        zip(raw_round_two_calls, audit["backend_calls"], strict=True)
    ):
        if not isinstance(raw_call, Mapping) or not isinstance(
            normalized_call, Mapping
        ):
            raise ValueError(f"{source} raw round-two call {index} is malformed")
        require_exact_keys(
            normalized_call,
            P1_NORMALIZED_ROUND_TWO_CALL_KEYS,
            source=f"{source}.backend_calls[{index}]",
        )
        context = raw_call.get("terminal_review")
        if (
            not isinstance(context, Mapping)
            or set(context) != set(P1_TERMINAL_REVIEW_CONTEXT_KEYS)
        ):
            raise ValueError(f"{source} raw round-two call lacks exact review context")
        region = (
            context.get("review_region_id")
        )
        if "raw_response" not in raw_call:
            raise ValueError(f"{source} raw round-two call {index} lacks raw_response")
        expected = {
            "round_index": context.get("round_index"),
            "stage": raw_call.get("stage"),
            "attempt_kind": raw_call.get("attempt_kind"),
            "prompt_sha256": raw_call.get("prompt_sha256"),
            "raw_response": raw_call.get("raw_response"),
            "region_id": region,
            "parsed_output": raw_call.get("parsed_output"),
            "visual_input_scope": raw_call.get("visual_input_scope"),
            "network_retry_attempt_count": raw_call.get(
                "network_retry_attempt_count"
            ),
        }
        if any(normalized_call.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{source} normalized round-two call {index} differs")
        expected_contract = region_contracts.get(normalized_call["region_id"])
        if (
            expected_contract is None
            or region != normalized_call["region_id"]
            or context.get("proposal_version") != expected_contract[1]
            or not _protocol_visual_scope_valid(
                scan=expected_contract[0] == "scan",
                attempt_kind=normalized_call["attempt_kind"],
                visual_input_scope=normalized_call["visual_input_scope"],
            )
        ):
            raise ValueError(
                f"{source} round-two proposal_version differs from region identity"
            )
        if normalized_call["attempt_kind"] == "initial":
            round_two_identities.add(
                (normalized_call["region_id"], normalized_call["prompt_sha256"])
            )
        if audit.get("trigger") == "plan_null_after_allowed_repair":
            assert expected_plan_null_selection is not None
            expected_region, expected_prompt, expected_branch = (
                expected_plan_null_selection
            )
            if (
                region != expected_region
                or context.get("preferred_region_id") != expected_region
                or context.get("prompt_version") != expected_prompt
                or context.get("selection_branch") != expected_branch
            ):
                raise ValueError(f"{source} plan-null selection differs from round one")
        elif (
            not isinstance(context, Mapping)
            or context.get("preferred_region_id") is not None
            or context.get("prompt_version") is not None
            or context.get("selection_branch") is not None
        ):
            raise ValueError(f"{source} finish review carries plan-null metadata")
    if not round_two_identities or not (round_two_identities - prior_identities):
        raise ValueError(f"{source} does not independently prove a new prompt/region")


def normalize_p1_harness_audit(
    payload: Mapping[str, Any],
    *,
    source: str,
    samples: Sequence[str],
    experiment_arm: str,
    harness_protocol_id: str,
    arm_contract: Mapping[str, Any],
    summary_samples: Mapping[str, Any],
    episode_p1_samples: Mapping[str, Any],
    trajectories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    top_level_keys = (
        "schema_id",
        "factor_id",
        "harness_protocol_id",
        "experiment_arm_id",
        "arm",
        "requested_sample_ids",
        "per_sample",
        "totals",
        "gt_boundary_violation_count",
        "safety_boundary_relaxation_count",
        "tool_failure_count",
        "written_before_rollout_complete",
        "gt_loaded",
    )
    require_exact_keys(payload, top_level_keys, source=source)
    expected_identity = {
        "schema_id": P1_AUDIT_SCHEMA_VERSION,
        "factor_id": P1_FACTOR_ID,
        "harness_protocol_id": harness_protocol_id,
        "experiment_arm_id": experiment_arm,
        "arm": arm_contract["arm"],
        "requested_sample_ids": list(samples),
        "written_before_rollout_complete": True,
        "gt_loaded": False,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"{source}.{key} differs from the frozen P1 contract")
    per_sample_payload = payload.get("per_sample")
    if not isinstance(per_sample_payload, Mapping) or list(per_sample_payload) != list(
        samples
    ):
        raise ValueError(f"{source}.per_sample identity/order differs")

    configured = arm_contract["configured_budgets"]
    totals = {key: 0 for key in P1_BUDGET_COUNTER_FIELDS}
    normalized_samples: dict[str, Any] = {}
    final_region_protocol_valid_count = 0
    round_two_invoked_count = 0
    for sample in samples:
        sample_source = f"{source}.per_sample.{sample}"
        raw_sample = per_sample_payload.get(sample)
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"{sample_source} must be an object")
        require_exact_keys(
            raw_sample,
            (
                "budget_accounting",
                "gt_boundary_audit",
                "safety_boundary_audit",
                "terminal_review",
                "terminal_reviews",
                "final_region_protocol_valid",
            ),
            source=sample_source,
        )
        counters = normalize_budget_accounting(
            raw_sample.get("budget_accounting"),
            source=f"{sample_source}.budget_accounting",
            configured=configured,
        )
        if counters["verifier_round_count"] < 1:
            raise ValueError(f"{sample_source} has no verifier round")
        summary_sample = summary_samples.get(sample)
        if not isinstance(summary_sample, Mapping):
            raise ValueError(f"summary is missing P1 sample {sample}")
        episode_sample = episode_p1_samples.get(sample)
        if not isinstance(episode_sample, Mapping) or dict(raw_sample) != dict(
            episode_sample
        ):
            raise ValueError(f"{sample_source} differs from episode summary evidence")
        expected_tool_attempts = (
            require_nonnegative_int(
                summary_sample, "tool_action_count", source=f"summary.{sample}"
            )
            + require_nonnegative_int(
                summary_sample, "tool_failure_count", source=f"summary.{sample}"
            )
        )
        if counters["tool_action_attempt_count"] != expected_tool_attempts:
            raise ValueError(f"{sample_source} tool-attempt accounting differs")
        if counters["tool_failure_count"] != summary_sample["tool_failure_count"]:
            raise ValueError(f"{sample_source} tool-failure accounting differs")
        if counters["environment_step_count"] != summary_sample["candidate_count"]:
            raise ValueError(f"{sample_source} environment-step accounting differs")

        gt_audit = normalize_gt_boundary_audit(
            raw_sample.get("gt_boundary_audit"),
            source=f"{sample_source}.gt_boundary_audit",
        )
        safety_audit = normalize_safety_boundary_audit(
            raw_sample.get("safety_boundary_audit"),
            source=f"{sample_source}.safety_boundary_audit",
        )
        raw_terminal_reviews = raw_sample.get("terminal_reviews")
        if not isinstance(raw_terminal_reviews, list):
            raise ValueError(f"{sample_source}.terminal_reviews must be a list")
        if arm_contract["arm"] == "h1" and not raw_terminal_reviews:
            raise ValueError(f"{sample_source} lacks H1 terminal-review history")
        if arm_contract["arm"] == "h0" and raw_terminal_reviews:
            raise ValueError(f"{sample_source} H0 history must be empty")
        trajectory_reviews = terminal_review_records_from_trajectory(
            trajectories[sample], source=f"trajectory.{sample}"
        )
        trajectory_audits = [item["audit"] for item in trajectory_reviews]
        if raw_terminal_reviews != trajectory_audits:
            raise ValueError(
                f"{sample_source}.terminal_reviews differs from raw trajectory evidence"
            )
        terminal_reviews: list[dict[str, Any]] = []
        invoked_state_ids: set[str] = set()
        for index, raw_terminal in enumerate(raw_terminal_reviews):
            normalized = normalize_terminal_review_audit(
                raw_terminal,
                source=f"{sample_source}.terminal_reviews[{index}]",
                experiment_arm=experiment_arm,
                arm_contract=arm_contract,
                budget_accounting=counters,
            )
            if normalized is None:
                raise ValueError(f"{sample_source}.terminal_reviews[{index}] is null")
            rebuilt_trigger = rebuild_terminal_trigger_from_round_one(
                trajectory_reviews[index]["round_one_evidence"]
            )
            if normalized.get("trigger") != rebuilt_trigger:
                raise ValueError(
                    f"{sample_source}.terminal_reviews[{index}] trigger differs "
                    "from raw round-one evidence"
                )
            if normalized["round_two_invoked"]:
                rebuilt_outcome = rebuild_round_two_outcome(
                    trajectory_reviews[index]["round_two_evidence"]
                )
                if normalized.get("observed_outcome") != rebuilt_outcome:
                    raise ValueError(
                        f"{sample_source}.terminal_reviews[{index}] outcome differs "
                        "from raw round-two evidence"
                    )
            state_id = str(normalized["accepted_state_id"])
            if normalized["round_two_invoked"]:
                if state_id in invoked_state_ids:
                    raise ValueError(
                        f"{sample_source} invokes round two twice for one accepted state"
                    )
                invoked_state_ids.add(state_id)
            terminal_reviews.append(normalized)
            round_two_invoked_count += int(normalized["round_two_invoked"])
            raw_round_two_evidence = trajectory_reviews[index]["round_two_evidence"]
            if normalized["round_two_invoked"] != isinstance(
                raw_round_two_evidence, Mapping
            ):
                raise ValueError(
                    f"{sample_source}.terminal_reviews[{index}] round-two evidence differs"
                )
            validate_terminal_review_independence(
                trajectory_reviews[index],
                normalized,
                source=f"{sample_source}.terminal_reviews[{index}]",
            )
        terminal_audit = raw_sample.get("terminal_review")
        expected_last = terminal_reviews[-1] if terminal_reviews else None
        if terminal_audit != expected_last:
            raise ValueError(f"{sample_source}.terminal_review is not the history tail")
        protocol_valid = raw_sample.get("final_region_protocol_valid")
        if not isinstance(protocol_valid, bool):
            raise ValueError(f"{sample_source}.final_region_protocol_valid is malformed")
        rebuilt_protocol_valid = rebuild_final_region_protocol_valid(
            trajectories[sample],
            arm=arm_contract["arm"],
            review_records=trajectory_reviews,
        )
        if protocol_valid is not rebuilt_protocol_valid:
            raise ValueError(
                f"{sample_source}.final_region_protocol_valid differs from "
                "raw staged evidence"
            )
        final_region_protocol_valid_count += int(protocol_valid)
        for key, value in counters.items():
            totals[key] += value
        normalized_samples[sample] = {
            "budget_accounting": counters,
            "gt_boundary_audit": gt_audit,
            "safety_boundary_audit": safety_audit,
            "terminal_review": terminal_audit,
            "terminal_reviews": terminal_reviews,
            "final_region_protocol_valid": protocol_valid,
        }

    raw_totals = payload.get("totals")
    if not isinstance(raw_totals, Mapping):
        raise ValueError(f"{source}.totals must be an object")
    require_exact_keys(raw_totals, P1_BUDGET_COUNTER_FIELDS, source=f"{source}.totals")
    normalized_raw_totals = {
        key: require_nonnegative_int(raw_totals, key, source=f"{source}.totals")
        for key in P1_BUDGET_COUNTER_FIELDS
    }
    if normalized_raw_totals != totals:
        raise ValueError(f"{source}.totals differ from per-sample budget sums")
    gt_violations = require_nonnegative_int(
        payload, "gt_boundary_violation_count", source=source
    )
    safety_relaxations = require_nonnegative_int(
        payload, "safety_boundary_relaxation_count", source=source
    )
    tool_failures = require_nonnegative_int(payload, "tool_failure_count", source=source)
    if gt_violations or safety_relaxations:
        raise ValueError(f"{source} fails the GT or safety-boundary hard gate")
    if tool_failures != totals["tool_failure_count"] or tool_failures:
        raise ValueError(f"{source} records a non-zero P1 tool-failure hard gate")
    return {
        "schema_version": "qwen3vl_e2e_summary_p1_contract_v1",
        "source_audit_schema_id": P1_AUDIT_SCHEMA_VERSION,
        "factor_id": P1_FACTOR_ID,
        "arm": arm_contract["arm"],
        "harness_protocol_id": harness_protocol_id,
        "experiment_arm": experiment_arm,
        "terminal_review": arm_contract["terminal_review"],
        "model_io_protocol": P1_MODEL_IO_PROTOCOL,
        "configured_budgets": dict(configured),
        "budget_accounting_totals": totals,
        "frozen_safety_boundaries": list(P1_SAFETY_BOUNDARIES),
        "gt_loaded_before_rollout_complete": False,
        "final_region_protocol_valid_count": final_region_protocol_valid_count,
        "round_two_invoked_count": round_two_invoked_count,
        "per_sample": normalized_samples,
        "hard_gates": {
            "requested_scored_mismatch_count": 0,
            "pre_rollout_gt_violation_count": gt_violations,
            "safety_boundary_relaxation_count": safety_relaxations,
            "tool_failure_count": tool_failures,
            "all_passed": True,
        },
    }


def build_artifact_bindings(
    run_dir: Path,
    samples: Sequence[str],
    *,
    p1_harness_audit: Path | None,
) -> dict[str, Any]:
    per_sample: dict[str, Any] = {}
    for sample in samples:
        tool_root = run_dir / "tool_runs" / sample
        call_dirs = (
            sorted(
                (
                    path
                    for path in tool_root.iterdir()
                    if path.is_dir()
                    and (path.name.startswith("point_") or path.name.startswith("box_"))
                ),
                key=lambda path: path.name,
            )
            if tool_root.is_dir()
            else []
        )
        reports = [artifact_binding(path / "report.json") for path in call_dirs]
        per_sample[sample] = {
            "trajectory": artifact_binding(
                run_dir / "trajectories" / sample / "trajectory.json"
            ),
            "episode_summary": artifact_binding(
                run_dir / "trajectories" / sample / "episode_summary.json"
            ),
            "tool_reports": reports,
        }
    result: dict[str, Any] = {
        "algorithm": "sha256",
        "run_dir": str(run_dir.resolve()),
        "rollout_marker": artifact_binding(run_dir / "rollout_complete.json"),
        "source_metrics": artifact_binding(run_dir / "per_sample_metrics.json"),
        "per_sample": per_sample,
    }
    if p1_harness_audit is not None:
        result["p1_harness_audit"] = artifact_binding(p1_harness_audit)
    return result


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    expected_samples = [str(sample) for sample in args.expected_samples]
    expected_evaluation_id, canonical_samples = EVALUATION_PROFILES[
        args.evaluation_profile
    ]
    if args.evaluation_id != expected_evaluation_id:
        raise ValueError(
            "evaluation ID does not match evaluation profile: "
            f"{args.evaluation_id!r} != {expected_evaluation_id!r}"
        )
    if expected_samples != list(canonical_samples):
        raise ValueError(
            "expected samples do not match evaluation profile: "
            f"{expected_samples!r} != {list(canonical_samples)!r}"
        )
    if len(expected_samples) != len(set(expected_samples)):
        raise ValueError("expected sample IDs must be unique")
    experiment_arm = str(args.experiment_arm)
    harness_protocol_id = str(getattr(args, "harness_protocol_id", ""))
    p1_arm_contract = P1_ARM_CONTRACTS.get(experiment_arm)
    p1_protocol_to_arm = {
        contract["harness_protocol_id"]: arm
        for arm, contract in P1_ARM_CONTRACTS.items()
    }
    if p1_arm_contract is not None:
        if harness_protocol_id != p1_arm_contract["harness_protocol_id"]:
            raise ValueError("P1 experiment arm and harness protocol ID differ")
        if str(args.protocol_version) != P1_MODEL_IO_PROTOCOL:
            raise ValueError("P1 model-I/O protocol must be v12.1")
        if args.model_state != "base":
            raise ValueError("P1 H0/H1 harness evaluation must use the base model")
        if args.evaluation_profile not in {"p1_smoke1", "e2e3"}:
            raise ValueError("P1 H0/H1 evaluation profile must be p1_smoke1 or e2e3")
    elif harness_protocol_id in p1_protocol_to_arm:
        raise ValueError("P1 harness protocol ID is paired with the wrong experiment arm")
    elif args.evaluation_profile == "p1_smoke1":
        raise ValueError("p1_smoke1 requires a frozen P1 H0/H1 experiment arm")
    if (run_dir / ".run_incomplete").exists():
        raise ValueError(f"run is still marked incomplete: {run_dir}")

    rollout = read_object(run_dir / "rollout_complete.json")
    if rollout.get("execution_status") != "completed" or rollout.get("gt_loaded") is not False:
        raise ValueError("rollout marker must say completed with gt_loaded=false")
    rollout_samples = rollout.get("samples")
    expected_rollout_samples = sorted(expected_samples)
    if (
        not isinstance(rollout_samples, list)
        or rollout_samples != expected_rollout_samples
    ):
        raise ValueError(
            "rollout sample identity/canonical order differs: "
            f"{rollout_samples!r} != {expected_rollout_samples!r}"
        )

    metrics = read_object(run_dir / "per_sample_metrics.json")
    sample_metrics = metrics.get("samples")
    if not isinstance(sample_metrics, dict) or list(sample_metrics) != expected_samples:
        observed = list(sample_metrics) if isinstance(sample_metrics, dict) else sample_metrics
        raise ValueError(
            f"scored sample order differs: {observed!r} != {expected_samples!r}"
        )

    initial_counts = {key: 0 for key in COUNT_KEYS}
    selected_counts = {key: 0 for key in COUNT_KEYS}
    action_totals = Counter()
    outcome_counts: Counter[str] = Counter()
    per_sample: dict[str, Any] = {}
    initial_macro_counts: list[dict[str, int]] = []
    selected_macro_counts: list[dict[str, int]] = []
    candidate_oracle_totals = empty_candidate_oracle_totals()
    degraded_sample_count = 0
    finish_count = 0
    episode_p1_samples: dict[str, Any] = {}
    trajectories: dict[str, Mapping[str, Any]] = {}

    for sample in expected_samples:
        sample_payload = sample_metrics[sample]
        if not isinstance(sample_payload, dict):
            raise ValueError(f"sample metrics must be an object: {sample}")
        initial = sample_payload.get("initial")
        selected = sample_payload.get("verifier_selected")
        if not isinstance(initial, dict) or not isinstance(selected, dict):
            raise ValueError(f"sample {sample} lacks initial/verifier_selected metrics")
        initial_sample_counts = counts_from_metrics(initial, source=f"{sample}.initial")
        selected_sample_counts = counts_from_metrics(selected, source=f"{sample}.selected")
        if not close_metrics(
            initial,
            metrics_from_counts(initial_sample_counts),
        ):
            raise ValueError(f"sample {sample} initial rates disagree with its pixel counts")
        if not close_metrics(
            selected,
            metrics_from_counts(selected_sample_counts),
        ):
            raise ValueError(f"sample {sample} selected rates disagree with its pixel counts")
        add_counts(initial_counts, initial_sample_counts)
        add_counts(selected_counts, selected_sample_counts)
        initial_macro_counts.append(initial_sample_counts)
        selected_macro_counts.append(selected_sample_counts)
        degraded = is_degraded(initial_sample_counts, selected_sample_counts)
        degraded_sample_count += int(degraded)

        episode = read_object(run_dir / "trajectories" / sample / "episode_summary.json")
        if episode.get("sample") != sample or episode.get("execution_status") != "completed":
            raise ValueError(f"episode summary is not a completed record for {sample}")
        if p1_arm_contract is not None:
            expected_episode_identity = {
                "harness_protocol_id": harness_protocol_id,
                "experiment_arm_id": experiment_arm,
                "terminal_review_factor_id": P1_FACTOR_ID,
                "terminal_review_arm": p1_arm_contract["arm"],
            }
            for key, expected in expected_episode_identity.items():
                if episode.get(key) != expected:
                    raise ValueError(f"{sample}.episode.{key} differs from P1 identity")
            episode_p1_samples[sample] = {
                key: episode.get(key)
                for key in (
                    "terminal_review",
                    "terminal_reviews",
                    "budget_accounting",
                    "gt_boundary_audit",
                    "safety_boundary_audit",
                    "final_region_protocol_valid",
                )
            }
        candidate_count = require_nonnegative_int(
            episode, "candidate_count", source=f"{sample}.episode"
        )
        accepted = require_nonnegative_int(
            episode, "accepted_candidate_count", source=f"{sample}.episode"
        )
        rejected = require_nonnegative_int(
            episode, "rejected_candidate_count", source=f"{sample}.episode"
        )
        if candidate_count != accepted + rejected:
            raise ValueError(
                f"{sample} candidate total differs: {candidate_count} != {accepted}+{rejected}"
            )
        tool_actions = require_nonnegative_int(
            episode, "tool_action_count", source=f"{sample}.episode"
        )
        attempts = require_nonnegative_int(
            episode, "action_attempt_count", source=f"{sample}.episode"
        )
        errors = require_nonnegative_int(
            episode, "action_attempt_error_count", source=f"{sample}.episode"
        )
        fallback_actions = require_nonnegative_int(
            episode, "empty_mask_scan_fallback_action_count", source=f"{sample}.episode"
        )
        if attempts < candidate_count or errors > attempts:
            raise ValueError(f"{sample} has inconsistent action-attempt counts")
        outcome = episode.get("episode_outcome")
        stop_reason = episode.get("stop_reason")
        if not isinstance(outcome, str) or not isinstance(stop_reason, str):
            raise ValueError(f"{sample} lacks episode outcome/stop reason")
        outcome_counts[outcome] += 1
        finish_count += int(stop_reason in FINISH_REASONS)
        action_totals.update(
            candidates=candidate_count,
            accepted=accepted,
            rejected=rejected,
            tool_actions=tool_actions,
            action_attempts=attempts,
            action_attempt_errors=errors,
            fallback_actions=fallback_actions,
        )
        trajectory = read_object(
            run_dir / "trajectories" / sample / "trajectory.json"
        )
        trajectories[sample] = trajectory
        candidate_oracle = candidate_oracle_summary(
            trajectory,
            candidate_count=candidate_count,
            source=f"{sample}.trajectory",
        )
        add_candidate_oracle_totals(candidate_oracle_totals, candidate_oracle)
        tool_root = run_dir / "tool_runs" / sample
        action_call_dirs = (
            sorted(
                (
                    path
                    for path in tool_root.iterdir()
                    if path.is_dir()
                    and (path.name.startswith("point_") or path.name.startswith("box_"))
                ),
                key=lambda path: path.name,
            )
            if tool_root.is_dir()
            else []
        )
        action_successes = 0
        action_failures = 0
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
        action_totals.update(
            successful_tool_actions=action_successes,
            tool_failure_count=action_failures,
        )
        if action_successes != tool_actions:
            raise ValueError(
                f"{sample} tool action evidence differs: reports={action_successes}, "
                f"episode_summary={tool_actions}"
            )
        per_sample[sample] = {
            "initial": {key: initial[key] for key in (*METRIC_KEYS, *COUNT_KEYS)},
            "selected": {key: selected[key] for key in (*METRIC_KEYS, *COUNT_KEYS)},
            "delta_iou": round(float(selected["iou"]) - float(initial["iou"]), 8),
            "delta_f1": round(float(selected["f1"]) - float(initial["f1"]), 8),
            "delta_counts": count_deltas(
                initial_sample_counts, selected_sample_counts
            ),
            "degraded": degraded,
            "episode_outcome": outcome,
            "stop_reason": stop_reason,
            "candidate_count": candidate_count,
            "accepted_candidate_count": accepted,
            "rejected_candidate_count": rejected,
            "tool_action_count": tool_actions,
            "action_attempt_count": attempts,
            "action_attempt_error_count": errors,
            "successful_tool_action_count": action_successes,
            "tool_failure_count": action_failures,
            "candidate_oracle": candidate_oracle,
        }

    initial_aggregate = metrics_from_counts(initial_counts)
    selected_aggregate = metrics_from_counts(selected_counts)
    initial_macro = macro_metrics(initial_macro_counts)
    selected_macro = macro_metrics(selected_macro_counts)
    recorded_aggregate = metrics.get("aggregate")
    if not isinstance(recorded_aggregate, dict) or not close_metrics(
        recorded_aggregate, selected_aggregate
    ):
        raise ValueError("reconstructed selected aggregate differs from runner evidence")

    p1_audit_path: Path | None = None
    p1_contract: dict[str, Any] | None = None
    if p1_arm_contract is not None:
        configured_path = getattr(args, "p1_harness_audit", None)
        p1_audit_path = (
            Path(configured_path).resolve()
            if configured_path is not None
            else (run_dir / "p1_harness_audit.json").resolve()
        )
        expected_audit_path = (run_dir / "p1_harness_audit.json").resolve()
        if p1_audit_path != expected_audit_path:
            raise ValueError(
                "P1 harness audit path must be run_dir/p1_harness_audit.json"
            )
        p1_contract = normalize_p1_harness_audit(
            read_object(p1_audit_path),
            source=str(p1_audit_path),
            samples=expected_samples,
            experiment_arm=experiment_arm,
            harness_protocol_id=harness_protocol_id,
            arm_contract=p1_arm_contract,
            summary_samples=per_sample,
            episode_p1_samples=episode_p1_samples,
            trajectories=trajectories,
        )

    artifact_bindings = build_artifact_bindings(
        run_dir,
        expected_samples,
        p1_harness_audit=p1_audit_path,
    )
    if p1_contract is not None:
        p1_contract["provenance"] = {
            "model_state": str(args.model_state),
            "runtime_commit": str(args.runtime_commit),
            "harness_commit": str(args.harness_commit),
            "job_id": str(args.job_id),
            "artifact_bindings": artifact_bindings,
        }

    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "model_state": args.model_state,
        "job_id": str(args.job_id),
        "experiment_arm": experiment_arm,
        "protocol_version": str(args.protocol_version),
        "harness_protocol_id": harness_protocol_id,
        "runtime_commit": str(args.runtime_commit),
        "harness_commit": str(args.harness_commit),
        "requested_samples": len(expected_samples),
        "scored_samples": len(expected_samples),
        "sample_order": expected_samples,
        "gt_boundary": {
            "rollout_marker_claims_gt_free": rollout.get("gt_loaded") is False,
            "offline_metrics_are_post_rollout": True,
            "interpretation": "the decision path is GT-free; final trajectory contains post-rollout offline GT metrics",
        },
        "rollout_marker": str(run_dir / "rollout_complete.json"),
        "aggregate": {
            "aggregation": "micro_count_sum",
            "initial": initial_aggregate,
            "selected": selected_aggregate,
            "delta_iou": round(
                float(selected_aggregate["iou"]) - float(initial_aggregate["iou"]), 8
            ),
            "delta_f1": round(
                float(selected_aggregate["f1"]) - float(initial_aggregate["f1"]), 8
            ),
            "delta_counts": count_deltas(initial_counts, selected_counts),
        },
        "macro": {
            "aggregation": "sample_arithmetic_mean",
            "initial": initial_macro,
            "selected": selected_macro,
            "delta_iou": round(
                float(selected_macro["iou"]) - float(initial_macro["iou"]), 8
            ),
            "delta_f1": round(
                float(selected_macro["f1"]) - float(initial_macro["f1"]), 8
            ),
        },
        "actions": {
            **dict(sorted(action_totals.items())),
            "finish_count": finish_count,
            "tool_success_rate": (
                actions_successes / (actions_successes + actions_failures)
                if (actions_successes := int(action_totals["successful_tool_actions"]))
                + (actions_failures := int(action_totals["tool_failure_count"]))
                else None
            ),
        },
        "episode_outcome_counts": dict(sorted(outcome_counts.items())),
        "invalid_or_no_action_samples": (
            outcome_counts.get("verifier_invalid", 0)
            + outcome_counts.get("no_executable_action", 0)
        ),
        "degraded_sample_count": degraded_sample_count,
        "candidate_oracle": candidate_oracle_totals,
        "per_sample": per_sample,
        "source_metrics": str(run_dir / "per_sample_metrics.json"),
        "artifact_bindings": artifact_bindings,
    }
    if p1_contract is not None:
        summary["p1_contract"] = p1_contract
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize(args)
    except ValueError as error:
        print(f"qwen3vl_e2e_summary: {error}", file=sys.stderr)
        return 3
    atomic_write(args.output, summary)
    aggregate = summary["aggregate"]
    actions = summary["actions"]
    print(
        "status=completed;"
        f"scored={summary['scored_samples']}/{summary['requested_samples']};"
        f"initial_iou={aggregate['initial']['iou']};"
        f"selected_iou={aggregate['selected']['iou']};"
        f"initial_f1={aggregate['initial']['f1']};"
        f"selected_f1={aggregate['selected']['f1']};"
        f"accepted={actions['accepted']};rejected={actions['rejected']};"
        f"finish={actions['finish_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
