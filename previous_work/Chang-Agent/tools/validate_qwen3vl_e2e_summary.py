#!/usr/bin/env python3
"""Independently validate a frozen P1 Qwen3-VL E2E summary.

The validator intentionally does not import the summarizer.  It re-opens the
bound rollout marker, raw metrics, trajectories, episode summaries, tool
reports, and P1 harness audit; verifies every SHA-256 binding; reconstructs the
reported metrics and action totals; and applies the preregistered P1 hard
gates.  A summary is accepted only when the caller also supplies the exact
expected evaluation and arm identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATION_PROFILES = {
    "e2e5": (
        "DEV-EXTENDED-5-V1",
        ("test_20_15", "test_78_13", "test_85_16", "test_1_1", "test_50_8"),
    ),
    "e2e3": ("DEV-CORE-3-V1", ("test_20_15", "test_1_1", "test_50_8")),
    "smoke1": ("DEV-SMOKE-1-V1", ("test_20_15",)),
    "p1_smoke1": ("DEV-SMOKE-1-H1-V1", ("test_1_1",)),
}
SUMMARY_SCHEMA_VERSION = "qwen3vl_e2e_summary_v2"
P1_SUMMARY_CONTRACT_VERSION = "qwen3vl_e2e_summary_p1_contract_v1"
P1_AUDIT_SCHEMA_VERSION = "qwen3vl_p1_harness_audit_v1"
P1_TERMINAL_AUDIT_SCHEMA_VERSION = "qwen3vl_terminal_review_audit_v1"
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
COUNT_KEYS = ("tp", "fp", "fn", "tn")
DELTA_COUNT_KEYS = ("tp", "fp", "fn")
METRIC_KEYS = ("iou", "precision", "recall", "f1")
ACTION_KEYS = (
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
FINISH_REASONS = {
    "initial_verifier_authorized_finish",
    "verifier_authorized_finish",
    "confirmed_finish",
}
CANDIDATE_CLASSIFICATIONS = ("beneficial", "harmful", "indeterminate")
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
COMPARISON_EPSILON = 1e-12
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
P1_OUTCOMES = (
    "finish",
    "unique_safe_plan",
    "plan_null",
    "unresolved",
    "unsafe_or_nonunique_plan",
    "schema_invalid_after_allowed_repair",
    "budget_exhausted",
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
P1_TERMINAL_AUDIT_KEYS = (
    "schema_version", "factor_id", "policy", "accepted_state_id", "gt_free",
    "trigger_input_fields", "forbidden_trigger_inputs_present", "eligible",
    "trigger", "round_two_invoked", "round_two_count", "round_index",
    "round_two_kind", "observed_outcome", "effective_outcome", "transition",
    "fresh_backend_context", "inherits_hidden_state", "copied_round_one_raw_response",
    "nonduplicate_prompt_or_region", "round_three_allowed", "contract_violation",
    "safe_plan_returned_to_existing_flow", "tool_executed_by_scheduler",
    "counter_scope", "counters_before", "counters_after", "backend_calls",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-profile", choices=tuple(EVALUATION_PROFILES), required=True
    )
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--expected-samples", nargs="+", required=True)
    parser.add_argument("--experiment-arm", choices=tuple(P1_ARM_CONTRACTS), required=True)
    parser.add_argument("--harness-protocol-id", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--job-id", required=True)
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


def require_exact_keys(value: Mapping[str, Any], keys: Sequence[str], *, name: str) -> None:
    if set(value) != set(keys):
        raise ValueError(f"{name} keys differ from the frozen contract")


def nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, found {value!r}")
    return value


def finite_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite, found {value!r}")
    return float(value)


def nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
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


def binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"bound artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def rates_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def metric_counts(value: Any, *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    counts = {
        key: nonnegative_int(value.get(key), name=f"{name}.{key}") for key in COUNT_KEYS
    }
    expected = rates_from_counts(counts)
    for key in METRIC_KEYS:
        observed = finite_number(value.get(key), name=f"{name}.{key}")
        if not 0.0 <= observed <= 1.0 or not math.isclose(
            observed, expected[key], rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"{name}.{key} disagrees with pixel counts")
    return counts


def close_number(value: Any, expected: float, *, name: str) -> None:
    observed = finite_number(value, name=name)
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{name} disagrees with reconstructed evidence")


def count_deltas(initial: Mapping[str, int], selected: Mapping[str, int]) -> dict[str, int]:
    return {key: selected[key] - initial[key] for key in DELTA_COUNT_KEYS}


def degraded(initial: Mapping[str, int], selected: Mapping[str, int]) -> bool:
    initial_rates = rates_from_counts(initial)
    selected_rates = rates_from_counts(selected)
    delta_iou = selected_rates["iou"] - initial_rates["iou"]
    delta_f1 = selected_rates["f1"] - initial_rates["f1"]
    if delta_iou < -COMPARISON_EPSILON or delta_f1 < -COMPARISON_EPSILON:
        return True
    tied = (
        abs(delta_iou) <= COMPARISON_EPSILON
        and abs(delta_f1) <= COMPARISON_EPSILON
    )
    return tied and (
        selected["tp"] < initial["tp"]
        or selected["fp"] > initial["fp"]
        or selected["fn"] > initial["fn"]
    )


def macro_metrics(values: Sequence[Mapping[str, int]]) -> dict[str, float]:
    if not values:
        raise ValueError("macro metrics require at least one sample")
    rates = [rates_from_counts(item) for item in values]
    return {
        key: round(sum(item[key] for item in rates) / len(rates), 8)
        for key in METRIC_KEYS
    }


def add_counts(target: dict[str, int], value: Mapping[str, int]) -> None:
    for key in COUNT_KEYS:
        target[key] += value[key]


def optional_oracle_counts(step: Any) -> dict[str, int] | None:
    if not isinstance(step, Mapping) or not isinstance(step.get("offline_metrics"), Mapping):
        return None
    metrics = step["offline_metrics"]
    result: dict[str, int] = {}
    for key in DELTA_COUNT_KEYS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        result[key] = value
    return result


def candidate_decision(step: Any) -> str:
    if not isinstance(step, Mapping):
        return "unavailable"
    decisions: list[bool] = []
    direct = step.get("candidate_accepted")
    if isinstance(direct, bool):
        decisions.append(direct)
    execution = step.get("execution")
    if isinstance(execution, Mapping) and isinstance(
        execution.get("candidate_accepted"), bool
    ):
        decisions.append(execution["candidate_accepted"])
    if not decisions or any(value != decisions[0] for value in decisions[1:]):
        return "unavailable"
    return "accepted" if decisions[0] else "rejected"


def hard_gate_before_qwen(step: Any) -> bool:
    if not isinstance(step, Mapping) or not isinstance(step.get("execution"), Mapping):
        return False
    execution = step["execution"]
    if execution.get("verifier_skipped_by_hard_gate") is True:
        return True
    evidence = execution.get("verifier_evidence")
    return bool(
        isinstance(evidence, Mapping)
        and evidence.get("type") == "runtime_hard_gate_precheck"
        and evidence.get("verifier_skipped") is True
    )


def decision_path(step: Any, *, hard_gate: bool) -> str:
    if not isinstance(step, Mapping) or not isinstance(step.get("execution"), Mapping):
        return "unavailable"
    execution = step["execution"]
    evidence = execution.get("verifier_evidence")
    qwen_evidence = bool(
        execution.get("verifier_skipped_by_hard_gate") is False
        or isinstance(evidence, Mapping)
        and evidence.get("type") == "staged_qwen_verifier"
    )
    if hard_gate and qwen_evidence:
        return "unavailable"
    if hard_gate:
        return "pre_qwen_hard_gate"
    return "qwen" if qwen_evidence else "unavailable"


def leaf_class(
    classification: str, oracle_status: str, decision: str, path: str
) -> str:
    if oracle_status != "available" or decision == "unavailable":
        return "unavailable"
    if decision == "accepted":
        if path != "qwen":
            return "unavailable"
        return {
            "beneficial": "true_accept",
            "harmful": "false_accept",
            "indeterminate": "indeterminate_accepted",
        }.get(classification, "unavailable")
    if decision == "rejected" and path == "qwen":
        return {
            "beneficial": "qwen_false_reject",
            "harmful": "true_reject",
            "indeterminate": "indeterminate_qwen_rejected",
        }.get(classification, "unavailable")
    if decision == "rejected" and path == "pre_qwen_hard_gate":
        return {
            "beneficial": "hard_gate_false_reject",
            "harmful": "hard_gate_correct_reject",
            "indeterminate": "indeterminate_hard_gate_rejected",
        }.get(classification, "unavailable")
    return "unavailable"


def classify_candidate(reference: Mapping[str, int], candidate: Mapping[str, int]) -> str:
    initial = rates_from_counts({**reference, "tn": 0})
    selected = rates_from_counts({**candidate, "tn": 0})
    delta_iou = selected["iou"] - initial["iou"]
    delta_f1 = selected["f1"] - initial["f1"]
    if delta_iou > COMPARISON_EPSILON and delta_f1 > COMPARISON_EPSILON:
        return "beneficial"
    tied = (
        abs(delta_iou) <= COMPARISON_EPSILON
        and abs(delta_f1) <= COMPARISON_EPSILON
    )
    worse = (
        candidate["tp"] < reference["tp"]
        or candidate["fp"] > reference["fp"]
        or candidate["fn"] > reference["fn"]
    )
    if delta_iou < -COMPARISON_EPSILON or delta_f1 < -COMPARISON_EPSILON or tied and worse:
        return "harmful"
    return "indeterminate"


def empty_oracle_totals(*, include_candidates: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_count": 0,
        "classification_counts": {key: 0 for key in CANDIDATE_CLASSIFICATIONS},
        "decision_counts": {key: 0 for key in CANDIDATE_DECISIONS},
        "oracle_status_counts": {key: 0 for key in CANDIDATE_ORACLE_STATUSES},
        "decision_path_counts": {key: 0 for key in CANDIDATE_DECISION_PATHS},
        "leaf_class_counts": {key: 0 for key in CANDIDATE_LEAF_CLASSES},
        "false_accept_count": 0,
        "qwen_false_reject_count": 0,
        "hard_gate_before_qwen_count": 0,
        "hard_gate_false_reject_count": 0,
        "all_false_reject_count": 0,
        "false_reject_count": 0,
        "false_reject_count_semantics": FALSE_REJECT_COUNT_SEMANTICS,
    }
    if include_candidates:
        result["candidates"] = []
    return result


def candidate_oracle(trajectory: Mapping[str, Any], candidate_count: int, *, name: str) -> dict[str, Any]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or len(steps) != candidate_count + 1:
        raise ValueError(f"{name} trajectory candidate accounting differs")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("step_index") != index:
            raise ValueError(f"{name}.steps[{index}] has the wrong step index")
    result = empty_oracle_totals(include_candidates=True)
    reference = optional_oracle_counts(steps[0])
    reference_index: int | None = 0
    for index in range(1, candidate_count + 1):
        step = steps[index]
        candidate = optional_oracle_counts(step)
        decision = candidate_decision(step)
        hard_gate = hard_gate_before_qwen(step)
        path = decision_path(step, hard_gate=hard_gate)
        unavailable_reason: str | None = None
        delta: dict[str, int] | None = None
        delta_iou: float | None = None
        delta_f1: float | None = None
        if reference is None:
            classification = "indeterminate"
            oracle_status = "unavailable"
            unavailable_reason = "reference_offline_counts_unavailable"
        elif candidate is None:
            classification = "indeterminate"
            oracle_status = "unavailable"
            unavailable_reason = "candidate_offline_counts_unavailable"
        else:
            classification = classify_candidate(reference, candidate)
            oracle_status = "available"
            delta = count_deltas(reference, candidate)
            initial_rates = rates_from_counts({**reference, "tn": 0})
            selected_rates = rates_from_counts({**candidate, "tn": 0})
            delta_iou = round(selected_rates["iou"] - initial_rates["iou"], 8)
            delta_f1 = round(selected_rates["f1"] - initial_rates["f1"], 8)
        leaf = leaf_class(classification, oracle_status, decision, path)
        false_accept = leaf == "false_accept"
        qwen_false_reject = leaf == "qwen_false_reject"
        hard_gate_false_reject = leaf == "hard_gate_false_reject"
        all_false_reject = qwen_false_reject or hard_gate_false_reject
        result["candidate_count"] += 1
        result["classification_counts"][classification] += 1
        result["decision_counts"][decision] += 1
        result["oracle_status_counts"][oracle_status] += 1
        result["decision_path_counts"][path] += 1
        result["leaf_class_counts"][leaf] += 1
        result["false_accept_count"] += int(false_accept)
        result["qwen_false_reject_count"] += int(qwen_false_reject)
        result["hard_gate_before_qwen_count"] += int(hard_gate)
        result["hard_gate_false_reject_count"] += int(hard_gate_false_reject)
        result["all_false_reject_count"] += int(all_false_reject)
        result["false_reject_count"] += int(all_false_reject)
        result["candidates"].append(
            {
                "candidate_index": index,
                "step_index": index,
                "reference_step_index": reference_index,
                "decision": decision,
                "decision_path": path,
                "classification": classification,
                "oracle_status": oracle_status,
                "leaf_class": leaf,
                "reference_counts": dict(reference) if reference is not None else None,
                "candidate_counts": dict(candidate) if candidate is not None else None,
                "delta_counts": delta,
                "delta_iou": delta_iou,
                "delta_f1": delta_f1,
                "hard_gate_before_qwen": hard_gate,
                "false_accept": false_accept,
                "qwen_false_reject": qwen_false_reject,
                "all_false_reject": all_false_reject,
                "false_reject": all_false_reject,
                "hard_gate_false_reject": hard_gate_false_reject,
                "unavailable_reason": unavailable_reason,
            }
        )
        if decision == "accepted":
            reference = candidate
            reference_index = index
        elif decision == "unavailable":
            reference = None
            reference_index = None
    return result


def add_oracle(target: dict[str, Any], sample: Mapping[str, Any]) -> None:
    target["candidate_count"] += sample["candidate_count"]
    for group, keys in (
        ("classification_counts", CANDIDATE_CLASSIFICATIONS),
        ("decision_counts", CANDIDATE_DECISIONS),
        ("oracle_status_counts", CANDIDATE_ORACLE_STATUSES),
        ("decision_path_counts", CANDIDATE_DECISION_PATHS),
        ("leaf_class_counts", CANDIDATE_LEAF_CLASSES),
    ):
        for key in keys:
            target[group][key] += sample[group][key]
    for key in (
        "false_accept_count",
        "qwen_false_reject_count",
        "hard_gate_before_qwen_count",
        "hard_gate_false_reject_count",
        "all_false_reject_count",
        "false_reject_count",
    ):
        target[key] += sample[key]


def expected_tool_report_bindings(run_dir: Path, sample: str) -> list[dict[str, str]]:
    root = run_dir / "tool_runs" / sample
    call_dirs = (
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                and (path.name.startswith("point_") or path.name.startswith("box_"))
            ),
            key=lambda path: path.name,
        )
        if root.is_dir()
        else []
    )
    return [binding(path / "report.json") for path in call_dirs]


def validate_artifact_bindings(
    value: Any, *, run_dir: Path, samples: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("artifact_bindings must be an object")
    require_exact_keys(
        value,
        (
            "algorithm",
            "run_dir",
            "rollout_marker",
            "source_metrics",
            "per_sample",
            "p1_harness_audit",
        ),
        name="artifact_bindings",
    )
    expected: dict[str, Any] = {
        "algorithm": "sha256",
        "run_dir": str(run_dir),
        "rollout_marker": binding(run_dir / "rollout_complete.json"),
        "source_metrics": binding(run_dir / "per_sample_metrics.json"),
        "p1_harness_audit": binding(run_dir / "p1_harness_audit.json"),
        "per_sample": {},
    }
    observed_samples = value.get("per_sample")
    if not isinstance(observed_samples, Mapping) or list(observed_samples) != list(samples):
        raise ValueError("artifact_bindings per-sample identity/order differs")
    for sample in samples:
        expected["per_sample"][sample] = {
            "trajectory": binding(run_dir / "trajectories" / sample / "trajectory.json"),
            "episode_summary": binding(
                run_dir / "trajectories" / sample / "episode_summary.json"
            ),
            "tool_reports": expected_tool_report_bindings(run_dir, sample),
        }
    if dict(value) != expected:
        raise ValueError("artifact path/hash binding differs from raw run artifacts")
    return expected


def validate_terminal_review(
    value: Any,
    *,
    name: str,
    arm: str,
    arm_contract: Mapping[str, Any],
    episode_counters: Mapping[str, int],
) -> dict[str, Any] | None:
    if value is None:
        if arm_contract["arm"] == "h1":
            raise ValueError(f"{name} must retain an H1 scheduler audit")
        return None
    if arm_contract["arm"] == "h0":
        raise ValueError(f"{name} must be null for H0")
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object or null")
    require_exact_keys(value, P1_TERMINAL_AUDIT_KEYS, name=name)
    if value.get("schema_version") != P1_TERMINAL_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"{name}.schema_version differs")
    if value.get("factor_id") != P1_FACTOR_ID or value.get("gt_free") is not True:
        raise ValueError(f"{name} lacks the frozen factor/GT-free identity")
    policy = value.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{name}.policy must be an object")
    expected_policy = {
        "factor_id": P1_FACTOR_ID,
        "arm": arm_contract["arm"],
        "harness_protocol_id": arm_contract["harness_protocol_id"],
        "experiment_arm_id": arm,
        "terminal_review": arm_contract["terminal_review"],
        **arm_contract["configured_budgets"],
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"{name}.policy.{key} differs")
    require_exact_keys(policy, tuple(expected_policy), name=f"{name}.policy")
    if value.get("forbidden_trigger_inputs_present") != []:
        raise ValueError(f"{name} contains forbidden trigger inputs")
    if value.get("trigger_input_fields") != list(P1_TRIGGER_INPUT_FIELDS):
        raise ValueError(f"{name}.trigger_input_fields differs")
    state_id = value.get("accepted_state_id")
    if not isinstance(state_id, str) or SHA256_RE.fullmatch(state_id) is None:
        raise ValueError(f"{name}.accepted_state_id is malformed")
    if value.get("counter_scope") != P1_COUNTER_SCOPE:
        raise ValueError(f"{name}.counter_scope differs")
    for key in (
        "inherits_hidden_state",
        "copied_round_one_raw_response",
        "round_three_allowed",
        "tool_executed_by_scheduler",
    ):
        if value.get(key) is not False:
            raise ValueError(f"{name}.{key} must be false")
    invoked = value.get("round_two_invoked")
    if not isinstance(invoked, bool):
        raise ValueError(f"{name}.round_two_invoked is malformed")
    round_count = 1 if invoked else 0
    if (
        isinstance(value.get("round_two_count"), bool)
        or not isinstance(value.get("round_two_count"), int)
        or value.get("round_two_count") != round_count
    ):
        raise ValueError(f"{name}.round_two_count disagrees")
    eligible = value.get("eligible")
    if not isinstance(eligible, bool):
        raise ValueError(f"{name}.eligible is malformed")
    trigger = value.get("trigger")
    observed_outcome = value.get("observed_outcome")
    effective_outcome = value.get("effective_outcome")
    contract_violation = value.get("contract_violation")
    if contract_violation is not None and (
        not isinstance(contract_violation, str) or not contract_violation
    ):
        raise ValueError(f"{name}.contract_violation is malformed")
    passthrough = trigger is None
    if passthrough:
        if (
            eligible or invoked or value.get("round_two_kind") is not None
            or observed_outcome is not None or effective_outcome is not None
            or value.get("transition") is not None or value.get("round_index") is not None
            or value.get("fresh_backend_context") is not None
            or value.get("nonduplicate_prompt_or_region") is not None
            or contract_violation is not None
            or value.get("safe_plan_returned_to_existing_flow") is not False
        ):
            raise ValueError(f"{name} has malformed passthrough semantics")
    elif trigger not in P1_TRIGGERS:
        raise ValueError(f"{name}.trigger is not registered")
    else:
        expected_kind = {
            "provisional_finish": "finish_confirmation",
            "plan_null_after_allowed_repair": "plan_null_alternative",
        }[trigger]
        if value.get("round_two_kind") != expected_kind:
            raise ValueError(f"{name}.round_two_kind differs")
    if invoked:
        if passthrough or not eligible:
            raise ValueError(f"{name} invoked round two while ineligible")
        if (
            value.get("round_index") != 2
            or value.get("fresh_backend_context") is not True
            or value.get("nonduplicate_prompt_or_region") is not True
        ):
            raise ValueError(f"{name} does not prove an independent second round")
        if observed_outcome not in P1_OUTCOMES or effective_outcome not in P1_OUTCOMES:
            raise ValueError(f"{name} has an unregistered outcome")
        if contract_violation not in {None, "environment_step_budget_exhausted_after_review"}:
            raise ValueError(f"{name} has an unregistered invoked violation")
    elif not passthrough and value.get("round_index") is not None:
        raise ValueError(f"{name}.round_index must be null")
    elif not passthrough:
        if (
            not eligible or observed_outcome is not None
            or effective_outcome != "budget_exhausted"
            or contract_violation not in P1_FAIL_CLOSED_REASONS
            or value.get("fresh_backend_context") is not None
            or value.get("nonduplicate_prompt_or_region") is not None
        ):
            raise ValueError(f"{name} lacks structured fail-closed semantics")

    configured = arm_contract["configured_budgets"]
    snapshots: dict[str, dict[str, int]] = {}
    for snapshot in ("counters_before", "counters_after"):
        raw = value.get(snapshot)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name}.{snapshot} must be an object")
        require_exact_keys(raw, P1_BUDGET_COUNTER_FIELDS, name=f"{name}.{snapshot}")
        counters = {
            key: nonnegative_int(raw.get(key), name=f"{name}.{snapshot}.{key}")
            for key in P1_BUDGET_COUNTER_FIELDS
        }
        if counters["verifier_round_count"] > configured["max_verifier_rounds"]:
            raise ValueError(f"{name}.{snapshot} exceeds verifier-round budget")
        if counters["environment_step_count"] > configured["max_environment_steps"]:
            raise ValueError(f"{name}.{snapshot} exceeds environment-step budget")
        if counters["network_retry_attempt_count"] > configured["effective_network_retry_limit"]:
            raise ValueError(f"{name}.{snapshot} exceeds network-retry budget")
        if counters["schema_repair_attempt_count"] > counters["stage_initial_attempt_count"]:
            raise ValueError(f"{name}.{snapshot} has excessive schema repairs")
        if counters["tool_failure_count"] > counters["tool_action_attempt_count"]:
            raise ValueError(f"{name}.{snapshot} has excessive tool failures")
        if any(counters[key] > episode_counters[key] for key in P1_BUDGET_COUNTER_FIELDS):
            raise ValueError(f"{name}.{snapshot} exceeds episode accounting")
        snapshots[snapshot] = counters
    before, after = snapshots["counters_before"], snapshots["counters_after"]
    if any(after[key] < before[key] for key in P1_BUDGET_COUNTER_FIELDS):
        raise ValueError(f"{name} counter snapshots are not monotonic")
    if before["verifier_round_count"] != 1 or after["verifier_round_count"] != 1 + int(invoked):
        raise ValueError(f"{name} verifier round snapshots are inconsistent")
    if before["tool_action_attempt_count"] or after["tool_action_attempt_count"]:
        raise ValueError(f"{name} scheduler executed a tool attempt")
    if before["tool_failure_count"] or after["tool_failure_count"]:
        raise ValueError(f"{name} scheduler recorded a tool failure")
    if before["environment_step_count"] != after["environment_step_count"]:
        raise ValueError(f"{name} review changed environment-step accounting")
    if invoked:
        exhausted = (
            observed_outcome == "unique_safe_plan"
            and before["environment_step_count"] >= configured["max_environment_steps"]
        )
        expected_effective = "budget_exhausted" if exhausted else observed_outcome
        expected_violation = (
            "environment_step_budget_exhausted_after_review" if exhausted else None
        )
        if effective_outcome != expected_effective or contract_violation != expected_violation:
            raise ValueError(f"{name} effective outcome/violation differs")

    calls = value.get("backend_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{name}.backend_calls must be a list")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, call in enumerate(calls):
        call_name = f"{name}.backend_calls[{index}]"
        if not isinstance(call, Mapping) or call.get("round_index") != 2:
            raise ValueError(f"{call_name} is malformed")
        require_exact_keys(
            call,
            P1_NORMALIZED_ROUND_TWO_CALL_KEYS,
            name=call_name,
        )
        stage = nonempty_string(call.get("stage"), name=f"{call_name}.stage")
        region = nonempty_string(call.get("region_id"), name=f"{call_name}.region_id")
        attempt = call.get("attempt_kind")
        if attempt not in {"initial", "repair"}:
            raise ValueError(f"{call_name}.attempt_kind differs")
        retries = nonnegative_int(
            call.get("network_retry_attempt_count"),
            name=f"{call_name}.network_retry_attempt_count",
        )
        if retries > configured["effective_network_retry_limit"]:
            raise ValueError(f"{call_name} exceeds network retry budget")
        if not isinstance(call.get("raw_response"), str):
            raise ValueError(f"{call_name}.raw_response is not retained")
        parsed_output = call.get("parsed_output")
        if parsed_output is not None and (
            not isinstance(parsed_output, Mapping)
            or parsed_output.get("region_id") != region
        ):
            raise ValueError(f"{call_name}.parsed_output differs from region_id")
        visual_input_scope = call.get("visual_input_scope")
        if not isinstance(visual_input_scope, str) or not visual_input_scope:
            raise ValueError(f"{call_name}.visual_input_scope is not retained")
        prompt = call.get("prompt_sha256")
        if not isinstance(prompt, str) or SHA256_RE.fullmatch(prompt) is None:
            raise ValueError(f"{call_name}.prompt_sha256 is malformed")
        grouped.setdefault((stage, region), []).append(call)
    for key, grouped_calls in grouped.items():
        attempts = [call["attempt_kind"] for call in grouped_calls]
        if (
            len(attempts) > configured["max_stage_attempts_per_round"]
            or attempts[0] != "initial"
            or attempts.count("initial") != 1
            or attempts.count("repair") > 1
        ):
            raise ValueError(f"{name} stage/region {key!r} exceeds attempt budget")
        if grouped_calls[-1].get("parsed_output") is None or (
            len(grouped_calls) == 2
            and grouped_calls[0].get("parsed_output") is not None
        ):
            raise ValueError(f"{name} stage/region {key!r} has invalid parse evidence")
    observed_delta = {key: after[key] - before[key] for key in P1_BUDGET_COUNTER_FIELDS}
    expected_delta = {
        "verifier_round_count": round_count,
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
    if observed_delta != expected_delta:
        raise ValueError(f"{name} counter delta differs from round-two calls")
    if passthrough:
        if calls or before != after:
            raise ValueError(f"{name} passthrough consumed scheduler counters")
    else:
        expected_transition = _expected_p1_transition(trigger, effective_outcome)
        if value.get("transition") != expected_transition:
            raise ValueError(f"{name}.transition differs from frozen table")
        if value.get("safe_plan_returned_to_existing_flow") is not (
            effective_outcome == "unique_safe_plan"
        ):
            raise ValueError(f"{name}.safe_plan_returned_to_existing_flow differs")
    return dict(value)


def _expected_p1_transition(trigger: str, outcome: str) -> dict[str, str]:
    if trigger == "provisional_finish" and outcome == "finish":
        return {
            "round_one_trigger": trigger,
            "round_two_outcome": outcome,
            "review_decision": "confirmed_finish",
            "next_state": "terminate_confirmed_finish",
            "tool_action_authorization": "forbidden",
        }
    if outcome == "unique_safe_plan":
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


def trajectory_backend_accounting(
    trajectory: Mapping[str, Any], terminal_review: Mapping[str, Any] | None, *, name: str
) -> tuple[int, int, int]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{name}.steps must be a list")
    calls: list[Mapping[str, Any]] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"{name}.steps[{step_index}] must be an object")
        execution = step.get("execution")
        if not isinstance(execution, Mapping):
            continue
        post_review = execution.get("post_accept_terminal_review")
        selected = (
            post_review.get("verifier_evidence")
            if isinstance(post_review, Mapping)
            else execution.get("verifier_evidence")
        )
        if not isinstance(selected, Mapping):
            continue
        review = selected.get("terminal_review")
        evidence_blocks: list[Mapping[str, Any]] = []
        if isinstance(review, Mapping):
            for key in ("round_one_evidence", "round_two_evidence"):
                block = review.get(key)
                if isinstance(block, Mapping):
                    evidence_blocks.append(block)
        else:
            evidence_blocks.append(selected)
        for block in evidence_blocks:
            backend_calls = block.get("backend_calls", [])
            if backend_calls is None:
                backend_calls = []
            if not isinstance(backend_calls, list):
                raise ValueError(f"{name}.steps[{step_index}] backend_calls is malformed")
            for call in backend_calls:
                if not isinstance(call, Mapping):
                    raise ValueError(f"{name}.steps[{step_index}] contains a malformed call")
                calls.append(call)
    initial = 0
    repair = 0
    network_retries = 0
    for index, call in enumerate(calls):
        attempt = call.get("attempt_kind")
        if attempt == "initial":
            initial += 1
        elif attempt == "repair":
            repair += 1
        else:
            raise ValueError(f"{name} backend call {index} lacks a frozen attempt kind")
        network_retries += nonnegative_int(
            call.get("network_retry_attempt_count"),
            name=(
                f"{name}.backend_calls[{index}]."
                "network_retry_attempt_count"
            ),
        )
    return initial, repair, network_retries


def _independent_region_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        region_id = item.get("region_id") if isinstance(item, Mapping) else None
        if not isinstance(region_id, str) or not region_id or region_id in result:
            return None
        expected_version = _independent_proposal_version(region_id)
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


def _independent_proposal_version(region_id: str) -> str | None:
    return (
        P1_EMPTY_MASK_SCAN_PROPOSAL_VERSION
        if P1_SCAN_REGION_RE.fullmatch(region_id) is not None
        else None
    )


def _independent_visual_scope_valid(
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


def _independent_trace_region_contracts(
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
        region_ids = _independent_region_ids(raw_records)
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


def _independent_trace_proposal_versions(
    trace: Mapping[str, Any],
) -> dict[str, str | None] | None:
    contracts = _independent_trace_region_contracts(trace)
    return (
        {region_id: contract[1] for region_id, contract in contracts.items()}
        if contracts is not None
        else None
    )


def _independent_id_list(value: Any) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return tuple(value)


def _independent_records_match(value: Any, ids: tuple[str, ...]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(ids)
        and all(
            isinstance(item, Mapping) and item.get("region_id") == region_id
            for item, region_id in zip(value, ids, strict=True)
        )
    )


def _independent_trace_groups(
    trace: Mapping[str, Any],
) -> tuple[tuple[str, str], ...] | None:
    if "plan" not in trace:
        return None
    if trace.get("mode") == "initial":
        ids = _independent_region_ids(trace.get("evidence"))
        if (
            not ids
            or _independent_id_list(trace.get("audited_region_ids")) != ids
            or not _independent_records_match(trace.get("judgments"), ids)
            or not _independent_records_match(trace.get("diagnoses"), ids)
        ):
            return None
        return tuple(("audit", region_id) for region_id in ids)
    if trace.get("mode") != "candidate":
        return None
    candidate_ids = _independent_region_ids(trace.get("evidence"))
    if (
        not candidate_ids
        or _independent_id_list(trace.get("audited_region_ids")) != candidate_ids
        or not _independent_records_match(trace.get("judgments"), candidate_ids)
        or trace.get("diagnoses") != []
    ):
        return None
    groups = [("candidate_evidence", region_id) for region_id in candidate_ids]
    replan_ids = _independent_region_ids(trace.get("replan_evidence"))
    if replan_ids is None:
        return None
    if replan_ids:
        if (
            _independent_id_list(trace.get("replan_audited_region_ids"))
            != replan_ids
            or not _independent_records_match(
                trace.get("replan_judgments"), replan_ids
            )
            or not _independent_records_match(
                trace.get("replan_diagnoses"), replan_ids
            )
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


def independently_validate_staged_region_evidence(
    evidence: Any,
    *,
    require_round_two: bool,
) -> bool:
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
    groups = _independent_trace_groups(trace)
    region_contracts = _independent_trace_region_contracts(trace)
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
        attempts = [initial]
        call_index += 1
        if (
            call_index < len(calls)
            and isinstance(calls[call_index], Mapping)
            and calls[call_index].get("stage") == expected_stage
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
                or not _independent_visual_scope_valid(
                    scan=region_contracts[expected_region][0] == "scan",
                    attempt_kind=call.get("attempt_kind"),
                    visual_input_scope=call.get("visual_input_scope"),
                )
                or isinstance(retry_count, bool)
                or retry_count != 0
                or not isinstance(prompt, str)
                or SHA256_RE.fullmatch(prompt) is None
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


def _independent_round_one_groups(
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
    expected = _independent_trace_groups(trace)
    region_contracts = _independent_trace_region_contracts(trace)
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
                or not _independent_visual_scope_valid(
                    scan=region_contracts[region_id][0] == "scan",
                    attempt_kind=call.get("attempt_kind"),
                    visual_input_scope=call.get("visual_input_scope"),
                )
                or isinstance(retry_count, bool)
                or retry_count != 0
                or not isinstance(prompt, str)
                or SHA256_RE.fullmatch(prompt) is None
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
    errors = evidence.get("regional_validation_errors")
    if not isinstance(errors, list):
        return None
    repair_regions = {
        region_id
        for stage, region_id, attempts in result
        if stage == "audit" and len(attempts) == 2
    }
    seen: set[str] = set()
    for item in errors:
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


def independently_rebuild_terminal_trigger(evidence: Any) -> str | None:
    if _independent_round_one_groups(evidence) is None or not isinstance(
        evidence, Mapping
    ):
        return None
    trace = evidence.get("stage_trace")
    if not isinstance(trace, Mapping):
        return None
    plan = trace.get("plan")
    mode = trace.get("mode")
    diagnoses = trace.get(
        "diagnoses" if mode == "initial" else "replan_diagnoses"
    )
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
        and isinstance(diagnoses, list)
        and diagnoses
        and all(
            isinstance(item, Mapping) and item.get("error_type") == "none"
            for item in diagnoses
        )
        and production_trace_decision
        and evidence.get("regional_validation_errors") == []
    ):
        return "provisional_finish"
    if "plan" in trace and plan is None:
        return "plan_null_after_allowed_repair"
    return None


def independently_rebuild_round_two_outcome(evidence: Any) -> str | None:
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


def independently_validate_review_identity_and_selection(
    raw_review: Mapping[str, Any],
    normalized: Mapping[str, Any],
    *,
    name: str,
) -> None:
    round_two = raw_review.get("round_two_evidence")
    if normalized.get("round_two_invoked") is not True:
        if round_two is not None or normalized.get("backend_calls") != []:
            raise ValueError(f"{name} retains calls without round-two invocation")
        return
    round_one = raw_review.get("round_one_evidence")
    if not isinstance(round_one, Mapping) or not isinstance(round_two, Mapping):
        raise ValueError(f"{name} lacks raw round-one/round-two evidence")
    raw_two_calls = round_two.get("backend_calls")
    normalized_calls = normalized.get("backend_calls")
    if (
        not isinstance(raw_two_calls, list)
        or not isinstance(normalized_calls, list)
        or len(raw_two_calls) != len(normalized_calls)
    ):
        raise ValueError(f"{name} raw/normalized backend calls are malformed")
    round_one_groups = _independent_round_one_groups(round_one)
    if round_one_groups is None:
        raise ValueError(f"{name} raw round-one calls are malformed")
    round_two_trace = round_two.get("stage_trace")
    region_contracts = (
        _independent_trace_region_contracts(round_two_trace)
        if isinstance(round_two_trace, Mapping)
        else None
    )
    if region_contracts is None:
        raise ValueError(f"{name} round-two proposal versions are malformed")
    prior: set[tuple[str, str]] = set()
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
                raise ValueError(f"{name} validated round-one identity is malformed")
            prior.add((region, prompt))
    expected_selection: tuple[str, str, str] | None = None
    if normalized.get("trigger") == "plan_null_after_allowed_repair":
        trace = round_one.get("stage_trace")
        if not isinstance(trace, Mapping):
            raise ValueError(f"{name} plan-null round one lacks a trace")
        if trace.get("mode") == "initial":
            raw_regions = trace.get("evidence")
            raw_audited = trace.get("audited_region_ids")
            raw_diagnoses = trace.get("diagnoses")
        elif trace.get("mode") == "candidate":
            raw_regions = trace.get("replan_evidence")
            raw_audited = trace.get("replan_audited_region_ids")
            raw_diagnoses = trace.get("replan_diagnoses")
        else:
            raise ValueError(f"{name} plan-null trace mode is malformed")
        region_ids = _independent_region_ids(raw_regions)
        audited_ids = _independent_id_list(raw_audited)
        if not region_ids or audited_ids is None:
            raise ValueError(f"{name} plan-null region order is malformed")
        next_unreviewed = next(
            (region_id for region_id in region_ids if region_id not in audited_ids),
            None,
        )
        if next_unreviewed is not None:
            expected_selection = (
                next_unreviewed,
                "qwen3vl_plan_null_next_unreviewed_region_v1",
                "next_unreviewed_environment_region",
            )
        else:
            diagnoses = {
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
                    or diagnoses.get(region_id, {}).get("error_type") != "none"
                ),
                None,
            )
            if unresolved is not None:
                expected_selection = (
                    unresolved,
                    "qwen3vl_plan_null_alternative_diagnosis_v1",
                    "first_unresolved_environment_region",
                )
        if expected_selection is None:
            raise ValueError(f"{name} plan-null selection cannot be rebuilt")
    observed: set[tuple[str, str]] = set()
    for index, (raw_call, normalized_call) in enumerate(
        zip(raw_two_calls, normalized_calls, strict=True)
    ):
        if not isinstance(raw_call, Mapping) or not isinstance(
            normalized_call, Mapping
        ):
            raise ValueError(f"{name} backend call {index} is malformed")
        context = raw_call.get("terminal_review")
        if (
            not isinstance(context, Mapping)
            or set(context) != set(P1_TERMINAL_REVIEW_CONTEXT_KEYS)
        ):
            raise ValueError(f"{name} raw round-two call lacks exact review context")
        require_exact_keys(
            normalized_call,
            P1_NORMALIZED_ROUND_TWO_CALL_KEYS,
            name=f"{name}.backend_calls[{index}]",
        )
        region = context.get("review_region_id")
        if "raw_response" not in raw_call:
            raise ValueError(f"{name} raw round-two call {index} lacks raw_response")
        expected = {
            "round_index": context.get("round_index"),
            "stage": raw_call.get("stage"),
            "region_id": region,
            "attempt_kind": raw_call.get("attempt_kind"),
            "prompt_sha256": raw_call.get("prompt_sha256"),
            "raw_response": raw_call.get("raw_response"),
            "parsed_output": raw_call.get("parsed_output"),
            "visual_input_scope": raw_call.get("visual_input_scope"),
            "network_retry_attempt_count": raw_call.get(
                "network_retry_attempt_count"
            ),
        }
        if any(normalized_call.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name} normalized call {index} differs from raw")
        expected_contract = region_contracts.get(normalized_call["region_id"])
        if (
            expected_contract is None
            or region != normalized_call["region_id"]
            or context.get("proposal_version") != expected_contract[1]
            or not _independent_visual_scope_valid(
                scan=expected_contract[0] == "scan",
                attempt_kind=normalized_call["attempt_kind"],
                visual_input_scope=normalized_call["visual_input_scope"],
            )
        ):
            raise ValueError(
                f"{name} round-two proposal_version differs from region identity"
            )
        if normalized_call.get("attempt_kind") == "initial":
            observed.add(
                (
                    str(normalized_call.get("region_id")),
                    str(normalized_call.get("prompt_sha256")),
                )
            )
        if expected_selection is not None:
            expected_region, expected_prompt, expected_branch = expected_selection
            if (
                region != expected_region
                or context.get("preferred_region_id") != expected_region
                or context.get("prompt_version") != expected_prompt
                or context.get("selection_branch") != expected_branch
            ):
                raise ValueError(f"{name} plan-null selection differs from round one")
        elif (
            context.get("preferred_region_id") is not None
            or context.get("prompt_version") is not None
            or context.get("selection_branch") is not None
        ):
            raise ValueError(f"{name} finish review carries plan-null metadata")
    if not observed or not (observed - prior):
        raise ValueError(f"{name} does not prove a new prompt/region identity")


def independent_terminal_review_records(
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("trajectory.steps must be a list")
    records: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"trajectory.steps[{step_index}] must be an object")
        execution = step.get("execution")
        if not isinstance(execution, Mapping):
            continue
        if "post_accept_terminal_review" in execution:
            post = execution.get("post_accept_terminal_review")
            if not isinstance(post, Mapping) or "verifier_evidence" not in post:
                raise ValueError(
                    f"trajectory.steps[{step_index}] post-review envelope is malformed"
                )
            selected = post.get("verifier_evidence")
            if not isinstance(selected, Mapping):
                raise ValueError(
                    f"trajectory.steps[{step_index}] post-review evidence is malformed"
                )
        else:
            selected = execution.get("verifier_evidence")
            if "verifier_evidence" in execution and not isinstance(selected, Mapping):
                raise ValueError(
                    f"trajectory.steps[{step_index}] verifier evidence is malformed"
                )
        blocks: list[Mapping[str, Any]] = []
        if isinstance(selected, Mapping):
            blocks.append(selected)
        if "rollback_replan" in execution:
            rollback = execution.get("rollback_replan")
            if not isinstance(rollback, Mapping) or "verifier_evidence" not in rollback:
                raise ValueError(
                    f"trajectory.steps[{step_index}] rollback envelope is malformed"
                )
            rollback_evidence = rollback.get("verifier_evidence")
            if not isinstance(rollback_evidence, Mapping):
                raise ValueError(
                    f"trajectory.steps[{step_index}] rollback evidence is malformed"
                )
            blocks.append(rollback_evidence)
        for block_index, block in enumerate(blocks):
            if "terminal_review" not in block:
                continue
            review = block.get("terminal_review")
            if not isinstance(review, Mapping):
                raise ValueError(
                    f"trajectory.steps[{step_index}] terminal review "
                    f"{block_index} is malformed"
                )
            round_one = review.get("round_one_evidence")
            if not isinstance(round_one, Mapping):
                raise ValueError(
                    f"trajectory.steps[{step_index}] terminal review lacks "
                    "round-one evidence"
                )
            round_two = review.get("round_two_evidence")
            if round_two is not None and not isinstance(round_two, Mapping):
                raise ValueError(
                    f"trajectory.steps[{step_index}] terminal review has malformed "
                    "round-two evidence"
                )
            records.append(
                {
                    "audit": {
                        key: value
                        for key, value in review.items()
                        if key not in {"round_one_evidence", "round_two_evidence"}
                    },
                    "round_one_evidence": round_one,
                    "round_two_evidence": round_two,
                }
            )
    return records


def independently_rebuild_final_region_protocol_valid(
    trajectory: Mapping[str, Any],
    *,
    arm: str,
) -> tuple[bool, list[dict[str, Any]]]:
    reviews = independent_terminal_review_records(trajectory)
    if arm not in {"h0", "h1"} or (arm == "h0" and reviews):
        return False, reviews
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return False, reviews
    final_block: Any = None
    block_seen = False
    for step in steps:
        if not isinstance(step, Mapping):
            return False, reviews
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
        return False, reviews
    if arm == "h1":
        raw_review = (
            final_block.get("terminal_review")
            if isinstance(final_block, Mapping)
            else None
        )
        if not isinstance(raw_review, Mapping) or not reviews:
            return False, reviews
        final_record = {
            "audit": {
                key: value
                for key, value in raw_review.items()
                if key not in {"round_one_evidence", "round_two_evidence"}
            },
            "round_one_evidence": raw_review.get("round_one_evidence"),
            "round_two_evidence": raw_review.get("round_two_evidence"),
        }
        if final_record != reviews[-1]:
            return False, reviews
        audit = final_record["audit"]
        if (
            audit.get("eligible") is not True
            or audit.get("round_two_invoked") is not True
            or audit.get("round_two_count") != 1
            or audit.get("round_index") != 2
        ):
            return False, reviews
        return (
            independently_validate_staged_region_evidence(
                final_record["round_two_evidence"],
                require_round_two=True,
            ),
            reviews,
        )
    return (
        independently_validate_staged_region_evidence(
            final_block,
            require_round_two=False,
        ),
        reviews,
    )


def validate_p1_audit(
    raw: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    samples: Sequence[str],
    experiment_arm: str,
    harness_protocol_id: str,
    arm_contract: Mapping[str, Any],
    summary_samples: Mapping[str, Any],
    episode_p1_samples: Mapping[str, Any],
    trajectories: Mapping[str, Mapping[str, Any]],
    tool_report_counts: Mapping[str, tuple[int, int]],
    artifact_bindings: Mapping[str, Any],
) -> None:
    require_exact_keys(
        raw,
        (
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
        ),
        name="p1_harness_audit",
    )
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
        if raw.get(key) != expected:
            raise ValueError(f"p1_harness_audit.{key} differs from frozen identity")
    raw_samples = raw.get("per_sample")
    if not isinstance(raw_samples, Mapping) or list(raw_samples) != list(samples):
        raise ValueError("p1_harness_audit per-sample identity/order differs")

    totals = {key: 0 for key in P1_BUDGET_COUNTER_FIELDS}
    normalized_samples: dict[str, Any] = {}
    round_two_count = 0
    valid_region_count = 0
    configured = arm_contract["configured_budgets"]
    for sample in samples:
        name = f"p1_harness_audit.per_sample.{sample}"
        item = raw_samples.get(sample)
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be an object")
        require_exact_keys(
            item,
            (
                "terminal_review",
                "terminal_reviews",
                "budget_accounting",
                "gt_boundary_audit",
                "safety_boundary_audit",
                "final_region_protocol_valid",
            ),
            name=name,
        )
        episode_item = episode_p1_samples.get(sample)
        if not isinstance(episode_item, Mapping) or dict(item) != dict(episode_item):
            raise ValueError(f"{name} differs from episode summary evidence")
        raw_budget = item.get("budget_accounting")
        if not isinstance(raw_budget, Mapping):
            raise ValueError(f"{name}.budget_accounting must be an object")
        require_exact_keys(
            raw_budget, P1_BUDGET_COUNTER_FIELDS, name=f"{name}.budget_accounting"
        )
        counters = {
            key: nonnegative_int(
                raw_budget.get(key), name=f"{name}.budget_accounting.{key}"
            )
            for key in P1_BUDGET_COUNTER_FIELDS
        }
        if not 1 <= counters["verifier_round_count"] <= configured["max_verifier_rounds"]:
            raise ValueError(f"{name} verifier-round budget is invalid")
        if counters["environment_step_count"] > configured["max_environment_steps"]:
            raise ValueError(f"{name} environment-step budget is exceeded")
        if counters["network_retry_attempt_count"] > configured["effective_network_retry_limit"]:
            raise ValueError(f"{name} network-retry budget is exceeded")
        if counters["schema_repair_attempt_count"] > counters["stage_initial_attempt_count"]:
            raise ValueError(f"{name} schema repair accounting is invalid")
        if counters["tool_failure_count"] > counters["tool_action_attempt_count"]:
            raise ValueError(f"{name} tool accounting is invalid")

        terminal_review = validate_terminal_review(
            item.get("terminal_review"),
            name=f"{name}.terminal_review",
            arm=experiment_arm,
            arm_contract=arm_contract,
            episode_counters=counters,
        )
        invoked = bool(
            terminal_review is not None and terminal_review.get("round_two_invoked") is True
        )
        round_two_count += int(invoked)
        if counters["verifier_round_count"] != 1 + int(invoked):
            raise ValueError(f"{name} verifier-round count disagrees with terminal review")
        initial, repair, network = trajectory_backend_accounting(
            trajectories[sample], terminal_review, name=f"trajectory.{sample}"
        )
        if (
            counters["stage_initial_attempt_count"] != initial
            or counters["schema_repair_attempt_count"] != repair
            or counters["network_retry_attempt_count"] != network
        ):
            raise ValueError(f"{name} stage-attempt accounting differs from raw calls")
        summary_sample = summary_samples[sample]
        successes, failures = tool_report_counts[sample]
        if counters["tool_action_attempt_count"] != successes + failures:
            raise ValueError(f"{name} tool attempts differ from raw reports")
        if counters["tool_failure_count"] != failures:
            raise ValueError(f"{name} tool failures differ from raw reports")
        if counters["environment_step_count"] != summary_sample.get("candidate_count"):
            raise ValueError(f"{name} environment steps differ from trajectory steps")

        gt = item.get("gt_boundary_audit")
        if not isinstance(gt, Mapping):
            raise ValueError(f"{name}.gt_boundary_audit must be an object")
        require_exact_keys(
            gt,
            (
                "pre_rollout_gt_access_count",
                "violation_ids",
                "rollout_completed_before_offline_scoring",
            ),
            name=f"{name}.gt_boundary_audit",
        )
        gt_count = nonnegative_int(
            gt.get("pre_rollout_gt_access_count"),
            name=f"{name}.gt_boundary_audit.pre_rollout_gt_access_count",
        )
        violation_ids = string_list(
            gt.get("violation_ids"), name=f"{name}.gt_boundary_audit.violation_ids"
        )
        if gt_count != len(violation_ids) or gt_count:
            raise ValueError(f"{name} fails the pre-rollout GT hard gate")
        if gt.get("rollout_completed_before_offline_scoring") is not True:
            raise ValueError(f"{name} lacks the offline-scoring boundary")

        safety = item.get("safety_boundary_audit")
        if not isinstance(safety, Mapping):
            raise ValueError(f"{name}.safety_boundary_audit must be an object")
        require_exact_keys(
            safety,
            ("relaxation_count", "relaxed_boundaries"),
            name=f"{name}.safety_boundary_audit",
        )
        relaxation_count = nonnegative_int(
            safety.get("relaxation_count"),
            name=f"{name}.safety_boundary_audit.relaxation_count",
        )
        relaxed = string_list(
            safety.get("relaxed_boundaries"),
            name=f"{name}.safety_boundary_audit.relaxed_boundaries",
        )
        if (
            relaxation_count != len(relaxed)
            or relaxation_count
            or any(boundary not in P1_SAFETY_BOUNDARIES for boundary in relaxed)
        ):
            raise ValueError(f"{name} relaxes a frozen safety boundary")
        protocol_valid = item.get("final_region_protocol_valid")
        if not isinstance(protocol_valid, bool):
            raise ValueError(f"{name}.final_region_protocol_valid is malformed")
        rebuilt_protocol_valid, raw_reviews = (
            independently_rebuild_final_region_protocol_valid(
                trajectories[sample],
                arm=arm_contract["arm"],
            )
        )
        validated_history: list[dict[str, Any]] = []
        invoked_state_ids: set[str] = set()
        for review_index, raw_review in enumerate(raw_reviews):
            rebuilt_review = validate_terminal_review(
                raw_review["audit"],
                name=f"{name}.raw_terminal_reviews[{review_index}]",
                arm=experiment_arm,
                arm_contract=arm_contract,
                episode_counters=counters,
            )
            if rebuilt_review is None:
                raise ValueError(f"{name} has a null H1 raw terminal review")
            rebuilt_trigger = independently_rebuild_terminal_trigger(
                raw_review.get("round_one_evidence")
            )
            if rebuilt_review.get("trigger") != rebuilt_trigger:
                raise ValueError(
                    f"{name}.raw_terminal_reviews[{review_index}] trigger differs "
                    "from raw round one"
                )
            invoked_review = rebuilt_review.get("round_two_invoked") is True
            raw_round_two = raw_review.get("round_two_evidence")
            if invoked_review != isinstance(raw_round_two, Mapping):
                raise ValueError(
                    f"{name}.raw_terminal_reviews[{review_index}] round-two "
                    "presence differs"
                )
            if invoked_review:
                rebuilt_outcome = independently_rebuild_round_two_outcome(
                    raw_round_two
                )
                if rebuilt_review.get("observed_outcome") != rebuilt_outcome:
                    raise ValueError(
                        f"{name}.raw_terminal_reviews[{review_index}] outcome "
                        "differs from raw round two"
                    )
                state_id = str(rebuilt_review.get("accepted_state_id"))
                if state_id in invoked_state_ids:
                    raise ValueError(
                        f"{name} invokes round two twice for one accepted state"
                    )
                invoked_state_ids.add(state_id)
            independently_validate_review_identity_and_selection(
                raw_review,
                rebuilt_review,
                name=f"{name}.raw_terminal_reviews[{review_index}]",
            )
            validated_history.append(rebuilt_review)
        raw_tail_audit = validated_history[-1] if validated_history else None
        declared_history = item.get("terminal_reviews")
        if not isinstance(declared_history, list) or declared_history != validated_history:
            raise ValueError(
                f"{name}.terminal_reviews differs from the raw trajectory history"
            )
        if raw_tail_audit != terminal_review:
            raise ValueError(
                f"{name}.terminal_review differs from the raw trajectory tail"
            )
        if protocol_valid is not rebuilt_protocol_valid:
            raise ValueError(
                f"{name}.final_region_protocol_valid differs from raw staged evidence"
            )
        valid_region_count += int(protocol_valid)
        for key in P1_BUDGET_COUNTER_FIELDS:
            totals[key] += counters[key]
        normalized_samples[sample] = {
            "budget_accounting": counters,
            "gt_boundary_audit": dict(gt),
            "safety_boundary_audit": dict(safety),
            "terminal_review": terminal_review,
            "terminal_reviews": declared_history,
            "final_region_protocol_valid": protocol_valid,
        }

    raw_totals = raw.get("totals")
    if not isinstance(raw_totals, Mapping):
        raise ValueError("p1_harness_audit.totals must be an object")
    require_exact_keys(raw_totals, P1_BUDGET_COUNTER_FIELDS, name="p1_harness_audit.totals")
    normalized_totals = {
        key: nonnegative_int(raw_totals.get(key), name=f"p1_harness_audit.totals.{key}")
        for key in P1_BUDGET_COUNTER_FIELDS
    }
    if normalized_totals != totals:
        raise ValueError("p1_harness_audit totals differ from sample sums")
    for key in (
        "gt_boundary_violation_count",
        "safety_boundary_relaxation_count",
        "tool_failure_count",
    ):
        if nonnegative_int(raw.get(key), name=f"p1_harness_audit.{key}") != 0:
            raise ValueError(f"p1_harness_audit.{key} hard gate failed")
    if raw.get("tool_failure_count") != totals["tool_failure_count"]:
        raise ValueError("p1_harness_audit tool failure total differs")

    expected_contract = {
        "schema_version": P1_SUMMARY_CONTRACT_VERSION,
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
        "final_region_protocol_valid_count": valid_region_count,
        "round_two_invoked_count": round_two_count,
        "per_sample": normalized_samples,
        "hard_gates": {
            "requested_scored_mismatch_count": 0,
            "pre_rollout_gt_violation_count": 0,
            "safety_boundary_relaxation_count": 0,
            "tool_failure_count": 0,
            "all_passed": True,
        },
        "provenance": {
            "model_state": "base",
            "runtime_commit": summary.get("runtime_commit"),
            "harness_commit": summary.get("harness_commit"),
            "job_id": summary.get("job_id"),
            "artifact_bindings": artifact_bindings,
        },
    }
    if summary.get("p1_contract") != expected_contract:
        raise ValueError("summary.p1_contract differs from independently rebuilt evidence")


def validate_summary(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    summary_path = args.summary.resolve()
    if (run_dir / ".run_incomplete").exists():
        raise ValueError("run is still marked incomplete")
    summary = read_object(summary_path)
    samples = [str(item) for item in args.expected_samples]
    evaluation_id, canonical_samples = EVALUATION_PROFILES[args.evaluation_profile]
    if args.evaluation_id != evaluation_id:
        raise ValueError("expected evaluation ID differs from profile")
    if samples != list(canonical_samples) or len(samples) != len(set(samples)):
        raise ValueError("expected sample identity/order differs from profile")
    arm_contract = P1_ARM_CONTRACTS[args.experiment_arm]
    if args.harness_protocol_id != arm_contract["harness_protocol_id"]:
        raise ValueError("experiment arm and harness protocol ID differ")
    if not COMMIT_RE.fullmatch(args.runtime_commit):
        raise ValueError("expected runtime commit must be a 40-hex commit")
    if not COMMIT_RE.fullmatch(args.harness_commit):
        raise ValueError("expected harness commit must be a 40-hex commit")
    expected_identity = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "model_state": "base",
        "job_id": str(args.job_id),
        "experiment_arm": args.experiment_arm,
        "protocol_version": P1_MODEL_IO_PROTOCOL,
        "harness_protocol_id": args.harness_protocol_id,
        "runtime_commit": args.runtime_commit,
        "harness_commit": args.harness_commit,
        "requested_samples": len(samples),
        "scored_samples": len(samples),
        "sample_order": samples,
        "rollout_marker": str(run_dir / "rollout_complete.json"),
        "source_metrics": str(run_dir / "per_sample_metrics.json"),
    }
    for key, expected in expected_identity.items():
        if summary.get(key) != expected:
            raise ValueError(f"summary.{key} differs from expected P1 identity")
    if summary.get("requested_samples") != summary.get("scored_samples"):
        raise ValueError("requested/scored hard gate failed")
    boundary = summary.get("gt_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("rollout_marker_claims_gt_free") is not True
        or boundary.get("offline_metrics_are_post_rollout") is not True
    ):
        raise ValueError("summary lacks the GT-free rollout boundary")
    artifact_bindings = validate_artifact_bindings(
        summary.get("artifact_bindings"), run_dir=run_dir, samples=samples
    )

    marker = read_object(run_dir / "rollout_complete.json")
    if (
        marker.get("execution_status") != "completed"
        or marker.get("gt_loaded") is not False
        or marker.get("samples") != sorted(samples)
    ):
        raise ValueError("rollout marker is not canonical/completed/GT-free")
    raw_metrics = read_object(run_dir / "per_sample_metrics.json")
    raw_samples = raw_metrics.get("samples")
    summary_samples = summary.get("per_sample")
    if (
        not isinstance(raw_samples, Mapping)
        or list(raw_samples) != samples
        or not isinstance(summary_samples, Mapping)
        or list(summary_samples) != samples
    ):
        raise ValueError("raw or summary per-sample identity/order differs")

    initial_total = {key: 0 for key in COUNT_KEYS}
    selected_total = {key: 0 for key in COUNT_KEYS}
    initial_macro: list[dict[str, int]] = []
    selected_macro: list[dict[str, int]] = []
    action_totals: Counter[str] = Counter()
    outcome_totals: Counter[str] = Counter()
    oracle_total = empty_oracle_totals(include_candidates=False)
    degraded_count = 0
    episode_p1_samples: dict[str, Any] = {}
    trajectories: dict[str, Mapping[str, Any]] = {}
    tool_report_counts: dict[str, tuple[int, int]] = {}
    for sample in samples:
        name = f"sample.{sample}"
        raw = raw_samples[sample]
        observed = summary_samples[sample]
        if not isinstance(raw, Mapping) or not isinstance(observed, Mapping):
            raise ValueError(f"{name} raw/summary evidence is malformed")
        raw_initial = metric_counts(raw.get("initial"), name=f"{name}.raw.initial")
        raw_selected = metric_counts(
            raw.get("verifier_selected"), name=f"{name}.raw.selected"
        )
        summary_initial = metric_counts(observed.get("initial"), name=f"{name}.initial")
        summary_selected = metric_counts(observed.get("selected"), name=f"{name}.selected")
        if raw_initial != summary_initial or raw_selected != summary_selected:
            raise ValueError(f"{name} summary metrics differ from raw metrics")
        add_counts(initial_total, raw_initial)
        add_counts(selected_total, raw_selected)
        initial_macro.append(raw_initial)
        selected_macro.append(raw_selected)
        initial_rates = rates_from_counts(raw_initial)
        selected_rates = rates_from_counts(raw_selected)
        close_number(
            observed.get("delta_iou"),
            round(selected_rates["iou"] - initial_rates["iou"], 8),
            name=f"{name}.delta_iou",
        )
        close_number(
            observed.get("delta_f1"),
            round(selected_rates["f1"] - initial_rates["f1"], 8),
            name=f"{name}.delta_f1",
        )
        if observed.get("delta_counts") != count_deltas(raw_initial, raw_selected):
            raise ValueError(f"{name}.delta_counts differs")
        is_degraded = degraded(raw_initial, raw_selected)
        if observed.get("degraded") is not is_degraded:
            raise ValueError(f"{name}.degraded differs")
        degraded_count += int(is_degraded)

        episode = read_object(run_dir / "trajectories" / sample / "episode_summary.json")
        if episode.get("sample") != sample or episode.get("execution_status") != "completed":
            raise ValueError(f"{name} episode is not canonical/completed")
        expected_episode_identity = {
            "harness_protocol_id": args.harness_protocol_id,
            "experiment_arm_id": args.experiment_arm,
            "terminal_review_factor_id": P1_FACTOR_ID,
            "terminal_review_arm": arm_contract["arm"],
        }
        for key, expected in expected_episode_identity.items():
            if episode.get(key) != expected:
                raise ValueError(f"{name}.episode.{key} differs")
        outcome = nonempty_string(episode.get("episode_outcome"), name=f"{name}.outcome")
        stop_reason = nonempty_string(episode.get("stop_reason"), name=f"{name}.stop_reason")
        if observed.get("episode_outcome") != outcome or observed.get("stop_reason") != stop_reason:
            raise ValueError(f"{name} summary outcome differs from episode")
        outcome_totals[outcome] += 1
        episode_values = {
            "candidate_count": nonnegative_int(
                episode.get("candidate_count"), name=f"{name}.candidate_count"
            ),
            "accepted_candidate_count": nonnegative_int(
                episode.get("accepted_candidate_count"), name=f"{name}.accepted"
            ),
            "rejected_candidate_count": nonnegative_int(
                episode.get("rejected_candidate_count"), name=f"{name}.rejected"
            ),
            "tool_action_count": nonnegative_int(
                episode.get("tool_action_count"), name=f"{name}.tool_actions"
            ),
            "action_attempt_count": nonnegative_int(
                episode.get("action_attempt_count"), name=f"{name}.action_attempts"
            ),
            "action_attempt_error_count": nonnegative_int(
                episode.get("action_attempt_error_count"), name=f"{name}.attempt_errors"
            ),
            "fallback_actions": nonnegative_int(
                episode.get("empty_mask_scan_fallback_action_count"),
                name=f"{name}.fallback_actions",
            ),
        }
        if episode_values["candidate_count"] != (
            episode_values["accepted_candidate_count"]
            + episode_values["rejected_candidate_count"]
        ):
            raise ValueError(f"{name} candidate accounting differs")
        if (
            episode_values["action_attempt_count"] < episode_values["candidate_count"]
            or episode_values["action_attempt_error_count"]
            > episode_values["action_attempt_count"]
        ):
            raise ValueError(f"{name} action-attempt accounting differs")
        for key in (
            "candidate_count",
            "accepted_candidate_count",
            "rejected_candidate_count",
            "tool_action_count",
            "action_attempt_count",
            "action_attempt_error_count",
        ):
            if observed.get(key) != episode_values[key]:
                raise ValueError(f"{name}.{key} differs from episode")
        trajectory = read_object(run_dir / "trajectories" / sample / "trajectory.json")
        trajectories[sample] = trajectory
        oracle = candidate_oracle(
            trajectory, episode_values["candidate_count"], name=name
        )
        if observed.get("candidate_oracle") != oracle:
            raise ValueError(f"{name}.candidate_oracle differs from trajectory")
        add_oracle(oracle_total, oracle)

        reports = artifact_bindings["per_sample"][sample]["tool_reports"]
        successes = 0
        failures = 0
        for report_binding in reports:
            report = read_object(Path(report_binding["path"]))
            if report.get("status") == "success":
                successes += 1
            else:
                failures += 1
        if successes != episode_values["tool_action_count"]:
            raise ValueError(f"{name} successful tool reports differ from episode")
        if observed.get("successful_tool_action_count") != successes:
            raise ValueError(f"{name} successful-tool count differs")
        if observed.get("tool_failure_count") != failures:
            raise ValueError(f"{name} tool-failure count differs")
        tool_report_counts[sample] = (successes, failures)
        action_totals.update(
            candidates=episode_values["candidate_count"],
            accepted=episode_values["accepted_candidate_count"],
            rejected=episode_values["rejected_candidate_count"],
            tool_actions=episode_values["tool_action_count"],
            action_attempts=episode_values["action_attempt_count"],
            action_attempt_errors=episode_values["action_attempt_error_count"],
            fallback_actions=episode_values["fallback_actions"],
            successful_tool_actions=successes,
            tool_failure_count=failures,
            finish_count=int(stop_reason in FINISH_REASONS),
        )
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

    raw_aggregate = raw_metrics.get("aggregate")
    if metric_counts(raw_aggregate, name="raw.aggregate") != selected_total:
        raise ValueError("raw aggregate counts differ from per-sample sums")
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping) or aggregate.get("aggregation") != "micro_count_sum":
        raise ValueError("summary aggregate contract is malformed")
    aggregate_initial = metric_counts(aggregate.get("initial"), name="aggregate.initial")
    aggregate_selected = metric_counts(aggregate.get("selected"), name="aggregate.selected")
    if aggregate_initial != initial_total or aggregate_selected != selected_total:
        raise ValueError("summary aggregate counts differ from per-sample sums")
    initial_rates = rates_from_counts(initial_total)
    selected_rates = rates_from_counts(selected_total)
    close_number(
        aggregate.get("delta_iou"),
        round(selected_rates["iou"] - initial_rates["iou"], 8),
        name="aggregate.delta_iou",
    )
    close_number(
        aggregate.get("delta_f1"),
        round(selected_rates["f1"] - initial_rates["f1"], 8),
        name="aggregate.delta_f1",
    )
    if aggregate.get("delta_counts") != count_deltas(initial_total, selected_total):
        raise ValueError("aggregate.delta_counts differs")

    macro = summary.get("macro")
    if not isinstance(macro, Mapping) or macro.get("aggregation") != "sample_arithmetic_mean":
        raise ValueError("summary macro contract is malformed")
    expected_initial_macro = macro_metrics(initial_macro)
    expected_selected_macro = macro_metrics(selected_macro)
    for state, expected in (
        ("initial", expected_initial_macro),
        ("selected", expected_selected_macro),
    ):
        observed = macro.get(state)
        if not isinstance(observed, Mapping):
            raise ValueError(f"macro.{state} must be an object")
        for key in METRIC_KEYS:
            close_number(observed.get(key), expected[key], name=f"macro.{state}.{key}")
    close_number(
        macro.get("delta_iou"),
        round(expected_selected_macro["iou"] - expected_initial_macro["iou"], 8),
        name="macro.delta_iou",
    )
    close_number(
        macro.get("delta_f1"),
        round(expected_selected_macro["f1"] - expected_initial_macro["f1"], 8),
        name="macro.delta_f1",
    )

    actions = summary.get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError("summary.actions must be an object")
    for key in ACTION_KEYS:
        if nonnegative_int(actions.get(key), name=f"actions.{key}") != action_totals[key]:
            raise ValueError(f"actions.{key} differs from raw evidence")
    tool_denominator = action_totals["successful_tool_actions"] + action_totals["tool_failure_count"]
    expected_rate = (
        action_totals["successful_tool_actions"] / tool_denominator
        if tool_denominator
        else None
    )
    if expected_rate is None:
        if actions.get("tool_success_rate") is not None:
            raise ValueError("actions.tool_success_rate must be null")
    else:
        close_number(
            actions.get("tool_success_rate"), expected_rate, name="actions.tool_success_rate"
        )
    if action_totals["tool_failure_count"] != 0:
        raise ValueError("tool-failure hard gate failed")
    if summary.get("candidate_oracle") != oracle_total:
        raise ValueError("summary candidate-oracle totals differ")
    if summary.get("episode_outcome_counts") != dict(sorted(outcome_totals.items())):
        raise ValueError("summary episode outcomes differ")
    expected_invalid = outcome_totals.get("verifier_invalid", 0) + outcome_totals.get(
        "no_executable_action", 0
    )
    if summary.get("invalid_or_no_action_samples") != expected_invalid:
        raise ValueError("summary invalid/no-action count differs")
    if summary.get("degraded_sample_count") != degraded_count:
        raise ValueError("summary degraded-sample count differs")

    validate_p1_audit(
        read_object(run_dir / "p1_harness_audit.json"),
        summary=summary,
        samples=samples,
        experiment_arm=args.experiment_arm,
        harness_protocol_id=args.harness_protocol_id,
        arm_contract=arm_contract,
        summary_samples=summary_samples,
        episode_p1_samples=episode_p1_samples,
        trajectories=trajectories,
        tool_report_counts=tool_report_counts,
        artifact_bindings=artifact_bindings,
    )
    return {
        "status": "valid",
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "experiment_arm": args.experiment_arm,
        "harness_protocol_id": args.harness_protocol_id,
        "requested_samples": len(samples),
        "scored_samples": len(samples),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_summary(args)
    except ValueError as error:
        print(f"qwen3vl_e2e_summary_validator: {error}", file=sys.stderr)
        return 3
    print(
        "status=valid;"
        f"profile={result['evaluation_profile']};"
        f"arm={result['experiment_arm']};"
        f"scored={result['scored_samples']}/{result['requested_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
