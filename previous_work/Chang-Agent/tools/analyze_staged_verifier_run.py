#!/usr/bin/env python3
"""Summarize staged-verifier protocol behavior from saved GT-free trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
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
P1_MODEL_IO_PROTOCOL = "v12.1"
P1_ARM_PROTOCOLS = {
    "v12_1-h0-legacy-single-round-base-0802": (
        "qwen3vl_harness_h0_legacy_single_round_v1"
    ),
    "v12_1-h1-gtfree-terminal-review-base-0802": (
        "qwen3vl_harness_h1_gtfree_terminal_review_v1"
    ),
}
ALLOWED_BACKEND_STAGES = frozenset(("audit", "candidate_evidence", "decision"))
ALLOWED_ATTEMPT_KINDS = frozenset(("initial", "repair"))
ALLOWED_VISUAL_INPUT_SCOPES = frozenset(
    (
        "standard",
        "authoritative_pair_rgb_only_scan",
        "authoritative_pair_rgb_only_evidence_repair",
        "unreported",
    )
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPORTED_STAGES = ("select", *sorted(ALLOWED_BACKEND_STAGES))
REPORTED_ATTEMPT_KINDS = ("initial", "repair")
PROGRAMMATIC_NO_TRACE_MODES = {
    "programmatic_identical_state",
    "programmatic_alternate_region_after_rollback",
    "programmatic_no_alternate_region_after_rollback",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-profile", choices=tuple(EVALUATION_PROFILES), required=True
    )
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--expected-samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--experiment-arm", default="")
    parser.add_argument("--protocol-version", default="")
    parser.add_argument("--harness-protocol-id", default="")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--runtime-commit", default="")
    parser.add_argument("--harness-commit", default="")
    return parser.parse_args(argv)


def read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"required JSON must be an object: {path}")
    return payload


def nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, found {value!r}")
    return value


def finite_nonnegative(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative, found {value!r}")
    return float(value)


def record_ids(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result: list[str] = []
    for item in value:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("region_id"), str)
            or not item["region_id"]
        ):
            raise ValueError(f"{name} contains a malformed region record")
        result.append(item["region_id"])
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate region IDs")
    return result


def string_ids(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must contain non-empty string IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate region IDs")
    return list(value)


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


def summarize_sample(
    run_dir: Path, sample: str, *, strict_p1_calls: bool = False
) -> dict[str, Any]:
    sample_dir = run_dir / "trajectories" / sample
    trajectory = read_object(sample_dir / "trajectory.json")
    episode = read_object(sample_dir / "episode_summary.json")
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"trajectory has no steps: {sample}")
    if (
        episode.get("sample") != sample
        or episode.get("execution_status") != "completed"
    ):
        raise ValueError(f"episode summary sample mismatch: {sample}")
    outcome = episode.get("episode_outcome")
    stop_reason = episode.get("stop_reason")
    if not isinstance(outcome, str) or not outcome:
        raise ValueError(f"episode outcome is missing: {sample}")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError(f"episode stop reason is missing: {sample}")
    if not isinstance(episode.get("converged"), bool):
        raise ValueError(f"episode converged flag is malformed: {sample}")
    elapsed = finite_nonnegative(
        episode.get("elapsed_seconds"), name=f"{sample}.elapsed_seconds"
    )
    episode_counts = {
        name: nonnegative_int(episode.get(name), name=f"{sample}.{name}")
        for name in (
            "loop_count",
            "candidate_count",
            "accepted_candidate_count",
            "rejected_candidate_count",
            "tool_action_count",
            "empty_mask_scan_fallback_action_count",
            "action_attempt_count",
            "action_attempt_error_count",
        )
    }
    if episode_counts["candidate_count"] != (
        episode_counts["accepted_candidate_count"]
        + episode_counts["rejected_candidate_count"]
    ):
        raise ValueError(f"candidate accounting is inconsistent: {sample}")
    if (
        episode_counts["action_attempt_error_count"]
        > episode_counts["action_attempt_count"]
        or episode_counts["action_attempt_count"]
        < episode_counts["candidate_count"]
    ):
        raise ValueError(f"action-attempt accounting is inconsistent: {sample}")

    stage_calls: Counter[str] = Counter()
    attempt_kinds: Counter[str] = Counter()
    visual_input_scopes: Counter[str] = Counter()
    stage_latency: defaultdict[str, float] = defaultdict(float)
    stage_response_chars: Counter[str] = Counter()
    diagnosis_types: Counter[str] = Counter()
    plans: Counter[str] = Counter()
    validation_error_count = 0
    regional_error_count = 0
    atomic_audit_records = 0
    atomic_audit_valid_records = 0
    audited_region_records = 0
    initial_environment_region_records = 0
    initial_atomic_audit_records = 0
    initial_audited_region_ids: list[str] | None = None
    verifier_invalid_steps = 0
    localization_invalid_steps = 0
    hard_gate_skipped_steps = 0
    schema_versions: set[str] = set()

    def accumulate_evidence(
        evidence: Mapping[str, Any], *, evidence_name: str
    ) -> tuple[Mapping[str, Any] | None, list[str], list[str]]:
        nonlocal validation_error_count, regional_error_count
        schema = evidence.get("schema_version")
        if not isinstance(schema, str) or not schema:
            raise ValueError(f"{evidence_name} lacks a schema version: {sample}")
        schema_versions.add(schema)
        verifier_valid = evidence.get("verifier_valid")
        localization_valid = evidence.get("localization_valid")
        if not isinstance(verifier_valid, bool) or not isinstance(
            localization_valid, bool
        ):
            raise ValueError(f"step {index} has malformed validity flags: {sample}")
        if not isinstance(verifier_valid, bool) or not isinstance(
            localization_valid, bool
        ):
            raise ValueError(f"{evidence_name} has malformed validity flags: {sample}")
        validation_errors = evidence.get("validation_errors")
        if not isinstance(validation_errors, list) or any(
            not isinstance(item, str) or not item for item in validation_errors
        ):
            raise ValueError(f"{evidence_name} validation_errors is malformed: {sample}")
        validation_error_count += len(validation_errors)
        trace = evidence.get("stage_trace")
        regional_errors = evidence.get("regional_validation_errors")
        if regional_errors is None and trace is None:
            regional_errors = []
        if not isinstance(regional_errors, list):
            raise ValueError(f"{evidence_name} regional errors are malformed: {sample}")
        regional_ids: list[str] = []
        for item in regional_errors:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("region_id"), str)
                or not item["region_id"]
                or not isinstance(item.get("error"), str)
                or not item["error"]
            ):
                raise ValueError(f"{evidence_name} has a malformed regional error: {sample}")
            regional_ids.append(item["region_id"])
        if len(regional_ids) != len(set(regional_ids)):
            raise ValueError(f"{evidence_name} repeats a regional error ID: {sample}")
        regional_error_count += len(regional_ids)
        backend_calls = evidence.get("backend_calls")
        if not isinstance(backend_calls, list):
            raise ValueError(f"{evidence_name} backend_calls is not a list: {sample}")
        for call_index, call in enumerate(backend_calls):
            if not isinstance(call, Mapping):
                raise ValueError(f"{evidence_name} has a malformed backend call: {sample}")
            backend = call.get("backend")
            stage = call.get("stage")
            attempt = call.get("attempt_kind")
            visual_input_scope = call.get("visual_input_scope")
            retry_count = call.get("network_retry_attempt_count")
            prompt_sha256 = call.get("prompt_sha256")
            if strict_p1_calls and (not isinstance(backend, str) or not backend):
                raise ValueError(f"backend identity is malformed: {evidence_name}")
            if not isinstance(stage, str) or stage not in ALLOWED_BACKEND_STAGES:
                raise ValueError(f"backend stage is malformed: {evidence_name}")
            if not isinstance(attempt, str) or attempt not in ALLOWED_ATTEMPT_KINDS:
                raise ValueError(f"backend attempt is malformed: {evidence_name}")
            if strict_p1_calls and (
                isinstance(retry_count, bool)
                or not isinstance(retry_count, int)
                or retry_count < 0
            ):
                raise ValueError(f"backend retry counter is malformed: {evidence_name}")
            if strict_p1_calls and (
                not isinstance(prompt_sha256, str)
                or SHA256_RE.fullmatch(prompt_sha256) is None
            ):
                raise ValueError(f"backend prompt hash is malformed: {evidence_name}")
            if visual_input_scope is None and not strict_p1_calls:
                visual_input_scope = "unreported"
            if (
                not isinstance(visual_input_scope, str)
                or visual_input_scope not in (
                    ALLOWED_VISUAL_INPUT_SCOPES - {"unreported"}
                    if strict_p1_calls
                    else ALLOWED_VISUAL_INPUT_SCOPES
                )
            ):
                raise ValueError(f"backend visual input scope is unsupported: {evidence_name}")
            latency = finite_nonnegative(
                call.get("latency_seconds"),
                name=f"{evidence_name}.call_{call_index}.{stage}.latency",
            )
            if "raw_response" not in call or not isinstance(call["raw_response"], str):
                raise ValueError(f"backend raw_response is malformed: {evidence_name}")
            if (
                visual_input_scope == "authoritative_pair_rgb_only_evidence_repair"
                and (stage != "audit" or attempt != "repair")
            ):
                raise ValueError(
                    "evidence-repair visual input scope requires an audit repair"
                )
            if visual_input_scope == "authoritative_pair_rgb_only_scan" and stage != "audit":
                raise ValueError("scan visual input scope requires an audit call")
            stage_calls[stage] += 1
            attempt_kinds[attempt] += 1
            visual_input_scopes[visual_input_scope] += 1
            stage_latency[stage] += latency
            stage_response_chars[stage] += len(call["raw_response"])
        return trace if isinstance(trace, Mapping) else None, regional_ids, validation_errors

    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"trajectory step {index} is not an object: {sample}")
        execution = step.get("execution")
        evidence = (
            execution.get("verifier_evidence")
            if isinstance(execution, Mapping)
            else None
        )
        if not isinstance(evidence, Mapping):
            raise ValueError(f"step {index} lacks verifier evidence: {sample}")
        evidence_type = evidence.get("type")
        if evidence_type == "runtime_hard_gate_precheck":
            if evidence.get("verifier_skipped") is not True:
                raise ValueError(
                    f"step {index} hard-gate evidence is not marked skipped: {sample}"
                )
            if evidence.get("gt_available") is not False:
                raise ValueError(
                    f"step {index} hard-gate evidence has an invalid GT flag: {sample}"
                )
            reasons = evidence.get("rejection_reasons")
            if not isinstance(reasons, list) or any(
                not isinstance(item, str) or not item for item in reasons
            ):
                raise ValueError(
                    f"step {index} hard-gate rejection reasons are malformed: {sample}"
                )
            if not reasons:
                raise ValueError(
                    f"step {index} hard-gate evidence has no rejection reason: {sample}"
                )
            hard_gate_skipped_steps += 1
            continue
        if evidence_type != "staged_qwen_verifier":
            raise ValueError(
                f"step {index} has an unsupported verifier evidence type: {sample}"
            )
        verifier_valid = evidence.get("verifier_valid")
        localization_valid = evidence.get("localization_valid")
        verifier_invalid_steps += int(not verifier_valid)
        localization_invalid_steps += int(not localization_valid)
        terminal_review = evidence.get("terminal_review")
        if "terminal_review" in evidence:
            if not isinstance(terminal_review, Mapping):
                raise ValueError(f"step {index} terminal_review is malformed: {sample}")
            round_one = terminal_review.get("round_one_evidence")
            if not isinstance(round_one, Mapping):
                raise ValueError(f"step {index} lacks round-one evidence: {sample}")
            trace, regional_ids, validation_errors = accumulate_evidence(
                round_one,
                evidence_name=f"{sample}.step_{index}.round_one_evidence",
            )
            round_two = terminal_review.get("round_two_evidence")
            round_two_invoked = terminal_review.get("round_two_invoked")
            if not isinstance(round_two_invoked, bool):
                raise ValueError(f"step {index} round-two flag is malformed: {sample}")
            if round_two_invoked:
                if not isinstance(round_two, Mapping):
                    raise ValueError(f"step {index} lacks invoked round-two evidence: {sample}")
                accumulate_evidence(
                    round_two,
                    evidence_name=f"{sample}.step_{index}.round_two_evidence",
                )
                final_source = round_two
            else:
                if round_two is not None:
                    raise ValueError(f"step {index} has uninvoked round-two evidence: {sample}")
                final_source = round_one
            for field in (
                "schema_version",
                "verifier_valid",
                "localization_valid",
                "validation_errors",
                "regional_validation_errors",
                "stage_trace",
                "backend_calls",
            ):
                if evidence.get(field) != final_source.get(field):
                    raise ValueError(
                        f"step {index} final evidence {field} differs from terminal-review tail: {sample}"
                    )
        else:
            trace, regional_ids, validation_errors = accumulate_evidence(
                evidence,
                evidence_name=f"{sample}.step_{index}",
            )
        backend_calls = evidence.get("backend_calls")
        if not isinstance(backend_calls, list):
            raise ValueError(f"step {index} backend_calls is not a list: {sample}")

        if not isinstance(trace, Mapping):
            decision_mode = evidence.get("decision_mode")
            if regional_ids:
                raise ValueError(
                    f"step {index} has regional errors without a trace: {sample}"
                )
            if evidence.get("verifier_valid") is False:
                if (
                    evidence.get("localization_valid") is not False
                    or not validation_errors
                    or decision_mode is not None
                ):
                    raise ValueError(
                        f"step {index} has malformed invalid no-trace evidence: {sample}"
                    )
                continue
            if decision_mode not in PROGRAMMATIC_NO_TRACE_MODES:
                raise ValueError(
                    f"step {index} lacks a trace without a programmatic mode: {sample}"
                )
            if initial_audited_region_ids is None:
                raise ValueError(
                    f"step {index} programmatic no-trace evidence precedes initial audit: {sample}"
                )
            if validation_errors or backend_calls:
                raise ValueError(
                    f"step {index} programmatic no-trace evidence contains model activity or errors: {sample}"
                )
            if decision_mode == "programmatic_identical_state":
                if "replan" in evidence:
                    raise ValueError(
                        f"step {index} identical-state evidence contains a replan: {sample}"
                    )
            else:
                replan = evidence.get("replan")
                if not isinstance(replan, Mapping):
                    raise ValueError(
                        f"step {index} rollback evidence lacks replan metadata: {sample}"
                    )
                selected_region_id = replan.get("selected_region_id")
                if decision_mode == "programmatic_alternate_region_after_rollback":
                    if evidence.get("localization_valid") is not True:
                        raise ValueError(
                            f"step {index} alternate rollback has invalid localization flag: {sample}"
                        )
                    if not isinstance(selected_region_id, str) or not selected_region_id:
                        raise ValueError(
                            f"step {index} alternate rollback lacks a selected region: {sample}"
                        )
                    if selected_region_id not in initial_audited_region_ids:
                        raise ValueError(
                            f"step {index} alternate rollback selected an unaudited region: {sample}"
                        )
                elif (
                    evidence.get("localization_valid") is not False
                    or selected_region_id is not None
                ):
                    raise ValueError(
                        f"step {index} exhausted rollback unexpectedly selected a region: {sample}"
                    )
                rejected_action_count = nonnegative_int(
                    replan.get("rejected_action_count"),
                    name=f"{sample}.step_{index}.rejected_action_count",
                )
                if rejected_action_count < 1:
                    raise ValueError(
                        f"step {index} rollback has no rejected action: {sample}"
                    )
                rejection_reasons = replan.get("rejection_reasons")
                if (
                    not rejection_reasons
                    or not isinstance(rejection_reasons, list)
                    or any(
                        not isinstance(item, str) or not item
                        for item in rejection_reasons
                    )
                ):
                    raise ValueError(
                        f"step {index} rollback rejection reasons are malformed: {sample}"
                    )
            continue
        mode = trace.get("mode")
        if mode not in {"initial", "candidate"}:
            raise ValueError(f"step {index} has an invalid trace mode: {sample}")
        primary_records = record_ids(
            trace.get("evidence"), name=f"{sample}.step_{index}.evidence"
        )
        primary_audited = string_ids(
            trace.get(
                "audited_region_ids",
                trace.get("selected_region_ids"),
            ),
            name=f"{sample}.step_{index}.audited_region_ids",
        )
        if set(primary_audited) != set(primary_records):
            raise ValueError(f"step {index} primary audit coverage differs: {sample}")
        if mode == "initial" and not primary_records:
            raise ValueError(f"step {index} initial trace has no audit regions: {sample}")
        audited_region_records += len(primary_records)

        def parse_diagnoses(value: Any, *, name: str) -> list[str]:
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            ids: list[str] = []
            for diagnosis in value:
                if (
                    not isinstance(diagnosis, Mapping)
                    or not isinstance(diagnosis.get("region_id"), str)
                    or not diagnosis["region_id"]
                    or not isinstance(diagnosis.get("error_type"), str)
                    or not diagnosis["error_type"]
                ):
                    raise ValueError(f"{name} contains a malformed diagnosis")
                ids.append(diagnosis["region_id"])
                diagnosis_types[diagnosis["error_type"]] += 1
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} contains duplicate region IDs")
            return ids

        def parse_judgments(value: Any, *, name: str) -> list[str]:
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            ids: list[str] = []
            for judgment in value:
                if (
                    not isinstance(judgment, Mapping)
                    or not isinstance(judgment.get("region_id"), str)
                    or not judgment["region_id"]
                ):
                    raise ValueError(f"{name} contains a malformed judgment")
                ids.append(judgment["region_id"])
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} contains duplicate region IDs")
            return ids

        judgment_ids = parse_judgments(
            trace.get("judgments"), name=f"{sample}.step_{index}.judgments"
        )
        if set(judgment_ids) != set(primary_records):
            raise ValueError(f"step {index} judgment coverage differs: {sample}")

        diagnosis_ids = parse_diagnoses(
            trace.get("diagnoses"), name=f"{sample}.step_{index}.diagnoses"
        )
        if mode == "initial" and set(diagnosis_ids) != set(primary_records):
            raise ValueError(f"step {index} diagnosis coverage differs: {sample}")
        if mode == "candidate" and diagnosis_ids:
            raise ValueError(f"step {index} candidate trace has diagnoses: {sample}")
        if mode == "initial":
            if initial_audited_region_ids is not None:
                raise ValueError(f"sample has more than one initial trace: {sample}")
            initial_audited_region_ids = sorted(primary_audited)
            initial_environment_region_records += len(primary_records)
            initial_atomic_audit_records += len(diagnosis_ids)

        replan_records = record_ids(
            trace.get("replan_evidence"),
            name=f"{sample}.step_{index}.replan_evidence",
        )
        replan_audited = string_ids(
            trace.get(
                "replan_audited_region_ids",
                trace.get("replan_selected_region_ids"),
            ),
            name=f"{sample}.step_{index}.replan_audited_region_ids",
        )
        if set(replan_audited) != set(replan_records):
            raise ValueError(f"step {index} replan audit coverage differs: {sample}")
        audited_region_records += len(replan_records)
        replan_diagnosis_ids = parse_diagnoses(
            trace.get("replan_diagnoses"),
            name=f"{sample}.step_{index}.replan_diagnoses",
        )
        if set(replan_diagnosis_ids) != set(replan_records):
            raise ValueError(f"step {index} replan diagnosis coverage differs: {sample}")
        replan_judgment_ids = parse_judgments(
            trace.get("replan_judgments"),
            name=f"{sample}.step_{index}.replan_judgments",
        )
        if set(replan_judgment_ids) != set(replan_records):
            raise ValueError(
                f"step {index} replan judgment coverage differs: {sample}"
            )

        all_diagnosis_ids = diagnosis_ids + replan_diagnosis_ids
        if not set(regional_ids) <= set(all_diagnosis_ids):
            raise ValueError(
                f"step {index} regional errors lack diagnoses: {sample}"
            )
        atomic_audit_records += len(all_diagnosis_ids)
        atomic_audit_valid_records += len(all_diagnosis_ids) - len(regional_ids)
        plan = trace.get("plan")
        if isinstance(plan, Mapping):
            action = plan.get("action")
            if not isinstance(action, str) or not action:
                raise ValueError(f"step {index} has a malformed plan: {sample}")
            plans[action] += 1
        elif plan is None:
            plans["none"] += 1
        else:
            raise ValueError(f"step {index} has a malformed plan: {sample}")

    backend_call_count = sum(stage_calls.values())
    total_latency = sum(stage_latency.values())
    total_response_chars = sum(stage_response_chars.values())
    for stage in REPORTED_STAGES:
        stage_calls.setdefault(stage, 0)
        stage_latency.setdefault(stage, 0.0)
        stage_response_chars.setdefault(stage, 0)
    for attempt_kind in REPORTED_ATTEMPT_KINDS:
        attempt_kinds.setdefault(attempt_kind, 0)
    if initial_audited_region_ids is None:
        raise ValueError(f"sample lacks an initial staged-verifier trace: {sample}")
    if stage_calls["audit"] < atomic_audit_records:
        raise ValueError(
            f"audit call count is below atomic audit records: {sample}"
        )
    if elapsed + 0.01 < total_latency:
        raise ValueError(
            f"episode elapsed seconds are below summed backend latency: {sample}"
        )
    return {
        "trajectory_steps": len(steps),
        "schema_versions": sorted(schema_versions),
        "backend_call_count": backend_call_count,
        "stage_calls": dict(sorted(stage_calls.items())),
        "attempt_kinds": dict(sorted(attempt_kinds.items())),
        "visual_input_scopes": dict(sorted(visual_input_scopes.items())),
        "stage_latency_seconds": {
            key: round(value, 6) for key, value in sorted(stage_latency.items())
        },
        "backend_latency_seconds": round(total_latency, 6),
        "stage_raw_response_chars": dict(sorted(stage_response_chars.items())),
        "raw_response_chars": total_response_chars,
        "validation_error_count": validation_error_count,
        "regional_validation_error_count": regional_error_count,
        "atomic_audit_records": atomic_audit_records,
        "atomic_audit_valid_records": atomic_audit_valid_records,
        "audited_region_records": audited_region_records,
        "initial_environment_region_records": initial_environment_region_records,
        "initial_atomic_audit_records": initial_atomic_audit_records,
        "initial_audited_region_ids": initial_audited_region_ids,
        "verifier_invalid_steps": verifier_invalid_steps,
        "localization_invalid_steps": localization_invalid_steps,
        "hard_gate_skipped_steps": hard_gate_skipped_steps,
        "diagnosis_types": dict(sorted(diagnosis_types.items())),
        "plans": dict(sorted(plans.items())),
        "episode_outcome": outcome,
        "stop_reason": stop_reason,
        "episode_elapsed_seconds": elapsed,
        "action_attempt_error_count": episode_counts["action_attempt_error_count"],
        "tool_action_count": episode_counts["tool_action_count"],
    }


def add_counters(target: Counter[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target[key] += nonnegative_int(value, name=f"counter.{key}")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    samples = [str(item) for item in args.expected_samples]
    expected_evaluation_id, canonical_samples = EVALUATION_PROFILES[
        args.evaluation_profile
    ]
    if args.evaluation_id != expected_evaluation_id:
        raise ValueError(
            "evaluation ID does not match evaluation profile: "
            f"{args.evaluation_id!r} != {expected_evaluation_id!r}"
        )
    if samples != list(canonical_samples):
        raise ValueError(
            "expected samples do not match evaluation profile: "
            f"{samples!r} != {list(canonical_samples)!r}"
        )
    if len(samples) != len(set(samples)):
        raise ValueError("expected sample IDs must be unique")
    experiment_arm = str(args.experiment_arm)
    harness_protocol_id = str(getattr(args, "harness_protocol_id", ""))
    expected_harness_protocol = P1_ARM_PROTOCOLS.get(experiment_arm)
    if expected_harness_protocol is not None:
        if harness_protocol_id != expected_harness_protocol:
            raise ValueError("P1 experiment arm and harness protocol ID differ")
        if str(args.protocol_version) != P1_MODEL_IO_PROTOCOL:
            raise ValueError("P1 model-I/O protocol must be v12.1")
        if args.evaluation_profile not in {"p1_smoke1", "e2e3"}:
            raise ValueError("P1 H0/H1 evaluation profile must be p1_smoke1 or e2e3")
    elif harness_protocol_id in set(P1_ARM_PROTOCOLS.values()):
        raise ValueError("P1 harness protocol ID is paired with the wrong experiment arm")
    elif args.evaluation_profile == "p1_smoke1":
        raise ValueError("p1_smoke1 requires a frozen P1 H0/H1 experiment arm")
    rollout = read_object(run_dir / "rollout_complete.json")
    if rollout.get("execution_status") != "completed" or rollout.get("gt_loaded") is not False:
        raise ValueError("rollout marker must be completed and GT-free")
    rollout_samples = rollout.get("samples")
    if (
        not isinstance(rollout_samples, list)
        or rollout_samples != sorted(samples)
    ):
        raise ValueError(
            "rollout sample identity/canonical order differs from expected samples"
        )

    per_sample = {
        sample: summarize_sample(
            run_dir,
            sample,
            strict_p1_calls=expected_harness_protocol is not None,
        )
        for sample in samples
    }
    count_fields = (
        "stage_calls",
        "attempt_kinds",
        "visual_input_scopes",
        "stage_raw_response_chars",
        "diagnosis_types",
        "plans",
    )
    counters = {field: Counter() for field in count_fields}
    latency: defaultdict[str, float] = defaultdict(float)
    scalar_fields = (
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
    scalar_totals = {field: 0 for field in scalar_fields}
    elapsed = 0.0
    outcomes: Counter[str] = Counter()
    schema_versions: set[str] = set()
    for result in per_sample.values():
        for field in count_fields:
            add_counters(counters[field], result[field])
        for stage, value in result["stage_latency_seconds"].items():
            latency[stage] += float(value)
        for field in scalar_fields:
            scalar_totals[field] += nonnegative_int(
                result.get(field), name=f"aggregate.{field}"
            )
        elapsed += finite_nonnegative(
            result.get("episode_elapsed_seconds"),
            name="aggregate.episode_elapsed_seconds",
        )
        outcomes[result["episode_outcome"]] += 1
        schema_versions.update(result["schema_versions"])

    totals: dict[str, Any] = dict(sorted(scalar_totals.items()))
    totals.update(
        {
            field: dict(sorted(counter.items()))
            for field, counter in counters.items()
        }
    )
    totals["stage_latency_seconds"] = {
        key: round(value, 6) for key, value in sorted(latency.items())
    }
    totals["backend_latency_seconds"] = round(sum(latency.values()), 6)
    totals["episode_elapsed_seconds"] = round(elapsed, 6)
    totals["episode_outcomes"] = dict(sorted(outcomes.items()))
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "evaluation_profile": str(args.evaluation_profile),
        "evaluation_id": str(args.evaluation_id),
        "experiment_arm": experiment_arm,
        "protocol_version": str(args.protocol_version),
        "harness_protocol_id": harness_protocol_id,
        "job_id": str(args.job_id),
        "runtime_commit": str(args.runtime_commit),
        "harness_commit": str(args.harness_commit),
        "sample_order": samples,
        "schema_versions": sorted(schema_versions),
        "totals": totals,
        "per_sample": per_sample,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize(args)
    except ValueError as error:
        print(f"staged_verifier_analysis: {error}", file=sys.stderr)
        return 3
    atomic_write(args.output, summary)
    totals = summary["totals"]
    print(
        "protocol=completed;"
        f"calls={totals.get('backend_call_count', 0)};"
        f"audit={totals.get('stage_calls', {}).get('audit', 0)};"
        f"repair={totals.get('attempt_kinds', {}).get('repair', 0)};"
        f"regional_errors={totals.get('regional_validation_error_count', 0)};"
        f"latency={totals.get('backend_latency_seconds', 0.0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
