#!/usr/bin/env python3
"""Validate a hash-bound Qwen3-VL P1 evaluation receipt v2.

The receipt is an immutable index over artifacts produced by one H0 or H1
evaluation.  Validation is deliberately fail closed: completion is accepted
only when every required artifact exists, matches its recorded SHA-256, and
agrees on the frozen arm, protocol, sample order, commit, Slurm identity,
budget accounting, GT boundary, safety boundaries and tool-failure gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "qwen3vl_evaluation_receipt_v2"
SUMMARY_SCHEMA_ID = "qwen3vl_e2e_summary_v2"
P1_AUDIT_SCHEMA_ID = "qwen3vl_p1_harness_audit_v1"
FACTOR_ID = "terminal_review_scheduler_v1"
MODEL_IO_PROTOCOL = "v12.1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

PROFILES = {
    "p1_smoke1": ("DEV-SMOKE-1-H1-V1", ("test_1_1",)),
    "e2e3": (
        "DEV-CORE-3-V1",
        ("test_20_15", "test_1_1", "test_50_8"),
    ),
}
ARM_CONTRACTS = {
    "h0": {
        "experiment_arm_id": "v12_1-h0-legacy-single-round-base-0802",
        "harness_protocol_id": (
            "qwen3vl_harness_h0_legacy_single_round_v1"
        ),
        "terminal_review": False,
        "max_verifier_rounds": 1,
    },
    "h1": {
        "experiment_arm_id": (
            "v12_1-h1-gtfree-terminal-review-base-0802"
        ),
        "harness_protocol_id": (
            "qwen3vl_harness_h1_gtfree_terminal_review_v1"
        ),
        "terminal_review": True,
        "max_verifier_rounds": 2,
    },
}
CONFIGURED_BUDGETS = {
    "max_stage_attempts_per_round": 2,
    "max_environment_steps": 3,
    "effective_network_retry_limit": 0,
    "effective_tool_attempts_per_authorized_action": 1,
}
BUDGET_COUNTERS = (
    "verifier_round_count",
    "stage_initial_attempt_count",
    "schema_repair_attempt_count",
    "network_retry_attempt_count",
    "tool_action_attempt_count",
    "tool_failure_count",
    "environment_step_count",
)
EPISODE_TOTALS = (
    "candidates",
    "accepted",
    "rejected",
    "action_attempts",
    "action_attempt_errors",
    "fallback_actions",
    "tool_actions",
    "successful_tool_actions",
    "tool_failure_count",
)
ARTIFACT_NAMES = (
    "baseline_summary",
    "protocol_summary",
    "p1_harness_audit",
    "resolved_run_config",
    "p1_amendment",
    "evaluation_input_manifest",
    "runtime_input_manifest",
    "run_manifest",
    "launch_command",
    "rollout_marker",
    "offline_scorer_input",
    "submission_receipt",
    "status_receipt",
    "summary_validation",
)
JSON_ARTIFACTS = frozenset(
    {
        "baseline_summary",
        "protocol_summary",
        "p1_harness_audit",
        "resolved_run_config",
        "p1_amendment",
        "evaluation_input_manifest",
        "runtime_input_manifest",
        "rollout_marker",
        "offline_scorer_input",
    }
)
SAFETY_BOUNDARIES = (
    "parser",
    "geometry_authorization",
    "locality_hard_gate",
    "candidate_benefit_without_harm",
    "rollback",
)


class ReceiptValidationError(ValueError):
    """Raised when receipt evidence does not satisfy the frozen contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def _object(value: Any, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptValidationError(f"{source} must be a JSON object")
    return value


def _string(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptValidationError(f"{source} must be a non-empty string")
    return value


def _integer(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptValidationError(
            f"{source} must be a non-negative integer"
        )
    return value


def _frozen_hashes(value: Any, *, source: str) -> dict[str, str]:
    mapping = _object(value, source=source)
    if not mapping:
        raise ReceiptValidationError(f"{source} must not be empty")
    result: dict[str, str] = {}
    for name, digest in mapping.items():
        key = _string(name, source=f"{source} key")
        if key in result:
            raise ReceiptValidationError(f"{source} contains duplicate names")
        observed = _string(digest, source=f"{source}.{key}")
        if SHA256_RE.fullmatch(observed) is None:
            raise ReceiptValidationError(f"{source}.{key} is malformed")
        result[key] = observed
    return result


def _config_hash_bindings(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    bindings = _object(
        config.get("freeze_commit_hash_bindings"),
        source="resolved config freeze-commit hash bindings",
    )
    if bindings.get("status") != "frozen":
        raise ReceiptValidationError(
            "resolved config freeze-commit hash bindings are not frozen"
        )
    return (
        _frozen_hashes(
            bindings.get("model_files_sha256"),
            source="resolved config model_files_sha256",
        ),
        _frozen_hashes(
            bindings.get("tool_files_sha256"),
            source="resolved config tool_files_sha256",
        ),
        _frozen_hashes(
            bindings.get("worker_sha256"),
            source="resolved config worker_sha256",
        ),
    )


def _amendment_model_hashes(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        value = {"model.safetensors": value}
    return _frozen_hashes(value, source="P1 amendment model_hash")


def _amendment_worker_hashes(value: Any) -> dict[str, str]:
    result = _frozen_hashes(value, source="P1 amendment worker_sha256")
    if set(result) != {
        "cpu_fixtures",
        "evaluation",
        "dependency_guard",
    }:
        raise ReceiptValidationError(
            "P1 amendment worker_sha256 keys differ from canonical mapping"
        )
    return result


def _string_list(value: Any, *, source: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ReceiptValidationError(
            f"{source} must be a list of non-empty strings"
        )
    if len(value) != len(set(value)):
        raise ReceiptValidationError(f"{source} contains duplicates")
    return list(value)


def _read_json(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReceiptValidationError(f"{source} is missing: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptValidationError(
            f"{source} is not readable JSON: {path}: {error}"
        ) from error
    return dict(_object(payload, source=source))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ReceiptValidationError(
            f"cannot hash required artifact {path}: {error}"
        ) from error
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReceiptValidationError(
            f"cannot canonicalize JSON evidence: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _resolve_bound_path(value: Any, *, source: str) -> Path:
    path_text = _string(value, source=source)
    path = Path(path_text)
    if not path.is_absolute():
        raise ReceiptValidationError(f"{source} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReceiptValidationError(
            f"{source} does not resolve to an existing artifact: {path}"
        ) from error
    if not resolved.is_file():
        raise ReceiptValidationError(f"{source} must resolve to a file")
    return resolved


def _artifact_bindings(
    value: Any,
) -> tuple[dict[str, Path], dict[str, Any]]:
    bindings = _object(value, source="receipt.artifacts")
    if list(bindings) != list(ARTIFACT_NAMES):
        raise ReceiptValidationError(
            "receipt.artifacts must use the frozen artifact order"
        )
    paths: dict[str, Path] = {}
    payloads: dict[str, Any] = {}
    seen_paths: set[Path] = set()
    for name in ARTIFACT_NAMES:
        source = f"receipt.artifacts.{name}"
        binding = _object(bindings[name], source=source)
        if list(binding) != ["path", "sha256", "size_bytes"]:
            raise ReceiptValidationError(
                f"{source} must contain ordered path, sha256 and size_bytes"
            )
        path = _resolve_bound_path(binding.get("path"), source=f"{source}.path")
        expected_hash = _string(binding.get("sha256"), source=f"{source}.sha256")
        if SHA256_RE.fullmatch(expected_hash) is None:
            raise ReceiptValidationError(f"{source}.sha256 is malformed")
        if _sha256(path) != expected_hash:
            raise ReceiptValidationError(f"{source}.sha256 does not match its file")
        size = _integer(binding.get("size_bytes"), source=f"{source}.size_bytes")
        if size != path.stat().st_size:
            raise ReceiptValidationError(
                f"{source}.size_bytes does not match its file"
            )
        if path in seen_paths:
            raise ReceiptValidationError(
                "independent receipt artifacts must not alias one file"
            )
        seen_paths.add(path)
        paths[name] = path
        if name in JSON_ARTIFACTS:
            payloads[name] = _read_json(path, source=source)
    return paths, payloads


def _parse_key_value(path: Path, *, source: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ReceiptValidationError(
            f"{source} is not readable text: {path}: {error}"
        ) from error
    result: dict[str, str] = {}
    for index, line in enumerate(lines, 1):
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ReceiptValidationError(
                f"{source} line {index} is not key=value"
            )
        key, raw = line.split("=", 1)
        if not key or key in result:
            raise ReceiptValidationError(
                f"{source} line {index} has an empty or duplicate key"
            )
        result[key] = raw
    return result


def _identity(value: Any) -> dict[str, Any]:
    identity = dict(_object(value, source="receipt.identity"))
    required = {
        "factor_id",
        "arm",
        "experiment_arm_id",
        "harness_protocol_id",
        "model_io_protocol",
        "evaluation_profile",
        "evaluation_id",
        "sample_order",
        "runtime_commit",
        "harness_commit",
        "job_id",
        "model_state",
    }
    if set(identity) != required:
        raise ReceiptValidationError(
            "receipt.identity keys differ from the frozen contract"
        )
    if identity.get("factor_id") != FACTOR_ID:
        raise ReceiptValidationError("receipt.identity.factor_id is not frozen")
    arm = identity.get("arm")
    if arm not in ARM_CONTRACTS:
        raise ReceiptValidationError("receipt.identity.arm must be h0 or h1")
    contract = ARM_CONTRACTS[str(arm)]
    for key in ("experiment_arm_id", "harness_protocol_id"):
        if identity.get(key) != contract[key]:
            raise ReceiptValidationError(f"receipt.identity.{key} is not frozen")
    if identity.get("model_io_protocol") != MODEL_IO_PROTOCOL:
        raise ReceiptValidationError(
            "receipt.identity.model_io_protocol is not frozen"
        )
    profile = identity.get("evaluation_profile")
    if profile not in PROFILES:
        raise ReceiptValidationError(
            "receipt.identity.evaluation_profile is not a P1 profile"
        )
    evaluation_id, samples = PROFILES[str(profile)]
    if identity.get("evaluation_id") != evaluation_id:
        raise ReceiptValidationError(
            "receipt identity evaluation ID/profile mismatch"
        )
    if identity.get("sample_order") != list(samples):
        raise ReceiptValidationError(
            "receipt identity sample order/profile mismatch"
        )
    for key in ("runtime_commit", "harness_commit"):
        commit = _string(identity.get(key), source=f"receipt.identity.{key}")
        if GIT_SHA_RE.fullmatch(commit) is None:
            raise ReceiptValidationError(f"receipt.identity.{key} is malformed")
    if identity["runtime_commit"] != identity["harness_commit"]:
        raise ReceiptValidationError(
            "runtime and harness commit must be the same exact implementation"
        )
    _string(identity.get("job_id"), source="receipt.identity.job_id")
    if identity.get("model_state") != "base":
        raise ReceiptValidationError("P1 H0/H1 receipt model_state must be base")
    return identity


def _validate_provenance(
    value: Any, identity: Mapping[str, Any]
) -> dict[str, Any]:
    provenance = dict(_object(value, source="receipt.provenance"))
    required = {
        "implementation_commit",
        "freeze_commit",
        "run_worktree",
        "git_dirty",
        "git_worktree_sha256",
        "model_path",
        "model_sha256",
        "tool_sha256",
        "worker_sha256",
        "evaluation_input_manifest_canonical_sha256",
    }
    if set(provenance) != required:
        raise ReceiptValidationError(
            "receipt.provenance keys differ from the frozen contract"
        )
    if provenance.get("freeze_commit") != identity["runtime_commit"]:
        raise ReceiptValidationError(
            "receipt freeze commit differs from runtime commit"
        )
    implementation_commit = _string(
        provenance.get("implementation_commit"),
        source="provenance.implementation_commit",
    )
    if GIT_SHA_RE.fullmatch(implementation_commit) is None:
        raise ReceiptValidationError("receipt implementation commit is malformed")
    if implementation_commit == provenance["freeze_commit"]:
        raise ReceiptValidationError(
            "implementation and freeze commits must remain distinct"
        )
    worktree = Path(
        _string(provenance.get("run_worktree"), source="run_worktree")
    )
    if not worktree.is_absolute() or not worktree.is_dir():
        raise ReceiptValidationError("receipt run_worktree must be an existing path")
    if provenance.get("git_dirty") is not False:
        raise ReceiptValidationError("complete receipt requires git_dirty=false")
    for key in (
        "git_worktree_sha256",
        "model_sha256",
        "worker_sha256",
        "evaluation_input_manifest_canonical_sha256",
    ):
        digest = _string(provenance.get(key), source=f"provenance.{key}")
        if SHA256_RE.fullmatch(digest) is None:
            raise ReceiptValidationError(f"provenance.{key} is malformed")
    try:
        head = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD^"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_bytes = subprocess.run(
            [
                "git",
                "-C",
                str(worktree),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReceiptValidationError(
            "receipt run_worktree is not independently Git-verifiable"
        ) from error
    if head != identity["runtime_commit"]:
        raise ReceiptValidationError("live run_worktree HEAD differs from receipt")
    if parent != provenance["implementation_commit"]:
        raise ReceiptValidationError(
            "live freeze commit is not a direct child of the receipt implementation commit"
        )
    if status_bytes:
        raise ReceiptValidationError("live run_worktree is no longer clean")
    if hashlib.sha256(status_bytes).hexdigest() != provenance[
        "git_worktree_sha256"
    ]:
        raise ReceiptValidationError(
            "provenance.git_worktree_sha256 differs from live clean status"
        )
    model_path = Path(
        _string(provenance.get("model_path"), source="provenance.model_path")
    )
    if not model_path.is_absolute() or not model_path.is_dir():
        raise ReceiptValidationError("bound base model path is missing")
    tools = _object(provenance.get("tool_sha256"), source="provenance.tool_sha256")
    if not tools:
        raise ReceiptValidationError("provenance.tool_sha256 must not be empty")
    for name, digest in tools.items():
        _string(name, source="provenance.tool_sha256 key")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ReceiptValidationError(
                f"provenance.tool_sha256.{name} is malformed"
            )
    return provenance


def _validate_budget_accounting(
    value: Any, identity: Mapping[str, Any]
) -> tuple[
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, int],
    dict[str, int],
]:
    budget = _object(value, source="receipt.budget_accounting")
    required = ["configured", "terminal_review_scheduler", "episode_totals"]
    if list(budget) != required:
        raise ReceiptValidationError(
            "receipt.budget_accounting must keep configured, scheduler and "
            "episode totals in the frozen order"
        )
    configured = dict(
        _object(budget.get("configured"), source="budget.configured")
    )
    expected_configured = {
        **CONFIGURED_BUDGETS,
        "max_verifier_rounds": ARM_CONTRACTS[str(identity["arm"])][
            "max_verifier_rounds"
        ],
    }
    if configured != expected_configured:
        raise ReceiptValidationError("configured budgets are not frozen")
    scheduler = _object(
        budget.get("terminal_review_scheduler"),
        source="budget.terminal_review_scheduler",
    )
    if list(scheduler) != ["totals", "per_sample"]:
        raise ReceiptValidationError(
            "terminal-review scheduler accounting must contain totals and per_sample"
        )
    totals = _counter_record(scheduler.get("totals"), source="scheduler.totals")
    per_sample_raw = _object(
        scheduler.get("per_sample"), source="scheduler.per_sample"
    )
    samples = list(identity["sample_order"])
    if list(per_sample_raw) != samples:
        raise ReceiptValidationError("budget per-sample order differs from identity")
    reconstructed = {key: 0 for key in BUDGET_COUNTERS}
    normalized_samples: dict[str, dict[str, int]] = {}
    max_rounds = expected_configured["max_verifier_rounds"]
    max_steps = expected_configured["max_environment_steps"]
    for sample in samples:
        record = _counter_record(
            per_sample_raw[sample], source=f"budget.per_sample.{sample}"
        )
        if record["verifier_round_count"] > max_rounds:
            raise ReceiptValidationError(
                f"budget.per_sample.{sample} exceeds verifier round budget"
            )
        if record["environment_step_count"] > max_steps:
            raise ReceiptValidationError(
                f"budget.per_sample.{sample} exceeds environment step budget"
            )
        if record["network_retry_attempt_count"] != 0:
            raise ReceiptValidationError("P1 local network retry budget is zero")
        if record["schema_repair_attempt_count"] > record[
            "stage_initial_attempt_count"
        ]:
            raise ReceiptValidationError(
                f"budget.per_sample.{sample} has excess repairs"
            )
        if record["tool_failure_count"] > record["tool_action_attempt_count"]:
            raise ReceiptValidationError(
                f"budget.per_sample.{sample} has excess tool failures"
            )
        for key in BUDGET_COUNTERS:
            reconstructed[key] += record[key]
        normalized_samples[sample] = record
    if reconstructed != totals:
        raise ReceiptValidationError(
            "budget consumed totals do not sum from per-sample records"
        )
    episode_raw = _object(
        budget.get("episode_totals"), source="budget.episode_totals"
    )
    if set(episode_raw) != set(EPISODE_TOTALS):
        raise ReceiptValidationError(
            "episode totals differ from the frozen aggregate fields"
        )
    episode = {
        key: _integer(episode_raw[key], source=f"episode_totals.{key}")
        for key in EPISODE_TOTALS
    }
    if episode["candidates"] != episode["accepted"] + episode["rejected"]:
        raise ReceiptValidationError("episode candidate totals are inconsistent")
    if episode["action_attempt_errors"] > episode["action_attempts"]:
        raise ReceiptValidationError("episode action-attempt totals are inconsistent")
    if episode["successful_tool_actions"] != episode["tool_actions"]:
        raise ReceiptValidationError("episode tool report totals are inconsistent")
    return totals, normalized_samples, episode, expected_configured


def _counter_record(value: Any, *, source: str) -> dict[str, int]:
    record = _object(value, source=source)
    if set(record) != set(BUDGET_COUNTERS):
        raise ReceiptValidationError(f"{source} keys differ from frozen counters")
    return {
        key: _integer(record[key], source=f"{source}.{key}")
        for key in BUDGET_COUNTERS
    }


def _validate_gates(
    value: Any,
    identity: Mapping[str, Any],
    scheduler_totals: Mapping[str, int],
    episode_totals: Mapping[str, int],
) -> None:
    gates = _object(value, source="receipt.gates")
    expected = {
        "gt_loaded_before_rollout_complete": False,
        "pre_rollout_gt_access_count": 0,
        "safety_boundary_relaxation_count": 0,
        "requested_sample_count": len(identity["sample_order"]),
        "scored_sample_count": len(identity["sample_order"]),
        "requested_scored_mismatch_count": 0,
        "tool_failure_count": 0,
        "offline_scoring_after_rollout_complete": True,
        "summary_validator_passed": True,
    }
    if gates != expected:
        raise ReceiptValidationError("receipt hard gates are absent or failed")
    if scheduler_totals["tool_failure_count"] != 0:
        raise ReceiptValidationError("scheduler evidence records a tool failure")
    if episode_totals["tool_failure_count"] != 0:
        raise ReceiptValidationError("episode evidence records a tool failure")


def _validate_summary(
    summary: Mapping[str, Any],
    identity: Mapping[str, Any],
    configured: Mapping[str, int],
    scheduler_totals: Mapping[str, int],
    scheduler_samples: Mapping[str, Mapping[str, int]],
    episode_totals: Mapping[str, int],
) -> None:
    schema = summary.get("summary_schema_version", summary.get("schema_id"))
    if schema != SUMMARY_SCHEMA_ID:
        raise ReceiptValidationError("bound summary is not schema v2")
    expected = {
        "status": "completed",
        "evaluation_profile": identity["evaluation_profile"],
        "evaluation_id": identity["evaluation_id"],
        "model_state": identity["model_state"],
        "job_id": identity["job_id"],
        "experiment_arm": identity["experiment_arm_id"],
        "protocol_version": identity["model_io_protocol"],
        "runtime_commit": identity["runtime_commit"],
        "harness_commit": identity["harness_commit"],
        "requested_samples": len(identity["sample_order"]),
        "scored_samples": len(identity["sample_order"]),
        "sample_order": identity["sample_order"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ReceiptValidationError(f"bound summary.{key} is not bound")
    if summary.get("harness_protocol_id") != identity["harness_protocol_id"]:
        raise ReceiptValidationError(
            "bound summary harness protocol is not receipt-bound"
        )
    gt_boundary = _object(summary.get("gt_boundary"), source="summary.gt_boundary")
    if gt_boundary.get("rollout_marker_claims_gt_free") is not True:
        raise ReceiptValidationError("summary does not preserve GT-free rollout")
    if gt_boundary.get("offline_metrics_are_post_rollout") is not True:
        raise ReceiptValidationError("summary offline scoring boundary is absent")
    p1 = _object(summary.get("p1_contract"), source="summary.p1_contract")
    expected_p1 = {
        "schema_version": "qwen3vl_e2e_summary_p1_contract_v1",
        "source_audit_schema_id": P1_AUDIT_SCHEMA_ID,
        "factor_id": FACTOR_ID,
        "arm": identity["arm"],
        "harness_protocol_id": identity["harness_protocol_id"],
        "experiment_arm": identity["experiment_arm_id"],
        "terminal_review": ARM_CONTRACTS[str(identity["arm"])][
            "terminal_review"
        ],
        "model_io_protocol": MODEL_IO_PROTOCOL,
        "gt_loaded_before_rollout_complete": False,
        "frozen_safety_boundaries": list(SAFETY_BOUNDARIES),
    }
    for key, value in expected_p1.items():
        if p1.get(key) != value:
            raise ReceiptValidationError(f"summary.p1_contract.{key} is not bound")
    hard_gates = _object(p1.get("hard_gates"), source="summary hard gates")
    if hard_gates != {
        "requested_scored_mismatch_count": 0,
        "pre_rollout_gt_violation_count": 0,
        "safety_boundary_relaxation_count": 0,
        "tool_failure_count": 0,
        "all_passed": True,
    }:
        raise ReceiptValidationError("summary P1 hard gates are absent or failed")
    if p1.get("configured_budgets") != dict(configured):
        raise ReceiptValidationError("summary configured budgets differ from receipt")
    if p1.get("budget_accounting_totals") != dict(scheduler_totals):
        raise ReceiptValidationError("summary scheduler totals differ from receipt")
    p1_samples = _object(p1.get("per_sample"), source="summary P1 per_sample")
    if list(p1_samples) != list(identity["sample_order"]):
        raise ReceiptValidationError("summary P1 per-sample order differs")
    for sample in identity["sample_order"]:
        record = _object(p1_samples[sample], source=f"summary P1 {sample}")
        if record.get("budget_accounting") != dict(scheduler_samples[sample]):
            raise ReceiptValidationError(
                f"summary P1 {sample} scheduler accounting differs"
            )
        gt = _object(record.get("gt_boundary_audit"), source=f"summary GT {sample}")
        if gt.get("pre_rollout_gt_access_count") != 0:
            raise ReceiptValidationError(f"summary P1 {sample} records GT access")
        safety = _object(
            record.get("safety_boundary_audit"),
            source=f"summary safety {sample}",
        )
        if safety.get("relaxation_count") != 0:
            raise ReceiptValidationError(
                f"summary P1 {sample} records safety relaxation"
            )
        if not isinstance(record.get("final_region_protocol_valid"), bool):
            raise ReceiptValidationError(
                f"summary P1 {sample} protocol validity is malformed"
            )
        terminal = record.get("terminal_review")
        if identity["arm"] == "h0" and terminal is not None:
            audit = _object(terminal, source=f"summary H0 audit {sample}")
            if audit.get("round_two_invoked") is not False:
                raise ReceiptValidationError("summary H0 audit invoked round two")
    valid_count = _integer(
        p1.get("final_region_protocol_valid_count"),
        source="summary final-region validity count",
    )
    round_two_count = _integer(
        p1.get("round_two_invoked_count"),
        source="summary round-two invocation count",
    )
    sample_count = len(identity["sample_order"])
    if valid_count > sample_count or round_two_count > sample_count:
        raise ReceiptValidationError("summary P1 counts exceed requested samples")
    if identity["arm"] == "h0" and round_two_count != 0:
        raise ReceiptValidationError("summary H0 round-two count is non-zero")
    provenance = _object(p1.get("provenance"), source="summary P1 provenance")
    expected_provenance = {
        "model_state": identity["model_state"],
        "runtime_commit": identity["runtime_commit"],
        "harness_commit": identity["harness_commit"],
        "job_id": identity["job_id"],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ReceiptValidationError(
                f"summary P1 provenance.{key} differs from receipt"
            )
    if provenance.get("artifact_bindings") != summary.get("artifact_bindings"):
        raise ReceiptValidationError("summary duplicates inconsistent artifact bindings")
    actions = _object(summary.get("actions"), source="summary.actions")
    for key, value in episode_totals.items():
        if actions.get(key) != value:
            raise ReceiptValidationError(
                f"summary episode total {key} differs from receipt"
            )


def _validate_protocol_summary(
    protocol: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    expected = {
        "status": "completed",
        "evaluation_profile": identity["evaluation_profile"],
        "evaluation_id": identity["evaluation_id"],
        "experiment_arm": identity["experiment_arm_id"],
        "protocol_version": identity["model_io_protocol"],
        "job_id": identity["job_id"],
        "runtime_commit": identity["runtime_commit"],
        "harness_commit": identity["harness_commit"],
        "sample_order": identity["sample_order"],
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ReceiptValidationError(
                f"bound protocol summary.{key} is not bound"
            )
    harness_id = protocol.get("harness_protocol_id")
    if harness_id != identity["harness_protocol_id"]:
        raise ReceiptValidationError(
            "bound protocol summary harness protocol is not receipt-bound"
        )


def _validate_p1_audit(
    audit: Mapping[str, Any],
    identity: Mapping[str, Any],
    totals: Mapping[str, int],
) -> None:
    expected = {
        "schema_id": P1_AUDIT_SCHEMA_ID,
        "factor_id": FACTOR_ID,
        "harness_protocol_id": identity["harness_protocol_id"],
        "experiment_arm_id": identity["experiment_arm_id"],
        "arm": identity["arm"],
        "requested_sample_ids": identity["sample_order"],
        "totals": dict(totals),
        "gt_boundary_violation_count": 0,
        "safety_boundary_relaxation_count": 0,
        "tool_failure_count": 0,
        "written_before_rollout_complete": True,
        "gt_loaded": False,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ReceiptValidationError(f"bound P1 audit.{key} is not bound")
    per_sample = _object(audit.get("per_sample"), source="p1_audit.per_sample")
    if list(per_sample) != list(identity["sample_order"]):
        raise ReceiptValidationError("P1 audit sample order differs from receipt")
    for sample, record_value in per_sample.items():
        record = _object(record_value, source=f"p1_audit.per_sample.{sample}")
        gt = _object(record.get("gt_boundary_audit"), source="gt boundary audit")
        if gt.get("pre_rollout_gt_access_count") != 0:
            raise ReceiptValidationError("P1 audit records pre-rollout GT access")
        if gt.get("violation_ids") != []:
            raise ReceiptValidationError("P1 audit records GT violation IDs")
        if gt.get("rollout_completed_before_offline_scoring") is not True:
            raise ReceiptValidationError("P1 audit lacks offline boundary proof")
        safety = _object(
            record.get("safety_boundary_audit"), source="safety boundary audit"
        )
        if safety.get("relaxation_count") != 0:
            raise ReceiptValidationError("P1 audit records safety relaxation")
        if safety.get("relaxed_boundaries") != []:
            raise ReceiptValidationError("P1 audit records relaxed boundaries")
        terminal = record.get("terminal_review")
        if identity["arm"] == "h0" and terminal is not None:
            terminal_payload = _object(terminal, source="H0 terminal review audit")
            if terminal_payload.get("round_two_invoked") is not False:
                raise ReceiptValidationError("H0 audit invoked round two")


def _validate_evaluation_input_manifest(
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    if manifest.get("schema_id") != "qwen3vl_evaluation_input_manifest_v1":
        raise ReceiptValidationError("evaluation input manifest schema is not frozen")
    if manifest.get("design_status") != "frozen":
        raise ReceiptValidationError("evaluation input manifest is not frozen")
    observed_digest = _canonical_json_sha256(manifest)
    if observed_digest != provenance[
        "evaluation_input_manifest_canonical_sha256"
    ]:
        raise ReceiptValidationError(
            "evaluation input manifest canonical digest differs from receipt"
        )
    profiles = manifest.get("evaluation_profiles")
    if not isinstance(profiles, list):
        raise ReceiptValidationError(
            "evaluation input manifest profiles must be a list"
        )
    expected_phase = (
        "smoke1" if identity["evaluation_profile"] == "p1_smoke1" else "core3"
    )
    matches = [
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("phase") == expected_phase
    ]
    if len(matches) != 1:
        raise ReceiptValidationError(
            "evaluation input manifest lacks one frozen evaluation profile"
        )
    profile = matches[0]
    if profile.get("evaluation_id") != identity["evaluation_id"]:
        raise ReceiptValidationError(
            "evaluation input manifest evaluation ID differs from receipt"
        )
    if profile.get("ordered_sample_ids") != identity["sample_order"]:
        raise ReceiptValidationError(
            "evaluation input manifest sample order differs from receipt"
        )
    if profile.get("arm_order") != ["h0", "h1"]:
        raise ReceiptValidationError(
            "evaluation input manifest arm order is not frozen"
        )
    source = _object(
        manifest.get("source_contract"), source="evaluation input source contract"
    )
    input_root = _string(
        source.get("input_root"), source="evaluation input source root"
    )
    if not Path(input_root).is_absolute():
        raise ReceiptValidationError("evaluation input root must be absolute")
    gt = _object(manifest.get("gt_boundary"), source="evaluation input GT boundary")
    if gt.get("a_and_b_online_inputs") is not True:
        raise ReceiptValidationError("evaluation input A/B online contract is absent")
    if gt.get("label_cvt_online_input") is not False:
        raise ReceiptValidationError("evaluation input exposes labels online")
    if gt.get("label_cvt_offline_scorer_only") is not True:
        raise ReceiptValidationError("evaluation input offline-label boundary is absent")
    samples = _object(manifest.get("samples"), source="evaluation input samples")
    for sample in identity["sample_order"]:
        record = _object(samples.get(sample), source=f"evaluation input {sample}")
        for key in ("a_sha256", "b_sha256", "label_sha256"):
            digest = _string(record.get(key), source=f"evaluation input {sample}.{key}")
            if SHA256_RE.fullmatch(digest) is None:
                raise ReceiptValidationError(
                    f"evaluation input {sample}.{key} is malformed"
                )
        if record.get("label_access") != "offline_scorer_only_after_rollout_complete":
            raise ReceiptValidationError(
                f"evaluation input {sample} label-access contract differs"
            )
    return input_root


def _validate_runtime_input_manifest(
    runtime: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    config: Mapping[str, Any],
    amendment: Mapping[str, Any],
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    input_root: str,
    *,
    evaluation_manifest_path: Path,
    resolved_config_path: Path,
) -> None:
    frozen_models, frozen_tools, frozen_workers = _config_hash_bindings(config)
    expected = {
        "evaluation_profile": identity["evaluation_profile"],
        "evaluation_id": identity["evaluation_id"],
        "experiment_arm": identity["experiment_arm_id"],
        "protocol_version": MODEL_IO_PROTOCOL,
        "harness_protocol_id": identity["harness_protocol_id"],
        "terminal_review_arm": identity["arm"],
        "sample_order": identity["sample_order"],
        "input_root": input_root,
        "slurm_job_id": identity["job_id"],
        "model_path": provenance["model_path"],
        "gt_hashed_after_rollout": True,
    }
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise ReceiptValidationError(
                f"runtime input manifest {key} differs from receipt"
            )
    evaluation_binding = _object(
        runtime.get("evaluation_input_manifest"),
        source="runtime evaluation-input binding",
    )
    if evaluation_binding.get("path") != str(evaluation_manifest_path):
        raise ReceiptValidationError("runtime evaluation-input path differs")
    if evaluation_binding.get("canonical_sha256") != _canonical_json_sha256(
        evaluation
    ):
        raise ReceiptValidationError("runtime evaluation-input digest differs")
    config_binding = _object(
        runtime.get("resolved_run_config"),
        source="runtime resolved-config binding",
    )
    if config_binding.get("path") != str(resolved_config_path):
        raise ReceiptValidationError("runtime resolved-config path differs")
    if config_binding.get("sha256") != _sha256(resolved_config_path):
        raise ReceiptValidationError("runtime resolved-config digest differs")
    static_samples = _object(evaluation.get("samples"), source="static samples")
    runtime_files = _object(runtime.get("files_sha256"), source="runtime files")
    if list(runtime_files) != list(identity["sample_order"]):
        raise ReceiptValidationError("runtime input file order differs from receipt")
    for sample in identity["sample_order"]:
        observed = _object(runtime_files[sample], source=f"runtime files {sample}")
        frozen = _object(static_samples[sample], source=f"static files {sample}")
        expected_hashes = {
            "A": frozen.get("a_sha256"),
            "B": frozen.get("b_sha256"),
            "label_cvt": frozen.get("label_sha256"),
        }
        if dict(observed) != expected_hashes:
            raise ReceiptValidationError(
                f"runtime input hashes for {sample} differ from static manifest"
            )
    model_files = _object(
        runtime.get("model_files_sha256"), source="runtime model files"
    )
    if model_files.get("model.safetensors") != provenance["model_sha256"]:
        raise ReceiptValidationError("runtime base-model hash differs from receipt")
    for name, digest in model_files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ReceiptValidationError("runtime model file hash is malformed")
    if {name: model_files.get(name) for name in frozen_models} != frozen_models:
        raise ReceiptValidationError(
            "runtime model subset differs from resolved config"
        )
    model_root = Path(str(provenance["model_path"]))
    for name, expected_digest in frozen_models.items():
        model_file = _resolve_bound_path(
            str(model_root / name), source=f"live frozen model file {name}"
        )
        if _sha256(model_file) != expected_digest:
            raise ReceiptValidationError(f"live model file drifted: {name}")
    tool_files = _object(
        runtime.get("tool_files_sha256"), source="runtime tool files"
    )
    if dict(tool_files) != dict(provenance["tool_sha256"]):
        raise ReceiptValidationError("runtime tool hashes differ from receipt")
    if dict(tool_files) != frozen_tools:
        raise ReceiptValidationError("runtime tool hashes differ from resolved config")
    if _frozen_hashes(
        amendment.get("tool_hashes"), source="P1 amendment tool_hashes"
    ) != frozen_tools:
        raise ReceiptValidationError("P1 amendment tool hashes differ from config")
    if _amendment_model_hashes(amendment.get("model_hash")) != frozen_models:
        raise ReceiptValidationError("P1 amendment model hashes differ from config")
    if _amendment_worker_hashes(
        amendment.get("worker_sha256")
    ) != frozen_workers:
        raise ReceiptValidationError("P1 amendment worker hashes differ from config")
    if runtime.get("executed_worker_sha256") != provenance["worker_sha256"]:
        raise ReceiptValidationError("runtime worker hash differs from receipt")
    if tool_files.get("harness_worker") != provenance["worker_sha256"]:
        raise ReceiptValidationError("runtime harness-worker hash is inconsistent")
    if (
        frozen_workers.get("evaluation") != provenance["worker_sha256"]
        or frozen_tools.get("harness_worker") != provenance["worker_sha256"]
    ):
        raise ReceiptValidationError(
            "frozen evaluation-worker bindings differ from receipt"
        )
    tool_paths = _object(
        runtime.get("tool_file_paths"), source="runtime tool file paths"
    )
    if set(tool_paths) != set(frozen_tools):
        raise ReceiptValidationError(
            "runtime tool path names differ from frozen exact set"
        )
    for name, expected_digest in frozen_tools.items():
        tool_path = _resolve_bound_path(
            tool_paths.get(name), source=f"runtime tool path {name}"
        )
        if _sha256(tool_path) != expected_digest:
            raise ReceiptValidationError(f"live runtime tool drifted: {name}")
    source_hashes = _object(
        runtime.get("runtime_source_sha256"), source="runtime source hashes"
    )
    if not source_hashes:
        raise ReceiptValidationError("runtime source hash set must not be empty")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ReceiptValidationError("runtime source hash is malformed")
    commits = _object(runtime.get("runtime_commits"), source="runtime commits")
    for key in (
        "chang_agent",
        "expected_chang_agent",
        "harness",
        "expected_harness",
    ):
        if commits.get(key) != identity["runtime_commit"]:
            raise ReceiptValidationError(f"runtime commit {key} differs from receipt")
    environment = _object(
        runtime.get("runtime_environment"), source="runtime environment"
    )
    device = _string(environment.get("device"), source="runtime device")
    if "A800" not in device:
        raise ReceiptValidationError("runtime device is not an NVIDIA A800")


def _validate_resolved_run_config(
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_path: Path,
) -> None:
    if config.get("schema_id") != "qwen3vl_p1_resolved_run_config_v1":
        raise ReceiptValidationError("resolved run config schema is not frozen")
    if (
        config.get("configuration_status") != "frozen"
        or config.get("execution_readiness") != "execution_ready"
        or config.get("p1_execution_authorized") is not True
    ):
        raise ReceiptValidationError(
            "resolved run config is not frozen, execution-ready and authorized"
        )
    frozen_models, frozen_tools, frozen_workers = _config_hash_bindings(config)
    if frozen_models.get("model.safetensors") != provenance["model_sha256"]:
        raise ReceiptValidationError("resolved config model hash differs from receipt")
    if frozen_tools != dict(provenance["tool_sha256"]):
        raise ReceiptValidationError("resolved config tool hashes differ from receipt")
    if (
        frozen_workers.get("evaluation") != provenance["worker_sha256"]
        or frozen_tools.get("harness_worker") != provenance["worker_sha256"]
    ):
        raise ReceiptValidationError(
            "resolved config evaluation worker hash differs from receipt"
        )
    paths = _object(config.get("paths"), source="resolved config paths")
    for key, expected in {
        "clean_run_worktree": provenance["run_worktree"],
        "base_model": provenance["model_path"],
    }.items():
        if paths.get(key) != expected:
            raise ReceiptValidationError(f"resolved config path {key} differs")
    frozen = _object(config.get("frozen_identity"), source="resolved config identity")
    if frozen.get("factor_id") != FACTOR_ID:
        raise ReceiptValidationError("resolved config factor ID differs")
    if frozen.get("model_state") != "base" or frozen.get("model_io_protocol") != MODEL_IO_PROTOCOL:
        raise ReceiptValidationError("resolved config model identity differs")
    if frozen.get("seed") != 42:
        raise ReceiptValidationError("resolved config seed differs")
    arm = _object(frozen.get(identity["arm"]), source="resolved config arm")
    expected_arm = {
        "terminal_review_arm": identity["arm"],
        "experiment_arm_id": identity["experiment_arm_id"],
        "harness_protocol_id": identity["harness_protocol_id"],
        "max_verifier_rounds": ARM_CONTRACTS[str(identity["arm"])]
        ["max_verifier_rounds"],
    }
    for key, expected in expected_arm.items():
        if arm.get(key) != expected:
            raise ReceiptValidationError(f"resolved config arm {key} differs")
    profiles = _object(config.get("profiles"), source="resolved config profiles")
    profile = _object(profiles.get(identity["evaluation_profile"]), source="resolved config profile")
    if profile.get("evaluation_id") != identity["evaluation_id"]:
        raise ReceiptValidationError("resolved config evaluation ID differs")
    if profile.get("ordered_samples") != identity["sample_order"]:
        raise ReceiptValidationError("resolved config sample order differs")
    slurm = _object(config.get("slurm"), source="resolved config Slurm")
    if slurm.get("partition") != "gre" or slurm.get("fixed_node") != "gpu48":
        raise ReceiptValidationError("resolved config Slurm placement differs")
    if slurm.get("maximum_ca0802_concurrent_gpus") != 3:
        raise ReceiptValidationError("resolved config maximum GPU budget differs")
    if slurm.get("maximum_p1_concurrent_gpus_by_dependency_design") != 1:
        raise ReceiptValidationError("resolved config P1 concurrency differs")
    allocations = config.get("allocations")
    if not isinstance(allocations, list):
        raise ReceiptValidationError("resolved config allocations are absent")
    expected_allocation_id = {
        ("p1_smoke1", "h0"): "CA_0802(3)",
        ("p1_smoke1", "h1"): "CA_0802(4)",
        ("e2e3", "h0"): "CA_0802(5)",
        ("e2e3", "h1"): "CA_0802(6)",
    }[(str(identity["evaluation_profile"]), str(identity["arm"]))]
    matches = [
        item
        for item in allocations
        if isinstance(item, Mapping) and item.get("allocation_id") == expected_allocation_id
    ]
    if len(matches) != 1:
        raise ReceiptValidationError("resolved config allocation differs")
    allocation = matches[0]
    if allocation.get("arm") != identity["arm"] or allocation.get("evaluation_profile") != identity["evaluation_profile"]:
        raise ReceiptValidationError("resolved config allocation identity differs")
    resources = _object(allocation.get("resources"), source="resolved config resources")
    if resources.get("gpus") != 1:
        raise ReceiptValidationError("P1 evaluation allocation must use one GPU")
    if not config_path.is_file() or _sha256(config_path) == hashlib.sha256(b"").hexdigest():
        raise ReceiptValidationError("resolved config is empty")


def _validate_p1_amendment(
    amendment: Mapping[str, Any],
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    amendment_path: Path,
    config_path: Path,
    submission_path: Path,
) -> None:
    if amendment.get("schema_id") != "qwen3vl_p1_execution_amendment_v1":
        raise ReceiptValidationError("P1 amendment schema differs")
    if (
        amendment.get("record_status") != "frozen"
        or amendment.get("execution_readiness") != "execution_ready"
        or amendment.get("p1_execution_authorized") is not True
    ):
        raise ReceiptValidationError("P1 amendment is not frozen and execution-ready")
    if amendment.get("implementation_commit") != provenance["implementation_commit"]:
        raise ReceiptValidationError("P1 amendment implementation commit differs")
    if Path(str(amendment.get("clean_run_worktree", ""))).resolve() != Path(
        str(provenance["run_worktree"])
    ).resolve():
        raise ReceiptValidationError("P1 amendment run-worktree binding differs")
    if amendment.get("resolved_run_config_sha256") != _sha256(config_path):
        raise ReceiptValidationError("P1 amendment resolved-config hash differs")
    frozen_models, frozen_tools, frozen_workers = _config_hash_bindings(config)
    if _amendment_model_hashes(amendment.get("model_hash")) != frozen_models:
        raise ReceiptValidationError("P1 amendment model hashes differ from config")
    if _frozen_hashes(
        amendment.get("tool_hashes"), source="P1 amendment tool_hashes"
    ) != frozen_tools:
        raise ReceiptValidationError("P1 amendment tool hashes differ from config")
    amendment_workers = _amendment_worker_hashes(
        amendment.get("worker_sha256")
    )
    if amendment_workers != frozen_workers:
        raise ReceiptValidationError("P1 amendment worker hashes differ from config")
    if (
        amendment_workers.get("evaluation") != provenance["worker_sha256"]
        or frozen_workers.get("evaluation") != provenance["worker_sha256"]
        or frozen_tools.get("harness_worker") != provenance["worker_sha256"]
    ):
        raise ReceiptValidationError(
            "P1 amendment evaluation worker binding differs from receipt"
        )
    config_paths = _object(config.get("paths"), source="resolved config paths")
    relative = config_paths.get("p1_amendment")
    if not isinstance(relative, str) or not relative:
        raise ReceiptValidationError("resolved config P1 amendment path is absent")
    expected_path = (config_path.parents[3] / relative).resolve()
    if expected_path != amendment_path.resolve():
        raise ReceiptValidationError("resolved config P1 amendment path differs")
    submission = _parse_key_value(
        submission_path,
        source="submission receipt",
    )
    if submission.get("expected_commit") != identity["runtime_commit"]:
        raise ReceiptValidationError("P1 amendment submission freeze commit differs")
    if submission.get("implementation_commit") != provenance["implementation_commit"]:
        raise ReceiptValidationError(
            "P1 amendment submission implementation commit differs"
        )
    if submission.get("amendment_canonical_sha256") != _canonical_json_sha256(
        amendment
    ):
        raise ReceiptValidationError("submission P1 amendment digest differs")


def _validate_rollout(marker: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if marker.get("execution_status") != "completed":
        raise ReceiptValidationError("rollout marker is not completed")
    if marker.get("gt_loaded") is not False:
        raise ReceiptValidationError("rollout marker does not prove gt_loaded=false")
    samples = marker.get("samples")
    if samples not in (
        identity["sample_order"],
        sorted(identity["sample_order"]),
    ):
        raise ReceiptValidationError("rollout marker sample identity differs")


def _validate_summary_artifact_bindings(
    summary: Mapping[str, Any],
    identity: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    bindings = _object(
        summary.get("artifact_bindings"), source="summary.artifact_bindings"
    )
    if bindings.get("algorithm") != "sha256":
        raise ReceiptValidationError("summary artifact hash algorithm is not sha256")
    marker = _object(bindings.get("rollout_marker"), source="summary rollout binding")
    metrics = _object(bindings.get("source_metrics"), source="summary metrics binding")
    audit = _object(bindings.get("p1_harness_audit"), source="summary audit binding")
    expected = {
        "rollout_marker": (paths["rollout_marker"], marker),
        "offline_scorer_input": (paths["offline_scorer_input"], metrics),
        "p1_harness_audit": (paths["p1_harness_audit"], audit),
    }
    for name, (path, binding) in expected.items():
        if binding.get("path") != str(path):
            raise ReceiptValidationError(
                f"summary artifact binding {name} path differs from receipt"
            )
        if binding.get("sha256") != _sha256(path):
            raise ReceiptValidationError(
                f"summary artifact binding {name} hash differs from receipt"
            )
    per_sample = _object(bindings.get("per_sample"), source="summary per-sample bindings")
    if list(per_sample) != list(identity["sample_order"]):
        raise ReceiptValidationError("summary trajectory binding order differs")
    for sample, sample_value in per_sample.items():
        record = _object(sample_value, source=f"summary bindings {sample}")
        for artifact_name in ("trajectory", "episode_summary"):
            binding = _object(
                record.get(artifact_name),
                source=f"summary bindings {sample}.{artifact_name}",
            )
            path = _resolve_bound_path(
                binding.get("path"),
                source=f"summary bindings {sample}.{artifact_name}.path",
            )
            if binding.get("sha256") != _sha256(path):
                raise ReceiptValidationError(
                    f"summary bindings {sample}.{artifact_name} hash mismatch"
                )
        reports = record.get("tool_reports")
        if not isinstance(reports, list):
            raise ReceiptValidationError(
                f"summary bindings {sample}.tool_reports must be a list"
            )
        for index, report_value in enumerate(reports):
            report = _object(report_value, source="tool report binding")
            report_path = _resolve_bound_path(
                report.get("path"), source=f"tool report {sample}[{index}].path"
            )
            if report.get("sha256") != _sha256(report_path):
                raise ReceiptValidationError("tool report binding hash mismatch")


def _validate_run_manifest(path: Path, identity: Mapping[str, Any]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReceiptValidationError(f"run manifest is unreadable: {error}") from error
    required_fragments = (
        f"terminal_review_factor_id: `{FACTOR_ID}`",
        f"terminal_review_arm: `{identity['arm']}`",
        f"harness_protocol_id: `{identity['harness_protocol_id']}`",
        f"experiment_arm_id: `{identity['experiment_arm_id']}`",
        f"git_commit: `{identity['runtime_commit']}`",
        "git_dirty: `False`",
        "seed: `42`",
    )
    for fragment in required_fragments:
        if fragment not in text:
            raise ReceiptValidationError(
                f"run manifest lacks frozen binding {fragment}"
            )


def _validate_launch(path: Path, identity: Mapping[str, Any]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReceiptValidationError(f"launch command is unreadable: {error}") from error
    tokens = []
    for token in shlex.split(text):
        tokens.append(token.removeprefix("launch="))
    required_pairs = {
        "--terminal-review-arm": identity["arm"],
        "--seed": "42",
    }
    for option, expected in required_pairs.items():
        try:
            observed = tokens[tokens.index(option) + 1]
        except (ValueError, IndexError) as error:
            raise ReceiptValidationError(
                f"launch command lacks {option} {expected}"
            ) from error
        if observed != expected:
            raise ReceiptValidationError(
                f"launch command {option} differs from receipt"
            )


def _validate_summary_validation(
    path: Path, identity: Mapping[str, Any]
) -> None:
    try:
        line = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise ReceiptValidationError(
            f"summary validation evidence is unreadable: {error}"
        ) from error
    expected = (
        "status=valid;"
        f"profile={identity['evaluation_profile']};"
        f"arm={identity['experiment_arm_id']};"
        f"scored={len(identity['sample_order'])}/{len(identity['sample_order'])}"
    )
    if line != expected:
        raise ReceiptValidationError(
            "summary validation evidence is not exactly receipt-bound"
        )


def _validate_text_receipts(
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    submission = _parse_key_value(
        paths["submission_receipt"], source="submission receipt"
    )
    status = _parse_key_value(paths["status_receipt"], source="status receipt")
    expected_submission_keys = [
        "submission_status",
        "job_id",
        "job_name",
        "allocation_id",
        "evaluation_profile",
        "evaluation_id",
        "samples",
        "arm",
        "experiment_arm_id",
        "harness_protocol_id",
        "expected_commit",
        "implementation_commit",
        "worker_sha256",
        "config_sha256",
        "amendment_canonical_sha256",
        "partition",
        "node",
        "resources",
        "dependency",
        "output",
    ]
    expected_status_keys = [
        "status",
        "exit_code",
        "elapsed_seconds",
        "job_id",
        "job_name",
        "allocation_id",
        "evaluation_profile",
        "evaluation_id",
        "samples",
        "arm",
        "experiment_arm_id",
        "harness_protocol_id",
        "commit",
        "worker_sha256",
        "result",
        "error",
        "path",
    ]
    if list(submission) != expected_submission_keys:
        raise ReceiptValidationError("submission receipt keys/order are not frozen")
    if list(status) != expected_status_keys:
        raise ReceiptValidationError("status receipt keys/order are not frozen")
    allocation = {
        ("p1_smoke1", "h0"): ("CA_0802(3)", "CA0802P1SmokeH0"),
        ("p1_smoke1", "h1"): ("CA_0802(4)", "CA0802P1SmokeH1"),
        ("e2e3", "h0"): ("CA_0802(5)", "CA0802P1CoreH0"),
        ("e2e3", "h1"): ("CA_0802(6)", "CA0802P1CoreH1"),
    }[(str(identity["evaluation_profile"]), str(identity["arm"]))]
    output_parent = str(paths["baseline_summary"].parent)
    common = {
        "job_id": identity["job_id"],
        "job_name": allocation[1],
        "allocation_id": allocation[0],
        "evaluation_profile": identity["evaluation_profile"],
        "evaluation_id": identity["evaluation_id"],
        "samples": " ".join(identity["sample_order"]),
        "arm": identity["arm"],
        "experiment_arm_id": identity["experiment_arm_id"],
        "harness_protocol_id": identity["harness_protocol_id"],
        "worker_sha256": provenance["worker_sha256"],
    }
    for name, payload in (("submission", submission), ("status", status)):
        for key, expected in common.items():
            if payload.get(key) != str(expected):
                raise ReceiptValidationError(
                    f"{name} receipt {key} is not bound"
                )
    if submission.get("submission_status") != "submitted":
        raise ReceiptValidationError("submission receipt is not submitted")
    if submission.get("expected_commit") != identity["runtime_commit"]:
        raise ReceiptValidationError("submission receipt commit is not bound")
    if submission.get("implementation_commit") != provenance["implementation_commit"]:
        raise ReceiptValidationError(
            "submission receipt implementation commit is not bound"
        )
    if submission.get("config_sha256") != _sha256(
        paths["resolved_run_config"]
    ):
        raise ReceiptValidationError("submission receipt config hash is not bound")
    amendment_digest = submission.get("amendment_canonical_sha256", "")
    if SHA256_RE.fullmatch(amendment_digest) is None:
        raise ReceiptValidationError("submission amendment digest is malformed")
    if submission.get("partition") != "gre" or submission.get("node") != "gpu48":
        raise ReceiptValidationError("submission Slurm placement is not frozen")
    if submission.get("output") != output_parent:
        raise ReceiptValidationError("submission output path is not receipt-bound")
    if status.get("status") != "success" or status.get("exit_code") != "0":
        raise ReceiptValidationError("status receipt is not successful")
    if status.get("commit") != identity["runtime_commit"]:
        raise ReceiptValidationError("status receipt commit is not bound")
    if status.get("path") != output_parent:
        raise ReceiptValidationError("status receipt output path is not bound")
    try:
        elapsed = int(status.get("elapsed_seconds", ""))
    except ValueError as error:
        raise ReceiptValidationError("status elapsed_seconds is malformed") from error
    if elapsed < 0:
        raise ReceiptValidationError("status elapsed_seconds is negative")
    if not status.get("result") or status.get("error"):
        raise ReceiptValidationError("status receipt lacks a clean success result")


def validate_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.resolve()
    receipt = _read_json(receipt_path, source="evaluation receipt")
    if list(receipt) != [
        "schema_id",
        "status",
        "terminal_state",
        "identity",
        "provenance",
        "budget_accounting",
        "gates",
        "artifacts",
    ]:
        raise ReceiptValidationError(
            "receipt top-level keys or order differ from the v2 contract"
        )
    if receipt.get("schema_id") != SCHEMA_ID:
        raise ReceiptValidationError("receipt.schema_id is not v2")
    if receipt.get("status") != "complete":
        raise ReceiptValidationError("only status=complete may pass validation")
    if receipt.get("terminal_state") != "success":
        raise ReceiptValidationError("complete receipt terminal_state must be success")
    identity = _identity(receipt.get("identity"))
    provenance = _validate_provenance(receipt.get("provenance"), identity)
    totals, per_sample, episode_totals, configured = _validate_budget_accounting(
        receipt.get("budget_accounting"), identity
    )
    _validate_gates(
        receipt.get("gates"), identity, totals, episode_totals
    )
    paths, payloads = _artifact_bindings(receipt.get("artifacts"))
    _validate_summary(
        payloads["baseline_summary"],
        identity,
        configured,
        totals,
        per_sample,
        episode_totals,
    )
    _validate_summary_artifact_bindings(
        payloads["baseline_summary"], identity, paths
    )
    _validate_protocol_summary(payloads["protocol_summary"], identity)
    _validate_p1_audit(payloads["p1_harness_audit"], identity, totals)
    _validate_resolved_run_config(
        payloads["resolved_run_config"],
        identity,
        provenance,
        paths["resolved_run_config"],
    )
    _validate_p1_amendment(
        payloads["p1_amendment"],
        payloads["resolved_run_config"],
        identity,
        provenance,
        amendment_path=paths["p1_amendment"],
        config_path=paths["resolved_run_config"],
        submission_path=paths["submission_receipt"],
    )
    input_root = _validate_evaluation_input_manifest(
        payloads["evaluation_input_manifest"],
        identity,
        provenance,
    )
    _validate_runtime_input_manifest(
        payloads["runtime_input_manifest"],
        payloads["evaluation_input_manifest"],
        payloads["resolved_run_config"],
        payloads["p1_amendment"],
        identity,
        provenance,
        input_root,
        evaluation_manifest_path=paths["evaluation_input_manifest"],
        resolved_config_path=paths["resolved_run_config"],
    )
    _validate_rollout(payloads["rollout_marker"], identity)
    _validate_run_manifest(paths["run_manifest"], identity)
    metrics = payloads["offline_scorer_input"]
    samples = _object(metrics.get("samples"), source="offline scorer input samples")
    if list(samples) != list(identity["sample_order"]):
        raise ReceiptValidationError("offline scorer sample order differs")
    _validate_launch(paths["launch_command"], identity)
    _validate_text_receipts(paths, identity, provenance)
    _validate_summary_validation(paths["summary_validation"], identity)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = validate_receipt(args.receipt)
    except ReceiptValidationError as error:
        print(f"qwen3vl_evaluation_receipt_v2: {error}", file=sys.stderr)
        return 3
    identity = receipt["identity"]
    print(
        "status=complete;"
        f"arm={identity['arm']};"
        f"profile={identity['evaluation_profile']};"
        f"job_id={identity['job_id']};"
        f"samples={len(identity['sample_order'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
