#!/usr/bin/env python3
"""Build a hash-bound Qwen3-VL P1 evaluation receipt v2.

The builder is deliberately narrow.  It does not score an evaluation or
repair any producer artifact.  It verifies the frozen identity, clean Git
worktree, successful text receipts, post-rollout runtime manifest and summary
hard gates, then publishes an immutable index over the already-written
artifacts.  The independently maintained receipt validator remains the final
authority on whether the resulting receipt is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "qwen3vl_evaluation_receipt_v2"
SUMMARY_SCHEMA_ID = "qwen3vl_e2e_summary_v2"
P1_AUDIT_SCHEMA_ID = "qwen3vl_p1_harness_audit_v1"
FACTOR_ID = "terminal_review_scheduler_v1"
MODEL_IO_PROTOCOL = "v12.1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

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
        "harness_protocol_id": "qwen3vl_harness_h0_legacy_single_round_v1",
        "terminal_review": False,
        "max_verifier_rounds": 1,
    },
    "h1": {
        "experiment_arm_id": "v12_1-h1-gtfree-terminal-review-base-0802",
        "harness_protocol_id": "qwen3vl_harness_h1_gtfree_terminal_review_v1",
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


class ReceiptBuildError(ValueError):
    """Raised when source evidence cannot support a complete receipt."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--expected-samples", nargs="+", required=True)
    parser.add_argument("--arm", choices=tuple(ARM_CONTRACTS), required=True)
    parser.add_argument("--experiment-arm", required=True)
    parser.add_argument("--harness-protocol-id", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-worktree", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--evaluation-input-manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-input-manifest-canonical-sha256", required=True
    )
    parser.add_argument("--runtime-input-manifest", type=Path, required=True)
    parser.add_argument("--resolved-run-config", type=Path, required=True)
    parser.add_argument("--resolved-run-config-sha256", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--launch-command", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--status-receipt", type=Path, required=True)
    parser.add_argument("--summary-validation", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReceiptBuildError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def read_object(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptBuildError(f"cannot read {source} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ReceiptBuildError(f"{source} must be a JSON object")
    return payload


def require_file(path: Path, *, source: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReceiptBuildError(f"missing {source}: {path}") from error
    if not resolved.is_file():
        raise ReceiptBuildError(f"{source} must be a file: {resolved}")
    return resolved


def require_dir(path: Path, *, source: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReceiptBuildError(f"missing {source}: {path}") from error
    if not resolved.is_dir():
        raise ReceiptBuildError(f"{source} must be a directory: {resolved}")
    return resolved


def require_sha256(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReceiptBuildError(f"{source} is not a SHA-256 digest")
    return value


def frozen_hashes(value: Any, *, source: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ReceiptBuildError(f"{source} must be a non-empty object")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name or name in result:
            raise ReceiptBuildError(f"{source} contains a malformed name")
        result[name] = require_sha256(digest, source=f"{source}.{name}")
    return result


def config_hash_bindings(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if config.get("schema_id") != "qwen3vl_p1_resolved_run_config_v1":
        raise ReceiptBuildError("resolved run config schema drifted")
    if (
        config.get("configuration_status") != "frozen"
        or config.get("execution_readiness") != "execution_ready"
        or config.get("p1_execution_authorized") is not True
    ):
        raise ReceiptBuildError(
            "resolved run config is not frozen, execution-ready and authorized"
        )
    bindings = config.get("freeze_commit_hash_bindings")
    if not isinstance(bindings, Mapping) or bindings.get("status") != "frozen":
        raise ReceiptBuildError("resolved run config hash bindings are not frozen")
    return (
        frozen_hashes(
            bindings.get("model_files_sha256"),
            source="resolved config model_files_sha256",
        ),
        frozen_hashes(
            bindings.get("tool_files_sha256"),
            source="resolved config tool_files_sha256",
        ),
        frozen_hashes(
            bindings.get("worker_sha256"),
            source="resolved config worker_sha256",
        ),
    )


def amendment_model_hashes(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        value = {"model.safetensors": value}
    return frozen_hashes(value, source="P1 amendment model_hash")


def require_nonnegative_int(value: Any, *, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptBuildError(f"{source} must be a non-negative integer")
    return value


def read_key_value(path: Path, *, source: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ReceiptBuildError(f"cannot read {source} {path}: {error}") from error
    result: dict[str, str] = {}
    for index, line in enumerate(lines, 1):
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ReceiptBuildError(f"malformed {source} line {index}")
        result[key] = value
    return result


def artifact_binding(path: Path) -> dict[str, Any]:
    resolved = require_file(path, source="receipt artifact")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def git_evidence(
    worktree: Path,
    expected_freeze_commit: str,
    expected_implementation_commit: str,
) -> tuple[Path, str]:
    root = require_dir(worktree, source="run worktree")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD^"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReceiptBuildError("run worktree is not Git-verifiable") from error
    if head != expected_freeze_commit:
        raise ReceiptBuildError("run worktree HEAD differs from the runtime commit")
    if parent != expected_implementation_commit:
        raise ReceiptBuildError(
            "run worktree freeze commit is not a direct child of the implementation commit"
        )
    if status:
        raise ReceiptBuildError("complete receipt requires a clean run worktree")
    return root, hashlib.sha256(status).hexdigest()


def validate_identity(args: argparse.Namespace) -> tuple[str, list[str], dict[str, Any]]:
    if GIT_SHA_RE.fullmatch(args.runtime_commit) is None:
        raise ReceiptBuildError("runtime commit is malformed")
    if args.harness_commit != args.runtime_commit:
        raise ReceiptBuildError("runtime and harness commits must be identical")
    if GIT_SHA_RE.fullmatch(args.implementation_commit) is None:
        raise ReceiptBuildError("implementation commit is malformed")
    if args.implementation_commit == args.runtime_commit:
        raise ReceiptBuildError(
            "implementation and freeze/runtime commits must remain distinct"
        )
    expected_id, expected_samples = PROFILES[args.evaluation_profile]
    samples = list(args.expected_samples)
    if args.evaluation_id != expected_id or samples != list(expected_samples):
        raise ReceiptBuildError("evaluation profile, ID or sample order drifted")
    arm_contract = ARM_CONTRACTS[args.arm]
    if args.experiment_arm != arm_contract["experiment_arm_id"]:
        raise ReceiptBuildError("experiment arm ID drifted")
    if args.harness_protocol_id != arm_contract["harness_protocol_id"]:
        raise ReceiptBuildError("harness protocol ID drifted")
    require_sha256(args.worker_sha256, source="worker SHA-256")
    return expected_id, samples, arm_contract


def validate_source_receipts(
    args: argparse.Namespace,
    *,
    output_parent: Path,
    config_hash: str,
    amendment_hash: str,
) -> None:
    submission = read_key_value(args.submission_receipt, source="submission receipt")
    status = read_key_value(args.status_receipt, source="status receipt")
    expected_common = {
        "job_id": str(args.job_id),
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "samples": " ".join(args.expected_samples),
        "arm": args.arm,
        "experiment_arm_id": args.experiment_arm,
        "harness_protocol_id": args.harness_protocol_id,
        "worker_sha256": args.worker_sha256,
    }
    for name, payload in (("submission", submission), ("status", status)):
        for key, expected in expected_common.items():
            if payload.get(key) != expected:
                raise ReceiptBuildError(f"{name} receipt {key} drifted")
    if submission.get("submission_status") != "submitted":
        raise ReceiptBuildError("submission receipt is not submitted")
    if submission.get("expected_commit") != args.runtime_commit:
        raise ReceiptBuildError("submission receipt commit drifted")
    if submission.get("implementation_commit") != args.implementation_commit:
        raise ReceiptBuildError("submission receipt implementation commit drifted")
    if submission.get("config_sha256") != config_hash:
        raise ReceiptBuildError("submission receipt config hash drifted")
    if submission.get("amendment_canonical_sha256") != amendment_hash:
        raise ReceiptBuildError("submission amendment canonical SHA-256 drifted")
    if submission.get("partition") != "gre" or submission.get("node") != "gpu48":
        raise ReceiptBuildError("submission placement is not frozen")
    if submission.get("output") != str(output_parent):
        raise ReceiptBuildError("submission output path drifted")
    if status.get("status") != "success" or status.get("exit_code") != "0":
        raise ReceiptBuildError("status receipt is not a successful terminal record")
    if status.get("commit") != args.runtime_commit:
        raise ReceiptBuildError("status receipt commit drifted")
    if status.get("path") != str(output_parent):
        raise ReceiptBuildError("status receipt output path drifted")
    try:
        elapsed = int(status.get("elapsed_seconds", ""))
    except ValueError as error:
        raise ReceiptBuildError("status elapsed_seconds is malformed") from error
    if elapsed < 0 or not status.get("result") or status.get("error"):
        raise ReceiptBuildError("status receipt does not contain a clean result")
    expected_validation = (
        "status=valid;"
        f"profile={args.evaluation_profile};"
        f"arm={args.experiment_arm};"
        f"scored={len(args.expected_samples)}/{len(args.expected_samples)}"
    )
    try:
        observed_validation = args.summary_validation.read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, UnicodeError) as error:
        raise ReceiptBuildError("summary validation receipt is unreadable") from error
    if observed_validation != expected_validation:
        raise ReceiptBuildError("summary validation receipt is not identity-bound")


def normalized_counters(value: Any, *, source: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(BUDGET_COUNTERS):
        raise ReceiptBuildError(f"{source} differs from the frozen counter set")
    return {
        key: require_nonnegative_int(value[key], source=f"{source}.{key}")
        for key in BUDGET_COUNTERS
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    _, samples, arm_contract = validate_identity(args)
    output_parent = require_dir(args.output_parent, source="output parent")
    output = args.output.resolve()
    if output.parent != output_parent:
        raise ReceiptBuildError("receipt output is not directly under output-parent")
    if output.exists():
        raise ReceiptBuildError(f"evaluation receipt already exists: {output}")
    run_dir = require_dir(args.run_dir, source="run directory")
    if run_dir.parent != output_parent:
        raise ReceiptBuildError("run directory is not directly under output-parent")
    logs_dir = require_dir(output_parent / "logs", source="output logs directory")

    config = require_file(args.resolved_run_config, source="resolved run config")
    config_hash = require_sha256(
        args.resolved_run_config_sha256, source="resolved config SHA-256"
    )
    if sha256_file(config) != config_hash:
        raise ReceiptBuildError("resolved run config hash drifted")
    config_payload = read_object(config, source="resolved run config")
    frozen_models, frozen_tools, frozen_workers = config_hash_bindings(
        config_payload
    )
    if frozen_workers.get("evaluation") != args.worker_sha256:
        raise ReceiptBuildError(
            "resolved config evaluation worker hash drifted"
        )
    if frozen_tools.get("harness_worker") != args.worker_sha256:
        raise ReceiptBuildError(
            "resolved config harness-worker hash drifted"
        )
    config_paths = config_payload.get("paths")
    if not isinstance(config_paths, Mapping):
        raise ReceiptBuildError("resolved run config lacks paths")
    amendment_relative = config_paths.get("p1_amendment")
    if not isinstance(amendment_relative, str) or not amendment_relative:
        raise ReceiptBuildError("resolved run config lacks the P1 amendment path")
    amendment_path = require_file(
        config.parents[3] / amendment_relative,
        source="P1 amendment",
    )
    amendment = read_object(amendment_path, source="P1 amendment")
    if (
        amendment.get("schema_id") != "qwen3vl_p1_execution_amendment_v1"
        or amendment.get("record_status") != "frozen"
        or amendment.get("execution_readiness") != "execution_ready"
        or amendment.get("p1_execution_authorized") is not True
    ):
        raise ReceiptBuildError("P1 amendment is not frozen and execution-ready")
    if amendment.get("implementation_commit") != args.implementation_commit:
        raise ReceiptBuildError("P1 amendment implementation commit drifted")
    if Path(str(amendment.get("clean_run_worktree", ""))).resolve() != args.run_worktree.resolve():
        raise ReceiptBuildError("P1 amendment run-worktree binding drifted")
    if amendment.get("resolved_run_config_sha256") != config_hash:
        raise ReceiptBuildError("P1 amendment resolved-config hash drifted")
    if amendment_model_hashes(amendment.get("model_hash")) != frozen_models:
        raise ReceiptBuildError("P1 amendment model hashes differ from config")
    if frozen_hashes(
        amendment.get("tool_hashes"), source="P1 amendment tool_hashes"
    ) != frozen_tools:
        raise ReceiptBuildError("P1 amendment tool hashes differ from config")
    amendment_workers = frozen_hashes(
        amendment.get("worker_sha256"),
        source="P1 amendment worker_sha256",
    )
    if amendment_workers != frozen_workers or set(amendment_workers) != {
        "cpu_fixtures",
        "evaluation",
        "dependency_guard",
    }:
        raise ReceiptBuildError(
            "P1 amendment worker hashes differ from config exact mapping"
        )
    if amendment_workers["evaluation"] != args.worker_sha256:
        raise ReceiptBuildError(
            "P1 amendment evaluation worker hash drifted"
        )
    amendment_hash = canonical_sha256(amendment)
    evaluation_path = require_file(
        args.evaluation_input_manifest, source="evaluation input manifest"
    )
    evaluation = read_object(evaluation_path, source="evaluation input manifest")
    evaluation_hash = require_sha256(
        args.evaluation_input_manifest_canonical_sha256,
        source="evaluation input manifest canonical SHA-256",
    )
    if canonical_sha256(evaluation) != evaluation_hash:
        raise ReceiptBuildError("evaluation input manifest canonical hash drifted")

    baseline = require_file(
        output_parent / "baseline_summary.json", source="baseline summary"
    )
    protocol_path = require_file(
        output_parent / "protocol_summary.json", source="protocol summary"
    )
    audit_path = require_file(
        run_dir / "p1_harness_audit.json", source="P1 harness audit"
    )
    runtime_path = require_file(
        args.runtime_input_manifest, source="runtime input manifest"
    )
    rollout_path = require_file(
        run_dir / "rollout_complete.json", source="rollout marker"
    )
    metrics_path = require_file(
        run_dir / "per_sample_metrics.json", source="offline scorer input"
    )
    run_manifest = require_file(args.run_manifest, source="run manifest")
    launch = require_file(args.launch_command, source="launch command")
    submission = require_file(args.submission_receipt, source="submission receipt")
    status = require_file(args.status_receipt, source="status receipt")
    summary_validation = require_file(
        args.summary_validation, source="summary validation"
    )

    for name, path in (
        ("baseline summary", baseline),
        ("protocol summary", protocol_path),
        ("runtime input manifest", runtime_path),
        ("launch command", launch),
        ("submission receipt", submission),
        ("summary validation", summary_validation),
    ):
        if path.parent != output_parent:
            raise ReceiptBuildError(f"{name} is outside output-parent")
    if status.parent != logs_dir:
        raise ReceiptBuildError("status receipt is outside output-parent/logs")
    for name, path in (
        ("P1 harness audit", audit_path),
        ("rollout marker", rollout_path),
        ("offline scorer input", metrics_path),
        ("run manifest", run_manifest),
    ):
        if path.parent != run_dir:
            raise ReceiptBuildError(f"{name} is outside run-dir")

    summary = read_object(baseline, source="baseline summary")
    protocol = read_object(protocol_path, source="protocol summary")
    audit = read_object(audit_path, source="P1 harness audit")
    runtime = read_object(runtime_path, source="runtime input manifest")
    rollout = read_object(rollout_path, source="rollout marker")

    expected_identity_fields = {
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "experiment_arm": args.experiment_arm,
        "harness_protocol_id": args.harness_protocol_id,
        "job_id": str(args.job_id),
        "runtime_commit": args.runtime_commit,
        "harness_commit": args.harness_commit,
        "sample_order": samples,
    }
    if summary.get("summary_schema_version", summary.get("schema_id")) != SUMMARY_SCHEMA_ID:
        raise ReceiptBuildError("baseline summary is not schema v2")
    if summary.get("status") != "completed" or summary.get("model_state") != "base":
        raise ReceiptBuildError("baseline summary is not a completed base-model record")
    for key, expected in expected_identity_fields.items():
        if summary.get(key) != expected:
            raise ReceiptBuildError(f"baseline summary {key} drifted")
    if summary.get("requested_samples") != len(samples) or summary.get(
        "scored_samples"
    ) != len(samples):
        raise ReceiptBuildError("baseline summary requested/scored gate failed")
    for key, expected in expected_identity_fields.items():
        if key == "harness_protocol_id":
            observed = protocol.get(key)
        else:
            observed = protocol.get(key)
        if observed != expected:
            raise ReceiptBuildError(f"protocol summary {key} drifted")
    if protocol.get("status") != "completed":
        raise ReceiptBuildError("protocol summary is not completed")

    p1 = summary.get("p1_contract")
    if not isinstance(p1, Mapping):
        raise ReceiptBuildError("baseline summary lacks P1 contract")
    expected_p1 = {
        "schema_version": "qwen3vl_e2e_summary_p1_contract_v1",
        "source_audit_schema_id": P1_AUDIT_SCHEMA_ID,
        "factor_id": FACTOR_ID,
        "arm": args.arm,
        "harness_protocol_id": args.harness_protocol_id,
        "experiment_arm": args.experiment_arm,
        "terminal_review": arm_contract["terminal_review"],
        "model_io_protocol": MODEL_IO_PROTOCOL,
        "gt_loaded_before_rollout_complete": False,
    }
    for key, expected in expected_p1.items():
        if p1.get(key) != expected:
            raise ReceiptBuildError(f"baseline summary P1 {key} drifted")
    hard_gates = {
        "requested_scored_mismatch_count": 0,
        "pre_rollout_gt_violation_count": 0,
        "safety_boundary_relaxation_count": 0,
        "tool_failure_count": 0,
        "all_passed": True,
    }
    if p1.get("hard_gates") != hard_gates:
        raise ReceiptBuildError("baseline summary P1 hard gates failed")

    configured = {
        **CONFIGURED_BUDGETS,
        "max_verifier_rounds": arm_contract["max_verifier_rounds"],
    }
    if p1.get("configured_budgets") != configured:
        raise ReceiptBuildError("baseline summary configured budgets drifted")
    totals = normalized_counters(
        p1.get("budget_accounting_totals"), source="P1 budget totals"
    )
    per_sample_raw = p1.get("per_sample")
    if not isinstance(per_sample_raw, Mapping) or list(per_sample_raw) != samples:
        raise ReceiptBuildError("baseline summary P1 sample order drifted")
    per_sample: dict[str, dict[str, int]] = {}
    reconstructed = {key: 0 for key in BUDGET_COUNTERS}
    for sample in samples:
        record = per_sample_raw[sample]
        if not isinstance(record, Mapping):
            raise ReceiptBuildError(f"baseline summary P1 sample is malformed: {sample}")
        counters = normalized_counters(
            record.get("budget_accounting"), source=f"P1 budget {sample}"
        )
        per_sample[sample] = counters
        for key in BUDGET_COUNTERS:
            reconstructed[key] += counters[key]
    if reconstructed != totals:
        raise ReceiptBuildError("P1 per-sample counters do not sum to totals")

    actions = summary.get("actions")
    if not isinstance(actions, Mapping):
        raise ReceiptBuildError("baseline summary lacks episode totals")
    episode_totals = {
        key: require_nonnegative_int(actions.get(key), source=f"actions.{key}")
        for key in EPISODE_TOTALS
    }
    if episode_totals["candidates"] != (
        episode_totals["accepted"] + episode_totals["rejected"]
    ):
        raise ReceiptBuildError("episode candidate totals are inconsistent")
    if episode_totals["successful_tool_actions"] != episode_totals["tool_actions"]:
        raise ReceiptBuildError("episode tool totals contain a failure")

    if audit.get("schema_id") != P1_AUDIT_SCHEMA_ID:
        raise ReceiptBuildError("P1 harness audit schema drifted")
    if audit.get("totals") != totals or audit.get("requested_sample_ids") != samples:
        raise ReceiptBuildError("P1 harness audit counters or sample order drifted")
    if any(
        audit.get(key) != 0
        for key in (
            "gt_boundary_violation_count",
            "safety_boundary_relaxation_count",
            "tool_failure_count",
        )
    ):
        raise ReceiptBuildError("P1 harness audit records a hard-gate failure")
    if rollout.get("execution_status") != "completed" or rollout.get("gt_loaded") is not False:
        raise ReceiptBuildError("rollout marker does not prove GT-free completion")

    model_path = require_dir(args.model_path, source="base model")
    runtime_expectations = {
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "sample_order": samples,
        "terminal_review_arm": args.arm,
        "experiment_arm": args.experiment_arm,
        "harness_protocol_id": args.harness_protocol_id,
        "protocol_version": MODEL_IO_PROTOCOL,
        "slurm_job_id": str(args.job_id),
        "model_path": str(model_path),
        "gt_hashed_after_rollout": True,
        "executed_worker_sha256": args.worker_sha256,
    }
    for key, expected in runtime_expectations.items():
        if runtime.get(key) != expected:
            raise ReceiptBuildError(f"runtime input manifest {key} drifted")
    runtime_config = runtime.get("resolved_run_config")
    if not isinstance(runtime_config, Mapping) or runtime_config.get("path") != str(config):
        raise ReceiptBuildError("runtime resolved-config path drifted")
    if runtime_config.get("sha256") != config_hash:
        raise ReceiptBuildError("runtime resolved-config hash drifted")
    runtime_evaluation = runtime.get("evaluation_input_manifest")
    if not isinstance(runtime_evaluation, Mapping) or runtime_evaluation.get(
        "path"
    ) != str(evaluation_path):
        raise ReceiptBuildError("runtime evaluation-manifest path drifted")
    if runtime_evaluation.get("canonical_sha256") != evaluation_hash:
        raise ReceiptBuildError("runtime evaluation-manifest hash drifted")
    model_files = runtime.get("model_files_sha256")
    if not isinstance(model_files, Mapping):
        raise ReceiptBuildError("runtime input manifest lacks model hashes")
    model_hash = require_sha256(
        model_files.get("model.safetensors"), source="base model SHA-256"
    )
    model_file = require_file(
        model_path / "model.safetensors", source="base model weights"
    )
    if sha256_file(model_file) != model_hash:
        raise ReceiptBuildError("base model hash differs from runtime manifest")
    if {name: model_files.get(name) for name in frozen_models} != frozen_models:
        raise ReceiptBuildError("runtime model subset differs from frozen config")
    for name, expected in frozen_models.items():
        if sha256_file(
            require_file(model_path / name, source=f"frozen model file {name}")
        ) != expected:
            raise ReceiptBuildError(f"live model file drifted: {name}")
    tool_files = runtime.get("tool_files_sha256")
    if not isinstance(tool_files, Mapping) or not tool_files:
        raise ReceiptBuildError("runtime input manifest lacks tool hashes")
    tool_hashes = {
        str(name): require_sha256(digest, source=f"runtime tool {name}")
        for name, digest in tool_files.items()
    }
    if tool_hashes != frozen_tools:
        raise ReceiptBuildError("runtime tool hashes differ from frozen config")
    if tool_hashes.get("harness_worker") != args.worker_sha256:
        raise ReceiptBuildError("runtime harness-worker hash drifted")
    tool_paths = runtime.get("tool_file_paths")
    if not isinstance(tool_paths, Mapping) or set(tool_paths) != set(frozen_tools):
        raise ReceiptBuildError("runtime tool paths differ from frozen tool set")
    for name, expected in frozen_tools.items():
        raw_path = tool_paths.get(name)
        if not isinstance(raw_path, str) or not raw_path:
            raise ReceiptBuildError(f"runtime tool path is malformed: {name}")
        if sha256_file(
            require_file(Path(raw_path), source=f"runtime tool file {name}")
        ) != expected:
            raise ReceiptBuildError(f"live runtime tool drifted: {name}")

    worktree, worktree_hash = git_evidence(
        args.run_worktree,
        args.runtime_commit,
        args.implementation_commit,
    )
    validate_source_receipts(
        args,
        output_parent=output_parent,
        config_hash=config_hash,
        amendment_hash=amendment_hash,
    )

    identity = {
        "factor_id": FACTOR_ID,
        "arm": args.arm,
        "experiment_arm_id": args.experiment_arm,
        "harness_protocol_id": args.harness_protocol_id,
        "model_io_protocol": MODEL_IO_PROTOCOL,
        "evaluation_profile": args.evaluation_profile,
        "evaluation_id": args.evaluation_id,
        "sample_order": samples,
        "runtime_commit": args.runtime_commit,
        "harness_commit": args.harness_commit,
        "job_id": str(args.job_id),
        "model_state": "base",
    }
    provenance = {
        "implementation_commit": args.implementation_commit,
        "freeze_commit": args.runtime_commit,
        "run_worktree": str(worktree),
        "git_dirty": False,
        "git_worktree_sha256": worktree_hash,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "tool_sha256": tool_hashes,
        "worker_sha256": args.worker_sha256,
        "evaluation_input_manifest_canonical_sha256": evaluation_hash,
    }
    budget_accounting = {
        "configured": configured,
        "terminal_review_scheduler": {
            "totals": totals,
            "per_sample": per_sample,
        },
        "episode_totals": episode_totals,
    }
    gates = {
        "gt_loaded_before_rollout_complete": False,
        "pre_rollout_gt_access_count": 0,
        "safety_boundary_relaxation_count": 0,
        "requested_sample_count": len(samples),
        "scored_sample_count": len(samples),
        "requested_scored_mismatch_count": 0,
        "tool_failure_count": 0,
        "offline_scoring_after_rollout_complete": True,
        "summary_validator_passed": True,
    }
    artifact_paths = {
        "baseline_summary": baseline,
        "protocol_summary": protocol_path,
        "p1_harness_audit": audit_path,
        "resolved_run_config": config,
        "p1_amendment": amendment_path,
        "evaluation_input_manifest": evaluation_path,
        "runtime_input_manifest": runtime_path,
        "run_manifest": run_manifest,
        "launch_command": launch,
        "rollout_marker": rollout_path,
        "offline_scorer_input": metrics_path,
        "submission_receipt": submission,
        "status_receipt": status,
        "summary_validation": summary_validation,
    }
    if tuple(artifact_paths) != ARTIFACT_NAMES:
        raise AssertionError("internal artifact order drifted")
    artifacts = {
        name: artifact_binding(path) for name, path in artifact_paths.items()
    }
    return {
        "schema_id": SCHEMA_ID,
        "status": "complete",
        "terminal_state": "success",
        "identity": identity,
        "provenance": provenance,
        "budget_accounting": budget_accounting,
        "gates": gates,
        "artifacts": artifacts,
    }


def atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    if target.exists():
        raise ReceiptBuildError(f"evaluation receipt already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise ReceiptBuildError(f"evaluation receipt appeared during publish: {target}")
        os.link(temporary, target)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_payload(args)
        atomic_publish(args.output, payload)
    except (OSError, ReceiptBuildError) as error:
        print(f"qwen3vl_evaluation_receipt_v2_builder: {error}", file=sys.stderr)
        return 3
    print(
        "status=complete;"
        f"arm={payload['identity']['arm']};"
        f"profile={payload['identity']['evaluation_profile']};"
        f"job_id={payload['identity']['job_id']};"
        f"samples={len(payload['identity']['sample_order'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
