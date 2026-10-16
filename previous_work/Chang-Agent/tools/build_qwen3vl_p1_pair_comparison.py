#!/usr/bin/env python3
"""Build one hash-bound H0/H1 P1 comparison after both receipts validate.

This is an offline finalizer.  It never opens LEVIR source images or labels;
all diagnostic metrics come from the independently validated summaries bound
by the two evaluation receipts.  The smoke profile additionally enforces the
pre-registered live-H1 gate: exactly one fresh, non-duplicate round two must
have been observed and its final region protocol must be valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "qwen3vl_h0_h1_pair_comparison_v1"
PROFILES = {
    "p1_smoke1": ("DEV-SMOKE-1-H1-V1", ("test_1_1",)),
    "e2e3": (
        "DEV-CORE-3-V1",
        ("test_20_15", "test_1_1", "test_50_8"),
    ),
}
ARM_IDS = {
    "h0": "v12_1-h0-legacy-single-round-base-0802",
    "h1": "v12_1-h1-gtfree-terminal-review-base-0802",
}
PROTOCOL_IDS = {
    "h0": "qwen3vl_harness_h0_legacy_single_round_v1",
    "h1": "qwen3vl_harness_h1_gtfree_terminal_review_v1",
}
CONTROLLED_PROVENANCE_KEYS = (
    "implementation_commit",
    "freeze_commit",
    "run_worktree",
    "git_dirty",
    "git_worktree_sha256",
    "model_path",
    "model_sha256",
    "tool_sha256",
    "evaluation_input_manifest_canonical_sha256",
)
RUNTIME_MANIFEST_CONTROLLED_KEYS = (
    "evaluation_profile",
    "evaluation_id",
    "sample_order",
    "input_root",
    "files_sha256",
    "model_path",
    "model_files_sha256",
    "tool_files_sha256",
    "runtime_source_sha256",
)


class PairComparisonError(ValueError):
    """Raised when two artifacts are not the frozen P1 pair."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h0-receipt", type=Path, required=True)
    parser.add_argument("--h1-receipt", type=Path, required=True)
    parser.add_argument("--evaluation-profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--receipt-validator",
        type=Path,
        default=Path(__file__).with_name("validate_qwen3vl_evaluation_receipt.py"),
    )
    return parser.parse_args(argv)


def read_object(path: Path, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairComparisonError(f"cannot read {source} JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PairComparisonError(f"{source} must be a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PairComparisonError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise PairComparisonError(f"bound path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_receipt(path: Path, validator: Path) -> dict[str, Any]:
    receipt = path.resolve(strict=True)
    validator = validator.resolve(strict=True)
    completed = subprocess.run(
        [sys.executable, str(validator), "--receipt", str(receipt)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PairComparisonError(
            f"independent receipt validation failed for {receipt}: {detail}"
        )
    return read_object(receipt, source="evaluation receipt")


def require_mapping(value: Any, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairComparisonError(f"{source} must be an object")
    return value


def summary_for(receipt: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifacts = require_mapping(receipt.get("artifacts"), source="receipt.artifacts")
    summary_binding = require_mapping(
        artifacts.get("baseline_summary"), source="receipt baseline summary binding"
    )
    raw_path = summary_binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise PairComparisonError("receipt baseline summary path is missing")
    path = Path(raw_path).resolve(strict=True)
    if summary_binding.get("sha256") != sha256(path):
        raise PairComparisonError("receipt baseline summary binding drifted")
    return path, read_object(path, source="baseline summary")


def artifact_for(
    receipt: Mapping[str, Any], name: str, *, source: str
) -> tuple[Path, dict[str, Any] | None]:
    artifacts = require_mapping(receipt.get("artifacts"), source=f"{source}.artifacts")
    artifact = require_mapping(
        artifacts.get(name), source=f"{source}.artifacts.{name}"
    )
    path_text = artifact.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise PairComparisonError(f"{source}.{name} path is missing")
    path = Path(path_text).resolve(strict=True)
    if artifact.get("sha256") != sha256(path):
        raise PairComparisonError(f"{source}.{name} hash binding drifted")
    payload = (
        read_object(path, source=f"{source}.{name}")
        if name in {"evaluation_input_manifest", "runtime_input_manifest"}
        else None
    )
    return path, payload


def receipt_gate_checks(
    receipt: Mapping[str, Any], *, sample_count: int, source: str
) -> dict[str, bool]:
    gates = require_mapping(receipt.get("gates"), source=f"{source}.gates")
    checks = {
        f"{source}_gt_loaded_before_rollout_complete_false": (
            gates.get("gt_loaded_before_rollout_complete") is False
        ),
        f"{source}_pre_rollout_gt_access_zero": (
            gates.get("pre_rollout_gt_access_count") == 0
        ),
        f"{source}_safety_relaxation_zero": (
            gates.get("safety_boundary_relaxation_count") == 0
        ),
        f"{source}_requested_count_exact": (
            gates.get("requested_sample_count") == sample_count
        ),
        f"{source}_scored_count_exact": (
            gates.get("scored_sample_count") == sample_count
        ),
        f"{source}_requested_scored_mismatch_zero": (
            gates.get("requested_scored_mismatch_count") == 0
        ),
        f"{source}_tool_failure_zero": gates.get("tool_failure_count") == 0,
        f"{source}_offline_scoring_after_rollout": (
            gates.get("offline_scoring_after_rollout_complete") is True
        ),
        f"{source}_summary_validator_passed": (
            gates.get("summary_validator_passed") is True
        ),
    }
    return checks


def smoke_round_two_checks(p1: Mapping[str, Any], sample: str) -> dict[str, bool]:
    samples = require_mapping(p1.get("per_sample"), source="h1.p1_contract.per_sample")
    record = require_mapping(samples.get(sample), source=f"h1.p1_contract.{sample}")
    terminal = require_mapping(
        record.get("terminal_review"), source=f"h1.p1_contract.{sample}.terminal_review"
    )
    schema_valid_outcomes = {
        "finish",
        "unique_safe_plan",
        "plan_null",
        "unresolved",
        "unsafe_or_nonunique_plan",
    }
    calls = terminal.get("backend_calls")
    return {
        "h1_smoke_round_two_invoked": terminal.get("round_two_invoked") is True,
        "h1_smoke_round_two_count_one": terminal.get("round_two_count") == 1,
        "h1_smoke_round_index_two": terminal.get("round_index") == 2,
        "h1_smoke_round_two_schema_valid": (
            terminal.get("observed_outcome") in schema_valid_outcomes
            and terminal.get("contract_violation") is None
        ),
        "h1_smoke_fresh_backend_context": (
            terminal.get("fresh_backend_context") is True
        ),
        "h1_smoke_nonduplicate_prompt_or_region": (
            terminal.get("nonduplicate_prompt_or_region") is True
        ),
        "h1_smoke_gt_free": terminal.get("gt_free") is True,
        "h1_smoke_no_forbidden_trigger_inputs": (
            terminal.get("forbidden_trigger_inputs_present") == []
        ),
        "h1_smoke_backend_call_retained": (
            isinstance(calls, list) and len(calls) > 0
        ),
    }


def metric_slice(summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = require_mapping(summary.get("aggregate"), source="summary.aggregate")
    initial = require_mapping(aggregate.get("initial"), source="aggregate.initial")
    selected = require_mapping(aggregate.get("selected"), source="aggregate.selected")
    return {
        "initial_iou": initial.get("iou"),
        "initial_f1": initial.get("f1"),
        "selected_iou": selected.get("iou"),
        "selected_f1": selected.get("f1"),
        "delta_iou": aggregate.get("delta_iou"),
        "delta_f1": aggregate.get("delta_f1"),
    }


def build_payload(
    *,
    h0_path: Path,
    h1_path: Path,
    h0: Mapping[str, Any],
    h1: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    evaluation_id, sample_order = PROFILES[profile]
    receipts = {"h0": h0, "h1": h1}
    identities: dict[str, Mapping[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    summary_paths: dict[str, Path] = {}
    for arm, receipt in receipts.items():
        identity = require_mapping(receipt.get("identity"), source=f"{arm}.identity")
        expected = {
            "arm": arm,
            "experiment_arm_id": ARM_IDS[arm],
            "harness_protocol_id": PROTOCOL_IDS[arm],
            "evaluation_profile": profile,
            "evaluation_id": evaluation_id,
            "sample_order": list(sample_order),
            "model_state": "base",
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise PairComparisonError(f"{arm} identity {key} is not frozen")
        identities[arm] = identity
        summary_paths[arm], summaries[arm] = summary_for(receipt)

    if identities["h0"].get("job_id") == identities["h1"].get("job_id"):
        raise PairComparisonError("H0 and H1 must be independent Slurm jobs")
    if identities["h0"].get("runtime_commit") != identities["h1"].get(
        "runtime_commit"
    ):
        raise PairComparisonError("paired arms do not use the same freeze/runtime commit")

    provenance = {
        arm: require_mapping(receipt.get("provenance"), source=f"{arm}.provenance")
        for arm, receipt in receipts.items()
    }
    controlled_checks: dict[str, bool] = {}
    for key in CONTROLLED_PROVENANCE_KEYS:
        controlled_checks[f"same_{key}"] = provenance["h0"].get(key) == provenance[
            "h1"
        ].get(key)
    if not all(controlled_checks.values()):
        failed = sorted(key for key, passed in controlled_checks.items() if not passed)
        raise PairComparisonError(f"paired controlled provenance differs: {failed}")

    evaluation_manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    runtime_manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for arm, receipt in receipts.items():
        evaluation_path, evaluation_payload = artifact_for(
            receipt, "evaluation_input_manifest", source=arm
        )
        runtime_path, runtime_payload = artifact_for(
            receipt, "runtime_input_manifest", source=arm
        )
        if evaluation_payload is None or runtime_payload is None:
            raise PairComparisonError(f"{arm} input manifest is not JSON")
        evaluation_manifests[arm] = (evaluation_path, evaluation_payload)
        runtime_manifests[arm] = (runtime_path, runtime_payload)
        # These two evidence files close important independent-review holes;
        # require them even though their contents are validated elsewhere.
        artifact_for(receipt, "run_manifest", source=arm)
        artifact_for(receipt, "summary_validation", source=arm)
    if binding(evaluation_manifests["h0"][0])["sha256"] != binding(
        evaluation_manifests["h1"][0]
    )["sha256"]:
        raise PairComparisonError("paired evaluation-input manifests differ")
    for key in RUNTIME_MANIFEST_CONTROLLED_KEYS:
        if key == "evaluation_profile":
            same = all(
                runtime_manifests[arm][1].get(key) == profile
                for arm in ("h0", "h1")
            )
        elif key == "evaluation_id":
            same = all(
                runtime_manifests[arm][1].get(key) == evaluation_id
                for arm in ("h0", "h1")
            )
        elif key == "sample_order":
            same = all(
                runtime_manifests[arm][1].get(key) == list(sample_order)
                for arm in ("h0", "h1")
            )
        else:
            same = runtime_manifests["h0"][1].get(key) == runtime_manifests["h1"][1].get(
                key
            )
        controlled_checks[f"same_runtime_input_{key}"] = same
    if not all(controlled_checks.values()):
        failed = sorted(key for key, passed in controlled_checks.items() if not passed)
        raise PairComparisonError(f"paired runtime inputs differ: {failed}")

    p1 = {
        arm: require_mapping(summary.get("p1_contract"), source=f"{arm}.p1_contract")
        for arm, summary in summaries.items()
    }
    valid_counts = {
        arm: p1[arm].get("final_region_protocol_valid_count") for arm in ("h0", "h1")
    }
    round_two_counts = {
        arm: p1[arm].get("round_two_invoked_count") for arm in ("h0", "h1")
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in valid_counts.values()):
        raise PairComparisonError("final-region protocol validity counts are malformed")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in round_two_counts.values()):
        raise PairComparisonError("round-two invocation counts are malformed")

    hard_gates = {
        "individual_receipts_validated": True,
        "controlled_provenance_identical": True,
        "requested_scored_exactly_once": all(
            summary.get("requested_samples") == len(sample_order)
            and summary.get("scored_samples") == len(sample_order)
            for summary in summaries.values()
        ),
        "h0_round_two_invocation_count_zero": round_two_counts["h0"] == 0,
        "h1_protocol_validity_delta_nonnegative": (
            valid_counts["h1"] - valid_counts["h0"] >= 0
        ),
        "h1_smoke_exactly_one_round_two": (
            round_two_counts["h1"] == 1 if profile == "p1_smoke1" else True
        ),
        "h1_smoke_final_region_protocol_valid": (
            valid_counts["h1"] == 1 if profile == "p1_smoke1" else True
        ),
    }
    for arm, receipt in receipts.items():
        hard_gates.update(
            receipt_gate_checks(receipt, sample_count=len(sample_order), source=arm)
        )
    if profile == "p1_smoke1":
        hard_gates.update(smoke_round_two_checks(p1["h1"], sample_order[0]))
    if not all(hard_gates.values()):
        failed = sorted(key for key, passed in hard_gates.items() if not passed)
        raise PairComparisonError(f"P1 pair hard gates failed: {failed}")

    return {
        "schema_id": SCHEMA_ID,
        "status": "complete",
        "evaluation_profile": profile,
        "evaluation_id": evaluation_id,
        "sample_order": list(sample_order),
        "implementation_commit": provenance["h0"].get("implementation_commit"),
        "freeze_commit": identities["h0"].get("runtime_commit"),
        "arms": {
            arm: {
                "receipt": binding(path),
                "baseline_summary": binding(summary_paths[arm]),
                "job_id": identities[arm].get("job_id"),
                "experiment_arm_id": ARM_IDS[arm],
                "harness_protocol_id": PROTOCOL_IDS[arm],
            }
            for arm, path in (("h0", h0_path), ("h1", h1_path))
        },
        "controlled_variable_checks": controlled_checks,
        "hard_gates": hard_gates,
        "comparison": {
            "h0": {
                **metric_slice(summaries["h0"]),
                "final_region_protocol_valid_count": valid_counts["h0"],
                "round_two_invoked_count": round_two_counts["h0"],
            },
            "h1": {
                **metric_slice(summaries["h1"]),
                "final_region_protocol_valid_count": valid_counts["h1"],
                "round_two_invoked_count": round_two_counts["h1"],
            },
            "final_region_protocol_validity_delta_h1_minus_h0": (
                valid_counts["h1"] - valid_counts["h0"]
            ),
            "diagnostic_delta_selected_iou_h1_minus_h0": round(
                float(metric_slice(summaries["h1"])["selected_iou"])
                - float(metric_slice(summaries["h0"])["selected_iou"]),
                8,
            ),
            "diagnostic_delta_selected_f1_h1_minus_h0": round(
                float(metric_slice(summaries["h1"])["selected_f1"])
                - float(metric_slice(summaries["h0"])["selected_f1"]),
                8,
            ),
        },
        "interpretation": (
            "This fixed diagnostic pair evaluates harness protocol and safety only; "
            "its IoU/F1 deltas are not evidence of LEVIR-CD generalization."
        ),
    }


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise PairComparisonError(f"comparison output already exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        h0_path = args.h0_receipt.resolve(strict=True)
        h1_path = args.h1_receipt.resolve(strict=True)
        h0 = validate_receipt(h0_path, args.receipt_validator)
        h1 = validate_receipt(h1_path, args.receipt_validator)
        payload = build_payload(
            h0_path=h0_path,
            h1_path=h1_path,
            h0=h0,
            h1=h1,
            profile=args.evaluation_profile,
        )
        atomic_write(args.output, payload)
    except (OSError, PairComparisonError) as error:
        print(f"qwen3vl_p1_pair_comparison: {error}", file=sys.stderr)
        return 3
    print(
        "status=complete;"
        f"profile={payload['evaluation_profile']};"
        f"samples={len(payload['sample_order'])};"
        "hard_gates=passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
