#!/usr/bin/env python3
"""Validate the independent, hash-bound frozen v12.1/P0 H0 golden receipt.

This module deliberately does not import the staged verifier, terminal-review
decorator, Environment, runner, or their test helpers.  The checked-in receipt
is an immutable behavioral oracle, not a value regenerated from the current
implementation.  Validation binds it to:

* exact behavior vectors embedded below;
* the completed P0 acceptance receipt (raw and canonical JSON digests);
* source blobs at the successful P0 validation commit; and
* raw/canonical receipt digests plus this validator's digest in the CPU fixture
  manifest.

Only JSON files and Git blobs are read.  No image, model, label, rollout, or
offline metric artifact is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_ID = "qwen3vl_h0_golden_fixture_receipt_v1"
MANIFEST_SCHEMA_ID = "qwen3vl_h0_h1_cpu_fixture_manifest_v1"
P0_RECEIPT_SCHEMA_ID = "qwen3vl_p0_acceptance_receipt_v1"
GOLDEN_RECEIPT_RELPATH = (
    "doc/training_pipeline/0802_third_try/"
    "QWEN3VL_H0_V12_1_GOLDEN_FIXTURE_RECEIPT_0802.json"
)
VALIDATOR_RELPATH = "tools/validate_qwen3vl_h0_golden_fixture.py"
P0_RECEIPT_RELPATH = (
    "doc/training_pipeline/0802_third_try/"
    "QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json"
)
FROZEN_P0_COMMIT = "e9fbcd690a9af2e91384231ced46555fe400d9ba"
FROZEN_P0_RAW_SHA256 = (
    "<hash omitted>"
)
FROZEN_P0_CANONICAL_SHA256 = (
    "<hash omitted>"
)
FROZEN_SOURCE_BLOBS = (
    (
        "change_agent/adapters/stage_backends.py",
        "<hash omitted>",
    ),
    (
        "change_agent/adapters/staged_verifier.py",
        "<hash omitted>",
    ),
    (
        "change_agent/environment.py",
        "<hash omitted>",
    ),
    (
        "change_agent/state.py",
        "<hash omitted>",
    ),
    (
        "change_agent/verifier_protocol.py",
        "<hash omitted>",
    ),
    (
        "change_agent/verifier_regions.py",
        "<hash omitted>",
    ),
    (
        "tests/test_staged_verifier.py",
        "<hash omitted>",
    ),
    (
        "tools/run_levir_change_agent.py",
        "<hash omitted>",
    ),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_CASE_IDS = ("initial_finish", "initial_plan_null")
FORBIDDEN_GT_MARKERS = (
    "label_cvt",
    "ground_truth",
    "ground truth",
    "oracle_mask",
    "intersection_over_union",
    "f1_score",
)

_PASS_CHECKLIST = {
    "evidence_sufficient": "pass",
    "target_class_only": "pass",
    "white_pixels_supported": "pass",
    "boundary_alignment": "pass",
    "internal_holes_absent": "pass",
    "changed_object_extent_complete": "pass",
    "fragment_artifacts_absent": "pass",
}
_MIXED_CHECKLIST = {
    **_PASS_CHECKLIST,
    "white_pixels_supported": "fail",
    "changed_object_extent_complete": "fail",
}
_COMMON_REGION_RECORD = {
    "region_id": "r0",
    "audit_kind": "present",
    "change_mask_state": "white",
    "temporal_difference_state": "present",
    "box_normalized_1000": [0, 0, 800, 800],
    "component_seed_normalized_1000": [400, 400],
    "t1_mask_pixels": 0,
    "t2_mask_pixels": 25,
    "component_area": 25,
    "editable_seed_white": {"t1": False, "t2": True},
    "allowed_actions": ["positive_point", "negative_point", "box"],
    "transition_mask_facts": {},
}
_COMMON_PAYLOAD_CONTRACT = {
    "stage": "audit",
    "region_id": "r0",
    "proposal_catalog_region_ids": ["r0"],
    "schema": "environment_ordered_compact_region_audit_v12_1",
    "target_class": "building",
    "visual_context": "hybrid",
    "terminal_review_present": False,
}
_COMMON_MODEL_CALLS = {
    "count": 1,
    "stages": ["audit"],
    "region_ids": ["r0"],
    "attempt_kinds": ["initial"],
    "schema_repair_attempt_count": 0,
    "network_retry_attempt_count": 0,
    "payload_contracts": [_COMMON_PAYLOAD_CONTRACT],
}
_COMMON_H0_ROUNDS = {
    "max_verifier_rounds": 1,
    "verifier_round_count": 1,
    "round_two_invocations": 0,
    "round_three_invocations": 0,
}


def _output(
    *,
    quality_score: float,
    error_type: str,
    suggested_action: str | None,
    feedback: str,
    accept: bool,
    localization_valid: bool,
    stop: bool,
) -> dict[str, Any]:
    return {
        "quality_score": quality_score,
        "progress_score": 0.0,
        "score_delta": 0.0,
        "comparison": "initial",
        "error_type": error_type,
        "target_view": "t2",
        "error_region": None,
        "coordinate_space": "normalized_0_1000",
        "suggested_action": suggested_action,
        "feedback": feedback,
        "accept": accept,
        "verifier_valid": True,
        "localization_valid": localization_valid,
        "stop": stop,
    }


EXPECTED_BEHAVIOR = {
    "initial_finish": {
        "region_order": ["r0"],
        "region_records": [_COMMON_REGION_RECORD],
        "model_calls": _COMMON_MODEL_CALLS,
        "runtime_diagnoses": [
            {
                "region_id": "r0",
                "error_type": "none",
                "target_view": None,
                "audit_checklist": _PASS_CHECKLIST,
                "summary": "Paired RGB and exact component are observable.",
            }
        ],
        "plan": {
            "region_id": None,
            "action": "finish",
            "target_view": None,
            "coordinate_normalized_1000": None,
            "box_normalized_1000": None,
        },
        "verifier_output": _output(
            quality_score=1.0,
            error_type="none",
            suggested_action="finish",
            feedback="All Environment audit regions passed the compact checklist.",
            accept=True,
            localization_valid=True,
            stop=True,
        ),
        "safety_gates": {
            "completion_gate_passed": True,
            "completion_gate_reason": (
                "all Environment audit regions are covered and diagnosed none"
            ),
            "regional_validation_errors": [],
        },
        "candidate_and_termination": {
            "planned_tool_action": None,
            "tool_action_attempt_count": 0,
            "candidate_state_count": 0,
            "legacy_stop_reason": "initial_verifier_authorized_finish",
            "episode_outcome": "converged",
        },
        "h0_rounds": _COMMON_H0_ROUNDS,
    },
    "initial_plan_null": {
        "region_order": ["r0"],
        "region_records": [_COMMON_REGION_RECORD],
        "model_calls": _COMMON_MODEL_CALLS,
        "runtime_diagnoses": [
            {
                "region_id": "r0",
                "error_type": "mixed_error",
                "target_view": None,
                "audit_checklist": _MIXED_CHECKLIST,
                "summary": "Paired RGB and exact component are observable.",
            }
        ],
        "plan": None,
        "verifier_output": _output(
            quality_score=0.6666666666666666,
            error_type="mixed_error",
            suggested_action=None,
            feedback=(
                "Paired RGB and exact component are observable.; completion gate: "
                "at least one diagnosed error remains"
            ),
            accept=False,
            localization_valid=False,
            stop=False,
        ),
        "safety_gates": {
            "completion_gate_passed": False,
            "completion_gate_reason": "at least one diagnosed error remains",
            "regional_validation_errors": [],
        },
        "candidate_and_termination": {
            "planned_tool_action": None,
            "tool_action_attempt_count": 0,
            "candidate_state_count": 0,
            "legacy_stop_reason": "verifier_no_executable_action",
            "episode_outcome": "no_executable_action",
        },
        "h0_rounds": _COMMON_H0_ROUNDS,
    },
}


class GoldenFixtureValidationError(ValueError):
    """Raised when the frozen independent oracle fails closed."""


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
        raise GoldenFixtureValidationError(
            f"cannot canonicalize JSON value: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise GoldenFixtureValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _object(value: Any, *, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoldenFixtureValidationError(f"{source} must be an object")
    return value


def _read_json_bytes(path: Path, *, source: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GoldenFixtureValidationError(
            f"cannot read {source} {path}: {error}"
        ) from error
    return dict(_object(value, source=source)), raw


def _string(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise GoldenFixtureValidationError(f"{source} must be a non-empty string")
    return value


def _digest(value: Any, *, source: str) -> str:
    result = _string(value, source=source)
    if not SHA256_RE.fullmatch(result):
        raise GoldenFixtureValidationError(f"{source} is not a SHA-256 digest")
    return result


def _commit(value: Any, *, source: str) -> str:
    result = _string(value, source=source)
    if not COMMIT_RE.fullmatch(result):
        raise GoldenFixtureValidationError(f"{source} is not a commit SHA")
    return result


def _repo_file(repo_root: Path, relative_path: str, *, source: str) -> Path:
    relative = Path(_string(relative_path, source=source))
    if relative.is_absolute() or ".." in relative.parts:
        raise GoldenFixtureValidationError(f"{source} must be repository-relative")
    root = repo_root.resolve(strict=True)
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as error:
        raise GoldenFixtureValidationError(
            f"{source} does not resolve to a repository file"
        ) from error
    if not candidate.is_file():
        raise GoldenFixtureValidationError(f"{source} is not a file")
    return candidate


def _reject_forbidden_gt(value: Any, *, source: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_forbidden_gt(str(key), source=f"{source}.<key>")
            _reject_forbidden_gt(item, source=f"{source}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_gt(item, source=f"{source}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if any(marker in lowered for marker in FORBIDDEN_GT_MARKERS):
            raise GoldenFixtureValidationError(
                f"{source} contains forbidden GT/offline-metric material"
            )


def _git_blob_sha256(repo_root: Path, commit: str, relative_path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GoldenFixtureValidationError(
            f"cannot read frozen source blob {commit}:{relative_path}"
        ) from error
    return sha256_bytes(result.stdout)


def _validate_exact_receipt_payload(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_id") != SCHEMA_ID:
        raise GoldenFixtureValidationError("H0 golden receipt schema_id mismatch")
    if receipt.get("status") != "frozen":
        raise GoldenFixtureValidationError("H0 golden receipt is not frozen")
    if receipt.get("artifact_role") != "read_only_behavioral_oracle":
        raise GoldenFixtureValidationError("H0 golden receipt is not read-only")
    if receipt.get("model_io_protocol") != "v12.1":
        raise GoldenFixtureValidationError("H0 golden receipt protocol mismatch")
    if (
        receipt.get("harness_protocol_id")
        != "qwen3vl_harness_h0_legacy_single_round_v1"
    ):
        raise GoldenFixtureValidationError("H0 golden receipt harness mismatch")
    provenance = _object(
        receipt.get("oracle_provenance"), source="oracle_provenance"
    )
    expected_provenance = {
        "origin": "frozen_v12_1_p0_behavioral_snapshot",
        "current_h0_decorator_used_as_oracle": False,
        "current_legacy_verifier_used_as_oracle": False,
        "regeneration_allowed": False,
        "reference_annotations_used": False,
    }
    if dict(provenance) != expected_provenance:
        raise GoldenFixtureValidationError("golden oracle provenance drifted")

    p0 = _object(
        receipt.get("p0_acceptance_receipt"), source="p0_acceptance_receipt"
    )
    if dict(p0) != {
        "path": P0_RECEIPT_RELPATH,
        "raw_sha256": FROZEN_P0_RAW_SHA256,
        "canonical_sha256": FROZEN_P0_CANONICAL_SHA256,
    }:
        raise GoldenFixtureValidationError("P0 acceptance receipt binding drifted")

    anchor = _object(receipt.get("source_anchor"), source="source_anchor")
    expected_files = [
        {"path": path, "sha256": digest}
        for path, digest in FROZEN_SOURCE_BLOBS
    ]
    if anchor.get("commit") != FROZEN_P0_COMMIT:
        raise GoldenFixtureValidationError("frozen P0 source commit drifted")
    if anchor.get("files") != expected_files:
        raise GoldenFixtureValidationError("frozen P0 source-blob binding drifted")

    fixture = _object(receipt.get("fixture_definition"), source="fixture_definition")
    expected_fixture = {
        "fixture_id": "qwen3vl_v12_1_single_region_atomic_audit_v1",
        "state": {
            "image_shape_hwc": [16, 16, 3],
            "query": "building",
            "t1_image_fill": 0,
            "t2_image_fill": 0,
            "t2_image_patch_xyxy_exclusive": [4, 4, 9, 9],
            "t2_image_patch_value": 180,
            "t1_mask_rectangles_xyxy_exclusive": [],
            "t2_mask_rectangles_xyxy_exclusive": [[4, 4, 9, 9]],
            "change_mask_rule": "logical_xor_t1_t2",
        },
        "region_builder": {
            "factory": "attach_verifier_regions",
            "max_regions": 6,
            "min_component_area": 1,
            "padding_ratio": 0.25,
            "empty_mask_scan_grid_size": 0,
        },
        "verifier": {
            "factory": "StagedQwenVerifier",
            "max_regions": 6,
            "max_stage_attempts_including_initial": 2,
            "visual_context": "hybrid",
        },
    }
    if dict(fixture) != expected_fixture:
        raise GoldenFixtureValidationError("frozen fixture definition drifted")

    cases = receipt.get("fixture_cases")
    if not isinstance(cases, list) or len(cases) != len(EXPECTED_CASE_IDS):
        raise GoldenFixtureValidationError("golden fixture case count drifted")
    if [item.get("case_id") for item in cases if isinstance(item, Mapping)] != list(
        EXPECTED_CASE_IDS
    ):
        raise GoldenFixtureValidationError("golden fixture cases/order drifted")
    expected_sources = {
        "initial_finish": (
            "tests/test_staged_verifier.py::"
            "test_correct_appearance_change_can_finish_without_action_plan"
        ),
        "initial_plan_null": (
            "tests/test_staged_verifier.py::"
            "test_initial_without_plan_cannot_authorize_external_candidate"
        ),
    }
    expected_checklists = {
        "initial_finish": _PASS_CHECKLIST,
        "initial_plan_null": _MIXED_CHECKLIST,
    }
    for index, value in enumerate(cases):
        case = _object(value, source=f"fixture_cases[{index}]")
        case_id = EXPECTED_CASE_IDS[index]
        if set(case) != {
            "case_id",
            "source_test_id",
            "backend_audit_response",
            "expected_behavior",
        }:
            raise GoldenFixtureValidationError(f"{case_id} fields drifted")
        if case.get("case_id") != case_id:
            raise GoldenFixtureValidationError(f"{case_id} identity drifted")
        if case.get("source_test_id") != expected_sources[case_id]:
            raise GoldenFixtureValidationError(f"{case_id} source contract drifted")
        expected_response = {
            "visual_judgment": {
                "t1_state": "background",
                "t2_state": "building",
                "evidence": "Paired RGB and exact component are observable.",
            },
            "diagnosis": {"audit_checklist": expected_checklists[case_id]},
        }
        if case.get("backend_audit_response") != expected_response:
            raise GoldenFixtureValidationError(f"{case_id} backend input drifted")
        if case.get("expected_behavior") != EXPECTED_BEHAVIOR[case_id]:
            raise GoldenFixtureValidationError(f"{case_id} behavior oracle drifted")
    _reject_forbidden_gt(receipt, source="H0 golden receipt")


def validate_receipt_payload(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    observed_receipt_raw_sha256: str,
    repo_root: Path,
) -> dict[str, str]:
    """Validate already-decoded documents; the file wrapper supplies raw bytes.

    Keeping the exact-oracle check in this pure function lets tamper tests prove
    that coordinated receipt + manifest hash changes still fail closed.  The
    CLI wrapper additionally guarantees that ``observed_receipt_raw_sha256``
    came from the exact receipt path named by the manifest.
    """

    root = repo_root.resolve(strict=True)
    if manifest.get("schema_id") != MANIFEST_SCHEMA_ID:
        raise GoldenFixtureValidationError("CPU fixture manifest schema mismatch")
    if manifest.get("status") != "frozen":
        raise GoldenFixtureValidationError("CPU fixture manifest is not frozen")
    _validate_exact_receipt_payload(receipt)

    contract = _object(
        manifest.get("golden_fixture_contract"),
        source="manifest.golden_fixture_contract",
    )
    required_contract = {
        "h0_receipt_required": True,
        "receipt_schema_id": SCHEMA_ID,
        "receipt_path": GOLDEN_RECEIPT_RELPATH,
        "validator_path": VALIDATOR_RELPATH,
        "p0_acceptance_receipt_path": P0_RECEIPT_RELPATH,
        "p0_acceptance_receipt_raw_sha256": FROZEN_P0_RAW_SHA256,
        "p0_acceptance_receipt_canonical_sha256": FROZEN_P0_CANONICAL_SHA256,
        "h0_round_two_invocations": 0,
        "h1_eligible_cases_round_two_invocations": 1,
        "h1_third_round_allowed": False,
        "dangerous_action_must_not_execute": True,
        "safe_plan_must_be_checked_by_existing_environment_gates": True,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise GoldenFixtureValidationError(
                f"CPU manifest golden contract drifted at {key}"
            )
    observed_raw = _digest(
        observed_receipt_raw_sha256, source="observed golden receipt raw hash"
    )
    observed_canonical = canonical_sha256(receipt)
    if observed_raw != _digest(
        contract.get("receipt_raw_sha256"),
        source="manifest golden receipt raw hash",
    ):
        raise GoldenFixtureValidationError("H0 golden receipt raw hash mismatch")
    if observed_canonical != _digest(
        contract.get("receipt_canonical_sha256"),
        source="manifest golden receipt canonical hash",
    ):
        raise GoldenFixtureValidationError(
            "H0 golden receipt canonical hash mismatch"
        )
    validator = _repo_file(
        root,
        _string(contract.get("validator_path"), source="validator_path"),
        source="validator_path",
    )
    if sha256_file(validator) != _digest(
        contract.get("validator_sha256"), source="manifest validator hash"
    ):
        raise GoldenFixtureValidationError("H0 golden validator hash mismatch")

    p0_path = _repo_file(root, P0_RECEIPT_RELPATH, source="P0 receipt path")
    p0_payload, p0_raw = _read_json_bytes(p0_path, source="P0 acceptance receipt")
    if p0_payload.get("schema_id") != P0_RECEIPT_SCHEMA_ID:
        raise GoldenFixtureValidationError("P0 acceptance receipt schema drifted")
    if p0_payload.get("status") != "complete":
        raise GoldenFixtureValidationError("P0 acceptance receipt is incomplete")
    if p0_payload.get("validation_commit") != FROZEN_P0_COMMIT:
        raise GoldenFixtureValidationError("P0 validation commit drifted")
    if sha256_bytes(p0_raw) != FROZEN_P0_RAW_SHA256:
        raise GoldenFixtureValidationError("P0 acceptance receipt raw hash drifted")
    if canonical_sha256(p0_payload) != FROZEN_P0_CANONICAL_SHA256:
        raise GoldenFixtureValidationError(
            "P0 acceptance receipt canonical hash drifted"
        )

    for relative_path, expected_digest in FROZEN_SOURCE_BLOBS:
        observed = _git_blob_sha256(root, FROZEN_P0_COMMIT, relative_path)
        if observed != expected_digest:
            raise GoldenFixtureValidationError(
                f"frozen source blob drifted: {relative_path}"
            )
    return {
        "receipt_raw_sha256": observed_raw,
        "receipt_canonical_sha256": observed_canonical,
        "validator_sha256": sha256_file(validator),
        "p0_receipt_raw_sha256": FROZEN_P0_RAW_SHA256,
        "p0_receipt_canonical_sha256": FROZEN_P0_CANONICAL_SHA256,
        "source_anchor_commit": FROZEN_P0_COMMIT,
    }


def validate_receipt(
    receipt_path: Path,
    manifest_path: Path,
    *,
    repo_root: Path,
) -> dict[str, str]:
    root = repo_root.resolve(strict=True)
    expected_receipt_path = _repo_file(
        root, GOLDEN_RECEIPT_RELPATH, source="golden receipt path"
    )
    if receipt_path.resolve(strict=True) != expected_receipt_path:
        raise GoldenFixtureValidationError(
            "validator was not given the canonical H0 golden receipt path"
        )
    receipt, receipt_raw = _read_json_bytes(
        expected_receipt_path, source="H0 golden receipt"
    )
    manifest, _ = _read_json_bytes(
        manifest_path.resolve(strict=True), source="CPU fixture manifest"
    )
    return validate_receipt_payload(
        receipt,
        manifest,
        observed_receipt_raw_sha256=sha256_bytes(receipt_raw),
        repo_root=root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_receipt(
            args.receipt,
            args.manifest,
            repo_root=args.repo_root,
        )
    except (GoldenFixtureValidationError, OSError) as error:
        print(f"H0_GOLDEN_FIXTURE_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "valid", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
