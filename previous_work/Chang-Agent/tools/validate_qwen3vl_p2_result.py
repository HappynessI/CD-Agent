#!/usr/bin/env python3
"""Fail-closed validation for the frozen Qwen3-VL P2 actionability result.

The validator deliberately re-reads every authoritative artifact and recomputes
the oracle metrics from the generated masks and offline labels.  It does not
import the P2 scorer or any of its metric/classification helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


RESULT_SCHEMA = "qwen3vl_p2_actionability_result_v1"
IMPLEMENTATION_A = "8252a57b191a2a7c4aa176a3f82cca5218119533"
FREEZE_B = "cda314ce9caeb6e9dd69ba0d08fe5254a8689fdd"
ACCEPTANCE_PARENT_C = "d412759857d030b8ab766a077ea54b603585e72a"
ACCEPTANCE_D = "a094cc59d9104471b70b0624b4e10cbacfefe2e5"
P1_ACCEPTANCE = "17e9bb9914266691aee0c3d2b5e78b6bf64d7a00"
P1_FREEZE_B4 = "5210f89f5ca13ee1e636f5aa6a44631206a80123"
GENERATION_JOB = "53272"
SCORING_JOB = "53273"
CONTRACT_JOB = "53269"
MATERIALIZATION_JOB = "53268"
ORDERED_SAMPLES = ("test_20_15", "test_1_1", "test_50_8")
TARGET_SAMPLES = ("test_1_1", "test_50_8")
COUNTS = {"total": 36, "primary": 27, "diagnostic_executor_only": 9}
FLOW_COUNTS = {"requested": 36, "generated": 36, "tool_success": 36}
SCORED_FLOW_COUNTS = {**FLOW_COUNTS, "scored": 36}
ORACLE_COUNTS = {"beneficial": 4, "harmful": 31, "indeterminate": 1}
PRIMARY_ORACLE_COUNTS = {"beneficial": 3, "harmful": 23, "indeterminate": 1}
DIAGNOSTIC_ORACLE_COUNTS = {"beneficial": 1, "harmful": 8, "indeterminate": 0}
GATE_LEAF_COUNTS = {
    "harmful_gate_pass": 15,
    "hard_gate_correct_reject": 16,
    "beneficial_gate_pass": 2,
    "indeterminate_gate_pass": 1,
    "hard_gate_false_reject": 2,
}
GATE_REJECTION_REASONS = {
    "point_proposal_combined_change_gate_exceeded": 12,
    "component_count_delta_exceeded": 6,
}
PRIMARY_BENEFICIAL_IDS = {
    "test_20_15--primary--r7--negative_point--t2",
    "test_50_8--primary--s8--box--t1",
    "test_50_8--primary--s8--box--t2",
}
DIAGNOSTIC_BENEFICIAL_IDS = {
    "test_1_1--diagnostic_executor_only--r0--positive_point--t1"
}
RESULT_RELATIVE = Path(
    "doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.json"
)
RESULT_MARKDOWN_RELATIVE = Path(
    "doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.md"
)
ACCEPTANCE_RELATIVE = Path(
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json"
)
ACCEPTANCE_D_BLOBS = {
    ACCEPTANCE_RELATIVE: (
        "<hash omitted>",
        744,
    ),
    RESULT_RELATIVE: (
        "<hash omitted>",
        10062,
    ),
}
PREREG_RELATIVE = Path(
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
)
EXECUTION_RELATIVE = Path(
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
)
HANDOFF_RELATIVE = Path(
    "doc/training_pipeline/0802_third_try/QWEN3VL_SESSION_HANDOFF_0802.md"
)
REGISTRY_RELATIVE = Path("experiments/experiment_registry.json")
WORKBOOK_RELATIVE = Path("experiments/experiment_registry.xlsx")
REMOVED_IMPLEMENTATION_A_WORKTREE = Path(
    "/guisongxia01/pangchao/wangyihan/wyh/Chang-Agent-qwen3vl-p2-a-0803-run"
)
REMOVED_IMPLEMENTATION_A_BINDINGS = {
    REMOVED_IMPLEMENTATION_A_WORKTREE
    / "tools/build_qwen3vl_p2_preregistration.py": "tools/build_qwen3vl_p2_preregistration.py",
    REMOVED_IMPLEMENTATION_A_WORKTREE
    / "tools/validate_qwen3vl_p2_preregistration.py": "tools/validate_qwen3vl_p2_preregistration.py",
    REMOVED_IMPLEMENTATION_A_WORKTREE
    / "tools/verify_qwen3vl_p2_release.py": "tools/verify_qwen3vl_p2_release.py",
}

PREREG_RAW_SHA256 = "<hash omitted>"
PREREG_CANONICAL_SHA256 = "<hash omitted>"
EXECUTION_RAW_SHA256 = "<hash omitted>"
EXECUTION_CANONICAL_SHA256 = "<hash omitted>"

FORMAL_SHA256 = {
    "submission_intent.json": "<hash omitted>",
    "submission_receipt.json": "<hash omitted>",
    "generation_receipt.json": "<hash omitted>",
    "generation/candidate_generation_complete.json": "<hash omitted>",
    "generation/candidates.json": "<hash omitted>",
    "scoring_receipt.json": "<hash omitted>",
    "scoring/runtime_scoring_manifest.json": "<hash omitted>",
    "scoring/score_report.json": "<hash omitted>",
    "scoring/actionability_decision.json": "<hash omitted>",
    "logs/generation-status-53272.txt": "<hash omitted>",
    "logs/scoring-status-53273.txt": "<hash omitted>",
}
MATERIALIZATION_RECEIPT_SHA256 = (
    "<hash omitted>"
)
CONTRACT_RECEIPT_SHA256 = (
    "<hash omitted>"
)
GENERATION_WORKER_SHA256 = (
    "<hash omitted>"
)
SCORING_WORKER_SHA256 = (
    "<hash omitted>"
)
MATERIALIZATION_WORKER_SHA256 = (
    "<hash omitted>"
)
CONTRACT_WORKER_SHA256 = (
    "<hash omitted>"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ATTEMPT_RE = re.compile(r"attempt-[0-9]{4,}")
COMPARISON_EPSILON = 1e-12

CLOSEOUT_CHECKS = {
    "clean_detached_exact_commit": True,
    "freeze_b_ancestor": True,
    "frozen_manifests_unchanged": True,
    "frozen_execution_surface_unchanged": True,
    "formal_evidence_and_result": "passed",
    "negative_gate_and_no_data": "passed",
    "workbook_check": "passed",
    "p2_result_targeted_tests": "passed",
    "registry_targeted_tests": "passed",
    "shell_syntax": "passed",
    "full_regression": "passed",
    "repository_unchanged": True,
}
CLOSEOUT_CANDIDATE_BINDINGS = {
    "result_json",
    "result_markdown",
    "handoff",
    "registry",
    "workbook",
    "formal_submission_receipt",
    "generation_receipt",
    "scoring_receipt",
    "score_report",
    "decision",
    "status",
    "submission_intent",
    "submission_receipt",
    "test_log",
}


class ValidationError(ValueError):
    """Raised when an artifact does not satisfy the frozen P2 contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_exact(actual: Any, expected: Any, location: str) -> None:
    if not same_json(actual, expected):
        raise ValidationError(f"{location} differs")


def same_json(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
            return False
        if isinstance(expected, float) or isinstance(actual, float):
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15)
        return actual == expected
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(actual) == set(expected) and all(
            same_json(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            same_json(left, right) for left, right in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_nonfinite(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"non-finite JSON number at {location}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{location}[{index}]")


def read_json(path: Path, location: str | None = None) -> dict[str, Any]:
    label = location or str(path)

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    _reject_nonfinite(value, label)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"cannot canonicalize JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def require_sha(value: Any, location: str, *, commit: bool = False) -> str:
    pattern = COMMIT_RE if commit else SHA256_RE
    require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{location} is malformed")
    return value


def require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{location} must be an object")
    return value


def canonical_path(path: Path, location: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValidationError(f"{location} does not exist: {error}") from error
    absolute = Path(os.path.abspath(path))
    require(absolute == resolved, f"{location} must be a canonical non-symlink path")
    return resolved


def exact_path(raw: Any, expected: Path, location: str) -> Path:
    require(isinstance(raw, str) and raw, f"{location} path is missing")
    observed = canonical_path(Path(raw), location)
    wanted = canonical_path(expected, f"expected {location}")
    require(observed == wanted, f"{location} path differs")
    return observed


def relative_file(root: Path, raw: Any, location: str) -> Path:
    require(isinstance(raw, str) and raw, f"{location} path is missing")
    relative = Path(raw)
    require(not relative.is_absolute(), f"{location} must be relative")
    require(".." not in relative.parts and "." not in relative.parts, f"{location} is not canonical")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"{location} escapes its root") from error
    require(path.is_file(), f"{location} is not a file")
    return path


def exact_commit_blob(
    repository: Path,
    commit: str,
    relative: str,
    location: str,
) -> bytes:
    """Read one exact relative file as a Git blob from one exact commit."""

    require_sha(commit, f"{location} commit", commit=True)
    relative_path = Path(relative)
    require(
        not relative_path.is_absolute()
        and bool(relative_path.parts)
        and ".." not in relative_path.parts
        and "\\" not in relative,
        f"{location} relative path is invalid",
    )
    repository = canonical_path(repository, f"{location} repository")
    commit_check = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", f"{commit}^{{commit}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        commit_check.returncode == 0 and commit_check.stdout.strip() == commit,
        f"{location} exact commit object is unavailable",
    )
    object_spec = f"{commit}:{relative_path.as_posix()}"
    type_check = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-t", object_spec],
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        type_check.returncode == 0 and type_check.stdout.strip() == "blob",
        f"{location} exact object is not a blob",
    )
    blob = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "blob", object_spec],
        capture_output=True,
        check=False,
    )
    require(blob.returncode == 0, f"{location} cannot read exact blob")
    return blob.stdout


def removed_implementation_a_blob(raw_path: Path, location: str) -> bytes | None:
    """Recover an exact deleted-worktree binding from commit A's Git blob.

    This is deliberately not a filesystem relocation mechanism.  It accepts
    only three fully enumerated historical paths and reads the corresponding
    fixed path from the exact 40-hex implementation commit.  Thus a matching
    basename elsewhere, a different commit, a tree, or a missing Git object
    all fail closed.
    """

    relative = REMOVED_IMPLEMENTATION_A_BINDINGS.get(raw_path)
    if relative is None:
        return None
    require(not raw_path.is_symlink(), f"{location} deleted binding is a symlink")
    repository = Path(__file__).resolve().parents[1]
    return exact_commit_blob(
        repository,
        IMPLEMENTATION_A,
        relative,
        f"{location} exact implementation A",
    )


def verify_binding(
    value: Any,
    expected_path: Path | None,
    location: str,
    expected_sha256: str | None = None,
) -> Path:
    binding = require_mapping(value, location)
    require(set(binding) == {"path", "sha256", "size_bytes"}, f"{location} keys differ")
    require_sha(binding.get("sha256"), f"{location}.sha256")
    raw = binding.get("path")
    if expected_path is None:
        require(isinstance(raw, str) and raw, f"{location}.path is missing")
        raw_path = Path(raw)
        if not raw_path.exists():
            payload = removed_implementation_a_blob(raw_path, f"{location}.path")
            if payload is not None:
                observed = hashlib.sha256(payload).hexdigest()
                require(binding.get("sha256") == observed, f"{location} SHA256 differs")
                require(
                    type(binding.get("size_bytes")) is int
                    and binding.get("size_bytes") == len(payload),
                    f"{location} size differs",
                )
                if expected_sha256 is not None:
                    require(observed == expected_sha256, f"{location} frozen SHA256 differs")
                return raw_path
        path = canonical_path(raw_path, f"{location}.path")
    else:
        path = exact_path(raw, expected_path, f"{location}.path")
    observed = sha256_file(path)
    require(binding.get("sha256") == observed, f"{location} SHA256 differs")
    require(type(binding.get("size_bytes")) is int and binding.get("size_bytes") == path.stat().st_size, f"{location} size differs")
    if expected_sha256 is not None:
        require(observed == expected_sha256, f"{location} frozen SHA256 differs")
    return path


def verify_all_bindings(value: Any, location: str) -> None:
    bindings = require_mapping(value, location)
    for name, binding in bindings.items():
        verify_binding(binding, None, f"{location}.{name}")


def parse_status(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read status {path}: {error}") from error
    for line in lines:
        require("=" in line, f"status line lacks '=' in {path}")
        key, value = line.split("=", 1)
        require(bool(key) and key not in result, f"empty or duplicate status key in {path}")
        result[key] = value
    return result


def run_git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        capture_output=True,
        check=check,
    )


def verify_git_lineage(repo: Path, acceptance_parent: str | None) -> None:
    parents_b = run_git(repo, "rev-list", "--parents", "-n", "1", FREEZE_B).stdout.split()
    require(parents_b == [FREEZE_B, IMPLEMENTATION_A], "freeze B is not the single-child freeze of A")
    parents_a = run_git(repo, "rev-list", "--parents", "-n", "1", IMPLEMENTATION_A).stdout.split()
    require(parents_a == [IMPLEMENTATION_A, P1_ACCEPTANCE], "implementation A parent differs")
    changes = tuple(
        line
        for line in run_git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            FREEZE_B,
        ).stdout.splitlines()
        if line
    )
    expected = (f"A\t{PREREG_RELATIVE}", f"A\t{EXECUTION_RELATIVE}")
    require(changes == expected, "freeze B did not add exactly the two frozen manifests")
    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    require(COMMIT_RE.fullmatch(head) is not None, "validated HEAD is malformed")
    if acceptance_parent is None:
        lineage_target = head
        parents_head = run_git(
            repo, "rev-list", "--parents", "-n", "1", head
        ).stdout.split()
        require(
            parents_head == [head, FREEZE_B],
            "candidate C must be a single-parent direct child of freeze B",
        )
    else:
        require_exact(
            acceptance_parent,
            ACCEPTANCE_PARENT_C,
            "fixed P2 acceptance parent C",
        )
        acceptance_parents = run_git(
            repo, "rev-list", "--parents", "-n", "1", ACCEPTANCE_D
        ).stdout.split()
        require(
            acceptance_parents == [ACCEPTANCE_D, ACCEPTANCE_PARENT_C],
            "fixed P2 acceptance D parent differs",
        )
        acceptance_ancestor = run_git(
            repo,
            "merge-base",
            "--is-ancestor",
            ACCEPTANCE_D,
            head,
            check=False,
        )
        require(
            acceptance_ancestor.returncode == 0,
            "validated HEAD is not a descendant of fixed P2 acceptance D",
        )
        lineage_target = ACCEPTANCE_D
    ancestor = run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        FREEZE_B,
        lineage_target,
        check=False,
    )
    require(
        ancestor.returncode == 0,
        "freeze B is not an ancestor of the validated P2 lineage target",
    )
    frozen_execution_files = (
        "change_agent",
        str(PREREG_RELATIVE),
        "doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.md",
        str(EXECUTION_RELATIVE),
        "tests/test_qwen3vl_p2_oracle.py",
        "tests/test_qwen3vl_p2_preregistration.py",
        "tools/build_qwen3vl_p2_preregistration.py",
        "tools/run_qwen3vl_p2_contract_validation.slurm",
        "tools/run_qwen3vl_p2_manifest_build.slurm",
        "tools/run_qwen3vl_p2_oracle.py",
        "tools/run_qwen3vl_p2_oracle_generation.slurm",
        "tools/run_qwen3vl_p2_oracle_scoring.slurm",
        "tools/score_qwen3vl_p2_oracle.py",
        "tools/submit_qwen3vl_p2_manifest_build.sh",
        "tools/submit_qwen3vl_p2_oracle.sh",
        "tools/validate_qwen3vl_p2_preregistration.py",
        "tools/verify_qwen3vl_p2_release.py",
    )
    frozen_diff = run_git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        FREEZE_B,
        lineage_target,
        "--",
        *frozen_execution_files,
    ).stdout.splitlines()
    require(not frozen_diff, "frozen P2 execution surface changed after freeze B")
    if acceptance_parent is None:
        return
    allowed = {
        str(HANDOFF_RELATIVE),
        str(ACCEPTANCE_RELATIVE),
        str(RESULT_MARKDOWN_RELATIVE),
    }
    changes = {
        line
        for line in run_git(
            repo,
            "diff",
            "--name-only",
            "--no-renames",
            ACCEPTANCE_PARENT_C,
            ACCEPTANCE_D,
        ).stdout.splitlines()
        if line
    }
    require(changes, "acceptance D commit is empty")
    require(changes <= allowed, "acceptance D changed files outside the allowlist")
    require(
        {
            str(ACCEPTANCE_RELATIVE),
            str(RESULT_MARKDOWN_RELATIVE),
            str(HANDOFF_RELATIVE),
        }
        <= changes,
        "acceptance D lacks the required acceptance/Markdown/handoff changes",
    )
    for relative, (expected_sha256, expected_size) in ACCEPTANCE_D_BLOBS.items():
        historical = exact_commit_blob(
            repo,
            ACCEPTANCE_D,
            relative.as_posix(),
            f"fixed acceptance D {relative}",
        )
        require(
            len(historical) == expected_size
            and hashlib.sha256(historical).hexdigest() == expected_sha256,
            f"fixed acceptance D blob/hash differs for {relative}",
        )
        current = canonical_path(repo / relative, f"current {relative}")
        require(
            current.stat().st_size == expected_size
            and sha256_file(current) == expected_sha256,
            f"current checkout changed fixed acceptance artifact {relative}",
        )


def verify_result_claims(result: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_id",
        "created_at_utc",
        "result_status",
        "engineering_outcome",
        "scientific_outcome",
        "interpretation",
        "git_lineage",
        "frozen_manifests",
        "manifest_materialization",
        "contract_validation",
        "formal_submission",
        "formal_jobs",
        "candidate_generation",
        "oracle_scoring",
        "target_results",
        "gt_access_boundary",
        "decision",
        "training_data_hypothesis",
        "interpretation_limits",
        "registry_closeout_target",
        "closeout_status",
        "attempts",
    }
    require(set(result) == expected_keys, "result top-level inventory differs")
    require_exact(result.get("schema_id"), RESULT_SCHEMA, "result schema")
    require_exact(result.get("created_at_utc"), "2026-08-03", "result date")
    require_exact(result.get("result_status"), "complete", "result status")
    require_exact(result.get("engineering_outcome"), "pass", "engineering outcome")
    require_exact(result.get("scientific_outcome"), "negative", "scientific outcome")
    require(isinstance(result.get("interpretation"), str) and "test_1_1" in result["interpretation"], "result interpretation is missing the failed target")

    lineage = require_mapping(result.get("git_lineage"), "git_lineage")
    require_exact(lineage.get("p1_execution_freeze_b4"), P1_FREEZE_B4, "P1 freeze B4")
    require_exact(lineage.get("p1_closeout_acceptance_commit"), P1_ACCEPTANCE, "P1 acceptance")
    require_exact(lineage.get("implementation_commit_a"), IMPLEMENTATION_A, "implementation A")
    require_exact(lineage.get("freeze_commit_b"), FREEZE_B, "freeze B")
    for key in (
        "a_parent_is_p1_closeout_acceptance",
        "b_parent_is_a",
        "freeze_b_two_manifest_add_only",
        "detached_run_worktree_exact_b_clean_detached",
        "p1_b4_worktree_preserved",
    ):
        require(lineage.get(key) is True, f"git_lineage.{key} differs")
    require(isinstance(lineage.get("detached_run_worktree"), str) and lineage["detached_run_worktree"], "detached run worktree is missing")

    require_exact(
        result.get("candidate_generation"),
        {
            "ordered_samples": list(ORDERED_SAMPLES),
            "requested": 36,
            "generated": 36,
            "tool_success": 36,
            "primary": 27,
            "diagnostic_executor_only": 9,
            "gt_loaded": False,
        },
        "candidate generation summary",
    )
    require_exact(
        result.get("oracle_scoring"),
        {
            "candidate_count": 36,
            "action_spec_count": 36,
            "unique_candidate_state_count": 36,
            "candidate_state_duplicate_count": 0,
            "qwen_invoked": False,
            "all_tiers_oracle_counts": ORACLE_COUNTS,
            "primary_oracle_counts": PRIMARY_ORACLE_COUNTS,
            "diagnostic_executor_only_oracle_counts": DIAGNOSTIC_ORACLE_COUNTS,
            "all_tiers_gate_leaf_counts": GATE_LEAF_COUNTS,
        },
        "oracle scoring summary",
    )
    require_exact(
        result.get("gt_access_boundary"),
        {
            "generation_gt_loaded": False,
            "candidate_generation_complete_verified_before_label_access": True,
            "all_candidate_records_and_hashes_verified_before_label_access": True,
            "runtime_scoring_manifest_not_available_to_generator": True,
            "offline_labels_used_only_for_scoring": True,
        },
        "GT access boundary",
    )
    require_exact(
        result.get("decision"),
        {
            "decision_basis": "primary_candidates_only",
            "diagnostic_candidates_excluded": True,
            "gate_modification_authorized": False,
            "no_primary_beneficial_candidate_samples": ["test_1_1"],
            "proceed_to_c_iso_and_t_traj_data_construction": False,
            "next_action": "stop_training_and_review_frozen_action_space",
        },
        "result decision",
    )
    require_exact(
        result.get("registry_closeout_target"),
        {
            "experiment_rows": 15,
            "field_dictionary_rows": 52,
            "data_rows": 21,
            "attempt_rows": 40,
        },
        "registry closeout target",
    )
    require_exact(result.get("closeout_status"), "pending_exact_candidate_validation", "closeout status")
    require_exact(result.get("attempts"), [], "closeout attempts")
    hypothesis = require_mapping(result.get("training_data_hypothesis"), "training data hypothesis")
    require_exact(hypothesis.get("status"), "registered_but_not_tested", "training-data hypothesis status")
    require(isinstance(hypothesis.get("p2_implication"), str) and "must not begin" in hypothesis["p2_implication"], "training-data stop boundary is missing")
    limits = result.get("interpretation_limits")
    require(isinstance(limits, list) and len(limits) == 5, "interpretation limits differ")
    require(any("larger, multi-step" in str(item) for item in limits), "larger-action-space limit is missing")
    require(any("not evidence of LEVIR-CD generalization" in str(item) for item in limits), "generalization limit is missing")


def verify_result_manifests(result: Mapping[str, Any], repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = require_mapping(result.get("frozen_manifests"), "frozen_manifests")
    require(set(manifests) == {"preregistration", "execution"}, "frozen manifest inventory differs")
    prereg_spec = require_mapping(manifests.get("preregistration"), "frozen preregistration")
    execution_spec = require_mapping(manifests.get("execution"), "frozen execution")
    require_exact(
        prereg_spec,
        {
            "path": str(PREREG_RELATIVE),
            "raw_sha256": PREREG_RAW_SHA256,
            "canonical_sha256": PREREG_CANONICAL_SHA256,
        },
        "frozen preregistration identity",
    )
    require_exact(
        execution_spec,
        {
            "path": str(EXECUTION_RELATIVE),
            "raw_sha256": EXECUTION_RAW_SHA256,
            "canonical_sha256": EXECUTION_CANONICAL_SHA256,
            "gt_free_generator_input": True,
        },
        "frozen execution identity",
    )
    prereg_path = canonical_path(repo / PREREG_RELATIVE, "preregistration")
    execution_path = canonical_path(repo / EXECUTION_RELATIVE, "execution manifest")
    require(sha256_file(prereg_path) == PREREG_RAW_SHA256, "preregistration raw SHA256 differs")
    require(sha256_file(execution_path) == EXECUTION_RAW_SHA256, "execution raw SHA256 differs")
    prereg = read_json(prereg_path, "preregistration")
    execution = read_json(execution_path, "execution manifest")
    require(canonical_sha256(prereg) == PREREG_CANONICAL_SHA256, "preregistration canonical SHA256 differs")
    require(canonical_sha256(execution) == EXECUTION_CANONICAL_SHA256, "execution canonical SHA256 differs")
    require_exact(execution.get("source_freeze_commit"), P1_FREEZE_B4, "execution source freeze")
    require_exact(execution.get("implementation_commit"), IMPLEMENTATION_A, "execution implementation")
    require_exact(execution.get("ordered_samples"), list(ORDERED_SAMPLES), "execution sample order")
    require_exact(execution.get("expected_candidate_counts"), COUNTS, "execution candidate counts")
    return prereg, execution


def assert_execution_is_gt_free(value: Any, location: str = "execution") -> None:
    forbidden_keys = {
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
    if isinstance(value, Mapping):
        for key, child in value.items():
            require(str(key).lower() not in forbidden_keys, f"GT key leaked into {location}.{key}")
            assert_execution_is_gt_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_execution_is_gt_free(child, f"{location}[{index}]")
    elif isinstance(value, str) and ("/label_cvt/" in value.lower() or "/ground_truth/" in value.lower()):
        raise ValidationError(f"GT path leaked into {location}")


def verify_result_preflight_receipts(result: Mapping[str, Any]) -> None:
    materialization = require_mapping(result.get("manifest_materialization"), "manifest_materialization")
    require_exact(materialization.get("job_id"), MATERIALIZATION_JOB, "materialization Job ID")
    require_exact(materialization.get("job_name"), "CA0803P2Manifest", "materialization Job name")
    require_exact(materialization.get("state"), "COMPLETED", "materialization state")
    require_exact(materialization.get("exit_code"), "0:0", "materialization exit")
    require_exact(materialization.get("node"), "gpu48", "materialization node")
    require_exact(materialization.get("partition"), "gre", "materialization partition")
    require_exact(materialization.get("resources"), "0GPU/2CPU/16G", "materialization resources")
    require(materialization.get("label_file_accessed") is False, "materialization accessed labels")
    require_exact(materialization.get("worker_sha256"), MATERIALIZATION_WORKER_SHA256, "materialization worker")
    materialization_path = verify_binding(
        materialization.get("receipt"), None, "materialization receipt", MATERIALIZATION_RECEIPT_SHA256
    )
    materialization_receipt = read_json(materialization_path, "materialization receipt")
    require_exact(materialization_receipt.get("schema_id"), "qwen3vl_p2_manifest_materialization_receipt_v2", "materialization receipt schema")
    require_exact(materialization_receipt.get("status"), "complete", "materialization receipt status")
    require_exact(materialization_receipt.get("job_id"), MATERIALIZATION_JOB, "materialization receipt Job ID")
    require_exact(materialization_receipt.get("implementation_commit_a"), IMPLEMENTATION_A, "materialization receipt implementation")
    require(materialization_receipt.get("label_file_accessed") is False, "materialization receipt label boundary differs")
    require_exact(materialization_receipt.get("worker_sha256"), MATERIALIZATION_WORKER_SHA256, "materialization receipt worker")
    verify_all_bindings(materialization_receipt.get("bindings"), "materialization receipt bindings")

    contract = require_mapping(result.get("contract_validation"), "contract_validation")
    require_exact(contract.get("job_id"), CONTRACT_JOB, "contract Job ID")
    require_exact(contract.get("job_name"), "CA0803P2Contract", "contract Job name")
    require_exact(contract.get("state"), "COMPLETED", "contract state")
    require_exact(contract.get("exit_code"), "0:0", "contract exit")
    require_exact(contract.get("node"), "gpu48", "contract node")
    require_exact(contract.get("partition"), "gre", "contract partition")
    require_exact(contract.get("resources"), "0GPU/2CPU/16G", "contract resources")
    require_exact(contract.get("p2_targeted_tests"), "37/37", "contract P2 tests")
    require_exact(contract.get("registry_targeted_tests"), "15/15", "contract registry tests")
    require_exact(contract.get("full_regression"), "508/508 passed; 3 skipped", "contract regression")
    require_exact(contract.get("worker_sha256"), CONTRACT_WORKER_SHA256, "contract worker")
    contract_path = verify_binding(contract.get("receipt"), None, "contract receipt", CONTRACT_RECEIPT_SHA256)
    contract_receipt = read_json(contract_path, "contract receipt")
    require_exact(contract_receipt.get("schema_id"), "qwen3vl_p2_contract_validation_receipt_v1", "contract receipt schema")
    require_exact(contract_receipt.get("status"), "complete", "contract receipt status")
    require_exact(contract_receipt.get("job_id"), CONTRACT_JOB, "contract receipt Job ID")
    require_exact(contract_receipt.get("freeze_commit_b"), FREEZE_B, "contract receipt freeze")
    require_exact(contract_receipt.get("implementation_commit_a"), IMPLEMENTATION_A, "contract receipt implementation")
    require(contract_receipt.get("freeze_b_parent_is_a") is True, "contract receipt parent flag differs")
    require_exact(contract_receipt.get("worker_sha256"), CONTRACT_WORKER_SHA256, "contract receipt worker")
    checks = contract_receipt.get("checks")
    require(isinstance(checks, Mapping) and all(value is True or value == "passed" for value in checks.values()), "contract checks are incomplete")
    verify_all_bindings(contract_receipt.get("bindings"), "contract receipt bindings")


def verify_formal_headers(
    result: Mapping[str, Any], attempt: Path, execution: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    formal = require_mapping(result.get("formal_submission"), "formal_submission")
    require_exact(formal.get("attempt_id"), "attempt-0001", "formal attempt ID")
    exact_path(formal.get("attempt_root"), attempt, "formal attempt root")
    require_exact(formal.get("submission_intent_sha256"), FORMAL_SHA256["submission_intent.json"], "formal intent SHA256")
    require_exact(formal.get("submission_receipt_sha256"), FORMAL_SHA256["submission_receipt.json"], "formal receipt SHA256")
    require_exact(formal.get("dependency"), f"afterok:{GENERATION_JOB}", "formal dependency")
    require(formal.get("cross_attempt_composition") is False, "cross-attempt composition is forbidden")

    for relative, expected_hash in FORMAL_SHA256.items():
        path = canonical_path(attempt / relative, f"formal {relative}")
        require(sha256_file(path) == expected_hash, f"formal {relative} SHA256 differs")

    intent = read_json(attempt / "submission_intent.json", "formal submission intent")
    submission = read_json(attempt / "submission_receipt.json", "formal submission receipt")
    generation_receipt = read_json(attempt / "generation_receipt.json", "generation receipt")
    scoring_receipt = read_json(attempt / "scoring_receipt.json", "scoring receipt")
    require_exact(intent.get("schema_id"), "qwen3vl_p2_formal_submission_intent_v1", "formal intent schema")
    require_exact(intent.get("status"), "prepared", "formal intent status")
    require_exact(intent.get("attempt_id"), attempt.name, "formal intent attempt")
    exact_path(intent.get("attempt_root"), attempt, "formal intent root")
    require_exact(intent.get("freeze_commit_b"), FREEZE_B, "formal intent freeze")
    require_exact(intent.get("implementation_commit_a"), IMPLEMENTATION_A, "formal intent implementation")
    require_exact(
        intent.get("job_contract"),
        {
            "dependency": "afterok",
            "generation_name": "CA0803P2Generate",
            "node": "gpu48",
            "partition": "gre",
            "scoring_name": "CA0803P2Score",
            "user": "guisongxia",
        },
        "formal intent job contract",
    )
    require_exact(
        intent.get("manifests"),
        {
            "execution": {"canonical_sha256": EXECUTION_CANONICAL_SHA256, "sha256": EXECUTION_RAW_SHA256},
            "preregistration": {"canonical_sha256": PREREG_CANONICAL_SHA256, "sha256": PREREG_RAW_SHA256},
        },
        "formal intent manifests",
    )

    require_exact(submission.get("schema_id"), "qwen3vl_p2_oracle_submission_receipt_v2", "formal receipt schema")
    require_exact(submission.get("submission_status"), "formal_submitted", "formal receipt status")
    require_exact(submission.get("attempt_id"), attempt.name, "formal receipt attempt")
    exact_path(submission.get("attempt_root"), attempt, "formal receipt root")
    require_exact(submission.get("freeze_commit_b"), FREEZE_B, "formal receipt freeze")
    require_exact(submission.get("implementation_commit_a"), IMPLEMENTATION_A, "formal receipt implementation")
    require_exact(submission.get("jobs"), {"contract_validation": CONTRACT_JOB, "generation": GENERATION_JOB, "scoring": SCORING_JOB}, "formal jobs")
    require_exact(submission.get("dependency"), f"afterok:{GENERATION_JOB}", "formal receipt dependency")
    require_exact(submission.get("job_user"), "guisongxia", "formal receipt user")
    require_exact(submission.get("node"), "gpu48", "formal receipt node")
    require_exact(submission.get("partition"), "gre", "formal receipt partition")
    require_exact(submission.get("workers"), intent.get("workers"), "formal worker binding")
    require_exact(submission.get("manifests"), intent.get("manifests"), "formal manifest binding")
    verify_binding(submission.get("intent"), attempt / "submission_intent.json", "formal intent binding", FORMAL_SHA256["submission_intent.json"])

    contract_intent = require_mapping(intent.get("contract_validation"), "intent contract validation")
    contract_submission = require_mapping(submission.get("contract_validation"), "submission contract validation")
    require_exact(contract_intent.get("job_id"), CONTRACT_JOB, "intent contract Job ID")
    require_exact(contract_submission.get("job_id"), CONTRACT_JOB, "submission contract Job ID")
    contract_path = Path(str(require_mapping(contract_intent.get("receipt"), "intent contract receipt").get("path")))
    verify_binding(contract_intent.get("receipt"), contract_path, "intent contract receipt", CONTRACT_RECEIPT_SHA256)
    verify_binding(contract_submission.get("receipt"), contract_path, "submission contract receipt", CONTRACT_RECEIPT_SHA256)

    jobs = result.get("formal_jobs")
    require(isinstance(jobs, list) and len(jobs) == 2, "result formal jobs differ")
    generation_job, scoring_job = jobs
    require_exact(
        generation_job,
        {
            "role": "candidate_generation",
            "job_id": GENERATION_JOB,
            "job_name": "CA0803P2Generate",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed": "00:19:45",
            "node": "gpu48",
            "partition": "gre",
            "resources": "1xA800/4CPU/48G",
            "worker_sha256": GENERATION_WORKER_SHA256,
            "receipt_sha256": FORMAL_SHA256["generation_receipt.json"],
            "complete_marker_sha256": FORMAL_SHA256["generation/candidate_generation_complete.json"],
            "candidate_index_sha256": FORMAL_SHA256["generation/candidates.json"],
        },
        "generation job result",
    )
    require_exact(
        scoring_job,
        {
            "role": "offline_scoring",
            "job_id": SCORING_JOB,
            "job_name": "CA0803P2Score",
            "depends_on": f"afterok:{GENERATION_JOB}",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed": "00:00:15",
            "node": "gpu48",
            "partition": "gre",
            "resources": "0GPU/2CPU/16G",
            "worker_sha256": SCORING_WORKER_SHA256,
            "receipt_sha256": FORMAL_SHA256["scoring_receipt.json"],
            "runtime_scoring_manifest_sha256": FORMAL_SHA256["scoring/runtime_scoring_manifest.json"],
            "score_report_sha256": FORMAL_SHA256["scoring/score_report.json"],
            "actionability_decision_sha256": FORMAL_SHA256["scoring/actionability_decision.json"],
        },
        "scoring job result",
    )

    verify_generation_receipt(generation_receipt, attempt)
    verify_scoring_receipt(scoring_receipt, attempt)
    verify_statuses(attempt)
    require_exact(execution.get("expected_candidate_counts"), COUNTS, "formal execution counts")
    return intent, submission, generation_receipt, scoring_receipt


def verify_generation_receipt(receipt: Mapping[str, Any], attempt: Path) -> None:
    require_exact(receipt.get("schema_id"), "qwen3vl_p2_oracle_generation_receipt_v2", "generation receipt schema")
    require_exact(receipt.get("status"), "complete", "generation receipt status")
    require_exact(receipt.get("job_id"), GENERATION_JOB, "generation receipt Job ID")
    require_exact(receipt.get("freeze_commit_b"), FREEZE_B, "generation receipt freeze")
    require_exact(receipt.get("implementation_commit_a"), IMPLEMENTATION_A, "generation receipt implementation")
    require_exact(receipt.get("worker_sha256"), GENERATION_WORKER_SHA256, "generation receipt worker")
    require(receipt.get("gt_loaded") is False, "generation receipt GT flag differs")
    require_exact(receipt.get("counts"), COUNTS, "generation receipt counts")
    require_exact(receipt.get("candidate_flow_counts"), FLOW_COUNTS, "generation flow counts")
    bindings = require_mapping(receipt.get("bindings"), "generation receipt bindings")
    require(set(bindings) == {"candidate_index", "complete_marker", "execution_manifest", "formal_submission_receipt"}, "generation receipt binding inventory differs")
    verify_binding(bindings.get("candidate_index"), attempt / "generation/candidates.json", "generation candidate index", FORMAL_SHA256["generation/candidates.json"])
    verify_binding(bindings.get("complete_marker"), attempt / "generation/candidate_generation_complete.json", "generation marker", FORMAL_SHA256["generation/candidate_generation_complete.json"])
    verify_binding(bindings.get("formal_submission_receipt"), attempt / "submission_receipt.json", "generation formal receipt", FORMAL_SHA256["submission_receipt.json"])
    execution_binding = verify_binding(bindings.get("execution_manifest"), None, "generation execution manifest", EXECUTION_RAW_SHA256)
    require(execution_binding.name == EXECUTION_RELATIVE.name, "generation execution manifest path differs")
    contract = require_mapping(receipt.get("contract_validation"), "generation contract validation")
    require_exact(contract.get("job_id"), CONTRACT_JOB, "generation contract Job ID")
    verify_binding(contract.get("receipt"), None, "generation contract receipt", CONTRACT_RECEIPT_SHA256)


def verify_scoring_receipt(receipt: Mapping[str, Any], attempt: Path) -> None:
    require_exact(receipt.get("schema_id"), "qwen3vl_p2_oracle_scoring_receipt_v2", "scoring receipt schema")
    require_exact(receipt.get("status"), "complete", "scoring receipt status")
    require_exact(receipt.get("job_id"), SCORING_JOB, "scoring receipt Job ID")
    require_exact(receipt.get("generation_job_id"), GENERATION_JOB, "scoring generation Job ID")
    require_exact(receipt.get("generation_state"), "COMPLETED 0:0 CA0803P2Generate guisongxia gpu48", "scoring generation state")
    require_exact(receipt.get("freeze_commit_b"), FREEZE_B, "scoring receipt freeze")
    require_exact(receipt.get("implementation_commit_a"), IMPLEMENTATION_A, "scoring receipt implementation")
    require_exact(receipt.get("worker_sha256"), SCORING_WORKER_SHA256, "scoring receipt worker")
    require(receipt.get("gt_access_after_complete_candidate_validation") is True, "scoring GT boundary differs")
    require(receipt.get("proceed_to_c_iso_and_t_traj_data_construction") is False, "scoring receipt gate differs")
    require_exact(receipt.get("candidate_flow_counts"), SCORED_FLOW_COUNTS, "scoring flow counts")
    bindings = require_mapping(receipt.get("bindings"), "scoring receipt bindings")
    expected = {
        "decision": (attempt / "scoring/actionability_decision.json", FORMAL_SHA256["scoring/actionability_decision.json"]),
        "formal_submission_receipt": (attempt / "submission_receipt.json", FORMAL_SHA256["submission_receipt.json"]),
        "generation_receipt": (attempt / "generation_receipt.json", FORMAL_SHA256["generation_receipt.json"]),
        "runtime_scoring_manifest": (attempt / "scoring/runtime_scoring_manifest.json", FORMAL_SHA256["scoring/runtime_scoring_manifest.json"]),
        "score_report": (attempt / "scoring/score_report.json", FORMAL_SHA256["scoring/score_report.json"]),
    }
    require(set(bindings) == set(expected) | {"preregistration", "execution_manifest"}, "scoring receipt binding inventory differs")
    for name, (path, digest) in expected.items():
        verify_binding(bindings.get(name), path, f"scoring {name}", digest)
    verify_binding(bindings.get("preregistration"), None, "scoring preregistration", PREREG_RAW_SHA256)
    verify_binding(bindings.get("execution_manifest"), None, "scoring execution manifest", EXECUTION_RAW_SHA256)


def verify_statuses(attempt: Path) -> None:
    generation_path = attempt / f"logs/generation-status-{GENERATION_JOB}.txt"
    scoring_path = attempt / f"logs/scoring-status-{SCORING_JOB}.txt"
    generation = parse_status(generation_path)
    scoring = parse_status(scoring_path)
    require_exact(
        {key: generation.get(key) for key in ("status", "exit_code", "job_id", "job_name", "job_user", "commit", "implementation_commit", "worker_sha256", "result", "error")},
        {"status": "success", "exit_code": "0", "job_id": GENERATION_JOB, "job_name": "CA0803P2Generate", "job_user": "guisongxia", "commit": FREEZE_B, "implementation_commit": IMPLEMENTATION_A, "worker_sha256": GENERATION_WORKER_SHA256, "result": "p2_candidate_generation_complete", "error": ""},
        "generation status",
    )
    require_exact(
        {key: scoring.get(key) for key in ("status", "exit_code", "job_id", "job_name", "job_user", "generation_job_id", "generation_state", "commit", "implementation_commit", "worker_sha256", "result", "error")},
        {"status": "success", "exit_code": "0", "job_id": SCORING_JOB, "job_name": "CA0803P2Score", "job_user": "guisongxia", "generation_job_id": GENERATION_JOB, "generation_state": "COMPLETED 0:0 CA0803P2Generate guisongxia gpu48", "commit": FREEZE_B, "implementation_commit": IMPLEMENTATION_A, "worker_sha256": SCORING_WORKER_SHA256, "result": "p2_offline_scoring_complete", "error": ""},
        "scoring status",
    )
    exact_path(generation.get("output"), attempt / "generation", "generation status output")
    exact_path(generation.get("receipt"), attempt / "generation_receipt.json", "generation status receipt")
    exact_path(scoring.get("output"), attempt / "scoring", "scoring status output")
    exact_path(scoring.get("receipt"), attempt / "scoring_receipt.json", "scoring status receipt")


def load_mask(root: Path, spec_value: Any, location: str) -> np.ndarray:
    spec = require_mapping(spec_value, location)
    require(set(spec) == {"path", "sha256", "mask_sha256", "shape"}, f"{location} keys differ")
    path = relative_file(root, spec.get("path"), f"{location}.path")
    require(sha256_file(path) == require_sha(spec.get("sha256"), f"{location}.sha256"), f"{location} raw SHA256 differs")
    try:
        mask = np.asarray(np.load(path, allow_pickle=False), dtype=bool)
    except (OSError, ValueError) as error:
        raise ValidationError(f"cannot load {location}: {error}") from error
    require(mask.ndim == 2, f"{location} must be a two-dimensional mask")
    require_exact(list(mask.shape), spec.get("shape"), f"{location} shape")
    require(mask_sha256(mask) == require_sha(spec.get("mask_sha256"), f"{location}.mask_sha256"), f"{location} semantic SHA256 differs")
    return mask


def verify_candidate_audit(record: Mapping[str, Any], candidate_id: str) -> None:
    gate = require_mapping(record.get("hard_gate"), f"{candidate_id}.hard_gate")
    no_qwen = require_mapping(record.get("no_qwen_audit"), f"{candidate_id}.no_qwen_audit")
    passed, rejected = gate.get("passed"), gate.get("rejected")
    require(type(passed) is bool and type(rejected) is bool and passed != rejected, f"{candidate_id} gate flags differ")
    require(gate.get("verifier_skipped_by_hard_gate") == rejected, f"{candidate_id} gate skip flag differs")
    reasons = gate.get("rejection_reasons")
    require(isinstance(reasons, list) and bool(reasons) == rejected, f"{candidate_id} rejection reasons differ")
    require(no_qwen.get("qwen_invoked") is False, f"{candidate_id} invoked Qwen")
    require(type(no_qwen.get("initial_stub_calls")) is int and no_qwen.get("initial_stub_calls") == 1, f"{candidate_id} initial stub calls differ")
    require(type(no_qwen.get("candidate_stub_calls")) is int and no_qwen.get("candidate_stub_calls") == (0 if rejected else 1), f"{candidate_id} candidate stub calls differ")
    require_exact(no_qwen.get("decision_path"), "pre_qwen_hard_gate" if rejected else "eligible_for_qwen_not_evaluated", f"{candidate_id} decision path")
    evidence = require_mapping(record.get("tool_evidence"), f"{candidate_id}.tool_evidence")
    expected_tool = "sam3" if record.get("action") == "box" else "simpleclick"
    expected_mode = "box" if record.get("action") == "box" else "point"
    require_exact(evidence.get("tool"), expected_tool, f"{candidate_id} tool")
    require(isinstance(evidence.get("tool_input"), Mapping), f"{candidate_id} tool input is missing")
    tool_result = require_mapping(evidence.get("tool_result"), f"{candidate_id} tool result")
    require_exact(tool_result.get("worker_mode"), expected_mode, f"{candidate_id} worker mode")
    worker_report = require_mapping(tool_result.get("worker_report"), f"{candidate_id} worker report")
    require_exact(worker_report.get("status"), "success", f"{candidate_id} tool status")
    require_exact(worker_report.get("mode"), expected_mode, f"{candidate_id} tool report mode")
    require(evidence.get("candidate_accepted") is False, f"{candidate_id} no-Qwen sentinel acceptance differs")
    require_exact(evidence.get("pairwise_comparison"), "worse", f"{candidate_id} no-Qwen pairwise sentinel")
    require(evidence.get("verifier_skipped_by_hard_gate") == rejected, f"{candidate_id} verifier skip evidence differs")
    verifier = require_mapping(evidence.get("verifier_evidence"), f"{candidate_id} verifier evidence")
    require_exact(verifier.get("type"), "runtime_hard_gate_precheck" if rejected else "p2_no_qwen_reject_verifier", f"{candidate_id} verifier evidence type")
    if rejected:
        require(verifier.get("verifier_skipped") is True and verifier.get("rejection_reasons") == reasons, f"{candidate_id} hard-gate verifier evidence differs")
    else:
        require(verifier.get("qwen_invoked") is False, f"{candidate_id} verifier invoked Qwen")


def verify_generation_and_load_masks(
    attempt: Path, execution: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], list[tuple[dict[str, Any], np.ndarray]]]:
    generation_root = canonical_path(attempt / "generation", "generation root")
    marker = read_json(generation_root / "candidate_generation_complete.json", "generation marker")
    index = read_json(generation_root / "candidates.json", "candidate index")
    require_exact(marker.get("schema_id"), "qwen3vl_p2_candidate_generation_complete_v1", "marker schema")
    require_exact(marker.get("execution_status"), "completed", "marker status")
    require(marker.get("gt_loaded") is False, "marker GT flag differs")
    require_exact(marker.get("ordered_samples"), list(ORDERED_SAMPLES), "marker sample order")
    require_exact(marker.get("counts"), COUNTS, "marker counts")
    require_exact(marker.get("p2_runtime_commit"), FREEZE_B, "marker runtime commit")
    require_exact(marker.get("execution_manifest"), {"sha256": EXECUTION_RAW_SHA256}, "marker execution binding")
    require_exact(marker.get("preregistration_canonical_sha256"), PREREG_CANONICAL_SHA256, "marker preregistration binding")
    marker_index = require_mapping(marker.get("candidate_index"), "marker candidate index")
    require_exact(marker_index, {"path": "candidates.json", "sha256": FORMAL_SHA256["generation/candidates.json"]}, "marker candidate index")

    require_exact(index.get("schema_id"), "qwen3vl_p2_oracle_candidates_v1", "candidate index schema")
    require(index.get("gt_loaded") is False, "candidate index GT flag differs")
    require_exact(index.get("ordered_samples"), list(ORDERED_SAMPLES), "candidate index sample order")
    require_exact(index.get("counts"), COUNTS, "candidate index counts")
    require_exact(index.get("p2_runtime_commit"), FREEZE_B, "candidate index runtime commit")
    require_exact(index.get("source_freeze_commit"), P1_FREEZE_B4, "candidate index source freeze")
    require_exact(index.get("preregistration_canonical_sha256"), PREREG_CANONICAL_SHA256, "candidate index preregistration")
    execution_identity = require_mapping(index.get("execution_manifest"), "candidate index execution")
    require_exact(execution_identity.get("sha256"), EXECUTION_RAW_SHA256, "candidate index execution raw hash")
    require_exact(execution_identity.get("canonical_sha256"), EXECUTION_CANONICAL_SHA256, "candidate index execution canonical hash")

    initial_specs = require_mapping(index.get("initial_states"), "candidate initial states")
    require(list(initial_specs) == list(ORDERED_SAMPLES), "candidate initial-state order differs")
    initial_masks: dict[str, np.ndarray] = {}
    for sample in ORDERED_SAMPLES:
        state = require_mapping(initial_specs.get(sample), f"initial.{sample}")
        require(set(state) == {"t1_mask", "t2_mask", "change_mask"}, f"initial.{sample} inventory differs")
        t1 = load_mask(generation_root, state.get("t1_mask"), f"initial.{sample}.t1")
        t2 = load_mask(generation_root, state.get("t2_mask"), f"initial.{sample}.t2")
        change = load_mask(generation_root, state.get("change_mask"), f"initial.{sample}.change")
        manifest_state = require_mapping(require_mapping(execution.get("samples"), "execution samples").get(sample), f"execution sample {sample}")
        initial_hashes = require_mapping(manifest_state.get("initial_state_mask_sha256"), f"execution initial hashes {sample}")
        require_exact(
            {"t1": mask_sha256(t1), "t2": mask_sha256(t2), "change": mask_sha256(change)},
            initial_hashes,
            f"initial semantic state {sample}",
        )
        initial_masks[sample] = change

    identities = index.get("candidates")
    require(isinstance(identities, list) and len(identities) == 36, "candidate inventory size differs")
    expected_actions = [
        action
        for sample in ORDERED_SAMPLES
        for action in require_mapping(require_mapping(execution.get("samples"), "execution samples").get(sample), f"execution sample {sample}").get("actions", [])
    ]
    require(len(expected_actions) == 36, "execution action count differs")
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    seen_ids: set[str] = set()
    state_identities: set[tuple[str, str, str, str]] = set()
    tier_counts: Counter[str] = Counter()
    for number, (identity_value, expected_action) in enumerate(zip(identities, expected_actions)):
        identity = require_mapping(identity_value, f"candidate identity {number}")
        record_path = relative_file(generation_root, identity.get("record_path"), f"candidate identity {number}.record_path")
        require(sha256_file(record_path) == require_sha(identity.get("record_sha256"), f"candidate identity {number}.record_sha256"), f"candidate record {number} raw SHA256 differs")
        record = read_json(record_path, f"candidate record {number}")
        candidate_id = record.get("candidate_id")
        require(isinstance(candidate_id, str) and candidate_id not in seen_ids, f"candidate ID {number} is missing or duplicated")
        seen_ids.add(candidate_id)
        require_exact(identity.get("candidate_id"), candidate_id, f"candidate identity {number} ID")
        require_exact(identity.get("sample"), record.get("sample"), f"candidate identity {number} sample")
        require_exact(identity.get("tier"), record.get("tier"), f"candidate identity {number} tier")
        for key, value in require_mapping(expected_action, f"execution action {number}").items():
            require_exact(record.get(key), value, f"candidate {candidate_id}.{key}")
        tier = record.get("tier")
        require(tier in ("primary", "diagnostic_executor_only"), f"candidate {candidate_id} tier differs")
        tier_counts[str(tier)] += 1
        mask_specs = require_mapping(record.get("candidate_masks"), f"candidate {candidate_id} masks")
        require(set(mask_specs) == {"t1", "t2", "change"}, f"candidate {candidate_id} mask inventory differs")
        t1 = load_mask(generation_root, mask_specs.get("t1"), f"candidate.{candidate_id}.t1")
        t2 = load_mask(generation_root, mask_specs.get("t2"), f"candidate.{candidate_id}.t2")
        change = load_mask(generation_root, mask_specs.get("change"), f"candidate.{candidate_id}.change")
        hashes = {"t1": mask_sha256(t1), "t2": mask_sha256(t2), "change": mask_sha256(change)}
        require_exact(record.get("candidate_state_mask_sha256"), hashes, f"candidate {candidate_id} state hashes")
        require_exact(identity.get("candidate_state_mask_sha256"), hashes, f"candidate identity {candidate_id} state hashes")
        state_identity = (str(record.get("sample")), hashes["t1"], hashes["t2"], hashes["change"])
        require(state_identity not in state_identities, f"duplicate candidate semantic state: {candidate_id}")
        state_identities.add(state_identity)
        verify_candidate_audit(record, candidate_id)
        candidates.append((record, change))
    require_exact(dict(tier_counts), {"primary": 27, "diagnostic_executor_only": 9}, "candidate tier counts")
    require(len(seen_ids) == 36 and len(state_identities) == 36, "candidate uniqueness differs")
    return marker, index, initial_masks, candidates


def metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, int | float]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    require(prediction.shape == reference.shape, "prediction/reference shapes differ")
    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, np.logical_not(reference)).sum())
    fn = int(np.logical_and(np.logical_not(prediction), reference).sum())
    tn = int(np.logical_and(np.logical_not(prediction), np.logical_not(reference)).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classify_candidate(
    initial: Mapping[str, int | float], candidate: Mapping[str, int | float], epsilon: float
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


def gate_leaf(label: str, rejected: bool) -> str:
    leaves = {
        ("beneficial", True): "hard_gate_false_reject",
        ("beneficial", False): "beneficial_gate_pass",
        ("harmful", True): "hard_gate_correct_reject",
        ("harmful", False): "harmful_gate_pass",
        ("indeterminate", True): "indeterminate_hard_gate_rejected",
        ("indeterminate", False): "indeterminate_gate_pass",
    }
    try:
        return leaves[(label, rejected)]
    except KeyError as error:
        raise ValidationError(f"unsupported oracle/gate state: {label}/{rejected}") from error


def load_labels_after_candidates_verified(scoring_manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    labels = require_mapping(scoring_manifest.get("labels"), "scoring labels")
    require(list(labels) == list(ORDERED_SAMPLES), "scoring label order differs")
    result: dict[str, np.ndarray] = {}
    for sample in ORDERED_SAMPLES:
        spec = require_mapping(labels.get(sample), f"label {sample}")
        require(set(spec) == {"path", "sha256"}, f"label {sample} identity keys differ")
        path = canonical_path(Path(str(spec.get("path"))), f"label {sample}")
        require(sha256_file(path) == require_sha(spec.get("sha256"), f"label {sample}.sha256"), f"label {sample} SHA256 differs")
        try:
            with Image.open(path) as image:
                result[sample] = np.asarray(image) > 0
        except OSError as error:
            raise ValidationError(f"cannot load label {sample}: {error}") from error
        require(result[sample].ndim == 2, f"label {sample} must be two-dimensional")
    return result


def build_target_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for sample in TARGET_SAMPLES:
        primary = [row for row in rows if row["sample"] == sample and row["tier"] == "primary"]
        beneficial = [row for row in primary if row["oracle_label"] == "beneficial"]
        gate_pass = [row for row in beneficial if row["hard_gate"]["passed"]]
        best_primary = max(primary, key=lambda row: (row["metric_and_count_deltas"]["iou"], row["metric_and_count_deltas"]["f1"], row["candidate_id"]), default=None)
        best_beneficial = max(gate_pass, key=lambda row: (row["metric_and_count_deltas"]["iou"], row["metric_and_count_deltas"]["f1"], row["candidate_id"]), default=None)
        decision = "primary_actionable_and_gate_pass" if gate_pass else "primary_actionable_but_all_beneficial_hard_gated" if beneficial else "no_primary_beneficial_candidate_in_frozen_enumeration"
        targets[sample] = {
            "decision": decision,
            "primary_candidate_count": len(primary),
            "primary_beneficial_count": len(beneficial),
            "primary_beneficial_gate_pass_count": len(gate_pass),
            "best_primary_candidate_id": best_primary["candidate_id"] if best_primary else None,
            "best_primary_candidate_oracle_label": best_primary["oracle_label"] if best_primary else None,
            "best_primary_candidate_gate_passed": best_primary["hard_gate"]["passed"] if best_primary else None,
            "best_primary_candidate_deltas": best_primary["metric_and_count_deltas"] if best_primary else None,
            "best_primary_beneficial_candidate_id": best_beneficial["candidate_id"] if best_beneficial else None,
            "best_primary_beneficial_deltas": best_beneficial["metric_and_count_deltas"] if best_beneficial else None,
        }
    return targets


def verify_scoring(
    result: Mapping[str, Any],
    attempt: Path,
    initial_masks: Mapping[str, np.ndarray],
    candidates: list[tuple[dict[str, Any], np.ndarray]],
) -> None:
    scoring_manifest = read_json(attempt / "scoring/runtime_scoring_manifest.json", "runtime scoring manifest")
    report = read_json(attempt / "scoring/score_report.json", "score report")
    decision = read_json(attempt / "scoring/actionability_decision.json", "actionability decision")
    require_exact(scoring_manifest.get("schema_id"), "qwen3vl_p2_oracle_scoring_manifest_v1", "scoring manifest schema")
    require_exact(scoring_manifest.get("status"), "frozen", "scoring manifest status")
    require_exact(scoring_manifest.get("p2_runtime_commit"), FREEZE_B, "scoring runtime commit")
    require_exact(scoring_manifest.get("ordered_samples"), list(ORDERED_SAMPLES), "scoring sample order")
    require_exact(scoring_manifest.get("target_samples"), list(TARGET_SAMPLES), "scoring targets")
    require_exact(scoring_manifest.get("expected_candidate_counts"), COUNTS, "scoring candidate counts")
    require_exact(scoring_manifest.get("candidate_flow_counts"), FLOW_COUNTS, "scoring flow counts")
    require_exact(scoring_manifest.get("candidate_generation_complete_sha256"), FORMAL_SHA256["generation/candidate_generation_complete.json"], "scoring marker binding")
    require_exact(scoring_manifest.get("candidate_index_sha256"), FORMAL_SHA256["generation/candidates.json"], "scoring index binding")
    require_exact(scoring_manifest.get("execution_manifest_sha256"), EXECUTION_RAW_SHA256, "scoring execution binding")
    require_exact(scoring_manifest.get("preregistration_canonical_sha256"), PREREG_CANONICAL_SHA256, "scoring preregistration binding")
    require_exact(scoring_manifest.get("comparison_epsilon"), COMPARISON_EPSILON, "scoring epsilon")
    require_exact(
        scoring_manifest.get("gt_access_boundary"),
        {
            "generation_verified_before_manifest_creation": True,
            "all_candidate_records_and_masks_verified_before_manifest_creation": True,
            "runtime_scoring_manifest_not_available_to_generator": True,
        },
        "scoring manifest GT boundary",
    )

    # This is intentionally the first label I/O in the validator.  Marker, index,
    # all 36 records, all raw hashes, and all semantic mask hashes were verified by
    # verify_generation_and_load_masks before this function was entered.
    labels = load_labels_after_candidates_verified(scoring_manifest)
    initial_metrics = {sample: metrics(initial_masks[sample], labels[sample]) for sample in ORDERED_SAMPLES}
    epsilon = float(scoring_manifest["comparison_epsilon"])
    recomputed: list[dict[str, Any]] = []
    for record, candidate_mask in candidates:
        sample = str(record["sample"])
        candidate_metrics = metrics(candidate_mask, labels[sample])
        oracle_label, deltas = classify_candidate(initial_metrics[sample], candidate_metrics, epsilon)
        rejected = bool(record["hard_gate"]["rejected"])
        recomputed.append(
            {
                "candidate_id": record["candidate_id"],
                "sample": sample,
                "tier": record["tier"],
                "region_id": record["region_id"],
                "action": record["action"],
                "target_view": record["target_view"],
                "candidate_state_mask_sha256": record["candidate_state_mask_sha256"],
                "hard_gate": record["hard_gate"],
                "decision_path": record["no_qwen_audit"]["decision_path"],
                "initial_metrics": initial_metrics[sample],
                "candidate_metrics": candidate_metrics,
                "metric_and_count_deltas": deltas,
                "oracle_label": oracle_label,
                "gate_leaf": gate_leaf(oracle_label, rejected),
            }
        )

    require_exact(report.get("schema_id"), "qwen3vl_p2_oracle_score_report_v1", "score report schema")
    require_exact(report.get("comparison_epsilon"), COMPARISON_EPSILON, "score report epsilon")
    require_exact(report.get("initial_metrics"), initial_metrics, "score report initial metrics")
    require_exact(report.get("candidate_count"), 36, "score report candidate count")
    require_exact(report.get("action_spec_count"), 36, "score report action count")
    require_exact(report.get("unique_candidate_state_count"), 36, "score report unique states")
    require_exact(report.get("candidate_state_duplicate_count"), 0, "score report duplicate states")
    require_exact(report.get("state_identity_policy"), "sample+t1_mask_sha256+t2_mask_sha256+change_mask_sha256", "score report state identity policy")
    require_exact(
        report.get("gt_access_boundary"),
        {
            "candidate_generation_complete_verified_before_label_access": True,
            "all_candidate_hashes_verified_before_label_access": True,
            "generator_reported_gt_loaded": False,
        },
        "score report GT boundary",
    )
    require_exact(
        report.get("qwen_evaluation"),
        {
            "qwen_invoked": False,
            "qwen_false_accept_reject_available": False,
            "gate_pass_semantics": "eligible_for_qwen_not_evaluated",
        },
        "score report Qwen boundary",
    )
    require_exact(report.get("p2_pass_stop_decision_scope"), "primary_only", "score report decision scope")
    require_exact(report.get("candidates"), recomputed, "independently recomputed candidate scores")

    all_oracle = dict(Counter(row["oracle_label"] for row in recomputed))
    primary_rows = [row for row in recomputed if row["tier"] == "primary"]
    diagnostic_rows = [row for row in recomputed if row["tier"] == "diagnostic_executor_only"]
    primary_oracle = dict(Counter(row["oracle_label"] for row in primary_rows))
    diagnostic_oracle = dict(Counter(row["oracle_label"] for row in diagnostic_rows))
    all_oracle = {name: all_oracle.get(name, 0) for name in ("beneficial", "harmful", "indeterminate")}
    primary_oracle = {name: primary_oracle.get(name, 0) for name in ("beneficial", "harmful", "indeterminate")}
    diagnostic_oracle = {name: diagnostic_oracle.get(name, 0) for name in ("beneficial", "harmful", "indeterminate")}
    leaves = dict(Counter(row["gate_leaf"] for row in recomputed))
    require_exact(all_oracle, ORACLE_COUNTS, "recomputed all-tier oracle counts")
    require_exact(primary_oracle, PRIMARY_ORACLE_COUNTS, "recomputed primary oracle counts")
    require_exact(diagnostic_oracle, DIAGNOSTIC_ORACLE_COUNTS, "recomputed diagnostic oracle counts")
    require_exact(leaves, GATE_LEAF_COUNTS, "recomputed gate-leaf counts")
    require_exact(report.get("oracle_counts"), ORACLE_COUNTS, "score report oracle counts")
    require_exact(report.get("gate_leaf_counts"), GATE_LEAF_COUNTS, "score report gate-leaf counts")
    require_exact(require_mapping(report.get("aggregate_all_tiers_diagnostic_only"), "all-tier summary").get("oracle_counts"), ORACLE_COUNTS, "all-tier summary oracle counts")
    require_exact(require_mapping(report.get("aggregate_all_tiers_diagnostic_only"), "all-tier summary").get("gate_leaf_counts"), GATE_LEAF_COUNTS, "all-tier summary gate leaves")
    by_tier = require_mapping(report.get("by_tier"), "score report tiers")
    primary_summary = require_mapping(by_tier.get("primary"), "primary score summary")
    diagnostic_summary = require_mapping(by_tier.get("diagnostic_executor_only"), "diagnostic score summary")
    require_exact(primary_summary.get("candidate_count"), 27, "primary score count")
    require_exact(primary_summary.get("oracle_counts"), PRIMARY_ORACLE_COUNTS, "primary score oracle counts")
    require_exact(diagnostic_summary.get("candidate_count"), 9, "diagnostic score count")
    require_exact(diagnostic_summary.get("oracle_counts"), DIAGNOSTIC_ORACLE_COUNTS, "diagnostic score oracle counts")

    primary_beneficial = {row["candidate_id"] for row in primary_rows if row["oracle_label"] == "beneficial"}
    diagnostic_beneficial = {row["candidate_id"] for row in diagnostic_rows if row["oracle_label"] == "beneficial"}
    require_exact(primary_beneficial, PRIMARY_BENEFICIAL_IDS, "primary beneficial candidate IDs")
    require_exact(diagnostic_beneficial, DIAGNOSTIC_BENEFICIAL_IDS, "diagnostic beneficial candidate IDs")
    rejection_reasons: Counter[str] = Counter()
    for record, _mask in candidates:
        rejection_reasons.update(record["hard_gate"]["rejection_reasons"])
    require_exact(dict(rejection_reasons), GATE_REJECTION_REASONS, "hard-gate rejection reasons")

    targets = build_target_decisions(recomputed)
    require_exact(report.get("target_primary_actionability"), targets, "score report target decisions")
    verify_decision(decision, targets)
    verify_result_targets(result, initial_metrics, targets, recomputed)


def verify_decision(decision: Mapping[str, Any], targets: Mapping[str, Any]) -> None:
    require_exact(decision.get("schema_id"), "qwen3vl_p2_actionability_decision_v1", "decision schema")
    require_exact(decision.get("status"), "complete", "decision status")
    require_exact(decision.get("decision_basis"), "primary_candidates_only", "decision basis")
    require(decision.get("diagnostic_candidates_excluded") is True, "diagnostic candidates were not excluded")
    require_exact(decision.get("target_samples"), list(TARGET_SAMPLES), "decision targets")
    require_exact(decision.get("target_results"), targets, "decision target results")
    require(decision.get("proceed_to_c_iso_and_t_traj_data_construction") is False, "P2 proceed gate is not false")
    require_exact(decision.get("next_action"), "stop_training_and_review_frozen_action_space", "decision next action")
    require_exact(decision.get("beneficial_but_hard_gated_samples"), [], "decision hard-gated targets")
    require_exact(decision.get("no_primary_beneficial_candidate_samples"), ["test_1_1"], "decision missing-beneficial targets")
    require(decision.get("gate_modification_authorized") is False, "gate modification was authorized")
    require_exact(decision.get("scope_limit"), "decision applies only to the frozen finite single-step CORE3 inventory", "decision scope")


def verify_result_targets(
    result: Mapping[str, Any],
    initial_metrics: Mapping[str, Mapping[str, int | float]],
    targets: Mapping[str, Mapping[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    result_targets = require_mapping(result.get("target_results"), "result targets")
    require(list(result_targets) == list(TARGET_SAMPLES), "result target order differs")
    test_1 = require_mapping(result_targets.get("test_1_1"), "result target test_1_1")
    test_50 = require_mapping(result_targets.get("test_50_8"), "result target test_50_8")
    metric_projection = ("iou", "f1", "tp", "fp", "fn")
    require_exact(test_1.get("initial_metrics"), {key: initial_metrics["test_1_1"][key] for key in metric_projection}, "test_1_1 initial metrics")
    require_exact(test_50.get("initial_metrics"), {key: initial_metrics["test_50_8"][key] for key in metric_projection}, "test_50_8 initial metrics")
    for sample, payload in (("test_1_1", test_1), ("test_50_8", test_50)):
        for key in (
            "primary_candidate_count",
            "primary_beneficial_count",
            "primary_beneficial_gate_pass_count",
            "decision",
        ):
            require_exact(payload.get(key), targets[sample][key], f"result {sample}.{key}")
    for key in (
        "best_primary_candidate_id",
        "best_primary_candidate_oracle_label",
        "best_primary_candidate_gate_passed",
        "best_primary_candidate_deltas",
    ):
        require_exact(test_1.get(key), targets["test_1_1"][key], f"result test_1_1.{key}")
    for key in ("best_primary_beneficial_candidate_id", "best_primary_beneficial_deltas"):
        require_exact(test_50.get(key), targets["test_50_8"][key], f"result test_50_8.{key}")
    require_exact(test_1.get("diagnostic_beneficial_candidate_excluded_from_primary_decision"), "test_1_1--diagnostic_executor_only--r0--positive_point--t1", "test_1_1 diagnostic ID")
    require_exact(test_1.get("diagnostic_candidate_hard_gate_rejection_reason"), "point_proposal_combined_change_gate_exceeded", "test_1_1 diagnostic gate reason")
    require_exact(test_50.get("beneficial_but_hard_gated_candidate_id"), "test_50_8--primary--s8--box--t1", "test_50_8 hard-gated ID")
    require_exact(test_50.get("beneficial_but_hard_gated_rejection_reason"), "component_count_delta_exceeded", "test_50_8 hard-gated reason")
    ids = {row["candidate_id"]: row for row in rows}
    require(ids["test_1_1--diagnostic_executor_only--r0--positive_point--t1"]["gate_leaf"] == "hard_gate_false_reject", "test_1_1 diagnostic leaf differs")
    require(ids["test_50_8--primary--s8--box--t1"]["gate_leaf"] == "hard_gate_false_reject", "test_50_8 hard-gated leaf differs")
    require(ids["test_50_8--primary--s8--box--t2"]["gate_leaf"] == "beneficial_gate_pass", "test_50_8 gate-pass leaf differs")


def verify_parent_closeout_receipt(
    receipt_path: Path,
    expected_parent_commit: str,
    result_path: Path,
    repo: Path,
    formal_attempt: Path,
) -> None:
    require_sha(expected_parent_commit, "expected parent commit", commit=True)
    receipt_path = canonical_path(receipt_path, "parent closeout receipt")
    receipt = read_json(receipt_path, "parent closeout receipt")
    expected_keys = {
        "schema_id",
        "status",
        "job_id",
        "job_name",
        "job_user",
        "node",
        "partition",
        "validation_mode",
        "commit",
        "parent_commit",
        "freeze_commit_b",
        "worker_sha256",
        "validator_sha256",
        "resources",
        "checks",
        "bindings",
    }
    require(set(receipt) == expected_keys, "parent closeout receipt inventory differs")
    require_exact(receipt.get("schema_id"), "qwen3vl_p2_closeout_validation_receipt_v1", "parent closeout schema")
    require_exact(receipt.get("status"), "complete", "parent closeout status")
    require_exact(receipt.get("validation_mode"), "candidate", "parent closeout mode")
    require_exact(receipt.get("commit"), expected_parent_commit, "parent closeout commit")
    require_exact(receipt.get("parent_commit"), FREEZE_B, "parent closeout parent")
    require_exact(receipt.get("freeze_commit_b"), FREEZE_B, "parent closeout freeze")
    require_exact(receipt.get("job_name"), "CA0803P2Closeout", "parent closeout job name")
    require_exact(receipt.get("job_user"), "guisongxia", "parent closeout user")
    require_exact(receipt.get("node"), "gpu48", "parent closeout node")
    require_exact(receipt.get("partition"), "gre", "parent closeout partition")
    job_id = receipt.get("job_id")
    require(isinstance(job_id, str) and job_id.isdecimal(), "parent closeout Job ID is malformed")
    require_sha(receipt.get("worker_sha256"), "parent closeout worker")
    require_sha(receipt.get("validator_sha256"), "parent closeout validator")
    historical_worker = exact_commit_blob(
        repo,
        expected_parent_commit,
        "tools/run_qwen3vl_p2_closeout_validation.slurm",
        "parent closeout worker",
    )
    historical_validator = exact_commit_blob(
        repo,
        expected_parent_commit,
        "tools/validate_qwen3vl_p2_result.py",
        "parent result validator",
    )
    require_exact(
        receipt.get("worker_sha256"),
        hashlib.sha256(historical_worker).hexdigest(),
        "parent/exact-commit closeout worker",
    )
    require_exact(
        receipt.get("validator_sha256"),
        hashlib.sha256(historical_validator).hexdigest(),
        "parent/exact-commit result validator",
    )
    require_exact(receipt.get("resources"), {"gpus": 0, "cpus": 2, "memory": "16G"}, "parent closeout resources")
    require_exact(receipt.get("checks"), CLOSEOUT_CHECKS, "parent closeout checks")
    bindings = require_mapping(receipt.get("bindings"), "parent closeout bindings")
    require(set(bindings) == CLOSEOUT_CANDIDATE_BINDINGS, "parent closeout binding inventory differs")

    # Candidate and acceptance commits intentionally keep the scientific result
    # byte-identical.  The candidate worktree path may differ, so compare its
    # bound bytes with the current result rather than imposing the current path.
    bound_result = verify_binding(bindings.get("result_json"), None, "parent result JSON")
    require(sha256_file(bound_result) == sha256_file(result_path), "parent/current result JSON differs")
    # Acceptance D may append its receipt record to the Markdown/handoff.  Their
    # C-era bindings must remain independently valid, but are intentionally not
    # required to equal the D-era documentation bytes.
    verify_binding(bindings.get("result_markdown"), None, "parent result Markdown")
    verify_binding(bindings.get("handoff"), None, "parent handoff")
    # The machine result and registry/workbook were frozen byte-for-byte across
    # C -> D.  Later maintenance may relocate evidence paths, so validate the
    # receipt against the exact candidate commit's Git blobs rather than against
    # mutable current-checkout bytes.  Current registry policy is independently
    # exercised by the registry contract suite.
    for name, relative in (
        ("registry", REGISTRY_RELATIVE),
        ("workbook", WORKBOOK_RELATIVE),
    ):
        bound = verify_binding(bindings.get(name), None, f"parent {name}")
        historical = exact_commit_blob(
            repo,
            expected_parent_commit,
            relative.as_posix(),
            f"parent {name}",
        )
        require(
            sha256_file(bound) == hashlib.sha256(historical).hexdigest()
            and bound.stat().st_size == len(historical),
            f"parent/exact-commit {name} differs",
        )
    expected_formal = {
        "formal_submission_receipt": formal_attempt / "submission_receipt.json",
        "generation_receipt": formal_attempt / "generation_receipt.json",
        "scoring_receipt": formal_attempt / "scoring_receipt.json",
        "score_report": formal_attempt / "scoring/score_report.json",
        "decision": formal_attempt / "scoring/actionability_decision.json",
    }
    for name, path in expected_formal.items():
        verify_binding(bindings.get(name), path, f"parent {name}", sha256_file(path))

    attempt_root = receipt_path.parent.parent
    require(ATTEMPT_RE.fullmatch(attempt_root.name) is not None, "parent closeout attempt name differs")
    expected_status = attempt_root / f"status-{job_id}.txt"
    expected_intent = attempt_root / "submission_intent.json"
    expected_submission = attempt_root / "submission_receipt.json"
    expected_log = attempt_root / f"job-{job_id}/validation_tests.log"
    status_path = verify_binding(bindings.get("status"), expected_status, "parent closeout status binding")
    verify_binding(bindings.get("submission_intent"), expected_intent, "parent closeout intent binding")
    verify_binding(bindings.get("submission_receipt"), expected_submission, "parent closeout submission binding")
    verify_binding(bindings.get("test_log"), expected_log, "parent closeout test-log binding")
    status = parse_status(status_path)
    require_exact(
        {key: status.get(key) for key in ("status", "exit_code", "job_id", "job_name", "job_user", "node", "partition", "validation_mode", "commit", "parent_commit", "freeze_commit_b", "worker_sha256", "validator_sha256", "result", "error")},
        {"status": "success", "exit_code": "0", "job_id": job_id, "job_name": "CA0803P2Closeout", "job_user": "guisongxia", "node": "gpu48", "partition": "gre", "validation_mode": "candidate", "commit": expected_parent_commit, "parent_commit": FREEZE_B, "freeze_commit_b": FREEZE_B, "worker_sha256": receipt["worker_sha256"], "validator_sha256": receipt["validator_sha256"], "result": "p2_closeout_validation_complete", "error": ""},
        "parent closeout status content",
    )
    exact_path(status.get("receipt"), receipt_path, "parent closeout status receipt")
    intent = read_json(expected_intent, "parent closeout submission intent")
    submission = read_json(expected_submission, "parent closeout submission receipt")
    require_exact(intent.get("schema_id"), "qwen3vl_p2_closeout_submission_intent_v1", "parent closeout intent schema")
    require_exact(intent.get("status"), "prepared", "parent closeout intent status")
    require_exact(submission.get("schema_id"), "qwen3vl_p2_closeout_submission_receipt_v1", "parent closeout submission schema")
    require_exact(submission.get("submission_status"), "closeout_submitted", "parent closeout submission status")
    for payload, label in ((intent, "intent"), (submission, "submission")):
        require_exact(payload.get("mode"), "candidate", f"parent closeout {label} mode")
        require_exact(payload.get("expected_commit"), expected_parent_commit, f"parent closeout {label} commit")
        require_exact(payload.get("parent_commit"), FREEZE_B, f"parent closeout {label} parent")
        require_exact(payload.get("freeze_commit_b"), FREEZE_B, f"parent closeout {label} freeze")
        require_exact(payload.get("worker_sha256"), receipt["worker_sha256"], f"parent closeout {label} worker")
        require_exact(payload.get("validator_sha256"), receipt["validator_sha256"], f"parent closeout {label} validator")


def verify_acceptance_document(
    repo: Path,
    parent_receipt_path: Path,
    expected_parent_commit: str,
) -> None:
    """Validate D's no-self-reference acceptance record against C's receipt."""

    parent_receipt_path = canonical_path(parent_receipt_path, "parent closeout receipt")
    # Verify the candidate receipt first.  The acceptance JSON is then checked
    # against the already-authenticated receipt identity, not used to choose it.
    parent_receipt = read_json(parent_receipt_path, "P2 acceptance parent receipt")
    require_exact(parent_receipt.get("schema_id"), "qwen3vl_p2_closeout_validation_receipt_v1", "P2 acceptance parent receipt schema")
    require_exact(parent_receipt.get("status"), "complete", "P2 acceptance parent receipt status")
    require_exact(parent_receipt.get("validation_mode"), "candidate", "P2 acceptance parent receipt mode")
    require_exact(parent_receipt.get("commit"), expected_parent_commit, "P2 acceptance parent receipt commit")

    acceptance_path = canonical_path(repo / ACCEPTANCE_RELATIVE, "P2 acceptance record")
    acceptance = read_json(acceptance_path, "P2 acceptance record")
    require(
        set(acceptance)
        == {
            "schema_id",
            "status",
            "scientific_decision_unchanged",
            "proceed_to_c_iso_and_t_traj_data_construction",
            "candidate_closeout",
            "next_boundary",
        },
        "P2 acceptance top-level inventory differs",
    )
    require_exact(
        acceptance.get("schema_id"),
        "qwen3vl_p2_actionability_acceptance_v1",
        "P2 acceptance schema",
    )
    require_exact(acceptance.get("status"), "accepted", "P2 acceptance status")
    require(
        acceptance.get("scientific_decision_unchanged") is True,
        "P2 acceptance changed the scientific decision",
    )
    require(
        acceptance.get("proceed_to_c_iso_and_t_traj_data_construction") is False,
        "P2 acceptance proceed gate is not false",
    )
    require_exact(
        acceptance.get("next_boundary"),
        "stop_before_c_iso_and_t_traj_data_construction",
        "P2 acceptance next boundary",
    )
    closeout = require_mapping(
        acceptance.get("candidate_closeout"), "P2 acceptance candidate closeout"
    )
    require(
        set(closeout) == {"validated_commit", "validation_mode", "job_id", "receipt"},
        "P2 acceptance candidate-closeout inventory differs",
    )
    require_exact(
        closeout.get("validated_commit"),
        expected_parent_commit,
        "P2 acceptance validated commit",
    )
    require_exact(
        closeout.get("validation_mode"), "candidate", "P2 acceptance validation mode"
    )
    job_id = closeout.get("job_id")
    require(
        isinstance(job_id, str) and job_id.isdecimal(),
        "P2 acceptance candidate Job ID is malformed",
    )
    verify_binding(
        closeout.get("receipt"),
        parent_receipt_path,
        "P2 acceptance candidate receipt",
        sha256_file(parent_receipt_path),
    )
    require_exact(parent_receipt.get("job_id"), job_id, "P2 acceptance parent receipt Job ID")


def validate(
    result_path: Path,
    repo_root: Path,
    formal_attempt: Path,
    *,
    parent_closeout_receipt: Path | None = None,
    expected_parent_commit: str | None = None,
) -> dict[str, Any]:
    require((parent_closeout_receipt is None) == (expected_parent_commit is None), "parent receipt and expected parent commit must be supplied together")
    repo = canonical_path(repo_root, "repository root")
    expected_result = repo / RESULT_RELATIVE
    result_path = exact_path(str(result_path), expected_result, "result")
    attempt = canonical_path(formal_attempt, "formal attempt")
    require(ATTEMPT_RE.fullmatch(attempt.name) is not None, "formal attempt name differs")
    result = read_json(result_path, "P2 result")
    verify_result_claims(result)
    verify_git_lineage(repo, expected_parent_commit)
    acceptance_path = repo / ACCEPTANCE_RELATIVE
    if parent_closeout_receipt is None:
        require(
            not acceptance_path.exists(),
            "candidate C must not contain the acceptance record",
        )
    else:
        require(expected_parent_commit is not None, "acceptance parent commit is missing")
        # The complete parent receipt is authenticated before the D acceptance
        # JSON can consume any of its identities.
        verify_parent_closeout_receipt(
            parent_closeout_receipt,
            expected_parent_commit,
            result_path,
            repo,
            attempt,
        )
        verify_acceptance_document(repo, parent_closeout_receipt, expected_parent_commit)
    prereg, execution = verify_result_manifests(result, repo)
    del prereg
    assert_execution_is_gt_free(execution)
    verify_result_preflight_receipts(result)
    verify_formal_headers(result, attempt, execution)
    _marker, _index, initial_masks, candidates = verify_generation_and_load_masks(attempt, execution)
    verify_scoring(result, attempt, initial_masks, candidates)
    return {
        "mode": "acceptance" if parent_closeout_receipt is not None else "candidate",
        "candidate_count": len(candidates),
        "unique_candidate_state_count": 36,
        "proceed": False,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--result", type=Path, required=True)
    root.add_argument("--repo-root", type=Path, required=True)
    root.add_argument("--formal-attempt", type=Path, required=True)
    root.add_argument("--parent-closeout-receipt", type=Path)
    root.add_argument("--expected-parent-commit")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = validate(
            args.result,
            args.repo_root,
            args.formal_attempt,
            parent_closeout_receipt=args.parent_closeout_receipt,
            expected_parent_commit=args.expected_parent_commit,
        )
    except (
        OSError,
        UnicodeError,
        ValidationError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        print(f"qwen3vl_p2_result: {error}", file=sys.stderr)
        return 3
    print(
        f"status=valid;mode={summary['mode']};candidates={summary['candidate_count']};"
        f"unique_states={summary['unique_candidate_state_count']};proceed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
