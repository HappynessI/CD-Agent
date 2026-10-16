#!/usr/bin/env python3
"""Fail-closed release checks shared by the P2 Slurm submission chain.

This module intentionally does not use Python ``assert``: every condition is a
runtime check even when the interpreter is started with optimization enabled.
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


SHA40 = re.compile(r"[0-9a-f]{40}")
SHA64 = re.compile(r"[0-9a-f]{64}")
ATTEMPT = re.compile(r"attempt-[0-9]{4,}")
COUNTS = {"total": 36, "primary": 27, "diagnostic_executor_only": 9}
FREEZE_FILES = (
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json",
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json",
)
CONTRACT_CHECKS = {
    "clean_detached_exact_b": True,
    "single_parent_freeze_b": True,
    "two_manifest_add_only_freeze": True,
    "manifest_independent_validation": "passed",
    "p1_runtime_unchanged": True,
    "shell_syntax": "passed",
    "p2_targeted_tests": "passed",
    "registry_targeted_tests": "passed",
    "full_regression": "passed",
    "repository_unchanged": True,
}


class ReleaseError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def require_sha(value: Any, length: int, field: str) -> str:
    pattern = SHA40 if length == 40 else SHA64
    require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{field} is malformed")
    return str(value)


def read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read {field}: {error}") from error
    require(isinstance(value, dict), f"{field} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        text=True,
        capture_output=True,
    )


def exact_path(raw: Path, expected: Path, field: str) -> Path:
    absolute = Path(os.path.abspath(raw))
    resolved = raw.resolve(strict=True)
    require(absolute == resolved, f"{field} must be a canonical non-symlink path")
    require(resolved == expected.resolve(strict=True), f"{field} is not the canonical expected path")
    return resolved


def verify_binding(spec: Any, expected_path: Path, field: str, expected_hash: str | None = None) -> str:
    require(isinstance(spec, Mapping), f"{field} binding is missing")
    require(set(spec) == {"path", "sha256", "size_bytes"}, f"{field} binding keys differ")
    path = exact_path(Path(str(spec.get("path"))), expected_path, f"{field}.path")
    observed = sha256_file(path)
    require(spec.get("sha256") == observed, f"{field} binding SHA256 differs")
    require(spec.get("size_bytes") == path.stat().st_size, f"{field} binding size differs")
    if expected_hash is not None:
        require(observed == expected_hash, f"{field} expected SHA256 differs")
    return observed


def parse_status(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        require("=" in line, "status line lacks equals separator")
        key, value = line.split("=", 1)
        require(bool(key) and key not in result, "status contains an empty or duplicate key")
        result[key] = value
    return result


def verify_freeze(repo: Path, freeze: str, implementation: str) -> None:
    freeze = require_sha(freeze, 40, "freeze commit B")
    implementation = require_sha(implementation, 40, "implementation commit A")
    repo = repo.resolve(strict=True)
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    require(head == freeze, "checkout is not exact freeze commit B")
    parents = git(repo, "rev-list", "--parents", "-n", "1", freeze).stdout.split()
    require(parents == [freeze, implementation], "freeze B must have exactly one parent, implementation A")
    changes = tuple(
        line
        for line in git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            freeze,
        ).stdout.splitlines()
        if line
    )
    expected = tuple(f"A\t{path}" for path in FREEZE_FILES)
    require(changes == expected, "freeze B must add exactly the two manifests")
    for relative in FREEZE_FILES:
        absent = git(repo, "cat-file", "-e", f"{implementation}:{relative}", check=False)
        require(absent.returncode != 0, f"manifest already existed in implementation A: {relative}")
        kind = git(repo, "cat-file", "-t", f"{freeze}:{relative}").stdout.strip()
        require(kind == "blob", f"frozen manifest is not a Git blob: {relative}")


def canonical_contract_paths(validation_base: Path, freeze: str, job_id: str, receipt: Path) -> tuple[Path, Path]:
    base = validation_base.resolve(strict=True)
    resolved = receipt.resolve(strict=True)
    attempt = resolved.parent.parent
    require(ATTEMPT.fullmatch(attempt.name) is not None, "contract receipt attempt directory is malformed")
    expected_attempt_parent = base / f"freeze-{freeze}"
    require(attempt.parent == expected_attempt_parent, "contract receipt is outside the exact freeze validation root")
    expected = attempt / f"job-{job_id}" / "contract_validation_receipt.json"
    exact_path(receipt, expected, "contract receipt")
    return attempt, resolved


def verify_contract_submission(
    attempt: Path,
    job_id: str,
    freeze: str,
    implementation: str,
    worker_sha: str,
    prereg_sha: str,
    execution_sha: str,
    expected_user: str,
    expected_node: str,
    expected_job_name: str,
) -> tuple[Path, Path]:
    intent_path = attempt / "submission_intent.json"
    submission_path = attempt / "submission_receipt.json"
    intent = read_json(intent_path, "contract submission intent")
    submission = read_json(submission_path, "contract submission receipt")
    require(intent.get("schema_id") == "qwen3vl_p2_contract_submission_intent_v1", "contract intent schema differs")
    require(intent.get("status") == "prepared", "contract intent status differs")
    require(intent.get("attempt_id") == attempt.name, "contract intent attempt differs")
    require(Path(str(intent.get("attempt_root"))).resolve() == attempt, "contract intent root differs")
    require(intent.get("freeze_commit_b") == freeze, "contract intent freeze differs")
    require(intent.get("implementation_commit_a") == implementation, "contract intent implementation differs")
    require(intent.get("worker_sha256") == worker_sha, "contract intent worker differs")
    require(intent.get("manifests", {}).get("preregistration", {}).get("sha256") == prereg_sha, "contract intent preregistration differs")
    require(intent.get("manifests", {}).get("execution", {}).get("sha256") == execution_sha, "contract intent execution differs")
    expected_job = {
        "name": expected_job_name,
        "user": expected_user,
        "node": expected_node,
        "partition": "gre",
        "gpus": 0,
        "cpus": 2,
        "memory": "16G",
    }
    require(intent.get("job_contract") == expected_job, "contract intent job contract differs")
    require(submission.get("schema_id") == "qwen3vl_p2_contract_submission_receipt_v1", "contract submission schema differs")
    require(submission.get("submission_status") == "contract_validation_submitted", "contract submission status differs")
    require(submission.get("attempt_id") == attempt.name, "contract submission attempt differs")
    require(Path(str(submission.get("attempt_root"))).resolve() == attempt, "contract submission root differs")
    require(submission.get("freeze_commit_b") == freeze, "contract submission freeze differs")
    require(submission.get("implementation_commit_a") == implementation, "contract submission implementation differs")
    require(submission.get("job") == {**expected_job, "job_id": job_id}, "contract submitted job binding differs")
    require(submission.get("worker_sha256") == worker_sha, "contract submission worker differs")
    require(submission.get("manifests", {}).get("preregistration", {}).get("sha256") == prereg_sha, "contract submission preregistration differs")
    require(submission.get("manifests", {}).get("execution", {}).get("sha256") == execution_sha, "contract submission execution differs")
    verify_binding(submission.get("intent"), intent_path, "contract submission intent", sha256_file(intent_path))
    return intent_path, submission_path


def verify_contract_receipt(args: argparse.Namespace) -> None:
    freeze = require_sha(args.freeze_commit, 40, "freeze commit B")
    implementation = require_sha(args.implementation_commit, 40, "implementation commit A")
    worker = require_sha(args.worker_sha256, 64, "contract worker SHA256")
    prereg = require_sha(args.prereg_sha256, 64, "preregistration SHA256")
    execution = require_sha(args.execution_sha256, 64, "execution SHA256")
    require(re.fullmatch(r"[0-9]+", args.job_id) is not None, "contract Job ID is malformed")
    attempt, receipt_path = canonical_contract_paths(
        args.validation_base, freeze, args.job_id, args.receipt
    )
    intent_path, submission_path = verify_contract_submission(
        attempt,
        args.job_id,
        freeze,
        implementation,
        worker,
        prereg,
        execution,
        args.expected_user,
        args.expected_node,
        args.expected_job_name,
    )
    receipt = read_json(receipt_path, "contract validation receipt")
    require(receipt.get("schema_id") == "qwen3vl_p2_contract_validation_receipt_v1", "contract receipt schema differs")
    require(receipt.get("status") == "complete", "contract receipt status differs")
    require(receipt.get("job_id") == args.job_id, "contract receipt Job ID differs")
    require(receipt.get("job_name") == args.expected_job_name, "contract receipt job name differs")
    require(receipt.get("job_user") == args.expected_user, "contract receipt user differs")
    require(receipt.get("node") == args.expected_node, "contract receipt node differs")
    require(receipt.get("partition") == "gre", "contract receipt partition differs")
    require(receipt.get("freeze_commit_b") == freeze, "contract receipt freeze differs")
    require(receipt.get("implementation_commit_a") == implementation, "contract receipt implementation differs")
    require(receipt.get("freeze_b_parent_is_a") is True, "contract receipt A/B parent flag differs")
    require(receipt.get("worker_sha256") == worker, "contract receipt worker differs")
    require(receipt.get("resources") == {"gpus": 0, "cpus": 2, "memory": "16G"}, "contract resources differ")
    require(receipt.get("checks") == CONTRACT_CHECKS, "contract receipt checks are not the exact complete set")
    bindings = receipt.get("bindings")
    require(isinstance(bindings, Mapping), "contract receipt bindings are missing")
    require(
        set(bindings)
        == {
            "preregistration",
            "execution_manifest",
            "status",
            "submission_intent",
            "submission_receipt",
            "release_helper",
        },
        "contract receipt binding inventory differs",
    )
    prereg_path = Path(args.repo_root).resolve(strict=True) / FREEZE_FILES[0]
    execution_path = Path(args.repo_root).resolve(strict=True) / FREEZE_FILES[1]
    verify_binding(bindings.get("preregistration"), prereg_path, "preregistration", prereg)
    verify_binding(bindings.get("execution_manifest"), execution_path, "execution manifest", execution)
    verify_binding(bindings.get("submission_intent"), intent_path, "submission intent")
    verify_binding(bindings.get("submission_receipt"), submission_path, "submission receipt")
    verify_binding(
        bindings.get("release_helper"),
        Path(args.repo_root).resolve(strict=True) / "tools/verify_qwen3vl_p2_release.py",
        "release helper",
    )
    status_path = attempt / f"status-{args.job_id}.txt"
    verify_binding(bindings.get("status"), status_path, "contract status")
    status = parse_status(status_path)
    require(status.get("status") == "success" and status.get("exit_code") == "0", "contract status is not success/0")
    require(status.get("job_id") == args.job_id, "contract status Job ID differs")
    require(status.get("commit") == freeze, "contract status freeze differs")
    require(status.get("implementation_commit") == implementation, "contract status implementation differs")
    require(status.get("worker_sha256") == worker, "contract status worker differs")
    require(status.get("result") == "p2_contract_validation_complete", "contract status result differs")
    require(Path(str(status.get("receipt"))).resolve() == receipt_path, "contract status receipt path differs")


def verify_contract_submission_command(args: argparse.Namespace) -> None:
    attempt = args.attempt_root.resolve(strict=True)
    require(ATTEMPT.fullmatch(attempt.name) is not None, "contract attempt directory is malformed")
    verify_contract_submission(
        attempt,
        args.job_id,
        require_sha(args.freeze_commit, 40, "freeze commit B"),
        require_sha(args.implementation_commit, 40, "implementation commit A"),
        require_sha(args.worker_sha256, 64, "contract worker SHA256"),
        require_sha(args.prereg_sha256, 64, "preregistration SHA256"),
        require_sha(args.execution_sha256, 64, "execution SHA256"),
        args.expected_user,
        args.expected_node,
        args.expected_job_name,
    )


def verify_formal_submission(args: argparse.Namespace) -> None:
    attempt = args.attempt_root.resolve(strict=True)
    require(ATTEMPT.fullmatch(attempt.name) is not None, "formal attempt directory is malformed")
    intent_path = attempt / "submission_intent.json"
    receipt_path = attempt / "submission_receipt.json"
    intent = read_json(intent_path, "formal submission intent")
    receipt = read_json(receipt_path, "formal submission receipt")
    require(intent.get("schema_id") == "qwen3vl_p2_formal_submission_intent_v1", "formal intent schema differs")
    require(intent.get("status") == "prepared", "formal intent status differs")
    require(intent.get("attempt_id") == attempt.name, "formal intent attempt differs")
    require(Path(str(intent.get("attempt_root"))).resolve() == attempt, "formal intent root differs")
    require(intent.get("freeze_commit_b") == args.freeze_commit, "formal intent freeze differs")
    require(intent.get("implementation_commit_a") == args.implementation_commit, "formal intent implementation differs")
    require(receipt.get("schema_id") == "qwen3vl_p2_oracle_submission_receipt_v2", "formal receipt schema differs")
    require(receipt.get("submission_status") == "formal_submitted", "formal submission is not complete")
    require(receipt.get("attempt_id") == attempt.name, "formal receipt attempt differs")
    require(Path(str(receipt.get("attempt_root"))).resolve() == attempt, "formal receipt root differs")
    require(receipt.get("freeze_commit_b") == args.freeze_commit, "formal receipt freeze differs")
    require(receipt.get("implementation_commit_a") == args.implementation_commit, "formal receipt implementation differs")
    jobs = receipt.get("jobs")
    require(isinstance(jobs, Mapping), "formal receipt jobs are missing")
    require(
        set(jobs) == {"contract_validation", "generation", "scoring"},
        "formal receipt job inventory differs",
    )
    require(jobs.get("generation") == args.generation_job_id, "formal generation Job ID differs")
    own_key = "generation" if args.role == "generation" else "scoring"
    require(jobs.get(own_key) == args.job_id, f"formal {own_key} Job ID differs")
    require(re.fullmatch(r"[0-9]+", str(jobs.get("scoring"))) is not None, "formal scoring Job ID is malformed")
    require(receipt.get("dependency") == f"afterok:{args.generation_job_id}", "formal dependency is not exact afterok")
    require(receipt.get("job_user") == args.expected_user, "formal receipt user differs")
    require(receipt.get("node") == args.expected_node and receipt.get("partition") == "gre", "formal allocation binding differs")
    expected_job_contract = {
        "user": args.expected_user,
        "node": args.expected_node,
        "partition": "gre",
        "generation_name": "CA0803P2Generate",
        "scoring_name": "CA0803P2Score",
        "dependency": "afterok",
    }
    require(intent.get("job_contract") == expected_job_contract, "formal intent job contract differs")
    workers = receipt.get("workers")
    intent_workers = intent.get("workers")
    expected_worker_keys = {
        "generation",
        "scoring",
        "action_generator",
        "validator",
        "scorer",
        "release_helper",
    }
    require(
        isinstance(workers, Mapping)
        and isinstance(intent_workers, Mapping)
        and set(workers) == expected_worker_keys
        and workers == intent_workers
        and all(SHA64.fullmatch(str(value)) is not None for value in workers.values()),
        "formal worker bindings differ",
    )
    require(workers.get(own_key) == args.worker_sha256, f"formal {own_key} worker SHA differs")
    intent_manifests = intent.get("manifests")
    receipt_manifests = receipt.get("manifests")
    require(
        isinstance(intent_manifests, Mapping)
        and isinstance(receipt_manifests, Mapping)
        and intent_manifests == receipt_manifests
        and set(intent_manifests) == {"preregistration", "execution"},
        "formal manifest bindings differ",
    )
    for name, binding in intent_manifests.items():
        require(
            isinstance(binding, Mapping)
            and set(binding) == {"sha256", "canonical_sha256"}
            and SHA64.fullmatch(str(binding.get("sha256"))) is not None
            and SHA64.fullmatch(str(binding.get("canonical_sha256"))) is not None,
            f"formal {name} manifest binding is malformed",
        )
    verify_binding(receipt.get("intent"), intent_path, "formal submission intent")
    contract = receipt.get("contract_validation")
    require(isinstance(contract, Mapping), "formal contract validation binding is missing")
    require(contract.get("job_id") == args.contract_job_id, "formal contract Job ID differs")
    verify_binding(contract.get("receipt"), args.contract_receipt.resolve(strict=True), "formal contract receipt", args.contract_receipt_sha256)
    intent_contract = intent.get("contract_validation")
    require(
        isinstance(intent_contract, Mapping)
        and intent_contract.get("job_id") == args.contract_job_id,
        "formal intent contract validation differs",
    )
    verify_binding(
        intent_contract.get("receipt"),
        args.contract_receipt.resolve(strict=True),
        "formal intent contract receipt",
        args.contract_receipt_sha256,
    )


def inside(root: Path, raw: Any, field: str) -> Path:
    require(isinstance(raw, str) and raw and not Path(raw).is_absolute(), f"{field} must be relative")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReleaseError(f"{field} escapes generation root") from error
    return path


def verify_generation_output(args: argparse.Namespace) -> None:
    root = args.candidate_root.resolve(strict=True)
    marker_path = root / "candidate_generation_complete.json"
    index_path = root / "candidates.json"
    manifest_path = args.execution_manifest.resolve(strict=True)
    marker = read_json(marker_path, "generation marker")
    index = read_json(index_path, "candidate index")
    manifest = read_json(manifest_path, "execution manifest")
    require(marker.get("schema_id") == "qwen3vl_p2_candidate_generation_complete_v1", "marker schema differs")
    require(marker.get("execution_status") == "completed" and marker.get("gt_loaded") is False, "marker completion/GT differs")
    require(marker.get("counts") == COUNTS, "marker counts differ")
    require(marker.get("p2_runtime_commit") == args.runtime_commit, "marker runtime commit differs")
    require(marker.get("execution_manifest", {}).get("sha256") == sha256_file(manifest_path), "marker execution binding differs")
    require(marker.get("preregistration_canonical_sha256") == manifest.get("preregistration_canonical_sha256"), "marker preregistration binding differs")
    require(index.get("schema_id") == "qwen3vl_p2_oracle_candidates_v1", "candidate index schema differs")
    require(index.get("gt_loaded") is False and index.get("counts") == COUNTS, "candidate index GT/counts differ")
    candidates = index.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 36, "candidate index size differs")
    require(index.get("p2_runtime_commit") == args.runtime_commit, "candidate index runtime commit differs")
    require(index.get("execution_manifest", {}).get("sha256") == sha256_file(manifest_path), "candidate index execution raw binding differs")
    require(index.get("execution_manifest", {}).get("canonical_sha256") == canonical_sha256(manifest), "candidate index execution canonical binding differs")
    require(index.get("preregistration_canonical_sha256") == manifest.get("preregistration_canonical_sha256"), "candidate index preregistration binding differs")
    require(marker.get("candidate_index", {}).get("sha256") == sha256_file(index_path), "marker index binding differs")
    expected_actions = [
        action
        for sample in manifest.get("ordered_samples", [])
        for action in manifest.get("samples", {}).get(sample, {}).get("actions", [])
    ]
    require(len(expected_actions) == 36, "execution manifest action count differs")
    for identity, expected in zip(candidates, expected_actions):
        require(isinstance(identity, Mapping), "candidate identity is malformed")
        record_path = inside(root, identity.get("record_path"), "candidate record path")
        require(sha256_file(record_path) == identity.get("record_sha256"), "candidate record SHA differs")
        record = read_json(record_path, "candidate record")
        require(all(record.get(key) == value for key, value in expected.items()), "candidate action identity differs")
        require(identity.get("candidate_id") == record.get("candidate_id"), "candidate ID differs")
        require(identity.get("sample") == record.get("sample") and identity.get("tier") == record.get("tier"), "candidate sample/tier differs")
        evidence = record.get("tool_evidence")
        expected_tool = "sam3" if record.get("action") == "box" else "simpleclick"
        require(isinstance(evidence, Mapping) and evidence.get("tool") == expected_tool, "candidate tool evidence differs")
        require(isinstance(evidence.get("tool_input"), Mapping), "candidate tool input is missing")


def verify_generation_receipt(args: argparse.Namespace) -> None:
    generation_root = args.candidate_root.resolve(strict=True)
    require(generation_root.name == "generation", "candidate root is not the formal attempt generation directory")
    attempt = generation_root.parent
    receipt_path = exact_path(args.receipt, attempt / "generation_receipt.json", "generation receipt")
    formal_receipt = exact_path(
        args.formal_submission_receipt,
        attempt / "submission_receipt.json",
        "formal submission receipt",
    )
    marker_path = generation_root / "candidate_generation_complete.json"
    index_path = generation_root / "candidates.json"
    execution_path = args.execution_manifest.resolve(strict=True)
    receipt = read_json(receipt_path, "generation receipt")
    require(receipt.get("schema_id") == "qwen3vl_p2_oracle_generation_receipt_v2", "generation receipt schema differs")
    require(receipt.get("status") == "complete", "generation receipt is not complete")
    require(receipt.get("job_id") == args.generation_job_id, "generation receipt Job ID differs")
    require(receipt.get("freeze_commit_b") == args.freeze_commit, "generation receipt freeze differs")
    require(receipt.get("implementation_commit_a") == args.implementation_commit, "generation receipt implementation differs")
    require(receipt.get("worker_sha256") == args.worker_sha256, "generation receipt worker differs")
    require(receipt.get("gt_loaded") is False and receipt.get("counts") == COUNTS, "generation receipt GT/counts differ")
    require(receipt.get("candidate_flow_counts") == {"requested": 36, "generated": 36, "tool_success": 36}, "generation flow counts differ")
    bindings = receipt.get("bindings")
    require(isinstance(bindings, Mapping), "generation receipt bindings are missing")
    verify_binding(bindings.get("execution_manifest"), execution_path, "generation execution manifest", args.execution_sha256)
    verify_binding(bindings.get("complete_marker"), marker_path, "generation marker")
    verify_binding(bindings.get("candidate_index"), index_path, "generation candidate index")
    verify_binding(bindings.get("formal_submission_receipt"), formal_receipt, "generation formal submission receipt")


def build_decision(args: argparse.Namespace) -> None:
    report = read_json(args.report.resolve(strict=True), "score report")
    require(report.get("schema_id") == "qwen3vl_p2_oracle_score_report_v1", "score report schema differs")
    targets = report.get("target_primary_actionability")
    expected = ["test_1_1", "test_50_8"]
    require(isinstance(targets, Mapping) and list(targets) == expected, "score report target order differs")
    candidates = report.get("candidates")
    require(report.get("candidate_count") == 36 and isinstance(candidates, list) and len(candidates) == 36, "score report candidate count differs")
    require(report.get("action_spec_count") == 36, "score report action-spec count differs")
    require(len({item.get("candidate_id") for item in candidates if isinstance(item, Mapping)}) == 36, "score report candidate IDs differ")
    require(report.get("by_tier", {}).get("primary", {}).get("candidate_count") == 27, "primary score count differs")
    require(report.get("by_tier", {}).get("diagnostic_executor_only", {}).get("candidate_count") == 9, "diagnostic score count differs")
    require(report.get("qwen_evaluation", {}).get("qwen_invoked") is False, "Qwen invocation flag differs")
    proceed = all(targets[name].get("decision") == "primary_actionable_and_gate_pass" for name in expected)
    hard_gated = [name for name in expected if targets[name].get("decision") == "primary_actionable_but_all_beneficial_hard_gated"]
    missing = [name for name in expected if targets[name].get("decision") == "no_primary_beneficial_candidate_in_frozen_enumeration"]
    payload = {
        "schema_id": "qwen3vl_p2_actionability_decision_v1",
        "status": "complete",
        "decision_basis": "primary_candidates_only",
        "diagnostic_candidates_excluded": True,
        "target_samples": expected,
        "target_results": targets,
        "proceed_to_c_iso_and_t_traj_data_construction": proceed,
        "next_action": (
            "construct_matched_c_iso_and_t_traj"
            if proceed
            else "report_beneficial_hard_gate_conflict_without_gate_change"
            if hard_gated and not missing
            else "stop_training_and_review_frozen_action_space"
        ),
        "beneficial_but_hard_gated_samples": hard_gated,
        "no_primary_beneficial_candidate_samples": missing,
        "gate_modification_authorized": False,
        "scope_limit": "decision applies only to the frozen finite single-step CORE3 inventory",
    }
    target = args.output.resolve()
    require(not target.exists(), "refusing to overwrite actionability decision")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("verify-freeze")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--freeze-commit", required=True)
    freeze.add_argument("--implementation-commit", required=True)

    contract = commands.add_parser("verify-contract-receipt")
    contract.add_argument("--receipt", type=Path, required=True)
    contract.add_argument("--validation-base", type=Path, required=True)
    contract.add_argument("--repo-root", type=Path, required=True)
    contract.add_argument("--job-id", required=True)
    contract.add_argument("--freeze-commit", required=True)
    contract.add_argument("--implementation-commit", required=True)
    contract.add_argument("--worker-sha256", required=True)
    contract.add_argument("--prereg-sha256", required=True)
    contract.add_argument("--execution-sha256", required=True)
    contract.add_argument("--expected-user", default="guisongxia")
    contract.add_argument("--expected-node", default="gpu48")
    contract.add_argument("--expected-job-name", default="CA0803P2Contract")

    contract_submission = commands.add_parser("verify-contract-submission")
    contract_submission.add_argument("--attempt-root", type=Path, required=True)
    contract_submission.add_argument("--job-id", required=True)
    contract_submission.add_argument("--freeze-commit", required=True)
    contract_submission.add_argument("--implementation-commit", required=True)
    contract_submission.add_argument("--worker-sha256", required=True)
    contract_submission.add_argument("--prereg-sha256", required=True)
    contract_submission.add_argument("--execution-sha256", required=True)
    contract_submission.add_argument("--expected-user", default="guisongxia")
    contract_submission.add_argument("--expected-node", default="gpu48")
    contract_submission.add_argument("--expected-job-name", default="CA0803P2Contract")

    formal = commands.add_parser("verify-formal-submission")
    formal.add_argument("--attempt-root", type=Path, required=True)
    formal.add_argument("--role", choices=("generation", "scoring"), required=True)
    formal.add_argument("--job-id", required=True)
    formal.add_argument("--generation-job-id", required=True)
    formal.add_argument("--contract-job-id", required=True)
    formal.add_argument("--freeze-commit", required=True)
    formal.add_argument("--implementation-commit", required=True)
    formal.add_argument("--worker-sha256", required=True)
    formal.add_argument("--contract-receipt", type=Path, required=True)
    formal.add_argument("--contract-receipt-sha256", required=True)
    formal.add_argument("--expected-user", default="guisongxia")
    formal.add_argument("--expected-node", default="gpu48")

    generated = commands.add_parser("verify-generation-output")
    generated.add_argument("--candidate-root", type=Path, required=True)
    generated.add_argument("--execution-manifest", type=Path, required=True)
    generated.add_argument("--runtime-commit", required=True)

    generation_receipt = commands.add_parser("verify-generation-receipt")
    generation_receipt.add_argument("--receipt", type=Path, required=True)
    generation_receipt.add_argument("--candidate-root", type=Path, required=True)
    generation_receipt.add_argument("--execution-manifest", type=Path, required=True)
    generation_receipt.add_argument("--formal-submission-receipt", type=Path, required=True)
    generation_receipt.add_argument("--generation-job-id", required=True)
    generation_receipt.add_argument("--freeze-commit", required=True)
    generation_receipt.add_argument("--implementation-commit", required=True)
    generation_receipt.add_argument("--worker-sha256", required=True)
    generation_receipt.add_argument("--execution-sha256", required=True)

    decision = commands.add_parser("build-decision")
    decision.add_argument("--report", type=Path, required=True)
    decision.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "verify-freeze":
            verify_freeze(args.repo_root, args.freeze_commit, args.implementation_commit)
        elif args.command == "verify-contract-receipt":
            verify_contract_receipt(args)
        elif args.command == "verify-contract-submission":
            verify_contract_submission_command(args)
        elif args.command == "verify-formal-submission":
            verify_formal_submission(args)
        elif args.command == "verify-generation-output":
            verify_generation_output(args)
        elif args.command == "verify-generation-receipt":
            verify_generation_receipt(args)
        elif args.command == "build-decision":
            build_decision(args)
        else:
            raise ReleaseError(f"unsupported command: {args.command}")
    except (OSError, ReleaseError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"qwen3vl_p2_release: {error}", file=sys.stderr)
        return 3
    print(f"status=valid;command={args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
