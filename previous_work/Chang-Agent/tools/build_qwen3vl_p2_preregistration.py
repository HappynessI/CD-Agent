#!/usr/bin/env python3
"""Build the two P2 freeze-B manifest candidates from an exact clean commit A.

The first candidate is the offline preregistration/scoring contract and is the
only repository artifact that contains label identities.  The second candidate
is a strictly GT-free execution manifest.  The latter is the only manifest that
the candidate generator may receive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor
from change_agent.state import ChangeState
from change_agent.trajectory import mask_sha256
from change_agent.verifier_regions import attach_verifier_regions
from tools.run_qwen3vl_p2_oracle import enumerate_oracle_actions


PREREG_SCHEMA_ID = "qwen3vl_p2_actionability_preregistration_v1"
EXECUTION_SCHEMA_ID = "qwen3vl_p2_oracle_execution_manifest_v1"
SOURCE_FREEZE_COMMIT = "5210f89f5ca13ee1e636f5aa6a44631206a80123"
P1_ACCEPTANCE_COMMIT = "17e9bb9914266691aee0c3d2b5e78b6bf64d7a00"
ORDERED_SAMPLES = ("test_20_15", "test_1_1", "test_50_8")
TARGET_SAMPLES = ("test_1_1", "test_50_8")
EXPECTED_COUNTS = {"total": 36, "primary": 27, "diagnostic_executor_only": 9}
EXPECTED_PRIMARY = {"test_20_15": 8, "test_1_1": 1, "test_50_8": 18}
EXPECTED_DIAGNOSTIC = {"test_20_15": 8, "test_1_1": 1, "test_50_8": 0}
H0_SOURCE_OUTPUT = Path(
    "/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/"
    "CA_0802(5)-qwen3vl-h0-base-core3-retry3"
)
H1_SOURCE_OUTPUT = Path(
    "/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/"
    "CA_0802(6)-qwen3vl-h1-base-core3-retry3"
)
P1_ARCHIVE_OUTPUT_BASE = H0_SOURCE_OUTPUT.parent / "CA_0802_archive"
H0_ARCHIVE_OUTPUT = P1_ARCHIVE_OUTPUT_BASE / H0_SOURCE_OUTPUT.name
H1_ARCHIVE_OUTPUT = P1_ARCHIVE_OUTPUT_BASE / H1_SOURCE_OUTPUT.name
DATASET = Path("/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256")
WORKSPACE = Path("/guisongxia01/pangchao/wangyihan")
H0_SOURCE_FILE_HASHES = {
    "rollout_complete": "<hash omitted>",
    "evaluation_receipt": "<hash omitted>",
}
H1_SOURCE_FILE_HASHES = {
    "rollout_complete": "<hash omitted>",
    "evaluation_receipt": "<hash omitted>",
    "pair_comparison_receipt": "<hash omitted>",
}
IMAGE_HASHES = {
    "test_20_15": {
        "A": "<hash omitted>",
        "B": "<hash omitted>",
        "label": "<hash omitted>",
    },
    "test_1_1": {
        "A": "<hash omitted>",
        "B": "<hash omitted>",
        "label": "<hash omitted>",
    },
    "test_50_8": {
        "A": "<hash omitted>",
        "B": "<hash omitted>",
        "label": "<hash omitted>",
    },
}
MASK_FILE_HASHES = {
    "test_20_15": (
        "<hash omitted>",
        "<hash omitted>",
    ),
    "test_1_1": (
        "<hash omitted>",
        "<hash omitted>",
    ),
    "test_50_8": (
        "<hash omitted>",
        "<hash omitted>",
    ),
}
TOOL_IDENTITIES = {
    "segagent_python": (
        WORKSPACE / "conda_envs/segagent-env/bin/python3.10",
        "<hash omitted>",
    ),
    "omniovcd_python": (
        WORKSPACE / "conda_envs/omniovcd-env/bin/python3.12",
        "<hash omitted>",
    ),
    "simpleclick_checkpoint": (
        WORKSPACE / "models/SimpleClick/cocolvis_vit_large.pth",
        "<hash omitted>",
    ),
    "sam3_checkpoint": (
        WORKSPACE / "models/sam3/sam3.pt",
        "<hash omitted>",
    ),
    "sam3_bpe": (
        WORKSPACE / "Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "<hash omitted>",
    ),
}
RUNTIME_SOURCES = (
    "change_agent/action_parser.py",
    "change_agent/adapters/omniovcd_adapter.py",
    "change_agent/adapters/sam3_adapter.py",
    "change_agent/adapters/segagent_adapter.py",
    "change_agent/adapters/staged_verifier.py",
    "change_agent/adapters/subprocess_adapters.py",
    "change_agent/environment.py",
    "change_agent/executor.py",
    "change_agent/state.py",
    "change_agent/trajectory.py",
    "change_agent/verifier.py",
    "change_agent/verifier_protocol.py",
    "change_agent/verifier_regions.py",
    "change_agent/coordinates.py",
    "tools/seeded_segmentation_worker.py",
    "tools/segmentation_worker.py",
)


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


def identity(path: Path, expected: str | None = None) -> dict[str, str]:
    path = path.resolve(strict=True)
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(f"frozen file changed: {path}: {observed}")
    return {"path": str(path), "sha256": observed}


def p1_materialized_file(logical_path: Path) -> Path:
    """Resolve one frozen P1 file after the exact output roots were archived.

    The frozen manifest identity remains the original absolute path.  This
    resolver only changes where bytes are read: direct evidence wins, and the
    sole fallback is the same relative file below the two explicit archive
    roots.  No basename search or cross-run substitution is allowed.
    """

    logical_path = Path(os.path.abspath(logical_path))
    if logical_path.is_file():
        return logical_path.resolve(strict=True)
    relocations = (
        (H0_SOURCE_OUTPUT, H0_ARCHIVE_OUTPUT),
        (H1_SOURCE_OUTPUT, H1_ARCHIVE_OUTPUT),
    )
    for frozen_root, archive_root in relocations:
        try:
            relative = logical_path.relative_to(frozen_root)
        except ValueError:
            continue
        resolved_archive_root = archive_root.resolve(strict=True)
        materialized = (resolved_archive_root / relative).resolve(strict=True)
        if not materialized.is_relative_to(resolved_archive_root):
            raise ValueError(f"archived P1 file escapes its exact root: {logical_path}")
        if not materialized.is_file():
            raise ValueError(f"archived P1 evidence is not a file: {materialized}")
        return materialized
    raise ValueError(f"frozen P1 evidence is unavailable: {logical_path}")


def frozen_p1_identity(logical_path: Path, expected: str) -> dict[str, str]:
    """Verify relocated bytes while preserving the already frozen identity."""

    logical_path = Path(os.path.abspath(logical_path))
    materialized = p1_materialized_file(logical_path)
    observed = sha256_file(materialized)
    if observed != expected:
        raise ValueError(
            f"frozen P1 file changed: {logical_path} -> {materialized}: {observed}"
        )
    return {"path": str(logical_path), "sha256": observed}


def repo_identity(repo_root: Path, relative: str) -> dict[str, str]:
    path = (repo_root / relative).resolve(strict=True)
    return {"path": relative, "sha256": sha256_file(path)}


def git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_exact_clean_commit_a(repo_root: Path, implementation_commit: str) -> None:
    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ValueError("implementation commit A must be one exact lowercase SHA")
    if git(repo_root, "rev-parse", "HEAD") != implementation_commit:
        raise ValueError("builder checkout is not exact implementation commit A")
    if git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("builder checkout must be clean before manifest generation")


def _sample(repo_root: Path, name: str) -> dict[str, Any]:
    del repo_root
    source = H0_SOURCE_OUTPUT / "run/tool_runs" / name / "sam3_initialization/initialize_000"
    t1_mask_logical_path = source / "t1_mask.npy"
    t2_mask_logical_path = source / "t2_mask.npy"
    t1_mask_path = p1_materialized_file(t1_mask_logical_path)
    t2_mask_path = p1_materialized_file(t2_mask_logical_path)
    expected_t1, expected_t2 = MASK_FILE_HASHES[name]
    t1_file = frozen_p1_identity(t1_mask_logical_path, expected_t1)
    t2_file = frozen_p1_identity(t2_mask_logical_path, expected_t2)
    t1_mask = np.asarray(np.load(t1_mask_path), dtype=bool)
    t2_mask = np.asarray(np.load(t2_mask_path), dtype=bool)
    processor = MaskPairProcessor(
        overlap_threshold=0.25,
        matching_mode="overlap_presence",
        t12_min_instance_area=0,
        cd_min_instance_area=0,
    )
    update = processor.rebuild(t1_mask, t2_mask, {"initializer": "p1_cached_sam3"})
    a_path = DATASET / "A" / f"{name}.png"
    b_path = DATASET / "B" / f"{name}.png"
    a_file = identity(a_path, IMAGE_HASHES[name]["A"])
    b_file = identity(b_path, IMAGE_HASHES[name]["B"])
    state = ChangeState(
        t1_image=np.asarray(Image.open(a_path).convert("RGB")),
        t2_image=np.asarray(Image.open(b_path).convert("RGB")),
        query="building",
        t1_mask=t1_mask,
        t2_mask=t2_mask,
        change_mask=update.change_mask,
        t1_instances=update.t1_instances,
        t2_instances=update.t2_instances,
        matching=update.matching,
        evidence=update.evidence,
        step_index=0,
    )
    proposals = attach_verifier_regions(
        state,
        None,
        max_regions=1,
        max_delta_regions=1,
        min_component_area=4,
        padding_ratio=0.25,
        empty_mask_scan_grid_size=3,
    )
    actions = enumerate_oracle_actions(name, proposals)
    t1_file["mask_sha256"] = mask_sha256(t1_mask)
    t2_file["mask_sha256"] = mask_sha256(t2_mask)
    return {
        "t1_image": a_file,
        "t2_image": b_file,
        "t1_mask": t1_file,
        "t2_mask": t2_file,
        "initial_state_mask_sha256": {
            "t1": mask_sha256(t1_mask),
            "t2": mask_sha256(t2_mask),
            "change": mask_sha256(update.change_mask),
        },
        # Copy the already frozen P1 label identity without opening, statting, or
        # hashing the label.  P2's first label-file access belongs to the
        # post-generation validator after every candidate artifact is verified.
        "offline_label": {
            "path": str(DATASET / "label_cvt" / f"{name}.png"),
            "sha256": IMAGE_HASHES[name]["label"],
        },
        "proposals_canonical_sha256": canonical_sha256(proposals),
        "actions_canonical_sha256": canonical_sha256(actions),
        "proposals": proposals,
        "actions": actions,
    }


def _contracts() -> dict[str, Any]:
    return {
        "schema_contracts": {
            "execution_manifest": EXECUTION_SCHEMA_ID,
            "candidate_index": "qwen3vl_p2_oracle_candidates_v1",
            "generation_complete_marker": "qwen3vl_p2_candidate_generation_complete_v1",
            "scoring_manifest": "qwen3vl_p2_oracle_scoring_manifest_v1",
            "score_report": "qwen3vl_p2_oracle_score_report_v1",
            "generation_receipt": "qwen3vl_p2_oracle_generation_receipt_v1",
            "decision": "qwen3vl_p2_actionability_decision_v1",
            "scoring_receipt": "qwen3vl_p2_oracle_scoring_receipt_v1",
            "submission_receipt": "qwen3vl_p2_oracle_submission_receipt_v1",
        },
        "metric_contract": {
            "comparison_epsilon": 1e-12,
            "comparison_occurs_before_rounding": True,
            "beneficial_if": "delta_iou_above_epsilon_and_delta_f1_above_epsilon",
            "harmful_if": (
                "either_metric_decreases_below_negative_epsilon_or_both_metrics_tie_"
                "with_tp_decrease_or_fp_increase_or_fn_increase"
            ),
            "otherwise": "indeterminate",
            "empty_gt_and_empty_prediction": {
                "iou": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            "empty_prediction_nonempty_gt": {
                "precision": 0.0,
                "recall": 0.0,
                "iou": 0.0,
                "f1": 0.0,
            },
            "empty_gt_nonempty_prediction": {
                "precision": 0.0,
                "recall": 1.0,
                "iou": 0.0,
                "f1": 0.0,
            },
            "rate_contract": (
                "Every rate records numerator, denominator, and Wilson 95% interval; "
                "a zero denominator is represented by null rate and bounds."
            ),
        },
        "decision_contract": {
            "allow_data_construction_if": (
                "test_1_1 and test_50_8 each have at least one primary beneficial "
                "candidate whose unchanged pre-Qwen hard gate passed"
            ),
            "proceed_target": (
                "construct matched C-ISO and T-TRAJ data only after this condition passes"
            ),
            "stop_if": [
                "either target has no primary beneficial candidate in the complete frozen inventory",
                "either target has beneficial primary candidates but none pass the unchanged pre-Qwen hard gate",
                "candidate generation or offline scoring evidence is incomplete or invalid",
            ],
            "beneficial_but_hard_gated_policy": (
                "Report the action and gate reason for review; do not modify, relax, or "
                "retune the gate in this P2 run."
            ),
            "diagnostic_candidates_affect_decision": False,
            "test_20_15_affects_target_gate": False,
            "negative_result_interpretation": (
                "No reachability was observed only within this preregistered finite "
                "single-step action inventory; it is not proof that no useful action "
                "exists in a broader space."
            ),
        },
        "gt_boundary": {
            "generation_manifest_contains_labels": False,
            "generation_process_receives_input_root": False,
            "generation_process_receives_scoring_manifest": False,
            "generation_process_receives_only_explicit_a_b_and_frozen_mask_files": True,
            "candidate_generation_complete_marker_required_before_any_label_file_is_opened": True,
            "all_candidate_records_and_masks_hash_verified_before_any_label_file_is_opened": True,
            "labels_are_offline_scorer_only": True,
        },
    }


def build(repo_root: Path, implementation_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve(strict=True)
    assert_exact_clean_commit_a(repo_root, implementation_commit)
    h0_source_files = {
        "rollout_complete": frozen_p1_identity(
            H0_SOURCE_OUTPUT / "run/rollout_complete.json",
            H0_SOURCE_FILE_HASHES["rollout_complete"],
        ),
        "evaluation_receipt": frozen_p1_identity(
            H0_SOURCE_OUTPUT / "evaluation_receipt.json",
            H0_SOURCE_FILE_HASHES["evaluation_receipt"],
        ),
    }
    h1_source_files = {
        "rollout_complete": frozen_p1_identity(
            H1_SOURCE_OUTPUT / "run/rollout_complete.json",
            H1_SOURCE_FILE_HASHES["rollout_complete"],
        ),
        "evaluation_receipt": frozen_p1_identity(
            H1_SOURCE_OUTPUT / "evaluation_receipt.json",
            H1_SOURCE_FILE_HASHES["evaluation_receipt"],
        ),
        "pair_comparison_receipt": frozen_p1_identity(
            H1_SOURCE_OUTPUT / "pair_comparison_receipt.json",
            H1_SOURCE_FILE_HASHES["pair_comparison_receipt"],
        ),
    }
    samples = {name: _sample(repo_root, name) for name in ORDERED_SAMPLES}
    all_proposals = [item for name in ORDERED_SAMPLES for item in samples[name]["proposals"]]
    all_actions = [item for name in ORDERED_SAMPLES for item in samples[name]["actions"]]
    counts = {
        "total": len(all_actions),
        "primary": sum(item["tier"] == "primary" for item in all_actions),
        "diagnostic_executor_only": sum(
            item["tier"] == "diagnostic_executor_only" for item in all_actions
        ),
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"mechanical action inventory count changed: {counts}")
    primary_by_sample = {
        name: sum(item["tier"] == "primary" for item in samples[name]["actions"])
        for name in ORDERED_SAMPLES
    }
    diagnostic_by_sample = {
        name: sum(
            item["tier"] == "diagnostic_executor_only"
            for item in samples[name]["actions"]
        )
        for name in ORDERED_SAMPLES
    }
    if primary_by_sample != EXPECTED_PRIMARY or diagnostic_by_sample != EXPECTED_DIAGNOSTIC:
        raise ValueError("mechanical per-sample action inventory changed")

    action_generator = repo_identity(repo_root, "tools/run_qwen3vl_p2_oracle.py")
    scorer = repo_identity(repo_root, "tools/score_qwen3vl_p2_oracle.py")
    runtime_sources = [repo_identity(repo_root, path) for path in RUNTIME_SOURCES]
    tools = {
        "workspace_root": {"path": str(WORKSPACE)},
        **{
            name: identity(path, expected)
            for name, (path, expected) in TOOL_IDENTITIES.items()
        },
    }
    protocol = {
        "seed": 42,
        "query": "building",
        "single_step_independent_reset": True,
        "max_actions_per_candidate": 1,
        "candidate_to_candidate_state_carryover": False,
        "qwen_invoked": False,
        "tool_device": "cuda",
        "sam3_resolution": 1008,
        "tool_timeout_seconds": 900,
        "authorized_negative_point_input_scope": "full_session",
        "matching": {
            "mode": "overlap_presence",
            "overlap_threshold": 0.25,
            "t12_min_instance_area": 0,
            "cd_min_instance_area": 0,
        },
        "proposal_config": {
            "max_regions": 1,
            "max_delta_regions": 1,
            "min_component_area": 4,
            "padding_ratio": 0.25,
            "empty_mask_scan_grid_size": 3,
        },
        "gate_config": {
            "max_selection_area_delta": 0.25,
            "max_locality_outside_ratio": 0.1,
            "max_target_mask_change_ratio": 0.25,
            "max_component_count_delta": 4,
            "max_point_proposal_change_ratio": 0.25,
            "max_point_proposal_change_pixels": 256,
            "max_point_image_change_ratio": 0.004,
            "point_proposal_change_gate_enabled": True,
        },
    }
    preregistration: dict[str, Any] = {
        "schema_id": PREREG_SCHEMA_ID,
        "design_status": "frozen",
        "execution_status": "not_run",
        "created_utc": "2026-08-03",
        "purpose": (
            "Test whether the frozen production action space contains at least one "
            "single-step primary candidate that improves each P1 false-finish target "
            "and passes the unchanged pre-Qwen hard gates before any C-ISO or T-TRAJ "
            "data are constructed."
        ),
        "interpretation_scope": (
            "Conclusions apply only to the frozen CORE3 source states, enumerator, "
            "tools, single-step budget, and primary action inventory. They are not a "
            "LEVIR-CD generalization result."
        ),
        "anchors": {
            "p1_runtime_source_freeze_commit_b4": SOURCE_FREEZE_COMMIT,
            "p1_acceptance_commit": P1_ACCEPTANCE_COMMIT,
            "implementation_commit": implementation_commit,
            "freeze_commit_policy": (
                "The manifest records implementation commit A only. Freeze commit B "
                "must be its direct child and is recorded only by external run receipts."
            ),
        },
        "source_h0_retry3": {
            "job_id": "53158",
            "state": "COMPLETED",
            "arm": "h0",
            "experiment_arm_id": "v12_1-h0-legacy-single-round-base-0802",
            "harness_protocol_id": "qwen3vl_harness_h0_legacy_single_round_v1",
            "runtime_commit": SOURCE_FREEZE_COMMIT,
            "output_root": str(H0_SOURCE_OUTPUT),
            **h0_source_files,
            "source_state_policy": (
                "Every candidate resets independently to the exact cached H0 retry3 "
                "selected-step-zero T1/T2 semantic masks. No candidate starts from "
                "another candidate."
            ),
        },
        "source_h1_retry3": {
            "job_id": "53159",
            "state": "COMPLETED",
            "arm": "h1",
            "experiment_arm_id": "v12_1-h1-gtfree-terminal-review-base-0802",
            "harness_protocol_id": "qwen3vl_harness_h1_gtfree_terminal_review_v1",
            "runtime_commit": SOURCE_FREEZE_COMMIT,
            "output_root": str(H1_SOURCE_OUTPUT),
            **h1_source_files,
            "provenance_policy": (
                "The frozen H0/H1 pair proves that H1 selected the same step-zero state "
                "and metrics as the H0 masks used for P2, while binding the two P1 "
                "false-finish targets to the completed H1 evaluation."
            ),
            "paired_h0_output_root": str(H0_SOURCE_OUTPUT),
            "pair_equivalence_requirement": (
                "The fixed CORE3 pair must bind independent H0 Job 53158 and H1 Job "
                "53159 receipts, and the bound summaries/source metrics must show the "
                "same verifier-selected step and selected metrics for every sample."
            ),
        },
        "ordered_samples": list(ORDERED_SAMPLES),
        "target_samples": list(TARGET_SAMPLES),
        "execution_protocol": protocol,
        "action_enumeration_contract": {
            "authority": (
                "MaskPairProcessor.rebuild followed by attach_verifier_regions on the "
                "frozen masks, then tools.run_qwen3vl_p2_oracle.enumerate_oracle_actions"
            ),
            "manual_coordinate_selection_forbidden": True,
            "present_proposal_primary": (
                "one negative_point on the exactly one white target view at component_seed_pixels"
            ),
            "present_proposal_diagnostic": (
                "one positive_point on the exactly one black target view at the same component_seed_pixels"
            ),
            "scan_proposal_primary": "one box action for each of t1 and t2 at box_pixels",
            "diagnostic_tier_is_executor_only": True,
            "primary_tier_is_only_decision_tier": True,
            "action_spec_deduplication": (
                "exact candidate_id and exact sample/tier/region/action/target/geometry identity"
            ),
            "candidate_state_deduplication": (
                "not used to reduce the preregistered action inventory; identical "
                "resulting masks remain separately auditable action outcomes"
            ),
            "expected_candidate_counts": counts,
            "expected_primary_by_sample": primary_by_sample,
            "expected_diagnostic_by_sample": diagnostic_by_sample,
            "all_proposals_canonical_sha256": canonical_sha256(all_proposals),
            "all_actions_canonical_sha256": canonical_sha256(all_actions),
            "combined_inventory_canonical_sha256": canonical_sha256(
                {"proposals": all_proposals, "actions": all_actions}
            ),
        },
        **_contracts(),
        "runtime_files": {
            "action_generator": action_generator,
            "scorer": scorer,
            "generation_runtime_sources": runtime_sources,
        },
        "tools": tools,
        "samples": samples,
        "canonical_json_contract": {
            "hash_scope": "The entire parsed preregistration object; no field is excluded.",
            "serialization": (
                "UTF-8 json.dumps(payload, ensure_ascii=False, sort_keys=True, "
                "separators=(',', ':'), allow_nan=False) with no trailing newline."
            ),
            "digest": "sha256",
            "self_digest_field_forbidden": True,
            "formal_expected_digest_is_supplied_externally_to_submitter_and_workers": True,
        },
        "formal_execution": {
            "authorized_after_freeze_b_validation": True,
            "freeze_b_contract": "B^ must equal anchors.implementation_commit",
            "generation_resources": {
                "node": "gpu48",
                "accelerator": "A800",
                "gpus": 1,
                "cpus": 4,
                "memory": "48G",
                "time": "02:00:00",
            },
            "scoring_resources": {
                "node": "gpu48",
                "gpus": 0,
                "cpus": 2,
                "memory": "16G",
                "time": "00:45:00",
            },
            "dependency": "scoring afterok generation",
            "output": str(
                WORKSPACE
                / "Chang-Agent/outputs/"
                "CA_0803(1)-qwen3vl-p2-oracle-actionability"
            ),
        },
    }
    execution_samples = {
        name: {
            key: value
            for key, value in sample.items()
            if key != "offline_label"
        }
        for name, sample in samples.items()
    }
    execution: dict[str, Any] = {
        "schema_id": EXECUTION_SCHEMA_ID,
        "status": "frozen",
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "implementation_commit": implementation_commit,
        "preregistration_canonical_sha256": canonical_sha256(preregistration),
        "action_generator_sha256": action_generator["sha256"],
        "runtime_source_files": runtime_sources,
        "ordered_samples": list(ORDERED_SAMPLES),
        "expected_candidate_counts": counts,
        "seed": protocol["seed"],
        "query": protocol["query"],
        "single_step_independent_reset": True,
        "authorized_negative_point_input_scope": protocol[
            "authorized_negative_point_input_scope"
        ],
        "matching": protocol["matching"],
        "proposal_config": protocol["proposal_config"],
        "gate_config": protocol["gate_config"],
        "tool_device": protocol["tool_device"],
        "sam3_resolution": protocol["sam3_resolution"],
        "tool_timeout_seconds": protocol["tool_timeout_seconds"],
        "tools": tools,
        "samples": execution_samples,
    }
    return preregistration, execution


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite manifest candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        # Preserve the frozen sample/action insertion order in the raw artifact.
        # Canonical identity remains sort-key based and is computed separately.
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--preregistration-output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        preregistration, execution = build(args.repo_root, args.implementation_commit)
        write_new_json(args.preregistration_output, preregistration)
        write_new_json(args.execution_output, execution)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"qwen3vl_p2_preregistration_builder: {error}", file=__import__("sys").stderr)
        return 3
    print(
        "status=complete;"
        f"implementation_commit={args.implementation_commit};"
        f"preregistration_canonical_sha256={canonical_sha256(preregistration)};"
        f"execution_canonical_sha256={canonical_sha256(execution)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
