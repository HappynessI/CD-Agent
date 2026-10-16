from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "tools/validate_qwen3vl_oracle_trajectory.py"
    spec = importlib.util.spec_from_file_location(
        "validate_qwen3vl_oracle_trajectory_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def load_runner():
    path = ROOT / "tools/run_qwen3vl_oracle_trajectory.py"
    spec = importlib.util.spec_from_file_location(
        "run_qwen3vl_oracle_trajectory_validator_contract_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mask(root: Path, relative: str, value: np.ndarray) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value, dtype=np.uint8))
    mask = np.asarray(value, dtype=bool)
    return {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "mask_sha256": VALIDATOR.mask_sha256(mask),
        "shape": list(mask.shape),
    }


def write_state(
    root: Path,
    prefix: str,
    change: np.ndarray,
    *,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
) -> dict[str, object]:
    t1 = np.zeros_like(change, dtype=bool) if t1 is None else np.asarray(t1, dtype=bool)
    t2 = np.asarray(change, dtype=bool) if t2 is None else np.asarray(t2, dtype=bool)
    specs = {
        "t1_mask": write_mask(root, f"{prefix}/t1.npy", t1),
        "t2_mask": write_mask(root, f"{prefix}/t2.npy", t2),
        "change_mask": write_mask(root, f"{prefix}/change.npy", change),
    }
    identity = VALIDATOR.state_sha256(
        {
            "t1": specs["t1_mask"]["mask_sha256"],
            "t2": specs["t2_mask"]["mask_sha256"],
            "change": specs["change_mask"]["mask_sha256"],
        }
    )
    return {"masks": specs, "state_sha256": identity}


def expected_sample(gt_pixels: int, *, split: str = "train") -> dict[str, object]:
    return {
        "sample_id": "sample",
        "parent_id": "train_1" if split == "train" else "val_1",
        "split": split,
        "crop": {"gt_pixels": gt_pixels},
        "semantic": {
            "t1_state": "background",
            "t2_state": "building" if gt_pixels else "background",
        },
    }


def runner_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    metrics = VALIDATOR.compute_metrics(prediction, reference)
    return {
        key: metrics[key]
        for key in ("pixel_error", "iou", "f1", "precision", "recall", "tp", "fp", "fn")
    } | {
        "tn": int(prediction.size)
        - int(metrics["tp"])
        - int(metrics["fp"])
        - int(metrics["fn"])
    }


def runner_metric_delta(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    return {
        "pixel_error_reduction": int(before["pixel_error"])
        - int(after["pixel_error"]),
        "iou": float(after["iou"]) - float(before["iou"]),
        "f1": float(after["f1"]) - float(before["f1"]),
        "tp": int(after["tp"]) - int(before["tp"]),
        "fp": int(after["fp"]) - int(before["fp"]),
        "fn": int(after["fn"]) - int(before["fn"]),
    }


def runner_metric_gate(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    delta = runner_metric_delta(before, after)
    nonempty = int(before["tp"]) + int(before["fn"]) > 0
    pixel_pass = int(delta["pixel_error_reduction"]) > 0
    iou_pass = float(delta["iou"]) > 1e-12
    f1_pass = float(delta["f1"]) > 1e-12
    fp_pass = int(after["fp"]) < int(before["fp"])
    return {
        "passed": pixel_pass and ((iou_pass and f1_pass) if nonempty else fp_pass),
        "pixel_error_strictly_decreased": pixel_pass,
        "nonempty_gt": nonempty,
        "iou_strictly_increased": iou_pass if nonempty else None,
        "f1_strictly_increased": f1_pass if nonempty else None,
        "fp_strictly_decreased": fp_pass if not nonempty else None,
        "comparison_epsilon": 1e-12,
    }


def successful_tool_evidence(
    action: dict[str, object],
    region_id: str,
    target_mask_pixels: int,
    hard_evidence: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    point = action["action"] in {"positive_point", "negative_point"}
    mode = "point" if point else "box"
    tool_input: dict[str, object] = {"authorized_region_id": region_id}
    receipt: dict[str, object] = {
        "tool": "simpleclick" if point else "sam3",
        "tool_input": tool_input,
        "tool_result": {
            "worker_mode": mode,
            "worker_report": {"status": "success", "mode": mode},
        },
    }
    if point:
        point_evidence = copy.deepcopy(hard_evidence["point_authorization"])
        tool_input.update(
            {
                "coordinate": action["coordinate"],
                "is_positive": action["action"] == "positive_point",
                "accepted_click_history": [],
                "proposal_box_xyxy": point_evidence["proposal_box_xyxy"],
                "proposal_component_area": point_evidence[
                    "proposal_component_area"
                ],
                "effective_roi_xyxy": point_evidence["effective_roi_xyxy"],
            }
        )
        receipt["point_authorization"] = {
            "region_id": region_id,
            **point_evidence,
        }
    else:
        receipt["box_authorization"] = {
            "region_id": region_id,
            "proposal_box_xyxy": action["box"],
            "geometry_match": True,
        }
    receipt["locality"] = copy.deepcopy(hard_evidence["locality"])
    reasons = list(hard_evidence["rejection_reasons"])
    rejected = bool(reasons)
    execution = {
        **copy.deepcopy(receipt),
        "candidate_accepted": not rejected,
        "verifier_skipped_by_hard_gate": rejected,
        "candidate_rejection_reasons": reasons,
        "previous_area_ratio": hard_evidence["previous_area_ratio"],
        "candidate_area_ratio": hard_evidence["candidate_area_ratio"],
        "candidate_area_delta": hard_evidence["candidate_area_delta"],
        "point_session_update": {
            "policy": "rebase_to_accepted_live_mask",
            "target_view": action["target_view"],
            "base_mask_pixels": target_mask_pixels,
        },
    }
    if rejected:
        execution["verifier_evidence"] = {
            "type": "runtime_hard_gate_precheck",
            "verifier_skipped": True,
            "rejection_reasons": reasons,
            "gt_available": False,
        }
    hard_gate = {
        "passed": not rejected,
        "rejected": rejected,
        "rejection_reasons": reasons,
        "verifier_skipped_by_hard_gate": rejected,
    }
    return receipt, execution, hard_gate


def valid_episode(
    root: Path,
    *,
    empty_gt: bool = False,
    already_exact: bool = False,
    sample_id: str = "sample",
    parent_id: str = "train_1",
    split: str = "train",
    artifact_prefix: str = "",
    source_hashes: dict[str, str] | None = None,
    endpoint_success: bool = True,
    already_endpoint: bool = False,
    empty_gt_max_fp: int = 0,
) -> dict[str, object]:
    def artifact_path(relative: str) -> str:
        return f"{artifact_prefix}/{relative}" if artifact_prefix else relative

    gt = np.zeros((4, 4), dtype=bool)
    if not empty_gt:
        gt[0, :2] = True
    if already_exact:
        initial_change = np.array(gt, copy=True)
    elif empty_gt:
        initial_change = np.zeros((4, 4), dtype=bool)
        initial_change[0, :2] = True
    else:
        initial_change = np.zeros((4, 4), dtype=bool)
    gt_spec = write_mask(root, artifact_path("gt.npy"), gt)
    initial = write_state(root, artifact_path("initial"), initial_change)
    initial["initializer_receipt"] = {
        "worker_mode": "initialize",
        "worker_report": {
            "status": "success",
            "mode": "initialize",
            "tool": "sam3",
            "operation": "fresh_dual_view_text_initialization",
        },
    }
    episode: dict[str, object] = {
        "schema_id": "qwen3vl_oracle_trajectory_episode_v1",
        "episode_id": sample_id,
        "sample_id": sample_id,
        "parent_id": parent_id,
        "split": split,
        "query": "building",
        "qwen_invoked": False,
        "qwen_call_count": 0,
        "source_hashes": source_hashes
        or {"t1": "1" * 64, "t2": "2" * 64, "label": "3" * 64},
        "gt_mask": gt_spec,
        "initial_state": initial,
        "max_steps": 3,
        "budget_hit": False,
        "steps": [],
    }
    initial_metrics = runner_metrics(initial_change, gt)
    initial["metrics"] = copy.deepcopy(initial_metrics)
    initial_endpoint_gate = VALIDATOR.endpoint_gate(
        VALIDATOR.compute_metrics(initial_change, gt),
        minimum_iou=0.9,
        minimum_f1=18 / 19,
        empty_gt_max_fp=empty_gt_max_fp,
        epsilon=1e-12,
    )
    final_change = np.array(initial_change, copy=True)
    region_id = "region-0"
    proposal = {
        "region_id": region_id,
        "audit_kind": "present",
        "component_seed_pixels": [0, 0],
        "box_pixels": [0, 0, 1, 1],
        "component_area": 2,
        "component_seed_t1_mask_white": False,
        "component_seed_t2_mask_white": bool(empty_gt),
    }
    frozen_proposals: list[dict[str, object]] = (
        [] if already_exact or already_endpoint else [proposal]
    )
    frozen_actions: list[dict[str, object]] = []
    action_catalog: list[dict[str, object]] = []
    execution_shortlist: list[str] = []
    candidates: list[dict[str, object]] = []
    beneficial_shortlist: list[str] = []
    selected_candidate_id: str | None = None
    canonical_replay: dict[str, object] | None = None

    if already_exact or already_endpoint:
        episode["episode_status"] = (
            "already_exact" if already_exact else "already_endpoint"
        )
    else:
        next_change = np.array(initial_change, copy=True)
        if empty_gt:
            next_change[0, 1] = False
            action = {
                "action": "negative_point",
                "target_view": "t2",
                "coordinate_space": "pixel",
                "coordinate": [0, 0],
            }
            observed_rgb = "background"
        else:
            next_change[0, 0] = True
            if endpoint_success:
                next_change[0, 1] = True
            action = {
                "action": "box",
                "target_view": "t2",
                "coordinate_space": "pixel",
                "box": [0, 0, 1, 1],
            }
            observed_rgb = "building"
        candidate_id = f"{sample_id}--step-001--{region_id}--{action['action']}--t2"
        frozen_actions = VALIDATOR._rebuild_useful_action_inventory(
            sample_id, 1, [proposal], "fixture"
        )
        frozen_action = next(
            item
            for item in frozen_actions
            if item["candidate_id"] == candidate_id
        )
        rgb_gate = {
            "passed": True,
            "target_view": "t2",
            "action": action["action"],
            "observed_rgb_state": observed_rgb,
            "required_rgb_state": observed_rgb,
            "policy": "negative_requires_background_positive_or_box_requires_building",
        }
        action_catalog = []
        for item in frozen_actions:
            item_action = item["action"]
            observed = (
                "background"
                if item_action["action"] == "negative_point"
                else "building"
                if item_action["target_view"] == "t2" and not empty_gt
                else "background"
            )
            required = (
                "background"
                if item_action["action"] == "negative_point"
                else "building"
            )
            item_gate = {
                "passed": observed == required,
                "target_view": item_action["target_view"],
                "action": item_action["action"],
                "observed_rgb_state": observed,
                "required_rgb_state": required,
                "policy": "negative_requires_background_positive_or_box_requires_building",
            }
            action_catalog.append(
                {
                    **copy.deepcopy(item),
                    "rgb_semantic_gate": item_gate,
                    "proposal_box_current_change_error_pixels": 2,
                    "selected_for_execution": False,
                }
            )
        execution_shortlist = sorted(
            item["candidate_id"]
            for item in action_catalog
            if item["rgb_semantic_gate"]["passed"]
        )[:4]
        for item in action_catalog:
            item["selected_for_execution"] = (
                item["candidate_id"] in execution_shortlist
            )
        assert execution_shortlist[0] == candidate_id
        candidate = write_state(
            root, artifact_path("step1/candidate"), next_change
        )
        canonical = write_state(
            root, artifact_path("step1/canonical"), next_change
        )
        replay = write_state(root, artifact_path("step1/replay"), next_change)
        candidate_metrics = runner_metrics(next_change, gt)
        candidate_endpoint_gate = VALIDATOR.endpoint_gate(
            VALIDATOR.compute_metrics(next_change, gt),
            minimum_iou=0.9,
            minimum_f1=18 / 19,
            empty_gt_max_fp=empty_gt_max_fp,
            epsilon=1e-12,
        )
        delta = runner_metric_delta(initial_metrics, candidate_metrics)
        metric_gate = runner_metric_gate(initial_metrics, candidate_metrics)
        source_state = VALIDATOR._load_state(root, initial, "fixture.initial")
        selected_candidate_state = VALIDATOR._load_state(
            root, candidate, "fixture.candidate"
        )
        selected_hard_evidence = VALIDATOR._recompute_hard_gate_evidence(
            source_state,
            selected_candidate_state,
            action,
            proposal,
            "fixture",
        )
        branch_receipt, branch_execution, hard_gate = successful_tool_evidence(
            action,
            region_id,
            int(next_change.sum()),
            selected_hard_evidence,
        )
        replay_receipt, replay_execution, replay_hard_gate = successful_tool_evidence(
            action,
            region_id,
            int(next_change.sum()),
            selected_hard_evidence,
        )
        candidate_record = {
            **copy.deepcopy(frozen_action),
            "ordinal": 0,
            "source_state_mask_sha256": {
                name: initial["masks"][f"{name}_mask"]["mask_sha256"]
                for name in ("t1", "t2", "change")
            },
            "candidate_state": candidate,
            "hard_gate": hard_gate,
            "rgb_semantic_gate": rgb_gate,
            "tool_receipt": branch_receipt,
            "runtime_execution": branch_execution,
            "no_qwen_audit": {"qwen_invoked": False, "qwen_calls": 0},
            "metrics": candidate_metrics,
            "endpoint_gate": candidate_endpoint_gate,
            "metric_delta": delta,
            "per_step_metric_gate": metric_gate,
        }
        candidates = [candidate_record]
        for ordinal, alternate_id in enumerate(execution_shortlist[1:], start=1):
            alternate_catalog = next(
                item
                for item in action_catalog
                if item["candidate_id"] == alternate_id
            )
            alternate_action = alternate_catalog["action"]
            alternate_state = write_state(
                root,
                artifact_path(f"step1/candidate-{ordinal}"),
                next_change,
            )
            alternate_receipt, alternate_execution, alternate_gate = (
                successful_tool_evidence(
                    alternate_action,
                    region_id,
                    int(next_change.sum()),
                    VALIDATOR._recompute_hard_gate_evidence(
                        source_state,
                        VALIDATOR._load_state(
                            root,
                            alternate_state,
                            f"fixture.candidate-{ordinal}",
                        ),
                        alternate_action,
                        alternate_catalog["proposal"],
                        "fixture",
                    ),
                )
            )
            candidates.append(
                {
                    **{
                        key: copy.deepcopy(alternate_catalog[key])
                        for key in (
                            "candidate_id",
                            "region_id",
                            "reachability_tier",
                            "action",
                            "proposal",
                        )
                    },
                    "ordinal": ordinal,
                    "source_state_mask_sha256": copy.deepcopy(
                        candidate_record["source_state_mask_sha256"]
                    ),
                    "candidate_state": alternate_state,
                    "hard_gate": alternate_gate,
                    "rgb_semantic_gate": copy.deepcopy(
                        alternate_catalog["rgb_semantic_gate"]
                    ),
                    "tool_receipt": alternate_receipt,
                    "runtime_execution": alternate_execution,
                    "no_qwen_audit": {
                        "qwen_invoked": False,
                        "qwen_calls": 0,
                    },
                    "metrics": copy.deepcopy(candidate_metrics),
                    "endpoint_gate": copy.deepcopy(candidate_endpoint_gate),
                    "metric_delta": copy.deepcopy(delta),
                    "per_step_metric_gate": copy.deepcopy(metric_gate),
                }
            )
        beneficial_shortlist = list(execution_shortlist)
        selected_candidate_id = candidate_id
        mask_hashes = {
            name: candidate["masks"][f"{name}_mask"]["mask_sha256"]
            for name in ("t1", "t2", "change")
        }
        canonical_replay = {
            "branch_candidate_state_mask_sha256": mask_hashes,
            "canonical_state_mask_sha256": mask_hashes,
            "all_three_mask_hashes_match": True,
            "state": canonical,
            "metrics": candidate_metrics,
            "endpoint_gate": candidate_endpoint_gate,
            "tool_receipt": replay_receipt,
            "hard_gate": replay_hard_gate,
            "runtime_execution": replay_execution,
        }
        episode["episode_status"] = (
            "endpoint_success" if candidate_endpoint_gate["passed"]
            else "locally_reachable_only"
        )
        episode["steps"] = [
            {
                "step_index": 1,
                "action": action,
                "proposal": proposal,
                "candidate_id": candidate_id,
                "reachability_tier": frozen_action["reachability_tier"],
                "source_state_sha256": initial["state_sha256"],
                "source_state_mask_sha256": candidate_record[
                    "source_state_mask_sha256"
                ],
                "qwen_invoked": False,
                "qwen_calls": 0,
                "tool_success": True,
                "tool_receipt": replay_receipt,
                "endpoint_gate": candidate_endpoint_gate,
                "hard_gate": hard_gate,
                "rgb_semantic": rgb_gate,
                "candidate_accepted_by_oracle_selector": True,
                "branch_replay_mask_hash_match": True,
                "candidate_state": candidate,
                "canonical_state": canonical,
                "replay_state": replay,
                "budget_hit": False,
            }
        ]
        final_change = np.array(next_change, copy=True)

    episode_prefix = Path(artifact_prefix).parent if artifact_prefix else Path()
    catalog_relative = (episode_prefix / "catalog_freeze_before_gt.json").as_posix()
    initial_hashes = {
        name: initial["masks"][f"{name}_mask"]["mask_sha256"]
        for name in ("t1", "t2", "change")
    }
    catalog_payload = {
        "schema_id": VALIDATOR.CATALOG_SCHEMA,
        "episode_id": sample_id,
        "step_index": 1,
        "source_state_sha256": initial["state_sha256"],
        "source_state_mask_sha256": initial_hashes,
        "live_proposals": frozen_proposals,
        "live_proposals_canonical_sha256": VALIDATOR.canonical_sha256(
            frozen_proposals
        ),
        "actions": frozen_actions,
        "action_catalog_canonical_sha256": VALIDATOR.canonical_sha256(
            frozen_actions
        ),
        "action_count": len(frozen_actions),
        "gt_access_count_before": 0,
        "contains_gt_derived_fields": False,
        "qwen_invoked": False,
        "qwen_calls": 0,
    }
    catalog_path = root / catalog_relative
    write_json(catalog_path, catalog_payload)
    catalog_identity = {
        "path": catalog_relative,
        "sha256": sha256(catalog_path),
        "canonical_sha256": VALIDATOR.canonical_sha256(catalog_payload),
        "size_bytes": catalog_path.stat().st_size,
        "step_index": 1,
    }
    selection_rounds = [
        {
            "step_index": 1,
            "source_state_mask_sha256": initial_hashes,
            "source_state_sha256": initial["state_sha256"],
            "source_metrics": initial_metrics,
            "source_endpoint_gate": initial_endpoint_gate,
            "live_proposals": frozen_proposals,
            "live_proposals_canonical_sha256": VALIDATOR.canonical_sha256(
                frozen_proposals
            ),
            "action_catalog": action_catalog,
            "gt_free_catalog_freeze": catalog_identity,
            "action_catalog_count": len(action_catalog),
            "execution_shortlist": execution_shortlist,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "beneficial_shortlist": beneficial_shortlist,
            "selected_candidate_id": selected_candidate_id,
            "canonical_replay": canonical_replay,
            "candidate_batch_complete_before_final_gt_scoring": True,
        }
    ]
    final_metrics = runner_metrics(final_change, gt)
    final_endpoint_gate = VALIDATOR.endpoint_gate(
        VALIDATOR.compute_metrics(final_change, gt),
        minimum_iou=0.9,
        minimum_f1=18 / 19,
        empty_gt_max_fp=empty_gt_max_fp,
        epsilon=1e-12,
    )
    initial_error = int(initial_metrics["pixel_error"])
    reduction_ratio = (
        None
        if initial_error == 0
        else (initial_error - int(final_metrics["pixel_error"])) / initial_error
    )
    episode["final_metrics"] = final_metrics
    final_artifact = initial if not episode["steps"] else canonical
    episode["final_state_mask_sha256"] = {
        name: final_artifact["masks"][f"{name}_mask"]["mask_sha256"]
        for name in ("t1", "t2", "change")
    }
    episode["final_state_sha256"] = final_artifact["state_sha256"]
    episode["final_error_reduction_ratio"] = reduction_ratio
    episode["local_reachability"] = {
        "passed": bool(episode["steps"] and reduction_ratio is not None and reduction_ratio >= 0.05),
        "minimum_final_error_reduction_ratio": 0.05,
    }
    episode["endpoint_reachability"] = {
        "initial_gate": initial_endpoint_gate,
        "final_gate": final_endpoint_gate,
        "success_step": (
            1 if episode["episode_status"] == "endpoint_success" else None
        ),
        "exact_recovery": bool(np.array_equal(final_change, gt)),
    }
    catalog_freezes = [catalog_identity]
    if episode["episode_status"] == "locally_reachable_only":
        terminal_catalog_payload = {
            "schema_id": VALIDATOR.CATALOG_SCHEMA,
            "episode_id": sample_id,
            "step_index": 2,
            "source_state_sha256": final_artifact["state_sha256"],
            "source_state_mask_sha256": copy.deepcopy(
                episode["final_state_mask_sha256"]
            ),
            "live_proposals": [],
            "live_proposals_canonical_sha256": VALIDATOR.canonical_sha256([]),
            "actions": [],
            "action_catalog_canonical_sha256": VALIDATOR.canonical_sha256([]),
            "action_count": 0,
            "gt_access_count_before": 1,
            "contains_gt_derived_fields": False,
            "qwen_invoked": False,
            "qwen_calls": 0,
        }
        terminal_catalog_relative = (
            episode_prefix / "catalog_freeze_step-002.json"
        ).as_posix()
        terminal_catalog_path = root / terminal_catalog_relative
        write_json(terminal_catalog_path, terminal_catalog_payload)
        terminal_catalog_identity = {
            "path": terminal_catalog_relative,
            "sha256": sha256(terminal_catalog_path),
            "canonical_sha256": VALIDATOR.canonical_sha256(
                terminal_catalog_payload
            ),
            "size_bytes": terminal_catalog_path.stat().st_size,
            "step_index": 2,
        }
        catalog_freezes.append(terminal_catalog_identity)
        selection_rounds.append(
            {
                "step_index": 2,
                "source_state_mask_sha256": copy.deepcopy(
                    episode["final_state_mask_sha256"]
                ),
                "source_state_sha256": final_artifact["state_sha256"],
                "source_metrics": copy.deepcopy(final_metrics),
                "source_endpoint_gate": copy.deepcopy(final_endpoint_gate),
                "live_proposals": [],
                "live_proposals_canonical_sha256": VALIDATOR.canonical_sha256(
                    []
                ),
                "action_catalog": [],
                "gt_free_catalog_freeze": terminal_catalog_identity,
                "action_catalog_count": 0,
                "execution_shortlist": [],
                "candidate_count": 0,
                "candidates": [],
                "beneficial_shortlist": [],
                "selected_candidate_id": None,
                "canonical_replay": None,
                "candidate_batch_complete_before_final_gt_scoring": True,
            }
        )
    episode["selection_rounds"] = selection_rounds
    episode["gt_access_audit"] = {
        "access_count": 1,
        "first_access_phase": "after_initial_live_action_catalog_freeze_before_offline_top4_shortlist",
        "catalog_freeze_before_first_access": catalog_identity,
        "catalog_freezes": catalog_freezes,
        "label_passed_to_runtime_factory": False,
        "label_passed_to_environment": False,
        "label_passed_to_initializer_or_tools": False,
        "runtime_factory_sample_type": "RuntimeEpisode_GT_free",
        "offline_label_loader_only": True,
    }
    episode["termination"] = {
        "reason": (
            "already_exact"
            if already_exact
            else "already_endpoint"
            if already_endpoint
            else "exact_recovery"
            if np.array_equal(final_change, gt)
            else "endpoint_reached"
            if final_endpoint_gate["passed"]
            else "no_beneficial"
        ),
        "selected_step_count": len(episode["steps"]),
        "candidate_count": sum(
            len(round_value["candidates"])
            for round_value in selection_rounds
        ),
        "exact": bool(np.array_equal(final_change, gt)),
        "budget_hit": False,
    }
    episode["budget"] = {
        "hit": False,
        "max_candidates_per_step": 4,
        "max_distinct_candidates": 12,
        "max_total_tool_executions": 24,
        "distinct_candidate_count": len(candidates),
        "tool_execution_count": len(candidates) + len(episode["steps"]),
    }
    episode["rgb_semantics"] = {
        "default": {
            "t1": "background",
            "t2": "background" if empty_gt else "building",
        },
        "evidence": "",
        "source": "",
        "status": "",
        "by_region": {},
        "by_step": {},
    }
    return episode


def validate_fixture(
    root: Path,
    episode: dict[str, object],
    gt_pixels: int,
    *,
    empty_gt_max_fp: int = 0,
):
    sample = expected_sample(gt_pixels, split=str(episode.get("split", "train")))
    sample["sample_id"] = episode.get("sample_id", episode.get("episode_id"))
    sample["parent_id"] = episode.get("parent_id")
    return VALIDATOR.validate_episode(
        episode,
        root=root,
        expected_sample=sample,
        epsilon=1e-12,
        minimum_reduction=0.05,
        source_episode_sha256="a" * 64,
        endpoint_empty_gt_max_fp=empty_gt_max_fp,
    )


def bind_matching_to_report(
    report: dict[str, object], matching: dict[str, object]
) -> None:
    report["matching"] = {
        key: copy.deepcopy(matching[key])
        for key in (
            "endpoint_success_episode_count",
            "transition_count",
            "bijection_100_percent",
            "split_artifacts",
            "training_authorized",
            "authorization_scope",
            "authorized_splits",
            "raw_artifacts_sft_ready",
            "requires_deterministic_gt_free_format_conversion",
        )
    }


def minimal_config(
    *,
    status: str = "preregistered_pending_implementation_and_runtime_validation",
) -> dict[str, object]:
    ordered = [
        "train_407_r3c2",
        "train_114_r3c1",
        "train_364_r0c2",
        "val_43_r1c1",
        "val_41_r3c2",
        "val_60_r3c0",
    ]
    samples = {
        sample_id: {"split": "train" if sample_id.startswith("train_") else "val"}
        for sample_id in ordered
    }
    return {
        "schema_id": VALIDATOR.CONFIG_SCHEMA,
        "status": status,
        "ordered_samples": ordered,
        "samples": samples,
        "execution_protocol": {
            "seed": 42,
            "query": "building",
            "sam3_resolution": 1008,
            "tool_device": "cuda",
            "tool_timeout_seconds": 900,
            "max_steps": 3,
            "accepted_actions_per_step": 1,
            "qwen_policy": {
                "enabled": False,
                "model_loaded": False,
                "allowed_call_count": 0,
            },
        },
        "candidate_and_replay_budget": {
            "max_candidate_actions_per_step": 4,
            "max_steps_per_sample": 3,
            "max_distinct_candidate_actions_per_sample": 12,
            "max_total_tool_executions_including_selected_action_replay_per_sample": 24,
        },
        "proposal_config": {
            "max_regions": 1,
            "max_delta_regions": 1,
            "min_component_area": 4,
            "padding_ratio": 0.25,
            "empty_mask_scan_grid_size": 3,
        },
        "authorized_negative_point_input_scope": "full_session",
        "matching": {
            "mode": "overlap_presence",
            "overlap_threshold": 0.25,
            "t12_min_instance_area": 0,
            "cd_min_instance_area": 0,
        },
        "hard_gate_config": {
            "max_selection_area_delta": 0.25,
            "max_locality_outside_ratio": 0.1,
            "max_target_mask_change_ratio": 0.25,
            "max_component_count_delta": 4,
            "max_point_proposal_change_ratio": 0.25,
            "max_point_proposal_change_pixels": 256,
            "max_point_image_change_ratio": 0.004,
            "point_proposal_change_gate_enabled": True,
            "modification_or_retuning_authorized": False,
        },
        "gt_boundary": {
            "gt_visible_to_sam3_initializer": False,
            "gt_visible_to_simpleclick": False,
            "gt_visible_to_sam3_box_tool": False,
            "gt_visible_to_environment_proposal_enumeration": False,
            "gt_visible_to_hard_gates": False,
            "gt_visible_to_qwen": False,
            "action_catalog_complete_before_gt_shortlisting": True,
            "shortlisted_candidate_batch_complete_before_final_gt_selection": True,
        },
        "data_derivation_contract": {
            "enabled_only_for_endpoint_success_episodes": True,
            "same_source_episode_required": True,
            "cross_episode_splicing_forbidden": True,
            "already_exact_generates_data": False,
            "already_endpoint_generates_data": False,
            "locally_reachable_only_generates_data": False,
            "inconclusive_generates_data": False,
            "failed_episode_generates_data": False,
        },
        "artifact_plan": {"output_root": VALIDATOR.EXPECTED_OUTPUT_ROOT},
        "tools_and_models": {
            "qwen_base_model_disabled": {"load_for_pilot": False}
        },
        "resource_plan": {
            "slurm_required": True,
            "partition": "gre",
            "maximum_total_gpus": 3,
            "planned_generation_shards": 3,
            "maximum_generate_select_rounds": 3,
            "gpus_per_shard": 1,
            "accelerator": "A800",
            "cpus_per_shard": 4,
            "memory_per_shard": "48G",
            "time_per_shard": "02:00:00",
            "peak_gpu_concurrency": 3,
            "independent_post_generation_validation": {
                "gpus": 0,
                "cpus": 2,
                "memory": "16G",
            },
            "shards": [
                {"shard_id": 0, "samples": [ordered[0], ordered[3]]},
                {"shard_id": 1, "samples": [ordered[1], ordered[4]]},
                {"shard_id": 2, "samples": [ordered[2], ordered[5]]},
            ],
        },
        "metric_and_success_contract": {
            "comparison_epsilon": 1e-12,
            "already_endpoint": {
                "episode_status": "already_endpoint",
                "generate_c_iso_or_t_traj": False,
            },
            "locally_reachable": {
                "minimum_final_error_reduction_ratio": 0.05,
                "episode_status_if_endpoint_not_reached": "locally_reachable_only",
                "may_generate_c_iso_or_t_traj": False,
            },
            "endpoint_success": {
                "episode_status": "endpoint_success",
                "nonempty_gt_minimum_final_iou": 0.9,
                "nonempty_gt_minimum_final_f1": 18 / 19,
                "empty_gt_maximum_final_fp": 64,
                "may_generate_matched_candidate_data": True,
            },
            "global_pilot_release_gate": {
                "required_endpoint_successes": 6,
                "required_evaluated_episodes": 6,
                "maximum_inconclusive_or_not_evaluated": 0,
                "minimum_nontrivial_accepted_trajectories": 6,
                "proceed_to_next_training_data_iteration_only_if_passed": True,
            },
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_source_file(path: Path, value: int) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((4, 4), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
    return {"path": str(path), "sha256": sha256(path)}


def write_shard(
    root: Path,
    *,
    shard_id: int,
    num_shards: int,
    sample_ids: list[str],
    samples: dict[str, dict[str, object]],
    config_sha256: str,
    config_canonical_sha256: str,
    runtime_commit: str = "a" * 40,
) -> Path:
    shard_root = root / f"shard-{shard_id}"
    entries = []
    for sample_id in sample_ids:
        sample = samples[sample_id]
        source_hashes = {
            name: sample["source_parent"][name]["sha256"]
            for name in ("t1", "t2", "label")
        }
        episode = valid_episode(
            shard_root,
            sample_id=sample_id,
            parent_id=sample["parent_id"],
            split=sample["split"],
            artifact_prefix=f"episodes/{sample_id}/artifacts",
            source_hashes=source_hashes,
        )
        episode["source_parent"] = copy.deepcopy(sample["source_parent"])
        relative = f"episodes/{sample_id}/episode.json"
        episode_path = shard_root / relative
        write_json(episode_path, episode)
        entries.append(
            {
                "episode_id": sample_id,
                "path": relative,
                "sha256": sha256(episode_path),
                "canonical_sha256": VALIDATOR.canonical_sha256(episode),
                "size_bytes": episode_path.stat().st_size,
            }
        )
    runner_path = ROOT / "tools/run_qwen3vl_oracle_trajectory.py"
    index = {
        "schema_id": "qwen3vl_oracle_trajectory_shard_index_v1",
        "status": "completed",
        "shard_index": shard_id,
        "num_shards": num_shards,
        "config": {
            "path": str(root / "pilot.json"),
            "sha256": config_sha256,
            "canonical_sha256": config_canonical_sha256,
        },
        "runtime_commit": runtime_commit,
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        "qwen_invoked": False,
        "qwen_calls": 0,
        "episodes": entries,
    }
    index_path = shard_root / "shard_index.json"
    write_json(index_path, index)
    index_artifact = {
        "path": "shard_index.json",
        "sha256": sha256(index_path),
        "canonical_sha256": VALIDATOR.canonical_sha256(index),
        "size_bytes": index_path.stat().st_size,
    }
    write_json(
        shard_root / "generation_complete.json",
        {
            "schema_id": "qwen3vl_oracle_trajectory_generation_complete_v1",
            "status": "completed",
            "shard_index": shard_id,
            "num_shards": num_shards,
            "episode_ids": sample_ids,
            "runtime_commit": runtime_commit,
            "qwen_invoked": False,
            "qwen_calls": 0,
            "shard_index_sha256": sha256(index_path),
            "shard_index_artifact": index_artifact,
        },
    )
    return shard_root


def build_sharded_fixture(
    root: Path,
    ordered_samples: list[str],
    shard_samples: list[list[str]],
    *,
    runtime_commits: list[str] | None = None,
):
    samples: dict[str, dict[str, object]] = {}
    for sample_index, sample_id in enumerate(ordered_samples):
        split = "train" if sample_id.startswith("train_") else "val"
        parent_id = sample_id.rsplit("_r", 1)[0]
        sources = {
            name: write_source_file(
                root
                / "sources"
                / f"{split}_original"
                / name
                / f"{sample_id}.png",
                sample_index * 3 + source_index + 1,
            )
            for source_index, name in enumerate(("t1", "t2", "label"))
        }
        samples[sample_id] = {
            "split": split,
            "parent_id": parent_id,
            "crop": {"gt_pixels": 2},
            "semantic": {
                "t1_state": "background",
                "t2_state": "building",
            },
            "source_parent": sources,
        }
    config = minimal_config(status="frozen_pending_exact_commit_validation")
    resource_plan = copy.deepcopy(config["resource_plan"])
    resource_plan["shards"] = [
        {"shard_id": shard_id, "samples": sample_ids}
        for shard_id, sample_ids in enumerate(shard_samples)
    ]
    config.update(
        {
            "ordered_samples": ordered_samples,
            "samples": samples,
            "resource_plan": resource_plan,
        }
    )
    config_path = root / "pilot.json"
    write_json(config_path, config)
    config_sha256 = sha256(config_path)
    config_canonical_sha256 = VALIDATOR.canonical_sha256(config)
    runtime_commits = runtime_commits or ["a" * 40] * len(shard_samples)
    shard_roots = [
        write_shard(
            root,
            shard_id=shard_id,
            num_shards=len(shard_samples),
            sample_ids=sample_ids,
            samples=samples,
            config_sha256=config_sha256,
            config_canonical_sha256=config_canonical_sha256,
            runtime_commit=runtime_commits[shard_id],
        )
        for shard_id, sample_ids in enumerate(shard_samples)
    ]
    return config, config_sha256, shard_roots


def initialize_git_repository(path: Path, payload: str) -> str:
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tracked = path / "tracked.py"
    tracked.write_text(payload, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "tracked.py"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Preflight Test",
            "-c",
            "user.email=preflight@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def save_png(path: Path, pixels: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L").save(path)
    return {"path": str(path), "sha256": sha256(path)}


def build_preflight_fixture(root: Path):
    workspace = root / "workspace"
    repository = workspace / "repository"
    configs = repository / "configs"
    configs.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(repository)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dataset = workspace / "dataset"
    curation = workspace / "archive" / "curation"
    tools_dir = workspace / "identities"
    tools_dir.mkdir(parents=True)

    tool_specs: dict[str, dict[str, str]] = {}
    for name, filename, executable in (
        ("segagent_python", "segagent-python", True),
        ("omniovcd_python", "omniovcd-python", True),
        ("simpleclick_checkpoint", "simpleclick.pth", False),
        ("sam3_checkpoint", "sam3.pt", False),
        ("sam3_bpe", "bpe.txt.gz", False),
    ):
        path = tools_dir / filename
        path.write_bytes(f"{name}\n".encode("utf-8"))
        if executable:
            path.chmod(0o755)
        tool_specs[name] = {"path": str(path), "sha256": sha256(path)}

    external_specs = []
    for name, dirname in (
        ("SegAgent_with_bundled_SimpleClick", "SegAgent"),
        ("OmniOVCD_SAM3", "OmniOVCD"),
    ):
        path = workspace / "reference" / dirname
        commit = initialize_git_repository(path, f"{name}\n")
        external_specs.append(
            {
                "name": name,
                "path": str(path),
                "commit": commit,
                "tracked_worktree_must_be_clean": True,
            }
        )

    runtime_specs = []
    for relative, payload in (
        ("change_agent/environment.py", "environment\n"),
        ("tools/seeded_segmentation_worker.py", "worker\n"),
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        runtime_specs.append({"path": relative, "sha256": sha256(path)})

    ordered = [
        "train_407_r3c2",
        "train_114_r3c1",
        "train_364_r0c2",
        "val_43_r1c1",
        "val_41_r3c2",
        "val_60_r3c0",
    ]
    samples: dict[str, dict[str, object]] = {}
    manual_samples = []
    manifest_samples = []
    for index, sample_id in enumerate(ordered):
        split = "train" if sample_id.startswith("train_") else "val"
        parent_id = sample_id.rsplit("_r", 1)[0]
        source_split = f"{split}_original"
        legacy_id = f"{split}_{sample_id}_changed"
        t1_pixels = np.full((4, 4), 10 + 2 * index, dtype=np.uint8)
        t2_pixels = np.full((4, 4), 11 + 2 * index, dtype=np.uint8)
        label_pixels = np.zeros((4, 4), dtype=np.uint8)
        label_pixels.reshape(-1)[: index + 1] = 255
        source_parent = {
            "t1": save_png(dataset / source_split / "A" / f"{parent_id}.png", t1_pixels),
            "t2": save_png(dataset / source_split / "B" / f"{parent_id}.png", t2_pixels),
            "label": save_png(
                dataset / source_split / "label" / f"{parent_id}.png",
                label_pixels,
            ),
        }
        artifact_root = curation / "samples" / legacy_id
        artifacts = {
            "t1": save_png(artifact_root / "t1.png", t1_pixels),
            "t2": save_png(artifact_root / "t2.png", t2_pixels),
            "gt_change": save_png(artifact_root / "gt_change.png", label_pixels),
        }
        prior = {
            "t1": artifacts["t1"]["sha256"],
            "t2": artifacts["t2"]["sha256"],
            "label": artifacts["gt_change"]["sha256"],
        }
        evidence = f"fixture RGB evidence {index}"
        samples[sample_id] = {
            "legacy_candidate_id": legacy_id,
            "split": split,
            "split_role": (
                "train_data_derivation_candidate"
                if split == "train"
                else "heldout_validation_only"
            ),
            "parent_id": parent_id,
            "kind": "changed",
            "semantic": {
                "t1_state": "background",
                "t2_state": "building",
                "rgb_evidence": evidence,
                "source": "fixture",
                "status": "provisional_non_blinded",
            },
            "source_parent": source_parent,
            "crop": {
                "row": 0,
                "column": 0,
                "xyxy_exclusive": [0, 0, 4, 4],
                "shape": [4, 4],
                "gt_pixels": index + 1,
                "largest_curated_component_pixels": index + 1,
                "prior_curation_png_sha256": prior,
            },
        }
        manual_samples.append(
            {
                "candidate_id": legacy_id,
                "use": True,
                "t1_state": "background",
                "t2_state": "building",
                "evidence": evidence,
            }
        )
        legacy_root = Path("/retired/output/curation/samples") / legacy_id
        manifest_samples.append(
            {
                "candidate_id": legacy_id,
                "split": split,
                "parent_id": parent_id,
                "patch_row": 0,
                "patch_column": 0,
                "kind": "changed",
                "gt_pixels": index + 1,
                "component_pixels": index + 1,
                "source_paths": {
                    channel: identity["path"]
                    for channel, identity in source_parent.items()
                },
                "source_sha256": {
                    channel: identity["sha256"]
                    for channel, identity in source_parent.items()
                },
                "artifacts": {
                    "t1": str(legacy_root / "t1.png"),
                    "t2": str(legacy_root / "t2.png"),
                    "gt_change": str(legacy_root / "gt_change.png"),
                },
                "artifact_sha256": {
                    "t1": artifacts["t1"]["sha256"],
                    "t2": artifacts["t2"]["sha256"],
                    "gt_change": artifacts["gt_change"]["sha256"],
                },
            }
        )

    manual_path = configs / "manual.json"
    write_json(
        manual_path,
        {"schema": "manual_schema_v1", "samples": manual_samples},
    )
    manifest_path = curation / "curation_manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "curation_schema_v1",
            "seed": 801,
            "dataset_root": str(dataset),
            "source_splits": ["train_original", "val_original"],
            "test_data_used": False,
            "sam3": {
                "checkpoint": tool_specs["sam3_checkpoint"]["path"],
                "checkpoint_sha256": tool_specs["sam3_checkpoint"]["sha256"],
                "bpe": tool_specs["sam3_bpe"]["path"],
                "bpe_sha256": tool_specs["sam3_bpe"]["sha256"],
                "resolution": 1008,
                "query": "building",
            },
            "samples": manifest_samples,
        },
    )
    config: dict[str, object] = {
        "schema_id": VALIDATOR.CONFIG_SCHEMA,
        "status": "frozen_pending_exact_commit_validation",
        "scope_and_interpretation": {
            "test_split_used": False,
            "qwen_invoked": False,
        },
        "provenance": {
            "manual_semantic_config": {
                "path": "configs/manual.json",
                "sha256": sha256(manual_path),
                "schema": "manual_schema_v1",
            },
            "curation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "schema": "curation_schema_v1",
                "curation_seed": 801,
            },
        },
        "dataset_contract": {
            "dataset_root": str(dataset),
            "allowed_source_splits": ["train_original", "val_original"],
            "forbidden_source_splits": ["test", "test_256"],
            "parent_shape": [4, 4],
            "crop_shape": [4, 4],
            "crop_coordinates": "xyxy_exclusive_in_parent_pixels",
            "parent_split_isolation": {
                "required": True,
                "train_parent_ids": ["train_407", "train_114", "train_364"],
                "validation_parent_ids": ["val_43", "val_41", "val_60"],
                "intersection_must_be_empty": True,
            },
        },
        "ordered_samples": ordered,
        "samples": samples,
        "execution_protocol": {
            "query": "building",
            "sam3_resolution": 1008,
            "max_steps": 3,
            "accepted_actions_per_step": 1,
            "qwen_policy": {
                "enabled": False,
                "model_loaded": False,
                "allowed_call_count": 0,
            },
        },
        "tools_and_models": {
            "workspace_root": str(workspace),
            "repository": str(repository),
            **tool_specs,
            "qwen_base_model_disabled": {
                "path": str(workspace / "disabled-qwen"),
                "load_for_pilot": False,
            },
            "external_source_repositories": external_specs,
            "runtime_sources": runtime_specs,
        },
    }
    frozen = minimal_config(status="frozen_pending_exact_commit_validation")
    for key in (
        "execution_protocol",
        "candidate_and_replay_budget",
        "proposal_config",
        "authorized_negative_point_input_scope",
        "matching",
        "hard_gate_config",
        "gt_boundary",
        "data_derivation_contract",
        "artifact_plan",
        "resource_plan",
        "metric_and_success_contract",
    ):
        config[key] = copy.deepcopy(frozen[key])
    config_path = configs / "pilot.json"
    write_json(config_path, config)
    return config, config_path, {
        "dataset": dataset,
        "curation": curation,
        "manual": manual_path,
        "manifest": manifest_path,
        "repository": repository,
        "external": external_specs,
    }


class PreflightContractTests(unittest.TestCase):
    def test_preflight_verifies_all_frozen_inputs_and_archive_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path, _ = build_preflight_fixture(Path(directory))
            receipt = VALIDATOR.run_preflight(config, config_path)

        self.assertEqual(receipt["schema_id"], VALIDATOR.PREFLIGHT_SCHEMA)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["dataset"]["sample_count"], 6)
        self.assertEqual(receipt["dataset"]["parent_file_count"], 18)
        self.assertEqual(receipt["dataset"]["curation_artifact_count"], 18)
        self.assertEqual(len(receipt["runtime_sources"]), 2)
        self.assertEqual(len(receipt["external_source_repositories"]), 2)
        self.assertTrue(
            all(
                artifact["relocated_from_manifest_path"]
                for sample in receipt["dataset"]["samples"]
                for artifact in sample["curation_artifacts"].values()
            )
        )
        self.assertEqual(receipt["qwen"], {
            "loaded": False,
            "calls": 0,
            "policy_enabled": False,
        })

    def test_parent_hash_gt_and_curation_artifact_tampering_fail_closed(self):
        mutations = (
            (
                "source t1 SHA256 differs",
                lambda config, paths: config["samples"]["train_407_r3c2"][
                    "source_parent"
                ]["t1"].update({"sha256": "0" * 64}),
            ),
            (
                "GT pixel count differs",
                lambda config, paths: config["samples"]["train_407_r3c2"][
                    "crop"
                ].update({"gt_pixels": 2}),
            ),
            (
                "curation artifact t1 SHA256 differs",
                lambda config, paths: save_png(
                    paths["curation"]
                    / "samples"
                    / "train_train_407_r3c2_changed"
                    / "t1.png",
                    np.full((4, 4), 255, dtype=np.uint8),
                ),
            ),
        )
        for pattern, mutate in mutations:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                config, config_path, paths = build_preflight_fixture(Path(directory))
                mutate(config, paths)
                write_json(config_path, config)
                with self.assertRaisesRegex(VALIDATOR.ValidationError, pattern):
                    VALIDATOR.run_preflight(config, config_path)

    def test_preflight_rejects_missing_budget_and_wrong_hard_gate_before_hashing(self):
        mutations = (
            lambda config: config["candidate_and_replay_budget"].pop(
                "max_candidate_actions_per_step"
            ),
            lambda config: config["hard_gate_config"].update(
                {"max_selection_area_delta": 0.3}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                config, config_path, _ = build_preflight_fixture(Path(directory))
                mutate(config)
                write_json(config_path, config)
                with self.assertRaisesRegex(
                    VALIDATOR.ValidationError,
                    "candidate/replay budget|hard-gate contract",
                ):
                    VALIDATOR.run_preflight(config, config_path)

    def test_provenance_runtime_source_and_qwen_tampering_fail_closed(self):
        mutations = (
            (
                "manual semantic config SHA256 differs",
                lambda config, paths: paths["manual"].write_text(
                    '{}\n', encoding="utf-8"
                ),
            ),
            (
                "runtime source SHA256 differs",
                lambda config, paths: (
                    paths["repository"] / "change_agent/environment.py"
                ).write_text("tampered\n", encoding="utf-8"),
            ),
            (
                "config Qwen policy differs",
                lambda config, paths: config["execution_protocol"][
                    "qwen_policy"
                ].update({"allowed_call_count": 1}),
            ),
        )
        for pattern, mutate in mutations:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                config, config_path, paths = build_preflight_fixture(Path(directory))
                mutate(config, paths)
                write_json(config_path, config)
                with self.assertRaisesRegex(VALIDATOR.ValidationError, pattern):
                    VALIDATOR.run_preflight(config, config_path)

    def test_test_content_and_external_repository_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path, paths = build_preflight_fixture(Path(directory))
            selected = Path(
                config["samples"]["train_407_r3c2"]["source_parent"]["t1"]["path"]
            )
            forbidden = paths["dataset"] / "test_original" / "A" / "copied.png"
            forbidden.parent.mkdir(parents=True)
            forbidden.write_bytes(selected.read_bytes())
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "forbidden test split"
            ):
                VALIDATOR.run_preflight(config, config_path)

        with tempfile.TemporaryDirectory() as directory:
            config, config_path, paths = build_preflight_fixture(Path(directory))
            external = Path(paths["external"][0]["path"])
            (external / "tracked.py").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "tracked worktree is dirty"
            ):
                VALIDATOR.run_preflight(config, config_path)

    def test_preflight_cli_mode_and_atomic_receipt(self):
        receipt = {"schema_id": VALIDATOR.PREFLIGHT_SCHEMA, "status": "passed"}
        with (
            mock.patch.object(VALIDATOR, "read_json", return_value={"config": True}),
            mock.patch.object(VALIDATOR, "run_preflight", return_value=receipt) as run,
            mock.patch.object(VALIDATOR, "write_preflight_receipt") as write,
            mock.patch("builtins.print"),
        ):
            result = VALIDATOR.main(
                [
                    "--config",
                    "pilot.json",
                    "--preflight-only",
                    "--preflight-receipt",
                    "preflight.json",
                ]
            )
        self.assertEqual(result, 0)
        run.assert_called_once_with({"config": True}, Path("pilot.json"))
        write.assert_called_once_with(Path("preflight.json"), receipt)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            VALIDATOR.write_preflight_receipt(path, receipt)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), receipt)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "overwrite"):
                VALIDATOR.write_preflight_receipt(path, receipt)

    def test_zero_gpu_contract_invokes_and_binds_preflight_receipt(self):
        worker = (
            ROOT / "tools/run_qwen3vl_oracle_trajectory_contract.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("--preflight-only", worker)
        self.assertIn("--preflight-receipt \"$PREFLIGHT_RECEIPT\"", worker)
        self.assertIn(
            '"schema_id": "qwen3vl_oracle_trajectory_contract_receipt_v3"',
            worker,
        )
        self.assertIn('payload["preflight"] = preflight', worker)
        self.assertIn('"sha256": hashlib.sha256(preflight_bytes).hexdigest()', worker)
        self.assertIn('preflight["passed"]', worker)
        self.assertLess(
            worker.index("trap bootstrap_finish EXIT"),
            worker.index('ROOT="$(realpath -e'),
        )
        self.assertIn("bootstrap-contract-${BOOTSTRAP_JOB_ID}.json", worker)


class MetricContractTests(unittest.TestCase):
    def test_nonempty_and_empty_metric_edges_are_explicit(self):
        empty = np.zeros((2, 2), dtype=bool)
        nonempty = np.array([[1, 0], [0, 0]], dtype=bool)
        self.assertEqual(
            VALIDATOR.compute_metrics(empty, empty),
            {
                "iou": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pixel_error": 0,
                "predicted_pixels": 0,
                "gt_pixels": 0,
            },
        )
        empty_gt_error = VALIDATOR.compute_metrics(nonempty, empty)
        self.assertEqual((empty_gt_error["iou"], empty_gt_error["f1"]), (0.0, 0.0))
        self.assertEqual((empty_gt_error["fp"], empty_gt_error["pixel_error"]), (1, 1))

    def test_wilson_rate_preserves_counts_and_zero_denominator(self):
        rate = VALIDATOR.wilson_rate(4, 5)
        self.assertEqual((rate["numerator"], rate["denominator"], rate["rate"]), (4, 5, 0.8))
        self.assertLess(rate["wilson95_lower"], rate["rate"])
        self.assertGreater(rate["wilson95_upper"], rate["rate"])
        self.assertEqual(
            VALIDATOR.wilson_rate(0, 0),
            {
                "numerator": 0,
                "denominator": 0,
                "rate": None,
                "wilson95_lower": None,
                "wilson95_upper": None,
            },
        )

    def test_metric_delta_tolerates_float_representation_but_returns_recompute(self):
        before = {
            "pixel_error": 9,
            "iou": 1 / 7,
            "f1": 1 / 4,
            "tp": 2,
            "fp": 3,
            "fn": 6,
        }
        after = {
            "pixel_error": 4,
            "iou": 2 / 3,
            "f1": 4 / 5,
            "tp": 6,
            "fp": 1,
            "fn": 3,
        }
        expected = VALIDATOR._metric_delta(before, after)
        recorded = copy.deepcopy(expected)
        recorded["f1"] = math.nextafter(float(recorded["f1"]), math.inf)
        self.assertNotEqual(recorded, expected)
        self.assertEqual(
            VALIDATOR._require_metric_delta(
                recorded, before, after, "equivalent runner representation"
            ),
            expected,
        )


class EpisodeValidationTests(unittest.TestCase):
    def test_nonempty_success_recomputes_metrics_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = validate_fixture(root, valid_episode(root), 2)
        self.assertEqual(result.status, "endpoint_success")
        self.assertEqual(result.success_step, 1)
        self.assertEqual(result.initial_metrics["pixel_error"], 2)
        self.assertEqual(result.final_metrics["pixel_error"], 0)
        self.assertAlmostEqual(result.final_metrics["iou"], 1.0)
        self.assertTrue(result.exact_recovery)

    def test_empty_gt_uses_strict_fp_decrease_not_unchanged_iou_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = validate_fixture(root, valid_episode(root, empty_gt=True), 0)
        self.assertEqual(result.status, "locally_reachable_only")
        self.assertIsNone(result.success_step)
        self.assertEqual(result.initial_metrics["fp"], 2)
        self.assertEqual(result.final_metrics["fp"], 1)
        self.assertEqual(result.initial_metrics["iou"], 0.0)
        self.assertEqual(result.final_metrics["iou"], 0.0)
        self.assertEqual(result.termination_reason, "no_beneficial")

    def test_nonexact_endpoint_reached_reason_is_independently_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = validate_fixture(
                root,
                valid_episode(
                    root, empty_gt=True, empty_gt_max_fp=1
                ),
                0,
                empty_gt_max_fp=1,
            )
        self.assertEqual(result.status, "endpoint_success")
        self.assertEqual(result.termination_reason, "endpoint_reached")
        self.assertFalse(result.exact_recovery)

    def test_termination_reason_is_uniquely_derived_from_round_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root, empty_gt=True)
            episode["termination"]["reason"] = "max_steps"
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "termination receipt differs"
            ):
                validate_fixture(root, episode, 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root, empty_gt=True)
            episode["selection_rounds"].pop()
            episode["gt_access_audit"]["catalog_freezes"].pop()
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError,
                "neither independently no-beneficial nor max_steps",
            ):
                validate_fixture(root, episode, 0)

    def test_already_exact_generates_no_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = validate_fixture(
                root, valid_episode(root, already_exact=True), 2
            )
        self.assertEqual(result.status, "already_exact")
        self.assertTrue(result.exact_recovery)
        self.assertEqual(result.transitions, [])

    def test_already_endpoint_is_not_trajectory_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = validate_fixture(
                root,
                valid_episode(
                    root,
                    empty_gt=True,
                    already_endpoint=True,
                    empty_gt_max_fp=64,
                ),
                0,
                empty_gt_max_fp=64,
            )
        self.assertEqual(result.status, "already_endpoint")
        self.assertFalse(result.exact_recovery)
        self.assertIsNone(result.success_step)
        self.assertEqual(result.transitions, [])

    def test_multiple_actions_and_more_than_three_steps_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root)
            episode["steps"][0]["actions"] = [
                episode["steps"][0]["action"],
                episode["steps"][0]["action"],
            ]
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "exactly one action"
            ):
                validate_fixture(root, episode, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root)
            episode["steps"] = episode["steps"] * 4
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "max_steps=3"):
                validate_fixture(root, episode, 2)

    def test_state_continuity_and_three_mask_replay_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root)
            episode["steps"][0]["source_state_sha256"] = "f" * 64
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "source evidence"):
                validate_fixture(root, episode, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root)
            wrong = np.zeros((4, 4), dtype=bool)
            wrong[1, 1] = True
            episode["steps"][0]["replay_state"] = write_state(root, "wrong", wrong)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "round/replay"):
                validate_fixture(root, episode, 2)

    def test_gate_semantic_qwen_and_monotonic_failures_are_rejected(self):
        mutations = (
            ("hard[- ]gate", lambda value: value["steps"][0]["hard_gate"].update({"passed": False, "rejected": True})),
            ("RGB[- ]semantic", lambda value: value["steps"][0]["rgb_semantic"].update({"valid": False})),
            ("qwen_invoked", lambda value: value["steps"][0].update({"qwen_invoked": True})),
        )
        for message, mutate in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode = valid_episode(root)
                mutate(episode)
                with self.assertRaisesRegex(VALIDATOR.ValidationError, message):
                    validate_fixture(root, episode, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root, empty_gt=True)
            unchanged = np.zeros((4, 4), dtype=bool)
            unchanged[0, :2] = True
            for key, prefix in (
                ("candidate_state", "unchanged/candidate"),
                ("canonical_state", "unchanged/canonical"),
                ("replay_state", "unchanged/replay"),
            ):
                episode["steps"][0][key] = write_state(root, prefix, unchanged)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "selected/candidate"):
                validate_fixture(root, episode, 0)

    def test_episode_source_hashes_must_exactly_match_config(self):
        expected = {"t1": "1" * 64, "t2": "2" * 64, "label": "3" * 64}
        mutations = {
            "missing": lambda value: value.pop("label"),
            "extra": lambda value: value.update({"unexpected": "4" * 64}),
            "tampered": lambda value: value.update({"label": "4" * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode = valid_episode(root)
                mutate(episode["source_hashes"])
                with self.assertRaisesRegex(
                    VALIDATOR.ValidationError,
                    "source_hashes (inventory differs|differ from config)",
                ):
                    VALIDATOR.validate_episode(
                        episode,
                        root=root,
                        expected_sample=expected_sample(2),
                        epsilon=1e-12,
                        minimum_reduction=0.05,
                        source_episode_sha256="a" * 64,
                        expected_source_hashes=expected,
                    )


class IndependentEvidenceTests(unittest.TestCase):
    def assert_tamper_rejected(self, pattern, mutate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root)
            mutate(root, episode)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, pattern):
                validate_fixture(root, episode, 2)

    def test_initializer_gt_audit_batch_and_selection_tampering_fail_closed(self):
        mutations = (
            (
                "fresh dual-view SAM3",
                lambda root, value: value["initial_state"][
                    "initializer_receipt"
                ]["worker_report"].update({"status": "failed"}),
            ),
            (
                "GT access count",
                lambda root, value: value["gt_access_audit"].update(
                    {"access_count": 2}
                ),
            ),
            (
                "candidate batch is incomplete",
                lambda root, value: value["selection_rounds"][0].update(
                    {"candidate_batch_complete_before_final_gt_scoring": False}
                ),
            ),
            (
                r"beneficial_shortlist\[0\]",
                lambda root, value: value["selection_rounds"][0].update(
                    {"selected_candidate_id": "tampered-candidate"}
                ),
            ),
            (
                "qwen_calls must be zero",
                lambda root, value: value["selection_rounds"][0]["candidates"][
                    0
                ]["no_qwen_audit"].update({"qwen_calls": 1}),
            ),
        )
        for pattern, mutate in mutations:
            with self.subTest(pattern=pattern):
                self.assert_tamper_rejected(pattern, mutate)

    def test_catalog_gt_field_and_candidate_tool_receipt_tampering_fail_closed(self):
        def inject_gt_field(root, episode):
            identity = episode["selection_rounds"][0]["gt_free_catalog_freeze"]
            path = root / identity["path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["live_proposals"][0]["gt_mask"] = "forbidden"
            for action in payload["actions"]:
                if action["region_id"] == payload["live_proposals"][0]["region_id"]:
                    action["proposal"]["gt_mask"] = "forbidden"
            payload["live_proposals_canonical_sha256"] = (
                VALIDATOR.canonical_sha256(payload["live_proposals"])
            )
            payload["action_catalog_canonical_sha256"] = (
                VALIDATOR.canonical_sha256(payload["actions"])
            )
            write_json(path, payload)
            identity.update(
                {
                    "sha256": sha256(path),
                    "canonical_sha256": VALIDATOR.canonical_sha256(payload),
                    "size_bytes": path.stat().st_size,
                }
            )

        self.assert_tamper_rejected("forbidden GT-derived field", inject_gt_field)
        self.assert_tamper_rejected(
            "worker receipt is not successful",
            lambda root, value: value["selection_rounds"][0]["candidates"][0][
                "tool_receipt"
            ]["tool_result"]["worker_report"].update({"status": "failed"}),
        )

    def test_action_inventory_top4_and_unselected_rgb_are_independently_rebuilt(self):
        def mutate_inventory(root, episode):
            round_record = episode["selection_rounds"][0]
            identity = round_record["gt_free_catalog_freeze"]
            path = root / identity["path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["actions"][0]["reachability_tier"] = "tampered-tier"
            payload["action_catalog_canonical_sha256"] = (
                VALIDATOR.canonical_sha256(payload["actions"])
            )
            write_json(path, payload)
            identity.update(
                {
                    "sha256": sha256(path),
                    "canonical_sha256": VALIDATOR.canonical_sha256(payload),
                    "size_bytes": path.stat().st_size,
                }
            )
            round_record["action_catalog"][0][
                "reachability_tier"
            ] = "tampered-tier"

        self.assert_tamper_rejected(
            "frozen action inventory differs from useful executor policy",
            mutate_inventory,
        )
        self.assert_tamper_rejected(
            "proposal-box error pixels differ",
            lambda root, value: value["selection_rounds"][0][
                "action_catalog"
            ][0].update({"proposal_box_current_change_error_pixels": 999}),
        )
        self.assert_tamper_rejected(
            "execution shortlist order differs from independent top-4",
            lambda root, value: value["selection_rounds"][0].update(
                {
                    "execution_shortlist": list(
                        reversed(
                            value["selection_rounds"][0][
                                "execution_shortlist"
                            ]
                        )
                    )
                }
            ),
        )

        def mutate_unselected_rgb(root, episode):
            del root
            catalog = episode["selection_rounds"][0]["action_catalog"]
            unselected = next(
                item
                for item in catalog
                if item["selected_for_execution"] is False
            )
            unselected["rgb_semantic_gate"].update(
                {"passed": True, "observed_rgb_state": "building"}
            )

        self.assert_tamper_rejected(
            "RGB-semantic receipt differs from config",
            mutate_unselected_rgb,
        )

    def test_independent_inventory_never_adds_scan_negative_on_white_drift(self):
        actions = VALIDATOR._rebuild_useful_action_inventory(
            "episode",
            1,
            [
                {
                    "region_id": "scan",
                    "audit_kind": "scan",
                    "component_seed_pixels": [1, 1],
                    "box_pixels": [0, 0, 2, 2],
                    "component_seed_t1_mask_white": True,
                    "component_seed_t2_mask_white": True,
                }
            ],
            "fixture",
        )
        self.assertEqual(
            [
                (item["action"]["action"], item["action"]["target_view"])
                for item in actions
            ],
            [
                ("positive_point", "t1"),
                ("box", "t1"),
                ("positive_point", "t2"),
                ("box", "t2"),
            ],
        )
        self.assertTrue(
            all(item["action"]["coordinate_space"] == "pixel" for item in actions)
        )

    def test_runner_and_validator_action_inventories_serialize_exactly_alike(self):
        proposals = [
            {
                "region_id": "present",
                "audit_kind": "present",
                "component_seed_pixels": [3, 4],
                "box_pixels": [1, 2, 7, 8],
                "component_seed_t1_mask_white": True,
                "component_seed_t2_mask_white": False,
            },
            {
                "region_id": "missing",
                "audit_kind": "missing",
                "component_seed_pixels": [5, 6],
                "box_pixels": [2, 3, 9, 10],
                "component_seed_t1_mask_white": False,
                "component_seed_t2_mask_white": True,
            },
            {
                "region_id": "mixed",
                "audit_kind": "mixed",
                "component_seed_pixels": [7, 8],
                "box_pixels": [3, 4, 11, 12],
                "component_seed_t1_mask_white": True,
                "component_seed_t2_mask_white": True,
            },
            {
                "region_id": "scan",
                "audit_kind": "scan",
                "component_seed_pixels": [9, 10],
                "box_pixels": [4, 5, 13, 14],
                "component_seed_t1_mask_white": True,
                "component_seed_t2_mask_white": True,
            },
        ]
        runner_actions = RUNNER.enumerate_live_actions("episode", 2, proposals)
        serialized_runner = [
            {
                "candidate_id": item.candidate_id,
                "region_id": item.region_id,
                "reachability_tier": item.reachability_tier,
                "action": item.action.to_dict(),
                "proposal": dict(item.proposal),
            }
            for item in runner_actions
        ]
        rebuilt = VALIDATOR._rebuild_useful_action_inventory(
            "episode", 2, proposals, "runner-validator parity"
        )
        self.assertEqual(rebuilt, serialized_runner)
        for item in rebuilt:
            action = item["action"]
            geometry_key = (
                "box" if action["action"] == "box" else "coordinate"
            )
            self.assertEqual(
                set(action),
                {"target_view", "action", "coordinate_space", geometry_key},
            )
            self.assertEqual(action["coordinate_space"], "pixel")

    def test_independent_inventory_rejects_unknown_audit_kind(self):
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "unsupported audit_kind"
        ):
            VALIDATOR._rebuild_useful_action_inventory(
                "episode",
                1,
                [
                    {
                        "region_id": "unknown",
                        "audit_kind": "future_kind",
                        "component_seed_pixels": [1, 1],
                        "box_pixels": [0, 0, 2, 2],
                        "component_seed_t1_mask_white": False,
                        "component_seed_t2_mask_white": False,
                    }
                ],
                "fixture",
            )

    def test_coordinated_hard_gate_flag_and_receipt_forgery_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_mask = np.zeros((16, 16), dtype=bool)
            candidate_mask = np.zeros((16, 16), dtype=bool)
            candidate_mask[15, 15] = True
            source = VALIDATOR._load_state(
                root,
                write_state(root, "hard/source", source_mask),
                "hard.source",
            )
            candidate = VALIDATOR._load_state(
                root,
                write_state(root, "hard/candidate", candidate_mask),
                "hard.candidate",
            )
            action = {
                "action": "positive_point",
                "target_view": "t2",
                "coordinate": [0, 0],
            }
            proposal = {
                "region_id": "r0",
                "audit_kind": "missing",
                "component_area": 4,
                "component_seed_pixels": [0, 0],
                "box_pixels": [0, 0, 1, 1],
                "component_seed_t1_mask_white": False,
                "component_seed_t2_mask_white": False,
            }
            forged_gate = {
                "passed": True,
                "rejected": False,
                "rejection_reasons": [],
                "verifier_skipped_by_hard_gate": False,
            }
            forged_execution = {
                "candidate_accepted": True,
                "candidate_rejection_reasons": [],
                "verifier_skipped_by_hard_gate": False,
            }
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError,
                "hard-gate receipt differs from independent recomputation",
            ):
                VALIDATOR._validate_hard_gate(
                    forged_gate,
                    forged_execution,
                    "forged",
                    source=source,
                    candidate=candidate,
                    action=action,
                    proposal=proposal,
                )

    def test_non_target_and_rebuilt_change_mask_tampering_fail_closed(self):
        def change_non_target(root, episode):
            change = np.zeros((4, 4), dtype=bool)
            change[0, 0] = True
            wrong_t1 = np.zeros((4, 4), dtype=bool)
            wrong_t1[3, 3] = True
            wrong = write_state(
                root,
                "tampered/non-target",
                change,
                t1=wrong_t1,
                t2=change,
            )
            episode["selection_rounds"][0]["candidates"][0][
                "candidate_state"
            ] = wrong
            episode["steps"][0]["candidate_state"] = wrong

        def change_rebuild(root, episode):
            t2 = np.zeros((4, 4), dtype=bool)
            t2[0, 0] = True
            wrong = write_state(
                root,
                "tampered/rebuild",
                np.zeros((4, 4), dtype=bool),
                t2=t2,
            )
            episode["selection_rounds"][0]["candidates"][0][
                "candidate_state"
            ] = wrong
            episode["steps"][0]["candidate_state"] = wrong

        self.assert_tamper_rejected("non-target view mask changed", change_non_target)
        self.assert_tamper_rejected("MaskPairProcessor rebuild", change_rebuild)

    def test_point_history_and_canonical_session_rebase_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = valid_episode(root, empty_gt=True)
            episode["selection_rounds"][0]["candidates"][0]["tool_receipt"][
                "tool_input"
            ]["accepted_click_history"] = [
                {"coordinate": [1, 1], "is_positive": True}
            ]
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "point tool input differs"
            ):
                validate_fixture(root, episode, 0)

        self.assert_tamper_rejected(
            "accepted-mask session rebase differs",
            lambda root, value: value["selection_rounds"][0][
                "canonical_replay"
            ]["runtime_execution"]["point_session_update"].update(
                {"base_mask_pixels": 999}
            ),
        )

    def test_final_state_termination_and_budget_receipts_fail_closed(self):
        mutations = (
            (
                "final state identity differs",
                lambda root, value: value.update(
                    {"final_state_sha256": "0" * 64}
                ),
            ),
            (
                "final state identity differs",
                lambda root, value: value["final_state_mask_sha256"].update(
                    {"change": "0" * 64}
                ),
            ),
            (
                "termination receipt differs",
                lambda root, value: value["termination"].update(
                    {"candidate_count": 999}
                ),
            ),
            (
                "termination receipt differs",
                lambda root, value: value["termination"].update(
                    {"reason": "max_steps"}
                ),
            ),
            (
                "budget receipt differs",
                lambda root, value: value["budget"].update(
                    {"tool_execution_count": 999}
                ),
            ),
            (
                "budget receipt differs",
                lambda root, value: value["budget"].update(
                    {"distinct_candidate_count": 999}
                ),
            ),
            (
                "initial_state.metrics.pixel_error differs",
                lambda root, value: value["initial_state"]["metrics"].update(
                    {"pixel_error": 999}
                ),
            ),
        )
        for pattern, mutate in mutations:
            with self.subTest(pattern=pattern):
                self.assert_tamper_rejected(pattern, mutate)


class DerivationAndReportTests(unittest.TestCase):
    def test_false_reject_diagnostic_counts_metric_benefit_and_reason(self):
        diagnostics = VALIDATOR._episode_diagnostics(
            [
                {
                    "action_catalog": [],
                    "beneficial_shortlist": [],
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "action": {
                                "action": "box",
                                "target_view": "t2",
                                "box": [0, 0, 1, 1],
                            },
                            "rgb_semantic_gate": {"passed": True},
                            "hard_gate": {
                                "rejected": True,
                                "rejection_reasons": [
                                    "locality_outside_roi_exceeded"
                                ],
                            },
                            "per_step_metric_gate": {"passed": True},
                        }
                    ],
                    "selected_candidate_id": None,
                }
            ]
        )
        self.assertEqual(
            diagnostics["hard_gate_false_reject"],
            {"would_improve_but_hard_rejected": 1},
        )
        self.assertEqual(
            diagnostics["hard_gate_false_reject_reasons"],
            {"locality_outside_roi_exceeded": 1},
        )
        self.assertEqual(diagnostics["candidate_metric_gate"], {"passed": 1})
        self.assertEqual(
            diagnostics["candidate_beneficial"], {"not_beneficial": 1}
        )

    def test_only_endpoint_success_produces_rich_bijective_matched_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = validate_fixture(root, valid_episode(root), 2)
            exact_root = root / "exact"
            exact_root.mkdir()
            exact = validate_fixture(
                exact_root, valid_episode(exact_root, already_exact=True), 2
            )
            c_iso, t_traj, manifest = VALIDATOR.derive_matched_data(
                [success, exact], global_release_passed=False
            )
        self.assertEqual(len(c_iso), 1)
        self.assertEqual(len(t_traj), 1)
        self.assertEqual(c_iso[0]["transition_id"], t_traj[0]["transitions"][0]["transition_id"])
        self.assertIn("source_context", c_iso[0])
        self.assertIn("accepted_step_evidence", c_iso[0])
        self.assertIn("selection_round_evidence", c_iso[0])
        for state_field in (
            "source_state",
            "candidate_state",
            "next_state",
            "replay_state",
        ):
            self.assertIn(state_field, c_iso[0])
            self.assertIn("masks", c_iso[0][state_field])
        self.assertEqual(c_iso[0]["source_context"]["query"], "building")
        self.assertEqual(
            c_iso[0]["source_context"]["rgb_semantics"]["default"],
            {"t1": "background", "t2": "building"},
        )
        self.assertEqual(
            c_iso[0]["source_context"], t_traj[0]["source_context"]
        )
        self.assertEqual(manifest["endpoint_success_episode_count"], 1)
        self.assertTrue(manifest["bijection_100_percent"])
        self.assertEqual(manifest["failed_episode_records_written"], 0)
        self.assertFalse(manifest["training_authorized"])
        self.assertEqual(manifest["authorized_splits"], [])
        self.assertTrue(all(not item["training_authorized"] for item in c_iso))
        self.assertTrue(all(not item["training_authorized"] for item in t_traj))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_only = validate_fixture(
                root, valid_episode(root, endpoint_success=False), 2
            )
            c_iso, t_traj, manifest = VALIDATOR.derive_matched_data(
                [local_only], global_release_passed=False
            )
        self.assertEqual((c_iso, t_traj), ([], []))
        self.assertEqual(manifest["endpoint_success_episode_count"], 0)

    def test_report_has_all_required_rates_with_wilson_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = validate_fixture(root, valid_episode(root), 2)
            report = VALIDATOR.build_report([success])
        self.assertEqual(report["failure_taxonomy"]["endpoint_success"], 1)
        for name in (
            "endpoint_success_at_1",
            "endpoint_success_at_2",
            "endpoint_success_at_3",
            "final_endpoint_reachability",
            "exact_recovery",
            "tool_success",
            "hard_gate_pass",
            "rgb_semantic_valid",
            "monotonic_step",
            "selected_action_replay",
        ):
            with self.subTest(name=name):
                rate = report["rates"][name]
                self.assertIn("numerator", rate)
                self.assertIn("denominator", rate)
                self.assertIn("wilson95_lower", rate)
                self.assertIn("wilson95_upper", rate)
        self.assertFalse(report["global_release_gate"]["passed"])
        self.assertEqual(report["termination_reason_distribution"], {"exact_recovery": 1})
        diagnostics = report["candidate_diagnostics"]
        self.assertGreater(diagnostics["candidate_metric_gate"]["passed"], 0)
        self.assertGreater(diagnostics["candidate_beneficial"]["beneficial"], 0)
        self.assertEqual(diagnostics["hard_gate_false_reject"], {})
        self.assertEqual(diagnostics["selected_replay_tools"], {"SAM3": 1})

    def test_six_of_six_endpoint_release_gate_passes(self):
        episodes = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                episode_root = root / str(index)
                episode_root.mkdir()
                episodes.append(
                    validate_fixture(
                        episode_root,
                        valid_episode(
                            episode_root,
                            sample_id=f"sample-{index}",
                            parent_id=f"train-{index}",
                        ),
                        2,
                    )
                )
        report = VALIDATOR.build_report(episodes)
        self.assertTrue(report["global_release_gate"]["passed"])
        self.assertTrue(
            report["global_release_gate"][
                "proceed_to_next_training_data_iteration"
            ]
        )

    def test_six_of_six_authorizes_only_train_records_and_never_validation(self):
        episodes = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                split = "train" if index < 3 else "val"
                episode_root = root / str(index)
                episode_root.mkdir()
                episodes.append(
                    validate_fixture(
                        episode_root,
                        valid_episode(
                            episode_root,
                            sample_id=f"sample-{index}",
                            parent_id=f"{split}-{index}",
                            split=split,
                        ),
                        2,
                    )
                )
        report = VALIDATOR.build_report(episodes)
        self.assertTrue(report["training_authorized"])
        c_iso, t_traj, manifest = VALIDATOR.derive_matched_data(
            episodes, global_release_passed=report["global_release_gate"]["passed"]
        )
        for record in c_iso:
            self.assertIs(
                record["training_authorized"], record["split"] == "train"
            )
        for record in t_traj:
            expected = record["split"] == "train"
            self.assertIs(record["training_authorized"], expected)
            self.assertTrue(
                all(
                    step["training_authorized"] is expected
                    for step in record["transitions"]
                )
            )
        self.assertTrue(manifest["training_authorized"])
        self.assertEqual(manifest["authorized_splits"], ["train"])
        self.assertTrue(
            manifest["split_artifacts"]["train"]["training_authorized"]
        )
        self.assertFalse(
            manifest["split_artifacts"]["val"]["training_authorized"]
        )
        self.assertTrue(
            all(
                item["training_authorized"] is (item["split"] == "train")
                for item in manifest["transitions"]
            )
        )

    def test_no_go_with_successful_diagnostic_episode_authorizes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = validate_fixture(root, valid_episode(root), 2)
        report = VALIDATOR.build_report([success])
        c_iso, t_traj, manifest = VALIDATOR.derive_matched_data(
            [success],
            global_release_passed=report["global_release_gate"]["passed"],
        )
        self.assertFalse(report["training_authorized"])
        self.assertTrue(c_iso)
        self.assertTrue(t_traj)
        self.assertTrue(all(not item["training_authorized"] for item in c_iso))
        self.assertTrue(all(not item["training_authorized"] for item in t_traj))
        self.assertFalse(manifest["training_authorized"])
        self.assertEqual(manifest["authorized_splits"], [])

    def test_writer_rejects_training_authorization_tampering_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_root = root / "episode"
            episode_root.mkdir()
            success = validate_fixture(
                episode_root, valid_episode(episode_root), 2
            )
            base_report = VALIDATOR.build_report([success])
            base_c_iso, base_t_traj, base_matching = (
                VALIDATOR.derive_matched_data(
                    [success], global_release_passed=False
                )
            )
            bind_matching_to_report(base_report, base_matching)
            mutations = (
                lambda report, c_iso, t_traj, matching: c_iso[0].update(
                    {"training_authorized": True}
                ),
                lambda report, c_iso, t_traj, matching: t_traj[0].update(
                    {"training_authorized": True}
                ),
                lambda report, c_iso, t_traj, matching: matching[
                    "split_artifacts"
                ]["train"].update({"training_authorized": True}),
                lambda report, c_iso, t_traj, matching: report.update(
                    {"training_authorized": True}
                ),
            )
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    report = copy.deepcopy(base_report)
                    c_iso = copy.deepcopy(base_c_iso)
                    t_traj = copy.deepcopy(base_t_traj)
                    matching = copy.deepcopy(base_matching)
                    mutate(report, c_iso, t_traj, matching)
                    output = root / f"tampered-{index}"
                    with self.assertRaisesRegex(
                        VALIDATOR.ValidationError, "authorization"
                    ):
                        VALIDATOR.write_outputs(
                            output, report, c_iso, t_traj, matching
                        )
                    self.assertFalse(output.exists())

    def test_exact_recovery_rate_excludes_already_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success = validate_fixture(root, valid_episode(root), 2)
            exact_root = root / "exact"
            exact_root.mkdir()
            already_exact = validate_fixture(
                exact_root, valid_episode(exact_root, already_exact=True), 2
            )
            report = VALIDATOR.build_report([success, already_exact])
        self.assertEqual(
            report["rates"]["exact_recovery"],
            VALIDATOR.wilson_rate(1, 1),
        )

    def test_atomic_writer_refuses_overwrite_and_writes_required_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            report = VALIDATOR.build_report([])
            c_iso, t_traj, matching = VALIDATOR.derive_matched_data(
                [], global_release_passed=False
            )
            bind_matching_to_report(report, matching)
            VALIDATOR.write_outputs(
                output,
                report,
                c_iso,
                t_traj,
                matching,
            )
            self.assertTrue((output / "validation_report.json").is_file())
            self.assertTrue((output / "data/train/c_iso.jsonl").is_file())
            self.assertTrue((output / "data/train/t_traj.jsonl").is_file())
            self.assertTrue((output / "data/val/c_iso.jsonl").is_file())
            self.assertTrue((output / "data/val/t_traj.jsonl").is_file())
            self.assertFalse((output / "data/c_iso.jsonl").exists())
            self.assertFalse((output / "data/t_traj.jsonl").exists())
            self.assertTrue((output / "data/matching_manifest.json").is_file())
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "overwrite"):
                VALIDATOR.write_outputs(
                    output, report, c_iso, t_traj, matching
                )


class SourceAndJsonIntegrityTests(unittest.TestCase):
    def test_duplicate_json_keys_and_mask_hash_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "duplicate.json"
            path.write_text('{"schema_id":"x","schema_id":"y"}\n', encoding="utf-8")
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "duplicate JSON key"):
                VALIDATOR.read_json(path)
            episode = valid_episode(root)
            episode["gt_mask"]["mask_sha256"] = "0" * 64
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "semantic SHA256"):
                validate_fixture(root, episode, 2)

    def test_source_hashes_and_test_paths_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png", "label.png"):
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(root / name)
            sample = {
                "source_parent": {
                    "t1": {"path": str(root / "a.png"), "sha256": sha256(root / "a.png")},
                    "t2": {"path": str(root / "b.png"), "sha256": sha256(root / "b.png")},
                    "label": {"path": str(root / "label.png"), "sha256": sha256(root / "label.png")},
                }
            }
            self.assertEqual(set(VALIDATOR._source_parent_hashes(sample)), {"t1", "t2", "label"})
            sample["source_parent"]["label"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "SHA256 differs"):
                VALIDATOR._source_parent_hashes(sample)
            for forbidden in ("test", "test_256", "test_original"):
                with self.subTest(forbidden=forbidden):
                    forbidden_sample = copy.deepcopy(sample)
                    forbidden_sample["source_parent"]["label"] = {
                        "path": f"/forbidden/{forbidden}/label.png",
                        "sha256": "0" * 64,
                    }
                    with self.assertRaisesRegex(
                        VALIDATOR.ValidationError, "test source"
                    ):
                        VALIDATOR._source_parent_hashes(forbidden_sample)


class ConfigAndShardContractTests(unittest.TestCase):
    def test_config_accepts_exact_commit_freeze_status_only_from_allowlist(self):
        contract = VALIDATOR._validate_config(
            minimal_config(status="frozen_pending_exact_commit_validation")
        )
        self.assertEqual(
            (contract.epsilon, contract.minimum_local_reduction),
            (1e-12, 0.05),
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "pilot config status invalid"
        ):
            VALIDATOR._validate_config(minimal_config(status="frozen_and_done"))

    def test_config_rejects_missing_or_wrong_frozen_runtime_fields(self):
        mutations = (
            lambda value: value["candidate_and_replay_budget"].pop(
                "max_candidate_actions_per_step"
            ),
            lambda value: value["candidate_and_replay_budget"].update(
                {"max_total_tool_executions_including_selected_action_replay_per_sample": 23}
            ),
            lambda value: value["hard_gate_config"].update(
                {"max_locality_outside_ratio": 0.2}
            ),
            lambda value: value["resource_plan"]["shards"][0].update(
                {"samples": ["train_407_r3c2", "val_41_r3c2"]}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                config = minimal_config()
                mutate(config)
                with self.assertRaises(VALIDATOR.ValidationError):
                    VALIDATOR._validate_config(config)

    def test_three_shards_are_reordered_by_config_not_input_root_order(self):
        ordered_samples = [
            "train_407_r3c2",
            "train_114_r3c1",
            "train_364_r0c2",
            "val_43_r1c1",
            "val_41_r3c2",
            "val_60_r3c0",
        ]
        shard_samples = [
            ["train_407_r3c2", "val_43_r1c1"],
            ["train_114_r3c1", "val_41_r3c2"],
            ["train_364_r0c2", "val_60_r3c0"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_sha256, shard_roots = build_sharded_fixture(
                root, ordered_samples, shard_samples
            )
            report, c_iso, t_traj, matching = VALIDATOR.validate_inputs(
                config,
                [shard_roots[2], shard_roots[0], shard_roots[1]],
                config_raw_sha256=config_sha256,
            )

        self.assertEqual(
            [item["sample_id"] for item in report["episodes"]], ordered_samples
        )
        self.assertEqual([item["sample_id"] for item in c_iso], ordered_samples)
        self.assertEqual([item["sample_id"] for item in t_traj], ordered_samples)
        self.assertEqual(
            [item["episode_id"] for item in matching["transitions"]],
            ordered_samples,
        )
        self.assertEqual(report["bindings"]["runtime_commit"], "a" * 40)
        self.assertEqual(report["bindings"]["config"]["sha256"], config_sha256)
        self.assertEqual(
            report["bindings"]["runner"]["sha256"],
            sha256(ROOT / "tools/run_qwen3vl_oracle_trajectory.py"),
        )

    def test_config_runner_and_cross_shard_commit_tampering_fail_closed(self):
        ordered = [
            "train_407_r3c2",
            "train_114_r3c1",
            "train_364_r0c2",
            "val_43_r1c1",
            "val_41_r3c2",
            "val_60_r3c0",
        ]
        shards = [
            [ordered[0], ordered[3]],
            [ordered[1], ordered[4]],
            [ordered[2], ordered[5]],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_sha256, shard_roots = build_sharded_fixture(
                root,
                ordered,
                shards,
                runtime_commits=["a" * 40, "b" * 40, "a" * 40],
            )
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "runtime commit differs across shards"
            ):
                VALIDATOR.validate_inputs(
                    config,
                    shard_roots,
                    config_raw_sha256=config_sha256,
                )

        tampering = (
            ("config binding differs", ("config", "sha256")),
            ("runner SHA256 differs", ("runner", "sha256")),
        )
        for pattern, keys in tampering:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, config_sha256, shard_roots = build_sharded_fixture(
                    root, ordered, shards
                )
                index_path = shard_roots[0] / "shard_index.json"
                index = json.loads(index_path.read_text(encoding="utf-8"))
                index[keys[0]][keys[1]] = "0" * 64
                write_json(index_path, index)
                with self.assertRaisesRegex(VALIDATOR.ValidationError, pattern):
                    VALIDATOR.validate_inputs(
                        config,
                        shard_roots,
                        config_raw_sha256=config_sha256,
                    )

    def test_main_validates_inputs_once(self):
        config = minimal_config()
        validated = ({"report": True}, [], [], {"matching": True})
        with (
            mock.patch.object(VALIDATOR, "read_json", return_value=config),
            mock.patch.object(VALIDATOR, "sha256_file", return_value="f" * 64),
            mock.patch.object(
                VALIDATOR, "validate_inputs", return_value=validated
            ) as validate_inputs,
            mock.patch.object(VALIDATOR, "write_outputs") as write_outputs,
            mock.patch("builtins.print"),
        ):
            result = VALIDATOR.main(
                [
                    "--config",
                    "pilot.json",
                    "--input-root",
                    "shard-0",
                    "--output",
                    "validated",
                ]
            )
        self.assertEqual(result, 0)
        validate_inputs.assert_called_once_with(
            config, [Path("shard-0")], config_raw_sha256="f" * 64
        )
        write_outputs.assert_called_once_with(Path("validated"), *validated)


if __name__ == "__main__":
    unittest.main()
