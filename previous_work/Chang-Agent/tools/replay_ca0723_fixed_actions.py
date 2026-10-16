#!/usr/bin/env python3
"""Three-way SimpleClick replay of frozen CA_0723(5) actions.

The replay has no planner, Verifier, ratio gate, or fresh SAM initialization.
It first completes and audits every SimpleClick call, writes
``replay_complete.json`` with ``gt_loaded=false``, and only then opens LEVIR
ground truth for offline metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor
from change_agent.adapters.subprocess_adapters import SubprocessPointBackend
from change_agent.trajectory import default_run_metadata, mask_sha256


SCHEMA_VERSION = "ca0723_fixed_action_replay_v1"
MANIFEST_SCHEMA_VERSION = "ca0723_fixed_action_manifest_v1"
ARMS = (
    "legacy_session_legacy_write",
    "component_input_component_write",
    "legacy_session_component_write",
)
LEGACY_ARM, COMPONENT_ARM, HYBRID_ARM = ARMS


@dataclass(frozen=True)
class MaskState:
    t1: np.ndarray
    t2: np.ndarray
    change: np.ndarray


@dataclass(frozen=True)
class SourceSample:
    name: str
    expected: dict[str, Any]
    trajectory_path: Path
    images: dict[str, np.ndarray]
    initial_masks: dict[str, np.ndarray]
    processor: MaskPairProcessor


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Replay fixed CA_0723 actions under three input/write scopes."
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=(
            root
            / "outputs/CA_0723_archive"
            / "CA_0723(5)-bailian-proposal-atomic-v11-5sample"
            / "qwen37-no-thinking"
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256"
        ),
    )
    parser.add_argument(
        "--fixed-action-manifest",
        type=Path,
        default=Path(__file__).with_name("ca0723_fixed_action_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--segagent-python",
        type=Path,
        default=Path("/guisongxia01/pangchao/wangyihan/segagent-env/bin/python"),
    )
    parser.add_argument(
        "--simpleclick-checkpoint",
        type=Path,
        default=Path(
            "/guisongxia01/pangchao/wangyihan/models/SimpleClick/"
            "cocolvis_vit_large.pth"
        ),
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("seeded_segmentation_worker.py"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    incomplete = output.with_name(f"{output.name}.incomplete")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if incomplete.exists():
        raise FileExistsError(
            f"refusing to overwrite existing incomplete replay: {incomplete}"
        )
    _validate_required_paths(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir(parents=False, exist_ok=False)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        (incomplete / ".slurm_job_id").write_text(
            slurm_job_id + "\n", encoding="utf-8"
        )
    start = time.monotonic()
    try:
        summary = run_replay(args, incomplete, start=start)
        if output.exists():
            raise FileExistsError(f"output appeared during replay: {output}")
        incomplete.rename(output)
    except BaseException as error:
        _record_failure(incomplete, error, start)
        raise
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(output),
                "comparison": str(output / "comparison.json"),
                "micro_iou": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_replay(
    args: argparse.Namespace, output: Path, *, start: float
) -> dict[str, float]:
    manifest = _load_json(args.fixed_action_manifest)
    _validate_fixed_manifest(manifest, args)
    source_samples = _load_and_validate_source(args, manifest)
    runtime_manifest = _runtime_manifest(args, manifest, output, start)

    replay_results: dict[str, dict[str, Any]] = {}
    action_records: dict[str, Any] = {}
    safety = {
        arm: {
            "action_count": 0,
            "negative_added_pixels": 0,
            "changed_outside_authorized_pixels": 0,
            "authorized_mask_pixels": 0,
            "authorized_live_pixels": 0,
            "removed_pixels": 0,
        }
        for arm in ARMS
    }
    simpleclick_calls = 0
    source_hash_match_count = 0

    for sample in manifest["replay"]["samples"]:
        source = source_samples[sample]
        sample_result, sample_records, sample_calls, sample_matches = _replay_sample(
            args, output, source, safety
        )
        replay_results[sample] = sample_result
        action_records[sample] = sample_records
        simpleclick_calls += sample_calls
        source_hash_match_count += sample_matches

    _require_equal(
        "SimpleClick call count",
        simpleclick_calls,
        manifest["replay"]["simpleclick_call_count"],
    )
    _require_equal(
        "legacy source candidate hash match count",
        source_hash_match_count,
        manifest["replay"]["action_count"],
    )
    for arm in ARMS:
        _require_equal(
            f"{arm} negative additions",
            safety[arm]["negative_added_pixels"],
            0,
        )
        _require_equal(
            f"{arm} writes outside authorization",
            safety[arm]["changed_outside_authorized_pixels"],
            0,
        )

    action_manifest = {
        "schema_version": SCHEMA_VERSION,
        "fixed_action_manifest": str(args.fixed_action_manifest.resolve()),
        "fixed_action_manifest_sha256": _file_sha256(
            args.fixed_action_manifest.resolve()
        ),
        "source_run": str(args.source_run.resolve()),
        "validation": {
            "trajectory_files_hash_matched": len(source_samples),
            "pre_barrier_trajectory_policy": (
                "opaque sha256 only; archived trajectory JSON is not decoded because it contains offline_metrics"
            ),
            "action_and_accept_schedule_field_validation": (
                "deferred_until_after_replay_complete"
            ),
            "source_proposal_geometry_field_validation": (
                "deferred_until_after_replay_complete"
            ),
            "legacy_raw_outputs_strictly_matched": manifest["replay"][
                "action_count"
            ],
            "legacy_candidate_state_triples_strictly_matched": (
                source_hash_match_count
            ),
        },
        "samples": action_records,
    }
    _write_json(output / "action_manifest.json", action_manifest)

    runtime_manifest.update(
        {
            "status": "replay_complete_gt_not_loaded",
            "simpleclick_call_count": simpleclick_calls,
            "legacy_shared_call_count": manifest["replay"]["action_count"],
            "component_input_call_count": manifest["replay"]["action_count"],
            "safety": safety,
        }
    )
    _write_json(output / "experiment_manifest.json", runtime_manifest)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "status": "replay_complete",
        "gt_loaded": False,
        "samples": list(manifest["replay"]["samples"]),
        "arms": list(ARMS),
        "fixed_action_count": manifest["replay"]["action_count"],
        "simpleclick_call_count": simpleclick_calls,
        "legacy_raw_shared_by_arms": [LEGACY_ARM, HYBRID_ARM],
        "legacy_source_raw_hashes_matched": source_hash_match_count,
        "legacy_source_candidate_state_triples_matched": source_hash_match_count,
        "trajectory_json_decoded_pre_barrier": False,
        "gt_derived_counts_or_metrics_loaded_pre_barrier": False,
        "all_negative_additions_zero": all(
            value["negative_added_pixels"] == 0 for value in safety.values()
        ),
        "all_changed_outside_authorization_zero": all(
            value["changed_outside_authorized_pixels"] == 0
            for value in safety.values()
        ),
        "final_state_hashes": {
            sample: {
                arm: _state_hashes(replay_results[sample][arm]["final"])
                for arm in ARMS
            }
            for sample in manifest["replay"]["samples"]
        },
    }
    _write_json(output / "replay_complete.json", marker)

    # This is the only transition into code that opens label_cvt ground truth.
    metrics = _evaluate_after_replay(
        args, output, manifest, source_samples, replay_results, safety
    )
    action_manifest["validation"].update(
        {
            "action_and_accept_schedule_field_validation": True,
            "source_proposal_geometry_field_validation": True,
            "trajectory_field_validation_performed_after_replay_complete": True,
        }
    )
    _write_json(output / "action_manifest.json", action_manifest)
    comparison = _build_comparison(manifest, metrics, safety)
    _write_json(output / "per_sample_metrics.json", metrics)
    _write_json(output / "comparison.json", comparison)

    runtime_manifest.update(
        {
            "status": "success",
            "completed_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "gt_loaded_after_replay_complete": True,
            "metrics_file": "per_sample_metrics.json",
            "comparison_file": "comparison.json",
            "micro_metrics": {
                arm: metrics["arms"][arm]["aggregate"] for arm in ARMS
            },
            "acceptance": {
                arm: comparison["arms"][arm]["acceptance"] for arm in ARMS
            },
        }
    )
    _write_json(output / "experiment_manifest.json", runtime_manifest)
    summary = "; ".join(
        f"{arm} IoU={metrics['arms'][arm]['aggregate']['iou']:.8f}, "
        f"relative_1pct_pass={str(comparison['arms'][arm]['acceptance']['within_relative_1_percent_line']).lower()}"
        for arm in ARMS
    )
    (output / "completion_summary.txt").write_text(summary + "\n", encoding="utf-8")
    return {arm: metrics["arms"][arm]["aggregate"]["iou"] for arm in ARMS}


def _replay_sample(
    args: argparse.Namespace,
    output: Path,
    source: SourceSample,
    safety: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    root = Path(__file__).resolve().parents[1]
    common_backend = SubprocessPointBackend(
        args.segagent_python,
        args.worker,
        output / "tool_runs" / source.name / "legacy_shared",
        checkpoint=args.simpleclick_checkpoint,
        pythonpath=(
            root,
            root.parent / "SegAgent",
            root.parent / "SegAgent/third_party/SimpleClick",
        ),
        device=args.device,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
    )
    component_backend = SubprocessPointBackend(
        args.segagent_python,
        args.worker,
        output / "tool_runs" / source.name / "component_input",
        checkpoint=args.simpleclick_checkpoint,
        pythonpath=(
            root,
            root.parent / "SegAgent",
            root.parent / "SegAgent/third_party/SimpleClick",
        ),
        device=args.device,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
    )
    initial = _build_state(
        source.initial_masks["t1"], source.initial_masks["t2"], source.processor
    )
    live = {arm: initial for arm in ARMS}
    result: dict[str, Any] = {}
    for arm in ARMS:
        files = _persist_state(
            output, arm, source.name, 0, "committed", initial
        )
        result[arm] = {"initial": initial, "steps": [], "files": {"0": files}}

    records: list[dict[str, Any]] = []
    source_matches = 0
    for expected in source.expected["actions"]:
        index = int(expected["step_index"])
        target = str(expected["target_view"])
        coordinate = tuple(int(value) for value in expected["coordinate"])
        history = tuple(
            (
                tuple(int(value) for value in item["coordinate"]),
                bool(item["is_positive"]),
            )
            for item in expected["accepted_click_history"]
        )
        archived_raw = _validate_archived_point_call(source, expected, args.seed)
        session_mask = source.initial_masks[target]
        raw_shared = common_backend.refine(
            source.images[target], session_mask, coordinate, False, history
        )
        _validate_live_worker_call(
            common_backend.last_evidence,
            expected,
            session_mask,
            raw_shared,
            history,
            args,
            context=f"{source.name} step {index} legacy shared",
        )
        _assert_mask_equal(
            raw_shared,
            archived_raw,
            f"{source.name} step {index} live raw vs archived raw",
        )
        _require_equal(
            f"{source.name} step {index} live raw hash",
            mask_sha256(raw_shared),
            expected["source_raw_sha256"],
        )

        proposal_box = tuple(
            int(value) for value in expected["source_proposal_box_xyxy"]
        )
        source_component = _clicked_component_in_box(
            _target_mask(live[LEGACY_ARM], target), proposal_box, coordinate
        )
        _require_equal(
            f"{source.name} step {index} source proposal component area",
            int(source_component.sum()),
            int(expected["source_proposal_component_area"]),
        )

        component_input = _clicked_component_in_box(
            _target_mask(live[COMPONENT_ARM], target), proposal_box, coordinate
        )
        # Hold the frozen accepted click history constant across arms so this
        # treatment changes only the prev_mask/model-input scope.  Production
        # ActionExecutor also forwards point_click_history for every input scope.
        raw_component = component_backend.refine(
            source.images[target], component_input, coordinate, False, history
        )
        _validate_live_worker_call(
            component_backend.last_evidence,
            expected,
            component_input,
            raw_component,
            history,
            args,
            context=f"{source.name} step {index} component input",
        )
        hybrid_component = _clicked_component_in_box(
            _target_mask(live[HYBRID_ARM], target), proposal_box, coordinate
        )
        height, width = session_mask.shape
        legacy_box = _legacy_point_box(
            coordinate,
            (width, height),
            0.125,
        )
        legacy_authorization = _box_mask(session_mask.shape, legacy_box)
        arm_inputs = {
            LEGACY_ARM: {
                "raw": raw_shared,
                "authorization": legacy_authorization,
                "model_input": session_mask,
                "history": history,
                "input_scope": "archived_initial_target_mask",
                "write_scope": "legacy_point_roi",
                "effective_box": legacy_box,
                "raw_family": "legacy_shared",
                "raw_call_index": index - 1,
            },
            COMPONENT_ARM: {
                "raw": raw_component,
                "authorization": component_input,
                "model_input": component_input,
                "history": history,
                "input_scope": "live_clicked_component",
                "write_scope": "live_clicked_component",
                "effective_box": proposal_box,
                "raw_family": "component_input",
                "raw_call_index": index - 1,
            },
            HYBRID_ARM: {
                "raw": raw_shared,
                "authorization": hybrid_component,
                "model_input": session_mask,
                "history": history,
                "input_scope": "archived_initial_target_mask",
                "write_scope": "live_clicked_component",
                "effective_box": proposal_box,
                "raw_family": "legacy_shared",
                "raw_call_index": index - 1,
            },
        }
        step_record = {
            "step_index": index,
            "source_action": {
                "tool": expected["tool"],
                "action": expected["action"],
                "target_view": target,
                "coordinate": list(coordinate),
                "coordinate_space": expected["coordinate_space"],
                "accepted": bool(expected["accepted"]),
                "accepted_click_history": expected["accepted_click_history"],
                "source_region_id": expected["source_region_id"],
                "source_proposal_box_xyxy": list(proposal_box),
                "source_proposal_component_area": int(source_component.sum()),
            },
            "raw_sharing": {
                "shared_by": [LEGACY_ARM, HYBRID_ARM],
                "shared_raw_sha256": mask_sha256(raw_shared),
                "component_raw_sha256": mask_sha256(raw_component),
            },
            "arms": {},
        }
        for arm in ARMS:
            before = live[arm]
            configuration = arm_inputs[arm]
            candidate, audit = _negative_candidate(
                before,
                target,
                configuration["raw"],
                configuration["authorization"],
                source.processor,
            )
            audit.update(
                {
                    "arm": arm,
                    "input_scope": configuration["input_scope"],
                    "write_scope": configuration["write_scope"],
                    "effective_box_xyxy": list(configuration["effective_box"]),
                    "model_input_mask_pixels": int(
                        configuration["model_input"].sum()
                    ),
                    "model_input_mask_sha256": mask_sha256(
                        configuration["model_input"]
                    ),
                    "accepted_click_history": [
                        {
                            "coordinate": list(item[0]),
                            "is_positive": bool(item[1]),
                        }
                        for item in configuration["history"]
                    ],
                    "authorization_mask_pixels": int(
                        configuration["authorization"].sum()
                    ),
                    "authorization_mask_sha256": mask_sha256(
                        configuration["authorization"]
                    ),
                    "raw_family": configuration["raw_family"],
                    "raw_call_index": configuration["raw_call_index"],
                    "raw_output_pixels": int(configuration["raw"].sum()),
                    "raw_output_sha256": mask_sha256(configuration["raw"]),
                    "source_schedule_accepted": bool(expected["accepted"]),
                }
            )
            if audit["negative_added_pixels"] != 0:
                raise ValueError(
                    f"negative point added pixels for {source.name} step {index} {arm}"
                )
            if audit["changed_outside_authorized_pixels"] != 0:
                raise ValueError(
                    "write escaped authorization for "
                    f"{source.name} step {index} {arm}"
                )
            if arm == LEGACY_ARM:
                _require_equal(
                    f"{source.name} step {index} source removed pixels",
                    audit["removed_pixels"],
                    int(expected["source_removed_pixels"]),
                )
                _require_equal(
                    f"{source.name} step {index} candidate hashes",
                    _state_hashes(candidate),
                    expected["candidate_hashes"],
                )
                archived_change = _source_change_mask(source, index)
                _assert_mask_equal(
                    candidate.change,
                    archived_change,
                    f"{source.name} step {index} candidate change vs archive",
                )
                source_matches += 1
                audit["legacy_source_reproduction"] = {
                    "raw_hash_match": True,
                    "t1_hash_match": True,
                    "t2_hash_match": True,
                    "change_hash_match": True,
                    "archived_change_array_match": True,
                }
            committed = candidate if expected["accepted"] else before
            live[arm] = committed
            candidate_files = _persist_state(
                output, arm, source.name, index, "candidate", candidate
            )
            committed_files = _persist_state(
                output, arm, source.name, index, "committed", committed
            )
            audit["candidate_files"] = candidate_files
            audit["committed_files"] = committed_files
            audit_path = (
                output / "audits" / arm / source.name / f"step_{index:03d}.json"
            )
            audit["audit_file"] = str(audit_path.relative_to(output))
            _write_json(audit_path, audit)
            step_record["arms"][arm] = audit
            result[arm]["steps"].append(
                {
                    "step_index": index,
                    "accepted": bool(expected["accepted"]),
                    "candidate": candidate,
                    "committed": committed,
                    "audit": audit,
                }
            )
            safety[arm]["action_count"] += 1
            safety[arm]["negative_added_pixels"] += audit[
                "negative_added_pixels"
            ]
            safety[arm]["changed_outside_authorized_pixels"] += audit[
                "changed_outside_authorized_pixels"
            ]
            safety[arm]["authorized_mask_pixels"] += audit[
                "authorized_mask_pixels"
            ]
            safety[arm]["authorized_live_pixels"] += audit[
                "authorized_live_pixels"
            ]
            safety[arm]["removed_pixels"] += audit["removed_pixels"]
        records.append(step_record)

    _require_equal(f"{source.name} legacy call count", common_backend.call_index, 3)
    _require_equal(
        f"{source.name} component call count", component_backend.call_index, 3
    )
    for arm in ARMS:
        result[arm]["final"] = live[arm]
    return result, records, common_backend.call_index + component_backend.call_index, source_matches


def _load_and_validate_source(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> dict[str, SourceSample]:
    source_root = args.source_run.resolve()
    rollout = source_root / "rollout_complete.json"
    _require_equal(
        "source rollout_complete file hash",
        _file_sha256(rollout),
        manifest["source"]["rollout_complete_sha256"],
    )
    rollout_payload = _load_json(rollout)
    _require_equal("source rollout gt_loaded", rollout_payload.get("gt_loaded"), False)
    for sample in manifest["replay"]["samples"]:
        if sample not in rollout_payload.get("samples", []):
            raise ValueError(f"source rollout marker omits frozen sample: {sample}")

    result: dict[str, SourceSample] = {}
    for sample in manifest["replay"]["samples"]:
        expected = manifest["samples"][sample]
        trajectory_path = source_root / "trajectories" / sample / "trajectory.json"
        _require_equal(
            f"{sample} trajectory file hash",
            _file_sha256(trajectory_path),
            expected["trajectory_sha256"],
        )
        # The archived trajectory contains post-rollout ``offline_metrics``.
        # Before the replay barrier it is deliberately treated as opaque bytes:
        # its pinned file hash is checked, but the JSON is not decoded.  The
        # field-level action/schedule/proposal validation happens after
        # replay_complete.json has been durably written.
        init = source_root / "tool_runs" / sample / "sam3_initialization/initialize_000"
        images = {
            "t1": _load_rgb(init / "t1_image.png"),
            "t2": _load_rgb(init / "t2_image.png"),
        }
        initial_masks = {
            "t1": _load_mask(init / "t1_mask.npy"),
            "t2": _load_mask(init / "t2_mask.npy"),
        }
        for view, dataset_dir in (("t1", "A"), ("t2", "B")):
            image_path = init / f"{view}_image.png"
            _require_equal(
                f"{sample} archived {view} image file hash",
                _file_sha256(image_path),
                expected["archived_image_file_sha256"][view],
            )
            dataset_image = _load_rgb(args.input_root / dataset_dir / f"{sample}.png")
            _assert_array_equal(
                images[view], dataset_image, f"{sample} archived vs dataset {view} image"
            )
            _require_equal(
                f"{sample} initial {view} hash",
                mask_sha256(initial_masks[view]),
                expected["initial_hashes"][view],
            )
        processor = MaskPairProcessor(
            overlap_threshold=float(manifest["source"]["overlap_threshold"]),
            matching_mode=str(manifest["source"]["matching_mode"]),
            t12_min_instance_area=int(
                manifest["source"]["t12_min_instance_area"]
            ),
            cd_min_instance_area=int(manifest["source"]["cd_min_instance_area"]),
        )
        initial = _build_state(initial_masks["t1"], initial_masks["t2"], processor)
        _require_equal(
            f"{sample} rebuilt initial hashes",
            _state_hashes(initial),
            expected["initial_hashes"],
        )
        archived_initial_change = _load_mask(
            source_root / "masks" / sample / "step_000.npy"
        )
        _assert_mask_equal(
            initial.change,
            archived_initial_change,
            f"{sample} rebuilt initial change vs archive",
        )
        result[sample] = SourceSample(
            name=sample,
            expected=expected,
            trajectory_path=trajectory_path,
            images=images,
            initial_masks=initial_masks,
            processor=processor,
        )
    return result


def _validate_trajectory(
    sample: str,
    trajectory: dict[str, Any],
    expected: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    _require_equal(
        f"{sample} trajectory selection policy",
        trajectory.get("selection_policy"),
        manifest["source"]["selection_policy"],
    )
    _require_equal(
        f"{sample} trajectory best step",
        trajectory.get("best_step"),
        expected["source_best_step"],
    )
    _require_equal(
        f"{sample} trajectory verifier best step",
        trajectory.get("verifier_best_step"),
        expected["source_verifier_best_step"],
    )
    metadata = trajectory.get("metadata", {})
    for key, value in (
        ("sample", f"{sample}.png"),
        ("git_commit", manifest["source"]["git_commit"]),
        ("git_worktree_sha256", manifest["source"]["git_worktree_sha256"]),
        ("matching_mode", manifest["source"]["matching_mode"]),
        ("overlap_threshold", manifest["source"]["overlap_threshold"]),
        ("t12_min_instance_area", manifest["source"]["t12_min_instance_area"]),
        ("cd_min_instance_area", manifest["source"]["cd_min_instance_area"]),
        ("gt_available_during_rollout", False),
    ):
        _require_equal(f"{sample} trajectory metadata {key}", metadata.get(key), value)
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{sample} trajectory steps is not a list")
    _require_equal(f"{sample} trajectory step count", len(steps), 4)
    initial = steps[0]
    _require_equal(f"{sample} initial step index", initial.get("step_index"), 0)
    _require_equal(f"{sample} initial parsed action", initial.get("parsed_action"), None)
    _require_equal(f"{sample} initial accepted", initial.get("candidate_accepted"), True)
    _require_equal(
        f"{sample} trajectory initial hashes",
        {
            "t1": initial.get("t1_mask_sha256"),
            "t2": initial.get("t2_mask_sha256"),
            "change": initial.get("change_mask_sha256"),
        },
        expected["initial_hashes"],
    )
    for expected_action, step in zip(expected["actions"], steps[1:], strict=True):
        index = expected_action["step_index"]
        _require_equal(f"{sample} step {index} index", step.get("step_index"), index)
        _require_equal(
            f"{sample} step {index} tool", step.get("tool"), expected_action["tool"]
        )
        parsed = {
            "target_view": expected_action["target_view"],
            "action": expected_action["action"],
            "coordinate_space": expected_action["coordinate_space"],
            "coordinate": expected_action["coordinate"],
        }
        _require_equal(
            f"{sample} step {index} parsed action", step.get("parsed_action"), parsed
        )
        _require_equal(
            f"{sample} step {index} accept schedule",
            step.get("candidate_accepted"),
            expected_action["accepted"],
        )
        execution = step.get("execution", {})
        _require_equal(
            f"{sample} step {index} duplicated execution acceptance",
            execution.get("candidate_accepted"),
            expected_action["accepted"],
        )
        for location, tool_input in (
            ("top-level", step.get("tool_input", {})),
            ("execution", execution.get("tool_input", {})),
        ):
            _require_equal(
                f"{sample} step {index} {location} coordinate",
                tool_input.get("coordinate"),
                expected_action["coordinate"],
            )
            _require_equal(
                f"{sample} step {index} {location} click polarity",
                tool_input.get("is_positive"),
                False,
            )
            _require_equal(
                f"{sample} step {index} {location} accepted history",
                tool_input.get("accepted_click_history"),
                expected_action["accepted_click_history"],
            )
        _require_equal(
            f"{sample} step {index} candidate hashes",
            {
                "t1": step.get("t1_mask_sha256"),
                "t2": step.get("t2_mask_sha256"),
                "change": step.get("change_mask_sha256"),
            },
            expected_action["candidate_hashes"],
        )
        proposal = _trajectory_proposal_record(trajectory, index)
        _require_equal(
            f"{sample} step {index} source region id",
            proposal["region_id"],
            expected_action["source_region_id"],
        )
        _require_equal(
            f"{sample} step {index} source proposal area",
            proposal["component_area"],
            expected_action["source_proposal_component_area"],
        )
        _require_equal(
            f"{sample} step {index} source proposal box",
            _normalized_box_to_pixels(proposal["box_normalized_1000"], (256, 256)),
            expected_action["source_proposal_box_xyxy"],
        )
        _require_equal(
            f"{sample} step {index} source proposal seed",
            _normalized_point_to_pixel(
                proposal["component_seed_normalized_1000"], (256, 256)
            ),
            expected_action["coordinate"],
        )


def _trajectory_proposal_record(
    trajectory: dict[str, Any], step_index: int
) -> dict[str, Any]:
    previous = trajectory["steps"][step_index - 1]
    execution = previous.get("execution", {})
    rollback = execution.get("rollback_replan") or {}
    region_id = (
        ((rollback.get("verifier_evidence") or {}).get("replan") or {}).get(
            "selected_region_id"
        )
    )
    if region_id is None:
        region_id = (
            (((execution.get("verifier_evidence") or {}).get("stage_trace") or {}).get("plan") or {}).get("region_id")
        )
    if not region_id:
        raise ValueError(f"source trajectory has no proposal for step {step_index}")
    records: list[dict[str, Any]] = []
    for step in reversed(trajectory["steps"][:step_index]):
        trace = (
            ((step.get("execution") or {}).get("verifier_evidence") or {}).get(
                "stage_trace"
            )
            or {}
        )
        for key in ("evidence", "replan_evidence"):
            records.extend(
                item
                for item in trace.get(key, [])
                if item.get("region_id") == region_id
                and item.get("audit_kind") == "present"
            )
    if not records:
        raise ValueError(
            f"source trajectory has no present proposal record {region_id} for step {step_index}"
        )
    return records[0]


def _validate_archived_point_call(
    source: SourceSample, expected: dict[str, Any], seed: int
) -> np.ndarray:
    index = int(expected["step_index"])
    point = (
        source.trajectory_path.parents[2]
        / "tool_runs"
        / source.name
        / f"point_{index - 1:03d}"
    )
    target = expected["target_view"]
    initial = _load_mask(point / "initial_mask.npy")
    _assert_mask_equal(
        initial,
        source.initial_masks[target],
        f"{source.name} step {index} archived point initial mask",
    )
    image = _load_rgb(point / "image.png")
    _assert_array_equal(
        image,
        source.images[target],
        f"{source.name} step {index} archived point image",
    )
    raw = _load_mask(point / "output_mask.npy")
    _require_equal(
        f"{source.name} step {index} archived raw hash",
        mask_sha256(raw),
        expected["source_raw_sha256"],
    )
    report = _load_json(point / "report.json")
    for key, value in (
        ("status", "success"),
        ("mode", "point"),
        ("coordinate", expected["coordinate"]),
        ("is_positive", False),
        ("accepted_click_history", expected["accepted_click_history"]),
        ("initial_mask_pixels", int(initial.sum())),
        ("output_pixels", int(raw.sum())),
    ):
        _require_equal(
            f"{source.name} step {index} archived report {key}",
            report.get(key),
            value,
        )
    _require_equal(
        f"{source.name} step {index} archived worker seed",
        (report.get("seed_runtime") or {}).get("seed"),
        seed,
    )
    return raw


def _validate_live_worker_call(
    evidence: dict[str, Any],
    expected: dict[str, Any],
    initial_mask: np.ndarray,
    raw: np.ndarray,
    history: tuple[tuple[tuple[int, int], bool], ...],
    args: argparse.Namespace,
    *,
    context: str,
) -> None:
    report = evidence.get("worker_report", {})
    expected_history = [
        {"coordinate": list(coordinate), "is_positive": bool(is_positive)}
        for coordinate, is_positive in history
    ]
    for key, value in (
        ("status", "success"),
        ("mode", "point"),
        ("device", args.device),
        ("coordinate", expected["coordinate"]),
        ("is_positive", False),
        ("accepted_click_history", expected_history),
        ("initial_mask_pixels", int(initial_mask.sum())),
        ("output_pixels", int(raw.sum())),
    ):
        _require_equal(f"{context} report {key}", report.get(key), value)
    _require_equal(
        f"{context} worker seed",
        (report.get("seed_runtime") or {}).get("seed"),
        args.seed,
    )


def _negative_candidate(
    before: MaskState,
    target: str,
    raw: np.ndarray,
    authorization: np.ndarray,
    processor: MaskPairProcessor,
) -> tuple[MaskState, dict[str, int]]:
    before_target = _target_mask(before, target)
    raw = np.asarray(raw, dtype=bool)
    authorization = np.asarray(authorization, dtype=bool)
    if raw.shape != before_target.shape or authorization.shape != before_target.shape:
        raise ValueError("raw, authorization, and live target masks must have equal shape")
    removable = np.logical_and(np.logical_and(before_target, ~raw), authorization)
    candidate_target = np.logical_and(before_target, ~removable)
    negative_added = np.logical_and(candidate_target, ~before_target)
    changed = np.logical_xor(candidate_target, before_target)
    outside = np.logical_and(changed, ~authorization)
    t1 = candidate_target if target == "t1" else before.t1
    t2 = candidate_target if target == "t2" else before.t2
    candidate = _build_state(t1, t2, processor)
    return candidate, {
        "before_target_pixels": int(before_target.sum()),
        "candidate_target_pixels": int(candidate_target.sum()),
        "raw_removed_within_authorization_pixels": int(removable.sum()),
        "removed_pixels": int(np.logical_and(before_target, ~candidate_target).sum()),
        "negative_added_pixels": int(negative_added.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_outside_authorized_pixels": int(outside.sum()),
        "authorized_mask_pixels": int(authorization.sum()),
        "authorized_live_pixels": int(
            np.logical_and(before_target, authorization).sum()
        ),
        "before_t1_sha256": mask_sha256(before.t1),
        "before_t2_sha256": mask_sha256(before.t2),
        "before_change_sha256": mask_sha256(before.change),
        "candidate_t1_sha256": mask_sha256(candidate.t1),
        "candidate_t2_sha256": mask_sha256(candidate.t2),
        "candidate_change_sha256": mask_sha256(candidate.change),
    }


def _evaluate_after_replay(
    args: argparse.Namespace,
    output: Path,
    manifest: dict[str, Any],
    source_samples: dict[str, SourceSample],
    replay_results: dict[str, dict[str, Any]],
    safety: dict[str, dict[str, int]],
) -> dict[str, Any]:
    marker_path = output / "replay_complete.json"
    if not marker_path.is_file():
        raise RuntimeError("refusing GT evaluation before replay_complete.json")
    marker = _load_json(marker_path)
    _require_equal("replay barrier status", marker.get("status"), "replay_complete")
    _require_equal("replay barrier gt_loaded", marker.get("gt_loaded"), False)
    _validate_source_trajectories_after_barrier(
        source_samples, manifest, marker_path
    )
    source_baseline = _validate_source_metrics_after_barrier(
        args.source_run, manifest, marker_path
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gt_policy": (
            "label_cvt opened only after replay_complete.json with gt_loaded=false"
        ),
        "primary_selection": "frozen_source_acceptance_schedule",
        "oracle_policy": "diagnostic_only_per_sample_best_among_initial_and_step_candidates",
        "source_baseline_loaded_after_barrier": source_baseline,
        "arms": {arm: {"samples": {}} for arm in ARMS},
    }
    ground_truth: dict[str, np.ndarray] = {}
    for sample in manifest["replay"]["samples"]:
        ground_truth[sample] = _load_gt(
            args.input_root / "label_cvt" / f"{sample}.png", marker_path
        )
        for arm in ARMS:
            arm_result = replay_results[sample][arm]
            initial_metrics = _mask_metrics(arm_result["initial"].change, ground_truth[sample])
            steps = []
            oracle_options = [
                {
                    "selection": "initial",
                    "step_index": 0,
                    "state": arm_result["initial"],
                    "metrics": initial_metrics,
                }
            ]
            for step in arm_result["steps"]:
                candidate_metrics = _mask_metrics(
                    step["candidate"].change, ground_truth[sample]
                )
                committed_metrics = _mask_metrics(
                    step["committed"].change, ground_truth[sample]
                )
                steps.append(
                    {
                        "step_index": step["step_index"],
                        "source_schedule_accepted": step["accepted"],
                        "candidate": candidate_metrics,
                        "committed": committed_metrics,
                    }
                )
                oracle_options.append(
                    {
                        "selection": "candidate",
                        "step_index": step["step_index"],
                        "state": step["candidate"],
                        "metrics": candidate_metrics,
                    }
                )
            final_metrics = _mask_metrics(
                arm_result["final"].change, ground_truth[sample]
            )
            oracle = max(
                oracle_options,
                key=lambda item: (item["metrics"]["iou"], -item["step_index"]),
            )
            result["arms"][arm]["samples"][sample] = {
                "initial": initial_metrics,
                "steps": steps,
                "final_fixed_schedule": final_metrics,
                "oracle_best_diagnostic": {
                    "selection": oracle["selection"],
                    "step_index": oracle["step_index"],
                    "metrics": oracle["metrics"],
                },
                "_oracle_state": oracle["state"],
            }

    for arm in ARMS:
        final_rows = [
            result["arms"][arm]["samples"][sample]["final_fixed_schedule"]
            for sample in manifest["replay"]["samples"]
        ]
        oracle_rows = [
            result["arms"][arm]["samples"][sample]["oracle_best_diagnostic"][
                "metrics"
            ]
            for sample in manifest["replay"]["samples"]
        ]
        result["arms"][arm]["aggregate"] = _aggregate_metric_rows(final_rows)
        result["arms"][arm]["oracle_aggregate_diagnostic"] = _aggregate_metric_rows(
            oracle_rows
        )
        result["arms"][arm]["safety"] = safety[arm]
        for sample in manifest["replay"]["samples"]:
            result["arms"][arm]["samples"][sample].pop("_oracle_state")

    for sample in manifest["replay"]["samples"]:
        actual = result["arms"][LEGACY_ARM]["samples"][sample][
            "final_fixed_schedule"
        ]
        expected_counts = source_baseline["samples"][sample]["counts"]
        _require_equal(
            f"{sample} replay baseline final counts",
            {key: actual[key] for key in ("tp", "fp", "fn", "tn")},
            expected_counts,
        )
    baseline = result["arms"][LEGACY_ARM]["aggregate"]
    _require_equal(
        "replay baseline aggregate counts",
        {key: baseline[key] for key in ("tp", "fp", "fn", "tn")},
        source_baseline["aggregate_counts"],
    )
    return result


def _validate_source_trajectories_after_barrier(
    source_samples: dict[str, SourceSample],
    manifest: dict[str, Any],
    marker_path: Path,
) -> None:
    _assert_replay_barrier(marker_path)
    for sample in manifest["replay"]["samples"]:
        source = source_samples[sample]
        _require_equal(
            f"{sample} post-barrier trajectory file hash",
            _file_sha256(source.trajectory_path),
            source.expected["trajectory_sha256"],
        )
        trajectory = _load_json(source.trajectory_path)
        _validate_trajectory(sample, trajectory, source.expected, manifest)


def _validate_source_metrics_after_barrier(
    source_run: Path, manifest: dict[str, Any], marker_path: Path
) -> dict[str, Any]:
    _assert_replay_barrier(marker_path)
    path = source_run.resolve() / "per_sample_metrics.json"
    metrics = _load_json(path)
    result: dict[str, Any] = {
        "file": str(path),
        "file_sha256_computed_after_barrier": _file_sha256(path),
        "samples": {},
    }
    for sample in manifest["replay"]["samples"]:
        selected = metrics["samples"][sample]["verifier_selected"]
        result["samples"][sample] = {
            "step_index": selected["step_index"],
            "counts": {
                key: int(selected[key]) for key in ("tp", "fp", "fn", "tn")
            },
            "reported_metrics": {
                key: float(selected[key])
                for key in ("iou", "precision", "recall", "f1")
            },
        }
        _require_equal(
            f"{sample} archived selected step",
            selected["step_index"],
            manifest["samples"][sample]["source_best_step"],
        )
    result["aggregate_counts"] = {
        key: sum(result["samples"][sample]["counts"][key] for sample in manifest["replay"]["samples"])
        for key in ("tp", "fp", "fn", "tn")
    }
    result["aggregate_metrics"] = _metrics_from_counts(
        *(result["aggregate_counts"][key] for key in ("tp", "fp", "fn", "tn"))
    )
    return result


def _build_comparison(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    safety: dict[str, dict[str, int]],
) -> dict[str, Any]:
    baseline = metrics["arms"][LEGACY_ARM]["aggregate"]
    evaluation = manifest["evaluation"]
    absolute_tolerance = float(evaluation["absolute_iou_drop_tolerance"])
    relative_tolerance = float(evaluation["relative_iou_drop_tolerance"])
    absolute_line = float(baseline["iou"]) - absolute_tolerance
    relative_line = float(baseline["iou"]) * (1.0 - relative_tolerance)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "primary_result": "final_fixed_source_acceptance_schedule",
        "baseline": {
            "source": evaluation["baseline_name"],
            "arm": LEGACY_ARM,
            "metrics": baseline,
        },
        "acceptance_lines": {
            "absolute_iou_drop_at_most_0_01": absolute_line,
            "relative_iou_drop_at_most_1_percent": relative_line,
            "working_line": "relative_iou_drop_at_most_1_percent",
            "secondary_improvement_required_when_iou_regresses": True,
        },
        "arms": {},
        "caveat": (
            "This isolates model input and write scope under frozen actions and a "
            "frozen accept/rollback schedule; it is not a new live Verifier trajectory."
        ),
    }
    for arm in ARMS:
        aggregate = metrics["arms"][arm]["aggregate"]
        deltas = {
            key: float(aggregate[key]) - float(baseline[key])
            for key in ("iou", "precision", "recall", "f1")
        }
        improvements = [
            key for key in ("precision", "recall", "f1") if deltas[key] > 1e-12
        ]
        authorized_scope_reduction = (
            safety[LEGACY_ARM]["authorized_live_pixels"]
            - safety[arm]["authorized_live_pixels"]
        )
        safety_improvements = (
            ["authorized_live_scope_reduction_pixels"]
            if authorized_scope_reduction > 0
            else []
        )
        has_secondary_improvement = bool(improvements or safety_improvements)
        iou_not_lower = deltas["iou"] >= -1e-12
        within_absolute = aggregate["iou"] + 1e-12 >= absolute_line
        within_relative = aggregate["iou"] + 1e-12 >= relative_line
        result["arms"][arm] = {
            "metrics": aggregate,
            "delta_vs_baseline": deltas,
            "improved_secondary_quality_metrics": improvements,
            "improved_secondary_safety_metrics": safety_improvements,
            "safety": safety[arm],
            "authorized_live_scope_reduction_vs_legacy_pixels": (
                authorized_scope_reduction
            ),
            "oracle_aggregate_diagnostic": metrics["arms"][arm][
                "oracle_aggregate_diagnostic"
            ],
            "acceptance": {
                "iou_at_or_above_baseline": iou_not_lower,
                "within_absolute_0_01_line": within_absolute,
                "within_relative_1_percent_line": within_relative,
                "has_secondary_quality_improvement": bool(improvements),
                "has_secondary_safety_improvement": bool(safety_improvements),
                "has_any_secondary_improvement": has_secondary_improvement,
                "eligible_under_absolute_line_with_quality_tradeoff": (
                    iou_not_lower or (within_absolute and bool(improvements))
                ),
                "eligible_under_relative_line_with_quality_tradeoff": (
                    iou_not_lower or (within_relative and bool(improvements))
                ),
                "eligible_under_absolute_line_with_secondary_tradeoff": (
                    iou_not_lower
                    or (within_absolute and has_secondary_improvement)
                ),
                "eligible_under_relative_line_with_secondary_tradeoff": (
                    iou_not_lower
                    or (within_relative and has_secondary_improvement)
                ),
                "negative_additions_zero": (
                    safety[arm]["negative_added_pixels"] == 0
                ),
                "changed_outside_authorization_zero": (
                    safety[arm]["changed_outside_authorized_pixels"] == 0
                ),
            },
        }
    return result


def _validate_fixed_manifest(
    manifest: dict[str, Any], args: argparse.Namespace
) -> None:
    forbidden_gt_keys = {
        "offline_metrics",
        "baseline",
        "baseline_final_counts",
        "tp",
        "fp",
        "fn",
        "tn",
        "iou",
        "precision",
        "recall",
        "f1",
    }
    violations = _find_mapping_keys(manifest, forbidden_gt_keys)
    if violations:
        raise ValueError(
            "pre-barrier fixed action manifest contains GT-derived fields: "
            + ", ".join(violations)
        )
    _require_equal(
        "fixed manifest schema",
        manifest.get("schema_version"),
        MANIFEST_SCHEMA_VERSION,
    )
    _require_equal(
        "fixed replay arms",
        [item.get("name") for item in manifest["replay"]["arms"]],
        list(ARMS),
    )
    _require_equal("fixed replay seed", manifest["replay"]["seed"], args.seed)
    _require_equal(
        "fixed replay legacy ROI fraction",
        manifest["replay"]["legacy_point_roi_fraction"],
        0.125,
    )
    _require_equal(
        "fixed replay legacy minimum radius",
        manifest["replay"]["legacy_point_roi_min_radius"],
        2,
    )
    _require_equal("fixed replay action count", manifest["replay"]["action_count"], 9)
    _require_equal(
        "fixed replay SimpleClick call count",
        manifest["replay"]["simpleclick_call_count"],
        18,
    )
    _require_equal("fixed replay Qwen disabled", manifest["replay"]["qwen_enabled"], False)
    _require_equal(
        "fixed replay ratio gate disabled",
        manifest["replay"]["ratio_gate_enabled"],
        False,
    )
    samples = manifest["replay"]["samples"]
    _require_equal(
        "fixed replay samples",
        samples,
        ["test_20_15", "test_78_13", "test_85_16"],
    )
    if set(manifest["samples"]) != set(samples):
        raise ValueError("fixed manifest sample records do not match replay samples")
    action_count = sum(len(manifest["samples"][sample]["actions"]) for sample in samples)
    _require_equal("fixed manifest listed actions", action_count, 9)
    _require_equal(
        "fixed manifest evaluation baseline arm",
        manifest["evaluation"]["baseline_arm"],
        LEGACY_ARM,
    )
    _require_equal(
        "fixed manifest absolute IoU tolerance",
        manifest["evaluation"]["absolute_iou_drop_tolerance"],
        0.01,
    )
    _require_equal(
        "fixed manifest relative IoU tolerance",
        manifest["evaluation"]["relative_iou_drop_tolerance"],
        0.01,
    )
    _require_equal(
        "fixed manifest secondary quality metrics",
        manifest["evaluation"]["secondary_quality_metrics"],
        ["precision", "recall", "f1"],
    )
    for sample in samples:
        indexes = [
            item["step_index"] for item in manifest["samples"][sample]["actions"]
        ]
        _require_equal(f"{sample} fixed step indexes", indexes, [1, 2, 3])
        for item in manifest["samples"][sample]["actions"]:
            _require_equal(
                f"{sample} step {item['step_index']} fixed tool",
                item["tool"],
                "simpleclick",
            )
            _require_equal(
                f"{sample} step {item['step_index']} fixed action",
                item["action"],
                "negative_point",
            )


def _runtime_manifest(
    args: argparse.Namespace,
    fixed_manifest: dict[str, Any],
    output: Path,
    start: float,
) -> dict[str, Any]:
    runtime = default_run_metadata()
    checkpoint = args.simpleclick_checkpoint.resolve()
    stat = checkpoint.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "elapsed_seconds_at_manifest": round(time.monotonic() - start, 3),
        "output": str(output.with_name(output.name.removesuffix(".incomplete"))),
        "working_directory": str(output),
        "source_run": str(args.source_run.resolve()),
        "input_root": str(args.input_root.resolve()),
        "fixed_action_manifest": str(args.fixed_action_manifest.resolve()),
        "fixed_action_manifest_sha256": _file_sha256(
            args.fixed_action_manifest.resolve()
        ),
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": _file_sha256(Path(__file__).resolve()),
        "worker": str(args.worker.resolve()),
        "worker_sha256": _file_sha256(args.worker.resolve()),
        "segagent_python": str(args.segagent_python.resolve()),
        "simpleclick_checkpoint": {
            "path": str(checkpoint),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "device": args.device,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "samples": list(fixed_manifest["replay"]["samples"]),
        "arms": list(ARMS),
        "simpleclick_call_plan": (
            "9 legacy raw calls shared by A/C plus 9 component-input calls"
        ),
        "planner": None,
        "verifier": None,
        "ratio_gate": None,
        "acceptance_schedule": "strictly copied from pinned source trajectories",
        "gt_policy": (
            "before replay_complete.json the fixed manifest has no GT-derived "
            "counts/metrics and archived trajectories are checked only as opaque "
            "sha256-pinned bytes; trajectory JSON, source per_sample_metrics, and "
            "label_cvt are decoded only by _evaluate_after_replay after the barrier"
        ),
        "git_commit": runtime.get("git_commit"),
        "git_dirty": runtime.get("git_dirty"),
        "git_worktree_sha256": runtime.get("git_worktree_sha256"),
        "python": runtime.get("python"),
        "platform": runtime.get("platform"),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }


def _persist_state(
    output: Path,
    arm: str,
    sample: str,
    step_index: int,
    kind: str,
    state: MaskState,
) -> dict[str, Any]:
    directory = output / "masks" / arm / sample / f"step_{step_index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for name, mask in (("t1", state.t1), ("t2", state.t2), ("change", state.change)):
        path = directory / f"{kind}_{name}.npy"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite replay mask: {path}")
        np.save(path, np.asarray(mask, dtype=bool), allow_pickle=False)
        files[name] = {
            "file": str(path.relative_to(output)),
            "sha256": mask_sha256(mask),
            "pixels": int(mask.sum()),
        }
    preview = directory / f"{kind}_change.png"
    Image.fromarray(np.asarray(state.change, dtype=np.uint8) * 255, mode="L").save(
        preview
    )
    files["change_preview"] = str(preview.relative_to(output))
    return files


def _build_state(
    t1: np.ndarray, t2: np.ndarray, processor: MaskPairProcessor
) -> MaskState:
    t1_value = np.asarray(t1, dtype=bool).copy()
    t2_value = np.asarray(t2, dtype=bool).copy()
    if t1_value.shape != t2_value.shape:
        raise ValueError("T1 and T2 masks must have equal shape")
    update = processor.rebuild(t1_value, t2_value)
    return MaskState(
        t1=t1_value,
        t2=t2_value,
        change=np.asarray(update.change_mask, dtype=bool).copy(),
    )


def _state_hashes(state: MaskState) -> dict[str, str]:
    return {
        "t1": mask_sha256(state.t1),
        "t2": mask_sha256(state.t2),
        "change": mask_sha256(state.change),
    }


def _target_mask(state: MaskState, target: str) -> np.ndarray:
    if target == "t1":
        return state.t1
    if target == "t2":
        return state.t2
    raise ValueError(f"invalid target view: {target}")


def _clicked_component_in_box(
    live_mask: np.ndarray,
    box: tuple[int, int, int, int],
    coordinate: tuple[int, int],
) -> np.ndarray:
    roi = _box_mask(live_mask.shape, box)
    component = _component_containing(np.logical_and(live_mask, roi), coordinate)
    if not component.any():
        raise ValueError(
            f"frozen proposal click {coordinate} is not white in live proposal {box}"
        )
    return component


def _component_containing(
    mask: np.ndarray, coordinate: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    x, y = coordinate
    component = np.zeros_like(source)
    if not (0 <= x < source.shape[1] and 0 <= y < source.shape[0]):
        return component
    if not source[y, x]:
        return component
    component[y, x] = True
    stack = [(x, y)]
    while stack:
        current_x, current_y = stack.pop()
        for next_x, next_y in (
            (current_x - 1, current_y),
            (current_x + 1, current_y),
            (current_x, current_y - 1),
            (current_x, current_y + 1),
        ):
            if (
                0 <= next_x < source.shape[1]
                and 0 <= next_y < source.shape[0]
                and source[next_y, next_x]
                and not component[next_y, next_x]
            ):
                component[next_y, next_x] = True
                stack.append((next_x, next_y))
    return component


def _legacy_point_box(
    coordinate: tuple[int, int],
    image_size: tuple[int, int],
    fraction: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    radius = max(2, round(min(width, height) * fraction))
    x, y = coordinate
    return (
        max(0, x - radius),
        max(0, y - radius),
        min(width - 1, x + radius),
        min(height - 1, y + radius),
    )


def _box_mask(
    shape: tuple[int, ...], box: tuple[int, int, int, int]
) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("authorization mask shape must be two-dimensional")
    height, width = shape
    x1, y1, x2, y2 = box
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ValueError(f"invalid inclusive box {box} for shape {shape}")
    result = np.zeros((height, width), dtype=bool)
    result[y1 : y2 + 1, x1 : x2 + 1] = True
    return result


def _source_change_mask(source: SourceSample, step_index: int) -> np.ndarray:
    source_root = source.trajectory_path.parents[2]
    return _load_mask(source_root / "masks" / source.name / f"step_{step_index:03d}.npy")


def _normalized_point_to_pixel(
    point: list[int], image_size: tuple[int, int]
) -> list[int]:
    width, height = image_size
    return [
        round(int(point[0]) / 1000 * (width - 1)),
        round(int(point[1]) / 1000 * (height - 1)),
    ]


def _normalized_box_to_pixels(
    box: list[int], image_size: tuple[int, int]
) -> list[int]:
    left = _normalized_point_to_pixel(box[:2], image_size)
    right = _normalized_point_to_pixel(box[2:], image_size)
    return [left[0], left[1], right[0], right[1]]


def _mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and GT must have equal shape")
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    return _metrics_from_counts(tp, fp, fn, tn)


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    iou = tp / iou_denominator if iou_denominator else 1.0
    precision = (
        tp / precision_denominator
        if precision_denominator
        else 1.0
        if recall_denominator == 0
        else 0.0
    )
    recall = tp / recall_denominator if recall_denominator else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _metrics_from_counts(
        *(sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn"))
    )


def _load_gt(path: Path, marker_path: Path) -> np.ndarray:
    _assert_replay_barrier(marker_path)
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _assert_replay_barrier(marker_path: Path) -> None:
    if not marker_path.is_file():
        raise RuntimeError("GT barrier marker disappeared before evaluation")
    marker = _load_json(marker_path)
    if marker.get("status") != "replay_complete" or marker.get("gt_loaded") is not False:
        raise RuntimeError("GT barrier marker is invalid")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2:
        raise ValueError(f"mask must be two-dimensional: {path}")
    return np.asarray(value, dtype=bool)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_mask_equal(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    actual_value = np.asarray(actual, dtype=bool)
    expected_value = np.asarray(expected, dtype=bool)
    if actual_value.shape != expected_value.shape or not np.array_equal(
        actual_value, expected_value
    ):
        raise ValueError(f"mask array mismatch: {context}")


def _assert_array_equal(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"array mismatch: {context}")


def _require_equal(context: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: {actual!r} != {expected!r}")


def _find_mapping_keys(value: Any, forbidden: set[str], path: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in forbidden:
                result.append(child)
            result.extend(_find_mapping_keys(item, forbidden, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_find_mapping_keys(item, forbidden, f"{path}[{index}]"))
    return result


def _validate_required_paths(args: argparse.Namespace) -> None:
    for name, path, kind in (
        ("source run", args.source_run, "directory"),
        ("input root", args.input_root, "directory"),
        ("fixed action manifest", args.fixed_action_manifest, "file"),
        ("SegAgent Python", args.segagent_python, "file"),
        ("SimpleClick checkpoint", args.simpleclick_checkpoint, "file"),
        ("seeded segmentation worker", args.worker, "file"),
    ):
        exists = path.is_dir() if kind == "directory" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"missing required {name}: {path}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")


def _record_failure(output: Path, error: BaseException, start: float) -> None:
    if not output.is_dir():
        return
    message = _safe_failure_message(error)
    failure = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete",
        "failed_at": _utc_now(),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "error_type": type(error).__name__,
        "error": message,
        "output_is_incomplete": True,
    }
    try:
        _write_json(output / "failure.json", failure)
        (output / "failure_summary.txt").write_text(
            f"{type(error).__name__}: {message}\n", encoding="utf-8"
        )
    except Exception:
        pass


def _safe_failure_message(error: BaseException) -> str:
    message = str(error)
    message = re.sub(r"\bsk-[A-Za-z0-9._-]+\b", "sk-[redacted]", message)
    message = re.sub(
        r"(?i)\bBearer\s+[^\s\"',}\]]+", "Bearer [redacted]", message
    )
    return " ".join(message.split())[:1000]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"fixed replay failed: {type(error).__name__}: {_safe_failure_message(error)}", file=sys.stderr)
        raise
