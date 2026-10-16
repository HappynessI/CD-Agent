#!/usr/bin/env python3
"""Run the Verifier-planned Environment → Executor loop on LEVIR samples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from change_agent.action_parser import ActionValidationError
from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.terminal_review_verifier import TerminalReviewVerifier
from change_agent.adapters.stage_backends import (
    BailianQwen3VLStageBackend,
    LocalQwen3VLStageBackend,
)
from change_agent.adapters.direct_verifier import DirectQwenVerifier
from change_agent.adapters.subprocess_adapters import (
    SubprocessBoxBackend,
    SubprocessPointBackend,
    SubprocessSAM3Initializer,
)
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentObservation
from change_agent.trajectory import default_run_metadata
from change_agent.verifier import RuleBasedVerifier
from change_agent.verifier_regions import EMPTY_MASK_SCAN_PROPOSAL_VERSION
from change_agent.terminal_review import (
    FACTOR_ID,
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
    TerminalReviewPolicy,
)


def _staged_protocol_version() -> str:
    return os.environ.get("CHANGE_AGENT_STAGED_PROTOCOL_VERSION", "v12.1")


def _effective_empty_mask_scan_grid_size(args: argparse.Namespace) -> int:
    return (
        args.empty_mask_scan_grid_size
        if args.verifier == "qwen_staged" and args.proposal_mode != "direct"
        else 0
    )


SAMPLES = ("test_20_15.png", "test_78_13.png", "test_85_16.png")


def parse_args() -> argparse.Namespace:
    root = Path("/Data/wyh/CD-SegAgent")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "outputs/omniovcd_levir_3sample_vis_20260717_145327/input_subset",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="building")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(SAMPLES),
        help="Optional subset of the fixed LEVIR smoke samples (filenames or stems).",
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--selection-policy",
        choices=("verifier_best", "conservative_best", "initial"),
        default="conservative_best",
    )
    parser.add_argument("--selection-epsilon", type=float, default=0.0)
    parser.add_argument("--max-selection-area-delta", type=float, default=0.25)
    parser.add_argument("--max-locality-outside-ratio", type=float, default=0.1)
    parser.add_argument("--max-target-mask-change-ratio", type=float, default=0.25)
    parser.add_argument("--max-component-count-delta", type=int, default=4)
    parser.add_argument(
        "--max-point-proposal-change-ratio", type=float, default=0.25
    )
    parser.add_argument(
        "--max-point-proposal-change-pixels", type=int, default=256
    )
    parser.add_argument(
        "--max-point-image-change-ratio", type=float, default=0.004
    )
    parser.add_argument(
        "--disable-point-proposal-change-gate", action="store_true"
    )
    parser.add_argument(
        "--authorized-negative-point-input-scope",
        choices=("full_session", "proposal_roi", "clicked_component"),
        default="full_session",
        help=(
            "SimpleClick prev_mask for proposal-authorized negative points. "
            "The write boundary remains the clicked live component in all modes."
        ),
    )
    parser.add_argument(
        "--verifier-max-regions",
        type=int,
        default=1,
        help=(
            "Deprecated v10 batching metadata. V11 atomically audits every "
            "initial component in separate model calls."
        ),
    )
    parser.add_argument(
        "--verifier-max-delta-regions-per-batch",
        "--verifier-max-delta-regions",
        dest="verifier_max_delta_regions",
        type=int,
        default=1,
        help=(
            "Deprecated v10 batching metadata. V11 candidate evidence uses "
            "complete action-scoped deltas."
        ),
    )
    parser.add_argument(
        "--verifier-min-region-area",
        type=int,
        default=4,
        help=(
            "Deprecated compatibility option. Full-coverage v11 does not drop "
            "small connected components."
        ),
    )
    parser.add_argument("--verifier-region-padding-ratio", type=float, default=0.25)
    parser.add_argument(
        "--empty-mask-scan-grid-size",
        type=int,
        default=3,
        help=(
            "Full-frame grid used only when both temporal object masks are empty; "
            "0 disables the recovery scan."
        ),
    )
    parser.add_argument("--staged-verifier-max-total-regions", type=int, default=128)
    parser.add_argument(
        "--verifier-max-selected-regions",
        type=int,
        default=3,
        help=(
            "Deprecated v11.1 screening batch size. V12.1 audits every region "
            "once in deterministic Environment order and ignores this value."
        ),
    )
    parser.add_argument("--matching-mode", choices=sorted(MaskPairProcessor.MODES), default="overlap_presence")
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--t12-min-instance-area", type=int, default=0)
    parser.add_argument("--cd-min-instance-area", type=int, default=0)
    parser.add_argument("--model-path", default=str(root / "models/Qwen3-VL-2B-Instruct"))
    parser.add_argument(
        "--agent-backend",
        choices=("local", "bailian"),
        default="local",
        help=(
            "Deprecated compatibility option. The production runner is "
            "Verifier-planned and has no independent Agent action generator."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Deprecated compatibility option for the retired Agent generator.",
    )
    parser.add_argument(
        "--verifier",
        choices=("qwen_zero_shot", "qwen_staged", "rule"),
        default="qwen_zero_shot",
    )
    parser.add_argument(
        "--staged-verifier-backend",
        choices=("local", "bailian"),
        default="local",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("direct", "proposal", "hybrid"),
        default="hybrid",
        help=(
            "Direct uses full-state Qwen diagnosis/action geometry without Proposals; "
            "Proposal and Hybrid exhaustively audit Environment-numbered regions with "
            "local/exact-component crops; Environment owns execution geometry."
        ),
    )
    parser.add_argument("--bailian-model", default="qwen3-vl-plus")
    parser.add_argument(
        "--verifier-bailian-model",
        default=None,
        help=(
            "Hosted staged-Verifier model. When omitted, the legacy "
            "--bailian-model value is used as a compatibility alias."
        ),
    )
    parser.add_argument("--bailian-base-url", default=None)
    parser.add_argument("--bailian-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--bailian-network-retries", type=int, default=2)
    parser.add_argument(
        "--bailian-network-backoff-seconds", type=float, default=1.0
    )
    parser.add_argument(
        "--bailian-enable-thinking",
        action="store_true",
        help="Enable hosted hybrid-model reasoning before each structured stage response.",
    )
    parser.add_argument(
        "--bailian-thinking-budget",
        type=int,
        default=None,
        help="Optional positive reasoning-token budget; used only with --bailian-enable-thinking.",
    )
    parser.add_argument("--verifier-max-new-tokens", type=int, default=1024)
    parser.add_argument("--verifier-accept-threshold", type=float, default=0.82)
    parser.add_argument(
        "--verifier-min-visual-confidence",
        type=float,
        default=0.6,
        help=(
            "Deprecated compatibility option. Staged v12.1 derives evidence quality "
            "from a compact atomic audit checklist instead of "
            "model-authored numeric confidence."
        ),
    )
    parser.add_argument("--verifier-retries", type=int, default=2)
    parser.add_argument("--verifier-repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--terminal-review-arm",
        choices=("legacy", "h0", "h1"),
        default="legacy",
        help=(
            "P1 scheduler arm. legacy preserves the pre-P1 runner; h0 binds the "
            "single-round control; h1 enables exactly one GT-free terminal review."
        ),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tool-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-retries",
        type=int,
        default=3,
        help=(
            "Deprecated compatibility option. Verifier-authorized geometry is "
            "executed once; only Verifier schema/network retries remain active."
        ),
    )
    parser.add_argument("--segagent-python", default=str(root / "segagent-env/bin/python"))
    parser.add_argument("--omniovcd-python", default=str(root / "omniovcd-env/bin/python"))
    parser.add_argument("--simpleclick-checkpoint", default=str(root / "models/SimpleClick/cocolvis_vit_large.pth"))
    parser.add_argument("--sam3-checkpoint", default=str(root / "models/sam3/sam3.pt"))
    parser.add_argument(
        "--sam3-bpe",
        default=str(
            root.parent / "Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        ),
    )
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Save per-step T1/T2 binary masks and each connected-component "
            "instance mask as PNG files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staged_protocol_version = _staged_protocol_version()
    start = time.monotonic()
    seed_runtime = _seed_runtime(args.seed)
    _validate_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "trajectories", "masks", "verifier_feedback", "logs", "tool_runs"):
        (args.output / name).mkdir()
    for name in ("initial", "verifier_best", "last", "selected"):
        (args.output / "predictions" / name).mkdir()
    if args.visualize:
        (args.output / "visualizations").mkdir()
    manifest = _base_manifest(args, seed_runtime)
    p1_policy = _terminal_review_policy(args)
    if p1_policy is not None:
        manifest.update(
            terminal_review_factor_id=FACTOR_ID,
            terminal_review_arm=p1_policy.arm.value,
            harness_protocol_id=p1_policy.harness_protocol_id,
            experiment_arm_id=p1_policy.experiment_arm_id,
            max_verifier_rounds=p1_policy.max_verifier_rounds,
            max_stage_attempts_per_round=(
                p1_policy.max_stage_attempts_per_round
            ),
        )
    _atomic_write_text(
        args.output / "run_manifest.md", _render_manifest(manifest)
    )
    (args.output / ".run_incomplete").write_text(
        "The run has not reached GT-free rollout completion.\n",
        encoding="utf-8",
    )
    try:
        _run_prepared(args, staged_protocol_version, start, manifest)
    except BaseException as error:
        _record_run_failure(args.output, manifest, error, start)
        raise


def _run_prepared(
    args: argparse.Namespace,
    staged_protocol_version: str,
    start: float,
    manifest: dict[str, Any],
) -> None:
    p1_policy = _terminal_review_policy(args)
    local_qwen = None
    needs_local_qwen = (
        args.verifier == "qwen_zero_shot"
        or (
            args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "local"
        )
    )
    if needs_local_qwen:
        local_qwen = GroundingModelQwen3VL(
            args.model_path,
            device_map=args.device_map,
            max_new_tokens=args.verifier_max_new_tokens,
        )
    verifier_hosted_client = None
    if (
        args.verifier == "qwen_staged"
        and args.staged_verifier_backend == "bailian"
    ):
        verifier_hosted_client = BailianQwen3VLStageBackend(
            model=args.verifier_bailian_model or args.bailian_model,
            base_url=args.bailian_base_url,
            api_key_env=args.bailian_api_key_env,
            max_completion_tokens=args.verifier_max_new_tokens,
            seed=args.seed,
            enable_thinking=args.bailian_enable_thinking,
            thinking_budget=args.bailian_thinking_budget,
            max_network_retries=args.bailian_network_retries,
            retry_backoff_seconds=args.bailian_network_backoff_seconds,
        )
    verifier = _build_verifier(args, local_qwen, verifier_hosted_client)
    rollout_records: dict[str, dict[str, Any]] = {}
    sample_files = tuple(
        name if str(name).endswith(".png") else f"{name}.png"
        for name in args.samples
    )
    for sample_file in sample_files:
        sample_start = time.monotonic()
        sample = Path(sample_file).stem
        image1 = np.asarray(Image.open(args.input_root / "A" / sample_file).convert("RGB"))
        image2 = np.asarray(Image.open(args.input_root / "B" / sample_file).convert("RGB"))
        tool_root = args.output / "tool_runs" / sample
        point_backend = SubprocessPointBackend(
            args.segagent_python,
            Path(__file__).with_name("seeded_segmentation_worker.py"),
            tool_root,
            checkpoint=args.simpleclick_checkpoint,
            pythonpath=(
                Path(__file__).parents[1],
                Path(__file__).parents[2] / "Chang-Agent/third_party/SegAgent",
                Path(__file__).parents[2] / "Chang-Agent/third_party/SegAgent/third_party/SimpleClick",
            ),
            device=args.tool_device,
            timeout_seconds=900,
            seed=args.seed,
        )
        box_backend = SubprocessBoxBackend(
            args.omniovcd_python,
            Path(__file__).with_name("seeded_segmentation_worker.py"),
            tool_root,
            checkpoint=args.sam3_checkpoint,
            bpe=args.sam3_bpe,
            resolution=args.sam3_resolution,
            pythonpath=(
                Path(__file__).parents[1],
                Path(__file__).parents[2] / "Chang-Agent/third_party/OmniOVCD",
            ),
            device=args.tool_device,
            timeout_seconds=900,
            seed=args.seed,
        )
        initializer = SubprocessSAM3Initializer(
            args.omniovcd_python,
            Path(__file__).with_name("seeded_segmentation_worker.py"),
            tool_root / "sam3_initialization",
            checkpoint=args.sam3_checkpoint,
            bpe=args.sam3_bpe,
            resolution=args.sam3_resolution,
            pythonpath=(
                Path(__file__).parents[1],
                Path(__file__).parents[2] / "Chang-Agent/third_party/OmniOVCD",
            ),
            device=args.tool_device,
            timeout_seconds=900,
            seed=args.seed,
        )
        processor = MaskPairProcessor(
            overlap_threshold=args.overlap_threshold,
            matching_mode=args.matching_mode,
            t12_min_instance_area=args.t12_min_instance_area,
            cd_min_instance_area=args.cd_min_instance_area,
        )
        backend = OmniOVCDAdapter(
            initializer.initialize_masks, box_backend.segment_box, processor
        )
        environment = ChangeAgentEnvironment(
            backend,
            ActionExecutor(
                point_backend,
                box_backend,
                authorized_negative_point_input_scope=(
                    args.authorized_negative_point_input_scope
                ),
            ),
            verifier,
            max_steps=args.max_steps,
            selection_policy=args.selection_policy,
            selection_epsilon=args.selection_epsilon,
            max_selection_area_delta=args.max_selection_area_delta,
            max_locality_outside_ratio=args.max_locality_outside_ratio,
            max_target_mask_change_ratio=args.max_target_mask_change_ratio,
            max_component_count_delta=args.max_component_count_delta,
            max_point_proposal_change_ratio=(
                args.max_point_proposal_change_ratio
            ),
            max_point_proposal_change_pixels=(
                args.max_point_proposal_change_pixels
            ),
            max_point_image_change_ratio=(
                args.max_point_image_change_ratio
            ),
            point_proposal_change_gate_enabled=(
                not args.disable_point_proposal_change_gate
            ),
            verifier_max_regions=args.verifier_max_regions,
            verifier_max_delta_regions=args.verifier_max_delta_regions,
            verifier_min_region_area=args.verifier_min_region_area,
            verifier_region_padding_ratio=args.verifier_region_padding_ratio,
            empty_mask_scan_grid_size=(
                _effective_empty_mask_scan_grid_size(args)
            ),
            enable_verifier_regions=args.proposal_mode != "direct",
            run_metadata={
                "sample": sample_file,
                "query": args.query,
                "controller_architecture": (
                    "verifier_planner_environment_executor"
                ),
                "action_planner": "verifier",
                "executor": "deterministic_simpleclick_or_sam3_dispatch",
                "action_generation_policy": (
                    "direct_verifier_full_context"
                    if args.proposal_mode == "direct"
                    else "verifier_region_id_programmatic_geometry"
                ),
                "verifier": args.verifier,
                "verifier_model": (
                    args.verifier_bailian_model or args.bailian_model
                    if args.staged_verifier_backend == "bailian"
                    else args.model_path
                ),
                "proposal_mode": args.proposal_mode,
                "proposal_grounding_enabled": args.proposal_mode != "direct",
                "verifier_decision_mode": (
                    "qwen_full_context_direct_binary_rubric"
                    if args.proposal_mode == "direct"
                    else f"qwen_staged_deterministic_target_resolution_{staged_protocol_version}"
                    if args.proposal_mode == "proposal"
                    else f"qwen_staged_deterministic_target_resolution_{staged_protocol_version}_full_context"
                    if args.verifier == "qwen_staged"
                    else "qwen_rich_region_diagnosis_and_global_synthesis"
                    if args.verifier == "qwen_zero_shot"
                    else "legacy_rule_score"
                ),
                "verifier_region_ordering": "environment_deterministic_exhaustive",
                "legacy_verifier_max_selected_regions_ignored": (
                    args.verifier_max_selected_regions
                ),
                "staged_verifier_max_total_regions": (
                    args.staged_verifier_max_total_regions
                ),
                "verifier_do_sample": False,
                "bailian_enable_thinking": args.bailian_enable_thinking,
                "bailian_thinking_budget": args.bailian_thinking_budget,
                "bailian_network_retries": args.bailian_network_retries,
                "bailian_network_backoff_seconds": (
                    args.bailian_network_backoff_seconds
                ),
                "verifier_repetition_penalty": args.verifier_repetition_penalty,
                "verifier_min_visual_confidence": None,
                "verifier_confidence_policy": (
                    "compact checklist statuses plus one evidence string; runtime-derived quality"
                ),
                "verifier_candidate_evidence_modes": list(
                    Qwen3VLZeroShotVerifier.CANDIDATE_EVIDENCE_MODES
                )
                if args.verifier == "qwen_zero_shot"
                else None,
                "verifier_component_coverage_policy": (
                    "all_components_including_single_pixel"
                ),
                "verifier_region_padding_ratio": args.verifier_region_padding_ratio,
                "empty_mask_scan_grid_size": (
                    _effective_empty_mask_scan_grid_size(args)
                ),
                "coordinate_protocol": "public normalized_0_1000; environment pixel_xy",
                "matching_mode": args.matching_mode,
                "overlap_threshold": args.overlap_threshold,
                "t12_min_instance_area": args.t12_min_instance_area,
                "cd_min_instance_area": args.cd_min_instance_area,
                "gt_available_during_rollout": False,
                "authorized_negative_point_input_scope": (
                    args.authorized_negative_point_input_scope
                ),
                "point_proposal_change_gate_enabled": (
                    not args.disable_point_proposal_change_gate
                ),
                "max_point_proposal_change_ratio": (
                    args.max_point_proposal_change_ratio
                ),
                "max_point_proposal_change_pixels": (
                    args.max_point_proposal_change_pixels
                ),
                "max_point_image_change_ratio": (
                    args.max_point_image_change_ratio
                ),
                "terminal_review_factor_id": (
                    FACTOR_ID if p1_policy is not None else None
                ),
                "terminal_review_arm": (
                    p1_policy.arm.value if p1_policy is not None else "legacy"
                ),
                "harness_protocol_id": (
                    p1_policy.harness_protocol_id if p1_policy is not None else None
                ),
                "experiment_arm_id": (
                    p1_policy.experiment_arm_id if p1_policy is not None else None
                ),
            },
        )
        observation = environment.reset(image1, image2, args.query)
        invalid_outputs: list[dict[str, Any]] = []
        episode_stop_reason = (
            _terminal_review_stop_reason(verifier)
            or _initial_verifier_stop_reason(observation)
        )
        loop_index = 0
        while not environment.done and episode_stop_reason is None:
            loop_index += 1
            observation, attempt_errors, action_executed = _execute_verifier_action(
                environment,
                observation,
                loop_index=loop_index,
                source=(
                    "direct_verifier_full_context"
                    if args.proposal_mode == "direct"
                    else "verifier_region_id_programmatic_geometry"
                ),
            )
            invalid_outputs.extend(attempt_errors)
            reviewed_stop_reason = _terminal_review_stop_reason(verifier)
            if reviewed_stop_reason is not None:
                episode_stop_reason = reviewed_stop_reason
                break
            if not action_executed:
                episode_stop_reason = "verifier_no_executable_action"
                break
        if episode_stop_reason is None:
            last_entry = environment.trajectory.entries[-1]
            reviewed_stop_reason = _terminal_review_stop_reason(verifier)
            final_feedback = observation.feedback
            legacy_finish = bool(
                last_entry.parsed_action is not None
                and last_entry.parsed_action.action == "finish"
                and last_entry.execution.get("candidate_accepted")
                and final_feedback is not None
                and final_feedback.stop
            )
            if (
                reviewed_stop_reason is None
                and legacy_finish
                and p1_policy is not None
                and p1_policy.terminal_review
            ):
                raise RuntimeError(
                    "H1 accepted finish lacks a registered terminal-review transition"
                )
            episode_stop_reason = (
                reviewed_stop_reason
                or ("verifier_authorized_finish" if legacy_finish else None)
                or "max_steps_reached"
            )
        tool_steps = [entry for entry in environment.trajectory.entries if entry.execution.get("tool")]
        fallback_steps = [
            entry
            for entry in tool_steps
            if entry.execution.get("action_generation", {}).get(
                "empty_mask_scan_fallback"
            )
        ]

        trajectory_dir = args.output / "trajectories" / sample
        trajectory_path = environment.trajectory.save(
            trajectory_dir,
            args.output / "masks" / sample,
            (
                args.output / "visualizations" / sample
                if args.visualize
                else None
            ),
        )
        feedback_path = args.output / "verifier_feedback" / f"{sample}.json"
        feedback_path.write_text(
            json.dumps(
                [entry.verifier.to_dict() for entry in environment.trajectory.entries],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if invalid_outputs:
            _atomic_write_text(
                trajectory_dir / "action_attempt_errors.json",
                json.dumps(invalid_outputs, indent=2, ensure_ascii=False),
            )
        accepted_count = sum(
            bool(entry.execution.get("candidate_accepted"))
            for entry in environment.trajectory.entries[1:]
        )
        rejected_count = sum(
            entry.execution.get("candidate_accepted") is False
            for entry in environment.trajectory.entries[1:]
        )
        episode_outcome = _episode_outcome(episode_stop_reason)
        terminal_reviews = _terminal_review_history(environment)
        terminal_review_audit = (
            copy.deepcopy(terminal_reviews[-1]) if terminal_reviews else None
        )
        verifier_last_audit = copy.deepcopy(
            getattr(verifier, "last_terminal_review_audit", None)
        )
        if verifier_last_audit != terminal_review_audit:
            raise RuntimeError(
                "terminal-review history does not end at verifier.last_terminal_review_audit"
            )
        if p1_policy is not None:
            if p1_policy.terminal_review and not terminal_reviews:
                raise RuntimeError("H1 episode did not retain its terminal-review audit")
            if not p1_policy.terminal_review and terminal_reviews:
                raise RuntimeError("H0 episode unexpectedly retained terminal-review audits")
        budget_accounting = _episode_budget_accounting(
            environment,
            terminal_reviews,
        )
        safety_relaxations = _safety_boundary_relaxations(environment)
        episode_summary = {
            "sample": sample,
            "execution_status": "completed",
            "episode_outcome": episode_outcome,
            "converged": episode_outcome == "converged",
            "stop_reason": episode_stop_reason,
            "elapsed_seconds": round(time.monotonic() - sample_start, 3),
            "loop_count": loop_index,
            "candidate_count": len(environment.trajectory.entries) - 1,
            "accepted_candidate_count": accepted_count,
            "rejected_candidate_count": rejected_count,
            "tool_action_count": len(tool_steps),
            "empty_mask_scan_fallback_action_count": len(fallback_steps),
            "action_attempt_count": (
                len(environment.trajectory.entries) - 1 + len(invalid_outputs)
            ),
            "action_attempt_error_count": len(invalid_outputs),
            "action_attempt_errors": invalid_outputs,
            "selected_step": environment.trajectory.best_entry.step_index,
            "verifier_best_step": environment.trajectory.verifier_best_entry.step_index,
            "trajectory": str(trajectory_path),
            "terminal_review_factor_id": (
                FACTOR_ID if p1_policy is not None else None
            ),
            "terminal_review_arm": (
                p1_policy.arm.value if p1_policy is not None else "legacy"
            ),
            "harness_protocol_id": (
                p1_policy.harness_protocol_id if p1_policy is not None else None
            ),
            "experiment_arm_id": (
                p1_policy.experiment_arm_id if p1_policy is not None else None
            ),
            "terminal_review": terminal_review_audit,
            "terminal_reviews": terminal_reviews,
            "budget_accounting": budget_accounting,
            "gt_boundary_audit": {
                "pre_rollout_gt_access_count": 0,
                "violation_ids": [],
                "rollout_completed_before_offline_scoring": True,
            },
            "safety_boundary_audit": {
                "relaxation_count": len(safety_relaxations),
                "relaxed_boundaries": safety_relaxations,
            },
            "final_region_protocol_valid": bool(
                _final_region_protocol_valid(environment, p1_policy)
            ),
        }
        episode_summary_path = trajectory_dir / "episode_summary.json"
        _atomic_write_text(
            episode_summary_path,
            json.dumps(episode_summary, indent=2, ensure_ascii=False),
        )
        selected = environment.best_state.change_mask
        verifier_best = environment.trajectory.verifier_best_entry.state.change_mask
        initial = environment.trajectory.entries[0].state.change_mask
        last = environment.trajectory.entries[-1].state.change_mask
        Image.fromarray(selected.astype(np.uint8) * 255, mode="L").save(
            args.output / "predictions" / sample_file
        )
        for name, mask in (
            ("selected", selected),
            ("verifier_best", verifier_best),
            ("initial", initial),
            ("last", last),
        ):
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                args.output / "predictions" / name / sample_file
            )
        rollout_records[sample] = {
            "trajectory": trajectory_path,
            "best_step": environment.trajectory.best_entry.step_index,
            "verifier_best_step": environment.trajectory.verifier_best_entry.step_index,
            "tool_action_count": len(tool_steps),
            "fallback_action_count": len(fallback_steps),
            "fallback_actions": [
                entry.parsed_action.to_dict()
                for entry in fallback_steps
                if entry.parsed_action is not None
            ],
            "episode_stop_reason": episode_stop_reason,
            "episode_outcome": episode_outcome,
            "episode_summary": episode_summary_path,
            "raw_actions": [entry.raw_action for entry in environment.trajectory.entries[1:]],
            "p1_harness_audit": {
                key: copy.deepcopy(episode_summary[key])
                for key in (
                    "terminal_review",
                    "terminal_reviews",
                    "budget_accounting",
                    "gt_boundary_audit",
                    "safety_boundary_audit",
                    "final_region_protocol_valid",
                )
            },
        }

    # This marker is written before label_cvt is ever opened. Everything above is the
    # GT-free runtime path; only the offline evaluator below receives the GT directory.
    if p1_policy is not None:
        _write_p1_harness_audit(
            args.output / "p1_harness_audit.json",
            p1_policy,
            rollout_records,
        )
    rollout_marker = args.output / "rollout_complete.json"
    _atomic_write_text(
        rollout_marker,
        json.dumps({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "samples": sorted(rollout_records),
            "execution_status": "completed",
            "gt_loaded": False,
        }, indent=2),
    )
    _validate_offline_labels_after_rollout(
        input_root=args.input_root,
        samples=tuple(rollout_records),
        rollout_marker=rollout_marker,
    )
    metrics = evaluate_after_rollout(rollout_records, args.input_root / "label_cvt")
    _atomic_write_text(
        args.output / "per_sample_metrics.json",
        json.dumps(metrics, indent=2),
    )
    outcome_counts = _outcome_counts(rollout_records)
    converged_count = outcome_counts.get("converged", 0)
    sample_count = len(rollout_records)
    manifest.update(
        status="completed",
        execution_status="completed",
        episode_outcome_counts=outcome_counts,
        converged_samples=converged_count,
        convergence_rate=(
            converged_count / sample_count if sample_count else 0.0
        ),
        elapsed_seconds=round(time.monotonic() - start, 3),
        rollout_marker=str(rollout_marker),
        per_sample_metrics=str(args.output / "per_sample_metrics.json"),
        rss_peak_mb=round(_rss_mb(), 2),
    )
    _atomic_write_text(
        args.output / "run_manifest.md", _render_manifest(manifest, metrics)
    )
    (args.output / ".run_incomplete").unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "execution_status": "completed",
                "output": str(args.output),
                "episode_outcome_counts": outcome_counts,
                **metrics["aggregate"],
            },
            indent=2,
        )
    )


def _build_verifier(
    args: argparse.Namespace,
    local_qwen: GroundingModelQwen3VL | None,
    hosted_client: BailianQwen3VLStageBackend | None,
):
    if args.verifier == "rule":
        return RuleBasedVerifier(accept_threshold=args.verifier_accept_threshold)
    if args.verifier == "qwen_staged":
        if args.staged_verifier_backend == "local":
            if local_qwen is None:
                raise RuntimeError("local staged verifier requires local Qwen weights")
            stage_backend = LocalQwen3VLStageBackend(
                model=local_qwen.model,
                processor=local_qwen.processor,
                max_new_tokens=args.verifier_max_new_tokens,
                do_sample=False,
                repetition_penalty=args.verifier_repetition_penalty,
            )
        else:
            if hosted_client is None:
                raise RuntimeError("BaiLian staged verifier client was not initialized")
            stage_backend = hosted_client
        if args.proposal_mode == "direct":
            if args.terminal_review_arm != "legacy":
                raise ValueError(
                    "P1 terminal review requires proposal or hybrid staged verification"
                )
            return DirectQwenVerifier(
                stage_backend,
                accept_threshold=args.verifier_accept_threshold,
                max_retries=args.verifier_retries,
            )
        staged = StagedQwenVerifier(
            stage_backend,
            max_regions=args.staged_verifier_max_total_regions,
            max_retries=args.verifier_retries,
            visual_context=args.proposal_mode,
        )
        policy = _terminal_review_policy(args)
        if policy is not None and policy.terminal_review:
            return TerminalReviewVerifier(staged, policy=policy)
        return staged
    if local_qwen is None:
        raise RuntimeError("legacy qwen_zero_shot verifier requires local Qwen weights")
    return Qwen3VLZeroShotVerifier(
        model=local_qwen.model,
        processor=local_qwen.processor,
        max_new_tokens=args.verifier_max_new_tokens,
        accept_threshold=args.verifier_accept_threshold,
        max_retries=args.verifier_retries,
        do_sample=False,
        repetition_penalty=args.verifier_repetition_penalty,
    )


def _terminal_review_policy(
    args: argparse.Namespace,
) -> TerminalReviewPolicy | None:
    arm = getattr(args, "terminal_review_arm", "legacy")
    if arm == "legacy":
        return None
    if args.verifier != "qwen_staged":
        raise ValueError("P1 terminal review requires verifier=qwen_staged")
    policy = TerminalReviewPolicy.h0() if arm == "h0" else TerminalReviewPolicy.h1()
    if args.max_steps != policy.max_environment_steps:
        raise ValueError(
            "P1 max-steps differs from the frozen environment-step budget"
        )
    if args.verifier_retries != policy.max_stage_attempts_per_round:
        raise ValueError(
            "P1 verifier retries must mean initial plus at most one repair"
        )
    if args.staged_verifier_backend != "local":
        raise ValueError("P1 H0/H1 is frozen to the same local base model backend")
    return policy


def _initial_verifier_stop_reason(observation: Any) -> str | None:
    feedback = getattr(observation, "feedback", None)
    if feedback is not None and not feedback.verifier_valid:
        return "initial_verifier_invalid"
    if (
        feedback is not None
        and bool(getattr(feedback, "accept", False))
        and bool(getattr(feedback, "stop", False))
        and getattr(feedback, "comparison", None) == "initial"
        and getattr(feedback, "error_type", None) == "none"
    ):
        return "initial_verifier_authorized_finish"
    return None


def _terminal_review_stop_reason(verifier: Any) -> str | None:
    audit = getattr(verifier, "last_terminal_review_audit", None)
    if not isinstance(audit, Mapping):
        return None
    transition = audit.get("transition")
    if not isinstance(transition, Mapping):
        return None
    next_state = transition.get("next_state")
    if next_state == "terminate_confirmed_finish":
        return "confirmed_finish"
    if next_state == "terminate_no_executable_action_after_review":
        return "no_executable_action_after_review"
    return None


def _episode_outcome(stop_reason: str) -> str:
    if stop_reason in {
        "initial_verifier_authorized_finish",
        "verifier_authorized_finish",
        "confirmed_finish",
    }:
        return "converged"
    if stop_reason == "max_steps_reached":
        return "budget_exhausted"
    if stop_reason == "initial_verifier_invalid":
        return "verifier_invalid"
    if stop_reason in {
        "verifier_no_executable_action",
        "action_retry_exhaustion_without_state_change",
        "no_executable_action_after_review",
    }:
        return "no_executable_action"
    return "stopped_without_convergence"


def _outcome_counts(
    rollout_records: dict[str, dict[str, Any]]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in rollout_records.values():
        outcome = str(record["episode_outcome"])
        counts[outcome] = counts.get(outcome, 0) + 1
    return dict(sorted(counts.items()))


_P1_COUNTER_KEYS = (
    "verifier_round_count",
    "stage_initial_attempt_count",
    "schema_repair_attempt_count",
    "network_retry_attempt_count",
    "tool_action_attempt_count",
    "tool_failure_count",
    "environment_step_count",
)


def _episode_budget_accounting(
    environment: ChangeAgentEnvironment,
    terminal_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    entries = list(environment.trajectory.entries)
    backend_calls: list[Mapping[str, Any]] = []
    review_round_counts = [1]
    for entry in entries:
        evidence = entry.execution.get("verifier_evidence")
        post_review = entry.execution.get("post_accept_terminal_review")
        post_evidence = (
            post_review.get("verifier_evidence")
            if isinstance(post_review, Mapping)
            else None
        )
        evidence_candidates: list[Mapping[str, Any]] = []
        # The post-review envelope already embeds the candidate/round-one
        # evidence, so selecting both would double-count the same backend calls.
        selected_evidence = (
            post_evidence if isinstance(post_evidence, Mapping) else evidence
        )
        if isinstance(selected_evidence, Mapping):
            evidence_candidates.append(selected_evidence)
        rollback = entry.execution.get("rollback_replan")
        rollback_evidence = (
            rollback.get("verifier_evidence")
            if isinstance(rollback, Mapping)
            else None
        )
        if isinstance(rollback_evidence, Mapping):
            evidence_candidates.append(rollback_evidence)
        for selected_evidence in evidence_candidates:
            review = selected_evidence.get("terminal_review")
            if isinstance(review, Mapping):
                counters_after = review.get("counters_after")
                if isinstance(counters_after, Mapping):
                    value = counters_after.get("verifier_round_count")
                    if isinstance(value, int) and not isinstance(value, bool):
                        review_round_counts.append(value)
                for key in ("round_one_evidence", "round_two_evidence"):
                    round_evidence = review.get(key)
                    if not isinstance(round_evidence, Mapping):
                        continue
                    calls = round_evidence.get("backend_calls")
                    if isinstance(calls, list):
                        backend_calls.extend(
                            item for item in calls if isinstance(item, Mapping)
                        )
            else:
                calls = selected_evidence.get("backend_calls")
                if isinstance(calls, list):
                    backend_calls.extend(
                        item for item in calls if isinstance(item, Mapping)
                    )
    for terminal_review_audit in terminal_reviews:
        counters_after = terminal_review_audit.get("counters_after")
        if isinstance(counters_after, Mapping):
            value = counters_after.get("verifier_round_count")
            if isinstance(value, int) and not isinstance(value, bool):
                review_round_counts.append(value)
    verifier_round_count = max(review_round_counts)
    tool_entries = [entry for entry in entries if entry.execution.get("tool")]
    tool_failures = sum(
        entry.execution.get("tool_success") is False
        or bool(entry.execution.get("tool_error"))
        for entry in tool_entries
    )
    calls_for_attempts = backend_calls
    return {
        "verifier_round_count": verifier_round_count,
        "stage_initial_attempt_count": sum(
            item.get("attempt_kind") == "initial" for item in calls_for_attempts
        ),
        "schema_repair_attempt_count": sum(
            item.get("attempt_kind") == "repair" for item in calls_for_attempts
        ),
        "network_retry_attempt_count": sum(
            _backend_network_retry_count(item)
            for item in calls_for_attempts
        ),
        "tool_action_attempt_count": len(tool_entries),
        "tool_failure_count": tool_failures,
        "environment_step_count": len(entries) - 1,
    }


def _terminal_review_history(
    environment: ChangeAgentEnvironment,
) -> list[dict[str, Any]]:
    """Collect every accepted-state audit once, in trajectory order.

    A post-accept envelope contains the same round-one evidence already exposed
    on the trajectory entry, so it replaces that primary envelope for audit
    collection.  The scheduler audit stored in the episode deliberately omits
    the nested round evidence; the hash-bound trajectory remains the raw source
    that the independent validator replays.
    """

    result: list[dict[str, Any]] = []
    for entry in environment.trajectory.entries:
        execution = entry.execution
        primary = execution.get("verifier_evidence")
        post_review = execution.get("post_accept_terminal_review")
        post_evidence = (
            post_review.get("verifier_evidence")
            if isinstance(post_review, Mapping)
            else None
        )
        evidence_blocks: list[Mapping[str, Any]] = []
        selected = post_evidence if isinstance(post_evidence, Mapping) else primary
        if isinstance(selected, Mapping):
            evidence_blocks.append(selected)
        rollback = execution.get("rollback_replan")
        rollback_evidence = (
            rollback.get("verifier_evidence")
            if isinstance(rollback, Mapping)
            else None
        )
        if isinstance(rollback_evidence, Mapping):
            evidence_blocks.append(rollback_evidence)
        for evidence in evidence_blocks:
            if "terminal_review" not in evidence:
                continue
            review = evidence.get("terminal_review")
            if not isinstance(review, Mapping):
                raise ValueError("trajectory contains a malformed terminal-review envelope")
            if not isinstance(review.get("round_one_evidence"), Mapping):
                raise ValueError("terminal-review envelope lacks raw round-one evidence")
            round_two_evidence = review.get("round_two_evidence")
            if round_two_evidence is not None and not isinstance(
                round_two_evidence, Mapping
            ):
                raise ValueError("terminal-review envelope has malformed round-two evidence")
            result.append(
                {
                    key: copy.deepcopy(value)
                    for key, value in review.items()
                    if key not in {"round_one_evidence", "round_two_evidence"}
                }
            )
    return result


def _backend_network_retry_count(call: Mapping[str, Any]) -> int:
    explicit = call.get("network_retry_attempt_count")
    if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
        raise ValueError(
            "backend call lacks a valid explicit network retry counter"
        )
    return explicit


_P1_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_P1_SCAN_REGION_RE = re.compile(r"^s[0-9]+$")
_P1_NON_SCAN_AUDIT_KINDS = {
    "present",
    "missing",
    "delta_added",
    "delta_removed",
    "mixed",
}
_P1_TERMINAL_REVIEW_CONTEXT_KEYS = {
    "round_index",
    "round_kind",
    "trigger",
    "fresh_backend_context",
    "inherits_hidden_state",
    "copies_round_one_raw_response",
    "selection_policy_version",
    "preferred_region_id",
    "prompt_version",
    "selection_branch",
    "review_region_id",
    "proposal_version",
}
_P1_PLAN_NULL_PROMPT_BRANCHES = {
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION: (
        "next_unreviewed_environment_region"
    ),
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION: (
        "first_unresolved_environment_region"
    ),
}
_P1_RAW_CALL_REQUIRED_KEYS = {
    "backend",
    "stage",
    "attempt_kind",
    "network_retry_attempt_count",
    "prompt_sha256",
    "raw_response",
    "visual_input_scope",
}


def _raw_proposal_version(region_id: str) -> str | None:
    return (
        EMPTY_MASK_SCAN_PROPOSAL_VERSION
        if _P1_SCAN_REGION_RE.fullmatch(region_id) is not None
        else None
    )


def _raw_visual_scope_valid(
    *, scan: bool, attempt_kind: Any, visual_input_scope: Any
) -> bool:
    if not isinstance(visual_input_scope, str) or not visual_input_scope:
        return False
    if attempt_kind == "initial":
        expected = (
            "authoritative_pair_rgb_only_scan" if scan else "standard"
        )
        return visual_input_scope == expected
    if attempt_kind == "repair":
        expected = (
            "authoritative_pair_rgb_only_scan" if scan else "standard"
        )
        return visual_input_scope in {
            expected,
            "authoritative_pair_rgb_only_evidence_repair",
        }
    return False


def _raw_region_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        region_id = item.get("region_id") if isinstance(item, Mapping) else None
        if not isinstance(region_id, str) or not region_id or region_id in result:
            return None
        expected_version = _raw_proposal_version(region_id)
        if (
            item.get("proposal_version") != expected_version
            or (
                item.get("audit_kind") != "scan"
                if expected_version is not None
                else item.get("audit_kind") not in _P1_NON_SCAN_AUDIT_KINDS
            )
        ):
            return None
        result.append(region_id)
    return tuple(result)


def _raw_id_list(value: Any) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return tuple(value)


def _raw_records_match(value: Any, ids: tuple[str, ...]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(ids)
        and all(
            isinstance(item, Mapping) and item.get("region_id") == region_id
            for item, region_id in zip(value, ids, strict=True)
        )
    )


def _raw_trace_region_contracts(
    trace: Mapping[str, Any],
) -> dict[str, tuple[str, str | None]] | None:
    mode = trace.get("mode")
    raw_groups = (
        (trace.get("evidence"),)
        if mode == "initial"
        else (trace.get("evidence"), trace.get("replan_evidence"))
        if mode == "candidate"
        else ()
    )
    if not raw_groups:
        return None
    result: dict[str, tuple[str, str | None]] = {}
    for raw_records in raw_groups:
        region_ids = _raw_region_ids(raw_records)
        if region_ids is None or not isinstance(raw_records, list):
            return None
        for raw_record, region_id in zip(raw_records, region_ids, strict=True):
            if region_id in result:
                return None
            result[region_id] = (
                str(raw_record["audit_kind"]),
                raw_record.get("proposal_version"),
            )
    return result or None


def _raw_trace_call_groups(
    trace: Mapping[str, Any],
) -> tuple[tuple[str, str], ...] | None:
    if "plan" not in trace:
        return None
    mode = trace.get("mode")
    if mode == "initial":
        ids = _raw_region_ids(trace.get("evidence"))
        audited = _raw_id_list(trace.get("audited_region_ids"))
        if (
            not ids
            or audited != ids
            or not _raw_records_match(trace.get("judgments"), ids)
            or not _raw_records_match(trace.get("diagnoses"), ids)
        ):
            return None
        return tuple(("audit", region_id) for region_id in ids)
    if mode != "candidate":
        return None
    candidate_ids = _raw_region_ids(trace.get("evidence"))
    candidate_audited = _raw_id_list(trace.get("audited_region_ids"))
    if (
        not candidate_ids
        or candidate_audited != candidate_ids
        or not _raw_records_match(trace.get("judgments"), candidate_ids)
        or trace.get("diagnoses") != []
    ):
        return None
    groups = [
        ("candidate_evidence", region_id) for region_id in candidate_ids
    ]
    replan_ids = _raw_region_ids(trace.get("replan_evidence"))
    if replan_ids is None:
        return None
    if replan_ids:
        if (
            _raw_id_list(trace.get("replan_audited_region_ids")) != replan_ids
            or not _raw_records_match(trace.get("replan_judgments"), replan_ids)
            or not _raw_records_match(trace.get("replan_diagnoses"), replan_ids)
        ):
            return None
        groups.extend(("audit", region_id) for region_id in replan_ids)
    elif any(
        trace.get(key) != []
        for key in (
            "replan_audited_region_ids",
            "replan_judgments",
            "replan_diagnoses",
        )
    ):
        return None
    return tuple(groups)


def _raw_staged_region_protocol_valid(
    evidence: Any,
    *,
    require_round_two: bool,
) -> bool:
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("type") != "staged_qwen_verifier"
        or evidence.get("schema_version") != StagedQwenVerifier.SCHEMA_VERSION
        or evidence.get("verifier_valid") is not True
        or evidence.get("validation_errors") != []
        or evidence.get("regional_validation_errors") != []
    ):
        return False
    trace = evidence.get("stage_trace")
    if not isinstance(trace, Mapping):
        return False
    expected_groups = _raw_trace_call_groups(trace)
    region_contracts = _raw_trace_region_contracts(trace)
    calls = evidence.get("backend_calls")
    if (
        not expected_groups
        or region_contracts is None
        or not isinstance(calls, list)
    ):
        return False
    if require_round_two:
        round_meta = evidence.get("terminal_review_round")
        if (
            not isinstance(round_meta, Mapping)
            or round_meta.get("schema_valid") is not True
        ):
            return False
    call_index = 0
    for expected_stage, expected_region in expected_groups:
        if call_index >= len(calls):
            return False
        group: list[Mapping[str, Any]] = []
        initial = calls[call_index]
        if (
            not isinstance(initial, Mapping)
            or initial.get("stage") != expected_stage
            or initial.get("attempt_kind") != "initial"
        ):
            return False
        group.append(initial)
        call_index += 1
        if (
            call_index < len(calls)
            and isinstance(calls[call_index], Mapping)
            and calls[call_index].get("stage") == expected_stage
            and calls[call_index].get("attempt_kind") == "repair"
        ):
            group.append(calls[call_index])
            call_index += 1
        parsed_states: list[bool] = []
        for call in group:
            retry_count = call.get("network_retry_attempt_count")
            prompt = call.get("prompt_sha256")
            raw_response = call.get("raw_response")
            if (
                any(key not in call for key in _P1_RAW_CALL_REQUIRED_KEYS)
                or not isinstance(call.get("backend"), str)
                or not call.get("backend")
                or not _raw_visual_scope_valid(
                    scan=region_contracts[expected_region][0] == "scan",
                    attempt_kind=call.get("attempt_kind"),
                    visual_input_scope=call.get("visual_input_scope"),
                )
                or isinstance(retry_count, bool)
                or retry_count != 0
                or not isinstance(prompt, str)
                or _P1_SHA256_RE.fullmatch(prompt) is None
                or "raw_response" not in call
                or not isinstance(raw_response, str)
            ):
                return False
            if "parsed_output" in call:
                parsed = call.get("parsed_output")
                if (
                    "parse_error" in call
                    or not isinstance(parsed, Mapping)
                    or parsed.get("region_id") != expected_region
                ):
                    return False
                parsed_states.append(True)
            else:
                if (
                    not isinstance(call.get("parse_error"), str)
                    or not call.get("parse_error")
                ):
                    return False
                parsed_states.append(False)
            context = call.get("terminal_review")
            if require_round_two:
                if (
                    not isinstance(context, Mapping)
                    or set(context) != _P1_TERMINAL_REVIEW_CONTEXT_KEYS
                    or context.get("round_index") != 2
                    or context.get("review_region_id") != expected_region
                    or context.get("proposal_version")
                    != region_contracts[expected_region][1]
                    or context.get("fresh_backend_context") is not True
                    or context.get("inherits_hidden_state") is not False
                    or context.get("copies_round_one_raw_response") is not False
                    or context.get("selection_policy_version")
                    != "environment_ordered_terminal_review_v1"
                ):
                    return False
                trigger = context.get("trigger")
                if trigger == "plan_null_after_allowed_repair":
                    prompt_version = context.get("prompt_version")
                    if (
                        len(expected_groups) != 1
                        or context.get("round_kind") != "plan_null_alternative"
                        or context.get("preferred_region_id") != expected_region
                        or _P1_PLAN_NULL_PROMPT_BRANCHES.get(prompt_version)
                        != context.get("selection_branch")
                    ):
                        return False
                elif trigger == "provisional_finish":
                    if (
                        context.get("round_kind") != "finish_confirmation"
                        or context.get("preferred_region_id") is not None
                        or context.get("prompt_version") is not None
                        or context.get("selection_branch") is not None
                    ):
                        return False
                else:
                    return False
            elif "terminal_review" in call:
                return False
        if not parsed_states[-1] or (
            len(parsed_states) == 2 and parsed_states[0]
        ):
            return False
    return call_index == len(calls)


def _trajectory_verifier_evidence_blocks(
    environment: ChangeAgentEnvironment,
) -> list[Mapping[str, Any]] | None:
    result: list[Mapping[str, Any]] = []
    for entry in environment.trajectory.entries:
        execution = entry.execution
        selected: Any = None
        if "post_accept_terminal_review" in execution:
            post = execution.get("post_accept_terminal_review")
            if not isinstance(post, Mapping) or "verifier_evidence" not in post:
                return None
            selected = post.get("verifier_evidence")
            if not isinstance(selected, Mapping):
                return None
        elif "verifier_evidence" in execution:
            selected = execution.get("verifier_evidence")
            if not isinstance(selected, Mapping):
                return None
        if selected is not None:
            result.append(selected)
        if "rollback_replan" in execution:
            rollback = execution.get("rollback_replan")
            if not isinstance(rollback, Mapping) or "verifier_evidence" not in rollback:
                return None
            rollback_evidence = rollback.get("verifier_evidence")
            if not isinstance(rollback_evidence, Mapping):
                return None
            result.append(rollback_evidence)
    return result


def _final_region_protocol_valid(
    environment: ChangeAgentEnvironment,
    policy: TerminalReviewPolicy | None,
) -> bool:
    """Rebuild the final-region gate from raw staged evidence, never feedback."""

    blocks = _trajectory_verifier_evidence_blocks(environment)
    if blocks is None or not blocks:
        return False
    final_block = blocks[-1]
    if policy is not None and policy.terminal_review:
        tail = final_block.get("terminal_review")
        if (
            not isinstance(tail, Mapping)
            or tail.get("eligible") is not True
            or tail.get("round_two_invoked") is not True
            or tail.get("round_two_count") != 1
            or tail.get("round_index") != 2
        ):
            return False
        return _raw_staged_region_protocol_valid(
            tail.get("round_two_evidence"),
            require_round_two=True,
        )
    return _raw_staged_region_protocol_valid(
        final_block,
        require_round_two=False,
    )


def _safety_boundary_relaxations(
    environment: ChangeAgentEnvironment,
) -> list[str]:
    expected = {
        "parser": True,
        "geometry_authorization": True,
        "locality_hard_gate": True,
        "candidate_benefit_without_harm": True,
        "rollback": True,
    }
    observed = {
        "parser": True,
        "geometry_authorization": bool(environment.enable_verifier_regions),
        "locality_hard_gate": True,
        "candidate_benefit_without_harm": True,
        "rollback": True,
    }
    return sorted(name for name, enabled in expected.items() if observed[name] != enabled)


def _write_p1_harness_audit(
    path: Path,
    policy: TerminalReviewPolicy,
    rollout_records: Mapping[str, Mapping[str, Any]],
) -> None:
    per_sample = {
        sample: copy.deepcopy(record["p1_harness_audit"])
        for sample, record in rollout_records.items()
    }
    totals = {key: 0 for key in _P1_COUNTER_KEYS}
    for sample in per_sample.values():
        counts = sample["budget_accounting"]
        for key in _P1_COUNTER_KEYS:
            value = counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"P1 budget counter {key} is invalid")
            totals[key] += value
    payload = {
        "schema_id": "qwen3vl_p1_harness_audit_v1",
        "factor_id": FACTOR_ID,
        "harness_protocol_id": policy.harness_protocol_id,
        "experiment_arm_id": policy.experiment_arm_id,
        "arm": policy.arm.value,
        "requested_sample_ids": list(per_sample),
        "per_sample": per_sample,
        "totals": totals,
        "gt_boundary_violation_count": sum(
            sample["gt_boundary_audit"]["pre_rollout_gt_access_count"]
            for sample in per_sample.values()
        ),
        "safety_boundary_relaxation_count": sum(
            sample["safety_boundary_audit"]["relaxation_count"]
            for sample in per_sample.values()
        ),
        "tool_failure_count": totals["tool_failure_count"],
        "written_before_rollout_complete": True,
        "gt_loaded": False,
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _execute_verifier_action(
    environment: ChangeAgentEnvironment,
    observation: AgentObservation,
    *,
    loop_index: int,
    source: str = "verifier_region_id_programmatic_geometry",
) -> tuple[AgentObservation, list[dict[str, Any]], bool]:
    """Execute geometry already authorized by the verifier/runtime contract."""

    feedback = observation.feedback
    if (
        feedback is None
        or not feedback.verifier_valid
        or feedback.suggested_action is None
    ):
        return observation, [
            {
                "loop_index": loop_index,
                "error": "verifier did not authorize an executable action",
            }
        ], False
    payload: dict[str, Any] = {
        "target_view": feedback.target_view,
        "action": feedback.suggested_action,
    }
    empty_mask_scan_fallback = bool(
        feedback.suggested_action == "box"
        and getattr(environment, "state", None) is not None
        and environment.state.evidence.get("verifier_mask_facts", {})
        .get("proposal_config", {})
        .get("empty_dual_mask_scan")
    )
    if feedback.suggested_action in {"positive_point", "negative_point"}:
        if feedback.error_region is None:
            return observation, [
                {
                    "loop_index": loop_index,
                    "error": "verifier point action lacks normalized geometry",
                }
            ], False
        payload["coordinate"] = list(feedback.error_region[:2])
    elif feedback.suggested_action == "box":
        if feedback.error_region is None:
            return observation, [
                {
                    "loop_index": loop_index,
                    "error": "verifier box action lacks normalized geometry",
                }
            ], False
        payload["box"] = list(feedback.error_region)
    raw = json.dumps(payload, separators=(",", ":"))
    authorized_region_id = _authorized_action_region_id(environment, feedback)
    proposal_authorization_required = bool(
        source == "verifier_region_id_programmatic_geometry"
        and feedback.suggested_action
        in {"positive_point", "negative_point", "box"}
    )
    if proposal_authorization_required and authorized_region_id is None:
        return observation, [
            {
                "loop_index": loop_index,
                "raw_verifier_action": raw,
                "error": (
                    "proposal-grounded verifier action lacks an authorized "
                    "live Environment region ID"
                ),
            }
        ], False
    try:
        if authorized_region_id is None:
            next_observation, _ = environment.step(raw)
        else:
            next_observation, _ = environment.step(
                raw, authorized_region_id=authorized_region_id
            )
    except ActionValidationError as error:
        return observation, [
            {
                "loop_index": loop_index,
                "raw_verifier_action": raw,
                "error": str(error),
            }
        ], False
    environment.trajectory.entries[-1].execution["action_generation"] = {
        "loop_index": loop_index,
        "source": (
            "verifier_empty_mask_scan_environment_box"
            if empty_mask_scan_fallback
            else source
        ),
        "empty_mask_scan_fallback": empty_mask_scan_fallback,
        "authorized_region_id": authorized_region_id,
    }
    return next_observation, [], True


def _authorized_action_region_id(
    environment: ChangeAgentEnvironment,
    feedback: Any,
) -> str | None:
    """Recover staged point/box region ID without exposing geometry authorship."""

    if feedback.suggested_action not in {
        "positive_point",
        "negative_point",
        "box",
    }:
        return None
    verifier = getattr(environment, "verifier", None)
    evidence = getattr(verifier, "last_evidence", None)
    if not isinstance(evidence, Mapping):
        return None
    replan = evidence.get("replan")
    if isinstance(replan, Mapping):
        region_id = replan.get("selected_region_id")
        if isinstance(region_id, str) and region_id:
            return region_id
    stage_trace = evidence.get("stage_trace")
    if isinstance(stage_trace, Mapping):
        plan = stage_trace.get("plan")
        if isinstance(plan, Mapping):
            region_id = plan.get("region_id")
            if isinstance(region_id, str) and region_id:
                return region_id
    state = getattr(environment, "state", None)
    state_evidence = getattr(state, "evidence", None)
    proposals = (
        state_evidence.get("verifier_region_proposals", [])
        if isinstance(state_evidence, Mapping)
        else []
    )
    geometry = getattr(feedback, "error_region", None)
    if geometry is None:
        return None
    if feedback.suggested_action in {"positive_point", "negative_point"}:
        expected_geometry = tuple(int(value) for value in geometry[:2])
        proposal_key = "component_seed_normalized"
    else:
        expected_geometry = tuple(int(value) for value in geometry)
        proposal_key = "box_normalized"
    matches: list[str] = []
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        proposal_geometry = proposal.get(proposal_key)
        region_id = proposal.get("region_id")
        if not isinstance(proposal_geometry, (list, tuple)) or not isinstance(
            region_id, str
        ):
            continue
        try:
            normalized_geometry = tuple(
                int(value) for value in proposal_geometry
            )
        except (TypeError, ValueError):
            continue
        if normalized_geometry == expected_geometry:
            matches.append(region_id)
    if len(matches) == 1:
        return matches[0]
    return None


def _execute_direct_verifier_action(
    environment: ChangeAgentEnvironment,
    observation: AgentObservation,
    *,
    loop_index: int,
) -> tuple[AgentObservation, list[dict[str, Any]], bool]:
    """Backward-compatible wrapper for the Direct ablation."""

    return _execute_verifier_action(
        environment,
        observation,
        loop_index=loop_index,
        source="direct_verifier_full_context",
    )


def evaluate_after_rollout(
    rollout_records: dict[str, dict[str, Any]], gt_dir: Path
) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    aggregate_counts = np.zeros(4, dtype=np.int64)
    for sample, record in rollout_records.items():
        trajectory_path = Path(record["trajectory"])
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        gt = np.asarray(Image.open(gt_dir / f"{sample}.png")) > 0
        step_metrics = []
        selected_metrics = None
        verifier_best_metrics = None
        for step in payload["steps"]:
            mask_path = (trajectory_path.parent / step["change_mask_file"]).resolve()
            prediction = np.asarray(np.load(mask_path), dtype=bool)
            values, counts = _metrics(prediction, gt)
            step["offline_metrics"] = values
            step_metrics.append({"step_index": step["step_index"], **values})
            if step["step_index"] == payload["best_step"]:
                selected_metrics = values
                aggregate_counts += counts
            if step["step_index"] == payload["verifier_best_step"]:
                verifier_best_metrics = values
        trajectory_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        samples[sample] = {
            "verifier_selected_step": payload["best_step"],
            "tool_action_count": record["tool_action_count"],
            "fallback_action_count": record["fallback_action_count"],
            "steps": step_metrics,
            "initial": step_metrics[0],
            "verifier_selected": {"step_index": payload["best_step"], **selected_metrics},
            "verifier_best_step": payload["verifier_best_step"],
            "verifier_best": {
                "step_index": payload["verifier_best_step"], **verifier_best_metrics
            },
            "last": step_metrics[-1],
        }
    aggregate, _ = _metrics_from_counts(aggregate_counts)
    return {"samples": samples, "aggregate": aggregate}


def _validate_offline_labels_after_rollout(
    *,
    input_root: Path,
    samples: Sequence[str],
    rollout_marker: Path,
) -> None:
    """Validate offline labels only after the atomic rollout marker exists.

    Reading the marker is deliberately the first filesystem operation in this
    helper.  Missing, malformed, incomplete, or sample-mismatched markers fail
    before a label path is stat'ed or opened.
    """

    try:
        marker = json.loads(rollout_marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "offline label validation requires a readable rollout_complete.json"
        ) from error
    if not isinstance(marker, Mapping):
        raise RuntimeError(
            "offline label validation requires an object rollout marker"
        )
    normalized_samples = tuple(Path(str(sample)).stem for sample in samples)
    marker_samples = marker.get("samples")
    if (
        marker.get("execution_status") != "completed"
        or marker.get("gt_loaded") is not False
        or not isinstance(marker_samples, list)
        or any(not isinstance(item, str) for item in marker_samples)
        or sorted(marker_samples) != sorted(normalized_samples)
    ):
        raise RuntimeError(
            "offline label validation requires an exact completed GT-free rollout marker"
        )
    label_root = input_root / "label_cvt"
    missing: list[str] = []
    unreadable: list[str] = []
    for sample in normalized_samples:
        label_path = label_root / f"{sample}.png"
        if not label_path.is_file():
            missing.append(str(label_path))
            continue
        try:
            with Image.open(label_path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            unreadable.append(f"{label_path}: {type(error).__name__}")
    if missing or unreadable:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unreadable:
            details.append(f"unreadable={unreadable}")
        raise FileNotFoundError(
            "offline labels failed post-rollout validation: " + "; ".join(details)
        )


def _metrics(prediction: np.ndarray, gt: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    if prediction.shape != gt.shape:
        raise ValueError(f"prediction shape {prediction.shape} != GT shape {gt.shape}")
    counts = np.asarray([
        np.logical_and(prediction, gt).sum(),
        np.logical_and(prediction, ~gt).sum(),
        np.logical_and(~prediction, gt).sum(),
        np.logical_and(~prediction, ~gt).sum(),
    ], dtype=np.int64)
    return _metrics_from_counts(counts)


def _metrics_from_counts(counts: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    tp, fp, fn, tn = (int(value) for value in counts)
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": round(iou, 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }, counts


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.proposal_mode != "hybrid" and args.verifier != "qwen_staged":
        raise ValueError("proposal ablations require --verifier qwen_staged")
    if not 0 <= args.empty_mask_scan_grid_size <= 16:
        raise ValueError("--empty-mask-scan-grid-size must be within [0,16]")
    if args.bailian_network_retries < 0:
        raise ValueError("--bailian-network-retries must be non-negative")
    if args.bailian_network_backoff_seconds < 0:
        raise ValueError("--bailian-network-backoff-seconds must be non-negative")
    if not 0 < args.max_point_proposal_change_ratio <= 1:
        raise ValueError(
            "--max-point-proposal-change-ratio must be within (0,1]"
        )
    if args.max_point_proposal_change_pixels < 0:
        raise ValueError(
            "--max-point-proposal-change-pixels must be non-negative"
        )
    if not 0 <= args.max_point_image_change_ratio <= 1:
        raise ValueError(
            "--max-point-image-change-ratio must be within [0,1]"
        )
    needs_local_qwen = (
        args.verifier == "qwen_zero_shot"
        or (
            args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "local"
        )
    )
    files = [Path(args.simpleclick_checkpoint), Path(args.sam3_checkpoint), Path(args.sam3_bpe)]
    if needs_local_qwen:
        files.append(Path(args.model_path))
    sample_files = [
        str(sample) if str(sample).endswith(".png") else f"{sample}.png"
        for sample in args.samples
    ]
    for sample in sample_files:
        files.extend([
            args.input_root / "A" / sample,
            args.input_root / "B" / sample,
        ])
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")


def _base_manifest(
    args: argparse.Namespace, seed_runtime: dict[str, Any]
) -> dict[str, Any]:
    source = default_run_metadata()
    return {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": [
            str(sample) if str(sample).endswith(".png") else f"{sample}.png"
            for sample in args.samples
        ],
        "query": args.query,
        "input_root": str(args.input_root),
        "initialization": "fresh dual-view SAM3 text prompting for every sample; no cached masks",
        "initialization_artifacts": str(args.output / "tool_runs" / "<sample>" / "sam3_initialization"),
        "controller_architecture": "verifier_planner_environment_executor",
        "action_planner": "verifier",
        "executor_policy": "deterministic SimpleClick/SAM3 dispatch",
        "action_generation_policy": (
            "direct_verifier_full_context"
            if args.proposal_mode == "direct"
            else "verifier_region_id_programmatic_geometry"
        ),
        "point_executor": f"SimpleClick {args.simpleclick_checkpoint}",
        "box_executor": f"SAM3 {args.sam3_checkpoint}",
        "verifier": args.verifier,
        "proposal_mode": args.proposal_mode,
        "proposal_semantics": {
            "direct": "full-state Qwen diagnosis and model-authored action geometry; no Proposal attachment",
            "proposal": "environment-ordered compact region audit over marked RGB/change overview plus exact component/local crops, runtime candidate transition rubric, and programmatic geometry",
            "hybrid": "environment-ordered compact region audit over marked full mask state plus exact component/local crops, runtime candidate transition rubric, and programmatic geometry",
        }[args.proposal_mode],
        "verifier_model": (
            "shared local Qwen3-VL weights"
            if args.verifier == "qwen_zero_shot"
            else args.model_path
            if args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "local"
            else args.verifier_bailian_model or args.bailian_model
            if args.verifier == "qwen_staged"
            else None
        ),
        "staged_verifier_backend": (
            args.staged_verifier_backend if args.verifier == "qwen_staged" else None
        ),
        "verifier_decision_mode": (
            "qwen_rich_region_diagnosis_and_global_synthesis"
            if args.verifier == "qwen_zero_shot"
            else "qwen_full_context_direct_binary_rubric"
            if args.verifier == "qwen_staged" and args.proposal_mode == "direct"
            else f"qwen_staged_deterministic_target_resolution_{_staged_protocol_version()}"
            if args.verifier == "qwen_staged" and args.proposal_mode == "proposal"
            else f"qwen_staged_deterministic_target_resolution_{_staged_protocol_version()}_full_context"
            if args.verifier == "qwen_staged" and args.proposal_mode == "hybrid"
            else "legacy_rule_score"
        ),
        "verifier_region_ordering": "environment_deterministic_exhaustive",
        "legacy_verifier_max_selected_regions_ignored": (
            args.verifier_max_selected_regions
        ),
        "staged_verifier_max_total_regions": (
            args.staged_verifier_max_total_regions
        ),
        "verifier_do_sample": False,
        "bailian_enable_thinking": args.bailian_enable_thinking,
        "bailian_thinking_budget": args.bailian_thinking_budget,
        "bailian_network_retries": args.bailian_network_retries,
        "bailian_network_backoff_seconds": (
            args.bailian_network_backoff_seconds
        ),
        "verifier_repetition_penalty": args.verifier_repetition_penalty,
        "verifier_min_visual_confidence": None,
        "verifier_confidence_policy": (
            "compact checklist statuses plus one evidence string; runtime-derived quality"
        ),
        "verifier_candidate_evidence_modes": list(
            Qwen3VLZeroShotVerifier.CANDIDATE_EVIDENCE_MODES
        )
        if args.verifier == "qwen_zero_shot"
        else None,
        "verifier_component_coverage_policy": (
            "all_components_including_single_pixel"
        ),
        "verifier_region_padding_ratio": args.verifier_region_padding_ratio,
        "empty_mask_scan_grid_size": (
            _effective_empty_mask_scan_grid_size(args)
        ),
        "coordinate_protocol": "Verifier normalized_0_1000; Environment pixel_xy",
        "target_view_policy": "runtime derivation from RGB states and Environment seed editability",
        "matching_mode": args.matching_mode,
        "overlap_threshold": args.overlap_threshold,
        "t12_min_instance_area": args.t12_min_instance_area,
        "cd_min_instance_area": args.cd_min_instance_area,
        "visualize": args.visualize,
        "visualization_artifacts": (
            str(args.output / "visualizations" / "<sample>" / "step_<index>")
            if args.visualize
            else None
        ),
        "max_steps": args.max_steps,
        "selection_policy": args.selection_policy,
        "selection_epsilon": args.selection_epsilon,
        "max_selection_area_delta": args.max_selection_area_delta,
        "max_locality_outside_ratio": args.max_locality_outside_ratio,
        "max_target_mask_change_ratio": args.max_target_mask_change_ratio,
        "max_component_count_delta": args.max_component_count_delta,
        "max_point_proposal_change_ratio": (
            args.max_point_proposal_change_ratio
        ),
        "max_point_proposal_change_pixels": (
            args.max_point_proposal_change_pixels
        ),
        "max_point_image_change_ratio": args.max_point_image_change_ratio,
        "point_proposal_change_gate_enabled": (
            not args.disable_point_proposal_change_gate
        ),
        "authorized_negative_point_input_scope": (
            args.authorized_negative_point_input_scope
        ),
        "seed": args.seed,
        "seed_runtime": seed_runtime,
        "python": sys.version,
        "platform": platform.platform(),
        "model_identity": (
            _model_identity(Path(args.model_path))
            if args.verifier == "qwen_zero_shot"
            or (
                args.verifier == "qwen_staged"
                and args.staged_verifier_backend == "local"
            )
            else None
        ),
        "verifier_model_identity": (
            {"provider": "bailian", "model": args.verifier_bailian_model or args.bailian_model}
            if args.verifier == "qwen_staged"
            and args.staged_verifier_backend == "bailian"
            else None
        ),
        "git_commit": source["git_commit"],
        "git_dirty": source["git_dirty"],
        "git_worktree_sha256": source["git_worktree_sha256"],
        "gt_policy": "label_cvt is opened only by evaluate_after_rollout after rollout_complete.json",
    }


def _seed_runtime(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    result: dict[str, Any] = {
        "python_random": seed,
        "numpy": seed,
        "torch": None,
        "deterministic_algorithms": False,
    }
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result["torch"] = seed
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
    except ImportError:
        result["torch_version"] = None
        result["cuda_version"] = None
    return result


def _model_identity(model_path: Path) -> dict[str, Any]:
    resolved = model_path.resolve()
    identity: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    candidates = (
        [resolved]
        if resolved.is_file()
        else [
            resolved / "config.json",
            resolved / "model.safetensors.index.json",
            resolved / "generation_config.json",
        ]
    )
    checksums: dict[str, str] = {}
    for candidate in candidates:
        if candidate.is_file():
            checksums[candidate.name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    identity["metadata_sha256"] = checksums
    return identity


def _render_manifest(manifest: dict[str, Any], metrics: dict[str, Any] | None = None) -> str:
    lines = ["# LEVIR-CD Verifier-planned Change-Agent run", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in manifest.items())
    if metrics:
        lines.extend(["", "## Aggregate verifier-selected metrics", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in metrics["aggregate"].items())
    lines.extend([
        "",
        "## Runtime boundary",
        "",
        "The final entry point is `tools/run_levir_change_agent.py`, not OmniOVCD/eval.py. ",
        "Each sample executes a fresh SAM3 initialization, Environment.reset, the configured ",
        "GT-free Verifier as diagnosis/action planner, ActionParser, a deterministic SimpleClick ",
        "or SAM3 Executor, and Environment.step/rebuild. There is no independent Agent policy ",
        "in this production path. SAM3 masks, confidence maps, scores, prompts, and ",
        "worker parameters are persisted below tool_runs/<sample>/sam3_initialization. ",
        "Ground truth is loaded only after all trajectories and rollout_complete.json exist.",
    ])
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _safe_failure_message(error: BaseException) -> str:
    message = str(error)
    message = re.sub(r"\bsk-[A-Za-z0-9._-]+\b", "sk-[redacted]", message)
    message = re.sub(
        r"(?i)\bBearer\s+[^\s\"',}\]]+",
        "Bearer [redacted]",
        message,
    )
    return " ".join(message.split())[:1000]


def _record_run_failure(
    output: Path,
    manifest: dict[str, Any],
    error: BaseException,
    start: float,
) -> None:
    failure = {
        "execution_status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "error_type": type(error).__name__,
        "error": _safe_failure_message(error),
    }
    manifest.update(
        status="failed",
        execution_status="failed",
        failure=failure,
        elapsed_seconds=failure["elapsed_seconds"],
    )
    try:
        _atomic_write_text(
            output / "failure.json",
            json.dumps(failure, indent=2, ensure_ascii=False),
        )
        _atomic_write_text(
            output / "run_manifest.md", _render_manifest(manifest)
        )
    except Exception as recording_error:
        print(
            "warning: failed to persist run failure metadata: "
            f"{type(recording_error).__name__}",
            file=sys.stderr,
        )


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if value > 10_000 else value / (1024.0 * 1024.0)


if __name__ == "__main__":
    main()
