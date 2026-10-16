#!/usr/bin/env python3
"""Legacy replay for archived pre-v11 candidates.

This implementation removes an entire clicked connected component for a
negative point and therefore does not reproduce the current ROI-clipped
SimpleClick Executor. It must not be used to validate v11 online trajectories.
GT is opened only after each Verifier response has been generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor
from change_agent.adapters.qwen3vl_adapter import GroundingModelQwen3VL
from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.state import AgentAction, ChangeState
from change_agent.trajectory import mask_sha256
from change_agent.verifier_regions import attach_verifier_regions


SAMPLES = ("test_20_15.png", "test_78_13.png", "test_85_16.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="building")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--verifier-retries", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--comparison-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-regions", type=int, default=6)
    parser.add_argument(
        "--acknowledge-legacy-executor-mismatch",
        action="store_true",
        help=(
            "Required acknowledgement that this replay does not match the "
            "current negative-point Executor."
        ),
    )
    parser.add_argument(
        "--max-delta-regions-per-batch",
        "--max-delta-regions",
        dest="max_delta_regions",
        type=int,
        default=3,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_legacy_executor_mismatch:
        raise SystemExit(
            "legacy replay is not valid for v11; pass "
            "--acknowledge-legacy-executor-mismatch only for archived studies"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {args.run_dir}")

    def build(temp_output: Path) -> None:
        qwen = GroundingModelQwen3VL(
            args.model_path,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=qwen.model,
            processor=qwen.processor,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.verifier_retries,
        )
        result = replay_run(
            args.run_dir,
            args.input_root,
            verifier,
            query=args.query,
            comparison_epsilon=args.comparison_epsilon,
            max_regions=args.max_regions,
            max_delta_regions=args.max_delta_regions,
        )
        temp_output.mkdir(parents=True, exist_ok=True)
        (temp_output / "replay_report.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    atomic_output(args.output, build)
    print(json.dumps({"status": "success", "output": str(args.output)}, indent=2))


def replay_run(
    run_dir: Path,
    input_root: Path,
    verifier: Any,
    *,
    query: str = "building",
    comparison_epsilon: float = 1e-6,
    max_regions: int = 6,
    max_delta_regions: int = 3,
) -> dict[str, Any]:
    """Replay all saved tool candidates without exposing GT to ``verifier``."""

    if comparison_epsilon < 0:
        raise ValueError("comparison_epsilon must be non-negative")
    samples: dict[str, Any] = {}
    for sample_file in SAMPLES:
        sample = Path(sample_file).stem
        trajectory_path = run_dir / "trajectories" / sample / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        image1 = np.asarray(Image.open(input_root / "A" / sample_file).convert("RGB"))
        image2 = np.asarray(Image.open(input_root / "B" / sample_file).convert("RGB"))
        initial_t1, initial_t2 = _initial_masks(run_dir, sample)
        metadata = trajectory.get("metadata", {})
        processor = MaskPairProcessor(
            overlap_threshold=float(metadata.get("overlap_threshold", 0.25)),
            matching_mode=metadata.get("matching_mode", "overlap_presence"),
            t12_min_instance_area=int(metadata.get("t12_min_instance_area", 0)),
            cd_min_instance_area=int(metadata.get("cd_min_instance_area", 0)),
        )
        initial = _make_state(
            image1,
            image2,
            query,
            initial_t1,
            initial_t2,
            processor,
        )
        accepted = initial
        verifier.reset()
        entries: list[dict[str, Any]] = []
        pending_offline: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
        for step in trajectory["steps"][1:]:
            if not step.get("tool") or not step.get("parsed_action"):
                continue
            candidate_t1, candidate_t2 = _candidate_masks(
                run_dir, sample, step, accepted.t1_mask, accepted.t2_mask
            )
            candidate = _make_state(
                image1,
                image2,
                query,
                candidate_t1,
                candidate_t2,
                processor,
            )
            saved_change_path = (trajectory_path.parent / step["change_mask_file"]).resolve()
            saved_change = np.asarray(np.load(saved_change_path), dtype=bool)
            if saved_change.shape != candidate.change_mask.shape:
                raise ValueError(
                    f"saved candidate mask shape {saved_change.shape} does not match "
                    f"reconstructed state {candidate.change_mask.shape}"
                )
            expected_hashes = {
                "t1": step.get("t1_mask_sha256"),
                "t2": step.get("t2_mask_sha256"),
                "change": step.get("change_mask_sha256") or mask_sha256(saved_change),
            }
            actual_hashes = {
                "t1": mask_sha256(candidate.t1_mask),
                "t2": mask_sha256(candidate.t2_mask),
                "change": mask_sha256(candidate.change_mask),
            }
            assert_replay_hashes(
                expected_hashes,
                actual_hashes,
                context=f"{sample} step {step['step_index']}",
            )

            action = _action_from_dict(step["parsed_action"])
            skipped = bool(
                step.get("execution", {}).get("verifier_skipped_by_hard_gate")
            )
            output = None
            evidence: dict[str, Any] = {}
            if not skipped:
                attach_verifier_regions(
                    candidate,
                    accepted,
                    max_regions=max_regions,
                    max_delta_regions=max_delta_regions,
                )
                output = verifier.verify(candidate, None, action, accepted)
                evidence = getattr(verifier, "last_evidence", {})
            entry = {
                "step_index": step["step_index"],
                "candidate_accepted_in_original_run": step.get("candidate_accepted"),
                "verifier_skipped_by_hard_gate": skipped,
                "replay_hash_match": True,
                "replay_candidate_hashes": actual_hashes,
                "verifier_output": output.to_dict() if output is not None else None,
                "verifier_evidence": evidence,
            }
            entries.append(entry)
            pending_offline.append((entry, accepted.change_mask.copy(), candidate.change_mask.copy()))
            if step.get("candidate_accepted"):
                accepted = candidate

        # Ground truth is opened only after every Verifier call for this sample.
        gt = np.asarray(Image.open(input_root / "label_cvt" / sample_file)) > 0
        initial_iou = _iou(initial.change_mask, gt)
        for entry, previous_change, candidate_change in pending_offline:
            previous_iou = _iou(previous_change, gt)
            candidate_iou = _iou(candidate_change, gt)
            expected = comparison_label(
                candidate_iou - previous_iou, epsilon=comparison_epsilon
            )
            output = entry["verifier_output"]
            entry["offline_after_verifier"] = {
                "previous_iou": previous_iou,
                "candidate_iou": candidate_iou,
                "expected_comparison": expected,
                "comparison_match": (
                    output["comparison"] == expected if output is not None else None
                ),
            }
        samples[sample] = {
            "initial_iou": initial_iou,
            "original_verifier_best_step": trajectory.get("verifier_best_step"),
            "entries": entries,
        }
    summary = _summary(samples)
    metrics = json.loads((run_dir / "per_sample_metrics.json").read_text())
    return {
        "decision_mode": "qwen_rich_delta_diagnosis_then_global_synthesis",
        "gt_policy": "GT opened only after every verifier output for a sample",
        "run_dir": str(run_dir),
        "samples": samples,
        "summary": summary,
        "source_metrics": metrics,
    }


def atomic_output(output: Path, writer: Callable[[Path], None]) -> None:
    """Commit a generated output directory only when ``writer`` succeeds."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        writer(temporary)
        if output.exists():
            raise FileExistsError(f"output appeared during run: {output}")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def comparison_label(delta: float, *, epsilon: float = 1e-6) -> str:
    if delta > epsilon:
        return "better"
    if delta < -epsilon:
        return "worse"
    return "unchanged"


def assert_replay_hashes(
    expected: dict[str, str | None], actual: dict[str, str], *, context: str
) -> None:
    mismatches = [
        name
        for name, expected_hash in expected.items()
        if expected_hash is not None and expected_hash != actual.get(name)
    ]
    if mismatches:
        raise ValueError(
            f"replay candidate hash mismatch for {context}: " + ", ".join(mismatches)
        )


def _initial_masks(run_dir: Path, sample: str) -> tuple[np.ndarray, np.ndarray]:
    directory = run_dir / "tool_runs" / sample / "sam3_initialization" / "initialize_000"
    return (
        np.asarray(np.load(directory / "t1_mask.npy"), dtype=bool),
        np.asarray(np.load(directory / "t2_mask.npy"), dtype=bool),
    )


def _candidate_masks(
    run_dir: Path,
    sample: str,
    step: dict[str, Any],
    initial_t1: np.ndarray,
    initial_t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t1 = np.array(initial_t1, dtype=bool, copy=True)
    t2 = np.array(initial_t2, dtype=bool, copy=True)
    parsed = step["parsed_action"]
    target = parsed["target_view"]
    worker_command = step.get("execution", {}).get("tool_result", {}).get("worker_command", [])
    output_mask = _argument_value(worker_command, "--output-mask")
    if output_mask is None:
        # Older artifacts retain the same deterministic point directory layout.
        output_mask = str(
            run_dir / "tool_runs" / sample / "point_{:03d}".format(step["step_index"] - 1) / "output_mask.npy"
        )
    result = np.asarray(np.load(output_mask), dtype=bool)
    target_initial = t1 if target == "t1" else t2
    action_name = parsed["action"]
    if action_name == "positive_point":
        coordinate = tuple(parsed["coordinate"])
        result = np.logical_or(
            target_initial, _component_containing(result, coordinate)
        )
    elif action_name == "negative_point":
        coordinate = tuple(parsed["coordinate"])
        result = np.logical_and(
            target_initial, ~_component_containing(target_initial, coordinate)
        )
    elif action_name == "box":
        x1, y1, x2, y2 = parsed["box"]
        composed = np.array(target_initial, copy=True)
        composed[y1 : y2 + 1, x1 : x2 + 1] = result[
            y1 : y2 + 1, x1 : x2 + 1
        ]
        result = composed
    if target == "t1":
        t1 = result
    else:
        t2 = result
    return t1, t2


def _component_containing(
    mask: np.ndarray, coordinate: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    x, y = coordinate
    component = np.zeros_like(source)
    if not (0 <= x < source.shape[1] and 0 <= y < source.shape[0]) or not source[y, x]:
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


def _make_state(
    image1: np.ndarray,
    image2: np.ndarray,
    query: str,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    processor: MaskPairProcessor,
) -> ChangeState:
    update = processor.rebuild(t1_mask, t2_mask)
    return ChangeState(
        image1,
        image2,
        query,
        t1_mask,
        t2_mask,
        update.change_mask,
        t1_instances=update.t1_instances,
        t2_instances=update.t2_instances,
        matching=update.matching,
        evidence=update.evidence,
    )


def _action_from_dict(value: dict[str, Any]) -> AgentAction:
    return AgentAction(
        target_view=value["target_view"],
        action=value["action"],
        coordinate=tuple(value["coordinate"]) if value.get("coordinate") else None,
        box=tuple(value["box"]) if value.get("box") else None,
    )


def _argument_value(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return intersection / union if union else 1.0


def _summary(samples: dict[str, Any]) -> dict[str, Any]:
    rows = [
        entry["offline_after_verifier"]
        for sample in samples.values()
        for entry in sample["entries"]
        if entry["offline_after_verifier"]["comparison_match"] is not None
    ]
    return {
        "candidate_count": len(rows),
        "comparison_match_count": sum(bool(row["comparison_match"]) for row in rows),
        "comparison_accuracy": (
            sum(bool(row["comparison_match"]) for row in rows) / len(rows)
            if rows
            else None
        ),
    }


if __name__ == "__main__":
    main()
