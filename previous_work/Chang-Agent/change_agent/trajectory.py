"""Trajectory recording, best-state tracking, and reproducibility artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .state import AgentAction, ChangeState, VerifierOutput


@dataclass
class TrajectoryEntry:
    step_index: int
    raw_action: str | None
    parsed_action: AgentAction | None
    verifier: VerifierOutput
    state: ChangeState
    execution: dict[str, Any]

    def to_dict(
        self,
        mask_file: str | None = None,
        visualization_dir: str | None = None,
    ) -> dict[str, Any]:
        parsed = self.parsed_action.to_dict() if self.parsed_action else None
        verifier = self.verifier.to_dict()
        return {
            "step_index": self.step_index,
            "raw_action": self.raw_action,
            "raw_action_payload": _json_safe(self.execution.get("raw_action_payload")),
            "coordinate_warning": self.execution.get("coordinate_warning"),
            "parsed_action": parsed,
            "target_view": parsed["target_view"] if parsed else None,
            "tool": self.execution.get("tool"),
            "tool_input": _json_safe(self.execution.get("tool_input")),
            "quality_score": verifier["quality_score"],
            "progress_score": verifier["progress_score"],
            "score_delta": verifier["score_delta"],
            "comparison": verifier["comparison"],
            "error_type": verifier["error_type"],
            "suggested_action": verifier["suggested_action"],
            "accept": verifier["accept"],
            "candidate_accepted": self.execution.get("candidate_accepted"),
            "candidate_rejection_reasons": _json_safe(
                self.execution.get("candidate_rejection_reasons", [])
            ),
            "verifier": verifier,
            "execution": _json_safe(self.execution),
            "matching_evidence": _json_safe(self.state.evidence.get("matching")),
            "t1_mask_sha256": mask_sha256(self.state.t1_mask),
            "t2_mask_sha256": mask_sha256(self.state.t2_mask),
            "change_mask_sha256": mask_sha256(self.state.change_mask),
            "change_mask_file": mask_file,
            "visualization_dir": visualization_dir,
        }


class Trajectory:
    def __init__(
        self,
        run_metadata: dict[str, Any] | None = None,
        *,
        selection_policy: str = "conservative_best",
        selection_epsilon: float = 0.0,
        max_area_delta: float = 0.25,
    ):
        if selection_policy not in {"verifier_best", "conservative_best", "initial"}:
            raise ValueError("unsupported selection_policy")
        self.entries: list[TrajectoryEntry] = []
        self.run_metadata = default_run_metadata()
        self.run_metadata.update(run_metadata or {})
        self.selection_policy = selection_policy
        self.selection_epsilon = selection_epsilon
        self.max_area_delta = max_area_delta

    def append(self, entry: TrajectoryEntry) -> None:
        if self.entries and entry.step_index <= self.entries[-1].step_index:
            raise ValueError("trajectory step indices must be strictly increasing")
        self.entries.append(entry)

    @property
    def best_entry(self) -> TrajectoryEntry:
        if not self.entries:
            raise RuntimeError("trajectory is empty")
        if self.selection_policy == "initial":
            return self.entries[0]
        accepted = [
            item
            for item in self.entries
            if item.execution.get("candidate_accepted", True)
        ]
        if not accepted:
            return self.entries[0]
        # Pairwise candidates form a monotonic accepted chain: only Qwen-judged
        # ``better`` candidates are committed, so the latest accepted entry remains
        # the closed-loop best. Rich quality scores are retained for audit/history.
        if any(item.verifier.comparison is not None for item in accepted):
            return accepted[-1]
        return max(
            accepted,
            key=lambda item: (
                item.verifier.quality_score
                if item.verifier.quality_score is not None
                else float("-inf")
            ),
        )

    @property
    def verifier_best_entry(self) -> TrajectoryEntry:
        if not self.entries:
            raise RuntimeError("trajectory is empty")
        pairwise_accepted = [
            item
            for item in self.entries
            if item.execution.get("candidate_accepted", True)
            and item.verifier.comparison in {"initial", "better"}
        ]
        if pairwise_accepted:
            return pairwise_accepted[-1]
        return max(
            self.entries,
            key=lambda item: (
                item.verifier.quality_score
                if item.verifier.quality_score is not None
                else float("-inf")
            ),
        )

    def history_summary(self, limit: int = 4) -> str:
        recent = self.entries[-limit:]
        parts = []
        for item in recent:
            action = item.parsed_action.action if item.parsed_action else "reset"
            accepted = item.execution.get("candidate_accepted", True)
            score = (
                f"{item.verifier.quality_score:.3f}"
                if item.verifier.quality_score is not None
                else "n/a"
            )
            summary = (
                f"step={item.step_index}, action={action}, "
                f"score={score}, comparison={item.verifier.comparison}, "
                f"progress={item.verifier.progress_score}, error={item.verifier.error_type}, "
                f"accepted={accepted}"
            )
            rejected_payload = item.execution.get("raw_action_payload")
            if accepted is False and isinstance(rejected_payload, dict):
                summary += ", rejected_action=" + json.dumps(
                    rejected_payload, ensure_ascii=False, separators=(",", ":")
                )
            parts.append(summary)
        return "; ".join(parts)

    def rejected_action_history(self, limit: int = 4) -> list[dict[str, Any]]:
        """Return bounded GT-free rejection facts for action replanning."""

        if limit < 1:
            return []
        history: list[dict[str, Any]] = []
        for item in self.entries:
            if item.execution.get("candidate_accepted") is not False:
                continue
            action = item.parsed_action.to_dict() if item.parsed_action else None
            locality = item.execution.get("locality", {})
            history.append(
                {
                    "step_index": item.step_index,
                    "action": action,
                    "rejection_reasons": list(
                        item.execution.get("candidate_rejection_reasons", [])
                    ),
                    "candidate_comparison": item.verifier.comparison,
                    "candidate_accept": bool(item.verifier.accept),
                    "candidate_feedback": item.verifier.feedback[:400],
                    "candidate_area_delta": item.execution.get(
                        "candidate_area_delta"
                    ),
                    "changed_pixels": locality.get("changed_pixels"),
                    "outside_roi_ratio": locality.get("outside_roi_ratio"),
                }
            )
        return history[-limit:]

    def save(
        self,
        output_dir: str | Path,
        mask_output_dir: str | Path | None = None,
        visualization_output_dir: str | Path | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = Path(mask_output_dir) if mask_output_dir else output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        visualizations_dir = (
            Path(visualization_output_dir)
            if visualization_output_dir is not None
            else None
        )
        serialized = []
        for entry in self.entries:
            mask_path = masks_dir / f"step_{entry.step_index:03d}.npy"
            relative = Path(os.path.relpath(mask_path, output_dir))
            np.save(mask_path, entry.state.change_mask.astype(np.uint8))
            relative_visualization_dir = None
            if visualizations_dir is not None:
                step_dir = visualizations_dir / f"step_{entry.step_index:03d}"
                _save_state_visualizations(entry.state, step_dir)
                relative_visualization_dir = str(
                    Path(os.path.relpath(step_dir, output_dir))
                )
            serialized.append(
                entry.to_dict(str(relative), relative_visualization_dir)
            )
        payload = {
            "metadata": _json_safe(self.run_metadata),
            "best_step": self.best_entry.step_index if self.entries else None,
            "selection_policy": self.selection_policy,
            "verifier_best_step": self.verifier_best_entry.step_index if self.entries else None,
            "steps": serialized,
        }
        path = output_dir / "trajectory.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def _save_state_visualizations(state: ChangeState, output_dir: Path) -> None:
    """Persist binary temporal masks and the instances used by pair matching."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_binary_mask(state.t1_mask, output_dir / "t1_mask.png")
    _save_binary_mask(state.t2_mask, output_dir / "t2_mask.png")
    for view, instances in (
        ("t1", state.t1_instances),
        ("t2", state.t2_instances),
    ):
        instances_dir = output_dir / f"{view}_instances"
        instances_dir.mkdir(exist_ok=True)
        for index, instance in enumerate(instances):
            _save_binary_mask(instance, instances_dir / f"instance_{index:03d}.png")


def _save_binary_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def default_run_metadata() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "git_worktree_sha256": _git_worktree_sha256(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def mask_sha256(mask: np.ndarray) -> str:
    """Hash mask values together with shape so replay identity is auditable."""

    value = np.ascontiguousarray(mask, dtype=np.uint8)
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    return _git_output(["rev-parse", "HEAD"])


def _git_dirty() -> bool | None:
    status = _git_output(["status", "--short"])
    return bool(status) if status is not None else None


def _git_worktree_sha256() -> str | None:
    """Fingerprint tracked diffs and untracked file contents without storing them."""

    root = Path(__file__).resolve().parents[1]
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "."],
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return None
    digest = hashlib.sha256(diff)
    for raw_path in sorted(item for item in untracked if item):
        path = root / raw_path.decode("utf-8", errors="surrogateescape")
        digest.update(b"\0untracked\0")
        digest.update(raw_path)
        if path.is_file():
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_output(arguments: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
