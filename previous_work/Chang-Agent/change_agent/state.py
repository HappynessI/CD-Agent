"""Typed state and protocol objects shared by all Change-Agent components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from .coordinates import PROTOCOL_COORDINATE_SPACE, validate_normalized_box

TargetView = Literal["t1", "t2"]
ActionName = Literal["positive_point", "negative_point", "box", "finish"]
ComparisonLabel = Literal["initial", "better", "worse", "unchanged", "uncertain"]


def _copy_array(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.array(array, copy=True)


@dataclass(frozen=True)
class AgentAction:
    """A validated action in pixel coordinates."""

    target_view: TargetView
    action: ActionName
    coordinate: tuple[int, int] | None = None
    box: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target_view": self.target_view,
            "action": self.action,
            "coordinate_space": "pixel",
        }
        if self.coordinate is not None:
            result["coordinate"] = list(self.coordinate)
        if self.box is not None:
            result["box"] = list(self.box)
        return result


@dataclass(frozen=True)
class VerifierOutput:
    # ``quality_score`` remains optional for rule/trained baselines and old artifacts.
    # The rich Qwen verifier supplies it; older rule/trained outputs may leave it unset.
    quality_score: float | None = None
    score_delta: float = 0.0
    progress_score: float | None = None
    comparison: ComparisonLabel | None = None
    error_type: str = "none"
    target_view: TargetView = "t2"
    error_region: tuple[int, int, int, int] | None = None
    suggested_action: ActionName | None = "finish"
    feedback: str = ""
    accept: bool = False
    verifier_valid: bool = True
    localization_valid: bool = True
    stop: bool | None = None

    def __post_init__(self) -> None:
        if self.quality_score is not None and not 0.0 <= float(self.quality_score) <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        if self.progress_score is not None and not -1.0 <= float(self.progress_score) <= 1.0:
            raise ValueError("progress_score must be in [-1, 1]")
        if self.error_region is not None:
            validate_normalized_box(self.error_region)
        if self.comparison not in {
            None,
            "initial",
            "better",
            "worse",
            "unchanged",
            "uncertain",
        }:
            raise ValueError("unsupported comparison label")
        if self.stop is None:
            object.__setattr__(self, "stop", bool(self.accept))
        if not self.verifier_valid:
            object.__setattr__(self, "accept", False)
            object.__setattr__(self, "stop", False)
            object.__setattr__(self, "suggested_action", None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": (
                float(self.quality_score) if self.quality_score is not None else None
            ),
            "progress_score": (
                float(self.progress_score) if self.progress_score is not None else None
            ),
            "score_delta": float(self.score_delta),
            "comparison": self.comparison,
            "error_type": self.error_type,
            "target_view": self.target_view,
            "error_region": list(self.error_region) if self.error_region else None,
            "coordinate_space": PROTOCOL_COORDINATE_SPACE,
            "suggested_action": self.suggested_action,
            "feedback": self.feedback,
            "accept": bool(self.accept),
            "verifier_valid": bool(self.verifier_valid),
            "localization_valid": bool(self.localization_valid),
            "stop": bool(self.stop),
        }


@dataclass
class ChangeState:
    """Environment state. Only :meth:`public_observation` is exposed to the Agent."""

    t1_image: np.ndarray
    t2_image: np.ndarray
    query: str
    t1_mask: np.ndarray
    t2_mask: np.ndarray
    change_mask: np.ndarray
    t1_instances: tuple[np.ndarray, ...] = ()
    t2_instances: tuple[np.ndarray, ...] = ()
    matching: tuple[tuple[int, int], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    step_index: int = 0

    def __post_init__(self) -> None:
        self.t1_image = _copy_array(self.t1_image)
        self.t2_image = _copy_array(self.t2_image)
        self.t1_mask = _copy_array(self.t1_mask, dtype=bool)
        self.t2_mask = _copy_array(self.t2_mask, dtype=bool)
        self.change_mask = _copy_array(self.change_mask, dtype=bool)
        expected = self.t1_image.shape[:2]
        if self.t2_image.shape[:2] != expected:
            raise ValueError("T1 and T2 images must have the same spatial size")
        for name, mask in (
            ("t1_mask", self.t1_mask),
            ("t2_mask", self.t2_mask),
            ("change_mask", self.change_mask),
        ):
            if mask.shape != expected:
                raise ValueError(f"{name} shape {mask.shape} does not match image shape {expected}")
        if not self.query.strip():
            raise ValueError("query must not be empty")

    @property
    def image_size(self) -> tuple[int, int]:
        height, width = self.t1_image.shape[:2]
        return width, height

    def clone(self) -> "ChangeState":
        return ChangeState(
            t1_image=self.t1_image,
            t2_image=self.t2_image,
            query=self.query,
            t1_mask=self.t1_mask,
            t2_mask=self.t2_mask,
            change_mask=self.change_mask,
            t1_instances=tuple(_copy_array(x, dtype=bool) for x in self.t1_instances),
            t2_instances=tuple(_copy_array(x, dtype=bool) for x in self.t2_instances),
            matching=tuple(self.matching),
            evidence=dict(self.evidence),
            step_index=self.step_index,
        )


@dataclass(frozen=True)
class AgentObservation:
    """The GT-free model predictions and imagery supplied to an Agent."""

    t1_image: np.ndarray
    t2_image: np.ndarray
    query: str
    change_mask: np.ndarray
    feedback: VerifierOutput | None = None
    history_summary: str = ""
    t1_mask: np.ndarray | None = None
    t2_mask: np.ndarray | None = None

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "t1_image": self.t1_image,
            "t2_image": self.t2_image,
            "query": self.query,
            "current_change_mask": self.change_mask,
            "predicted_t1_mask": self.t1_mask,
            "predicted_t2_mask": self.t2_mask,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "history_summary": self.history_summary,
        }
