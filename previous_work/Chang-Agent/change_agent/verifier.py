"""GT-free rule baseline and interfaces for a learned Change Verifier."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .coordinates import pixel_box_to_normalized
from .state import AgentAction, ChangeState, VerifierOutput


class Verifier(Protocol):
    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput: ...


class RuleBasedVerifier:
    """A transparent no-GT baseline, not a substitute for the trained verifier."""

    def __init__(
        self,
        accept_threshold: float = 0.82,
        min_change_ratio: float = 0.0005,
        max_change_ratio: float = 0.65,
    ):
        self.accept_threshold = accept_threshold
        self.min_change_ratio = min_change_ratio
        self.max_change_ratio = max_change_ratio

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        mask = state.change_mask
        ratio = float(mask.mean())
        confidence = state.evidence.get("change_confidence")
        if confidence is None:
            confidence_score = 0.5
        else:
            confidence_array = np.asarray(confidence, dtype=float)
            if confidence_array.shape == mask.shape and mask.any():
                confidence_score = float(np.clip(confidence_array[mask].mean(), 0, 1))
            else:
                confidence_score = float(np.clip(confidence_array.mean(), 0, 1))

        if ratio < self.min_change_ratio:
            score = 0.15 * confidence_score
            error_type = "false_negative"
            suggested = "positive_point"
            feedback = "The predicted change is nearly empty; inspect the suggested target view for a missing instance."
            region = _normalized_region(_full_region(mask.shape), mask.shape)
        elif ratio > self.max_change_ratio:
            score = 0.2 * confidence_score
            error_type = "false_positive_change"
            suggested = "negative_point"
            feedback = "The change region is implausibly broad; remove unsupported foreground."
            region = _normalized_region(_mask_bbox(mask), mask.shape)
        else:
            # Confidence dominates; a mild area prior prevents empty/full-mask shortcuts.
            area_prior = 1.0 - min(abs(ratio - 0.12) / 0.53, 1.0)
            score = float(np.clip(0.75 * confidence_score + 0.25 * area_prior, 0, 1))
            error_type = "none" if score >= self.accept_threshold else "uncertain_region"
            suggested = "finish" if score >= self.accept_threshold else "positive_point"
            feedback = (
                "The candidate is supported by the available model evidence."
                if score >= self.accept_threshold
                else "Model evidence remains uncertain in the current change region."
            )
            region = (
                None
                if score >= self.accept_threshold
                else _normalized_region(_mask_bbox(mask), mask.shape)
            )

        delta = score - previous_score if previous_score is not None else 0.0
        comparison = (
            "initial"
            if previous_state is None
            else "better"
            if delta > 0
            else "worse"
            if delta < 0
            else "unchanged"
        )
        target_view = _target_view(state, previous_action)
        return VerifierOutput(
            quality_score=score,
            progress_score=delta,
            score_delta=delta,
            comparison=comparison,
            error_type=error_type,
            target_view=target_view,
            error_region=region,
            suggested_action=suggested,
            feedback=feedback,
            accept=score >= self.accept_threshold,
        )


def _target_view(state: ChangeState, previous_action: AgentAction | None) -> str:
    hint = state.evidence.get("target_view_hint")
    if hint in {"t1", "t2"}:
        return hint
    if previous_action is not None:
        return previous_action.target_view
    return "t2"


def _full_region(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    return 0, 0, width - 1, height - 1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _normalized_region(
    region: tuple[int, int, int, int] | None, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if region is None:
        return None
    height, width = shape
    return pixel_box_to_normalized(region, (width, height))
