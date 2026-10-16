"""Action dispatch with hard validation at the tool boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .state import AgentAction


class PointBackend(Protocol):
    def refine(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        coordinate: tuple[int, int],
        is_positive: bool,
        click_history: tuple[tuple[tuple[int, int], bool], ...] = (),
    ) -> np.ndarray: ...


class BoxBackend(Protocol):
    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray: ...


@dataclass
class ExecutionResult:
    mask: np.ndarray
    evidence: dict[str, Any]


class ActionExecutor:
    def __init__(
        self,
        point_backend: PointBackend,
        box_backend: BoxBackend,
        *,
        point_roi_fraction: float = 0.125,
        point_roi_min_padding: int = 4,
        authorized_negative_point_input_scope: str = "full_session",
    ):
        if not 0 < point_roi_fraction <= 0.5:
            raise ValueError("point_roi_fraction must be in (0, 0.5]")
        if point_roi_min_padding < 0:
            raise ValueError("point_roi_min_padding must be non-negative")
        if authorized_negative_point_input_scope not in {
            "full_session",
            "proposal_roi",
            "clicked_component",
        }:
            raise ValueError(
                "authorized_negative_point_input_scope must be one of "
                "full_session, proposal_roi, or clicked_component"
            )
        self.point_backend = point_backend
        self.box_backend = box_backend
        self.point_roi_fraction = point_roi_fraction
        self.point_roi_min_padding = point_roi_min_padding
        self.authorized_negative_point_input_scope = (
            authorized_negative_point_input_scope
        )

    def execute(
        self,
        action: AgentAction,
        image: np.ndarray,
        initial_mask: np.ndarray,
        query: str,
        *,
        point_session_mask: np.ndarray | None = None,
        point_click_history: tuple[tuple[tuple[int, int], bool], ...] = (),
        point_region_id: str | None = None,
        point_proposal_box: tuple[int, int, int, int] | None = None,
        point_proposal_component_area: int | None = None,
        box_region_id: str | None = None,
    ) -> ExecutionResult:
        height, width = image.shape[:2]
        if initial_mask.shape != (height, width):
            raise ValueError("initial mask and target image must have identical spatial size")

        if action.action in {"positive_point", "negative_point"}:
            if action.coordinate is None:
                raise ValueError("validated point action has no coordinate")
            x, y = action.coordinate
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("pixel coordinate is outside the target image")
            if (
                point_proposal_component_area is not None
                and point_proposal_component_area < 1
            ):
                raise ValueError("point proposal component area must be positive")
            session_source = (
                initial_mask
                if point_session_mask is None
                else np.asarray(point_session_mask, dtype=bool)
            )
            if session_source.shape != initial_mask.shape:
                raise ValueError(
                    "point session mask and target mask must have identical spatial size"
                )
            roi = _point_roi(
                action.coordinate,
                (width, height),
                self.point_roi_fraction,
                proposal_box=point_proposal_box,
                min_padding=self.point_roi_min_padding,
            )
            inside_roi = _roi_mask(initial_mask.shape, roi)
            proposal_session_mask = (
                np.logical_and(session_source, inside_roi)
                if point_proposal_box is not None
                else session_source
            )
            proposal_live_mask = (
                np.logical_and(initial_mask, inside_roi)
                if point_proposal_box is not None
                else np.asarray(initial_mask, dtype=bool)
            )
            model_input_mask = proposal_session_mask
            authorized_write_mask = proposal_live_mask
            session_localization_policy = (
                "proposal_roi_mask"
                if point_proposal_box is not None
                else "legacy_full_mask"
            )
            if (
                point_proposal_box is not None
                and action.action == "negative_point"
            ):
                clicked_component = _component_containing(
                    proposal_live_mask, action.coordinate
                )
                if not clicked_component.any():
                    raise ValueError(
                        "authorized negative point seed is not white in the live proposal"
                    )
                authorized_write_mask = clicked_component
                if self.authorized_negative_point_input_scope == "full_session":
                    model_input_mask = session_source
                    session_localization_policy = "full_session_mask"
                elif (
                    self.authorized_negative_point_input_scope
                    == "clicked_component"
                ):
                    model_input_mask = clicked_component
                    session_localization_policy = (
                        "clicked_component_within_proposal_roi"
                    )
            raw_mask = self.point_backend.refine(
                image,
                model_input_mask,
                action.coordinate,
                action.action == "positive_point",
                point_click_history,
            )
            raw_mask = _validated_tool_mask(raw_mask, initial_mask.shape)
            if action.action == "positive_point":
                component = _component_containing(raw_mask, action.coordinate)
                if point_proposal_box is not None:
                    component = np.logical_and(component, inside_roi)
                authorized_write_mask = component
                mask = np.logical_or(initial_mask, component)
                composition_mode = (
                    "merge_clicked_prediction_component_within_proposal_roi"
                    if point_proposal_box is not None
                    else "merge_clicked_prediction_component"
                )
            else:
                # SimpleClick returns a refined full mask.  A negative click must
                # use that prediction instead of deleting the entire connected
                # initial component, which may join real and false-positive
                # regions.  Keep the operation subtraction-only and clip it to
                # the deterministic point ROI so one click cannot add pixels or
                # silently rewrite distant parts of the accepted state.
                removable = np.logical_and(
                    np.logical_and(authorized_write_mask, ~raw_mask), inside_roi
                )
                mask = np.logical_and(initial_mask, ~removable)
                composition_mode = (
                    "remove_simpleclick_delta_within_clicked_component"
                    if point_proposal_box is not None
                    else "remove_simpleclick_delta_within_roi"
                )
            tool = "simpleclick"
            tool_input: dict[str, Any] = {
                "coordinate": list(action.coordinate),
                "is_positive": action.action == "positive_point",
                "accepted_click_history": [
                    {
                        "coordinate": list(history_coordinate),
                        "is_positive": history_is_positive,
                    }
                    for history_coordinate, history_is_positive in point_click_history
                ],
                "authorized_region_id": point_region_id,
                "proposal_box_xyxy": (
                    list(point_proposal_box)
                    if point_proposal_box is not None
                    else None
                ),
                "proposal_component_area": point_proposal_component_area,
                "effective_roi_xyxy": list(roi),
                "session_source_mask_pixels": int(session_source.sum()),
                "proposal_roi_mask_pixels": int(proposal_session_mask.sum()),
                "proposal_live_mask_pixels": int(proposal_live_mask.sum()),
                "session_initial_mask_pixels": int(model_input_mask.sum()),
                "model_input_mask_pixels": int(model_input_mask.sum()),
                "authorized_write_mask_pixels": int(
                    authorized_write_mask.sum()
                ),
                "session_localized_to_proposal": (
                    point_proposal_box is not None
                    and session_localization_policy != "full_session_mask"
                ),
                "session_localization_policy": session_localization_policy,
                "model_input_scope_policy": session_localization_policy,
                "write_scope_policy": (
                    "clicked_component_within_proposal_roi"
                    if point_proposal_box is not None
                    and action.action == "negative_point"
                    else "clicked_prediction_component_within_proposal_roi"
                    if point_proposal_box is not None
                    and action.action == "positive_point"
                    else "legacy_point_roi"
                ),
                "discarded_proposal_mask_pixels": int(
                    max(
                        0,
                        int(proposal_session_mask.sum())
                        - int(model_input_mask.sum()),
                    )
                ),
                "model_input_outside_proposal_pixels": int(
                    np.logical_and(model_input_mask, ~inside_roi).sum()
                ),
                "raw_output_mask_pixels": int(raw_mask.sum()),
            }
            if action.action == "negative_point":
                tool_input["composed_removed_pixels"] = int(removable.sum())
        elif action.action == "box":
            if action.box is None:
                raise ValueError("validated box action has no box")
            cxcywh = xyxy_to_normalized_cxcywh(action.box, (width, height))
            raw_mask = self.box_backend.segment_box(image, cxcywh, query)
            raw_mask = _validated_tool_mask(raw_mask, initial_mask.shape)
            x1, y1, x2, y2 = action.box
            mask = np.array(initial_mask, dtype=bool, copy=True)
            mask[y1 : y2 + 1, x1 : x2 + 1] = raw_mask[
                y1 : y2 + 1, x1 : x2 + 1
            ]
            roi = action.box
            composition_mode = "replace_box_roi_only"
            tool = "sam3"
            tool_input = {
                "authorized_region_id": box_region_id,
                "box_xyxy": list(action.box),
                "box_cxcywh_normalized": list(cxcywh),
            }
        else:
            raise ValueError("finish is handled by the environment, not an executor")

        result = np.asarray(mask, dtype=bool)
        if result.shape != initial_mask.shape:
            raise ValueError(f"tool returned mask shape {result.shape}, expected {initial_mask.shape}")
        backend = self.point_backend if tool == "simpleclick" else self.box_backend
        tool_result = getattr(backend, "last_evidence", None)
        changed_pixels = int(
            np.logical_xor(np.asarray(initial_mask, dtype=bool), result).sum()
        )
        evidence = {"tool": tool, "tool_input": tool_input}
        if tool == "simpleclick":
            evidence["point_authorization"] = {
                "region_id": point_region_id,
                "proposal_box_xyxy": (
                    list(point_proposal_box)
                    if point_proposal_box is not None
                    else None
                ),
                "proposal_component_area": point_proposal_component_area,
                "proposal_change_ratio": (
                    changed_pixels / point_proposal_component_area
                    if point_proposal_component_area is not None
                    else None
                ),
                "effective_roi_xyxy": list(roi),
                "model_input_scope_policy": session_localization_policy,
                "write_scope_policy": tool_input["write_scope_policy"],
                "scope_policy": (
                    "environment_proposal_clicked_component"
                    if point_proposal_box is not None
                    and action.action == "negative_point"
                    else "environment_proposal_local"
                    if point_proposal_box is not None
                    else "legacy_fixed_fraction"
                ),
            }
        else:
            evidence["box_authorization"] = {
                "region_id": box_region_id,
                "proposal_box_xyxy": list(action.box),
                "geometry_match": box_region_id is not None,
                "scope_policy": (
                    "environment_live_proposal"
                    if box_region_id is not None
                    else "legacy_action_box"
                ),
            }
        evidence.update(
            _locality_evidence(
                np.asarray(initial_mask, dtype=bool),
                result,
                roi,
                composition_mode,
            )
        )
        if tool_result:
            evidence["tool_result"] = tool_result
        return ExecutionResult(result, evidence)


def xyxy_to_normalized_cxcywh(
    box: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert pixel XYXY to SAM3 normalized center-size format."""

    x1, y1, x2, y2 = box
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
        raise ValueError("box must be ordered and inside the image")
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _point_roi(
    coordinate: tuple[int, int],
    image_size: tuple[int, int],
    fraction: float,
    *,
    proposal_box: tuple[int, int, int, int] | None = None,
    min_padding: int = 4,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x, y = coordinate
    if proposal_box is not None:
        x1, y1, x2, y2 = proposal_box
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ValueError("point proposal box must be ordered and inside the image")
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            raise ValueError("point coordinate must be inside its proposal box")
        return (
            max(0, min(x1, x - min_padding)),
            max(0, min(y1, y - min_padding)),
            min(width - 1, max(x2, x + min_padding)),
            min(height - 1, max(y2, y + min_padding)),
        )
    radius = max(2, round(min(width, height) * fraction))
    return (
        max(0, x - radius),
        max(0, y - radius),
        min(width - 1, x + radius),
        min(height - 1, y + radius),
    )


def _roi_mask(
    shape: tuple[int, int], roi: tuple[int, int, int, int]
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = roi
    result[y1 : y2 + 1, x1 : x2 + 1] = True
    return result


def _validated_tool_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.shape != expected_shape:
        raise ValueError(
            f"tool returned mask shape {result.shape}, expected {expected_shape}"
        )
    return result


def _component_containing(
    mask: np.ndarray, coordinate: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    x, y = coordinate
    component = np.zeros_like(source)
    if not (0 <= x < width and 0 <= y < height) or not source[y, x]:
        return component
    stack = [(x, y)]
    component[y, x] = True
    while stack:
        current_x, current_y = stack.pop()
        for next_x, next_y in (
            (current_x - 1, current_y),
            (current_x + 1, current_y),
            (current_x, current_y - 1),
            (current_x, current_y + 1),
        ):
            if (
                0 <= next_x < width
                and 0 <= next_y < height
                and source[next_y, next_x]
                and not component[next_y, next_x]
            ):
                component[next_y, next_x] = True
                stack.append((next_x, next_y))
    return component


def _locality_evidence(
    before: np.ndarray,
    after: np.ndarray,
    roi: tuple[int, int, int, int],
    composition_mode: str,
) -> dict[str, Any]:
    changed = np.logical_xor(before, after)
    x1, y1, x2, y2 = roi
    inside = np.zeros_like(changed)
    inside[y1 : y2 + 1, x1 : x2 + 1] = True
    changed_pixels = int(changed.sum())
    outside_pixels = int(np.logical_and(changed, ~inside).sum())
    inside_pixels = changed_pixels - outside_pixels
    before_components = _component_count(before)
    after_components = _component_count(after)
    largest_changed = max(
        (int(component.sum()) for component in _components(changed)),
        default=0,
    )
    return {
        "locality": {
            "composition_mode": composition_mode,
            "roi_xyxy": [x1, y1, x2, y2],
            "changed_pixels": changed_pixels,
            "inside_roi_pixels": inside_pixels,
            "outside_roi_pixels": outside_pixels,
            "local_changed_pixels": inside_pixels,
            "nonlocal_changed_pixels": outside_pixels,
            "outside_roi_ratio": (
                outside_pixels / changed_pixels if changed_pixels else 0.0
            ),
            "target_mask_change_ratio": changed_pixels / changed.size,
            "largest_changed_component_pixels": largest_changed,
            "components_before": before_components,
            "components_after": after_components,
            "component_count_delta": after_components - before_components,
        }
    }


def _components(mask: np.ndarray) -> list[np.ndarray]:
    source = np.asarray(mask, dtype=bool)
    remaining = np.array(source, copy=True)
    components: list[np.ndarray] = []
    while remaining.any():
        y, x = np.argwhere(remaining)[0]
        component = _component_containing(remaining, (int(x), int(y)))
        components.append(component)
        remaining[component] = False
    return components


def _component_count(mask: np.ndarray) -> int:
    return len(_components(mask))
