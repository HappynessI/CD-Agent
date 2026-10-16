"""Environment-owned local region proposals for Change Verifier inspection."""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters.omniovcd_adapter import connected_components
from .coordinates import pixel_box_to_normalized, pixel_point_to_normalized
from .state import ChangeState


EMPTY_MASK_SCAN_PROPOSAL_VERSION = "qwen3vl_empty_dual_mask_scan_grid_v1"


def attach_verifier_regions(
    state: ChangeState,
    previous_state: ChangeState | None = None,
    *,
    max_regions: int = 6,
    max_delta_regions: int = 3,
    min_component_area: int = 4,
    padding_ratio: float = 0.25,
    empty_mask_scan_grid_size: int = 0,
) -> list[dict[str, Any]]:
    """Attach deterministic full-coverage proposals and exact mask facts to ``state``.

    Proposals come from current change components, T1/T2 semantic-mask difference,
    and (for edited candidates) the change-mask delta against the previous accepted
    state. The Verifier selects/classifies these proposals; it does not invent boxes.
    """

    if not 0 <= empty_mask_scan_grid_size <= 16:
        raise ValueError("empty_mask_scan_grid_size must be within [0,16]")
    empty_dual_mask_scan = bool(
        previous_state is None
        and empty_mask_scan_grid_size > 0
        and not np.asarray(state.t1_mask, dtype=bool).any()
        and not np.asarray(state.t2_mask, dtype=bool).any()
    )
    proposals = (
        build_empty_mask_scan_regions(
            state,
            grid_size=empty_mask_scan_grid_size,
            max_regions=max_regions,
        )
        if empty_dual_mask_scan
        else build_verifier_regions(
            state,
            None,
            max_regions=max_regions,
            min_component_area=min_component_area,
            padding_ratio=padding_ratio,
        )
        if previous_state is None
        else build_candidate_delta_regions(
            state,
            previous_state,
            max_regions=max_delta_regions,
            min_component_area=min_component_area,
            padding_ratio=padding_ratio,
        )
    )
    temporal_difference = np.logical_xor(state.t1_mask, state.t2_mask)
    candidate_delta = (
        np.zeros_like(state.change_mask)
        if previous_state is None
        else np.logical_xor(previous_state.change_mask, state.change_mask)
    )
    state.evidence = dict(state.evidence)
    state.evidence["verifier_region_proposals"] = proposals
    state.evidence["verifier_mask_facts"] = {
        "height": int(state.change_mask.shape[0]),
        "width": int(state.change_mask.shape[1]),
        "change_pixels": int(state.change_mask.sum()),
        "change_area_ratio": float(state.change_mask.mean()),
        "t1_mask_pixels": int(state.t1_mask.sum()),
        "t2_mask_pixels": int(state.t2_mask.sum()),
        "temporal_difference_pixels": int(temporal_difference.sum()),
        "candidate_delta_pixels": int(candidate_delta.sum()),
        "candidate_added_pixels": int(
            np.logical_and(state.change_mask, ~previous_state.change_mask).sum()
        )
        if previous_state is not None
        else 0,
        "candidate_removed_pixels": int(
            np.logical_and(previous_state.change_mask, ~state.change_mask).sum()
        )
        if previous_state is not None
        else 0,
        "candidate_delta_covered_pixels": int(
            sum(item["component_area"] for item in proposals)
        )
        if previous_state is not None
        else 0,
        "proposal_count": len(proposals),
        "proposal_config": {
            "schema_version": "full_component_batched_v5",
            "max_regions": max_regions if previous_state is None else max_delta_regions,
            "max_regions_per_batch": (
                max_regions if previous_state is None else max_delta_regions
            ),
            "total_component_count": len(proposals),
            "batch_count": (
                0 if not proposals else (
                    (len(proposals) + max_regions - 1) // max_regions
                    if previous_state is None
                    else (len(proposals) + max_delta_regions - 1)
                    // max_delta_regions
                )
            ),
            "component_filtering": "disabled_full_coverage",
            "legacy_min_component_area_argument": min_component_area,
            "padding_ratio": padding_ratio,
            "empty_mask_scan_grid_size": empty_mask_scan_grid_size,
            "empty_dual_mask_scan": empty_dual_mask_scan,
            "empty_mask_scan_proposal_version": (
                EMPTY_MASK_SCAN_PROPOSAL_VERSION
                if empty_dual_mask_scan
                else None
            ),
        },
    }
    if previous_state is None:
        audit_mask = (
            np.ones_like(state.change_mask, dtype=bool)
            if empty_dual_mask_scan
            else np.logical_or(
                state.change_mask,
                np.logical_and(temporal_difference, ~state.change_mask),
            )
        )
        covered_mask = np.zeros_like(audit_mask)
        for proposal in proposals:
            audit_kind = proposal.get("audit_kind")
            if audit_kind == "scan":
                x1, y1, x2, y2 = (
                    int(value) for value in proposal["box_pixels"]
                )
                covered_mask[y1 : y2 + 1, x1 : x2 + 1] = True
                continue
            source = (
                state.change_mask
                if audit_kind == "present"
                else np.logical_and(temporal_difference, ~state.change_mask)
                if audit_kind == "missing"
                else audit_mask
            )
            covered_mask |= _component_containing_seed(
                source, proposal["component_seed_pixels"]
            )
        covered_audit = np.logical_and(audit_mask, covered_mask)
        uncovered_audit = np.logical_and(audit_mask, ~covered_mask)
        total = int(audit_mask.sum())
        uncovered_box = None
        if uncovered_audit.any():
            largest = max(
                connected_components(uncovered_audit), key=lambda item: int(item.sum())
            )
            uncovered_box = list(
                pixel_box_to_normalized(_mask_box(largest), state.image_size)
            )
        state.evidence["verifier_mask_facts"].update(
            {
                "initial_audit_pixels": total,
                "initial_audit_covered_pixels": int(covered_audit.sum()),
                "initial_audit_uncovered_pixels": int(uncovered_audit.sum()),
                "initial_audit_coverage_ratio": (
                    int(covered_audit.sum()) / total if total else 1.0
                ),
                "initial_audit_uncovered_box_normalized": uncovered_box,
            }
        )
    else:
        covered = int(state.evidence["verifier_mask_facts"]["candidate_delta_covered_pixels"])
        total = int(candidate_delta.sum())
        state.evidence["verifier_mask_facts"]["candidate_delta_uncovered_pixels"] = max(
            0, total - covered
        )
        state.evidence["verifier_mask_facts"]["candidate_delta_coverage_ratio"] = (
            covered / total if total else 1.0
        )
    return proposals


def build_empty_mask_scan_regions(
    state: ChangeState,
    *,
    grid_size: int = 3,
    max_regions: int = 6,
) -> list[dict[str, Any]]:
    """Tile the full frame when both temporal object masks are empty.

    The tiles are deterministic Environment geometry. The Verifier may diagnose
    a black tile as a false negative, but it cannot invent a point or box.
    """

    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if np.asarray(state.t1_mask, dtype=bool).any() or np.asarray(
        state.t2_mask, dtype=bool
    ).any():
        raise ValueError("empty-mask scan requires both temporal masks to be empty")
    if np.asarray(state.change_mask, dtype=bool).any():
        raise ValueError("empty-mask scan requires the change mask to be empty")
    height, width = state.change_mask.shape
    if grid_size > min(height, width):
        raise ValueError("grid_size cannot exceed the image dimensions")

    result: list[dict[str, Any]] = []
    for row in range(grid_size):
        y1 = row * height // grid_size
        y2 = (row + 1) * height // grid_size - 1
        for column in range(grid_size):
            x1 = column * width // grid_size
            x2 = (column + 1) * width // grid_size - 1
            seed_x = (x1 + x2) // 2
            seed_y = (y1 + y2) // 2
            index = row * grid_size + column
            box = (x1, y1, x2, y2)
            result.append(
                {
                    "region_id": f"s{index}",
                    "proposal_version": EMPTY_MASK_SCAN_PROPOSAL_VERSION,
                    "batch_index": index // max_regions,
                    "sources": ["empty_dual_mask_full_frame_scan"],
                    "audit_kind": "scan",
                    "component_area": (x2 - x1 + 1) * (y2 - y1 + 1),
                    "component_seed_pixels": [seed_x, seed_y],
                    "component_seed_normalized": list(
                        pixel_point_to_normalized(
                            (seed_x, seed_y), (width, height)
                        )
                    ),
                    "box_pixels": list(box),
                    "box_normalized": list(
                        pixel_box_to_normalized(box, (width, height))
                    ),
                    "change_pixels": 0,
                    "temporal_difference_pixels": 0,
                    "candidate_delta_pixels": 0,
                    "candidate_added_pixels": 0,
                    "candidate_removed_pixels": 0,
                    "t1_mask_pixels": 0,
                    "t2_mask_pixels": 0,
                    "component_t1_mask_pixels": 0,
                    "component_t2_mask_pixels": 0,
                    "component_seed_t1_mask_white": False,
                    "component_seed_t2_mask_white": False,
                    "scan_tile_row": row,
                    "scan_tile_column": column,
                }
            )
    return result


def build_candidate_delta_regions(
    state: ChangeState,
    previous_state: ChangeState,
    *,
    max_regions: int = 3,
    min_component_area: int = 1,
    padding_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Build compact, polarity-preserving panels for a candidate edit.

    Point-tool output is already clipped to one authoritative action ROI.  In that
    case all same-polarity fragments are aggregated into one complete action-delta
    record so the verifier judges the edit as a whole instead of making many calls
    for one-pixel fragments.  Generic/offline callers without execution locality
    retain component-separated records.  Exact delta coverage remains auditable in
    both cases. ``min_component_area`` is retained only for archived callers;
    full-coverage v11 deliberately does not filter small components.
    """

    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    change = np.asarray(state.change_mask, dtype=bool)
    previous = np.asarray(previous_state.change_mask, dtype=bool)
    candidate_added = np.logical_and(change, ~previous)
    candidate_removed = np.logical_and(previous, ~change)
    temporal_difference = np.logical_xor(state.t1_mask, state.t2_mask)
    action_scope = _candidate_action_scope(state, change.shape)
    raw: list[tuple[str, np.ndarray, bool, int]] = []
    for effect_kind, mask in (
        ("added", candidate_added),
        ("removed", candidate_removed),
    ):
        components = sorted(
            connected_components(mask), key=lambda item: int(item.sum()), reverse=True
        )
        if action_scope is not None and components:
            aggregate = np.logical_or.reduce(components)
            raw.append((effect_kind, aggregate, True, len(components)))
        else:
            raw.extend(
                (effect_kind, component, False, 1) for component in components
            )
    raw.sort(key=lambda item: int(item[1].sum()), reverse=True)

    height, width = change.shape
    result: list[dict[str, Any]] = []
    for index, (effect_kind, component, action_scoped, component_count) in enumerate(raw):
        action_coordinate = action_scope[1] if action_scope is not None else None
        if (
            action_coordinate is not None
            and component[action_coordinate[1], action_coordinate[0]]
        ):
            seed_x, seed_y = action_coordinate
        else:
            seed_x, seed_y = _distance_transform_seed(component)
        crop_box = _padded_box(_mask_box(component), (height, width), padding_ratio)
        if action_scope is not None:
            roi = action_scope[0]
            crop_box = (
                min(crop_box[0], roi[0]),
                min(crop_box[1], roi[1]),
                max(crop_box[2], roi[2]),
                max(crop_box[3], roi[3]),
            )
        x1, y1, x2, y2 = crop_box
        crop = np.zeros_like(change)
        crop[y1 : y2 + 1, x1 : x2 + 1] = True
        transition_mask_facts = {
            "previous_t1_crop_pixels": int(
                np.logical_and(previous_state.t1_mask, crop).sum()
            ),
            "previous_t2_crop_pixels": int(
                np.logical_and(previous_state.t2_mask, crop).sum()
            ),
            "candidate_t1_crop_pixels": int(
                np.logical_and(state.t1_mask, crop).sum()
            ),
            "candidate_t2_crop_pixels": int(
                np.logical_and(state.t2_mask, crop).sum()
            ),
            "previous_t1_delta_pixels": int(
                np.logical_and(previous_state.t1_mask, component).sum()
            ),
            "previous_t2_delta_pixels": int(
                np.logical_and(previous_state.t2_mask, component).sum()
            ),
            "candidate_t1_delta_pixels": int(
                np.logical_and(state.t1_mask, component).sum()
            ),
            "candidate_t2_delta_pixels": int(
                np.logical_and(state.t2_mask, component).sum()
            ),
            "previous_seed_t1_white": bool(previous_state.t1_mask[seed_y, seed_x]),
            "previous_seed_t2_white": bool(previous_state.t2_mask[seed_y, seed_x]),
            "candidate_seed_t1_white": bool(state.t1_mask[seed_y, seed_x]),
            "candidate_seed_t2_white": bool(state.t2_mask[seed_y, seed_x]),
            "action_scoped_delta": action_scoped,
            "delta_component_count": component_count,
        }
        result.append(
            {
                "region_id": f"d{index}",
                "batch_index": index // max_regions,
                "effect_kind": effect_kind,
                "sources": [f"candidate_{effect_kind}"],
                "component_area": int(component.sum()),
                "component_seed_pixels": [int(seed_x), int(seed_y)],
                "component_seed_normalized": list(
                    pixel_point_to_normalized((int(seed_x), int(seed_y)), (width, height))
                ),
                "box_pixels": list(crop_box),
                "box_normalized": list(
                    pixel_box_to_normalized(crop_box, (width, height))
                ),
                "change_pixels": int(np.logical_and(change, crop).sum()),
                "temporal_difference_pixels": int(
                    np.logical_and(temporal_difference, crop).sum()
                ),
                "candidate_delta_pixels": int(component.sum()),
                "delta_pixels": int(component.sum()),
                "candidate_added_pixels": int(component.sum())
                if effect_kind == "added"
                else 0,
                "candidate_removed_pixels": int(component.sum())
                if effect_kind == "removed"
                else 0,
                "t1_mask_pixels": int(np.logical_and(state.t1_mask, crop).sum()),
                "t2_mask_pixels": int(np.logical_and(state.t2_mask, crop).sum()),
                "component_t1_mask_pixels": int(
                    np.logical_and(state.t1_mask, component).sum()
                ),
                "component_t2_mask_pixels": int(
                    np.logical_and(state.t2_mask, component).sum()
                ),
                "component_seed_t1_mask_white": bool(state.t1_mask[seed_y, seed_x]),
                "component_seed_t2_mask_white": bool(state.t2_mask[seed_y, seed_x]),
                "transition_mask_facts": transition_mask_facts,
            }
        )
    return result


def _candidate_action_scope(
    state: ChangeState, shape: tuple[int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int] | None] | None:
    """Return validated Environment-owned point/box locality for a candidate."""

    locality = state.evidence.get("locality")
    if not isinstance(locality, dict):
        return None
    mode = str(locality.get("composition_mode", ""))
    if mode not in {
        "merge_clicked_prediction_component",
        "merge_clicked_prediction_component_within_proposal_roi",
        "remove_simpleclick_delta_within_roi",
        "remove_simpleclick_delta_within_proposal_roi",
        "remove_simpleclick_delta_within_clicked_component",
        "replace_box_roi_only",
    }:
        return None
    roi = locality.get("roi_xyxy")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return None
    height, width = shape
    x1, y1, x2, y2 = (int(value) for value in roi)
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        return None
    coordinate: tuple[int, int] | None = None
    tool_input = state.evidence.get("tool_input")
    if isinstance(tool_input, dict):
        raw_coordinate = tool_input.get("coordinate")
        if isinstance(raw_coordinate, (list, tuple)) and len(raw_coordinate) == 2:
            x, y = (int(value) for value in raw_coordinate)
            if 0 <= x < width and 0 <= y < height:
                coordinate = (x, y)
    return (x1, y1, x2, y2), coordinate


def build_verifier_regions(
    state: ChangeState,
    previous_state: ChangeState | None = None,
    *,
    max_regions: int = 6,
    min_component_area: int = 4,
    padding_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    """Return every initial audit component, grouped by ``max_regions`` per batch.

    ``min_component_area`` is a legacy compatibility argument. V11 keeps every
    component, including a single pixel, so completion coverage cannot hide it.
    """

    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative")

    change = np.asarray(state.change_mask, dtype=bool)
    temporal_difference = np.logical_xor(state.t1_mask, state.t2_mask)
    missing_from_change = np.logical_and(temporal_difference, ~change)
    candidate_added = np.zeros_like(change)
    candidate_removed = np.zeros_like(change)
    if previous_state is not None:
        candidate_added = np.logical_and(change, ~previous_state.change_mask)
        candidate_removed = np.logical_and(previous_state.change_mask, ~change)

    source_masks = [
        ("candidate_added", candidate_added, 0),
        ("candidate_removed", candidate_removed, 0),
        ("change_component", change, 1),
        ("temporal_difference_missing", missing_from_change, 2),
        ("temporal_difference", temporal_difference, 3),
    ]
    raw: list[dict[str, Any]] = []
    for source, source_mask, priority in source_masks:
        components = sorted(
            connected_components(source_mask),
            key=lambda item: int(item.sum()),
            reverse=True,
        )
        for component in components:
            raw.append(
                {
                    "source": source,
                    "priority": priority,
                    "area": int(component.sum()),
                    "component": component,
                    "box": _mask_box(component),
                }
            )

    raw.sort(key=lambda item: (item["priority"], -item["area"]))
    selected: list[dict[str, Any]] = []
    for item in raw:
        duplicate = next(
            (
                existing
                for existing in selected
                if _box_overlap_fraction(existing["box"], item["box"]) >= 0.8
            ),
            None,
        )
        if duplicate is not None:
            duplicate["sources"].add(item["source"])
            duplicate["component_area"] = max(
                duplicate["component_area"], item["area"]
            )
            continue
        selected.append(
            {
                "box": item["box"],
                "sources": {item["source"]},
                "component_area": item["area"],
                "component": item["component"],
                "primary_source": item["source"],
            }
        )

    height, width = change.shape
    result: list[dict[str, Any]] = []
    candidate_delta = np.logical_or(candidate_added, candidate_removed)
    for index, item in enumerate(selected):
        seed_x, seed_y = _distance_transform_seed(item["component"])
        crop_box = _padded_box(item["box"], (height, width), padding_ratio)
        x1, y1, x2, y2 = crop_box
        crop = np.zeros_like(change)
        crop[y1 : y2 + 1, x1 : x2 + 1] = True
        result.append(
            {
                "region_id": f"r{index}",
                "batch_index": index // max_regions,
                "sources": sorted(item["sources"]),
                "audit_kind": (
                    "present"
                    if item["primary_source"] == "change_component"
                    else "missing"
                    if item["primary_source"] == "temporal_difference_missing"
                    else "mixed"
                ),
                "component_area": int(item["component_area"]),
                "component_seed_pixels": [int(seed_x), int(seed_y)],
                "component_seed_normalized": list(
                    pixel_point_to_normalized((int(seed_x), int(seed_y)), (width, height))
                ),
                "box_pixels": list(crop_box),
                "box_normalized": list(
                    pixel_box_to_normalized(crop_box, (width, height))
                ),
                "change_pixels": int(np.logical_and(change, crop).sum()),
                "temporal_difference_pixels": int(
                    np.logical_and(temporal_difference, crop).sum()
                ),
                "candidate_delta_pixels": int(
                    np.logical_and(candidate_delta, crop).sum()
                ),
                "candidate_added_pixels": int(
                    np.logical_and(candidate_added, crop).sum()
                ),
                "candidate_removed_pixels": int(
                    np.logical_and(candidate_removed, crop).sum()
                ),
                "t1_mask_pixels": int(np.logical_and(state.t1_mask, crop).sum()),
                "t2_mask_pixels": int(np.logical_and(state.t2_mask, crop).sum()),
                "component_t1_mask_pixels": int(
                    np.logical_and(state.t1_mask, item["component"]).sum()
                ),
                "component_t2_mask_pixels": int(
                    np.logical_and(state.t2_mask, item["component"]).sum()
                ),
                "component_seed_t1_mask_white": bool(state.t1_mask[seed_y, seed_x]),
                "component_seed_t2_mask_white": bool(state.t2_mask[seed_y, seed_x]),
            }
        )
    return result


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot build a box for an empty component")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _distance_transform_seed(mask: np.ndarray) -> tuple[int, int]:
    """Return stable XY argmax of an 8-neighbour interior distance transform.

    Repeated binary erosion is equivalent to a discrete distance transform for
    this purpose: pixels surviving the most erosion rounds are farthest from
    the component boundary.  The implementation keeps proposal generation
    dependency-free and ``np.argmax`` gives deterministic row-major tie-breaks.
    """

    source = np.asarray(mask, dtype=bool)
    if not source.any():
        raise ValueError("cannot choose a seed for an empty component")
    remaining = np.array(source, copy=True)
    distance = np.zeros(source.shape, dtype=np.int32)
    while remaining.any():
        distance[remaining] += 1
        padded = np.pad(remaining, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(remaining)
        for offset_y in range(3):
            for offset_x in range(3):
                eroded &= padded[
                    offset_y : offset_y + remaining.shape[0],
                    offset_x : offset_x + remaining.shape[1],
                ]
        remaining = eroded
    seed_y, seed_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(seed_x), int(seed_y)


def _component_containing_seed(
    mask: np.ndarray, seed_xy: list[int] | tuple[int, int]
) -> np.ndarray:
    seed_x, seed_y = (int(value) for value in seed_xy)
    component = next(
        (
            item
            for item in connected_components(np.asarray(mask, dtype=bool))
            if item[seed_y, seed_x]
        ),
        None,
    )
    if component is None:
        raise ValueError("proposal seed is outside its source component")
    return component


def _padded_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    height, width = shape
    padding = max(4, round(max(x2 - x1 + 1, y2 - y1 + 1) * padding_ratio))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    )


def _box_overlap_fraction(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1) + 1) * max(
        0, min(ly2, ry2) - max(ly1, ry1) + 1
    )
    if not intersection:
        return 0.0
    left_area = (lx2 - lx1 + 1) * (ly2 - ly1 + 1)
    right_area = (rx2 - rx1 + 1) * (ry2 - ry1 + 1)
    return intersection / min(left_area, right_area)
