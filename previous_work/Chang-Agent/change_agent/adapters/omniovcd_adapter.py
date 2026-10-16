"""Stable mask-level boundary around OmniOVCD/SAM3 functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import numpy as np


@dataclass
class PairUpdate:
    t1_instances: tuple[np.ndarray, ...]
    t2_instances: tuple[np.ndarray, ...]
    matching: tuple[tuple[int, int], ...]
    change_mask: np.ndarray
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitializationResult:
    t1_mask: np.ndarray
    t2_mask: np.ndarray
    update: PairUpdate


class MaskPairProcessor:
    """Rebuild change masks with configurable OmniOVCD-style instance matching."""

    MODES = {"overlap_presence", "greedy_one_to_one"}

    def __init__(
        self,
        overlap_threshold: float = 0.25,
        min_instance_area: int | None = None,
        *,
        matching_mode: Literal["overlap_presence", "greedy_one_to_one"] = "overlap_presence",
        t12_min_instance_area: int = 0,
        cd_min_instance_area: int = 0,
    ):
        if not 0 <= overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if matching_mode not in self.MODES:
            raise ValueError(f"matching_mode must be one of {sorted(self.MODES)}")
        # Keep the original constructor keyword working while giving the two OmniOVCD
        # area filters distinct, explicit meanings.
        if min_instance_area is not None:
            if t12_min_instance_area != 0:
                raise ValueError("use either min_instance_area or t12_min_instance_area")
            t12_min_instance_area = min_instance_area
        if t12_min_instance_area < 0 or cd_min_instance_area < 0:
            raise ValueError("instance area thresholds must be non-negative")
        self.overlap_threshold = overlap_threshold
        self.matching_mode = matching_mode
        self.t12_min_instance_area = t12_min_instance_area
        self.cd_min_instance_area = cd_min_instance_area

    def rebuild(
        self,
        t1_mask: np.ndarray,
        t2_mask: np.ndarray,
        evidence: Mapping[str, Any] | None = None,
    ) -> PairUpdate:
        t1_mask = np.asarray(t1_mask, dtype=bool)
        t2_mask = np.asarray(t2_mask, dtype=bool)
        if t1_mask.shape != t2_mask.shape or t1_mask.ndim != 2:
            raise ValueError("T1/T2 masks must be same-shaped 2-D arrays")
        # OmniOVCD extracts every component. t12_min_instance_area only suppresses
        # small unmatched components from the change mask; it does not remove them
        # from overlap checks against the opposite view.
        t1_instances = connected_components(t1_mask)
        t2_instances = connected_components(t2_mask)
        candidates = self._candidate_pairs(t1_instances, t2_instances)
        if self.matching_mode == "overlap_presence":
            matching, matched_t1, matched_t2 = self._overlap_presence(candidates)
        else:
            matching = self._greedy_one_to_one(candidates)
            matched_t1 = {left for left, _ in matching}
            matched_t2 = {right for _, right in matching}
        change = np.zeros_like(t1_mask, dtype=bool)
        for index, instance in enumerate(t1_instances):
            if index not in matched_t1 and int(instance.sum()) >= self.t12_min_instance_area:
                change |= instance
        for index, instance in enumerate(t2_instances):
            if index not in matched_t2 and int(instance.sum()) >= self.t12_min_instance_area:
                change |= instance
        if self.cd_min_instance_area > 0:
            filtered = np.zeros_like(change)
            for instance in connected_components(change):
                if int(instance.sum()) >= self.cd_min_instance_area:
                    filtered |= instance
            change = filtered

        diagnostic_candidates = [
            {
                "t1_id": left,
                "t2_id": right,
                "t1_coverage": round(t1_coverage, 8),
                "t2_coverage": round(t2_coverage, 8),
                "coverage": round(max(t1_coverage, t2_coverage), 8),
            }
            for left, right, t1_coverage, t2_coverage in candidates
            if max(t1_coverage, t2_coverage) >= self.overlap_threshold
        ]
        t1_degree: dict[int, int] = {}
        t2_degree: dict[int, int] = {}
        for pair in diagnostic_candidates:
            t1_degree[pair["t1_id"]] = t1_degree.get(pair["t1_id"], 0) + 1
            t2_degree[pair["t2_id"]] = t2_degree.get(pair["t2_id"], 0) + 1
        matching_evidence = {
            "matching_mode": self.matching_mode,
            "overlap_threshold": self.overlap_threshold,
            "t12_min_instance_area": self.t12_min_instance_area,
            "cd_min_instance_area": self.cd_min_instance_area,
            "t1_instance_count": len(t1_instances),
            "t2_instance_count": len(t2_instances),
            "candidate_pairs": diagnostic_candidates,
            "split_merge_ambiguity": any(value > 1 for value in t1_degree.values())
            or any(value > 1 for value in t2_degree.values()),
        }
        merged_evidence = dict(evidence or {})
        merged_evidence["matching"] = matching_evidence
        return PairUpdate(
            t1_instances=t1_instances,
            t2_instances=t2_instances,
            matching=matching,
            change_mask=change,
            evidence=merged_evidence,
        )

    @staticmethod
    def _candidate_pairs(
        t1_instances: tuple[np.ndarray, ...],
        t2_instances: tuple[np.ndarray, ...],
    ) -> list[tuple[int, int, float, float]]:
        candidates: list[tuple[int, int, float, float]] = []
        for left, mask1 in enumerate(t1_instances):
            area1 = int(mask1.sum())
            for right, mask2 in enumerate(t2_instances):
                intersection = int(np.logical_and(mask1, mask2).sum())
                if not intersection:
                    continue
                area2 = int(mask2.sum())
                candidates.append((left, right, intersection / area1, intersection / area2))
        return candidates

    def _overlap_presence(
        self, candidates: list[tuple[int, int, float, float]]
    ) -> tuple[tuple[tuple[int, int], ...], set[int], set[int]]:
        matched_t1 = {
            left for left, _, coverage, _ in candidates if coverage >= self.overlap_threshold
        }
        matched_t2 = {
            right for _, right, _, coverage in candidates if coverage >= self.overlap_threshold
        }
        # The pair list is diagnostic: include a relationship when it supplies
        # presence evidence in either direction. Change construction uses the two
        # directional matched sets above, exactly as OmniOVCD does.
        matching = tuple(
            sorted(
                (left, right)
                for left, right, t1_coverage, t2_coverage in candidates
                if max(t1_coverage, t2_coverage) >= self.overlap_threshold
            )
        )
        return matching, matched_t1, matched_t2

    def _greedy_one_to_one(
        self, candidates: list[tuple[int, int, float, float]]
    ) -> tuple[tuple[int, int], ...]:
        ranked = [
            (min(t1_coverage, t2_coverage), left, right)
            for left, right, t1_coverage, t2_coverage in candidates
            if min(t1_coverage, t2_coverage) >= self.overlap_threshold
        ]
        matches: list[tuple[int, int]] = []
        used_left: set[int] = set()
        used_right: set[int] = set()
        for _, left, right in sorted(ranked, reverse=True):
            if left not in used_left and right not in used_right:
                matches.append((left, right))
                used_left.add(left)
                used_right.add(right)
        return tuple(sorted(matches))


def connected_components(mask: np.ndarray, min_area: int = 1) -> tuple[np.ndarray, ...]:
    """Extract 8-connected instances without requiring OpenCV/skimage."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= min_area:
            component = np.zeros_like(mask, dtype=bool)
            ys, xs = zip(*pixels)
            component[np.asarray(ys), np.asarray(xs)] = True
            components.append(component)
    return tuple(components)


class OmniOVCDAdapter:
    """Injectable adapter for expensive OmniOVCD initialization and SAM3 boxes.

    The callbacks are constructed inside the OmniOVCD environment so this package
    never imports incompatible CUDA stacks into the Agent process.
    """

    def __init__(
        self,
        initialize_masks: Callable[[np.ndarray, np.ndarray, str], Any],
        segment_box_callback: Callable[
            [np.ndarray, tuple[float, float, float, float], str], Any
        ],
        pair_processor: MaskPairProcessor | None = None,
    ):
        self.initialize_masks = initialize_masks
        self.segment_box_callback = segment_box_callback
        self.pair_processor = pair_processor or MaskPairProcessor()

    def initialize(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> InitializationResult:
        raw = self.initialize_masks(t1_image, t2_image, query)
        if isinstance(raw, Mapping):
            t1_mask, t2_mask = raw["t1_mask"], raw["t2_mask"]
            evidence = dict(raw.get("evidence", {}))
        elif isinstance(raw, tuple) and len(raw) in {2, 3}:
            t1_mask, t2_mask = raw[:2]
            evidence = dict(raw[2]) if len(raw) == 3 else {}
        else:
            raise TypeError("initializer must return a mapping or (t1_mask, t2_mask[, evidence])")
        t1_mask = np.asarray(t1_mask, dtype=bool)
        t2_mask = np.asarray(t2_mask, dtype=bool)
        update = self.pair_processor.rebuild(t1_mask, t2_mask, evidence)
        return InitializationResult(t1_mask=t1_mask, t2_mask=t2_mask, update=update)

    def rebuild(
        self, t1_mask: np.ndarray, t2_mask: np.ndarray, evidence: Mapping[str, Any]
    ) -> PairUpdate:
        return self.pair_processor.rebuild(t1_mask, t2_mask, evidence)

    def segment_box(
        self,
        image: np.ndarray,
        box_cxcywh_normalized: tuple[float, float, float, float],
        query: str,
    ) -> np.ndarray:
        return np.asarray(
            self.segment_box_callback(image, box_cxcywh_normalized, query), dtype=bool
        )
