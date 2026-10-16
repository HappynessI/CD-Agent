#!/usr/bin/env python3
"""Run the GT-direct, no-Qwen, teacher-only tool upper-bound pilot.

This runner deliberately differs from the deployable 0803 oracle.  It opens the
change label only after a fresh SAM3 initialization, derives point/box geometry
from current FN/FP change-error components, and injects that geometry into a real
``ChangeAgentEnvironment``.  Raw label pixels are never passed to the runtime
factory, Environment, Executor, SAM3, or SimpleClick.  The complete GT-derived
teacher proposal is retained in Environment evidence for audit.  The Executor
receives action geometry, Proposal ROI/area, and a teacher-encoded authorized
region ID; the SAM3/SimpleClick backends receive only the real-call geometry and,
for SimpleClick, the current-mask context localized by that ROI.

Four tracks are adaptive and independent: ``matched_safe`` first, then
``wide_safe``, ``wide_hard``, and ``wide_raw`` only while every earlier track
misses the endpoint.  Safe tracks require production hard and RGB gates; hard
ignores only RGB; raw is a metric-only counterfactual.  Every selected action is
executed twice and all T1/T2/change mask hashes must match before its state may be
carried forward.  Artifacts are permanently teacher-only and never authorize SFT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # Direct ``python tools/...`` execution.
    import run_qwen3vl_oracle_trajectory as base
except ModuleNotFoundError:  # Import through a test/module loader from repo root.
    from tools import run_qwen3vl_oracle_trajectory as base

from change_agent.adapters.omniovcd_adapter import connected_components
from change_agent.environment import ChangeAgentEnvironment
from change_agent.state import AgentAction, ChangeState
from change_agent.trajectory import mask_sha256
from change_agent.verifier_regions import _distance_transform_seed


CONFIG_SCHEMA_ID = "qwen3vl_gt_direct_teacher_preregistration_v1"
EPISODE_SCHEMA_ID = "qwen3vl_gt_direct_teacher_episode_v1"
INDEX_SCHEMA_ID = "qwen3vl_gt_direct_teacher_shard_index_v1"
COMPLETE_SCHEMA_ID = "qwen3vl_gt_direct_teacher_generation_complete_v1"
DIAGNOSTIC_SMOKE_EPISODE_SCHEMA_ID = (
    "qwen3vl_gt_direct_teacher_diagnostic_smoke_episode_v1"
)
DIAGNOSTIC_SMOKE_INDEX_SCHEMA_ID = (
    "qwen3vl_gt_direct_teacher_diagnostic_smoke_index_v1"
)
DIAGNOSTIC_SMOKE_COMPLETE_SCHEMA_ID = (
    "qwen3vl_gt_direct_teacher_diagnostic_smoke_complete_v1"
)
TRACK_NAMES = ("matched_safe", "wide_safe", "wide_hard", "wide_raw")
FORMAL_SAMPLE_IDS = (
    "train_407_r3c2",
    "train_364_r0c2",
    "val_43_r1c1",
    "val_41_r3c2",
)
MAX_STEPS = 3
MATCHED_MAX_CANDIDATES = 4
WIDE_MAX_CANDIDATES = 96
MATCHED_PADDING_RATIO = 0.25
WIDE_PADDING_RATIOS = (0.0, 0.1, 0.25, 0.5)
MAX_ERROR_COMPONENTS = 4
DIAGNOSTIC_SMOKE_SAMPLE_ID = "train_407_r3c2"
DIAGNOSTIC_SMOKE_STEP_INDEX = 1
DIAGNOSTIC_SMOKE_MAX_REAL_CANDIDATES = 2
EXPECTED_CHANGE_COMPOSITION = {
    "implementation": "MaskPairProcessor",
    "mode": "overlap_presence",
    "overlap_threshold": 0.25,
    "t12_min_instance_area": 0,
    "cd_min_instance_area": 0,
}
RUNTIME_BOUNDARY_LAYERS = {
    "environment": {
        "complete_gt_derived_proposal_metadata_present": True,
        "raw_label_pixels_or_path_present": False,
    },
    "executor_and_receipt": {
        "gt_derived_fields": [
            "point_coordinate_or_box",
            "proposal_box",
            "component_area",
            "authorized_region_id_encoding_teacher_route",
        ],
        "raw_label_pixels_or_path_present": False,
    },
    "sam3_backend": {
        "gt_derived_fields": ["box"],
        "raw_label_pixels_or_path_present": False,
    },
    "simpleclick_backend": {
        "gt_derived_fields": [
            "point_coordinate",
            "proposal_localized_current_mask_context",
        ],
        "raw_label_pixels_or_path_present": False,
    },
}


@dataclass(frozen=True)
class TeacherContract:
    base_config_path: Path
    base_manifest: Mapping[str, Any]
    ordered_samples: tuple[str, ...]
    expected_initial_hashes: Mapping[str, Mapping[str, str]]
    max_steps: int
    matched_max_candidates: int
    wide_max_candidates: int
    max_error_components: int
    matched_padding_ratio: float
    wide_padding_ratios: tuple[float, ...]
    endpoint_iou: float
    endpoint_f1: float
    endpoint_empty_fp: int
    comparison_epsilon: float


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _identity_path(value: Any, field: str, config_path: Path) -> Path:
    item = _as_mapping(value, field)
    raw_path = item.get("path")
    expected = item.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field}.path must be a non-empty string")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{field}.sha256 must be a full SHA256")
    path = Path(raw_path)
    if not path.is_absolute():
        repo_candidate = Path(__file__).resolve().parents[1] / path
        config_candidate = config_path.parent / path
        path = repo_candidate if repo_candidate.exists() else config_candidate
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{field}.path is not a file: {path}")
    if base.sha256_file(path) != expected.lower():
        raise ValueError(f"{field} SHA256 differs")
    return path


def _exact_hashes(value: Any, field: str) -> dict[str, str]:
    item = _as_mapping(value, field)
    if set(item) != {"t1", "t2", "change"}:
        raise ValueError(f"{field} must contain exactly t1/t2/change")
    result: dict[str, str] = {}
    for name in ("t1", "t2", "change"):
        digest = item[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError(f"{field}.{name} must be a full SHA256")
        result[name] = digest.lower()
    return result


def load_contract(config_path: Path) -> tuple[dict[str, Any], TeacherContract]:
    """Read and freeze the small 0804 contract plus immutable 0803 base config."""

    config_path = config_path.resolve()
    manifest = base.read_strict_json(config_path)
    if manifest.get("schema_id") != CONFIG_SCHEMA_ID:
        raise ValueError(f"schema_id must be {CONFIG_SCHEMA_ID}")
    if manifest.get("teacher_only") is not True:
        raise ValueError("top-level teacher_only must be true")
    if manifest.get("training_authorized") is not False:
        raise ValueError("top-level training_authorized must be false")
    scope = _as_mapping(
        manifest.get("scope_and_access_audit"), "scope_and_access_audit"
    )
    if scope.get("teacher_only") is not True:
        raise ValueError("teacher_only must be true")
    if scope.get("training_authorized") is not False:
        raise ValueError("training_authorized must be false")
    if scope.get("raw_label_pixels_passed_to_tools") is not False:
        raise ValueError("raw label pixels must be forbidden at the tool boundary")
    if scope.get("gt_derived_geometry_passed_to_environment") is not True:
        raise ValueError("teacher geometry must be authorized in Environment")
    if scope.get("gt_derived_geometry_passed_to_tools") is not True:
        raise ValueError("teacher geometry must reach real tools")
    base_path = _identity_path(
        manifest.get("base_config_0803"), "base_config_0803", config_path
    )
    base_manifest = base.read_strict_json(base_path)
    if base_manifest.get("schema_id") != base.PILOT_SCHEMA_ID:
        raise ValueError("base_config_0803 has the wrong schema")
    # Keep production model/tool/gate construction anchored to the validated 0803
    # object.  New teacher fields never enter ProductionRuntimeFactory.
    base._validate_frozen_runtime_contract(base_manifest)

    ordered = manifest.get("ordered_samples")
    if not isinstance(ordered, list) or set(ordered) != set(FORMAL_SAMPLE_IDS):
        raise ValueError("ordered_samples must contain exactly the four formal episodes")
    if len(ordered) != len(set(ordered)):
        raise ValueError("ordered_samples contains duplicates")
    base_specs = base._pilot_sample_specs(base_manifest)
    if any(sample_id not in base_specs for sample_id in ordered):
        raise ValueError("ordered_samples contains an episode absent from 0803 base config")

    raw_hashes = _as_mapping(
        manifest.get("expected_initial_state_mask_sha256"),
        "expected_initial_state_mask_sha256",
    )
    if set(raw_hashes) != set(ordered):
        raise ValueError("expected initial hashes must cover exactly ordered_samples")
    initial_hashes = {
        sample_id: _exact_hashes(
            raw_hashes[sample_id],
            f"expected_initial_state_mask_sha256.{sample_id}",
        )
        for sample_id in ordered
    }

    protocol = _as_mapping(manifest.get("execution_protocol"), "execution_protocol")
    if int(protocol.get("max_steps", -1)) != MAX_STEPS:
        raise ValueError("execution_protocol.max_steps must equal 3")
    qwen_policy = protocol.get("qwen_policy", {"enabled": False})
    if isinstance(qwen_policy, Mapping) and qwen_policy.get("enabled") is not False:
        raise ValueError("Qwen must be disabled")
    composition = dict(
        _as_mapping(
            protocol.get("change_composition"),
            "execution_protocol.change_composition",
        )
    )
    if composition != EXPECTED_CHANGE_COMPOSITION:
        raise ValueError(
            "change composition must freeze production MaskPairProcessor "
            "overlap_presence settings"
        )
    base_matching = dict(
        _as_mapping(base_manifest.get("matching"), "base_config_0803.matching")
    )
    if {
        key: composition[key]
        for key in (
            "mode",
            "overlap_threshold",
            "t12_min_instance_area",
            "cd_min_instance_area",
        )
    } != base_matching:
        raise ValueError("teacher change composition differs from bound 0803 runtime")

    geometry = _as_mapping(manifest.get("teacher_geometry"), "teacher_geometry")
    matched = _as_mapping(geometry.get("matched"), "teacher_geometry.matched")
    wide = _as_mapping(geometry.get("wide"), "teacher_geometry.wide")
    max_components = int(
        geometry.get(
            "max_error_components",
            matched.get("max_error_components", wide.get("max_error_components", -1)),
        )
    )
    if max_components != MAX_ERROR_COMPONENTS:
        raise ValueError("teacher geometry max_error_components must equal 4")
    matched_padding = float(matched.get("padding_ratio", -1))
    matched_cap = int(matched.get("max_candidates_per_step", -1))
    wide_paddings = tuple(float(item) for item in wide.get("padding_ratios", ()))
    wide_cap = int(wide.get("max_candidates_per_step", -1))
    if matched_padding != MATCHED_PADDING_RATIO or matched_cap != MATCHED_MAX_CANDIDATES:
        raise ValueError("matched geometry must freeze padding=0.25 and cap=4")
    if wide_paddings != WIDE_PADDING_RATIOS or wide_cap != WIDE_MAX_CANDIDATES:
        raise ValueError("wide geometry must freeze paddings [0,0.1,0.25,0.5] and cap=96")

    base_metric = _as_mapping(
        base_manifest.get("metric_and_success_contract", {}),
        "base metric_and_success_contract",
    )
    endpoint = _as_mapping(manifest.get("endpoint"), "endpoint")
    epsilon = float(base_metric.get("comparison_epsilon", 1e-12))
    endpoint_iou = float(
        endpoint.get("nonempty_gt_minimum_iou", base.DEFAULT_ENDPOINT_IOU)
    )
    endpoint_f1 = float(
        endpoint.get("nonempty_gt_minimum_f1", base.DEFAULT_ENDPOINT_F1)
    )
    endpoint_empty_fp = int(
        endpoint.get("empty_gt_maximum_fp", base.DEFAULT_EMPTY_GT_MAX_FP)
    )
    if not (0 < endpoint_iou <= 1 and 0 < endpoint_f1 <= 1 and epsilon >= 0):
        raise ValueError("metric endpoint contract is invalid")
    if not 0 <= endpoint_empty_fp < base.PATCH_SIZE * base.PATCH_SIZE:
        raise ValueError("empty-GT endpoint is invalid")

    contract = TeacherContract(
        base_config_path=base_path,
        base_manifest=base_manifest,
        ordered_samples=tuple(str(item) for item in ordered),
        expected_initial_hashes=initial_hashes,
        max_steps=MAX_STEPS,
        matched_max_candidates=MATCHED_MAX_CANDIDATES,
        wide_max_candidates=WIDE_MAX_CANDIDATES,
        max_error_components=MAX_ERROR_COMPONENTS,
        matched_padding_ratio=MATCHED_PADDING_RATIO,
        wide_padding_ratios=WIDE_PADDING_RATIOS,
        endpoint_iou=endpoint_iou,
        endpoint_f1=endpoint_f1,
        endpoint_empty_fp=endpoint_empty_fp,
        comparison_epsilon=epsilon,
    )
    return manifest, contract


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        raise ValueError("cannot box an empty component")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pad_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    ratio: float,
    *,
    production_minimum: bool,
) -> tuple[tuple[int, int, int, int], int, bool]:
    if ratio < 0:
        raise ValueError("padding ratio must be non-negative")
    x1, y1, x2, y2 = box
    height, width = shape
    span = max(x2 - x1 + 1, y2 - y1 + 1)
    padding = round(span * ratio)
    if production_minimum:
        padding = max(4, padding)
    padded = [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    ]
    adjusted = False
    # AgentAction boxes require positive width and height.  A one-pixel, zero-pad
    # component therefore gets the smallest deterministic executor-valid extent.
    if padded[0] == padded[2]:
        adjusted = True
        if padded[2] < width - 1:
            padded[2] += 1
        else:
            padded[0] -= 1
    if padded[1] == padded[3]:
        adjusted = True
        if padded[3] < height - 1:
            padded[3] += 1
        else:
            padded[1] -= 1
    return tuple(int(item) for item in padded), int(padding), adjusted


def _component_inventory(
    state: ChangeState, ground_truth: np.ndarray, limit: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    change = np.asarray(state.change_mask, dtype=bool)
    gt = np.asarray(ground_truth, dtype=bool)
    if change.shape != gt.shape:
        raise ValueError("GT and current change mask shapes differ")
    masks = {
        "fn": np.logical_and(gt, ~change),
        "fp": np.logical_and(change, ~gt),
    }
    raw: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for kind in ("fn", "fp"):
        components = connected_components(masks[kind])
        totals[f"{kind}_pixels"] = int(masks[kind].sum())
        totals[f"{kind}_component_count"] = len(components)
        for component in components:
            box = _mask_box(component)
            raw.append(
                {
                    "error_kind": kind,
                    "component": component,
                    "component_area": int(component.sum()),
                    "component_box": box,
                    "component_mask_sha256": mask_sha256(component),
                }
            )
    raw.sort(
        key=lambda item: (
            -int(item["component_area"]),
            str(item["error_kind"]),
            (
                int(item["component_box"][1]),
                int(item["component_box"][0]),
                int(item["component_box"][3]),
                int(item["component_box"][2]),
            ),
            str(item["component_mask_sha256"]),
        )
    )
    selected = raw[:limit]
    for index, item in enumerate(selected):
        item["component_rank"] = index
        item["component_id"] = f"{item['error_kind']}-c{index:03d}"
    totals["selected_component_count"] = len(selected)
    totals["all_component_count"] = len(raw)
    totals["unselected_component_count"] = len(raw) - len(selected)
    return selected, totals


def _padding_token(value: float) -> str:
    return f"p{round(value * 1000):04d}"


def build_teacher_candidates(
    episode: base.RuntimeEpisode,
    state: ChangeState,
    ground_truth: np.ndarray,
    step_index: int,
    *,
    geometry_mode: str,
    max_error_components: int = MAX_ERROR_COMPONENTS,
    matched_padding_ratio: float = MATCHED_PADDING_RATIO,
    wide_padding_ratios: Sequence[float] = WIDE_PADDING_RATIOS,
    candidate_limit: int | None = None,
) -> tuple[list[base.CandidateAction], dict[str, Any]]:
    """Create deterministic action-specific proposals from GT change error."""

    if geometry_mode not in {"matched", "wide"}:
        raise ValueError("geometry_mode must be matched or wide")
    components, inventory = _component_inventory(
        state, ground_truth, max_error_components
    )
    paddings = (
        (float(matched_padding_ratio),)
        if geometry_mode == "matched"
        else tuple(float(item) for item in wide_padding_ratios)
    )
    candidates_by_component: list[list[base.CandidateAction]] = []
    proposal_count = 0
    for component in components:
        component_candidates: list[base.CandidateAction] = []
        component_mask = np.asarray(component["component"], dtype=bool)
        seed_x, seed_y = _distance_transform_seed(component_mask)
        for padding_index, padding_ratio in enumerate(paddings):
            box, padding_pixels, extent_adjusted = _pad_box(
                tuple(component["component_box"]),
                component_mask.shape,
                padding_ratio,
                production_minimum=geometry_mode == "matched",
            )
            base_region_id = (
                f"teacher-{component['component_id']}-{_padding_token(padding_ratio)}"
            )
            generic = {
                "region_id": base_region_id,
                "proposal_version": "qwen3vl_gt_direct_action_geometry_v1",
                "teacher_only": True,
                "gt_derived_geometry": True,
                "geometry_source": "offline_gt_direct_teacher",
                "geometry_family": geometry_mode,
                "geometry_mode": geometry_mode,
                "gt_used_for": "offline_teacher_geometry_and_scoring",
                "error_kind": component["error_kind"],
                "error_definition": (
                    "GT_change AND NOT current_change"
                    if component["error_kind"] == "fn"
                    else "current_change AND NOT GT_change"
                ),
                "audit_kind": (
                    "missing" if component["error_kind"] == "fn" else "present"
                ),
                "component_id": component["component_id"],
                "component_rank": int(component["component_rank"]),
                "component_area": int(component["component_area"]),
                "component_error_pixels": int(component["component_area"]),
                "component_mask_sha256": component["component_mask_sha256"],
                "component_box_pixels": list(component["component_box"]),
                "component_seed_pixels": [seed_x, seed_y],
                "box_pixels": list(box),
                "padding_ratio": padding_ratio,
                "padding_pixels": padding_pixels,
                "padding_index": padding_index,
                "minimum_extent_adjusted": extent_adjusted,
                "seed_algorithm": "8_neighbor_repeated_erosion_distance_argmax_row_major",
                "box_algorithm": (
                    "component_bbox_ratio_padding_production_min4"
                    if geometry_mode == "matched"
                    else "component_bbox_ratio_padding_no_minimum"
                ),
                "component_seed_t1_mask_white": bool(
                    state.t1_mask[seed_y, seed_x]
                ),
                "component_seed_t2_mask_white": bool(
                    state.t2_mask[seed_y, seed_x]
                ),
            }
            proposal_count += 1
            routes = base.enumerate_live_actions(
                episode.episode_id, step_index, [generic]
            )
            for route_index, route in enumerate(routes):
                route_action = route.action
                reachability_tier = route.reachability_tier
                action_region_id = "--".join(
                    (base_region_id, route_action.action, route_action.target_view)
                )
                proposal = dict(generic)
                proposal.update(
                    {
                        "region_id": action_region_id,
                        "action_specific": True,
                        "authorized_action": route_action.action,
                        "authorized_target_view": route_action.target_view,
                        "route_index": route_index,
                    }
                )
                candidate_id = "--".join(
                    (
                        episode.episode_id,
                        geometry_mode,
                        f"step-{step_index:03d}",
                        action_region_id,
                    )
                )
                component_candidates.append(
                    base.CandidateAction(
                        candidate_id=candidate_id,
                        region_id=action_region_id,
                        reachability_tier=reachability_tier,
                        action=route_action,
                        proposal=proposal,
                    )
                )
        component_candidates.sort(
            key=lambda item: (
                int(item.proposal["padding_index"]),
                int(item.proposal["route_index"]),
                item.candidate_id,
            )
        )
        candidates_by_component.append(component_candidates)

    if geometry_mode == "matched":
        flattened = [item for group in candidates_by_component for item in group]
        flattened.sort(
            key=lambda item: (
                -int(item.proposal["component_area"]),
                item.candidate_id,
            )
        )
        # Match 0803: apply RGB compatibility before freezing the top four.
        candidates = [
            item
            for item in flattened
            if base.rgb_semantic_gate(
                item.action,
                base.resolve_rgb_state(
                    episode, step_index, item.region_id, item.action.target_view
                ),
            )["passed"]
        ]
    else:
        # Round-robin components fixes a deterministic order.  The frozen 96-action
        # cap covers the complete declared family: at most four components times
        # four paddings times at most six production-reachable routes.
        candidates = []
        max_group = max((len(group) for group in candidates_by_component), default=0)
        for index in range(max_group):
            for group in candidates_by_component:
                if index < len(group):
                    candidates.append(group[index])
    if candidate_limit is not None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        candidates = candidates[:candidate_limit]
    catalog = {
        "schema_id": "qwen3vl_gt_direct_teacher_catalog_v1",
        "geometry_mode": geometry_mode,
        "step_index": step_index,
        "source_state_mask_sha256": base._state_hashes(state),
        "source_state_sha256": base.state_sha256(base._state_hashes(state)),
        "error_inventory": inventory,
        "selected_components": [
            {
                key: base._json_safe(value)
                for key, value in item.items()
                if key != "component"
            }
            for item in components
        ],
        "padding_ratios": list(paddings),
        "proposal_geometry_count": proposal_count,
        "action_count_before_cap": sum(len(group) for group in candidates_by_component),
        "execution_candidate_count": len(candidates),
        "candidate_ids": [item.candidate_id for item in candidates],
        "gt_derived_geometry": True,
        "raw_gt_mask_in_catalog": False,
        "qwen_invoked": False,
    }
    return candidates, catalog


def _candidate_tool(candidate: base.CandidateAction) -> str:
    if candidate.action.action == "box":
        return "sam3"
    if candidate.action.action in {"positive_point", "negative_point"}:
        return "simpleclick"
    raise ValueError(f"unsupported teacher action {candidate.action.action!r}")


def _diagnostic_smoke_candidates(
    candidates: Sequence[base.CandidateAction],
) -> tuple[list[base.CandidateAction], dict[str, Any]]:
    """Pick at most one real SAM3 box and one positive SimpleClick action.

    Candidate construction is cheap and may expose many action routes.  The smoke
    is deliberately bounded to two real model calls from the same fresh step-1
    state; it is a dependency/authorization gate, not a miniature endpoint run.
    """

    requested = (("sam3", "box"), ("simpleclick", "positive_point"))
    selected: list[base.CandidateAction] = []
    availability: dict[str, bool] = {}
    for tool, action_name in requested:
        match = next(
            (item for item in candidates if item.action.action == action_name),
            None,
        )
        availability[tool] = match is not None
        if match is not None:
            selected.append(match)
    if len(selected) > DIAGNOSTIC_SMOKE_MAX_REAL_CANDIDATES:
        raise RuntimeError("diagnostic smoke candidate cap was exceeded")
    return selected, {
        "selection_policy": "first_deterministic_RGB_compatible_action_per_tool",
        "candidate_count_before_diagnostic_cap": len(candidates),
        "max_real_candidates": DIAGNOSTIC_SMOKE_MAX_REAL_CANDIDATES,
        "requested_action_types": [item[1] for item in requested],
        "requested_tool_types": [item[0] for item in requested],
        "available_requested_tool_types": [
            tool for tool, _ in requested if availability[tool]
        ],
        "unavailable_requested_tool_types": [
            tool for tool, _ in requested if not availability[tool]
        ],
        "selected_candidate_ids": [item.candidate_id for item in selected],
        "selected_action_types": [item.action.action for item in selected],
        "selected_tool_types": [_candidate_tool(item) for item in selected],
    }


def _install_teacher_proposal(
    environment: ChangeAgentEnvironment, proposal: Mapping[str, Any]
) -> None:
    if environment.state is None:
        raise RuntimeError("cannot install teacher proposal before Environment reset")
    state = environment.state
    state.evidence = dict(state.evidence)
    state.evidence["verifier_region_proposals"] = [dict(proposal)]
    state.evidence["gt_direct_teacher_boundary"] = {
        "teacher_only": True,
        "raw_label_pixels_present": False,
        "gt_derived_geometry_present": True,
        "authorized_region_id": proposal["region_id"],
    }


def _assert_source_masks(environment: ChangeAgentEnvironment, source: ChangeState) -> None:
    if environment.state is None:
        raise RuntimeError("branch Environment has no reset state")
    if base._state_hashes(environment.state) != base._state_hashes(source):
        raise RuntimeError("branch did not start from exact accepted masks")


def _raw_metric_gate(
    before: Mapping[str, Any], after: Mapping[str, Any], *, epsilon: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    delta = base.metric_delta(before, after)
    nonempty = int(before["gt_pixels"]) > 0
    pixel_pass = int(delta["pixel_error_reduction"]) > 0
    iou_pass = float(delta["iou"]) > epsilon
    f1_pass = float(delta["f1"]) > epsilon
    fp_pass = int(after["fp"]) < int(before["fp"])
    passed = pixel_pass and ((iou_pass and f1_pass) if nonempty else fp_pass)
    return delta, {
        "passed": bool(passed),
        "pixel_error_strictly_decreased": pixel_pass,
        "nonempty_gt": nonempty,
        "iou_strictly_increased": iou_pass if nonempty else None,
        "f1_strictly_increased": f1_pass if nonempty else None,
        "fp_strictly_decreased": fp_pass if not nonempty else None,
        "comparison_epsilon": epsilon,
    }


def _track_policy(track_name: str) -> dict[str, Any]:
    if track_name not in TRACK_NAMES:
        raise ValueError(f"unknown track: {track_name}")
    return {
        "require_hard_gate": track_name != "wide_raw",
        "require_rgb_semantic_gate": track_name in {"matched_safe", "wide_safe"},
        "require_strict_metric_improvement": True,
        "geometry_mode": "matched" if track_name == "matched_safe" else "wide",
        "counterfactual_gate_bypass_allowed": track_name == "wide_raw",
        "policy_id": {
            "matched_safe": "matched_geometry_hard_rgb_metric",
            "wide_safe": "wide_geometry_hard_rgb_metric",
            "wide_hard": "wide_geometry_hard_metric_ignore_rgb",
            "wide_raw": "wide_geometry_metric_only_ignore_hard_and_rgb",
        }[track_name],
    }


def _candidate_eligibility(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if policy["require_hard_gate"] and not record["hard_gate"]["passed"]:
        reasons.append("hard_gate_failed")
    if policy["require_rgb_semantic_gate"] and not record["rgb_semantic_gate"]["passed"]:
        reasons.append("rgb_semantic_gate_failed")
    if not record["raw_metric_beneficial"]["passed"]:
        reasons.append("strict_metric_improvement_failed")
    return not reasons, reasons


def _execute_candidate(
    episode: base.RuntimeEpisode,
    source_state: ChangeState,
    candidate: base.CandidateAction,
    runtime_factory: base.EpisodeRuntimeFactory,
    artifact_root: Path,
    output_root: Path,
    ground_truth: np.ndarray,
    step_index: int,
    ordinal: int,
    epsilon: float,
) -> tuple[dict[str, Any], ChangeState]:
    branch = runtime_factory.create_branch_environment(
        episode, source_state.clone(), artifact_root / "tool_runs"
    )
    branch.reset(episode.t1_image, episode.t2_image, episode.query)
    _assert_source_masks(branch, source_state)
    _install_teacher_proposal(branch, candidate.proposal)
    branch.step(candidate.action, authorized_region_id=candidate.region_id)
    entry = branch.trajectory.entries[-1]
    candidate_state = entry.state.clone()
    state_artifact = base._state_artifact(
        candidate_state, artifact_root / "masks", output_root
    )
    hard = base._hard_gate(entry.execution)
    observed = base.resolve_rgb_state(
        episode, step_index, candidate.region_id, candidate.action.target_view
    )
    rgb = base.rgb_semantic_gate(candidate.action, observed)
    before = base.mask_metrics(source_state.change_mask, ground_truth)
    after = base.mask_metrics(candidate_state.change_mask, ground_truth)
    delta, metric_gate = _raw_metric_gate(before, after, epsilon=epsilon)
    authorization = (
        entry.execution.get("point_authorization", {})
        if candidate.action.action in {"positive_point", "negative_point"}
        else entry.execution.get("box_authorization", {})
    )
    geometry_match = bool(
        isinstance(authorization, Mapping)
        and authorization.get("region_id") == candidate.region_id
        and authorization.get("proposal_box_xyxy")
        == list(candidate.proposal["box_pixels"])
        and (
            candidate.action.action in {"positive_point", "negative_point"}
            or authorization.get("geometry_match") is True
        )
    )
    if not geometry_match:
        raise RuntimeError("GT-derived geometry did not reach authorized tool boundary")
    verifier = branch.verifier
    record = {
        "candidate_id": candidate.candidate_id,
        "ordinal": ordinal,
        "region_id": candidate.region_id,
        "reachability_tier": candidate.reachability_tier,
        "teacher_geometry": base._json_safe(candidate.proposal),
        "proposal": base._json_safe(candidate.proposal),
        "action": candidate.action.to_dict(),
        "source_state_mask_sha256": base._state_hashes(source_state),
        "source_state_sha256": base.state_sha256(base._state_hashes(source_state)),
        "candidate_state": state_artifact,
        "raw_metrics": after,
        "metrics": after,
        "metric_delta": delta,
        "raw_metric_beneficial": metric_gate,
        "per_step_metric_gate": metric_gate,
        "hard_gate": hard,
        "rgb_semantic_gate": rgb,
        "environment_candidate_accepted": bool(entry.execution.get("candidate_accepted")),
        "tool_receipt": base._tool_receipt(entry.execution),
        "runtime_execution": base._json_safe(entry.execution),
        "teacher_boundary_audit": {
            "raw_label_pixels_passed_to_environment": False,
            "raw_label_pixels_passed_to_tools": False,
            "raw_label_path_passed_to_runtime": False,
            "gt_derived_geometry_passed_to_environment": True,
            "gt_derived_geometry_passed_to_tools": True,
            "gt_derived_metadata_passed_to_environment": True,
            "executor_receipt_contains_teacher_encoded_region_id": True,
            "runtime_boundary_layers": base._json_safe(RUNTIME_BOUNDARY_LAYERS),
            "geometry_match": geometry_match,
        },
        "no_qwen_audit": {
            "qwen_invoked": False,
            "qwen_calls": int(getattr(verifier, "qwen_calls", 0)),
        },
        "eligible_under_track": None,
        "eligibility_rejection_reasons": None,
        "selected": False,
    }
    return record, candidate_state


def _validate_diagnostic_smoke_record(
    record: Mapping[str, Any], candidate: base.CandidateAction
) -> None:
    """Fail closed unless a bounded smoke candidate reached its real backend."""

    expected_tool = _candidate_tool(candidate)
    receipt = _as_mapping(record.get("tool_receipt"), "diagnostic tool_receipt")
    if receipt.get("tool") != expected_tool:
        raise RuntimeError("diagnostic smoke tool receipt differs from action type")
    if not isinstance(receipt.get("tool_input"), Mapping):
        raise RuntimeError("diagnostic smoke lacks a real tool input receipt")
    if not isinstance(receipt.get("tool_result"), Mapping):
        raise RuntimeError("diagnostic smoke lacks a real backend result receipt")
    authorization_key = (
        "box_authorization" if expected_tool == "sam3" else "point_authorization"
    )
    authorization = _as_mapping(
        receipt.get(authorization_key), f"diagnostic {authorization_key}"
    )
    if authorization.get("region_id") != candidate.region_id:
        raise RuntimeError("diagnostic smoke authorization region differs")
    boundary = _as_mapping(
        record.get("teacher_boundary_audit"), "diagnostic teacher_boundary_audit"
    )
    if (
        boundary.get("geometry_match") is not True
        or boundary.get("raw_label_pixels_passed_to_environment") is not False
        or boundary.get("raw_label_pixels_passed_to_tools") is not False
        or boundary.get("raw_label_path_passed_to_runtime") is not False
        or boundary.get("gt_derived_geometry_passed_to_environment") is not True
        or boundary.get("gt_derived_geometry_passed_to_tools") is not True
        or boundary.get("gt_derived_metadata_passed_to_environment") is not True
        or boundary.get("executor_receipt_contains_teacher_encoded_region_id")
        is not True
        or boundary.get("runtime_boundary_layers") != RUNTIME_BOUNDARY_LAYERS
    ):
        raise RuntimeError("diagnostic smoke GT/tool boundary audit failed")
    no_qwen = _as_mapping(record.get("no_qwen_audit"), "diagnostic no_qwen_audit")
    if no_qwen.get("qwen_invoked") is not False or no_qwen.get("qwen_calls") != 0:
        raise RuntimeError("diagnostic smoke made a forbidden Qwen call")
    for gate_name in ("hard_gate", "rgb_semantic_gate"):
        gate = _as_mapping(record.get(gate_name), f"diagnostic {gate_name}")
        if type(gate.get("passed")) is not bool:
            raise RuntimeError(f"diagnostic smoke lacks {gate_name} safety flag")


def _replay_selected(
    episode: base.RuntimeEpisode,
    source_state: ChangeState,
    candidate: base.CandidateAction,
    expected_state: ChangeState,
    runtime_factory: base.EpisodeRuntimeFactory,
    artifact_root: Path,
    output_root: Path,
    ground_truth: np.ndarray,
    track_name: str,
    step_index: int,
    canonical_environment: ChangeAgentEnvironment | None,
) -> tuple[dict[str, Any], ChangeState, ChangeAgentEnvironment | None]:
    use_canonical = bool(
        canonical_environment is not None
        and canonical_environment.state is not None
        and base._state_hashes(canonical_environment.state)
        == base._state_hashes(source_state)
    )
    if use_canonical:
        replay = canonical_environment
    else:
        replay = runtime_factory.create_branch_environment(
            episode, source_state.clone(), artifact_root / "tool_runs"
        )
        replay.reset(episode.t1_image, episode.t2_image, episode.query)
        _assert_source_masks(replay, source_state)
    _install_teacher_proposal(replay, candidate.proposal)
    replay.step(candidate.action, authorized_region_id=candidate.region_id)
    entry = replay.trajectory.entries[-1]
    replay_state = entry.state.clone()
    expected_hashes = base._state_hashes(expected_state)
    replay_hashes = base._state_hashes(replay_state)
    if replay_hashes != expected_hashes:
        raise RuntimeError("selected replay masks differ from candidate branch")
    hard = base._hard_gate(entry.execution)
    accepted = bool(entry.execution.get("candidate_accepted"))
    if track_name != "wide_raw" and not accepted:
        raise RuntimeError("safe/hard track selected an Environment-rejected candidate")
    counterfactual = not accepted
    if accepted:
        if replay.state is None or base._state_hashes(replay.state) != replay_hashes:
            raise RuntimeError("accepted canonical Environment did not carry replay masks")
        next_canonical: ChangeAgentEnvironment | None = replay if use_canonical else None
    else:
        # A rejected Environment correctly rolled back.  Also, factory branch
        # Environments have max_steps=1 and are done after replay even when accepted.
        # Either case must start the next step from a new branch; retaining this
        # exhausted object would fail with "episode has already finished".
        next_canonical = None
    state_artifact = base._state_artifact(
        replay_state, artifact_root / "masks", output_root
    )
    metrics = base.mask_metrics(replay_state.change_mask, ground_truth)
    return {
        "mode": (
            "canonical_environment_counterfactual_gate_bypass"
            if counterfactual and use_canonical
            else "second_branch_counterfactual_gate_bypass"
            if counterfactual
            else "persistent_canonical_environment_replay"
            if use_canonical
            else "second_branch_canonical_environment_replay"
        ),
        "candidate_branch_state_mask_sha256": expected_hashes,
        "canonical_replay_state_mask_sha256": replay_hashes,
        "all_three_mask_hashes_match": True,
        "environment_candidate_accepted": accepted,
        "counterfactual_gate_bypass": counterfactual,
        "carried_state_source": (
            "rejected_candidate_state_teacher_only" if counterfactual else "accepted_environment_state"
        ),
        "state": state_artifact,
        "metrics": metrics,
        "hard_gate": hard,
        "tool_receipt": base._tool_receipt(entry.execution),
        "runtime_execution": base._json_safe(entry.execution),
    }, replay_state, next_canonical


def _endpoint(
    metrics: Mapping[str, Any], contract: TeacherContract
) -> dict[str, Any]:
    return base.endpoint_gate(
        metrics,
        minimum_iou=contract.endpoint_iou,
        minimum_f1=contract.endpoint_f1,
        empty_gt_max_fp=contract.endpoint_empty_fp,
        epsilon=contract.comparison_epsilon,
    )


def _fresh_track_environment(
    episode: base.RuntimeEpisode,
    runtime_factory: base.EpisodeRuntimeFactory,
    directory: Path,
    expected_hashes: Mapping[str, str],
    *,
    max_steps: int = MAX_STEPS,
) -> ChangeAgentEnvironment:
    environment = runtime_factory.create_canonical_environment(
        episode, directory / "fresh_canonical_tool_runs", max_steps
    )
    environment.reset(episode.t1_image, episode.t2_image, episode.query)
    if environment.state is None:
        raise RuntimeError("fresh SAM3 reset produced no state")
    actual = base._state_hashes(environment.state)
    if actual != dict(expected_hashes):
        raise RuntimeError(
            "fresh SAM3 initial hashes differ from preregistered 0803 hashes"
        )
    return environment


def run_track(
    track_name: str,
    episode: base.RuntimeEpisode,
    ground_truth: np.ndarray,
    runtime_factory: base.EpisodeRuntimeFactory,
    output_root: Path,
    episode_dir: Path,
    contract: TeacherContract,
    expected_initial_hashes: Mapping[str, str],
    *,
    initial_environment: ChangeAgentEnvironment | None = None,
) -> dict[str, Any]:
    policy = _track_policy(track_name)
    track_dir = episode_dir / "tracks" / track_name
    canonical = initial_environment or _fresh_track_environment(
        episode, runtime_factory, track_dir, expected_initial_hashes
    )
    if canonical.state is None:
        raise RuntimeError("track canonical Environment has no initial state")
    initial_state = canonical.state.clone()
    if base._state_hashes(initial_state) != dict(expected_initial_hashes):
        raise RuntimeError("track did not begin from exact fresh SAM3 state")
    source_state = initial_state.clone()
    initial_metrics = base.mask_metrics(initial_state.change_mask, ground_truth)
    initial_endpoint = _endpoint(initial_metrics, contract)
    rounds: list[dict[str, Any]] = []
    selected_steps: list[dict[str, Any]] = []
    candidate_count = 0
    replay_count = 0
    stop_reason = "max_steps"

    if initial_endpoint["passed"]:
        stop_reason = (
            "already_exact" if int(initial_metrics["pixel_error"]) == 0 else "already_endpoint"
        )
    else:
        for step_index in range(1, contract.max_steps + 1):
            geometry_mode = str(policy["geometry_mode"])
            limit = (
                contract.matched_max_candidates
                if geometry_mode == "matched"
                else contract.wide_max_candidates
            )
            actions, catalog = build_teacher_candidates(
                episode,
                source_state,
                ground_truth,
                step_index,
                geometry_mode=geometry_mode,
                max_error_components=contract.max_error_components,
                matched_padding_ratio=contract.matched_padding_ratio,
                wide_padding_ratios=contract.wide_padding_ratios,
                candidate_limit=limit,
            )
            source_metrics = base.mask_metrics(source_state.change_mask, ground_truth)
            source_endpoint = _endpoint(source_metrics, contract)
            records: list[dict[str, Any]] = []
            states: list[ChangeState] = []
            for ordinal, candidate in enumerate(actions):
                record, candidate_state = _execute_candidate(
                    episode,
                    source_state,
                    candidate,
                    runtime_factory,
                    track_dir
                    / "candidate_branches"
                    / f"step-{step_index:03d}"
                    / f"candidate-{ordinal:04d}",
                    output_root,
                    ground_truth,
                    step_index,
                    ordinal,
                    contract.comparison_epsilon,
                )
                eligible, reasons = _candidate_eligibility(record, policy)
                record["eligible_under_track"] = eligible
                record["eligibility_rejection_reasons"] = reasons
                records.append(record)
                states.append(candidate_state)
            candidate_count += len(records)
            eligible_records = [item for item in records if item["eligible_under_track"]]
            eligible_records.sort(
                key=lambda item: (
                    -int(item["metric_delta"]["pixel_error_reduction"]),
                    str(item["candidate_id"]),
                )
            )
            round_record: dict[str, Any] = {
                "step_index": step_index,
                "source_state_mask_sha256": base._state_hashes(source_state),
                "source_state_sha256": base.state_sha256(base._state_hashes(source_state)),
                "source_metrics": source_metrics,
                "source_endpoint_gate": source_endpoint,
                "teacher_catalog": catalog,
                "candidate_count": len(records),
                "candidates": records,
                "candidate_batch_complete_before_selection": True,
                "beneficial_candidate_ids": [item["candidate_id"] for item in eligible_records],
                "selected_candidate_id": None,
                "canonical_replay": None,
            }
            if not eligible_records:
                stop_reason = "no_eligible_beneficial"
                rounds.append(round_record)
                break
            selected_record = eligible_records[0]
            selected_record["selected"] = True
            selected_index = int(selected_record["ordinal"])
            selected_candidate = actions[selected_index]
            replay, next_state, canonical = _replay_selected(
                episode,
                source_state,
                selected_candidate,
                states[selected_index],
                runtime_factory,
                track_dir / "canonical_replays" / f"step-{step_index:03d}",
                output_root,
                ground_truth,
                track_name,
                step_index,
                canonical,
            )
            replay_count += 1
            replay_metrics = replay["metrics"]
            replay_endpoint = _endpoint(replay_metrics, contract)
            replay["endpoint_gate"] = replay_endpoint
            round_record["selected_candidate_id"] = selected_candidate.candidate_id
            round_record["canonical_replay"] = replay
            selected_steps.append(
                {
                    "step_index": step_index,
                    "candidate_id": selected_candidate.candidate_id,
                    "region_id": selected_candidate.region_id,
                    "action": selected_candidate.action.to_dict(),
                    "teacher_geometry": base._json_safe(selected_candidate.proposal),
                    "source_state_mask_sha256": base._state_hashes(source_state),
                    "candidate_state": selected_record["candidate_state"],
                    "canonical_replay": replay,
                    "metrics_before": source_metrics,
                    "metrics_after": replay_metrics,
                    "metric_delta": selected_record["metric_delta"],
                    "raw_metric_beneficial": selected_record["raw_metric_beneficial"],
                    "hard_gate": selected_record["hard_gate"],
                    "rgb_semantic_gate": selected_record["rgb_semantic_gate"],
                    "counterfactual_gate_bypass": replay["counterfactual_gate_bypass"],
                    "all_three_mask_hashes_match": True,
                    "qwen_invoked": False,
                }
            )
            rounds.append(round_record)
            source_state = next_state.clone()
            if replay_endpoint["passed"]:
                stop_reason = (
                    "exact_recovery"
                    if int(replay_metrics["pixel_error"]) == 0
                    else "endpoint_reached"
                )
                break

    final_metrics = base.mask_metrics(source_state.change_mask, ground_truth)
    final_endpoint = _endpoint(final_metrics, contract)
    final_state_artifact = base._state_artifact(
        source_state, track_dir / "final_state", output_root
    )
    if stop_reason == "max_steps" and len(selected_steps) != contract.max_steps:
        raise RuntimeError("max_steps termination did not consume three selected actions")
    status = (
        "endpoint_success"
        if final_endpoint["passed"] and not initial_endpoint["passed"]
        else "already_endpoint"
        if initial_endpoint["passed"]
        else "not_reachable_within_track"
    )
    return {
        "track_name": track_name,
        "executed": True,
        "teacher_only": True,
        "training_authorized": False,
        "gate_policy": policy,
        "max_steps": contract.max_steps,
        "candidate_limit_per_step": (
            contract.matched_max_candidates
            if policy["geometry_mode"] == "matched"
            else contract.wide_max_candidates
        ),
        "initial_state_mask_sha256": base._state_hashes(initial_state),
        "expected_initial_state_mask_sha256": dict(expected_initial_hashes),
        "fresh_initial_hash_match": True,
        "initial_metrics": initial_metrics,
        "initial_endpoint_gate": initial_endpoint,
        "selection_rounds": rounds,
        "steps": selected_steps,
        "termination": {
            "reason": stop_reason,
            "selected_step_count": len(selected_steps),
            "candidate_count": candidate_count,
            "replay_count": replay_count,
        },
        "final_state": final_state_artifact,
        "final_state_mask_sha256": base._state_hashes(source_state),
        "final_state_sha256": base.state_sha256(base._state_hashes(source_state)),
        "final_metrics": final_metrics,
        "endpoint_reachability": {
            "initial_gate": initial_endpoint,
            "final_gate": final_endpoint,
            "success_step": (
                len(selected_steps)
                if final_endpoint["passed"] and not initial_endpoint["passed"]
                else None
            ),
            "exact_recovery": int(final_metrics["pixel_error"]) == 0,
        },
        "track_status": status,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "counterfactual_selected_steps": sum(
            bool(item["counterfactual_gate_bypass"]) for item in selected_steps
        ),
    }


def _skipped_track(track_name: str, prior_winner: str) -> dict[str, Any]:
    return {
        "track_name": track_name,
        "executed": False,
        "teacher_only": True,
        "training_authorized": False,
        "gate_policy": _track_policy(track_name),
        "skip_reason": "prior_track_endpoint_success",
        "prior_winner": prior_winner,
        "track_status": "skipped_adaptive",
        "selection_rounds": [],
        "steps": [],
        "qwen_invoked": False,
        "qwen_calls": 0,
    }


def _episode_conclusion(winner: str | None) -> dict[str, Any]:
    if winner == "matched_safe":
        conclusion = "proposal_geometry_or_coverage_bottleneck_supported_matched"
        plain = "Matched GT-derived geometry reaches endpoint under production gates; current Proposal is at least one bottleneck."
    elif winner == "wide_safe":
        conclusion = "proposal_geometry_or_coverage_bottleneck_supported"
        plain = "Wider GT-derived geometry succeeds under production gates; current Proposal is at least one bottleneck."
    elif winner == "wide_hard":
        conclusion = "rgb_semantic_gate_bottleneck_supported"
        plain = "Real tools and hard gates succeed only after ignoring frozen RGB routing."
    elif winner == "wide_raw":
        conclusion = "hard_gate_bottleneck_supported_counterfactual"
        plain = "Metric-only real-tool candidates succeed after wide_hard failed; production hard gate blocks the bounded trajectory."
    else:
        conclusion = "tool_upper_bound_not_reached_within_frozen_protocol"
        plain = "The bounded greedy wide GT-derived protocol did not find an endpoint in three real-tool steps; this does not isolate a tool-only ceiling."
    return {
        "classification": conclusion,
        "plain_language": plain,
        "winning_track": winner,
        "proposal_bottleneck_supported": winner in {"matched_safe", "wide_safe"},
        "proposal_only_claim_allowed": False,
        "tool_upper_bound_reached": winner is not None,
        "production_safe_upper_bound_reached": winner in {"matched_safe", "wide_safe"},
    }


def run_episode(
    pilot: base.PilotEpisode,
    runtime_factory: base.EpisodeRuntimeFactory,
    output_root: Path,
    contract: TeacherContract,
    *,
    label_loader: Callable[[base.LabelSource], np.ndarray] = base.load_label_mask,
    expected_gt_pixels: int | None = None,
) -> dict[str, Any]:
    episode = pilot.runtime
    episode_dir = output_root / "episodes" / episode.episode_id
    expected_hashes = contract.expected_initial_hashes[episode.episode_id]

    # The only fresh initializer built before label access becomes matched_safe's
    # canonical Environment.  No object passed to the factory contains label data.
    first_environment = _fresh_track_environment(
        episode,
        runtime_factory,
        episode_dir / "tracks" / "matched_safe",
        expected_hashes,
    )
    if first_environment.state is None:
        raise RuntimeError("fresh initial Environment lost state")
    initial_state = first_environment.state.clone()
    initial_artifact = base._state_artifact(
        initial_state, episode_dir / "initial_state", output_root
    )
    initial_artifact["initializer_receipt"] = base._json_safe(
        initial_state.evidence.get("sam3_initialization")
    )

    ground_truth = np.asarray(label_loader(pilot.label), dtype=bool)
    if ground_truth.shape != initial_state.change_mask.shape:
        raise ValueError("offline label shape differs from fresh SAM3 state")
    if expected_gt_pixels is not None and int(ground_truth.sum()) != expected_gt_pixels:
        raise ValueError("offline label pixels differ from preregistration")
    gt_artifact = base._atomic_save_mask(
        episode_dir / "offline_ground_truth.npy", ground_truth, output_root
    )
    initial_metrics = base.mask_metrics(initial_state.change_mask, ground_truth)
    initial_endpoint = _endpoint(initial_metrics, contract)
    if initial_endpoint["passed"]:
        raise RuntimeError("formal teacher denominator contains an initial endpoint episode")

    tracks: dict[str, dict[str, Any]] = {}
    executed: list[str] = []
    winner: str | None = None
    for track_name in TRACK_NAMES:
        if winner is not None:
            tracks[track_name] = _skipped_track(track_name, winner)
            continue
        track = run_track(
            track_name,
            episode,
            ground_truth,
            runtime_factory,
            output_root,
            episode_dir,
            contract,
            expected_hashes,
            initial_environment=(first_environment if track_name == "matched_safe" else None),
        )
        tracks[track_name] = track
        executed.append(track_name)
        if track["endpoint_reachability"]["final_gate"]["passed"]:
            winner = track_name

    if int(getattr(runtime_factory, "qwen_calls", 0)) != 0:
        raise RuntimeError("GT-direct teacher runner made a forbidden Qwen call")
    conclusion = _episode_conclusion(winner)
    return {
        "schema_id": EPISODE_SCHEMA_ID,
        "status": "completed",
        "episode_id": episode.episode_id,
        "sample_id": episode.sample_id,
        "split": episode.split,
        "parent_id": episode.parent_id,
        "query": episode.query,
        "crop_xyxy_exclusive": list(episode.crop_xyxy_exclusive),
        "teacher_only": True,
        "training_authorized": False,
        "claim_scope": "LEVIR_change_mask_only_teacher_tool_upper_bound",
        "qwen_invoked": False,
        "qwen_calls": 0,
        "source_hashes": {
            "t1": episode.t1_source.sha256,
            "t2": episode.t2_source.sha256,
            "label": pilot.label.identity.sha256,
        },
        "source_parent": {
            "t1": episode.t1_source.to_dict(),
            "t2": episode.t2_source.to_dict(),
            "label": pilot.label.identity.to_dict(),
        },
        "rgb_semantics": {
            "default": dict(episode.rgb_states),
            "evidence": episode.rgb_semantic_evidence,
            "source": episode.rgb_semantic_source,
            "status": episode.rgb_semantic_status,
        },
        "gt_mask": gt_artifact,
        "gt_access_audit": {
            "access_count": 1,
            "first_access_phase": "after_fresh_sam3_initialization_and_hash_match",
            "label_passed_to_runtime_factory": False,
            "raw_label_pixels_passed_to_environment": False,
            "raw_label_pixels_passed_to_tools": False,
            "raw_label_pixels_passed_to_qwen": False,
            "raw_label_path_passed_to_runtime": False,
            "gt_derived_geometry_passed_to_environment": True,
            "gt_derived_geometry_passed_to_tools": True,
            "gt_derived_metadata_passed_to_environment": True,
            "executor_receipt_contains_teacher_encoded_region_id": True,
            "runtime_boundary_layers": base._json_safe(RUNTIME_BOUNDARY_LAYERS),
            "runtime_factory_sample_type": "RuntimeEpisode_GT_free",
            "offline_label_loader_only": True,
        },
        "initial_state": initial_artifact,
        "expected_initial_state_mask_sha256": dict(expected_hashes),
        "fresh_initial_hash_match": True,
        "initial_metrics": initial_metrics,
        "initial_endpoint_gate": initial_endpoint,
        "adaptive_execution": {
            "track_order": list(TRACK_NAMES),
            "executed_tracks": executed,
            "stops_after_first_endpoint_success": True,
            "winning_track": winner,
        },
        "tracks": tracks,
        "conclusion": conclusion,
        "episode_status": (
            "teacher_endpoint_success" if winner is not None else "teacher_endpoint_not_reached"
        ),
    }


def run_diagnostic_smoke_episode(
    pilot: base.PilotEpisode,
    runtime_factory: base.EpisodeRuntimeFactory,
    output_root: Path,
    contract: TeacherContract,
    *,
    label_loader: Callable[[base.LabelSource], np.ndarray] = base.load_label_mask,
    expected_gt_pixels: int | None = None,
) -> dict[str, Any]:
    """Execute the fixed, bounded dependency smoke without searching an endpoint."""

    episode = pilot.runtime
    if episode.episode_id != DIAGNOSTIC_SMOKE_SAMPLE_ID:
        raise ValueError("diagnostic smoke is fixed to train_407_r3c2")
    episode_dir = output_root / "diagnostic_episodes" / episode.episode_id
    expected_hashes = contract.expected_initial_hashes[episode.episode_id]

    # Preserve the same GT firewall as the formal runner: initialize SAM3 and
    # verify all three fresh state hashes before opening the offline label.
    initial_environment = _fresh_track_environment(
        episode,
        runtime_factory,
        episode_dir,
        expected_hashes,
        max_steps=1,
    )
    if initial_environment.state is None:
        raise RuntimeError("diagnostic fresh Environment lost state")
    initial_state = initial_environment.state.clone()
    initial_artifact = base._state_artifact(
        initial_state, episode_dir / "initial_state", output_root
    )
    initial_artifact["initializer_receipt"] = base._json_safe(
        initial_state.evidence.get("sam3_initialization")
    )

    ground_truth = np.asarray(label_loader(pilot.label), dtype=bool)
    if ground_truth.shape != initial_state.change_mask.shape:
        raise ValueError("offline label shape differs from diagnostic fresh state")
    if expected_gt_pixels is not None and int(ground_truth.sum()) != expected_gt_pixels:
        raise ValueError("diagnostic offline label pixels differ from preregistration")
    gt_artifact = base._atomic_save_mask(
        episode_dir / "offline_ground_truth.npy", ground_truth, output_root
    )

    available, catalog = build_teacher_candidates(
        episode,
        initial_state,
        ground_truth,
        DIAGNOSTIC_SMOKE_STEP_INDEX,
        geometry_mode="matched",
        max_error_components=contract.max_error_components,
        matched_padding_ratio=contract.matched_padding_ratio,
        wide_padding_ratios=contract.wide_padding_ratios,
        candidate_limit=None,
    )
    selected, selection = _diagnostic_smoke_candidates(available)
    required_tools = ["sam3", "simpleclick"]
    if selection["available_requested_tool_types"] != required_tools:
        raise RuntimeError(
            "fixed diagnostic sample does not expose both SAM3 box and "
            "SimpleClick positive-point candidates"
        )
    if selection["selected_tool_types"] != required_tools or len(selected) != 2:
        raise RuntimeError("diagnostic smoke did not select exactly one candidate per tool")
    catalog = dict(catalog)
    catalog["candidate_count_before_diagnostic_cap"] = catalog[
        "execution_candidate_count"
    ]
    catalog["execution_candidate_count"] = len(selected)
    catalog["candidate_ids"] = [item.candidate_id for item in selected]
    catalog["diagnostic_smoke_selection"] = selection

    records: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(selected):
        record, _ = _execute_candidate(
            episode,
            initial_state,
            candidate,
            runtime_factory,
            episode_dir / "candidate_branches" / f"candidate-{ordinal:02d}",
            output_root,
            ground_truth,
            DIAGNOSTIC_SMOKE_STEP_INDEX,
            ordinal,
            contract.comparison_epsilon,
        )
        _validate_diagnostic_smoke_record(record, candidate)
        records.append(record)
    if int(getattr(runtime_factory, "qwen_calls", 0)) != 0:
        raise RuntimeError("diagnostic smoke made a forbidden Qwen call")

    executed_tools = [str(item["tool_receipt"]["tool"]) for item in records]
    executed_actions = [str(item["action"]["action"]) for item in records]
    if executed_tools != required_tools or executed_actions != ["box", "positive_point"]:
        raise RuntimeError("diagnostic smoke executed an unexpected tool/action sequence")
    return {
        "schema_id": DIAGNOSTIC_SMOKE_EPISODE_SCHEMA_ID,
        "status": "completed",
        "diagnostic_smoke": True,
        "scientific_result_authoritative": False,
        "output_scope": "diagnostics_only",
        "episode_id": episode.episode_id,
        "sample_id": episode.sample_id,
        "split": episode.split,
        "parent_id": episode.parent_id,
        "query": episode.query,
        "crop_xyxy_exclusive": list(episode.crop_xyxy_exclusive),
        "teacher_only": True,
        "training_authorized": False,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "diagnostic_contract": {
            "fresh_sam3_initialization": True,
            "fresh_state_hash_match_before_gt_access": True,
            "geometry_mode": "matched",
            "step_indices_executed": [DIAGNOSTIC_SMOKE_STEP_INDEX],
            "max_real_candidates": DIAGNOSTIC_SMOKE_MAX_REAL_CANDIDATES,
            "independent_candidates_from_same_fresh_state": True,
            "endpoint_search_performed": False,
            "candidate_selection_or_replay_performed": False,
        },
        "gt_access_audit": {
            "access_count": 1,
            "first_access_phase": "after_fresh_sam3_initialization_and_hash_match",
            "label_passed_to_runtime_factory": False,
            "raw_label_pixels_passed_to_environment": False,
            "raw_label_pixels_passed_to_tools": False,
            "raw_label_pixels_passed_to_qwen": False,
            "raw_label_path_passed_to_runtime": False,
            "gt_derived_geometry_passed_to_environment": True,
            "gt_derived_geometry_passed_to_tools": True,
            "gt_derived_metadata_passed_to_environment": True,
            "executor_receipt_contains_teacher_encoded_region_id": True,
            "runtime_boundary_layers": base._json_safe(RUNTIME_BOUNDARY_LAYERS),
            "runtime_factory_sample_type": "RuntimeEpisode_GT_free",
            "offline_label_loader_only": True,
        },
        "initial_state": initial_artifact,
        "expected_initial_state_mask_sha256": dict(expected_hashes),
        "fresh_initial_hash_match": True,
        "initial_metrics": base.mask_metrics(initial_state.change_mask, ground_truth),
        "gt_mask": gt_artifact,
        "teacher_catalog": catalog,
        "candidate_selection": selection,
        "candidates": records,
        "actual_execution": {
            "candidate_count": len(records),
            "tool_types": executed_tools,
            "action_types": executed_actions,
            "all_gt_geometry_authorized": all(
                item["teacher_boundary_audit"]["geometry_match"] is True
                for item in records
            ),
            "all_real_tool_receipts_present": all(
                isinstance(item["tool_receipt"].get("tool_result"), Mapping)
                for item in records
            ),
            "hard_gate_flags_observed": all(
                type(item["hard_gate"].get("passed")) is bool for item in records
            ),
            "rgb_semantic_gate_flags_observed": all(
                type(item["rgb_semantic_gate"].get("passed")) is bool
                for item in records
            ),
        },
    }


def _sample_specs(contract: TeacherContract) -> Mapping[str, Any]:
    return base._pilot_sample_specs(contract.base_manifest)


def _assigned_samples(
    contract: TeacherContract, shard_index: int, num_shards: int
) -> list[str]:
    if num_shards != len(contract.ordered_samples):
        raise ValueError("formal teacher run requires exactly four shards")
    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index")
    return [contract.ordered_samples[shard_index]]


def run_diagnostic_smoke(
    config_path: Path,
    output: Path,
    runtime_commit: str,
    *,
    shard_index: int,
    num_shards: int,
    runtime_factory: base.EpisodeRuntimeFactory | None = None,
    label_loader: Callable[[base.LabelSource], np.ndarray] = base.load_label_mask,
) -> dict[str, Any]:
    """Run one bounded step-1 candidate from each real editing backend."""

    if shard_index != 0 or num_shards != len(FORMAL_SAMPLE_IDS):
        raise ValueError("diagnostic smoke requires shard_index=0 and num_shards=4")
    config_path = config_path.resolve()
    manifest, contract = load_contract(config_path)
    if DIAGNOSTIC_SMOKE_SAMPLE_ID not in contract.ordered_samples:
        raise ValueError("diagnostic smoke sample is absent from the frozen contract")
    runtime_commit = base._runtime_commit(runtime_commit)
    output = base._prepare_output(output)
    factory = runtime_factory or base.ProductionRuntimeFactory(
        base.build_runtime_factory_config(contract.base_manifest)
    )
    specs = _sample_specs(contract)
    raw_spec = specs[DIAGNOSTIC_SMOKE_SAMPLE_ID]
    base_protocol = _as_mapping(
        contract.base_manifest.get("execution_protocol", {}),
        "base execution_protocol",
    )
    query = str(base_protocol.get("query", "building"))
    parsed, label_source = base.parse_pilot_episode(raw_spec, 0, query)
    runtime_episode = base.load_runtime_episode(parsed)
    crop = _as_mapping(
        raw_spec.get("crop", {}),
        f"samples.{DIAGNOSTIC_SMOKE_SAMPLE_ID}.crop",
    )
    record = run_diagnostic_smoke_episode(
        base.PilotEpisode(runtime_episode, label_source),
        factory,
        output,
        contract,
        label_loader=label_loader,
        expected_gt_pixels=int(crop["gt_pixels"]),
    )
    if int(getattr(factory, "qwen_calls", 0)) != 0:
        raise RuntimeError("diagnostic smoke made a forbidden Qwen call")
    episode_path = (
        output
        / "diagnostic_episodes"
        / DIAGNOSTIC_SMOKE_SAMPLE_ID
        / "diagnostic_smoke.json"
    )
    base._atomic_write_json(episode_path, record)
    identity = {
        "episode_id": DIAGNOSTIC_SMOKE_SAMPLE_ID,
        "path": str(episode_path.relative_to(output)),
        "sha256": base.sha256_file(episode_path),
        "canonical_sha256": base.canonical_sha256(record),
        "size_bytes": episode_path.stat().st_size,
        "diagnostic_smoke": True,
        "scientific_result_authoritative": False,
    }
    actual = _as_mapping(record["actual_execution"], "actual_execution")
    counts = {
        "episodes": 1,
        "candidates": int(actual["candidate_count"]),
        "candidates_executed": int(actual["candidate_count"]),
        "sam3_box_candidates_executed": sum(
            tool == "sam3" and action == "box"
            for tool, action in zip(actual["tool_types"], actual["action_types"])
        ),
        "simpleclick_positive_point_candidates_executed": sum(
            tool == "simpleclick" and action == "positive_point"
            for tool, action in zip(actual["tool_types"], actual["action_types"])
        ),
    }
    diagnostic_contract = {
        "sample_id": DIAGNOSTIC_SMOKE_SAMPLE_ID,
        "fresh_sam3_initialization": True,
        "geometry_modes_executed": ["matched"],
        "step_indices_executed": [DIAGNOSTIC_SMOKE_STEP_INDEX],
        "max_real_candidates": DIAGNOSTIC_SMOKE_MAX_REAL_CANDIDATES,
        "independent_candidates_from_same_fresh_state": True,
        "endpoint_search_performed": False,
        "candidate_selection_or_replay_performed": False,
    }
    common = {
        "status": "completed",
        "diagnostic_smoke": True,
        "scientific_result_authoritative": False,
        "output_scope": "diagnostics_only",
        "teacher_only": True,
        "training_authorized": False,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "episode_ids": [DIAGNOSTIC_SMOKE_SAMPLE_ID],
        "runtime_commit": runtime_commit,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "diagnostic_contract": diagnostic_contract,
        "executed_tool_types": list(actual["tool_types"]),
        "executed_action_types": list(actual["action_types"]),
        "all_gt_geometry_authorized": actual["all_gt_geometry_authorized"],
        "all_real_tool_receipts_present": actual[
            "all_real_tool_receipts_present"
        ],
        "hard_gate_flags_observed": actual["hard_gate_flags_observed"],
        "rgb_semantic_gate_flags_observed": actual[
            "rgb_semantic_gate_flags_observed"
        ],
        "counts": counts,
    }
    index = {
        "schema_id": DIAGNOSTIC_SMOKE_INDEX_SCHEMA_ID,
        **common,
        "config": {
            "path": str(config_path),
            "sha256": base.sha256_file(config_path),
            "canonical_sha256": base.canonical_sha256(manifest),
        },
        "base_config_0803": {
            "path": str(contract.base_config_path),
            "sha256": base.sha256_file(contract.base_config_path),
            "canonical_sha256": base.canonical_sha256(contract.base_manifest),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "episodes": [identity],
    }
    index_path = output / "shard_index.json"
    base._atomic_write_json(index_path, index)
    marker = {
        "schema_id": DIAGNOSTIC_SMOKE_COMPLETE_SCHEMA_ID,
        **common,
        "shard_index_artifact": {
            "path": str(index_path.relative_to(output)),
            "sha256": base.sha256_file(index_path),
            "canonical_sha256": base.canonical_sha256(index),
            "size_bytes": index_path.stat().st_size,
        },
        "shard_index_sha256": base.sha256_file(index_path),
    }
    base._atomic_write_json(output / "generation_complete.json", marker)
    return marker


def run(
    config_path: Path,
    output: Path,
    runtime_commit: str,
    *,
    shard_index: int,
    num_shards: int,
    runtime_factory: base.EpisodeRuntimeFactory | None = None,
    label_loader: Callable[[base.LabelSource], np.ndarray] = base.load_label_mask,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    manifest, contract = load_contract(config_path)
    runtime_commit = base._runtime_commit(runtime_commit)
    output = base._prepare_output(output)
    factory = runtime_factory or base.ProductionRuntimeFactory(
        base.build_runtime_factory_config(contract.base_manifest)
    )
    assigned = _assigned_samples(contract, shard_index, num_shards)
    specs = _sample_specs(contract)
    base_protocol = _as_mapping(
        contract.base_manifest.get("execution_protocol", {}), "base execution_protocol"
    )
    query = str(base_protocol.get("query", "building"))
    identities: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, sample_id in enumerate(assigned):
        raw_spec = specs[sample_id]
        parsed, label_source = base.parse_pilot_episode(raw_spec, index, query)
        runtime_episode = base.load_runtime_episode(parsed)
        crop = _as_mapping(raw_spec.get("crop", {}), f"samples.{sample_id}.crop")
        record = run_episode(
            base.PilotEpisode(runtime_episode, label_source),
            factory,
            output,
            contract,
            label_loader=label_loader,
            expected_gt_pixels=int(crop["gt_pixels"]),
        )
        episode_path = output / "episodes" / sample_id / "episode.json"
        base._atomic_write_json(episode_path, record)
        identity = {
            "episode_id": sample_id,
            "path": str(episode_path.relative_to(output)),
            "sha256": base.sha256_file(episode_path),
            "canonical_sha256": base.canonical_sha256(record),
            "size_bytes": episode_path.stat().st_size,
            "episode_status": record["episode_status"],
            "winning_track": record["adaptive_execution"]["winning_track"],
            "conclusion": record["conclusion"]["classification"],
        }
        identities.append(identity)
        records.append(record)
    if int(getattr(factory, "qwen_calls", 0)) != 0:
        raise RuntimeError("teacher shard made a forbidden Qwen call")
    counts = {
        "episodes": len(records),
        "teacher_endpoint_success": sum(
            item["episode_status"] == "teacher_endpoint_success" for item in records
        ),
        "teacher_endpoint_not_reached": sum(
            item["episode_status"] == "teacher_endpoint_not_reached" for item in records
        ),
        "executed_tracks": sum(
            len(item["adaptive_execution"]["executed_tracks"]) for item in records
        ),
        "selected_steps": sum(
            sum(len(track["steps"]) for track in item["tracks"].values())
            for item in records
        ),
        "candidates": sum(
            sum(
                sum(round_["candidate_count"] for round_ in track["selection_rounds"])
                for track in item["tracks"].values()
            )
            for item in records
        ),
    }
    index = {
        "schema_id": INDEX_SCHEMA_ID,
        "status": "completed",
        "teacher_only": True,
        "training_authorized": False,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "episode_ids": assigned,
        "config": {
            "path": str(config_path),
            "sha256": base.sha256_file(config_path),
            "canonical_sha256": base.canonical_sha256(manifest),
        },
        "base_config_0803": {
            "path": str(contract.base_config_path),
            "sha256": base.sha256_file(contract.base_config_path),
            "canonical_sha256": base.canonical_sha256(contract.base_manifest),
        },
        "runtime_commit": runtime_commit,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "qwen_invoked": False,
        "qwen_calls": 0,
        "counts": counts,
        "episodes": identities,
    }
    index_path = output / "shard_index.json"
    base._atomic_write_json(index_path, index)
    marker = {
        "schema_id": COMPLETE_SCHEMA_ID,
        "status": "completed",
        "teacher_only": True,
        "training_authorized": False,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "episode_ids": assigned,
        "runtime_commit": runtime_commit,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "counts": counts,
        "shard_index_artifact": {
            "path": str(index_path.relative_to(output)),
            "sha256": base.sha256_file(index_path),
            "canonical_sha256": base.canonical_sha256(index),
            "size_bytes": index_path.stat().st_size,
        },
        "shard_index_sha256": base.sha256_file(index_path),
    }
    base._atomic_write_json(output / "generation_complete.json", marker)
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument(
        "--diagnostic-smoke",
        action="store_true",
        help=(
            "run only the fixed train_407 step-1 dependency smoke: matched "
            "geometry, one SAM3 box and one SimpleClick positive point"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diagnostic_smoke:
        marker = run_diagnostic_smoke(
            args.config,
            args.output,
            args.runtime_commit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        print(
            "status=complete;diagnostic_smoke=true;"
            "scientific_result_authoritative=false;teacher_only=true;"
            "training_authorized=false;qwen_calls=0;"
            f"episodes={marker['counts']['episodes']};"
            f"candidates_executed={marker['counts']['candidates_executed']};"
            f"tools={','.join(marker['executed_tool_types'])}"
        )
    else:
        marker = run(
            args.config,
            args.output,
            args.runtime_commit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        print(
            "status=complete;teacher_only=true;training_authorized=false;qwen_calls=0;"
            f"episodes={marker['counts']['episodes']};"
            f"teacher_endpoint_success={marker['counts']['teacher_endpoint_success']}"
        )


if __name__ == "__main__":
    main()
