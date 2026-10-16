#!/usr/bin/env python3
"""Build strict v12.1 few-shot SFT data from manually curated LEVIR regions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from change_agent.adapters.stage_backends import (
    _stage_images,
    _stage_prompt,
    _stage_visual_input_scope,
)
from change_agent.adapters.staged_verifier import (
    StagedQwenVerifier,
    _parse_atomic_audit,
    _parse_judgment,
    _plan_agent_action,
)
from change_agent.state import ChangeState
from change_agent.verifier_regions import attach_verifier_regions
from change_agent.verifier_protocol import EvidenceRecord
from tools.prepare_qwen3vl_v121_curation import component_box, components


DATASET_NAMES = ("levir_qwen3vl_v121_train_v1", "levir_qwen3vl_v121_val_v1")
CHECKS = (
    "evidence_sufficient", "target_class_only", "white_pixels_supported",
    "boundary_alignment", "internal_holes_absent",
    "changed_object_extent_complete", "fragment_artifacts_absent",
)
WHITE_NO_TRANSITION_VALIDATION_ERROR = (
    "a white region with no target transition cannot pass as correct"
)
V6_FROZEN_CORE_HANDOFF_SHA256 = (
    "<hash omitted>"
)
V6_FROZEN_CORE_CANONICAL_SHA256 = (
    "<hash omitted>"
)
V6_CORE_CONTENT_FIELDS = (
    "id",
    "prompt_sha256",
    "target_sha256",
    "image_sha256",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curation-root", type=Path, required=True)
    p.add_argument("--curation-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-commit", required=True)
    p.add_argument("--seed", type=int, default=801)
    p.add_argument(
        "--round",
        choices=(
            "v1",
            "v2-hardcase",
            "v3-action",
            "v4-residual",
            "v5-contrast-repair",
            "v6-matched-contrast",
            "v7-weighted-contrast",
        ),
        default="v1",
    )
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_core_content_sha256(records: list[dict[str, Any]]) -> str:
    """Hash every model-visible v3 core component in deterministic row order.

    The historical handoff recorded a separate ``e10dee1...`` receipt without
    preserving its serialization recipe.  This reproducible companion receipt
    covers each row id, prompt, target, and ordered image bytes through the
    hashes already stored in the manifest.  It is pinned to Job 52132 and lets
    v6 fail closed on any real content drift instead of trusting only an id set.
    """
    payload = [
        {field: record[field] for field in V6_CORE_CONTENT_FIELDS}
        for record in records
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalized_point(x: int, y: int) -> tuple[int, int]:
    return min(999, max(0, round(x * 1000 / 256))), min(999, max(0, round(y * 1000 / 256)))


def normalized_box(box: tuple[int, int, int, int], padding: int = 10) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (*normalized_point(max(0, x1 - padding), max(0, y1 - padding)),
            *normalized_point(min(255, x2 - 1 + padding), min(255, y2 - 1 + padding)))


def seed_for(mask: np.ndarray) -> tuple[int, int]:
    x, y = pixel_seed_for(mask)
    return normalized_point(x, y)


def pixel_seed_for(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("cannot choose a seed for an empty mask")
    mid = len(xs) // 2
    return int(xs[mid]), int(ys[mid])


def scan_grid_geometry(index: int) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Return the exact 3x3 normalized geometry used by the frozen runtime."""
    if not 0 <= index < 9:
        raise ValueError(index)
    row, column = divmod(index, 3)
    starts = (0, 333, 667)
    ends = (329, 663, 1000)
    centers = (165, 498, 831)
    box = (starts[column], starts[row], ends[column], ends[row])
    return box, (centers[column], centers[row])


def unchanged_transition_case(
    t1: np.ndarray,
    t2: np.ndarray,
    component: np.ndarray,
    *,
    source_view: str,
) -> dict[str, Any]:
    """Build a production-shaped one-sided SAM transition for an unchanged scene.

    The selected component is copied from one real SAM3 view.  The other view is
    empty in the local state, so the change is a genuine one-sided XOR rather
    than the old fixed synthetic square.  Removing that component returns the
    local state to an unchanged pair and is the positive ``delta_removed``
    candidate example.
    """
    if source_view not in {"t1", "t2"}:
        raise ValueError(source_view)
    empty = np.zeros_like(component, dtype=bool)
    t1_mask = component if source_view == "t1" else empty
    t2_mask = component if source_view == "t2" else empty
    false_transition = ChangeState(
        t1, t2, "building", t1_mask, t2_mask, component
    )
    corrected = ChangeState(
        t1, t2, "building", empty, empty, np.zeros_like(component)
    )
    return {
        "false_transition": false_transition,
        "corrected": corrected,
        "target_view": source_view,
        "corrective_action": "negative_point",
        "corrective_delta": "delta_removed",
    }


def directional_component(
    sam1: np.ndarray,
    sam2: np.ndarray,
    source_view: str,
    rank: int,
    expected_area: int | None = None,
) -> np.ndarray:
    """Select a deterministic one-sided connected component from SAM3 masks."""
    if source_view == "t1":
        source = np.logical_and(sam1, ~sam2)
    elif source_view == "t2":
        source = np.logical_and(sam2, ~sam1)
    else:
        raise ValueError(source_view)
    ordered = sorted(components(source), key=lambda item: int(item.sum()), reverse=True)
    if not 0 <= rank < len(ordered):
        raise ValueError(
            f"{source_view} component rank {rank} is unavailable (count={len(ordered)})"
        )
    selected = ordered[rank]
    if expected_area is not None and int(selected.sum()) != int(expected_area):
        raise ValueError(
            f"{source_view} component rank {rank} area changed: "
            f"expected {expected_area}, got {int(selected.sum())}"
        )
    return selected


def component_sha256(component: np.ndarray) -> str:
    """Hash a binary component with an explicit shape/dtype contract."""
    mask = np.ascontiguousarray(np.asarray(component, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(f"{mask.shape[0]}x{mask.shape[1]}:uint8:".encode())
    digest.update(mask.tobytes())
    return digest.hexdigest()


def residual_component(
    sam1: np.ndarray,
    sam2: np.ndarray,
    gt: np.ndarray,
    *,
    source_view: str,
    rank: int,
    expected_area: int,
    expected_gt_overlap: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve one pinned, GT-disjoint, one-sided SAM3 residual.

    The rank is defined by :func:`directional_component`: descending component
    area inside ``sam(source_view) & ~sam(other_view)``.  The explicit area and
    overlap locks make a silent SAM artifact or curation-root drift fail closed.
    GT is used only for this offline assertion and is never put in a prompt.
    """
    component = directional_component(
        sam1,
        sam2,
        source_view,
        rank,
        expected_area,
    )
    area = int(component.sum())
    overlap = int(np.logical_and(component, np.asarray(gt, dtype=bool)).sum())
    if overlap != int(expected_gt_overlap):
        raise ValueError(
            f"{source_view} component rank {rank} GT overlap changed: "
            f"expected {expected_gt_overlap}, got {overlap}"
        )
    if not 200 <= area <= 2000:
        raise ValueError(
            f"v4 residual component area must be in [200, 2000], got {area}"
        )
    return component, {
        "source_view": source_view,
        "component_rank": int(rank),
        "component_area": area,
        "component_box_xyxy": list(component_box(component)),
        "component_sha256": component_sha256(component),
        "gt_change_overlap_pixels": overlap,
    }


def environment_record(
    state: ChangeState,
    previous_state: ChangeState | None = None,
    *,
    expected_kind: str,
) -> tuple[EvidenceRecord, dict[str, Any]]:
    """Return one typed record directly from frozen Environment proposals."""
    proposals = attach_verifier_regions(
        state,
        previous_state,
        max_regions=32,
        max_delta_regions=3,
        min_component_area=1,
    )
    matches = [
        proposal
        for proposal in proposals
        if str(proposal.get("audit_kind")) == expected_kind
        or (
            expected_kind == "delta_added"
            and proposal.get("effect_kind") == "added"
        )
        or (
            expected_kind == "delta_removed"
            and proposal.get("effect_kind") == "removed"
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {expected_kind} proposal, found {len(matches)}"
        )
    proposal = matches[0]
    return EvidenceRecord.from_proposal(proposal), proposal


def scan_environment_record(
    state: ChangeState, index: int
) -> tuple[EvidenceRecord, dict[str, Any]]:
    """Return one exact 3x3 empty-mask scan tile from the runtime."""
    proposals = attach_verifier_regions(
        state,
        max_regions=9,
        min_component_area=1,
        empty_mask_scan_grid_size=3,
    )
    region_id = f"s{index}"
    matches = [p for p in proposals if p.get("region_id") == region_id]
    if len(matches) != 1:
        raise ValueError(f"missing production scan tile {region_id}")
    proposal = matches[0]
    return EvidenceRecord.from_proposal(proposal), proposal


def with_action_locality(
    state: ChangeState,
    proposal: dict[str, Any],
    *,
    action: str,
) -> ChangeState:
    """Attach the same locality facts that the Environment records after a click."""
    if action not in {"positive_point", "negative_point", "box"}:
        raise ValueError(action)
    result = state.clone()
    box = tuple(int(value) for value in proposal["box_pixels"])
    result.evidence.update(
        {
            "locality": {
                "composition_mode": (
                    "merge_clicked_prediction_component_within_proposal_roi"
                    if action == "positive_point"
                    else "remove_simpleclick_delta_within_clicked_component"
                    if action == "negative_point"
                    else "replace_box_roi_only"
                ),
                "roi_xyxy": list(box),
            },
            "tool_input": {
                "coordinate": list(proposal["component_seed_pixels"])
                if action != "box"
                else None,
                "is_positive": action == "positive_point",
            },
        }
    )
    return result


def checklist(kind: str) -> dict[str, str]:
    if kind == "correct":
        return {name: "pass" for name in CHECKS}
    if kind == "false_negative":
        return {
            "evidence_sufficient":"pass", "target_class_only":"pass",
            "white_pixels_supported":"not_applicable", "boundary_alignment":"not_applicable",
            "internal_holes_absent":"not_applicable", "changed_object_extent_complete":"fail",
            "fragment_artifacts_absent":"not_applicable",
        }
    if kind == "false_positive":
        return {
            "evidence_sufficient":"pass", "target_class_only":"fail",
            "white_pixels_supported":"fail", "boundary_alignment":"fail",
            "internal_holes_absent":"not_applicable", "changed_object_extent_complete":"pass",
            "fragment_artifacts_absent":"pass",
        }
    if kind == "false_positive_background":
        return {
            "evidence_sufficient":"pass", "target_class_only":"fail",
            "white_pixels_supported":"fail", "boundary_alignment":"fail",
            "internal_holes_absent":"not_applicable", "changed_object_extent_complete":"not_applicable",
            "fragment_artifacts_absent":"pass",
        }
    if kind == "false_positive_transition":
        return {
            "evidence_sufficient":"pass", "target_class_only":"pass",
            "white_pixels_supported":"fail", "boundary_alignment":"fail",
            "internal_holes_absent":"not_applicable", "changed_object_extent_complete":"not_applicable",
            "fragment_artifacts_absent":"pass",
        }
    raise ValueError(kind)


def audit_target(region_id: str, annotation: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "region_id": region_id,
        "visual_judgment": {"t1_state":annotation["t1_state"], "t2_state":annotation["t2_state"], "evidence":annotation["evidence"]},
        "diagnosis": {"audit_checklist": checklist(kind)},
    }


def record_for(region_id: str, audit_kind: str, box: tuple[int,int,int,int], seed: tuple[int,int], state: ChangeState, area: int) -> EvidenceRecord:
    x, y = seed
    px = min(255, round(x * 256 / 1000)); py = min(255, round(y * 256 / 1000))
    return EvidenceRecord(
        region_id=region_id, audit_kind=audit_kind,
        change_mask_state="white" if audit_kind in {"present","delta_added"} else "black",
        temporal_difference_state="present" if bool(np.logical_xor(state.t1_mask,state.t2_mask).any()) else "absent",
        box_normalized_1000=box, component_seed_normalized_1000=seed,
        t1_mask_pixels=int(state.t1_mask.sum()), t2_mask_pixels=int(state.t2_mask.sum()), component_area=area,
        editable_seed_white={"t1":bool(state.t1_mask[py,px]), "t2":bool(state.t2_mask[py,px])},
        allowed_actions=("positive_point","negative_point","box"), transition_mask_facts={},
    )


def save_example(output: Path, split: str, sample_id: str, stage: str, attempt: str,
                 state: ChangeState, previous: ChangeState | None, payload: dict[str,Any],
                 target: dict[str,Any], validation_error: str | None, metadata: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    images = _stage_images(stage, state, previous, payload, validation_error)
    image_paths=[]; prompt_parts=[]
    for idx,(label,image) in enumerate(images):
        path=(output/"images"/f"{sample_id}_{idx}.png").resolve(); image.save(path)
        image_paths.append(str(path)); prompt_parts.append(f"{label}\n<image>")
    prompt=_stage_prompt(stage,payload,validation_error); prompt_parts.append(prompt)
    target_text=json.dumps(target,ensure_ascii=False,separators=(",",":"))
    row={"id":sample_id,"conversations":[{"from":"human","value":"\n".join(prompt_parts)},{"from":"gpt","value":target_text}],"images":image_paths}
    scope=_stage_visual_input_scope(stage,payload,validation_error)
    manifest={**metadata,"id":sample_id,"split":split,"stage":stage,"attempt_kind":attempt,"visual_input_scope":scope,
              "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"target_sha256":hashlib.sha256(target_text.encode()).hexdigest(),
              "image_sha256":[sha256_file(Path(p)) for p in image_paths]}
    region=payload.get("region")
    if isinstance(region, dict):
        manifest.update({
            "region_id": region.get("region_id"),
            "region_audit_kind": region.get("audit_kind"),
            "region_component_area": region.get("component_area"),
            "region_transition_mask_facts": region.get("transition_mask_facts", {}),
        })
    action=payload.get("attempted_action")
    if isinstance(action, dict):
        manifest["attempted_action"] = dict(action)
    return row,manifest


def append_v6_true_transition_pair(
    *,
    output: Path,
    rows: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    candidate_id: str,
    id_prefix: str,
    state: ChangeState,
    component: np.ndarray,
    annotation: dict[str, Any],
    metadata: dict[str, Any],
    target_view: str,
    expected_component_area: int,
) -> None:
    """Append a correct repair and the harmful removal it must reject.

    The repair is the semantic counterweight to a background/background
    false-positive repair: a real T1/T2 transition stays all-pass.  The paired
    candidate replays the erroneous negative-point path and proves, using the
    frozen runtime assessor, that deleting the real transition is ``worse``.
    """
    area = int(component.sum())
    if area != int(expected_component_area):
        raise ValueError(
            f"v6 true-transition area changed for {candidate_id}: "
            f"expected {expected_component_area}, got {area}"
        )
    expected_view = (
        "t1"
        if annotation["t1_state"] == "building"
        and annotation["t2_state"] == "background"
        else "t2"
        if annotation["t2_state"] == "building"
        and annotation["t1_state"] == "background"
        else None
    )
    if target_view != expected_view:
        raise ValueError(
            f"v6 true-transition target mismatch for {candidate_id}: "
            f"expected {expected_view}, got {target_view}"
        )

    record, proposal = environment_record(state, expected_kind="present")
    payload = {
        "target_class": "building",
        "region": record.to_dict(),
        "proposal_catalog": [record.to_dict()],
        "schema": "environment_ordered_compact_region_audit_v12_1",
        "visual_context": "proposal",
    }
    repair_target = audit_target(record.region_id, annotation, "correct")
    _, repair_diagnosis = _parse_atomic_audit(repair_target, record)
    repair_plan = StagedQwenVerifier._plan(
        None, state, (record,), repair_diagnosis
    )
    if (
        repair_diagnosis.error_type != "none"
        or repair_plan is None
        or repair_plan.action != "finish"
        or repair_plan.target_view is not None
    ):
        raise ValueError(
            f"v6 true-transition repair did not derive finish: {candidate_id}"
        )

    repair_id = f"{id_prefix}_repair_matched_true_transition_{target_view}"
    candidate_sample_id = (
        f"{id_prefix}_candidate_harm_matched_true_transition_"
        f"delta_removed_{target_view}"
    )
    pair_meta = {
        **metadata,
        "construction": "production_matched_true_transition_v6",
        "v6_extension": True,
        "target_view": target_view,
        "expected_component_area": area,
        "origin_validation_error": WHITE_NO_TRANSITION_VALIDATION_ERROR,
    }
    row, item = save_example(
        output,
        "train",
        repair_id,
        "audit",
        "repair",
        state,
        None,
        payload,
        repair_target,
        WHITE_NO_TRANSITION_VALIDATION_ERROR,
        {
            **pair_meta,
            "v6_example_kind": "true_transition_repair",
            "paired_candidate_id": candidate_sample_id,
            "expected_repair_action": "finish",
        },
    )
    rows["train"].append(row)
    records.append(item)

    # Reconstruct the precise erroneous FP plan that the contrast candidate
    # guards against.  Its one-sided white seed must localize the negative
    # point to the real building view even though the RGB states truly differ.
    wrong_target = audit_target(
        record.region_id, annotation, "false_positive_transition"
    )
    _, wrong_diagnosis = _parse_atomic_audit(wrong_target, record)
    wrong_plan = StagedQwenVerifier._plan(
        None, state, (record,), wrong_diagnosis
    )
    if (
        wrong_diagnosis.error_type != "false_positive_change"
        or wrong_diagnosis.target_view != target_view
        or wrong_plan is None
        or wrong_plan.action != "negative_point"
        or wrong_plan.target_view != target_view
    ):
        raise ValueError(
            f"v6 counterfactual FP path did not derive {target_view} "
            f"negative point: {candidate_id}"
        )
    attempted_action = _plan_agent_action(wrong_plan, state.image_size)

    harmful_t1 = state.t1_mask.copy()
    harmful_t2 = state.t2_mask.copy()
    if target_view == "t1":
        harmful_t1[component] = False
    else:
        harmful_t2[component] = False
    harmful = ChangeState(
        state.t1_image,
        state.t2_image,
        state.query,
        harmful_t1,
        harmful_t2,
        np.zeros_like(component),
    )
    harmful = with_action_locality(harmful, proposal, action="negative_point")
    removed, _ = environment_record(
        harmful, state, expected_kind="delta_removed"
    )
    candidate_payload = {
        "target_class": "building",
        "region": removed.to_dict(),
        "proposal_catalog": [removed.to_dict()],
        "schema": "candidate_transition_evidence_v1",
        "visual_context": "proposal",
        "attempted_action": attempted_action.to_dict(),
        "persisted_initial_diagnosis": wrong_diagnosis.to_dict(),
    }
    candidate_target = {
        "region_id": removed.region_id,
        "visual_judgment": {
            "t1_state": annotation["t1_state"],
            "t2_state": annotation["t2_state"],
            "evidence_quality": "clear",
        },
    }
    judgment = _parse_judgment(candidate_target, removed.region_id)
    verifier = object.__new__(StagedQwenVerifier)
    verifier._planned_diagnosis = wrong_diagnosis
    verifier._planned_action = attempted_action
    assessment = verifier._assess_candidate_transition(
        (removed,), (judgment,), attempted_action
    )
    if (
        not assessment.evidence_sufficient
        or assessment.intended_error_improved
        or not assessment.introduced_false_negative
        or assessment.comparison != "worse"
    ):
        raise ValueError(
            f"v6 true-transition removal was not runtime-derived harm: {candidate_id}"
        )
    row, item = save_example(
        output,
        "train",
        candidate_sample_id,
        "candidate_evidence",
        "initial",
        harmful,
        state,
        candidate_payload,
        candidate_target,
        None,
        {
            **pair_meta,
            "v6_example_kind": "true_transition_harmful_removal",
            "candidate_outcome": "harm",
            "expected_candidate_comparison": "worse",
            "paired_repair_id": repair_id,
        },
    )
    rows["train"].append(row)
    records.append(item)


def sampling_content_key(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    """Return the model-visible identity used by the v7 no-new-content gate."""
    return (
        str(record["prompt_sha256"]),
        str(record["target_sha256"]),
        tuple(str(value) for value in record["image_sha256"]),
    )


def append_v7_sampling_repeats(
    *,
    rows: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    sampling_spec: dict[str, Any],
) -> dict[str, Any]:
    """Repeat the ten v6 extension rows without inventing model-visible content."""
    expected_unique=int(sampling_spec["expected_unique_train_rows"])
    expected_val=int(sampling_spec["expected_val_rows"])
    expected_sources=int(sampling_spec["expected_weighted_source_rows"])
    repeat_copies=int(sampling_spec["sampling_repeat_copies"])
    weight_factor=int(sampling_spec["sampling_weight_factor"])
    expected_exposures=int(sampling_spec["expected_train_exposures"])
    selector_field=str(sampling_spec["repeat_selector_field"])
    if sampling_spec.get("base_round") != "v6-matched-contrast":
        raise ValueError("v7 sampling must reuse the v6-matched-contrast recipe")
    if selector_field != "v6_extension":
        raise ValueError("v7 repeat selector must remain v6_extension")
    if repeat_copies != 2 or weight_factor != 3:
        raise ValueError("v7 requires two repeats and an effective weight factor of three")
    if 1 + repeat_copies != weight_factor:
        raise ValueError("v7 repeat count does not equal its sampling weight factor")
    if expected_exposures != expected_unique + expected_sources * repeat_copies:
        raise ValueError("v7 configured exposure arithmetic is inconsistent")

    unique_train=[item for item in records if item["split"] == "train"]
    val_records=[item for item in records if item["split"] == "val"]
    if len(unique_train) != expected_unique or len(rows["train"]) != expected_unique:
        raise ValueError("v7 base must contain exactly the 63 unique v6 train rows")
    if len(val_records) != expected_val or len(rows["val"]) != expected_val:
        raise ValueError("v7 validation rows drifted from v6")
    unique_keys=[sampling_content_key(item) for item in unique_train]
    if len(set(unique_keys)) != expected_unique:
        raise ValueError("v7 base train rows do not contain 63 unique contents")
    weighted_sources=[item for item in unique_train if item.get(selector_field) is True]
    if len(weighted_sources) != expected_sources:
        raise ValueError(
            f"v7 expected ten weighted v6 extensions, got {len(weighted_sources)}"
        )
    if Counter(item.get("construction") for item in weighted_sources) != Counter({
        "production_residual_fp_contract_repair_v4":6,
        "production_matched_true_transition_v6":4,
    }):
        raise ValueError("v7 weighted sources are not the six residual plus four matched rows")

    row_by_id={row["id"]:row for row in rows["train"]}
    if set(row_by_id) != {item["id"] for item in unique_train}:
        raise ValueError("v7 base row/manifest ids disagree")
    source_ids=[str(item["id"]) for item in weighted_sources]
    unique_train_content_sha256=canonical_core_content_sha256(unique_train)
    weighted_source_content_sha256=canonical_core_content_sha256(weighted_sources)
    repeat_records=[]
    for source_record in weighted_sources:
        source_id=str(source_record["id"])
        source_row=row_by_id[source_id]
        source_construction=str(source_record["construction"])
        source_record.update({
            "v7_weighted_source":True,
            "sampling_repeat_count":repeat_copies,
            "sampling_weight_factor":weight_factor,
        })
        for repeat_index in range(1,repeat_copies+1):
            repeat_id=f"{source_id}__sampling_repeat_v7_{repeat_index}"
            repeat_row=copy.deepcopy(source_row)
            repeat_row["id"]=repeat_id
            if (
                repeat_row["conversations"] != source_row["conversations"]
                or repeat_row["images"] != source_row["images"]
            ):
                raise ValueError(f"v7 repeat content changed while copying {source_id}")
            repeat_record=copy.deepcopy(source_record)
            repeat_record.pop(selector_field,None)
            repeat_record.pop("v7_weighted_source",None)
            repeat_record.pop("sampling_repeat_count",None)
            repeat_record.update({
                "id":repeat_id,
                "construction":"sampling_weight_repeat_v7",
                "repeated_source_construction":source_construction,
                "repeated_from_id":source_id,
                "sampling_repeat_index":repeat_index,
                "sampling_weight_factor":weight_factor,
                "sampling_repeat_only":True,
                "unique_content":False,
            })
            if sampling_content_key(repeat_record) != sampling_content_key(source_record):
                raise ValueError(f"v7 repeat hashes changed for {source_id}")
            rows["train"].append(repeat_row)
            records.append(repeat_record)
            repeat_records.append(repeat_record)

    train_records=[item for item in records if item["split"] == "train"]
    if len(train_records) != expected_exposures or len(rows["train"]) != expected_exposures:
        raise ValueError("v7 train exposure count mismatch")
    if len({item["id"] for item in train_records}) != expected_exposures:
        raise ValueError("v7 train exposure ids are not unique")
    if len(repeat_records) != expected_sources * repeat_copies:
        raise ValueError("v7 sampling repeat row count mismatch")
    repeats_by_source=Counter(item["repeated_from_id"] for item in repeat_records)
    if repeats_by_source != Counter({source_id:repeat_copies for source_id in source_ids}):
        raise ValueError("v7 source repeat multiplicities changed")
    for source_id in source_ids:
        indexes={
            int(item["sampling_repeat_index"])
            for item in repeat_records
            if item["repeated_from_id"] == source_id
        }
        if indexes != {1,2}:
            raise ValueError(f"v7 sampling repeat indexes changed for {source_id}")
    content_counts=Counter(sampling_content_key(item) for item in train_records)
    if len(content_counts) != expected_unique:
        raise ValueError("v7 introduced new unique model-visible content")
    weighted_keys={sampling_content_key(item) for item in weighted_sources}
    if any(
        count != (weight_factor if key in weighted_keys else 1)
        for key,count in content_counts.items()
    ):
        raise ValueError("v7 effective per-content sampling weights changed")

    return {
        "schema":"qwen3vl_v121_sampling_receipt_v1",
        "strategy":"explicit_row_repetition",
        "base_round":"v6-matched-contrast",
        "unique_train_rows":expected_unique,
        "train_exposures":expected_exposures,
        "val_rows":expected_val,
        "weighted_source_rows":expected_sources,
        "repeat_rows":len(repeat_records),
        "sampling_repeat_copies":repeat_copies,
        "sampling_weight_factor":weight_factor,
        "unique_content_count":len(content_counts),
        "unique_train_content_sha256":unique_train_content_sha256,
        "weighted_source_content_sha256":weighted_source_content_sha256,
        "weighted_source_ids":source_ids,
        "repeat_content_exact":True,
        "test_data_used":False,
    }


def main() -> None:
    args=parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    manifest=json.loads((args.curation_root/"curation_manifest.json").read_text())
    config=json.loads(args.curation_config.read_text())
    annotations={x["candidate_id"]:x for x in config["samples"] if x.get("use") is True}
    hardcase=config.get("hardcase_v2", {})
    action_hardcase=config.get("hardcase_v3", {})
    residual_hardcase=config.get("hardcase_v4", {})
    contrast_hardcase=config.get("hardcase_v5", {})
    matched_hardcase=config.get("hardcase_v6", {})
    sampling_hardcase=config.get("hardcase_v7", {})
    hardcase_round=args.round in {
        "v2-hardcase", "v3-action", "v4-residual", "v5-contrast-repair",
        "v6-matched-contrast", "v7-weighted-contrast",
    }
    v3_round=args.round in {
        "v3-action", "v4-residual", "v5-contrast-repair",
        "v6-matched-contrast", "v7-weighted-contrast",
    }
    v4_round=args.round in {
        "v4-residual", "v5-contrast-repair", "v6-matched-contrast",
        "v7-weighted-contrast",
    }
    v5_round=args.round == "v5-contrast-repair"
    v6_round=args.round == "v6-matched-contrast"
    v7_round=args.round == "v7-weighted-contrast"
    matched_round=v6_round or v7_round
    transition_specs=action_hardcase.get("unchanged_transition_cases", [])
    transition_by_candidate={}
    for spec in transition_specs:
        transition_by_candidate.setdefault(spec["candidate_id"], []).append(spec)
    v3_scan_tile_ids={str(key): int(value) for key, value in action_hardcase.get("unchanged_scan_tile_ids", {}).items()}
    v3_scan_ids=set(v3_scan_tile_ids)
    v3_harm_ids=set(action_hardcase.get("delta_removed_harm_candidate_ids", []))
    v3_reverse_specs=action_hardcase.get("time_reverse_cases", [])
    v4_residual_specs=residual_hardcase.get("residual_false_positive_cases", [])
    v5_semantic_specs=contrast_hardcase.get("semantic_anchor_cases", [])
    v5_actionable_specs=contrast_hardcase.get("actionable_repair_cases", [])
    v5_semantic_by_candidate={str(spec["candidate_id"]):spec for spec in v5_semantic_specs}
    v5_actionable_by_key={
        (str(spec["candidate_id"]),str(spec["source_view"]),int(spec["component_rank"])):spec
        for spec in v5_actionable_specs
    }
    v6_core_spec=matched_hardcase.get("frozen_v3_core", {})
    v6_residual_specs=matched_hardcase.get("residual_false_positive_cases", [])
    v6_true_specs=matched_hardcase.get("true_transition_cases", [])
    v6_true_by_key={
        (str(spec["candidate_id"]),str(spec["variant"])):spec
        for spec in v6_true_specs
    }
    include_background_fp_pairs=(
        bool(matched_hardcase.get("include_background_fp_pairs"))
        if matched_round else True
    )
    v3_transition_ids=set(transition_by_candidate)
    sources={x["candidate_id"]:x for x in manifest["samples"]}
    if set(annotations)-set(sources): raise ValueError("curation config contains unknown candidates")
    if v4_round:
        if len(v4_residual_specs) != 2:
            raise ValueError(
                f"v4 requires exactly two residual cases, got {len(v4_residual_specs)}"
            )
        residual_ids=[str(spec.get("candidate_id")) for spec in v4_residual_specs]
        if len(set(residual_ids)) != len(residual_ids):
            raise ValueError("v4 residual cases must come from distinct candidates")
        if set(residual_ids)-set(sources):
            raise ValueError("v4 residual config contains unknown candidates")
    if v5_round:
        if len(v5_semantic_specs) != 2 or len(v5_actionable_specs) != 2:
            raise ValueError(
                "v5 requires exactly two semantic anchors and two actionable repairs"
            )
        v5_ids=[str(spec.get("candidate_id")) for spec in (*v5_semantic_specs,*v5_actionable_specs)]
        if len(set(v5_ids)) != 4:
            raise ValueError("v5 contrast cases must come from four distinct candidates")
        if set(v5_ids)-set(sources):
            raise ValueError("v5 contrast config contains unknown candidates")
    if matched_round:
        if matched_hardcase.get("base_round") != "v3-action":
            raise ValueError("v6 must be based on the frozen v3-action core")
        if include_background_fp_pairs:
            raise ValueError("v6 must exclude the six background FP pair rows")
        if int(v6_core_spec.get("expected_rows", -1)) != 53:
            raise ValueError("v6 frozen core must declare exactly 53 rows")
        if v6_core_spec.get("handoff_content_sha256") != V6_FROZEN_CORE_HANDOFF_SHA256:
            raise ValueError("v6 historical handoff core fingerprint changed")
        if v6_core_spec.get("canonical_content_sha256") != V6_FROZEN_CORE_CANONICAL_SHA256:
            raise ValueError("v6 canonical core fingerprint changed")
        if tuple(v6_core_spec.get("canonical_content_fields", ())) != V6_CORE_CONTENT_FIELDS:
            raise ValueError("v6 canonical core fingerprint fields changed")
        configured_residual_keys={
            (
                str(spec.get("candidate_id")),
                str(spec.get("source_view")),
                int(spec.get("component_rank", -1)),
            )
            for spec in v6_residual_specs
        }
        frozen_residual_keys={
            (
                str(spec.get("candidate_id")),
                str(spec.get("source_view")),
                int(spec.get("component_rank", -1)),
            )
            for spec in v4_residual_specs
        }
        if configured_residual_keys != frozen_residual_keys or len(v6_residual_specs) != 2:
            raise ValueError("v6 must reuse exactly the train51/T2 and train85/T1 residuals")
        expected_true_keys={
            ("train_train_407_r3c2_changed", "forward", "t2"),
            ("train_train_231_r3c2_changed", "time_reverse", "t1"),
        }
        configured_true_keys={
            (
                str(spec.get("candidate_id")),
                str(spec.get("variant")),
                str(spec.get("target_view")),
            )
            for spec in v6_true_specs
        }
        if configured_true_keys != expected_true_keys or len(v6_true_specs) != 2:
            raise ValueError("v6 true-transition contrast cases changed")
        if {str(spec.get("candidate_id")) for spec in v6_true_specs}-set(sources):
            raise ValueError("v6 true-transition config contains unknown candidates")
    if v7_round:
        expected_sampling_spec={
            "base_round":"v6-matched-contrast",
            "repeat_selector_field":"v6_extension",
            "expected_unique_train_rows":63,
            "expected_val_rows":24,
            "expected_weighted_source_rows":10,
            "sampling_repeat_copies":2,
            "sampling_weight_factor":3,
            "expected_train_exposures":83,
        }
        if sampling_hardcase != expected_sampling_spec:
            raise ValueError("v7 controlled sampling recipe changed")
    args.output.mkdir(parents=True); (args.output/"images").mkdir()
    rows={"train":[],"val":[]}; records=[]
    v6_core_receipt: dict[str, Any] | None = None
    v7_sampling_receipt: dict[str, Any] | None = None
    for cid,ann in annotations.items():
        src=sources[cid]; split=src["split"]; art={k:Path(v) for k,v in src["artifacts"].items()}
        t1=np.asarray(Image.open(art["t1"]).convert("RGB")); t2=np.asarray(Image.open(art["t2"]).convert("RGB"))
        sam1=np.asarray(Image.open(art["sam3_t1_mask"]).convert("L"))>0; sam2=np.asarray(Image.open(art["sam3_t2_mask"]).convert("L"))>0
        gt=np.asarray(Image.open(art["gt_change"]).convert("L"))>0
        meta={"parent_id":src["parent_id"],"candidate_id":cid,"source_kind":src["kind"],"construction":"frozen_sam3_plus_local_gt_audited_perturbation",
              "source_sha256":src["artifact_sha256"],"t1_state":ann["t1_state"],"t2_state":ann["t2_state"]}
        if src["kind"]=="changed":
            box0=tuple(src["component_box_xyxy"]); comps=components(gt)
            comp=max(comps,key=lambda m:int(m[box0[1]:box0[3],box0[0]:box0[2]].sum()))
            box=normalized_box(component_box(comp)); seed=seed_for(comp)
            m1=sam1.copy();m2=sam2.copy()
            if ann["t1_state"]=="building": m1|=comp
            else: m1&=~comp
            if ann["t2_state"]=="building": m2|=comp
            else: m2&=~comp
            correct=ChangeState(t1,t2,"building",m1,m2,comp)
            missing=ChangeState(t1,t2,"building",m1,m2,np.zeros_like(comp))
            scan=ChangeState(t1,t2,"building",np.zeros_like(comp),np.zeros_like(comp),np.zeros_like(comp))
            variants=[("audit_missing","audit","initial",missing,None,"missing","false_negative",None),
                      ("audit_correct","audit","initial",correct,None,"present","correct",None),
                      ("scan_false_negative","audit","initial",scan,None,"scan","false_negative",None)]
            if v3_round and split == "train":
                variants = [variant for variant in variants if variant[0] != "scan_false_negative"]
            elif hardcase_round and split == "train" and cid not in set(hardcase["keep_scan_positive_ids"]):
                variants = [variant for variant in variants if variant[0] != "scan_false_negative"]
            for suffix,stage,attempt,state,prev,akind,label_kind,error in variants:
                rec=record_for("r0",akind,box,seed,state,int(comp.sum()))
                payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                target=audit_target("r0",ann,label_kind); _parse_atomic_audit(target,rec)
                row,m=save_example(args.output,split,f"{cid}_{suffix}",stage,attempt,state,prev,payload,target,error,meta); rows[split].append(row);records.append(m)
            if matched_round and split == "train":
                true_spec=v6_true_by_key.get((str(cid),"forward"))
                if true_spec is not None:
                    append_v6_true_transition_pair(
                        output=args.output,
                        rows=rows,
                        records=records,
                        candidate_id=str(cid),
                        id_prefix=str(true_spec["id_prefix"]),
                        state=correct,
                        component=comp,
                        annotation=ann,
                        metadata=meta,
                        target_view=str(true_spec["target_view"]),
                        expected_component_area=int(true_spec["expected_component_area"]),
                    )
            if v5_round and split == "train" and cid in v5_semantic_by_candidate:
                spec=v5_semantic_by_candidate[cid]
                expected_area=int(spec["expected_component_area"])
                if int(comp.sum()) != expected_area:
                    raise ValueError(
                        f"v5 semantic anchor area changed for {cid}: "
                        f"expected {expected_area}, got {int(comp.sum())}"
                    )
                anchor_record,_=environment_record(correct,expected_kind="present")
                anchor_payload={"target_class":"building","region":anchor_record.to_dict(),"proposal_catalog":[anchor_record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                anchor_target=audit_target(anchor_record.region_id,ann,"correct")
                _,anchor_diagnosis=_parse_atomic_audit(anchor_target,anchor_record)
                anchor_plan=StagedQwenVerifier._plan(None,correct,(anchor_record,),anchor_diagnosis)
                if anchor_diagnosis.error_type != "none" or anchor_plan.action != "finish":
                    raise ValueError(f"v5 semantic anchor is not a valid finish repair: {cid}")
                anchor_meta={
                    **meta,
                    "construction":"production_contrast_repair_v5",
                    "v5_example_kind":"semantic_anchor_repair",
                    "origin_validation_error":WHITE_NO_TRANSITION_VALIDATION_ERROR,
                    "paired_candidate_id":str(spec["paired_candidate_id"]),
                    "expected_component_area":expected_area,
                }
                row,item=save_example(
                    args.output,split,f"{cid}_repair_contrast_semantic_anchor",
                    "audit","repair",correct,None,anchor_payload,anchor_target,
                    WHITE_NO_TRANSITION_VALIDATION_ERROR,anchor_meta,
                );rows[split].append(row);records.append(item)
            if v3_round and split == "train" and cid in set(hardcase["keep_scan_positive_ids"]):
                scan_index=int(action_hardcase.get("scan_positive_tile_ids", {}).get(cid, -1))
                if scan_index < 0:
                    sx,sy=pixel_seed_for(comp); scan_index=min(2, sy // 85) * 3 + min(2, sx // 85)
                scan_record,scan_proposal=scan_environment_record(scan, scan_index)
                scan_payload={"target_class":"building","region":scan_record.to_dict(),"proposal_catalog":[scan_record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                scan_target=audit_target(f"s{scan_index}",ann,"false_negative");scan_target["region_id"]=f"s{scan_index}";_parse_atomic_audit(scan_target,scan_record)
                scan_meta={**meta,"construction":"production_scale_scan_positive_v3","scan_tile_index":scan_index}
                row,item=save_example(args.output,split,f"{cid}_scan_production_s{scan_index}","audit","initial",scan,None,scan_payload,scan_target,None,scan_meta);rows[split].append(row);records.append(item)
            rec=record_for("r0","missing",box,seed,missing,int(comp.sum()))
            payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal",
                     "repair_context":{"mode":"atomic_evidence_only","origin_validation_error":"atomic audit evidence copied schema boilerplate","preserve_fields":["region_id","visual_judgment.t1_state","visual_judgment.t2_state","diagnosis.audit_checklist"]}}
            target={"region_id":"r0","visual_judgment":{"evidence":ann["evidence"]},"diagnosis":{}}
            err="atomic audit evidence copied schema boilerplate"
            row,m=save_example(args.output,split,f"{cid}_evidence_repair","audit","repair",missing,None,payload,target,err,meta);rows[split].append(row);records.append(m)
            if v3_round and split == "train":
                target_view="t1" if ann["t1_state"] == "building" else "t2"
                before_m1=m1.copy();before_m2=m2.copy()
                if target_view == "t1": before_m1[comp]=False
                else: before_m2[comp]=False
                candidate_previous=ChangeState(t1,t2,"building",before_m1,before_m2,np.zeros_like(comp))
                _,initial_proposal=environment_record(correct,expected_kind="present")
                candidate=with_action_locality(correct,initial_proposal,action="positive_point")
                delta,delta_proposal=environment_record(candidate,candidate_previous,expected_kind="delta_added")
                payload={"target_class":"building","region":delta.to_dict(),"proposal_catalog":[delta.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":target_view,"action":"positive_point"},"persisted_initial_diagnosis":{"error_type":"false_negative"}}
                target={"region_id":delta.region_id,"visual_judgment":{"t1_state":ann["t1_state"],"t2_state":ann["t2_state"],"evidence_quality":"clear"}};_parse_judgment(target,delta.region_id)
                candidate_meta={**meta,"candidate_outcome":"improvement","construction":"production_delta_added_v3","target_view":target_view}
                row,m=save_example(args.output,split,f"{cid}_candidate_improvement","candidate_evidence","initial",candidate,candidate_previous,payload,target,None,candidate_meta);rows[split].append(row);records.append(m)
            else:
                rec=record_for("d0","delta_added",box,seed,correct,int(comp.sum()))
                payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":"t2","action":"positive_point"},"persisted_initial_diagnosis":{"error_type":"false_negative"}}
                target={"region_id":"d0","visual_judgment":{"t1_state":ann["t1_state"],"t2_state":ann["t2_state"],"evidence_quality":"clear"}};_parse_judgment(target,"d0")
                row,m=save_example(args.output,split,f"{cid}_candidate_improvement","candidate_evidence","initial",correct,missing,payload,target,None,meta);rows[split].append(row);records.append(m)
            if v3_round and split == "train" and cid in v3_harm_ids:
                target_view="t1" if ann["t1_state"] == "building" else "t2"
                _,initial_proposal=environment_record(correct,expected_kind="present")
                harm_m1=m1.copy();harm_m2=m2.copy()
                if target_view == "t1": harm_m1[comp]=False
                else: harm_m2[comp]=False
                harmful=ChangeState(t1,t2,"building",harm_m1,harm_m2,np.zeros_like(comp))
                harmful=with_action_locality(harmful,initial_proposal,action="negative_point")
                delta,delta_proposal=environment_record(harmful,correct,expected_kind="delta_removed")
                payload={"target_class":"building","region":delta.to_dict(),"proposal_catalog":[delta.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":target_view,"action":"negative_point"},"persisted_initial_diagnosis":{"error_type":"false_positive_change"}}
                target={"region_id":delta.region_id,"visual_judgment":{"t1_state":ann["t1_state"],"t2_state":ann["t2_state"],"evidence_quality":"clear"}};_parse_judgment(target,delta.region_id)
                harm_meta={**meta,"candidate_outcome":"harm","construction":"production_delta_removed_harm_v3","target_view":target_view}
                row,m=save_example(args.output,split,f"{cid}_candidate_harm_delta_removed","candidate_evidence","initial",harmful,correct,payload,target,None,harm_meta);rows[split].append(row);records.append(m)
        else:
            if v3_round and split == "train" and cid in v3_scan_ids:
                empty=np.zeros((256,256),dtype=bool)
                scan=ChangeState(t1,t2,"building",empty,empty,empty)
                scan_index=v3_scan_tile_ids[cid]
                scan_record,scan_proposal=scan_environment_record(scan,scan_index)
                scan_payload={"target_class":"building","region":scan_record.to_dict(),"proposal_catalog":[scan_record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                scan_target=audit_target(f"s{scan_index}",ann,"correct");_parse_atomic_audit(scan_target,scan_record)
                scan_meta={**meta,"construction":"production_scale_scan_correct_v3","scan_tile_index":scan_index}
                row,item=save_example(args.output,split,f"{cid}_scan_correct_s{scan_index}","audit","initial",scan,None,scan_payload,scan_target,None,scan_meta);rows[split].append(row);records.append(item)
                for spec in transition_by_candidate.get(cid, []):
                    component=directional_component(sam1,sam2,spec["source_view"],int(spec["component_rank"]),spec.get("expected_area"))
                    case=unchanged_transition_case(t1,t2,component,source_view=spec["source_view"])
                    false_transition=case["false_transition"];corrected=case["corrected"];target_view=case["target_view"]
                    transition_meta={**meta,"construction":"production_one_sided_transition_v3","source_view":target_view,"component_rank":int(spec["component_rank"]),"component_area":int(component.sum())}
                    record,initial_proposal=environment_record(false_transition,expected_kind="present")
                    audit_payload={"target_class":"building","region":record.to_dict(),"proposal_catalog":[record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                    target=audit_target(record.region_id,ann,"false_positive_transition");_parse_atomic_audit(target,record)
                    row,item=save_example(args.output,split,f"{cid}_audit_false_transition_{target_view}_r{spec['component_rank']}","audit","initial",false_transition,None,audit_payload,target,None,transition_meta);rows[split].append(row);records.append(item)
                    corrected=with_action_locality(corrected,initial_proposal,action="negative_point")
                    delta,delta_proposal=environment_record(corrected,false_transition,expected_kind="delta_removed")
                    candidate_payload={"target_class":"building","region":delta.to_dict(),"proposal_catalog":[delta.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":target_view,"action":"negative_point"},"persisted_initial_diagnosis":{"error_type":"false_positive_change"}}
                    candidate_target={"region_id":delta.region_id,"visual_judgment":{"t1_state":ann["t1_state"],"t2_state":ann["t2_state"],"evidence_quality":"clear"}};_parse_judgment(candidate_target,delta.region_id)
                    candidate_meta={**transition_meta,"candidate_outcome":"improvement"}
                    row,item=save_example(args.output,split,f"{cid}_candidate_improvement_delta_removed_r{spec['component_rank']}","candidate_evidence","initial",corrected,false_transition,candidate_payload,candidate_target,None,candidate_meta);rows[split].append(row);records.append(item)
                    if v5_round:
                        v5_key=(str(cid),str(spec["source_view"]),int(spec["component_rank"]))
                        v5_spec=v5_actionable_by_key.get(v5_key)
                        if v5_spec is not None:
                            paired_id=f"{cid}_candidate_improvement_delta_removed_r{spec['component_rank']}"
                            if str(v5_spec["paired_candidate_id"]) != paired_id:
                                raise ValueError(f"v5 actionable pair id mismatch for {cid}")
                            if int(component.sum()) != int(v5_spec["expected_area"]):
                                raise ValueError(f"v5 actionable area changed for {cid}")
                            repair_target=audit_target(record.region_id,ann,"false_positive_transition")
                            _,repair_diagnosis=_parse_atomic_audit(repair_target,record)
                            repair_plan=StagedQwenVerifier._plan(None,false_transition,(record,),repair_diagnosis)
                            if (
                                repair_diagnosis.error_type != "false_positive_change"
                                or repair_diagnosis.target_view != target_view
                                or repair_plan is None
                                or repair_plan.action != "negative_point"
                                or repair_plan.target_view != target_view
                            ):
                                raise ValueError(f"v5 actionable repair did not derive negative point: {cid}")
                            repair_payload={"target_class":"building","region":record.to_dict(),"proposal_catalog":[record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                            repair_meta={
                                **transition_meta,
                                "construction":"production_contrast_repair_v5",
                                "v5_example_kind":"actionable_transition_repair",
                                "origin_validation_error":WHITE_NO_TRANSITION_VALIDATION_ERROR,
                                "paired_candidate_id":paired_id,
                            }
                            row,item=save_example(
                                args.output,"train",
                                f"{cid}_repair_contrast_actionable_{spec['source_view']}_r{spec['component_rank']}",
                                "audit","repair",false_transition,None,repair_payload,
                                repair_target,WHITE_NO_TRANSITION_VALIDATION_ERROR,
                                repair_meta,
                            );rows["train"].append(row);records.append(item)
                continue
            x1,y1,x2,y2=(104,104,152,152); blob=np.zeros((256,256),bool);blob[y1:y2,x1:x2]=True
            box=normalized_box((x1,y1,x2,y2),24);seed=seed_for(blob)
            m1=sam1.copy();m2=sam2.copy(); same=ann["t1_state"]=="building"
            m1[blob]=same;m2[blob]=same
            fp=ChangeState(t1,t2,"building",m1,m2,blob); base=ChangeState(t1,t2,"building",m1,m2,np.zeros_like(blob)); scan=ChangeState(t1,t2,"building",np.zeros_like(blob),np.zeros_like(blob),np.zeros_like(blob))
            for suffix,state,akind,label_kind in (("audit_false_positive",fp,"present","false_positive"),("scan_correct",scan,"scan","correct")):
                rec=record_for("r0",akind,box,seed,state,int(blob.sum()));payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                target=audit_target("r0",ann,label_kind);_parse_atomic_audit(target,rec)
                row,m=save_example(args.output,split,f"{cid}_{suffix}","audit","initial",state,None,payload,target,None,meta);rows[split].append(row);records.append(m)
            rec=record_for("d0","delta_added",box,seed,fp,int(blob.sum()));payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":"t2","action":"positive_point"},"persisted_initial_diagnosis":{"error_type":"none"}}
            target={"region_id":"d0","visual_judgment":{"t1_state":ann["t1_state"],"t2_state":ann["t2_state"],"evidence_quality":"clear"}};_parse_judgment(target,"d0")
            row,m=save_example(args.output,split,f"{cid}_candidate_harm","candidate_evidence","initial",fp,base,payload,target,None,meta);rows[split].append(row);records.append(m)

    if v3_round:
        # A small time-reversed mirror removes the v2 bias toward only
        # background->building / t2-positive actions without adding new parents.
        for spec in v3_reverse_specs:
            cid=spec["candidate_id"]
            src=sources[cid]
            if src["split"] != "train" or src["kind"] != "changed":
                raise ValueError(f"time-reverse source must be a changed train candidate: {cid}")
            ann=annotations[cid]
            art={key:Path(value) for key,value in src["artifacts"].items()}
            t1_original=np.asarray(Image.open(art["t1"]).convert("RGB"));t2_original=np.asarray(Image.open(art["t2"]).convert("RGB"))
            sam1_original=np.asarray(Image.open(art["sam3_t1_mask"]).convert("L"))>0;sam2_original=np.asarray(Image.open(art["sam3_t2_mask"]).convert("L"))>0
            gt=np.asarray(Image.open(art["gt_change"]).convert("L"))>0
            box0=tuple(src["component_box_xyxy"]);comps=components(gt);comp=max(comps,key=lambda m:int(m[box0[1]:box0[3],box0[0]:box0[2]].sum()))
            t1,t2=t2_original,t1_original;sam1,sam2=sam2_original,sam1_original
            reversed_ann={"t1_state":ann["t2_state"],"t2_state":ann["t1_state"],"evidence":spec["evidence"]}
            meta={"parent_id":src["parent_id"],"candidate_id":cid,"source_kind":src["kind"],"construction":"time_reversed_production_v3","source_sha256":src["artifact_sha256"],"t1_state":reversed_ann["t1_state"],"t2_state":reversed_ann["t2_state"],"time_reversed":True}
            m1=sam1.copy();m2=sam2.copy()
            if reversed_ann["t1_state"]=="building": m1|=comp
            else: m1&=~comp
            if reversed_ann["t2_state"]=="building": m2|=comp
            else: m2&=~comp
            correct=ChangeState(t1,t2,"building",m1,m2,comp);missing=ChangeState(t1,t2,"building",m1,m2,np.zeros_like(comp))
            box=normalized_box(component_box(comp));seed=seed_for(comp)
            for suffix,state,akind,label_kind in (("audit_missing",missing,"missing","false_negative"),("audit_correct",correct,"present","correct")):
                rec=record_for("r0",akind,box,seed,state,int(comp.sum()));payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                target=audit_target("r0",reversed_ann,label_kind);_parse_atomic_audit(target,rec)
                row,item=save_example(args.output,"train",f"{cid}_time_reverse_{suffix}","audit","initial",state,None,payload,target,None,meta);rows["train"].append(row);records.append(item)
            rec=record_for("r0","missing",box,seed,missing,int(comp.sum()));payload={"target_class":"building","region":rec.to_dict(),"proposal_catalog":[rec.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal","repair_context":{"mode":"atomic_evidence_only","origin_validation_error":"atomic audit evidence copied schema boilerplate","preserve_fields":["region_id","visual_judgment.t1_state","visual_judgment.t2_state","diagnosis.audit_checklist"]}}
            target={"region_id":"r0","visual_judgment":{"evidence":reversed_ann["evidence"]},"diagnosis":{}}
            row,item=save_example(args.output,"train",f"{cid}_time_reverse_evidence_repair","audit","repair",missing,None,payload,target,"atomic audit evidence copied schema boilerplate",meta);rows["train"].append(row);records.append(item)
            if matched_round:
                true_spec=v6_true_by_key.get((str(cid),"time_reverse"))
                if true_spec is not None:
                    append_v6_true_transition_pair(
                        output=args.output,
                        rows=rows,
                        records=records,
                        candidate_id=str(cid),
                        id_prefix=str(true_spec["id_prefix"]),
                        state=correct,
                        component=comp,
                        annotation=reversed_ann,
                        metadata=meta,
                        target_view=str(true_spec["target_view"]),
                        expected_component_area=int(true_spec["expected_component_area"]),
                    )
            target_view="t1" if reversed_ann["t1_state"]=="building" else "t2"
            before_m1=m1.copy();before_m2=m2.copy()
            if target_view=="t1": before_m1[comp]=False
            else: before_m2[comp]=False
            candidate_previous=ChangeState(t1,t2,"building",before_m1,before_m2,np.zeros_like(comp))
            _,initial_proposal=environment_record(correct,expected_kind="present")
            candidate=with_action_locality(correct,initial_proposal,action="positive_point")
            delta,delta_proposal=environment_record(candidate,candidate_previous,expected_kind="delta_added")
            payload={"target_class":"building","region":delta.to_dict(),"proposal_catalog":[delta.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":target_view,"action":"positive_point"},"persisted_initial_diagnosis":{"error_type":"false_negative"}}
            target={"region_id":delta.region_id,"visual_judgment":{"t1_state":reversed_ann["t1_state"],"t2_state":reversed_ann["t2_state"],"evidence_quality":"clear"}};_parse_judgment(target,delta.region_id)
            meta={**meta,"candidate_outcome":"improvement","target_view":target_view}
            row,item=save_example(args.output,"train",f"{cid}_time_reverse_candidate_improvement","candidate_evidence","initial",candidate,candidate_previous,payload,target,None,meta);rows["train"].append(row);records.append(item)

    if hardcase_round:
        cid=hardcase["scan_negative_candidate_id"];ann=annotations[cid];src=sources[cid]
        if src["split"] != "train" or src["kind"] != "unchanged":
            raise ValueError("v2 hard-negative source must be an unchanged train candidate")
        art={key:Path(value) for key,value in src["artifacts"].items()}
        t1=np.asarray(Image.open(art["t1"]).convert("RGB"));t2=np.asarray(Image.open(art["t2"]).convert("RGB"))
        empty=np.zeros((256,256),dtype=bool);scan=ChangeState(t1,t2,"building",empty,empty,empty)
        meta={"parent_id":src["parent_id"],"candidate_id":cid,"source_kind":src["kind"],
              "construction":"production_scale_scan_hard_negative_v3" if v3_round else "production_scale_scan_hard_negative_v2","source_sha256":src["artifact_sha256"],
              "t1_state":"background","t2_state":"background"}
        negative_annotation={"t1_state":"background","t2_state":"background","evidence":hardcase["scan_negative_evidence"]}
        candidate_harm_tiles=set(action_hardcase.get("grid_candidate_harm_tile_ids", hardcase["candidate_harm_tile_ids"])) if v3_round else set(hardcase["candidate_harm_tile_ids"])
        scan_tile_ids=sorted(set(action_hardcase.get("grid_scan_tile_ids", range(9)))) if v3_round else list(range(9))
        for index in scan_tile_ids:
            box,seed=scan_grid_geometry(index);row_index,column_index=divmod(index,3)
            pixel_starts=(0,85,170);pixel_ends=(85,170,256)
            x1=pixel_starts[column_index];y1=pixel_starts[row_index];x2=pixel_ends[column_index];y2=pixel_ends[row_index]
            area=(x2-x1)*(y2-y1)
            if v3_round:
                record,scan_proposal=scan_environment_record(scan,index)
            else:
                record=record_for(f"s{index}","scan",box,seed,scan,area)
            payload={"target_class":"building","region":record.to_dict(),"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
            target=audit_target(f"s{index}",negative_annotation,"correct");_parse_atomic_audit(target,record)
            row,item=save_example(args.output,"train",f"{cid}_scan_grid_correct_s{index}","audit","initial",scan,None,payload,target,None,meta)
            rows["train"].append(row);records.append(item)
            if index not in candidate_harm_tiles:
                continue
            blob=np.zeros((256,256),dtype=bool);blob[y1:y2,x1:x2]=True
            candidate=ChangeState(t1,t2,"building",empty,blob,blob)
            if v3_round:
                candidate=with_action_locality(candidate,scan_proposal,action="box")
                if include_background_fp_pairs:
                    # v3-v5 pair each large background audit with a safe removal.
                    # v6 deliberately drops these six rows while retaining the
                    # production scan and the three grid-harm box candidates.
                    background_candidate=ChangeState(t1,t2,"building",empty,blob,blob)
                    fp_record,fp_proposal=environment_record(background_candidate,expected_kind="present")
                    fp_payload={"target_class":"building","region":fp_record.to_dict(),"proposal_catalog":[fp_record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
                    fp_target=audit_target(fp_record.region_id,negative_annotation,"false_positive_background");_parse_atomic_audit(fp_target,fp_record)
                    fp_meta={**meta,"construction":"production_background_false_transition_v3","scan_tile_index":index}
                    row,item=save_example(args.output,"train",f"{cid}_audit_false_positive_background_s{index}","audit","initial",background_candidate,None,fp_payload,fp_target,None,fp_meta);rows["train"].append(row);records.append(item)
                    corrected=with_action_locality(scan,fp_proposal,action="negative_point")
                    removed,removed_proposal=environment_record(corrected,background_candidate,expected_kind="delta_removed")
                    removed_payload={"target_class":"building","region":removed.to_dict(),"proposal_catalog":[removed.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":"t2","action":"negative_point"},"persisted_initial_diagnosis":{"error_type":"false_positive_change"}}
                    removed_target={"region_id":removed.region_id,"visual_judgment":{"t1_state":"background","t2_state":"background","evidence_quality":"clear"}};_parse_judgment(removed_target,removed.region_id)
                    removed_meta={**fp_meta,"candidate_outcome":"improvement"}
                    row,item=save_example(args.output,"train",f"{cid}_candidate_improvement_background_delta_removed_s{index}","candidate_evidence","initial",corrected,background_candidate,removed_payload,removed_target,None,removed_meta);rows["train"].append(row);records.append(item)
                delta,delta_proposal=environment_record(candidate,scan,expected_kind="delta_added")
            else:
                delta=record_for(f"d{index}","delta_added",box,seed,candidate,int(blob.sum()))
            candidate_payload={"target_class":"building","region":delta.to_dict(),"proposal_catalog":[delta.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":"t2","action":"box"},"persisted_initial_diagnosis":{"error_type":"false_negative"}}
            candidate_target={"region_id":delta.region_id,"visual_judgment":{"t1_state":"background","t2_state":"background","evidence_quality":"clear"}};_parse_judgment(candidate_target,delta.region_id)
            candidate_meta={**meta,"candidate_outcome":"harm"} if v3_round else meta
            row,item=save_example(args.output,"train",f"{cid}_candidate_grid_harm_d{index}","candidate_evidence","initial",candidate,scan,candidate_payload,candidate_target,None,candidate_meta)
            rows["train"].append(row);records.append(item)

    if matched_round:
        core_records=[
            item for item in records
            if item["split"] == "train" and not item.get("v6_extension")
        ]
        if len(core_records) != int(v6_core_spec["expected_rows"]):
            raise ValueError(
                f"v6 frozen core row count changed: {len(core_records)}"
            )
        if any(
            item.get("construction") == "production_background_false_transition_v3"
            for item in core_records
        ):
            raise ValueError("v6 frozen core retained a background FP pair row")
        core_content_sha256=canonical_core_content_sha256(core_records)
        if core_content_sha256 != str(v6_core_spec["canonical_content_sha256"]):
            raise ValueError(
                "v6 frozen v3 core content drift: "
                f"expected {v6_core_spec['canonical_content_sha256']}, "
                f"got {core_content_sha256}"
            )
        for item in core_records:
            item["v6_core"] = True
        v6_core_receipt={
            "rows": len(core_records),
            "source_job_id": str(v6_core_spec["source_job_id"]),
            "handoff_content_sha256": str(
                v6_core_spec["handoff_content_sha256"]
            ),
            "canonical_content_sha256": core_content_sha256,
            "canonical_content_fields": list(V6_CORE_CONTENT_FIELDS),
        }

    if v4_round:
        # v4 is deliberately a six-row extension of v3.  Each pinned real SAM3
        # residual teaches the same semantic correction in three production
        # contexts: initial atomic audit, full-context repair after the exact
        # runtime all-pass rejection, and candidate evidence after removal.
        for spec in v4_residual_specs:
            cid=str(spec["candidate_id"]);src=sources[cid]
            if src["split"] != "train":
                raise ValueError(f"v4 residual source must be train-only: {cid}")
            if spec.get("t1_state") != "background" or spec.get("t2_state") != "background":
                raise ValueError(f"v4 residual must be manually curated background/background: {cid}")
            background_annotation={
                "t1_state":"background",
                "t2_state":"background",
                "evidence":str(spec["evidence"]),
            }
            art={key:Path(value) for key,value in src["artifacts"].items()}
            t1=np.asarray(Image.open(art["t1"]).convert("RGB"));t2=np.asarray(Image.open(art["t2"]).convert("RGB"))
            sam1=np.asarray(Image.open(art["sam3_t1_mask"]).convert("L"))>0;sam2=np.asarray(Image.open(art["sam3_t2_mask"]).convert("L"))>0
            gt=np.asarray(Image.open(art["gt_change"]).convert("L"))>0
            source_view=str(spec["source_view"]);rank=int(spec["component_rank"])
            component,residual_facts=residual_component(
                sam1,sam2,gt,
                source_view=source_view,
                rank=rank,
                expected_area=int(spec["expected_area"]),
                expected_gt_overlap=int(spec.get("expected_gt_overlap",0)),
            )
            expected_box=[int(value) for value in spec["expected_box_xyxy"]]
            if residual_facts["component_box_xyxy"] != expected_box:
                raise ValueError(
                    f"{cid} {source_view} component rank {rank} box changed: "
                    f"expected {expected_box}, got {residual_facts['component_box_xyxy']}"
                )
            case=unchanged_transition_case(t1,t2,component,source_view=source_view)
            false_transition=case["false_transition"];corrected=case["corrected"]
            record,initial_proposal=environment_record(false_transition,expected_kind="present")
            payload={"target_class":"building","region":record.to_dict(),"proposal_catalog":[record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
            residual_label_kind=(
                "false_positive_background"
                if v5_round or matched_round else "false_positive_transition"
            )
            target=audit_target(record.region_id,background_annotation,residual_label_kind)
            _,diagnosis=_parse_atomic_audit(target,record)
            plan=StagedQwenVerifier._plan(None,false_transition,(record,),diagnosis)
            if diagnosis.error_type != "false_positive_change" or diagnosis.target_view != source_view:
                raise ValueError(f"v4 parser did not derive a unique false-positive target: {cid}")
            if plan is None or plan.action != "negative_point" or plan.target_view != source_view:
                raise ValueError(f"v4 runtime did not derive the expected negative point: {cid}")
            if record.editable_seed_white != {"t1":source_view == "t1","t2":source_view == "t2"}:
                raise ValueError(f"v4 residual seed is not one-sided white: {cid}")
            base_meta={
                "parent_id":src["parent_id"],
                "candidate_id":cid,
                "source_kind":src["kind"],
                "construction":"production_residual_fp_contract_repair_v4",
                "source_sha256":src["artifact_sha256"],
                "t1_state":"background",
                "t2_state":"background",
                "rejected_initial_pattern":"background_background_all_pass",
                "rejected_validation_error":WHITE_NO_TRANSITION_VALIDATION_ERROR,
                "expected_action":"negative_point",
                "expected_delta":"delta_removed",
                "expected_component_area":int(spec["expected_area"]),
                "expected_gt_change_overlap_pixels":int(spec.get("expected_gt_overlap",0)),
                "v5_residual_label_correction":bool(v5_round),
                "residual_label_kind":residual_label_kind,
                **residual_facts,
            }
            if matched_round:
                base_meta.update({
                    "v6_residual_label_correction":True,
                    "v6_extension":True,
                })
            suffix=f"residual_{source_view}_r{rank}"
            initial_meta={**base_meta,"v4_example_kind":"audit_initial"}
            row,item=save_example(args.output,"train",f"{cid}_audit_false_positive_{suffix}","audit","initial",false_transition,None,payload,target,None,initial_meta);rows["train"].append(row);records.append(item)

            # This is the exact repair shape produced when the model saw the
            # same component, returned background/background plus all-pass, and
            # the frozen parser rejected it.  The gold answer flips only the
            # precision/scope statuses required by that semantic contradiction.
            repair_meta={**base_meta,"v4_example_kind":"audit_repair_replay","origin_validation_error":WHITE_NO_TRANSITION_VALIDATION_ERROR}
            row,item=save_example(args.output,"train",f"{cid}_repair_false_positive_{suffix}","audit","repair",false_transition,None,payload,target,WHITE_NO_TRANSITION_VALIDATION_ERROR,repair_meta);rows["train"].append(row);records.append(item)

            corrected=with_action_locality(corrected,initial_proposal,action="negative_point")
            removed,removed_proposal=environment_record(corrected,false_transition,expected_kind="delta_removed")
            removed_payload={"target_class":"building","region":removed.to_dict(),"proposal_catalog":[removed.to_dict()],"schema":"candidate_transition_evidence_v1","visual_context":"proposal","attempted_action":{"target_view":source_view,"action":"negative_point"},"persisted_initial_diagnosis":{"error_type":"false_positive_change"}}
            removed_target={"region_id":removed.region_id,"visual_judgment":{"t1_state":"background","t2_state":"background","evidence_quality":"clear"}}
            removed_judgment=_parse_judgment(removed_target,removed.region_id)
            verifier=object.__new__(StagedQwenVerifier)
            verifier._planned_diagnosis=diagnosis
            verifier._planned_action=_plan_agent_action(plan,false_transition.image_size)
            assessment=verifier._assess_candidate_transition(
                (removed,),(removed_judgment,),verifier._planned_action
            )
            if assessment.comparison != "better" or not assessment.intended_error_improved or assessment.introduced_harm:
                raise ValueError(f"v4 delta removal is not a runtime-derived improvement: {cid}")
            removed_meta={**base_meta,"v4_example_kind":"candidate_improvement","candidate_outcome":"improvement","expected_candidate_comparison":"better"}
            row,item=save_example(args.output,"train",f"{cid}_candidate_improvement_delta_removed_{suffix}","candidate_evidence","initial",corrected,false_transition,removed_payload,removed_target,None,removed_meta);rows["train"].append(row);records.append(item)

    if v7_round:
        v7_sampling_receipt=append_v7_sampling_repeats(
            rows=rows,
            records=records,
            sampling_spec=sampling_hardcase,
        )

    for split in rows:
        (args.output/f"{split}.json").write_text(json.dumps(rows[split],indent=2,ensure_ascii=False)+"\n")
    info={DATASET_NAMES[0]:{"file_name":"train.json","formatting":"sharegpt","columns":{"messages":"conversations","images":"images"},"tags":{"role_tag":"from","content_tag":"value","user_tag":"human","assistant_tag":"gpt"}},
          DATASET_NAMES[1]:{"file_name":"val.json","formatting":"sharegpt","columns":{"messages":"conversations","images":"images"},"tags":{"role_tag":"from","content_tag":"value","user_tag":"human","assistant_tag":"gpt"}}}
    (args.output/"dataset_info.json").write_text(json.dumps(info,indent=2)+"\n")
    parents={s:{r["parent_id"] for r in records if r["split"]==s} for s in rows}
    if parents["train"]&parents["val"]: raise ValueError("parent leakage")
    if v3_round:
        if v7_round:
            expected_rows=(83,24)
            expected_stages=Counter({"audit":54,"candidate_evidence":29})
            expected_scopes=Counter({"standard":66,"authoritative_pair_rgb_only_scan":9,"authoritative_pair_rgb_only_evidence_repair":8})
        elif v6_round:
            expected_rows=(63,24)
            expected_stages=Counter({"audit":42,"candidate_evidence":21})
            expected_scopes=Counter({"standard":46,"authoritative_pair_rgb_only_scan":9,"authoritative_pair_rgb_only_evidence_repair":8})
        elif v5_round:
            expected_rows=(69,24)
            expected_stages=Counter({"audit":47,"candidate_evidence":22})
            expected_scopes=Counter({"standard":52,"authoritative_pair_rgb_only_scan":9,"authoritative_pair_rgb_only_evidence_repair":8})
        elif v4_round:
            expected_rows=(65,24)
            expected_stages=Counter({"audit":43,"candidate_evidence":22})
            expected_scopes=Counter({"standard":48,"authoritative_pair_rgb_only_scan":9,"authoritative_pair_rgb_only_evidence_repair":8})
        else:
            expected_rows=(59,24)
            expected_stages=Counter({"audit":39,"candidate_evidence":20})
            expected_scopes=Counter({"standard":42,"authoritative_pair_rgb_only_scan":9,"authoritative_pair_rgb_only_evidence_repair":8})
        if (len(rows["train"]), len(rows["val"])) != expected_rows:
            raise ValueError(
                f"{args.round} row contract mismatch: train={len(rows['train'])}, val={len(rows['val'])}"
            )
        train_records=[r for r in records if r["split"] == "train"]
        stage_counts=Counter(r["stage"] for r in train_records)
        scope_counts=Counter(r["visual_input_scope"] for r in train_records)
        if stage_counts != expected_stages:
            raise ValueError(f"{args.round} stage distribution mismatch: {stage_counts}")
        if scope_counts != expected_scopes:
            raise ValueError(f"{args.round} scope distribution mismatch: {scope_counts}")
        for item in train_records:
            if item.get("construction") == "production_one_sided_transition_v3":
                if int(item.get("region_component_area") or 0) in {0, 2304}:
                    raise ValueError("v3 transition geometry is not production-sized")
        if v4_round:
            residual_records=[item for item in train_records if str(item.get("construction","")).endswith("_v4")]
            if len(residual_records) != 6:
                raise ValueError(f"v4 residual row contract mismatch: {len(residual_records)}")
            if Counter(item["attempt_kind"] for item in residual_records) != Counter({"initial":4,"repair":2}):
                raise ValueError("v4 residual initial/repair distribution mismatch")
            if Counter(item.get("v4_example_kind") for item in residual_records) != Counter({"audit_initial":2,"audit_repair_replay":2,"candidate_improvement":2}):
                raise ValueError("v4 residual example-kind distribution mismatch")
            if Counter((item["stage"],item["attempt_kind"],item["visual_input_scope"]) for item in residual_records) != Counter({("audit","initial","standard"):2,("audit","repair","standard"):2,("candidate_evidence","initial","standard"):2}):
                raise ValueError("v4 residual stage/attempt/scope contract mismatch")
            if len({item.get("component_sha256") for item in residual_records}) != 2:
                raise ValueError("v4 residual component identity contract mismatch")
            if any(int(item.get("gt_change_overlap_pixels",-1)) != 0 for item in residual_records):
                raise ValueError("v4 residual overlaps offline GT change")
            if (v5_round or matched_round) and any(item.get("residual_label_kind") != "false_positive_background" for item in residual_records):
                raise ValueError(f"{args.round} residual checklist contract was not canonicalized")
        if v5_round:
            contrast_records=[item for item in train_records if item.get("construction") == "production_contrast_repair_v5"]
            if len(contrast_records) != 4:
                raise ValueError(f"v5 contrast row contract mismatch: {len(contrast_records)}")
            if Counter(item.get("v5_example_kind") for item in contrast_records) != Counter({"semantic_anchor_repair":2,"actionable_transition_repair":2}):
                raise ValueError("v5 contrast example-kind distribution mismatch")
            if any((item["stage"],item["attempt_kind"],item["visual_input_scope"],item.get("region_audit_kind")) != ("audit","repair","standard","present") for item in contrast_records):
                raise ValueError("v5 contrast stage/scope contract mismatch")
            if any(item.get("origin_validation_error") != WHITE_NO_TRANSITION_VALIDATION_ERROR for item in contrast_records):
                raise ValueError("v5 contrast validation-error contract mismatch")
            repair_states=Counter(
                (item.get("t1_state"),item.get("t2_state"))
                for item in train_records
                if item.get("attempt_kind") == "repair"
                and (
                    item.get("origin_validation_error") == WHITE_NO_TRANSITION_VALIDATION_ERROR
                    or item.get("rejected_validation_error") == WHITE_NO_TRANSITION_VALIDATION_ERROR
                )
            )
            if repair_states != Counter({("background","background"):2,("background","building"):2,("building","building"):2}):
                raise ValueError(f"v5 contrast semantic repair distribution mismatch: {repair_states}")
            record_ids={item["id"] for item in train_records}
            for spec in (*v5_semantic_specs,*v5_actionable_specs):
                paired_id=str(spec["paired_candidate_id"])
                if paired_id not in record_ids:
                    raise ValueError(f"v5 paired candidate is missing: {paired_id}")
                paired=next(item for item in train_records if item["id"] == paired_id)
                expected_outcome="harm" if spec in v5_semantic_specs else "improvement"
                if paired.get("candidate_outcome") != expected_outcome or paired.get("region_audit_kind") != "delta_removed":
                    raise ValueError(f"v5 paired candidate contract mismatch: {paired_id}")
        if matched_round:
            if v6_core_receipt is None:
                raise ValueError("v6 frozen core receipt is missing")
            if any(
                item.get("construction") == "production_background_false_transition_v3"
                for item in train_records
            ):
                raise ValueError("v6 unexpectedly contains a background FP pair")
            core_records=[item for item in train_records if item.get("v6_core") is True]
            if len(core_records) != 53:
                raise ValueError(f"v6 manifest core marker count mismatch: {len(core_records)}")
            if canonical_core_content_sha256(core_records) != V6_FROZEN_CORE_CANONICAL_SHA256:
                raise ValueError("v6 marked core content fingerprint changed")
            grid_harm_records=[
                item for item in train_records
                if "_candidate_grid_harm_d" in item["id"]
            ]
            if len(grid_harm_records) != 3 or any(
                item.get("candidate_outcome") != "harm"
                for item in grid_harm_records
            ):
                raise ValueError("v6 did not retain all three v3 grid-harm rows")
            residual_records=[
                item for item in train_records
                if item.get("construction") == "production_residual_fp_contract_repair_v4"
            ]
            if len(residual_records) != 6 or any(
                item.get("v6_extension") is not True
                or item.get("residual_label_kind") != "false_positive_background"
                for item in residual_records
            ):
                raise ValueError("v6 residual triplet contract mismatch")
            matched_records=[
                item for item in train_records
                if item.get("construction") == "production_matched_true_transition_v6"
            ]
            if len(matched_records) != 4:
                raise ValueError(f"v6 matched contrast row mismatch: {len(matched_records)}")
            if Counter(item.get("v6_example_kind") for item in matched_records) != Counter({
                "true_transition_repair":2,
                "true_transition_harmful_removal":2,
            }):
                raise ValueError("v6 matched contrast kind distribution mismatch")
            repair_records=[
                item for item in matched_records
                if item.get("v6_example_kind") == "true_transition_repair"
            ]
            if any(
                (
                    item["stage"],
                    item["attempt_kind"],
                    item["visual_input_scope"],
                    item.get("region_audit_kind"),
                    item.get("expected_repair_action"),
                ) != ("audit","repair","standard","present","finish")
                or item.get("origin_validation_error") != WHITE_NO_TRANSITION_VALIDATION_ERROR
                for item in repair_records
            ):
                raise ValueError("v6 true-transition repair contract mismatch")
            harm_records=[
                item for item in matched_records
                if item.get("v6_example_kind") == "true_transition_harmful_removal"
            ]
            if any(
                item["stage"] != "candidate_evidence"
                or item.get("region_audit_kind") != "delta_removed"
                or item.get("candidate_outcome") != "harm"
                or item.get("expected_candidate_comparison") != "worse"
                or item.get("attempted_action", {}).get("action") != "negative_point"
                or item.get("attempted_action", {}).get("target_view") != item.get("target_view")
                for item in harm_records
            ):
                raise ValueError("v6 true-transition harmful-removal contract mismatch")
            if Counter((item.get("t1_state"),item.get("t2_state")) for item in repair_records) != Counter({
                ("background","building"):1,
                ("building","background"):1,
            }):
                raise ValueError("v6 true-transition direction balance changed")
        if v7_round:
            if v7_sampling_receipt is None:
                raise ValueError("v7 sampling receipt is missing")
            repeat_records=[
                item for item in train_records
                if item.get("construction") == "sampling_weight_repeat_v7"
            ]
            unique_records=[
                item for item in train_records
                if item.get("construction") != "sampling_weight_repeat_v7"
            ]
            weighted_sources=[
                item for item in unique_records
                if item.get("v7_weighted_source") is True
            ]
            if (len(unique_records),len(repeat_records)) != (63,20):
                raise ValueError("v7 unique/repeat row contract mismatch")
            if len({sampling_content_key(item) for item in train_records}) != 63:
                raise ValueError("v7 unique content count is not 63")
            if len(weighted_sources) != 10:
                raise ValueError("v7 weighted source count is not ten")
            if [item["id"] for item in weighted_sources] != v7_sampling_receipt["weighted_source_ids"]:
                raise ValueError("v7 receipt weighted source order changed")
            if canonical_core_content_sha256(unique_records) != v7_sampling_receipt["unique_train_content_sha256"]:
                raise ValueError("v7 unique-train content receipt changed")
            if canonical_core_content_sha256(weighted_sources) != v7_sampling_receipt["weighted_source_content_sha256"]:
                raise ValueError("v7 weighted-source content receipt changed")
            manifest_by_id={item["id"]:item for item in train_records}
            train_row_by_id={row["id"]:row for row in rows["train"]}
            for repeat in repeat_records:
                source_id=str(repeat.get("repeated_from_id"))
                if source_id not in manifest_by_id or source_id not in train_row_by_id:
                    raise ValueError(f"v7 repeat source is missing: {source_id}")
                source=manifest_by_id[source_id]
                source_row=train_row_by_id[source_id]
                repeat_row=train_row_by_id[repeat["id"]]
                if (
                    repeat.get("sampling_repeat_index") not in {1,2}
                    or repeat.get("sampling_weight_factor") != 3
                    or repeat.get("sampling_repeat_only") is not True
                    or repeat.get("unique_content") is not False
                    or repeat.get("v6_extension") is not None
                    or sampling_content_key(repeat) != sampling_content_key(source)
                    or repeat_row["conversations"] != source_row["conversations"]
                    or repeat_row["images"] != source_row["images"]
                ):
                    raise ValueError(f"v7 sampling repeat drifted from {source_id}")
            if Counter(
                (item.get("t1_state"),item.get("t2_state"))
                for item in train_records
            ) != Counter({
                ("background","building"):36,
                ("background","background"):25,
                ("building","background"):14,
                ("building","building"):8,
            }):
                raise ValueError("v7 semantic exposure distribution mismatch")
            if Counter(
                item.get("candidate_outcome")
                for item in train_records
                if item["stage"] == "candidate_evidence"
            ) != Counter({"improvement":17,"harm":12}):
                raise ValueError("v7 candidate exposure distribution mismatch")
    semantic_distribution=Counter(
        f'{r["split"]}|{r.get("t1_state")}->{r.get("t2_state")}' for r in records
    )
    audit_distribution=Counter(
        f'{r["split"]}|{r.get("region_audit_kind") or "none"}' for r in records
    )
    candidate_distribution=Counter(
        f'{r["split"]}|{r.get("candidate_outcome") or "unspecified"}'
        for r in records if r["stage"] == "candidate_evidence"
    )
    parent_distribution=Counter(
        f'{r["split"]}|{r["parent_id"]}' for r in records
    )
    report={"valid":True,"data_round":args.round,"train_rows":len(rows["train"]),"val_rows":len(rows["val"]),"train_parents":len(parents["train"]),"val_parents":len(parents["val"]),
            "distribution":dict(Counter(f'{r["split"]}|{r["stage"]}|{r["visual_input_scope"]}' for r in records)),
            "semantic_state_distribution":dict(semantic_distribution),"audit_kind_distribution":dict(audit_distribution),
            "candidate_outcome_distribution":dict(candidate_distribution),"parent_row_distribution":dict(parent_distribution),"test_data_used":False}
    full={"schema":"qwen3vl_v121_fewshot_manifest_v1","data_round":args.round,"runtime_commit":args.runtime_commit,"seed":args.seed,"curation_manifest_sha256":sha256_file(args.curation_root/"curation_manifest.json"),
          "curation_config_sha256":sha256_file(args.curation_config),"test_data_used":False,"raw_gt_model_visible":False,"samples":records}
    if v6_core_receipt is not None:
        report["v6_frozen_core"] = dict(v6_core_receipt)
        full["v6_frozen_core"] = dict(v6_core_receipt)
    if v7_sampling_receipt is not None:
        report["v7_sampling_receipt"] = dict(v7_sampling_receipt)
        full["v7_sampling_receipt"] = dict(v7_sampling_receipt)
    (args.output/"manifest.json").write_text(json.dumps(full,indent=2,ensure_ascii=False)+"\n");(args.output/"validation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (args.output/".data_complete").write_text("complete\n");print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__": main()
