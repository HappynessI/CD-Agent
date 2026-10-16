#!/usr/bin/env python3
"""Build deterministic no-Qwen oracle trajectories with the real runtime gates.

The online half of this runner receives only RGB crops and accepted model masks.
It enumerates actions from the Environment-owned live proposal catalog, executes
every shortlisted candidate in a fresh branch Environment, and records the
unchanged runtime hard gates.  After the GT-free live catalog is frozen, the
offline half opens the label to form the frozen top-4 shortlist.  It executes the
whole shortlist before scoring completed masks and choosing one action.  The
chosen action is replayed immediately in the single canonical Environment and all
three semantic mask hashes must match the shortlisted branch before another step.

Production uses fresh SAM3 initialization plus the real SimpleClick/SAM3 workers.
``EpisodeRuntimeFactory`` is intentionally injectable so the policy and isolation
contract can be tested without loading model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.adapters.subprocess_adapters import (
    SubprocessBoxBackend,
    SubprocessPointBackend,
    SubprocessSAM3Initializer,
)
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction, ChangeState, VerifierOutput
from change_agent.trajectory import mask_sha256


PILOT_SCHEMA_ID = "qwen3vl_oracle_trajectory_pilot_preregistration_v1"
EPISODE_SCHEMA_ID = "qwen3vl_oracle_trajectory_episode_v1"
INDEX_SCHEMA_ID = "qwen3vl_oracle_trajectory_shard_index_v1"
COMPLETE_SCHEMA_ID = "qwen3vl_oracle_trajectory_generation_complete_v1"
PATCH_SIZE = 256
MAX_CANONICAL_STEPS = 3
MAX_CANDIDATES_PER_STEP = 4
MAX_DISTINCT_CANDIDATES = MAX_CANONICAL_STEPS * MAX_CANDIDATES_PER_STEP
MAX_TOOL_EXECUTIONS = 24
DEFAULT_LOCAL_ERROR_REDUCTION_RATIO = 0.05
DEFAULT_ENDPOINT_IOU = 0.90
DEFAULT_ENDPOINT_F1 = 18.0 / 19.0
DEFAULT_EMPTY_GT_MAX_FP = 64
RGB_STATES = frozenset({"building", "background", "mixed", "uncertain"})
FULL_LIVE_AUDIT_KINDS = frozenset({"present", "missing", "mixed", "scan"})
EVIDENCE_ONLY_AUDIT_KINDS = frozenset({"delta_added", "delta_removed"})
KNOWN_AUDIT_KINDS = FULL_LIVE_AUDIT_KINDS | EVIDENCE_ONLY_AUDIT_KINDS
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class LabelSource:
    identity: FileIdentity
    crop_xyxy_exclusive: tuple[int, int, int, int]


@dataclass(frozen=True)
class RuntimeEpisode:
    """GT-free episode input.  This is the only sample object runtime factories see."""

    episode_id: str
    sample_id: str
    split: str
    parent_id: str
    t1_image: np.ndarray
    t2_image: np.ndarray
    t1_source: FileIdentity
    t2_source: FileIdentity
    crop_xyxy_exclusive: tuple[int, int, int, int]
    query: str
    rgb_states: Mapping[str, str]
    rgb_semantic_evidence: str
    rgb_semantic_source: str
    rgb_semantic_status: str
    rgb_states_by_region: Mapping[str, Mapping[str, str]]
    rgb_states_by_step: Mapping[str, Mapping[str, Mapping[str, str]]]


@dataclass(frozen=True)
class PilotEpisode:
    runtime: RuntimeEpisode
    label: LabelSource


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    region_id: str
    reachability_tier: str
    action: AgentAction
    proposal: Mapping[str, Any]


class EpisodeRuntimeFactory(Protocol):
    """Factory boundary that cannot receive a label or an oracle metric."""

    @property
    def qwen_calls(self) -> int: ...

    def create_canonical_environment(
        self, episode: RuntimeEpisode, artifact_dir: Path, max_steps: int
    ) -> ChangeAgentEnvironment: ...

    def create_branch_environment(
        self,
        episode: RuntimeEpisode,
        accepted_state: ChangeState,
        artifact_dir: Path,
    ) -> ChangeAgentEnvironment: ...


class NoQwenAcceptVerifier:
    """Accept every gate-passing candidate without importing or invoking Qwen."""

    def __init__(self) -> None:
        self.initial_call_count = 0
        self.candidate_call_count = 0
        self.qwen_calls = 0
        self.last_evidence: dict[str, Any] = {}

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        del state, previous_score, previous_state
        initial = previous_action is None
        if initial:
            self.initial_call_count += 1
        else:
            self.candidate_call_count += 1
        self.last_evidence = {
            "type": "oracle_no_qwen_accept_verifier",
            "qwen_invoked": False,
            "phase": "initial" if initial else "candidate",
        }
        return VerifierOutput(
            quality_score=0.0,
            progress_score=0.0 if initial else 1.0,
            comparison="initial" if initial else "better",
            error_type="none",
            target_view=(previous_action.target_view if previous_action else "t2"),
            suggested_action=None,
            feedback=(
                "No-Qwen oracle initial-state sentinel."
                if initial
                else "No-Qwen oracle accepts candidates that passed runtime hard gates."
            ),
            accept=not initial,
            verifier_valid=True,
            localization_valid=True,
            stop=False,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def read_strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_float=_finite_float,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot be serialized")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _atomic_write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    encoded = (json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_mask(path: Path, mask: np.ndarray, root: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite mask artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    value = np.ascontiguousarray(mask, dtype=np.uint8)
    try:
        with temporary.open("xb") as stream:
            np.save(stream, value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "mask_sha256": mask_sha256(value),
        "shape": list(value.shape),
    }


def _file_identity(value: Any, field: str) -> FileIdentity:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a file identity object")
    raw_path, expected = value.get("path"), value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field}.path must be a non-empty string")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected)
    ):
        raise ValueError(f"{field}.sha256 must be a SHA256 hex digest")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    # Do not resolve/stat here: a label identity must remain opaque until the
    # explicit offline label loader runs after the GT-free catalog freeze.
    return FileIdentity(path.absolute(), expected.lower())


def _verified_file(identity: FileIdentity, field: str) -> Path:
    if not identity.path.is_file():
        raise FileNotFoundError(f"{field} is not a file: {identity.path}")
    if sha256_file(identity.path) != identity.sha256:
        raise ValueError(f"{field} SHA256 mismatch")
    return identity.path


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _validate_frozen_runtime_contract(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Fail closed if a formal run drifts from the preregistered runtime contract."""

    protocol = _mapping(manifest.get("execution_protocol"), "execution_protocol")
    qwen = _mapping(protocol.get("qwen_policy"), "execution_protocol.qwen_policy")
    if not (
        protocol.get("seed") == 42
        and protocol.get("query") == "building"
        and protocol.get("sam3_resolution") == 1008
        and protocol.get("tool_device") == "cuda"
        and protocol.get("tool_timeout_seconds") == 900
        and protocol.get("max_steps") == MAX_CANONICAL_STEPS
        and protocol.get("accepted_actions_per_step") == 1
        and qwen.get("enabled") is False
        and qwen.get("model_loaded") is False
        and qwen.get("allowed_call_count") == 0
    ):
        raise ValueError("frozen execution/Qwen contract differs")
    budget = _mapping(
        manifest.get("candidate_and_replay_budget"),
        "candidate_and_replay_budget",
    )
    if not (
        budget.get("max_candidate_actions_per_step") == MAX_CANDIDATES_PER_STEP
        and budget.get("max_steps_per_sample") == MAX_CANONICAL_STEPS
        and budget.get("max_distinct_candidate_actions_per_sample")
        == MAX_DISTINCT_CANDIDATES
        and budget.get(
            "max_total_tool_executions_including_selected_action_replay_per_sample"
        )
        == MAX_TOOL_EXECUTIONS
    ):
        raise ValueError("pilot candidate/replay budget must equal 4 x 3, 12, and 24")
    proposal_config = _mapping(manifest.get("proposal_config"), "proposal_config")
    if not (
        proposal_config.get("max_regions") == 1
        and proposal_config.get("max_delta_regions") == 1
        and proposal_config.get("min_component_area") == 4
        and proposal_config.get("padding_ratio") == 0.25
        and proposal_config.get("empty_mask_scan_grid_size") == 3
        and manifest.get("authorized_negative_point_input_scope") == "full_session"
    ):
        raise ValueError("frozen proposal/negative-point contract differs")
    matching = _mapping(manifest.get("matching"), "matching")
    if not (
        matching.get("mode") == "overlap_presence"
        and matching.get("overlap_threshold") == 0.25
        and matching.get("t12_min_instance_area") == 0
        and matching.get("cd_min_instance_area") == 0
    ):
        raise ValueError("frozen matching contract differs")
    metric = _mapping(
        manifest.get("metric_and_success_contract"),
        "metric_and_success_contract",
    )
    local = _mapping(
        metric.get("locally_reachable"),
        "metric_and_success_contract.locally_reachable",
    )
    endpoint = _mapping(
        metric.get("endpoint_success"),
        "metric_and_success_contract.endpoint_success",
    )
    global_gate = _mapping(
        metric.get("global_pilot_release_gate"),
        "metric_and_success_contract.global_pilot_release_gate",
    )
    if not (
        metric.get("comparison_epsilon") == 1e-12
        and local.get("minimum_final_error_reduction_ratio")
        == DEFAULT_LOCAL_ERROR_REDUCTION_RATIO
        and endpoint.get("nonempty_gt_minimum_final_iou")
        == DEFAULT_ENDPOINT_IOU
        and endpoint.get("nonempty_gt_minimum_final_f1")
        == DEFAULT_ENDPOINT_F1
        and endpoint.get("empty_gt_maximum_final_fp")
        == DEFAULT_EMPTY_GT_MAX_FP
        and global_gate.get("required_endpoint_successes") == 6
        and global_gate.get("required_evaluated_episodes") == 6
        and global_gate.get("maximum_inconclusive_or_not_evaluated") == 0
        and global_gate.get("minimum_nontrivial_accepted_trajectories") == 6
    ):
        raise ValueError("frozen metric/endpoint release contract differs")
    hard_gate = _mapping(manifest.get("hard_gate_config"), "hard_gate_config")
    expected_hard_gate = {
        "max_selection_area_delta": 0.25,
        "max_locality_outside_ratio": 0.1,
        "max_target_mask_change_ratio": 0.25,
        "max_component_count_delta": 4,
        "max_point_proposal_change_ratio": 0.25,
        "max_point_proposal_change_pixels": 256,
        "max_point_image_change_ratio": 0.004,
        "point_proposal_change_gate_enabled": True,
        "modification_or_retuning_authorized": False,
    }
    if not all(hard_gate.get(key) == value for key, value in expected_hard_gate.items()):
        raise ValueError("frozen production hard-gate contract differs")
    tools = _mapping(manifest.get("tools_and_models"), "tools_and_models")
    qwen_model = _mapping(
        tools.get("qwen_base_model_disabled"),
        "tools_and_models.qwen_base_model_disabled",
    )
    if qwen_model.get("load_for_pilot") is not False:
        raise ValueError("frozen Qwen model-load policy differs")
    return protocol, budget


def _source(spec: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in spec:
            return spec[name]
    raise ValueError(f"missing required source field; expected one of {names}")


def _crop_box(spec: Mapping[str, Any]) -> tuple[int, int, int, int]:
    crop_spec = spec.get("crop", spec)
    crop_spec = _mapping(crop_spec, "sample.crop")
    raw = crop_spec.get(
        "xyxy_exclusive",
        crop_spec.get("crop_xyxy_exclusive", crop_spec.get("crop_xyxy")),
    )
    if raw is None and "row" in crop_spec and "column" in crop_spec:
        row, column = int(crop_spec["row"]), int(crop_spec["column"])
        raw = [column * PATCH_SIZE, row * PATCH_SIZE, (column + 1) * PATCH_SIZE, (row + 1) * PATCH_SIZE]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("sample crop must be crop_xyxy_exclusive or row/column")
    x1, y1, x2, y2 = (int(value) for value in raw)
    if x1 < 0 or y1 < 0 or x2 - x1 != PATCH_SIZE or y2 - y1 != PATCH_SIZE:
        raise ValueError("every pilot crop must be an in-frame 256x256 exclusive box")
    return x1, y1, x2, y2


def _validate_rgb_state(value: Any, field: str) -> str:
    if not isinstance(value, str) or value not in RGB_STATES:
        raise ValueError(f"{field} must be one of {sorted(RGB_STATES)}")
    return value


def _rgb_pair(value: Any, field: str) -> dict[str, str]:
    mapping = _mapping(value, field)
    return {
        view: _validate_rgb_state(
            mapping.get(view, mapping.get(f"{view}_state", "uncertain")),
            f"{field}.{view}",
        )
        for view in ("t1", "t2")
    }


def _parse_rgb_semantics(
    spec: Mapping[str, Any], field: str
) -> tuple[
    Mapping[str, str],
    Mapping[str, Mapping[str, str]],
    Mapping[str, Mapping[str, Mapping[str, str]]],
]:
    raw = spec.get(
        "semantic",
        spec.get(
            "rgb_semantics",
            {"t1_state": spec.get("t1_state", "uncertain"), "t2_state": spec.get("t2_state", "uncertain")},
        ),
    )
    mapping = _mapping(raw, f"{field}.rgb_semantics")
    defaults = _rgb_pair(mapping, f"{field}.rgb_semantics")
    by_region_raw = mapping.get("by_region", spec.get("rgb_states_by_region", {}))
    by_region = {
        str(region_id): _rgb_pair(value, f"{field}.rgb_semantics.by_region.{region_id}")
        for region_id, value in _mapping(by_region_raw, f"{field}.rgb_semantics.by_region").items()
    }
    by_step_raw = mapping.get("by_step", spec.get("rgb_states_by_step", {}))
    by_step: dict[str, dict[str, Mapping[str, str]]] = {}
    for step, regions in _mapping(by_step_raw, f"{field}.rgb_semantics.by_step").items():
        by_step[str(step)] = {
            str(region_id): _rgb_pair(
                value, f"{field}.rgb_semantics.by_step.{step}.{region_id}"
            )
            for region_id, value in _mapping(
                regions, f"{field}.rgb_semantics.by_step.{step}"
            ).items()
        }
    return defaults, by_region, by_step


def parse_pilot_episode(spec: Any, index: int, query: str) -> tuple[dict[str, Any], LabelSource]:
    item = _mapping(spec, f"samples[{index}]")
    episode_id = str(item.get("episode_id", item.get("sample_id", "")))
    sample_id = str(item.get("sample_id", episode_id))
    if not SAFE_ID.fullmatch(episode_id) or episode_id in {".", ".."}:
        raise ValueError(f"samples[{index}].episode_id is not a safe artifact ID")
    if not sample_id:
        raise ValueError(f"samples[{index}].sample_id is required")
    source_parent = _mapping(item.get("source_parent", item), f"samples[{index}].source_parent")
    t1_identity = _file_identity(
        _source(source_parent, "parent_a", "parent_A", "t1_parent", "t1"),
        f"samples[{index}].parent_a",
    )
    t2_identity = _file_identity(
        _source(source_parent, "parent_b", "parent_B", "t2_parent", "t2"),
        f"samples[{index}].parent_b",
    )
    label_identity = _file_identity(
        _source(source_parent, "parent_label", "label_parent", "label"),
        f"samples[{index}].parent_label",
    )
    crop = _crop_box(item)
    rgb_states, by_region, by_step = _parse_rgb_semantics(item, f"samples[{index}]")
    parsed = {
        "episode_id": episode_id,
        "sample_id": sample_id,
        "split": str(item.get("split", "unknown")),
        "parent_id": str(item.get("parent_id", Path(t1_identity.path).stem)),
        "t1_source": t1_identity,
        "t2_source": t2_identity,
        "crop": crop,
        "query": str(item.get("query", query)),
        "rgb_states": rgb_states,
        "rgb_semantic_evidence": str(
            _mapping(item.get("semantic", {}), f"samples[{index}].semantic").get(
                "rgb_evidence", ""
            )
        ),
        "rgb_semantic_source": str(
            _mapping(item.get("semantic", {}), f"samples[{index}].semantic").get(
                "source", ""
            )
        ),
        "rgb_semantic_status": str(
            _mapping(item.get("semantic", {}), f"samples[{index}].semantic").get(
                "status", ""
            )
        ),
        "rgb_states_by_region": by_region,
        "rgb_states_by_step": by_step,
    }
    if not parsed["query"].strip():
        raise ValueError(f"samples[{index}].query must not be empty")
    return parsed, LabelSource(label_identity, crop)


def load_runtime_episode(parsed: Mapping[str, Any]) -> RuntimeEpisode:
    """Open only RGB parents; label identity is deliberately absent from this API."""

    t1_identity = parsed["t1_source"]
    t2_identity = parsed["t2_source"]
    if not isinstance(t1_identity, FileIdentity) or not isinstance(t2_identity, FileIdentity):
        raise TypeError("parsed RGB identities are invalid")
    t1_parent = Image.open(_verified_file(t1_identity, "parent_a")).convert("RGB")
    t2_parent = Image.open(_verified_file(t2_identity, "parent_b")).convert("RGB")
    crop = tuple(parsed["crop"])
    x1, y1, x2, y2 = crop
    if x2 > t1_parent.width or y2 > t1_parent.height:
        raise ValueError("T1 crop exceeds its parent image")
    if x2 > t2_parent.width or y2 > t2_parent.height:
        raise ValueError("T2 crop exceeds its parent image")
    t1 = np.asarray(t1_parent.crop(crop), dtype=np.uint8)
    t2 = np.asarray(t2_parent.crop(crop), dtype=np.uint8)
    if t1.shape != (PATCH_SIZE, PATCH_SIZE, 3) or t2.shape != t1.shape:
        raise ValueError("deterministic parent crops must both be 256x256 RGB")
    return RuntimeEpisode(
        episode_id=str(parsed["episode_id"]),
        sample_id=str(parsed["sample_id"]),
        split=str(parsed["split"]),
        parent_id=str(parsed["parent_id"]),
        t1_image=t1,
        t2_image=t2,
        t1_source=t1_identity,
        t2_source=t2_identity,
        crop_xyxy_exclusive=crop,
        query=str(parsed["query"]),
        rgb_states=parsed["rgb_states"],
        rgb_semantic_evidence=str(parsed["rgb_semantic_evidence"]),
        rgb_semantic_source=str(parsed["rgb_semantic_source"]),
        rgb_semantic_status=str(parsed["rgb_semantic_status"]),
        rgb_states_by_region=parsed["rgb_states_by_region"],
        rgb_states_by_step=parsed["rgb_states_by_step"],
    )


def load_label_mask(source: LabelSource) -> np.ndarray:
    """Offline-only label loader, first called after the live catalog is frozen."""

    path = _verified_file(source.identity, "parent_label")
    parent = Image.open(path).convert("L")
    x1, y1, x2, y2 = source.crop_xyxy_exclusive
    if x2 > parent.width or y2 > parent.height:
        raise ValueError("label crop exceeds its parent image")
    value = np.asarray(parent.crop(source.crop_xyxy_exclusive), dtype=np.uint8) > 0
    if value.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("deterministic label crop must be 256x256")
    return value


def _proposal_flags(proposal: Mapping[str, Any]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for view in ("t1", "t2"):
        value = proposal.get(f"component_seed_{view}_mask_white")
        if type(value) is not bool:
            raise ValueError(f"proposal {proposal.get('region_id')} lacks a boolean {view} seed fact")
        result[view] = value
    return result


def enumerate_live_actions(
    episode_id: str,
    step_index: int,
    proposals: Sequence[Mapping[str, Any]],
) -> list[CandidateAction]:
    """Enumerate the frozen useful executor surface from live proposal geometry.

    Positive clicks are valid on either a black seed (add a missing object) or
    an already-white seed (refine/expand that object).  For ordinary full-live
    proposals, negative clicks require a white live seed because the production
    Executor rejects a black clicked component.  Scan proposals deliberately
    expose only their positive click and box actions, even if a malformed or
    future proposal reports a white seed flag.  A proposal-owned box is
    executable on either temporal view.
    The offline RGB gate later removes physically inconsistent target/action
    directions before the top-k execution shortlist is frozen.
    """

    result: list[CandidateAction] = []
    seen: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise ValueError("live proposal catalog contains a non-object")
        region_id = proposal.get("region_id")
        audit_kind = proposal.get("audit_kind")
        if not isinstance(region_id, str) or not region_id or region_id in seen:
            raise ValueError("live proposal region IDs must be non-empty and unique")
        if audit_kind not in KNOWN_AUDIT_KINDS:
            raise ValueError(
                f"proposal {region_id} has unsupported audit_kind {audit_kind!r}"
            )
        seen.add(region_id)
        planned: list[tuple[str, AgentAction]] = []
        if audit_kind in FULL_LIVE_AUDIT_KINDS:
            raw_seed = proposal.get("component_seed_pixels")
            raw_box = proposal.get("box_pixels")
            if not isinstance(raw_seed, list) or len(raw_seed) != 2:
                raise ValueError(f"proposal {region_id} lacks a pixel seed")
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                raise ValueError(f"proposal {region_id} lacks a pixel box")
            seed = tuple(int(value) for value in raw_seed)
            box = tuple(int(value) for value in raw_box)
            flags = _proposal_flags(proposal)
            unique_views = sum(bool(flags[view]) for view in ("t1", "t2")) == 1
            for view in ("t1", "t2"):
                if flags[view] and audit_kind != "scan":
                    planned.append(
                        (
                            "runtime_planner_primary"
                            if unique_views and audit_kind == "present"
                            else "executor_reachable_extended_negative",
                            AgentAction(view, "negative_point", coordinate=seed),
                        )
                    )
                planned.append(
                    (
                        "executor_reachable_scan_point"
                        if audit_kind == "scan"
                        else "runtime_planner_primary"
                        if audit_kind == "missing"
                        and unique_views
                        and not flags[view]
                        else "executor_reachable_refinement"
                        if flags[view]
                        else "executor_reachable_complement"
                        if audit_kind == "present"
                        else "executor_reachable_extended_positive",
                        AgentAction(view, "positive_point", coordinate=seed),
                    )
                )
                planned.append(
                    (
                        "runtime_planner_primary"
                        if audit_kind == "scan"
                        else "executor_reachable_box",
                        AgentAction(view, "box", box=box),
                    )
                )
        # Non-unique and candidate-delta proposals deliberately authorize nothing.
        for reachability_tier, action in planned:
            candidate_id = "--".join(
                (
                    episode_id,
                    f"step-{step_index:03d}",
                    region_id,
                    action.action,
                    action.target_view,
                )
            )
            result.append(
                CandidateAction(
                    candidate_id,
                    region_id,
                    reachability_tier,
                    action,
                    dict(proposal),
                )
            )
    return result


def resolve_rgb_state(
    episode: RuntimeEpisode,
    step_index: int,
    region_id: str,
    target_view: str,
) -> str:
    step_mapping = episode.rgb_states_by_step.get(str(step_index), {})
    region_mapping = step_mapping.get(region_id)
    if region_mapping is None:
        region_mapping = episode.rgb_states_by_region.get(region_id)
    value = (region_mapping or episode.rgb_states).get(target_view, "uncertain")
    return _validate_rgb_state(value, "resolved RGB state")


def rgb_semantic_gate(action: AgentAction, observed_state: str) -> dict[str, Any]:
    expected = "background" if action.action == "negative_point" else "building"
    passed = observed_state == expected
    return {
        "passed": passed,
        "target_view": action.target_view,
        "action": action.action,
        "observed_rgb_state": observed_state,
        "required_rgb_state": expected,
        "policy": "negative_requires_background_positive_or_box_requires_building",
    }


def freeze_candidate_shortlist(
    episode: RuntimeEpisode,
    step_index: int,
    state: ChangeState,
    ground_truth: np.ndarray,
    actions: Sequence[CandidateAction],
    limit: int,
) -> tuple[list[dict[str, Any]], list[CandidateAction]]:
    """Offline pre-execution selector frozen by the preregistered top-4 policy."""

    if limit < 1:
        raise ValueError("candidate shortlist limit must be positive")
    error_mask = np.logical_xor(state.change_mask, np.asarray(ground_truth, dtype=bool))
    catalog: list[dict[str, Any]] = []
    by_id = {action.candidate_id: action for action in actions}
    for candidate in actions:
        observed = resolve_rgb_state(
            episode, step_index, candidate.region_id, candidate.action.target_view
        )
        semantic = rgb_semantic_gate(candidate.action, observed)
        raw_box = candidate.proposal.get("box_pixels")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError(f"proposal {candidate.region_id} lacks shortlist box_pixels")
        x1, y1, x2, y2 = (int(value) for value in raw_box)
        height, width = error_mask.shape
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ValueError(f"proposal {candidate.region_id} shortlist box is out of frame")
        catalog.append(
            {
                "candidate_id": candidate.candidate_id,
                "region_id": candidate.region_id,
                "reachability_tier": candidate.reachability_tier,
                "proposal": _json_safe(candidate.proposal),
                "action": candidate.action.to_dict(),
                "rgb_semantic_gate": semantic,
                "proposal_box_current_change_error_pixels": int(
                    error_mask[y1 : y2 + 1, x1 : x2 + 1].sum()
                ),
                "selected_for_execution": False,
            }
        )
    eligible = [item for item in catalog if item["rgb_semantic_gate"]["passed"]]
    eligible.sort(
        key=lambda item: (
            -int(item["proposal_box_current_change_error_pixels"]),
            str(item["candidate_id"]),
        )
    )
    selected_ids = [item["candidate_id"] for item in eligible[:limit]]
    selected_set = set(selected_ids)
    for item in catalog:
        item["selected_for_execution"] = item["candidate_id"] in selected_set
    return catalog, [by_id[candidate_id] for candidate_id in selected_ids]


def mask_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, int | float]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if prediction.shape != reference.shape:
        raise ValueError("prediction/reference shapes differ")
    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, ~reference).sum())
    fn = int(np.logical_and(~prediction, reference).sum())
    tn = int(np.logical_and(~prediction, ~reference).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0 if tp + fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pixel_error": fp + fn,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": tp + fp,
        "gt_pixels": tp + fn,
    }


def metric_delta(
    before: Mapping[str, int | float], after: Mapping[str, int | float]
) -> dict[str, int | float]:
    return {
        "pixel_error_reduction": int(before["pixel_error"]) - int(after["pixel_error"]),
        "iou": float(after["iou"]) - float(before["iou"]),
        "f1": float(after["f1"]) - float(before["f1"]),
        "tp": int(after["tp"]) - int(before["tp"]),
        "fp": int(after["fp"]) - int(before["fp"]),
        "fn": int(after["fn"]) - int(before["fn"]),
    }


def rank_beneficial_candidates(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = [
        record
        for record in records
        if record["hard_gate"]["passed"]
        and record["rgb_semantic_gate"]["passed"]
        and record["per_step_metric_gate"]["passed"]
    ]
    return sorted(
        eligible,
        key=lambda record: (
            -int(record["metric_delta"]["pixel_error_reduction"]),
            -float(record["metric_delta"]["iou"]),
            -float(record["metric_delta"]["f1"]),
            str(record["candidate_id"]),
        ),
    )


def _state_hashes(state: ChangeState) -> dict[str, str]:
    return {
        "t1": mask_sha256(state.t1_mask),
        "t2": mask_sha256(state.t2_mask),
        "change": mask_sha256(state.change_mask),
    }


def state_sha256(hashes: Mapping[str, str]) -> str:
    return canonical_sha256(
        {
            "t1_mask_sha256": hashes["t1"],
            "t2_mask_sha256": hashes["t2"],
            "change_mask_sha256": hashes["change"],
        }
    )


def _state_artifact(
    state: ChangeState, directory: Path, output_root: Path
) -> dict[str, Any]:
    masks = {
        "t1": _atomic_save_mask(directory / "t1.npy", state.t1_mask, output_root),
        "t2": _atomic_save_mask(directory / "t2.npy", state.t2_mask, output_root),
        "change": _atomic_save_mask(
            directory / "change.npy", state.change_mask, output_root
        ),
    }
    hashes = {name: masks[name]["mask_sha256"] for name in ("t1", "t2", "change")}
    return {
        "masks": masks,
        "mask_sha256": hashes,
        "state_sha256": state_sha256(hashes),
    }


def _proposal_catalog(state: ChangeState) -> tuple[dict[str, Any], ...]:
    raw = state.evidence.get("verifier_region_proposals")
    if not isinstance(raw, list):
        raise RuntimeError("Environment did not attach a live proposal catalog")
    return tuple(dict(item) for item in raw)


def _hard_gate(execution: Mapping[str, Any]) -> dict[str, Any]:
    rejected = bool(execution.get("verifier_skipped_by_hard_gate"))
    reasons = list(execution.get("candidate_rejection_reasons", []))
    if rejected and not reasons:
        evidence = execution.get("verifier_evidence", {})
        if isinstance(evidence, Mapping):
            reasons = list(evidence.get("rejection_reasons", []))
    return {
        "passed": not rejected,
        "rejected": rejected,
        "rejection_reasons": reasons,
        "verifier_skipped_by_hard_gate": rejected,
    }


def _tool_receipt(execution: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            key: execution.get(key)
            for key in (
                "tool",
                "tool_input",
                "tool_result",
                "locality",
                "point_authorization",
                "box_authorization",
            )
            if key in execution
        }
    )


def _assert_same_accepted_start(branch: ChangeAgentEnvironment, state: ChangeState) -> None:
    if branch.state is None:
        raise RuntimeError("branch Environment has no reset state")
    if _state_hashes(branch.state) != _state_hashes(state):
        raise RuntimeError("fresh branch did not start from the exact accepted masks")
    if canonical_sha256(_proposal_catalog(branch.state)) != canonical_sha256(
        _proposal_catalog(state)
    ):
        raise RuntimeError("fresh branch live proposal catalog differs from canonical state")


def generate_candidate_batch(
    episode: RuntimeEpisode,
    canonical_state: ChangeState,
    step_index: int,
    actions: Sequence[CandidateAction],
    proposals: Sequence[Mapping[str, Any]],
    runtime_factory: EpisodeRuntimeFactory,
    episode_dir: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[ChangeState], tuple[dict[str, Any], ...]]:
    """Execute the frozen GT-shortlisted actions through GT-free runtime APIs."""

    records: list[dict[str, Any]] = []
    states: list[ChangeState] = []
    for ordinal, candidate in enumerate(actions):
        candidate_root = episode_dir / "candidate_branches" / f"step-{step_index:03d}" / f"candidate-{ordinal:04d}"
        branch = runtime_factory.create_branch_environment(
            episode, canonical_state.clone(), candidate_root / "tool_runs"
        )
        branch.reset(episode.t1_image, episode.t2_image, episode.query)
        _assert_same_accepted_start(branch, canonical_state)
        branch.step(candidate.action, authorized_region_id=candidate.region_id)
        entry = branch.trajectory.entries[-1]
        state = entry.state.clone()
        state_artifact = _state_artifact(state, candidate_root / "masks", output_root)
        verifier = branch.verifier
        hard_gate = _hard_gate(entry.execution)
        if hard_gate["passed"] != bool(entry.execution.get("candidate_accepted")):
            raise RuntimeError("no-Qwen branch acceptance does not equal original hard-gate pass")
        observed_rgb = resolve_rgb_state(
            episode, step_index, candidate.region_id, candidate.action.target_view
        )
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "ordinal": ordinal,
                "region_id": candidate.region_id,
                "reachability_tier": candidate.reachability_tier,
                "proposal": _json_safe(candidate.proposal),
                "action": candidate.action.to_dict(),
                "source_state_mask_sha256": _state_hashes(canonical_state),
                "candidate_state": state_artifact,
                "hard_gate": hard_gate,
                "rgb_semantic_gate": rgb_semantic_gate(candidate.action, observed_rgb),
                "tool_receipt": _tool_receipt(entry.execution),
                "runtime_execution": _json_safe(entry.execution),
                "no_qwen_audit": {
                    "qwen_invoked": False,
                    "qwen_calls": int(getattr(verifier, "qwen_calls", 0)),
                    "initial_stub_calls": int(getattr(verifier, "initial_call_count", 0)),
                    "candidate_stub_calls": int(getattr(verifier, "candidate_call_count", 0)),
                },
                # Filled only by the offline phase after this complete loop.
                "metrics": None,
                "metric_delta": None,
            }
        )
        states.append(state)
    return records, states, tuple(dict(item) for item in proposals)


def score_completed_candidates(
    current_state: ChangeState,
    candidate_records: list[dict[str, Any]],
    candidate_states: Sequence[ChangeState],
    ground_truth: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[dict[str, int | float], list[Mapping[str, Any]]]:
    """Offline-only phase; no Environment, verifier, executor, or backend is reachable."""

    if len(candidate_records) != len(candidate_states):
        raise ValueError("candidate record/state count mismatch")
    current = mask_metrics(current_state.change_mask, ground_truth)
    for record, state in zip(candidate_records, candidate_states):
        metrics = mask_metrics(state.change_mask, ground_truth)
        record["metrics"] = metrics
        delta = metric_delta(current, metrics)
        record["metric_delta"] = delta
        nonempty_gt = bool(np.asarray(ground_truth, dtype=bool).any())
        pixel_error_pass = int(delta["pixel_error_reduction"]) > 0
        iou_pass = float(delta["iou"]) > epsilon
        f1_pass = float(delta["f1"]) > epsilon
        fp_pass = int(metrics["fp"]) < int(current["fp"])
        record["per_step_metric_gate"] = {
            "passed": bool(
                pixel_error_pass
                and ((iou_pass and f1_pass) if nonempty_gt else fp_pass)
            ),
            "pixel_error_strictly_decreased": pixel_error_pass,
            "nonempty_gt": nonempty_gt,
            "iou_strictly_increased": iou_pass if nonempty_gt else None,
            "f1_strictly_increased": f1_pass if nonempty_gt else None,
            "fp_strictly_decreased": fp_pass if not nonempty_gt else None,
            "comparison_epsilon": epsilon,
        }
    return current, rank_beneficial_candidates(candidate_records)


def endpoint_gate(
    metrics: Mapping[str, Any],
    *,
    minimum_iou: float = DEFAULT_ENDPOINT_IOU,
    minimum_f1: float = DEFAULT_ENDPOINT_F1,
    empty_gt_max_fp: int = DEFAULT_EMPTY_GT_MAX_FP,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate the preregistered absolute endpoint, separate from local progress."""

    nonempty_gt = int(metrics["gt_pixels"]) > 0
    if nonempty_gt:
        iou_passed = float(metrics["iou"]) + epsilon >= minimum_iou
        f1_passed = float(metrics["f1"]) + epsilon >= minimum_f1
        passed = iou_passed and f1_passed
        fp_passed = None
    else:
        iou_passed = None
        f1_passed = None
        fp_passed = int(metrics["fp"]) <= empty_gt_max_fp
        passed = fp_passed
    return {
        "passed": bool(passed),
        "nonempty_gt": nonempty_gt,
        "minimum_iou": minimum_iou if nonempty_gt else None,
        "minimum_f1": minimum_f1 if nonempty_gt else None,
        "empty_gt_max_fp": empty_gt_max_fp if not nonempty_gt else None,
        "iou_passed": iou_passed,
        "f1_passed": f1_passed,
        "fp_residual_passed": fp_passed,
        "comparison_epsilon": epsilon,
    }


def run_episode(
    pilot: PilotEpisode,
    runtime_factory: EpisodeRuntimeFactory,
    output_root: Path,
    *,
    max_steps: int = MAX_CANONICAL_STEPS,
    max_candidates_per_step: int = MAX_CANDIDATES_PER_STEP,
    max_tool_executions: int = MAX_TOOL_EXECUTIONS,
    comparison_epsilon: float = 1e-12,
    minimum_local_error_reduction_ratio: float = DEFAULT_LOCAL_ERROR_REDUCTION_RATIO,
    endpoint_minimum_iou: float = DEFAULT_ENDPOINT_IOU,
    endpoint_minimum_f1: float = DEFAULT_ENDPOINT_F1,
    endpoint_empty_gt_max_fp: int = DEFAULT_EMPTY_GT_MAX_FP,
    expected_gt_pixels: int | None = None,
    label_loader: Callable[[LabelSource], np.ndarray] = load_label_mask,
) -> dict[str, Any]:
    if max_steps != MAX_CANONICAL_STEPS:
        raise ValueError("canonical episode max_steps must equal 3")
    if max_candidates_per_step != MAX_CANDIDATES_PER_STEP:
        raise ValueError("max_candidates_per_step must equal 4")
    if max_tool_executions != MAX_TOOL_EXECUTIONS:
        raise ValueError("max_tool_executions must equal 24")
    episode = pilot.runtime
    episode_dir = output_root / "episodes" / episode.episode_id
    canonical = runtime_factory.create_canonical_environment(
        episode, episode_dir / "canonical_tool_runs", max_steps
    )
    canonical.reset(episode.t1_image, episode.t2_image, episode.query)
    if canonical.state is None:
        raise RuntimeError("canonical Environment reset produced no state")
    initial_state = canonical.state.clone()
    initial_artifact = _state_artifact(
        initial_state, episode_dir / "canonical_states" / "step-000", output_root
    )
    initial_artifact["initializer_receipt"] = _json_safe(
        initial_state.evidence.get("sam3_initialization")
    )
    rounds: list[dict[str, Any]] = []
    canonical_steps: list[dict[str, Any]] = []
    ground_truth: np.ndarray | None = None
    gt_artifact: dict[str, Any] | None = None
    stop_reason = "max_steps"
    budget_hit = False
    tool_execution_count = 0
    distinct_candidates: set[str] = set()
    gt_access_count = 0
    catalog_freezes: list[dict[str, Any]] = []
    initial_endpoint_gate: dict[str, Any] | None = None
    endpoint_success_step: int | None = None

    for step_index in range(1, max_steps + 1):
        if canonical.state is None:
            raise RuntimeError("canonical Environment lost its state")
        source_state = canonical.state.clone()
        # Complete the GT-free live catalog before the offline selector may open GT.
        proposals = _proposal_catalog(source_state)
        actions = enumerate_live_actions(episode.episode_id, step_index, proposals)
        gt_free_actions = [
            {
                "candidate_id": item.candidate_id,
                "region_id": item.region_id,
                "reachability_tier": item.reachability_tier,
                "action": item.action.to_dict(),
                "proposal": _json_safe(item.proposal),
            }
            for item in actions
        ]
        catalog_payload = {
            "schema_id": "qwen3vl_oracle_trajectory_catalog_freeze_v1",
            "episode_id": episode.episode_id,
            "step_index": step_index,
            "source_state_sha256": state_sha256(_state_hashes(source_state)),
            "source_state_mask_sha256": _state_hashes(source_state),
            "live_proposals": _json_safe(proposals),
            "live_proposals_canonical_sha256": canonical_sha256(proposals),
            "actions": gt_free_actions,
            "action_catalog_canonical_sha256": canonical_sha256(gt_free_actions),
            "action_count": len(gt_free_actions),
            "gt_access_count_before": gt_access_count,
            "contains_gt_derived_fields": False,
            "qwen_invoked": False,
            "qwen_calls": 0,
        }
        catalog_name = (
            "catalog_freeze_before_gt.json"
            if gt_access_count == 0
            else f"catalog_freeze_step-{step_index:03d}.json"
        )
        catalog_path = episode_dir / catalog_name
        _atomic_write_json(catalog_path, catalog_payload)
        catalog_identity = {
            "path": str(catalog_path.relative_to(output_root)),
            "sha256": sha256_file(catalog_path),
            "canonical_sha256": canonical_sha256(catalog_payload),
            "size_bytes": catalog_path.stat().st_size,
            "step_index": step_index,
        }
        catalog_freezes.append(catalog_identity)
        if ground_truth is None:
            ground_truth = np.asarray(label_loader(pilot.label), dtype=bool)
            gt_access_count += 1
            if ground_truth.shape != source_state.change_mask.shape:
                raise ValueError("offline label shape differs from canonical state")
            if expected_gt_pixels is not None and int(ground_truth.sum()) != expected_gt_pixels:
                raise ValueError("offline label pixels differ from preregistered crop")
            gt_artifact = _atomic_save_mask(
                episode_dir / "offline_ground_truth.npy", ground_truth, output_root
            )
        current_metrics = mask_metrics(source_state.change_mask, ground_truth)
        current_endpoint = endpoint_gate(
            current_metrics,
            minimum_iou=endpoint_minimum_iou,
            minimum_f1=endpoint_minimum_f1,
            empty_gt_max_fp=endpoint_empty_gt_max_fp,
            epsilon=comparison_epsilon,
        )
        if initial_endpoint_gate is None:
            initial_endpoint_gate = current_endpoint
        action_catalog, execution_shortlist = freeze_candidate_shortlist(
            episode,
            step_index,
            source_state,
            ground_truth,
            actions,
            max_candidates_per_step,
        )
        round_record: dict[str, Any] = {
            "step_index": step_index,
            "source_state_mask_sha256": _state_hashes(source_state),
            "source_state_sha256": state_sha256(_state_hashes(source_state)),
            "source_metrics": current_metrics,
            "source_endpoint_gate": current_endpoint,
            "live_proposals": _json_safe(proposals),
            "live_proposals_canonical_sha256": canonical_sha256(proposals),
            "action_catalog": action_catalog,
            "gt_free_catalog_freeze": catalog_identity,
            "action_catalog_count": len(action_catalog),
            "execution_shortlist": [item.candidate_id for item in execution_shortlist],
            "candidate_count": 0,
            "candidates": [],
            "beneficial_shortlist": [],
            "selected_candidate_id": None,
            "canonical_replay": None,
            "candidate_batch_complete_before_final_gt_scoring": False,
        }
        if current_endpoint["passed"]:
            stop_reason = (
                "already_exact"
                if int(current_metrics["pixel_error"]) == 0
                else "already_endpoint"
            )
            round_record["candidate_batch_complete_before_final_gt_scoring"] = True
            rounds.append(round_record)
            break
        if tool_execution_count + len(execution_shortlist) > max_tool_executions:
            stop_reason = "budget_hit"
            budget_hit = True
            rounds.append(round_record)
            break
        records, states, _ = generate_candidate_batch(
            episode,
            source_state,
            step_index,
            execution_shortlist,
            proposals,
            runtime_factory,
            episode_dir,
            output_root,
        )
        tool_execution_count += len(records)
        distinct_candidates.update(record["candidate_id"] for record in records)
        if len(distinct_candidates) > max_candidates_per_step * max_steps:
            raise RuntimeError("distinct candidate budget invariant was exceeded")
        current_metrics, beneficial = score_completed_candidates(
            source_state,
            records,
            states,
            ground_truth,
            epsilon=comparison_epsilon,
        )
        round_record["candidate_count"] = len(records)
        round_record["candidates"] = records
        round_record["beneficial_shortlist"] = [
            record["candidate_id"] for record in beneficial
        ]
        round_record["candidate_batch_complete_before_final_gt_scoring"] = True
        if not beneficial:
            stop_reason = "no_beneficial"
            rounds.append(round_record)
            break

        selected = beneficial[0]
        selected_ordinal = int(selected["ordinal"])
        selected_state = states[selected_ordinal]
        selected_action = AgentAction(
            str(selected["action"]["target_view"]),
            str(selected["action"]["action"]),
            coordinate=(
                tuple(int(value) for value in selected["action"]["coordinate"])
                if "coordinate" in selected["action"]
                else None
            ),
            box=(
                tuple(int(value) for value in selected["action"]["box"])
                if "box" in selected["action"]
                else None
            ),
        )
        if tool_execution_count + 1 > max_tool_executions:
            stop_reason = "budget_hit"
            budget_hit = True
            rounds.append(round_record)
            break
        canonical.step(selected_action, authorized_region_id=str(selected["region_id"]))
        tool_execution_count += 1
        replay_entry = canonical.trajectory.entries[-1]
        if not replay_entry.execution.get("candidate_accepted"):
            raise RuntimeError("shortlisted gate-pass action was rejected during canonical replay")
        branch_hashes = _state_hashes(selected_state)
        replay_hashes = _state_hashes(replay_entry.state)
        if replay_hashes != branch_hashes:
            raise RuntimeError("canonical replay masks differ from shortlisted branch")
        if canonical.state is None or _state_hashes(canonical.state) != branch_hashes:
            raise RuntimeError("canonical accepted live state differs from replay candidate")
        replay_artifact = _state_artifact(
            canonical.state,
            episode_dir / "canonical_states" / f"step-{step_index:03d}",
            output_root,
        )
        replay_metrics = mask_metrics(canonical.state.change_mask, ground_truth)
        replay_endpoint = endpoint_gate(
            replay_metrics,
            minimum_iou=endpoint_minimum_iou,
            minimum_f1=endpoint_minimum_f1,
            empty_gt_max_fp=endpoint_empty_gt_max_fp,
            epsilon=comparison_epsilon,
        )
        round_record["selected_candidate_id"] = selected["candidate_id"]
        round_record["canonical_replay"] = {
            "branch_candidate_state_mask_sha256": branch_hashes,
            "canonical_state_mask_sha256": replay_hashes,
            "all_three_mask_hashes_match": True,
            "state": replay_artifact,
            "metrics": replay_metrics,
            "endpoint_gate": replay_endpoint,
            "tool_receipt": _tool_receipt(replay_entry.execution),
            "hard_gate": _hard_gate(replay_entry.execution),
            "runtime_execution": _json_safe(replay_entry.execution),
        }
        canonical_steps.append(
            {
                "step_index": step_index,
                "action": selected_action.to_dict(),
                "proposal": selected["proposal"],
                "candidate_id": selected["candidate_id"],
                "reachability_tier": selected["reachability_tier"],
                "source_state_sha256": state_sha256(_state_hashes(source_state)),
                "source_state_mask_sha256": _state_hashes(source_state),
                "candidate_state": selected["candidate_state"],
                "canonical_state": replay_artifact,
                "replay_state": replay_artifact,
                "metrics_before": current_metrics,
                "metrics_after": replay_metrics,
                "endpoint_gate": replay_endpoint,
                "metric_delta": selected["metric_delta"],
                "per_step_metric_gate": selected["per_step_metric_gate"],
                "hard_gate": selected["hard_gate"],
                "rgb_semantic": selected["rgb_semantic_gate"],
                "tool_receipt": _tool_receipt(replay_entry.execution),
                "tool_success": True,
                "candidate_accepted_by_oracle_selector": True,
                "branch_replay_mask_hash_match": True,
                "budget_hit": False,
                "qwen_invoked": False,
                "qwen_calls": 0,
            }
        )
        rounds.append(round_record)
        if replay_endpoint["passed"]:
            endpoint_success_step = step_index
            stop_reason = (
                "exact_recovery"
                if int(replay_metrics["pixel_error"]) == 0
                else "endpoint_reached"
            )
            break

    if ground_truth is None or gt_artifact is None:
        raise RuntimeError("episode ended without the offline scoring phase")
    if canonical.state is None:
        raise RuntimeError("canonical Environment has no final state")
    final_metrics = mask_metrics(canonical.state.change_mask, ground_truth)
    final_endpoint_gate = endpoint_gate(
        final_metrics,
        minimum_iou=endpoint_minimum_iou,
        minimum_f1=endpoint_minimum_f1,
        empty_gt_max_fp=endpoint_empty_gt_max_fp,
        epsilon=comparison_epsilon,
    )
    if stop_reason == "max_steps" and len(canonical_steps) != max_steps:
        raise RuntimeError("max_steps termination did not consume the canonical budget")
    if int(getattr(runtime_factory, "qwen_calls", 0)) != 0:
        raise RuntimeError("oracle trajectory contract forbids every Qwen call")
    initial_artifact["metrics"] = mask_metrics(initial_state.change_mask, ground_truth)
    initial_error = int(initial_artifact["metrics"]["pixel_error"])
    reduction_ratio = (
        (initial_error - int(final_metrics["pixel_error"])) / initial_error
        if initial_error
        else None
    )
    if initial_endpoint_gate is None:
        raise RuntimeError("initial endpoint gate was not evaluated")
    locally_reachable = bool(
        canonical_steps
        and reduction_ratio is not None
        and reduction_ratio + comparison_epsilon
        >= minimum_local_error_reduction_ratio
    )
    if initial_error == 0:
        episode_status = "already_exact"
    elif initial_endpoint_gate["passed"]:
        episode_status = "already_endpoint"
    elif budget_hit:
        episode_status = "inconclusive_budget_exhausted"
    elif canonical_steps and final_endpoint_gate["passed"]:
        episode_status = "endpoint_success"
    elif locally_reachable:
        episode_status = "locally_reachable_only"
    else:
        episode_status = "not_reachable_within_frozen_pilot"
    return {
        "schema_id": EPISODE_SCHEMA_ID,
        "episode_id": episode.episode_id,
        "sample_id": episode.sample_id,
        "split": episode.split,
        "parent_id": episode.parent_id,
        "query": episode.query,
        "crop_xyxy_exclusive": list(episode.crop_xyxy_exclusive),
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
        "gt_mask": gt_artifact,
        "rgb_semantics": {
            "default": dict(episode.rgb_states),
            "evidence": episode.rgb_semantic_evidence,
            "source": episode.rgb_semantic_source,
            "status": episode.rgb_semantic_status,
            "by_region": _json_safe(episode.rgb_states_by_region),
            "by_step": _json_safe(episode.rgb_states_by_step),
        },
        "episode_status": episode_status,
        "max_steps": max_steps,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "gt_access_audit": {
            "access_count": gt_access_count,
            "first_access_phase": "after_initial_live_action_catalog_freeze_before_offline_top4_shortlist",
            "catalog_freeze_before_first_access": catalog_freezes[0],
            "catalog_freezes": catalog_freezes,
            "label_passed_to_runtime_factory": False,
            "label_passed_to_environment": False,
            "label_passed_to_initializer_or_tools": False,
            "runtime_factory_sample_type": "RuntimeEpisode_GT_free",
            "offline_label_loader_only": True,
        },
        "initial_state": initial_artifact,
        "selection_rounds": rounds,
        "steps": canonical_steps,
        "termination": {
            "reason": stop_reason,
            "selected_step_count": len(canonical_steps),
            "candidate_count": sum(len(item["candidates"]) for item in rounds),
            "exact": int(final_metrics["pixel_error"]) == 0,
            "budget_hit": budget_hit,
        },
        "budget_hit": budget_hit,
        "budget": {
            "hit": budget_hit,
            "max_candidates_per_step": max_candidates_per_step,
            "max_distinct_candidates": max_candidates_per_step * max_steps,
            "max_total_tool_executions": max_tool_executions,
            "distinct_candidate_count": len(distinct_candidates),
            "tool_execution_count": tool_execution_count,
        },
        "final_state_mask_sha256": _state_hashes(canonical.state),
        "final_state_sha256": state_sha256(_state_hashes(canonical.state)),
        "final_metrics": final_metrics,
        "final_error_reduction_ratio": reduction_ratio,
        "local_reachability": {
            "passed": locally_reachable,
            "minimum_final_error_reduction_ratio": minimum_local_error_reduction_ratio,
        },
        "endpoint_reachability": {
            "initial_gate": initial_endpoint_gate,
            "final_gate": final_endpoint_gate,
            "success_step": endpoint_success_step,
            "exact_recovery": int(final_metrics["pixel_error"]) == 0,
        },
    }


def _pair_processor(config: Mapping[str, Any]) -> MaskPairProcessor:
    raw = _mapping(config.get("matching", {}), "matching")
    return MaskPairProcessor(
        overlap_threshold=float(raw.get("overlap_threshold", 0.25)),
        matching_mode=str(raw.get("mode", "overlap_presence")),
        t12_min_instance_area=int(raw.get("t12_min_instance_area", 0)),
        cd_min_instance_area=int(raw.get("cd_min_instance_area", 0)),
    )


def _proposal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(config.get("proposal_config", {}), "proposal_config")
    return {
        "verifier_max_regions": int(raw.get("max_regions", 6)),
        "verifier_max_delta_regions": int(raw.get("max_delta_regions", 3)),
        "verifier_min_region_area": int(raw.get("min_component_area", 4)),
        "verifier_region_padding_ratio": float(raw.get("padding_ratio", 0.25)),
        "empty_mask_scan_grid_size": int(raw.get("empty_mask_scan_grid_size", 3)),
    }


def _gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(
        config.get("hard_gate_config", config.get("gate_config", {})),
        "hard_gate_config",
    )
    return {
        "selection_policy": "conservative_best",
        "selection_epsilon": 0.0,
        "max_selection_area_delta": float(raw.get("max_selection_area_delta", 0.25)),
        "max_locality_outside_ratio": float(raw.get("max_locality_outside_ratio", 0.1)),
        "max_target_mask_change_ratio": float(raw.get("max_target_mask_change_ratio", 0.25)),
        "max_component_count_delta": int(raw.get("max_component_count_delta", 4)),
        "max_point_proposal_change_ratio": float(raw.get("max_point_proposal_change_ratio", 0.25)),
        "max_point_proposal_change_pixels": int(raw.get("max_point_proposal_change_pixels", 256)),
        "max_point_image_change_ratio": float(raw.get("max_point_image_change_ratio", 0.004)),
        "point_proposal_change_gate_enabled": bool(raw.get("point_proposal_change_gate_enabled", True)),
    }


def _tool_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    raw = _mapping(
        config.get("tools_and_models", config.get("tools")), "tools_and_models"
    )
    result: dict[str, Path] = {}
    for name in (
        "segagent_python",
        "omniovcd_python",
        "simpleclick_checkpoint",
        "sam3_checkpoint",
        "sam3_bpe",
    ):
        identity = _file_identity(raw.get(name), f"tools_and_models.{name}")
        result[name] = _verified_file(identity, f"tools_and_models.{name}")
    workspace_raw = raw.get("workspace_root")
    workspace_path = (
        workspace_raw.get("path") if isinstance(workspace_raw, Mapping) else workspace_raw
    )
    if not isinstance(workspace_path, str) or not Path(workspace_path).resolve().is_dir():
        raise ValueError("tools_and_models.workspace_root must be an existing directory")
    result["workspace_root"] = Path(workspace_path).resolve()
    return result


def build_runtime_factory_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit GT-free subset that production runtime may receive.

    The preregistration also contains label identities, GT pixel counts, sample
    kinds, offline metric thresholds, and data-derivation policy.  Passing the
    whole object to ``ProductionRuntimeFactory`` would make the claimed runtime
    boundary false even if today's factory happened not to read those fields.
    Keep this allowlist narrow so future runtime code cannot accidentally gain
    access to offline-only data by looking elsewhere in the manifest.
    """

    tools = _mapping(
        manifest.get("tools_and_models", manifest.get("tools")),
        "tools_and_models",
    )
    protocol = _mapping(manifest.get("execution_protocol", {}), "execution_protocol")
    matching = _mapping(manifest.get("matching", {}), "matching")
    proposals = _mapping(manifest.get("proposal_config", {}), "proposal_config")
    gates = _mapping(
        manifest.get("hard_gate_config", manifest.get("gate_config", {})),
        "hard_gate_config",
    )

    def selected(source: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
        return {name: source[name] for name in names if name in source}

    return {
        "tools_and_models": selected(
            tools,
            (
                "workspace_root",
                "segagent_python",
                "omniovcd_python",
                "simpleclick_checkpoint",
                "sam3_checkpoint",
                "sam3_bpe",
            ),
        ),
        "execution_protocol": selected(
            protocol,
            ("seed", "sam3_resolution", "tool_device", "tool_timeout_seconds"),
        ),
        "matching": selected(
            matching,
            (
                "mode",
                "overlap_threshold",
                "t12_min_instance_area",
                "cd_min_instance_area",
            ),
        ),
        "proposal_config": selected(
            proposals,
            (
                "max_regions",
                "max_delta_regions",
                "min_component_area",
                "padding_ratio",
                "empty_mask_scan_grid_size",
            ),
        ),
        "hard_gate_config": selected(
            gates,
            (
                "max_selection_area_delta",
                "max_locality_outside_ratio",
                "max_target_mask_change_ratio",
                "max_component_count_delta",
                "max_point_proposal_change_ratio",
                "max_point_proposal_change_pixels",
                "max_point_image_change_ratio",
                "point_proposal_change_gate_enabled",
            ),
        ),
        "authorized_negative_point_input_scope": manifest.get(
            "authorized_negative_point_input_scope", "full_session"
        ),
    }


class ProductionRuntimeFactory:
    """Construct real isolated tools and fresh Environment instances."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.tools = _tool_paths(config)
        self.repo_root = Path(__file__).resolve().parents[1]
        self.worker = self.repo_root / "tools" / "seeded_segmentation_worker.py"
        protocol = _mapping(config.get("execution_protocol", {}), "execution_protocol")
        self.seed = int(protocol.get("seed", config.get("seed", 42)))
        self.resolution = int(protocol.get("sam3_resolution", config.get("sam3_resolution", 1008)))
        self.timeout = int(protocol.get("tool_timeout_seconds", config.get("tool_timeout_seconds", 900)))
        self.device = str(protocol.get("tool_device", config.get("tool_device", "cuda")))
        self._verifiers: list[NoQwenAcceptVerifier] = []

    @property
    def qwen_calls(self) -> int:
        return sum(verifier.qwen_calls for verifier in self._verifiers)

    def _tool_backends(
        self, artifact_dir: Path
    ) -> tuple[SubprocessPointBackend, SubprocessBoxBackend]:
        workspace = self.tools["workspace_root"]
        point = SubprocessPointBackend(
            self.tools["segagent_python"],
            self.worker,
            artifact_dir / "simpleclick",
            checkpoint=self.tools["simpleclick_checkpoint"],
            pythonpath=(
                self.repo_root,
                workspace / "reference" / "SegAgent",
                workspace / "reference" / "SegAgent" / "third_party" / "SimpleClick",
            ),
            device=self.device,
            timeout_seconds=self.timeout,
            seed=self.seed,
        )
        box = SubprocessBoxBackend(
            self.tools["omniovcd_python"],
            self.worker,
            artifact_dir / "sam3_box",
            checkpoint=self.tools["sam3_checkpoint"],
            bpe=self.tools["sam3_bpe"],
            resolution=self.resolution,
            pythonpath=(self.repo_root, workspace / "reference" / "OmniOVCD"),
            device=self.device,
            timeout_seconds=self.timeout,
            seed=self.seed,
        )
        return point, box

    def _environment(
        self,
        initialize_masks: Callable[[np.ndarray, np.ndarray, str], Any],
        point: Any,
        box: Any,
        max_steps: int,
    ) -> ChangeAgentEnvironment:
        verifier = NoQwenAcceptVerifier()
        self._verifiers.append(verifier)
        backend = OmniOVCDAdapter(initialize_masks, box.segment_box, _pair_processor(self.config))
        return ChangeAgentEnvironment(
            backend,
            ActionExecutor(
                point,
                box,
                authorized_negative_point_input_scope=str(
                    self.config.get("authorized_negative_point_input_scope", "full_session")
                ),
            ),
            verifier,
            max_steps=max_steps,
            require_tool_before_finish=False,
            enable_verifier_regions=True,
            **_proposal_config(self.config),
            **_gate_config(self.config),
        )

    def create_canonical_environment(
        self, episode: RuntimeEpisode, artifact_dir: Path, max_steps: int
    ) -> ChangeAgentEnvironment:
        del episode
        workspace = self.tools["workspace_root"]
        initializer = SubprocessSAM3Initializer(
            self.tools["omniovcd_python"],
            self.worker,
            artifact_dir / "fresh_sam3_initializer",
            checkpoint=self.tools["sam3_checkpoint"],
            bpe=self.tools["sam3_bpe"],
            resolution=self.resolution,
            pythonpath=(self.repo_root, workspace / "reference" / "OmniOVCD"),
            device=self.device,
            timeout_seconds=self.timeout,
            seed=self.seed,
        )
        point, box = self._tool_backends(artifact_dir)
        return self._environment(initializer.initialize_masks, point, box, max_steps)

    def create_branch_environment(
        self,
        episode: RuntimeEpisode,
        accepted_state: ChangeState,
        artifact_dir: Path,
    ) -> ChangeAgentEnvironment:
        del episode
        t1_mask = np.array(accepted_state.t1_mask, dtype=bool, copy=True)
        t2_mask = np.array(accepted_state.t2_mask, dtype=bool, copy=True)

        def cached_initializer(
            t1_image: np.ndarray, t2_image: np.ndarray, query: str
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            del t1_image, t2_image, query
            return (
                np.array(t1_mask, copy=True),
                np.array(t2_mask, copy=True),
                {
                    "initializer": "accepted_live_mask_branch",
                    "source_state_mask_sha256": _state_hashes(accepted_state),
                },
            )

        point, box = self._tool_backends(artifact_dir)
        return self._environment(cached_initializer, point, box, 1)


def _runtime_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("runtime_commit must be a full 40-character Git SHA")
    return value.lower()


def _prepare_output(path: Path) -> Path:
    target = path.resolve()
    if target.exists():
        raise FileExistsError(f"output path already exists: {target}")
    target.mkdir(parents=True)
    return target


def _pilot_sample_specs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    samples = manifest.get("samples")
    if isinstance(samples, list):
        result: dict[str, Any] = {}
        for index, item in enumerate(samples):
            mapping = dict(_mapping(item, f"samples[{index}]"))
            sample_id = mapping.get("sample_id", mapping.get("episode_id"))
            if not isinstance(sample_id, str) or not sample_id or sample_id in result:
                raise ValueError("list samples require unique sample_id values")
            mapping.setdefault("episode_id", sample_id)
            result[sample_id] = mapping
        return result
    if isinstance(samples, Mapping):
        order = manifest.get("ordered_samples")
        if not isinstance(order, list) or order != list(samples):
            raise ValueError("mapping samples require exact ordered_samples order")
        return {
            name: {"sample_id": name, "episode_id": name, **dict(samples[name])}
            for name in order
        }
    raise ValueError("pilot manifest samples must be a list or ordered mapping")


def _assigned_sample_ids(
    manifest: Mapping[str, Any], shard_index: int, num_shards: int
) -> list[str]:
    ordered = manifest.get("ordered_samples")
    if not isinstance(ordered, list):
        ordered = list(_pilot_sample_specs(manifest))
    resource = manifest.get("resource_plan")
    if isinstance(resource, Mapping) and isinstance(resource.get("shards"), list):
        shards = resource["shards"]
        if num_shards != len(shards):
            raise ValueError("--num-shards differs from the preregistered resource plan")
        match = next(
            (
                item
                for item in shards
                if isinstance(item, Mapping) and item.get("shard_id") == shard_index
            ),
            None,
        )
        if match is None or not isinstance(match.get("samples"), list):
            raise ValueError("--shard-index is absent from the preregistered resource plan")
        assigned = list(match["samples"])
    else:
        if not 0 <= shard_index < num_shards:
            raise ValueError("invalid shard index/count")
        assigned = [name for index, name in enumerate(ordered) if index % num_shards == shard_index]
    if any(name not in ordered for name in assigned) or len(assigned) != len(set(assigned)):
        raise ValueError("preregistered shard sample assignment is invalid")
    return assigned


def run(
    config_path: Path,
    output: Path,
    runtime_commit: str,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    runtime_factory: EpisodeRuntimeFactory | None = None,
    label_loader: Callable[[LabelSource], np.ndarray] = load_label_mask,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    manifest = read_strict_json(config_path)
    if manifest.get("schema_id") != PILOT_SCHEMA_ID:
        raise ValueError(f"pilot schema_id must be {PILOT_SCHEMA_ID}")
    runtime_commit = _runtime_commit(runtime_commit)
    protocol, budget = _validate_frozen_runtime_contract(manifest)
    max_steps = int(protocol["max_steps"])
    max_candidates = int(budget["max_candidate_actions_per_step"])
    max_tool_executions = int(
        budget[
            "max_total_tool_executions_including_selected_action_replay_per_sample"
        ]
    )
    metric_contract = _mapping(
        manifest.get("metric_and_success_contract", {}),
        "metric_and_success_contract",
    )
    epsilon = float(metric_contract.get("comparison_epsilon", 1e-12))
    local_contract = _mapping(
        metric_contract.get("locally_reachable", {}),
        "metric_and_success_contract.locally_reachable",
    )
    minimum_reduction = float(
        local_contract.get(
            "minimum_final_error_reduction_ratio",
            DEFAULT_LOCAL_ERROR_REDUCTION_RATIO,
        )
    )
    endpoint_contract = _mapping(
        metric_contract.get("endpoint_success", {}),
        "metric_and_success_contract.endpoint_success",
    )
    endpoint_iou = float(
        endpoint_contract.get("nonempty_gt_minimum_final_iou", DEFAULT_ENDPOINT_IOU)
    )
    endpoint_f1 = float(
        endpoint_contract.get("nonempty_gt_minimum_final_f1", DEFAULT_ENDPOINT_F1)
    )
    endpoint_empty_fp = int(
        endpoint_contract.get("empty_gt_maximum_final_fp", DEFAULT_EMPTY_GT_MAX_FP)
    )
    if not 0.0 < minimum_reduction <= 1.0:
        raise ValueError("local reachability reduction ratio must be within (0, 1]")
    if not 0.0 < endpoint_iou <= 1.0 or not 0.0 < endpoint_f1 <= 1.0:
        raise ValueError("non-empty endpoint IoU/F1 must be within (0, 1]")
    if not 0 <= endpoint_empty_fp < PATCH_SIZE * PATCH_SIZE:
        raise ValueError("empty-GT endpoint FP residual must be within the crop")
    output = _prepare_output(output)
    factory = runtime_factory or ProductionRuntimeFactory(
        build_runtime_factory_config(manifest)
    )
    query = str(protocol.get("query", manifest.get("query", "building")))
    episodes: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    sample_specs = _pilot_sample_specs(manifest)
    assigned = _assigned_sample_ids(manifest, shard_index, num_shards)
    for index, sample_id in enumerate(assigned):
        raw_spec = sample_specs[sample_id]
        parsed, label_source = parse_pilot_episode(raw_spec, index, query)
        runtime_episode = load_runtime_episode(parsed)
        if runtime_episode.episode_id in seen:
            raise ValueError(f"duplicate episode_id: {runtime_episode.episode_id}")
        seen.add(runtime_episode.episode_id)
        record = run_episode(
            PilotEpisode(runtime_episode, label_source),
            factory,
            output,
            max_steps=max_steps,
            max_candidates_per_step=max_candidates,
            max_tool_executions=max_tool_executions,
            comparison_epsilon=epsilon,
            minimum_local_error_reduction_ratio=minimum_reduction,
            endpoint_minimum_iou=endpoint_iou,
            endpoint_minimum_f1=endpoint_f1,
            endpoint_empty_gt_max_fp=endpoint_empty_fp,
            expected_gt_pixels=int(
                _mapping(raw_spec.get("crop", {}), f"samples.{sample_id}.crop").get(
                    "gt_pixels", -1
                )
            ),
            label_loader=label_loader,
        )
        episode_path = output / "episodes" / runtime_episode.episode_id / "episode.json"
        _atomic_write_json(episode_path, record)
        identity = {
            "episode_id": runtime_episode.episode_id,
            "path": str(episode_path.relative_to(output)),
            "sha256": sha256_file(episode_path),
            "canonical_sha256": canonical_sha256(record),
            "size_bytes": episode_path.stat().st_size,
            "termination_reason": record["termination"]["reason"],
            "selected_step_count": record["termination"]["selected_step_count"],
            "candidate_count": record["termination"]["candidate_count"],
        }
        episodes.append(record)
        identities.append(identity)

    qwen_calls = int(getattr(factory, "qwen_calls", 0))
    if qwen_calls != 0:
        raise RuntimeError("oracle trajectory run made a forbidden Qwen call")
    counts = {
        "episodes": len(episodes),
        "selected_steps": sum(len(item["steps"]) for item in episodes),
        "candidates": sum(item["termination"]["candidate_count"] for item in episodes),
        "endpoint_success": sum(
            item["episode_status"] == "endpoint_success" for item in episodes
        ),
        "locally_reachable_only": sum(
            item["episode_status"] == "locally_reachable_only" for item in episodes
        ),
        "already_exact": sum(item["episode_status"] == "already_exact" for item in episodes),
        "already_endpoint": sum(
            item["episode_status"] == "already_endpoint" for item in episodes
        ),
        "not_reachable_within_frozen_pilot": sum(
            item["episode_status"] == "not_reachable_within_frozen_pilot" for item in episodes
        ),
        "inconclusive_budget_exhausted": sum(
            item["episode_status"] == "inconclusive_budget_exhausted" for item in episodes
        ),
    }
    index = {
        "schema_id": INDEX_SCHEMA_ID,
        "status": "completed",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "episode_ids": assigned,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_sha256(manifest),
        },
        "runtime_commit": runtime_commit,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "qwen_invoked": False,
        "qwen_calls": 0,
        "counts": counts,
        "episodes": identities,
    }
    index_path = output / "shard_index.json"
    _atomic_write_json(index_path, index)
    marker = {
        "schema_id": COMPLETE_SCHEMA_ID,
        "status": "completed",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "episode_ids": assigned,
        "runtime_commit": runtime_commit,
        "qwen_invoked": False,
        "qwen_calls": 0,
        "counts": counts,
        "shard_index_artifact": {
            "path": str(index_path.relative_to(output)),
            "sha256": sha256_file(index_path),
            "canonical_sha256": canonical_sha256(index),
            "size_bytes": index_path.stat().st_size,
        },
        "shard_index_sha256": sha256_file(index_path),
    }
    _atomic_write_json(output / "generation_complete.json", marker)
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--runtime-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker = run(
        args.config,
        args.output,
        args.runtime_commit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(
        "status=complete;qwen_calls=0;"
        f"episodes={marker['counts']['episodes']};"
        f"steps={marker['counts']['selected_steps']};"
        f"candidates={marker['counts']['candidates']}"
    )


if __name__ == "__main__":
    main()
