#!/usr/bin/env python3
"""Generate the frozen P2 actionability candidates without loading Qwen or GT.

The execution manifest is deliberately label-free.  Every action is replayed from
the same cached P1 T1/T2 masks in a fresh ChangeAgentEnvironment, so the production
geometry authorization, Executor composition, pair rebuild, and pre-Qwen hard gates
remain the authority for the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.adapters.subprocess_adapters import SubprocessBoxBackend, SubprocessPointBackend
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction, ChangeState, VerifierOutput
from change_agent.trajectory import mask_sha256
from change_agent.verifier_protocol import EvidenceRecord
from change_agent.verifier_regions import attach_verifier_regions


SCHEMA_ID = "qwen3vl_p2_oracle_execution_manifest_v1"
CANDIDATE_INDEX_SCHEMA_ID = "qwen3vl_p2_oracle_candidates_v1"
COMPLETE_SCHEMA_ID = "qwen3vl_p2_candidate_generation_complete_v1"
FORBIDDEN_GT_KEYS = frozenset(
    {"gt", "gt_path", "ground_truth", "ground_truth_path", "label", "label_path", "label_cvt"}
)
FORBIDDEN_PATH_PARTS = frozenset({"label", "labels", "label_cvt", "ground_truth", "gt"})


@dataclass(frozen=True)
class LoadedSample:
    name: str
    t1_image: np.ndarray
    t2_image: np.ndarray
    t1_mask: np.ndarray
    t2_mask: np.ndarray
    query: str
    proposals: tuple[dict[str, Any], ...]
    initial_state: ChangeState


class NoQwenRejectVerifier:
    """A deterministic verifier stub used only to expose the pre-Qwen gate path."""

    def __init__(self) -> None:
        self.initial_call_count = 0
        self.candidate_call_count = 0
        self.last_evidence: dict[str, Any] = {}

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        del state, previous_score, previous_state
        if previous_action is None:
            self.initial_call_count += 1
            comparison = "initial"
            feedback = "P2 no-Qwen initial-state sentinel."
        else:
            self.candidate_call_count += 1
            comparison = "worse"
            feedback = "P2 no-Qwen sentinel rejects every candidate that passes hard gates."
        self.last_evidence = {
            "type": "p2_no_qwen_reject_verifier",
            "qwen_invoked": False,
            "comparison": comparison,
        }
        return VerifierOutput(
            quality_score=0.0,
            progress_score=0.0 if previous_action is None else -1.0,
            comparison=comparison,
            error_type="none",
            target_view="t2",
            suggested_action=None,
            feedback=feedback,
            accept=False,
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
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported non-JSON evidence value: {type(value).__name__}")


def assert_execution_manifest_is_gt_free(value: Any, location: str = "manifest") -> None:
    """Reject both explicit GT keys and paths that would let the generator probe GT."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in FORBIDDEN_GT_KEYS
                or normalized.startswith("label_")
                or normalized.startswith("ground_truth_")
                or normalized.startswith("gt_path")
            ):
                raise ValueError(f"GT reference is forbidden in execution manifest: {location}.{key}")
            assert_execution_manifest_is_gt_free(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_execution_manifest_is_gt_free(item, f"{location}[{index}]")
        return
    if isinstance(value, str):
        parts = {part.lower() for part in Path(value).parts}
        if parts & FORBIDDEN_PATH_PARTS or any(
            part.startswith(("label_", "ground_truth_", "gt_")) for part in parts
        ):
            raise ValueError(f"GT path is forbidden in execution manifest: {location}")


def _file_from_spec(spec: Any, *, field: str, require_hash: bool = True) -> Path:
    if not isinstance(spec, Mapping) or not isinstance(spec.get("path"), str):
        raise ValueError(f"{field} must be a file identity object")
    path = Path(spec["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} is not a file: {path}")
    expected = spec.get("sha256")
    if require_hash and not isinstance(expected, str):
        raise ValueError(f"{field}.sha256 is required")
    if expected is not None and sha256_file(path) != expected:
        raise ValueError(f"{field} SHA256 mismatch")
    return path


def _directory_from_spec(spec: Any, *, field: str) -> Path:
    raw = spec.get("path") if isinstance(spec, Mapping) else spec
    if not isinstance(raw, str):
        raise ValueError(f"{field} must contain a path")
    path = Path(raw).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{field} is not a directory: {path}")
    return path


def _load_array(spec: Mapping[str, Any], *, field: str, dtype: Any) -> np.ndarray:
    path = _file_from_spec(spec, field=field)
    value = np.asarray(np.load(path), dtype=dtype)
    expected_mask_hash = spec.get("mask_sha256")
    if expected_mask_hash is not None and mask_sha256(value) != expected_mask_hash:
        raise ValueError(f"{field} mask SHA256 mismatch")
    return value


def _load_image(spec: Mapping[str, Any], *, field: str) -> np.ndarray:
    path = _file_from_spec(spec, field=field)
    return np.asarray(Image.open(path).convert("RGB"))


def _proposal_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("proposal_config", {})
    if not isinstance(raw, Mapping):
        raise ValueError("proposal_config must be an object")
    result = {
        "max_regions": int(raw.get("max_regions", 1)),
        "max_delta_regions": int(raw.get("max_delta_regions", 1)),
        "min_component_area": int(raw.get("min_component_area", 4)),
        "padding_ratio": float(raw.get("padding_ratio", 0.25)),
        "empty_mask_scan_grid_size": int(raw.get("empty_mask_scan_grid_size", 3)),
    }
    return result


def _pair_processor(manifest: Mapping[str, Any]) -> MaskPairProcessor:
    raw = manifest.get("matching", {})
    if not isinstance(raw, Mapping):
        raise ValueError("matching must be an object")
    return MaskPairProcessor(
        overlap_threshold=float(raw.get("overlap_threshold", 0.25)),
        matching_mode=str(raw.get("mode", "overlap_presence")),
        t12_min_instance_area=int(raw.get("t12_min_instance_area", 0)),
        cd_min_instance_area=int(raw.get("cd_min_instance_area", 0)),
    )


def load_frozen_sample(
    name: str, spec: Mapping[str, Any], manifest: Mapping[str, Any]
) -> LoadedSample:
    t1_image = _load_image(spec["t1_image"], field=f"samples.{name}.t1_image")
    t2_image = _load_image(spec["t2_image"], field=f"samples.{name}.t2_image")
    t1_mask = _load_array(spec["t1_mask"], field=f"samples.{name}.t1_mask", dtype=bool)
    t2_mask = _load_array(spec["t2_mask"], field=f"samples.{name}.t2_mask", dtype=bool)
    processor = _pair_processor(manifest)
    update = processor.rebuild(t1_mask, t2_mask, {"initializer": "p1_cached_sam3"})
    state = ChangeState(
        t1_image=t1_image,
        t2_image=t2_image,
        query=str(manifest.get("query", "building")),
        t1_mask=t1_mask,
        t2_mask=t2_mask,
        change_mask=update.change_mask,
        t1_instances=update.t1_instances,
        t2_instances=update.t2_instances,
        matching=update.matching,
        evidence=update.evidence,
        step_index=0,
    )
    attach_verifier_regions(state, None, **_proposal_config(manifest))
    expected = spec.get("initial_state_mask_sha256")
    observed = {
        "t1": mask_sha256(state.t1_mask),
        "t2": mask_sha256(state.t2_mask),
        "change": mask_sha256(state.change_mask),
    }
    if expected != observed:
        raise ValueError(f"samples.{name} frozen initial state hashes mismatch")
    proposals = tuple(dict(item) for item in state.evidence["verifier_region_proposals"])
    expected_proposals = spec.get("proposals")
    if not isinstance(expected_proposals, list) or canonical_sha256(proposals) != canonical_sha256(expected_proposals):
        raise ValueError(f"samples.{name} proposal catalog mismatch")
    return LoadedSample(
        name=name,
        t1_image=t1_image,
        t2_image=t2_image,
        t1_mask=t1_mask,
        t2_mask=t2_mask,
        query=state.query,
        proposals=proposals,
        initial_state=state,
    )


def enumerate_oracle_actions(
    sample: str, proposals: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return 27 planner-primary plus 9 executor-only diagnostic actions."""

    primary: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    for proposal in proposals:
        record = EvidenceRecord.from_proposal(proposal)
        if record.audit_kind == "scan":
            pixel_box = proposal.get("box_pixels")
            if not isinstance(pixel_box, list) or len(pixel_box) != 4:
                raise ValueError(f"scan proposal {record.region_id} lacks pixel box")
            for target_view in ("t1", "t2"):
                primary.append(
                    _action_record(
                        sample,
                        record.region_id,
                        "primary",
                        "box",
                        target_view,
                        box=tuple(int(value) for value in pixel_box),
                    )
                )
            continue
        if record.audit_kind != "present":
            continue
        pixel_seed = proposal.get("component_seed_pixels")
        if not isinstance(pixel_seed, list) or len(pixel_seed) != 2:
            raise ValueError(f"present proposal {record.region_id} lacks pixel seed")
        seed = tuple(int(value) for value in pixel_seed)
        white_views = [view for view in ("t1", "t2") if record.editable_seed_white.get(view)]
        black_views = [view for view in ("t1", "t2") if not record.editable_seed_white.get(view)]
        if len(white_views) != 1 or len(black_views) != 1:
            raise ValueError(
                f"present proposal {record.region_id} is not a one-sided planner-reachable component"
            )
        primary.append(
            _action_record(
                sample,
                record.region_id,
                "primary",
                "negative_point",
                white_views[0],
                coordinate=seed,
            )
        )
        diagnostic.append(
            _action_record(
                sample,
                record.region_id,
                "diagnostic_executor_only",
                "positive_point",
                black_views[0],
                coordinate=seed,
            )
        )
    return primary + diagnostic


def _action_record(
    sample: str,
    region_id: str,
    tier: str,
    action: str,
    target_view: str,
    *,
    coordinate: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    geometry = list(coordinate if coordinate is not None else box or ())
    candidate_id = "--".join((sample, tier, region_id, action, target_view))
    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "sample": sample,
        "tier": tier,
        "region_id": region_id,
        "action": action,
        "target_view": target_view,
        "geometry_pixels": geometry,
    }
    if coordinate is not None:
        result["coordinate"] = list(coordinate)
    if box is not None:
        result["box"] = list(box)
    return result


def _agent_action(record: Mapping[str, Any]) -> AgentAction:
    coordinate = record.get("coordinate")
    box = record.get("box")
    return AgentAction(
        str(record["target_view"]),
        str(record["action"]),
        coordinate=tuple(int(value) for value in coordinate) if coordinate is not None else None,
        box=tuple(int(value) for value in box) if box is not None else None,
    )


def _gate_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("gate_config", {})
    if not isinstance(raw, Mapping):
        raise ValueError("gate_config must be an object")
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


def execute_one_candidate(
    sample: LoadedSample,
    action_record: Mapping[str, Any],
    point_backend: Any,
    box_backend: Any,
    manifest: Mapping[str, Any],
) -> tuple[Any, NoQwenRejectVerifier]:
    """Execute one candidate in a fresh real Environment from cached masks."""

    processor = _pair_processor(manifest)

    def cached_initializer(
        t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del t1_image, t2_image, query
        return (
            np.array(sample.t1_mask, copy=True),
            np.array(sample.t2_mask, copy=True),
            {"initializer": "p1_cached_sam3", "gt_available": False},
        )

    backend = OmniOVCDAdapter(cached_initializer, box_backend.segment_box, processor)
    verifier = NoQwenRejectVerifier()
    environment = ChangeAgentEnvironment(
        backend,
        ActionExecutor(
            point_backend,
            box_backend,
            authorized_negative_point_input_scope=str(
                manifest.get("authorized_negative_point_input_scope", "full_session")
            ),
        ),
        verifier,
        max_steps=1,
        require_tool_before_finish=False,
        verifier_max_regions=_proposal_config(manifest)["max_regions"],
        verifier_max_delta_regions=_proposal_config(manifest)["max_delta_regions"],
        verifier_min_region_area=_proposal_config(manifest)["min_component_area"],
        verifier_region_padding_ratio=_proposal_config(manifest)["padding_ratio"],
        empty_mask_scan_grid_size=_proposal_config(manifest)["empty_mask_scan_grid_size"],
        enable_verifier_regions=True,
        **_gate_config(manifest),
    )
    environment.reset(sample.t1_image, sample.t2_image, sample.query)
    observed_initial = environment.trajectory.entries[0].state
    for observed, expected in (
        (observed_initial.t1_mask, sample.t1_mask),
        (observed_initial.t2_mask, sample.t2_mask),
        (observed_initial.change_mask, sample.initial_state.change_mask),
    ):
        if not np.array_equal(observed, expected):
            raise RuntimeError("candidate did not start from the exact frozen initial state")
    environment.step(_agent_action(action_record), authorized_region_id=str(action_record["region_id"]))
    return environment.trajectory.entries[-1], verifier


def _save_mask(path: Path, mask: np.ndarray, root: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=np.uint8))
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "mask_sha256": mask_sha256(mask),
        "shape": list(mask.shape),
    }


def _tool_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    tools = manifest.get("tools")
    if not isinstance(tools, Mapping):
        raise ValueError("tools must be an object")
    result = {
        key: _file_from_spec(tools[key], field=f"tools.{key}")
        for key in (
            "segagent_python",
            "omniovcd_python",
            "simpleclick_checkpoint",
            "sam3_checkpoint",
            "sam3_bpe",
        )
    }
    result["workspace_root"] = _directory_from_spec(tools["workspace_root"], field="tools.workspace_root")
    return result


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_id") != SCHEMA_ID or manifest.get("status") != "frozen":
        raise ValueError("execution manifest must be the frozen P2 v1 schema")
    assert_execution_manifest_is_gt_free(manifest)
    preregistration_hash = manifest.get("preregistration_canonical_sha256")
    if not isinstance(preregistration_hash, str) or len(preregistration_hash) != 64:
        raise ValueError("preregistration canonical SHA256 binding is required")
    expected_self = manifest.get("action_generator_sha256")
    if not isinstance(expected_self, str) or sha256_file(Path(__file__).resolve()) != expected_self:
        raise ValueError("action generator SHA256 mismatch")
    sources = manifest.get("runtime_source_files", [])
    if not isinstance(sources, list):
        raise ValueError("runtime_source_files must be a list")
    for index, source in enumerate(sources):
        _file_from_spec(source, field=f"runtime_source_files[{index}]")
    ordered = manifest.get("ordered_samples")
    samples = manifest.get("samples")
    if not isinstance(ordered, list) or not isinstance(samples, Mapping):
        raise ValueError("ordered_samples and samples are required")
    if ordered != list(samples) or len(ordered) != len(set(ordered)):
        raise ValueError("sample mapping must preserve the unique frozen order")
    if manifest.get("expected_candidate_counts") != {
        "total": 36,
        "primary": 27,
        "diagnostic_executor_only": 9,
    }:
        raise ValueError("P2 v1 freezes exactly 27 primary plus 9 diagnostic candidates")


def _prepare_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"output must be absent or an empty directory: {path}")
    else:
        path.mkdir(parents=True)
    return path


def validate_runtime_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("runtime commit must be a full 40-character Git SHA")
    return value.lower()


def run(manifest_path: Path, output: Path, runtime_commit: str) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    runtime_commit = validate_runtime_commit(runtime_commit)
    output = _prepare_output(output)
    tools = _tool_paths(manifest)
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tools["workspace_root"]
    seed = int(manifest.get("seed", 42))
    resolution = int(manifest.get("sam3_resolution", 1008))
    timeout = int(manifest.get("tool_timeout_seconds", 900))

    loaded: dict[str, LoadedSample] = {}
    all_actions: list[dict[str, Any]] = []
    initial_index: dict[str, Any] = {}
    for name in manifest["ordered_samples"]:
        sample = load_frozen_sample(name, manifest["samples"][name], manifest)
        loaded[name] = sample
        actions = enumerate_oracle_actions(name, sample.proposals)
        expected_actions = manifest["samples"][name].get("actions")
        if expected_actions != actions:
            raise ValueError(f"samples.{name} frozen action inventory mismatch")
        all_actions.extend(actions)
        base = output / "initial_states" / name
        initial_index[name] = {
            "t1_mask": _save_mask(base / "t1_mask.npy", sample.t1_mask, output),
            "t2_mask": _save_mask(base / "t2_mask.npy", sample.t2_mask, output),
            "change_mask": _save_mask(
                base / "change_mask.npy", sample.initial_state.change_mask, output
            ),
        }
    primary_count = sum(item["tier"] == "primary" for item in all_actions)
    diagnostic_count = sum(item["tier"] == "diagnostic_executor_only" for item in all_actions)
    expected_counts = manifest.get("expected_candidate_counts")
    observed_counts = {
        "total": len(all_actions),
        "primary": primary_count,
        "diagnostic_executor_only": diagnostic_count,
    }
    if expected_counts != observed_counts:
        raise ValueError(f"frozen candidate counts mismatch: {observed_counts}")

    candidate_records: list[dict[str, Any]] = []
    for action_record in all_actions:
        sample = loaded[action_record["sample"]]
        candidate_dir = output / "candidates" / action_record["candidate_id"]
        tool_root = candidate_dir / "tool_runs"
        point_backend = SubprocessPointBackend(
            tools["segagent_python"],
            repo_root / "tools" / "seeded_segmentation_worker.py",
            tool_root,
            checkpoint=tools["simpleclick_checkpoint"],
            pythonpath=(
                repo_root,
                workspace / "reference" / "SegAgent",
                workspace / "reference" / "SegAgent" / "third_party" / "SimpleClick",
            ),
            device=str(manifest.get("tool_device", "cuda")),
            timeout_seconds=timeout,
            seed=seed,
        )
        box_backend = SubprocessBoxBackend(
            tools["omniovcd_python"],
            repo_root / "tools" / "seeded_segmentation_worker.py",
            tool_root,
            checkpoint=tools["sam3_checkpoint"],
            bpe=tools["sam3_bpe"],
            resolution=resolution,
            pythonpath=(repo_root, workspace / "reference" / "OmniOVCD"),
            device=str(manifest.get("tool_device", "cuda")),
            timeout_seconds=timeout,
            seed=seed,
        )
        entry, verifier = execute_one_candidate(
            sample, action_record, point_backend, box_backend, manifest
        )
        masks_dir = candidate_dir / "masks"
        hard_gate_rejected = bool(entry.execution.get("verifier_skipped_by_hard_gate"))
        gate_evidence = entry.execution.get("verifier_evidence", {})
        hard_reasons = (
            list(gate_evidence.get("rejection_reasons", []))
            if hard_gate_rejected and isinstance(gate_evidence, Mapping)
            else []
        )
        candidate_masks = {
            "t1": _save_mask(masks_dir / "t1.npy", entry.state.t1_mask, output),
            "t2": _save_mask(masks_dir / "t2.npy", entry.state.t2_mask, output),
            "change": _save_mask(masks_dir / "change.npy", entry.state.change_mask, output),
        }
        candidate_state_mask_sha256 = {
            name: candidate_masks[name]["mask_sha256"]
            for name in ("t1", "t2", "change")
        }
        record = {
            **action_record,
            "initial_state_mask_sha256": manifest["samples"][sample.name][
                "initial_state_mask_sha256"
            ],
            "candidate_masks": candidate_masks,
            "candidate_state_mask_sha256": candidate_state_mask_sha256,
            "hard_gate": {
                "passed": not hard_gate_rejected,
                "rejected": hard_gate_rejected,
                "rejection_reasons": hard_reasons,
                "verifier_skipped_by_hard_gate": hard_gate_rejected,
            },
            "no_qwen_audit": {
                "qwen_invoked": False,
                "initial_stub_calls": verifier.initial_call_count,
                "candidate_stub_calls": verifier.candidate_call_count,
                "decision_path": (
                    "pre_qwen_hard_gate"
                    if hard_gate_rejected
                    else "eligible_for_qwen_not_evaluated"
                ),
            },
            "tool_evidence": _json_safe(entry.execution),
        }
        record_path = candidate_dir / "candidate.json"
        _write_json(record_path, record)
        candidate_records.append(
            {
                "candidate_id": record["candidate_id"],
                "sample": record["sample"],
                "tier": record["tier"],
                "candidate_state_mask_sha256": candidate_state_mask_sha256,
                "record_path": str(record_path.relative_to(output)),
                "record_sha256": sha256_file(record_path),
            }
        )

    index = {
        "schema_id": CANDIDATE_INDEX_SCHEMA_ID,
        "execution_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "canonical_sha256": canonical_sha256(manifest),
        },
        "preregistration_canonical_sha256": manifest.get(
            "preregistration_canonical_sha256"
        ),
        "source_freeze_commit": manifest.get("source_freeze_commit"),
        "p2_runtime_commit": runtime_commit,
        "gt_loaded": False,
        "ordered_samples": list(manifest["ordered_samples"]),
        "counts": observed_counts,
        "initial_states": initial_index,
        "candidates": candidate_records,
    }
    index_path = output / "candidates.json"
    _write_json(index_path, index)
    marker = {
        "schema_id": COMPLETE_SCHEMA_ID,
        "execution_status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "gt_loaded": False,
        "ordered_samples": list(manifest["ordered_samples"]),
        "counts": observed_counts,
        "p2_runtime_commit": runtime_commit,
        "execution_manifest": {
            "sha256": sha256_file(manifest_path),
        },
        "preregistration_canonical_sha256": manifest.get(
            "preregistration_canonical_sha256"
        ),
        "candidate_index": {
            "path": str(index_path.relative_to(output)),
            "sha256": sha256_file(index_path),
        },
    }
    _write_json(output / "candidate_generation_complete.json", marker)
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker = run(args.execution_manifest, args.output, args.runtime_commit)
    print(
        "status=complete;gt_loaded=false;"
        f"candidates={marker['counts']['total']};"
        f"primary={marker['counts']['primary']};"
        f"diagnostic={marker['counts']['diagnostic_executor_only']}"
    )


if __name__ == "__main__":
    main()
