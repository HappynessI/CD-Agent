#!/usr/bin/env python3
"""Export the image inputs assembled for staged Verifier calls.

This script reads an existing run. It does not load a model or call a provider.
For each sample it exports the global proposal overview, every local proposal
crop, the binary mask crops, candidate deltas, and a JSON manifest describing
what each exported image represents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from change_agent.adapters.stage_backends import (
    _as_image,
    _delta_only_contour,
    _mask_image,
    _normalized_crop_box,
    _proposal_overview,
)
from change_agent.state import ChangeState
from change_agent.verifier_regions import attach_verifier_regions


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Visualize staged Verifier image inputs from a completed run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/visualization",
        help="Output directory; existing files with the same names are replaced.",
    )
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument(
        "--initial-only",
        action="store_true",
        help="Export only the initial Verifier input. Useful for a first presentation.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_from_init(run_dir: Path, sample: str, view: str) -> Image.Image:
    path = run_dir / "tool_runs" / sample / "sam3_initialization" / "initialize_000" / f"{view}_image.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing initialization image: {path}")
    return Image.open(path).convert("RGB")


def _initial_masks(run_dir: Path, sample: str) -> tuple[np.ndarray, np.ndarray]:
    base = run_dir / "tool_runs" / sample / "sam3_initialization" / "initialize_000"
    paths = (base / "t1_mask.npy", base / "t2_mask.npy")
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("missing SAM3 initialization masks under " + str(base))
    return tuple(np.asarray(np.load(path), dtype=bool) for path in paths)  # type: ignore[return-value]


def _change_mask(run_dir: Path, sample: str, step: int) -> np.ndarray:
    path = run_dir / "masks" / sample / f"step_{step:03d}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing change mask: {path}")
    return np.asarray(np.load(path), dtype=bool)


def _normalized_alias(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Environment proposal names to stage-backend overview names."""

    result = dict(proposal)
    result["box_normalized_1000"] = list(
        proposal.get("box_normalized_1000", proposal.get("box_normalized", ()))
    )
    result["component_seed_normalized_1000"] = list(
        proposal.get(
            "component_seed_normalized_1000",
            proposal.get("component_seed_normalized", ()),
        )
    )
    if "effect_kind" in result and "audit_kind" not in result:
        result["audit_kind"] = f"delta_{result['effect_kind']}"
    return result


def _state(
    t1_image: Image.Image,
    t2_image: Image.Image,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    change_mask: np.ndarray,
    step: int,
    query: str,
) -> ChangeState:
    return ChangeState(
        t1_image=np.asarray(t1_image),
        t2_image=np.asarray(t2_image),
        query=query,
        t1_mask=t1_mask,
        t2_mask=t2_mask,
        change_mask=change_mask,
        step_index=step,
    )


def _point_roi(coordinate: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    radius = max(2, round(min(width, height) * 0.125))
    x, y = coordinate
    return max(0, x - radius), max(0, y - radius), min(width - 1, x + radius), min(height - 1, y + radius)


def _component_containing(mask: np.ndarray, coordinate: tuple[int, int]) -> np.ndarray:
    x, y = coordinate
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
        return np.zeros_like(mask, dtype=bool)
    result = np.zeros_like(mask, dtype=bool)
    stack = [(x, y)]
    result[y, x] = True
    while stack:
        current_x, current_y = stack.pop()
        for next_x, next_y in (
            (current_x - 1, current_y),
            (current_x + 1, current_y),
            (current_x, current_y - 1),
            (current_x, current_y + 1),
        ):
            if (
                0 <= next_x < mask.shape[1]
                and 0 <= next_y < mask.shape[0]
                and mask[next_y, next_x]
                and not result[next_y, next_x]
            ):
                result[next_y, next_x] = True
                stack.append((next_x, next_y))
    return result


def _compose_tool_mask(
    before: np.ndarray,
    action: Mapping[str, Any],
    raw_mask: np.ndarray,
) -> np.ndarray:
    """Reproduce Environment's local point/box composition for visualization."""

    result = np.asarray(before, dtype=bool).copy()
    raw = np.asarray(raw_mask, dtype=bool)
    action_name = action.get("action")
    if action_name == "positive_point":
        coordinate = tuple(int(value) for value in action["coordinate"])
        return np.logical_or(result, _component_containing(raw, coordinate))
    if action_name == "negative_point":
        coordinate = tuple(int(value) for value in action["coordinate"])
        x1, y1, x2, y2 = _point_roi(coordinate, before.shape)
        roi = np.zeros_like(result)
        roi[y1 : y2 + 1, x1 : x2 + 1] = True
        return np.logical_and(result, ~np.logical_and(result, np.logical_and(~raw, roi)))
    if action_name == "box":
        x1, y1, x2, y2 = (int(value) for value in action["box"])
        result[y1 : y2 + 1, x1 : x2 + 1] = raw[y1 : y2 + 1, x1 : x2 + 1]
        return result
    return result


def _reconstruct_candidate_masks(
    run_dir: Path,
    sample: str,
    step: int,
    accepted_t1: np.ndarray,
    accepted_t2: np.ndarray,
    entry: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    action = entry.get("parsed_action")
    if not isinstance(action, Mapping):
        return accepted_t1.copy(), accepted_t2.copy()
    tool = entry.get("tool")
    if not tool:
        return accepted_t1.copy(), accepted_t2.copy()
    tool_dir = run_dir / "tool_runs" / sample / f"point_{step - 1:03d}"
    if tool == "sam3":
        tool_dir = run_dir / "tool_runs" / sample / f"box_{step - 1:03d}"
    output_path = tool_dir / "output_mask.npy"
    if not output_path.is_file():
        return accepted_t1.copy(), accepted_t2.copy()
    raw = np.asarray(np.load(output_path), dtype=bool)
    target = accepted_t1 if action.get("target_view") == "t1" else accepted_t2
    composed = _compose_tool_mask(target, action, raw)
    if action.get("target_view") == "t1":
        return composed, accepted_t2.copy()
    return accepted_t1.copy(), composed


def _contact_sheet(images: Iterable[tuple[str, Image.Image]], columns: int = 2) -> Image.Image:
    items = list(images)
    if not items:
        return Image.new("RGB", (32, 32), "black")
    thumb_w = max(image.width for _, image in items)
    thumb_h = max(image.height for _, image in items) + 22
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        x = (index % columns) * thumb_w
        y = (index // columns) * thumb_h
        rgb = image.convert("RGB")
        sheet.paste(rgb.resize((thumb_w, thumb_h - 22)), (x, y + 22))
        draw.text((x + 3, y + 3), label, fill="white")
    return sheet


def _save_stage_inputs(
    output_dir: Path,
    state: ChangeState,
    previous_state: ChangeState | None,
    proposals: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = _proposal_overview(state, previous_state, proposals)
    overview.save(output_dir / "00_global_overview.png")
    manifest: list[dict[str, Any]] = []
    for proposal in proposals:
        region_id = str(proposal["region_id"])
        region_dir = output_dir / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        box = tuple(int(value) for value in proposal["box_normalized_1000"])
        crop_box = _normalized_crop_box(box, state.image_size)
        if previous_state is None:
            images: list[tuple[str, Image.Image]] = [
                ("T1 RGB crop", _as_image(state.t1_image).crop(crop_box)),
                ("T2 RGB crop", _as_image(state.t2_image).crop(crop_box)),
                ("T1 object mask", _mask_image(state.t1_mask).crop(crop_box)),
                ("T2 object mask", _mask_image(state.t2_mask).crop(crop_box)),
                ("change mask", _mask_image(state.change_mask).crop(crop_box)),
            ]
        else:
            images = [
                (
                    "T1 delta-only original RGB with cyan contour",
                    _delta_only_contour(
                        state.t1_image,
                        np.logical_or(
                            np.logical_and(state.change_mask, ~previous_state.change_mask),
                            np.logical_and(previous_state.change_mask, ~state.change_mask),
                        ),
                    ).crop(crop_box),
                ),
                (
                    "T2 delta-only original RGB with cyan contour",
                    _delta_only_contour(
                        state.t2_image,
                        np.logical_or(
                            np.logical_and(state.change_mask, ~previous_state.change_mask),
                            np.logical_and(previous_state.change_mask, ~state.change_mask),
                        ),
                    ).crop(crop_box),
                ),
                ("T1 object mask", _mask_image(state.t1_mask).crop(crop_box)),
                ("T2 object mask", _mask_image(state.t2_mask).crop(crop_box)),
                ("change mask", _mask_image(state.change_mask).crop(crop_box)),
            ]
        if previous_state is not None:
            added = np.logical_and(state.change_mask, ~previous_state.change_mask)
            removed = np.logical_and(previous_state.change_mask, ~state.change_mask)
            delta = np.logical_or(added, removed)
            images.extend(
                [
                    ("previous T1 mask", _mask_image(previous_state.t1_mask).crop(crop_box)),
                    ("previous T2 mask", _mask_image(previous_state.t2_mask).crop(crop_box)),
                    ("previous change mask", _mask_image(previous_state.change_mask).crop(crop_box)),
                    ("candidate added", _mask_image(added).crop(crop_box)),
                    ("candidate removed", _mask_image(removed).crop(crop_box)),
                ]
            )
        for label, image in images:
            filename = label.lower().replace(" ", "_") + ".png"
            image.save(region_dir / filename)
        _contact_sheet(images).save(region_dir / "contact_sheet.png")
        manifest.append(
            {
                "region_id": region_id,
                "selected_for_local_inspection": region_id in selected_ids,
                "proposal": proposal,
                "crop_box_pixels_pil": list(crop_box),
                "files": sorted(str(path.relative_to(output_dir)) for path in region_dir.glob("*.png")),
            }
        )
    return manifest


def _sample_names(run_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [Path(name).stem for name in requested]
    return sorted(path.name for path in (run_dir / "trajectories").iterdir() if path.is_dir())


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_samples = _sample_names(run_dir, args.samples)
    run_manifest: dict[str, Any] = {
        "run_dir": str(run_dir),
        "output_dir": str(output_root),
        "samples": selected_samples,
        "notes": [
            "PNG files are reconstructed staged Verifier visual inputs; no model call is made.",
            "00_global_overview is selection-stage input; each region/contact_sheet is local evidence input.",
        ],
    }
    for sample in selected_samples:
        trajectory_path = run_dir / "trajectories" / sample / "trajectory.json"
        trajectory = _load_json(trajectory_path)
        metadata = trajectory.get("metadata", {})
        query = str(metadata.get("query", "building"))
        initial_batch_size = int(metadata.get("verifier_max_regions", 6))
        delta_batch_size = int(metadata.get("verifier_max_delta_regions_per_batch", metadata.get("verifier_max_delta_regions", 3)))
        min_component_area = int(metadata.get("verifier_min_region_area", 4))
        padding_ratio = float(metadata.get("verifier_region_padding_ratio", 0.25))
        t1_image = _image_from_init(run_dir, sample, "t1")
        t2_image = _image_from_init(run_dir, sample, "t2")
        accepted_t1, accepted_t2 = _initial_masks(run_dir, sample)
        sample_root = output_root / sample
        initial_change = _change_mask(run_dir, sample, 0)
        initial_state = _state(t1_image, t2_image, accepted_t1, accepted_t2, initial_change, 0, query)
        accepted_change = initial_change.copy()
        initial_props = [
            _normalized_alias(item)
            for item in attach_verifier_regions(initial_state, None, max_regions=initial_batch_size, max_delta_regions=delta_batch_size, min_component_area=min_component_area, padding_ratio=padding_ratio)
        ]
        initial_trace = trajectory["steps"][0].get("execution", {}).get("verifier_evidence", {}).get("stage_trace", {})
        selected = {
            str(value)
            for value in initial_trace.get(
                "audited_region_ids",
                initial_trace.get("selected_region_ids", []),
            )
        }
        sample_manifest: dict[str, Any] = {
            "initial": _save_stage_inputs(sample_root / "step_000_initial", initial_state, None, initial_props, selected),
            "steps": [],
        }
        if not args.initial_only:
            for entry in trajectory.get("steps", [])[1:]:
                step = int(entry["step_index"])
                candidate_t1, candidate_t2 = _reconstruct_candidate_masks(
                    run_dir, sample, step, accepted_t1, accepted_t2, entry
                )
                candidate_change = _change_mask(run_dir, sample, step)
                previous_state = _state(t1_image, t2_image, accepted_t1, accepted_t2, accepted_change, step - 1, query)
                candidate_state = _state(t1_image, t2_image, candidate_t1, candidate_t2, candidate_change, step, query)
                props = [
                    _normalized_alias(item)
                    for item in attach_verifier_regions(candidate_state, previous_state, max_regions=delta_batch_size, max_delta_regions=delta_batch_size, min_component_area=min_component_area, padding_ratio=padding_ratio)
                ]
                trace = entry.get("execution", {}).get("verifier_evidence", {}).get("stage_trace", {})
                selected = {
                    str(value)
                    for value in trace.get(
                        "audited_region_ids",
                        trace.get("selected_region_ids", []),
                    )
                }
                step_dir = sample_root / f"step_{step:03d}_candidate"
                records = _save_stage_inputs(step_dir, candidate_state, previous_state, props, selected)
                sample_manifest["steps"].append({"step": step, "records": records})
                if bool(entry.get("candidate_accepted")):
                    accepted_t1, accepted_t2, accepted_change = (
                        candidate_t1,
                        candidate_t2,
                        candidate_change,
                    )
        (sample_root / "manifest.json").write_text(json.dumps(sample_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        run_manifest.setdefault("sample_manifests", {})[sample] = str((sample_root / "manifest.json").relative_to(output_root))
    (output_root / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
