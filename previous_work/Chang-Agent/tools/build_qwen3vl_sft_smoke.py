#!/usr/bin/env python3
"""Build the deterministic 32-example Qwen3-VL select-stage SFT smoke set.

This is an offline-only data builder.  It reads LEVIR-CD train labels to select
balanced patches and to synthesize a deliberately incomplete/over-complete
change mask. The raw GT label image is not supplied, but GT-derived candidate
masks, proposal geometry, and mask facts are intentionally model-visible in
this overfit pipeline smoke. The resulting target is the current select-stage
JSON envelope, not a free coordinate, action, or target-view label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from change_agent.adapters.stage_backends import _proposal_overview, _stage_prompt
from change_agent.state import ChangeState


TRAIN_DATASET_NAME = "levir_qwen3vl_sft32_train"
EVAL_DATASET_NAME = "levir_qwen3vl_sft32_eval"
DATASET_NAMES = (TRAIN_DATASET_NAME, EVAL_DATASET_NAME)
SCHEMA_VERSION = "levir_select_sft_smoke_v1"
DEFAULT_DATASET_ROOT = Path(
    "/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD"
)
DEFAULT_OUTPUT_DIR = Path(
    "/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/"
    "CA_0729(1)-qwen3vl-sft32-smoke/data"
)
PATCH_SIZE = 256
GRID_SIZE = 4
SAMPLES_PER_CLASS = 16
TOTAL_SAMPLES = SAMPLES_PER_CLASS * 2
PROPOSALS_PER_SAMPLE = 4
MIN_POSITIVE_PIXELS = 64


@dataclass(frozen=True)
class PatchRecord:
    parent_id: str
    parent_index: int
    row: int
    column: int
    gt_pixels: int
    t1_path: Path
    t2_path: Path
    label_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.parent_id}_r{self.row}c{self.column}"

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        left = self.column * PATCH_SIZE
        top = self.row * PATCH_SIZE
        return left, top, left + PATCH_SIZE, top + PATCH_SIZE


@dataclass(frozen=True)
class Component:
    mask: np.ndarray
    area: int
    box: tuple[int, int, int, int]
    seed: tuple[int, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=729)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing output directory without reading LEVIR-CD",
    )
    return parser.parse_args()


def _numeric_suffix(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"unexpected LEVIR-CD filename: {path.name}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bool(path: Path, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").crop(crop_box), dtype=np.uint8) > 0


def _load_rgb(path: Path, crop_box: tuple[int, int, int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").crop(crop_box)


def scan_train_patches(dataset_root: Path) -> list[PatchRecord]:
    split_root = dataset_root / "train_original"
    labels = sorted((split_root / "label").glob("*.png"), key=_numeric_suffix)
    if len(labels) != 445:
        raise ValueError(
            f"expected 445 LEVIR-CD train labels, found {len(labels)} under {split_root}"
        )
    records: list[PatchRecord] = []
    for label_path in labels:
        parent_index = _numeric_suffix(label_path)
        parent_id = f"train_{parent_index}"
        t1_path = split_root / "A" / label_path.name
        t2_path = split_root / "B" / label_path.name
        if not t1_path.is_file() or not t2_path.is_file():
            raise FileNotFoundError(f"missing paired image for {label_path.name}")
        with Image.open(label_path) as image:
            if image.size != (PATCH_SIZE * GRID_SIZE, PATCH_SIZE * GRID_SIZE):
                raise ValueError(f"unexpected label size for {label_path}: {image.size}")
            label = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                crop = label[
                    row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                    column * PATCH_SIZE : (column + 1) * PATCH_SIZE,
                ]
                records.append(
                    PatchRecord(
                        parent_id=parent_id,
                        parent_index=parent_index,
                        row=row,
                        column=column,
                        gt_pixels=int(crop.sum()),
                        t1_path=t1_path,
                        t2_path=t2_path,
                        label_path=label_path,
                    )
                )
    return records


def _stratified_pick(
    pool: Iterable[PatchRecord], count: int, *, seed: int, excluded_parents: set[str]
) -> list[PatchRecord]:
    """Pick centered area quantiles with a deterministic seeded tie break."""

    rng = np.random.default_rng(seed)
    keyed = [
        (record, float(rng.random()))
        for record in pool
        if record.parent_id not in excluded_parents
    ]
    keyed.sort(
        key=lambda item: (
            item[0].gt_pixels,
            item[1],
            item[0].parent_index,
            item[0].row,
            item[0].column,
        )
    )
    if len(keyed) < count:
        raise ValueError(f"cannot choose {count} records from pool of {len(keyed)}")
    targets = [round((index + 0.5) * len(keyed) / count - 0.5) for index in range(count)]
    selected: list[PatchRecord] = []
    used = set(excluded_parents)
    available = set(range(len(keyed)))
    for target in targets:
        candidates = sorted(
            available,
            key=lambda index: (
                keyed[index][0].parent_id in used,
                abs(index - target),
                keyed[index][1],
            ),
        )
        chosen_index = next(
            (index for index in candidates if keyed[index][0].parent_id not in used),
            None,
        )
        if chosen_index is None:
            raise ValueError("not enough distinct parent scenes for deterministic selection")
        record = keyed[chosen_index][0]
        selected.append(record)
        used.add(record.parent_id)
        available.remove(chosen_index)
    return selected


def choose_smoke_records(
    records: list[PatchRecord], *, seed: int
) -> tuple[list[PatchRecord], list[PatchRecord]]:
    empty = [record for record in records if record.gt_pixels == 0]
    positive = [
        record for record in records if record.gt_pixels >= MIN_POSITIVE_PIXELS
    ]
    selected_positive = _stratified_pick(
        positive, SAMPLES_PER_CLASS, seed=seed, excluded_parents=set()
    )
    selected_empty = _stratified_pick(
        empty,
        SAMPLES_PER_CLASS,
        seed=seed + 1,
        excluded_parents={record.parent_id for record in selected_positive},
    )
    return selected_positive, selected_empty


def _fp_mask_for_empty(sample_id: str) -> np.ndarray:
    """Create a stable, obvious compact false-positive component without GT input."""

    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    width = 36 + digest[0] % 29
    height = 36 + digest[1] % 29
    left = 16 + digest[2] % (PATCH_SIZE - width - 32)
    top = 16 + digest[3] % (PATCH_SIZE - height - 32)
    yy, xx = np.ogrid[:PATCH_SIZE, :PATCH_SIZE]
    center_x = left + width / 2
    center_y = top + height / 2
    ellipse = (
        ((xx - center_x) / max(1.0, width / 2)) ** 2
        + ((yy - center_y) / max(1.0, height / 2)) ** 2
        <= 1.0
    )
    return np.asarray(ellipse, dtype=bool)


def _components(mask: np.ndarray) -> list[Component]:
    """Return 8-connected components using a compact dependency-free flood fill."""

    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    visited = np.zeros_like(source, dtype=bool)
    result: list[Component] = []
    for start_y, start_x in zip(*np.nonzero(source)):
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
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and source[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        ys = np.asarray([item[0] for item in pixels], dtype=np.int32)
        xs = np.asarray([item[1] for item in pixels], dtype=np.int32)
        component_mask = np.zeros_like(source, dtype=bool)
        component_mask[ys, xs] = True
        seed_offset = len(pixels) // 2
        result.append(
            Component(
                mask=component_mask,
                area=len(pixels),
                box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                seed=(int(xs[seed_offset]), int(ys[seed_offset])),
            )
        )
    result.sort(key=lambda item: (-item.area, item.box))
    return result


def _connected_chunk(component: Component, sample_id: str) -> np.ndarray:
    """Select one deterministic connected subregion while retaining other GT pixels."""

    source = np.asarray(component.mask, dtype=bool)
    ys, xs = np.nonzero(source)
    if not len(xs):
        raise ValueError("cannot remove a chunk from an empty component")
    digest = hashlib.sha256((sample_id + ":fn").encode("utf-8")).digest()
    start_index = int.from_bytes(digest[:4], "big") % len(xs)
    start = (int(ys[start_index]), int(xs[start_index]))
    desired = min(max(16, component.area // 3), 1024)
    if component.area > 1:
        desired = min(desired, component.area - 1)
    desired = max(1, desired)
    visited = np.zeros_like(source, dtype=bool)
    visited[start] = True
    queue = [start]
    pixels: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(queue) and len(pixels) < desired:
        y, x = queue[cursor]
        cursor += 1
        pixels.append((y, x))
        neighbours: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < source.shape[0]
                    and 0 <= nx < source.shape[1]
                    and source[ny, nx]
                    and not visited[ny, nx]
                ):
                    neighbours.append((ny, nx))
        neighbours.sort(
            key=lambda item: hashlib.sha256(
                f"{sample_id}:{item[0]}:{item[1]}".encode("utf-8")
            ).digest()
        )
        for item in neighbours:
            visited[item] = True
            queue.append(item)
    chunk = np.zeros_like(source, dtype=bool)
    chunk[tuple(np.asarray(pixels, dtype=np.int32).T)] = True
    return chunk


def _normalized_point(point: tuple[int, int]) -> list[int]:
    return [round(value / (PATCH_SIZE - 1) * 1000) for value in point]


def _normalized_box(box: tuple[int, int, int, int]) -> list[int]:
    return [round(value / (PATCH_SIZE - 1) * 1000) for value in box]


def _component_geometry(mask: np.ndarray, *, padding: int = 6) -> tuple[
    tuple[int, int, int, int], tuple[int, int], int
]:
    components = _components(mask)
    if not components:
        raise ValueError("proposal geometry mask is empty")
    component = components[0]
    x1, y1, x2, y2 = component.box
    box = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(PATCH_SIZE - 1, x2 + padding),
        min(PATCH_SIZE - 1, y2 + padding),
    )
    return box, component.seed, component.area


def _decoy_boxes(target_mask: np.ndarray, sample_id: str) -> list[tuple[int, int, int, int]]:
    """Choose three non-overlapping scan boxes away from the supervised error."""

    side = 48
    anchors = (8, 72, 136, 200)
    candidates: list[tuple[int, tuple[int, int, int, int], bytes]] = []
    for top in anchors:
        for left in anchors:
            box = (left, top, min(PATCH_SIZE - 1, left + side - 1), min(PATCH_SIZE - 1, top + side - 1))
            x1, y1, x2, y2 = box
            overlap = int(target_mask[y1 : y2 + 1, x1 : x2 + 1].sum())
            tie = hashlib.sha256(
                f"{sample_id}:decoy:{left}:{top}".encode("utf-8")
            ).digest()
            candidates.append((overlap, box, tie))
    candidates.sort(key=lambda item: (item[0], item[2]))
    chosen: list[tuple[int, int, int, int]] = []
    for _, box, _ in candidates:
        if all(_box_intersection(box, existing) == 0 for existing in chosen):
            chosen.append(box)
        if len(chosen) == PROPOSALS_PER_SAMPLE - 1:
            break
    if len(chosen) != PROPOSALS_PER_SAMPLE - 1:
        raise ValueError("could not construct three independent decoy proposals")
    return chosen


def _box_intersection(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> int:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def _proposal_record(
    *,
    audit_kind: str,
    box: tuple[int, int, int, int],
    seed: tuple[int, int],
    component_area: int,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    region_id: str = "pending",
) -> dict[str, Any]:
    x1, y1, x2, y2 = box
    crop = np.zeros_like(t1_mask, dtype=bool)
    crop[y1 : y2 + 1, x1 : x2 + 1] = True
    seed_x, seed_y = seed
    seed_state = {
        "t1": bool(t1_mask[seed_y, seed_x]),
        "t2": bool(t2_mask[seed_y, seed_x]),
    }
    return {
        "region_id": region_id,
        "audit_kind": audit_kind,
        "change_mask_state": "white" if audit_kind == "present" else "black",
        "temporal_difference_state": "present" if audit_kind in {"present", "missing"} else "absent",
        "box_normalized_1000": _normalized_box(box),
        "component_seed_normalized_1000": _normalized_point(seed),
        "t1_mask_pixels": int(np.logical_and(t1_mask, crop).sum()),
        "t2_mask_pixels": int(np.logical_and(t2_mask, crop).sum()),
        "component_area": int(component_area),
        "editable_seed_white": seed_state,
        "allowed_actions": ["box"],
        "transition_mask_facts": {},
    }


def _proposal_catalog(
    predicted_change: np.ndarray,
    gt_change: np.ndarray,
    error_type: str,
    sample_id: str,
    *,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], str]:
    source = (
        np.logical_and(gt_change, ~predicted_change)
        if error_type == "false_negative"
        else np.logical_and(predicted_change, ~gt_change)
    )
    if not source.any():
        raise ValueError(f"synthetic {error_type} mask contains no error component")
    target_box, target_seed, target_area = _component_geometry(source)
    proposals = [
        (
            True,
            _proposal_record(
                audit_kind="missing" if error_type == "false_negative" else "present",
                box=target_box,
                seed=target_seed,
                component_area=target_area,
                t1_mask=t1_mask,
                t2_mask=t2_mask,
            ),
        )
    ]
    for box in _decoy_boxes(source, sample_id):
        seed = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        proposals.append(
            (
                False,
                _proposal_record(
                    audit_kind="scan",
                    box=box,
                    seed=seed,
                    component_area=(box[2] - box[0] + 1) * (box[3] - box[1] + 1),
                    t1_mask=t1_mask,
                    t2_mask=t2_mask,
                ),
            )
        )
    if len(proposals) != PROPOSALS_PER_SAMPLE:
        raise AssertionError("proposal construction must produce exactly four regions")
    order = sorted(
        range(len(proposals)),
        key=lambda index: hashlib.sha256(
            f"{sample_id}:proposal-order:{index}".encode("utf-8")
        ).digest(),
    )
    catalog: list[dict[str, Any]] = []
    target_region_id = ""
    for region_index, logical_index in enumerate(order):
        is_target, proposal = proposals[logical_index]
        proposal = dict(proposal)
        proposal["region_id"] = f"r{region_index}"
        catalog.append(proposal)
        if is_target:
            target_region_id = proposal["region_id"]
    if target_region_id not in {"r0", "r1", "r2", "r3"}:
        raise AssertionError("shuffled target region id is invalid")
    return catalog, target_region_id


def _stage_payload(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "target_class": "building",
        "proposal_catalog": catalog,
        "max_selected_regions": 1,
        "schema": "persistent_region_screening_v11",
    }


def _assistant_target(error_type: str, region_id: str) -> str:
    reason = (
        "This black proposal contains visible building-change extent missing from the current mask."
        if error_type == "false_negative"
        else "This white proposal marks unchanged scene content as a building change."
    )
    return json.dumps(
        {"selection": {"region_ids": [region_id], "reason": reason}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def dataset_info_payload() -> dict[str, Any]:
    entry = {
        "file_name": "sharegpt.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
    return {dataset_name: dict(entry) for dataset_name in DATASET_NAMES}


def _build_one(
    record: PatchRecord,
    error_type: str,
    images_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_change = _load_bool(record.label_path, record.crop_box)
    t1 = _load_rgb(record.t1_path, record.crop_box)
    t2 = _load_rgb(record.t2_path, record.crop_box)
    if error_type == "false_negative":
        if not gt_change.any():
            raise ValueError(f"positive sample has empty GT: {record.sample_id}")
        largest = _components(gt_change)[0]
        removed_chunk = _connected_chunk(largest, record.sample_id)
        predicted_change = np.logical_and(gt_change, ~removed_chunk)
        t1_mask = np.array(gt_change, copy=True)
        t2_mask = np.zeros_like(gt_change, dtype=bool)
    elif error_type == "false_positive_change":
        if gt_change.any():
            raise ValueError(f"empty sample has non-empty GT: {record.sample_id}")
        predicted_change = _fp_mask_for_empty(record.sample_id)
        t1_mask = np.zeros_like(gt_change, dtype=bool)
        t2_mask = np.array(predicted_change, copy=True)
    else:
        raise ValueError(f"unsupported class: {error_type}")
    catalog, target_region_id = _proposal_catalog(
        predicted_change,
        gt_change,
        error_type,
        record.sample_id,
        t1_mask=t1_mask,
        t2_mask=t2_mask,
    )
    payload = _stage_payload(catalog)
    state = ChangeState(
        t1_image=np.asarray(t1),
        t2_image=np.asarray(t2),
        query="building",
        t1_mask=t1_mask,
        t2_mask=t2_mask,
        change_mask=predicted_change,
    )
    image_path = (images_dir / f"{record.sample_id}_{error_type}.png").resolve()
    overview = _proposal_overview(state, None, catalog)
    overview.save(image_path, format="PNG", optimize=True)
    user_text = (
        "Numbered global proposal overview: T1 | T2 | current change mask\n"
        "<image>\n"
        + _stage_prompt("select", payload)
    )
    assistant_text = _assistant_target(error_type, target_region_id)
    example = {
        "id": f"{record.sample_id}_{error_type}",
        "conversations": [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ],
        "images": [str(image_path)],
    }
    manifest = {
        "id": example["id"],
        "source_split": "train_original",
        "parent_id": record.parent_id,
        "patch_row": record.row,
        "patch_column": record.column,
        "crop_xyxy_exclusive": list(record.crop_box),
        "gt_change_pixels": int(gt_change.sum()),
        "candidate_change_pixels": int(predicted_change.sum()),
        "synthetic_error_pixels": int(
            np.logical_xor(predicted_change, gt_change).sum()
        ),
        "synthetic_error_type": error_type,
        "selected_region_ids": [target_region_id],
        "proposal_count": len(catalog),
        "model_visible_image": str(image_path),
        "source_t1_sha256": _sha256(record.t1_path),
        "source_t2_sha256": _sha256(record.t2_path),
        "source_label_sha256": _sha256(record.label_path),
        "assistant_target_sha256": hashlib.sha256(
            assistant_text.encode("utf-8")
        ).hexdigest(),
    }
    return example, manifest


def validate_output(output_dir: Path) -> dict[str, Any]:
    sharegpt_path = output_dir / "sharegpt.json"
    dataset_info_path = output_dir / "dataset_info.json"
    manifest_path = output_dir / "manifest.json"
    with sharegpt_path.open(encoding="utf-8") as handle:
        examples = json.load(handle)
    with dataset_info_path.open(encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    errors: list[str] = []
    if not isinstance(examples, list) or len(examples) != TOTAL_SAMPLES:
        errors.append(f"sharegpt must contain exactly {TOTAL_SAMPLES} examples")
    expected_entry = {
        "file_name": "sharegpt.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
    if set(dataset_info) != set(DATASET_NAMES):
        errors.append(f"dataset_info must contain exactly {list(DATASET_NAMES)}")
    for dataset_name in DATASET_NAMES:
        if dataset_info.get(dataset_name) != expected_entry:
            errors.append(
                f"dataset_info entry {dataset_name} does not match the ShareGPT schema"
            )
    ids: set[str] = set()
    parents: set[str] = set()
    classes: dict[str, int] = {}
    image_sha256: dict[str, str] = {}
    image_paths: set[str] = set()
    target_by_id: dict[str, str] = {}
    images_root = (output_dir / "images").resolve()
    for index, example in enumerate(examples if isinstance(examples, list) else []):
        sample_id = str(example.get("id", ""))
        if not sample_id or sample_id in ids:
            errors.append(f"example {index} has an empty or duplicate id")
        ids.add(sample_id)
        conversations = example.get("conversations")
        images = example.get("images")
        if not isinstance(conversations, list) or len(conversations) != 2:
            errors.append(f"{sample_id}: conversations must contain one user/assistant pair")
            continue
        user = conversations[0]
        assistant = conversations[1]
        if user.get("from") != "human" or assistant.get("from") != "gpt":
            errors.append(f"{sample_id}: invalid ShareGPT roles")
        placeholder_count = str(user.get("value", "")).count("<image>")
        if not isinstance(images, list) or placeholder_count != len(images):
            errors.append(f"{sample_id}: image placeholders and paths do not match")
        if not isinstance(images, list) or len(images) != 1:
            errors.append(f"{sample_id}: exactly one overview image is required")
        for raw_image_path in images or []:
            image_path = Path(raw_image_path)
            if not image_path.is_absolute():
                errors.append(f"{sample_id}: image path must be absolute")
            else:
                resolved_image = image_path.resolve()
                if images_root not in resolved_image.parents:
                    errors.append(
                        f"{sample_id}: image must be contained by {images_root}"
                    )
            if str(image_path) in image_paths:
                errors.append(f"{sample_id}: duplicate model-visible image path")
            image_paths.add(str(image_path))
            if not image_path.is_file():
                errors.append(f"{sample_id}: missing image {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    if image.size != (PATCH_SIZE * 3, PATCH_SIZE):
                        errors.append(
                            f"{sample_id}: overview size must be "
                            f"{PATCH_SIZE * 3}x{PATCH_SIZE}, found {image.size}"
                        )
            except OSError as error:
                errors.append(f"{sample_id}: unreadable overview image: {error}")
                continue
            image_sha256[str(image_path)] = _sha256(image_path)
        try:
            target = json.loads(str(assistant.get("value", "")))
            selection = target["selection"]
            region_ids = selection["region_ids"]
            if set(target) != {"selection"} or set(selection) != {"region_ids", "reason"}:
                raise ValueError("wrong target keys")
            if not isinstance(region_ids, list) or len(region_ids) != 1:
                raise ValueError("invalid region_ids")
            if region_ids[0] not in {"r0", "r1", "r2", "r3"}:
                raise ValueError("unknown region id")
            if not isinstance(selection["reason"], str) or not selection["reason"].strip():
                raise ValueError("reason must be a non-empty string")
            target_by_id[sample_id] = region_ids[0]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{sample_id}: invalid select target: {error}")
        forbidden = ("target_view", '"action"', '"coordinate"', '"accept"')
        combined = str(user.get("value", "")) + str(assistant.get("value", ""))
        present = [token for token in forbidden if token in combined]
        if present:
            errors.append(f"{sample_id}: forbidden supervision keys: {present}")
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or len(samples) != TOTAL_SAMPLES:
        errors.append(f"manifest must contain exactly {TOTAL_SAMPLES} samples")
    manifest_ids: set[str] = set()
    target_region_ids: set[str] = set()
    for item in samples if isinstance(samples, list) else []:
        manifest_id = str(item.get("id", ""))
        if manifest_id in manifest_ids:
            errors.append(f"duplicate manifest id: {manifest_id}")
        manifest_ids.add(manifest_id)
        parent_id = str(item.get("parent_id", ""))
        if parent_id in parents:
            errors.append(f"parent scene leakage inside smoke set: {parent_id}")
        parents.add(parent_id)
        error_type = str(item.get("synthetic_error_type", ""))
        classes[error_type] = classes.get(error_type, 0) + 1
        if item.get("proposal_count") != PROPOSALS_PER_SAMPLE:
            errors.append(f"{manifest_id}: expected exactly four proposals")
        selected_region_ids = item.get("selected_region_ids")
        if (
            not isinstance(selected_region_ids, list)
            or len(selected_region_ids) != 1
            or selected_region_ids[0] not in {"r0", "r1", "r2", "r3"}
        ):
            errors.append(f"{manifest_id}: invalid shuffled target region id")
        else:
            target_region_ids.add(selected_region_ids[0])
            if target_by_id.get(manifest_id) != selected_region_ids[0]:
                errors.append(
                    f"{manifest_id}: ShareGPT target and manifest target disagree"
                )
    expected_classes = {
        "false_negative": SAMPLES_PER_CLASS,
        "false_positive_change": SAMPLES_PER_CLASS,
    }
    if classes != expected_classes:
        errors.append(f"unexpected class distribution: {classes}")
    if len(parents) != TOTAL_SAMPLES:
        errors.append(f"expected {TOTAL_SAMPLES} unique parent scenes, found {len(parents)}")
    if ids != manifest_ids:
        errors.append("ShareGPT and manifest IDs do not match")
    if manifest.get("dataset_names") != list(DATASET_NAMES):
        errors.append("manifest dataset_names do not match the registered datasets")
    if manifest.get("source_split") != "train_original":
        errors.append("manifest source_split must be train_original")
    if manifest.get("test_data_used") is not False:
        errors.append("manifest must state that test data was not used")
    if manifest.get("raw_gt_label_image_model_visible") is not False:
        errors.append("manifest must state that the raw GT label image is not model-visible")
    if manifest.get("gt_derived_synthetic_facts_model_visible") is not True:
        errors.append("manifest must state that GT-derived smoke facts are model-visible")
    if len(target_region_ids) < 2:
        errors.append("target region IDs were not shuffled across examples")
    report = {
        "valid": not errors,
        "errors": errors,
        "examples": len(examples) if isinstance(examples, list) else 0,
        "unique_parent_scenes": len(parents),
        "class_distribution": classes,
        "target_region_ids": sorted(target_region_ids),
        "image_placeholder_mismatches": sum(
            "placeholders" in item for item in errors
        ),
        "unknown_region_ids": sum("unknown region id" in item for item in errors),
        "sharegpt_sha256": _sha256(sharegpt_path),
        "dataset_info_sha256": _sha256(dataset_info_path),
        "model_visible_image_sha256": image_sha256,
    }
    return report


def build_dataset(dataset_root: Path, output_dir: Path, *, seed: int, overwrite: bool) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; pass --overwrite to replace it"
            )
        resolved = output_dir.resolve()
        allowed_parent = DEFAULT_OUTPUT_DIR.parent.resolve()
        if resolved == allowed_parent or allowed_parent not in resolved.parents:
            raise ValueError(f"refusing to overwrite unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    records = scan_train_patches(dataset_root)
    positives, empties = choose_smoke_records(records, seed=seed)
    examples: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] = []
    for record, error_type in [
        *((item, "false_negative") for item in positives),
        *((item, "false_positive_change") for item in empties),
    ]:
        example, item = _build_one(record, error_type, images_dir)
        examples.append(example)
        sample_manifest.append(item)
    dataset_info = dataset_info_payload()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_names": list(DATASET_NAMES),
        "seed": seed,
        "source_dataset_root": str(dataset_root.resolve()),
        "source_split": "train_original",
        "test_data_used": False,
        "raw_gt_label_image_model_visible": False,
        "gt_derived_synthetic_facts_model_visible": True,
        "sample_selection": {
            "patch_size": PATCH_SIZE,
            "grid_size": GRID_SIZE,
            "total_train_patches": len(records),
            "empty_train_patches": sum(item.gt_pixels == 0 for item in records),
            "positive_train_patches": sum(item.gt_pixels > 0 for item in records),
            "strategy": "centered area quantiles with seeded tie break and unique parents",
        },
        "target_contract": "current select-stage JSON only",
        "omitted_supervision": [
            "target_view",
            "action",
            "coordinate",
            "accept",
            "quality_score",
            "hidden_chain_of_thought",
        ],
        "samples": sample_manifest,
    }
    for path, payload in (
        (output_dir / "sharegpt.json", examples),
        (output_dir / "dataset_info.json", dataset_info),
        (output_dir / "manifest.json", manifest),
    ):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    report = validate_output(output_dir)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not report["valid"]:
        raise ValueError("generated dataset failed validation: " + "; ".join(report["errors"]))
    return report


def main() -> None:
    args = _parse_args()
    try:
        report = (
            validate_output(args.output_dir)
            if args.validate_only
            else build_dataset(
                args.dataset_root,
                args.output_dir,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
