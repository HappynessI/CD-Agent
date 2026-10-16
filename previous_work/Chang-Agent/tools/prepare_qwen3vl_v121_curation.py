#!/usr/bin/env python3
"""Select parent-isolated LEVIR regions and run frozen SAM3 initialization.

This is the first, GPU-backed phase of the v12.1 few-shot data pipeline.  It
creates curator previews and frozen-predictor masks only; it does not author
SFT targets.  GT is used solely to select diagnostic regions and is never
passed to SAM3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PATCH_SIZE = 256
GRID_SIZE = 4


@dataclass(frozen=True)
class Candidate:
    split: str
    parent_id: str
    row: int
    column: int
    kind: str
    gt_pixels: int
    component_pixels: int
    component_box_xyxy: tuple[int, int, int, int]
    t1_path: Path
    t2_path: Path
    label_path: Path

    @property
    def candidate_id(self) -> str:
        return f"{self.split}_{self.parent_id}_r{self.row}c{self.column}_{self.kind}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-bpe", type=Path, required=True)
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--train-positive", type=int, default=8)
    parser.add_argument("--train-negative", type=int, default=4)
    parser.add_argument("--val-positive", type=int, default=3)
    parser.add_argument("--val-negative", type=int, default=3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask)
    result: list[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        todo = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while todo:
            y, x = todo.pop()
            points.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not seen[ny, nx]
                ):
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        component = np.zeros_like(mask)
        ys, xs = zip(*points)
        component[np.asarray(ys), np.asarray(xs)] = True
        result.append(component)
    return result


def component_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def candidate_rank(candidate: Candidate, seed: int) -> tuple[str, str]:
    key = f"{seed}:{candidate.candidate_id}".encode()
    return hashlib.sha256(key).hexdigest(), candidate.candidate_id


def discover_split(dataset_root: Path, split: str) -> list[Candidate]:
    split_root = dataset_root / f"{split}_original"
    a_root, b_root, label_root = (split_root / name for name in ("A", "B", "label"))
    candidates: list[Candidate] = []
    for t1_path in sorted(a_root.glob("*.png")):
        parent_id = t1_path.stem
        t2_path = b_root / t1_path.name
        label_path = label_root / t1_path.name
        if not t2_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"incomplete LEVIR triplet for {parent_id}")
        with Image.open(label_path) as image:
            label = np.asarray(image.convert("L")) > 0
        if label.shape != (PATCH_SIZE * GRID_SIZE, PATCH_SIZE * GRID_SIZE):
            raise ValueError(f"unexpected label shape for {label_path}: {label.shape}")
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                y1, x1 = row * PATCH_SIZE, column * PATCH_SIZE
                patch = label[y1 : y1 + PATCH_SIZE, x1 : x1 + PATCH_SIZE]
                gt_pixels = int(patch.sum())
                if gt_pixels == 0:
                    candidates.append(
                        Candidate(
                            split,
                            parent_id,
                            row,
                            column,
                            "unchanged",
                            0,
                            0,
                            (72, 72, 184, 184),
                            t1_path,
                            t2_path,
                            label_path,
                        )
                    )
                    continue
                local_components = components(patch)
                if not local_components:
                    continue
                largest = max(local_components, key=lambda item: int(item.sum()))
                pixels = int(largest.sum())
                x1b, y1b, x2b, y2b = component_box(largest)
                width, height = x2b - x1b, y2b - y1b
                if not (
                    180 <= pixels <= 14000
                    and 10 <= width <= 180
                    and 10 <= height <= 180
                    and x1b >= 8
                    and y1b >= 8
                    and x2b <= PATCH_SIZE - 8
                    and y2b <= PATCH_SIZE - 8
                ):
                    continue
                candidates.append(
                    Candidate(
                        split,
                        parent_id,
                        row,
                        column,
                        "changed",
                        gt_pixels,
                        pixels,
                        (x1b, y1b, x2b, y2b),
                        t1_path,
                        t2_path,
                        label_path,
                    )
                )
    return candidates


def select_candidates(
    candidates: list[Candidate], *, kind: str, count: int, seed: int
) -> list[Candidate]:
    pool = sorted(
        (item for item in candidates if item.kind == kind),
        key=lambda item: candidate_rank(item, seed),
    )
    selected: list[Candidate] = []
    parents: set[str] = set()
    for item in pool:
        if item.parent_id in parents:
            continue
        selected.append(item)
        parents.add(item.parent_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"requested {count} {kind} candidates, found {len(selected)}")
    return selected


def crop_triplet(candidate: Candidate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = (
        candidate.column * PATCH_SIZE,
        candidate.row * PATCH_SIZE,
        (candidate.column + 1) * PATCH_SIZE,
        (candidate.row + 1) * PATCH_SIZE,
    )
    with Image.open(candidate.t1_path) as image:
        t1 = np.asarray(image.convert("RGB").crop(box))
    with Image.open(candidate.t2_path) as image:
        t2 = np.asarray(image.convert("RGB").crop(box))
    with Image.open(candidate.label_path) as image:
        label = np.asarray(image.convert("L").crop(box)) > 0
    return t1, t2, label


def mask_panel(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[np.asarray(mask, dtype=bool)] = color
    return Image.fromarray(panel)


def make_preview(
    candidate: Candidate,
    t1: np.ndarray,
    t2: np.ndarray,
    label: np.ndarray,
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
) -> Image.Image:
    panels = [
        Image.fromarray(t1),
        Image.fromarray(t2),
        mask_panel(label, (255, 255, 255)),
        mask_panel(t1_mask, (255, 255, 255)),
        mask_panel(t2_mask, (255, 255, 255)),
    ]
    preview = Image.new("RGB", (PATCH_SIZE * len(panels), PATCH_SIZE + 36), "white")
    draw = ImageDraw.Draw(preview)
    for index, (panel, title) in enumerate(
        zip(panels, ("T1 RGB", "T2 RGB", "GT change (curator only)", "SAM3 T1", "SAM3 T2"))
    ):
        preview.paste(panel, (PATCH_SIZE * index, 36))
        draw.text((PATCH_SIZE * index + 6, 10), title, fill="black")
    x1, y1, x2, y2 = candidate.component_box_xyxy
    for index in (0, 1, 2):
        offset = PATCH_SIZE * index
        draw.rectangle((offset + x1, 36 + y1, offset + x2, 36 + y2), outline="yellow", width=2)
    return preview


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    for path in (args.dataset_root, args.sam3_checkpoint, args.sam3_bpe):
        if not path.exists():
            raise FileNotFoundError(path)
    started = time.monotonic()
    discovered = {
        split: discover_split(args.dataset_root, split) for split in ("train", "val")
    }
    chosen: list[Candidate] = []
    for split, positive_count, negative_count in (
        ("train", args.train_positive, args.train_negative),
        ("val", args.val_positive, args.val_negative),
    ):
        chosen.extend(
            select_candidates(
                discovered[split], kind="changed", count=positive_count, seed=args.seed
            )
        )
        chosen.extend(
            select_candidates(
                discovered[split], kind="unchanged", count=negative_count, seed=args.seed + 1
            )
        )
    if len({(item.split, item.parent_id) for item in chosen}) != len(chosen):
        raise RuntimeError("candidate selection violated parent isolation")

    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    from change_agent.adapters.sam3_adapter import SAM3ProcessorAdapter

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")
    device = torch.device("cuda")
    model = build_sam3_image_model(
        bpe_path=str(args.sam3_bpe),
        checkpoint_path=str(args.sam3_checkpoint),
        device="cuda",
        load_from_HF=False,
    )
    model.to(device)
    model._device = device
    adapter = SAM3ProcessorAdapter(
        Sam3Processor(model, resolution=args.sam3_resolution, device="cuda")
    )

    args.output.mkdir(parents=True)
    samples_root = args.output / "samples"
    previews_root = args.output / "previews"
    samples_root.mkdir()
    previews_root.mkdir()
    rows: list[dict[str, object]] = []
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for index, candidate in enumerate(chosen):
        t1, t2, label = crop_triplet(candidate)
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            t1_mask, _ = adapter.segment_text(t1, "building")
            t2_mask, _ = adapter.segment_text(t2, "building")
        t1_mask = np.asarray(t1_mask, dtype=bool)
        t2_mask = np.asarray(t2_mask, dtype=bool)
        sample_dir = samples_root / candidate.candidate_id
        sample_dir.mkdir()
        paths = {
            "t1": sample_dir / "t1.png",
            "t2": sample_dir / "t2.png",
            "gt_change": sample_dir / "gt_change.png",
            "sam3_t1_mask": sample_dir / "sam3_t1_mask.png",
            "sam3_t2_mask": sample_dir / "sam3_t2_mask.png",
        }
        Image.fromarray(t1).save(paths["t1"])
        Image.fromarray(t2).save(paths["t2"])
        Image.fromarray(label.astype(np.uint8) * 255).save(paths["gt_change"])
        Image.fromarray(t1_mask.astype(np.uint8) * 255).save(paths["sam3_t1_mask"])
        Image.fromarray(t2_mask.astype(np.uint8) * 255).save(paths["sam3_t2_mask"])
        preview_path = previews_root / f"{index:02d}_{candidate.candidate_id}.png"
        make_preview(candidate, t1, t2, label, t1_mask, t2_mask).save(preview_path)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "split": candidate.split,
                "parent_id": candidate.parent_id,
                "patch_row": candidate.row,
                "patch_column": candidate.column,
                "kind": candidate.kind,
                "gt_pixels": candidate.gt_pixels,
                "component_pixels": candidate.component_pixels,
                "component_box_xyxy": list(candidate.component_box_xyxy),
                "source_paths": {
                    "t1": str(candidate.t1_path),
                    "t2": str(candidate.t2_path),
                    "label": str(candidate.label_path),
                },
                "source_sha256": {
                    "t1": sha256_file(candidate.t1_path),
                    "t2": sha256_file(candidate.t2_path),
                    "label": sha256_file(candidate.label_path),
                },
                "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
                "artifact_sha256": {key: sha256_file(path) for key, path in paths.items()},
                "preview": str(preview_path.resolve()),
                "sam3_pixels": {
                    "t1": int(t1_mask.sum()),
                    "t2": int(t2_mask.sum()),
                },
            }
        )
    manifest = {
        "schema": "qwen3vl_v121_curation_candidates_v1",
        "seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()),
        "source_splits": ["train_original", "val_original"],
        "test_data_used": False,
        "gt_used_for": ["candidate_selection", "curator_preview", "offline_target_validation"],
        "gt_visible_to_sam3": False,
        "sam3": {
            "checkpoint": str(args.sam3_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.sam3_checkpoint),
            "bpe": str(args.sam3_bpe.resolve()),
            "bpe_sha256": sha256_file(args.sam3_bpe),
            "resolution": args.sam3_resolution,
            "query": "building",
        },
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "cuda_device": torch.cuda.get_device_name(0),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "samples": rows,
    }
    (args.output / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / ".curation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"status": "success", "samples": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
