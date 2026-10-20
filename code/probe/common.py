"""Shared paths, data loading and object matching for the CD_1016 probe experiments.

Task definition: doc/change_dection_agent.md. Only numpy/PIL here so the module can be
imported from both the SAM3 environment and the analysis environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path("/guisongxia01/pangchao/wangyihan")
OUT = WORKSPACE / "Chang-Agent/outputs/CD_1016_probe"
PRED = OUT / "preds"

LIM1 = WORKSPACE / "datasets/lim1"
# dirs = (t1 folder, t2 folder, binary change label folder); prefix selects the split.
# task: "building" (building-change datasets) or "general" (any land-cover change); it selects
# the MLLM prompt in experiment 3.
DATASETS = {
    "levir": {
        "root": WORKSPACE / "datasets/LEVIR-CD/test_original", "dirs": ("A", "B", "label"), "prefix": "",
        "schanger": "levir", "task": "building",
        "detectors": ("schanger_small", "schanger_base", "sam3_omniovcd"),
    },
    "whu": {
        "root": LIM1 / "WHU-CD/test", "dirs": ("A", "B", "label"), "prefix": "",
        "schanger": "whu", "task": "building",
        "detectors": ("schanger_small", "schanger_base", "sam3_omniovcd"),
        # Per-date building labels: decide new vs demolished for the SAM3 oracle.
        "sem1": LIM1 / "WHU-CD/slabel1", "sem2": LIM1 / "WHU-CD/slabel2",
    },
    "sysu": {
        "root": LIM1 / "SYSU-CD", "dirs": ("img1", "img2", "label_unch0chi255"), "prefix": "test_",
        "schanger": "sysu", "task": "general",
        "detectors": ("schanger_small", "schanger_base"),
    },
    "second": {
        "root": LIM1 / "SECOND", "dirs": ("img1", "img2", "label_unch0chi255"), "prefix": "test_",
        "task": "general", "detectors": ("mambascd_base",),
        # Per-date land-cover classes inside changed pixels (0 = no change): GT objects are
        # split by (t1 class, t2 class) transition before taking connected components.
        "class1": LIM1 / "SECOND/slabel1", "class2": LIM1 / "SECOND/slabel2",
        "classes": {1: "low vegetation", 2: "non-vegetated surface", 3: "tree", 4: "water",
                    5: "building", 6: "playground"},
    },
}
DATASETS["second_cv"] = {**DATASETS["second"], "prefix": "train_"}
DATASETS["second_scannet"] = {**DATASETS["second"], "detectors": ("scannet",)}
RS_DATASETS = Path("/guisongxia01/pangchao/haoxiangyu/rs_datasets")
for dataset, folder, classes in (
    ("lsscd", "lsSCD", {0: "Unknown", 1: "Bareland", 2: "Rangeland", 3: "Developed space",
                       4: "Road", 5: "Tree", 6: "Water", 7: "Agriculture land", 8: "Building"}),
    ("scscd7", "SCSCD7", {0: "Bareland", 1: "Water", 2: "Building", 3: "Structure",
                         4: "Farmland", 5: "Vegetation", 6: "Road"}),
):
    DATASETS[dataset] = {
        "root": RS_DATASETS / folder, "dirs": ("img1", "img2", "label_unch0chi255"),
        "prefix": "test_", "task": "general", "detectors": ("mambascd_base",),
        "class1": RS_DATASETS / folder / "slabel1", "class2": RS_DATASETS / folder / "slabel2",
        "classes": classes,
    }

# Used only to estimate train object geometry, never to train a transfer detector.
# Keep LEVIR/WHU test registrations and their label semantics unchanged.
TRAIN_DATASETS = {
    "levir": {**DATASETS["levir"], "root": WORKSPACE / "datasets/LEVIR-CD/train_original"},
    "whu": {**DATASETS["whu"], "root": LIM1 / "WHU-CD",
            "dirs": ("img1", "img2", "label_unch0chi255"), "prefix": "train_"},
    **{d: {**DATASETS[d], "prefix": "train_"} for d in ("lsscd", "scscd7")},
}
DETECTORS = ("schanger_small", "schanger_base", "sam3_omniovcd")   # LEVIR/WHU set
PROB_DETECTORS = ("schanger_small", "schanger_base", "mambascd_base", "scannet")
AREA_BUCKETS = ((0, 100, "<100"), (100, 1000, "100-1000"), (1000, None, ">1000"))
DISPLAY = {"levir": "LEVIR-CD", "whu": "WHU-CD", "sysu": "SYSU-CD", "second": "SECOND",
           "second_cv": "SECOND train (cross-fitted)", "second_scannet": "SECOND (SCanNet)",
           "lsscd": "lsSCD", "scscd7": "SCSCD7"}


def detectors(dataset: str) -> tuple[str, ...]:
    return DATASETS[dataset]["detectors"]


def names(dataset: str) -> list[str]:
    spec = DATASETS[dataset]
    return sorted(p.name for p in (spec["root"] / spec["dirs"][0]).glob(spec["prefix"] + "*.png"))


def load_rgb(dataset: str, name: str, view: str) -> np.ndarray:
    spec = DATASETS[dataset]
    folder = spec["dirs"][{"t1": 0, "t2": 1}[view]]
    return np.asarray(Image.open(spec["root"] / folder / name).convert("RGB"))


def load_label(dataset: str, name: str) -> np.ndarray:
    spec = DATASETS[dataset]
    return np.asarray(Image.open(spec["root"] / spec["dirs"][2] / name)) > 127


def load_semantic(dataset: str, name: str):
    """Return (building_t1, building_t2) or None when the dataset has no per-date building labels."""
    spec = DATASETS[dataset]
    if "sem1" not in spec:
        return None
    return tuple(np.asarray(Image.open(spec[key] / name)) > 0 for key in ("sem1", "sem2"))


def load_classes(dataset: str, name: str):
    """Return (class_t1, class_t2) maps for semantic-change datasets, else None."""
    spec = DATASETS[dataset]
    if "class1" not in spec:
        return None
    return tuple(np.asarray(Image.open(spec[key] / name)) for key in ("class1", "class2"))


def load_pred(dataset: str, detector: str, name: str, threshold: float | None = None) -> np.ndarray:
    base = PRED / dataset / detector
    if threshold is None or detector not in PROB_DETECTORS:
        return np.asarray(Image.open(base / "mask" / name)) > 0
    return load_prob(dataset, detector, name) > threshold


def load_prob(dataset: str, detector: str, name: str) -> np.ndarray:
    return np.load(PRED / dataset / detector / "prob" / (Path(name).stem + ".npy")).astype(np.float32)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pixel_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def area_bucket(area: int) -> str:
    for low, high, label in AREA_BUCKETS:
        if area >= low and (high is None or area < high):
            return label
    raise ValueError(area)
