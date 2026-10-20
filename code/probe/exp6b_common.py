"""Shared pool indexing, feature batches, MLP and metrics for experiment 6b."""
from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import OUT, names
from exp6_pool_io import DETECTOR  # noqa: F401 (used by exp6b_eval)

ROOT = OUT / "exp6"
SOURCES = ("B", "G0.05", "C1", "C2", "H1", "pred")
BACKBONES = ("none", "sam_vith", "remoteclip", "dinov3")
SEED = 1020


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def run_dir(backbone, include_pred=False):
    return ROOT / "verifier" / (backbone + ("_pred" if include_pred else ""))


def source_of(row):
    return "pred" if row["kind"] == "pred" else row["subsource"] or row["source"]


def rows_by_image(dataset):
    """The merged batch2 JSONL is sorted by (image, piece_id)."""
    with (ROOT / "pool" / dataset / "pieces.jsonl").open() as handle:
        for name, rows in groupby(map(json.loads, handle), key=lambda r: r["image"]):
            yield name, list(rows)


def verifier_split():
    manifest = json.loads((ROOT / "second_cv/manifest.json").read_text())
    train, val = (manifest[k] for k in ("verifier_train_images", "verifier_val_images"))
    if (len(set(train)) != len(train) or len(set(val)) != len(val)
            or set(train) & set(val) or set(train) | set(val) != set(names("second_cv"))):
        raise ValueError("Verifier train/validation images must partition second_cv")
    return sorted(train), sorted(val)


class Pool:
    """Compact inference metadata; GT fields are kept only in a separate label array."""

    def __init__(self, dataset):
        self.dataset = dataset
        summary = json.loads((ROOT / "pool" / dataset / "pieces_summary.json").read_text())
        self.images = [r["image"] for r in summary["per_image"]]
        if self.images != names(dataset):
            raise ValueError(f"{dataset}: pool summary does not cover the complete image split")
        self.image_index = {name: i for i, name in enumerate(self.images)}
        count = int(summary["pieces"])
        self.image_id = np.empty(count, np.int32)
        self.piece_id = np.empty(count, np.int32)
        self.source = np.empty(count, np.int8)
        self.label = np.empty(count, np.int8)
        # mean/max/min/p90, log(area), six source indicators, split.
        self.meta = np.zeros((count, 12), np.float32)
        cursor = 0
        self.slices = {name: slice(0, 0) for name in self.images}
        expected = {r["image"]: r["pieces"] for r in summary["per_image"]}
        previous = ""
        for name, rows in rows_by_image(dataset):
            if name not in self.image_index or name <= previous or len(rows) != expected[name]:
                raise ValueError(f"{dataset}/{name}: pool order or summary count mismatch")
            previous = name
            start = cursor
            for row in rows:
                i = cursor
                self.image_id[i] = self.image_index[name]
                self.piece_id[i] = row["piece_id"]
                self.source[i] = SOURCES.index(source_of(row))
                self.label[i] = -1 if row["label"] is None else row["label"]
                self.meta[i, :4] = [row["det_prob"][k] for k in ("mean", "max", "min", "p90")]
                self.meta[i, 4] = np.log(row["area"])
                self.meta[i, 5 + self.source[i]] = 1
                self.meta[i, 11] = float(row["split"])
                cursor += 1
            self.slices[name] = slice(start, cursor)
        if cursor != count:
            raise ValueError(f"{dataset}: expected {count} pieces, read {cursor}")
        self.stride = int(self.piece_id.max()) + 1 if count else 1
        self.keys = self.image_id.astype(np.int64) * self.stride + self.piece_id
        if len(self.keys) > 1 and np.any(self.keys[1:] <= self.keys[:-1]):
            raise ValueError("Pool (image, piece_id) must be unique and sorted")
        for name, expected_count in expected.items():
            sl = self.slices[name]
            if sl.stop - sl.start != expected_count:
                raise ValueError(f"{name}: missing pieces")
        if not np.isfinite(self.meta).all():
            raise ValueError("Non-finite inference metadata")

    def __len__(self):
        return len(self.piece_id)

    def image_mask(self, images):
        ids = [self.image_index[name] for name in images]
        return np.isin(self.image_id, ids)

    def feature_indices(self, image, piece_id):
        ids = np.fromiter((self.image_index[str(n)] for n in image), np.int64, len(image))
        keys = ids * self.stride + piece_id
        idx = np.searchsorted(self.keys, keys)
        if np.any(idx >= len(self)) or not np.array_equal(self.keys[idx], keys):
            raise ValueError("Feature index does not match the current pool")
        if len(np.unique(idx)) != len(idx):
            raise ValueError("Duplicate feature indices")
        return idx

    def score_arrays(self, scores):
        return dict(image=np.asarray(self.images)[self.image_id], piece_id=self.piece_id,
                    score=np.asarray(scores, np.float32))


class Features:
    """Read one NPZ shard at a time; never materialize the full 6D input matrix."""

    def __init__(self, pool, backbone, expected_config=None):
        self.pool, self.backbone = pool, backbone
        if backbone == "none":
            self.paths, self.config, self.input_dim = [None], {"backbone": "none"}, 11
        else:
            base = ROOT / "features" / backbone / pool.dataset
            self.paths = sorted((p for p in base.glob("shard_*.npz") if p.stem[6:].isdigit()),
                                key=lambda p: int(p.stem[6:]))
            if not self.paths:
                raise FileNotFoundError(f"No feature shards in {base}")
            with np.load(self.paths[0], allow_pickle=False) as z:
                self.config = json.loads(str(z["config"]))
                shards = int(z["shards"])
            if [p.name for p in self.paths] != [f"shard_{i}.npz" for i in range(shards)]:
                raise ValueError(f"{base}: incomplete or mixed feature shards")
            self.input_dim = 6 * self.config["dim"] + 13
        if expected_config is not None and self.config != expected_config:
            raise ValueError("Training and inference feature configurations differ")

    def batches(self, selected, batch_size, rng=None):
        paths = list(self.paths)
        if rng is not None:
            rng.shuffle(paths)
        seen = np.zeros(len(self.pool), bool)
        for path in paths:
            if path is None:
                indices, vectors = np.arange(len(self.pool)), None
            else:
                with np.load(path, allow_pickle=False) as z:
                    if (json.loads(str(z["config"])) != self.config
                            or int(z["shards"]) != len(self.paths)
                            or int(z["shard"]) != int(path.stem[6:])
                            or str(z["dataset"]) != self.pool.dataset):
                        raise ValueError(f"{path}: inconsistent feature metadata")
                    indices = self.pool.feature_indices(z["image"], z["piece_id"])
                    vectors = z["features"]
                if vectors.shape != (len(indices), 4, self.config["dim"]):
                    raise ValueError(f"{path}: feature shape mismatch")
                if not np.isfinite(vectors).all():
                    raise ValueError(f"{path}: non-finite features")
            if seen[indices].any():
                raise ValueError("Duplicate pieces across feature shards")
            seen[indices] = True
            local = np.flatnonzero(selected[indices])
            if rng is not None:
                rng.shuffle(local)
            for start in range(0, len(local), batch_size):
                take = local[start:start + batch_size]
                idx = indices[take]
                meta = self.pool.meta[idx]
                if vectors is None:
                    x = meta[:, :11].copy()
                else:
                    f1, f2, c1, c2 = vectors[take].astype(np.float32).transpose(1, 0, 2)
                    denom = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1)
                    cosine = (f1 * f2).sum(1) / np.maximum(denom, 1e-12)
                    x = np.concatenate((f1, f2, np.abs(f1 - f2), c1, c2,
                                        np.abs(c1 - c2), cosine[:, None], meta), axis=1)
                yield idx, x
            del vectors
        if not seen.all():
            raise ValueError(f"Missing features for {int((~seen).sum())} pieces")


def classifier(input_dim, dropout=0.15):
    return nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(256, 1))


def normalization(features, selected, batch_size):
    total = np.zeros(features.input_dim, np.float64)
    squares = np.zeros_like(total)
    count = 0
    for _, x in features.batches(selected, batch_size):
        values = x.astype(np.float64)
        total += values.sum(0)
        squares += (values * values).sum(0)
        count += len(x)
    if not count:
        raise ValueError("No training samples")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    std[std < 1e-6] = 1
    return mean.astype(np.float32), std.astype(np.float32)


@torch.inference_mode()
def predict(model, features, selected, mean, std, batch_size, device):
    model.eval()
    scores = np.full(len(features.pool), np.nan, np.float32)
    for idx, x in features.batches(selected, batch_size):
        x = torch.from_numpy((x - mean) / std).to(device)
        scores[idx] = model(x).squeeze(1).sigmoid().cpu().numpy()
    if not np.isfinite(scores[selected]).all():
        raise ValueError("Non-finite or missing model scores")
    return scores


def binary_metrics(labels, scores):
    """Tie-aware ROC AUC, stepwise AP, and the exp4b >= threshold FPR convention."""
    y, s = np.asarray(labels, np.int64), np.asarray(scores, np.float64)
    if len(y) != len(s) or not np.isfinite(s).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Metrics require finite scores and binary labels")
    positive, negative = int(y.sum()), int(len(y) - y.sum())
    result = dict(n=len(y), positive=positive, negative=negative, auc=None, ap=None)
    for recall in (80, 90):
        result[f"recall_{recall}"] = dict(threshold=None, fpr=None, achieved_recall=None)
    if not len(y):
        return result
    order = np.argsort(-s, kind="stable")
    yy, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.cumsum(yy)[ends]
    fp = ends + 1 - tp
    if positive:
        result["ap"] = float(np.sum(np.diff(np.r_[0, tp]) / positive * tp / (ends + 1)))
        for recall in (80, 90):
            i = int(np.searchsorted(tp, int(np.ceil(positive * recall / 100))))
            result[f"recall_{recall}"] = dict(
                threshold=float(ss[ends[i]]), fpr=float(fp[i] / negative) if negative else None,
                achieved_recall=float(tp[i] / positive))
    if positive and negative:
        tpr, fpr = np.r_[0, tp / positive], np.r_[0, fp / negative]
        result["auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return result


def grouped_metrics(pool, scores, selected):
    recall = selected & (pool.source != SOURCES.index("pred")) & (pool.label >= 0)
    report = {"overall": binary_metrics(pool.label[recall], scores[recall]), "by_source": {}}
    for i, source in enumerate(SOURCES[:-1]):
        take = recall & (pool.source == i)
        report["by_source"][source] = binary_metrics(pool.label[take], scores[take])
    take = recall & np.isin(pool.source, [SOURCES.index("C1"), SOURCES.index("C2")])
    report["by_source"]["C"] = binary_metrics(pool.label[take], scores[take])
    return report
