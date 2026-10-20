"""Paths, fold manifests and the experiment 6a time budget."""
from __future__ import annotations

import json
import math
from pathlib import Path

WORKSPACE = Path('/guisongxia01/pangchao/wangyihan')
OUT = WORKSPACE / 'Chang-Agent/outputs/CD_1016_probe'
CV = OUT / 'exp6/second_cv'
SECOND = WORKSPACE / 'datasets/lim1/SECOND'
VENDOR = WORKSPACE / 'Vlm_vs_OneJev/vendor/ChangeMamba'
CFG = VENDOR / 'changedetection/configs/vssm1/vssm_base_224.yaml'
INIT = WORKSPACE / 'models/VMamba/vssm_base_0229_ckpt_epoch_237.pth'
PRED = OUT / 'preds/second_cv/mambascd_base'
DIRS = dict(T1='img1', T2='img2', GT_T1='slabel1', GT_T2='slabel2', GT_CD='label_unch0chi255')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_fold(root, index):
    manifest = json.loads((Path(root) / 'manifest.json').read_text())
    if not 0 <= index < manifest['k']:
        raise ValueError(f'fold must be in [0, {manifest["k"] - 1}]')
    fold = manifest['folds'][index]
    train, val, heldout = (set(fold[key]) for key in ('train_images', 'val_images', 'heldout_images'))
    if train & val or train & heldout or val & heldout:
        raise ValueError('Fold train/validation/heldout images overlap')
    if train | val | heldout != set(manifest['all_images']):
        raise ValueError('Fold does not cover SECOND train')
    return manifest, fold


def budget_seconds(manifest, timing, max_iters):
    """Full schedule estimate with reserve; do not assume early stopping."""
    k = manifest['k']
    val_images = sum(len(f['val_images']) for f in manifest['folds'])
    train = k * max_iters * timing['mean_step_seconds']
    validation = math.ceil(max_iters / 500) * val_images * timing['validation_seconds_per_image']
    inference = len(manifest['all_images']) * timing['validation_seconds_per_image']
    cold_overhead = max(0, timing['warmup_seconds'] - timing['warmup_steps'] * timing['mean_step_seconds'])
    startup = k * (timing['startup_seconds'] + cold_overhead)
    # Checkpoint I/O is not benchmarked: reserve 20% for it and runtime variation.
    return 1.2 * (train + validation + inference + startup)
