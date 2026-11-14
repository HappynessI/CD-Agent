"""Paths and small data helpers shared by the offline OVCD review tools."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path('/guisongxia01/pangchao/wangyihan')
REPO = WORKSPACE / 'wyh/CD-Agent'
RUN = WORKSPACE / 'wyh/exp6_staging/ovcd_v0_run'
DATA = WORKSPACE / 'datasets/LEVIR-CD'
STATES = ('M0', 'A', 'B', 'C')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_image(path, mode='RGB'):
    with Image.open(path) as im:
        return np.array(im.convert(mode))


def image_f1(pred, gt):
    """Same per-image convention as ovcd_v0/common.py; empty/empty is 1."""
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def call_records(steps):
    """Expand the agent's combined T3/T5 step into individual model calls.

    F1 belongs to the entire recorded step: there is no measured intermediate
    mask between verification and box generation. C/T6 has no model call.
    """
    calls = []
    for step in steps:
        if step.get('raw') is None and step.get('p_yes') is None:
            continue
        row = {k: step.get(k) for k in (
            'step', 'stage', 'uid', 'bbox', 'action', 'op', 'raw', 'p_yes',
            'f1_before', 'f1_after', 'calls_used')}
        row['kind'] = 'T3'
        row['images'] = step['images']
        row['f1_scope'] = '整步（含 T3 + T5）' if step.get('box_raw') is not None else '本步'
        calls.append(row)
        if step.get('box_raw') is not None:
            calls.append(dict(row, kind='T5', raw=step['box_raw'], p_yes=None))
    return calls
