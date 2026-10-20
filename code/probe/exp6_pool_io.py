"""Experiment 6 piece-pool locations and mask reader (numpy only, any conda env)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import OUT

POOL = OUT / "exp6/pool"
# Detector whose 0.5 mask is corrected, per pool dataset (LEVIR/WHU: main detectors of exp1-3).
DETECTOR = {"second": "mambascd_base", "second_cv": "mambascd_base", "sysu": "schanger_small",
            "second_scannet": "scannet", "levir": "schanger_base", "whu": "schanger_base",
            "lsscd": "mambascd_base", "scscd7": "mambascd_base"}


def load_piece_masks(dataset, stem):
    """Return {piece_id: (bbox [x0,y0,x1,y1), local boolean mask)}; no pickle."""
    path = POOL / dataset / "masks" / (Path(stem).stem + ".npz")
    with np.load(path, allow_pickle=False) as z:
        data, offsets = z["packed"], z["offsets"]
        masks = {}
        for index, piece_id in enumerate(z["piece_ids"]):
            shape = tuple(map(int, z["shapes"][index]))
            local = np.unpackbits(data[offsets[index]:offsets[index + 1]],
                                  count=shape[0] * shape[1]).reshape(shape).astype(bool)
            masks[int(piece_id)] = (z["bboxes"][index].tolist(), local)
        return masks
