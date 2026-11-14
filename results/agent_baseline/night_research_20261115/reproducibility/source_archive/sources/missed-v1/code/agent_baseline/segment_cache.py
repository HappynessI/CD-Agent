"""Shared explicit-image cache address used by preparation, workers and readers."""
from pathlib import Path
from urllib.parse import quote
import numpy as np
from . import instances


def manifest_cache_path(run, image_path, text, tile, conf, pixel_thr):
    source = Path(image_path).resolve()
    slug = quote(' '.join(text.lower().split()), safe='')
    return (Path(run) / 'sam3_manifest_cache' / f'{slug}_t{tile}_c{conf:g}_p{pixel_thr:g}' /
            source.relative_to(source.anchor) / 'instances.npz')


def read_instances(path, score_thr):
    with np.load(path) as z:
        h, w = map(int, z['shape'])
        keep = z['scores'] >= score_thr
        dense = np.unpackbits(z['masks'][keep], axis=1, count=h*w).reshape(-1, h, w).astype(bool)
        return instances.from_dense(dense, z['scores'][keep])
