"""Compute-node SAM3 building cache preparation and undirected T5 given masks.

Use the existing HTTP worker or instantiate the same Segmenter locally in its SAM3
environment. No planner or verifier API is involved. Frozen: tile=512, pixel=.85,
instance score=.5, confidence=.1, overlap=.25 (same as LEVIR).
"""
import argparse
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
from agent_baseline.fixed_routes import RUN, run_manifest
from agent_baseline.tasks import read_tasks
from agent_baseline.backends import SegmentBackend


def local_segmenter(run):
    import sys
    sys.path[:0] = ['/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD',
                    '/guisongxia01/pangchao/wangyihan/Chang-Agent/probe']
    import sam3_tools
    from agent_baseline.workers.sam3_worker import Segmenter
    return Segmenter(sam3_tools.build_processor(), sam3_tools, run)


def prepare(rows, run, worker=None, segmenter=None, shard=0, shards=1):
    from PIL import Image
    if not 0 <= shard < shards:
        raise ValueError('Require 0 <= shard < shards')
    chosen = rows[shard::shards]
    for row in chosen:
        paths = [row['image_t1'], row['image_t2']]
        backend = SegmentBackend('val', Path(paths[0]).name, worker, .85, image_paths=paths)
        for time, path in zip(('t1', 't2'), paths):
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError('This preparation expects 512 x 512 images')
            if segmenter is not None:
                segmenter.infer('val', Path(path).name, time, 'building', 512,
                                conf=.1, pixel_thr=.85, image_path=path)
            _, meta = backend(time, 'building', 512, .5)
            print(row['id'], time, meta['backend'], flush=True)
    # Every shard produces its own count/settings files; mask directory is finalized later.
    if shards == 1:
        return run_manifest(chosen, Path(run) / 'init/second_sam3_imc_p85/val', direction='any', pixel_thr=.85)
    return run_manifest(chosen, Path(run) / f'second/prepare_shards/{shard}', direction='any', pixel_thr=.85)


def main():
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--sam3-worker')
    mode.add_argument('--local', action='store_true')
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    os.environ['AGENT_RUN'] = str(a.run_root)
    rows = read_tasks(a.tasks)
    prepare(rows, a.run_root, a.sam3_worker, local_segmenter(a.run_root) if a.local else None, a.shard, a.shards)


if __name__ == '__main__':
    main()
