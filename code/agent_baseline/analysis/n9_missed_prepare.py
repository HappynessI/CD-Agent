"""N9 restartable feature fill and N8 physical reconstruction; no label inputs."""
import argparse
import json
import os
from pathlib import Path
import time
from .n9_missed_assets import OUT
from .n8_missed_assets import read, RUN, CLASSES
from .n8_missed_pipeline import cost_total


def atomic_write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); temp.replace(path)


def fill(a):
    import numpy as np
    import torch
    from agent_baseline.t13_worker import DynamicBackend
    from .io_retry import read_image
    if not torch.cuda.is_available(): raise RuntimeError('Feature fill requires GPU Slurm')
    assets=read(a.out/'asset_inventory.json')['assets']
    previous=read(a.out/'asset_fill.json') if (a.out/'asset_fill.json').exists() else {}
    completed=list(previous.get('completed',[])); started=time.monotonic()
    prior_elapsed=previous.get('elapsed_seconds', previous.get('prior_elapsed_seconds',0)+previous.get('attempt_seconds',0))
    pending=[r for r in assets if any(not Path(r[k]).is_file() for k in ('semantic','features'))]
    if pending:
        runtime=a.out/'model_runtime'; runtime.mkdir(exist_ok=True)
        backend=DynamicBackend(read(RUN/'t3x/vision.json')['dynamic'],runtime)
    for row in pending:
        needs=[k for k in ('semantic','features') if not Path(row[k]).is_file()]
        image=read_image(row['image'],rgb=True)
        for key in needs:
            began=time.monotonic()
            with torch.inference_mode(): value=backend.semantic(image,CLASSES) if key=='semantic' else backend.features(image)
            if key=='semantic':
                assert value.shape==(7,*image.shape[:2]) and np.isfinite(value).all()
                assert value.min()>=0 and value.max()<=1 and np.allclose(value.sum(axis=0,dtype=np.float32),1.,atol=.002)
            dest=Path(row[key]); dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_suffix('.partial.npz')
            np.savez_compressed(temporary,data=value); temporary.replace(dest)
            completed.append(dict(image=row['image'],kind=key,path=str(dest),seconds=time.monotonic()-began,
                                  shape=list(value.shape),slurm_job=os.environ['SLURM_JOB_ID']))
            atomic_write(a.out/'asset_fill.json',dict(status='running',completed=completed,
                attempt_seconds=time.monotonic()-started,prior_elapsed_seconds=prior_elapsed))
    elapsed=prior_elapsed+time.monotonic()-started
    atomic_write(a.out/'asset_fill.json',dict(status='complete',completed=completed,elapsed_seconds=elapsed,
        available_images=len(assets),slurm_job=os.environ['SLURM_JOB_ID']))
    atomic_write(a.out/'metric.json',dict(stage='fill_assets',status='complete',completed=len(completed),
        cost_cny=cost_total(a.out),cumulative=True))


def prepare(a):
    from . import n8_missed_prepare
    # N8 writes each NPZ fully before its JSON record. Atomic completion markers
    # preserve that per-image restart contract in N9 without editing N8 sources.
    original=n8_missed_prepare.write
    n8_missed_prepare.write=atomic_write
    try: n8_missed_prepare.prepare(a)
    finally: n8_missed_prepare.write=original


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('fill','prepare'))
    p.add_argument('--out',type=Path,default=OUT); a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    globals()[a.command](a)

if __name__=='__main__': main()
