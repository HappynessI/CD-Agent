"""Slurm-only equivalence check on a completed LEVIR cache; no source writes."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
from .night_missed_candidates import visual_instances,rematch_policies,decode
from .missed_sources import load_npz
from .io_retry import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--gates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stem',default='train_106');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    group='levir_any';base=load_npz(a.work/'base'/group/(a.stem+'.npz'))['start']
    saved=load_npz(a.gates/'prepared'/group/(a.stem+'.npz'))
    started=time.monotonic();found,count=visual_instances(a.work,group,a.stem,base.shape)
    read_seconds=time.monotonic()-started;started=time.monotonic()
    results=rematch_policies(found,base,'any');elapsed=time.monotonic()-started
    checks={}
    for key,mask in results.items():
        target='baseline' if key=='old_025' else key
        reference=decode(saved,target)
        checks[key]=dict(equal=bool(np.array_equal(mask,reference)),different_pixels=int(np.count_nonzero(mask!=reference)))
    report=dict(stem=a.stem,group=group,visual_files=count,instances={k:len(v) for k,v in found.items()},
        read_seconds=read_seconds,shared_match_seconds=elapsed,checks=checks,
        reference='Completed original four-independent-match prepared NPZ; old_025 compared to historical cached S5',
        no_gt_read=True,no_cache_modified=True)
    write_json(a.output,report)
    if not all(v['equal'] for v in checks.values()):raise RuntimeError('Recorded mismatch; do not deploy optimization')


if __name__=='__main__':main()
