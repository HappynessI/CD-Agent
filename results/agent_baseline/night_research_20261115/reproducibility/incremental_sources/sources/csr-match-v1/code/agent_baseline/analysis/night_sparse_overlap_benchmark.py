"""One existing LEVIR case; stage timing, exact pair/mask checks, no API/GT."""
import argparse
import os
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
from agent_baseline import instances
from .night_sparse_overlap import sparse_overlaps
from .night_missed_candidates import visual_instances,matched_masks,rematch,MATCH_POLICIES
from .missed_sources import load_npz
from .io_retry import write_json


def main():
    p=argparse.ArgumentParser()
    for key in ('work','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stem',choices=['train_106','train_124'],default='train_106')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    group='levir_any';result=dict(stem=a.stem,group=group,api_requests=0,gt_read=False,
        limitation='One-case timing; loading and matching separated, no claim of overall speedup or matched cache state')
    def progress(stage,values=None):
        if values:result.update(values)
        result.update(stage=stage,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(a.out/(a.stem+'.json'),result)
    progress('loading_base')
    base=load_npz(a.work/'base'/group/(a.stem+'.npz'))['start']
    stats={};progress('loading_visual');started=perf_counter()
    found,count=visual_instances(a.work,group,a.stem,base.shape,stats)
    progress('loaded_visual',dict(load_decode_seconds=perf_counter()-started,decode_stats=stats,files=count,
        instance_counts={k:len(v) for k,v in found.items()}))
    # Reference computed once, not four times. Both use this identical loaded set.
    progress('reference_pairs');started=perf_counter();reference=instances.overlaps(found['t1'],found['t2'])
    progress('reference_pairs_complete',dict(reference_pair_seconds=perf_counter()-started,reference_pairs=len(reference)))
    stats={};candidate=sparse_overlaps(found,base.shape,stats,progress)
    same=reference==candidate;progress('pair_comparison',dict(csr=stats,pairs_equal=same))
    checks={};started=perf_counter()
    for direction in ('any','appeared','disappeared'):
        expected=matched_masks(found,base,direction,reference);actual=matched_masks(found,base,direction,candidate)
        for key in expected:checks[direction+'/'+key]=int(np.count_nonzero(expected[key]!=actual[key]))
    progress('complete',dict(mask_difference_pixels=checks,rule_seconds=perf_counter()-started,
        all_equal=same and all(v==0 for v in checks.values())))
    if not result['all_equal']:raise RuntimeError('Saved pair/mask mismatch; do not enable CSR')


if __name__=='__main__':main()
