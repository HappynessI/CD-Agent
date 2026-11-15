"""CPU check: fixed agent10, whitelist config, exact cached backend reproduction (no API)."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json
from .t3_routes import load_dev
from .t3_routes_agent import sample_ten
from .t3_routes_cache import CachedVision,read_npz
from .n8_t3_common import packed_load
from .n8_t3_online_provider import proposals


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--semantic',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    rows=[json.loads(line) for line in read_text(a.tasks).splitlines() if line.strip()]
    if rows!=sample_ten(load_dev(a.run)):raise ValueError('Fixed historical agent10 differs')
    selected=json.loads(read_text(a.out/'step2/system_frozen.json'))
    config=dict(rank=selected['instance']['rank'],system_method=selected['system_method'],k=5,
        method='predeclared reconstructed pool Top5; no task-relative gate or internal VLM; agent decides subsequent verification/edit/submission')
    write_json(a.out/'online/inference_config.json',config)
    split=json.loads(read_text(a.night/'audit.json'))['split'];cache=CachedVision(a.run/'t3_diag/cache/vision');records=[]
    for row in rows:
        for field in ('image_t1','image_t2','exemplar_mask'):
            if not Path(row[field]).is_file():raise ValueError('Missing online input '+field)
        images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];ex=read_image(row['exemplar_mask'])>0
        base=cache.for_row(row);memo={}
        def backend(operation,**kwargs):
            key=(operation,kwargs.get('time'))
            if key not in memo:
                memo[key]=read_npz(a.semantic/f"{row['id']}_{kwargs['time']}.npz")['data'] if operation=='classify' else base(operation,**kwargs)
            return memo[key]
        old=read_npz(a.night/'retrieval/B0/tasks'/row['id']/'prepared.npz');indices=old['order1'].tolist()[:8]
        expected_base=(old['r1'][indices].any(axis=0) if indices else np.zeros_like(ex))&~ex
        pool,meta=packed_load(a.out/'step3/prepare'/split[row['id']]/'tasks'/row['id']/'pool.npz');ids=meta['order'].tolist()[:5]
        expected_n8=pool[ids].any(axis=0) if ids else np.zeros_like(ex)
        results={}
        for arm,expected in [('baseline',expected_base),('n8',expected_n8)]:
            masks,_,audit=proposals(images,ex,backend,arm,config);actual=masks.any(axis=0) if len(masks) else np.zeros_like(ex)
            same=bool(np.array_equal(actual,expected));results[arm]=dict(exact_mask_reproduction=same,returned_candidates=len(masks),inference=audit)
            if not same:raise ValueError(f'Provider reproduction mismatch: {row["id"]} {arm}')
        records.append(dict(id=row['id'],development_split=split[row['id']],results=results))
        write_json(a.out/'online/provider_check.json',dict(tasks=len(records),expected=10,records=records,api_calls=0,
            exposure='Same previously reviewed dev10; 7 original confirm and 3 tune. Not independent validation.'))
        print('N8 online CPU check',len(records),row['id'],flush=True)


if __name__=='__main__':main()
