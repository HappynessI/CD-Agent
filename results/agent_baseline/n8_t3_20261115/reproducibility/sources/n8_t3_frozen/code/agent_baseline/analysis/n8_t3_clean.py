"""T3-1 actual pre-TopK self-match removal and clean-support feature recomputation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,group
from .t3_routes_cache import CachedVision,read_npz
from .n8_t3_common import clean_pool,ranked_features,observable_example,packed_save,evaluate,report,RANKS,K_VALUES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--semantic',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Submit through Slurm')
    rows=load_dev(a.run);splits=json.loads(read_text(a.night/'audit.json'))['split'];out=a.out/'step1'
    cache=CachedVision(a.run/'t3_diag/cache/vision');records=[]
    write_json(out/'settings.json',dict(overlap_threshold=.15,iou_threshold=.15,min_net_area=16,K=list(K_VALUES),
        feature_policy='Recompute delta/joint on every retained net mask C minus E before sorting; original pool identities retained in mapping only.',
        whitelist=['T1 RGB','T2 RGB','E','cached DINOv2','cached seven-channel SegEarth probabilities'],
        semantic_source=str(a.semantic),forbidden='No GT/class-pair/task-ID inference; GT only scoring. No text oracle cache read.',api_calls=0))
    for row in rows:
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        prior=a.night/'retrieval/B0/tasks'/row['id'];saved=read_npz(prior/'prepared.npz')
        masks=saved['r1'].astype(bool);example=read_image(row['exemplar_mask'])>0
        clean,ids,audit=clean_pool(masks,example)
        values,orders=ranked_features(clean,example,cache.for_row(row))
        probabilities=[read_npz(a.semantic/f"{row['id']}_{t}.npz")['data'] for t in ('t1','t2')]
        observable,dense=observable_example(probabilities,example)
        observable['probability_range']=[[float(p.min()),float(p.max())] for p in probabilities]
        observable['channel_sum_max_error']=[float(np.abs(p.sum(axis=0)-1).max()) for p in probabilities]
        packed_save(folder/'clean_pool.npz',clean,original_ids=np.asarray(ids,int),**values,
                    **{'order_'+r:np.asarray(orders[r],int) for r in RANKS})
        write_json(folder/'inference.json',dict(observable=observable,cleaning=audit,
            original_candidates=len(masks),clean_candidates=len(clean),removed=len(masks)-len(clean),
            modified_retained=sum(audit[i]['modified'] for i in ids),
            feature_recomputed=True,semantic_files=[str(a.semantic/f"{row['id']}_{t}.npz") for t in ('t1','t2')]))
        # GT enters after pool/masks/features/ranks and inference observables have been persisted.
        truth=read_image(row['gt_mask'])>0
        original=json.loads(read_text(prior/'metrics.json'))
        baseline=next(r for r in original['scores'] if r['method']=='R1_geometry_dino' and r['k']==8)
        reproduced=evaluate(masks,saved['order1'].tolist()[:8],truth,example)
        if reproduced['counts']!=baseline['counts']:raise ValueError('Original R1 counts differ')
        methods={'original_R1_top8':reproduced,'clean_pool_upper':evaluate(clean,list(range(len(clean))),truth,example)}
        for rank in RANKS:
            for k in K_VALUES:methods[f'clean_{rank}_top{k}']=evaluate(clean,orders[rank][:k],truth,example)
        record=dict(id=row['id'],split=splits[row['id']],group=group(row),remaining_gt_pixels=int((truth&~example).sum()),
                    methods=methods,original_candidates=len(masks),clean_candidates=len(clean),
                    removed_candidates=len(masks)-len(clean),modified_retained=sum(audit[i]['modified'] for i in ids),
                    original_pool_upper=original['geometry_proposal_upper'])
        write_json(final,record);records.append(record)
        write_json(out/'progress.json',dict(tasks=len(records),expected=60,last=row['id']))
        print('N8 clean',len(records),row['id'],len(masks),'->',len(clean),flush=True)
    report(out,records,'T3-1 clean pool',api_calls=0)


if __name__=='__main__':main()
