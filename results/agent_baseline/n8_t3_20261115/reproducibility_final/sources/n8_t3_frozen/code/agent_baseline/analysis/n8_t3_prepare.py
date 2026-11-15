"""T3-3 reconstruct routed candidate pools, recompute scores, freeze relative gate on tune."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,pooled
from .t3_routes_cache import CachedVision,read_npz
from .n8_t3_common import packed_load,packed_save,observable_example,evaluate,report
from .n8_t3_system import construct_pool,feature_evidence,passes_gate,Z_GRID,P_GRID,VERIFY_K


def choose_gate(records):
    if any(r['split']!='tune' for r in records):raise ValueError('Gate selection requires tune only')
    grid=[]
    for z in Z_GRID:
        for p in P_GRID:
            chosen=[r['methods']['pool_top5'] if passes_gate(r['candidates'],r['z'],r['top1_ptrans'],z,p) else r['methods']['always_empty'] for r in records]
            grid.append(dict(z=z,p=p,metrics=pooled(chosen),triggered=sum(passes_gate(r['candidates'],r['z'],r['top1_ptrans'],z,p) for r in records)))
    best=max(grid,key=lambda r:(r['metrics']['f1_c'],-r['z'],-r['p']))
    return dict(selected=best,grid=grid,objective='tune pooled all-task F1 of unverified Top5; ties lower z then p',
        z_definition='(highest score - mean(other scores))/(population std(other scores)+1e-6); N0/1 z=0; empty pool returns empty',
        p_definition='mean over top candidate of P_T1(example-predicted c1) * P_T2(example-predicted c2)',
        gate_rule='accept task iff nonempty pool AND z >= threshold AND top1 ptrans >= threshold')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--night',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['tune','confirm'],required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    frozen=json.loads(read_text(a.out/'step2/system_frozen.json'));rank=frozen['instance']['rank']
    previous=json.loads(read_text(a.out/'step1/summary.json'))['records'];prior={r['id']:r for r in previous if r['split']==a.split}
    out=a.out/'step3/prepare'/a.split;cache=CachedVision(a.run/'t3_diag/cache/vision');records=[]
    for row in load_dev(a.run):
        if row['id'] not in prior:continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        clean,_=packed_load(a.out/'step1/tasks'/row['id']/'clean_pool.npz');ex=read_image(row['exemplar_mask'])>0
        probabilities=[read_npz(a.semantic/f"{row['id']}_{t}.npz")['data'] for t in ('t1','t2')]
        observable,dense=observable_example(probabilities,ex)
        masks,sources=construct_pool(clean,dense,probabilities,observable,frozen['system_method'])
        values,order,candidates,relative=feature_evidence(masks,ex,probabilities,observable,cache.for_row(row),rank)
        original=read_npz(a.night/'retrieval/B0/tasks'/row['id']/'prepared.npz')['r1'].astype(bool)
        original_keys={np.packbits(mask).tobytes() for mask in original}
        new_flags=[np.packbits(mask).tobytes() not in original_keys for mask in masks]
        packed_save(folder/'pool.npz',masks,order=np.asarray(order,int),**values)
        inference=dict(observable=observable,rank=rank,system_method=frozen['system_method'],relative=relative,
            candidates=candidates,sources=sources,order=order,new_mask_vs_original_R1=new_flags,
            new_candidates=sum(new_flags),candidate_count=len(masks),
            construction='Full clean instance pool for structural route; all 8-connected dense components for nonstructural route; hybrid union adds high-confidence dense components, intersection transforms each clean instance. Empty/exact duplicates only removed; all component sizes retained.',
            whitelist=['T1','T2','E','DINO features','predicted SegEarth probabilities'])
        write_json(folder/'inference.json',inference)
        truth=read_image(row['gt_mask'])>0
        methods={'original_R1_top8':prior[row['id']]['methods']['original_R1_top8'],
                 'always_empty':evaluate(masks,[],truth,ex),'pool_upper':evaluate(masks,range(len(masks)),truth,ex)}
        for k in VERIFY_K:methods[f'pool_top{k}']=evaluate(masks,order[:k],truth,ex)
        record=dict(id=row['id'],split=a.split,group=prior[row['id']]['group'],remaining_gt_pixels=prior[row['id']]['remaining_gt_pixels'],methods=methods,
            candidates=len(masks),new_candidates=sum(new_flags),z=relative['z'],top1_ptrans=candidates[order[0]]['ptrans'] if order else 0,
            branch='structural' if observable['structural'] else 'surface')
        write_json(final,record);records.append(record);print('N8 rebuilt pool',a.split,len(records),row['id'],len(masks),'new',sum(new_flags),flush=True)
    if a.split=='tune':write_json(a.out/'step3/gate_frozen.json',choose_gate(records))
    gate=json.loads(read_text(a.out/'step3/gate_frozen.json'))
    for record in records:
        triggered=passes_gate(record['candidates'],record['z'],record['top1_ptrans'],gate['selected']['z'],gate['selected']['p'])
        record['triggered']=triggered
        for k in VERIFY_K:record['methods'][f'gate_top{k}']=record['methods'][f'pool_top{k}'] if triggered else record['methods']['always_empty']
        write_json(out/'tasks'/record['id']/'metrics.json',record)
    report(out,records,'T3-3 rebuilt pool '+a.split,gate=gate,api_calls=0)


if __name__=='__main__':main()
