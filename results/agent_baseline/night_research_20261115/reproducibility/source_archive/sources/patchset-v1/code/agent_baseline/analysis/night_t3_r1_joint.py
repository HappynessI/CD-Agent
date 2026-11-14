"""Fixed R1 geometry pool: cached delta Top8 versus predefined equal endpoint/delta Top8."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.tools_t13 import Features,pooled,pool_features
from agent_baseline.instances import from_dense
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,group
from .t3_routes_cache import CachedVision,read_npz,write_npz
from .night_t3_endpoint import representations
from .night_t3_retrieval import score,summaries


def joint_order(candidate1,candidate2,example1,example2):
    values=representations(candidate1,candidate2,example1,example2)
    order=sorted(range(len(candidate1)),key=lambda i:(-float(values['equal_joint'][i]),i))
    return order,{k:values[k] for k in ('delta','endpoint','equal_joint')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm'],default='tune');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute cached features through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));out=a.night/'r1_endpoint'/a.split;source=a.night/'retrieval/B0/tasks';records=[]
    settings=dict(code=str(Path(__file__).resolve()),split=a.split,k=8,source=str(source),model_calls=0,api_cost_cny=0,
        pool='exact existing prepared.r1 geometry pool; no new gate or exemplar-overlap removal',
        baseline='exact cached order1 delta Top8 and original R1 counts',
        joint='0.5*cos(candidateT2-candidateT1,exampleT2-exampleT1) + 0.25*(cos(candidateT1,exampleT1)+cos(candidateT2,exampleT2))',
        policy='predefined equal delta/endpoint weights; tune30 first; confirm only after tuning evidence; no direct4 verifier',
        reference='historical fullpool equal_joint@8 is different-pool context, not the strict scoring contrast')
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Existing R1 joint settings differ')
    write_json(out/'settings.json',settings);cache=CachedVision(a.run/'t3_diag/cache/vision')
    for row in load_dev(a.run):
        if audit['split'][row['id']]!=a.split:continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        saved=read_npz(source/row['id']/'prepared.npz');masks=saved['r1'].astype(bool);baseline_indices=saved['order1'].tolist()[:8]
        ex=read_image(row['exemplar_mask'])>0;ranking=folder/'ranking.npz'
        if exists(ranking):order=read_npz(ranking)['order_joint'].tolist()
        else:
            backend=cache.for_row(row);features=[Features(backend('extract_features',time=t),ex.shape,t) for t in ('t1','t2')]
            if len(masks):
                items=from_dense(masks,np.ones(len(masks)));indices=np.flatnonzero(ex)
                candidates=[pooled(f,items) for f in features];examples=[pool_features(f,indices) for f in features]
                order,values=joint_order(*candidates,*examples)
            else:order=[];values={k:np.zeros(0) for k in ('delta','endpoint','equal_joint')}
            write_npz(ranking,order_joint=np.asarray(order,int),**values)
            write_json(folder/'ranking_provenance.json',dict(source_prepared=str(source/row['id']/'prepared.npz'),candidates=len(masks),
                cached_delta_top8=baseline_indices,recomputed_delta_top8=sorted(range(len(masks)),key=lambda i:(-float(values['delta'][i]),i))[:8]))
        prior=json.loads(read_text(source/row['id']/'metrics.json'))
        baseline=next(r for r in prior['scores'] if r['method']=='R1_geometry_dino' and r['k']==8)
        truth=read_image(row['gt_mask'])>0
        combined=score(masks,order[:8],truth,ex,'R1_equal_joint',8)
        endpoint=json.loads(read_text(a.night/'endpoint/tasks'/row['id']/'metrics.json'))
        full=next(r for r in endpoint['scores'] if r['method']=='equal_joint' and r['k']==8)
        full=dict(full,method='fullpool_equal_joint_reference')
        record=dict(id=row['id'],split=a.split,group=group(row),remaining_gt_pixels=prior['remaining_gt_pixels'],scores=[baseline,combined,full],
            geometry_proposal_upper=prior['geometry_proposal_upper'],verification={},
            baseline_candidate_ids=baseline_indices,joint_candidate_ids=order[:8],
            paired_counts_delta=dict(zip(('tp','fp','fn'),[new-old for new,old in zip(combined['counts'],baseline['counts'])])),api_cost_cny=0)
        write_json(final,record);records.append(record)
        write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),model_calls=0,api_cost_cny=0))
        print('R1 joint',a.split,len(records),row['id'],flush=True)
    write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),model_calls=0,api_cost_cny=0))


if __name__=='__main__':main()
