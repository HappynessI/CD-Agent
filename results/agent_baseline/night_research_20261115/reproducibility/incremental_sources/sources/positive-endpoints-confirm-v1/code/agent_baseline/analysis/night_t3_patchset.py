"""Fixed R1 pool, weighted bidirectional patch-delta matching; cache-only, tune first."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.tools_t13 import Features, pool_features, cosine
from .io_retry import read_text, read_image, write_json, exists
from .t3_routes import load_dev, group
from .t3_routes_cache import CachedVision, read_npz, write_npz
from .night_t3_retrieval import score, summaries

EPS = 1e-12


def support_weights(indices, shape, grid_shape):
    """Exact align_corners bilinear mask mass used by existing pool_features."""
    h,w=shape;gh,gw=grid_shape;idx=np.asarray(indices,np.int64)
    if not len(idx): return np.zeros(gh*gw,np.float32)
    y=(idx//w)*((gh-1)/max(h-1,1));x=(idx%w)*((gw-1)/max(w-1,1))
    y0,x0=y.astype(int),x.astype(int);y1=np.minimum(y0+1,gh-1);x1=np.minimum(x0+1,gw-1)
    dy,dx=y-y0,x-x0;weights=np.zeros(gh*gw)
    for yy,xx,value in ((y0,x0,(1-dy)*(1-dx)),(y0,x1,(1-dy)*dx),(y1,x0,dy*(1-dx)),(y1,x1,dy*dx)):
        weights+=np.bincount(yy*gw+xx,weights=value,minlength=gh*gw)
    return (weights/len(idx)).astype(np.float32)


def patch_scores(delta, candidate_weights, example_weights):
    """Fixed symmetric weighted max cosine; zero-delta mass contributes zero."""
    delta=np.asarray(delta,np.float32);norm=np.linalg.norm(delta,axis=1);valid=norm>EPS
    unit=np.divide(delta,norm[:,None],out=np.zeros_like(delta),where=valid[:,None])
    example=np.flatnonzero((example_weights>0)&valid)
    normal=[];swapped=[]
    sim=np.clip(unit@unit[example].T,-1,1)
    for weights in candidate_weights:
        candidate=np.flatnonzero((weights>0)&valid)
        if not len(candidate) or not len(example): normal.append(0.);swapped.append(0.);continue
        cross=sim[candidate]
        def evaluate(matrix):
            return .5*(float(weights[candidate]@matrix.max(axis=1))+float(example_weights[example]@matrix.max(axis=0)))
        normal.append(evaluate(cross));swapped.append(evaluate(-cross))
    return np.asarray(normal),np.asarray(swapped)


def ranking(values):
    return sorted(range(len(values)),key=lambda i:(-float(values[i]),i))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm'],default='tune');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run cache computation through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));source=a.night/'retrieval/B0/tasks';out=a.night/'r1_patchset'/a.split;records=[]
    settings=dict(code=str(Path(__file__).resolve()),source=str(source),split=a.split,k=8,near_zero_norm=EPS,
        score='0.5*(sum_p wC[p]*max_q cos(deltaC[p],deltaE[q])+sum_q wE[q]*max_p cos(deltaC[p],deltaE[q]))',
        support='align_corners bilinear mask mass/area, exactly existing pool_features; all positive-mass patches, no truncation',
        zero='norm<=1e-12 vectors contribute zero; their original weight is not redistributed; no valid vectors on either side gives zero',
        pool='unchanged prepared.r1 and stable candidate indices; exemplar overlaps retained',
        policy='single predefined patch-set score; no mixing, parameter search, extra proposals, models or API; GT only after ranking',
        confirmation='If requested, already multiply exposed exploratory confirmation, never unseen holdout',
        counterfactual='swap example dates only: deltaE negated; original GT retained for ranking sensitivity, not new truth')
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Patchset settings differ')
    write_json(out/'settings.json',settings);cache=CachedVision(a.run/'t3_diag/cache/vision')
    def save():
        write_json(out/'summary.json',dict(tasks=len(records),expected=30,status='complete' if len(records)==30 else 'partial',records=records,summary=summaries(records),model_calls=0,api_cost_cny=0))
    for row in load_dev(a.run):
        if audit['split'][row['id']]!=a.split:continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        prepared=read_npz(source/row['id']/'prepared.npz');masks=prepared['r1'].astype(bool);base_order=prepared['order1'].tolist()[:8]
        ex=read_image(row['exemplar_mask'])>0;rankfile=folder/'ranking.npz'
        if not exists(rankfile):
            backend=cache.for_row(row);features=[Features(backend('extract_features',time=t),ex.shape,t) for t in ('t1','t2')]
            if features[0].data.shape!=features[1].data.shape:raise ValueError('Two-date feature grids differ')
            grid=features[0].data.shape[:2];exidx=np.flatnonzero(ex)
            indices=[np.flatnonzero(mask) for mask in masks]
            weights=[support_weights(idx,ex.shape,grid) for idx in indices];ew=support_weights(exidx,ex.shape,grid)
            delta=(features[1].data.astype(np.float32)-features[0].data.astype(np.float32)).reshape(-1,features[0].data.shape[-1])
            patch,swap=patch_scores(delta,weights,ew)
            # Retain original per-date float32 pooling order for strict delta reproduction.
            example=[pool_features(f,exidx) for f in features]
            difference=np.asarray([pool_features(features[1],idx)-pool_features(features[0],idx) for idx in indices],np.float32).reshape(len(indices),delta.shape[1])
            original=cosine(difference,example[1]-example[0])
            reproduced=ranking(original)[:8]
            write_json(folder/'ranking_provenance.json',dict(cached_delta_top8=base_order,recomputed_delta_top8=reproduced,
                delta_top8_equal=reproduced==base_order,grid_shape=list(grid),candidates=len(masks),
                example_support_patches=int((ew>0).sum()),example_nonzero_delta_mass=float(ew[np.linalg.norm(delta,axis=1)>EPS].sum())))
            if reproduced!=base_order:raise ValueError('Original delta Top8 reproduction failed; inspect provenance')
            write_npz(rankfile,delta=original,patchset=patch,swapped_example=swap,
                order_patchset=np.asarray(ranking(patch),int),order_swapped=np.asarray(ranking(swap),int))
        ranks=read_npz(rankfile);prior=json.loads(read_text(source/row['id']/'metrics.json'))
        baseline=next(r for r in prior['scores'] if r['method']=='R1_geometry_dino' and r['k']==8)
        previous=json.loads(read_text(a.night/'r1_endpoint'/a.split/'tasks'/row['id']/'metrics.json'))
        joint=next(r for r in previous['scores'] if r['method']=='R1_equal_joint' and r['k']==8)
        truth=read_image(row['gt_mask'])>0
        new=score(masks,ranks['order_patchset'].tolist()[:8],truth,ex,'R1_patchset',8)
        swap=score(masks,ranks['order_swapped'].tolist()[:8],truth,ex,'R1_patchset_swapped_example',8)
        record=dict(id=row['id'],split=a.split,group=group(row),remaining_gt_pixels=prior['remaining_gt_pixels'],
            scores=[baseline,joint,new,swap],geometry_proposal_upper=prior['geometry_proposal_upper'],verification={},
            delta_top8_reproduced=True,baseline_candidate_ids=base_order,patchset_candidate_ids=ranks['order_patchset'].tolist()[:8],
            swapped_candidate_ids=ranks['order_swapped'].tolist()[:8],api_cost_cny=0)
        write_json(final,record);records.append(record);save();print('patchset',a.split,len(records),row['id'],flush=True)
    save()


if __name__=='__main__':main()
