"""Development-only geometry/colour filtering before DINO ranking; no model/API calls."""
import argparse
from pathlib import Path
import numpy as np
from agent_baseline.tools_t13 import Features
from .t3_routes_runtime import load_dev, read_image, group, pooled, pixel_counts, GRID, GROUPS
from .t3_routes_cache import CachedVision
from .probe_t3 import task_scores
from .io_retry import write_json


def descriptor(mask, images):
    y,x=np.nonzero(mask)
    if not len(x): return None
    return (len(x), (x.max()-x.min()+1)/(y.max()-y.min()+1),
            np.concatenate([im[mask].mean(axis=0)/255 for im in images]))


def eligible(masks, exemplar, images):
    ref=descriptor(exemplar,images)
    if ref is None: return []
    kept=[]
    for i,m in enumerate(masks):
        d=descriptor(m,images)
        if d is None: continue
        # Broad predeclared gates; labels are not consulted.
        if .25<=d[0]/ref[0]<=4 and .333333<=d[1]/ref[1]<=3 and np.linalg.norm(d[2]-ref[2])/np.sqrt(6)<=.25:
            kept.append(i)
    return kept


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Use a new output file')
    rows=load_dev(args.run);cache=CachedVision(args.run/'t3_diag/cache/vision');records=[]
    for row in rows:
        backend=cache.for_row(row);ex=read_image(row['exemplar_mask'])>0
        images=[read_image(row[k]) for k in ('image_t1','image_t2')]
        masks=np.concatenate([backend('segment_everything',time=t) for t in ('t1','t2')])
        features=[Features(backend('extract_features',time=t),ex.shape,t) for t in ('t1','t2')]
        similarity=task_scores(masks,*features,dict(instances=[np.flatnonzero(ex).tolist()],exemplar_index=0))
        kept=eligible(masks,ex,images)
        truth=read_image(row['gt_mask'])>0
        for variant,indices in [('dino',range(len(masks))),('geometry_colour',kept)]:
            ranked=sorted(indices,key=lambda i:(-float(similarity[i]),i))
            for k in GRID:
                chosen=ranked[:k]; pred=np.any(masks[chosen],axis=0) if chosen else np.zeros_like(ex)
                records.append(dict(id=row['id'],group=group(row),variant=variant,k=k,available=len(ranked),
                    counts=pixel_counts(pred,truth,ex)))
        print(row['id'],len(masks),len(kept),flush=True)
    summary={}
    for variant in ('dino','geometry_colour'):
        summary[variant]={str(k):{g:pooled([r for r in records if r['variant']==variant and r['k']==k and (g=='全部' or r['group']==g)]) for g in ('全部',*GROUPS)} for k in GRID}
    write_json(args.output,dict(policy=dict(area_ratio=[.25,4],aspect_ratio=[1/3,3],colour_rms=.25),records=records,summary=summary,api_cost_cny=0))

if __name__=='__main__': main()
