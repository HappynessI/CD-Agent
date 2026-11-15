"""FM-1 full-mask reconstruction and tune-only semantic calibration."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from agent_baseline.tools_t13 import Features,pool_features,cosine
from .n8_missed_assets import read,write,OUT,RUN,WORK,SOURCES
from .missed_sources import specs,rows_for,pixel_counts
from .missed_verifier_study import metrics

SEM_GRID=(0.,.02,.05,.10,.20)
AREA_POWERS=(0.,.5,1.)
RESIDUAL_GRID=(25,256,1024)

def union(masks,shape):
    out=np.zeros(shape,bool)
    for m in masks:out|=m
    return out

def components(mask,min_area=1):
    labels,n=ndi.label(mask,structure=np.ones((3,3)))
    areas=np.bincount(labels.ravel())
    return [labels==i for i in range(1,n+1) if areas[i]>=min_area]

def load_union(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(z['shape']);packed=z['masks']
        if not len(packed):return np.zeros(shape,bool)
        return np.unpackbits(np.bitwise_or.reduce(packed,axis=0),count=int(np.prod(shape))).reshape(shape).astype(bool)

def load_masks(path):
    with np.load(path,allow_pickle=False) as z:
        shape=tuple(z['shape']);packed=z['masks']
        return np.unpackbits(packed,axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)

def save_masks(path,masks,shape,**kwargs):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    packed=np.packbits(np.asarray(masks,bool).reshape(len(masks),int(np.prod(shape))),axis=1) if len(masks) else np.zeros((0,(int(np.prod(shape))+7)//8),np.uint8)
    np.savez_compressed(path,masks=packed,shape=shape,**kwargs)

def directional_delta(p1,p2,direction):
    if direction in ('appeared','build'):return p2-p1
    if direction in ('disappeared','demolish'):return p1-p2
    if direction=='any':return np.abs(p2-p1)
    raise ValueError(direction)

def active_sources(direction):
    return ('S1_02','S2','S5') if direction in ('disappeared','demolish') else SOURCES

def feature_difference(mask,features):
    idx=np.flatnonzero(mask)
    return float(1-cosine(pool_features(features[0],idx),pool_features(features[1],idx)))

def reconstruct(source_masks,base,direction,probabilities,features):
    """Only inference-observable inputs; no labels or source-dataset argument."""
    y,x=np.ogrid[-3:4,-3:4]
    shape=base.shape;near=ndi.binary_dilation(base,structure=x*x+y*y<=9)
    enabled=active_sources(direction);delta=directional_delta(*probabilities,direction)
    selected={};s5_seen=0;s5_kept=0
    for source in SOURCES:
        if source=='S5':
            raw=components(source_masks[source]);s5_seen=len(raw)
            kept=[m for m in raw if feature_difference(m,features)>=.20];s5_kept=len(kept)
            selected[source]=union(kept,shape)
        else:selected[source]=source_masks[source]
    routed=union([selected[s] for s in enabled],shape)
    by_source={s:components(selected[s]&~near,25) for s in enabled}
    candidates=[];seen={}
    for source in enabled:
        for mask in by_source[source]:
            identity=np.packbits(mask).tobytes()
            if identity in seen:
                candidates[seen[identity]]['sources'].append(source);continue
            idx=np.flatnonzero(mask);y,x=np.nonzero(mask)
            record=dict(index=len(candidates),sources=[source],area=len(idx),bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
                semantic_delta=float(delta[mask].mean()),p1=float(probabilities[0][mask].mean()),p2=float(probabilities[1][mask].mean()),
                dino=feature_difference(mask,features),mask=mask)
            seen[identity]=len(candidates);candidates.append(record)
    stages=dict(original_all=union(source_masks.values(),shape),s5_dino020_all=union(selected.values(),shape),
                direction_routed=routed,dilation_only=routed&~near,area25=union([r['mask'] for r in candidates],shape))
    return candidates,stages,dict(enabled_sources=list(enabled),s5_before=s5_seen,s5_after_dino020=s5_kept,
                                 source_components_after_dilation25={s:len(v) for s,v in by_source.items()})

def prepare(a):
    started=time.monotonic();manifest=[]
    for item in read(a.out/'asset_manifest.json'):
        path=a.out/'fm1'/item['group']/(item['stem']+'.json')
        if path.exists():manifest.append(read(path));continue
        with np.load(item['base'],allow_pickle=False) as z:base=z['start'].astype(bool)
        sources={s:load_union(item['sources'][s]) for s in SOURCES}
        probs=[];features=[]
        for t in ('t1','t2'):
            with np.load(item['semantic'][t],allow_pickle=False) as z:
                probability=z['data']
                assert probability.shape==(7,*base.shape), 'Require background plus six aligned channels'
                assert np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1, 'Require probability values, not logits'
                assert np.allclose(probability.sum(axis=0,dtype=np.float32),1.,atol=.002), 'Require channel-normalized probabilities'
                probs.append(probability[5].astype(np.float32))
            with np.load(item['features'][t],allow_pickle=False) as z:features.append(Features(z['data'],base.shape,t))
        candidates,stages,audit=reconstruct(sources,base,item['direction'],probs,features)
        masks=[r.pop('mask') for r in candidates];npz=path.with_suffix('.npz')
        save_masks(npz,masks,base.shape,**{k:np.packbits(v) for k,v in stages.items()})
        record=dict(item,candidates=candidates,masks=str(npz),audit=audit)
        write(path,record);manifest.append(record)
        write(a.out/'metric.json',dict(stage='fm1_reconstruct',status='running',completed=len(manifest),cost_cny=0,cumulative=True))
    write(a.out/'fm1_manifest.json',manifest)
    write(a.out/'fm1_execution.json',dict(status='complete',tasks=len(manifest),elapsed_seconds=time.monotonic()-started,slurm_job=os.environ['SLURM_JOB_ID']))

def gt_lookup():
    return {(s['group'],Path(r['id']).stem):r['gt_mask'] for s in specs(RUN,WORK) if s['group'] in ('levir_any','second_appeared','second_disappeared') for r in rows_for(RUN,s)}

def truth(path,group):
    from .io_retry import read_image
    # Preserve the historical label reader exactly; this is offline scoring only.
    return read_image(path)>(127 if group=='levir_any' else 0)

def aggregate(rows):
    return metrics([sum(r[k] for r in rows) for k in ('tp','fp','fn')])

def semantic_select(a):
    lookup=gt_lookup();rows=[]
    for item in read(a.out/'fm1_manifest.json'):
        if item['split']!='tune':continue
        gt=truth(lookup[item['group'],item['stem']],item['group'])
        with np.load(item['base'],allow_pickle=False) as z:base=z['start'].astype(bool)
        masks=load_masks(item['masks']);record=dict(group=item['group'],stem=item['stem'],baseline=metrics(pixel_counts(base,gt)),stages={},thresholds={})
        with np.load(item['masks'],allow_pickle=False) as z:
            for key in ('original_all','s5_dino020_all','direction_routed','dilation_only','area25'):
                mask=np.unpackbits(z[key],count=gt.size).reshape(gt.shape).astype(bool)
                record['stages'][key]=dict(candidate_tp=int((mask&gt&~base).sum()),candidate_fp=int((mask&~gt&~base).sum()),add_all=metrics(pixel_counts(base|mask,gt)))
        for threshold in SEM_GRID:
            chosen=[m for c,m in zip(item['candidates'],masks) if c['semantic_delta']>=threshold]
            mask=union(chosen,base.shape)
            record['thresholds'][str(threshold)]=dict(candidates=len(chosen),candidate_tp=int((mask&gt&~base).sum()),candidate_fp=int((mask&~gt&~base).sum()),add_all=metrics(pixel_counts(base|mask,gt)))
        rows.append(record)
    reports=[];reference=sum(r['thresholds']['0.0']['candidate_tp'] for r in rows)
    for threshold in SEM_GRID:
        groups={g:aggregate([r['thresholds'][str(threshold)]['add_all'] for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
        tp=sum(r['thresholds'][str(threshold)]['candidate_tp'] for r in rows)
        reports.append(dict(threshold=threshold,tp_retention=tp/reference if reference else 1.,candidate_tp=tp,
            candidate_fp=sum(r['thresholds'][str(threshold)]['candidate_fp'] for r in rows),groups=groups,
            group_pooled_mean_f1=float(np.mean([v['f1'] for v in groups.values()]))))
    selected=max(reports,key=lambda r:(r['group_pooled_mean_f1'],-r['candidate_fp'],-r['threshold']))
    write(a.out/'fm1_tune_scores.json',rows)
    write(a.out/'frozen_semantic.json',dict(threshold=selected['threshold'],grid=reports,criterion='tune three-group mean add-all F1; ties fewer FP then lower threshold; TP retention reported without eligibility constraint',exposure='historically exposed development pool'))
    write(a.out/'metric.json',dict(stage='fm1_complete',status='complete',completed=30,cost_cny=0,cumulative=True,semantic_threshold=selected['threshold']))
    print('FM1 complete; semantic threshold='+str(selected['threshold']))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','semantic-select']);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    {'prepare':prepare,'semantic-select':semantic_select}[a.command](a)

if __name__=='__main__':main()
