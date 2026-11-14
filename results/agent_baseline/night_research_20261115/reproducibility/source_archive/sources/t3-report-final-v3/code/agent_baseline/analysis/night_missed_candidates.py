"""Bounded S5 denoising comparisons using existing caches; no model inference.

Extraction never reads GT. Evaluation consumes labels separately and freezes a
single tuning-selected gate before reporting confirmation retention.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from scipy import ndimage as ndi
from agent_baseline import instances
from agent_baseline.tools_t13 import Features,pool_features,cosine
from .io_retry import read_image,write_json
from .missed_sources import load_npz,save_npz,unpack,pack,split_components,specs,rows_for,pixel_counts
from .missed_verifier_study import metrics

SOURCES=('S1_02','S2','S3','S4','S5')
GATES=('baseline','overlap_010','overlap_050','iou_025','dilate_1','dilate_3','dino_010','dino_020','rgb_003','rgb_006','dino010_rgb003')


def load(path):return json.loads(Path(path).read_text())


def mask_union(masks,shape):
    out=np.zeros(shape,bool)
    for mask in masks:out|=mask
    return out


def frozen_rows(a):
    known=load(a.views/'settings.json')['splits'];pilot=load(a.views/'manifest.json')
    selected={(r['group'],r['stem']) for r in pilot};allrows=[]
    for spec in specs(a.run,a.work):
        if spec['dataset']=='whu':continue
        rows=rows_for(a.run,spec)
        for row in rows:allrows.append((spec,row))
    mapping=dict(known)
    for dataset in ('second','levir'):
        extra=sorted({Path(row['id']).stem for spec,row in allrows if spec['dataset']==dataset and dataset+':'+Path(row['id']).stem not in mapping})
        random.Random(1016).shuffle(extra)
        for i,stem in enumerate(extra):mapping[dataset+':'+stem]='tune' if i<len(extra)//2 else 'confirm'
    return [(s,r,mapping[s['dataset']+':'+Path(r['id']).stem]) for s,r in allrows if a.all_dev or (s['group'],Path(r['id']).stem) in selected]


def visual_instances(work,group,stem,shape):
    folder=work/'gpu_cache'/'visual'/group/stem;found={'t1':[],'t2':[]};seen={'t1':set(),'t2':set()}
    files=sorted(folder.glob('*.npz'))
    for path in files:
        time=path.stem.rsplit('_',1)[-1]
        if time not in found:continue
        z=load_npz(path)
        for mask in unpack(z['masks'],tuple(z['shape'])):
            idx=np.flatnonzero(mask).astype(np.int32);identity=idx.tobytes()
            if not len(idx) or identity in seen[time]:continue
            seen[time].add(identity);y,x=np.divmod(idx,shape[1])
            found[time].append(dict(idx=idx,area=len(idx),score=1.,bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]))
    return found,len(files)


def rematch(found,base,direction,threshold,mode):
    appeared,disappeared,_=instances.match(found['t1'],found['t2'],threshold,mode)
    chosen=appeared+disappeared if direction=='any' else appeared if direction=='appeared' else disappeared
    return instances.to_mask(chosen,base.shape)&~base


def gate_features(masks,rgb,feature_pair):
    result=[]
    for mask in masks:
        idx=np.flatnonzero(mask)
        values=[im.reshape(-1,3)[idx].astype(np.float32)/255 for im in rgb]
        mean_rms=float(np.sqrt(np.mean((values[0].mean(axis=0)-values[1].mean(axis=0))**2)))
        pixel_rms=float(np.sqrt(np.mean((values[0]-values[1])**2)))
        dino=float(1-cosine(pool_features(feature_pair[0],idx),pool_features(feature_pair[1],idx))) if feature_pair else None
        result.append(dict(area=len(idx),rgb_mean_rms=mean_rms,rgb_pixel_rms=pixel_rms,dino=dino))
    return result


def feature_paths(a,dataset,stem):
    paths=[]
    for time in ('t1','t2'):
        old=a.work/'gpu_cache'/'dynamic'/dataset/stem/f'extract_features_{time}.npz'
        paths.append(old if old.exists() else a.out/'feature_cache'/dataset/stem/f'extract_features_{time}.npz')
    return paths


def fill_features(a):
    from .missed_sources_gpu import LineWorker
    cfg=load(a.run/'t3x'/'vision.json')['dynamic']
    worker=LineWorker([cfg['python'],'-B','-m','agent_baseline.workers.missed_sources_worker','dynamic'],a.out/'feature_worker')
    records=[]
    try:
        for spec,row,split in frozen_rows(a):
            stem=Path(row['id']).stem
            for time,path in zip(('t1','t2'),feature_paths(a,spec['dataset'],stem)):
                if path.exists():continue
                worker(dict(operation='extract_features',paths=[row['image_t1'],row['image_t2']],time=time,config=cfg,output=str(path)))
                records.append(dict(group=spec['group'],stem=stem,time=time,path=str(path)))
                write_json(a.out/'feature_fill.json',dict(completed=records,cost_cny=0))
    finally:worker.close()
    write_json(a.out/'feature_fill.json',dict(completed=records,cost_cny=0,status='complete'))


def extract(a):
    rows=frozen_rows(a)
    policy=dict(gates=GATES,scope='all_development' if a.all_dev else 'same_pilot_images',threshold_policy='bounded one-factor plus one predeclared joint feature gate',
        sources=SOURCES,dino_metric='1-cosine of mean DINOv2 region features',rgb_metric='RMS of RGB mean differences normalized to [0,1]',
        dilation='square Chebyshev radius1/3 native pixels',split_source=str(a.views/'settings.json'))
    settings=a.out/'settings.json'
    if settings.exists() and load(settings)!=json.loads(json.dumps(policy)):raise ValueError('Candidate policy differs; new output required')
    write_json(settings,policy)
    manifest=[]
    for spec,row,split in rows:
        group=spec['group'];stem=Path(row['id']).stem;dest=a.out/'prepared'/group/(stem+'.json')
        if dest.exists():
            old=load(dest)
            if old['feature_cache_available'] or not all(p.exists() for p in feature_paths(a,spec['dataset'],stem)):
                manifest.append(old);continue
        base=load_npz(a.work/'base'/group/(stem+'.npz'))['start'];shape=base.shape
        sources={}
        for source in SOURCES:
            z=load_npz(a.work/'candidates'/group/source/(stem+'.npz'))
            sources[source]=list(unpack(z['masks'],tuple(z['shape'])))
        s5=sources['S5'];original=mask_union(s5,shape)
        found,files=visual_instances(a.work,group,stem,shape)
        paths=feature_paths(a,spec['dataset'],stem)
        features=[Features(load_npz(path)['data'],shape,time) for path,time in zip(paths,('t1','t2'))] if all(p.exists() for p in paths) else None
        rgb=[read_image(row['image_'+time],rgb=True) for time in ('t1','t2')]
        values=gate_features(s5,rgb,features)
        masks=dict(baseline=original,dilate_1=original&~ndi.binary_dilation(base,structure=np.ones((3,3))),
                   dilate_3=original&~ndi.binary_dilation(base,structure=np.ones((7,7))))
        status={}
        if files:
            old=rematch(found,base,spec['direction'],.25,'overlap')
            status['old_rematch_equals_cache']=bool(np.array_equal(old,original))
            for key,thr,mode in (('overlap_010',.10,'overlap'),('overlap_050',.50,'overlap'),('iou_025',.25,'iou')):
                masks[key]=rematch(found,base,spec['direction'],thr,mode)
        else:
            with np.load(a.work/'raw'/group/(stem+'.npz'),allow_pickle=False) as raw:
                seeds=int(raw['yes_seed_count'])
            status['matching']='no_verified_seeds' if seeds==0 else 'missing_visual_cache'
            if seeds==0:
                for key in ('overlap_010','overlap_050','iou_025'):masks[key]=original.copy()
        for key in ('rgb_003','rgb_006','dino_010','dino_020','dino010_rgb003'):
            if key.startswith('dino') and features is None and s5:
                status[key]='missing_feature_cache';continue
            def keep(v):
                if key=='rgb_003':return v['rgb_mean_rms']>=.03
                if key=='rgb_006':return v['rgb_mean_rms']>=.06
                if key=='dino_010':return v['dino']>=.10
                if key=='dino_020':return v['dino']>=.20
                return v['dino']>=.10 and v['rgb_mean_rms']>=.03
            masks[key]=mask_union([m for m,v in zip(s5,values) if keep(v)],shape)
        arrays=dict(shape=shape,**{key:np.packbits(mask) for key,mask in masks.items()})
        for source in SOURCES:arrays['source_'+source]=np.packbits(mask_union(sources[source],shape))
        save_npz(dest.with_suffix('.npz'),**arrays)
        record=dict(group=group,stem=stem,split=split,arrays=str(dest.with_suffix('.npz')),source_counts={s:len(v) for s,v in sources.items()},
            visual_cache_files=files,feature_cache_available=features is not None,status=status,s5_features=values,gates=list(masks))
        write_json(dest,record);manifest.append(record)
        write_json(a.out/'progress.json',dict(stage='extracting',completed=len(manifest),expected=len(rows),last=stem,cost_cny=0))
    write_json(a.out/'manifest.json',manifest)


def decode(data,key):return np.unpackbits(data[key],count=int(np.prod(data['shape']))).reshape(tuple(data['shape'])).astype(bool)


def evaluate(a):
    rows=load(a.out/'manifest.json');lookup={s['group']:{Path(r['id']).stem:r for r in rows_for(a.run,s)} for s in specs(a.run,a.work) if s['dataset']!='whu'}
    detail=[]
    for item in rows:
        group,stem=item['group'],item['stem'];row=lookup[group][stem]
        truth=read_image(row['gt_mask'])>(127 if group.startswith('levir') else 0)
        base=load_npz(a.work/'base'/group/(stem+'.npz'))['start'];data=load_npz(Path(item['arrays']))
        sources={s:decode(data,'source_'+s) for s in SOURCES};full=mask_union(sources.values(),base.shape)
        without=mask_union([m for s,m in sources.items() if s!='S4'],base.shape)
        result=dict(group=group,stem=stem,split=item['split'],baseline=metrics(pixel_counts(base,truth)),gates={},
            s4_unique_tp=int((full&~without&truth).sum()),s4_unique_fp=int((full&~without&~truth).sum()),
            all_source_tp=int((full&truth).sum()),all_source_fp=int((full&~truth).sum()),
            remaining_fn_pixels=int((truth&~base).sum()),status=item['status'])
        original=decode(data,'baseline');tp0=int((original&truth).sum());fp0=int((original&~truth).sum())
        for gate in item['gates']:
            mask=decode(data,gate);tp=int((mask&truth).sum());fp=int((mask&~truth).sum())
            result['gates'][gate]=dict(tp=tp,fp=fp,tp_retention=tp/tp0 if tp0 else None,fp_retention=fp/fp0 if fp0 else None,
                candidates=len(split_components(mask)),add_all=metrics(pixel_counts(base|mask,truth)))
        detail.append(result);write_json(a.out/'scores'/group/(stem+'.json'),result)
    summaries={};selection={}
    for group in sorted({r['group'] for r in detail}):
        summaries[group]={}
        for split in ('tune','confirm'):
            rows=[r for r in detail if r['group']==group and r['split']==split];stats={}
            base_tp=sum(r['gates']['baseline']['tp'] for r in rows);base_fp=sum(r['gates']['baseline']['fp'] for r in rows)
            for gate in GATES:
                usable=[r for r in rows if gate in r['gates']]
                tp=sum(r['gates'][gate]['tp'] for r in usable);fp=sum(r['gates'][gate]['fp'] for r in usable)
                stats[gate]=dict(images=len(usable),expected=len(rows),complete=len(usable)==len(rows),tp=tp,fp=fp,
                    matching_cache_mismatch_images=sum(r['status'].get('old_rematch_equals_cache') is False for r in usable) if gate in ('overlap_010','overlap_050','iou_025') else 0,
                    tp_retention=tp/base_tp if base_tp else None,fp_retention=fp/base_fp if base_fp else None,
                    add_all=metrics([sum(r['gates'][gate]['add_all'][key] for r in usable) for key in ('tp','fp','fn')]) if usable else None)
                if usable:
                    before=np.asarray([[r['gates']['baseline']['add_all'][k] for k in ('tp','fp','fn')] for r in usable])
                    after=np.asarray([[r['gates'][gate]['add_all'][k] for k in ('tp','fp','fn')] for r in usable])
                    rng=np.random.default_rng(1016);draws=[]
                    for _ in range(2000):
                        index=rng.integers(0,len(usable),len(usable))
                        draws.append(metrics(after[index].sum(axis=0))['f1']-metrics(before[index].sum(axis=0))['f1'])
                    stats[gate]['unverified_addall_delta_vs_s5']=metrics(after.sum(axis=0))['f1']-metrics(before.sum(axis=0))['f1']
                    stats[gate]['paired_image_bootstrap95']=np.quantile(draws,[.025,.975]).tolist()
            summaries[group][split]=stats
        # Selection is made ONLY from tune. Confirmation never changes this key.
        tune=summaries[group]['tune'];eligible=[g for g,v in tune.items() if v['complete'] and not v['matching_cache_mismatch_images'] and v['tp_retention'] is not None and v['tp_retention']>=.80]
        choice=min(eligible,key=lambda g:(tune[g]['fp'],GATES.index(g))) if eligible else 'baseline'
        selection[group]=dict(gate=choice,rule='tune TP retention >=80%, then smallest FP; ties predeclared order',
            tune=tune[choice],confirmation=summaries[group]['confirm'].get(choice))
    write_json(a.out/'score_summary.json',dict(groups=summaries,selection=selection,images=detail,
        interpretation='unverified candidate gates; add-all is not VLM or Agent performance; changed masks need fresh verification',
        no_source_pruned=True))


def whu(a):
    """Rescore historical source pools against official-v3 labels; leave history untouched."""
    from agent_baseline.evaluation import load_reference
    summary={}
    for spec in specs(a.run,a.work):
        if spec['dataset']!='whu':continue
        group=spec['group'];records=[]
        for row in rows_for(a.run,spec):
            stem=Path(row['id']).stem
            directory=a.run/'whu'/'gt_official_v3';kind={'any':'c0','appeared':'new','disappeared':'demolished'}[spec['direction']]
            gtpath=directory/kind/(stem+'.png');ignore=directory/'ignore'/(stem+'.png') if spec['direction']!='any' else None
            truth,valid=load_reference(gtpath,ignore)
            base=load_npz(a.work/'base'/group/(stem+'.npz'))['start']&valid;sources={}
            candidate_counts={}
            for source in SOURCES:
                z=load_npz(a.work/'candidates'/group/source/(stem+'.npz'))
                sources[source]=mask_union(unpack(z['masks'],tuple(z['shape'])),base.shape)&valid
                candidate_counts[source]=len(z['masks'])
            values={}
            missing=truth&~base;components=split_components(missing)
            for source,mask in sources.items():
                other=mask_union([m for s,m in sources.items() if s!=source],base.shape);unique=mask&~other
                values[source]=dict(tp=int((mask&truth).sum()),fp=int((mask&~truth).sum()),
                    candidates=candidate_counts[source],missing_pixels=int(missing.sum()),
                    missing_pixels_covered=int((mask&missing).sum()),
                    missing_components=len(components),components_covered=int(sum((m&mask).sum()>=.5*m.sum() for m in components)),
                    unique_tp=int((unique&truth).sum()),unique_fp=int((unique&~truth).sum()),
                    add_all=metrics(pixel_counts(base|mask,truth&valid)))
            record=dict(stem=stem,gt=str(gtpath),ignore=str(ignore) if ignore else None,baseline=metrics(pixel_counts(base,truth&valid)),sources=values)
            write_json(a.out/'whu_official'/group/(stem+'.json'),record);records.append(record)
        summary[group]=dict(images=len(records),records=records,sources={s:{key:sum(r['sources'][s][key] for r in records) for key in (
            'tp','fp','unique_tp','unique_fp','candidates','missing_pixels','missing_pixels_covered','missing_components','components_covered')} for s in SOURCES})
    write_json(a.out/'whu_official_summary.json',dict(gt_version='official_v3',candidate_provenance='unchanged historical batch99 candidates, historical thresholds',groups=summary))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['extract','evaluate','whu','fill_features']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--views',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--all-dev',action='store_true');a=p.parse_args();globals()[a.command](a)

if __name__=='__main__':main()
