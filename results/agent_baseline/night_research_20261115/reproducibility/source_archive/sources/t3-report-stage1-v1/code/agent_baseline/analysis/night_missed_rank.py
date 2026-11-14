"""Fixed multi-source rankings with selected S5 gate and exact-mask answer reuse."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from agent_baseline.planner import Planner,usage_tokens,request_cost
from agent_baseline.tools_t13 import Features
from .io_retry import write_json,read_image
from .missed_sources import load_npz,save_npz,pack,unpack,split_components,specs,rows_for,pixel_counts
from .missed_verifier_study import ranked_candidates,parse_response,metrics
from .night_missed_candidates import SOURCES,gate_features,feature_paths,decode
from .night_missed_views import render,MODEL,CONSERVATIVE,BLUE_LEGEND

RANKINGS=('balanced','area','evidence')


def load(path):return json.loads(Path(path).read_text())


def old_answer(a,uid):
    return (a.pilot/'answers'/'four' if a.variant=='V0' else a.views/'answers'/a.variant)/(uid.replace(':','__')+'.json')


def ordering(records,kind):
    if kind=='balanced':return list(range(len(records)))
    if kind=='area':return sorted(range(len(records)),key=lambda i:(-records[i]['area'],i))
    if any(r['dino'] is None for r in records):return None
    return sorted(range(len(records)),key=lambda i:(-records[i]['evidence_score'],i))


def candidate_images(rgb,mask,base,bbox,variant):
    from agent_baseline.view import ask_images
    # V0/V3 must never enter the spotlight renderer, including new masks.
    return ask_images(rgb,mask,bbox,True) if variant in ('V0','V3') else render(rgb,mask,base,bbox,variant)[0]


def prepare(a):
    selected=load(a.gates/'score_summary.json')['selection'];items=load(a.gates/'manifest.json')
    old={(r['group'],r['stem']):r for r in load(a.pilot/'manifest.json')}
    settings=dict(version=1,variant=a.variant,s5_selection=selected,rankings=RANKINGS,k=[3,5,10,20],sources=SOURCES,
        evidence_score='DINO(1-cos) * RGBmeanRMS * (1-base_dilation3_fraction) * (1+other_source_support/4)',
        gt_role='selection inherited tune only; no labels read in candidate preparation',gate_source=str(a.gates),views=str(a.views))
    path=a.out/'settings.json'
    if path.exists() and load(path)!=json.loads(json.dumps(settings)):raise ValueError('Frozen rank settings differ')
    write_json(path,settings);manifest=[]
    for item in items:
        group,stem=item['group'],item['stem'];dest=a.out/'prepared'/group/(stem+'.json')
        if dest.exists():manifest.append(load(dest));continue
        reference=old[(group,stem)];z=load_npz(Path(item['arrays']));shape=tuple(int(v) for v in z['shape'])
        choice=selected[group]['gate'];unions={s:decode(z,'source_'+s) for s in SOURCES}
        unions['S5']=decode(z,choice)
        ranked=ranked_candidates({s:split_components(unions[s]) for s in SOURCES})
        base=load_npz(a.work/'base'/group/(stem+'.npz'))['start'];near=ndi.binary_dilation(base,structure=np.ones((7,7)))
        arg=argparse.Namespace(work=a.work,out=a.gates);dataset=group.split('_')[0]
        paths=feature_paths(arg,dataset,stem)
        features=[Features(load_npz(p)['data'],shape,t) for p,t in zip(paths,('t1','t2'))] if all(p.exists() for p in paths) else None
        rgb=[read_image(reference['image_'+t],rgb=True) for t in ('t1','t2')]
        values=gate_features([r['mask'] for r in ranked],rgb,features)
        prior=load_npz(Path(reference['masks']));prior_masks=list(unpack(prior['masks'],tuple(prior['shape'])))
        identities={np.packbits(m).tobytes():r for r,m in zip(reference['candidates'],prior_masks)}
        candidates=[]
        for index,(record,value) in enumerate(zip(ranked,values)):
            mask=record['mask'];y,x=np.nonzero(mask);area=int(mask.sum())
            agreement=sum(float((mask&unions[s]).sum()/area) for s in SOURCES if s!=record['source'])
            proximity=float((mask&near).sum()/area)
            previous=identities.get(np.packbits(mask).tobytes());uid=previous['uid'] if previous else f'{group}:{stem}:gated:{index+1}'
            evidence=value['dino']*value['rgb_mean_rms']*(1-proximity)*(1+agreement/4) if value['dino'] is not None else None
            candidates.append(dict(uid=uid,source=record['source'],sources=record['sources'],
                bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
                **value,other_source_support=agreement,base_dilation3_fraction=proximity,evidence_score=evidence,
                exact_old_candidate=previous is not None,old_answer=str(old_answer(a,uid)) if previous else None))
        ranks={kind:ordering(candidates,kind) for kind in RANKINGS}
        needed=sorted(set(i for indexes in ranks.values() if indexes is not None for i in indexes[:20]))
        full=np.logical_or.reduce([r['mask'] for r in ranked]) if ranked else np.zeros(shape,bool)
        npz=dest.with_suffix('.npz');save_npz(npz,masks=pack([ranked[i]['mask'] for i in needed],shape),shape=shape,full_union=np.packbits(full))
        remap={original:i for i,original in enumerate(needed)}
        record=dict(group=group,stem=stem,split=item['split'],s5_gate=choice,shape=shape,masks=str(npz),
            candidates=[candidates[i] for i in needed],rankings={kind:[remap[i] for i in indexes[:20]] if indexes is not None else None for kind,indexes in ranks.items()},
            pre_rank_count=len(candidates),pre_rank_area=int(np.logical_or.reduce([r['mask'] for r in ranked]).sum()) if ranked else 0,
            image_t1=reference['image_t1'],image_t2=reference['image_t2'],direction=reference['direction'],scope=reference['scope'],
            template_uid=reference['candidates'][0]['uid'] if reference['candidates'] else None,
            original_candidate_reference=reference)
        write_json(dest,record);manifest.append(record)
        write_json(a.out/'progress.json',dict(stage='prepare_rank',completed=len(manifest),expected=len(items)))
    write_json(a.out/'manifest.json',manifest)


def verify(a):
    settings=load(a.out/'settings.json')
    if settings['variant']!=a.variant:raise ValueError('Verifier differs from frozen rank policy')
    stop=a.out/'STOP.json'
    if stop.exists():
        if not a.clear_stop:raise RuntimeError('Resolve STOP before explicit resume')
        stop.rename(a.out/('previous_STOP_'+str(int(time.time()))+'.json'))
    for item in load(a.out/'manifest.json'):
        if a.split!='all' and item['split']!=a.split:continue
        z=load_npz(Path(item['masks']));masks=list(unpack(z['masks'],tuple(z['shape'])))
        rgb=None;base=None
        for candidate,mask in zip(item['candidates'],masks):
            uid=candidate['uid'];name=uid.replace(':','__');dest=a.out/'answers'/(name+'.json')
            if dest.exists():continue
            source=Path(candidate['old_answer']) if candidate['old_answer'] else None
            if source is not None and source.exists():
                old=load(source);write_json(dest,dict(old,reused_from=str(source),original_cost_cny=old.get('cost_cny'),cost_cny=0.));continue
            event_path=a.out/'usage'/(name+'.json')
            if event_path.exists() and load(event_path).get('raw'):
                event=load(event_path);write_json(dest,dict(uid=uid,**parse_response(event['raw']),cost_cny=event['cost_cny'],usage=event['usage'],recovered=True));continue
            if rgb is None:
                rgb=[read_image(item['image_'+t],rgb=True) for t in ('t1','t2')]
                base=load_npz(a.work/'base'/item['group']/(item['stem']+'.npz'))['start']
            template=load(a.pilot/'answers'/'four'/(item['template_uid'].replace(':','__')+'.json'))
            request=load(Path(template['input_path'])/'request.json');messages=json.loads(json.dumps(request['messages']))
            directory=a.out/'inputs'/name;directory.mkdir(parents=True,exist_ok=True)
            images=candidate_images(rgb,mask,base,candidate['bbox'],a.variant)
            blocks=messages[1]['content'];n=0
            for block in blocks:
                if block.get('type')=='image_url':
                    file=directory/f'{n}.jpg';images[n].save(file,quality=92);block['image_url']['url']=str(file);n+=1
            if a.variant in ('V3','V4'):blocks[-1]['text']=CONSERVATIVE+' Requested direction: '+item['direction']+'.'
            if a.variant in ('V2','V4'):blocks[-1]['text']+=BLUE_LEGEND
            parameters=dict(request['parameters']);parameters.pop('seed',None);parameters['model']=MODEL
            seed=sum(map(ord,uid))%(2**31)
            write_json(directory/'request.json',dict(messages=messages,parameters=dict(parameters,seed=seed),variant=a.variant,uid=uid))
            def capture(event):
                body=event.get('error_body');secret=os.environ.get('DASHSCOPE_API_KEY')
                if body and secret:body=body.replace(secret,'[credential omitted]')
                tokens=usage_tokens(event.get('usage'))
                write_json(event_path,dict(raw=event.get('raw'),usage=event.get('usage'),cost_cny=request_cost(MODEL,tokens),
                    http_status=event.get('http_status'),error_type=event.get('error_type'),error_body=body,model=MODEL))
                if event.get('error_type'):
                    write_json(a.out/'failures'/name/(str(int(time.time()*1000))+'.json'),load(event_path))
                    if event.get('http_status')!=429 or event.get('quota_error'):raise RuntimeError('Request failed')
            try:
                result=Planner(seed=seed,parameters=parameters,max_attempts=3,record=capture).complete(messages,[],1)
                event=load(event_path);write_json(dest,dict(uid=uid,**parse_response(result['raw']),cost_cny=event['cost_cny'],usage=event['usage'],input_path=str(directory)))
            except Exception as exc:
                write_json(stop,dict(uid=uid,error_type=type(exc).__name__,evidence=str(event_path)));raise RuntimeError('Saved request failure') from None
        costs=[load(p).get('cost_cny',0.) for p in (a.out/'usage').glob('*.json')]
        write_json(a.out/'metric.json',dict(stage='rank_verify',cost_cny=sum(costs),cumulative=True,requests=len(costs),last=item['stem']))


def union(masks,shape):
    result=np.zeros(shape,bool)
    for mask in masks:result|=mask
    return result


def score(a):
    lookup={s['group']:{Path(r['id']).stem:r for r in rows_for(a.run,s)} for s in specs(a.run,a.work) if s['dataset']!='whu'}
    records=[]
    for item in load(a.out/'manifest.json'):
        group,stem=item['group'],item['stem'];row=lookup[group][stem]
        gt=read_image(row['gt_mask'])>(127 if group.startswith('levir') else 0);base=load_npz(a.work/'base'/group/(stem+'.npz'))['start']
        z=load_npz(Path(item['masks']));masks=list(unpack(z['masks'],tuple(z['shape'])))
        old=item['original_candidate_reference'];prior=load_npz(Path(old['masks']));prior_masks=list(unpack(prior['masks'],tuple(prior['shape'])))
        full=np.unpackbits(z['full_union'],count=int(np.prod(base.shape))).reshape(base.shape).astype(bool)
        result=dict(group=group,stem=stem,split=item['split'],baseline=metrics(pixel_counts(base,gt)),rankings={},reference={},
            pre_rank_candidate_tp=int((full&gt).sum()),pre_rank_candidate_fp=int((full&~gt).sum()),
            pre_rank_fn_recall=float((full&gt&~base).sum()/(gt&~base).sum()) if (gt&~base).any() else None)
        for k in (3,5,10,20):
            pred=base.copy();missing=0
            for candidate,mask in zip(old['candidates'][:k],prior_masks[:k]):
                path=old_answer(a,candidate['uid'])
                if not path.exists():missing+=1;continue
                if load(path).get('answer')=='yes':pred|=mask
            result['reference'][str(k)]=metrics(pixel_counts(pred,gt)) if missing==0 else None
        for kind,indexes in item['rankings'].items():
            if indexes is None:result['rankings'][kind]=dict(status='missing_DINO_features');continue
            result['rankings'][kind]={}
            for k in (3,5,10,20):
                pred=base.copy();covered=np.zeros(base.shape,bool);missing=0;unknown=0
                for i in indexes[:k]:
                    mask=masks[i];covered|=mask;path=a.out/'answers'/(item['candidates'][i]['uid'].replace(':','__')+'.json')
                    if not path.exists():missing+=1;continue
                    answer=load(path).get('answer')
                    if answer=='yes':pred|=mask
                    if answer not in ('yes','no'):unknown+=1
                score=metrics(pixel_counts(pred,gt)) if missing==0 else None;reference=result['reference'][str(k)]
                result['rankings'][kind][str(k)]=dict(metrics=score,missing=missing,unknown=unknown,
                    delta_vs_initial=score['f1']-result['baseline']['f1'] if score else None,
                    delta_vs_ungated_same_verifier=score['f1']-reference['f1'] if score and reference else None,
                    candidate_tp=int((covered&gt).sum()),candidate_fp=int((covered&~gt).sum()),
                    candidate_fn_recall=float((covered&gt&~base).sum()/(gt&~base).sum()) if (gt&~base).any() else None)
        records.append(result);write_json(a.out/'scores'/group/(stem+'.json'),result)
    groups={}
    for group in sorted({r['group'] for r in records}):
        groups[group]={}
        for split in ('tune','confirm'):
            rows=[r for r in records if r['group']==group and r['split']==split];report={}
            for ranking in RANKINGS:
                report[ranking]={}
                for k in (3,5,10,20):
                    usable=[r for r in rows if r['rankings'][ranking].get(str(k),{}).get('metrics') and r['reference'][str(k)]]
                    if len(usable)!=len(rows) or not usable:
                        report[ranking][str(k)]=dict(complete=False,available=len(usable),expected=len(rows));continue
                    after=np.asarray([[r['rankings'][ranking][str(k)]['metrics'][n] for n in ('tp','fp','fn')] for r in rows])
                    before=np.asarray([[r['reference'][str(k)][n] for n in ('tp','fp','fn')] for r in rows])
                    rng=np.random.default_rng(1016);delta=[]
                    for _ in range(2000):
                        ids=rng.integers(0,len(rows),len(rows));delta.append(metrics(after[ids].sum(axis=0))['f1']-metrics(before[ids].sum(axis=0))['f1'])
                    report[ranking][str(k)]=dict(complete=True,metrics=metrics(after.sum(axis=0)),
                        delta_vs_ungated_same_verifier=metrics(after.sum(axis=0))['f1']-metrics(before.sum(axis=0))['f1'],paired_image_bootstrap95=np.quantile(delta,[.025,.975]).tolist())
            groups[group][split]=report
    write_json(a.out/'score_summary.json',dict(groups=groups,images=records,verifier=a.variant,sources_preserved=list(SOURCES),
        interpretation='S5 gate plus ranking at frozen verifier; comparison uses equal K and same verifier on old ungated balanced pool'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','score'])
    for key in ('out','gates','views','pilot','work','run'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant',choices=['V0','V1','V2','V3','V4'],required=True)
    p.add_argument('--split',choices=['tune','confirm','all'],default='all');p.add_argument('--clear-stop',action='store_true')
    a=p.parse_args();globals()[a.command](a)

if __name__=='__main__':main()
