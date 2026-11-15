"""N3 frozen-candidate V0--V4 ablation; GT enters score only."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image
from .io_retry import write_json,read_image
from .missed_sources import load_npz,unpack,pixel_counts,specs,rows_for,split_components
from .missed_verifier_study import parse_response,metrics
from agent_baseline.planner import Planner,usage_tokens,request_cost

MODEL='qwen3.7-flash'
VARIANTS=('V0','V1','V2','V3','V4')
BLUE_LEGEND=' Blue pixels mark regions already included in the initial answer, not the candidate; judge only the unoccluded red-contour candidate.'
CONSERVATIVE=('Judge only the area INSIDE the red contour, not adjacent buildings or changes outside it. '
    'Does this area show clear evidence of the requested building change between T1 and T2? '
    'A roof fragment can be part of a real change, but do not infer change from a different object nearby. '
    'Answer yes only when the two dates support that change inside the candidate; otherwise answer no. '
    'Follow the scope card and requested direction. Reply JSON only: {"change":"yes" or "no","reason":"one short evidence sentence"}.')


def load(path):return json.loads(Path(path).read_text())


def split_images(manifest):
    mapping={}
    for dataset in ('second','levir'):
        names=sorted({r['stem'] for r in manifest if r['group'].startswith(dataset+'_')})
        random.Random(1016).shuffle(names)
        for index,name in enumerate(names):mapping[dataset+':'+name]='tune' if index<len(names)//2 else 'confirm'
    return mapping


def prepare(a):
    manifest=load(a.pilot/'manifest.json');splits=split_images(manifest)
    settings=dict(version=1,pilot=str(a.pilot.resolve()),work=str(a.work.resolve()),split_seed=1016,
        split_rule='dataset-image level; SECOND directions share split',splits=splits,candidates=sum(len(r['candidates']) for r in manifest),
        variants=dict(V0='exact archived four images/prompt/answers; no API',V1='70% outside darkening; old question',
            V2='70% outside darkening, 40% blue baseline overlay; old question plus color legend',
            V3='exact archived images; conservative binary question',V4='V2 image; conservative question plus color legend'),
        outside_brightness=.30,baseline_blue_alpha=.40,model=MODEL,max_tokens=256,
        caution='Current image split controls this round selection only; historical pilot exposure is not an untouched holdout')
    path=a.out/'settings.json'
    if path.exists() and load(path)!=settings:raise ValueError('Settings differ; choose new output')
    write_json(path,settings)
    write_json(a.out/'manifest.json',[dict(r,split=splits[r['group'].split('_')[0]+':'+r['stem']]) for r in manifest])
    write_json(a.out/'metric.json',dict(stage='prepared',candidates=settings['candidates'],cost_cny=0,cumulative=True))


def selected(a):
    rows=[r for r in load(a.out/'manifest.json') if a.split=='all' or r['split']==a.split]
    if a.smoke:
        first={}
        for r in rows:first.setdefault(r['group'].split('_')[0],r['stem'])
        rows=[r for r in rows if r['stem']==first[r['group'].split('_')[0]]]
    return rows


def render(rgb,mask,base,bbox,variant):
    from render import crop_box,edge
    boxes=[crop_box(bbox,mask.shape),crop_box(bbox,mask.shape,scale=10,min_side=256)]
    images=[]
    for box in boxes:
        l,t,r,b=box
        support=np.asarray(Image.fromarray(mask[t:b,l:r]).resize((384,384),Image.Resampling.NEAREST))
        already=np.asarray(Image.fromarray((base&~mask)[t:b,l:r]).resize((384,384),Image.Resampling.NEAREST))
        outline=edge(mask[t:b,l:r])
        for image in rgb:
            original=np.asarray(Image.fromarray(image[t:b,l:r]).resize((384,384),Image.Resampling.BICUBIC),dtype=float)
            display=original.copy();display[~support]*=.30
            if variant in ('V2','V4'):
                display[already]=.60*original[already]+.40*np.asarray([0,80,255])
            display[outline]=[255,0,0]
            images.append(Image.fromarray(np.clip(display,0,255).astype(np.uint8)))
    return images,boxes


def answer_path(a,variant,uid):
    return a.pilot/'answers'/'four'/(uid.replace(':','__')+'.json') if variant=='V0' else a.out/'answers'/variant/(uid.replace(':','__')+'.json')


def summarize_usage(a,stage):
    rows=[load(p) for p in (a.out/'usage').glob('*/*.json')]
    write_json(a.out/'metric.json',dict(stage=stage,cost_cny=sum(r.get('cost_cny',0) for r in rows),
        cumulative=True,requests=len(rows),result_path=str(a.out)))


def verify(a):
    if a.variant=='V0':raise ValueError('V0 is reused, never re-requested')
    stop=a.out/'STOP.json'
    if stop.exists():
        if not a.clear_stop:raise RuntimeError('Review STOP then resume explicitly')
        stop.rename(a.out/('STOP.previous.'+str(int(time.time()))+'.json'))
    for item in selected(a):
        z=load_npz(Path(item['masks']));masks=list(unpack(z['masks'],tuple(z['shape'])))
        base=load_npz(a.work/'base'/item['group']/(item['stem']+'.npz'))['start']
        rgb=None
        if a.variant!='V3':rgb=[read_image(item['image_'+t],rgb=True) for t in ('t1','t2')]
        for candidate,mask in zip(item['candidates'],masks):
            uid=candidate['uid'];name=uid.replace(':','__');dest=answer_path(a,a.variant,uid)
            if dest.exists():continue
            usage_path=a.out/'usage'/a.variant/(name+'.json')
            if usage_path.exists() and load(usage_path).get('raw'):
                event=load(usage_path)
                write_json(dest,dict(uid=uid,**parse_response(event['raw']),cost_cny=event['cost_cny'],usage=event['usage'],recovered_response=True));continue
            original_answer=load(answer_path(a,'V0',uid))
            original_dir=Path(original_answer.get('input_path') or a.pilot/'answers'/'four'/'inputs'/name)
            old=load(original_dir/'request.json')
            messages=json.loads(json.dumps(old['messages']))
            directory=a.out/'inputs'/a.variant/name;directory.mkdir(parents=True,exist_ok=True)
            if a.variant=='V3':
                for i in range(4):shutil.copyfile(original_dir/f'{i}.jpg',directory/f'{i}.jpg')
                boxes=None
            else:
                images,boxes=render(rgb,mask,base,candidate['bbox'],a.variant)
                for i,image in enumerate(images):image.save(directory/f'{i}.jpg',quality=92)
            blocks=messages[1]['content'];number=0
            for block in blocks:
                if block.get('type')=='image_url':
                    block['image_url']['url']=str(directory/f'{number}.jpg');number+=1
            if a.variant in ('V3','V4'):
                blocks[-1]['text']=CONSERVATIVE+' Requested direction: '+item['direction']+'.'
            if a.variant in ('V2','V4'):blocks[-1]['text']+=BLUE_LEGEND
            parameters=dict(old['parameters']);parameters['model']=MODEL
            # Planner accepts seed separately; all other baseline request settings persist.
            seed=parameters.pop('seed')
            write_json(directory/'request.json',dict(messages=messages,parameters=dict(parameters,seed=seed),variant=a.variant,
                uid=uid,old_request=str(original_dir/'request.json'),crop_boxes=boxes,
                close_equals_wide=boxes[0]==boxes[1] if boxes else None))
            def record(event):
                body=event.get('error_body')
                secret=os.environ.get('DASHSCOPE_API_KEY')
                if body and secret:body=body.replace(secret,'[credential omitted]')
                tokens=usage_tokens(event.get('usage'))
                write_json(usage_path,dict(model=MODEL,raw=event.get('raw'),usage=event.get('usage'),tokens=tokens,
                    cost_cny=request_cost(MODEL,tokens),http_status=event.get('http_status'),error_type=event.get('error_type'),error_body=body))
                if event.get('error_type'):
                    write_json(a.out/'failures'/a.variant/name/(str(int(time.time()*1000))+'.json'),load(usage_path))
                    if event.get('http_status')!=429 or event.get('quota_error'):
                        raise RuntimeError('API failed; preserve response and resume explicitly')
            try:
                result=Planner(seed=seed,parameters=parameters,max_attempts=3,record=record).complete(messages,[],1)
                event=load(usage_path)
                write_json(dest,dict(uid=uid,**parse_response(result['raw']),usage=result['usage'],cost_cny=event['cost_cny'],input_path=str(directory)))
            except Exception as exc:
                write_json(stop,dict(uid=uid,variant=a.variant,error_type=type(exc).__name__,usage_path=str(usage_path)))
                summarize_usage(a,'failed_'+a.variant);raise RuntimeError('Verifier stopped; inspect saved failure') from None
        summarize_usage(a,'verified_'+a.variant+'_'+item['group']+'_'+item['stem'])
    summarize_usage(a,'complete_'+a.variant+'_'+a.split+('_smoke' if a.smoke else ''))


def score(a):
    # Label access is confined to this offline command.
    lookup={s['group']:{Path(r['id']).stem:r for r in rows_for(a.run,s)} for s in specs(a.run,a.work) if not s['group'].startswith('whu')}
    records=[]
    for item in selected(a):
        gt=read_image(lookup[item['group']][item['stem']]['gt_mask'])>(127 if item['group'].startswith('levir') else 0)
        base=load_npz(a.work/'base'/item['group']/(item['stem']+'.npz'))['start']
        z=load_npz(Path(item['masks']));masks=list(unpack(z['masks'],tuple(z['shape'])))
        row=dict(group=item['group'],stem=item['stem'],split=item['split'],baseline=metrics(pixel_counts(base,gt)),variants={})
        for variant in VARIANTS:
            outcomes=[]
            for c,m in zip(item['candidates'],masks):
                path=answer_path(a,variant,c['uid']);answer=load(path) if path.exists() else None
                outcomes.append((c,m,answer))
            budgets={}
            for k in (3,5,10,20):
                chosen=outcomes[:k];pred=base.copy();covered=np.zeros(base.shape,bool)
                errors=dict(proxy_true_yes=0,proxy_false_yes=0,proxy_true_no=0,proxy_false_no=0,
                    proxy_positive_total=0,proxy_negative_total=0,unknown=0,missing=0,truncated=0)
                for c,m,answer in chosen:
                    covered|=m
                    tp=int((m&gt).sum());fp=int((m&~gt).sum())
                    errors['proxy_positive_total' if tp>fp else 'proxy_negative_total']+=1
                    if answer is None:errors['missing']+=1;continue
                    if answer.get('finish_reason')=='length':errors['truncated']+=1
                    decision=answer.get('answer')
                    if decision=='yes':pred|=m
                    if decision not in ('yes','no'):errors['unknown']+=1
                    else:errors['proxy_'+('true' if ((tp>fp)==(decision=='yes')) else 'false')+'_'+decision]+=1
                count=metrics(pixel_counts(pred,gt)) if errors['missing']==0 else None
                budgets[str(k)]=dict(metrics=count,delta_f1=count['f1']-row['baseline']['f1'] if count else None,
                    candidate_missing_pixel_recall=float((covered&gt&~base).sum()/(gt&~base).sum()) if (gt&~base).any() else None,
                    errors=errors,criterion='TP>FP is a pixel-benefit proxy, not semantic annotation')
            row['variants'][variant]=budgets
        for variant in VARIANTS:
            for k in (3,5,10,20):
                result=row['variants'][variant][str(k)];reference=row['variants']['V0'][str(k)]['metrics']
                result['delta_vs_v0']=result['metrics']['f1']-reference['f1'] if result['metrics'] and reference else None
        records.append(row);write_json(a.out/'scores'/item['group']/(item['stem']+'.json'),row)
    summaries={}
    for group in sorted({r['group'] for r in records}):
        for split in ('tune','confirm'):
            rows=[r for r in records if r['group']==group and r['split']==split]
            if not rows:continue
            output={}
            for variant in VARIANTS:
                output[variant]={}
                for k in (3,5,10,20):
                    usable=[r for r in rows if r['variants'][variant][str(k)]['metrics']]
                    if len(usable)!=len(rows):output[variant][str(k)]=dict(complete=False,available=len(usable),expected=len(rows));continue
                    before=np.asarray([[r['baseline'][n] for n in ('tp','fp','fn')] for r in rows])
                    after=np.asarray([[r['variants'][variant][str(k)]['metrics'][n] for n in ('tp','fp','fn')] for r in rows])
                    v0_ready=all(r['variants']['V0'][str(k)]['metrics'] for r in rows)
                    v0=np.asarray([[r['variants']['V0'][str(k)]['metrics'][n] for n in ('tp','fp','fn')] for r in rows]) if v0_ready else None
                    delta=metrics(after.sum(axis=0))['f1']-metrics(before.sum(axis=0))['f1']
                    rng=np.random.default_rng(1016);boot=[];boot_v0=[]
                    for _ in range(2000):
                        index=rng.integers(0,len(rows),len(rows));boot.append(metrics(after[index].sum(axis=0))['f1']-metrics(before[index].sum(axis=0))['f1'])
                        if v0_ready:boot_v0.append(metrics(after[index].sum(axis=0))['f1']-metrics(v0[index].sum(axis=0))['f1'])
                    totals={key:sum(r['variants'][variant][str(k)]['errors'][key] for r in rows) for key in rows[0]['variants'][variant][str(k)]['errors']}
                    accepted=totals['proxy_true_yes']+totals['proxy_false_yes'];positive=totals['proxy_positive_total']
                    output[variant][str(k)]=dict(complete=True,images=len(rows),metrics=metrics(after.sum(axis=0)),delta_f1=delta,
                        paired_image_bootstrap_95_percentile=np.quantile(boot,[.025,.975]).tolist(),
                        delta_vs_v0=metrics(after.sum(axis=0))['f1']-metrics(v0.sum(axis=0))['f1'] if v0_ready else None,
                        paired_vs_v0_bootstrap95=np.quantile(boot_v0,[.025,.975]).tolist() if boot_v0 else None,
                        candidate_proxy_counts=totals,candidate_proxy_precision=totals['proxy_true_yes']/accepted if accepted else None,
                        candidate_proxy_recall=totals['proxy_true_yes']/positive if positive else None,
                        candidate_proxy_denominators=dict(precision_accepted_candidates=accepted,recall_all_tp_gt_fp_candidates=positive,
                            policy='TP>FP proxy, not semantic labels; missing/unknown are not accepted but remain in recall denominator'),
                        positive_images=sum(r['variants'][variant][str(k)]['delta_f1']>0 for r in rows))
            summaries[group+'/'+split]=output
    write_json(a.out/('score_smoke.json' if a.smoke else 'score_summary.json'),dict(groups=summaries,images=records,
        confidence_interval='paired image bootstrap, 2000 draws, seed1016; tiny split intervals are exploratory',
        v0_source=str(a.pilot/'answers'/'four')))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','score'])
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,default='V1');p.add_argument('--split',choices=['all','tune','confirm'],default='all')
    p.add_argument('--smoke',action='store_true');p.add_argument('--clear-stop',action='store_true')
    a=p.parse_args();globals()[a.command](a)

if __name__=='__main__':main()
