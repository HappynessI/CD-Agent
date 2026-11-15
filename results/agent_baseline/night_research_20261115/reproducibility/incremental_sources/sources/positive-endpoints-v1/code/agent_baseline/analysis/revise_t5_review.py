"""Versioned T5 mask-quality candidates and isolated self-contained review pages.

Offline GT-simulated feedback, not human annotations. GT is only used for task
construction/QC, never provided as online masks. No formal evaluation or API calls.
Slurm only. Existing T4 review service/store and all historical files are untouched.
"""
import argparse
from collections import Counter
import base64
import html
import io
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from revise_whu_official import read, write, counts, metrics

S8=np.ones((3,3))


def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def point(mask):
    distance=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    y,x=np.unravel_index(distance.argmax(),mask.shape)
    return [int(x),int(y)]


def analyse(g,p):
    gl,gn=ndi.label(g,S8);pl,pn=ndi.label(p,S8)
    errors=[]
    for kind,error,parents,other in [('missed',g & ~p,gl,p),('false_positive',p & ~g,pl,g)]:
        lab,n=ndi.label(error,S8)
        for i,sl in enumerate(ndi.find_objects(lab),1):
            region=lab==i;area=int(region.sum())
            if area<32:continue
            parent_ids=np.unique(parents[region]);parent_ids=parent_ids[parent_ids>0]
            partial=any(bool(other[parents==pid].any()) for pid in parent_ids)
            xy=point(region)
            errors.append(dict(type=kind,area=area,point=xy,bbox=[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop],
                               parent_ids=[int(v) for v in parent_ids],partial_object=partial,
                               region_id=f'{kind}_{i}',region=region))
    errors.sort(key=lambda e:(not e['partial_object'],-e['area'],e['type'],e['point'][1],e['point'][0]))
    split=sum(len(set(pl[gl==i])- {0})>1 for i in range(1,gn+1))
    merge=sum(len(set(gl[pl==i])- {0})>1 for i in range(1,pn+1))
    partial=[e for e in errors if e['partial_object'] and e['area']>=64]
    mixed={e['type'] for e in errors}=={'missed','false_positive'}
    selected=errors[:6]
    # Preserve both error signs when mixed, so feedback really covers both.
    if mixed and len({e['type'] for e in selected})<2:
        selected[-1]=next(e for e in errors if e['type']!=selected[0]['type'])
    core=bool(partial and (mixed or len(partial)>=2) and sum(e['area'] for e in selected)>=128)
    categories=[]
    if partial:categories.append('partial_shape_boundary')
    if mixed:categories.append('mixed_omission_commission')
    if split:categories.append('fragmented_objects')
    if merge:categories.append('merged_objects')
    return errors,selected,dict(core_candidate=core,categories=categories,partial_error_regions=len(partial),
        mixed_error_signs=mixed,split_gt_objects=split,merged_initial_objects=merge,
        qualifying_error_regions=len(errors),initial=metrics(counts(p,g)),
        exclusion_reason=None if core else 'no_substantial_partial_shape_error_or_insufficient_combined_correction')


def clean(e):return {k:v for k,v in e.items() if k!='region'}


def instruction(task, selected, shape):
    lead='Improve the supplied building-change mask using T1 and T2. Correct the indicated errors while preserving valid building changes. Review object shape and boundaries; do not delete an entire object merely because one boundary point is wrong. '
    feedback=[]
    for e in selected:
        x,y=e['point'];loc=['upper','middle','lower'][min(2,3*y//shape[0])]+'-'+['left','center','right'][min(2,3*x//shape[1])]
        kind=('missing part of a changed building' if e['partial_object'] else 'missed building change') if e['type']=='missed' else ('excess mask around a partly correct object' if e['partial_object'] else 'false building-change region')
        if task=='T5.2':feedback.append(f'Point ({x}, {y}): {kind}.')
        else:feedback.append(f'In the {loc} area, near ({x}, {y}), there is {kind}; check the image evidence and adjust its outline.')
    return lead+'Feedback: '+' '.join(feedback)+' Submit the improved full mask and explain the main corrections.'


def t5(a):
    original=rows(a.run/'t52/tasks/T5.2_k3.jsonl');old53={r['id']:r for r in rows(a.run/'t53/tasks/T5.3.jsonl')}
    audit=[]; candidates=[]
    for r in original:
        g=read(r['gt_mask']);p=read(r['init_mask']);errors,chosen,qc=analyse(g,p)
        oldfeedback=old53[r['id']].get('meta',{}).get('feedback',[])
        info=dict(id=r['id'],qc=qc,old_t53_feedback_count=len(oldfeedback),
                  feedback=[clean(e) for e in chosen],source_gt=r['gt_mask'],source_initial=r['init_mask'])
        audit.append(info)
        if not qc['core_candidate']:continue
        for task in ('T5.2','T5.3'):
            taskrow=dict(r,task_type=task,instruction=instruction(task,chosen,g.shape))
            taskrow['meta']=dict(data_version='t5_mask_quality_v2_20261113',review_status='pending_review',
                feedback_source='offline_simulated_from_existing_LEVIR_GT_and_archived_SAM3_initial_mask',
                source_manifest=str(a.run/('t52/tasks/T5.2_k3.jsonl' if task=='T5.2' else 't53/tasks/T5.3.jsonl')),
                feedback=[clean(e) for e in chosen],qc=qc,formal_evaluation_eligible=False,
                limitation='Nontrivial geometry is a candidate-selection proxy; actual correction difficulty requires human review. Language feedback includes coarse area plus coordinates to avoid ambiguous references.')
            candidates.append(taskrow)
        for e in chosen:
            dest=a.output/'t5/feedback_regions'/Path(r['id']).stem/(e['region_id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(e['region'].astype('uint8')*255).save(dest)
    for task in ('T5.2','T5.3'):
        path=a.output/'t5/tasks'/f'{task}_v2_pending.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in candidates if r['task_type']==task))
    summary=dict(status='pending_review',api_calls=0,source_images=len(original),core_images=sum(r['qc']['core_candidate'] for r in audit),
        candidates_per_task=dict(Counter(r['task_type'] for r in candidates)),excluded_images=sum(not r['qc']['core_candidate'] for r in audit),
        old_t53_no_feedback=sum(not r['old_t53_feedback_count'] for r in audit),
        categories=dict(Counter(c for r in audit if r['qc']['core_candidate'] for c in r['qc']['categories'])),
        feedback_provenance='GT-simulated, fully reproducible, not collected human feedback',
        selection='existing LEVIR validation 64 only; candidate pool pending human review, not a new formal test set',
        protocol=dict(min_error_region_px=32,substantial_partial_error_px=64,min_selected_error_px=128,max_feedback_regions=6,
                      core='at least one substantial partial-object error plus mixed FP/FN or multiple substantial partial errors',
                      primary_evaluation='full-mask quality F1/IoU, precision/recall; boundary/object measures are secondary; no strict local compliance emphasis'))
    write(a.output/'t5/audit.json',audit);write(a.output/'t5/summary.json',summary)
    return candidates,audit


def anomaly(a):
    selection=json.loads((a.review/'selection.json').read_text());out=[]
    for cid in ('case_04','case_07','case_08','case_13','case_17'):
        case=next(r for r in selection if r['case_id']==cid);session=Path(case['session'])
        inp=json.loads((session/'input.json').read_text());final=json.loads((session/'final.json').read_text())
        gt=read(inp['gt_mask_path']);name=Path(inp['source_images'][1]).name
        pred=read(session/final['mask_path']);dataset=Path(inp['source_images'][1]).parent.parent
        with Image.open(dataset/'slabel1'/name) as im: b1=np.asarray(im)
        with Image.open(dataset/'slabel2'/name) as im: b2=np.asarray(im)
        rec=dict(case_id=cid,task=inp['task_id'],source_images=inp['source_images'],old_gt=inp['gt_mask_path'],
                 prediction=str(session/final['mask_path']),instruction=inp['instruction'],source_labels=[str(dataset/k/name) for k in ('slabel1','slabel2')],
                 review_status='pending_review',gt_edited=False,previous_review=case.get('review'),historical_final=metrics(counts(pred,gt)))
        if 'SECOND' in str(dataset):
            changed=(b1!=b2)&(b1!=0)&(b2!=0);new=changed&(b2==5)&(b1!=5);gone=changed&(b1==5)&(b2!=5)
            full=new if inp['change_direction']=='appeared' else new|gone
            match=re.search(r'box \[([\d, ]+)\]',inp['instruction']);box=[int(v) for v in match.group(1).split(',')]
            x0,y0,x1,y1=box;roi=np.zeros_like(gt);roi[y0:y1,x0:x1]=True;derived=full&roi
            labs,n=ndi.label(full,S8)
            crossing=[i for i in range(1,n+1) if (roi & (labs==i)).any() and (~roi & (labs==i)).any()]
            rec.update(roi=box,gt_vs_reconstructed_disagreement_px=int((gt!=derived).sum()),
                full_source_pixels=int(full.sum()),cropped_source_pixels=int(derived.sum()),roi_cut_components=crossing,
                conclusion='Task GT is compared against native semantic-change labels clipped to the instructed ROI. ROI intersections can cut buildings; source annotation accuracy cannot be inferred from shape alone.',
                new_gt=inp['gt_mask_path'],source_mask=str(a.output/'anomalies'/f'{cid}_uncropped.png'))
            dest=Path(rec['source_mask']);dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(full.astype('uint8')*255).save(dest)
        else:
            official=read(dataset/'label_unch0chi1'/name);xor=(b1>0)^(b2>0)
            rec.update(old_matches_slabel_xor=bool(np.array_equal(gt,xor)),old_vs_official=metrics(counts(gt,official)),
                       official_final=metrics(counts(pred,official)),official_gt=str(dataset/'label_unch0chi1'/name))
            if inp['task_id']=='T5.4':
                rec.update(new_gt=inp['gt_mask_path'],conclusion='Actual case_08 is T5.4 old-map update, not C2/C4. Its target is the difference between building inventories; official image-change GT is shown as a diagnostic, not substituted automatically.')
            else:
                rec.update(new_gt=str(a.output/'gt/c0_official'/name),conclusion='C0 target uses official WHU image-change annotation in v2; legacy XOR outline changes were not reliable change truth.')
        out.append(rec)
    write(a.output/'anomalies/audit.json',out);return out


def imgdata(path,mask=None,points=None,box=None):
    image=Image.open(path).convert('RGB')
    if mask is not None:
        arr=np.asarray(image).copy();m=read(mask) if isinstance(mask,(str,Path)) else mask
        arr[m]=(arr[m]*.45+np.array([255,40,60])*.55).astype('uint8');image=Image.fromarray(arr)
    draw=ImageDraw.Draw(image)
    if box:draw.rectangle(tuple(box),outline='#00ffff',width=3)
    for e in points or []:
        x,y=e['point'];color='#00ff66' if e['type']=='missed' else '#ffdd00';draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
    image.thumbnail((640,640));buf=io.BytesIO();image.save(buf,'JPEG',quality=83)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()


def page(a,cases):
    css='body{background:#edf0f5;color:#172133;font:16px system-ui;margin:0}header,main{max-width:1400px;margin:auto;padding:24px}.panel{background:white;padding:22px;margin:20px 0;border-radius:14px;border:1px solid #d9e0ea}figure{margin:0}img{width:100%;height:auto} .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figcaption{padding:8px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#1869aa}.badge{background:#fff1c4;padding:5px 10px;border-radius:10px}@media(max-width:800px){.grid{grid-template-columns:1fr}}'
    links=[]
    for i,c in enumerate(cases,1):
        key=f'review_{i:02d}';panels=[('T1 / old map',c['t1'],None),('T2',c['t2'],None),('Old GT overlay',c['t2'],c['old_gt']),('New / proposed GT overlay',c['t2'],c['new_gt'])]
        if c.get('initial'):panels.append(('Initial mask + feedback',c['t2'],c['initial']))
        if c.get('prediction'):panels.append(('Historical prediction',c['t2'],c['prediction']))
        if c.get('source_mask'):panels.append(('Uncropped native-label target',c['t2'],c['source_mask']))
        if c.get('official_gt'):panels.append(('Official C0 diagnostic',c['t2'],c['official_gt']))
        figures=''.join('<figure><img src="'+imgdata(p,m,c.get('feedback') if label.startswith('Initial') else None,c.get('roi'))+'"><figcaption>'+html.escape(label)+'</figcaption></figure>' for label,p,m in panels)
        body=f'<h1>{html.escape(c["title"])}</h1><p><span class="badge">pending_review · no formal evaluation</span></p><p>{html.escape(c["reason"])}</p><p>{html.escape(c["instruction"])}</p>'
        body+='<p>Red = mask overlay; cyan = ROI; green points = missed pixels; yellow points = excess pixels. Overlay previews are resized; source paths retain original resolution.</p><div class="grid">'+figures+'</div><details open><summary>Source / QC / limitations</summary><pre>'+html.escape(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></details>'
        (a.output/f'{key}.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(c['title'])+'</title><style>'+css+'</style><header><a href="index.html">Review index</a></header><main><section class="panel">'+body+'</section></main>')
        links.append(f'<li><a href="{key}.html">{i:02d} · {html.escape(c["title"])}</a> — {html.escape(c["reason"])}</li>')
    summary={'cases':len(cases),'api_calls':0,'status':'pending_review','t4_store_touched':False}
    write(a.output/'review_selection.json',cases);write(a.output/'review_summary.json',summary)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Data revision review</title><style>'+css+'</style><header><h1>WHU / T5 data revision review</h1><p>Version 20261113 v2 · '+str(len(cases))+' actual examples · all pending human review. Existing predictions are rescored offline only.</p></header><main><section class="panel"><p>Purposeful examples, not a representative sample. No candidate is approved or used for formal Agent experiments. Feedback is simulated reproducibly from existing GT, not human feedback. C1 conflict requires direction review.</p><ol>'+''.join(links)+'</ol></section></main>')


def build(a):
    raise SystemExit('Historical WHU comparison renderer retired after XOR GT removal; retain the published embedded review pages. Use finalize_whu_gt for current references.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--whu',type=Path,required=True);build(p.parse_args())
