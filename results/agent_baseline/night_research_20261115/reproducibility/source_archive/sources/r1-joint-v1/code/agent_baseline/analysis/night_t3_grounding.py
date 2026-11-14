"""N2 fixed dev60: original-support audit and B0/B1/B2 grounding comparison."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
from render import crop_box, SIZE
from agent_baseline.exemplar_semantics import request, canonical, CLASSES, PARSE_QUESTION
from .io_retry import read_image, read_text, write_json, write_text, exists
from .t3_routes import load_dev, pair_id, group, GROUPS, pooled
from .t3_routes_match import bbox
from .t3_exemplar_geometry_audit import geometry

SPLIT_SEED=20261114
API_SEED=1016
DARKEN=.70


def fixed_split(rows):
    images=sorted({str(Path(row['image_t1']).resolve()) for row in rows})
    if len(images)!=len(rows): raise ValueError('Require one task per image for this fixed split')
    random.Random(SPLIT_SEED).shuffle(images)
    tune=set(images[:len(images)//2])
    return {row['id']:('tune' if str(Path(row['image_t1']).resolve()) in tune else 'confirm') for row in rows}


def spotlight(image,support,darken=DARKEN):
    result=np.rint(np.asarray(image,dtype=float)*(1-darken)).astype(np.uint8)
    result[support]=image[support]
    return result


def raw_views(image,support):
    box=bbox(support); output=[]
    for label,kwargs in [('close',{}),('wide',dict(scale=10,min_side=256))]:
        l,t,r,b=crop_box(box,support.shape,**kwargs)
        output.append((label,Image.fromarray(image[t:b,l:r]).resize((SIZE,SIZE),Image.Resampling.BICUBIC)))
    return output


def views(images,support,time,variant):
    target=images[time]
    output=[(f'Target date T{time+1} spotlight {label}',image)
        for label,image in raw_views(spotlight(target,support),support)]
    if variant=='B2':
        for index,original in enumerate(images):
            output.extend((f'Unmodified reference T{index+1} {label}',image) for label,image in raw_views(original,support))
    return output


def summarize_parses(records):
    result=[]
    for split in ('all','tune','confirm'):
        for label in ('全部',*GROUPS):
            for holes in ('all','holes','no_holes'):
                selected=[r for r in records if (split=='all' or r['split']==split)
                    and (label=='全部' or r['group']==label)
                    and (holes=='all' or r['has_holes']==(holes=='holes'))]
                result.append(dict(split=split,group=label,holes=holes,tasks=len(selected),
                    before_label_agreement=np.mean([r['before_agrees'] for r in selected]).item() if selected else None,
                    after_label_agreement=np.mean([r['after_agrees'] for r in selected]).item() if selected else None,
                    pair_label_agreement=np.mean([r['pair_agrees'] for r in selected]).item() if selected else None))
    return result


def audit(rows,old,output):
    split=fixed_split(rows); old_by={t['id']:t for t in old['per_task']}; records=[]
    for row in rows:
        ex=read_image(row['exemplar_mask'])>0; truth=read_image(row['gt_mask'])>0
        prior=old_by[row['id']]
        records.append(dict(id=row['id'],split=split[row['id']],group=group(row),
            **geometry(ex,truth),gt_components=prior['proposal_upper']['gt_instances'],
            proposal_upper=prior['proposal_upper'],baseline_records=prior['records']))
    strata=[]
    for part in ('all','tune','confirm'):
        for support in ('all','positive','empty','at_least_16_diagnostic'):
            selected=[r for r in records if (part=='all' or r['split']==part) and
                (support=='all' or (support=='positive' and r['remaining_gt_pixels']>0) or
                 (support=='empty' and r['remaining_gt_pixels']==0) or
                 (support=='at_least_16_diagnostic' and r['remaining_gt_pixels']>=16))]
            for label in ('全部',*GROUPS):
                chosen=[r for r in selected if label=='全部' or r['group']==label]
                methods={}
                for method,k in sorted({(v['method'],v['k']) for r in chosen for v in r['baseline_records']}):
                    metrics=[v for r in chosen for v in r['baseline_records'] if v['method']==method and v['k']==k]
                    methods[f'{method}@{k}']=dict(pooled(metrics),
                        empty_prediction_rate=np.mean([v['counts'][0]+v['counts'][1]==0 for v in metrics]).item() if metrics else None,
                        per_task_counts=[dict(id=r['id'],counts=v['counts']) for r in chosen for v in r['baseline_records'] if v['method']==method and v['k']==k])
                strata.append(dict(split=part,support=support,group=label,tasks=len(chosen),methods=methods))
    result=dict(policy=dict(split_seed=SPLIT_SEED,split_rule='shuffle sorted unique image paths, first half tune; remainder confirm',
        confirmation_scope='selection-isolated for new night variants only; old dev60 has already been inspected, not an unseen holdout',
        diagnostic_16='make_t3_tasks.MIN_AREA=16 is original component construction cutoff; remaining total>=16 is diagnostic only, not a revised benchmark'),
        split=split,records=records,strata=strata)
    write_json(output/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['audit','B1','B2'],required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Execute through Slurm')
    rows=load_dev(a.run);old=json.loads(read_text(a.baseline));prior={r['id']:r for r in old['per_task']}
    if len(prior)!=60: raise ValueError('B0 must be complete original dev60')
    audited=json.loads(read_text(a.output/'audit.json')) if exists(a.output/'audit.json') else audit(rows,old,a.output)
    geometry_by={r['id']:r for r in audited['records']}
    policy=dict(model='qwen3.7-flash',seed=API_SEED,temperature=0,darken=DARKEN,
        frozen_code=str(Path(__file__).resolve()),baseline_source=str(a.baseline),run_root=str(a.run),
        split=audited['split'],B0='reuse existing per-date contour parses, no new requests',
        B1='single date, support unchanged before resize; outside and holes 70% darkened; no red contour',
        B2='same spotlight target date plus unmodified T1/T2 close/wide crops; has cross-date information')
    write_json(a.output/'grounding_policy.json',policy)
    variants=['B0'] if a.variant=='audit' else ['B0',a.variant]
    for variant in variants:
        records=[]
        for row in rows:
            if variant=='B0':
                interpreted=prior[row['id']]['interpretation'];pair=interpreted['pair'];new_cost=0.
            else:
                ex=read_image(row['exemplar_mask'])>0;images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')]
                answers=[]
                for time in (0,1):
                    system=('Identify land cover only on the bright, unmodified support pixels in the target-date spotlight views. '
                        'All darkened pixels, including dark holes surrounded by bright support, are excluded. '
                        'Do not classify objects inside excluded holes. ' +
                        ('Raw reference images show both dates for context; answer only for the stated target date.' if variant=='B2' else 'Both views show only this same date.'))
                    answers.append(request(a.output/variant/'tasks'/row['id']/f't{time+1}',system,
                        f'Target date is T{time+1}. '+PARSE_QUESTION,views(images,ex,time,variant),API_SEED))
                reverse={name:i for i,name in CLASSES.items()}
                pair=[reverse.get(canonical(answer['parsed'].get('class'))) for answer in answers]
                interpreted=dict(pair=pair,classes=[canonical(v['parsed'].get('class')) for v in answers],evidence=[v['parsed'].get('evidence') for v in answers])
                new_cost=sum(v['cost_cny'] for v in answers)
            gt=list(pair_id(row));geo=geometry_by[row['id']]
            records.append(dict(id=row['id'],split=audited['split'][row['id']],group=group(row),has_holes=geo['has_holes'],
                remaining_gt_pixels=geo['remaining_gt_pixels'],interpretation=interpreted,true_pair=gt,
                before_agrees=pair[0]==gt[0],after_agrees=pair[1]==gt[1],pair_agrees=pair==gt,
                semantic_review_required=pair!=gt,new_cost_cny=new_cost))
            write_json(a.output/variant/'summary.json',dict(variant=variant,tasks=len(records),records=records,
                label_agreement_only=True,summary=summarize_parses(records),new_cost_cny=sum(r['new_cost_cny'] for r in records)))
            print(variant,len(records),row['id'],flush=True)
    write_text(a.output/'grounding_notes.md','# N2 Grounding对照\n\n固定原dev60与图像级30/30调参/确认划分。B0复用；B1为单时相支持区保真、其余及孔洞压暗70%；B2额外看两期原始裁图，因此不能将差异仅归因显示。标签一致只作诊断，人工grounding复核独立。完整/正目标/空目标分层见audit.json；原完整评分未改变。\n')


if __name__=='__main__':main()
