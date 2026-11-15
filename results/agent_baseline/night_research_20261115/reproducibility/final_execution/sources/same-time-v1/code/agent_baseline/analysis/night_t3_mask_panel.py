"""B3: single-date aligned RGB and binary support panels; tune first."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from render import crop_box,SIZE
from agent_baseline.exemplar_semantics import request,canonical,CLASSES,PARSE_QUESTION
from .io_retry import read_image,read_text,write_json,exists
from .t3_routes import load_dev,pair_id,group
from .t3_routes_match import bbox
from .night_t3_grounding import API_SEED,summarize_parses


def panels(image,support):
    result=[];box=bbox(support)
    for label,kwargs in [('close',{}),('wide',dict(scale=10,min_side=256))]:
        l,t,r,b=crop_box(box,support.shape,**kwargs)
        rgb=Image.fromarray(image[t:b,l:r]).resize((SIZE,SIZE),Image.Resampling.BICUBIC)
        binary=Image.fromarray(support[t:b,l:r].astype(np.uint8)*255).resize((SIZE,SIZE),Image.Resampling.NEAREST).convert('RGB')
        result.extend([(f'{label} original RGB',rgb),(f'{label} aligned binary support: WHITE included, BLACK excluded including holes',binary)])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm'],default='tune');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Execute through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));rows=load_dev(a.run)
    geometries={r['id']:r for r in audit['records']};out=a.night/'B3'
    policy=dict(model='qwen3.7-flash',temperature=0,seed=API_SEED,
        intervention='single-date original RGB close/wide plus separate aligned binary support panels and explicit support prompt; joint representation/prompt intervention',
        comparison='not an isolated pure-visual causal intervention; B0/B1 differ in input count and support instruction',
        confirmation='run only after tune evidence and manual grounding review support promotion; not automatic',
        split=audit['split'],code=str(Path(__file__).resolve()))
    write_json(out/'settings.json',policy)
    prior=json.loads(read_text(out/'summary.json'))['records'] if exists(out/'summary.json') else []
    by_id={r['id']:r for r in prior};reverse={name:i for i,name in CLASSES.items()}
    for row in rows:
        if audit['split'][row['id']]!=a.split:continue
        support=read_image(row['exemplar_mask'])>0;answers=[]
        for time in ('t1','t2'):
            image=read_image(row['image_'+time],rgb=True)
            system=('All four images refer to ONE date. Each original RGB crop is followed by a binary support panel with identical coordinates and size. '
                'Identify land cover in RGB only at coordinates where the paired panel is WHITE. BLACK is excluded, including black holes enclosed by white regions. '
                'Objects inside black holes are NOT the target. Do not classify the entire RGB crop or the shape of the white panel itself. '
                'Explain the visible land cover on the included white-supported pixels.')
            answers.append(request(out/'tasks'/row['id']/time,system,PARSE_QUESTION,panels(image,support),API_SEED))
        pair=[reverse.get(canonical(v['parsed'].get('class'))) for v in answers];truth=list(pair_id(row));geo=geometries[row['id']]
        by_id[row['id']]=dict(id=row['id'],split=a.split,group=group(row),has_holes=geo['has_holes'],remaining_gt_pixels=geo['remaining_gt_pixels'],
            interpretation=dict(pair=pair,classes=[canonical(v['parsed'].get('class')) for v in answers],evidence=[v['parsed'].get('evidence') for v in answers]),
            true_pair=truth,before_agrees=pair[0]==truth[0],after_agrees=pair[1]==truth[1],pair_agrees=pair==truth,
            semantic_review_required=pair!=truth,new_cost_cny=sum(v['cost_cny'] for v in answers))
        all_records=[by_id[r['id']] for r in rows if r['id'] in by_id]
        current=[r for r in all_records if r['split']==a.split]
        for name,records in [('summary.json',all_records),(f'summary_{a.split}.json',current)]:
            write_json(out/name,dict(variant='B3',tasks=len(records),records=records,label_agreement_only=True,
                summary=summarize_parses(records),new_cost_cny=sum(r['new_cost_cny'] for r in records)))
        print('B3',a.split,len(current),row['id'],flush=True)


if __name__=='__main__':main()
