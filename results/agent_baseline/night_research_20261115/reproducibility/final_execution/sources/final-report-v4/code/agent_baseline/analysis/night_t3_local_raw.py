"""Tune-first raw local contour verification on exactly the existing B0 R2 Top8."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from render import crop_box,SIZE
from agent_baseline.exemplar_semantics import request,CLASSES
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,group
from .t3_routes_match import bbox
from .t3_routes_cache import read_npz
from .night_t3_grounding import API_SEED
from .night_t3_retrieval import score,summaries,verification_stats


def local_raw_views(images,mask):
    l,t,r,b=crop_box(bbox(mask),mask.shape,scale=1.5,min_side=64)
    support=np.asarray(Image.fromarray(mask[t:b,l:r]).resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
    boundary=support&~ndi.binary_erosion(support,border_value=0)
    result=[]
    for i,image in enumerate(images):
        pixels=np.asarray(Image.fromarray(image[t:b,l:r]).resize((SIZE,SIZE),Image.Resampling.BICUBIC)).copy()
        pixels[boundary]=(255,0,0)
        result.append((f'T{i+1} local raw RGB with thin candidate contour',Image.fromarray(pixels)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm'],default='tune');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));source=a.night/'retrieval/B0';out=a.night/'retrieval/B0_local_raw'
    complete=json.loads(read_text(source/'summary_all.json'))
    if complete['tasks']!=60:raise ValueError('Complete original B0 diagnostic before paired view intervention')
    parses={r['id']:r for r in json.loads(read_text(a.night/'B0/summary.json'))['records']}
    settings=dict(code=str(Path(__file__).resolve()),source=str(source),model='qwen3.7-flash',seed=API_SEED,temperature=0,max_tokens=768,k=8,
        intervention='same local crop scale1.5/min64; raw RGB plus one-display-pixel inner red contour and necessary support wording; joint view/prompt intervention',
        fixed='same B0 parsed pair, R2 pool, DINO Top8 and verifier response schema; no GT to model',
        old_cost='original spotlight answers reused with zero additional charge; original costs remain in B0 request artifacts',
        promotion='tune30 first; confirm only if positive-GT evidence and manual grounding support promotion; 1136 remaining3px is mechanism only')
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Existing raw view settings differ')
    write_json(out/'settings.json',settings);records=[]
    for row in load_dev(a.run):
        if audit['split'][row['id']]!=a.split:continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        original=source/'tasks'/row['id'];old=json.loads(read_text(original/'metrics.json'));saved=read_npz(original/'prepared.npz')
        masks=saved['r2'].astype(bool);indices=saved['order3'].tolist()[:8];pair=parses[row['id']]['interpretation']['pair']
        images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];answers={}
        for index in indices:
            if str(index) not in old['verification']:raise ValueError('Original matched candidate answer is missing')
            before,after=[CLASSES[c] for c in pair]
            question=(f'Does the candidate region delineated by the thin red contour change from {before} at T1 to {after} at T2? '
                'Judge the included region; enclosed holes are excluded. Unchanged or reversed transitions are no. '
                'Reply JSON {"change":"yes" or "no", "before":"visible class", "after":"visible class", "reason":"local visual evidence"}.')
            response=request(folder/'verification'/str(index),'Judge only the candidate delineated by the thin red contour in the two local-date images.',question,local_raw_views(images,masks[index]),API_SEED)
            label=response['parsed'].get('change');answers[index]=dict(answer=label if label in ('yes','no') else 'unparsed',cost_cny=response['cost_cny'])
        truth=read_image(row['gt_mask'])>0;ex=read_image(row['exemplar_mask'])>0
        scores=[r for r in old['scores'] if r['method'] in ('R3_verified','R3_drop_unparsed_diagnostic') and r['k']==8]
        raw=score(masks,[i for i in indices if answers[i]['answer']!='no'],truth,ex,'R3_raw_contour_verified',8)
        scores.extend([raw,score(masks,[i for i in indices if answers[i]['answer']=='yes'],truth,ex,'R3_raw_drop_unparsed_diagnostic',8)])
        baseline=next(r for r in scores if r['method']=='R3_verified')
        record=dict(id=row['id'],split=a.split,group=group(row),remaining_gt_pixels=old['remaining_gt_pixels'],scores=scores,
            verification=answers,verification_stats=verification_stats(answers),spotlight_verification_stats=verification_stats(old['verification']),
            paired_counts_delta=dict(zip(('tp','fp','fn'),[new-prior for new,prior in zip(raw['counts'],baseline['counts'])])),
            candidate_ids=indices,spotlight_reused_from=str(original),spotlight_additional_cost_cny=0,cost_cny=sum(v['cost_cny'] for v in answers.values()))
        write_json(final,record);records.append(record)
        write_json(out/f'summary_{a.split}.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))
        print('local raw',a.split,len(records),row['id'],flush=True)
    write_json(out/f'summary_{a.split}.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))


if __name__=='__main__':main()
