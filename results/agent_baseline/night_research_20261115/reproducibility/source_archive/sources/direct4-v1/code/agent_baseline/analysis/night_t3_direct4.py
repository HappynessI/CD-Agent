"""Tune-first direct exemplar comparison on fixed R1 geometry/DINO candidates."""
import argparse
import json
import os
from pathlib import Path
from agent_baseline.exemplar_semantics import request
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,group
from .t3_routes_cache import read_npz
from .t3_routes_match import T3_QUESTION
from .night_t3_grounding import API_SEED
from .night_t3_local_raw import local_raw_views
from .night_t3_retrieval import score,summaries,verification_stats

SYSTEM=('Compare land-cover changes in two separate regions, Example and Candidate. '
    'Each has two local original-RGB views, T1 then T2. Thin red contours mark included support; enclosed holes are excluded. '
    'The drawn support boundary itself is not an object structure. Infer land cover before and after in each region from RGB evidence. '
    'Reply JSON only. Reverse transitions and unchanged regions are not matches.')


def four_views(images,exemplar,candidate,swap_example=False):
    return [(f'{role} {label.replace("candidate contour","support contour")}',view) for role,mask in [('Example',exemplar),('Candidate',candidate)] for label,view in local_raw_views(images[::-1] if swap_example and role=='Example' else images,mask)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm'],default='tune');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));source=a.night/'retrieval/B0';out=a.night/'r1_direct4'/a.split
    original=json.loads(read_text(source/'summary_all.json'))
    if original['tasks']!=60:raise ValueError('Complete original fixed R1 diagnostic required')
    settings=dict(code=str(Path(__file__).resolve()),source=str(source),split=a.split,model='qwen3.7-flash',seed=API_SEED,temperature=0,max_tokens=768,k=8,
        pool='existing geometry-filtered R1 masks/order1; candidate IDs are indices in prepared r1, not original SAM pool',
        overlap='R1 eligible does not exclude exemplar overlap. Candidate can contain the example; retain fixed pool and exclude exemplar pixels only during scoring.',
        intervention='4 local original RGB views with thin support contour and explicit support instruction; joint view/prompt intervention',
        semantics='direct example/candidate comparison; no parsed class pair, task ID, or GT to model',
        contrast='same R1 Top8 without verification; old8 original-pool verifier and R3 parsed-text two-view verifier are contextual references, not strict paired controls',
        selection='fixed tune30 first; confirm only after positive-target evidence; no weight or TopK tuning')
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Existing direct4 settings differ')
    write_json(out/'settings.json',settings);records=[]
    for row in load_dev(a.run):
        if audit['split'][row['id']]!=a.split:continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        prior=source/'tasks'/row['id'];old=json.loads(read_text(prior/'metrics.json'));saved=read_npz(prior/'prepared.npz')
        masks=saved['r1'].astype(bool);indices=saved['order1'].tolist()[:8]
        images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];exemplar=read_image(row['exemplar_mask'])>0;answers={}
        for index in indices:
            response=request(folder/'verification'/str(index),SYSTEM,T3_QUESTION,four_views(images,exemplar,masks[index]),API_SEED)
            label=response['parsed'].get('change');answers[index]=dict(answer=label if label in ('yes','no') else 'unparsed',cost_cny=response['cost_cny'])
        truth=read_image(row['gt_mask'])>0
        baseline=next(r for r in old['scores'] if r['method']=='R1_geometry_dino' and r['k']==8)
        verified=score(masks,[i for i in indices if answers[i]['answer']!='no'],truth,exemplar,'R1_direct4_verified',8)
        scores=[baseline,verified,score(masks,[i for i in indices if answers[i]['answer']=='yes'],truth,exemplar,'R1_direct4_drop_unparsed_diagnostic',8)]
        record=dict(id=row['id'],split=a.split,group=group(row),remaining_gt_pixels=old['remaining_gt_pixels'],scores=scores,
            verification=answers,verification_stats=verification_stats(answers),geometry_proposal_upper=old['geometry_proposal_upper'],
            paired_counts_delta=dict(zip(('tp','fp','fn'),[new-previous for new,previous in zip(verified['counts'],baseline['counts'])])),
            candidate_ids=indices,unverified_reused_from=str(prior),cost_cny=sum(v['cost_cny'] for v in answers.values()))
        write_json(final,record);records.append(record)
        write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))
        print('direct4',a.split,len(records),row['id'],flush=True)
    write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))


if __name__=='__main__':main()
