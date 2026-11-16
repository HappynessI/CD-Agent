"""Versioned N9 correction: S5 construction gate only for directed profiles.

Never mutates the running parent version. Reuse requires actual image bytes,
all non-image message content, and complete request parameters to match.
"""
import argparse
import copy
import os
from pathlib import Path
import time
import numpy as np
from .n8_missed_assets import read, write
from .n8_missed_pipeline import cost_total as local_cost
from .n9_missed_assets import OUT as PARENT
from .n9_missed_protocol import configurations as original_configurations, candidate_indices as original_indices
from .n9_missed_prepare import atomic_write

OUT=PARENT/'method_revision_s5_only'
VERSION='s5_only_directed_v1'


def configurations(profile):
    result=copy.deepcopy(original_configurations(profile))
    if profile!='any':
        for c in result:c['dino']=None
    return result


def candidate_indices(candidates,config):
    if config['profile']=='any':return original_indices(candidates,config)
    enabled=('S1_02','S2','S5') if config['profile']=='demolish' else ('S1_02','S2','S3','S4','S5')
    chosen=[c for c in candidates if set(c['sources'])&set(enabled) and c['semantic_delta']>=config['semantic']]
    chosen.sort(key=lambda c:(-max(c['semantic_delta'],0.)*max(c['dino'],0.)*c['area']**config['alpha'],c['index']))
    return [c['index'] for c in chosen[:config['k']]]


def cumulative_cost(out):
    return local_cost(Path(out).parent)+local_cost(out)


def initialize(out):
    out=Path(out); parent=out.parent
    for name in ('asset_manifest.json','evaluation_manifest.json','fm1_manifest.json','frozen_protocol.json'):
        value=read(parent/name)
        if (out/name).exists() and read(out/name)!=value:raise ValueError('Immutable shared inputs differ: '+name)
        atomic_write(out/name,value)
    atomic_write(out/'method_amendment.json',dict(version=VERSION,parent=str(parent),
        reason='Specification audit before reading tune F1: directed dino_020 belongs to S5 construction, not every net candidate',
        changed='build/demolish net-candidate DINO hard threshold removed; S5 pre-dilation .20 unchanged',
        unchanged=['any','source combinations','semantic grid','DINO ranking factor','alpha','K','verifiers','splits','online selection archive','tune selection rule'],
        parent_role='Preserved extra-gate sensitivity, not a competing configuration to select by outcomes',
        online_authorized=False,gt_read_for_amendment=False))


def equivalent_requests(expected,previous):
    if expected['parameters']!=previous['parameters']:return False
    a=expected['messages']; b=previous['messages']
    if len(a)!=len(b):return False
    for ma,mb in zip(a,b):
        if ma.get('role')!=mb.get('role'):return False
        ca,cb=ma['content'],mb['content']
        if isinstance(ca,str) or isinstance(cb,str):
            if ca!=cb:return False
            continue
        if len(ca)!=len(cb):return False
        for ba,bb in zip(ca,cb):
            if ba.get('type')=='image_url' and bb.get('type')=='image_url':
                aa=copy.deepcopy(ba); ab=copy.deepcopy(bb)
                pa=aa['image_url'].pop('url'); pb=ab['image_url'].pop('url')
                if aa!=ab or Path(pa).read_bytes()!=Path(pb).read_bytes():return False
            elif ba!=bb:return False
    return True


def reuse(a):
    from .n9_missed_pipeline import make_request
    from .n8_missed_prepare import load_masks
    from .night_missed_views import render
    from .io_retry import read_image
    from agent_baseline.view import ask_images
    records=[]; loaded=None
    for row in read(a.out/f'requests_{a.split}.json'):
        for variant in row['variants']:
            destination=a.out/'answers'/variant/(row['uid']+'.json')
            if destination.exists():continue
            source=a.out.parent/'answers'/variant/(row['uid']+'.json')
            source_request=a.out.parent/'inputs'/variant/row['uid']/'request.json'
            if not source.exists() or not source_request.exists():continue
            previous_answer=read(source)
            if previous_answer.get('answer') not in ('yes','no'):continue
            if loaded!=row['masks']:
                masks=load_masks(row['masks']); rgb=[read_image(row['image_'+t],rgb=True) for t in ('t1','t2')]
                with np.load(row['base'],allow_pickle=False) as z:base=z['start'].astype(bool)
                loaded=row['masks']
            style='blue' if variant in ('V2','V4') else 'original'
            viewdir=a.out/'views'/row['uid']/style;viewdir.mkdir(parents=True,exist_ok=True)
            paths=[viewdir/f'{i}.jpg' for i in range(4)]
            if not all(p.exists() for p in paths):
                mask=masks[row['mask_index']]
                images=render(rgb,mask,base,row['candidate']['bbox'],'V2')[0] if style=='blue' else ask_images(rgb,mask,row['candidate']['bbox'],True)
                for path,image in zip(paths,images):image.save(path,quality=92)
            expected=make_request(row,paths,variant); previous=read(source_request)
            same=equivalent_requests(expected,previous)
            record=dict(uid=row['uid'],variant=variant,source_answer=str(source),source_request=str(source_request),
                image_bytes_text_parameters_seed_equal=same,actual_new_cost_cny=0)
            records.append(record)
            if not same:continue
            folder=a.out/'inputs'/variant/row['uid']
            atomic_write(folder/'request.json',dict(expected,uid=row['uid'],variant=variant,reuse_source=str(source_request)))
            atomic_write(destination,dict(previous_answer,input_path=str(folder),reused_from=str(source),
                reuse_contract='four image bytes + complete text + parameters including model and seed equal',
                actual_new_cost_cny=0))
    old=read(a.out/f'reuse_{a.split}.json') if (a.out/f'reuse_{a.split}.json').exists() else []
    atomic_write(a.out/f'reuse_{a.split}.json',old+records)


def install():
    from . import n9_missed_pipeline as pipeline, n9_missed_score as score
    pipeline.configurations=configurations
    pipeline.candidate_indices=candidate_indices
    pipeline.cost_total=cumulative_cost;score.cost_total=cumulative_cost
    # configs_for uses pipeline's globals, including versioned configurations.
    return pipeline,score


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('initialize','prepare','reuse','verify','score'))
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--split',choices=('tune','confirm'),default='tune')
    p.add_argument('--budget',type=float,default=20.);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.command=='initialize':initialize(a.out);return
    if a.command=='reuse':reuse(a);return
    pipeline,score=install()
    if a.command=='score':
        a.allow_abstain=True
        score.score(a)
        if a.split=='confirm':
            path=a.out/'offline_decision.json';decisions=read(path)
            for row in decisions.values():
                row['offline_positive']=row.pop('advance');row['advance']=False
                row['online_authorized']=False;row['online_status']='prohibited_by_latest_user_instruction'
            atomic_write(path,decisions)
    else:getattr(pipeline,a.command)(a)

if __name__=='__main__':main()
