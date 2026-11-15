"""Frozen-threshold direct4 verification with exact rendered-input answer reuse."""
import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from agent_baseline.exemplar_semantics import request, MODEL
from agent_baseline.backends import encode_image
from .io_retry import read_text, read_image, write_json, write_text, exists
from .t3_routes import load_dev
from .t3_routes_cache import read_npz
from .night_t3_direct4 import SYSTEM, four_views
from .t3_routes_match import T3_QUESTION
from .night_t3_grounding import API_SEED
from .n7_t3_rejection import RANKS, evaluate, summarize


def exact_reuse(folder, views):
    if not exists(folder/'answer.json'):return None
    previous=json.loads(read_text(folder/'input.json'))
    settings=dict(system=SYSTEM,question=T3_QUESTION,model=MODEL,seed=API_SEED,max_tokens=768,temperature=0)
    if any(previous.get(k)!=v for k,v in settings.items()):return None
    if len(previous['images'])!=len(views):return None
    for prior,(label,view) in zip(previous['images'],views):
        if prior['label']!=label:return None
        with Image.open(BytesIO(base64.b64decode(encode_image(view)))) as image:
            current=np.asarray(image)
        if not np.array_equal(read_image(prior['path'],rgb=True),current):return None
    return json.loads(read_text(folder/'answer.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['tune','confirm'],required=True)
    p.add_argument('--reuse-only',action='store_true',help='Inventory missing requests; do not fabricate missing answers or aggregate incomplete verification')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    baseline=json.loads(read_text(a.out/f'{a.split}_summary.json'))
    frozen=json.loads(read_text(a.out/'frozen.json'))
    if baseline['frozen']!=frozen:raise ValueError('Frozen thresholds differ')
    sources={r['id']:r for r in baseline['records']}
    rows=[r for r in load_dev(a.run) if r['id'] in sources]
    out=a.out/'verification'/a.split
    records=[];inventory=[];started=time.monotonic()
    for row in rows:
        source=sources[row['id']];folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        masks=read_npz(Path(source['source_prepared']))['r1'].astype(bool)
        indices={rank:source['methods'][rank+'_threshold']['candidate_ids'] for rank in RANKS}
        union=sorted(set().union(*map(set,indices.values())))
        images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')]
        ex=read_image(row['exemplar_mask'])>0;answers={};missing=[]
        for index in union:
            target=folder/'requests'/str(index)
            reference=target/'reference.json'
            if exists(reference):
                answers[index]=json.loads(read_text(reference));continue
            views=four_views(images,ex,masks[index])
            old=a.night/'r1_direct4'/a.split/'tasks'/row['id']/'verification'/str(index)
            response=exact_reuse(old,views)
            reused=response is not None
            if response is None:
                if a.reuse_only:
                    missing.append(index);continue
                response=request(target,SYSTEM,T3_QUESTION,views,API_SEED)
            answer=response['parsed'].get('change')
            answers[index]=dict(answer=answer if answer in ('yes','no') else 'unknown',
                reused=reused,source=str(old/'answer.json' if reused else target/'answer.json'),
                new_cost_cny=0 if reused else response['cost_cny'],
                historical_cost_cny=response['cost_cny'] if reused else 0,
                exact_reuse_check='settings, ordered labels and every decoded rendered pixel' if reused else None)
            write_json(reference,answers[index])
        inventory.append(dict(id=row['id'],required=len(union),reused=sum(v['reused'] for v in answers.values()),missing=missing))
        write_json(out/'inventory.json',dict(records=inventory,missing=sum(len(r['missing']) for r in inventory),reuse_only=a.reuse_only))
        if missing:continue
        truth=read_image(row['gt_mask'])>0
        record=dict(source)
        record['methods']=dict(source['methods'])
        for rank in RANKS:
            accepted=[i for i in indices[rank] if answers[i]['answer']=='yes']
            record['methods'][rank+'_threshold_verified']=evaluate(masks,accepted,truth,ex)
        record['verification']={str(k):v for k,v in answers.items()}
        record['verification_cost']=dict(unique_slots=len(answers),new_calls=sum(not v['reused'] for v in answers.values()),
            reused_calls=sum(v['reused'] for v in answers.values()),cost_cny=sum(v['new_cost_cny'] for v in answers.values()),
            unknown=sum(v['answer']=='unknown' for v in answers.values()),
            per_arm_slots={rank:len(indices[rank]) for rank in RANKS})
        write_json(final,record);records.append(record)
        print('N7 verifier',a.split,len(records),row['id'],flush=True)
    if len(records)!=30:
        write_json(out/'status.json',dict(status='missing_answers',completed=len(records),expected=30,api_calls=0 if a.reuse_only else None))
        return
    result=dict(records=records,tasks=len(records),summary=summarize(records),frozen=frozen,
        new_calls=sum(r['verification_cost']['new_calls'] for r in records),
        reused_calls=sum(r['verification_cost']['reused_calls'] for r in records),
        cost_cny=sum(r['verification_cost']['cost_cny'] for r in records),elapsed_seconds=time.monotonic()-started,
        model=MODEL,policy='Accept explicit yes only. Same fixed threshold and candidate pool/order as unverified comparator. Calls shared by identical candidates across rank arms.',
        limitation='Historical deterministic-setting responses and fresh requests have different time-of-request; reuse eliminates new cost but not historical model-version uncertainty.')
    write_json(out/'summary.json',result)
    previous_status=out/'status.json'
    if exists(previous_status):
        previous=json.loads(read_text(previous_status))
        if previous.get('status')=='missing_answers' and not exists(out/'reuse_only_status.json'):
            write_json(out/'reuse_only_status.json',previous)
    write_json(previous_status,dict(status='complete',completed=len(records),expected=30,
        new_calls=result['new_calls'],reused_calls=result['reused_calls'],cost_cny=result['cost_cny'],
        summary_source=str(out/'summary.json')))
    if a.split=='confirm' and exists(a.out/'verification/tune/summary.json'):
        tune=json.loads(read_text(a.out/'verification/tune/summary.json'))
        combined=tune['records']+records
        write_json(a.out/'verification/all_summary.json',dict(tasks=len(combined),records=combined,
            summary=summarize(combined),frozen=frozen,model=MODEL,
            new_calls=tune['new_calls']+result['new_calls'],
            reused_calls=tune['reused_calls']+result['reused_calls'],
            cost_cny=tune['cost_cny']+result['cost_cny'],policy=result['policy'],limitation=result['limitation']))
    lines=['# N7 frozen-threshold local verification','',result['policy'],result['limitation'],
        f"New calls {result['new_calls']}; exact reuse {result['reused_calls']}; new usage cost CNY {result['cost_cny']}.",'',
        '| Split | Support | Group | N | Method | F1 | P | R | TP | FP | FN | Empty | Positive fully rejected | TP loss vs Top8 |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        if r['split']=='all':continue
        for name,m in r['methods'].items():
            lines.append('| '+' | '.join(map(str,[r['split'],r['support'],r['group'],r['tasks'],name,*[round(m[k],5) for k in ('f1_c','precision','recall')],m['tp'],m['fp'],m['fn'],m['empty_predictions'],m['positive_fully_rejected'],m.get('tp_lost_vs_top8','n/a')]))+' |')
    write_text(out/'report.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
