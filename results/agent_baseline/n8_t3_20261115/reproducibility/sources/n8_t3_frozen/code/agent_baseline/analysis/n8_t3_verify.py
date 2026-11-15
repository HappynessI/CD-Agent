"""T3-3 real evidence-card verification on rebuilt pools; tune-frozen K then confirm."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from agent_baseline.exemplar_semantics import request,MODEL
from .night_t3_direct4 import SYSTEM as BASE_SYSTEM,four_views
from .night_t3_grounding import API_SEED
from .t3_routes_match import T3_QUESTION
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,pooled
from .n8_t3_common import packed_load,packed_save,evaluate,report
from .n8_t3_system import evidence_card,passes_gate,VERIFY_K
from .n8_t3_cost import request_ledger,line_ledger,stage_ledger

SYSTEM=BASE_SYSTEM+' Two numerical evidence lines provide auxiliary model predictions, not ground truth. Check the directed change in RGB; probabilities and similarity alone do not establish a match.'


def choose_k(records):
    if any(r['split']!='tune' for r in records):raise ValueError('K selection requires tune exclusively')
    table=[dict(k=k,metrics=pooled([r['methods'][f'verified_top{k}'] for r in records])) for k in VERIFY_K]
    best=max(table,key=lambda r:(r['metrics']['f1_c'],-r['k']))
    return dict(k=best['k'],candidates=table,objective='tune verified all-task pooled F1, tie smaller K')


def budget_cost(root):
    return line_ledger(root)['known_cost_cny']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['tune','confirm'],required=True)
    p.add_argument('--budget-cny',type=float,default=50.);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    root=a.out/'step3';out=root/'verification'/a.split
    gate=json.loads(read_text(root/'gate_frozen.json'));system=json.loads(read_text(a.out/'step2/system_frozen.json'))
    frozen=json.loads(read_text(root/'final_frozen.json')) if a.split=='confirm' else None
    ks=VERIFY_K if a.split=='tune' else (frozen['k'],);budget_k=max(ks)
    prepared=json.loads(read_text(root/'prepare'/a.split/'summary.json'))['records'];prior={r['id']:r for r in prepared}
    spend=budget_cost(root/'verification');records=[];started=time.monotonic()
    write_json(out/'settings.json',dict(model=MODEL,seed=API_SEED,temperature=0,max_tokens=768,gate=gate['selected'],system=system['system_method'],
        request_top_k=budget_k,policy='Explicit yes accepts; no/unknown rejects. Evidence input differs from N7, all requests new except own completed N8 resume.',
        system_prompt=SYSTEM,budget_cny=a.budget_cny))
    for row in load_dev(a.run):
        if row['id'] not in prior:continue
        source=prior[row['id']];folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        prep=root/'prepare'/a.split/'tasks'/row['id'];inference=json.loads(read_text(prep/'inference.json'))
        masks,saved=packed_load(prep/'pool.npz');order=saved['order'].tolist()
        triggered=passes_gate(source['candidates'],source['z'],source['top1_ptrans'],gate['selected']['z'],gate['selected']['p'])
        ids=order[:budget_k] if triggered else []
        ex=read_image(row['exemplar_mask'])>0;answers={}
        images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')] if ids else None
        for index in ids:
            target=folder/'requests'/str(index);already=exists(target/'answer.json')
            if not already and spend>=a.budget_cny:raise RuntimeError('Actual N8 T3 usage reached allocated budget; request reallocation')
            card=evidence_card(inference['observable'],inference['candidates'][index],inference['rank'])
            question=T3_QUESTION+'\n'+card
            before=request_ledger(target)['known_cost_cny']
            try:
                response=request(target,SYSTEM,question,four_views(images,ex,masks[index]),API_SEED)
            finally:
                spend+=request_ledger(target)['known_cost_cny']-before
                write_json(a.out/'api_cost_status.json',line_ledger(root/'verification'))
            label=response['parsed'].get('change')
            answers[index]=dict(answer=label if label in ('yes','no') else 'unknown',cost_cny=request_ledger(target)['known_cost_cny'],
                new_mask_vs_original_R1=inference['new_mask_vs_original_R1'][index],source=str(target/'answer.json'))
            write_json(folder/'request_progress.json',dict(completed=len(answers),planned=len(ids),line_actual_cost_cny=spend))
        # Inputs contain no task class or GT; GT scoring starts only after every planned response is saved.
        truth=read_image(row['gt_mask'])>0;methods={'original_R1_top8':source['methods']['original_R1_top8'],
            'always_empty':source['methods']['always_empty'],'pool_top5':source['methods']['pool_top5']}
        predictions={}
        for k in ks:
            selected=order[:k] if triggered else []
            accepted=[i for i in selected if answers[i]['answer']=='yes']
            methods[f'gate_top{k}']=evaluate(masks,selected,truth,ex)
            methods[f'verified_top{k}']=evaluate(masks,accepted,truth,ex)
            predictions[f'gate_top{k}']=masks[selected].any(axis=0) if selected else np.zeros_like(ex)
            predictions[f'verified_top{k}']=masks[accepted].any(axis=0) if accepted else np.zeros_like(ex)
        packed_save(folder/'predictions.npz',np.stack(list(predictions.values())),methods=np.asarray(list(predictions)))
        record=dict(id=row['id'],split=a.split,group=source['group'],remaining_gt_pixels=source['remaining_gt_pixels'],methods=methods,
            candidates=source['candidates'],new_candidates=source['new_candidates'],triggered=triggered,z=source['z'],top1_ptrans=source['top1_ptrans'],
            verification={str(i):v for i,v in answers.items()},new_calls=len(answers),
            new_mask_calls=sum(v['new_mask_vs_original_R1'] for v in answers.values()),unknown=sum(v['answer']=='unknown' for v in answers.values()),
            cost_cny=sum(v['cost_cny'] for v in answers.values()))
        write_json(final,record);records.append(record);print('N8 evidence',a.split,len(records),row['id'],len(answers),flush=True)
    if a.split=='tune':
        choice=choose_k(records);frozen=dict(**choice,system=system,gate=gate,model=MODEL,online_selection='This frozen main configuration only; no confirm winner selection')
        write_json(root/'final_frozen.json',frozen)
    for record in records:
        record['methods']['N8_gate_unverified']=record['methods'][f"gate_top{frozen['k']}"]
        record['methods']['N8_verified']=record['methods'][f"verified_top{frozen['k']}"]
        write_json(out/'tasks'/record['id']/'metrics.json',record)
    ledger=stage_ledger(out)
    extra=dict(selection=frozen,new_calls=sum(r['new_calls'] for r in records),new_mask_calls=sum(r['new_mask_calls'] for r in records),
        cost_cny=ledger['known_cost_cny'],usage_ledger=ledger,unknown=sum(r['unknown'] for r in records),elapsed_seconds=time.monotonic()-started)
    report(out,records,'T3-3 evidence verification '+a.split,**extra)
    if a.split=='confirm':
        tune=json.loads(read_text(root/'verification/tune/summary.json'))
        report(root,tune['records']+records,'N8 final dev60',selection=frozen,new_calls=tune['new_calls']+extra['new_calls'],
               cost_cny=tune['cost_cny']+extra['cost_cny'],new_mask_calls=tune['new_mask_calls']+extra['new_mask_calls'])
        positive=[r for r in records if r['remaining_gt_pixels']>0];empty=[r for r in records if r['remaining_gt_pixels']==0]
        old=pooled([r['methods']['original_R1_top8'] for r in positive]);new=pooled([r['methods']['N8_verified'] for r in positive])
        old_empty=pooled([r['methods']['original_R1_top8'] for r in empty]);new_empty=pooled([r['methods']['N8_verified'] for r in empty])
        reduction=1-new_empty['fp']/old_empty['fp'] if old_empty['fp'] else 0
        passed=new['f1_c']>old['f1_c'] and reduction>.5
        write_json(a.out/'advancement.json',dict(passed=passed,positive_baseline=old,positive_N8=new,empty_baseline=old_empty,empty_N8=new_empty,
            empty_fp_reduction=reduction,frozen_k=frozen['k'],frozen_system=system['system_method'],
            next='10 tasks x 2 arms x 3 seeds true online' if passed else 'ten-case visualization diagnostics; no online',
            cost_cny=tune['cost_cny']+extra['cost_cny']))
    write_json(out/'status.json',dict(status='complete',tasks=len(records),new_calls=extra['new_calls'],cost_cny=extra['cost_cny']))


if __name__=='__main__':main()
