"""Offline-only labels: tune joint choice, frozen confirm, complete group metrics."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from .n8_missed_assets import read,write,OUT
from .n8_missed_prepare import load_masks,union,gt_lookup,truth,aggregate,AREA_POWERS,RESIDUAL_GRID
from .n8_missed_pipeline import VARIANTS,cost_total,should_verify
from .missed_sources import pixel_counts
from .missed_verifier_study import metrics

def configurations(split,frozen=None):
    if split=='confirm':
        return [dict(alpha=frozen['alpha'],residual_area=frozen['residual_area'],k=k,variant=v) for v in VARIANTS for k in (5,10)]
    return [dict(alpha=alpha,residual_area=area,k=k,variant=v) for alpha in AREA_POWERS for area in RESIDUAL_GRID for v in VARIANTS for k in (5,10)]

def config_id(c):return f"alpha{c['alpha']}_res{c['residual_area']}_{c['variant']}_k{c['k']}"

def summary(rows):
    result=dict(tasks=len(rows),pooled=aggregate([r['metrics'] for r in rows]),baseline=aggregate([r['baseline'] for r in rows]),
        macro_f1=float(np.mean([r['metrics']['f1'] for r in rows])),baseline_macro_f1=float(np.mean([r['baseline']['f1'] for r in rows])),
        added_tp=sum(r['added_tp'] for r in rows),added_fp=sum(r['added_fp'] for r in rows),
        wins=sum(r['delta_f1']>1e-9 for r in rows),losses=sum(r['delta_f1']< -1e-9 for r in rows),
        vlm_trigger_tasks=sum(r['trigger'] for r in rows),vlm_trigger_rate=sum(r['trigger'] for r in rows)/len(rows),
        find_missed_generation_rate=1.,verifier_candidate_calls=sum(r['verifier_calls'] for r in rows),
        counterfactual_answer_cost_cny=sum(r['answer_cost_cny'] for r in rows))
    result['ties']=len(rows)-result['wins']-result['losses'];result['harm_rate']=result['losses']/len(rows)
    result['pooled_delta']=result['pooled']['f1']-result['baseline']['f1']
    result['macro_delta']=result['macro_f1']-result['baseline_macro_f1']
    empty=[r for r in rows if r['baseline']['tp']+r['baseline']['fn']==0]
    result.update(empty_gt_tasks=len(empty),empty_gt_fp=sum(r['metrics']['fp'] for r in empty),empty_gt_added_fp=sum(r['added_fp'] for r in empty))
    return result

def all_groups(rows):
    result=summary(rows)
    result['groups']={g:summary([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
    result['group_pooled_mean_f1']=float(np.mean([g['pooled']['f1'] for g in result['groups'].values()]))
    result['baseline_group_pooled_mean_f1']=float(np.mean([g['baseline']['f1'] for g in result['groups'].values()]))
    result['group_macro_mean_f1']=float(np.mean([g['macro_f1'] for g in result['groups'].values()]))
    result['baseline_group_macro_mean_f1']=float(np.mean([g['baseline_macro_f1'] for g in result['groups'].values()]))
    result['group_pooled_mean_delta']=result['group_pooled_mean_f1']-result['baseline_group_pooled_mean_f1']
    result['group_macro_mean_delta']=result['group_macro_mean_f1']-result['baseline_group_macro_mean_f1']
    return result

def score(a):
    started=time.monotonic();lookup=gt_lookup();frozen=read(a.out/'frozen_config.json') if a.split=='confirm' else None
    configs=configurations(a.split,frozen);cases={config_id(c):[] for c in configs}
    states={(r['group'],r['stem']):r for r in read(a.out/f'fm2_states_{a.split}.json')}
    diagnostics=[]
    for item in read(a.out/'fm1_manifest.json'):
        if item['split']!=a.split:continue
        gt=truth(lookup[item['group'],item['stem']],item['group']);masks=load_masks(item['masks'])
        with np.load(item['base'],allow_pickle=False) as z:base=z['start'].astype(bool)
        baseline=metrics(pixel_counts(base,gt));state=states[item['group'],item['stem']]
        for c in configs:
            trigger=should_verify(state,c['residual_area'])
            indexes=state['rankings'][str(c['alpha'])][:c['k']] if trigger else []
            pred=base.copy();covered=np.zeros(base.shape,bool);accepted=[];charge=0
            for i in indexes:
                uid=f"{item['group']}__{item['stem']}__n8_{i:04d}"
                answer=read(a.out/'answers'/c['variant']/(uid+'.json'))
                if answer.get('answer') not in ('yes','no'):raise RuntimeError('Unresolved verifier answer: '+uid+' '+c['variant'])
                covered|=masks[i];charge+=answer.get('cost_cny',0.)
                if answer['answer']=='yes':pred|=masks[i];accepted.append(i)
            value=metrics(pixel_counts(pred,gt))
            cases[config_id(c)].append(dict(group=item['group'],stem=item['stem'],split=a.split,config=c,baseline=baseline,metrics=value,
                delta_f1=value['f1']-baseline['f1'],added_tp=value['tp']-baseline['tp'],added_fp=value['fp']-baseline['fp'],trigger=trigger,
                residual_area=state['residual_area'],residual_components=state['residual_components'],net_candidates=state['net_candidates'],
                submitted_indexes=indexes,accepted_indexes=accepted,verifier_calls=len(indexes),answer_cost_cny=charge,
                topk_candidate_tp=int((covered&gt&~base).sum()),topk_candidate_fp=int((covered&~gt&~base).sum()),
                baseline_path=item['base'],candidate_masks=item['masks'],image_t1=item['image_t1'],image_t2=item['image_t2'],gt_path=lookup[item['group'],item['stem']]))
        # Stage-quality diagnostics are offline only and include every intervention.
        stage={}
        with np.load(item['masks'],allow_pickle=False) as z:
            for key in ('original_all','s5_dino020_all','direction_routed','dilation_only','area25'):
                m=np.unpackbits(z[key],count=gt.size).reshape(gt.shape).astype(bool)
                stage[key]=dict(tp=int((m&gt&~base).sum()),fp=int((m&~gt&~base).sum()))
        tau=read(a.out/'frozen_semantic.json')['threshold']
        m=union([mask for mask,candidate in zip(masks,item['candidates']) if candidate['semantic_delta']>=tau],base.shape)
        stage['semantic']=dict(tp=int((m&gt&~base).sum()),fp=int((m&~gt&~base).sum()))
        diagnostics.append(dict(group=item['group'],stem=item['stem'],remaining_fn=int((gt&~base).sum()),stages=stage))
    reports={key:dict(config=next(r['config'] for r in rows),**all_groups(rows)) for key,rows in cases.items()}
    write(a.out/f'scores_{a.split}.json',dict(cases=cases,summaries=reports));write(a.out/f'candidate_diagnostics_{a.split}.json',diagnostics)
    if a.split=='tune':
        eligible=[r for r in reports.values() if r['pooled_delta']>0 and r['group_pooled_mean_delta']>0]
        chosen=max(eligible or list(reports.values()),key=lambda r:((r['pooled_delta']+r['group_pooled_mean_delta'])/2,-r['verifier_candidate_calls'],-r['config']['k']))
        frozen=dict(chosen['config'],semantic_threshold=tau,tune_id=config_id(chosen['config']),tune_both_positive=bool(eligible),
            criterion='maximize mean of pooled and three-group-pooled deltas; prefer jointly positive when available; ties fewer calls then smaller K then fixed enumeration',
            tune_metrics={k:chosen[k] for k in ('pooled','macro_f1','pooled_delta','macro_delta','group_pooled_mean_f1','group_macro_mean_f1')})
        write(a.out/'frozen_config.json',frozen)
    else:
        selected=reports[config_id(frozen)]
        decisions={key:dict(pooled_pass=r['pooled_delta']>0,group_pooled_pass=r['group_pooled_mean_delta']>0,
                           group_macro_pass=r['group_macro_mean_delta']>0) for key,r in reports.items()}
        primary=decisions[config_id(frozen)]
        write(a.out/'offline_decision.json',dict(primary_config=frozen,primary=primary,
            promote_primary=primary['pooled_pass'] and primary['group_pooled_pass'],
            promote_primary_strict_both_group_definitions=all(primary.values()),arms=decisions,
            cost_cny=cost_total(a.out),reason='N8 threshold is pooled and three-group mean F1 versus never; report both meanings of group mean'))
        write(a.out/'selected_cases.json',cases[config_id(frozen)])
    write(a.out/f'score_execution_{a.split}.json',dict(status='complete',tasks=15,configurations=len(configs),elapsed_seconds=time.monotonic()-started,slurm_job=os.environ['SLURM_JOB_ID']))
    write(a.out/'metric.json',dict(stage='scored_'+a.split,status='complete',completed=15,cost_cny=cost_total(a.out),cumulative=True,primary_config=config_id(frozen)))
    print('Scored '+a.split+'; frozen '+config_id(frozen))

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['tune','confirm'],required=True);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    score(a)

if __name__=='__main__':main()
