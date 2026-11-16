"""Offline-only legacy gate sensitivity; invalid answers abstain, never become no."""
import argparse
from collections import Counter
import os
from pathlib import Path
import numpy as np
from agent_baseline.evaluation import load_reference
from .n8_missed_assets import read,write
from .n8_missed_prepare import load_masks
from .n8_missed_pipeline import cost_total
from .n9_missed_assets import OUT as SOURCE
from .n9_missed_protocol import configurations
from .n9_missed_pipeline import config_key
from .n9_missed_score import metric,summarize_line

OUT=SOURCE/'legacy_all_source_dino020_abstain'


def decision(answer):
    value=answer.get('answer')
    return 'accept' if value=='yes' else 'reject' if value=='no' else 'invalid_abstain'


def audit(source,out):
    counts=Counter();invalid=[];all_rows=[]
    for path in sorted((source/'answers').glob('*/*.json')):
        answer=read(path);state=decision(answer)
        row=dict(path=str(path),uid=answer.get('uid',path.stem),variant=path.parent.name,
                 answer=answer.get('answer'),decision=state,finish_reason=answer.get('finish_reason'),
                 parse_failure=answer.get('parse_failure'))
        counts['total_answers']+=1;counts[state]+=1
        if row['finish_reason']=='length':counts['finish_reason_length']+=1
        if state=='invalid_abstain':invalid.append(row)
        all_rows.append(row)
    result=dict(counts=dict(counts),invalid_rate=counts['invalid_abstain']/max(1,counts['total_answers']),
        invalid_answers=invalid,all_answers=all_rows,raw_and_answers_modified=False,new_api_requests=0,
        policy='Use parsed answer field only; unknown/null stays invalid_abstain; no prefix/regex recovery')
    write(out/'answer_failure_audit.json',result)
    return result


def score(a):
    a.out.mkdir(parents=True,exist_ok=True);audited=audit(a.source,a.out)
    references={(r['group'],r['stem']):r['task'] for r in read(a.source/'evaluation_manifest.json')}
    states={(r['group'],r['stem']):r for r in read(a.source/'states_tune.json')}
    cases={};metadata={}
    for item in read(a.source/'fm1_manifest.json'):
        if item['split']!='tune':continue
        ref=references[item['group'],item['stem']]
        truth,valid=load_reference(ref['gt_mask'],ref.get('gt_ignore_mask'))
        with np.load(item['base'],allow_pickle=False) as z:base=z['start'].astype(bool)
        masks=load_masks(item['masks']);before=metric(base,truth,valid);state=states[item['group'],item['stem']]
        for config in configurations(item['profile']):
            key=config_key(config);metadata[key]=config;indexes=state['rankings'][key]
            pred=base.copy();accepted=[];invalid=[];rejected=[];charge=0.
            for index in indexes:
                uid=f"{item['group']}__{item['stem']}__n9_{index:04d}"
                path=a.source/'answers'/config['verifier']/(uid+'.json');answer=read(path)
                outcome=decision(answer);charge+=answer.get('cost_cny',0.)
                if outcome=='accept':pred|=masks[index];accepted.append(index)
                elif outcome=='reject':rejected.append(index)
                else:invalid.append(dict(index=index,uid=uid,path=str(path),answer=answer.get('answer'),
                    finish_reason=answer.get('finish_reason'),decision='invalid_abstain'))
            after=metric(pred,truth,valid)
            cases.setdefault(key,[]).append(dict(group=item['group'],stem=item['stem'],profile=item['profile'],split='tune',
                config=config,baseline=before,after=after,delta_f1=after['f1']-before['f1'],
                added_tp=after['tp']-before['tp'],added_fp=after['fp']-before['fp'],calls=len(indexes),
                submitted_indexes=indexes,accepted_indexes=accepted,rejected_indexes=rejected,invalid_answers=invalid,
                answer_cost_cny=charge,baseline_path=item['base'],candidate_masks=item['masks']))
    summaries={}
    for key,rows in cases.items():
        lines={}
        for group in sorted({r['group'] for r in rows}):
            chosen=[r for r in rows if r['group']==group];value=summarize_line(chosen)
            value.update(invalid_calls=sum(len(r['invalid_answers']) for r in chosen),
                tasks_with_invalid=sum(bool(r['invalid_answers']) for r in chosen))
            lines[group]=value
        summaries[key]=dict(config=metadata[key],split='tune',lines=lines)
    write(a.out/'scores_tune.json',dict(cases=cases,summaries=summaries,
        role='Legacy all-source DINO .20 sensitivity only; not formal S5-only calibration',
        invalid_policy='Preserve baseline at invalid candidate; leave status unknown/invalid_abstain, never relabel no',
        no_configuration_frozen=True,online_authorized=False,new_api_requests=0))
    write(a.out/'metric.json',dict(stage='legacy_abstain_sensitivity',status='complete',
        actual_parent_cost_cny=cost_total(a.source),new_api_cost_cny=0,cost_cny=cost_total(a.source),
        cumulative=True,total_answers=audited['counts']['total_answers'],
        invalid_answers=audited['counts'].get('invalid_abstain',0),invalid_rate=audited['invalid_rate']))
    print(dict(status='complete',counts=audited['counts'],new_api_requests=0),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for audit and scoring')
    score(a)

if __name__=='__main__':main()
