"""Post-session reporting only. No inference, parameter selection, or API calls."""
import argparse
import json
import os
from pathlib import Path
from collections import Counter
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import pooled,GROUPS
from .n8_t3_online_metrics import SEEDS
from .n8_t3_cost import line_ledger

SUPPORTS=('all','positive','empty','remaining_gt>=16_diagnostic')

def matches(row,support):
    n=row['remaining_gt_pixels']
    return support=='all' or support=='positive' and n>0 or support=='empty' and n==0 or support=='remaining_gt>=16_diagnostic' and n>=16


def summarize(rows):
    scored=[r for r in rows if 'counts' in r];states=Counter(r['state'] for r in rows)
    attempted=sum(r['retrieve_calls']>0 for r in rows);returned=sum(r['retrieve_returns']>0 for r in rows)
    positive=[r for r in scored if r['remaining_gt_pixels']>0];empty=[r for r in scored if r['remaining_gt_pixels']==0]
    correct=sum(r['counts'][1]==0 for r in empty)
    return dict(expected=len(rows),scored=len(scored),remaining_gt_pixels=sum(r['remaining_gt_pixels'] for r in rows),
        scored_gt_pixels=sum(r['remaining_gt_pixels'] for r in scored),**pooled(scored),
        states={k:states[k] for k in ('successful_submit','fallback','error','missing_final')},
        no_model_submit=sum(r['state']!='missing_final' and r.get('submitted_by')!='model' for r in rows),
        termination_reasons=dict(Counter(r.get('termination_reason','missing') for r in rows)),
        empty_tasks_scored=len(empty),empty_correct=correct,empty_correct_rate=correct/len(empty) if empty else None,
        empty_fp=sum(r['counts'][1] for r in empty),positive_tasks_scored=len(positive),
        positive_fully_empty=sum(r['counts'][0]+r['counts'][1]==0 for r in positive),
        retrieve_attempted_tasks=attempted,retrieve_returned_tasks=returned,
        retrieve_attempt_rate=attempted/len(rows) if rows else None,retrieve_return_rate=returned/len(rows) if rows else None,
        retrieve_calls=sum(r['retrieve_calls'] for r in rows),retrieve_returns=sum(r['retrieve_returns'] for r in rows),
        current_session_cost_cny=sum(r['cost_cny'] for r in rows))


def tables(records):
    result=[]
    for seed in (*SEEDS,'all_seeds'):
        for arm in ('baseline','n8'):
            for group in ('全部',*GROUPS):
                for support in SUPPORTS:
                    rows=[r for r in records if r['arm']==arm and (seed=='all_seeds' or r['seed_offset']==seed)
                          and (group=='全部' or r['group']==group) and matches(r,support)]
                    result.append(dict(seed=seed,arm=arm,group=group,support=support,**summarize(rows)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run reporting via Slurm')
    source=json.loads(read_text(a.output/'online_summary.json'))
    manifest=[json.loads(s) for s in read_text(a.tasks).splitlines() if s.strip()]
    support={r['id']:int(((read_image(r['gt_mask'])>0)&~(read_image(r['exemplar_mask'])>0)).sum()) for r in manifest}
    records=[]
    for old in source['records']:
        row=dict(old);folder=Path(row['final_path']).parent
        calls=[]
        if exists(folder/'steps.jsonl'):
            for line in read_text(folder/'steps.jsonl').splitlines():
                if line.strip():calls.extend(json.loads(line).get('calls',[]))
        row.update(remaining_gt_pixels=support[row['id']],retrieve_calls=sum(c.get('tool')=='retrieve_example' for c in calls),
                   retrieve_returns=sum(c.get('tool')=='retrieve_example' and c.get('ok',False) for c in calls),
                   retrieve_materializations=len(list((folder/'n8_retrieval').glob('call_*/inference.json'))))
        records.append(row)
    summary=tables(records);pairs=[]
    for seed in SEEDS:
        for task in support:
            pair={r['arm']:r for r in records if r['seed_offset']==seed and r['id']==task}
            if all('counts' in pair.get(arm,{}) for arm in ('baseline','n8')):
                b,n=pair['baseline'],pair['n8'];bm,nm=pooled([b]),pooled([n])
                pairs.append(dict(id=task,seed=seed,delta_f1=nm['f1_c']-bm['f1_c'],delta_tp=nm['tp']-bm['tp'],delta_fp=nm['fp']-bm['fp'],
                    baseline_state=b['state'],n8_state=n['state'],baseline_retrieve_calls=b['retrieve_calls'],n8_retrieve_calls=n['retrieve_calls']))
    costs=source['cost_audit'];offline=line_ledger(a.output/'step3/verification')['known_cost_cny']
    cost=dict(offline_cny=offline,online_cny=source['cost_cny'],total_cny=offline+source['cost_cny'],
        known_failed_cny=sum(c.get('known_failed_cost_cny',0) for c in costs),
        usage_missing_failures=sum(c.get('usage_missing_failures',0) for c in costs),
        interrupted_attempts_cny=sum(c['cost_cny'] for c in costs if c.get('interrupted')),
        note='Current-session table costs exclude interrupted attempts; global total includes them. Missing usage is unknown, not asserted free.')
    value=dict(summary=summary,records=records,paired_differences=pairs,cost=cost,
        interpretation='All sessions retained. Retrieval-use subsets are descriptive, not randomized. Same exposed dev10; three seeds do not add independent images.')
    write_json(a.output/'online_report.json',value)
    lines=['# N8 T3 online paired report','',value['interpretation'],'',
        'Main gate/K3 failed offline. This online arm is the predeclared ungated pool Top5 control that met both original numerical gates in tune and confirm; eligibility was recognized after confirm exposure.',
        '', '| Seed | Arm | Group | Support | Expected/scored | GT pixels | P | R | F1 | Empty correct/scored | Empty FP | Positive empty/scored | No model submit | Retrieve attempted/returned tasks | Calls/returns | Current cost |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if not r['expected']:continue
        values=[r['seed'],r['arm'],r['group'],r['support'],f"{r['expected']}/{r['scored']}",r['remaining_gt_pixels'],
                *[f"{r[k]:.5f}" for k in ('precision','recall','f1_c')],f"{r['empty_correct']}/{r['empty_tasks_scored']}",r['empty_fp'],
                f"{r['positive_fully_empty']}/{r['positive_tasks_scored']}",r['no_model_submit'],
                f"{r['retrieve_attempted_tasks']}/{r['retrieve_returned_tasks']}",f"{r['retrieve_calls']}/{r['retrieve_returns']}",f"{r['current_session_cost_cny']:.7f}"]
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines+=['','Rates use expected tasks as denominator; JSON preserves explicit rates, termination reasons and four completion states. Empty/no-submit results are not removed. Zero scored support is missing evidence, not measured zero performance.',
            '',f"Known total CNY {cost['total_cny']:.7f}; offline {offline:.7f}; online {cost['online_cny']:.7f}; paid failures {cost['known_failed_cny']:.7f}; missing-usage failures {cost['usage_missing_failures']}.",
            '', 'Final interpretation: compare overall paired end states first, then identify whether retrieve_example was actually called and proposals returned. A non-use session cannot establish provider efficacy. Inspect candidate-to-selection-to-verification-to-submit traces for claimed mechanisms.']
    write_text(a.output/'online_report.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
