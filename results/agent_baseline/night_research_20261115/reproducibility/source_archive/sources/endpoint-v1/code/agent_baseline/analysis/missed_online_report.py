"""Incremental paired Agent metrics and spending, including failed attempts."""
import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from agent_baseline.planner import usage_tokens,request_cost
from .io_retry import write_json


def load(path): return json.loads(path.read_text())


def lines(path,errors=None):
    if not path.exists():return []
    rows=[]
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        try:
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('Expected an object')
            rows.append(row)
        except (ValueError,TypeError) as exc:
            if errors is not None:errors.append(dict(path=str(path),line=number,error_type=type(exc).__name__,kind='parse_error'))
    return rows


def complete_final(final):
    return isinstance(final,dict) and bool(final.get('eval')) and final.get('submitted_by')!='error'


def ledger_cost(run):
    # Current RunGuard removes .txt and writes one hostname-pid file per process.
    directory=sum(float(p.read_text()) for p in (run/'budget'/'missed_online').glob('*.txt'))
    legacy=run/'budget'/'missed_online.txt'
    single=float(legacy.read_text()) if legacy.is_file() else 0.
    return max(directory,single),dict(process_directory_cny=directory,legacy_single_file_cny=single)


def metrics(values):
    counts={k:sum(v[k] for v in values) for k in ('tp','fp','fn')};tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return dict(counts,f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        precision=100*tp/(tp+fp) if tp+fp else None,recall=100*tp/(tp+fn) if tp+fn else None)


def referenced(value):
    if isinstance(value,str):return {value} if '#' in value else set()
    if isinstance(value,list):return set().union(*(referenced(v) for v in value)) if value else set()
    if isinstance(value,dict):return set().union(*(referenced(v) for v in value.values())) if value else set()
    return set()


def trace(directory,errors=None):
    calls=[c for row in lines(directory/'steps.jsonl',errors) for c in row.get('calls',[])]
    descendants=set();offered=0;selected=0;answer_use=False;submitted=False
    for call in calls:
        if call.get('tool')=='find_missed' and call.get('ok'):
            offered+=1;descendants.update(call.get('handles_created',[]));continue
        if referenced(call.get('args'))&descendants and call.get('ok'):
            descendants.update(call.get('handles_created',[]))
            selected+=call.get('tool') in ('select','to_mask')
            answer_use |= call.get('current_after') in descendants
            submitted |= call.get('tool')=='submit'
    return dict(total_tool_calls=len(calls),tool_calls=dict(Counter(c.get('tool') for c in calls)),
        tool_errors=[dict(tool=c.get('tool'),code=c.get('error_code')) for c in calls if not c.get('ok',True)],
        find_missed_calls=offered,candidate_selection_calls=selected,candidate_descendant_became_current=answer_use,
        candidate_descendant_submitted=submitted,
        adoption_definition='trace handle ancestry; does not assert positive pixel benefit')


def request_events(directory,errors=None):
    # Backend snapshots cover paid responses before recorder.step flushes them.
    events=lines(directory/'messages.jsonl',errors)
    for path in (directory/'missed_verifier').glob('*/response.json'):
        events.append(load(path))
    return events


def offline_cost(root):
    seen=set();cost=0.
    for phase in ('smoke','pilot'):
        for path in (root/phase/'answers').glob('*/usage/*.json'):
            row=load(path);raw=row.get('raw');raw=raw if isinstance(raw,dict) else {}
            identity=raw.get('id') or row.get('request_id') or str(path)
            if identity in seen:continue
            seen.add(identity);cost+=float(row.get('cost_cny') or 0.)
    return cost


def report(out,publish_root=False):
    out=Path(out);settings=load(out/'settings.json');sessions=[];seen=set();spent=0.;unpriced=0;api_errors=[];parse_errors=[]
    for task in settings['tasks']:
        for seed in settings['seeds']:
            for arm in settings['arms']:
                prefix=out/'run'/task['group']/f'seed{seed}'/arm/task['stem'];history=[]
                for attempt in sorted(prefix.glob('attempt_*')):
                    directory=attempt/task['task_type']/'val'/task['stem']
                    final=load(directory/'final.json') if (directory/'final.json').exists() else None
                    item=dict(attempt=attempt.name,directory=str(directory),final=final,trace=trace(directory,parse_errors),usage_cost_cny=0.)
                    for index,event in enumerate(request_events(directory,parse_errors)):
                        raw=event.get('raw');raw=raw if isinstance(raw,dict) else {}
                        uid=event.get('request_id') or raw.get('id') or (str(directory),event.get('backend'),event.get('requested_at'),event.get('total_attempt_id'),index)
                        if uid in seen:continue
                        seen.add(uid);usage=event.get('usage') or raw.get('usage')
                        if isinstance(usage,dict) and usage:
                            cost=request_cost('qwen3.7-flash',usage_tokens(usage));spent+=cost;item['usage_cost_cny']+=cost
                        elif event.get('error_type'):unpriced+=1
                        if event.get('error_type'):
                            api_errors.append(dict(directory=str(directory),error_type=event['error_type'],http_status=event.get('http_status')))
                    item['accounted_cost_cny']=max(item['usage_cost_cny'],(final or {}).get('cost_cny',0))
                    history.append(item)
                successful=[h for h in history if complete_final(h['final'])]
                selected=successful[-1] if successful else None
                sessions.append(dict(group=task['group'],seed=seed,arm=arm,stem=task['stem'],complete=bool(selected),
                    selected=selected,attempts=history,status='complete' if selected else 'failed_or_incomplete' if history else 'not_started'))
    ledger,ledger_details=ledger_cost(out/'run')
    final_cost=sum((attempt.get('final') or {}).get('cost_cny',0) for session in sessions for attempt in session['attempts'])
    groups={};pairs=[];pairing_errors=[];seed_summary={}
    for group in sorted({r['group'] for r in sessions}):
        groups[group]={}
        for seed in settings['seeds']:
            rows=[r for r in sessions if r['group']==group and r['seed']==seed];by={(r['stem'],r['arm']):r for r in rows}
            paired=[stem for stem in sorted({r['stem'] for r in rows}) if all(by[(stem,a)]['complete'] for a in settings['arms'])]
            aggregate={}
            for arm in settings['arms']:
                completed=[r for r in rows if r['arm']==arm and r['complete']]
                finals=[r['selected']['final'] for r in completed];traces=[r['selected']['trace'] for r in completed]
                aggregate[arm]=dict(completed=len(completed),expected=2,all_tasks_complete=len(completed)==2,
                    cost_cny=sum(h['accounted_cost_cny'] for r in rows if r['arm']==arm for h in r['attempts']),
                    all_attempt_tool_calls=sum(h['trace']['total_tool_calls'] for r in rows if r['arm']==arm for h in r['attempts']),
                    completed_tool_calls=sum(t['total_tool_calls'] for t in traces),
                    available_case_metrics=metrics([r['eval'] for r in finals]) if finals else None,
                    paired_metrics=metrics([by[(stem,arm)]['selected']['final']['eval'] for stem in paired]) if paired else None,
                    find_missed_adoption_rate=sum(t['find_missed_calls']>0 for t in traces)/len(traces) if traces else None,
                    descendant_current_rate=sum(t['candidate_descendant_became_current'] for t in traces)/len(traces) if traces else None,
                    termination_counts=dict(Counter(r.get('termination_reason') for r in finals)),
                    submitted_by_counts=dict(Counter(r.get('submitted_by') for r in finals)),
                    fallback_counts=dict(Counter(r.get('fallback_kind') for r in finals)),
                    model_submitted_metrics=metrics([r['eval'] for r in finals if r.get('submitted_by')=='model']) if any(r.get('submitted_by')=='model' for r in finals) else None,
                    tool_error_count=sum(len(t['tool_errors']) for t in traces))
            groups[group][str(seed)]=dict(paired_tasks=len(paired),arms=aggregate,cost_cny=sum(a['cost_cny'] for a in aggregate.values()))
            for stem in paired:
                baseline=by[(stem,'baseline')]['selected'];changed=by[(stem,'find_missed')]['selected']
                inputs=[]
                for entry in (baseline,changed):
                    path=Path(entry['directory'])/'input.json'
                    inputs.append(load(path) if path.exists() else {})
                for field in ('seed','model_parameters','init_mask_path'):
                    if not inputs[0].get(field) or inputs[0].get(field)!=inputs[1].get(field):
                        pairing_errors.append(dict(group=group,seed=seed,stem=stem,field=field,reason='missing_or_mismatched_paired_metadata'))
                pairs.append(dict(group=group,seed=seed,stem=stem,
                    baseline=metrics([baseline['final']['eval']]),find_missed=metrics([changed['final']['eval']]),
                    delta_f1_points=100*(changed['final']['eval']['f1']-baseline['final']['eval']['f1'])))
        seed_summary[group]={}
        for arm in settings['arms']:
            values=[v['arms'][arm]['paired_metrics']['f1'] for v in groups[group].values() if v['paired_tasks']==2]
            seed_summary[group][arm]=dict(complete_seed_pairs=len(values),expected_seeds=3,
                mean_f1=statistics.mean(values) if values else None,
                std_f1=statistics.stdev(values) if len(values)>1 else None)
    attributed_cost=sum(h['accounted_cost_cny'] for r in sessions for h in r['attempts'])
    total_cost=max(spent,ledger,final_cost,attributed_cost)
    result=dict(groups=groups,pairs=pairs,pairing_errors=pairing_errors,seed_summary=seed_summary,sessions=sessions,expected_sessions=36,completed=sum(r['complete'] for r in sessions),
        cost_cny=total_cost,usage_derived_cost_cny=spent,ledger_cost_cny=ledger,all_final_cost_cny=final_cost,
        ledger_details=ledger_details,group_attributed_cost_cny=attributed_cost,unattributed_ledger_cost_cny=max(0,total_cost-attributed_cost),
        total_tool_calls=sum(h['trace']['total_tool_calls'] for r in sessions for h in r['attempts']),parse_errors=parse_errors,
        cost_basis='actual response usage under repository price table; includes failed attempts; not provider invoice',
        failures_without_usage=unpriced,api_errors=api_errors,
        interpretation='joint candidate-tool and verifier intervention; missing pairs excluded and explicitly counted')
    write_json(out/'paired_report.json',result)
    write_json(out/'metric.json',dict(stage='online_incremental_report',cumulative=True,cost_cny=total_cost,
        completed=result['completed'],expected=36,paired=len(pairs),api_errors=len(api_errors),result_path=str(out/'paired_report.json')))
    if publish_root:
        prior=offline_cost(out.parent)
        write_json(out.parent/'metric.json',dict(stage='online_incremental_report',cumulative=True,cost_cny=prior+total_cost,
            offline_cost_cny=prior,online_cost_cny=total_cost,completed=result['completed'],expected=36,paired=len(pairs),
            api_errors=len(api_errors),parse_errors=len(parse_errors),result_path=str(out/'paired_report.json')))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--publish-root',action='store_true');args=p.parse_args()
    report(args.out,args.publish_root)

if __name__=='__main__':main()
