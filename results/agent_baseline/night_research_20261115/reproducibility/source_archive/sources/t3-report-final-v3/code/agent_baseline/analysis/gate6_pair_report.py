"""Paired development-only Gate6 evidence; preserve errors and incomplete arms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from .coordinate_audit import audit,coordinate_error,write_json


def retries(directory):
    meta=json.loads((directory/'input.json').read_text());failed={};events=[];current=None
    def signature(name,args,answer):
        context=answer if name=='ask' and not (args or {}).get('regions') else None
        return name,json.dumps(args,sort_keys=True),context
    for line in (directory/'steps.jsonl').read_text().splitlines():
        row=json.loads(line);raw=(row.get('planner') or {}).get('raw') or {}
        choices=raw.get('choices',[]) if isinstance(raw,dict) else []
        native=(choices[0].get('message',{}).get('tool_calls') or []) if choices else []
        calls=row.get('calls',[]);requested=[]
        for item in native:
            function=item.get('function') or {}
            try:args=json.loads(function.get('arguments','{}'))
            except (ValueError,TypeError):continue
            requested.append((function.get('name'),args))
        if not native:requested=[(c['tool'],c.get('args')) for c in calls]
        for name,args in requested:
            key=signature(name,args,current)
            if key in failed:
                events.append(dict(step=row['step'],tool=name,args=args,prior_failure=failed[key],
                    filtered_by_harness=not bool(calls)))
        for call in calls:
            key=signature(call['tool'],call.get('args'),call.get('current_before',current))
            if not call.get('ok',True):failed[key]=dict(step=row['step'],error_code=call.get('error_code'),coordinate=bool(coordinate_error(call,meta.get('image_shape'))))
            else:failed.pop(key,None)
        current=row.get('current_mask',current)
    return events


def report(run,manifest,output):
    tasks=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    arms={};errors=[];all_rows={}
    for arm in ('off','on'):
        rows=[];counts=Counter();cost=0
        for task in tasks:
            name=Path(task['id']).stem;directory=run/f'gate6_{arm}'/'T1.3'/'val'/name
            try:
                row=audit((f'gate6_{arm}','T1.3','val',directory));final=json.loads((directory/'final.json').read_text())
                if not final.get('eval'):raise ValueError('Final has no evaluation; inspect input/backend failure')
                meta=json.loads((directory/'input.json').read_text())
                if bool(meta.get('backends',{}).get('gate6'))!=(arm=='on'):raise ValueError('Wrong Gate6 arm metadata')
                retry=retries(directory);row.update(invalid_retries=len(retry),invalid_coordinate_retries=sum(e['prior_failure']['coordinate'] for e in retry),retry_evidence=retry,
                    final=final,seed=meta.get('seed'),model_parameters=meta.get('model_parameters'))
                rows.append(row);cost+=final['cost_cny']
                for field in ('tp','fp','fn'):counts[field]+=final['eval'][field]
            except (OSError,ValueError,KeyError) as error:errors.append(dict(arm=arm,name=name,error=str(error),directory=str(directory)))
        denominator=2*counts['tp']+counts['fp']+counts['fn']
        arms[arm]=dict(completed=len(rows),coordinate_bad_args=sum(r['coordinate_bad_args'] for r in rows),
            affected_sessions=sum(bool(r['coordinate_bad_args']) for r in rows),invalid_retries=sum(r['invalid_retries'] for r in rows),
            invalid_coordinate_retries=sum(r['invalid_coordinate_retries'] for r in rows),
            no_submit=sum(r['termination']=='no_submit' for r in rows),other_bad_args=sum(r['other_bad_args'] for r in rows),
            termination_counts=dict(Counter(r['termination'] for r in rows)),
            pooled_f1=200*counts['tp']/denominator if denominator else None,
            mean_image_f1=100*statistics.mean(r['f1'] for r in rows) if rows else None,cost_cny=cost,counts=dict(counts))
        all_rows[arm]={r['name']:r for r in rows}
    pairs=[]
    for name in sorted(set(all_rows['off']) & set(all_rows['on'])):
        off,on=all_rows['off'][name],all_rows['on'][name]
        if off['seed']!=on['seed'] or off['model_parameters']!=on['model_parameters']:
            errors.append(dict(name=name,error='Paired seed/model mismatch'))
        pairs.append(dict(name=name,off=off,on=on,f1_delta_points=100*(on['f1']-off['f1'])))
    delta={k:arms['on'][k]-arms['off'][k] for k in ('coordinate_bad_args','affected_sessions','invalid_retries','invalid_coordinate_retries','no_submit')}
    if arms['on']['pooled_f1'] is not None and arms['off']['pooled_f1'] is not None:delta['pooled_f1_points']=arms['on']['pooled_f1']-arms['off']['pooled_f1']
    output.mkdir(parents=True,exist_ok=True)
    value=dict(expected_tasks=len(tasks),arms=arms,on_minus_off=delta,pairs=pairs,errors=errors,
        scope='Same development tasks, seed and settings; one paired run. Gate6 includes coordinate metadata plus feature quantiles; not a coordinate-only causal ablation.',
        invalid_retry_definition='Request repeats identical tool/arguments after an unresolved failure, including requests filtered by the harness; implicit ask uses the same current-answer state.')
    write_json(output/'paired_results.json',value)
    ledger=sum(float(p.read_text()) for p in (run/'budget/gate6_actual').glob('*.txt'))
    recorded=sum(a['cost_cny'] for a in arms.values())
    write_json(output/'metric.json',dict(paired_tasks=len(pairs),read_failures=len(errors),
        cost_cny=max(ledger,recorded),ledger_cost_cny=ledger,final_records_cost_cny=recorded,cumulative=True))
    if errors or len(pairs)!=len(tasks):raise SystemExit('Incomplete paired experiment; partial results archived for repair')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=report(a.run,a.manifest,a.output)
    print(json.dumps(dict(arms=result['arms'],on_minus_off=result['on_minus_off']),ensure_ascii=False))


if __name__=='__main__':main()
