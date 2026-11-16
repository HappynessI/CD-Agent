"""N9 online progress/cost and independent dataset-line paired summaries."""
import argparse
from pathlib import Path
from .n8_missed_assets import read, write
from .n8_missed_pipeline import cost_total
from .n9_missed_assets import OUT
from .missed_online_report import complete_final, lines, trace
from .missed_verifier_study import metrics
from agent_baseline.planner import usage_tokens, request_cost


def pool(rows):
    return metrics([sum(r['eval'][k] for r in rows) for k in ('tp','fp','fn')])


def report(root):
    root=Path(root); online=root/'online'; settings=read(online/'settings.json')
    sessions=[]; events_seen=set(); usage_cost=0.; failure_events=[]
    for task in settings['tasks']:
        for seed in settings['seeds']:
            for arm in settings['arms']:
                prefix=online/'run'/task['group']/f'seed{seed}'/arm/task['stem']
                chosen=None
                for attempt in sorted(prefix.glob('attempt_*')):
                    directory=attempt/task['task_type']/'val'/task['stem']
                    final=read(directory/'final.json') if (directory/'final.json').exists() else None
                    if final and complete_final(final): chosen=(directory,final)
                    events=[(str(directory/'messages.jsonl')+':'+str(i),e) for i,e in enumerate(lines(directory/'messages.jsonl'))]
                    events += [(str(p),read(p)) for p in (directory/'n9_missed').glob('call_*/candidate_*/attempt_*.json')]
                    for origin,event in events:
                        raw=event.get('raw') or {}; raw=raw if isinstance(raw,dict) else {}
                        uid=event.get('request_id') or raw.get('id') or origin
                        if uid in events_seen: continue
                        events_seen.add(uid)
                        usage=event.get('usage') or raw.get('usage')
                        if usage: usage_cost+=request_cost('qwen3.7-flash',usage_tokens(usage))
                        if event.get('error_type'): failure_events.append(dict(origin=origin,error_type=event['error_type'],http_status=event.get('http_status')))
                row=dict(group=task['group'],stem=task['stem'],seed=seed,arm=arm,complete=chosen is not None)
                if chosen:
                    directory,final=chosen
                    row.update(directory=str(directory),eval=final['eval'],cost_cny=final.get('cost_cny',0.),
                        submitted_by=final.get('submitted_by'),trace=trace(directory))
                sessions.append(row)
    ledger=sum(float(p.read_text()) for p in (online/'run/budget/n9_missed_online').glob('*.txt'))
    complete_cost=sum(r.get('cost_cny',0.) for r in sessions)
    accounted=max(usage_cost,ledger,complete_cost)
    summaries={}
    for group in sorted({t['group'] for t in settings['tasks']}):
        summaries[group]={}
        for seed in settings['seeds']:
            before=[r for r in sessions if r['group']==group and r['seed']==seed and r['arm']=='baseline' and r['complete']]
            after=[r for r in sessions if r['group']==group and r['seed']==seed and r['arm']=='find_missed' and r['complete']]
            paired=sorted({r['stem'] for r in before}&{r['stem'] for r in after})
            b={r['stem']:r for r in before}; f={r['stem']:r for r in after}
            if paired:
                bm=pool([b[n] for n in paired]); fm=pool([f[n] for n in paired])
                delta=[metrics([f[n]['eval'][k] for k in ('tp','fp','fn')])['f1']-
                       metrics([b[n]['eval'][k] for k in ('tp','fp','fn')])['f1'] for n in paired]
                summaries[group][str(seed)]=dict(paired_tasks=len(paired),expected=10,complete=len(paired)==10,
                    baseline=bm,find_missed=fm,delta_f1=fm['f1']-bm['f1'],macro_delta=sum(delta)/len(delta),
                    wins=sum(d>1e-9 for d in delta),losses=sum(d< -1e-9 for d in delta),ties=sum(abs(d)<=1e-9 for d in delta),
                    find_missed_calls=sum(f[n]['trace']['find_missed_calls'] for n in paired))
    result=dict(status='complete' if all(r['complete'] for r in sessions) else 'partial',
        sessions=sessions,lines=summaries,expected_sessions=settings['expected_sessions'],
        completed_sessions=sum(r['complete'] for r in sessions),online_usage_cost_cny=usage_cost,
        online_guard_ledger_cny=ledger,online_accounted_cost_cny=accounted,
        offline_cost_cny=cost_total(root),cumulative_cost_cny=cost_total(root)+accounted,api_failures=failure_events,
        note='Each seed summary pools only ten paired tasks from one dataset/task line; seeds are repeated measurements, not independent images')
    write(online/'report.json',result)
    write(root/'metric.json',dict(stage='online',status=result['status'],completed_sessions=result['completed_sessions'],
        expected_sessions=settings['expected_sessions'],cost_cny=result['cumulative_cost_cny'],cumulative=True))
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUT); a=p.parse_args()
    import os
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    report(a.out)

if __name__=='__main__': main()
