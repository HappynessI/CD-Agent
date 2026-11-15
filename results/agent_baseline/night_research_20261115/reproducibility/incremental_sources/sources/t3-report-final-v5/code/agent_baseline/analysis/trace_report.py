"""Consolidate completed journals without rescanning images or rerunning agents."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
from .coordinate_audit import load_journal,write_json
from .trace_extended import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--coordinates',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=list({r['session']:r for r in load_journal(args.checkpoint/'sessions.jsonl')}.values())
    audited={r['session']:r for r in load_journal(args.coordinates/'sessions.jsonl')} if args.coordinates else {}
    totals=Counter();category_events=Counter();category_sessions=Counter();examples=defaultdict(list);gaps=Counter();gap_examples={}
    for row in rows:
        # Upgrade only coordinate classification from the audited journal. This
        # includes nested exemplar.point without repeating expensive spatial work.
        if row['session'] in audited:
            row['events']=[e for e in row['events'] if e['kind']!='coordinate_out_of_bounds']
            row['events'].extend(dict(e,kind='coordinate_out_of_bounds',category='tool_failure',attribution='agent_coordinate_error_with_interface_ambiguity') for e in audited[row['session']]['events'])
        # A successful verification report is expected behavior, not termination failure.
        row['normal_verification_reports']=sum(e['kind']=='submit_report_not_rejection' for e in row['events'])
        row['events']=[e for e in row['events'] if e['kind']!='submit_report_not_rejection']
        detail=row.get('verifier_spatial',{})
        if detail.get('available'):
            detail['superseded_outlines']=detail.get('verified_outlines',0)-sum(detail.get('decisions',{}).values())
        else:
            reason=detail.get('reason','missing final.json; final-mask comparison unavailable' if not row.get('complete') else 'spatial analysis not requested');gaps[reason]+=1;gap_examples.setdefault(reason,row['session'])
    groups=summarize(rows)
    for row in rows:
        categories=set()
        for event in row['events']:
            kind=event['kind'];totals[kind]+=1;category_events[event['category']]+=1;categories.add(event['category'])
            if len(examples[kind])<5:examples[kind].append(dict(session=row['session'],evidence=row['evidence'],event=event))
        category_sessions.update(categories)
    failures=json.loads((args.checkpoint/'summary.json').read_text()).get('read_failures',[])
    value=dict(sessions=len(rows),groups=groups,event_counts=dict(totals),category_event_counts=dict(category_events),
        category_affected_sessions=dict(category_sessions),examples=dict(examples),read_failures=failures,
        spatial_gaps=[dict(reason=r,count=n,example=gap_examples[r]) for r,n in gaps.items()],
        definitions=dict(counts='Signals can overlap within a call; affected sessions are deduplicated. No signal is automatically a confirmed bug.',
            normal_reports='Expected pre-submission verification reports are excluded from anomaly counts.',
            superseded_outlines='Exact earlier outlines fully covered by later verdicts are excluded from final-decision counts.',
            blind_trust='Following a verifier without a subsequent view is an observation, not proof of blind trust.',
            overrides='Latest-overlap pixel counterfactual relative to obeying verifier; not causal contribution.'),
        journal=str(args.checkpoint/'sessions.jsonl'),coordinate_journal=str(args.coordinates/'sessions.jsonl') if args.coordinates else None,
        coordinate_classification='strict audit overlay' if args.coordinates else 'provisional trace classification; strict audit pending')
    write_json(args.output/'behavior_evidence.json',value)
    with (args.output/'task_config_counts.csv').open('w',newline='') as sink:
        writer=csv.writer(sink);writer.writerow(['dataset','task','config','split','sessions','tool_calls','no_submit','incomplete','spatial_unavailable'])
        for g in groups:writer.writerow([g[k] for k in ('dataset','task','config','split','sessions','tool_calls','no_submit','incomplete','spatial_unavailable_sessions')])
    write_json(args.output/'report_metric.json',dict(sessions=len(rows),read_failures=len(failures),spatial_unavailable=sum(gaps.values()),cost_cny=0))
    print(json.dumps(dict(sessions=len(rows),event_counts=dict(totals),category_affected_sessions=dict(category_sessions),read_failures=len(failures),spatial_unavailable=sum(gaps.values()))))
    if failures:raise SystemExit('Read failures still require review')


if __name__=='__main__':main()
