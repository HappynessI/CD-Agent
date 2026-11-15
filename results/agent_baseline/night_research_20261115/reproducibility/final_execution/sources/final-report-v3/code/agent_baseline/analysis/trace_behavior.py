"""Read-only, API-free trace events. Recorded scores are diagnostic, never causal."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics

GROUPS = {'repeat_call':'redundancy', 'retry_unchanged':'redundancy',
          'blocked_calls':'redundancy', 'view_only_run':'excess_calls',
          'tool_error':'tool_failure', 'use_empty_result':'tool_failure',
          'answer_regression':'state_termination', 'submit_without_current':'state_termination',
          'fallback_latest_mismatch':'state_termination', 'abnormal_end':'state_termination',
          'empty_submission':'state_termination'}

def read(path):
    return json.loads(path.read_text())

def signature(call):
    return (call.get('tool'), json.dumps(call.get('args'),sort_keys=True,ensure_ascii=False))

def analyze(directory,final=None,steps=None):
    directory=Path(directory)
    final=read(directory/'final.json') if final is None else final
    steps=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines() if line.strip()] if steps is None else steps
    events=[]; calls=[]; previous=None; repeated=0; longest=0; views=0; max_views=0
    first_mask=None; latest=None; current=None; empty=set(); errors=Counter(); tools=Counter(); reactions=Counter(); answers=Counter()
    parents={}; creators={}; unresolved=False
    def emit(kind, step, **evidence):
        events.append(dict(kind=kind,category=GROUPS[kind],step=step,**evidence))
    for row in steps:
        step=row['step']; batch=row.get('calls',[])
        if row.get('note'): continue
        if first_mask is None and row.get('current_mask') is not None: first_mask=step
        if row.get('current_mask') is not None: current=row['current_mask']
        if batch and all(c['tool']=='view' for c in batch):
            views+=1; max_views=max(max_views,views)
            if views==3: emit('view_only_run',step,length_at_least=3)
        else: views=0
        raw=(row.get('planner') or {}).get('raw') or {}
        if isinstance(raw,dict):
            choices=raw.get('choices') or []
            native=(choices[0].get('message',{}).get('tool_calls') or []) if choices else []
            if native and not batch: emit('blocked_calls',step,count=len(native),attribution='harness_filter_candidate')
        for c in batch:
            idx=len(calls);calls.append(c); tools[c['tool']]+=1
            sig=signature(c)
            if previous:
                if sig==signature(previous):
                    repeated+=1;emit('repeat_call',step,tool=c['tool'],args=c.get('args'))
                else: repeated=1
                if not previous.get('ok',True):
                    reaction='same_args' if sig==signature(previous) else 'changed_args' if c['tool']==previous['tool'] else 'changed_tool'
                    reactions[reaction]+=1
                    if reaction=='same_args': emit('retry_unchanged',step,tool=c['tool'],error_code=previous.get('error_code'))
            else: repeated=1
            longest=max(longest,repeated);previous=c
            if not c.get('ok',True):
                code=c.get('error_code') or 'unknown'; errors[(c['tool'],code)]+=1
                emit('tool_error',step,tool=c['tool'],error_code=code,args=c.get('args'),
                     result=c.get('result_text'),attribution='backend_or_tool' if code.startswith('backend') else 'needs_context_review')
            try: public=json.loads(c.get('result_text') or '{}')
            except (ValueError,TypeError): public={}
            if not isinstance(public,dict): public={}
            handles=c.get('handles_created',c.get('output_handles',[]))
            for h in handles:
                parents[h]=c.get('input_handles',[]);creators[h]=idx
                if h.startswith('mask#'):
                    latest=h
                    if first_mask is None: first_mask=step
                if public.get('area')==0 or public.get('n')==0: empty.add(h)
            used=sorted(empty.intersection(c.get('input_handles',[])))
            if used and c['tool'] not in ('inspect','view','ask'):
                emit('use_empty_result',step,tool=c['tool'],handles=used,attribution='review_needed_empty_may_be_valid')
            metric=c.get('eval') or {}
            if c.get('current_before') != c.get('current_after') and (metric.get('delta_f1') or 0)<0:
                emit('answer_regression',step,tool=c['tool'],before=c.get('current_before'),after=c.get('current_after'),delta_f1=metric['delta_f1'],attribution='agent_edit_candidate_not_proven_bug')
            if c['tool']=='submit' and c.get('current_before') is None:
                emit('submit_without_current',step,attribution='valid_explicit_submit_possible')
            for answer in public.get('results',[]):
                if isinstance(answer,dict): answers[answer.get('answer','unknown')]+=1
    reason=final.get('termination_reason')
    if reason!='submit': emit('abnormal_end',steps[-1]['step'] if steps else 0,reason=reason)
    if final.get('change_area_px')==0: emit('empty_submission',steps[-1]['step'] if steps else 0)
    if final.get('submitted_by')!='model' and current and latest and current!=latest:
        emit('fallback_latest_mismatch',steps[-1]['step'],current=current,latest=latest,attribution='not_necessarily_bug_current_may_be_intentional')
    ancestry=set(); pending=[final.get('final_handle')]
    while pending:
        h=pending.pop()
        if not h or h in ancestry: continue
        ancestry.add(h);pending.extend(parents.get(h,[]))
    producers=set(creators.values()); used_producers={creators[h] for h in ancestry if h in creators}
    return dict(session=str(directory),task=final.get('task_id'),f1=(final.get('eval') or {}).get('f1'),
        rounds=final.get('planning_rounds'),termination=reason,first_mask_step=first_mask,
        longest_exact_call_run=longest,longest_view_only_run=max_views,
        structural_unused_producer_fraction=(len(producers-used_producers)/len(producers) if producers else None),
        tools=dict(tools),errors=[dict(tool=t,error=e,count=n) for (t,e),n in errors.items()],
        error_reactions=dict(reactions),verifier_answers=dict(answers),events=events,
        unavailable=['near_duplicate_semantics','verifier_spatial_override_and_net_gain','direction_class_violation_without_labels'])

def discover(root):
    for directory, dirs, files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in ('masks','images','insts','.git') and 'test' not in d.lower() and not d.startswith('code'))
        if 'steps.jsonl' in files and 'final.json' in files: yield Path(directory)

def summarize(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[(r['config'],r['task'])].append(r)
    result=[]
    for (config,task),items in sorted(grouped.items(),key=str):
        normal=[r['rounds'] for r in items if r['termination']=='submit' and r['rounds'] is not None]
        median=statistics.median(normal) if normal else None
        for r in items: r['round_ratio_to_normal_median']=r['rounds']/median if median and r['rounds'] is not None else None
        contrasts={}
        for category in sorted(set(GROUPS.values())):
            yes=[r['f1'] for r in items if r['f1'] is not None and any(e['category']==category for e in r['events'])]
            no=[r['f1'] for r in items if r['f1'] is not None and not any(e['category']==category for e in r['events'])]
            contrasts[category]=dict(flagged=len(yes),unflagged=len(no),mean_f1_difference=statistics.mean(yes)-statistics.mean(no) if yes and no else None,
                examples=[dict(session=r['session'],event=e) for r in items for e in r['events'] if e['category']==category][:3])
        result.append(dict(config=config,task=task,n=len(items),normal_round_median=median,contrasts=contrasts))
    return result

def main():
    import sys
    if '--extended' in sys.argv:
        from .trace_extended import main as extended_main
        return extended_main()
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',action='append',help='Exact configuration directory; repeatable.')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not args.resume:
        p.error('Output exists; use --resume or a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    journal=args.output/'sessions.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    done={r['session'] for r in rows};failures=[]
    # Online Recorder stores config / task / split / session. Do not recurse
    # into model caches, candidate archives or unrelated datasets in run root.
    configs=[args.root/c for c in args.config] if args.config else sorted(
        c for c in args.root.iterdir() if c.is_dir() and 'test' not in c.name.lower()
        and not c.name.startswith('code'))
    with journal.open('a') as sink:
        for config in configs:
            if 'test' in config.name.lower(): continue
            tasks=sorted(t for t in config.iterdir() if t.is_dir() and t.name.startswith('T'))
            for task in tasks:
                for directory in discover(task):
                    if str(directory) in done: continue
                    try:
                        row=analyze(directory);row['config']=config.name
                    except (OSError,ValueError,KeyError) as e:
                        failures.append(dict(session=str(directory),error=str(e)))
                        (args.output/'read_failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
                        continue
                    sink.write(json.dumps(row,ensure_ascii=False)+'\n');sink.flush()
                    rows.append(row);done.add(str(directory))
                    print(f"saved={len(rows)} session={directory}",flush=True)
    summary=summarize(rows)
    temp=args.output/'summary.tmp.json'
    temp.write_text(json.dumps(dict(groups=summary,read_failures=failures,
        excluded='test paths; non-task directories; code snapshots',causal=False),ensure_ascii=False,indent=2))
    temp.replace(args.output/'summary.json')
    print(f'sessions={len(rows)} groups={len(summary)} read_failures={len(failures)} output={args.output}',flush=True)

if __name__=='__main__': main()
