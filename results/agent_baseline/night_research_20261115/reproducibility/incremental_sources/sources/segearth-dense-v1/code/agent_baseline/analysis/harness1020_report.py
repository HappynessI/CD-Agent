"""Offline incremental development-arm report. Never imported by the online agent."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from .gate6_pair_report import retries


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def aggregate(rows):
    counts=Counter()
    for row in rows:counts.update({k:row['final']['eval'][k] for k in ('tp','fp','fn')})
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    calls=sum(r['final']['tool_calls'] for r in rows)
    errors=Counter()
    for row in rows:errors.update(row['error_codes'])
    return dict(n=len(rows),mean_f1=100*statistics.mean(r['final']['eval']['f1'] for r in rows) if rows else None,
        pooled_f1=200*tp/(2*tp+fp+fn) if tp*2+fp+fn else (100 if rows else None),
        precision=100*tp/(tp+fp) if tp+fp else None,recall=100*tp/(tp+fn) if tp+fn else None,
        counts=dict(counts),calls=calls,rounds=sum(r['final']['planning_rounds'] for r in rows),
        tokens={k:sum(r['final'].get('tokens',{}).get(k,0) for r in rows) for k in ('input','cached','output')},
        errors=dict(errors),error_rate=sum(errors.values())/calls if calls else 0,
        invalid_retries=sum(len(r['invalid_retries']) for r in rows),
        termination=dict(Counter(r['final']['termination_reason'] for r in rows)),
        abnormal=sum(r['final']['termination_reason']!='submit' for r in rows),
        seconds=sum(r['final'].get('elapsed_seconds',0) for r in rows),
        token_metered_cost_cny=sum(r['final']['cost_cny'] for r in rows),
        set_answer_calls=sum(r['tools'].get('set_answer',0) for r in rows),
        pair_tool_calls=sum(r['tools'].get('segment_text_pair',0) for r in rows),
        segmentation_backend_invocations=sum(r.get('segmentation_backend_invocations',0) for r in rows))


def collect(run,selection,output,config,task=None):
    selected=[r for r in json.loads(selection.read_text()) if r['pilot'] and (task is None or r['task']==task)]
    rows=[];missing=[]
    for case in selected:
        name=Path(case['manifest']['id']).stem
        directory=run/config/case['task']/'val'/name
        saved=output/'sessions'/config/case['task']/(name+'.json')
        try:
            if saved.exists():row=json.loads(saved.read_text())
            else:
                final=json.loads((directory/'final.json').read_text());meta=json.loads((directory/'input.json').read_text())
                steps=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines() if line.strip()]
                calls=[c for step in steps for c in step.get('calls',[])]
                errors=[dict(step=step['step'],tool=c['tool'],args=c.get('args'),code=c.get('error_code'),message=c.get('result_text'))
                        for step in steps for c in step.get('calls',[]) if not c.get('ok',True)]
                row=dict(config=config,task=case['task'],name=name,cohort=case['cohort'],session=str(directory),
                    final=final,error_codes=dict(Counter(e['code'] or 'unknown' for e in errors)),errors=errors,
                    tools=dict(Counter(c['tool'] for c in calls)),invalid_retries=retries(directory),
                    segmentation_backend_invocations=sum(len(c.get('provenance',{}).get('temporal_backends',{})) if c['tool']=='segment_text_pair' else int(c['tool']=='segment_text') for c in calls),
                    seed=meta.get('seed'),model_parameters=meta.get('model_parameters'),backends=meta.get('backends'),
                    gt_mask_path=meta.get('gt_mask_path'),
                    f1_drop_calls=[dict(step=st['step'],tool=c['tool'],args=c.get('args'),before=c.get('current_before'),after=c.get('current_after'),eval=c['eval']) for st in steps for c in st.get('calls',[]) if (c.get('eval') or {}).get('delta_f1') is not None and c['eval']['delta_f1'] < -1e-9],
                    successful_after_error=[dict(error_step=e['step'],step=st['step'],tool=c['tool'],args=c.get('args')) for e in errors for st in steps if e['step'] < st['step'] <= e['step']+4 for c in st.get('calls',[]) if c.get('ok') and c['tool'] in ('ask','inspect','set_answer')],
                    state_recovery=[dict(step=s['step'],tool=c['tool'],args=c.get('args'),ok=c.get('ok'),current=s.get('current_mask')) for s in steps for c in s.get('calls',[]) if c['tool']=='set_answer'])
                if 'eval' not in final:raise ValueError('final has no evaluation')
                write(saved,row)
            rows.append(row)
        except (OSError,ValueError,KeyError) as e:missing.append(dict(session=str(directory),error=str(e)))
    summary=dict(config=config,groups={k:aggregate([r for r in rows if k=='all' or r['cohort']==k]) for k in ['all','hard','simple']},
                 rows=rows,missing=missing,cost_note='Metered real API tokens valued with repository price table, not provider invoice.',expected=len(selected))
    write(output/'arms'/f'{config}.json',summary)
    arms={p.stem:json.loads(p.read_text()) for p in (output/'arms').glob('*.json')}
    ledger=sum(float(p.read_text()) for p in (run/'budget'/'harness1020').glob('*.txt'))
    recorded=sum(a['groups']['all']['token_metered_cost_cny'] for a in arms.values())
    write(output/'metric.json',dict(stage=config,completed=len(rows),expected=len(selected),missing=len(missing),
        cost_cny=max(ledger,recorded),cumulative=True,groups=summary['groups']))
    pairs=[]
    for arm in sorted(set(arms)-{'baseline'}):
        for reference in ['baseline']+{'p1':['p0'],'p2':['p1'],'p0v2':['p0'],'p1v2':['p0v2'],'p2v2':['p1v2']}.get(arm,[]):
            lookup={(r['task'],r['name']):r for r in arms[arm]['rows']}
            for old in arms.get(reference,{}).get('rows',[]):
                new=lookup.get((old['task'],old['name']))
                if new:
                    match=all(old.get(k)==new.get(k) for k in ['seed','model_parameters','gt_mask_path'])
                    pairs.append(dict(reference=reference,arm=arm,task=old['task'],name=old['name'],cohort=old['cohort'],paired_inputs_match=match,
                        old_f1=100*old['final']['eval']['f1'],new_f1=100*new['final']['eval']['f1'],
                        delta_f1=100*(new['final']['eval']['f1']-old['final']['eval']['f1']),
                        old_session=old['session'],new_session=new['session']))
    write(output/'comparison.json',dict(arms={k:v['groups'] for k,v in arms.items()},pairs=pairs,
        caveat='Frozen purposive development cases, one seed. Staged pilots can contain fewer cases: compare matched rows and their cohort counts, never all-six versus three aggregate. F1 changes are observations, not causal proof or test results.'))
    print(json.dumps(dict(stage=config,groups=summary['groups'],missing=len(missing)),ensure_ascii=False))
    if missing:raise SystemExit('Incomplete arm; successful sessions and metered spending preserved')


def main():
    p=argparse.ArgumentParser()
    for key in ['run','selection','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--task',help='Restrict reporting to a predeclared frozen task subset.')
    a=p.parse_args();collect(a.run,a.selection,a.output,a.config,a.task)

if __name__=='__main__':main()
