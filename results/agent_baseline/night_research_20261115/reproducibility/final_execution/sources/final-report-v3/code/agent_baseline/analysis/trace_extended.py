"""Incremental full-inventory behavioral evidence. Offline, never imported online."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import re
import statistics

from .coordinate_audit import coordinate_error,public,sessions,load_journal,write_json,empty_session
from .trace_behavior import analyze as base_analyze

VERSION=2
EDITORS={'to_mask','combine','morph','remove_region','add_region','refine_region','python','select'}
OBSERVERS={'view','inspect','ask'}


def normalize(value):
    if isinstance(value,str):return ' '.join(value.lower().split())
    if isinstance(value,dict):return {k:normalize(v) for k,v in sorted(value.items())}
    if isinstance(value,list):return [normalize(x) for x in value]
    return value


def semantic_signature(call):
    args=normalize(call.get('args') or {})
    # Only genuinely equivalent spellings/selection order, never claim arbitrary
    # natural-language queries or different coordinates mean the same thing.
    if isinstance(args,dict) and 'ids' in args:args['ids']=sorted(args['ids']) if isinstance(args['ids'],list) else args['ids']
    return call.get('tool'),json.dumps(args,sort_keys=True,ensure_ascii=False)


def report_answers(text):
    handle=re.search(r'Verifier answers on (mask#\d+):',text or '')
    if not handle:return []
    result=[]
    for answer in ('yes','no'):
        ids=re.search(r'IDs answered '+answer+r': (\[[\d, ]*\])',text)
        if ids:result.extend(dict(mask=handle[1],id=i,answer=answer) for i in json.loads(ids[1]))
    return result


def behavioral_events(meta,final,steps):
    events=[];calls=[];seen={};last_failed={};last_view=-1;pending_answers={};mask_scores={};mask_step={}
    raw_repeats=0;last_raw=None;last_current=None;switches=0;current_switches=[];shape=meta.get('image_shape')
    def emit(kind,category,step,**detail):events.append(dict(kind=kind,category=category,step=step,**detail))
    for row in steps:
        step=row['step']
        if row.get('note'):continue
        raw=(row.get('planner') or {}).get('raw') or {}
        choices=raw.get('choices',[]) if isinstance(raw,dict) else []
        requested=(choices[0].get('message',{}).get('tool_calls') or []) if choices else []
        raw_sig=json.dumps([(c.get('function') or {}) for c in requested],sort_keys=True)
        if requested and not row.get('calls'):
            raw_repeats+=1
            emit('harness_filtered_request','redundancy',step,requested_tools=[(c.get('function') or {}).get('name') for c in requested],
                 same_as_previous=raw_sig==last_raw,attribution='harness_policy_and_agent_repetition',evidence='planner.raw contains tool calls; executed calls empty')
        if requested:last_raw=raw_sig
        if not requested and row.get('planner') and not row.get('calls'):
            emit('no_tool_reply','state_termination',step,attribution='agent_protocol_or_provider_output')
        for within,c in enumerate(row.get('calls',[]),1):
            index=len(calls);calls.append((step,c));name=c['tool'];args=c.get('args') or {};sig=semantic_signature(c);pub=public(c)
            error=coordinate_error(c,shape)
            if error:emit('coordinate_out_of_bounds','tool_failure',step,call=within,tool=name,args=args,**error,attribution='agent_coordinate_error_with_interface_ambiguity')
            previous=seen.get(sig)
            # Stateful ask/submit/plan and creation of a new current answer can be
            # intentional; repeats are candidates, not proof of wasted work.
            if previous is not None and name not in ('ask','submit','plan','set_current'):
                emit('equivalent_repeated_call','redundancy',step,previous_step=calls[previous][0],call=within,tool=name,args=args,
                     attribution='agent_redundancy_candidate',cached=c.get('cache_hit'))
            seen[sig]=index
            signatures=[semantic_signature(x[1]) for x in calls]
            for period in (2,3,4):
                if len(signatures)>=2*period and signatures[-period:]==signatures[-2*period:-period]:
                    emit('call_cycle','redundancy',step,period=period,start_step=calls[-2*period][0],tools=[x[0] for x in signatures[-period:]],attribution='agent_loop_candidate');break
            if not c.get('ok',True):
                previous_failure=last_failed.get(name)
                if previous_failure:
                    emit('retry_after_failure','tool_failure',step,tool=name,previous_step=previous_failure[0],
                         unchanged=semantic_signature(previous_failure[1])==sig,error_code=c.get('error_code'),attribution='agent_retry_strategy')
                last_failed[name]=(step,c)
                if c.get('error_code','').startswith('backend'):
                    traceback=(c.get('provenance') or {}).get('traceback','')
                    emit('backend_failure','tool_failure',step,tool=name,error_code=c.get('error_code'),
                         traceback_excerpt=traceback[-1200:],attribution='harness_or_backend_failure_requires_traceback')
            elif name in last_failed:last_failed.pop(name)
            if name=='view' and c.get('ok'):last_view=index
            for a in pub.get('results',[])+report_answers(pub.get('report','')):
                if isinstance(a,dict) and a.get('answer') in ('yes','no') and a.get('mask'):
                    pending_answers[(a['mask'],a.get('id'))]=(a['answer'],index,step)
            if name=='remove_region' and c.get('ok'):
                for cid in args.get('ids',[]):
                    answer=pending_answers.get((args.get('mask'),cid))
                    if answer:
                        emit('verifier_follow_or_override','tool_failure',step,mask=args.get('mask'),id=cid,answer=answer[0],
                             answer_step=answer[2],action='remove',view_after_answer=last_view>answer[1],
                             attribution='agent_verifier_reliance_observation_not_causal')
            before,after=c.get('current_before'),c.get('current_after')
            if before!=after and after:
                switches+=1;current_switches.append((step,before,after))
                metric=c.get('eval') or {}
                if metric.get('f1') is not None:mask_scores[after]=metric['f1'];mask_step[after]=step
            for candidate in c.get('candidate_eval',[]):
                if candidate.get('f1') is not None:mask_scores[candidate['handle']]=candidate['f1'];mask_step[candidate['handle']]=step
            if name=='submit' and c.get('ok') and pub.get('submitted') is False:
                emit('submit_report_not_rejection','state_termination',step,attribution='expected_harness_verification_stage')
        if row.get('current_mask') and (row.get('eval') or {}).get('f1') is not None:
            mask_scores[row['current_mask']]=row['eval']['f1'];mask_step[row['current_mask']]=step
        last_current=row.get('current_mask',last_current)
    final_score=(final.get('eval') or {}).get('f1');best=max(mask_scores,key=mask_scores.get) if mask_scores else None
    if best and final_score is not None and mask_scores[best]>final_score+1e-8:
        emit('better_candidate_abandoned','state_termination',steps[-1]['step'] if steps else 0,
             best_handle=best,best_step=mask_step[best],best_f1=mask_scores[best],final_handle=final.get('final_handle'),final_f1=final_score,
             loss=mask_scores[best]-final_score,attribution='offline_diagnostic_agent_cannot_observe_gt')
    if len(current_switches)>=2:
        for prior,now in zip(current_switches,current_switches[1:]):
            if now[2]==prior[1]:emit('answer_handle_cycle','state_termination',now[0],before=now[1],after=now[2],previous_step=prior[0],attribution='agent_state_cycle_candidate')
    return dict(events=events,harness_filtered_rounds=raw_repeats,current_switches=switches,
                candidate_best_f1=mask_scores.get(best),candidate_best_step=mask_step.get(best),
                final_vs_best_candidate=(final_score-mask_scores[best]) if final_score is not None and best else None,
                tools_seconds=sum(c.get('seconds') or 0 for _,c in calls),planner_seconds=sum((r.get('planner') or {}).get('seconds') or 0 for r in steps),
                calls=len(calls),no_submit=final.get('termination_reason')=='no_submit')


def spatial_verifier(directory,meta,final,steps):
    """Compare exact verified outlines to final pixels; no assumptions from handle IDs.

    Count following/overriding the final yes/no verdict and the exact F1 change of
    the observed final decision versus following that verdict on those pixels.
    This is a pixel counterfactual, not an estimated causal effect of the verifier.
    """
    import numpy as np
    from PIL import Image
    from .error_taxonomy import Archive
    votes=[]
    for row in steps:
        votes.extend(dict(a,step=row['step']) for a in report_answers(row.get('report','')))
        for c in row.get('calls',[]):
            p=public(c)
            votes.extend(dict(a,step=row['step']) for a in p.get('results',[])+report_answers(p.get('report','')) if isinstance(a,dict) and a.get('answer') in ('yes','no') and a.get('mask'))
    if not votes:return dict(verified_outlines=0,events=[],available=True)
    def mask(handle):
        with Image.open(directory/'masks'/(handle.replace('#','_')+'.png')) as im:return np.asarray(im)>0
    truth_path=meta.get('gt_mask_path')
    if not truth_path:
        sources=meta.get('source_images') or []
        if sources and 'LEVIR' in sources[0]:truth_path=str(Path(sources[0]).parent.parent/'label'/Path(sources[0]).name)
    if not truth_path:return dict(available=False,reason='No archived ground-truth path for offline attribution',events=[])
    with Image.open(truth_path) as im:gt=np.asarray(im)>0
    pred=mask(final['final_handle']);regions={};last_handle=None;labels=None
    archive=Archive(directory,gt.shape,steps)
    for vote in votes:
        h=vote['mask']
        if h!=last_handle:labels=archive.component_labels(h);last_handle=h
        ids=np.flatnonzero(labels==vote.get('id'));key=ids.tobytes()
        if len(ids):regions[key]=(ids,vote)
    observed=dict(yes_keep=0,yes_remove=0,no_keep=0,no_remove=0,partial=0)
    alternative=pred.copy();events=[];claimed=np.zeros(pred.size,bool)
    # Later verdicts supersede earlier verdicts on overlapping changed outlines.
    for ids,vote in sorted(regions.values(),key=lambda item:item[1]['step'],reverse=True):
        remaining=ids[~claimed[ids]];claimed[ids]=True
        if not len(remaining):continue
        kept=int(pred.ravel()[ids].sum());answer=vote['answer'];action='keep' if kept==len(ids) else 'remove' if kept==0 else 'partial'
        observed['partial' if action=='partial' else answer+'_'+action]+=1
        tp=int(gt.ravel()[ids].sum());fp=len(ids)-tp
        if action=='partial' or (answer=='no' and action=='keep') or (answer=='yes' and action=='remove'):
            events.append(dict(kind='verifier_spatial_override',category='tool_failure',step=vote['step'],mask=vote['mask'],id=vote['id'],
                answer=answer,final_action=action,area=len(ids),retained_px=kept,tp=tp,fp=fp,
                attribution='agent_override_of_verifier' if action!='partial' else 'agent_partial_edit'))
        if answer=='no' and action=='remove' and tp>fp:
            events.append(dict(kind='verifier_followed_harmful_removal',category='tool_failure',step=vote['step'],mask=vote['mask'],id=vote['id'],tp=tp,fp=fp,
                attribution='verifier_error_and_agent_reliance_not_proven_blindness'))
        alternative.ravel()[remaining]=(answer=='yes')
    def f1(a):
        tp=int((a & gt).sum());fp=int((a & ~gt).sum());fn=int((~a & gt).sum());den=2*tp+fp+fn
        return 2*tp/den if den else 1.
    return dict(available=True,verified_outlines=len(regions),decisions=observed,events=events,
        final_f1=f1(pred),follow_verifier_f1=f1(alternative),override_pixel_counterfactual_delta=f1(pred)-f1(alternative),
        caveat='Offline pixel counterfactual. Overlapping regions use latest verdict; not causal verifier contribution.')


def analyze(item,base=None,spatial=False):
    config,task,split,directory=item
    meta=json.loads((directory/'input.json').read_text())
    final_path=directory/'final.json';complete=final_path.exists()
    final=json.loads(final_path.read_text()) if complete else dict(task_id=task,termination_reason='missing_final')
    steps_path=directory/'steps.jsonl'
    steps=[json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()] if steps_path.exists() else []
    result=dict(base or base_analyze(directory,final,steps));result['events']=list(result['events'])
    extra=behavioral_events(meta,final,steps);result['events'].extend(extra.pop('events'));result.update(extra)
    sources=' '.join(meta.get('source_images') or [])
    result.update(version=VERSION,config=config,task=task,split=split,complete=complete,harness=meta.get('backends',{}),
        dataset=next((d for d in ('SECOND','LEVIR','WHU','xBD') if d.lower() in sources.lower()),'unknown'),
        evidence=dict(steps=str(steps_path),input=str(directory/'input.json'),final=str(final_path)),cost_cny=final.get('cost_cny'))
    if spatial and complete:
        try:
            detail=spatial_verifier(directory,meta,final,steps);result['events'].extend(detail.pop('events'));result['verifier_spatial']=detail
        except (OSError,ValueError,KeyError) as error:result['verifier_spatial']=dict(available=False,reason=str(error))
    result['unavailable']=['arbitrary_language_semantic_equivalence','causal_attribution_from_observation_alone']
    return result


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['dataset'],row['task'],row['config'],row['split'])].append(row)
    output=[]
    for key,items in sorted(groups.items()):
        normal=[r['calls'] for r in items if r['termination']=='submit'];median=statistics.median(normal) if normal else None
        counts=Counter();affected=Counter();attributions=Counter();examples=defaultdict(list)
        for row in items:
            if median and row['calls']>2*median:
                row['events'].append(dict(kind='excess_calls_relative_to_group',category='excess_calls',step=None,calls=row['calls'],group_normal_median=median,attribution='descriptive_outlier_not_proof_of_waste'))
            kinds=set()
            for event in row['events']:
                kind=event['kind'];counts[kind]+=1;kinds.add(kind)
                attributions[event.get('attribution','unclassified_observation')]+=1
                if len(examples[kind])<3:examples[kind].append(dict(session=row['session'],evidence=row['evidence'],event=event))
            affected.update(kinds)
        available=[r['verifier_spatial'] for r in items if r.get('verifier_spatial',{}).get('available')]
        delta=[r['override_pixel_counterfactual_delta'] for r in available if 'override_pixel_counterfactual_delta' in r]
        output.append(dict(zip(('dataset','task','config','split'),key),sessions=len(items),event_counts=dict(counts),affected_sessions=dict(affected),
            examples=dict(examples),attributions=dict(attributions),normal_calls_median=median,tool_calls=sum(r['calls'] for r in items),
            no_submit=sum(r['no_submit'] for r in items),incomplete=sum(not r['complete'] for r in items),
            spatial_available_sessions=len(available),spatial_unavailable_sessions=len(items)-len(available),
            mean_override_pixel_counterfactual_delta=statistics.mean(delta) if delta else None))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--extended',action='store_true');p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--base-journal',type=Path);p.add_argument('--include-test-logs',action='store_true');p.add_argument('--spatial',action='store_true');p.add_argument('--config',action='append')
    p.add_argument('--resume-static-history',action='store_true',help='Reuse all recorded historical sessions, including terminated sessions without final.json.')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);journal=args.output/'sessions.jsonl'
    previous={r['session']:r for r in load_journal(journal)}
    rows=list(previous.values());done={r['session'] for r in rows if r['complete'] or args.resume_static_history};base={}
    if args.base_journal:
        # The old scan may still be appending. Read a snapshot without modifying its journal.
        with args.base_journal.open() as f:
            for line in f:
                try:r=json.loads(line);base[r['session']]=r
                except ValueError:break
    failures=[];excluded=[]
    def work(item):
        try:
            if empty_session(item[3]):return None,dict(session=str(item[3]),excluded='initialization_only_no_metadata_or_steps')
            return analyze(item,base.get(str(item[3])),args.spatial),None
        except (OSError,ValueError,KeyError,TypeError) as error:return None,dict(session=str(item[3]),error=str(error))
    pending=(item for item in sessions(args.root,args.config,args.include_test_logs) if str(item[3]) not in done)
    with journal.open('a') as sink,ThreadPoolExecutor(max_workers=args.workers) as pool:
        while batch:=list(itertools.islice(pending,args.workers*4)):
            for row,error in pool.map(work,batch):
                if error and error.get('excluded'):excluded.append(error);continue
                if error:failures.append(error);write_json(args.output/'read_failures.json',failures);continue
                sink.write(json.dumps(row,ensure_ascii=False)+'\n');sink.flush();rows.append(row)
                if len(rows)%100==0:print('saved='+str(len(rows)),flush=True)
    rows=list({r['session']:r for r in rows}.values())
    write_json(args.output/'read_failures.json',failures)
    write_json(args.output/'empty_sessions.json',excluded)
    groups=summarize(rows)
    write_json(args.output/'summary.json',dict(version=VERSION,sessions=len(rows),groups=groups,read_failures=failures,
        scope='Existing logs only; development/report/test are separate; no new test inference',causal=False))
    write_json(args.output/'metric.json',dict(sessions=len(rows),groups=len(groups),read_failures=len(failures),empty_sessions=len(excluded),cost_cny=0))
    print(f'sessions={len(rows)} groups={len(groups)} read_failures={len(failures)}',flush=True)
    if failures:raise SystemExit('Incomplete reads retained for repair; not full coverage yet')


if __name__=='__main__':main()
