"""Bounded, resumable offline verifier; never called by estimate/inventory.

Use the frozen f37_corr model, crops, prompts, seed and scope of second_rule.
All HTTP 429/403/402 and balance errors trip STOP before any retry/new request.
Only already-in-flight calls may finish; their answers and cost are retained.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
from pathlib import Path

import numpy as np

from .io_retry import write_json, read_image, retry_io
from .missed_sources import (exists, load_json, ensure_settings, candidate_records,
    answer_map, reference, ledger_unit, score_answers, S1)


class AskStopped(RuntimeError):
    pass


class Budget:
    """One process per pool; reserve a conservative request allowance in advance.

    The observed ledger average is used only for estimates. It is not a safe
    per-request bound. The explicit allowance prevents launching a batch whose
    reservations would exceed the remaining cap. Actual charges are persisted.
    """
    def __init__(self,directory,max_cost,allowance,clear_stop=False):
        self.directory=Path(directory)
        self.stop_file=self.directory/'STOP.json'
        self.state_file=self.directory/'budget.json'
        self.lock=threading.Lock()
        self.halted=threading.Event()
        self.max_cost,self.allowance=float(max_cost),float(allowance)
        if self.max_cost<=0 or self.allowance<=0:
            raise ValueError('Positive cost cap and request allowance required')
        self.charges=load_json(self.state_file).get('charges',{}) if exists(self.state_file) else {}
        self.reserved={}
        if clear_stop and exists(self.stop_file):
            # Explicit flag acknowledges provider recovery/budget increase; no silent clearing.
            old=load_json(self.stop_file)
            write_json(self.directory/'previous_STOP.json',old)
            retry_io(lambda:self.stop_file.unlink())
        self.check()

    @property
    def spent(self):
        return float(sum(self.charges.values()))

    def check(self):
        if self.halted.is_set() or exists(self.stop_file):
            raise AskStopped('STOP exists; resolve cause and use --clear-stop --resume')

    def stop(self,reason):
        self.halted.set()
        write_json(self.stop_file,dict(reason=reason,spent_cny=self.spent,max_cost_cny=self.max_cost))

    def trip(self,reason,batch_only=False):
        with self.lock:
            self.stop(reason)
        raise AskStopped(reason)

    def reserve(self,uid):
        with self.lock:
            self.check()
            if self.spent+sum(self.reserved.values())+self.allowance > self.max_cost+1e-12:
                return False
            self.reserved[uid]=self.allowance
        return True

    def charge(self,uid,cost):
        with self.lock:
            self.charges[uid]=self.charges.get(uid,0.)+float(cost)
            write_json(self.state_file,dict(charges=self.charges,spent_cny=self.spent))
            if cost > self.allowance+1e-12:
                self.stop('Observed request cost exceeds reserved allowance; increase --request-cost-cny')
            elif self.spent > self.max_cost+1e-12:
                self.stop('Actual cumulative cost exceeded budget')

    def release(self,uid):
        with self.lock:
            self.reserved.pop(uid,None)


class RequestGuard:
    def __init__(self,budget,uid):
        self.budget,self.uid=budget,uid
    def check(self):
        self.budget.check()
    def charge(self,cost):
        self.budget.charge(self.uid,cost)
    def trip(self,reason,batch_only=False):
        self.budget.trip(reason,batch_only)


def run_requests(records,backend,directory,max_cost,concurrency=1,allowance=.02,
                 resume=False,clear_stop=False,settings=None):
    """Testable fake-backend entry; backend(record,guard) -> answer dictionary."""
    if concurrency<1:
        raise ValueError('concurrency must be positive')
    directory=Path(directory)
    answers_file=directory/'answers.json'
    if exists(answers_file) and not resume:
        raise ValueError('Answers already exist; use --resume')
    if settings is not None:
        ensure_settings(directory/'settings.json',settings)
    done=answer_map(answers_file)
    budget=Budget(directory,max_cost,allowance,clear_stop)
    # Restore answers from per-request snapshots after a crash between atomic writes.
    files=retry_io(lambda:list((directory/'answer_parts').glob('*.json')))
    for path in files:
        result=load_json(path)
        done[result['uid']]=result
    if done:
        write_json(answers_file,done)
    # A single staging pool is owned by one job, not by multiple shards.
    # Per-image candidate generation is sharded; verifier parallelism is threads.
    failure=None
    def one(record):
        uid=record['uid'];guard=RequestGuard(budget,uid)
        try:
            guard.check()
            result=dict(uid=uid,**backend(record,guard))
            # API backend charges via Planner. Fakes may use guard.charge too.
            safe_uid=uid.replace(':','__')
            write_json(directory/'answer_parts'/(safe_uid+'.json'),result)
            return result
        except Exception as exc:
            # Never archive arbitrary exception text (may contain transport details).
            if not exists(budget.stop_file):
                code=getattr(exc,'code',getattr(exc,'status_code',None))
                reason=f'HTTP {code}' if code in (429,403,402) else 'Verifier failure; inspect local log; resume explicitly'
                with budget.lock:
                    budget.stop(reason)
            raise
        finally:
            budget.release(uid)
    iterator=iter(r for r in records if r['uid'] not in done)
    exhausted=False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while not exhausted:
            budget.check()
            batch=[]
            for _ in range(concurrency):
                try:
                    record=next(iterator)
                except StopIteration:
                    exhausted=True;break
                if not budget.reserve(record['uid']):
                    # Finish existing calls first, then retry this record in the next batch.
                    iterator=iter_pending(record,iterator)
                    break
                batch.append(record)
            if not batch:
                if exhausted:
                    break
                budget.trip('Remaining budget cannot reserve one request; raise cap to resume')
            futures=[pool.submit(one,r) for r in batch]
            for future in as_completed(futures):
                try:
                    result=future.result()
                    done[result['uid']]=result
                    write_json(answers_file,done)
                except Exception as exc:
                    failure=failure or exc
            if failure is not None:
                raise AskStopped('Verifier stopped; successful in-flight answers retained') from failure
            budget.check()
    # Empty pools are valid complete pools with zero requests.
    write_json(answers_file,done)
    result=dict(answers=len(done),cost_cny=budget.spent,status='complete')
    write_json(directory/'summary.json',result)
    return result


def iter_pending(first,rest):
    yield first
    yield from rest


def existing_verifier(record,guard):
    from agent_baseline.ask_eval import api_answer, VARIANTS
    from agent_baseline.state import Session
    from agent_baseline.view import ask_images
    row=record['row']
    if 'mask' in record:
        mask=record['mask']
    else:
        mask=np.zeros(record['shape'],bool)
        mask.ravel()[record['idx']]=True
    rgb=[read_image(row['image_'+t],rgb=True) for t in ('t1','t2')]
    session=retry_io(lambda:Session(mask.shape,target='building',
        change_direction=row['change_direction'],scope=row['scope']))
    images=ask_images(rgb,mask,record['bbox'],True)
    # Same seed formula as second_rule, independent of threads and resume order.
    seed=sum(map(ord,record['uid'])) % (2**31)
    return api_answer(images,dict(VARIANTS['f37_corr'],strict_stop=True),guard,seed,session)


def run_ask_cli(args,run,work,spec,report):
    if args.shards!=1 or args.shard!=0:
        raise ValueError('Do not shard one ask pool; use --concurrency')
    if args.whu_fixed:
        if spec['dataset']!='whu':
            raise ValueError('--whu-fixed requires a WHU development group')
        sources=['fixed'];directory=work/'whu_rule'/spec['group']
        estimates=report.get('whu_fixed_cost_estimates',{}).get(spec['group'],{}).get('pools',{})
    else:
        if not args.pool:
            raise ValueError('ask requires --pool from the saved estimate')
        sources=args.pool.split('+');directory=work/'ask'/spec['group']/args.pool
        group=report['groups'][spec['group']]
        allowed=set(group.get('selected_for_ask',[]))
        if not set(sources)<=allowed or len(sources)>2 or (len(sources)==2 and sources[0]!=S1):
            raise ValueError('Pool must be S1, selected marginal source, or S1+selected source')
        estimates=report.get('cost_estimates',{}).get(spec['group'],{}).get('pools',{})
    pool='+'.join(sources)
    if pool not in estimates or estimates[pool].get('cost_cny') is None:
        raise ValueError('Run estimate and review candidate count/cost before invoking ask')
    records=list(candidate_records(run,work,spec,sources,combined=len(sources)>1,fixed=args.whu_fixed))
    if len(records)!=estimates[pool]['candidates']:
        raise ValueError('Candidate count changed after estimate; regenerate estimate')
    # Explicit readable identities, no labels or GT path enter any verifier record.
    settings=dict(group=spec['group'],pool=pool,scope=spec['scope'],variant='f37_corr',
        direction=spec['direction'],candidate_count=len(records),
        candidates=[dict(uid=r['uid'],bbox=r['bbox'],pixels=len(r['idx'])) for r in records])
    if not args.whu_fixed:
        settings['thresholds']=report['groups'][spec['group']]['thresholds']
    result=run_requests(records,existing_verifier,directory,args.max_cost_cny,
        args.concurrency,args.request_cost_cny,args.resume,args.clear_stop,settings)
    if args.whu_fixed:
        result['rule']=score_answers(run,work,spec,sources,fixed=True)
    write_json(directory/'summary.json',result)
    return result
