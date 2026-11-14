#!/usr/bin/env python3
"""Event-driven experiment queues. Public workflow: init / add / status.
advance is a Slurm hook; tick repairs hooks only; finalize is host-side I/O.
Technical documentation lives in doc/agent_baseline/README_ai.md.
"""
import fcntl
import glob
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager

TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
            'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
QQ='/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq'
WGIT='/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-git'
SELF=str(Path(__file__).resolve())


def run(args, cwd=None, env=None, shell=False):
    p=subprocess.run(args,cwd=cwd,env=env,shell=shell,capture_output=True,text=True)
    return p.returncode,p.stdout.strip(),p.stderr.strip()


def now(): return time.strftime('%F %T')
def cfg_of(exp): return json.loads((exp/'exp.json').read_text())
def load(exp):
    return json.loads((exp/'state.json').read_text()) if (exp/'state.json').exists() else None

def save(exp,s):
    p=exp/'.state.json.tmp';p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');p.replace(exp/'state.json')

@contextmanager
def lock(path):
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def job_state(jid):
    rc,out,_=run(['sacct','-j',jid,'-P','-n','-X','-o','State,ExitCode'])
    if rc or not out: return None,None
    parts=out.splitlines()[0].split('|')
    return parts[0].split()[0].rstrip('+'),parts[1]


def finish(s,reason,failed=False):
    s.update(status='failed' if failed else 'done',stop_reason=reason,
             finalize_requested=True,finalized=False,finished_at=now())
    started=s.get('started_epoch')
    if started is None: started=time.mktime(time.strptime(s['started_at'],'%Y-%m-%d %H:%M:%S'))
    s['elapsed_seconds']=round(time.time()-started,1)
    if failed: s['error']=reason


def stop_reason(cfg,s,check_queue=True):
    rule=cfg.get('stop',{})
    if s['cost_cny']>=rule.get('budget_cny',100): return 'budget reached'
    if check_queue and not s['queue']: return 'queue exhausted'
    return None


def request_review(exp,s,kind,error=None):
    """Record an observation for the researcher; never infer a research stopping rule."""
    number=s.get('review_count',0)+1
    event=dict(id=f'event-{number:04d}',kind=kind,error=error,at=now(),
               job_id=s.get('job_id'),hook_id=s.get('hook_id'),current=s.get('current'),
               latest=s['history'][-1] if kind=='round_finished' and s['history'] else None,
               queued_jobs=list(s['queue']),cost_cny=s['cost_cny'],delivery='pending')
    s.update(status='awaiting_agent',review_count=number,review=event,finalize_requested=False)
    events=exp/'events';events.mkdir(exist_ok=True)
    (events/(event['id']+'.json')).write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n')
    save(exp,s)


def notify_agent(exp):
    """One event-triggered HTTP delivery attempt, outside the state lock. Cron only retries delivery."""
    s=load(exp)
    if not s or s['status']!='awaiting_agent' or s['review'].get('delivery')=='accepted': return
    try:
        binding=cfg_of(exp).get('agent_bridge')
        if not binding: raise ValueError('No agent_bridge configured; result has NOT reached the agent')
        bridge=json.loads(Path(binding).read_text())
        request=urllib.request.Request(
            f"http://{bridge['host']}:{bridge['port']}/event",
            data=json.dumps(dict(experiment=str(exp),event_id=s['review']['id'])).encode(),
            headers={'Authorization':'Bearer '+bridge['token'],'Content-Type':'application/json'})
        # Cluster event traffic is local, never route it through the model/API proxy.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=15) as response:
            response.read()
    except (OSError,ValueError,KeyError) as error:
        with lock(exp/'.lock'):
            current=load(exp)
            if current.get('review',{}).get('id')==s['review']['id'] and current['review'].get('delivery')!='accepted':
                current['review']['delivery_error']=str(error)[-400:];save(exp,current)


def ensure_hook(exp,cfg,s):
    hooks=exp/'hooks';hooks.mkdir(exist_ok=True)
    event=s.get('job_id') or 'start'
    n=s.get('hook_attempts',0)+1;s['hook_attempts']=n
    path=hooks/f"round-{s.get('rounds_submitted',0):03d}-attempt-{n}.sbatch"
    node_line = f"#SBATCH --nodelist={cfg['hook_node']}\n" if cfg.get('hook_node') else ''
    path.write_text(f'''#!/usr/bin/env bash
#SBATCH --partition={cfg.get('hook_partition',cfg.get('partition','gre'))}
{node_line}#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:05:00
#SBATCH --output={hooks}/hook-%j.out
#SBATCH --error={hooks}/hook-%j.err
set -euo pipefail
/usr/bin/python3 {shlex.quote(SELF)} advance {shlex.quote(str(exp))} {shlex.quote(event)}
''')
    command=['sbatch','--parsable']
    if s.get('job_id'): command.append('--dependency=afterany:'+s['job_id'])
    rc,out,err=run(command+[str(path)])
    s['hook_id']=None if rc else out.split(';')[0]
    if rc: s['hook_error']=(err or out)[-500:]
    else: s.pop('hook_error',None)
    save(exp,s)


def submit_round(exp,cfg,s):
    reason=stop_reason(cfg,s)
    if reason:
        request_review(exp,s,'submission_blocked',reason);return
    job=s['queue'][0]
    rc,out,err=run(['scontrol','ping'])
    if rc or ' is DOWN' in out:
        request_review(exp,s,'submission_failed','Slurm control unavailable: '+(err or out)[-300:]);return
    rc,out,err=run(['sbatch','--parsable',job],cwd=cfg.get('repo') or str(exp))
    if rc:
        request_review(exp,s,'submission_failed','sbatch failed: '+(err or out)[-300:]);return
    s['queue'].pop(0)
    s.update(current=job,job_id=out.split(';')[0],hook_id=None,recorded_current=False,
             submitted_at=now(),last_advance_at=now(),rounds_submitted=s.get('rounds_submitted',0)+1)
    # Persist the parent before attempting its hook: cron can repair a failed hook submission.
    save(exp,s)
    ensure_hook(exp,cfg,s)


def read_metric(exp,cfg,s):
    cmd=cfg.get('metric')
    if not cmd: return {}  # Engineering/diagnostic experiments need not have a scalar score.
    env=dict(os.environ,EXP_LOOP_JOB_ID=s['job_id'],EXP_LOOP_JOB=s['current'],
             EXP_LOOP_ROUND=str(s['rounds']+1),EXP_LOOP_DIR=str(exp))
    rc,out,err=run(cmd,cwd=str(exp),env=env,shell=isinstance(cmd,str))
    if rc: raise ValueError('metric failed: '+(err or out)[-400:])
    metric=json.loads(out.splitlines()[-1])
    primary=metric.get('primary');cost=metric.get('cost_cny',0)
    if primary is not None and (isinstance(primary,bool) or not isinstance(primary,(int,float)) or not math.isfinite(primary)):
        raise ValueError('metric.primary must be a finite number')
    if isinstance(cost,bool) or not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0:
        raise ValueError('metric.cost_cny must be a nonnegative finite number')
    return metric


def accounted_cost(history):
    """Replay measured costs; cumulative metrics already include earlier rounds."""
    total=0.
    for row in history:
        metric=row.get('metric') or {}
        cost=metric.get('cost_cny',row.get('cost_cny',0))
        total=max(total,cost) if metric.get('cumulative') else total+cost
    return round(total,6)


def record(exp,cfg,s,state,ec):
    error=None
    try: metric=read_metric(exp,cfg,s)
    except (ValueError,IndexError,TypeError,OSError) as e: metric={};error=str(e)
    primary=metric.get('primary');cost=metric.get('cost_cny',0)
    previous_cost=accounted_cost(s['history'])
    s['cost_cny']=accounted_cost([*s['history'],dict(metric=metric)])
    round_cost=round(s['cost_cny']-previous_cost,6)
    gain=None
    if primary is not None:
        old=s['primary_best'];gain=None if old is None else primary-old
        s['primary_best']=primary if old is None else max(old,primary)
    s['history'].append(dict(job=Path(s['current']).name,job_id=s['job_id'],hook_id=s.get('hook_id'),
        state=state,exit=ec,primary=primary,gain=gain,cost_cny=round_cost,metric=metric,metric_error=error,at=now()))
    s['done'].append(s['current']);s['rounds']+=1;s['recorded_current']=True
    request_review(exp,s,'round_finished',f'{state} exit={ec}' if state!='COMPLETED' else error)


def init(exp):
    with lock(exp/'.lock'):
        if load(exp): raise ValueError('Already initialized; use add/status, never overwrite state')
        cfg=cfg_of(exp)
        jobs=sorted(glob.glob(str(exp/cfg.get('jobs_glob','jobs/*.sbatch'))))
        s=dict(name=cfg.get('name',exp.name),status='running' if jobs and cfg.get('auto_start',True) else 'ready',
            queue=jobs,done=[],current=None,job_id=None,hook_id=None,rounds=0,rounds_submitted=0,
            primary_best=cfg.get('initial_primary'),cost_cny=0.,finalized=False,
            finalize_requested=False,started_at=now(),started_epoch=time.time(),history=[],driver='dependency',recorded_current=False)
        save(exp,s)
        if s['status']=='running': submit_round(exp,cfg,s)
    notify_agent(exp)
    return status(exp)


def advance(exp,event):
    with lock(exp/'.lock'):
        s=load(exp)
        if not s or s['status']!='running' or event!=(s.get('job_id') or 'start'): return 0
        cfg=cfg_of(exp)
        if event!='start' and not s.get('recorded_current'):
            state,ec=job_state(s['job_id'])
            if state not in TERMINAL: return 0  # Accounting delay: completed hook will be repaired by cron.
            record(exp,cfg,s,state,ec)
        if s['status']=='running': submit_round(exp,cfg,s)
        print(json.dumps(dict(name=s['name'],rounds=s['rounds'],status=s['status']),ensure_ascii=False))
    notify_agent(exp)
    return 0


def tick(exp):
    """Repair controller hooks only. Never read metrics or submit an experiment."""
    with lock(exp/'.lock'):
        s=load(exp)
        if not s or s['status']!='running': return 0
        hook=s.get('hook_id')
        if hook:
            state,_=job_state(hook)  # Only this experiment's recorded hook ID.
            if state not in TERMINAL: return 0
        ensure_hook(exp,cfg_of(exp),s)
    return 0


def add(exp,job=None,decision=None,reason=None):
    if job:
        job=str(Path(job).resolve())
        if not Path(job).is_file(): raise ValueError('Job script does not exist')
    with lock(exp/'.lock'):
        s=load(exp)
        if not s: raise ValueError('Initialize first')
        s['cost_cny']=accounted_cost(s.get('history',[]))
        cfg=cfg_of(exp)
        if s['status']=='running':
            if decision: raise ValueError('Wait for the current job event before deciding')
            if job and job not in s['queue']: s['queue'].append(job)
            save(exp,s);return 0
        if s['status']=='awaiting_agent' or decision:
            if not reason: raise ValueError('Record the evidence/reason for this decision with --reason')
            if decision=='finish':
                s.setdefault('decisions',[]).append(dict(event_id=s.get('review',{}).get('id'),action='finish',reason=reason,at=now()))
                finish(s,reason);save(exp,s);return status(exp)
            if decision=='retry':
                job=(s['queue'][0] if s.get('review',{}).get('kind')=='submission_failed' else s.get('current'))
                if not job: raise ValueError('No failed/current script to retry')
            if not job and not s['queue']: raise ValueError('No next experiment; add a script or explicitly --finish')
            if stop_reason(cfg,s,False): raise ValueError('Actual budget reached; cannot submit more in this research budget')
            s.setdefault('decisions',[]).append(dict(event_id=s.get('review',{}).get('id'),action=decision or 'add',job=job,reason=reason,at=now()))
            if job:
                if job in s['queue']: s['queue'].remove(job)
                s['queue'].insert(0,job)
        elif job and job not in s['queue']: s['queue'].append(job)
        if s['status'] in ('awaiting_agent','ready','empty','done','failed','stopped'):
            if stop_reason(cfg,s,False): raise ValueError('Actual budget reached')
            s.update(status='running',current=None,job_id=None,hook_id=None,recorded_current=False,
                     finalized=False,finalize_requested=False,finalize_attempts=0,git_committed=False,git_pushed=False,notified=False)
            for k in ('stop_reason','finished_at','finalize_error','finalize_blocked','finalize_retry_at'): s.pop(k,None)
            save(exp,s)
            ensure_hook(exp,cfg,s)  # A start hook, not cron, launches the next experiment.
        else: save(exp,s)
    return status(exp)


def status(exp):
    s=load(exp)
    if not s: print('Not initialized');return 1
    print(json.dumps(s,ensure_ascii=False,indent=2))
    return 0


def git_finalize(exp,cfg,s):
    settings=cfg.get('git') or {}
    if not settings: return
    repo=Path(settings['repo']);paths=settings.get('paths')
    if not paths or any(p in ('.','*') or Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
        raise ValueError('git.paths must explicitly name experiment files inside repo')
    export=settings.get('state_export')
    if export and not s.get('git_committed'):
        if Path(export).is_absolute() or '..' in Path(export).parts:
            raise ValueError('state_export must be inside repo')
        target=repo/export;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    def git(*args):
        rc,out,err=run(['git','-C',str(repo),*args])
        if rc: raise ValueError('git '+args[0]+': '+(err or out)[-400:])
        return out
    gitdir=Path(git('rev-parse','--absolute-git-dir'))
    with lock(gitdir/'exp-loop.lock'):
        if git('branch','--show-current')!=settings['branch']: raise ValueError('Unexpected repository branch')
        if not s.get('git_committed'):
            git('add','-A','--',*paths)
            rc,_,err=run(['git','-C',str(repo),'diff','--cached','--quiet','--',*paths])
            if rc==1:
                git('commit','--only','-m',f"auto: {s['name']} {s['rounds']} rounds",'--',*paths)
            elif rc: raise ValueError(err)
            s['git_committed']=True;save(exp,s)
        if not s.get('git_pushed'):
            rc,out,err=run([WGIT,'-C',str(repo),'push','origin',settings['branch']])
            if rc: raise ValueError('push failed: '+(err or out)[-400:])
            s['git_pushed']=True;save(exp,s)


def finalize(exp):
    with lock(exp/'.lock'):
        s=load(exp)
        if not s or not s.get('finalize_requested') or s.get('finalized'): return 0
        if s.get('finalize_blocked') or time.time()<s.get('finalize_retry_at',0): return 0
        cfg=cfg_of(exp)
        if cfg.get('dry_run'):
            print('dry-run: no git/QQ, finalization remains pending');return 0
        s['finalize_attempts']=s.get('finalize_attempts',0)+1
        try:
            git_finalize(exp,cfg,s)
            if cfg.get('notify',True) and not s.get('notified'):
                body=(f"{s['name']}: {s['status']}; rounds={s['rounds']}; best={s['primary_best']}; "
                      f"cost={s['cost_cny']} CNY; reason={s.get('stop_reason')}; "
                      f"elapsed_seconds={s.get('elapsed_seconds')}; started={s.get('started_at')}; ended={s.get('finished_at')}; artifacts={exp}; "
                      f"repo={(cfg.get('git') or {}).get('repo','-')}")
                rc,_,err=run([QQ],env=dict(os.environ,QQ_SUB=f"[{s['name']}] {s['status']}",QQ_BODY=body))
                if rc: raise ValueError('QQ failed: '+err[-300:])
                s['notified']=True;save(exp,s)
        except (ValueError,OSError) as e:
            s['finalize_error']=str(e);s['finalize_retry_at']=time.time()+cfg.get('finalize_retry_seconds',60)
            s['finalize_blocked']=s['finalize_attempts']>=cfg.get('max_finalize_attempts',5)
            if s['finalize_blocked'] and cfg.get('notify',True) and not s.get('finalize_failure_notified'):
                body=f"{s['name']}: finalization failed after {s['finalize_attempts']} attempts; error={e}; artifacts={exp}; started={s.get('started_at')}; ended={s.get('finished_at')}"
                rc,_,_=run([QQ],env=dict(os.environ,QQ_SUB=f"[{s['name']}] finalization failed",QQ_BODY=body))
                s['finalize_failure_notified']=(rc==0)
            save(exp,s);print(str(e));return 1
        s.update(finalized=True,finalize_requested=False,finalized_at=now());s.pop('finalize_error',None)
        save(exp,s)
    return 0


def main():
    if len(sys.argv)<3: print(__doc__);return 2
    cmd=sys.argv[1];exp=Path(sys.argv[2]).resolve()
    try:
        if cmd=='advance': return advance(exp,sys.argv[3])
        if cmd=='add':
            import argparse
            parser=argparse.ArgumentParser()
            parser.add_argument('job',nargs='?')
            group=parser.add_mutually_exclusive_group()
            for action in ('continue','retry','finish'): group.add_argument('--'+action,dest='decision',action='store_const',const=action)
            parser.add_argument('--reason')
            args=parser.parse_args(sys.argv[3:])
            return add(exp,args.job,args.decision,args.reason)
        return {'init':init,'status':status,'tick':tick,'finalize':finalize}[cmd](exp)
    except (ValueError,KeyError,IndexError,OSError) as e:
        print(str(e),file=sys.stderr);return 1

if __name__=='__main__': sys.exit(main())
