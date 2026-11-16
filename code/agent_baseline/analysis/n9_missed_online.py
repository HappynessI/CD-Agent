"""Prepare and run only independently positive N9 lines: 10 images x 2 arms x 3 seeds."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import numpy as np
from PIL import Image
from .n8_missed_assets import ROOT, read, write
from .n8_missed_pipeline import cost_total
from .n9_missed_assets import OUT
from .n9_missed_protocol import profile_for
from .n9_api_preflight import check
from .missed_online_report import complete_final

ARMS = ('baseline', 'find_missed')
CODE = ROOT/'cd-agent-worktrees/n7-missed/code'
PYTHON = '/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python'


def prepare(a):
    decisions = read(a.out/'offline_decision.json')
    evaluation = {(r['group'],r['stem']):r['task'] for r in read(a.out/'evaluation_manifest.json')}
    assets = {(r['group'],r['stem']):r for r in read(a.out/'asset_manifest.json')}
    online = a.out/'online'; tasks=[]
    for line, decision in decisions.items():
        if not decision['advance']: continue
        assert decision['delta_f1']>0 and len(decision['online_stems'])==10
        for stem in decision['online_stems']:
            task = dict(evaluation[line,stem]); asset=assets[line,stem]
            assert asset['split']=='confirm' and profile_for(task['change_direction'])==decision['config']['profile']
            start_path=online/'initial_masks'/line/(stem+'.png'); start_path.parent.mkdir(parents=True,exist_ok=True)
            with np.load(asset['base'],allow_pickle=False) as z: current=z['start'].astype(bool)
            if start_path.exists():
                assert np.array_equal(np.asarray(Image.open(start_path))>0,current)
            else: Image.fromarray(current.astype(np.uint8)*255).save(start_path)
            task.setdefault('task_type','T1.1xC0' if task['change_direction']=='any' else 'T1.1xC1')
            task.setdefault('instruction','Find all buildings that changed between T1 and T2 (newly built or demolished). Output a binary change mask.')
            task.setdefault('constraint',None); task['init_mask']=str(start_path)
            task['meta']=dict(task.get('meta') or {}, n9_line=line, n9_initial_source=asset['base'],
                              n9_online_selection='fixed before N9 scoring')
            task_path=online/'tasks'/line/(stem+'.jsonl'); task_path.parent.mkdir(parents=True,exist_ok=True)
            task_path.write_text(json.dumps(task,ensure_ascii=False)+'\n')
            tasks.append(dict(group=line, stem=stem, task_type=task['task_type'], tasks=str(task_path),
                initial_mask=str(start_path), asset=asset, config=decision['config'], seeds=decision['seeds']))
    settings=dict(tasks=tasks, arms=list(ARMS), seeds=[1016,1017,1018], code=str(a.code),
        expected_sessions=len(tasks)*6, profile_source=str(a.out/'frozen_profiles.json'),
        selection_source=str(a.out/'frozen_protocol.json'),
        interpretation='Only find_missed availability differs; ordinary ask/prompt/tools and starting mask match. Agent chooses calls and adoption.',
        no_offline_answer_reuse=True, online_status='prepared_not_run')
    target=online/'settings.json'
    if target.exists() and read(target)!=settings: raise ValueError('Frozen online settings differ')
    write(target,settings)
    write(a.out/'metric.json',dict(stage='online_prepared',status='complete',positive_lines=sum(d['advance'] for d in decisions.values()),
        expected_sessions=len(tasks)*6,cost_cny=cost_total(a.out),cumulative=True))


def arguments(row, config, seed, worker, remaining):
    return ['--tasks',row['tasks'],'--config',config,'--split','val','--concurrency','1',
        '--seed-offset',str(seed),'--planner-model','qwen3.7-flash','--ask-backend','api',
        '--ask-model','qwen3.7-flash','--ask-correspondence','--pixel-thr','0.85',
        '--prompt-version','v4','--status','--verify-stage','--gate5','--sam-prompt-device','cpu',
        '--sam3-worker',worker,'--batch','n9_missed_online','--max-cost-cny',str(remaining)]


def final_path(online,row,seed,arm,attempt):
    return online/'run'/row['group']/f'seed{seed}'/arm/row['stem']/f'attempt_{attempt:03d}'/row['task_type']/'val'/row['stem']/'final.json'


def start_worker(code, online, env, worker_module=None):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    endpoint=f'http://127.0.0.1:{port}'
    log=online/'logs'/f'sam3-{os.environ["SLURM_JOB_ID"]}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    stream=log.open('a')
    process=subprocess.Popen(['/guisongxia01/pangchao/wangyihan/conda_envs/omniovcd-env/bin/python','-B',
        *(['-m',worker_module] if worker_module else [str(code/'agent_baseline/workers/sam3_worker.py')]),'--port',str(port)],cwd=code/'agent_baseline',env=env,stdout=stream,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError('SAM3 worker exited; inspect its log')
            try:
                with urllib.request.urlopen(endpoint+'/health',timeout=2) as response:
                    if json.load(response).get('ok'): return process,stream,endpoint
            except OSError: pass
            time.sleep(1)
        raise RuntimeError('SAM3 worker readiness deadline exceeded')
    except Exception:
        process.terminate(); process.wait(timeout=30); stream.close(); raise


def run(a):
    online=a.out/'online'; settings=read(online/'settings.json'); code=Path(settings['code'])
    selected=[r for r in settings['tasks'] if r['group']==a.group]
    if len(selected)!=10: raise ValueError('Only a confirmed positive line with ten frozen tasks can run')
    check(online/'preflight'/f'{a.group}-{a.seed}-{os.environ["SLURM_JOB_ID"]}.json')
    remaining=a.budget-cost_total(a.out)
    if remaining<=0: raise RuntimeError('Actual offline spend exhausted FM allocation; main agent must reallocate')
    env=os.environ.copy(); env.update(AGENT_RUN=str(online/'run'),AGENT_STOP_FILE=str(online/'run/STOP.json'),
        PYTHONPATH=str(code)+os.pathsep+env.get('PYTHONPATH',''),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    # Workers share exactly this node/allocation; no public endpoint or off-Slurm process.
    worker,stream,endpoint=start_worker(code,online,env)
    completed=0
    try:
        for row in selected:
            for arm in ARMS:
                prefix=online/'run'/row['group']/f'seed{a.seed}'/arm/row['stem']
                prior=sorted(prefix.glob('attempt_*'))
                finals=[p/row['task_type']/'val'/row['stem']/'final.json' for p in prior]
                if any(p.exists() and complete_final(read(p)) for p in finals): completed+=1; continue
                if prior and not a.retry_failed: raise RuntimeError('Incomplete attempt preserved; review before --retry-failed')
                attempt=len(prior); config=f"{row['group']}/seed{a.seed}/{arm}/{row['stem']}/attempt_{attempt:03d}"
                from agent_baseline.tasks import seed_for
                task= json.loads(Path(row['tasks']).read_text())
                from agent_baseline.tasks import image_name
                name=image_name(task)
                runtime=dict(arm=arm,asset=row['asset'],config=row['config'],session_seed=seed_for(name,a.seed))
                runtime_path=online/'runtime'/row['group']/f'seed{a.seed}'/arm/(row['stem']+f'_attempt{attempt}.json')
                write(runtime_path,runtime); env['N9_MISSED_RUNTIME']=str(runtime_path)
                cmd=[sys.executable,'-B','-m','agent_baseline.analysis.n9_missed_agent_entry',
                     *arguments(row,config,a.seed,endpoint,remaining)]
                launch=online/'launches'/row['group']/f'seed{a.seed}'/arm/(row['stem']+f'_attempt{attempt}.json')
                write(launch,dict(command=cmd,arm=arm,seed=a.seed,stem=row['stem'],state='started',slurm_job=os.environ['SLURM_JOB_ID']))
                result=subprocess.run(cmd,cwd=code/'agent_baseline',env=env,check=False)
                final=final_path(online,row,a.seed,arm,attempt)
                complete=final.exists() and complete_final(read(final))
                write(launch,dict(command=cmd,arm=arm,seed=a.seed,stem=row['stem'],returncode=result.returncode,
                    state='complete' if complete else 'failed',final=str(final),slurm_job=os.environ['SLURM_JOB_ID']))
                if result.returncode or not complete: raise RuntimeError('Online session failed; original attempt preserved')
                completed+=1
                write(online/'progress'/f'{a.group}-{a.seed}.json',dict(completed=completed,expected=20,status='running'))
        write(online/'progress'/f'{a.group}-{a.seed}.json',dict(completed=completed,expected=20,status='complete'))
    finally:
        worker.terminate()
        try: worker.wait(timeout=30)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()
        stream.close()
        from .n9_missed_online_report import report
        report(a.out)


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('prepare','run'))
    p.add_argument('--out',type=Path,default=OUT); p.add_argument('--code',type=Path,default=CODE)
    p.add_argument('--group'); p.add_argument('--seed',type=int,choices=(1016,1017,1018))
    p.add_argument('--budget',type=float,default=20.); p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    globals()[a.command](a)

if __name__=='__main__': main()
