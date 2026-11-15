"""Prepare/execute serial, paired genuine Agent runs; never selects using labels.

Run prepare and run only in Slurm. prepare freezes six existing development tasks
and their cached candidate records; it does not submit jobs or call an API.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
from agent_baseline.find_missed import CachedMissedCandidates, GROUPS, SOURCES
from .io_retry import write_json, write_text
from .missed_online_report import complete_final

SEEDS=(1016,2027,3038)
ARMS=('baseline','find_missed')
MODEL='qwen3.7-flash'


def read_json(path): return json.loads(Path(path).read_text())


def common_arguments(tasks,config,seed):
    return ['--tasks',str(tasks),'--config',config,'--split','val','--concurrency','1',
        '--seed-offset',str(seed),'--planner-model',MODEL,'--ask-backend','api','--ask-model',MODEL,
        '--ask-correspondence','--pixel-thr','0.85','--prompt-version','v4','--status',
        '--verify-stage','--gate5','--sam-prompt-device','cpu','--batch','missed_online']


def arm_arguments(arm,variant,cache):
    if arm=='baseline':return []
    return ['--missed-candidates-root',str(cache),'--missed-verifier-variant',variant]


def prepare(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run prepare through Slurm')
    design=read_json(args.selection/'design.json')
    if design['selection_seed']!=1016 or design['seed_offsets']!=list(SEEDS):
        raise ValueError('Use the frozen six-task selection and three seeds')
    tasks=[]
    for group in GROUPS:
        rows=[json.loads(s) for s in (args.selection/(group+'.jsonl')).read_text().splitlines() if s.strip()]
        if len(rows)!=2 or [r['id'] for r in rows]!=design['selected'][group]:
            raise ValueError('Frozen task selection mismatch')
        for row in rows:
            stem=Path(row['id']).stem
            with np.load(args.cache/'candidates'/group/'S1_02'/(stem+'.npz'),allow_pickle=False) as z:
                shape=tuple(int(v) for v in z['shape'])
            provider=CachedMissedCandidates(args.cache,group,stem,shape)
            ranked=provider.load()
            # Inference-only candidate archive: no GT and no new source inference.
            folder=args.out/'candidate_manifest'/group;folder.mkdir(parents=True,exist_ok=True)
            candidates=[]
            for rank,r in enumerate(ranked,1):
                candidates.append(dict(rank=rank,source=r['source'],sources=r['sources'],original=r['original'],area=len(r['idx'])))
            write_json(folder/(stem+'.json'),dict(group=group,stem=stem,shape=shape,candidates=candidates,
                initial_top20=candidates[:20],ranking='fixed_source_round_robin_area',
                cache=str(args.cache),gt_role='no_label_access',
                online_rule='subtract current answer; skip emptied or identical residuals, retain original source order'))
            taskpath=args.out/'tasks'/group/(stem+'.jsonl')
            write_text(taskpath,json.dumps(row,ensure_ascii=False)+'\n')
            tasks.append(dict(group=group,stem=stem,task_type=row['task_type'],tasks=str(taskpath.resolve()),
                              initial_mask=row['init_mask'],image_t1=row['image_t1'],image_t2=row['image_t2']))
    settings=dict(version=1,seeds=list(SEEDS),arms=list(ARMS),variant=args.variant,tasks=tasks,
        cache=str(args.cache.resolve()),code=str(args.code.resolve()),output=str(args.out.resolve()),
        common_arguments=common_arguments('TASK','CONFIG',1016),selection=str(args.selection.resolve()),
        only_arm_differences=['find_missed candidate tool','controlled verifier variant '+args.variant],
        interpretation='joint intervention; not separate attribution to candidate tool versus verifier',
        sources=SOURCES,model=MODEL,expected_sessions=36)
    settings=json.loads(json.dumps(settings))
    path=args.out/'settings.json'
    if path.exists() and read_json(path)!=settings:raise ValueError('Frozen settings differ; use a new online output')
    write_json(path,settings)
    for group in GROUPS:
        for seed in SEEDS:
            path=args.out/'jobs'/f'{group}-seed{seed}.sbatch'
            cmd=[args.python,'-B','-m','agent_baseline.analysis.missed_online_runner','run','--out',str(args.out.resolve()),'--group',group,'--seed',str(seed)]
            script='\n'.join(['#!/usr/bin/env bash','#SBATCH --partition=gre','#SBATCH --nodelist=gpu42',
                '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=48G','#SBATCH --time=04:00:00',
                f'#SBATCH --output={args.out.resolve()}/logs/{group}-seed{seed}-%j.out',
                f'#SBATCH --error={args.out.resolve()}/logs/{group}-seed{seed}-%j.err','set -euo pipefail',
                ': "${BAILIAN_CSV:?Existing workspace endpoint credential CSV}"',
                'export PYTHONPATH='+shlex.quote(str(args.code.resolve().parent)),
                'export OVCD_CODE='+shlex.quote(str(args.ovcd.resolve())),
                'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
                'report_on_exit() { status=$?; trap - EXIT; '+shlex.join([args.python,'-B','-m','agent_baseline.analysis.missed_online_report','--out',str(args.out.resolve()),'--publish-root'])+' || true; exit "$status"; }',
                'trap report_on_exit EXIT',
                shlex.join(cmd),''])
            write_text(path,script)
    (args.out/'logs').mkdir(parents=True,exist_ok=True)
    write_json(args.out/'prepare_summary.json',dict(tasks=6,sessions=36,jobs=9,status='prepared_not_submitted'))


def attempts(out,group,seed,arm,stem):
    return sorted((out/'run'/group/f'seed{seed}'/arm/stem).glob('attempt_*'))


def successful(path,task):
    final=path/task/'val'/path.parent.name/'final.json'
    return final.exists() and complete_final(read_json(final))


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    settings=read_json(args.out/'settings.json');code=Path(settings['code'])
    env=os.environ.copy();env.update(AGENT_RUN=str((args.out/'run').resolve()),AGENT_CODE=str(code),
        AGENT_STOP_FILE=str((args.out/'run'/'STOP.json').resolve()),USE_ASK_WORKER='0',USE_LOCAL_WORKERS='1',
        AGENT_ASK_SLOTS='1',
        PYTHONPATH=str(code.parent)+os.pathsep+env.get('PYTHONPATH',''))
    stop=Path(env['AGENT_STOP_FILE'])
    if stop.exists():raise RuntimeError('Shared STOP exists; resolve it before resuming')
    work=[];commands=[]
    for row in settings['tasks']:
        if row['group']!=args.group:continue
        for arm in ARMS:
            prior=attempts(args.out,args.group,args.seed,arm,row['stem'])
            if any(successful(p,row['task_type']) for p in prior):continue
            if prior and not args.retry_failed:continue
            number=len(prior);config=f"{args.group}/seed{args.seed}/{arm}/{row['stem']}/attempt_{number:03d}"
            destination=args.out/'run'/config/row['task_type']/'val'/row['stem']
            arguments=common_arguments(row['tasks'],config,args.seed)+arm_arguments(arm,settings['variant'],settings['cache'])
            commands.extend(([':::'] if commands else [])+arguments)
            item=dict(group=args.group,seed=args.seed,arm=arm,stem=row['stem'],attempt=number,directory=str(destination),arguments=arguments)
            write_json(destination.parent.parent.parent/'attempt.json',dict(item,status='scheduled',started_at=time.time()))
            work.append(item)
    if not commands:
        from .missed_online_report import report
        summary=report(args.out,publish_root=True)
        current=[r for r in summary['sessions'] if r['group']==args.group and r['seed']==args.seed]
        if not all(r['complete'] for r in current):
            raise SystemExit('Previous failed attempts preserved; use --retry-failed after review')
        return
    command=['bash',str(code/'run_agent.sbatch'),*commands]
    write_json(args.out/'launches'/f'{args.group}-seed{args.seed}-{os.environ["SLURM_JOB_ID"]}.json',dict(command=command,sessions=work))
    result=subprocess.run(command,env=env,check=False)
    for item in work:
        destination=Path(item['directory']);final=destination/'final.json'
        status='complete' if final.exists() and complete_final(read_json(final)) else 'failed_or_not_reached'
        write_json(destination.parent.parent.parent/'attempt.json',dict(item,status=status,returncode=result.returncode,ended_at=time.time()))
    from .missed_online_report import report
    report(args.out,publish_root=True)
    if result.returncode:raise SystemExit(result.returncode)
    if any(not (Path(item['directory'])/'final.json').exists() or not complete_final(read_json(Path(item['directory'])/'final.json')) for item in work):
        raise SystemExit('At least one Agent session failed or remained incomplete; artifacts preserved')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--code',type=Path)
    p.add_argument('--ovcd',type=Path,default=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/ovcd_v0'))
    p.add_argument('--python',default='/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python')
    p.add_argument('--variant',choices=['four','som','structured'],default='structured')
    p.add_argument('--group',choices=GROUPS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
