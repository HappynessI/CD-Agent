"""One explicitly requested illustrative online run; not a benchmark or new tuning."""
import os,json,sys,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from .n8_missed_assets import read,write,ROOT
from .n9_missed_online import start_worker,arguments
from .n9_api_preflight import check

def main():
    assert os.environ.get('SLURM_JOB_ID')
    out=ROOT/'CD-Agent/results/agent_baseline/demo_find_missed_replay'
    source=ROOT/'CD-Agent/results/agent_baseline/n9_missed_20261116/method_revision_s5_only'
    group,stem='second_appeared','train_2255'
    asset=next(r for r in read(source/'asset_manifest.json') if (r['group'],r['stem'])==(group,stem))
    task=dict(next(r['task'] for r in read(source/'evaluation_manifest.json') if (r['group'],r['stem'])==(group,stem)))
    config=read(source/'frozen_profiles.json')['profiles']['build']['config']
    out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    check(out/'preflight.json')
    with np.load(asset['base'],allow_pickle=False) as z:initial=z['start'].astype(bool)
    init=out/'initial.png';Image.fromarray(initial.astype('uint8')*255).save(init)
    task['init_mask']=str(init)
    task['instruction']+=' Review the supplied initial answer. For this tool demonstration, call find_missed once, inspect its evidence, decide which proposals to adopt, and submit your final mask. Do not call find_missed repeatedly.'
    tasks=out/'task.jsonl';tasks.write_text(json.dumps(task)+'\n')
    from agent_baseline.tasks import seed_for,image_name
    runtime=dict(arm='find_missed',asset=asset,config=config,session_seed=seed_for(image_name(task),1016),export_funnel=True)
    write(out/'runtime.json',runtime)
    write(out/'selection.json',dict(group=group,stem=stem,reason='User-authorized illustrative selection from highest positive SECOND confirm gain; visually inspected T2',representative=False,explicit_tool_request=True,online_sessions_authorized=1,model='qwen3.7-flash'))
    code=ROOT/'CD-Agent/code'
    env=os.environ.copy();env.update(AGENT_RUN=str(out/'run'),AGENT_STOP_FILE=str(out/'run/STOP.json'),N9_MISSED_RUNTIME=str(out/'runtime.json'),PYTHONPATH=str(code)+os.pathsep+env.get('PYTHONPATH',''))
    if list((out/'run').glob('**/final.json')):raise RuntimeError('Single run already exists; do not repeat')
    worker,stream,endpoint=start_worker(code,out,env,worker_module='agent_baseline.analysis.missed_replay_worker')
    try:
        args=arguments(dict(tasks=str(tasks)),'illustrative',1016,endpoint,20)
        # No estimated API cost gate; exactly one supplied task and one session.
        pos=args.index('--max-cost-cny');args=args[:pos]
        cmd=[sys.executable,'-B','-m','agent_baseline.analysis.n9_missed_agent_entry',*args]
        write(out/'launch.json',dict(command=cmd,job=os.environ['SLURM_JOB_ID']))
        r=subprocess.run(cmd,cwd=code/'agent_baseline',env=env)
        if r.returncode:raise RuntimeError(f'Agent process failed: {r.returncode}')
        finals=list((out/'run').glob('**/final.json'))
        write(out/'online_status.json',dict(status='finished',finals=[str(p) for p in finals]))
    finally:
        worker.terminate()
        try:worker.wait(timeout=30)
        except subprocess.TimeoutExpired:worker.kill();worker.wait()
        stream.close()
if __name__=='__main__':main()
