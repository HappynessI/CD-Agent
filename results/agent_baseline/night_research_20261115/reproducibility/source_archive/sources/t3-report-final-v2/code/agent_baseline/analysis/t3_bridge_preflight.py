"""Small online prerequisites and fixed-task support audit; execute in Slurm."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def readable(path, executable=False):
    path=Path(path)
    if not path.exists() or not os.access(path,os.R_OK):
        raise ValueError(f'Required path is unavailable: {path}')
    if executable and not os.access(path,os.X_OK):
        raise ValueError(f'Interpreter is not executable: {path}')
    return str(path)


def task_support(row):
    """Scoring-side audit only; these counts never enter the planner prompt."""
    if row['task_type']!='T3.1': raise ValueError('Expected existing T3.1 tasks')
    for key in ('image_t1','image_t2','exemplar_mask','gt_mask'): readable(row[key])
    with Image.open(row['image_t1']) as image: shape=(image.height,image.width)
    with Image.open(row['image_t2']) as image:
        if (image.height,image.width)!=shape: raise ValueError('Image pair sizes differ')
    exemplar=np.asarray(Image.open(row['exemplar_mask']))>0
    truth=np.asarray(Image.open(row['gt_mask']))>0
    if exemplar.shape!=shape or truth.shape!=shape: raise ValueError('Task mask dimensions differ')
    if not exemplar.any(): raise ValueError('Empty supplied visual exemplar')
    return dict(id=row['id'],shape=list(shape),exemplar_pixels=int(exemplar.sum()),
        gt_pixels=int(truth.sum()),remaining_gt_pixels=int((truth & ~exemplar).sum()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run preflight through Slurm')
    root=Path('/guisongxia01/pangchao/wangyihan')
    ovcd=Path(os.environ.get('OVCD_CODE',args.repo/'code/ovcd_v0'))
    checked=[readable(ovcd/name) for name in ('common.py','prompts.py','render.py','mllm.py','sam_tools.py')]
    checked += [readable(args.repo/'code/agent_baseline'/name) for name in
        ('run_agent.sbatch','loop_json.py','exemplar_semantics.py','semantic_bridge_tools.py','prompts/system_v4.txt','workers/sam3_worker.py')]
    config_path=args.repo/'code/agent_baseline/vision_t13.example.json'
    config=json.loads(config_path.read_text())
    for values in config.values():
        for name,path in values.items():
            if name!='env': checked.append(readable(path,executable=name=='python'))
    checked += [readable(root/'conda_envs'/name/'bin/python',executable=True) for name in ('exp5-env','omniovcd-env')]
    checked += [readable(root/path) for path in ('Chang-Agent/tools/run_with_bailian_csv.py',
        'Chang-Agent/probe/sam3_tools.py','models/sam3/sam3.pt',
        'Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz')]
    # Only availability is checked; credential contents never enter this report.
    readable(os.environ['BAILIAN_CSV'])
    from .io_retry import write_json
    from agent_baseline.state import Session
    from agent_baseline.tools import session_schemas
    from .t3_routes_agent import sample_ten
    from .t3_routes import load_dev
    rows=[json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    expected=sample_ten(load_dev(root/'wyh/exp6_staging/agent_baseline_run'))
    if rows!=expected: raise ValueError('Online manifest differs from fixed historical agent10 selection')
    support=[task_support(row) for row in rows]
    session=Session((512,512),target='any change')
    base=session_schemas(session,tools_t13=True)
    bridge=session_schemas(session,tools_t13=True,t3_semantic_bridge=True)
    base_names={s['function']['name'] for s in base}
    bridge_names={s['function']['name'] for s in bridge}
    if bridge_names-base_names!={'interpret_exemplar','rank_semantic_transition'}:
        raise ValueError('Bridge schema difference is not the two opt-in tools')
    write_json(args.output,dict(status='ready_for_real_worker_start',source=str(args.repo),ovcd_source=str(ovcd),
        checked_paths=checked,credentials_file_readable=True,tasks=support,
        empty_remaining_gt_tasks=sum(r['remaining_gt_pixels']==0 for r in support),
        schema_additions=sorted(bridge_names-base_names),api_model='qwen3.7-flash',
        limitation='No API, CUDA inference, environment import or model load is performed by this check.'))


if __name__=='__main__': main()
