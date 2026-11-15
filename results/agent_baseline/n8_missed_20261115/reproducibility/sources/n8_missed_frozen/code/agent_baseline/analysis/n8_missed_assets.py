"""Inventory N8 full source assets, then fill missing semantic/DINO maps on GPU."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path('/guisongxia01/pangchao/wangyihan/wyh')
RUN=ROOT/'exp6_staging/agent_baseline_run'
WORK=ROOT/'exp6_staging/batch99_find_missed_pre/artifacts'
HISTORY=ROOT/'CD-Agent/results/agent_baseline/night_missed_20261114'
OUT=ROOT/'CD-Agent/results/agent_baseline/n8_missed_20261115'
SOURCES=('S1_02','S2','S3','S4','S5')
CLASSES=['low vegetation','non-vegetated ground','tree','water','building','playground']

def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def inventory(a):
    old=read(HISTORY/'rank_V0/manifest.json')
    tasks=[json.loads(l) for l in (RUN/'results/agent_baseline/data/t3_tasks_dev_0of1/tasks.jsonl').read_text().splitlines() if l.strip()]
    known={}
    for task in tasks:
        for t in ('t1','t2'):
            p=ROOT/'exp6_staging/batch98_t3_routes/results/route_a/semantic'/f"{task['id']}_{t}.npz"
            if p.exists(): known[str(Path(task['image_'+t]).resolve())]=str(p)
    rows=[];missing=[];unique={}
    for oldrow in old:
        group,stem=oldrow['group'],oldrow['stem'];source=group.split('_')[0]
        row={k:oldrow[k] for k in ('group','stem','split','shape','direction','scope','image_t1','image_t2')}
        row['base']=str(WORK/'base'/group/(stem+'.npz'))
        row['sources']={s:str(WORK/'candidates'/group/s/(stem+'.npz')) for s in SOURCES}
        row['semantic']={};row['features']={}
        for t in ('t1','t2'):
            image=str(Path(row['image_'+t]).resolve())
            sem=known.get(image,str(a.out/'semantic'/source/(stem+'_'+t+'.npz')))
            feature=WORK/'gpu_cache/dynamic'/source/stem/f'extract_features_{t}.npz'
            if not feature.exists(): feature=HISTORY/'candidate_gates_pilot/feature_cache'/source/stem/f'extract_features_{t}.npz'
            if not feature.exists(): feature=a.out/'features'/source/(stem+'_'+t+'.npz')
            row['semantic'][t]=str(sem);row['features'][t]=str(feature)
            unique[image]=dict(image=image,semantic=str(sem),features=str(feature))
        for p in (row['base'],*row['sources'].values(),row['image_t1'],row['image_t2']):
            if not Path(p).is_file():missing.append(p)
        rows.append(row)
    assert len(rows)==30 and sum(r['split']=='tune' for r in rows)==15
    write(a.out/'asset_manifest.json',rows)
    report=dict(status='complete' if not missing else 'missing_required_source',tasks=len(rows),images=len(unique),
        missing_required=missing,semantic_available=sum(Path(r['semantic']).is_file() for r in unique.values()),
        semantic_missing=sum(not Path(r['semantic']).is_file() for r in unique.values()),
        dino_available=sum(Path(r['features']).is_file() for r in unique.values()),
        dino_missing=sum(not Path(r['features']).is_file() for r in unique.values()),assets=list(unique.values()),
        config=str(RUN/'t3x/vision.json'),probability_classes=['background',*CLASSES],building_channel=5)
    write(a.out/'asset_inventory.json',report)
    write(a.out/'metric.json',dict(stage='inventory',status=report['status'],cost_cny=0,completed=len(rows),cumulative=True))
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets','missing_required')}))
    if missing: raise RuntimeError('Missing full source inputs; see asset_inventory.json')

def fill(a):
    import numpy as np
    import torch
    from agent_baseline.t13_worker import DynamicBackend
    from agent_baseline.analysis.io_retry import read_image
    assert torch.cuda.is_available(),'GPU Slurm required'
    report=read(a.out/'asset_inventory.json');cfg=read(RUN/'t3x/vision.json')['dynamic']
    backend=DynamicBackend(cfg,a.out/'model_runtime');(a.out/'model_runtime').mkdir(exist_ok=True)
    completed=[];started=time.monotonic()
    for row in report['assets']:
        needs=[key for key in ('semantic','features') if not Path(row[key]).is_file()]
        if not needs:continue
        image=read_image(row['image'],rgb=True)
        for key in needs:
            start=time.monotonic()
            with torch.inference_mode(): value=backend.semantic(image,CLASSES) if key=='semantic' else backend.features(image)
            if key=='semantic':assert value.shape==(7,*image.shape[:2]) and np.isfinite(value).all()
            dest=Path(row[key]);dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,data=value)
            completed.append(dict(image=row['image'],kind=key,path=str(dest),seconds=time.monotonic()-start,shape=list(value.shape)))
            write(a.out/'asset_fill.json',dict(status='running',completed=completed,elapsed_seconds=time.monotonic()-started))
    write(a.out/'asset_fill.json',dict(status='complete',completed=completed,elapsed_seconds=time.monotonic()-started,slurm_job=os.environ['SLURM_JOB_ID']))
    write(a.out/'metric.json',dict(stage='fill_assets',status='complete',cost_cny=0,completed=len(completed),cumulative=True))
    print('N8 missing semantic/DINO maps filled.')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inventory','fill']);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    a.out.mkdir(parents=True,exist_ok=True);globals()[a.command](a)

if __name__=='__main__':main()
