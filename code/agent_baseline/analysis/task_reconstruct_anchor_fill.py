"""Offline local-model road-aware semantic anchors; no API or Agent inference."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from agent_baseline.t13_worker import DynamicBackend
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/agent_baseline/task_reconstruct_audit/t44/road_stage'
CLASSES=['low vegetation','non-vegetated ground','tree','water','building','playground','road']
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    cfg=json.loads((ROOT.parent/'exp6_staging/agent_baseline_run/t3x/vision.json').read_text())['dynamic']
    runtime=OUT/'model_runtime';runtime.mkdir(parents=True,exist_ok=True)
    backend=DynamicBackend(cfg,runtime)
    original=json.loads((ROOT/'results/agent_baseline/n9_missed_20261116/asset_manifest.json').read_text())
    rows=[];seen=set();started=time.monotonic()
    for item in original:
        dataset=item['group'].split('_')[0]
        if dataset not in ('whu','second'):continue
        row=dict(item);row['semantic']={}
        for t in ('t1','t2'):
            dest=OUT/'semantic'/dataset/(item['stem']+'_'+t+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with Image.open(item['image_'+t]) as image:rgb=np.asarray(image.convert('RGB'))
                with torch.inference_mode():value=backend.semantic(rgb,CLASSES)
                np.savez_compressed(dest,data=value)
            row['semantic'][t]=str(dest);seen.add(str(dest))
        rows.append(row)
        (OUT/'fill_progress.json').write_text(json.dumps(dict(completed_images=len(seen),elapsed_seconds=time.monotonic()-started,api_cost_cny=0)))
    (OUT/'asset_manifest.json').write_text(json.dumps(rows))
    (OUT/'protocol.json').write_text(json.dumps(dict(classes=CLASSES,reason='Existing six-class caches have no road channel; high-confidence stable water/playground yielded zero valid queries',inference='local SegEarth, offline construction only',api_cost_cny=0),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
