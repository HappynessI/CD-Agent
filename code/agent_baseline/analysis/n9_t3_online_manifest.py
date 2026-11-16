"""Explicit approved-task freeze; no automatic conversion of QC into approval."""
import argparse
import json
import random
import shutil
import os
import numpy as np
from PIL import Image
from pathlib import Path
from .io_retry import write_json,write_text

def select_tasks(rows, count=20):
    """Fixed seed, image diversity first; never receives method scores."""
    rows=sorted((r for r in rows if r['task_type']=='T3.1'),key=lambda r:r['id']);random.Random(20261116).shuffle(rows)
    chosen=[];seen=set();remaining=[]
    for row in rows:
        image=row['image_t1']
        if image not in seen and len(chosen)<count:
            chosen.append(row);seen.add(image)
        else:remaining.append(row)
    chosen += remaining[:max(0,count-len(chosen))]
    return [r['id'] for r in chosen]

def validate_frozen(path):
    root=Path(path);value=json.loads((root/'freeze.json').read_text())
    if value['approval']['status']!='approved' or not value['approval'].get('approval_reference'):raise ValueError('Explicit human approval required')
    rows=[json.loads(x) for x in (root/'tasks.jsonl').read_text().splitlines() if x]
    if len(rows)<20 or len({r['id'] for r in rows})!=len(rows):raise ValueError('At least 20 distinct frozen tasks required')
    if any(r['task_type']!='T3.1' for r in rows):raise ValueError('Online quota is exclusively T3.1-clean')
    if value['seeds']!=[1016,2027,3038]:raise ValueError('Three fixed seeds required')
    return value

def freeze(data,approval,selection,config,output):
    if (output/'freeze.json').exists():raise ValueError('Freeze exists; preserve existing manifest')
    approved=json.loads(approval.read_text());method=json.loads(config.read_text())
    if approved.get('status')!='approved' or not approved.get('approval_reference'):raise ValueError('Explicit human approval required')
    rows={r['id']:r for r in json.loads((data/'t3_1_clean_confirm.json').read_text())}
    if any(r['task_type']!='T3.1' or r['meta']['split']!='confirm' for r in rows.values()):raise ValueError('Require T3.1 confirm source')
    ids=select_tasks(list(rows.values()))
    if selection.exists() and json.loads(selection.read_text())!=ids:raise ValueError('Selection differs from seed-fixed image-diverse list')
    if len(ids)<20:raise ValueError('Require >=20 available tasks')
    output.mkdir(parents=True,exist_ok=True)
    result=[];mapping=[]
    for i,key in enumerate(ids):
        row=dict(rows[key]);opaque=f'n9_online_{i+1:03d}';mapping.append(dict(id=opaque,original_id=key,meta=row['meta']))
        row.update(id=opaque,meta={},review_status='approved');result.append(row)
        folder=output/'inputs'/opaque;folder.mkdir(parents=True,exist_ok=True)
        def alias(path,name):
            dest=folder/name
            if not dest.exists():shutil.copyfile(path,dest)
            return str(dest.absolute())
        for field in ('image_t1','image_t2','gt_mask','gt_ignore_mask','exemplar_mask'):
            if row.get(field):row[field]=alias(row[field],field+'.png')
        ignored=np.asarray(Image.open(row['exemplar_mask']))>0
        if row.get('gt_ignore_mask'):ignored|=np.asarray(Image.open(row['gt_ignore_mask']))>0
        ignore_path=folder/'evaluation_ignore.png'
        Image.fromarray(ignored.astype(np.uint8)*255).save(ignore_path)
        row['gt_ignore_mask']=str(ignore_path.absolute())
        row['instruction']='Find all other land-cover changes with the same directed transition as the marked example. The example is reference support, not an output target. Output a binary mask excluding the example itself.'
        if row.get('examples'):
            row['examples']=[dict(role=e['role'],**{k:alias(e[k],'example_'+k+'.png') for k in ('image_t1','image_t2','mask')}) for e in row['examples']]
    for kind in {r['task_type'] for r in result}:
        assert method[kind]['rank'] in ('delta','joint') and method[kind]['k'] in (5,8)
    output.mkdir(parents=True,exist_ok=True)
    write_text(output/'tasks.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    write_json(output/'scoring_mapping.json',mapping);write_json(output/'inference_config.json',method)
    write_json(output/'freeze.json',dict(approval=approved,seeds=[1016,2027,3038],tasks=len(result),selection_ids=ids,
        independent_query_images=len({rows[k]['image_t1'] for k in ids}),selection_seed=20261116,
        selection_policy='Seed-shuffle sorted IDs; first distinct query images, then remaining transitions only if fewer than 20 images; no scores',arms=['N9-Agent'],fixed_comparators=['Baseline-Fixed','N9-Fixed'],configuration=method))
    validate_frozen(output)

if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Manifest input copy/ignore generation requires Slurm')
    p=argparse.ArgumentParser()
    for key in ('data','approval','selection','config','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();freeze(a.data,a.approval,a.selection,a.config,a.output)
