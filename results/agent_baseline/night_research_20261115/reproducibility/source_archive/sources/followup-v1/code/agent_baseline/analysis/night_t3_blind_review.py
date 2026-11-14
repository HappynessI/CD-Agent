"""Fixed, correctness-blind B1/B2 review package; image work only in Slurm."""
import argparse
import html
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image
from render import crop_box,SIZE
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import load_dev,group,GROUPS
from .t3_routes_match import bbox

SEED=2026111401


def choose(rows,split,per_group=3):
    rng=random.Random(SEED);chosen=[]
    for part in ('tune','confirm'):
        for name in GROUPS:
            eligible=sorted([r for r in rows if split[r['id']]==part and group(r)==name],key=lambda r:r['id'])
            rng.shuffle(eligible);chosen.extend(eligible[:per_group])
    rng.shuffle(chosen)
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--key',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Build image package through Slurm')
    if exists(a.output/'index.html'):raise ValueError('Review package exists; keep annotations and use a new output path')
    split=json.loads(read_text(a.night/'audit.json'))['split']
    rows=choose(load_dev(a.run),split);mapping=[];sections=[];rng=random.Random(SEED+1)
    for number,row in enumerate(rows,1):
        case=f'case_{number:02d}';folder=a.output/case;folder.mkdir(parents=True,exist_ok=True)
        mask=read_image(row['exemplar_mask'])>0;box=bbox(mask)
        shutil.copyfile(row['exemplar_mask'],folder/'support_native.png')
        supports=[]
        for label,kwargs in [('close',{}),('wide',dict(scale=10,min_side=256))]:
            l,t,r,b=crop_box(box,mask.shape,**kwargs)
            Image.fromarray(mask[t:b,l:r].astype(np.uint8)*255).resize((SIZE,SIZE),Image.Resampling.NEAREST).save(folder/f'support_{label}.png')
            supports.append(f'<figure><img src="{case}/support_{label}.png"><figcaption>Binary support {label}: white included, black excluded</figcaption></figure>')
        variants=['B1','B2'];rng.shuffle(variants);panels=[];variant_key={}
        for label,variant in zip(('A','B'),variants):
            variant_key[label]=variant;parts=[]
            for time in ('t1','t2'):
                source=a.night/variant/'tasks'/row['id']/time
                metadata=json.loads(read_text(source/'input.json'));answer=json.loads(read_text(source/'answer.json'))
                destination=folder/label/time;destination.mkdir(parents=True,exist_ok=True)
                images=[]
                for index,image in enumerate(metadata['images']):
                    filename=f'input_{index:02d}.png';shutil.copyfile(image['path'],destination/filename)
                    images.append(dict(label=image['label'],path=filename))
                # Do not copy original paths/IDs into the reviewer-facing package.
                write_json(destination/'input.json',dict(system=metadata['system'],question=metadata['question'],images=images))
                write_json(destination/'prediction.json',dict(parsed=answer['parsed'],raw=answer['raw']))
                figures=''.join(f'<figure><img loading="lazy" src="{case}/{label}/{time}/{img["path"]}"><figcaption>{html.escape(img["label"])}</figcaption></figure>' for img in images)
                parts.append(f'<h4>{time}</h4><pre>{html.escape(json.dumps(answer["parsed"],ensure_ascii=False,indent=2))}</pre><div class="images">{figures}</div>')
            panels.append(f'<h3>Condition {label}</h3>'+''.join(parts))
        mapping.append(dict(case=case,task_id=row['id'],split=split[row['id']],group=group(row),variants=variant_key,
            exemplar_path=row['exemplar_mask'],selection='fixed seed within split and target/surface group; no model correctness used'))
        sections.append(f'<section><h2>{case}</h2><div class="images">'+''.join(supports)+'</div>'+''.join(panels)+
            '<p>Review: Does the prediction describe white support or excluded holes/background? Record visible before/after states, support-grounding evidence, and uncertainty. Do not infer correctness from a class label alone.</p></section>')
    write_json(a.key,dict(seed=SEED,per_group=3,cases=mapping))
    write_json(a.output/'review_template.json',[dict(case=r['case'],A=dict(grounding='',visible_before='',visible_after='',uncertainty=''),B=dict(grounding='',visible_before='',visible_after='',uncertainty='')) for r in mapping])
    write_text(a.output/'index.html','<!doctype html><meta charset="utf-8"><title>Anonymous support review</title><style>body{font-family:system-ui;max-width:1400px;margin:24px auto}section{border-top:2px solid #555;padding:20px}figure{margin:8px}img{width:300px;max-width:100%}.images{display:flex;flex-wrap:wrap}figcaption{max-width:300px;font-size:12px}pre{white-space:pre-wrap;background:#eee;padding:12px}</style><h1>Anonymous B1/B2 support review</h1><p>Cases were fixed without consulting prediction correctness. No task ID or GT label is shown. Conditions A/B are randomized separately for each case; number and type of inputs remain visible. Binary support is an audit aid and was not an extra model input.</p>'+''.join(sections))


if __name__=='__main__':main()
