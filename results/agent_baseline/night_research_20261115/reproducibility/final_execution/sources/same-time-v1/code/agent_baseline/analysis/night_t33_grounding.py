"""Existing cross-image T3.3 dev audit and fixed 12-query grounding pilot."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import numpy as np
from scipy import ndimage as ndi
from agent_baseline.exemplar_semantics import request,canonical,CLASSES,PARSE_SYSTEM,PARSE_QUESTION,single_date_views
from .io_retry import read_image,read_text,write_json,exists
from .t3_routes import pair_id,group
from .t3_exemplar_geometry_audit import geometry
from .night_t3_grounding import views,SPLIT_SEED,API_SEED,summarize_parses


def selection(rows):
    by_query=defaultdict(list)
    for row in rows:by_query[Path(row['image_t1']).name].append(row)
    rng=random.Random(SPLIT_SEED)
    chosen=[rng.choice(sorted(by_query[name],key=lambda r:r['id'])) for name in sorted(by_query)]
    # External example source scenes stay together as well as query images.
    by_source=defaultdict(list)
    for row in chosen:by_source[row['meta']['example_provenance'][0]['name']].append(row)
    source_names=sorted(by_source);rng.shuffle(source_names);counts={'tune':0,'confirm':0};split={}
    for source in source_names:
        part=min(counts,key=lambda key:counts[key]);items=by_source[source]
        for row in items:split[row['id']]=part
        counts[part]+=len(items)
    pilot=[]
    for part in ('tune','confirm'):
        items=sorted([row for row in chosen if split[row['id']]==part],key=lambda row:row['id'])
        rng.shuffle(items);pilot.extend(items[:6])
    return chosen,pilot,split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--variant',choices=['audit','B0','B1','B2'],required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Execute through Slurm')
    manifest=a.run/'t3x/tasks/dev/tasks.jsonl'
    rows=[json.loads(line) for line in read_text(manifest).splitlines() if line.strip()]
    rows=[row for row in rows if row['task_type']=='T3.3']
    dev_names=set(read_text(a.run/'second_dev/names.txt').split())
    if {Path(row['image_t1']).name for row in rows}!=dev_names:raise ValueError('T3.3 queries must be exactly original dev images')
    chosen,pilot,split=selection(rows);out=a.night/'t33';old=json.loads(read_text(a.run/'t3x/dev.json'))
    if not old['complete']:raise ValueError('Historical T3.3 dev baseline is incomplete')
    if not exists(out/'audit.json'):
        records=[]
        for row in rows:
            example=row['examples'][0];ex=read_image(example['mask'])>0;truth=read_image(row['gt_mask'])>0
            _,components=ndi.label(truth,np.ones((3,3),bool))
            records.append(dict(id=row['id'],group=group(row),query_image=row['image_t1'],
                example_source=row['meta']['example_provenance'][0]['name'],query_gt_pixels=int(truth.sum()),query_gt_components=int(components),
                external_example_geometry=geometry(ex,np.zeros_like(ex)),
                historical_fixed_counts=old['per_task'][row['id']]['counts'][str(old['selected']['T3.3'])]))
        write_json(out/'audit.json',dict(tasks=len(rows),records=records,selected_query_tasks=[r['id'] for r in chosen],
            pilot_ids=[r['id'] for r in pilot],split=split,seed=SPLIT_SEED,
            split_scope='one preselected task per query; shared external source scenes kept in same split; old development data already exposed',
            scoring='external example has separate coordinates: do not subtract its mask from query GT',
            original_manifest=str(manifest),T32_T34='unchanged and not run'))
    if a.variant=='audit':return
    records=[]
    for row in pilot:
        example=row['examples'][0];ex=read_image(example['mask'])>0
        images=[read_image(example[k],rgb=True) for k in ('image_t1','image_t2')];answers=[]
        for time in (0,1):
            if a.variant=='B0':pictures=single_date_views(images[time],ex);system=PARSE_SYSTEM;question=PARSE_QUESTION
            else:
                pictures=views(images,ex,time,a.variant)
                system=('Identify only bright supported target-date pixels. Darkened outside and enclosed holes are excluded. '
                    'The raw paired-date reference images, if present, are context only. Do not classify objects inside excluded holes.')
                question=f'Target date T{time+1}. '+PARSE_QUESTION
            answers.append(request(out/a.variant/'tasks'/row['id']/f't{time+1}',system,question,pictures,API_SEED))
        reverse={name:i for i,name in CLASSES.items()};pair=[reverse.get(canonical(r['parsed'].get('class'))) for r in answers];truth=list(pair_id(row))
        records.append(dict(id=row['id'],split=split[row['id']],group=group(row),has_holes=bool(geometry(ex,np.zeros_like(ex))['has_holes']),
            pair=pair,true_pair=truth,before_agrees=pair[0]==truth[0],after_agrees=pair[1]==truth[1],pair_agrees=pair==truth,
            cost_cny=sum(v['cost_cny'] for v in answers)))
        write_json(out/a.variant/'summary.json',dict(tasks=len(records),records=records,summary=summarize_parses(records),
            cost_cny=sum(r['cost_cny'] for r in records),intervention=('original contour prompt, new external-example requests' if a.variant=='B0' else 'view plus support prompt; B2 also has cross-date reference information')))
        print('T3.3',a.variant,len(records),row['id'],flush=True)


if __name__=='__main__':main()
