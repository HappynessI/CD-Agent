"""Offline whole-instance spatial audit; never publishes evaluation tasks."""
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from agent_baseline.evaluation import load_reference

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'results/agent_baseline/n9_missed_20261116'
OUT = ROOT/'results/agent_baseline/task_reconstruct_audit/t44'
ROAD_STAGE=os.environ.get('T44_ROAD_STAGE')=='1'
if ROAD_STAGE: OUT=OUT/'road_stage'
RULE = dict(buffer_pixels=32, minimum_overlap_fraction=.1, instance_min_pixels=150,
            target_min_pixels=400, anchor_min_pixels=400, semantic_probability=.55,
            temporal_rule='same semantic class argmax and probability >= .55 in both dates',
            topology='Euclidean buffer; retain entire filled instance, no clipping')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)

def mask(path, value):
    Image.fromarray(value.astype('uint8')*255).save(path)
    return str(path)

def components(binary, minimum):
    labels, n = ndi.label(binary, np.ones((3,3)))
    result=[]
    for i in range(1,n+1):
        raw=labels==i
        filled=ndi.binary_fill_holes(raw)
        if filled.sum()>=minimum:
            result.append((filled, int((filled & ~raw).sum())))
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assets=json.loads(((OUT if ROAD_STAGE else SOURCE)/'asset_manifest.json').read_text())
    evaluation={(r['group'],r['stem']):r['task'] for r in json.loads((SOURCE/'evaluation_manifest.json').read_text())}
    scenes={}
    for row in assets:
        dataset=row['group'].split('_')[0]
        if dataset not in ('second','whu'): continue
        scenes.setdefault((dataset,row['stem']),[]).append(row)
    candidates=[]; reasons=Counter(); scanned=Counter()
    for (dataset,stem),rows in sorted(scenes.items()):
        row=rows[0]; scanned[dataset]+=1
        checkpoint=OUT/'scan'/f'{dataset}_{stem}.json'
        if checkpoint.exists():
            data=json.loads(checkpoint.read_text()); candidates.extend(data['candidates']);reasons.update(data['reasons']);continue
        local=Counter(); truths=[]; sources=[]
        for r in rows:
            task=evaluation[r['group'],stem]
            gt,_=load_reference(task['gt_mask'],task.get('gt_ignore_mask'))
            truths.append(gt);sources.append(task['gt_mask'])
        truth=np.logical_or.reduce(truths)
        buildings=components(truth,150)
        options=[]
        if len(buildings)<2: local['fewer_than_two_building_instances']+=1
        else:
            probabilities=[]
            for t in ('t1','t2'):
                with np.load(row['semantic'][t],allow_pickle=False) as z: probabilities.append(z['data'].astype('float32'))
            # The six-class model has no road class. Never relabel generic ground as road.
            for class_id,name,cn in [(4,'water','水体'),(6,'playground','运动场')]+([(7,'road','道路')] if ROAD_STAGE else []):
                stable=np.logical_and.reduce([(p.argmax(0)==class_id)&(p[class_id]>=.55) for p in probabilities])
                anchors=components(stable,400)
                if not anchors: local['no_stable_'+name+'_anchor']+=1
                for ai,(anchor,_) in enumerate(anchors):
                    if name=='road':
                        yy,xx=np.where(anchor)
                        if max((yy.max()-yy.min()+1)/anchor.shape[0],(xx.max()-xx.min()+1)/anchor.shape[1])<.6:
                            local['road_not_salient_spanning']+=1;continue
                    buffer=ndi.distance_transform_edt(~anchor)<=32
                    hits=[i for i,(b,_) in enumerate(buildings) if (b & buffer).sum()/b.sum()>=.1]
                    if not hits or len(hits)==len(buildings):
                        local['no_target_or_no_distractor']+=1;continue
                    target=np.logical_or.reduce([buildings[i][0] for i in hits])
                    if target.sum()<400: local['target_below_400']+=1;continue
                    excluded=[i for i in range(len(buildings)) if i not in hits]
                    distractor=np.logical_or.reduce([buildings[i][0] for i in excluded])
                    options.append(dict(id=f'T44_{dataset}_{stem}_{name}_{ai}', dataset=dataset,image_id=stem,
                        image_t1=row['image_t1'],image_t2=row['image_t2'],anchor_class=name,anchor_class_cn=cn,anchor_index=ai,
                        target_instance_ids=hits,distractor_instance_ids=excluded,target_instances=len(hits),distractor_instances=len(excluded),
                        target_pixels=int(target.sum()),distractor_pixels=int(distractor.sum()),anchor_pixels=int(anchor.sum()),
                        filled_pixels=sum(x[1] for x in buildings),source_gt=sources,source_semantic=row['semantic'],rule=RULE,
                        instruction=f'找出紧邻{cn}周边的建筑变化。',status='pending_review',
                        anchor_source='SegEarth same-class high-confidence intersection of both dates; visual approval required',
                        legacy_relation='Replaces directional/new-class R3 examples; no old task approved automatically'))
        write(checkpoint,dict(candidates=options,reasons=dict(local)))
        candidates.extend(options);reasons.update(local)
    # Prefer substantial selected and excluded populations, and large stable visible anchors.
    ranked=sorted(candidates,key=lambda r:(-min(r['target_pixels'],r['distractor_pixels']),-r['anchor_pixels'],r['id']))
    selected=[];used=set()
    for item in ranked:
        key=item['dataset'],item['image_id']
        if key in used:continue
        used.add(key);selected.append(item)
        if len(selected)==3:break
    for item in selected:
        p=OUT/'masks'/item['id'];p.mkdir(parents=True,exist_ok=True)
        truths=[load_reference(g)[0] for g in item['source_gt']]
        buildings=components(np.logical_or.reduce(truths),150)
        probs=[]
        for t in ('t1','t2'):
            with np.load(item['source_semantic'][t],allow_pickle=False) as z:probs.append(z['data'])
        ci={'water':4,'playground':6,'road':7}[item['anchor_class']]
        stable=np.logical_and.reduce([(v.argmax(0)==ci)&(v[ci]>=.55) for v in probs])
        anchor=components(stable,400)[item['anchor_index']][0]
        values=dict(anchor_mask=anchor,buffer_mask=ndi.distance_transform_edt(~anchor)<=32,
            target_mask=np.logical_or.reduce([buildings[i][0] for i in item['target_instance_ids']]),
            distractor_mask=np.logical_or.reduce([buildings[i][0] for i in item['distractor_instance_ids']]))
        item['evaluation']={k:mask(p/(k+'.png'),v) for k,v in values.items()}
        item['inference_input']={k:item[k] for k in ('image_t1','image_t2','instruction')}
    write(OUT/'selected.json',selected)
    write(OUT/'pool_summary.json',dict(scanned_images=dict(scanned),qualified_configurations=len(candidates),
        qualified_images=len({(c['dataset'],c['image_id']) for c in candidates}),
        by_dataset=dict(Counter(c['dataset'] for c in candidates)),rejections=dict(reasons),
        selected=len(selected),gap=3-len(selected),rule=RULE,api_cost_cny=0,
        limitation=('Explicit road-aware local SegEarth; stable anchors require visual review' if ROAD_STAGE else 'No road channel available; only actual water/playground semantic anchors used; visual review pending')))
    print(json.dumps(dict(stage='t44',status='complete',selected=len(selected),pool=len(candidates),cost_cny=0)))
if __name__=='__main__':main()
