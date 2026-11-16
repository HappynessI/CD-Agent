"""Versioned whole-GT interior audit; never removes border instances from GT."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
import numpy as np
from agent_baseline.tools_t13 import Features
from . import task_reconstruct_t3 as original

OUT=original.OUT/'revision_interior_examples'
POLICY=dict(original.POLICY,revision='whole_gt_interior_v1',
    border_rule='Reject image/category/direction task if ANY full-GT instance touches an image boundary; retain all instances in audit/full GT',
    fallback='none; fewer than three distinct strict images reported without relaxation')


def touches_border(mask):
    return bool(mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any())


def strict_example(masks,records,vectors):
    annotated=[dict(r,touches_border=touches_border(m)) for r,m in zip(records,masks)]
    border=sum(r['touches_border'] for r in annotated)
    if border:
        return None,Counter(whole_gt_has_border_instances=1,border_instances=border),0,annotated
    best,failures,n=original.best_example(masks,annotated,vectors)
    if best:best.update(whole_gt_interior=True,border_instances=0)
    return best,failures,n,annotated


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    items=original.tasks();groups={};candidates=[];lookup={};counts=Counter();missing=[]
    for dataset,stem,task in items:
        for t in ('t1','t2'):
            path=original.feature_path(dataset,stem,t)
            if not path.is_file():missing.append(dict(image=task['image_'+t],feature_path=str(path)))
    original.write(a.out/'policy.json',POLICY)
    if missing:
        original.write(a.out/'pool_summary.json',dict(status='needs_feature_fill',missing_features=missing,policy=POLICY))
        raise RuntimeError('Missing features; no image skipped')
    for dataset,stem,task in items:
        lookup[dataset,stem]=task;path=a.out/'checkpoints'/dataset/(stem+'.json')
        identity=dict(dataset=dataset,stem=stem,image_t1=task['image_t1'],image_t2=task['image_t2'],policy=POLICY)
        if path.exists():
            saved=original.read(path)
            if saved['identity']!=identity:raise ValueError('Checkpoint identity differs')
            counts['images_reused']+=1
        else:
            rgb={t:original.raster(task['image_'+t])[...,:3] for t in ('t1','t2')};features={}
            for t in ('t1','t2'):
                with np.load(original.feature_path(dataset,stem,t),allow_pickle=False) as z:features[t]=Features(z['data'],rgb[t].shape[:2],t)
            saved=dict(identity=identity,candidates=[],groups={},instance_audit={})
            for category,direction,t,truth in original.direction_truths(dataset,task):
                key=f'{dataset}/{category}/{direction}'
                masks,records,vectors=original.analyze_instances(original.component_masks(truth),rgb[t],features[t])
                best,failures,n,annotated=strict_example(masks,records,vectors)
                border=sum(r['touches_border'] for r in annotated)
                saved['instance_audit'][key]=annotated
                saved['groups'][key]=dict(images=1,original_instances=len(records),border_instances=border,
                    whole_gt_interior_images=int(bool(records) and border==0),eligible_examples=n,
                    eligible_images=int(best is not None),failures=dict(failures))
                if best:saved['candidates'].append(dict(best,dataset=dataset,stem=stem,category=category,direction=direction,example_time=t))
            original.write(path,saved);counts['images_computed_this_attempt']+=1
        candidates.extend(saved['candidates'])
        for key,value in saved['groups'].items():
            g=groups.setdefault(key,dict(images=0,original_instances=0,border_instances=0,whole_gt_interior_images=0,
                                         eligible_examples=0,eligible_images=0,failures=Counter()))
            for field in ('images','original_instances','border_instances','whole_gt_interior_images','eligible_examples','eligible_images'):g[field]+=value[field]
            g['failures'].update(value['failures'])
        counts['images_processed']+=1
        original.write(a.out/'progress.json',dict(**counts,strict_candidates=len(candidates)))
    chosen=original.select_three(candidates)
    selected=[original.export(row,lookup[row['dataset'],row['stem']],a.out,i+1) for i,row in enumerate(chosen)]
    for row in selected:
        row['selection_revision']='whole_gt_interior_v1'
        row['audit_fields'].update(whole_gt_interior=True,border_instances=0)
    original.write(a.out/'selected.json',selected)
    distinct=len({(r['dataset'],r['stem']) for r in candidates})
    original.write(a.out/'pool_summary.json',dict(status='complete_pending_review',policy=POLICY,groups=groups,
        images_scanned=len(items),strict_eligible_candidates=len(candidates),strict_eligible_physical_images=distinct,
        candidates_at_least_85_percent=sum(r['cluster_area_fraction']>=.85 for r in candidates),selected=len(selected),
        shortage_reason=None if len(selected)==3 else 'Strict whole-GT interior pool has fewer than three distinct valid images; no relaxation or removal of border instances',
        candidate_summaries=[{k:r[k] for k in ('dataset','stem','category','direction','example_id','cluster_area_fraction','remaining_pixels','whole_gt_interior')} for r in candidates],
        original_results_preserved=str(original.OUT),checkpoint_counts=dict(counts),api_cost_cny=0,
        elapsed_seconds=time.monotonic()-started,slurm_job=os.environ['SLURM_JOB_ID'],review_status='pending_review'))
    print(dict(strict_images=distinct,selected=len(selected),api_cost_cny=0),flush=True)

if __name__=='__main__':main()
