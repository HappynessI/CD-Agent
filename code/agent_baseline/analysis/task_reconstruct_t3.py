"""Pending-review T3.1 homogeneous object clusters, real dev images, no API.

Only the three selected examples and two companion masks per example are saved.
Every original direction-component remains in full/rem GT and per-instance audit.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from PIL import Image
from agent_baseline.tools_t13 import Features,pool_features
from agent_baseline.whu_gt import load_truth
from .n8_missed_assets import ROOT,RUN,WORK

OUT=ROOT/'CD-Agent/results/agent_baseline/task_reconstruct_audit/t3'
POLICY=dict(example_area=[200,5000],max_original_hole_ratio=.05,min_rotated_rectangle_fill=.45,
    single_date_dino_cosine=.75,rgb_mean_euclidean_normalized_max=.15,
    rgb_normalization='euclidean RGB mean distance / (sqrt(3)*255)',
    minimum_other_matching_instances=2,min_full_remaining_pixels=400,
    preferred_cluster_area_fraction=.85,connectivity=8,
    datasets='WHU official dev60 and SECOND frozen dev60 only; no report/test',
    full_gt='union of ALL hole-filled same-category same-direction components, including nonmatches',
    labels_role='offline construction/review only',review_status='pending_review')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def raster(path):
    with Image.open(path) as im:return np.asarray(im)
def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def geometry(mask):
    filled=ndi.binary_fill_holes(mask);raw_area=int(mask.sum());area=int(filled.sum())
    yy,xx=np.nonzero(filled);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    boundary=filled&~ndi.binary_erosion(filled)
    y,x=np.nonzero(boundary);points=np.stack([x,y],axis=1).astype(float)
    corners=np.concatenate([points+offset for offset in ((0,0),(1,0),(0,1),(1,1))])
    hull=corners[ConvexHull(corners).vertices]
    edges=np.roll(hull,-1,axis=0)-hull
    angles=np.unique(np.mod(np.arctan2(edges[:,1],edges[:,0]),np.pi/2))
    rectangle=float((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
    for angle in angles:
        c,s=np.cos(angle),np.sin(angle);rotated=hull@np.array([[c,-s],[s,c]])
        rectangle=min(rectangle,float(np.ptp(rotated[:,0])*np.ptp(rotated[:,1])))
    return filled,dict(raw_area=raw_area,area=area,bbox=bbox,hole_ratio=(area-raw_area)/area,
                       filled_pixels=area-raw_area,compactness=min(1.,area/rectangle),min_rectangle_area=rectangle)


def component_masks(truth):
    labels,n=ndi.label(truth,np.ones((3,3)))
    return [labels==i for i in range(1,n+1)]


def direction_truths(dataset,task):
    root=Path(task['image_t1']).parent.parent;name=Path(task['image_t1']).name
    if dataset=='whu':
        official=load_truth(root,name)
        return [('building','build','t2',official['new']),('building','demolish','t1',official['demolished'])]
    a=raster(root/'slabel1'/name);b=raster(root/'slabel2'/name)
    if np.any((a==0)!=(b==0)):raise ValueError('SECOND unchanged-label support differs')
    changed=(a!=b)&(a>0)&(b>0)
    return [(category,direction,t,changed&((b==code) if direction=='build' else (a==code)))
            for category,code in (('building',5),('playground',6)) for direction,t in (('build','t2'),('demolish','t1'))]


def tasks():
    result=[]
    for dataset,path,names in (
        ('whu',RUN/'whu/tasks/T1.1xC1_appeared_dev.jsonl',RUN/'whu/names_dev60.txt'),
        ('second',RUN/'second_dev/second/tasks/T1.1xC1_appeared.jsonl',RUN/'second_dev/names.txt')):
        allowed={Path(x).stem for x in names.read_text().split()}
        rows=jsonl(path);lookup={Path(r['image_t1']).stem:r for r in rows}
        assert len(allowed)==60 and set(lookup)==allowed
        for stem in sorted(allowed):result.append((dataset,stem,lookup[stem]))
    return result


def feature_path(dataset,stem,t):return WORK/'gpu_cache/dynamic'/dataset/stem/f'extract_features_{t}.npz'


def analyze_instances(raw_masks,rgb,feature):
    masks=[];records=[];vectors=[]
    for index,raw in enumerate(raw_masks,1):
        mask,g=geometry(raw);masks.append(mask)
        vector=pool_features(feature,np.flatnonzero(mask));norm=float(np.linalg.norm(vector))
        vectors.append(vector/max(norm,1e-12))
        records.append(dict(id=index,**g,rgb_mean=rgb[mask].mean(axis=0).tolist()))
    return masks,records,np.asarray(vectors)


def best_example(masks,records,vectors):
    if not masks:return None,Counter(empty_direction=1),0
    full=np.logical_or.reduce(masks);total=int(full.sum());failures=Counter();best=None;eligible=0
    colors=np.asarray([r['rgb_mean'] for r in records]);areas=np.asarray([r['area'] for r in records])
    for i,row in enumerate(records):
        reasons=[]
        if not 200<=row['area']<=5000:reasons.append('example_area')
        if row['hole_ratio']>.05:reasons.append('example_holes')
        if row['compactness']<.45:reasons.append('example_compactness')
        remaining=int((full&~masks[i]).sum())
        if remaining<400:reasons.append('remaining_pixels')
        if reasons:
            failures.update(reasons);continue
        cos=np.clip(vectors@vectors[i],-1,1)
        distance=np.linalg.norm(colors-colors[i],axis=1)/(np.sqrt(3)*255)
        matches=(cos>=.75)&(distance<=.15)
        # Other members also need intact, non-ring object geometry; unlike E,
        # they are not restricted to E's 200..5000 size interval.
        intact=np.asarray([r['hole_ratio']<=.05 and r['compactness']>=.45 for r in records])
        matches &= intact;matches[i]=True
        others=[j for j in range(len(records)) if matches[j] and j!=i]
        if len(others)<2:
            failures['fewer_than_two_other_matches']+=1;continue
        cluster=np.zeros_like(full)
        for j in np.flatnonzero(matches):cluster|=masks[j]
        fraction=float(cluster.sum()/total);eligible+=1
        instances=[dict(r,cosine=float(cos[j]),rgb_distance=float(distance[j]),is_example=j==i,
                        is_match=bool(matches[j]),intact_geometry=bool(intact[j])) for j,r in enumerate(records)]
        key=(fraction>=.85,fraction,int((cluster&~masks[i]).sum()),row['compactness'],row['area'],-row['id'])
        if best is None or key>best[0]:
            best=(key,dict(example_id=row['id'],example_area=row['area'],bbox=row['bbox'],hole_ratio=row['hole_ratio'],
                compactness=row['compactness'],cluster_area_fraction=fraction,full_pixels=total,
                remaining_pixels=remaining,remaining_instances=len(records)-1,matching_other_instances=len(others),
                matching_other_pixels=int((cluster&~masks[i]).sum()),instances=instances,
                sample_instance_ids=[records[j]['id'] for j in sorted(others,key=lambda j:(-records[j]['area'],-cos[j],records[j]['id']))[:2]]))
    return (best[1] if best else None),failures,eligible


def quality(row):
    return (row['cluster_area_fraction']>=.85,row['cluster_area_fraction'],row['matching_other_pixels'],row['compactness'])


def select_three(candidates):
    ordered=sorted(candidates,key=lambda r:(tuple(-float(x) for x in quality(r)),r['dataset'],r['stem'],r['category'],r['direction']))
    selected=[];used=set()
    # Prefer the clean >=85% stratum; within it cover both datasets if available.
    clean=[r for r in ordered if r['cluster_area_fraction']>=.85]
    pool=clean if len({(r['dataset'],r['stem']) for r in clean})>=3 else ordered
    for dataset in ('whu','second'):
        row=next((r for r in pool if r['dataset']==dataset and (r['dataset'],r['stem']) not in used),None)
        if row is not None:selected.append(row);used.add((row['dataset'],row['stem']))
    for row in pool:
        if len(selected)==3:break
        if (row['dataset'],row['stem']) not in used:selected.append(row);used.add((row['dataset'],row['stem']))
    return selected[:3]


def export(row,task,out,number):
    raw=next(m for category,direction,t,m in direction_truths(row['dataset'],task) if category==row['category'] and direction==row['direction'])
    masks=[ndi.binary_fill_holes(m) for m in component_masks(raw)]
    example=masks[row['example_id']-1];full=np.logical_or.reduce(masks);remaining=full&~example
    folder=out/'samples'/f't3_{number:02d}';folder.mkdir(parents=True,exist_ok=True)
    def save(name,mask):
        path=folder/(name+'.png');Image.fromarray(mask.astype(np.uint8)*255).save(path);return str(path)
    result=dict(row,id=f't3_{number:02d}',image_id=row['stem'],status='pending_review',image_t1=task['image_t1'],image_t2=task['image_t2'],
        example_mask=save('example',example),gt_full=save('gt_full',full),gt_rem=save('gt_rem',remaining),
        review_status='pending_review',task_type='T3.1',
        instruction='请参照图中标出的完整示例，找出图内其他同类、同款外观且变化方式相同的目标，输出二值变化掩码。')
    for item in result['instances']:
        item['mask']=result['example_mask'] if item['is_example'] else (save('instance_'+str(item['id']),masks[item['id']-1]) if item['id'] in row['sample_instance_ids'] else None)
    result['inference_inputs']={k:result[k] for k in ('image_t1','image_t2','example_mask','instruction')}
    result['audit_fields']={k:result[k] for k in ('dataset','stem','direction','category','example_time','gt_full','gt_rem','bbox','example_area','hole_ratio','compactness','cluster_area_fraction','remaining_pixels','remaining_instances','instances')}
    result['source_labels']=dict(dataset_root=str(Path(task['image_t1']).parent.parent),
        definition='official WHU change intersect reliable semantic direction' if row['dataset']=='whu' else 'SECOND changed semantic labels, class5 building/class6 playground',
        processing='8-connected complete direction components; holes filled; no bbox/grid cutting; all components retained in full GT',
        original_task_id=task['id'],label_revision='new pending-review derived masks, never overwrites existing GT')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('All construction/feature analysis must use Slurm')
    a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic();items=tasks();missing=[]
    for dataset,stem,task in items:
        for t in ('t1','t2'):
            path=feature_path(dataset,stem,t)
            if not path.is_file():missing.append(dict(dataset=dataset,stem=stem,time=t,image=task['image_'+t],feature_path=str(path)))
    write(a.out/'policy.json',POLICY)
    if missing:
        write(a.out/'pool_summary.json',dict(status='needs_feature_fill',missing_features=missing,api_cost_cny=0,review_status='pending_review'))
        raise RuntimeError('Missing DINO assets listed; fill via Slurm, do not skip images')
    candidates=[];groups={};lookup={};counts=Counter()
    for dataset,stem,task in items:
        lookup[dataset,stem]=task
        checkpoint=a.out/'checkpoints'/dataset/(stem+'.json')
        identity=dict(dataset=dataset,stem=stem,image_t1=task['image_t1'],image_t2=task['image_t2'],policy=POLICY)
        if checkpoint.exists():
            saved=read(checkpoint)
            if saved['identity']!=identity:raise ValueError('Checkpoint inputs/policy differ; preserve old output and use a new directory')
            image_candidates=saved['candidates'];image_groups=saved['groups']
            counts['images_reused']+=1
        else:
            image_candidates=[];image_groups={}
            rgb={t:raster(task['image_'+t])[...,:3] for t in ('t1','t2')};features={}
            for t in ('t1','t2'):
                with np.load(feature_path(dataset,stem,t),allow_pickle=False) as z:features[t]=Features(z['data'],rgb[t].shape[:2],t)
            for category,direction,t,truth in direction_truths(dataset,task):
                key=f'{dataset}/{category}/{direction}'
                masks,records,vectors=analyze_instances(component_masks(truth),rgb[t],features[t])
                best,failures,n=best_example(masks,records,vectors)
                image_groups[key]=dict(images=1,original_instances=len(records),eligible_examples=n,
                                       eligible_images=int(best is not None),failures=dict(failures))
                if best:
                    image_candidates.append(dict(best,dataset=dataset,stem=stem,category=category,direction=direction,example_time=t))
            write(checkpoint,dict(status='complete',identity=identity,candidates=image_candidates,groups=image_groups))
            counts['images_computed_this_attempt']+=1
        candidates.extend(image_candidates)
        for key,value in image_groups.items():
            g=groups.setdefault(key,dict(images=0,original_instances=0,eligible_examples=0,eligible_images=0,failures=Counter()))
            for field in ('images','original_instances','eligible_examples','eligible_images'):g[field]+=value[field]
            g['failures'].update(value['failures'])
        counts['images_processed']+=1
        write(a.out/'progress.json',dict(**counts,eligible_image_directions=len(candidates),elapsed_seconds=time.monotonic()-started))
    selected=select_three(candidates)
    exports=[export(row,lookup[row['dataset'],row['stem']],a.out,i+1) for i,row in enumerate(selected)]
    write(a.out/'selected.json',exports)
    write(a.out/'pool_summary.json',dict(status='complete_pending_review',policy=POLICY,images_scanned=len(items),groups=groups,
        eligible_image_direction_candidates=len(candidates),eligible_physical_images=len({(r['dataset'],r['stem']) for r in candidates}),
        candidates_at_least_85_percent=sum(r['cluster_area_fraction']>=.85 for r in candidates),selected=len(exports),
        selection='Prefer >=85% area homogeneity; cover WHU/SECOND if available; distinct images; then larger homogeneous remaining support and compactness',
        candidate_summaries=[{k:r[k] for k in ('dataset','stem','category','direction','example_id','example_area','cluster_area_fraction','matching_other_instances','remaining_pixels')} for r in candidates],
        shortage_reason=None if len(exports)==3 else 'Fewer than three distinct qualifying physical images; no threshold relaxation or new split supplementation',
        api_cost_cny=0,elapsed_seconds=time.monotonic()-started,checkpoint_counts=dict(counts),slurm_job=os.environ['SLURM_JOB_ID'],review_status='pending_review',
        note='No new Agent run or formal evaluation. GT full/rem retain every same-category same-direction filled component, not only matched cluster. Non-selected masks are not written.'))
    print(json.dumps(dict(status='complete_pending_review',selected=len(exports),eligible=len(candidates),api_cost_cny=0)),flush=True)

if __name__=='__main__':main()
