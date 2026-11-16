"""N9 construction/QC only: fixed label geometry, no models or task scores."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from .io_retry import read_text,read_image,write_json,write_text,save_image
from .second_labels import DATASET,check_encoding,load_labels

SEED=20261116
PARTS=('tune','confirm','report')
FG=np.ones((3,3),bool)
BG=ndi.generate_binary_structure(2,1)
CN={1:'低矮植被',2:'非植被地表',3:'树木',4:'水体',5:'建筑',6:'运动场'}
POLICY=dict(seed=SEED,example_area=[64,4096],max_hole_ratio=.1,min_bbox_fill=.25,
            foreground_connectivity=8,hole_background_connectivity=4,min_query_pixels=64,min_query_component=32,
            external_source_counts=dict(tune=36,confirm=36,report=37),status='pending_review',
            rule='One task per query image/directed transition. Uniform seeded choice among geometry-eligible components; no model scores. Query labels unchanged; T3.1 scoring excludes E. No N9 evaluation before explicit human approval.')


def components(mask):
    labels,_=ndi.label(mask,FG)
    flat=labels.ravel();indices=np.flatnonzero(flat)
    if not len(indices):return []
    # One foreground gather; stable grouping preserves component label order
    # and original row-major pixel order without rescanning the entire grid.
    ordered=indices[np.argsort(flat[indices],kind='stable')]
    cuts=np.flatnonzero(np.diff(flat[ordered]))+1
    return list(np.split(ordered,cuts))


def geometry(mask):
    mask=np.asarray(mask,bool);area=int(mask.sum())
    if not area:return dict(area=0,hole_pixels=0,hole_ratio=0.,bbox=None,bbox_fill=0.,components=0)
    y,x=np.nonzero(mask);box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    filled=ndi.binary_fill_holes(mask,structure=BG);holes=int((filled&~mask).sum())
    return dict(area=area,hole_pixels=holes,hole_ratio=holes/int(filled.sum()),bbox=box,
                bbox_fill=area/((box[2]-box[0])*(box[3]-box[1])),components=int(ndi.label(mask,FG)[1]))


def example_reasons(g):
    reasons=[]
    if g['area']<64:reasons.append('example_area_below64')
    if g['area']>4096:reasons.append('example_area_above4096')
    if g['hole_ratio']>.1:reasons.append('example_hole_ratio_above0.1')
    if g['bbox_fill']<.25:reasons.append('example_bbox_fill_below0.25')
    if g['components']!=1:reasons.append('example_not_one_8connected_component')
    return reasons


def component_geometry(indices,shape):
    y,x=np.divmod(indices,shape[1]);x0,x1=int(x.min()),int(x.max()+1);y0,y1=int(y.min()),int(y.max()+1)
    crop=np.zeros((y1-y0,x1-x0),bool);crop[y-y0,x-x0]=True
    value=geometry(crop);value['bbox']=[x0,y0,x1,y1]
    return value


def support(mask):
    items=components(mask)
    return dict(pixels=int(mask.sum()),components=len(items),largest_component=max(map(len,items),default=0))


def query_reasons(s):
    return (['query_pixels_below64'] if s['pixels']<64 else [])+(['query_no_component32'] if s['largest_component']<32 else [])


def source_partition(names):
    names=sorted(names)
    if len(names)!=109 or len(set(names))!=109:raise ValueError('Require original 109 unique example-source images')
    random.Random(SEED).shuffle(names)
    return dict(tune=sorted(names[:36]),confirm=sorted(names[36:72]),report=sorted(names[72:]))


def choose(items,key):
    seed=SEED+sum((i+1)*ord(c) for i,c in enumerate(key))
    return random.Random(seed).choice(items)


def mask_from(indices,shape):
    mask=np.zeros(shape,bool);mask.ravel()[indices]=True;return mask


def transitions(a,b):
    changed=(a!=0)&(b!=0)&(a!=b)
    return [(int(x),int(y)) for x,y in np.unique(np.stack([a[changed],b[changed]],axis=1),axis=0)]


def rows(path):return [json.loads(s) for s in read_text(path).splitlines() if s.strip()]


def inventory(run,night):
    dev=read_text(run/'second_dev/names.txt').split();report=read_text(run/'second/second_val128.txt').split()
    if len(dev)!=60 or len(set(dev))!=60 or len(report)!=128 or len(set(report))!=128 or set(dev)&set(report):raise ValueError('Original query inventories changed')
    old=json.loads(read_text(night/'audit.json'))['split']
    by_image={k.rsplit('_',1)[0]+'.png':v for k,v in old.items()}
    if set(by_image)!=set(dev) or Counter(by_image.values())!=dict(tune=30,confirm=30):raise ValueError('Original query 30/30 split mismatch')
    query={part:sorted(n for n in dev if by_image[n]==part) for part in ('tune','confirm')};query['report']=sorted(report)
    rest=set(read_text(run/'second_dev/rest.txt').split())-set(dev)-set(report)
    return query,source_partition(rest)


def old_tasks(run):
    out=[]
    for phase in ('dev','val'):
        out+=rows(run/f'results/agent_baseline/data/t3_tasks_{phase}_0of1/tasks.jsonl')
        out += [r for r in rows(run/f't3x/tasks/{phase}/tasks.jsonl') if r['task_type']=='T3.3']
    return out


def audit_old(old,parts):
    records=[];empty=[]
    for row in old:
        kind=row['task_type'];example=read_image(row['exemplar_mask'] if kind=='T3.1' else row['examples'][0]['mask'])>0
        gt=read_image(row['gt_mask'])>0;remaining=gt&~example if kind=='T3.1' else gt
        g=geometry(example);s=support(remaining);reasons=example_reasons(g)+query_reasons(s)
        record=dict(id=row['id'],task_type=kind,split=parts[Path(row['image_t1']).name],query_image=Path(row['image_t1']).name,
                    geometry=g,query_support=s,reasons=reasons,automatic_geometry_pass=not reasons,original_task=row)
        records.append(record)
        if s['pixels']==0:empty.append(record)
    return records,empty


def catalogue(dataset,sources):
    result={p:defaultdict(list) for p in PARTS};audit=[]
    for part in PARTS:
        for name in sources[part]:
            a,b=load_labels(dataset,name)
            for source,target in transitions(a,b):
                for index,indices in enumerate(components((a==source)&(b==target)),1):
                    g=component_geometry(indices,a.shape);reasons=example_reasons(g)
                    descriptor=dict(name=name,source=source,target=target,component=index,geometry=g)
                    audit.append(dict(split=part,**descriptor,reasons=reasons))
                    if not reasons:result[part][(source,target)].append(descriptor)
    return result,audit


def save_external(dataset,descriptor,destination):
    # Full native paired images preserve context and avoid changing example support.
    a,b=load_labels(dataset,descriptor['name']);items=components((a==descriptor['source'])&(b==descriptor['target']))
    ex=mask_from(items[descriptor['component']-1],a.shape);path=destination/'example.png';save_image(path,ex.astype(np.uint8)*255)
    return dict(role='positive',image_t1=str((dataset/'img1'/descriptor['name']).resolve()),
                image_t2=str((dataset/'img2'/descriptor['name']).resolve()),mask=str(path.resolve()))


def rebuild(dataset,query,pool,old,out):
    old_map={(r['task_type'],Path(r['image_t1']).name,r['id'].rsplit('_',1)[-1]):r['id'] for r in old}
    manifest={kind:{p:[] for p in PARTS} for kind in ('T3.1','T3.3')};audit=[];mapping=[]
    for part in PARTS:
        for name in query[part]:
            a,b=load_labels(dataset,name)
            for source,target in transitions(a,b):
                truth=(a==source)&(b==target);items=components(truth);key=f'{Path(name).stem}_{source}to{target}'
                sizes=[len(v) for v in items];gt_support=dict(pixels=sum(sizes),components=len(sizes),largest_component=max(sizes))
                descending=sorted(sizes,reverse=True)
                for kind in ('T3.1','T3.3'):
                    reasons=[];eligible=[];component_audit=[]
                    if kind=='T3.1':
                        for i,indices in enumerate(items):
                            g=component_geometry(indices,a.shape)
                            # E is an entire original 8-connected component, so
                            # removing it leaves every other component unchanged.
                            largest=(descending[1] if len(descending)>1 else 0) if len(indices)==descending[0] else descending[0]
                            rem=dict(pixels=gt_support['pixels']-len(indices),components=len(items)-1,largest_component=largest)
                            bad=example_reasons(g)+query_reasons(rem)
                            component_audit.append(dict(component=i+1,geometry=g,query_support=rem,reasons=bad))
                            if not bad:eligible.append(dict(component=i+1,geometry=g,query_support=rem,name=name,source=source,target=target))
                        if not eligible:reasons=['no_eligible_same_image_example']
                    else:
                        reasons=query_reasons(gt_support)
                        eligible=pool[part].get((source,target),[])
                        if not eligible:reasons.append('no_eligible_external_example_in_assigned_source_partition')
                    old_id=old_map.get((kind,name,f'{source}to{target}'))
                    record=dict(task_type=kind,split=part,query_image=name,transition=[source,target],old_id=old_id,
                                full_gt=gt_support,candidate_examples=len(eligible),component_audit=component_audit,reasons=reasons,
                                decision='excluded' if reasons else 'pending_review')
                    if not reasons:
                        descriptor=choose(eligible,kind+key);new_id=f'n9_{kind.replace(".","")}_{key}_e{descriptor["component"]}'
                        folder=out/'data'/kind.replace('.','')/part/new_id;gt_path=folder/'gt.png';save_image(gt_path,truth.astype(np.uint8)*255)
                        task=dict(id=new_id,task_type=kind,target='any change',change_direction='any',
                                  instruction=('Find all land-cover changes with the same directed transition as the marked example. Output a binary mask.' if kind=='T3.1' else 'Find all changes with the same directed transition as the positive external before/after example. Output a binary mask.'),
                                  constraint=None,init_mask=None,image_t1=str((dataset/'img1'/name).resolve()),image_t2=str((dataset/'img2'/name).resolve()),
                                  gt_mask=str(gt_path.resolve()),review_status='pending_review',
                                  meta=dict(review_status='pending_review',data_version='N9-clean-candidate',split=part,old_id=old_id,
                                            transition=f'{source}->{target}',transition_cn=f'{CN[source]}→{CN[target]}',
                                            example_provenance=[descriptor],geometry=descriptor['geometry'],
                                            query_support=descriptor['query_support'] if kind=='T3.1' else gt_support,
                                            inference_whitelist=['image_t1','image_t2','instruction','exemplar_mask' if kind=='T3.1' else 'examples'],
                                            gt_policy='full original directed-transition pixels; exclude E only at T3.1 scoring'))
                        if kind=='T3.1':
                            ex=mask_from(items[descriptor['component']-1],a.shape);ep=folder/'example.png';save_image(ep,ex.astype(np.uint8)*255);task['exemplar_mask']=str(ep.resolve())
                            remaining_path=folder/'remaining_gt_review_only.png';save_image(remaining_path,(truth&~ex).astype(np.uint8)*255)
                            task['meta']['remaining_gt_review_only']=str(remaining_path.resolve())
                        else:task['examples']=[save_external(dataset,descriptor,folder)]
                        manifest[kind][part].append(task);record.update(new_id=new_id,selected_example=descriptor)
                    mapping.append(dict(old_id=old_id,new_id=record.get('new_id'),task_type=kind,split=part,query_image=name,
                                        transition=[source,target],status=record['decision'],reasons=reasons))
                    audit.append(record)
            print('N9 QC',part,name,flush=True)
    return manifest,audit,mapping


def summarize(old,audit,source_audit,manifest):
    table=[]
    for kind in ('T3.1','T3.3'):
        for part in PARTS:
            before=[r for r in old if r['task_type']==kind and r['split']==part]
            now=[r for r in audit if r['task_type']==kind and r['split']==part]
            table.append(dict(task_type=kind,split=part,old_tasks=len(before),all_observed_query_transitions=len(now),
                              pending_review=len(manifest[kind][part]),excluded=sum(bool(r['reasons']) for r in now),
                              exclusion_reasons=dict(Counter(x for r in now for x in r['reasons'])),
                              component_failure_reasons=dict(Counter(x for r in now for c in r['component_audit'] for x in c['reasons'])),
                              before_transitions=dict(Counter(r['original_task']['id'].rsplit('_',1)[-1] for r in before)),
                              after_transitions=dict(Counter(r['meta']['transition'] for r in manifest[kind][part]))))
    return dict(status='pending_review',policy=POLICY,table=table,external_component_qc=dict(total=len(source_audit),passed=sum(not r['reasons'] for r in source_audit)),
                formal_evaluation_allowed=False,api_calls=0,warning='Automatic QC and six-case display are not approval. Report pool geometry is construction-only; no performance or model scores used.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=DATASET);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('N9 image work must run on Slurm')
    check_encoding(a.dataset);query,sources=inventory(a.run,a.night)
    parts={name:part for part in PARTS for name in query[part]};old=old_tasks(a.run)
    write_json(a.output/'predeclaration.json',dict(**POLICY,query_images=query,external_source_images=sources,
        bbox_definition='native support area / tight axis-aligned bounding rectangle area; inclusive min .25',
        example_display='T3.3 new examples use full native source images with exact component mask, not legacy 4x crops'))
    write_json(a.output/'partition.json',dict(query=query,external_sources=sources))
    prior,empty=audit_old(old,parts);write_json(a.output/'old_task_audit.json',prior)
    write_json(a.output/'t3_empty_abstention_diag.json',dict(status='sealed_not_evaluated',tasks=empty))
    pool,source_audit=catalogue(a.dataset,sources);write_json(a.output/'external_component_audit.json',source_audit)
    manifests,audit,mapping=rebuild(a.dataset,query,pool,old,a.output)
    for kind in manifests:
        for part,tasks in manifests[kind].items():
            base=a.output/f'{kind.lower().replace(".","_")}_clean_{part}'
            write_json(base.with_suffix('.json'),tasks)
            write_text(base.with_suffix('.jsonl'),''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks))
    write_json(a.output/'rebuild_audit.json',audit);write_json(a.output/'old_new_mapping.json',mapping)
    summary=summarize(prior,audit,source_audit,manifests);write_json(a.output/'summary.json',summary)
    lines=['# N9 T3 全候选构造QC','',
           '状态：全部新任务待人工审核。无模型/API调用，无正式confirm/online。report仅几何数据审计。',
           '', '| 类型 | 划分 | 旧任务数 | 全部观测转换数 | 新候选待审核 | 剔除 |', '|---|---|---:|---:|---:|---:|']
    for r in summary['table']:
        lines.append('| '+' | '.join(str(r[k]) for k in ('task_type','split','old_tasks','all_observed_query_transitions','pending_review','excluded'))+' |')
    lines += ['', '剔除原因可重叠；逐组件失败原因、前后转换类别分布见summary.json。全部原任务QC见old_task_audit.json；旧空目标仅封存在t3_empty_abstention_diag.json，本轮不测。',
              '', '确认候选不足20条时如实报告，不跨划分补齐；即使足够，也必须等待用户对新版数据明确批准。']
    write_text(a.output/'qc_report.md','\n'.join(lines)+'\n')
    from .n9_t3_review import render_review
    render_review(prior,manifests,a.output,a.review)

if __name__=='__main__':main()
