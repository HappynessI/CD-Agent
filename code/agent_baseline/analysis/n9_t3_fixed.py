"""Explicit exploratory dev_tune fixed comparison. No confirm/report/online option."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json,write_text,exists
from .n9_t3_retrieval import retrieve,RANKS,KS
from .t3_routes_cache import CachedVision
from .n8_t3_common import packed_save,packed_load
from .second_labels import pixel_counts
from .t3_routes import pooled

METHODS=('baseline_R1_delta_top8','clean_delta_top5','clean_delta_top8','clean_joint_top5','clean_joint_top8')


def input_identity(row,same):
    spec=row if same else row['examples'][0]
    paths=[row['image_t1'],row['image_t2'],spec['image_t1'],spec['image_t2'],row['exemplar_mask'] if same else spec['mask']]
    return [dict(path=str(Path(p).resolve()),size=Path(p).stat().st_size,mtime=Path(p).stat().st_mtime_ns) for p in paths]


def tune_tasks(data,kinds):
    partition=json.loads(read_text(data/'partition.json'));query=set(partition['query']['tune']);sources=set(partition['external_sources']['tune'])
    tasks=[]
    for kind in kinds:
        path=data/f'{kind.lower().replace(".","_")}_clean_tune.json';current=json.loads(read_text(path))
        for row in current:
            if row['meta']['split']!='tune' or Path(row['image_t1']).name not in query:raise ValueError('Only fixed dev_tune is authorized')
            if row['task_type']=='T3.3':
                e=row['examples'][0]
                if Path(e['image_t1']).name not in sources or Path(e['image_t2']).name not in sources:raise ValueError('External example outside assigned tune sources')
        tasks.extend(current)
    return tasks


class Backends:
    def __init__(self,run,out,mode,config=None):
        roots=[run/'t3_diag/cache/vision',run/'vision_t13',out/'vision_cache']
        self.caches=[(root,CachedVision(root)) for root in roots if exists(root/'dynamic/inventory.json')]
        self.pool=None;self.mode=mode;self.out=out;self.config=config;self.calls=[]
    def locate(self,row,operation,time):
        paths=[str(Path(row[k]).resolve()) for k in ('image_t1','image_t2')]
        for root,cache in self.caches:
            hits=[r for r in cache.inventory if [s['path'] for s in r['sources']]==paths and r['operation']==operation and r.get('time')==time and r.get('classes') is None]
            if len(hits)==1:return root,cache.for_row(row)
        return None
    def for_row(self,row):
        minimal={k:row[k] for k in ('image_t1','image_t2')}
        def call(operation,**kwargs):
            found=self.locate(minimal,operation,kwargs['time'])
            if found:
                root,backend=found;result=backend(operation,**kwargs)
                self.calls.append(dict(operation=operation,time=kwargs['time'],source='exact_source_cache',cache=str(root)))
                return result
            if self.mode=='cache-only':raise RuntimeError('Missing exact-source vision cache; run authorized GPU fill-missing entry, not a substitute old E cache')
            if self.pool is None:
                from agent_baseline.backends_t13 import WorkerPool
                self.pool=WorkerPool(self.config,self.out/'vision_cache')
            self.calls.append(dict(operation=operation,time=kwargs['time'],source='live_local_worker',cache=str(self.out/'vision_cache')))
            return self.pool.for_pair([minimal[k] for k in ('image_t1','image_t2')])(operation,**kwargs)
        return call
    def close(self):
        if self.pool is not None:self.pool.close()


def requirements(tasks,backend):
    needed={}
    for row in tasks:
        specs=[(row,('segment_everything','extract_features'))]
        if row['task_type']=='T3.3':specs.append((row['examples'][0],('extract_features',)))
        for spec,operations in specs:
            for op in operations:
                for time in ('t1','t2'):
                    key=(spec['image_t1'],spec['image_t2'],op,time)
                    if key not in needed:needed[key]=dict(image_t1=key[0],image_t2=key[1],operation=op,time=time,cached=backend.locate(spec,op,time) is not None)
    values=list(needed.values())
    return dict(status='exploratory_dev_tune_only',tasks=len(tasks),operations=len(values),missing=sum(not r['cached'] for r in values),records=values)


def evaluate(masks,indices,truth,excluded):
    pred=masks[indices].any(axis=0) if indices else np.zeros_like(truth)
    pred=pred&~excluded;truth=truth&~excluded
    return dict(counts=pixel_counts(pred,truth),predicted_pixels=int(pred.sum()))


def summarize(records):
    table=[]
    for kind in ('T3.1','T3.3'):
        for group in ('全部','目标类','地表类'):
            rows=[r for r in records if r['task_type']==kind and (group=='全部' or r['group']==group)]
            if not rows:continue
            methods={name:dict(**pooled([r['methods'][name] for r in rows]),
                       empty_predictions=sum(r['methods'][name]['predicted_pixels']==0 for r in rows)) for name in METHODS}
            table.append(dict(task_type=kind,group=group,tasks=len(rows),query_images=len({r['query_image'] for r in rows}),
                              gt_pixels=sum(r['gt_pixels'] for r in rows),methods=methods,
                              original_pool_covered=sum(r['original_pool_covered'] for r in rows),clean_pool_covered=sum(r['clean_pool_covered'] for r in rows)))
    selected={}
    for item in table:
        if item['group']!='全部':continue
        candidates=[(rank,k) for rank in RANKS for k in KS]
        rank,k=max(candidates,key=lambda v:(item['methods'][f'clean_{v[0]}_top{v[1]}']['f1_c'],-v[1],v[0]=='delta'))
        selected[item['task_type']]=dict(rank=rank,k=k,objective='own-task-type dev_tune pooled pixel F1; ties smaller K, then delta',
                                      status='exploratory_recommendation_pending_data_approval')
    return table,selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--task-type',action='append',choices=['T3.1','T3.3']);p.add_argument('--backend',choices=['cache-only','fill-missing'],default='cache-only')
    p.add_argument('--vision-config',type=Path);p.add_argument('--inventory-only',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for all tests/model/array work')
    if a.backend=='fill-missing' and not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES')):raise RuntimeError('Local model fill needs Slurm GPU')
    kinds=a.task_type or ['T3.1','T3.3'];tasks=tune_tasks(a.data,kinds);out=a.data/'exploratory_fixed_tune'
    config=json.loads(read_text(a.vision_config)) if a.vision_config else None
    if a.backend=='fill-missing' and config is None:raise ValueError('GPU fill requires explicit local vision config')
    backend=Backends(a.run,out,a.backend,config);name='_'.join(k.replace('.','') for k in kinds)
    write_json(out/f'cache_requirements_{name}.json',requirements(tasks,backend))
    if a.inventory_only:return
    settings=dict(status='exploratory_dev_tune_only_pending_review',task_types=kinds,task_ids=[r['id'] for r in tasks],methods=METHODS,
                  baseline='Original R1 geometry from current E, original support delta Top8; no self-match removal; T3.1 score excludes E',
                  geometry='candidate/example area .25..4; aspect .333333..3; paired-RGB RMS<=.25; recomputed using new E',
                  clean='T3.1 overlap>=.15 OR IoU>=.15 removed before rank; otherwise C minus E, no tiny-net filter. T3.3 no removal/subtraction.',
                  delta='cos(c2-c1,e2-e1)',joint='.5*(delta+.5*(cos(c1,e1)+cos(c2,e2)))',
                  gates='No z/p global rejection; no SegEarth routing; no GT classes; no VLM verifier in this phase',api_calls=0,
                  next_stage='Formal confirm/online still requires explicit clean-data human approval; no automatic promotion')
    write_json(out/f'settings_{name}.json',settings);records=[]
    try:
        for row in tasks:
            folder=out/'tasks'/row['id'];metrics=folder/'metrics.json'
            same=row['task_type']=='T3.1';identity=input_identity(row,same)
            if exists(metrics):
                saved=json.loads(read_text(folder/'inference.json'))
                if saved['input_identity']!=identity:raise ValueError('N9 task RGB/E changed; prior metrics/inference need a new run')
                records.append(json.loads(read_text(metrics)));continue
            qimages=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')]
            spec=row if same else row['examples'][0]
            example=read_image(row['exemplar_mask'] if same else spec['mask'])>0
            refs=qimages if same else [read_image(spec[k],rgb=True) for k in ('image_t1','image_t2')]
            if exists(folder/'inference.json'):
                original,_=packed_load(folder/'original_pool.npz');clean,_=packed_load(folder/'clean_pool.npz')
                inference=json.loads(read_text(folder/'inference.json'))
                if inference['input_identity']!=identity:raise ValueError('N9 task RGB/E changed; do not reuse prior fixed inference')
            else:
                before=len(backend.calls)
                original,clean,inference=retrieve(qimages,refs,example,backend.for_row(row),backend.for_row(spec),same)
                packed_save(folder/'original_pool.npz',original);packed_save(folder/'clean_pool.npz',clean)
                write_json(folder/'inference.json',dict(**inference,input_identity=identity,vision_calls=backend.calls[before:]))
            # The GT is loaded only after inference arrays/scores/orders have been persisted.
            truth=read_image(row['gt_mask'])>0;excluded=example if same else np.zeros_like(truth)
            methods={METHODS[0]:evaluate(original,inference['original_delta_order'][:8],truth,excluded)}
            for rank in RANKS:
                for k in KS:methods[f'clean_{rank}_top{k}']=evaluate(clean,inference['clean_orders'][rank][:k],truth,excluded)
            effective=truth&~excluded;pair=list(map(int,row['meta']['transition'].split('->')))
            covered=lambda masks:int(((masks.any(axis=0) if len(masks) else np.zeros_like(truth))&effective).sum())
            record=dict(id=row['id'],task_type=row['task_type'],split='tune',query_image=Path(row['image_t1']).name,
                        group='目标类' if set(pair)&{5,6} else '地表类',gt_pixels=int(effective.sum()),methods=methods,
                        original_candidates=len(original),clean_candidates=len(clean),original_pool_covered=covered(original),clean_pool_covered=covered(clean))
            write_json(metrics,record);records.append(record)
            write_json(out/f'progress_{name}.json',dict(tasks=len(records),expected=len(tasks),status='exploratory_dev_tune_only'))
            print('N9 exploratory fixed',len(records),len(tasks),row['id'],flush=True)
    finally:backend.close()
    table,selected=summarize(records)
    write_json(out/f'summary_{name}.json',dict(status='exploratory_dev_tune_only_pending_review',records=records,summary=table,selected=selected,api_calls=0))
    write_json(out/f'tune_candidate_selection_{name}.json',selected)
    lines=['# N9 待审核数据上的 dev_tune 固定检索探索','', '不是正式confirm或online结果；不与旧N7/N8任务分布直接比较。',
           '', '| 类型 | 分组 | 任务/图像 | 方法 | P | R | F1 | TP | FP | FN | 空预测 |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in table:
        for method,m in item['methods'].items():
            fields=[item['task_type'],item['group'],f"{item['tasks']}/{item['query_images']}",method,*[round(m[k],5) for k in ('precision','recall','f1_c')],m['tp'],m['fp'],m['fn'],m['empty_predictions']]
            lines.append('| '+' | '.join(map(str,fields))+' |')
    write_text(out/f'report_{name}.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
