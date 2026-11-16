"""Full approved clean-confirm evaluation with per-type tune-frozen methods."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .io_retry import read_image,write_json,write_text,save_image
from .n9_t3_fixed import Backends,input_identity,evaluate
from .n9_t3_fixed_confirm import report
from .n9_t3_retrieval import retrieve
from .t3_routes_cache import CachedVision
from .n8_t3_common import packed_save,packed_load

KINDS=('T3.1','T3.3')

def inputs(data,approval,selection_paths):
    approved=json.loads(approval.read_text())
    if approved.get('status')!='approved' or not approved.get('approval_reference'):raise ValueError('Explicit human approval required for full clean confirm')
    methods={}
    for path in selection_paths:
        for kind,value in json.loads(path.read_text()).items():
            if kind not in KINDS:continue
            if kind in methods and methods[kind]!=value:raise ValueError('Conflicting tune selections')
            if value['rank'] not in ('delta','joint') or value['k'] not in (5,8):raise ValueError('Unexpected tune-selected method')
            methods[kind]=value
    if set(methods)!=set(KINDS):raise ValueError('Saved tune configuration required for both task types')
    partition=json.loads((data/'partition.json').read_text());tasks=[]
    for kind in KINDS:
        rows=json.loads((data/f'{kind.lower().replace(".","_")}_clean_confirm.json').read_text())
        for row in rows:
            if row['task_type']!=kind or row['meta']['split']!='confirm' or Path(row['image_t1']).name not in partition['query']['confirm']:raise ValueError('Only original clean confirm queries allowed')
            if kind=='T3.3' and any(Path(row['examples'][0][k]).name not in partition['external_sources']['confirm'] for k in ('image_t1','image_t2')):raise ValueError('External example violates confirm source partition')
        tasks.extend(rows)
    return tasks,methods,approved

def run(a):
    tasks,methods,approval=inputs(a.data,a.approval,a.tune_selection)
    counts={kind:sum(r['task_type']==kind for r in tasks) for kind in KINDS}
    if counts!={'T3.1':106,'T3.3':156}:raise ValueError(f'Full clean-confirm inventory changed: {counts}')
    out=a.data/'fixed_full_confirm';frozen=dict(task_ids=[r['id'] for r in tasks],configuration=methods)
    if (out/'frozen.json').exists():
        prior=json.loads((out/'frozen.json').read_text())
        if any(prior[k]!=v for k,v in frozen.items()):raise ValueError('Existing full confirm freeze differs; preserve prior results')
    else:write_json(out/'frozen.json',dict(frozen,approval=approval,selection_sources=[str(p) for p in a.tune_selection],task_counts=counts,parameter_search=False,report_pool_access=False))
    config=json.loads(a.vision_config.read_text()) if a.vision_config else None
    backend=Backends(a.run,out,a.backend,config)
    extra=a.data/'exploratory_fixed_tune/vision_cache'
    if (extra/'dynamic/inventory.json').exists():backend.caches.append((extra,CachedVision(extra)))
    records=[]
    try:
        for row in tasks:
            same=row['task_type']=='T3.1';method=methods[row['task_type']];folder=out/'tasks'/row['id'];identity=input_identity(row,same)
            if (folder/'metrics.json').exists():
                if json.loads((folder/'inference.json').read_text())['input_identity']!=identity:raise ValueError('Inputs changed')
                records.append(json.loads((folder/'metrics.json').read_text()));continue
            images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];spec=row if same else row['examples'][0]
            example=read_image(row['exemplar_mask'] if same else spec['mask'])>0
            refs=images if same else [read_image(spec[k],rgb=True) for k in ('image_t1','image_t2')]
            if (folder/'inference.json').exists():
                inference=json.loads((folder/'inference.json').read_text())
                if inference['input_identity']!=identity:raise ValueError('Inputs changed')
                original,_=packed_load(folder/'original.npz');clean,_=packed_load(folder/'clean.npz')
            else:
                before=len(backend.calls)
                original,clean,inference=retrieve(images,refs,example,backend.for_row(row),backend.for_row(spec),same)
                packed_save(folder/'original.npz',original);packed_save(folder/'clean.npz',clean)
                write_json(folder/'inference.json',dict(inference,input_identity=identity,vision_calls=backend.calls[before:]))
            # Evaluation-only labels follow durable, GT-free inference; external E is never subtracted.
            truth=read_image(row['gt_mask'])>0;excluded=example.copy() if same else np.zeros_like(truth)
            if row.get('gt_ignore_mask'):excluded|=read_image(row['gt_ignore_mask'])>0
            arms={'Baseline-Fixed':(original,inference['original_delta_order'][:8]),'N9-Fixed':(clean,inference['clean_orders'][method['rank']][:method['k']])}
            metrics={}
            for name,(masks,indices) in arms.items():
                metrics[name]=evaluate(masks,indices,truth,excluded)
                pred=(masks[indices].any(axis=0) if indices else np.zeros_like(truth))&~excluded
                save_image(folder/(name+'.png'),pred.astype(np.uint8)*255)
            effective=truth&~excluded;pair=set(map(int,row['meta']['transition'].split('->')))
            covered=lambda masks:int(((masks.any(axis=0) if len(masks) else np.zeros_like(truth))&effective).sum())
            record=dict(id=row['id'],task_type=row['task_type'],query_image=Path(row['image_t1']).name,split='confirm',group='目标类' if pair&{5,6} else '地表类',gt_pixels=int(effective.sum()),methods=metrics,original_pool_covered=covered(original),clean_pool_covered=covered(clean),original_candidates=len(original),clean_candidates=len(clean))
            write_json(folder/'metrics.json',record);records.append(record)
            write_json(out/'progress.json',dict(tasks=len(records),expected=len(tasks),last_id=row['id']))
    finally:backend.close()
    table=[]
    for kind in KINDS:
        subset=[r for r in records if r['task_type']==kind]
        for item in report(subset):
            selected=[r for r in subset if item['group']=='全部' or r['group']==item['group']]
            item.update(task_type=kind,original_pool_covered=sum(r['original_pool_covered'] for r in selected),clean_pool_covered=sum(r['clean_pool_covered'] for r in selected))
            item['original_pool_coverage_recall']=item['original_pool_covered']/max(1,item['gt_pixels']);item['clean_pool_coverage_recall']=item['clean_pool_covered']/max(1,item['gt_pixels']);table.append(item)
    write_json(out/'summary.json',dict(status='complete',configuration=methods,records=records,summary=table,task_counts=counts,api_calls=0,report_pool_access=False))
    lines=['# N9 完整 clean-confirm 固定检索确认','', '分别采用两类任务的tune冻结配置，不在confirm选参；T3.1评分扣E，T3.3外部示例不从查询图扣除。与在线20任务同名单fixed报告独立保留。','', '| 类型 | 分组 | 任务/图像 | 方法 | P | R | F1 | TP | FP | FN | 空预测 |','|---|---|---|---|---|---|---|---|---|---|---|']
    for item in table:
        for name,m in item['methods'].items():lines.append('| '+' | '.join(map(str,[item['task_type'],item['group'],f"{item['tasks']}/{item['independent_images']}",name,*[round(m[k],5) for k in ('precision','recall','f1_c')],m['tp'],m['fp'],m['fn'],m['empty_predictions']]))+' |')
    write_text(out/'report.md','\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser()
    for key in ('data','run','approval'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tune-selection',type=Path,action='append',required=True);p.add_argument('--backend',choices=('cache-only','fill-missing'),default='cache-only');p.add_argument('--vision-config',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    if a.backend=='fill-missing' and (not a.vision_config or not any(os.environ.get(k) for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES'))):raise ValueError('GPU config/allocation required')
    run(a)

if __name__=='__main__':main()
