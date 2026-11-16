"""Approved T3.1 frozen online cohort: two fixed comparators, no parameter search."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .io_retry import read_image,write_json,write_text,save_image
from .n9_t3_online_manifest import validate_frozen
from .n9_t3_fixed import Backends,input_identity,evaluate
from .n9_t3_retrieval import retrieve
from .n8_t3_common import packed_save,packed_load
from .t3_routes import pooled

def cohort(data,frozen,tune_selection):
    freeze=validate_frozen(frozen);method=freeze['configuration']['T3.1']
    if json.loads(tune_selection.read_text())['T3.1']!=method:raise ValueError('Method must equal saved tune selection, without confirm reselection')
    source={r['id']:r for r in json.loads((data/'t3_1_clean_confirm.json').read_text())}
    mapping=json.loads((frozen/'scoring_mapping.json').read_text())
    tasks=[(m['id'],source[m['original_id']]) for m in mapping]
    frozen_rows=[json.loads(s) for s in (frozen/'tasks.jsonl').read_text().splitlines() if s]
    if [x[0] for x in tasks]!=[r['id'] for r in frozen_rows]:raise ValueError('Fixed/online cohort mismatch')
    if any(r['task_type']!='T3.1' or r['meta']['split']!='confirm' for _,r in tasks):raise ValueError('T3.1 confirm only')
    return tasks,method,freeze

def report(records):
    table=[]
    for group in ('全部','目标类','地表类'):
        chosen=[r for r in records if group=='全部' or r['group']==group]
        if not chosen:continue
        methods={name:dict(**pooled([r['methods'][name] for r in chosen]),empty_predictions=sum(r['methods'][name]['predicted_pixels']==0 for r in chosen)) for name in ('Baseline-Fixed','N9-Fixed')}
        table.append(dict(group=group,tasks=len(chosen),independent_images=len({r['query_image'] for r in chosen}),gt_pixels=sum(r['gt_pixels'] for r in chosen),methods=methods))
    return table

def main():
    p=argparse.ArgumentParser()
    for key in ('data','run','frozen','tune-selection'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=('cache-only','fill-missing'),default='cache-only');p.add_argument('--vision-config',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    tasks,method,freeze=cohort(a.data,a.frozen,a.tune_selection);out=a.data/'fixed_confirm_online_cohort'
    config=json.loads(a.vision_config.read_text()) if a.vision_config else None
    if a.backend=='fill-missing' and (config is None or not any(os.environ.get(k) for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES'))):raise ValueError('GPU config and allocation required')
    backend=Backends(a.run,out,a.backend,config);records=[]
    if (out/'preregistration.json').exists():
        prior=json.loads((out/'preregistration.json').read_text())
        if prior['configuration']!=method or prior['task_ids']!=[x[0] for x in tasks]:raise ValueError('Existing fixed cohort/configuration differs; preserve prior run')
    write_json(out/'preregistration.json',dict(status='approved_frozen_cohort',approval=freeze['approval'],configuration=method,task_ids=[x[0] for x in tasks],methods=['Baseline-Fixed','N9-Fixed'],parameter_search=False))
    try:
        for opaque,row in tasks:
            folder=out/'tasks'/opaque;identity=input_identity(row,True)
            if (folder/'metrics.json').exists():
                if json.loads((folder/'inference.json').read_text())['input_identity']!=identity:raise ValueError('Inputs changed')
                records.append(json.loads((folder/'metrics.json').read_text()));continue
            images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];example=read_image(row['exemplar_mask'])>0
            if (folder/'inference.json').exists():
                inference=json.loads((folder/'inference.json').read_text())
                if inference['input_identity']!=identity:raise ValueError('Inputs changed')
                original,_=packed_load(folder/'original.npz');clean,_=packed_load(folder/'clean.npz')
            else:
                original,clean,inference=retrieve(images,images,example,backend.for_row(row),backend.for_row(row),True)
                packed_save(folder/'original.npz',original);packed_save(folder/'clean.npz',clean)
                write_json(folder/'inference.json',dict(inference,input_identity=identity))
            # Evaluation labels enter only after all inference outputs are durable.
            truth=read_image(row['gt_mask'])>0
            arms={'Baseline-Fixed':(original,inference['original_delta_order'][:8]),'N9-Fixed':(clean,inference['clean_orders'][method['rank']][:method['k']])}
            metrics={}
            for name,(masks,indices) in arms.items():
                metrics[name]=evaluate(masks,indices,truth,example)
                pred=(masks[indices].any(axis=0) if indices else np.zeros_like(truth))&~example
                save_image(folder/(name+'.png'),pred.astype(np.uint8)*255)
            pair=set(map(int,row['meta']['transition'].split('->')))
            record=dict(id=opaque,task_type='T3.1',query_image=Path(row['image_t1']).name,group='目标类' if pair&{5,6} else '地表类',gt_pixels=int((truth&~example).sum()),methods=metrics)
            write_json(folder/'metrics.json',record);records.append(record)
            write_json(out/'progress.json',dict(tasks=len(records),expected=len(tasks)))
    finally:backend.close()
    table=report(records);write_json(out/'summary.json',dict(configuration=method,records=records,summary=table,api_calls=0,cohort='exact frozen N9-Agent task IDs',status='complete'))
    lines=['# N9 T3.1 同冻结在线任务的 fixed 对照','', '仅使用人工批准后的T3.1 confirm冻结名单与tune冻结配置，无confirm调参。在线N9-Agent三seed必须逐项使用相同opaque ID。T3.3不占在线配额。','', '| 分组 | 任务/独立图 | 方法 | P | R | F1 | TP | FP | FN | 空预测 |','|---|---|---|---|---|---|---|---|---|---|']
    for item in table:
        for name,m in item['methods'].items():lines.append('| '+' | '.join(map(str,[item['group'],f"{item['tasks']}/{item['independent_images']}",name,*[round(m[k],5) for k in ('precision','recall','f1_c')],m['tp'],m['fp'],m['fn'],m['empty_predictions']]))+' |')
    write_text(out/'report.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
