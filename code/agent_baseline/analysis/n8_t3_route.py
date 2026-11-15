"""T3-2 train-free example routing and tune-frozen structural dense fusion."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,pooled
from .t3_routes_cache import read_npz
from .n8_t3_common import packed_load,packed_save,observable_example,mask_score,report,RANKS,K_VALUES

HIGH_CONFIDENCE=(.7,.85)


def select_instance(records):
    if any(r['split']!='tune' for r in records):raise ValueError('Tune only selection')
    candidates=[]
    for rank in RANKS:
        for k in K_VALUES:
            method=f'clean_{rank}_top{k}';metrics=pooled([r['methods'][method] for r in records])
            candidates.append(dict(rank=rank,k=k,method=method,metrics=metrics))
    selected=max(candidates,key=lambda r:(r['metrics']['f1_c'],-r['k'],r['rank']=='delta'))
    return dict(selected=selected,candidates=candidates,objective='tune all-task pooled F1; tie smaller K then delta')


def dense_high(probabilities,pair,dense,tau):
    return dense&(probabilities[0][pair[0]]>=tau)&(probabilities[1][pair[1]]>=tau)


def routed_predictions(instance,dense,probabilities,observable):
    structural=observable['structural'];result=dict(branch_inst=instance,branch_dense=dense,
        route=instance if structural else dense)
    for tau in HIGH_CONFIDENCE:
        high=dense_high(probabilities,observable['predicted_pair'],dense,tau)
        for op in ('intersection','union'):
            combined=instance&high if op=='intersection' else instance|high
            result[f'hybrid_{op}_{tau}']=combined if structural else dense
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--semantic',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm'],required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    root=a.out/'step2';out=root/a.split
    originals=json.loads(read_text(a.out/'step1/summary.json'))['records'];prior={r['id']:r for r in originals if r['split']==a.split}
    if a.split=='tune':
        choice=select_instance([r for r in originals if r['split']=='tune']);write_json(root/'instance_frozen.json',choice)
    else:choice=json.loads(read_text(root/'instance_frozen.json'))
    fixed=choice['selected'];frozen=json.loads(read_text(root/'system_frozen.json')) if a.split=='confirm' else None
    records=[]
    for row in load_dev(a.run):
        if row['id'] not in prior:continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        masks,saved=packed_load(a.out/'step1/tasks'/row['id']/'clean_pool.npz');ex=read_image(row['exemplar_mask'])>0
        order=saved['order_'+fixed['rank']].tolist();instance=masks[order[:fixed['k']]].any(axis=0) if order else np.zeros_like(ex)
        probabilities=[read_npz(a.semantic/f"{row['id']}_{t}.npz")['data'] for t in ('t1','t2')]
        observable,dense=observable_example(probabilities,ex)
        predictions=routed_predictions(instance,dense,probabilities,observable)
        if frozen:predictions={k:v for k,v in predictions.items() if k in ('branch_inst','branch_dense','route',frozen['hybrid_method'])}
        names=list(predictions);packed_save(folder/'predictions.npz',np.stack(list(predictions.values())),methods=np.asarray(names))
        write_json(folder/'inference.json',dict(observable=observable,instance_config=fixed,instance_candidate_ids=order[:fixed['k']],
            route='instance' if observable['structural'] else 'dense',methods=names,
            high_confidence_rule='Both example-predicted endpoint probabilities at each pixel >= tau; intersect with rebuilt routeA mask.'))
        truth=read_image(row['gt_mask'])>0
        methods={'original_R1_top8':prior[row['id']]['methods']['original_R1_top8']}
        methods.update({name:mask_score(mask,truth,ex) for name,mask in predictions.items()})
        record=dict(id=row['id'],split=a.split,group=prior[row['id']]['group'],remaining_gt_pixels=prior[row['id']]['remaining_gt_pixels'],
            methods=methods,routed_to='instance' if observable['structural'] else 'dense',observable=observable)
        write_json(final,record);records.append(record);print('N8 route',a.split,len(records),row['id'],record['routed_to'],flush=True)
    if a.split=='tune':
        candidates=[]
        for name in ['route',*[f'hybrid_{op}_{tau}' for tau in HIGH_CONFIDENCE for op in ('intersection','union')]]:
            candidates.append(dict(method=name,metrics=pooled([r['methods'][name] for r in records])))
        hybrid=max(candidates[1:],key=lambda c:c['metrics']['f1_c'])
        best=max(candidates,key=lambda c:c['metrics']['f1_c'])
        frozen=dict(instance=fixed,hybrid_method=hybrid['method'],system_method=best['method'],candidates=candidates,
            objective='tune all-task pooled F1; ties preserve declared route-first/config order; no confirm selection',
            high_confidence_thresholds=list(HIGH_CONFIDENCE),channels=records[0]['observable']['channels'])
        write_json(root/'system_frozen.json',frozen)
    report(out,records,'T3-2 '+a.split,selection=frozen,api_calls=0)
    if a.split=='confirm':report(root,json.loads(read_text(root/'tune/summary.json'))['records']+records,'T3-2 full dev60',selection=frozen,api_calls=0)


if __name__=='__main__':main()
