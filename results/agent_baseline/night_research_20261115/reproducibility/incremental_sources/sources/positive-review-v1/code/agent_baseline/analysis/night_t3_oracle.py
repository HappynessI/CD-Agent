"""Reuse historical GT-text oracle JSON and cache-only loss/gain diagnostics; no model calls."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.exemplar_semantics import CLASSES
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import load_dev,pair_id,group,pooled,GROUPS
from .t3_routes_cache import read_npz


def cached_union(run,night,row,category,time,shape):
    paths=sorted((run/'t3_diag/cache/t3_sam3/sam3_text'/row['id']).glob('*.json'))
    paths.append(night/'text_cache/t3_sam3/sam3_text'/(row['id']+'_'+category.replace(' ','_'))/f't1_{time}.json')
    for path in paths:
        if not exists(path):continue
        meta=json.loads(read_text(path));request=meta['request']
        if request.get('text')!=category or request['query']!=row['image_'+time]:continue
        if meta['pixel_thr']!=.85 or meta['score_thr']!=.5 or meta['resolution']!=1008:continue
        packed=read_npz(path.with_suffix('.npz'));dimensions=tuple(map(int,packed['shape']))
        if dimensions!=shape:raise ValueError('Cached proposal coordinates differ')
        masks=np.unpackbits(packed['masks'],axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)
        return masks.any(axis=0) if len(masks) else np.zeros(shape,bool),str(path)
    return None,None


def constraint_counts(unions,truth):
    source1,source2,target1,target2=unions
    joint=source1&target2
    return dict(truth_pixels=int(truth.sum()),source_at_t1=int((source1&truth).sum()),target_at_t2=int((target2&truth).sum()),
        positive_intersection=int((joint&truth).sum()),after_source_loss=int((joint&~source2&truth).sum()),
        after_target_gain=int((joint&~target1&truth).sum()),strict_loss_and_gain=int((joint&~source2&~target1&truth).sum()))


def strata(records):
    result=[]
    for split in ('all','tune','confirm'):
        for support in ('all','positive','empty'):
            for label in ('全部',*GROUPS):
                rows=[r for r in records if (split=='all' or r['split']==split) and (label=='全部' or r['group']==label)
                    and (support=='all' or (r['remaining_gt_pixels']>0)==(support=='positive'))]
                methods={}
                for method in ('historical_oracle','predicted'):
                    upper={key:sum(r[method]['upper'][key] for r in rows) for key in ['truth_pixels','covered_pixels','gt_instances','aligned_instances','proposals']}
                    methods[method]=dict(pooled([dict(counts=r[method]['counts']) for r in rows]),**upper,
                        empty_prediction_rate=sum(r[method]['counts'][0]+r[method]['counts'][1]==0 for r in rows)/len(rows) if rows else None)
                result.append(dict(split=split,support=support,group=label,tasks=len(rows),methods=methods))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run cache processing through Slurm')
    old_path=a.run/'t3_diag/report.json';old=json.loads(read_text(old_path));old_rows={r['id']:r for r in old['tasks']}
    audit=json.loads(read_text(a.night/'audit.json'));parses={r['id']:r for r in json.loads(read_text(a.night/'B0/summary.json'))['records']}
    predicted=json.loads(read_text(a.night/'retrieval/B0/summary_all.json'));new_rows={r['id']:r for r in predicted['records']}
    rows=load_dev(a.run)
    if set(old_rows)!={r['id'] for r in rows} or set(new_rows)!=set(old_rows):raise ValueError('Both historical and predicted records must cover fixed60')
    out=a.night/'oracle_cache';records=[]
    write_json(out/'settings.json',dict(code=str(Path(__file__).resolve()),historical_source=str(old_path),model_calls=0,api_cost_cny=0,
        oracle='Historical SAM3 text prompts used GT transition classes. Oracle diagnosis only, never online or a claimed predicted pipeline.',
        cache_only='Existing exact text/query masks only; missing caches recorded, never filled by model inference',
        constraint='positive intersection sourceT1 AND targetT2; remove sourceT2 and targetT1 separately/jointly; diagnostic coverage, not a tuned new route'))
    for row in rows:
        ex=read_image(row['exemplar_mask'])>0;truth=(read_image(row['gt_mask'])>0)&~ex
        old_method=old_rows[row['id']]['methods']['sam3_text'];new=new_rows[row['id']]
        new_score=next(r for r in new['scores'] if r['method']=='R2_all')
        pred_pair=parses[row['id']]['interpretation']['pair'];truth_pair=list(pair_id(row));diagnostics={}
        for name,pair,expected in [('historical_oracle',truth_pair,old_method['proposals']['covered_pixels']),('predicted',pred_pair,new['text_proposal_upper']['covered_pixels'])]:
            if None in pair or pair[0]==pair[1]:
                diagnostics[name]=dict(status='unknown_or_same_class_skip',covered_pixels=0);continue
            unions=[];paths=[]
            for category in pair:
                for time in ('t1','t2'):
                    union,path=cached_union(a.run,a.night,row,CLASSES[category],time,truth.shape)
                    unions.append(union);paths.append(path)
            if any(x is None for x in unions):diagnostics[name]=dict(status='missing_cache',paths=paths);continue
            counts=constraint_counts(unions,truth)
            if counts['strict_loss_and_gain']!=expected:raise ValueError(f'Cached strict coverage differs from archived result: {row["id"]} {name}')
            diagnostics[name]=dict(status='complete',paths=paths,**counts)
        record=dict(id=row['id'],split=audit['split'][row['id']],group=group(row),remaining_gt_pixels=int(truth.sum()),
            predicted_pair=pred_pair,oracle_pair=truth_pair,pair_label_agreement=pred_pair==truth_pair,
            historical_oracle=dict(counts=old_method['scores']['default']['counts'],upper=old_method['proposals']),
            predicted=dict(counts=new_score['counts'],upper=new['text_proposal_upper']),constraint_diagnostics=diagnostics)
        records.append(record);write_json(out/'tasks'/row['id']/'diagnostic.json',record);print('oracle cache',len(records),row['id'],flush=True)
    summary=strata(records);write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summary,model_calls=0,api_cost_cny=0))
    lines=['# Historical oracle versus predicted text proposals','',
        'Oracle uses GT transition labels in historical SAM3 requests: diagnostic only. No new model/API calls. Existing strict directed-proposal coverage is checked against cached masks; relaxed intersections diagnose constraints, not a selected new method.','',
        '| Split | GT | Group | N | Method | F1 | P | R | Covered/GT pixels | Aligned/GT instances |','|---|---|---|---:|---|---:|---:|---:|---|---|']
    for r in summary:
        for method,v in r['methods'].items():lines.append(f"| {r['split']} | {r['support']} | {r['group']} | {r['tasks']} | {method} | {v['f1_c']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['covered_pixels']}/{v['truth_pixels']} | {v['aligned_instances']}/{v['gt_instances']} |")
    lines+=['','## Fixed tune positive examples','', '| Task | Pair label agreement | Predicted proposals | Oracle covered pixels | Predicted covered pixels | Predicted constraint evidence |','|---|---|---:|---:|---:|---|']
    for r in records:
        if r['split']=='tune' and r['remaining_gt_pixels']>0:lines.append(f"| {r['id']} | {r['pair_label_agreement']} | {r['predicted']['upper']['proposals']} | {r['historical_oracle']['upper']['covered_pixels']} | {r['predicted']['upper']['covered_pixels']} | {json.dumps({k:v for k,v in r['constraint_diagnostics']['predicted'].items() if k!='paths'},ensure_ascii=False)} |")
    write_text(out/'report.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
