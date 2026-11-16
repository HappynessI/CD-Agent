"""One-pass tune-only exact SAM cache diagnosis; frozen filters, no new inference."""
import argparse,json,os
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .n9_t3_fixed import Backends,tune_tasks
from .t3_geometry import descriptor
from .n8_t3_common import packed_load
from .io_retry import read_image,write_json,write_text,save_image
from .second_labels import pixel_counts

REASONS=('empty','area_below','area_above','aspect_below','aspect_above','color_above')
def failures(d,ref):
    if d is None:return ['empty']
    area=d[0]/ref[0];aspect=d[1]/ref[1];color=float(np.linalg.norm(d[2]-ref[2])/np.sqrt(6))
    return [k for k,failed in [('area_below',area<.25),('area_above',area>4),('aspect_below',aspect<.333333),('aspect_above',aspect>3),('color_above',color>.25)] if failed]

def main():
    p=argparse.ArgumentParser()
    for key in ('data','run','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    tasks=tune_tasks(a.data,['T3.1','T3.3'])
    if len(tasks)!=298:raise ValueError('Expected original 298 tune tasks')
    groups=defaultdict(list)
    for row in tasks:groups[(row['image_t1'],row['image_t2'])].append(row)
    source=a.data/'exploratory_fixed_tune';backend=Backends(a.run,source,'cache-only');records=[]
    try:
        for paths,rows in sorted(groups.items()):
            pending=[]
            for row in rows:
                path=a.output/'tasks'/row['id']/'diagnostic.json'
                if path.exists():records.append(json.loads(path.read_text()))
                else:pending.append(row)
            if not pending:continue
            images=[read_image(path,rgb=True) for path in paths]
            call=backend.for_row(rows[0]);raw=np.concatenate([call('segment_everything',time=t) for t in ('t1','t2')])
            descriptions=[descriptor(mask,images) for mask in raw];raw_union=raw.any(axis=0)
            for row in pending:
                same=row['task_type']=='T3.1';spec=row if same else row['examples'][0]
                e=read_image(row['exemplar_mask'] if same else spec['mask'])>0
                refs=images if same else [read_image(spec[k],rgb=True) for k in ('image_t1','image_t2')]
                ref=descriptor(e,refs);reasons=[failures(d,ref) for d in descriptions];kept=[i for i,r in enumerate(reasons) if not r]
                folder=source/'tasks'/row['id'];inference=json.loads((folder/'inference.json').read_text())
                if kept!=inference['r1_raw_ids']:raise ValueError('Exact cached raw pool/filter disagrees with frozen inference: '+row['id'])
                original,_=packed_load(folder/'original_pool.npz');original_union=original.any(axis=0) if len(original) else np.zeros_like(raw_union)
                clean,_=packed_load(folder/'clean_pool.npz');clean_union=clean.any(axis=0) if len(clean) else np.zeros_like(raw_union)
                # All candidate supports and reasons fixed before evaluation labels are loaded.
                truth=read_image(row['gt_mask'])>0
                if same:truth&=~e
                if row.get('gt_ignore_mask'):truth&=~(read_image(row['gt_ignore_mask'])>0)
                metrics=json.loads((folder/'metrics.json').read_text());exports=[]
                for method,value in metrics['methods'].items():
                    if method=='baseline_R1_delta_top8':pool=original;ids=inference['original_delta_order'][:8]
                    else:
                        _,rank,k=method.split('_');pool=clean;ids=inference['clean_orders'][rank][:int(k[3:])]
                    pred=pool[ids].any(axis=0) if ids else np.zeros_like(truth)
                    if same:pred&=~e
                    if row.get('gt_ignore_mask'):pred&=~(read_image(row['gt_ignore_mask'])>0)
                    counts=list(pixel_counts(pred,truth))
                    if counts!=value['counts']:raise ValueError('Export differs from existing metrics: '+row['id']+' '+method)
                    destination=a.output/'compact_predictions'/row['id']/(method+'.png')
                    save_image(destination,pred.astype(np.uint8)*255)
                    exports.append(dict(method=method,path=str(destination),counts=counts,exact_metrics_match=True))
                del original,clean,pool
                coverage={k:int((u&truth).sum()) for k,u in [('raw',raw_union),('original',original_union),('clean',clean_union)]}
                counter=Counter(k for rs in reasons for k in rs);reason_evidence={}
                for reason in REASONS:
                    ids=[i for i,rs in enumerate(reasons) if reason in rs];union=np.zeros_like(truth)
                    for i in ids:union|=raw[i]
                    reason_evidence[reason]=dict(candidates=len(ids),rejected_union_gt=int((union&truth).sum()),gt_lost_from_original_covered_by_rejected_reason=int((union&truth&~original_union).sum()))
                pair=set(map(int,row['meta']['transition'].split('->')))
                record=dict(id=row['id'],task_type=row['task_type'],query_image=Path(paths[0]).name,group='目标类' if pair&{5,6} else '地表类',gt_pixels=int(truth.sum()),coverage=coverage,
                    raw_candidates=len(raw),original_candidates=len(kept),clean_candidates=inference['clean_candidates'],geometry_loss=coverage['raw']-coverage['original'],cleaning_loss=coverage['original']-coverage['clean'],
                    filter_reason_counts=dict(counter),reason_evidence=reason_evidence,kept_raw_ids=kept,rejected_raw_ids_by_reason={r:[i for i,rs in enumerate(reasons) if r in rs] for r in REASONS},
                    inference_path=str(folder/'inference.json'),compact_predictions=exports,cache_policy='exact query SAM cache; no live fallback',limitations='Reasons overlap: rejected candidates can fail several gates and cover the same GT. Per-reason losses cannot be added or interpreted as causal ablations.')
                write_json(a.output/'tasks'/row['id']/'diagnostic.json',record);records.append(record)
                write_json(a.output/'progress.json',dict(tasks=len(records),expected=298,last_query=Path(paths[0]).name))
            del raw,raw_union,descriptions,images
    finally:backend.close()
    table=[];severe=[]
    for kind in ('T3.1','T3.3'):
        for group in ('全部','目标类','地表类'):
            rows=[r for r in records if r['task_type']==kind and (group=='全部' or r['group']==group)];gt=sum(r['gt_pixels'] for r in rows)
            cover={k:sum(r['coverage'][k] for r in rows) for k in ('raw','original','clean')}
            table.append(dict(task_type=kind,group=group,tasks=len(rows),images=len({r['query_image'] for r in rows}),gt_pixels=gt,coverage=cover,coverage_recall={k:v/max(1,gt) for k,v in cover.items()},
                geometry_loss=cover['raw']-cover['original'],cleaning_loss=cover['original']-cover['clean'],reason_counts={k:sum(r['filter_reason_counts'].get(k,0) for r in rows) for k in REASONS},
                reason_lost_gt_nonadditive={k:sum(r['reason_evidence'][k]['gt_lost_from_original_covered_by_rejected_reason'] for r in rows) for k in REASONS}))
            if group!='全部':severe.extend(sorted(rows,key=lambda r:(-r['geometry_loss'],r['id']))[:3])
    write_json(a.output/'summary.json',dict(status='tune_diagnostic_complete',tasks=len(records),table=table,severe=severe,records=records,new_inference_calls=0,api_calls=0,method_changes=False,
        compact_prediction_files=sum(len(r['compact_predictions']) for r in records),exact_metrics_matches=sum(e['exact_metrics_match'] for r in records for e in r['compact_predictions']),original_pool_files_preserved=True))
    lines=['# N9 tune 原始SAM到R1候选池覆盖诊断','', '仅298 tune，准确缓存读取，无新推理/调参；候选固定后GT仅评分。下列覆盖不是最终检索召回，各几何原因可重叠，不可相加或当作单因素消融。','', '| 类型 | 分组 | 任务/图 | raw覆盖 | R1覆盖 | clean覆盖 | R1损失像素 | 清理损失像素 |','|---|---|---|---|---|---|---|---|']
    for t in table:lines.append('| '+' | '.join(map(str,[t['task_type'],t['group'],f"{t['tasks']}/{t['images']}",*[f"{t['coverage_recall'][k]:.4%}" for k in ('raw','original','clean')],t['geometry_loss'],t['cleaning_loss']]))+' |')
    lines+=['','## 严重R1损失（每类型/分组最多3例）','']
    for r in severe:lines += [f"- {r['id']}：raw/R1/clean GT覆盖 {r['coverage']}；GT {r['gt_pixels']}；过滤原因候选计数 {r['filter_reason_counts']}。各原因涉及的R1丢失GT（非加和）：{ {k:v['gt_lost_from_original_covered_by_rejected_reason'] for k,v in r['reason_evidence'].items()} }。证据：[inference]({r['inference_path']})；[逐候选诊断]({a.output/'tasks'/r['id']/'diagnostic.json'})。"]
    write_text(a.output/'report.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
