"""N8 T3 whitelist-only geometry/features plus offline reporting helpers."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from agent_baseline.tools_t13 import Features, pooled as feature_pool, pool_features
from agent_baseline.instances import from_dense
from .night_t3_endpoint import representations
from .io_retry import read_text,read_image,write_json,write_text
from .t3_routes import pooled,pixel_counts,semantic_route
from .t3_diag import proposal_metrics
from .t3_routes_cache import read_npz,write_npz

CHANNELS=['background','low vegetation','non-vegetated ground','tree','water','building','playground']
K_VALUES=(1,3,5,8)
RANKS=('delta','joint')


def clean_pool(masks,example):
    kept=[];ids=[];audit=[];earea=int(example.sum())
    for index,mask in enumerate(masks):
        area=int(mask.sum());intersection=int((mask&example).sum())
        overlap=intersection/max(area,1);iou=intersection/max(area+earea-intersection,1)
        net=mask&~example
        reason='overlap_or_iou' if overlap>=.15 or iou>=.15 else ('net_area_below16' if intersection and net.sum()<16 else None)
        audit.append(dict(original_id=index,raw_area=area,intersection=intersection,overlap=overlap,iou=iou,
                          net_area=int(net.sum()),removed=reason,modified=bool(intersection)))
        if reason is None and net.any():kept.append(net);ids.append(index)
    return np.asarray(kept,bool).reshape(-1,*example.shape),ids,audit


def packed_save(path,masks,**arrays):
    write_npz(path,masks=np.packbits(masks.reshape(len(masks),-1),axis=1) if len(masks) else np.zeros((0,(int(np.prod(masks.shape[1:]))+7)//8),np.uint8),
              shape=np.asarray(masks.shape[1:]),**arrays)


def packed_load(path):
    saved=read_npz(path);shape=tuple(map(int,saved['shape']))
    return np.unpackbits(saved['masks'],axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool),saved


def ranked_features(masks,example,backend):
    features=[Features(backend('extract_features',time=t),example.shape,t) for t in ('t1','t2')]
    examples=[pool_features(f,np.flatnonzero(example)) for f in features]
    if len(masks):
        items=from_dense(masks,np.ones(len(masks)));candidates=[feature_pool(f,items) for f in features]
        rep=representations(*candidates,*examples)
        values={'delta':rep['delta'],'joint':rep['equal_joint']}
    else:values={r:np.zeros(0,np.float32) for r in RANKS}
    orders={r:sorted(range(len(masks)),key=lambda i:(-float(values[r][i]),i)) for r in RANKS}
    return values,orders


def observable_example(probabilities,example):
    pair,dense=semantic_route(probabilities,example)
    means=[p[:,example].mean(axis=1).tolist() for p in probabilities]
    yy,xx=np.nonzero(example);area=int(example.sum());bbox_area=int((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1))
    return dict(channels=CHANNELS,predicted_pair=pair,example_mean_probabilities=means,area=area,bbox_fill=area/bbox_area,
                structural=bool(5 in pair or 6 in pair)),dense


def connected_masks(mask):
    labels,n=ndi.label(mask,np.ones((3,3),bool))
    return np.asarray([labels==i for i in range(1,n+1)],bool).reshape(-1,*mask.shape)


def evaluate(masks,indices,truth,example):
    indices=list(map(int,indices));pred=masks[indices].any(axis=0) if indices else np.zeros_like(example)
    return dict(counts=pixel_counts(pred,truth,example),candidate_ids=indices,coverage=proposal_metrics(masks[indices],truth,example))


def mask_score(pred,truth,example):
    return evaluate(np.asarray([pred],bool),[0],truth,example)


def aggregates(records):
    result=[]
    for split in ('tune','confirm','all'):
        for support in ('all','positive','empty','remaining_gt>=16_diagnostic'):
            for label in ('全部','目标类','地表类'):
                rows=[r for r in records if (split=='all' or r['split']==split) and (label=='全部' or r['group']==label)
                      and (support=='all' or support=='positive' and r['remaining_gt_pixels']>0 or support=='empty' and r['remaining_gt_pixels']==0
                           or support=='remaining_gt>=16_diagnostic' and r['remaining_gt_pixels']>=16)]
                if not rows:continue
                methods={}
                for method in sorted(set.intersection(*(set(r['methods']) for r in rows))):
                    vals=[r['methods'][method] for r in rows];m=pooled(vals);base=pooled([r['methods']['original_R1_top8'] for r in rows])
                    m.update(macro_f1=float(np.mean([pooled([v])['f1_c'] for v in vals])),
                        empty_predictions=sum(sum(v['counts'][:2])==0 for v in vals),
                        empty_prediction_rate=sum(sum(v['counts'][:2])==0 for v in vals)/len(vals),
                        positive_fully_rejected=sum(r['remaining_gt_pixels']>0 and sum(r['methods'][method]['counts'][:2])==0 for r in rows),
                        tp_lost_vs_original=base['tp']-m['tp'],fp_removed_vs_original=base['fp']-m['fp'],
                        tp_retention=m['tp']/base['tp'] if base['tp'] else None,
                        tp_by_task={r['id']:r['methods'][method]['counts'][0] for r in rows if r['methods'][method]['counts'][0]})
                    if all('coverage' in v for v in vals):
                        covered=sum(v['coverage']['covered_pixels'] for v in vals);total=sum(v['coverage']['truth_pixels'] for v in vals)
                        m.update(covered_pixels=covered,truth_pixels=total,coverage_recall=covered/total if total else None,
                            aligned_instances=sum(v['coverage']['aligned_instances'] for v in vals),gt_instances=sum(v['coverage']['gt_instances'] for v in vals))
                    methods[method]=m
                result.append(dict(split=split,support=support,group=label,tasks=len(rows),methods=methods))
    return result


def report(out,records,stage,**extra):
    summary=aggregates(records);result=dict(stage=stage,tasks=len(records),records=records,summary=summary,**extra)
    write_json(out/'summary.json',result)
    lines=[f'# N8 T3 {stage}','', 'Original dev60 and split are unchanged. GT is used only for offline scoring/isolated tune selection. Development images have prior exposure.', '',
        '| Split | Support | Group | N | Method | F1 | P | R | TP | FP | FN | Empty predictions | Positive fully rejected |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        for name,m in row['methods'].items():
            lines.append('| '+' | '.join(map(str,[row['split'],row['support'],row['group'],row['tasks'],name,*[round(m[k],5) for k in ('f1_c','precision','recall')],m['tp'],m['fp'],m['fn'],m['empty_predictions'],m['positive_fully_rejected']]))+' |')
    write_text(out/'report.md','\n'.join(lines)+'\n')
