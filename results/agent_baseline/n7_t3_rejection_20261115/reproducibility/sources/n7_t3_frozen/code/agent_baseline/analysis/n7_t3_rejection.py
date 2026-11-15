"""N7: fixed-pool T3 rejection; tune-only selection, separately invoked confirmation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text, read_image, write_json, write_text, exists
from .t3_routes import load_dev, pooled, pixel_counts, group
from .t3_routes_cache import read_npz
from .t3_diag import proposal_metrics

GRID = (-1.0, 0.0, 0.2, 0.4, 0.6, 0.8)
RANKS = ('delta', 'joint')


def threshold_indices(order, values, threshold):
    # Keep the exact Top8 budget; filtering never backfills lower-ranked candidates.
    return [int(i) for i in order[:8] if float(values[i]) >= threshold]


def select_threshold(records, rank):
    if not records or any(r['split'] != 'tune' for r in records):
        raise ValueError('Selection requires tune records exclusively')
    baseline = sum(r['methods'][rank+'_top8']['counts'][0] for r in records)
    candidates = []
    for threshold in GRID:
        key = rank+'_threshold_'+str(threshold)
        metrics = pooled([r['grid'][key] for r in records])
        candidates.append(dict(threshold=threshold, **metrics,
            tp_retention=metrics['tp']/baseline if baseline else None,
            eligible=metrics['tp'] >= .95*baseline))
    selected = max((v for v in candidates if v['eligible']), key=lambda v:(v['f1_c'], -v['threshold']))
    return dict(threshold=selected['threshold'], baseline_tp=baseline, candidates=candidates,
                rule='maximize tune all-task micro F1 subject to >=95% baseline Top8 TP; tie lower threshold')


def evaluate(masks, indices, truth, exemplar):
    pred = masks[indices].any(axis=0) if indices else np.zeros_like(exemplar)
    return dict(counts=pixel_counts(pred, truth, exemplar), candidate_ids=list(indices),
                coverage=proposal_metrics(masks[indices], truth, exemplar))


def summarize(records):
    result = []
    for split in ('tune','confirm','all'):
        for support in ('all','positive','empty','remaining_gt>=16_diagnostic'):
            for label in ('全部','目标类','地表类'):
                chosen = [r for r in records if (split=='all' or r['split']==split)
                          and (label=='全部' or r['group']==label) and
                          (support=='all' or support=='positive' and r['remaining_gt_pixels']>0 or
                           support=='empty' and r['remaining_gt_pixels']==0 or
                           support=='remaining_gt>=16_diagnostic' and r['remaining_gt_pixels']>=16)]
                if not chosen: continue
                methods = {}
                for method in sorted(set.intersection(*(set(r['methods']) for r in chosen))):
                    vals=[r['methods'][method] for r in chosen]
                    m=pooled(vals)
                    m.update(empty_predictions=sum(sum(v['counts'][:2])==0 for v in vals),
                        empty_prediction_rate=sum(sum(v['counts'][:2])==0 for v in vals)/len(vals),
                        positive_fully_rejected=sum(r['remaining_gt_pixels']>0 and sum(r['methods'][method]['counts'][:2])==0 for r in chosen),
                        zero_tp_positive=sum(r['remaining_gt_pixels']>0 and r['methods'][method]['counts'][0]==0 for r in chosen),
                        tp_by_task={r['id']:r['methods'][method]['counts'][0] for r in chosen if r['methods'][method]['counts'][0]})
                    if all('coverage' in v for v in vals):
                        cov=[v['coverage'] for v in vals]
                        m['covered_pixels']=sum(v['covered_pixels'] for v in cov)
                        m['coverage_recall']=m['covered_pixels']/max(1,sum(v['truth_pixels'] for v in cov))
                        m['aligned_instances']=sum(v['aligned_instances'] for v in cov)
                        m['gt_instances']=sum(v['gt_instances'] for v in cov)
                    if method.startswith(('delta_','joint_')):
                        rank=method.split('_')[0]
                        base=pooled([r['methods'][rank+'_top8'] for r in chosen])
                        m.update(tp_lost_vs_top8=base['tp']-m['tp'], fp_removed_vs_top8=base['fp']-m['fp'],
                                 tp_retention_vs_top8=m['tp']/base['tp'] if base['tp'] else None)
                    methods[method]=m
                result.append(dict(split=split,support=support,group=label,tasks=len(chosen),methods=methods))
    return result


def checkpoint(out, records, frozen, stage):
    summaries=summarize(records)
    result=dict(stage=stage,tasks=len(records),records=records,summary=summaries,frozen=frozen,
                api_calls=0,api_cost_cny=0,online_agent=False,
                interpretation='Multiply exposed dev60; original split/GT fixed. routeA is different-pool historical context, not same-pool ablation.')
    write_json(out/f'{stage}_summary.json',result)
    lines=['# N7 T3 fixed-pool rejection', '',result['interpretation'],
           'Threshold grid and 95% tune TP constraint were fixed before this run. Confirmation is never used for selection. Pure Top1 has no verifier. No dynamic routing or online Agent was run.', '',
           '| Split | Support | Group | N | Method | F1 | P | R | TP | FP | FN | Empty predictions | Positive fully rejected | TP lost vs Top8 |',
           '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        for name,m in r['methods'].items():
            lines.append('| '+' | '.join(map(str,[r['split'],r['support'],r['group'],r['tasks'],name,*[round(m[k],5) for k in ('f1_c','precision','recall')],m['tp'],m['fp'],m['fn'],m['empty_predictions'],m['positive_fully_rejected'],m.get('tp_lost_vs_top8','n/a')]))+' |')
    lines+=['','Threshold selection, full-pool coverage, candidate indices, per-task TP/FP/FN, TP concentration and rejection losses are retained in JSON. Empty-prediction rate on empty tasks is empty-set accuracy; on all/positive tasks it is only prediction frequency. Zero-call phase does not claim to test local verification.']
    write_text(out/f'{stage}_report.md','\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['tune','confirm'],required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for cached scoring')
    audit=json.loads(read_text(a.night/'audit.json'));rows=[r for r in load_dev(a.run) if audit['split'][r['id']]==a.split]
    if len(rows)!=30:raise ValueError('Expected original split30')
    frozen=json.loads(read_text(a.out/'frozen.json')) if a.split=='confirm' else None
    records=[]
    for row in rows:
        target=a.out/a.split/'tasks'/row['id']/'metrics.json'
        if exists(target):records.append(json.loads(read_text(target)));continue
        prior=a.night/'retrieval/B0/tasks'/row['id']
        old=json.loads(read_text(prior/'metrics.json'));saved=read_npz(prior/'prepared.npz');masks=saved['r1'].astype(bool)
        ranks=read_npz(a.night/'r1_endpoint'/a.split/'tasks'/row['id']/'ranking.npz')
        orders={'delta':saved['order1'].tolist(),'joint':ranks['order_joint'].tolist()}
        values={'delta':ranks['delta'],'joint':ranks['equal_joint']}
        if any(len(v)!=len(masks) for v in values.values()):raise ValueError('Candidate identities differ')
        ex=read_image(row['exemplar_mask'])>0;truth=read_image(row['gt_mask'])>0
        methods={'always_empty':evaluate(masks,[],truth,ex)};grid={}
        for rank in RANKS:
            for k in (8,1):methods[f'{rank}_top{k}']=evaluate(masks,orders[rank][:k],truth,ex)
            thresholds=GRID if a.split=='tune' else (frozen[rank]['threshold'],)
            for threshold in thresholds:
                value=evaluate(masks,threshold_indices(orders[rank],values[rank],threshold),truth,ex)
                grid[rank+'_threshold_'+str(threshold)]=value
                if a.split=='confirm':methods[rank+'_threshold']=value
        expected=next(v['counts'] for v in old['scores'] if v['method']=='R1_geometry_dino' and v['k']==8)
        if methods['delta_top8']['counts']!=expected:raise ValueError('Original delta Top8 reproduction failed')
        oldjoint=json.loads(read_text(a.night/'r1_endpoint'/a.split/'tasks'/row['id']/'metrics.json'))
        expected=next(v['counts'] for v in oldjoint['scores'] if v['method']=='R1_equal_joint')
        if methods['joint_top8']['counts']!=expected:raise ValueError('Original joint Top8 reproduction failed')
        dense=json.loads(read_text(a.night/f'segearth_dense_{a.split}'/'tasks'/row['id']/'metrics.json'))
        methods['routeA_historical']=next(v for v in dense['scores'] if v['method']=='routeA_historical')
        record=dict(id=row['id'],split=a.split,group=group(row),remaining_gt_pixels=int((truth&~ex).sum()),
                    methods=methods,grid=grid,geometry_proposal_upper=old['geometry_proposal_upper'],
                    source_prepared=str(prior/'prepared.npz'),ranking_source=str(a.night/'r1_endpoint'/a.split/'tasks'/row['id']/'ranking.npz'),
                    top8_scores={rank:[float(values[rank][i]) for i in orders[rank][:8]] for rank in RANKS},
                    top8_exemplar_overlap_pixels={rank:[int((masks[i]&ex).sum()) for i in orders[rank][:8]] for rank in RANKS})
        write_json(target,record);records.append(record)
        print('N7 rejection',a.split,len(records),row['id'],flush=True)
    if a.split=='tune':
        frozen={rank:select_threshold(records,rank) for rank in RANKS}
        write_json(a.out/'frozen.json',frozen)
        for r in records:
            for rank in RANKS:r['methods'][rank+'_threshold']=r['grid'][rank+'_threshold_'+str(frozen[rank]['threshold'])]
            write_json(a.out/'tune/tasks'/r['id']/'metrics.json',r)
    checkpoint(a.out,records,frozen,a.split)
    if a.split=='confirm':
        tune=json.loads(read_text(a.out/'tune_summary.json'))['records']
        checkpoint(a.out,tune+records,frozen,'all')


if __name__=='__main__':main()
