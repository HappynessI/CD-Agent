"""Post-primary precision/recall tradeoff: tune-optimal fixed thresholds, one confirmation."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import load_dev
from .t3_routes_cache import read_npz
from .n7_t3_rejection import RANKS,threshold_indices,evaluate,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    original=json.loads(read_text(a.out/'all_summary.json'))
    frozen=json.loads(read_text(a.out/'frozen.json'));out=a.out/'tradeoff_diagnostic'
    selection={rank:max(frozen[rank]['candidates'],key=lambda r:(r['f1_c'],-r['threshold'])) for rank in RANKS}
    thresholds={rank:selection[rank]['threshold'] for rank in RANKS}
    if thresholds!={'delta':.6,'joint':.8}:raise ValueError('Does not match separately declared fixed tradeoff thresholds')
    write_json(out/'selection.json',dict(thresholds=thresholds,tune_selection=selection,
        rule='maximum tune pooled all-task F1 from the already computed six thresholds; tie lower threshold; no TP constraint',
        scope='Post-primary exploratory tradeoff, prior confirm Top8 outcomes already exposed. Original >=95% TP primary rule unchanged. No online promotion.'))
    rows={r['id']:r for r in load_dev(a.run)};records=[]
    for source in original['records']:
        folder=out/'tasks'/source['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        record=dict(source);record['methods']=dict(source['methods'])
        if source['split']=='tune':
            for rank in RANKS:
                record['methods'][rank+'_tradeoff']=source['grid'][rank+'_threshold_'+str(thresholds[rank])]
        else:
            row=rows[source['id']];saved=read_npz(Path(source['source_prepared']))
            ranking=read_npz(Path(source['ranking_source']));masks=saved['r1'].astype(bool)
            orders={'delta':saved['order1'].tolist(),'joint':ranking['order_joint'].tolist()}
            values={'delta':ranking['delta'],'joint':ranking['equal_joint']}
            ex=read_image(row['exemplar_mask'])>0;truth=read_image(row['gt_mask'])>0
            for rank in RANKS:
                ids=threshold_indices(orders[rank],values[rank],thresholds[rank])
                record['methods'][rank+'_tradeoff']=evaluate(masks,ids,truth,ex)
        record['tradeoff_thresholds']=thresholds
        record['tradeoff_losses']={rank:dict(tp_lost=source['methods'][rank+'_top8']['counts'][0]-record['methods'][rank+'_tradeoff']['counts'][0],
            fp_removed=source['methods'][rank+'_top8']['counts'][1]-record['methods'][rank+'_tradeoff']['counts'][1]) for rank in RANKS}
        write_json(final,record);records.append(record)
        print('T3 fixed tradeoff',len(records),source['id'],flush=True)
    summaries=summarize(records)
    result=dict(tasks=len(records),records=records,summary=summaries,selection=selection,api_calls=0,api_cost_cny=0,
        interpretation='Exploratory recall/precision tradeoff after the primary experiment; original >=95% TP criterion remains unchanged. No threshold sweep on confirm, no verifier or route change, no automatic online advancement.')
    write_json(out/'summary.json',result)
    lines=['# T3 fixed recall–precision tradeoff diagnostic','',result['interpretation'],
           'Selected on the already computed tune grid: delta=0.6, joint=0.8. Confirm Top8 and primary-gate results were already exposed before this supplemental declaration. This is not a fresh independent confirmation.', '',
           '| Split | Support | Group | N | Method | F1 | P | R | TP | FP | FN | Empty | Positive fully rejected | TP retention vs Top8 |',
           '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        for name,m in r['methods'].items():
            if not name.endswith(('_tradeoff','_top8')):continue
            lines.append('| '+' | '.join(map(str,[r['split'],r['support'],r['group'],r['tasks'],name,*[round(m[k],5) for k in ('f1_c','precision','recall')],m['tp'],m['fp'],m['fn'],m['empty_predictions'],m['positive_fully_rejected'],m.get('tp_retention_vs_top8')]))+' |')
    lines+=['','All per-task losses are retained in metrics.json. Higher pooled F1 alone does not erase lost nonempty TP or establish an acceptable operating point. This diagnostic asks whether the original high-recall constraint hid a real precision/recall tradeoff. It does not replace the original primary hypothesis.']
    write_text(out/'report.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
