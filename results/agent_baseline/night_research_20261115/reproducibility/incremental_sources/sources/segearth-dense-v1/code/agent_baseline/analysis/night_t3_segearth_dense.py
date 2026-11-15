"""Cached SegEarth dense retrieval: historical routeA/B0 reproduction, new B2/B3 tune."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.exemplar_semantics import CLASSES
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import load_dev,semantic_route,pixel_counts,pooled,group
from .t3_routes_cache import read_npz,write_npz

METHODS=('routeA_historical','B0_historical','B2_dense','B3_dense')


def directed_dense(maps,pair,exemplar):
    if None in pair or pair[0]==pair[1]:return np.zeros_like(exemplar,dtype=bool)
    return (maps[0]==pair[0])&(maps[1]==pair[1])&~exemplar


def aggregate(records):
    output=[]
    for support in ('all','positive','empty','remaining_gt>=16_diagnostic'):
        for label in ('全部','目标类','地表类'):
            chosen=[r for r in records if (label=='全部' or r['group']==label) and
                (support=='all' or (support=='positive' and r['remaining_gt_pixels']>0) or
                 (support=='empty' and r['remaining_gt_pixels']==0) or
                 (support=='remaining_gt>=16_diagnostic' and r['remaining_gt_pixels']>=16))]
            methods={}
            for method in METHODS:
                values=[v for r in chosen for v in r['scores'] if v['method']==method]
                methods[method]=dict(pooled(values),empty_prediction_rate=sum(v['counts'][0]+v['counts'][1]==0 for v in values)/len(values) if values else None)
            output.append(dict(split='tune',support=support,group=label,tasks=len(chosen),methods=methods))
    return output


def checkpoint(output,records):
    result=dict(tasks=len(records),expected=30,status='complete' if len(records)==30 else 'partial',records=records,summary=aggregate(records),
        historical_reproduction=dict(routeA=sum(r['routeA_reproduced'] for r in records),B0=sum(r['B0_reproduced'] for r in records)),api_calls=0,new_model_calls=0,new_cost_cny=0,
        interpretation='routeA and B0 dense are historical methods reproduced from cached probabilities; only B2/B3 parsed-pair dense masks are new. Offline fixed tune30, no model input GT, no online Agent run.')
    write_json(output/'summary.json',result)
    lines=['# Cached SegEarth dense parsed-pair comparison','',result['interpretation'],
        f"Status: {result['status']}, {len(records)}/30. Reproduction: {result['historical_reproduction']}.",
        'Common seven-channel argmax retains background. No threshold/class/model changes. Unknown or same-class VLM pairs give empty predictions. Exemplar is removed from prediction and truth. The >=16 remaining-total-GT subset is sensitivity only; the original MIN_AREA=16 was a component construction threshold. B2 sees both dates, so differences cannot be called a pure spotlight intervention.','',
        '| GT | Group | N | Method | F1 | P | R | TP | FP | FN | Empty prediction rate |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['summary']:
        for method,m in row['methods'].items():
            lines.append('| '+' | '.join(map(str,[row['support'],row['group'],row['tasks'],method,m['f1_c'],m['precision'],m['recall'],m['tp'],m['fp'],m['fn'],m['empty_prediction_rate']]))+' |')
    lines+=['','Per-task masks, pairs, counts, source paths and exact old/new reproduction counts are retained under tasks/. Missing caches are errors, never zero predictions or omitted cases. No configuration is selected by this report.']
    write_text(output/'report.md','\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True);p.add_argument('--old-bridge',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Cached array work requires Slurm')
    out=a.night/'segearth_dense_tune';audit=json.loads(read_text(a.night/'audit.json'))
    rows=[r for r in load_dev(a.run) if audit['split'][r['id']]=='tune'];ids={r['id'] for r in rows}
    if len(rows)!=30:raise ValueError('Expected original tune30')
    old_a=json.loads(read_text(a.historical/'route_a/progress.json'))
    if old_a['policy']['classes']!=list(CLASSES.values()):raise ValueError('Historical vocabulary/order differs')
    old_a={r['id']:r for r in old_a['tasks']}
    old_b={r['id']:r for r in json.loads(read_text(a.old_bridge))['per_task']}
    parses={v:{r['id']:r for r in json.loads(read_text(a.night/v/'summary.json'))['records'] if r['split']=='tune'} for v in ('B0','B2','B3')}
    if any(set(v)!=ids for v in parses.values()):raise ValueError('Parse tune cohorts differ')
    settings=dict(code=str(Path(__file__).resolve()),split='fixed tune30',classes=list(CLASSES.values()),background_channel=0,
        cache=str(a.historical/'route_a/semantic'),historical_routeA=str(a.historical/'route_a/progress.json'),historical_B0=str(a.old_bridge),
        routeA='original semantic_route: foreground mean probability over example; dense argmax includes background',
        VLM='existing B0/B2/B3 predicted pair; argmax(T1)==source AND argmax(T2)==target; unknown/same empty',
        selection='no confidence thresholds, morphological filters, GT-driven class corrections or new models',
        new_scope='B2/B3 only; B0 bridge_dense and routeA already evaluated historically, now reproducibility references',api_calls=0,new_model_calls=0)
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Settings differ')
    write_json(out/'settings.json',settings)
    missing=[str(a.historical/'route_a/semantic'/f"{row['id']}_{t}.npz") for row in rows for t in ('t1','t2') if not exists(a.historical/'route_a/semantic'/f"{row['id']}_{t}.npz")]
    write_json(out/'cache_inventory.json',dict(expected=60,missing=missing,model_fill_allowed=False))
    if missing:raise RuntimeError('Missing existing probability cache; see cache_inventory.json')
    records=[]
    for row in rows:
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        ex=read_image(row['exemplar_mask'])>0
        paths=[a.historical/'route_a/semantic'/f"{row['id']}_{t}.npz" for t in ('t1','t2')]
        probabilities=[read_npz(path)['data'] for path in paths]
        pair_a,pred_a=semantic_route(probabilities,ex) # validates shape and finite probabilities
        maps=[p.argmax(axis=0) for p in probabilities]
        pairs={'routeA_historical':pair_a};predictions={'routeA_historical':pred_a}
        for variant in ('B0','B2','B3'):
            name=variant+'_historical' if variant=='B0' else variant+'_dense'
            pair=parses[variant][row['id']]['interpretation']['pair'];pairs[name]=pair
            predictions[name]=directed_dense(maps,pair,ex)
        if pairs['B0_historical']!=old_b[row['id']]['interpretation']['pair']:raise ValueError('Reused B0 parsing differs from historical record')
        write_npz(folder/'predictions.npz',shape=np.asarray(ex.shape),methods=np.asarray(METHODS),
            masks=np.packbits(np.stack([predictions[k].ravel() for k in METHODS]),axis=1))
        # Ground truth enters only after all masks have been fixed and persisted.
        truth=read_image(row['gt_mask'])>0
        scores=[dict(method=k,k=0,counts=pixel_counts(predictions[k],truth,ex)) for k in METHODS]
        expected_a=old_a[row['id']]['counts'];expected_b=next(r['counts'] for r in old_b[row['id']]['records'] if r['method']=='bridge_dense' and r['k']==0)
        reproduced_a=pair_a==old_a[row['id']]['predicted_pair'] and scores[0]['counts']==expected_a
        reproduced_b=scores[1]['counts']==expected_b
        reproduction=dict(routeA=reproduced_a,B0=reproduced_b,old_routeA=expected_a,new_routeA=scores[0]['counts'],old_B0=expected_b,new_B0=scores[1]['counts'])
        write_json(folder/'reproduction.json',reproduction)
        if not reproduced_a or not reproduced_b:raise ValueError('Historical dense reproduction failed')
        record=dict(id=row['id'],split='tune',group=group(row),remaining_gt_pixels=int((truth&~ex).sum()),pairs=pairs,scores=scores,
            routeA_reproduced=reproduced_a,B0_reproduced=reproduced_b,probability_sources=list(map(str,paths)),exemplar_source=row['exemplar_mask'],api_calls=0)
        write_json(final,record);records.append(record);checkpoint(out,records);print('dense cached',len(records),row['id'],flush=True)
    checkpoint(out,records)


if __name__=='__main__':main()
