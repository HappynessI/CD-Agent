"""Fixed B2 SAM/dense mask overlap diagnosis, no new inference or policy selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import load_dev,group,pixel_counts,pooled
from .t3_routes_cache import read_npz,write_npz

METHODS=('SAM_only','dense_only','intersection','union')


def regions(sam,dense):
    return dict(SAM_only=sam&~dense,dense_only=dense&~sam,intersection=sam&dense,union=sam|dense)


def unpack(saved,key):
    shape=tuple(map(int,saved['shape']))
    return np.unpackbits(saved[key],axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)


def summarize(records):
    rows=[]
    for support in ('all','positive','empty'):
        for label in ('全部','目标类','地表类'):
            selected=[r for r in records if (label=='全部' or r['group']==label) and
                (support=='all' or (r['remaining_gt_pixels']>0)==(support=='positive'))]
            rows.append(dict(support=support,group=label,tasks=len(selected),methods={method:pooled([v for r in selected for v in r['scores'] if v['method']==method]) for method in METHODS}))
    return rows


def checkpoint(out,records,pending,split):
    data=dict(split=split,tasks=len(records),expected=30,status='complete' if len(records)==30 else ('partial' if records else 'pending'),pending=pending,records=records,
        summary=summarize(records),api_calls=0,new_model_calls=0,
        interpretation='Fixed B2 SAM positive-only full-union mask versus B2 SegEarth dense mask. Disjoint SAM-only/dense-only/intersection and fixed union are descriptive regions, not GT-selected deployment policies. GT only scores these fixed masks. FN for each region is relative to the full remaining GT; regional FN counts must not be added.',
        scope='Original split retained; available-case tables are partial until30/30, never a reduced benchmark. Confirmation images were multiply exposed. No online Agent evaluation.')
    write_json(out/'summary.json',data)
    lines=['# B2 fixed mask complementarity', '',data['interpretation'],data['scope'],
        f"{split}: {data['status']} {len(records)}/30; pending {len(pending)}.",'',
        '| GT | Group | Available N | Region | F1 | P | R | TP | FP | FN |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in data['summary']:
        for method,m in row['methods'].items():lines.append('| '+' | '.join(map(str,[row['support'],row['group'],row['tasks'],method,m['f1_c'],m['precision'],m['recall'],m['tp'],m['fp'],m['fn']]))+' |')
    lines+=['','Per-case regions and counts remain under tasks/. The union is specified before scoring and is not chosen per case from GT. No claim of calibrated abstention or deployable combination follows from overlap counts. Pending cases are listed in summary.json.']
    write_text(out/'report.md','\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--split',choices=['tune','confirm','all'],default='all');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Mask diagnostics require Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));allrows=load_dev(a.run)
    for split in (('tune','confirm') if a.split=='all' else (a.split,)):
        rows=[r for r in allrows if audit['split'][r['id']]==split]
        if len(rows)!=30:raise ValueError('Require original split30')
        out=a.night/'b2_mask_complementarity'/split;records=[];pending=[]
        write_json(out/'settings.json',dict(code=str(Path(__file__).resolve()),split=split,variant='B2',SAM='positive-only full candidate union, not Top8',dense='original seven-channel directed argmax mask',
            regions=list(METHODS),prediction_exemplar_removed=True,GT_usage='offline scoring only',selection='no per-case GT selection, no threshold change or new configuration',api_calls=0))
        for row in rows:
            folder=out/'tasks'/row['id'];final=folder/'metrics.json'
            if exists(final):records.append(json.loads(read_text(final)));continue
            samfolder=a.night/f'positive_endpoints_{split}'/'B2/tasks'/row['id']
            densefolder=a.night/f'segearth_dense_{split}'/'tasks'/row['id']
            paths=[samfolder/'proposals.npz',samfolder/'metrics.json',densefolder/'predictions.npz',densefolder/'metrics.json']
            missing=[str(p) for p in paths if not exists(p)]
            if missing:
                item=dict(id=row['id'],status='pending_sources',missing=missing);pending.append(item)
                write_json(folder/'pending.json',item);checkpoint(out,records,pending,split);continue
            ex=read_image(row['exemplar_mask'])>0
            saved_s=read_npz(paths[0]);saved_d=read_npz(paths[2])
            if tuple(saved_s['shape'])!=ex.shape or tuple(saved_d['shape'])!=ex.shape:raise ValueError('Cached prediction coordinates differ')
            stack=unpack(saved_s,'positive_only');sam=(stack.any(axis=0) if len(stack) else np.zeros_like(ex))&~ex
            methods=saved_d['methods'].tolist();dense=unpack(saved_d,'masks')[methods.index('B2_dense')]&~ex
            parts=regions(sam,dense)
            write_npz(folder/'regions.npz',shape=np.asarray(ex.shape),methods=np.asarray(METHODS),masks=np.packbits(np.stack([parts[k].ravel() for k in METHODS]),axis=1))
            # Masks above do not use GT. Existing counts are checked only afterward.
            truth=read_image(row['gt_mask'])>0
            old_s=json.loads(read_text(paths[1]));old_d=json.loads(read_text(paths[3]))
            expected_s=next(r['counts'] for r in old_s['scores'] if r['method']=='positive_only_all')
            expected_d=next(r['counts'] for r in old_d['scores'] if r['method']=='B2_dense')
            actual_s=pixel_counts(sam,truth,ex);actual_d=pixel_counts(dense,truth,ex)
            if actual_s!=expected_s or actual_d!=expected_d:raise ValueError('Saved prediction differs from original source counts')
            record=dict(id=row['id'],split=split,group=group(row),remaining_gt_pixels=int((truth&~ex).sum()),scores=[dict(method=k,counts=pixel_counts(parts[k],truth,ex)) for k in METHODS],
                source_counts=dict(SAM=actual_s,dense=actual_d),source_reproduced=True,sources=list(map(str,paths)),api_calls=0)
            write_json(final,record);records.append(record)
            if exists(folder/'pending.json'):write_json(folder/'pending.json',dict(id=row['id'],status='resolved',metrics=str(final)))
            checkpoint(out,records,pending,split);print('complementarity',split,len(records),row['id'],flush=True)
        checkpoint(out,records,pending,split)


if __name__=='__main__':main()
