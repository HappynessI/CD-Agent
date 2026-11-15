"""Zero-API fixed geometry routing, selected from tune diagnostics only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_image,write_json
from .missed_sources import load_npz,unpack,specs,rows_for,pixel_counts
from .missed_verifier_study import metrics
from .night_candidate_geometry import geometry,base_geometry
from .night_missed_views import answer_path

REPAIR=('internal_hole','adjacent_edge_fragment')


def load(p):return json.loads(Path(p).read_text())


def route(group,stratum,task_variant):
    return 'V0' if group=='second_appeared' and stratum in REPAIR else task_variant


def freeze(tune):
    primary=tune['groups']['second_appeared']['10']
    if not primary['complete'] or primary['delta_vs_task']<=0:raise ValueError('No positive complete tune K10 gain; do not freeze')
    return dict(status='frozen',primary_k=10,descriptive_k=20,affected_group='second_appeared',strata=list(REPAIR),
        route='V0 for affected geometry; otherwise original frozen task variant',
        task_selection=tune['task_selection'],tune_gain=primary['delta_vs_task'],confirmation_used=False,
        limitation='Secondary exploratory; geometry rule derived from tune, prior all-arm confirmation aggregates already exposed')


def evaluate(a):
    task=load(a.views/'selected_verifier_by_task.json');full=load(a.views/'score_summary.json')
    if a.split=='confirm':
        frozen=load(a.out/'selection.json')
        if frozen['task_selection']!=task:raise ValueError('Frozen task policy differs')
    rows=[r for r in full['images'] if r['split']==a.split]
    manifests={(r['group'],r['stem']):r for r in load(a.views/'manifest.json')}
    spec=next(s for s in specs(a.run,a.work) if s['group']=='second_appeared')
    lookup={Path(r['id']).stem:r for r in rows_for(a.run,spec)}
    records=[]
    for row in rows:
        group,stem=row['group'],row['stem'];variant=task['groups'][group]['variant']
        record=dict(group=group,stem=stem,baseline=row['baseline'],budgets={})
        if group!='second_appeared':
            for k in ('10','20'):
                record['budgets'][k]=dict(metrics=row['variants'][variant][k]['metrics'],task=row['variants'][variant][k]['metrics'],
                    v0=row['variants']['V0'][k]['metrics'],routed_candidates=0,missing=0)
        else:
            item=manifests[(group,stem)];base=load_npz(a.work/'base'/group/(stem+'.npz'))['start'];context=base_geometry(base)
            z=load_npz(Path(item['masks']));masks=list(unpack(z['masks'],tuple(z['shape'])))
            truth=read_image(lookup[stem]['gt_mask'])>0;outcomes=[]
            for candidate,mask in zip(item['candidates'],masks):
                geo=geometry(mask,base,context);chosen=route(group,geo['stratum'],variant)
                path=answer_path(argparse.Namespace(pilot=a.pilot,out=a.views),chosen,candidate['uid'])
                answer=load(path) if path.exists() else None
                outcomes.append(dict(uid=candidate['uid'],geometry=geo,variant=chosen,answer=answer.get('answer') if answer else None,
                    missing=answer is None,path=str(path)))
            record['candidates']=outcomes
            for k in (10,20):
                pred=base.copy();routed=0;missing=0
                for o,mask in zip(outcomes[:k],masks[:k]):
                    routed+=o['variant']!=variant;missing+=o['missing']
                    if o['answer']=='yes':pred|=mask
                record['budgets'][str(k)]=dict(metrics=metrics(pixel_counts(pred,truth)) if not missing else None,
                    task=row['variants'][variant][str(k)]['metrics'],v0=row['variants']['V0'][str(k)]['metrics'],
                    routed_candidates=routed,missing=missing)
        records.append(record);write_json(a.out/a.split/'images'/group/(stem+'.json'),record)
    groups={}
    for group in sorted({r['group'] for r in records}):
        group_rows=[r for r in records if r['group']==group];groups[group]={}
        for k in ('10','20'):
            if any(not r['budgets'][k]['metrics'] for r in group_rows):groups[group][k]=dict(complete=False);continue
            after=np.asarray([[r['budgets'][k]['metrics'][n] for n in ('tp','fp','fn')] for r in group_rows])
            entry=dict(complete=True,metrics=metrics(after.sum(axis=0)),images=len(group_rows),
                routed_candidates=sum(r['budgets'][k]['routed_candidates'] for r in group_rows))
            for control in ('task','v0','initial'):
                before=np.asarray([[(r['baseline'] if control=='initial' else r['budgets'][k][control])[n] for n in ('tp','fp','fn')] for r in group_rows])
                delta=metrics(after.sum(axis=0))['f1']-metrics(before.sum(axis=0))['f1']
                rng=np.random.default_rng(1016);draws=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(group_rows),len(group_rows));draws.append(metrics(after[ix].sum(axis=0))['f1']-metrics(before[ix].sum(axis=0))['f1'])
                entry['delta_vs_'+control]=delta;entry['paired95_vs_'+control]=np.quantile(draws,[.025,.975]).tolist()
                entry['pixel_delta_vs_'+control]=dict(zip(('tp','fp','fn'),map(int,after.sum(axis=0)-before.sum(axis=0))))
            groups[group][k]=entry
    report=dict(split=a.split,task_selection=task,groups=groups,images=records,api_requests=0,
        rule='SECOND appeared internal_hole/adjacent_edge_fragment use V0; all other candidates retain frozen task variant',
        geometry_uses_gt=False,gt_role='offline score only',confirmation_status='exploratory previously exposed; no confirm selection')
    write_json(a.out/a.split/'summary.json',report)
    lines=['# 固定几何路由：'+a.split,'','零新增API；K10主指标，K20描述；确认集历史曝光，不用于改规则。','',
        '|组|K|完整|F1/P/R|Δ原任务|95%CI|ΔTP/FP/FN vs任务|Δ初始|ΔV0|路由数|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for group,budgets in groups.items():
        for k,v in budgets.items():
            if not v['complete']:lines.append(f'|{group}|{k}|False|待完成|—|—|—|—|—|—|');continue
            lines.append('|'+ '|'.join((group,k,'True','/'.join(f"{v['metrics'][x]:.4f}" for x in ('f1','precision','recall')),
                f"{v['delta_vs_task']:.4f}",str(v['paired95_vs_task']),str(v['pixel_delta_vs_task']),
                f"{v['delta_vs_initial']:.4f}",f"{v['delta_vs_v0']:.4f}",str(v['routed_candidates'])))+'|')
    (a.out/a.split/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['evaluate','freeze']);p.add_argument('--split',choices=['tune','confirm'],default='tune')
    for key in ('views','pilot','work','run','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    if a.command=='evaluate':evaluate(a)
    else:
        selection=freeze(load(a.out/'tune/summary.json'));path=a.out/'selection.json'
        if path.exists() and load(path)!=selection:raise ValueError('Frozen route differs')
        write_json(path,selection)


if __name__=='__main__':main()
