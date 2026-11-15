"""Secondary exploratory task-conditioned verifier; tune freezes all K."""
import argparse
import json
from pathlib import Path
from .io_retry import write_json

VARIANTS=('V0','V1','V2','V3','V4')


def choose(report):
    groups={key[:-5]:value for key,value in report['groups'].items() if key.endswith('/tune')}
    if len(groups)!=3:raise ValueError('Require three complete tuning groups')
    selected={}
    for group,arms in groups.items():
        values={}
        for variant in VARIANTS:
            entry=arms[variant]['10']
            if not entry.get('complete'):raise ValueError('Incomplete tuning arm')
            values[variant]=entry['metrics']['f1']
        selected[group]=dict(variant=max(VARIANTS,key=values.get),tune_k10_f1=values)
    return dict(status='frozen',analysis='secondary exploratory; historical exposure disclosed',
        selection_k=10,applied_k=[3,5,10,20],groups=selected,confirmation_used=False,
        rule='Per group maximum K10 tune pooled F1; ties V0,V1,V2,V3,V4; same variant for all K',
        global_selection_unchanged=True,online_status='diagnostic only; not promoted')


def score(decision,report):
    groups={};images=[]
    for group,selection in decision['groups'].items():
        variant=selection['variant'];source=report.get('groups',{}).get(group+'/confirm',{}).get(variant,{})
        groups[group]=dict(variant=variant,confirmation={str(k):source.get(str(k),dict(complete=False,status='pending')) for k in decision['applied_k']})
    for row in report.get('images',[]):
        if row['split']!='confirm' or row['group'] not in decision['groups']:continue
        variant=decision['groups'][row['group']]['variant']
        images.append(dict(group=row['group'],stem=row['stem'],variant=variant,baseline=row['baseline'],
                           selected=row['variants'][variant],v0=row['variants']['V0']))
    return dict(analysis=decision['analysis'],selection=decision,groups=groups,images=images,
        complete=all(v.get('complete',False) for group in groups.values() for v in group['confirmation'].values()),
        comparison='Existing same-image paired scores vs initial and V0; existing bootstrap intervals retained; no new requests',
        policy='Confirmation evaluates frozen task variants; never reselect variant or K; no online claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','score'])
    p.add_argument('--report',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();report=json.loads(a.report.read_text())
    if a.command=='select':
        decision=choose(report)
        if a.selection.exists() and json.loads(a.selection.read_text())!=decision:raise ValueError('Frozen task choice differs')
        write_json(a.selection,decision)
    else:
        if a.output is None:raise ValueError('Score needs separate output')
        write_json(a.output,score(json.loads(a.selection.read_text()),report))


if __name__=='__main__':main()
