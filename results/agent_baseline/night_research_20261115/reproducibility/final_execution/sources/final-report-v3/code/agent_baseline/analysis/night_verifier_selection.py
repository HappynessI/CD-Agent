"""Freeze the predeclared tuning-only verifier choice before confirmation."""
import argparse
import json
from pathlib import Path
from .io_retry import write_json

VARIANTS=('V0','V1','V2','V3','V4')


def choose(report):
    groups={name:row for name,row in report['groups'].items() if name.endswith('/tune')}
    if len(groups)!=3:
        raise ValueError('Require all three complete tuning groups')
    scores={}
    for variant in VARIANTS:
        effects={}
        for name,row in groups.items():
            reference,current=row['V0']['10'],row[variant]['10']
            if not reference['complete'] or not current['complete']:
                raise ValueError('All five tuning arms must be complete before selection')
            effects[name]=current['metrics']['f1']-reference['metrics']['f1']
        scores[variant]=dict(group_delta_f1_vs_V0=effects,mean_delta_f1=sum(effects.values())/len(effects))
    winner=max(VARIANTS,key=lambda variant:scores[variant]['mean_delta_f1'])
    return dict(selected_verifier=winner,k=10,scores=scores,
        rule='Largest equal-weight mean tuning-group pooled F1 improvement over V0 at fixed K10; ties preserve V0,V1,V2,V3,V4 order',
        confirmation_used=False,online_status='not_promoted; confirmation and mechanism review still required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();decision=choose(json.loads(a.report.read_text()))
    if a.output.exists() and json.loads(a.output.read_text())!=decision:
        raise ValueError('Existing frozen verifier choice differs; do not overwrite it')
    write_json(a.output,decision)


if __name__=='__main__':main()
