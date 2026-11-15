"""One frozen morphology follow-up after coverage failure, using cached states."""
import argparse
import os
import time
from pathlib import Path
from .n7_missed_trigger import read, write, summarize, pooled


def concentrated(state, shape):
    return state['mask_pixels']==0 or shape['largest_component_fraction']>=0.5


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    started=time.monotonic();out=a.root/'round2_concentration';out.mkdir(exist_ok=True)
    original=read(a.root/'paired_cases.json')
    states={(r['group'],r['stem']):r for r in read(a.root/'initial_shape_states.json')}
    lookup={(r['endpoint'],r['policy'],r['group'],r['stem']):r for r in original}
    rows=[]
    for r in original:
        if r['policy']!='always': continue
        state=states[r['group'],r['stem']]
        call=concentrated(state['state'],state['shape_features'])
        chosen=lookup[r['endpoint'],'always' if call else 'never',r['group'],r['stem']]
        coverage=lookup[r['endpoint'],'coverage',r['group'],r['stem']]
        row=dict(chosen,policy='concentration',shape_features=state['shape_features'],
                 coverage_metrics=coverage['metrics'],delta_vs_coverage=chosen['metrics']['f1']-coverage['metrics']['f1'])
        rows.append(row)
    write(out/'paired_cases.json',rows)
    summary={}
    for endpoint in sorted({r['endpoint'] for r in rows}):
        for split in ('tune','confirm'):
            selected=[r for r in rows if r['endpoint']==endpoint and r['split']==split]
            for group in ('all',*sorted({r['group'] for r in selected})):
                subset=[r for r in selected if group=='all' or r['group']==group]
                s=summarize(subset)
                s['macro_delta_vs_coverage']=sum(r['delta_vs_coverage'] for r in subset)/len(subset)
                s['pooled_delta_vs_coverage']=s['metrics']['f1']-pooled([r['coverage_metrics'] for r in subset])['f1']
                summary['/'.join((endpoint,split,group))]=s
    write(out/'summary.json',summary)
    write(out/'frozen_rule.json',dict(rule='mask_pixels == 0 or largest_component_fraction >= 0.5',threshold=.5,
        selection='one majority threshold motivated by tune shape audit, no sweep; confirm previously exposed',new_api_requests=0))
    lines=['# N7 concentration follow-up','',
        'One predeclared rule: call for empty current answer or largest component >= half of current-mask area.',
        'Same cached endpoints; no new API. Exposed-development internal confirmation only.','',
        '| endpoint | split | group | F1/P/R | macro Δ never | pooled Δ never | macro Δ always | pooled Δ always | macro Δ coverage | pooled Δ coverage | calls | W/L/T | added TP/FP | verifier calls | replay CNY |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key,s in summary.items():
        m=s['metrics'];cells=key.split('/')+[f"{m['f1']:.4f}/{m['precision']:.4f}/{m['recall']:.4f}"]
        cells += [f"{s[k]:.4f}" for k in ('macro_delta','pooled_delta','macro_delta_vs_fixed','pooled_delta_vs_fixed','macro_delta_vs_coverage','pooled_delta_vs_coverage')]
        cells += [f"{s['calls']}/{s['images']}",f"{s['wins']}/{s['losses']}/{s['ties']}",f"{s['added_tp']}/{s['added_fp']}",str(s['verifier_requests']),f"{s['token_usage_cost_cny']:.6f}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['','All called-harm and skipped-benefit records are retained in paired_cases.json; empty-GT FP counts are in summary.json.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(out/'execution.json',dict(status='complete',slurm_job=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,new_api_requests=0))
    print('Frozen concentration replay complete.')

if __name__=='__main__': main()
