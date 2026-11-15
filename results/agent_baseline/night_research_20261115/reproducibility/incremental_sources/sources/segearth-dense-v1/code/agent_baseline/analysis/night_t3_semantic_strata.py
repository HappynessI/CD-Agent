"""Read-only tune label-agreement audit; no image loading or inference."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import read_text, write_json, write_text

VARIANTS=('B0','B1','B2','B3')


def audit_semantics(audit, variants):
    cohort={r['id']:r for r in audit['records'] if r['split']=='tune'}
    if len(cohort)!=30:raise ValueError('Expected the original fixed 30-task tune cohort')
    aligned={}
    for name in VARIANTS:
        records=[r for r in variants[name]['records'] if r['split']=='tune']
        by_id={r['id']:r for r in records}
        if len(records)!=30 or set(by_id)!=set(cohort):raise ValueError('Variants must contain exactly the same fixed tune30')
        for key,row in by_id.items():
            if row['remaining_gt_pixels']!=cohort[key]['remaining_gt_pixels']:raise ValueError('Remaining GT metadata differs')
        aligned[name]=by_id
    strata={
        'all':sorted(cohort),
        'positive':sorted(k for k,r in cohort.items() if r['remaining_gt_pixels']>0),
        'remaining_gt>=16_diagnostic':sorted(k for k,r in cohort.items() if r['remaining_gt_pixels']>=16),
        'empty':sorted(k for k,r in cohort.items() if r['remaining_gt_pixels']==0),
    }
    rows=[]
    for support,ids in strata.items():
        for group in ('全部','目标类','地表类'):
            selected=[key for key in ids if group=='全部' or cohort[key]['group']==group]
            base={key for key in selected if aligned['B0'][key]['pair_agrees']}
            for variant in VARIANTS:
                counts={label:sum(bool(aligned[variant][key][label+'_agrees']) for key in selected) for label in ('before','after','pair')}
                correct=sorted(key for key in selected if aligned[variant][key]['pair_agrees'])
                rows.append(dict(support=support,group=group,variant=variant,tasks=len(selected),task_ids=selected,
                    correct_counts=counts,label_agreement={key:value/len(selected) if selected else None for key,value in counts.items()},
                    pair_correct_ids=correct,gained_pair_ids_vs_B0=sorted(set(correct)-base),lost_pair_ids_vs_B0=sorted(base-set(correct))))
    return dict(scope='Original fixed tune30 only; confirmation records are ignored',api_calls=0,new_cost_cny=0,
        interpretation='Before/after/pair agreement with existing GT labels is not independently audited grounding accuracy. Remaining GT only stratifies offline diagnostics; original parses, labels and task cohort remain unchanged.',
        diagnostic_16='Original make_t3_tasks.MIN_AREA=16 is a component construction cutoff; remaining total>=16 is sensitivity only and is not the original definition.',
        cohort_sizes={key:len(value) for key,value in strata.items()},rows=rows)


def markdown(data):
    lines=['# Fixed tune semantic label-agreement strata','',data['scope'],data['interpretation'],data['diagnostic_16'],'',
        '| Remaining GT | Group | Variant | N | Before count/rate | After count/rate | Pair count/rate |',
        '|---|---|---|---:|---|---|---|']
    for row in data['rows']:
        values=[f"{row['correct_counts'][key]}/{row['tasks']} ({row['label_agreement'][key]:.4f})" if row['tasks'] else '0/0 (NA)' for key in ('before','after','pair')]
        lines.append('| '+' | '.join(map(str,[row['support'],row['group'],row['variant'],row['tasks'],*values]))+' |')
    lines+=['','## Correct pair identities and changes from B0','']
    for row in data['rows']:
        if row['group']!='全部':continue
        lines.append(f"- {row['support']} / {row['variant']}: correct={row['pair_correct_ids']}; gained={row['gained_pair_ids_vs_B0']}; lost={row['lost_pair_ids_vs_B0']}")
    b0=next(r for r in data['rows'] if r['support']=='positive' and r['group']=='全部' and r['variant']=='B0')
    b3=next(r for r in data['rows'] if r['support']=='positive' and r['group']=='全部' and r['variant']=='B3')
    lines+=['','## Interpretation of the earlier B3 decision','',
        f"Earlier aggregate pair agreement (B3 6/30 versus B0 8/30) alone is insufficient to characterize agreement on retrievable positive tasks. This audit gives B3 {b3['correct_counts']['pair']}/{b3['tasks']} versus B0 {b0['correct_counts']['pair']}/{b0['tasks']} on remaining-GT positives. Use the subgroup identities above to qualify the earlier rationale; do not claim all strata degraded solely from the aggregate.",
        'This is a retrospective diagnosis of the recorded decision, not an automatic promotion, a new parsing run, a retrieval-benefit result, or confirmation-based winner selection. Correct label pairs still require support-grounding review; later oracle text-proposal coverage limitations remain separate evidence.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--night',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run diagnostic aggregation through Slurm')
    load=lambda path:json.loads(read_text(path))
    data=audit_semantics(load(a.night/'audit.json'),{name:load(a.night/name/'summary.json') for name in VARIANTS})
    write_json(a.output/'summary.json',data);write_text(a.output/'report.md',markdown(data))


if __name__=='__main__':main()
