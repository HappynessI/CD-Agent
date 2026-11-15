"""JSON-only paired failure attribution after frozen rejection/verification scoring."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import read_text, write_json, write_text, exists
from .n7_t3_rejection import RANKS
from .t3_routes import pooled


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run analysis through Slurm')
    records=[]
    for split in ('tune','confirm'):
        verify=a.out/'verification'/split/'summary.json';baseline=a.out/f'{split}_summary.json'
        if exists(verify):source=verify
        elif exists(baseline):source=baseline
        else:continue
        records.extend(json.loads(read_text(source))['records'])
    details=[];paired=[]
    for r in records:
        pool=r['geometry_proposal_upper'];gt=r['remaining_gt_pixels']
        for rank in RANKS:
            base=r['methods'][rank+'_top8'];gated=r['methods'][rank+'_threshold']
            verified=r['methods'].get(rank+'_threshold_verified')
            causes=[]
            if gt and pool['covered_pixels']==0:causes.append('original_geometry_pool_zero_GT_coverage')
            if gt and pool['covered_pixels']>base['counts'][0]:causes.append('Top8_truncation_loses_pool_coverage')
            if gated['counts'][0]<base['counts'][0]:causes.append('similarity_threshold_loses_TP')
            if verified and verified['counts'][0]<gated['counts'][0]:causes.append('local_verifier_loses_TP')
            if gt==0 and gated['counts'][1]>0:causes.append('similarity_gate_retains_empty_target_FP')
            if gt==0 and verified and verified['counts'][1]>0:causes.append('local_verifier_retains_empty_target_FP')
            details.append(dict(id=r['id'],split=r['split'],group=r['group'],rank=rank,remaining_gt_pixels=gt,
                pool_covered_pixels=pool['covered_pixels'],top8=base['counts'],threshold=gated['counts'],
                verified=verified['counts'] if verified else None,causes=causes,
                exemplar_overlap_top8=r['top8_exemplar_overlap_pixels'][rank],
                overlap_interpretation='support-overlap diagnostic only; not causal proof and not a gate'))
    for split in ('tune','confirm'):
        chosen=[r for r in records if r['split']==split]
        if not chosen:continue
        for rank in RANKS:
            for suffix in ('threshold','threshold_verified'):
                name=rank+'_'+suffix
                if not all(name in r['methods'] for r in chosen):continue
                positive=[r for r in chosen if r['remaining_gt_pixels']>0];empty=[r for r in chosen if r['remaining_gt_pixels']==0]
                b=pooled([r['methods'][rank+'_top8'] for r in positive]);m=pooled([r['methods'][name] for r in positive])
                empty_before=pooled([r['methods'][rank+'_top8'] for r in empty]);empty_after=pooled([r['methods'][name] for r in empty])
                retention=m['tp']/b['tp'] if b['tp'] else None
                wins=losses=ties=0
                for r in chosen:
                    before=pooled([r['methods'][rank+'_top8']])['f1_c'];after=pooled([r['methods'][name]])['f1_c']
                    wins+=after>before;losses+=after<before;ties+=after==before
                passed=bool(retention is not None and retention>=.95 and m['f1_c']>=b['f1_c'] and empty_after['fp']<empty_before['fp'])
                paired.append(dict(split=split,method=name,positive_baseline=b,positive_method=m,tp_retention=retention,
                    empty_fp_before=empty_before['fp'],empty_fp_after=empty_after['fp'],
                    image_f1_wins=wins,image_f1_losses=losses,image_f1_ties=ties,
                    positive_entirely_rejected=[r['id'] for r in positive if sum(r['methods'][name]['counts'][:2])==0],
                    positive_tp_losses={r['id']:r['methods'][rank+'_top8']['counts'][0]-r['methods'][name]['counts'][0] for r in positive if r['methods'][rank+'_top8']['counts'][0]>r['methods'][name]['counts'][0]},
                    passes_preregistered_aggregate_criteria=passed,
                    promotion='eligible for review, not automatic online promotion' if split=='confirm' and passed else 'not promoted'))
    result=dict(tasks=len(records),paired=paired,per_task=details,
        limitation='Failure attribution is count-based. Segmentation errors vs semantic errors require visual evidence; overlap is not causal evidence. Empty-task pixel F1 is zero under the unchanged historical convention, so use empty accuracy/FP separately.')
    write_json(a.out/'diagnostics.json',result)
    lines=['# N7 T3 paired diagnostics','',result['limitation'],'',
        '| Split | Method | Positive TP retention | Positive F1 before | Positive F1 after | Empty FP before | Empty FP after | Image wins/losses/ties | Criteria |',
        '|---|---|---:|---:|---:|---:|---:|---|---|']
    for r in paired:
        lines.append('| '+' | '.join(map(str,[r['split'],r['method'],r['tp_retention'],r['positive_baseline']['f1_c'],r['positive_method']['f1_c'],r['empty_fp_before'],r['empty_fp_after'],f"{r['image_f1_wins']}/{r['image_f1_losses']}/{r['image_f1_ties']}",r['promotion']]))+' |')
    lines+=['','## Per-task losses and bottlenecks','', '| Split | Task | Rank | Pool covered | Top8 TP/FP/FN | Threshold TP/FP/FN | Verified TP/FP/FN | Count-based causes |','|---|---|---|---:|---|---|---|---|']
    for r in details:lines.append('| '+' | '.join(map(str,[r['split'],r['id'],r['rank'],r['pool_covered_pixels'],r['top8'],r['threshold'],r['verified'],', '.join(r['causes'])]))+' |')
    write_text(a.out/'diagnostics.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
