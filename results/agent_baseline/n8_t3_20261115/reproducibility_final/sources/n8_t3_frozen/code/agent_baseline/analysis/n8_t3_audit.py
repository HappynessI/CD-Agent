"""JSON-only final N8 stage counts, failure decomposition and returned-usage audit."""
import argparse
import json
import os
from .io_retry import read_text,write_json,write_text
from .n8_t3_cost import line_ledger
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run report analysis via Slurm')
    stage1=json.loads(read_text(a.out/'step1/summary.json'));stage2=json.loads(read_text(a.out/'step2/summary.json'))
    stage3=json.loads(read_text(a.out/'step3/summary.json'));frozen=json.loads(read_text(a.out/'step3/final_frozen.json'));k=frozen['k']
    details=[];summary=[]
    for split in ('tune','confirm'):
        first=[r for r in stage1['records'] if r['split']==split];second=[r for r in stage2['records'] if r['split']==split]
        prep=json.loads(read_text(a.out/'step3/prepare'/split/'summary.json'))['records'];prepared={r['id']:r for r in prep}
        final=[r for r in stage3['records'] if r['split']==split]
        summary.append(dict(split=split,tasks=len(final),original_candidates=sum(r['original_candidates'] for r in first),
            clean_candidates=sum(r['clean_candidates'] for r in first),removed_candidates=sum(r['removed_candidates'] for r in first),
            modified_retained=sum(r['modified_retained'] for r in first),routed_to_instance=sum(r['routed_to']=='instance' for r in second),
            routed_to_dense=sum(r['routed_to']=='dense' for r in second),rebuilt_candidates=sum(r['candidates'] for r in prep),
            new_masks_vs_original=sum(r['new_candidates'] for r in prep),gated_tasks=sum(r['triggered'] for r in final),
            request_slots=sum(r['new_calls'] for r in final),new_mask_request_slots=sum(r['new_mask_calls'] for r in final),
            unknown_answers=sum(r['unknown'] for r in final),
            positive_fully_rejected=sum(r['remaining_gt_pixels']>0 and sum(r['methods']['N8_verified']['counts'][:2])==0 for r in final)))
        for r in final:
            source=prepared[r['id']];pool=source['methods']['pool_upper']['counts'];top=source['methods'][f'pool_top{k}']['counts']
            gate=r['methods']['N8_gate_unverified']['counts'];verified=r['methods']['N8_verified']['counts'];base=r['methods']['original_R1_top8']['counts']
            reasons=[]
            if r['remaining_gt_pixels'] and pool[0]==0:reasons.append('rebuilt_pool_zero_GT_coverage')
            if pool[0]>top[0]:reasons.append('topK_truncation_loses_covered_TP')
            if top[0]>gate[0]:reasons.append('relative_gate_loses_TP')
            if gate[0]>verified[0]:reasons.append('evidence_verifier_loses_TP')
            details.append(dict(id=r['id'],split=split,remaining_gt_pixels=r['remaining_gt_pixels'],baseline=base,pool=pool,topk=top,
                gated=gate,verified=verified,pool_to_topk_tp_loss=pool[0]-top[0],gate_tp_loss=top[0]-gate[0],verifier_tp_loss=gate[0]-verified[0],
                verifier_fp_removed=gate[1]-verified[1],triggered=r['triggered'],z=r['z'],top1_ptrans=r['top1_ptrans'],reasons=reasons))
    ledger=line_ledger(a.out/'step3/verification')
    result=dict(stages=summary,per_task=details,usage_ledger=ledger,selection=frozen,advancement=json.loads(read_text(a.out/'advancement.json')),
        interpretation='All stage outputs retained; count-based failure decomposition is offline diagnosis, not a GT-based inference rule. Request slots differ from HTTP attempts; returned-usage failures are charged once.')
    write_json(a.out/'final_audit.json',result)
    lines=['# N8 T3 final counts and failure decomposition','',result['interpretation'],'',
        '| Split | Tasks | Original candidates | Clean | Removed | Modified retained | Inst routes | Dense routes | Rebuilt | New masks | Triggered | Request slots | New-mask slots | Unknown answers |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+' | '.join(str(r[k]) for k in ('split','tasks','original_candidates','clean_candidates','removed_candidates','modified_retained','routed_to_instance','routed_to_dense','rebuilt_candidates','new_masks_vs_original','gated_tasks','request_slots','new_mask_request_slots','unknown_answers'))+' |')
    lines+=['',f"Returned-usage known cost CNY {ledger['known_cost_cny']}; known failed-response cost {ledger['known_failed_cost_cny']}; HTTP attempts {ledger['attempts']}; missing-usage attempts {ledger['usage_missing_attempts']}. Not an account invoice.", '',
        '| Split | Task | Remaining GT | Original TP/FP/FN | Pool TP/FP/FN | TopK TP/FP/FN | Gate TP/FP/FN | Verified TP/FP/FN | Cause |',
        '|---|---|---:|---|---|---|---|---|---|']
    for r in details:lines.append('| '+' | '.join(map(str,[r['split'],r['id'],r['remaining_gt_pixels'],r['baseline'],r['pool'],r['topk'],r['gated'],r['verified'],', '.join(r['reasons'])]))+' |')
    write_text(a.out/'final_audit.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
