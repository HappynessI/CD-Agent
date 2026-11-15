"""Post hoc JSON-only association audit of Top1 loss and exemplar overlap."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import read_text, write_json, write_text
from .n7_t3_rejection import RANKS
from .t3_routes import pooled


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run diagnostics through Slurm')
    records=json.loads(read_text(a.out/'all_summary.json'))['records'];details=[];summary=[]
    for r in records:
        for rank in RANKS:
            top1=r['methods'][rank+'_top1'];top8=r['methods'][rank+'_top8']
            if top1['candidate_ids']!=top8['candidate_ids'][:1]:raise ValueError('Top1 candidate differs from first Top8')
            overlap=r['top8_exemplar_overlap_pixels'][rank][0] if top1['candidate_ids'] else 0
            residual=sum(top1['counts'][:2]);raw=overlap+residual
            details.append(dict(id=r['id'],split=r['split'],group=r['group'],rank=rank,remaining_gt_pixels=r['remaining_gt_pixels'],
                candidate_id=top1['candidate_ids'][0] if top1['candidate_ids'] else None,
                exemplar_overlap_pixels=overlap,post_exemplar_prediction_pixels=residual,
                raw_candidate_pixels=raw,overlap_fraction_of_raw_candidate=overlap/raw if raw else None,
                top1_counts=top1['counts'],top8_counts=top8['counts'],tp_lost=top8['counts'][0]-top1['counts'][0]))
    for split in ('tune','confirm','all'):
        for rank in RANKS:
            for support in ('all','positive','empty'):
                for overlap_group in ('all','no_overlap','overlap','majority_overlap'):
                    rows=[r for r in details if (split=='all' or r['split']==split) and r['rank']==rank
                          and (support=='all' or support=='positive' and r['remaining_gt_pixels']>0 or support=='empty' and r['remaining_gt_pixels']==0)
                          and (overlap_group=='all' or overlap_group=='no_overlap' and r['exemplar_overlap_pixels']==0
                               or overlap_group=='overlap' and r['exemplar_overlap_pixels']>0
                               or overlap_group=='majority_overlap' and (r['overlap_fraction_of_raw_candidate'] or 0)>.5)]
                    if not rows:continue
                    summary.append(dict(split=split,rank=rank,support=support,overlap_group=overlap_group,tasks=len(rows),
                        top1=pooled([dict(counts=r['top1_counts']) for r in rows]),
                        top8=pooled([dict(counts=r['top8_counts']) for r in rows]),
                        positive_tp_tasks=sum(r['top1_counts'][0]>0 for r in rows),
                        raw_candidate_pixels=sum(r['raw_candidate_pixels'] for r in rows),
                        exemplar_overlap_pixels=sum(r['exemplar_overlap_pixels'] for r in rows),
                        post_exemplar_prediction_pixels=sum(r['post_exemplar_prediction_pixels'] for r in rows)))
    caveat=('Post hoc association only. All masks/GT/scores unchanged; overlap and majority-overlap groups are descriptive, not inference gates. '
            'Raw candidate area is exactly the single-candidate scored area plus exemplar intersection; it is not exemplar area. '
            'Overlap does not establish same-instance self-matching, nor prove that removing an overlapping candidate restores recall.')
    write_json(a.out/'top1_overlap_audit.json',dict(tasks=len(records),records=details,summary=summary,interpretation=caveat))
    lines=['# Top1 and exemplar-overlap association audit','',caveat,'',
           '| Split | Rank | Support | Overlap group | N | Top1 TP | Top8 TP | Top1 FP | Nonzero-TP tasks | Overlap pixels | Post-exemplar predicted pixels |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+' | '.join(map(str,[r['split'],r['rank'],r['support'],r['overlap_group'],r['tasks'],r['top1']['tp'],r['top8']['tp'],r['top1']['fp'],r['positive_tp_tasks'],r['exemplar_overlap_pixels'],r['post_exemplar_prediction_pixels']]))+' |')
    lines+=['','## Largest Top8-to-Top1 TP losses','',
        '| Split | Task | Rank | TP lost | Top8 TP | Top1 TP | Exemplar overlap pixels | Fraction of raw candidate overlapping exemplar | Remaining prediction pixels |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(details,key=lambda r:-r['tp_lost'])[:20]:
        lines.append('| '+' | '.join(map(str,[r['split'],r['id'],r['rank'],r['tp_lost'],r['top8_counts'][0],r['top1_counts'][0],r['exemplar_overlap_pixels'],r['overlap_fraction_of_raw_candidate'],r['post_exemplar_prediction_pixels']]))+' |')
    write_text(a.out/'top1_overlap_audit.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
