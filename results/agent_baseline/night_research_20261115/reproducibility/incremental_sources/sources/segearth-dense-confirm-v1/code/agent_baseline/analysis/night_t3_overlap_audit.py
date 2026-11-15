"""Cache-only R1 Top8 exemplar-overlap audit; no new verification or pool changes."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text,read_image,write_json,write_text,exists
from .t3_routes import load_dev,group,pixel_counts,pooled,GROUPS
from .t3_routes_cache import read_npz


def candidate_geometry(mask,exemplar,truth):
    overlap=int((mask&exemplar).sum());area=int(mask.sum());residual=mask&~exemplar
    return dict(candidate_pixels=area,example_overlap_pixels=overlap,overlap_fraction=overlap/area if area else None,
        remaining_support_pixels=int(residual.sum()),raw_counts_including_example=pixel_counts(mask,truth,np.zeros_like(exemplar)),
        remaining_counts=pixel_counts(mask,truth,exemplar),
        category='empty_residual' if not residual.any() else ('overlap_nonempty' if overlap else 'no_overlap'))


def cached_complete(path):
    if not exists(path):return None
    record=json.loads(read_text(path))
    return record if record['answers_complete'] else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Read and process masks through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));records=[]
    def checkpoint(record):
        records.append(record)
        write_json(a.output/'progress.json',dict(tasks=len(records),answers_complete_tasks=sum(r['answers_complete'] for r in records),
            ids=[r['id'] for r in records],model_calls=0,api_cost_cny=0))
        print('overlap audit',len(records),record['id'],record['answers_complete'],flush=True)
    for row in load_dev(a.run):
        if audit['split'][row['id']]!='tune':continue
        task_output=a.output/'tasks'/row['id']/'diagnostic.json'
        cached=cached_complete(task_output)
        if cached is not None:
            checkpoint(cached);continue
        source=a.night/'retrieval/B0/tasks'/row['id'];prior=json.loads(read_text(source/'metrics.json'));prepared=read_npz(source/'prepared.npz')
        masks=prepared['r1'].astype(bool);indices=prepared['order1'].tolist()[:8]
        ex=read_image(row['exemplar_mask'])>0;truth=read_image(row['gt_mask'])>0;candidates=[]
        for rank,index in enumerate(indices,1):
            path=a.night/'r1_direct4/tune/tasks'/row['id']/'verification'/str(index)/'answer.json'
            value=json.loads(read_text(path)) if exists(path) else None
            label=value['parsed'].get('change') if value else 'pending'
            if value and label not in ('yes','no'):label='unparsed'
            candidates.append(dict(id=index,rank=rank,answer=label,answer_path=str(path) if value else None,
                **candidate_geometry(masks[index],ex,truth)))
        union=lambda ids:masks[ids].any(axis=0) if ids else np.zeros_like(ex)
        baseline=pixel_counts(union(indices),truth,ex)
        original=next(r for r in prior['scores'] if r['method']=='R1_geometry_dino' and r['k']==8)
        if baseline!=original['counts']:raise ValueError('Audit Top8 differs from original R1')
        known=all(r['answer']!='pending' for r in candidates)
        accepted=[r['id'] for r in candidates if r['answer'] in ('yes','unparsed')]
        overlap=[r['id'] for r in candidates if r['answer'] in ('yes','unparsed') and r['example_overlap_pixels']>0]
        record=dict(id=row['id'],split='tune',group=group(row),remaining_gt_pixels=int((truth&~ex).sum()),candidates=candidates,
            answers_complete=known,baseline_counts=baseline,
            verified_counts=pixel_counts(union(accepted),truth,ex) if known else None,
            accepted_overlap_union_counts=pixel_counts(union(overlap),truth,ex) if known else None)
        write_json(task_output,record);checkpoint(record)
    strata=[]
    for support in ('all','positive','empty'):
        for label in ('全部',*GROUPS):
            rows=[r for r in records if (label=='全部' or r['group']==label) and (support=='all' or (r['remaining_gt_pixels']>0)==(support=='positive'))]
            counts={}
            for category in ('no_overlap','overlap_nonempty','empty_residual'):
                items=[c for r in rows for c in r['candidates'] if c['category']==category]
                accepted=[c for c in items if c['answer'] in ('yes','unparsed')]
                counts[category]=dict(candidates=len(items),answers={v:sum(c['answer']==v for c in items) for v in ('yes','no','unparsed','pending')},
                    accepted_candidates=len(accepted),accepted_fp_candidates=sum(c['remaining_counts'][1]>0 for c in accepted),
                    accepted_candidate_sum_counts=np.asarray([c['remaining_counts'] for c in accepted],dtype=np.int64).reshape(-1,3).sum(axis=0).tolist(),
                    accepted_overlap_fractions=[c['overlap_fraction'] for c in accepted],
                    candidate_sum_counts=np.asarray([c['remaining_counts'] for c in items],dtype=np.int64).reshape(-1,3).sum(axis=0).tolist())
            complete=[r for r in rows if r['answers_complete']]
            strata.append(dict(support=support,group=label,tasks=len(rows),answers_complete_tasks=len(complete),categories=counts,
                complete_verified=pooled([dict(counts=r['verified_counts']) for r in complete]),
                accepted_overlap_union=pooled([dict(counts=r['accepted_overlap_union_counts']) for r in complete])))
    result=dict(tasks=len(records),records=records,strata=strata,api_cost_cny=0,model_calls=0,
        interpretation='Candidate-summed counts double count overlaps; complete-task unions are pooled separately. Accepted overlap FP association does not prove self-matching caused decisions. Pending answers stay pending.',
        candidate_policy='Original fixed R1 Top8 retained; live match_exemplar and eligible do not remove exemplar overlap. Scoring alone excludes exemplar pixels.')
    write_json(a.output/'summary.json',result)
    lines=['# R1 direct4 exemplar-overlap audit','',result['candidate_policy'],result['interpretation'],'',
        '| GT | Group | N | Answers complete | Candidate category | Candidates | Yes/no/unparsed/pending | Accepted candidate-sum TP/FP/FN |','|---|---|---:|---:|---|---:|---|---|']
    for r in strata:
        for category,c in r['categories'].items():lines.append(f"| {r['support']} | {r['group']} | {r['tasks']} | {r['answers_complete_tasks']} | {category} | {c['candidates']} | {c['answers']} | {c['accepted_candidate_sum_counts']} |")
    lines+=['','| Task | GT pixels | Known answers | Accepted overlap union TP/FP/FN | Full verified TP/FP/FN |','|---|---:|---:|---|---|']
    for r in records:lines.append(f"| {r['id']} | {r['remaining_gt_pixels']} | {sum(c['answer']!='pending' for c in r['candidates'])}/{len(r['candidates'])} | {r['accepted_overlap_union_counts']} | {r['verified_counts']} |")
    write_text(a.output/'report.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
