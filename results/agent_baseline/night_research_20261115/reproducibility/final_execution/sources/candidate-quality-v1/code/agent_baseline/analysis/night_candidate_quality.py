"""Offline common-pool pixel-quality diagnosis; never used for selection."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import read_image,write_json
from .missed_sources import load_npz,unpack,specs,rows_for
from .night_missed_views import VARIANTS,answer_path

BINS=('[0]','(0,.1]','(.1,.5]','(.5,.9]','(.9,1]')


def quality_bin(tp,fp):
    if tp+fp==0:return 'empty_after_subtraction'
    if tp==0:return BINS[0]
    # Integer comparisons preserve the declared inclusive boundaries.
    if 10*tp<=tp+fp:return BINS[1]
    if 2*tp<=tp+fp:return BINS[2]
    if 10*tp<=9*(tp+fp):return BINS[3]
    return BINS[4]


def aggregate(records):
    groups={}
    for row in records:
        for variant in VARIANTS:
            key='/'.join((row['group'],row['split'],variant))
            bins=groups.setdefault(key,{b:dict(candidates=0,yes=0,no=0,unknown=0,missing=0,
                accepted_candidate_tp_sum=0,accepted_candidate_fp_sum=0) for b in (*BINS,'empty_after_subtraction')})
            cell=bins[row['quality_bin']];cell['candidates']+=1
            answer=row['answers'][variant];decision=answer['answer']
            cell[decision if decision in ('yes','no') else 'unknown']+=1
            cell['missing']+=int(answer['missing'])
            if decision=='yes':
                cell['accepted_candidate_tp_sum']+=row['tp'];cell['accepted_candidate_fp_sum']+=row['fp']
    return groups


def load(path):return json.loads(Path(path).read_text())


def run(a):
    items=load(a.views/'manifest.json')
    if len(items)!=30 or sum(len(r['candidates']) for r in items)!=526:
        raise ValueError('This diagnosis requires the original complete 30-record / 526-candidate pool')
    lookup={s['group']:{Path(r['id']).stem:r for r in rows_for(a.run,s)} for s in specs(a.run,a.work) if not s['group'].startswith('whu')}
    records=[]
    for item in items:
        group,stem=item['group'],item['stem']
        base=load_npz(a.work/'base'/group/(stem+'.npz'))['start']
        data=load_npz(Path(item['masks']));masks=unpack(data['masks'],tuple(data['shape']))
        if len(masks)!=len(item['candidates']):raise ValueError('Candidate/mask count differs')
        truth=read_image(lookup[group][stem]['gt_mask'])>(127 if group.startswith('levir') else 0)
        rows=[]
        for rank,(candidate,mask) in enumerate(zip(item['candidates'],masks),1):
            novel=mask&~base;tp=int((novel&truth).sum());fp=int((novel&~truth).sum());answers={}
            for variant in VARIANTS:
                path=answer_path(argparse.Namespace(pilot=a.pilot,out=a.views),variant,candidate['uid'])
                answer=load(path) if path.exists() else None
                answers[variant]=dict(answer=answer.get('answer') if answer else None,missing=answer is None,
                    finish_reason=answer.get('finish_reason') if answer else None,
                    parse_failure=answer.get('parse_failure') if answer else None,answer_path=str(path))
            rows.append(dict(uid=candidate['uid'],group=group,stem=stem,split=item['split'],direction=item['direction'],
                rank=rank,source=candidate.get('source'),sources=candidate.get('sources'),tp=tp,fp=fp,
                removed_baseline_pixels=int((mask&base).sum()),pixel_precision=tp/(tp+fp) if tp+fp else None,
                quality_bin=quality_bin(tp,fp),answers=answers))
        write_json(a.out/'images'/group/(stem+'.json'),dict(candidates=rows));records.extend(rows)
        write_json(a.out/'progress.json',dict(completed_candidates=len(records),expected_candidates=526,last=stem))
    if len({r['uid'] for r in records})!=526:raise ValueError('Expected 526 distinct candidate UIDs')
    summary=dict(records=30,candidates=len(records),api_requests=0,groups=aggregate(records),images=records,
        policy='Original common pool; GT offline explanation only; fixed bins never select methods. Pixels exclude initial baseline. Candidate sums count overlapping pixels repeatedly, not accepted union. Missing is a subset of unknown. Empty masks retain denominator separately. No semantic ground-truth annotation.')
    write_json(a.out/'summary.json',summary)
    lines=['# 共同526候选新增像素质量诊断','',summary['policy'],'',
        '全部30记录、原tune/confirm和V0–V4均保留。每臂跨全部组与划分的区间候选数之和均为526；接受/拒绝/unknown之和等于候选数。TP/FP为候选内真实新增像素，TP>FP不是人工语义真值；对象事件yes不保证提交边界像素正确。','',
        '|组/划分/臂|精度区间|候选数|接受|拒绝|unknown（含missing）|missing|接受候选TP累加|接受候选FP累加|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,bins in summary['groups'].items():
        for b,x in bins.items():
            lines.append('|'+ '|'.join(map(str,(key,b,x['candidates'],x['yes'],x['no'],x['unknown'],x['missing'],x['accepted_candidate_tp_sum'],x['accepted_candidate_fp_sum'])))+'|')
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    for name in ('views','pilot','work','run','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run via Slurm')
    run(a)


if __name__=='__main__':main()
