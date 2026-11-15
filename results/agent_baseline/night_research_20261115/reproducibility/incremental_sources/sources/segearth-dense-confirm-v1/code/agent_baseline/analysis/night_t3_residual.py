"""Fixed R1 direct4 paired support intervention: show candidate minus exemplar only."""
import argparse
import json
import os
from pathlib import Path
from agent_baseline.exemplar_semantics import request,MODEL
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,group
from .t3_routes_cache import read_npz
from .night_t3_direct4 import four_views,SYSTEM,T3_QUESTION
from .night_t3_grounding import API_SEED
from .night_t3_retrieval import score,summaries,verification_stats


def candidate_mode(mask,exemplar):
    residual=mask&~exemplar
    return ('skip_empty' if not residual.any() else ('request_residual' if (mask&exemplar).any() else 'reuse')),residual


def reuse_answer(folder):
    inputs=json.loads(read_text(folder/'input.json'))
    expected=dict(system=SYSTEM,question=T3_QUESTION,model=MODEL,seed=API_SEED,temperature=0,max_tokens=768)
    if any(inputs.get(k)!=v for k,v in expected.items()):raise ValueError('Original direct4 request parameters differ')
    original=json.loads(read_text(folder/'answer.json'));label=original['parsed'].get('change')
    return dict(answer=label if label in ('yes','no') else 'unparsed',cost_cny=0,new_api_request=False,
        reused_from=str(folder/'answer.json'),input_source=str(folder/'input.json'),original_cost_cny=original['cost_cny'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Execute through Slurm')
    original=a.night/'r1_direct4/tune';complete=json.loads(read_text(original/'summary.json'))
    if complete['tasks']!=30:raise ValueError('Complete fixed tune direct4 answers before the paired intervention')
    audit=json.loads(read_text(a.night/'audit.json'));source=a.night/'retrieval/B0/tasks';out=a.night/'r1_residual/tune';records=[]
    settings=dict(code=str(Path(__file__).resolve()),source=str(original),model=MODEL,seed=API_SEED,temperature=0,max_tokens=768,k=8,
        intervention='Candidate support C minus Example E, including resulting local crop; Example views, system/question, seed, Top8 and original ranking unchanged',
        reuse='nonoverlapping C has identical rendering and request parameters; reference original answer at zero new cost; empty residual makes no request',
        scoring='original direct4 is strict control; R1 without verification is reference; official prediction scoring already removes E, so no-verification support is unchanged',
        scope='fixed tune30 only; no GT-based candidate selection; overlap-associated FP is not exclusive or predicted removable FP; no confirm promotion')
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Existing residual settings differ')
    write_json(out/'settings.json',settings)
    for row in load_dev(a.run):
        if audit['split'][row['id']]!='tune':continue
        folder=out/'tasks'/row['id'];final=folder/'metrics.json'
        if exists(final):records.append(json.loads(read_text(final)));continue
        prior_folder=original/'tasks'/row['id'];prior=json.loads(read_text(prior_folder/'metrics.json'))
        prepared=read_npz(source/row['id']/'prepared.npz');masks=prepared['r1'].astype(bool);indices=prepared['order1'].tolist()[:8]
        if indices!=prior['candidate_ids']:raise ValueError('Candidate order differs from direct4')
        exemplar=read_image(row['exemplar_mask'])>0;images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];answers={};support=[]
        for index in indices:
            mode,residual=candidate_mode(masks[index],exemplar);request_folder=folder/'verification'/str(index)
            support.append(dict(id=index,mode=mode,candidate_pixels=int(masks[index].sum()),overlap_pixels=int((masks[index]&exemplar).sum()),residual_pixels=int(residual.sum())))
            if mode=='reuse':
                answer=reuse_answer(prior_folder/'verification'/str(index));write_json(request_folder/'reused.json',answer)
            elif mode=='skip_empty':
                answer=dict(answer='empty_residual_skip',cost_cny=0,new_api_request=False);write_json(request_folder/'skipped.json',answer)
            else:
                response=request(request_folder,SYSTEM,T3_QUESTION,four_views(images,exemplar,residual),API_SEED)
                label=response['parsed'].get('change');answer=dict(answer=label if label in ('yes','no') else 'unparsed',cost_cny=response['cost_cny'],new_api_request=True,
                    original_answer=str(prior_folder/'verification'/str(index)/'answer.json'))
            answers[index]=answer
            write_json(request_folder/'support.json',dict(support[-1],prepared_source=str(source/row['id']/'prepared.npz'),exemplar_source=row['exemplar_mask']))
        truth=read_image(row['gt_mask'])>0
        baseline=next(r for r in prior['scores'] if r['method']=='R1_direct4_verified')
        noverify=next(r for r in prior['scores'] if r['method']=='R1_geometry_dino')
        # Scoring original masks with E excluded equals scoring their residuals.
        verified=score(masks,[i for i in indices if answers[i]['answer'] in ('yes','unparsed')],truth,exemplar,'R1_residual_verified',8)
        scores=[noverify,baseline,verified,score(masks,[i for i in indices if answers[i]['answer']=='yes'],truth,exemplar,'R1_residual_drop_unparsed_diagnostic',8)]
        new_requests=sum(v['new_api_request'] for v in answers.values());reused=sum('reused_from' in v for v in answers.values())
        stats=verification_stats(answers);stats.update(requests=new_requests,decisions=len(answers),reused_answers=reused)
        record=dict(id=row['id'],split='tune',group=group(row),remaining_gt_pixels=prior['remaining_gt_pixels'],scores=scores,
            verification=answers,verification_stats=stats,new_api_requests=new_requests,reused_answers=reused,
            empty_residual_skips=sum(v['answer']=='empty_residual_skip' for v in answers.values()),geometry_proposal_upper=prior['geometry_proposal_upper'],
            candidate_ids=indices,support=support,paired_counts_delta=dict(zip(('tp','fp','fn'),[new-old for new,old in zip(verified['counts'],baseline['counts'])])),
            cost_cny=sum(v['cost_cny'] for v in answers.values()))
        write_json(final,record);records.append(record)
        write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records),new_api_requests=sum(r['new_api_requests'] for r in records),reused_answers=sum(r['reused_answers'] for r in records)))
        print('residual',len(records),row['id'],flush=True)
    write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records),new_api_requests=sum(r['new_api_requests'] for r in records),reused_answers=sum(r['reused_answers'] for r in records)))


if __name__=='__main__':main()
