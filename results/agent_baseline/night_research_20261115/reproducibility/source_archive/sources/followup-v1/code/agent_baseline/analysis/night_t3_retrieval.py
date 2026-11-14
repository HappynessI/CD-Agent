"""N2 R1/R2/R3: cached geometry/DINO and parsed-text proposals, Slurm GPU only."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from render import crop_box, SIZE
from agent_baseline.exemplar_semantics import request, CLASSES
from agent_baseline.tools_t13 import Features
from .io_retry import read_image,read_text,write_json,exists
from .t3_routes import load_dev,pixel_counts,pooled,GROUPS,GRID,group
from .t3_routes_cache import CachedVision,read_npz,write_npz
from .t3_diag import Sam3Backend,directed_proposals,proposal_metrics
from .t3_geometry import eligible
from .probe_t3 import task_scores
from .t3_routes_match import bbox
from .night_t3_grounding import spotlight,API_SEED


def unpack(path):
    saved=read_npz(path);shape=tuple(map(int,saved['shape']))
    return np.unpackbits(saved['masks'],axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)


class TextProposals:
    def __init__(self,run,out):
        self.old=run/'t3_diag/cache/t3_sam3/sam3_text';self.reused=[]
        root=Path('/guisongxia01/pangchao/wangyihan')
        self.pool=Sam3Backend(SimpleNamespace(cache=out/'text_cache',device='cuda',
            sam3_python=root/'conda_envs/omniovcd-env/bin/python',sam3_root=root/'Chang-Agent/third_party/OmniOVCD',
            sam3_checkpoint=root/'models/sam3/sam3.pt',sam3_bpe=root/'Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz'))

    def get(self,row,text,time):
        query=row['image_'+time]
        for path in sorted((self.old/row['id']).glob('*.json')):
            meta=json.loads(read_text(path));req=meta['request']
            if req.get('text')==text and req['query']==query and meta['pixel_thr']==.85 and meta['score_thr']==.5:
                self.reused.append(str(path));return unpack(path.with_suffix('.npz'))
        # Class-keyed namespace avoids collision between different predicted pairs.
        keyed=dict(row,id=row['id']+'_'+text.replace(' ','_'))
        return self.pool.for_row(keyed)(mode='sam3_text',time=time,category=text,reference_time='t1',box=[])


def ranked(masks,features,exemplar):
    if not len(masks):return []
    scores=task_scores(masks,*features,dict(instances=[np.flatnonzero(exemplar).tolist()],exemplar_index=0))
    return sorted(range(len(masks)),key=lambda index:(-float(scores[index]),index))


def local_views(images,mask):
    l,t,r,b=crop_box(bbox(mask),mask.shape,scale=1.5,min_side=64)
    return [(f'T{index+1} local support spotlight',Image.fromarray(spotlight(image,mask)[t:b,l:r]).resize((SIZE,SIZE),Image.Resampling.BICUBIC))
        for index,image in enumerate(images)]


def score(masks,indices,truth,exemplar,method,k):
    pred=masks[indices].any(axis=0) if indices else np.zeros_like(exemplar)
    return dict(method=method,k=k,counts=pixel_counts(pred,truth,exemplar),
        coverage=proposal_metrics(masks[indices],truth,exemplar),candidates=indices)


def summaries(records):
    output=[]
    for split in ('all','tune','confirm'):
        for support in ('all','positive','empty'):
            for label in ('全部',*GROUPS):
                tasks=[t for t in records if (split=='all' or t['split']==split) and (label=='全部' or t['group']==label)
                    and (support=='all' or (t['remaining_gt_pixels']>0)==(support=='positive'))]
                methods={}
                for method,k in sorted({(r['method'],r['k']) for t in tasks for r in t['scores']}):
                    rows=[r for t in tasks for r in t['scores'] if r['method']==method and r['k']==k]
                    methods[f'{method}@{k}']=dict(pooled(rows),empty_prediction_rate=float(np.mean([r['counts'][0]+r['counts'][1]==0 for r in rows])),
                        covered_pixels=sum(r['coverage']['covered_pixels'] for r in rows),
                        truth_pixels=sum(r['coverage']['truth_pixels'] for r in rows),
                        aligned_instances=sum(r['coverage']['aligned_instances'] for r in rows),gt_instances=sum(r['coverage']['gt_instances'] for r in rows))
                output.append(dict(split=split,support=support,group=label,tasks=len(tasks),methods=methods))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--parse-variant',choices=['B0','B1','B2'],default='B1');p.add_argument('--split',choices=['tune','confirm','all'],default='all')
    p.add_argument('--verify-k',type=int,default=8,choices=[0,4,8,16]);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run in a Slurm GPU allocation')
    source=json.loads(read_text(a.night/a.parse_variant/'summary.json'))
    if source['tasks']!=60:raise ValueError('Complete grounding output required before retrieval')
    parses={r['id']:r for r in source['records']};audit=json.loads(read_text(a.night/'audit.json'))
    rows=[r for r in load_dev(a.run) if a.split=='all' or audit['split'][r['id']]==a.split]
    out=a.night/'retrieval'/a.parse_variant;out.mkdir(parents=True,exist_ok=True)
    settings=dict(parse_variant=a.parse_variant,verify_k=a.verify_k,model='qwen3.7-flash',source=str(a.night/a.parse_variant/'summary.json'),
        R1='same broad geometry/colour gates as historical geometry; DINO change-vector ranking',
        R2='predicted source/target text each at both dates; directed loss/gain overlap; all proposals union',
        R3='same R2 pool; DINO reorder and top-k; fixed top8 local two-date spotlight verifier',
        verifier='joint local view/support-prompt intervention, not a pure single-factor comparison to old eight-view verifier',
        code=str(Path(__file__).resolve()))
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Existing retrieval settings differ')
    write_json(out/'settings.json',settings)
    cache=CachedVision(a.run/'t3_diag/cache/vision');text=TextProposals(a.run,a.night);records=[]
    try:
        for row in rows:
            folder=out/'tasks'/row['id'];result=folder/'metrics.json'
            if exists(result):records.append(json.loads(read_text(result)));continue
            ex=read_image(row['exemplar_mask'])>0;images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')]
            backend=cache.for_row(row);features=[Features(backend('extract_features',time=t),ex.shape,t) for t in ('t1','t2')]
            prepared=folder/'prepared.npz'
            if exists(prepared):
                saved=read_npz(prepared);r1=saved['r1'].astype(bool);r2=saved['r2'].astype(bool);order1=saved['order1'].tolist();order3=saved['order3'].tolist()
            else:
                proposals=np.concatenate([backend('segment_everything',time=t) for t in ('t1','t2')])
                r1=proposals[eligible(proposals,ex,images)];order1=ranked(r1,features,ex)
                pair=parses[row['id']]['interpretation']['pair']
                if None in pair or pair[0]==pair[1]:r2=np.zeros((0,*ex.shape),bool)
                else:
                    found=[text.get(row,CLASSES[category],time) for category in pair for time in ('t1','t2')]
                    r2=directed_proposals(*found,ex.shape)
                order3=ranked(r2,features,ex)
                write_npz(prepared,r1=r1,r2=r2,order1=np.asarray(order1,dtype=int),order3=np.asarray(order3,dtype=int))
                write_json(folder/'provenance.json',dict(predicted_pair=pair,text_cache_reused=text.reused))
                text.reused=[]
            pair=parses[row['id']]['interpretation']['pair'];answers={}
            if a.verify_k and order3:
                before,after=[CLASSES[c] for c in pair]
                question=(f'Does the bright supported candidate region change from {before} at T1 to {after} at T2? '
                    'Ignore darkened outside pixels and holes. Unchanged or reversed transitions are no. '
                    'Reply JSON {"change":"yes" or "no", "before":"visible class", "after":"visible class", "reason":"local visual evidence"}.')
                for index in order3[:a.verify_k]:
                    response=request(folder/'verification'/str(index),'Judge only the supported candidate region in the two local-date images.',question,local_views(images,r2[index]),API_SEED)
                    answer=response['parsed'].get('change');answers[index]=dict(answer=answer if answer in ('yes','no') else 'unparsed',cost_cny=response['cost_cny'])
            truth=read_image(row['gt_mask'])>0;scores=[]
            for k in GRID:
                scores.extend([score(r1,order1[:k],truth,ex,'R1_geometry_dino',k),
                    score(r2,list(range(min(k,len(r2)))),truth,ex,'R2_native_order',k),score(r2,order3[:k],truth,ex,'R3_dino_noverify',k)])
            scores.append(score(r2,list(range(len(r2))),truth,ex,'R2_all',0))
            if a.verify_k:
                chosen=[i for i in order3[:a.verify_k] if answers[i]['answer']!='no']
                scores.append(score(r2,chosen,truth,ex,'R3_verified',a.verify_k))
            record=dict(id=row['id'],split=audit['split'][row['id']],group=group(row),remaining_gt_pixels=int((truth&~ex).sum()),
                scores=scores,text_proposal_upper=proposal_metrics(r2,truth,ex),geometry_proposal_upper=proposal_metrics(r1,truth,ex),
                verification=answers,cost_cny=sum(v['cost_cny'] for v in answers.values()))
            write_json(result,record);records.append(record)
            write_json(out/f'summary_{a.split}.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))
            print('retrieval',len(records),row['id'],flush=True)
    finally:text.pool.close()
    write_json(out/f'summary_{a.split}.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))


if __name__=='__main__':main()
