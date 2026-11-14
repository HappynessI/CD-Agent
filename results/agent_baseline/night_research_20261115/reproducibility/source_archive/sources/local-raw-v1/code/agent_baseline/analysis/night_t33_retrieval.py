"""R1/R2/R3 on the frozen cross-image T3.3 pilot; all GT pixels retained."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.tools_t13 import Features,pooled,pool_features,cosine
from agent_baseline.instances import from_dense
from agent_baseline.exemplar_semantics import request,CLASSES
from .io_retry import read_text,read_image,write_json,exists
from .t3_routes import load_dev,group,GRID
from .t3_routes_cache import CachedVision,read_npz,write_npz
from .t3_diag import directed_proposals,proposal_metrics
from .t3_geometry import descriptor
from .night_t3_retrieval import TextProposals,local_views,score,summaries
from .night_t3_grounding import API_SEED


def external_scores(masks,query_features,reference_features,reference_mask):
    if not len(masks):return np.zeros(0)
    items=from_dense(masks,np.ones(len(masks)))
    difference=pooled(query_features[1],items)-pooled(query_features[0],items)
    indices=np.flatnonzero(reference_mask)
    vector=pool_features(reference_features[1],indices)-pool_features(reference_features[0],indices)
    return cosine(difference,vector)


def geometry_indices(masks,query_images,reference_mask,reference_images):
    reference=descriptor(reference_mask,reference_images);indices=[]
    for index,mask in enumerate(masks):
        candidate=descriptor(mask,query_images)
        if candidate and .25<=candidate[0]/reference[0]<=4 and .333333<=candidate[1]/reference[1]<=3 and np.linalg.norm(candidate[2]-reference[2])/np.sqrt(6)<=.25:
            indices.append(index)
    return indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--parse-variant',choices=['B0','B1','B2'],default='B1');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm GPU')
    audit=json.loads(read_text(a.night/'t33/audit.json'));pilot=audit['pilot_ids']
    source=json.loads(read_text(a.night/'t33'/a.parse_variant/'summary.json'))
    parses={r['id']:r for r in source['records']}
    if set(parses)!=set(pilot):raise ValueError('Complete fixed T3.3 pilot parsing required')
    rows=[json.loads(line) for line in read_text(a.run/'t3x/tasks/dev/tasks.jsonl').splitlines() if line.strip()]
    rows=[r for r in rows if r['id'] in pilot]
    query_cache=CachedVision(a.run/'t3_diag/cache/vision');reference_cache=CachedVision(a.run/'vision_t13')
    cache_rows={str(Path(r['image_t1']).resolve()):r for r in load_dev(a.run)}
    text=TextProposals(a.run,a.night);out=a.night/'t33/retrieval'/a.parse_variant;records=[]
    write_json(out/'settings.json',dict(pilot_ids=pilot,parse_variant=a.parse_variant,verify_k=8,
        model='qwen3.7-flash',source=str(a.night/'t33'/a.parse_variant/'summary.json'),
        query_cache=str(a.run/'t3_diag/cache/vision'),reference_cache=str(a.run/'vision_t13'),
        scoring='external exemplar coordinates never subtracted from query truth',
        R1='external change vector plus same area/aspect/colour gates',
        R2='parsed directed text proposals',R3='same text pool, external DINO ranking, local support-prompt verification',
        code=str(Path(__file__).resolve())))
    try:
        for row in rows:
            folder=out/'tasks'/row['id'];final=folder/'metrics.json'
            if exists(final):records.append(json.loads(read_text(final)));continue
            images=[read_image(row[k],rgb=True) for k in ('image_t1','image_t2')];shape=images[0].shape[:2]
            example=row['examples'][0];ref_images=[read_image(example[k],rgb=True) for k in ('image_t1','image_t2')]
            ref_mask=read_image(example['mask'])>0;zero=np.zeros(shape,bool)
            query=query_cache.for_row(row);reference=reference_cache.for_row(example)
            qfeatures=[Features(query('extract_features',time=t),shape,t) for t in ('t1','t2')]
            rfeatures=[Features(reference('extract_features',time=t),ref_mask.shape,t) for t in ('t1','t2')]
            prepared=folder/'prepared.npz';pair=parses[row['id']]['pair']
            if exists(prepared):
                saved=read_npz(prepared);r1=saved['r1'];r2=saved['r2'];order1=saved['order1'].tolist();order3=saved['order3'].tolist()
            else:
                all_masks=np.concatenate([query('segment_everything',time=t) for t in ('t1','t2')])
                r1=all_masks[geometry_indices(all_masks,images,ref_mask,ref_images)]
                similarities=external_scores(r1,qfeatures,rfeatures,ref_mask)
                order1=sorted(range(len(r1)),key=lambda i:(-float(similarities[i]),i))
                if None in pair or pair[0]==pair[1]:r2=np.zeros((0,*shape),bool)
                else:
                    cached_row=cache_rows[str(Path(row['image_t1']).resolve())]
                    masks=[text.get(cached_row,CLASSES[c],t) for c in pair for t in ('t1','t2')]
                    r2=directed_proposals(*masks,shape)
                similarities=external_scores(r2,qfeatures,rfeatures,ref_mask)
                order3=sorted(range(len(r2)),key=lambda i:(-float(similarities[i]),i))
                write_npz(prepared,r1=r1,r2=r2,order1=np.asarray(order1,int),order3=np.asarray(order3,int))
                write_json(folder/'provenance.json',dict(predicted_pair=pair,text_cache_reused=text.reused));text.reused=[]
            answers={}
            for index in order3[:8]:
                before,after=[CLASSES[c] for c in pair]
                question=(f'Does the bright supported candidate change from {before} at T1 to {after} at T2? '
                    'Ignore darkened outside and holes. Reply JSON {"change":"yes" or "no", "before":"class", "after":"class", "reason":"local visual evidence"}.')
                response=request(folder/'verification'/str(index),'Judge only the supported candidate in the two local-date images.',question,local_views(images,r2[index]),API_SEED)
                answer=response['parsed'].get('change');answers[index]=dict(answer=answer if answer in ('yes','no') else 'unparsed',cost_cny=response['cost_cny'])
            truth=read_image(row['gt_mask'])>0;scores=[]
            for k in GRID:
                scores.extend([score(r1,order1[:k],truth,zero,'R1_geometry_dino',k),
                    score(r2,list(range(min(k,len(r2)))),truth,zero,'R2_native_order',k),score(r2,order3[:k],truth,zero,'R3_dino_noverify',k)])
            scores.extend([score(r2,list(range(len(r2))),truth,zero,'R2_all',0),
                score(r2,[i for i in order3[:8] if answers[i]['answer']!='no'],truth,zero,'R3_verified',8)])
            record=dict(id=row['id'],split=audit['split'][row['id']],group=group(row),remaining_gt_pixels=int(truth.sum()),
                scores=scores,text_proposal_upper=proposal_metrics(r2,truth,zero),geometry_proposal_upper=proposal_metrics(r1,truth,zero),
                verification=answers,cost_cny=sum(v['cost_cny'] for v in answers.values()))
            write_json(final,record);records.append(record)
            write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))
            print('T3.3 retrieval',len(records),row['id'],flush=True)
    finally:text.pool.close()
    write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summaries(records),cost_cny=sum(r['cost_cny'] for r in records)))


if __name__=='__main__':main()
