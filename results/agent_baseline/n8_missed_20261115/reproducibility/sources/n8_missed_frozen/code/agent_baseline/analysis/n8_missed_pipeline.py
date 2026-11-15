"""FM-2 residual gate and FM-3 real V0/evidence calls on N8 reconstructed masks."""
import argparse
import copy
import os
from pathlib import Path
import time
import numpy as np
from scipy import ndimage as ndi
from .n8_missed_assets import read,write,OUT,ROOT,HISTORY
from .n8_missed_prepare import load_masks,union,AREA_POWERS,RESIDUAL_GRID

VARIANTS=('V0','V0_evidence')
PILOT=ROOT/'CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot'

def cost_total(out):
    return sum(read(p).get('cost_cny',0.) for p in (out/'usage').glob('*.json'))

def rank_candidates(candidates,threshold,alpha):
    chosen=[c for c in candidates if c['semantic_delta']>=threshold]
    return sorted(chosen,key=lambda c:(-max(c['semantic_delta'],0)*max(c['dino'],0)*c['area']**alpha,c['index']))

def residual_state(candidates,masks,shape,threshold):
    gated=[c for c in candidates if c['semantic_delta']>=threshold]
    residual=union([masks[c['index']] for c in gated if c['dino']>=.20],shape)
    _,n=ndi.label(residual,structure=np.ones((3,3)))
    return dict(net_candidates=len(gated),residual_area=int(residual.sum()),residual_components=n)

def should_verify(state,minimum):
    return state['net_candidates']>0 and state['residual_components']>0 and state['residual_area']>=minimum

def make_request(template,image_paths,candidate,variant,index):
    messages=copy.deepcopy(template['messages']);number=0
    for block in messages[1]['content']:
        if block.get('type')=='image_url':block['image_url']['url']=str(image_paths[number]);number+=1
    if variant=='V0_evidence':
        c=candidate
        messages[1]['content'][-1]['text']+=(f"\nQuantitative evidence for the red-contour region only: area={c['area']} native pixels; "
            f"SegEarth mean building probability T1={c['p1']:.4f} -> T2={c['p2']:.4f}; "
            f"DINO temporal difference (1-cosine)={c['dino']:.4f}. These model-derived values are evidence, not ground truth; verify the two-date visual correspondence.")
    parameters=dict(template['parameters']);parameters['model']='qwen3.7-flash';parameters['seed']=1016+index
    return dict(messages=messages,parameters=parameters)

def prepare(a):
    tau=read(a.out/'frozen_semantic.json')['threshold']
    selection=read(a.out/'frozen_config.json') if a.split=='confirm' else None
    alphas=[selection['alpha']] if selection else AREA_POWERS
    area_min=selection['residual_area'] if selection else min(RESIDUAL_GRID)
    old={(r['group'],r['stem']):r for r in read(PILOT/'manifest.json')}
    requests=[];states=[]
    for item in read(a.out/'fm1_manifest.json'):
        if item['split']!=a.split:continue
        masks=load_masks(item['masks']);shape=tuple(item['shape'])
        residual=residual_state(item['candidates'],masks,shape,tau)
        rankings={str(alpha):[c['index'] for c in rank_candidates(item['candidates'],tau,alpha)[:10]] for alpha in alphas}
        state=dict(group=item['group'],stem=item['stem'],split=a.split,semantic_threshold=tau,
            **residual,rankings=rankings,
            generated_find_missed=True,vlm_eligible=should_verify(residual,area_min))
        states.append(state)
        if not state['vlm_eligible']:continue
        previous=old[item['group'],item['stem']]
        template_answer=read(PILOT/'answers/four'/(previous['candidates'][0]['uid'].replace(':','__')+'.json'))
        template=str(Path(template_answer['input_path'])/'request.json')
        for index in sorted({i for ids in rankings.values() for i in ids}):
            candidate=item['candidates'][index];uid=f"{item['group']}__{item['stem']}__n8_{index:04d}"
            requests.append(dict(uid=uid,group=item['group'],stem=item['stem'],split=a.split,mask_index=index,masks=item['masks'],
                candidate=candidate,image_t1=item['image_t1'],image_t2=item['image_t2'],template=template))
    write(a.out/f'fm2_states_{a.split}.json',states);write(a.out/f'requests_{a.split}.json',requests)
    write(a.out/'metric.json',dict(stage='fm2_prepared_'+a.split,status='complete',completed=len(states),candidate_requests=len(requests),
        cost_cny=cost_total(a.out),cumulative=True,find_missed_generation_rate=1.,vlm_task_eligible=sum(s['vlm_eligible'] for s in states)))
    print(f'FM2 {a.split}: {len(requests)} unique masks for two verifier arms.')

def verify(a):
    from agent_baseline.planner import Planner,usage_tokens,request_cost
    from .missed_verifier_study import parse_response
    from agent_baseline.view import ask_images
    from .io_retry import read_image
    cost=cost_total(a.out);request_count=len(list((a.out/'usage').glob('*.json')))
    rows=read(a.out/f'requests_{a.split}.json');loaded=None;rgb=None;masks=None
    started=time.monotonic();completed=0;initial_requests=request_count;initial_cost=cost
    for row in rows:
        for variant in VARIANTS:
            dest=a.out/'answers'/variant/(row['uid']+'.json')
            if dest.exists():completed+=1;continue
            # Recover a returned response if the process stopped before answer serialization.
            prior=sorted((a.out/'usage').glob(row['uid']+'__'+variant+'__*.json'))
            recovered=next((read(p) for p in reversed(prior) if read(p).get('raw') and not read(p).get('error_type')),None)
            if recovered:
                write(dest,dict(uid=row['uid'],variant=variant,**parse_response(recovered['raw']),usage=recovered['usage'],cost_cny=recovered['cost_cny'],recovered=True));completed+=1;continue
            if cost>=50:raise RuntimeError('Actual N8 find_missed budget reached; request allocation from main agent')
            if loaded!=row['masks']:
                masks=load_masks(row['masks']);rgb=[read_image(row['image_'+t],rgb=True) for t in ('t1','t2')];loaded=row['masks']
            template=read(row['template'])
            folder=a.out/'inputs'/variant/row['uid'];folder.mkdir(parents=True,exist_ok=True)
            image_folder=a.out/'views'/row['uid'];image_folder.mkdir(parents=True,exist_ok=True)
            image_paths=[image_folder/f'{i}.jpg' for i in range(4)]
            if not all(p.exists() for p in image_paths):
                images=ask_images(rgb,masks[row['mask_index']],row['candidate']['bbox'],True)
                for path,image in zip(image_paths,images):image.save(path,quality=92)
            request=make_request(template,image_paths,row['candidate'],variant,row['mask_index'])
            write(folder/'request.json',dict(request,uid=row['uid'],variant=variant))
            messages=request['messages'];parameters=dict(request['parameters']);seed=parameters.pop('seed')
            last={}
            def capture(event):
                nonlocal cost,request_count,last
                tokens=usage_tokens(event.get('usage'));charge=request_cost('qwen3.7-flash',tokens)
                body=event.get('error_body');secret=os.environ.get('DASHSCOPE_API_KEY')
                if body and secret:body=body.replace(secret,'[credential omitted]')
                last=dict(uid=row['uid'],variant=variant,split=a.split,model='qwen3.7-flash',raw=event.get('raw'),usage=event.get('usage'),tokens=tokens,
                    cost_cny=charge,http_status=event.get('http_status'),error_type=event.get('error_type'),error_body=body)
                write(a.out/'usage'/f"{row['uid']}__{variant}__{time.time_ns()}.json",last)
                cost+=charge;request_count+=1
                write(a.out/'metric.json',dict(stage='fm3_verify_'+a.split,status='running',completed=completed,requests=request_count,cost_cny=cost,cumulative=True))
                if event.get('error_type') and (event.get('http_status')!=429 or event.get('quota_error')):
                    raise RuntimeError('API failed; saved evidence, no answer assumed')
            try:
                answer=Planner(seed=seed,parameters=parameters,max_attempts=3,record=capture).complete(messages,[],1)
                write(dest,dict(uid=row['uid'],variant=variant,**parse_response(answer['raw']),usage=answer['usage'],cost_cny=last['cost_cny'],input_path=str(folder)))
            except Exception as exc:
                write(a.out/'verify_failure.json',dict(uid=row['uid'],variant=variant,error_type=type(exc).__name__,split=a.split))
                raise RuntimeError('N8 verifier failed; inspect archived usage and request') from None
            completed+=1
    write(a.out/f'fm3_verify_{a.split}.json',dict(status='complete',completed_answers=completed,new_requests=request_count-initial_requests,new_cost_cny=cost-initial_cost,cumulative_requests=request_count,
        cumulative_cost_cny=cost,elapsed_seconds=time.monotonic()-started,slurm_job=os.environ['SLURM_JOB_ID']))
    write(a.out/'metric.json',dict(stage='fm3_verified_'+a.split,status='complete',completed=completed,requests=request_count,cost_cny=cost,cumulative=True))
    print(f'FM3 {a.split} complete: {completed} answers; cumulative actual usage cost {cost:.6f} CNY.')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify']);p.add_argument('--split',choices=['tune','confirm'],required=True);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    globals()[a.command](a)

if __name__=='__main__':main()
