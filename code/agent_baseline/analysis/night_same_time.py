"""Six frozen cases, two same-time counterfactuals; no GT or method selection."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.view import ask_images
from agent_baseline.planner import Planner,usage_tokens,request_cost
from .io_retry import read_image,write_json
from .missed_sources import load_npz,unpack
from .missed_verifier_study import parse_response

MODEL='qwen3.7-flash'
UIDS=('second_appeared:train_1065:1','second_appeared:train_1065:4',
      'second_disappeared:train_1065:1','second_disappeared:train_1065:2',
      'levir_any:train_106:1','levir_any:train_106:3')


def load(path):return json.loads(Path(path).read_text())


def same_time_images(rgb,mask,bbox,time):
    pair=[rgb[time].copy(),rgb[time].copy()]
    if not np.array_equal(pair[0],pair[1]):raise ValueError('Counterfactual RGB dates differ')
    # ask_images explicitly orders close T1/T2, wider T1/T2; render anew so
    # positional date labels remain correct rather than copying old panels.
    return ask_images(pair,mask,bbox,True)


def summarize(out):
    rows=[load(p) for p in sorted((out/'answers').glob('*.json'))]
    events=[load(p) for p in sorted((out/'events').glob('*.json'))]
    write_json(out/'summary.json',dict(expected=12,completed=len(rows),attempted=len(list((out/'attempted').glob('*.json'))),
        recorded_cost_cny=sum(e['cost_cny'] for e in events),responses=rows,
        interpretation='Same RGB at both dates; yes contradicts temporal invariance, not a measured natural-task benefit or semantic pixel label. Fixed cases have prior visual exposure.'))
    lines=['# 固定六例同日期反事实','',
        '同一RGB分别作为T1/T1或T2/T2；保留原mask/bbox/方向/问法/模型参数。原四图重新渲染，GT不参与。yes违反此构造的无时相变化条件，但不等于自然任务收益、人工语义标签或线上晋级证据。固定6例来自既有review_pack，已有视觉曝光。','',
        '|UID|同日期输入|原V0|反事实回答|finish_reason|响应证据|','|---|---|---|---|---|---|']
    for r in rows:lines.append('|'+ '|'.join(str(r.get(k)) for k in ('uid','condition','original_answer','answer','finish_reason','event_path'))+'|')
    lines+=['',f'完成 {len(rows)}/12；已记录usage费用 ¥{sum(e["cost_cny"] for e in events):.7f}。每配置最多一次调用；未完成/失败不自动重试。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def run(a):
    design=dict(uids=list(UIDS),conditions=['T1_T1','T2_T2'],max_requests=12,max_attempts_per_configuration=1,
        model=MODEL,gt_access=False,selection='fixed six UIDs from existing review_pack/review.md; not selected by new results',
        rendering='ask_images([same RGB,same RGB], original mask, original bbox, context=True)',
        parameters='exact original V0 request parameters including seed; fresh API, no offline answer reuse',
        purpose='temporal invariance counterfactual only; no natural-task benefit claim or promotion')
    write_json(a.out/'design.json',design)
    items=load(a.views/'manifest.json')
    lookup={c['uid']:(item,i,c) for item in items for i,c in enumerate(item['candidates'])}
    for uid in UIDS:
        item,index,candidate=lookup[uid]
        z=load_npz(Path(item['masks']));mask=unpack(z['masks'],tuple(z['shape']))[index]
        rgb=[read_image(item['image_'+t],rgb=True) for t in ('t1','t2')]
        original_path=a.pilot/'answers/four'/(uid.replace(':','__')+'.json');original=load(original_path)
        request_path=Path(original['input_path'])/'request.json';request=load(request_path)
        if request['parameters']['model']!=MODEL:raise ValueError('Original model differs')
        for time,condition in enumerate(('T1_T1','T2_T2')):
            name=uid.replace(':','__')+'__'+condition;dest=a.out/'answers'/(name+'.json');attempt=a.out/'attempted'/(name+'.json')
            if dest.exists():continue
            if attempt.exists():raise RuntimeError('Configuration already attempted; no automatic retry')
            directory=a.out/'inputs'/name;directory.mkdir(parents=True,exist_ok=True)
            images=same_time_images(rgb,mask,candidate['bbox'],time)
            messages=json.loads(json.dumps(request['messages']));blocks=[b for m in messages if isinstance(m['content'],list) for b in m['content'] if b.get('type')=='image_url']
            if len(blocks)!=4 or len(images)!=4:raise ValueError('Expected original four-image order')
            for n,(block,image) in enumerate(zip(blocks,images)):
                path=directory/f'{n}.jpg';image.save(path,quality=92);block['image_url']['url']=str(path)
            parameters=dict(request['parameters']);seed=parameters.pop('seed')
            write_json(directory/'request.json',dict(messages=messages,parameters=dict(parameters,seed=seed),
                uid=uid,condition=condition,original_request=str(request_path),original_answer=str(original_path),
                same_rgb_pixel_equal=True,source_image=item['image_'+('t1' if time==0 else 't2')],
                bbox=candidate['bbox'],direction=item['direction'],panel_order=['T1 close-up','T2 close-up','T1 wider view','T2 wider view']))
            event_path=a.out/'events'/(name+'.json')
            def capture(event):
                body=event.get('error_body');secret=os.environ.get('DASHSCOPE_API_KEY')
                if body and secret:body=body.replace(secret,'[credential omitted]')
                write_json(event_path,dict(raw=event.get('raw'),usage=event.get('usage'),http_status=event.get('http_status'),
                    error_type=event.get('error_type'),error_body=body,model=MODEL,
                    cost_cny=request_cost(MODEL,usage_tokens(event.get('usage')))))
            write_json(attempt,dict(uid=uid,condition=condition,input_path=str(directory),max_attempts=1))
            try:
                result=Planner(seed=seed,parameters=parameters,max_attempts=1,record=capture).complete(messages,[],1)
                write_json(dest,dict(uid=uid,condition=condition,original_answer=original.get('answer'),
                    **parse_response(result['raw']),input_path=str(directory),event_path=str(event_path)))
            finally:summarize(a.out)
    summarize(a.out)


def main():
    p=argparse.ArgumentParser()
    for key in ('views','pilot','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    run(a)

if __name__=='__main__':main()
