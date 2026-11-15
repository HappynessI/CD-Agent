"""Independent opt-in verifier adapter. Default ApiAsk remains unchanged.

Toolbox.verify must call backend.render(rgb, region, bbox) when this adapter is
explicitly selected. Calls are serial; source candidates and labels never enter
its prompts. All decisions remain advisory to the Agent.
"""
from pathlib import Path
import os
from .analysis.missed_verifier_study import marked_images, parse_response, MODEL
from .analysis.io_retry import write_json
from .scope_t13 import session_prompts
from .planner import Planner, RunStopped


class MissedVerifier:
    context=True

    def __init__(self,directory,variant='four',session=None,client=None,guard=None,seed=0):
        if variant not in ('four','som','structured'): raise ValueError('Unknown verifier variant')
        self.directory=Path(directory);self.variant=variant;self.session=session
        self.client,self.guard,self.seed=client,guard,seed
        self.pending=[];self.number=0;self.halted=False

    def render(self,rgb,region,bbox):
        return marked_images(rgb,region,bbox,self.variant)

    def __call__(self,pairs,question,level='L1'):
        if self.halted:
            raise RunStopped('Verifier session halted after a configuration or quota error')
        if self.guard is not None:
            self.guard.check()
        shared_stop=os.environ.get('AGENT_STOP_FILE')
        if shared_stop and Path(shared_stop).exists():
            raise RunStopped('Shared verifier stop exists; resolve recorded error before resuming')
        settings=session_prompts(self.session)
        question=settings['CORRESPONDENCE']
        if self.variant!='four':
            question+='\nThe yellow mark 1 identifies the candidate inside the existing red contour in all four views; it is an annotation, not image evidence.'
        if self.variant=='structured':
            question+=('\nUse the same change definition above. Before the final change decision, describe the object at T1 and T2, '
                'whether they are the same persisting object, and any appearance-only explanation. '
                'Return JSON with t1_state, t2_state, same_object, appearance_only, change, type, reason. '
                'The change field must still be yes or no, using the same decision rule as above.')
        answers=[];events=[];total=dict(input=0,cached=0,output=0);cost=0.
        for images in pairs:
            self.number+=1;folder=self.directory/f'{self.number:06d}';folder.mkdir(parents=True,exist_ok=True)
            content=[]
            for i,(label,image) in enumerate(zip(('T1 close-up','T2 close-up','T1 wider view','T2 wider view'),images)):
                path=folder/f'{i}.jpg';image.save(path,quality=92)
                content.extend([dict(type='text',text=label),dict(type='image_url',image_url=dict(url=str(path)))])
            content.append(dict(type='text',text=question))
            messages=[dict(role='system',content=settings['SYSTEM']),dict(role='user',content=content)]
            def record(event):
                safe={k:v for k,v in event.items() if k != 'request'}
                if safe.get('error_body'):
                    secret=os.environ.get('DASHSCOPE_API_KEY')
                    if secret: safe['error_body']=safe['error_body'].replace(secret,'[credential omitted]')
                events.append(safe);write_json(folder/'response.json',safe)
                code=event.get('http_status')
                body=(event.get('error_body') or '').lower()
                fatal=code in (401,402,403,404) or event.get('quota_error') or any(
                    word in body for word in ('insufficient balance','payment required','arrearage','欠费'))
                if event.get('error_type') and fatal:
                    self.halted=True
                    halt=dict(reason=f'Verifier configuration/quota error: HTTP {code}',
                              http_status=code,response_path=str(folder/'response.json'))
                    if self.session is not None: self.session.verifier_halted=halt
                    write_json(self.directory/'HALT.json',halt)
                    if self.guard is not None:
                        self.guard.trip(halt['reason'])
                    elif shared_stop:
                        write_json(Path(shared_stop),halt)
                    raise RunStopped(halt['reason'])
                if event.get('error_type'): raise RuntimeError('Verifier request failed; no automatic retry')
            planner=Planner(seed=self.seed+self.number,client=self.client,guard=self.guard,record=record,max_attempts=1,
                parameters=dict(model=MODEL,temperature=0,enable_thinking=False,max_tokens=256,
                    logprobs=True,top_logprobs=5,response_format={'type':'json_object'}))
            write_json(folder/'request.json',dict(messages=messages,parameters=planner.parameters,variant=self.variant))
            try:
                result=planner.complete(messages,[],self.number)
            except Exception as exc:
                exc.backend_meta=dict(backend='missed_verifier',model=MODEL,variant=self.variant,
                    attempts=events,tokens={k:total[k]+planner.tokens[k] for k in total},cost_cny=cost+planner.cost,halted=self.halted)
                raise
            parsed=parse_response(result['raw'])
            answers.append(dict(answer=parsed['answer'],type=parsed['parsed'].get('type'),reason=parsed['parsed'].get('reason'),
                raw=parsed['raw'],usage=result['usage'],level=level,p_yes=None,finish_reason=parsed['finish_reason'],
                parse_failure=parsed['parse_failure'],input_path=str(folder)))
            cost+=planner.cost
            for k in total:total[k]+=planner.tokens[k]
        return answers,dict(backend='missed_verifier',model=MODEL,variant=self.variant,attempts=events,tokens=total,cost_cny=cost)
