"""Explicit N9-only process adapter; default tools and other task launches unchanged."""
import json
import os
import numpy as np
from PIL import Image, ImageDraw
from agent_baseline.state import Mask, Instances, ToolError
from agent_baseline.instances import from_dense
from agent_baseline.tools_t13 import Features
from .n9_t3_retrieval import retrieve, clean_candidates, scores, order
from .io_retry import write_json, save_image

COUNTS=['Only changes with the SAME directed land-cover transition as the provided visual example: both the T1 source category and T2 destination category must match. Infer these from the images, not from surrounding changes.']
NOT_COUNTS=['A different transition, even if it is a real change; unchanged appearance, shadow, season or misregistration. Surrounding objects outside the marked support are not the target.']
QUESTION='In every image the upper panel is the task example and the lower panel is the candidate. The first two images are T1 and T2 close-ups; the next two are T1 and T2 wider context. Compare ONLY marked support. Infer both directed transitions. Does the candidate have the SAME source AND destination land-cover categories as the example? Reply with JSON only: {"change":"yes" or "no","type":"real" or "other","reason":"state the example T1→T2 and candidate T1→T2, then explain match or mismatch"}.'

def align_user_instruction(messages):
    for message in messages:
        if message.get('role')!='user':continue
        content=message.get('content')
        if isinstance(content,list):
            for block in content:
                if block.get('type')=='text':block['text']=block['text'].replace('including the example itself.', 'excluding the example itself; it is reference support, not an output target.')
    return messages

def reference(tool):
    from agent_baseline.examples_t3x import Example
    handles=getattr(tool.session,'external_examples',())
    if handles:
        value=tool.get(handles[0][0],Example)
        return [value.images[t] for t in ('t1','t2')],value.mask,value.backend,False
    mask=tool.get(tool.session.exemplar_handle,Mask).data
    return tool.rgb,mask,tool._vision,True

def configuration(tool):
    config=json.loads(open(os.environ['N9_INFERENCE_CONFIG']).read())
    kind='T3.3' if getattr(tool.session,'external_examples',()) else 'T3.1'
    value=config[kind]
    if value['rank'] not in ('delta','joint') or value['k'] not in (5,8):raise ValueError('Frozen N9 configuration required')
    return value

def retrieve_example(self):
    images,example,backend,same=reference(self);config=configuration(self)
    _,masks,audit=retrieve(self.rgb,images,example,self._vision,backend,same)
    chosen=audit['clean_orders'][config['rank']][:config['k']]
    values=[audit['clean_scores'][config['rank']][i] for i in chosen]
    result=self._props(from_dense(masks[chosen],values),'t2',())
    folder=self.directory/'n9_retrieval';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/f'call_{len(list(folder.glob("call_*.json"))):03d}.json',dict(audit,selected=chosen,configuration=config))
    return dict(result,proposal_only=True,ranked_ids=list(range(1,len(chosen)+1)))

def match_exemplar(self,props,exemplar,mode='change',negative=None):
    if mode!='change' or negative is not None:raise ToolError('bad_args','N9 uses the provided positive directed-transition example and change mode.')
    images,example,backend,same=reference(self)
    allowed={'mask':self.session.exemplar_handle} if same else {'example':self.session.external_examples[0][0]}
    external_mask={'mask':self.session.external_examples[0][1]} if not same else allowed
    if exemplar not in (allowed,external_mask):raise ToolError('bad_args','Use the task-provided positive example handle.')
    item=self.get(props,Instances);masks=np.zeros((len(item.items),*self.session.shape),bool)
    for i,row in enumerate(item.items):masks[i].ravel()[row['idx']]=True
    clean,ids,audit=clean_candidates(masks,example,same)
    q=[self._feature(t)[1] for t in ('t1','t2')]
    r=q if same else [Features(backend('extract_features',time=t),example.shape,t) for t in ('t1','t2')]
    rank=configuration(self)['rank'];values=scores(clean,q,r,example)[rank];indices=order(values)
    rows=from_dense(clean[indices],[float(values[i]) for i in indices]);result=self._props(rows,item.time,(props,))
    return dict(result,similarities=[dict(id=x['id'],score=x['score']) for x in rows],mode='change',n9_rank=rank,removed_self_matches=sum(x['removed'] for x in audit))

def install():
    from agent_baseline import tools,backends,loop_json,scope_t13
    from agent_baseline.planner import Planner
    if os.environ.get('N9_ONLINE_OPT_IN')!='1':raise RuntimeError('Explicit N9 opt-in required')
    original_complete=Planner.complete
    def complete(self,messages,*args,**kwargs):
        return original_complete(self,align_user_instruction(messages),*args,**kwargs)
    Planner.complete=complete
    original_card=scope_t13.session_card
    def card(session):
        return dict(original_card(session),target='the example-directed transition',plural='changes matching the example-directed transition',counts=COUNTS,not_counts=NOT_COUNTS)
    def prompts(session):
        return dict(SYSTEM='Verify whether the candidate matches the visual example directed transition. '+COUNTS[0]+' '+NOT_COUNTS[0],VERIFY=QUESTION,CORRESPONDENCE=QUESTION,PSEUDO_TYPES=['real','other'],SCOPE_CARD=dict(counts=COUNTS,not_counts=NOT_COUNTS))
    scope_t13.session_card=tools.session_card=loop_json.session_card=card
    scope_t13.session_prompts=tools.session_prompts=backends.session_prompts=prompts
    original_system=loop_json.system_prompt
    def system_prompt(*args,**kwargs):
        path,template,text=original_system(*args,**kwargs)
        text=text.replace('It sees only crops of the two images and knows nothing of your reasoning.',
            'It sees paired example and candidate crops at T1 and T2 and independently checks whether their directed transitions match; it knows nothing of your reasoning.')
        text+='\nN9 task: '+COUNTS[0]+' '+NOT_COUNTS[0]+' Retrieve proposals with retrieve_example, or segment and rank your own proposals with match_exemplar. Neither tool chooses your final answer.'
        return path,template,text
    loop_json.system_prompt=system_prompt
    original_verify=tools.Toolbox.verify
    def verify(self,expanded,source):
        from agent_baseline.view import ask_images
        images,example,_,_=reference(self);y,x=np.nonzero(example)
        refs=ask_images(images,example,[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],True)
        backend=self.ask_backend;old=getattr(backend,'render',None)
        def render(rgb,region,bbox):
            candidates=ask_images(rgb,region,bbox,True);out=[]
            root=self.directory/'n9_verifier_inputs';root.mkdir(parents=True,exist_ok=True)
            call=len(list(root.glob('pair_*')));folder=root/f'pair_{call:04d}';folder.mkdir()
            for i,(ref,candidate) in enumerate(zip(refs,candidates)):
                ref=Image.fromarray(np.asarray(ref));candidate=Image.fromarray(np.asarray(candidate))
                canvas=Image.new('RGB',(max(ref.width,candidate.width),ref.height+candidate.height+48),'white')
                canvas.paste(ref,(0,24));canvas.paste(candidate,(0,ref.height+48));draw=ImageDraw.Draw(canvas)
                draw.text((4,4),'EXAMPLE '+('T1' if i%2==0 else 'T2'),fill='black');draw.text((4,ref.height+28),'CANDIDATE',fill='black')
                save_image(folder/f'{i}.png',np.asarray(canvas));out.append(canvas)
            write_json(folder/'input.json',dict(source=source,example_and_candidate=True,same_transition_question=QUESTION,candidate_bbox=list(bbox)))
            return out
        backend.render=render
        try:return original_verify(self,expanded,source)
        finally:
            if old is None:del backend.render
            else:backend.render=old
    tools.Toolbox.verify=verify;tools.Toolbox.match_exemplar=match_exemplar;tools.Toolbox.retrieve_example=retrieve_example
    entry=tools.schema('retrieve_example','Retrieve frozen N9 candidates matching the provided visual example directed transition. Returns proposals only; inspect, verify, edit and select your own answer.',{},[])
    tools.SCHEMA_BY_NAME['retrieve_example']=entry['function']['parameters'];previous=tools.session_schemas
    def schemas(*args,**kwargs):return previous(*args,**kwargs)+[entry]
    tools.session_schemas=schemas
