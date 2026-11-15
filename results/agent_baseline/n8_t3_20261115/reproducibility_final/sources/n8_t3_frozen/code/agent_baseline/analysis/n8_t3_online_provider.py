"""Opt-in live retrieval provider. No GT, offline prediction or verifier-answer inputs."""
import os
import json
from pathlib import Path
import numpy as np
from agent_baseline.instances import from_dense
from agent_baseline.state import Mask,ToolError
from .t3_geometry import eligible
from .n8_t3_common import clean_pool,ranked_features,observable_example,packed_save,CHANNELS
from .n8_t3_system import construct_pool
from .io_retry import read_text,write_json


def proposals(images,example,backend,arm,config):
    raw=np.concatenate([backend('segment_everything',time=t) for t in ('t1','t2')])
    geometry=raw[eligible(raw,example,images)]
    if arm=='baseline':
        values,orders=ranked_features(geometry,example,backend);order=orders['delta'];selected=order[:8]
        masks=geometry[selected]&~example
        scores=[float(values['delta'][i]) for i in selected]
        audit=dict(arm=arm,rank='delta',k=8,raw_candidates=len(raw),geometry_candidates=len(geometry),
            candidate_pool_size=len(geometry),selected_pool_ids=selected,similarities=scores,
            whitelist=['RGB T1/T2','example E','DINO features','SAM proposals'])
    elif arm=='n8':
        clean,old_ids,cleaning=clean_pool(geometry,example)
        probabilities=[backend('classify',time=t,classes=CHANNELS[1:]) for t in ('t1','t2')]
        observable,dense=observable_example(probabilities,example)
        pool,sources=construct_pool(clean,dense,probabilities,observable,config['system_method'])
        values,orders=ranked_features(pool,example,backend);order=orders[config['rank']];selected=order[:config['k']]
        masks=pool[selected]&~example;scores=[float(values[config['rank']][i]) for i in selected]
        audit=dict(arm=arm,rank=config['rank'],k=config['k'],system_method=config['system_method'],raw_candidates=len(raw),
            geometry_candidates=len(geometry),clean_candidates=len(clean),candidate_pool_size=len(pool),
            selected_pool_ids=selected,similarities=scores,observable=observable,clean_original_ids=old_ids,
            cleaning=cleaning,selected_sources=[sources[i] for i in selected],
            global_gate=False,internal_verifier=False,whitelist=['RGB T1/T2','example E','DINO features','SAM proposals','predicted SegEarth probabilities'])
    else:raise ValueError('Expected baseline or n8')
    return masks,scores,audit


def schema():
    from agent_baseline.tools import schema as make_schema,arg
    return make_schema('retrieve_example',
        'Retrieve candidate regions showing the same directed change as the supplied example. Returns a ranked props handle and IDs; does not select the answer or submit. Inspect candidates, independently verify change, and choose/edit the regions you consider justified. An empty proposal set is allowed.',
        dict(mask=arg('string',description='The task-provided visual-example mask handle.')),['mask'])


def retrieve_example(self,mask):
    example=self.get(mask,Mask).data
    if not example.any():raise ToolError('bad_args','Example mask is empty')
    arm=os.environ['N8_ONLINE_ARM'];config=json.loads(read_text(Path(os.environ['N8_INFERENCE_CONFIG'])))
    masks,scores,audit=proposals(self.rgb,example,self._vision,arm,config)
    root=self.directory/'n8_retrieval';folder=root/f'call_{len(list(root.glob("call_*"))):03d}'
    packed_save(folder/'selected_candidates.npz',masks,scores=np.asarray(scores))
    write_json(folder/'inference.json',dict(**audit,example_handle=mask,
        policy='Provider only returns proposals. No current-answer mutation, GT access, offline mask replay or inherited verifier answer.'))
    rows=from_dense(masks,scores)
    result=self._props(rows,'t2',(mask,))
    result.update(ranked_ids=[r['id'] for r in rows],scores=[dict(id=r['id'],score=r['score'],area=r['area']) for r in rows],
                  proposal_only=True)
    return result


def install():
    """Only called by the explicit N8 online launcher; default tools stay unchanged."""
    from agent_baseline import tools
    entry=schema();tools.SCHEMA_BY_NAME['retrieve_example']=entry['function']['parameters']
    tools.Toolbox.retrieve_example=retrieve_example
    previous=tools.session_schemas
    def schemas(*args,**kwargs):return previous(*args,**kwargs)+[entry]
    tools.session_schemas=schemas
