"""Opt-in frozen development candidate provider; no labels and no answer edits.

Integration is explicit: callers supply a provider for the current development
image, then expose find_missed_schema() and delegate the tool method to offer().
This adapter does not generate new candidates or change the shared tool registry.
"""
from pathlib import Path
import numpy as np
from .state import Mask, ToolError
from .analysis.io_retry import retry_io

SOURCES=('S1_02','S2','S3','S4','S5')
GROUPS=('second_appeared','second_disappeared','levir_any')


def find_missed_schema():
    return dict(type='function',function=dict(name='find_missed',description=(
        'Discover cached multi-source change proposals outside an existing answer. '
        'Returns a proposal handle and provenance, without modifying the answer. '
        'Inspect/select proposals, convert selected proposals with to_mask(set_current=false), '
        'ask for verification, and decide whether to combine them with the answer.'),parameters=dict(
        type='object',properties=dict(current=dict(type='string',description='Existing answer mask handle; defaults to current answer.'),
            limit=dict(type='integer',enum=[5,10,20],default=20)),additionalProperties=False)))


class CachedMissedCandidates:
    def __init__(self,work,group,stem,shape):
        if group not in GROUPS:
            raise ValueError('Frozen development candidate provider supports SECOND and LEVIR only')
        self.work,self.group,self.stem,self.shape=Path(work),group,stem,tuple(shape)
        self.ranked=None

    def load(self):
        if self.ranked is not None: return self.ranked
        queues={}
        for source in SOURCES:
            path=self.work/'candidates'/self.group/source/(self.stem+'.npz')
            def read():
                with np.load(path,allow_pickle=False) as z:
                    shape=tuple(int(v) for v in z['shape']);bits=z['masks'].copy()
                return shape,bits
            shape,bits=retry_io(read)
            if shape!=self.shape: raise ValueError('Cached candidate shape differs from task image')
            rows=[]
            for i,packed in enumerate(bits):
                mask=np.unpackbits(packed)[:int(np.prod(shape))].reshape(shape).astype(bool)
                idx=np.flatnonzero(mask).astype(np.int32)
                rows.append(dict(source=source,sources=[source],original=i,idx=idx))
            queues[source]=sorted(rows,key=lambda r:(-len(r['idx']),r['original']))
        ranked=[];seen={}
        for i in range(max((len(v) for v in queues.values()),default=0)):
            for source in SOURCES:
                if i>=len(queues[source]): continue
                row=queues[source][i];identity=row['idx'].tobytes()
                if identity in seen:
                    seen[identity]['sources'].append(source);continue
                seen[identity]=row;ranked.append(row)
        self.ranked=ranked
        return ranked

    def candidates(self,current,limit):
        if limit not in (5,10,20): raise ToolError('bad_args','limit must be 5, 10 or 20')
        if current.shape!=self.shape: raise ToolError('bad_args','Current answer shape differs from task image')
        rows=[];seen=set()
        for rank,original in enumerate(self.load(),1):
            idx=original['idx'][~current.ravel()[original['idx']]]
            if not len(idx): continue
            identity=idx.tobytes()
            if identity in seen: continue
            seen.add(identity)
            # One original proposal remains one proposal after subtraction. It
            # may become disconnected: to_mask/ask will use its resulting
            # connected components, whose outlines get fresh verifier records.
            # Never carry historical verifier IDs or answers into this adapter.
            yy,xx=np.divmod(idx,self.shape[1])
            rows.append(dict(id=len(rows)+1,idx=idx,area=len(idx),score=1./rank,
                bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                sources=list(original['sources']),source_rank=rank,original=original['original']))
            if len(rows)==limit:break
        return rows


def offer(toolbox,provider,current=None,limit=20):
    handle=current or toolbox.session.current_mask
    if handle is None: raise ToolError('bad_args','Choose an existing answer mask before finding missed candidates')
    item=toolbox.get(handle,Mask)
    rows=provider.candidates(item.data,limit)
    result=toolbox._props(rows,'t1' if toolbox.session.change_direction=='disappeared' else 't2',parents=(handle,))
    result.update(candidates=[dict(id=r['id'],area=r['area'],sources=r['sources'],rank=r['source_rank']) for r in rows],
        reference_answer=handle,ranking='frozen_source_round_robin_area',answer_changed=False,
        candidate_scope='cached_development_sources_minus_current_answer',
        granularity='one cropped source proposal; to_mask/ask recomputes connected components')
    return result
