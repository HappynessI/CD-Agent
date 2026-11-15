"""Opt-in evidence@10 proposals only; no answer or verifier-cache access."""
import json
from pathlib import Path
import numpy as np
from .find_missed import CachedMissedCandidates
from .state import ToolError


class FrozenRankedCandidates(CachedMissedCandidates):
    ranking='frozen_evidence_top10_no_refill'

    def __init__(self,manifest,group,stem,shape):
        self.manifest=Path(manifest)
        super().__init__(self.manifest.parent,group,stem,shape)

    def load(self):
        if self.ranked is not None:return self.ranked
        settings=json.loads((self.manifest.parent/'settings.json').read_text())
        if settings['variant']!='V0':raise ValueError('This online provider requires frozen V0/four')
        items=json.loads(self.manifest.read_text())
        matches=[r for r in items if r['group']==self.group and r['stem']==self.stem]
        if len(matches)!=1:raise ValueError('Frozen ranked task missing or duplicated; never substitute pilot tasks')
        item=matches[0];indexes=item['rankings']['evidence']
        if indexes is None:raise ValueError('Frozen evidence ranking unavailable')
        if item['s5_gate']!=settings['s5_selection'][self.group]['gate']:raise ValueError('Frozen gate differs')
        path=Path(item['masks']);path=path if path.is_absolute() else self.manifest.parent/path
        with np.load(path,allow_pickle=False) as z:
            shape=tuple(map(int,z['shape']));packed=z['masks'].copy()
        if shape!=self.shape:raise ValueError('Frozen ranked mask shape differs')
        rows=[]
        for rank,index in enumerate(indexes[:10],1):
            candidate=item['candidates'][index]
            idx=np.flatnonzero(np.unpackbits(packed[index],count=int(np.prod(shape)))).astype(np.int32)
            # Deliberately whitelist provenance. Never read old_answer, GT,
            # reference labels or cached API responses from the offline record.
            rows.append(dict(idx=idx,source=candidate['source'],sources=list(candidate['sources']),
                original=index,source_rank=rank))
        self.ranked=rows
        return rows

    def candidates(self,current,limit=10):
        # Parent implementation subtracts current and returns proposal rows;
        # load() is already capped at original Top10, so emptied slots cannot
        # refill from ranks 11+. No current/answer mutation occurs.
        if limit not in (5,10,20):raise ToolError('bad_args','Use limit 5, 10 or 20; ranked pool is capped at 10')
        return super().candidates(current,min(limit,10))


def offer_ranked(toolbox,provider,current=None,limit=10):
    from .find_missed import offer
    result=offer(toolbox,provider,current,limit)
    result.update(ranking=provider.ranking,frozen_pool_limit=10,refill_outside_top10=False,
        verifier_policy='V0/four fresh per session; no offline or cross-seed answer reuse')
    return result
