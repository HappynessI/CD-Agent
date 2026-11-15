"""Paired image bootstrap of the delete-rejected rule between verifiers on one pool. Labels only score."""
import json, sys
from pathlib import Path
import numpy as np
C='/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/'
sys.path[:0]=[C+'ovcd_v0',C]
import agent_baseline
from agent_baseline import ask_eval as A
from common import names
A.INIT=sys.argv[1]; base=sys.argv[2]; others=sys.argv[3:]
table,tot=A.truth('val')
stems=[Path(n).stem for n in names('val')]
def per_image(v):
    rows={json.loads(l)['uid']:json.loads(l) for l in open(A.folder('val')/f'{v}.jsonl')}
    gone={s:np.zeros(2) for s in stems}
    miss=0
    for u,t in table.items():
        if u not in rows: miss+=1
        if rows.get(u,{}).get('answer')=='no': gone[u.split(':')[0]]+=[t['tp'],t['fp']]
    return np.array([gone[s] for s in stems]),miss
kept={s:np.zeros(3) for s in stems}
# per-image base counts
from common import load_gt
for n in names('val'):
    s=Path(n).stem; gt=load_gt('val',n); m=A.load_init('val',s); kept[s]=np.array([(m&gt).sum(),(m&~gt).sum(),(~m&gt).sum()],float)
B=np.array([kept[s] for s in stems])
def rule(g,idx=None):
    b=B if idx is None else B[idx]; g=g if idx is None else g[idx]
    tp,fp,fn=b[:,0].sum()-g[:,0].sum(),b[:,1].sum()-g[:,1].sum(),b[:,2].sum()+g[:,0].sum()
    return 200*tp/(2*tp+fp+fn)
g0,m0=per_image(base); print(base,'rule',round(rule(g0),2),'missing',m0,'removes TP FP',g0.sum(0).astype(int).tolist())
rng=np.random.default_rng(0)
for v in others:
    g,m=per_image(v); ds=[]
    for _ in range(4000):
        i=rng.integers(0,len(stems),len(stems)); ds.append(rule(g,i)-rule(g0,i))
    lo,hi=np.percentile(ds,[2.5,97.5]); print(v,'rule',round(rule(g),2),'missing',m,'removes TP FP',g.sum(0).astype(int).tolist(),'diff',round(rule(g)-rule(g0),2),'CI',round(lo,2),round(hi,2),'P>0',round(float(np.mean(np.array(ds)>0)),3))
