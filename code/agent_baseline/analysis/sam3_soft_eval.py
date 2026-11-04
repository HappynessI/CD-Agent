"""Fixed route on the soft SAM3 output: tile, overlap, score threshold and pixel threshold. Labels only score."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
C='/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/'
sys.path[:0]=[C+'ovcd_v0',C]
import agent_baseline
from common import names, load_gt, confusion, scores
from agent_baseline import instances as I
R=Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
TAUS=(.5,.6,.7,.75,.8,.85,.9,.95); SCORES=(.3,.4,.5,.6)
def rows(z,t,tile,step,overlap,s,tau,keep_grid):
    sc=z[f'scores{t}']; org=z[f'origins{t}']; off=z[f'offs{t}']; idx=z[f'idx{t}']; val=z[f'val{t}']
    H,W=z['shape']; out=[]
    for i in np.flatnonzero(sc>=s):
        x,y=org[i]
        if keep_grid and (x%keep_grid or y%keep_grid): continue
        a=idx[off[i]:off[i+1]][val[off[i]:off[i+1]]>=round(tau*255)]
        if not len(a): continue
        ys,xs=a//W,a%W
        if overlap:
            # Each point of the image belongs to the central cell of one tile; the instance centre decides.
            m=(tile-step)//2; cx,cy=xs.mean(),ys.mean()
            if not ((x==0 or cx>=x+m) and (x+tile==W or cx<x+tile-m) and (y==0 or cy>=y+m) and (y+tile==H or cy<y+tile-m)): continue
        out.append(dict(id=len(out)+1,idx=np.sort(a),area=len(a),score=float(sc[i]),bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]))
    return out
CFG=[('t512',512,256,512,False),('t512ov',512,256,0,True)]
def work(name):
    gt=load_gt('val',name); res={}
    for label,tile,step,grid,ov in CFG:
        z=dict(np.load(R/f'sam3_soft/t{tile}s{step}/val/{Path(name).stem}.npz'))
        for s in SCORES:
            for tau in TAUS:
                a=rows(z,0,tile,step,ov,s,tau,grid); b=rows(z,1,tile,step,ov,s,tau,grid)
                ap,dis,_=I.match(a,b,.25,'overlap')
                res[f'{label}|{s}|{tau}']=confusion(I.to_mask(ap+dis,gt.shape),gt)
    return res
if __name__=='__main__':
    with Pool(int(sys.argv[1])) as p: all_=p.map(work,names('val'))
    out={}
    for k in all_[0]:
        tot={c:sum(r[k][c] for r in all_) for c in all_[0][k]}
        out[k]=scores(tot) if callable(scores) else tot
    (R/'probe/sam3_soft_eval2.json').write_text(json.dumps(out,indent=0,default=float))
    best=sorted(out.items(),key=lambda kv:-kv[1]['f1'])
    for k,v in best[:15]: print(k,{m:round(v[m],4) for m in v if m in('f1','precision','recall','p','r')})
    for k in ('t512|0.5|0.5','t512ov|0.5|0.5'): print('ref',k,out[k])
