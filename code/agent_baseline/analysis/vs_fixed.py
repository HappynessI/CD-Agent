"""Agent against the fixed route on the same masks: paired bootstrap of images and runs. Labels only score."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
C='/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/'
sys.path[:0]=[C+'ovcd_v0',C]
import agent_baseline
from common import names, load_gt
R=Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
stems=[Path(n).stem for n in names('val')]
def fixed():
    out=[]
    for n,s in zip(names('val'),stems):
        gt=load_gt('val',n); m=np.asarray(Image.open(R/f'init/sam3_imc_p85/val/{s}.png'))>0
        out.append([(m&gt).sum(),(m&~gt).sum(),(~m&gt).sum()])
    return np.array(out,float)
def run(cfg,t):
    rows=[]
    for n in (1,2,3):
        a=[]
        for s in stems:
            e=json.load(open(R/f'{cfg}_s{n}/{t}/val/{s}/final.json'))['eval']; a.append([e['tp'],e['fp'],e['fn']])
        rows.append(a)
    return np.array(rows,float)
f1=lambda c:200*c[...,0].sum(-1)/(2*c[...,0].sum(-1)+c[...,1].sum(-1)+c[...,2].sum(-1))
F=fixed(); print('fixed p85',round(float(f1(F)),2))
rng=np.random.default_rng(0)
cfgs=[('F0','json_qwen37flash_f0'),('F1','json_qwen37flash_f1'),('F2','json_qwen37flash_f2'),('plus r6','json_qwen37plus_r6'),('plus r4','json_qwen37plus_r4')]
res={}
for t in ('T1.1xC0','T5.1'):
    for name,c in cfgs:
        X=run(c,t); res[(name,t)]=X
        ds=[]
        for _ in range(4000):
            i=rng.integers(0,64,64); k=rng.integers(0,3,3)
            ds.append(np.mean([f1(X[j][i]) for j in k])-f1(F[i]))
        lo,hi=np.percentile(ds,[2.5,97.5]); per=[round(float(f1(X[j])),2) for j in range(3)]
        print(t,name,per,'mean',round(float(np.mean(per)),2),'sd',round(float(np.std(per,ddof=1)),2),'vs fixed',round(float(np.mean(per)-f1(F)),2),'CI',round(lo,2),round(hi,2),'P>0',round(float(np.mean(np.array(ds)>0)),3))
    for a,b in (('F1','F0'),('F2','F1'),('F2','F0'),('F0','plus r6'),('F1','plus r6'),('F2','plus r6')):
        A,B=res[(a,t)],res[(b,t)]; ds=[]
        for _ in range(4000):
            i=rng.integers(0,64,64); k=rng.integers(0,3,3); l=rng.integers(0,3,3)
            ds.append(np.mean([f1(A[j][i]) for j in k])-np.mean([f1(B[j][i]) for j in l]))
        lo,hi=np.percentile(ds,[2.5,97.5]); print(' ',t,a,'-',b,round(float(np.mean(f1(A))-np.mean(f1(B))),2),'CI',round(lo,2),round(hi,2),'P>0',round(float(np.mean(np.array(ds)>0)),3))
