"""Tool layer, val only: can the boundaries of correct SAM3 instances be improved without labels?
Every method below uses images and instances only; labels score the result.
  sam_box   : SAM ViT-H box prompt per instance at its own time, replace it (optionally only when quality is high)
  sam_and   : SAM3 instance AND SAM box mask
  slic      : superpixels on that time's image, keep a superpixel when most of it is inside the instance
  shift     : per-image best global shift of the mask (uses labels: a ceiling, shows misregistration)
Reports both on the whole mask and after the oracle deletion (which shows what is left once decisions are right).
"""
import json, sys
from multiprocessing import Pool
import numpy as np
from skimage.segmentation import slic
C='/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/'
sys.path[:0]=[C+'ovcd_v0',C]
import agent_baseline
from common import names, load_gt, load_rgb, confusion, scores
import sam3_t7, sam_tools
from agent_baseline import instances as I
P=None
def init():
    global P
    import torch; torch.set_num_threads(1)
    P=sam_tools.Prompter()
def rows_of(name):
    a,b=[I.from_dense(m[s>=.5],s[s>=.5]) for m,s in sam3_t7.read('val',name,512,conf=.1)]
    ap,dis,_=I.match(a,b,.25,'overlap')
    return [(r,1) for r in ap]+[(r,0) for r in dis]
def dense(r,shape):
    m=np.zeros(shape[0]*shape[1],bool); m[r['idx']]=True; return m.reshape(shape)
def work(name):
    gt=load_gt('val',name); rgb=load_rgb('val',name); rows=rows_of(name); shape=gt.shape; g=gt.ravel()
    sp=[slic(im,n_segments=2500,compactness=10,start_label=1,channel_axis=-1) for im in rgb]
    out={k:np.zeros(shape,bool) for k in('base','sam_box','sam_box_q90','sam_and','slic','slic_or','base_del','sam_box_del','sam_and_del','slic_del')}
    for r,t in rows:
        m=dense(r,shape); x0,y0,x1,y1=r['bbox']
        pad=2; box=[max(0,x0-pad),max(0,y0-pad),min(shape[1],x1+pad),min(shape[0],y1+pad)]
        s,q=P.box('val',name,t,box)
        roi=np.zeros(shape,bool); roi[box[1]:box[3],box[0]:box[2]]=True; s=s&roi
        # superpixel snap
        lab=sp[t]; ids=np.unique(lab[box[1]:box[3],box[0]:box[2]])
        sn=np.zeros(shape,bool)
        for i in ids:
            mm=lab==i
            if (mm&m).sum()/mm.sum()>=.5: sn|=mm
        true=g[r['idx']].mean()>=.5
        out['base']|=m; out['sam_box']|=s; out['sam_box_q90']|=(s if q>=.9 else m); out['sam_and']|=(s&m); out['slic']|=sn; out['slic_or']|=(sn|m)
        if true:
            out['base_del']|=m; out['sam_box_del']|=s; out['sam_and_del']|=(s&m); out['slic_del']|=sn
    res={k:confusion(v,gt) for k,v in out.items()}
    # shift ceiling
    best=None
    for dy in range(-4,5):
        for dx in range(-4,5):
            c=confusion(np.roll(np.roll(out['base'],dy,0),dx,1),gt)
            f=2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1)
            if best is None or f>best[0]: best=(f,dx,dy,c)
    res['shift_oracle']=best[3]; res['_shift']=(best[1],best[2],bool(gt.any()))
    return res
if __name__=='__main__':
    with Pool(int(sys.argv[1]),initializer=init) as pool: rs=pool.map(work,names('val'),chunksize=1)
    for k in [k for k in rs[0] if not k.startswith('_')]:
        t={x:sum(r[k][x] for r in rs) for x in('tp','fp','fn')}; s=scores(t); print(f"{k:14s} F1 {100*s['f1']:.2f} P {100*s['precision']:.2f} R {100*s['recall']:.2f}",flush=True)
    sh=[r['_shift'] for r in rs if r['_shift'][2]]
    print('best shift median dx,dy',np.median([s[0] for s in sh]),np.median([s[1] for s in sh]),'mean |dx|,|dy|',np.mean([abs(s[0]) for s in sh]).round(2),np.mean([abs(s[1]) for s in sh]).round(2))
