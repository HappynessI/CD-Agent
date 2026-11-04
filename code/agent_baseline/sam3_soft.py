"""Save SAM3 "building" instances with their soft pixel values, for tile sizes and overlaps not in the cache.
Run in omniovcd-env. Output per image: for each time, instance scores, tile origins and sparse values >= 0.2 (uint8)."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'ovcd_v0'),'/guisongxia01/pangchao/wangyihan/Chang-Agent/probe']
from common import names, load_rgb
import sam3_tools
p=argparse.ArgumentParser(); p.add_argument('--tile',type=int); p.add_argument('--step',type=int); p.add_argument('--split',default='val'); p.add_argument('--names-file')
p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=1); a=p.parse_args()
OUT=Path(os.environ.get('AGENT_RUN','/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))/f'sam3_soft/t{a.tile}s{a.step}/{a.split}'; OUT.mkdir(parents=True,exist_ok=True)
proc=sam3_tools.build_processor(); proc.confidence_threshold=.3
todo=Path(a.names_file).read_text().split() if a.names_file else names(a.split)
for name in todo[a.shard::a.shards]:
    out=OUT/(Path(name).stem+'.npz')
    if out.exists(): continue
    t0=time.time(); arrays={}
    for t,im in enumerate(load_rgb(a.split,name)):
        H,W=im.shape[:2]; scores,origins,offs,idxs,vals=[],[],[0],[],[]
        for y in range(0,H-a.tile+1,a.step):
            for x in range(0,W-a.tile+1,a.step):
                with torch.inference_mode(), sam3_tools.autocast():
                    s=proc.set_image(Image.fromarray(im[y:y+a.tile,x:x+a.tile]))
                    s=proc.set_text_prompt(prompt='building',state=s)
                    if not len(s['scores']): continue
                    soft=(s['masks_logits'][:,0].float()*255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
                    sc=s['scores'].float().cpu().numpy()
                for m,v in zip(soft,sc):
                    yy,xx=np.nonzero(m>=51)
                    scores.append(v); origins.append((x,y)); idxs.append((yy+y)*W+xx+x); vals.append(m[yy,xx]); offs.append(offs[-1]+len(yy))
        arrays[f'scores{t}']=np.array(scores,np.float32); arrays[f'origins{t}']=np.array(origins,np.int32).reshape(-1,2); arrays[f'offs{t}']=np.array(offs,np.int64)
        arrays[f'idx{t}']=np.concatenate(idxs).astype(np.int32) if idxs else np.zeros(0,np.int32); arrays[f'val{t}']=np.concatenate(vals) if vals else np.zeros(0,np.uint8)
    arrays['shape']=np.array([H,W]); arrays['tile']=np.array([a.tile,a.step])
    tmp=out.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**arrays); tmp.replace(out)
    print(name,len(arrays['scores0']),len(arrays['scores1']),f'{time.time()-t0:.0f}s',flush=True)
