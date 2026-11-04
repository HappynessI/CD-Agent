"""Area-weighted random sample of pure false positives the verifier accepted, for manual classification."""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw
C='/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/'
sys.path[:0]=[C+'ovcd_v0',C]
import agent_baseline
from agent_baseline.ask_eval import load_init, RUN, crops, truth
from common import components, load_rgb, names
table,_=truth('val')
rows={json.loads(l)['uid']:json.loads(l) for l in open(RUN/'ask_eval/val/q37_ms.jsonl')}
yes=[u for u,r in rows.items() if u in table and table[u]['tp']/table[u]['area']<.1 and r['answer']=='yes']
rng=random.Random(20261103); w=[table[u]['area'] for u in yes]; picked=[]
while len(picked)<36:
    u=rng.choices(yes,w)[0]
    if u not in picked: picked.append(u)
by={n.split('.')[0]:n for n in names('val')}
print('yes pure FP',len(yes),'area',sum(w),'sampled area',sum(table[u]['area'] for u in picked))
for s in range(2):
    sheet=Image.new('RGB',(4*260,6*280),'white'); d=ImageDraw.Draw(sheet)
    for k,u in enumerate(picked[s*12:(s+1)*12] if False else picked[s*18:(s+1)*18][:18]):
        pass
    part=picked[s*18:(s+1)*18]
    sheet=Image.new('RGB',(6*200,6*(200+18)),'white'); d=ImageDraw.Draw(sheet)
    for k,u in enumerate(part):
        stem,i=u.split(':'); rgb=load_rgb('val',by[stem]); labels,comps=components(load_init('val',stem)); c=next(c for c in comps if c['id']==int(i))
        ims=crops(rgb,labels,dict(id=c['id'],bbox=[int(v) for v in c['bbox']]),False)
        x=(k%3)*400; y=(k//3)*218
        for j,im in enumerate(ims): sheet.paste(im.resize((198,198)),(x+j*200,y+18))
        d.text((x+2,y+3),f"#{s*18+k} {u} a{table[u]['area']} {rows[u]['type']}",fill='black')
    sheet.save(RUN/f'probe/ask_fp_sample{s}.png')
json.dump(picked,open(RUN/'probe/ask_fp_sample.json','w'))
