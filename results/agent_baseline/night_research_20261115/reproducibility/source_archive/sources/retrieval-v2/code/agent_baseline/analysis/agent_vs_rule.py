"""Where does the Agent differ from the rule 'delete components the verifier rejects'? Pixels, per image. Labels only score."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
C='/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/'
sys.path[:0]=[C+'ovcd_v0',C]
import agent_baseline
from common import names, load_gt, components
R=Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
import os; PREFIX=os.environ.get('PREFIX','json_qwen37plus'); CFG=sys.argv[1]; TASK=sys.argv[2]; INIT=sys.argv[3]; ASK=sys.argv[4]
rows={json.loads(l)['uid']:json.loads(l) for l in open(R/ASK/(os.environ.get('RULE','q37_ms')+'.jsonl'))}
def f1(c): return 200*c[0]/max(1,2*c[0]+c[1]+c[2])
def work(a):
    n,name=a; stem=Path(name).stem; gt=load_gt('val',name)
    init=np.asarray(Image.open(R/'init'/INIT/'val'/f'{stem}.png'))>0
    lab,comps=components(init)
    no=[c['id'] for c in comps if rows.get(f"{stem}:{c['id']}",{}).get('answer')=='no']
    rule=init&~np.isin(lab,no)
    d=R/f'{PREFIX}_{CFG}_s{n}'/TASK/'val'/stem
    if not (d/'final.json').exists(): return None
    fin=json.load(open(d/'final.json')); agent=np.asarray(Image.open(d/fin['mask_path']))>0
    asked=set(); asks=0
    for l in open(d/'steps.jsonl'):
        for call in json.loads(l)['calls']:
            if call['tool']=='ask' and call.get('ok'):
                asks+=1
                for e in call['args'].get('regions',[]):
                    if e['mask']=='mask#0': asked|=set(e['ids'] or [])
    cnt=lambda m:(int((m&gt).sum()),int((m&~gt).sum()))
    return dict(stem=stem,n=n,comps=len(comps),asked=len(asked),asks=asks,by=fin['submitted_by'],
        agent=(*cnt(agent),int((~agent&gt).sum())),rule=(*cnt(rule),int((~rule&gt).sum())),
        rule_only_removed=cnt(init&~rule&agent),agent_only_removed=cnt(init&rule&~agent),added=cnt(agent&~init),same_init=bool((agent==init).all()))
if __name__=='__main__':
    todo=[(n,x) for n in (1,2,3) for x in names('val')]
    with Pool(8) as p: res=[r for r in p.map(work,todo) if r]
    json.dump(res,open(R/f'probe/agent_vs_rule_{PREFIX}_{CFG}_{TASK}.json','w'))
    print('sessions',len(res))
    sum2=lambda k:[sum(r[k][i] for r in res)/3 for i in range(len(res[0][k]))]
    a,b=sum2('agent'),sum2('rule'); print('agent tp fp fn',[round(x) for x in a],'F1',round(f1(a),2)); print('rule tp fp fn',[round(x) for x in b],'F1',round(f1(b),2))
    print('rule removed, agent kept (TP,FP)',[round(x) for x in sum2('rule_only_removed')]); print('agent removed, rule kept (TP,FP)',[round(x) for x in sum2('agent_only_removed')]); print('agent added (TP,FP)',[round(x) for x in sum2('added')])
    # by coverage of the given components
    for lo,hi,tag in ((-1,.001,'no component asked'),(.001,.5,'under half'),(.5,.999,'half or more'),(.999,9,'all asked')):
        g=[r for r in res if lo<=(r['asked']/max(1,r['comps']))<hi]
        if g:
            aa=[sum(r['agent'][i] for r in g)/3 for i in range(3)]; rr=[sum(r['rule'][i] for r in g)/3 for i in range(3)]
            print(tag,'sessions',len(g),'same_init',sum(r['same_init'] for r in g),'agent',[round(x) for x in aa],'rule',[round(x) for x in rr],'gap F1 terms',round(f1(rr)-f1(aa),2),'comps mean',round(np.mean([r['comps'] for r in g]),1))
    # per image gap on the pooled score
    tot=np.array(a); by={}
    for r in res:
        by.setdefault(r['stem'],[np.zeros(3),np.zeros(3)]); by[r['stem']][0]+=np.array(r['agent'])/3; by[r['stem']][1]+=np.array(r['rule'])/3
    gaps=sorted(((f1(tot-v[0]+v[1])-f1(tot)),s,round(f1(v[0]),1),round(f1(v[1]),1)) for s,v in by.items())
    print('largest rule advantage',[(round(g,2),s,x,y) for g,s,x,y in gaps[::-1][:10]]); print('largest agent advantage',[(round(g,2),s,x,y) for g,s,x,y in gaps[:6]])
