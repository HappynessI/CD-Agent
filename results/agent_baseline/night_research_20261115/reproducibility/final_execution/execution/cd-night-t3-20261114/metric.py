import json
from pathlib import Path
out=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114')
cost=0.
for root in ('B1','B2','B3','t33','retrieval','r1_direct4'):
 for pattern in ('**/answer.json','**/failure_*.json'):
  for p in (out/root).glob(pattern):cost+=json.loads(p.read_text()).get('cost_cny',0.)
completed={}
for root in ('B0','B1','B2','B3','t33/B0','t33/B1','t33/B2'):
 p=out/root/'summary.json'
 if p.exists():completed[root]=json.loads(p.read_text())['tasks']
for p in (out/'retrieval').glob('*/summary_*.json'):
 completed[str(p.relative_to(out))]=json.loads(p.read_text())['tasks']
for name in ('t33/retrieval/B0','r1_direct4/tune','r1_direct4/confirm','r1_endpoint/tune','r1_endpoint/confirm'):
 p=out/name/'summary.json'
 if p.exists():completed[name]=json.loads(p.read_text())['tasks']
print(json.dumps(dict(stage='night_t3',cost_cny=cost,cumulative=True,completed=completed,result_path=str(out))))
