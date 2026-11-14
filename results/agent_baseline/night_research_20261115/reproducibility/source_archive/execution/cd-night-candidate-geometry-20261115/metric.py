import json
from pathlib import Path
r=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114')
s={}
for name in ("candidate_geometry_tune/progress.json","geometry_route/tune/summary.json","geometry_route/confirm/summary.json"):
 p=r/name
 if p.exists():
  d=json.loads(p.read_text());s[name]={k:d[k] for k in ("completed","expected","groups") if k in d}
print(json.dumps(dict(stage="geometry_and_routing",cost_cny=0,results=s)))
