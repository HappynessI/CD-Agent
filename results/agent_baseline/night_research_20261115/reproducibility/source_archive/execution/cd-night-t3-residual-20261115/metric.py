import json
from pathlib import Path
r=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_residual')
cost=sum(json.loads(p.read_text()).get("cost_cny",0) for pattern in ("**/answer.json","**/failure_*.json") for p in r.glob(pattern))
p=r/"tune/summary.json"
s=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="paired_C_minus_E",cost_cny=cost,cumulative=True,tasks=s.get("tasks",0),new_requests=s.get("new_api_requests",0),reused=s.get("reused_answers",0),cost_scope="Only residual new requests; mainT3 cost excludes this independent root")))
