import json
from pathlib import Path
o=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114')
stages={}
for split in ("tune","confirm"):
 p=o/("positive_endpoints_"+split);q=p/"summary.json";i=p/"inventory.json"
 d=json.loads(q.read_text()) if q.exists() else {};v=json.loads(i.read_text()) if i.exists() else {}
 stages[split]=dict(status=d.get("status"),missing_queries=v.get("unique_missing"))
print(json.dumps(dict(stage="predicted_endpoint_mechanism",cost_cny=0,stages=stages)))
