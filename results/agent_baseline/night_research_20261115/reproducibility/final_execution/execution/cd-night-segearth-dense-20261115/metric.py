import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/segearth_dense_confirm/summary.json')
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="cached_SegEarth_B2_confirmation",cost_cny=0,status=d.get("status"),tasks=d.get("tasks",0),reproduction=d.get("historical_reproduction"))))
