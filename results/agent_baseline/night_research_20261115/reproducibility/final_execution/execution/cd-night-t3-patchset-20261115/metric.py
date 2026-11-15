import json
from pathlib import Path
r=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/r1_patchset')
s={}
for split in ("tune","confirm"):
 p=r/split/"summary.json"
 if p.exists():s[split]=json.loads(p.read_text()).get("tasks",0)
print(json.dumps(dict(stage="fixed_R1_patchset",cost_cny=0,completed=s)))
