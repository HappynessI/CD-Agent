import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/task_reconstruct_audit/t44/pool_summary.json')
r=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage='t44',selected=r.get('selected'),cost_cny=0,cumulative=True)))
