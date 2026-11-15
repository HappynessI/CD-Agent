import json,os
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_missed_trigger_20261115/reproducibility/inventory.json')
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(cost_cny=0,archive_complete=d.get("complete",False))))
