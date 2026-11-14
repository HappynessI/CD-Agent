import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/night_report.json')
s=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="offline_report",cost_cny=0,report_exists=p.exists(),views_status=s.get("views",{}).get("status"),rankings=list(s.get("rankings",{})))))
