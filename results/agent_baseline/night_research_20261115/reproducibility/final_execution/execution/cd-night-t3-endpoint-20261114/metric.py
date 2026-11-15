import json
from pathlib import Path
r=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114')
s=json.loads((r/"endpoint/summary.json").read_text())
p=r/"report_stage1/report.json"
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="endpoint_and_stage1_report",cost_cny=0,tasks=s["tasks"],report_exists=p.exists(),delta_reproduction={k:v for k,v in d.get("delta_reproduction",{}).items() if k!="records"})))
