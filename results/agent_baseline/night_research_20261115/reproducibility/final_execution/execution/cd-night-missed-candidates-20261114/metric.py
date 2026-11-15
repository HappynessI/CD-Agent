import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/candidate_gates_pilot')
result={"cost_cny":0,"stage":"candidate_gates","result_path":str(p)}
for name in ("score_summary.json","whu_official_summary.json"):
 f=p/name
 if f.exists():result[name]="complete"
print(json.dumps(result))
