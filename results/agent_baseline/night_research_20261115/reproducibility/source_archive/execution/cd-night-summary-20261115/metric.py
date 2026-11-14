import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/report_final/report.json')
print(json.dumps(dict(stage="night_final_reports",cost_cny=0,t3_report=p.exists())))
