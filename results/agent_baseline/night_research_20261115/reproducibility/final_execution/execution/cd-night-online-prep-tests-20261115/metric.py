import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/online_preparation_tests/result.json')
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="online_provider_unit_preparation_only",cost_cny=0,result=d)))
