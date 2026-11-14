import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/oracle_cache/summary.json')
s=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="cached_GT_text_oracle_diagnostic",cost_cny=0,tasks=s.get("tasks",0),model_calls=0,summary=s.get("summary",[]))))
