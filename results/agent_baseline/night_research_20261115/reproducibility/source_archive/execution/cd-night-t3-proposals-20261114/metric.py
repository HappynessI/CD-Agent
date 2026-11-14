import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/retrieval/B0/prepare_progress_all.json')
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="GPU_proposals_only",cost_cny=0,progress=d)))
