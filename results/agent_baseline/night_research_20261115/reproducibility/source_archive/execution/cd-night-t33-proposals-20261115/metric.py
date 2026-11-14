import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/t33/retrieval/B0/prepare_progress.json')
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="T33_GPU_proposals_only",cost_cny=0,progress=d)))
