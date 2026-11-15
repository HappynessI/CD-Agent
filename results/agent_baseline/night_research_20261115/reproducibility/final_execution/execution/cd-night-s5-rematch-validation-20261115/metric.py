import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114')
print(json.dumps(dict(stage="S5_equivalence",cost_cny=0,results={name:json.loads((p/name).read_text()) for name in ["rematch_equivalence_train106.json","rematch_packed_equivalence_train106.json"] if (p/name).exists()})))
