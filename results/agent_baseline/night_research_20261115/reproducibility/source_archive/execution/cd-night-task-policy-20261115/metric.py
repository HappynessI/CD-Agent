import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114')
print(json.dumps(dict(stage="task_policy",cost_cny=0,frozen=(p/"selected_verifier_by_task.json").exists(),scored=(p/"score_verifier_by_task.json").exists())))
