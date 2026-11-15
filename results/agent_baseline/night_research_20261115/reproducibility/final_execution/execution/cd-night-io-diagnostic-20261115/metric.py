import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/io_diagnostic/progress.json')
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(stage="fixed_four_file_IO",cost_cny=0,completed=d.get("completed",0))))
