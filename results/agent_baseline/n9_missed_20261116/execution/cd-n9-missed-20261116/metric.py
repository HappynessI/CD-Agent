import json
from pathlib import Path
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n9_missed_20261116/legacy_all_source_dino020_abstain/metric.json')
print(json.dumps(json.loads(p.read_text())))
