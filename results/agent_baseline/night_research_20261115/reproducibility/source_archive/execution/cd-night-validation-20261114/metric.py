import json
from pathlib import Path
print(json.dumps(json.loads(Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_validation_20261114/metric.json').read_text())))
