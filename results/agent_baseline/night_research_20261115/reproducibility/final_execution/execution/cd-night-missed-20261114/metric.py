import sys,json
from pathlib import Path
sys.path.insert(0,'/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/rank-v1/code')
from agent_baseline.analysis.night_missed_report import usage
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114')
roots=[p]
u=[usage(x) for x in roots]
print(json.dumps(dict(stage='views',cost_cny=sum(x["recorded_cost_cny"] for x in u),cumulative=True,requests=sum(x["successful_requests"] for x in u),failed_attempts=sum(x["failed_attempts"] for x in u),missing_cost_records=sum(x["missing_cost_records"] for x in u),completed_scores=[str(x/"score_summary.json") for x in roots if (x/"score_summary.json").exists()])))
