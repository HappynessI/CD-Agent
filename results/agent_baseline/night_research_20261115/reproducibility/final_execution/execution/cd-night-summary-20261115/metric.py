import json
from pathlib import Path
s=json.loads(Path("state.json").read_text())
if Path(s.get("current") or "").name=="13-same-time.sbatch":
 p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/same_time_counterfactual/summary.json')
 d=json.loads(p.read_text()) if p.exists() else {}
 print(json.dumps(dict(stage="same_time_counterfactual",cost_cny=d.get("recorded_cost_cny",0),cumulative=True,completed=d.get("completed",0),expected=12)))
else:
 print(json.dumps(dict(stage="night_report_and_archive",cost_cny=0)))
