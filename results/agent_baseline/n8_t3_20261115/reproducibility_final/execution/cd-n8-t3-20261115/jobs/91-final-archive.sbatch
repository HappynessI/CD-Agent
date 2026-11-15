#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-t3-20261115/logs/archive-final-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-t3-20261115/logs/archive-final-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 - <<'CHECK'
import json,re
from pathlib import Path
repo=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent');root=repo/'results/agent_baseline/n8_t3_20261115'
report=json.loads((root/'online_report.json').read_text());summary=json.loads((root/'online_summary.json').read_text())
assert summary['complete'] and len(report['records'])==60
assert all(r['state'] in ('successful_submit','fallback') for r in report['records'])
missing=[];links=0
p=repo/'results/agent_baseline/n8_summary_report.md'
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
 if '://' in target or target.startswith('#'):continue
 links+=1
 if not (p.parent/target.split('#')[0]).exists():missing.append(target)
assert not missing,missing
cost=report['cost']['total_cny']+json.loads((repo/'results/agent_baseline/n8_missed_20261115/metric.json').read_text())['cost_cny'] if (repo/'results/agent_baseline/n8_missed_20261115/metric.json').exists() else report['cost']['total_cny']+0.0990168
assert cost<=100
result=dict(sessions=60,complete=True,summary_local_links=links,missing_links=missing,total_cost_cny=cost,known_online_requests=sum(x['known_requests'] for x in summary['cost_audit']))
(root/'final_delivery_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
CHECK
/usr/bin/python3 - <<'SNAPSHOT'
from pathlib import Path
import shutil
repo=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent');snap=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/n8-t3-source')
for pattern in ['code/agent_baseline/analysis/n8_t3*.py', 'code/agent_baseline/tests/test_n8_t3*.py', 'tools/run_n8_t3*.sh', 'code/agent_baseline/analysis/run_n8_t3*.sbatch']:
 for source in repo.glob(pattern):
  target=snap/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
SNAPSHOT
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.night_research_archive --plan /guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-t3-20261115/archive_final_plan.json --output /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115/reproducibility_final
