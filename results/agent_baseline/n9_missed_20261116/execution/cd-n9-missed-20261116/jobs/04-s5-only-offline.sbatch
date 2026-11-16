#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/04-s5-only-offline-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n9-missed-20261116/logs/04-s5-only-offline-%j.err
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/CD-Agent
export PYTHONPATH="$PWD/code:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m unittest agent_baseline.tests.test_n9_missed_s5_only
"$PY" -B -m agent_baseline.analysis.n9_missed_s5_only initialize
"$PY" -B -m agent_baseline.analysis.n9_missed_s5_only prepare --split tune
"$PY" -B -m agent_baseline.analysis.n9_missed_s5_only reuse --split tune
"$PY" -B - <<'PY'
from pathlib import Path
import json
out=Path('results/agent_baseline/n9_missed_20261116/method_revision_s5_only')
rows=json.loads((out/'requests_tune.json').read_text())
missing=[dict(uid=r['uid'],variant=v) for r in rows for v in r['variants'] if not (out/'answers'/v/(r['uid']+'.json')).exists()]
slots=sum(len(r['variants']) for r in rows)
report=dict(status='awaiting_new_api_authorization' if missing else 'cached_answers_complete',requested_slots=slots,reused_slots=slots-len(missing),missing_new_verifications=missing,new_api_calls=0,online_agent_runs=0,reason='Automatic approval review rejected new external API requests under latest round restrictions; offline preparation and exact paid-answer reuse only. No incomplete F1 presented.')
(out/'offline_reuse_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='missing_new_verifications'}))
PY
