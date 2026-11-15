#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-validation-20261114/logs/contacts-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-validation-20261114/logs/contacts-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m agent_baseline.analysis.night_grounding_review_contact /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/grounding_blind_v1
"$PY" -c 'import json;from pathlib import Path;Path("/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_validation_20261114/metric.json").write_text(json.dumps(dict(stage="anonymous_review_contacts",completed=12,cost_cny=0)))'
QQ_SUB="Night anonymous review contacts" QQ_BODY="status=success; 12 contacts; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_t3_20261114/grounding_blind_v1" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true
