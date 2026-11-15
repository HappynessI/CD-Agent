#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-t3-20261115/logs/archive-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-t3-20261115/logs/archive-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code PYTHONDONTWRITEBYTECODE=1
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.n8_t3_audit --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.n8_t3_refresh_html --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
/usr/bin/python3 - <<'SNAPSHOT'
from pathlib import Path
import shutil
repo=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent');snap=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/n8-t3-source')
for pattern in ['code/agent_baseline/analysis/n8_t3*.py', 'code/agent_baseline/tests/test_n8_t3*.py', 'tools/run_n8_t3*.sh', 'code/agent_baseline/analysis/run_n8_t3*.sbatch']:
 for source in repo.glob(pattern):
  target=snap/source.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
SNAPSHOT
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.night_research_archive --plan /guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-t3-20261115/archive_plan.json --output /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115/reproducibility
