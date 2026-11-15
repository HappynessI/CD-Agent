#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-missed-trigger-20261115/logs/archive-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-missed-trigger-20261115/logs/archive-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code PYTHONDONTWRITEBYTECODE=1
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.night_research_archive --plan /guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-missed-trigger-20261115/archive_plan.json --output /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/n7_missed_trigger_20261115/reproducibility
