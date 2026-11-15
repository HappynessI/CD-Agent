#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/offline-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n7-t3-rejection-20261115/logs/offline-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/code
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m unittest agent_baseline.tests.test_n7_t3_rejection
bash /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-t3/tools/run_n7_t3_rejection.sh tune
