#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
set -euo pipefail
WORK=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/n7-missed
export PYTHONPATH="$WORK/code:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -m agent_baseline.analysis.task_reconstruct_t44_interior
