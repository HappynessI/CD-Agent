#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-missed-20261115/logs/mechanism-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-n8-missed-20261115/logs/mechanism-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="${N8_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.n8_missed_mechanism
