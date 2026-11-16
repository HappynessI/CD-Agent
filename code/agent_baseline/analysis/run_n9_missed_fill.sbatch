#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="${N9_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m agent_baseline.analysis.n9_missed_prepare fill
