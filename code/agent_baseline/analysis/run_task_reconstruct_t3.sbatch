#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Main agent submits through exp-loop}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
export PYTHONPATH="$ROOT/cd-agent-worktrees/n7-missed/code:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
"$PY" -B -m unittest agent_baseline.tests.test_task_reconstruct_t3
"$PY" -B -m agent_baseline.analysis.task_reconstruct_t3
