#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu25
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_JOB_ID:?Parent Slurm submission}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for script in run_n9_t3_online.sh run_n9_t3_agent_worker.sh run_n9_t3_data_resume.sh run_n9_t3_fixed_confirm.sh; do bash -n "$REPO/tools/$script"; done
"$ROOT/conda_envs/exp5-env/bin/python" -B -m unittest agent_baseline.tests.test_n9_t3_data agent_baseline.tests.test_n9_t3_retrieval agent_baseline.tests.test_n9_t3_online
