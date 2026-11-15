#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N8_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
TASKS=$ROOT/wyh/exp6_staging/batch98_t3_routes/results/agent10_tasks.jsonl
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B -m unittest agent_baseline.tests.test_n8_t3_online_report
"$PY" -B -m agent_baseline.analysis.n8_t3_online_metrics --output "$OUT" --tasks "$TASKS"
"$PY" -B -m agent_baseline.analysis.n8_t3_online_report --output "$OUT" --tasks "$TASKS"
