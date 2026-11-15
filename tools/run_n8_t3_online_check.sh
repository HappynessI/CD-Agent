#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=$ROOT/wyh/cd-agent-worktrees/n7-t3
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 OVCD_CODE=$REPO/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B -m unittest agent_baseline.tests.test_n8_t3_online_provider
"$PY" -B -m agent_baseline.analysis.n8_t3_online_check --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --night "$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114" --semantic "$ROOT/wyh/exp6_staging/batch98_t3_routes/results/route_a/semantic" --tasks "$ROOT/wyh/exp6_staging/batch98_t3_routes/results/agent10_tasks.jsonl" --out "$OUT"
