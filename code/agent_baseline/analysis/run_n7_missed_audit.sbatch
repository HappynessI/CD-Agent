#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run through sbatch}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${N7_MISSED_CODE:-$ROOT/cd-agent-worktrees/n7-missed/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python
export PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$ROOT/CD-Agent/results/agent_baseline/n7_missed_trigger_20261115"
"$PY" -B -m agent_baseline.analysis.n7_missed_cluster --out "$OUT"
"$PY" -B -m agent_baseline.analysis.n7_missed_state_audit --out "$OUT" --base "$ROOT/exp6_staging/batch99_find_missed_pre/artifacts/base"
