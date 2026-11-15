#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${T3_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n7_t3_rejection_20261115
PY=$ROOT/conda_envs/exp5-env/bin/python
export OVCD_CODE=$REPO/code/ovcd_v0
export PYTHONPATH=$REPO/code:$OVCD_CODE PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish() {
  status=$?; trap - EXIT
  QQ_SUB="N7 T3 rejection $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=$OUT" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true
  exit "$status"
}
trap finish EXIT
"$PY" -B -m agent_baseline.analysis.n7_t3_rejection --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --night "$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114" --out "$OUT" --split "${1:-tune}"
