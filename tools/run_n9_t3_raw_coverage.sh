#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu25
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Parent exp-loop submission only}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="N9 tune raw SAM coverage $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$OUT/raw_sam_coverage_diagnostic" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$ROOT/conda_envs/exp5-env/bin/python" -B -m agent_baseline.analysis.n9_t3_raw_coverage --data "$OUT" --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --output "$OUT/raw_sam_coverage_diagnostic"
