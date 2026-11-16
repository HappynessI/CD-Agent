#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through main-agent exp-loop}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116
REVIEW=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_data_review
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="N9 T3 data QC $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$OUT review=$REVIEW; all new tasks pending_review" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_n9_t3_data
"$PY" -B -m agent_baseline.analysis.n9_t3_data --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --night "$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114" --output "$OUT" --review "$REVIEW"
