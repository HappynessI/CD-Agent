#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Main-agent Slurm submission only}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
DATA=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="N9 T3 tune exploration $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$DATA/exploratory_fixed_tune; pending-review tune only" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_n9_t3_retrieval
args=(--run "$ROOT/wyh/exp6_staging/agent_baseline_run" --data "$DATA")
"$PY" -B -m agent_baseline.analysis.n9_t3_fixed "${args[@]}" --inventory-only
"$PY" -B -m agent_baseline.analysis.n9_t3_fixed "${args[@]}" --task-type T3.1 --backend cache-only
