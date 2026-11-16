#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu25
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
: "${SLURM_JOB_ID:?Parent submits frozen script}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${T5_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
BASE=$ROOT/wyh/CD-Agent/results/agent_baseline/task_reconstruct_audit/t5
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="T5 visual selection $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$BASE/visual_selection; 5 candidate + 1 diagnostic, pending_review" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$ROOT/conda_envs/exp5-env/bin/python" -B -m agent_baseline.analysis.task_reconstruct_t5_visual --source "$BASE/revision_actual_error" --output "$BASE/visual_selection"
