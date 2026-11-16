#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu25
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
set -euo pipefail
: "${SLURM_JOB_ID:?Parent submits frozen script}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${T5_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/task_reconstruct_audit/t5/revision_actual_error
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export AGENT_RUN=$ROOT/wyh/exp6_staging/agent_baseline_run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="T5 paired data reconstruction $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$OUT; pending_review offline only" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$ROOT/conda_envs/exp5-env/bin/python" -B -m unittest agent_baseline.tests.test_task_reconstruct_t5_actual
"$ROOT/conda_envs/exp5-env/bin/python" -B -m agent_baseline.analysis.task_reconstruct_t5_actual --run "$AGENT_RUN" --output "$OUT"
