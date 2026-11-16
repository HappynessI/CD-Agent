#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu46
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Parent exp-loop submission only}"
: "${N9_APPROVAL_FILE:?Explicit human approval required}"
: "${N9_TUNE_SELECTION_T31:?Saved T3.1 tune selection required}"
: "${N9_TUNE_SELECTION_T33:?Saved T3.3 tune selection required}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="N9 T3 full confirm $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$OUT/fixed_full_confirm" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_n9_t3_full_confirm
"$PY" -B -m agent_baseline.analysis.n9_t3_full_confirm --data "$OUT" --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --approval "$N9_APPROVAL_FILE" --tune-selection "$N9_TUNE_SELECTION_T31" --tune-selection "$N9_TUNE_SELECTION_T33" --backend fill-missing --vision-config "$REPO/code/agent_baseline/vision_t13.example.json"
