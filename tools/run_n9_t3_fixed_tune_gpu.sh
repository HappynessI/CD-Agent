#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu46
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Main-agent Slurm submission only}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
DATA=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 OVCD_CODE=$REPO/code/ovcd_v0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="N9 T3 tune local feature fill $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$DATA/exploratory_fixed_tune; no API; no formal confirm/online" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_n9_t3_retrieval
"$PY" -B -m agent_baseline.analysis.n9_t3_fixed --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --data "$DATA" --backend fill-missing --vision-config "$REPO/code/agent_baseline/vision_t13.example.json"
