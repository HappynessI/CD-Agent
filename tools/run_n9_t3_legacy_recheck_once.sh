#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu25
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
: "${SLURM_JOB_ID:?Parent submission only}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${N9_CODE:-$ROOT/wyh/cd-agent-worktrees/n7-t3}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n9_t3_20261116
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish(){ status=$?; trap - EXIT; QQ_SUB="N9 legacy one-time recheck $SLURM_JOB_ID" QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$OUT/legacy_recheck_once; no clean-data changes" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
timeout --signal=TERM --kill-after=10s 220s "$ROOT/conda_envs/exp5-env/bin/python" -B -m agent_baseline.analysis.n9_t3_legacy_recheck_once --data "$OUT"
