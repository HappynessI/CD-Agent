#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=$ROOT/wyh/cd-agent-worktrees/n7-t3
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
PY=$ROOT/conda_envs/exp5-env/bin/python
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 OVCD_CODE=$REPO/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
started=$SECONDS
finish() { status=$?;trap - EXIT; QQ_SUB="N8 T3 diagnostics $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=$OUT/diagnostics10" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_n8_t3_clean agent_baseline.tests.test_n8_t3_route agent_baseline.tests.test_n8_t3_system agent_baseline.tests.test_n8_t3_cost
"$PY" -B -m agent_baseline.analysis.n8_t3_audit --out "$OUT"
"$PY" -B -m agent_baseline.analysis.n8_t3_visualize --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --night "$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114" --out "$OUT"
