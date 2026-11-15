#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
COMMON=(--views "$OUT" --run "$ROOT/exp6_staging/agent_baseline_run" --work "$ROOT/exp6_staging/batch99_find_missed_pre/artifacts" --out "$OUT/candidate_gates_pilot")
MODE=${1:-pilot}
shift || true
if [[ "$MODE" == pilot ]]; then
  "$PY" -B -m unittest agent_baseline.tests.test_night_missed_candidates
  "$PY" -B -m agent_baseline.analysis.night_missed_candidates extract "${COMMON[@]}"
  "$PY" -B -m agent_baseline.analysis.night_missed_candidates evaluate "${COMMON[@]}"
else
  "$PY" -B -m agent_baseline.analysis.night_missed_candidates "$MODE" "${COMMON[@]}" "$@"
fi
