#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${NIGHT_VERIFIER:?Freeze V0/V1/V2/V3/V4 from tuning evidence first}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
BASE="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
COMMON=(--views "$BASE" --gates "$BASE/candidate_gates_pilot" --pilot "$ROOT/CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot" --run "$ROOT/exp6_staging/agent_baseline_run" --work "$ROOT/exp6_staging/batch99_find_missed_pre/artifacts" --out "$BASE/rank_$NIGHT_VERIFIER" --variant "$NIGHT_VERIFIER")
MODE=${1:-prepare}
shift || true
if [[ "$MODE" == prepare ]]; then
  "$PY" -B -m unittest agent_baseline.tests.test_night_missed_rank agent_baseline.tests.test_night_missed_rank_inventory
fi
if [[ "$MODE" == verify ]]; then
  : "${BAILIAN_CSV:?Existing workspace credential export}"
  "$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" "$PY" -B -m agent_baseline.analysis.night_missed_rank "$MODE" "${COMMON[@]}" "$@"
else
  "$PY" -B -m agent_baseline.analysis.night_missed_rank "$MODE" "${COMMON[@]}" "$@"
fi
if [[ "$MODE" == prepare || "$MODE" == verify ]]; then
  "$PY" -B -m agent_baseline.analysis.night_missed_rank_inventory --out "$BASE/rank_$NIGHT_VERIFIER"
fi
