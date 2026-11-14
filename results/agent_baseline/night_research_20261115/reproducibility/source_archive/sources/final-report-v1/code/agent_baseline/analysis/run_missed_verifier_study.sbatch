#!/usr/bin/env bash
#SBATCH --job-name=missed-verifier
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through exp-loop}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE="$ROOT/cd-missed-verifier"
RUN="$ROOT/exp6_staging/agent_baseline_run"
WORK="$ROOT/exp6_staging/batch99_find_missed_pre/artifacts"
OUT="$ROOT/CD-Agent/results/agent_baseline/missed_verifier_20261114"
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
export PYTHONPATH="$CODE/code:$RUN/code_w/ovcd_v0"
export OVCD_CODE="$RUN/code_w/ovcd_v0" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
cd "$CODE"
MODE=${1:-smoke}
BASE=(--run "$RUN" --work "$WORK")
api() {
  : "${BAILIAN_CSV:?Set existing credential export path}"
  "$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" \
    "$PY" -B -m agent_baseline.analysis.missed_verifier_study "$@"
}
if [[ "$MODE" == tests ]]; then
  "$PY" -B -m unittest agent_baseline.tests.test_missed_verifier_study agent_baseline.tests.test_find_missed
elif [[ "$MODE" == smoke ]]; then
  "$PY" -B -m unittest agent_baseline.tests.test_missed_verifier_study agent_baseline.tests.test_find_missed
  "$PY" -B -m agent_baseline.analysis.missed_verifier_study prepare "${BASE[@]}" --out "$OUT/smoke" --images 1
  for variant in four som structured; do
    api verify "${BASE[@]}" --out "$OUT/smoke" --variant "$variant"
  done
  "$PY" -B -m agent_baseline.analysis.missed_verifier_study score "${BASE[@]}" --out "$OUT/smoke"
else
  shift
  if [[ "$MODE" == verify ]]; then
    api "$MODE" "${BASE[@]}" --out "$OUT/pilot" "$@"
  else
    "$PY" -B -m agent_baseline.analysis.missed_verifier_study "$MODE" "${BASE[@]}" --out "$OUT/pilot" "$@"
  fi
fi
