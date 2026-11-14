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
OUT="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export OVCD_CODE="$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
COMMON=(--pilot "$ROOT/CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot" --work "$ROOT/exp6_staging/batch99_find_missed_pre/artifacts" --run "$ROOT/exp6_staging/agent_baseline_run" --out "$OUT")
MODE=${1:-smoke}
shift || true
run() { "$PY" -B -m agent_baseline.analysis.night_missed_views "$@" "${COMMON[@]}"; }
ask() { : "${BAILIAN_CSV:?Existing workspace API credential export}"; "$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" "$PY" -B -m agent_baseline.analysis.night_missed_views verify "${COMMON[@]}" "$@"; }
if [[ "$MODE" == smoke ]]; then
  "$PY" -B -m unittest agent_baseline.tests.test_night_missed_views
  run prepare
  for variant in V1 V2 V3 V4; do ask --variant "$variant" --split tune --smoke; done
  run score --split tune --smoke
elif [[ "$MODE" == verify ]]; then ask "$@"
else run "$MODE" "$@"
fi
