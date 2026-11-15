#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
OUT="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export NIGHT_MISSED_CODE="$CODE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# The gate and verifier decisions must already exist; this stage never selects
# from confirmation or invokes an external model.
for file in selected_verifier.json score_tune_frozen.json candidate_gates_pilot/manifest.json candidate_gates_pilot/score_summary.json; do
  test -s "$OUT/$file"
done
NIGHT_VERIFIER=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_verifier"])' "$OUT/selected_verifier.json")
export NIGHT_VERIFIER
"$PY" -B -m unittest agent_baseline.tests.test_night_missed_report agent_baseline.tests.test_night_missed_candidates
bash "$CODE/agent_baseline/analysis/run_night_missed_rank.sbatch" prepare
"$PY" -B -m agent_baseline.analysis.night_missed_report --out "$OUT"
