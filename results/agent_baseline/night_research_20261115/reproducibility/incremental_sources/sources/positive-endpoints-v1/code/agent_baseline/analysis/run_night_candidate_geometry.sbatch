#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
BASE="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -B -m unittest agent_baseline.tests.test_night_candidate_geometry
"$PY" -B -m agent_baseline.analysis.night_candidate_geometry \
 --views "$BASE" --pilot "$ROOT/CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot" \
 --work "$ROOT/exp6_staging/batch99_find_missed_pre/artifacts" --run "$ROOT/exp6_staging/agent_baseline_run" \
 --out "$BASE/candidate_geometry_tune"
