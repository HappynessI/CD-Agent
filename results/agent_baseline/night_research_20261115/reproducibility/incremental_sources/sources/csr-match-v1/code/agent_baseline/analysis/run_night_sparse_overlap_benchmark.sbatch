#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STEM=${1:-train_106}
"$PY" -B -m unittest agent_baseline.tests.test_night_sparse_overlap
"$PY" -B -m agent_baseline.analysis.night_sparse_overlap_benchmark \
 --work "$ROOT/exp6_staging/batch99_find_missed_pre/artifacts" \
 --out "$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114/csr_benchmark" --stem "$STEM"
