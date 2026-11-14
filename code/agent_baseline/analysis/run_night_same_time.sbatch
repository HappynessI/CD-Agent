#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${BAILIAN_CSV:?Existing workspace credential CSV}"
ROOT=/guisongxia01/pangchao/wangyihan/wyh
CODE=${NIGHT_MISSED_CODE:-$ROOT/cd-missed-verifier/code}
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
BASE="$ROOT/CD-Agent/results/agent_baseline/night_missed_20261114"
export PYTHONPATH="$CODE:$ROOT/exp6_staging/agent_baseline_run/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
trap 'status=$?; QQ_SUB="same-time six-case diagnosis" QQ_BODY="job=$SLURM_JOB_ID status=$status output=$BASE/same_time_counterfactual" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || true' EXIT
"$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" "$PY" -B -m agent_baseline.analysis.night_same_time \
 --views "$BASE" --pilot "$ROOT/CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot" \
 --out "$BASE/same_time_counterfactual"
