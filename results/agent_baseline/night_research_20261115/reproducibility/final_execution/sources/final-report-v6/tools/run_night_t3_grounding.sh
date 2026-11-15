#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${T3_CODE:-$ROOT/wyh/cd-t3-bridge}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114
R=$ROOT/wyh/exp6_staging/agent_baseline_run
PY=$ROOT/conda_envs/exp5-env/bin/python
variant=${1:-all}
export OVCD_CODE=${T3_OVCD_CODE:-$REPO/code/ovcd_v0}
export PYTHONPATH=$REPO/code:$OVCD_CODE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
mkdir -p "$OUT/logs"
started=$SECONDS
finish() {
    status=$?;trap - EXIT
    QQ_SUB="T3 night grounding $variant $SLURM_JOB_ID" \
    QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=$OUT; results/errors: B0/B1/B2 summaries and logs" \
      "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true
    exit "$status"
}
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_night_t3_grounding
variants=("$variant")
if [[ "$variant" == all ]]; then variants=(audit B1 B2); fi
for current in "${variants[@]}"; do
    "$PY" "$ROOT/Chang-Agent/tools/run_with_bailian_csv.py" "$ROOT/wyh/默认业务空间-apiKey-6214720.csv" \
      "$PY" -B -m agent_baseline.analysis.night_t3_grounding --run "$R" \
      --baseline "$ROOT/wyh/CD-Agent/results/agent_baseline/t3_bridge_20261114/dev60/summary.json" \
      --output "$OUT" --variant "$current"
done
