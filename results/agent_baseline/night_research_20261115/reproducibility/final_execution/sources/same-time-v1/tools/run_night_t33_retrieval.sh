#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=${T3_CODE:-$ROOT/wyh/cd-t3-bridge}
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114
PY=$ROOT/conda_envs/exp5-env/bin/python
export OVCD_CODE=${T3_OVCD_CODE:-$REPO/code/ovcd_v0}
export PYTHONPATH=$REPO/code:$OVCD_CODE PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
started=$SECONDS
finish() {
    status=$?;trap - EXIT
    QQ_SUB="T3.3 night retrieval $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=$OUT/t33/retrieval; summary and logs" \
      "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true
    exit "$status"
}
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_night_t33_retrieval
"$PY" "$ROOT/Chang-Agent/tools/run_with_bailian_csv.py" "$ROOT/wyh/默认业务空间-apiKey-6214720.csv" \
  "$PY" -B -m agent_baseline.analysis.night_t33_retrieval --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --night "$OUT" --parse-variant "${1:-B1}"
