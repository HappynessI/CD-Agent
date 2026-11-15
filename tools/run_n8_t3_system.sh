#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit via Slurm}"
ROOT=/guisongxia01/pangchao/wangyihan
REPO=$ROOT/wyh/cd-agent-worktrees/n7-t3
OUT=$ROOT/wyh/CD-Agent/results/agent_baseline/n8_t3_20261115
PY=$ROOT/conda_envs/exp5-env/bin/python
split=${1:-tune}
export PYTHONPATH=$REPO/code:$REPO/code/ovcd_v0 OVCD_CODE=$REPO/code/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export AGENT_STOP_FILE=$OUT/API_STOP.json
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
started=$SECONDS
finish() { status=$?;trap - EXIT; QQ_SUB="N8 T3 system $split $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=$OUT/step3" "$ROOT/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m unittest agent_baseline.tests.test_n8_t3_system agent_baseline.tests.test_n8_t3_cost
"$PY" -B -m agent_baseline.analysis.n8_t3_prepare --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --night "$ROOT/wyh/CD-Agent/results/agent_baseline/night_t3_20261114" --semantic "$ROOT/wyh/exp6_staging/batch98_t3_routes/results/route_a/semantic" --out "$OUT" --split "$split"
"$PY" "$ROOT/Chang-Agent/tools/run_with_bailian_csv.py" "$ROOT/wyh/默认业务空间-apiKey-6214720.csv" "$PY" -B -m agent_baseline.analysis.n8_t3_verify --run "$ROOT/wyh/exp6_staging/agent_baseline_run" --out "$OUT" --split "$split" --budget-cny 50
