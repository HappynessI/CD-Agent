#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-candidate-geometry-20261115/logs/routing-confirm-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-candidate-geometry-20261115/logs/routing-confirm-%j.err
set -euo pipefail
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/.runtime/night-20261114/geometry-route-v1/code:/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/code_w/ovcd_v0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python
ARGS=(--views /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114 --pilot /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/missed_verifier_20261114/pilot --work /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/batch99_find_missed_pre/artifacts --run /guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run --out /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/geometry_route)
started=$SECONDS
finish() { status=$?; trap - EXIT; QQ_SUB="Night geometry routing confirm $SLURM_JOB_ID" QQ_BODY="status=$status; elapsed=$((SECONDS-started))s; output=/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/geometry_route/confirm; API0; exploratory historical exposure" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/dev/null 2>&1 || true; exit "$status"; }
trap finish EXIT
"$PY" -B -m agent_baseline.analysis.night_geometry_route freeze "${ARGS[@]}"
"$PY" -B -m agent_baseline.analysis.night_geometry_route evaluate --split confirm "${ARGS[@]}"
