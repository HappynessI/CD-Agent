#!/usr/bin/env bash
#SBATCH --job-name=t3-routes
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit through sbatch}"
mode=${1:?route-a|prepare-b|verify|score-b|report|agent-metrics}
shift
: "${R:?Set R to the existing run root}"
export OVCD_CODE="${OVCD_CODE:-$R/code_w/ovcd_v0}"
# The login-node proxy does not exist on compute nodes; the API is reached directly.
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
export PYTHONPATH="$PWD/code:$OVCD_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$PWD/.runtime/t3-routes-$SLURM_JOB_ID"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" results/logs
started=$SECONDS
finish() {
  code=$?
  trap - EXIT
  state=success
  [[ "$code" == 0 ]] || state=failed
  QQ_SUB="T3 routes $mode: $state" \
  QQ_BODY="task=$mode; status=$state; exit=$code; seconds=$((SECONDS-started)); artifacts=$PWD/results; log=results/logs/$SLURM_JOB_ID.log; key result or error: $(tail -n 3 "results/logs/$SLURM_JOB_ID.log")" \
    /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || echo 'QQ notification failed' >&2
  exit "$code"
}
trap finish EXIT
python_bin=${PYTHON_BIN:-/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python}
if [[ "$mode" == agent-metrics ]]; then
  "$python_bin" -B code/t3_routes_bootstrap.py agent_baseline.analysis.t3_routes_agent metrics --run-root "$R" --output-dir results "$@" 2>&1 | tee "results/logs/$SLURM_JOB_ID.log"
else
  "$python_bin" -B code/t3_routes_bootstrap.py agent_baseline.analysis.t3_routes "$mode" --run-root "$R" --output-dir results "$@" 2>&1 | tee "results/logs/$SLURM_JOB_ID.log"
fi
