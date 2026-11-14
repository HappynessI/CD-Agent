#!/usr/bin/env bash
#SBATCH --job-name=t3x
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
mode=${1:?Supply make, fixed or metrics}
outputs=${2:?Supply output directory for completion notice}
shift 2
case "$mode" in
  make) module=agent_baseline.analysis.make_t3x_tasks ;;
  fixed) module=agent_baseline.analysis.t3x_fixed ;;
  metrics) module=agent_baseline.analysis.t3x_metrics ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export OVCD_CODE="$PWD/code/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
runtime="$PWD/code/.runtime/t3x-$SLURM_JOB_ID"
mkdir -p "$runtime/tmp" "$runtime/matplotlib"
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib"
started=$SECONDS
notify() {
  result=$?
  trap - EXIT
  state=success
  [[ "$result" == 0 ]] || state=failed
  detail=$(tail -n 3 "$runtime/run.log")
  QQ_SUB="T3x $mode: $state" \
  QQ_BODY="task=$module; status=$state; exit=$result; seconds=$((SECONDS-started)); outputs=$outputs; job=$SLURM_JOB_ID; result or error: $detail; log=$runtime/run.log" \
    /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || { echo 'QQ notification failed' >&2; [[ "$result" != 0 ]] || result=1; }
  exit "$result"
}
trap notify EXIT
"${PYTHON_BIN:-/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python}" -B -m "$module" "$@" 2>&1 | tee "$runtime/run.log"
