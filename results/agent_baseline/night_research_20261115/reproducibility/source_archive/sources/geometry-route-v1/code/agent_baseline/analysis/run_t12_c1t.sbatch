#!/usr/bin/env bash
#SBATCH --job-name=t12-c1t
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
mode=${1:?Supply task mode}
outputs=${2:?Supply output directory for completion notice}
shift 2
case "$mode" in
  make-t12) module=agent_baseline.analysis.make_t12_tasks ;;
  make-c1t) module=agent_baseline.analysis.make_c1t_tasks ;;
  sam3) module=agent_baseline.analysis.second_prepare ;;
  sam3-fixed) module=agent_baseline.fixed_routes ;;
  t12-fixed) module=agent_baseline.analysis.t12_fixed ;;
  c1t-fixed) module=agent_baseline.analysis.c1t_fixed ;;
  rule) module=agent_baseline.analysis.t12_rule ;;
  t13-ask) module=agent_baseline.analysis.t13_ask_select ;;
  metrics) module=agent_baseline.analysis.second_metrics ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
runtime="$PWD/code/.runtime/t12-c1t-$SLURM_JOB_ID"
mkdir -p "$runtime/tmp" "$runtime/matplotlib"
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib"
started=$SECONDS
notify() {
  result=$?
  trap - EXIT
  state=success
  [[ "$result" == 0 ]] || state=failed
  QQ_SUB="$mode: $state" \
  QQ_BODY="task=$module; status=$state; exit=$result; seconds=$((SECONDS-started)); outputs=$outputs; logs=Slurm stdout/stderr for job $SLURM_JOB_ID; inspect output report or error traceback" \
    /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || { echo 'QQ notification failed' >&2; [[ "$result" != 0 ]] || result=1; }
  exit "$result"
}
trap notify EXIT
"${PYTHON_BIN:-/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python}" -B -m "$module" "$@"
