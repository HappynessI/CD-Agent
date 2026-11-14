#!/usr/bin/env bash
#SBATCH --job-name=vision-t13-probe
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Run through sbatch}"
mode=${1:?Supply t13, t12, make-t3 or t3}
shift
case "$mode" in
  t13) module=agent_baseline.analysis.probe_t13 ;;
  t12) module=agent_baseline.analysis.probe_t12_segearth ;;
  make-t3) module=agent_baseline.analysis.make_t3_tasks ;;
  t3) module=agent_baseline.analysis.probe_t3 ;;
  *) echo "Unknown probe: $mode" >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime="$PWD/.runtime/probe-$SLURM_JOB_ID"
mkdir -p "$runtime" "$runtime/tmp" "$runtime/matplotlib"
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib"
started=$SECONDS
notify() {
  code=$?
  trap - EXIT
  state=success
  [[ "$code" == 0 ]] || state=failed
  QQ_SUB="vision-$mode: $state" \
  QQ_BODY="task=$module; status=$state; exit=$code; seconds=$((SECONDS-started)); outputs=$PWD/results/agent_baseline/data; logs=$PWD/.runtime/probe-logs; job=$SLURM_JOB_ID" \
    /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || { echo 'QQ notification failed' >&2; [[ "$code" != 0 ]] || code=1; }
  exit "$code"
}
trap notify EXIT
/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python -B -m "$module" "$@"
