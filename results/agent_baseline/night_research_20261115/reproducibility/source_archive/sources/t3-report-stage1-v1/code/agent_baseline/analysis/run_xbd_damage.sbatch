#!/usr/bin/env bash
#SBATCH --job-name=xbd-probe10
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${R:?Set the existing run root}"
: "${XBD_DATA_ROOT:?Set the read-only xBD data root}"
mode=${1:-ask}
if [[ $# -gt 0 ]]; then shift; fi
export OVCD_CODE="${OVCD_CODE:-$R/code_w/ovcd_v0}"
export PYTHONPATH="$PWD/code:$OVCD_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$PWD/.runtime/xbd-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" results/logs
# Follow the existing compute-node wrapper's direct network configuration.
while IFS= read -r var; do
  if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi
done < <(compgen -e)
python_bin=${PYTHON_BIN:-/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env/bin/python}
started=$SECONDS
finish() {
  exit_code=$?
  trap - EXIT
  state=success
  [[ $exit_code == 0 ]] || state=failed
  printf 'status=%s; exit=%s; seconds=%s\n' "$state" "$exit_code" "$((SECONDS-started))"
  notifier=${QQ_NOTIFIER:-/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq}
  if [[ -x "$notifier" ]]; then
    QQ_SUB="xBD probe10 $mode: $state" \
    QQ_BODY="status=$state; exit=$exit_code; seconds=$((SECONDS-started)); result: results/xbd_probe10_report.json; logs: results/logs/xbd-$SLURM_JOB_ID.log; artifacts=$PWD/results" \
      "$notifier" || printf 'Completion notification failed\n' >&2
  fi
  exit "$exit_code"
}
trap finish EXIT
command=("$python_bin" -B -m agent_baseline.analysis.xbd_damage "$mode" --data-root "$XBD_DATA_ROOT")
if [[ "$mode" == ask ]]; then
  : "${KEY_FILE:?Provide your own key-file path for the existing injection wrapper}"
  "$python_bin" -B "$R/code_t/agent_baseline/run_with_key.py" "$KEY_FILE" \
    "${command[@]}" "$@" 2>&1 | tee "results/logs/xbd-$SLURM_JOB_ID.log"
else
  "${command[@]}" "$@" 2>&1 | tee "results/logs/xbd-$SLURM_JOB_ID.log"
fi
