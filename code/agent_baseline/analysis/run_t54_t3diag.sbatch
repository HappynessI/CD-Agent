#!/usr/bin/env bash
#SBATCH --job-name=t54-t3diag
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# Caller supplies shared logs via --output and --error. GPU only for t3 mode.
set -euo pipefail
MODE=${1:?Usage: run_t54_t3diag.sbatch compare|noisy|t3 OUT [arguments]}
OUT=${2:?Output directory required}
shift 2
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${CODE_ROOT:?Set CODE_ROOT to the code directory to review/run}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the host Python}"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" OVCD_CODE="$CODE_ROOT/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The login-node proxy does not exist on compute nodes; the API is reached directly.
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
export TMPDIR="$OUT/.tmp/$SLURM_JOB_ID"
export MPLCONFIGDIR="$TMPDIR/matplotlib" HF_HOME="$TMPDIR/hf" TORCH_HOME="$TMPDIR/torch"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME" "$OUT/logs"
START=$SECONDS
finish() {
    RESULT=$?
    trap - EXIT
    STATUS=success
    if [[ "$RESULT" != 0 ]]; then STATUS=failed; fi
    QQ_SUB="T5.4/T3 $MODE $STATUS" \
    QQ_BODY="task=$MODE; status=$STATUS; exit=$RESULT; elapsed=$((SECONDS-START))s; artifacts=$OUT; logs=$OUT/logs" \
        /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || echo 'QQ notification failed' >&2
    exit "$RESULT"
}
trap finish EXIT
cd "$CODE_ROOT"
case "$MODE" in
    compare)
        : "${BAILIAN_CSV:?Set the existing credential export path; do not put credentials in this script}"
        "$PYTHON_BIN" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" \
            "$PYTHON_BIN" -B -m agent_baseline.analysis.t54_ask_compare --output-dir "$OUT" "$@"
        ;;
    noisy)
        "$PYTHON_BIN" -B -m agent_baseline.analysis.make_t54_noisy --output-root "$OUT" "$@"
        ;;
    t3)
        "$PYTHON_BIN" -B -m agent_baseline.analysis.t3_diag --output "$OUT/report.json" "$@"
        ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
