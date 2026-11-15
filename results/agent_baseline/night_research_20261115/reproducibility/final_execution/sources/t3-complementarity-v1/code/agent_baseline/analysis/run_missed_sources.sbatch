#!/usr/bin/env bash
#SBATCH --job-name=missed-sources
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# GPU resources and shared log paths are supplied by the caller in NOTES.md.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${STAGING:?Set staging directory}"
: "${R:?Set read-only experiment root}"
: "${PYTHON_BIN:?Set host Python}"
MODE=${1:?inventory|fixed|baseline|generate|statistics|estimate|ask|report}
shift
cd "$STAGING"
export PYTHONPATH="$STAGING/code${PYTHONPATH:+:$PYTHONPATH}"
export OVCD_CODE="$R/code_w/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The login-node proxy does not exist on compute nodes; the API is reached directly.
while IFS= read -r var; do if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi; done < <(compgen -e)
export TMPDIR="$STAGING/tmp/slurm_$SLURM_JOB_ID"
export MPLCONFIGDIR="$TMPDIR/matplotlib" HF_HOME="$TMPDIR/hf" TORCH_HOME="$TMPDIR/torch"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME"
ARGS=("$MODE" --run "$R" --work "$STAGING/artifacts" --report "$STAGING/results/missed_sources.json")
if [[ "$MODE" == fixed || "$MODE" == generate ]]; then
    : "${SAM3_PYTHON:?Set existing SAM3 interpreter}"
    : "${SAM3_ROOT:?Set existing installed sam3 package root}"
    : "${SAM3_CHECKPOINT:?Set existing local checkpoint}"
    : "${SAM3_BPE:?Set existing local tokenizer resource}"
    ARGS+=(--sam3-python "$SAM3_PYTHON" --sam3-root "$SAM3_ROOT"
           --sam3-checkpoint "$SAM3_CHECKPOINT" --sam3-bpe "$SAM3_BPE")
    if [[ "$MODE" == generate ]]; then
        : "${VISION_CONFIG:?Set existing local vision config}"
        ARGS+=(--vision-config "$VISION_CONFIG")
    fi
fi
# Optional authorized job-completion notification; no credential values in script.
START=$SECONDS
finish() {
    RESULT=$?
    trap - EXIT
    STATUS=success
    if [[ "$RESULT" != 0 ]]; then STATUS=failed; fi
    if [[ -n "${NOTIFY_BIN:-}" ]]; then
        QQ_SUB="missed-sources $MODE $STATUS" \
        QQ_BODY="task=$MODE; status=$STATUS; exit=$RESULT; elapsed=$((SECONDS-START))s; output=$STAGING/artifacts; logs=$STAGING/tmp/logs" \
            "$NOTIFY_BIN" || true
    fi
    exit "$RESULT"
}
trap finish EXIT
if [[ "$MODE" == ask && -n "${CREDENTIAL_WRAPPER_FILE:-}" ]]; then
    # Credentials stay in the existing wrapper input, never in command text or logs.
    "$PYTHON_BIN" -B -m agent_baseline.run_with_key "$CREDENTIAL_WRAPPER_FILE" \
        "$PYTHON_BIN" -B -m agent_baseline.analysis.missed_sources "${ARGS[@]}" "$@"
else
    "$PYTHON_BIN" -B -m agent_baseline.analysis.missed_sources "${ARGS[@]}" "$@"
fi
