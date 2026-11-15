#!/usr/bin/env bash
#SBATCH --job-name=ab-t54
#SBATCH --partition=gre
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
# Pass --output/--error explicitly, pointing to the experiment's logs/ directory.
set -euo pipefail
MODE=${1:?Usage: run_t54.sbatch prepare|fixed|rule|metrics OUTPUT_DIR [arguments]}
OUT=${2:?Output directory required}
shift 2
C=${CODE_ROOT:-${SLURM_SUBMIT_DIR:?}/code}
PY=${PYTHON_BIN:-/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env/bin/python}
SAM_PY=${SAM3_PYTHON_BIN:-/guisongxia01/pangchao/wangyihan/conda_envs/omniovcd-env/bin/python}
export PYTHONPATH="$C${PYTHONPATH:+:$PYTHONPATH}" OVCD_CODE="$C/ovcd_v0"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$OUT/.tmp/${SLURM_JOB_ID:?Submit this script through sbatch}"
mkdir -p "$TMPDIR" "$OUT/logs"
START=$SECONDS
WORKER=
finish() {
    RESULT=$?
    trap - EXIT
    if [[ -n "$WORKER" ]]; then
        if kill -0 "$WORKER" 2>/dev/null; then kill "$WORKER"; fi
        if wait "$WORKER"; then :; else :; fi
    fi
    if ! rmdir "$TMPDIR"; then echo "Temporary files remain: $TMPDIR" >&2; fi
    STATUS=success
    if [[ "$RESULT" != 0 ]]; then STATUS=failed; fi
    QQ_SUB="T5.4 $MODE $STATUS" \
    QQ_BODY="task=T5.4/$MODE; status=$STATUS; exit=$RESULT; elapsed=$((SECONDS-START))s; artifacts=$OUT; logs=$OUT/logs" \
        /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq || { echo 'QQ notification failed' >&2; if [[ "$RESULT" == 0 ]]; then RESULT=1; fi; }
    exit "$RESULT"
}
trap finish EXIT
cd "$C"
case "$MODE" in
    prepare) "$PY" -B -m agent_baseline.analysis.make_t54_tasks "$@" ;;
    fixed)
        : "${AGENT_RUN:?Set AGENT_RUN to the cache/run root}"
        PORT=$("$PY" -B -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
        "$SAM_PY" -B -m agent_baseline.workers.sam3_worker --port "$PORT" --run "$AGENT_RUN" > "$OUT/logs/sam3-$SLURM_JOB_ID.log" 2>&1 &
        WORKER=$!
        READY=0
        for ((i=0; i<180; i++)); do
            if ! kill -0 "$WORKER" 2>/dev/null; then
                echo "SAM3 worker exited; see $OUT/logs/sam3-$SLURM_JOB_ID.log" >&2
                exit 1
            fi
            if curl --silent --fail --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health" > /dev/null; then
                READY=1
                break
            fi
            sleep 2
        done
        if [[ "$READY" != 1 ]]; then echo 'SAM3 worker did not become ready' >&2; exit 1; fi
        NO_PROXY='*' no_proxy='*' "$PY" -B -m agent_baseline.analysis.t54_fixed \
            --sam3-worker "http://127.0.0.1:$PORT" --output-dir "$OUT" "$@"
        ;;
    rule)
        : "${BAILIAN_CSV:?Set BAILIAN_CSV to the existing local credential export path}"
        "$PY" -B -m agent_baseline.run_with_key "$BAILIAN_CSV" \
            "$PY" -B -m agent_baseline.analysis.t54_rule --output-dir "$OUT" "$@"
        ;;
    metrics) "$PY" -B -m agent_baseline.analysis.t54_metrics "$@" ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
