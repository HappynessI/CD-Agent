#!/usr/bin/env bash
#SBATCH --job-name=agent-bf
#SBATCH --partition=gre
#SBATCH --exclude=gpu04,gpu06,gpu16,gpu20,gpu37
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/logs/%x-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run/logs/%x-%j.err
# Pre-create logs before submission; custom AGENT_RUN also needs sbatch output/error overrides.
# Multiple loop argument groups may be separated by :::.
set -euo pipefail
WORKSPACE=${WORKSPACE:-/guisongxia01/pangchao/wangyihan}
export AGENT_RUN=${AGENT_RUN:-$WORKSPACE/wyh/exp6_staging/agent_baseline_run}
CODE=${AGENT_CODE:-$WORKSPACE/wyh/CD-Agent/code/agent_baseline}
PY=$WORKSPACE/conda_envs/exp5-env/bin/python
mkdir -p "$AGENT_RUN/logs" "$AGENT_RUN/.tmp/${SLURM_JOB_ID:-manual}"
exec >>"$AGENT_RUN/logs/${SLURM_JOB_NAME:-agent-bf}-${SLURM_JOB_ID:-manual}.runtime.log" 2>&1
started=$SECONDS
pids=()
finish() {
    status=$?; trap - EXIT
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    QQ_SUB="agent-bf ${SLURM_JOB_ID:-manual}" \
    QQ_BODY="status=$status elapsed=$((SECONDS-started))s output=$AGENT_RUN; results/errors in logs" \
        "$WORKSPACE/.local/bin/wangyihan-send-qq" >/dev/null 2>&1 || true
    exit "$status"
}
trap finish EXIT
: "${BAILIAN_CSV:?BAILIAN_CSV must be supplied in the submission environment}"
if (($# == 0)); then
    echo 'Supply loop_json.py arguments, optionally in groups separated by :::.' >&2
    exit 2
fi
# Clear every exported proxy variant without printing its value.
while IFS= read -r var; do
    if [[ ${var,,} == *proxy* ]]; then unset "$var"; fi
done < <(compgen -e)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/guisongxia01/pangchao/haoxiangyu/project/DynamicEarth/third_party/segment_anything${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=$AGENT_RUN/.tmp/${SLURM_JOB_ID:-manual}
cd "$CODE"
worker_args=()
if [[ ${USE_LOCAL_WORKERS:-1} == 1 ]]; then
    read -r ASK_PORT SAM3_PORT < <("$PY" - <<'PY'
import socket
sockets = [socket.socket(), socket.socket()]
for sock in sockets:
    sock.bind(('127.0.0.1', 0))
print(*(sock.getsockname()[1] for sock in sockets))
for sock in sockets:
    sock.close()
PY
    )
    # An API verifier needs no local model, which saves GPU memory and loading time.
    if [[ ${USE_ASK_WORKER:-1} == 1 ]]; then
        "$WORKSPACE/conda_envs/vlm-vs-onejev-env/bin/python" -B workers/ask_worker.py --port "$ASK_PORT" \
            >"$AGENT_RUN/logs/ask-${SLURM_JOB_ID:-manual}.log" 2>&1 &
        pids+=("$!")
    fi
    "$WORKSPACE/conda_envs/omniovcd-env/bin/python" -B workers/sam3_worker.py --port "$SAM3_PORT" \
        >"$AGENT_RUN/logs/sam3-${SLURM_JOB_ID:-manual}.log" 2>&1 &
    pids+=("$!")
    wait_ports=("$SAM3_PORT")
    [[ ${USE_ASK_WORKER:-1} == 1 ]] && wait_ports+=("$ASK_PORT")
    "$PY" - "${wait_ports[@]}" <<'PY'
import json
import sys
import time
import urllib.request
pending = set(sys.argv[1:])
deadline = time.monotonic() + 900
while pending and time.monotonic() < deadline:
    for port in list(pending):
        try:
            with urllib.request.urlopen('http://127.0.0.1:' + port + '/health', timeout=2) as response:
                if json.load(response).get('ok'):
                    pending.remove(port)
        except OSError:
            pass
    if pending:
        time.sleep(1)
if pending:
    raise SystemExit('Worker readiness deadline exceeded; inspect worker logs.')
PY
    worker_args=(--sam3-worker "http://127.0.0.1:$SAM3_PORT")
    [[ ${USE_ASK_WORKER:-1} == 1 ]] && worker_args+=(--ask-worker "http://127.0.0.1:$ASK_PORT")
fi
run_group() {
    "$PY" "$WORKSPACE/Chang-Agent/tools/run_with_bailian_csv.py" "$BAILIAN_CSV" \
        "$PY" -B loop_json.py "${worker_args[@]}" "$@"
}
group=()
for value in "$@" :::; do
    if [[ $value == ::: ]]; then
        if ((${#group[@]})); then run_group "${group[@]}"; group=(); fi
    else
        group+=("$value")
    fi
done
