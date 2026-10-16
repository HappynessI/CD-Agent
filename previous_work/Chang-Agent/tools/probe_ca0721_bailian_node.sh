#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
SEGAGENT_PYTHON="/guisongxia01/pangchao/wangyihan/segagent-env/bin/python"
CSV="${CSV:-/guisongxia01/pangchao/wangyihan/wyh/bailian-apikey.csv}"
SIMPLECLICK_CHECKPOINT="/guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth"
SAM3_CHECKPOINT="/guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt"
SAM3_BPE="/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
JOB_NAME="CA21Probe"
SHARED_LOG_DIR="${PROBE_LOG_DIR:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs}"

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE is required" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy" >&2
  exit 2
fi
for executable in "$PYTHON" "$SEGAGENT_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "a required interpreter is missing or not executable" >&2
    exit 2
  fi
done
for required in \
  "$ROOT" \
  "$CSV" \
  "$ROOT/tools/run_with_bailian_csv.py" \
  "$ROOT/tools/run_levir_change_agent.py" \
  "$SIMPLECLICK_CHECKPOINT" \
  "$SAM3_CHECKPOINT" \
  "$SAM3_BPE"; do
  if [[ ! -e "$required" ]]; then
    echo "a required probe path is missing" >&2
    exit 2
  fi
done
if ! SLURM_PING="$(scontrol ping 2>&1)" || ! grep -q UP <<<"$SLURM_PING"; then
  echo "Slurm controller is not UP; no probe submitted" >&2
  exit 3
fi
if ! sinfo -N -n "$NODE" -h -o '%T' | grep -Eq 'idle|mix|alloc'; then
  echo "node is not available: $NODE" >&2
  exit 2
fi
if ! DUPLICATE_JOBS="$(
  squeue -h -u guisongxia --name="$JOB_NAME" -o '%i %T' 2>&1
)"; then
  echo "duplicate-job lookup failed; no probe submitted" >&2
  exit 3
fi
if [[ -n "$DUPLICATE_JOBS" ]]; then
  echo "an active probe with the same name already exists; no probe submitted" >&2
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  exit 2
fi

proxy_fingerprint() {
  env | LC_ALL=C awk -F= \
    'tolower($1) ~ /^(http_proxy|https_proxy|all_proxy|no_proxy)$/ {print}' \
    | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

proxy_configured() {
  [[ -n "${HTTP_PROXY:-}${HTTPS_PROXY:-}${ALL_PROXY:-}${http_proxy:-}${https_proxy:-}${all_proxy:-}" ]]
}

PROXY_LISTENER_CHECK='import os,socket; from urllib.parse import urlsplit; names=("HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy"); value=next((os.environ.get(name) for name in names if os.environ.get(name)), ""); parsed=urlsplit(value if "://" in value else "//"+value); host=parsed.hostname; port=parsed.port or {"http":80,"https":443,"socks":1080,"socks5":1080}.get(parsed.scheme); assert value and host and port; connection=socket.create_connection((host,port),3); connection.close()'
if [[ "$BAILIAN_NETWORK_MODE" == "proxy" ]] && ! proxy_configured; then
  echo "proxy mode requires a configured management-node proxy" >&2
  exit 3
fi
if [[ "$BAILIAN_NETWORK_MODE" == "proxy" ]] \
  && ! "$PYTHON" -c "$PROXY_LISTENER_CHECK" 2>/dev/null; then
  echo "management-node proxy listener is unavailable" >&2
  exit 3
fi
MANAGEMENT_PROXY_FINGERPRINT="$(proxy_fingerprint)"
mkdir -p "$SHARED_LOG_DIR"

PROBE_PY_Q=$(printf '%q' "$PYTHON")
SEGAGENT_PYTHON_Q=$(printf '%q' "$SEGAGENT_PYTHON")
ROOT_Q=$(printf '%q' "$ROOT")
CSV_Q=$(printf '%q' "$CSV")
SIMPLECLICK_CHECKPOINT_Q=$(printf '%q' "$SIMPLECLICK_CHECKPOINT")
SAM3_CHECKPOINT_Q=$(printf '%q' "$SAM3_CHECKPOINT")
SAM3_BPE_Q=$(printf '%q' "$SAM3_BPE")
NODE_Q=$(printf '%q' "$NODE")
MANAGEMENT_FP_Q=$(printf '%q' "$MANAGEMENT_PROXY_FINGERPRINT")
PROXY_LISTENER_CHECK_Q=$(printf '%q' "$PROXY_LISTENER_CHECK")
ENDPOINT_CHECK='set -eu; test -n "${DASHSCOPE_BASE_URL:-}"; endpoint_http="$(curl -q --silent --max-time 20 --output /dev/null --write-out "%{http_code}" "$DASHSCOPE_BASE_URL")" || { echo endpoint_request_failed >&2; exit 4; }; case "$endpoint_http" in [1-5][0-9][0-9]) ;; *) echo endpoint_http_invalid >&2; exit 4;; esac; printf "endpoint_http=%s\n" "$endpoint_http"'
ENDPOINT_CHECK_Q=$(printf '%q' "$ENDPOINT_CHECK")

if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX='env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy'
else
  NETWORK_PREFIX='env'
fi

WRAP="set -eu; proxy_fingerprint() { env | LC_ALL=C awk -F= 'tolower(\$1) ~ /^(http_proxy|https_proxy|all_proxy|no_proxy)$/ {print}' | LC_ALL=C sort | sha256sum | awk '{print \$1}'; }; proxy_configured() { [ -n \"\${HTTP_PROXY:-}\${HTTPS_PROXY:-}\${ALL_PROXY:-}\${http_proxy:-}\${https_proxy:-}\${all_proxy:-}\" ]; }; worker_node=\$(hostname -s); if [ \"\$worker_node\" != ${NODE_Q} ]; then echo fixed_node_mismatch >&2; exit 3; fi; echo worker_node=\$worker_node; test -d ${ROOT_Q}; test -x ${PROBE_PY_Q}; test -x ${SEGAGENT_PYTHON_Q}; test -f ${CSV_Q}; test -f ${ROOT_Q}/tools/run_with_bailian_csv.py; test -f ${ROOT_Q}/tools/run_levir_change_agent.py; test -f ${SIMPLECLICK_CHECKPOINT_Q}; test -f ${SAM3_CHECKPOINT_Q}; test -f ${SAM3_BPE_Q}; echo required_paths=ok; ${PROBE_PY_Q} -c 'import numpy, PIL; print(\"python_dependencies=ok\")'; nvidia-smi -L >/dev/null; echo gpu_visible=ok; worker_proxy_fingerprint=\$(proxy_fingerprint); echo worker_proxy_fingerprint=\$worker_proxy_fingerprint; if [ \"\$worker_proxy_fingerprint\" != ${MANAGEMENT_FP_Q} ]; then echo proxy_fingerprint_mismatch >&2; exit 3; fi; if [ ${BAILIAN_NETWORK_MODE} = proxy ]; then proxy_configured || { echo worker_proxy_not_configured >&2; exit 3; }; ${PROBE_PY_Q} -c ${PROXY_LISTENER_CHECK_Q} 2>/dev/null || { echo worker_proxy_listener_unavailable >&2; exit 3; }; echo worker_proxy_listener=ok; fi; echo network_mode=${BAILIAN_NETWORK_MODE}; ${NETWORK_PREFIX} ${PROBE_PY_Q} ${ROOT_Q}/tools/run_with_bailian_csv.py ${CSV_Q} /bin/bash -c ${ENDPOINT_CHECK_Q}; echo endpoint_reachable=ok"

JOB=$(sbatch --parsable --job-name="$JOB_NAME" --partition=gre \
  --nodelist="$NODE" --gres=gpu:1 \
  --cpus-per-task=1 --mem=4G --time=00:05:00 \
  --output="${SHARED_LOG_DIR}/.ca_0721_bailian_probe-%j.out" \
  --error="${SHARED_LOG_DIR}/.ca_0721_bailian_probe-%j.err" --wrap="$WRAP")
JOB_ID="${JOB%%;*}"
printf 'job=%s\nnode=%s\nnetwork_mode=%s\nmanagement_proxy_fingerprint=%s\nstdout=%s/.ca_0721_bailian_probe-%s.out\nstderr=%s/.ca_0721_bailian_probe-%s.err\n' \
  "$JOB_ID" "$NODE" "$BAILIAN_NETWORK_MODE" "$MANAGEMENT_PROXY_FINGERPRINT" \
  "$SHARED_LOG_DIR" "$JOB_ID" "$SHARED_LOG_DIR" "$JOB_ID"
