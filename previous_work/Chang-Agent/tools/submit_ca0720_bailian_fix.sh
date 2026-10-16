#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
CSV="$ROOT/bailian-apikey.csv"
RUN_INDEX="${1:-6}"
SAMPLE="${2:-test_20_15}"
OUTPUT="$ROOT/outputs/CA_0720(${RUN_INDEX})-bailian-fix"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be set to the previously probed Slurm node (for example gpu43)" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy, based on the node probe" >&2
  exit 2
fi

if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi

NETWORK_PREFIX=()
if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy
  )
fi

COMMAND=(
  "${NETWORK_PREFIX[@]}"
  "$PYTHON" "$ROOT/tools/run_with_bailian_csv.py" "$CSV"
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  --input-root /guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256
  --samples "$SAMPLE"
  --output "$OUTPUT"
  --query building
  --max-steps 3
  --selection-policy conservative_best
  --verifier-max-regions 1
  --verifier-max-delta-regions-per-batch 1
  --staged-verifier-max-total-regions 32
  --verifier-min-region-area 4
  --verifier-region-padding-ratio 0.25
  --matching-mode overlap_presence
  --overlap-threshold 0.25
  --t12-min-instance-area 0
  --cd-min-instance-area 0
  --agent-backend bailian
  --verifier qwen_staged
  --staged-verifier-backend bailian
  --bailian-model qwen3-vl-plus
  --max-new-tokens 128
  --verifier-max-new-tokens 512
  --verifier-accept-threshold 0.82
  --verifier-retries 2
  --verifier-repetition-penalty 1.05
  --device-map auto
  --tool-device cuda
  --seed 42
  --action-retries 3
  --segagent-python /guisongxia01/pangchao/wangyihan/segagent-env/bin/python
  --omniovcd-python /guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python
  --simpleclick-checkpoint /guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth
  --sam3-checkpoint /guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt
  --sam3-bpe /guisongxia01/pangchao/wangyihan/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz
  --sam3-resolution 1008
)

COMMAND_STR=$(printf '%q ' "${COMMAND[@]}")
OUTPUT_Q=$(printf '%q' "$OUTPUT")
LOG_DIR_Q=$(printf '%q' "$OUTPUT/logs")
TMP_PREFIX="/tmp/CA_0720_${RUN_INDEX}_bailian_fix"
WRAP="set +e; ${COMMAND_STR}; status=\$?; if [ -d $OUTPUT_Q ]; then mkdir -p $LOG_DIR_Q; cp -f ${TMP_PREFIX}-\${SLURM_JOB_ID}.out $LOG_DIR_Q/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${TMP_PREFIX}-\${SLURM_JOB_ID}.err $LOG_DIR_Q/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; fi; QQ_SUB=\"CA 0720(${RUN_INDEX}) Bailian crop fix\" QQ_BODY=\"status=\$status; output=$OUTPUT; node=\${SLURM_JOB_NODELIST}; job=\${SLURM_JOB_ID}\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/CA_0720_${RUN_INDEX}_bailian_fix.qq.log 2>&1; rm -f ${TMP_PREFIX}-\${SLURM_JOB_ID}.out ${TMP_PREFIX}-\${SLURM_JOB_ID}.err; exit \$status"

SBATCH_EXTRA_ARGS=(--nodelist="$NODE")
if [[ -n "${CA_EXCLUDE_NODE:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--exclude="$CA_EXCLUDE_NODE")
fi
JOB=$(sbatch --parsable --job-name=CA0720BF --gres=gpu:1 --cpus-per-task=8 \
  "${SBATCH_EXTRA_ARGS[@]}" \
  --mem=64G --time=01:00:00 \
  --output="/tmp/CA_0720_${RUN_INDEX}_bailian_fix-%j.out" \
  --error="/tmp/CA_0720_${RUN_INDEX}_bailian_fix-%j.err" \
  --wrap="$WRAP")

printf 'job=%s\noutput=%s\nsample=%s\n' "$JOB" "$OUTPUT" "$SAMPLE"
