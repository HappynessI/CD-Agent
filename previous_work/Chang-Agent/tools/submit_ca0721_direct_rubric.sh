#!/usr/bin/env bash
set -euo pipefail

# Standard Direct-rubric Bailian submission.
# Usage:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0721_direct_rubric.sh 11 test_20_15 test_78_13 test_85_16
# Override OUTPUT, JOB_NAME, MAX_STEPS, or SAMPLES_CSV when needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
CSV="${CSV:-$ROOT/bailian-apikey.csv}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
RUN_INDEX="${1:-11}"
shift || true

if (( $# > 0 )); then
  SAMPLES=("$@")
else
  SAMPLES_CSV="${SAMPLES_CSV:-test_20_15,test_78_13,test_85_16}"
  IFS=',' read -r -a SAMPLES <<< "$SAMPLES_CSV"
fi

NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0721(${RUN_INDEX})-bailian-direct-rubric-v3}"
JOB_NAME="${JOB_NAME:-CA21${RUN_INDEX}DR}"
MAX_STEPS="${MAX_STEPS:-3}"

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be set to the node passed by tools/probe_ca0721_bailian_node.sh" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy" >&2
  exit 2
fi
if [[ ! "$RUN_INDEX" =~ ^[0-9]+$ || ! "$MAX_STEPS" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX and MAX_STEPS must be non-negative integers" >&2
  exit 2
fi
if (( ${#SAMPLES[@]} == 0 )); then
  echo "at least one sample is required" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
for required in "$PYTHON" "$CSV" "$INPUT_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

# Fail fast when this management shell cannot reach Slurm controller.  Without
# this check sbatch may retry stream creation for a long time and hide root cause.
if ! SLURM_PING=$(scontrol ping 2>&1); then
  echo "Slurm preflight failed; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
if ! grep -q 'UP' <<< "$SLURM_PING"; then
  echo "Slurm controller is not UP; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
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
  --input-root "$INPUT_ROOT"
  --samples "${SAMPLES[@]}"
  --output "$OUTPUT"
  --query building
  --max-steps "$MAX_STEPS"
  --selection-policy conservative_best
  --proposal-mode direct
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

COMMAND_STRING=$(printf '%q ' "${COMMAND[@]}")
OUTPUT_Q=$(printf '%q' "$OUTPUT")
LOG_DIR_Q=$(printf '%q' "$OUTPUT/logs")
TMP_PREFIX="/tmp/CA_0721_${RUN_INDEX}_direct_rubric"
QQ_LOG="/tmp/CA_0721_${RUN_INDEX}_direct_rubric-\${SLURM_JOB_ID}.qq.log"
WRAP="set +e; ${COMMAND_STRING}; status=\$?; if [ -d ${OUTPUT_Q} ]; then mkdir -p ${LOG_DIR_Q}; cp -f ${TMP_PREFIX}-\${SLURM_JOB_ID}.out ${LOG_DIR_Q}/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${TMP_PREFIX}-\${SLURM_JOB_ID}.err ${LOG_DIR_Q}/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; fi; QQ_SUB=\"CA 0721(${RUN_INDEX}) Direct rubric\" QQ_BODY=\"status=\$status; output=${OUTPUT}; node=\${SLURM_JOB_NODELIST}; job=\${SLURM_JOB_ID}\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >${QQ_LOG} 2>&1; rm -f ${TMP_PREFIX}-\${SLURM_JOB_ID}.out ${TMP_PREFIX}-\${SLURM_JOB_ID}.err ${QQ_LOG}; exit \$status"

JOB=$(sbatch --parsable --job-name="$JOB_NAME" --nodelist="$NODE" \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
  --output="${TMP_PREFIX}-%j.out" --error="${TMP_PREFIX}-%j.err" \
  --wrap="$WRAP")

printf 'job=%s\noutput=%s\nsamples=%s\nnode=%s\nnetwork_mode=%s\n' \
  "$JOB" "$OUTPUT" "${SAMPLES[*]}" "$NODE" "$BAILIAN_NETWORK_MODE"
