#!/usr/bin/env bash
set -euo pipefail

# Single-sample hosted smoke for the v10 batched discrete audit protocol.
# Usage after a successful node/network probe:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0723_batched_rubric_v10.sh 2 test_85_16

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
CSV="${CSV:-$ROOT/bailian-apikey.csv}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
RUN_INDEX="${1:-2}"
SAMPLE="${2:-test_85_16}"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
BAILIAN_MODEL="${BAILIAN_MODEL:-qwen3.7-plus-2026-05-26}"
BAILIAN_ENABLE_THINKING="${BAILIAN_ENABLE_THINKING:-1}"
BAILIAN_THINKING_BUDGET="${BAILIAN_THINKING_BUDGET:-256}"
PARENT="${OUTPUT:-$ROOT/outputs/CA_0723(${RUN_INDEX})-bailian-proposal-batched-rubric-v10}"
CHILD="$PARENT/proposal"
SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0723_${RUN_INDEX}_batched_rubric_v10}"

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be set to a node that passed the Bailian probe" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy" >&2
  exit 2
fi
if [[ ! "$RUN_INDEX" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$SAMPLE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "SAMPLE contains unsupported characters" >&2
  exit 2
fi
if [[ "$BAILIAN_ENABLE_THINKING" != "0" && "$BAILIAN_ENABLE_THINKING" != "1" ]]; then
  echo "BAILIAN_ENABLE_THINKING must be 0 or 1" >&2
  exit 2
fi
if [[ "$BAILIAN_ENABLE_THINKING" == "1" && ! "$BAILIAN_THINKING_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "BAILIAN_THINKING_BUDGET must be positive" >&2
  exit 2
fi
if [[ -e "$PARENT" ]]; then
  echo "output already exists: $PARENT" >&2
  exit 2
fi
for required in "$PYTHON" "$CSV" "$INPUT_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done
if ! scontrol ping | grep -q UP; then
  echo "Slurm controller is not UP" >&2
  exit 3
fi

mkdir -p "$PARENT" "$SHARED_LOG_ROOT"
printf '%s\n' \
  '# CA_0723 staged verifier v10 smoke' \
  '' \
  "- node: \`$NODE\`" \
  "- network_mode: \`$BAILIAN_NETWORK_MODE\`" \
  "- sample: \`$SAMPLE\`" \
  "- model: \`$BAILIAN_MODEL\`" \
  "- thinking: \`$BAILIAN_ENABLE_THINKING\`, budget \`$BAILIAN_THINKING_BUDGET\`" \
  '- protocol: `v10 batched exhaustive audit + discrete checklist + runtime-derived diagnosis/quality`' \
  '- prompt: `active audit hard-negative instruction + three abstract text few-shot contrasts`' \
  '- confidence: `no model-authored numeric confidence or initial quality`' \
  '- max action steps: `1`; audit batches do not consume action steps' \
  '- GT policy: `loaded only after rollout completion`' \
  >"$PARENT/experiment_manifest.md"

NETWORK_PREFIX=()
if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy
    CHANGE_AGENT_STAGED_PROTOCOL_VERSION=v10
  )
else
  NETWORK_PREFIX=(env CHANGE_AGENT_STAGED_PROTOCOL_VERSION=v10)
fi

COMMAND=(
  "${NETWORK_PREFIX[@]}"
  "$PYTHON" "$ROOT/tools/run_with_bailian_csv.py" "$CSV"
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  --input-root "$INPUT_ROOT"
  --samples "$SAMPLE"
  --output "$CHILD"
  --query building
  --max-steps 1
  --selection-policy conservative_best
  --proposal-mode proposal
  --verifier-max-regions 1
  --verifier-max-delta-regions-per-batch 1
  --staged-verifier-max-total-regions 32
  --verifier-max-selected-regions 3
  --verifier-min-region-area 4
  --verifier-region-padding-ratio 0.25
  --matching-mode overlap_presence
  --overlap-threshold 0.25
  --agent-backend bailian
  --verifier qwen_staged
  --staged-verifier-backend bailian
  --bailian-model "$BAILIAN_MODEL"
  --max-new-tokens 128
  --verifier-max-new-tokens 1024
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
if [[ "$BAILIAN_ENABLE_THINKING" == "1" ]]; then
  COMMAND+=(--bailian-enable-thinking --bailian-thinking-budget "$BAILIAN_THINKING_BUDGET")
fi

COMMAND_STRING=$(printf '%q ' "${COMMAND[@]}")
CHILD_LOGS_Q=$(printf '%q' "$CHILD/logs")
SHARED_PREFIX="$SHARED_LOG_ROOT/proposal"
WRAP="set +e; ${COMMAND_STRING}; status=\$?; mkdir -p ${CHILD_LOGS_Q}; cp -f ${SHARED_PREFIX}-\${SLURM_JOB_ID}.out ${CHILD_LOGS_Q}/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${SHARED_PREFIX}-\${SLURM_JOB_ID}.err ${CHILD_LOGS_Q}/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; QQ_SUB=\"CA 0723(${RUN_INDEX}) batched rubric v10\" QQ_BODY=\"status=\$status; sample=${SAMPLE}; output=${CHILD}; node=\${SLURM_JOB_NODELIST}; job=\${SLURM_JOB_ID}\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/ca0723_${RUN_INDEX}_\${SLURM_JOB_ID}.qq.log 2>&1; rm -f ${SHARED_PREFIX}-\${SLURM_JOB_ID}.out ${SHARED_PREFIX}-\${SLURM_JOB_ID}.err /tmp/ca0723_${RUN_INDEX}_\${SLURM_JOB_ID}.qq.log; rmdir ${SHARED_LOG_ROOT} 2>/dev/null || true; exit \$status"

JOB=$(sbatch --parsable --job-name="CA23${RUN_INDEX}V10" --nodelist="$NODE" \
  --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=01:00:00 \
  --output="${SHARED_PREFIX}-%j.out" --error="${SHARED_PREFIX}-%j.err" \
  --wrap="$WRAP")
printf 'job=%s\noutput=%s\nnode=%s\nnetwork_mode=%s\n' \
  "$JOB" "$PARENT" "$NODE" "$BAILIAN_NETWORK_MODE"
