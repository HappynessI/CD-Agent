#!/usr/bin/env bash
set -euo pipefail

# Submit the marked-transition Direct/Proposal/Hybrid smoke ablation.
# Usage:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0722_context_fix_ablation.sh 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
CSV="${CSV:-$ROOT/bailian-apikey.csv}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
RUN_INDEX="${1:-2}"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
MEMORY="${MEMORY:-16G}"
PARENT="${OUTPUT:-$ROOT/outputs/CA_0722(${RUN_INDEX})-bailian-context-fix-3arm}"
SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0722_${RUN_INDEX}_marked_transition}"
PROTOCOL_VERSION="${PROTOCOL_VERSION:-v8}"
PROTOCOL_EVIDENCE="${PROTOCOL_EVIDENCE:-action-scoped delta masks plus delta-highlighted T1/T2 RGB and previous/candidate masks}"
PROTOCOL_TRANSITION="${PROTOCOL_TRANSITION:-v8 deterministic T1/T2 facts plus unique editable-target resolution and action-scoped delta RGB evidence}"
BAILIAN_MODEL="${BAILIAN_MODEL:-qwen3-vl-plus}"
BAILIAN_ENABLE_THINKING="${BAILIAN_ENABLE_THINKING:-0}"
BAILIAN_THINKING_BUDGET="${BAILIAN_THINKING_BUDGET:-}"
SAMPLES=(test_20_15 test_78_13 test_85_16)
read -r -a MODES <<< "${MODES:-direct proposal hybrid}"

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be set to the node passed by tools/probe_ca0721_bailian_node.sh" >&2
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
if [[ "$BAILIAN_ENABLE_THINKING" != "0" && "$BAILIAN_ENABLE_THINKING" != "1" ]]; then
  echo "BAILIAN_ENABLE_THINKING must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$BAILIAN_THINKING_BUDGET" && ! "$BAILIAN_THINKING_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "BAILIAN_THINKING_BUDGET must be a positive integer" >&2
  exit 2
fi
if [[ "$BAILIAN_ENABLE_THINKING" == "0" && -n "$BAILIAN_THINKING_BUDGET" ]]; then
  echo "BAILIAN_THINKING_BUDGET requires BAILIAN_ENABLE_THINKING=1" >&2
  exit 2
fi
for mode in "${MODES[@]}"; do
  if [[ "$mode" != "direct" && "$mode" != "proposal" && "$mode" != "hybrid" ]]; then
    echo "unsupported proposal mode: $mode" >&2
    exit 2
  fi
done
if [[ -e "$PARENT" ]]; then
  echo "output parent already exists: $PARENT" >&2
  exit 2
fi
for required in "$PYTHON" "$CSV" "$INPUT_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

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

mkdir -p "$PARENT"
mkdir -p "$SHARED_LOG_ROOT"
printf '%s\n' \
  '# BaiLian marked-transition verifier ablation' \
  '' \
  "- node: \`$NODE\`" \
  "- network_mode: \`$BAILIAN_NETWORK_MODE\`" \
  "- memory_per_arm: \`$MEMORY\`" \
  '- samples: `test_20_15 test_78_13 test_85_16`' \
  "- arms: \`${MODES[*]}\`" \
  "- bailian_model: \`$BAILIAN_MODEL\`" \
  "- bailian_enable_thinking: \`$BAILIAN_ENABLE_THINKING\`" \
  "- bailian_thinking_budget: \`${BAILIAN_THINKING_BUDGET:-none}\`" \
  '- regional evidence: `active-region-marked global overview plus exact local crops`' \
  "- candidate evidence: \`$PROTOCOL_EVIDENCE\`" \
  "- transition policy: \`$PROTOCOL_TRANSITION\`" \
  '- finish policy: `all Environment audit regions require sufficient evidence and diagnosis=none`' \
  '- negative point: `SimpleClick-supported subtraction inside deterministic point ROI`' \
  '- GT policy: `loaded only after each arm rollout completes`' \
  > "$PARENT/experiment_manifest.md"

NETWORK_PREFIX=()
if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy
    "CHANGE_AGENT_STAGED_PROTOCOL_VERSION=$PROTOCOL_VERSION"
  )
else
  NETWORK_PREFIX=(env "CHANGE_AGENT_STAGED_PROTOCOL_VERSION=$PROTOCOL_VERSION")
fi

submit_arm() {
  local mode="$1"
  local child="$PARENT/$mode"
  local shared_prefix="$SHARED_LOG_ROOT/${mode}"
  local command=(
    "${NETWORK_PREFIX[@]}"
    "$PYTHON" "$ROOT/tools/run_with_bailian_csv.py" "$CSV"
    "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
    --input-root "$INPUT_ROOT"
    --samples "${SAMPLES[@]}"
    --output "$child"
    --query building
    --max-steps 3
    --selection-policy conservative_best
    --proposal-mode "$mode"
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
    --bailian-model "$BAILIAN_MODEL"
    --max-new-tokens 128
    --verifier-max-new-tokens 512
    --verifier-accept-threshold 0.82
    --verifier-min-visual-confidence 0.6
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
    command+=(--bailian-enable-thinking)
  fi
  if [[ -n "$BAILIAN_THINKING_BUDGET" ]]; then
    command+=(--bailian-thinking-budget "$BAILIAN_THINKING_BUDGET")
  fi
  local command_string child_q log_dir_q
  command_string=$(printf '%q ' "${command[@]}")
  child_q=$(printf '%q' "$child")
  log_dir_q=$(printf '%q' "$child/logs")
  local wrap
  wrap="set +e; ${command_string}; status=\$?; mkdir -p ${log_dir_q}; cp -f ${shared_prefix}-\${SLURM_JOB_ID}.out ${log_dir_q}/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${shared_prefix}-\${SLURM_JOB_ID}.err ${log_dir_q}/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; QQ_SUB=\"CA 0722(${RUN_INDEX}) marked transition ${mode}\" QQ_BODY=\"status=\$status; arm=${mode}; output=${child}; node=\${SLURM_JOB_NODELIST}; job=\${SLURM_JOB_ID}\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/CA_0722_${RUN_INDEX}_context_${mode}.qq.log 2>&1; rm -f ${shared_prefix}-\${SLURM_JOB_ID}.out ${shared_prefix}-\${SLURM_JOB_ID}.err /tmp/CA_0722_${RUN_INDEX}_context_${mode}.qq.log; rmdir ${SHARED_LOG_ROOT} 2>/dev/null || true; exit \$status"
  sbatch --parsable --job-name="CA22${RUN_INDEX}${mode:0:1}" --nodelist="$NODE" \
    --gres=gpu:1 --cpus-per-task=8 --mem="$MEMORY" --time=01:00:00 \
    --output="${shared_prefix}-%j.out" --error="${shared_prefix}-%j.err" \
    --wrap="$wrap"
}

declare -A jobs
for mode in "${MODES[@]}"; do
  jobs["$mode"]=$(submit_arm "$mode")
done

{
  printf 'parent=%s\nnode=%s\nnetwork_mode=%s\n' \
    "$PARENT" "$NODE" "$BAILIAN_NETWORK_MODE"
  for mode in "${MODES[@]}"; do
    printf '%s_job=%s\n' "$mode" "${jobs[$mode]}"
  done
} | tee "$PARENT/submission.txt"
