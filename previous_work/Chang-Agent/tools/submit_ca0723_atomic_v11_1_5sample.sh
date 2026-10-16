#!/usr/bin/env bash
set -euo pipefail

# Protocol v11.1: three controlled Verifier-planned controller arms over the same five samples.
# Diagnosis and action planning come from the configured Verifier; the
# Environment/Executor implementation is fixed across arms.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
CSV="${CSV:-$ROOT/bailian-apikey.csv}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
RUN_INDEX="${1:-3}"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
PARENT="${OUTPUT:-$ROOT/outputs/CA_0723(${RUN_INDEX})-bailian-proposal-atomic-v11.1-5sample}"
SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0723_${RUN_INDEX}_atomic_v11_1}"
SAMPLES=(test_20_15 test_78_13 test_85_16 test_1_1 test_50_8)
ARMS=(qwen37-thinking qwen37-no-thinking qwen3vl-plus)

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be set to the GPU node that passed the complete Bailian probe" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy based on the fixed-node probe" >&2
  exit 2
fi
if [[ ! "$RUN_INDEX" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX must be a non-negative integer" >&2
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
for sample in "${SAMPLES[@]}"; do
  for directory in A B label_cvt; do
    if [[ ! -f "$INPUT_ROOT/$directory/$sample.png" ]]; then
      echo "missing sample input: $directory/$sample.png" >&2
      exit 2
    fi
  done
done
if ! scontrol ping | grep -q UP; then
  echo "Slurm controller is not UP" >&2
  exit 3
fi

mkdir -p "$PARENT" "$SHARED_LOG_ROOT"
printf '%s\n' \
  '# CA_0723 atomic grounded verifier v11.1 five-sample ablation' \
  '' \
  "- node: \`$NODE\`" \
  "- network_mode: \`$BAILIAN_NETWORK_MODE\`" \
  '- samples: `test_20_15 test_78_13 test_85_16 test_1_1 test_50_8`' \
  '- architecture: `Verifier planner -> Environment -> deterministic Executor`' \
  '- arms: `qwen37-thinking`, `qwen37-no-thinking`, `qwen3vl-plus`' \
  '- protocol: `v11.1 = v11 atomic audit + empty-mask scan + rollback/status/security/network hardening`' \
  '- empty-mask fallback: `3x3 full-frame scan; Environment-owned tile box`' \
  '- Bailian transient retry: `2 retries; exponential backoff from 1s`' \
  '- max action steps: `3`; rejected candidates roll back and replan' \
  '- GT policy: `loaded only after each complete rollout`' \
  >"$PARENT/experiment_manifest.md"

NETWORK_PREFIX=()
if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy
    CHANGE_AGENT_STAGED_PROTOCOL_VERSION=v11.1
  )
else
  NETWORK_PREFIX=(env CHANGE_AGENT_STAGED_PROTOCOL_VERSION=v11.1)
fi

submit_arm() {
  local arm="$1"
  local verifier_model="$2"
  local enable_thinking="$3"
  local child="$PARENT/$arm"
  local shared_prefix="$SHARED_LOG_ROOT/$arm"
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
    --proposal-mode proposal
    --staged-verifier-max-total-regions 32
    --verifier-max-selected-regions 3
    --verifier-region-padding-ratio 0.25
    --empty-mask-scan-grid-size 3
    --matching-mode overlap_presence
    --overlap-threshold 0.25
    --verifier qwen_staged
    --staged-verifier-backend bailian
    --verifier-bailian-model "$verifier_model"
    --bailian-network-retries 2
    --bailian-network-backoff-seconds 1
    --verifier-max-new-tokens 1536
    --verifier-retries 2
    --verifier-repetition-penalty 1.05
    --tool-device cuda
    --seed 42
    --segagent-python /guisongxia01/pangchao/wangyihan/segagent-env/bin/python
    --omniovcd-python /guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python
    --simpleclick-checkpoint /guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth
    --sam3-checkpoint /guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt
    --sam3-bpe /guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz
    --sam3-resolution 1008
  )
  if [[ "$enable_thinking" == "1" ]]; then
    command+=(--bailian-enable-thinking --bailian-thinking-budget 256)
  fi

  local command_string child_logs_q
  command_string="$(printf '%q ' "${command[@]}")"
  child_logs_q="$(printf '%q' "$child/logs")"
  local wrap
  wrap="set +e; ${command_string}; status=\$?; mkdir -p ${child_logs_q}; cp -f ${shared_prefix}-\${SLURM_JOB_ID}.out ${child_logs_q}/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${shared_prefix}-\${SLURM_JOB_ID}.err ${child_logs_q}/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; QQ_SUB=\"CA 0723(${RUN_INDEX}) atomic v11.1 ${arm}\" QQ_BODY=\"status=\$status; samples=5; metrics=${child}/per_sample_metrics.json; output=${child}; node=\${SLURM_JOB_NODELIST}; job=\${SLURM_JOB_ID}\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/ca0723_${RUN_INDEX}_${arm}_\${SLURM_JOB_ID}.qq.log 2>&1; rm -f ${shared_prefix}-\${SLURM_JOB_ID}.out ${shared_prefix}-\${SLURM_JOB_ID}.err /tmp/ca0723_${RUN_INDEX}_${arm}_\${SLURM_JOB_ID}.qq.log; rmdir ${SHARED_LOG_ROOT} 2>/dev/null || true; exit \$status"

  sbatch --parsable --job-name="CA23V111-${arm}" --nodelist="$NODE" \
    --gres=gpu:1 --cpus-per-task=8 --mem=24G --time=03:00:00 \
    --output="${shared_prefix}-%j.out" --error="${shared_prefix}-%j.err" \
    --wrap="$wrap"
}

JOBS=()
JOBS+=("$(submit_arm qwen37-thinking qwen3.7-plus-2026-05-26 1)")
JOBS+=("$(submit_arm qwen37-no-thinking qwen3.7-plus-2026-05-26 0)")
JOBS+=("$(submit_arm qwen3vl-plus qwen3-vl-plus 0)")

for index in "${!ARMS[@]}"; do
  printf 'arm=%s job=%s output=%s/%s node=%s network_mode=%s\n' \
    "${ARMS[$index]}" "${JOBS[$index]}" "$PARENT" "${ARMS[$index]}" \
    "$NODE" "$BAILIAN_NETWORK_MODE"
done
