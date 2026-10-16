#!/usr/bin/env bash
set -euo pipefail

# Submit the CA_0730 live-trajectory validation after a complete fixed-node
# Bailian probe. This is one Qwen3.7 non-thinking proposal-v11.1 arm and tests
# the already fixed-action-validated SimpleClick default under autonomous Qwen
# planning; it is not the first validation of the input/write mechanism.
# Usage:
#   NODE=gpuXX BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0730_simpleclick_live_3sample.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
CSV="${CSV:-/guisongxia01/pangchao/wangyihan/wyh/bailian-apikey.csv}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
SEGAGENT_PYTHON="${SEGAGENT_PYTHON:-/guisongxia01/pangchao/wangyihan/segagent-env/bin/python}"
OMNIOVCD_PYTHON="${OMNIOVCD_PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
SIMPLECLICK_CHECKPOINT="${SIMPLECLICK_CHECKPOINT:-/guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt}"
SAM3_BPE="${SAM3_BPE:-/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
QQ_NOTIFIER="${QQ_NOTIFIER:-/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq}"

NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
PARENT="${OUTPUT:-$ROOT/outputs/CA_0730(1)-simpleclick-full-session-component-write-live-3sample}"
ARM="qwen37-no-thinking"
CHILD="$PARENT/$ARM"
JOB_NAME="${JOB_NAME:-CA30-SC-LIVE3}"
SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0730_1_simpleclick_live_3sample}"
SHARED_PREFIX="$SHARED_LOG_ROOT/$ARM"
SAMPLES=(test_20_15 test_78_13 test_85_16)

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be set to the GPU node that passed the complete Bailian probe" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy based on the fixed-node probe" >&2
  exit 2
fi
if [[ -z "$JOB_NAME" || ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "JOB_NAME contains unsupported characters" >&2
  exit 2
fi
if [[ -e "$PARENT" ]]; then
  echo "output already exists: $PARENT" >&2
  exit 2
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "missing input root: $INPUT_ROOT" >&2
  exit 2
fi
for executable in "$PYTHON" "$SEGAGENT_PYTHON" "$OMNIOVCD_PYTHON" "$QQ_NOTIFIER"; do
  if [[ ! -x "$executable" ]]; then
    echo "missing executable: $executable" >&2
    exit 2
  fi
done
for required_file in \
  "$CSV" \
  "$ROOT/tools/run_with_bailian_csv.py" \
  "$ROOT/tools/run_levir_change_agent.py" \
  "$SIMPLECLICK_CHECKPOINT" \
  "$SAM3_CHECKPOINT" \
  "$SAM3_BPE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing required file: $required_file" >&2
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

if ! SLURM_PING="$(scontrol ping 2>&1)"; then
  echo "Slurm preflight failed; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
if ! grep -q 'UP' <<<"$SLURM_PING"; then
  echo "Slurm controller is not UP; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
if ! DUPLICATE_JOBS="$(
  squeue -h -u guisongxia --name="$JOB_NAME" -o '%i %T' 2>&1
)"; then
  echo "Slurm duplicate-job preflight failed; no job submitted" >&2
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  exit 3
fi
if [[ -n "$DUPLICATE_JOBS" ]]; then
  echo "active job with the same name already exists; no job submitted" >&2
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  exit 2
fi

# mkdir without -p makes the output non-existence check atomic with respect to
# another invocation using the same parent directory.
mkdir "$PARENT"
mkdir -p "$SHARED_LOG_ROOT"
printf '%s\n' \
  '# CA_0730 SimpleClick new-default live-trajectory validation' \
  '' \
  "- node: \`$NODE\`" \
  "- network_mode: \`$BAILIAN_NETWORK_MODE\`" \
  '- samples: `test_20_15 test_78_13 test_85_16`' \
  '- arm: `qwen37-no-thinking` (`qwen3.7-plus-2026-05-26`, thinking disabled)' \
  '- protocol: `v11.1 proposal`' \
  '- SimpleClick model input: `full_session`' \
  '- negative-point write scope: `live clicked component within proposal ROI`' \
  '- point proposal gate: `enabled`' \
  '- point gate: `proposal_ratio > 0.25 AND changed_pixels > 256 AND image_ratio > 0.004`' \
  '- max action steps: `3`; rejected candidates roll back and replan' \
  '- GT policy: `loaded only after each complete rollout`' \
  '- purpose: `validate autonomous Qwen trajectory behavior of the fixed-action-validated SimpleClick default`' \
  '- acceptance: `selected micro IoU >= 0.7155976055; zero writes outside authorized component; no generalization claim`' \
  '- reference: `CA_0723(5) qwen37-no-thinking selected micro IoU 0.7228258641`' \
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

COMMAND=(
  "${NETWORK_PREFIX[@]}"
  "$PYTHON" "$ROOT/tools/run_with_bailian_csv.py" "$CSV"
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  --input-root "$INPUT_ROOT"
  --samples "${SAMPLES[@]}"
  --output "$CHILD"
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
  --authorized-negative-point-input-scope full_session
  --max-point-proposal-change-ratio 0.25
  --max-point-proposal-change-pixels 256
  --max-point-image-change-ratio 0.004
  --verifier qwen_staged
  --staged-verifier-backend bailian
  --verifier-bailian-model qwen3.7-plus-2026-05-26
  --bailian-network-retries 2
  --bailian-network-backoff-seconds 1
  --verifier-max-new-tokens 1536
  --verifier-retries 2
  --verifier-repetition-penalty 1.05
  --tool-device cuda
  --seed 42
  --segagent-python "$SEGAGENT_PYTHON"
  --omniovcd-python "$OMNIOVCD_PYTHON"
  --simpleclick-checkpoint "$SIMPLECLICK_CHECKPOINT"
  --sam3-checkpoint "$SAM3_CHECKPOINT"
  --sam3-bpe "$SAM3_BPE"
  --sam3-resolution 1008
)
# The point proposal gate is enabled by default. Deliberately do not append
# --disable-point-proposal-change-gate to this fixed validation arm.

COMMAND_STRING="$(printf '%q ' "${COMMAND[@]}")"
CHILD_LOGS_Q="$(printf '%q' "$CHILD/logs")"
QQ_NOTIFIER_Q="$(printf '%q' "$QQ_NOTIFIER")"
SHARED_PREFIX_Q="$(printf '%q' "$SHARED_PREFIX")"
SHARED_LOG_ROOT_Q="$(printf '%q' "$SHARED_LOG_ROOT")"
QQ_LOG_PREFIX_Q="$(printf '%q' /tmp/ca0730_1_simpleclick_live)"
WRAP="set +e; started_epoch=\$(date +%s); ${COMMAND_STRING}; exit_code=\$?; finished_epoch=\$(date +%s); elapsed_seconds=\$((finished_epoch-started_epoch)); if [ \"\$exit_code\" -eq 0 ]; then final_status=success; else final_status=failure; fi; QQ_SUB=\"CA 0730(1) SimpleClick live validation\" QQ_BODY=\"status=\${final_status}; exit_code=\${exit_code}; metrics=${CHILD}/per_sample_metrics.json; elapsed_seconds=\${elapsed_seconds}; output=${CHILD}; job=\${SLURM_JOB_ID}; node=\${SLURM_JOB_NODELIST}\" ${QQ_NOTIFIER_Q} >${QQ_LOG_PREFIX_Q}-\${SLURM_JOB_ID}.qq.log 2>&1; mkdir -p ${CHILD_LOGS_Q}; cp -f ${SHARED_PREFIX_Q}-\${SLURM_JOB_ID}.out ${CHILD_LOGS_Q}/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${SHARED_PREFIX_Q}-\${SLURM_JOB_ID}.err ${CHILD_LOGS_Q}/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; rm -f ${SHARED_PREFIX_Q}-\${SLURM_JOB_ID}.out ${SHARED_PREFIX_Q}-\${SLURM_JOB_ID}.err ${QQ_LOG_PREFIX_Q}-\${SLURM_JOB_ID}.qq.log; rmdir ${SHARED_LOG_ROOT_Q} 2>/dev/null || true; exit \$exit_code"

JOB="$(
  sbatch --parsable --job-name="$JOB_NAME" --nodelist="$NODE" \
    --gres=gpu:1 --cpus-per-task=8 --mem=24G --time=03:00:00 \
    --output="${SHARED_PREFIX}-%j.out" --error="${SHARED_PREFIX}-%j.err" \
    --wrap="$WRAP"
)"

printf 'job=%s\nparent=%s\noutput=%s\nsamples=%s\nnode=%s\nnetwork_mode=%s\n' \
  "$JOB" "$PARENT" "$CHILD" "${SAMPLES[*]}" "$NODE" "$BAILIAN_NETWORK_MODE"
