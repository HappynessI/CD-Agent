#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
CSV="$ROOT/bailian-apikey.csv"
RUN_INDEX="${1:-6}"
NODE="${NODE:-}"
BAILIAN_NETWORK_MODE="${BAILIAN_NETWORK_MODE:-}"
PARENT="$ROOT/experiments/CA_0721(${RUN_INDEX})-bailian-proposal-ablation"
SAMPLES=(test_20_15 test_78_13 test_85_16)
MODES=(direct proposal hybrid)

if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must be the node already passed by tools/probe_ca0721_bailian_node.sh" >&2
  exit 2
fi
if [[ "$BAILIAN_NETWORK_MODE" != "direct" && "$BAILIAN_NETWORK_MODE" != "proxy" ]]; then
  echo "BAILIAN_NETWORK_MODE must be direct or proxy" >&2
  exit 2
fi
if [[ -e "$PARENT" ]]; then
  echo "experiment parent already exists: $PARENT" >&2
  exit 2
fi

mkdir -p "$PARENT"
printf '%s\n' \
  '# BaiLian Proposal ablation' \
  '' \
  "- node: \`$NODE\`" \
  "- network_mode: \`$BAILIAN_NETWORK_MODE\`" \
  '- samples: `test_20_15 test_78_13 test_85_16`' \
  '- arms: `direct`, `proposal`, `hybrid`' \
  '- GT policy: `loaded only after each arm rollout completes`' \
  > "$PARENT/experiment_manifest.md"

NETWORK_PREFIX=()
if [[ "$BAILIAN_NETWORK_MODE" == "direct" ]]; then
  NETWORK_PREFIX=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy
  )
fi

submit_arm() {
  local mode="$1"
  local child="$PARENT/$mode"
  local tmp_prefix="/tmp/CA_0721_${RUN_INDEX}_proposal_${mode}"
  local command=(
    "${NETWORK_PREFIX[@]}"
    "$PYTHON" "$ROOT/tools/run_with_bailian_csv.py" "$CSV"
    "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
    --input-root /guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256
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
  local command_string child_q log_dir_q
  command_string=$(printf '%q ' "${command[@]}")
  child_q=$(printf '%q' "$child")
  log_dir_q=$(printf '%q' "$child/logs")
  local wrap
  wrap="set +e; ${command_string}; status=\$?; if [ -d ${child_q} ]; then mkdir -p ${log_dir_q}; cp -f ${tmp_prefix}-\${SLURM_JOB_ID}.out ${log_dir_q}/slurm-\${SLURM_JOB_ID}.out 2>/dev/null; cp -f ${tmp_prefix}-\${SLURM_JOB_ID}.err ${log_dir_q}/slurm-\${SLURM_JOB_ID}.err 2>/dev/null; fi; QQ_SUB=\"CA 0721(${RUN_INDEX}) proposal ${mode}\" QQ_BODY=\"status=\$status; arm=${mode}; output=${child}; node=\${SLURM_JOB_NODELIST}; job=\${SLURM_JOB_ID}\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/CA_0721_${RUN_INDEX}_proposal_${mode}.qq.log 2>&1; rm -f ${tmp_prefix}-\${SLURM_JOB_ID}.out ${tmp_prefix}-\${SLURM_JOB_ID}.err; exit \$status"
  sbatch --parsable --job-name="CA21${RUN_INDEX}${mode:0:1}" --nodelist="$NODE" \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
    --output="${tmp_prefix}-%j.out" --error="${tmp_prefix}-%j.err" --wrap="$wrap"
}

declare -A jobs
for mode in "${MODES[@]}"; do
  jobs["$mode"]=$(submit_arm "$mode")
done

printf 'parent=%s\nnode=%s\nnetwork_mode=%s\ndirect_job=%s\nproposal_job=%s\nhybrid_job=%s\n' \
  "$PARENT" "$NODE" "$BAILIAN_NETWORK_MODE" \
  "${jobs[direct]}" "${jobs[proposal]}" "${jobs[hybrid]}" \
  | tee "$PARENT/submission.txt"
