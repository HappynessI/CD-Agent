#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
RUN_INDEX="${1:-2}"
OUTPUT="$ROOT/outputs/CA_0720(${RUN_INDEX})-local-fix"

if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi

COMMAND=(
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  --input-root /guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256
  --samples test_85_16
  --output "$OUTPUT"
  --query building
  --max-steps 2
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
  --model-path /guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct
  --agent-backend local
  --verifier qwen_staged
  --staged-verifier-backend local
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
WRAP="set +e; ${COMMAND_STR}; status=\$?; QQ_SUB='CA 0720(${RUN_INDEX}) local staged fix' QQ_BODY=\"status=\$status; output=$OUTPUT\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/CA_0720_${RUN_INDEX}_local.qq.log 2>&1; rm -f /tmp/CA_0720_${RUN_INDEX}_local-\${SLURM_JOB_ID}.out /tmp/CA_0720_${RUN_INDEX}_local-\${SLURM_JOB_ID}.err; exit \$status"

JOB=$(sbatch --parsable --job-name=CA0720L2 --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=00:30:00 \
  --output="/tmp/CA_0720_${RUN_INDEX}_local-%j.out" \
  --error="/tmp/CA_0720_${RUN_INDEX}_local-%j.err" \
  --wrap="$WRAP")

printf 'job=%s\noutput=%s\n' "$JOB" "$OUTPUT"
