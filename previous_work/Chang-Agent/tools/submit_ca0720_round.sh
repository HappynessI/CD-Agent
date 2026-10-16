#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-fresh}"
PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
INPUT_ROOT="/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256"
SEG_PYTHON="/guisongxia01/pangchao/wangyihan/segagent-env/bin/python"
OMNIOVCD_PYTHON="/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python"
API_CSV="$ROOT/bailian-apikey.csv"
ARCHIVE="$ROOT/outputs_archive_0720_before_CA"

if [[ "$MODE" == "fresh" ]]; then
  if [[ -e "$ARCHIVE" ]]; then
    echo "archive already exists: $ARCHIVE" >&2
    exit 2
  fi
  if [[ ! -d "$ROOT/outputs" ]]; then
    echo "outputs directory is missing: $ROOT/outputs" >&2
    exit 2
  fi
  mv "$ROOT/outputs" "$ARCHIVE"
  mkdir -p "$ROOT/outputs"
elif [[ "$MODE" == "retry-bailian" ]]; then
  if [[ ! -d "$ARCHIVE" || ! -d "$ROOT/outputs" ]]; then
    echo "retry requires the existing archive and outputs directories" >&2
    exit 2
  fi
else
  echo "usage: $0 [fresh|retry-bailian]" >&2
  exit 2
fi

COMMON=(
  --input-root "$INPUT_ROOT"
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
  --max-new-tokens 128
  --verifier-max-new-tokens 512
  --verifier-accept-threshold 0.82
  --verifier-retries 2
  --verifier-repetition-penalty 1.05
  --device-map auto
  --tool-device cuda
  --seed 42
  --action-retries 3
  --segagent-python "$SEG_PYTHON"
  --omniovcd-python "$OMNIOVCD_PYTHON"
  --simpleclick-checkpoint /guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth
  --sam3-checkpoint /guisongxia01/pangchao/wangyihan/models/sam3/sam3.pt
  --sam3-bpe /guisongxia01/pangchao/wangyihan/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz
  --sam3-resolution 1008
)

LOCAL_OUT="$ROOT/outputs/CA_0720(1)-local"
BAILIAN_OUT="$ROOT/outputs/CA_0720(1)-bailian"

LOCAL_CMD=(
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  "${COMMON[@]}"
  --output "$LOCAL_OUT"
  --model-path /guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct
  --agent-backend local
  --verifier qwen_staged
  --staged-verifier-backend local
)

BAILIAN_CMD=(
  "$PYTHON" "$ROOT/tools/run_with_bailian_csv.py" "$API_CSV"
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  "${COMMON[@]}"
  --output "$BAILIAN_OUT"
  --agent-backend bailian
  --verifier qwen_staged
  --staged-verifier-backend bailian
  --bailian-model qwen3-vl-plus
  --bailian-api-key-env DASHSCOPE_API_KEY
)

LOCAL_CMD_STR=$(printf '%q ' "${LOCAL_CMD[@]}")
BAILIAN_CMD_STR=$(printf '%q ' "${BAILIAN_CMD[@]}")
LOCAL_WRAP="set +e; ${LOCAL_CMD_STR}; status=\$?; QQ_SUB='CA 0720(1) local staged' QQ_BODY=\"status=\$status; output=$LOCAL_OUT\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/CA_0720_1_local.qq.log 2>&1; rm -f /tmp/CA_0720_1_local-\${SLURM_JOB_ID}.out /tmp/CA_0720_1_local-\${SLURM_JOB_ID}.err; exit \$status"
BAILIAN_WRAP="set +e; ${BAILIAN_CMD_STR}; status=\$?; QQ_SUB='CA 0720(1) Bailian staged' QQ_BODY=\"status=\$status; output=$BAILIAN_OUT\" /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq >/tmp/CA_0720_1_bailian.qq.log 2>&1; rm -f /tmp/CA_0720_1_bailian-\${SLURM_JOB_ID}.out /tmp/CA_0720_1_bailian-\${SLURM_JOB_ID}.err; exit \$status"

if [[ "$MODE" == "retry-bailian" ]]; then
  LOCAL_JOB="skipped"
else
  LOCAL_JOB=$(sbatch --parsable --job-name=CA0720L --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
    --output="/tmp/CA_0720_1_local-%j.out" --error="/tmp/CA_0720_1_local-%j.err" --wrap="$LOCAL_WRAP")
fi
if [[ "$MODE" == "retry-bailian" && -e "$BAILIAN_OUT" ]]; then
  echo "bailian output already exists: $BAILIAN_OUT" >&2
  exit 2
fi
BAILIAN_JOB=$(sbatch --parsable --job-name=CA0720B --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
  --output="/tmp/CA_0720_1_bailian-%j.out" --error="/tmp/CA_0720_1_bailian-%j.err" --wrap="$BAILIAN_WRAP")

printf 'archive=%s\nlocal_job=%s\nlocal_output=%s\nbailian_job=%s\nbailian_output=%s\n' \
  "$ARCHIVE" "$LOCAL_JOB" "$LOCAL_OUT" "$BAILIAN_JOB" "$BAILIAN_OUT"
