#!/usr/bin/env bash
#SBATCH --job-name=CA0801V121CURATE
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00

set -euo pipefail

ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}")"
OUTPUT="$(realpath -m -- "${1:?usage: run_qwen3vl_v121_curation.slurm OUTPUT}")"
OUTPUT_ROOT="$(realpath -e -- "${OUTPUT_ROOT:?OUTPUT_ROOT is required}")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE/datasets/LEVIR-CD}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$WORKSPACE/models/sam3/sam3.pt}"
SAM3_BPE="${SAM3_BPE:-$WORKSPACE/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""

die() { ERROR_TEXT="$*"; echo "v12.1 curation: $*" >&2; return 1; }
record_error() { local rc="$?"; [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"; return "$rc"; }
finish() {
  local rc="$?"
  trap - ERR EXIT
  set +e
  [[ "$rc" -eq 0 ]] && STATUS="success"
  local elapsed="$((SECONDS - START_SECONDS))"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\nkey_result=%s\nerror=%s\noutput=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$KEY_RESULT" "$ERROR_TEXT" "$OUTPUT" \
    >"$OUTPUT/logs/status-${JOB_ID}.txt"
  QQ_SUB="Qwen3-VL v12.1 curation candidates" \
  QQ_BODY="status=${STATUS}; result=${KEY_RESULT}; error=${ERROR_TEXT:-none}; job=${JOB_ID}; elapsed=${elapsed}s; output=${OUTPUT}" \
    "$QQ_NOTIFIER" >"/tmp/qwen3vl_v121_curation_${JOB_ID}.qq.log" 2>&1 || true
  rm -f -- "/tmp/qwen3vl_v121_curation_${JOB_ID}.qq.log"
  exit "$rc"
}
trap record_error ERR
trap finish EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run only through Slurm"
[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || die "output escapes declared output root"
[[ -d "$OUTPUT/logs" ]] || die "submitter did not reserve output/logs"
for path in "$PYTHON" "$ROOT/tools/prepare_qwen3vl_v121_curation.py" "$SAM3_CHECKPOINT" "$SAM3_BPE"; do
  [[ -e "$path" ]] || die "missing required path: $path"
done
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU lacks BF16 support")
print(f"gpu_contract=ok device={torch.cuda.get_device_name(0)}")
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:$WORKSPACE/Chang-Agent/third_party/OmniOVCD:$WORKSPACE/Chang-Agent/third_party/SegAgent:$WORKSPACE/Chang-Agent/third_party/SegAgent/third_party/SimpleClick"
"$PYTHON" "$ROOT/tools/prepare_qwen3vl_v121_curation.py" \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUTPUT/curation" \
  --sam3-checkpoint "$SAM3_CHECKPOINT" \
  --sam3-bpe "$SAM3_BPE" \
  --seed 801
[[ -f "$OUTPUT/curation/.curation_complete" ]] || die "curation completion marker missing"
KEY_RESULT="samples=$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["samples"]))' "$OUTPUT/curation/curation_manifest.json")"
STATUS="success"
