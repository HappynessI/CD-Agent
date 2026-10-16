#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl-sft32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

# A single-GPU smoke worker for the pinned LLaMA-Factory/Qwen3-VL path. The
# submitter may export all paths below; defaults make direct `sbatch` use
# predictable while still refusing missing installations.
ROOT="${REPO_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent}"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${1:-$ROOT/outputs/CA_0729(1)-qwen3vl-sft32-smoke}"
TRAIN_ENV="${TRAIN_ENV:-$WORKSPACE/conda_envs/qwen3vl-sft-env}"
PYTHON="${PYTHON:-$TRAIN_ENV/bin/python}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/qwen3vl_sft_smoke_0729.yaml}"
SCORER="${SCORER:-$ROOT/tools/score_qwen3vl_sft_smoke.py}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-$WORKSPACE/Chang-Agent/third_party/LLaMA-Factory}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/data}"
DATASET_NAME="${DATASET_NAME:-levir_qwen3vl_sft32_train}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-levir_qwen3vl_sft32_eval}"
COMMIT="${LLAMA_FACTORY_COMMIT:-9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
LOCAL_RUN_DIR=""
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""

mkdir -p "$RUN_DIR/logs"

die() {
  ERROR_TEXT="$*"
  echo "qwen3vl_sft_smoke: $*" >&2
  return 1
}

write_status() {
  local rc="$1"
  local elapsed="$((SECONDS - START_SECONDS))"
  local status_path="$RUN_DIR/logs/status-${JOB_ID}.txt"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\nrun_dir=%s\nkey_result=%s\nerror=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$RUN_DIR" "$KEY_RESULT" "$ERROR_TEXT" >"$status_path"
}

record_error() {
  local rc="$?"
  if [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

finish() {
  local rc="$?"
  trap - ERR EXIT
  set +e
  if [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  else
    STATUS="failed"
    if [[ -z "$ERROR_TEXT" ]]; then
      ERROR_TEXT="pipeline exited with code $rc; inspect shared Slurm/LLaMA-Factory logs"
    fi
  fi
  write_status "$rc"
  if [[ -n "$LOCAL_RUN_DIR" && -d "$LOCAL_RUN_DIR" ]]; then
    # The adapter and all intermediate checkpoints are intentionally never
    # copied to the shared workspace for a smoke test.
    rm -rf -- "$LOCAL_RUN_DIR"
  fi
  local elapsed="$((SECONDS - START_SECONDS))"
  QQ_SUB="Qwen3-VL-2B SFT smoke 0729" \
  QQ_BODY="status=${STATUS}; key_result=${KEY_RESULT}; error=${ERROR_TEXT:-none}; job=${JOB_ID}; elapsed=${elapsed}s; output=${RUN_DIR}" \
    "$WORKSPACE/.local/bin/wangyihan-send-qq" \
    >"/tmp/qwen3vl_sft_${JOB_ID}.qq.log" 2>&1 || true
  rm -f -- "/tmp/qwen3vl_sft_${JOB_ID}.qq.log"
  exit "$rc"
}
trap record_error ERR
trap finish EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run this worker only through a Slurm GPU allocation"
[[ -d "$ROOT/outputs" ]] || die "repository root is invalid: $ROOT"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid pinned LLaMA-Factory commit: $COMMIT"
[[ -x "$PYTHON" ]] || die "training interpreter is missing: $PYTHON"
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || die "model is missing: $MODEL_PATH"
[[ -f "$TRAIN_CONFIG" ]] || die "training config is missing: $TRAIN_CONFIG"
[[ -f "$SCORER" ]] || die "smoke scorer is missing: $SCORER"
[[ -f "$DATASET_DIR/dataset_info.json" ]] || die "dataset_info.json is missing: $DATASET_DIR"
[[ -f "$DATASET_DIR/sharegpt.json" ]] || die "sharegpt.json is missing: $DATASET_DIR"
[[ -d "$LLAMA_FACTORY_ROOT/.git" ]] || die "LLaMA-Factory checkout is missing: $LLAMA_FACTORY_ROOT"
[[ "$($WORKSPACE/.local/bin/wangyihan-git -C "$LLAMA_FACTORY_ROOT" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "LLaMA-Factory checkout is not pinned to $COMMIT"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is empty; request one GPU through Slurm"

export PYTHONPATH="$LLAMA_FACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# This worker has already enforced exactly one visible GPU. Keep LLaMA-Factory
# on its native single-process path so the pinned overlay interpreter remains
# authoritative. In particular, do not inherit a caller's distributed-launch
# variables: FORCE_TORCHRUN would make the launcher search PATH for a separate
# torchrun entry point, which may belong to the base environment instead of
# TRAIN_ENV.
unset FORCE_TORCHRUN NPROC_PER_NODE NNODES NODE_RANK LOCAL_RANK RANK WORLD_SIZE
unset MASTER_ADDR MASTER_PORT

# Verify the dataset contract before loading the 4GB model. This catches
# malformed ShareGPT roles, missing images and image-token mismatches early.
"$PYTHON" - "$DATASET_DIR" "$DATASET_NAME" "$EVAL_DATASET_NAME" <<'PY'
import json
import pathlib
import sys

data_dir = pathlib.Path(sys.argv[1])
train_name = sys.argv[2]
eval_name = sys.argv[3]
info = json.loads((data_dir / "dataset_info.json").read_text())
for name in (train_name, eval_name):
    if name not in info:
        raise SystemExit(f"dataset entry {name!r} is missing")
attr = info[train_name]
eval_attr = info[eval_name]
if attr.get("file_name") != eval_attr.get("file_name"):
    raise SystemExit("train/eval smoke entries must point at the same 32-row file")
file_name = attr.get("file_name", "sharegpt.json")
data_path = data_dir / file_name
if not data_path.is_file():
    raise SystemExit(f"dataset file is missing: {data_path}")
rows = json.loads(data_path.read_text())
if not isinstance(rows, list):
    raise SystemExit("ShareGPT smoke data must be a JSON list")
if len(rows) != 32:
    raise SystemExit(f"expected exactly 32 rows, found {len(rows)}")
for idx, row in enumerate(rows):
    columns = attr.get("columns", {})
    messages = row.get(columns.get("messages", "conversations"), row.get("conversations"))
    images = row.get(columns.get("images", "images"), row.get("images", []))
    if not isinstance(messages, list) or not messages:
        raise SystemExit(f"row {idx}: missing messages")
    if not isinstance(images, list) or not images:
        raise SystemExit(f"row {idx}: missing images")
    roles = [str(m.get("from", m.get("role", ""))).lower() for m in messages]
    if roles[-1] not in {"gpt", "assistant"}:
        raise SystemExit(f"row {idx}: final message is not assistant: {roles[-1]!r}")
    placeholder_count = sum(str(m.get("value", m.get("content", ""))).count("<image>") for m in messages)
    if placeholder_count != len(images):
        raise SystemExit(f"row {idx}: {placeholder_count} image tokens for {len(images)} images")
    for image in images:
        path = pathlib.Path(str(image))
        if not path.is_absolute():
            path = data_dir / path
        if not path.is_file():
            raise SystemExit(f"row {idx}: image is missing: {path}")
print(f"dataset_contract=ok rows={len(rows)}")
PY

# Lightweight package/model check. The setup script records the exact pins;
# this worker repeats the check so a stale compute-node environment fails fast.
"$PYTHON" - "$LLAMA_FACTORY_ROOT" "$MODEL_PATH" <<'PY'
import importlib.metadata as md
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model = pathlib.Path(sys.argv[2])
expected = {
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "transformers": "4.57.6",
    "datasets": "4.0.0",
    "accelerate": "1.11.0",
    "peft": "0.18.1",
    "trl": "0.24.0",
    "torchdata": "0.11.0",
    "qwen-vl-utils": "0.0.14",
}
required_present = (
    "torchaudio",
    "matplotlib",
    "fire",
    "tyro",
    "omegaconf",
    "sentencepiece",
    "tiktoken",
    "jieba",
    "nltk",
    "rouge-chinese",
)
bad = []
for name, want in expected.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "MISSING"
    print(f"{name}={got}")
    if got != want:
        bad.append(f"{name}={got} expected={want}")
for name in required_present:
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = "MISSING"
    print(f"{name}={got}")
    if got == "MISSING":
        bad.append(f"{name}=MISSING")
config = json.loads((model / "config.json").read_text())
if config.get("model_type") != "qwen3_vl":
    bad.append(f"model_type={config.get('model_type')!r}")
if not (root / "src" / "llamafactory" / "cli.py").is_file():
    bad.append(f"missing LLaMA-Factory source under {root}")
if bad:
    raise SystemExit("dependency/model check failed: " + "; ".join(bad))
PY

"$PYTHON" - <<'PY'
import llamafactory.data.mm_plugin  # noqa: F401
import llamafactory.train.sft.workflow  # noqa: F401
print("llamafactory_import=ok")
PY

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU does not report BF16 support")
print(f"gpu_contract=ok device={torch.cuda.get_device_name(0)}")
PY

TMP_BASE="${SLURM_TMPDIR:-/tmp}"
[[ -d "$TMP_BASE" && -w "$TMP_BASE" ]] || TMP_BASE="/tmp"
LOCAL_RUN_DIR="$(mktemp -d "$TMP_BASE/qwen3vl_sft_${JOB_ID}.XXXXXX")"
LOCAL_OUTPUT="$LOCAL_RUN_DIR/adapter"
mkdir -p "$LOCAL_OUTPUT"

cp -- "$TRAIN_CONFIG" "$RUN_DIR/config_used.yaml"
printf '%s\n' "$COMMIT" >"$RUN_DIR/llamafactory_commit.txt"
"$PYTHON" - <<'PY' >"$RUN_DIR/environment_versions.txt"
import importlib.metadata as md
import platform
import torch
import transformers
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
print(f"transformers={transformers.__version__}")
for name in (
    "numpy",
    "scipy",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "torchdata",
    "qwen-vl-utils",
    "sentencepiece",
    "tiktoken",
    "jieba",
    "nltk",
    "rouge-chinese",
):
    print(f"{name}={md.version(name)}")
PY

CMD=(
  "$PYTHON" -m llamafactory.cli train "$TRAIN_CONFIG"
  "output_dir=$LOCAL_OUTPUT"
  "model_name_or_path=$MODEL_PATH"
  "dataset_dir=$DATASET_DIR"
  "media_dir=$DATASET_DIR"
  "dataset=$DATASET_NAME"
  "eval_dataset=$EVAL_DATASET_NAME"
  "max_samples=32"
)
printf 'launch=%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/logs/llamafactory-${JOB_ID}.log"

[[ -f "$LOCAL_OUTPUT/adapter_config.json" ]] || die "adapter_config.json was not produced"
[[ -f "$LOCAL_OUTPUT/adapter_model.safetensors" ]] || die "adapter_model.safetensors was not produced"

for file in train_results.json eval_results.json predict_results.json generated_predictions.jsonl trainer_state.json all_results.json; do
  if [[ -f "$LOCAL_OUTPUT/$file" ]]; then
    cp -- "$LOCAL_OUTPUT/$file" "$RUN_DIR/$file"
  fi
done

set +e
KEY_RESULT="$("$PYTHON" "$SCORER" \
  --generated-predictions "$LOCAL_OUTPUT/generated_predictions.jsonl" \
  --train-metrics "$LOCAL_OUTPUT/train_results.json" \
  --eval-metrics "$LOCAL_OUTPUT/eval_results.json" \
  --predict-metrics "$LOCAL_OUTPUT/predict_results.json" \
  --smoke-metrics-output "$RUN_DIR/smoke_metrics.json" \
  --smoke-manifest-output "$RUN_DIR/smoke_manifest.json" \
  --completion-summary-output "$RUN_DIR/completion_summary.json" \
  --artifact-dir "$RUN_DIR" \
  --local-output-dir "$LOCAL_OUTPUT" \
  --dataset "$DATASET_NAME" \
  --eval-dataset "$EVAL_DATASET_NAME" \
  --job-id "$JOB_ID" \
  --model-path "$MODEL_PATH" \
  --train-config "$TRAIN_CONFIG" \
  --llamafactory-commit "$COMMIT")"
SCORER_RC="$?"
set -e

if [[ -z "$KEY_RESULT" ]]; then
  KEY_RESULT="pipeline=failed;acceptance=not_evaluated;scorer_exit=${SCORER_RC}"
fi
echo "$KEY_RESULT"
if [[ "$SCORER_RC" -ne 0 ]]; then
  ERROR_TEXT="smoke scorer failed with exit code $SCORER_RC; retained metrics describe the failure"
  exit "$SCORER_RC"
fi
STATUS="success"
