#!/usr/bin/env bash
set -euo pipefail

# Submit the bounded Qwen3-VL-2B LoRA SFT smoke after environment setup and
# deterministic 32-example preprocessing have completed.  This launcher never
# creates the environment or dataset and never runs training on the management
# node.

ROOT="${REPO_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent}"
ROOT="$(realpath -e -- "$ROOT")"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/CA_0729(1)-qwen3vl-sft32-smoke}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/data}"
DATASET_NAME="${DATASET_NAME:-levir_qwen3vl_sft32_train}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-levir_qwen3vl_sft32_eval}"
TRAIN_ENV="${TRAIN_ENV:-/guisongxia01/pangchao/wangyihan/conda_envs/qwen3vl-sft-env}"
PYTHON="${PYTHON:-$TRAIN_ENV/bin/python}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent/third_party/LLaMA-Factory}"
LLAMA_FACTORY_COMMIT="${LLAMA_FACTORY_COMMIT:-9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-$TRAIN_ENV/bin/llamafactory-cli}"
MODEL_PATH="${MODEL_PATH:-/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/qwen3vl_sft_smoke_0729.yaml}"
SCORER="${SCORER:-$ROOT/tools/score_qwen3vl_sft_smoke.py}"
WORKER="${WORKER:-$ROOT/tools/run_qwen3vl_sft_smoke.slurm}"
DATA_BUILDER="${DATA_BUILDER:-$ROOT/tools/build_qwen3vl_sft_smoke.py}"
JOB_NAME="${JOB_NAME:-CA29Q3VLSFT32}"
NODE="${NODE:-}"
PARTITION="gre"

fail() {
  printf '%s\n' "$1" >&2
  exit "${2:-2}"
}

OUTPUTS_ROOT="$(realpath -e -- "$ROOT/outputs")"
RUN_DIR="$(realpath -m -- "$RUN_DIR")"
DATASET_DIR="$(realpath -m -- "$DATASET_DIR")"
if [[ "$RUN_DIR" == "$OUTPUTS_ROOT" || "$RUN_DIR" != "$OUTPUTS_ROOT/"* ]]; then
  fail "RUN_DIR must be a child of the repository outputs directory: $RUN_DIR"
fi
if [[ "$DATASET_DIR" != "$RUN_DIR/data" ]]; then
  fail "DATASET_DIR must be the data child of RUN_DIR: expected $RUN_DIR/data, found $DATASET_DIR"
fi

if [[ ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  fail "JOB_NAME contains unsupported characters: $JOB_NAME"
fi
if [[ -n "$NODE" && ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  fail "NODE contains unsupported characters: $NODE"
fi
if [[ ! "$LLAMA_FACTORY_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  fail "LLAMA_FACTORY_COMMIT must be a full 40-character lowercase Git commit"
fi

for required_file in \
  "$PYTHON" \
  "$LLAMAFACTORY_CLI" \
  "$WORKER" \
  "$DATA_BUILDER" \
  "$SCORER" \
  "$TRAIN_CONFIG" \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/model.safetensors" \
  "$MODEL_PATH/preprocessor_config.json" \
  "$DATASET_DIR/sharegpt.json" \
  "$DATASET_DIR/dataset_info.json" \
  "$DATASET_DIR/manifest.json"; do
  if [[ ! -f "$required_file" ]]; then
    fail "missing required file: $required_file; finish setup/preprocessing before submission"
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  fail "training Python is not executable: $PYTHON"
fi
if [[ ! -x "$LLAMAFACTORY_CLI" ]]; then
  fail "LLaMA-Factory CLI is not executable: $LLAMAFACTORY_CLI"
fi
if [[ ! -d "$DATASET_DIR/images" ]]; then
  fail "missing preprocessed image directory: $DATASET_DIR/images"
fi
if [[ ! -d "$LLAMA_FACTORY_ROOT/.git" ]]; then
  fail "missing pinned LLaMA-Factory checkout: $LLAMA_FACTORY_ROOT"
fi

if ! ACTUAL_LLAMA_FACTORY_COMMIT="$(git -C "$LLAMA_FACTORY_ROOT" rev-parse HEAD 2>/dev/null)"; then
  fail "cannot read LLaMA-Factory Git commit: $LLAMA_FACTORY_ROOT"
fi
if [[ "$ACTUAL_LLAMA_FACTORY_COMMIT" != "$LLAMA_FACTORY_COMMIT" ]]; then
  fail "LLaMA-Factory commit mismatch: expected $LLAMA_FACTORY_COMMIT, found $ACTUAL_LLAMA_FACTORY_COMMIT"
fi

# The run root may contain only data/, the successful prepare markers, and the
# prepare logs produced by the matching CPU job. Any training marker, result, or
# unrelated entry means this is not a fresh smoke submission.
if [[ ! -d "$RUN_DIR" ]]; then
  fail "run directory is missing: $RUN_DIR; preprocess the 32 examples into $DATASET_DIR first"
fi
for completed_path in \
  "$RUN_DIR/.training_submitted" \
  "$RUN_DIR/smoke_metrics.json" \
  "$RUN_DIR/completion_summary.json"; do
  if [[ -e "$completed_path" ]]; then
    fail "refusing duplicate or overwrite; run state already exists: $completed_path"
  fi
done
if [[ ! -f "$RUN_DIR/.prepare_complete" ]]; then
  fail "missing successful prepare marker: $RUN_DIR/.prepare_complete"
fi
while IFS= read -r existing_entry; do
  case "$existing_entry" in
    "$DATASET_DIR"|"$RUN_DIR/logs"|"$RUN_DIR/.prepare_submitted"|"$RUN_DIR/.prepare_complete") ;;
    *) fail "unexpected run entry before training submission: $existing_entry" ;;
  esac
done < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print)

# Re-run the authoritative deterministic validator.  It checks exactly 32
# records, balanced classes, unique source scenes, strict JSON targets, image
# existence, and one-to-one image placeholders without rebuilding any data.
if ! "$PYTHON" "$DATA_BUILDER" \
  --output-dir "$DATASET_DIR" \
  --validate-only >/dev/null; then
  fail "32-example dataset validation failed: $DATASET_DIR"
fi

if ! SLURM_PING="$(scontrol ping 2>&1)"; then
  printf '%s\n' "Slurm preflight failed; no job submitted" "$SLURM_PING" >&2
  exit 3
fi
if ! grep -q 'UP' <<<"$SLURM_PING"; then
  printf '%s\n' "Slurm controller is not UP; no job submitted" "$SLURM_PING" >&2
  exit 3
fi
if ! DUPLICATES="$(squeue -u guisongxia --noheader --name="$JOB_NAME" --format='%i|%j|%T|%R' 2>&1)"; then
  printf '%s\n' "failed to query duplicate jobs for Slurm user guisongxia" "$DUPLICATES" >&2
  exit 3
fi
if [[ -n "${DUPLICATES//[[:space:]]/}" ]]; then
  printf '%s\n' "refusing duplicate smoke submission; active matching job:" "$DUPLICATES" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs"
SUBMISSION_LOCK="$RUN_DIR/.submission_in_progress"
if ! mkdir "$SUBMISSION_LOCK"; then
  fail "another submission is already claiming this run: $SUBMISSION_LOCK"
fi
cleanup_submission_lock() {
  rmdir "$SUBMISSION_LOCK" 2>/dev/null || true
}
trap cleanup_submission_lock EXIT

NODE_ARGS=()
if [[ -n "$NODE" ]]; then
  NODE_ARGS+=(--nodelist="$NODE")
fi

export RUN_DIR DATASET_DIR DATASET_NAME EVAL_DATASET_NAME TRAIN_ENV PYTHON
export LLAMA_FACTORY_ROOT LLAMA_FACTORY_COMMIT LLAMAFACTORY_CLI
export MODEL_PATH TRAIN_CONFIG SCORER
REPO_ROOT="$ROOT"
export REPO_ROOT

if ! JOB_ID="$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  "${NODE_ARGS[@]}" \
  --output="$RUN_DIR/logs/slurm-%j.out" \
  --error="$RUN_DIR/logs/slurm-%j.err" \
  "$WORKER" "$RUN_DIR")"; then
  rmdir "$RUN_DIR/logs" 2>/dev/null || true
  echo "sbatch failed; no Qwen3-VL SFT smoke job was recorded" >&2
  exit 3
fi

printf '%s\n' \
  "job=$JOB_ID" \
  "job_name=$JOB_NAME" \
  "run_dir=$RUN_DIR" \
  "dataset=$DATASET_NAME" \
  "eval_dataset=$EVAL_DATASET_NAME" \
  "dataset_dir=$DATASET_DIR" \
  "partition=$PARTITION" \
  "node=${NODE:-slurm-selected}" \
  "resources=gpu:1,cpu:4,mem:32G,time:01:00:00" \
  >"$SUBMISSION_LOCK/training_submitted"
mv "$SUBMISSION_LOCK/training_submitted" "$RUN_DIR/.training_submitted"
rmdir "$SUBMISSION_LOCK"
trap - EXIT
cat "$RUN_DIR/.training_submitted"
