#!/usr/bin/env bash
set -euo pipefail

# Management-node launcher for the CPU-only environment/data preparation job.
# It does no package installation, image processing, or test execution itself.
ROOT="${REPO_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent}"
ROOT="$(realpath -e -- "$ROOT")"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/CA_0729(1)-qwen3vl-sft32-smoke}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE/datasets/LEVIR-CD}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/data}"
BASE_ENV="${BASE_ENV:-$WORKSPACE/conda_envs/omniovcd-env}"
TRAIN_ENV="${TRAIN_ENV:-$WORKSPACE/conda_envs/qwen3vl-sft-env}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-$WORKSPACE/Chang-Agent/third_party/LLaMA-Factory}"
GIT_WRAPPER="${GIT_WRAPPER:-$WORKSPACE/.local/bin/wangyihan-git}"
SETUP_SCRIPT="${SETUP_SCRIPT:-$ROOT/tools/setup_qwen3vl_sft_env.sh}"
DATA_BUILDER="${DATA_BUILDER:-$ROOT/tools/build_qwen3vl_sft_smoke.py}"
DATA_TEST="${DATA_TEST:-$ROOT/tests/test_qwen3vl_sft_smoke_data.py}"
SCORER="${SCORER:-$ROOT/tools/score_qwen3vl_sft_smoke.py}"
SCORE_TEST="${SCORE_TEST:-$ROOT/tests/test_qwen3vl_sft_smoke_score.py}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
WORKER="${WORKER:-$ROOT/tools/run_qwen3vl_sft_prepare.slurm}"
JOB_NAME="${JOB_NAME:-CA29Q3VLPREP32}"
PARTITION="gre"
NODE="${NODE:-}"

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
if [[ ! "$PARTITION" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  fail "PARTITION contains unsupported characters: $PARTITION"
fi
if [[ -n "$NODE" && ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  fail "NODE contains unsupported characters: $NODE"
fi

for required_file in \
  "$WORKER" \
  "$SETUP_SCRIPT" \
  "$DATA_BUILDER" \
  "$DATA_TEST" \
  "$SCORER" \
  "$SCORE_TEST" \
  "$MODEL_PATH/config.json" \
  "$GIT_WRAPPER" \
  "$QQ_NOTIFIER"; do
  [[ -f "$required_file" ]] || fail "missing required file: $required_file"
done
[[ -d "$BASE_ENV" ]] || fail "base environment is missing: $BASE_ENV"
[[ -d "$DATASET_ROOT/train_original/A" ]] || fail "LEVIR-CD train A directory is missing"
[[ -d "$DATASET_ROOT/train_original/B" ]] || fail "LEVIR-CD train B directory is missing"
[[ -d "$DATASET_ROOT/train_original/label" ]] || fail "LEVIR-CD train label directory is missing"
[[ -d "$LLAMA_FACTORY_ROOT/.git" ]] || fail "pinned LLaMA-Factory checkout is missing"
[[ -x "$GIT_WRAPPER" ]] || fail "Git wrapper is not executable: $GIT_WRAPPER"
[[ -x "$QQ_NOTIFIER" ]] || fail "QQ notifier is not executable: $QQ_NOTIFIER"

# Refuse ambiguous continuation states. Failed logs and the previous submission
# receipt may be retained for audit; retry is allowed only when no
# data/completion/training marker exists. A same-name active job is rejected by
# the Slurm duplicate check below before the receipt is replaced.
if [[ -e "$DATASET_DIR" ]]; then
  fail "dataset directory already exists; refusing to overwrite: $DATASET_DIR"
fi
if [[ -e "$RUN_DIR/.prepare_complete" ]]; then
  fail "prepare completion marker already exists: $RUN_DIR/.prepare_complete"
fi
if [[ -e "$RUN_DIR/.training_submitted" ]]; then
  fail "training was already submitted for this run: $RUN_DIR/.training_submitted"
fi
if [[ -e "$RUN_DIR" && ! -d "$RUN_DIR" ]]; then
  fail "RUN_DIR exists but is not a directory: $RUN_DIR"
fi
if [[ -d "$RUN_DIR" ]]; then
  while IFS= read -r existing_entry; do
    case "$existing_entry" in
      "$RUN_DIR/logs"|"$RUN_DIR/.prepare_submitted") ;;
      *) fail "RUN_DIR contains an unexpected entry before prepare retry: $existing_entry" ;;
    esac
  done < <(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print)
  if [[ -d "$RUN_DIR/logs" ]]; then
    while IFS= read -r existing_log; do
      case "$existing_log" in
        "$RUN_DIR/logs/prepare-slurm-"*.out|\
        "$RUN_DIR/logs/prepare-slurm-"*.err|\
        "$RUN_DIR/logs/prepare-status-"*.txt|\
        "$RUN_DIR/logs/prepare-submitted-"*.txt) ;;
        *) fail "unexpected existing prepare log before retry: $existing_log" ;;
      esac
    done < <(find "$RUN_DIR/logs" -mindepth 1 -maxdepth 1 -print)
  fi
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
  printf '%s\n' "refusing duplicate prepare submission; active matching job:" "$DUPLICATES" >&2
  exit 2
fi

mkdir -p -- "$RUN_DIR/logs"
SUBMISSION_LOCK="$RUN_DIR/.prepare_submission_in_progress"
if ! mkdir "$SUBMISSION_LOCK"; then
  fail "another prepare submission is already claiming this run: $SUBMISSION_LOCK"
fi
cleanup_submission_lock() {
  rmdir "$SUBMISSION_LOCK" 2>/dev/null || true
}
trap cleanup_submission_lock EXIT

NODE_ARGS=()
if [[ -n "$NODE" ]]; then
  NODE_ARGS+=(--nodelist="$NODE")
fi

export RUN_DIR DATASET_ROOT DATASET_DIR BASE_ENV TRAIN_ENV MODEL_PATH
export LLAMA_FACTORY_ROOT GIT_WRAPPER SETUP_SCRIPT DATA_BUILDER DATA_TEST
export SCORER SCORE_TEST
export QQ_NOTIFIER
REPO_ROOT="$ROOT"
export REPO_ROOT

if ! JOB_ID="$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --cpus-per-task=2 \
  --mem=16G \
  --time=01:00:00 \
  "${NODE_ARGS[@]}" \
  --output="$RUN_DIR/logs/prepare-slurm-%j.out" \
  --error="$RUN_DIR/logs/prepare-slurm-%j.err" \
  "$WORKER" "$RUN_DIR")"; then
  echo "sbatch failed; no Qwen3-VL SFT prepare job was recorded" >&2
  exit 3
fi

printf '%s\n' \
  "job=$JOB_ID" \
  "job_name=$JOB_NAME" \
  "run_dir=$RUN_DIR" \
  "dataset_dir=$DATASET_DIR" \
  "partition=$PARTITION" \
  "node=${NODE:-slurm-selected}" \
  "resources=gpu:0,cpu:2,mem:16G,time:01:00:00" \
  >"$SUBMISSION_LOCK/prepare_submitted"
mv "$SUBMISSION_LOCK/prepare_submitted" "$RUN_DIR/.prepare_submitted"
rmdir "$SUBMISSION_LOCK"
trap - EXIT
cat "$RUN_DIR/.prepare_submitted"
