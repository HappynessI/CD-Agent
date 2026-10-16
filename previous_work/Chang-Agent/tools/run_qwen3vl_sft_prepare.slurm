#!/usr/bin/env bash
#SBATCH --job-name=CA29Q3VLPREP32
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

# CPU-only worker for the one-time Qwen3-VL SFT environment setup and the
# deterministic LEVIR-CD 32-example preprocessing gate. Submit it through
# submit_qwen3vl_sft_prepare.sh so stdout/stderr land in RUN_DIR/logs.
ROOT="${REPO_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent}"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${1:-$ROOT/outputs/CA_0729(1)-qwen3vl-sft32-smoke}"
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
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
LOG_DIR=""
STATUS="failed"
KEY_RESULT="preflight_not_started"
ERROR_TEXT="none"

die() {
  ERROR_TEXT="$*"
  printf 'qwen3vl_sft_prepare: %s\n' "$*" >&2
  return 1
}

write_status() {
  local rc="$1"
  local elapsed="$((SECONDS - START_SECONDS))"
  local temporary_status
  [[ -n "$LOG_DIR" && -d "$LOG_DIR" ]] || return 0
  temporary_status="$LOG_DIR/.prepare-status-${JOB_ID}.tmp"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\nrun_dir=%s\nkey_result=%s\nerror=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$RUN_DIR" "$KEY_RESULT" "$ERROR_TEXT" \
    >"$temporary_status"
  mv -- "$temporary_status" "$LOG_DIR/prepare-status-${JOB_ID}.txt"
}

record_error() {
  local rc="$?"
  if [[ "$ERROR_TEXT" == "none" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

finish() {
  local rc="$?"
  local elapsed="$((SECONDS - START_SECONDS))"
  local qq_log="/tmp/qwen3vl_sft_prepare_${JOB_ID}.qq.log"
  trap - ERR EXIT
  set +e
  if [[ "$rc" -eq 0 ]]; then
    STATUS="success"
    ERROR_TEXT="none"
  fi
  write_status "$rc"
  if [[ -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="Qwen3-VL-2B SFT32 prepare 0729" \
    QQ_BODY="status=${STATUS}; key_result=${KEY_RESULT}; error=${ERROR_TEXT}; job=${JOB_ID}; elapsed=${elapsed}s; output=${RUN_DIR}" \
      "$QQ_NOTIFIER" >"$qq_log" 2>&1 || true
    rm -f -- "$qq_log"
  fi
  exit "$rc"
}
trap record_error ERR
trap finish EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this worker must run inside a Slurm allocation"
[[ -d "$ROOT/outputs" ]] || die "repository root is invalid: $ROOT"

OUTPUTS_ROOT="$(realpath -e -- "$ROOT/outputs")"
RUN_DIR="$(realpath -m -- "$RUN_DIR")"
DATASET_DIR="$(realpath -m -- "$DATASET_DIR")"
if [[ "$RUN_DIR" == "$OUTPUTS_ROOT" || "$RUN_DIR" != "$OUTPUTS_ROOT/"* ]]; then
  die "RUN_DIR must be a child of the repository outputs directory: $RUN_DIR"
fi
if [[ "$DATASET_DIR" != "$RUN_DIR/data" ]]; then
  die "DATASET_DIR must be the data child of RUN_DIR: $DATASET_DIR"
fi

mkdir -p -- "$RUN_DIR/logs"
LOG_DIR="$RUN_DIR/logs"
KEY_RESULT="preflight"

for required_file in \
  "$SETUP_SCRIPT" \
  "$DATA_BUILDER" \
  "$DATA_TEST" \
  "$SCORER" \
  "$SCORE_TEST" \
  "$MODEL_PATH/config.json" \
  "$GIT_WRAPPER"; do
  [[ -f "$required_file" ]] || die "missing required file: $required_file"
done
[[ -x "$GIT_WRAPPER" ]] || die "Git wrapper is not executable: $GIT_WRAPPER"
[[ -x "$QQ_NOTIFIER" ]] || die "QQ notifier is missing or not executable: $QQ_NOTIFIER"
[[ -d "$BASE_ENV" ]] || die "base environment is missing: $BASE_ENV"
[[ -d "$DATASET_ROOT/train_original/A" ]] || die "LEVIR-CD train A directory is missing"
[[ -d "$DATASET_ROOT/train_original/B" ]] || die "LEVIR-CD train B directory is missing"
[[ -d "$DATASET_ROOT/train_original/label" ]] || die "LEVIR-CD train label directory is missing"
[[ ! -e "$DATASET_DIR" ]] || die "refusing to replace an existing dataset directory: $DATASET_DIR"
[[ ! -e "$RUN_DIR/.prepare_complete" ]] || die "prepare completion marker already exists"
[[ ! -e "$RUN_DIR/.training_submitted" ]] || die "training was already submitted for this run"

# Propagate only path/configuration values needed by the setup script. Proxy or
# credential variables are neither inspected nor printed by this worker.
export BASE_ENV TRAIN_ENV MODEL_PATH LLAMA_FACTORY_ROOT GIT_WRAPPER
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

KEY_RESULT="environment_installing"
bash "$SETUP_SCRIPT" install
PYTHON="$TRAIN_ENV/bin/python"
[[ -x "$PYTHON" ]] || die "training interpreter was not created: $PYTHON"
KEY_RESULT="environment_ready"

export PYTHONPATH="$ROOT:$LLAMA_FACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
(
  cd "$ROOT"
  "$PYTHON" "$DATA_TEST"
  "$PYTHON" "$SCORE_TEST"
)
KEY_RESULT="data_and_scorer_unit_tests_passed"

"$PYTHON" "$DATA_BUILDER" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$DATASET_DIR" \
  --seed 729
KEY_RESULT="dataset_built"

# Re-open all persisted artifacts through the same validator used by the GPU
# launcher; this is intentionally separate from the builder's in-memory gate.
"$PYTHON" "$DATA_BUILDER" \
  --output-dir "$DATASET_DIR" \
  --validate-only

KEY_RESULT="$("$PYTHON" - "$DATASET_DIR/validation.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"false_negative": 16, "false_positive_change": 16}
if report.get("valid") is not True:
    raise SystemExit("validation.json does not report valid=true")
if report.get("examples") != 32:
    raise SystemExit(f"expected 32 examples, found {report.get('examples')!r}")
if report.get("unique_parent_scenes") != 32:
    raise SystemExit("expected 32 unique parent scenes")
if report.get("class_distribution") != expected:
    raise SystemExit(f"unexpected class distribution: {report.get('class_distribution')!r}")
print("dataset_valid_rows=32_classes=16fn+16fp_parents=32")
PY
)"

COMPLETION_TMP="$RUN_DIR/.prepare_complete.tmp-${JOB_ID}"
printf 'status=success\njob_id=%s\nrun_dir=%s\ndataset_dir=%s\nresult=%s\n' \
  "$JOB_ID" "$RUN_DIR" "$DATASET_DIR" "$KEY_RESULT" >"$COMPLETION_TMP"
mv -- "$COMPLETION_TMP" "$RUN_DIR/.prepare_complete"
printf 'qwen3vl_sft_prepare: %s\n' "$KEY_RESULT"
