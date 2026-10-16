#!/usr/bin/env bash
#SBATCH --job-name=CA0803P2Workbook
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ROOT_RAW="${1:-}"
ARTIFACT_ROOT_RAW="${2:-}"
ATTEMPT_ROOT_RAW="${3:-}"
PYTHON_RAW="${4:-}"
QQ_NOTIFIER_RAW="${5:-}"
EXPECTED_HEAD="${6:-missing}"
EXPECTED_WORKER_SHA256="${7:-missing}"
EXPECTED_SOURCE_SNAPSHOT_SHA256="${8:-missing}"
EXPECTED_STATUS_SHA256="${9:-missing}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
JOB_USER="unknown"
START_SECONDS="$SECONDS"
RESULT_STAGE="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
STATUS="${ATTEMPT_ROOT_RAW%/}/status-$JOB_ID.txt"
RECEIPT="${ATTEMPT_ROOT_RAW%/}/job-$JOB_ID/registry_preparation_receipt.json"
TEST_LOG="${ATTEMPT_ROOT_RAW%/}/job-$JOB_ID/registry_preparation.log"

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

write_status() {
  local rc="$1" state="$2" temporary="${STATUS}.tmp-${BASHPID}"
  [[ -d "$ATTEMPT_ROOT_RAW" ]] || return 0
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\njob_user=%s\nnode=%s\npartition=%s\nhead=%s\nworker_sha256=%s\nsource_snapshot_sha256=%s\nresult=%s\nerror=%s\nreceipt=%s\n' \
    "$state" "$rc" "$((SECONDS - START_SECONDS))" "$JOB_ID" "$JOB_NAME" "$JOB_USER" \
    "${SLURMD_NODENAME:-missing}" "${SLURM_JOB_PARTITION:-missing}" "$EXPECTED_HEAD" \
    "$ACTUAL_WORKER_SHA256" "$EXPECTED_SOURCE_SNAPSHOT_SHA256" "$RESULT_STAGE" \
    "$ERROR_TEXT" "$RECEIPT" >"$temporary"
  mv -f -- "$temporary" "$STATUS"
}

finish() {
  local rc="$?" state="failed"
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state="success"
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state="cancelled"
  write_status "$rc" "$state" || true
  if [[ -x "$QQ_NOTIFIER_RAW" ]]; then
    QQ_SUB="CA0803 P2 registry workbook preparation" \
    QQ_BODY="job_id=$JOB_ID; status=$state; result=$RESULT_STAGE; elapsed_seconds=$((SECONDS - START_SECONDS)); artifact_path=${ATTEMPT_ROOT_RAW%/}" \
      "$QQ_NOTIFIER_RAW" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap record_error ERR
trap finish EXIT
trap 'ERROR_TEXT="received_SIGTERM"; exit 143' TERM
trap 'ERROR_TEXT="received_SIGINT"; exit 130' INT

for forbidden in PYTHONOPTIMIZE PYTHONHOME PYTHONPATH OPENAI_API_KEY HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy SMTP_PASSWORD SMTP_AUTH_CODE; do
  [[ ! -v "$forbidden" ]] || { ERROR_TEXT="forbidden_inherited_environment_$forbidden"; exit 2; }
done
JOB_USER="$(id -un)"

ROOT="$(realpath -e -- "$ROOT_RAW")"
ARTIFACT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT_RAW")"
ATTEMPT_ROOT="$(realpath -e -- "$ATTEMPT_ROOT_RAW")"
PYTHON="$(realpath -e -- "$PYTHON_RAW")"
QQ_NOTIFIER="$(realpath -e -- "$QQ_NOTIFIER_RAW")"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p2_registry_prepare.slurm"
WORKBOOK="$ROOT/experiments/experiment_registry.xlsx"
STATUS="$ATTEMPT_ROOT/status-$JOB_ID.txt"
RECEIPT="$ATTEMPT_ROOT/job-$JOB_ID/registry_preparation_receipt.json"
TEST_LOG="$ATTEMPT_ROOT/job-$JOB_ID/registry_preparation.log"
SOURCE_FILES=(
  doc/training_pipeline/0802_third_try/QWEN3VL_SESSION_HANDOFF_0802.md
  doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.json
  doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.md
  experiments/README.md
  experiments/build_experiment_workbook.py
  experiments/experiment_registry.json
  tests/test_experiment_registry.py
  tests/test_qwen3vl_p2_result.py
  tools/run_qwen3vl_p2_closeout_validation.slurm
  tools/run_qwen3vl_p2_registry_prepare.slurm
  tools/submit_qwen3vl_p2_closeout.sh
  tools/validate_qwen3vl_p2_result.py
)

[[ "$JOB_ID" =~ ^[0-9]+$ && "$JOB_NAME" == "CA0803P2Workbook" \
    && "$JOB_USER" == "guisongxia" ]] || { ERROR_TEXT="unexpected_job_identity"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" && "${SLURMD_NODENAME:-}" == "gpu48" \
    && "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || { ERROR_TEXT="unexpected_allocation"; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT="unexpected_visible_gpu"; exit 2 ;; esac
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] \
  || { ERROR_TEXT="unexpected_gpu_allocation"; exit 2; }
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || { ERROR_TEXT="malformed_expected_head"; exit 2; }
for value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_SOURCE_SNAPSHOT_SHA256" "$EXPECTED_STATUS_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { ERROR_TEXT="malformed_expected_sha256"; exit 2; }
done
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" && -f "$PINNED_WORKER" ]] \
  || { ERROR_TEXT="required_preparation_input_missing"; exit 2; }
[[ ! -e "$ATTEMPT_ROOT/job-$JOB_ID" && ! -e "$STATUS" ]] \
  || { ERROR_TEXT="append_only_job_artifact_exists"; exit 2; }
mkdir -- "$ATTEMPT_ROOT/job-$JOB_ID"

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" \
    && "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="worker_hash_drift"; exit 2; }

RESULT_STAGE="verifying_dirty_source_snapshot"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" \
    && "$(git -C "$ROOT" branch --show-current)" == "feat/local-qwen3vl-p2-oracle-0803" ]] \
  || { ERROR_TEXT="development_checkout_identity_drift"; exit 2; }
git -C "$ROOT" diff --cached --quiet || { ERROR_TEXT="staged_changes_not_allowed"; exit 2; }
[[ -L "$ROOT/outputs" \
    && "$(realpath -e -- "$ROOT/outputs")" == "$(realpath -e -- "$ARTIFACT_ROOT/outputs")" ]] \
  || { ERROR_TEXT="outputs_symlink_target_drift"; exit 2; }
ACTUAL_STATUS_SHA256="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all | sha256sum | awk '{print $1}')"
[[ "$ACTUAL_STATUS_SHA256" == "$EXPECTED_STATUS_SHA256" ]] \
  || { ERROR_TEXT="development_status_snapshot_drift"; exit 2; }
ACTUAL_SOURCE_SNAPSHOT_SHA256="$(cd "$ROOT" && sha256sum -- "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_SNAPSHOT_SHA256" == "$EXPECTED_SOURCE_SNAPSHOT_SHA256" ]] \
  || { ERROR_TEXT="source_snapshot_drift"; exit 2; }

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"
RESULT_STAGE="building_workbook"
"$PYTHON" experiments/build_experiment_workbook.py >"$TEST_LOG" 2>&1
RESULT_STAGE="checking_workbook"
"$PYTHON" experiments/build_experiment_workbook.py --check >>"$TEST_LOG" 2>&1
RESULT_STAGE="running_registry_targeted_tests"
"$PYTHON" -m unittest tests.test_experiment_registry >>"$TEST_LOG" 2>&1
RESULT_STAGE="running_p2_result_targeted_tests"
"$PYTHON" -m unittest tests.test_qwen3vl_p2_result >>"$TEST_LOG" 2>&1
RESULT_STAGE="checking_shell_syntax"
bash -n tools/run_qwen3vl_p2_registry_prepare.slurm \
  tools/run_qwen3vl_p2_closeout_validation.slurm tools/submit_qwen3vl_p2_closeout.sh \
  >>"$TEST_LOG" 2>&1

RESULT_STAGE="verifying_postbuild_source_snapshot"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] \
  || { ERROR_TEXT="development_head_changed"; exit 2; }
[[ "$(cd "$ROOT" && sha256sum -- "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')" \
    == "$EXPECTED_SOURCE_SNAPSHOT_SHA256" ]] || { ERROR_TEXT="source_changed_during_build"; exit 2; }
[[ -f "$WORKBOOK" ]] || { ERROR_TEXT="workbook_missing_after_build"; exit 2; }

RESULT_STAGE="writing_registry_preparation_receipt"
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$EXPECTED_HEAD" "$ACTUAL_WORKER_SHA256" \
  "$EXPECTED_SOURCE_SNAPSHOT_SHA256" "$WORKBOOK" "$TEST_LOG" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
workbook = pathlib.Path(sys.argv[6])
test_log = pathlib.Path(sys.argv[7])
def binding(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
payload = {
    "schema_id": "qwen3vl_p2_registry_preparation_receipt_v1",
    "status": "complete",
    "job_id": sys.argv[2],
    "job_name": "CA0803P2Workbook",
    "job_user": "guisongxia",
    "node": "gpu48",
    "partition": "gre",
    "head_before_candidate_commit": sys.argv[3],
    "worker_sha256": sys.argv[4],
    "source_snapshot_sha256": sys.argv[5],
    "resources": {"gpus": 0, "cpus": 2, "memory": "16G"},
    "checks": {
        "registry_build": "passed",
        "workbook_check": "passed",
        "registry_targeted_tests": "passed",
        "p2_result_targeted_tests": "passed",
        "shell_syntax": "passed",
        "source_snapshot_unchanged": True,
    },
    "bindings": {"workbook": binding(workbook), "test_log": binding(test_log)},
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY

RESULT_STAGE="p2_registry_preparation_complete"
exit 0
