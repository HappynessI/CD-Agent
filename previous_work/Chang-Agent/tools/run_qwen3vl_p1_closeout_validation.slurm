#!/usr/bin/env bash
#SBATCH --job-name=CA0802P1Closeout
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@60

set -euo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p1_closeout_validation.slurm"
ARTIFACT_ROOT="$(realpath -e -- "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}")"
PYTHON="$(realpath -e -- "${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}")"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
EXPECTED_WORKER_SHA256="${EXPECTED_WORKER_SHA256:?EXPECTED_WORKER_SHA256 is required}"
FREEZE_COMMIT="5210f89f5ca13ee1e636f5aa6a44631206a80123"
ANALYSIS_RECEIPT="$WORKSPACE/wyh/diagnostics/qwen3vl_p1_retry3_result_0802/trajectory_analysis.json"
ANALYSIS_SHA256="<hash omitted>"
RESULT_JSON="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_RETRY3_RESULT_0802.json"
PARENT_CLOSEOUT_RECEIPT="$WORKSPACE/wyh/diagnostics/qwen3vl_p1_closeout_validation_0802/job-53178/closeout_validation_receipt.json"
PARENT_CLOSEOUT_RECEIPT_SHA256="<hash omitted>"
DIAG_ROOT="$(realpath -e -- "${DIAG_ROOT:?DIAG_ROOT is required}")"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_DIR="$DIAG_ROOT/job-$JOB_ID"
STATUS="$DIAG_ROOT/status-$JOB_ID.txt"
RECEIPT="$JOB_DIR/closeout_validation_receipt.json"
START_SECONDS="$SECONDS"
RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"

mkdir -p -- "$JOB_DIR"

record_error() {
  local rc="$?"
  if [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

finish() {
  local rc="$?"
  trap - EXIT ERR TERM INT
  local state="failed"
  if [[ "$rc" -eq 0 ]]; then
    state="success"
  elif [[ "$rc" -eq 130 || "$rc" -eq 143 ]]; then
    state="cancelled"
  fi
  local elapsed="$((SECONDS - START_SECONDS))"
  local temporary="${STATUS}.tmp-${BASHPID}"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\ncommit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\nreceipt=%s\n' \
    "$state" "$rc" "$elapsed" "$JOB_ID" "$EXPECTED_COMMIT" \
    "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" "$RECEIPT" >"$temporary"
  mv -f -- "$temporary" "$STATUS"
  if [[ -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0802 P1 closeout validation" \
    QQ_BODY="job_id=${JOB_ID}; status=${state}; result=${RESULT}; elapsed_seconds=${elapsed}; artifact_path=${JOB_DIR}" \
      "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap record_error ERR
trap finish EXIT
trap 'ERROR_TEXT="received_SIGTERM"; exit 143' TERM
trap 'ERROR_TEXT="received_SIGINT"; exit 130' INT

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT="not_running_under_slurm"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" ]] || { ERROR_TEXT="unexpected_partition"; exit 2; }
[[ "${SLURMD_NODENAME:-}" == "gpu48" ]] || { ERROR_TEXT="unexpected_node"; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || { ERROR_TEXT="unexpected_cpu_count"; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) ERROR_TEXT="unexpected_visible_gpu"; exit 2 ;;
esac
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] \
  || { ERROR_TEXT="unexpected_slurm_gpu_allocation"; exit 2; }
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { ERROR_TEXT="malformed_expected_commit"; exit 2; }
[[ "$EXPECTED_WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { ERROR_TEXT="malformed_expected_worker_sha256"; exit 2; }
[[ -f "$PINNED_WORKER" && -f "$RESULT_JSON" && -f "$ANALYSIS_RECEIPT" \
    && -f "$PARENT_CLOSEOUT_RECEIPT" ]] \
  || { ERROR_TEXT="required_closeout_input_missing"; exit 2; }
[[ -x "$PYTHON" ]] || { ERROR_TEXT="python_missing"; exit 2; }
[[ -x "$QQ_NOTIFIER" ]] || { ERROR_TEXT="qq_notifier_missing"; exit 2; }

ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="executed_worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="repository_worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$ANALYSIS_RECEIPT" | awk '{print $1}')" == "$ANALYSIS_SHA256" ]] \
  || { ERROR_TEXT="trajectory_analysis_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$PARENT_CLOSEOUT_RECEIPT" | awk '{print $1}')" == "$PARENT_CLOSEOUT_RECEIPT_SHA256" ]] \
  || { ERROR_TEXT="parent_closeout_receipt_hash_drift"; exit 2; }
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] \
  || { ERROR_TEXT="repository_commit_drift"; exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_not_clean"; exit 2; }
git -C "$ROOT" merge-base --is-ancestor "$FREEZE_COMMIT" "$EXPECTED_COMMIT" \
  || { ERROR_TEXT="freeze_not_ancestor"; exit 2; }
[[ -L "$ROOT/outputs" ]] || { ERROR_TEXT="outputs_symlink_missing"; exit 2; }
[[ "$(realpath -e -- "$ROOT/outputs")" == "$(realpath -e -- "$ARTIFACT_ROOT/outputs")" ]] \
  || { ERROR_TEXT="outputs_symlink_target_drift"; exit 2; }

FROZEN_FILES=(
  doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_PREREGISTRATION_0802.json
  doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json
  doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json
  doc/training_pipeline/0802_third_try/QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json
  doc/training_pipeline/0802_third_try/QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json
)
git -C "$ROOT" diff --exit-code "$FREEZE_COMMIT" -- "${FROZEN_FILES[@]}" \
  || { ERROR_TEXT="frozen_machine_contract_changed"; exit 2; }

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT"
cd "$ROOT"

RESULT="validating_result_and_analysis_receipts"
"$PYTHON" - "$RESULT_JSON" "$ANALYSIS_RECEIPT" "$PARENT_CLOSEOUT_RECEIPT" <<'PY'
import hashlib
import json
import pathlib
import sys

def load(path_text):
    path = pathlib.Path(path_text)
    seen = []
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise SystemExit(f"duplicate JSON key in {path}: {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)

result = load(sys.argv[1])
analysis = load(sys.argv[2])
parent_closeout = load(sys.argv[3])
assert result["schema_id"] == "qwen3vl_p1_retry3_result_v1"
assert result["result_status"] == "complete"
assert result["engineering_outcome"] == "pass"
assert result["scientific_outcome"] == "diagnostic_no_gain"
assert result["git_lineage"]["final_freeze_commit_b4"] == "5210f89f5ca13ee1e636f5aa6a44631206a80123"
assert [item["job_id"] for item in result["formal_jobs"]] == [
    "53155", "53156", "53157", "53158", "53159", "53160"
]
assert all(item["state"] == "COMPLETED" for item in result["formal_jobs"])
comparison = result["core3_comparison"]
assert comparison["h0"]["micro_iou"] == comparison["h1"]["micro_iou"] == 0.7422312
assert comparison["h0"]["micro_f1"] == comparison["h1"]["micro_f1"] == 0.85204673
assert comparison["delta_h1_minus_h0"]["micro_iou"] == 0.0
assert comparison["delta_h1_minus_h0"]["micro_f1"] == 0.0
assert result["registry_closeout_target"] == {
    "experiment_rows": 14,
    "field_dictionary_rows": 52,
    "data_rows": 21,
    "attempt_rows": 38,
}
closeout = result["registry_closeout_artifacts"]
assert closeout["closeout_status"] == "validated"
assert closeout["workbook_preparation_job_id"] == "53171"
assert closeout["workbook_raw_sha256"] == "<hash omitted>"
assert [(item["job_id"], item["state"]) for item in closeout["candidate_validation_attempts"]] == [
    ("53174", "FAILED"), ("53178", "COMPLETED")
]
assert closeout["candidate_validation_attempts"][-1]["receipt_sha256"] == (
    "<hash omitted>"
)
assert result["post_p1_authorization"]["authorization_effective"] is True
assert analysis["schema_id"] == "qwen3vl_p1_retry3_trajectory_analysis_v1"
assert analysis["status"] == "complete"
assert analysis["slurm_job_id"] == "53165"
assert analysis["independent_gates"]["controlled_variable_checks_passed"] == 18
assert analysis["independent_gates"]["hard_gates_passed"] == 25
assert analysis["comparison"]["round_two_invocations"] == {"h0": 0, "h1": 2}
assert analysis["conclusions"]["training_data_hypothesis_status"] == (
    "consistent_with_observations_and_registered_for_controlled_test_not_proven_causal"
)
assert parent_closeout["schema_id"] == "qwen3vl_p1_closeout_validation_receipt_v1"
assert parent_closeout["status"] == "complete"
assert parent_closeout["job_id"] == "53178"
assert parent_closeout["commit"] == "d64ec522de1d46df27891cea1f9f9ab0fa1b7be0"
assert all(value == "passed" for key, value in parent_closeout["checks"].items() if key not in {"frozen_machine_contract_unchanged", "repository_unchanged"})
assert parent_closeout["checks"]["frozen_machine_contract_unchanged"] is True
assert parent_closeout["checks"]["repository_unchanged"] is True
print("result_analysis_contract=passed")
PY

RESULT="checking_workbook_determinism_and_external_evidence"
"$PYTHON" experiments/build_experiment_workbook.py --check

RESULT="running_registry_targeted_tests"
"$PYTHON" -m unittest tests.test_experiment_registry

RESULT="checking_closeout_shell_syntax"
bash -n tools/run_qwen3vl_p1_closeout_validation.slurm

RESULT="running_full_regression"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'

RESULT="verifying_repository_immutability"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] \
  || { ERROR_TEXT="repository_commit_changed_during_validation"; exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_changed_during_validation"; exit 2; }

RESULT="writing_closeout_validation_receipt"
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$EXPECTED_COMMIT" "$ACTUAL_WORKER_SHA256" \
  "$RESULT_JSON" "$ANALYSIS_RECEIPT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
job_id, commit, worker_sha256 = sys.argv[2:5]
result_path = pathlib.Path(sys.argv[5])
analysis_path = pathlib.Path(sys.argv[6])
def binding(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
payload = {
    "schema_id": "qwen3vl_p1_closeout_validation_receipt_v1",
    "status": "complete",
    "job_id": job_id,
    "commit": commit,
    "worker_sha256": worker_sha256,
    "resources": {"gpus": 0, "cpus": 2, "memory": "8G"},
    "checks": {
        "frozen_machine_contract_unchanged": True,
        "result_analysis_contract": "passed",
        "workbook_check": "passed",
        "registry_targeted_tests": "passed",
        "full_regression": "passed",
        "repository_unchanged": True,
    },
    "bindings": {
        "result": binding(result_path),
        "trajectory_analysis": binding(analysis_path),
    },
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

RESULT="p1_closeout_validation_complete"
exit 0
