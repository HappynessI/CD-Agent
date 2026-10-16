#!/usr/bin/env bash
#SBATCH --job-name=CA0803P2Contract
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Positional arguments are used with sbatch --export=NIL, so the submitting
# shell's credentials, proxies, and Python controls never enter the job.
ROOT_RAW="${1:-}"
WORKSPACE_RAW="${2:-}"
ARTIFACT_ROOT_RAW="${3:-}"
ATTEMPT_ROOT_RAW="${4:-}"
PYTHON_RAW="${5:-}"
QQ_NOTIFIER_RAW="${6:-}"
EXPECTED_COMMIT="${7:-missing}"
EXPECTED_IMPLEMENTATION_COMMIT="${8:-missing}"
EXPECTED_WORKER_SHA256="${9:-missing}"
EXPECTED_RELEASE_HELPER_SHA256="${10:-missing}"
EXPECTED_PREREG_RAW_SHA256="${11:-missing}"
EXPECTED_PREREG_CANONICAL_SHA256="${12:-missing}"
EXPECTED_EXECUTION_RAW_SHA256="${13:-missing}"
EXPECTED_EXECUTION_CANONICAL_SHA256="${14:-missing}"
SUBMISSION_INTENT_RAW="${15:-}"
EXPECTED_INTENT_SHA256="${16:-missing}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
JOB_USER="unknown"
START_SECONDS="$SECONDS"
RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
STATUS_FINALIZED=0
STATUS="${ATTEMPT_ROOT_RAW%/}/status-$JOB_ID.txt"
RECEIPT="${ATTEMPT_ROOT_RAW%/}/job-$JOB_ID/contract_validation_receipt.json"

write_status() {
  local rc="$1" state="$2" elapsed="$((SECONDS - START_SECONDS))"
  [[ -d "$ATTEMPT_ROOT_RAW" ]] || return 0
  local temporary="${STATUS}.tmp-${BASHPID}"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\njob_user=%s\ncommit=%s\nimplementation_commit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\nreceipt=%s\n' \
    "$state" "$rc" "$elapsed" "$JOB_ID" "$JOB_NAME" "$JOB_USER" "$EXPECTED_COMMIT" \
    "$EXPECTED_IMPLEMENTATION_COMMIT" "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" "$RECEIPT" >"$temporary"
  mv -f -- "$temporary" "$STATUS"
}

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" state="failed"
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state="success"
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state="cancelled"
  [[ "$STATUS_FINALIZED" -eq 1 && "$rc" -eq 0 ]] || write_status "$rc" "$state" || true
  if [[ -x "$QQ_NOTIFIER_RAW" ]]; then
    QQ_SUB="CA0803 P2 contract validation" \
    QQ_BODY="job_id=$JOB_ID; status=$state; result=$RESULT; elapsed_seconds=$((SECONDS - START_SECONDS)); attempt=$ATTEMPT_ROOT_RAW" \
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
WORKSPACE="$(realpath -e -- "$WORKSPACE_RAW")"
ARTIFACT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT_RAW")"
ATTEMPT_ROOT="$(realpath -e -- "$ATTEMPT_ROOT_RAW")"
PYTHON="$(realpath -e -- "$PYTHON_RAW")"
QQ_NOTIFIER="$(realpath -e -- "$QQ_NOTIFIER_RAW")"
SUBMISSION_INTENT="$(realpath -e -- "$SUBMISSION_INTENT_RAW")"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p2_contract_validation.slurm"
RELEASE_HELPER="$ROOT/tools/verify_qwen3vl_p2_release.py"
PREREG="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_preregistration.py"
SOURCE_COMMIT="5210f89f5ca13ee1e636f5aa6a44631206a80123"
JOB_DIR="$ATTEMPT_ROOT/job-$JOB_ID"
STATUS="$ATTEMPT_ROOT/status-$JOB_ID.txt"
RECEIPT="$JOB_DIR/contract_validation_receipt.json"
SUBMISSION_RECEIPT="$ATTEMPT_ROOT/submission_receipt.json"

[[ "$JOB_ID" =~ ^[0-9]+$ && "$JOB_NAME" == "CA0803P2Contract" && "$JOB_USER" == "guisongxia" ]] \
  || { ERROR_TEXT="unexpected_job_identity"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" && "${SLURMD_NODENAME:-}" == "gpu48" \
    && "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || { ERROR_TEXT="unexpected_allocation"; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT="unexpected_visible_gpu"; exit 2 ;; esac
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || { ERROR_TEXT="unexpected_gpu_allocation"; exit 2; }
[[ "$ATTEMPT_ROOT" != "$ROOT" && "$ATTEMPT_ROOT" != "$ROOT/"* \
    && "$ATTEMPT_ROOT" != "$ARTIFACT_ROOT" && "$ATTEMPT_ROOT" != "$ARTIFACT_ROOT/"* ]] \
  || { ERROR_TEXT="diagnostics_inside_repository"; exit 2; }
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$EXPECTED_IMPLEMENTATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { ERROR_TEXT="malformed_commit_identity"; exit 2; }
for value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_RELEASE_HELPER_SHA256" \
  "$EXPECTED_PREREG_RAW_SHA256" "$EXPECTED_PREREG_CANONICAL_SHA256" \
  "$EXPECTED_EXECUTION_RAW_SHA256" "$EXPECTED_EXECUTION_CANONICAL_SHA256" "$EXPECTED_INTENT_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { ERROR_TEXT="malformed_expected_sha256"; exit 2; }
done
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" && -f "$PINNED_WORKER" && -f "$RELEASE_HELPER" \
    && -f "$PREREG" && -f "$EXECUTION" && -f "$VALIDATOR" ]] \
  || { ERROR_TEXT="required_contract_input_missing"; exit 2; }
[[ ! -e "$JOB_DIR" && ! -e "$STATUS" ]] || { ERROR_TEXT="attempt_job_artifact_exists"; exit 2; }
mkdir -- "$JOB_DIR"

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" \
    && "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$RELEASE_HELPER" | awk '{print $1}')" == "$EXPECTED_RELEASE_HELPER_SHA256" ]] \
  || { ERROR_TEXT="release_helper_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$PREREG" | awk '{print $1}')" == "$EXPECTED_PREREG_RAW_SHA256" \
    && "$(sha256sum -- "$EXECUTION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_RAW_SHA256" \
    && "$(sha256sum -- "$SUBMISSION_INTENT" | awk '{print $1}')" == "$EXPECTED_INTENT_SHA256" ]] \
  || { ERROR_TEXT="input_hash_drift"; exit 2; }
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { ERROR_TEXT="repository_commit_drift"; exit 2; }
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then ERROR_TEXT="run_worktree_is_not_detached"; exit 2; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { ERROR_TEXT="repository_not_clean"; exit 2; }
[[ -L "$ROOT/outputs" && "$(realpath -e -- "$ROOT/outputs")" == "$(realpath -e -- "$ARTIFACT_ROOT/outputs")" ]] \
  || { ERROR_TEXT="outputs_symlink_drift"; exit 2; }
git -C "$ROOT" check-ignore -q -- outputs || { ERROR_TEXT="outputs_symlink_not_ignored"; exit 2; }

RESULT="verifying_single_parent_add_only_freeze"
"$PYTHON" "$RELEASE_HELPER" verify-freeze --repo-root "$ROOT" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT"

RESULT="waiting_for_append_only_submission_receipt"
for _ in $(seq 1 30); do [[ -f "$SUBMISSION_RECEIPT" ]] && break; sleep 1; done
[[ -f "$SUBMISSION_RECEIPT" ]] || { ERROR_TEXT="submission_receipt_missing"; exit 2; }
"$PYTHON" "$RELEASE_HELPER" verify-contract-submission --attempt-root "$ATTEMPT_ROOT" \
  --job-id "$JOB_ID" --freeze-commit "$EXPECTED_COMMIT" \
  --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" --worker-sha256 "$EXPECTED_WORKER_SHA256" \
  --prereg-sha256 "$EXPECTED_PREREG_RAW_SHA256" --execution-sha256 "$EXPECTED_EXECUTION_RAW_SHA256"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"
RESULT="validating_frozen_manifests"
"$PYTHON" "$VALIDATOR" --preregistration "$PREREG" --execution-manifest "$EXECUTION" \
  --repo-root "$ROOT" --phase frozen \
  --expected-preregistration-canonical-sha256 "$EXPECTED_PREREG_CANONICAL_SHA256" \
  --expected-execution-canonical-sha256 "$EXPECTED_EXECUTION_CANONICAL_SHA256"

RESULT="checking_frozen_runtime_diff"
FROZEN_RUNTIME=(
  change_agent/action_parser.py
  change_agent/adapters/omniovcd_adapter.py
  change_agent/adapters/sam3_adapter.py
  change_agent/adapters/segagent_adapter.py
  change_agent/adapters/staged_verifier.py
  change_agent/adapters/subprocess_adapters.py
  change_agent/environment.py
  change_agent/executor.py
  change_agent/state.py
  change_agent/trajectory.py
  change_agent/verifier.py
  change_agent/verifier_protocol.py
  change_agent/verifier_regions.py
  change_agent/coordinates.py
  tools/seeded_segmentation_worker.py
  tools/segmentation_worker.py
)
git -C "$ROOT" diff --exit-code "$SOURCE_COMMIT" -- "${FROZEN_RUNTIME[@]}" \
  || { ERROR_TEXT="p1_runtime_source_changed"; exit 2; }

RESULT="checking_shell_syntax"
bash -n tools/run_qwen3vl_p2_manifest_build.slurm tools/run_qwen3vl_p2_contract_validation.slurm \
  tools/run_qwen3vl_p2_oracle_generation.slurm tools/run_qwen3vl_p2_oracle_scoring.slurm \
  tools/submit_qwen3vl_p2_manifest_build.sh tools/submit_qwen3vl_p2_oracle.sh

RESULT="running_p2_targeted_tests"
"$PYTHON" -m unittest tests.test_qwen3vl_p2_preregistration tests.test_qwen3vl_p2_oracle
RESULT="running_registry_targeted_tests"
"$PYTHON" -m unittest tests.test_experiment_registry
RESULT="running_full_regression"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'

RESULT="verifying_repository_immutability"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_changed_during_validation"; exit 2; }

RESULT="p2_contract_validation_complete"
write_status 0 success
STATUS_FINALIZED=1
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$JOB_NAME" "$JOB_USER" "$EXPECTED_COMMIT" \
  "$EXPECTED_IMPLEMENTATION_COMMIT" "$ACTUAL_WORKER_SHA256" "$PREREG" "$EXECUTION" \
  "$STATUS" "$SUBMISSION_INTENT" "$SUBMISSION_RECEIPT" "$RELEASE_HELPER" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1])
paths=[pathlib.Path(value) for value in sys.argv[8:14]]
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
checks={"clean_detached_exact_b":True,"single_parent_freeze_b":True,"two_manifest_add_only_freeze":True,
 "manifest_independent_validation":"passed","p1_runtime_unchanged":True,"shell_syntax":"passed",
 "p2_targeted_tests":"passed","registry_targeted_tests":"passed","full_regression":"passed","repository_unchanged":True}
payload={"schema_id":"qwen3vl_p2_contract_validation_receipt_v1","status":"complete",
 "job_id":sys.argv[2],"job_name":sys.argv[3],"job_user":sys.argv[4],"node":"gpu48","partition":"gre",
 "freeze_commit_b":sys.argv[5],"implementation_commit_a":sys.argv[6],"freeze_b_parent_is_a":True,
 "worker_sha256":sys.argv[7],"resources":{"gpus":0,"cpus":2,"memory":"16G"},"checks":checks,
 "bindings":dict(zip(["preregistration","execution_manifest","status","submission_intent","submission_receipt","release_helper"],map(bind,paths)))}
temporary=target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
os.replace(temporary,target)
PY
exit 0
