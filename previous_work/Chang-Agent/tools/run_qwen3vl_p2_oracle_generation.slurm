#!/usr/bin/env bash
#SBATCH --job-name=CA0803P2Generate
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ROOT_RAW="${1:-}"
WORKSPACE_RAW="${2:-}"
RUN_ROOT_RAW="${3:-}"
MANIFEST_RAW="${4:-}"
PYTHON_RAW="${5:-}"
QQ_NOTIFIER_RAW="${6:-}"
EXPECTED_COMMIT="${7:-missing}"
EXPECTED_IMPLEMENTATION_COMMIT="${8:-missing}"
EXPECTED_WORKER_SHA256="${9:-missing}"
EXPECTED_ACTION_GENERATOR_SHA256="${10:-missing}"
EXPECTED_RELEASE_HELPER_SHA256="${11:-missing}"
EXPECTED_MANIFEST_RAW_SHA256="${12:-missing}"
EXPECTED_MANIFEST_CANONICAL_SHA256="${13:-missing}"
CONTRACT_JOB_ID="${14:-missing}"
CONTRACT_RECEIPT_RAW="${15:-}"
EXPECTED_CONTRACT_RECEIPT_SHA256="${16:-missing}"
EXPECTED_CONTRACT_WORKER_SHA256="${17:-missing}"
EXPECTED_PREREG_RAW_SHA256="${18:-missing}"
VALIDATION_BASE_RAW="${19:-}"
SUBMISSION_INTENT_RAW="${20:-}"
EXPECTED_INTENT_SHA256="${21:-missing}"
FORMAL_SUBMISSION_RAW="${22:-}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
JOB_USER="unknown"
START_SECONDS="$SECONDS"
RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
STATUS="${RUN_ROOT_RAW%/}/logs/generation-status-$JOB_ID.txt"
RECEIPT="${RUN_ROOT_RAW%/}/generation_receipt.json"
OUTPUT="${RUN_ROOT_RAW%/}/generation"

record_error() { local rc="$?"; [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"; return "$rc"; }

write_failure_receipt() {
  [[ -d "$RUN_ROOT_RAW" && ! -e "$RECEIPT" && -x "$PYTHON_RAW" ]] || return 0
  "$PYTHON_RAW" - "$RECEIPT" "$1" "$JOB_ID" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" \
    "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" "$CONTRACT_JOB_ID" "$FORMAL_SUBMISSION_RAW" <<'PY' || true
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);formal=pathlib.Path(sys.argv[10])
binding=None
if formal.is_file(): binding={"path":str(formal.resolve()),"sha256":hashlib.sha256(formal.read_bytes()).hexdigest(),"size_bytes":formal.stat().st_size}
payload={"schema_id":"qwen3vl_p2_oracle_generation_receipt_v2","status":sys.argv[2],"job_id":sys.argv[3],
 "freeze_commit_b":sys.argv[4],"implementation_commit_a":sys.argv[5],"worker_sha256":sys.argv[6],
 "result":sys.argv[7],"error":sys.argv[8] or None,"contract_validation":{"job_id":sys.argv[9]},
 "formal_submission_receipt":binding,"gt_loaded":False}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

finish() {
  local rc="$?" state="failed"
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state="success"
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state="cancelled"
  if [[ -d "${RUN_ROOT_RAW%/}/logs" ]]; then
    printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\njob_user=%s\ncommit=%s\nimplementation_commit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\noutput=%s\nreceipt=%s\n' \
      "$state" "$rc" "$((SECONDS - START_SECONDS))" "$JOB_ID" "$JOB_NAME" "$JOB_USER" "$EXPECTED_COMMIT" \
      "$EXPECTED_IMPLEMENTATION_COMMIT" "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" "$OUTPUT" "$RECEIPT" \
      >"$STATUS.tmp-$$" && mv -f -- "$STATUS.tmp-$$" "$STATUS" || true
  fi
  [[ -e "$RECEIPT" ]] || write_failure_receipt "$state"
  if [[ -x "$QQ_NOTIFIER_RAW" ]]; then
    QQ_SUB="CA0803 P2 candidate generation" \
    QQ_BODY="job_id=$JOB_ID; status=$state; result=$RESULT; elapsed_seconds=$((SECONDS - START_SECONDS)); attempt=$RUN_ROOT_RAW" \
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
RUN_ROOT="$(realpath -e -- "$RUN_ROOT_RAW")"
MANIFEST="$(realpath -e -- "$MANIFEST_RAW")"
PYTHON="$(realpath -e -- "$PYTHON_RAW")"
QQ_NOTIFIER="$(realpath -e -- "$QQ_NOTIFIER_RAW")"
CONTRACT_RECEIPT="$(realpath -e -- "$CONTRACT_RECEIPT_RAW")"
VALIDATION_BASE="$(realpath -e -- "$VALIDATION_BASE_RAW")"
SUBMISSION_INTENT="$(realpath -e -- "$SUBMISSION_INTENT_RAW")"
FORMAL_SUBMISSION="$FORMAL_SUBMISSION_RAW"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p2_oracle_generation.slurm"
ACTION_GENERATOR="$ROOT/tools/run_qwen3vl_p2_oracle.py"
RELEASE_HELPER="$ROOT/tools/verify_qwen3vl_p2_release.py"
OUTPUT="$RUN_ROOT/generation"
LOG_DIR="$RUN_ROOT/logs"
STATUS="$LOG_DIR/generation-status-$JOB_ID.txt"
RECEIPT="$RUN_ROOT/generation_receipt.json"

[[ "$JOB_ID" =~ ^[0-9]+$ && "$CONTRACT_JOB_ID" =~ ^[0-9]+$ \
    && "$JOB_NAME" == "CA0803P2Generate" && "$JOB_USER" == "guisongxia" ]] \
  || { ERROR_TEXT="malformed_job_identity"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" && "${SLURMD_NODENAME:-}" == "gpu48" \
    && "${SLURM_CPUS_PER_TASK:-}" == "4" ]] || { ERROR_TEXT="unexpected_allocation"; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" \
    && -n "${SLURM_JOB_GPUS:-}" ]] || { ERROR_TEXT="single_gpu_not_visible_or_allocated"; exit 2; }
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$EXPECTED_IMPLEMENTATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { ERROR_TEXT="malformed_commit_identity"; exit 2; }
for value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_ACTION_GENERATOR_SHA256" \
  "$EXPECTED_RELEASE_HELPER_SHA256" "$EXPECTED_MANIFEST_RAW_SHA256" \
  "$EXPECTED_MANIFEST_CANONICAL_SHA256" "$EXPECTED_CONTRACT_RECEIPT_SHA256" \
  "$EXPECTED_CONTRACT_WORKER_SHA256" "$EXPECTED_PREREG_RAW_SHA256" "$EXPECTED_INTENT_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { ERROR_TEXT="malformed_expected_sha256"; exit 2; }
done
[[ -d "$LOG_DIR" && ! -e "$OUTPUT" && ! -e "$RECEIPT" && ! -e "$STATUS" ]] \
  || { ERROR_TEXT="generation_attempt_artifact_exists"; exit 2; }
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" && -f "$PINNED_WORKER" && -f "$ACTION_GENERATOR" && -f "$RELEASE_HELPER" ]] \
  || { ERROR_TEXT="required_runtime_input_missing"; exit 2; }

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" \
    && "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$ACTION_GENERATOR" | awk '{print $1}')" == "$EXPECTED_ACTION_GENERATOR_SHA256" \
    && "$(sha256sum -- "$RELEASE_HELPER" | awk '{print $1}')" == "$EXPECTED_RELEASE_HELPER_SHA256" ]] \
  || { ERROR_TEXT="generator_or_helper_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_RAW_SHA256" \
    && "$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_CONTRACT_RECEIPT_SHA256" \
    && "$(sha256sum -- "$SUBMISSION_INTENT" | awk '{print $1}')" == "$EXPECTED_INTENT_SHA256" ]] \
  || { ERROR_TEXT="manifest_contract_or_intent_hash_drift"; exit 2; }
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { ERROR_TEXT="repository_commit_drift"; exit 2; }
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then ERROR_TEXT="run_worktree_is_not_detached"; exit 2; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { ERROR_TEXT="repository_not_clean"; exit 2; }

RESULT="verifying_single_parent_add_only_freeze"
"$PYTHON" "$RELEASE_HELPER" verify-freeze --repo-root "$ROOT" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT"

RESULT="waiting_for_complete_formal_submission_receipt"
for _ in $(seq 1 30); do [[ -f "$FORMAL_SUBMISSION" ]] && break; sleep 1; done
[[ -f "$FORMAL_SUBMISSION" ]] || { ERROR_TEXT="formal_submission_receipt_missing"; exit 2; }
FORMAL_SUBMISSION="$(realpath -e -- "$FORMAL_SUBMISSION")"
"$PYTHON" "$RELEASE_HELPER" verify-formal-submission --attempt-root "$RUN_ROOT" --role generation \
  --job-id "$JOB_ID" --generation-job-id "$JOB_ID" --contract-job-id "$CONTRACT_JOB_ID" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" \
  --worker-sha256 "$EXPECTED_WORKER_SHA256" --contract-receipt "$CONTRACT_RECEIPT" \
  --contract-receipt-sha256 "$EXPECTED_CONTRACT_RECEIPT_SHA256"
"$PYTHON" "$RELEASE_HELPER" verify-contract-receipt --receipt "$CONTRACT_RECEIPT" \
  --validation-base "$VALIDATION_BASE" --repo-root "$ROOT" --job-id "$CONTRACT_JOB_ID" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" \
  --worker-sha256 "$EXPECTED_CONTRACT_WORKER_SHA256" --prereg-sha256 "$EXPECTED_PREREG_RAW_SHA256" \
  --execution-sha256 "$EXPECTED_MANIFEST_RAW_SHA256"

RESULT="verifying_manifest_canonical_hash"
"$PYTHON" - "$MANIFEST" "$EXPECTED_MANIFEST_CANONICAL_SHA256" <<'PY'
import hashlib,json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
b=json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
if hashlib.sha256(b).hexdigest()!=sys.argv[2]: raise SystemExit("execution manifest canonical SHA256 differs")
PY

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"
RESULT="running_gt_free_candidate_generation"
"$PYTHON" "$ACTION_GENERATOR" --execution-manifest "$MANIFEST" --output "$OUTPUT" --runtime-commit "$EXPECTED_COMMIT"

RESULT="validating_generation_outputs"
"$PYTHON" "$RELEASE_HELPER" verify-generation-output --candidate-root "$OUTPUT" \
  --execution-manifest "$MANIFEST" --runtime-commit "$EXPECTED_COMMIT"

RESULT="verifying_repository_immutability"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_changed_during_generation"; exit 2; }

RESULT="writing_generation_receipt"
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" \
  "$ACTUAL_WORKER_SHA256" "$MANIFEST" "$OUTPUT/candidate_generation_complete.json" \
  "$OUTPUT/candidates.json" "$CONTRACT_JOB_ID" "$CONTRACT_RECEIPT" "$FORMAL_SUBMISSION" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);manifest=pathlib.Path(sys.argv[6]);marker=pathlib.Path(sys.argv[7]);index=pathlib.Path(sys.argv[8])
contract=pathlib.Path(sys.argv[10]);formal=pathlib.Path(sys.argv[11])
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
payload={"schema_id":"qwen3vl_p2_oracle_generation_receipt_v2","status":"complete","job_id":sys.argv[2],
 "freeze_commit_b":sys.argv[3],"implementation_commit_a":sys.argv[4],"worker_sha256":sys.argv[5],
 "resources":{"node":"gpu48","accelerator":"A800","gpus":1,"cpus":4,"memory":"48G"},
 "gt_loaded":False,"counts":{"total":36,"primary":27,"diagnostic_executor_only":9},
 "candidate_flow_counts":{"requested":36,"generated":36,"tool_success":36},
 "contract_validation":{"job_id":sys.argv[9],"receipt":bind(contract)},
 "bindings":{"execution_manifest":bind(manifest),"complete_marker":bind(marker),"candidate_index":bind(index),
             "formal_submission_receipt":bind(formal)}}
temporary=target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
os.replace(temporary,target)
PY
RESULT="p2_candidate_generation_complete"
exit 0
