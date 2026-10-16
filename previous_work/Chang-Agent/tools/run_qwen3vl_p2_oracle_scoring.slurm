#!/usr/bin/env bash
#SBATCH --job-name=CA0803P2Score
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ROOT_RAW="${1:-}"
WORKSPACE_RAW="${2:-}"
RUN_ROOT_RAW="${3:-}"
PREREG_RAW_PATH="${4:-}"
EXECUTION_RAW_PATH="${5:-}"
PYTHON_RAW="${6:-}"
QQ_NOTIFIER_RAW="${7:-}"
EXPECTED_COMMIT="${8:-missing}"
EXPECTED_IMPLEMENTATION_COMMIT="${9:-missing}"
EXPECTED_WORKER_SHA256="${10:-missing}"
EXPECTED_GENERATION_WORKER_SHA256="${11:-missing}"
EXPECTED_VALIDATOR_SHA256="${12:-missing}"
EXPECTED_SCORER_SHA256="${13:-missing}"
EXPECTED_RELEASE_HELPER_SHA256="${14:-missing}"
EXPECTED_PREREG_RAW_SHA256="${15:-missing}"
EXPECTED_PREREG_CANONICAL_SHA256="${16:-missing}"
EXPECTED_EXECUTION_RAW_SHA256="${17:-missing}"
EXPECTED_EXECUTION_CANONICAL_SHA256="${18:-missing}"
CONTRACT_JOB_ID="${19:-missing}"
GENERATION_JOB_ID="${20:-missing}"
GENERATION_RECEIPT_RAW="${21:-}"
CONTRACT_RECEIPT_RAW="${22:-}"
EXPECTED_CONTRACT_RECEIPT_SHA256="${23:-missing}"
EXPECTED_CONTRACT_WORKER_SHA256="${24:-missing}"
VALIDATION_BASE_RAW="${25:-}"
SUBMISSION_INTENT_RAW="${26:-}"
EXPECTED_INTENT_SHA256="${27:-missing}"
FORMAL_SUBMISSION_RAW="${28:-}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
JOB_USER="unknown"
START_SECONDS="$SECONDS"
RESULT="not_started"
ERROR_TEXT=""
GENERATION_STATE="unknown"
ACTUAL_WORKER_SHA256="unknown"
STATUS="${RUN_ROOT_RAW%/}/logs/scoring-status-$JOB_ID.txt"
RECEIPT="${RUN_ROOT_RAW%/}/scoring_receipt.json"
OUTPUT="${RUN_ROOT_RAW%/}/scoring"

record_error() { local rc="$?"; [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"; return "$rc"; }

write_failure_receipt() {
  [[ -d "$RUN_ROOT_RAW" && ! -e "$RECEIPT" && -x "$PYTHON_RAW" ]] || return 0
  "$PYTHON_RAW" - "$RECEIPT" "$1" "$JOB_ID" "$GENERATION_JOB_ID" "$EXPECTED_COMMIT" \
    "$EXPECTED_IMPLEMENTATION_COMMIT" "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" \
    "$GENERATION_STATE" "$GENERATION_RECEIPT_RAW" "$FORMAL_SUBMISSION_RAW" <<'PY' || true
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);generation=pathlib.Path(sys.argv[11]);formal=pathlib.Path(sys.argv[12])
def optional(path):
 return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size} if path.is_file() else None
payload={"schema_id":"qwen3vl_p2_oracle_scoring_receipt_v2","status":sys.argv[2],"job_id":sys.argv[3],
 "generation_job_id":sys.argv[4],"freeze_commit_b":sys.argv[5],"implementation_commit_a":sys.argv[6],
 "worker_sha256":sys.argv[7],"result":sys.argv[8],"error":sys.argv[9] or None,
 "generation_state":sys.argv[10],"gt_accessed":False,"generation_receipt":optional(generation),
 "formal_submission_receipt":optional(formal)}
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
    printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\njob_user=%s\ngeneration_job_id=%s\ngeneration_state=%s\ncommit=%s\nimplementation_commit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\noutput=%s\nreceipt=%s\n' \
      "$state" "$rc" "$((SECONDS - START_SECONDS))" "$JOB_ID" "$JOB_NAME" "$JOB_USER" \
      "$GENERATION_JOB_ID" "$GENERATION_STATE" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" \
      "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" "$OUTPUT" "$RECEIPT" \
      >"$STATUS.tmp-$$" && mv -f -- "$STATUS.tmp-$$" "$STATUS" || true
  fi
  [[ -e "$RECEIPT" ]] || write_failure_receipt "$state"
  if [[ -x "$QQ_NOTIFIER_RAW" ]]; then
    QQ_SUB="CA0803 P2 offline scoring" \
    QQ_BODY="job_id=$JOB_ID; status=$state; generation_job=$GENERATION_JOB_ID; generation_state=$GENERATION_STATE; result=$RESULT; elapsed_seconds=$((SECONDS - START_SECONDS)); attempt=$RUN_ROOT_RAW" \
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
PREREG="$(realpath -e -- "$PREREG_RAW_PATH")"
EXECUTION="$(realpath -e -- "$EXECUTION_RAW_PATH")"
PYTHON="$(realpath -e -- "$PYTHON_RAW")"
QQ_NOTIFIER="$(realpath -e -- "$QQ_NOTIFIER_RAW")"
CONTRACT_RECEIPT="$(realpath -e -- "$CONTRACT_RECEIPT_RAW")"
VALIDATION_BASE="$(realpath -e -- "$VALIDATION_BASE_RAW")"
SUBMISSION_INTENT="$(realpath -e -- "$SUBMISSION_INTENT_RAW")"
FORMAL_SUBMISSION="$FORMAL_SUBMISSION_RAW"
GENERATION_RECEIPT="$GENERATION_RECEIPT_RAW"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p2_oracle_scoring.slurm"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_preregistration.py"
SCORER="$ROOT/tools/score_qwen3vl_p2_oracle.py"
RELEASE_HELPER="$ROOT/tools/verify_qwen3vl_p2_release.py"
CANDIDATE_ROOT="$RUN_ROOT/generation"
OUTPUT="$RUN_ROOT/scoring"
RUNTIME_MANIFEST="$OUTPUT/runtime_scoring_manifest.json"
REPORT="$OUTPUT/score_report.json"
DECISION="$OUTPUT/actionability_decision.json"
RECEIPT="$RUN_ROOT/scoring_receipt.json"
LOG_DIR="$RUN_ROOT/logs"
STATUS="$LOG_DIR/scoring-status-$JOB_ID.txt"

[[ "$JOB_ID" =~ ^[0-9]+$ && "$GENERATION_JOB_ID" =~ ^[0-9]+$ && "$CONTRACT_JOB_ID" =~ ^[0-9]+$ \
    && "$JOB_NAME" == "CA0803P2Score" && "$JOB_USER" == "guisongxia" ]] \
  || { ERROR_TEXT="malformed_job_identity"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" && "${SLURMD_NODENAME:-}" == "gpu48" \
    && "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || { ERROR_TEXT="unexpected_allocation"; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT="unexpected_visible_gpu"; exit 2 ;; esac
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || { ERROR_TEXT="unexpected_gpu_allocation"; exit 2; }
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$EXPECTED_IMPLEMENTATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { ERROR_TEXT="malformed_commit_identity"; exit 2; }
for value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_GENERATION_WORKER_SHA256" \
  "$EXPECTED_VALIDATOR_SHA256" "$EXPECTED_SCORER_SHA256" "$EXPECTED_RELEASE_HELPER_SHA256" \
  "$EXPECTED_PREREG_RAW_SHA256" "$EXPECTED_PREREG_CANONICAL_SHA256" \
  "$EXPECTED_EXECUTION_RAW_SHA256" "$EXPECTED_EXECUTION_CANONICAL_SHA256" \
  "$EXPECTED_CONTRACT_RECEIPT_SHA256" "$EXPECTED_CONTRACT_WORKER_SHA256" "$EXPECTED_INTENT_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { ERROR_TEXT="malformed_expected_sha256"; exit 2; }
done
[[ -d "$LOG_DIR" && ! -e "$OUTPUT" && ! -e "$RECEIPT" && ! -e "$STATUS" ]] \
  || { ERROR_TEXT="scoring_attempt_artifact_exists"; exit 2; }
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" && -f "$PINNED_WORKER" && -f "$VALIDATOR" \
    && -f "$SCORER" && -f "$RELEASE_HELPER" ]] || { ERROR_TEXT="required_scoring_input_missing"; exit 2; }

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" \
    && "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$VALIDATOR" | awk '{print $1}')" == "$EXPECTED_VALIDATOR_SHA256" \
    && "$(sha256sum -- "$SCORER" | awk '{print $1}')" == "$EXPECTED_SCORER_SHA256" \
    && "$(sha256sum -- "$RELEASE_HELPER" | awk '{print $1}')" == "$EXPECTED_RELEASE_HELPER_SHA256" ]] \
  || { ERROR_TEXT="validator_scorer_or_helper_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$PREREG" | awk '{print $1}')" == "$EXPECTED_PREREG_RAW_SHA256" \
    && "$(sha256sum -- "$EXECUTION" | awk '{print $1}')" == "$EXPECTED_EXECUTION_RAW_SHA256" \
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
"$PYTHON" "$RELEASE_HELPER" verify-formal-submission --attempt-root "$RUN_ROOT" --role scoring \
  --job-id "$JOB_ID" --generation-job-id "$GENERATION_JOB_ID" --contract-job-id "$CONTRACT_JOB_ID" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" \
  --worker-sha256 "$EXPECTED_WORKER_SHA256" --contract-receipt "$CONTRACT_RECEIPT" \
  --contract-receipt-sha256 "$EXPECTED_CONTRACT_RECEIPT_SHA256"

GENERATION_STATE=""
for _ in $(seq 1 30); do
  GENERATION_STATE="$(sacct -n -X -P -j "$GENERATION_JOB_ID" \
    --format=JobIDRaw,State,ExitCode,JobName%64,User%32,NodeList%64 \
    | awk -F '|' -v id="$GENERATION_JOB_ID" '$1==id {print $2" "$3" "$4" "$5" "$6; exit}')"
  [[ "$GENERATION_STATE" == "COMPLETED 0:0 CA0803P2Generate guisongxia gpu48" ]] && break
  sleep 1
done
if [[ "$GENERATION_STATE" != "COMPLETED 0:0 CA0803P2Generate guisongxia gpu48" ]]; then
  RESULT="generation_dependency_not_completed"
  ERROR_TEXT="generation_job_not_completed_0"
  exit 4
fi

RESULT="verifying_generation_receipt"
GENERATION_RECEIPT="$(realpath -e -- "$GENERATION_RECEIPT")"
"$PYTHON" "$RELEASE_HELPER" verify-generation-receipt --receipt "$GENERATION_RECEIPT" \
  --candidate-root "$CANDIDATE_ROOT" --execution-manifest "$EXECUTION" \
  --formal-submission-receipt "$FORMAL_SUBMISSION" --generation-job-id "$GENERATION_JOB_ID" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" \
  --worker-sha256 "$EXPECTED_GENERATION_WORKER_SHA256" --execution-sha256 "$EXPECTED_EXECUTION_RAW_SHA256"
"$PYTHON" "$RELEASE_HELPER" verify-contract-receipt --receipt "$CONTRACT_RECEIPT" \
  --validation-base "$VALIDATION_BASE" --repo-root "$ROOT" --job-id "$CONTRACT_JOB_ID" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" \
  --worker-sha256 "$EXPECTED_CONTRACT_WORKER_SHA256" --prereg-sha256 "$EXPECTED_PREREG_RAW_SHA256" \
  --execution-sha256 "$EXPECTED_EXECUTION_RAW_SHA256"

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"
RESULT="verifying_generation_then_creating_runtime_scoring_manifest"
"$PYTHON" "$VALIDATOR" --preregistration "$PREREG" --execution-manifest "$EXECUTION" \
  --repo-root "$ROOT" --phase post_generation \
  --expected-preregistration-canonical-sha256 "$EXPECTED_PREREG_CANONICAL_SHA256" \
  --expected-execution-canonical-sha256 "$EXPECTED_EXECUTION_CANONICAL_SHA256" \
  --candidate-root "$CANDIDATE_ROOT" --expected-runtime-commit "$EXPECTED_COMMIT" \
  --runtime-scoring-output "$RUNTIME_MANIFEST"

RESULT="running_offline_scorer"
"$PYTHON" "$SCORER" --scoring-manifest "$RUNTIME_MANIFEST" --candidate-root "$CANDIDATE_ROOT" --output "$REPORT"

RESULT="building_primary_only_decision"
"$PYTHON" "$RELEASE_HELPER" build-decision --report "$REPORT" --output "$DECISION"

RESULT="verifying_repository_immutability"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_changed_during_scoring"; exit 2; }

RESULT="writing_scoring_receipt"
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$GENERATION_JOB_ID" "$EXPECTED_COMMIT" \
  "$EXPECTED_IMPLEMENTATION_COMMIT" "$ACTUAL_WORKER_SHA256" "$PREREG" "$EXECUTION" \
  "$GENERATION_RECEIPT" "$FORMAL_SUBMISSION" "$RUNTIME_MANIFEST" "$REPORT" "$DECISION" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);paths=[pathlib.Path(x) for x in sys.argv[7:14]]
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
decision=json.loads(paths[-1].read_text(encoding="utf-8"))
payload={"schema_id":"qwen3vl_p2_oracle_scoring_receipt_v2","status":"complete","job_id":sys.argv[2],
 "generation_job_id":sys.argv[3],"generation_state":"COMPLETED 0:0 CA0803P2Generate guisongxia gpu48",
 "freeze_commit_b":sys.argv[4],"implementation_commit_a":sys.argv[5],"worker_sha256":sys.argv[6],
 "resources":{"node":"gpu48","gpus":0,"cpus":2,"memory":"16G"},
 "gt_access_after_complete_candidate_validation":True,
 "candidate_flow_counts":{"requested":36,"generated":36,"tool_success":36,"scored":36},
 "proceed_to_c_iso_and_t_traj_data_construction":decision["proceed_to_c_iso_and_t_traj_data_construction"],
 "bindings":dict(zip(["preregistration","execution_manifest","generation_receipt","formal_submission_receipt",
                       "runtime_scoring_manifest","score_report","decision"],map(bind,paths)))}
temporary=target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
os.replace(temporary,target)
PY
RESULT="p2_offline_scoring_complete"
exit 0
