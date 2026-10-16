#!/usr/bin/env bash
#SBATCH --job-name=CA0803P2Closeout
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

ROOT_RAW="${1:-}"
WORKSPACE_RAW="${2:-}"
ARTIFACT_ROOT_RAW="${3:-}"
ATTEMPT_ROOT_RAW="${4:-}"
FORMAL_ATTEMPT_RAW="${5:-}"
PYTHON_RAW="${6:-}"
QQ_NOTIFIER_RAW="${7:-}"
VALIDATION_MODE="${8:-missing}"
EXPECTED_COMMIT="${9:-missing}"
EXPECTED_PARENT_COMMIT="${10:-missing}"
FREEZE_COMMIT_B="${11:-missing}"
EXPECTED_WORKER_SHA256="${12:-missing}"
EXPECTED_VALIDATOR_SHA256="${13:-missing}"
SUBMISSION_INTENT_RAW="${14:-}"
EXPECTED_INTENT_SHA256="${15:-missing}"
SUBMISSION_RECEIPT_RAW="${16:-}"
PARENT_CLOSEOUT_RECEIPT_RAW="${17:-}"
ACCEPTANCE_JSON_RAW="${18:-}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
JOB_USER="unknown"
START_SECONDS="$SECONDS"
RESULT_STAGE="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
STATUS_FINALIZED=0
STATUS="${ATTEMPT_ROOT_RAW%/}/status-$JOB_ID.txt"
RECEIPT="${ATTEMPT_ROOT_RAW%/}/job-$JOB_ID/closeout_validation_receipt.json"
TEST_LOG="${ATTEMPT_ROOT_RAW%/}/job-$JOB_ID/validation_tests.log"

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

write_status() {
  local rc="$1" state="$2" temporary="${STATUS}.tmp-${BASHPID}"
  [[ -d "$ATTEMPT_ROOT_RAW" ]] || return 0
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\njob_user=%s\nnode=%s\npartition=%s\nvalidation_mode=%s\ncommit=%s\nparent_commit=%s\nfreeze_commit_b=%s\nworker_sha256=%s\nvalidator_sha256=%s\nresult=%s\nerror=%s\nreceipt=%s\n' \
    "$state" "$rc" "$((SECONDS - START_SECONDS))" "$JOB_ID" "$JOB_NAME" "$JOB_USER" \
    "${SLURMD_NODENAME:-missing}" "${SLURM_JOB_PARTITION:-missing}" "$VALIDATION_MODE" \
    "$EXPECTED_COMMIT" "$EXPECTED_PARENT_COMMIT" "$FREEZE_COMMIT_B" "$ACTUAL_WORKER_SHA256" \
    "$EXPECTED_VALIDATOR_SHA256" "$RESULT_STAGE" "$ERROR_TEXT" "$RECEIPT" >"$temporary"
  mv -f -- "$temporary" "$STATUS"
}

finish() {
  local rc="$?" state="failed"
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state="success"
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state="cancelled"
  [[ "$STATUS_FINALIZED" -eq 1 && "$rc" -eq 0 ]] || write_status "$rc" "$state" || true
  if [[ -x "$QQ_NOTIFIER_RAW" ]]; then
    QQ_SUB="CA0803 P2 closeout ${VALIDATION_MODE}" \
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
WORKSPACE="$(realpath -e -- "$WORKSPACE_RAW")"
ARTIFACT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT_RAW")"
ATTEMPT_ROOT="$(realpath -e -- "$ATTEMPT_ROOT_RAW")"
FORMAL_ATTEMPT="$(realpath -e -- "$FORMAL_ATTEMPT_RAW")"
PYTHON="$(realpath -e -- "$PYTHON_RAW")"
QQ_NOTIFIER="$(realpath -e -- "$QQ_NOTIFIER_RAW")"
SUBMISSION_INTENT="$(realpath -e -- "$SUBMISSION_INTENT_RAW")"
SUBMISSION_RECEIPT="$SUBMISSION_RECEIPT_RAW"
PARENT_CLOSEOUT_RECEIPT=""
if [[ -n "$PARENT_CLOSEOUT_RECEIPT_RAW" ]]; then
  PARENT_CLOSEOUT_RECEIPT="$(realpath -e -- "$PARENT_CLOSEOUT_RECEIPT_RAW")"
fi
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p2_closeout_validation.slurm"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_result.py"
RESULT_JSON="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.json"
RESULT_MD="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.md"
ACCEPTANCE_JSON_EXPECTED="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json"
ACCEPTANCE_RELATIVE="doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json"
ACCEPTANCE_JSON=""
if [[ -n "$ACCEPTANCE_JSON_RAW" ]]; then
  ACCEPTANCE_JSON="$(realpath -e -- "$ACCEPTANCE_JSON_RAW")"
fi
HANDOFF="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_SESSION_HANDOFF_0802.md"
REGISTRY="$ROOT/experiments/experiment_registry.json"
WORKBOOK="$ROOT/experiments/experiment_registry.xlsx"
PREREG="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
FORMAL_SUBMISSION="$FORMAL_ATTEMPT/submission_receipt.json"
GENERATION_RECEIPT="$FORMAL_ATTEMPT/generation_receipt.json"
SCORING_RECEIPT="$FORMAL_ATTEMPT/scoring_receipt.json"
SCORE_REPORT="$FORMAL_ATTEMPT/scoring/score_report.json"
DECISION="$FORMAL_ATTEMPT/scoring/actionability_decision.json"
STATUS="$ATTEMPT_ROOT/status-$JOB_ID.txt"
RECEIPT="$ATTEMPT_ROOT/job-$JOB_ID/closeout_validation_receipt.json"
TEST_LOG="$ATTEMPT_ROOT/job-$JOB_ID/validation_tests.log"
PRE_VALIDATION_FROZEN_SHA256=""
[[ ! -e "$ATTEMPT_ROOT/job-$JOB_ID" && ! -e "$STATUS" ]] \
  || { ERROR_TEXT="closeout_job_artifact_exists"; exit 2; }
mkdir -- "$ATTEMPT_ROOT/job-$JOB_ID"

[[ "$JOB_ID" =~ ^[0-9]+$ && "$JOB_NAME" == "CA0803P2Closeout" && "$JOB_USER" == "guisongxia" ]] \
  || { ERROR_TEXT="malformed_job_identity"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" && "${SLURMD_NODENAME:-}" == "gpu48" \
    && "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || { ERROR_TEXT="unexpected_allocation"; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT="unexpected_visible_gpu"; exit 2 ;; esac
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] \
  || { ERROR_TEXT="unexpected_gpu_allocation"; exit 2; }
[[ "$VALIDATION_MODE" == "candidate" || "$VALIDATION_MODE" == "acceptance" ]] \
  || { ERROR_TEXT="invalid_validation_mode"; exit 2; }
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && "$EXPECTED_PARENT_COMMIT" =~ ^[0-9a-f]{40}$ \
    && "$FREEZE_COMMIT_B" =~ ^[0-9a-f]{40}$ ]] || { ERROR_TEXT="malformed_commit_identity"; exit 2; }
for value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_VALIDATOR_SHA256" "$EXPECTED_INTENT_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { ERROR_TEXT="malformed_expected_sha256"; exit 2; }
done
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" && -f "$PINNED_WORKER" && -f "$VALIDATOR" \
    && -f "$RESULT_JSON" && -f "$RESULT_MD" && -f "$HANDOFF" && -f "$REGISTRY" \
    && -f "$WORKBOOK" && -f "$PREREG" && -f "$EXECUTION" && -f "$FORMAL_SUBMISSION" \
    && -f "$GENERATION_RECEIPT" && -f "$SCORING_RECEIPT" && -f "$SCORE_REPORT" && -f "$DECISION" ]] \
  || { ERROR_TEXT="required_closeout_input_missing"; exit 2; }
[[ "$FORMAL_ATTEMPT" == "$(realpath -e -- "$ARTIFACT_ROOT/outputs/CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001")" ]] \
  || { ERROR_TEXT="formal_attempt_is_not_canonical"; exit 2; }

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" \
    && "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$VALIDATOR" | awk '{print $1}')" == "$EXPECTED_VALIDATOR_SHA256" \
    && "$(sha256sum -- "$SUBMISSION_INTENT" | awk '{print $1}')" == "$EXPECTED_INTENT_SHA256" ]] \
  || { ERROR_TEXT="validator_or_intent_hash_drift"; exit 2; }

RESULT_STAGE="verifying_repository_identity"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && "$(git -C "$ROOT" rev-parse HEAD^)" == "$EXPECTED_PARENT_COMMIT" ]] \
  || { ERROR_TEXT="repository_commit_or_parent_drift"; exit 2; }
read -r -a COMMIT_PARENTS <<<"$(git -C "$ROOT" rev-list --parents -n 1 "$EXPECTED_COMMIT")"
[[ "${#COMMIT_PARENTS[@]}" -eq 2 && "${COMMIT_PARENTS[0]}" == "$EXPECTED_COMMIT" \
    && "${COMMIT_PARENTS[1]}" == "$EXPECTED_PARENT_COMMIT" ]] \
  || { ERROR_TEXT="closeout_commit_is_not_single_parent"; exit 2; }
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then ERROR_TEXT="worktree_is_not_detached"; exit 2; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_not_clean"; exit 2; }
git -C "$ROOT" merge-base --is-ancestor "$FREEZE_COMMIT_B" "$EXPECTED_COMMIT" \
  || { ERROR_TEXT="freeze_b_not_ancestor"; exit 2; }
[[ -L "$ROOT/outputs" && "$(realpath -e -- "$ROOT/outputs")" == "$(realpath -e -- "$ARTIFACT_ROOT/outputs")" ]] \
  || { ERROR_TEXT="outputs_symlink_target_drift"; exit 2; }
FROZEN_EXECUTION_FILES=(
  change_agent
  doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json
  doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.md
  doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json
  tests/test_qwen3vl_p2_oracle.py
  tests/test_qwen3vl_p2_preregistration.py
  tools/build_qwen3vl_p2_preregistration.py
  tools/run_qwen3vl_p2_contract_validation.slurm
  tools/run_qwen3vl_p2_manifest_build.slurm
  tools/run_qwen3vl_p2_oracle.py
  tools/run_qwen3vl_p2_oracle_generation.slurm
  tools/run_qwen3vl_p2_oracle_scoring.slurm
  tools/score_qwen3vl_p2_oracle.py
  tools/submit_qwen3vl_p2_manifest_build.sh
  tools/submit_qwen3vl_p2_oracle.sh
  tools/validate_qwen3vl_p2_preregistration.py
  tools/verify_qwen3vl_p2_release.py
)
git -C "$ROOT" diff --exit-code "$FREEZE_COMMIT_B" -- "${FROZEN_EXECUTION_FILES[@]}" \
  || { ERROR_TEXT="frozen_p2_execution_surface_changed"; exit 2; }

if [[ "$VALIDATION_MODE" == "acceptance" ]]; then
  [[ -n "$PARENT_CLOSEOUT_RECEIPT" && -f "$PARENT_CLOSEOUT_RECEIPT" ]] \
    || { ERROR_TEXT="parent_closeout_receipt_missing"; exit 2; }
  [[ -n "$ACCEPTANCE_JSON" && -f "$ACCEPTANCE_JSON" && "$ACCEPTANCE_JSON" == "$ACCEPTANCE_JSON_EXPECTED" ]] \
    || { ERROR_TEXT="acceptance_record_missing"; exit 2; }
  [[ "$EXPECTED_PARENT_COMMIT" != "$FREEZE_COMMIT_B" ]] \
    || { ERROR_TEXT="acceptance_directly_inherits_freeze_b"; exit 2; }
  [[ "$(git -C "$ROOT" cat-file -t "$EXPECTED_COMMIT:$ACCEPTANCE_RELATIVE")" == "blob" ]] \
    || { ERROR_TEXT="acceptance_record_not_blob_in_exact_commit"; exit 2; }
  if git -C "$ROOT" cat-file -e "$EXPECTED_PARENT_COMMIT:$ACCEPTANCE_RELATIVE" 2>/dev/null; then
    ERROR_TEXT="candidate_parent_contains_acceptance_record"; exit 2
  fi
  [[ "$(git hash-object -- "$ACCEPTANCE_JSON")" == \
      "$(git -C "$ROOT" rev-parse "$EXPECTED_COMMIT:$ACCEPTANCE_RELATIVE")" ]] \
    || { ERROR_TEXT="acceptance_record_does_not_match_exact_commit"; exit 2; }
  ACCEPTANCE_ALLOWED=(
    doc/training_pipeline/0802_third_try/QWEN3VL_SESSION_HANDOFF_0802.md
    doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json
    doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.md
  )
  mapfile -t ACCEPTANCE_CHANGED < <(git -C "$ROOT" diff --name-only --no-renames "$EXPECTED_PARENT_COMMIT" "$EXPECTED_COMMIT")
  [[ "${#ACCEPTANCE_CHANGED[@]}" -eq "${#ACCEPTANCE_ALLOWED[@]}" ]] \
    || { ERROR_TEXT="acceptance_diff_inventory_mismatch"; exit 2; }
  for changed in "${ACCEPTANCE_CHANGED[@]}"; do
    allowed=0
    for expected in "${ACCEPTANCE_ALLOWED[@]}"; do
      [[ "$changed" == "$expected" ]] && allowed=1
    done
    [[ "$allowed" -eq 1 ]] || { ERROR_TEXT="acceptance_diff_outside_allowlist_$changed"; exit 2; }
  done
  for expected in "${ACCEPTANCE_ALLOWED[@]}"; do
    present=0
    for changed in "${ACCEPTANCE_CHANGED[@]}"; do
      [[ "$changed" == "$expected" ]] && present=1
    done
    [[ "$present" -eq 1 ]] || { ERROR_TEXT="acceptance_required_change_missing_$expected"; exit 2; }
  done
else
  [[ "$EXPECTED_PARENT_COMMIT" == "$FREEZE_COMMIT_B" ]] \
    || { ERROR_TEXT="candidate_is_not_direct_child_of_freeze_b"; exit 2; }
  [[ -z "$PARENT_CLOSEOUT_RECEIPT" ]] \
    || { ERROR_TEXT="candidate_received_parent_closeout_receipt"; exit 2; }
  [[ -z "$ACCEPTANCE_JSON" && ! -e "$ACCEPTANCE_JSON_EXPECTED" && ! -L "$ACCEPTANCE_JSON_EXPECTED" ]] \
    || { ERROR_TEXT="candidate_contains_acceptance_record"; exit 2; }
  if git -C "$ROOT" cat-file -e "$EXPECTED_COMMIT:$ACCEPTANCE_RELATIVE" 2>/dev/null; then
    ERROR_TEXT="candidate_git_tree_contains_acceptance_record"; exit 2
  fi
fi

RESULT_STAGE="capturing_frozen_closeout_files"
PRE_VALIDATION_FROZEN_SHA256="$(sha256sum -- "$RESULT_JSON" "$REGISTRY" "$WORKBOOK" | sha256sum | awk '{print $1}')"

RESULT_STAGE="waiting_for_append_only_submission_receipt"
for _ in $(seq 1 30); do [[ -f "$SUBMISSION_RECEIPT" ]] && break; sleep 1; done
[[ -f "$SUBMISSION_RECEIPT" ]] || { ERROR_TEXT="submission_receipt_missing"; exit 2; }
SUBMISSION_RECEIPT="$(realpath -e -- "$SUBMISSION_RECEIPT")"
"$PYTHON" - "$SUBMISSION_INTENT" "$SUBMISSION_RECEIPT" "$ATTEMPT_ROOT" "$JOB_ID" \
  "$VALIDATION_MODE" "$EXPECTED_COMMIT" "$EXPECTED_PARENT_COMMIT" "$FREEZE_COMMIT_B" \
  "$ACTUAL_WORKER_SHA256" "$EXPECTED_VALIDATOR_SHA256" "$FORMAL_ATTEMPT" \
  "$PARENT_CLOSEOUT_RECEIPT" "$ACCEPTANCE_JSON" <<'PY'
import hashlib,json,os,pathlib,sys
def need(value,message):
 if not value: raise SystemExit(message)
def read(path):
 def pairs(items):
  out={}
  for key,value in items:
   need(key not in out,f"duplicate key: {key}");out[key]=value
  return out
 return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=pairs,
                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
intent_path,receipt_path,attempt=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]).resolve()
intent,receipt=read(intent_path),read(receipt_path)
def canonical(path_text,label):
 need(isinstance(path_text,str) and path_text,f"{label} path missing")
 path=pathlib.Path(path_text);resolved=path.resolve(strict=True)
 need(pathlib.Path(os.path.abspath(path))==resolved,f"{label} path is non-canonical")
 return resolved
def expected_binding(path_text):
 if not path_text: return None
 path=canonical(path_text,"expected binding")
 return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
need(intent.get("schema_id")=="qwen3vl_p2_closeout_submission_intent_v1" and intent.get("status")=="prepared","intent schema/status differs")
need(receipt.get("schema_id")=="qwen3vl_p2_closeout_submission_receipt_v1" and receipt.get("submission_status")=="closeout_submitted","receipt schema/status differs")
for payload in (intent,receipt):
 need(payload.get("attempt_id")==attempt.name and pathlib.Path(payload.get("attempt_root","")).resolve()==attempt,"attempt identity differs")
 need(payload.get("mode")==sys.argv[5] and payload.get("expected_commit")==sys.argv[6],"mode/commit differs")
 need(payload.get("parent_commit")==sys.argv[7] and payload.get("freeze_commit_b")==sys.argv[8],"parent/freeze differs")
 need(payload.get("worker_sha256")==sys.argv[9] and payload.get("validator_sha256")==sys.argv[10],"worker/validator differs")
formal=canonical(sys.argv[11],"formal attempt")
need(canonical(intent.get("formal_attempt"),"intent formal attempt")==formal,"intent formal attempt differs")
need(canonical(receipt.get("formal_attempt"),"receipt formal attempt")==formal,"receipt formal attempt differs")
need(receipt.get("job_id")==sys.argv[4],"submitted Job ID differs")
job=receipt.get("job",{});need(job=={"name":"CA0803P2Closeout","user":"guisongxia","node":"gpu48","partition":"gre","gpus":0,"cpus":2,"memory":"16G"},"job contract differs")
need(intent.get("job_contract")==job,"intent job contract differs")
binding=receipt.get("intent",{});need(pathlib.Path(binding.get("path","")).resolve()==intent_path.resolve(),"intent path differs")
need(binding.get("sha256")==hashlib.sha256(intent_path.read_bytes()).hexdigest() and binding.get("size_bytes")==intent_path.stat().st_size,"intent binding differs")
parent_binding=expected_binding(sys.argv[12]);acceptance_binding=expected_binding(sys.argv[13])
need(intent.get("parent_closeout_receipt")==parent_binding and receipt.get("parent_closeout_receipt")==parent_binding,
     "parent receipt binding differs")
need(intent.get("acceptance_json")==acceptance_binding and receipt.get("acceptance_json")==acceptance_binding,
     "acceptance JSON binding differs")
need((sys.argv[5]=="candidate" and parent_binding is None and acceptance_binding is None) or
     (sys.argv[5]=="acceptance" and parent_binding is not None and acceptance_binding is not None),
     "mode-specific closeout bindings differ")
PY

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"

RESULT_STAGE="validating_result_and_formal_evidence"
VALIDATOR_ARGS=(--result "$RESULT_JSON" --repo-root "$ROOT" --formal-attempt "$FORMAL_ATTEMPT")
if [[ "$VALIDATION_MODE" == "acceptance" ]]; then
  VALIDATOR_ARGS+=(--parent-closeout-receipt "$PARENT_CLOSEOUT_RECEIPT" \
    --expected-parent-commit "$EXPECTED_PARENT_COMMIT")
fi
"$PYTHON" "$VALIDATOR" "${VALIDATOR_ARGS[@]}" >"$TEST_LOG" 2>&1

RESULT_STAGE="checking_workbook_determinism"
"$PYTHON" experiments/build_experiment_workbook.py --check >>"$TEST_LOG" 2>&1
RESULT_STAGE="running_p2_result_targeted_tests"
"$PYTHON" -m unittest tests.test_qwen3vl_p2_result >>"$TEST_LOG" 2>&1
RESULT_STAGE="running_registry_targeted_tests"
"$PYTHON" -m unittest tests.test_experiment_registry >>"$TEST_LOG" 2>&1
RESULT_STAGE="checking_closeout_shell_syntax"
bash -n tools/run_qwen3vl_p2_closeout_validation.slurm \
  tools/run_qwen3vl_p2_registry_prepare.slurm tools/submit_qwen3vl_p2_closeout.sh \
  >>"$TEST_LOG" 2>&1
RESULT_STAGE="running_full_regression"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' >>"$TEST_LOG" 2>&1

RESULT_STAGE="verifying_repository_immutability"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="repository_changed_during_validation"; exit 2; }
[[ "$(sha256sum -- "$RESULT_JSON" "$REGISTRY" "$WORKBOOK" | sha256sum | awk '{print $1}')" \
    == "$PRE_VALIDATION_FROZEN_SHA256" ]] \
  || { ERROR_TEXT="frozen_closeout_files_changed_during_validation"; exit 2; }

RESULT_STAGE="p2_closeout_validation_complete"
write_status 0 success
STATUS_FINALIZED=1
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$JOB_NAME" "$JOB_USER" "$VALIDATION_MODE" \
  "$EXPECTED_COMMIT" "$EXPECTED_PARENT_COMMIT" "$FREEZE_COMMIT_B" "$ACTUAL_WORKER_SHA256" \
  "$EXPECTED_VALIDATOR_SHA256" "$RESULT_JSON" "$RESULT_MD" "$HANDOFF" "$REGISTRY" "$WORKBOOK" \
  "$FORMAL_SUBMISSION" "$GENERATION_RECEIPT" "$SCORING_RECEIPT" "$SCORE_REPORT" "$DECISION" \
  "$STATUS" "$SUBMISSION_INTENT" "$SUBMISSION_RECEIPT" "$TEST_LOG" "$PARENT_CLOSEOUT_RECEIPT" \
  "$ACCEPTANCE_JSON" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);paths=[pathlib.Path(value) for value in sys.argv[11:25]]
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
keys=["result_json","result_markdown","handoff","registry","workbook","formal_submission_receipt",
      "generation_receipt","scoring_receipt","score_report","decision","status","submission_intent",
      "submission_receipt","test_log"]
bindings=dict(zip(keys,map(bind,paths)))
if sys.argv[25]: bindings["parent_closeout_receipt"]=bind(pathlib.Path(sys.argv[25]))
if sys.argv[26]: bindings["acceptance_json"]=bind(pathlib.Path(sys.argv[26]))
payload={"schema_id":"qwen3vl_p2_closeout_validation_receipt_v1","status":"complete",
 "job_id":sys.argv[2],"job_name":sys.argv[3],"job_user":sys.argv[4],"node":"gpu48","partition":"gre",
 "validation_mode":sys.argv[5],"commit":sys.argv[6],"parent_commit":sys.argv[7],"freeze_commit_b":sys.argv[8],
 "worker_sha256":sys.argv[9],"validator_sha256":sys.argv[10],
 "resources":{"gpus":0,"cpus":2,"memory":"16G"},
 "checks":{"clean_detached_exact_commit":True,"freeze_b_ancestor":True,"frozen_manifests_unchanged":True,
           "frozen_execution_surface_unchanged":True,
           "formal_evidence_and_result":"passed","negative_gate_and_no_data":"passed",
           "workbook_check":"passed","p2_result_targeted_tests":"passed","registry_targeted_tests":"passed",
           "shell_syntax":"passed","full_regression":"passed","repository_unchanged":True},
 "bindings":bindings}
temporary=target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
os.replace(temporary,target)
PY
exit 0
