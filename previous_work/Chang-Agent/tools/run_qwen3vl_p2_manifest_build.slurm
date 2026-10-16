#!/usr/bin/env bash
#SBATCH --job-name=CA0803P2Manifest
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
TARGET_REPO_RAW="${2:-}"
WORKSPACE_RAW="${3:-}"
ATTEMPT_ROOT_RAW="${4:-}"
MANIFEST_ROOT_RAW="${5:-}"
STAGING_ROOT_RAW="${6:-}"
LOCK_FILE_RAW="${7:-}"
PYTHON_RAW="${8:-}"
QQ_NOTIFIER_RAW="${9:-}"
EXPECTED_COMMIT="${10:-missing}"
EXPECTED_WORKER_SHA256="${11:-missing}"
EXPECTED_BUILDER_SHA256="${12:-missing}"
EXPECTED_VALIDATOR_SHA256="${13:-missing}"
EXPECTED_RELEASE_HELPER_SHA256="${14:-missing}"
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
RECEIPT="${ATTEMPT_ROOT_RAW%/}/materialization_receipt.json"
STAGED_PREREG="${STAGING_ROOT_RAW%/}/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
STAGED_EXECUTION="${STAGING_ROOT_RAW%/}/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
PREREG="${MANIFEST_ROOT_RAW%/}/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION="${MANIFEST_ROOT_RAW%/}/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"

record_error() { local rc="$?"; [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"; return "$rc"; }

write_status() {
  local rc="$1" state="$2"
  [[ -d "$ATTEMPT_ROOT_RAW" ]] || return 0
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\njob_user=%s\nimplementation_commit_a=%s\nworker_sha256=%s\nresult=%s\nerror=%s\nreceipt=%s\n' \
    "$state" "$rc" "$((SECONDS - START_SECONDS))" "$JOB_ID" "$JOB_NAME" "$JOB_USER" "$EXPECTED_COMMIT" \
    "$ACTUAL_WORKER_SHA256" "$RESULT" "$ERROR_TEXT" "$RECEIPT" >"$STATUS.tmp-$$"
  mv -f -- "$STATUS.tmp-$$" "$STATUS"
}

cleanup_partial_publication() {
  [[ -x "$PYTHON_RAW" && -d "$TARGET_REPO_RAW" ]] || return 0
  "$PYTHON_RAW" - "$TARGET_REPO_RAW" "$EXPECTED_COMMIT" "$STAGED_PREREG" "$STAGED_EXECUTION" "$PREREG" "$EXECUTION" <<'PY' || true
import hashlib,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);expected=sys.argv[2]
staged=[pathlib.Path(sys.argv[3]),pathlib.Path(sys.argv[4])];published=[pathlib.Path(sys.argv[5]),pathlib.Path(sys.argv[6])]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
head=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],text=True,capture_output=True).stdout.strip()
branch=subprocess.run(["git","-C",str(repo),"branch","--show-current"],text=True,capture_output=True).stdout.strip()
if head!=expected or branch!="feat/local-qwen3vl-p2-oracle-0803": raise SystemExit(0)
for source,target in zip(staged,published):
 flag=source.parent/f".published-{target.name}"
 tracked=subprocess.run(["git","-C",str(repo),"ls-files","--error-unmatch","--",str(target.relative_to(repo))],capture_output=True).returncode==0
 if flag.is_file() and target.is_file() and source.is_file() and not tracked and sha(target)==sha(source): target.unlink()
 if flag.is_file(): flag.unlink()
for target in published:
 for temporary in target.parent.glob(f".{target.name}.publish-{sys.argv[2][:12]}-*"):
  if temporary.is_file(): temporary.unlink()
PY
}

finish() {
  local rc="$?" state="failed"
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state="success"
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state="cancelled"
  if [[ "$rc" -ne 0 && ! -e "$RECEIPT" ]]; then cleanup_partial_publication; fi
  [[ "$STATUS_FINALIZED" -eq 1 && "$rc" -eq 0 ]] || write_status "$rc" "$state" || true
  if [[ -x "$QQ_NOTIFIER_RAW" ]]; then
    QQ_SUB="CA0803 P2 manifest materialization" \
    QQ_BODY="job_id=$JOB_ID; status=$state; result=$RESULT; elapsed_seconds=$((SECONDS - START_SECONDS)); attempt=$ATTEMPT_ROOT_RAW; target=$MANIFEST_ROOT_RAW" \
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
TARGET_REPO="$(realpath -e -- "$TARGET_REPO_RAW")"
WORKSPACE="$(realpath -e -- "$WORKSPACE_RAW")"
ATTEMPT_ROOT="$(realpath -e -- "$ATTEMPT_ROOT_RAW")"
MANIFEST_ROOT="$(realpath -e -- "$MANIFEST_ROOT_RAW")"
STAGING_ROOT="$(realpath -e -- "$STAGING_ROOT_RAW")"
LOCK_FILE="$(realpath -m -- "$LOCK_FILE_RAW")"
PYTHON="$(realpath -e -- "$PYTHON_RAW")"
QQ_NOTIFIER="$(realpath -e -- "$QQ_NOTIFIER_RAW")"
SUBMISSION_INTENT="$(realpath -e -- "$SUBMISSION_INTENT_RAW")"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p2_manifest_build.slurm"
BUILDER="$ROOT/tools/build_qwen3vl_p2_preregistration.py"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_preregistration.py"
RELEASE_HELPER="$ROOT/tools/verify_qwen3vl_p2_release.py"
SUBMISSION_RECEIPT="$ATTEMPT_ROOT/submission_receipt.json"
STATUS="$ATTEMPT_ROOT/status-$JOB_ID.txt"
RECEIPT="$ATTEMPT_ROOT/materialization_receipt.json"
STAGED_PREREG="$STAGING_ROOT/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
STAGED_EXECUTION="$STAGING_ROOT/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
PREREG="$MANIFEST_ROOT/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION="$MANIFEST_ROOT/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"

[[ "$JOB_ID" =~ ^[0-9]+$ && "$JOB_NAME" == "CA0803P2Manifest" && "$JOB_USER" == "guisongxia" ]] \
  || { ERROR_TEXT="unexpected_job_identity"; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == "gre" && "${SLURMD_NODENAME:-}" == "gpu48" \
    && "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || { ERROR_TEXT="unexpected_allocation"; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT="unexpected_visible_gpu"; exit 2 ;; esac
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || { ERROR_TEXT="unexpected_gpu_allocation"; exit 2; }
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { ERROR_TEXT="malformed_implementation_commit"; exit 2; }
for value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_BUILDER_SHA256" "$EXPECTED_VALIDATOR_SHA256" \
  "$EXPECTED_RELEASE_HELPER_SHA256" "$EXPECTED_INTENT_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { ERROR_TEXT="malformed_expected_sha256"; exit 2; }
done
[[ "$ATTEMPT_ROOT" != "$ROOT" && "$ATTEMPT_ROOT" != "$ROOT/"* \
    && "$ATTEMPT_ROOT" != "$TARGET_REPO" && "$ATTEMPT_ROOT" != "$TARGET_REPO/"* \
    && "$STAGING_ROOT" == "$ATTEMPT_ROOT/staging" ]] || { ERROR_TEXT="attempt_or_staging_location_invalid"; exit 2; }
[[ "$LOCK_FILE" == "$(realpath -m -- "$ATTEMPT_ROOT/../locks/materialization.lock")" \
    && "$LOCK_FILE" != "$ROOT/"* && "$LOCK_FILE" != "$TARGET_REPO/"* ]] \
  || { ERROR_TEXT="materialization_lock_location_invalid"; exit 2; }
[[ "$MANIFEST_ROOT" == "$TARGET_REPO/doc/training_pipeline/0803_fourth_try" ]] \
  || { ERROR_TEXT="manifest_root_is_not_target_doc_directory"; exit 2; }
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" && -f "$PINNED_WORKER" && -f "$BUILDER" \
    && -f "$VALIDATOR" && -f "$RELEASE_HELPER" ]] || { ERROR_TEXT="required_materialization_input_missing"; exit 2; }
[[ ! -e "$PREREG" && ! -e "$EXECUTION" && ! -e "$STAGED_PREREG" && ! -e "$STAGED_EXECUTION" \
    && ! -e "$RECEIPT" && ! -e "$STATUS" ]] || { ERROR_TEXT="attempt_or_manifest_artifact_exists"; exit 2; }

exec 9>>"$LOCK_FILE"
flock -n 9 || { ERROR_TEXT="materialization_lock_busy"; exit 2; }

EXECUTED_WORKER="$(realpath -e -- "${BASH_SOURCE[0]}")"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" \
    && "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || { ERROR_TEXT="worker_hash_drift"; exit 2; }
[[ "$(sha256sum -- "$BUILDER" | awk '{print $1}')" == "$EXPECTED_BUILDER_SHA256" \
    && "$(sha256sum -- "$VALIDATOR" | awk '{print $1}')" == "$EXPECTED_VALIDATOR_SHA256" \
    && "$(sha256sum -- "$RELEASE_HELPER" | awk '{print $1}')" == "$EXPECTED_RELEASE_HELPER_SHA256" \
    && "$(sha256sum -- "$SUBMISSION_INTENT" | awk '{print $1}')" == "$EXPECTED_INTENT_SHA256" ]] \
  || { ERROR_TEXT="builder_validator_helper_or_intent_hash_drift"; exit 2; }

RESULT="waiting_for_append_only_submission_receipt"
for _ in $(seq 1 30); do [[ -f "$SUBMISSION_RECEIPT" ]] && break; sleep 1; done
[[ -f "$SUBMISSION_RECEIPT" ]] || { ERROR_TEXT="submission_receipt_missing"; exit 2; }
"$PYTHON" - "$SUBMISSION_INTENT" "$SUBMISSION_RECEIPT" "$ATTEMPT_ROOT" "$JOB_ID" \
  "$EXPECTED_COMMIT" "$EXPECTED_WORKER_SHA256" "$ROOT" "$TARGET_REPO" "$MANIFEST_ROOT" \
  "$STAGING_ROOT" "$LOCK_FILE" "$EXPECTED_BUILDER_SHA256" "$EXPECTED_VALIDATOR_SHA256" \
  "$EXPECTED_RELEASE_HELPER_SHA256" <<'PY'
import hashlib,json,pathlib,sys
intent_path,receipt_path=(pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]));attempt=pathlib.Path(sys.argv[3]).resolve()
intent=json.loads(intent_path.read_text(encoding="utf-8"));receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
def need(condition,message):
 if not condition: raise SystemExit(message)
need(intent.get("schema_id")=="qwen3vl_p2_manifest_submission_intent_v1" and intent.get("status")=="prepared","manifest intent differs")
need(intent.get("attempt_id")==attempt.name and pathlib.Path(intent.get("attempt_root","")).resolve()==attempt,"manifest intent attempt differs")
need(intent.get("implementation_commit_a")==sys.argv[5],"manifest intent commit differs")
need(pathlib.Path(intent.get("source_repo","")).resolve()==pathlib.Path(sys.argv[7]).resolve(),"manifest intent source differs")
need(pathlib.Path(intent.get("target_repo","")).resolve()==pathlib.Path(sys.argv[8]).resolve(),"manifest intent target differs")
need(pathlib.Path(intent.get("manifest_root","")).resolve()==pathlib.Path(sys.argv[9]).resolve(),"manifest intent output differs")
need(pathlib.Path(intent.get("staging_root","")).resolve()==pathlib.Path(sys.argv[10]).resolve(),"manifest intent staging differs")
need(pathlib.Path(intent.get("lock_file","")).resolve()==pathlib.Path(sys.argv[11]).resolve(),"manifest intent lock differs")
need(intent.get("workers")=={"materializer":sys.argv[6],"builder":sys.argv[12],"validator":sys.argv[13],"release_helper":sys.argv[14]},"manifest intent workers differ")
need(receipt.get("schema_id")=="qwen3vl_p2_manifest_submission_receipt_v1" and receipt.get("submission_status")=="materialization_submitted","manifest submission differs")
need(receipt.get("attempt_id")==attempt.name and pathlib.Path(receipt.get("attempt_root","")).resolve()==attempt,"manifest receipt attempt differs")
need(receipt.get("implementation_commit_a")==sys.argv[5] and receipt.get("worker_sha256")==sys.argv[6],"manifest receipt binding differs")
expected_job={"job_id":sys.argv[4],"name":"CA0803P2Manifest","user":"guisongxia","node":"gpu48","partition":"gre","gpus":0,"cpus":2,"memory":"16G"}
need(receipt.get("job")==expected_job,"manifest submitted job differs")
binding=receipt.get("intent",{});need(binding.get("path")==str(intent_path.resolve()),"manifest receipt intent path differs")
need(binding.get("sha256")==hashlib.sha256(intent_path.read_bytes()).hexdigest() and binding.get("size_bytes")==intent_path.stat().st_size,"manifest receipt intent hash differs")
PY

[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { ERROR_TEXT="source_commit_drift"; exit 2; }
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then ERROR_TEXT="source_worktree_not_detached"; exit 2; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { ERROR_TEXT="source_worktree_not_clean"; exit 2; }
[[ "$(git -C "$TARGET_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && "$(git -C "$TARGET_REPO" branch --show-current)" == "feat/local-qwen3vl-p2-oracle-0803" \
    && -z "$(git -C "$TARGET_REPO" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="target_worktree_precondition_drift"; exit 2; }

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"
RESULT="building_two_manifests_in_external_staging"
"$PYTHON" "$BUILDER" --repo-root "$ROOT" --implementation-commit "$EXPECTED_COMMIT" \
  --preregistration-output "$STAGED_PREREG" --execution-output "$STAGED_EXECUTION"
RESULT="validating_staged_manifests"
"$PYTHON" "$VALIDATOR" --preregistration "$STAGED_PREREG" --execution-manifest "$STAGED_EXECUTION" \
  --repo-root "$ROOT" --phase structure

RESULT="rechecking_both_worktrees_before_publication"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="source_changed_during_staging"; exit 2; }
[[ "$(git -C "$TARGET_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && "$(git -C "$TARGET_REPO" branch --show-current)" == "feat/local-qwen3vl-p2-oracle-0803" \
    && -z "$(git -C "$TARGET_REPO" status --porcelain --untracked-files=all)" ]] \
  || { ERROR_TEXT="target_changed_during_staging"; exit 2; }

RESULT="publishing_validated_manifests_with_rollback_guard"
"$PYTHON" - "$STAGED_PREREG" "$STAGED_EXECUTION" "$PREREG" "$EXECUTION" "$EXPECTED_COMMIT" "$JOB_ID" <<'PY'
import hashlib,os,pathlib,shutil,sys
sources=[pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])];targets=[pathlib.Path(sys.argv[3]),pathlib.Path(sys.argv[4])]
temporaries=[target.with_name(f".{target.name}.publish-{sys.argv[5][:12]}-{sys.argv[6]}") for target in targets]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for source,target,temporary in zip(sources,targets,temporaries):
 if target.exists() or temporary.exists(): raise SystemExit("publication target already exists")
 with source.open("rb") as incoming,temporary.open("xb") as outgoing:
  shutil.copyfileobj(incoming,outgoing);outgoing.flush();os.fsync(outgoing.fileno())
 if sha(source)!=sha(temporary): raise SystemExit("publication staging copy hash differs")
for source,temporary,target in zip(sources,temporaries,targets):
 os.link(temporary,target)
 temporary.unlink()
 flag=source.parent/f".published-{target.name}"
 with flag.open("x",encoding="utf-8") as stream:
  stream.write(str(target.resolve())+"\n");stream.flush();os.fsync(stream.fileno())
directory_fd=os.open(str(targets[0].parent),os.O_RDONLY)
try: os.fsync(directory_fd)
finally: os.close(directory_fd)
PY

RESULT="verifying_final_target_state"
EXPECTED_STATUS=$'?? doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json\n?? doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json'
[[ "$(git -C "$TARGET_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" \
    && "$(git -C "$TARGET_REPO" branch --show-current)" == "feat/local-qwen3vl-p2-oracle-0803" \
    && "$(git -C "$TARGET_REPO" status --porcelain --untracked-files=all)" == "$EXPECTED_STATUS" ]] \
  || { ERROR_TEXT="target_final_state_differs"; exit 2; }
[[ "$(sha256sum -- "$PREREG" | awk '{print $1}')" == "$(sha256sum -- "$STAGED_PREREG" | awk '{print $1}')" \
    && "$(sha256sum -- "$EXECUTION" | awk '{print $1}')" == "$(sha256sum -- "$STAGED_EXECUTION" | awk '{print $1}')" ]] \
  || { ERROR_TEXT="published_manifest_hash_drift"; exit 2; }

RESULT="p2_manifest_materialization_complete"
write_status 0 success
STATUS_FINALIZED=1
"$PYTHON" - "$RECEIPT" "$JOB_ID" "$JOB_NAME" "$JOB_USER" "$EXPECTED_COMMIT" "$ACTUAL_WORKER_SHA256" \
  "$ROOT" "$TARGET_REPO" "$STATUS" "$SUBMISSION_INTENT" "$SUBMISSION_RECEIPT" \
  "$BUILDER" "$VALIDATOR" "$RELEASE_HELPER" "$STAGED_PREREG" "$STAGED_EXECUTION" "$PREREG" "$EXECUTION" <<'PY'
import hashlib,json,os,pathlib,subprocess,sys
target=pathlib.Path(sys.argv[1]);paths=[pathlib.Path(x) for x in sys.argv[9:19]]
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
target_repo=pathlib.Path(sys.argv[8]);head=subprocess.run(["git","-C",str(target_repo),"rev-parse","HEAD"],check=True,text=True,capture_output=True).stdout.strip()
branch=subprocess.run(["git","-C",str(target_repo),"branch","--show-current"],check=True,text=True,capture_output=True).stdout.strip()
payload={"schema_id":"qwen3vl_p2_manifest_materialization_receipt_v2","status":"complete",
 "job_id":sys.argv[2],"job_name":sys.argv[3],"job_user":sys.argv[4],"node":"gpu48","partition":"gre",
 "implementation_commit_a":sys.argv[5],"worker_sha256":sys.argv[6],"resources":{"gpus":0,"cpus":2,"memory":"16G"},
 "label_file_accessed":False,"source_repository":{"path":str(pathlib.Path(sys.argv[7]).resolve()),"head":sys.argv[5],"detached":True,"clean":True},
 "target_repository":{"path":str(target_repo.resolve()),"head":head,"branch":branch,"only_two_untracked_manifests":True},
 "bindings":dict(zip(["status","submission_intent","submission_receipt","builder","validator","release_helper",
                       "staged_preregistration","staged_execution_manifest","preregistration","execution_manifest"],map(bind,paths)))}
temporary=target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
os.replace(temporary,target)
PY
exit 0
