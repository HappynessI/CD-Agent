#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
WORKSPACE="/guisongxia01/pangchao/wangyihan"
ARTIFACT_ROOT="$(realpath -e -- "$WORKSPACE/Chang-Agent")"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
PYTHON="$(realpath -e -- "$WORKSPACE/conda_envs/omniovcd-env/bin/python3.12")"
QQ_NOTIFIER="$(realpath -e -- "$WORKSPACE/.local/bin/wangyihan-send-qq")"
WORKER="$ROOT/tools/run_qwen3vl_p2_closeout_validation.slurm"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_result.py"
RESULT="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.json"
ACCEPTANCE="$ROOT/doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json"
ACCEPTANCE_RELATIVE="doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json"
FORMAL_ATTEMPT="$OUTPUT_ROOT/CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001"
DIAGNOSTIC_BASE="$ROOT/diagnostics/legacy/qwen3vl_p2_closeout_validation_0803"
FREEZE_B="cda314ce9caeb6e9dd69ba0d08fe5254a8689fdd"
EXPECTED_COMMIT_INPUT="${EXPECTED_COMMIT:-}"
PARTITION="gre"
NODE="gpu48"
JOB_USER="guisongxia"
JOB_NAME="CA0803P2Closeout"
MODE="candidate"
CHECK_ONLY=0
PARENT_RECEIPT_RAW=""

usage() {
  printf 'usage: EXPECTED_COMMIT=<exact-commit> %s [--check-only | --candidate | --acceptance PARENT_CLOSEOUT_RECEIPT]\n' "$0" >&2
}

case "$#" in
  0) ;;
  1)
    case "$1" in
      --check-only) MODE="candidate"; CHECK_ONLY=1 ;;
      --candidate) MODE="candidate" ;;
      *) usage; exit 2 ;;
    esac
    ;;
  2)
    [[ "$1" == "--acceptance" ]] || { usage; exit 2; }
    MODE="acceptance"
    PARENT_RECEIPT_RAW="$2"
    ;;
  *) usage; exit 2 ;;
esac

fail() { printf 'qwen3vl_p2_closeout_submit: %s\n' "$*" >&2; exit 2; }

reserve_attempt() {
  local base="$1" number candidate
  mkdir -p -- "$base"
  [[ -d "$base" ]] || return 1
  for number in $(seq 1 9999); do
    candidate="$base/$(printf 'attempt-%04d' "$number")"
    if mkdir -- "$candidate" 2>/dev/null; then
      mkdir -- "$candidate/logs"
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

write_intent() {
  "$PYTHON" - "$1" "$MODE" "$EXPECTED_COMMIT" "$PARENT_COMMIT" "$FREEZE_B" \
    "$WORKER_SHA" "$VALIDATOR_SHA" "$FORMAL_ATTEMPT" "$PARENT_RECEIPT" \
    "$ACCEPTANCE_JSON" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent
def bind(path_text):
 if not path_text: return None
 path=pathlib.Path(path_text).resolve(strict=True)
 return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
payload={"schema_id":"qwen3vl_p2_closeout_submission_intent_v1","status":"prepared",
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"mode":sys.argv[2],
 "expected_commit":sys.argv[3],"parent_commit":sys.argv[4],"freeze_commit_b":sys.argv[5],
 "worker_sha256":sys.argv[6],"validator_sha256":sys.argv[7],
 "formal_attempt":str(pathlib.Path(sys.argv[8]).resolve(strict=True)),
 "parent_closeout_receipt":bind(sys.argv[9]),
 "acceptance_json":bind(sys.argv[10]),
 "job_contract":{"name":"CA0803P2Closeout","user":"guisongxia","node":"gpu48",
                 "partition":"gre","gpus":0,"cpus":2,"memory":"16G"}}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

write_submission() {
  "$PYTHON" - "$1" "$2" "$3" "$4" "$MODE" "$EXPECTED_COMMIT" "$PARENT_COMMIT" \
    "$FREEZE_B" "$WORKER_SHA" "$VALIDATOR_SHA" "$FORMAL_ATTEMPT" "$PARENT_RECEIPT" \
    "$ACCEPTANCE_JSON" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent;intent=pathlib.Path(sys.argv[4]).resolve(strict=True)
def bind(path): return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
def optional(path_text):
 return bind(pathlib.Path(path_text).resolve(strict=True)) if path_text else None
payload={"schema_id":"qwen3vl_p2_closeout_submission_receipt_v1","submission_status":sys.argv[2],
 "job_id":sys.argv[3],"attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"mode":sys.argv[5],
 "expected_commit":sys.argv[6],"parent_commit":sys.argv[7],"freeze_commit_b":sys.argv[8],
 "worker_sha256":sys.argv[9],"validator_sha256":sys.argv[10],
 "formal_attempt":str(pathlib.Path(sys.argv[11]).resolve(strict=True)),
 "parent_closeout_receipt":optional(sys.argv[12]),"acceptance_json":optional(sys.argv[13]),
 "job":{"name":"CA0803P2Closeout","user":"guisongxia","node":"gpu48","partition":"gre",
        "gpus":0,"cpus":2,"memory":"16G"},"intent":bind(intent)}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

verify_acceptance() {
  "$PYTHON" - "$ACCEPTANCE_JSON" "$PARENT_RECEIPT" "$PARENT_COMMIT" "$FREEZE_B" <<'PY'
import hashlib,json,os,pathlib,sys
def need(value,message):
 if not value: raise SystemExit(message)
def read(path,label):
 def pairs(items):
  out={}
  for key,value in items:
   need(key not in out,f"duplicate key in {label}: {key}");out[key]=value
  return out
 try:
  value=json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=pairs,
                   parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
 except (OSError,UnicodeError,json.JSONDecodeError,ValueError) as error:
  raise SystemExit(f"cannot read {label}: {error}") from error
 need(isinstance(value,dict),f"{label} must be an object")
 return value
def canonical(path_text,label):
 need(isinstance(path_text,str) and path_text,f"{label} path is missing")
 path=pathlib.Path(path_text);resolved=path.resolve(strict=True)
 need(pathlib.Path(os.path.abspath(path))==resolved,f"{label} path is not canonical")
 return resolved
acceptance_path=canonical(sys.argv[1],"acceptance JSON")
receipt_path=canonical(sys.argv[2],"candidate receipt")
acceptance=read(acceptance_path,"acceptance JSON")
receipt=read(receipt_path,"candidate receipt")
need(set(acceptance)=={"schema_id","status","scientific_decision_unchanged",
 "proceed_to_c_iso_and_t_traj_data_construction","candidate_closeout","next_boundary"},
 "acceptance JSON inventory differs")
need(acceptance.get("schema_id")=="qwen3vl_p2_actionability_acceptance_v1","acceptance schema differs")
need(acceptance.get("status")=="accepted","acceptance status differs")
need(acceptance.get("scientific_decision_unchanged") is True,"acceptance changed science")
need(acceptance.get("proceed_to_c_iso_and_t_traj_data_construction") is False,"acceptance proceed gate differs")
need(acceptance.get("next_boundary")=="stop_before_c_iso_and_t_traj_data_construction","acceptance boundary differs")
closeout=acceptance.get("candidate_closeout")
need(isinstance(closeout,dict) and set(closeout)=={"validated_commit","validation_mode","job_id","receipt"},
 "acceptance candidate-closeout inventory differs")
need(closeout.get("validated_commit")==sys.argv[3],"acceptance candidate commit is not D's parent")
need(closeout.get("validation_mode")=="candidate","acceptance validation mode differs")
job_id=closeout.get("job_id");need(isinstance(job_id,str) and job_id.isdecimal(),"acceptance Job ID is malformed")
binding=closeout.get("receipt")
need(isinstance(binding,dict) and set(binding)=={"path","sha256","size_bytes"},"candidate receipt binding differs")
need(canonical(binding.get("path"),"bound candidate receipt")==receipt_path,"bound candidate receipt path differs")
need(binding.get("sha256")==hashlib.sha256(receipt_path.read_bytes()).hexdigest(),"bound candidate receipt SHA256 differs")
need(type(binding.get("size_bytes")) is int and binding.get("size_bytes")==receipt_path.stat().st_size,
     "bound candidate receipt size differs")
need(receipt.get("schema_id")=="qwen3vl_p2_closeout_validation_receipt_v1","candidate receipt schema differs")
need(receipt.get("status")=="complete" and receipt.get("validation_mode")=="candidate","candidate receipt status/mode differs")
need(receipt.get("commit")==sys.argv[3] and receipt.get("parent_commit")==sys.argv[4],"candidate receipt Git lineage differs")
need(receipt.get("freeze_commit_b")==sys.argv[4],"candidate receipt freeze differs")
need(receipt.get("job_id")==job_id,"acceptance/candidate receipt Job ID differs")
PY
}

[[ "$EXPECTED_COMMIT_INPUT" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_COMMIT must be one exact lowercase commit"
EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse --verify "$EXPECTED_COMMIT_INPUT^{commit}")"
[[ "$EXPECTED_COMMIT" == "$EXPECTED_COMMIT_INPUT" ]] || fail "EXPECTED_COMMIT did not remain exact"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not exact EXPECTED_COMMIT"
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then fail "checkout must be detached"; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || fail "checkout must be clean"
[[ -L "$ROOT/outputs" ]] || fail "detached worktree must provide an ignored outputs symlink"
[[ "$(realpath -e -- "$ROOT/outputs")" == "$OUTPUT_ROOT" ]] || fail "outputs symlink target differs"
git -C "$ROOT" check-ignore -q -- outputs || fail "outputs symlink is not ignored"

for file in "$PYTHON" "$QQ_NOTIFIER" "$WORKER" "$VALIDATOR" "$RESULT" "$FORMAL_ATTEMPT"; do
  [[ -e "$file" ]] || fail "required closeout input missing: $file"
done
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" ]] || fail "python or QQ notifier is not executable"
for forbidden in PYTHONOPTIMIZE PYTHONHOME PYTHONPATH OPENAI_API_KEY HTTP_PROXY HTTPS_PROXY ALL_PROXY \
  http_proxy https_proxy all_proxy SMTP_PASSWORD SMTP_AUTH_CODE; do
  [[ ! -v "$forbidden" ]] || fail "forbidden submit environment: $forbidden"
done

PARENTS=( $(git -C "$ROOT" rev-list --parents -n 1 "$EXPECTED_COMMIT") )
[[ "${#PARENTS[@]}" -eq 2 && "${PARENTS[0]}" == "$EXPECTED_COMMIT" ]] \
  || fail "closeout commit must have exactly one parent"
PARENT_COMMIT="${PARENTS[1]}"
git -C "$ROOT" merge-base --is-ancestor "$FREEZE_B" "$EXPECTED_COMMIT" || fail "P2 freeze B is not an ancestor"
PARENT_RECEIPT=""
ACCEPTANCE_JSON=""
if [[ "$MODE" == "acceptance" ]]; then
  [[ "$PARENT_COMMIT" != "$FREEZE_B" ]] || fail "acceptance D cannot directly inherit freeze B"
  PARENT_RECEIPT="$(realpath -e -- "$PARENT_RECEIPT_RAW")"
  [[ -f "$PARENT_RECEIPT" ]] || fail "parent closeout receipt is missing"
  ACCEPTANCE_JSON="$(realpath -e -- "$ACCEPTANCE")"
  [[ -f "$ACCEPTANCE_JSON" && "$ACCEPTANCE_JSON" == "$ACCEPTANCE" ]] \
    || fail "canonical acceptance JSON is missing or non-canonical"
  [[ "$(git -C "$ROOT" cat-file -t "$EXPECTED_COMMIT:$ACCEPTANCE_RELATIVE")" == "blob" ]] \
    || fail "acceptance JSON is not a blob in exact D"
  if git -C "$ROOT" cat-file -e "$PARENT_COMMIT:$ACCEPTANCE_RELATIVE" 2>/dev/null; then
    fail "candidate C already contains the acceptance JSON"
  fi
  [[ "$(git hash-object -- "$ACCEPTANCE_JSON")" == \
      "$(git -C "$ROOT" rev-parse "$EXPECTED_COMMIT:$ACCEPTANCE_RELATIVE")" ]] \
    || fail "acceptance JSON does not match exact D"
  ACCEPTANCE_ALLOWED=(
    doc/training_pipeline/0802_third_try/QWEN3VL_SESSION_HANDOFF_0802.md
    "$ACCEPTANCE_RELATIVE"
    doc/training_pipeline/0803_fourth_try/QWEN3VL_P2_ACTIONABILITY_RESULT_0803.md
  )
  mapfile -t ACCEPTANCE_CHANGED < <(git -C "$ROOT" diff --name-only --no-renames \
    "$PARENT_COMMIT" "$EXPECTED_COMMIT")
  [[ "${#ACCEPTANCE_CHANGED[@]}" -eq "${#ACCEPTANCE_ALLOWED[@]}" ]] \
    || fail "acceptance D must change exactly the three acceptance documents"
  for expected in "${ACCEPTANCE_ALLOWED[@]}"; do
    present=0
    for changed in "${ACCEPTANCE_CHANGED[@]}"; do [[ "$changed" == "$expected" ]] && present=1; done
    [[ "$present" -eq 1 ]] || fail "acceptance D changed-file inventory differs: missing $expected"
  done
  verify_acceptance
elif [[ "$MODE" == "candidate" && "$PARENT_COMMIT" != "$FREEZE_B" ]]; then
  fail "candidate C must be a direct child of freeze B"
elif [[ "$MODE" == "candidate" && ( -e "$ACCEPTANCE" || -L "$ACCEPTANCE" ) ]]; then
  fail "candidate C must not contain an acceptance record"
elif git -C "$ROOT" cat-file -e "$EXPECTED_COMMIT:$ACCEPTANCE_RELATIVE" 2>/dev/null; then
  fail "candidate C Git tree contains the acceptance JSON"
fi

WORKER_SHA="$(sha256sum -- "$WORKER" | awk '{print $1}')"
VALIDATOR_SHA="$(sha256sum -- "$VALIDATOR" | awk '{print $1}')"
bash -n "$WORKER" "$0"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  printf 'status=check_only_passed\nmode=%s\ncommit=%s\nparent_commit=%s\n' "$MODE" "$EXPECTED_COMMIT" "$PARENT_COMMIT"
  exit 0
fi

PING="$(scontrol ping 2>&1)" || { printf '%s\n' "$PING" >&2; fail "cannot reach Slurm controller"; }
grep -q 'UP' <<<"$PING" || fail "Slurm controller is not UP"
NODE_INFO="$(scontrol show node "$NODE" -o 2>&1)" || { printf '%s\n' "$NODE_INFO" >&2; fail "cannot inspect fixed node"; }
[[ "$NODE_INFO" =~ State=([^[:space:]]+) ]] || fail "fixed node state missing"
case "${BASH_REMATCH[1]}" in *DOWN*|*DRAIN*|*FAIL*|*MAINT*|*UNKNOWN*) fail "fixed node is unusable" ;; esac
ACTIVE="$(squeue -h -u "$JOB_USER" --name="$JOB_NAME" -o '%i %T')" || fail "cannot inspect duplicate jobs"
[[ -z "$ACTIVE" ]] || fail "active duplicate closeout job exists: $ACTIVE"

ATTEMPT_ROOT="$(reserve_attempt "$DIAGNOSTIC_BASE/commit-$EXPECTED_COMMIT")" || fail "cannot reserve closeout attempt"
INTENT="$ATTEMPT_ROOT/submission_intent.json"
SUBMISSION="$ATTEMPT_ROOT/submission_receipt.json"
write_intent "$INTENT"
INTENT_SHA="$(sha256sum -- "$INTENT" | awk '{print $1}')"

set +e
SUBMIT_OUTPUT="$(sbatch --parsable --export=NIL --job-name="$JOB_NAME" --partition="$PARTITION" \
  --nodelist="$NODE" --ntasks=1 --cpus-per-task=2 --mem=16G --time=01:00:00 \
  --output="$ATTEMPT_ROOT/logs/slurm-%j.out" --error="$ATTEMPT_ROOT/logs/slurm-%j.err" \
  "$WORKER" "$ROOT" "$WORKSPACE" "$ARTIFACT_ROOT" "$ATTEMPT_ROOT" "$FORMAL_ATTEMPT" \
  "$PYTHON" "$QQ_NOTIFIER" "$MODE" "$EXPECTED_COMMIT" "$PARENT_COMMIT" "$FREEZE_B" \
  "$WORKER_SHA" "$VALIDATOR_SHA" "$INTENT" "$INTENT_SHA" "$SUBMISSION" "$PARENT_RECEIPT" \
  "$ACCEPTANCE_JSON" 2>&1)"
SUBMIT_RC="$?"
set -e
if [[ "$SUBMIT_RC" -ne 0 ]]; then
  write_submission "$SUBMISSION" "closeout_submit_failed" "not_submitted" "$INTENT"
  printf '%s\n' "$SUBMIT_OUTPUT" >&2
  exit "$SUBMIT_RC"
fi
JOB_ID="$(cut -d ';' -f 1 <<<"$SUBMIT_OUTPUT")"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fail "sbatch returned an invalid Job ID"
write_submission "$SUBMISSION" "closeout_submitted" "$JOB_ID" "$INTENT"
printf 'status=closeout_submitted\nmode=%s\njob_id=%s\nattempt=%s\ncommit=%s\n' \
  "$MODE" "$JOB_ID" "$ATTEMPT_ROOT" "$EXPECTED_COMMIT"
