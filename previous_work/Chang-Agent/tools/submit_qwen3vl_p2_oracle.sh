#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
WORKSPACE="/guisongxia01/pangchao/wangyihan"
ARTIFACT_ROOT="$(realpath -e -- "$WORKSPACE/Chang-Agent")"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
PYTHON="$(realpath -e -- "$WORKSPACE/conda_envs/omniovcd-env/bin/python3.12")"
QQ_NOTIFIER="$WORKSPACE/.local/bin/wangyihan-send-qq"
DOC_ROOT="$ROOT/doc/training_pipeline/0803_fourth_try"
PREREG="$DOC_ROOT/QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION="$DOC_ROOT/QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"
VALIDATOR="$ROOT/tools/validate_qwen3vl_p2_preregistration.py"
RELEASE_HELPER="$ROOT/tools/verify_qwen3vl_p2_release.py"
CONTRACT_WORKER="$ROOT/tools/run_qwen3vl_p2_contract_validation.slurm"
GENERATION_WORKER="$ROOT/tools/run_qwen3vl_p2_oracle_generation.slurm"
SCORING_WORKER="$ROOT/tools/run_qwen3vl_p2_oracle_scoring.slurm"
ACTION_GENERATOR="$ROOT/tools/run_qwen3vl_p2_oracle.py"
SCORER="$ROOT/tools/score_qwen3vl_p2_oracle.py"
OUTPUT_BASE="$OUTPUT_ROOT/CA_0803(1)-qwen3vl-p2-oracle-actionability"
VALIDATION_BASE="$ROOT/diagnostics/legacy/qwen3vl_p2_contract_validation_0803"
EXPECTED_COMMIT_INPUT="${EXPECTED_COMMIT:-}"
PARTITION="gre"
NODE="gpu48"
JOB_USER="guisongxia"
CONTRACT_NAME="CA0803P2Contract"
GENERATION_NAME="CA0803P2Generate"
SCORING_NAME="CA0803P2Score"
MODE="validation"
CONTRACT_JOB_ARG=""
CONTRACT_RECEIPT_ARG=""

usage() {
  printf 'usage: EXPECTED_COMMIT=<freeze-B> %s [--check-only | --formal CONTRACT_JOB_ID CONTRACT_RECEIPT]\n' "$0" >&2
}

case "$#" in
  0) ;;
  1) [[ "$1" == "--check-only" ]] || { usage; exit 2; }; MODE="check_only" ;;
  3)
    [[ "$1" == "--formal" ]] || { usage; exit 2; }
    MODE="formal"; CONTRACT_JOB_ARG="$2"; CONTRACT_RECEIPT_ARG="$3"
    ;;
  *) usage; exit 2 ;;
esac

fail() { printf 'qwen3vl_p2_submit: %s\n' "$*" >&2; exit 2; }

canonical_hash() {
  "$PYTHON" - "$1" <<'PY'
import hashlib,json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
b=json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
print(hashlib.sha256(b).hexdigest())
PY
}

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

write_contract_intent() {
  "$PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" <<'PY'
import json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent
payload={
 "schema_id":"qwen3vl_p2_contract_submission_intent_v1","status":"prepared",
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),
 "freeze_commit_b":sys.argv[2],"implementation_commit_a":sys.argv[3],
 "worker_sha256":sys.argv[4],
 "manifests":{"preregistration":{"sha256":sys.argv[5],"canonical_sha256":sys.argv[6]},
              "execution":{"sha256":sys.argv[7],"canonical_sha256":sys.argv[8]}},
 "job_contract":{"name":sys.argv[9],"user":sys.argv[10],"node":sys.argv[11],
                 "partition":"gre","gpus":0,"cpus":2,"memory":"16G"}}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

write_contract_submission() {
  local target="$1" status="$2" job_id="$3" intent="$4"
  "$PYTHON" - "$target" "$status" "$job_id" "$intent" "$EXPECTED_COMMIT" \
    "$IMPLEMENTATION_COMMIT" "$CONTRACT_WORKER_SHA" "$PREREG_RAW" "$PREREG_CANONICAL" \
    "$EXECUTION_RAW" "$EXECUTION_CANONICAL" "$CONTRACT_NAME" "$JOB_USER" "$NODE" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent;intent=pathlib.Path(sys.argv[4])
def bind(path):
 return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
job={"name":sys.argv[12],"user":sys.argv[13],"node":sys.argv[14],"partition":"gre","gpus":0,"cpus":2,"memory":"16G","job_id":sys.argv[3]}
payload={"schema_id":"qwen3vl_p2_contract_submission_receipt_v1","submission_status":sys.argv[2],
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"freeze_commit_b":sys.argv[5],
 "implementation_commit_a":sys.argv[6],"worker_sha256":sys.argv[7],"job":job,
 "manifests":{"preregistration":{"sha256":sys.argv[8],"canonical_sha256":sys.argv[9]},
              "execution":{"sha256":sys.argv[10],"canonical_sha256":sys.argv[11]}},"intent":bind(intent)}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

write_formal_intent() {
  local target="$1"
  "$PYTHON" - "$target" "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT" "$CONTRACT_JOB_ARG" \
    "$CONTRACT_RECEIPT" "$CONTRACT_RECEIPT_SHA" "$PREREG_RAW" "$PREREG_CANONICAL" \
    "$EXECUTION_RAW" "$EXECUTION_CANONICAL" "$GENERATION_WORKER_SHA" "$SCORING_WORKER_SHA" \
    "$ACTION_GENERATOR_SHA" "$VALIDATOR_SHA" "$SCORER_SHA" "$RELEASE_HELPER_SHA" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent;contract=pathlib.Path(sys.argv[5])
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
if hashlib.sha256(contract.read_bytes()).hexdigest()!=sys.argv[6]: raise SystemExit("contract receipt changed before formal intent")
payload={"schema_id":"qwen3vl_p2_formal_submission_intent_v1","status":"prepared",
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"freeze_commit_b":sys.argv[2],
 "implementation_commit_a":sys.argv[3],"contract_validation":{"job_id":sys.argv[4],"receipt":bind(contract)},
 "manifests":{"preregistration":{"sha256":sys.argv[7],"canonical_sha256":sys.argv[8]},
              "execution":{"sha256":sys.argv[9],"canonical_sha256":sys.argv[10]}},
 "workers":{"generation":sys.argv[11],"scoring":sys.argv[12],"action_generator":sys.argv[13],
            "validator":sys.argv[14],"scorer":sys.argv[15],"release_helper":sys.argv[16]},
 "job_contract":{"user":"guisongxia","node":"gpu48","partition":"gre",
                 "generation_name":"CA0803P2Generate","scoring_name":"CA0803P2Score","dependency":"afterok"}}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

write_formal_submission() {
  local target="$1" status="$2" generation_job="$3" scoring_job="$4" failure_stage="$5" intent="$6"
  "$PYTHON" - "$target" "$status" "$generation_job" "$scoring_job" "$failure_stage" "$intent" \
    "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT" "$CONTRACT_JOB_ARG" "$CONTRACT_RECEIPT" \
    "$PREREG_RAW" "$PREREG_CANONICAL" "$EXECUTION_RAW" "$EXECUTION_CANONICAL" \
    "$GENERATION_WORKER_SHA" "$SCORING_WORKER_SHA" "$ACTION_GENERATOR_SHA" "$VALIDATOR_SHA" \
    "$SCORER_SHA" "$RELEASE_HELPER_SHA" <<'PY'
import hashlib,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]);attempt=target.parent;intent=pathlib.Path(sys.argv[6]);contract=pathlib.Path(sys.argv[10])
def bind(path): return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}
payload={"schema_id":"qwen3vl_p2_oracle_submission_receipt_v2","submission_status":sys.argv[2],
 "attempt_id":attempt.name,"attempt_root":str(attempt.resolve()),"freeze_commit_b":sys.argv[7],
 "implementation_commit_a":sys.argv[8],"job_user":"guisongxia","node":"gpu48","partition":"gre",
 "jobs":{"contract_validation":sys.argv[9],"generation":sys.argv[3],"scoring":sys.argv[4]},
 "dependency":("afterok:"+sys.argv[3] if sys.argv[3].isdigit() and sys.argv[4].isdigit() else None),
 "failure_stage":None if sys.argv[5]=="none" else sys.argv[5],"output":str(attempt.resolve()),
 "contract_validation":{"job_id":sys.argv[9],"receipt":bind(contract)},"intent":bind(intent),
 "manifests":{"preregistration":{"sha256":sys.argv[11],"canonical_sha256":sys.argv[12]},
              "execution":{"sha256":sys.argv[13],"canonical_sha256":sys.argv[14]}},
 "workers":{"generation":sys.argv[15],"scoring":sys.argv[16],"action_generator":sys.argv[17],
            "validator":sys.argv[18],"scorer":sys.argv[19],"release_helper":sys.argv[20]}}
with target.open("x",encoding="utf-8") as stream:
 json.dump(payload,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
PY
}

[[ "$EXPECTED_COMMIT_INPUT" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_COMMIT must be one exact lowercase commit"
EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse --verify "$EXPECTED_COMMIT_INPUT^{commit}")"
[[ "$EXPECTED_COMMIT" == "$EXPECTED_COMMIT_INPUT" ]] || fail "EXPECTED_COMMIT did not remain exact"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not exact freeze B"
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then fail "checkout must be detached"; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || fail "checkout must be clean"
[[ -L "$ROOT/outputs" ]] || fail "detached run worktree must provide an outputs symlink"
[[ "$(realpath -e -- "$ROOT/outputs")" == "$OUTPUT_ROOT" ]] || fail "outputs symlink target differs"
git -C "$ROOT" check-ignore -q -- outputs || fail "outputs symlink is not ignored"

for file in "$PYTHON" "$QQ_NOTIFIER" "$PREREG" "$EXECUTION" "$VALIDATOR" "$RELEASE_HELPER" \
  "$CONTRACT_WORKER" "$GENERATION_WORKER" "$SCORING_WORKER" "$ACTION_GENERATOR" "$SCORER"; do
  [[ -e "$file" ]] || fail "required file is missing: $file"
done
[[ -x "$PYTHON" && -x "$QQ_NOTIFIER" ]] || fail "python or QQ notifier is not executable"
[[ -z "${PYTHONOPTIMIZE:-}" ]] || fail "PYTHONOPTIMIZE is forbidden for release submission"

IMPLEMENTATION_COMMIT="$("$PYTHON" - "$PREREG" <<'PY'
import json,pathlib,sys
value=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("anchors",{}).get("implementation_commit")
if not isinstance(value,str): raise SystemExit("implementation commit missing")
print(value)
PY
)"
[[ "$IMPLEMENTATION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "implementation commit A is malformed"
# Keep a direct shell lineage check as a readable operator guard; the helper also proves single-parent/add-only.
[[ "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_COMMIT" ]] || fail "freeze B first parent is not A"
"$PYTHON" "$RELEASE_HELPER" verify-freeze --repo-root "$ROOT" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$IMPLEMENTATION_COMMIT"

PREREG_RAW="$(sha256sum -- "$PREREG" | awk '{print $1}')"
PREREG_CANONICAL="$(canonical_hash "$PREREG")"
EXECUTION_RAW="$(sha256sum -- "$EXECUTION" | awk '{print $1}')"
EXECUTION_CANONICAL="$(canonical_hash "$EXECUTION")"
"$PYTHON" "$VALIDATOR" --preregistration "$PREREG" --execution-manifest "$EXECUTION" \
  --repo-root "$ROOT" --phase structure \
  --expected-preregistration-canonical-sha256 "$PREREG_CANONICAL" \
  --expected-execution-canonical-sha256 "$EXECUTION_CANONICAL"

if ! PING="$(scontrol ping 2>&1)" || ! grep -q 'UP' <<<"$PING"; then printf '%s\n' "$PING" >&2; fail "Slurm controller is not UP"; fi
NODE_INFO="$(scontrol show node "$NODE" -o 2>&1)" || { printf '%s\n' "$NODE_INFO" >&2; fail "cannot inspect fixed node"; }
[[ "$NODE_INFO" =~ State=([^[:space:]]+) ]] || fail "fixed node state is absent"
case "${BASH_REMATCH[1]}" in *DOWN*|*DRAIN*|*FAIL*|*MAINT*|*UNKNOWN*) fail "fixed node is not usable" ;; esac
[[ "$NODE_INFO" =~ gres/gpu=([0-9]+) ]] && (( BASH_REMATCH[1] >= 1 )) || fail "fixed node has no GPU"
for name in "$CONTRACT_NAME" "$GENERATION_NAME" "$SCORING_NAME"; do
  ACTIVE="$(squeue -h -u "$JOB_USER" --name="$name" -o '%i %T')" || fail "cannot inspect duplicate job $name"
  [[ -z "$ACTIVE" ]] || fail "active duplicate job exists for $name: $ACTIVE"
done

CONTRACT_WORKER_SHA="$(sha256sum -- "$CONTRACT_WORKER" | awk '{print $1}')"
GENERATION_WORKER_SHA="$(sha256sum -- "$GENERATION_WORKER" | awk '{print $1}')"
SCORING_WORKER_SHA="$(sha256sum -- "$SCORING_WORKER" | awk '{print $1}')"
ACTION_GENERATOR_SHA="$(sha256sum -- "$ACTION_GENERATOR" | awk '{print $1}')"
VALIDATOR_SHA="$(sha256sum -- "$VALIDATOR" | awk '{print $1}')"
SCORER_SHA="$(sha256sum -- "$SCORER" | awk '{print $1}')"
RELEASE_HELPER_SHA="$(sha256sum -- "$RELEASE_HELPER" | awk '{print $1}')"

if [[ "$MODE" == "check_only" ]]; then
  printf 'status=check_only_passed\nfreeze_commit_b=%s\nimplementation_commit_a=%s\nnext=submit append-only zero-GPU validation attempt\n' "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT"
  exit 0
fi

if [[ "$MODE" == "validation" ]]; then
  ATTEMPT_ROOT="$(reserve_attempt "$VALIDATION_BASE/freeze-$EXPECTED_COMMIT")" || fail "cannot reserve validation attempt"
  INTENT="$ATTEMPT_ROOT/submission_intent.json"
  SUBMISSION="$ATTEMPT_ROOT/submission_receipt.json"
  write_contract_intent "$INTENT" "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT" "$CONTRACT_WORKER_SHA" \
    "$PREREG_RAW" "$PREREG_CANONICAL" "$EXECUTION_RAW" "$EXECUTION_CANONICAL" "$CONTRACT_NAME" "$JOB_USER" "$NODE"
  INTENT_SHA="$(sha256sum -- "$INTENT" | awk '{print $1}')"
  set +e
  SUBMIT_OUTPUT="$(sbatch --parsable --export=NIL --job-name="$CONTRACT_NAME" --partition="$PARTITION" \
    --nodelist="$NODE" --ntasks=1 --cpus-per-task=2 --mem=16G --time=01:00:00 \
    --output="$ATTEMPT_ROOT/logs/slurm-%j.out" --error="$ATTEMPT_ROOT/logs/slurm-%j.err" \
    "$CONTRACT_WORKER" "$ROOT" "$WORKSPACE" "$ARTIFACT_ROOT" "$ATTEMPT_ROOT" "$PYTHON" "$QQ_NOTIFIER" \
    "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT" "$CONTRACT_WORKER_SHA" "$RELEASE_HELPER_SHA" \
    "$PREREG_RAW" "$PREREG_CANONICAL" "$EXECUTION_RAW" "$EXECUTION_CANONICAL" "$INTENT" "$INTENT_SHA" 2>&1)"
  SUBMIT_RC="$?"; set -e
  if [[ "$SUBMIT_RC" -ne 0 ]]; then
    write_contract_submission "$SUBMISSION" "contract_submit_failed" "not_submitted" "$INTENT"
    printf '%s\n' "$SUBMIT_OUTPUT" >&2; exit "$SUBMIT_RC"
  fi
  CONTRACT_JOB="$(cut -d ';' -f 1 <<<"$SUBMIT_OUTPUT")"
  [[ "$CONTRACT_JOB" =~ ^[0-9]+$ ]] || fail "contract sbatch returned an invalid Job ID"
  write_contract_submission "$SUBMISSION" "contract_validation_submitted" "$CONTRACT_JOB" "$INTENT"
  printf 'status=contract_validation_submitted_only\njob_id=%s\nattempt=%s\nformal_jobs_submitted=0\n' "$CONTRACT_JOB" "$ATTEMPT_ROOT"
  exit 0
fi

[[ "$CONTRACT_JOB_ARG" =~ ^[0-9]+$ ]] || fail "--formal contract Job ID is malformed"
CONTRACT_RECEIPT="$(realpath -e -- "$CONTRACT_RECEIPT_ARG")"
CONTRACT_STATE="$(sacct -n -X -P -j "$CONTRACT_JOB_ARG" \
  --format=JobIDRaw,State,ExitCode,JobName%64,User%32,NodeList%64 \
  | awk -F '|' -v id="$CONTRACT_JOB_ARG" '$1==id {print $2" "$3" "$4" "$5" "$6; exit}')"
[[ "$CONTRACT_STATE" == "COMPLETED 0:0 $CONTRACT_NAME $JOB_USER $NODE" ]] || fail "contract Job metadata/state differs: $CONTRACT_STATE"
"$PYTHON" "$RELEASE_HELPER" verify-contract-receipt --receipt "$CONTRACT_RECEIPT" \
  --validation-base "$VALIDATION_BASE" --repo-root "$ROOT" --job-id "$CONTRACT_JOB_ARG" \
  --freeze-commit "$EXPECTED_COMMIT" --implementation-commit "$IMPLEMENTATION_COMMIT" \
  --worker-sha256 "$CONTRACT_WORKER_SHA" --prereg-sha256 "$PREREG_RAW" \
  --execution-sha256 "$EXECUTION_RAW" --expected-user "$JOB_USER" --expected-node "$NODE" \
  --expected-job-name "$CONTRACT_NAME"
CONTRACT_RECEIPT_SHA="$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')"

ATTEMPT_ROOT="$(reserve_attempt "$OUTPUT_BASE")" || fail "cannot reserve formal attempt"
INTENT="$ATTEMPT_ROOT/submission_intent.json"
SUBMISSION="$ATTEMPT_ROOT/submission_receipt.json"
write_formal_intent "$INTENT"
INTENT_SHA="$(sha256sum -- "$INTENT" | awk '{print $1}')"

set +e
GEN_OUTPUT="$(sbatch --parsable --export=NIL --job-name="$GENERATION_NAME" --partition="$PARTITION" \
  --nodelist="$NODE" --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=02:00:00 \
  --output="$ATTEMPT_ROOT/logs/generation-slurm-%j.out" --error="$ATTEMPT_ROOT/logs/generation-slurm-%j.err" \
  "$GENERATION_WORKER" "$ROOT" "$WORKSPACE" "$ATTEMPT_ROOT" "$EXECUTION" "$PYTHON" "$QQ_NOTIFIER" \
  "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT" "$GENERATION_WORKER_SHA" "$ACTION_GENERATOR_SHA" \
  "$RELEASE_HELPER_SHA" "$EXECUTION_RAW" "$EXECUTION_CANONICAL" "$CONTRACT_JOB_ARG" \
  "$CONTRACT_RECEIPT" "$CONTRACT_RECEIPT_SHA" "$CONTRACT_WORKER_SHA" "$PREREG_RAW" \
  "$VALIDATION_BASE" "$INTENT" "$INTENT_SHA" "$SUBMISSION" 2>&1)"
GEN_RC="$?"; set -e
if [[ "$GEN_RC" -ne 0 ]]; then
  write_formal_submission "$SUBMISSION" "generation_submit_failed" "not_submitted" "not_submitted" "generation_sbatch" "$INTENT"
  printf '%s\n' "$GEN_OUTPUT" >&2; exit "$GEN_RC"
fi
GENERATION_JOB="$(cut -d ';' -f 1 <<<"$GEN_OUTPUT")"
[[ "$GENERATION_JOB" =~ ^[0-9]+$ ]] || fail "generation sbatch returned an invalid Job ID"

set +e
SCORE_OUTPUT="$(sbatch --parsable --export=NIL --dependency="afterok:$GENERATION_JOB" \
  --job-name="$SCORING_NAME" --partition="$PARTITION" --nodelist="$NODE" \
  --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:45:00 \
  --output="$ATTEMPT_ROOT/logs/scoring-slurm-%j.out" --error="$ATTEMPT_ROOT/logs/scoring-slurm-%j.err" \
  "$SCORING_WORKER" "$ROOT" "$WORKSPACE" "$ATTEMPT_ROOT" "$PREREG" "$EXECUTION" "$PYTHON" "$QQ_NOTIFIER" \
  "$EXPECTED_COMMIT" "$IMPLEMENTATION_COMMIT" "$SCORING_WORKER_SHA" "$GENERATION_WORKER_SHA" \
  "$VALIDATOR_SHA" "$SCORER_SHA" "$RELEASE_HELPER_SHA" "$PREREG_RAW" "$PREREG_CANONICAL" \
  "$EXECUTION_RAW" "$EXECUTION_CANONICAL" "$CONTRACT_JOB_ARG" "$GENERATION_JOB" \
  "$ATTEMPT_ROOT/generation_receipt.json" "$CONTRACT_RECEIPT" "$CONTRACT_RECEIPT_SHA" \
  "$CONTRACT_WORKER_SHA" "$VALIDATION_BASE" "$INTENT" "$INTENT_SHA" "$SUBMISSION" 2>&1)"
SCORE_RC="$?"; set -e
if [[ "$SCORE_RC" -ne 0 ]]; then
  write_formal_submission "$SUBMISSION" "scoring_submit_failed_generation_already_submitted" "$GENERATION_JOB" "not_submitted" "scoring_sbatch" "$INTENT"
  printf '%s\n' "$SCORE_OUTPUT" >&2; exit "$SCORE_RC"
fi
SCORING_JOB="$(cut -d ';' -f 1 <<<"$SCORE_OUTPUT")"
[[ "$SCORING_JOB" =~ ^[0-9]+$ ]] || fail "scoring sbatch returned an invalid Job ID"
write_formal_submission "$SUBMISSION" "formal_submitted" "$GENERATION_JOB" "$SCORING_JOB" "none" "$INTENT"
printf 'status=formal_submitted\ncontract_validation_job=%s\ngeneration_job=%s\nscoring_job=%s\ndependency=afterok:%s\nattempt=%s\n' \
  "$CONTRACT_JOB_ARG" "$GENERATION_JOB" "$SCORING_JOB" "$GENERATION_JOB" "$ATTEMPT_ROOT"
