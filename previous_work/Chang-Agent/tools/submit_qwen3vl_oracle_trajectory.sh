#!/usr/bin/env bash
set -Eeuo pipefail
umask 027
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

MODE="${1:-}"
[[ "$MODE" == contract || "$MODE" == pilot ]] \
  || { echo "usage: $0 {contract|pilot}" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(realpath -e -- /guisongxia01/pangchao/wangyihan)"
PRIMARY_REPO="$(realpath -e -- "$WORKSPACE/Chang-Agent")"
OUTPUT_ROOT="$(realpath -e -- "$PRIMARY_REPO/outputs")"
CAMPAIGN_ROOT="$OUTPUT_ROOT/CA_0803(2)-qwen3vl-oracle-trajectory-pilot"
CONFIG="$(realpath -e -- "$ROOT/configs/qwen3vl_oracle_trajectory_pilot_0803.json")"
PYTHON="$(realpath -e -- "$WORKSPACE/conda_envs/omniovcd-env/bin/python3.12")"
QQ_NOTIFIER="$(realpath -e -- "$WORKSPACE/.local/bin/wangyihan-send-qq")"
NODE=gpu48
PARTITION=gre
FREEZE_B="$(git -C "$ROOT" rev-parse HEAD)"
IMPLEMENTATION_A="$(git -C "$ROOT" rev-parse HEAD^)"
CONFIG_SHA256="$(sha256sum -- "$CONFIG" | awk '{print $1}')"
DIAGNOSTICS_BASE="$ROOT/diagnostics/legacy/qwen3vl_oracle_trajectory_contract_0803"
DIAGNOSTICS_FREEZE_ROOT="$DIAGNOSTICS_BASE/freeze-$FREEZE_B"
PILOT_SUBMISSION_ROOT="$DIAGNOSTICS_FREEZE_ROOT/pilot-submissions"
CONTRACT_ATTEMPT=""
SUBMISSION_AUDIT_ROOT=""
ATTEMPT=""
FREEZE_MANIFEST=""
CONTRACT_RECEIPT=""
FORMAL_ATTEMPT_RESERVED=false
FORMAL_JOB_ACCEPTED=false
SUBMISSION_COMPLETE=false
FAILURE_PHASE=initialization
ERROR_TEXT=""
CONTRACT_JOB_ID=""
ARRAY_JOB_ID=""
VALIDATION_JOB_ID=""
CANCEL_SUMMARY="not_requested"
RELOCATED_FORMAL_ATTEMPT=""
LOCK_FD=""
RECOVERED_JOB_IDS=""
AMBIGUOUS_SUBMISSION=false
AMBIGUOUS_JOB_NAME=""
AMBIGUOUS_RECOVERY_STATUS="not_applicable"

reserve_next_attempt() {
  local base="$1" variable="$2" index candidate
  mkdir -p -- "$base"
  for ((index = 1; index <= 9999; index++)); do
    candidate="$(printf '%s/attempt-%04d' "$base" "$index")"
    [[ -e "$candidate" ]] && continue
    if mkdir -- "$candidate"; then
      printf -v "$variable" '%s' "$candidate"
      return 0
    fi
    return 1
  done
  return 1
}

acquire_submission_lock() {
  local path="$1"
  mkdir -p -- "$(dirname -- "$path")"
  exec {LOCK_FD}>"$path"
  flock -n "$LOCK_FD"
}

active_campaign_jobs() {
  local queue job_name
  queue="$(squeue -h -u guisongxia -o '%i|%j|%T')" || return 2
  while IFS='|' read -r _ job_name _; do
    case "$job_name" in
      CA0803TrajPilot*|CA0803TrajValidate*) return 0 ;;
    esac
  done <<<"$queue"
  return 1
}

active_contract_jobs_for_freeze() {
  local queue job_name prefix="CA0803TrajContract-${FREEZE_B:0:8}-"
  queue="$(squeue -h -u guisongxia -o '%i|%j|%T')" || return 2
  while IFS='|' read -r _ job_name _; do
    [[ "$job_name" == CA0803TrajContract || "$job_name" == "$prefix"* ]] && return 0
  done <<<"$queue"
  return 1
}

recover_job_ids_by_name() {
  local expected_name="$1" variable="$2" queue job_id job_name recovered=""
  queue="$(squeue -h -u guisongxia -o '%i|%j')" || return 2
  while IFS='|' read -r job_id job_name; do
    [[ "$job_name" == "$expected_name" && "$job_id" =~ ^[0-9]+(_[0-9]+|_\[[^]]+\])?$ ]] || continue
    recovered="${recovered:+$recovered }$job_id"
  done <<<"$queue"
  printf -v "$variable" '%s' "$recovered"
  [[ -n "$recovered" ]]
}

recover_ambiguous_submission_by_name() {
  local expected_name="$1" poll recovery_rc=1 query_failed=false
  AMBIGUOUS_SUBMISSION=true
  AMBIGUOUS_JOB_NAME="$expected_name"
  RECOVERED_JOB_IDS=""
  for poll in 1 2 3; do
    if recover_job_ids_by_name "$expected_name" RECOVERED_JOB_IDS; then
      AMBIGUOUS_RECOVERY_STATUS="active_jobs_recovered"
      return 0
    else
      recovery_rc="$?"
    fi
    [[ "$recovery_rc" -eq 2 ]] && query_failed=true
    [[ "$poll" -lt 3 ]] && sleep 2
  done
  if [[ "$query_failed" == true ]]; then
    AMBIGUOUS_RECOVERY_STATUS="squeue_recovery_query_failed"
  else
    AMBIGUOUS_RECOVERY_STATUS="no_active_job_observed_after_retries"
  fi
  return 0
}

cancel_and_wait_one() {
  local job_id="$1" label="$2" cancel_rc=0 poll active observed=unknown
  [[ -n "$job_id" ]] || return 0
  scancel "$job_id" >/dev/null 2>&1 || cancel_rc="$?"
  for poll in 1 2 3 4 5; do
    if ! active="$(squeue -h -j "$job_id" -o '%i' 2>/dev/null)"; then
      observed=squeue_query_failed
      break
    fi
    if [[ -z "$active" ]]; then
      observed=squeue_absent
      break
    fi
    sleep 2
  done
  CANCEL_SUMMARY="${CANCEL_SUMMARY};${label}:${job_id}:scancel_rc=${cancel_rc}:observed=${observed}"
}

cancel_submitted_jobs() {
  local item
  CANCEL_SUMMARY="requested"
  cancel_and_wait_one "$VALIDATION_JOB_ID" validation
  for item in ${RECOVERED_JOB_IDS:-}; do
    cancel_and_wait_one "$item" recovered
  done
  cancel_and_wait_one "$ARRAY_JOB_ID" pilot_array
  cancel_and_wait_one "$CONTRACT_JOB_ID" contract
}

relocate_unsubmitted_formal_attempt() {
  local destination
  if [[ "$FORMAL_ATTEMPT_RESERVED" == true && "$FORMAL_JOB_ACCEPTED" == false \
    && "$AMBIGUOUS_SUBMISSION" == false \
    && -d "$ATTEMPT" && -n "$SUBMISSION_AUDIT_ROOT" ]]; then
    destination="$SUBMISSION_AUDIT_ROOT/formal-reservation-not-submitted"
    if [[ ! -e "$destination" ]] && mv -- "$ATTEMPT" "$destination"; then
      RELOCATED_FORMAL_ATTEMPT="$destination"
      FORMAL_ATTEMPT_RESERVED=false
    fi
  fi
  [[ "$FORMAL_JOB_ACCEPTED" == false ]] && rmdir -- "$CAMPAIGN_ROOT" 2>/dev/null || true
}

write_failure_receipt() {
  local phase="$1" error="$2"
  local directory target contract_receipt_sha
  [[ -n "$SUBMISSION_AUDIT_ROOT" ]] || return 0
  directory="$SUBMISSION_AUDIT_ROOT/status"
  mkdir -p -- "$directory"
  target="$directory/submission-failure-${phase}.json"
  contract_receipt_sha=""
  [[ -n "$CONTRACT_RECEIPT" && -f "$CONTRACT_RECEIPT" ]] \
    && contract_receipt_sha="$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')"
  "$PYTHON" - "$target" "$phase" "$error" "$FREEZE_B" "$IMPLEMENTATION_A" \
    "$CONFIG_SHA256" "$ATTEMPT" "$CONTRACT_ATTEMPT" "$CONTRACT_RECEIPT" \
    "$contract_receipt_sha" "$CONTRACT_JOB_ID" "$ARRAY_JOB_ID" \
    "$VALIDATION_JOB_ID" "$CANCEL_SUMMARY" "$RELOCATED_FORMAL_ATTEMPT" \
    "$AMBIGUOUS_SUBMISSION" "$AMBIGUOUS_JOB_NAME" "$AMBIGUOUS_RECOVERY_STATUS" \
    "$RECOVERED_JOB_IDS" <<'PY' || true
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_submission_failure_v2",
    "status": "failed",
    "phase": sys.argv[2],
    "error": sys.argv[3],
    "freeze_commit": sys.argv[4],
    "implementation_commit": sys.argv[5],
    "config_sha256": sys.argv[6],
    "formal_attempt": sys.argv[7],
    "diagnostics_contract_attempt": sys.argv[8],
    "diagnostics_contract_receipt": {"path": sys.argv[9], "sha256": sys.argv[10] or None},
    "submitted_jobs": {
        "contract": sys.argv[11] or None,
        "pilot_array": sys.argv[12] or None,
        "validation": sys.argv[13] or None,
    },
    "cancellation_summary": sys.argv[14],
    "relocated_unsubmitted_formal_attempt": sys.argv[15] or None,
    "ambiguous_sbatch": {
        "possible_controller_acceptance": sys.argv[16] == "true",
        "unique_job_name": sys.argv[17] or None,
        "recovery_status": sys.argv[18],
        "recovered_job_ids": sys.argv[19].split(),
    },
    "fixed_resources": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "accelerator": "A800", "peak_gpus": 3},
}
if target.exists():
    raise SystemExit("failure receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  if [[ ( "$FORMAL_JOB_ACCEPTED" == true || "$AMBIGUOUS_SUBMISSION" == true ) \
    && -d "$ATTEMPT/status" ]]; then
    cp -n -- "$target" "$ATTEMPT/status/submission-failure-${phase}.json" 2>/dev/null || true
  fi
}

notify_submission_failure() {
  local phase="$1" error="$2"
  QQ_SUB="CA0803 oracle trajectory submission failed" \
  QQ_BODY="status=failed; phase=$phase; error=$error; ambiguous_sbatch=$AMBIGUOUS_SUBMISSION; recovery=$AMBIGUOUS_RECOVERY_STATUS; job_name=$AMBIGUOUS_JOB_NAME; freeze=$FREEZE_B; diagnostics=$CONTRACT_ATTEMPT; formal=$ATTEMPT" \
    "$QQ_NOTIFIER" >/dev/null 2>&1 || true
}

submission_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

submission_finish() {
  local rc="$?"
  trap - EXIT ERR TERM INT
  set +e
  if [[ "$rc" -ne 0 && "$SUBMISSION_COMPLETE" != true ]]; then
    cancel_submitted_jobs
    relocate_unsubmitted_formal_attempt
    write_failure_receipt "$FAILURE_PHASE" "${ERROR_TEXT:-submission_failed}"
    notify_submission_failure "$FAILURE_PHASE" "${ERROR_TEXT:-submission_failed}"
  fi
  exit "$rc"
}

trap submission_error ERR
trap submission_finish EXIT
trap 'FAILURE_PHASE=signal; ERROR_TEXT=received_SIGTERM; exit 143' TERM
trap 'FAILURE_PHASE=signal; ERROR_TEXT=received_SIGINT; exit 130' INT

verify_freeze_delta() {
  local expected_config expected_doc observed
  expected_config="configs/qwen3vl_oracle_trajectory_pilot_0803.json"
  expected_doc="doc/training_pipeline/0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md"
  observed="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r \
    --no-renames "$IMPLEMENTATION_A" "$FREEZE_B" | LC_ALL=C sort)"
  [[ "$observed" == "$expected_config"$'\n'"$expected_doc" ]] \
    || { ERROR_TEXT=freeze_B_delta_must_be_exact_config_and_0803_doc; return 1; }
  if git -C "$ROOT" diff --quiet "$IMPLEMENTATION_A" "$FREEZE_B" -- "$expected_config"; then
    ERROR_TEXT=freeze_B_config_delta_missing
    return 1
  fi
  if git -C "$ROOT" diff --quiet "$IMPLEMENTATION_A" "$FREEZE_B" -- "$expected_doc"; then
    ERROR_TEXT=freeze_B_0803_doc_delta_missing
    return 1
  fi
  return 0
}

verify_exact_freeze_state() {
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$FREEZE_B" ]] \
    || { ERROR_TEXT=freeze_commit_drift; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_A" ]] \
    || { ERROR_TEXT=implementation_parent_drift; return 1; }
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
    || { ERROR_TEXT=formal_run_worktree_is_dirty; return 1; }
  verify_freeze_delta
  [[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] \
    || { ERROR_TEXT=config_hash_drift; return 1; }
}

validate_freeze_config() {
  "$PYTHON" - "$CONFIG" "$IMPLEMENTATION_A" "$PRIMARY_REPO" "$CAMPAIGN_ROOT" <<'PY'
import json, pathlib, sys
config_path = pathlib.Path(sys.argv[1])
implementation_a, primary_repo, campaign_root = sys.argv[2:5]
config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("status") != "frozen_pending_exact_commit_validation":
    raise SystemExit("pilot config status must be frozen_pending_exact_commit_validation")
anchors = config.get("anchors", {})
if anchors.get("pilot_implementation_commit") != implementation_a:
    raise SystemExit("config pilot_implementation_commit must equal B^")
if anchors.get("pilot_freeze_commit") is not None:
    raise SystemExit("config pilot_freeze_commit must remain null; B is external-receipt-bound")
resource = config.get("resource_plan", {})
expected = {
    "partition": "gre",
    "maximum_total_gpus": 3,
    "planned_generation_shards": 3,
    "gpus_per_shard": 1,
    "accelerator": "A800",
    "peak_gpu_concurrency": 3,
    "cpus_per_shard": 4,
    "memory_per_shard": "48G",
    "time_per_shard": "02:00:00",
}
for key, value in expected.items():
    if resource.get(key) != value:
        raise SystemExit(f"frozen resource plan differs: {key}")
if config.get("execution_protocol", {}).get("tool_device") != "cuda":
    raise SystemExit("frozen tool device must be cuda")
formal = config.get("formal_execution", {})
if formal.get("status") != "not_started" or formal.get("authorized_only_after_all_runtime_validation_pending_items_pass") is not True:
    raise SystemExit("formal execution authorization fields differ")
if pathlib.Path(config.get("tools_and_models", {}).get("repository", "")).resolve() != pathlib.Path(primary_repo).resolve():
    raise SystemExit("primary repository binding differs")
declared_output = pathlib.Path(config.get("artifact_plan", {}).get("output_root", "")).resolve()
if declared_output != pathlib.Path(campaign_root).resolve():
    raise SystemExit("formal campaign output root differs")
PY
}

create_freeze_manifest() {
  "$PYTHON" - "$ROOT" "$CONFIG" "$FREEZE_MANIFEST" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$CONFIG_SHA256" <<'PY'
import hashlib, json, os, pathlib, sys
root, config, output = map(pathlib.Path, sys.argv[1:4])
freeze_b, implementation_a, config_sha = sys.argv[4:7]
files = {
    "runner": "tools/run_qwen3vl_oracle_trajectory.py",
    "validator": "tools/validate_qwen3vl_oracle_trajectory.py",
    "seeded_worker": "tools/seeded_segmentation_worker.py",
    "contract_worker": "tools/run_qwen3vl_oracle_trajectory_contract.slurm",
    "pilot_worker": "tools/run_qwen3vl_oracle_trajectory_pilot.slurm",
    "validation_worker": "tools/run_qwen3vl_oracle_trajectory_validate.slurm",
    "submitter": "tools/submit_qwen3vl_oracle_trajectory.sh",
    "runner_test": "tests/test_qwen3vl_oracle_trajectory.py",
    "validator_test": "tests/test_validate_qwen3vl_oracle_trajectory.py",
}
frozen = {}
for name, relative in files.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"frozen file missing: {relative}")
    data = path.read_bytes()
    frozen[name] = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_freeze_manifest_v1",
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "freeze_policy": "B_directly_follows_A_and_changes_exactly_the_config_and_0803_handoff_doc; B_is_bound_by_external_manifest_and_receipts; config_pilot_freeze_commit_is_null",
    "freeze_delta": {
        "allowed_and_required_paths": [
            "configs/qwen3vl_oracle_trajectory_pilot_0803.json",
            "doc/training_pipeline/0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md",
        ],
        "observed_paths_equal_allowed": True,
    },
    "config": {"path": str(config), "sha256": config_sha, "size_bytes": config.stat().st_size},
    "files": frozen,
}
if output.exists():
    raise SystemExit("freeze manifest already exists")
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}

select_successful_contract_attempt() {
  local candidate selected="" receipt manifest
  shopt -s nullglob
  for candidate in "$DIAGNOSTICS_FREEZE_ROOT"/attempt-[0-9][0-9][0-9][0-9]; do
    receipt="$candidate/contract/receipt.json"
    manifest="$candidate/freeze_manifest.json"
    [[ -f "$receipt" && -f "$manifest" ]] || continue
    if "$PYTHON" - "$receipt" "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" <<'PY'
import json, pathlib, sys
receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "success" or receipt.get("result") != "all_contracts_passed":
    raise SystemExit(1)
if receipt.get("freeze_commit") != sys.argv[2] or receipt.get("implementation_commit") != sys.argv[3]:
    raise SystemExit(1)
if receipt.get("config_sha256") != sys.argv[4]:
    raise SystemExit(1)
PY
    then
      selected="$candidate"
    fi
  done
  shopt -u nullglob
  [[ -n "$selected" ]] || { ERROR_TEXT=no_successful_exact_freeze_contract_attempt; return 1; }
  CONTRACT_ATTEMPT="$selected"
  FREEZE_MANIFEST="$CONTRACT_ATTEMPT/freeze_manifest.json"
  CONTRACT_RECEIPT="$CONTRACT_ATTEMPT/contract/receipt.json"
}

validate_campaign_contents() {
  local child name
  [[ -e "$CAMPAIGN_ROOT" ]] || return 0
  [[ -d "$CAMPAIGN_ROOT" ]] || { ERROR_TEXT=formal_campaign_root_is_not_directory; return 1; }
  shopt -s nullglob dotglob
  for child in "$CAMPAIGN_ROOT"/*; do
    name="${child##*/}"
    [[ -d "$child" && "$name" =~ ^attempt-[0-9]{4}$ ]] || {
      shopt -u nullglob dotglob
      ERROR_TEXT=formal_campaign_contains_unmanaged_artifact
      return 1
    }
  done
  shopt -u nullglob dotglob
}

validate_manifest_and_contract() {
  local manifest_sha="$1" contract_sha="$2"
  "$PYTHON" - "$ROOT" "$FREEZE_MANIFEST" "$manifest_sha" "$CONTRACT_RECEIPT" \
    "$contract_sha" "$ATTEMPT" "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" <<'PY'
import hashlib, json, pathlib, sys
root, manifest_path = map(pathlib.Path, sys.argv[1:3])
manifest_sha = sys.argv[3]
receipt_path = pathlib.Path(sys.argv[4])
receipt_sha = sys.argv[5]
formal_attempt = pathlib.Path(sys.argv[6]).resolve()
contract_attempt = receipt_path.resolve().parent.parent
freeze_b, implementation_a, config_sha = sys.argv[7:10]
if manifest_path.resolve().parent != contract_attempt:
    raise SystemExit("freeze manifest is not inside receipt diagnostics attempt")
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha:
    raise SystemExit("freeze manifest raw hash mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("freeze_commit") != freeze_b or manifest.get("implementation_commit") != implementation_a:
    raise SystemExit("freeze manifest commit mismatch")
expected_delta = [
    "configs/qwen3vl_oracle_trajectory_pilot_0803.json",
    "doc/training_pipeline/0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md",
]
delta = manifest.get("freeze_delta", {})
if delta.get("allowed_and_required_paths") != expected_delta or delta.get("observed_paths_equal_allowed") is not True:
    raise SystemExit("freeze manifest A-to-B delta contract mismatch")
if manifest.get("config", {}).get("sha256") != config_sha:
    raise SystemExit("freeze manifest config mismatch")
for name, spec in manifest.get("files", {}).items():
    path = (root / spec.get("path", "")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"frozen path escapes root: {name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.get("sha256") or len(data) != spec.get("size_bytes"):
        raise SystemExit(f"frozen file differs: {name}")
if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != receipt_sha:
    raise SystemExit("contract receipt raw hash mismatch")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("schema_id") != "qwen3vl_oracle_trajectory_contract_receipt_v3":
    raise SystemExit("contract receipt schema mismatch")
if receipt.get("status") != "success" or receipt.get("result") != "all_contracts_passed":
    raise SystemExit("contract did not pass")
if receipt.get("exit_code") != 0 or not str(receipt.get("job_id", "")).isdigit():
    raise SystemExit("contract terminal/job identity mismatch")
if receipt.get("resources", {}).get("gpus") != 0:
    raise SystemExit("contract unexpectedly used a GPU")
if receipt.get("freeze_commit") != freeze_b or receipt.get("implementation_commit") != implementation_a:
    raise SystemExit("contract exact B/A binding mismatch")
if receipt.get("config_sha256") != config_sha or receipt.get("freeze_manifest", {}).get("sha256") != manifest_sha:
    raise SystemExit("contract config/manifest binding mismatch")
if receipt.get("diagnostics_contract_attempt") != str(contract_attempt):
    raise SystemExit("contract diagnostics attempt binding mismatch")
if receipt.get("frozen_files") != manifest.get("files"):
    raise SystemExit("contract worker hashes differ from freeze manifest")
submission_path = contract_attempt / "contract_submission.json"
submission = json.loads(submission_path.read_text(encoding="utf-8"))
expected_submission = {
    "schema_id": "qwen3vl_oracle_trajectory_submission_v3",
    "phase": "contract",
    "job_id": str(receipt.get("job_id")),
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest_sha256": manifest_sha,
    "diagnostics_contract_attempt": str(contract_attempt),
}
for key, value in expected_submission.items():
    if submission.get(key) != value:
        raise SystemExit(f"contract submission binding mismatch: {key}")
preflight = receipt.get("preflight", {})
if (
    preflight.get("exists") is not True
    or preflight.get("passed") is not True
    or preflight.get("schema_id") != "qwen3vl_oracle_trajectory_preflight_receipt_v1"
    or preflight.get("status") != "passed"
    or preflight.get("config_sha256") != config_sha
):
    raise SystemExit("contract preflight receipt did not pass or is unbound")
for label, entry in (("receipt", preflight), ("log", preflight.get("log", {}))):
    path = (contract_attempt / entry.get("path", "")).resolve()
    try:
        path.relative_to(contract_attempt)
    except ValueError:
        raise SystemExit(f"contract preflight {label} escapes attempt")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("size_bytes"):
        raise SystemExit(f"contract preflight {label} hash/size mismatch")
tests = receipt.get("tests", {})
required = {"targeted_runner", "targeted_validator", "full_regression"}
if set(tests) != required:
    raise SystemExit("contract test set mismatch")
for name in sorted(required):
    entry = tests[name]
    if entry.get("passed") is not True or entry.get("test_count", 0) <= 0:
        raise SystemExit(f"zero or failed tests: {name}")
    if entry.get("failures") != 0 or entry.get("errors") != 0:
        raise SystemExit(f"nonzero failure/error count: {name}")
    log = (contract_attempt / entry.get("path", "")).resolve()
    try:
        log.relative_to(contract_attempt)
    except ValueError:
        raise SystemExit(f"contract log escapes attempt: {name}")
    data = log.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("size_bytes"):
        raise SystemExit(f"contract log hash/size mismatch: {name}")
PY
}

FAILURE_PHASE=freeze_validation
[[ "$(id -un)" == guisongxia ]] || { ERROR_TEXT=wrong_unix_user; exit 2; }
[[ "$FREEZE_B" =~ ^[0-9a-f]{40}$ && "$IMPLEMENTATION_A" =~ ^[0-9a-f]{40}$ ]] \
  || { ERROR_TEXT=malformed_B_or_A_commit; exit 2; }
[[ "$(git -C "$ROOT" rev-list --parents -n 1 HEAD | wc -w)" -eq 2 ]] \
  || { ERROR_TEXT=freeze_B_must_have_exactly_one_parent_A; exit 2; }
if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
  ERROR_TEXT=formal_run_worktree_must_be_detached_at_exact_freeze_B
  exit 2
fi
verify_exact_freeze_state
validate_freeze_config
scontrol ping | grep -q UP || { ERROR_TEXT=slurm_controller_not_UP; exit 3; }
NODE_INFO="$(scontrol show node "$NODE")"
grep -Eq 'State=[^[:space:]]*(DOWN|DRAIN|FAIL|MAINT|UNKNOWN)' <<<"$NODE_INFO" \
  && { ERROR_TEXT=fixed_node_gpu48_unavailable; exit 3; }

if [[ "$MODE" == contract ]]; then
  FAILURE_PHASE=contract_reservation
  mkdir -p -- "$DIAGNOSTICS_FREEZE_ROOT"
  acquire_submission_lock "$DIAGNOSTICS_FREEZE_ROOT/.contract-submission.lock" \
    || { ERROR_TEXT=concurrent_contract_submitter; exit 2; }
  reserve_next_attempt "$DIAGNOSTICS_FREEZE_ROOT" CONTRACT_ATTEMPT \
    || { ERROR_TEXT=cannot_reserve_next_contract_attempt; exit 2; }
  SUBMISSION_AUDIT_ROOT="$CONTRACT_ATTEMPT"
  FREEZE_MANIFEST="$CONTRACT_ATTEMPT/freeze_manifest.json"
  CONTRACT_RECEIPT="$CONTRACT_ATTEMPT/contract/receipt.json"
  mkdir -- "$CONTRACT_ATTEMPT/logs" "$CONTRACT_ATTEMPT/contract" "$CONTRACT_ATTEMPT/status"
  create_freeze_manifest
  MANIFEST_SHA256="$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')"
  "$PYTHON" - "$CONTRACT_ATTEMPT/owner.json" "$CONTRACT_ATTEMPT" "$CAMPAIGN_ROOT" \
    "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$MANIFEST_SHA256" <<'PY'
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_contract_attempt_owner_v2",
    "status": "reserved",
    "diagnostics_contract_attempt": sys.argv[2],
    "formal_campaign_root": sys.argv[3],
    "formal_attempt_policy": "first_absent_append_only_attempt_NNNN_selected_only_after_contract_pass",
    "freeze_commit": sys.argv[4],
    "implementation_commit": sys.argv[5],
    "config_sha256": sys.argv[6],
    "freeze_manifest_sha256": sys.argv[7],
    "fixed_resources": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "accelerator": "A800", "peak_gpus": 3},
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  if active_contract_jobs_for_freeze; then
    ERROR_TEXT=active_duplicate_exact_freeze_contract
    exit 2
  elif [[ "$?" -ne 1 ]]; then
    ERROR_TEXT=cannot_query_active_contract_jobs
    exit 3
  fi
  ATTEMPT_NAME="${CONTRACT_ATTEMPT##*/}"
  CONTRACT_JOB_NAME="CA0803TrajContract-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"
  FAILURE_PHASE=contract_submission
  verify_exact_freeze_state
  if ! JOB_RAW="$(sbatch --parsable --export=NIL --job-name="$CONTRACT_JOB_NAME" \
    --partition=gre --nodelist=gpu48 --cpus-per-task=2 --mem=16G --time=00:30:00 \
    --output="$CONTRACT_ATTEMPT/logs/contract-%j.out" --error="$CONTRACT_ATTEMPT/logs/contract-%j.err" \
    "$ROOT/tools/run_qwen3vl_oracle_trajectory_contract.slurm" \
    "$ROOT" "$CONTRACT_ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256")"; then
    recover_ambiguous_submission_by_name "$CONTRACT_JOB_NAME"
    ERROR_TEXT=sbatch_contract_failed
    exit 3
  fi
  JOB="${JOB_RAW%%;*}"
  JOB="${JOB%%$'\n'*}"
  if [[ ! "$JOB" =~ ^[0-9]+$ ]]; then
    recover_job_ids_by_name "$CONTRACT_JOB_NAME" RECOVERED_JOB_IDS || true
    ERROR_TEXT=malformed_contract_job_id_after_sbatch_success
    exit 3
  fi
  CONTRACT_JOB_ID="$JOB"
  "$PYTHON" - "$CONTRACT_ATTEMPT/contract_submission.json" "$JOB" "$CONTRACT_JOB_NAME" \
    "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$MANIFEST_SHA256" \
    "$CONTRACT_ATTEMPT" "$CAMPAIGN_ROOT" <<'PY'
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_submission_v3",
    "phase": "contract",
    "job_id": sys.argv[2],
    "job_name": sys.argv[3],
    "freeze_commit": sys.argv[4],
    "implementation_commit": sys.argv[5],
    "config_sha256": sys.argv[6],
    "freeze_manifest_sha256": sys.argv[7],
    "environment_export": "NIL",
    "fixed_resources": {"gpus": 0, "cpus": 2, "memory": "16G", "time": "00:30:00", "node": "gpu48", "partition": "gre"},
    "diagnostics_contract_attempt": sys.argv[8],
    "formal_campaign_root_not_created_by_contract": sys.argv[9],
    "diagnostics_contract_receipt_expected": {"path": str(pathlib.Path(sys.argv[8]) / "contract" / "receipt.json"), "sha256": None},
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  verify_exact_freeze_state
  SUBMISSION_COMPLETE=true
  printf 'contract_job=%s\ndiagnostics_attempt=%s\nformal_attempt_created=false\nfreeze_B=%s\nimplementation_A=%s\n' \
    "$JOB" "$CONTRACT_ATTEMPT" "$FREEZE_B" "$IMPLEMENTATION_A"
  exit 0
fi

FAILURE_PHASE=pilot_submission_reservation
mkdir -p -- "$PILOT_SUBMISSION_ROOT"
acquire_submission_lock "$DIAGNOSTICS_BASE/.pilot-submission.lock" \
  || { ERROR_TEXT=concurrent_formal_pilot_submitter; exit 2; }
reserve_next_attempt "$PILOT_SUBMISSION_ROOT" SUBMISSION_AUDIT_ROOT \
  || { ERROR_TEXT=cannot_reserve_pilot_submission_audit_attempt; exit 2; }
mkdir -- "$SUBMISSION_AUDIT_ROOT/status"
select_successful_contract_attempt
[[ -f "$CONTRACT_ATTEMPT/owner.json" && -f "$FREEZE_MANIFEST" && -f "$CONTRACT_RECEIPT" ]] \
  || { ERROR_TEXT=successful_contract_attempt_is_incomplete; exit 2; }
MANIFEST_SHA256="$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')"
CONTRACT_SHA256="$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')"
validate_manifest_and_contract "$MANIFEST_SHA256" "$CONTRACT_SHA256" \
  || { ERROR_TEXT=contract_manifest_receipt_or_log_validation_failed; exit 2; }
"$PYTHON" - "$CONTRACT_ATTEMPT/owner.json" "$CONTRACT_ATTEMPT" "$CAMPAIGN_ROOT" \
  "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$MANIFEST_SHA256" <<'PY'
import json, pathlib, sys
owner = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "diagnostics_contract_attempt": sys.argv[2],
    "formal_campaign_root": sys.argv[3],
    "freeze_commit": sys.argv[4],
    "implementation_commit": sys.argv[5],
    "config_sha256": sys.argv[6],
    "freeze_manifest_sha256": sys.argv[7],
}
for key, value in expected.items():
    if owner.get(key) != value:
        raise SystemExit(f"contract attempt owner differs: {key}")
PY
validate_campaign_contents
if active_campaign_jobs; then
  ERROR_TEXT=active_duplicate_formal_campaign_job
  exit 2
elif [[ "$?" -ne 1 ]]; then
  ERROR_TEXT=cannot_query_active_formal_campaign_jobs
  exit 3
fi

FAILURE_PHASE=formal_attempt_reservation
reserve_next_attempt "$CAMPAIGN_ROOT" ATTEMPT \
  || { ERROR_TEXT=cannot_reserve_next_formal_attempt; exit 2; }
FORMAL_ATTEMPT_RESERVED=true
mkdir -- "$ATTEMPT/logs" "$ATTEMPT/status" "$ATTEMPT/shards"
ATTEMPT_NAME="${ATTEMPT##*/}"
PILOT_JOB_NAME="CA0803TrajPilot-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"
VALIDATION_JOB_NAME="CA0803TrajValidate-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"
"$PYTHON" - "$SUBMISSION_AUDIT_ROOT/reservation.json" "$ATTEMPT" "$CONTRACT_ATTEMPT" \
  "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$MANIFEST_SHA256" "$CONTRACT_SHA256" <<'PY'
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_pilot_submission_reservation_v1",
    "status": "reserved",
    "formal_attempt": sys.argv[2],
    "diagnostics_contract_attempt": sys.argv[3],
    "freeze_commit": sys.argv[4],
    "implementation_commit": sys.argv[5],
    "config_sha256": sys.argv[6],
    "freeze_manifest_sha256": sys.argv[7],
    "contract_receipt_sha256": sys.argv[8],
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

FAILURE_PHASE=pilot_array_submission
verify_exact_freeze_state
if ! ARRAY_RAW="$(sbatch --parsable --export=NIL --job-name="$PILOT_JOB_NAME" \
  --partition=gre --nodelist=gpu48 --array=0-2%3 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --time=02:00:00 \
  --output="$ATTEMPT/logs/pilot-%A_%a.out" --error="$ATTEMPT/logs/pilot-%A_%a.err" \
  "$ROOT/tools/run_qwen3vl_oracle_trajectory_pilot.slurm" \
  "$ROOT" "$ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
  "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256" \
  "$CONTRACT_RECEIPT" "$CONTRACT_SHA256")"; then
  recover_ambiguous_submission_by_name "$PILOT_JOB_NAME"
  ERROR_TEXT=sbatch_array_failed
  exit 3
fi
FORMAL_JOB_ACCEPTED=true
ARRAY="${ARRAY_RAW%%;*}"
ARRAY="${ARRAY%%$'\n'*}"
if [[ ! "$ARRAY" =~ ^[0-9]+$ ]]; then
  recover_job_ids_by_name "$PILOT_JOB_NAME" RECOVERED_JOB_IDS || true
  ERROR_TEXT=malformed_array_job_id_after_sbatch_success
  exit 3
fi
ARRAY_JOB_ID="$ARRAY"

FAILURE_PHASE=validator_submission
verify_exact_freeze_state
if ! VALIDATE_RAW="$(sbatch --parsable --export=NIL --dependency="afterany:$ARRAY" \
  --job-name="$VALIDATION_JOB_NAME" --partition=gre --nodelist=gpu48 \
  --cpus-per-task=2 --mem=16G --time=00:30:00 \
  --output="$ATTEMPT/logs/validation-%j.out" --error="$ATTEMPT/logs/validation-%j.err" \
  "$ROOT/tools/run_qwen3vl_oracle_trajectory_validate.slurm" \
  "$ROOT" "$ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
  "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256" \
  "$CONTRACT_RECEIPT" "$CONTRACT_SHA256" "$ARRAY")"; then
  recover_ambiguous_submission_by_name "$VALIDATION_JOB_NAME"
  ERROR_TEXT=sbatch_validator_failed
  exit 3
fi
VALIDATE="${VALIDATE_RAW%%;*}"
VALIDATE="${VALIDATE%%$'\n'*}"
if [[ ! "$VALIDATE" =~ ^[0-9]+$ ]]; then
  recover_job_ids_by_name "$VALIDATION_JOB_NAME" RECOVERED_JOB_IDS || true
  ERROR_TEXT=malformed_validator_job_id_after_sbatch_success
  exit 3
fi
VALIDATION_JOB_ID="$VALIDATE"

FAILURE_PHASE=pilot_submission_recording
"$PYTHON" - "$ATTEMPT/pilot_submission.json" "$ARRAY" "$VALIDATE" "$PILOT_JOB_NAME" \
  "$VALIDATION_JOB_NAME" "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" \
  "$MANIFEST_SHA256" "$CONTRACT_SHA256" "$CONTRACT_ATTEMPT" "$CONTRACT_RECEIPT" \
  "$ATTEMPT" "$SUBMISSION_AUDIT_ROOT" <<'PY'
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_submission_v3",
    "phase": "pilot",
    "array_job_id": sys.argv[2],
    "validation_job_id": sys.argv[3],
    "job_names": {"pilot": sys.argv[4], "validation": sys.argv[5]},
    "validation_dependency": "afterany",
    "freeze_commit": sys.argv[6],
    "implementation_commit": sys.argv[7],
    "config_sha256": sys.argv[8],
    "freeze_manifest_sha256": sys.argv[9],
    "contract_receipt_sha256": sys.argv[10],
    "diagnostics_contract_attempt": sys.argv[11],
    "diagnostics_contract_receipt": {"path": sys.argv[12], "sha256": sys.argv[10]},
    "pilot_submission_audit_attempt": sys.argv[14],
    "contract_logs_and_nonzero_test_counts_reverified": True,
    "contract_receipt_schema": "qwen3vl_oracle_trajectory_contract_receipt_v3",
    "contract_preflight_receipt_and_log_reverified": True,
    "environment_export": "NIL",
    "fixed_resources": {
        "user": "guisongxia", "node": "gpu48", "partition": "gre",
        "accelerator": "A800", "bf16_required": True,
        "shards": 3, "gpus_per_shard": 1, "peak_gpus": 3,
        "cpus_per_shard": 4, "memory_per_shard": "48G", "time_per_shard": "02:00:00",
    },
    "append_only_attempt": pathlib.Path(sys.argv[13]).name,
    "attempt": sys.argv[13],
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
cp -- "$ATTEMPT/pilot_submission.json" "$SUBMISSION_AUDIT_ROOT/pilot_submission.json"
verify_exact_freeze_state
SUBMISSION_COMPLETE=true
printf 'pilot_array_job=%s\nvalidation_job=%s\ndependency=afterany\nformal_attempt=%s\ndiagnostics_contract=%s\npeak_gpus=3\n' \
  "$ARRAY" "$VALIDATE" "$ATTEMPT" "$CONTRACT_ATTEMPT"
