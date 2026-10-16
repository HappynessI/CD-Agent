#!/usr/bin/env bash
set -Eeuo pipefail
umask 027
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

MODE="${1:-}"
[[ "$MODE" == contract || "$MODE" == smoke || "$MODE" == pilot ]] ||
  { echo "usage: $0 {contract|smoke|pilot}" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(realpath -e -- /guisongxia01/pangchao/wangyihan)"
PRIMARY_REPO="$(realpath -e -- "$WORKSPACE/Chang-Agent")"
OUTPUT_ROOT="$(realpath -e -- "$PRIMARY_REPO/outputs")"
CAMPAIGN_ROOT="$OUTPUT_ROOT/CA_0804(1)-qwen3vl-gt-direct-teacher"
DIAGNOSTICS_ROOT="$OUTPUT_ROOT/CA_0804_diagnostics/CA_0804(1)-qwen3vl-gt-direct-teacher-contract"
CONFIG="$(realpath -e -- "$ROOT/configs/qwen3vl_gt_direct_teacher_0804.json")"
PYTHON="$(realpath -e -- "$WORKSPACE/conda_envs/omniovcd-env/bin/python3.12")"
QQ_NOTIFIER="$(realpath -e -- "$WORKSPACE/.local/bin/wangyihan-send-qq")"
NODE=gpu48
PARTITION=gre
FREEZE_B="$(git -C "$ROOT" rev-parse HEAD)"
IMPLEMENTATION_A="$(git -C "$ROOT" rev-parse HEAD^)"
CONFIG_SHA256="$(sha256sum -- "$CONFIG" | awk '{print $1}')"
DIAGNOSTICS_FREEZE_ROOT="$DIAGNOSTICS_ROOT/freeze-$FREEZE_B"
SMOKE_ROOT="$DIAGNOSTICS_FREEZE_ROOT/smoke-attempts"
ATTEMPT=""
CONTRACT_ATTEMPT=""
FREEZE_MANIFEST=""
CONTRACT_RECEIPT=""
MANIFEST_SHA256=""
CONTRACT_SHA256=""
ARRAY_JOB_ID=""
VALIDATION_JOB_ID=""
CONTRACT_JOB_ID=""
SMOKE_ATTEMPT=""
SMOKE_RECEIPT=""
SMOKE_SHA256=""
SMOKE_JOB_ID=""
NODE_STATE=""
NODE_GPU_COUNT=""
NODE_PROBE_UTC=""
FAILURE_PHASE=initialization
ERROR_TEXT=""
SUBMISSION_COMPLETE=false
LOCK_FD=""

reserve_attempt() {
  local base="$1" variable="$2" index candidate
  mkdir -p -- "$base"
  for ((index = 1; index <= 9999; index++)); do
    candidate="$(printf '%s/attempt-%04d' "$base" "$index")"
    if mkdir -- "$candidate" 2>/dev/null; then
      printf -v "$variable" '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

acquire_lock() {
  local path="$1"
  mkdir -p -- "$(dirname -- "$path")"
  exec {LOCK_FD}>"$path"
  flock -n "$LOCK_FD"
}

verify_freeze() {
  local observed
  [[ "$(id -un)" == guisongxia ]] || { ERROR_TEXT=wrong_unix_user; return 1; }
  [[ "$FREEZE_B" =~ ^[0-9a-f]{40}$ && "$IMPLEMENTATION_A" =~ ^[0-9a-f]{40}$ ]] ||
    { ERROR_TEXT=malformed_freeze_commits; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$FREEZE_B" &&
    "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_A" ]] ||
    { ERROR_TEXT=freeze_parent_binding_failed; return 1; }
  [[ "$(git -C "$ROOT" rev-list --parents -n 1 HEAD | wc -w)" -eq 2 ]] ||
    { ERROR_TEXT=freeze_B_must_have_exactly_one_parent_A; return 1; }
  observed="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r --no-renames \
    "$IMPLEMENTATION_A" "$FREEZE_B" | LC_ALL=C sort)"
  [[ "$observed" == "configs/qwen3vl_gt_direct_teacher_0804.json"$'\n'"doc/training_pipeline/0804_fifith_try/0804.md" ]] ||
    { ERROR_TEXT=freeze_B_delta_must_be_exact_config_and_0804_doc; return 1; }
  if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
    ERROR_TEXT=formal_run_worktree_must_be_detached_at_exact_freeze_B
    return 1
  fi
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] ||
    { ERROR_TEXT=formal_run_worktree_dirty; return 1; }
  [[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] ||
    { ERROR_TEXT=config_hash_drift; return 1; }
}

validate_smoke_preregistration() {
  "$PYTHON" - "$CONFIG" <<'PY'
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "required_before_formal_pilot": True,
    "sample_id": "train_407_r3c2",
    "shard_index": 0,
    "num_shards": 4,
    "gpus": 1,
    "time_limit": "12:00:00",
    "output_scope": "diagnostics_only",
    "scientific_result_authoritative": False,
    "geometry_modes_executed": ["matched"],
    "step_indices_executed": [1],
    "max_real_candidates": 2,
    "required_tool_action_pairs": [["sam3", "box"], ["simpleclick", "positive_point"]],
    "endpoint_search_performed": False,
}
if config.get("resource_plan", {}).get("diagnostic_smoke") != expected:
    raise SystemExit("frozen smoke preregistration differs")
PY
}

active_jobs() {
  local kind="$1" queue _ job_name state
  queue="$(squeue -h -u guisongxia -o '%i|%j|%T')" || return 2
  while IFS='|' read -r _ job_name state; do
    case "$kind:$job_name" in
      contract:CA0804TeacherContract-*) return 0 ;;
      smoke:CA0804TeacherSmoke-*|smoke:CA0804TeacherPilot-*|smoke:CA0804TeacherValidate-*) return 0 ;;
      pilot:CA0804TeacherSmoke-*|pilot:CA0804TeacherPilot-*|pilot:CA0804TeacherValidate-*) return 0 ;;
    esac
  done <<<"$queue"
  return 1
}

create_manifest() {
  "$PYTHON" - "$ROOT" "$FREEZE_MANIFEST" "$FREEZE_B" "$IMPLEMENTATION_A" "$CONFIG_SHA256" <<'PY'
import hashlib, json, os, pathlib, sys
root, target = map(pathlib.Path, sys.argv[1:3])
freeze_b, implementation_a, config_sha = sys.argv[3:6]
paths = {
    "runner": "tools/run_qwen3vl_gt_direct_teacher.py",
    "validator": "tools/validate_qwen3vl_gt_direct_teacher.py",
    "contract_worker": "tools/run_qwen3vl_gt_direct_teacher_contract.slurm",
    "smoke_worker": "tools/run_qwen3vl_gt_direct_teacher_smoke.slurm",
    "pilot_worker": "tools/run_qwen3vl_gt_direct_teacher_pilot.slurm",
    "validation_worker": "tools/run_qwen3vl_gt_direct_teacher_validate.slurm",
    "submitter": "tools/submit_qwen3vl_gt_direct_teacher.sh",
    "runner_test": "tests/test_qwen3vl_gt_direct_teacher.py",
    "validator_test": "tests/test_validate_qwen3vl_gt_direct_teacher.py",
    "config": "configs/qwen3vl_gt_direct_teacher_0804.json",
    "handoff_doc": "doc/training_pipeline/0804_fifith_try/0804.md",
}
files = {}
for name, relative in paths.items():
    data = (root / relative).read_bytes()
    files[name] = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_freeze_manifest_v1",
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_policy": "B_directly_follows_A_and_changes_exactly_config_plus_0804_doc",
    "freeze_delta": {
        "allowed_and_required_paths": [
            "configs/qwen3vl_gt_direct_teacher_0804.json",
            "doc/training_pipeline/0804_fifith_try/0804.md",
        ],
        "observed_paths_equal_allowed": True,
    },
    "teacher_only": True,
    "training_authorized": False,
    "files": files,
}
if target.exists():
    raise SystemExit("append-only freeze manifest already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

select_contract() {
  local selected
  if ! selected="$("$PYTHON" - "$DIAGNOSTICS_FREEZE_ROOT" "$ROOT" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$CONFIG_SHA256" <<'PY'
import hashlib, json, pathlib, sys
base, root = map(pathlib.Path, sys.argv[1:3])
freeze_b, implementation_a, config_sha = sys.argv[3:6]
for attempt in sorted(base.glob("attempt-[0-9][0-9][0-9][0-9]"), reverse=True):
    manifest_path = attempt / "freeze_manifest.json"
    receipt_path = attempt / "contract" / "receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        continue
    try:
        manifest_data = manifest_path.read_bytes()
        manifest = json.loads(manifest_data)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_id") != "qwen3vl_gt_direct_teacher_contract_receipt_v1": continue
        if receipt.get("status") != "success" or receipt.get("result") != "all_contracts_passed": continue
        if receipt.get("freeze_commit") != freeze_b or receipt.get("implementation_commit") != implementation_a: continue
        if receipt.get("config_sha256") != config_sha: continue
        if receipt.get("freeze_manifest_sha256") != hashlib.sha256(manifest_data).hexdigest(): continue
        if receipt.get("preflight", {}).get("passed") is not True: continue
        valid = True
        for name in ("targeted_runner", "targeted_validator", "full_regression"):
            entry = receipt.get("logs", {}).get(name, {})
            path = attempt / entry.get("path", "")
            if entry.get("passed") is not True or not path.is_file() or entry.get("test_count", 0) <= 0:
                valid = False; break
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("size_bytes"):
                valid = False; break
        if not valid: continue
        for spec in manifest.get("files", {}).values():
            data = (root / spec["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != spec["size_bytes"]:
                valid = False; break
        if not valid: continue
        print(attempt); print(manifest_path); print(receipt_path)
        raise SystemExit(0)
    except Exception:
        continue
raise SystemExit("no successful exact-freeze contract receipt")
PY
)"; then
    ERROR_TEXT=no_successful_exact_freeze_contract_receipt
    return 1
  fi
  mapfile -t selected_lines <<<"$selected"
  [[ "${#selected_lines[@]}" -eq 3 ]] || { ERROR_TEXT=contract_selection_malformed; return 1; }
  CONTRACT_ATTEMPT="${selected_lines[0]}"
  FREEZE_MANIFEST="${selected_lines[1]}"
  CONTRACT_RECEIPT="${selected_lines[2]}"
  MANIFEST_SHA256="$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')"
  CONTRACT_SHA256="$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')"
}

select_smoke() {
  local selected
  if ! selected="$("$PYTHON" - "$SMOKE_ROOT" "$FREEZE_B" "$IMPLEMENTATION_A" \
    "$CONFIG_SHA256" "$MANIFEST_SHA256" "$CONTRACT_SHA256" "$CONTRACT_RECEIPT" <<'PY'
import hashlib, json, pathlib, sys
base = pathlib.Path(sys.argv[1])
freeze_b, implementation_a, config_sha, manifest_sha, contract_sha = sys.argv[2:7]
contract_receipt = str(pathlib.Path(sys.argv[7]).resolve())
for attempt in sorted(base.glob("attempt-[0-9][0-9][0-9][0-9]"), reverse=True):
    receipt_path = attempt / "status" / "smoke.json"
    marker_path = attempt / "smoke" / "shard-0" / "generation_complete.json"
    index_path = attempt / "smoke" / "shard-0" / "shard_index.json"
    submission_path = attempt / "smoke_submission.json"
    if not all(path.is_file() for path in (receipt_path, marker_path, index_path, submission_path)):
        continue
    try:
        receipt_data = receipt_path.read_bytes()
        receipt = json.loads(receipt_data)
        marker_data = marker_path.read_bytes()
        index_data = index_path.read_bytes()
        marker = json.loads(marker_data)
        index = json.loads(index_data)
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        expected = {
            "schema_id": "qwen3vl_gt_direct_teacher_smoke_receipt_v1",
            "status": "success",
            "result": "smoke_complete",
            "freeze_commit": freeze_b,
            "implementation_commit": implementation_a,
            "config_sha256": config_sha,
            "freeze_manifest_sha256": manifest_sha,
            "contract_receipt_sha256": contract_sha,
            "contract_receipt": contract_receipt,
            "sample_id": "train_407_r3c2",
            "shard_index": 0,
            "num_shards": 4,
            "teacher_only": True,
            "training_authorized": False,
            "qwen_invoked": False,
            "output_scope": "diagnostics_only",
            "scientific_result_authoritative": False,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            continue
        if not receipt.get("job_id", "").isdigit():
            continue
        marker_spec = receipt.get("generation_complete", {})
        index_spec = receipt.get("shard_index_artifact", {})
        if hashlib.sha256(marker_data).hexdigest() != marker_spec.get("sha256"):
            continue
        if len(marker_data) != marker_spec.get("size_bytes"):
            continue
        if hashlib.sha256(index_data).hexdigest() != index_spec.get("sha256"):
            continue
        if len(index_data) != index_spec.get("size_bytes"):
            continue
        for payload in (marker, index):
            if payload.get("status") != "completed" or payload.get("runtime_commit") != freeze_b:
                raise ValueError("smoke payload binding")
            if payload.get("shard_index") != 0 or payload.get("num_shards") != 4:
                raise ValueError("smoke shard binding")
            if payload.get("episode_ids") != ["train_407_r3c2"]:
                raise ValueError("smoke sample binding")
            if payload.get("teacher_only") is not True or payload.get("training_authorized") is not False:
                raise ValueError("smoke safety binding")
        if (
            submission.get("phase") != "smoke"
            or submission.get("freeze_commit") != freeze_b
            or submission.get("smoke_job_id") != receipt["job_id"]
            or submission.get("config_sha256") != config_sha
        ):
            continue
        print(attempt)
        print(receipt_path)
        print(hashlib.sha256(receipt_data).hexdigest())
        raise SystemExit(0)
    except Exception:
        continue
raise SystemExit("no successful exact-freeze one-GPU smoke receipt")
PY
)"; then
    ERROR_TEXT=no_successful_exact_freeze_smoke_receipt
    return 1
  fi
  mapfile -t selected_lines <<<"$selected"
  [[ "${#selected_lines[@]}" -eq 3 ]] || { ERROR_TEXT=smoke_selection_malformed; return 1; }
  SMOKE_ATTEMPT="${selected_lines[0]}"
  SMOKE_RECEIPT="${selected_lines[1]}"
  SMOKE_SHA256="${selected_lines[2]}"
  [[ "$(sha256sum -- "$SMOKE_RECEIPT" | awk '{print $1}')" == "$SMOKE_SHA256" ]] ||
    { ERROR_TEXT=smoke_receipt_hash_drift; return 1; }
}

write_record() {
  local target="$1" phase="$2"
  "$PYTHON" - "$target" "$phase" "$ATTEMPT" "$CONTRACT_ATTEMPT" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$MANIFEST_SHA256" "$CONTRACT_SHA256" \
    "$ARRAY_JOB_ID" "$VALIDATION_JOB_ID" "$CONTRACT_JOB_ID" "$SMOKE_ATTEMPT" \
    "$SMOKE_RECEIPT" "$SMOKE_SHA256" "$SMOKE_JOB_ID" <<'PY'
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
phase = sys.argv[2]
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_submission_v1",
    "phase": phase,
    "attempt": sys.argv[3] or None,
    "formal_attempt": sys.argv[3] if phase == "pilot" else None,
    "diagnostics_contract_attempt": sys.argv[4] or None,
    "freeze_commit": sys.argv[5],
    "implementation_commit": sys.argv[6],
    "config_sha256": sys.argv[7],
    "freeze_manifest_sha256": sys.argv[8] or None,
    "contract_receipt_sha256": sys.argv[9] or None,
    "array_job_id": sys.argv[10] or None,
    "validation_job_id": sys.argv[11] or None,
    "contract_job_id": sys.argv[12] or None,
    "smoke_attempt": sys.argv[13] or None,
    "smoke_receipt": sys.argv[14] or None,
    "smoke_receipt_sha256": sys.argv[15] or None,
    "smoke_job_id": sys.argv[16] or None,
    "validation_dependency": "afterany" if phase == "pilot" else None,
    "fixed_resources": {
        "user": "guisongxia", "node": "gpu48", "partition": "gre",
        "array": "0-3%4" if phase == "pilot" else None,
        "gpus_per_shard": 1 if phase in {"smoke", "pilot"} else 0,
        "peak_gpus": 4 if phase == "pilot" else 1 if phase == "smoke" else 0,
        "time_per_shard": "12:00:00" if phase in {"smoke", "pilot"} else None,
    },
    "output_scope": "diagnostics_only" if phase == "smoke" else "formal" if phase == "pilot" else "diagnostics",
    "teacher_only": True,
    "training_authorized": False,
}
if target.exists():
    raise SystemExit("append-only submission record already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

write_node_preflight() {
  local target="$1"
  "$PYTHON" - "$target" "$NODE_PROBE_UTC" "$NODE_STATE" "$NODE_GPU_COUNT" \
    "$NODE" "$PARTITION" "$FREEZE_B" <<'PY'
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_submission_preflight_v1",
    "status": "passed",
    "checked_at_utc": sys.argv[2],
    "slurm_controller": "UP",
    "node": {
        "name": sys.argv[5],
        "state": sys.argv[3],
        "partition": sys.argv[6],
        "configured_gpus": int(sys.argv[4]),
        "minimum_required_gpus": 4,
        "same_node_for_all_jobs": True,
    },
    "unix_and_slurm_user": "guisongxia",
    "freeze_commit": sys.argv[7],
}
if target.exists():
    raise SystemExit("append-only node preflight receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

submission_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

submission_finish() {
  local rc="$?" target job_id
  trap - EXIT ERR TERM INT
  set +e
  if [[ "$rc" -ne 0 && "$SUBMISSION_COMPLETE" != true ]]; then
    for job_id in "$VALIDATION_JOB_ID" "$ARRAY_JOB_ID" "$SMOKE_JOB_ID" "$CONTRACT_JOB_ID"; do
      [[ "$job_id" =~ ^[0-9]+$ ]] && scancel "$job_id" >/dev/null 2>&1 || true
    done
    if [[ -n "$ATTEMPT" && -d "$ATTEMPT/status" ]]; then
      target="$ATTEMPT/status/submission-failure.json"
      "$PYTHON" - "$target" "$FAILURE_PHASE" "${ERROR_TEXT:-submission_failed}" "$FREEZE_B" \
        "$CONTRACT_JOB_ID" "$SMOKE_JOB_ID" "$ARRAY_JOB_ID" "$VALIDATION_JOB_ID" <<'PY' || true
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {"schema_id": "qwen3vl_gt_direct_teacher_submission_failure_v1",
           "status": "failed", "phase": sys.argv[2], "error": sys.argv[3],
           "freeze_commit": sys.argv[4],
           "confirmed_job_ids_cancel_requested": {
               "contract": sys.argv[5] or None,
               "smoke": sys.argv[6] or None,
               "pilot_array": sys.argv[7] or None,
               "validation": sys.argv[8] or None,
           }}
if not target.exists():
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
PY
    fi
    QQ_SUB="CA0804 GT-direct submission failed" \
    QQ_BODY="status=failed; phase=$FAILURE_PHASE; error=${ERROR_TEXT:-submission_failed}; freeze=$FREEZE_B; attempt=${ATTEMPT:-none}" \
      "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap submission_error ERR
trap submission_finish EXIT
trap 'FAILURE_PHASE=signal; ERROR_TEXT=received_SIGTERM; exit 143' TERM
trap 'FAILURE_PHASE=signal; ERROR_TEXT=received_SIGINT; exit 130' INT

FAILURE_PHASE=freeze_validation
verify_freeze
validate_smoke_preregistration
scontrol ping | grep -q UP || { ERROR_TEXT=slurm_controller_not_UP; exit 3; }
NODE_INFO="$(scontrol show node "$NODE")"
grep -Eq 'State=[^[:space:]]*(DOWN|DRAIN|FAIL|MAINT|UNKNOWN)' <<<"$NODE_INFO" &&
  { ERROR_TEXT=fixed_node_gpu48_unavailable; exit 3; }
grep -Eq 'Partitions=([^[:space:]]*,)*gre(,[^[:space:]]*)?([[:space:]]|$)' <<<"$NODE_INFO" ||
  { ERROR_TEXT=fixed_node_gpu48_not_in_gre; exit 3; }
grep -Eq 'CfgTRES=.*gres/gpu=([4-9]|[1-9][0-9]+)(,|[[:space:]]|$)' <<<"$NODE_INFO" ||
  { ERROR_TEXT=fixed_node_gpu48_has_fewer_than_four_gpus; exit 3; }
[[ "$NODE_INFO" =~ State=([^[:space:]]+) ]] ||
  { ERROR_TEXT=fixed_node_gpu48_state_parse_failed; exit 3; }
NODE_STATE="${BASH_REMATCH[1]}"
[[ "$NODE_INFO" =~ gres/gpu=([0-9]+) ]] ||
  { ERROR_TEXT=fixed_node_gpu48_gpu_count_parse_failed; exit 3; }
NODE_GPU_COUNT="${BASH_REMATCH[1]}"
NODE_PROBE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$MODE" == contract ]]; then
  FAILURE_PHASE=contract_reservation
  acquire_lock "$DIAGNOSTICS_FREEZE_ROOT/.contract-submit.lock" ||
    { ERROR_TEXT=concurrent_contract_submitter; exit 2; }
  if active_jobs contract; then ERROR_TEXT=active_duplicate_contract; exit 2
  elif [[ "$?" -ne 1 ]]; then ERROR_TEXT=cannot_query_active_contract_jobs; exit 3; fi
  reserve_attempt "$DIAGNOSTICS_FREEZE_ROOT" ATTEMPT ||
    { ERROR_TEXT=cannot_reserve_contract_attempt; exit 2; }
  mkdir -- "$ATTEMPT/logs" "$ATTEMPT/status" "$ATTEMPT/contract"
  write_node_preflight "$ATTEMPT/status/submission-preflight.json"
  CONTRACT_ATTEMPT="$ATTEMPT"
  FREEZE_MANIFEST="$ATTEMPT/freeze_manifest.json"
  CONTRACT_RECEIPT="$ATTEMPT/contract/receipt.json"
  create_manifest
  MANIFEST_SHA256="$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')"
  FAILURE_PHASE=contract_submission
  verify_freeze
  ATTEMPT_NAME="${ATTEMPT##*/}"
  JOB_NAME="CA0804TeacherContract-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"
  JOB_RAW="$(sbatch --parsable --export=NIL --job-name="$JOB_NAME" \
    --partition="$PARTITION" --nodelist="$NODE" --cpus-per-task=2 --mem=16G --time=00:45:00 \
    --output="$ATTEMPT/logs/contract-%j.out" --error="$ATTEMPT/logs/contract-%j.err" \
    "$ROOT/tools/run_qwen3vl_gt_direct_teacher_contract.slurm" \
    "$ROOT" "$ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256")"
  CONTRACT_JOB_ID="${JOB_RAW%%;*}"; CONTRACT_JOB_ID="${CONTRACT_JOB_ID%%$'\n'*}"
  [[ "$CONTRACT_JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=malformed_contract_job_id; exit 3; }
  write_record "$ATTEMPT/contract_submission.json" contract
  SUBMISSION_COMPLETE=true
  printf 'contract_job=%s\ndiagnostics_attempt=%s\nfreeze_B=%s\nimplementation_A=%s\n' \
    "$CONTRACT_JOB_ID" "$ATTEMPT" "$FREEZE_B" "$IMPLEMENTATION_A"
  exit 0
fi

if [[ "$MODE" == smoke ]]; then
  FAILURE_PHASE=smoke_preflight
  acquire_lock "$DIAGNOSTICS_ROOT/.gpu-submit.lock" ||
    { ERROR_TEXT=concurrent_gpu_submitter; exit 2; }
  select_contract
  if active_jobs smoke; then ERROR_TEXT=active_duplicate_smoke_or_formal_job; exit 2
  elif [[ "$?" -ne 1 ]]; then ERROR_TEXT=cannot_query_active_gpu_jobs; exit 3; fi
  verify_freeze
  FAILURE_PHASE=smoke_attempt_reservation
  reserve_attempt "$SMOKE_ROOT" ATTEMPT ||
    { ERROR_TEXT=cannot_reserve_smoke_attempt; exit 2; }
  mkdir -- "$ATTEMPT/logs" "$ATTEMPT/status" "$ATTEMPT/smoke"
  write_node_preflight "$ATTEMPT/status/submission-preflight.json"
  SMOKE_ATTEMPT="$ATTEMPT"
  SMOKE_RECEIPT="$ATTEMPT/status/smoke.json"
  ATTEMPT_NAME="${ATTEMPT##*/}"
  SMOKE_NAME="CA0804TeacherSmoke-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"
  FAILURE_PHASE=smoke_submission
  verify_freeze
  SMOKE_RAW="$(sbatch --parsable --export=NIL --job-name="$SMOKE_NAME" \
    --partition="$PARTITION" --nodelist="$NODE" --gres=gpu:1 \
    --cpus-per-task=4 --mem=48G --time=12:00:00 \
    --output="$ATTEMPT/logs/smoke-%j.out" --error="$ATTEMPT/logs/smoke-%j.err" \
    "$ROOT/tools/run_qwen3vl_gt_direct_teacher_smoke.slurm" \
    "$ROOT" "$ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256" \
    "$CONTRACT_RECEIPT" "$CONTRACT_SHA256")"
  SMOKE_JOB_ID="${SMOKE_RAW%%;*}"; SMOKE_JOB_ID="${SMOKE_JOB_ID%%$'\n'*}"
  [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=malformed_smoke_job_id; exit 3; }
  FAILURE_PHASE=smoke_submission_record
  write_record "$ATTEMPT/smoke_submission.json" smoke
  verify_freeze
  SUBMISSION_COMPLETE=true
  printf 'smoke_job=%s\nsmoke_attempt=%s\ndiagnostics_contract=%s\nsample=train_407_r3c2\npeak_gpus=1\n' \
    "$SMOKE_JOB_ID" "$ATTEMPT" "$CONTRACT_ATTEMPT"
  exit 0
fi

FAILURE_PHASE=pilot_preflight
acquire_lock "$DIAGNOSTICS_ROOT/.gpu-submit.lock" ||
  { ERROR_TEXT=concurrent_gpu_submitter; exit 2; }
select_contract
select_smoke
if active_jobs pilot; then ERROR_TEXT=active_duplicate_pilot_or_validator; exit 2
elif [[ "$?" -ne 1 ]]; then ERROR_TEXT=cannot_query_active_pilot_jobs; exit 3; fi
verify_freeze
FAILURE_PHASE=formal_attempt_reservation
reserve_attempt "$CAMPAIGN_ROOT" ATTEMPT ||
  { ERROR_TEXT=cannot_reserve_formal_attempt; exit 2; }
mkdir -- "$ATTEMPT/logs" "$ATTEMPT/status" "$ATTEMPT/shards"
write_node_preflight "$ATTEMPT/status/submission-preflight.json"
ATTEMPT_NAME="${ATTEMPT##*/}"
PILOT_NAME="CA0804TeacherPilot-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"
VALIDATE_NAME="CA0804TeacherValidate-${FREEZE_B:0:8}-${ATTEMPT_NAME#attempt-}"

FAILURE_PHASE=pilot_array_submission
verify_freeze
ARRAY_RAW="$(sbatch --parsable --export=NIL --job-name="$PILOT_NAME" \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-3%4 --gres=gpu:1 \
  --cpus-per-task=4 --mem=48G --time=12:00:00 \
  --output="$ATTEMPT/logs/pilot-%A_%a.out" --error="$ATTEMPT/logs/pilot-%A_%a.err" \
  "$ROOT/tools/run_qwen3vl_gt_direct_teacher_pilot.slurm" \
  "$ROOT" "$ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
  "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256" \
  "$CONTRACT_RECEIPT" "$CONTRACT_SHA256" "$SMOKE_RECEIPT" "$SMOKE_SHA256")"
ARRAY_JOB_ID="${ARRAY_RAW%%;*}"; ARRAY_JOB_ID="${ARRAY_JOB_ID%%$'\n'*}"
[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=malformed_array_job_id; exit 3; }

FAILURE_PHASE=validator_submission
verify_freeze
VALIDATE_RAW="$(sbatch --parsable --export=NIL --dependency="afterany:$ARRAY_JOB_ID" \
  --job-name="$VALIDATE_NAME" --partition="$PARTITION" --nodelist="$NODE" \
  --cpus-per-task=2 --mem=16G --time=00:45:00 \
  --output="$ATTEMPT/logs/validation-%j.out" --error="$ATTEMPT/logs/validation-%j.err" \
  "$ROOT/tools/run_qwen3vl_gt_direct_teacher_validate.slurm" \
  "$ROOT" "$ATTEMPT" "$CONFIG" "$PYTHON" "$QQ_NOTIFIER" "$FREEZE_B" \
  "$IMPLEMENTATION_A" "$CONFIG_SHA256" "$FREEZE_MANIFEST" "$MANIFEST_SHA256" \
  "$CONTRACT_RECEIPT" "$CONTRACT_SHA256" "$SMOKE_RECEIPT" "$SMOKE_SHA256" \
  "$ARRAY_JOB_ID")"
VALIDATION_JOB_ID="${VALIDATE_RAW%%;*}"; VALIDATION_JOB_ID="${VALIDATION_JOB_ID%%$'\n'*}"
[[ "$VALIDATION_JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=malformed_validation_job_id; exit 3; }

FAILURE_PHASE=submission_record
write_record "$ATTEMPT/pilot_submission.json" pilot
verify_freeze
SUBMISSION_COMPLETE=true
printf 'pilot_array_job=%s\nvalidation_job=%s\ndependency=afterany\nformal_attempt=%s\ndiagnostics_contract=%s\nsmoke_receipt=%s\npeak_gpus=4\n' \
  "$ARRAY_JOB_ID" "$VALIDATION_JOB_ID" "$ATTEMPT" "$CONTRACT_ATTEMPT" "$SMOKE_RECEIPT"
