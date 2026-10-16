#!/usr/bin/env bash
#SBATCH --job-name=CA0804TeacherSmoke
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -Eeuo pipefail
umask 027
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ROOT_RAW="${1:-}"
ATTEMPT_RAW="${2:-}"
CONFIG_RAW="${3:-}"
PYTHON_RAW="${4:-}"
QQ_RAW="${5:-}"
FREEZE_B="${6:-missing}"
IMPLEMENTATION_A="${7:-missing}"
EXPECTED_CONFIG_SHA256="${8:-missing}"
MANIFEST_RAW="${9:-}"
EXPECTED_MANIFEST_SHA256="${10:-missing}"
CONTRACT_RAW="${11:-}"
EXPECTED_CONTRACT_SHA256="${12:-missing}"
JOB_ID="${SLURM_JOB_ID:-missing}"
START_SECONDS="$SECONDS"
RESULT=starting
ERROR_TEXT=""
ROOT="$ROOT_RAW"
ATTEMPT="$ATTEMPT_RAW"
CONFIG="$CONFIG_RAW"
PYTHON="$PYTHON_RAW"
QQ_NOTIFIER="$QQ_RAW"
FREEZE_MANIFEST="$MANIFEST_RAW"
CONTRACT_RECEIPT="$CONTRACT_RAW"
SHARD_ROOT="${ATTEMPT_RAW:+$ATTEMPT_RAW/smoke/shard-0}"
STATUS_RECEIPT="${ATTEMPT_RAW:+$ATTEMPT_RAW/status/smoke.json}"

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" state=failed elapsed fallback_python marker_sha marker_size index_sha index_size
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  elapsed="$((SECONDS - START_SECONDS))"
  fallback_python="$PYTHON"
  [[ -x "$fallback_python" ]] || fallback_python=/usr/bin/python3
  marker_sha=""; marker_size=0; index_sha=""; index_size=0
  if [[ -n "$SHARD_ROOT" && -f "$SHARD_ROOT/generation_complete.json" ]]; then
    marker_sha="$(sha256sum -- "$SHARD_ROOT/generation_complete.json" | awk '{print $1}')"
    marker_size="$(stat -c %s -- "$SHARD_ROOT/generation_complete.json")"
  fi
  if [[ -n "$SHARD_ROOT" && -f "$SHARD_ROOT/shard_index.json" ]]; then
    index_sha="$(sha256sum -- "$SHARD_ROOT/shard_index.json" | awk '{print $1}')"
    index_size="$(stat -c %s -- "$SHARD_ROOT/shard_index.json")"
  fi
  if [[ -n "$STATUS_RECEIPT" && "$STATUS_RECEIPT" == /* && -x "$fallback_python" ]]; then
    mkdir -p -- "$(dirname -- "$STATUS_RECEIPT")" 2>/dev/null || true
    "$fallback_python" - "$STATUS_RECEIPT" "$state" "$rc" "$JOB_ID" "$FREEZE_B" \
      "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_MANIFEST_SHA256" \
      "$EXPECTED_CONTRACT_SHA256" "$RESULT" "$ERROR_TEXT" "$elapsed" "$SHARD_ROOT" \
      "$marker_sha" "$marker_size" "$index_sha" "$index_size" "$CONTRACT_RECEIPT" <<'PY' || true
import json, os, pathlib, sys
(
    target_raw, status, exit_code, job_id, freeze_b, implementation_a,
    config_sha, manifest_sha, contract_sha, result, error, elapsed, shard_root,
    marker_sha, marker_size, index_sha, index_size, contract_receipt,
) = sys.argv[1:]
target = pathlib.Path(target_raw)
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_smoke_receipt_v1",
    "status": status,
    "exit_code": int(exit_code),
    "result": result,
    "error": error or None,
    "job_id": job_id,
    "elapsed_seconds": int(elapsed),
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
    "diagnostic_smoke": True,
    "scientific_result_authoritative": False,
    "bounded_protocol": {
        "sample_id": "train_407_r3c2",
        "geometry_modes_executed": ["matched"],
        "step_indices_executed": [1],
        "max_real_candidates": 2,
        "executed_tool_types": ["sam3", "simpleclick"],
        "executed_action_types": ["box", "positive_point"],
        "endpoint_search_performed": False,
    },
    "resources": {
        "user": "guisongxia", "node": "gpu48", "partition": "gre",
        "gpus": 1, "time": "12:00:00",
    },
    "output_scope": "diagnostics_only",
    "shard_root": shard_root,
    "generation_complete": {"sha256": marker_sha or None, "size_bytes": int(marker_size)},
    "shard_index_artifact": {"sha256": index_sha or None, "size_bytes": int(index_size)},
}
if target.exists():
    raise SystemExit("append-only smoke receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  fi
  if [[ -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0804 GT-direct 1-GPU smoke" \
    QQ_BODY="status=$state; result=$RESULT; error=${ERROR_TEXT:-none}; job=$JOB_ID; sample=train_407_r3c2; elapsed=${elapsed}s; output=$SHARD_ROOT" \
      "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap record_error ERR
trap finish EXIT
trap 'ERROR_TEXT=received_SIGTERM; exit 143' TERM
trap 'ERROR_TEXT=received_SIGINT; exit 130' INT

ROOT="$(realpath -e -- "${ROOT_RAW:?missing ROOT}")"
ATTEMPT="$(realpath -e -- "${ATTEMPT_RAW:?missing ATTEMPT}")"
CONFIG="$(realpath -e -- "${CONFIG_RAW:?missing CONFIG}")"
PYTHON="$(realpath -e -- "${PYTHON_RAW:?missing PYTHON}")"
QQ_NOTIFIER="$(realpath -e -- "${QQ_RAW:?missing QQ notifier}")"
FREEZE_MANIFEST="$(realpath -e -- "${MANIFEST_RAW:?missing freeze manifest}")"
CONTRACT_RECEIPT="$(realpath -e -- "${CONTRACT_RAW:?missing contract receipt}")"
SHARD_ROOT="$ATTEMPT/smoke/shard-0"
STATUS_RECEIPT="$ATTEMPT/status/smoke.json"

reject_sensitive_environment() {
  local name upper ignored
  while IFS='=' read -r name ignored; do
    upper="${name^^}"
    case "$upper" in
      *_API_KEY|*_ACCESS_KEY|*_TOKEN|*_ACCESS_TOKEN|*_AUTH_TOKEN|*_SECRET|*_PASSWORD|*_PRIVATE_KEY)
        ERROR_TEXT="sensitive_environment_inherited_${name}"; return 1 ;;
    esac
  done < <(env)
}

verify_binding() {
  local observed
  [[ "$(id -un)" == guisongxia && "${SLURM_JOB_USER:-}" == guisongxia ]] ||
    { ERROR_TEXT=wrong_unix_or_slurm_user; return 1; }
  [[ "$(hostname -s)" == gpu48 && "${SLURM_JOB_NODELIST:-}" == gpu48 &&
    "${SLURM_JOB_PARTITION:-}" == gre ]] ||
    { ERROR_TEXT=wrong_node_or_partition; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$FREEZE_B" &&
    "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_A" ]] ||
    { ERROR_TEXT=freeze_parent_binding_failed; return 1; }
  observed="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r --no-renames \
    "$IMPLEMENTATION_A" "$FREEZE_B" | LC_ALL=C sort)"
  [[ "$observed" == "configs/qwen3vl_gt_direct_teacher_0804.json"$'\n'"doc/training_pipeline/0804_fifith_try/0804.md" ]] ||
    { ERROR_TEXT=freeze_delta_not_exact_config_and_0804_doc; return 1; }
  ! git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1 ||
    { ERROR_TEXT=run_worktree_not_detached; return 1; }
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] ||
    { ERROR_TEXT=run_worktree_dirty; return 1; }
  [[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" &&
    "$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_SHA256" &&
    "$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_CONTRACT_SHA256" ]] ||
    { ERROR_TEXT=config_manifest_or_contract_hash_drift; return 1; }
  "$PYTHON" - "$ROOT" "$CONFIG" "$FREEZE_MANIFEST" "$CONTRACT_RECEIPT" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_MANIFEST_SHA256" <<'PY'
import hashlib, json, pathlib, sys
root, config_path, manifest_path, receipt_path = map(pathlib.Path, sys.argv[1:5])
freeze_b, implementation_a, config_sha, manifest_sha = sys.argv[5:9]
config = json.loads(config_path.read_text(encoding="utf-8"))
smoke = config.get("resource_plan", {}).get("diagnostic_smoke", {})
expected_smoke = {
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
if smoke != expected_smoke:
    raise SystemExit("frozen smoke preregistration differs")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if manifest.get("freeze_commit") != freeze_b or manifest.get("implementation_commit") != implementation_a:
    raise SystemExit("manifest binding")
if receipt.get("schema_id") != "qwen3vl_gt_direct_teacher_contract_receipt_v1":
    raise SystemExit("contract schema")
if receipt.get("status") != "success" or receipt.get("result") != "all_contracts_passed":
    raise SystemExit("contract failed")
if receipt.get("freeze_commit") != freeze_b or receipt.get("implementation_commit") != implementation_a:
    raise SystemExit("contract commits")
if receipt.get("config_sha256") != config_sha or receipt.get("freeze_manifest_sha256") != manifest_sha:
    raise SystemExit("contract hashes")
if receipt.get("preflight", {}).get("passed") is not True:
    raise SystemExit("contract preflight did not pass")
for name in ("targeted_runner", "targeted_validator", "full_regression"):
    entry = receipt.get("logs", {}).get(name, {})
    if entry.get("passed") is not True or entry.get("test_count", 0) <= 0:
        raise SystemExit(f"contract test evidence missing: {name}")
for spec in manifest.get("files", {}).values():
    data = (root / spec["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != spec["size_bytes"]:
        raise SystemExit("frozen file drift")
PY
}

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=not_a_slurm_job; exit 2; }
[[ ! -e "$SHARD_ROOT" && ! -e "$STATUS_RECEIPT" ]] ||
  { ERROR_TEXT=append_only_smoke_output_exists; exit 2; }
reject_sensitive_environment
verify_binding
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU lacks BF16 support")
print(f"gpu_contract=ok;device={torch.cuda.get_device_name(0)};visible_gpus=1")
PY
unset OPENAI_API_KEY DASHSCOPE_API_KEY ANTHROPIC_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
RESULT=running_real_tool_smoke
"$PYTHON" tools/run_qwen3vl_gt_direct_teacher.py --config "$CONFIG" \
  --output "$SHARD_ROOT" --shard-index 0 --num-shards 4 --runtime-commit "$FREEZE_B" \
  --diagnostic-smoke
[[ -f "$SHARD_ROOT/generation_complete.json" && -f "$SHARD_ROOT/shard_index.json" ]] ||
  { ERROR_TEXT=smoke_completion_artifacts_missing; exit 2; }
"$PYTHON" - "$SHARD_ROOT/generation_complete.json" "$SHARD_ROOT/shard_index.json" "$FREEZE_B" "$SHARD_ROOT" <<'PY'
import hashlib, json, pathlib, sys
marker_path = pathlib.Path(sys.argv[1])
index_path = pathlib.Path(sys.argv[2])
marker = json.loads(marker_path.read_text(encoding="utf-8"))
index = json.loads(index_path.read_text(encoding="utf-8"))
freeze_b = sys.argv[3]
shard_root = pathlib.Path(sys.argv[4]).resolve()
expected_schemas = (
    "qwen3vl_gt_direct_teacher_diagnostic_smoke_complete_v1",
    "qwen3vl_gt_direct_teacher_diagnostic_smoke_index_v1",
)
expected_contract = {
    "sample_id": "train_407_r3c2",
    "fresh_sam3_initialization": True,
    "geometry_modes_executed": ["matched"],
    "step_indices_executed": [1],
    "max_real_candidates": 2,
    "independent_candidates_from_same_fresh_state": True,
    "endpoint_search_performed": False,
    "candidate_selection_or_replay_performed": False,
}
expected_counts = {
    "episodes": 1,
    "candidates": 2,
    "candidates_executed": 2,
    "sam3_box_candidates_executed": 1,
    "simpleclick_positive_point_candidates_executed": 1,
}
for payload, schema in zip((marker, index), expected_schemas):
    if payload.get("schema_id") != schema:
        raise SystemExit("smoke payload schema differs")
    if payload.get("status") != "completed":
        raise SystemExit("smoke payload status differs")
    if payload.get("runtime_commit") != freeze_b:
        raise SystemExit("smoke runtime commit differs")
    if payload.get("shard_index") != 0 or payload.get("num_shards") != 4:
        raise SystemExit("smoke shard identity differs")
    if payload.get("episode_ids") != ["train_407_r3c2"]:
        raise SystemExit("smoke sample differs")
    if payload.get("teacher_only") is not True or payload.get("training_authorized") is not False:
        raise SystemExit("smoke safety contract differs")
    if payload.get("qwen_invoked") is not False or payload.get("qwen_calls") != 0:
        raise SystemExit("smoke Qwen audit differs")
    if payload.get("diagnostic_smoke") is not True:
        raise SystemExit("bounded diagnostic smoke flag differs")
    if payload.get("scientific_result_authoritative") is not False:
        raise SystemExit("diagnostic smoke was marked scientifically authoritative")
    if payload.get("output_scope") != "diagnostics_only":
        raise SystemExit("diagnostic smoke output scope differs")
    if payload.get("diagnostic_contract") != expected_contract:
        raise SystemExit("bounded diagnostic smoke contract differs")
    if payload.get("counts") != expected_counts:
        raise SystemExit("bounded diagnostic smoke counts differ")
    if payload.get("executed_tool_types") != ["sam3", "simpleclick"]:
        raise SystemExit("diagnostic smoke tool coverage differs")
    if payload.get("executed_action_types") != ["box", "positive_point"]:
        raise SystemExit("diagnostic smoke action coverage differs")
    for field in (
        "all_gt_geometry_authorized",
        "all_real_tool_receipts_present",
        "hard_gate_flags_observed",
        "rgb_semantic_gate_flags_observed",
    ):
        if payload.get(field) is not True:
            raise SystemExit(f"diagnostic smoke audit failed: {field}")
if marker.get("shard_index_sha256") != hashlib.sha256(index_path.read_bytes()).hexdigest():
    raise SystemExit("diagnostic smoke index hash differs")
episodes = index.get("episodes")
if not isinstance(episodes, list) or len(episodes) != 1:
    raise SystemExit("diagnostic smoke episode identity differs")
episode_spec = episodes[0]
if episode_spec.get("episode_id") != "train_407_r3c2":
    raise SystemExit("diagnostic smoke episode ID differs")
episode_path = (shard_root / episode_spec.get("path", "")).resolve()
if shard_root not in episode_path.parents or not episode_path.is_file():
    raise SystemExit("diagnostic smoke episode path differs")
episode_data = episode_path.read_bytes()
if hashlib.sha256(episode_data).hexdigest() != episode_spec.get("sha256"):
    raise SystemExit("diagnostic smoke episode hash differs")
episode = json.loads(episode_data)
if (
    episode.get("diagnostic_smoke") is not True
    or episode.get("scientific_result_authoritative") is not False
    or episode.get("teacher_only") is not True
    or episode.get("training_authorized") is not False
    or episode.get("qwen_invoked") is not False
    or episode.get("qwen_calls") != 0
):
    raise SystemExit("diagnostic smoke episode safety flags differ")
contract = episode.get("diagnostic_contract", {})
if (
    contract.get("geometry_mode") != "matched"
    or contract.get("step_indices_executed") != [1]
    or contract.get("max_real_candidates") != 2
    or contract.get("endpoint_search_performed") is not False
):
    raise SystemExit("diagnostic smoke episode bounds differ")
candidates = episode.get("candidates")
if not isinstance(candidates, list) or len(candidates) != 2:
    raise SystemExit("diagnostic smoke candidate count differs")
for candidate, expected_tool, expected_action in zip(
    candidates, ("sam3", "simpleclick"), ("box", "positive_point")
):
    if candidate.get("action", {}).get("action") != expected_action:
        raise SystemExit("diagnostic candidate action differs")
    receipt = candidate.get("tool_receipt", {})
    if (
        receipt.get("tool") != expected_tool
        or not isinstance(receipt.get("tool_input"), dict)
        or not isinstance(receipt.get("tool_result"), dict)
    ):
        raise SystemExit("diagnostic real-tool receipt differs")
    boundary = candidate.get("teacher_boundary_audit", {})
    if (
        boundary.get("geometry_match") is not True
        or boundary.get("raw_label_pixels_passed_to_environment") is not False
        or boundary.get("raw_label_pixels_passed_to_tools") is not False
        or boundary.get("raw_label_path_passed_to_runtime") is not False
        or boundary.get("gt_derived_geometry_passed_to_environment") is not True
        or boundary.get("gt_derived_geometry_passed_to_tools") is not True
        or boundary.get("gt_derived_metadata_passed_to_environment") is not True
        or boundary.get("executor_receipt_contains_teacher_encoded_region_id") is not True
        or not isinstance(boundary.get("runtime_boundary_layers"), dict)
    ):
        raise SystemExit("diagnostic GT geometry authorization differs")
    if type(candidate.get("hard_gate", {}).get("passed")) is not bool:
        raise SystemExit("diagnostic hard-gate flag missing")
    if type(candidate.get("rgb_semantic_gate", {}).get("passed")) is not bool:
        raise SystemExit("diagnostic RGB-gate flag missing")
PY
verify_binding
RESULT=smoke_complete
