#!/usr/bin/env bash
#SBATCH --job-name=CA0803TrajPilot
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -Eeuo pipefail
umask 027
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ROOT="${1:-}"
ATTEMPT="${2:-}"
CONFIG="${3:-}"
PYTHON="${4:-}"
QQ_NOTIFIER="${5:-}"
FREEZE_B="${6:-missing}"
IMPLEMENTATION_A="${7:-missing}"
EXPECTED_CONFIG_SHA256="${8:-missing}"
FREEZE_MANIFEST="${9:-}"
EXPECTED_MANIFEST_SHA256="${10:-missing}"
CONTRACT_RECEIPT="${11:-}"
EXPECTED_CONTRACT_SHA256="${12:-missing}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-missing}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-missing}"
JOB_ID="${SLURM_JOB_ID:-missing}"
START_SECONDS="$SECONDS"
STATUS=failed
RESULT=not_started
ERROR_TEXT=""
SHARD_ROOT="$ATTEMPT/shards/shard-$SHARD_INDEX"
STATUS_RECEIPT="$ATTEMPT/status/pilot-shard-$SHARD_INDEX.json"
BOOTSTRAP_ERROR=argument_resolution_failed

bootstrap_error() {
  local rc="$?"
  BOOTSTRAP_ERROR="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

bootstrap_finish() {
  local rc="$?" state=failed fallback_python target
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  fallback_python="$PYTHON"
  [[ -x "$fallback_python" ]] || fallback_python=/usr/bin/python3
  if [[ "$rc" -ne 0 && "$ATTEMPT" == /* && -d "$ATTEMPT/status" \
    && -x "$fallback_python" ]]; then
    target="$ATTEMPT/status/bootstrap-pilot-${SHARD_INDEX}-${JOB_ID}.json"
    "$fallback_python" - "$target" "$state" "$rc" "$JOB_ID" "$ARRAY_JOB_ID" \
      "$SHARD_INDEX" "$FREEZE_B" "$IMPLEMENTATION_A" "$BOOTSTRAP_ERROR" <<'PY' || true
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_worker_bootstrap_terminal_v1",
    "worker": "pilot_shard",
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "job_id": sys.argv[4],
    "array_job_id": sys.argv[5],
    "shard_index": sys.argv[6],
    "freeze_commit": sys.argv[7],
    "implementation_commit": sys.argv[8],
    "error": sys.argv[9],
}
if not target.exists():
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
PY
  fi
  if [[ "$rc" -ne 0 && -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0803 oracle trajectory shard $SHARD_INDEX bootstrap" \
    QQ_BODY="status=$state; error=$BOOTSTRAP_ERROR; job=$JOB_ID; shard=$SHARD_INDEX/3; attempt=$ATTEMPT" \
      "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap bootstrap_error ERR
trap bootstrap_finish EXIT
trap 'BOOTSTRAP_ERROR=received_SIGTERM; exit 143' TERM
trap 'BOOTSTRAP_ERROR=received_SIGINT; exit 130' INT

ROOT="$(realpath -e -- "${ROOT:?missing ROOT}")"
ATTEMPT="$(realpath -e -- "${ATTEMPT:?missing ATTEMPT}")"
CONFIG="$(realpath -e -- "${CONFIG:?missing CONFIG}")"
PYTHON="$(realpath -e -- "${PYTHON:?missing PYTHON}")"
QQ_NOTIFIER="$(realpath -e -- "${QQ_NOTIFIER:?missing QQ notifier}")"
[[ "$FREEZE_B" != missing && "$IMPLEMENTATION_A" != missing \
  && "$EXPECTED_CONFIG_SHA256" != missing && "$EXPECTED_MANIFEST_SHA256" != missing \
  && "$EXPECTED_CONTRACT_SHA256" != missing ]] || false
FREEZE_MANIFEST="$(realpath -e -- "${FREEZE_MANIFEST:?missing freeze manifest}")"
CONTRACT_RECEIPT="$(realpath -e -- "${CONTRACT_RECEIPT:?missing contract receipt}")"
SHARD_ROOT="$ATTEMPT/shards/shard-$SHARD_INDEX"
STATUS_RECEIPT="$ATTEMPT/status/pilot-shard-$SHARD_INDEX.json"

reject_sensitive_environment() {
  local name upper ignored
  while IFS='=' read -r name ignored; do
    upper="${name^^}"
    case "$upper" in
      *_API_KEY|*_ACCESS_KEY|*_TOKEN|*_ACCESS_TOKEN|*_AUTH_TOKEN|*_SECRET|*_PASSWORD|*_PRIVATE_KEY)
        ERROR_TEXT="sensitive_environment_inherited_${name}"
        return 1
        ;;
    esac
  done < <(env)
}

verify_binding() {
  local observed_delta
  [[ "$(id -un)" == guisongxia ]] || { ERROR_TEXT=wrong_unix_user; return 1; }
  [[ "${SLURM_JOB_USER:-}" == guisongxia ]] || { ERROR_TEXT=wrong_slurm_user; return 1; }
  [[ "$(hostname -s)" == gpu48 && "${SLURM_JOB_NODELIST:-}" == gpu48 ]] \
    || { ERROR_TEXT=wrong_slurm_node; return 1; }
  [[ "${SLURM_JOB_PARTITION:-}" == gre ]] || { ERROR_TEXT=wrong_slurm_partition; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$FREEZE_B" ]] \
    || { ERROR_TEXT=freeze_commit_drift; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_A" ]] \
    || { ERROR_TEXT=implementation_parent_drift; return 1; }
  observed_delta="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r \
    --no-renames "$IMPLEMENTATION_A" "$FREEZE_B" | LC_ALL=C sort)"
  [[ "$observed_delta" == "configs/qwen3vl_oracle_trajectory_pilot_0803.json"$'\n'"doc/training_pipeline/0803_fourth_try/QWEN3VL_ORACLE_TRAJECTORY_REACHABILITY_0803.md" ]] \
    || { ERROR_TEXT=freeze_delta_not_exact_config_and_0803_doc; return 1; }
  if git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
    ERROR_TEXT=run_worktree_not_detached
    return 1
  fi
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
    || { ERROR_TEXT=run_worktree_dirty; return 1; }
  [[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] \
    || { ERROR_TEXT=config_hash_drift; return 1; }
  [[ "$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_SHA256" ]] \
    || { ERROR_TEXT=freeze_manifest_hash_drift; return 1; }
  [[ "$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_CONTRACT_SHA256" ]] \
    || { ERROR_TEXT=contract_receipt_hash_drift; return 1; }
  "$PYTHON" - "$ROOT" "$CONFIG" "$FREEZE_MANIFEST" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$CONTRACT_RECEIPT" \
    "$EXPECTED_MANIFEST_SHA256" "$ATTEMPT" <<'PY'
import hashlib, json, pathlib, sys
root, config, manifest_path = map(pathlib.Path, sys.argv[1:4])
freeze_b, implementation_a, config_sha = sys.argv[4:7]
receipt_path = pathlib.Path(sys.argv[7]).resolve()
manifest_sha, formal_attempt_raw = sys.argv[8:10]
formal_attempt = pathlib.Path(formal_attempt_raw).resolve()
contract_attempt = receipt_path.parent.parent
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest_path.resolve().parent != contract_attempt:
    raise SystemExit("freeze manifest is not inside receipt diagnostics attempt")
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
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec.get("sha256"):
        raise SystemExit(f"frozen file mismatch: {name}")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("schema_id") != "qwen3vl_oracle_trajectory_contract_receipt_v3":
    raise SystemExit("contract receipt schema mismatch")
if receipt.get("status") != "success" or receipt.get("result") != "all_contracts_passed":
    raise SystemExit("contract did not pass")
if receipt.get("freeze_commit") != freeze_b or receipt.get("implementation_commit") != implementation_a:
    raise SystemExit("contract commit binding mismatch")
if receipt.get("config_sha256") != config_sha:
    raise SystemExit("contract config binding mismatch")
if receipt.get("freeze_manifest", {}).get("sha256") != manifest_sha:
    raise SystemExit("contract manifest binding mismatch")
if receipt.get("diagnostics_contract_attempt") != str(contract_attempt):
    raise SystemExit("contract diagnostics attempt binding mismatch")
if receipt.get("frozen_files") != manifest.get("files"):
    raise SystemExit("contract worker hashes differ from freeze manifest")
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
        raise SystemExit(f"contract preflight {label} binding mismatch")
tests = receipt.get("tests", {})
required = {"targeted_runner", "targeted_validator", "full_regression"}
if set(tests) != required:
    raise SystemExit("contract test set mismatch")
for name in sorted(required):
    entry = tests[name]
    if entry.get("passed") is not True or entry.get("test_count", 0) <= 0:
        raise SystemExit(f"contract tests did not pass: {name}")
    if entry.get("failures") != 0 or entry.get("errors") != 0:
        raise SystemExit(f"contract test failure count differs: {name}")
    path = (contract_attempt / entry.get("path", "")).resolve()
    try:
        path.relative_to(contract_attempt)
    except ValueError:
        raise SystemExit(f"contract log escapes attempt: {name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("size_bytes"):
        raise SystemExit(f"contract log binding mismatch: {name}")
PY
}

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" elapsed state marker_hash marker_size
  trap - EXIT ERR TERM INT
  elapsed="$((SECONDS - START_SECONDS))"
  state=failed
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  STATUS="$state"
  marker_hash=""
  marker_size=0
  if [[ -f "$SHARD_ROOT/generation_complete.json" ]]; then
    marker_hash="$(sha256sum -- "$SHARD_ROOT/generation_complete.json" | awk '{print $1}')"
    marker_size="$(stat -c %s -- "$SHARD_ROOT/generation_complete.json")"
  fi
  "$PYTHON" - "$STATUS_RECEIPT" "$STATUS" "$rc" "$JOB_ID" "$ARRAY_JOB_ID" \
    "$SHARD_INDEX" "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONTRACT_SHA256" "$RESULT" "$ERROR_TEXT" \
    "$elapsed" "$marker_hash" "$marker_size" "$SHARD_ROOT" "$CONTRACT_RECEIPT" <<'PY' || true
import json, os, pathlib, sys
(
    output_raw, status, exit_code, job_id, array_job_id, shard_index, freeze_b,
    implementation_a, config_sha, manifest_sha, contract_sha, result, error,
    elapsed, marker_sha, marker_size, shard_root, contract_receipt,
) = sys.argv[1:]
contract_receipt_path = pathlib.Path(contract_receipt).resolve()
output = pathlib.Path(output_raw)
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_pilot_shard_receipt_v1",
    "status": status,
    "exit_code": int(exit_code),
    "job_id": job_id,
    "array_job_id": array_job_id,
    "shard_index": int(shard_index) if shard_index.isdigit() else shard_index,
    "slurm": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "gpus": 1, "accelerator": "A800", "bf16_required": True},
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest_sha256": manifest_sha,
    "contract_receipt_sha256": contract_sha,
    "contract_receipt_schema": "qwen3vl_oracle_trajectory_contract_receipt_v3",
    "contract_preflight_receipt_and_log_reverified": True,
    "diagnostics_contract_attempt": str(contract_receipt_path.parent.parent),
    "diagnostics_contract_receipt": {"path": str(contract_receipt_path), "sha256": contract_sha},
    "result": result,
    "error": error or None,
    "elapsed_seconds": int(elapsed),
    "shard_root": shard_root,
    "generation_complete": {
        "sha256": marker_sha or None,
        "size_bytes": int(marker_size),
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise SystemExit("pilot shard receipt already exists")
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
  QQ_SUB="CA0803 oracle trajectory shard $SHARD_INDEX" \
  QQ_BODY="status=$STATUS; result=$RESULT; error=${ERROR_TEXT:-none}; job=$JOB_ID; shard=$SHARD_INDEX/3; elapsed=${elapsed}s; output=$SHARD_ROOT" \
    "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  exit "$rc"
}

trap record_error ERR
trap finish EXIT
trap 'ERROR_TEXT=received_SIGTERM; exit 143' TERM
trap 'ERROR_TEXT=received_SIGINT; exit 130' INT

[[ "$SHARD_INDEX" =~ ^[0-2]$ && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
  && "${SLURM_ARRAY_TASK_MAX:-}" == 2 && "${SLURM_ARRAY_TASK_COUNT:-}" == 3 ]] \
  || { ERROR_TEXT=invalid_three_shard_array_identity; exit 2; }
[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ && "$JOB_ID" =~ ^[0-9]+$ ]] \
  || { ERROR_TEXT=invalid_slurm_job_identity; exit 2; }
[[ ! -e "$SHARD_ROOT" && ! -e "$STATUS_RECEIPT" ]] \
  || { ERROR_TEXT=append_only_shard_output_exists; exit 2; }
reject_sensitive_environment
verify_binding

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
if "A800" not in name.upper():
    raise SystemExit(f"expected NVIDIA A800, found {name}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit(f"BF16 is not supported by {name}")
print(f"gpu_contract=ok;accelerator=A800;bf16=true;visible_gpus=1")
PY

unset OPENAI_API_KEY DASHSCOPE_API_KEY ANTHROPIC_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

RESULT=running_real_tool_oracle_trajectory
"$PYTHON" "$ROOT/tools/run_qwen3vl_oracle_trajectory.py" \
  --config "$CONFIG" \
  --output "$SHARD_ROOT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards 3 \
  --runtime-commit "$FREEZE_B"
[[ -f "$SHARD_ROOT/generation_complete.json" ]] \
  || { ERROR_TEXT=generation_complete_marker_missing; exit 2; }
verify_binding
RESULT=shard_complete
