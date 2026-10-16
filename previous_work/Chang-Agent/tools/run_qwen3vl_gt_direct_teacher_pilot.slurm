#!/usr/bin/env bash
#SBATCH --job-name=CA0804TeacherPilot
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
SMOKE_RAW="${13:-}"
EXPECTED_SMOKE_SHA256="${14:-missing}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-missing}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-missing}"
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
SMOKE_RECEIPT="$SMOKE_RAW"
SHARD_ROOT="${ATTEMPT_RAW:+$ATTEMPT_RAW/shards/shard-$SHARD_INDEX}"
STATUS_RECEIPT="${ATTEMPT_RAW:+$ATTEMPT_RAW/status/pilot-shard-$SHARD_INDEX.json}"

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" state=failed elapsed fallback_python marker_sha marker_size
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  elapsed="$((SECONDS - START_SECONDS))"
  fallback_python="$PYTHON"
  [[ -x "$fallback_python" ]] || fallback_python=/usr/bin/python3
  marker_sha=""; marker_size=0
  if [[ -n "$SHARD_ROOT" && -f "$SHARD_ROOT/generation_complete.json" ]]; then
    marker_sha="$(sha256sum -- "$SHARD_ROOT/generation_complete.json" | awk '{print $1}')"
    marker_size="$(stat -c %s -- "$SHARD_ROOT/generation_complete.json")"
  fi
  if [[ -n "$STATUS_RECEIPT" && "$STATUS_RECEIPT" == /* && -x "$fallback_python" ]]; then
    mkdir -p -- "$(dirname -- "$STATUS_RECEIPT")" 2>/dev/null || true
    "$fallback_python" - "$STATUS_RECEIPT" "$state" "$rc" "$JOB_ID" "$ARRAY_JOB_ID" \
      "$SHARD_INDEX" "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
      "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONTRACT_SHA256" "$RESULT" "$ERROR_TEXT" \
      "$elapsed" "$SHARD_ROOT" "$marker_sha" "$marker_size" "$CONTRACT_RECEIPT" \
      "$SMOKE_RECEIPT" "$EXPECTED_SMOKE_SHA256" <<'PY' || true
import json, os, pathlib, sys
(
    target_raw, status, exit_code, job_id, array_job_id, shard_index, freeze_b,
    implementation_a, config_sha, manifest_sha, contract_sha, result, error,
    elapsed, shard_root, marker_sha, marker_size, contract_receipt,
    smoke_receipt, smoke_sha,
) = sys.argv[1:]
target = pathlib.Path(target_raw)
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_pilot_receipt_v1",
    "status": status,
    "exit_code": int(exit_code),
    "result": result,
    "error": error or None,
    "job_id": job_id,
    "array_job_id": array_job_id,
    "shard_index": int(shard_index) if shard_index.isdigit() else shard_index,
    "elapsed_seconds": int(elapsed),
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest_sha256": manifest_sha,
    "contract_receipt_sha256": contract_sha,
    "contract_receipt": contract_receipt,
    "smoke_receipt": smoke_receipt,
    "smoke_receipt_sha256": smoke_sha,
    "smoke_gate_reverified": True,
    "resources": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "gpus": 1},
    "shard_root": shard_root,
    "generation_complete": {"sha256": marker_sha or None, "size_bytes": int(marker_size)},
}
if target.exists():
    raise SystemExit("append-only pilot receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  fi
  if [[ -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0804 GT-direct shard $SHARD_INDEX" \
    QQ_BODY="status=$state; result=$RESULT; error=${ERROR_TEXT:-none}; job=$JOB_ID; index=$SHARD_INDEX; total=4; elapsed=${elapsed}s; output=$SHARD_ROOT" \
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
SMOKE_RECEIPT="$(realpath -e -- "${SMOKE_RAW:?missing smoke receipt}")"
SHARD_ROOT="$ATTEMPT/shards/shard-$SHARD_INDEX"
STATUS_RECEIPT="$ATTEMPT/status/pilot-shard-$SHARD_INDEX.json"

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
  [[ "$(id -un)" == guisongxia && "${SLURM_JOB_USER:-}" == guisongxia ]] \
    || { ERROR_TEXT=wrong_unix_or_slurm_user; return 1; }
  [[ "$(hostname -s)" == gpu48 && "${SLURM_JOB_NODELIST:-}" == gpu48 \
    && "${SLURM_JOB_PARTITION:-}" == gre ]] \
    || { ERROR_TEXT=wrong_node_or_partition; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$FREEZE_B" \
    && "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_A" ]] \
    || { ERROR_TEXT=freeze_parent_binding_failed; return 1; }
  observed="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r --no-renames \
    "$IMPLEMENTATION_A" "$FREEZE_B" | LC_ALL=C sort)"
  [[ "$observed" == "configs/qwen3vl_gt_direct_teacher_0804.json"$'\n'"doc/training_pipeline/0804_fifith_try/0804.md" ]] \
    || { ERROR_TEXT=freeze_delta_not_exact_config_and_0804_doc; return 1; }
  ! git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1 \
    || { ERROR_TEXT=run_worktree_not_detached; return 1; }
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
    || { ERROR_TEXT=run_worktree_dirty; return 1; }
  [[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" \
    && "$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_SHA256" \
    && "$(sha256sum -- "$CONTRACT_RECEIPT" | awk '{print $1}')" == "$EXPECTED_CONTRACT_SHA256" \
    && "$(sha256sum -- "$SMOKE_RECEIPT" | awk '{print $1}')" == "$EXPECTED_SMOKE_SHA256" ]] \
    || { ERROR_TEXT=config_manifest_contract_or_smoke_hash_drift; return 1; }
  "$PYTHON" - "$ROOT" "$FREEZE_MANIFEST" "$CONTRACT_RECEIPT" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_MANIFEST_SHA256" \
    "$SMOKE_RECEIPT" "$EXPECTED_SMOKE_SHA256" <<'PY'
import hashlib, json, pathlib, sys
root, manifest_path, receipt_path = map(pathlib.Path, sys.argv[1:4])
freeze_b, implementation_a, config_sha, manifest_sha = sys.argv[4:8]
smoke_path = pathlib.Path(sys.argv[8])
smoke_sha = sys.argv[9]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if manifest.get("freeze_commit") != freeze_b or manifest.get("implementation_commit") != implementation_a: raise SystemExit("manifest binding")
if receipt.get("schema_id") != "qwen3vl_gt_direct_teacher_contract_receipt_v1": raise SystemExit("contract schema")
if receipt.get("status") != "success" or receipt.get("result") != "all_contracts_passed": raise SystemExit("contract failed")
if receipt.get("freeze_commit") != freeze_b or receipt.get("implementation_commit") != implementation_a: raise SystemExit("contract commits")
if receipt.get("config_sha256") != config_sha or receipt.get("freeze_manifest_sha256") != manifest_sha: raise SystemExit("contract hashes")
if receipt.get("preflight", {}).get("passed") is not True: raise SystemExit("preflight did not pass")
for name in ("targeted_runner", "targeted_validator", "full_regression"):
    entry = receipt.get("logs", {}).get(name, {})
    if entry.get("passed") is not True or not isinstance(entry.get("test_count"), int) or entry["test_count"] <= 0: raise SystemExit(f"test evidence missing: {name}")
for spec in manifest.get("files", {}).values():
    data = (root / spec["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != spec["size_bytes"]: raise SystemExit("frozen file drift")
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
expected_smoke = {
    "schema_id": "qwen3vl_gt_direct_teacher_smoke_receipt_v1",
    "status": "success",
    "result": "smoke_complete",
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest_sha256": manifest_sha,
    "contract_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "sample_id": "train_407_r3c2",
    "shard_index": 0,
    "num_shards": 4,
    "teacher_only": True,
    "training_authorized": False,
    "qwen_invoked": False,
    "output_scope": "diagnostics_only",
}
if any(smoke.get(key) != value for key, value in expected_smoke.items()):
    raise SystemExit("smoke gate binding differs")
if hashlib.sha256(smoke_path.read_bytes()).hexdigest() != smoke_sha:
    raise SystemExit("smoke receipt raw hash differs")
PY
}

[[ "$SHARD_INDEX" =~ ^[0-3]$ && "${SLURM_ARRAY_TASK_MIN:-}" == 0 \
  && "${SLURM_ARRAY_TASK_MAX:-}" == 3 && "${SLURM_ARRAY_TASK_COUNT:-}" == 4 ]] \
  || { ERROR_TEXT=invalid_four_shard_array_identity; exit 2; }
[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ && "$JOB_ID" =~ ^[0-9]+$ ]] \
  || { ERROR_TEXT=invalid_slurm_job_identity; exit 2; }
[[ ! -e "$SHARD_ROOT" && ! -e "$STATUS_RECEIPT" ]] \
  || { ERROR_TEXT=append_only_shard_output_exists; exit 2; }
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
RESULT=running_gt_direct_teacher
"$PYTHON" tools/run_qwen3vl_gt_direct_teacher.py --config "$CONFIG" \
  --output "$SHARD_ROOT" --shard-index "$SHARD_INDEX" --num-shards 4 \
  --runtime-commit "$FREEZE_B"
[[ -f "$SHARD_ROOT/generation_complete.json" ]] \
  || { ERROR_TEXT=generation_complete_marker_missing; exit 2; }
verify_binding
RESULT=teacher_shard_complete
