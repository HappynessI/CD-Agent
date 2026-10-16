#!/usr/bin/env bash
#SBATCH --job-name=CA0804TeacherValidate
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00

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
ARRAY_JOB_ID="${15:-missing}"
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
OUTPUT="${ATTEMPT_RAW:+$ATTEMPT_RAW/validation}"
STATUS_RECEIPT="${ATTEMPT_RAW:+$ATTEMPT_RAW/status/validation.json}"

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" state=failed elapsed fallback_python report_sha report_size
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  elapsed="$((SECONDS - START_SECONDS))"
  fallback_python="$PYTHON"
  [[ -x "$fallback_python" ]] || fallback_python=/usr/bin/python3
  report_sha=""; report_size=0
  if [[ -n "$OUTPUT" && -f "$OUTPUT/validation_report.json" ]]; then
    report_sha="$(sha256sum -- "$OUTPUT/validation_report.json" | awk '{print $1}')"
    report_size="$(stat -c %s -- "$OUTPUT/validation_report.json")"
  fi
  if [[ -n "$STATUS_RECEIPT" && "$STATUS_RECEIPT" == /* && -x "$fallback_python" ]]; then
    mkdir -p -- "$(dirname -- "$STATUS_RECEIPT")" 2>/dev/null || true
    "$fallback_python" - "$STATUS_RECEIPT" "$state" "$rc" "$JOB_ID" "$ARRAY_JOB_ID" \
      "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
      "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONTRACT_SHA256" "$RESULT" "$ERROR_TEXT" \
      "$elapsed" "$OUTPUT" "$report_sha" "$report_size" "$CONTRACT_RECEIPT" \
      "$SMOKE_RECEIPT" "$EXPECTED_SMOKE_SHA256" <<'PY' || true
import json, os, pathlib, sys
(
    target_raw, status, exit_code, job_id, array_job_id, freeze_b, implementation_a,
    config_sha, manifest_sha, contract_sha, result, error, elapsed, output_raw,
    report_sha, report_size, contract_receipt, smoke_receipt, smoke_sha,
) = sys.argv[1:]
target = pathlib.Path(target_raw)
output = pathlib.Path(output_raw)
conclusion = None
if (output / "validation_report.json").is_file():
    try:
        conclusion = json.loads((output / "validation_report.json").read_text(encoding="utf-8")).get("scientific_conclusion")
    except Exception:
        pass
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_validation_receipt_v1",
    "status": status,
    "exit_code": int(exit_code),
    "result": result,
    "error": error or None,
    "job_id": job_id,
    "upstream_array_job_id": array_job_id,
    "dependency_policy": "afterany_then_fail_closed_on_four_receipts",
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
    "resources": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "gpus": 0},
    "validation_root": output_raw,
    "validation_report": {"sha256": report_sha or None, "size_bytes": int(report_size)},
    "scientific_conclusion": conclusion,
    "training_authorized": False,
    "teacher_only": True,
}
if target.exists():
    raise SystemExit("append-only validation receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  fi
  if [[ -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0804 GT-direct validation" \
    QQ_BODY="status=$state; result=$RESULT; error=${ERROR_TEXT:-none}; job=$JOB_ID; upstream=$ARRAY_JOB_ID; elapsed=${elapsed}s; output=$OUTPUT" \
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
OUTPUT="$ATTEMPT/validation"
STATUS_RECEIPT="$ATTEMPT/status/validation.json"

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
  case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT=validator_must_be_zero_gpu; return 1 ;; esac
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
  "$PYTHON" - "$FREEZE_MANIFEST" "$CONTRACT_RECEIPT" "$SMOKE_RECEIPT" \
    "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONTRACT_SHA256" \
    "$EXPECTED_SMOKE_SHA256" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, contract_path, smoke_path = map(pathlib.Path, sys.argv[1:4])
freeze_b, implementation_a, config_sha, manifest_sha, contract_sha, smoke_sha = sys.argv[4:10]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
if manifest.get("freeze_commit") != freeze_b or manifest.get("implementation_commit") != implementation_a:
    raise SystemExit("manifest binding")
if hashlib.sha256(contract_path.read_bytes()).hexdigest() != contract_sha:
    raise SystemExit("contract raw hash")
if contract.get("schema_id") != "qwen3vl_gt_direct_teacher_contract_receipt_v1":
    raise SystemExit("contract schema")
if contract.get("status") != "success" or contract.get("result") != "all_contracts_passed":
    raise SystemExit("contract failed")
if contract.get("freeze_commit") != freeze_b or contract.get("implementation_commit") != implementation_a:
    raise SystemExit("contract commits")
if contract.get("config_sha256") != config_sha or contract.get("freeze_manifest_sha256") != manifest_sha:
    raise SystemExit("contract hashes")
if contract.get("preflight", {}).get("passed") is not True:
    raise SystemExit("contract preflight")
if hashlib.sha256(smoke_path.read_bytes()).hexdigest() != smoke_sha:
    raise SystemExit("smoke raw hash")
expected_smoke = {
    "schema_id": "qwen3vl_gt_direct_teacher_smoke_receipt_v1",
    "status": "success",
    "result": "smoke_complete",
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest_sha256": manifest_sha,
    "contract_receipt_sha256": contract_sha,
    "sample_id": "train_407_r3c2",
    "shard_index": 0,
    "num_shards": 4,
    "teacher_only": True,
    "training_authorized": False,
    "qwen_invoked": False,
    "output_scope": "diagnostics_only",
    "scientific_result_authoritative": False,
}
if any(smoke.get(key) != value for key, value in expected_smoke.items()):
    raise SystemExit("smoke binding")
PY
}

[[ "$JOB_ID" =~ ^[0-9]+$ && "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]] \
  || { ERROR_TEXT=invalid_slurm_job_identity; exit 2; }
[[ ! -e "$OUTPUT" && ! -e "$STATUS_RECEIPT" ]] \
  || { ERROR_TEXT=append_only_validation_output_exists; exit 2; }
reject_sensitive_environment
verify_binding

RESULT=checking_afterany_upstream_shards
for shard in 0 1 2 3; do
  receipt="$ATTEMPT/status/pilot-shard-$shard.json"
  marker="$ATTEMPT/shards/shard-$shard/generation_complete.json"
  index="$ATTEMPT/shards/shard-$shard/shard_index.json"
  if ! "$PYTHON" - "$receipt" "$marker" "$index" "$shard" "$ARRAY_JOB_ID" \
    "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONTRACT_SHA256" \
    "$SMOKE_RECEIPT" "$EXPECTED_SMOKE_SHA256" <<'PY'
import hashlib, json, pathlib, sys
receipt_path, marker_path, index_path = map(pathlib.Path, sys.argv[1:4])
(
    shard, array_job_id, freeze_b, implementation_a, config_sha, manifest_sha,
    contract_sha, smoke_receipt, smoke_sha,
) = sys.argv[4:13]
if not receipt_path.is_file() or not marker_path.is_file() or not index_path.is_file():
    raise SystemExit("upstream artifact missing")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected = {
    "schema_id": "qwen3vl_gt_direct_teacher_pilot_receipt_v1",
    "status": "success", "result": "teacher_shard_complete",
    "array_job_id": array_job_id, "shard_index": int(shard),
    "freeze_commit": freeze_b, "implementation_commit": implementation_a,
    "config_sha256": config_sha, "freeze_manifest_sha256": manifest_sha,
    "contract_receipt_sha256": contract_sha,
    "smoke_receipt": str(pathlib.Path(smoke_receipt).resolve()),
    "smoke_receipt_sha256": smoke_sha,
    "smoke_gate_reverified": True,
}
for key, value in expected.items():
    if receipt.get(key) != value: raise SystemExit(f"receipt differs: {key}")
data = marker_path.read_bytes()
artifact = receipt.get("generation_complete", {})
if hashlib.sha256(data).hexdigest() != artifact.get("sha256") or len(data) != artifact.get("size_bytes"):
    raise SystemExit("generation marker binding")
marker = json.loads(data)
index = json.loads(index_path.read_text(encoding="utf-8"))
if marker.get("runtime_commit") != freeze_b or index.get("runtime_commit") != freeze_b:
    raise SystemExit("runtime commit binding")
if index.get("shard_index") != int(shard) or index.get("num_shards") != 4:
    raise SystemExit("shard identity binding")
PY
  then
    ERROR_TEXT="upstream_shard_${shard}_failed_missing_or_unbound"
    exit 2
  fi
done

unset OPENAI_API_KEY DASHSCOPE_API_KEY ANTHROPIC_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"
RESULT=independent_teacher_validation
"$PYTHON" tools/validate_qwen3vl_gt_direct_teacher.py --config "$CONFIG" \
  --input-root "$ATTEMPT/shards/shard-0" \
  --input-root "$ATTEMPT/shards/shard-1" \
  --input-root "$ATTEMPT/shards/shard-2" \
  --input-root "$ATTEMPT/shards/shard-3" \
  --output "$OUTPUT"
[[ -f "$OUTPUT/validation_report.json" ]] \
  || { ERROR_TEXT=validation_report_missing; exit 2; }
"$PYTHON" - "$OUTPUT/validation_report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("teacher_only") is not True or report.get("training_authorized") is not False:
    raise SystemExit("teacher-only safety contract missing")
if report.get("scientific_conclusion") not in {
    "tool_upper_bound_supported", "greedy_protocol_inconclusive",
    "proposal_bottleneck_supported", "inconclusive",
}:
    raise SystemExit("invalid scientific conclusion")
PY
verify_binding
RESULT=validation_complete
