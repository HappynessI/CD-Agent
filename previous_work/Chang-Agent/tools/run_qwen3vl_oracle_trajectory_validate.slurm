#!/usr/bin/env bash
#SBATCH --job-name=CA0803TrajValidate
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

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
ARRAY_JOB_ID="${13:-missing}"
JOB_ID="${SLURM_JOB_ID:-missing}"
START_SECONDS="$SECONDS"
STATUS=failed
RESULT=not_started
ERROR_TEXT=""
OUTPUT="$ATTEMPT/validation"
STATUS_RECEIPT="$ATTEMPT/status/validation.json"
BOOTSTRAP_ERROR=argument_resolution_failed
SCIENTIFIC_CONCLUSION=not_available
GLOBAL_RELEASE_PASSED=unknown

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
    target="$ATTEMPT/status/bootstrap-validation-${JOB_ID}.json"
    "$fallback_python" - "$target" "$state" "$rc" "$JOB_ID" "$ARRAY_JOB_ID" \
      "$FREEZE_B" "$IMPLEMENTATION_A" "$BOOTSTRAP_ERROR" <<'PY' || true
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_worker_bootstrap_terminal_v1",
    "worker": "validation",
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "job_id": sys.argv[4],
    "upstream_array_job_id": sys.argv[5],
    "freeze_commit": sys.argv[6],
    "implementation_commit": sys.argv[7],
    "error": sys.argv[8],
}
if not target.exists():
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
PY
  fi
  if [[ "$rc" -ne 0 && -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0803 oracle trajectory validation bootstrap" \
    QQ_BODY="status=$state; error=$BOOTSTRAP_ERROR; job=$JOB_ID; upstream=$ARRAY_JOB_ID; attempt=$ATTEMPT" \
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
  && "$EXPECTED_CONTRACT_SHA256" != missing && "$ARRAY_JOB_ID" != missing ]] || false
FREEZE_MANIFEST="$(realpath -e -- "${FREEZE_MANIFEST:?missing freeze manifest}")"
CONTRACT_RECEIPT="$(realpath -e -- "${CONTRACT_RECEIPT:?missing contract receipt}")"
OUTPUT="$ATTEMPT/validation"
STATUS_RECEIPT="$ATTEMPT/status/validation.json"

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
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|NoDevFiles) ;;
    *) ERROR_TEXT=unexpected_gpu_in_validation; return 1 ;;
  esac
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
  "$PYTHON" - "$ROOT" "$FREEZE_MANIFEST" "$CONTRACT_RECEIPT" "$ATTEMPT" \
    "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_MANIFEST_SHA256" <<'PY'
import hashlib, json, pathlib, sys
root, manifest_path, receipt_path, formal_attempt = map(pathlib.Path, sys.argv[1:5])
freeze_b, implementation_a, config_sha, manifest_sha = sys.argv[5:9]
root, formal_attempt = root.resolve(), formal_attempt.resolve()
receipt_path = receipt_path.resolve()
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
        path.relative_to(root)
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
if receipt.get("config_sha256") != config_sha or receipt.get("freeze_manifest", {}).get("sha256") != manifest_sha:
    raise SystemExit("contract config/manifest binding mismatch")
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
for name in ("targeted_runner", "targeted_validator", "full_regression"):
    entry = tests.get(name, {})
    if entry.get("passed") is not True or entry.get("test_count", 0) <= 0:
        raise SystemExit(f"contract tests did not pass: {name}")
    if entry.get("failures") != 0 or entry.get("errors") != 0:
        raise SystemExit(f"contract test count mismatch: {name}")
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
  local rc="$?" elapsed state report_hash report_size matching_hash matching_size
  trap - EXIT ERR TERM INT
  elapsed="$((SECONDS - START_SECONDS))"
  state=failed
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  STATUS="$state"
  report_hash=""; report_size=0; matching_hash=""; matching_size=0
  if [[ -f "$OUTPUT/validation_report.json" ]]; then
    report_hash="$(sha256sum -- "$OUTPUT/validation_report.json" | awk '{print $1}')"
    report_size="$(stat -c %s -- "$OUTPUT/validation_report.json")"
  fi
  if [[ -f "$OUTPUT/data/matching_manifest.json" ]]; then
    matching_hash="$(sha256sum -- "$OUTPUT/data/matching_manifest.json" | awk '{print $1}')"
    matching_size="$(stat -c %s -- "$OUTPUT/data/matching_manifest.json")"
  fi
  "$PYTHON" - "$STATUS_RECEIPT" "$STATUS" "$rc" "$JOB_ID" "$ARRAY_JOB_ID" \
    "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_CONTRACT_SHA256" "$RESULT" "$ERROR_TEXT" \
    "$elapsed" "$OUTPUT" "$report_hash" "$report_size" "$matching_hash" "$matching_size" \
    "$CONTRACT_RECEIPT" "$SCIENTIFIC_CONCLUSION" "$GLOBAL_RELEASE_PASSED" <<'PY' || true
import json, os, pathlib, sys
(
    output_raw, status, exit_code, job_id, array_job_id, freeze_b,
    implementation_a, config_sha, manifest_sha, contract_sha, result, error,
    elapsed, validation_root, report_sha, report_size, matching_sha, matching_size,
    contract_receipt, scientific_conclusion, global_release_passed,
) = sys.argv[1:]
contract_receipt_path = pathlib.Path(contract_receipt).resolve()
output = pathlib.Path(output_raw)
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_validation_worker_receipt_v1",
    "status": status,
    "exit_code": int(exit_code),
    "job_id": job_id,
    "upstream_array_job_id": array_job_id,
    "dependency_policy": "afterany_then_fail_closed_on_every_shard_receipt",
    "slurm": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "gpus": 0},
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
    "scientific_conclusion": (
        None if scientific_conclusion == "not_available" else scientific_conclusion
    ),
    "global_release_passed": (
        True if global_release_passed == "true"
        else False if global_release_passed == "false"
        else None
    ),
    "error": error or None,
    "elapsed_seconds": int(elapsed),
    "validation_root": validation_root,
    "validation_report": {"sha256": report_sha or None, "size_bytes": int(report_size)},
    "matching_manifest": {"sha256": matching_sha or None, "size_bytes": int(matching_size)},
}
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise SystemExit("validation worker receipt already exists")
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
  QQ_SUB="CA0803 oracle trajectory validation" \
  QQ_BODY="status=$STATUS; result=$RESULT; scientific_conclusion=$SCIENTIFIC_CONCLUSION; global_release_passed=$GLOBAL_RELEASE_PASSED; error=${ERROR_TEXT:-none}; job=$JOB_ID; upstream=$ARRAY_JOB_ID; elapsed=${elapsed}s; output=$OUTPUT" \
    "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  exit "$rc"
}

trap record_error ERR
trap finish EXIT
trap 'ERROR_TEXT=received_SIGTERM; exit 143' TERM
trap 'ERROR_TEXT=received_SIGINT; exit 130' INT

[[ "$JOB_ID" =~ ^[0-9]+$ && "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]] \
  || { ERROR_TEXT=invalid_slurm_job_identity; exit 2; }
[[ ! -e "$OUTPUT" && ! -e "$STATUS_RECEIPT" ]] \
  || { ERROR_TEXT=append_only_validation_output_exists; exit 2; }
reject_sensitive_environment
verify_binding

RESULT=checking_afterany_upstream_shards
for shard in 0 1 2; do
  receipt="$ATTEMPT/status/pilot-shard-$shard.json"
  marker="$ATTEMPT/shards/shard-$shard/generation_complete.json"
  index="$ATTEMPT/shards/shard-$shard/shard_index.json"
  if ! "$PYTHON" - "$receipt" "$marker" "$index" "$shard" "$ARRAY_JOB_ID" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_MANIFEST_SHA256" \
    "$EXPECTED_CONTRACT_SHA256" "$CONTRACT_RECEIPT" <<'PY'
import hashlib, json, pathlib, sys
receipt_path, marker_path, index_path = map(pathlib.Path, sys.argv[1:4])
shard, array_job_id, freeze_b, implementation_a, config_sha, manifest_sha, contract_sha, contract_receipt = sys.argv[4:12]
contract_receipt_path = pathlib.Path(contract_receipt).resolve()
if not receipt_path.is_file() or not marker_path.is_file() or not index_path.is_file():
    raise SystemExit("upstream receipt, marker, or shard index missing")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected = {
    "status": "success",
    "result": "shard_complete",
    "array_job_id": array_job_id,
    "shard_index": int(shard),
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest_sha256": manifest_sha,
    "contract_receipt_sha256": contract_sha,
    "contract_receipt_schema": "qwen3vl_oracle_trajectory_contract_receipt_v3",
    "contract_preflight_receipt_and_log_reverified": True,
    "diagnostics_contract_attempt": str(contract_receipt_path.parent.parent),
    "diagnostics_contract_receipt": {"path": str(contract_receipt_path), "sha256": contract_sha},
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"upstream receipt differs: {key}")
data = marker_path.read_bytes()
artifact = receipt.get("generation_complete", {})
if hashlib.sha256(data).hexdigest() != artifact.get("sha256") or len(data) != artifact.get("size_bytes"):
    raise SystemExit("upstream generation marker differs")
marker = json.loads(data)
index = json.loads(index_path.read_text(encoding="utf-8"))
if marker.get("runtime_commit") != freeze_b:
    raise SystemExit("upstream generation marker runtime commit differs from exact freeze B")
if index.get("runtime_commit") != freeze_b:
    raise SystemExit("upstream shard index runtime commit differs from exact freeze B")
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

RESULT=independent_recompute_and_matched_derivation
"$PYTHON" "$ROOT/tools/validate_qwen3vl_oracle_trajectory.py" \
  --config "$CONFIG" \
  --input-root "$ATTEMPT/shards/shard-0" \
  --input-root "$ATTEMPT/shards/shard-1" \
  --input-root "$ATTEMPT/shards/shard-2" \
  --output "$OUTPUT"
[[ -f "$OUTPUT/validation_report.json" && -f "$OUTPUT/data/matching_manifest.json" ]] \
  || { ERROR_TEXT=validation_required_artifacts_missing; exit 2; }
verify_binding
RESULT=reading_validation_scientific_conclusion
if ! SCIENTIFIC_RESULT="$("$PYTHON" - "$OUTPUT/validation_report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
conclusion = report.get("scientific_conclusion")
passed = report.get("global_release_gate", {}).get("passed")
if conclusion not in {"global_release_passed", "scientific_no_go"}:
    raise SystemExit("validation report scientific conclusion is invalid")
if type(passed) is not bool or passed != (conclusion == "global_release_passed"):
    raise SystemExit("validation report global release result is inconsistent")
print(f"{conclusion}|{'true' if passed else 'false'}")
PY
)"; then
  ERROR_TEXT=validation_scientific_conclusion_invalid
  exit 2
fi
IFS='|' read -r SCIENTIFIC_CONCLUSION GLOBAL_RELEASE_PASSED <<<"$SCIENTIFIC_RESULT"
[[ "$SCIENTIFIC_CONCLUSION" == global_release_passed || "$SCIENTIFIC_CONCLUSION" == scientific_no_go ]] \
  || { ERROR_TEXT=validation_scientific_conclusion_parse_failed; exit 2; }
[[ "$GLOBAL_RELEASE_PASSED" == true || "$GLOBAL_RELEASE_PASSED" == false ]] \
  || { ERROR_TEXT=validation_global_release_parse_failed; exit 2; }
RESULT="validation_and_derivation_complete;scientific_conclusion=$SCIENTIFIC_CONCLUSION;global_release_passed=$GLOBAL_RELEASE_PASSED"
