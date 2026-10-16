#!/usr/bin/env bash
#SBATCH --job-name=CA0804TeacherContract
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
RECEIPT="${ATTEMPT_RAW:+$ATTEMPT_RAW/contract/receipt.json}"

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" state=failed elapsed fallback_python target
  trap - EXIT ERR TERM INT
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  elapsed="$((SECONDS - START_SECONDS))"
  fallback_python="$PYTHON"
  [[ -x "$fallback_python" ]] || fallback_python=/usr/bin/python3
  target="$RECEIPT"
  if [[ -n "$target" && "$target" == /* && -x "$fallback_python" ]]; then
    mkdir -p -- "$(dirname -- "$target")" 2>/dev/null || true
    "$fallback_python" - "$target" "$state" "$rc" "$JOB_ID" "$FREEZE_B" \
      "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_MANIFEST_SHA256" \
      "$RESULT" "$ERROR_TEXT" "$elapsed" "$ATTEMPT" <<'PY' || true
import hashlib, json, os, pathlib, re, sys
(
    target_raw, status, exit_code, job_id, freeze_b, implementation_a,
    config_sha, manifest_sha, result, error, elapsed, attempt_raw,
) = sys.argv[1:]
target = pathlib.Path(target_raw)
attempt = pathlib.Path(attempt_raw)
contract = attempt / "contract"
logs = {}
for name in ("preflight", "targeted-runner", "targeted-validator", "full-regression"):
    path = contract / f"{name}.log"
    entry = {"path": str(path.relative_to(attempt)), "exists": path.is_file()}
    if path.is_file():
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        counts = re.findall(r"Ran\s+(\d+)\s+tests?", text)
        entry.update({
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "test_count": int(counts[-1]) if counts else None,
            "passed": (
                name == "preflight"
                or (
                    bool(counts)
                    and "FAILED (" not in text
                    and re.search(r"(?m)^OK(?:\s|$)", text) is not None
                )
            ),
        })
    logs[name.replace("-", "_")] = entry
preflight_path = contract / "preflight-receipt.json"
preflight = {"path": str(preflight_path.relative_to(attempt)), "exists": preflight_path.is_file()}
if preflight_path.is_file():
    data = preflight_path.read_bytes()
    preflight.update({"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    try:
        preflight["payload"] = json.loads(data)
    except Exception:
        preflight["payload"] = None
    payload_value = preflight["payload"]
    preflight["passed"] = bool(
        isinstance(payload_value, dict)
        and payload_value.get("schema_id") == "qwen3vl_gt_direct_teacher_preflight_receipt_v1"
        and payload_value.get("status") == "preflight_passed"
        and payload_value.get("teacher_only") is True
        and payload_value.get("training_authorized") is False
        and payload_value.get("config", {}).get("sha256") == config_sha
    )
payload = {
    "schema_id": "qwen3vl_gt_direct_teacher_contract_receipt_v1",
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
    "resources": {"user": "guisongxia", "node": "gpu48", "partition": "gre", "gpus": 0},
    "preflight": preflight,
    "logs": logs,
}
if target.exists():
    raise SystemExit("append-only contract receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
status_target = attempt / "status" / "contract-terminal.json"
if not status_target.exists():
    temporary = status_target.with_name(f".{status_target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, status_target)
PY
  fi
  if [[ -x "$QQ_NOTIFIER" ]]; then
    QQ_SUB="CA0804 GT-direct contract" \
    QQ_BODY="status=$state; result=$RESULT; error=${ERROR_TEXT:-none}; job=$JOB_ID; elapsed=${elapsed}s; attempt=$ATTEMPT" \
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
RECEIPT="$ATTEMPT/contract/receipt.json"

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

verify_freeze() {
  local observed
  [[ "$(id -un)" == guisongxia && "${SLURM_JOB_USER:-}" == guisongxia ]] \
    || { ERROR_TEXT=wrong_unix_or_slurm_user; return 1; }
  [[ "$(hostname -s)" == gpu48 && "${SLURM_JOB_NODELIST:-}" == gpu48 \
    && "${SLURM_JOB_PARTITION:-}" == gre ]] \
    || { ERROR_TEXT=wrong_node_or_partition; return 1; }
  case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) ERROR_TEXT=contract_must_be_zero_gpu; return 1 ;; esac
  [[ "$FREEZE_B" =~ ^[0-9a-f]{40}$ && "$IMPLEMENTATION_A" =~ ^[0-9a-f]{40}$ ]] \
    || { ERROR_TEXT=malformed_freeze_commits; return 1; }
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$FREEZE_B" \
    && "$(git -C "$ROOT" rev-parse HEAD^)" == "$IMPLEMENTATION_A" ]] \
    || { ERROR_TEXT=freeze_parent_binding_failed; return 1; }
  [[ "$(git -C "$ROOT" rev-list --parents -n 1 HEAD | wc -w)" -eq 2 ]] \
    || { ERROR_TEXT=freeze_B_must_have_one_parent; return 1; }
  observed="$(git -C "$ROOT" diff-tree --no-commit-id --name-only -r --no-renames \
    "$IMPLEMENTATION_A" "$FREEZE_B" | LC_ALL=C sort)"
  [[ "$observed" == "configs/qwen3vl_gt_direct_teacher_0804.json"$'\n'"doc/training_pipeline/0804_fifith_try/0804.md" ]] \
    || { ERROR_TEXT=freeze_delta_not_exact_config_and_0804_doc; return 1; }
  ! git -C "$ROOT" symbolic-ref -q HEAD >/dev/null 2>&1 \
    || { ERROR_TEXT=run_worktree_not_detached; return 1; }
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
    || { ERROR_TEXT=run_worktree_dirty; return 1; }
  [[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" \
    && "$(sha256sum -- "$FREEZE_MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_SHA256" ]] \
    || { ERROR_TEXT=config_or_manifest_hash_drift; return 1; }
  "$PYTHON" - "$ROOT" "$FREEZE_MANIFEST" "$FREEZE_B" "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" <<'PY'
import hashlib, json, pathlib, sys
root, manifest_path = map(pathlib.Path, sys.argv[1:3])
freeze_b, implementation_a, config_sha = sys.argv[3:6]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_id") != "qwen3vl_gt_direct_teacher_freeze_manifest_v1": raise SystemExit("manifest schema")
if manifest.get("freeze_commit") != freeze_b or manifest.get("implementation_commit") != implementation_a: raise SystemExit("manifest commits")
if manifest.get("config_sha256") != config_sha: raise SystemExit("manifest config")
for spec in manifest.get("files", {}).values():
    path = (root / spec["path"]).resolve()
    path.relative_to(root.resolve())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != spec["size_bytes"]: raise SystemExit("frozen file drift")
PY
}

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=not_a_slurm_job; exit 2; }
[[ -d "$ATTEMPT/contract" && -d "$ATTEMPT/logs" && -d "$ATTEMPT/status" && ! -e "$RECEIPT" ]] \
  || { ERROR_TEXT=append_only_contract_attempt_invalid; exit 2; }
reject_sensitive_environment
verify_freeze
unset OPENAI_API_KEY DASHSCOPE_API_KEY ANTHROPIC_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"

RESULT=preflight
"$PYTHON" tools/validate_qwen3vl_gt_direct_teacher.py --config "$CONFIG" \
  --preflight-only --preflight-receipt "$ATTEMPT/contract/preflight-receipt.json" \
  >"$ATTEMPT/contract/preflight.log" 2>&1
RESULT=targeted_runner_tests
"$PYTHON" -m unittest discover -s tests -p 'test_qwen3vl_gt_direct_teacher.py' -v \
  >"$ATTEMPT/contract/targeted-runner.log" 2>&1
RESULT=targeted_validator_tests
"$PYTHON" -m unittest discover -s tests -p 'test_validate_qwen3vl_gt_direct_teacher.py' -v \
  >"$ATTEMPT/contract/targeted-validator.log" 2>&1
RESULT=full_regression
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v \
  >"$ATTEMPT/contract/full-regression.log" 2>&1
verify_freeze
RESULT=all_contracts_passed
