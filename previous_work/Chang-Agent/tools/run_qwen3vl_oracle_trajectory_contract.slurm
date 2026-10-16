#!/usr/bin/env bash
#SBATCH --job-name=CA0803TrajContract
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -Eeuo pipefail
umask 027
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BOOTSTRAP_ATTEMPT_RAW="${2:-}"
BOOTSTRAP_PYTHON_RAW="${4:-}"
BOOTSTRAP_QQ_RAW="${5:-}"
BOOTSTRAP_JOB_ID="${SLURM_JOB_ID:-missing}"
[[ "$BOOTSTRAP_JOB_ID" =~ ^[0-9]+$ ]] || BOOTSTRAP_JOB_ID=missing
BOOTSTRAP_START_SECONDS="$SECONDS"
BOOTSTRAP_ARMED=1

bootstrap_finish() {
  local rc="$?" state attempt_resolved status_dir target python_candidate
  trap - EXIT ERR TERM INT
  [[ "$BOOTSTRAP_ARMED" == 1 ]] || exit "$rc"
  state=failed
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  attempt_resolved=""
  if [[ -n "$BOOTSTRAP_ATTEMPT_RAW" ]]; then
    attempt_resolved="$(realpath -e -- "$BOOTSTRAP_ATTEMPT_RAW" 2>/dev/null || true)"
  fi
  if [[ -n "$attempt_resolved" && -d "$attempt_resolved" ]]; then
    status_dir="$attempt_resolved/status"
    mkdir -p -- "$status_dir" 2>/dev/null || true
    target="$status_dir/bootstrap-contract-${BOOTSTRAP_JOB_ID}.json"
    python_candidate="$BOOTSTRAP_PYTHON_RAW"
    if [[ ! -x "$python_candidate" ]]; then
      python_candidate=/usr/bin/python3
    fi
    if [[ -x "$python_candidate" && -d "$status_dir" ]]; then
      "$python_candidate" - "$target" "$state" "$rc" "$BOOTSTRAP_JOB_ID" \
        "$((SECONDS - BOOTSTRAP_START_SECONDS))" <<'PY' >/dev/null 2>&1 || true
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[1])
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_bootstrap_failure_v1",
    "worker": "contract",
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "job_id": sys.argv[4],
    "error": "startup_failed_before_contract_receipt_trap",
    "elapsed_seconds": int(sys.argv[5]),
}
if target.exists():
    raise SystemExit("bootstrap receipt already exists")
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
    fi
  fi
  if [[ -x "$BOOTSTRAP_QQ_RAW" ]]; then
    QQ_SUB="CA0803 oracle trajectory contract bootstrap failure" \
    QQ_BODY="status=$state; error=startup_failed_before_contract_receipt_trap; job=$BOOTSTRAP_JOB_ID; elapsed=$((SECONDS - BOOTSTRAP_START_SECONDS))s; attempt=${attempt_resolved:-unresolved}" \
      "$BOOTSTRAP_QQ_RAW" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

trap bootstrap_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ROOT="$(realpath -e -- "${1:?usage: worker ROOT ATTEMPT CONFIG PYTHON QQ FREEZE_B IMPLEMENTATION_A CONFIG_SHA MANIFEST MANIFEST_SHA}")"
ATTEMPT="$(realpath -e -- "${2:?missing ATTEMPT}")"
CONFIG="$(realpath -e -- "${3:?missing CONFIG}")"
PYTHON="$(realpath -e -- "${4:?missing PYTHON}")"
QQ_NOTIFIER="$(realpath -e -- "${5:?missing QQ notifier}")"
FREEZE_B="${6:?missing freeze commit B}"
IMPLEMENTATION_A="${7:?missing implementation commit A}"
EXPECTED_CONFIG_SHA256="${8:?missing config SHA256}"
FREEZE_MANIFEST="$(realpath -e -- "${9:?missing freeze manifest}")"
EXPECTED_MANIFEST_SHA256="${10:?missing freeze manifest SHA256}"
JOB_ID="${SLURM_JOB_ID:-missing}"
START_SECONDS="$SECONDS"
STATUS=failed
RESULT=not_started
ERROR_TEXT=""
CONTRACT="$ATTEMPT/contract"
LOG_DIR="$ATTEMPT/logs"
RECEIPT="$CONTRACT/receipt.json"
PREFLIGHT_RECEIPT="$CONTRACT/preflight-receipt.json"

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

verify_freeze() {
  local observed_delta
  [[ "$(id -un)" == guisongxia ]] || { ERROR_TEXT=wrong_unix_user; return 1; }
  [[ "${SLURM_JOB_USER:-}" == guisongxia ]] || { ERROR_TEXT=wrong_slurm_user; return 1; }
  [[ "$(hostname -s)" == gpu48 && "${SLURM_JOB_NODELIST:-}" == gpu48 ]] \
    || { ERROR_TEXT=wrong_slurm_node; return 1; }
  [[ "${SLURM_JOB_PARTITION:-}" == gre ]] || { ERROR_TEXT=wrong_slurm_partition; return 1; }
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|NoDevFiles) ;;
    *) ERROR_TEXT=unexpected_gpu_in_contract; return 1 ;;
  esac
  [[ "$FREEZE_B" =~ ^[0-9a-f]{40}$ && "$IMPLEMENTATION_A" =~ ^[0-9a-f]{40}$ ]] \
    || { ERROR_TEXT=malformed_freeze_commits; return 1; }
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
  "$PYTHON" - "$ROOT" "$CONFIG" "$FREEZE_MANIFEST" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" <<'PY'
import hashlib, json, pathlib, sys
root, config, manifest_path = map(pathlib.Path, sys.argv[1:4])
freeze_b, implementation_a, config_sha = sys.argv[4:7]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config_payload = json.loads(config.read_text(encoding="utf-8"))
anchors = config_payload.get("anchors", {})
if config_payload.get("status") != "frozen_pending_exact_commit_validation":
    raise SystemExit("frozen config status mismatch")
if anchors.get("pilot_implementation_commit") != implementation_a or anchors.get("pilot_freeze_commit") is not None:
    raise SystemExit("frozen config A/B anchor mismatch")
if manifest.get("schema_id") != "qwen3vl_oracle_trajectory_freeze_manifest_v1":
    raise SystemExit("freeze manifest schema mismatch")
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
if hashlib.sha256(config.read_bytes()).hexdigest() != config_sha:
    raise SystemExit("config raw hash mismatch")
for name, spec in manifest.get("files", {}).items():
    rel = pathlib.Path(spec.get("path", ""))
    if rel.is_absolute() or not rel.parts:
        raise SystemExit(f"invalid frozen path: {name}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"frozen path escapes root: {name}")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec.get("sha256"):
        raise SystemExit(f"frozen file mismatch: {name}")
required = {
    "runner", "validator", "seeded_worker", "contract_worker", "pilot_worker",
    "validation_worker", "submitter", "runner_test", "validator_test",
}
if set(manifest.get("files", {})) != required:
    raise SystemExit("freeze manifest file set mismatch")
PY
}

record_error() {
  local rc="$?"
  [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  return "$rc"
}

finish() {
  local rc="$?" elapsed state
  trap - EXIT ERR TERM INT
  elapsed="$((SECONDS - START_SECONDS))"
  state=failed
  [[ "$rc" -eq 0 ]] && state=success
  [[ "$rc" -eq 130 || "$rc" -eq 143 ]] && state=cancelled
  STATUS="$state"
  "$PYTHON" - "$RECEIPT" "$STATUS" "$rc" "$JOB_ID" "$FREEZE_B" \
    "$IMPLEMENTATION_A" "$EXPECTED_CONFIG_SHA256" "$EXPECTED_MANIFEST_SHA256" \
    "$FREEZE_MANIFEST" "$RESULT" "$ERROR_TEXT" "$elapsed" "$CONTRACT" <<'PY' || true
import hashlib, json, os, pathlib, re, sys
(
    receipt_raw, status, exit_code, job_id, freeze_b, implementation_a,
    config_sha, manifest_sha, manifest_raw, result, error, elapsed, contract_raw,
) = sys.argv[1:]
receipt = pathlib.Path(receipt_raw)
contract = pathlib.Path(contract_raw)
tests = {}
for name, filename in (
    ("targeted_runner", "targeted-runner.log"),
    ("targeted_validator", "targeted-validator.log"),
    ("full_regression", "full-regression.log"),
):
    path = contract / filename
    entry = {"path": str(path.relative_to(contract.parent)), "exists": path.is_file()}
    if path.is_file():
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        matches = re.findall(r"Ran\s+(\d+)\s+tests?", text)
        entry.update({
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "test_count": int(matches[-1]) if matches else 0,
            "failures": 0 if "FAILED (" not in text else None,
            "errors": 0 if "FAILED (" not in text else None,
            "passed": bool(matches) and "FAILED (" not in text and re.search(r"(?m)^OK(?:\s|$)", text) is not None,
        })
    tests[name] = entry
manifest_path = pathlib.Path(manifest_raw)
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    manifest = {}
payload = {
    "schema_id": "qwen3vl_oracle_trajectory_contract_receipt_v3",
    "status": status,
    "exit_code": int(exit_code),
    "job_id": job_id,
    "slurm": {"user": "guisongxia", "node": "gpu48", "partition": "gre"},
    "freeze_commit": freeze_b,
    "implementation_commit": implementation_a,
    "config_sha256": config_sha,
    "freeze_manifest": {
        "path": str(manifest_path.relative_to(contract.parent)),
        "sha256": manifest_sha,
    },
    "diagnostics_contract_attempt": str(contract.parent),
    "frozen_files": manifest.get("files", {}),
    "result": result,
    "error": error or None,
    "elapsed_seconds": int(elapsed),
    "resources": {"gpus": 0, "cpus": 2, "memory": "16G"},
    "tests": tests,
}
preflight_path = contract / "preflight-receipt.json"
preflight_log = contract / "preflight.log"
preflight = {
    "path": str(preflight_path.relative_to(contract.parent)),
    "exists": preflight_path.is_file(),
    "passed": False,
}
if preflight_path.is_file():
    preflight_bytes = preflight_path.read_bytes()
    preflight.update({
        "sha256": hashlib.sha256(preflight_bytes).hexdigest(),
        "size_bytes": len(preflight_bytes),
    })
    try:
        preflight_payload = json.loads(preflight_bytes)
    except Exception:
        preflight_payload = {}
    preflight.update({
        "schema_id": preflight_payload.get("schema_id"),
        "status": preflight_payload.get("status"),
        "config_sha256": preflight_payload.get("config", {}).get("sha256"),
    })
    preflight["passed"] = (
        preflight["schema_id"]
        == "qwen3vl_oracle_trajectory_preflight_receipt_v1"
        and preflight["status"] == "passed"
        and preflight["config_sha256"] == config_sha
    )
if preflight_log.is_file():
    preflight_log_bytes = preflight_log.read_bytes()
    preflight["log"] = {
        "path": str(preflight_log.relative_to(contract.parent)),
        "sha256": hashlib.sha256(preflight_log_bytes).hexdigest(),
        "size_bytes": len(preflight_log_bytes),
    }
payload["preflight"] = preflight
receipt.parent.mkdir(parents=True, exist_ok=True)
if receipt.exists():
    raise SystemExit("contract receipt already exists")
temporary = receipt.with_name(f".{receipt.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, receipt)
PY
  QQ_SUB="CA0803 oracle trajectory contract" \
  QQ_BODY="status=$STATUS; result=$RESULT; error=${ERROR_TEXT:-none}; job=$JOB_ID; elapsed=${elapsed}s; output=$ATTEMPT" \
    "$QQ_NOTIFIER" >/dev/null 2>&1 || true
  exit "$rc"
}

trap record_error ERR
trap finish EXIT
trap 'ERROR_TEXT=received_SIGTERM; exit 143' TERM
trap 'ERROR_TEXT=received_SIGINT; exit 130' INT
BOOTSTRAP_ARMED=0

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { ERROR_TEXT=not_a_slurm_job; exit 2; }
[[ -d "$CONTRACT" && -d "$LOG_DIR" && ! -e "$RECEIPT" ]] \
  || { ERROR_TEXT=contract_output_not_reserved_or_not_empty; exit 2; }
reject_sensitive_environment
verify_freeze

unset OPENAI_API_KEY DASHSCOPE_API_KEY ANTHROPIC_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 PYTHONPATH="$ROOT"
cd "$ROOT"

RESULT=runtime_input_preflight
"$PYTHON" tools/validate_qwen3vl_oracle_trajectory.py \
  --config "$CONFIG" \
  --preflight-only \
  --preflight-receipt "$PREFLIGHT_RECEIPT" \
  >"$CONTRACT/preflight.log" 2>&1
"$PYTHON" - "$PREFLIGHT_RECEIPT" "$EXPECTED_CONFIG_SHA256" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected_config_sha256 = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema_id") != "qwen3vl_oracle_trajectory_preflight_receipt_v1":
    raise SystemExit("preflight receipt schema mismatch")
if payload.get("status") != "passed":
    raise SystemExit("preflight did not pass")
if payload.get("config", {}).get("sha256") != expected_config_sha256:
    raise SystemExit("preflight config binding mismatch")
if payload.get("dataset", {}).get("parent_file_count") != 18:
    raise SystemExit("preflight parent file coverage mismatch")
if payload.get("dataset", {}).get("curation_artifact_count") != 18:
    raise SystemExit("preflight curation artifact coverage mismatch")
if len(payload.get("external_source_repositories", [])) != 2:
    raise SystemExit("preflight external source coverage mismatch")
PY

RESULT=targeted_runner_contracts
"$PYTHON" -m unittest discover -s tests -p 'test_qwen3vl_oracle_trajectory.py' -v \
  >"$CONTRACT/targeted-runner.log" 2>&1
RESULT=targeted_validator_contracts
"$PYTHON" -m unittest discover -s tests -p 'test_validate_qwen3vl_oracle_trajectory.py' -v \
  >"$CONTRACT/targeted-validator.log" 2>&1
RESULT=full_regression
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v \
  >"$CONTRACT/full-regression.log" 2>&1

"$PYTHON" - "$CONTRACT/targeted-runner.log" "$CONTRACT/targeted-validator.log" \
  "$CONTRACT/full-regression.log" <<'PY'
import pathlib, re, sys
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Ran\s+(\d+)\s+tests?", text)
    if not matches or int(matches[-1]) <= 0:
        raise SystemExit(f"zero tests discovered: {path.name}")
    if "FAILED (" in text or re.search(r"(?m)^OK(?:\s|$)", text) is None:
        raise SystemExit(f"test failures or incomplete summary: {path.name}")
PY

verify_freeze
RESULT=all_contracts_passed
