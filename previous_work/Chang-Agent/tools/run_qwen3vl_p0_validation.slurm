#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail
umask 027

EXECUTED_WORKER_COPY="$(realpath -e -- "${BASH_SOURCE[0]}")"
SCRIPT_ROOT="$(cd "$(dirname "$EXECUTED_WORKER_COPY")/.." && pwd -P)"
ROOT="$(realpath -m -- "${REPO_ROOT:-$SCRIPT_ROOT}")"
PINNED_WORKER="$(realpath -m -- "$ROOT/tools/run_qwen3vl_p0_validation.slurm")"
WORKSPACE="$(realpath -m -- "${WORKSPACE_ROOT:-$(cd "$ROOT/.." && pwd -P)}")"
CANONICAL_ARTIFACT_ROOT="$(realpath -m -- "$WORKSPACE/Chang-Agent")"
ARTIFACT_ROOT="$(realpath -m -- "${ARTIFACT_ROOT:-$CANONICAL_ARTIFACT_ROOT}")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
MODE="${1:-missing}"
INITIAL_OUTPUT="$(realpath -m -- "$ARTIFACT_ROOT/outputs/CA_0802(1)-qwen3vl-p0-contract-validation")"
PRIOR_OUTPUT="$(realpath -m -- "$ARTIFACT_ROOT/outputs/CA_0802(1R1)-qwen3vl-p0-contract-validation")"
EXPECTED_OUTPUT="$(realpath -m -- "$ARTIFACT_ROOT/outputs/CA_0802(1R2)-qwen3vl-p0-contract-validation")"
OUTPUT="$(realpath -m -- "${2:-$EXPECTED_OUTPUT}")"
LOG_DIR="$OUTPUT/logs"
EVIDENCE_DIR="$OUTPUT/evidence"
ARCHIVE_RUN_DIR="$ARTIFACT_ROOT/outputs/CA_0801_archive/CA_0801(10)-qwen3vl-v121-post-sft-e2e3-v1/run"
AMENDMENT="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P0_VALIDATION_AMENDMENT_R2_0802.json"
INPUT_MANIFEST="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json"
WORKBOOK="$ROOT/experiments/experiment_registry.xlsx"
R2_AMENDMENT_CANONICAL_SHA256="<hash omitted>"
INPUT_MANIFEST_CANONICAL_SHA256="<hash omitted>"
WORKBOOK_SHA256="<hash omitted>"
SUMMARY_OUTPUT="$EVIDENCE_DIR/v1_post_sft_summary_v2.json"
ARCHIVE_MANIFEST_BEFORE="$EVIDENCE_DIR/v1_archive_sha256_before.json"
ARCHIVE_MANIFEST_AFTER="$EVIDENCE_DIR/v1_archive_sha256_after.json"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
EXPECTED_WORKER_SHA256="${EXPECTED_WORKER_SHA256:-}"
JOB_ID="${SLURM_JOB_ID:-unknown}"
JOB_NAME="${SLURM_JOB_NAME:-unknown}"
SAFE_JOB_ID="${JOB_ID//[![:alnum:]_.-]/_}"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""
ACTUAL_COMMIT="unknown"
ACTUAL_WORKER_SHA256="unknown"
TERMINATION_SIGNAL=""
QQ_TITLE="CA0802 P0 contract validation R2"

case "$MODE" in
  targeted)
    EXPECTED_JOB_NAME="CA0802P0TargetR2"
    RESULT_RECEIPT="$EVIDENCE_DIR/targeted_result.json"
    ;;
  regression)
    EXPECTED_JOB_NAME="CA0802P0RegressR2"
    RESULT_RECEIPT="$EVIDENCE_DIR/regression_result.json"
    ;;
  *)
    EXPECTED_JOB_NAME="invalid"
    RESULT_RECEIPT="$EVIDENCE_DIR/invalid_result.json"
    ;;
esac
STATUS_RECEIPT="$LOG_DIR/status-${SAFE_JOB_ID}.txt"

die() {
  ERROR_TEXT="$*"
  echo "qwen3vl_p0_worker: $*" >&2
  return 1
}

record_error() {
  local rc="$?"
  if [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

write_status_receipt() {
  local rc="$1"
  local elapsed="$2"
  local temporary="${STATUS_RECEIPT}.tmp-${BASHPID}"

  [[ -d "$LOG_DIR" ]] || return 1
  if ! printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\nmode=%s\ncommit=%s\nexpected_commit=%s\nworker_sha256=%s\nexpected_worker_sha256=%s\nresult=%s\nerror=%s\npath=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$JOB_NAME" "$MODE" \
    "$ACTUAL_COMMIT" "$EXPECTED_COMMIT" "$ACTUAL_WORKER_SHA256" \
    "$EXPECTED_WORKER_SHA256" "$KEY_RESULT" "$ERROR_TEXT" "$OUTPUT" \
    >"$temporary"; then
    rm -f -- "$temporary"
    return 1
  fi
  if ! mv -f -- "$temporary" "$STATUS_RECEIPT"; then
    rm -f -- "$temporary"
    return 1
  fi
}

write_result_receipt() {
  local rc="$1"
  local elapsed="$2"

  [[ -x "$PYTHON" && -d "$EVIDENCE_DIR" ]] || return 1
  "$PYTHON" - \
    "$RESULT_RECEIPT" "$MODE" "$STATUS" "$rc" "$elapsed" \
    "$JOB_ID" "$JOB_NAME" "$ACTUAL_COMMIT" "$EXPECTED_COMMIT" \
    "$ACTUAL_WORKER_SHA256" "$EXPECTED_WORKER_SHA256" "$KEY_RESULT" \
    "$ERROR_TEXT" "$OUTPUT" "$INITIAL_OUTPUT" "$PRIOR_OUTPUT" "$SUMMARY_OUTPUT" \
    "$ARCHIVE_MANIFEST_BEFORE" "$ARCHIVE_MANIFEST_AFTER" <<'PY'
import json
import os
import pathlib
import sys

(
    target_text,
    mode,
    status,
    exit_code,
    elapsed_seconds,
    job_id,
    job_name,
    actual_commit,
    expected_commit,
    actual_worker_sha256,
    expected_worker_sha256,
    result,
    error,
    output,
    initial_failed_output,
    retry_of_output,
    *artifact_candidates,
) = sys.argv[1:]
target = pathlib.Path(target_text)
artifacts = []
if mode == "targeted":
    artifacts = [
        str(pathlib.Path(candidate))
        for candidate in artifact_candidates
        if pathlib.Path(candidate).is_file()
    ]
payload = {
    "schema_version": "qwen3vl_p0_validation_result_v1",
    "mode": mode,
    "status": status,
    "exit_code": int(exit_code),
    "elapsed_seconds": int(elapsed_seconds),
    "job_id": job_id,
    "job_name": job_name,
    "actual_commit": actual_commit,
    "expected_commit": expected_commit,
    "actual_worker_sha256": actual_worker_sha256,
    "expected_worker_sha256": expected_worker_sha256,
    "resources": {"gpus": 0, "cpus": 2, "memory": "8G"},
    "result": result,
    "error": error or None,
    "artifact_path": output,
    "retry_of_artifact_path": retry_of_output,
    "failed_attempt_artifact_paths": [initial_failed_output, retry_of_output],
    "evidence_artifacts": artifacts,
}
temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
try:
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY
}

write_terminal_receipts() {
  local rc="$1"
  local elapsed="$2"
  local failed=0
  write_result_receipt "$rc" "$elapsed" || failed=1
  write_status_receipt "$rc" "$elapsed" || failed=1
  return "$failed"
}

handle_termination() {
  local signal="$1"
  local exit_code="$2"
  TERMINATION_SIGNAL="$signal"
  STATUS="cancelled"
  ERROR_TEXT="pipeline received SIG${signal}"
  trap '' TERM INT
  exit "$exit_code"
}

finish() {
  local rc="$1"
  trap '' TERM INT
  trap - ERR EXIT
  set +e
  local elapsed="$((SECONDS - START_SECONDS))"

  if [[ "$MODE" == "targeted" && -f "$ARCHIVE_MANIFEST_BEFORE" ]]; then
    if ! write_archive_manifest "$ARCHIVE_MANIFEST_AFTER" \
      || ! compare_archive_manifests; then
      KEY_RESULT="archive_immutability_check_failed"
      ERROR_TEXT="${ERROR_TEXT:+$ERROR_TEXT; }historical v1 archive changed or could not be rehashed"
      rc=70
    fi
  fi

  if [[ -n "$TERMINATION_SIGNAL" ]]; then
    STATUS="cancelled"
  elif [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  else
    STATUS="failed"
    if [[ -z "$ERROR_TEXT" ]]; then
      ERROR_TEXT="pipeline exited with code $rc; inspect shared logs"
    fi
  fi

  if ! write_terminal_receipts "$rc" "$elapsed"; then
    STATUS="failed"
    KEY_RESULT="terminal_receipt_write_failed"
    ERROR_TEXT="${ERROR_TEXT:+$ERROR_TEXT; }could not atomically write terminal receipts"
    rc=74
    write_terminal_receipts "$rc" "$elapsed" || true
  fi

  local key_result_or_error="${ERROR_TEXT:-$KEY_RESULT}"
  local qq_log="/tmp/qwen3vl_p0_validation_${SAFE_JOB_ID}_${MODE}.qq.log"
  QQ_SUB="$QQ_TITLE" \
  QQ_BODY="job=${JOB_NAME}/${JOB_ID}; status=${STATUS}; result=${KEY_RESULT}; error=${ERROR_TEXT:-none}; elapsed=${elapsed}s; path=${OUTPUT}; job_name=${JOB_NAME}; job_id=${JOB_ID}; terminal_status=${STATUS}; elapsed_seconds=${elapsed}; key_result_or_error=${key_result_or_error}; artifact_path=${OUTPUT}" \
    "$QQ_NOTIFIER" >"$qq_log" 2>&1 || true
  rm -f -- "$qq_log"
  exit "$rc"
}

trap record_error ERR
trap 'finish "$?"' EXIT
trap 'handle_termination TERM 143' TERM
trap 'handle_termination INT 130' INT

verify_repo_identity() {
  local observed_commit
  local observed_copy_sha256
  local observed_pinned_sha256

  observed_commit="$(git -C "$ROOT" rev-parse HEAD)"
  [[ "$observed_commit" == "$EXPECTED_COMMIT" ]] \
    || die "repository commit drifted: expected $EXPECTED_COMMIT, found $observed_commit"
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "repository has uncommitted changes: $ROOT"
  observed_copy_sha256="$(sha256sum -- "$EXECUTED_WORKER_COPY" | awk '{print $1}')"
  observed_pinned_sha256="$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')"
  [[ "$observed_copy_sha256" == "$EXPECTED_WORKER_SHA256" ]] \
    || die "Slurm spool copy differs from the submitter-pinned worker bytes"
  [[ "$observed_pinned_sha256" == "$EXPECTED_WORKER_SHA256" ]] \
    || die "repository worker differs from the submitter-pinned worker bytes"
  ACTUAL_COMMIT="$observed_commit"
  ACTUAL_WORKER_SHA256="$observed_pinned_sha256"
}

verify_attempt_file() {
  local attempt_root="$1"
  local attempt_id="$2"
  local relative_path="$3"
  local expected_sha256="$4"
  local source="$attempt_root/$relative_path"
  local observed_sha256

  [[ -f "$source" ]] || die "$attempt_id P0 attempt evidence is missing: $source"
  observed_sha256="$(sha256sum -- "$source" | awk '{print $1}')"
  [[ "$observed_sha256" == "$expected_sha256" ]] \
    || die "$attempt_id P0 attempt evidence changed: $relative_path"
}

verify_failed_attempts() {
  [[ -d "$INITIAL_OUTPUT" ]] || die "initial failed P0 attempt directory is missing"
  [[ -d "$PRIOR_OUTPUT" ]] || die "R1 failed P0 attempt directory is missing"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "submission.txt" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "evidence/targeted_result.json" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "evidence/regression_result.json" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "logs/status-52553.txt" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "logs/status-52554.txt" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "logs/slurm-52553.err" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "logs/slurm-52553.out" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "logs/slurm-52554.err" \
    "<hash omitted>"
  verify_attempt_file "$INITIAL_OUTPUT" "initial" \
    "logs/slurm-52554.out" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "submission.txt" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "evidence/targeted_result.json" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "evidence/regression_result.json" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "evidence/v1_archive_sha256_before.json" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "evidence/v1_archive_sha256_after.json" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "logs/status-52562.txt" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "logs/status-52563.txt" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "logs/slurm-52562.err" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "logs/slurm-52562.out" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "logs/slurm-52563.err" \
    "<hash omitted>"
  verify_attempt_file "$PRIOR_OUTPUT" "R1" \
    "logs/slurm-52563.out" \
    "<hash omitted>"
}

verify_canonical_json_file() {
  local source="$1"
  local expected_sha256="$2"
  local label="$3"

  "$PYTHON" - "$source" "$expected_sha256" "$label" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
label = sys.argv[3]
payload = json.loads(source.read_text(encoding="utf-8"))
serialized = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(serialized).hexdigest()
if observed != expected:
    raise SystemExit(f"{label} canonical SHA256 changed: {observed}")
PY
}

ARCHIVE_RELATIVE_FILES=(
  "rollout_complete.json"
  "per_sample_metrics.json"
  "trajectories/test_20_15/episode_summary.json"
  "trajectories/test_20_15/trajectory.json"
  "trajectories/test_1_1/episode_summary.json"
  "trajectories/test_1_1/trajectory.json"
  "trajectories/test_50_8/episode_summary.json"
  "trajectories/test_50_8/trajectory.json"
)

write_archive_manifest() {
  local destination="$1"
  "$PYTHON" - "$ARCHIVE_RUN_DIR" "$destination" \
    "${ARCHIVE_RELATIVE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import sys

archive = pathlib.Path(sys.argv[1]).resolve(strict=True)
destination = pathlib.Path(sys.argv[2])
relative_paths = sys.argv[3:]
if len(relative_paths) != 8 or len(relative_paths) != len(set(relative_paths)):
    raise SystemExit("archive manifest must contain exactly eight unique paths")

records = []
for relative_text in relative_paths:
    relative = pathlib.Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe archive-relative path: {relative_text}")
    source = (archive / relative).resolve(strict=True)
    try:
        source.relative_to(archive)
    except ValueError as error:
        raise SystemExit(f"archive path escapes source root: {relative_text}") from error
    if not source.is_file():
        raise SystemExit(f"archive input is not a file: {relative_text}")
    stat_before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat_after = source.stat()
    identity_before = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    identity_after = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise SystemExit(f"archive input changed while hashing: {relative_text}")
    value = digest.hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"malformed SHA-256 for {relative_text}")
    records.append(
        {"path": relative.as_posix(), "sha256": value, "size_bytes": stat_after.st_size}
    )

payload = {
    "schema_version": "qwen3vl_archive_sha256_manifest_v1",
    "archive_run_dir": str(archive),
    "file_count": len(records),
    "files": records,
}
temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
try:
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY
}

compare_archive_manifests() {
  "$PYTHON" - "$ARCHIVE_MANIFEST_BEFORE" "$ARCHIVE_MANIFEST_AFTER" \
    "${ARCHIVE_RELATIVE_FILES[@]}" <<'PY'
import json
import pathlib
import re
import sys

before_path = pathlib.Path(sys.argv[1])
after_path = pathlib.Path(sys.argv[2])
expected_paths = set(sys.argv[3:])

def load_digest_map(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "qwen3vl_archive_sha256_manifest_v1":
        raise SystemExit(f"unexpected archive manifest schema: {path}")
    records = payload.get("files")
    if not isinstance(records, list) or payload.get("file_count") != len(records):
        raise SystemExit(f"invalid archive manifest records: {path}")
    digest_map = {}
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit(f"non-object archive manifest record: {path}")
        relative = record.get("path")
        digest = record.get("sha256")
        if not isinstance(relative, str) or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise SystemExit(f"invalid archive manifest path/digest: {path}")
        if relative in digest_map:
            raise SystemExit(f"duplicate archive manifest path: {relative}")
        digest_map[relative] = digest
    if set(digest_map) != expected_paths:
        raise SystemExit(
            f"archive manifest path set drifted: {set(digest_map)!r} != {expected_paths!r}"
        )
    return digest_map

before = load_digest_map(before_path)
after = load_digest_map(after_path)
if before != after:
    changed = sorted(path for path in before if before.get(path) != after.get(path))
    raise SystemExit(f"historical archive changed during summarization: {changed}")
print("archive_immutability=passed;files=8")
PY
}

assert_v1_summary() {
  "$PYTHON" - "$SUMMARY_OUTPUT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(f"independent v1 summary assertion failed: {message}")

require(payload.get("summary_schema_version") == "qwen3vl_e2e_summary_v2", "schema")
require(payload.get("model_state") == "post_sft", "model_state")
require(payload.get("evaluation_profile") == "e2e3", "evaluation_profile")
require(payload.get("evaluation_id") == "DEV-CORE-3-V1", "evaluation_id")
require(
    payload.get("sample_order") == ["test_20_15", "test_1_1", "test_50_8"],
    "sample_order",
)
per_sample = payload.get("per_sample")
require(isinstance(per_sample, dict), "per_sample object")
sample = per_sample.get("test_50_8")
require(isinstance(sample, dict), "test_50_8 record")
require(sample.get("delta_counts") == {"tp": 0, "fp": 6518, "fn": 0}, "delta_counts")
require(sample.get("degraded") is True, "degraded")
oracle = sample.get("candidate_oracle")
require(isinstance(oracle, dict), "sample candidate_oracle")
candidates = oracle.get("candidates")
require(isinstance(candidates, list) and len(candidates) == 1, "candidate count")
candidate = candidates[0]
require(isinstance(candidate, dict), "candidate record")
require(candidate.get("classification") == "harmful", "candidate classification")
require(candidate.get("leaf_class") == "false_accept", "candidate leaf_class")
require(candidate.get("false_accept") is True, "candidate false_accept")
require(oracle.get("false_accept_count") == 1, "sample false_accept_count")
totals = payload.get("candidate_oracle")
require(isinstance(totals, dict), "aggregate candidate_oracle")
require(totals.get("false_accept_count") == 1, "aggregate false_accept_count")
print("v1_summary_contract=passed;test_50_8=false_accept")
PY
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "worker must run under Slurm"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be numeric"
[[ "$MODE" == "targeted" || "$MODE" == "regression" ]] \
  || die "mode must be exactly targeted or regression"
[[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]] \
  || die "Slurm job name drifted: expected $EXPECTED_JOB_NAME, found $JOB_NAME"
[[ "${SLURM_CPUS_PER_TASK:-}" == "2" ]] \
  || die "P0 worker requires exactly two allocated CPUs"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || die "P0 CPU validation unexpectedly received CUDA visibility"
[[ -z "${SLURM_JOB_GPUS:-}" ]] \
  || die "P0 CPU validation unexpectedly received SLURM_JOB_GPUS"
[[ -z "${SLURM_STEP_GPUS:-}" ]] \
  || die "P0 CPU validation unexpectedly received SLURM_STEP_GPUS"

[[ -n "${REPO_ROOT:-}" ]] || die "REPO_ROOT was not exported by the submitter"
[[ -d "$ROOT" ]] || die "repository root is missing: $ROOT"
[[ -f "$PINNED_WORKER" ]] || die "pinned repository worker is missing"
[[ "$ARTIFACT_ROOT" == "$CANONICAL_ARTIFACT_ROOT" && -d "$ARTIFACT_ROOT" ]] \
  || die "artifact repository must remain pinned to $CANONICAL_ARTIFACT_ROOT"
[[ -e "$ROOT/outputs" ]] \
  || die "run worktree is missing its ignored outputs symlink"
[[ "$(realpath -e -- "$ROOT/outputs")" == "$(realpath -e -- "$ARTIFACT_ROOT/outputs")" ]] \
  || die "run worktree outputs path does not resolve to the fixed artifact root"
if [[ "$ROOT" != "$ARTIFACT_ROOT" ]]; then
  [[ -L "$ROOT/outputs" ]] \
    || die "separate run worktree must expose outputs through a symlink"
  git -C "$ROOT" check-ignore -q -- outputs \
    || die "run worktree outputs symlink must be ignored by Git"
fi
[[ "$OUTPUT" == "$EXPECTED_OUTPUT" ]] \
  || die "output must remain the amended CA_0802(1R2) reservation"
[[ -d "$OUTPUT" && -d "$LOG_DIR" && -d "$EVIDENCE_DIR" ]] \
  || die "fixed output reservation or its logs/evidence directories are missing"
[[ "$(realpath -e -- "$OUTPUT")" == "$EXPECTED_OUTPUT" ]] \
  || die "fixed output reservation resolves unexpectedly"
[[ "$(realpath -e -- "$LOG_DIR")" == "$EXPECTED_OUTPUT/logs" ]] \
  || die "fixed log directory resolves outside the output reservation"
[[ "$(realpath -e -- "$EVIDENCE_DIR")" == "$EXPECTED_OUTPUT/evidence" ]] \
  || die "fixed evidence directory resolves outside the output reservation"
[[ ! -e "$RESULT_RECEIPT" ]] \
  || die "mode result receipt already exists: $RESULT_RECEIPT"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || die "EXPECTED_COMMIT must be an exact lowercase Git commit"
[[ "$EXPECTED_WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "EXPECTED_WORKER_SHA256 is malformed"
for executable in "$PYTHON" "$QQ_NOTIFIER"; do
  [[ -x "$executable" ]] || die "required executable is missing: $executable"
done
for required_file in \
  "$ROOT/experiments/build_experiment_workbook.py" \
  "$ROOT/tools/summarize_qwen3vl_e2e_run.py" \
  "$ROOT/tests/test_qwen3vl_p0_contracts.py" \
  "$ROOT/tests/test_experiment_registry.py" \
  "$ROOT/tests/test_local_verifier_analysis.py" \
  "$AMENDMENT" \
  "$INPUT_MANIFEST" \
  "$WORKBOOK" \
  "${ARCHIVE_RELATIVE_FILES[@]/#/$ARCHIVE_RUN_DIR/}"; do
  [[ -f "$required_file" ]] || die "required P0 input is missing: $required_file"
done
verify_failed_attempts
verify_canonical_json_file \
  "$AMENDMENT" "$R2_AMENDMENT_CANONICAL_SHA256" "R2 amendment"
verify_canonical_json_file \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_CANONICAL_SHA256" "input manifest"
verify_attempt_file \
  "$ROOT" "frozen workbook" "experiments/experiment_registry.xlsx" "$WORKBOOK_SHA256"
verify_repo_identity

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT"
cd "$ROOT"

case "$MODE" in
  targeted)
    [[ ! -e "$SUMMARY_OUTPUT" ]] \
      || die "target v2 summary already exists: $SUMMARY_OUTPUT"
    [[ ! -e "$ARCHIVE_MANIFEST_BEFORE" && ! -e "$ARCHIVE_MANIFEST_AFTER" ]] \
      || die "archive immutability receipts already exist"
    KEY_RESULT="hashing_v1_archive_before_all_targeted_checks"
    if ! write_archive_manifest "$ARCHIVE_MANIFEST_BEFORE"; then
      die "could not create the before-validation archive SHA-256 manifest"
    fi

    TEST_PATTERNS=(
      "test_qwen3vl_p0_contracts.py"
      "test_experiment_registry.py"
      "test_local_verifier_analysis.py"
    )
    FAILED_PATTERNS=()
    for pattern in "${TEST_PATTERNS[@]}"; do
      KEY_RESULT="running_unittest_pattern=$pattern"
      set +e
      "$PYTHON" -m unittest discover -s tests -p "$pattern"
      pattern_rc="$?"
      set -e
      if [[ "$pattern_rc" -ne 0 ]]; then
        FAILED_PATTERNS+=("$pattern")
      fi
    done
    if [[ "${#FAILED_PATTERNS[@]}" -ne 0 ]]; then
      die "failed unittest patterns: ${FAILED_PATTERNS[*]}"
    fi

    KEY_RESULT="checking_p0_shell_syntax"
    if ! bash -n \
      tools/submit_qwen3vl_p0_validation.sh \
      tools/run_qwen3vl_p0_validation.slurm; then
      die "P0 shell syntax check failed"
    fi

    KEY_RESULT="running_experiment_workbook_check"
    if ! "$PYTHON" experiments/build_experiment_workbook.py --check; then
      die "experiment workbook checker failed"
    fi

    KEY_RESULT="summarizing_v1_post_sft_archive"
    set +e
    "$PYTHON" tools/summarize_qwen3vl_e2e_run.py \
      --run-dir "$ARCHIVE_RUN_DIR" \
      --output "$SUMMARY_OUTPUT" \
      --evaluation-profile e2e3 \
      --evaluation-id DEV-CORE-3-V1 \
      --model-state post_sft \
      --job-id 51903 \
      --experiment-arm v12_1-compact-post-sft \
      --protocol-version v12.1 \
      --runtime-commit 3020d8df83b15e6062508784cea63f32638683d5 \
      --harness-commit 3020d8df83b15e6062508784cea63f32638683d5 \
      --expected-samples test_20_15 test_1_1 test_50_8
    summarizer_rc="$?"
    set -e

    KEY_RESULT="hashing_v1_archive_after"
    if ! write_archive_manifest "$ARCHIVE_MANIFEST_AFTER"; then
      die "could not create the after-summary archive SHA-256 manifest"
    fi
    KEY_RESULT="comparing_v1_archive_manifests"
    if ! compare_archive_manifests; then
      die "historical v1 archive immutability check failed"
    fi
    if [[ "$summarizer_rc" -ne 0 ]]; then
      die "v1 archive summarizer exited with code $summarizer_rc"
    fi

    KEY_RESULT="asserting_v1_summary_contract"
    if ! assert_v1_summary; then
      die "independent v1 summary contract assertions failed"
    fi
    verify_repo_identity
    KEY_RESULT="targeted_contracts_passed;archive_unchanged=true;v1_false_accept=true"
    ;;
  regression)
    KEY_RESULT="running_full_unittest_regression"
    if ! "$PYTHON" -m unittest discover -s tests -p 'test_*.py'; then
      die "full unittest regression failed"
    fi
    verify_repo_identity
    KEY_RESULT="full_unittest_regression_passed"
    ;;
esac

exit 0
