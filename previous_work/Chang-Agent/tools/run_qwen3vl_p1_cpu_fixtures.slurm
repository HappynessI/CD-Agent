#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -euo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

EXECUTED_WORKER_COPY="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}")"
ARTIFACT_ROOT="$(realpath -e -- "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}")"
OUTPUT="$(realpath -e -- "${1:?usage: worker OUTPUT}")"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
LOG_DIR="$(realpath -e -- "$OUTPUT/logs")"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p1_cpu_fixtures.slurm"
CONFIG="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
AMENDMENT="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json"
FIXTURE_MANIFEST="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json"
H0_GOLDEN_RECEIPT="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_H0_V12_1_GOLDEN_FIXTURE_RECEIPT_0802.json"
H0_GOLDEN_VALIDATOR="$ROOT/tools/validate_qwen3vl_h0_golden_fixture.py"
PYTHON="$(realpath -e -- "${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}")"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
EXPECTED_IMPLEMENTATION_COMMIT="${EXPECTED_IMPLEMENTATION_COMMIT:?EXPECTED_IMPLEMENTATION_COMMIT is required}"
EXPECTED_WORKER_SHA256="${EXPECTED_WORKER_SHA256:?EXPECTED_WORKER_SHA256 is required}"
EXPECTED_CONFIG_SHA256="${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}"
EXPECTED_AMENDMENT_CANONICAL_SHA256="${EXPECTED_AMENDMENT_CANONICAL_SHA256:?EXPECTED_AMENDMENT_CANONICAL_SHA256 is required}"
EXPECTED_FIXTURE_MANIFEST_CANONICAL_SHA256="${EXPECTED_FIXTURE_MANIFEST_CANONICAL_SHA256:?EXPECTED_FIXTURE_MANIFEST_CANONICAL_SHA256 is required}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
ALLOCATION_ID="CA_0802(2)"
EXPECTED_JOB_NAME="CA0802P1CPUFixtures"
QQ_TITLE="CA0802 P1 H0-H1 CPU fixtures"
STATUS_RECEIPT="$LOG_DIR/status-${JOB_ID}.txt"
RESULT_RECEIPT="$OUTPUT/cpu_fixture_receipt.json"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
TERMINATION_SIGNAL=""

die() {
  ERROR_TEXT="$*"
  echo "qwen3vl_p1_cpu_fixtures: $*" >&2
  return 1
}

record_error() {
  local rc="$?"
  if [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

write_status() {
  local rc="$1"
  local elapsed="$2"
  local temporary="${STATUS_RECEIPT}.tmp-${BASHPID}"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\nallocation_id=%s\nevaluation_profile=cpu_fixtures\nevaluation_id=P1-H0-H1-CPU-FIXTURES-V1\nsamples=cpu_fixture_cases\narm=h0_h1\nexperiment_arm_id=not_applicable\nharness_protocol_id=not_applicable\ncommit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\npath=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$JOB_NAME" "$ALLOCATION_ID" \
    "$EXPECTED_COMMIT" "$ACTUAL_WORKER_SHA256" "$KEY_RESULT" "$ERROR_TEXT" \
    "$OUTPUT" >"$temporary" || return 1
  mv -f -- "$temporary" "$STATUS_RECEIPT"
}

handle_signal() {
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
  if [[ -n "$TERMINATION_SIGNAL" ]]; then
    STATUS="cancelled"
  elif [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  elif [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="pipeline exited with code $rc; inspect shared logs"
  fi
  if ! write_status "$rc" "$elapsed"; then
    STATUS="failed"
    ERROR_TEXT="${ERROR_TEXT:+$ERROR_TEXT; }could not atomically write terminal status"
    rc=74
    write_status "$rc" "$elapsed" || true
  fi
  local qq_log="/tmp/qwen3vl_p1_cpu_${JOB_ID}.qq.log"
  QQ_SUB="$QQ_TITLE" \
  QQ_BODY="job_name=${JOB_NAME}; job_id=${JOB_ID}; terminal_status=${STATUS}; elapsed_seconds=${elapsed}; key_result_or_error=${ERROR_TEXT:-$KEY_RESULT}; artifact_path=${OUTPUT}" \
    "$QQ_NOTIFIER" >"$qq_log" 2>&1 || true
  rm -f -- "$qq_log"
  exit "$rc"
}

trap record_error ERR
trap 'finish "$?"' EXIT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal INT 130' INT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run this worker only through Slurm"
[[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || die "unexpected Job name: $JOB_NAME"
[[ "${SLURM_JOB_PARTITION:-}" == "gre" ]] || die "partition must be gre"
[[ "${SLURMD_NODENAME:-}" == "gpu48" ]] || die "CPU fixture node must be gpu48"
[[ "${SLURM_CPUS_PER_TASK:-}" == "2" ]] || die "CPU fixture must receive two CPUs"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) die "CPU fixture allocation unexpectedly received a visible GPU" ;;
esac
[[ -x "$PYTHON" ]] || die "required Python is missing: $PYTHON"
[[ -x "$QQ_NOTIFIER" ]] || die "QQ notifier is missing"
for required in "$PINNED_WORKER" "$CONFIG" "$AMENDMENT" "$FIXTURE_MANIFEST" \
  "$H0_GOLDEN_RECEIPT" "$H0_GOLDEN_VALIDATOR" "$OUTPUT/submission.txt"; do
  [[ -f "$required" ]] || die "required CPU fixture input is missing: $required"
done
[[ "$EXPECTED_WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "worker SHA-256 is malformed"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER_COPY" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "executed Slurm worker differs from the pinned worker"
[[ "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "repository worker hash drifted"
[[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] \
  || die "resolved run config byte hash drifted"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit drifted"
[[ "$(git -C "$ROOT" rev-parse "${EXPECTED_COMMIT}^")" == "$EXPECTED_IMPLEMENTATION_COMMIT" ]] \
  || die "freeze commit is not the direct child of the pinned implementation commit"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "run worktree is dirty"

mapfile -t TEST_PATTERNS < <(
  "$PYTHON" - "$CONFIG" "$AMENDMENT" "$FIXTURE_MANIFEST" \
    "$ROOT" "$WORKSPACE" "$ARTIFACT_ROOT" "$OUTPUT_ROOT" "$OUTPUT" "$PYTHON" \
    "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" "$EXPECTED_AMENDMENT_CANONICAL_SHA256" \
    "$EXPECTED_FIXTURE_MANIFEST_CANONICAL_SHA256" "$JOB_ID" "$OUTPUT/submission.txt" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    config_text,
    amendment_text,
    fixture_text,
    root_text,
    workspace_text,
    artifact_text,
    output_root_text,
    output_text,
    python_text,
    expected_commit,
    expected_implementation_commit,
    expected_amendment_hash,
    expected_fixture_hash,
    job_id,
    submission_text,
) = sys.argv[1:]

def load(path_text):
    return json.loads(pathlib.Path(path_text).read_text(encoding="utf-8"))

def canonical(value):
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

config = load(config_text)
amendment = load(amendment_text)
fixture = load(fixture_text)

def resolved(path_text, *, strict=True):
    return pathlib.Path(path_text).resolve(strict=strict)

paths = config.get("paths")
if config.get("configuration_status") != "frozen":
    raise SystemExit("resolved config is not frozen")
if config.get("execution_readiness") != "execution_ready":
    raise SystemExit("resolved config is not execution-ready")
if config.get("p1_execution_authorized") is not True:
    raise SystemExit("resolved config does not authorize P1")
if not isinstance(paths, dict):
    raise SystemExit("resolved config paths are missing")
for key, observed in (
    ("workspace_root", workspace_text),
    ("clean_run_worktree", root_text),
    ("artifact_repository", artifact_text),
    ("output_root", output_root_text),
    ("omniovcd_python", python_text),
):
    declared = paths.get(key)
    if not isinstance(declared, str) or resolved(declared) != resolved(observed):
        raise SystemExit(f"resolved config paths.{key} differs from the CPU worker")
if resolved(sys.executable) != resolved(python_text):
    raise SystemExit("CPU worker PYTHON executable identity drifted")
cpu_allocations = [
    item for item in config.get("allocations", [])
    if isinstance(item, dict) and item.get("allocation_id") == "CA_0802(2)"
]
if len(cpu_allocations) != 1:
    raise SystemExit("resolved config must contain one CPU fixture allocation")
declared_output = cpu_allocations[0].get("output_path")
if not isinstance(declared_output, str) or resolved(declared_output) != resolved(output_text):
    raise SystemExit("CPU worker output differs from the resolved allocation")
if canonical(amendment) != expected_amendment_hash:
    raise SystemExit("P1 amendment canonical hash drifted")
if amendment.get("record_status") != "frozen":
    raise SystemExit("P1 amendment is not frozen")
if amendment.get("execution_readiness") != "execution_ready":
    raise SystemExit("P1 amendment is not execution-ready")
if amendment.get("p1_execution_authorized") is not True:
    raise SystemExit("P1 execution is not authorized")
if amendment.get("implementation_commit") != expected_implementation_commit:
    raise SystemExit("P1 amendment implementation commit binding drifted")
if canonical(fixture) != expected_fixture_hash:
    raise SystemExit("CPU fixture manifest canonical hash drifted")
if fixture.get("schema_id") != "qwen3vl_h0_h1_cpu_fixture_manifest_v1":
    raise SystemExit("CPU fixture manifest schema drifted")
if fixture.get("status") != "frozen":
    raise SystemExit("CPU fixture manifest is not frozen")
expected_patterns = config.get("cpu_fixture_suite", {}).get("targeted_unittest_patterns")
if fixture.get("targeted_unittest_patterns") != expected_patterns:
    raise SystemExit("CPU fixture pattern list differs from resolved config")
for test_id in fixture.get("required_test_ids", []):
    relative, separator, method = str(test_id).partition("::")
    if not separator or not method:
        raise SystemExit(f"malformed required fixture test ID: {test_id}")
    source = pathlib.Path(config_text).resolve().parents[3] / relative
    if not source.is_file() or f"def {method}(" not in source.read_text(encoding="utf-8"):
        raise SystemExit(f"required fixture test ID is absent: {test_id}")
submission = {}
for line in pathlib.Path(submission_text).read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        key, separator, value = line.partition("=")
        if not separator or key in submission:
            raise SystemExit("submission receipt is malformed")
        submission[key] = value
if submission.get("submission_status") != "submitted":
    raise SystemExit("submission receipt is not complete")
if submission.get("job_id") != job_id:
    raise SystemExit("submission receipt Job ID drifted")
if submission.get("expected_commit") != expected_commit:
    raise SystemExit("submission receipt freeze commit drifted")
if submission.get("implementation_commit") != expected_implementation_commit:
    raise SystemExit("submission receipt implementation commit drifted")
for pattern in expected_patterns:
    print(pattern)
PY
)
[[ "${#TEST_PATTERNS[@]}" -gt 0 ]] || die "CPU fixture pattern list is empty"

GOLDEN_VALIDATION_LOG="$LOG_DIR/h0-golden-fixture-validation.log"
[[ ! -e "$GOLDEN_VALIDATION_LOG" ]] \
  || die "H0 golden validation log already exists"
KEY_RESULT="validating_independent_h0_golden_receipt"
"$PYTHON" "$H0_GOLDEN_VALIDATOR" \
  --receipt "$H0_GOLDEN_RECEIPT" \
  --manifest "$FIXTURE_MANIFEST" \
  --repo-root "$ROOT" \
  >"$GOLDEN_VALIDATION_LOG" 2>&1 \
  || die "independent H0 golden receipt validation failed"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT"
cd "$ROOT"

TEST_LOGS=()
for pattern in "${TEST_PATTERNS[@]}"; do
  [[ "$pattern" =~ ^test_[A-Za-z0-9_]+\.py$ ]] || die "unsafe test pattern: $pattern"
  [[ -f "$ROOT/tests/$pattern" ]] || die "targeted test file is missing: $pattern"
  log="$LOG_DIR/unittest-${pattern%.py}.log"
  [[ ! -e "$log" ]] || die "targeted test log already exists: $log"
  KEY_RESULT="running_targeted_pattern=${pattern}"
  "$PYTHON" -m unittest discover -s tests -p "$pattern" >"$log" 2>&1 \
    || die "targeted unittest pattern failed: $pattern"
  if ! grep -Eq 'Ran [1-9][0-9]* tests?' "$log"; then
    die "targeted unittest pattern discovered zero tests: $pattern"
  fi
  TEST_LOGS+=("$log")
done

KEY_RESULT="checking_p1_shell_syntax"
bash -n \
  tools/run_qwen3vl_p1_cpu_fixtures.slurm \
  tools/run_qwen3vl_p1_evaluation.slurm \
  tools/run_qwen3vl_p1_dependency_guard.slurm \
  tools/submit_qwen3vl_p1.sh \
  || die "P1 shell syntax check failed"

FULL_LOG="$LOG_DIR/unittest-full-regression.log"
[[ ! -e "$FULL_LOG" ]] || die "full regression log already exists"
KEY_RESULT="running_full_unittest_regression"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' >"$FULL_LOG" 2>&1 \
  || die "full unittest regression failed"
if ! grep -Eq 'Ran [1-9][0-9]* tests?' "$FULL_LOG"; then
  die "full unittest regression discovered zero tests"
fi
TEST_LOGS+=("$FULL_LOG")

[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit changed during fixtures"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "worktree changed during fixtures"

KEY_RESULT="targeted_and_full_regression_passed"
"$PYTHON" - "$RESULT_RECEIPT" "$JOB_ID" "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" \
  "$ACTUAL_WORKER_SHA256" "$CONFIG" "$FIXTURE_MANIFEST" "${TEST_LOGS[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import sys

target = pathlib.Path(sys.argv[1])
job_id, freeze_commit, implementation_commit, worker_hash = sys.argv[2:6]
config = pathlib.Path(sys.argv[6]).resolve(strict=True)
fixture = pathlib.Path(sys.argv[7]).resolve(strict=True)
logs = [pathlib.Path(item).resolve(strict=True) for item in sys.argv[8:]]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = []
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Ran ([0-9]+) tests?", text)
    if not matches or int(matches[-1]) <= 0:
        raise SystemExit(f"cannot recover positive unittest count from {path}")
    records.append(
        {"path": str(path), "sha256": sha256(path), "tests_run": int(matches[-1])}
    )
payload = {
    "schema_id": "qwen3vl_h0_h1_cpu_fixture_receipt_v1",
    "status": "complete",
    "job_id": job_id,
    "implementation_commit": implementation_commit,
    "freeze_commit": freeze_commit,
    "worker_sha256": worker_hash,
    "resolved_run_config": {"path": str(config), "sha256": sha256(config)},
    "fixture_manifest": {"path": str(fixture), "sha256": sha256(fixture)},
    "targeted_and_full_regression_passed": True,
    "test_logs": records,
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

exit 0
