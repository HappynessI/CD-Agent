#!/usr/bin/env bash
#SBATCH --job-name=CA31V121Tests
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT must be exported by submit_local_verifier_tests.sh}")"
HARNESS_ROOT="$(realpath -e -- "${HARNESS_ROOT:-$ROOT}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:-$(cd "$ROOT/.." && pwd -P)}")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
SUITE="${1:?usage: run_local_verifier_tests.slurm SUITE OUTPUT}"
OUTPUT="$(realpath -m -- "${2:?usage: run_local_verifier_tests.slurm SUITE OUTPUT}")"
LOG_DIR="$OUTPUT/logs"
OUTPUT_ROOT="$(realpath -e -- "$HARNESS_ROOT/outputs")"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""

finish() {
  local rc="$?"
  trap - ERR EXIT
  set +e
  local elapsed="$((SECONDS - START_SECONDS))"
  if [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  elif [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="test suite exited with code $rc"
  fi
  QQ_SUB="Local verifier ${SUITE} tests ${STATUS}" \
  QQ_BODY="status=${STATUS}; suite=${SUITE}; result=${KEY_RESULT}; error=${ERROR_TEXT:-none}; job=${JOB_ID}; elapsed=${elapsed}s; output=${OUTPUT}" \
    "$QQ_NOTIFIER" >"/tmp/local_verifier_tests_${JOB_ID}.qq.log" 2>&1 || true
  rm -f -- "/tmp/local_verifier_tests_${JOB_ID}.qq.log"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\nsuite=%s\ncommit=%s\nkey_result=%s\nerror=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$SUITE" "${ACTUAL_COMMIT:-unknown}" \
    "$KEY_RESULT" "$ERROR_TEXT" >"$LOG_DIR/status-${SUITE}-${JOB_ID}.txt"
  exit "$rc"
}
trap finish EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  ERROR_TEXT="worker must run under Slurm"
  exit 2
}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ERROR_TEXT="CPU test job unexpectedly received a GPU allocation"
  exit 2
fi
[[ -d "$LOG_DIR" ]] || {
  ERROR_TEXT="shared log directory is missing: $LOG_DIR"
  exit 2
}
[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || {
  ERROR_TEXT="output must be inside the repository outputs directory"
  exit 2
}
[[ -x "$PYTHON" ]] || {
  ERROR_TEXT="Python is not executable: $PYTHON"
  exit 2
}
[[ -x "$QQ_NOTIFIER" ]] || {
  ERROR_TEXT="QQ notifier is not executable: $QQ_NOTIFIER"
  exit 2
}
ACTUAL_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
if [[ -n "$EXPECTED_COMMIT" && "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  ERROR_TEXT="repository commit drifted: expected $EXPECTED_COMMIT, found $ACTUAL_COMMIT"
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then
  ERROR_TEXT="repository has uncommitted changes: $ROOT"
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

"$PYTHON" - "$ROOT" <<'PY'
import pathlib
import sys

import change_agent

root = pathlib.Path(sys.argv[1]).resolve()
package = pathlib.Path(change_agent.__file__).resolve()
try:
    package.relative_to(root / "change_agent")
except ValueError as error:
    raise SystemExit(
        f"change_agent import escaped test repository: {package} not under {root}"
    ) from error
print(f"change_agent_import={package}")
PY

case "$SUITE" in
  targeted)
    PATTERNS=(
      test_stage_backends.py
      test_bailian_adapter.py
      test_staged_verifier.py
      test_local_verifier_analysis.py
      test_verifier_regions.py
      test_environment.py
      test_authorized_action_protocol.py
      test_runner_action_protocol.py
      test_qwen3vl_verifier.py
      test_executor_locality.py
    )
    ;;
  regression)
    PATTERNS=('test_*.py')
    ;;
  *)
    ERROR_TEXT="unsupported suite: $SUITE"
    exit 2
    ;;
esac

FAILED=()
for pattern in "${PATTERNS[@]}"; do
  set +e
  "$PYTHON" -m unittest discover -s tests -p "$pattern"
  rc="$?"
  set -e
  if [[ "$rc" -ne 0 ]]; then
    FAILED+=("$pattern")
  fi
done
if [[ "${#FAILED[@]}" -ne 0 ]]; then
  KEY_RESULT="failed_patterns=${FAILED[*]}"
  ERROR_TEXT="$KEY_RESULT"
  exit 1
fi
KEY_RESULT="passed_patterns=${PATTERNS[*]}"
