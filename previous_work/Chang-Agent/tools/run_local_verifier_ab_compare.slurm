#!/usr/bin/env bash
#SBATCH --job-name=CA31V121Compare
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:-$(cd "$ROOT/.." && pwd -P)}")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
CONTROL="$(realpath -e -- "${1:?control output is required}")"
TREATMENT="$(realpath -e -- "${2:?treatment output is required}")"
OUTPUT="$(realpath -e -- "${3:?comparison output is required}")"
OUTPUT_ROOT="$(realpath -e -- "$ROOT/outputs")"
LOG_DIR="$OUTPUT/logs"
COMPARISON="$OUTPUT/comparison.json"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""

for path in "$CONTROL" "$TREATMENT" "$OUTPUT"; do [[ "$path" == "$OUTPUT_ROOT/"* ]] || { echo "comparison path escapes $OUTPUT_ROOT: $path" >&2; exit 2; }; done
[[ "$CONTROL" != "$TREATMENT" && "$OUTPUT" != "$CONTROL" && "$OUTPUT" != "$TREATMENT" ]] || { echo "comparison paths must differ" >&2; exit 2; }
[[ -d "$LOG_DIR" ]] || { echo "shared log directory is missing: $LOG_DIR" >&2; exit 2; }

finish() {
  local rc="$?"
  trap - EXIT
  set +e
  local elapsed="$((SECONDS - START_SECONDS))"
  if [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  elif [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="comparison exited with code $rc"
  fi
  QQ_SUB="Local verifier A/B comparison ${STATUS}" \
  QQ_BODY="status=${STATUS}; result=${KEY_RESULT}; error=${ERROR_TEXT:-none}; job=${JOB_ID}; elapsed=${elapsed}s; output=${OUTPUT}" \
    "$QQ_NOTIFIER" >"/tmp/local_verifier_compare_${JOB_ID}.qq.log" 2>&1 || true
  rm -f -- "/tmp/local_verifier_compare_${JOB_ID}.qq.log"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\ncommit=%s\nkey_result=%s\nerror=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$EXPECTED_COMMIT" "$KEY_RESULT" \
    "$ERROR_TEXT" >"$LOG_DIR/status-${JOB_ID}.txt"
  exit "$rc"
}
trap finish EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  ERROR_TEXT="worker must run under Slurm"
  exit 2
}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ERROR_TEXT="CPU comparison job unexpectedly received a GPU allocation"
  exit 2
fi
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  ERROR_TEXT="comparison harness commit drifted"
  exit 2
}
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || {
  ERROR_TEXT="comparison harness has uncommitted changes"
  exit 2
}
[[ -x "$PYTHON" ]] || { ERROR_TEXT="Python is not executable: $PYTHON"; exit 2; }
[[ -x "$QQ_NOTIFIER" ]] || { ERROR_TEXT="QQ notifier is not executable: $QQ_NOTIFIER"; exit 2; }
for required in \
  "$ROOT/tools/compare_local_verifier_ab.py" \
  "$CONTROL/input_manifest.json" \
  "$CONTROL/protocol_summary.json" \
  "$CONTROL/baseline_summary.json" \
  "$TREATMENT/input_manifest.json" \
  "$TREATMENT/protocol_summary.json" \
  "$TREATMENT/baseline_summary.json"; do
  [[ -f "$required" ]] || {
    ERROR_TEXT="required comparison input is missing: $required"
    exit 2
  }
done

export PYTHONDONTWRITEBYTECODE=1
KEY_RESULT="$(
  "$PYTHON" "$ROOT/tools/compare_local_verifier_ab.py" \
    --control "$CONTROL" \
    --treatment "$TREATMENT" \
    --expected-harness-commit "$EXPECTED_COMMIT" \
    --output "$COMPARISON"
)"
echo "$KEY_RESULT"
