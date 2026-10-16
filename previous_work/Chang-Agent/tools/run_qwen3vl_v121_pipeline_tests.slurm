#!/usr/bin/env bash
#SBATCH --job-name=CA0801V121TEST
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
OUTPUT="$(realpath -e -- "${1:?usage: worker OUTPUT SUITE}")"
SUITE="${2:?usage: worker OUTPUT SUITE}"
OUTPUT_ROOT="$(realpath -e -- "${OUTPUT_ROOT:?OUTPUT_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
STATUS="failed"
RESULT="not_started"

finish() {
  local rc="$?"
  trap - EXIT
  set +e
  [[ "$rc" -eq 0 ]] && STATUS="success"
  local elapsed="$((SECONDS - START_SECONDS))"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\nsuite=%s\ncommit=%s\nresult=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$SUITE" "$EXPECTED_COMMIT" "$RESULT" \
    >"$OUTPUT/logs/status-${SUITE}-${JOB_ID}.txt"
  QQ_SUB="Qwen3-VL v12.1 pipeline tests ${SUITE}" \
  QQ_BODY="status=${STATUS}; result=${RESULT}; job=${JOB_ID}; elapsed=${elapsed}s; output=${OUTPUT}" \
    "$QQ_NOTIFIER" >"/tmp/qwen3vl_v121_tests_${JOB_ID}.qq.log" 2>&1 || true
  rm -f -- "/tmp/qwen3vl_v121_tests_${JOB_ID}.qq.log"
  exit "$rc"
}
trap finish EXIT

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "run only through Slurm" >&2; exit 2; }
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CPU tests received GPU" >&2; exit 2; }
[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || { echo "output escapes output root" >&2; exit 2; }
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "commit drift" >&2; exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { echo "worktree dirty" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT"
cd "$ROOT"
case "$SUITE" in
  targeted) PATTERN='test_qwen3vl_v121_curation.py' ;;
  regression) PATTERN='test_*.py' ;;
  *) echo "unsupported suite: $SUITE" >&2; exit 2 ;;
esac
"$PYTHON" -m unittest discover -s tests -p "$PATTERN"
RESULT="passed_pattern=${PATTERN}"
