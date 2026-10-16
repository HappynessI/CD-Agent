#!/usr/bin/env bash
set -euo pipefail

ROOT="/guisongxia01/pangchao/wangyihan/Chang-Agent"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
SUITE="${1:?suite is required}"
OUTPUT="${2:?output directory is required}"
START_SECONDS="$SECONDS"

cd "$ROOT"
STATUS=0
run_tests() {
  set +e
  "$PYTHON" -m unittest "$@"
  local code="$?"
  set -e
  if [[ "$code" -ne 0 ]]; then
    STATUS="$code"
  fi
}

case "$SUITE" in
  targeted)
    run_tests discover -s tests -p test_executor_locality.py
    run_tests discover -s tests -p test_environment.py
    run_tests discover -s tests -p test_stage_backends.py
    run_tests discover -s tests -p test_staged_verifier.py
    run_tests discover -s tests -p test_bailian_adapter.py
    run_tests discover -s tests -p test_verifier_regions.py
    run_tests discover -s tests -p test_subprocess_initialization.py
    run_tests discover -s tests -p test_runner_action_protocol.py
    ;;
  regression)
    run_tests discover -s tests -p 'test_*.py'
    ;;
  *)
    echo "unsupported suite: $SUITE" >&2
    exit 2
    ;;
esac

ELAPSED="$((SECONDS - START_SECONDS))"
if [[ "$STATUS" -eq 0 ]]; then
  RESULT=success
else
  RESULT=failed
fi

QQ_SUB="CA 0723 verifier ${SUITE} ${RESULT}" \
QQ_BODY="status=${RESULT}; suite=${SUITE}; elapsed=${ELAPSED}s; output=${OUTPUT}; job=${SLURM_JOB_ID}" \
  /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq \
  >"/tmp/ca0723_verifier_${SLURM_JOB_ID}.qq.log" 2>&1 || true
rm -f "/tmp/ca0723_verifier_${SLURM_JOB_ID}.qq.log"
exit "$STATUS"
