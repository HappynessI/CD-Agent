#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(cd "$ROOT/.." && pwd -P)"
WORKER="${WORKER:-$ROOT/tools/run_local_verifier_tests.slurm}"
OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0731(5)-local-verifier-v12_1-tests}"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
TARGETED_NAME="${TARGETED_NAME:-CA31V121Target}"
REGRESSION_NAME="${REGRESSION_NAME:-CA31V121Regress}"
PARTITION="${PARTITION:-gre}"
OUTPUT_ROOT="$(realpath -e -- "$ROOT/outputs")"
OUTPUT="$(realpath -m -- "$OUTPUT")"

[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || { echo "output must be inside $OUTPUT_ROOT" >&2; exit 2; }
if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "repository has uncommitted changes; commit before submitting tests" >&2
  exit 2
fi
[[ "$PARTITION" == gre ]] || { echo "test partition must remain pinned to gre" >&2; exit 2; }
for executable in "$PYTHON" "$WORKER" "$QQ_NOTIFIER"; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is missing: $executable" >&2
    exit 2
  fi
done
if ! SLURM_PING="$(scontrol ping 2>&1)" || ! grep -q 'UP' <<<"$SLURM_PING"; then
  echo "Slurm controller is not UP; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
for job_name in "$TARGETED_NAME" "$REGRESSION_NAME"; do
  ACTIVE="$(squeue -h -u guisongxia --name="$job_name" -o '%i %T')"
  if [[ -n "$ACTIVE" ]]; then
    echo "active duplicate job exists for $job_name: $ACTIVE" >&2
    exit 2
  fi
done

mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
COMMON_EXPORT="ALL,REPO_ROOT=$ROOT,HARNESS_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,PYTHON=$PYTHON,QQ_NOTIFIER=$QQ_NOTIFIER,EXPECTED_COMMIT=$EXPECTED_COMMIT"
set +e
TARGETED_OUTPUT="$(
  sbatch --parsable \
    --job-name="$TARGETED_NAME" \
    --partition="$PARTITION" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:20:00 \
    --output="$OUTPUT/logs/targeted-%j.out" \
    --error="$OUTPUT/logs/targeted-%j.err" \
    --export="$COMMON_EXPORT" \
    "$WORKER" targeted "$OUTPUT" 2>&1
)"
TARGETED_RC="$?"
set -e
if [[ "$TARGETED_RC" -ne 0 ]]; then
  rmdir "$OUTPUT/logs" "$OUTPUT" 2>/dev/null || true
  echo "targeted sbatch failed; empty output reservation removed" >&2
  printf '%s\n' "$TARGETED_OUTPUT" >&2
  exit "$TARGETED_RC"
fi
TARGETED_JOB="${TARGETED_OUTPUT%%;*}"
if [[ ! "$TARGETED_JOB" =~ ^[0-9]+$ ]]; then
  echo "targeted sbatch returned an unexpected job id: $TARGETED_OUTPUT" >&2
  exit 3
fi

set +e
REGRESSION_OUTPUT="$(
  sbatch --parsable \
    --dependency="afterany:$TARGETED_JOB" \
    --job-name="$REGRESSION_NAME" \
    --partition="$PARTITION" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --output="$OUTPUT/logs/regression-%j.out" \
    --error="$OUTPUT/logs/regression-%j.err" \
    --export="$COMMON_EXPORT" \
    "$WORKER" regression "$OUTPUT" 2>&1
)"
REGRESSION_RC="$?"
set -e
if [[ "$REGRESSION_RC" -ne 0 ]]; then
  printf 'targeted_job=%s\nregression_submit_error=%s\n' \
    "$TARGETED_JOB" "$REGRESSION_OUTPUT" >"$OUTPUT/submission.txt"
  echo "regression sbatch failed; targeted job remains submitted" >&2
  printf '%s\n' "$REGRESSION_OUTPUT" >&2
  exit "$REGRESSION_RC"
fi
REGRESSION_JOB="${REGRESSION_OUTPUT%%;*}"
if [[ ! "$REGRESSION_JOB" =~ ^[0-9]+$ ]]; then
  echo "regression sbatch returned an unexpected job id: $REGRESSION_OUTPUT" >&2
  exit 3
fi

printf 'expected_commit=%s\npartition=%s\ntargeted_job=%s\nregression_job=%s\nresources=cpu:2,mem:8G\noutput=%s\n' \
  "$EXPECTED_COMMIT" "$PARTITION" "$TARGETED_JOB" "$REGRESSION_JOB" "$OUTPUT" \
  >"$OUTPUT/submission.txt"
printf 'output=%s\ntargeted_job=%s\nregression_job=%s\n' \
  "$OUTPUT" "$TARGETED_JOB" "$REGRESSION_JOB"
