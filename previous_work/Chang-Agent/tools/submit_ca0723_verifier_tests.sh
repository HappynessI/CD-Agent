#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_INDEX="${1:-1}"
OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0723(${RUN_INDEX})-verifier-batched-rubric-tests}"
TEST_ATTEMPT="${TEST_ATTEMPT:-1}"

if [[ ! "$RUN_INDEX" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX must be a non-negative integer" >&2
  exit 2
fi
if [[ -e "$OUTPUT" && "$TEST_ATTEMPT" == "1" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
if ! scontrol ping | grep -q UP; then
  echo "Slurm controller is not UP" >&2
  exit 3
fi

mkdir -p "$OUTPUT/logs"
TARGETED_JOB=$(sbatch --parsable --job-name=CA23VTarget \
  --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --output="$OUTPUT/logs/targeted-a${TEST_ATTEMPT}-%j.out" \
  --error="$OUTPUT/logs/targeted-a${TEST_ATTEMPT}-%j.err" \
  "$ROOT/tools/run_ca0723_verifier_tests.slurm" targeted "$OUTPUT")
REGRESSION_JOB=$(sbatch --parsable --job-name=CA23VRegress \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --output="$OUTPUT/logs/regression-a${TEST_ATTEMPT}-%j.out" \
  --error="$OUTPUT/logs/regression-a${TEST_ATTEMPT}-%j.err" \
  "$ROOT/tools/run_ca0723_verifier_tests.slurm" regression "$OUTPUT")

if [[ ! -e "$OUTPUT/test_manifest.md" ]]; then
  printf '%s\n' \
    '# CA_0723 verifier hardening test run' \
    '' \
    '- targeted: `test_executor_locality test_environment test_stage_backends test_staged_verifier test_bailian_adapter test_verifier_regions test_subprocess_initialization test_runner_action_protocol`' \
    '- regression: `unittest discover -s tests`' \
    '- compute policy: `Slurm CPU jobs; no management-node test execution`' \
    >"$OUTPUT/test_manifest.md"
fi
printf '%s\n' \
  '' \
  "## Attempt $TEST_ATTEMPT" \
  '' \
  "- status: \`submitted\`" \
  "- targeted_job: \`$TARGETED_JOB\`" \
  "- regression_job: \`$REGRESSION_JOB\`" \
  >>"$OUTPUT/test_manifest.md"

printf 'output=%s\ntargeted_job=%s\nregression_job=%s\n' \
  "$OUTPUT" "$TARGETED_JOB" "$REGRESSION_JOB"
