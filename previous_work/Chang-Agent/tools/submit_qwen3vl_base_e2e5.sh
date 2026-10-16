#!/usr/bin/env bash
set -euo pipefail

# Submit the fixed five-sample, local-Qwen staged-verifier base baseline.
# The output parent is reserved atomically and the actual rollout is written to
# its absent `run/` child so run_levir_change_agent.py can enforce no overwrite.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
WORKER="${WORKER:-$ROOT/tools/run_qwen3vl_base_e2e5.slurm}"
OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0730(4)-qwen3vl-base-e2e5}"
JOB_NAME="${JOB_NAME:-CA30-Q3VL-BE2E5}"
PARTITION="${PARTITION:-gre}"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}"
EXPECTED_MODEL_PATH="$WORKSPACE/models/Qwen3-VL-2B-Instruct"
INPUT_ROOT="${INPUT_ROOT:-$WORKSPACE/datasets/LEVIR-CD/test_256}"
SAMPLES=(test_20_15 test_78_13 test_85_16 test_1_1 test_50_8)

if [[ -z "$JOB_NAME" || ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "JOB_NAME contains unsupported characters" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
NODE="${NODE:-}"
if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must name the preflighted A800 node (for example gpu25)" >&2
  exit 2
fi
if [[ "$(realpath -e -- "$MODEL_PATH")" != "$(realpath -e -- "$EXPECTED_MODEL_PATH")" ]]; then
  echo "base diagnostic must use the pinned model path: $EXPECTED_MODEL_PATH" >&2
  exit 2
fi
for executable in "$PYTHON" "$WORKER"; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is missing: $executable" >&2
    exit 2
  fi
done
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "base model is missing: $MODEL_PATH" >&2
  exit 2
fi
for sample in "${SAMPLES[@]}"; do
  for directory in A B label_cvt; do
    if [[ ! -f "$INPUT_ROOT/$directory/$sample.png" ]]; then
      echo "fixed evaluation input is missing: $directory/$sample.png" >&2
      exit 2
    fi
  done
done

if ! SLURM_PING="$(scontrol ping 2>&1)"; then
  echo "Slurm preflight failed; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
if ! grep -q 'UP' <<<"$SLURM_PING"; then
  echo "Slurm controller is not UP; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
if ! DUPLICATE_JOBS="$(squeue -h -u guisongxia --name="$JOB_NAME" -o '%i %T' 2>&1)"; then
  echo "Slurm duplicate-job preflight failed; no job submitted" >&2
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  exit 3
fi
if [[ -n "$DUPLICATE_JOBS" ]]; then
  echo "active job with the same name already exists; no job submitted" >&2
  printf '%s\n' "$DUPLICATE_JOBS" >&2
  exit 2
fi
if ! NODE_INFO="$(scontrol show node "$NODE" 2>&1)"; then
  echo "cannot inspect fixed node: $NODE" >&2
  printf '%s\n' "$NODE_INFO" >&2
  exit 3
fi
if ! grep -q 'State=IDLE' <<<"$NODE_INFO"; then
  echo "fixed node is not idle at submission time: $NODE" >&2
  printf '%s\n' "$NODE_INFO" >&2
  exit 2
fi

mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
set +e
JOB_OUTPUT="$(
  REPO_ROOT="$ROOT" PYTHON="$PYTHON" MODEL_PATH="$MODEL_PATH" INPUT_ROOT="$INPUT_ROOT" \
    sbatch --parsable \
      --job-name="$JOB_NAME" \
      --partition="$PARTITION" \
      --nodelist="$NODE" \
      --gres=gpu:1 \
      --cpus-per-task=2 \
      --mem=32G \
      --time=03:00:00 \
      --output="$OUTPUT/logs/slurm-%j.out" \
      --error="$OUTPUT/logs/slurm-%j.err" \
      --export=ALL,REPO_ROOT,PYTHON,MODEL_PATH,INPUT_ROOT \
      "$WORKER" "$OUTPUT" 2>&1
)"
SUBMIT_RC="$?"
set -e
if [[ "$SUBMIT_RC" -ne 0 ]]; then
  rmdir "$OUTPUT/logs" "$OUTPUT" 2>/dev/null || true
  echo "sbatch failed; reserved empty output was removed" >&2
  printf '%s\n' "$JOB_OUTPUT" >&2
  exit "$SUBMIT_RC"
fi
JOB="${JOB_OUTPUT%%;*}"
if [[ ! "$JOB" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an unexpected job id: $JOB_OUTPUT" >&2
  exit 3
fi

printf 'job_id=%s\njob_name=%s\npartition=%s\nnode=%s\nresources=gpu:1,cpu:2,mem:32G,time=03:00:00\nmodel_state=base\nevaluation_id=DEV-EXTENDED-5-V1\nsamples=%s\noutput=%s\n' \
  "$JOB" "$JOB_NAME" "$PARTITION" "$NODE" "${SAMPLES[*]}" "$OUTPUT" >"$OUTPUT/submission.txt"
printf 'job=%s\noutput=%s\nrun=%s\nsamples=%s\n' \
  "$JOB" "$OUTPUT" "$OUTPUT/run" "${SAMPLES[*]}"
