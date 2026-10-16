#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_INDEX="${1:-2}"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/outputs/CA_0723_archive/CA_0723(5)-bailian-proposal-atomic-v11-5sample/qwen37-no-thinking}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
FIXED_ACTION_MANIFEST="${FIXED_ACTION_MANIFEST:-$ROOT/tools/ca0723_fixed_action_manifest.json}"
SEGAGENT_PYTHON="${SEGAGENT_PYTHON:-/guisongxia01/pangchao/wangyihan/segagent-env/bin/python}"
SIMPLECLICK_CHECKPOINT="${SIMPLECLICK_CHECKPOINT:-/guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth}"
OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0726(${RUN_INDEX})-simpleclick-input-write-fixed-replay-3sample}"
JOB_NAME="${JOB_NAME:-CA26R${RUN_INDEX}Fixed}"
NODE="${NODE:-}"
SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0726_${RUN_INDEX}_fixed_replay}"
SAMPLES=(test_20_15 test_78_13 test_85_16)

if [[ ! "$RUN_INDEX" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "JOB_NAME contains unsupported characters" >&2
  exit 2
fi
if [[ -n "$NODE" && ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE contains unsupported characters" >&2
  exit 2
fi
if [[ -e "$OUTPUT" || -e "${OUTPUT}.incomplete" ]]; then
  echo "refusing to overwrite existing complete or incomplete output: $OUTPUT" >&2
  exit 2
fi
for required in \
  "$PYTHON" \
  "$SOURCE_RUN" \
  "$INPUT_ROOT" \
  "$FIXED_ACTION_MANIFEST" \
  "$SEGAGENT_PYTHON" \
  "$SIMPLECLICK_CHECKPOINT" \
  "$ROOT/tools/replay_ca0723_fixed_actions.py" \
  "$ROOT/tools/seeded_segmentation_worker.py" \
  "$ROOT/tools/run_ca0726_fixed_replay.slurm"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required replay path: $required" >&2
    exit 2
  fi
done
for sample in "${SAMPLES[@]}"; do
  for directory in A B label_cvt; do
    if [[ ! -f "$INPUT_ROOT/$directory/$sample.png" ]]; then
      echo "missing frozen sample input: $directory/$sample.png" >&2
      exit 2
    fi
  done
  if [[ ! -f "$SOURCE_RUN/trajectories/$sample/trajectory.json" ]]; then
    echo "missing frozen source trajectory: $sample" >&2
    exit 2
  fi
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
if ! DUPLICATES="$(squeue -u guisongxia --noheader --name="$JOB_NAME" --format='%i|%j|%T' 2>&1)"; then
  echo "failed to query duplicate jobs for Slurm user guisongxia" >&2
  printf '%s\n' "$DUPLICATES" >&2
  exit 3
fi
if [[ -n "${DUPLICATES//[[:space:]]/}" ]]; then
  echo "refusing duplicate fixed replay submission; active matching job:" >&2
  printf '%s\n' "$DUPLICATES" >&2
  exit 2
fi

mkdir -p "$SHARED_LOG_ROOT"
export PYTHON SOURCE_RUN INPUT_ROOT FIXED_ACTION_MANIFEST
export SEGAGENT_PYTHON SIMPLECLICK_CHECKPOINT
NODE_ARGS=()
if [[ -n "$NODE" ]]; then
  NODE_ARGS+=(--nodelist="$NODE")
fi
if ! JOB_ID="$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --partition=gre \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  "${NODE_ARGS[@]}" \
  --output="$SHARED_LOG_ROOT/slurm-%j.out" \
  --error="$SHARED_LOG_ROOT/slurm-%j.err" \
  "$ROOT/tools/run_ca0726_fixed_replay.slurm" "$OUTPUT" "$SHARED_LOG_ROOT")"; then
  rmdir "$SHARED_LOG_ROOT" 2>/dev/null || true
  echo "sbatch failed; no fixed replay job was recorded" >&2
  exit 3
fi

printf 'job=%s\noutput=%s\nsamples=%s\nnode=%s\npartition=gre\nresources=gpu:1,cpu:2,mem:8G,time:00:30:00\n' \
  "$JOB_ID" "$OUTPUT" "${SAMPLES[*]}" "${NODE:-slurm-selected}"
