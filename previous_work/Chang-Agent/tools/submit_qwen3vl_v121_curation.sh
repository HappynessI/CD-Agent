#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="${WORKSPACE_ROOT:-/guisongxia01/pangchao/wangyihan}"
WORKSPACE="$(realpath -e -- "$WORKSPACE")"
PRIMARY_REPO="${PRIMARY_REPO:-$WORKSPACE/Chang-Agent}"
OUTPUT_ROOT="$(realpath -e -- "$PRIMARY_REPO/outputs")"
OUTPUT="$(realpath -m -- "${OUTPUT:-$OUTPUT_ROOT/CA_0801(8)-qwen3vl-v121-baseline-data-v1}")"
WORKER="$ROOT/tools/run_qwen3vl_v121_curation.slurm"
NODE="${NODE:?NODE must name a preflighted GPU node}"
PARTITION="${PARTITION:-gre}"
JOB_NAME="${JOB_NAME:-CA0801V121CURATE}"

[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || { echo "output escapes $OUTPUT_ROOT" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 2; }
[[ -x "$WORKER" ]] || { echo "worker is not executable: $WORKER" >&2; exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { echo "experiment worktree is dirty" >&2; exit 2; }
scontrol ping | grep -q UP || { echo "Slurm controller is not UP" >&2; exit 3; }
ACTIVE="$(squeue -h -u guisongxia --name="$JOB_NAME" -o '%i %T')"
[[ -z "$ACTIVE" ]] || { echo "active duplicate exists: $ACTIVE" >&2; exit 2; }
NODE_INFO="$(scontrol show node "$NODE")"
grep -Eq 'State=[^[:space:]]*(DOWN|DRAIN|FAIL|MAINT|UNKNOWN)' <<<"$NODE_INFO" && { echo "node unavailable: $NODE" >&2; exit 2; }

mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
set +e
SUBMIT="$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --nodelist="$NODE" \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=48G \
  --time=01:30:00 \
  --output="$OUTPUT/logs/slurm-%j.out" \
  --error="$OUTPUT/logs/slurm-%j.err" \
  --export="ALL,REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,OUTPUT_ROOT=$OUTPUT_ROOT" \
  "$WORKER" "$OUTPUT" 2>&1)"
RC="$?"
set -e
if [[ "$RC" -ne 0 ]]; then
  rmdir "$OUTPUT/logs" "$OUTPUT" 2>/dev/null || true
  printf '%s\n' "$SUBMIT" >&2
  exit "$RC"
fi
JOB_ID="${SUBMIT%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch result: $SUBMIT" >&2; exit 3; }
printf 'job_id=%s\njob_name=%s\nnode=%s\nresources=gpu:1,cpu:4,mem:48G,time=01:30:00\ncode_root=%s\ncode_commit=%s\noutput=%s\n' \
  "$JOB_ID" "$JOB_NAME" "$NODE" "$ROOT" "$(git -C "$ROOT" rev-parse HEAD)" "$OUTPUT" \
  >"$OUTPUT/submission.txt"
printf 'job=%s\noutput=%s\n' "$JOB_ID" "$OUTPUT"
