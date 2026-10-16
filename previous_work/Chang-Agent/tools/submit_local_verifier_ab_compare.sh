#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(cd "$ROOT/.." && pwd -P)"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
WORKER="${WORKER:-$ROOT/tools/run_local_verifier_ab_compare.slurm}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
CONTROL="${CONTROL:-$ROOT/outputs/CA_0731(6)-qwen3vl-v11_3-control-v12_1-harness-e2e5}"
TREATMENT="${TREATMENT:-$ROOT/outputs/CA_0731(7)-qwen3vl-v12_1-compact-e2e5}"
OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0731(8)-qwen3vl-v11_3-v12_1-comparison}"
JOB_NAME="${JOB_NAME:-CA31V121Compare}"
PARTITION="${PARTITION:-gre}"
OUTPUT_ROOT="$(realpath -e -- "$ROOT/outputs")"
CONTROL="$(realpath -e -- "$CONTROL")"
TREATMENT="$(realpath -e -- "$TREATMENT")"
OUTPUT="$(realpath -m -- "$OUTPUT")"

for path in "$CONTROL" "$TREATMENT"; do [[ "$path" == "$OUTPUT_ROOT/"* ]] || { echo "A/B input escapes $OUTPUT_ROOT: $path" >&2; exit 2; }; done
[[ "$CONTROL" != "$TREATMENT" ]] || { echo "control and treatment must differ" >&2; exit 2; }
[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || { echo "comparison output escapes $OUTPUT_ROOT" >&2; exit 2; }

if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "comparison harness has uncommitted changes" >&2
  exit 2
fi
[[ "$PARTITION" == gre ]] || { echo "comparison partition must remain pinned to gre" >&2; exit 2; }
for executable in "$PYTHON" "$WORKER" "$QQ_NOTIFIER"; do [[ -x "$executable" ]] || { echo "required executable is missing: $executable" >&2; exit 2; }; done
for required in \
  "$CONTROL/input_manifest.json" \
  "$CONTROL/protocol_summary.json" \
  "$CONTROL/baseline_summary.json" \
  "$TREATMENT/input_manifest.json" \
  "$TREATMENT/protocol_summary.json" \
  "$TREATMENT/baseline_summary.json"; do
  if [[ ! -f "$required" ]]; then
    echo "required comparison input is missing: $required" >&2
    exit 2
  fi
done
if ! SLURM_PING="$(scontrol ping 2>&1)" || ! grep -q 'UP' <<<"$SLURM_PING"; then
  echo "Slurm controller is not UP; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
ACTIVE="$(squeue -h -u guisongxia --name="$JOB_NAME" -o '%i %T')"
if [[ -n "$ACTIVE" ]]; then
  echo "active duplicate job exists: $ACTIVE" >&2
  exit 2
fi

mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
EXPORTS="ALL,REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,PYTHON=$PYTHON,QQ_NOTIFIER=$QQ_NOTIFIER,EXPECTED_COMMIT=$EXPECTED_COMMIT"
set +e
JOB_OUTPUT="$(
  sbatch --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:10:00 \
    --output="$OUTPUT/logs/slurm-%j.out" \
    --error="$OUTPUT/logs/slurm-%j.err" \
    --export="$EXPORTS" \
    "$WORKER" "$CONTROL" "$TREATMENT" "$OUTPUT" 2>&1
)"
SUBMIT_RC="$?"
set -e
if [[ "$SUBMIT_RC" -ne 0 ]]; then
  rmdir "$OUTPUT/logs" "$OUTPUT" 2>/dev/null || true
  echo "sbatch failed; empty output reservation removed" >&2
  printf '%s\n' "$JOB_OUTPUT" >&2
  exit "$SUBMIT_RC"
fi
JOB="${JOB_OUTPUT%%;*}"
if [[ ! "$JOB" =~ ^[0-9]+$ ]]; then
  echo "sbatch returned an unexpected job id: $JOB_OUTPUT" >&2
  exit 3
fi
printf 'job_id=%s\nexpected_commit=%s\npartition=%s\nresources=cpu:2,mem:8G,time=00:10:00\ncontrol=%s\ntreatment=%s\noutput=%s\n' \
  "$JOB" "$EXPECTED_COMMIT" "$PARTITION" "$CONTROL" "$TREATMENT" "$OUTPUT" \
  >"$OUTPUT/submission.txt"
printf 'job=%s\noutput=%s\n' "$JOB" "$OUTPUT"
