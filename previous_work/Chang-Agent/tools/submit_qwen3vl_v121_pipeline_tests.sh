#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:-/guisongxia01/pangchao/wangyihan}")"
PRIMARY_REPO="$(realpath -e -- "${PRIMARY_REPO:-$WORKSPACE/Chang-Agent}")"
OUTPUT_ROOT="$(realpath -e -- "$PRIMARY_REPO/outputs")"
OUTPUT="$(realpath -m -- "${OUTPUT:-$OUTPUT_ROOT/CA_0801(7)-qwen3vl-v121-baseline-pipeline-tests-rerun1}")"
WORKER="$ROOT/tools/run_qwen3vl_v121_pipeline_tests.slurm"
EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

[[ ! -e "$OUTPUT" ]] || { echo "output exists: $OUTPUT" >&2; exit 2; }
[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || { echo "output escapes output root" >&2; exit 2; }
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { echo "worktree dirty" >&2; exit 2; }
scontrol ping | grep -q UP || { echo "Slurm controller not UP" >&2; exit 3; }
for name in CA0801V121Target CA0801V121Regress; do
  ACTIVE="$(squeue -h -u guisongxia --name="$name" -o '%i %T')"
  [[ -z "$ACTIVE" ]] || { echo "active duplicate: $ACTIVE" >&2; exit 2; }
done
mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
EXPORTS="ALL,REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,OUTPUT_ROOT=$OUTPUT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT"
TARGET="$(sbatch --parsable --job-name=CA0801V121Target --partition=gre --cpus-per-task=2 --mem=8G --time=00:20:00 --output="$OUTPUT/logs/targeted-%j.out" --error="$OUTPUT/logs/targeted-%j.err" --export="$EXPORTS" "$WORKER" "$OUTPUT" targeted)"
TARGET="${TARGET%%;*}"
REGRESSION="$(sbatch --parsable --dependency="afterany:$TARGET" --job-name=CA0801V121Regress --partition=gre --cpus-per-task=2 --mem=8G --time=00:30:00 --output="$OUTPUT/logs/regression-%j.out" --error="$OUTPUT/logs/regression-%j.err" --export="$EXPORTS" "$WORKER" "$OUTPUT" regression)"
REGRESSION="${REGRESSION%%;*}"
printf 'commit=%s\ntargeted_job=%s\nregression_job=%s\noutput=%s\n' "$EXPECTED_COMMIT" "$TARGET" "$REGRESSION" "$OUTPUT" >"$OUTPUT/submission.txt"
printf 'targeted_job=%s\nregression_job=%s\noutput=%s\n' "$TARGET" "$REGRESSION" "$OUTPUT"
