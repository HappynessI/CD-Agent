#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd -P)"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:-/guisongxia01/pangchao/wangyihan}")"
RUN_DIR="$(realpath -m -- "${RUN_DIR:?RUN_DIR must name an output run}")"
CURATION_ROOT="$(realpath -e -- "${CURATION_ROOT:-$RUN_DIR/curation}")"
DATA_ROUND="${DATA_ROUND:-v1}"
WORKER="$ROOT/tools/run_qwen3vl_v121_data_build.slurm";COMMIT="$(git -C "$ROOT" rev-parse HEAD)";JOB_NAME="${JOB_NAME:-CA0801V121BUILD}"
[[ -f "$CURATION_ROOT/.curation_complete" ]] || { echo "curation incomplete" >&2;exit 2; }
if [[ ! -e "$RUN_DIR" ]]; then mkdir "$RUN_DIR"; mkdir "$RUN_DIR/logs"; fi
if [[ -e "$RUN_DIR/data" && ! -f "$RUN_DIR/data/.data_complete" ]]; then echo "incomplete data output exists" >&2;exit 2; fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || { echo "worktree dirty" >&2;exit 2; }
scontrol ping|grep -q UP || exit 3
ACTIVE="$(squeue -h -u guisongxia --name="$JOB_NAME" -o '%i %T')";[[ -z "$ACTIVE" ]]||{ echo "duplicate: $ACTIVE" >&2;exit 2; }
JOB="$(sbatch --parsable --job-name="$JOB_NAME" --partition=gre --cpus-per-task=2 --mem=16G --time=00:30:00 --output="$RUN_DIR/logs/data-build-%j.out" --error="$RUN_DIR/logs/data-build-%j.err" --export="ALL,REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,EXPECTED_COMMIT=$COMMIT,CURATION_ROOT=$CURATION_ROOT,DATA_ROUND=$DATA_ROUND" "$WORKER" "$RUN_DIR")";JOB="${JOB%%;*}"
printf 'data_build_job=%s\ncode_commit=%s\nrun_dir=%s\n' "$JOB" "$COMMIT" "$RUN_DIR" >"$RUN_DIR/data-build-submission-${JOB}.txt";printf 'job=%s\nrun_dir=%s\n' "$JOB" "$RUN_DIR"
