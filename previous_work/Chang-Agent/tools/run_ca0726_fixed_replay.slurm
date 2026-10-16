#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="/guisongxia01/pangchao/wangyihan/Chang-Agent"
PYTHON="${PYTHON:-/guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/outputs/CA_0723_archive/CA_0723(5)-bailian-proposal-atomic-v11-5sample/qwen37-no-thinking}"
INPUT_ROOT="${INPUT_ROOT:-/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256}"
FIXED_ACTION_MANIFEST="${FIXED_ACTION_MANIFEST:-$ROOT/tools/ca0723_fixed_action_manifest.json}"
SEGAGENT_PYTHON="${SEGAGENT_PYTHON:-/guisongxia01/pangchao/wangyihan/segagent-env/bin/python}"
SIMPLECLICK_CHECKPOINT="${SIMPLECLICK_CHECKPOINT:-/guisongxia01/pangchao/wangyihan/models/SimpleClick/cocolvis_vit_large.pth}"
OUTPUT="${1:?output directory is required}"
SHARED_LOG_ROOT="${2:?shared Slurm log directory is required}"
START_SECONDS="$SECONDS"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run_ca0726_fixed_replay.slurm must run inside a Slurm allocation" >&2
  exit 2
fi

finish() {
  local status="$1"
  trap - EXIT
  set +e
  local result target detail elapsed shared_out shared_err qq_log owner_job
  elapsed="$((SECONDS - START_SECONDS))"
  if [[ "$status" -eq 0 ]]; then
    result="success"
  else
    result="failed"
  fi
  if [[ "$status" -eq 0 && -d "$OUTPUT" ]]; then
    target="$OUTPUT"
  elif [[ -d "${OUTPUT}.incomplete" ]]; then
    owner_job=""
    if [[ -f "${OUTPUT}.incomplete/.slurm_job_id" ]]; then
      IFS= read -r owner_job <"${OUTPUT}.incomplete/.slurm_job_id"
    fi
    if [[ "$owner_job" == "$SLURM_JOB_ID" ]]; then
      target="${OUTPUT}.incomplete"
    else
      target="${OUTPUT}.job-${SLURM_JOB_ID}.incomplete"
      mkdir -p "$target"
      printf '%s\n' \
        "status=incomplete" \
        "reason=atomic replay directory is owned by another Slurm job" \
        "job=${SLURM_JOB_ID}" \
        "owner_job=${owner_job:-unknown}" \
        >"$target/failure_summary.txt"
    fi
  else
    target="${OUTPUT}.job-${SLURM_JOB_ID}.incomplete"
    mkdir -p "$target"
    printf '%s\n' \
      "status=incomplete" \
      "reason=replay driver exited before creating its atomic work directory" \
      "job=${SLURM_JOB_ID}" \
      >"$target/failure_summary.txt"
  fi
  mkdir -p "$target/logs"
  shared_out="$SHARED_LOG_ROOT/slurm-${SLURM_JOB_ID}.out"
  shared_err="$SHARED_LOG_ROOT/slurm-${SLURM_JOB_ID}.err"
  [[ -f "$shared_out" ]] && cp -f "$shared_out" "$target/logs/slurm-${SLURM_JOB_ID}.out"
  [[ -f "$shared_err" ]] && cp -f "$shared_err" "$target/logs/slurm-${SLURM_JOB_ID}.err"
  printf 'status=%s\nelapsed_seconds=%s\njob=%s\nnode=%s\n' \
    "$result" "$elapsed" "$SLURM_JOB_ID" "${SLURM_JOB_NODELIST:-unknown}" \
    >"$target/logs/job_status.txt"
  if [[ "$status" -eq 0 && -f "$OUTPUT/completion_summary.txt" ]]; then
    IFS= read -r detail <"$OUTPUT/completion_summary.txt"
  elif [[ -f "$target/failure_summary.txt" ]]; then
    IFS= read -r detail <"$target/failure_summary.txt"
  else
    detail="see archived Slurm stderr"
  fi
  detail="${detail:0:700}"
  qq_log="/tmp/ca0726_fixed_replay_${SLURM_JOB_ID}.qq.log"
  QQ_SUB="CA 0726 fixed replay ${result}" \
  QQ_BODY="status=${result}; result=${detail}; elapsed=${elapsed}s; output=${target}; job=${SLURM_JOB_ID}; node=${SLURM_JOB_NODELIST:-unknown}" \
    /guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq \
    >"$qq_log" 2>&1 || true
  rm -f "$qq_log" "$shared_out" "$shared_err"
  rmdir "$SHARED_LOG_ROOT" 2>/dev/null || true
  exit "$status"
}
trap 'finish $?' EXIT

cd "$ROOT"
"$PYTHON" "$ROOT/tools/replay_ca0723_fixed_actions.py" \
  --source-run "$SOURCE_RUN" \
  --input-root "$INPUT_ROOT" \
  --fixed-action-manifest "$FIXED_ACTION_MANIFEST" \
  --output "$OUTPUT" \
  --segagent-python "$SEGAGENT_PYTHON" \
  --simpleclick-checkpoint "$SIMPLECLICK_CHECKPOINT" \
  --device cuda \
  --seed 42 \
  --timeout-seconds 300
