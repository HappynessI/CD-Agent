#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(cd "$ROOT/.." && pwd -P)"
EXPECTED_HARNESS_COMMIT="${EXPECTED_HARNESS_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
WORKER="$ROOT/tools/run_qwen3vl_base_e2e5.slurm"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}"
INPUT_ROOT="${INPUT_ROOT:-$WORKSPACE/datasets/LEVIR-CD/test_256}"
PARTITION="${PARTITION:-gre}"
NODE="${NODE:-}"
DEPENDENCY="${DEPENDENCY:-}"
ARM="${1:?usage: NODE=gpuXX submit_local_verifier_e2e5.sh control|compact [ARM_REPO]}"
ARM_REPO="${2:-$ROOT}"
OUTPUT_ROOT="$(realpath -e -- "$ROOT/outputs")"
# e2e3 is a fixed fast-iteration gate.  Only e2e5 outputs are eligible for
# compare_local_verifier_ab.py and the final freeze decision.
EVALUATION_PROFILE="${EVALUATION_PROFILE-e2e5}"
case "$EVALUATION_PROFILE" in
  e2e5)
    EVALUATION_ID="DEV-EXTENDED-5-V1"
    SAMPLES=(test_20_15 test_78_13 test_85_16 test_1_1 test_50_8)
    ;;
  e2e3)
    EVALUATION_ID="DEV-CORE-3-V1"
    SAMPLES=(test_20_15 test_1_1 test_50_8)
    ;;
  smoke1)
    EVALUATION_ID="DEV-SMOKE-1-V1"
    SAMPLES=(test_20_15)
    ;;
  *)
    echo "EVALUATION_PROFILE must be exactly e2e5, e2e3, or smoke1" >&2
    exit 2
    ;;
esac
if [[ "$EVALUATION_PROFILE" != e2e5 && -z "${OUTPUT:-}" ]]; then
  echo "$EVALUATION_PROFILE requires an explicit non-empty OUTPUT to avoid an e2e5-labelled path" >&2
  exit 2
fi

case "$ARM" in
  control)
    PROTOCOL_VERSION="${PROTOCOL_VERSION:-v11.3}"
    PINNED_COMMIT="$(git -C "$ROOT" rev-parse f80e270^{commit})"
    EXPECTED_COMMIT="${EXPECTED_COMMIT:-$PINNED_COMMIT}"
    OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0731(6)-qwen3vl-v11_3-control-v12_1-harness-e2e5}"
    JOB_NAME="${JOB_NAME:-CA31Q3V11Ctrl}"
    EXPERIMENT_ARM="v11_3-control"
    ;;
  compact)
    PROTOCOL_VERSION="${PROTOCOL_VERSION:-v12.1}"
    PINNED_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
    EXPECTED_COMMIT="${EXPECTED_COMMIT:-$PINNED_COMMIT}"
    OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0731(7)-qwen3vl-v12_1-compact-e2e5}"
    JOB_NAME="${JOB_NAME:-CA31Q3V121Cmp}"
    EXPERIMENT_ARM="v12_1-compact"
    ;;
  *)
    echo "arm must be control or compact" >&2
    exit 2
    ;;
esac

[[ "$EXPECTED_COMMIT" == "$PINNED_COMMIT" ]] || { echo "arm commit must remain pinned to $PINNED_COMMIT" >&2; exit 2; }
if { [[ "$ARM" == control && "$PROTOCOL_VERSION" != v11.3 ]]; } || { [[ "$ARM" == compact && "$PROTOCOL_VERSION" != v12.1 ]]; }; then echo "protocol override disagrees with fixed A/B arm" >&2; exit 2; fi
if [[ -z "$NODE" || ! "$NODE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "NODE must name a preflighted A800 node" >&2
  exit 2
fi
[[ "$PARTITION" == gre ]] || { echo "A/B partition must remain pinned to gre" >&2; exit 2; }
if [[ -n "$DEPENDENCY" && ! "$DEPENDENCY" =~ ^afterany:[0-9]+$ ]]; then echo "DEPENDENCY must be empty or afterany:<job-id>" >&2; exit 2; fi
if [[ -z "$JOB_NAME" || ! "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "JOB_NAME contains unsupported characters" >&2
  exit 2
fi
if [[ ! -d "$ARM_REPO" ]]; then
  echo "arm repository is missing: $ARM_REPO" >&2
  exit 2
fi
ARM_REPO="$(realpath -e -- "$ARM_REPO")"
MODEL_PATH="$(realpath -e -- "$MODEL_PATH")"
INPUT_ROOT="$(realpath -e -- "$INPUT_ROOT")"
OUTPUT="$(realpath -m -- "$OUTPUT")"
[[ "$OUTPUT" == "$OUTPUT_ROOT/"* ]] || { echo "output escapes $OUTPUT_ROOT" >&2; exit 2; }
[[ "$MODEL_PATH" == "$(realpath -e -- "$WORKSPACE/models/Qwen3-VL-2B-Instruct")" ]] || { echo "baseline model path is not pinned" >&2; exit 2; }
[[ "$INPUT_ROOT" == "$(realpath -e -- "$WORKSPACE/datasets/LEVIR-CD/test_256")" ]] || { echo "diagnostic input root is not pinned" >&2; exit 2; }
ACTUAL_COMMIT="$(git -C "$ARM_REPO" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "arm commit mismatch: expected $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$ARM_REPO" status --porcelain --untracked-files=all)" ]]; then
  echo "arm repository has uncommitted changes: $ARM_REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "experiment harness has uncommitted changes: $ROOT" >&2
  exit 2
fi
if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$EXPECTED_HARNESS_COMMIT" ]]; then
  echo "harness commit mismatch: expected $EXPECTED_HARNESS_COMMIT" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
for executable in "$PYTHON" "$WORKER" "$QQ_NOTIFIER"; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is missing: $executable" >&2
    exit 2
  fi
done
WORKER_SHA256="$(sha256sum -- "$WORKER" | awk '{print $1}')"
if [[ ! "$WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "cannot compute the pinned harness worker SHA-256" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "pinned base model is incomplete: $MODEL_PATH" >&2
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
if ! NODE_INFO="$(scontrol show node "$NODE" 2>&1)"; then
  echo "cannot inspect fixed node: $NODE" >&2
  printf '%s\n' "$NODE_INFO" >&2
  exit 3
fi
if grep -Eq 'State=[^[:space:]]*(DOWN|DRAIN|FAIL|MAINT|UNKNOWN)' <<<"$NODE_INFO"; then
  echo "fixed node is unavailable: $NODE" >&2
  printf '%s\n' "$NODE_INFO" >&2
  exit 2
fi

mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
EXPORTS="ALL,REPO_ROOT=$ARM_REPO,HARNESS_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,PYTHON=$PYTHON,QQ_NOTIFIER=$QQ_NOTIFIER,MODEL_PATH=$MODEL_PATH,INPUT_ROOT=$INPUT_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,EXPECTED_HARNESS_COMMIT=$EXPECTED_HARNESS_COMMIT,EXPECTED_WORKER_SHA256=$WORKER_SHA256,EXPERIMENT_ARM=$EXPERIMENT_ARM,PROTOCOL_VERSION=$PROTOCOL_VERSION,EVALUATION_PROFILE=$EVALUATION_PROFILE"
SBATCH_DEPENDENCY=()
[[ -z "$DEPENDENCY" ]] || SBATCH_DEPENDENCY=(--dependency="$DEPENDENCY")
set +e
JOB_OUTPUT="$(
  sbatch --parsable \
    --job-name="$JOB_NAME" \
    "${SBATCH_DEPENDENCY[@]}" \
    --partition="$PARTITION" \
    --nodelist="$NODE" \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=32G \
    --time=03:00:00 \
    --output="$OUTPUT/logs/slurm-%j.out" \
    --error="$OUTPUT/logs/slurm-%j.err" \
    --export="$EXPORTS" \
    "$WORKER" "$OUTPUT" 2>&1
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

printf 'job_id=%s\njob_name=%s\npartition=%s\nnode=%s\ndependency=%s\narm=%s\nprotocol=%s\narm_repo=%s\nexpected_commit=%s\nexpected_harness_commit=%s\nworker=%s\nworker_sha256=%s\nresources=gpu:1,cpu:2,mem:32G,time=03:00:00\nevaluation_profile=%s\nevaluation_id=%s\nsamples=%s\noutput=%s\n' \
  "$JOB" "$JOB_NAME" "$PARTITION" "$NODE" "${DEPENDENCY:-none}" "$EXPERIMENT_ARM" \
  "$PROTOCOL_VERSION" "$ARM_REPO" "$EXPECTED_COMMIT" "$EXPECTED_HARNESS_COMMIT" \
  "$WORKER" "$WORKER_SHA256" "$EVALUATION_PROFILE" "$EVALUATION_ID" \
  "${SAMPLES[*]}" "$OUTPUT" \
  >"$OUTPUT/submission.txt"
printf 'job=%s\narm=%s\nprotocol=%s\nevaluation_profile=%s\nevaluation_id=%s\ncommit=%s\noutput=%s\nrun=%s\n' \
  "$JOB" "$EXPERIMENT_ARM" "$PROTOCOL_VERSION" "$EVALUATION_PROFILE" \
  "$EVALUATION_ID" "$EXPECTED_COMMIT" "$OUTPUT" "$OUTPUT/run"
