#!/usr/bin/env bash
#SBATCH --job-name=CA30-Q3VL-BE2E5
#SBATCH --partition=gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail

ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT must be exported by submit_local_verifier_e2e5.sh}")"
HARNESS_ROOT="$(realpath -e -- "${HARNESS_ROOT:-$ROOT}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:-$(cd "$ROOT/.." && pwd -P)}")"
PARENT="$(realpath -m -- "${1:?usage: sbatch run_qwen3vl_base_e2e5.slurm OUTPUT_PARENT}")"
RUN_DIR="$PARENT/run"
LOG_DIR="$PARENT/logs"
OUTPUT_ROOT="$(realpath -e -- "$HARNESS_ROOT/outputs")"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}"
EXPECTED_MODEL_PATH="$WORKSPACE/models/Qwen3-VL-2B-Instruct"
INPUT_ROOT="${INPUT_ROOT:-$WORKSPACE/datasets/LEVIR-CD/test_256}"
EXPECTED_INPUT_ROOT="$WORKSPACE/datasets/LEVIR-CD/test_256"
SEGAGENT_PYTHON="${SEGAGENT_PYTHON:-$WORKSPACE/segagent-env/bin/python}"
OMNIOVCD_PYTHON="${OMNIOVCD_PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
SIMPLECLICK_CHECKPOINT="${SIMPLECLICK_CHECKPOINT:-$WORKSPACE/models/SimpleClick/cocolvis_vit_large.pth}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$WORKSPACE/models/sam3/sam3.pt}"
SAM3_BPE="${SAM3_BPE:-$WORKSPACE/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
SUMMARIZER="${SUMMARIZER:-$HARNESS_ROOT/tools/summarize_qwen3vl_e2e_run.py}"
ANALYZER="${ANALYZER:-$HARNESS_ROOT/tools/analyze_staged_verifier_run.py}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPERIMENT_ARM="${EXPERIMENT_ARM:-local-base}"
PROTOCOL_VERSION="${PROTOCOL_VERSION:-v12.1}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
EXPECTED_HARNESS_COMMIT="${EXPECTED_HARNESS_COMMIT:-}"
EXPECTED_WORKER_SHA256="${EXPECTED_WORKER_SHA256:?EXPECTED_WORKER_SHA256 must be exported by submit_local_verifier_e2e5.sh}"
JOB_ID="${SLURM_JOB_ID:-manual-$$}"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256=""
TERMINATION_SIGNAL=""
TERMINATION_EXIT_CODE=""
FINISHING=0
# Keep this profile table identical to submit_local_verifier_e2e5.sh.  The
# manifest and both summaries persist the resolved identity independently.
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
    echo "qwen3vl_base_e2e5: EVALUATION_PROFILE must be exactly e2e5, e2e3, or smoke1" >&2
    exit 2
    ;;
esac

if [[ "$PARENT" != "$OUTPUT_ROOT/"* || ! -d "$PARENT" || ! -d "$LOG_DIR" ]]; then echo "qwen3vl_base_e2e5: invalid output reservation under $OUTPUT_ROOT" >&2; exit 2; fi

die() {
  ERROR_TEXT="$*"
  echo "qwen3vl_base_e2e5: $*" >&2
  return 1
}

record_error() {
  local rc="$?"
  if [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

write_status_receipt() {
  local rc="$1"
  local elapsed="$2"
  local receipt_path="$LOG_DIR/status-${JOB_ID}.txt"
  local temporary_path="${receipt_path}.tmp-${BASHPID}"
  if ! printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\nrun_dir=%s\nworker_sha256=%s\nevaluation_profile=%s\nevaluation_id=%s\nsamples=%s\nkey_result=%s\nerror=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$RUN_DIR" \
    "$ACTUAL_WORKER_SHA256" "$EVALUATION_PROFILE" "$EVALUATION_ID" \
    "${SAMPLES[*]}" "$KEY_RESULT" "$ERROR_TEXT" \
    >"$temporary_path"; then
    rm -f -- "$temporary_path"
    return 1
  fi
  if ! mv -f -- "$temporary_path" "$receipt_path"; then
    rm -f -- "$temporary_path"
    return 1
  fi
}

mark_receipt_failure() {
  STATUS="failed"
  if [[ "$ERROR_TEXT" != *"terminal status receipt"* ]]; then
    ERROR_TEXT="${ERROR_TEXT:+$ERROR_TEXT; }could not atomically write terminal status receipt"
  fi
}

handle_termination() {
  local signal="$1"
  local exit_code="$2"
  TERMINATION_SIGNAL="$signal"
  TERMINATION_EXIT_CODE="$exit_code"
  STATUS="cancelled"
  ERROR_TEXT="pipeline received SIG${signal}"
  trap '' TERM INT
  if [[ "$FINISHING" -eq 1 ]]; then
    trap - ERR EXIT
    set +e
    local elapsed="$((SECONDS - START_SECONDS))"
    if ! write_status_receipt "$exit_code" "$elapsed"; then
      mark_receipt_failure
      exit_code=74
      write_status_receipt "$exit_code" "$elapsed" || true
    fi
  fi
  exit "$exit_code"
}

finish() {
  trap '' TERM INT
  trap - ERR EXIT
  local rc="$1"
  FINISHING=1
  set +e
  local elapsed="$((SECONDS - START_SECONDS))"
  if [[ -n "$TERMINATION_SIGNAL" ]]; then
    STATUS="cancelled"
    if [[ "$rc" -eq 0 ]]; then
      rc="$TERMINATION_EXIT_CODE"
    fi
    if [[ -z "$ERROR_TEXT" ]]; then
      ERROR_TEXT="pipeline received SIG${TERMINATION_SIGNAL}"
    fi
  elif [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  elif [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="pipeline exited with code $rc; inspect shared logs"
  fi
  # Slurm may send SIGKILL shortly after TERM. Persist a fail-closed receipt
  # before best-effort notification so cancellation cannot be mistaken for a
  # successful run even if the remaining epilogue is interrupted.
  if [[ "$STATUS" != "success" ]] && ! write_status_receipt "$rc" "$elapsed"; then
    mark_receipt_failure
    rc=74
    write_status_receipt "$rc" "$elapsed" || true
  fi
  QQ_SUB="Qwen3-VL verifier ${EVALUATION_PROFILE} ${EXPERIMENT_ARM}" \
  QQ_BODY="status=${STATUS}; arm=${EXPERIMENT_ARM}; protocol=${PROTOCOL_VERSION}; key_result=${KEY_RESULT}; error=${ERROR_TEXT:-none}; job=${JOB_ID}; elapsed=${elapsed}s; output=${PARENT}" \
    "$QQ_NOTIFIER" >"/tmp/qwen3vl_base_e2e5_${JOB_ID}.qq.log" 2>&1 || true
  rm -f -- "/tmp/qwen3vl_base_e2e5_${JOB_ID}.qq.log"
  # A same-directory rename makes the terminal receipt atomic for readers.
  # Successful runs publish only after the worker epilogue reaches this point.
  if ! write_status_receipt "$rc" "$elapsed"; then
    mark_receipt_failure
    rc=74
    write_status_receipt "$rc" "$elapsed" || true
  fi
  exit "$rc"
}
trap record_error ERR
trap 'finish "$?"' EXIT
trap 'handle_termination TERM 143' TERM
trap 'handle_termination INT 130' INT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run this worker only through a Slurm GPU allocation"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is empty"
[[ "$EXPECTED_WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || die "expected worker SHA-256 is malformed"
ACTUAL_WORKER_SHA256="$(sha256sum -- "${BASH_SOURCE[0]}" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "executed Slurm worker differs from the pinned harness worker"
[[ ! -e "$RUN_DIR" ]] || die "run output already exists: $RUN_DIR"
ACTUAL_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
HARNESS_COMMIT="$(git -C "$HARNESS_ROOT" rev-parse HEAD)"
if [[ -n "$EXPECTED_COMMIT" && "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  die "repository commit drifted: expected $EXPECTED_COMMIT, found $ACTUAL_COMMIT"
fi
if [[ -n "$EXPECTED_HARNESS_COMMIT" && "$HARNESS_COMMIT" != "$EXPECTED_HARNESS_COMMIT" ]]; then
  die "harness commit drifted: expected $EXPECTED_HARNESS_COMMIT, found $HARNESS_COMMIT"
fi
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || die "repository has uncommitted changes: $ROOT"
[[ -z "$(git -C "$HARNESS_ROOT" status --porcelain --untracked-files=all)" ]] \
  || die "harness has uncommitted changes: $HARNESS_ROOT"
[[ "$(realpath -e -- "$MODEL_PATH")" == "$(realpath -e -- "$EXPECTED_MODEL_PATH")" ]] \
  || die "base diagnostic must use the pinned model path: $EXPECTED_MODEL_PATH"
[[ "$(realpath -e -- "$INPUT_ROOT")" == "$(realpath -e -- "$EXPECTED_INPUT_ROOT")" ]] \
  || die "base diagnostic must use the pinned input root: $EXPECTED_INPUT_ROOT"
for executable in "$PYTHON" "$SEGAGENT_PYTHON" "$OMNIOVCD_PYTHON" "$QQ_NOTIFIER"; do
  [[ -x "$executable" ]] || die "required executable is missing: $executable"
done
for required_directory in \
  "$WORKSPACE/Chang-Agent/third_party/OmniOVCD" \
  "$WORKSPACE/Chang-Agent/third_party/SegAgent" \
  "$WORKSPACE/Chang-Agent/third_party/SegAgent/third_party/SimpleClick"; do
  [[ -d "$required_directory" ]] \
    || die "required workspace tool source is missing: $required_directory"
done
for required_file in \
  "$ROOT/tools/run_levir_change_agent.py" \
  "$SUMMARIZER" \
  "$ANALYZER" \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/model.safetensors" \
  "$MODEL_PATH/preprocessor_config.json" \
  "$SIMPLECLICK_CHECKPOINT" \
  "$SAM3_CHECKPOINT" \
  "$SAM3_BPE"; do
  [[ -f "$required_file" ]] || die "required file is missing: $required_file"
done
for sample in "${SAMPLES[@]}"; do
  for directory in A B label_cvt; do
    [[ -f "$INPUT_ROOT/$directory/$sample.png" ]] \
      || die "fixed evaluation input is missing: $directory/$sample.png"
  done
done

cd "$ROOT"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONHASHSEED=42
export PYTHONPATH="$ROOT:$WORKSPACE/Chang-Agent/third_party/OmniOVCD:$WORKSPACE/Chang-Agent/third_party/SegAgent:$WORKSPACE/Chang-Agent/third_party/SegAgent/third_party/SimpleClick"
export CHANGE_AGENT_STAGED_PROTOCOL_VERSION="$PROTOCOL_VERSION"

"$PYTHON" - "$ROOT" <<'PY'
import pathlib
import sys

import change_agent

root = pathlib.Path(sys.argv[1]).resolve()
package = pathlib.Path(change_agent.__file__).resolve()
try:
    package.relative_to(root / "change_agent")
except ValueError as error:
    raise SystemExit(
        f"change_agent import escaped arm repository: {package} not under {root}"
    ) from error
print(f"change_agent_import={package}")
PY

"$PYTHON" - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys
import torch

model = pathlib.Path(sys.argv[1])
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
if config.get("model_type") != "qwen3_vl":
    raise SystemExit(f"unexpected model_type: {config.get('model_type')!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one CUDA GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU does not report BF16 support")
device_name = torch.cuda.get_device_name(0)
if "A800" not in device_name:
    raise SystemExit(f"expected an A800 for the pinned baseline, found {device_name!r}")
print(f"gpu_contract=ok device={device_name}")
PY

"$PYTHON" - <<'PY' >"$PARENT/environment_versions.txt"
import importlib.metadata as md
import platform
import torch
import change_agent

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"change_agent={change_agent.__file__}")
for package in ("transformers", "accelerate", "numpy", "pillow"):
    print(f"{package}={md.version(package)}")
PY

COMMAND=(
  "$PYTHON" "$ROOT/tools/run_levir_change_agent.py"
  --input-root "$INPUT_ROOT"
  --samples "${SAMPLES[@]}"
  --output "$RUN_DIR"
  --query building
  --max-steps 3
  --selection-policy conservative_best
  --proposal-mode proposal
  --staged-verifier-max-total-regions 32
  --verifier-max-selected-regions 3
  --verifier-region-padding-ratio 0.25
  --empty-mask-scan-grid-size 3
  --matching-mode overlap_presence
  --overlap-threshold 0.25
  --t12-min-instance-area 0
  --cd-min-instance-area 0
  --authorized-negative-point-input-scope full_session
  --max-point-proposal-change-ratio 0.25
  --max-point-proposal-change-pixels 256
  --max-point-image-change-ratio 0.004
  --model-path "$MODEL_PATH"
  --agent-backend local
  --verifier qwen_staged
  --staged-verifier-backend local
  --verifier-max-new-tokens 1536
  --verifier-accept-threshold 0.82
  --verifier-retries 2
  --verifier-repetition-penalty 1.05
  --device-map auto
  --tool-device cuda
  --seed 42
  --action-retries 3
  --segagent-python "$SEGAGENT_PYTHON"
  --omniovcd-python "$OMNIOVCD_PYTHON"
  --simpleclick-checkpoint "$SIMPLECLICK_CHECKPOINT"
  --sam3-checkpoint "$SAM3_CHECKPOINT"
  --sam3-bpe "$SAM3_BPE"
  --sam3-resolution 1008
)
printf 'launch=%q ' "${COMMAND[@]}" >"$PARENT/launch_command.txt"
printf '\n' >>"$PARENT/launch_command.txt"
"${COMMAND[@]}"

[[ -f "$RUN_DIR/rollout_complete.json" ]] || die "rollout completion marker is missing"
[[ -f "$RUN_DIR/per_sample_metrics.json" ]] || die "per-sample metrics are missing"
[[ ! -e "$RUN_DIR/.run_incomplete" ]] || die "run retained an incomplete marker"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$ACTUAL_COMMIT" ]] \
  || die "repository commit changed during rollout"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || die "repository gained tracked modifications during rollout"
[[ "$(git -C "$HARNESS_ROOT" rev-parse HEAD)" == "$HARNESS_COMMIT" ]] \
  || die "harness commit changed during rollout"
[[ -z "$(git -C "$HARNESS_ROOT" status --porcelain --untracked-files=all)" ]] \
  || die "harness gained tracked modifications during rollout"

# Hash GT only after the runner has written its GT-free completion marker and
# finished offline scoring. Opening label_cvt before that point would violate
# the rollout boundary even if the bytes were used only for identity checks.
"$PYTHON" - "$PARENT/input_manifest.json" "$ROOT" "$HARNESS_ROOT" \
  "$INPUT_ROOT" "$MODEL_PATH" "$SIMPLECLICK_CHECKPOINT" "$SAM3_CHECKPOINT" \
  "$SAM3_BPE" "$EXPERIMENT_ARM" "$PROTOCOL_VERSION" "$ACTUAL_COMMIT" \
  "$ACTUAL_WORKER_SHA256" "$EVALUATION_PROFILE" "$EVALUATION_ID" \
  "${SAMPLES[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

output = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2]).resolve()
harness = pathlib.Path(sys.argv[3]).resolve()
root = pathlib.Path(sys.argv[4]).resolve()
model = pathlib.Path(sys.argv[5]).resolve()
workspace = model.parents[1]
simpleclick = pathlib.Path(sys.argv[6]).resolve()
sam3 = pathlib.Path(sys.argv[7]).resolve()
bpe = pathlib.Path(sys.argv[8]).resolve()
experiment_arm = sys.argv[9]
protocol_version = sys.argv[10]
actual_commit = sys.argv[11]
actual_worker_sha256 = sys.argv[12]
evaluation_profile = sys.argv[13]
evaluation_id = sys.argv[14]
samples = sys.argv[15:]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def git_head(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

files = {}
for sample in samples:
    files[sample] = {
        directory: sha256(root / directory / f"{sample}.png")
        for directory in ("A", "B", "label_cvt")
    }
payload = {
    "evaluation_profile": evaluation_profile,
    "evaluation_id": evaluation_id,
    "experiment_arm": experiment_arm,
    "protocol_version": protocol_version,
    "sample_order": samples,
    "input_root": str(root),
    "files_sha256": files,
    "gt_hashed_after_rollout": True,
    "slurm_node": os.environ.get("SLURMD_NODENAME"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "model_path": str(model),
    "model_files_sha256": {
        path.name: sha256(path)
        for path in sorted(model.iterdir())
        if path.is_file()
    },
    "tool_files_sha256": {
        "simpleclick_checkpoint": sha256(simpleclick),
        "sam3_checkpoint": sha256(sam3),
        "sam3_bpe": sha256(bpe),
        "harness_worker": sha256(
            harness / "tools/run_qwen3vl_base_e2e5.slurm"
        ),
        "segmentation_worker": sha256(repo / "tools/segmentation_worker.py"),
        "seeded_segmentation_worker": sha256(repo / "tools/seeded_segmentation_worker.py"),
    },
    "runtime_source_sha256": {
        relative: sha256(repo / relative)
        for relative in (
            "change_agent/adapters/stage_backends.py",
            "change_agent/adapters/staged_verifier.py",
            "change_agent/verifier_protocol.py",
            "tools/run_levir_change_agent.py",
        )
    },
    "runtime_commits": {
        "chang_agent": git_head(repo),
        "expected_chang_agent": actual_commit,
        "harness": git_head(harness),
        "expected_harness": os.environ.get("EXPECTED_HARNESS_COMMIT"),
        "omniovcd": git_head(workspace / "Chang-Agent/third_party/OmniOVCD"),
    },
    "executed_worker_sha256": actual_worker_sha256,
}
environment_path = output.parent / "environment_versions.txt"
if environment_path.is_file():
    environment = {}
    for line in environment_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            environment[key] = value
    payload["runtime_environment"] = {
        key: value for key, value in environment.items() if key != "change_agent"
    }
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

KEY_RESULT="$(
  "$PYTHON" "$SUMMARIZER" \
    --run-dir "$RUN_DIR" \
    --output "$PARENT/baseline_summary.json" \
    --evaluation-profile "$EVALUATION_PROFILE" \
    --evaluation-id "$EVALUATION_ID" \
    --model-state base \
    --job-id "$JOB_ID" \
    --experiment-arm "$EXPERIMENT_ARM" \
    --protocol-version "$PROTOCOL_VERSION" \
    --runtime-commit "$ACTUAL_COMMIT" \
    --harness-commit "$HARNESS_COMMIT" \
    --expected-samples "${SAMPLES[@]}"
)"
echo "$KEY_RESULT"
PROTOCOL_RESULT="$(
  "$PYTHON" "$ANALYZER" \
    --run-dir "$RUN_DIR" \
    --output "$PARENT/protocol_summary.json" \
    --evaluation-profile "$EVALUATION_PROFILE" \
    --evaluation-id "$EVALUATION_ID" \
    --experiment-arm "$EXPERIMENT_ARM" \
    --protocol-version "$PROTOCOL_VERSION" \
    --job-id "$JOB_ID" \
    --runtime-commit "$ACTUAL_COMMIT" \
    --harness-commit "$HARNESS_COMMIT" \
    --expected-samples "${SAMPLES[@]}"
)"
echo "$PROTOCOL_RESULT"
KEY_RESULT="${KEY_RESULT};${PROTOCOL_RESULT}"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$ACTUAL_COMMIT" && -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "arm repository drifted during postprocessing"
[[ "$(git -C "$HARNESS_ROOT" rev-parse HEAD)" == "$HARNESS_COMMIT" && -z "$(git -C "$HARNESS_ROOT" status --porcelain --untracked-files=all)" ]] || die "harness drifted during postprocessing"
STATUS="success"
