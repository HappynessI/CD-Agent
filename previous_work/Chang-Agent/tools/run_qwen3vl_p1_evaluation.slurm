#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu48
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

EXECUTED_WORKER_COPY="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}")"
ARTIFACT_ROOT="$(realpath -e -- "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}")"
OUTPUT="$(realpath -e -- "${1:?usage: worker OUTPUT PROFILE ARM [H0_RECEIPT]}")"
EVALUATION_PROFILE="${2:?evaluation profile is required}"
ARM="${3:?arm is required}"
H0_RECEIPT_OVERRIDE="${4:-}"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
LOG_DIR="$(realpath -e -- "$OUTPUT/logs")"
RUN_DIR="$OUTPUT/run"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p1_evaluation.slurm"
CONFIG="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
AMENDMENT="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json"
INPUT_MANIFEST="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json"
PYTHON="$(realpath -e -- "${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}")"
SEGAGENT_PYTHON="$(realpath -e -- "${SEGAGENT_PYTHON:-$WORKSPACE/segagent-env/bin/python}")"
OMNIOVCD_PYTHON="$(realpath -e -- "${OMNIOVCD_PYTHON:-$WORKSPACE/omniovcd-env/bin/python}")"
MODEL_PATH="$(realpath -e -- "${MODEL_PATH:-$WORKSPACE/models/Qwen3-VL-2B-Instruct}")"
INPUT_ROOT="$(realpath -e -- "${INPUT_ROOT:-$WORKSPACE/datasets/LEVIR-CD/test_256}")"
SIMPLECLICK_CHECKPOINT="$(realpath -e -- "${SIMPLECLICK_CHECKPOINT:-$WORKSPACE/models/SimpleClick/cocolvis_vit_large.pth}")"
SAM3_CHECKPOINT="$(realpath -e -- "${SAM3_CHECKPOINT:-$WORKSPACE/models/sam3/sam3.pt}")"
SAM3_BPE="$(realpath -e -- "${SAM3_BPE:-$WORKSPACE/Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz}")"
SUMMARIZER="${SUMMARIZER:-$ROOT/tools/summarize_qwen3vl_e2e_run.py}"
ANALYZER="${ANALYZER:-$ROOT/tools/analyze_staged_verifier_run.py}"
SUMMARY_VALIDATOR="${SUMMARY_VALIDATOR:-$ROOT/tools/validate_qwen3vl_e2e_summary.py}"
RECEIPT_BUILDER="${RECEIPT_BUILDER:-$ROOT/tools/build_qwen3vl_evaluation_receipt.py}"
RECEIPT_VALIDATOR="${RECEIPT_VALIDATOR:-$ROOT/tools/validate_qwen3vl_evaluation_receipt.py}"
RUNTIME_MANIFEST_BUILDER="${RUNTIME_MANIFEST_BUILDER:-$ROOT/tools/build_qwen3vl_runtime_input_manifest.py}"
PAIR_BUILDER="${PAIR_BUILDER:-$ROOT/tools/build_qwen3vl_p1_pair_comparison.py}"
PAIR_VALIDATOR="${PAIR_VALIDATOR:-$ROOT/tools/validate_qwen3vl_p1_pair_comparison.py}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
EXPECTED_IMPLEMENTATION_COMMIT="${EXPECTED_IMPLEMENTATION_COMMIT:?EXPECTED_IMPLEMENTATION_COMMIT is required}"
EXPECTED_WORKER_SHA256="${EXPECTED_WORKER_SHA256:?EXPECTED_WORKER_SHA256 is required}"
EXPECTED_CONFIG_SHA256="${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}"
EXPECTED_AMENDMENT_CANONICAL_SHA256="${EXPECTED_AMENDMENT_CANONICAL_SHA256:?EXPECTED_AMENDMENT_CANONICAL_SHA256 is required}"
EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256="${EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256:?EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256 is required}"
JOB_ID="${SLURM_JOB_ID:-missing}"
JOB_NAME="${SLURM_JOB_NAME:-missing}"
START_SECONDS="$SECONDS"
STATUS="failed"
STATUS_PUBLISHED=0
KEY_RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
TERMINATION_SIGNAL=""
TERMINATION_EXIT_CODE=""

case "$EVALUATION_PROFILE:$ARM" in
  p1_smoke1:h0)
    ALLOCATION_ID="CA_0802(3)"
    EVALUATION_ID="DEV-SMOKE-1-H1-V1"
    SAMPLES=(test_1_1)
    EXPERIMENT_ARM="v12_1-h0-legacy-single-round-base-0802"
    HARNESS_PROTOCOL_ID="qwen3vl_harness_h0_legacy_single_round_v1"
    EXPECTED_JOB_NAME="CA0802P1SmokeH0"
    QQ_TITLE="CA0802 P1 H0 base smoke1"
    PAIR_REQUIRED=0
    H0_RECEIPT=""
    TIME_LIMIT="01:00:00"
    ;;
  p1_smoke1:h1)
    ALLOCATION_ID="CA_0802(4)"
    EVALUATION_ID="DEV-SMOKE-1-H1-V1"
    SAMPLES=(test_1_1)
    EXPERIMENT_ARM="v12_1-h1-gtfree-terminal-review-base-0802"
    HARNESS_PROTOCOL_ID="qwen3vl_harness_h1_gtfree_terminal_review_v1"
    EXPECTED_JOB_NAME="CA0802P1SmokeH1"
    QQ_TITLE="CA0802 P1 H1 base smoke1"
    PAIR_REQUIRED=1
    H0_RECEIPT="${H0_RECEIPT_OVERRIDE:?paired H0 receipt override is required}"
    TIME_LIMIT="01:00:00"
    ;;
  e2e3:h0)
    ALLOCATION_ID="CA_0802(5)"
    EVALUATION_ID="DEV-CORE-3-V1"
    SAMPLES=(test_20_15 test_1_1 test_50_8)
    EXPERIMENT_ARM="v12_1-h0-legacy-single-round-base-0802"
    HARNESS_PROTOCOL_ID="qwen3vl_harness_h0_legacy_single_round_v1"
    EXPECTED_JOB_NAME="CA0802P1CoreH0"
    QQ_TITLE="CA0802 P1 H0 base CORE3"
    PAIR_REQUIRED=0
    H0_RECEIPT=""
    TIME_LIMIT="03:00:00"
    ;;
  e2e3:h1)
    ALLOCATION_ID="CA_0802(6)"
    EVALUATION_ID="DEV-CORE-3-V1"
    SAMPLES=(test_20_15 test_1_1 test_50_8)
    EXPERIMENT_ARM="v12_1-h1-gtfree-terminal-review-base-0802"
    HARNESS_PROTOCOL_ID="qwen3vl_harness_h1_gtfree_terminal_review_v1"
    EXPECTED_JOB_NAME="CA0802P1CoreH1"
    QQ_TITLE="CA0802 P1 H1 base CORE3"
    PAIR_REQUIRED=1
    H0_RECEIPT="${H0_RECEIPT_OVERRIDE:?paired H0 receipt override is required}"
    TIME_LIMIT="03:00:00"
    ;;
  *)
    echo "qwen3vl_p1_evaluation: unsupported profile/arm $EVALUATION_PROFILE/$ARM" >&2
    exit 2
    ;;
esac
if [[ "$PAIR_REQUIRED" -eq 1 ]]; then
  H0_RECEIPT="$(realpath -e -- "$H0_RECEIPT")"
fi

STATUS_RECEIPT="$LOG_DIR/status-${JOB_ID}.txt"
SUMMARY_VALIDATION_RECEIPT="$OUTPUT/summary_validation.txt"
RECEIPT_VALIDATION_RECEIPT="$OUTPUT/receipt_validation.txt"
EVALUATION_RECEIPT="$OUTPUT/evaluation_receipt.json"
PAIR_RECEIPT="$OUTPUT/pair_comparison_receipt.json"

die() {
  ERROR_TEXT="$*"
  echo "qwen3vl_p1_evaluation: $*" >&2
  return 1
}

record_error() {
  local rc="$?"
  if [[ -z "$ERROR_TEXT" ]]; then
    ERROR_TEXT="command_failed_at_line_${BASH_LINENO[0]:-unknown}"
  fi
  return "$rc"
}

sanitize() {
  local value="$1"
  value="${value//$'\n'/ }"
  value="${value//$'\r'/ }"
  printf '%s' "$value"
}

write_status() {
  local rc="$1"
  local elapsed="$2"
  local temporary="${STATUS_RECEIPT}.tmp-${BASHPID}"
  local safe_error
  safe_error="$(sanitize "$ERROR_TEXT")"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\nallocation_id=%s\nevaluation_profile=%s\nevaluation_id=%s\nsamples=%s\narm=%s\nexperiment_arm_id=%s\nharness_protocol_id=%s\ncommit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\npath=%s\n' \
    "$STATUS" "$rc" "$elapsed" "$JOB_ID" "$JOB_NAME" "$ALLOCATION_ID" \
    "$EVALUATION_PROFILE" "$EVALUATION_ID" "${SAMPLES[*]}" "$ARM" \
    "$EXPERIMENT_ARM" "$HARNESS_PROTOCOL_ID" "$EXPECTED_COMMIT" \
    "$ACTUAL_WORKER_SHA256" "$KEY_RESULT" "$safe_error" "$OUTPUT" \
    >"$temporary" || return 1
  mv -f -- "$temporary" "$STATUS_RECEIPT"
}

handle_signal() {
  local signal="$1"
  local exit_code="$2"
  TERMINATION_SIGNAL="$signal"
  TERMINATION_EXIT_CODE="$exit_code"
  STATUS="cancelled"
  ERROR_TEXT="pipeline received SIG${signal}"
  trap '' TERM INT
  exit "$exit_code"
}

finish() {
  local rc="$1"
  trap '' TERM INT
  trap - ERR EXIT
  set +e
  local elapsed="$((SECONDS - START_SECONDS))"
  if [[ -n "$TERMINATION_SIGNAL" ]]; then
    STATUS="cancelled"
    [[ "$rc" -eq 0 ]] && rc="$TERMINATION_EXIT_CODE"
  elif [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  else
    # A failure after the provisional success publication must invalidate the
    # hash-bound status (and therefore any complete receipt that referenced it).
    STATUS="failed"
    STATUS_PUBLISHED=0
    if [[ -z "$ERROR_TEXT" ]]; then
      ERROR_TEXT="pipeline exited with code $rc; inspect shared logs"
    fi
  fi
  # A successful status is already hash-bound by the receipt builder.  Never
  # rewrite it in the successful EXIT trap, because changing elapsed_seconds
  # would invalidate an otherwise complete receipt.  Any later failure clears
  # the publication bit and deliberately invalidates that receipt.
  if [[ "$STATUS_PUBLISHED" -ne 1 || "$STATUS" != "success" ]]; then
    if ! write_status "$rc" "$elapsed"; then
      STATUS="failed"
      ERROR_TEXT="${ERROR_TEXT:+$ERROR_TEXT; }could not atomically write terminal status"
      rc=74
      write_status "$rc" "$elapsed" || true
    fi
  fi
  local qq_log="/tmp/qwen3vl_p1_${JOB_ID}.qq.log"
  QQ_SUB="$QQ_TITLE" \
  QQ_BODY="job_name=${JOB_NAME}; job_id=${JOB_ID}; terminal_status=${STATUS}; elapsed_seconds=${elapsed}; key_result_or_error=$(sanitize "${ERROR_TEXT:-$KEY_RESULT}"); artifact_path=${OUTPUT}" \
    "$QQ_NOTIFIER" >"$qq_log" 2>&1 || true
  rm -f -- "$qq_log"
  exit "$rc"
}

trap record_error ERR
trap 'finish "$?"' EXIT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal INT 130' INT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run this worker only through Slurm"
[[ "$JOB_NAME" == "$EXPECTED_JOB_NAME" ]] || die "unexpected Job name: $JOB_NAME"
[[ "${SLURM_JOB_PARTITION:-}" == "gre" ]] || die "partition must be gre"
[[ "${SLURMD_NODENAME:-}" == "gpu48" ]] || die "GPU worker must run on gpu48"
[[ "${SLURM_CPUS_PER_TASK:-}" == "4" ]] || die "GPU worker must receive four CPUs"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]] \
  || die "CUDA_VISIBLE_DEVICES is empty"
[[ -x "$PYTHON" && -x "$SEGAGENT_PYTHON" && -x "$OMNIOVCD_PYTHON" ]] \
  || die "required Python executable is missing"
[[ -x "$QQ_NOTIFIER" ]] || die "QQ notifier is missing"
for required in \
  "$PINNED_WORKER" "$CONFIG" "$AMENDMENT" "$INPUT_MANIFEST" \
  "$SUMMARIZER" "$ANALYZER" "$SUMMARY_VALIDATOR" "$RECEIPT_BUILDER" \
  "$RECEIPT_VALIDATOR" "$RUNTIME_MANIFEST_BUILDER" "$PAIR_BUILDER" \
  "$PAIR_VALIDATOR" \
  "$OUTPUT/submission.txt" "$MODEL_PATH/config.json" "$SIMPLECLICK_CHECKPOINT" \
  "$SAM3_CHECKPOINT" "$SAM3_BPE"; do
  [[ -f "$required" ]] || die "required P1 input is missing: $required"
done
[[ "$EXPECTED_WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "worker SHA-256 is malformed"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER_COPY" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "executed worker differs from the submitter-pinned bytes"
[[ "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "repository worker differs from pinned bytes"
[[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] \
  || die "resolved run config hash drifted"
[[ "$ARM" == h0 || "$ARM" == h1 ]] || die "arm must be h0 or h1"
[[ ! -e "$RUN_DIR" ]] || die "run child already exists: $RUN_DIR"
[[ ! -e "$EVALUATION_RECEIPT" && ! -e "$PAIR_RECEIPT" ]] || die "evaluation receipt already exists"

"$PYTHON" - "$CONFIG" "$AMENDMENT" "$EXPECTED_AMENDMENT_CANONICAL_SHA256" \
  "$EXPECTED_COMMIT" "$EXPECTED_IMPLEMENTATION_COMMIT" \
  "$ROOT" "$WORKSPACE" "$ARTIFACT_ROOT" "$OUTPUT_ROOT" "$OUTPUT" \
  "$INPUT_ROOT" "$MODEL_PATH" "$SIMPLECLICK_CHECKPOINT" "$SAM3_CHECKPOINT" "$SAM3_BPE" \
  "$PYTHON" "$OMNIOVCD_PYTHON" "$SEGAGENT_PYTHON" "$INPUT_MANIFEST" \
  "$ALLOCATION_ID" "$EVALUATION_PROFILE" "$ARM" "$JOB_NAME" "$H0_RECEIPT" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    config_text, amendment_text, expected, freeze_commit, implementation_commit,
    root, workspace, artifact_root, output_root, output, input_root, model,
    simpleclick, sam3, sam3_bpe, python, omniovcd_python, segagent_python,
    input_manifest, allocation_id, profile, arm, job_name, h0_receipt,
) = sys.argv[1:]

def resolved(path_text, *, strict=True):
    return pathlib.Path(path_text).resolve(strict=strict)

config = json.loads(pathlib.Path(config_text).read_text(encoding="utf-8"))
path = pathlib.Path(amendment_text)
payload = json.loads(path.read_text(encoding="utf-8"))
serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
if config.get("configuration_status") != "frozen":
    raise SystemExit("resolved config is not frozen")
if config.get("execution_readiness") != "execution_ready":
    raise SystemExit("resolved config is not execution-ready")
if config.get("p1_execution_authorized") is not True:
    raise SystemExit("resolved config does not authorize P1")
paths = config.get("paths")
if not isinstance(paths, dict):
    raise SystemExit("resolved config paths are missing")
for key, observed in (
    ("workspace_root", workspace),
    ("clean_run_worktree", root),
    ("artifact_repository", artifact_root),
    ("output_root", output_root),
    ("input_root", input_root),
    ("base_model", model),
    ("simpleclick_checkpoint", simpleclick),
    ("sam3_checkpoint", sam3),
    ("sam3_bpe", sam3_bpe),
    ("omniovcd_python", omniovcd_python),
    ("segagent_python", segagent_python),
):
    declared = paths.get(key)
    if not isinstance(declared, str) or resolved(declared) != resolved(observed):
        raise SystemExit(f"resolved config paths.{key} differs from the evaluation worker")
configured_omniovcd = resolved(paths["omniovcd_python"])
if resolved(python) != configured_omniovcd or resolved(sys.executable) != configured_omniovcd:
    raise SystemExit("evaluation worker PYTHON is not the resolved config omniovcd_python")
declared_input_manifest = paths.get("evaluation_input_manifest")
if (
    not isinstance(declared_input_manifest, str)
    or (resolved(root) / declared_input_manifest).resolve() != resolved(input_manifest)
):
    raise SystemExit("evaluation input manifest path differs from the resolved config")
allocations = [
    item for item in config.get("allocations", [])
    if isinstance(item, dict) and item.get("allocation_id") == allocation_id
]
if len(allocations) != 1:
    raise SystemExit("resolved config allocation identity is missing or duplicated")
allocation = allocations[0]
if (
    allocation.get("evaluation_profile") != profile
    or allocation.get("arm") != arm
    or allocation.get("job_name") != job_name
    or not isinstance(allocation.get("output_path"), str)
    or resolved(allocation["output_path"]) != resolved(output)
):
    raise SystemExit("evaluation worker differs from its resolved allocation")
paired_id = allocation.get("paired_h0_allocation")
if arm == "h1":
    paired = [
        item for item in config.get("allocations", [])
        if isinstance(item, dict) and item.get("allocation_id") == paired_id
    ]
    if len(paired) != 1 or not isinstance(paired[0].get("output_path"), str):
        raise SystemExit("resolved paired H0 allocation is missing")
    expected_h0_receipt = resolved(
        pathlib.Path(paired[0]["output_path"]) / "evaluation_receipt.json"
    )
    if resolved(h0_receipt) != expected_h0_receipt:
        raise SystemExit("paired H0 receipt path differs from the resolved config")
elif h0_receipt:
    raise SystemExit("H0 evaluation must not receive a paired receipt override")
if hashlib.sha256(serialized).hexdigest() != expected:
    raise SystemExit("P1 amendment canonical hash drifted")
if payload.get("record_status") != "frozen" or payload.get("execution_readiness") != "execution_ready":
    raise SystemExit("P1 amendment is not frozen and execution-ready")
if payload.get("p1_execution_authorized") is not True:
    raise SystemExit("P1 execution is not authorized")
if payload.get("implementation_commit") != implementation_commit:
    raise SystemExit("P1 implementation commit binding drifted")
worktree = pathlib.Path(payload.get("clean_run_worktree", "")).resolve()
if worktree != pathlib.Path(root).resolve():
    raise SystemExit("P1 clean worktree binding drifted")
PY

[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit drifted"
[[ "$(git -C "$ROOT" rev-parse "${EXPECTED_COMMIT}^")" == "$EXPECTED_IMPLEMENTATION_COMMIT" ]] \
  || die "freeze commit is not the direct child of the pinned implementation commit"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "run worktree is dirty"

for sample in "${SAMPLES[@]}"; do
  [[ -f "$INPUT_ROOT/A/$sample.png" && -f "$INPUT_ROOT/B/$sample.png" ]] \
    || die "online A/B input is missing for $sample"
done

RUNTIME_DEVICE="$("$PYTHON" - "$MODEL_PATH" <<'PY'
import json
import pathlib
import sys
import torch

model = pathlib.Path(sys.argv[1])
config = json.loads((model / "config.json").read_text(encoding="utf-8"))
if config.get("model_type") != "qwen3_vl":
    raise SystemExit(f"unexpected model_type: {config.get('model_type')!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU does not report BF16 support")
name = torch.cuda.get_device_name(0)
if "A800" not in name:
    raise SystemExit(f"expected an A800 device, found {name!r}")
print(name)
PY
)"
echo "gpu_contract=ok;bf16=true;device_model=${RUNTIME_DEVICE}"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=42
export PYTHONPATH="$ROOT:$WORKSPACE/Chang-Agent/third_party/OmniOVCD:$WORKSPACE/Chang-Agent/third_party/SegAgent:$WORKSPACE/Chang-Agent/third_party/SegAgent/third_party/SimpleClick"
export CHANGE_AGENT_STAGED_PROTOCOL_VERSION="v12.1"
cd "$ROOT"

KEY_RESULT="writing_pre_rollout_runtime_input_manifest"
RUNTIME_TOOL_ARGS=(
  --tool-file "simpleclick_checkpoint=$SIMPLECLICK_CHECKPOINT"
  --tool-file "sam3_checkpoint=$SAM3_CHECKPOINT"
  --tool-file "sam3_bpe=$SAM3_BPE"
  --tool-file "runner=$ROOT/tools/run_levir_change_agent.py"
  --tool-file "summarizer=$SUMMARIZER"
  --tool-file "analyzer=$ANALYZER"
  --tool-file "summary_validator=$SUMMARY_VALIDATOR"
  --tool-file "receipt_builder=$RECEIPT_BUILDER"
  --tool-file "receipt_validator=$RECEIPT_VALIDATOR"
  --tool-file "pair_builder=$PAIR_BUILDER"
  --tool-file "pair_validator=$PAIR_VALIDATOR"
  --tool-file "runtime_manifest_builder=$RUNTIME_MANIFEST_BUILDER"
  --tool-file "harness_worker=$PINNED_WORKER"
  --tool-file "terminal_review_scheduler=$ROOT/change_agent/terminal_review.py"
  --tool-file "terminal_review_verifier=$ROOT/change_agent/adapters/terminal_review_verifier.py"
  --tool-file "staged_verifier=$ROOT/change_agent/adapters/staged_verifier.py"
  --tool-file "stage_backends=$ROOT/change_agent/adapters/stage_backends.py"
  --tool-file "environment=$ROOT/change_agent/environment.py"
  --tool-file "verifier_regions=$ROOT/change_agent/verifier_regions.py"
  --tool-file "h0_golden_fixture_validator=$ROOT/tools/validate_qwen3vl_h0_golden_fixture.py"
  --tool-file "submitter=$ROOT/tools/submit_qwen3vl_p1.sh"
  --runtime-source "runner=tools/run_levir_change_agent.py"
  --runtime-source "environment=change_agent/environment.py"
  --runtime-source "staged_verifier=change_agent/adapters/staged_verifier.py"
  --runtime-source "terminal_review=change_agent/terminal_review.py"
  --runtime-source "terminal_review_verifier=change_agent/adapters/terminal_review_verifier.py"
  --runtime-source "verifier_protocol=change_agent/verifier_protocol.py"
)
"$PYTHON" "$RUNTIME_MANIFEST_BUILDER" \
  --phase pre_rollout \
  --output "$OUTPUT/input_manifest.json" \
  --run-dir "$RUN_DIR" \
  --input-root "$INPUT_ROOT" \
  --model-path "$MODEL_PATH" \
  --evaluation-input-manifest "$INPUT_MANIFEST" \
  --evaluation-input-manifest-canonical-sha256 "$EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256" \
  --resolved-run-config "$CONFIG" \
  --resolved-run-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --evaluation-profile "$EVALUATION_PROFILE" \
  --evaluation-id "$EVALUATION_ID" \
  --arm "$ARM" \
  --experiment-arm-id "$EXPERIMENT_ARM" \
  --harness-protocol-id "$HARNESS_PROTOCOL_ID" \
  --model-io-protocol v12.1 \
  --runtime-commit "$EXPECTED_COMMIT" \
  --worker-sha256 "$ACTUAL_WORKER_SHA256" \
  --job-id "$JOB_ID" \
  --runtime-device "$RUNTIME_DEVICE" \
  --samples "${SAMPLES[@]}" \
  "${RUNTIME_TOOL_ARGS[@]}"

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
  --terminal-review-arm "$ARM"
)
printf 'launch=%q ' "${COMMAND[@]}" >"$OUTPUT/launch_command.txt"
printf '\n' >>"$OUTPUT/launch_command.txt"
KEY_RESULT="running_rollout"
"${COMMAND[@]}"

[[ -f "$RUN_DIR/rollout_complete.json" ]] || die "rollout completion marker is missing"
[[ -f "$RUN_DIR/per_sample_metrics.json" ]] || die "per-sample metrics are missing"
[[ -f "$RUN_DIR/p1_harness_audit.json" ]] || die "P1 harness audit is missing"
[[ ! -e "$RUN_DIR/.run_incomplete" ]] || die "run retained incomplete marker"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit changed during rollout"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "worktree changed during rollout"

KEY_RESULT="writing_post_rollout_runtime_input_manifest"
"$PYTHON" "$RUNTIME_MANIFEST_BUILDER" \
  --phase post_rollout \
  --output "$OUTPUT/input_manifest.json" \
  --run-dir "$RUN_DIR" \
  --input-root "$INPUT_ROOT" \
  --model-path "$MODEL_PATH" \
  --evaluation-input-manifest "$INPUT_MANIFEST" \
  --evaluation-input-manifest-canonical-sha256 "$EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256" \
  --resolved-run-config "$CONFIG" \
  --resolved-run-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --evaluation-profile "$EVALUATION_PROFILE" \
  --evaluation-id "$EVALUATION_ID" \
  --arm "$ARM" \
  --experiment-arm-id "$EXPERIMENT_ARM" \
  --harness-protocol-id "$HARNESS_PROTOCOL_ID" \
  --model-io-protocol v12.1 \
  --runtime-commit "$EXPECTED_COMMIT" \
  --worker-sha256 "$ACTUAL_WORKER_SHA256" \
  --job-id "$JOB_ID" \
  --runtime-device "$RUNTIME_DEVICE" \
  --samples "${SAMPLES[@]}" \
  "${RUNTIME_TOOL_ARGS[@]}"

KEY_RESULT="building_summary"
"$PYTHON" "$SUMMARIZER" \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT/baseline_summary.json" \
  --evaluation-profile "$EVALUATION_PROFILE" \
  --evaluation-id "$EVALUATION_ID" \
  --model-state base \
  --job-id "$JOB_ID" \
  --experiment-arm "$EXPERIMENT_ARM" \
  --protocol-version v12.1 \
  --harness-protocol-id "$HARNESS_PROTOCOL_ID" \
  --runtime-commit "$EXPECTED_COMMIT" \
  --harness-commit "$EXPECTED_COMMIT" \
  --p1-harness-audit "$RUN_DIR/p1_harness_audit.json" \
  --expected-samples "${SAMPLES[@]}"

KEY_RESULT="building_protocol_summary"
"$PYTHON" "$ANALYZER" \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT/protocol_summary.json" \
  --evaluation-profile "$EVALUATION_PROFILE" \
  --evaluation-id "$EVALUATION_ID" \
  --experiment-arm "$EXPERIMENT_ARM" \
  --protocol-version v12.1 \
  --harness-protocol-id "$HARNESS_PROTOCOL_ID" \
  --job-id "$JOB_ID" \
  --runtime-commit "$EXPECTED_COMMIT" \
  --harness-commit "$EXPECTED_COMMIT" \
  --expected-samples "${SAMPLES[@]}"

KEY_RESULT="validating_summary"
SUMMARY_VALIDATION_OUTPUT="$(
  "$PYTHON" "$SUMMARY_VALIDATOR" \
    --summary "$OUTPUT/baseline_summary.json" \
    --run-dir "$RUN_DIR" \
    --evaluation-profile "$EVALUATION_PROFILE" \
    --evaluation-id "$EVALUATION_ID" \
    --expected-samples "${SAMPLES[@]}" \
    --experiment-arm "$EXPERIMENT_ARM" \
    --harness-protocol-id "$HARNESS_PROTOCOL_ID" \
    --runtime-commit "$EXPECTED_COMMIT" \
    --harness-commit "$EXPECTED_COMMIT" \
    --job-id "$JOB_ID"
 )"
[[ "$SUMMARY_VALIDATION_OUTPUT" == status=valid\;* ]] || die "summary validator did not return status=valid"
printf '%s\n' "$SUMMARY_VALIDATION_OUTPUT" >"$SUMMARY_VALIDATION_RECEIPT"

[[ ! -e "$EVALUATION_RECEIPT" ]] || die "evaluation receipt already exists"
KEY_RESULT="publishing_success_status_before_receipt"
STATUS="success"
KEY_RESULT="all_rollout_and_summary_gates_passed"
elapsed_before_receipt="$((SECONDS - START_SECONDS))"
write_status 0 "$elapsed_before_receipt" || die "could not publish success status before receipt"
STATUS_PUBLISHED=1

KEY_RESULT="building_evaluation_receipt"
"$PYTHON" "$RECEIPT_BUILDER" \
  --output "$EVALUATION_RECEIPT" \
  --output-parent "$OUTPUT" \
  --run-dir "$RUN_DIR" \
  --evaluation-profile "$EVALUATION_PROFILE" \
  --evaluation-id "$EVALUATION_ID" \
  --expected-samples "${SAMPLES[@]}" \
  --arm "$ARM" \
  --experiment-arm "$EXPERIMENT_ARM" \
  --harness-protocol-id "$HARNESS_PROTOCOL_ID" \
  --runtime-commit "$EXPECTED_COMMIT" \
  --harness-commit "$EXPECTED_COMMIT" \
  --implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT" \
  --job-id "$JOB_ID" \
  --run-worktree "$ROOT" \
  --model-path "$MODEL_PATH" \
  --worker-sha256 "$ACTUAL_WORKER_SHA256" \
  --evaluation-input-manifest "$INPUT_MANIFEST" \
  --evaluation-input-manifest-canonical-sha256 "$EXPECTED_INPUT_MANIFEST_CANONICAL_SHA256" \
  --runtime-input-manifest "$OUTPUT/input_manifest.json" \
  --resolved-run-config "$CONFIG" \
  --resolved-run-config-sha256 "$EXPECTED_CONFIG_SHA256" \
  --run-manifest "$RUN_DIR/run_manifest.md" \
  --launch-command "$OUTPUT/launch_command.txt" \
  --submission-receipt "$OUTPUT/submission.txt" \
  --status-receipt "$STATUS_RECEIPT" \
  --summary-validation "$SUMMARY_VALIDATION_RECEIPT"

KEY_RESULT="validating_evaluation_receipt"
RECEIPT_VALIDATION_OUTPUT="$(
  "$PYTHON" "$RECEIPT_VALIDATOR" --receipt "$EVALUATION_RECEIPT"
)"
printf '%s\n' "$RECEIPT_VALIDATION_OUTPUT" >"$RECEIPT_VALIDATION_RECEIPT"

if [[ "$PAIR_REQUIRED" -eq 1 ]]; then
  [[ -f "$H0_RECEIPT" ]] || die "paired H0 evaluation receipt is missing: $H0_RECEIPT"
  [[ ! -e "$PAIR_RECEIPT" ]] || die "pair comparison receipt already exists"
  KEY_RESULT="building_h0_h1_pair_comparison"
  "$PYTHON" "$PAIR_BUILDER" \
    --h0-receipt "$H0_RECEIPT" \
    --h1-receipt "$EVALUATION_RECEIPT" \
    --evaluation-profile "$EVALUATION_PROFILE" \
    --receipt-validator "$RECEIPT_VALIDATOR" \
    --output "$PAIR_RECEIPT"
  [[ -f "$PAIR_RECEIPT" ]] || die "pair comparison receipt is missing"
  "$PYTHON" "$PAIR_VALIDATOR" \
    --pair "$PAIR_RECEIPT" \
    --h0-receipt "$H0_RECEIPT" \
    --h1-receipt "$EVALUATION_RECEIPT" \
    --evaluation-profile "$EVALUATION_PROFILE" \
    --receipt-validator "$RECEIPT_VALIDATOR" \
    >"$OUTPUT/pair_validation.txt"
  KEY_RESULT="pair_hard_gates_passed"
fi

[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit changed during finalization"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "worktree changed during finalization"
exit 0
