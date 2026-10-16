#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(cd "$ROOT/.." && pwd -P)"
ARTIFACT_ROOT="$(realpath -e -- "$WORKSPACE/Chang-Agent")"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
INITIAL_OUTPUT="$(realpath -m -- "$OUTPUT_ROOT/CA_0802(1)-qwen3vl-p0-contract-validation")"
PRIOR_OUTPUT="$(realpath -m -- "$OUTPUT_ROOT/CA_0802(1R1)-qwen3vl-p0-contract-validation")"
OUTPUT="$(realpath -m -- "$OUTPUT_ROOT/CA_0802(1R2)-qwen3vl-p0-contract-validation")"
WORKER="$ROOT/tools/run_qwen3vl_p0_validation.slurm"
PYTHON="${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}"
QQ_NOTIFIER="${QQ_NOTIFIER:-$WORKSPACE/.local/bin/wangyihan-send-qq}"
PARTITION="${PARTITION:-gre}"
TARGETED_NAME="CA0802P0TargetR2"
REGRESSION_NAME="CA0802P0RegressR2"
EXPECTED_COMMIT_INPUT="${EXPECTED_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
ARCHIVE_RUN_DIR="$ARTIFACT_ROOT/outputs/CA_0801_archive/CA_0801(10)-qwen3vl-v121-post-sft-e2e3-v1/run"
AMENDMENT="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P0_VALIDATION_AMENDMENT_R2_0802.json"
INPUT_MANIFEST="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json"
WORKBOOK="$ROOT/experiments/experiment_registry.xlsx"
R2_AMENDMENT_CANONICAL_SHA256="<hash omitted>"
INPUT_MANIFEST_CANONICAL_SHA256="<hash omitted>"
WORKBOOK_SHA256="<hash omitted>"

fail() {
  echo "qwen3vl_p0_submit: $*" >&2
  exit 2
}

verify_attempt_file() {
  local attempt_root="$1"
  local attempt_id="$2"
  local relative_path="$3"
  local expected_sha256="$4"
  local source="$attempt_root/$relative_path"
  local observed_sha256

  [[ -f "$source" ]] || fail "$attempt_id P0 evidence is missing: $source"
  observed_sha256="$(sha256sum -- "$source" | awk '{print $1}')"
  [[ "$observed_sha256" == "$expected_sha256" ]] \
    || fail "$attempt_id P0 evidence changed: $relative_path"
}

verify_canonical_json_file() {
  local source="$1"
  local expected_sha256="$2"
  local label="$3"

  "$PYTHON" - "$source" "$expected_sha256" "$label" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
label = sys.argv[3]
payload = json.loads(source.read_text(encoding="utf-8"))
serialized = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
observed = hashlib.sha256(serialized).hexdigest()
if observed != expected:
    raise SystemExit(
        f"qwen3vl_p0_submit: {label} canonical SHA256 changed: {observed}"
    )
PY
}

write_submission_receipt() {
  local submission_status="$1"
  local targeted_job="$2"
  local regression_job="$3"
  local failure_stage="$4"
  local failure_code="$5"
  local receipt="$OUTPUT/submission.txt"
  local temporary="${receipt}.tmp-${BASHPID}"

  if ! printf 'submission_status=%s\nexpected_commit=%s\nworker=%s\nworker_sha256=%s\npartition=%s\nresources=targeted:cpu=2,mem=8G,time=00:20:00;regression:cpu=2,mem=8G,time=00:45:00;gpu=0\nretry_of_output=%s\nfailed_attempt_outputs=%s,%s\nretry_reason=canonical_sample_order_contract_conflict\ntargeted_job_name=%s\ntargeted_job_id=%s\nregression_job_name=%s\nregression_job_id=%s\ndependency=%s\nfailure_stage=%s\nfailure_code=%s\noutput=%s\n' \
    "$submission_status" "$EXPECTED_COMMIT" "$WORKER" "$WORKER_SHA256" \
    "$PARTITION" "$PRIOR_OUTPUT" "$INITIAL_OUTPUT" "$PRIOR_OUTPUT" \
    "$TARGETED_NAME" "$targeted_job" "$REGRESSION_NAME" "$regression_job" \
    "afterany:${targeted_job}" "$failure_stage" \
    "$failure_code" "$OUTPUT" >"$temporary"; then
    rm -f -- "$temporary"
    return 1
  fi
  if ! mv -f -- "$temporary" "$receipt"; then
    rm -f -- "$temporary"
    return 1
  fi
}

[[ "$PARTITION" == "gre" ]] || fail "P0 CPU validation partition must remain pinned to gre"
[[ "$OUTPUT" == "$OUTPUT_ROOT/CA_0802(1R2)-qwen3vl-p0-contract-validation" ]] \
  || fail "amended P0 retry output path drifted"
[[ ! -e "$OUTPUT" ]] || fail "reserved output already exists: $OUTPUT"
[[ -d "$INITIAL_OUTPUT" ]] || fail "initial failed P0 output is missing: $INITIAL_OUTPUT"
[[ -d "$PRIOR_OUTPUT" ]] || fail "prior failed P0 output is missing: $PRIOR_OUTPUT"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "submission.txt" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "evidence/targeted_result.json" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "evidence/regression_result.json" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "logs/status-52553.txt" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "logs/status-52554.txt" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "logs/slurm-52553.err" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "logs/slurm-52553.out" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "logs/slurm-52554.err" \
  "<hash omitted>"
verify_attempt_file "$INITIAL_OUTPUT" "initial" \
  "logs/slurm-52554.out" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "submission.txt" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "evidence/targeted_result.json" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "evidence/regression_result.json" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "evidence/v1_archive_sha256_before.json" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "evidence/v1_archive_sha256_after.json" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "logs/status-52562.txt" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "logs/status-52563.txt" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "logs/slurm-52562.err" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "logs/slurm-52562.out" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "logs/slurm-52563.err" \
  "<hash omitted>"
verify_attempt_file "$PRIOR_OUTPUT" "R1" \
  "logs/slurm-52563.out" \
  "<hash omitted>"
[[ -e "$ROOT/outputs" ]] \
  || fail "run worktree must provide an ignored outputs symlink to $OUTPUT_ROOT"
[[ "$(realpath -e -- "$ROOT/outputs")" == "$OUTPUT_ROOT" ]] \
  || fail "run worktree outputs path does not resolve to the fixed artifact root"
if [[ "$ROOT" != "$ARTIFACT_ROOT" ]]; then
  [[ -L "$ROOT/outputs" ]] \
    || fail "separate run worktree must expose outputs through a symlink"
  git -C "$ROOT" check-ignore -q -- outputs \
    || fail "run worktree outputs symlink must be ignored by Git"
fi

[[ "$EXPECTED_COMMIT_INPUT" =~ ^[0-9a-f]{40}$ ]] \
  || fail "EXPECTED_COMMIT must be one exact 40-character lowercase Git commit"
if ! EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse --verify "${EXPECTED_COMMIT_INPUT}^{commit}" 2>/dev/null)"; then
  fail "EXPECTED_COMMIT does not resolve to a commit in $ROOT"
fi
[[ "$EXPECTED_COMMIT" == "$EXPECTED_COMMIT_INPUT" ]] \
  || fail "EXPECTED_COMMIT did not remain exact after resolution"
ACTUAL_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] \
  || fail "repository commit mismatch: expected $EXPECTED_COMMIT, found $ACTUAL_COMMIT"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail "repository must be clean before P0 submission: $ROOT"

for executable in "$PYTHON" "$WORKER" "$QQ_NOTIFIER"; do
  [[ -x "$executable" ]] || fail "required executable is missing: $executable"
done
for required_file in \
  "$ROOT/experiments/build_experiment_workbook.py" \
  "$ROOT/tools/summarize_qwen3vl_e2e_run.py" \
  "$ROOT/tests/test_qwen3vl_p0_contracts.py" \
  "$ROOT/tests/test_experiment_registry.py" \
  "$ROOT/tests/test_local_verifier_analysis.py" \
  "$AMENDMENT" \
  "$INPUT_MANIFEST" \
  "$WORKBOOK" \
  "$ARCHIVE_RUN_DIR/rollout_complete.json" \
  "$ARCHIVE_RUN_DIR/per_sample_metrics.json" \
  "$ARCHIVE_RUN_DIR/trajectories/test_20_15/episode_summary.json" \
  "$ARCHIVE_RUN_DIR/trajectories/test_20_15/trajectory.json" \
  "$ARCHIVE_RUN_DIR/trajectories/test_1_1/episode_summary.json" \
  "$ARCHIVE_RUN_DIR/trajectories/test_1_1/trajectory.json" \
  "$ARCHIVE_RUN_DIR/trajectories/test_50_8/episode_summary.json" \
  "$ARCHIVE_RUN_DIR/trajectories/test_50_8/trajectory.json"; do
  [[ -f "$required_file" ]] || fail "required P0 input is missing: $required_file"
done
verify_canonical_json_file \
  "$AMENDMENT" "$R2_AMENDMENT_CANONICAL_SHA256" "R2 amendment"
verify_canonical_json_file \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_CANONICAL_SHA256" "input manifest"
verify_attempt_file \
  "$ROOT" "frozen workbook" "experiments/experiment_registry.xlsx" "$WORKBOOK_SHA256"

WORKER_SHA256="$(sha256sum -- "$WORKER" | awk '{print $1}')"
[[ "$WORKER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "cannot compute the P0 worker SHA-256"

if ! SLURM_PING="$(scontrol ping 2>&1)" || ! grep -q 'UP' <<<"$SLURM_PING"; then
  echo "qwen3vl_p0_submit: Slurm controller is not UP; no job submitted" >&2
  printf '%s\n' "$SLURM_PING" >&2
  exit 3
fi
for job_name in "$TARGETED_NAME" "$REGRESSION_NAME"; do
  if ! ACTIVE="$(squeue -h -u guisongxia --name="$job_name" -o '%i %T' 2>&1)"; then
    echo "qwen3vl_p0_submit: cannot inspect duplicate jobs for $job_name" >&2
    printf '%s\n' "$ACTIVE" >&2
    exit 3
  fi
  [[ -z "$ACTIVE" ]] || fail "active duplicate job exists for $job_name: $ACTIVE"
done

mkdir -- "$OUTPUT"
mkdir -- "$OUTPUT/logs" "$OUTPUT/evidence"

# Export an explicitly empty CUDA visibility value.  The worker also rejects
# any non-empty CUDA or Slurm GPU-allocation metadata.
export CUDA_VISIBLE_DEVICES=""
COMMON_EXPORT="ALL,REPO_ROOT=$ROOT,WORKSPACE_ROOT=$WORKSPACE,ARTIFACT_ROOT=$ARTIFACT_ROOT,PYTHON=$PYTHON,QQ_NOTIFIER=$QQ_NOTIFIER,EXPECTED_COMMIT=$EXPECTED_COMMIT,EXPECTED_WORKER_SHA256=$WORKER_SHA256"

set +e
TARGETED_OUTPUT="$(
  sbatch --parsable \
    --job-name="$TARGETED_NAME" \
    --partition="$PARTITION" \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:20:00 \
    --output="$OUTPUT/logs/slurm-%j.out" \
    --error="$OUTPUT/logs/slurm-%j.err" \
    --export="$COMMON_EXPORT" \
    "$WORKER" targeted "$OUTPUT" 2>&1
)"
TARGETED_RC="$?"
set -e
if [[ "$TARGETED_RC" -ne 0 ]]; then
  write_submission_receipt \
    "targeted_submit_failed" "not_submitted" "not_submitted" \
    "targeted_sbatch" "$TARGETED_RC" || {
      echo "qwen3vl_p0_submit: could not atomically write the targeted submission failure receipt" >&2
      exit 74
    }
  echo "qwen3vl_p0_submit: targeted sbatch failed; output reservation retained for audit" >&2
  printf '%s\n' "$TARGETED_OUTPUT" >&2
  exit "$TARGETED_RC"
fi
TARGETED_JOB="${TARGETED_OUTPUT%%;*}"
if [[ ! "$TARGETED_JOB" =~ ^[0-9]+$ ]]; then
  write_submission_receipt \
    "targeted_submit_uncertain" "unknown" "not_submitted" \
    "targeted_job_id_parse" "3" || {
      echo "qwen3vl_p0_submit: could not atomically write the uncertain targeted submission receipt" >&2
      exit 74
    }
  echo "qwen3vl_p0_submit: targeted sbatch returned an unexpected job id; output reservation retained" >&2
  printf '%s\n' "$TARGETED_OUTPUT" >&2
  exit 3
fi

set +e
REGRESSION_OUTPUT="$(
  sbatch --parsable \
    --dependency="afterany:$TARGETED_JOB" \
    --job-name="$REGRESSION_NAME" \
    --partition="$PARTITION" \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:45:00 \
    --output="$OUTPUT/logs/slurm-%j.out" \
    --error="$OUTPUT/logs/slurm-%j.err" \
    --export="$COMMON_EXPORT" \
    "$WORKER" regression "$OUTPUT" 2>&1
)"
REGRESSION_RC="$?"
set -e
if [[ "$REGRESSION_RC" -ne 0 ]]; then
  write_submission_receipt \
    "regression_submit_failed" "$TARGETED_JOB" "not_submitted" \
    "regression_sbatch" "$REGRESSION_RC" || {
      echo "qwen3vl_p0_submit: could not atomically write the regression submission failure receipt" >&2
      exit 74
    }
  echo "qwen3vl_p0_submit: regression sbatch failed; targeted job remains submitted" >&2
  printf '%s\n' "$REGRESSION_OUTPUT" >&2
  exit "$REGRESSION_RC"
fi
REGRESSION_JOB="${REGRESSION_OUTPUT%%;*}"
if [[ ! "$REGRESSION_JOB" =~ ^[0-9]+$ ]]; then
  write_submission_receipt \
    "regression_submit_uncertain" "$TARGETED_JOB" "unknown" \
    "regression_job_id_parse" "3" || {
      echo "qwen3vl_p0_submit: could not atomically write the uncertain regression submission receipt" >&2
      exit 74
    }
  echo "qwen3vl_p0_submit: regression sbatch returned an unexpected job id; submitted state is uncertain" >&2
  printf '%s\n' "$REGRESSION_OUTPUT" >&2
  exit 3
fi

write_submission_receipt \
  "submitted" "$TARGETED_JOB" "$REGRESSION_JOB" "none" "0" || {
    echo "qwen3vl_p0_submit: both jobs were submitted but the submission receipt could not be written" >&2
    exit 74
  }
printf 'output=%s\nexpected_commit=%s\ntargeted_job=%s\nregression_job=%s\ndependency=afterany:%s\n' \
  "$OUTPUT" "$EXPECTED_COMMIT" "$TARGETED_JOB" "$REGRESSION_JOB" "$TARGETED_JOB"
