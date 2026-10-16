#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

set -euo pipefail
umask 027
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SCONTROL_BIN="${SCONTROL_BIN:-$(command -v scontrol || true)}"

EXECUTED_WORKER_COPY="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "${REPO_ROOT:?REPO_ROOT is required}")"
WORKSPACE="$(realpath -e -- "${WORKSPACE_ROOT:?WORKSPACE_ROOT is required}")"
ARTIFACT_ROOT="$(realpath -e -- "${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}")"
OUTPUT_ROOT="$(realpath -e -- "$ARTIFACT_ROOT/outputs")"
PINNED_WORKER="$ROOT/tools/run_qwen3vl_p1_dependency_guard.slurm"
CONFIG="$ROOT/doc/training_pipeline/0802_third_try/QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
PYTHON="$(realpath -e -- "${PYTHON:-$WORKSPACE/omniovcd-env/bin/python}")"
mapfile -t OUTPUT_NAMES < <(
  "$PYTHON" - "$CONFIG" "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
output_root = pathlib.Path(sys.argv[2]).resolve()
allocations = config.get("allocations")
if not isinstance(allocations, list) or len(allocations) != 5:
    raise SystemExit("dependency guard requires five resolved allocations")
for item in allocations:
    output = pathlib.Path(item.get("output_path", ""))
    if not output.is_absolute() or output.resolve().parent != output_root:
        raise SystemExit("dependency guard allocation output escapes output root")
    print(output.name)
PY
)
[[ "${#OUTPUT_NAMES[@]}" -eq 5 ]] || {
  echo "qwen3vl_p1_dependency_guard: config did not resolve five outputs" >&2
  exit 2
}
CPU_OUTPUT="$(realpath -e -- "$OUTPUT_ROOT/${OUTPUT_NAMES[0]}")"
LOG_DIR="$(realpath -e -- "$CPU_OUTPUT/logs")"
SACCT_BIN="${SACCT_BIN:-$(command -v sacct || true)}"
QQ_NOTIFIER="${QQ_NOTIFIER:-/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
EXPECTED_IMPLEMENTATION_COMMIT="${EXPECTED_IMPLEMENTATION_COMMIT:?EXPECTED_IMPLEMENTATION_COMMIT is required}"
EXPECTED_WORKER_SHA256="${EXPECTED_WORKER_SHA256:?EXPECTED_WORKER_SHA256 is required}"
EXPECTED_CPU_WORKER_SHA256="${EXPECTED_CPU_WORKER_SHA256:?EXPECTED_CPU_WORKER_SHA256 is required}"
EXPECTED_GPU_WORKER_SHA256="${EXPECTED_GPU_WORKER_SHA256:?EXPECTED_GPU_WORKER_SHA256 is required}"
EXPECTED_CONFIG_SHA256="${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}"
GUARD_JOB_ID="${SLURM_JOB_ID:-missing}"
GUARD_JOB_NAME="${SLURM_JOB_NAME:-missing}"
EXPECTED_GUARD_JOB_NAME="CA0802P1DependencyGuard"
GUARD_STATUS="$LOG_DIR/dependency-guard-status-${GUARD_JOB_ID}.txt"
GUARD_RECEIPT="$CPU_OUTPUT/dependency_guard_receipt.txt"
GUARD_QQ_RECEIPT="$LOG_DIR/dependency-guard-qq-attempt-${GUARD_JOB_ID}.txt"
GUARD_SUBMISSION="$CPU_OUTPUT/dependency_guard_submission.txt"
START_SECONDS="$SECONDS"
STATUS="failed"
KEY_RESULT="not_started"
ERROR_TEXT=""
ACTUAL_WORKER_SHA256="unknown"
TERMINATION_SIGNAL=""

[[ "$#" -eq 5 ]] || {
  echo "usage: $0 CPU_JOB SMOKE_H0_JOB SMOKE_H1_JOB CORE_H0_JOB CORE_H1_JOB" >&2
  exit 2
}

FORMAL_JOB_IDS=("$@")
ALLOCATION_IDS=("CA_0802(2)" "CA_0802(3)" "CA_0802(4)" "CA_0802(5)" "CA_0802(6)")
JOB_NAMES=("CA0802P1CPUFixtures" "CA0802P1SmokeH0" "CA0802P1SmokeH1" "CA0802P1CoreH0" "CA0802P1CoreH1")
EVALUATION_PROFILES=("cpu_fixtures" "p1_smoke1" "p1_smoke1" "e2e3" "e2e3")
EVALUATION_IDS=("P1-H0-H1-CPU-FIXTURES-V1" "DEV-SMOKE-1-H1-V1" "DEV-SMOKE-1-H1-V1" "DEV-CORE-3-V1" "DEV-CORE-3-V1")
SAMPLES=("cpu_fixture_cases" "test_1_1" "test_1_1" "test_20_15 test_1_1 test_50_8" "test_20_15 test_1_1 test_50_8")
ARMS=("h0_h1" "h0" "h1" "h0" "h1")
EXPERIMENT_ARMS=("not_applicable" "v12_1-h0-legacy-single-round-base-0802" "v12_1-h1-gtfree-terminal-review-base-0802" "v12_1-h0-legacy-single-round-base-0802" "v12_1-h1-gtfree-terminal-review-base-0802")
PROTOCOL_IDS=("not_applicable" "qwen3vl_harness_h0_legacy_single_round_v1" "qwen3vl_harness_h1_gtfree_terminal_review_v1" "qwen3vl_harness_h0_legacy_single_round_v1" "qwen3vl_harness_h1_gtfree_terminal_review_v1")
QQ_TITLES=("CA0802 P1 H0-H1 CPU fixtures" "CA0802 P1 H0 base smoke1" "CA0802 P1 H1 base smoke1" "CA0802 P1 H0 base CORE3" "CA0802 P1 H1 base CORE3")

declare -a ACCOUNTING_STATES=("" "" "" "" "")
declare -a ACCOUNTING_EXIT_CODES=("" "" "" "" "")
declare -a ACCOUNTING_STARTS=("" "" "" "" "")
declare -a ACCOUNTING_ENDS=("" "" "" "" "")
declare -a ACCOUNTING_REASONS=("" "" "" "" "")
declare -a DEPENDENCY_EVIDENCE_SOURCES=("" "" "" "" "")
declare -a CONTROLLER_JOB_STATES=("" "" "" "" "")
declare -a CONTROLLER_REASONS=("" "" "" "" "")
declare -a CONTROLLER_DEPENDENCIES=("" "" "" "" "")
declare -a EXISTING_STATUS_IDS=()
declare -a SYNTHESIZED_STATUS_IDS=()
declare -a UNRESOLVED_STATUS_IDS=()

die() {
  ERROR_TEXT="$*"
  echo "qwen3vl_p1_dependency_guard: $*" >&2
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

join_ids() {
  if [[ "$#" -eq 0 ]]; then
    printf '%s' "none"
    return
  fi
  local IFS=,
  printf '%s' "$*"
}

atomic_publish_absent() {
  local temporary="$1"
  local destination="$2"
  if ln -- "$temporary" "$destination" 2>/dev/null; then
    rm -f -- "$temporary"
    return 0
  fi
  rm -f -- "$temporary"
  return 1
}

attempt_qq() {
  local receipt="$1"
  local title="$2"
  local target_job_name="$3"
  local target_job_id="$4"
  local terminal_status="$5"
  local elapsed="$6"
  local key_result_or_error="$7"
  local artifact_path="$8"
  local temporary="${receipt}.tmp-${BASHPID}-${RANDOM}"
  local qq_log="/tmp/qwen3vl_p1_dependency_guard_${GUARD_JOB_ID}_${target_job_id}.qq.log"
  local notifier_rc=127
  local attempted_at
  local body
  attempted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  key_result_or_error="$(sanitize "$key_result_or_error")"
  body="job_name=${target_job_name}; job_id=${target_job_id}; terminal_status=${terminal_status}; elapsed_seconds=${elapsed}; key_result_or_error=${key_result_or_error}; artifact_path=${artifact_path}"
  if [[ -x "$QQ_NOTIFIER" ]]; then
    if QQ_SUB="$title" QQ_BODY="$body" "$QQ_NOTIFIER" >"$qq_log" 2>&1; then
      notifier_rc=0
    else
      notifier_rc="$?"
    fi
  fi
  rm -f -- "$qq_log"
  printf 'schema_id=qwen3vl_p1_qq_attempt_v1\nattempted=true\nattempted_at_utc=%s\nqq_subject=%s\njob_name=%s\njob_id=%s\nterminal_status=%s\nelapsed_seconds=%s\nkey_result_or_error=%s\nartifact_path=%s\nnotifier_exit_code=%s\ndelivery_claimed=false\nguard_job_id=%s\n' \
    "$attempted_at" "$title" "$target_job_name" "$target_job_id" \
    "$terminal_status" "$elapsed" "$key_result_or_error" "$artifact_path" \
    "$notifier_rc" "$GUARD_JOB_ID" >"$temporary" || return 1
  atomic_publish_absent "$temporary" "$receipt"
}

write_synthetic_status() {
  local index="$1"
  local output="$2"
  local status_receipt="$3"
  local predecessor_index="$((index - 1))"
  local temporary="${status_receipt}.tmp-${BASHPID}-${RANDOM}"
  local target_job_id="${FORMAL_JOB_IDS[$index]}"
  local predecessor_job_id="${FORMAL_JOB_IDS[$predecessor_index]}"
  local worker_sha256="$EXPECTED_GPU_WORKER_SHA256"
  local safe_reason safe_controller_state safe_controller_reason
  local safe_controller_dependency
  safe_reason="$(sanitize "${ACCOUNTING_REASONS[$index]}")"
  safe_controller_state="$(sanitize "${CONTROLLER_JOB_STATES[$index]}")"
  safe_controller_reason="$(sanitize "${CONTROLLER_REASONS[$index]}")"
  safe_controller_dependency="$(sanitize "${CONTROLLER_DEPENDENCIES[$index]}")"
  if [[ "$index" -eq 0 ]]; then
    worker_sha256="$EXPECTED_CPU_WORKER_SHA256"
  fi
  printf 'status=cancelled\nexit_code=125\nelapsed_seconds=0\njob_id=%s\njob_name=%s\nallocation_id=%s\nevaluation_profile=%s\nevaluation_id=%s\nsamples=%s\narm=%s\nexperiment_arm_id=%s\nharness_protocol_id=%s\ncommit=%s\nworker_sha256=%s\nresult=not_started_due_to_failed_dependency\nerror=Slurm cancelled this job before worker start because its afterok dependency was not satisfied\npath=%s\nstatus_writer=qwen3vl_p1_dependency_guard_v1\nguard_job_id=%s\npredecessor_job_id=%s\npredecessor_slurm_state=%s\nexpected_dependency=afterok:%s\ndependency_evidence_source=%s\nslurm_state=%s\nslurm_exit_code=%s\nslurm_start=%s\nslurm_end=%s\nslurm_reason=%s\ncontroller_job_state=%s\ncontroller_reason=%s\ncontroller_dependency=%s\n' \
    "$target_job_id" "${JOB_NAMES[$index]}" "${ALLOCATION_IDS[$index]}" \
    "${EVALUATION_PROFILES[$index]}" "${EVALUATION_IDS[$index]}" \
    "${SAMPLES[$index]}" "${ARMS[$index]}" "${EXPERIMENT_ARMS[$index]}" \
    "${PROTOCOL_IDS[$index]}" "$EXPECTED_COMMIT" "$worker_sha256" "$output" \
    "$GUARD_JOB_ID" "$predecessor_job_id" "${ACCOUNTING_STATES[$predecessor_index]}" \
    "$predecessor_job_id" "${DEPENDENCY_EVIDENCE_SOURCES[$index]}" \
    "${ACCOUNTING_STATES[$index]}" "${ACCOUNTING_EXIT_CODES[$index]}" \
    "${ACCOUNTING_STARTS[$index]}" "${ACCOUNTING_ENDS[$index]}" "$safe_reason" \
    "$safe_controller_state" "$safe_controller_reason" \
    "$safe_controller_dependency" \
    >"$temporary" || return 1
  atomic_publish_absent "$temporary" "$status_receipt"
}

read_accounting() {
  local index="$1"
  local job_id="${FORMAL_JOB_IDS[$index]}"
  local output
  local matches=0
  local raw_job_id state exit_code start end reason extra
  if ! output="$("$SACCT_BIN" -n -X -P -j "$job_id" --format=JobIDRaw,State,ExitCode,Start,End,Reason)"; then
    die "sacct failed for formal job $job_id"
    return 1
  fi
  while IFS='|' read -r raw_job_id state exit_code start end reason extra; do
    [[ "$raw_job_id" == "$job_id" ]] || continue
    [[ -z "${extra:-}" ]] || {
      die "sacct returned a malformed exact row for formal job $job_id"
      return 1
    }
    matches="$((matches + 1))"
    ACCOUNTING_STATES[$index]="$state"
    ACCOUNTING_EXIT_CODES[$index]="$exit_code"
    ACCOUNTING_STARTS[$index]="$start"
    ACCOUNTING_ENDS[$index]="$end"
    ACCOUNTING_REASONS[$index]="$reason"
  done <<<"$output"
  [[ "$matches" -eq 1 ]] || {
    die "sacct must return exactly one top-level row for formal job $job_id"
    return 1
  }
}

state_is_completed() {
  local state="${1%% *}"
  state="${state%+}"
  [[ "$state" == "COMPLETED" ]]
}

state_is_cancelled() {
  local state="${1%% *}"
  state="${state%+}"
  [[ "$state" == "CANCELLED" ]]
}

state_is_terminal_failure() {
  local state="${1%% *}"
  state="${state%+}"
  case "$state" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT) return 0 ;;
    *) return 1 ;;
  esac
}

start_is_missing() {
  case "$1" in
    ""|Unknown|UNKNOWN|None|NONE|N/A) return 0 ;;
    *) return 1 ;;
  esac
}

reason_is_dependency() {
  case "$1" in
    *Dependency*|*dependency*) return 0 ;;
    *) return 1 ;;
  esac
}

cancelled_job_has_dependency_evidence() {
  local index="$1"
  local job_id="${FORMAL_JOB_IDS[$index]}"
  if reason_is_dependency "${ACCOUNTING_REASONS[$index]}"; then
    DEPENDENCY_EVIDENCE_SOURCES[$index]="sacct_reason"
    return 0
  fi
  # Some SlurmDBD versions persist a dependency-cancelled top-level row as
  # CANCELLED/Start=None/Reason=None.  In that exact ambiguous case, consult
  # the live controller record and require both the scheduler reason and the
  # frozen direct predecessor edge.  We never infer dependency cancellation
  # merely from an empty accounting reason.
  case "${ACCOUNTING_REASONS[$index]}" in
    ""|Unknown|UNKNOWN|None|NONE|N/A) ;;
    *) return 1 ;;
  esac
  local predecessor_job_id="${FORMAL_JOB_IDS[$((index - 1))]}"
  local control_record
  [[ -x "$SCONTROL_BIN" ]] || return 1
  control_record="$("$SCONTROL_BIN" show job "$job_id" -o 2>/dev/null)" || return 1
  [[ -n "$control_record" && "$control_record" != *$'\n'* ]] || return 1

  local token key value
  local parsed_job_id="" parsed_state="" parsed_reason="" parsed_dependency=""
  local job_id_count=0 state_count=0 reason_count=0 dependency_count=0
  local -a control_fields=()
  read -r -a control_fields <<<"$control_record"
  for token in "${control_fields[@]}"; do
    [[ "$token" == *=* ]] || continue
    key="${token%%=*}"
    value="${token#*=}"
    case "$key" in
      JobId)
        job_id_count="$((job_id_count + 1))"
        parsed_job_id="$value"
        ;;
      JobState)
        state_count="$((state_count + 1))"
        parsed_state="$value"
        ;;
      Reason)
        reason_count="$((reason_count + 1))"
        parsed_reason="$value"
        ;;
      Dependency)
        dependency_count="$((dependency_count + 1))"
        parsed_dependency="$value"
        ;;
    esac
  done
  [[ "$job_id_count" -eq 1 && "$state_count" -eq 1 \
    && "$reason_count" -eq 1 && "$dependency_count" -eq 1 ]] || return 1
  [[ "$parsed_job_id" == "$job_id" ]] || return 1
  [[ "$parsed_state" == "CANCELLED" ]] || return 1
  [[ "$parsed_reason" == "DependencyNeverSatisfied" ]] || return 1
  [[ "$parsed_dependency" =~ ^afterok:${predecessor_job_id}\([^,\(\)[:space:]]+\)$ ]] \
    || return 1

  DEPENDENCY_EVIDENCE_SOURCES[$index]="scontrol_exact"
  CONTROLLER_JOB_STATES[$index]="$parsed_state"
  CONTROLLER_REASONS[$index]="$parsed_reason"
  CONTROLLER_DEPENDENCIES[$index]="$parsed_dependency"
  return 0
}

validate_submission_binding() {
  local index="$1"
  local output="$2"
  local receipt="$output/submission.txt"
  local expected_worker_sha256="$EXPECTED_GPU_WORKER_SHA256"
  local expected_dependency="none"
  [[ "$index" -eq 0 ]] && expected_worker_sha256="$EXPECTED_CPU_WORKER_SHA256"
  if [[ "$index" -gt 0 ]]; then
    expected_dependency="afterok:${FORMAL_JOB_IDS[$((index - 1))]}"
  fi
  [[ -f "$receipt" ]] || return 1
  grep -Fqx 'submission_status=submitted' "$receipt" \
    && grep -Fqx "job_id=${FORMAL_JOB_IDS[$index]}" "$receipt" \
    && grep -Fqx "job_name=${JOB_NAMES[$index]}" "$receipt" \
    && grep -Fqx "allocation_id=${ALLOCATION_IDS[$index]}" "$receipt" \
    && grep -Fqx "worker_sha256=${expected_worker_sha256}" "$receipt" \
    && grep -Fqx "dependency=${expected_dependency}" "$receipt"
}

write_guard_artifacts() {
  local rc="$1"
  local elapsed="$2"
  local status_text="$3"
  local temporary_status="${GUARD_STATUS}.tmp-${BASHPID}"
  local temporary_receipt="${GUARD_RECEIPT}.tmp-${BASHPID}"
  local formal_ids existing_ids synthesized_ids unresolved_ids safe_error
  formal_ids="$(join_ids "${FORMAL_JOB_IDS[@]}")"
  existing_ids="$(join_ids "${EXISTING_STATUS_IDS[@]}")"
  synthesized_ids="$(join_ids "${SYNTHESIZED_STATUS_IDS[@]}")"
  unresolved_ids="$(join_ids "${UNRESOLVED_STATUS_IDS[@]}")"
  safe_error="$(sanitize "$ERROR_TEXT")"
  printf 'status=%s\nexit_code=%s\nelapsed_seconds=%s\njob_id=%s\njob_name=%s\ncommit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\npath=%s\n' \
    "$status_text" "$rc" "$elapsed" "$GUARD_JOB_ID" "$GUARD_JOB_NAME" \
    "$EXPECTED_COMMIT" "$ACTUAL_WORKER_SHA256" "$KEY_RESULT" "$safe_error" \
    "$CPU_OUTPUT" >"$temporary_status" || return 1
  mv -f -- "$temporary_status" "$GUARD_STATUS" || return 1
  printf 'schema_id=qwen3vl_p1_dependency_guard_receipt_v1\nstatus=%s\nguard_job_id=%s\nguard_job_name=%s\nformal_job_ids=%s\nexisting_worker_status_job_ids=%s\nsynthesized_dependency_status_job_ids=%s\nunresolved_missing_status_job_ids=%s\ncommit=%s\nworker_sha256=%s\nresult=%s\nerror=%s\n' \
    "$status_text" "$GUARD_JOB_ID" "$GUARD_JOB_NAME" "$formal_ids" \
    "$existing_ids" "$synthesized_ids" "$unresolved_ids" "$EXPECTED_COMMIT" \
    "$ACTUAL_WORKER_SHA256" "$KEY_RESULT" "$safe_error" >"$temporary_receipt" \
    || return 1
  mv -f -- "$temporary_receipt" "$GUARD_RECEIPT"
}

handle_signal() {
  local signal="$1"
  local exit_code="$2"
  TERMINATION_SIGNAL="$signal"
  STATUS="cancelled"
  ERROR_TEXT="dependency guard received SIG${signal}"
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
  elif [[ "$rc" -eq 0 ]]; then
    STATUS="success"
  else
    STATUS="failed"
    [[ -n "$ERROR_TEXT" ]] || ERROR_TEXT="dependency guard exited with code $rc"
  fi
  if ! write_guard_artifacts "$rc" "$elapsed" "$STATUS"; then
    STATUS="failed"
    ERROR_TEXT="${ERROR_TEXT:+$ERROR_TEXT; }could not atomically publish dependency guard artifacts"
    rc=74
    write_guard_artifacts "$rc" "$elapsed" "$STATUS" || true
  fi
  attempt_qq "$GUARD_QQ_RECEIPT" "CA0802 P1 dependency guard" \
    "$GUARD_JOB_NAME" "$GUARD_JOB_ID" "$STATUS" "$elapsed" \
    "${ERROR_TEXT:-$KEY_RESULT}" "$CPU_OUTPUT" || true
  exit "$rc"
}

trap record_error ERR
trap 'finish "$?"' EXIT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal INT 130' INT

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run this worker only through Slurm"
[[ "$GUARD_JOB_NAME" == "$EXPECTED_GUARD_JOB_NAME" ]] || die "unexpected guard Job name: $GUARD_JOB_NAME"
[[ "${SLURM_JOB_PARTITION:-}" == "gre" ]] || die "dependency guard partition must be gre"
[[ "${SLURM_CPUS_PER_TASK:-}" == "1" ]] || die "dependency guard must receive one CPU"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) die "dependency guard unexpectedly received a visible GPU" ;;
esac
[[ -x "$SACCT_BIN" ]] || die "sacct executable is missing"
for hash_value in "$EXPECTED_WORKER_SHA256" "$EXPECTED_CPU_WORKER_SHA256" \
  "$EXPECTED_GPU_WORKER_SHA256" "$EXPECTED_CONFIG_SHA256"; do
  [[ "$hash_value" =~ ^[0-9a-f]{64}$ ]] || die "expected SHA-256 is malformed"
done
for job_id in "${FORMAL_JOB_IDS[@]}"; do
  [[ "$job_id" =~ ^[0-9]+$ ]] || die "formal Job ID is malformed: $job_id"
done
[[ "$(printf '%s\n' "${FORMAL_JOB_IDS[@]}" | sort -u | wc -l)" -eq 5 ]] \
  || die "formal Job IDs must be unique"
[[ -f "$PINNED_WORKER" && -f "$CONFIG" && -f "$GUARD_SUBMISSION" ]] \
  || die "required dependency guard input is missing"
ACTUAL_WORKER_SHA256="$(sha256sum -- "$EXECUTED_WORKER_COPY" | awk '{print $1}')"
[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "executed dependency guard differs from the pinned hash"
[[ "$(sha256sum -- "$PINNED_WORKER" | awk '{print $1}')" == "$EXPECTED_WORKER_SHA256" ]] \
  || die "repository dependency guard hash drifted"
[[ "$(sha256sum -- "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] \
  || die "resolved run config byte hash drifted"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit drifted"
[[ "$(git -C "$ROOT" rev-parse "${EXPECTED_COMMIT}^")" == "$EXPECTED_IMPLEMENTATION_COMMIT" ]] \
  || die "freeze commit is not the direct child of the pinned implementation commit"
"$PYTHON" - "$CONFIG" "$ROOT" "$WORKSPACE" "$ARTIFACT_ROOT" "$OUTPUT_ROOT" \
  "$CPU_OUTPUT" "${OUTPUT_NAMES[@]}" <<'PY'
import json
import pathlib
import sys

(
    config_text, root_text, workspace_text, artifact_text, output_root_text,
    cpu_output_text, *output_names,
) = sys.argv[1:]

def resolved(path_text, *, strict=True):
    return pathlib.Path(path_text).resolve(strict=strict)

config = json.loads(pathlib.Path(config_text).read_text(encoding="utf-8"))
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
    ("workspace_root", workspace_text),
    ("clean_run_worktree", root_text),
    ("artifact_repository", artifact_text),
    ("output_root", output_root_text),
    ("omniovcd_python", sys.executable),
):
    declared = paths.get(key)
    if not isinstance(declared, str) or resolved(declared) != resolved(observed):
        raise SystemExit(f"resolved config paths.{key} differs from the dependency guard")
if resolved(sys.executable) != resolved(paths["omniovcd_python"]):
    raise SystemExit("dependency guard PYTHON executable identity drifted")
allocations = config.get("allocations")
if not isinstance(allocations, list) or len(allocations) != len(output_names):
    raise SystemExit("resolved config formal allocations differ from the dependency guard")
for item, output_name in zip(allocations, output_names, strict=True):
    declared_output = item.get("output_path") if isinstance(item, dict) else None
    observed_output = resolved(pathlib.Path(output_root_text) / output_name)
    if not isinstance(declared_output, str) or resolved(declared_output) != observed_output:
        raise SystemExit("dependency guard output differs from the resolved allocation")
if resolved(cpu_output_text) != resolved(allocations[0]["output_path"]):
    raise SystemExit("dependency guard CPU output identity drifted")
PY
grep -Fqx 'submission_status=submitted' "$GUARD_SUBMISSION" \
  && grep -Fqx "job_id=${GUARD_JOB_ID}" "$GUARD_SUBMISSION" \
  && grep -Fqx "worker_sha256=${EXPECTED_WORKER_SHA256}" "$GUARD_SUBMISSION" \
  || die "dependency guard submission receipt is not bound to this worker"

for index in "${!FORMAL_JOB_IDS[@]}"; do
  output="$(realpath -e -- "$OUTPUT_ROOT/${OUTPUT_NAMES[$index]}")"
  validate_submission_binding "$index" "$output" \
    || die "formal submission receipt binding failed for ${FORMAL_JOB_IDS[$index]}"
  read_accounting "$index"
done

KEY_RESULT="auditing_downstream_worker_statuses"
for index in 1 2 3 4; do
  target_job_id="${FORMAL_JOB_IDS[$index]}"
  predecessor_index="$((index - 1))"
  output="$(realpath -e -- "$OUTPUT_ROOT/${OUTPUT_NAMES[$index]}")"
  log_dir="$(realpath -e -- "$output/logs")"
  status_receipt="$log_dir/status-${target_job_id}.txt"
  qq_receipt="$log_dir/qq-attempt-${target_job_id}.txt"
  if [[ -f "$status_receipt" ]]; then
    EXISTING_STATUS_IDS+=("$target_job_id")
    continue
  fi
  if state_is_cancelled "${ACCOUNTING_STATES[$index]}" \
    && start_is_missing "${ACCOUNTING_STARTS[$index]}" \
    && cancelled_job_has_dependency_evidence "$index" \
    && state_is_terminal_failure "${ACCOUNTING_STATES[$predecessor_index]}"; then
    if write_synthetic_status "$index" "$output" "$status_receipt"; then
      SYNTHESIZED_STATUS_IDS+=("$target_job_id")
      attempt_qq "$qq_receipt" "${QQ_TITLES[$index]}" "${JOB_NAMES[$index]}" \
        "$target_job_id" "cancelled" "0" \
        "not_started_due_to_failed_dependency; predecessor_job_id=${FORMAL_JOB_IDS[$predecessor_index]}; predecessor_state=${ACCOUNTING_STATES[$predecessor_index]}" \
        "$output" || die "could not publish QQ attempt receipt for $target_job_id"
      continue
    fi
    if [[ -f "$status_receipt" ]]; then
      EXISTING_STATUS_IDS+=("$target_job_id")
      continue
    fi
  fi
  UNRESOLVED_STATUS_IDS+=("$target_job_id")
done

if [[ "${#UNRESOLVED_STATUS_IDS[@]}" -ne 0 ]]; then
  die "missing worker status was not eligible for dependency synthesis: $(join_ids "${UNRESOLVED_STATUS_IDS[@]}")"
fi

KEY_RESULT="dependency_audit_complete; synthesized=$(join_ids "${SYNTHESIZED_STATUS_IDS[@]}")"
exit 0
