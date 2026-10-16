#!/usr/bin/env bash
set -uo pipefail

# Lightweight management-node monitor for one Slurm job and the effective
# per-user GPU association quota. It sends transition-based QQ mail and keeps
# its state under the matching experiment logs directory.

ROOT="${REPO_ROOT:-/guisongxia01/pangchao/wangyihan/Chang-Agent}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/CA_0729(1)-qwen3vl-sft32-smoke}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
JOB_ID="${JOB_ID:-49550}"
SLURM_USER="${SLURM_USER:-guisongxia}"
GPU_LIMIT="${GPU_LIMIT:-24}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
QQ_NOTIFIER="${QQ_NOTIFIER:-/guisongxia01/pangchao/wangyihan/.local/bin/wangyihan-send-qq}"
DRY_RUN="${DRY_RUN:-0}"
ONCE=0

usage() {
  cat <<'EOF'
Usage: bash tools/monitor_qwen3vl_sft_job.sh [options]

Options:
  --job-id ID         Slurm job to watch (default: 49550)
  --gpu-limit COUNT   Effective per-user GPU limit (default: 24)
  --interval SECONDS  Poll interval, at least 60 seconds (default: 120)
  --once              Query once and exit
  --dry-run           Log notifications without sending mail
  -h, --help          Show this help

Environment overrides: REPO_ROOT, RUN_DIR, LOG_DIR, JOB_ID, SLURM_USER,
GPU_LIMIT, INTERVAL_SECONDS, QQ_NOTIFIER, DRY_RUN.
EOF
}

fail() {
  printf 'qwen3vl_monitor: %s\n' "$*" >&2
  exit 2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --job-id)
      [[ "$#" -ge 2 ]] || fail "--job-id requires a value"
      JOB_ID="$2"
      shift 2
      ;;
    --gpu-limit)
      [[ "$#" -ge 2 ]] || fail "--gpu-limit requires a value"
      GPU_LIMIT="$2"
      shift 2
      ;;
    --interval)
      [[ "$#" -ge 2 ]] || fail "--interval requires a value"
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fail "JOB_ID must be numeric: $JOB_ID"
[[ "$GPU_LIMIT" =~ ^[0-9]+$ && "$GPU_LIMIT" -gt 0 ]] \
  || fail "GPU_LIMIT must be a positive integer: $GPU_LIMIT"
[[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ && "$INTERVAL_SECONDS" -ge 60 ]] \
  || fail "INTERVAL_SECONDS must be an integer >= 60: $INTERVAL_SECONDS"
[[ "$SLURM_USER" == "guisongxia" ]] \
  || fail "this workspace monitor must query Slurm user guisongxia"
[[ -x "$QQ_NOTIFIER" ]] || fail "QQ notifier is missing or not executable: $QQ_NOTIFIER"

mkdir -p -- "$LOG_DIR"
STATE_FILE="$LOG_DIR/gpu-monitor-${JOB_ID}.state"
LOCK_FILE="$LOG_DIR/gpu-monitor-${JOB_ID}.lock"
MONITOR_LOG="$LOG_DIR/gpu-monitor-${JOB_ID}.log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  fail "another monitor already holds $LOCK_FILE"
fi

log() {
  printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$MONITOR_LOG"
}

availability_notified=0
start_notified=0
terminal_notified=0

load_state() {
  local key value
  [[ -f "$STATE_FILE" ]] || return 0
  while IFS='=' read -r key value; do
    case "$key" in
      availability_notified)
        [[ "$value" == "0" || "$value" == "1" ]] && availability_notified="$value"
        ;;
      start_notified)
        [[ "$value" == "0" || "$value" == "1" ]] && start_notified="$value"
        ;;
      terminal_notified)
        [[ "$value" == "0" || "$value" == "1" ]] && terminal_notified="$value"
        ;;
    esac
  done <"$STATE_FILE"
}

save_state() {
  local temporary
  temporary="$(mktemp "$LOG_DIR/.gpu-monitor-${JOB_ID}.state.XXXXXX")"
  printf 'availability_notified=%s\nstart_notified=%s\nterminal_notified=%s\n' \
    "$availability_notified" "$start_notified" "$terminal_notified" \
    >"$temporary"
  mv -- "$temporary" "$STATE_FILE"
}

send_mail() {
  local subject="$1"
  local body="$2"
  local qq_log
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry_run_notification subject=$(printf '%q' "$subject") body=$(printf '%q' "$body")"
    return 0
  fi
  qq_log="$(mktemp "/tmp/qwen3vl-monitor-${JOB_ID}.XXXXXX")"
  if QQ_SUB="$subject" QQ_BODY="$body" "$QQ_NOTIFIER" >"$qq_log" 2>&1; then
    log "notification_sent subject=$(printf '%q' "$subject")"
    rm -f -- "$qq_log"
    return 0
  fi
  log "notification_failed subject=$(printf '%q' "$subject")"
  rm -f -- "$qq_log"
  return 1
}

GPU_USED=0
GPU_AVAILABLE=0
JOB_STATE="UNKNOWN"
JOB_NODE="None"
JOB_ELAPSED="00:00:00"
JOB_REASON="unknown"

query_gpu_usage() {
  local running_gres line part
  local -a parts
  if ! running_gres="$(squeue -h -u "$SLURM_USER" -t RUNNING -o '%b' 2>/dev/null)"; then
    return 1
  fi
  GPU_USED=0
  while IFS= read -r line; do
    IFS=',' read -r -a parts <<<"$line"
    for part in "${parts[@]}"; do
      if [[ "$part" =~ gres/gpu(:[^,:\(\)]+)?:([0-9]+) ]]; then
        GPU_USED="$((GPU_USED + BASH_REMATCH[2]))"
      fi
    done
  done <<<"$running_gres"
  GPU_AVAILABLE="$((GPU_LIMIT - GPU_USED))"
  if [[ "$GPU_AVAILABLE" -lt 0 ]]; then
    GPU_AVAILABLE=0
  fi
}

query_job() {
  local job_line accounting_line
  if ! job_line="$(squeue -h -j "$JOB_ID" -o '%T|%N|%M|%R' 2>/dev/null)"; then
    return 1
  fi
  if [[ -n "$job_line" ]]; then
    IFS='|' read -r JOB_STATE JOB_NODE JOB_ELAPSED JOB_REASON <<<"$job_line"
    return 0
  fi
  if ! accounting_line="$(sacct -n -X -j "$JOB_ID" \
    --format=State,NodeList,Elapsed,Reason -P 2>/dev/null | sed -n '1p')"; then
    return 1
  fi
  if [[ -z "$accounting_line" ]]; then
    JOB_STATE="NOT_FOUND"
    JOB_NODE="None"
    JOB_ELAPSED="00:00:00"
    JOB_REASON="not found in squeue or sacct"
    return 0
  fi
  IFS='|' read -r JOB_STATE JOB_NODE JOB_ELAPSED JOB_REASON <<<"$accounting_line"
  JOB_STATE="${JOB_STATE%%+*}"
}

is_active_state() {
  [[ "$JOB_STATE" == "RUNNING" || "$JOB_STATE" == "CONFIGURING" ]]
}

is_terminal_state() {
  case "$JOB_STATE" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

handle_snapshot() {
  local now body
  now="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  log "snapshot gpu_used=${GPU_USED}/${GPU_LIMIT} gpu_available=${GPU_AVAILABLE} job=${JOB_ID} state=${JOB_STATE} node=${JOB_NODE} elapsed=${JOB_ELAPSED} reason=$(printf '%q' "$JOB_REASON")"

  if is_active_state && [[ "$start_notified" == "0" ]]; then
    body="event=job_started; job=${JOB_ID}; state=${JOB_STATE}; node=${JOB_NODE}; elapsed=${JOB_ELAPSED}; gpu_used=${GPU_USED}/${GPU_LIMIT}; time=${now}; output=${RUN_DIR}"
    if send_mail "Qwen3-VL SFT Job ${JOB_ID} started" "$body"; then
      start_notified=1
    fi
  fi

  if ! is_active_state && ! is_terminal_state; then
    if [[ "$GPU_AVAILABLE" -gt 0 && "$availability_notified" == "0" ]]; then
      body="event=gpu_available; available=${GPU_AVAILABLE}; used=${GPU_USED}/${GPU_LIMIT}; job=${JOB_ID}; state=${JOB_STATE}; reason=${JOB_REASON}; time=${now}; output=${RUN_DIR}"
      if send_mail "GPU available for Qwen3-VL SFT Job ${JOB_ID}" "$body"; then
        availability_notified=1
      fi
    elif [[ "$GPU_AVAILABLE" -eq 0 ]]; then
      availability_notified=0
    fi
  fi

  if is_terminal_state && [[ "$terminal_notified" == "0" ]]; then
    body="event=job_terminal; job=${JOB_ID}; state=${JOB_STATE}; node=${JOB_NODE}; elapsed=${JOB_ELAPSED}; time=${now}; output=${RUN_DIR}"
    if send_mail "Qwen3-VL SFT Job ${JOB_ID} ${JOB_STATE}" "$body"; then
      terminal_notified=1
    fi
  fi
  save_state
}

load_state
log "monitor_started job=${JOB_ID} user=${SLURM_USER} gpu_limit=${GPU_LIMIT} interval=${INTERVAL_SECONDS}s dry_run=${DRY_RUN}"

while true; do
  if query_gpu_usage && query_job; then
    handle_snapshot
    if is_terminal_state; then
      log "monitor_stopped terminal_state=${JOB_STATE}"
      exit 0
    fi
  else
    log "query_failed; will retry without changing notification state"
  fi
  if [[ "$ONCE" == "1" ]]; then
    log "monitor_stopped once=true"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
