#!/usr/bin/env bash
# Host fallback only: repair hooks, deliver missed observations, retry finalization.
set -uo pipefail
ROOT="${EXP_LOOP_ROOT:?Set the installed exp-loop directory}"
ENGINE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/loop.py"
mkdir -p "$ROOT/logs"
for exp in "$ROOT"/experiments/*/; do
    name="$(basename "$exp")"
    case "$name" in _*|.*) continue ;; esac
    [[ -f "$exp/state.json" ]] || continue
    # Touch only experiments explicitly bound to this project's event receiver.
    /usr/bin/python3 -B -c 'import json,sys; sys.exit(not bool(json.load(open(sys.argv[1])).get("agent_bridge")))' "$exp/exp.json" || continue
    /usr/bin/python3 -B "$ENGINE" tick "$exp" >> "$ROOT/logs/watchdog.log" 2>&1
    /usr/bin/python3 -B "$(dirname "$ENGINE")/bridge.py" fallback "$exp" >> "$ROOT/logs/watchdog.log" 2>&1
    /usr/bin/python3 -B "$ENGINE" finalize "$exp" >> "$ROOT/logs/watchdog.log" 2>&1
done
