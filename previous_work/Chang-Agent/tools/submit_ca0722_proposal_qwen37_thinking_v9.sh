#!/usr/bin/env bash
set -euo pipefail

# Strict follow-up to CA_0722(10): same Qwen3.7 snapshot and Proposal v9,
# with bounded hosted thinking enabled for every Agent/Verifier stage.
# Usage:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0722_proposal_qwen37_thinking_v9.sh 11

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_INDEX="${1:-11}"

export MODES=proposal
export PROTOCOL_VERSION=v9
export BAILIAN_MODEL=qwen3.7-plus-2026-05-26
export BAILIAN_ENABLE_THINKING=1
export BAILIAN_THINKING_BUDGET="${BAILIAN_THINKING_BUDGET:-256}"
export PROTOCOL_EVIDENCE='v9 action-scoped delta masks plus original-color delta-only T1/T2 RGB crops with cyan contour'
export PROTOCOL_TRANSITION="v9 runtime gates unchanged; fixed Qwen3.7-Plus snapshot with thinking enabled and reasoning budget ${BAILIAN_THINKING_BUDGET}"
export OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0722(${RUN_INDEX})-bailian-proposal-qwen37-thinking-v9}"
export SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0722_${RUN_INDEX}_proposal_qwen37_thinking_v9}"

exec bash "$ROOT/tools/submit_ca0722_context_fix_ablation.sh" "$RUN_INDEX"
