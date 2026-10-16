#!/usr/bin/env bash
set -euo pipefail

# Submit Proposal v9 with only the hosted model changed to Qwen3.7-Plus.
# Usage:
#   NODE=gpu46 BAILIAN_NETWORK_MODE=direct \
#     bash tools/submit_ca0722_proposal_qwen37_v9.sh 10

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_INDEX="${1:-10}"

export MODES=proposal
export PROTOCOL_VERSION=v9
export BAILIAN_MODEL=qwen3.7-plus-2026-05-26
export PROTOCOL_EVIDENCE='v9 action-scoped delta masks plus original-color delta-only T1/T2 RGB crops with cyan contour'
export PROTOCOL_TRANSITION='v9 runtime gates unchanged; hosted Agent and staged Verifier use fixed Qwen3.7-Plus snapshot with thinking disabled'
export OUTPUT="${OUTPUT:-$ROOT/outputs/CA_0722(${RUN_INDEX})-bailian-proposal-qwen37-v9}"
export SHARED_LOG_ROOT="${SHARED_LOG_ROOT:-/guisongxia01/pangchao/wangyihan/wyh/.slurm-logs/CA_0722_${RUN_INDEX}_proposal_qwen37_v9}"

exec bash "$ROOT/tools/submit_ca0722_context_fix_ablation.sh" "$RUN_INDEX"
